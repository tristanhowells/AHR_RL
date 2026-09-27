"""Gymnasium environment: one episode = one race, from ~10 min before the
scheduled start until the market turns in-play.

Action space (per runner, every ``decision_every`` tape steps = 2s by default)
------------------------------------------------------------------------------
    0  NOOP
    1  CLOSE         cancel everything on the runner and flatten it (cross the
                     spread, re-trying each decision until flat)
    2  CANCEL_ENTRY  pull an unfilled entry order, keep managing any position
    3+ OPEN(side, entry, stake, tp)
         side  : BACK or LAY
         entry : TAKE (cross the spread now) or JOIN (queue at the best price
                 on our side of the book)
         stake : one of EnvConfig.stakes
         tp    : take-profit distance in ticks

OPEN is a bracket trade, which is how greening traders work: once the entry is
(partly) matched, an exit order is kept resting ``tp`` ticks better than the
entry price, sized to green that runner completely; if the price moves
``sl_mult * tp`` ticks against the entry, the position is closed at market.
Every leg is an ordinary Betfair back or lay at a ladder price, the funds
check applies to each one, and the agent can CLOSE at any time.

Safety net
----------
From ``auto_green_s`` (relative to the scheduled start) no new trades are
opened and every runner is closed at market each decision until the off.
The same rule should run in the live bot.

Reward
------
Potential-based on the mark-to-market green value G (worst-case net P&L if
every runner were flattened now), blended towards the unhedged worst case as
the scheduled start approaches; the terminal potential is the actual
worst-case net P&L at the off. The undiscounted episode return is therefore
exactly the guaranteed (green) profit across all outcomes.
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from .exchange import BACK, LAY, Exchange, ExchangeConfig
from .features import (N_GLOBAL_FEATURES, N_RUNNER_FEATURES, R_MAX, TapeHistory,
                       global_features, runner_features)
from .ladder import N_TICKS, PRICES, price_to_tick
from .tape import Tape

NOOP, CLOSE, CANCEL_ENTRY = 0, 1, 2
N_FIXED = 3
TAKE, JOIN = 0, 1
ENTRY_MODES = (TAKE, JOIN)


@dataclass
class EnvConfig:
    stakes: tuple = (5.0, 10.0, 25.0, 50.0)
    tp_ticks: tuple = (1, 2, 4)
    sl_mult: float = 2.0
    max_hold_s: float = 180.0
    decision_every: int = 4  # tape steps per decision (tape dt=0.5s -> 2s)
    reward_scale: float = 5.0  # $ per unit reward
    random_start_s: float = 0.0  # randomise episode start within this many seconds
    phi_ramp_start_s: float = -120.0
    phi_ramp_end_s: float = 0.0
    auto_green_s: float | None = 0.0
    close_slippage_ticks: int = 2
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)

    @property
    def n_actions(self) -> int:
        return N_FIXED + 2 * len(ENTRY_MODES) * len(self.stakes) * len(self.tp_ticks)


def decode_open(a: int, cfg: EnvConfig):
    """action id (>= N_FIXED) -> (side, entry_mode, stake, tp)"""
    a -= N_FIXED
    n_tp, n_st = len(cfg.tp_ticks), len(cfg.stakes)
    tp = cfg.tp_ticks[a % n_tp]
    a //= n_tp
    stake = cfg.stakes[a % n_st]
    a //= n_st
    mode = ENTRY_MODES[a % len(ENTRY_MODES)]
    side = BACK if a // len(ENTRY_MODES) == 0 else LAY
    return side, mode, stake, tp


def encode_open(side: int, mode: int, stake_idx: int, tp_idx: int, cfg: EnvConfig) -> int:
    n_tp, n_st = len(cfg.tp_ticks), len(cfg.stakes)
    s = 0 if side == BACK else 1
    return N_FIXED + ((s * len(ENTRY_MODES) + mode) * n_st + stake_idx) * n_tp + tp_idx


def entry_tick(ex: Exchange, runner: int, side: int, mode: int) -> int | None:
    bb, bl = ex.best(ex.step, runner)
    if side == BACK:
        if mode == TAKE:
            return bb if bb >= 0 else None
        return bl if bl >= 0 else (bb + 1 if bb >= 0 else None)
    if mode == TAKE:
        return bl if bl >= 0 else None
    return bb if bb >= 0 else (bl - 1 if bl >= 0 else None)


@dataclass
class Bracket:
    side: int  # side of the entry
    tp: int
    sl: int
    opened_step: int
    entry_oid: int | None = None
    exit_oid: int | None = None
    closing: bool = False


class BetfairPreRaceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tapes, cfg: EnvConfig | None = None, seed: int | None = None,
                 cache_tapes: bool = True, sequential: bool = False):
        self.cfg = cfg or EnvConfig()
        self.sources = list(tapes)
        self.cache_tapes = cache_tapes
        self._cache: dict[str, Tape] = {}
        self.sequential = sequential
        self._next = 0
        self.rng = random.Random(seed)
        self.action_space = gym.spaces.MultiDiscrete([self.cfg.n_actions] * R_MAX)
        self.observation_space = gym.spaces.Dict({
            "runners": gym.spaces.Box(-10, 10, (R_MAX, N_RUNNER_FEATURES), np.float32),
            "global": gym.spaces.Box(-10, 10, (N_GLOBAL_FEATURES,), np.float32),
            "mask": gym.spaces.MultiBinary(R_MAX),
        })

    # ------------------------------------------------------------------ helpers
    def _load(self, src) -> Tape:
        if isinstance(src, Tape):
            return src
        t = self._cache.get(src)
        if t is None:
            t = Tape.load(src)
            if self.cache_tapes:
                self._cache[src] = t
        return t

    def _obs(self):
        R, mask = runner_features(self.hist, self.ex.step, self.ex)
        g = global_features(self.hist, self.ex.step, self.ex, self._phi)
        return {"runners": R, "global": g, "mask": mask.astype(np.int8)}

    def in_auto_green(self) -> bool:
        g = self.cfg.auto_green_s
        return g is not None and float(self.tape.t_rel[self.ex.step]) >= g

    # ------------------------------------------------------------------ gym api
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        if options and "tape" in options:
            src = options["tape"]
        elif self.sequential:
            src = self.sources[self._next % len(self.sources)]
            self._next += 1
        else:
            src = self.rng.choice(self.sources)
        self.tape = self._load(src)
        self.hist = TapeHistory(self.tape)
        start = 0
        if self.cfg.random_start_s > 0:
            start = self.rng.randrange(0, max(1, int(self.cfg.random_start_s / self.tape.dt)))
            start = min(start, self.tape.n_steps - 2)
        self.ex = Exchange(self.tape, self.cfg.exchange, start_step=start)
        self.ex.brackets = {}  # runner -> Bracket (read by features)
        self._phi = 0.0
        self._rphi = np.zeros(self.tape.n_runners)
        self.ep_return = 0.0
        self.n_opens = 0
        self.n_closes = 0
        self.n_stops = 0
        return self._obs(), {"market": self.tape.name}

    def step(self, action):
        ex, cfg = self.ex, self.cfg
        self.decide(action)
        for _ in range(cfg.decision_every):
            ex.advance()
            if ex.step >= self.tape.n_steps - 1:
                break
        done = ex.step >= self.tape.n_steps - 1
        info = {}
        # per-runner potentials (for per-runner credit assignment in training)
        rv = self._runner_potential(done)
        runner_rewards = np.zeros(R_MAX, np.float32)
        n = min(len(rv), R_MAX)
        runner_rewards[:n] = (rv[:n] - self._rphi[:n]) / cfg.reward_scale
        self._rphi = rv
        info["runner_rewards"] = runner_rewards
        if done:
            res = ex.settle()
            phi_new = res["worst"]
            info.update(res)
            info.update(market=self.tape.name, turnover=ex.turnover, n_fills=len(ex.fills),
                        n_rejected=ex.n_rejected, n_opens=self.n_opens, n_closes=self.n_closes,
                        n_stops=self.n_stops)
        else:
            phi_new = self._potential()
        reward = (phi_new - self._phi) / cfg.reward_scale
        self._phi = phi_new
        self.ep_return += reward * cfg.reward_scale
        return self._obs(), float(reward), bool(done), False, info

    def decide(self, action):
        """Turn the agent's action at the current step into orders (no time passes).
        Shared by training (via step) and the live bot."""
        action = np.asarray(action).reshape(-1)
        if self.tape.suspended[self.ex.step]:
            return
        if self.in_auto_green():
            for r in range(self.tape.n_runners):
                self._close(r)
        else:
            self._apply(action)
        self._manage()

    def attach(self, tape: Tape, ex: Exchange):
        """Point the env at an externally driven tape/exchange (live bot)."""
        self.tape, self.ex = tape, ex
        self.hist = TapeHistory(tape)
        if not hasattr(ex, "brackets"):
            ex.brackets = {}
            self._phi = 0.0
            self.ep_return = 0.0
            self.n_opens = self.n_closes = self.n_stops = 0

    def observe(self):
        return self._obs()

    # ------------------------------------------------------------------ internals
    def _ramp_w(self) -> float:
        cfg = self.cfg
        t = float(self.tape.t_rel[self.ex.step])
        return float(np.clip((t - cfg.phi_ramp_start_s) / max(cfg.phi_ramp_end_s - cfg.phi_ramp_start_s, 1e-6),
                             0.0, 1.0))

    def _runner_potential(self, terminal: bool) -> np.ndarray:
        if not self.ex.bet_runner:
            return np.zeros(self.tape.n_runners)
        return self.ex.runner_values(w_unhedged=1.0 if terminal else self._ramp_w())

    def _potential(self) -> float:
        ex, cfg = self.ex, self.cfg
        if not ex.bet_runner:
            return 0.0
        t = float(self.tape.t_rel[ex.step])
        w = np.clip((t - cfg.phi_ramp_start_s) / max(cfg.phi_ramp_end_s - cfg.phi_ramp_start_s, 1e-6), 0.0, 1.0)
        g = ex.green_value()
        return g if w <= 0 else (1 - w) * g + w * ex.worst_net()

    def _apply(self, action):
        ex, cfg = self.ex, self.cfg
        for r in range(min(self.tape.n_runners, R_MAX)):
            a = int(action[r])
            if a == NOOP or not self.tape.active[ex.step, r] or ex.void_runner[r]:
                continue
            if a == CLOSE:
                self.n_closes += 1
                self._close(r)
            elif a == CANCEL_ENTRY:
                b = ex.brackets.get(r)
                if b and b.entry_oid is not None:
                    ex.cancel_order(b.entry_oid)
            else:
                self._open(r, *decode_open(a, cfg))

    def _open(self, r, side, mode, stake, tp):
        ex = self.ex
        flat = abs(ex.W[r] - ex.L[r]) < 0.5
        if r in ex.brackets or not flat or any(o.runner == r for o in ex.orders):
            ex.n_rejected += 1  # one trade at a time per runner
            return
        tk = entry_tick(ex, r, side, mode)
        if tk is None:
            ex.n_rejected += 1
            return
        o = ex.submit(r, side, tk, stake)
        if o is None:
            return
        self.n_opens += 1
        ex.brackets[r] = Bracket(side, tp, max(1, int(round(self.cfg.sl_mult * tp))), ex.step, entry_oid=o.oid)

    def _close(self, r):
        ex = self.ex
        b = ex.brackets.get(r)
        if b is None:
            if abs(ex.W[r] - ex.L[r]) < 0.01 and not any(o.runner == r for o in ex.orders):
                return
            b = ex.brackets[r] = Bracket(BACK, 0, 0, ex.step)
        b.closing = True

    def _entry_vwap_tick(self, r, b: Bracket) -> int | None:
        fills = [f for f in self.ex.fills if f.runner == r and f.side == b.side and f.step >= b.opened_step]
        if not fills:
            return None
        st = sum(f.stake for f in fills)
        p = sum(f.price * f.stake for f in fills) / st
        return price_to_tick(p)

    def _manage(self):
        """Maintain exits / stops / closes for every open bracket."""
        ex, cfg = self.ex, self.cfg
        live = {o.oid: o for o in ex.orders}
        for r, b in list(ex.brackets.items()):
            if ex.void_runner[r] or not self.tape.active[ex.step, r]:
                del ex.brackets[r]
                continue
            d = ex.W[r] - ex.L[r]
            entry_live = b.entry_oid in live
            if b.closing:
                if abs(d) < 0.01:
                    if not any(o.runner == r for o in ex.orders):
                        del ex.brackets[r]
                    else:
                        ex.cancel_runner(r)
                    continue
                ex.cancel_runner(r)
                plan = ex.hedge_plan(r)
                if plan is not None:
                    side, tk, stake = plan
                    tk = tk - cfg.close_slippage_ticks if side == BACK else tk + cfg.close_slippage_ticks
                    ex.submit(r, side, int(np.clip(tk, 0, N_TICKS - 1)), stake, is_hedge=True)
                continue
            if abs(d) < 0.01:
                if not entry_live and b.exit_oid not in live:
                    del ex.brackets[r]  # entry never filled / fully exited
                continue
            # position open: stop-loss and time stop
            e_tk = self._entry_vwap_tick(r, b)
            bb, bl = ex.best(ex.step, r)
            held_s = (ex.step - b.opened_step) * self.tape.dt
            if e_tk is not None:
                adverse = (bl >= 0 and bl >= e_tk + b.sl) if b.side == BACK else (bb >= 0 and bb <= e_tk - b.sl)
                if adverse or held_s > cfg.max_hold_s:
                    self.n_stops += 1
                    b.closing = True
                    ex.cancel_runner(r)
                    continue
                # keep one exit order resting at the target, sized to green the runner
                tgt = e_tk - b.tp if b.side == BACK else e_tk + b.tp
                tgt = int(np.clip(tgt, 0, N_TICKS - 1))
                want = abs(d) / PRICES[tgt]
                ex_o = live.get(b.exit_oid)
                if ex_o is not None and abs(ex_o.size - want) <= max(0.05, 0.05 * want):
                    continue
                if ex_o is not None:
                    ex.cancel_order(ex_o.oid)
                exit_side = LAY if d > 0 else BACK
                o = ex.submit(r, exit_side, tgt, want, is_hedge=True)
                b.exit_oid = o.oid if o is not None else None


def list_tapes(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern))


def split_by_date(paths: list[str], val_frac: float = 0.15, test_frac: float = 0.15):
    """Chronological split on the YYYYMMDD filename prefix (no look-ahead leakage)."""
    days = sorted({os.path.basename(p)[:8] for p in paths})
    n = len(days)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_days = set(days[n - n_test:])
    val_days = set(days[n - n_test - n_val: n - n_test])
    tr, va, te = [], [], []
    for p in paths:
        d = os.path.basename(p)[:8]
        (te if d in test_days else va if d in val_days else tr).append(p)
    return tr, va, te
