"""SingleRunnerTradingEnv: trade ONE runner of a Betfair race, observe the whole race.

Data     = stream tapes (ahr_rl.tape, .npz): 0.5 s grid, 8-level ladders, every
           trade, projected BSP, scratchings, optional catalogue form features.
Episode  = (tape, target rank). The target is the runner holding that
           favouritism rank (by fair price) at the episode's first step; it stays
           the target even if the market reorders. Every runner of every race is
           its own episode.
Start    = $100 balance (configurable). Betfair funds check: worst-case loss over
           outcomes, including unmatched orders, never exceeds the balance.
Actions  = Box(-1, 1, (6,)) every `decision_every` tape steps (default 4 = 2 s):
             0 back size   : stake = max(a, 0) * max affordable back stake
             1 back price  : round(a * tick_range) ticks from the best atb price
                             (+ = higher odds = passive, - = aggressive)
             2 lay size    : stake = max(a, 0) * max affordable lay stake
             3 lay price   : round(a * tick_range) ticks from the best atl price
                             (+ = lower odds = passive, - = aggressive)
             4 cancel backs: a > 0 pulls all unmatched backs on the target
             5 cancel lays : a > 0 pulls all unmatched lays on the target
           Opening orders need the $5 minimum; an order that only reduces the
           position may go below it (the standard cancel-down workaround).
Matching = ahr_rl.exchange (latency, queue position, fills from recorded trades,
           liquidity memory, scratchings, base-rate commission) + cross-matching
           (exchange_x.py).
End      = the in-play transition (last tape row). Unmatched orders lapse; the
           position is greened at the last tradeable pre-off snapshot at the
           unbiased fair price (or by walking the book with greenup_mode='cross').
Reward   = change in greened, post-commission equity / balance * reward_scale.
           It telescopes: episode return = final net P&L / balance * reward_scale,
           and doing nothing scores exactly 0. The race result is never used.
"""
from collections import OrderedDict
from dataclasses import dataclass, asdict
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ahr_rl.exchange import BACK, LAY, ExchangeConfig
from ahr_rl.ladder import N_TICKS, PRICES, price_to_tick
from ahr_rl.tape import Tape

from .exchange_x import CrossMatchingExchange
from .features import MAX_RUNNERS, N_GLOBAL_FEATURES, N_RUNNER_FEATURES, build_features

N_TARGET_EXTRA = 4
N_SLOT_EXTRA = 2
N_AGENT_STATE = 19
_SEARCH_BACK = 240          # steps (2 min) to look back for a tradeable book


@dataclass
class EnvConfig:
    initial_balance: float = 100.0
    min_stake: float = 5.0             # Betfair AU minimum for opening orders
    tick_range: int = 10               # price action spans +/- this many ladder ticks
    max_open_orders: int = 20          # per side
    decision_every: int = 4            # tape steps per decision (0.5 s grid -> 2 s)
    latency_steps: int = 1             # orders/cancels land one tape step later
    fill_mode: str = "realistic"       # ahr_rl fill model: realistic | no_queue | touch
    cross_matching: bool = True
    max_virtual_ticks: int = 3         # ignore virtual prices further than this past the best
    greenup_mode: str = "fair"         # 'fair' (unbiased) | 'cross' (walk the book, pays spread)
    commission: float = None           # None = the market's base rate from the tape
    reward_scale: float = 10.0
    start_s: float = 600.0             # start this many seconds before the scheduled off
    random_start_s: float = 0.0        # training augmentation: random later start
    min_decisions: int = 10
    runner_ranks: tuple = None         # e.g. (1, 2, 3); None = every runner
    cache_size: int = 256              # tapes (+ features, ~6 MB each) kept in memory

    def exchange_config(self):
        return ExchangeConfig(bankroll=self.initial_balance, min_stake=self.min_stake,
                              commission=self.commission, latency_steps=self.latency_steps,
                              fill_mode=self.fill_mode)


def obs_dim():
    return (N_GLOBAL_FEATURES + N_RUNNER_FEATURES + N_TARGET_EXTRA
            + MAX_RUNNERS * (N_RUNNER_FEATURES + N_SLOT_EXTRA) + N_AGENT_STATE)


class SingleRunnerTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tapes, config=None, seed=None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.sources = list(tapes)
        if not self.sources:
            raise ValueError("no tapes given")
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim(),), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        self._cache = OrderedDict()
        self._bad = set()
        self.rng = np.random.default_rng(seed)
        self.tape = None

    # ------------------------------------------------------------ data
    def get(self, src):
        """(tape, features) for a path or Tape object, cached."""
        key = src if isinstance(src, str) else id(src)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        tape = Tape.load(src) if isinstance(src, str) else src
        comm = self.cfg.commission if self.cfg.commission is not None else tape.base_rate / 100.0
        item = (tape, build_features(tape, comm))
        self._cache[key] = item
        if len(self._cache) > self.cfg.cache_size:
            self._cache.popitem(last=False)
        return item

    def start_step(self, tape, rng=None):
        T = tape.n_steps
        s = int(np.searchsorted(tape.t_rel, -self.cfg.start_s))
        if rng is not None and self.cfg.random_start_s > 0:
            s += int(rng.integers(0, max(1, int(self.cfg.random_start_s / tape.dt))))
        while s < T - 1 and tape.suspended[s]:
            s += 1
        return min(s, T - 1)

    def eligible_ranks(self, tape, feats, start):
        if not tape.went_in_play:
            return []      # abandoned / closed before the off: every bet is void
        if (tape.n_steps - 1 - start) // self.cfg.decision_every < self.cfg.min_decisions:
            return []
        ranks = list(range(1, int(feats.priced[start].sum()) + 1))
        if self.cfg.runner_ranks is not None:
            ranks = [k for k in ranks if k in self.cfg.runner_ranks]
        return ranks

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        forced = options.get("tape")
        for _ in range(200):
            src = forced if forced is not None else self.sources[int(self.rng.integers(len(self.sources)))]
            key = src if isinstance(src, str) else id(src)
            if key in self._bad and forced is None:
                continue
            try:
                tape, feats = self.get(src)
            except Exception:
                if forced is not None:
                    raise
                self._bad.add(key)
                continue
            start = self.start_step(tape, None if forced is not None else self.rng)
            ranks = self.eligible_ranks(tape, feats, start)
            if not ranks:
                if forced is not None:
                    raise ValueError(f"no eligible episode in {key}")
                self._bad.add(key)
                continue
            rank = options.get("rank") or ranks[int(self.rng.integers(len(ranks)))]
            if rank not in ranks:
                raise ValueError(f"rank {rank} not eligible (eligible {ranks})")
            break
        else:
            raise RuntimeError("could not find a usable tape")

        self.tape, self.feats, self.src = tape, feats, key
        self.t0, self.t_end = start, tape.n_steps - 1
        self.target = int(np.nonzero(feats.rank[start] == rank)[0][0])
        self.target_rank = rank
        self.slot_order = np.argsort(feats.rank[start], kind="stable")[:min(tape.n_runners, MAX_RUNNERS)]
        self.ex = CrossMatchingExchange(tape, self.cfg.exchange_config(), start, self.cfg.cross_matching,
                                        self.cfg.max_virtual_ticks)
        self.start_fair = max(feats.fair[start, self.target], 1.01)
        self.equity = 0.0
        self.last_fill = (0.0, 0.0)
        self.n_submitted = [0, 0]
        self.n_decisions = 0
        return self._obs(), {"tape": key, "target_runner": self.target, "target_rank": rank,
                             "decisions": (self.t_end - start) // self.cfg.decision_every}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        ex, r = self.ex, self.target
        n_fills = len(ex.fills)
        tradeable = (not self.tape.suspended[ex.step] and self.tape.active[ex.step, r]
                     and not ex.void_runner[r])
        if tradeable:
            for side, cancel in ((BACK, a[4]), (LAY, a[5])):
                if cancel > 0:
                    for o in ex.orders:
                        if o.runner == r and o.side == side:
                            ex.cancel_order(o.oid)
            self._place(BACK, a[0], a[1])
            self._place(LAY, a[2], a[3])
        for _ in range(self.cfg.decision_every):
            if ex.step >= self.t_end:
                break
            ex.advance()
        self.n_decisions += 1
        new = [f for f in ex.fills[n_fills:] if f.runner == r]
        self.last_fill = (sum(f.stake for f in new if f.side == BACK),
                          sum(f.stake for f in new if f.side == LAY))

        terminal = ex.step >= self.t_end or bool(ex.void_runner[r])
        new_equity = self._equity()
        reward = (new_equity - self.equity) / self.cfg.initial_balance * self.cfg.reward_scale
        self.equity = new_equity
        info = self._final_info() if terminal else {}
        return self._obs(), float(reward), terminal, False, info

    # ------------------------------------------------------------ internals
    def _place(self, side, size_a, price_a):
        frac = float(max(size_a, 0.0))
        if frac <= 0.0:
            return
        ex, r = self.ex, self.target
        if sum(1 for o in ex.orders if o.runner == r and o.side == side) >= self.cfg.max_open_orders:
            return
        bb, bl = ex.best(ex.step, r)
        off = int(round(float(price_a) * self.cfg.tick_range))
        fair = self.feats.fair[ex.step, r]
        if side == BACK:
            ref = bb if bb >= 0 else (bl - 1 if bl >= 0 else (price_to_tick(fair) if fair > 1 else -1))
            tick = ref + off
        else:
            ref = bl if bl >= 0 else (bb + 1 if bb >= 0 else (price_to_tick(fair) if fair > 1 else -1))
            tick = ref - off
        if ref < 0:
            return
        tick = int(np.clip(tick, 0, N_TICKS - 1))
        cap = ex.max_stake(r, side, tick)
        if not np.isfinite(cap):
            cap = self.cfg.initial_balance
        stake = frac * cap
        d = float(ex.W[r] - ex.L[r])
        hedge_size = abs(d) / PRICES[tick]
        is_hedge = ((side == LAY and d > 0) or (side == BACK and d < 0)) and stake <= hedge_size + 0.01
        if ex.submit(r, side, tick, stake, is_hedge=is_hedge) is not None:
            self.n_submitted[0 if side == BACK else 1] += 1

    def _value_step(self, step):
        """Latest step <= `step` where the target has a tradeable book."""
        t, r = self.tape, self.target
        for s in range(step, max(self.t0, step - _SEARCH_BACK) - 1, -1):
            if not t.suspended[s] and self.feats.fair[s, r] > 1.0:
                return s
        return step

    def _equity(self):
        g, _ = self.ex.green_value(self.target, self._value_step(self.ex.step), self.cfg.greenup_mode)
        return self.ex.net(g)

    def _final_info(self):
        ex, r, tape = self.ex, self.target, self.tape
        gs = self._value_step(ex.step)
        gross, hedge_price = ex.green_value(r, gs, self.cfg.greenup_mode)
        lapsed = sum(1 for o in ex.orders if o.runner == r)
        bs, ab, ls, al = ex.runner_bets(r)
        fills = [f for f in ex.fills if f.runner == r]
        W, L = float(ex.W[r]), float(ex.L[r])
        won = tape.winner == r if tape.winner >= 0 else None
        ex.settle()
        return {
            "tape": self.src if isinstance(self.src, str) else str(self.src),
            "race": os.path.basename(self.src) if isinstance(self.src, str) else tape.name,
            "race_date": tape.name[:8], "market_id": tape.market_id,
            "target_runner": r, "target_rank": self.target_rank,
            "pnl": ex.net(gross), "pnl_gross": gross, "commission": gross - ex.net(gross),
            "commission_rate": ex.commission, "hedge_price": hedge_price,
            "green_secs_before_off": float(-tape.t_rel[gs]),
            "back_matched": bs, "avg_back": ab, "lay_matched": ls, "avg_lay": al,
            "n_fills": len(fills), "n_passive": sum(f.passive for f in fills),
            "n_aggressive": sum(not f.passive for f in fills),
            "n_back_orders": self.n_submitted[0], "n_lay_orders": self.n_submitted[1],
            "n_rejected": ex.n_rejected, "lapsed_orders": lapsed,
            "decisions": self.n_decisions, "void": bool(ex.void_runner[r]),
            # diagnostics only, never used in reward/obs: the un-greened book on the real result
            "ungreened_pnl_actual": (ex.net(W) if won else ex.net(L)) if won is not None else float("nan"),
            "target_won": won,
        }

    def _obs(self):
        tape, f, ex, r = self.tape, self.feats, self.ex, self.target
        s, B = ex.step, self.cfg.initial_balance
        rf = f.runner[s]
        fair_now = f.fair[s, r] if f.fair[s, r] > 1 else self.start_fair
        tgt_extra = np.array([
            self.target_rank / MAX_RUNNERS,
            (f.rank[s, r] - self.target_rank) / MAX_RUNNERS,
            np.log(fair_now / self.start_fair),
            (s - self.t0) * tape.dt / 600.0,
        ], dtype=np.float32)
        n = len(self.slot_order)
        slots = np.zeros((MAX_RUNNERS, N_RUNNER_FEATURES + N_SLOT_EXTRA), np.float32)
        slots[:n, :N_RUNNER_FEATURES] = rf[self.slot_order]
        slots[:n, N_RUNNER_FEATURES] = self.slot_order == r
        slots[:n, N_RUNNER_FEATURES + 1] = (f.rank[s, self.slot_order] - f.rank[self.t0, self.slot_order]) / MAX_RUNNERS

        bs, ab, ls, al = ex.runner_bets(r)
        ub = [o for o in ex.orders if o.runner == r and o.side == BACK]
        ul = [o for o in ex.orders if o.runner == r and o.side == LAY]

        def rel(p):
            return float(np.log(p / fair_now)) if p > 1.0 else 0.0

        exposure = max(0.0, -ex.worst_case())
        state = np.array([
            ex.W[r] / B, ex.L[r] / B, exposure / B, ex.available_funds() / B,
            bs / B, rel(ab), ls / B, rel(al), self.equity / B,
            sum(o.size for o in ub) / B, rel(PRICES[min(o.tick for o in ub)]) if ub else 0.0,
            np.log1p(min(min((o.queue_ahead for o in ub), default=0.0), 1e6)),
            sum(o.size for o in ul) / B, rel(PRICES[max(o.tick for o in ul)]) if ul else 0.0,
            np.log1p(min(min((o.queue_ahead for o in ul), default=0.0), 1e6)),
            len(ub) / self.cfg.max_open_orders, len(ul) / self.cfg.max_open_orders,
            self.last_fill[0] / B, self.last_fill[1] / B,
        ], dtype=np.float32)
        obs = np.concatenate([f.glob[s], rf[r], tgt_extra, slots.ravel(), state])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def episode_specs(env, tapes):
    """Every eligible (tape, rank) episode, for exhaustive evaluation."""
    specs = []
    for src in tapes:
        try:
            tape, feats = env.get(src)
        except Exception:
            continue
        start = env.start_step(tape)
        specs += [{"tape": src, "rank": k} for k in env.eligible_ranks(tape, feats, start)]
    return specs


def config_dict(cfg):
    return asdict(cfg)
