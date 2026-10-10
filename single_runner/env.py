"""SingleRunnerTradingEnv — trade ONE runner of a Betfair race, observe the whole race.

Episode  = (race, target runner). The target is chosen by favouritism rank at
           the episode's first step (1 = favourite, 2 = 2nd favourite, ...).
           Every runner of every race is its own episode, so the agent learns
           from all of them while only ever trading one per episode.
Start    = $100 balance (configurable). Worst-case loss across outcomes,
           including unmatched orders, may never exceed the balance.
Actions  = Box(-1, 1, (6,)) each step:
             0 back size   : stake = max(a, 0) * max affordable back stake
             1 back price  : round(a * tick_range) ticks from best available-to-back
                             (+ = higher odds = more passive, - = more aggressive)
             2 lay size    : stake = max(a, 0) * max affordable lay stake
             3 lay price   : round(a * tick_range) ticks from best available-to-lay
                             (+ = lower odds = more passive, - = more aggressive)
             4 cancel backs: a > 0 cancels all unmatched backs (before placing)
             5 cancel lays : a > 0 cancels all unmatched lays (before placing)
           So any size up to the bank and any price within +/- tick_range ticks
           of the touch, on the real Betfair ladder, every step.
Matching = see matching.py (book walking, cross-matching, queue + traded-volume
           passive fills).
End      = the in-play transition. Unmatched orders lapse and the position is
           greened at the last pre-race snapshot (unbiased fair price by default).
Reward   = change in greened, post-commission equity / balance * reward_scale.
           It telescopes, so the episode return equals final net P&L / balance
           * reward_scale. The race result is never used.
"""
from collections import OrderedDict
from dataclasses import dataclass, asdict
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .features import (MAX_RUNNERS, N_GLOBAL_FEATURES, N_RUNNER_FEATURES, load_race)
from .ladder import shift, snap
from .matching import MatchingEngine, commission_on, snapshot_from_race

N_TARGET_EXTRA = 4
N_SLOT_EXTRA = 2
N_AGENT_STATE = 19


@dataclass
class EnvConfig:
    initial_balance: float = 100.0
    tick_range: int = 10               # price action covers +/- this many ticks
    min_stake: float = 1.0             # set to your jurisdiction's Betfair minimum
    max_open_orders: int = 20          # per side
    fill_optimism: float = 1.0         # share of traded volume available to passive orders
    cross_matching: bool = True
    greenup_mode: str = "fair"         # 'fair' (unbiased) | 'cross' (pay spread) | 'ltp'
    reward_scale: float = 10.0
    max_episode_steps: int = None      # use only the last N pre-race snapshots
    random_start: bool = False         # randomise start inside the window (training aug.)
    min_episode_steps: int = 10
    runner_ranks: tuple = None         # e.g. (1, 2, 3); None = every runner
    cache_size: int = 256              # races kept in memory


def obs_dim():
    return (N_GLOBAL_FEATURES + N_RUNNER_FEATURES + N_TARGET_EXTRA
            + MAX_RUNNERS * (N_RUNNER_FEATURES + N_SLOT_EXTRA) + N_AGENT_STATE)


class SingleRunnerTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, race_files, config=None, seed=None):
        super().__init__()
        self.cfg = config or EnvConfig()
        self.race_files = list(race_files)
        if not self.race_files:
            raise ValueError("race_files is empty")
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim(),), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        self._cache = OrderedDict()
        self._bad = set()
        self.engine = MatchingEngine(self.cfg.cross_matching, self.cfg.fill_optimism,
                                     self.cfg.greenup_mode)
        self.np_random_ = np.random.default_rng(seed)
        self.race = None

    # ------------------------------------------------------------ data
    def get_race(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        race = load_race(path)
        self._cache[path] = race
        if len(self._cache) > self.cfg.cache_size:
            self._cache.popitem(last=False)
        return race

    def _episode_bounds(self, race, rng):
        end = race.T - 1
        n = self.cfg.max_episode_steps
        start = 0 if n is None else max(0, race.T - n)
        if self.cfg.random_start and end - start > self.cfg.min_episode_steps:
            start = int(rng.integers(start, end - self.cfg.min_episode_steps + 1))
        return start, end

    def eligible_ranks(self, race, start):
        n_active = int(race.active[start].sum())
        ranks = range(1, n_active + 1)
        if self.cfg.runner_ranks is not None:
            ranks = [k for k in ranks if k in self.cfg.runner_ranks]
        return list(ranks)

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random_ = np.random.default_rng(seed)
        rng = self.np_random_
        options = options or {}
        for _ in range(100):
            path = options.get("race_path") or self.race_files[int(rng.integers(len(self.race_files)))]
            if path in self._bad:
                if "race_path" in options:
                    raise ValueError(f"unusable race {path}")
                continue
            try:
                race = self.get_race(path)
            except Exception:
                self._bad.add(path)
                continue
            start, end = self._episode_bounds(race, rng)
            ranks = self.eligible_ranks(race, start)
            if end - start + 1 < self.cfg.min_episode_steps or not ranks:
                if "race_path" in options:
                    raise ValueError(f"race {path} has no eligible episode")
                self._bad.add(path)
                continue
            rank = options.get("rank") or ranks[int(rng.integers(len(ranks)))]
            if rank not in ranks:
                raise ValueError(f"rank {rank} not eligible in {path} (eligible {ranks})")
            break
        else:
            raise RuntimeError("could not find a usable race episode")

        self.race, self.path = race, path
        self.t0, self.t, self.t_end = start, start, end
        self.target = int(np.where(race.rank[start] == rank)[0][0])
        self.target_rank = rank
        # all-runner slots ordered by favouritism at episode start (stable for the episode)
        self.slot_order = np.argsort(race.rank[start], kind="stable")
        self.engine.reset()
        self.start_fair = race.fair[start, self.target]
        self.balance = self.cfg.initial_balance
        self.equity = 0.0
        self.last_fill = (0.0, 0.0)
        self.n_orders = [0, 0]
        self.snap = snapshot_from_race(race, self.t, self.target)
        return self._obs(), {"race_path": path, "target_runner": self.target,
                             "target_rank": rank, "steps": end - start + 1}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        cfg, eng, s = self.cfg, self.engine, self.snap
        if a[4] > 0:
            eng.cancel("B")
        if a[5] > 0:
            eng.cancel("L")
        fb = self._place("B", a[0], a[1], s)
        fl = self._place("L", a[2], a[3], s)

        terminal = self.t >= self.t_end
        if terminal:
            lapsed = eng.lapse_all()
        else:
            self.t += 1
            new = snapshot_from_race(self.race, self.t, self.target)
            pb, pl = eng.on_new_snapshot(s, new, self.t)
            fb, fl = fb + pb, fl + pl
            self.snap = new
        self.last_fill = (fb, fl)

        new_equity = self._equity(self.snap)
        reward = (new_equity - self.equity) / cfg.initial_balance * cfg.reward_scale
        self.equity = new_equity
        info = {}
        if terminal:
            info = self._final_info(lapsed)
        return self._obs(), float(reward), terminal, False, info

    # ------------------------------------------------------------ internals
    def _place(self, side, size_a, price_a, s):
        frac = float(max(size_a, 0.0))
        if frac <= 0.0:
            return 0.0
        ticks = int(round(float(price_a) * self.cfg.tick_range))
        if side == "B":
            ref = s.back_p[0] if not np.isnan(s.back_p[0]) else snap(s.fair)
            price = shift(ref, ticks)            # + = higher odds = passive
        else:
            ref = s.lay_p[0] if not np.isnan(s.lay_p[0]) else snap(s.fair)
            price = shift(ref, -ticks)           # + = lower odds = passive
        stake = math.floor(frac * self.engine.max_affordable(side, price, self.balance) * 100) / 100
        if stake < self.cfg.min_stake:
            return 0.0
        self.n_orders[0 if side == "B" else 1] += 1
        return self.engine.place(side, price, stake, s, self.t, self.cfg.max_open_orders)

    def _equity(self, s):
        g, *_ = self.engine.green_up(s)
        return g - commission_on(g, self.race.commission)

    def _final_info(self, lapsed):
        eng, race = self.engine, self.race
        gross, hedge_price, hedge_side, hedge_stake = eng.green_up(self.snap)
        comm = commission_on(gross, race.commission)
        W, L = eng.pos.win_payoff, eng.pos.lose_payoff
        won = bool(race.is_winner[self.target] > 0.5)
        fills = eng.fills
        return {
            "race_path": self.path, "market_id": race.market_id, "race_date": race.race_date,
            "target_runner": self.target, "target_rank": self.target_rank,
            "pnl": gross - comm, "pnl_gross": gross, "commission": comm,
            "hedge_price": hedge_price, "hedge_side": hedge_side, "hedge_stake": hedge_stake,
            "back_matched": eng.pos.back_stake, "lay_matched": eng.pos.lay_stake,
            "avg_back": eng.pos.avg_back, "avg_lay": eng.pos.avg_lay,
            "n_fills": len(fills),
            "n_aggressive": sum(f.kind == "aggressive" for f in fills),
            "n_passive": sum(f.kind != "aggressive" for f in fills),
            "n_back_orders": self.n_orders[0], "n_lay_orders": self.n_orders[1],
            "lapsed_orders": lapsed, "steps": self.t_end - self.t0 + 1,
            # diagnostics only — never used in reward/obs: what the un-greened book would have paid
            "ungreened_pnl_actual": W if won else L, "target_won": won,
        }

    def _obs(self):
        race, t, r, B = self.race, self.t, self.target, self.cfg.initial_balance
        eng = self.engine
        rf = race.runner_feats[t]
        tgt_extra = np.array([
            self.target_rank / MAX_RUNNERS,
            (race.rank[t, r] - self.target_rank) / MAX_RUNNERS,
            np.log(race.fair[t, r] / self.start_fair),
            (t - self.t0) / 500.0,
        ], dtype=np.float32)
        slots = np.zeros((MAX_RUNNERS, N_RUNNER_FEATURES + N_SLOT_EXTRA), dtype=np.float32)
        n = min(race.n_runners, MAX_RUNNERS)
        order = self.slot_order[:n]
        slots[:n, :N_RUNNER_FEATURES] = rf[order]
        slots[:n, N_RUNNER_FEATURES] = (order == r)
        slots[:n, N_RUNNER_FEATURES + 1] = (race.rank[t, order] - race.rank[self.t0, order]) / MAX_RUNNERS

        pos = eng.pos
        fair = self.snap.fair
        ub = [o for o in eng.orders if o.side == "B"]
        ul = [o for o in eng.orders if o.side == "L"]

        def rel(p):
            return float(np.log(p / fair)) if p > 1.0 else 0.0

        ub_stake = sum(o.size for o in ub)
        ul_stake = sum(o.size for o in ul)
        state = np.array([
            pos.win_payoff / B, pos.lose_payoff / B, eng.exposure() / B,
            (self.balance - eng.exposure()) / B,
            pos.back_stake / B, rel(pos.avg_back), pos.lay_stake / B, rel(pos.avg_lay),
            self.equity / B,
            ub_stake / B, rel(min((o.price for o in ub), default=0.0)),
            np.log1p(min((o.queue for o in ub), default=0.0)),
            ul_stake / B, rel(max((o.price for o in ul), default=0.0)),
            np.log1p(min((o.queue for o in ul), default=0.0)),
            len(ub) / self.cfg.max_open_orders, len(ul) / self.cfg.max_open_orders,
            self.last_fill[0] / B, self.last_fill[1] / B,
        ], dtype=np.float32)
        obs = np.concatenate([race.global_feats[t], rf[r], tgt_extra, slots.ravel(), state])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def episode_specs(env, race_files):
    """Every (race, rank) episode in `race_files` — for exhaustive evaluation."""
    specs = []
    for path in race_files:
        try:
            race = env.get_race(path)
        except Exception:
            continue
        start, end = env._episode_bounds(race, np.random.default_rng(0))
        if end - start + 1 < env.cfg.min_episode_steps:
            continue
        specs += [{"race_path": path, "rank": k} for k in env.eligible_ranks(race, start)]
    return specs


def config_dict(cfg):
    return asdict(cfg)
