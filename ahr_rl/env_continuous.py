"""Alternative action spec: continuous allocation (for comparison with the
discrete bracket spec in env.py).

Action = 25 floats
    a[0:24]  in [-1, 1]  one per runner slot; > 0 back, < 0 lay, |a| = conviction
    a[24]    in [ 0, 1]  fraction of currently available funds to wager this step

Each decision (2s) the budget ``f * available_funds`` is split across runners
with ``softmax(|a_r| / temperature)`` over runners whose |a_r| clears a small
dead-zone. Each runner's share is placed as an aggressive order at the best
price (back: the share is the stake; lay: the share is the *liability*, so the
amount wagered is the amount at risk on either side). Orders that don't fill
immediately are cancelled at the next decision. Shares below the minimum stake
are skipped.

Everything else is identical to the bracket env so the comparison is fair:
same tapes, matching engine, funds check, commission, auto-green safety net
from the scheduled start, and reward (change in mark-to-market green value).
Note there is no explicit exit/hedge action here: to green up the agent must
itself place the offsetting bets.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from .env import BetfairPreRaceEnv, EnvConfig
from .exchange import BACK, LAY
from .features import R_MAX
from .ladder import PRICES

N_CONT = R_MAX + 1


@dataclass
class ContinuousConfig(EnvConfig):
    deadzone: float = 0.05  # |a_r| below this -> runner gets nothing
    min_fraction: float = 0.01  # f below this -> no bets this step
    temperature: float = 0.25  # softmax temperature over |a_r|
    max_fraction: float = 1.0  # cap on f (share of available funds wagered per step)


class ContinuousAllocEnv(BetfairPreRaceEnv):
    def __init__(self, tapes, cfg: ContinuousConfig | None = None, **kw):
        super().__init__(tapes, cfg or ContinuousConfig(), **kw)
        lo = np.full(N_CONT, -1.0, np.float32)
        lo[-1] = 0.0
        self.action_space = gym.spaces.Box(lo, np.ones(N_CONT, np.float32), dtype=np.float32)

    def _apply(self, action):
        ex, cfg = self.ex, self.cfg
        a = np.clip(np.asarray(action, np.float64)[:R_MAX], -1, 1)
        f = float(np.clip(action[R_MAX], 0.0, cfg.max_fraction))
        ex.cancel_all()  # aggressive-only: leftovers from the last decision are pulled
        if f < cfg.min_fraction:
            return
        R = min(self.tape.n_runners, R_MAX)
        elig = []
        for r in range(R):
            if abs(a[r]) < cfg.deadzone or not self.tape.active[ex.step, r] or ex.void_runner[r]:
                continue
            bb, bl = ex.best(ex.step, r)
            if (a[r] > 0 and bb >= 0) or (a[r] < 0 and bl >= 0):
                elig.append(r)
        if not elig:
            return
        z = np.abs(a[elig]) / cfg.temperature
        w = np.exp(z - z.max())
        w /= w.sum()
        budget = f * max(ex.available_funds(), 0.0)
        for r, wr in zip(elig, w):
            amt = budget * wr
            bb, bl = ex.best(ex.step, r)
            if a[r] > 0:
                o = ex.submit(r, BACK, bb, amt)
            else:
                o = ex.submit(r, LAY, bl, amt / max(PRICES[bl] - 1.0, 0.01))
            if o is not None:
                self.n_opens += 1
