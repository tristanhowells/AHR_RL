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

Liquidity gate (``min_best_usd`` > 0): a runner may only be *opened* while the money
at its best back + lay prices, averaged over the last ``liq_window_s`` (live, no
look-ahead), is at least ``min_best_usd``. A runner below the gate that the agent
already holds may still be traded, but only to reduce that position (capped at the
stake that flattens it). The observation gains ``act_mask``: the runners the agent
can act on (tradeable or held). ``mask`` is unchanged, so the agent still sees the
whole market. With ``gate_inputs`` (default on when gated) each runner's features gain
two columns (N_GATE_INPUTS), so the networks know the gate state rather than inferring it:
  above the cut  1 / 0 (0 for a held runner that is reduce-only)
  distance       log(trailing $ at best / cut), clipped to +-3 and divided by 3
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
N_GATE_INPUTS = 2


@dataclass
class ContinuousConfig(EnvConfig):
    deadzone: float = 0.05  # |a_r| below this -> runner gets nothing
    min_fraction: float = 0.01  # f below this -> no bets this step
    temperature: float = 0.25  # softmax temperature over |a_r|
    max_fraction: float = 1.0  # cap on f (share of available funds wagered per step)
    min_best_usd: float = 0.0  # liquidity gate ($ at best back + lay); 0 = off
    liq_window_s: float = 10.0  # trailing window the gate averages over
    gate_inputs: bool = True  # append (above cut, distance to cut) to each runner's features when gated


class ContinuousAllocEnv(BetfairPreRaceEnv):
    def __init__(self, tapes, cfg: ContinuousConfig | None = None, **kw):
        super().__init__(tapes, cfg or ContinuousConfig(), **kw)
        lo = np.full(N_CONT, -1.0, np.float32)
        lo[-1] = 0.0
        self.action_space = gym.spaces.Box(lo, np.ones(N_CONT, np.float32), dtype=np.float32)
        spaces = dict(self.observation_space.spaces, act_mask=gym.spaces.MultiBinary(R_MAX))
        if self.gate_inputs:
            r = spaces["runners"]
            spaces["runners"] = gym.spaces.Box(-10, 10, (R_MAX, r.shape[1] + N_GATE_INPUTS), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    @property
    def gate_inputs(self) -> bool:
        return self.cfg.min_best_usd > 0 and self.cfg.gate_inputs

    def best_usd(self, step: int | None = None) -> np.ndarray:
        """(n_runners,) money at the best back + lay prices, trailing mean over liq_window_s (live)."""
        t = self.tape
        s = self.ex.step if step is None else step
        k = max(1, int(round(self.cfg.liq_window_s / t.dt)))
        lo = max(0, s - k + 1)
        best = t.back_size[lo:s + 1, :, 0].astype(np.float64) + t.lay_size[lo:s + 1, :, 0]
        return best.mean(0)

    def liquid(self, step: int | None = None) -> np.ndarray:
        """Runners (n_runners,) passing the liquidity gate at ``step`` (all True if the gate is off)."""
        if self.cfg.min_best_usd <= 0:
            return np.ones(self.tape.n_runners, bool)
        return self.best_usd(step) >= self.cfg.min_best_usd

    def held(self) -> np.ndarray:
        return np.abs(self.ex.W - self.ex.L) > 1e-6

    def _obs(self):
        o = super()._obs()
        am = o["mask"].copy()
        if self.cfg.min_best_usd > 0:
            n = min(self.tape.n_runners, R_MAX)
            b = self.best_usd()[:n]
            liq = b >= self.cfg.min_best_usd
            ok = np.zeros(R_MAX, bool)
            ok[:n] = liq | self.held()[:n]
            am &= ok.astype(am.dtype)
            if self.gate_inputs:
                g = np.zeros((R_MAX, N_GATE_INPUTS), np.float32)
                g[:n, 0] = liq
                g[:n, 1] = np.clip(np.log(np.maximum(b, 1e-2) / self.cfg.min_best_usd), -3, 3) / 3
                g *= o["mask"][:, None]
                o["runners"] = np.concatenate([o["runners"], g], 1)
        o["act_mask"] = am
        return o

    def _apply(self, action):
        ex, cfg = self.ex, self.cfg
        a = np.clip(np.asarray(action, np.float64)[:R_MAX], -1, 1)
        f = float(np.clip(action[R_MAX], 0.0, cfg.max_fraction))
        ex.cancel_all()  # aggressive-only: leftovers from the last decision are pulled
        if f < cfg.min_fraction:
            return
        R = min(self.tape.n_runners, R_MAX)
        liq = self.liquid()
        expo = ex.W - ex.L  # > 0 long (backed), < 0 short (laid)
        elig = []
        for r in range(R):
            if abs(a[r]) < cfg.deadzone or not self.tape.active[ex.step, r] or ex.void_runner[r]:
                continue
            if not liq[r] and not ((a[r] > 0 and expo[r] < -1e-6) or (a[r] < 0 and expo[r] > 1e-6)):
                continue  # below the gate: only trades that reduce a held position
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
                stake = amt
            else:
                stake = amt / max(PRICES[bl] - 1.0, 0.01)
            if not liq[r]:  # reduce only: never past flat
                hp = ex.hedge_plan(r)
                if hp is None:
                    continue
                stake = min(stake, hp[2])
            o = ex.submit(r, BACK if a[r] > 0 else LAY, bb if a[r] > 0 else bl, stake)
            if o is not None:
                self.n_opens += 1
