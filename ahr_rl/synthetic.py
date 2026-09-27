"""Synthetic tapes with a planted, learnable edge - a sanity check that the RL
stack can find profit when profit exists (real markets may simply not offer much).

In each synthetic race one randomly chosen runner (a "steamer") shortens by one
tick every ``trend_every_s`` seconds; everything else random-walks with no drift.
Backing the steamer and exiting a tick or two lower is profitable; the agent
must *recognise* the steamer from its features (the runner index is random).
"""
from __future__ import annotations

import os

import numpy as np

from .ladder import N_TICKS
from .tape import Tape


def make_synthetic_tape(seed: int, n_runners: int = 6, dt: float = 0.5, pre_s: float = 600.0,
                        post_s: float = 60.0, trend_every_s: float = 6.0, k_levels: int = 8,
                        level_size: float = 40.0, trade_rate: float = 0.3) -> Tape:
    rng = np.random.default_rng(seed)
    T = int((pre_s + post_s) / dt)
    R, K = n_runners, k_levels
    t_rel = (-pre_s + np.arange(T) * dt).astype(np.float32)
    steamer = int(rng.integers(R))
    base = rng.integers(150, 230, size=R)  # ticks ~3.0 .. 20
    mid = np.zeros((T, R), np.int64)
    for r in range(R):
        if r == steamer:
            drift = -(np.arange(T) * dt // trend_every_s).astype(np.int64)
            mid[:, r] = base[r] + drift
        else:
            steps = rng.choice([-1, 0, 1], size=T, p=[0.004, 0.992, 0.004])
            mid[:, r] = base[r] + np.cumsum(steps)
    mid = np.clip(mid, K + 2, N_TICKS - K - 3)
    bt = np.zeros((T, R, K), np.int16)
    lt = np.zeros((T, R, K), np.int16)
    for k in range(K):
        bt[:, :, k] = mid - k
        lt[:, :, k] = mid + 1 + k
    bs = rng.uniform(0.5, 1.5, size=(T, R, K)).astype(np.float32) * level_size
    ls = rng.uniform(0.5, 1.5, size=(T, R, K)).astype(np.float32) * level_size
    trades = []
    for s in range(1, T):
        for r in range(R):
            if rng.random() < trade_rate:
                side = rng.integers(2)
                tk = bt[s, r, 0] if side == 0 else lt[s, r, 0]
                trades.append((s, r, tk, float(rng.uniform(1, 10))))
    tr = np.array(trades, np.float64).reshape(-1, 4)
    return Tape(
        market_id=f"syn.{seed}", name=f"20990101_synthetic_{seed}", dt=dt, t_rel=t_rel,
        back_tick=bt, back_size=bs, lay_tick=lt, lay_size=ls,
        ltp_tick=mid.astype(np.int16), tv=np.cumsum(np.ones((T, R), np.float32), 0),
        spn=np.zeros((T, R), np.float32), active=np.ones((T, R), bool), suspended=np.zeros(T, bool),
        total_matched=np.arange(T, dtype=np.float32) * 10,
        trade_step=tr[:, 0].astype(np.int32), trade_runner=tr[:, 1].astype(np.int16),
        trade_tick=tr[:, 2].astype(np.int16), trade_vol=tr[:, 3].astype(np.float32),
        removal_step=np.zeros(0, np.int32), removal_runner=np.zeros(0, np.int16),
        removal_factor=np.zeros(0, np.float32), selection_ids=np.arange(R, dtype=np.int64),
        base_rate=8.0, went_in_play=True, winner=int(rng.integers(R)), bsp=np.zeros(R, np.float32),
    )


def write_synthetic(out_dir: str, n: int, seed0: int = 0) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(out_dir, f"2099{(i % 28) + 1:04d}_syn_{seed0 + i}.npz")
        make_synthetic_tape(seed0 + i).save(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic"
    print(len(write_synthetic(out, int(sys.argv[2]) if len(sys.argv) > 2 else 120)), "synthetic tapes")
