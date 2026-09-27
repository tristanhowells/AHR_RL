"""Observation features. Pure functions of (tape history up to `step`, account state)
so the identical code runs in training and in the live bot.

Per-runner token (N_RUNNER_FEATURES) + global vector (N_GLOBAL_FEATURES). Runner
order is the tape's (sort priority); the policy is permutation-equivariant, so
order carries no meaning beyond identity.
"""
from __future__ import annotations

import numpy as np

from .ladder import N_TICKS, PRICES

R_MAX = 24
N_RUNNER_FEATURES = 38
N_GLOBAL_FEATURES = 12
_MID_LAGS_S = (5, 15, 30, 60)
_VOL_WIN_S = (5, 30)


class TapeHistory:
    """Per-episode derived series (mid price in ticks, cumulative volume)."""

    def __init__(self, tape):
        self.tape = tape
        bt = tape.back_tick[:, :, 0].astype(np.float32)
        lt = tape.lay_tick[:, :, 0].astype(np.float32)
        ltp = tape.ltp_tick.astype(np.float32)
        both = (bt >= 0) & (lt >= 0)
        mid = np.where(both, (bt + lt) / 2, np.where(bt >= 0, bt, np.where(lt >= 0, lt, ltp)))
        mid[mid < 0] = np.nan
        # forward fill gaps so lags are defined
        for r in range(mid.shape[1]):
            col = mid[:, r]
            idx = np.where(~np.isnan(col), np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            mid[:, r] = col[idx]
        self.mid = np.nan_to_num(mid, nan=-1.0)
        T, R = mid.shape
        vol = np.zeros((T, R), np.float32)
        np.add.at(vol, (tape.trade_step, tape.trade_runner.astype(np.int64)), tape.trade_vol)
        self.cumvol = np.cumsum(vol, axis=0)
        self.dt = tape.dt


def runner_features(h: TapeHistory, step: int, ex) -> tuple[np.ndarray, np.ndarray]:
    t = h.tape
    R = min(t.n_runners, R_MAX)
    F = np.zeros((R_MAX, N_RUNNER_FEATURES), np.float32)
    mask = np.zeros(R_MAX, bool)
    B = ex.cfg.bankroll
    bt, bs = t.back_tick[step, :R], t.back_size[step, :R]
    lt, ls = t.lay_tick[step, :R], t.lay_size[step, :R]
    act = t.active[step, :R] & ~ex.void_runner[:R] & ((bt[:, 0] >= 0) | (lt[:, 0] >= 0))
    mask[:R] = act
    mid = h.mid[step, :R]
    pb = np.where(bt[:, 0] >= 0, PRICES[np.clip(bt[:, 0], 0, None)], 0.0)
    pl = np.where(lt[:, 0] >= 0, PRICES[np.clip(lt[:, 0], 0, None)], 0.0)
    pm = np.where(mid >= 0, np.interp(mid, np.arange(N_TICKS), PRICES), 0.0)
    F[:R, 0] = act
    F[:R, 1] = np.where(bt[:, 0] >= 0, bt[:, 0] / N_TICKS, 0)
    F[:R, 2] = np.where(lt[:, 0] >= 0, lt[:, 0] / N_TICKS, 0)
    spread = np.where((bt[:, 0] >= 0) & (lt[:, 0] >= 0), lt[:, 0] - bt[:, 0], 10)
    F[:R, 3] = np.clip(spread, 0, 20) / 10
    F[:R, 4] = np.where(pm > 0, 1.0 / np.maximum(pm, 1.01), 0)
    F[:R, 5:8] = np.log1p(bs[:, :3]) / 6
    F[:R, 8:11] = np.log1p(ls[:, :3]) / 6
    sb, sl = bs[:, :3].sum(1), ls[:, :3].sum(1)
    F[:R, 11] = (sb - sl) / np.maximum(sb + sl, 1e-6)
    ltp = t.ltp_tick[step, :R].astype(np.float32)
    F[:R, 12] = np.where((ltp >= 0) & (mid >= 0), np.clip(ltp - mid, -10, 10) / 5, 0)
    for i, lag in enumerate(_MID_LAGS_S):
        s0 = max(0, step - int(lag / h.dt))
        m0 = h.mid[s0, :R]
        F[:R, 13 + i] = np.where((m0 >= 0) & (mid >= 0), np.clip(mid - m0, -20, 20) / 5, 0)
    for i, win in enumerate(_VOL_WIN_S):
        s0 = max(0, step - int(win / h.dt))
        F[:R, 17 + i] = np.log1p(h.cumvol[step, :R] - h.cumvol[s0, :R]) / 6
    F[:R, 19] = np.log1p(t.tv[step, :R]) / 10
    spn = t.spn[step, :R]
    has_spn = (spn > 1.0) & (pm > 0)
    spn_tick = np.searchsorted(PRICES, np.clip(spn, 1.01, 1000))
    F[:R, 20] = np.where(has_spn, np.clip(spn_tick - mid, -30, 30) / 10, 0)
    F[:R, 21] = has_spn
    F[:R, 22] = ex.W[:R] / B
    F[:R, 23] = ex.L[:R] / B
    F[:R, 24] = (ex.W[:R] - ex.L[:R]) / B
    for o in ex.orders:
        if o.runner >= R:
            continue
        if o.side == 1:
            F[o.runner, 25] += o.size / B
            ref = lt[o.runner, 0] if lt[o.runner, 0] >= 0 else o.tick
            F[o.runner, 26] = np.clip(o.tick - ref, -10, 10) / 5
            F[o.runner, 27] = np.log1p(min(o.queue_ahead, 1e5)) / 6
        else:
            F[o.runner, 28] += o.size / B
            ref = bt[o.runner, 0] if bt[o.runner, 0] >= 0 else o.tick
            F[o.runner, 29] = np.clip(o.tick - ref, -10, 10) / 5
            F[o.runner, 30] = np.log1p(min(o.queue_ahead, 1e5)) / 6
    # favourite rank and short-term volatility
    order = np.argsort(np.where(act, pm, 1e9))
    rank = np.empty(R)
    rank[order] = np.arange(R)
    F[:R, 31] = rank / 20
    s0 = max(0, step - int(60 / h.dt))
    win = h.mid[s0 : step + 1, :R]
    F[:R, 32] = np.clip(np.diff(win, axis=0).std(0) if len(win) > 2 else 0, 0, 5) / 2
    F[:R, 33] = np.where((pb > 0) & (pl > 0), (pl - pb) / np.maximum(pb, 1.01), 0)
    for r, b in getattr(ex, "brackets", {}).items():
        if r >= R:
            continue
        F[r, 34] = 1.0
        F[r, 35] = float(b.closing)
        F[r, 36] = min((step - b.opened_step) * h.dt / 120.0, 3.0)
        F[r, 37] = b.side * b.tp / 4.0
    F[~mask] = 0
    return F, mask


def global_features(h: TapeHistory, step: int, ex, green_value: float) -> np.ndarray:
    t = h.tape
    B = ex.cfg.bankroll
    g = np.zeros(N_GLOBAL_FEATURES, np.float32)
    act = t.active[step] & ~ex.void_runner
    bt, lt = t.back_tick[step, :, 0], t.lay_tick[step, :, 0]
    g[0] = t.t_rel[step] / 600
    g[1] = step / 1200
    g[2] = (1.0 / PRICES[bt[act & (bt >= 0)]]).sum() - 1.0 if (act & (bt >= 0)).any() else 0
    g[3] = (1.0 / PRICES[lt[act & (lt >= 0)]]).sum() - 1.0 if (act & (lt >= 0)).any() else 0
    g[4] = np.log1p(t.total_matched[step]) / 12
    g[5] = act.sum() / 20
    g[6] = ex.available_funds() / B
    g[7] = ex.worst_case() / B
    g[8] = green_value / B
    g[9] = float(t.suspended[step])
    g[10] = ex.commission
    g[11] = np.log1p(len(ex.fills)) / 4
    return np.clip(g, -5, 5)
