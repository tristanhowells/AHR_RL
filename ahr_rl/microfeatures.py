"""Engineered order-book / trade-flow features ("microstructure" features).

All are causal (use only data up to step s) and expressed in ticks or ratios so
they mean the same thing at 2.0 as at 20.0. Computed for a whole tape at once
as [T, R] arrays.

Conventions (Betfair stream):
  atb = "available to back"  = money from LAYERS waiting to be matched (tape.back_*)
  atl = "available to lay"   = money from BACKERS waiting to be matched (tape.lay_*)
  best back price bb = highest atb;  best lay price bl = lowest atl
  Lower tick = shorter price = runner more fancied.

Features
  wom{k}          (atb_k - atl_k) / (atb_k + atl_k) over the best k levels, k = 1, 3, 5, 8.
                  > 0: more money on the "back" column (layers offering) than the "lay" column.
  wom3_chg{W}     change of wom3 over the last W seconds (liquidity flow)
  max_{W}, min_{W} highest / lowest matched price in the last W s, in ticks relative to mid
                  (W = 30, 120 or "sess" = since the tape started)
  rng_pos_{W}     (mid - min) / (max - min): where the current price sits in its matched range
  wap_{W}         volume-weighted average matched price minus mid, in ticks
  ltp_wap_{W}     last traded price minus WAP, in ticks
  flow_{W}        aggressor imbalance: (backer-initiated - layer-initiated volume) / total,
                  where a trade at/below the previous best back price was a backer taking
                  it, at/above the previous best lay price a layer taking it
  lvol_{W}        log(1 + traded volume in window)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

WINDOWS_S = (30, 120)
FLOW_WINDOWS_S = (10, 30, 120)
WOM_LEVELS = (1, 3, 5, 8)


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(x, axis=0)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def _wom(bs, ls, k):
    b, l = bs[:, :, :k].sum(-1), ls[:, :, :k].sum(-1)
    return np.where(b + l > 0, (b - l) / np.maximum(b + l, 1e-9), 0.0)


def compute(tape, mid: np.ndarray) -> dict[str, np.ndarray]:
    """mid: [T, R] mid price in ticks (TapeHistory.mid). Returns name -> [T, R] float32."""
    T, R = mid.shape
    dt = tape.dt
    out: dict[str, np.ndarray] = {}
    bs, ls = tape.back_size.astype(np.float64), tape.lay_size.astype(np.float64)
    for k in WOM_LEVELS:
        out[f"wom{k}"] = _wom(bs, ls, min(k, bs.shape[2]))
    w3 = out["wom3"]
    for W in (10, 30):
        n = int(W / dt)
        prev = np.vstack([np.repeat(w3[:1], n, 0), w3[:-n]]) if n < T else np.repeat(w3[:1], T, 0)
        out[f"wom3_chg{W}"] = w3 - prev

    # ---- trades per step (volume, volume*tick, max/min tick, aggressor-signed volume)
    vol = np.zeros((T, R))
    vtk = np.zeros((T, R))
    hi = np.full((T, R), -np.inf)
    lo = np.full((T, R), np.inf)
    signed = np.zeros((T, R))
    s_idx, r_idx = tape.trade_step.astype(np.int64), tape.trade_runner.astype(np.int64)
    keep = r_idx < R
    s_idx, r_idx = s_idx[keep], r_idx[keep]
    tk = tape.trade_tick.astype(np.float64)[keep]
    v = tape.trade_vol.astype(np.float64)[keep]
    np.add.at(vol, (s_idx, r_idx), v)
    np.add.at(vtk, (s_idx, r_idx), v * tk)
    np.maximum.at(hi, (s_idx, r_idx), tk)
    np.minimum.at(lo, (s_idx, r_idx), tk)
    prev = np.maximum(s_idx - 1, 0)
    bb_prev = tape.back_tick[prev, r_idx, 0].astype(np.float64)
    bl_prev = tape.lay_tick[prev, r_idx, 0].astype(np.float64)
    sign = np.where((bb_prev >= 0) & (tk <= bb_prev), 1.0, np.where((bl_prev >= 0) & (tk >= bl_prev), -1.0, 0.0))
    np.add.at(signed, (s_idx, r_idx), sign * v)

    valid_mid = mid >= 0
    ltp = tape.ltp_tick.astype(np.float64)
    for W in list(WINDOWS_S) + ["sess"]:
        if W == "sess":
            cv, cvt = np.cumsum(vol, 0), np.cumsum(vtk, 0)
            mx = np.maximum.accumulate(hi, 0)
            mn = np.minimum.accumulate(lo, 0)
        else:
            n = int(W / dt)
            cv, cvt = _rolling_sum(vol, n), _rolling_sum(vtk, n)
            # origin shifts the window so it only looks backwards
            mx = maximum_filter1d(hi, size=n, axis=0, origin=(n - 1) // 2, mode="nearest")
            mn = minimum_filter1d(lo, size=n, axis=0, origin=(n - 1) // 2, mode="nearest")
        has = np.isfinite(mx) & np.isfinite(mn) & valid_mid
        wap = np.where(cv > 0, cvt / np.maximum(cv, 1e-9), np.nan)
        out[f"max_{W}"] = np.where(has, mx - mid, 0.0)
        out[f"min_{W}"] = np.where(has, mn - mid, 0.0)
        rng = np.where(has, mx - mn, 0.0)
        out[f"rng_pos_{W}"] = np.where(rng > 0, (mid - mn) / np.maximum(rng, 1e-9), 0.5)
        out[f"wap_{W}"] = np.where(np.isfinite(wap) & valid_mid, wap - mid, 0.0)
        out[f"ltp_wap_{W}"] = np.where(np.isfinite(wap) & (ltp >= 0), ltp - wap, 0.0)
    for W in FLOW_WINDOWS_S:
        n = int(W / dt)
        sv, tv = _rolling_sum(signed, n), _rolling_sum(vol, n)
        out[f"flow_{W}"] = np.where(tv > 0, sv / np.maximum(tv, 1e-9), 0.0)
        out[f"lvol_{W}"] = np.log1p(tv)
    return {k: np.clip(np.nan_to_num(a, nan=0.0), -50, 50).astype(np.float32) for k, a in out.items()}
