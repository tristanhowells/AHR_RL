"""Causal per-runner + global features computed from a stream tape (ahr_rl.tape.Tape).

A tape is the pre-race market on a fixed 0.5 s grid: 8-level atb/atl ladders,
every trade (runner, tick, single-counted volume), projected BSP, scratchings
and, optionally, catalogue form features. All features are computed once per
tape for every grid step; row s uses only ladder rows <= s and trades in steps
<= s, so there is no look-ahead. Settlement fields (winner, bsp) are never used.
"""
from dataclasses import dataclass
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ahr_rl.catalogue import N_STATIC
from ahr_rl.ladder import PRICES

MAX_RUNNERS = 24
RET_LAGS_S = (5, 15, 30, 60, 120, 300)
WIN_S = 60.0          # rolling VWAP / matched-range window
FLOW_S = 30.0         # trade-flow window
RECENT_S = 5.0        # "volume just now" window

RUNNER_FEATURES = (
    [f"log_back_p{k}" for k in (1, 2, 3)] + [f"log_lay_p{k}" for k in (1, 2, 3)]
    + [f"log1p_back_s{k}" for k in (1, 2, 3)] + [f"log1p_lay_s{k}" for k in (1, 2, 3)]
    + ["has_back", "has_lay", "log1p_back_depth", "log1p_lay_depth",
       "log_ltp", "log_fair", "fair_prob", "spread_ticks",
       "wom_l1", "wom_l3", "wom_all", "book_wap_rel",
       "vwap_rel", "vwap_win_rel", "max_matched_rel", "min_matched_rel",
       "max_matched_win_rel", "min_matched_win_rel",
       "log1p_tv", "log1p_vol_win", "log1p_vol_recent", "flow_imbalance", "vol_share",
       "log1p_secs_since_trade"]
    + [f"ret_{l}s" for l in RET_LAGS_S]
    + ["ltp_rel", "spn_rel", "has_spn", "rank_norm", "is_fav", "active"]
    + [f"static_{i}" for i in range(N_STATIC)]
)
N_RUNNER_FEATURES = len(RUNNER_FEATURES)

GLOBAL_FEATURES = [
    "t_rel", "log1p_total_matched", "dlog_total_matched_30s", "back_overround", "lay_overround",
    "prob_entropy", "fav_prob_gap", "mean_spread_ticks", "n_active", "runner_count",
    "commission", "suspended", "has_static",
]
N_GLOBAL_FEATURES = len(GLOBAL_FEATURES)


@dataclass
class TapeFeatures:
    runner: np.ndarray    # [T, R, N_RUNNER_FEATURES] float32
    glob: np.ndarray      # [T, N_GLOBAL_FEATURES] float32
    fair: np.ndarray      # [T, R] fair price (probability-space microprice), 0 = none
    rank: np.ndarray      # [T, R] 1 = favourite, R + 1 = inactive / no price
    priced: np.ndarray    # [T, R] bool: active with at least one price


def _ffill(a):
    """Forward-fill NaN along axis 0."""
    idx = np.where(~np.isnan(a), np.arange(a.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = a[idx, np.arange(a.shape[1])[None, :]]
    return out


def _window_reduce(a, n, fn):
    """Trailing window of n rows over axis 0; all-NaN windows give NaN (silently)."""
    x = np.concatenate([np.full((n - 1,) + a.shape[1:], np.nan), a])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(sliding_window_view(x, n, axis=0), axis=-1)


def _window_sum(cum, n):
    """Trailing n-row sum from a cumulative array."""
    out = cum.copy()
    out[n:] -= cum[:-n]
    return out


def fair_price(bt, bs, lt, ls):
    """Probability-space microprice from level 1. Inputs [..]; returns price, 0 = no book.

    Backing at the atb price p_b buys the outcome at probability 1/p_b; laying at
    the atl price p_l sells it at 1/p_l. Weighting each by the size on the other
    side gives the microprice in probability, which is what the green value is
    linear in.
    """
    has_b, has_l = bt >= 0, lt >= 0
    pb = np.where(has_b, PRICES[np.clip(bt, 0, None)], np.nan)
    pl = np.where(has_l, PRICES[np.clip(lt, 0, None)], np.nan)
    with np.errstate(all="ignore"):
        w = bs + ls
        prob = np.where(w > 0, (ls / pb + bs / pl) / w, np.nan)
        prob = np.where(has_b & has_l, prob, np.where(has_b, 1 / pb, np.where(has_l, 1 / pl, np.nan)))
        return np.nan_to_num(1.0 / prob, nan=0.0)


def build_features(tape, commission):
    T, R = tape.n_steps, tape.n_runners
    dt = float(tape.dt)
    steps = lambda s: max(1, int(round(s / dt)))
    bt, bs = tape.back_tick.astype(np.int64), tape.back_size.astype(np.float64)
    lt, ls = tape.lay_tick.astype(np.int64), tape.lay_size.astype(np.float64)
    bs = np.where(bt >= 0, bs, 0.0)
    ls = np.where(lt >= 0, ls, 0.0)
    has_b, has_l = bt[..., 0] >= 0, lt[..., 0] >= 0
    priced = tape.active & (has_b | has_l)

    ltp_tick = tape.ltp_tick.astype(np.int64)
    ltp_p = np.where(ltp_tick >= 0, PRICES[np.clip(ltp_tick, 0, None)], np.nan)
    fair = fair_price(bt[..., 0], bs[..., 0], lt[..., 0], ls[..., 0])
    fair = np.where(fair > 0, fair, np.nan_to_num(ltp_p, nan=0.0))
    fair = np.where(priced, fair, 0.0)
    fair_f = np.where(fair > 0, fair, np.nan)
    fair_ff = _ffill(fair_f)                                 # for lagged returns
    ref = np.where(fair > 0, fair, 1000.0)                   # denominator for *_rel features

    order = np.argsort(np.where(priced, ref, np.inf), axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(T)[:, None], order] = np.arange(1, R + 1)[None, :]
    rank = np.where(priced, rank, R + 1)

    # ---- trades -> per-step per-runner arrays (exact prices, single-counted volume)
    vol = np.zeros((T, R))
    pv = np.zeros((T, R))
    bvol = np.zeros((T, R))       # backers taking atl (trade at/above previous best lay)
    lvol = np.zeros((T, R))       # layers taking atb (trade at/below previous best back)
    hi = np.full((T, R), np.nan)
    lo = np.full((T, R), np.nan)
    if len(tape.trade_step):
        s_, r_ = tape.trade_step.astype(np.int64), tape.trade_runner.astype(np.int64)
        tk, v = tape.trade_tick.astype(np.int64), tape.trade_vol.astype(np.float64)
        keep = (s_ < T) & (r_ < R) & (tk >= 0)
        s_, r_, tk, v = s_[keep], r_[keep], tk[keep], v[keep]
        p = PRICES[tk]
        np.add.at(vol, (s_, r_), v)
        np.add.at(pv, (s_, r_), v * p)
        prev = np.maximum(s_ - 1, 0)
        pbl, pbb = lt[prev, r_, 0], bt[prev, r_, 0]
        is_b = (pbl >= 0) & (tk >= pbl)
        is_l = (pbb >= 0) & (tk <= pbb)
        amb = ~is_b & ~is_l
        np.add.at(bvol, (s_, r_), v * (is_b + 0.5 * amb))
        np.add.at(lvol, (s_, r_), v * (is_l + 0.5 * amb))
        np.fmax.at(hi, (s_, r_), p)
        np.fmin.at(lo, (s_, r_), p)
    cv, cpv = np.cumsum(vol, 0), np.cumsum(pv, 0)
    w = steps(WIN_S)
    vol_w, pv_w = _window_sum(cv, w), _window_sum(cpv, w)
    with np.errstate(all="ignore"):
        vwap = np.where(cv > 0, cpv / cv, ref)
        vwap_w = np.where(vol_w > 1e-9, pv_w / vol_w, ref)
        max_m = np.fmax.accumulate(hi, axis=0)
        min_m = np.fmin.accumulate(lo, axis=0)
    max_m, min_m = np.where(np.isnan(max_m), ref, max_m), np.where(np.isnan(min_m), ref, min_m)
    max_w, min_w = _window_reduce(hi, w, np.nanmax), _window_reduce(lo, w, np.nanmin)
    max_w, min_w = np.where(np.isnan(max_w), max_m, max_w), np.where(np.isnan(min_w), min_m, min_w)
    fb = _window_sum(np.cumsum(bvol, 0), steps(FLOW_S))
    fl = _window_sum(np.cumsum(lvol, 0), steps(FLOW_S))
    flow = np.where(fb + fl > 1e-9, (fb - fl) / np.maximum(fb + fl, 1e-9), 0.0)
    vol_recent = _window_sum(cv, steps(RECENT_S))
    last_trade = np.where(vol > 0, np.arange(T)[:, None], -1)
    np.maximum.accumulate(last_trade, axis=0, out=last_trade)
    secs_since = np.where(last_trade >= 0, (np.arange(T)[:, None] - last_trade) * dt, 3600.0)
    tv = np.maximum(cv, tape.tv.astype(np.float64))
    tv_tot = tv.sum(1, keepdims=True)

    # ---- book features
    def lp(tk):
        return np.where(tk >= 0, np.log(PRICES[np.clip(tk, 0, None)]), 0.0)

    sb3, sl3 = bs[..., :3].sum(-1), ls[..., :3].sum(-1)
    sba, sla = bs.sum(-1), ls.sum(-1)

    def wom(a, b):
        return np.where(a + b > 0, a / np.maximum(a + b, 1e-9), 0.5)

    pw = (np.where(bt >= 0, PRICES[np.clip(bt, 0, None)], 0) * bs).sum(-1) \
        + (np.where(lt >= 0, PRICES[np.clip(lt, 0, None)], 0) * ls).sum(-1)
    book_wap = np.where(sba + sla > 0, pw / np.maximum(sba + sla, 1e-9), ref)
    spread = np.where(has_b & has_l, lt[..., 0] - bt[..., 0], 0)
    prob = np.where(fair > 0, 1.0 / np.where(fair > 0, fair, 1.0), 0.0)
    fair_prob = prob / np.maximum(prob.sum(1, keepdims=True), 1e-9)
    spn = tape.spn.astype(np.float64)

    def rel(x):
        with np.errstate(all="ignore"):
            return np.nan_to_num(np.log(np.clip(x, 1.01, 1000.0) / ref))

    rets = {}
    for lag in RET_LAGS_S:
        k = steps(lag)
        r = np.zeros((T, R))
        with np.errstate(all="ignore"):
            r[k:] = np.log(fair_ff[k:] / fair_ff[:-k])
        rets[f"ret_{lag}s"] = 10 * np.nan_to_num(r)

    static = getattr(tape, "static", None)
    has_static = static is not None and static.shape[-1] == N_STATIC
    st = np.zeros((T, R, N_STATIC))
    if has_static:
        st[:] = np.nan_to_num(np.asarray(static, np.float64)[:R])[None]

    f = {
        **{f"log_back_p{k + 1}": lp(bt[..., k]) for k in range(3)},
        **{f"log_lay_p{k + 1}": lp(lt[..., k]) for k in range(3)},
        **{f"log1p_back_s{k + 1}": np.log1p(bs[..., k]) for k in range(3)},
        **{f"log1p_lay_s{k + 1}": np.log1p(ls[..., k]) for k in range(3)},
        "has_back": has_b, "has_lay": has_l,
        "log1p_back_depth": np.log1p(sba), "log1p_lay_depth": np.log1p(sla),
        "log_ltp": np.nan_to_num(np.log(np.where(np.isnan(ltp_p), 1.0, ltp_p))),
        "log_fair": np.log(np.where(fair > 0, fair, 1.0)),
        "fair_prob": fair_prob, "spread_ticks": np.clip(spread, 0, 50) / 10.0,
        "wom_l1": wom(bs[..., 0], ls[..., 0]), "wom_l3": wom(sb3, sl3), "wom_all": wom(sba, sla),
        "book_wap_rel": rel(book_wap), "vwap_rel": rel(vwap), "vwap_win_rel": rel(vwap_w),
        "max_matched_rel": rel(max_m), "min_matched_rel": rel(min_m),
        "max_matched_win_rel": rel(max_w), "min_matched_win_rel": rel(min_w),
        "log1p_tv": np.log1p(tv), "log1p_vol_win": np.log1p(vol_w),
        "log1p_vol_recent": np.log1p(vol_recent), "flow_imbalance": flow,
        "vol_share": np.where(tv_tot > 0, tv / np.maximum(tv_tot, 1e-9), 0.0),
        "log1p_secs_since_trade": np.log1p(secs_since),
        **rets,
        "ltp_rel": np.where(np.isnan(ltp_p), 0.0, rel(np.nan_to_num(ltp_p, nan=1.0))),
        "spn_rel": np.where(spn > 1.0, rel(spn), 0.0), "has_spn": spn > 1.0,
        "rank_norm": rank / float(MAX_RUNNERS), "is_fav": rank == 1, "active": priced,
        **{f"static_{i}": st[..., i] for i in range(N_STATIC)},
    }
    runner = np.stack([np.asarray(f[n], np.float64) for n in RUNNER_FEATURES], axis=2)
    runner = np.where(priced[..., None], runner, 0.0)
    runner = np.nan_to_num(runner, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---- global features
    with np.errstate(all="ignore"):
        bo = np.where(has_b & tape.active, 1.0 / PRICES[np.clip(bt[..., 0], 0, None)], 0).sum(1)
        lo_ = np.where(has_l & tape.active, 1.0 / PRICES[np.clip(lt[..., 0], 0, None)], 0).sum(1)
        ent = -(np.where(fair_prob > 0, fair_prob * np.log(np.maximum(fair_prob, 1e-12)), 0)).sum(1)
    sp = np.sort(fair_prob, axis=1)[:, ::-1]
    fav_gap = sp[:, 0] - (sp[:, 1] if R > 1 else 0)
    n_pr = priced.sum(1)
    tm = tape.total_matched.astype(np.float64)
    k30 = steps(30)
    dtm = np.zeros(T)
    dtm[k30:] = np.log1p(np.maximum(tm[k30:] - tm[:-k30], 0))
    g = {
        "t_rel": np.clip(tape.t_rel.astype(np.float64), -900, 600) / 600.0,
        "log1p_total_matched": np.log1p(tm) / 10.0, "dlog_total_matched_30s": dtm / 10.0,
        "back_overround": bo - 1.0, "lay_overround": lo_ - 1.0,
        "prob_entropy": ent, "fav_prob_gap": fav_gap,
        "mean_spread_ticks": np.where(n_pr > 0, (np.clip(spread, 0, 50) * priced).sum(1)
                                      / np.maximum(n_pr, 1), 0) / 10.0,
        "n_active": n_pr / float(MAX_RUNNERS), "runner_count": np.full(T, R / float(MAX_RUNNERS)),
        "commission": np.full(T, commission), "suspended": tape.suspended.astype(np.float64),
        "has_static": np.full(T, float(has_static)),
    }
    glob = np.stack([g[n] for n in GLOBAL_FEATURES], axis=1)
    glob = np.nan_to_num(glob, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return TapeFeatures(runner=runner, glob=glob, fair=fair, rank=rank, priced=priced)
