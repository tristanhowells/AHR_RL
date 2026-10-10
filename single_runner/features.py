"""Race loading + causal feature engineering for the single-runner environment.

A race parquet (schema 1.3.x, one row per ~1.3 s stream snapshot, columns
`run[i].*` per runner) is turned into dense numpy arrays once, so the
environment's step() is just array indexing.

Every feature at row t uses only rows <= t (cumulative / trailing-window ops),
so there is no look-ahead. Result columns (`run[i].is_winner`, `result_*`) are
never used as features; `is_winner` is kept separately for diagnostics only.
"""
from dataclasses import dataclass, field
import re
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd

from .ladder import price_to_tick

MAX_RUNNERS = 24
LEVELS = 3
ROLL_STEPS = 45          # ~60 s at the 1.33 s capture interval
RET_LAGS = (1, 5, 20, 60)

RUNNER_FEATURES = (
    [f"log_back_p{k}" for k in range(1, LEVELS + 1)]
    + [f"log_lay_p{k}" for k in range(1, LEVELS + 1)]
    + [f"log1p_back_s{k}" for k in range(1, LEVELS + 1)]
    + [f"log1p_lay_s{k}" for k in range(1, LEVELS + 1)]
    + ["has_back", "has_lay", "log_ltp", "log_micro", "prob_implied", "fair_prob",
       "rel_spread", "spread_ticks", "ob_imbalance", "wom_l1", "wom_3l",
       "book_wap_rel", "vwap_rel", "vwap_roll_rel",
       "max_matched_rel", "min_matched_rel", "max_matched_roll_rel", "min_matched_roll_rel",
       "log1p_tv", "log1p_tv_60s", "log1p_dvol", "vol_share", "log1p_secs_since_trade",
       "ret_std_5s", "ret_std_20s"]
    + [f"ret_{l}" for l in RET_LAGS]
    + ["ltp_rel", "rank_norm", "is_fav", "active"]
)
N_RUNNER_FEATURES = len(RUNNER_FEATURES)

GLOBAL_FEATURES = [
    "secs_to_off", "log1p_total_matched", "dlog_total_matched", "back_overround",
    "lay_overround", "overround_gap", "prob_entropy", "fav_prob_gap",
    "spread_mean_run", "spread_std_run", "runner_count", "n_active", "commission",
    "is_flat", "is_harness", "is_other_code", "distance", "dt",
]
N_GLOBAL_FEATURES = len(GLOBAL_FEATURES)

_RUN_RE = re.compile(r"run\[(\d+)\]\.")


@dataclass
class RaceData:
    path: str
    market_id: str
    race_date: str
    commission: float
    n_runners: int
    T: int                       # decision steps (pre-race OPEN rows)
    in_play_found: bool
    # raw book used by the matching engine, shape [T, R] (NaN = no offer)
    back_p: np.ndarray           # [T, R, 3] best available-to-BACK prices (desc)
    back_s: np.ndarray           # [T, R, 3]
    lay_p: np.ndarray            # [T, R, 3] best available-to-LAY prices (asc)
    lay_s: np.ndarray            # [T, R, 3]
    ltp: np.ndarray              # [T, R]
    tv: np.ndarray               # [T, R] cumulative traded volume
    fair: np.ndarray             # [T, R] fair price (microprice -> mid -> ltp)
    rank: np.ndarray             # [T, R] 1 = favourite, inactive = R+1
    active: np.ndarray           # [T, R] bool
    ts: np.ndarray               # [T] seconds
    runner_feats: np.ndarray     # [T, R, N_RUNNER_FEATURES] float32
    global_feats: np.ndarray     # [T, N_GLOBAL_FEATURES] float32
    is_winner: np.ndarray = field(default=None)   # [R] diagnostics only


def _col(df, name, default=np.nan):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
    return np.full(len(df), default, dtype=np.float64)


def _runner_count(df):
    idx = {int(m.group(1)) for c in df.columns if (m := _RUN_RE.match(c))}
    return (max(idx) + 1) if idx else 0


def _ffill(a):
    """Forward-fill NaNs along axis 0 (causal)."""
    a = a.copy()
    mask = np.isnan(a)
    idx = np.where(~mask, np.arange(a.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = a[idx, np.arange(a.shape[1])[None, :]]
    out[np.cumsum(~mask, axis=0) == 0] = np.nan
    return out


def _rolling(a, n, fn):
    """Trailing-window reduction over axis 0 with window n (causal, nan-aware).

    A window with no data for a runner (e.g. no trade in the last ~60 s) gives
    NaN; callers fall back to the running value. numpy's "All-NaN slice"
    warning for that case is expected and silenced.
    """
    x = np.concatenate([np.full((n - 1,) + a.shape[1:], np.nan), a])
    w = sliding_window_view(x, n, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(w, axis=-1)


def _lag_ret(x, lag):
    out = np.zeros_like(x)
    if x.shape[0] > lag:
        with np.errstate(all="ignore"):
            out[lag:] = np.log(x[lag:] / x[:-lag])
    return out


def find_decision_rows(df):
    """Rows the agent acts on: pre-race, market OPEN. Returns (rows, in_play_found)."""
    in_play = _col(df, "in_play", 0.0)
    status = df["market_status"].astype(str).to_numpy() if "market_status" in df.columns \
        else np.array(["OPEN"] * len(df))
    ip = np.where(in_play > 0.5)[0]
    end = int(ip[0]) if len(ip) else len(df)
    rows = np.array([i for i in range(end) if status[i] == "OPEN"], dtype=np.int64)
    return rows, bool(len(ip))


def load_race(path, max_runners=MAX_RUNNERS):
    df = pd.read_parquet(path)
    return build_race(df, path=str(path), max_runners=max_runners)


def build_race(df, path="<df>", max_runners=MAX_RUNNERS):
    rows, in_play_found = find_decision_rows(df)
    d = df.iloc[rows].reset_index(drop=True)
    T = len(d)
    R = min(_runner_count(df), max_runners)
    if T == 0 or R == 0:
        raise ValueError(f"{path}: no pre-race OPEN rows or no runners")

    def rc(i, name, default=np.nan):
        return _col(d, f"run[{i}].{name}", default)

    def stack(name, default=np.nan):
        return np.stack([rc(i, name, default) for i in range(R)], axis=1)

    back_p = np.stack([stack(f"back_price_{k}") for k in range(1, LEVELS + 1)], axis=2)
    back_s = np.stack([stack(f"back_size_{k}") for k in range(1, LEVELS + 1)], axis=2)
    lay_p = np.stack([stack(f"lay_price_{k}") for k in range(1, LEVELS + 1)], axis=2)
    lay_s = np.stack([stack(f"lay_size_{k}") for k in range(1, LEVELS + 1)], axis=2)
    # a size without a price (or vice versa) is no offer
    back_s = np.where(np.isnan(back_p), 0.0, np.nan_to_num(back_s))
    lay_s = np.where(np.isnan(lay_p), 0.0, np.nan_to_num(lay_s))

    ltp_raw = stack("last_traded_price")
    tv = np.nan_to_num(_ffill(stack("traded_vol_total")), nan=0.0)
    micro = stack("microprice")
    has_back = ~np.isnan(back_p[..., 0])
    has_lay = ~np.isnan(lay_p[..., 0])
    with np.errstate(all="ignore"):
        mid = np.where(has_back & has_lay, 0.5 * (back_p[..., 0] + lay_p[..., 0]), np.nan)
    ltp = _ffill(ltp_raw)
    fair = micro.copy()
    for alt in (mid, ltp, back_p[..., 0], lay_p[..., 0]):
        fair = np.where(np.isnan(fair), alt, fair)
    fair = np.clip(fair, 1.01, 1000.0)
    active = has_back | has_lay
    fair_f = np.where(active, fair, np.nan)
    fair_filled = np.nan_to_num(fair_f, nan=1000.0)
    ltp = np.where(np.isnan(ltp), fair_filled, ltp)

    # favouritism rank (1 = shortest price) among active runners
    order = np.argsort(np.where(active, fair_filled, np.inf), axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(T)[:, None], order] = np.arange(1, R + 1)[None, :]
    rank = np.where(active, rank, R + 1)

    # ---------------- engineered features (all causal) ----------------
    eps = 1e-9
    prob_implied = np.nan_to_num(stack("prob_implied"), nan=0.0)
    prob_from_fair = np.where(active, 1.0 / fair_filled, 0.0)
    prob_implied = np.where(prob_implied > 0, prob_implied, prob_from_fair)
    fair_prob = prob_from_fair / np.maximum(prob_from_fair.sum(1, keepdims=True), eps)

    sb, sl = back_s.sum(2), lay_s.sum(2)
    wom_3l = np.where(sb + sl > 0, sb / np.maximum(sb + sl, eps), 0.5)
    wom_l1 = np.where(back_s[..., 0] + lay_s[..., 0] > 0,
                      back_s[..., 0] / np.maximum(back_s[..., 0] + lay_s[..., 0], eps), 0.5)
    pw = np.nan_to_num(back_p) * back_s + np.nan_to_num(lay_p) * lay_s
    book_wap = np.where(sb + sl > 0, pw.sum(2) / np.maximum(sb + sl, eps), fair_filled)

    dvol = np.zeros_like(tv)
    dvol[1:] = np.maximum(tv[1:] - tv[:-1], 0.0)
    traded = dvol > 0
    ltp_tr = np.where(traded, ltp, np.nan)
    cum_v = np.cumsum(dvol, 0)
    cum_pv = np.cumsum(np.where(traded, dvol * ltp, 0.0), 0)
    vwap = np.where(cum_v > 0, cum_pv / np.maximum(cum_v, eps), fair_filled)
    k = ROLL_STEPS
    roll_v = cum_v - np.vstack([np.zeros((k, R)), cum_v[:-k]])[:T]
    roll_pv = cum_pv - np.vstack([np.zeros((k, R)), cum_pv[:-k]])[:T]
    vwap_roll = np.where(roll_v > 0, roll_pv / np.maximum(roll_v, eps), fair_filled)
    # max/min matched price: running + trailing window (ltp of the first row seeds it)
    seed = np.where(np.isnan(ltp_tr), np.nan, ltp_tr)
    seed[0] = np.where(np.isnan(seed[0]), ltp[0], seed[0])
    with np.errstate(all="ignore"):
        max_m = np.fmax.accumulate(seed, axis=0)
        min_m = np.fmin.accumulate(seed, axis=0)
    max_m = np.where(np.isnan(max_m), fair_filled, max_m)
    min_m = np.where(np.isnan(min_m), fair_filled, min_m)
    max_roll = _rolling(seed, k, np.nanmax)
    min_roll = _rolling(seed, k, np.nanmin)
    max_roll = np.where(np.isnan(max_roll), max_m, max_roll)
    min_roll = np.where(np.isnan(min_roll), min_m, min_roll)

    spread_ticks = np.zeros((T, R))
    for t in range(T):
        for r in range(R):
            if has_back[t, r] and has_lay[t, r]:
                spread_ticks[t, r] = price_to_tick(lay_p[t, r, 0]) - price_to_tick(back_p[t, r, 0])

    def lg(x):
        with np.errstate(all="ignore"):
            return np.log(np.clip(x, 1.0, 1000.0))

    def rel(x):
        with np.errstate(all="ignore"):
            return np.log(np.maximum(x, 1.0) / fair_filled)

    tv_total = tv.sum(1, keepdims=True)
    feats = {
        **{f"log_back_p{k + 1}": np.nan_to_num(lg(back_p[..., k])) for k in range(LEVELS)},
        **{f"log_lay_p{k + 1}": np.nan_to_num(lg(lay_p[..., k])) for k in range(LEVELS)},
        **{f"log1p_back_s{k + 1}": np.log1p(back_s[..., k]) for k in range(LEVELS)},
        **{f"log1p_lay_s{k + 1}": np.log1p(lay_s[..., k]) for k in range(LEVELS)},
        "has_back": has_back.astype(float), "has_lay": has_lay.astype(float),
        "log_ltp": lg(ltp), "log_micro": lg(fair_filled),
        "prob_implied": prob_implied, "fair_prob": fair_prob,
        "rel_spread": np.clip(np.nan_to_num(stack("rel_spread")), 0, 2),
        "spread_ticks": np.clip(spread_ticks, 0, 50) / 10.0,
        "ob_imbalance": np.nan_to_num(stack("ob_imbalance")),
        "wom_l1": wom_l1, "wom_3l": wom_3l,
        "book_wap_rel": rel(book_wap), "vwap_rel": rel(vwap), "vwap_roll_rel": rel(vwap_roll),
        "max_matched_rel": rel(max_m), "min_matched_rel": rel(min_m),
        "max_matched_roll_rel": rel(max_roll), "min_matched_roll_rel": rel(min_roll),
        "log1p_tv": np.log1p(tv), "log1p_tv_60s": np.log1p(np.nan_to_num(stack("traded_vol_60s"))),
        "log1p_dvol": np.log1p(dvol),
        "vol_share": np.where(tv_total > 0, tv / np.maximum(tv_total, eps), 0.0),
        "log1p_secs_since_trade": np.log1p(np.clip(np.nan_to_num(stack("secs_since_last_trade"), nan=999.0), 0, 1e4)),
        "ret_std_5s": 10 * np.nan_to_num(stack("ret_std_5s")),
        "ret_std_20s": 10 * np.nan_to_num(stack("ret_std_20s")),
        **{f"ret_{l}": 10 * _lag_ret(fair_filled, l) for l in RET_LAGS},
        "ltp_rel": rel(ltp),
        "rank_norm": rank / float(MAX_RUNNERS), "is_fav": (rank == 1).astype(float),
        "active": active.astype(float),
    }
    runner_feats = np.stack([feats[n] for n in RUNNER_FEATURES], axis=2)
    runner_feats = np.where(active[..., None], runner_feats, 0.0)
    runner_feats = np.nan_to_num(runner_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---------------- global features ----------------
    ts = _col(d, "ts_unix", np.nan) / 1000.0
    if np.isnan(ts).all():
        ts = np.arange(T, dtype=np.float64) * 1.33
    ts = _ffill(ts[:, None])[:, 0]
    dt = np.zeros(T)
    dt[1:] = np.clip(np.diff(ts), 0, 30)
    tm = np.nan_to_num(_ffill(_col(d, "total_matched_market")[:, None])[:, 0])
    dtm = np.zeros(T)
    dtm[1:] = np.log1p(np.maximum(np.diff(tm), 0))
    code = str(d["race_code"].iloc[0]) if "race_code" in d.columns else "?"
    commission = _col(d, "commission_rate", 0.05)[0]
    commission = 0.05 if np.isnan(commission) else float(commission)
    g = {
        "secs_to_off": np.clip(np.nan_to_num(_col(d, "secs_to_off")), -600, 3600) / 600.0,
        "log1p_total_matched": np.log1p(tm) / 10.0,
        "dlog_total_matched": dtm,
        "back_overround": np.nan_to_num(_col(d, "back_overround"), nan=100.0) / 100.0 - 1.0,
        "lay_overround": np.nan_to_num(_col(d, "lay_overround"), nan=100.0) / 100.0 - 1.0,
        "overround_gap": np.clip(np.nan_to_num(_col(d, "overround_gap")), -100, 100) / 100.0,
        "prob_entropy": np.nan_to_num(_col(d, "prob_entropy")),
        "fav_prob_gap": np.nan_to_num(_col(d, "fav_prob_gap")),
        "spread_mean_run": np.nan_to_num(_col(d, "spread_mean_run")),
        "spread_std_run": np.nan_to_num(_col(d, "spread_std_run")),
        "runner_count": np.full(T, R / float(MAX_RUNNERS)),
        "n_active": active.sum(1) / float(MAX_RUNNERS),
        "commission": np.full(T, commission),
        "is_flat": np.full(T, float(code == "Flat")),
        "is_harness": np.full(T, float(code == "Harness")),
        "is_other_code": np.full(T, float(code not in ("Flat", "Harness"))),
        "distance": np.full(T, np.nan_to_num(_col(d, "distance_m", 0.0)[0]) / 3000.0),
        "dt": dt / 5.0,
    }
    global_feats = np.stack([g[n] for n in GLOBAL_FEATURES], axis=1)
    global_feats = np.nan_to_num(global_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    is_winner = np.array([np.nan_to_num(_col(df, f"run[{i}].is_winner", 0.0)).max() for i in range(R)])
    market_id = str(df["file_market_id"].iloc[0]) if "file_market_id" in df.columns else ""
    race_date = str(df["race_date"].iloc[0]) if "race_date" in df.columns else ""

    return RaceData(
        path=path, market_id=market_id, race_date=race_date, commission=commission,
        n_runners=R, T=T, in_play_found=in_play_found,
        back_p=back_p, back_s=back_s, lay_p=lay_p, lay_s=lay_s,
        ltp=ltp, tv=tv, fair=fair_filled, rank=rank, active=active, ts=ts,
        runner_feats=runner_feats, global_feats=global_feats, is_winner=is_winner,
    )
