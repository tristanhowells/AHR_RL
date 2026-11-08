"""P22: trade every runner with perfect hindsight, describe where the hindsight profit is, then test whether any of
it transfers to comparable races that were not looked at.

A perfect-hindsight trade always shows a profit on a runner whose price moves at all - it picks the best entry
and exit after the fact - so the hindsight profit mostly measures how much a price moves, not whether the move
was predictable. The study therefore keeps two questions apart:

  DESCRIBE (what hindsight sees)
    O  full-race oracle, per runner: the best single round trip in the last 10 minutes (back then lay, or lay then
       back, crossing the spread both times, entry one row after the decision), its side, timing, hold, ticks and
       green % of stake; and whether the best prices had $10 on them
    A  hindsight profit by race category (race type, state, metro, distance, class, race number, field size,
       market volume, day, time of day) and by runner (favourite rank, price band, time to the start)
    B  the conditions just before the oracle's entries vs ordinary moments: feature lift (median at oracle
       entries vs all samples), separately for oracle backs and oracle lays

  TRANSFER (what could have been traded, scored on later races only)
    Samples: every 10s from 10 min to 10s before the advertised start, every runner with a two-sided book.
    Executable labels: cross now, cross back out H = 60 / 120s later (no hindsight), net of commission.
    Days are split chronologically: TRAIN (70%) / VAL (15%) / TEST (15%); TEST is scored once.
    C  feature IC with the executable returns on TRAIN vs TEST: does a signal keep its sign out of sample?
    D  category stability: each category's mean executable return (best side) on TRAIN vs TEST
    E  model: gradient-boosted trees on every feature (race + runner + market state) predict the four executable
       returns (back / lay x 60 / 120s); trade the best prediction above a threshold picked on VAL; TEST once,
       race-clustered t; plus the oracle-imitation version (predict the side hindsight would have taken)
    F  comparable races: for each TEST race, its k most similar TRAIN races (race type, distance, class, metro,
       field size, volume so far, venue); the cell (side x horizon x favourite rank x time bucket x 30s-move
       bucket) that paid best on those neighbours (>= --min-cell trades, mean > 0) is traded in the TEST race
    "transferable" = TEST mean > 0 with t > 2 over >= 30 races, for E or F.

    python -m ahr_rl.hindsight_study --tapes "data/tapes/*.npz" --catalogues "<drive>/catalogues" --out runs/hindsight
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .env import list_tapes, split_by_date
from .ladder import PRICES
from .race_study import add_categories, race_meta
from .tape import Tape

EVERY_S, LAST_S, MIN_LEFT_S = 10.0, 600.0, 10.0
HORIZONS = (60, 120)
STAKE = 10.0
RACE_CATS = ["race_type", "state", "metro", "distance_b", "class_group", "race_no_b", "field_b", "volume_b", "dow",
             "time_of_day"]
RUNNER_FEATS = ["rank", "log_price", "share", "spread_ticks", "best_usd_log", "wom", "mom_10", "mom_30", "mom_60",
                "vol_60", "traded_30_log", "spn_gap", "t_to_start", "overround", "matched_log"]
RACE_FEATS = ["field", "distance_m", "race_no", "is_metro", "race_type_c", "class_c", "state_c", "venue_c", "hour",
              "dow_c"]


def rt(v, race) -> dict:
    """Race-clustered: mean of per-race means, its t, and the number of races."""
    per = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna().groupby("race")["v"].mean()
    if len(per) < 2 or per.std(ddof=1) == 0:
        return dict(mean=float(per.mean()) if len(per) else np.nan, t=np.nan, races=len(per))
    return dict(mean=float(per.mean()), t=float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))), races=len(per))


def _prices(t: Tape):
    """Best back / lay prices per (row, runner), NaN where there is no price, forward-filled over <= 5s gaps."""
    ok = ~t.suspended[:, None] & t.active
    bt, lt = t.back_tick[:, :, 0].astype(int), t.lay_tick[:, :, 0].astype(int)
    bb = np.where(ok & (bt >= 0), PRICES[np.maximum(bt, 0)], np.nan)
    bl = np.where(ok & (lt >= 0), PRICES[np.maximum(lt, 0)], np.nan)
    bb = pd.DataFrame(bb).ffill(limit=10).to_numpy(copy=True)
    bl = pd.DataFrame(bl).ffill(limit=10).to_numpy(copy=True)
    both = np.isfinite(bb) & np.isfinite(bl) & (bl > bb)
    bb[~both], bl[~both] = np.nan, np.nan
    return bb, bl


def _net(g, c):
    return np.where(g > 0, g * (1 - c), g)


def full_oracle(t: Tape, bb, bl, rows: np.ndarray, end: int, comm: float) -> list[dict]:
    """Per runner: the best single round trip with entry at a decision row (filled one row later) and exit at any
    later row up to the off. back: P_back(e) / P_lay(x) - 1; lay: 1 - P_lay(e) / P_back(x)."""
    out = []
    span = np.arange(rows[0], end + 1)
    for r in range(t.n_runners):
        b, l = bb[span, r], bl[span, r]
        if np.isfinite(b).sum() < 20:
            continue
        # suffix extremes of the exit prices (strictly after the entry fill)
        lmin = np.minimum.accumulate(np.where(np.isfinite(l), l, np.inf)[::-1])[::-1]
        bmax = np.maximum.accumulate(np.where(np.isfinite(b), b, -np.inf)[::-1])[::-1]
        best = None
        for s in rows:
            i = s + 1 - span[0]  # entry fills one row later
            if i + 2 >= len(span) or not (np.isfinite(b[i]) and np.isfinite(l[i])):
                continue
            gb = b[i] / lmin[i + 2] - 1 if np.isfinite(lmin[i + 2]) else -np.inf
            gl = 1 - l[i] / bmax[i + 2] if np.isfinite(bmax[i + 2]) and bmax[i + 2] > 0 else -np.inf
            for side, g in (("back", gb), ("lay", gl)):
                if best is None or g > best[0]:
                    best = (g, side, s, i)
        if best is None:
            continue
        g, side, s, i = best
        # exit row: the first row after the entry where the extreme is reached
        seg = (l if side == "back" else b)[i + 2:]
        target = lmin[i + 2] if side == "back" else bmax[i + 2]
        x = int(np.nanargmin(np.abs(seg - target))) + i + 2 + span[0]
        p_in = b[i] if side == "back" else l[i]
        out.append(dict(runner=r, side=side if g > 0 else "none", green_pct=float(max(_net(g, comm), 0.0) * 100),
                        t_entry=float(t.t_rel[s]), t_exit=float(t.t_rel[x]), hold_s=float(t.t_rel[x] - t.t_rel[s]),
                        p_in=float(p_in), p_out=float(target),
                        size_in=float((t.back_size if side == "back" else t.lay_size)[s + 1, r, 0]),
                        size_out=float((t.lay_size if side == "back" else t.back_size)[x, r, 0])))
    return out


def sample_race(path: str, cat_idx: dict) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    t = Tape.load(path)
    if not t.went_in_play or t.n_steps < 200:
        return None
    meta = race_meta(path, cat_idx)
    race = meta["race"]
    comm = t.base_rate / 100.0
    dt = t.dt
    k = lambda sec: int(round(sec / dt))
    live = ~t.suspended
    end = int(np.where(live)[0].max())  # last live row before the off
    rows = np.where(live & (t.t_rel >= -LAST_S) & (t.t_rel <= -MIN_LEFT_S))[0][::max(1, k(EVERY_S))]
    rows = rows[rows >= k(60)]
    if len(rows) < 5:
        return None
    bb, bl = _prices(t)
    mid = np.sqrt(bb * bl)
    p = 1.0 / mid
    book = np.nansum(p, 1)
    share = p / book[:, None]
    # per-runner traded $ per row (single-counted), for the traded-in-last-30s feature
    vol = np.zeros((t.n_steps, t.n_runners))
    m = t.trade_step < t.n_steps
    np.add.at(vol, (t.trade_step[m], t.trade_runner[m].astype(int)), t.trade_vol[m])
    cvol = np.cumsum(vol, 0)
    lmid = np.log(mid)
    dl = np.diff(lmid, axis=0, prepend=np.nan)
    bs3, ls3 = t.back_size[:, :, :3].sum(-1), t.lay_size[:, :, :3].sum(-1)
    recs = []
    for s in rows:
        okr = np.where(np.isfinite(mid[s]))[0]
        if len(okr) < 2:
            continue
        rank = np.empty(t.n_runners, int)
        rank[okr] = (-p[s, okr]).argsort().argsort() + 1
        e = s + 1  # entry fills one row later
        if e > end:
            continue
        for r in okr:
            if not (np.isfinite(bb[e, r]) and np.isfinite(bl[e, r])):
                continue
            rec = dict(race=race, runner=int(r), t=float(t.t_rel[s]), rank=int(rank[r]),
                       log_price=float(np.log(mid[s, r])), share=float(share[s, r]),
                       spread_ticks=int(t.lay_tick[s, r, 0] - t.back_tick[s, r, 0]),
                       best_usd_log=float(np.log1p(t.back_size[s, r, 0] + t.lay_size[s, r, 0])),
                       wom=float(bs3[s, r] / (bs3[s, r] + ls3[s, r])) if bs3[s, r] + ls3[s, r] > 0 else np.nan,
                       vol_60=float(np.nanstd(dl[max(0, s - k(60)):s + 1, r])),
                       traded_30_log=float(np.log1p(cvol[s, r] - cvol[max(0, s - k(30)), r])),
                       spn_gap=float(np.log(t.spn[s, r] / mid[s, r])) if t.spn[s, r] > 1 else np.nan,
                       t_to_start=float(-t.t_rel[s]), overround=float(book[s]),
                       matched_log=float(np.log1p(t.total_matched[s])))
            for W in (10, 30, 60):
                s0 = s - k(W)
                rec[f"mom_{W}"] = float((p[s, r] - p[s0, r]) * 100) if s0 >= 0 and np.isfinite(p[s0, r]) else np.nan
            for H in HORIZONS:
                x = min(s + k(H), end)
                ex_l, ex_b = bl[x, r], bb[x, r]
                rec[f"exec_back_{H}"] = float(_net(bb[e, r] / ex_l - 1, comm) * 100) if np.isfinite(ex_l) else np.nan
                rec[f"exec_lay_{H}"] = float(_net(1 - bl[e, r] / ex_b, comm) * 100) if np.isfinite(ex_b) else np.nan
            # hindsight from here: the best exit within 120s (what a perfect trader entering now could make)
            w = slice(e + 2, min(s + k(120), end) + 1)
            lm, bm = np.nanmin(bl[w, r]) if np.isfinite(bl[w, r]).any() else np.nan, \
                np.nanmax(bb[w, r]) if np.isfinite(bb[w, r]).any() else np.nan
            rec["oracle_back"] = float(_net(bb[e, r] / lm - 1, comm) * 100) if np.isfinite(lm) else np.nan
            rec["oracle_lay"] = float(_net(1 - bl[e, r] / bm, comm) * 100) if np.isfinite(bm) else np.nan
            recs.append(rec)
    if not recs:
        return None
    d = pd.DataFrame(recs)
    for c_ in ("venue", "race_type", "state", "class_group", "distance_m", "race_no", "field", "local_time",
               "market_name"):
        d[c_] = meta.get(c_)
    d["final_matched"] = meta["matched_$"]  # description only (known after the off), never a model input
    o = pd.DataFrame(full_oracle(t, bb, bl, rows, end, comm))
    if len(o):
        o["race"] = race
        p0 = np.nan_to_num(p[rows[0]], nan=-1.0)  # favourite rank when the window opens (no price -> last)
        rk0 = (-p0).argsort().argsort() + 1
        o["rank"] = [int(rk0[r]) if p0[r] > 0 else 99 for r in o["runner"]]
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    return d, o


def _one(args):
    path, cat_idx = args
    try:
        return sample_race(path, cat_idx)
    except Exception as e:
        print(f"skip {os.path.basename(path)}: {e}", flush=True)
        return None


# ============================================================================ analysis
def add_race_categories(d: pd.DataFrame) -> pd.DataFrame:
    races = d.drop_duplicates("race")[["race", "venue", "race_type", "state", "class_group", "distance_m", "race_no",
                                        "field", "local_time", "final_matched"]].copy()
    races = add_categories(races)
    races["volume_b"] = pd.qcut(races["final_matched"].rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    keep = ["race", "metro", "distance_b", "dow", "weekend", "time_of_day", "race_no_b", "field_b", "volume_b"]
    d = d.merge(races[keep], on="race", how="left")
    lt = pd.to_datetime(d["local_time"])
    d["hour"] = (lt.dt.hour + lt.dt.minute / 60).astype("float32")
    d["is_metro"] = (d["metro"] == "metro").astype("float32")
    for src, dst in (("race_type", "race_type_c"), ("class_group", "class_c"), ("state", "state_c"),
                     ("venue", "venue_c"), ("dow", "dow_c")):
        c_ = d[src].astype("category").cat.codes.astype("float32")
        d[dst] = c_.where(c_ >= 0)  # unknown -> missing (the trees handle NaN)
    d["rank_g"] = pd.cut(d["rank"], [0, 1, 3, 6, 99], labels=["fav", "2-3", "4-6", "7+"])
    d["price_b"] = pd.cut(np.exp(d["log_price"]), [1, 3, 6, 11, 21, 1001], labels=["<3", "3-6", "6-11", "11-21", "21+"])
    d["time_b"] = pd.cut(d["t_to_start"], [0, 60, 180, 360, 601], labels=["<1m", "1-3m", "3-6m", "6-10m"])
    d["mom_b"] = pd.cut(d["mom_30"], [-1e9, -0.3, -0.05, 0.05, 0.3, 1e9], labels=["drift++", "drift", "flat", "firm", "firm++"])
    d["oracle_best"] = d[["oracle_back", "oracle_lay"]].max(axis=1)
    d["oracle_side"] = np.where(d["oracle_back"] >= d["oracle_lay"], "back", "lay")
    return d


def by_cat(d: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for lv, g in d.groupby(col, observed=True):
        rows.append({"category": col, "level": str(lv), "races": g["race"].nunique(), "samples": len(g),
                     "hindsight_best_pct": g["oracle_best"].mean(),
                     "hindsight_back_pct": g["oracle_back"].mean(), "hindsight_lay_pct": g["oracle_lay"].mean(),
                     "share_beats_0": (g["oracle_best"] > 0).mean() * 100,
                     "exec_back_60": g["exec_back_60"].mean(), "exec_lay_60": g["exec_lay_60"].mean()})
    return pd.DataFrame(rows)


def lift(d: pd.DataFrame) -> pd.DataFrame:
    """Median of each feature at hindsight-profitable entries (best side clears 2% within 120s) vs all samples."""
    rows = []
    good_b = d["oracle_back"] > 2
    good_l = d["oracle_lay"] > 2
    for f in RUNNER_FEATS:
        allv = d[f].median()
        rows.append(dict(feature=f, all_samples=allv, at_oracle_backs=d.loc[good_b, f].median(),
                         at_oracle_lays=d.loc[good_l, f].median(),
                         share_oracle_back_pct=good_b.mean() * 100, share_oracle_lay_pct=good_l.mean() * 100))
    return pd.DataFrame(rows)


def race_ic(d: pd.DataFrame, f: str, y: str) -> float:
    g = d[["race", f, y]].dropna()
    if len(g) < 50:
        return np.nan
    ics = g.groupby("race").apply(lambda h: h[f].rank().corr(h[y].rank()) if len(h) >= 8 else np.nan,
                                  include_groups=False).dropna()
    return float(ics.mean()) if len(ics) else np.nan


def ic_table(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in RUNNER_FEATS:
        for y in ("exec_back_60", "exec_lay_60", "exec_back_120", "exec_lay_120"):
            a, b = race_ic(tr, f, y), race_ic(te, f, y)
            rows.append(dict(feature=f, label=y, ic_train=a, ic_test=b,
                             same_sign=bool(np.sign(a) == np.sign(b)) if np.isfinite(a) and np.isfinite(b) else None))
    return pd.DataFrame(rows)


def category_transfer(tr: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = ["exec_back_60", "exec_lay_60", "exec_back_120", "exec_lay_120"]
    for col in RACE_CATS + ["rank_g", "price_b", "time_b", "mom_b"]:
        for lv, g in tr.groupby(col, observed=True):
            if g["race"].nunique() < 20:
                continue
            means = g[labels].mean()
            best = means.idxmax()
            h = te[te[col] == lv]
            r = rt(h[best], h["race"]) if len(h) else dict(mean=np.nan, t=np.nan, races=0)
            rows.append(dict(category=col, level=str(lv), best_label_train=best, train_mean=float(means[best]),
                             train_races=g["race"].nunique(), test_mean=r["mean"], test_t=r["t"], test_races=r["races"]))
    return pd.DataFrame(rows).sort_values("train_mean", ascending=False)


def pick_trades(df: pd.DataFrame, pred: np.ndarray, thr: float, labels) -> pd.DataFrame:
    """Per sample: the best predicted label; trade it when its prediction clears thr (at most one trade per runner
    per decision)."""
    j = pred.argmax(1)
    best = pred[np.arange(len(pred)), j]
    sel = best > thr
    out = df.loc[sel, ["race", "runner", "t"]].copy()
    vals = df[labels].to_numpy()[np.where(sel)[0], j[sel]]
    out["label"] = np.array(labels)[j[sel]]
    out["ret"] = vals
    return out.dropna(subset=["ret"])


def model_transfer(tr, va, te, seed=0) -> tuple[pd.DataFrame, dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    feats = [f for f in RUNNER_FEATS + RACE_FEATS if tr[f].notna().any() and tr[f].nunique() > 1]  # e.g. no catalogue
    cats = [feats.index(c) for c in ("race_type_c", "class_c", "state_c", "venue_c", "dow_c") if c in feats]
    cats = cats or None
    labels = ["exec_back_60", "exec_lay_60", "exec_back_120", "exec_lay_120"]
    Xtr, Xva, Xte = (x[feats].to_numpy(np.float32) for x in (tr, va, te))
    preds = {}
    for y in labels:
        m = tr[y].notna().to_numpy()
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                            l2_regularization=1.0, categorical_features=cats, random_state=seed)
        reg.fit(Xtr[m], np.clip(tr[y].to_numpy()[m], -50, 50))
        preds[y] = (reg.predict(Xva), reg.predict(Xte))
    Pva = np.stack([preds[y][0] for y in labels], 1)
    Pte = np.stack([preds[y][1] for y in labels], 1)
    rows, best = [], None
    min_r = min(20, max(3, va["race"].nunique() // 3))  # thresholds must trade in enough VAL races
    for thr in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        v = pick_trades(va, Pva, thr, labels)
        r = rt(v["ret"], v["race"]) if len(v) else dict(mean=np.nan, t=np.nan, races=0)
        rows.append(dict(model="executable-return trees", threshold=thr, val_trades=len(v), val_mean=r["mean"],
                         val_t=r["t"], val_races=r["races"]))
        if r["races"] >= min_r and np.isfinite(r["mean"]) and (best is None or r["mean"] > best[1]):
            best = (thr, r["mean"])
    res = {}
    if best is not None:
        tt = pick_trades(te, Pte, best[0], labels)
        r = rt(tt["ret"], tt["race"]) if len(tt) else dict(mean=np.nan, t=np.nan, races=0)
        res["trees"] = dict(threshold=best[0], test_trades=len(tt), test_mean_pct=r["mean"], test_t=r["t"],
                            test_races=r["races"], labels=tt["label"].value_counts().to_dict() if len(tt) else {})
    # oracle imitation: predict the side hindsight takes when it clears 2% within 120s, trade it at 120s
    ytr = np.where(tr["oracle_best"] <= 2, 0, np.where(tr["oracle_side"] == "back", 1, 2))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                         categorical_features=cats, random_state=seed)
    clf.fit(Xtr, ytr)
    pv, pt = clf.predict_proba(Xva), clf.predict_proba(Xte)
    bestc = None
    for q in (0.4, 0.5, 0.6, 0.7, 0.8):
        for name, P, df in (("val", pv, va),):
            side = np.where(P[:, 1] >= P[:, 2], 1, 2)
            conf = P[np.arange(len(P)), side]
            sel = conf >= q
            ret = np.where(side == 1, df["exec_back_120"], df["exec_lay_120"])[sel]
            r = rt(ret, df["race"].to_numpy()[sel]) if sel.any() else dict(mean=np.nan, t=np.nan, races=0)
            rows.append(dict(model="oracle imitation", threshold=q, val_trades=int(sel.sum()), val_mean=r["mean"],
                             val_t=r["t"], val_races=r["races"]))
            if r["races"] >= min_r and np.isfinite(r["mean"]) and (bestc is None or r["mean"] > bestc[1]):
                bestc = (q, r["mean"])
    if bestc is not None:
        side = np.where(pt[:, 1] >= pt[:, 2], 1, 2)
        sel = pt[np.arange(len(pt)), side] >= bestc[0]
        ret = np.where(side == 1, te["exec_back_120"], te["exec_lay_120"])[sel]
        r = rt(ret, te["race"].to_numpy()[sel]) if sel.any() else dict(mean=np.nan, t=np.nan, races=0)
        res["imitation"] = dict(threshold=bestc[0], test_trades=int(sel.sum()), test_mean_pct=r["mean"],
                                test_t=r["t"], test_races=r["races"])
    return pd.DataFrame(rows), res


def knn_transfer(tr: pd.DataFrame, te: pd.DataFrame, k: int = 25, min_cell: int = 30) -> tuple[pd.DataFrame, dict]:
    """For each TEST race: its k most similar TRAIN races; the cell that paid best on them is traded in it."""
    def profile(d):
        g = d.sort_values("t").groupby("race").first()
        x = pd.DataFrame(index=g.index)
        x["distance"] = g["distance_m"].fillna(g["distance_m"].median()) / 400.0
        x["field"] = g["field"].astype(float) / 2.0
        x["matched"] = g["matched_log"].astype(float)
        x["metro"] = g["is_metro"].astype(float) * 2
        for c in ("race_type", "class_group"):
            for lv in tr[c].dropna().unique():
                x[f"{c}={lv}"] = (g[c] == lv).astype(float) * 2
        return x, g["venue"]

    ptr, vtr = profile(tr)
    pte, vte = profile(te)
    pte = pte.reindex(columns=ptr.columns, fill_value=0.0)
    cells = ["rank_g", "time_b", "mom_b"]
    labels = ["exec_back_60", "exec_lay_60", "exec_back_120", "exec_lay_120"]
    long = tr.melt(id_vars=["race"] + cells, value_vars=labels, var_name="label", value_name="ret").dropna()
    A = ptr.to_numpy()
    trades = []
    for race, row in pte.iterrows():
        dist = np.sqrt(((A - row.to_numpy()) ** 2).sum(1)) + np.where(vtr.to_numpy() == vte[race], 0.0, 0.5)
        nb = ptr.index[np.argsort(dist)[:k]]
        g = long[long["race"].isin(nb)]
        st = g.groupby(cells + ["label"], observed=True)["ret"].agg(["mean", "count"])
        st = st[st["count"] >= min_cell]
        if not len(st) or st["mean"].max() <= 0:
            continue
        (rk, tb, mb, lab) = st["mean"].idxmax()
        h = te[(te["race"] == race) & (te["rank_g"] == rk) & (te["time_b"] == tb) & (te["mom_b"] == mb)]
        for v in h[lab].dropna():
            trades.append(dict(race=race, cell=f"{rk}|{tb}|{mb}|{lab}", neighbour_mean=float(st["mean"].max()), ret=v))
    t_ = pd.DataFrame(trades)
    if not len(t_):
        return t_, dict(test_trades=0, test_races=0, test_mean_pct=np.nan, test_t=np.nan)
    r = rt(t_["ret"], t_["race"])
    return t_, dict(test_trades=len(t_), test_races=r["races"], test_mean_pct=r["mean"], test_t=r["t"],
                    races_with_a_trade_pct=r["races"] / max(te["race"].nunique(), 1) * 100,
                    top_cells=t_["cell"].value_counts().head(10).to_dict())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--catalogues", default=None, help="market catalogue folder (race type, class, distance, venue)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=25, help="comparable races per TEST race")
    ap.add_argument("--min-cell", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    t0 = time.time()
    from .catalogue import index_catalogues

    cat_idx = index_catalogues(a.catalogues) if a.catalogues and os.path.isdir(a.catalogues) else {}
    paths = list_tapes(a.tapes)
    cache = os.path.join(a.out, "samples.parquet")
    if os.path.exists(cache) and os.path.exists(os.path.join(a.out, "oracle_runners.csv")):
        d, o = pd.read_parquet(cache), pd.read_csv(os.path.join(a.out, "oracle_runners.csv"))
        print(f"reusing {cache}: {len(d):,} samples", flush=True)
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        parts, oparts = [], []
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for i, res in enumerate(pool.map(_one, [(p, cat_idx) for p in paths], chunksize=4)):
                if res is not None:
                    parts.append(res[0])
                    oparts.append(res[1])
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(paths)} races, {time.time() - t0:.0f}s", flush=True)
        if not parts:
            raise SystemExit("no races")
        d, o = pd.concat(parts, ignore_index=True), pd.concat(oparts, ignore_index=True)
        d["local_time"] = d["local_time"].astype(str)
        d.to_parquet(cache, index=False)
        o.to_csv(os.path.join(a.out, "oracle_runners.csv"), index=False)
    d = add_race_categories(d)
    n_races = d["race"].nunique()
    print(f"=== P22: perfect hindsight on {n_races} races, {len(d):,} runner samples "
          f"({(d['race_type'] != 'Unknown').mean() * 100:.0f}% with catalogue metadata) ===", flush=True)

    print("\n=== O. Full-race oracle per runner: the best single round trip in the last 10 min (net of commission) ===")
    o["rank_g"] = pd.cut(o["rank"], [0, 1, 3, 6, 99], labels=["fav", "2-3", "4-6", "7+"])
    o["liquid"] = (o["size_in"] >= STAKE) & (o["size_out"] >= STAKE)
    os_ = o.groupby("rank_g", observed=True).agg(runners=("green_pct", "size"), mean_green_pct=("green_pct", "mean"),
                                                 median_green_pct=("green_pct", "median"),
                                                 back_share_pct=("side", lambda s: (s == "back").mean() * 100),
                                                 median_hold_s=("hold_s", "median"),
                                                 median_entry_s=("t_entry", "median"),
                                                 liquid_both_ends_pct=("liquid", lambda s: s.mean() * 100))
    print(os_.round(2).to_string())
    print(f"all runners: mean {o['green_pct'].mean():.2f}% of stake, median {o['green_pct'].median():.2f}%; "
          f"per race (sum over runners, $10 each): ${(o.groupby('race')['green_pct'].sum() / 10).mean():.2f}")

    print("\n=== A. Hindsight profit (best exit within 120s of each sample, % of stake) and the executable 60s round "
          "trip, by category ===")
    cat_rows = pd.concat([by_cat(d, c) for c in RACE_CATS + ["rank_g", "price_b", "time_b", "mom_b"]], ignore_index=True)
    cat_rows.to_csv(os.path.join(a.out, "by_category.csv"), index=False)
    print(cat_rows.round(3).to_string(index=False))

    print("\n=== B. Conditions at hindsight-profitable entries (median), vs all samples ===")
    lf = lift(d)
    lf.to_csv(os.path.join(a.out, "lift.csv"), index=False)
    print(lf.round(4).to_string(index=False))

    tr_r, va_r, te_r = split_by_date(sorted(d["race"].unique()))
    tr, va, te = d[d["race"].isin(tr_r)], d[d["race"].isin(va_r)], d[d["race"].isin(te_r)]
    print(f"\nsplit by day: TRAIN {len(tr_r)} / VAL {len(va_r)} / TEST {len(te_r)} races "
          f"({tr_r[0][:8]}..{tr_r[-1][:8]} | ..{va_r[-1][:8]} | ..{te_r[-1][:8]})")

    print("\n=== C. Does a signal keep its sign? within-race rank IC with the executable return, TRAIN vs TEST ===")
    ic = ic_table(tr, te)
    ic.to_csv(os.path.join(a.out, "ic_transfer.csv"), index=False)
    print(ic.round(4).to_string(index=False))

    print("\n=== D. Do categories that paid on TRAIN pay on TEST? (each level's best executable trade on TRAIN) ===")
    ct = category_transfer(tr, te)
    ct.to_csv(os.path.join(a.out, "category_transfer.csv"), index=False)
    print(ct.head(40).round(3).to_string(index=False))
    pos = ct[ct["train_mean"] > 0]
    print(f"levels positive on TRAIN: {len(pos)}; of those positive on TEST: {(pos['test_mean'] > 0).sum()}")

    print("\n=== E. Models on every feature: threshold picked on VAL, TEST once ===")
    mt, mres = model_transfer(tr, va, te)
    mt.to_csv(os.path.join(a.out, "model_val.csv"), index=False)
    print(mt.round(3).to_string(index=False))
    print(json.dumps(mres, indent=1, default=float))

    print(f"\n=== F. Comparable races: each TEST race traded with the cell that paid best on its {a.k} nearest TRAIN "
          f"races ===")
    kt, kres = knn_transfer(tr, te, a.k, a.min_cell)
    kt.to_csv(os.path.join(a.out, "knn_trades.csv"), index=False)
    print(json.dumps(kres, indent=1, default=float))

    ok = lambda r: bool(r and r.get("test_races", 0) >= 30 and (r.get("test_mean_pct") or -1) > 0 and (r.get("test_t") or 0) > 2)
    verdict = dict(races=int(n_races), samples=int(len(d)),
                   oracle_mean_green_pct=float(o["green_pct"].mean()),
                   signals_same_sign_pct=float(ic["same_sign"].dropna().mean() * 100) if ic["same_sign"].notna().any() else None,
                   categories_positive_train=int(len(pos)), categories_positive_train_and_test=int((pos["test_mean"] > 0).sum()),
                   model=mres, comparable_races=kres,
                   transferable=bool(ok(mres.get("trees")) or ok(mres.get("imitation")) or ok(kres)))
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    print("TRANSFERABLE: something learned on earlier races pays on later comparable races -> pre-register it and "
          "forward-test" if verdict["transferable"] else
          "NOT TRANSFERABLE: hindsight profits are everywhere, but nothing that describes them predicts a profitable "
          "trade on later races")
    print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
