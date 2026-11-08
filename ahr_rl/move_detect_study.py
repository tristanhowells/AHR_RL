"""P23: catch a move once it has started. Detect it early, act, and is there enough of it left to beat the costs?

The jump study (cell 2j) and P22 showed that a SUDDEN move is over by the time it is visible: after a 3-5 tick
jump in 5s the price comes back slightly more often than it carries on, and the last 10-60s move has no
predictive power (IC ~0). This study looks at what those did not cover:

  1. GRADUAL moves - a price that steps the same way several times in a row (persistence) rather than jumping.
     Every 5s, for every tradeable runner (both sides quoted, spread <= 3 ticks, price <= 50), over a look-back
     W = 30 / 60 / 90s:
        net_W      ticks moved (mid of best back / lay in tick space; + = shortened, i.e. steaming)
        persist_W  share of the tick changes in the window that went the same way as net_W
        jump_W     the largest 5s move inside the window (gradual = no single 5s step of 2+ ticks)
        traded_W   $ traded on the runner in the window
  2. KNOCK-ON moves - when one runner (the leader) steams, the rest of the book has to lengthen to make room.
     A runner that has not lengthened yet is a move that has started in the market but not on it:
        lead_dp    the biggest 30s rise in implied probability among the OTHER runners (pp)
        lag        this runner's 30s change in implied probability minus what rebalancing implies
                   (-lead_dp x share / (1 - leader's share)); > 0 = it has not drifted enough yet
  3. CONFIRMED moves - a gradual move plus an order-flow condition that agrees (or disagrees) with it:
        wom_with / wom_against   weight of money (best 3 levels) moved the same way / against over 30s
        wap_room / wap_past      the price is still on the far side of its session WAP (room to go) / already past
        vol_accel                $ traded in the last 30s is >= 2x the pace of the 60s before
        spn_with                 the projected SP is already beyond the price in the move's direction
     plus a gradient-boosted model on every feature (moves, flow, WOM, WAP, projected SP, knock-on, rank, price,
     spread, time) predicting the executable returns: threshold picked on VAL, TEST once.
  4. HOW MUCH IS LEFT at detection: the best further move the same way within 120s (MFE, ticks) and the worst
     move against (MAE), against the round-trip cost in ticks.

Executable trades: decided at the sample row, filled one row later (latency), crossing the spread; exit by
crossing H = 30 / 60 / 120s later, or a trailing exit (out once the mid gives back 2 ticks from its best since
entry, at most 120s). Net of commission, % of stake. "follow" = back a steamer / lay a drifter; "fade" = the
opposite. Knock-on: "follow" = lay a lagging runner (back one that has lengthened too much).

Rule search on TRAIN days only (look-back x move size x persistence x gradual x runners x exit x follow/fade, and
leader size x lag x exit for knock-on), each rule's race-clustered mean and t; the best TRAIN rule of each family is
pre-registered and scored once on VAL and TEST (chronological days). "edge" = TEST mean > 0, t > 2, >= 30 races,
VAL > 0. Section 0 runs the same pipeline on synthetic races with a planted gradual steamer first: the gradual
follow rules must find it, or a null result means little.

    python -m ahr_rl.move_detect_study --tapes "data/tapes/*.npz" --out runs/move_detect
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .env import list_tapes, split_by_date
from .hindsight_study import flow_arrays, flow_features
from .ladder import PRICES
from .tape import Tape

EVERY_S, LAST_S, MIN_LEFT_S = 5.0, 600.0, 10.0
WINDOWS = (30, 60, 90)
HORIZONS = (30, 60, 120)
MAX_SPREAD, MAX_PRICE = 3, 50.0
TRAIL_TICKS, TRAIL_MAX_S = 2, 120.0
MFE_S = 120.0


def rt(v, race) -> dict:
    """Race-clustered: mean of per-race means, its t, and the number of races."""
    per = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna().groupby("race")["v"].mean()
    if len(per) < 2 or per.std(ddof=1) == 0:
        return dict(mean=float(per.mean()) if len(per) else np.nan, t=np.nan, races=len(per))
    return dict(mean=float(per.mean()), t=float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))), races=len(per))


def _net(g, c):
    return np.where(g > 0, g * (1 - c), g)


def _book(t: Tape):
    """Best back / lay ticks and prices per (row, runner), NaN without a two-sided book, forward-filled <= 5s."""
    ok = ~t.suspended[:, None] & t.active
    bt = np.where(ok & (t.back_tick[:, :, 0] >= 0), t.back_tick[:, :, 0], np.nan).astype(float)
    lt = np.where(ok & (t.lay_tick[:, :, 0] >= 0), t.lay_tick[:, :, 0], np.nan).astype(float)
    bt = pd.DataFrame(bt).ffill(limit=10).to_numpy(copy=True)
    lt = pd.DataFrame(lt).ffill(limit=10).to_numpy(copy=True)
    bad = ~(np.isfinite(bt) & np.isfinite(lt) & (lt > bt))
    bt[bad], lt[bad] = np.nan, np.nan
    bb = np.where(np.isfinite(bt), PRICES[np.nan_to_num(bt, nan=0).astype(int)], np.nan)
    bl = np.where(np.isfinite(lt), PRICES[np.nan_to_num(lt, nan=0).astype(int)], np.nan)
    return bt, lt, bb, bl


def _trail(mt: np.ndarray, e: int, last: int, side: str, k: int) -> int:
    """Exit row of a trailing stop: the first row after e where the mid has given back k ticks from its best."""
    seg = mt[e:last + 1]
    if not len(seg) or not np.isfinite(seg).any():
        return last
    seg = np.where(np.isfinite(seg), seg, np.nan)
    s2 = pd.Series(seg).ffill().to_numpy()
    if side == "back":  # wants the tick index to fall (shorten)
        best = np.minimum.accumulate(np.nan_to_num(s2, nan=np.inf))
        hit = s2 >= best + k
    else:
        best = np.maximum.accumulate(np.nan_to_num(s2, nan=-np.inf))
        hit = s2 <= best - k
    hit[0] = False
    i = int(np.argmax(hit)) if hit.any() else len(seg) - 1
    return e + i


def sample_race(path: str) -> pd.DataFrame | None:
    t = Tape.load(path)
    if not t.went_in_play or t.n_steps < 200:
        return None
    race = os.path.basename(path).split(".npz")[0]
    comm = t.base_rate / 100.0
    k = lambda sec: int(round(sec / t.dt))
    live = ~t.suspended
    end = int(np.where(live)[0].max())
    rows = np.where(live & (t.t_rel >= -LAST_S) & (t.t_rel <= -MIN_LEFT_S))[0][::max(1, k(EVERY_S))]
    rows = rows[rows >= k(30)]
    if len(rows) < 5:
        return None
    bt, lt, bb, bl = _book(t)
    mt = (bt + lt) / 2.0  # mid in tick space
    mid = np.sqrt(bb * bl)
    p = 1.0 / mid
    share = p / np.nansum(p, 1, keepdims=True)
    vol = np.zeros((t.n_steps, t.n_runners))
    m = t.trade_step < t.n_steps
    np.add.at(vol, (t.trade_step[m], t.trade_runner[m].astype(int)), t.trade_vol[m])
    cvol = np.cumsum(vol, 0)
    d1 = np.diff(mt, axis=0, prepend=np.nan)  # tick change per row
    j5 = np.abs(mt - np.vstack([np.full((k(5), t.n_runners), np.nan), mt[:-k(5)]]))  # 5s move
    _, wap, wom = flow_arrays(t)
    recs = []
    for s in rows:
        okr = np.where(np.isfinite(mt[s]))[0]
        if len(okr) < 2:
            continue
        rank = np.empty(t.n_runners, int)
        rank[okr] = (-p[s, okr]).argsort().argsort() + 1
        s30 = s - k(30)
        dshare = share[s] - share[s30] if s30 >= 0 else np.full(t.n_runners, np.nan)
        e = s + 1
        if e > end:
            continue
        for r in okr:
            spread = lt[s, r] - bt[s, r]
            if spread > MAX_SPREAD or mid[s, r] > MAX_PRICE or not (np.isfinite(bb[e, r]) and np.isfinite(bl[e, r])):
                continue
            rec = dict(race=race, runner=int(r), t_to_start=float(-t.t_rel[s]), rank=int(rank[r]),
                       price=float(mid[s, r]), spread=int(spread), wom=float(wom[s, r]),
                       spn_gap=float(np.log(t.spn[s, r] / mid[s, r])) if t.spn[s, r] > 1 else np.nan)
            rec.update(flow_features(cvol, wap, wom, mid, s, r, k))
            for W in WINDOWS:
                s0 = s - k(W)
                if s0 < 0 or not np.isfinite(mt[s0, r]):
                    rec[f"net_{W}"] = rec[f"persist_{W}"] = rec[f"jump_{W}"] = rec[f"traded_{W}"] = np.nan
                    rec[f"steps_{W}"] = 0
                    continue
                net = -(mt[s, r] - mt[s0, r])  # + = shortened (steaming)
                ch = -d1[s0 + 1:s + 1, r]
                ch = ch[np.isfinite(ch) & (ch != 0)]
                same = (np.sign(ch) == np.sign(net)).mean() if len(ch) and net != 0 else np.nan
                rec[f"net_{W}"] = float(net)
                rec[f"persist_{W}"] = float(same)
                rec[f"steps_{W}"] = int(len(ch))
                rec[f"jump_{W}"] = float(np.nanmax(j5[s0 + k(5):s + 1, r])) if s - s0 > k(5) else np.nan
                rec[f"traded_{W}"] = float(cvol[s, r] - cvol[s0, r])
            # knock-on: the other runners' biggest 30s rise in implied probability, and this runner's lag
            others = np.setdiff1d(okr, [r])
            if s30 >= 0 and len(others) and np.isfinite(dshare[others]).any():
                li = others[int(np.nanargmax(dshare[others]))]
                ld = float(dshare[li])
                expect = -ld * share[s, r] / max(1 - share[s, li], 1e-6)
                rec["lead_dp"] = ld * 100
                rec["lag"] = float((dshare[r] - expect) * 100) if np.isfinite(dshare[r]) else np.nan
            else:
                rec["lead_dp"] = rec["lag"] = np.nan
            # executable round trips: cross in at e, cross out at the exit decision + 1 row
            for H in HORIZONS:
                x = min(s + k(H) + 1, end)
                rec[f"back_{H}"] = float(_net(bb[e, r] / bl[x, r] - 1, comm) * 100) if np.isfinite(bl[x, r]) else np.nan
                rec[f"lay_{H}"] = float(_net(1 - bl[e, r] / bb[x, r], comm) * 100) if np.isfinite(bb[x, r]) else np.nan
            last = min(e + k(TRAIL_MAX_S), end)
            for side in ("back", "lay"):
                x = min(_trail(mt[:, r], e, last, side, TRAIL_TICKS) + 1, end)
                if side == "back":
                    rec["back_trail"] = float(_net(bb[e, r] / bl[x, r] - 1, comm) * 100) if np.isfinite(bl[x, r]) else np.nan
                else:
                    rec["lay_trail"] = float(_net(1 - bl[e, r] / bb[x, r], comm) * 100) if np.isfinite(bb[x, r]) else np.nan
            # how much is left: best further move each way within 120s, in ticks (+ = shorten)
            seg = mt[e:min(e + k(MFE_S), end) + 1, r]
            if np.isfinite(seg).any():
                rec["max_shorten"] = float(mt[s, r] - np.nanmin(seg))
                rec["max_lengthen"] = float(np.nanmax(seg) - mt[s, r])
            else:
                rec["max_shorten"] = rec["max_lengthen"] = np.nan
            recs.append(rec)
    if not recs:
        return None
    d = pd.DataFrame(recs)
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    return d


def _one(path):
    try:
        return sample_race(path)
    except Exception as e:
        print(f"skip {os.path.basename(path)}: {e}", flush=True)
        return None


def collect(paths, workers: int) -> pd.DataFrame:
    os.environ["OMP_NUM_THREADS"] = "1"
    parts, t0 = [], time.time()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        for i, d in enumerate(pool.map(_one, paths, chunksize=4)):
            if d is not None:
                parts.append(d)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(paths)} races, {time.time() - t0:.0f}s", flush=True)
    if not parts:
        raise SystemExit("no races")
    return pd.concat(parts, ignore_index=True)


# ============================================================================ rules
EXITS = [f"{h}" for h in HORIZONS] + ["trail"]


def gradual_rules():
    for W, kk, pp, grad, runners, ex, act in itertools.product(WINDOWS, (2, 3, 4, 6), (0.0, 0.66, 0.8, 1.0),
                                                                (False, True), ("top3", "all"), EXITS,
                                                                ("follow", "fade")):
        yield dict(family="gradual", W=W, k=kk, persist=pp, gradual=grad, runners=runners, exit=ex, action=act)


CONFIRMS = ("none", "wom_with", "wom_against", "wap_room", "wap_past", "vol_accel", "spn_with")


def confirmed_rules():
    for W, kk, conf, runners, ex, act in itertools.product(WINDOWS, (2, 3, 4), CONFIRMS[1:], ("top3", "all"), EXITS,
                                                           ("follow", "fade")):
        yield dict(family="confirmed", W=W, k=kk, persist=0.66, gradual=True, confirm=conf, runners=runners, exit=ex,
                   action=act)


def _confirm(d: pd.DataFrame, conf: str, direction: np.ndarray) -> np.ndarray:
    """direction: +1 for a steamer (shortening), -1 for a drifter."""
    if conf == "none":
        return np.ones(len(d), bool)
    if conf in ("wom_with", "wom_against"):
        x = direction * d["wom_chg_30"].to_numpy()
        return x >= 0.05 if conf == "wom_with" else x <= -0.05
    if conf in ("wap_room", "wap_past"):
        x = direction * d["wap_gap"].to_numpy()  # steamer still longer than its WAP = room to shorten
        return x >= 1.0 if conf == "wap_room" else x <= -1.0
    if conf == "vol_accel":
        return d["vol_accel"].to_numpy() >= 2.0
    if conf == "spn_with":
        return direction * -d["spn_gap"].to_numpy() >= 0.02  # projected SP already shorter than a steamer's price
    raise ValueError(conf)


def knock_rules():
    for ld, lag, runners, ex, act in itertools.product((0.5, 1.0, 2.0, 4.0), (0.25, 0.5, 1.0, 2.0), ("top3", "all"),
                                                       EXITS, ("follow", "fade")):
        yield dict(family="knock-on", lead=ld, lag=lag, runners=runners, exit=ex, action=act)


def apply_rule(d: pd.DataFrame, r: dict) -> pd.DataFrame:
    """The trades a rule takes: (race, ret) rows.
    gradual  follow = back a steamer (net > 0) / lay a drifter; fade = the opposite
    knock-on follow = finish the rebalancing: lay a runner that has not lengthened enough (lag > 0), back one that
             has lengthened too much (lag < 0); fade = the opposite"""
    sel = (d["rank"] <= 3).to_numpy() if r["runners"] == "top3" else np.ones(len(d), bool)
    if r["family"] in ("gradual", "confirmed"):
        net = d[f"net_{r['W']}"].to_numpy()
        sel = sel & (np.abs(net) >= r["k"]) & (d[f"persist_{r['W']}"].to_numpy() >= r["persist"])
        if r["gradual"]:
            sel = sel & (d[f"jump_{r['W']}"].to_numpy() < 2)
        if r["family"] == "confirmed":
            sel = sel & _confirm(d, r["confirm"], np.sign(net))
        back_if_follow = net > 0
    else:
        lag = d["lag"].to_numpy()
        sel = sel & (d["lead_dp"].to_numpy() >= r["lead"]) & (np.abs(lag) >= r["lag"])
        back_if_follow = lag < 0
    back = back_if_follow if r["action"] == "follow" else ~back_if_follow
    ex = r["exit"]
    ret = np.where(back, d[f"back_{ex}"].to_numpy(), d[f"lay_{ex}"].to_numpy())
    return pd.DataFrame({"race": d["race"].to_numpy()[sel], "ret": ret[sel]}).dropna()


def rule_name(r: dict) -> str:
    if r["family"] == "confirmed":
        return f"confirmed W{r['W']} k{r['k']} {r['confirm']} {r['runners']} exit {r['exit']} {r['action']}"
    if r["family"] == "gradual":
        return (f"gradual W{r['W']} k{r['k']} p{r['persist']} {'nojump' if r['gradual'] else 'any'} {r['runners']} "
                f"exit {r['exit']} {r['action']}")
    return f"knock-on lead{r['lead']} lag{r['lag']} {r['runners']} exit {r['exit']} {r['action']}"


def search(tr: pd.DataFrame, rules, min_races: int) -> pd.DataFrame:
    rows = []
    for r in rules:
        tt = apply_rule(tr, r)
        if not len(tt):
            continue
        s = rt(tt["ret"], tt["race"])
        if s["races"] < min_races:
            continue
        rows.append(dict(rule=rule_name(r), **{k_: v for k_, v in r.items()}, trades=len(tt), races=s["races"],
                         mean=s["mean"], t=s["t"]))
    return pd.DataFrame(rows)


def score(d: pd.DataFrame, r: dict) -> dict:
    tt = apply_rule(d, r)
    s = rt(tt["ret"], tt["race"]) if len(tt) else dict(mean=np.nan, t=np.nan, races=0)
    return dict(trades=len(tt), races=s["races"], mean=s["mean"], t=s["t"])


def left_table(d: pd.DataFrame) -> pd.DataFrame:
    """How much of a detected move is left: further best move the same way vs against, within 120s (ticks)."""
    rows = []
    for W in WINDOWS:
        for kk in (2, 3, 4, 6):
            for grad in (False, True):
                net = d[f"net_{W}"]
                m = (net.abs() >= kk) & (d[f"persist_{W}"] >= 0.8)
                if grad:
                    m &= d[f"jump_{W}"] < 2
                g = d[m]
                if len(g) < 50:
                    continue
                st = g[f"net_{W}"] > 0
                fav = np.where(st, g["max_shorten"], g["max_lengthen"])
                adv = np.where(st, g["max_lengthen"], g["max_shorten"])
                cost = g["spread"].to_numpy() * 2  # ticks given up crossing in and out
                rows.append(dict(W=W, k=kk, gradual=grad, samples=len(g), races=g["race"].nunique(),
                                 median_moved=float(np.median(np.abs(g[f"net_{W}"]))),
                                 median_further_same_way=float(np.nanmedian(fav)),
                                 median_against=float(np.nanmedian(adv)),
                                 pct_further_beats_cost=float(np.nanmean(fav >= cost) * 100),
                                 pct_more_with_than_against=float(np.nanmean(fav > adv) * 100)))
    return pd.DataFrame(rows)


def run(d: pd.DataFrame, label: str, out: str, min_races: int, top: int = 15) -> dict:
    tr_r, va_r, te_r = split_by_date(sorted(d["race"].unique()))
    tr, va, te = d[d["race"].isin(tr_r)], d[d["race"].isin(va_r)], d[d["race"].isin(te_r)]
    print(f"[{label}] {d['race'].nunique()} races, {len(d):,} samples; TRAIN {len(tr_r)} / VAL {len(va_r)} / "
          f"TEST {len(te_r)} races", flush=True)
    res = {}
    for fam, gen in (("gradual", gradual_rules), ("confirmed", confirmed_rules), ("knock-on", knock_rules)):
        rules = list(gen())
        s = search(tr, rules, min_races)
        if not len(s):
            print(f"  {fam}: no rule trades in >= {min_races} TRAIN races")
            continue
        s = s.sort_values("t", ascending=False)
        s.to_csv(os.path.join(out, f"{label}_rules_{fam}.csv"), index=False)
        best = s.head(top).copy()
        for split_name, df in (("val", va), ("test", te)):
            sc = [score(df, {k_: row[k_] for k_ in row.index if k_ in (
                "family", "W", "k", "persist", "gradual", "confirm", "runners", "exit", "action", "lead", "lag")})
                for _, row in best.iterrows()]
            best[f"{split_name}_mean"] = [x["mean"] for x in sc]
            best[f"{split_name}_t"] = [x["t"] for x in sc]
            best[f"{split_name}_races"] = [x["races"] for x in sc]
        keep = ["rule", "trades", "races", "mean", "t", "val_mean", "val_t", "val_races", "test_mean", "test_t",
                "test_races"]
        print(f"\n  {fam}: top {top} of {len(s)} rules on TRAIN (by t), then VAL / TEST (pre-registered = the first)")
        print(best[keep].round(3).to_string(index=False))
        b = best.iloc[0]
        edge = bool(b["test_mean"] > 0 and (b["test_t"] or 0) > 2 and b["test_races"] >= 30 and b["val_mean"] > 0)
        res[fam] = dict(rule=b["rule"], train_mean=b["mean"], train_t=b["t"], val_mean=b["val_mean"],
                        val_t=b["val_t"], test_mean=b["test_mean"], test_t=b["test_t"],
                        test_races=int(b["test_races"]), edge=edge,
                        train_rules_positive_t2=int(((s["mean"] > 0) & (s["t"] > 2)).sum()), rules_tested=len(s))
    return res


MODEL_FEATS = ([f"{x}_{W}" for W in WINDOWS for x in ("net", "persist", "steps", "jump", "traded")] +
               ["wom", "wom_chg_30", "wap_gap", "vol_accel", "spn_gap", "lead_dp", "lag", "rank", "price", "spread",
                "t_to_start"])
MODEL_LABELS = ["back_60", "lay_60", "back_trail", "lay_trail"]


def model_test(d: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Gradient-boosted trees on every feature predict the four executable returns (TRAIN days); trade the best
    prediction above a threshold picked on VAL; TEST once."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    tr_r, va_r, te_r = split_by_date(sorted(d["race"].unique()))
    tr, va, te = d[d["race"].isin(tr_r)], d[d["race"].isin(va_r)], d[d["race"].isin(te_r)]
    feats = [f for f in MODEL_FEATS if f in d and tr[f].notna().any() and tr[f].nunique() > 1]
    Xtr, Xva, Xte = (x[feats].to_numpy(np.float32) for x in (tr, va, te))
    pv, pt = [], []
    for y in MODEL_LABELS:
        m = tr[y].notna().to_numpy()
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                            l2_regularization=1.0, random_state=seed)
        reg.fit(Xtr[m], np.clip(tr[y].to_numpy()[m], -50, 50))
        pv.append(reg.predict(Xva))
        pt.append(reg.predict(Xte))
    Pva, Pte = np.stack(pv, 1), np.stack(pt, 1)

    def trades(df, P, thr):
        j = P.argmax(1)
        sel = P[np.arange(len(P)), j] > thr
        ret = df[MODEL_LABELS].to_numpy()[np.where(sel)[0], j[sel]]
        return pd.DataFrame({"race": df["race"].to_numpy()[sel], "ret": ret}).dropna()

    rows, best = [], None
    min_r = min(20, max(3, va["race"].nunique() // 3))
    for thr in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        v = trades(va, Pva, thr)
        r = rt(v["ret"], v["race"]) if len(v) else dict(mean=np.nan, t=np.nan, races=0)
        rows.append(dict(threshold=thr, val_trades=len(v), val_mean=r["mean"], val_t=r["t"], val_races=r["races"]))
        if r["races"] >= min_r and np.isfinite(r["mean"]) and (best is None or r["mean"] > best[1]):
            best = (thr, r["mean"])
    res = {}
    if best is not None:
        tt = trades(te, Pte, best[0])
        r = rt(tt["ret"], tt["race"]) if len(tt) else dict(mean=np.nan, t=np.nan, races=0)
        res = dict(threshold=best[0], test_trades=len(tt), test_mean=r["mean"], test_t=r["t"], test_races=r["races"],
                   edge=bool(r["races"] >= 30 and (r["mean"] or -1) > 0 and (r["t"] or 0) > 2 and best[1] > 0))
    return pd.DataFrame(rows), res


def control_tapes(out_dir: str, n: int = 120) -> list[str]:
    """Synthetic races with a planted GRADUAL steamer (1 tick every 6s) among random walkers."""
    from .synthetic import make_synthetic_tape

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(out_dir, f"2099{i % 30 + 1:04d}_1200_Ctl_1_{i}.npz")
        if not os.path.exists(p):
            t = make_synthetic_tape(80_000 + i, trend_every_s=6.0)
            t.name = os.path.basename(p)[:-4]
            t.save(p)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-races", type=int, default=100, help="a TRAIN rule must trade in this many races")
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    verdict = {}
    if not a.skip_control:
        print("=== 0. Positive control: synthetic races with a planted gradual steamer ===", flush=True)
        dc = collect(control_tapes(os.path.join(a.out, "control_tapes")), a.workers)
        rc = run(dc, "control", a.out, min_races=10, top=5)
        g = rc.get("gradual", {})
        verdict["control_found_gradual_move"] = bool(g and g["test_mean"] > 0 and (g["test_t"] or 0) > 2)
        print("control: the gradual follow rules FIND the planted steamer" if verdict["control_found_gradual_move"]
              else "control: NOT found -> a null result on real races means little", flush=True)

    cache = os.path.join(a.out, "samples.parquet")
    d = pd.read_parquet(cache) if os.path.exists(cache) else None
    if d is not None and any(f not in d for f in MODEL_FEATS):
        print(f"\n{cache} is from an older version: rebuilding the samples", flush=True)
        d = None
    elif d is not None:
        print(f"\nreusing {cache}: {len(d):,} samples", flush=True)
    if d is None:
        d = collect(list_tapes(a.tapes), a.workers)
        d.to_parquet(cache, index=False)

    print("\n=== 1. How much of a detected move is left? (persistence >= 0.8; ticks; 'cost' = 2 x spread) ===")
    lt = left_table(d)
    lt.to_csv(os.path.join(a.out, "how_much_is_left.csv"), index=False)
    print(lt.round(2).to_string(index=False))

    print("\n=== 2. Rule search on TRAIN days, best rules scored on VAL and TEST ===")
    res = run(d, "real", a.out, a.min_races)

    print("\n=== 3. Gradient-boosted model on every feature (moves, order flow, WOM, WAP, projected SP, knock-on): "
          "threshold on VAL, TEST once ===")
    mv, mres = model_test(d)
    mv.to_csv(os.path.join(a.out, "model_val.csv"), index=False)
    print(mv.round(3).to_string(index=False))
    print(json.dumps(mres, indent=1, default=float))
    verdict.update(real=res, model=mres, edge=any(v["edge"] for v in res.values()) or bool(mres.get("edge")))
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    if verdict["edge"]:
        print("EDGE: a move caught early pays on unseen days -> pre-register it and forward-test")
    elif verdict.get("control_found_gradual_move") is False:
        print("INCONCLUSIVE: the rules could not find the planted gradual steamer either")
    else:
        print("NO EDGE: moves caught early - gradual or knock-on - do not leave enough to beat the round trip on "
              "unseen days")
    print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
