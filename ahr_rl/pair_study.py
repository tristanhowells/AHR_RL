"""P6: paired bets. Every action is a bet plus its own unbiased hedge.

Instead of individual bets, the agent's only choice per runner per decision
(every 10s) is BACK pair, LAY pair or nothing:

  BACK pair  back $10 now (crossing the spread, <= 2 ticks of slippage) and at once
             rest a LAY k ticks lower, sized to green the runner (equal profit on
             every outcome). If the price shortens k ticks the pair is green.
  LAY pair   the mirror: lay now for a $10 liability, rest a BACK k ticks higher.

Both sides put $10 at risk, so every pair locks the same $10 of the bank.

A hedge that hasn't filled by the off is settled one of three ways:

  close  green it at market at the scheduled start (the env's auto-green)
  bsp    the resting hedge converts to a BSP bet at the off (Betfair's
         MARKET_ON_CLOSE persistence)
  hold   it never gets hedged and the bet runs to the result

Hedge fills: 'through' (a trade one tick beyond the hedge price - conservative)
or 'touch' (a trade at the price - optimistic). The hedge lands two steps after
the decision (1s): one step for the entry, one for the hedge.

This answers the question before any RL is trained on the action spec:

  A  What each action is worth on average (side x k x settle x fill), by time
     to the off and favourite rank.
  B  Headroom: what a hindsight oracle picking back / lay / nothing would make.
  C  Learnability: gradient-boosted models predict each pair's P&L from the
     order-book state. The config and threshold are picked on the last quarter of
     TRAIN days (models fitted on the first three quarters), then refitted on all
     TRAIN days and scored once on HOLDOUT days. "edge" = holdout mean > 0, t > 2,
     >= 20 races.
  E  One k-tick pair vs a ladder of k=1 pairs (re-enter each time a hedge fills) to
     the same target: same gross move, but every leg pays the spread again.
  D  A $500 bank per race, $10 pairs, on HOLDOUT races: the model policy vs a
     random policy making as many pairs, vs the oracle. Each open pair locks its
     worst loss until it is hedged or settled; no pair if funds don't cover it.

P&L is per $1 at risk (back: stake, lay: liability), after commission.
Config selection in C uses the hedged settles (close / bsp) only: 'hold' is a bet
on the result and is shown for information.

    python -m ahr_rl.pair_study --tapes "data/tapes/*.npz" --out runs/pair_study
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import race_filter
from .env import list_tapes, split_by_date
from .jump_study import _round_trip, _walk
from .ladder import N_TICKS, PRICES
from .tape import Tape

STAKE = 10.0
BANK = 500.0
KS = (1, 2, 3, 5, 8)
SETTLES = ("close", "bsp", "hold")
FILLS = ("through", "touch")
SIDES = ("back", "lay")
FEATURES = ["t", "field", "rank", "prob", "logp", "spread", "wom", "wom_chg30", "mom_30", "mom_120", "wap_sess",
            "vol_rate", "vol_accel", "vol_share", "spn_gap", "depth3"]
TIME_BINS = [-601, -300, -120, -30, 0]
TIME_LABELS = ["10-5m", "5-2m", "2m-30s", "last 30s"]
RANK_BINS = [0, 1, 2, 3, 6, 99]
RANK_LABELS = ["1", "2", "3", "4-6", "7+"]


# ----------------------------------------------------------------- sampling
def sample_tape(path: str, every_s: float = 10.0, stake: float = STAKE) -> pd.DataFrame | None:
    tape = Tape.load(path)
    if not tape.went_in_play or tape.winner < 0:
        return None
    race = os.path.basename(path).split(".npz")[0]
    dt, T, R = tape.dt, tape.n_steps, tape.n_runners
    ok = ~tape.suspended.copy()
    if T > 1:
        ok[-1] = False  # the in-play snapshot
    if not ok.any():
        return None
    end = int(np.where(ok)[0].max())
    comm = tape.base_rate / 100.0
    t = tape.t_rel.astype(float)
    after0 = np.where(t >= 0)[0]
    s_close = min(int(after0[0]) if len(after0) else end, end)
    k = lambda sec: int(round(sec / dt))

    bt, lt = tape.back_tick[:, :, 0].astype(int), tape.lay_tick[:, :, 0].astype(int)
    valid = (bt >= 0) & (lt >= 0) & tape.active & ok[:, None]
    valid[end + 1:] = False
    midt = np.where(valid, (bt + lt) / 2.0, np.nan)
    midp = np.where(valid, np.sqrt(PRICES[np.maximum(bt, 0)] * PRICES[np.maximum(lt, 0)]), np.nan)
    ip = np.where(valid, 1.0 / midp, 0.0)
    share = ip / np.maximum(ip.sum(1, keepdims=True), 1e-12) * 100
    bs3, ls3 = tape.back_size[:, :, :3].sum(-1), tape.lay_size[:, :, :3].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        wom = np.where(bs3 + ls3 > 0, bs3 / (bs3 + ls3), np.nan)

    st = np.clip(tape.trade_step, 0, T - 1)
    vol, vtk = np.zeros((T, R)), np.zeros((T, R))
    np.add.at(vol, (st, tape.trade_runner), tape.trade_vol)
    np.add.at(vtk, (st, tape.trade_runner), tape.trade_vol * tape.trade_tick)
    cv, cvt = np.cumsum(vol, 0), np.cumsum(vtk, 0)
    tmin = np.full((T, R), 10 ** 6, np.int64)  # lowest / highest tick traded in (step-1, step]
    tmax = np.full((T, R), -1, np.int64)
    np.minimum.at(tmin, (st, tape.trade_runner), tape.trade_tick.astype(np.int64))
    np.maximum.at(tmax, (st, tape.trade_runner), tape.trade_tick.astype(np.int64))

    def window(c, s, w):
        return c[s] - (c[s - w] if s - w >= 0 else 0.0)

    bad = np.zeros(T, bool)
    for s in tape.removal_step:
        bad[max(0, s - k(10)): min(T, s + k(10) + 1)] = True
    last_removal = int(tape.removal_step.max()) if len(tape.removal_step) else -1

    in_book = dict(back=(tape.back_tick, tape.back_size), lay=(tape.lay_tick, tape.lay_size))
    rows = []
    for s in range(k(30), s_close - 2, k(every_s)):
        if bad[s] or s < last_removal:  # a later scratching would re-price (reduce) matched bets
            continue
        live = np.where(valid[s])[0]
        if len(live) < 2:
            continue
        order = live[np.argsort(midp[s, live])]
        rank = {int(r): i + 1 for i, r in enumerate(order)}
        mkt_v60 = sum(window(cv[:, r], s, k(60)) for r in live)
        s_in = s + 1
        for r in live:
            r = int(r)
            if not valid[s_in, r] or not tape.active[end, r]:
                continue
            v60 = window(cv[:, r], s, k(60))
            v30a, v30b = window(cv[:, r], s, k(30)), window(cv[:, r], s - k(30), k(30))
            wap_s = cvt[s, r] / cv[s, r] if cv[s, r] >= 20 else np.nan
            spn = float(tape.spn[s, r])
            rec = dict(
                race=race, day=race[:8], runner=r, s=s, s_close=s_close, end=end, t=t[s], field=len(live),
                rank=rank[r], prob=share[s, r], logp=np.log(midp[s, r]), spread=int(lt[s, r] - bt[s, r]),
                wom=wom[s, r], wom_chg30=wom[s, r] - wom[s - k(30), r],
                mom_30=midt[s, r] - midt[s - k(30), r],
                mom_120=(midt[s, r] - midt[s - k(120), r]) if s >= k(120) else np.nan,
                wap_sess=midt[s, r] - wap_s, vol_rate=np.log1p(v60), vol_accel=np.log((v30a + 1) / (v30b + 1)),
                vol_share=(v60 / mkt_v60 * 100 - share[s, r]) if mkt_v60 > 0 else np.nan,
                spn_gap=np.log(spn / midp[s, r]) if spn > 1 else np.nan,
                depth3=np.log1p(bs3[s, r] + ls3[s, r]),
                comm=comm, bsp=float(tape.bsp[r]), win=int(r == tape.winner),
            )
            for side in SIDES:
                ticks, sizes = in_book[side]
                e0 = int(ticks[s_in, r, 0])
                st_side = stake if side == "back" else stake / max(PRICES[e0] - 1, 0.01)  # $10 liability
                p_in, f = _walk(ticks[s_in, r], sizes[s_in, r], st_side, 2)
                rec[f"{side}_p"], rec[f"{side}_f"], rec[f"{side}_e0"] = p_in, f, e0
                rec[f"{side}_gclose"] = (_round_trip(tape, r, s_in, s_close, side, st_side, comm)[0]
                                         if np.isfinite(p_in) and s_close > s_in else np.nan)
                lo = s + 3  # hedge rests from here on
                seg = tmin[lo:end + 1, r] if side == "back" else tmax[lo:end + 1, r]
                cm = np.minimum.accumulate(seg) if side == "back" else np.maximum.accumulate(seg)
                for kk in KS:
                    h = e0 - kk if side == "back" else e0 + kk
                    for fill in FILLS:
                        if side == "back":
                            thr = h - 1 if fill == "through" else h
                            hit = cm <= thr
                        else:
                            thr = h + 1 if fill == "through" else h
                            hit = cm >= thr
                        rec[f"fs_{side}_{kk}_{fill}"] = lo + int(hit.argmax()) if len(hit) and hit.any() else -1
                legs = _ladder_legs(tape, valid, tmin if side == "back" else tmax, r, s, side, end, stake)
                for kk in KS:
                    for st in SETTLES:
                        v, n = _ladder_value(tape, r, legs, side, kk, st, s_close, comm, rec["win"] == 1)
                        rec[f"lad_{side}_{kk}_{st}"], rec[f"ladn_{side}_{kk}_{st}"] = v, n
            rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


# ----------------------------------------------------------------- k=1 ladder
MAX_LEGS = 25


def _ladder_legs(tape: Tape, valid, textreme, r: int, s: int, side: str, end: int, stake: float) -> list[tuple]:
    """Chain of k=1 pairs: enter (crossing the spread), rest the hedge one tick better; when
    it fills ('through'), re-enter at once at the new best price with another k=1 pair.
    Runs until a hedge doesn't fill, the book is gone, the price is max(KS) ticks past the
    first entry, or MAX_LEGS. Each leg is (s_in, p_in, filled fraction, stake, hedge tick,
    hedge fill step or -1). The first leg puts up to $10 at risk; every re-entry is sized
    to the amount the first leg actually got matched for, so exposure stays constant."""
    ticks, sizes = (tape.back_tick, tape.back_size) if side == "back" else (tape.lay_tick, tape.lay_size)
    legs, d, e_first, risk = [], s, None, stake
    while len(legs) < MAX_LEGS:
        s_in = d + 1
        if s_in > end or not valid[s_in, r]:
            break
        e0 = int(ticks[s_in, r, 0])
        st = risk if side == "back" else risk / max(PRICES[e0] - 1, 0.01)
        p_in, f = _walk(ticks[s_in, r], sizes[s_in, r], st, 2)
        if not np.isfinite(p_in) or f <= 0:
            break
        if e_first is None:
            e_first = e0
            risk = st * f * (1.0 if side == "back" else p_in - 1)
        h = int(np.clip(e0 - 1 if side == "back" else e0 + 1, 0, N_TICKS - 1))
        lo = d + 3  # entry lands at d+1, the hedge one step later
        seg = textreme[lo:end + 1, r]
        hit = seg <= h - 1 if side == "back" else seg >= h + 1
        fs = lo + int(hit.argmax()) if len(hit) and hit.any() else -1
        legs.append((s_in, p_in, f, st, h, fs, e_first))
        if fs < 0 or (h <= e_first - max(KS) if side == "back" else h >= e_first + max(KS)):
            break
        d = fs
    return legs


def _ladder_value(tape: Tape, r: int, legs, side: str, k: int, settle: str, s_close: int, comm: float,
                  win: bool):
    """The k=1 ladder that stops once a hedge k ticks past the first entry has filled.
    Returns (P&L per $1 at risk on the first leg, legs used). 'close' only uses legs
    entered before the scheduled start and greens the open one at market there; 'bsp' /
    'hold' settle the open leg at BSP / on the result. Commission is charged per leg
    (as for the single pair)."""
    if not legs:
        return np.nan, 0
    e_first = legs[0][6]
    s_in0, p0, f0, st0 = legs[0][:4]
    risk0 = st0 * f0 * (1.0 if side == "back" else p0 - 1)
    usd, n = 0.0, 0
    for s_in, p_in, f, st, h, fs, _ in legs:
        if settle == "close" and s_in >= s_close:
            break  # no new entries after the auto-green
        n += 1
        if fs >= 0 and (settle != "close" or fs <= s_close):
            usd += st * f * float(_green(side, p_in, PRICES[h], comm))
            if (h <= e_first - k) if side == "back" else (h >= e_first + k):
                break  # reached the single pair's target: stop, flat
            continue
        # this leg is still open at the off / scheduled start
        if settle == "close":
            g = _round_trip(tape, r, s_in, s_close, side, st, comm)[0]
        elif settle == "bsp":
            g = float(_green(side, p_in, float(tape.bsp[r]), comm))
        elif side == "back":
            g = (p_in - 1) * (1 - comm) if win else -1.0
        else:
            g = -(p_in - 1) if win else 1 - comm
        if not np.isfinite(g):
            return np.nan, n
        usd += st * f * g
        break
    return usd / risk0, n


def _one(args):
    path, kw = args
    try:
        return sample_tape(path, **kw)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}", flush=True)
        return None


# ----------------------------------------------------------------- pair P&L
def _green(side: str, p_in, p_out, comm):
    p_in, p_out = np.asarray(p_in, float), np.asarray(p_out, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = p_in / p_out - 1 if side == "back" else 1 - p_in / p_out
        g = np.where(g > 0, g * (1 - comm), g)
    return np.where((p_in > 1) & (p_out > 1), g, np.nan)


def pair_pnl(df: pd.DataFrame, side: str, k: int, settle: str, fill: str):
    """(P&L per $1 at risk, hedge filled, step the pair is done)."""
    p_in, comm = df[f"{side}_p"].to_numpy(float), df["comm"].to_numpy(float)
    e0 = df[f"{side}_e0"].to_numpy(int)
    h = np.clip(e0 - k if side == "back" else e0 + k, 0, N_TICKS - 1)
    fs = df[f"fs_{side}_{k}_{fill}"].to_numpy(int)
    last = df["s_close" if settle == "close" else "end"].to_numpy(int)
    filled = (fs >= 0) & (fs <= last)
    g_fill = _green(side, p_in, PRICES[h], comm)
    if settle == "close":
        g_open = df[f"{side}_gclose"].to_numpy(float)
    elif settle == "bsp":
        g_open = _green(side, p_in, df["bsp"].to_numpy(float), comm)
    else:
        win = df["win"].to_numpy(int) == 1
        g_open = (np.where(win, (p_in - 1) * (1 - comm), -1.0) if side == "back"
                  else np.where(win, -(p_in - 1), 1 - comm))
    g = np.where(filled, g_fill, g_open)
    if side == "lay":  # per $1 of liability
        g = g / (p_in - 1)
    g = np.where(np.isfinite(p_in) & (e0 >= 0), g, np.nan)
    return g, filled, np.where(filled, fs, last)


def ladder_table(df: pd.DataFrame, fill: str = "through") -> pd.DataFrame:
    """Single k-tick pair vs a chain of k=1 pairs that stops at the same target, on the
    same decisions. diff = ladder - single, race-clustered t on the per-decision gap."""
    out = []
    race = df["race"].to_numpy()
    for side in SIDES:
        for k in KS:
            for st in SETTLES:
                single = pair_pnl(df, side, k, st, fill)[0]
                lad = df[f"lad_{side}_{k}_{st}"].to_numpy(float)
                m = np.isfinite(single) & np.isfinite(lad)
                d = race_t((lad[m] - single[m]) * 100, race[m])
                out.append(dict(side=side, k=k, settle=st, single_pct=np.mean(single[m]) * 100,
                                ladder_pct=np.mean(lad[m]) * 100, diff_pct=d["mean"], diff_t=d["t"],
                                legs=df[f"ladn_{side}_{k}_{st}"].to_numpy(float)[m].mean(),
                                ladder_pos_pct=(lad[m] > 0).mean() * 100, n=int(m.sum())))
    return pd.DataFrame(out)


def configs():
    return [(k, st, fl) for k in KS for st in SETTLES for fl in FILLS]


def race_t(v, race) -> dict:
    x = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna()
    if len(x) < 2:
        return dict(n=len(x), races=x["race"].nunique(), mean=np.nan, t=np.nan)
    per = x.groupby("race")["v"].mean()
    sd = per.std(ddof=1)
    return dict(n=len(x), races=len(per), mean=float(x["v"].mean()),
                t=float(per.mean() / (sd / np.sqrt(len(per)))) if len(per) > 1 and sd > 0 else np.nan)


# ----------------------------------------------------------------- A, B
def value_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for k, st, fl in configs():
        for side in SIDES:
            g, filled, _ = pair_pnl(df, side, k, st, fl)
            m = np.isfinite(g)
            rt = race_t(g[m] * 100, df["race"].to_numpy()[m])
            out.append(dict(side=side, k=k, settle=st, fill=fl, hedged_pct=filled[m].mean() * 100,
                            mean_pct=rt["mean"], t=rt["t"], pos_pct=(g[m] > 0).mean() * 100, n=rt["n"]))
    return pd.DataFrame(out)


def segment_table(df: pd.DataFrame, k: int, settle: str, fill: str) -> pd.DataFrame:
    d = df[["race"]].copy()
    d["time_b"] = pd.cut(df["t"], TIME_BINS, labels=TIME_LABELS)
    d["rank_b"] = pd.cut(df["rank"], RANK_BINS, labels=RANK_LABELS)
    for side in SIDES:
        d[side] = pair_pnl(df, side, k, settle, fill)[0] * 100
    rows = []
    for (tb, rb), g in d.groupby(["time_b", "rank_b"], observed=True):
        row = dict(time=tb, rank=rb, n=len(g))
        for side in SIDES:
            rt = race_t(g[side], g["race"])
            row[f"{side}_mean_pct"], row[f"{side}_t"] = rt["mean"], rt["t"]
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for k, st, fl in configs():
        gb = pair_pnl(df, "back", k, st, fl)[0]
        gl = pair_pnl(df, "lay", k, st, fl)[0]
        best = np.fmax(np.fmax(np.nan_to_num(gb, nan=-9), np.nan_to_num(gl, nan=-9)), 0.0)
        out.append(dict(k=k, settle=st, fill=fl, any_positive_pct=(best > 0).mean() * 100,
                        oracle_mean_pct=best.mean() * 100, back_pos_pct=np.nanmean(gb > 0) * 100,
                        lay_pos_pct=np.nanmean(gl > 0) * 100))
    return pd.DataFrame(out)


# ----------------------------------------------------------------- C
def _fit(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = np.isfinite(y)
    lo, hi = np.nanpercentile(y[m], [1, 99])
    mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                        l2_regularization=1.0, random_state=0)
    mdl.fit(X[m], np.clip(y[m], lo, hi))
    return mdl


def _policy_pnl(pred_b, pred_l, gb, gl):
    """The agent takes the side with the higher prediction."""
    take_back = pred_b >= pred_l
    return np.where(take_back, pred_b, pred_l), np.where(take_back, gb, gl), take_back


def learnability(df_fit: pd.DataFrame, df_eval: pd.DataFrame, fill: str, qs=(0.95, 0.99),
                 settles=SETTLES) -> pd.DataFrame:
    from scipy.stats import spearmanr

    Xf, Xe = df_fit[FEATURES].to_numpy(float), df_eval[FEATURES].to_numpy(float)
    out = []
    for k in KS:
        for st in settles:
            preds, gs = {}, {}
            for side in SIDES:
                yf = pair_pnl(df_fit, side, k, st, fill)[0]
                mdl = _fit(Xf, yf)
                preds[side + "_fit"] = mdl.predict(Xf)
                preds[side] = mdl.predict(Xe)
                gs[side] = pair_pnl(df_eval, side, k, st, fill)[0]
            score_f = np.fmax(preds["back_fit"], preds["lay_fit"])
            score, g, _ = _policy_pnl(preds["back"], preds["lay"], gs["back"], gs["lay"])
            row = dict(k=k, settle=st, fill=fill)
            for side in SIDES:
                m = np.isfinite(gs[side])
                row[f"ic_{side}"] = float(spearmanr(preds[side][m], gs[side][m])[0]) if m.sum() > 10 else np.nan
            for q in qs:
                thr = float(np.quantile(score_f, q))
                sel = (score >= thr) & np.isfinite(g)
                rt = race_t(g[sel] * 100, df_eval["race"].to_numpy()[sel])
                row[f"top{100 - q * 100:.0f}_thr"] = thr * 100
                row[f"top{100 - q * 100:.0f}_mean_pct"] = rt["mean"]
                row[f"top{100 - q * 100:.0f}_t"] = rt["t"]
                row[f"top{100 - q * 100:.0f}_races"] = rt["races"]
                row[f"top{100 - q * 100:.0f}_n"] = rt["n"]
            out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------- D
def bank_sim(df: pd.DataFrame, act: np.ndarray, side_back: np.ndarray, g_back, g_lay, done_b, done_l) -> pd.DataFrame:
    """Per race: $500, $10-at-risk pairs in decision order; an open pair locks its
    $10 until it is done."""
    d = df[["race", "s", "back_p", "lay_p", "back_f", "lay_f"]].copy()
    d["act"], d["sb"] = act, side_back
    d["g"] = np.where(side_back, g_back, g_lay)
    d["done"] = np.where(side_back, done_b, done_l)
    d = d[d["act"] & np.isfinite(d["g"])]
    out = []
    for race, g in d.sort_values(["race", "s"]).groupby("race", sort=False):
        open_, pnl, n = [], 0.0, 0
        for row in g.itertuples(index=False):
            open_ = [(done, lock) for done, lock in open_ if done > row.s]
            f = row.back_f if row.sb else row.lay_f
            lock = STAKE * f
            if sum(x for _, x in open_) + lock > BANK:
                continue
            open_.append((row.done, lock))
            pnl += STAKE * f * row.g
            n += 1
        out.append(dict(race=race, pnl=pnl, pairs=n))
    return pd.DataFrame(out)


def _bank_summary(name, b: pd.DataFrame, races) -> dict:
    b = b.set_index("race").reindex(races).fillna(0.0)
    p = b["pnl"]
    t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p))) if len(p) > 1 and p.std(ddof=1) > 0 else np.nan
    return dict(policy=name, races=len(p), pairs_per_race=b["pairs"].mean(), mean_race_usd=p.mean(), t=t,
                total_usd=p.sum(), pct_races_up=(p > 0).mean() * 100, worst_race_usd=p.min(), best_race_usd=p.max())


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--cache", default=None, help="pairs.parquet to reuse / write (default: <out>/pairs.parquet)")
    race_filter.add_args(ap)
    ap.add_argument("--max-races", type=int, default=100000)
    ap.add_argument("--every-s", type=float, default=10.0)
    ap.add_argument("--fill", default="through", choices=FILLS, help="hedge fill rule for sections C and D")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    t0 = time.time()
    cache = a.cache or os.path.join(a.out, "pairs.parquet")
    paths = list_tapes(a.tapes)[: a.max_races]
    tr, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}  # same days as the unfiltered run
    paths = race_filter.filter_paths(paths, a)
    keep = {os.path.basename(p).split(".npz")[0] for p in paths}
    df = pd.read_parquet(cache) if os.path.exists(cache) else None
    if df is not None and "lad_back_1_close" not in df.columns:
        print(f"{cache} predates the ladder variant: rebuilding")
        df = None
    if df is not None:
        print(f"loaded {cache}")
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        parts = []
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for i, res in enumerate(pool.map(_one, [(p, dict(every_s=a.every_s)) for p in paths], chunksize=2)):
                if res is not None:
                    parts.append(res)
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(paths)} races, {time.time() - t0:.0f}s", flush=True)
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(cache, index=False)
    df = df[df["race"].isin(keep)].reset_index(drop=True)  # the race filter (a cache may hold more races)
    df["split"] = np.where(df["day"].isin(hold_days), "holdout", "train")
    train, hold = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "holdout"].reset_index(drop=True)
    print(f"\n{len(df)} decisions (runner x 10s) from {df['race'].nunique()} races: "
          f"train {train['race'].nunique()} races, holdout {hold['race'].nunique()} races  [{time.time() - t0:.0f}s]")

    # ---------------- A
    print("\n=== A. What each action is worth (all races; P&L % of the $10 at risk, race-clustered t) ===")
    vt = value_table(df)
    vt.to_csv(os.path.join(a.out, "value_table.csv"), index=False)
    for fl in FILLS:
        print(f"\nhedge fill = {fl}:")
        p = vt[vt["fill"] == fl].pivot_table(index=["side", "k"], columns="settle", values=["hedged_pct", "mean_pct", "t"])
        print(p.round(2).to_string())
    k_seg = 2
    for st in ("close", "bsp"):
        print(f"\nby time to the scheduled start x favourite rank (k={k_seg}, settle={st}, fill=through):")
        print(segment_table(df, k_seg, st, "through").round(2).to_string(index=False))

    # ---------------- B
    print("\n=== B. Headroom: a hindsight oracle picking back / lay / nothing at every decision ===")
    ot = oracle_table(df)
    ot.to_csv(os.path.join(a.out, "oracle.csv"), index=False)
    print(ot[ot["fill"] == a.fill].round(2).to_string(index=False))

    # ---------------- C
    print(f"\n=== C. Can the order-book state pick the good pairs? (fill={a.fill}) ===")
    tdays = sorted(train["day"].unique())
    cut = tdays[int(len(tdays) * 0.75)] if len(tdays) > 3 else tdays[-1]
    fit_part, val_part = train[train["day"] < cut], train[train["day"] >= cut]
    print(f"step 1: fit on {fit_part['race'].nunique()} train races, pick config on the last "
          f"{val_part['race'].nunique()} train races")
    lv = learnability(fit_part.reset_index(drop=True), val_part.reset_index(drop=True), a.fill)
    lv.to_csv(os.path.join(a.out, "learn_val.csv"), index=False)
    print(lv.round(3).to_string(index=False))
    elig = lv[(lv["top5_races"] >= 10) & lv["settle"].isin(["close", "bsp"])]
    if len(elig) == 0:
        elig = lv[lv["settle"].isin(["close", "bsp"])]
    best = elig.sort_values("top5_mean_pct", ascending=False).iloc[0]
    k_b, st_b = int(best["k"]), str(best["settle"])
    print(f"\npicked on validation: k={k_b} settle={st_b} (val top-5% mean {best['top5_mean_pct']:.2f}%, "
          f"t {best['top5_t']:.2f})")
    print("\nstep 2: refit on all train races, score once on HOLDOUT (every config shown, only the picked one counts):")
    lh = learnability(train, hold, a.fill)
    lh.to_csv(os.path.join(a.out, "learn_holdout.csv"), index=False)
    print(lh.round(3).to_string(index=False))
    pick = lh[(lh["k"] == k_b) & (lh["settle"] == st_b)].iloc[0]
    edge = bool(pick["top5_mean_pct"] > 0 and pick["top5_t"] > 2 and pick["top5_races"] >= 20)

    # ---------------- D
    print(f"\n=== D. $500 bank per HOLDOUT race, $10 pairs (k={k_b}, settle={st_b}, fill={a.fill}) ===")
    X_tr, X_ho = train[FEATURES].to_numpy(float), hold[FEATURES].to_numpy(float)
    res = {side: pair_pnl(hold, side, k_b, st_b, a.fill) for side in SIDES}
    preds, preds_tr = {}, {}
    for side in SIDES:
        mdl = _fit(X_tr, pair_pnl(train, side, k_b, st_b, a.fill)[0])
        preds[side], preds_tr[side] = mdl.predict(X_ho), mdl.predict(X_tr)
    thr = float(np.quantile(np.fmax(preds_tr["back"], preds_tr["lay"]), 0.95))
    score = np.fmax(preds["back"], preds["lay"])
    act_m = score >= thr
    sb_m = preds["back"] >= preds["lay"]
    gb, gl = res["back"][0], res["lay"][0]
    db, dl = res["back"][2], res["lay"][2]
    races = sorted(hold["race"].unique())
    rng = np.random.default_rng(0)
    act_r = rng.random(len(hold)) < max(act_m.mean(), 1e-3)
    sb_r = rng.random(len(hold)) < 0.5
    gb0, gl0 = np.nan_to_num(gb, nan=-9), np.nan_to_num(gl, nan=-9)
    act_o = np.fmax(gb0, gl0) > 0
    sb_o = gb0 >= gl0
    bank = pd.DataFrame([
        _bank_summary("model (top 5% predicted)", bank_sim(hold, act_m, sb_m, gb, gl, db, dl), races),
        _bank_summary("model, positive predictions", bank_sim(hold, score > 0, sb_m, gb, gl, db, dl), races),
        _bank_summary("random, same activity", bank_sim(hold, act_r, sb_r, gb, gl, db, dl), races),
        _bank_summary("always back the favourite", bank_sim(hold, (hold["rank"] == 1).to_numpy(),
                                                            np.ones(len(hold), bool), gb, gl, db, dl), races),
        _bank_summary("always lay the favourite", bank_sim(hold, (hold["rank"] == 1).to_numpy(),
                                                           np.zeros(len(hold), bool), gb, gl, db, dl), races),
        _bank_summary("ORACLE (hindsight, upper bound)", bank_sim(hold, act_o, sb_o, gb, gl, db, dl), races),
    ])
    bank.to_csv(os.path.join(a.out, "bank_sim.csv"), index=False)
    print(bank.round(2).to_string(index=False))

    # ---------------- E
    print("\n=== E. One k-tick pair vs a ladder of k=1 pairs to the same target (fill=through) ===")
    print("ladder: enter, hedge 1 tick better; each time a hedge fills, re-enter at the new best price with another")
    print("k=1 pair, until a hedge k ticks past the first entry fills. Same decisions as A; per $1 at risk on the")
    print("first leg. k=1 is a check (ladder == single). diff = ladder - single (race-clustered t).")
    lt = ladder_table(df)
    lt.to_csv(os.path.join(a.out, "ladder_vs_single.csv"), index=False)
    print(lt.round(2).to_string(index=False))
    print("\nfavourites only:")
    print(ladder_table(df[df["rank"] == 1]).round(2).to_string(index=False))

    print("\n=== Verdict ===")
    print(f"picked config k={k_b} settle={st_b} fill={a.fill}: holdout top-5% mean {pick['top5_mean_pct']:.2f}% "
          f"t {pick['top5_t']:.2f} over {int(pick['top5_races'])} races, IC back {pick['ic_back']:.3f} "
          f"lay {pick['ic_lay']:.3f}  ->  edge = {edge}")
    hedged = vt[(vt["fill"] == a.fill) & vt["settle"].isin(["close", "bsp"])]
    top = hedged.sort_values("mean_pct", ascending=False).iloc[0]
    print(f"Best unconditional hedged pair: {top['side']} k={int(top['k'])} settle={top['settle']} "
          f"{top['mean_pct']:.2f}% per pair (t {top['t']:.2f}); a paired action spec only pays if the agent can "
          "find pairs that beat that")
    print(f"\ndone in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
