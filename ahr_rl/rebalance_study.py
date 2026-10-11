"""P12: runner vs the rest of the market. Do runners lag the book's rebalancing?

A win market's implied probabilities add up to ~100% (the book). When some runners
shorten, others must drift to keep the book balanced, and vice versa. If some runners
adjust late, the rest of the market's move predicts theirs. This is a "runner vs the
market" signal, unlike the single-runner signals that failed (WoM, momentum, jumps).

Every 5s, for every runner r (p = implied probability from the mid price, B = book sum,
q_r = p_r / B, all over a look-back of W = 5 / 15 / 30s):

  sig_book  = -(change in B) x q_r
              the book's imbalance allocated to r in proportion to its share
              (B rose: someone shortened, the rest should drift)
  sig_lag   = -(net change of every OTHER runner's p) x q_r - (r's own change)
              how far r is behind its proportional share of the others' move
  sig_lag_still   sig_lag only where r itself has barely moved (|own change| < 0.1
              percentage points): the purest "hasn't caught up yet" case
  mom       r's own change (control: the old momentum signal)

Targets: r's change in implied probability over the next H = 5 / 15 / 30 / 60s (log, %;
positive = shortening), never past the off.

  A  Lead-lag: correlation of the favourite's 1s moves with the rest of the book's,
     at lags -5..+5s (who moves first?)
  B  Predictive power: within-race rank IC of each signal for each W, H; t across races,
     TRAIN and HOLDOUT days; also within the most liquid runners (P11: only those can pay)
  C  Trading it: when |signal| is in its top 5% (cut-off from TRAIN days), cross the
     spread in the signal's direction ($10, 1-step latency, walks the visible book,
     commission) and close aggressively H later; all runners or liquid runners only.
     Rules picked on TRAIN days by t, scored once on HOLDOUT; "edge" = holdout mean > 0,
     t > 2, >= 20 races.

    python -m ahr_rl.rebalance_study --tapes "data/tapes/*.npz" --out runs/rebalance --race-type flat --catalogues ...
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import race_filter
from .env import list_tapes, split_by_date
from .jump_study import _round_trip
from .ladder import PRICES
from .tape import Tape

WS = (5, 15, 30)
HS = (5, 15, 30, 60)
TRADE_H = (15, 30, 60)
SIGNALS = ("sig_book", "sig_lag", "sig_lag_still", "mom")
LAGS = range(-5, 6)
STILL = 0.1  # percentage points of implied probability
warnings.filterwarnings("ignore", message="Mean of empty slice")


# ----------------------------------------------------------------- per tape
def _prob_grid(t: Tape):
    """Implied probability per (step, runner) from the mid price, forward-filled over
    short gaps; NaN where the runner has no two-sided book."""
    T = t.n_steps
    ok = ~t.suspended.copy()
    if t.went_in_play and T > 1:
        ok[-1] = False
    end = int(np.where(ok)[0].max()) if ok.any() else 0
    bt, lt = t.back_tick[:, :, 0].astype(int), t.lay_tick[:, :, 0].astype(int)
    valid = (bt >= 0) & (lt >= 0) & t.active & ok[:, None]
    valid[end + 1:] = False
    midp = np.where(valid, np.sqrt(PRICES[np.maximum(bt, 0)] * PRICES[np.maximum(lt, 0)]), np.nan)
    p = pd.DataFrame(1.0 / midp).ffill(limit=10).to_numpy(copy=True)  # up to 5s
    p[~t.active] = np.nan
    p[end + 1:] = np.nan
    return p, valid, end, bt, lt


def lead_lag(t: Tape, p: np.ndarray, end: int) -> np.ndarray | None:
    """corr(favourite's 1s change at time x, rest-of-book's 1s change at x + lag), lags -5..5s."""
    k = max(1, int(round(1.0 / t.dt)))
    g = p[: end + 1: k]
    if len(g) < 60:
        return None
    last = np.nanmean(g[-30:], 0)
    if not np.isfinite(last).any():
        return None
    fav = int(np.nanargmax(last))
    d = np.diff(np.log(g), axis=0)
    f = d[:, fav]
    rest = np.nansum(np.delete(np.diff(g, axis=0), fav, axis=1), 1)  # net probability change of the others
    out = np.full(len(LAGS), np.nan)
    for i, L in enumerate(LAGS):
        a, b = (f[: len(f) - L], rest[L:]) if L >= 0 else (f[-L:], rest[: len(rest) + L])
        m = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
        if m.sum() > 20:
            out[i] = np.corrcoef(a[m], b[m])[0, 1]
    return out


def sample_tape(path: str, every_s: float = 5.0, stake: float = 10.0):
    t = Tape.load(path)
    if not t.went_in_play:
        return None, None
    race = os.path.basename(path).split(".npz")[0]
    p, valid, end, bt, lt = _prob_grid(t)
    dt = t.dt
    k = lambda sec: int(round(sec / dt))
    comm = t.base_rate / 100.0
    tr = t.t_rel.astype(float)
    bad = np.zeros(t.n_steps, bool)
    for s in t.removal_step:
        bad[max(0, s - k(30)): min(t.n_steps, s + k(30) + 1)] = True
    rows = []
    for s in range(k(max(WS)), end - k(5), k(every_s)):
        if bad[s] or tr[s] < -600:
            continue
        live = np.where(np.isfinite(p[s]))[0]
        if len(live) < 3:
            continue
        B = np.nansum(p[s, live])
        order = live[np.argsort(-p[s, live])]
        rank = {int(r): i + 1 for i, r in enumerate(order)}
        rec_w = {}
        for W in WS:
            s0 = s - k(W)
            both = live[np.isfinite(p[s0, live])]
            if len(both) < 3:
                rec_w[W] = None
                continue
            dpv = p[s, both] - p[s0, both]
            rec_w[W] = (dict(zip(both.tolist(), dpv.tolist())), float(dpv.sum()))
        for r in live:
            r = int(r)
            if not valid[s, r]:
                continue
            q = p[s, r] / B
            rec = dict(race=race, day=race[:8], runner=r, s=s, t=tr[s], rank=rank[r], field=len(live),
                       prob=q * 100, book=B * 100, spread=int(lt[s, r] - bt[s, r]),
                       best_usd=float(t.back_size[s, r, 0] + t.lay_size[s, r, 0]), comm=comm)
            for W in WS:
                if rec_w[W] is None or r not in rec_w[W][0]:
                    for sg in SIGNALS:
                        rec[f"{sg}_{W}"] = np.nan
                    continue
                dmap, dB = rec_w[W]
                own = dmap[r]
                others = dB - own
                rec[f"sig_book_{W}"] = -dB * q * 100
                lag = (-others * q - own) * 100
                rec[f"sig_lag_{W}"] = lag
                rec[f"sig_lag_still_{W}"] = lag if abs(own) * 100 < STILL else np.nan
                rec[f"mom_{W}"] = own * 100
            for H in HS:
                s2 = s + k(H)
                rec[f"fwd_{H}"] = (100 * np.log(p[s2, r] / p[s, r])
                                   if s2 <= end and np.isfinite(p[s2, r]) and p[s, r] > 0 else np.nan)
            for H in TRADE_H:
                s_in, s_out = s + 1, s + 1 + k(H)
                if s_out > end:
                    rec[f"back_{H}"] = rec[f"lay_{H}"] = np.nan
                    continue
                rec[f"back_{H}"] = _round_trip(t, r, s_in, s_out, "back", stake, comm)[0]
                rec[f"lay_{H}"] = _round_trip(t, r, s_in, s_out, "lay", stake, comm)[0]
            rows.append(rec)
    df = pd.DataFrame(rows) if rows else None
    if df is not None:
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
    return df, lead_lag(t, p, end)


def _one(args):
    path, kw = args
    try:
        return sample_tape(path, **kw)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}", flush=True)
        return None, None


# ----------------------------------------------------------------- stats
def race_ic(d: pd.DataFrame, sig: str, y: str, min_n: int = 8) -> dict:
    """Mean within-race Spearman IC and its t across races (vectorised)."""
    x = d[[sig, y, "race"]].dropna()
    if x.empty:
        return dict(ic=np.nan, t=np.nan, races=0)
    g = x.groupby("race")
    rx, ry = g[sig].rank(), g[y].rank()
    z = pd.DataFrame({"race": x["race"], "rx": rx, "ry": ry})
    z["rx"] -= z.groupby("race")["rx"].transform("mean")
    z["ry"] -= z.groupby("race")["ry"].transform("mean")
    agg = z.assign(xy=z.rx * z.ry, xx=z.rx ** 2, yy=z.ry ** 2).groupby("race").agg(
        n=("xy", "size"), xy=("xy", "sum"), xx=("xx", "sum"), yy=("yy", "sum"))
    agg = agg[(agg["n"] >= min_n) & (agg["xx"] > 0) & (agg["yy"] > 0)]
    ics = (agg["xy"] / np.sqrt(agg["xx"] * agg["yy"])).to_numpy()
    if len(ics) < 3:
        return dict(ic=np.nan, t=np.nan, races=len(ics))
    return dict(ic=float(ics.mean()), t=float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), races=len(ics))


def race_t(v, race) -> dict:
    x = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna()
    if len(x) < 2:
        return dict(n=len(x), races=x["race"].nunique(), mean=np.nan, t=np.nan)
    per = x.groupby("race")["v"].mean()
    sd = per.std(ddof=1)
    return dict(n=len(x), races=len(per), mean=float(x["v"].mean()),
                t=float(per.mean() / (sd / np.sqrt(len(per)))) if len(per) > 1 and sd > 0 else np.nan)


def trade_rules(df: pd.DataFrame, hold: set, liq_cut: float) -> pd.DataFrame:
    tr = ~df["day"].isin(hold)
    rows = []
    for sg in SIGNALS:
        for W in WS:
            col = f"{sg}_{W}"
            if col not in df or df.loc[tr, col].notna().sum() < 50:
                continue
            cut = float(np.nanquantile(df.loc[tr, col].abs(), 0.95))
            sel = df[col].abs() >= cut
            for H in TRADE_H:
                pnl = np.where(df[col] > 0, df[f"back_{H}"], df[f"lay_{H}"]) * 100  # signal > 0: expect shortening
                for gate in ("all", "liquid"):
                    m = sel & np.isfinite(pnl)
                    if gate == "liquid":
                        m &= df["best_usd"] >= liq_cut
                    rec = dict(signal=sg, W=W, H=H, runners=gate, cut=cut)
                    for sp, mm in (("train", m & tr), ("holdout", m & ~tr)):
                        rt = race_t(pnl[mm.to_numpy()], df.loc[mm, "race"])
                        rec.update({f"{sp}_n": rt["n"], f"{sp}_races": rt["races"], f"{sp}_mean_%": rt["mean"],
                                    f"{sp}_t": rt["t"]})
                    rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-s", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--min-races", type=int, default=30)
    race_filter.add_args(ap)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    paths = list_tapes(a.tapes)
    _, va, te = split_by_date(paths)
    hold = {os.path.basename(p)[:8] for p in va + te}  # same days as the unfiltered studies
    paths = race_filter.filter_paths(paths, a)
    cache = os.path.join(a.out, f"samples{race_filter.tag(a)}.parquet")
    ll_path = os.path.join(a.out, f"lead_lag{race_filter.tag(a)}.npy")
    if os.path.exists(cache) and os.path.exists(ll_path):
        df, LL = pd.read_parquet(cache), np.load(ll_path)
        print(f"loaded {cache}")
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        parts, lls = [], []
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for i, (d, ll) in enumerate(pool.map(_one, [(p, dict(every_s=a.every_s)) for p in paths], chunksize=2)):
                if d is not None:
                    parts.append(d)
                if ll is not None:
                    lls.append(ll)
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(paths)} races, {time.time() - t0:.0f}s", flush=True)
        if not parts:
            raise SystemExit("no samples")
        df = pd.concat(parts, ignore_index=True)
        LL = np.vstack(lls) if lls else np.zeros((0, len(LAGS)))
        df.to_parquet(cache, index=False)
        np.save(ll_path, LL)
    tr = ~df["day"].isin(hold)
    liq_cut = float(np.nanquantile(df.loc[tr, "best_usd"], 0.8))
    print(f"\n{len(df)} runner samples, {df['race'].nunique()} races (train {df.loc[tr, 'race'].nunique()}, "
          f"holdout {df.loc[~tr, 'race'].nunique()}); liquid runners = >= ${liq_cut:.0f} at the best prices "
          f"(train 80th pct)  [{time.time() - t0:.0f}s]")

    # ---------------- A
    print("\n=== A. Who moves first? corr(favourite's 1s move, rest of the book's 1s move `lag` s later) ===")
    print("(negative at lag 0 = rebalancing; a stronger value at positive lags = the favourite leads, negative "
          "lags = the rest leads)")
    m = np.nanmean(LL, 0) if len(LL) else np.full(len(LAGS), np.nan)
    n = np.sum(np.isfinite(LL), 0) if len(LL) else np.zeros(len(LAGS))
    sd = np.nanstd(LL, 0) if len(LL) else np.full(len(LAGS), np.nan)
    lagtab = pd.DataFrame(dict(lag_s=list(LAGS), corr=m, t=m / (sd / np.sqrt(np.maximum(n, 1))), races=n))
    lagtab.to_csv(os.path.join(a.out, "A_lead_lag.csv"), index=False)
    print(lagtab.round(3).to_string(index=False))

    # ---------------- B
    print("\n=== B. Does the rest of the market predict the runner? within-race rank IC, t across races ===")
    rows = []
    for sg in SIGNALS:
        for W in WS:
            col = f"{sg}_{W}"
            for H in HS:
                rec = dict(signal=sg, W=W, H=H)
                for sp, d in (("train", df[tr]), ("holdout", df[~tr]), ("liquid", df[df["best_usd"] >= liq_cut])):
                    r = race_ic(d, col, f"fwd_{H}")
                    rec.update({f"{sp}_ic": r["ic"], f"{sp}_t": r["t"]})
                rows.append(rec)
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(a.out, "B_ic.csv"), index=False)
    print(B.round(3).to_string(index=False))

    # ---------------- C
    print("\n=== C. Trading it: top 5% |signal|, cross the spread in its direction, close H later ($10, per trade %) ===")
    R = trade_rules(df, hold, liq_cut)
    R.to_csv(os.path.join(a.out, "C_rules.csv"), index=False)
    R = R[R["train_races"] >= a.min_races].sort_values("train_t", ascending=False)
    cols = ["signal", "W", "H", "runners", "train_n", "train_races", "train_mean_%", "train_t", "holdout_n",
            "holdout_races", "holdout_mean_%", "holdout_t"]
    print(R[cols].head(15).round(3).to_string(index=False))
    best = R.iloc[0].to_dict() if len(R) else {}
    edge = bool(best and best["holdout_mean_%"] > 0 and (best["holdout_t"] or 0) > 2 and best["holdout_races"] >= 20)
    best_ic = B.sort_values("train_t", ascending=False).iloc[0].to_dict() if len(B) else {}
    verdict = dict(best_rule={k: best.get(k) for k in cols}, edge=edge,
                   best_ic={k: best_ic.get(k) for k in ("signal", "W", "H", "train_ic", "train_t", "holdout_ic",
                                                         "holdout_t", "liquid_ic", "liquid_t")})
    json.dump(verdict, open(os.path.join(a.out, f"verdict{race_filter.tag(a)}.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    print("EDGE: the rest of the market predicts runners well enough to trade -> forward-test it like P3c" if edge else
          "NO EDGE after costs (see B for whether there is information at all)")
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
