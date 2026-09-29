"""Lead-lag study: does a price from OUTSIDE the exchange order book lead the
exchange, and can you trade the gap?

Bookmaker and tote prices often move before Betfair in Australian racing. No such
feed is recorded yet, so this study is written for any outside feed and, until
one exists, runs on the one outside-the-book signal the recordings do contain:
Betfair's projected BSP (``spn``). It includes Betfair SP bets, money that is
never shown in the order book.

External feed (optional, ``--external file.csv[,file2.csv]``), one row per price
update:
    market_id,selection_id,price,<time>
where <time> is ``secs_to_start`` (seconds relative to the scheduled start,
negative before it) or ``ts`` (epoch ms / ISO time; then ``--catalogues`` is
needed for each market's scheduled start). The feed is joined to the 0.5s tape
causally: at each row the latest update at or before that time is used.

Per runner, every ``--every-s`` seconds (tradeable runners only: spread <= 3
ticks, price <= 30):
  gap_%     100 * log(outside price / exchange mid): + = outside says longer
            (less likely) than the exchange
  d_out_30  change of the outside price over the last 30s (%), d_mid_30 the same
            for the exchange
  fwd_H     exchange mid move over H (%), H = 10s, 30s, 60s, 120s, to the start

  A  Lead-lag: IC of the gap and of the outside price's recent change with the
     exchange's next move; and the reverse (does the exchange lead the outside
     price?). If the outside price leads, gap -> fwd IC is positive and larger
     than fwd -> outside.
  B  Trading the gap: when |gap| is in the top X% (thresholds from TRAIN days),
     trade the exchange toward the outside price with real aggressive round trips
     (latency, visible depth, commission; same model as the jump study). Rules
     picked on TRAIN days, scored once on HOLDOUT days.

    python -m ahr_rl.leadlag_study --tapes "data/tapes/*.npz" --out runs/leadlag
    python -m ahr_rl.leadlag_study --tapes ... --out ... --external tab.csv --catalogues <drive>/catalogues
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
from .jump_study import _round_trip
from .ladder import PRICES
from .tape import Tape

H_LIST = ("10", "30", "60", "120", "start")
TRADE_H = ("30", "120", "start")


def load_external(paths: list[str], cat_dir: str | None) -> dict:
    """-> {market_id: DataFrame[selection_id, t_rel, price]} (t_rel = seconds vs scheduled start)."""
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["market_id"] = df["market_id"].astype(str)
    if "secs_to_start" in df:
        df["t_rel"] = df["secs_to_start"].astype(float)
    else:
        from .catalogue import index_catalogues, load_catalogue

        if not cat_dir:
            raise SystemExit("external feed has 'ts' but no --catalogues to find each market's scheduled start")
        idx = index_catalogues(cat_dir)
        starts = {}
        for mid in df["market_id"].unique():
            if mid in idx:
                st = load_catalogue(idx[mid]).get("marketStartTime")
                if st:
                    starts[mid] = pd.Timestamp(st).value / 1e6
        ts = df["ts"]
        ms = pd.to_numeric(ts, errors="coerce")
        ms = ms.where(ms.notna(), pd.to_datetime(ts, utc=True, errors="coerce").astype("int64") / 1e6)
        df["t_rel"] = (ms - df["market_id"].map(starts)) / 1000.0
        df = df.dropna(subset=["t_rel"])
    df = df[df["price"] > 1.0]
    return {m: g[["selection_id", "t_rel", "price"]].sort_values("t_rel") for m, g in df.groupby("market_id")}


def sample_tape(path: str, external: dict | None = None, every_s: float = 10.0, stake: float = 10.0,
                max_spread: int = 3, max_price: float = 30.0) -> list[dict]:
    t = Tape.load(path)
    race = os.path.basename(path).split(".npz")[0]
    dt, T, R = t.dt, t.n_steps, t.n_runners
    ok = ~t.suspended.copy()
    if t.went_in_play and T > 1:
        ok[-1] = False
    if not ok.any():
        return []
    end = int(np.where(ok)[0].max())
    comm = t.base_rate / 100.0
    tr = t.t_rel.astype(float)
    after0 = np.where(tr >= 0)[0]
    s_start = int(after0[0]) if len(after0) else end
    k = lambda sec: int(round(sec / dt))

    bt, lt = t.back_tick[:, :, 0].astype(int), t.lay_tick[:, :, 0].astype(int)
    valid = (bt >= 0) & (lt >= 0) & t.active & ok[:, None]
    valid[end + 1:] = False
    midp = np.where(valid, np.sqrt(PRICES[np.maximum(bt, 0)] * PRICES[np.maximum(lt, 0)]), np.nan)

    # outside price on the tape grid, causal (last update at or before each row)
    if external is None:
        out = np.where(t.spn > 1.0, t.spn, np.nan).astype(float)
        src = "projected_bsp"
    else:
        g = external.get(t.market_id)
        if g is None:
            return []
        out = np.full((T, R), np.nan)
        sid_to_r = {int(s): i for i, s in enumerate(t.selection_ids)}
        for sid, gg in g.groupby("selection_id"):
            r = sid_to_r.get(int(sid))
            if r is None:
                continue
            idx = np.searchsorted(gg["t_rel"].values, tr, side="right") - 1
            vals = gg["price"].values
            out[:, r] = np.where(idx >= 0, vals[np.clip(idx, 0, len(vals) - 1)], np.nan)
        src = "external"
    rows = []
    for s in range(k(30), end, k(every_s)):
        for r in range(R):
            if not valid[s, r] or not np.isfinite(out[s, r]) or lt[s, r] - bt[s, r] > max_spread:
                continue
            if midp[s, r] > max_price:
                continue
            rec = dict(race=race, day=race[:8], runner=r, t=tr[s], src=src, price=midp[s, r],
                       gap=100 * np.log(out[s, r] / midp[s, r]),
                       d_out_30=100 * np.log(out[s, r] / out[s - k(30), r]) if np.isfinite(out[s - k(30), r]) else np.nan,
                       d_mid_30=100 * np.log(midp[s, r] / midp[s - k(30), r]) if np.isfinite(midp[s - k(30), r]) else np.nan)
            for h in H_LIST:
                if h == "start":
                    s2 = s_start if s < s_start else None
                else:
                    s2 = min(s + k(int(h)), end)
                if s2 is None or s2 <= s:
                    rec[f"fwd_{h}"] = rec[f"out_fwd_{h}"] = np.nan
                else:
                    ix = np.where(np.isfinite(midp[s:s2 + 1, r]))[0]
                    rec[f"fwd_{h}"] = 100 * np.log(midp[s + ix[-1], r] / midp[s, r]) if len(ix) else np.nan
                    rec[f"out_fwd_{h}"] = 100 * np.log(out[s2, r] / out[s, r]) if np.isfinite(out[s2, r]) else np.nan
                if h in TRADE_H:
                    if s2 is None or min(s2 + 1, end) <= min(s + 1, end):
                        rec[f"back_{h}"] = rec[f"lay_{h}"] = np.nan
                    else:
                        rec[f"back_{h}"] = _round_trip(t, r, min(s + 1, end), min(s2 + 1, end), "back", stake, comm)[0]
                        rec[f"lay_{h}"] = _round_trip(t, r, min(s + 1, end), min(s2 + 1, end), "lay", stake, comm)[0]
            rows.append(rec)
    return rows


def _one(args):
    path, kw = args
    try:
        return sample_tape(path, **kw)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}", flush=True)
        return []


def ic(df: pd.DataFrame, f: str, y: str) -> dict:
    """Within-race rank IC, t clustered by race."""
    x = df[["race", f, y]].dropna()
    if x["race"].nunique() < 10:
        return dict(ic=np.nan, t=np.nan, n=len(x))
    g = x.groupby("race")
    a = (g[f].rank(pct=True) - 0.5).values
    b = (g[y].rank(pct=True) - 0.5).values
    r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
    per = pd.Series(a * b).groupby(x["race"].values).mean()
    t = float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)) + 1e-12))
    return dict(ic=r, t=t, n=len(x))


def race_t(v, race) -> dict:
    x = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna()
    if len(x) < 2:
        return dict(n=len(x), races=0, mean=np.nan, t=np.nan)
    per = x.groupby("race")["v"].mean()
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 and per.std(ddof=1) > 0 else np.nan
    return dict(n=len(x), races=len(per), mean=float(x["v"].mean()), t=float(t))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--external", default="", help="comma-separated CSVs of an outside price feed (see module doc)")
    ap.add_argument("--catalogues", default="")
    ap.add_argument("--every-s", type=float, default=10.0)
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--min-races", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-races", type=int, default=100000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    t0 = time.time()
    paths = list_tapes(a.tapes)[: a.max_races]
    tr, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    external = load_external(a.external.split(","), a.catalogues) if a.external else None
    source = "external feed" if external else "Betfair projected BSP (no external feed given)"
    print(f"{len(paths)} races; outside price = {source}", flush=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    rows = []
    kw = dict(external=external, every_s=a.every_s, stake=a.stake)
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, res in enumerate(pool.map(_one, [(p, kw) for p in paths], chunksize=4)):
            rows.extend(res)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(paths)} races, {len(rows)} samples, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no samples: does the outside feed cover these markets?")
    df["split"] = np.where(df["day"].isin(hold_days), "holdout", "train")
    df["phase"] = pd.cut(df["t"], [-1e9, -120, 0, 1e9], labels=["<T-2m", "T-2m..start", "after start"], right=False)
    df.to_parquet(os.path.join(a.out, "samples.parquet"), index=False)
    print(f"{len(df)} samples from {df['race'].nunique()} races\n")
    print("gap (%) distribution:", df["gap"].describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_dict())

    # ---------------- A: lead-lag
    print("\n" + "=" * 100 + "\nA. LEAD-LAG (within-race rank IC, t clustered by race)\n" + "=" * 100)
    rows_a = []
    for h in H_LIST:
        rows_a.append(dict(horizon=h,
                           gap_to_exch=ic(df, "gap", f"fwd_{h}")["ic"], gap_to_exch_t=ic(df, "gap", f"fwd_{h}")["t"],
                           outchg_to_exch=ic(df, "d_out_30", f"fwd_{h}")["ic"],
                           exchchg_to_out=ic(df, "d_mid_30", f"out_fwd_{h}")["ic"],
                           exch_momentum=ic(df, "d_mid_30", f"fwd_{h}")["ic"]))
    A = pd.DataFrame(rows_a)
    A.to_csv(os.path.join(a.out, "A_leadlag.csv"), index=False)
    print("gap_to_exch    : outside-vs-exchange gap -> next exchange move (+ = exchange moves toward the outside price)")
    print("outchg_to_exch : outside price's last-30s change -> next exchange move (+ = outside leads)")
    print("exchchg_to_out : exchange's last-30s change -> next outside move (+ = exchange leads)")
    print(A.round(3).to_string(index=False))
    print("\nby phase (gap -> next 30s / 120s):")
    for ph, g in df.groupby("phase", observed=True):
        print(f"  {ph:12s} 30s IC {ic(g, 'gap', 'fwd_30')['ic']:+.3f}   120s IC {ic(g, 'gap', 'fwd_120')['ic']:+.3f}   "
              f"n={len(g)}")

    # ---------------- B: trading the gap
    print("\n" + "=" * 100 + "\nB. TRADE THE EXCHANGE TOWARD THE OUTSIDE PRICE when |gap| is large "
          "(real round trips, $10; TRAIN-picked, HOLDOUT-scored)\n" + "=" * 100)
    train, hold = df[df["split"] == "train"], df[df["split"] == "holdout"]
    res = []
    for q in (0.5, 0.8, 0.9, 0.95, 0.99):
        thr = train["gap"].abs().quantile(q)
        for phase in ("all", "<T-2m", "T-2m..start", "after start"):
            for h in TRADE_H:
                rec = dict(top_pct=round((1 - q) * 100, 1), gap_thr=thr, phase=phase, horizon=h)
                for name, g in (("train", train), ("holdout", hold)):
                    if phase != "all":
                        g = g[g["phase"] == phase]
                    g = g[g["gap"].abs() >= thr]
                    # outside longer (gap > 0) -> exchange should drift -> lay first; else back first
                    pnl = np.where(g["gap"] > 0, g[f"lay_{h}"], g[f"back_{h}"])
                    rt = race_t(pnl, g["race"])
                    rec.update({f"{name}_trades": rt["n"], f"{name}_races": rt["races"], f"{name}_mean": rt["mean"],
                                f"{name}_t": rt["t"]})
                res.append(rec)
    B = pd.DataFrame(res)
    B = B[B["train_races"] >= a.min_races].sort_values("train_t", ascending=False)
    B.to_csv(os.path.join(a.out, "B_rules.csv"), index=False)
    print(B.head(15).round(4).to_string(index=False))
    best = B.iloc[0].to_dict() if len(B) else {}
    verdict = dict(source=source, samples=len(df),
                   gap_ic_30=float(A.loc[A["horizon"] == "30", "gap_to_exch"].iloc[0]),
                   outside_leads=bool(A.loc[A["horizon"] == "30", "outchg_to_exch"].iloc[0] >
                                      A.loc[A["horizon"] == "30", "exchchg_to_out"].iloc[0]),
                   best_rule=best,
                   edge=bool(best and best["holdout_mean"] > 0 and (best["holdout_t"] or 0) > 2))
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
