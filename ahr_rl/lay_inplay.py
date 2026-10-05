"""Lay (or back) the n-th favourite pre-off, hedge in-play: does it make money?

Both directions are tested:
  side=lay   lay pre-off, resting BACK at lay price x k (fills if the runner trades UP)
  side=back  back pre-off, resting LAY at back price / k (fills if the runner trades
             DOWN: winners almost always do in-running, and so do losers that lead and fade)

The idea: lay a fancied runner before the off, and leave a resting BACK order at a
higher price that persists into the race (Betfair "keep" bets). If the runner trades
up to that price in-running (most losers, and many winners at some point, do), the
back matches and the position is green on every outcome. If it never trades that
high (usually because it is winning), the lay is exposed.

No hindsight: the hedge price is fixed when the lay is placed (lay price x k).

  entry    lay $1 of stake at the best available lay price for the n-th favourite
           (by price at that moment) at T-10m / T-5m / T-1m / the last pre-off
           snapshot. Needs >= $5 shown at that price.
  hedge    resting back at the first tick >= lay price x k, k = 1.1, 1.25, 1.5, 2, 3,
           live from the moment of the lay through the race. Fill rule:
             touch:   any trade at or above the hedge price (optimistic: no queue)
             through: a trade at least one tick ABOVE the hedge price (conservative)
  P&L      per $1 of lay stake, after commission (market base rate) on a winning
           market result:
             hedged:   lay x at L, back x*L/B at B  ->  green  x * (1 - L/B)
             unhedged: runner loses -> +x, runner wins -> -x * (L - 1)
  back side: back $1 at the best back price P, resting lay at the first tick <= P / k
             (through = one tick lower). hedged: green (P/B - 1); unhedged: win -> P - 1,
             lose -> -1. P&L per $1 backed.
  also     "no hedge": the plain bet held to the result (is the bet itself value?).

Per-race results are averaged with t-stats across races; rules (with >= --min-races
train races) are picked on TRAIN days and scored once on HOLDOUT days; "edge" needs
holdout mean > 0, t > 2 and >= 20 holdout races, a conservative ('through') or no-hedge
rule, and a positive tail-stressed mean. A winner that never trades up costs (lay
price - 1) and can be rare enough to be missing from a sample entirely, which makes a
nearly-always-hedged rule look riskless (huge t). The stressed mean replaces the
observed rate of that worst outcome by its 95% upper bound (Clopper-Pearson, all days),
so a rule only passes if it still pays when the rare blow-up happens as often as the
data can't rule out.

    python -m ahr_rl.lay_inplay --recordings "<drive>/betfair stream data/recordings" --out runs/lay_inplay
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .env import split_by_date
from .ladder import PRICES, price_to_tick
from .stream import MarketCache, read_recording

CHECKPOINTS = {"T-10m": -600.0, "T-5m": -300.0, "T-1m": -60.0, "last": None}
KS = (1.1, 1.25, 1.5, 2.0, 3.0)
MAX_RANK = 6


def race_rows(path: str, max_rank: int = MAX_RANK) -> list[dict]:
    rec = read_recording(path)
    tr = rec.trailer or {}
    winners = tr.get("winners") or []
    if not rec.messages or len(winners) != 1:
        return []
    winner = int(winners[0])
    start = rec.market_start_ms
    cache = MarketCache(rec.market_id)
    snaps = {}       # checkpoint -> {sid: (best lay price, size shown, mid)}
    pending = dict(CHECKPOINTS)
    last_pre = None
    went_ip = False
    ip_max, ip_min = {}, {}          # in-play traded price range per runner
    pre_max_after = {}               # checkpoint -> {sid: max traded price after that checkpoint, pre-off}
    pre_min_after = {}               # same, min
    commission = None

    def snapshot():
        out = {}
        for r in cache.active_runners():
            bl, ls = r.best_lay()
            bb, bs = r.best_back()
            if bl > 0 and bb > 0:
                out[r.selection_id] = (bl, ls, (bl * bb) ** 0.5, bb, bs)
        return out

    for m in rec.messages:
        pt = m["pt"]
        rel = (pt - start) / 1000.0
        cache.apply_mcm(m)
        commission = cache.market_base_rate or commission
        trades = cache.drain_trades()
        if not went_ip and not cache.in_play and cache.status == "OPEN":
            for name, sec in list(pending.items()):
                if sec is not None and rel >= sec:
                    snap = snapshot()
                    if len(snap) >= 2:
                        snaps[name] = snap
                        pre_max_after[name] = {}
                        pre_min_after[name] = {}
                        del pending[name]
        if cache.in_play:
            went_ip = True
            for sid, price, vol in trades:
                ip_max[sid] = max(ip_max.get(sid, 0.0), price)
                ip_min[sid] = min(ip_min.get(sid, 1e9), price)
        else:
            for name in pre_max_after:
                d, e = pre_max_after[name], pre_min_after[name]
                for sid, price, vol in trades:
                    d[sid] = max(d.get(sid, 0.0), price)
                    e[sid] = min(e.get(sid, 1e9), price)
            if cache.status == "OPEN":
                last_pre = snapshot()
                if "last" in pre_max_after:
                    pre_max_after["last"] = {}
    if not went_ip or last_pre is None:
        return []
    snaps["last"] = last_pre
    pre_max_after.setdefault("last", {})
    pre_min_after.setdefault("last", {})
    comm = (commission or 8.0) / 100.0
    race = os.path.basename(path).split(".ndjson")[0]
    rows = []
    for cp, snap in snaps.items():
        ranked = sorted(snap.items(), key=lambda kv: kv[1][2])
        for rank, (sid, (L, size, mid, Pb, bsize)) in enumerate(ranked[:max_rank], 1):
            won = sid == winner
            # price range traded after entry: rest of the pre-off period, then in-running
            hi = max(pre_max_after.get(cp, {}).get(sid, 0.0), ip_max.get(sid, 0.0))
            lo = min(pre_min_after.get(cp, {}).get(sid, 1e9), ip_min.get(sid, 1e9))
            base = dict(race=race, day=race[:8], checkpoint=cp, rank=rank, won=won, comm=comm,
                        ip_max=ip_max.get(sid, np.nan), ip_min=ip_min.get(sid, np.nan), hi_after=hi, lo_after=lo)
            # ---- lay first, hedge with a back that fills if the price goes UP
            if size >= 5.0 and L <= 50:
                rec_ = dict(base, side="lay", price=L)
                unhedged = (1.0 * (1 - comm)) if not won else -(L - 1)
                rec_["no_hedge"] = unhedged
                for k in KS:
                    bt = price_to_tick(L * k)
                    if PRICES[bt] < L * k:
                        bt += 1
                    B = float(PRICES[min(bt, len(PRICES) - 1)])
                    B_through = float(PRICES[min(bt + 1, len(PRICES) - 1)])
                    green = (1 - L / B)
                    green = green * (1 - comm) if green > 0 else green
                    for mode, need in (("touch", B), ("through", B_through)):
                        hit = hi >= need - 1e-9
                        rec_[f"k{k}_{mode}_hit"] = hit
                        rec_[f"k{k}_{mode}"] = green if hit else unhedged
                rows.append(rec_)
            # ---- back first, hedge with a lay that fills if the price goes DOWN
            if bsize >= 5.0 and Pb <= 50:
                rec_ = dict(base, side="back", price=Pb)
                unhedged = ((Pb - 1) * (1 - comm)) if won else -1.0
                rec_["no_hedge"] = unhedged
                for k in KS:
                    bt = price_to_tick(Pb / k)
                    if PRICES[bt] > Pb / k:
                        bt -= 1
                    bt = max(bt, 0)
                    B = float(PRICES[bt])
                    B_through = float(PRICES[max(bt - 1, 0)])
                    green = Pb / B - 1
                    green = green * (1 - comm) if green > 0 else green
                    for mode, need in (("touch", B), ("through", B_through)):
                        hit = lo <= need + 1e-9 and B < Pb
                        rec_[f"k{k}_{mode}_hit"] = hit
                        rec_[f"k{k}_{mode}"] = green if hit else unhedged
                rows.append(rec_)
    return rows


def _one(p):
    try:
        return race_rows(p)
    except Exception as e:
        print(f"  skip {os.path.basename(p)}: {e}", flush=True)
        return []


def _race_t(v, race):
    x = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna()
    if len(x) < 2:
        return dict(n=len(x), mean=np.nan, t=np.nan, races=x["race"].nunique())
    per = x.groupby("race")["v"].mean()
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 and per.std(ddof=1) > 0 else np.nan
    return dict(n=len(x), mean=float(x["v"].mean()), t=float(t), races=len(per))


def _stress(g: pd.DataFrame, col: str, side: str) -> dict:
    """Worst outcome of the rule (lay: an unhedged winner, -(L-1); back: an unhedged
    loser, -1), its observed rate on all days, its 95% upper bound, and the mean P&L
    with the bound in place of the observed rate."""
    from scipy.stats import beta

    hit = g[col + "_hit"].to_numpy(bool) if col != "no_hedge" else np.zeros(len(g), bool)
    won = g["won"].to_numpy(bool)
    tail = (~hit & won) if side == "lay" else (~hit & ~won)
    loss = -(g["price"].to_numpy(float) - 1) if side == "lay" else -np.ones(len(g))
    n, x = len(g), int(tail.sum())
    if n == 0:
        return dict(tail_events=0, **{"tail_rate_%": np.nan, "tail_rate_95_%": np.nan}, stressed_mean=np.nan)
    p_up = float(beta.ppf(0.95, x + 1, n - x)) if x < n else 1.0
    v = g[col].to_numpy(float)
    rest = v[~tail].mean() if (~tail).any() else 0.0
    stressed = (1 - p_up) * rest + p_up * loss.mean()
    return {"tail_events": x, "tail_rate_%": x / n * 100, "tail_rate_95_%": p_up * 100, "stressed_mean": stressed}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", required=True, help="folder of *.ndjson.gz recordings (they include in-play)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-races", type=int, default=100000)
    ap.add_argument("--min-races", type=int, default=50, help="a rule needs this many train races to be ranked")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    t0 = time.time()
    paths = sorted(glob.glob(os.path.join(a.recordings, "**", "*.ndjson.gz"), recursive=True))[: a.max_races]
    tr, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    rows = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, res in enumerate(pool.map(_one, paths, chunksize=4)):
            rows.extend(res)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(paths)} recordings ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no usable races (need in-play data and a single known winner)")
    df["split"] = np.where(df["day"].isin(hold_days), "holdout", "train")
    df.to_parquet(os.path.join(a.out, "lays.parquet"), index=False)
    print(f"{df['race'].nunique()} races with in-play data and a winner\n")

    for side, word in (("lay", "UP (x k)"), ("back", "DOWN (/ k)")):
        print("=" * 100 + f"\n{side.upper()} FIRST: how often does the n-th favourite (at T-10m) trade {word} after "
              "entry? (all days, 'through' fills)\n" + "=" * 100)
        d10 = df[(df["checkpoint"] == "T-10m") & (df["side"] == side)]
        rows_h = []
        for rank, g in d10.groupby("rank"):
            rec = dict(rank=rank, races=len(g), win_rate=g["won"].mean() * 100, avg_price=g["price"].mean(),
                       implied_win=(1 / g["price"]).mean() * 100, no_hedge=g["no_hedge"].mean())
            for k in KS:
                rec[f"hedged k{k}"] = g[f"k{k}_through_hit"].mean() * 100
            rec["winners hedged k1.25"] = g.loc[g["won"], "k1.25_through_hit"].mean() * 100 if g["won"].any() else np.nan
            rec["losers hedged k1.25"] = g.loc[~g["won"], "k1.25_through_hit"].mean() * 100
            rows_h.append(rec)
        print(pd.DataFrame(rows_h).round(2).to_string(index=False))
        print()

    print("\n" + "=" * 100 + "\nP&L PER $1 STAKED (after commission), by side, favourite rank, entry time and hedge "
          "price; t across races\n" + "=" * 100)
    res = []
    for (side, cp, rank), g in df.groupby(["side", "checkpoint", "rank"]):
        for col in ["no_hedge"] + [f"k{k}_{m}" for k in KS for m in ("touch", "through")]:
            rec = dict(side=side, checkpoint=cp, rank=rank, rule=col)
            for sp, gg in g.groupby("split"):
                rt = _race_t(gg[col], gg["race"])
                rec.update({f"{sp}_n": rt["n"], f"{sp}_mean": rt["mean"], f"{sp}_t": rt["t"]})
            if col != "no_hedge":
                rec["hedged_%"] = g[col + "_hit"].mean() * 100
            rec.update(_stress(g, col, side))
            res.append(rec)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(a.out, "rules.csv"), index=False)
    piv = R[R["rule"].str.endswith("through") | (R["rule"] == "no_hedge")].copy()
    piv["all_mean"] = (piv["train_mean"] * piv["train_n"] + piv["holdout_mean"].fillna(0) * piv["holdout_n"].fillna(0)) / \
        (piv["train_n"] + piv["holdout_n"].fillna(0))
    for side in ("lay", "back"):
        for cp in CHECKPOINTS:
            t = piv[(piv["checkpoint"] == cp) & (piv["side"] == side)].pivot(index="rank", columns="rule",
                                                                              values="all_mean")
            if t.empty:
                continue
            t = t[["no_hedge"] + [f"k{k}_through" for k in KS]]
            print(f"\n--- {side} at {cp}: mean P&L per $1, all days (conservative 'through' fills) ---")
            print(t.round(4).to_string())

    ok_rule = R["rule"].str.endswith("through") | (R["rule"] == "no_hedge")
    best = R[(R["train_n"] >= a.min_races) & ok_rule].sort_values("train_t", ascending=False)
    if best.empty:
        raise SystemExit(f"no rule has >= {a.min_races} train races (lower --min-races)")
    print("\n" + "=" * 100 + "\nTOP 10 RULES ON TRAIN DAYS ('through' / no-hedge only), with HOLDOUT and the "
          "tail stress test\n" + "=" * 100)
    cols = ["side", "checkpoint", "rank", "rule", "hedged_%", "train_n", "train_mean", "train_t", "holdout_n", "holdout_mean",
            "holdout_t", "tail_events", "tail_rate_%", "tail_rate_95_%", "stressed_mean"]
    print(best[cols].head(10).round(4).to_string(index=False))
    b = best.iloc[0].to_dict()
    verdict = dict(best={k: b.get(k) for k in cols},
                   edge=bool((b.get("holdout_mean") or 0) > 0 and (b.get("holdout_t") or 0) > 2
                             and (b.get("holdout_n") or 0) >= 20 and (b.get("stressed_mean") or -1) > 0))
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=str)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=str))
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
