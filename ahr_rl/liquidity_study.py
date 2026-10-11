"""P11: does liquidity make markets, or individual runners, tradeable, and does it rescue an edge?

The race study (2h) found deep races cost less and move more relative to cost, but the
best signal (weight of money) was no more predictive there. This asks the same at the
level of the individual runner, and whether the candidate strategies do better where
liquidity is high. Reuses earlier outputs, nothing is rebuilt:

  --samples   samples.parquet from the market study (cell 2g): every runner every 10s
  --pairs     pairs.parquet from P6
  --attempts  attempts.parquet from P3c (the frozen drift fade, live top 3)
  --races-csv races.csv from the race study (cell 2h): race type, total matched

Liquidity measures:
  runner (live)  best_$  money at the runner's best back + lay prices (2g, P3c),
                 depth3  top-3 depth both sides (P6)
  market (live)  matched so far at the event (P3c, once P3c has been re-run)
  race (look-ahead, descriptive only)  total matched at the off (2h)
Quintile cut-offs come from TRAIN days only.

  A  Runner tradeability by runner liquidity: cost, typical 2-min move / cost, random
     and perfect-hindsight 2-min round trips, and the 30s predictive power (rank IC) of
     WoM and 30s momentum. All races and thoroughbreds. Plus race x runner liquidity.
  B  P6's model (boosted trees predicting pair P&L, trained on TRAIN days) scored on
     HOLDOUT days within each liquidity quintile: IC and the P&L of its top 5% picks.
  C  P3c's drift fade on the days before the forward test, by liquidity quintile.

Any slice that looks good here is a hypothesis, not a result: it was found by slicing
(5 quintiles x several measures), so it would need its own forward test (like P3c)
before it means anything.

    python -m ahr_rl.liquidity_study --samples runs/market/samples.parquet --pairs runs/pair_study/pairs.parquet \\
        --attempts runs/p3b_forward/attempts.parquet --races-csv runs/races/races.csv --out runs/liquidity
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from .env import split_by_date
from .pair_study import FEATURES, _fit, pair_pnl, race_t

Q_LABELS = ["Q1 thin", "Q2", "Q3", "Q4", "Q5 deep"]
warnings.filterwarnings("ignore", message="Mean of empty slice")


def hold_days_of(races) -> set[str]:
    _, va, te = split_by_date([f"{r}.x" for r in sorted(set(races))])
    return {r[:8] for r in va + te}


def quintiles(x: pd.Series, train_mask) -> pd.Series:
    """Quintiles of x with cut-offs from TRAIN rows only."""
    edges = np.nanquantile(x[train_mask], [0.2, 0.4, 0.6, 0.8])
    edges = np.unique(edges)
    bins = np.r_[-np.inf, edges, np.inf]
    labels = Q_LABELS if len(bins) == 6 else [f"Q{i + 1}" for i in range(len(bins) - 1)]
    return pd.cut(x, bins, labels=labels)


def spearman(a, b) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()))


def add_race_info(df: pd.DataFrame, races_csv: str | None) -> pd.DataFrame:
    if races_csv and os.path.exists(races_csv):
        r = pd.read_csv(races_csv, usecols=lambda c: c in ("race", "race_type", "matched_$"))
        df = df.merge(r, on="race", how="left")
    if "race_type" not in df:
        df["race_type"] = "Unknown"
    if "matched_$" not in df:
        df["matched_$"] = np.nan
    df["race_type"] = df["race_type"].fillna("Unknown")
    return df


# ----------------------------------------------------------------- A
def tradeability(s: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for q, g in s.groupby(by, observed=True):
        bk, ly = g["back_120"].to_numpy(float), g["lay_120"].to_numpy(float)
        cost = g["cost10"].median()
        rows.append({by: q, "samples": len(g), "races": g["race"].nunique(), "best_$_med": g["best_usd"].median(),
                     "cost_%": cost, "move/cost": g["fwd_120"].abs().median() / cost if cost > 0 else np.nan,
                     "random_%": np.nanmean(np.r_[bk, ly]) * 100, "hindsight_%": np.nanmean(np.fmax(bk, ly)) * 100,
                     "wom_ic": spearman(g["wom"].to_numpy(float), g["fwd_30"].to_numpy(float)),
                     "mom30_ic": spearman(g["mom_30"].to_numpy(float), g["fwd_30"].to_numpy(float))})
    return pd.DataFrame(rows)


def section_a(samples: str, races_csv: str | None, log) -> None:
    s = pd.read_parquet(samples)
    s = add_race_info(s, races_csv)
    hold = hold_days_of(s["race"].unique())
    tr = ~s["day"].isin(hold)
    s["runner_liq"] = quintiles(s["best_usd"], tr)
    s["race_liq"] = quintiles(s["matched_$"], tr) if s["matched_$"].notna().any() else np.nan
    log("\n=== A. Runner tradeability by the runner's own liquidity (money at its best back + lay prices) ===")
    log("(cost / move from the final 5 min, main runners and all others; 2-min round trips, $10, per trade)")
    for name, d in (("all races", s), ("thoroughbreds", s[s["race_type"] == "Flat"])):
        if d.empty:
            continue
        t = tradeability(d[d["t"] >= -300], "runner_liq")
        t.to_csv(os.path.join(OUT, f"A_runner_{name.split()[0]}.csv"), index=False)
        log(f"\n{name}:")
        log(t.round(3).to_string(index=False))
    f = s[(s["race_type"] == "Flat") & (s["t"] >= -300)]
    if f["race_liq"].notna().any() and len(f):
        log("\nthoroughbreds, race liquidity (total matched at the off; look-ahead, descriptive) x runner liquidity:")
        for col, name in (("hindsight", "perfect-hindsight 2-min trade %"), ("move", "move / cost")):
            g = f.groupby(["race_liq", "runner_liq"], observed=True)
            if col == "hindsight":
                v = g.apply(lambda x: np.nanmean(np.fmax(x["back_120"], x["lay_120"])) * 100, include_groups=False)
            else:
                v = g.apply(lambda x: x["fwd_120"].abs().median() / max(x["cost10"].median(), 1e-9), include_groups=False)
            log(f"  {name} (rows = race, columns = runner):")
            log(v.unstack().round(2).to_string())


# ----------------------------------------------------------------- B
def section_b(pairs: str, races_csv: str | None, log, ks=(2, 8)) -> dict:
    df = pd.read_parquet(pairs)
    df = add_race_info(df, races_csv)
    hold = hold_days_of(df["race"].unique())
    tr = ~df["day"].isin(hold).to_numpy()
    df["runner_liq"] = quintiles(df["depth3"], tr)
    df["race_liq"] = quintiles(df["matched_$"], tr) if df["matched_$"].notna().any() else np.nan
    X = df[FEATURES].to_numpy(float)
    log("\n=== B. P6's model (pair P&L, hedge k ticks, unfilled hedge -> BSP) by liquidity, HOLDOUT days ===")
    log("trained once on all TRAIN days; within each quintile: rank IC of predicted vs actual P&L (side picked by the "
        "model), mean P&L of all pairs, and of the model's top 5% picks (cut-off from TRAIN predictions)")
    best = None
    for k in ks:
        g = {side: pair_pnl(df, side, k, "bsp", "through")[0] for side in ("back", "lay")}
        pred = {side: _fit(X[tr], g[side][tr]).predict(X) for side in ("back", "lay")}
        score = np.fmax(pred["back"], pred["lay"])
        realised = np.where(pred["back"] >= pred["lay"], g["back"], g["lay"])
        thr = np.nanquantile(score[tr], 0.95)
        ho = ~tr
        for by in ("runner_liq", "race_liq"):
            if df[by].isna().all():
                continue
            rows = []
            for q in df[by].cat.categories:
                m = ho & (df[by] == q).to_numpy()
                top = m & (score >= thr) & np.isfinite(realised)
                rt = race_t(realised[top] * 100, df["race"].to_numpy()[top])
                rows.append({by: q, "pairs": int(m.sum()), "races": df.loc[m, "race"].nunique(),
                             "ic": spearman(score[m], realised[m]),
                             "all_pairs_mean_%": float(np.nanmean(realised[m]) * 100),  # every pair, on the side the model prefers
                             "top5_n": rt["n"], "top5_races": rt["races"], "top5_mean_%": rt["mean"], "top5_t": rt["t"]})
                if best is None or (rt["races"] >= 20 and np.nan_to_num(rt["t"], nan=-9) > best["t"]):
                    best = dict(k=k, by=by, quintile=str(q), mean=rt["mean"], t=rt["t"], races=rt["races"])
            t = pd.DataFrame(rows)
            t.to_csv(os.path.join(OUT, f"B_k{k}_{by}.csv"), index=False)
            what = "runner top-3 depth (live)" if by == "runner_liq" else "race total matched (look-ahead)"
            log(f"\nk={k}, by {what}:")
            log(t.round(3).to_string(index=False))
    return best or {}


# ----------------------------------------------------------------- C
def section_c(attempts: str, races_csv: str | None, since: str, log) -> dict:
    a = pd.read_parquet(attempts)
    a = add_race_info(a, races_csv)
    a = a[(a["day"] < since) & a["top3_live"]].copy()
    if a.empty:
        log("\n=== C. no P3c attempts before the forward period ===")
        return {}
    hold = hold_days_of(a["race"].unique())
    tr = ~a["day"].isin(hold)
    a["pnl"] = a["bsp_ev"].fillna(0) * a["filled"]
    measures = []
    if "best_usd" in a:
        measures.append(("best_usd", "runner money at best (live)"))
    if "mkt_matched_now" in a:
        measures.append(("mkt_matched_now", "market matched so far (live)"))
    if a["matched_$"].notna().any():
        measures.append(("matched_$", "race total matched (look-ahead, descriptive)"))
    log("\n=== C. P3c's frozen drift fade (live top 3) on the days before the forward test, by liquidity ===")
    if "best_usd" not in a:
        log("(this attempts.parquet predates the live liquidity columns: re-run cell P3c to add them)")
    best = {}
    for col, name in measures:
        a["q"] = quintiles(a[col], tr)
        for races_name, d in (("thoroughbreds", a[a["race_type"] == "Flat"]), ("all races", a)):
            rows = []
            for q, g in d.groupby("q", observed=True):
                rec = {"quintile": q, "median": g[col].median()}
                for sp, m in (("train", ~g["day"].isin(hold)), ("holdout", g["day"].isin(hold))):
                    rt = race_t(g.loc[m, "pnl"] * 100, g.loc[m, "race"])
                    rec.update({f"{sp}_n": rt["n"], f"{sp}_races": rt["races"], f"{sp}_mean_%": rt["mean"],
                                f"{sp}_t": rt["t"]})
                rows.append(rec)
                if races_name == "thoroughbreds" and rec["holdout_races"] >= 20 and \
                        np.nan_to_num(rec["holdout_t"], nan=-9) > best.get("holdout_t", -9) and rec["train_mean_%"] > 0:
                    best = dict(measure=col, quintile=str(q), **{k: rec[k] for k in rec if k != "quintile"})
            t = pd.DataFrame(rows)
            t.to_csv(os.path.join(OUT, f"C_{col}_{races_name.split()[0]}.csv"), index=False)
            log(f"\n{races_name}, by {name}:")
            log(t.round(3).to_string(index=False))
    return best


OUT = "."


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default=None, help="samples.parquet from the market study (2g)")
    ap.add_argument("--pairs", default=None, help="pairs.parquet from P6")
    ap.add_argument("--attempts", default=None, help="attempts.parquet from P3c")
    ap.add_argument("--races-csv", default=None, help="races.csv from the race study (2h)")
    ap.add_argument("--since", default="20261005", help="P3c forward start (attempts on/after it are left out)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    OUT = a.out
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    if a.samples and os.path.exists(a.samples):
        section_a(a.samples, a.races_csv, log)
    else:
        log("(no --samples: section A skipped; run cell 2g first)")
    best_b = section_b(a.pairs, a.races_csv, log) if a.pairs and os.path.exists(a.pairs) else {}
    best_c = section_c(a.attempts, a.races_csv, a.since, log) if a.attempts and os.path.exists(a.attempts) else {}
    verdict = dict(best_model_slice=best_b, best_drift_fade_slice=best_c,
                   note="slices found by searching quintiles are hypotheses: each needs its own forward test")
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    log("\n=== Summary ===")
    log(json.dumps(verdict, indent=1, default=float))
    log(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
