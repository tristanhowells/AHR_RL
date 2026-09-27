"""Do engineered microstructure features (WoM, matched-price range, WAP, trade
flow) carry signal the agent doesn't already have?

    python -m ahr_rl.feature_study --tapes "data/tapes/*.npz" --out runs/features

1. Univariate: per-race Spearman IC between each feature and the forward mid
   move in ticks (10/30/60s), averaged over races, with a t-stat across races.
   Reported for the new features and the agent's existing ones for reference.
   Sign convention: target < 0 means the price shortened, so a NEGATIVE IC means
   "high feature value -> price shortens -> backing is favoured".
2. Ablation: gradient-boosted model trained on train days with (a) the agent's
   current features, (b) current + engineered, (c) engineered only; scored on
   test days by IC on the mid move and by the val-selected trading rule.
3. Permutation importance (test days) for the extended 30s mid-move model.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .edge import build_dataset, fit_and_score
from .env import list_tapes, split_by_date

# agent features worth showing next to the new ones (see features.py)
REFERENCE = {"r11": "agent: WoM top-3", "r12": "agent: LTP - mid", "r13": "agent: mid chg 5s",
             "r14": "agent: mid chg 15s", "r15": "agent: mid chg 30s", "r16": "agent: mid chg 60s",
             "r17": "agent: vol 5s", "r18": "agent: vol 30s", "r20": "agent: proj. BSP - mid",
             "r3": "agent: spread", "r4": "agent: implied prob"}


def univariate(df: pd.DataFrame, horizons) -> pd.DataFrame:
    from scipy.stats import spearmanr

    cols = [c for c in df.columns if c.startswith("x_")] + [c for c in REFERENCE if c in df.columns]
    rows = []
    for H in horizons:
        tgt = f"mid_ticks_{H}"
        d = df.dropna(subset=[tgt])
        for c in cols:
            ics = []
            for _, g in d.groupby("race"):
                if len(g) < 30 or g[c].nunique() < 3 or g[tgt].nunique() < 3:
                    continue
                ics.append(spearmanr(g[c], g[tgt]).statistic)
            ics = np.array([x for x in ics if np.isfinite(x)])
            if len(ics) < 5:
                continue
            m, se = ics.mean(), ics.std(ddof=1) / np.sqrt(len(ics))
            rows.append(dict(horizon_s=H, feature=REFERENCE.get(c, c[2:]), mean_ic=m, t_races=m / se,
                             pct_races_same_sign=float((np.sign(ics) == np.sign(m)).mean() * 100),
                             n_races=len(ics)))
    out = pd.DataFrame(rows)
    out["abs_t"] = out["t_races"].abs()
    return out.sort_values(["horizon_s", "abs_t"], ascending=[True, False]).drop(columns="abs_t")


def importance(df: pd.DataFrame, split: dict, H: int = 30, seed: int = 0, n_test: int = 20000) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    tgt = f"mid_ticks_{H}"
    feat = [c for c in df.columns if (c[0] in "rg" and c[1:].isdigit()) or c.startswith("x_")]
    day_of = {os.path.basename(p).split(".npz")[0]: k for k, ps in split.items() for p in ps}
    d = df.assign(split=df["race"].map(day_of)).dropna(subset=[tgt])
    tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
    rng = np.random.default_rng(seed)
    if len(tr) > 400_000:
        tr = tr.iloc[rng.choice(len(tr), 400_000, replace=False)]
    if len(te) > n_test:
        te = te.iloc[rng.choice(len(te), n_test, replace=False)]
    y = tr[tgt].clip(tr[tgt].quantile(0.001), tr[tgt].quantile(0.999))
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=200,
                                      l2_regularization=1.0, early_stopping=True, random_state=seed)
    m.fit(tr[feat].values, y.values)
    pi = permutation_importance(m, te[feat].values, te[tgt].values, n_repeats=5, random_state=seed,
                                scoring="r2")
    names = [REFERENCE.get(c, c[2:] if c.startswith("x_") else f"agent: {c}") for c in feat]
    return (pd.DataFrame({"feature": names, "importance_r2": pi.importances_mean, "std": pi.importances_std})
            .sort_values("importance_r2", ascending=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", default="data/tapes/*.npz")
    ap.add_argument("--out", default="runs/features")
    ap.add_argument("--horizons", default="10,30,60")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    horizons = [int(x) for x in a.horizons.split(",")]
    paths = list_tapes(a.tapes)
    tr, va, te = split_by_date(paths)
    split = {"train": tr, "val": va, "test": te}
    df = build_dataset(paths, cache=os.path.join(a.out, "samples_extended.parquet"), workers=a.workers,
                       horizons_s=horizons, extended=True)
    print(f"[features] {len(df):,} samples, {df['race'].nunique()} races "
          f"(train {len(tr)} / val {len(va)} / test {len(te)})", flush=True)
    pd.set_option("display.width", 220)

    uni = univariate(df, horizons)
    uni.to_csv(os.path.join(a.out, "univariate.csv"), index=False)
    print("\n=== 1. Univariate signal vs forward mid move (ticks; IC < 0 = high value -> price shortens) ===")
    for H in horizons:
        print(f"\n-- {H}s --")
        print(uni[uni["horizon_s"] == H].head(20).round(3).to_string(index=False))

    tgts = ("mid_ticks", "back_take", "lay_take", "back_passive", "lay_passive")
    preds, rules = [], []
    for fs in ("base", "extended", "micro_only"):
        p, r = fit_and_score(df, split, horizons, feature_set=fs, targets=tgts, models_to_fit=("gbm",))
        preds.append(p)
        rules.append(r)
    preds, rules = pd.concat(preds), pd.concat(rules)
    preds.to_csv(os.path.join(a.out, "ablation_predictability.csv"), index=False)
    rules.to_csv(os.path.join(a.out, "ablation_rules.csv"), index=False)
    print("\n=== 2a. Ablation: test-day IC of the mid-move model ===")
    mp = preds[preds["target"] == "mid_ticks"].pivot(index="horizon_s", columns="features", values="test_ic")
    print(mp.round(4).to_string())
    print("\n=== 2b. Ablation: val-selected trading rule per family, scored on test ($10 stakes) ===")
    rows = []
    for fs, g in rules.groupby("features"):
        for fam, trades in (("cross_spread", ["back_take", "lay_take"]), ("passive_entry", ["back_passive", "lay_passive"])):
            f = g[g["trade"].isin(trades)]
            if not len(f):
                continue
            r = f.loc[f["val_usd_per_race"].idxmax()]
            rows.append(dict(features=fs, family=fam, rule=f"{r['trade']} {int(r['horizon_s'])}s top{r['top_pct']}%",
                             val_usd_per_race=r["val_usd_per_race"], test_usd_per_race=r["test_usd_per_race"],
                             test_tstat=r["test_tstat_races"], test_trades_per_race=r["test_trades_per_race"]))
    summ = pd.DataFrame(rows)
    print(summ.round(3).to_string(index=False))

    imp = importance(df, split, H=30)
    imp.to_csv(os.path.join(a.out, "importance_30s.csv"), index=False)
    print("\n=== 3. Permutation importance, extended 30s mid-move model (test days) ===")
    print(imp.head(20).round(5).to_string(index=False))
    json.dump({"ablation_ic": mp.round(4).to_dict(), "rules": summ.to_dict(orient="records")},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
