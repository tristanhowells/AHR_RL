"""Edge test: is there a *tradeable* short-term signal in the data at all?

Before spending more RL compute, check with plain supervised learning whether
the agent's own features predict price moves well enough to beat the cost of a
round trip. Same tapes, same features, same chronological train/val/test split
as the RL agent.

For every runner with a two-sided book, every ``every_s`` seconds, and each
horizon H, we compute the P&L per $1 of a round trip that opens now and greens
up H seconds later (net of commission on profit):

  conservative (cross the spread in and out - always fills):
    back_take  = P_back(t) / P_lay(t+H) - 1          (back now, lay later)
    lay_take   = 1 - P_lay(t) / P_back(t+H)          (lay now, back later)
  passive entry, aggressive exit (optimistic on fills: the entry counts as
  filled if the market trades at or through our price within H/2, ignoring
  queue position; adverse selection is kept because fills only happen when
  the market comes to us). Unfilled = no trade = 0:
    back_passive = P_lay(t) / P_lay(t+H) - 1         (offer to back at the lay price)
    lay_passive  = 1 - P_back(t) / P_back(t+H)       (offer to lay at the back price)
  frictionless:
    mid        = P_mid(t) / P_mid(t+H) - 1           (raw predictability only)

where P_back is the best price you can back at (best atb) and P_lay the best
price you can lay at (best atl). Only runners that are realistically tradeable
are sampled (spread <= ``max_spread_ticks``, price <= ``max_price``). Exits must happen before ``exit_cutoff_s``
(default 0 = the scheduled start, matching the auto-green rule).

Models (fit on train days): ridge and gradient-boosted trees. A trading rule
"take the trade when predicted return > theta" has theta picked on validation
days and is then scored once on test days. Results are aggregated per race so
the t-statistics aren't inflated by overlapping samples within a race.

    python -m ahr_rl.edge --tapes "data/tapes/*.npz" --out runs/edge
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .env import list_tapes, split_by_date
from .exchange import Exchange, ExchangeConfig
from .features import N_GLOBAL_FEATURES, N_RUNNER_FEATURES, R_MAX, TapeHistory, global_features, runner_features
from .ladder import PRICES
from .tape import Tape

TRADES = ("back_take", "lay_take", "back_passive", "lay_passive")
TARGETS = TRADES + ("mid",)
# features that describe *our* position/orders/brackets are always zero here
POSITION_FEATURES = list(range(22, 31)) + list(range(34, 38))


def build_samples(path: str, horizons_s=(10, 30, 60), every_s: float = 2.0, exit_cutoff_s: float = 0.0,
                  min_size: float = 5.0, max_spread_ticks: int = 3, max_price: float = 30.0) -> pd.DataFrame | None:
    t = Tape.load(path)
    h = TapeHistory(t)
    ex = Exchange(t, ExchangeConfig())
    ex.brackets = {}
    comm = t.base_rate / 100.0
    T, R = t.n_steps, min(t.n_runners, R_MAX)
    every = max(1, int(round(every_s / t.dt)))
    ok_exit = (t.t_rel < exit_cutoff_s) & (np.arange(T) < T - 1) & ~t.suspended
    last_ok = np.nonzero(ok_exit)[0]
    if not len(last_ok):
        return None
    last_ok = last_ok[-1]
    hsteps = {H: int(round(H / t.dt)) for H in horizons_s}
    bb, bl = t.back_tick[:, :, 0].astype(int), t.lay_tick[:, :, 0].astype(int)
    bsz, lsz = t.back_size[:, :, 0], t.lay_size[:, :, 0]
    # highest / lowest traded tick per (step, runner) for passive-fill checks
    tr_hi = np.full((T, t.n_runners), -1, np.int64)
    tr_lo = np.full((T, t.n_runners), 10 ** 6, np.int64)
    rr, tk = t.trade_runner.astype(np.int64), t.trade_tick.astype(np.int64)
    np.maximum.at(tr_hi, (t.trade_step, rr), tk)
    np.minimum.at(tr_lo, (t.trade_step, rr), tk)
    rows = []
    for s in range(0, last_ok + 1, every):
        if t.suspended[s]:
            continue
        ex.step = s
        F, mask = runner_features(h, s, ex)
        g = global_features(h, s, ex, 0.0)
        for r in range(R):
            if not mask[r] or bb[s, r] < 0 or bl[s, r] < 0:
                continue
            if bl[s, r] - bb[s, r] > max_spread_ticks or PRICES[bl[s, r]] > max_price:
                continue
            pb, pl = PRICES[bb[s, r]], PRICES[bl[s, r]]
            race = os.path.basename(path).split(".npz")[0]
            rec = {"race": race, "day": race[:8], "step": s, "runner": r, "t_rel": float(t.t_rel[s]),
                   "entry_size_back": float(bsz[s, r]), "entry_size_lay": float(lsz[s, r])}
            any_target = False
            for H, k in hsteps.items():
                e = s + k
                if e > last_ok or not t.active[e, r] or bb[e, r] < 0 or bl[e, r] < 0:
                    for name in TARGETS:
                        rec[f"{name}_{H}"] = np.nan
                    continue
                pb2, pl2 = PRICES[bb[e, r]], PRICES[bl[e, r]]
                w = slice(s + 1, s + max(1, k // 2) + 1)
                # back offer at bl[s] fills if layers trade at/above it or the book crosses it
                back_fill = (tr_hi[w, r] >= bl[s, r]).any() or (bb[w, r] >= bl[s, r]).any()
                lay_fill = (tr_lo[w, r] <= bb[s, r]).any() or ((bl[w, r] >= 0) & (bl[w, r] <= bb[s, r])).any()
                vals = {
                    "back_take": pb / pl2 - 1 if bsz[s, r] >= min_size else np.nan,
                    "lay_take": 1 - pl / pb2 if lsz[s, r] >= min_size else np.nan,
                    "back_passive": (pl / pl2 - 1) if back_fill else 0.0,
                    "lay_passive": (1 - pb / pb2) if lay_fill else 0.0,
                    "mid": (pb + pl) / (pb2 + pl2) - 1,
                }
                for name, v in vals.items():
                    if name != "mid" and np.isfinite(v) and v > 0:
                        v *= 1 - comm
                    rec[f"{name}_{H}"] = v
                any_target = True
            if not any_target:
                continue
            for i in range(N_RUNNER_FEATURES):
                if i not in POSITION_FEATURES:
                    rec[f"r{i}"] = F[r, i]
            for i in range(N_GLOBAL_FEATURES):
                if i not in (6, 7, 8, 11):  # funds / P&L / fills: always constant here
                    rec[f"g{i}"] = g[i]
            rows.append(rec)
    return pd.DataFrame(rows) if rows else None


def build_dataset(paths, cache: str | None = None, workers: int = 1, **kw) -> pd.DataFrame:
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(_build_one, [(p, kw) for p in paths], chunksize=4))
    else:
        parts = [_build_one((p, kw)) for p in paths]
    df = pd.concat([p for p in parts if p is not None], ignore_index=True)
    if cache:
        df.to_parquet(cache)
    return df


def _build_one(arg):
    p, kw = arg
    try:
        return build_samples(p, **kw)
    except Exception as e:  # a bad tape shouldn't kill the whole run
        print(f"[edge] skip {os.path.basename(p)}: {e!r}", flush=True)
        return None


def _per_race(df_sel: pd.DataFrame, col: str, races: list[str], stake: float):
    """Race-level P&L of a rule (sum over its trades in each race; 0 if none)."""
    per = df_sel.groupby("race")[col].sum() * stake
    per = per.reindex(races, fill_value=0.0)
    n = len(per)
    mean = per.mean()
    se = per.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return mean, (mean / se if se and se > 0 else np.nan), per


def fit_and_score(df: pd.DataFrame, split: dict, horizons, stake: float = 10.0, max_train: int = 400_000,
                  seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feat = [c for c in df.columns if c[0] in "rg" and c[1:].isdigit()]
    day_of = {os.path.basename(p).split(".npz")[0]: k for k, ps in split.items() for p in ps}
    df = df.assign(split=df["race"].map(day_of))
    parts = {k: df[df["split"] == k] for k in ("train", "val", "test")}
    test_races = sorted(parts["test"]["race"].unique())
    val_races = sorted(parts["val"]["race"].unique())
    rng = np.random.default_rng(seed)
    pred_rows, rule_rows = [], []
    for H in horizons:
        for name in TARGETS:
            col = f"{name}_{H}"
            tr = parts["train"].dropna(subset=[col])
            if len(tr) > max_train:
                tr = tr.iloc[rng.choice(len(tr), max_train, replace=False)]
            va, te = parts["val"].dropna(subset=[col]), parts["test"].dropna(subset=[col])
            if len(tr) < 1000 or len(va) < 100 or len(te) < 100:
                continue
            y = tr[col].clip(tr[col].quantile(0.001), tr[col].quantile(0.999))  # tame fat tails
            models = {
                "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
                "gbm": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                     min_samples_leaf=200, l2_regularization=1.0,
                                                     early_stopping=True, validation_fraction=0.1,
                                                     random_state=seed),
            }
            for mname, m in models.items():
                m.fit(tr[feat].values, y.values)
                pv, pt = m.predict(va[feat].values), m.predict(te[feat].values)
                ic = spearmanr(pt, te[col].values).statistic
                r2 = 1 - np.mean((te[col].values - pt) ** 2) / np.var(te[col].values)
                pred_rows.append(dict(horizon_s=H, target=name, model=mname, test_ic=ic, test_r2=r2,
                                      n_train=len(tr), n_test=len(te),
                                      mean_return_all=float(te[col].mean())))
                if name == "mid":
                    continue
                # trading rule: act only on the top-q% predictions; q (-> theta) chosen on validation
                best = None
                for q in (0.1, 0.5, 1, 2, 5, 10, 25):
                    th = np.percentile(pv, 100 - q)
                    sel = va[pv > th]
                    if len(sel) < 30:
                        continue
                    mr = sel[col].mean()
                    if best is None or mr > best[1]:
                        best = (q, mr, th)
                if best is None:
                    continue
                q, val_mean, th = best
                val_usd, _, _ = _per_race(va[pv > th], col, val_races, stake)
                sel = te[pt > th]
                race_mean, tstat, _ = _per_race(sel, col, test_races, stake)
                rule_rows.append(dict(
                    horizon_s=H, trade=name, model=mname, top_pct=q, val_mean_return=val_mean,
                    val_usd_per_race=val_usd,
                    test_trades=len(sel), test_trades_per_race=len(sel) / max(len(test_races), 1),
                    test_mean_return=float(sel[col].mean()) if len(sel) else np.nan,
                    test_hit_rate=float((sel[col] > 0).mean()) if len(sel) else np.nan,
                    test_usd_per_race=race_mean, test_tstat_races=tstat,
                ))
    return pd.DataFrame(pred_rows), pd.DataFrame(rule_rows)


def cost_table(df: pd.DataFrame, horizons) -> pd.DataFrame:
    """Unconditional round-trip economics: what trading blindly costs."""
    rows = []
    for H in horizons:
        for name in TRADES:
            v = df[f"{name}_{H}"].dropna()
            rows.append(dict(horizon_s=H, trade=name, n=len(v), mean_return=v.mean(), median_return=v.median(),
                             pct_profitable=(v > 0).mean() * 100))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tapes", default="data/tapes/*.npz")
    ap.add_argument("--out", default="runs/edge")
    ap.add_argument("--horizons", default="10,30,60")
    ap.add_argument("--every-s", type=float, default=2.0)
    ap.add_argument("--exit-cutoff-s", type=float, default=0.0)
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--max-train", type=int, default=400_000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    horizons = [int(x) for x in a.horizons.split(",")]
    paths = list_tapes(a.tapes)
    tr, va, te = split_by_date(paths)
    split = {"train": tr, "val": va, "test": te}
    print(f"[edge] races train {len(tr)} val {len(va)} test {len(te)}", flush=True)
    t0 = time.time()
    df = build_dataset(paths, cache=os.path.join(a.out, "samples.parquet"), workers=a.workers,
                       horizons_s=horizons, every_s=a.every_s, exit_cutoff_s=a.exit_cutoff_s)
    print(f"[edge] {len(df):,} samples from {df['race'].nunique()} races in {time.time() - t0:.0f}s", flush=True)

    pd.set_option("display.width", 200)
    costs = cost_table(df, horizons)
    print("\n=== Cost of trading blindly (mean return per $1 staked, net of commission) ===")
    print(costs.round(4).to_string(index=False))
    preds, rules = fit_and_score(df, split, horizons, a.stake, a.max_train)
    print("\n=== Predictability on TEST days (Spearman IC between prediction and outcome) ===")
    print(preds.round(4).to_string(index=False))
    print(f"\n=== Trading rules (threshold chosen on VAL, scored on TEST, ${a.stake:g} stake per trade) ===")
    print(rules.round(4).to_string(index=False))

    costs.to_csv(os.path.join(a.out, "costs.csv"), index=False)
    preds.to_csv(os.path.join(a.out, "predictability.csv"), index=False)
    rules.to_csv(os.path.join(a.out, "rules.csv"), index=False)
    def pick(trades):
        """The single rule chosen on VALIDATION for a family, reported on TEST
        (never select on test: with 12 rules per family one will look good by luck)."""
        if not len(rules):
            return None
        fam = rules[rules["trade"].isin(trades)]
        if not len(fam):
            return None
        r = fam.loc[fam["val_usd_per_race"].idxmax()]
        return {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in r.items()}

    verdict = {
        "n_test_races": int(len(split["test"])),
        "cross_spread_rule (val-selected)": pick(["back_take", "lay_take"]),
        "passive_entry_rule (val-selected)": pick(["back_passive", "lay_passive"]),
        "max_mid_ic": float(preds[preds["target"] == "mid"]["test_ic"].max()) if len(preds) else None,
    }
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1))
    print("Read test_usd_per_race and test_tstat_races of the two val-selected rules:\n"
          "  cross-spread > 0 with t > 2  -> a real edge the RL agent should be able to find\n"
          "  only passive-entry > 0, t > 2 -> edge depends on passive fills; queue/fill realism decides\n"
          "  both <= 0 or t < 2           -> nothing short-term to exploit with these features;\n"
          "                                  'do nothing' is the correct policy")


if __name__ == "__main__":
    main()
