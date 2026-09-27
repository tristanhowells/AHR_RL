"""Signal bot: the edge test's passive-entry signal, traded inside the full simulator.

The edge test (edge.py) found a small but consistent signal for *passive*
entries, under a generous fill assumption (any trade at our price = filled).
This bot checks whether that survives realistic execution:

  * gradient-boosted models (trained on TRAIN days, exactly as in edge.py)
    predict the passive back / lay round-trip return for horizon H
  * every 2s, for each tradeable runner that is flat, if a prediction clears the
    top-q% threshold (quantile of VALIDATION-day predictions) the bot opens a
    bracket: JOIN the queue at the best price on its side, $10 stake
  * the entry is cancelled if still unfilled after H/2; an open position is
    closed at market after H seconds (the bracket's own take-profit / stop can
    exit it sooner) - the same trade the edge test scored
  * everything else is the real simulator: 0.5s latency, queue position behind
    existing size, liquidity we took, funds check, auto-green from the
    scheduled start, commission

H and q are chosen on validation races *in the simulator*; the chosen config
(and, for context, the next-best ones) is scored once on test races.

    python -m ahr_rl.signal_bot --tapes "data/tapes/*.npz" --out runs/signal_bot \\
        --samples runs/edge/samples_base.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .edge import build_dataset, feature_record, tradeable
from .env import CANCEL_ENTRY, CLOSE, JOIN, BACK, LAY, BetfairPreRaceEnv, EnvConfig, encode_open, list_tapes, split_by_date
from .features import R_MAX, global_features, runner_features

SIDES = {"back": BACK, "lay": LAY}


# ============================================================================ models
def train_models(df: pd.DataFrame, split: dict, horizons, feature_set: str = "base", max_train: int = 400_000,
                 seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor

    feat = [c for c in df.columns if c[0] in "rg" and c[1:].isdigit()]
    if feature_set == "extended":
        feat += [c for c in df.columns if c.startswith("x_")]
    day_of = {os.path.basename(p).split(".npz")[0]: k for k, ps in split.items() for p in ps}
    df = df.assign(split=df["race"].map(day_of))
    tr_all, va_all = df[df["split"] == "train"], df[df["split"] == "val"]
    rng = np.random.default_rng(seed)
    bundle = {"feat": feat, "feature_set": feature_set, "models": {}, "val_pred": {}}
    for H in horizons:
        for side in SIDES:
            col = f"{side}_passive_{H}"
            tr = tr_all.dropna(subset=[col])
            if len(tr) > max_train:
                tr = tr.iloc[rng.choice(len(tr), max_train, replace=False)]
            y = tr[col].clip(tr[col].quantile(0.001), tr[col].quantile(0.999))
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=200, l2_regularization=1.0, early_stopping=True,
                                              validation_fraction=0.1, random_state=seed)
            m.fit(tr[feat].values, y.values)
            bundle["models"][(side, H)] = m
            va = va_all.dropna(subset=[col])
            bundle["val_pred"][(side, H)] = m.predict(va[feat].values)
            print(f"[signal_bot] trained {col} on {len(tr):,} rows", flush=True)
    return bundle


def thresholds(bundle: dict, H: int, q: float) -> dict:
    """Top-q% of validation-day predictions, per side (never lower than 0)."""
    return {side: max(0.0, float(np.percentile(bundle["val_pred"][(side, H)], 100 - q))) for side in SIDES}


# ============================================================================ policy
class SignalPolicy:
    def __init__(self, bundle: dict, H: int, q: float, stake_idx: int = 1, tp_idx: int = 2,
                 sides=("back", "lay")):
        self.b, self.H, self.q = bundle, H, q
        self.th = thresholds(bundle, H, q)
        self.stake_idx, self.tp_idx, self.sides = stake_idx, tp_idx, sides
        self._tape_key, self._micro = None, None
        self.n_signals = 0

    def _micro_for(self, env):
        key = (env.tape.name, id(env.tape))
        if key != self._tape_key:
            self._tape_key = key
            self._micro = None
            if self.b["feature_set"] == "extended":
                from .microfeatures import compute
                self._micro = compute(env.tape, env.hist.mid)
        return self._micro

    def __call__(self, obs, env):
        ex, t, cfg = env.ex, env.tape, env.cfg
        s = ex.step
        a = np.zeros(R_MAX, np.int64)
        R = min(t.n_runners, R_MAX)
        dt = t.dt
        # 1) manage open brackets: mirror the edge test's trade (fill window H/2, exit at H)
        busy = set()
        for r, br in list(ex.brackets.items()):
            if r >= R:
                continue
            busy.add(r)
            if br.closing:
                continue
            age = (s - br.opened_step) * dt
            flat = abs(ex.W[r] - ex.L[r]) < 0.01
            if not flat and age >= self.H:
                a[r] = CLOSE
            elif flat and age >= self.H / 2 and any(o.oid == br.entry_oid for o in ex.orders):
                a[r] = CANCEL_ENTRY
        if env.in_auto_green():
            return a
        # 2) new entries on tradeable, flat, idle runners
        cand = [r for r in range(R) if r not in busy and obs["mask"][r] and tradeable(t, s, r)
                and abs(ex.W[r] - ex.L[r]) < 0.01 and not any(o.runner == r for o in ex.orders)]
        if not cand:
            return a
        F, _ = runner_features(env.hist, s, ex)
        g = global_features(env.hist, s, ex, 0.0)
        micro = self._micro_for(env)
        X = pd.DataFrame([feature_record(F, g, micro, s, r) for r in cand])[self.b["feat"]].values
        preds = {side: self.b["models"][(side, self.H)].predict(X) for side in self.sides}
        for i, r in enumerate(cand):
            best_side, best_edge = None, 0.0
            for side in self.sides:
                edge = preds[side][i] - self.th[side]
                if preds[side][i] > self.th[side] and edge > best_edge:
                    best_side, best_edge = side, edge
            if best_side is not None:
                a[r] = encode_open(SIDES[best_side], JOIN, self.stake_idx, self.tp_idx, cfg)
                self.n_signals += 1
        return a


# ============================================================================ evaluation (parallel)
_W = {}


def _init_worker(bundle_path):
    with open(bundle_path, "rb") as fh:
        _W["bundle"] = pickle.load(fh)


def _run_one(args):
    tape, H, q, stake_idx, tp_idx = args
    env = BetfairPreRaceEnv([tape], EnvConfig(), cache_tapes=False)
    pol = SignalPolicy(_W["bundle"], H, q, stake_idx, tp_idx)
    obs, _ = env.reset(options={"tape": tape})
    done, info = False, {}
    while not done:
        obs, _, done, _, info = env.step(pol(obs, env))
    return {"market": info.get("market"), "worst": info["worst"], "expected": info["expected"],
            "void": info["void"], "turnover": info["turnover"], "n_opens": info["n_opens"],
            "n_fills": info["n_fills"], "n_stops": info["n_stops"], "signals": pol.n_signals}


def run_config(pool, tapes, H, q, stake_idx, tp_idx) -> pd.DataFrame:
    return pd.DataFrame(list(pool.map(_run_one, [(t, H, q, stake_idx, tp_idx) for t in tapes], chunksize=2)))


def summarise_races(df: pd.DataFrame) -> dict:
    live = df[~df["void"]]
    x = live["worst"].values
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return {"races": len(live), "usd_per_race": float(x.mean()), "tstat_races": float(x.mean() / se) if se else np.nan,
            "pct_green": float((x >= -0.005).mean() * 100), "pct_races_traded": float((live["turnover"] > 0).mean() * 100),
            "trades_per_race": float(live["n_opens"].mean()), "turnover_per_race": float(live["turnover"].mean()),
            "expected_usd_per_race": float(live["expected"].mean()), "worst_race": float(x.min()),
            "best_race": float(x.max())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", default="data/tapes/*.npz")
    ap.add_argument("--out", default="runs/signal_bot")
    ap.add_argument("--samples", default=None, help="reuse an edge-test samples parquet (must match --features)")
    ap.add_argument("--features", default="base", choices=["base", "extended"])
    ap.add_argument("--horizons", default="10,30,60")
    ap.add_argument("--top-pcts", default="0.5,1,2,5")
    ap.add_argument("--stake-idx", type=int, default=1, help="index into EnvConfig.stakes (1 = $10)")
    ap.add_argument("--tp-idx", type=int, default=2, help="index into EnvConfig.tp_ticks (2 = 4 ticks)")
    ap.add_argument("--test-top", type=int, default=3, help="also score the next-best val configs on test")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--max-train", type=int, default=400_000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    horizons = [int(x) for x in a.horizons.split(",")]
    qs = [float(x) for x in a.top_pcts.split(",")]
    paths = list_tapes(a.tapes)
    tr, va, te = split_by_date(paths)
    split = {"train": tr, "val": va, "test": te}
    print(f"[signal_bot] races train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    t0 = time.time()
    cache = a.samples or os.path.join(a.out, f"samples_{a.features}.parquet")
    df = build_dataset(paths, cache=cache, workers=a.workers, horizons_s=horizons, extended=a.features == "extended")
    if a.features == "extended" and not any(c.startswith("x_") for c in df.columns):
        raise SystemExit("--features extended needs samples built with extended features")
    bundle = train_models(df, split, horizons, a.features, a.max_train)
    del df
    bpath = os.path.join(a.out, f"models_{a.features}.pkl")
    with open(bpath, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"[signal_bot] models ready in {time.time() - t0:.0f}s", flush=True)

    pd.set_option("display.width", 220)
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(bpath,)) as pool:
        for H in horizons:
            for q in qs:
                t1 = time.time()
                s = summarise_races(run_config(pool, va, H, q, a.stake_idx, a.tp_idx))
                rows.append(dict(split="val", horizon_s=H, top_pct=q, **s))
                print(f"[val] H={H}s top{q}%: ${s['usd_per_race']:+.3f}/race (t={s['tstat_races']:+.2f}), "
                      f"{s['trades_per_race']:.1f} trades/race, {s['pct_green']:.0f}% green  [{time.time() - t1:.0f}s]",
                      flush=True)
        val = pd.DataFrame(rows).sort_values("usd_per_race", ascending=False)
        chosen = val.iloc[0]
        test_rows = []
        for rank, (_, r) in enumerate(val.head(max(1, a.test_top)).iterrows()):
            H, q = int(r["horizon_s"]), float(r["top_pct"])
            s = summarise_races(run_config(pool, te, H, q, a.stake_idx, a.tp_idx))
            test_rows.append(dict(split="test", val_rank=rank + 1, horizon_s=H, top_pct=q, **s))
            print(f"[test] (val rank {rank + 1}) H={H}s top{q}%: ${s['usd_per_race']:+.3f}/race "
                  f"(t={s['tstat_races']:+.2f})", flush=True)
    test = pd.DataFrame(test_rows)
    val.to_csv(os.path.join(a.out, f"val_{a.features}.csv"), index=False)
    test.to_csv(os.path.join(a.out, f"test_{a.features}.csv"), index=False)
    print("\n=== Validation (simulator; configs ranked by $/race) ===")
    print(val.round(3).to_string(index=False))
    print("\n=== Test (val-chosen config is val_rank 1; the others are context only) ===")
    print(test.round(3).to_string(index=False))
    ch = test[test["val_rank"] == 1].iloc[0].to_dict()
    verdict = {"features": a.features, "chosen_on_val": {"horizon_s": int(chosen["horizon_s"]),
                                                         "top_pct": float(chosen["top_pct"])},
               "test": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                        for k, v in ch.items()}}
    json.dump(verdict, open(os.path.join(a.out, f"verdict_{a.features}.json"), "w"), indent=1)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1))
    print("test usd_per_race > 0 with tstat_races > 2 -> the passive signal survives realistic execution;\n"
          "build the agent around it. Otherwise the edge-test profit was an artefact of generous fills.")


if __name__ == "__main__":
    main()
