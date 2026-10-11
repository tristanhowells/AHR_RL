"""Supervised price-move forecaster whose predictions become agent features.

RL is a poor way to *discover* a weak predictive signal from sparse P&L; a
gradient-boosted model fitted directly on price moves is far more sample
efficient. So we fit one per target (e.g. the mid move in ticks over the next
30s / 120s / 480s) on TRAIN days and hand the predictions to the agent, which
then only has to learn *what to do* with them (timing, side, size, exits).

Leakage control (cross-fitting): train-day predictions must be out-of-sample
too, otherwise the agent trains on over-confident forecasts and is
disappointed on test days. Train days are split into ``k`` folds by day; a race
on a train day is scored by the model fitted without its fold. Val/test races
use the model fitted on all train days.

    python -m ahr_rl.forecaster --samples runs/edge/samples_base.parquet[,runs/edge_long/samples_base.parquet] \\
        --tapes "data/tapes/*.npz" --out runs/forecaster.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd

from .edge import feature_record
from .env import list_tapes, split_by_date
from .features import R_MAX, global_features, runner_features

KEYS = ["race", "step", "runner"]


def _scale(target: str, v: np.ndarray) -> np.ndarray:
    """Put every forecast on a comparable O(1) scale for the policy."""
    if target.startswith("mid_ticks"):
        return np.clip(v / 5.0, -4, 4)
    return np.clip(v * 20.0, -4, 4)  # returns per $1 -> 5% = 1.0


def load_samples(paths: list[str]) -> pd.DataFrame:
    df = None
    for p in paths:
        d = pd.read_parquet(p)
        if df is None:
            df = d
        else:
            new = [c for c in d.columns if c not in df.columns]
            df = df.merge(d[KEYS + new], on=KEYS, how="left")
    return df


def fit(df: pd.DataFrame, split: dict, targets: list[str], folds: int = 3, max_train: int = 400_000,
        seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor

    feat = [c for c in df.columns if c[0] in "rg" and c[1:].isdigit()]
    train_races = {os.path.basename(p).split(".npz")[0] for p in split["train"]}
    tr = df[df["race"].isin(train_races)]
    days = sorted(tr["day"].unique())
    day_to_fold = {d: i % folds for i, d in enumerate(days)} if folds > 1 else {}
    rng = np.random.default_rng(seed)

    def gbm(X, y):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                          l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                          random_state=seed)
        return m.fit(X, y)

    def fit_on(rows, col):
        rows = rows.dropna(subset=[col])
        if len(rows) > max_train:
            rows = rows.iloc[rng.choice(len(rows), max_train, replace=False)]
        y = rows[col].clip(rows[col].quantile(0.001), rows[col].quantile(0.999))
        return gbm(rows[feat].values, y.values)

    full, by_fold = {}, {k: {} for k in range(folds)} if folds > 1 else {}
    for col in targets:
        t0 = time.time()
        full[col] = fit_on(tr, col)
        for k in by_fold:
            by_fold[k][col] = fit_on(tr[tr["day"].map(day_to_fold) != k], col)
        print(f"[forecaster] {col}: fitted ({len(by_fold)} folds + full) in {time.time() - t0:.0f}s", flush=True)
    return {"feat": feat, "targets": targets, "full": full, "by_fold": by_fold, "day_to_fold": day_to_fold}


class Forecaster:
    """Runtime wrapper used by the environment / live bot."""

    def __init__(self, bundle: dict):
        self.b = bundle
        self.targets = bundle["targets"]
        self.feat = bundle["feat"]

    @staticmethod
    def load(path: str) -> "Forecaster":
        with open(path, "rb") as fh:
            return Forecaster(pickle.load(fh))

    @property
    def n_outputs(self) -> int:
        return len(self.targets)

    def _models(self, race_day: str | None):
        k = self.b["day_to_fold"].get(race_day) if race_day else None
        return self.b["by_fold"][k] if k is not None else self.b["full"]

    def predict(self, h, step: int, ex, race_day: str | None, mask: np.ndarray) -> np.ndarray:
        """[R_MAX, n_outputs] scaled forecasts for the active runners (0 elsewhere)."""
        out = np.zeros((R_MAX, self.n_outputs), np.float32)
        idx = np.nonzero(mask)[0]
        if not len(idx):
            return out
        F, _ = runner_features(h, step, ex)
        g = global_features(h, step, ex, 0.0)
        X = np.array([[feature_record(F, g, None, step, r)[c] for c in self.feat] for r in idx], np.float64)
        models = self._models(race_day)
        for j, tgt in enumerate(self.targets):
            out[idx, j] = _scale(tgt, models[tgt].predict(X))
        return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="comma-separated edge-test samples parquet(s)")
    ap.add_argument("--tapes", required=True, help="glob of the tapes (defines the date split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="mid_ticks", help="column prefixes, comma separated")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=400_000)
    a = ap.parse_args(argv)
    df = load_samples(a.samples.split(","))
    prefixes = a.targets.split(",")
    targets = [c for c in df.columns if any(c.startswith(p + "_") for p in prefixes)]
    if not targets:
        raise SystemExit(f"no target columns matching {prefixes}")
    tr, va, te = split_by_date(list_tapes(a.tapes))
    bundle = fit(df, {"train": tr, "val": va, "test": te}, targets, a.folds, a.max_train)
    with open(a.out, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"[forecaster] saved {len(targets)} targets -> {a.out}: {targets}")


if __name__ == "__main__":
    main()
