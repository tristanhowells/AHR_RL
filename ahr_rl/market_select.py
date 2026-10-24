"""Pick training markets by size: $ matched at the scheduled start (the tape's market total, single-counted,
the same measure as the race study's matched_$). Known only once the race starts, so it is for choosing
TRAINING races; a live strategy needs something knowable beforehand (matched so far, class, venue)."""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .tape import Tape


def matched_at_off(path: str):
    """(path, $ matched at the scheduled start); NaN for races that never went in play."""
    try:
        t = Tape.load(path)
    except Exception:
        return path, np.nan
    if not t.went_in_play:
        return path, np.nan
    after = np.where(t.t_rel >= 0)[0]
    s = int(after[0]) if len(after) else t.n_steps - 1
    return path, float(np.nanmax(t.total_matched[: s + 1]))


def market_sizes(paths, workers: int, races_csv: str | None = None) -> pd.DataFrame:
    """race, path, matched for every tape (from races.csv if given, else by scanning the tapes)."""
    stem = {os.path.basename(p).split(".npz")[0]: p for p in paths}
    if races_csv and os.path.exists(races_csv):
        r = pd.read_csv(races_csv, usecols=["race", "matched_$"])
        r = r[r["race"].isin(stem)]
        r["path"] = r["race"].map(stem)
        d = r.rename(columns={"matched_$": "matched"})[["race", "path", "matched"]]
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        with ProcessPoolExecutor(max(1, workers), mp_context=mp.get_context("spawn")) as pool:
            res = list(pool.map(matched_at_off, paths, chunksize=8))
        d = pd.DataFrame(res, columns=["path", "matched"])
        d["race"] = d["path"].map(lambda p: os.path.basename(p).split(".npz")[0])
    return d.dropna().sort_values("matched", ascending=False).reset_index(drop=True)


def select_markets(paths, min_matched: float, workers: int, races_csv: str | None = None) -> list[str]:
    """Tape paths with at least min_matched $ matched at the scheduled start, in date order."""
    d = market_sizes(paths, workers, races_csv)
    return sorted(d.loc[d["matched"] >= min_matched, "path"])
