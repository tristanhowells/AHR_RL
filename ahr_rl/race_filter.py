"""Restrict a study to a kind of race (the race study, 2h, found harness markets
untradeable before the off: costs 5.6% vs 2.5% for thoroughbreds, and even perfect
hindsight on direction only breaks even).

Race type and metro / non-metro come from, in order of preference:
  1  the market catalogues (--catalogues folder; description.raceType, event.venue)
  2  the catalogue features attached to the tape (is_flat / is_harness, cell V2.1)
  3  a race table from the race study (--races-csv, races.csv written by cell 2h)
Races whose type can't be resolved are dropped when a filter is active.

Every study that accepts these options adds the same three flags:
  --race-type {all,flat,harness}  --metro-only  --catalogues DIR  --races-csv FILE
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .catalogue import STATIC_NAMES, index_catalogues, load_catalogue
from .race_study import METRO

RACE_TYPES = ("all", "flat", "harness")


def add_args(ap) -> None:
    g = ap.add_argument_group("race filter (see race_filter.py)")
    g.add_argument("--race-type", default="all", choices=RACE_TYPES, help="flat = thoroughbreds only")
    g.add_argument("--metro-only", action="store_true", help="metropolitan tracks only")
    g.add_argument("--catalogues", default=None, help="folder of market catalogue *.json")
    g.add_argument("--races-csv", default=None, help="races.csv from the race study (cell 2h)")


def _stem(path: str) -> str:
    b = os.path.basename(path)
    for ext in (".npz", ".ndjson.gz", ".ndjson"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return b


def _market_id(stem: str) -> str | None:
    parts = stem.rsplit("_", 2)
    return f"{parts[-2]}.{parts[-1]}" if len(parts) == 3 and parts[-2].isdigit() and parts[-1].isdigit() else None


def race_info(paths, catalogues: str | None = None, races_csv: str | None = None) -> pd.DataFrame:
    """race (file stem) -> race_type ('Flat' / 'Harness' / 'Unknown'), venue, metro."""
    cat_idx = index_catalogues(catalogues) if catalogues else {}
    table = {}
    if races_csv and os.path.exists(races_csv):
        r = pd.read_csv(races_csv, usecols=lambda c: c in ("race", "race_type", "venue"))
        table = {row.race: (row.race_type, row.venue) for row in r.itertuples(index=False)}
    rows = []
    for p in paths:
        stem = _stem(p)
        venue = stem.split("_")[2] if stem.count("_") >= 3 else "?"
        rtype = "Unknown"
        mid = _market_id(stem)
        if mid and mid in cat_idx:
            try:
                c = load_catalogue(cat_idx[mid])
                rtype = (c.get("description") or {}).get("raceType") or rtype
                venue = (c.get("event") or {}).get("venue") or venue
            except Exception:
                pass
        if rtype == "Unknown" and p.endswith(".npz") and os.path.exists(p):
            try:
                z = np.load(p, allow_pickle=False)
                if "static" in z.files:
                    st = z["static"]
                    if st[:, STATIC_NAMES.index("is_harness")].max() > 0:
                        rtype = "Harness"
                    elif st[:, STATIC_NAMES.index("is_flat")].max() > 0:
                        rtype = "Flat"
            except Exception:
                pass
        if rtype == "Unknown" and stem in table:
            rtype, venue = table[stem][0], table[stem][1] if isinstance(table[stem][1], str) else venue
        rows.append(dict(race=stem, race_type=rtype, venue=venue,
                         metro=str(venue).lower() in METRO))
    return pd.DataFrame(rows, columns=["race", "race_type", "venue", "metro"])


def keep_races(paths, race_type: str = "all", metro_only: bool = False, catalogues: str | None = None,
               races_csv: str | None = None, log=print) -> set[str]:
    """The race stems (file names without extension) that pass the filter."""
    info = race_info(paths, catalogues, races_csv)
    m = np.ones(len(info), bool)
    if race_type != "all":
        m &= (info["race_type"].str.lower() == race_type).to_numpy()
    if metro_only:
        m &= info["metro"].to_numpy(bool)
    if race_type != "all" or metro_only:
        counts = info["race_type"].value_counts().to_dict()
        log(f"race filter: race type {race_type}{', metro only' if metro_only else ''} -> {int(m.sum())} of "
            f"{len(info)} races kept (types found: {counts})")
    return set(info.loc[m, "race"])


def filter_paths(paths, args, log=print) -> list[str]:
    """Apply the CLI race filter to a list of tape / recording paths."""
    if getattr(args, "race_type", "all") == "all" and not getattr(args, "metro_only", False):
        return list(paths)
    keep = keep_races(paths, args.race_type, args.metro_only, args.catalogues, args.races_csv, log)
    return [p for p in paths if _stem(p) in keep]


def tag(args) -> str:
    """Suffix for output folders, e.g. '_flat' or '_flat_metro'."""
    t = "" if getattr(args, "race_type", "all") == "all" else f"_{args.race_type}"
    return t + ("_metro" if getattr(args, "metro_only", False) else "")
