"""Loading and parsing of Betfair AUS win price files (dwbfpricesauswinDDMMYYYY.csv).

One file per day; one row per runner. The file date is the UTC race date + 1.

Raw columns (lower-cased on load):
    event_id, menu_hint, event_name, event_dt, selection_id, selection_name,
    win_lose, bsp, ppwap, morningwap, ppmax, ppmin, ipmax, ipmin,
    morningtradedvol, pptradedvol, iptradedvol

Timing of each field relative to the off (important for leakage):
    morningwap / morningtradedvol  -> LEAKS THE RESULT (book ~140%, beats BSP at
                                      predicting winners); treat as post-off, do not use
    ppwap / ppmax / ppmin / pptradedvol -> whole pre-play window, final only at the off
    bsp                              -> determined at the off
    ip* , win_lose                   -> post-off; NEVER use as model inputs
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HARNESS_CLASSES = ("Pace", "Trot")
# ipmax/ipmin sentinels when nothing traded in-play
IP_NO_TRADE_MAX, IP_NO_TRADE_MIN = 1.0, 1001.0

_EVENT_RE = r"^R(?P<race_no>\d+)\s+(?P<distance_m>\d+)m\s+(?P<race_class>.+)$"
_MENU_RE = r"^(?P<venue>.*?)\s*\((?P<country>[A-Z]+)\)"
_SEL_RE = r"^(?P<saddlecloth>\d+)\.\s*(?P<runner_name>.+)$"


def load_raw(data_dir: str = "data/raw") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "dwbfpricesauswin*.csv")))
    if not files:
        raise FileNotFoundError(f"no dwbfpricesauswin*.csv files in {data_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.lower()  # older files use upper-case headers
        df["source_file"] = os.path.basename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def parse(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_dt"] = pd.to_datetime(df["event_dt"], format="%d-%m-%Y %H:%M")  # UTC
    df["race_date"] = df["event_dt"].dt.normalize()

    ev = df["event_name"].str.extract(_EVENT_RE)
    df["race_no"] = pd.to_numeric(ev["race_no"])
    df["distance_m"] = pd.to_numeric(ev["distance_m"])
    df["race_class"] = ev["race_class"].str.strip()
    df["race_code"] = np.where(
        df["race_class"].str.startswith(HARNESS_CLASSES, na=False), "Harness", "Thoroughbred"
    )

    mh = df["menu_hint"].str.extract(_MENU_RE)
    df["venue"] = mh["venue"].str.strip()

    sel = df["selection_name"].str.extract(_SEL_RE)
    df["saddlecloth"] = pd.to_numeric(sel["saddlecloth"])
    df["runner_name"] = sel["runner_name"].str.strip()

    for c in ("event_id", "selection_id"):
        df[c] = df[c].astype("int64")
    df["win_lose"] = df["win_lose"].astype("int8")

    df["ip_traded"] = ~((df["ipmax"] == IP_NO_TRADE_MAX) & (df["ipmin"] == IP_NO_TRADE_MIN))
    # same sentinel pattern for the pre-play range: no trade in the window -> NaN
    pp_none = (df["ppmax"] == IP_NO_TRADE_MAX) & (df["ppmin"] == IP_NO_TRADE_MIN)
    df["pp_range_missing"] = pp_none
    df.loc[pp_none, ["ppmax", "ppmin"]] = np.nan
    # morningwap == 1.0 coincides with zero morning volume -> NaN
    df.loc[df["morningtradedvol"] == 0, "morningwap"] = np.nan
    df["field_size"] = df.groupby("event_id")["selection_id"].transform("size")
    # selection_id is NOT a stable horse id in these files (the same horse appears under
    # several ids), so use name + code as the best available runner key
    df["runner_key"] = df["race_code"] + "|" + df["runner_name"].str.lower()
    return df


def load(data_dir: str = "data/raw") -> pd.DataFrame:
    df = parse(load_raw(data_dir))
    return df.drop_duplicates(["event_id", "selection_id"]).reset_index(drop=True)


def implied_probs(df: pd.DataFrame, price_col: str, normalise: bool = True) -> pd.Series:
    """1/price, optionally normalised to sum to 1 within each market."""
    p = 1.0 / df[price_col]
    if normalise:
        p = p / p.groupby(df["event_id"]).transform("sum")
    return p
