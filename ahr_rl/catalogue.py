"""Market catalogue (listMarketCatalogue) -> per-runner static features.

The recorder stores one JSON per market in ``betfair stream data/catalogues``:
    {"market_id": "1.26...", "catalogue": {<raw listMarketCatalogue record>}, "runners": {...}}
The live bot gets the same raw record from ``list_market_catalogue`` with the
RUNNER_METADATA / MARKET_DESCRIPTION projections, so both paths share
``static_features``.

Features (per runner, all roughly in [-2, 2], 0 = unknown):
  form_last      finishing position last start / 10 (0 unknown, 1.0 = 10th or worse)
  form_mean3     mean of the last 3 numeric finishes / 10
  form_wins5     wins in the last 5 starts / 5
  form_places5   top-3 finishes in the last 5 starts / 5
  form_starts    number of recorded starts / 10
  first_up       1 if the last form character is 'x' (resuming from a spell)
  draw_rel       barrier / field size
  days_log       log1p(days since last run) / 6
  age            age / 10
  weight_rel     (weight - race mean weight) / 5 kg   (thoroughbreds; 0 otherwise)
  claim          jockey claim / 4 kg
  female         1 for f / m (filly, mare)
  gelding        1 for g
  is_harness     race-level: harness race
  is_flat        race-level: thoroughbred flat race
  distance_km    race-level: distance / 1000, parsed from the market name
  field_size     race-level: runners in the catalogue / 20
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

STATIC_NAMES = ("form_last", "form_mean3", "form_wins5", "form_places5", "form_starts", "first_up",
                "draw_rel", "days_log", "age", "weight_rel", "claim", "female", "gelding",
                "is_harness", "is_flat", "distance_km", "field_size")
N_STATIC = len(STATIC_NAMES)


def _num(x, default=np.nan) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _form_positions(form: str | None) -> tuple[list[int], bool]:
    """'x1191' -> ([1, 1, 9, 1], last_is_spell=False). Digits are finishing
    positions (0 = 10th or worse), 'x' marks a spell; letters (f/l/d...) ignored."""
    if not form:
        return [], False
    pos = [10 if c == "0" else int(c) for c in form if c.isdigit()]
    return pos, form.strip().lower().endswith("x")


def static_features(catalogue: dict, selection_ids) -> np.ndarray:
    """catalogue = raw listMarketCatalogue record; returns [R, N_STATIC] float32
    in the order of ``selection_ids`` (runners missing from it get zeros)."""
    runners = {int(r["selectionId"]): r for r in catalogue.get("runners", [])}
    desc = catalogue.get("description") or {}
    race_type = str(desc.get("raceType") or "").lower()
    m = re.search(r"(\d{3,5})m", str(catalogue.get("marketName") or ""))
    distance = float(m.group(1)) / 1000 if m else 0.0
    field = len(runners)
    weights = [_num((r.get("metadata") or {}).get("WEIGHT_VALUE")) for r in runners.values()]
    weights = [w for w in weights if np.isfinite(w) and w > 0]
    mean_w = float(np.mean(weights)) if weights else np.nan

    out = np.zeros((len(selection_ids), N_STATIC), np.float32)
    for i, sid in enumerate(selection_ids):
        r = runners.get(int(sid))
        if r is None:
            continue
        md = r.get("metadata") or {}
        pos, spell = _form_positions(md.get("FORM"))
        last5 = pos[-5:]
        f = {
            "form_last": pos[-1] / 10 if pos else 0.0,
            "form_mean3": float(np.mean(pos[-3:])) / 10 if pos else 0.0,
            "form_wins5": sum(p == 1 for p in last5) / 5,
            "form_places5": sum(p <= 3 for p in last5) / 5,
            "form_starts": min(len(pos), 20) / 10,
            "first_up": float(spell),
            "draw_rel": _num(md.get("STALL_DRAW"), 0.0) / max(field, 1),
            "days_log": np.log1p(max(_num(md.get("DAYS_SINCE_LAST_RUN"), 0.0), 0.0)) / 6,
            "age": _num(md.get("AGE"), 0.0) / 10,
            "weight_rel": ((_num(md.get("WEIGHT_VALUE")) - mean_w) / 5
                           if np.isfinite(mean_w) and np.isfinite(_num(md.get("WEIGHT_VALUE"))) else 0.0),
            "claim": _num(md.get("JOCKEY_CLAIM"), 0.0) / 4,
            "female": float(str(md.get("SEX_TYPE") or "").lower() in ("f", "m")),
            "gelding": float(str(md.get("SEX_TYPE") or "").lower() == "g"),
            "is_harness": float("harness" in race_type),
            "is_flat": float("flat" in race_type),
            "distance_km": distance,
            "field_size": field / 20,
        }
        out[i] = [np.clip(f[n], -5, 5) for n in STATIC_NAMES]
    return out


def load_catalogue(path: str) -> dict:
    """Recorder file -> raw listMarketCatalogue record."""
    with open(path) as fh:
        d = json.load(fh)
    return d.get("catalogue", d)


def index_catalogues(cat_dir: str) -> dict[str, str]:
    """market_id ('1.262901771') -> catalogue path, from '<...>_1_262901771.json' names."""
    idx = {}
    for p in glob.glob(os.path.join(cat_dir, "**", "*.json"), recursive=True):
        m = re.search(r"_(\d)_(\d+)\.json$", os.path.basename(p))
        if m:
            idx[f"{m.group(1)}.{m.group(2)}"] = p
    return idx


def attach(tapes_dir: str, cat_dir: str, overwrite: bool = False) -> tuple[int, int]:
    """Add static features to already-built tapes in place (no rebuild needed)."""
    from .tape import Tape

    idx = index_catalogues(cat_dir)
    done = missing = 0
    for p in sorted(glob.glob(os.path.join(tapes_dir, "*.npz"))):
        t = Tape.load(p)
        if t.static is not None and not overwrite:
            continue
        cp = idx.get(t.market_id)
        if cp is None:
            missing += 1
            continue
        try:
            t.static = static_features(load_catalogue(cp), t.selection_ids)
        except Exception as e:  # malformed catalogue: leave the tape without static features
            print(f"[catalogue] {os.path.basename(cp)}: {e!r}")
            missing += 1
            continue
        t.save(p)
        done += 1
    return done, missing


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Attach catalogue static features to built tapes")
    ap.add_argument("tapes")
    ap.add_argument("catalogues")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    d, m = attach(a.tapes, a.catalogues, a.overwrite)
    print(f"attached {d} tapes, {m} without a catalogue")
