"""Held-out evaluation: trained policy vs baselines, broken down by segment,
plus the deployment gate.

    python -m ahr_rl.report --run runs/ppo            # uses runs/ppo/best.pt + split.json

Segments (a profitable niche can hide inside a losing average, and vice versa):
  race_type  harness / flat / unknown (from catalogue static features)
  liquidity  tercile of total matched at the off, within the evaluated races
  field      small (<= 8 runners) / large

Gate (written to <run>/gate.json): the agent passes only if, on the TEST split,
its mean guaranteed profit per race is > 0 with t > 2 across races. The live
bot refuses to trade real money with a checkpoint that has not passed.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .catalogue import STATIC_NAMES
from .evaluate import (evaluate, make_random_policy, make_scalper_policy, make_torch_policy, noop_policy,
                       summarise)
from .live.bot import load_model
from .tape import Tape

GATE_T = 2.0


def race_attributes(paths: list[str]) -> pd.DataFrame:
    rows = []
    ih, iff = STATIC_NAMES.index("is_harness"), STATIC_NAMES.index("is_flat")
    for p in paths:
        t = Tape.load(p)
        rtype = "unknown"
        if t.static is not None and len(t.static):
            rtype = "harness" if t.static[:, ih].max() > 0 else "flat" if t.static[:, iff].max() > 0 else "unknown"
        rows.append(dict(market=t.name, race_type=rtype, matched=float(t.total_matched[-1]),
                         field=int(t.active[0].sum())))
    df = pd.DataFrame(rows)
    df["liquidity"] = pd.qcut(df["matched"].rank(method="first"), 3, labels=["low", "mid", "high"]) \
        if len(df) >= 3 else "all"
    df["field_size"] = np.where(df["field"] <= 8, "small (<=8)", "large (9+)")
    return df


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    live = df[~df["void"]]
    rows = []
    for seg in ("race_type", "liquidity", "field_size"):
        for val, g in live.groupby(seg, observed=True):
            rows.append(dict(segment=seg, value=str(val), races=len(g), usd_per_race=g["worst"].mean(),
                             tstat=tstat(g["worst"].values), pct_green=(g["worst"] >= -0.005).mean() * 100,
                             trades_per_race=g["n_opens"].mean()))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-races", type=int, default=100000)
    a = ap.parse_args()
    split = json.load(open(os.path.join(a.run, "split.json")))[a.split][: a.max_races]
    model, cfg = load_model(os.path.join(a.run, a.ckpt))
    attrs = race_attributes(split)
    rows, frames = {}, []
    for name, pol in [("agent", make_torch_policy(model, True)), ("do_nothing", noop_policy),
                      ("random_1pct", make_random_policy(0.01)), ("scalper_rule", make_scalper_policy())]:
        df = evaluate(pol, split, cfg).merge(attrs, on="market", how="left")
        df.insert(0, "policy", name)
        frames.append(df)
        rows[name] = summarise(df)
        rows[name]["tstat_races"] = tstat(df.loc[~df["void"], "worst"].values)
        print(name, {k: round(v, 2) for k, v in rows[name].items()}, flush=True)
    pd.set_option("display.width", 200)
    out = pd.DataFrame(rows).T
    out.to_csv(os.path.join(a.run, f"report_{a.split}.csv"))
    races = pd.concat(frames)
    races.to_csv(os.path.join(a.run, f"report_{a.split}_races.csv"), index=False)
    print(out.round(2).to_string())

    seg = segment_table(frames[0])
    seg.to_csv(os.path.join(a.run, f"report_{a.split}_segments.csv"), index=False)
    print("\n=== Agent by segment (guaranteed $/race; t across races) ===")
    print(seg.round(3).to_string(index=False))
    print("Note: a positive segment found here is a hypothesis, not a result - it was not chosen in advance.\n"
          "Confirm it on races recorded after this run before trading it.")

    if a.split == "test":
        ag = rows["agent"]
        gate = {"passed": bool(ag["mean_green_$"] > 0 and ag["tstat_races"] > GATE_T),
                "mean_green_usd_per_race": ag["mean_green_$"], "tstat_races": ag["tstat_races"],
                "races": ag["races"], "checkpoint": a.ckpt, "rule": f"mean > 0 and t > {GATE_T} on test"}
        json.dump(gate, open(os.path.join(a.run, "gate.json"), "w"), indent=1)
        print("\n=== Deployment gate ===")
        print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
