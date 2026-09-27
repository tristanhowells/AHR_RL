"""Held-out evaluation: trained policy vs baselines on the TEST split.

    python -m ahr_rl.report --run runs/ppo            # uses runs/ppo/best.pt + split.json
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from .evaluate import (evaluate, make_random_policy, make_scalper_policy, make_torch_policy, noop_policy,
                       summarise)
from .live.bot import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-races", type=int, default=100000)
    a = ap.parse_args()
    split = json.load(open(os.path.join(a.run, "split.json")))[a.split][: a.max_races]
    model, cfg = load_model(os.path.join(a.run, a.ckpt))
    rows, frames = {}, []
    for name, pol in [("agent", make_torch_policy(model, True)), ("do_nothing", noop_policy),
                      ("random_1pct", make_random_policy(0.01)), ("scalper_rule", make_scalper_policy())]:
        df = evaluate(pol, split, cfg)
        df.insert(0, "policy", name)
        frames.append(df)
        rows[name] = summarise(df)
        print(name, {k: round(v, 2) for k, v in rows[name].items()}, flush=True)
    out = pd.DataFrame(rows).T
    out.to_csv(os.path.join(a.run, f"report_{a.split}.csv"))
    pd.concat(frames).to_csv(os.path.join(a.run, f"report_{a.split}_races.csv"), index=False)
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
