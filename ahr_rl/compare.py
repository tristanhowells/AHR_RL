"""Compare trained agents (discrete-bracket PPO, continuous PPO, continuous SAC)
and baselines on the same held-out TEST races.

    python -m ahr_rl.compare --name synthetic \\
        --discrete runs/synthetic4 --ppo-cont runs/syn_ppo_cont --sac runs/syn_sac
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from .continuous import Actor, make_policy_fn
from .env_continuous import ContinuousAllocEnv, ContinuousConfig
from .evaluate import (evaluate, make_random_policy, make_scalper_policy, make_torch_policy, noop_policy,
                       summarise)
from .live.bot import load_model


def load_actor(run):
    ck = torch.load(os.path.join(run, "best.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    actor = Actor(a["d_model"], a["n_layers"])
    actor.load_state_dict(ck["actor"])
    actor.eval()
    c = ck["cfg"]
    cfg = ContinuousConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items() if k != "exchange"})
    return actor, cfg, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--discrete")
    ap.add_argument("--ppo-cont")
    ap.add_argument("--sac")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="runs/compare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [r for r in (a.discrete, a.ppo_cont, a.sac) if r]
    splits = [json.load(open(os.path.join(r, "split.json")))[a.split] for r in runs]
    assert all(s == splits[0] for s in splits), "runs must share the same test races"
    races = splits[0]
    rows = {}

    def add(label, df):
        s = summarise(df)
        rows[label] = s
        print(label, {k: round(v, 2) for k, v in s.items()}, flush=True)

    rng_cont = np.random.default_rng(0)

    def rand_cont(obs, env):
        x = rng_cont.uniform(-1, 1, 25)
        x[:24] *= obs["mask"]
        x[-1] = 0.02
        return x

    add("do_nothing", evaluate(noop_policy, races))
    add("random (bracket spec)", evaluate(make_random_policy(0.01), races))
    add("random (continuous spec, f=0.02)", evaluate(rand_cont, races, ContinuousConfig(), env_cls=ContinuousAllocEnv))
    add("rule scalper (bracket spec)", evaluate(make_scalper_policy(), races))
    if a.discrete:
        model, cfg = load_model(os.path.join(a.discrete, "best.pt"))
        add("PPO - discrete bracket spec", evaluate(make_torch_policy(model, True), races, cfg))
    for label, run in (("PPO - continuous spec", a.ppo_cont), ("SAC - continuous spec", a.sac)):
        if run:
            actor, cfg, _ = load_actor(run)
            add(label, evaluate(make_policy_fn(actor), races, cfg, env_cls=ContinuousAllocEnv))
    df = pd.DataFrame(rows).T
    df.to_csv(os.path.join(a.out, f"{a.name}_{a.split}.csv"))
    print(df[["races", "mean_green_$", "pct_green", "mean_expected_$", "mean_turnover_$", "worst_race_$"]]
          .round(2).to_string())


if __name__ == "__main__":
    main()
