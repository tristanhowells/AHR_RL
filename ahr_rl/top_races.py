"""P18: a pure black box on the deepest markets only. Per-runner SAC, objective: a green book at the off.

The races with the most money matched (by default the top 10 on file, found by scanning the tapes)
are the cheapest and deepest markets there are (~1-2% to cross, $150k-$270k matched). The agent:
P17's per-runner agents with the full execution menu and nothing hand-made in the way -

  * every runner, every 10s: side, size ($5 - --order-cap at risk, <= --size-frac-cap of the money
    shown) and price (cross now, or rest 1..--max-offset ticks better than the best price); a
    repeated order keeps its queue place; |side| below the dead-zone cancels
  * no liquidity gate (all of these markets are deep), no signals, no BSP exit: the objective is a
    GREEN book before the off. From the scheduled start anything still open is greened by crossing
    (the env's safety net), and the race is scored on its worst-case outcome (its green value)
  * reward: each runner's own share of the green value; long random-strategy warm-up first

Resumable: each fold saves its scores; re-running the same command skips finished folds.

Ten races cannot carry a chronological train / validation / test split, so it is cross-validated:
the races (in date order) are cut into --folds folds; each fold trains on the others less one
(the validation race, for picking the checkpoint) and is scored ONCE on its held-out races. Every
race gets exactly one out-of-sample score; they are pooled. Each fold's checkpoint is also scored
on its own training races: the gap between the two is how much it memorised.

Read the result for what it is: an out-of-sample mean over 10 races has a huge error bar, and a
black box with this much freedom on this little data mostly learns the training races. It shows
what the agent does in the deepest markets and whether anything carries over; it cannot prove an
edge. For that, widen the set (--top 50 / 100) and the same code gives a far tighter estimate.

    python -m ahr_rl.top_races --tapes "data/tapes/*.npz" --out runs/top_races --top 10
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from .blackbox import forensics, race_t, score
from .env import list_tapes
from .evaluate import evaluate
from .passive_agents import N_ACT3, RestingAgentConfig, RestingAgentEnv, noop_policy, random_policy
from .runner_agents import load_runner_actor, make_runner_policy, train
from .tape import Tape


def matched_at_off(path: str):
    """(path, $ matched at the scheduled start, venue/name) - the tape's market total, single-counted."""
    try:
        t = Tape.load(path)
    except Exception:
        return path, np.nan
    if not t.went_in_play:
        return path, np.nan
    after = np.where(t.t_rel >= 0)[0]
    s = int(after[0]) if len(after) else t.n_steps - 1
    return path, float(np.nanmax(t.total_matched[: s + 1])) if s >= 0 else np.nan


def top_races(paths, n: int, workers: int, races_csv: str | None = None) -> pd.DataFrame:
    if races_csv and os.path.exists(races_csv):
        r = pd.read_csv(races_csv, usecols=["race", "matched_$"])
        stem = {os.path.basename(p).split(".npz")[0]: p for p in paths}
        r = r[r["race"].isin(stem)]
        r["path"] = r["race"].map(stem)
        d = r.rename(columns={"matched_$": "matched"})[["race", "path", "matched"]]
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            res = list(pool.map(matched_at_off, paths, chunksize=8))
        d = pd.DataFrame(res, columns=["path", "matched"])
        d["race"] = d["path"].map(lambda p: os.path.basename(p).split(".npz")[0])
    return d.dropna().sort_values("matched", ascending=False).head(n).reset_index(drop=True)


def folds_of(races: list[str], k: int) -> list[list[str]]:
    """k contiguous folds in date order (so a test race is never sandwiched in its own training days)."""
    races = sorted(races)
    return [list(f) for f in np.array_split(np.array(races, dtype=object), k) if len(f)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=10, help="how many of the highest-volume races")
    ap.add_argument("--races-csv", default=None, help="race study races.csv (matched_$); default: scan the tapes")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200_000, help="env steps per fold")
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-offset", type=int, default=5, help="rest up to this many ticks better than the best price")
    ap.add_argument("--order-cap", type=float, default=20.0)
    ap.add_argument("--runner-cap", type=float, default=50.0)
    ap.add_argument("--market-cap", type=float, default=150.0)
    ap.add_argument("--size-frac-cap", type=float, default=0.2)
    ap.add_argument("--decision-every", type=int, default=20, help="tape steps (0.5s) per decision; 20 = 10s")
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=max(4, os.cpu_count() or 4))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    if a.device == "cpu":
        print("WARNING: no GPU. Training will be very slow. Use a GPU runtime.", flush=True)
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - a.n_envs))

    top = top_races(list_tapes(a.tapes), a.top, a.workers, a.races_csv)
    top.to_csv(os.path.join(a.out, "races.csv"), index=False)
    print(f"=== P18: the {len(top)} highest-volume races on file (matched at the off, single-counted) ===")
    print(top[["race", "matched"]].round(0).to_string(index=False), flush=True)
    if len(top) < 3:
        raise SystemExit("need at least 3 races")
    path = dict(zip(top["race"], top["path"]))

    cfg = RestingAgentConfig(random_start_s=0.0, decision_every=a.decision_every, min_best_usd=0.0,
                             order_cap=a.order_cap, runner_cap=a.runner_cap, market_cap=a.market_cap,
                             size_frac_cap=a.size_frac_cap, max_offset=a.max_offset, exit_bsp=False)
    pess = replace(cfg, exchange=replace(cfg.exchange, queue_mult=2.0, trade_fill_frac=0.5))
    slow = replace(cfg, exchange=replace(cfg.exchange, latency_steps=2))
    print(f"\nper runner every {a.decision_every * 0.5:.0f}s: cross or rest 1-{a.max_offset} ticks better, $5-"
          f"${a.order_cap:.0f} at risk (<= {a.size_frac_cap:.0%} of the money shown); objective: green at the off "
          f"(open positions greened by crossing from the start); {a.folds}-fold cross-validation, {a.steps} steps per "
          f"fold ({a.device})\n", flush=True)

    folds = folds_of(list(top["race"]), min(a.folds, len(top)))
    races_sorted = sorted(top["race"])
    rows, fz_all = [], []
    for i, test in enumerate(folds):
        rest = [r for r in races_sorted if r not in test]
        val = [rest[-1]]  # the latest non-test race picks the checkpoint
        trn = [r for r in rest if r not in val]
        print(f"--- fold {i + 1}/{len(folds)}: train {len(trn)}, val {val[0]}, test {', '.join(test)} ---", flush=True)
        out = os.path.join(a.out, f"fold{i + 1}")
        done = os.path.join(out, "per_race.csv")
        if os.path.exists(done):  # resumable: a fold finished in an earlier session is not retrained
            rows += pd.read_csv(done).to_dict(orient="records")
            if os.path.exists(os.path.join(out, "fills.csv")):
                fz_all.append(pd.read_csv(os.path.join(out, "fills.csv")))
            print("  already done (loaded)", flush=True)
            continue
        split = ([path[r] for r in trn], [path[r] for r in val], [path[r] for r in test])
        ck = train("", out, cfg, a.steps, int(a.steps * a.warmup_frac), a.seed + i, a.n_envs, a.device,
                   buffer=a.buffer, eval_races=1, env_cls=RestingAgentEnv, n_act=N_ACT3, split=split)
        pol = make_runner_policy(load_runner_actor(ck, a.device), a.device)
        te_paths, tr_paths = split[2], split[0]
        for name, pf, c, races_, kind in (("agent", pol, cfg, te_paths, "test"),
                                          ("agent, 1s latency", pol, slow, te_paths, "test"),
                                          ("agent, pessimistic fills", pol, pess, te_paths, "test"),
                                          ("do nothing", noop_policy, cfg, te_paths, "test"),
                                          ("random", random_policy(seed=i), cfg, te_paths, "test"),
                                          ("agent on its OWN training races", pol, cfg, tr_paths, "train")):
            df = evaluate(pf, races_, c, env_cls=RestingAgentEnv)
            for r in df.itertuples():
                rows.append(dict(fold=i + 1, policy=name, kind=kind, race=r.market, green=r.worst,
                                 turnover=r.turnover, fills=r.n_fills))
        fz = forensics(pol, te_paths, cfg, env_cls=RestingAgentEnv)
        if len(fz):
            fz_all.append(fz.assign(fold=i + 1))
            fz_all[-1].to_csv(os.path.join(out, "fills.csv"), index=False)
        pd.DataFrame([x for x in rows if x["fold"] == i + 1]).to_csv(done, index=False)
        print(pd.DataFrame([x for x in rows if x["fold"] == i + 1]).groupby("policy", sort=False)[
            ["green", "turnover"]].mean().round(3).to_string(), flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "per_race.csv"), index=False)
    summ = res.groupby("policy", sort=False).agg(races=("race", "nunique"), mean_green=("green", "mean"),
                                                 pct_green=("green", lambda g: (g >= -0.005).mean() * 100),
                                                 worst_race=("green", "min"), best_race=("green", "max"),
                                                 turnover=("turnover", "mean"))
    summ["t"] = [race_t(res.loc[res["policy"] == p, "green"]) for p in summ.index]
    print("\n=== Out-of-sample (each race scored once, by the fold that never trained on it); $ per race, $500 bank ===")
    print(summ.round(3).to_string())
    oos = res[res["policy"] == "agent"]
    ins = res[res["policy"] == "agent on its OWN training races"]
    print(f"\nmemorisation gap: {ins['green'].mean():+.2f} $/race on its own training races vs {oos['green'].mean():+.2f} "
          f"on races it never saw")
    if fz_all:
        fz = pd.concat(fz_all, ignore_index=True)
        fz.to_csv(os.path.join(a.out, "fills.csv"), index=False)
        print(f"\n=== What the agents trade on their held-out races ({len(fz)} fills) ===")
        print(fz.pivot_table(index="rank", columns="side", values="stake", aggfunc="sum", fill_value=0).round(0).to_string())
        print(fz.pivot_table(index="time", columns="side", values="stake", aggfunc="sum", fill_value=0,
                             observed=False).round(0).to_string())
        print(f"resting (passive) fills: {fz.loc[fz['passive'], 'stake'].sum() / max(fz['stake'].sum(), 1e-9):.0%} of $ "
              f"matched")
    m = summ.loc["agent"]
    stress_ok = all(summ.loc[p, "mean_green"] > 0 for p in ("agent, 1s latency", "agent, pessimistic fills"))
    edge_hint = bool(m["mean_green"] > 0 and (m["t"] or 0) > 2 and stress_ok)
    verdict = dict(races=list(top["race"]), out_of_sample=summ.to_dict(orient="index"),
                   memorisation_gap=float(ins["green"].mean() - oos["green"].mean()),
                   promising=edge_hint,
                   note=f"{len(top)} races: an out-of-sample mean this small cannot establish an edge; widen with --top")
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    print("\n" + ("PROMISING (out of sample, t > 2, survives both stresses) - but on this few races, re-run with "
                  "--top 50 or more before believing it" if edge_hint else
                  "NOTHING THAT CARRIES OVER: the agents do not beat doing nothing on races they never trained on "
                  "(with both stresses)"))
    print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
