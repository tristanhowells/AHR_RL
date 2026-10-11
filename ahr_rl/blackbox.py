"""P8: a "black box" search. SAC with a long, consequence-free random warm-up.

No hand-made signal or rule: the agent sees the env features, acts in the continuous
allocation spec (back / lay any runner, any size up to a cap, every 10s) and is paid
the change in its green value. Before it learns anything it spends --warmup-frac of
its steps trying random strategies ("mixed": half the races a fresh random action
every step, half a random state-dependent strategy held for the whole race; see
continuous.RandomStrategies). Everything is simulated, so the warm-up costs nothing.
Whatever the critic finds in that data, it can then refine.

Because a black box only means something if it can find an edge that IS there:

  0  positive control   the same agent, same settings, on synthetic races with a
                        small planted edge (one runner shortens a tick every
                        --trend-every-s seconds). It must clearly beat doing nothing
                        on held-out synthetic races, or the real-data result is
                        uninformative.
  1  real races         --seeds independent runs, checkpoint picked on VAL races
  2  TEST races         (never used for training or selection): each seed vs do
                        nothing and vs random trading, with a race-clustered t; plus
                        a latency stress (1s instead of 0.5s) and a forensic table of
                        what the agent actually does (side, favourite rank, time).

Liquidity gate (--min-best-usd auto, or a $ value; default 0 = off): the agent sees every runner in every
race but may only open a position on a runner while the money at its best back + lay
prices (10s trailing mean) is at least the cut. It may still trade a held runner below
the cut, but only to reduce the position. "auto" = P11's top-quintile boundary: the 80th
percentile of that measure over all runners in the last 5 min on TRAIN days. 0 = no gate.

"edge" = every seed beats do-nothing on TEST with t > 2 AND survives the latency
stress. A strategy found by a black box that fails either is most likely an
artefact of the simulator or of the particular training races.

    python -m ahr_rl.blackbox --tapes "data/tapes/*.npz" --out runs/blackbox
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from .continuous import Actor, make_policy_fn, parse_args as sac_args, train_sac
from .env import list_tapes, split_by_date
from .env_continuous import N_CONT, ContinuousAllocEnv, ContinuousConfig
from .evaluate import evaluate, summarise
from .features import R_MAX
from .synthetic import write_synthetic
from .tape import Tape


def noop_policy(obs, env):
    return np.zeros(N_CONT, np.float32)


def random_cont_policy(f: float = 0.02, seed: int = 0):
    rng = np.random.default_rng(seed)

    def pol(obs, env):
        a = rng.uniform(-1, 1, N_CONT).astype(np.float32)
        a[:R_MAX] *= obs.get("act_mask", obs["mask"])
        a[-1] = f
        return a

    return pol


def best_usd_trailing(t: Tape, window_s: float = 10.0) -> np.ndarray:
    """[T, R] money at the best back + lay prices, trailing mean over window_s (live)."""
    best = t.back_size[:, :, 0].astype(np.float64) + t.lay_size[:, :, 0]
    k = max(1, int(round(window_s / t.dt)))
    c = np.cumsum(np.vstack([np.zeros((1, best.shape[1])), best]), 0)
    idx = np.arange(1, len(best) + 1)
    lo = np.maximum(idx - k, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


def liquidity_samples(paths, every_s: float = 10.0, last_s: float = 300.0, window_s: float = 10.0):
    """Gate measure for every active priced runner every every_s in the last last_s before the start."""
    out = []
    for p in paths:
        t = Tape.load(p)
        b = best_usd_trailing(t, window_s)
        priced = t.active & (t.back_tick[:, :, 0] >= 0) & (t.lay_tick[:, :, 0] >= 0) & ~t.suspended[:, None]
        steps = np.where((t.t_rel >= -last_s) & (t.t_rel <= 0))[0][:: max(1, int(round(every_s / t.dt)))]
        for s in steps:
            out.append((os.path.basename(p), b[s][priced[s]]))
    return out


def liquidity_cut(paths, q: float = 0.8, window_s: float = 10.0) -> float:
    v = np.concatenate([x for _, x in liquidity_samples(paths, window_s=window_s)] or [np.zeros(0)])
    return float(np.quantile(v, q)) if len(v) else 0.0


def gate_summary(paths, cut: float, window_s: float = 10.0) -> dict:
    """How often the gate is open on these races (last 5 min)."""
    if cut <= 0:
        return dict(cut=0.0)
    smp = liquidity_samples(paths, window_s=window_s)
    allv = np.concatenate([x for _, x in smp] or [np.zeros(0)])
    races = {}
    for name, x in smp:
        races[name] = max(races.get(name, 0), int((x >= cut).sum()))
    n_open = np.array([v for _, x in smp for v in [int((x >= cut).sum())]])
    return dict(cut=cut, runner_share_open=float((allv >= cut).mean()) if len(allv) else np.nan,
                races=len(races), races_with_any_open=int(sum(v > 0 for v in races.values())),
                mean_open_runners=float(n_open.mean()) if len(n_open) else np.nan)


def load_actor(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    ar = ck["args"]
    actor = Actor(ar["d_model"], ar["n_layers"], runner_extra=ar.get("runner_extra", 0)).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor


def race_t(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def score(name: str, df: pd.DataFrame) -> dict:
    s = summarise(df)
    live = df[~df["void"]]
    return dict(policy=name, races=s["races"], mean_green=s["mean_green_$"], t=race_t(live["worst"]),
                pct_green=s["pct_green"], mean_expected=s["mean_expected_$"], turnover=s["mean_turnover_$"],
                worst_race=float(live["worst"].min()) if len(live) else np.nan)


def forensics(pol, tapes, cfg, env_cls=ContinuousAllocEnv) -> pd.DataFrame:
    """What the agent trades: every fill by side, favourite rank and time to the start."""
    env = env_cls(tapes, cfg, cache_tapes=False)
    rows = []
    for tp in tapes:
        obs, _ = env.reset(options={"tape": tp})
        done = False
        while not done:
            obs, _, done, _, info = env.step(pol(obs, env))
        t = env.tape
        liq = best_usd_trailing(t, cfg.liq_window_s)
        for f in env.ex.fills:
            s = min(f.step, t.n_steps - 1)
            bt, lt = t.back_tick[s, :, 0], t.lay_tick[s, :, 0]
            mid = np.where((bt >= 0) & (lt >= 0), (bt + lt) / 2.0, np.inf)
            rank = int((mid < mid[f.runner]).sum()) + 1
            rows.append(dict(market=t.name, side="back" if f.side == 0 else "lay", rank=min(rank, 7),
                             best_usd=float(liq[s, f.runner]),
                             t=float(t.t_rel[s]), stake=f.stake, price=f.price,
                             race_green=float(info.get("worst", np.nan))))
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["time"] = pd.cut(d["t"], [-1e9, -300, -120, -30, 1e9], labels=["10-5m", "5-2m", "2m-30s", "last 30s"])
    return d


def run_sac(tapes_glob, out, steps, warmup_frac, seed, n_envs, device, max_fraction, buffer, eval_races,
            decision_every, update_every=2, grad_steps=1, min_best_usd=0.0, gate_inputs=True):
    argv = ["--algo", "sac", "--tapes", tapes_glob, "--out", out, "--total-steps", str(steps),
            "--n-envs", str(n_envs), "--seed", str(seed), "--device", device,
            "--warmup", str(int(steps * warmup_frac)), "--warmup-mode", "mixed",
            "--max-fraction", str(max_fraction), "--decision-every", str(decision_every),
            "--min-best-usd", str(min_best_usd), "--buffer", str(buffer),
            "--update-every", str(update_every), "--grad-steps", str(grad_steps),
            "--eval-races", str(eval_races),
            "--eval-every", str(max(1, int(steps * (1 - warmup_frac)) // n_envs // 10))]  # ~10 after the warm-up
    if not gate_inputs:
        argv.append("--no-gate-inputs")
    train_sac(sac_args(argv))
    return os.path.join(out, "best.pt")


def test_block(label, ckpt, test, cfg, device, rows, stress=True):
    """Scores the VAL-picked checkpoint (and its latency stress); returns (actor, agent row, stress row).
    The last checkpoint is scored too, for information only: it shows what training ended with even when
    nothing beat doing nothing on VAL."""
    actor = load_actor(ckpt, device)
    pol = make_policy_fn(actor, device)
    main = dict(run=label, **score("agent", evaluate(pol, test, cfg, env_cls=ContinuousAllocEnv)))
    rows.append(main)
    slow_row = None
    if stress:
        slow = replace(cfg, exchange=replace(cfg.exchange, latency_steps=2))
        slow_row = dict(run=label, **score("agent, 1s latency", evaluate(pol, test, slow, env_cls=ContinuousAllocEnv)))
        rows.append(slow_row)
    last = os.path.join(os.path.dirname(ckpt), "last.pt")
    if os.path.exists(last):
        lp = make_policy_fn(load_actor(last, device), device)
        rows.append(dict(run=label, **score("last checkpoint (info)", evaluate(lp, test, cfg,
                                                                             env_cls=ContinuousAllocEnv))))
    return actor, main, slow_row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600_000, help="env steps per real-data seed")
    ap.add_argument("--warmup-frac", type=float, default=0.4, help="share of steps spent on random strategies")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--max-fraction", type=float, default=0.1, help="cap on the share of funds wagered per decision")
    ap.add_argument("--decision-every", type=int, default=20, help="tape steps (0.5s) per decision; 20 = 10s")
    ap.add_argument("--update-every", type=int, default=2, help="SAC: env iterations per gradient update round")
    ap.add_argument("--grad-steps", type=int, default=1, help="SAC: gradient steps per update round")
    ap.add_argument("--min-best-usd", default="0",
                    help='liquidity gate in $ at best back + lay; "auto" = P11 top quintile on TRAIN days; 0 = off')
    ap.add_argument("--no-gate-inputs", action="store_true",
                    help="gated, but without the per-runner (above cut, distance to cut) network inputs")
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--control-steps", type=int, default=150_000)
    ap.add_argument("--control-races", type=int, default=200)
    ap.add_argument("--trend-every-s", type=float, default=60.0, help="planted edge strength (10 ticks / 10 min)")
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--control-only", action="store_true", help="run only the positive control")
    ap.add_argument("--n-envs", type=int, default=max(4, os.cpu_count() or 4))
    ap.add_argument("--eval-races", type=int, default=100, help="VAL races for checkpoint selection")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    tapes = list_tapes(a.tapes)
    tr, _, te = split_by_date(tapes)
    cut = liquidity_cut(tr) if a.min_best_usd == "auto" else float(a.min_best_usd)
    cfg = ContinuousConfig(random_start_s=0.0, max_fraction=a.max_fraction, decision_every=a.decision_every,
                           min_best_usd=cut, gate_inputs=not a.no_gate_inputs)
    gates = {}
    if cut > 0:
        gates["real_test"] = gate_summary(te, cut)
        print(f"=== Liquidity gate: open a runner only with >= ${cut:.2f} at best back + lay (10s mean)"
              f"{' [auto: P11 top quintile, TRAIN days]' if a.min_best_usd == 'auto' else ''}"
              f"{'' if a.no_gate_inputs else '; networks get (above cut, distance to cut) per runner'} ===")
        g = gates["real_test"]
        print(f"TEST races, last 5 min: {g['runner_share_open'] * 100:.0f}% of runner snapshots pass; "
              f"{g['races_with_any_open']} of {g['races']} races have a tradeable runner at some point; "
              f"{g['mean_open_runners']:.1f} tradeable runners per snapshot on average\n", flush=True)
    else:
        print("=== No liquidity gate ===\n", flush=True)
    rows = []

    # ---------------- 0. positive control
    control_ok = None
    if not a.skip_control:
        print(f"=== 0. Positive control: planted edge (a runner shortens 1 tick / {a.trend_every_s:.0f}s) ===",
              flush=True)
        syn_dir = os.path.join(a.out, "synthetic")
        write_synthetic(syn_dir, a.control_races, seed0=50_000, trend_every_s=a.trend_every_s)
        glob_ = os.path.join(syn_dir, "*.npz")
        _, _, cte = split_by_date(list_tapes(glob_))
        if cut > 0:
            gates["control_test"] = gate_summary(cte, cut)
            print(f"gate on synthetic races: {gates['control_test']['runner_share_open'] * 100:.0f}% of runner "
                  f"snapshots pass", flush=True)
        ck = run_sac(glob_, os.path.join(a.out, "control"), a.control_steps, a.warmup_frac, 0, a.n_envs,
                     a.device, a.max_fraction, a.buffer, min(a.eval_races, 30), a.decision_every,
                     a.update_every, a.grad_steps, cut, not a.no_gate_inputs)
        rows.append(dict(run="control", **score("do nothing", evaluate(noop_policy, cte, cfg, env_cls=ContinuousAllocEnv))))
        rows.append(dict(run="control", **score("random (f=0.02)", evaluate(random_cont_policy(), cte, cfg,
                                                                              env_cls=ContinuousAllocEnv))))
        _, ctl, _ = test_block("control", ck, cte, cfg, a.device, rows, stress=False)
        control_ok = bool(ctl["mean_green"] > 0 and (ctl["t"] or 0) > 2)
        print(pd.DataFrame([r for r in rows if r["run"] == "control"]).round(3).to_string(index=False))
        print("control: the black box FINDS a small planted edge" if control_ok else
              "control: the black box did NOT find the planted edge -> a null result on real data means little; "
              "try more --control-steps / --steps", flush=True)

    if a.control_only:
        json.dump(dict(control_found_planted_edge=control_ok, gate=gates), open(os.path.join(a.out, "verdict.json"), "w"))
        print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")
        return

    # ---------------- 1-2. real races
    print(f"\n=== 1. Real races: {len(tapes)} tapes, {len(te)} TEST races ===", flush=True)
    rows.append(dict(run="real", **score("do nothing", evaluate(noop_policy, te, cfg, env_cls=ContinuousAllocEnv))))
    rows.append(dict(run="real", **score("random (f=0.02)", evaluate(random_cont_policy(), te, cfg,
                                                                       env_cls=ContinuousAllocEnv))))
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    actors, results = {}, {}
    for sd in seeds:
        print(f"\n--- seed {sd}: {a.steps} steps, first {int(a.steps * a.warmup_frac)} on random strategies ---",
              flush=True)
        ck = run_sac(a.tapes, os.path.join(a.out, f"real_seed{sd}"), a.steps, a.warmup_frac, sd, a.n_envs, a.device,
                     a.max_fraction, a.buffer, a.eval_races, a.decision_every, a.update_every, a.grad_steps, cut,
                     not a.no_gate_inputs)
        actors[sd], main_row, slow_row = test_block(f"seed {sd}", ck, te, cfg, a.device, rows)
        results[sd] = (main_row, slow_row)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    print("\n=== 2. TEST races (never used for training or checkpoint selection); $ per race, $500 bank ===")
    print(res.round(3).to_string(index=False))

    # ---------------- forensics on the best seed
    best_sd = max(results, key=lambda s: results[s][0]["mean_green"]) if results else None
    if best_sd is not None:
        fz = forensics(make_policy_fn(actors[best_sd], a.device), te[:100], cfg)
        fz.to_csv(os.path.join(a.out, f"fills_seed{best_sd}.csv"), index=False)
        print(f"\n=== What seed {best_sd} trades on TEST races ({len(fz)} fills) ===")
        if len(fz):
            print(fz.pivot_table(index="rank", columns=["side"], values="stake", aggfunc="sum", fill_value=0)
                  .round(0).to_string())
            print(fz.pivot_table(index="time", columns=["side"], values="stake", aggfunc="sum", fill_value=0,
                                 observed=False).round(0).to_string())
            if cut > 0:
                print(f"share of $ staked on runners at/above the gate when filled: "
                      f"{fz.loc[fz['best_usd'] >= cut, 'stake'].sum() / max(fz['stake'].sum(), 1e-9) * 100:.0f}% "
                      f"(the rest are reduce-only exits after the runner fell below)")
        else:
            print("no fills: the agent learned not to trade")

    edge = bool(results) and all(r[0]["mean_green"] > 0 and (r[0]["t"] or 0) > 2 and r[1]["mean_green"] > 0
                                 for r in results.values())
    verdict = dict(control_found_planted_edge=control_ok, edge=edge, gate=gates,
                   seeds={str(s): dict(test_green=r[0]["mean_green"], t=r[0]["t"], turnover=r[0]["turnover"],
                                       latency_stress_green=r[1]["mean_green"]) for s, r in results.items()})
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    if edge:
        print("EDGE: every seed beats do-nothing on TEST races and survives the latency stress -> study the forensics, "
              "then forward-test it like P3c")
    elif control_ok is False:
        print("INCONCLUSIVE: the agent could not find a planted edge either")
    else:
        print("NO EDGE: given free rein, the black box finds nothing on unseen races that beats doing nothing")
    print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
