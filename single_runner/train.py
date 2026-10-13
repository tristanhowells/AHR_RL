"""Train the black-box SAC single-runner trader on stream tapes.

    python -m single_runner.train --tapes "/content/tapes/*.npz" --out-dir runs/sr_sac \
        --total-steps 1000000 --ranks 1,2,3 --race-type flat --catalogues "<drive>/catalogues"

Tapes come from `python -m ahr_rl.tape <recordings> <tapes>`. The split is the
research one (ahr_rl.env.split_by_date: chronological train / val / test by
the YYYYMMDD file prefix). Training samples a random (race, runner-rank) pair
per episode; validation runs the deterministic policy on every (val race,
eligible rank) pair, and `--eval-test` scores a checkpoint once on TEST.
"""
import argparse
import csv
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from ahr_rl import race_filter
from ahr_rl.env import list_tapes, split_by_date

from .env import EnvConfig, SingleRunnerTradingEnv, episode_specs
from .sac import ReplayBuffer, SACAgent, SACConfig


def evaluate(agent, env, specs, deterministic=True):
    rows = []
    for spec in specs:
        obs, _ = env.reset(options=spec)
        done, ret = False, 0.0
        while not done:
            obs, r, done, _, info = env.step(agent.act(obs, deterministic))
            ret += r
        info = {k: v for k, v in info.items() if not isinstance(v, (list, dict))}
        info["return"] = ret
        rows.append(info)
    return rows


def race_t(rows):
    """t-stat of mean P&L with races as the unit (a race's ranks are not independent)."""
    by = {}
    for r in rows:
        if not r.get("void"):
            by.setdefault(r["race"], []).append(r["pnl"])
    x = np.array([np.mean(v) for v in by.values()])
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def summarise(rows):
    pnl = np.array([r["pnl"] for r in rows])
    traded = np.array([r["back_matched"] + r["lay_matched"] > 0 for r in rows])
    fills = max(1, np.sum([r["n_fills"] for r in rows]))
    out = {
        "episodes": len(rows), "races": len({r["race"] for r in rows}),
        "mean_pnl": float(pnl.mean()), "median_pnl": float(np.median(pnl)),
        "std_pnl": float(pnl.std()), "total_pnl": float(pnl.sum()), "t_race": race_t(rows),
        "win_rate": float((pnl > 0).mean()), "trade_rate": float(traded.mean()),
        "mean_matched": float(np.mean([r["back_matched"] + r["lay_matched"] for r in rows])),
        "mean_fills": float(np.mean([r["n_fills"] for r in rows])),
        "passive_share": float(np.sum([r["n_passive"] for r in rows]) / fills),
    }
    for k in sorted({r["target_rank"] for r in rows}):
        out[f"mean_pnl_rank{k}"] = float(np.mean([r["pnl"] for r in rows if r["target_rank"] == k]))
    return out


def eval_specs(env, tapes, max_episodes=None, seed=0):
    specs = episode_specs(env, tapes)
    if max_episodes and len(specs) > max_episodes:
        rng = np.random.default_rng(seed)
        specs = [specs[i] for i in sorted(rng.choice(len(specs), max_episodes, replace=False))]
    return specs


def _append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def _device_banner(agent):
    if agent.device.type == "cuda":
        print(f"device: cuda ({torch.cuda.get_device_name(agent.device)})")
    else:
        print("device: CPU -- WARNING: SAC updates on CPU run at ~5-10 steps/s. "
              "In Colab use Runtime > Change runtime type > GPU.")


def train(train_tapes, val_tapes, out_dir, env_cfg, sac_cfg, total_steps=500_000,
          eval_every=25_000, val_episodes=300, seed=0, log_every_eps=50, final_full_val=True):
    """Train; checkpoint selection on a fixed validation subset, final check on full validation.

    `best.pt` is written only for a checkpoint whose validation mean P&L beats doing
    nothing (> 0). If none does, there is no best.pt and nothing should go to TEST.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SingleRunnerTradingEnv(train_tapes, env_cfg, seed=seed)
    eval_cfg = EnvConfig(**{**asdict(env_cfg), "random_start_s": 0.0,
                            "cache_size": max(env_cfg.cache_size, len(val_tapes) + 1)})
    val_env = SingleRunnerTradingEnv(val_tapes, eval_cfg, seed=seed + 1)
    all_specs = episode_specs(val_env, val_tapes)
    specs = eval_specs(val_env, val_tapes, val_episodes, seed)
    print(f"train races {len(train_tapes)} | val races {len(val_tapes)} | "
          f"val episodes {len(all_specs)} (periodic subset {len(specs)})")

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, sac_cfg)
    _device_banner(agent)
    buf = ReplayBuffer(obs_dim, act_dim, sac_cfg.buffer_size)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"env": asdict(env_cfg), "sac": asdict(sac_cfg), "obs_dim": obs_dim,
                   "train_races": len(train_tapes), "val_races": len(val_tapes)}, f, indent=2, default=str)
    best_path = os.path.join(out_dir, "best.pt")
    if os.path.exists(best_path):
        os.remove(best_path)          # never let a stale best.pt from an earlier run reach TEST
    obs, _ = env.reset()
    agent.norm.update(obs)
    ep, ep_ret, best, stats, recent = 0, 0.0, 0.0, {}, []
    t0, eval_secs = time.time(), 0.0
    for step in range(1, total_steps + 1):
        if step <= sac_cfg.learning_starts:
            act = env.action_space.sample()
        else:
            act = agent.act(obs)
        obs2, r, term, trunc, info = env.step(act)
        buf.add(obs, act, r, obs2, float(term))
        agent.norm.update(obs2)
        obs, ep_ret = obs2, ep_ret + r

        if step > sac_cfg.learning_starts:
            for _ in range(sac_cfg.updates_per_step):
                stats = agent.update(buf)

        if term or trunc:
            ep += 1
            recent.append(info)
            _append_csv(os.path.join(out_dir, "train_episodes.csv"),
                        {"step": step, "episode": ep, "return": ep_ret, **stats,
                         **{k: v for k, v in info.items() if not isinstance(v, (list, dict))}})
            if ep % log_every_eps == 0:
                s = summarise(recent)
                sps = step / max(time.time() - t0 - eval_secs, 1e-9)
                print(f"[{step:>8}] ep {ep:>6} | pnl {s['mean_pnl']:+7.3f} | win {s['win_rate']:.2f} "
                      f"| trade {s['trade_rate']:.2f} | matched {s['mean_matched']:7.1f} "
                      f"| alpha {stats.get('alpha', float('nan')):.3f} | {sps:.0f} sps")
                recent = []
            obs, _ = env.reset()
            agent.norm.update(obs)
            ep_ret = 0.0

        if step % eval_every == 0 or step == total_steps:
            te = time.time()
            s = summarise(evaluate(agent, val_env, specs))
            eval_secs += time.time() - te
            _append_csv(os.path.join(out_dir, "val_metrics.csv"), {"step": step, **s})
            print(f"  VAL @ {step}: mean pnl {s['mean_pnl']:+.3f}  t(race) {s['t_race']:+.2f}  "
                  f"median {s['median_pnl']:+.3f}  win {s['win_rate']:.2f}  trade {s['trade_rate']:.2f}  "
                  f"({time.time() - te:.0f}s)")
            agent.save(os.path.join(out_dir, "last.pt"), {"step": step, "val": s, "env": asdict(env_cfg)})
            if s["mean_pnl"] > best:
                best = s["mean_pnl"]
                agent.save(best_path, {"step": step, "val": s, "env": asdict(env_cfg)})
                print(f"  new best ({best:+.3f}, beats do-nothing) saved")

    if not os.path.exists(best_path):
        print("No checkpoint beat do-nothing (mean P&L > 0) on validation: no best.pt, nothing to send to TEST.")
    elif final_full_val:
        full, extra = evaluate_checkpoint(best_path, val_tapes, eval_cfg, greenup_modes=("fair",))
        f = full["fair"]
        with open(os.path.join(out_dir, "val_full_best.json"), "w") as fh:
            json.dump({"step": extra.get("step"), **f}, fh, indent=2, default=float)
        verdict = "beats" if f["mean_pnl"] > 0 else "does NOT beat"
        print(f"best.pt (step {extra.get('step')}) on ALL {f['episodes']} val episodes: mean {f['mean_pnl']:+.3f} "
              f"t(race) {f['t_race']:+.2f} -> {verdict} do-nothing")
    return agent


def evaluate_checkpoint(ckpt, tapes, env_cfg, out_csv=None, greenup_modes=("fair", "cross")):
    """Score a checkpoint on every (race, rank) episode of `tapes`, under each green-up rule."""
    agent, extra = SACAgent.load(ckpt)
    results = {}
    for mode in greenup_modes:
        cfg = EnvConfig(**{**asdict(env_cfg), "greenup_mode": mode, "random_start_s": 0.0})
        env = SingleRunnerTradingEnv(tapes, cfg)
        rows = evaluate(agent, env, episode_specs(env, tapes))
        results[mode] = summarise(rows)
        if out_csv:
            for r in rows:
                _append_csv(out_csv, {"greenup": mode, **r})
    return results, extra


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tapes", required=True, help='glob, e.g. "/content/tapes/*.npz"')
    p.add_argument("--out-dir", default="runs/single_runner_sac")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--ranks", default=None, help="comma list of favouritism ranks, e.g. 1,2,3 (default all)")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--val-episodes", type=int, default=300, help="fixed periodic validation subset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-test", default=None, metavar="CKPT",
                   help="skip training; score this checkpoint once on the TEST split")
    race_filter.add_args(p)
    # env
    p.add_argument("--balance", type=float, default=100.0)
    p.add_argument("--min-stake", type=float, default=5.0)
    p.add_argument("--tick-range", type=int, default=10)
    p.add_argument("--decision-every", type=int, default=20, help="tape steps (0.5 s) per decision (20 = 10 s)")
    p.add_argument("--latency-steps", type=int, default=1)
    p.add_argument("--fill-mode", choices=["realistic", "no_queue", "touch"], default="realistic")
    p.add_argument("--no-cross-matching", action="store_true")
    p.add_argument("--greenup", choices=["fair", "cross"], default="fair")
    p.add_argument("--commission", type=float, default=None, help="fraction; default = market base rate")
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--start-s", type=float, default=600.0)
    p.add_argument("--random-start-s", type=float, default=0.0)
    # sac
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--alpha-min", type=float, default=0.02, help="entropy coefficient floor (0 = off)")
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    tapes = race_filter.filter_paths(list_tapes(a.tapes), a)
    train_tapes, val_tapes, test_tapes = split_by_date(tapes, a.val_frac, a.test_frac)
    print(f"{len(tapes)} tapes -> train {len(train_tapes)} / val {len(val_tapes)} / test {len(test_tapes)}")
    env_cfg = EnvConfig(
        initial_balance=a.balance, min_stake=a.min_stake, tick_range=a.tick_range,
        decision_every=a.decision_every, latency_steps=a.latency_steps, fill_mode=a.fill_mode,
        cross_matching=not a.no_cross_matching, greenup_mode=a.greenup, commission=a.commission,
        reward_scale=a.reward_scale, start_s=a.start_s, random_start_s=a.random_start_s,
        runner_ranks=tuple(int(x) for x in a.ranks.split(",")) if a.ranks else None,
    )
    if a.eval_test:
        os.makedirs(a.out_dir, exist_ok=True)
        res, extra = evaluate_checkpoint(a.eval_test, test_tapes, env_cfg,
                                         os.path.join(a.out_dir, "test_episodes.csv"))
        print(json.dumps({"checkpoint_step": extra.get("step"), **res}, indent=2, default=float))
        with open(os.path.join(a.out_dir, "test_summary.json"), "w") as f:
            json.dump(res, f, indent=2, default=float)
        return
    sac_cfg = SACConfig(lr=a.lr, gamma=a.gamma, batch_size=a.batch_size, buffer_size=a.buffer_size,
                        learning_starts=a.learning_starts, updates_per_step=a.updates_per_step,
                        alpha_min=a.alpha_min, device=a.device)
    train(train_tapes, val_tapes, a.out_dir, env_cfg, sac_cfg, a.total_steps,
          a.eval_every, a.val_episodes, a.seed)


if __name__ == "__main__":
    main()
