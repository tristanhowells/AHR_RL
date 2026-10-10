"""Train the black-box SAC single-runner trader.

    python -m single_runner.train --data-dir /content/drive/MyDrive/race_out \
        --out-dir runs/sr_sac --total-steps 1000000 --ranks 1,2,3

Every training episode samples a random (race, runner-rank) pair, so each
runner of each race is a separate episode. Evaluation runs the deterministic
policy on EVERY (validation race, eligible rank) episode.
"""
import argparse
import csv
import glob
import json
import os
import re
import time
from dataclasses import asdict

import numpy as np
import torch

from .env import EnvConfig, SingleRunnerTradingEnv, episode_specs
from .sac import ReplayBuffer, SACAgent, SACConfig

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def list_race_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    insp = os.path.join(data_dir, "_inspection_results.csv")
    if os.path.exists(insp):
        with open(insp) as f:
            ok = {r["file"] for r in csv.DictReader(f) if r.get("is_suitable", "").strip() == "True"}
        files = [p for p in files if os.path.basename(p) in ok]
    return files


def temporal_split(files, val_frac=0.2, val_start=None):
    """Chronological split on the YYYY-MM-DD in the filename (no shuffling across time)."""
    dated = sorted((m.group(0), f) for f in files if (m := _DATE_RE.search(os.path.basename(f))))
    if val_start:
        train = [f for d, f in dated if d < val_start]
        val = [f for d, f in dated if d >= val_start]
    else:
        n_val = max(1, int(round(len(dated) * val_frac))) if len(dated) > 1 else 0
        train = [f for _, f in dated[:len(dated) - n_val]]
        val = [f for _, f in dated[len(dated) - n_val:]]
    return train, val


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


def summarise(rows):
    pnl = np.array([r["pnl"] for r in rows])
    traded = np.array([r["back_matched"] + r["lay_matched"] > 0 for r in rows])
    out = {
        "episodes": len(rows), "mean_pnl": pnl.mean(), "median_pnl": float(np.median(pnl)),
        "std_pnl": pnl.std(), "total_pnl": pnl.sum(), "win_rate": float((pnl > 0).mean()),
        "trade_rate": float(traded.mean()),
        "mean_matched": float(np.mean([r["back_matched"] + r["lay_matched"] for r in rows])),
        "mean_fills": float(np.mean([r["n_fills"] for r in rows])),
        "passive_share": float(np.sum([r["n_passive"] for r in rows]) / max(1, np.sum([r["n_fills"] for r in rows]))),
        "sharpe_per_ep": float(pnl.mean() / (pnl.std() + 1e-9)),
    }
    for k in sorted({r["target_rank"] for r in rows}):
        out[f"mean_pnl_rank{k}"] = float(np.mean([r["pnl"] for r in rows if r["target_rank"] == k]))
    return out


def _append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def train(train_files, val_files, out_dir, env_cfg, sac_cfg, total_steps=1_000_000,
          eval_every=50_000, max_eval_episodes=None, seed=0, log_every_eps=50):
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = SingleRunnerTradingEnv(train_files, env_cfg, seed=seed)
    eval_cfg = EnvConfig(**{**asdict(env_cfg), "random_start": False})
    val_env = SingleRunnerTradingEnv(val_files or train_files, eval_cfg, seed=seed + 1)
    specs = episode_specs(val_env, val_files or train_files)
    if max_eval_episodes and len(specs) > max_eval_episodes:
        rng = np.random.default_rng(seed)
        specs = [specs[i] for i in sorted(rng.choice(len(specs), max_eval_episodes, replace=False))]
    print(f"train races {len(train_files)} | val races {len(val_files)} | val episodes {len(specs)}")

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, sac_cfg)
    buf = ReplayBuffer(obs_dim, act_dim, sac_cfg.buffer_size)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"env": asdict(env_cfg), "sac": asdict(sac_cfg), "obs_dim": obs_dim,
                   "train_files": len(train_files), "val_files": len(val_files)}, f, indent=2, default=str)

    obs, _ = env.reset()
    agent.norm.update(obs)
    ep, ep_ret, best, t0, stats, recent = 0, 0.0, -np.inf, time.time(), {}, []
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
                print(f"[{step:>8}] ep {ep:>6} | pnl {s['mean_pnl']:+7.3f} | win {s['win_rate']:.2f} "
                      f"| trade {s['trade_rate']:.2f} | matched {s['mean_matched']:7.1f} "
                      f"| alpha {stats.get('alpha', float('nan')):.3f} | {step / (time.time() - t0):.0f} sps")
                recent = []
            obs, _ = env.reset()
            agent.norm.update(obs)
            ep_ret = 0.0

        if step % eval_every == 0 or step == total_steps:
            rows = evaluate(agent, val_env, specs)
            s = summarise(rows)
            _append_csv(os.path.join(out_dir, "val_metrics.csv"), {"step": step, **s})
            print(f"  VAL @ {step}: mean pnl {s['mean_pnl']:+.3f}  median {s['median_pnl']:+.3f} "
                  f"win {s['win_rate']:.2f}  trade {s['trade_rate']:.2f}  sharpe/ep {s['sharpe_per_ep']:+.3f}")
            agent.save(os.path.join(out_dir, "last.pt"), {"step": step, "val": s, "env": asdict(env_cfg)})
            if s["mean_pnl"] > best:
                best = s["mean_pnl"]
                agent.save(os.path.join(out_dir, "best.pt"), {"step": step, "val": s, "env": asdict(env_cfg)})
                print(f"  new best ({best:+.3f}) saved")
    return agent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="runs/single_runner_sac")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--val-start", default=None, help="YYYY-MM-DD; overrides --val-frac")
    p.add_argument("--ranks", default=None, help="comma list of favouritism ranks, e.g. 1,2,3 (default all)")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--max-eval-episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    # env
    p.add_argument("--balance", type=float, default=100.0)
    p.add_argument("--tick-range", type=int, default=10)
    p.add_argument("--min-stake", type=float, default=1.0)
    p.add_argument("--fill-optimism", type=float, default=1.0)
    p.add_argument("--no-cross-matching", action="store_true")
    p.add_argument("--greenup", choices=["fair", "cross", "ltp"], default="fair")
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--max-episode-steps", type=int, default=None)
    p.add_argument("--random-start", action="store_true")
    # sac
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    files = list_race_files(a.data_dir)
    train_files, val_files = temporal_split(files, a.val_frac, a.val_start)
    env_cfg = EnvConfig(
        initial_balance=a.balance, tick_range=a.tick_range, min_stake=a.min_stake,
        fill_optimism=a.fill_optimism, cross_matching=not a.no_cross_matching,
        greenup_mode=a.greenup, reward_scale=a.reward_scale,
        max_episode_steps=a.max_episode_steps, random_start=a.random_start,
        runner_ranks=tuple(int(x) for x in a.ranks.split(",")) if a.ranks else None,
    )
    sac_cfg = SACConfig(lr=a.lr, gamma=a.gamma, batch_size=a.batch_size, buffer_size=a.buffer_size,
                        learning_starts=a.learning_starts, updates_per_step=a.updates_per_step,
                        device=a.device)
    train(train_files or files, val_files, a.out_dir, env_cfg, sac_cfg, a.total_steps,
          a.eval_every, a.max_eval_episodes, a.seed)


if __name__ == "__main__":
    main()
