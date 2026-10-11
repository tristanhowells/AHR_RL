"""PPO training.

    python -m ahr_rl.train --tapes "data/tapes/*.npz" --out runs/ppo1 --total-steps 2000000

Data is split chronologically (train / val / test by race date). The best
checkpoint is chosen on validation mean green profit; the test split is only
touched by ``python -m ahr_rl.report``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch

from .env import BetfairPreRaceEnv, EnvConfig, list_tapes, split_by_date
from .evaluate import evaluate, make_torch_policy, summarise
from .policy import RunnerTransformerPolicy


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        bm, bv, bc = x.mean(), x.var(), x.size
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot


def make_env_fn(tapes, cfg, seed):
    def f():
        return BetfairPreRaceEnv(tapes, cfg, seed=seed)

    return f


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tapes = list_tapes(args.tapes)
    tr, va, te = split_by_date(tapes, args.val_frac, args.test_frac)
    print(f"tapes: train {len(tr)}  val {len(va)}  test {len(te)}")
    json.dump({"train": tr, "val": va, "test": te}, open(os.path.join(args.out, "split.json"), "w"), indent=1)

    if args.env == "v2":
        cfg = EnvConfig.v2(forecaster_path=args.forecaster, random_start_s=args.random_start_s)
    else:
        cfg = EnvConfig(random_start_s=args.random_start_s)
    args.n_runner_features = int(BetfairPreRaceEnv([], cfg).observation_space["runners"].shape[1])
    json.dump({**vars(args), "env": asdict(cfg)}, open(os.path.join(args.out, "config.json"), "w"), indent=1, default=str)
    fns = [make_env_fn(tr, cfg, args.seed + i) for i in range(args.n_envs)]
    if args.n_envs > 1 and not args.sync:
        venv = gym.vector.AsyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    else:
        venv = gym.vector.SyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    device = torch.device(args.device)
    model = RunnerTransformerPolicy(cfg, d_model=args.d_model, n_layers=args.n_layers, noop_bias=args.noop_bias,
                                    n_runner_features=args.n_runner_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    N, T = args.n_envs, args.n_steps
    obs, _ = venv.reset(seed=args.seed)
    R, F = obs["runners"].shape[1:]
    G = obs["global"].shape[1]
    buf_r = torch.zeros(T, N, R, F)
    buf_g = torch.zeros(T, N, G)
    buf_m = torch.zeros(T, N, R)
    buf_a = torch.zeros(T, N, R, 5, dtype=torch.long)
    buf_lp = torch.zeros(T, N, R)
    buf_v = torch.zeros(T, N, R)
    buf_rew = torch.zeros(T, N, R)
    buf_done = torch.zeros(T, N)

    ret_rms = RunningMeanStd()
    ret_acc = np.zeros((N, R))
    n_updates = args.total_steps // (N * T)
    log_path = os.path.join(args.out, "train_log.csv")
    logf = open(log_path, "w")
    logf.write("update,steps,sps,ep_green_mean,ep_turnover_mean,episodes,pg_loss,v_loss,entropy,kl,p_open,"
               "val_green,val_pct_green,val_turnover,val_trades\n")
    best_val = -np.inf
    ep_green, ep_turn = [], []
    global_step = 0
    t0 = time.time()

    def to_t(o):
        return (torch.as_tensor(o["runners"], dtype=torch.float32), torch.as_tensor(o["global"], dtype=torch.float32),
                torch.as_tensor(o["mask"], dtype=torch.float32))

    for update in range(1, n_updates + 1):
        frac = 1.0 - (update - 1) / n_updates
        for pg in opt.param_groups:
            pg["lr"] = args.lr * max(frac, 0.1)
        # exploration floor, annealed linearly to 0 over the first explore_anneal_frac of training
        prog = (update - 1) / n_updates
        model.explore_eps = args.explore_floor * max(0.0, 1.0 - prog / max(args.explore_anneal_frac, 1e-9))
        model.eval()
        for t in range(T):
            o_r, o_g, o_m = to_t(obs)
            with torch.no_grad():
                dist, v = model.dist(o_r.to(device), o_g.to(device), o_m.to(device))
                a = model.sample(dist) * o_m.long().to(device)[..., None]
                lp = model.runner_logprob(dist, a)
            buf_r[t], buf_g[t], buf_m[t] = o_r, o_g, o_m
            buf_a[t], buf_lp[t], buf_v[t] = a.cpu(), lp.cpu(), v.cpu()
            obs, rew, term, trunc, info = venv.step(model.codec.to_flat(a.cpu().numpy()))
            done_np = np.logical_or(term, trunc)
            rr = np.zeros((N, R), np.float32)
            if "runner_rewards" in info:
                sel = info["_runner_rewards"]
                rr[sel] = info["runner_rewards"][sel]
            fin = info.get("final_info")
            if fin is not None and "runner_rewards" in fin:
                sel = fin["_runner_rewards"]
                rr[sel] = fin["runner_rewards"][sel]
            if fin is not None and "worst" in fin:
                for i in np.nonzero(fin["_worst"])[0]:
                    ep_green.append(float(fin["worst"][i]))
                    ep_turn.append(float(fin["turnover"][i]))
            # scale by running std of per-runner discounted returns (keeps value
            # targets O(1) so the critic doesn't swamp the policy gradient)
            ret_acc = ret_acc * args.gamma + rr
            ret_rms.update(ret_acc[buf_m[t].numpy() > 0] if (buf_m[t] > 0).any() else ret_acc)
            ret_acc[done_np] = 0.0
            buf_rew[t] = torch.as_tensor(rr / np.sqrt(ret_rms.var + 1e-8), dtype=torch.float32)
            buf_done[t] = torch.as_tensor(done_np, dtype=torch.float32)
            global_step += N

        # ------------------------------------------------ per-runner GAE
        with torch.no_grad():
            o_r, o_g, o_m = to_t(obs)
            _, last_v = model.dist(o_r.to(device), o_g.to(device), o_m.to(device))
            last_v, last_m = last_v.cpu(), o_m
        adv = torch.zeros(T, N, R)
        gae = torch.zeros(N, R)
        for t in reversed(range(T)):
            nv, nm = (last_v, last_m) if t == T - 1 else (buf_v[t + 1], buf_m[t + 1])
            nonterm = (1.0 - buf_done[t])[:, None] * nm  # runner gone / episode over -> no bootstrap
            delta = buf_rew[t] + args.gamma * nv * nonterm - buf_v[t]
            gae = delta + args.gamma * args.gae_lambda * nonterm * gae
            adv[t] = gae * buf_m[t]
        ret = adv + buf_v

        # ------------------------------------------------ PPO update (per-runner ratios)
        model.train()
        B = T * N
        b_r, b_g, b_m = buf_r.reshape(B, R, F), buf_g.reshape(B, G), buf_m.reshape(B, R)
        b_a, b_lp, b_adv, b_ret, b_v = (buf_a.reshape(B, R, 5), buf_lp.reshape(B, R), adv.reshape(B, R),
                                        ret.reshape(B, R), buf_v.reshape(B, R))
        act = b_m > 0
        a_mean, a_std = b_adv[act].mean(), b_adv[act].std() + 1e-8
        idx = np.arange(B)
        stats = []
        for epoch in range(args.epochs):
            np.random.shuffle(idx)
            for s0 in range(0, B, args.minibatch):
                mb = torch.as_tensor(idx[s0:s0 + args.minibatch])
                m = b_m[mb].to(device)
                dist, v = model.dist(b_r[mb].to(device), b_g[mb].to(device), m)
                lp = model.runner_logprob(dist, b_a[mb].to(device))
                ratio = torch.exp(lp - b_lp[mb].to(device))
                A = (b_adv[mb].to(device) - a_mean) / a_std
                pg = -(torch.min(ratio * A, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * A) * m).sum() / m.sum().clamp(min=1)
                ov, rt = b_v[mb].to(device), b_ret[mb].to(device)
                v_clip = ov + torch.clamp(v - ov, -args.clip_v, args.clip_v)
                vl = 0.5 * (torch.max((v - rt) ** 2, (v_clip - rt) ** 2) * m).sum() / m.sum().clamp(min=1)
                ent = (model.runner_entropy(dist) * m).sum() / m.sum().clamp(min=1)
                tent = (model.type_entropy(dist) * m).sum() / m.sum().clamp(min=1)
                loss = pg + args.vf_coef * vl - args.ent_coef * ent - args.type_ent_coef * tent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kl = ((((ratio - 1) - torch.log(ratio)) * m).sum() / m.sum().clamp(min=1)).item()
                    p_open = ((dist[0].probs[..., 3]) * m).sum().item() / max(m.sum().item(), 1)
                stats.append((pg.item(), vl.item(), ent.item(), kl, p_open))
            if np.mean([s[3] for s in stats[-4:]]) > args.target_kl:
                break
        pg_l, v_l, ent_v, kl_v, p_open = np.mean(stats, 0)
        sps = global_step / (time.time() - t0)

        val_s = {"mean_green_$": np.nan, "pct_green": np.nan, "mean_turnover_$": np.nan, "mean_trades": np.nan}
        if update % args.eval_every == 0 or update == n_updates:
            model.eval()
            eps, model.explore_eps = model.explore_eps, 0.0  # evaluate the policy itself
            df = evaluate(make_torch_policy(model, True, device), va[: args.eval_races], cfg)
            model.explore_eps = eps
            val_s = summarise(df)
            ck = {"model": model.state_dict(), "cfg": asdict(cfg), "args": vars(args), "update": update,
                  "val": val_s}
            if val_s["mean_green_$"] > best_val:
                best_val = val_s["mean_green_$"]
                torch.save(ck, os.path.join(args.out, "best.pt"))
            torch.save(ck, os.path.join(args.out, "last.pt"))
        eg = np.mean(ep_green[-50:]) if ep_green else np.nan
        et = np.mean(ep_turn[-50:]) if ep_turn else np.nan
        logf.write(f"{update},{global_step},{sps:.0f},{eg:.3f},{et:.1f},{len(ep_green)},{pg_l:.4f},{v_l:.4f},"
                   f"{ent_v:.4f},{kl_v:.5f},{p_open:.4f},{val_s['mean_green_$']:.3f},{val_s['pct_green']:.1f},"
                   f"{val_s['mean_turnover_$']:.1f},{val_s['mean_trades']:.2f}\n")
        logf.flush()
        print(f"upd {update}/{n_updates} steps {global_step} sps {sps:.0f} | train green(50ep) {eg:+.2f} "
              f"turnover {et:.0f} | p_open {p_open:.4f} eps {model.explore_eps:.3f} ent {ent_v:.3f} kl {kl_v:.4f} "
              f"vloss {v_l:.3f}"
              + (f" | VAL green {val_s['mean_green_$']:+.2f} ({val_s['pct_green']:.0f}% green, "
                 f"{val_s['mean_trades']:.1f} trades)" if not np.isnan(val_s['mean_green_$']) else ""),
              flush=True)
    venv.close()
    logf.close()
    return os.path.join(args.out, "best.pt")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tapes", default="data/tapes/*.npz")
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--clip-v", type=float, default=1.0)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--noop-bias", type=float, default=3.0)
    ap.add_argument("--random-start-s", type=float, default=30.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-races", type=int, default=50)
    ap.add_argument("--env", default="v1", choices=["v1", "v2"])
    ap.add_argument("--forecaster", default=None, help="forecaster.pkl for --env v2")
    ap.add_argument("--explore-floor", type=float, default=0.0,
                    help="min P(OPEN) per runner-decision at the start of training (mixture; annealed)")
    ap.add_argument("--explore-anneal-frac", type=float, default=0.5)
    ap.add_argument("--type-ent-coef", type=float, default=0.0,
                    help="extra entropy bonus on the noop/close/cancel/open decision")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sync", action="store_true", help="single-process envs (debugging)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
