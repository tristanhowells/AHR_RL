"""PPO and SAC for the continuous allocation spec (env_continuous.py).

Both use the same runner-equivariant transformer backbone as the discrete
agent (policy.py) so that comparisons isolate the *action spec* and the
*algorithm*, not the architecture.

    python -m ahr_rl.continuous --algo sac --tapes "data/tapes/*.npz" --out runs/sac
    python -m ahr_rl.continuous --algo ppo --tapes "data/tapes/*.npz" --out runs/ppo_cont
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .env import list_tapes, split_by_date
from .env_continuous import N_CONT, ContinuousAllocEnv, ContinuousConfig
from .evaluate import evaluate, summarise
from .features import N_GLOBAL_FEATURES, N_RUNNER_FEATURES, R_MAX
from .train import RunningMeanStd

LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0


class Backbone(nn.Module):
    """Runner tokens (+ optional per-runner action) + market token -> transformer."""

    def __init__(self, d: int = 64, layers: int = 2, runner_extra: int = 0, global_extra: int = 0):
        super().__init__()
        self.r_in = nn.Sequential(nn.Linear(N_RUNNER_FEATURES + N_GLOBAL_FEATURES + runner_extra, d), nn.GELU(),
                                  nn.Linear(d, d))
        self.m_in = nn.Sequential(nn.Linear(N_GLOBAL_FEATURES + global_extra, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, runners, glob, mask, r_extra=None, g_extra=None):
        B, R, _ = runners.shape
        x = torch.cat([runners, glob[:, None].expand(B, R, -1)], -1)
        if r_extra is not None:
            x = torch.cat([x, r_extra], -1)
        g = glob if g_extra is None else torch.cat([glob, g_extra], -1)
        h = torch.cat([self.m_in(g)[:, None], self.r_in(x)], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask.bool()], 1)
        h = self.norm(self.enc(h, src_key_padding_mask=pad))
        return h[:, 0], h[:, 1:]


class Actor(nn.Module):
    """Squashed Gaussian over 24 runner actions (tanh) + fraction ((tanh+1)/2)."""

    def __init__(self, d=64, layers=2, frac_bias=-2.5, init_log_std=-0.5):
        super().__init__()
        self.bb = Backbone(d, layers)
        self.r_head = nn.Linear(d, 2)
        self.f_head = nn.Linear(2 * d, 2)
        with torch.no_grad():
            for h in (self.r_head, self.f_head):
                h.weight.mul_(0.01)
                h.bias.zero_()
                h.bias[1] = init_log_std
            self.f_head.bias[0] = frac_bias  # start by wagering ~1% of funds per step

    def forward(self, runners, glob, mask):
        hm, hr = self.bb(runners, glob, mask)
        m = mask[..., None].float()
        pooled = (hr * m).sum(1) / m.sum(1).clamp(min=1)
        r = self.r_head(hr)
        f = self.f_head(torch.cat([hm, pooled], -1))
        mu = torch.cat([r[..., 0], f[:, :1]], -1)  # [B, 25]
        log_std = torch.cat([r[..., 1], f[:, 1:]], -1).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    @staticmethod
    def dim_mask(mask):
        return torch.cat([mask.float(), torch.ones(mask.shape[0], 1, device=mask.device)], -1)

    @staticmethod
    def squash(u):
        a = torch.tanh(u)
        return torch.cat([a[:, :R_MAX], (a[:, R_MAX:] + 1) / 2], -1)

    def log_prob(self, mu, log_std, u, dmask):
        """log pi(a) for pre-tanh sample u, masked to active dims."""
        lp = -0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        lp = lp - (2 * (math.log(2) - u - Fn.softplus(-2 * u)))  # tanh Jacobian
        return (lp * dmask).sum(-1)

    def sample(self, runners, glob, mask, deterministic=False):
        mu, log_std = self(runners, glob, mask)
        u = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        dm = self.dim_mask(mask)
        return u, self.squash(u) * dm, self.log_prob(mu, log_std, u, dm), mu, log_std


class Critic(nn.Module):
    """Q(s, a): each runner token sees its own action; the market token sees f."""

    def __init__(self, d=64, layers=2):
        super().__init__()
        self.bb = Backbone(d, layers, runner_extra=1, global_extra=1)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, runners, glob, mask, act):
        hm, hr = self.bb(runners, glob, mask, act[:, :R_MAX, None], act[:, R_MAX:])
        m = mask[..., None].float()
        pooled = (hr * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(torch.cat([hm, pooled], -1)).squeeze(-1)


class ValueNet(nn.Module):
    def __init__(self, d=64, layers=2):
        super().__init__()
        self.bb = Backbone(d, layers)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, runners, glob, mask):
        hm, hr = self.bb(runners, glob, mask)
        m = mask[..., None].float()
        pooled = (hr * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(torch.cat([hm, pooled], -1)).squeeze(-1)


def obs_t(o, device):
    return (torch.as_tensor(o["runners"], dtype=torch.float32, device=device),
            torch.as_tensor(o["global"], dtype=torch.float32, device=device),
            torch.as_tensor(o["mask"], dtype=torch.float32, device=device))


def make_policy_fn(actor, device="cpu"):
    @torch.no_grad()
    def f(obs, env):
        r, g, m = obs_t(obs, device)
        _, a, _, _, _ = actor.sample(r[None], g[None], m[None], deterministic=True)
        return a[0].cpu().numpy()

    return f


def _setup(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tr, va, te = split_by_date(list_tapes(args.tapes))
    json.dump({"train": tr, "val": va, "test": te}, open(os.path.join(args.out, "split.json"), "w"), indent=1)
    cfg = ContinuousConfig(random_start_s=30.0, max_fraction=getattr(args, "max_fraction", 1.0),
                           decision_every=getattr(args, "decision_every", 4))
    json.dump({**vars(args), "env": asdict(cfg)}, open(os.path.join(args.out, "config.json"), "w"), indent=1,
              default=str)
    fns = [(lambda i=i: ContinuousAllocEnv(tr, cfg, seed=args.seed + i)) for i in range(args.n_envs)]
    venv = gym.vector.AsyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    print(f"[{args.algo}] tapes train {len(tr)} val {len(va)} test {len(te)}", flush=True)
    return cfg, venv, va


def _episode_stats(info, sink):
    fin = info.get("final_info")
    if fin is not None and "worst" in fin:
        for i in np.nonzero(fin["_worst"])[0]:
            sink.append((float(fin["worst"][i]), float(fin["turnover"][i])))


def _maybe_eval(args, actor, cfg, va, it, best, logf, stats, t0, steps, device):
    val = {"mean_green_$": np.nan, "pct_green": np.nan, "mean_trades": np.nan, "mean_turnover_$": np.nan}
    if it % args.eval_every == 0:
        actor.eval()
        val = summarise(evaluate(make_policy_fn(actor, device), va[: args.eval_races], cfg,
                                 env_cls=ContinuousAllocEnv))
        actor.train()
        ck = {"actor": actor.state_dict(), "cfg": asdict(cfg), "args": vars(args), "iter": it, "val": val}
        if val["mean_green_$"] > best[0]:
            best[0] = val["mean_green_$"]
            torch.save(ck, os.path.join(args.out, "best.pt"))
        torch.save(ck, os.path.join(args.out, "last.pt"))
    eg = np.mean([s[0] for s in stats[-50:]]) if stats else np.nan
    et = np.mean([s[1] for s in stats[-50:]]) if stats else np.nan
    sps = steps / (time.time() - t0)
    logf.write(f"{it},{steps},{sps:.0f},{eg:.3f},{et:.1f},{val['mean_green_$']:.3f},{val['pct_green']:.1f},"
               f"{val['mean_turnover_$']:.1f}\n")
    logf.flush()
    if not np.isnan(val["mean_green_$"]):
        print(f"[{args.algo}] it {it} steps {steps} sps {sps:.0f} | train green {eg:+.2f} turnover {et:.0f} | "
              f"VAL green {val['mean_green_$']:+.2f} ({val['pct_green']:.0f}% green, turnover "
              f"{val['mean_turnover_$']:.0f})", flush=True)


# ============================================================================ SAC
class RandomStrategies:
    """Warm-up exploration (no learning yet, nothing at stake).

    iid       a fresh random action every step, small wager fraction (the original
              warm-up). It churns: entries are undone by the next step's noise, so
              it rarely shows the critic what holding a position does.
    episodic  a random *strategy* held for a whole race: conviction per runner =
              tanh(gain * (w . runner features + v . market features + b)) with a
              random sparse w, v, b, gain and wager fraction. Like parameter-space
              noise / random search, it tries consistent, state-dependent behaviours
              (e.g. "back runners whose feature 7 is high, late in the market") that
              per-step noise can't express.
    mixed     each race picks one of the two (default for the black-box run)."""

    def __init__(self, n_envs: int, n_feat: int, n_glob: int, mode: str = "iid", max_fraction: float = 1.0,
                 seed: int = 0):
        self.n, self.F, self.G, self.mode = n_envs, n_feat, n_glob, mode
        self.fmax = min(max_fraction, 0.3)
        self.rng = np.random.default_rng(seed + 12345)
        self.kind = [""] * n_envs
        self.params = [None] * n_envs
        for i in range(n_envs):
            self._new(i)

    def _new(self, i):
        rng = self.rng
        kind = self.mode if self.mode != "mixed" else ("episodic" if rng.random() < 0.5 else "iid")
        self.kind[i] = kind
        if kind == "episodic":
            w = rng.normal(0, 1, self.F) * (rng.random(self.F) < rng.uniform(0.1, 0.5))
            v = rng.normal(0, 1, self.G) * (rng.random(self.G) < 0.3)
            self.params[i] = dict(w=w / np.sqrt(max((w != 0).sum(), 1)), v=v / np.sqrt(max((v != 0).sum(), 1)),
                                  b=rng.normal(0, 0.5), gain=rng.uniform(1, 4),
                                  f=rng.uniform(0, 1) ** 2 * self.fmax)

    def episode_done(self, done):
        for i in np.nonzero(done)[0]:
            self._new(int(i))

    def act(self, obs) -> np.ndarray:
        act = np.zeros((self.n, N_CONT), np.float32)
        for i in range(self.n):
            if self.kind[i] == "episodic":
                p = self.params[i]
                z = obs["runners"][i].astype(np.float64) @ p["w"] + float(obs["global"][i] @ p["v"]) + p["b"]
                act[i, :R_MAX] = np.tanh(p["gain"] * z)
                act[i, -1] = p["f"]
            else:
                # small wager fraction: uniform f would bet half the bank every 2s
                act[i, :R_MAX] = self.rng.uniform(-1, 1, R_MAX)
                act[i, -1] = self.rng.uniform(0, 1) ** 3 * self.fmax
        act[:, :R_MAX] *= obs["mask"]
        return act


def train_sac(args):
    cfg, venv, va = _setup(args)
    device = torch.device(args.device)
    actor = Actor(args.d_model, args.n_layers).to(device)
    q1, q2 = Critic(args.d_model, args.n_layers).to(device), Critic(args.d_model, args.n_layers).to(device)
    q1t, q2t = Critic(args.d_model, args.n_layers).to(device), Critic(args.d_model, args.n_layers).to(device)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    for p in list(q1t.parameters()) + list(q2t.parameters()):
        p.requires_grad_(False)
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    log_alpha = torch.tensor(math.log(args.init_alpha), device=device, requires_grad=True)
    al_opt = torch.optim.Adam([log_alpha], lr=args.lr)

    N, C = args.n_envs, args.buffer
    obs, _ = venv.reset(seed=args.seed)
    R, F = obs["runners"].shape[1:]
    G = obs["global"].shape[1]
    B_r = np.zeros((C, R, F), np.float16)
    B_g = np.zeros((C, G), np.float32)
    B_m = np.zeros((C, R), np.bool_)
    B_a = np.zeros((C, N_CONT), np.float32)
    B_rew = np.zeros(C, np.float32)
    B_d = np.zeros(C, np.float32)
    B_r2 = np.zeros((C, R, F), np.float16)
    B_g2 = np.zeros((C, G), np.float32)
    B_m2 = np.zeros((C, R), np.bool_)
    ptr, full = 0, False
    rms = RunningMeanStd()
    logf = open(os.path.join(args.out, "train_log.csv"), "w")
    logf.write("iter,steps,sps,ep_green_mean,ep_turnover_mean,val_green,val_pct_green,val_turnover\n")
    stats, best, steps, t0 = [], [-np.inf], 0, time.time()
    n_iters = args.total_steps // N
    explorer = RandomStrategies(N, F, G, args.warmup_mode, getattr(args, "max_fraction", 1.0), args.seed)
    for it in range(1, n_iters + 1):
        if steps < args.warmup:
            act = explorer.act(obs)
        else:
            with torch.no_grad():
                _, a, _, _, _ = actor.sample(*obs_t(obs, device))
            act = a.cpu().numpy()
        nobs, rew, term, trunc, info = venv.step(act)
        _episode_stats(info, stats)
        done = np.logical_or(term, trunc)
        explorer.episode_done(done)
        # SAME_STEP autoreset: the true next obs of finished envs is in final_obs
        nxt = {k: nobs[k].copy() for k in ("runners", "global", "mask")}
        if "final_obs" in info:
            for i in np.nonzero(info["_final_obs"])[0]:
                for k in nxt:
                    nxt[k][i] = info["final_obs"][i][k]
        rms.update(rew)
        for i in range(N):
            B_r[ptr], B_g[ptr], B_m[ptr] = obs["runners"][i], obs["global"][i], obs["mask"][i]
            B_a[ptr], B_rew[ptr], B_d[ptr] = act[i], rew[i], float(term[i])
            B_r2[ptr], B_g2[ptr], B_m2[ptr] = nxt["runners"][i], nxt["global"][i], nxt["mask"][i]
            ptr = (ptr + 1) % C
            full = full or ptr == 0
        obs = nobs
        steps += N
        size = C if full else ptr
        if steps >= args.warmup and it % args.update_every == 0:
            for _ in range(args.grad_steps):
                idx = np.random.randint(0, size, args.batch)
                t = lambda x, dt=torch.float32: torch.as_tensor(x[idx], dtype=dt, device=device)
                r, g, m, a = t(B_r), t(B_g), t(B_m), t(B_a)
                r2, g2, m2 = t(B_r2), t(B_g2), t(B_m2)
                rw = t(B_rew) / math.sqrt(rms.var + 1e-8)
                d = t(B_d)
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    _, a2, lp2, _, _ = actor.sample(r2, g2, m2)
                    qt = torch.min(q1t(r2, g2, m2, a2), q2t(r2, g2, m2, a2)) - alpha * lp2
                    y = rw + args.gamma * (1 - d) * qt
                ql = Fn.mse_loss(q1(r, g, m, a), y) + Fn.mse_loss(q2(r, g, m, a), y)
                q_opt.zero_grad()
                ql.backward()
                nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10.0)
                q_opt.step()
                _, an, lp, _, _ = actor.sample(r, g, m)
                qn = torch.min(q1(r, g, m, an), q2(r, g, m, an))
                al = (alpha * lp - qn).mean()
                a_opt.zero_grad()
                al.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                a_opt.step()
                # target entropy scales with the number of active action dims
                tgt = -args.target_entropy_per_dim * (m.sum(-1) + 1)
                alpha_loss = -(log_alpha * (lp.detach() + tgt)).mean()
                al_opt.zero_grad()
                alpha_loss.backward()
                al_opt.step()
                with torch.no_grad():
                    for net, tn in ((q1, q1t), (q2, q2t)):
                        for p, pt in zip(net.parameters(), tn.parameters()):
                            pt.mul_(1 - args.tau).add_(args.tau * p)
        _maybe_eval(args, actor, cfg, va, it, best, logf, stats, t0, steps, device)
    venv.close()


# ============================================================================ PPO
def train_ppo(args):
    cfg, venv, va = _setup(args)
    device = torch.device(args.device)
    actor = Actor(args.d_model, args.n_layers).to(device)
    critic = ValueNet(args.d_model, args.n_layers).to(device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr, eps=1e-5)
    N, T = args.n_envs, args.n_steps
    obs, _ = venv.reset(seed=args.seed)
    R, F = obs["runners"].shape[1:]
    G = obs["global"].shape[1]
    b_r, b_g, b_m = torch.zeros(T, N, R, F), torch.zeros(T, N, G), torch.zeros(T, N, R)
    b_u, b_lp, b_v = torch.zeros(T, N, N_CONT), torch.zeros(T, N), torch.zeros(T, N)
    b_rew, b_d = torch.zeros(T, N), torch.zeros(T, N)
    rms, racc = RunningMeanStd(), np.zeros(N)
    logf = open(os.path.join(args.out, "train_log.csv"), "w")
    logf.write("iter,steps,sps,ep_green_mean,ep_turnover_mean,val_green,val_pct_green,val_turnover\n")
    stats, best, steps, t0 = [], [-np.inf], 0, time.time()
    for it in range(1, args.total_steps // (N * T) + 1):
        actor.eval()
        for t in range(T):
            r, g, m = obs_t(obs, device)
            with torch.no_grad():
                u, a, lp, _, _ = actor.sample(r, g, m)
                v = critic(r, g, m)
            b_r[t], b_g[t], b_m[t] = r.cpu(), g.cpu(), m.cpu()
            b_u[t], b_lp[t], b_v[t] = u.cpu(), lp.cpu(), v.cpu()
            obs, rew, term, trunc, info = venv.step(a.cpu().numpy())
            _episode_stats(info, stats)
            done = np.logical_or(term, trunc)
            racc = racc * args.gamma + rew
            rms.update(racc)
            racc[done] = 0
            b_rew[t] = torch.as_tensor(rew / math.sqrt(rms.var + 1e-8), dtype=torch.float32)
            b_d[t] = torch.as_tensor(done, dtype=torch.float32)
            steps += N
        with torch.no_grad():
            lv = critic(*obs_t(obs, device)).cpu()
        adv, gae = torch.zeros(T, N), torch.zeros(N)
        for t in reversed(range(T)):
            nv = lv if t == T - 1 else b_v[t + 1]
            nt = 1 - b_d[t]
            delta = b_rew[t] + args.gamma * nv * nt - b_v[t]
            gae = delta + args.gamma * args.gae_lambda * nt * gae
            adv[t] = gae
        ret = adv + b_v
        B = T * N
        fr, fg, fm = b_r.reshape(B, R, F), b_g.reshape(B, G), b_m.reshape(B, R)
        fu, flp, fadv, fret = b_u.reshape(B, N_CONT), b_lp.reshape(B), adv.reshape(B), ret.reshape(B)
        actor.train()
        idx = np.arange(B)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s in range(0, B, args.minibatch):
                mb = torch.as_tensor(idx[s:s + args.minibatch])
                r, g, m = fr[mb].to(device), fg[mb].to(device), fm[mb].to(device)
                mu, ls = actor(r, g, m)
                lp = actor.log_prob(mu, ls, fu[mb].to(device), actor.dim_mask(m))
                ratio = torch.exp(lp - flp[mb].to(device))
                A = fadv[mb].to(device)
                A = (A - A.mean()) / (A.std() + 1e-8)
                pg = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
                vl = 0.5 * (critic(r, g, m) - fret[mb].to(device)).pow(2).mean()
                ent = (ls * actor.dim_mask(m)).sum(-1).mean()  # gaussian entropy up to a constant
                loss = pg + 0.5 * vl - args.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
                opt.step()
        _maybe_eval(args, actor, cfg, va, it, best, logf, stats, t0, steps, device)
    venv.close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["sac", "ppo"], required=True)
    ap.add_argument("--tapes", default="data/tapes/*.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total-steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-every", type=int, default=0, help="iterations; 0 = auto")
    ap.add_argument("--eval-races", type=int, default=50)
    # SAC
    ap.add_argument("--buffer", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=10_000)
    ap.add_argument("--warmup-mode", default="iid", choices=["iid", "episodic", "mixed"],
                    help="random exploration before learning starts (see RandomStrategies)")
    ap.add_argument("--max-fraction", type=float, default=1.0, help="cap on the share of funds wagered per step")
    ap.add_argument("--decision-every", type=int, default=4, help="tape steps (0.5s) per decision")
    ap.add_argument("--update-every", type=int, default=1)
    ap.add_argument("--grad-steps", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--init-alpha", type=float, default=0.05)
    ap.add_argument("--target-entropy-per-dim", type=float, default=1.0)
    # PPO
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--ent-coef", type=float, default=0.001)
    a = ap.parse_args(argv)
    if a.eval_every == 0:  # ~every 50k env steps
        a.eval_every = max(1, 50_000 // (a.n_envs * (a.n_steps if a.algo == "ppo" else 1)))
    return a


if __name__ == "__main__":
    args = parse_args()
    (train_sac if args.algo == "sac" else train_ppo)(args)
