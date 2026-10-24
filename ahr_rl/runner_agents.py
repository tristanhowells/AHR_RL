"""P13c: per-runner agents. One shared policy, a copy per runner above the liquidity cut.

P13/P13b use one agent with 24 runner slots: one reward for the whole market, one shared
wager fraction split across runners, and one critic over all 25 action dims (most of them
forced to zero by the gate). Here the unit is the runner instead:

  * a runner "comes online" when it crosses the liquidity cut (same gate as P13: $ at its
    best back + lay prices, 10s trailing mean, live) and goes offline once it is below the
    cut AND flat. Below the cut a held runner is reduce-only, as before.
  * each online runner gets 2 actions every decision:
        c in [-1, 1]  side and conviction (> 0 back, < 0 lay; |c| < deadzone = no bet)
        s in [ 0, 1]  size: $ at risk = min_stake + s * (order_cap - min_stake)
                      (back: the stake; lay: the liability), s < 0.05 = no bet
    Orders cross the spread at the best price; leftovers are cancelled at the next decision.
    An order that reduces the runner's position is capped at flat. One that adds is capped
    by the per-runner and per-market worst-case loss limits (runner_cap, market_cap).
  * each runner is paid its OWN P&L: the change in its share of the green value (the env's
    per-runner decomposition, `runner_rewards`). Cross-runner hedges are ignored.
  * SAC with independent learners and shared weights: the actor and twin critics run one
    transformer over the whole market (so every runner sees the market context and the
    gate inputs), then a per-runner head. Q_r(s, a_r) sees only its own runner's action.
    Losses average over online runners; a runner that goes offline stops bootstrapping.

Everything else is P13: random-strategy warm-up, planted-edge positive control first,
checkpoint picked on VAL, scored once on TEST vs do-nothing and random, latency stress,
forensics. "edge" = every seed beats do-nothing on TEST with t > 2 and survives the stress.

    python -m ahr_rl.runner_agents --tapes "data/tapes/*.npz" --out runs/runner_agents --min-best-usd auto
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .blackbox import forensics, gate_summary, liquidity_cut, score
from .continuous import LOG_STD_MAX, LOG_STD_MIN, Backbone, RandomStrategies, act_mask_t, obs_t
from .env import list_tapes, split_by_date
from .env_continuous import ContinuousAllocEnv, ContinuousConfig
from .evaluate import evaluate, summarise
from .exchange import BACK, LAY
from .features import N_RUNNER_FEATURES, R_MAX
from .ladder import PRICES
from .synthetic import write_synthetic
from .train import RunningMeanStd

N_ACT = 2  # per runner: conviction, size
MIN_SIZE = 0.05


@dataclass
class RunnerAgentConfig(ContinuousConfig):
    order_cap: float = 20.0  # $ at risk per order at s = 1
    runner_cap: float = 50.0  # no adding to a runner once its worst-case loss is this
    market_cap: float = 150.0  # no adding anywhere once the market's worst-case loss is this


class RunnerAgentEnv(ContinuousAllocEnv):
    """Action = 48 floats: c_0..c_23 in [-1, 1], then s_0..s_23 in [0, 1]."""

    def __init__(self, tapes, cfg: RunnerAgentConfig | None = None, **kw):
        super().__init__(tapes, cfg or RunnerAgentConfig(), **kw)
        lo = np.r_[np.full(R_MAX, -1.0), np.zeros(R_MAX)].astype(np.float32)
        self.action_space = gym.spaces.Box(lo, np.ones(N_ACT * R_MAX, np.float32), dtype=np.float32)

    def _apply(self, action):
        ex, cfg = self.ex, self.cfg
        action = np.asarray(action, np.float64).reshape(-1)
        c = np.clip(action[:R_MAX], -1, 1)
        s = np.clip(action[R_MAX:2 * R_MAX], 0, 1)
        ex.cancel_all()
        liq = self.liquid()
        expo = ex.W - ex.L  # > 0 long (backed), < 0 short (laid)
        mkt_loss = max(0.0, -ex.worst_case())
        for r in range(min(self.tape.n_runners, R_MAX)):
            if abs(c[r]) < cfg.deadzone or s[r] < MIN_SIZE or not self.tape.active[ex.step, r] or ex.void_runner[r]:
                continue
            back = c[r] > 0
            bb, bl = ex.best(ex.step, r)
            tick = bb if back else bl
            if tick < 0:
                continue
            reduces = (back and expo[r] < -1e-6) or (not back and expo[r] > 1e-6)
            if not liq[r] and not reduces:
                continue  # below the gate: reduce only
            risk = ex.cfg.min_stake + s[r] * max(cfg.order_cap - ex.cfg.min_stake, 0.0)
            if not reduces:
                run_loss = max(0.0, -min(ex.W[r], ex.L[r]))
                risk = min(risk, cfg.runner_cap - run_loss, cfg.market_cap - mkt_loss)
                if risk <= 0:
                    continue
            stake = risk if back else risk / max(PRICES[tick] - 1.0, 0.01)
            if reduces:
                hp = ex.hedge_plan(r)
                if hp is None:
                    continue
                stake = min(stake, hp[2])
            o = ex.submit(r, BACK if back else LAY, tick, stake, is_hedge=reduces)
            if o is not None:
                if reduces:
                    self.n_closes += 1
                else:
                    self.n_opens += 1
                    mkt_loss += risk


# ============================================================================ networks
class RunnerActor(nn.Module):
    """Per-runner squashed Gaussian over n_act actions, with the market seen through attention.
    Action 0 (conviction) is in [-1, 1]; the others (size, and P17's price offset) are in [0, 1]."""

    def __init__(self, d=64, layers=2, runner_extra=0, size_bias=-1.5, init_log_std=-0.5, n_act=N_ACT):
        super().__init__()
        self.n_act = n_act
        self.bb = Backbone(d, layers, runner_extra=runner_extra)
        self.head = nn.Linear(2 * d, 2 * n_act)
        with torch.no_grad():
            self.head.weight.mul_(0.01)
            self.head.bias.zero_()
            self.head.bias[1] = size_bias  # start small: s ~ 0.05
            self.head.bias[n_act:] = init_log_std

    def forward(self, runners, glob, mask):
        hm, hr = self.bb(runners, glob, mask)
        o = self.head(torch.cat([hr, hm[:, None].expand_as(hr)], -1))  # [B, R, 2 * n_act]
        return o[..., :self.n_act], o[..., self.n_act:].clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, runners, glob, mask, act_mask, deterministic=False):
        """-> pre-tanh u, action [B, R, n_act] (zero where act_mask is 0), log pi per runner [B, R]."""
        mu, log_std = self(runners, glob, mask)
        u = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        a = torch.tanh(u)
        a = torch.cat([a[..., :1], (a[..., 1:] + 1) / 2], -1) * act_mask[..., None]
        lp = -0.5 * ((u - mu) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        lp = lp - 2 * (math.log(2) - u - Fn.softplus(-2 * u))
        return u, a, lp.sum(-1)


class RunnerCritic(nn.Module):
    """Q_r(s, a_r) for every runner at once; runner r's value sees only its own action."""

    def __init__(self, d=64, layers=2, runner_extra=0, n_act=N_ACT):
        super().__init__()
        self.bb = Backbone(d, layers, runner_extra=runner_extra)
        self.out = nn.Sequential(nn.Linear(2 * d + n_act, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, runners, glob, mask, act):
        hm, hr = self.bb(runners, glob, mask)
        return self.out(torch.cat([hr, hm[:, None].expand_as(hr), act], -1)).squeeze(-1)


def flat_action(a) -> np.ndarray:
    """[.., R, n_act] -> [.., R * n_act]: all runners' action 0, then all action 1, ..."""
    a = np.asarray(a)
    return np.concatenate([a[..., i] for i in range(a.shape[-1])], -1).astype(np.float32)


def make_runner_policy(actor, device="cpu"):
    @torch.no_grad()
    def f(obs, env):
        r, g, m = obs_t(obs, device)
        am = act_mask_t(obs, device)
        _, a, _ = actor.sample(r[None], g[None], m[None], am[None], deterministic=True)
        return flat_action(a[0].cpu().numpy())

    return f


def noop_policy(obs, env):
    return np.zeros(N_ACT * R_MAX, np.float32)


def random_runner_policy(size: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)

    def pol(obs, env):
        c = rng.uniform(-1, 1, R_MAX) * obs.get("act_mask", obs["mask"])
        return np.r_[c, np.full(R_MAX, size)].astype(np.float32)

    return pol


def load_runner_actor(path: str, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    ar = ck["args"]
    actor = RunnerActor(ar["d_model"], ar["n_layers"], ar.get("runner_extra", 0), n_act=ar.get("n_act", N_ACT)).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    return actor


# ============================================================================ training
def train(tapes_glob: str, out: str, cfg: RunnerAgentConfig, steps: int, warmup: int, seed: int, n_envs: int,
          device: str, buffer: int = 400_000, batch: int = 256, update_every: int = 2, grad_steps: int = 1,
          lr: float = 3e-4, gamma: float = 0.995, tau: float = 0.005, init_alpha: float = 0.05,
          target_entropy_per_dim: float = 1.0, d_model: int = 64, n_layers: int = 2, eval_races: int = 100,
          warmup_mode: str = "mixed", env_cls=None, n_act: int = N_ACT, split=None) -> str:
    """Independent-learner SAC with shared weights. Returns the path of the VAL-best checkpoint.
    split: optional (train, val, test) lists of tape paths instead of the chronological split of tapes_glob."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(out, exist_ok=True)
    tr, va, te = split if split is not None else split_by_date(list_tapes(tapes_glob))
    json.dump({"train": tr, "val": va, "test": te}, open(os.path.join(out, "split.json"), "w"), indent=1)
    env_cls = env_cls or RunnerAgentEnv
    tcfg = replace(cfg, random_start_s=30.0)
    fns = [(lambda i=i: env_cls(tr, tcfg, seed=seed + i)) for i in range(n_envs)]
    venv = gym.vector.AsyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    dev = torch.device(device)
    obs, _ = venv.reset(seed=seed)
    N, R, F = obs["runners"].shape
    G = obs["global"].shape[1]
    extra = F - N_RUNNER_FEATURES
    args = dict(d_model=d_model, n_layers=n_layers, runner_extra=extra, n_act=n_act, steps=steps, warmup=warmup,
                seed=seed, env=asdict(cfg))
    json.dump(args, open(os.path.join(out, "config.json"), "w"), indent=1, default=str)
    actor = RunnerActor(d_model, n_layers, extra, n_act=n_act).to(dev)
    q1, q2, q1t, q2t = (RunnerCritic(d_model, n_layers, extra, n_act=n_act).to(dev) for _ in range(4))
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    for p in list(q1t.parameters()) + list(q2t.parameters()):
        p.requires_grad_(False)
    a_opt = torch.optim.Adam(actor.parameters(), lr=lr)
    q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    log_alpha = torch.tensor(math.log(init_alpha), device=dev, requires_grad=True)
    al_opt = torch.optim.Adam([log_alpha], lr=lr)
    tgt_ent = -target_entropy_per_dim * n_act

    C = buffer
    B_r, B_r2 = np.zeros((C, R, F), np.float16), np.zeros((C, R, F), np.float16)
    B_g, B_g2 = np.zeros((C, G), np.float32), np.zeros((C, G), np.float32)
    B_m, B_m2 = np.zeros((C, R), np.bool_), np.zeros((C, R), np.bool_)
    B_am, B_am2 = np.zeros((C, R), np.bool_), np.zeros((C, R), np.bool_)
    B_a = np.zeros((C, R, n_act), np.float32)
    B_rew = np.zeros((C, R), np.float32)
    B_d = np.zeros(C, np.float32)
    ptr, full = 0, False
    rms = RunningMeanStd()
    explorer = RandomStrategies(N, F, G, warmup_mode, 0.3, seed)
    logf = open(os.path.join(out, "train_log.csv"), "w")
    logf.write("iter,steps,sps,online_runners,ep_green_mean,ep_turnover_mean,val_green,val_pct_green,val_turnover\n")
    stats, online, best, n_steps, t0 = [], [], -np.inf, 0, time.time()
    n_iters = steps // N
    eval_every = max(1, (steps - warmup) // N // 10)  # ~10 checkpoints, all after the random warm-up
    for it in range(1, n_iters + 1):
        am_np = obs["act_mask"].astype(bool)
        if n_steps < warmup:
            ra = explorer.act(obs)  # convictions (masked) + one wager fraction per env
            act = np.zeros((N, R, n_act), np.float32)
            act[..., 0] = ra[:, :R_MAX]
            act[..., 1] = np.clip(ra[:, -1:] / 0.3, 0, 1) * am_np
            if n_act > 2:  # further actions (P17's price offset): uniform per runner and step
                act[..., 2:] = np.random.uniform(0, 1, (N, R, n_act - 2)) * am_np[..., None]
        else:
            with torch.no_grad():
                _, a, _ = actor.sample(*obs_t(obs, dev), act_mask_t(obs, dev))
            act = a.cpu().numpy()
        nobs, _, term, trunc, info = venv.step(flat_action(act))
        done = np.logical_or(term, trunc)
        explorer.episode_done(done)
        rr = np.array(info.get("runner_rewards", np.zeros((N, R_MAX))), np.float32)
        nxt = {k: nobs[k].copy() for k in ("runners", "global", "mask", "act_mask")}
        fin = info.get("final_info")
        if fin is not None and "worst" in fin:
            for i in np.nonzero(fin["_worst"])[0]:
                stats.append((float(fin["worst"][i]), float(fin["turnover"][i])))
        if "final_obs" in info:
            for i in np.nonzero(info["_final_obs"])[0]:
                for k in nxt:
                    nxt[k][i] = info["final_obs"][i][k]
                if fin is not None and "runner_rewards" in fin:
                    rr[i] = fin["runner_rewards"][i]
        if am_np.any():
            rms.update(rr[am_np])
        online.append(am_np.sum(1).mean())
        for i in range(N):
            B_r[ptr], B_g[ptr], B_m[ptr], B_am[ptr] = obs["runners"][i], obs["global"][i], obs["mask"][i], am_np[i]
            B_a[ptr], B_rew[ptr], B_d[ptr] = act[i], rr[i], float(term[i])
            B_r2[ptr], B_g2[ptr], B_m2[ptr], B_am2[ptr] = (nxt["runners"][i], nxt["global"][i], nxt["mask"][i],
                                                           nxt["act_mask"][i])
            ptr = (ptr + 1) % C
            full = full or ptr == 0
        obs = nobs
        n_steps += N
        size = C if full else ptr
        if n_steps >= warmup and it % update_every == 0:
            for _ in range(grad_steps):
                idx = np.random.randint(0, size, batch)
                idx = idx[B_am[idx].any(1)]  # steps with at least one online runner
                if len(idx) < 8:
                    continue
                t = lambda x: torch.as_tensor(x[idx], dtype=torch.float32, device=dev)
                r, g, m, am, a = t(B_r), t(B_g), t(B_m), t(B_am), t(B_a)
                r2, g2, m2, am2 = t(B_r2), t(B_g2), t(B_m2), t(B_am2)
                rw = t(B_rew) / math.sqrt(rms.var + 1e-8)
                d = t(B_d)[:, None]
                n_on = am.sum().clamp(min=1)
                alpha = log_alpha.exp().detach()
                with torch.no_grad():
                    _, a2, lp2 = actor.sample(r2, g2, m2, am2)
                    qt = torch.min(q1t(r2, g2, m2, a2), q2t(r2, g2, m2, a2)) - alpha * lp2
                    y = rw + gamma * (1 - d) * am2 * qt  # offline next step: no more value for that runner
                ql = ((((q1(r, g, m, a) - y) ** 2) + ((q2(r, g, m, a) - y) ** 2)) * am).sum() / n_on
                q_opt.zero_grad()
                ql.backward()
                nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10.0)
                q_opt.step()
                _, an, lp = actor.sample(r, g, m, am)
                qn = torch.min(q1(r, g, m, an), q2(r, g, m, an))
                al = ((alpha * lp - qn) * am).sum() / n_on
                a_opt.zero_grad()
                al.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                a_opt.step()
                alpha_loss = -(log_alpha * ((lp.detach() + tgt_ent) * am).sum() / n_on)
                al_opt.zero_grad()
                alpha_loss.backward()
                al_opt.step()
                with torch.no_grad():
                    for net, tn in ((q1, q1t), (q2, q2t)):
                        for p, pt in zip(net.parameters(), tn.parameters()):
                            pt.mul_(1 - tau).add_(tau * p)
        # no checkpoints during the warm-up: the untrained actor's deterministic action is "no bet" ($0 on VAL)
        # and would otherwise win "best" whenever every trained checkpoint loses
        if n_steps >= warmup and (it % eval_every == 0 or it == n_iters):
            actor.eval()
            val = summarise(evaluate(make_runner_policy(actor, dev), va[:eval_races], cfg, env_cls=env_cls))
            actor.train()
            ck = {"actor": actor.state_dict(), "args": args, "iter": it, "val": val}
            if val["mean_green_$"] > best:
                best = val["mean_green_$"]
                torch.save(ck, os.path.join(out, "best.pt"))
            torch.save(ck, os.path.join(out, "last.pt"))
            eg = np.mean([x[0] for x in stats[-50:]]) if stats else np.nan
            et = np.mean([x[1] for x in stats[-50:]]) if stats else np.nan
            sps = n_steps / (time.time() - t0)
            on = float(np.mean(online[-1000:]))
            logf.write(f"{it},{n_steps},{sps:.0f},{on:.2f},{eg:.3f},{et:.1f},{val['mean_green_$']:.3f},"
                       f"{val['pct_green']:.1f},{val['mean_turnover_$']:.1f}\n")
            logf.flush()
            print(f"[runner-sac] it {it} steps {n_steps} sps {sps:.0f} | online runners {on:.2f} | train green "
                  f"{eg:+.2f} turnover {et:.0f} | VAL green {val['mean_green_$']:+.2f} ({val['pct_green']:.0f}% "
                  f"green, turnover {val['mean_turnover_$']:.0f})", flush=True)
    venv.close()
    if not os.path.exists(os.path.join(out, "best.pt")):  # run too short for a learning-phase checkpoint
        ck = {"actor": actor.state_dict(), "args": args, "iter": n_iters, "val": None}
        torch.save(ck, os.path.join(out, "best.pt"))
        torch.save(ck, os.path.join(out, "last.pt"))
    return os.path.join(out, "best.pt")


def test_block(label, ckpt, test, cfg, device, rows, stress=True, env_cls=None, extra_stress=()):
    """VAL-picked checkpoint (+ latency stress, + any extra (name, cfg) stresses) -> (policy, agent row,
    [stress rows]); the last checkpoint is scored too, for information."""
    env_cls = env_cls or RunnerAgentEnv
    pol = make_runner_policy(load_runner_actor(ckpt, device), device)
    main = dict(run=label, **score("agent", evaluate(pol, test, cfg, env_cls=env_cls)))
    rows.append(main)
    stress_rows = []
    if stress:
        slow = replace(cfg, exchange=replace(cfg.exchange, latency_steps=2))
        for name, c in (("agent, 1s latency", slow), *extra_stress):
            stress_rows.append(dict(run=label, **score(name, evaluate(pol, test, c, env_cls=env_cls))))
            rows.append(stress_rows[-1])
    last = os.path.join(os.path.dirname(ckpt), "last.pt")
    if os.path.exists(last):
        lp = make_runner_policy(load_runner_actor(last, device), device)
        rows.append(dict(run=label, **score("last checkpoint (info)", evaluate(lp, test, cfg, env_cls=env_cls))))
    return pol, main, stress_rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-best-usd", default="auto",
                    help='liquidity gate in $ at best back + lay; "auto" = P11 top quintile on TRAIN days; 0 = off')
    ap.add_argument("--no-gate-inputs", action="store_true", help="without the (above cut, distance) inputs")
    ap.add_argument("--order-cap", type=float, default=20.0, help="$ at risk per order at full size")
    ap.add_argument("--runner-cap", type=float, default=50.0, help="max worst-case loss built on one runner")
    ap.add_argument("--market-cap", type=float, default=150.0, help="max worst-case loss built on the market")
    ap.add_argument("--steps", type=int, default=600_000, help="env steps (market decisions) per real-data seed")
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--decision-every", type=int, default=20, help="tape steps (0.5s) per decision; 20 = 10s")
    ap.add_argument("--update-every", type=int, default=2)
    ap.add_argument("--grad-steps", type=int, default=1)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--control-steps", type=int, default=150_000)
    ap.add_argument("--control-races", type=int, default=200)
    ap.add_argument("--trend-every-s", type=float, default=60.0)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--control-only", action="store_true")
    ap.add_argument("--n-envs", type=int, default=max(4, os.cpu_count() or 4))
    ap.add_argument("--eval-races", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    tapes = list_tapes(a.tapes)
    tr, _, te = split_by_date(tapes)
    cut = liquidity_cut(tr) if a.min_best_usd == "auto" else float(a.min_best_usd)
    cfg = RunnerAgentConfig(random_start_s=0.0, decision_every=a.decision_every, min_best_usd=cut,
                            gate_inputs=not a.no_gate_inputs, order_cap=a.order_cap, runner_cap=a.runner_cap,
                            market_cap=a.market_cap)
    gates = {}
    print(f"=== P13c per-runner agents: gate ${cut:.2f} at best back + lay (10s mean)"
          f"{' [auto: P11 top quintile, TRAIN days]' if a.min_best_usd == 'auto' else ''}; orders up to "
          f"${a.order_cap:.0f} at risk, runner cap ${a.runner_cap:.0f}, market cap ${a.market_cap:.0f} ===")
    if cut > 0:
        g = gates["real_test"] = gate_summary(te, cut)
        print(f"TEST races, last 5 min: {g['runner_share_open'] * 100:.0f}% of runner snapshots online; "
              f"{g['races_with_any_open']} of {g['races']} races have an online runner at some point; "
              f"{g['mean_open_runners']:.1f} online per snapshot on average\n", flush=True)
    kw = dict(n_envs=a.n_envs, device=a.device, buffer=a.buffer, update_every=a.update_every,
              grad_steps=a.grad_steps)
    rows = []

    control_ok = None
    if not a.skip_control:
        print(f"=== 0. Positive control: planted edge (a runner shortens 1 tick / {a.trend_every_s:.0f}s) ===",
              flush=True)
        syn = os.path.join(a.out, "synthetic")
        write_synthetic(syn, a.control_races, seed0=50_000, trend_every_s=a.trend_every_s)
        glob_ = os.path.join(syn, "*.npz")
        _, _, cte = split_by_date(list_tapes(glob_))
        if cut > 0:
            gates["control_test"] = gate_summary(cte, cut)
        ck = train(glob_, os.path.join(a.out, "control"), cfg, a.control_steps, int(a.control_steps * a.warmup_frac),
                   0, eval_races=min(a.eval_races, 30), **kw)
        rows.append(dict(run="control", **score("do nothing", evaluate(noop_policy, cte, cfg, env_cls=RunnerAgentEnv))))
        rows.append(dict(run="control", **score("random", evaluate(random_runner_policy(), cte, cfg,
                                                                     env_cls=RunnerAgentEnv))))
        _, ctl, _ = test_block("control", ck, cte, cfg, a.device, rows, stress=False)
        control_ok = bool(ctl["mean_green"] > 0 and (ctl["t"] or 0) > 2)
        print(pd.DataFrame([r for r in rows if r["run"] == "control"]).round(3).to_string(index=False))
        print("control: the per-runner agents FIND a small planted edge" if control_ok else
              "control: the planted edge was NOT found -> a null result on real data means little; "
              "try more --control-steps / --steps", flush=True)

    if a.control_only:
        json.dump(dict(control_found_planted_edge=control_ok, gate=gates), open(os.path.join(a.out, "verdict.json"),
                                                                                  "w"), indent=1, default=float)
        print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")
        return

    print(f"\n=== 1. Real races: {len(tapes)} tapes, {len(te)} TEST races ===", flush=True)
    rows.append(dict(run="real", **score("do nothing", evaluate(noop_policy, te, cfg, env_cls=RunnerAgentEnv))))
    rows.append(dict(run="real", **score("random", evaluate(random_runner_policy(), te, cfg, env_cls=RunnerAgentEnv))))
    pols, results = {}, {}
    for sd in [int(s) for s in a.seeds.split(",") if s.strip()]:
        print(f"\n--- seed {sd}: {a.steps} steps, first {int(a.steps * a.warmup_frac)} on random strategies ---",
              flush=True)
        ck = train(a.tapes, os.path.join(a.out, f"real_seed{sd}"), cfg, a.steps, int(a.steps * a.warmup_frac), sd,
                   eval_races=a.eval_races, **kw)
        pols[sd], main_row, stress_rows = test_block(f"seed {sd}", ck, te, cfg, a.device, rows)
        results[sd] = (main_row, stress_rows[0])

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    print("\n=== 2. TEST races (never used for training or checkpoint selection); $ per race, $500 bank ===")
    print(res.round(3).to_string(index=False))

    best_sd = max(results, key=lambda s: results[s][0]["mean_green"]) if results else None
    if best_sd is not None:
        fz = forensics(pols[best_sd], te[:100], cfg, env_cls=RunnerAgentEnv)
        fz.to_csv(os.path.join(a.out, f"fills_seed{best_sd}.csv"), index=False)
        print(f"\n=== What seed {best_sd} trades on TEST races ({len(fz)} fills) ===")
        if len(fz):
            print(fz.pivot_table(index="rank", columns=["side"], values="stake", aggfunc="sum", fill_value=0)
                  .round(0).to_string())
            print(fz.pivot_table(index="time", columns=["side"], values="stake", aggfunc="sum", fill_value=0,
                                 observed=False).round(0).to_string())
            if cut > 0:
                print(f"share of $ staked on runners at/above the gate when filled: "
                      f"{fz.loc[fz['best_usd'] >= cut, 'stake'].sum() / max(fz['stake'].sum(), 1e-9) * 100:.0f}%")
        else:
            print("no fills: the agents learned not to trade")

    edge = bool(results) and all(r[0]["mean_green"] > 0 and (r[0]["t"] or 0) > 2 and r[1]["mean_green"] > 0
                                 for r in results.values())
    verdict = dict(control_found_planted_edge=control_ok, edge=edge, gate=gates,
                   caps=dict(order=a.order_cap, runner=a.runner_cap, market=a.market_cap),
                   seeds={str(s): dict(test_green=r[0]["mean_green"], t=r[0]["t"], turnover=r[0]["turnover"],
                                       latency_stress_green=r[1]["mean_green"]) for s, r in results.items()})
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    if edge:
        print("EDGE: every seed beats do-nothing on TEST races and survives the latency stress -> study the forensics, "
              "then forward-test it like P3c")
    elif control_ok is False:
        print("INCONCLUSIVE: the agents could not find a planted edge either")
    else:
        print("NO EDGE: per-runner agents find nothing on unseen races that beats doing nothing")
    print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
