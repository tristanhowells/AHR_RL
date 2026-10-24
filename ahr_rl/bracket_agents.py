"""P17b: per-runner agents whose action is a whole trade - an entry AND its exit plan - rewarded when it closes.

P17 rewarded every decision with the change in mark-to-market value: crossing costs show at once, gains only
later, and with gamma = 0.995 a gain two minutes on is discounted ~5% against a cost paid now. Agents learned to
stop trading before they had seen enough of the later gains. P17b keeps the same market, gate, caps and stresses
but changes how the agent acts and is paid:

  ACTION (per free runner, every 10s; 6 numbers):
    c   side (> 0 back, < 0 lay); |c| < dead-zone = no trade
    s   size: $5 .. --order-cap at risk (back: stake, lay: liability), <= 20% of the money shown
    o   entry price: cross now, or rest 1 / 2 / 3 ticks better (a resting entry waits up to 120s)
    h   hold: 30 / 60 / 120 / 300s after the first fill, or until the off (then closed at BSP)
    tp  take-profit: none / 1 / 2 / 3 / 5 / 8 ticks - rests as the green-up order from the moment it fills
    sl  stop: none / 3 / 6 / 10 ticks against the entry - checked every decision, closes by crossing
  The env runs the trade: entry, take-profit, stop, time exit (crossing, up to 2 ticks through the best price),
  and closes anything still open at the off at BSP. While a runner's trade is live its actions are ignored;
  the observation shows the trade (live, share of the hold used, ticks in or out of the money).

  REWARD: zero until the trade closes, then its realised green (min of the runner's win / lose P&L after the
  close, net of commission on a profit), / $5, to that runner. No immediate crossing penalty: the cost is in
  the green at the close. Trained with gamma = 1 (episodes are one race) and n-step returns, so the close
  reaches the entry decision in one update.

  WARM START: --demo-frac of the warm-up episodes are played by P16 / arm C's drift-fade (back a 2nd/3rd
  favourite after a bottom-10% 30s drift, rest 3 ticks above best, hold to the off at BSP), so the replay
  buffer holds worked positive-EV trades before SAC learns anything.

Same checks as P17: resting-only positive control (bounce races), TEST once, 1s latency and pessimistic-fill
stresses, ceiling flag against P15/P16, forensics of the trades it takes (exit reasons, holds, greens).

    python -m ahr_rl.bracket_agents --tapes "data/tapes/*.npz" --out runs/bracket_agents --min-matched 20000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from .blackbox import gate_summary, liquidity_cut, score
from .env import list_tapes, split_by_date
from .env_continuous import ContinuousAllocEnv
from .evaluate import evaluate, run_episode
from .exchange import BACK, LAY
from .features import R_MAX
from .ladder import N_TICKS, PRICES
from .market_select import select_markets
from .rebalance_study import _prob_grid
from .runner_agents import MIN_SIZE, RunnerAgentConfig, test_block, train
from .synthetic import write_synthetic

N_ACT6 = 6
HOLDS = (30, 60, 120, 300, None)  # None = until the off, closed at BSP
TPS = (0, 1, 2, 3, 5, 8)
SLS = (0, 3, 6, 10)
ENTRY_WAIT_S = 120.0
N_BRACKET_INPUTS = 3
CEILING = 0.8


@dataclass
class BracketConfig(RunnerAgentConfig):
    size_frac_cap: float = 0.2
    close_slippage_ticks: int = 2
    demo_cut: float = -0.38  # drift-fade demo: mom_30 at or below this (pp of implied probability in 30s)


class BracketAgentEnv(ContinuousAllocEnv):
    """Action = 144 floats: c, s, o, h, tp, sl for each of 24 runner slots (all c first, then all s, ...)."""

    def __init__(self, tapes, cfg: BracketConfig | None = None, **kw):
        cfg = cfg or BracketConfig()
        super().__init__(tapes, replace(cfg, auto_green_s=None), **kw)
        lo = np.r_[np.full(R_MAX, -1.0), np.zeros((N_ACT6 - 1) * R_MAX)].astype(np.float32)
        self.action_space = gym.spaces.Box(lo, np.ones(N_ACT6 * R_MAX, np.float32), dtype=np.float32)
        r = self.observation_space.spaces["runners"]
        self.observation_space = gym.spaces.Dict(dict(self.observation_space.spaces, runners=gym.spaces.Box(
            -10, 10, (R_MAX, r.shape[1] + N_BRACKET_INPUTS), np.float32)))

    # ------------------------------------------------------------------ episode
    def reset(self, *, seed=None, options=None):
        o, info = super().reset(seed=seed, options=options)
        self.br = {}  # runner -> live trade
        self.trades = []  # closed trades (forensics)
        self._rr = np.zeros(R_MAX, np.float32)
        self._p = None  # implied-probability grid for the demo, built on first use
        return self._obs(), info

    def _obs(self):
        o = super()._obs()
        x = np.zeros((R_MAX, N_BRACKET_INPUTS), np.float32)
        am = o["act_mask"].copy()
        if hasattr(self, "br"):
            for r, b in self.br.items():
                if r >= R_MAX:
                    continue
                x[r, 0] = 1.0
                if b["hold_until"] is not None and b["opened"] is not None:
                    x[r, 1] = np.clip((self.ex.step - b["opened"]) / max(b["hold_until"] - b["opened"], 1), 0, 1)
                bb, bl = self.ex.best(self.ex.step, r)
                if b["entry_tick"] is not None and bb >= 0 and bl >= 0:
                    mid = (bb + bl) / 2.0
                    x[r, 2] = np.clip(((b["entry_tick"] - mid) if b["side"] == BACK else (mid - b["entry_tick"])) / 10, -1, 1)
                am[r] = 1  # a live trade keeps the runner in the episode (its reward comes at the close)
        o["runners"] = np.concatenate([o["runners"], x], 1)
        o["act_mask"] = am
        return o

    # ------------------------------------------------------------------ trades
    def _value(self, r) -> float:
        return float(min(self.ex.W[r], self.ex.L[r]))

    def _finish(self, r, reason: str):
        b = self.br.pop(r)
        g = self._value(r) - b["v0"]
        c = self.ex.commission
        g = g * (1 - c) if g > 0 else g
        if r < R_MAX:
            self._rr[r] += g / self.cfg.reward_scale
        self.trades.append(dict(runner=r, side="back" if b["side"] == BACK else "lay", k=b["k"], hold=b["hold_s"],
                                tp=b["tp"], sl=b["sl"], green=g, reason=reason, t_open=b["t_open"],
                                t_close=float(self.tape.t_rel[self.ex.step]), filled=b["entry_tick"] is not None))

    def _cross_out(self, r):
        """Close a runner's position by crossing (up to close_slippage_ticks through the best price)."""
        ex = self.ex
        ex.cancel_runner(r)
        plan = ex.hedge_plan(r)
        if plan is None:
            return
        side, tk, stake = plan
        tk = tk - self.cfg.close_slippage_ticks if side == BACK else tk + self.cfg.close_slippage_ticks
        ex.submit(r, side, int(np.clip(tk, 0, N_TICKS - 1)), stake, is_hedge=True)

    def _bsp_out(self, r):
        ex, t = self.ex, self.tape
        ex.cancel_runner(r)
        d = ex.W[r] - ex.L[r]
        if abs(d) < 1e-6 or ex.void_runner[r]:
            return
        if t.bsp[r] > 1.0:
            bsp = float(t.bsp[r])
            if d > 0:
                ex._add_bet(r, LAY, bsp, d / bsp, passive=False)
            else:
                ex._add_bet(r, BACK, bsp, -d / bsp, passive=False)
        else:
            self._cross_out(r)

    def _flat(self, r) -> bool:
        return abs(self.ex.W[r] - self.ex.L[r]) < 0.01

    def _manage(self):
        """Advance every live trade: entry fills -> exit orders; take-profit / stop / time exits; closes."""
        ex, step = self.ex, self.ex.step
        for r in list(self.br):
            b = self.br[r]
            if ex.void_runner[r] or not self.tape.active[step, r]:
                self._finish(r, "void")
                continue
            matched = sum(f.stake for f in ex.fills if f.runner == r and f.side == b["side"] and f.step >= b["placed"])
            if b["phase"] == "entering":
                done_wait = step >= b["entry_until"] or matched >= b["stake"] - 1e-6
                if not done_wait:
                    continue
                ex.cancel_runner(r)
                if matched < 0.01:
                    self.br.pop(r)  # never filled: no trade, no reward
                    continue
                fills = [f for f in ex.fills if f.runner == r and f.side == b["side"] and f.step >= b["placed"]]
                p_in = sum(f.price * f.stake for f in fills) / sum(f.stake for f in fills)
                b["entry_tick"] = int(np.argmin(np.abs(PRICES - p_in)))
                b["opened"] = step
                b["hold_until"] = None if b["hold_s"] is None else step + int(round(b["hold_s"] / self.tape.dt))
                b["phase"] = "open"
                if b["tp"] > 0:  # the green-up order rests from now on
                    tp_tick = b["entry_tick"] - b["tp"] if b["side"] == BACK else b["entry_tick"] + b["tp"]
                    tp_tick = int(np.clip(tp_tick, 0, N_TICKS - 1))
                    stake = abs(ex.W[r] - ex.L[r]) / PRICES[tp_tick]
                    ex.submit(r, LAY if b["side"] == BACK else BACK, tp_tick, stake, is_hedge=True)
                continue
            if b["phase"] == "open":
                if self._flat(r):
                    self._finish(r, "take-profit")
                    continue
                bb, bl = ex.best(step, r)
                stopped = b["sl"] > 0 and bb >= 0 and bl >= 0 and (
                    (b["side"] == BACK and bl >= b["entry_tick"] + b["sl"]) or
                    (b["side"] == LAY and bb <= b["entry_tick"] - b["sl"]))
                timed = b["hold_until"] is not None and step >= b["hold_until"]
                if stopped or timed:
                    b["phase"], b["why"] = "closing", "stop" if stopped else "time"
                    self._cross_out(r)
                continue
            if b["phase"] == "closing":
                if self._flat(r):
                    self._finish(r, b["why"])
                else:
                    self._cross_out(r)  # not all matched yet: re-price at the new best

    def _apply(self, action):
        ex, cfg, t = self.ex, self.cfg, self.tape
        a = np.asarray(action, np.float64).reshape(N_ACT6, R_MAX)
        c, s = np.clip(a[0], -1, 1), np.clip(a[1], 0, 1)
        o_, h_, tp_, sl_ = (np.clip(a[i], 0, 1) for i in range(2, 6))
        liq = self.liquid()
        mkt_loss = max(0.0, -ex.worst_case())
        step = ex.step
        for r in range(min(t.n_runners, R_MAX)):
            if r in self.br or abs(c[r]) < cfg.deadzone or s[r] < MIN_SIZE or not liq[r]:
                continue
            if not t.active[step, r] or ex.void_runner[r] or not self._flat(r):
                continue
            back = c[r] > 0
            k = min(int(o_[r] * 4), 3)
            bb, bl = ex.best(step, r)
            if bb < 0 or bl < 0:
                continue
            tick = (bb + k) if back else (bl - k)
            if not 0 <= tick < N_TICKS:
                continue
            risk = ex.cfg.min_stake + s[r] * max(cfg.order_cap - ex.cfg.min_stake, 0.0)
            run_loss = max(0.0, -min(ex.W[r], ex.L[r]))
            risk = min(risk, cfg.runner_cap - run_loss, cfg.market_cap - mkt_loss)
            if risk <= 0:
                continue
            p = PRICES[tick]
            stake = risk if back else risk / max(p - 1.0, 0.01)
            shown = float((t.back_size if (back == (k == 0)) else t.lay_size)[step, r, 0])
            stake = min(stake, max(ex.cfg.min_stake, cfg.size_frac_cap * shown))
            o = ex.submit(r, BACK if back else LAY, tick, stake)
            if o is None:
                continue
            self.n_opens += 1
            mkt_loss += risk
            wait = ENTRY_WAIT_S if k > 0 else cfg.decision_every * t.dt
            self.br[r] = dict(side=BACK if back else LAY, k=k, stake=stake, placed=step, phase="entering",
                              entry_until=step + int(round(wait / t.dt)), entry_tick=None, opened=None, hold_until=None,
                              hold_s=HOLDS[min(int(h_[r] * len(HOLDS)), len(HOLDS) - 1)],
                              tp=TPS[min(int(tp_[r] * len(TPS)), len(TPS) - 1)],
                              sl=SLS[min(int(sl_[r] * len(SLS)), len(SLS) - 1)], v0=self._value(r),
                              t_open=float(t.t_rel[step]), why=None)

    def decide(self, action):
        if self.tape.suspended[self.ex.step]:
            return
        if float(self.tape.t_rel[self.ex.step]) >= 0.0:  # the off: everything still live is closed at BSP
            for r in list(self.br):
                if self.br[r]["phase"] == "entering":
                    self.ex.cancel_runner(r)
                self._bsp_out(r)
            self._manage_after_bsp()
            for r in range(self.tape.n_runners):  # a late fill of a cancelled order: close that too
                if not self._flat(r):
                    self._bsp_out(r)
            return
        self._manage()
        self._apply(action)

    def _manage_after_bsp(self):
        for r in list(self.br):
            b = self.br[r]
            if b["entry_tick"] is None and sum(f.stake for f in self.ex.fills if f.runner == r and f.step >= b["placed"]) < 0.01:
                self.br.pop(r)
            elif self._flat(r):
                self._finish(r, "bsp")

    def step(self, action):
        ex, cfg = self.ex, self.cfg
        self._rr = np.zeros(R_MAX, np.float32)
        self.decide(action)
        for _ in range(cfg.decision_every):
            ex.advance()
            if ex.step >= self.tape.n_steps - 1:
                break
        done = ex.step >= self.tape.n_steps - 1
        info = {}
        if done:
            for r in list(self.br):  # the tape ended with a trade still live: close it at BSP
                self._bsp_out(r)
            self._manage_after_bsp()
            for r in list(self.br):
                self._finish(r, "end")
            res = ex.settle()
            info.update(res)
            info.update(market=self.tape.name, turnover=ex.turnover, n_fills=len(ex.fills), n_rejected=ex.n_rejected,
                        n_opens=self.n_opens, n_closes=len(self.trades), n_stops=sum(x["reason"] == "stop" for x in self.trades))
        info["runner_rewards"] = self._rr.copy()
        return self._obs(), float(self._rr.sum()), bool(done), False, info

    # ------------------------------------------------------------------ demo (warm start)
    def demo_action(self) -> np.ndarray:
        """P16 / arm C's drift-fade as a bracket: back a 2nd/3rd favourite whose own 30s move is at or below
        demo_cut, $10 at risk, resting 3 ticks above the best back price, held to the off (BSP)."""
        a = np.zeros((N_ACT6, R_MAX), np.float32)
        t, ex = self.tape, self.ex
        if self._p is None:
            self._p = _prob_grid(t)[0]
        s, k30 = ex.step, int(round(30.0 / t.dt))
        if s < k30 or float(t.t_rel[s]) < -600 or float(t.t_rel[s]) > -5:
            return a.reshape(-1)
        rb, rl = t.back_tick[s, :, 0], t.lay_tick[s, :, 0]
        ok = t.active[s] & (rb >= 0) & (rl >= 0)
        if ok.sum() < 3:
            return a.reshape(-1)
        mid = np.where(ok, np.sqrt(PRICES[np.maximum(rb, 0)] * PRICES[np.maximum(rl, 0)]), np.inf)
        rank = mid.argsort().argsort() + 1
        mom = (self._p[s] - self._p[s - k30]) * 100
        for r in range(min(t.n_runners, R_MAX)):
            if ok[r] and rank[r] in (2, 3) and np.isfinite(mom[r]) and mom[r] <= self.cfg.demo_cut and r not in self.br:
                a[:, r] = (1.0, (10.0 - 5.0) / 15.0, 0.9, 0.99, 0.0, 0.0)  # back, $10, k=3, hold to the off, no tp/sl
        return a.reshape(-1)


def noop_policy(obs, env):
    return np.zeros(N_ACT6 * R_MAX, np.float32)


def random_policy(seed: int = 0):
    rng = np.random.default_rng(seed)

    def pol(obs, env):
        a = rng.uniform(0, 1, (N_ACT6, R_MAX)).astype(np.float32)
        a[0] = rng.uniform(-1, 1, R_MAX) * obs.get("act_mask", obs["mask"]) * (rng.random(R_MAX) < 0.1)
        a[1] = 0.2
        return a.reshape(-1)

    return pol


def demo_policy(obs, env):
    return env.demo_action()


def trade_log(pol, tapes, cfg) -> pd.DataFrame:
    env = BracketAgentEnv(tapes, cfg, cache_tapes=False)
    rows = []
    for tp in tapes:
        run_episode(env, tp, pol)
        for x in env.trades:
            rows.append(dict(race=env.tape.name, **x))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-matched", type=float, default=0.0, help="only races with >= this $ matched (e.g. 20000)")
    ap.add_argument("--min-best-usd", default="auto", help='liquidity gate; "auto" = P11 top quintile on TRAIN days')
    ap.add_argument("--order-cap", type=float, default=20.0)
    ap.add_argument("--runner-cap", type=float, default=50.0)
    ap.add_argument("--market-cap", type=float, default=150.0)
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--demo-frac", type=float, default=0.3, help="share of warm-up episodes played by the drift-fade")
    ap.add_argument("--n-step", type=int, default=12, help="n-step returns (12 decisions = 2 min)")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--control-steps", type=int, default=300_000)
    ap.add_argument("--control-races", type=int, default=200)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--control-only", action="store_true")
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--eval-races", type=int, default=100)
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
    tapes = list_tapes(a.tapes)
    if a.min_matched > 0:
        n_all = len(tapes)
        tapes = select_markets(tapes, a.min_matched, a.workers)
        print(f"markets: {len(tapes)} of {n_all} races have >= ${a.min_matched:,.0f} matched at the off", flush=True)
    tr, va, te = split_by_date(tapes)
    cut = liquidity_cut(tr) if a.min_best_usd == "auto" else float(a.min_best_usd)
    cfg = BracketConfig(random_start_s=0.0, decision_every=20, min_best_usd=cut, order_cap=a.order_cap,
                        runner_cap=a.runner_cap, market_cap=a.market_cap)
    pess = replace(cfg, exchange=replace(cfg.exchange, queue_mult=2.0, trade_fill_frac=0.5))
    extra = (("agent, pessimistic fills", pess),)
    print(f"=== P17b bracket agents ({a.device}): a trade = entry + exit plan, reward at the close, gamma {a.gamma}, "
          f"{a.n_step}-step returns, {a.demo_frac:.0%} of warm-up episodes played by the drift-fade; gate ${cut:.2f}; "
          f"{len(tr)} train / {len(va)} val / {len(te)} test races ===", flush=True)
    if cut > 0:
        g = gate_summary(te, cut)
        print(f"TEST races: {g['runner_share_open'] * 100:.0f}% of runner snapshots pass the gate\n", flush=True)
    kw = dict(n_envs=a.n_envs, device=a.device, buffer=a.buffer, env_cls=BracketAgentEnv, n_act=N_ACT6,
              gamma=a.gamma, n_step=a.n_step, demo_frac=a.demo_frac)
    rows = []

    control_ok = None
    if not a.skip_control:
        print("=== 0. Positive control: 'bounce' races, where only resting orders can profit ===", flush=True)
        syn = os.path.join(a.out, "synthetic_bounce")
        write_synthetic(syn, a.control_races, seed0=70_000, kind="bounce")
        g_ = os.path.join(syn, "*.npz")
        ctr, cva, cte = split_by_date(list_tapes(g_))
        ck = train(g_, os.path.join(a.out, "control"), cfg, a.control_steps, int(a.control_steps * a.warmup_frac), 0,
                   eval_races=min(a.eval_races, 30), split=(ctr, cva, cte), **dict(kw, demo_frac=0.0))
        rows.append(dict(run="control", **score("do nothing", evaluate(noop_policy, cte, cfg, env_cls=BracketAgentEnv))))
        rows.append(dict(run="control", **score("random", evaluate(random_policy(), cte, cfg, env_cls=BracketAgentEnv))))
        _, ctl, _ = test_block("control", ck, cte, cfg, a.device, rows, stress=False, env_cls=BracketAgentEnv)
        control_ok = bool(ctl["mean_green"] > 0 and (ctl["t"] or 0) > 2)
        print(pd.DataFrame([r for r in rows if r["run"] == "control"]).round(3).to_string(index=False))
        print("control: the agents FIND the resting-only edge" if control_ok else
              "control: NOT found -> a null result on real races means little", flush=True)
    if a.control_only:
        json.dump(dict(control_found_planted_edge=control_ok), open(os.path.join(a.out, "verdict.json"), "w"), indent=1)
        print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")
        return

    print(f"\n=== 1. Real races: {len(te)} TEST races ===", flush=True)
    for name, p in (("do nothing", noop_policy), ("random", random_policy()), ("drift-fade (the demo)", demo_policy)):
        rows.append(dict(run="real", **score(name, evaluate(p, te, cfg, env_cls=BracketAgentEnv))))
    results, pols = {}, {}
    for sd in [int(x) for x in a.seeds.split(",") if x.strip()]:
        print(f"\n--- seed {sd}: {a.steps} steps ---", flush=True)
        ck = train(a.tapes, os.path.join(a.out, f"real_seed{sd}"), cfg, a.steps, int(a.steps * a.warmup_frac), sd,
                   eval_races=a.eval_races, split=(tr, va, te), **kw)
        pols[sd], main_row, stress_rows = test_block(f"seed {sd}", ck, te, cfg, a.device, rows, env_cls=BracketAgentEnv,
                                                     extra_stress=extra)
        results[sd] = (main_row, *stress_rows)

    res = pd.DataFrame(rows)
    res["per_$_traded_%"] = res["mean_green"] / res["turnover"].where(res["turnover"] > 0) * 100
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    print("\n=== 2. TEST races (never trained on); $ per race, $500 bank ===")
    print(res.round(3).to_string(index=False))
    for sd, pol in pols.items():
        tl = trade_log(pol, te, cfg)
        tl.to_csv(os.path.join(a.out, f"trades_seed{sd}.csv"), index=False)
        print(f"\n=== Trades seed {sd} took on TEST races: {len(tl)} ===")
        if len(tl):
            print(tl.groupby(["side", "reason"])["green"].agg(["count", "mean", "sum"]).round(3).to_string())
            print(tl.groupby("hold", dropna=False)["green"].agg(["count", "mean"]).round(3).to_string())
    edge = bool(results) and all(r[0]["mean_green"] > 0 and (r[0]["t"] or 0) > 2 and all(x["mean_green"] > 0 for x in r[1:])
                                 for r in results.values())
    per_traded = {str(s_): (r[0]["mean_green"] / r[0]["turnover"] * 100 if r[0]["turnover"] > 0 else np.nan)
                  for s_, r in results.items()}
    suspicious = any(np.isfinite(v) and v > 2 * CEILING for v in per_traded.values())
    verdict = dict(control_found_planted_edge=control_ok, edge=edge, per_dollar_traded_pct=per_traded,
                   above_measured_ceiling=suspicious,
                   seeds={str(s_): dict(test_green=r[0]["mean_green"], t=r[0]["t"], turnover=r[0]["turnover"],
                                        latency_green=r[1]["mean_green"], pessimistic_green=r[2]["mean_green"])
                          for s_, r in results.items()})
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    if edge and suspicious:
        print("EDGE ON PAPER, BUT ABOVE WHAT P15 / P16 MEASURED: treat as a likely simulator artefact first")
    elif edge:
        print("EDGE: every seed beats do-nothing on TEST races and survives both stresses -> pre-register and forward-test it")
    elif control_ok is False:
        print("INCONCLUSIVE: the agents could not find the resting-only planted edge either")
    else:
        print("NO EDGE: with whole-trade actions, rewards at the close and a drift-fade warm start, nothing on unseen "
              "races beats doing nothing")
    print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
