"""P17: per-runner agents that can REST orders, not just cross the spread, and exit at BSP.

Every black box so far (P8, P13, P13c) could only cross the spread, ~1-4% per round trip.
P15 showed the cheapest execution is resting 1-3 ticks better than the best price and
exiting at BSP (~0.2% per attempt on liquid runners), and P16 that book signals alone do
not pick good fills. P17 gives the P13c per-runner agents that full execution menu and
lets SAC look for state-dependent tactics (when to rest, where, how big, when to cancel).

Per online runner (same liquidity gate as P13c: $ at best back + lay, 10s mean, >= cut;
below it a held runner is reduce-only), every decision (10s), 3 actions:
    c in [-1, 1]  side (> 0 back, < 0 lay); |c| < deadzone = cancel this runner's orders
    s in [0, 1]   size: $ at risk = min_stake + s * (order_cap - min_stake)
    o in [0, 1]   price: k = floor(o * 4) -> 0 = cross now at the best price, 1..3 = rest
                  k ticks better (back above the best back price, lay below the best lay)
An order that is asked for again (same side and price) is left alone, keeping its queue
position; anything else replaces it (the cancel lands one step later, as live). Order size
is capped at --size-frac-cap of the money shown at the best level on its side, so the
agent stays where the simulator is credible. Per-runner and per-market loss caps count
resting orders too. From the scheduled start, any open position is closed AT BSP (the
realised SP, P3b's exit) and resting orders are cancelled. Reward: the runner's own P&L.

Credibility checks, because resting orders are where the simulator is weakest:
  0  positive control: synthetic "bounce" races (synthetic.make_bounce_tape) where only
     resting orders can profit; the agent must find it, or a null result means nothing
  *  every TEST score is repeated with 1s latency and with PESSIMISTIC fills (queue ahead
     x2, only half of the traded volume counts); an edge must survive both
  *  ceiling: P15 / P16 measured what resting entry is worth on liquid runners (about
     -0.2% .. +0.8% per attempt). A return per $ traded far above that is more likely a
     simulator artefact than a discovery; it is flagged.

--min-matched (e.g. 20000) restricts training, checkpoint selection and the test to races with at least that
much matched at the off (P18's market-size scan); the chronological split is then over those races.

    python -m ahr_rl.passive_agents --tapes "data/tapes/*.npz" --out runs/passive_agents --min-best-usd auto
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

from .blackbox import control_gate, forensics, gate_summary, liquidity_cut, score
from .env import list_tapes, split_by_date
from .env_continuous import ContinuousAllocEnv
from .evaluate import evaluate
from .exchange import BACK, LAY
from .features import R_MAX
from .ladder import N_TICKS, PRICES
from .market_select import select_markets
from .runner_agents import MIN_SIZE, RunnerAgentConfig, test_block, train
from .synthetic import write_synthetic

N_ACT3 = 3  # conviction, size, price offset
CEILING = 0.8  # % per $ traded: P15 / P16's best measured resting entry on liquid runners


@dataclass
class RestingAgentConfig(RunnerAgentConfig):
    max_offset: int = 3  # k = 1..max_offset ticks better than the best price
    size_frac_cap: float = 0.2  # order stake <= this x money shown at the best level on its side
    exit_bsp: bool = True  # close open positions at BSP from the scheduled start


class RestingAgentEnv(ContinuousAllocEnv):
    """Action = 72 floats: c_0..c_23 in [-1, 1], s_0..s_23 in [0, 1], o_0..o_23 in [0, 1]."""

    def __init__(self, tapes, cfg: RestingAgentConfig | None = None, **kw):
        super().__init__(tapes, cfg or RestingAgentConfig(), **kw)
        lo = np.r_[np.full(R_MAX, -1.0), np.zeros(2 * R_MAX)].astype(np.float32)
        self.action_space = gym.spaces.Box(lo, np.ones(N_ACT3 * R_MAX, np.float32), dtype=np.float32)

    def _runner_loss(self, r: int) -> float:
        """Worst-case loss on runner r from matched bets plus its resting orders."""
        ex = self.ex
        W, L = ex.W[r], ex.L[r]
        for o in ex.orders:
            if o.runner == r and o.size > 1e-9:
                if o.side == BACK:
                    L -= o.size
                else:
                    W -= o.size * (PRICES[o.tick] - 1.0)
        return max(0.0, -min(W, L))

    def _apply(self, action):
        ex, cfg, t = self.ex, self.cfg, self.tape
        a = np.asarray(action, np.float64).reshape(-1)
        c = np.clip(a[:R_MAX], -1, 1)
        s = np.clip(a[R_MAX:2 * R_MAX], 0, 1)
        o_ = np.clip(a[2 * R_MAX:3 * R_MAX], 0, 1)
        liq = self.liquid()
        expo = ex.W - ex.L
        mkt_loss = max(0.0, -ex.worst_case())  # includes resting orders
        step = ex.step
        for r in range(min(t.n_runners, R_MAX)):
            mine = [o for o in ex.orders if o.runner == r and o.size > 1e-9]
            if not t.active[step, r] or ex.void_runner[r]:
                continue
            if abs(c[r]) < cfg.deadzone or s[r] < MIN_SIZE:
                if mine:
                    ex.cancel_runner(r)
                continue
            back = c[r] > 0
            k = min(int(o_[r] * (cfg.max_offset + 1)), cfg.max_offset)
            bb, bl = ex.best(step, r)
            if bb < 0 or bl < 0:
                continue
            tick = (bb + k) if back else (bl - k)  # k = 0: cross at the best price
            if not 0 <= tick < N_TICKS:
                continue
            side = BACK if back else LAY
            reduces = (back and expo[r] < -1e-6) or (not back and expo[r] > 1e-6)
            if not liq[r] and not reduces:
                if mine:
                    ex.cancel_runner(r)
                continue
            if k > 0 and len(mine) == 1 and mine[0].side == side and mine[0].tick == tick:
                continue  # same order as before: keep its place in the queue
            risk = ex.cfg.min_stake + s[r] * max(cfg.order_cap - ex.cfg.min_stake, 0.0)
            if not reduces:
                risk = min(risk, cfg.runner_cap - self._runner_loss(r), cfg.market_cap - mkt_loss)
                if risk <= 0:
                    if mine:
                        ex.cancel_runner(r)
                    continue
            p = PRICES[tick]
            stake = risk if back else risk / max(p - 1.0, 0.01)
            if reduces:
                hp = ex.hedge_plan(r)
                if hp is None:
                    continue
                stake = min(stake, hp[2])
            # stay where the simulator is credible: no more than a fraction of the money shown on our side
            shown = float((t.back_size if (back == (k == 0)) else t.lay_size)[step, r, 0])
            stake = min(stake, max(ex.cfg.min_stake, cfg.size_frac_cap * shown))
            if mine:
                ex.cancel_runner(r)
            o = ex.submit(r, side, tick, stake, is_hedge=reduces)
            if o is not None:
                if reduces:
                    self.n_closes += 1
                else:
                    self.n_opens += 1
                    mkt_loss += risk

    def _close(self, r):
        """From the scheduled start: cancel the runner's orders and green any position at BSP."""
        ex, t = self.ex, self.tape
        if not (self.cfg.exit_bsp and t.bsp[r] > 1.0):
            return super()._close(r)
        if any(o.runner == r and o.size > 1e-9 for o in ex.orders):
            ex.cancel_runner(r)
        d = ex.W[r] - ex.L[r]
        if abs(d) < 1e-6 or ex.void_runner[r]:
            return
        bsp = float(t.bsp[r])
        if d > 0:  # long the runner: lay at BSP
            ex._add_bet(r, LAY, bsp, d / bsp, passive=False)
        else:  # short: back at BSP
            ex._add_bet(r, BACK, bsp, -d / bsp, passive=False)


def noop_policy(obs, env):
    return np.zeros(N_ACT3 * R_MAX, np.float32)


def random_policy(size: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)

    def pol(obs, env):
        c = rng.uniform(-1, 1, R_MAX) * obs.get("act_mask", obs["mask"])
        return np.r_[c, np.full(R_MAX, size), rng.uniform(0, 1, R_MAX)].astype(np.float32)

    return pol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-best-usd", default="auto",
                    help='liquidity gate in $ at best back + lay; "auto" = P11 top quintile on TRAIN days; 0 = off')
    ap.add_argument("--no-gate-inputs", action="store_true")
    ap.add_argument("--order-cap", type=float, default=20.0)
    ap.add_argument("--runner-cap", type=float, default=50.0)
    ap.add_argument("--market-cap", type=float, default=150.0)
    ap.add_argument("--size-frac-cap", type=float, default=0.2, help="order <= this x money shown at the best level")
    ap.add_argument("--min-matched", type=float, default=0.0,
                    help="train / select / test only on races with >= this $ matched at the off (e.g. 20000); 0 = all")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="for scanning market sizes")
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--decision-every", type=int, default=20, help="tape steps (0.5s) per decision; 20 = 10s")
    ap.add_argument("--update-every", type=int, default=2)
    ap.add_argument("--grad-steps", type=int, default=1)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--control-steps", type=int, default=300_000)
    ap.add_argument("--control-races", type=int, default=200)
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
    if a.device == "cpu":
        print("WARNING: no GPU. Training will be very slow (P13 on CPU: ~12h per seed). Use a GPU runtime.", flush=True)
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - a.n_envs))  # leave cores for the env processes
    tapes = list_tapes(a.tapes)
    if a.min_matched > 0:
        n_all = len(tapes)
        tapes = select_markets(tapes, a.min_matched, a.workers)
        print(f"markets: {len(tapes)} of {n_all} races have >= ${a.min_matched:,.0f} matched at the off "
              f"({len({os.path.basename(p)[:8] for p in tapes})} race days)", flush=True)
    tr, va, te = split_by_date(tapes)
    cut = liquidity_cut(tr) if a.min_best_usd == "auto" else float(a.min_best_usd)
    cfg = RestingAgentConfig(random_start_s=0.0, decision_every=a.decision_every, min_best_usd=cut,
                             gate_inputs=not a.no_gate_inputs, order_cap=a.order_cap, runner_cap=a.runner_cap,
                             market_cap=a.market_cap, size_frac_cap=a.size_frac_cap)
    pess = replace(cfg, exchange=replace(cfg.exchange, queue_mult=2.0, trade_fill_frac=0.5))
    extra = (("agent, pessimistic fills", pess),)
    print(f"=== P17 resting per-runner agents ({a.device}): gate ${cut:.2f}; cross or rest 1-{cfg.max_offset} ticks "
          f"better, $5-${a.order_cap:.0f} at risk (<= {a.size_frac_cap:.0%} of the money shown), runner cap "
          f"${a.runner_cap:.0f}, market cap ${a.market_cap:.0f}; open positions closed at BSP ===")
    gates = {}
    if cut > 0:
        g = gates["real_test"] = gate_summary(te, cut)
        print(f"TEST races, last 5 min: {g['runner_share_open'] * 100:.0f}% of runner snapshots online; "
              f"{g['races_with_any_open']} of {g['races']} races have an online runner\n", flush=True)
    kw = dict(n_envs=a.n_envs, device=a.device, buffer=a.buffer, update_every=a.update_every,
              grad_steps=a.grad_steps, env_cls=RestingAgentEnv, n_act=N_ACT3)
    rows = []

    control_ok = None
    if not a.skip_control:
        print("=== 0. Positive control: 'bounce' races, where only resting orders can profit ===", flush=True)
        syn = os.path.join(a.out, "synthetic_bounce")
        write_synthetic(syn, a.control_races, seed0=70_000, kind="bounce")
        glob_ = os.path.join(syn, "*.npz")
        _, _, cte = split_by_date(list_tapes(glob_))
        ccfg = replace(cfg, min_best_usd=control_gate(cut, list_tapes(glob_)))  # gate scaled to the control races
        if ccfg.min_best_usd != cut:
            print(f"control gate ${ccfg.min_best_usd:.2f} (the real gate ${cut:.2f} is above what the synthetic "
                  f"races ever show)", flush=True)
        ck = train(glob_, os.path.join(a.out, "control"), ccfg, a.control_steps, int(a.control_steps * a.warmup_frac),
                   0, eval_races=min(a.eval_races, 30), **kw)
        rows.append(dict(run="control", **score("do nothing", evaluate(noop_policy, cte, ccfg, env_cls=RestingAgentEnv))))
        rows.append(dict(run="control", **score("random", evaluate(random_policy(), cte, ccfg, env_cls=RestingAgentEnv))))
        _, ctl, _ = test_block("control", ck, cte, ccfg, a.device, rows, stress=False, env_cls=RestingAgentEnv)
        control_ok = bool(ctl["mean_green"] > 0 and (ctl["t"] or 0) > 2)
        print(pd.DataFrame([r for r in rows if r["run"] == "control"]).round(3).to_string(index=False))
        print("control: the agents FIND the resting-only edge" if control_ok else
              "control: the resting-only edge was NOT found -> a null result on real races means little; "
              "try more --control-steps", flush=True)

    if a.control_only:
        json.dump(dict(control_found_planted_edge=control_ok, gate=gates), open(os.path.join(a.out, "verdict.json"),
                                                                                  "w"), indent=1, default=float)
        print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")
        return

    print(f"\n=== 1. Real races: {len(tapes)} tapes, {len(te)} TEST races ===", flush=True)
    rows.append(dict(run="real", **score("do nothing", evaluate(noop_policy, te, cfg, env_cls=RestingAgentEnv))))
    rows.append(dict(run="real", **score("random", evaluate(random_policy(), te, cfg, env_cls=RestingAgentEnv))))
    pols, results = {}, {}
    for sd in [int(x) for x in a.seeds.split(",") if x.strip()]:
        print(f"\n--- seed {sd}: {a.steps} steps, first {int(a.steps * a.warmup_frac)} on random strategies ---",
              flush=True)
        ck = train(a.tapes, os.path.join(a.out, f"real_seed{sd}"), cfg, a.steps, int(a.steps * a.warmup_frac), sd,
                   eval_races=a.eval_races, split=(tr, va, te), **kw)
        pols[sd], main_row, stress_rows = test_block(f"seed {sd}", ck, te, cfg, a.device, rows,
                                                     env_cls=RestingAgentEnv, extra_stress=extra)
        results[sd] = (main_row, *stress_rows)

    res = pd.DataFrame(rows)
    res["per_$_traded_%"] = res["mean_green"] / res["turnover"].where(res["turnover"] > 0) * 100
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    print("\n=== 2. TEST races (never used for training or checkpoint selection); $ per race, $500 bank ===")
    print(res.round(3).to_string(index=False))

    best_sd = max(results, key=lambda x: results[x][0]["mean_green"]) if results else None
    if best_sd is not None:
        fz = forensics(pols[best_sd], te[:100], cfg, env_cls=RestingAgentEnv)
        fz.to_csv(os.path.join(a.out, f"fills_seed{best_sd}.csv"), index=False)
        print(f"\n=== What seed {best_sd} trades on TEST races ({len(fz)} fills) ===")
        if len(fz):
            print(fz.pivot_table(index="rank", columns=["side"], values="stake", aggfunc="sum", fill_value=0)
                  .round(0).to_string())
            print(fz.pivot_table(index="time", columns=["side"], values="stake", aggfunc="sum", fill_value=0,
                                 observed=False).round(0).to_string())
            print(f"resting (passive) fills: {fz.loc[fz['passive'], 'stake'].sum() / max(fz['stake'].sum(), 1e-9):.0%} "
                  f"of $ matched (the rest crossed the spread or closed at BSP)")
        else:
            print("no fills: the agents learned not to trade")

    edge = bool(results) and all(r[0]["mean_green"] > 0 and (r[0]["t"] or 0) > 2 and all(x["mean_green"] > 0 for x in r[1:])
                                 for r in results.values())
    per_traded = {str(s_): (r[0]["mean_green"] / r[0]["turnover"] * 100 if r[0]["turnover"] > 0 else np.nan)
                  for s_, r in results.items()}
    suspicious = any(np.isfinite(v) and v > 2 * CEILING for v in per_traded.values())
    verdict = dict(control_found_planted_edge=control_ok, edge=edge, gate=gates, per_dollar_traded_pct=per_traded,
                   above_measured_ceiling=suspicious,
                   seeds={str(s_): dict(test_green=r[0]["mean_green"], t=r[0]["t"], turnover=r[0]["turnover"],
                                        latency_green=r[1]["mean_green"], pessimistic_green=r[2]["mean_green"])
                          for s_, r in results.items()})
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    if edge and suspicious:
        print(f"EDGE ON PAPER, BUT ABOVE WHAT P15 / P16 MEASURED (> {2 * CEILING:.1f}% per $ traded): treat as a likely "
              "simulator artefact; study the forensics before anything else")
    elif edge:
        print("EDGE: every seed beats do-nothing on TEST races and survives the latency and pessimistic-fill stresses "
              "-> study the forensics, then pre-register and forward-test it like P3c")
    elif control_ok is False:
        print("INCONCLUSIVE: the agents could not find the resting-only planted edge either")
    else:
        print("NO EDGE: with resting orders and BSP exits too, the agents find nothing on unseen races that beats "
              "doing nothing")
    print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
