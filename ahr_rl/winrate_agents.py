"""P19: a black box whose objective is to show a PROFIT MORE OFTEN THAN NOT once the race has settled.

Every earlier objective was a green book before the off. Here positions are allowed to run: the agent
(P17's per-runner SAC: every runner, every 10s, side / size / price - cross, or rest 1-3 ticks better -
P13's liquidity gate) trades until the scheduled start, its unmatched orders are cancelled at the off, and
the book is settled on the result. The objective it is trained on:

    reward (at the end of the race) = P(profit) - P(loss)

over the possible winners, with each runner's chance taken from BSP (normalised 1/BSP; BSP is a fair
price, P10). It is positive exactly when the book shows a profit more often than not. It is used instead
of the single realised result because one race's winner is extremely noisy to learn from; the TEST scores
report both. Doing nothing scores 0. --reward mixed adds --ev-weight x the book's BSP-implied EV
(in units of $5), which pushes against the trap below.

THE WIN-RATE TRAP. "More often than not" is easy to buy: lay a few longshots and the book profits in most
races, then occasionally loses big. That is a high win rate with a negative expectation, worse than doing
nothing. So every score also reports the BSP-implied EV, the realised P&L, and the average win vs the
average loss, a hand-made "lay every runner priced 3.0-5.0" policy is scored as a reference (true longshots
are mostly out of reach: a $5 minimum stake at 10.0 is $45 of liability, above the order cap), and the
verdict separates the two:
    objective met   on TEST races the agent bets on, realised wins > losses (one-sided z > 2)
    useful          objective met AND the EV per race at BSP is > 0 with t > 2 AND realised P&L > 0
Positive control first: P17's synthetic 'bounce' races, where resting orders can lock a profit in every
outcome (the agent must reach P(profit) - P(loss) > 0, t > 2 there).

    python -m ahr_rl.winrate_agents --tapes "data/tapes/*.npz" --out runs/winrate --min-matched 20000
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from .blackbox import control_gate, gate_summary, liquidity_cut, race_t
from .env import list_tapes, split_by_date
from .evaluate import run_episode
from .features import R_MAX
from .market_select import select_markets
from .passive_agents import N_ACT3, RestingAgentConfig, RestingAgentEnv, noop_policy, random_policy
from .runner_agents import load_runner_actor, make_runner_policy, train
from .synthetic import write_synthetic


@dataclass
class HoldConfig(RestingAgentConfig):
    reward_mode: str = "winrate"  # "winrate": P(profit) - P(loss); "mixed": + ev_weight * EV / $5
    ev_weight: float = 0.5


class HoldEnv(RestingAgentEnv):
    """RestingAgentEnv without any green-up: trading stops (orders cancelled) at the scheduled start and the
    book is held to the result. Reward only at the end: the BSP-implied P(profit) - P(loss)."""

    def __init__(self, tapes, cfg: HoldConfig | None = None, **kw):
        cfg = cfg or HoldConfig()
        super().__init__(tapes, replace(cfg, auto_green_s=None, exit_bsp=False), **kw)

    def decide(self, action):
        if float(self.tape.t_rel[self.ex.step]) >= 0.0:  # the off: no new bets, nothing left resting
            if self.ex.orders:
                self.ex.cancel_all()
            return
        super().decide(action)

    def outcome(self) -> dict:
        """BSP-implied P(profit), P(loss) and EV of the book over the possible winners (net of commission)."""
        ex, t = self.ex, self.tape
        if not t.went_in_play or not ex.bet_runner:
            return dict(p_profit=0.0, p_loss=0.0, ev_bsp=0.0)
        m = ex.outcome_mask() & (t.bsp > 1.0)
        if not m.any():
            return dict(p_profit=0.0, p_loss=0.0, ev_bsp=0.0)
        pi = np.where(m, 1.0 / np.maximum(t.bsp, 1.0001), 0.0)
        pi /= pi.sum()
        net = ex.net_of_commission(ex.pnl_by_outcome())
        return dict(p_profit=float(pi[net > 0.005].sum()), p_loss=float(pi[net < -0.005].sum()),
                    ev_bsp=float((pi * net).sum()))

    def step(self, action):
        obs, _, done, trunc, info = super().step(action)
        rr = np.zeros(R_MAX, np.float32)
        reward = 0.0
        if done:
            o = self.outcome()
            info.update(o)
            reward = o["p_profit"] - o["p_loss"]
            if self.cfg.reward_mode == "mixed":
                reward += self.cfg.ev_weight * o["ev_bsp"] / 5.0
            for r in set(self.ex.bet_runner):  # every runner that was traded shares the race's result
                if r < R_MAX:
                    rr[r] = reward
        info["runner_rewards"] = rr
        return obs, float(reward), done, trunc, info


def lay_band_policy(min_price: float = 3.0, max_price: float = 5.0, size: float = 1.0):
    """Reference for the trap: lay (crossing) every runner whose best lay price is in [min_price, max_price].
    (Laying real longshots is mostly out of reach at these sizes: Betfair's $5 minimum stake means a lay at
    10.0 needs $45 of liability, above the order cap. Laying the mid-priced runners is the reachable version.)"""
    from .ladder import PRICES

    def pol(obs, env):
        a = np.zeros(N_ACT3 * R_MAX, np.float32)
        ex = env.ex
        for r in range(min(env.tape.n_runners, R_MAX)):
            if not obs["act_mask"][r]:
                continue
            bb, bl = ex.best(ex.step, r)
            if bl >= 0 and min_price <= PRICES[bl] <= max_price:
                a[r], a[R_MAX + r], a[2 * R_MAX + r] = -1.0, size, 0.0  # lay, crossing
        return a

    return pol


def score_races(pol, tapes, cfg) -> pd.DataFrame:
    env = HoldEnv(tapes, cfg, cache_tapes=False)
    rows = []
    for tp in tapes:
        info = run_episode(env, tp, pol)
        rows.append(dict(race=info.get("market"), traded=info.get("turnover", 0) > 0, turnover=info.get("turnover", 0),
                         p_profit=info.get("p_profit", 0.0), p_loss=info.get("p_loss", 0.0),
                         ev_bsp=info.get("ev_bsp", 0.0), realised=info.get("realised", np.nan),
                         worst=info.get("worst", np.nan), void=info.get("void", False)))
    return pd.DataFrame(rows)


def val_objective(pol, tapes, cfg) -> float:
    """Checkpoint selection on VAL races: the training objective itself (not the green value)."""
    d = score_races(pol, tapes, cfg)
    v = d["p_profit"] - d["p_loss"]
    if cfg.reward_mode == "mixed":
        v = v + cfg.ev_weight * d["ev_bsp"] / 5.0
    return float(v.mean())


def summarise(name: str, d: pd.DataFrame) -> dict:
    d = d[~d["void"]]
    tr = d[d["traded"]]
    w, l = tr["realised"] > 0.005, tr["realised"] < -0.005
    n_wl = int(w.sum() + l.sum())
    z = (w.sum() - n_wl / 2) / np.sqrt(n_wl / 4) if n_wl else np.nan  # one-sided test of wins > losses
    return dict(policy=name, races=len(d), pct_traded=len(tr) / max(len(d), 1) * 100,
                implied_p_profit=tr["p_profit"].mean() if len(tr) else np.nan,
                implied_p_loss=tr["p_loss"].mean() if len(tr) else np.nan,
                realised_win_pct=w.mean() * 100 if len(tr) else np.nan,
                realised_loss_pct=l.mean() * 100 if len(tr) else np.nan, wins_vs_losses_z=z,
                avg_win=tr.loc[w, "realised"].mean() if w.any() else np.nan,
                avg_loss=tr.loc[l, "realised"].mean() if l.any() else np.nan,
                ev_bsp_per_race=d["ev_bsp"].mean(), ev_t=race_t(d["ev_bsp"]),
                realised_per_race=d["realised"].mean(), realised_t=race_t(d["realised"].fillna(0)),
                worst_race=d["realised"].min(), turnover=d["turnover"].mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reward", default="winrate", choices=["winrate", "mixed"])
    ap.add_argument("--ev-weight", type=float, default=0.5, help="--reward mixed: weight on EV / $5")
    ap.add_argument("--min-matched", type=float, default=0.0, help="only races with >= this $ matched (e.g. 20000)")
    ap.add_argument("--min-best-usd", default="auto", help='liquidity gate; "auto" = P11 top quintile; 0 = off')
    ap.add_argument("--order-cap", type=float, default=20.0)
    ap.add_argument("--runner-cap", type=float, default=50.0)
    ap.add_argument("--market-cap", type=float, default=150.0)
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--warmup-frac", type=float, default=0.4)
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
    cfg = HoldConfig(random_start_s=0.0, decision_every=20, min_best_usd=cut, order_cap=a.order_cap,
                     runner_cap=a.runner_cap, market_cap=a.market_cap, reward_mode=a.reward, ev_weight=a.ev_weight)
    print(f"=== P19: profit more often than not after the race (reward: {a.reward}"
          f"{f', EV weight {a.ev_weight}' if a.reward == 'mixed' else ''}); positions held to the result, no green-up; "
          f"gate ${cut:.2f}; {len(tr)} train / {len(va)} val / {len(te)} test races ({a.device}) ===", flush=True)
    if cut > 0:
        g = gate_summary(te, cut)
        print(f"TEST races: {g['runner_share_open'] * 100:.0f}% of runner snapshots pass the gate\n", flush=True)
    kw = dict(n_envs=a.n_envs, device=a.device, buffer=a.buffer, env_cls=HoldEnv, n_act=N_ACT3,
              val_score=val_objective)
    rows = []

    control_ok = None
    if not a.skip_control:
        print("=== 0. Positive control: 'bounce' races, where resting orders can lock a profit in every outcome ===",
              flush=True)
        syn = os.path.join(a.out, "synthetic_bounce")
        write_synthetic(syn, a.control_races, seed0=70_000, kind="bounce")
        g_ = os.path.join(syn, "*.npz")
        ctr, cva, cte = split_by_date(list_tapes(g_))
        ccfg = replace(cfg, min_best_usd=control_gate(cut, list_tapes(g_)))  # gate scaled to the control races
        if ccfg.min_best_usd != cut:
            print(f"control gate ${ccfg.min_best_usd:.2f} (the real gate ${cut:.2f} is above what the synthetic "
                  f"races ever show)", flush=True)
        ck = train(g_, os.path.join(a.out, "control"), ccfg, a.control_steps, int(a.control_steps * a.warmup_frac), 0,
                   eval_races=min(a.eval_races, 30), split=(ctr, cva, cte), **kw)
        pol = make_runner_policy(load_runner_actor(ck, a.device), a.device)
        for name, p in (("do nothing", noop_policy), ("random", random_policy()), ("agent", pol)):
            rows.append(dict(run="control", **summarise(name, score_races(p, cte, ccfg))))
        ctl = score_races(pol, cte, ccfg)
        obj = ctl["p_profit"] - ctl["p_loss"]
        control_ok = bool(obj.mean() > 0 and (race_t(obj) or 0) > 2)
        print(pd.DataFrame([r for r in rows if r["run"] == "control"]).round(3).to_string(index=False))
        print("control: the agents FIND the planted way to profit in most outcomes" if control_ok else
              "control: NOT found -> a null result on real races means little", flush=True)
    if a.control_only:
        json.dump(dict(control_ok=control_ok), open(os.path.join(a.out, "verdict.json"), "w"), indent=1)
        print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")
        return

    print(f"\n=== 1. Real races: {len(te)} TEST races ===", flush=True)
    base = [("do nothing", noop_policy), ("random", random_policy()),
            ("lay every runner at 3.0-5.0 (a win-rate trap)", lay_band_policy())]
    for name, p in base:
        rows.append(dict(run="real", **summarise(name, score_races(p, te, cfg))))
    results = {}
    slow = replace(cfg, exchange=replace(cfg.exchange, latency_steps=2))
    pess = replace(cfg, exchange=replace(cfg.exchange, queue_mult=2.0, trade_fill_frac=0.5))
    for sd in [int(x) for x in a.seeds.split(",") if x.strip()]:
        print(f"\n--- seed {sd}: {a.steps} steps ---", flush=True)
        ck = train(a.tapes, os.path.join(a.out, f"real_seed{sd}"), cfg, a.steps, int(a.steps * a.warmup_frac), sd,
                   eval_races=a.eval_races, split=(tr, va, te), **kw)
        pol = make_runner_policy(load_runner_actor(ck, a.device), a.device)
        d = score_races(pol, te, cfg)
        d.to_csv(os.path.join(a.out, f"test_races_seed{sd}.csv"), index=False)
        main_row = dict(run=f"seed {sd}", **summarise("agent", d))
        rows.append(main_row)
        for name, c in (("agent, 1s latency", slow), ("agent, pessimistic fills", pess)):
            rows.append(dict(run=f"seed {sd}", **summarise(name, score_races(pol, te, c))))
        results[sd] = (main_row, rows[-2], rows[-1])

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    print("\n=== 2. TEST races (never trained on): win rate is over the races the policy bet on; $ per race ===")
    print(res.round(3).to_string(index=False))
    met = {s: bool((r[0]["wins_vs_losses_z"] or 0) > 2) for s, r in results.items()}
    useful = {s: bool(met[s] and r[0]["ev_bsp_per_race"] > 0 and (r[0]["ev_t"] or 0) > 2 and r[0]["realised_per_race"] > 0
                      and all(x["ev_bsp_per_race"] > 0 for x in r[1:])) for s, r in results.items()}
    verdict = dict(control_ok=control_ok, objective_met=met, useful=useful, reward=a.reward,
                   seeds={str(s): r[0] for s, r in results.items()})
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    for s in results:
        r = results[s][0]
        if useful[s]:
            msg = "USEFUL: wins more often than not AND makes money at BSP, surviving both stresses -> forward-test it"
        elif met[s]:
            msg = (f"WIN-RATE TRAP: wins more often than not ({r['realised_win_pct']:.0f}% vs {r['realised_loss_pct']:.0f}%) "
                   f"but EV per race {r['ev_bsp_per_race']:+.2f} (avg win {r['avg_win']:+.2f}, avg loss {r['avg_loss']:+.2f})")
        else:
            msg = "OBJECTIVE NOT MET on TEST races"
        print(f"seed {s}: {msg}")
    print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
