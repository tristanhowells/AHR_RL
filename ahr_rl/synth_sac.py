"""P20: black-box SAC pre-trained on endless synthetic races, fine-tuned on real TRAIN races, picked on real VAL,
scored once on real TEST.

Pipeline (each stage resumes: a finished stage is skipped on a re-run):
  split     chronological split of the real races by day (TRAIN / VAL / TEST); the generator and every
            training step only ever see TRAIN
  fit       synth_market: fit the market model on TRAIN races and calibrate it (vol by time to the off,
            60s variance ratio, trade-side information, 30s momentum)
  parity    synth_parity: real TRAIN vs synthetic races on what a trader can exploit - momentum / reversal at
            30s and 120s, the drift-fade, the cost of crossing, resting-order fill rate and adverse selection.
            parity_ok = every critical statistic matches. --require-parity stops here if it does not.
  pretrain  the P17b agent (per-runner SAC, a whole trade per action: side, size, entry offset, hold,
            take-profit, stop; reward at the close; gamma 1, n-step returns) on a FRESH synthetic race every
            episode (endless: no race is ever seen twice, so nothing can be memorised - P18's failure).
            Exploration: a long warm-up (--pretrain-warmup-frac of the steps) of random strategies plus
            drift-fade demonstrations, a higher entropy target and an entropy floor (--alpha-floor) so the
            policy keeps exploring however sure the critic gets. Checkpoints scored on real VAL races.
  finetune  start from the VAL-best pretrained checkpoint (actor, critics, alpha); the buffer is first filled by
            the pretrained policy itself, then SAC continues on real TRAIN races at a lower learning rate. The
            pretrained policy is scored on VAL first, so fine-tuning must beat it, not just replace it.
  baseline  the same agent trained on real TRAIN races only, from scratch (P17b): does synthetic data help?
  test      once, on real TEST races: do nothing, random, the drift-fade, the zero-shot pretrained agent, the
            fine-tuned agent (+ 1s latency and pessimistic-fill stresses) and the real-only baseline.
            Artefact check: the pretrained agent on 200 unseen synthetic races vs real VAL. A clear profit on
            synthetic races that real races do not show = it learned the generator, not the market.

    python -m ahr_rl.synth_sac --tapes "data/tapes/*.npz" --out runs/synth_sac --work /tmp/p20
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

from . import synth_market
from .blackbox import gate_summary, liquidity_cut, race_t, score
from .bracket_agents import (CEILING, N_ACT6, BracketAgentEnv, BracketConfig, demo_policy, noop_policy,
                             random_policy, trade_log)
from .env import list_tapes, split_by_date
from .evaluate import evaluate
from .market_select import select_markets
from .runner_agents import load_runner_actor, make_runner_policy, test_block, train
from .synth_parity import parity


class SynthBracketEnv(BracketAgentEnv):
    """BracketAgentEnv whose episodes are fresh synthetic races, generated on reset from a fitted market model.
    reset(options={"tape": ...}) still plays the given (real) race, so VAL / TEST scoring is unchanged."""

    MODEL_PATH: str | None = None
    GEN: dict | None = None

    def __init__(self, tapes, cfg: BracketConfig | None = None, model=None, gen=None, **kw):
        super().__init__(tapes, cfg, **kw)
        model = model if model is not None else self.MODEL_PATH
        self.synth = synth_market.load(model) if isinstance(model, str) else model
        self.gen = gen if gen is not None else self.GEN
        self._grng = np.random.default_rng([kw.get("seed") or 0, 20])
        self._n_gen = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._grng = np.random.default_rng([seed, 20])
        if (options and "tape" in options) or self.synth is None:
            return super().reset(seed=seed, options=options)
        self._n_gen += 1
        tape = synth_market.generate_market(self.synth, self._grng, f"20990101_LIVE_{self._n_gen}", self.gen)
        return super().reset(seed=seed, options={"tape": tape})


def synth_env_cls(model_path: str, gen: dict | None = None):
    """A SynthBracketEnv subclass bound to a model file (picklable into vector-env workers)."""
    return type("BoundSynthBracketEnv", (SynthBracketEnv,), {"MODEL_PATH": model_path, "GEN": gen})


def _done(d) -> bool:
    return os.path.exists(os.path.join(d, "DONE"))


def _mark(d, **info):
    os.makedirs(d, exist_ok=True)
    json.dump(dict(info, finished=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(d, "DONE"), "w"), indent=1)


def _per_race(pol, tapes, cfg) -> pd.Series:
    df = evaluate(pol, tapes, cfg, env_cls=BracketAgentEnv)
    return df.loc[~df["void"]].set_index("market")["worst"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="fast local dir for synthetic races (default: <out>/work)")
    ap.add_argument("--stages", default="fit,parity,pretrain,finetune,baseline,test")
    ap.add_argument("--min-matched", type=float, default=0.0, help="only races with >= this $ matched (VAL/TEST too)")
    ap.add_argument("--min-best-usd", default="auto", help='liquidity gate; "auto" = P11 top quintile on TRAIN days')
    ap.add_argument("--order-cap", type=float, default=20.0)
    ap.add_argument("--runner-cap", type=float, default=50.0)
    ap.add_argument("--market-cap", type=float, default=150.0)
    # generator
    ap.add_argument("--calibrate-iters", type=int, default=8)
    ap.add_argument("--calib-markets", type=int, default=64)
    ap.add_argument("--parity-n", type=int, default=300, help="real TRAIN and synthetic races in the parity check")
    ap.add_argument("--require-parity", action="store_true", help="stop before training if parity fails")
    # pretrain (synthetic)
    ap.add_argument("--pretrain-steps", type=int, default=1_500_000)
    ap.add_argument("--pretrain-warmup-frac", type=float, default=0.4, help="long warm-up: random strategies + demos")
    ap.add_argument("--demo-frac", type=float, default=0.2, help="share of warm-up episodes played by the drift-fade")
    ap.add_argument("--init-alpha", type=float, default=0.2)
    ap.add_argument("--alpha-floor", type=float, default=0.02, help="entropy weight never below this (pretrain)")
    ap.add_argument("--target-entropy-per-dim", type=float, default=0.25,
                    help="target entropy = -this x 6 (P17b used 1.0: lower = more exploration)")
    # finetune / baseline (real)
    ap.add_argument("--finetune-steps", type=int, default=300_000)
    ap.add_argument("--finetune-warmup", type=int, default=15_000, help="steps of the pretrained policy before updates")
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--finetune-alpha-floor", type=float, default=0.005)
    ap.add_argument("--baseline-steps", type=int, default=300_000, help="0 = skip the real-only baseline")
    ap.add_argument("--n-step", type=int, default=12)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--buffer", type=int, default=400_000)
    ap.add_argument("--eval-races", type=int, default=100)
    ap.add_argument("--holdout-synth", type=int, default=200, help="unseen synthetic races for the artefact check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=max(4, os.cpu_count() or 4))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    stages = {s.strip() for s in a.stages.split(",") if s.strip()}
    out = a.out
    work = a.work or os.path.join(out, "work")
    os.makedirs(out, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    if a.device == "cpu":
        print("WARNING: no GPU. Training will be very slow. Use a GPU runtime.", flush=True)
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - a.n_envs))

    # ------------------------------------------------------------------ split
    sp_path = os.path.join(out, "split.json")
    if os.path.exists(sp_path):
        sp = json.load(open(sp_path))
        tr, va, te = sp["train"], sp["val"], sp["test"]
        print(f"[split] reusing {sp_path}", flush=True)
    else:
        tapes = list_tapes(a.tapes)
        if a.min_matched > 0:
            n_all = len(tapes)
            tapes = select_markets(tapes, a.min_matched, a.workers)
            print(f"markets: {len(tapes)} of {n_all} races have >= ${a.min_matched:,.0f} matched at the off", flush=True)
        tr, va, te = split_by_date(tapes)
        json.dump({"train": tr, "val": va, "test": te}, open(sp_path, "w"), indent=1)
    cut = liquidity_cut(tr) if a.min_best_usd == "auto" else float(a.min_best_usd)
    cfg = BracketConfig(random_start_s=0.0, decision_every=20, min_best_usd=cut, order_cap=a.order_cap,
                        runner_cap=a.runner_cap, market_cap=a.market_cap)
    pess = replace(cfg, exchange=replace(cfg.exchange, queue_mult=2.0, trade_fill_frac=0.5))
    extra = (("agent, pessimistic fills", pess),)
    print(f"=== P20 synthetic-pretrained SAC ({a.device}): {len(tr)} TRAIN / {len(va)} VAL / {len(te)} TEST real races "
          f"({os.path.basename(tr[0])[:8]}..{os.path.basename(tr[-1])[:8]} | ..{os.path.basename(va[-1])[:8]} | "
          f"..{os.path.basename(te[-1])[:8]}); gate ${cut:.2f} ===", flush=True)

    # ------------------------------------------------------------------ fit
    model_path = os.path.join(out, "synth", "model.pkl")
    if "fit" in stages and not _done(os.path.join(out, "synth")):
        synth_market.fit(tr, model_path, threads=a.workers, seed=a.seed, calibrate_iters=a.calibrate_iters,
                         calib_markets=a.calib_markets)
        _mark(os.path.join(out, "synth"), n_train=len(tr))
    if not os.path.exists(model_path):
        raise SystemExit("no generator yet: run with the fit stage")

    # ------------------------------------------------------------------ parity
    par_dir = os.path.join(out, "parity")
    parity_ok, failed = None, []
    if "parity" in stages and not _done(par_dir):
        rng = np.random.default_rng(a.seed)
        rp = sorted(rng.choice(tr, min(a.parity_n, len(tr)), replace=False).tolist())
        spool = synth_market.write_pool(model_path, os.path.join(work, "parity_pool"), a.parity_n, seed=1000 + a.seed,
                                        workers=a.workers)
        df, parity_ok = parity(rp, spool, a.workers, par_dir)
        print("\n=== Parity: real TRAIN vs synthetic races ===")
        print(df.round(4).to_string(index=False))
        failed = df.loc[df["critical"] & ~df["ok"], "stat"].tolist()
        print("PARITY OK: synthetic races match real ones on every critical statistic" if parity_ok else
              "PARITY FAILED on: " + ", ".join(failed) + "  (an agent may learn these differences; the fine-tune on "
              "real races and VAL selection are the protection, and the artefact check below measures it)", flush=True)
        _mark(par_dir, parity_ok=parity_ok)
    if os.path.exists(os.path.join(par_dir, "parity.json")):
        pj = json.load(open(os.path.join(par_dir, "parity.json")))
        parity_ok = pj["parity_ok"]
        par = pd.read_csv(os.path.join(par_dir, "parity.csv"))
        failed = par.loc[par["critical"] & ~par["ok"], "stat"].tolist()
    if a.require_parity and parity_ok is False:
        print("--require-parity: stopping before training", flush=True)
        return

    kw = dict(n_envs=a.n_envs, device=a.device, buffer=a.buffer, n_act=N_ACT6, gamma=a.gamma, n_step=a.n_step,
              eval_races=a.eval_races, split=(tr, va, te))
    env_syn = synth_env_cls(model_path)

    # ------------------------------------------------------------------ pretrain
    pre_dir = os.path.join(out, "pretrain")
    if "pretrain" in stages and not _done(pre_dir):
        print(f"\n=== Pretrain: {a.pretrain_steps:,} steps on fresh synthetic races ({a.pretrain_warmup_frac:.0%} warm-up, "
              f"{a.demo_frac:.0%} of it drift-fade demos; alpha from {a.init_alpha}, floor {a.alpha_floor}, target "
              f"entropy -{a.target_entropy_per_dim} x 6); checkpoints scored on real VAL ===", flush=True)
        train("", pre_dir, cfg, a.pretrain_steps, int(a.pretrain_steps * a.pretrain_warmup_frac), a.seed,
              env_cls=env_syn, demo_frac=a.demo_frac, init_alpha=a.init_alpha, alpha_floor=a.alpha_floor,
              target_entropy_per_dim=a.target_entropy_per_dim, eval_every_n=15, **dict(kw, split=(["<synthetic>"], va, te)))
        _mark(pre_dir, steps=a.pretrain_steps)

    # ------------------------------------------------------------------ finetune
    ft_dir = os.path.join(out, "finetune")
    pre_ck = os.path.join(pre_dir, "best.pt")
    if "finetune" in stages and not _done(ft_dir) and os.path.exists(pre_ck):
        print(f"\n=== Fine-tune: {a.finetune_steps:,} steps on real TRAIN races from the VAL-best pretrained "
              f"checkpoint (lr {a.finetune_lr}, alpha floor {a.finetune_alpha_floor}) ===", flush=True)
        train("", ft_dir, cfg, a.finetune_steps, a.finetune_warmup, a.seed, env_cls=BracketAgentEnv, lr=a.finetune_lr,
              init_from=pre_ck, warmup_policy="actor", eval_initial=True, alpha_floor=a.finetune_alpha_floor,
              target_entropy_per_dim=a.target_entropy_per_dim, init_alpha=a.init_alpha, demo_frac=0.0, **kw)
        _mark(ft_dir, steps=a.finetune_steps)

    # ------------------------------------------------------------------ baseline
    bl_dir = os.path.join(out, "baseline")
    if "baseline" in stages and a.baseline_steps > 0 and not _done(bl_dir):
        print(f"\n=== Baseline: the same agent on real TRAIN races only, {a.baseline_steps:,} steps from scratch ===",
              flush=True)
        train("", bl_dir, cfg, a.baseline_steps, int(a.baseline_steps * 0.4), a.seed, env_cls=BracketAgentEnv,
              demo_frac=0.3, **kw)
        _mark(bl_dir, steps=a.baseline_steps)

    if "test" not in stages:
        print(f"\nsaved to {out} ({(time.time() - t0) / 60:.0f} min)")
        return

    # ------------------------------------------------------------------ test
    print(f"\n=== TEST: {len(te)} real races, scored once ===", flush=True)
    if cut > 0:
        g = gate_summary(te, cut)
        print(f"TEST races: {g['runner_share_open'] * 100:.0f}% of runner snapshots pass the gate", flush=True)
    rows = []
    for name, p in (("do nothing", noop_policy), ("random", random_policy()), ("drift-fade (the demo)", demo_policy)):
        rows.append(dict(run="reference", **score(name, evaluate(p, te, cfg, env_cls=BracketAgentEnv))))
    arms = {}
    for label, d, stress in (("pretrained, zero-shot", pre_dir, True), ("fine-tuned", ft_dir, True),
                             ("real-only baseline", bl_dir, True)):
        ck = os.path.join(d, "best.pt")
        if os.path.exists(ck):
            pol, main_row, stress_rows = test_block(label, ck, te, cfg, a.device, rows, stress=stress,
                                                    env_cls=BracketAgentEnv, extra_stress=extra)
            arms[label] = (pol, main_row, stress_rows)
    res = pd.DataFrame(rows)
    res["per_$_traded_%"] = res["mean_green"] / res["turnover"].where(res["turnover"] > 0) * 100
    res.to_csv(os.path.join(out, "results.csv"), index=False)
    print(res.round(3).to_string(index=False))

    # artefact check: the pretrained agent on unseen synthetic races vs real VAL
    artefact = None
    if os.path.exists(pre_ck):
        hold = synth_market.write_pool(model_path, os.path.join(work, "holdout_pool"), a.holdout_synth,
                                       seed=5000 + a.seed, workers=a.workers)
        pol = make_runner_policy(load_runner_actor(pre_ck, a.device), a.device)
        s_syn = score("pretrained on unseen synthetic races", evaluate(pol, hold, cfg, env_cls=BracketAgentEnv))
        s_val = score("pretrained on real VAL races", evaluate(pol, va, cfg, env_cls=BracketAgentEnv))
        artefact = bool(s_syn["mean_green"] > 0 and (s_syn["t"] or 0) > 2 and not (s_val["mean_green"] > 0))
        art = pd.DataFrame([s_syn, s_val])
        art.to_csv(os.path.join(out, "artefact_check.csv"), index=False)
        print("\n=== Artefact check ===")
        print(art.round(3).to_string(index=False))
        print("LEARNED THE GENERATOR: profitable on synthetic races, not on real ones" if artefact else
              "no sign that it learned something only the generator does", flush=True)

    # does synthetic pretraining help? fine-tuned vs real-only baseline, race by race
    helps = None
    if "fine-tuned" in arms and "real-only baseline" in arms:
        d = (_per_race(arms["fine-tuned"][0], te, cfg) - _per_race(arms["real-only baseline"][0], te, cfg)).dropna()
        helps = dict(mean_diff=float(d.mean()), t=race_t(d.to_numpy()), races=int(len(d)))
        print(f"\nfine-tuned minus real-only baseline: {helps['mean_diff']:+.3f} $/race (t {helps['t']:.2f}, "
              f"{helps['races']} races)", flush=True)
    if "fine-tuned" in arms:
        tl = trade_log(arms["fine-tuned"][0], te, cfg)
        tl.to_csv(os.path.join(out, "trades_finetuned.csv"), index=False)
        print(f"\n=== Trades the fine-tuned agent took on TEST races: {len(tl)} ===")
        if len(tl):
            print(tl.groupby(["side", "reason"])["green"].agg(["count", "mean", "sum"]).round(3).to_string())

    def arm(label):
        if label not in arms:
            return None
        m, st = arms[label][1], arms[label][2]
        return dict(test_green=m["mean_green"], t=m["t"], turnover=m["turnover"],
                    per_dollar_traded_pct=(m["mean_green"] / m["turnover"] * 100 if m["turnover"] > 0 else None),
                    **{r["policy"]: r["mean_green"] for r in st})

    ft = arms.get("fine-tuned")
    edge = bool(ft and ft[1]["mean_green"] > 0 and (ft[1]["t"] or 0) > 2 and all(r["mean_green"] > 0 for r in ft[2]))
    above = bool(ft and ft[1]["turnover"] > 0 and ft[1]["mean_green"] / ft[1]["turnover"] * 100 > 2 * CEILING)
    verdict = dict(parity_ok=parity_ok, parity_failed=failed, learned_generator_artefact=artefact, edge=edge,
                   above_measured_ceiling=above, synthetic_pretraining_helps=helps,
                   arms={k: arm(k) for k in ("pretrained, zero-shot", "fine-tuned", "real-only baseline")})
    json.dump(verdict, open(os.path.join(out, "verdict.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    if edge and (above or artefact):
        print("EDGE ON PAPER, BUT SUSPICIOUS (above what P15 / P16 measured, or a generator artefact): treat as an "
              "artefact first")
    elif edge:
        print("EDGE: the fine-tuned agent beats do-nothing on TEST races and survives both stresses -> pre-register "
              "and forward-test it")
    else:
        print("NO EDGE: pretraining on endless synthetic races and fine-tuning on real ones does not beat doing "
              "nothing on unseen real races")
    print(f"\nsaved to {out} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
