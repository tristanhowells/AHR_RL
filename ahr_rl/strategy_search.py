"""P9: black-box random-strategy search (no gradients, no hand-made signals).

A strategy is a random rule over the market features the RL agent sees (prices,
spreads, depth, weight of money, 5-60s price changes, volume, projected BSP,
favourite rank, volatility, time to the start, book percentages), standardised:

    score_r = w . runner features + v . market features + b       (w, v random, sparse)
    flat runner:  score > enter  -> OPEN (back, cross the spread)   [if the strategy backs]
                  score < -enter -> OPEN (lay,  cross the spread)   [if the strategy lays]
    in position:  score has turned against it by more than `exit` -> CLOSE

with a random stake ($5 / $10 / $25), exit style (take-profit 2 / 4 / 8 ticks, or hold
with a 12-tick stop), and a cap of 1-3 open positions. The bracket env does the rest
(funds check, take-profit, stops, auto-green from the scheduled start, commission,
queue-aware simulator), so every strategy is executable as written.

Search (all on TRAIN races; scores are $ green per race, $500 bank):
  1  sample --n-random strategies, score each on the same --races-per-eval races
  2  re-score the top 10% on fresh races (cuts the luck of stage 1)
  3  --generations rounds of mutation around the best (cross-entropy style),
     each scored on fresh races; a strategy's score is its mean over every
     race it has been scored on
  4  the top 5 are scored on every VAL race; the best on VAL is the pick, and if
     none beats doing nothing on VAL the pick is "do nothing"
  5  the pick is scored ONCE on TEST races vs doing nothing and a typical random
     strategy, plus a 1s-latency stress

A positive control runs the same search on synthetic races with a small planted
edge first: if the search can't find that, a null result on real races means little.

"edge" = the pick beats doing nothing on TEST (t > 2, >= 20 races) and still wins
under the latency stress.

    python -m ahr_rl.strategy_search --tapes "data/tapes/*.npz" --out runs/strategy_search
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from . import race_filter
from .env import BACK, CLOSE, LAY, TAKE, BetfairPreRaceEnv, EnvConfig, encode_open, list_tapes, split_by_date
from .features import R_MAX
from .synthetic import write_synthetic

# market-only features (no own-position / account features)
RUNNER_IDX = list(range(1, 22)) + [31, 32, 33]  # see RUNNER_NAMES
GLOBAL_IDX = [0, 1, 2, 3, 4, 5]
RUNNER_NAMES = ["back_tick", "lay_tick", "spread", "implied_prob", "back$1", "back$2", "back$3", "lay$1", "lay$2",
                "lay$3", "wom", "ltp_vs_mid", "chg5s", "chg15s", "chg30s", "chg60s", "vol5s", "vol30s", "traded_total",
                "spn_gap", "has_spn", "fav_rank", "volatility60s", "spread_pct"]
GLOBAL_NAMES = ["time_to_start", "elapsed", "back_book", "lay_book", "market_matched", "n_runners"]
STAKE_IDX = (0, 1, 2)  # EnvConfig.stakes -> $5, $10, $25
TP_TICKS = (2, 4, 8, 0)  # 0 = hold (12-tick stop) until the auto-green
SIDES = ("both", "back", "lay")


def env_config(latency_steps: int = 1) -> EnvConfig:
    cfg = EnvConfig(decision_every=20, tp_ticks=TP_TICKS, max_hold_s=900.0)
    return replace(cfg, exchange=replace(cfg.exchange, latency_steps=latency_steps))


# ----------------------------------------------------------------- strategies
def random_strategy(rng) -> dict:
    nR, nG = len(RUNNER_IDX), len(GLOBAL_IDX)
    w = rng.normal(0, 1, nR) * (rng.random(nR) < rng.uniform(0.05, 0.4))
    if not w.any():
        w[rng.integers(nR)] = rng.normal()
    v = rng.normal(0, 1, nG) * (rng.random(nG) < 0.3)
    return dict(w=w / np.linalg.norm(w), v=v * 0.5, b=float(rng.normal(0, 0.5)),
                enter=float(rng.uniform(1.0, 3.0)), exit=float(rng.choice([rng.uniform(0.0, 2.0), np.inf])),
                side=str(rng.choice(SIDES)), stake=int(rng.choice(STAKE_IDX)), tp=int(rng.integers(len(TP_TICKS))),
                max_open=int(rng.integers(1, 4)))


def mutate(s: dict, rng, scale: float) -> dict:
    m = dict(s)
    w = s["w"] + rng.normal(0, scale, len(s["w"])) * (s["w"] != 0)
    flip = rng.random(len(w)) < 0.03  # occasionally switch a feature on / off
    w[flip] = np.where(w[flip] == 0, rng.normal(0, 1, flip.sum()), 0.0)
    if not w.any():
        w[rng.integers(len(w))] = 1.0
    m["w"] = w / np.linalg.norm(w)
    m["v"] = s["v"] + rng.normal(0, scale * 0.5, len(s["v"])) * (s["v"] != 0)
    m["b"] = s["b"] + rng.normal(0, scale * 0.5)
    m["enter"] = float(np.clip(s["enter"] + rng.normal(0, scale), 0.3, 5.0))
    if np.isfinite(s["exit"]):
        m["exit"] = float(np.clip(s["exit"] + rng.normal(0, scale), 0.0, 4.0))
    for k, choices in (("side", SIDES), ("stake", STAKE_IDX), ("tp", range(len(TP_TICKS))), ("max_open", (1, 2, 3))):
        if rng.random() < 0.1:
            m[k] = type(s[k])(rng.choice(list(choices)))
    return m


def describe(s: dict, names_r, names_g) -> str:
    top = np.argsort(-np.abs(s["w"]))[:4]
    terms = " ".join(f"{s['w'][i]:+.2f}*{names_r[i]}" for i in top if s["w"][i] != 0)
    gt = " ".join(f"{s['v'][i]:+.2f}*{names_g[i]}" for i in np.nonzero(s["v"])[0])
    tp = TP_TICKS[s["tp"]]
    return (f"score = {terms} {gt} {s['b']:+.2f}; enter |score|>{s['enter']:.2f} side={s['side']}, "
            f"exit on reversal>{s['exit']:.2f}, stake ${[5, 10, 25][s['stake']]}, "
            f"{'hold' if tp == 0 else f'tp {tp} ticks'}, max {s['max_open']} open")


def make_policy(s: dict, mu_r, sd_r, mu_g, sd_g, cfg: EnvConfig):
    codes = {BACK: encode_open(BACK, TAKE, s["stake"], s["tp"], cfg), LAY: encode_open(LAY, TAKE, s["stake"], s["tp"], cfg)}

    def pol(obs, env):
        a = np.zeros(R_MAX, np.int64)
        X = (obs["runners"][:, RUNNER_IDX] - mu_r) / sd_r
        g = (obs["global"][GLOBAL_IDX] - mu_g) / sd_g
        z = X @ s["w"] + float(g @ s["v"]) + s["b"]
        br = env.ex.brackets
        n_open = len(br)
        for r in np.nonzero(obs["mask"])[0]:
            if r in br:
                d = 1 if br[r].side == BACK else -1
                if z[r] * d < -s["exit"]:
                    a[r] = CLOSE
            elif n_open < s["max_open"]:
                if z[r] > s["enter"] and s["side"] in ("both", "back"):
                    a[r], n_open = codes[BACK], n_open + 1
                elif z[r] < -s["enter"] and s["side"] in ("both", "lay"):
                    a[r], n_open = codes[LAY], n_open + 1
        return a

    return pol


# ----------------------------------------------------------------- evaluation
def _eval_race(args):
    """All strategies on one race -> green $ per strategy (None strategy = do nothing)."""
    path, strategies, stats, latency = args
    cfg = env_config(latency)
    env = BetfairPreRaceEnv([path], cfg, cache_tapes=True)
    out = np.zeros(len(strategies))
    for i, s in enumerate(strategies):
        pol = (lambda o, e: np.zeros(R_MAX, np.int64)) if s is None else make_policy(s, *stats, cfg)
        obs, _ = env.reset(options={"tape": path})
        done, info = False, {}
        while not done:
            obs, _, done, _, info = env.step(pol(obs, env))
        out[i] = 0.0 if info.get("void") else float(info.get("worst", 0.0))
    return out


def evaluate(pool, strategies, races, stats, latency: int = 1) -> np.ndarray:
    """[n_strategies, n_races] green $ per race."""
    if not races:
        return np.zeros((len(strategies), 0))
    cols = list(pool.map(_eval_race, [(p, strategies, stats, latency) for p in races]))
    return np.stack(cols, 1)


def feature_stats(paths, n: int = 12):
    cfg = env_config()
    R, G = [], []
    for p in paths[:n]:
        env = BetfairPreRaceEnv([p], cfg)
        obs, _ = env.reset(options={"tape": p})
        done = False
        while not done:
            m = obs["mask"].astype(bool)
            R.append(obs["runners"][m][:, RUNNER_IDX])
            G.append(obs["global"][GLOBAL_IDX][None])
            obs, _, done, _, _ = env.step(np.zeros(R_MAX, np.int64))
    R, G = np.concatenate(R), np.concatenate(G)
    return R.mean(0), R.std(0) + 1e-6, G.mean(0), G.std(0) + 1e-6


def t_stat(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std(ddof=1) > 0 else float("nan")


def search(paths, pool, a, rng, label: str, log) -> dict:
    tr, va, te = split_by_date(paths)
    log(f"[{label}] races: train {len(tr)}, val {len(va)}, test {len(te)}")
    stats = feature_stats(tr)
    k = min(a.races_per_eval, len(tr))

    def fresh():
        return list(rng.choice(tr, k, replace=False))

    pop = [random_strategy(rng) for _ in range(a.n_random)]
    scores = [[] for _ in pop]
    t0 = time.time()
    R1 = evaluate(pool, pop, fresh(), stats)
    for i in range(len(pop)):
        scores[i].extend(R1[i])
    log(f"[{label}] stage 1: {len(pop)} random strategies x {k} races ({time.time() - t0:.0f}s); "
        f"median {np.median(R1.mean(1)):+.2f} $/race, best {R1.mean(1).max():+.2f}")
    random_sample = [pop[i] for i in rng.choice(len(pop), min(20, len(pop)), replace=False)]

    keep = list(np.argsort(-R1.mean(1))[: max(5, len(pop) // 10)])
    R2 = evaluate(pool, [pop[i] for i in keep], fresh(), stats)
    for j, i in enumerate(keep):
        scores[i].extend(R2[j])
    mean = lambda i: float(np.mean(scores[i]))
    elite = sorted(keep, key=mean, reverse=True)[: a.elite]
    log(f"[{label}] stage 2: re-scored top {len(keep)}; elite mean {np.mean([mean(i) for i in elite]):+.2f} $/race")

    for gen in range(a.generations):
        scale = 0.5 * (0.7 ** gen)
        kids = [mutate(pop[elite[rng.integers(len(elite))]], rng, scale) for _ in range(a.children)]
        cand = elite + list(range(len(pop), len(pop) + len(kids)))
        pop.extend(kids)
        scores.extend([] for _ in kids)
        Rg = evaluate(pool, [pop[i] for i in cand], fresh(), stats)
        for j, i in enumerate(cand):
            scores[i].extend(Rg[j])
        elite = sorted(cand, key=lambda i: (np.mean(scores[i]) - 1.0 / np.sqrt(len(scores[i]))), reverse=True)[: a.elite]
        log(f"[{label}] generation {gen + 1}: best {mean(elite[0]):+.2f} $/race over {len(scores[elite[0]])} races")

    final = sorted(elite, key=mean, reverse=True)[:5]
    V = evaluate(pool, [pop[i] for i in final] + [None], va, stats)
    val = V.mean(1)
    best = int(np.argmax(val[:-1]))
    pick = pop[final[best]] if val[best] > val[-1] else None
    log(f"[{label}] VAL ({len(va)} races): " + ", ".join(f"{x:+.2f}" for x in val[:-1]) + f" | do nothing {val[-1]:+.2f}"
        + (" -> pick #%d" % (best + 1) if pick is not None else " -> pick: do nothing"))

    res = dict(label=label, train_score=mean(final[best]), val=float(val[best]))
    if pick is None:
        res.update(test_green=0.0, test_t=float("nan"), test_races=len(te), stress_green=0.0, random_median=np.nan,
                   pick=None)
        return res
    T = evaluate(pool, [pick, None] + random_sample, te, stats)
    S = evaluate(pool, [pick], te, stats, latency=2)
    names_r, names_g = RUNNER_NAMES, GLOBAL_NAMES
    res.update(test_green=float(T[0].mean()), test_t=t_stat(T[0]), test_races=len(te), test_pct_green=float((T[0] >= -0.005).mean() * 100),
               stress_green=float(S[0].mean()), stress_t=t_stat(S[0]),
               random_median=float(np.median(T[2:].mean(1))), pick=describe(pick, names_r, names_g))
    np.save(os.path.join(a.out, f"{label}_pick.npy"), np.array([pick], dtype=object), allow_pickle=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-random", type=int, default=1500)
    ap.add_argument("--races-per-eval", type=int, default=24)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--children", type=int, default=150)
    ap.add_argument("--elite", type=int, default=15)
    ap.add_argument("--control-races", type=int, default=200)
    ap.add_argument("--trend-every-s", type=float, default=60.0)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--control-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    race_filter.add_args(ap)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    logf = open(os.path.join(a.out, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    rng = np.random.default_rng(a.seed)
    results = []
    os.environ["OMP_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        if not a.skip_control:
            syn = write_synthetic(os.path.join(a.out, "synthetic"), a.control_races, seed0=70_000,
                                  trend_every_s=a.trend_every_s)
            log(f"=== 0. Positive control: planted edge (one runner shortens 1 tick / {a.trend_every_s:.0f}s) ===")
            results.append(search(sorted(syn), pool, a, rng, "control", log))
        if not a.control_only:
            log("\n=== 1. Real races ===")
            results.append(search(race_filter.filter_paths(list_tapes(a.tapes), a, log), pool, a, rng, "real", log))

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(a.out, "results.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 200)
    log("\n=== TEST races (never used for search or selection); $ green per race, $500 bank ===")
    log(res.drop(columns=["pick"]).round(3).to_string(index=False))
    for r in results:
        log(f"{r['label']} pick: {r['pick']}")
    ok = lambda r: bool(r["pick"] is not None and r["test_green"] > 0 and (r["test_t"] or 0) > 2
                        and r["test_races"] >= 20 and r["stress_green"] > 0)
    ctl = next((r for r in results if r["label"] == "control"), None)
    real = next((r for r in results if r["label"] == "real"), None)
    verdict = dict(control_found_planted_edge=ok(ctl) if ctl else None, edge=ok(real) if real else None)
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1)
    log("\n=== Verdict ===\n" + json.dumps(verdict))
    if real is not None:
        log("EDGE: the search found a rule that wins on unseen races -> study it, then forward-test like P3c"
            if verdict["edge"] else
            ("NO EDGE: thousands of random strategies, refined on training races, find nothing that beats "
             "doing nothing on unseen races" + ("" if verdict["control_found_planted_edge"] is not False else
                                                " (but the control failed too: inconclusive)")))
    log(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
