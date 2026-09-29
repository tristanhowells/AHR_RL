"""P3b: the post-sweep fade, tuned, with a BSP exit.

P3 found one near-break-even setup: after a >= 5 tick DRIFT sweep, rest a back
order at the best lay price (join the queue on the emptied side), take profit a
couple of ticks shorter. This refines only that family:

  events   sweeps of >= 5 ticks within 5s (drift and steam), and >= 3 tick drift
           sweeps for comparison; tradeable runners (spread <= 3, price <= 30)
  entry    fade side, JOIN the best price, cancel if unfilled after W = 10 / 30s
  take-profit  tp = 1 / 2 / 3 ticks, resting
  exit     at hold = 60s, 120s, or the scheduled start (the last point a bot can
           know about in advance), whatever is still open is closed either
             aggressive: cross the spread (up to 5 ticks), as in P3
             bsp:        a Betfair SP bet sized from the current price, matched at
                         BSP at the off (P1 found BSP is 0.3-1.2% cheaper than
                         crossing the spread near the off)
           No knowledge of when the race actually goes off is used (P3 clipped
           exits to 10s before the real off, a small look-ahead). If the off comes
           first, the position goes in-play as it stands.

A BSP exit is not an exact green (BSP is unknown when the bet is placed), so every
trade is scored on
  ev        expected P&L using BSP-implied probabilities (luck-free; main metric)
  worst     worst case over win / lose
  realised  P&L given the actual winner
all per $1 of matched entry, after commission. Rules are picked on TRAIN days by
race-clustered t of ev per attempt and scored once on HOLDOUT days.

    python -m ahr_rl.sweep_bsp --tapes "data/tapes/*.npz" --out runs/sweep_bsp
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .env import list_tapes, split_by_date
from .exchange import BACK, LAY, Exchange, ExchangeConfig
from .ladder import N_TICKS, PRICES
from .sweep_passive import find_events
from .tape import Tape

STAKE = 10.0
GRID = dict(W=(10, 30), tp=(1, 2, 3), hold=("60", "120", "start"))


def _score(W, L, comm, q, winner_is_r):
    """W/L: P&L if the runner wins / loses. Commission on a positive market result."""
    net = lambda x: x * (1 - comm) if x > 0 else x
    w, l = net(W), net(L)
    return dict(ev=q * w + (1 - q) * l, worst=min(w, l), realised=(w if winner_is_r else l) if winner_is_r is not None else np.nan)


def run_trade(t: Tape, end: int, s_start: int, s: int, r: int, side: int, W: int, tp: int, hold: str,
              fill_mode: str = "realistic") -> dict:
    dt = t.dt
    ex = Exchange(t, ExchangeConfig(fill_mode=fill_mode), start_step=s)
    bb, bl = ex.best(s, r)
    if bb < 0 or bl < 0:
        return dict(filled=0.0)
    tick = bl if side == BACK else bb
    o = ex.submit(r, side, tick, STAKE)
    if o is None:
        return dict(filled=0.0)
    s_entry_end = min(s + int(W / dt), end)
    while ex.step < s_entry_end and o.matched < STAKE - 1e-6:
        ex.advance()
    matched = o.matched
    if matched < 0.01:
        return dict(filled=0.0)
    ex.cancel_runner(r)
    ex.advance()
    p_in = PRICES[tick]
    hside = LAY if side == BACK else BACK
    tp_tick = int(np.clip(tick - tp if side == BACK else tick + tp, 0, N_TICKS - 1))
    o_tp = ex.submit(r, hside, tp_tick, matched * p_in / PRICES[tp_tick], is_hedge=True)
    if hold == "start":
        s_exit = s_start
    else:
        s_exit = ex.step + int(int(hold) / dt)
    hit_tp = False
    while ex.step < min(s_exit, end):
        ex.advance()
        # take-profit fully matched (stakes are rounded to cents, so W - L is only ~0, not exactly 0)
        if o_tp is not None and o_tp.size <= 1e-6 and o_tp.matched > 0:
            hit_tp = True
            break
    comm = ex.commission
    bsp = float(t.bsp[r]) if t.bsp[r] > 1 else np.nan
    q = 1.0 / bsp if np.isfinite(bsp) else np.nan
    win = (t.winner == r) if t.winner >= 0 else None
    out = dict(filled=matched / STAKE, hit_tp=hit_tp, reached_off=ex.step >= end)
    if hit_tp or ex.step >= end:  # green already, or the race went off first: position stands
        sc = _score(ex.W[r], ex.L[r], comm, q, win)
        for mode in ("aggressive", "bsp"):
            out.update({f"{mode}_{k}": v / matched for k, v in sc.items()})
        return out
    # ---- BSP exit (computed before the aggressive exit mutates the book state)
    ex.cancel_runner(r)
    Wr, Lr = float(ex.W[r]), float(ex.L[r])
    bb, bl = ex.best(ex.step, r)
    p_ref = (PRICES[bb] * PRICES[bl]) ** 0.5 if bb >= 0 and bl >= 0 else p_in
    d = Wr - Lr  # > 0: long the runner -> lay at BSP; < 0: back at BSP
    if np.isfinite(bsp) and abs(d) > 1e-6:
        x = abs(d) / p_ref  # stake that would green at the current price
        if d > 0:
            Wb, Lb = Wr - x * (bsp - 1), Lr + x
        else:
            Wb, Lb = Wr + x * (bsp - 1), Lr - x
        sc = _score(Wb, Lb, comm, q, win)
        out.update({f"bsp_{k}": v / matched for k, v in sc.items()})
    else:
        out.update({"bsp_ev": np.nan, "bsp_worst": np.nan, "bsp_realised": np.nan})
    # ---- aggressive exit
    for _ in range(6):
        ex.cancel_runner(r)
        plan = ex.hedge_plan(r)
        if plan is None or ex.step >= end:
            break
        hs, ht, hst = plan
        lim = int(np.clip(ht + 5 if hs == LAY else ht - 5, 0, N_TICKS - 1))
        ex.submit(r, hs, lim, hst, is_hedge=True)
        ex.advance()
        ex.advance()
    ex.cancel_runner(r)
    sc = _score(float(ex.W[r]), float(ex.L[r]), comm, q, win)
    out.update({f"aggressive_{k}": v / matched for k, v in sc.items()})
    return out


def study_tape(path: str, fill_mode: str = "realistic", seed: int = 0) -> list[dict]:
    t = Tape.load(path)
    if not t.went_in_play:
        return []
    race = os.path.basename(path).split(".npz")[0]
    rng = np.random.default_rng(abs(hash(race)) % (2**32) + seed)
    events, end = find_events(t, rng=rng, control_every_s=1e9)
    tr = t.t_rel
    after0 = np.where(tr >= 0)[0]
    s_start = int(after0[0]) if len(after0) else end
    rows = []
    for e in events:
        if e["kind"] != "sweep" or (e["J"] == 3 and e["direction"] < 0):
            continue
        side = BACK if e["direction"] > 0 else LAY  # fade: back after a drift, lay after a steam
        for W, tp, hold in itertools.product(*GRID.values()):
            if hold == "start" and e["step"] >= s_start - 10:
                continue
            res = run_trade(t, end, s_start, e["step"], e["runner"], side, W, tp, hold, fill_mode=fill_mode)
            rows.append(dict(race=race, day=race[:8], J=e["J"], dir="drift" if e["direction"] > 0 else "steam",
                             top3=e["top3"], t_rel=e["t_rel"], W=W, tp=tp, hold=hold, **res))
    return rows


def _one(args):
    path, kw = args
    try:
        return study_tape(path, **kw)
    except Exception as ex:
        print(f"  skip {os.path.basename(path)}: {ex}", flush=True)
        return []


def _race_t(v, race):
    x = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna()
    if len(x) < 2:
        return dict(n=len(x), races=x["race"].nunique(), mean=np.nan, t=np.nan)
    per = x.groupby("race")["v"].mean()
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 and per.std(ddof=1) > 0 else np.nan
    return dict(n=len(x), races=len(per), mean=float(x["v"].mean()), t=float(t))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-races", type=int, default=100000)
    ap.add_argument("--min-races", type=int, default=30)
    ap.add_argument("--fill-mode", default="realistic", choices=["realistic", "no_queue"])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    t0 = time.time()
    paths = list_tapes(a.tapes)[: a.max_races]
    tr, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    os.environ["OMP_NUM_THREADS"] = "1"
    rows = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, res in enumerate(pool.map(_one, [(p, dict(fill_mode=a.fill_mode)) for p in paths], chunksize=2)):
            rows.extend(res)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(paths)} races, {len(rows)} simulated trades, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["split"] = np.where(df["day"].isin(hold_days), "holdout", "train")
    for c in ("hit_tp", "reached_off"):  # missing for unfilled attempts -> real booleans
        df[c] = df[c].fillna(False).astype(bool)
    df.to_parquet(os.path.join(a.out, "trades.parquet"), index=False)
    f = df[df["filled"] > 0]
    print(f"\n{len(df)} simulated attempts ({df['race'].nunique()} races), fill mode {a.fill_mode}; "
          f"filled {len(f) / max(len(df), 1) * 100:.0f}%, take-profit hit on {f['hit_tp'].mean() * 100:.0f}% of fills, "
          f"{f['reached_off'].mean() * 100:.1f}% of fills still open at the off")

    # ---------------- rule table
    rows_r = []
    for keys, g in df.groupby(["J", "dir", "W", "tp", "hold"]):
        for top in ("all", "top3"):
            gg = g if top == "all" else g[g["top3"]]
            for mode in ("aggressive", "bsp"):
                rec = dict(zip(["J", "dir", "W", "tp", "hold"], keys), runners=top, exit=mode)
                for sp, s in gg.groupby("split"):
                    pa = _race_t(s[f"{mode}_ev"].fillna(0) * s["filled"], s["race"])  # per attempt
                    fl = s[s["filled"] > 0]
                    rec.update({f"{sp}_attempts": len(s), f"{sp}_fill_%": (s["filled"] > 0).mean() * 100,
                                f"{sp}_tp_%": fl["hit_tp"].mean() * 100 if len(fl) else np.nan,
                                f"{sp}_ev_per_fill": fl[f"{mode}_ev"].mean(),
                                f"{sp}_worst_per_fill": fl[f"{mode}_worst"].mean(),
                                f"{sp}_realised_per_fill": fl[f"{mode}_realised"].mean(),
                                f"{sp}_ev_per_attempt": pa["mean"], f"{sp}_t": pa["t"], f"{sp}_races": pa["races"]})
                rows_r.append(rec)
    R = pd.DataFrame(rows_r)
    R = R[R.get("train_races", 0) >= a.min_races].sort_values("train_t", ascending=False)
    R.to_csv(os.path.join(a.out, "rules.csv"), index=False)
    cols = [c for c in ["J", "dir", "W", "tp", "hold", "runners", "exit", "train_attempts", "train_fill_%", "train_tp_%",
                        "train_ev_per_fill", "train_ev_per_attempt", "train_t", "holdout_fill_%", "holdout_ev_per_fill",
                        "holdout_worst_per_fill", "holdout_realised_per_fill", "holdout_ev_per_attempt", "holdout_t",
                        "holdout_races"] if c in R]
    print("\n" + "=" * 100 + "\nRULES (ranked on TRAIN days by t of expected P&L per attempt; per $1, after commission)\n"
          + "=" * 100)
    print(R[cols].head(20).round(4).to_string(index=False))

    print("\n" + "=" * 100 + "\nBSP EXIT vs AGGRESSIVE EXIT on the same trades (all days, filled trades not closed by the "
          "take-profit)\n" + "=" * 100)
    open_ = f[~f["hit_tp"] & ~f["reached_off"]]
    for keys, g in open_.groupby(["J", "dir", "hold"]):
        print(f"  J>={keys[0]} {keys[1]:5s} hold {keys[2]:5s}: n={len(g):6d}  aggressive ev {g['aggressive_ev'].mean():+.4f}"
              f"  bsp ev {g['bsp_ev'].mean():+.4f}  (bsp better by {(g['bsp_ev'] - g['aggressive_ev']).mean():+.4f})")

    best = R.iloc[0].to_dict() if len(R) else {}
    verdict = dict(best_rule={k: best.get(k) for k in ["J", "dir", "W", "tp", "hold", "runners", "exit"]},
                   train_ev_per_attempt=best.get("train_ev_per_attempt"), train_t=best.get("train_t"),
                   holdout_ev_per_attempt=best.get("holdout_ev_per_attempt"), holdout_t=best.get("holdout_t"),
                   holdout_races=best.get("holdout_races"), holdout_realised_per_fill=best.get("holdout_realised_per_fill"))
    verdict["edge"] = bool(best and (verdict["holdout_ev_per_attempt"] or 0) > 0 and (verdict["holdout_t"] or 0) > 2
                           and (verdict["holdout_races"] or 0) >= 20)
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=str)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=str))
    print("EDGE: positive on unseen days (t > 2, >= 20 races) -> candidate for live paper trading" if verdict["edge"]
          else "NO EDGE on unseen days")
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
