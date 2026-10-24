"""P3c arm C: forward test of P16's drift-fade resting back. Registered 2026-10-24, before any forward day.

P16's rule search found one family that held up out of sample, and against its baseline (the same slice without
the signal) the HOLDOUT lift was +0.5 to +0.7 pp per attempt, t 2.7 - 3.9: back a 2nd or 3rd favourite that has
just drifted, resting above the best back price, and close at BSP. It missed the strict bar (holdout t on the mean
1.65 - 1.96 < 2) and it won a search over thousands of slices, so it is a hypothesis. This tests it on days that
did not exist when it was found.

FROZEN rule (do not change it; any change is a new arm with its own start date):
  decisions  every 20s in the last 10 min before the scheduled start (t <= -5s), not suspended
  runners    ranked 2nd or 3rd favourite by mid price among the runners with a two-sided book and a BSP
  signal     mom_30 = the runner's own change in implied probability (mid price) over the last 30s, x100,
             at or below CUT = its 10th percentile over every such runner-decision on days before 20261005
             (computed from those days only, so it is the same number on every re-run)
  order      back, $10 at risk, 3 ticks above the best back price, live one step later, cancelled after 120s
             or at the off; fills from P15's replay of the simulator's passive rules (+ cross-matching)
  value      each attempt = filled fraction x (green at BSP: price / BSP - 1, commission on a positive green)
  races      all race types (as found); thoroughbreds only is reported for information
  forward    days >= 20261025 (the first day after registration)

Decision, counted in RACES (the rule fires ~10 times a race, on the same drifting runner at successive decisions,
so races are the independent unit): >= 100 forward races (and >= 300 attempts): mean <= 0 FAIL; mean > 0 and
race-clustered t >= 2 PASS; otherwise continue to 200 races, then PASS only if t >= 2, else FAIL.

    python -m ahr_rl.arm_c_forward --tapes "data/tapes/*.npz" --out runs/p3b_forward/arm_c --catalogues ...
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import race_filter
from .crossmatch_study import virtual_books
from .env import list_tapes
from .ladder import N_TICKS, PRICES
from .passive_entry_study import ev_back, sim_order
from .rebalance_study import _prob_grid
from .tape import Tape

ARM_C = dict(registered="20261024", since="20261025", cut_before="20261005", runners=(2, 3), signal="mom_30",
             quantile=0.10, side="back", k=3, hold_s=120, stake=10.0, every_s=20.0, last_s=600.0, exit="bsp")
MIN_ATTEMPTS, MIN_RACES, MAX_RACES = 300, 100, 200
P16_HOLDOUT = dict(ev_per_attempt_pct=0.80, t=1.86, races=341, lift_pp=0.69, lift_t=2.75)  # mom_30, 120s, 10%


def _decisions(t: Tape):
    """(steps, mom_30 [T, R], rank [T, R], ok [T, R]) on the frozen decision grid."""
    p, _, _, _, _ = _prob_grid(t)
    dt = t.dt
    k30 = int(round(30.0 / dt))
    vb, vbs, vl, vls, live = virtual_books(t)
    rb, rl = t.back_tick[:, :, 0].astype(np.int32), t.lay_tick[:, :, 0].astype(np.int32)
    dec = np.where((t.t_rel >= -ARM_C["last_s"]) & (t.t_rel <= -5.0) & ~t.suspended)[0]
    dec = dec[::max(1, int(round(ARM_C["every_s"] / dt)))]
    return p, k30, live, rb, rl, (vb, vbs), dec


def tape_rows(path: str, cut: float | None):
    """mom_30 for every eligible runner-decision; with a cut, also the simulated arm-C attempts."""
    t = Tape.load(path)
    if not t.went_in_play:
        return None, None
    race = os.path.basename(path).split(".npz")[0]
    p, k30, live, rb, rl, (vb, vbs), dec = _decisions(t)
    T = t.n_steps
    after0 = np.where(t.t_rel >= 0)[0]
    off = int(after0[0]) if len(after0) else T - 1
    H = int(round(ARM_C["hold_s"] / t.dt))
    comm = t.base_rate / 100.0
    moms, atts = [], []
    for s0 in dec:
        ok = live[s0] & (rb[s0] >= 0) & (rl[s0] >= 0) & (t.bsp > 1.0)
        if ok.sum() < 2 or s0 < k30:
            continue
        mid = np.where(ok, np.sqrt(PRICES[np.maximum(rb[s0], 0)] * PRICES[np.maximum(rl[s0], 0)]), np.inf)
        rank = mid.argsort().argsort() + 1
        mom = (p[s0] - p[s0 - k30]) * 100
        for r in np.where(ok & np.isfinite(mom))[0]:
            moms.append((race, race[:8], int(r), float(mom[r])))
            if cut is None or rank[r] not in ARM_C["runners"] or mom[r] > cut:
                continue
            s1, end = s0 + 1, min(s0 + H, off, T - 1)
            P = int(rb[s0, r]) + ARM_C["k"]
            if s1 >= end or t.suspended[s1] or not 0 <= P < N_TICKS:
                continue
            price = float(PRICES[P])
            m = (t.trade_runner == r) & (t.trade_step > s1) & (t.trade_step <= end)
            W2 = np.arange(s1, end + 1)
            vsize = np.where((vb[W2, r] >= 0) & (vb[W2, r] >= P), vbs[W2, r], 0.0)
            cum, src = sim_order(t, r, "back", P, ARM_C["stake"], s1, end, t.trade_step[m],
                                 t.trade_tick[m].astype(np.int32), t.trade_vol[m].astype(np.float64), vsize, 10.0 / t.dt)
            filled = float(min(cum[-1], ARM_C["stake"]) / ARM_C["stake"])
            ev = float(ev_back(price, float(t.bsp[r]), comm)) * 100
            atts.append(dict(race=race, day=race[:8], runner=int(r), t=float(t.t_rel[s0]), rank=int(rank[r]),
                             mom_30=float(mom[r]), price=price, bsp=float(t.bsp[r]), filled=filled, ev_fill_pct=ev,
                             ev_attempt_pct=filled * ev, mkt_matched_now=float(t.total_matched[s0])))
    return (pd.DataFrame(moms, columns=["race", "day", "runner", "mom_30"]) if moms else None,
            pd.DataFrame(atts) if atts else None)


def _one(args):
    path, cut = args
    try:
        return tape_rows(path, cut)
    except Exception as e:
        print(f"skip {os.path.basename(path)}: {e}", flush=True)
        return None, None


def summary(d: pd.DataFrame) -> dict:
    if d is None or len(d) == 0:
        return dict(attempts=0, races=0, fill_pct=np.nan, ev_per_attempt_pct=np.nan, t=np.nan, ev_given_fill_pct=np.nan)
    per = d.groupby("race")["ev_attempt_pct"].mean()
    t = float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))) if len(per) > 1 and per.std(ddof=1) > 0 else np.nan
    f = d["filled"] > 0
    return dict(attempts=int(len(d)), races=int(len(per)), fill_pct=float(f.mean() * 100),
                ev_per_attempt_pct=float(d["ev_attempt_pct"].mean()), t=t,
                ev_given_fill_pct=float(np.average(d["ev_fill_pct"], weights=d["filled"])) if d["filled"].sum() else np.nan)


def decide(s: dict) -> str:
    n, R, m, t = s["attempts"], s["races"], s["ev_per_attempt_pct"], s["t"]
    if n < MIN_ATTEMPTS or R < MIN_RACES:
        return "CONTINUE"
    if not np.isfinite(m) or m <= 0:
        return "FAIL"
    if np.isfinite(t) and t >= 2:
        return "PASS"
    return "FAIL" if R >= MAX_RACES else "CONTINUE"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--since", default=ARM_C["since"], help="first forward day; keep the default")
    race_filter.add_args(ap)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    t0 = time.time()
    paths = list_tapes(a.tapes)
    day = lambda p: os.path.basename(p)[:8]
    os.environ["OMP_NUM_THREADS"] = "1"
    ctx = mp.get_context("spawn")

    # 1. the frozen cut: 10th percentile of mom_30 on days before cut_before (same number every run)
    cut_file = os.path.join(a.out, "cut.json")
    pre = [p for p in paths if day(p) < ARM_C["cut_before"]]
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as pool:
        moms = [m for m, _ in pool.map(_one, [(p, None) for p in pre], chunksize=4) if m is not None]
    cut = float(np.quantile(pd.concat(moms)["mom_30"], ARM_C["quantile"]))
    if os.path.exists(cut_file):
        old = json.load(open(cut_file))["cut"]
        if abs(old - cut) > 1e-6:
            print(f"WARNING: the cut changed ({old:.5f} -> {cut:.5f}): the tapes before {ARM_C['cut_before']} differ "
                  f"from the first run's. Using the first run's cut, as registered.")
            cut = old
    else:
        json.dump(dict(cut=cut, races=len(moms), computed=time.strftime("%Y%m%d")), open(cut_file, "w"), indent=1)
    print(f"arm C (frozen {ARM_C['registered']}): back 2nd-3rd favourites when mom_30 <= {cut:.4f} (10th pct, days < "
          f"{ARM_C['cut_before']}, {len(moms)} races), rest {ARM_C['k']} ticks above best for {ARM_C['hold_s']}s, "
          f"exit at BSP; forward from {a.since}", flush=True)

    # 2. attempts on every day (history for reference, forward for the decision)
    with ProcessPoolExecutor(a.workers, mp_context=ctx) as pool:
        parts = [d for _, d in pool.map(_one, [(p, cut) for p in paths], chunksize=4) if d is not None]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        info = race_filter.race_info(paths, a.catalogues, a.races_csv).set_index("race")
        df["race_type"] = df["race"].map(info["race_type"]).fillna("Unknown")
        df.to_parquet(os.path.join(a.out, "attempts.parquet"), index=False)
    periods = [("before 20261005 (in-sample: the cut's days)", lambda d: d["day"] < ARM_C["cut_before"]),
               (f"20261005 .. {ARM_C['registered']} (P16's holdout overlaps these)",
                lambda d: (d["day"] >= ARM_C["cut_before"]) & (d["day"] <= ARM_C["registered"])),
               (f"FORWARD (>= {a.since}): the decision", lambda d: d["day"] >= a.since)]
    rows = []
    for name, f in periods:
        d = df[f(df)] if len(df) else df
        rows.append(dict(period=name, **summary(d)))
        if len(d):
            rows.append(dict(period="  thoroughbreds only (info)", **summary(d[d["race_type"].str.lower() == "flat"])))
    tab = pd.DataFrame(rows)
    print("\n=== Arm C: EV per attempt (% of $ at risk, filled fraction x green at BSP) ===")
    print(tab.round(3).to_string(index=False))
    fwd = df[df["day"] >= a.since] if len(df) else df
    s = summary(fwd)
    status = decide(s)
    if len(fwd):
        print("\nforward days:")
        print(fwd.groupby("day").apply(lambda g: pd.Series(summary(g)), include_groups=False)
              [["attempts", "races", "fill_pct", "ev_per_attempt_pct"]].round(3).to_string())
    left = max(MIN_RACES - s["races"], 0)
    print(f"\n=== ARM C STATUS: {status} ({s['attempts']} forward attempts, {s['races']} races; {left} more races to "
          f"the first decision) ===")
    print(f"(P16 holdout for this rule: {P16_HOLDOUT['ev_per_attempt_pct']:+.2f}% per attempt, t {P16_HOLDOUT['t']}, "
          f"lift over its baseline +{P16_HOLDOUT['lift_pp']} pp, t {P16_HOLDOUT['lift_t']})")
    json.dump(dict(definition=ARM_C, cut=cut, forward=s, status=status, periods=tab.to_dict(orient="records"),
                   decision_rule=f">= {MIN_RACES} races & {MIN_ATTEMPTS} attempts: mean <= 0 FAIL; t >= 2 PASS; else "
                                 f"continue to {MAX_RACES} races, then PASS only if t >= 2"),
              open(os.path.join(a.out, "status.json"), "w"), indent=1, default=float)
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
