"""P3b forward test: the one rule that passed, frozen, scored only on days it has never seen.

The rule (fixed on 2026-10-05 from the P3b run; never re-tuned):

  event    a >= 5 tick DRIFT sweep within 5s (>= $20 traded, spread <= 3, price <= 30)
           on a top-3 runner by matched volume
  entry    rest a BACK at the best lay price (join the queue); cancel if unfilled after 30s
  exit     take-profit lay 3 ticks shorter, sized to green; otherwise at the scheduled
           start whatever is open is closed with a BSP bet
  stake    $10, full queue-aware simulator ('realistic' fills), commission
  score    expected P&L at BSP odds per attempt (luck-free), race-clustered t

One change from P3b, to remove look-ahead: P3b ranked "top 3" by matched volume AT
THE OFF, which a live bot can't know. Here the top 3 are ranked by volume matched up
to the event, and events in the last 40s before the off are kept (the bot doesn't
know when the off is). Section 1 re-scores the old days both ways so the effect of
that fix is visible.

Pre-registered decision (forward days only, i.e. day >= --since):
  fewer than 300 attempts or 100 races        CONTINUE (keep recording)
  >= 300 attempts: mean <= 0                   FAIL (stop)
                   mean > 0 and t >= 2         PASS (go to live paper trading)
                   mean > 0 and t < 2          CONTINUE to 600 attempts, then PASS only
                                               if t >= 2, else FAIL
Re-running this every few days is fine: the rule and the thresholds never change.

    python -m ahr_rl.p3b_forward --tapes "data/tapes/*.npz" --out runs/p3b_forward
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

from .env import list_tapes, split_by_date
from .exchange import BACK
from .sweep_bsp import _race_t, run_trade
from .sweep_passive import find_events
from .tape import Tape

FROZEN = dict(J=5, dir="drift", W=30, tp=3, hold="start", runners="top3", exit="bsp", fill_mode="realistic",
              frozen_on="20261005")
SINCE = "20261005"  # first day the P3b run never saw
MIN_ATTEMPTS, MIN_RACES, MAX_ATTEMPTS = 300, 100, 600
# P3b holdout (look-ahead top 3), for comparison
P3B_HOLDOUT = dict(ev_per_attempt=0.0102, t=2.15, races=178, fill_pct=70.9)


def study_tape(path: str) -> list[dict]:
    t = Tape.load(path)
    if not t.went_in_play:
        return []
    race = os.path.basename(path).split(".npz")[0]
    rng = np.random.default_rng(0)
    events, end = find_events(t, jumps=(FROZEN["J"],), rng=rng, control_every_s=1e9, live=True)
    after0 = np.where(t.t_rel >= 0)[0]
    s_start = int(after0[0]) if len(after0) else end
    rows = []
    for e in events:
        if e["kind"] != "sweep" or e["direction"] <= 0 or e["step"] >= s_start - 10:
            continue
        if not (e["top3"] or e["top3_final"]):
            continue
        res = run_trade(t, end, s_start, e["step"], e["runner"], BACK, FROZEN["W"], FROZEN["tp"], FROZEN["hold"],
                        fill_mode=FROZEN["fill_mode"])
        rows.append(dict(race=race, day=race[:8], runner=e["runner"], step=e["step"], t_rel=e["t_rel"],
                         top3_live=e["top3"], top3_final=e["top3_final"],
                         # the event set P3b scored: final-volume top 3, not in the last 40s before the off
                         in_p3b=bool(e["top3_final"] and e["step"] < end - int(40 / t.dt)), **res))
    return rows


def _one(path):
    try:
        return study_tape(path)
    except Exception as ex:
        print(f"  skip {os.path.basename(path)}: {ex}", flush=True)
        return []


def summary(d: pd.DataFrame) -> dict:
    """Per attempt: P&L at BSP odds x filled fraction (0 if unfilled)."""
    if len(d) == 0:
        return dict(attempts=0, races=0, fill_pct=np.nan, tp_pct=np.nan, ev_per_attempt=np.nan, t=np.nan,
                    realised_per_fill=np.nan, worst_per_fill=np.nan)
    pa = d["bsp_ev"].fillna(0) * d["filled"]
    rt = _race_t(pa, d["race"])
    f = d[d["filled"] > 0]
    return dict(attempts=len(d), races=rt["races"], fill_pct=len(f) / len(d) * 100,
                tp_pct=f["hit_tp"].mean() * 100 if len(f) else np.nan, ev_per_attempt=float(pa.mean()), t=rt["t"],
                realised_per_fill=f["bsp_realised"].mean() if len(f) else np.nan,
                worst_per_fill=f["bsp_worst"].mean() if len(f) else np.nan)


def decide(s: dict) -> str:
    n, m, t = s["attempts"], s["ev_per_attempt"], s["t"]
    if n < MIN_ATTEMPTS or s["races"] < MIN_RACES:
        return "CONTINUE"
    if not np.isfinite(m) or m <= 0:
        return "FAIL"
    if np.isfinite(t) and t >= 2:
        return "PASS"
    return "FAIL" if n >= MAX_ATTEMPTS else "CONTINUE"


def plot(fwd: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if fwd.empty:
        return
    d = fwd.sort_values(["day", "race", "step"])
    pa = (d["bsp_ev"].fillna(0) * d["filled"]).to_numpy() * 10  # $ per $10 attempt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(1, len(pa) + 1), np.cumsum(pa), lw=1.5, label="frozen rule, forward days")
    ax.plot(np.arange(1, len(pa) + 1), np.arange(1, len(pa) + 1) * P3B_HOLDOUT["ev_per_attempt"] * 10, ls="--",
            lw=1, color="grey", label="P3b holdout rate (+1.0% / attempt)")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(MIN_ATTEMPTS, color="red", lw=0.8, ls=":", label=f"decision at {MIN_ATTEMPTS} attempts")
    ax.set_xlabel("attempt")
    ax.set_ylabel("cumulative expected P&L ($, $10 stakes)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--since", default=SINCE, help="first forward day (YYYYMMDD); keep the default")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    paths = list_tapes(a.tapes)
    old = [p for p in paths if os.path.basename(p)[:8] < a.since]
    new = [p for p in paths if os.path.basename(p)[:8] >= a.since]
    print(f"frozen rule: {FROZEN}")
    print(f"{len(old)} races before {a.since} (seen by P3b), {len(new)} forward races")
    os.environ["OMP_NUM_THREADS"] = "1"
    rows = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, res in enumerate(pool.map(_one, paths, chunksize=2)):
            rows.extend(res)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(paths)} races, {len(rows)} attempts, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no events")
    df["hit_tp"] = df["hit_tp"].fillna(False).astype(bool)
    df.to_parquet(os.path.join(a.out, "attempts.parquet"), index=False)
    df["forward"] = df["day"] >= a.since

    # ---------------- 1. old days: does removing the look-ahead change the result?
    h = df[~df["forward"]]
    _, va, te = split_by_date(old) if old else ([], [], [])
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    print("\n=== 1. Days P3b already used: the rule with P3b's look-ahead top 3 vs the live top 3 ===")
    rows_s = []
    for name, sel in (("P3b events (top 3 at the off)", h["in_p3b"]), ("live (top 3 so far)", h["top3_live"])):
        for split in ("train", "holdout"):
            m = sel & (h["day"].isin(hold_days) if split == "holdout" else ~h["day"].isin(hold_days))
            rows_s.append(dict(definition=name, split=split, **summary(h[m])))
    old_tab = pd.DataFrame(rows_s)
    print(old_tab.round(4).to_string(index=False))
    print(f"(P3b reported holdout: {P3B_HOLDOUT['ev_per_attempt'] * 100:+.2f}% per attempt, t {P3B_HOLDOUT['t']:.2f}, "
          f"{P3B_HOLDOUT['races']} races; small differences come from the 40s-before-the-off events)")

    # ---------------- 2. forward days: the test
    fwd = df[df["forward"] & df["top3_live"]]
    s = summary(fwd)
    status = decide(s)
    print(f"\n=== 2. FORWARD TEST: days >= {a.since}, live top 3 ===")
    if len(fwd):
        print(f"{fwd['day'].min()} .. {fwd['day'].max()}, {fwd['day'].nunique()} days")
        by_day = fwd.groupby("day").apply(lambda g: pd.Series(summary(g)), include_groups=False)
        print(by_day[["attempts", "races", "fill_pct", "tp_pct", "ev_per_attempt"]].round(4).to_string())
    print("\noverall: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()))
    plot(fwd, os.path.join(a.out, "forward_cumulative.png"))

    # pace: how many more days until the first decision
    need = ""
    if status == "CONTINUE" and len(fwd):
        per_day = len(fwd) / fwd["day"].nunique()
        left = max(MIN_ATTEMPTS - len(fwd), 0)
        need = f" (~{left / max(per_day, 1e-9):.0f} more recording days at {per_day:.0f} attempts/day)" if left else ""
    verdict = dict(rule=FROZEN, since=a.since, forward=s, status=status,
                   decision_rule=f">= {MIN_ATTEMPTS} attempts & {MIN_RACES} races: mean <= 0 FAIL; t >= 2 PASS; "
                                 f"else continue to {MAX_ATTEMPTS}, then PASS only if t >= 2")
    json.dump(verdict, open(os.path.join(a.out, "forward_status.json"), "w"), indent=1, default=float)
    print(f"\n=== STATUS: {status}{need} ===")
    print({"CONTINUE": "keep recording; re-run this cell every few days. Do not change the rule.",
           "PASS": "forward days confirm it: next step is live paper trading of this exact rule.",
           "FAIL": "forward days do not confirm it: P3b's holdout result was luck or a passing regime."}[status])
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
