"""Jump study: after a sudden price jump, does the price keep going or snap back?

The race deep dives showed that big pre-off moves are rarely smooth trends.
They are *jumps*: a large order sweeps a thin book and the price moves several
ticks within a few seconds (e.g. Rosehill R2, 29 Aug: Fair Master 22 -> 19.5 in
3s at T-40s, then 16.5, then back to 18). Predicting a jump in advance looked
hopeless, but *reacting* to one only needs a pattern after the event. This
module measures that pattern over every race, and what it would pay after
realistic costs.

Event definition (per runner, on the 0.5s tape):
  * only tradeable runners, as in the edge test: both sides present, spread
    <= ``--max-spread`` ticks (3) before and after, price <= ``--max-price`` (30);
  * mid = midpoint of best back / best lay in *tick* space;
  * a jump is ``|mid(s) - mid(s - W)| >= J`` ticks over a window of ``W``
    seconds (default 5s), with J in ``--jumps`` (default 3 and 5);
  * one event per runner per ``cooldown`` seconds (the first step that
    crosses the threshold);
  * excluded: suspended rows, the in-play row, and anything within 10s of a
    scratching (reduction factors shift every price at once);
  * ``kind`` = "sweep" if at least ``min_sweep_vol`` dollars traded on that
    runner inside the window, else "quote" (the price moved because orders
    were pulled rather than matched).

direction: "steam" = price shortened (ticks fell), "drift" = price lengthened.

For every event and horizon H we record:
  * ``cont_H``: continuation in ticks = further mid move in the jump's
    direction from s to s+H (positive = kept going, negative = snapped back);
  * ``follow_H`` / ``fade_H``: green P&L per $1 of stake for an aggressive
    round trip, net of commission. The entry order is decided at s and lands
    one step later (latency). It takes up to ``--stake`` dollars from the
    visible *post-jump* book, which is often thin, no more than 2 ticks
    through the best price. The exit is decided at s+H and lands a step
    later, sized to green, up to 5 ticks through (any unfilled rest is charged
    at that limit). ``*_touch_H`` is the same trade at the best prices only,
    i.e. the spread cost without the depth cost.
    Follow = trade in the jump's direction (back a steamer, lay a drifter);
    fade = the opposite. Exits are clipped to the last pre-off row.
  * ``fill_frac``: share of the entry stake that the visible book could fill.

Control: the same round trip at random non-event moments with a random
direction. That is the plain cost of crossing the spread twice.

Verdict (no look-ahead): every rule (J, direction, kind, phase, horizon,
follow/fade) is scored on the TRAIN days; the best by race-clustered t
(>= ``min_races`` races) is then scored once on the HOLDOUT days (val + test
of the usual chronological split). The top 10 train rules are shown with their
holdout numbers so you can see how much of the train result survives.

    python -m ahr_rl.jump_study --tapes "data/tapes/*.npz" --out runs/jumps
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
from .ladder import PRICES
from .tape import Tape

PHASES = ((-1e9, -120.0, "early (<T-2m)"), (-120.0, 0.0, "late (T-2m..start)"), (0.0, 1e9, "after start"))


def _phase(t: float) -> str:
    for lo, hi, name in PHASES:
        if lo <= t < hi:
            return name
    return PHASES[-1][2]


def _walk(ticks: np.ndarray, sizes: np.ndarray, stake: float, max_ticks: int) -> tuple[float, float]:
    """Take up to `stake` from one side of the ladder (levels best first), but only
    from levels within `max_ticks` of the best price (a limit order, not a blind
    market order). Returns (volume-weighted price, filled fraction); nan if empty."""
    if ticks[0] < 0:
        return float("nan"), 0.0
    left, pv, filled = stake, 0.0, 0.0
    for k, sz in zip(ticks, sizes):
        if k < 0 or left <= 1e-9 or abs(int(k) - int(ticks[0])) > max_ticks:
            break
        q = min(float(sz), left)
        pv += q * PRICES[k]
        filled += q
        left -= q
    if filled <= 0:
        return float("nan"), 0.0
    return pv / filled, filled / stake


def _green(side_first: str, p_in: float, p_out: float, comm: float) -> float:
    """Green P&L per $1 of the opening stake, commission on a winning result.
    back then lay: p_in / p_out - 1; lay then back: 1 - p_in / p_out."""
    if not (np.isfinite(p_in) and np.isfinite(p_out)) or p_in <= 1 or p_out <= 1:
        return float("nan")
    g = p_in / p_out - 1 if side_first == "back" else 1 - p_in / p_out
    return g * (1 - comm) if g > 0 else g


def _round_trip(tape: Tape, r: int, s_in: int, s_out: int, side_first: str, stake: float, comm: float,
                entry_slip: int = 2, exit_slip: int = 5):
    """Aggressive entry at row s_in (limit `entry_slip` ticks through the best price;
    whatever the book can't fill there simply isn't traded) and an aggressive hedge,
    sized to green, at row s_out. The hedge must complete: any part the visible book
    can't fill within `exit_slip` ticks is charged at that limit price.
    Returns (green per $1 of filled stake, entry fill fraction, touch green per $1),
    where "touch" uses the best prices only (spread cost without depth cost)."""
    if side_first == "back":  # back now into the best atb prices; hedge by laying into atl
        in_t, in_s, out_t, out_s, sgn = tape.back_tick, tape.back_size, tape.lay_tick, tape.lay_size, 1
    else:  # lay now into atl; hedge by backing into atb
        in_t, in_s, out_t, out_s, sgn = tape.lay_tick, tape.lay_size, tape.back_tick, tape.back_size, -1
    nan = float("nan")
    p_in, f = _walk(in_t[s_in, r], in_s[s_in, r], stake, entry_slip)
    if not np.isfinite(p_in) or out_t[s_out, r, 0] < 0:
        return nan, f, nan
    best_out = int(out_t[s_out, r, 0])
    touch = _green(side_first, PRICES[in_t[s_in, r, 0]], PRICES[best_out], comm)
    hedge = stake * f * p_in / PRICES[best_out]  # greening stake at the best exit price
    p_out, g = _walk(out_t[s_out, r], out_s[s_out, r], hedge, exit_slip)
    if g < 1:  # rest of the hedge at the limit price (lay: higher, back: lower)
        p_lim = PRICES[int(np.clip(best_out + sgn * exit_slip, 0, len(PRICES) - 1))]
        p_out = (p_out * g + p_lim * (1 - g)) if np.isfinite(p_out) else p_lim
    return _green(side_first, p_in, p_out, comm), f, touch


def study_tape(path: str, jumps=(3, 5), window_s: float = 5.0, horizons=(5, 10, 30, 60, 120), stake: float = 10.0,
               cooldown_s: float = 15.0, max_spread: int = 3, min_sweep_vol: float = 20.0, max_price: float = 30.0,
               control_every_s: float = 20.0, seed: int = 0) -> list[dict]:
    tape = Tape.load(path)
    race = os.path.basename(path).split(".npz")[0]
    dt = tape.dt
    T = tape.n_steps
    # last pre-off row: drop suspended rows and the in-play snapshot at the end
    ok_row = ~tape.suspended.copy()
    if tape.went_in_play and T > 1:
        ok_row[-1] = False
    if not ok_row.any():
        return []
    end = int(np.where(ok_row)[0].max())
    comm = tape.base_rate / 100.0
    W = int(round(window_s / dt))
    cool = int(round(cooldown_s / dt))
    Hs = [int(round(h / dt)) for h in horizons]
    t = tape.t_rel

    bt, lt = tape.back_tick[:, :, 0].astype(float), tape.lay_tick[:, :, 0].astype(float)
    valid = (bt >= 0) & (lt >= 0) & tape.active & ok_row[:, None]
    spread = np.where(valid, lt - bt, np.inf)
    mid = np.where(valid & (spread <= max_spread), (bt + lt) / 2, np.nan)
    mid[end + 1:] = np.nan

    # rows near a scratching are excluded for every runner
    bad = np.zeros(T, bool)
    for s in tape.removal_step:
        bad[max(0, s - int(10 / dt)): min(T, s + int(10 / dt) + 1)] = True

    # traded $ per runner per row
    vol = np.zeros((T, tape.n_runners))
    np.add.at(vol, (np.clip(tape.trade_step, 0, T - 1), tape.trade_runner), tape.trade_vol)
    cvol = np.vstack([np.zeros((1, tape.n_runners)), np.cumsum(vol, 0)])
    matched = tape.tv[end] if end < T else tape.tv[-1]
    rank = np.argsort(np.argsort(-np.nan_to_num(matched)))  # 0 = most traded runner

    rng = np.random.default_rng(abs(hash(race)) % (2**32) + seed)
    rows = []

    def outcomes(s: int, r: int, direction: int, base: dict) -> dict:
        # direction: -1 = steam (ticks fell), +1 = drift
        rec = dict(base)
        follow_side = "back" if direction < 0 else "lay"
        fade_side = "lay" if direction < 0 else "back"
        s_in = min(s + 1, end)
        f_frac = float("nan")
        for h, H in zip(horizons, Hs):
            s_h = s + H
            clipped = s_h > end
            s_h = min(s_h, end)
            m_now, m_h = mid[s, r], mid[s_h, r]
            if not np.isfinite(m_h):  # fall back to the last valid mid before s_h
                k = np.where(np.isfinite(mid[s:s_h + 1, r]))[0]
                m_h = mid[s + k[-1], r] if len(k) else np.nan
            rec[f"cont_{h}"] = float(direction * (m_h - m_now))
            s_out = min(s_h + 1, end)
            if s_out <= s_in:
                for k in ("follow", "fade", "follow_touch", "fade_touch"):
                    rec[f"{k}_{h}"] = float("nan")
                continue
            rec[f"follow_{h}"], f_frac, rec[f"follow_touch_{h}"] = _round_trip(tape, r, s_in, s_out, follow_side,
                                                                                stake, comm)
            rec[f"fade_{h}"], _, rec[f"fade_touch_{h}"] = _round_trip(tape, r, s_in, s_out, fade_side, stake, comm)
            rec[f"clipped_{h}"] = bool(clipped)
        rec["fill_frac"] = f_frac
        return rec

    for r in range(tape.n_runners):
        m = mid[:, r]
        last_event = -10**9
        for J in jumps:
            last_event = -10**9
            for s in range(W, end):
                if s - last_event < cool or bad[s] or not np.isfinite(m[s]) or not np.isfinite(m[s - W]):
                    continue
                d = m[s] - m[s - W]
                if abs(d) < J:
                    continue
                p_now = PRICES[int(round(m[s]))]
                if p_now > max_price:
                    continue
                last_event = s
                swept = cvol[s + 1, r] - cvol[s - W + 1, r]
                base = dict(race=race, day=race[:8], runner=r, step=s, t_rel=float(t[s]), phase=_phase(float(t[s])),
                            J=J, jump_ticks=float(d), direction="steam" if d < 0 else "drift",
                            kind="sweep" if swept >= min_sweep_vol else "quote", swept_usd=float(swept),
                            price=float(p_now), spread=float(spread[s, r]), runner_rank=int(rank[r]),
                            top3=bool(rank[r] < 3), comm=comm)
                rows.append(outcomes(s, r, -1 if d < 0 else 1, base))
        # control: random moments, random direction, same filters (no jump in the window)
        ce = int(round(control_every_s / dt))
        for s in range(W + int(rng.integers(0, ce)), end, ce):
            if bad[s] or not np.isfinite(m[s]) or not np.isfinite(m[s - W]) or abs(m[s] - m[s - W]) >= min(jumps):
                continue
            p_now = PRICES[int(round(m[s]))]
            if p_now > max_price:
                continue
            direction = int(rng.choice([-1, 1]))
            base = dict(race=race, day=race[:8], runner=r, step=s, t_rel=float(t[s]), phase=_phase(float(t[s])),
                        J=0, jump_ticks=0.0, direction="control", kind="control", swept_usd=0.0, price=float(p_now),
                        spread=float(spread[s, r]), runner_rank=int(rank[r]), top3=bool(rank[r] < 3), comm=comm)
            rows.append(outcomes(s, r, direction, base))
    return rows


def _one(args):
    path, kw = args
    try:
        return study_tape(path, **kw)
    except Exception as e:  # a single bad tape shouldn't kill the study
        print(f"  skip {os.path.basename(path)}: {e}", flush=True)
        return []


def race_t(df: pd.DataFrame, col: str) -> dict:
    """Mean per event, and a t-stat on per-race means (events within a race overlap)."""
    x = df[["race", col]].dropna()
    if not len(x):
        return dict(events=0, races=0, mean=np.nan, t=np.nan)
    per = x.groupby("race")[col].mean()
    sd = per.std(ddof=1)
    tt = per.mean() / (sd / np.sqrt(len(per))) if len(per) > 1 and sd > 0 else np.nan
    return dict(events=len(x), races=len(per), mean=float(x[col].mean()), race_mean=float(per.mean()), t=float(tt))


def continuation_table(ev: pd.DataFrame, horizons) -> pd.DataFrame:
    rows = []
    for (J, direction, kind), g in ev[ev["kind"] != "control"].groupby(["J", "direction", "kind"]):
        row = dict(J=J, direction=direction, kind=kind, events=len(g), races=g["race"].nunique(),
                   jump=g["jump_ticks"].abs().mean(), fill=g["fill_frac"].mean())
        for h in horizons:
            c = g[f"cont_{h}"].dropna()
            row[f"cont_{h}s"] = c.mean()
            row[f"%kept_{h}s"] = (c > 0).mean() * 100
            row[f"%back_{h}s"] = (c < 0).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def rules(ev: pd.DataFrame, horizons, min_races: int) -> pd.DataFrame:
    """Score every rule on the given events."""
    out = []
    jumps = sorted(ev.loc[ev["kind"] != "control", "J"].unique())
    for J in jumps:
        for direction in ("steam", "drift", "both"):
            for kind in ("sweep", "quote", "any"):
                for phase in ("all",) + tuple(p[2] for p in PHASES):
                    for top in ("all", "top3"):
                        g = ev[(ev["J"] == J) & (ev["kind"] != "control")]
                        if direction != "both":
                            g = g[g["direction"] == direction]
                        if kind != "any":
                            g = g[g["kind"] == kind]
                        if phase != "all":
                            g = g[g["phase"] == phase]
                        if top == "top3":
                            g = g[g["top3"]]
                        if g["race"].nunique() < min_races:
                            continue
                        for h in horizons:
                            for act in ("follow", "fade"):
                                s = race_t(g, f"{act}_{h}")
                                out.append(dict(J=J, direction=direction, kind=kind, phase=phase, runners=top,
                                                horizon_s=h, action=act, **s))
    return pd.DataFrame(out)


def score_rule(ev: pd.DataFrame, rule: dict) -> dict:
    g = ev[(ev["J"] == rule["J"]) & (ev["kind"] != "control")]
    if rule["direction"] != "both":
        g = g[g["direction"] == rule["direction"]]
    if rule["kind"] != "any":
        g = g[g["kind"] == rule["kind"]]
    if rule["phase"] != "all":
        g = g[g["phase"] == rule["phase"]]
    if rule["runners"] == "top3":
        g = g[g["top3"]]
    return race_t(g, f"{rule['action']}_{rule['horizon_s']}")


def plot_curves(ev: pd.DataFrame, horizons, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    jev = ev[ev["kind"] != "control"]
    for (J, direction, kind), g in jev.groupby(["J", "direction", "kind"]):
        y = [g[f"cont_{h}"].mean() for h in horizons]
        ls = "-" if kind == "sweep" else ":"
        axes[0].plot(horizons, y, ls, marker="o", label=f"J>={J} {direction} ({kind}, n={len(g)})")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel("seconds after the jump")
    axes[0].set_ylabel("mean further move in jump direction (ticks)\n+ = kept going, - = snapped back")
    axes[0].set_title("Continuation after a jump")
    axes[0].legend(fontsize=7)
    ctl = ev[ev["kind"] == "control"]
    for J, g in jev.groupby("J"):
        axes[1].plot(horizons, [g[f"follow_{h}"].mean() * 100 for h in horizons], "-o", label=f"follow, J>={J}")
        axes[1].plot(horizons, [g[f"fade_{h}"].mean() * 100 for h in horizons], "--o", label=f"fade, J>={J}")
    if len(ctl):
        axes[1].plot(horizons, [ctl[f"follow_{h}"].mean() * 100 for h in horizons], "k:", marker="s",
                     label="control (random time & side)")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("holding time (s)")
    axes[1].set_ylabel("green P&L per $100 staked (after commission)")
    axes[1].set_title("Trading it: aggressive in and out, real book depth")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jumps", default="3,5", help="jump thresholds in ticks")
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--horizons", default="5,10,30,60,120")
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--max-price", type=float, default=30.0)
    ap.add_argument("--max-spread", type=int, default=3, help="ticks; runners must be this tight before and after the jump")
    ap.add_argument("--min-sweep-vol", type=float, default=20.0)
    ap.add_argument("--min-races", type=int, default=30, help="a rule needs events in at least this many races")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-races", type=int, default=100000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    horizons = [int(x) for x in a.horizons.split(",")]
    kw = dict(jumps=tuple(int(x) for x in a.jumps.split(",")), window_s=a.window_s, horizons=tuple(horizons),
              stake=a.stake, max_price=a.max_price, max_spread=a.max_spread, min_sweep_vol=a.min_sweep_vol)
    paths = list_tapes(a.tapes)[: a.max_races]
    tr, va, te = split_by_date(paths)
    holdout_days = {os.path.basename(p)[:8] for p in va + te}
    print(f"{len(paths)} races: {len(tr)} train, {len(va) + len(te)} holdout (val+test days)", flush=True)

    t0 = time.time()
    os.environ["OMP_NUM_THREADS"] = "1"
    rows = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, res in enumerate(pool.map(_one, [(p, kw) for p in paths], chunksize=4)):
            rows.extend(res)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(paths)} races, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    ev = pd.DataFrame(rows)
    if ev.empty:
        raise SystemExit("no events found")
    ev["split"] = np.where(ev["day"].isin(holdout_days), "holdout", "train")
    ev.to_parquet(os.path.join(a.out, "events.parquet"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    jev = ev[ev["kind"] != "control"]
    print(f"\n{len(jev)} jump events ({jev['race'].nunique()} races), {int((ev['kind'] == 'control').sum())} control "
          f"samples. Median post-jump entry fill for ${a.stake:.0f}: {jev['fill_frac'].median() * 100:.0f}%")
    print("\nEvents per race by phase and threshold:")
    print((jev.groupby(["J", "phase"]).size() / ev["race"].nunique()).round(2).unstack().to_string())

    cont = continuation_table(ev, horizons)
    cont.to_csv(os.path.join(a.out, "continuation.csv"), index=False)
    print("\n=== What happens after a jump (all days; ticks, + = kept going) ===")
    print(cont.round(2).to_string(index=False))

    ctl = ev[ev["kind"] == "control"]
    print("\n=== Control: round trip at random times, random side (the plain cost of trading) ===")
    print(pd.DataFrame([dict(horizon_s=h, **{f"{k}_{c}": v for c in ("follow", "follow_touch")
                                             for k, v in race_t(ctl, f"{c}_{h}").items() if k in ("mean", "t")},
                             events=race_t(ctl, f"follow_{h}")["events"]) for h in horizons]).round(4).to_string(index=False))

    train, hold = ev[ev["split"] == "train"], ev[ev["split"] == "holdout"]
    rt = rules(train, horizons, a.min_races)
    if rt.empty:
        raise SystemExit("not enough train events to score rules")
    rt = rt.sort_values("t", ascending=False)
    top = rt.head(10).copy()
    hold_scores = [score_rule(hold, r) for r in top.to_dict("records")]
    top["holdout_mean"] = [h["mean"] for h in hold_scores]
    top["holdout_t"] = [h["t"] for h in hold_scores]
    top["holdout_races"] = [h["races"] for h in hold_scores]
    rt.to_csv(os.path.join(a.out, "rules_train.csv"), index=False)
    top.to_csv(os.path.join(a.out, "top_rules.csv"), index=False)
    print(f"\n=== Top 10 rules on TRAIN days (green per $1 staked, race-clustered t), with HOLDOUT ===\n"
          f"({len(rt)} rules scored: expect the best of many to look good by chance)")
    print(top[["J", "direction", "kind", "phase", "runners", "horizon_s", "action", "events", "races", "mean", "t",
               "holdout_mean", "holdout_t", "holdout_races"]].round(4).to_string(index=False))

    best = top.iloc[0].to_dict()
    h = hold_scores[0]
    verdict = {
        "rule": {k: best[k] for k in ("J", "direction", "kind", "phase", "runners", "horizon_s", "action")},
        "train": {"mean_per_$1": best["mean"], "t": best["t"], "races": int(best["races"])},
        "holdout": {"mean_per_$1": h["mean"], "t": h["t"], "races": int(h["races"]), "events": int(h["events"])},
        "rules_scored": int(len(rt)),
        "edge": bool(np.isfinite(h["t"]) and h["mean"] > 0 and h["t"] > 2),
    }
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    plot_curves(ev, horizons, os.path.join(a.out, "jump_curves.png"))
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    print("EDGE: the best train rule held up on unseen days (mean > 0, t > 2)" if verdict["edge"] else
          "NO EDGE: the best train rule did not hold up on unseen days")
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
