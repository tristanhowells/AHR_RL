"""Event study: after a sudden price jump, or once a price has moved well away from
where its money traded, does the price keep going or come back, and can you trade it?

Two event types, one method.

1. Jumps. The race deep dives showed that big pre-off moves are rarely smooth
   trends. They are jumps: a large order sweeps a thin book and the price
   moves several ticks within a few seconds (e.g. Rosehill R2, 29 Aug: Fair
   Master 22 -> 19.5 in 3s at T-40s, then 16.5, then back to 18). A jump is
   ``|mid(s) - mid(s - W)| >= J`` ticks over ``--window-s`` (5s), J in
   ``--jumps`` (3, 5); one event per runner per 15s. ``kind`` = "sweep" if at
   least ``--min-sweep-vol`` dollars traded on that runner inside the window,
   else "quote" (the price moved because orders were pulled, not matched).

2. WAP displacement. The session volume-weighted average matched price (WAP)
   is where the money actually agreed the price was. A small sample suggested
   runners that drift well above their WAP come partly back, while steamers
   below it hold. An event is ``|mid - WAP| >= G`` ticks, G in ``--wap-gaps``
   (3, 5), once the runner has ``--wap-min-vol`` dollars matched; one event per
   runner per 60s.

direction: "steam" = price shortened (jump down / below WAP), "drift" = price
lengthened (jump up / above WAP). Follow = bet the move continues (back a
steamer, lay a drifter); fade = bet it comes back (lay a steamer, back a
drifter). So "WAP drift, fade" is: back a runner that has drifted above its
WAP, and green later when it shortens.

All events: only tradeable runners, as in the edge test (both sides present,
spread <= ``--max-spread`` ticks (3), price <= ``--max-price`` (30)); mid =
midpoint of best back / best lay in *tick* space; suspended rows, the in-play
row, and anything within 10s of a scratching (reduction factors shift every
price at once) are excluded.

For every event and horizon H (seconds, or "start" = the scheduled start,
where the RL env stops opening trades) we record:
  * ``cont_H``: further mid move in the event's direction from s to s+H, in
    ticks (positive = kept going, negative = came back);
  * ``follow_H`` / ``fade_H``: green P&L per $1 of stake for an aggressive
    round trip, net of commission. The entry order is decided at s and lands
    one step later (latency). It takes up to ``--stake`` dollars from the
    visible book, no more than 2 ticks through the best price. The exit is
    decided at s+H and lands a step later, sized to green, up to 5 ticks
    through (any unfilled rest is charged at that limit). ``*_touch_H`` is the
    same trade at the best prices only (spread cost without depth cost).
    Exits are clipped to the last pre-off row;
  * ``fill_frac``: share of the entry stake that the visible book could fill.

Control: the same round trip at random moments with a random direction, i.e.
the plain cost of crossing the spread twice.

Verdict (no look-ahead), separately for each event type: every rule
(threshold, direction, kind, phase, top-3 runners or all, horizon,
follow/fade) is scored on the TRAIN days; the best by race-clustered t (events
in >= ``--min-races`` races) is then scored once on the HOLDOUT days (val +
test of the usual chronological split). The top 10 train rules are shown
with their holdout numbers so you can see how much of the train result
survives.

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


def _hkey(h) -> str:
    return str(h)


def study_tape(path: str, jumps=(3, 5), window_s: float = 5.0, horizons=(5, 10, 30, 60, 120, 300, "start"),
               stake: float = 10.0, cooldown_s: float = 15.0, max_spread: int = 3, min_sweep_vol: float = 20.0,
               max_price: float = 30.0, control_every_s: float = 20.0, wap_gaps=(3, 5), wap_min_vol: float = 50.0,
               wap_cooldown_s: float = 60.0, seed: int = 0) -> list[dict]:
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
    t = tape.t_rel
    after0 = np.where(t >= 0)[0]
    s_start = int(after0[0]) if len(after0) else end  # first row at/after the scheduled start

    bt, lt = tape.back_tick[:, :, 0].astype(float), tape.lay_tick[:, :, 0].astype(float)
    valid = (bt >= 0) & (lt >= 0) & tape.active & ok_row[:, None]
    spread = np.where(valid, lt - bt, np.inf)
    mid = np.where(valid & (spread <= max_spread), (bt + lt) / 2, np.nan)
    mid[end + 1:] = np.nan

    # rows near a scratching are excluded for every runner
    bad = np.zeros(T, bool)
    for s in tape.removal_step:
        bad[max(0, s - int(10 / dt)): min(T, s + int(10 / dt) + 1)] = True

    # traded $ (and $ x tick) per runner per row; session WAP in tick space
    R = tape.n_runners
    st = np.clip(tape.trade_step, 0, T - 1)
    vol, vtk = np.zeros((T, R)), np.zeros((T, R))
    np.add.at(vol, (st, tape.trade_runner), tape.trade_vol)
    np.add.at(vtk, (st, tape.trade_runner), tape.trade_vol * tape.trade_tick)
    cvol = np.vstack([np.zeros((1, R)), np.cumsum(vol, 0)])
    csess, ctk = np.cumsum(vol, 0), np.cumsum(vtk, 0)
    wap = np.where(csess >= wap_min_vol, ctk / np.maximum(csess, 1e-9), np.nan)
    matched = tape.tv[end] if end < T else tape.tv[-1]
    rank = np.argsort(np.argsort(-np.nan_to_num(matched)))  # 0 = most traded runner

    rng = np.random.default_rng(abs(hash(race)) % (2**32) + seed)
    rows = []

    def exit_row(s: int, h):
        if h == "start":
            return (s_start, s_start > end) if s < s_start else (None, False)
        s_h = s + int(round(h / dt))
        return min(s_h, end), s_h > end

    def outcomes(s: int, r: int, direction: int, base: dict) -> dict:
        # direction: -1 = price has shortened (steam / below WAP), +1 = lengthened (drift / above WAP)
        rec = dict(base)
        follow_side = "back" if direction < 0 else "lay"
        fade_side = "lay" if direction < 0 else "back"
        s_in = min(s + 1, end)
        f_frac = float("nan")
        for h in horizons:
            k = _hkey(h)
            s_h, clipped = exit_row(s, h)
            names = ("cont", "follow", "fade", "follow_touch", "fade_touch")
            if s_h is None:
                for n in names:
                    rec[f"{n}_{k}"] = float("nan")
                continue
            m_now, m_h = mid[s, r], mid[s_h, r]
            if not np.isfinite(m_h):  # fall back to the last valid mid before s_h
                ix = np.where(np.isfinite(mid[s:s_h + 1, r]))[0]
                m_h = mid[s + ix[-1], r] if len(ix) else np.nan
            rec[f"cont_{k}"] = float(direction * (m_h - m_now))
            s_out = min(s_h + 1, end)
            if s_out <= s_in:
                for n in names[1:]:
                    rec[f"{n}_{k}"] = float("nan")
                continue
            rec[f"follow_{k}"], f_frac, rec[f"follow_touch_{k}"] = _round_trip(tape, r, s_in, s_out, follow_side,
                                                                                stake, comm)
            rec[f"fade_{k}"], _, rec[f"fade_touch_{k}"] = _round_trip(tape, r, s_in, s_out, fade_side, stake, comm)
            rec[f"clipped_{k}"] = bool(clipped)
        rec["fill_frac"] = f_frac
        return rec

    def base(s, r, event, J, size, direction, kind, swept, p_now):
        return dict(race=race, day=race[:8], runner=r, step=s, t_rel=float(t[s]), phase=_phase(float(t[s])),
                    event=event, J=J, jump_ticks=float(size), direction=direction, kind=kind, swept_usd=float(swept),
                    price=float(p_now), spread=float(spread[s, r]), runner_rank=int(rank[r]), top3=bool(rank[r] < 3),
                    comm=comm)

    for r in range(R):
        m = mid[:, r]
        # --- jumps: |mid(s) - mid(s-W)| >= J ticks
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
                rows.append(outcomes(s, r, -1 if d < 0 else 1,
                                     base(s, r, "jump", J, d, "steam" if d < 0 else "drift",
                                          "sweep" if swept >= min_sweep_vol else "quote", swept, p_now)))
        # --- WAP displacement: mid is >= G ticks away from the session WAP
        gap = m - wap[:, r]  # + = price longer than where the money traded (drifted above WAP)
        for G in wap_gaps:
            last_event = -10**9
            wcool = int(round(wap_cooldown_s / dt))
            for s in range(1, end):
                if s - last_event < wcool or bad[s] or not np.isfinite(gap[s]) or abs(gap[s]) < G:
                    continue
                p_now = PRICES[int(round(m[s]))]
                if p_now > max_price:
                    continue
                last_event = s
                rows.append(outcomes(s, r, 1 if gap[s] > 0 else -1,
                                     base(s, r, "wap", G, gap[s], "drift" if gap[s] > 0 else "steam", "wap", 0.0,
                                          p_now)))
        # --- control: random moments, random direction, same filters (no jump in the window)
        ce = int(round(control_every_s / dt))
        for s in range(W + int(rng.integers(0, ce)), end, ce):
            if bad[s] or not np.isfinite(m[s]) or not np.isfinite(m[s - W]) or abs(m[s] - m[s - W]) >= min(jumps):
                continue
            p_now = PRICES[int(round(m[s]))]
            if p_now > max_price:
                continue
            rows.append(outcomes(s, r, int(rng.choice([-1, 1])),
                                 base(s, r, "control", 0, 0.0, "control", "control", 0.0, p_now)))
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
    if col not in df:
        return dict(events=0, races=0, mean=np.nan, race_mean=np.nan, t=np.nan)
    x = df[["race", col]].dropna()
    if not len(x):
        return dict(events=0, races=0, mean=np.nan, race_mean=np.nan, t=np.nan)
    per = x.groupby("race")[col].mean()
    sd = per.std(ddof=1)
    tt = per.mean() / (sd / np.sqrt(len(per))) if len(per) > 1 and sd > 0 else np.nan
    return dict(events=len(x), races=len(per), mean=float(x[col].mean()), race_mean=float(per.mean()), t=float(tt))


def continuation_table(ev: pd.DataFrame, horizons) -> pd.DataFrame:
    rows = []
    for (event, J, direction, kind), g in ev[ev["event"] != "control"].groupby(["event", "J", "direction", "kind"]):
        row = dict(event=event, J=J, direction=direction, kind=kind, events=len(g), races=g["race"].nunique(),
                   size=g["jump_ticks"].abs().mean(), fill=g["fill_frac"].mean())
        for h in horizons:
            c = g[f"cont_{_hkey(h)}"].dropna()
            row[f"cont_{h}"] = c.mean()
            row[f"%kept_{h}"] = (c > 0).mean() * 100
            row[f"%back_{h}"] = (c < 0).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


RULE_KEYS = ("event", "J", "direction", "kind", "phase", "runners", "horizon", "action")


def _select(ev: pd.DataFrame, event, J, direction, kind, phase, runners) -> pd.DataFrame:
    g = ev[(ev["event"] == event) & (ev["J"] == J)]
    if direction != "both":
        g = g[g["direction"] == direction]
    if kind != "any":
        g = g[g["kind"] == kind]
    if phase != "all":
        g = g[g["phase"] == phase]
    if runners == "top3":
        g = g[g["top3"]]
    return g


def rules(ev: pd.DataFrame, horizons, min_races: int, event: str) -> pd.DataFrame:
    """Score every rule of one event type on the given events."""
    out = []
    kinds = ("sweep", "quote", "any") if event == "jump" else ("any",)
    for J in sorted(ev.loc[ev["event"] == event, "J"].unique()):
        for direction in ("steam", "drift", "both"):
            for kind in kinds:
                for phase in ("all",) + tuple(p[2] for p in PHASES):
                    for top in ("all", "top3"):
                        g = _select(ev, event, J, direction, kind, phase, top)
                        if g["race"].nunique() < min_races:
                            continue
                        for h in horizons:
                            for act in ("follow", "fade"):
                                s = race_t(g, f"{act}_{_hkey(h)}")
                                if s["races"] < min_races:
                                    continue
                                out.append(dict(event=event, J=J, direction=direction, kind=kind, phase=phase,
                                                runners=top, horizon=_hkey(h), action=act, **s))
    return pd.DataFrame(out)


def score_rule(ev: pd.DataFrame, rule: dict) -> dict:
    g = _select(ev, rule["event"], rule["J"], rule["direction"], rule["kind"], rule["phase"], rule["runners"])
    return race_t(g, f"{rule['action']}_{rule['horizon']}")


def plot_curves(ev: pd.DataFrame, horizons, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(horizons))
    labels = [f"{h}s" if h != "start" else "sched\nstart" for h in horizons]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, event, title in ((axes[0], "jump", "After a jump"), (axes[1], "wap", "After the price leaves its session WAP")):
        e = ev[ev["event"] == event]
        for (J, direction, kind), g in e.groupby(["J", "direction", "kind"]):
            y = [g[f"cont_{_hkey(h)}"].mean() for h in horizons]
            ls = ":" if kind == "quote" else "-"
            what = f"{kind}, " if event == "jump" else ""
            unit = "tick jump" if event == "jump" else "ticks from WAP"
            ax.plot(x, y, ls, marker="o", label=f">={J} {unit}, {direction} ({what}n={len(g)})")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x, labels)
        ax.set_xlabel("time after the event (\"sched start\": only events before the start, so a different set)")
        ax.set_ylabel("mean further move in the same direction (ticks)\n+ = kept going, - = came back")
        ax.set_title(title)
        ax.legend(fontsize=7)
    ax = axes[2]
    for event, mk in (("jump", "o"), ("wap", "^")):
        e = ev[ev["event"] == event]
        for J, g in e.groupby("J"):
            ax.plot(x, [g[f"follow_{_hkey(h)}"].mean() * 100 for h in horizons], "-", marker=mk,
                    label=f"{event} >={J}: follow")
            ax.plot(x, [g[f"fade_{_hkey(h)}"].mean() * 100 for h in horizons], "--", marker=mk,
                    label=f"{event} >={J}: fade")
    ctl = ev[ev["event"] == "control"]
    if len(ctl):
        ax.plot(x, [ctl[f"follow_{_hkey(h)}"].mean() * 100 for h in horizons], "k:", marker="s",
                label="control (random time & side)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("holding time (\"sched start\": events before the start only)")
    ax.set_ylabel("green P&L per $100 staked (after commission)")
    ax.set_title("Trading it: aggressive in and out, real book depth")
    ax.legend(fontsize=7, ncol=2)
    for a_ in axes:
        a_.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def _verdict_for(event, train, hold, horizons, min_races, out_dir):
    rt = rules(train, horizons, min_races, event)
    if rt.empty:
        print(f"\n[{event}] not enough train events to score rules")
        return None
    rt = rt.sort_values("t", ascending=False)
    top = rt.head(10).copy()
    hs = [score_rule(hold, r) for r in top.to_dict("records")]
    top["holdout_mean"] = [h["mean"] for h in hs]
    top["holdout_t"] = [h["t"] for h in hs]
    top["holdout_races"] = [h["races"] for h in hs]
    rt.to_csv(os.path.join(out_dir, f"rules_train_{event}.csv"), index=False)
    top.to_csv(os.path.join(out_dir, f"top_rules_{event}.csv"), index=False)
    print(f"\n=== [{event}] Top 10 rules on TRAIN days (green per $1 staked, race-clustered t), with HOLDOUT ===\n"
          f"({len(rt)} rules scored: expect the best of many to look good by chance)")
    print(top[list(RULE_KEYS) + ["events", "races", "mean", "t", "holdout_mean", "holdout_t", "holdout_races"]]
          .round(4).to_string(index=False))
    best, h = top.iloc[0].to_dict(), hs[0]
    return {
        "rule": {k: best[k] for k in RULE_KEYS},
        "train": {"mean_per_$1": best["mean"], "t": best["t"], "races": int(best["races"])},
        "holdout": {"mean_per_$1": h["mean"], "t": h["t"], "races": int(h["races"]), "events": int(h["events"])},
        "rules_scored": int(len(rt)),
        "edge": bool(np.isfinite(h["t"]) and h["mean"] > 0 and h["t"] > 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jumps", default="3,5", help="jump thresholds in ticks")
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--wap-gaps", default="3,5", help="ticks between mid and session WAP that count as an event")
    ap.add_argument("--wap-min-vol", type=float, default=50.0, help="$ matched on the runner before its WAP counts")
    ap.add_argument("--wap-cooldown-s", type=float, default=60.0)
    ap.add_argument("--horizons", default="5,10,30,60,120,300,start",
                    help="seconds, and/or 'start' = exit at the scheduled start")
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--max-price", type=float, default=30.0)
    ap.add_argument("--max-spread", type=int, default=3, help="ticks; runners must be this tight")
    ap.add_argument("--min-sweep-vol", type=float, default=20.0)
    ap.add_argument("--min-races", type=int, default=30, help="a rule needs events in at least this many races")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-races", type=int, default=100000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    horizons = [x if x == "start" else int(x) for x in a.horizons.split(",")]
    kw = dict(jumps=tuple(int(x) for x in a.jumps.split(",")), window_s=a.window_s, horizons=tuple(horizons),
              stake=a.stake, max_price=a.max_price, max_spread=a.max_spread, min_sweep_vol=a.min_sweep_vol,
              wap_gaps=tuple(float(x) for x in a.wap_gaps.split(",")), wap_min_vol=a.wap_min_vol,
              wap_cooldown_s=a.wap_cooldown_s)
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
    pd.set_option("display.max_columns", 60)

    nr = ev["race"].nunique()
    for event, what in (("jump", "jump"), ("wap", "WAP displacement")):
        e = ev[ev["event"] == event]
        print(f"\n{len(e)} {what} events ({e['race'].nunique()} races). Median entry fill for ${a.stake:.0f}: "
              f"{e['fill_frac'].median() * 100:.0f}%. Events per race by phase and threshold:")
        print((e.groupby(["J", "phase"]).size() / nr).round(2).unstack().to_string())

    cont = continuation_table(ev, horizons)
    cont.to_csv(os.path.join(a.out, "continuation.csv"), index=False)
    ccols = ["event", "J", "direction", "kind", "events", "races", "size", "fill"]
    for h in horizons:
        ccols += [f"cont_{h}", f"%kept_{h}", f"%back_{h}"]
    print("\n=== What happens next (all days; ticks, + = kept going in the same direction, - = came back) ===")
    print("jump: the price moved >= J ticks within the window.  wap: the price sits >= J ticks from its session WAP "
          "(drift = above it, i.e. longer; steam = below it)")
    print(cont[ccols].round(2).to_string(index=False))

    ctl = ev[ev["event"] == "control"]
    print("\n=== Control: round trip at random times, random side (the plain cost of trading) ===")
    print(pd.DataFrame([dict(horizon=h, **{f"{k}_{c}": v for c in ("follow", "follow_touch")
                                           for k, v in race_t(ctl, f"{c}_{_hkey(h)}").items() if k in ("mean", "t")},
                             events=race_t(ctl, f"follow_{_hkey(h)}")["events"]) for h in horizons])
          .round(4).to_string(index=False))

    train, hold = ev[ev["split"] == "train"], ev[ev["split"] == "holdout"]
    verdict = {e: _verdict_for(e, train, hold, horizons, a.min_races, a.out) for e in ("jump", "wap")}
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    plot_curves(ev, horizons, os.path.join(a.out, "jump_curves.png"))
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    for e, v in verdict.items():
        if v is None:
            continue
        print(f"{e}: " + ("EDGE: the best train rule held up on unseen days (mean > 0, t > 2)" if v["edge"] else
                          "NO EDGE: the best train rule did not hold up on unseen days"))
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
