"""Market study: which runners are tradeable pre-off, which signals carry
information, and how the two interact.

Every ``--every-s`` seconds (10) before the off, every runner with a two-sided
book is sampled. Per sample:

  segments   prob (normalised implied probability = market share, %), rank
             (1 = favourite, by price at that moment), field size, time to the
             scheduled start, time to the actual off (hindsight: known only
             after the race, so it is used to describe, never to trade)
  liquidity  spread (ticks, %), $ at the best prices, $ in the top 3 levels,
             cost of an instant $10 round trip (walk the book in, hedge out),
             the same at the best prices only ("touch"), $ matched per minute
  signals    wom       money waiting to be backed / (back + lay), top 3 levels
             wom_chg30 change of wom over 30s
             wap_sess  mid minus session WAP, ticks (+ = price longer than where
                       its money traded)
             wap_120   the same against the last 2 minutes' WAP
             mom_30, mom_120   price rate of change: mid move over 30s / 120s, ticks
             vol_rate  $ matched on the runner in the last 60s (log1p)
             vol_accel log((last 30s $ + 1) / (previous 30s $ + 1)): volume rate of change
             vol_share runner's share of the market's matched $ in the last 60s,
                       minus its market share (money arriving faster than its price implies)
  outcomes   fwd_H: mid move over H in % of price (100 * log ratio; - = shortened),
             H = 30s, 120s, to the scheduled start, to the off
             back_H / lay_H: green P&L per $1 of a real aggressive round trip
             (back then lay, or lay then back): order lands one step after the
             decision, walks the visible book (2 ticks max) for $10, hedge sized
             to green (5 ticks max), commission on profit. H = 30s, 120s, start.

Analyses (printed, saved as CSV and PNG in --out):
  A  Tradeability by market share, favouritism rank and time to off, and the
     market share x time cross: spread, depth, cost, typical move, move / cost,
     share of time a runner is "tradeable" (spread <= 2 ticks, >= $20 at the
     best prices, price <= 30), and the hit rate needed to break even.
  B  Signal strength: within-race rank correlation (IC) of each signal with
     future moves at each horizon, t-stat clustered by race.
  C  Crossed features: IC of each signal within market share, time-to-start
     and volume-rate segments, and the market share x time cross.
  D  How the signals relate to each other (rank correlation matrix).
  E  Can you trade it? For every signal x horizon x segment cell, trade the top
     and bottom 20% of the signal in the direction its TRAIN-day IC says, and
     score real round trips. Cells are ranked on TRAIN days; the top 15 are
     then scored once on HOLDOUT days.
  F  All signals and segments crossed at once: gradient-boosted models of the
     back and lay round-trip P&L, fitted on TRAIN days; on HOLDOUT days trade
     whenever a predicted P&L clears a threshold, reported by market share.

    python -m ahr_rl.market_study --tapes "data/tapes/*.npz" --out runs/market
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
from .jump_study import _round_trip
from .ladder import PRICES
from .tape import Tape

SIGNALS = ("wom", "wom_chg30", "wap_sess", "wap_120", "mom_30", "mom_120", "vol_rate", "vol_accel", "vol_share")
PROB_BINS = [0, 2, 5, 10, 20, 35, 101]
PROB_LABELS = ["<2%", "2-5%", "5-10%", "10-20%", "20-35%", ">35%"]
RANK_BINS = [0, 1, 2, 3, 4, 6, 9, 99]
RANK_LABELS = ["1 (fav)", "2", "3", "4", "5-6", "7-9", "10+"]
TIME_BINS = [-1e9, -300, -120, -60, 0, 1e9]
TIME_LABELS = ["<T-5m", "T-5m..-2m", "T-2m..-1m", "T-1m..start", "after start"]
OFF_BINS = [-1e9, -600, -300, -120, -60, 0.1]
OFF_LABELS = [">10m", "10-5m", "5-2m", "2-1m", "<1m"]
TRADE_H = ("30", "120", "start")
IC_H = ("30", "120", "start", "off")


def sample_tape(path: str, every_s: float = 10.0, stake: float = 10.0) -> pd.DataFrame | None:
    tape = Tape.load(path)
    race = os.path.basename(path).split(".npz")[0]
    dt, T, R = tape.dt, tape.n_steps, tape.n_runners
    ok = ~tape.suspended.copy()
    if tape.went_in_play and T > 1:
        ok[-1] = False
    if not ok.any():
        return None
    end = int(np.where(ok)[0].max())
    comm = tape.base_rate / 100.0
    t = tape.t_rel.astype(float)
    off_t = float(t[-1]) if tape.went_in_play else np.nan
    after0 = np.where(t >= 0)[0]
    s_start = int(after0[0]) if len(after0) else end
    k = lambda sec: int(round(sec / dt))

    bt, lt = tape.back_tick[:, :, 0].astype(int), tape.lay_tick[:, :, 0].astype(int)
    valid = (bt >= 0) & (lt >= 0) & tape.active & ok[:, None]
    valid[end + 1:] = False
    midt = np.where(valid, (bt + lt) / 2.0, np.nan)  # tick space
    midp = np.where(valid, np.sqrt(PRICES[np.maximum(bt, 0)] * PRICES[np.maximum(lt, 0)]), np.nan)
    ip = np.where(valid, 1.0 / midp, 0.0)
    share = ip / np.maximum(ip.sum(1, keepdims=True), 1e-12) * 100
    bs3, ls3 = tape.back_size[:, :, :3].sum(-1), tape.lay_size[:, :, :3].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        wom = np.where(bs3 + ls3 > 0, bs3 / (bs3 + ls3), np.nan)

    st = np.clip(tape.trade_step, 0, T - 1)
    vol, vtk = np.zeros((T, R)), np.zeros((T, R))
    np.add.at(vol, (st, tape.trade_runner), tape.trade_vol)
    np.add.at(vtk, (st, tape.trade_runner), tape.trade_vol * tape.trade_tick)
    cv, cvt = np.cumsum(vol, 0), np.cumsum(vtk, 0)

    def window(c, s, w):
        return c[s] - (c[s - w] if s - w >= 0 else 0.0)

    bad = np.zeros(T, bool)
    for s in tape.removal_step:
        bad[max(0, s - k(10)): min(T, s + k(10) + 1)] = True

    rows = []
    for s in range(k(30), end, k(every_s)):
        if bad[s]:
            continue
        live = np.where(valid[s])[0]
        if len(live) < 2:
            continue
        order = live[np.argsort(midp[s, live])]
        rank = {int(r): i + 1 for i, r in enumerate(order)}
        mkt_v60 = sum(window(cv[:, r], s, k(60)) for r in live)
        for r in live:
            r = int(r)
            v60 = window(cv[:, r], s, k(60))
            v30a, v30b = window(cv[:, r], s, k(30)), window(cv[:, r], s - k(30), k(30)) if s >= k(30) else 0.0
            wap_s = cvt[s, r] / cv[s, r] if cv[s, r] >= 20 else np.nan
            v120 = window(cv[:, r], s, k(120))
            wap_w = window(cvt[:, r], s, k(120)) / v120 if v120 >= 10 else np.nan
            rec = dict(
                race=race, day=race[:8], runner=r, t=t[s], to_off=(t[s] - off_t) if np.isfinite(off_t) else np.nan,
                field=len(live), rank=rank[r], prob=share[s, r], price=midp[s, r],
                spread=int(lt[s, r] - bt[s, r]), spread_pct=(PRICES[lt[s, r]] / PRICES[bt[s, r]] - 1) * 100,
                best_usd=float(tape.back_size[s, r, 0] + tape.lay_size[s, r, 0]), depth3_usd=float(bs3[s, r] + ls3[s, r]),
                vol_1m=v60,
                wom=wom[s, r],
                wom_chg30=wom[s, r] - wom[s - k(30), r],
                wap_sess=midt[s, r] - wap_s, wap_120=midt[s, r] - wap_w,
                mom_30=midt[s, r] - midt[s - k(30), r],
                mom_120=(midt[s, r] - midt[s - k(120), r]) if s >= k(120) else np.nan,
                vol_rate=np.log1p(v60), vol_accel=np.log((v30a + 1) / (v30b + 1)),
                vol_share=(v60 / mkt_v60 * 100 - share[s, r]) if mkt_v60 > 0 else np.nan,
                comm=comm,
            )
            g, f, touch = _round_trip(tape, r, s, s, "back", stake, comm)  # instant round trip = cost
            rec["cost10"], rec["cost_touch"], rec["fill10"] = -g * 100, -touch * 100, f
            g50, f50, _ = _round_trip(tape, r, s, s, "back", 50.0, comm)
            rec["cost50"], rec["fill50"] = -g50 * 100, f50
            for h in IC_H:
                if h == "start":
                    s2 = s_start if s < s_start else None
                elif h == "off":
                    s2 = end
                else:
                    s2 = min(s + k(int(h)), end)
                if s2 is None or s2 <= s:
                    rec[f"fwd_{h}"] = np.nan
                else:
                    ix = np.where(np.isfinite(midp[s:s2 + 1, r]))[0]
                    rec[f"fwd_{h}"] = 100 * np.log(midp[s + ix[-1], r] / midp[s, r]) if len(ix) else np.nan
                if h in TRADE_H:
                    if s2 is None or min(s2 + 1, end) <= min(s + 1, end):
                        rec[f"back_{h}"] = rec[f"lay_{h}"] = np.nan
                    else:
                        s_in, s_out = min(s + 1, end), min(s2 + 1, end)
                        rec[f"back_{h}"] = _round_trip(tape, r, s_in, s_out, "back", stake, comm)[0]
                        rec[f"lay_{h}"] = _round_trip(tape, r, s_in, s_out, "lay", stake, comm)[0]
            rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def _one(args):
    path, kw = args
    try:
        return sample_tape(path, **kw)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}", flush=True)
        return None


# ------------------------------------------------------------------ statistics
def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    df["prob_b"] = pd.cut(df["prob"], PROB_BINS, labels=PROB_LABELS, right=False)
    df["rank_b"] = pd.cut(df["rank"], RANK_BINS, labels=RANK_LABELS)
    df["time_b"] = pd.cut(df["t"], TIME_BINS, labels=TIME_LABELS, right=False)
    df["off_b"] = pd.cut(df["to_off"], OFF_BINS, labels=OFF_LABELS, right=False)
    q = df["vol_rate"].quantile([1 / 3, 2 / 3]).values
    df["busy_b"] = pd.cut(df["vol_rate"], [-1, q[0], q[1], 1e9], labels=["quiet", "normal", "busy"])
    df["tradeable"] = (df["spread"] <= 2) & (df["best_usd"] >= 20) & (df["price"] <= 30)
    return df


def add_race_ranks(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Within-race percentile ranks centred on 0, so ICs compare runners within a race."""
    g = df.groupby("race")
    for c in cols:
        df[f"z_{c}"] = (g[c].rank(pct=True) - 0.5).astype(np.float32)
    return df


def ic(df: pd.DataFrame, f: str, y: str, min_races: int = 20) -> dict:
    x = df[["race", f"z_{f}", f"z_{y}"]].dropna()
    if x["race"].nunique() < min_races or len(x) < 200:
        return dict(ic=np.nan, t=np.nan, n=len(x), races=x["race"].nunique())
    a, b = x[f"z_{f}"].values, x[f"z_{y}"].values
    a, b = a - a.mean(), b - b.mean()
    r = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
    per = pd.Series(a * b).groupby(x["race"].values).mean()
    t = float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)) + 1e-12))
    return dict(ic=r, t=t, n=len(x), races=len(per))


def race_t(v: pd.Series, race: pd.Series) -> tuple[float, float, int]:
    x = pd.DataFrame({"v": v.values, "race": race.values}).dropna()
    if len(x) < 2:
        return np.nan, np.nan, 0
    per = x.groupby("race")["v"].mean()
    if len(per) < 2 or per.std(ddof=1) == 0:
        return float(x["v"].mean()), np.nan, len(per)
    return float(x["v"].mean()), float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))), len(per)


def tradeability(df: pd.DataFrame, by) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    out = pd.DataFrame({
        "samples": g.size(),
        "price": g["price"].median(),
        "spread_ticks": g["spread"].median(),
        "spread_%": g["spread_pct"].median(),
        "best_$": g["best_usd"].median(),
        "top3_$": g["depth3_usd"].median(),
        "$/min": g["vol_1m"].median(),
        "cost_touch_%": g["cost_touch"].median(),
        "cost_$10_%": g["cost10"].median(),
        "cost_$50_%": g["cost50"].median(),
        "fill_$50_%": g["fill50"].median() * 100,
        "move_120s_%": g["fwd_120"].apply(lambda x: x.abs().median()),
        "move_start_%": g["fwd_start"].apply(lambda x: x.abs().median()),
        "tradeable_%": g["tradeable"].mean() * 100,
    })
    out["move/cost"] = out["move_120s_%"] / out["cost_$10_%"]
    out["hit_rate_needed_%"] = np.clip(50 * (1 + 1 / out["move/cost"]), 0, 999)
    vs = g["vol_1m"].sum()
    out["share_of_$_%"] = vs / vs.sum() * 100
    return out


# ------------------------------------------------------------------ plots
def _heat(ax, piv: pd.DataFrame, title: str, fmt: str = "{:.2f}", cmap="RdBu_r", center=None):
    v = piv.values.astype(float)
    if center is not None:
        lim = np.nanmax(np.abs(v - center)) or 1
        im = ax.imshow(v, cmap=cmap, vmin=center - lim, vmax=center + lim, aspect="auto")
    else:
        im = ax.imshow(v, cmap=cmap, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [str(c) for c in piv.columns], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]), [str(i) for i in piv.index], fontsize=8)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=10)
    return im


def plots(df, out_dir, ic_time: pd.DataFrame, cross_ic: dict, corr: pd.DataFrame):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1: tradeability heatmaps (market share x time to start)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    piv = lambda col, fn: df.pivot_table(index="prob_b", columns="time_b", values=col, aggfunc=fn, observed=True)
    _heat(axes[0], piv("cost10", "median"), "Cost of a $10 round trip (% of stake)", "{:.1f}", "Reds")
    mv = df.assign(am=df["fwd_120"].abs()).pivot_table(index="prob_b", columns="time_b", values="am",
                                                        aggfunc="median", observed=True)
    _heat(axes[1], mv / piv("cost10", "median"), "Typical 2-min move / cost (>1 = moves beat costs)", "{:.2f}",
          "RdYlGn", center=1.0)
    _heat(axes[2], piv("tradeable", "mean") * 100, "% of time tradeable (spread<=2, $20+ at best, price<=30)",
          "{:.0f}", "Greens")
    for ax in axes:
        ax.set_xlabel("time to scheduled start")
        ax.set_ylabel("market share (implied probability)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tradeability.png"), dpi=110)
    plt.close(fig)

    # 2: signal IC by time to start, per horizon
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    for ax, h in zip(axes, ("30", "120", "start")):
        sub = ic_time[ic_time["horizon"] == h]
        for f in SIGNALS:
            s = sub[sub["signal"] == f].set_index("time_b")["ic"].reindex(TIME_LABELS)
            ax.plot(range(len(TIME_LABELS)), s.values, marker="o", label=f)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(TIME_LABELS)), TIME_LABELS, rotation=20)
        ax.set_title(f"Signal IC vs next move ({h}{'s' if h.isdigit() else ''}), by time to start")
        ax.set_ylabel("within-race rank IC (+ = high signal -> price lengthens)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "signal_by_time.png"), dpi=110)
    plt.close(fig)

    # 3: crossed ICs (market share x time) for three key signal/horizon pairs + signal correlations
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax, (key, piv_ic) in zip(axes[:3], cross_ic.items()):
        _heat(ax, piv_ic, f"IC of {key} by market share x time", "{:.2f}", "RdBu_r", center=0.0)
        ax.set_xlabel("time to scheduled start")
        ax.set_ylabel("market share")
    _heat(axes[3], corr, "How the signals relate (rank correlation)", "{:.2f}", "RdBu_r", center=0.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "crossed.png"), dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-s", type=float, default=10.0)
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-races", type=int, default=100000)
    ap.add_argument("--min-races", type=int, default=30)
    ap.add_argument("--no-model", action="store_true", help="skip section F")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    t0 = time.time()

    paths = list_tapes(a.tapes)[: a.max_races]
    tr, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    print(f"{len(paths)} races: {len(tr)} train, {len(va) + len(te)} holdout days' races", flush=True)
    cache = os.path.join(a.out, "samples.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        print(f"loaded {len(df)} cached samples from {cache} (delete it to rebuild)")
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        parts = []
        kw = dict(every_s=a.every_s, stake=a.stake)
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for i, d in enumerate(pool.map(_one, [(p, kw) for p in paths], chunksize=4)):
                if d is not None:
                    parts.append(d)
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(paths)} races, {sum(len(p) for p in parts)} samples, "
                          f"{time.time() - t0:.0f}s", flush=True)
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(cache, index=False)
    df["split"] = np.where(df["day"].isin(hold_days), "holdout", "train")
    df = add_segments(df)
    ys = [f"fwd_{h}" for h in IC_H]
    df = add_race_ranks(df, list(SIGNALS) + ys)
    print(f"{len(df)} runner samples from {df['race'].nunique()} races ({time.time() - t0:.0f}s)\n")

    # ---------------- A: tradeability
    print("=" * 100 + "\nA. TRADEABILITY  (costs and moves in % of stake/price; medians)\n" + "=" * 100)
    for by, name in (("prob_b", "market share (implied probability)"), ("rank_b", "favouritism rank"),
                     ("time_b", "time to scheduled start"), ("off_b", "time to the ACTUAL off (hindsight)")):
        tb = tradeability(df, by)
        tb.to_csv(os.path.join(a.out, f"A_tradeability_{by}.csv"))
        print(f"\n--- by {name} ---")
        print(tb.round(2).to_string())
    for col, title in (("cost10", "cost of a $10 round trip, %"), ("tradeable", "% of time tradeable")):
        p = df.pivot_table(index="prob_b", columns="time_b", values=col, aggfunc="median" if col != "tradeable"
                           else "mean", observed=True)
        p = p * (100 if col == "tradeable" else 1)
        p.to_csv(os.path.join(a.out, f"A_cross_{col}.csv"))
        print(f"\n--- market share x time to start: {title} ---")
        print(p.round(1).to_string())
    mv = df.assign(am=df["fwd_120"].abs()).pivot_table(index="prob_b", columns="time_b", values="am",
                                                        aggfunc="median", observed=True)
    ratio = mv / df.pivot_table(index="prob_b", columns="time_b", values="cost10", aggfunc="median", observed=True)
    print("\n--- market share x time to start: typical 2-min move / cost (> 1 = moves bigger than costs) ---")
    print(ratio.round(2).to_string())

    # ---------------- B: signal strength
    print(f"\n[{time.time() - t0:.0f}s]\n" + "=" * 100 + "\nB. SIGNAL STRENGTH: within-race rank IC with the future move "
          "(+ = high signal -> price lengthens next), t clustered by race\n" + "=" * 100)
    rows = []
    for f in SIGNALS:
        for h in IC_H:
            rows.append(dict(signal=f, horizon=h, **ic(df, f, f"fwd_{h}", a.min_races)))
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(a.out, "B_signal_ic.csv"), index=False)
    print(B.pivot(index="signal", columns="horizon", values="ic").reindex(columns=list(IC_H)).round(3).to_string())
    print("\nt-stats:")
    print(B.pivot(index="signal", columns="horizon", values="t").reindex(columns=list(IC_H)).round(1).to_string())

    # ---------------- C: crossed features
    print(f"\n[{time.time() - t0:.0f}s]\n" + "=" * 100 + "\nC. CROSSED: signal IC within segments\n" + "=" * 100)
    crows = []
    for seg in ("prob_b", "rank_b", "time_b", "busy_b"):
        for val, g in df.groupby(seg, observed=True):
            for f in SIGNALS:
                for h in ("30", "120", "start"):
                    crows.append(dict(segment=seg, value=str(val), signal=f, horizon=h, **ic(g, f, f"fwd_{h}", a.min_races)))
    C = pd.DataFrame(crows)
    C.to_csv(os.path.join(a.out, "C_crossed_ic.csv"), index=False)
    for h in ("30", "120"):
        for seg in ("prob_b", "time_b", "busy_b"):
            p = C[(C["segment"] == seg) & (C["horizon"] == h)].pivot(index="signal", columns="value", values="ic")
            order = {"prob_b": PROB_LABELS, "time_b": TIME_LABELS, "busy_b": ["quiet", "normal", "busy"]}[seg]
            print(f"\n--- IC vs next {h}s move, by {seg} ---")
            print(p.reindex(columns=[c for c in order if c in p.columns]).round(3).to_string())
    ic_time = C[C["segment"] == "time_b"].rename(columns={"value": "time_b"})
    cross_ic = {}
    for f, h in (("wom", "30"), ("wap_sess", "120"), ("mom_30", "30")):
        cells = []
        for (pb, tb), g in df.groupby(["prob_b", "time_b"], observed=True):
            cells.append(dict(prob_b=pb, time_b=tb, **ic(g, f, f"fwd_{h}", a.min_races)))
        piv = pd.DataFrame(cells).pivot(index="prob_b", columns="time_b", values="ic")
        piv = piv.reindex(index=[p for p in PROB_LABELS if p in piv.index], columns=[t for t in TIME_LABELS if t in piv.columns])
        cross_ic[f"{f} -> {h}s"] = piv
        piv.to_csv(os.path.join(a.out, f"C_cross_{f}_{h}.csv"))
        print(f"\n--- IC of {f} vs next {h}s move: market share x time to start ---")
        print(piv.round(3).to_string())

    # ---------------- D: signal relationships
    zc = [f"z_{f}" for f in SIGNALS]
    corr = df[zc].corr()
    corr.index = corr.columns = list(SIGNALS)
    corr.to_csv(os.path.join(a.out, "D_signal_corr.csv"))
    print(f"\n[{time.time() - t0:.0f}s]\n" + "=" * 100 + "\nD. HOW THE SIGNALS RELATE (within-race rank correlation)\n" + "=" * 100)
    print(corr.round(2).to_string())
    seg_rel = df.groupby("prob_b", observed=True)[list(SIGNALS)].median()
    print("\nmedian signal value by market share:")
    print(seg_rel.round(3).to_string())

    # ---------------- E: can you trade it (cells picked on train, scored on holdout)
    print(f"\n[{time.time() - t0:.0f}s]\n" + "=" * 100 + "\nE. CAN YOU TRADE IT? top/bottom 20% of a signal, traded in its TRAIN-day direction, "
          "real round trips ($10)\n" + "=" * 100)
    train, hold = df[df["split"] == "train"], df[df["split"] == "holdout"]
    cells = [("all", "all", None)] + [(seg, str(v), None) for seg in ("prob_b", "time_b") for v in
                                         (PROB_LABELS if seg == "prob_b" else TIME_LABELS)]
    cells += [("prob_b x time_b", f"{pb} | {tb}", (pb, tb)) for pb in PROB_LABELS for tb in TIME_LABELS]

    def cell_rows(d, seg, val, pair):
        if seg == "all":
            return d
        if pair is not None:
            return d[(d["prob_b"] == pair[0]) & (d["time_b"] == pair[1])]
        return d[d[seg].astype(str) == val]

    res = []
    for seg, val, pair in cells:
        gtr, gho = cell_rows(train, seg, val, pair), cell_rows(hold, seg, val, pair)
        if gtr["race"].nunique() < a.min_races:
            continue
        for f in SIGNALS:
            x = gtr[f].dropna()
            if len(x) < 200:
                continue
            lo, hi = x.quantile(0.2), x.quantile(0.8)
            for h in TRADE_H:
                icv = ic(gtr, f, f"fwd_{h}", a.min_races)["ic"]
                if not np.isfinite(icv):
                    continue
                # icv > 0: high signal -> price lengthens -> lay first; low signal -> back first

                def pnl(g):
                    top, bot = g[g[f] >= hi], g[g[f] <= lo]
                    a_ = top[f"lay_{h}"] if icv > 0 else top[f"back_{h}"]
                    b_ = bot[f"back_{h}"] if icv > 0 else bot[f"lay_{h}"]
                    return pd.concat([a_, b_]), pd.concat([top["race"], bot["race"]])

                v, rc = pnl(gtr)
                m, tt, nr = race_t(v, rc)
                res.append(dict(segment=seg, cell=val, signal=f, horizon=h, train_ic=icv, lo=lo, hi=hi,
                                trades=int(v.notna().sum()), races=nr, mean=m, t=tt, _pair=pair))
    E = pd.DataFrame(res).sort_values("t", ascending=False)
    top = E.head(15).copy()
    hm, ht, hn = [], [], []
    for r in top.to_dict("records"):
        g = cell_rows(hold, r["segment"], r["cell"], r["_pair"])
        f, h = r["signal"], r["horizon"]
        topr, bot = g[g[f] >= r["hi"]], g[g[f] <= r["lo"]]
        a_ = topr[f"lay_{h}"] if r["train_ic"] > 0 else topr[f"back_{h}"]
        b_ = bot[f"back_{h}"] if r["train_ic"] > 0 else bot[f"lay_{h}"]
        m, tt, nr = race_t(pd.concat([a_, b_]), pd.concat([topr["race"], bot["race"]]))
        hm.append(m); ht.append(tt); hn.append(nr)
    top["holdout_mean"], top["holdout_t"], top["holdout_races"] = hm, ht, hn
    E.drop(columns="_pair").to_csv(os.path.join(a.out, "E_rules_train.csv"), index=False)
    top.drop(columns="_pair").to_csv(os.path.join(a.out, "E_top_rules.csv"), index=False)
    print(f"({len(E)} cells scored on train; green per $1 staked, t clustered by race)")
    print(top.drop(columns=["_pair", "lo", "hi"]).round(4).to_string(index=False))
    best_e = top.iloc[0]
    edge_e = bool(np.isfinite(best_e["holdout_t"]) and best_e["holdout_mean"] > 0 and best_e["holdout_t"] > 2)

    # ---------------- F: everything crossed in one model
    verdict_f = None
    if not a.no_model:
        from sklearn.ensemble import HistGradientBoostingRegressor

        print(f"\n[{time.time() - t0:.0f}s]\n" + "=" * 100 + "\nF. ALL SIGNALS AND SEGMENTS CROSSED: boosted trees of round-trip P&L, "
              "fitted on TRAIN days, traded on HOLDOUT days\n" + "=" * 100)
        X_cols = list(SIGNALS) + ["prob", "rank", "field", "t", "price", "spread", "spread_pct", "best_usd",
                                  "depth3_usd", "cost10"]
        rng = np.random.default_rng(0)
        rows_f = []
        for h in TRADE_H:
            preds = {}
            for side in ("back", "lay"):
                y = f"{side}_{h}"
                d = train.dropna(subset=[y])
                if len(d) > 400_000:
                    d = d.iloc[rng.choice(len(d), 400_000, replace=False)]
                m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                  min_samples_leaf=200, l2_regularization=1.0, random_state=0)
                m.fit(d[X_cols].values.astype(np.float32), d[y].values)
                preds[side] = m.predict(hold[X_cols].values.astype(np.float32))
            best_side = np.where(preds["back"] >= preds["lay"], "back", "lay")
            best_pred = np.maximum(preds["back"], preds["lay"])
            realised = np.where(best_side == "back", hold[f"back_{h}"].values, hold[f"lay_{h}"].values)
            for thr in (0.0, 0.01, 0.02, 0.05):
                sel = (best_pred > thr) & np.isfinite(realised)
                m_, t_, nr = race_t(pd.Series(realised[sel]), hold["race"][sel])
                rows_f.append(dict(horizon=h, threshold=thr, trades=int(sel.sum()), races=nr, mean=m_, t=t_,
                                   **{f"trades_{p}": int((sel & (hold["prob_b"].values == p)).sum()) for p in PROB_LABELS}))
                if thr == 0.0:
                    for p in PROB_LABELS:
                        sp = sel & (hold["prob_b"].values == p)
                        mp_, tp_, np_ = race_t(pd.Series(realised[sp]), hold["race"][sp])
                        rows_f.append(dict(horizon=h, threshold=f"0 | {p}", trades=int(sp.sum()), races=np_,
                                           mean=mp_, t=tp_))
        F = pd.DataFrame(rows_f)
        F.to_csv(os.path.join(a.out, "F_model_holdout.csv"), index=False)
        print("HOLDOUT results (trade when the better of predicted back/lay P&L > threshold; per $1 staked):")
        print(F.round(4).to_string(index=False))
        main_rows = F[F["threshold"].apply(lambda v: not isinstance(v, str))]
        best_f = main_rows.sort_values("t", ascending=False).iloc[0]
        verdict_f = dict(horizon=best_f["horizon"], threshold=float(best_f["threshold"]),
                         mean_per_1=float(best_f["mean"]), t=float(best_f["t"]), trades=int(best_f["trades"]),
                         note="threshold/horizon picked after seeing holdout: optimistic")

    plots(df, a.out, ic_time, cross_ic, corr)
    verdict = {
        "best_segment_rule": {**{k: best_e[k] for k in ("segment", "cell", "signal", "horizon")},
                              "train_mean": float(best_e["mean"]), "train_t": float(best_e["t"]),
                              "holdout_mean": float(best_e["holdout_mean"]), "holdout_t": float(best_e["holdout_t"]),
                              "edge": edge_e},
        "model": verdict_f,
    }
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=str)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=str))
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
