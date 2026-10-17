"""Cross-matching: how much would Betfair's virtual bets improve the prices our simulator sees?

The tapes are recorded from the raw ladder (EX_ALL_OFFERS: atb / atl), which holds real
offers only. Live, Betfair also cross-matches: it creates *virtual* offers on a runner from
the best offers on all the OTHER runners, because together they cover every outcome.

  virtual price to BACK runner i  (from the backers waiting on every other runner j, i.e.
  their best available-to-lay price q_j):   P = 1 / (1 - sum_{j != i} 1/q_j)
      size (backer stake) = min_j(size_j * q_j) / P
  virtual price to LAY runner i   (from the layers on every other runner, best
  available-to-back price b_j):             P = 1 / (1 - sum_{j != i} 1/b_j)
      size (backer stake) = min_j(size_j * b_j) / P

Only when every other active runner has an offer on that side (else the outcomes are not
covered) and the sum is < 1. Prices are rounded to a valid tick in the direction that is
WORSE for us (back: down, lay: up), so the improvement measured here is a lower bound.
The simulator does not include any of this, so crossing the spread in it is slightly harsher
than live.

  A  how often the virtual best beats the raw best (by ticks), adds size at the same price,
     or does nothing; and the round-trip cost (best lay / best back - 1) raw vs with virtual.
     Every 5s in the last 10 min, all active runners; by liquidity (P11/P13 cut: $ at best
     back + lay, 10s mean), favourite rank, field size, time to the off, book overround.
  B  check on the formula: real trades that printed strictly INSIDE the raw spread (no raw
     offer there half a second earlier). How many are explained by the virtual prices?
  C  what it would change for our own orders: the average improvement in the price paid
     crossing the spread, in ticks and %, for liquid runners, vs P12's ~3.9% round trip.

    python -m ahr_rl.crossmatch_study --tapes "data/tapes/*.npz" --out runs/crossmatch
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
from .blackbox import best_usd_trailing
from .env import list_tapes, split_by_date
from .ladder import N_TICKS, PRICES
from .tape import Tape

MAX_PRICE = float(PRICES[-1])


def _virtual(tick: np.ndarray, size: np.ndarray, live: np.ndarray, round_down: bool):
    """Virtual best offer on each runner from the best offers on all the others.

    tick, size: [T, R] best level on the side the virtual offer is built from (-1 = empty)
    live:       [T, R] runners that must be covered (active, not removed)
    -> (virtual tick [T, R] (-1 = none), virtual size [T, R] in backer stake)"""
    T, R = tick.shape
    has = tick >= 0
    p = np.where(has, PRICES[np.clip(tick, 0, N_TICKS - 1)], np.nan)
    inv = np.where(live & has, 1.0 / p, 0.0)
    covered_all = (has | ~live).all(1)  # every live runner has an offer on this side
    others = inv.sum(1, keepdims=True) - inv
    with np.errstate(divide="ignore", invalid="ignore"):
        vp = np.where(others < 1.0, 1.0 / (1.0 - others), np.inf)
    ok = covered_all[:, None] & live & (live.sum(1, keepdims=True) >= 2) & np.isfinite(vp) & (vp <= MAX_PRICE)
    vp_c = np.clip(np.where(ok, vp, PRICES[0]), PRICES[0], MAX_PRICE)
    if round_down:  # back: the best valid price not above the virtual one
        vt = np.searchsorted(PRICES, vp_c + 1e-9, side="right") - 1
    else:  # lay: the best valid price not below it
        vt = np.searchsorted(PRICES, vp_c - 1e-9, side="left")
    vt = np.where(ok, np.clip(vt, 0, N_TICKS - 1), -1)
    # payout each other runner's offer can cover; the thinnest leg limits the virtual size
    pay = np.where(live & has, size * np.nan_to_num(p), np.inf)
    o = np.argsort(pay, 1)
    m1 = np.take_along_axis(pay, o[:, :1], 1)
    m2 = np.take_along_axis(pay, o[:, 1:2], 1) if R > 1 else np.full((T, 1), np.inf)
    min_other = np.where(np.arange(R)[None] == o[:, :1], m2, m1)
    vsize = np.where(vt >= 0, min_other / np.where(vt >= 0, PRICES[np.clip(vt, 0, N_TICKS - 1)], 1.0), 0.0)
    return vt.astype(np.int32), vsize


def virtual_books(t: Tape):
    """(vb, vb_size, vl, vl_size) [T, R]: virtual best price to back / lay each runner."""
    live = t.active.copy()
    for s, r in zip(t.removal_step, t.removal_runner):
        live[s:, r] = False
    vb, vbs = _virtual(t.lay_tick[:, :, 0].astype(np.int32), t.lay_size[:, :, 0].astype(np.float64), live, True)
    vl, vls = _virtual(t.back_tick[:, :, 0].astype(np.int32), t.back_size[:, :, 0].astype(np.float64), live, False)
    return vb, vbs, vl, vls, live


def sample_tape(path: str, every_s: float = 5.0, last_s: float = 600.0):
    t = Tape.load(path)
    race = os.path.basename(path).split(".npz")[0]
    vb, vbs, vl, vls, live = virtual_books(t)
    rb, rl = t.back_tick[:, :, 0].astype(np.int32), t.lay_tick[:, :, 0].astype(np.int32)
    rbs, rls = t.back_size[:, :, 0].astype(np.float64), t.lay_size[:, :, 0].astype(np.float64)
    liq = best_usd_trailing(t, 10.0)
    ok_t = ~t.suspended & (t.t_rel >= -last_s) & (t.t_rel <= 0)
    rows = []
    k = max(1, int(round(every_s / t.dt)))
    for s in np.where(ok_t)[0][::k]:
        r_ok = live[s] & (rb[s] >= 0) & (rl[s] >= 0)
        if r_ok.sum() < 2:
            continue
        mid = np.where(r_ok, np.sqrt(PRICES[np.maximum(rb[s], 0)] * PRICES[np.maximum(rl[s], 0)]), np.inf)
        rank = mid.argsort().argsort() + 1
        bo = float((1.0 / PRICES[rb[s][r_ok]]).sum())  # back book %, raw
        lo = float((1.0 / PRICES[rl[s][r_ok]]).sum())
        for r in np.where(r_ok)[0]:
            rows.append((race, race[:8], float(t.t_rel[s]), int(r), int(rank[r]), int(r_ok.sum()), float(liq[s, r]),
                         int(rb[s, r]), int(rl[s, r]), float(rbs[s, r]), float(rls[s, r]),
                         int(vb[s, r]), int(vl[s, r]), float(vbs[s, r]), float(vls[s, r]), bo, lo))
    cols = ["race", "day", "t", "runner", "rank", "field", "best_usd", "rb", "rl", "rb_size", "rl_size",
            "vb", "vl", "vb_size", "vl_size", "back_book", "lay_book"]
    df = pd.DataFrame(rows, columns=cols) if rows else None

    # B: trades in (s-1, s] against the book at s-1
    tr = []
    sel = (t.trade_step >= 1) & ok_t[np.clip(t.trade_step, 0, t.n_steps - 1)]
    for s, r, tk, v in zip(t.trade_step[sel], t.trade_runner[sel], t.trade_tick[sel], t.trade_vol[sel]):
        p = s - 1
        if rb[p, r] < 0 or rl[p, r] < 0:
            continue
        tr.append((race, int(r), float(liq[p, r]), int(tk), float(v), int(rb[p, r]), int(rl[p, r]),
                   int(vb[p, r]), int(vl[p, r])))
    tdf = pd.DataFrame(tr, columns=["race", "runner", "best_usd", "tick", "vol", "rb", "rl", "vb", "vl"]) if tr else None
    return df, tdf


def _one(args):
    path, kw = args
    try:
        return sample_tape(path, **kw)
    except Exception as e:  # a corrupt tape should not kill the run
        print(f"skip {os.path.basename(path)}: {e}", flush=True)
        return None, None


def add_effective(df: pd.DataFrame) -> pd.DataFrame:
    """Best of raw and virtual on each side, and the derived improvements."""
    df = df.copy()
    has_vb, has_vl = df["vb"] >= 0, df["vl"] >= 0
    df["eb"] = np.where(has_vb, np.maximum(df["rb"], df["vb"]), df["rb"])
    df["el"] = np.where(has_vl, np.minimum(df["rl"], df["vl"]), df["rl"])
    df["back_gain_ticks"] = df["eb"] - df["rb"]  # better price to back (higher)
    df["lay_gain_ticks"] = df["rl"] - df["el"]  # better price to lay (lower)
    df["back_better"] = df["back_gain_ticks"] > 0
    df["lay_better"] = df["lay_gain_ticks"] > 0
    df["back_adds_size"] = has_vb & (df["vb"] == df["rb"])
    df["lay_adds_size"] = has_vl & (df["vl"] == df["rl"])
    df["any_better"] = df["back_better"] | df["lay_better"]
    # crossed: a virtual offer through the opposite raw best (an arbitrage the real engine would have matched)
    df["crossed"] = (has_vb & (df["vb"] >= df["rl"])) | (has_vl & (df["vl"] <= df["rb"]))
    pb, pl = PRICES[df["rb"].to_numpy()], PRICES[df["rl"].to_numpy()]
    eb = PRICES[np.minimum(df["eb"].to_numpy(), df["el"].to_numpy())]  # keep a crossed book at zero spread
    df["cost_raw_%"] = (pl / pb - 1) * 100
    df["cost_eff_%"] = (PRICES[df["el"].to_numpy()] / eb - 1) * 100
    df["cost_saved_%"] = df["cost_raw_%"] - df["cost_eff_%"]
    df["spread_raw"] = df["rl"] - df["rb"]
    df["spread_eff"] = np.maximum(df["el"] - df["eb"], 0)
    return df


def summary(df: pd.DataFrame, by: str | None) -> pd.DataFrame:
    g = df.groupby(by, observed=True) if by else [("all", df)]
    rows = []
    for k, d in g:
        rows.append({by or "slice": k, "snapshots": len(d), "races": d["race"].nunique(),
                     "virtual_back_%": (d["vb"] >= 0).mean() * 100, "virtual_lay_%": (d["vl"] >= 0).mean() * 100,
                     "back_better_%": d["back_better"].mean() * 100, "lay_better_%": d["lay_better"].mean() * 100,
                     "either_better_%": d["any_better"].mean() * 100,
                     "adds_size_%": (d["back_adds_size"] | d["lay_adds_size"]).mean() * 100,
                     "gain_ticks_when_better": float(np.nanmean(np.r_[d.loc[d["back_better"], "back_gain_ticks"],
                                                                      d.loc[d["lay_better"], "lay_gain_ticks"]]))
                     if d["any_better"].any() else np.nan,
                     "spread_raw": d["spread_raw"].mean(), "spread_eff": d["spread_eff"].mean(),
                     "cost_raw_%": d["cost_raw_%"].mean(), "cost_eff_%": d["cost_eff_%"].mean(),
                     "cost_saved_%": d["cost_saved_%"].mean(), "crossed_%": d["crossed"].mean() * 100})
    return pd.DataFrame(rows)


def trade_check(tdf: pd.DataFrame) -> dict:
    """Section B: volume printed strictly inside the raw spread, and how much the virtual prices explain."""
    inside = (tdf["tick"] > tdf["rb"]) & (tdf["tick"] < tdf["rl"])
    by_vb = (tdf["vb"] >= 0) & (tdf["tick"] <= tdf["vb"])  # a backer could have got this price virtually
    by_vl = (tdf["vl"] >= 0) & (tdf["tick"] >= tdf["vl"])
    v = tdf["vol"]
    vin = v[inside].sum()
    return dict(trades=int(len(tdf)), volume=float(v.sum()), inside_spread_vol_pct=float(vin / max(v.sum(), 1e-9) * 100),
                inside_explained_by_virtual_pct=float(v[inside & (by_vb | by_vl)].sum() / max(vin, 1e-9) * 100))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-s", type=float, default=5.0)
    ap.add_argument("--last-s", type=float, default=600.0, help="window before the scheduled start")
    ap.add_argument("--liq-cut", default="auto",
                    help='liquid runner cut in $ at best back + lay (10s mean); "auto" = 80th pct, last 5 min, TRAIN days')
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    race_filter.add_args(ap)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    tag = race_filter.tag(a)
    logf = open(os.path.join(a.out, f"log{tag}.txt"), "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")

    paths = list_tapes(a.tapes)
    tr_paths, _, _ = split_by_date(paths)
    train_days = {os.path.basename(p)[:8] for p in tr_paths}
    paths = race_filter.filter_paths(paths, a)
    cache, tcache = os.path.join(a.out, f"samples{tag}.parquet"), os.path.join(a.out, f"trades{tag}.parquet")
    if os.path.exists(cache) and os.path.exists(tcache):
        df, tdf = pd.read_parquet(cache), pd.read_parquet(tcache)
        log(f"loaded {cache}")
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        parts, tparts = [], []
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            jobs = [(p, dict(every_s=a.every_s, last_s=a.last_s)) for p in paths]
            for i, (d, td) in enumerate(pool.map(_one, jobs, chunksize=2)):
                if d is not None:
                    parts.append(d)
                if td is not None:
                    tparts.append(td)
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(paths)} races, {time.time() - t0:.0f}s", flush=True)
        if not parts:
            raise SystemExit("no samples")
        df, tdf = pd.concat(parts, ignore_index=True), pd.concat(tparts, ignore_index=True)
        df.to_parquet(cache, index=False)
        tdf.to_parquet(tcache, index=False)

    df = add_effective(df)
    if a.liq_cut == "auto":
        m = df["day"].isin(train_days) & (df["t"] >= -300)
        cut = float(np.nanquantile(df.loc[m if m.any() else df.index, "best_usd"], 0.8))
    else:
        cut = float(a.liq_cut)
    df["liquidity"] = np.where(df["best_usd"] >= cut, f">= ${cut:.0f} (liquid)", f"< ${cut:.0f}")
    df["fav_rank"] = pd.cut(df["rank"], [0, 1, 3, 6, 99], labels=["fav", "2-3", "4-6", "7+"])
    df["field_size"] = pd.cut(df["field"], [0, 6, 9, 12, 99], labels=["<=6", "7-9", "10-12", "13+"])
    df["time"] = pd.cut(df["t"], [-1e9, -300, -120, -30, 0.1], labels=["10-5m", "5-2m", "2m-30s", "last 30s"])
    df["back_book"] = pd.cut(df["back_book"] * 100, [0, 98, 100, 102, 105, 1e9],
                             labels=["<98%", "98-100%", "100-102%", "102-105%", ">105%"])
    log(f"{len(df)} runner snapshots, {df['race'].nunique()} races, every {a.every_s:.0f}s in the last "
        f"{a.last_s / 60:.0f} min; liquid = >= ${cut:.2f} at best back + lay (10s mean)  [{time.time() - t0:.0f}s]")
    log("prices rounded against us, so every improvement below is a lower bound\n")

    log("=== A. Virtual (cross-matched) best offers vs the raw ladder ===")
    tabs = {}
    df["liq_x_rank"] = df["liquidity"].str.replace(r" \(liquid\)", "", regex=True) + ", " + df["fav_rank"].astype(str)
    for by in (None, "liquidity", "fav_rank", "liq_x_rank", "field_size", "time", "back_book"):
        tb = summary(df, by)
        tabs[by or "all"] = tb
        tb.to_csv(os.path.join(a.out, f"A_{by or 'all'}{tag}.csv"), index=False)
        log(f"\n{'all runners' if by is None else 'by ' + by.replace('liq_x_rank', 'liquidity x favourite rank')}:")
        log(tb.round(3).to_string(index=False))

    log("\n=== B. Check: trades that printed strictly inside the raw spread ===")
    tdf = tdf.copy()
    tdf["liquid"] = tdf["best_usd"] >= cut
    b = {"all": trade_check(tdf), "liquid": trade_check(tdf[tdf["liquid"]]), "thin": trade_check(tdf[~tdf["liquid"]])}
    for k, v in b.items():
        log(f"{k:7s}: {v['trades']} trades, {v['inside_spread_vol_pct']:.1f}% of volume inside the raw spread; of that, "
            f"{v['inside_explained_by_virtual_pct']:.0f}% at a price the virtual offers make available")
    log("(inside-spread prints can also come from orders placed and hit within the same half-second)")

    liq = df[df["liquidity"].str.contains("liquid")]
    c = dict(liquid_snapshots=int(len(liq)),
             either_side_better_pct=float(liq["any_better"].mean() * 100),
             mean_cost_raw_pct=float(liq["cost_raw_%"].mean()), mean_cost_eff_pct=float(liq["cost_eff_%"].mean()),
             mean_cost_saved_pct=float(liq["cost_saved_%"].mean()),
             mean_spread_raw_ticks=float(liq["spread_raw"].mean()), mean_spread_eff_ticks=float(liq["spread_eff"].mean()))
    log("\n=== C. What it would change for our orders (liquid runners) ===")
    log(f"either side better than the raw best in {c['either_side_better_pct']:.1f}% of snapshots; spread "
        f"{c['mean_spread_raw_ticks']:.2f} -> {c['mean_spread_eff_ticks']:.2f} ticks; round-trip cost "
        f"{c['mean_cost_raw_pct']:.2f}% -> {c['mean_cost_eff_pct']:.2f}% (saves {c['mean_cost_saved_pct']:.2f} pp)")
    material = c["mean_cost_saved_pct"] >= 0.25
    log("MATERIAL: worth adding cross-matching to the simulator and re-scoring the crossing strategies" if material else
        "NOT MATERIAL: cross-matching barely changes the cost of crossing the spread; the simulator's raw ladder is "
        "a fair (slightly harsh) stand-in")
    verdict = dict(liq_cut=cut, overall=tabs["all"].iloc[0].to_dict(), trades=b, liquid=c, material=material,
                   note="prices rounded against us: improvements are lower bounds")
    json.dump(verdict, open(os.path.join(a.out, f"verdict{tag}.json"), "w"), indent=1, default=float)
    log(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
