"""P10: drop "all green before the off". Bets that run into the race and to the result.

Objective B (net positive after settlement, summed over a day) only beats the green
objectives if some bet has positive expected value on its own: a day is the sum of
its races. This looks for such bets in two places our earlier studies didn't reach.

  0  BASELINE  backing / laying every runner at BSP, by price band, after commission
               (the favourite-longshot bias).

  1  IN-PLAY RESTING ORDERS. Placed before the off with persistence (kept in play), so
     the in-play bet delay doesn't apply to placing them:
       lay @ L     L = 1.01 ... 2.0 on a runner priced well above L before the off:
                   fills only if it trades that short in running ("looks like winning")
       back @ m x  m = 2 ... 20 x its last pre-off price (max 1000): fills only if it
                   drifts that far in running ("looks beaten")
     Filled if at least the order's stake ($10 back stake / the lay stake for a $10
     liability) trades beyond its price in play ('through'; 'touch' = at the price,
     optimistic). The win rate when filled is what matters, and it already includes
     being filled by better-informed in-play players. P&L per $1 at risk (back stake /
     lay liability), commission on winnings. Grid: side x level x pre-off price band.

  2  WHAT EARLIER RACES AT THE MEETING REVEAL. Bets at BSP, informed only by races at
     the same meeting that have already been run:
       draw      did inside (or outside) barriers beat their BSP chances earlier today?
                 -> back runners drawn on the favoured side
       jockey    the jockey's wins beyond BSP chances earlier today ("hot hand")
       trainer   the same for the trainer
       fav       did favourites beat their BSP chances earlier today? -> back / lay the
                 favourite
     Each is tested as an information measure (does it predict who beats BSP?
     race-clustered t) and as a betting rule (back the top fifth / lay the bottom fifth
     of the signal at BSP).

Rules are picked on TRAIN days by t and scored once on HOLDOUT days. "edge" = holdout
mean > 0, t > 2, >= 20 races, AND a positive stressed mean on all days: for lays the
win rate is replaced by its 95% upper bound (a rare winner can be missing from the
sample entirely), for backs by its 95% lower bound (a few lucky long-shot winners can
make a rule look good), and it must stay positive without its single luckiest race.
The best rule of each angle is also shown day by day.

    python -m ahr_rl.day_study --recordings "<drive>/recordings" --catalogues "<drive>/catalogues" --out runs/day_study
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .catalogue import index_catalogues, load_catalogue
from .env import split_by_date
from .ladder import PRICES, price_to_tick
from .stream import MarketCache, read_recording

LAY_LEVELS = (1.01, 1.02, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0)
BACK_MULTS = (2, 3, 5, 10, 20)
BANDS = [1.0, 3.0, 6.0, 12.0, 25.0, 1001.0]
BAND_LABELS = ["<=3", "3-6", "6-12", "12-25", "25+"]


# ----------------------------------------------------------------- parsing
def _num(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def race_rows(path: str, cat_path: str | None) -> list[dict]:
    rec = read_recording(path)
    tr = rec.trailer or {}
    winners = tr.get("winners") or []
    if not rec.messages or len(winners) != 1:
        return []
    winner = int(winners[0])
    bsp = {int(k): _num(v) for k, v in (tr.get("bsp") or {}).items()}
    cache = MarketCache(rec.market_id)
    last_pre, comm = None, None
    went_ip, complete = False, False
    ip_min, ip_max, ip_vol = {}, {}, {}  # ip_vol: sid -> {price: matched $ (single-counted)}
    start = rec.market_start_ms
    for i, m in enumerate(rec.messages):
        cache.apply_mcm(m)
        comm = cache.market_base_rate or comm
        trades = cache.drain_trades()
        if cache.in_play:
            went_ip = True
            for sid, price, vol in trades:
                ip_min[sid] = min(ip_min.get(sid, 1e9), price)
                ip_max[sid] = max(ip_max.get(sid, 0.0), price)
                pv = ip_vol.setdefault(sid, {})
                pv[price] = pv.get(price, 0.0) + vol / 2.0  # stream trd counts both sides
        elif went_ip:
            complete = complete or cache.status in ("SUSPENDED", "CLOSED")
        elif cache.status == "OPEN" and ((m["pt"] - start) / 1000.0 >= -60 or i % 50 == 0):  # last pre-off book
            snap = {}
            for r in cache.active_runners():
                bb, _ = r.best_back()
                bl, _ = r.best_lay()
                if bb > 0 and bl > 0:
                    snap[r.selection_id] = (bb, bl)
            if len(snap) >= 2:
                last_pre = snap
        if went_ip and cache.status in ("SUSPENDED", "CLOSED"):
            complete = True
    if not went_ip or last_pre is None:
        return []
    base = os.path.basename(path).split(".ndjson")[0]
    venue = re.sub(r"^\d{8}_\d{4}_", "", base).rsplit("_", 2)[0]
    meta, field_draws = {}, []
    if cat_path:
        try:
            cat = load_catalogue(cat_path)
            venue = (cat.get("event") or {}).get("venue") or venue
            for r in cat.get("runners", []):
                md = r.get("metadata") or {}
                meta[int(r["selectionId"])] = dict(draw=_num(md.get("STALL_DRAW")), jockey=md.get("JOCKEY_NAME"),
                                                   trainer=md.get("TRAINER_NAME"))
        except Exception:
            pass
    sids = [s for s in last_pre if bsp.get(s, 0) > 1.0]
    if len(sids) < 2 or winner not in last_pre:
        return []
    mids = {s: (last_pre[s][0] * last_pre[s][1]) ** 0.5 for s in sids}
    order = sorted(sids, key=lambda s: mids[s])
    draws = [meta.get(s, {}).get("draw", np.nan) for s in sids]
    draw_rank = pd.Series(draws, index=sids).rank(method="average")
    n_draw = int(np.isfinite(draws).sum())
    inv = sum(1.0 / bsp[s] for s in sids)
    rows = []
    for s in sids:
        d = draw_rank.get(s)
        pv = ip_vol.get(s, {})
        pr = np.array(list(pv.keys()), float)
        vv = np.array(list(pv.values()), float)
        fills = {}
        for L in LAY_LEVELS:  # $ matched at or below / strictly below the lay price
            tk = price_to_tick(L)
            fills[f"lay_tch_{L:g}"] = float(vv[pr <= L + 1e-9].sum())
            fills[f"lay_thr_{L:g}"] = float(vv[pr <= PRICES[max(tk - 1, 0)] + 1e-9].sum())
        for mlt in BACK_MULTS:  # back price = first tick at or above mult x the last pre-off lay price
            B = min(last_pre[s][1] * mlt, 1000.0)
            tk = price_to_tick(B)
            tk = tk + 1 if PRICES[tk] < B - 1e-9 else tk
            tk = min(tk, len(PRICES) - 1)
            fills[f"back_px_{mlt}"] = float(PRICES[tk])
            fills[f"back_tch_{mlt}"] = float(vv[pr >= PRICES[tk] - 1e-9].sum())
            fills[f"back_thr_{mlt}"] = float(vv[pr >= PRICES[min(tk + 1, len(PRICES) - 1)] - 1e-9].sum())
        rows.append(dict(**fills,
            race=base, day=base[:8], venue=venue, start_ms=rec.market_start_ms, sid=s, won=int(s == winner),
            comm=(comm or 8.0) / 100.0, bsp=bsp[s], p_bsp=(1.0 / bsp[s]) / inv, rank=order.index(s) + 1,
            field=len(sids), pre_back=last_pre[s][0], pre_lay=last_pre[s][1], mid=mids[s],
            ip_min=ip_min.get(s, np.nan), ip_max=ip_max.get(s, np.nan), complete=complete,
            draw_rel=((d - 1) / max(n_draw - 1, 1)) if (d is not None and np.isfinite(d) and n_draw >= 2) else np.nan,
            jockey=meta.get(s, {}).get("jockey"), trainer=meta.get(s, {}).get("trainer")))
    return rows


def _one(args):
    try:
        return race_rows(*args)
    except Exception as e:
        print(f"  skip {os.path.basename(args[0])}: {e}", flush=True)
        return []


# ----------------------------------------------------------------- helpers
def back_ret(price, won, comm):
    return np.where(won == 1, (price - 1) * (1 - comm), -1.0)


def lay_ret_liab(price, won, comm):
    """Per $1 of liability: lose -> stake (1 / (price - 1)) less commission; win -> -1."""
    return np.where(won == 1, -1.0, (1 - comm) / (price - 1))


def race_t(v, race) -> dict:
    x = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna()
    if len(x) < 2:
        return dict(n=len(x), races=x["race"].nunique(), mean=np.nan, t=np.nan)
    per = x.groupby("race")["v"].sum()  # per-race P&L of the rule (sum of its bets)
    sd = per.std(ddof=1)
    return dict(n=len(x), races=len(per), mean=float(x["v"].mean()),
                t=float(per.mean() / (sd / np.sqrt(len(per)))) if len(per) > 1 and sd > 0 else np.nan)


def stressed(price, won, comm, side: str) -> float:
    """Mean P&L per bet with the win rate moved to its 95% bound against the rule."""
    from scipy.stats import beta

    n, x = len(won), int(np.sum(won))
    if n == 0:
        return np.nan
    if side == "lay":
        p = float(beta.ppf(0.95, x + 1, n - x)) if x < n else 1.0
        return float(p * -1.0 + (1 - p) * np.mean((1 - comm) / (price - 1)))
    p = float(beta.ppf(0.05, x, n - x + 1)) if x > 0 else 0.0
    return float(p * np.mean((price - 1) * (1 - comm)) - (1 - p))


def band(price):
    return pd.cut(price, BANDS, labels=BAND_LABELS)


# ----------------------------------------------------------------- 1. in-play resting orders
STAKE = 10.0  # $ at risk per order: back stake, lay liability


def inplay_bets(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (runner, order): side, level, filled ('through' / 'touch': at least the
    order's stake traded beyond / at its price in play), P&L per $1 at risk."""
    d = df[df["complete"]].copy()
    d["band"] = band(d["mid"])
    out = []
    for L in LAY_LEVELS:
        e = d[d["pre_back"] >= L * 1.25]  # rests in play, can't fill before the off
        need = STAKE / (L - 1)  # lay stake for a $10 liability
        pnl = lay_ret_liab(np.full(len(e), L), e["won"].to_numpy(), e["comm"].to_numpy())
        out.append(pd.DataFrame(dict(race=e["race"], day=e["day"], band=e["band"], side="lay", level=f"{L:g}",
                                     price=L, won=e["won"], comm=e["comm"], through=e[f"lay_thr_{L:g}"] >= need,
                                     touch=e[f"lay_tch_{L:g}"] >= need, pnl=pnl)))
    for mlt in BACK_MULTS:
        Bp = d[f"back_px_{mlt}"].to_numpy()
        pnl = back_ret(Bp, d["won"].to_numpy(), d["comm"].to_numpy())
        out.append(pd.DataFrame(dict(race=d["race"], day=d["day"], band=d["band"], side="back", level=f"{mlt}x",
                                     price=Bp, won=d["won"], comm=d["comm"], through=d[f"back_thr_{mlt}"] >= STAKE,
                                     touch=d[f"back_tch_{mlt}"] >= STAKE, pnl=pnl)))
    return pd.concat(out, ignore_index=True)


def rule_table(bets: pd.DataFrame, keys, hold_days, fill_col: str | None) -> pd.DataFrame:
    rows = []
    for k, g in bets.groupby(keys, observed=True):
        f = g[g[fill_col]] if fill_col else g
        if len(f) == 0:
            continue
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        rec.update(placed=len(g), fill_pct=len(f) / len(g) * 100, win_pct=f["won"].mean() * 100,
                   avg_price=f["price"].mean())
        for sp, m in (("train", ~f["day"].isin(hold_days)), ("holdout", f["day"].isin(hold_days))):
            rt = race_t(f.loc[m, "pnl"], f.loc[m, "race"])
            rec.update({f"{sp}_n": rt["n"], f"{sp}_races": rt["races"], f"{sp}_mean": rt["mean"], f"{sp}_t": rt["t"]})
        rec["all_mean"] = f["pnl"].mean()
        rec["stressed_mean"] = stressed(f["price"].to_numpy(), f["won"].to_numpy(), f["comm"].to_numpy(),
                                        rec.get("side", "back"))
        per = f.groupby("race")["pnl"].sum()
        rec["drop_best_mean"] = float((per.sum() - per.max()) / max(len(f) - (f["race"] == per.idxmax()).sum(), 1)) \
            if len(per) > 1 else np.nan  # without its single luckiest race
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- 2. meeting dynamics
def meeting_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Signals for each runner from races already run at the same meeting (venue + day)."""
    d = df.copy()
    d["surprise"] = d["won"] - d["p_bsp"]
    d["meeting"] = d["day"] + "_" + d["venue"].astype(str)
    races = d.groupby("race").agg(meeting=("meeting", "first"), start=("start_ms", "first")).reset_index()
    races["order"] = races.groupby("meeting")["start"].rank(method="first")
    d = d.merge(races[["race", "order"]], on="race")
    # race-level draw surprise: did inside (draw_rel 0) runners beat the market?
    d["_ds"] = d["surprise"] * (0.5 - d["draw_rel"])
    rs = d.groupby("race").agg(meeting=("meeting", "first"), order=("order", "first"), ds=("_ds", "sum"),
                               fav_s=("surprise", lambda s: s[d.loc[s.index, "rank"] == 1].sum())).reset_index()
    rs = rs.sort_values(["meeting", "order"])
    rs["draw_prior"] = rs.groupby("meeting")["ds"].transform(lambda x: x.shift().cumsum())
    rs["fav_prior"] = rs.groupby("meeting")["fav_s"].transform(lambda x: x.shift().cumsum())
    rs["n_prior"] = rs.groupby("meeting").cumcount()
    d = d.merge(rs[["race", "draw_prior", "fav_prior", "n_prior"]], on="race")
    d["sig_draw"] = np.where(d["n_prior"] >= 2, d["draw_prior"] * (0.5 - d["draw_rel"]), np.nan)
    d["sig_fav"] = np.where((d["n_prior"] >= 2) & (d["rank"] == 1), d["fav_prior"], np.nan)
    d = d.sort_values(["meeting", "order"])
    for who in ("jockey", "trainer"):
        key = d["meeting"] + "|" + d[who].fillna("?").astype(str)
        # surprise of this person's rides in EARLIER races at the meeting (not this one)
        per_race = d.assign(_k=key).groupby(["_k", "order"])["surprise"].sum().rename("s").reset_index()
        per_race["prior"] = per_race.groupby("_k")["s"].transform(lambda x: x.shift().cumsum()).fillna(0.0)
        d = d.assign(_k=key).merge(per_race[["_k", "order", "prior"]], on=["_k", "order"], how="left")
        d[f"sig_{who}"] = np.where(d[who].notna() & (d["n_prior"] >= 1), d["prior"], np.nan)
        d = d.drop(columns=["_k", "prior"])
    return d


def info_test(d: pd.DataFrame, sig: str) -> dict:
    """Does the signal predict who beats BSP? per-race sum of signal x surprise, t across races."""
    x = d[np.isfinite(d[sig]) & (d[sig] != 0)]
    if x["race"].nunique() < 5:
        return dict(signal=sig, runners=len(x), races=x["race"].nunique(), corr=np.nan, t=np.nan)
    per = (x[sig] * x["surprise"]).groupby(x["race"]).sum()
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if per.std(ddof=1) > 0 else np.nan
    return dict(signal=sig, runners=len(x), races=len(per), corr=float(np.corrcoef(x[sig], x["surprise"])[0, 1]),
                t=float(t))


def signal_bets(d: pd.DataFrame, hold_days) -> pd.DataFrame:
    """Back the top fifth / lay the bottom fifth of each signal at BSP (cut-offs from TRAIN days)."""
    out = []
    for sig in ("sig_draw", "sig_jockey", "sig_trainer", "sig_fav"):
        x = d[np.isfinite(d[sig]) & (d[sig] != 0)]
        tr = x[~x["day"].isin(hold_days)]
        if len(tr) < 20:
            continue
        hi, lo = np.quantile(tr[sig], 0.8), np.quantile(tr[sig], 0.2)
        for side, sel in (("back", x[sig] >= hi), ("lay", x[sig] <= lo)):
            g = x[sel]
            pnl = back_ret(g["bsp"], g["won"], g["comm"]) if side == "back" else lay_ret_liab(g["bsp"], g["won"], g["comm"])
            out.append(pd.DataFrame(dict(race=g["race"], day=g["day"], signal=sig, side=side, price=g["bsp"],
                                         won=g["won"], comm=g["comm"], pnl=pnl, all=True)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ----------------------------------------------------------------- main
def _pick(R: pd.DataFrame, min_races: int):
    R = R[(R["train_races"] >= min_races)].sort_values("train_t", ascending=False)
    if R.empty:
        return None, False
    b = R.iloc[0]
    edge = bool(b["holdout_mean"] > 0 and (b["holdout_t"] or 0) > 2 and b["holdout_races"] >= 20
                and b["stressed_mean"] > 0 and b["drop_best_mean"] > 0)
    return b, edge


def by_day(bets: pd.DataFrame, rule: dict, keys, fill_col) -> pd.DataFrame:
    m = np.ones(len(bets), bool)
    for k in keys:
        m &= (bets[k] == rule[k]).to_numpy()
    f = bets[m & (bets[fill_col].to_numpy() if fill_col else True)]
    return f.groupby("day")["pnl"].agg(bets="size", pnl="sum").reset_index()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", required=True, help="folder of *.ndjson.gz recordings (they include in-play)")
    ap.add_argument("--catalogues", default=None, help="folder of catalogue *.json (barrier, jockey, trainer)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--min-races", type=int, default=30)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    t0 = time.time()
    cache = os.path.join(a.out, "runners.parquet")
    paths = sorted(glob.glob(os.path.join(a.recordings, "**", "*.ndjson.gz"), recursive=True))
    tr, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        print(f"loaded {cache}")
    else:
        cat_idx = index_catalogues(a.catalogues) if a.catalogues else {}
        jobs = []
        for p in paths:
            m = re.search(r"_(\d)_(\d+)\.ndjson", os.path.basename(p))
            jobs.append((p, cat_idx.get(f"{m.group(1)}.{m.group(2)}") if m else None))
        rows = []
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for i, r in enumerate(pool.map(_one, jobs, chunksize=4)):
                rows.extend(r)
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(jobs)} recordings ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        df.to_parquet(cache, index=False)
    df["split"] = np.where(df["day"].isin(hold_days), "holdout", "train")
    nr = df["race"].nunique()
    print(f"\n{len(df)} runners, {nr} races, {df['day'].nunique()} days; in-play window complete in "
          f"{df.groupby('race')['complete'].first().mean() * 100:.0f}% of races; barrier known for "
          f"{np.isfinite(df['draw_rel']).mean() * 100:.0f}% of runners  [{time.time() - t0:.0f}s]")

    # ---------------- 0
    print("\n=== 0. Baseline: every runner at BSP, by BSP band (per $1 at risk, after commission) ===")
    df["bsp_band"] = band(df["bsp"])
    rows = []
    for b, g in df.groupby("bsp_band", observed=True):
        bk = back_ret(g["bsp"], g["won"], g["comm"])
        ly = lay_ret_liab(g["bsp"], g["won"], g["comm"])
        rows.append(dict(band=b, runners=len(g), win_pct=g["won"].mean() * 100, bsp_implied_pct=g["p_bsp"].mean() * 100,
                         back_roi_pct=bk.mean() * 100, back_t=race_t(bk, g["race"])["t"],
                         lay_roi_pct=ly.mean() * 100, lay_t=race_t(ly, g["race"])["t"]))
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # ---------------- 1
    print("\n=== 1. In-play resting orders (placed before the off, kept in play) ===")
    bets = inplay_bets(df)
    keys = ["side", "level", "band"]
    R1 = rule_table(bets, keys, hold_days, "through")
    R1.to_csv(os.path.join(a.out, "inplay_rules.csv"), index=False)
    T1 = rule_table(bets, keys, hold_days, "touch")
    T1.to_csv(os.path.join(a.out, "inplay_rules_touch.csv"), index=False)
    for side in ("lay", "back"):
        s = R1[R1["side"] == side]
        if s.empty:
            continue
        print(f"\n{side.upper()}: mean P&L per $1 at risk when filled, all days ('through' fills); "
              f"rows = level, columns = pre-off price band")
        print(s.pivot(index="level", columns="band", values="all_mean").round(3).to_string())
        print(f"{side.upper()}: win % when filled (break-even "
              f"{'below' if side == 'lay' else 'above'} roughly {'(1-c)/L' if side == 'lay' else '1/price'})")
        print(s.pivot(index="level", columns="band", values="win_pct").round(1).to_string())
    best1, edge1 = _pick(R1, a.min_races)
    cols = keys + ["placed", "fill_pct", "win_pct", "avg_price", "train_races", "train_mean", "train_t",
                   "holdout_races", "holdout_mean", "holdout_t", "stressed_mean", "drop_best_mean"]
    print("\ntop 10 in-play rules on TRAIN days, with HOLDOUT and the stress test:")
    print(R1[R1["train_races"] >= a.min_races].sort_values("train_t", ascending=False)[cols].head(10).round(3)
          .to_string(index=False))

    # ---------------- 2
    print("\n=== 2. What earlier races at the meeting reveal (bets at BSP) ===")
    d = meeting_signals(df)
    info = pd.DataFrame([info_test(d, s) for s in ("sig_draw", "sig_jockey", "sig_trainer", "sig_fav")])
    print("does the signal predict who beats BSP? (t across races; > 2 = yes)")
    print(info.round(3).to_string(index=False))
    sb = signal_bets(d, hold_days)
    R2 = rule_table(sb, ["signal", "side"], hold_days, None) if len(sb) else pd.DataFrame()
    if len(R2):
        R2.to_csv(os.path.join(a.out, "meeting_rules.csv"), index=False)
        print("\nbetting rules (top / bottom fifth of the signal at BSP):")
        print(R2[["signal", "side", "placed", "win_pct", "avg_price", "train_races", "train_mean", "train_t",
                  "holdout_races", "holdout_mean", "holdout_t", "stressed_mean", "drop_best_mean"]].round(3)
              .to_string(index=False))
    best2, edge2 = _pick(R2, a.min_races) if len(R2) else (None, False)

    # ---------------- day view + verdict
    print("\n=== Day by day (HOLDOUT days) for the best rule of each angle ===")
    for name, best, bt, k, fc in (("in-play", best1, bets, keys, "through"), ("meeting", best2, sb, ["signal", "side"], None)):
        if best is None:
            continue
        dd = by_day(bt[bt["day"].isin(hold_days)], best.to_dict(), k, fc)
        print(f"{name}: {dict((x, best[x]) for x in k)} -> {len(dd)} days, {(dd['pnl'] > 0).mean() * 100:.0f}% of days "
              f"up, mean {dd['pnl'].mean():+.3f} per day per $1 bet unit" if len(dd) else f"{name}: no holdout bets")
    verdict = dict(inplay_best={k: best1[k] for k in cols} if best1 is not None else None, inplay_edge=edge1,
                   meeting_best=best2[["signal", "side", "train_t", "holdout_mean", "holdout_t", "stressed_mean"]].to_dict()
                   if best2 is not None else None, meeting_edge=edge2)
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=str)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=str))
    print("EDGE found -> forward-test it like P3c before any money" if (edge1 or edge2) else
          "NO EDGE: neither in-play resting orders nor meeting information beats the market on unseen days")
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
