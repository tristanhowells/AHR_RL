"""P16: signal-conditioned passive entry. Rest a back only when a signal says "shortening next".

P15: resting k ticks better than the best price, then greening at BSP, costs only ~0.2% per attempt
on liquid runners (vs ~3.9% for crossing in and out) - but adverse selection eats the whole price
improvement: backs fill mostly when the runner is about to drift. A signal that predicts the next
move could screen those fills out. P12's signals have a small but real IC (~0.07); judged against
a ~0.2% hurdle instead of ~3.9%, they might now be enough.

Reuses P15's simulated orders (orders.parquet: every 20s in the last 10 min, every runner, back /
lay at k = 0..3 ticks, fills from the simulator's own passive rules + cross-matching, greened at
BSP) and attaches live signals computed at the same decision times (no look-ahead):

  sig_book_W  the book's imbalance over W s, allocated to the runner by its share (P12)
  sig_lag_W   how far the runner is behind its share of the others' move (P12)
  mom_W       the runner's own move over W s (P12: short-term reversal)
  wom         weight of money at the top 3 levels, back side share
  combo       linear blend of the above, fitted on TRAIN days to the next 30s move

Each signal is oriented on TRAIN days so that high = the runner is about to SHORTEN (a back
should rest there; a lay should rest at the low end).

  A  predictive power: within-race rank IC with the next 30s move, TRAIN / HOLDOUT, all / liquid
  B  does the signal shrink adverse selection? Liquid runners, k = 1, hold 60s, by signal
     quintile (cut-offs from TRAIN): fill rate, EV if all filled, EV given a fill, adverse
     selection, EV per attempt. A useful signal: backs in the top quintile keep more of their
     price improvement than the bottom one.
  C  rule search: signal x side x tail x k x hold x liquidity x threshold (top 20 / 10 / 5%) x rank,
     picked on TRAIN days (race-level t), scored once on HOLDOUT; "edge" = holdout mean > 0,
     t > 2, >= 20 races and still > 0 without its best race. A pass is a hypothesis for a
     pre-registered forward test (like P3c), not a result.

    python -m ahr_rl.signal_entry_study --tapes "data/tapes/*.npz" --orders runs/passive_entry/orders.parquet \\
        --out runs/signal_entry
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

from . import race_filter
from .env import list_tapes, split_by_date
from .passive_entry_study import HOLDS, add_values, rt
from .passive_entry_study import sample_tape as sample_orders
from .rebalance_study import _prob_grid
from .tape import Tape

RAW = ("sig_book_15", "sig_book_30", "sig_lag_15", "sig_lag_30", "mom_30", "wom")
SIGNALS = RAW + ("combo",)
THRESH = (0.20, 0.10, 0.05)


def signal_rows(path: str, every_s: float = 20.0, last_s: float = 600.0):
    """Signals at P15's decision steps (same grid), plus the next-30s move (for orienting / IC only)."""
    t = Tape.load(path)
    if not t.went_in_play:
        return None
    race = os.path.basename(path).split(".npz")[0]
    p, valid, end, _, _ = _prob_grid(t)
    dt = t.dt
    k = lambda sec: int(round(sec / dt))
    bs3, ls3 = t.back_size[:, :, :3].sum(-1), t.lay_size[:, :, :3].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        wom = np.where(bs3 + ls3 > 0, bs3 / (bs3 + ls3), np.nan)
    dec = np.where((t.t_rel >= -last_s) & (t.t_rel <= -5.0) & ~t.suspended)[0][::max(1, k(every_s))]
    rows = []
    for s in dec:
        live = np.where(np.isfinite(p[s]))[0]
        if len(live) < 3:
            continue
        B = np.nansum(p[s, live])
        ch = {}
        for W in (15, 30):
            s0 = s - k(W)
            if s0 < 0:
                ch[W] = None
                continue
            d = p[s] - p[s0]
            ch[W] = (d, float(np.nansum(d[live])))
        s2 = min(s + k(30), end)
        for r in live:
            q = p[s, r] / B
            rec = dict(race=race, runner=int(r), t=np.float32(t.t_rel[s]), wom=float(wom[s, r]),
                       fwd_30=float(100 * np.log(p[s2, r] / p[s, r])) if np.isfinite(p[s2, r]) and s2 > s else np.nan)
            for W in (15, 30):
                if ch[W] is None or not np.isfinite(ch[W][0][r]):
                    rec[f"sig_book_{W}"] = rec[f"sig_lag_{W}"] = rec[f"mom_{W}"] = np.nan
                    continue
                d, dB = ch[W]
                own = d[r]
                rec[f"sig_book_{W}"] = -dB * q * 100
                rec[f"sig_lag_{W}"] = (-(dB - own) * q - own) * 100
                rec[f"mom_{W}"] = own * 100
            rows.append(rec)
    if not rows:
        return None
    d = pd.DataFrame(rows).drop(columns=["mom_15"])
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    return d


def _one(args):
    fn, path, kw = args
    try:
        return fn(path, **kw)
    except Exception as e:
        print(f"skip {os.path.basename(path)}: {e}", flush=True)
        return None


def _pool(fn, paths, kw, workers, t0, label):
    os.environ["OMP_NUM_THREADS"] = "1"
    parts = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        for i, d in enumerate(pool.map(_one, [(fn, p, kw) for p in paths], chunksize=2)):
            if d is not None:
                parts.append(d)
            if (i + 1) % 200 == 0:
                print(f"  {label}: {i + 1}/{len(paths)} races, {time.time() - t0:.0f}s", flush=True)
    if not parts:
        raise SystemExit(f"no {label}")
    return pd.concat(parts, ignore_index=True)


def race_ic(x, y, race) -> float:
    d = pd.DataFrame({"x": x, "y": y, "race": race}).dropna()
    ics = d.groupby("race").apply(lambda g: g["x"].rank().corr(g["y"].rank()) if len(g) >= 4 else np.nan,
                                  include_groups=False).dropna()
    return float(ics.mean()) if len(ics) else np.nan


def orient(sig: pd.DataFrame, train: np.ndarray) -> dict:
    """z-score each raw signal on TRAIN rows, sign it so high = shortening next, and fit the combo."""
    info = {}
    tr = sig[train]
    for c in RAW:
        mu, sd = float(tr[c].mean()), float(tr[c].std()) or 1.0
        ic = race_ic(tr[c], tr["fwd_30"], tr["race"])
        sign = 1.0 if (np.isnan(ic) or ic >= 0) else -1.0
        sig[f"o_{c}"] = sign * (sig[c] - mu) / sd
        info[c] = dict(train_ic=ic, sign=sign)
    X = sig.loc[train, [f"o_{c}" for c in RAW]].fillna(0.0).to_numpy()
    y = sig.loc[train, "fwd_30"].to_numpy()
    m = np.isfinite(y)
    w = np.linalg.lstsq(np.c_[X[m], np.ones(m.sum())], y[m], rcond=None)[0]
    sig["o_combo"] = sig[[f"o_{c}" for c in RAW]].fillna(0.0).to_numpy() @ w[:-1]
    info["combo"] = dict(weights={c: float(v) for c, v in zip(RAW, w[:-1])})
    return info


def quintile_table(df: pd.DataFrame, sig: str, side: str, k: int = 1, hold: int = 60) -> pd.DataFrame:
    g = df[(df["side"] == side) & (df["k"] == k) & df["liquid"]]
    rows = []
    for q, d in g.groupby("q_" + sig, observed=True):
        w = d[f"f{hold}"]
        rows.append({"signal": sig, "side": side, "quintile": q, "attempts": len(d), "races": d["race"].nunique(),
                     f"fill{hold}_%": w.mean() * 100, "ev_if_all_filled_%": d["ev_fill"].mean(),
                     "ev_given_fill_%": float(np.average(d["ev_fill"], weights=w)) if w.sum() > 0 else np.nan,
                     "ev_attempt_%": d[f"ev_attempt_{hold}"].mean()})
    out = pd.DataFrame(rows)
    if len(out):
        out["adverse_sel_pp"] = out["ev_given_fill_%"] - out["ev_if_all_filled_%"]
    return out


def rule_search(df: pd.DataFrame, train: pd.Series, cuts: dict, min_races: int, top: int = 15) -> pd.DataFrame:
    res = []
    # both tails for both sides: the TRAIN orientation (from all runners) can differ within a slice
    for sig, side, tail, k, h, lq, th, fr in itertools.product(SIGNALS, ("back", "lay"), ("shorten", "drift"),
                                                               (0, 1, 2, 3), (60, 120), (False, True), THRESH,
                                                               (None, "fav", "2-3")):
        if k == 0 and h != 60:
            continue  # crossing fills at once
        hi, lo = cuts[(sig, lq, th)]
        o = df["o_" + sig]
        m = (df["side"] == side) & (df["k"] == k) & ((o >= hi) if tail == "shorten" else (o <= lo))
        if lq:
            m &= df["liquid"]
        if fr is not None:
            m &= df["fav_rank"] == fr
        col = f"ev_attempt_{h}" if k > 0 else "ev_fill"
        g = df.loc[m & train]
        if g["race"].nunique() < min_races:
            continue
        r_ = rt(g[col], g["race"])
        res.append(dict(signal=sig, side=side, when=tail, k=k, hold=h, liquid=lq, top=f"{int(th * 100)}%", fav_rank=fr or "all",
                        train_mean=r_["mean"], train_t=r_["t"], train_races=r_["races"], _m=m, _col=col))
    res = sorted(res, key=lambda x: -np.nan_to_num(x["train_t"], nan=-99))[:top]
    out = []
    for r in res:
        g = df.loc[r["_m"] & ~train]
        ht = rt(g[r["_col"]], g["race"])
        per = g.groupby("race")[r["_col"]].mean()
        drop = float(per.drop(per.idxmax()).mean()) if len(per) > 1 else np.nan
        out.append({k: v for k, v in r.items() if not k.startswith("_")} |
                   dict(holdout_mean=ht["mean"], holdout_t=ht["t"], holdout_races=ht["races"], holdout_drop_best=drop))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orders", default=None, help="P15's orders.parquet (else it is simulated here, ~20 min)")
    ap.add_argument("--every-s", type=float, default=20.0, help="must match the P15 run")
    ap.add_argument("--last-s", type=float, default=600.0, help="must match the P15 run")
    ap.add_argument("--liq-cut", default="auto")
    ap.add_argument("--min-races", type=int, default=100)
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
    _, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    paths = race_filter.filter_paths(paths, a)
    kw = dict(every_s=a.every_s, last_s=a.last_s)
    if a.orders and os.path.exists(a.orders):
        orders = pd.read_parquet(a.orders)
        log(f"loaded {len(orders)} orders from {a.orders}")
    else:
        orders = _pool(sample_orders, paths, kw, a.workers, t0, "orders")
        orders.to_parquet(os.path.join(a.out, f"orders{tag}.parquet"), index=False)
    keep = {os.path.basename(p).split(".npz")[0] for p in paths}
    orders = orders[orders["race"].isin(keep)]
    scache = os.path.join(a.out, f"signals{tag}.parquet")
    if os.path.exists(scache):
        sig = pd.read_parquet(scache)
    else:
        sig = _pool(signal_rows, paths, kw, a.workers, t0, "signals")
        sig.to_parquet(scache, index=False)
    sig["day"] = sig["race"].str[:8]
    train_s = ~sig["day"].isin(hold_days).to_numpy()
    info = orient(sig, train_s)

    df = add_values(orders.copy())
    df["t"] = df["t"].astype(np.float32)
    sig["t"] = sig["t"].astype(np.float32)
    df = df.merge(sig[["race", "runner", "t"] + [f"o_{c}" for c in SIGNALS]], on=["race", "runner", "t"], how="inner")
    train = ~df["day"].isin(hold_days)
    cut = (float(np.nanquantile(df.loc[train & (df["k"] == 0) & (df["t"] >= -300), "best_usd"], 0.8))
           if a.liq_cut == "auto" else float(a.liq_cut))
    df["liquid"] = df["best_usd"] >= cut
    log(f"{len(df)} orders with signals, {df['race'].nunique()} races; liquid = >= ${cut:.2f}  [{time.time() - t0:.0f}s]")
    log("signals oriented on TRAIN days: high = the runner is about to SHORTEN (good for a resting back)\n")

    log("=== A. Predictive power: within-race rank IC with the next 30s move (raw signal, before orienting) ===")
    sig_liq = sig.merge(df.loc[df["k"] == 0, ["race", "runner", "t", "liquid"]].drop_duplicates(),
                        on=["race", "runner", "t"], how="left")
    rows = []
    for c in SIGNALS:
        col = c if c in RAW else "o_combo"
        r = {"signal": c, "orientation": "+" if c == "combo" or info[c]["sign"] > 0 else "-"}
        for nm, m in (("train", train_s), ("holdout", ~train_s)):
            d = sig_liq[m]
            r[f"{nm}_ic"] = race_ic(d[col], d["fwd_30"], d["race"])
            dl = d[d["liquid"] == True]  # noqa: E712
            r[f"{nm}_ic_liquid"] = race_ic(dl[col], dl["fwd_30"], dl["race"])
        rows.append(r)
    ta = pd.DataFrame(rows)
    ta.to_csv(os.path.join(a.out, f"A_ic{tag}.csv"), index=False)
    log(ta.round(4).to_string(index=False))
    log(f"combo weights (oriented, TRAIN): {json.dumps({k: round(v, 4) for k, v in info['combo']['weights'].items()})}")

    # cut-offs from TRAIN rows: quintiles (B) and the top / bottom thresholds (C), all and liquid runners
    cuts, base = {}, df[(df["k"] == 0) & (df["side"] == "back")]
    for c in SIGNALS:
        o = df["o_" + c]
        edges = np.unique(np.nanquantile(base.loc[train & base["liquid"], "o_" + c], [0.2, 0.4, 0.6, 0.8]))
        labels = [f"Q{i + 1}" for i in range(len(edges) + 1)]  # ties (signal often exactly 0) merge quintiles
        labels[0], labels[-1] = labels[0] + " drift", labels[-1] + " shorten"
        df["q_" + c] = pd.cut(o, [-np.inf, *edges, np.inf], labels=labels)
        for lq in (False, True):
            b = base.loc[train & (base["liquid"] if lq else True), "o_" + c]
            for th in THRESH:
                cuts[(c, lq, th)] = (float(np.nanquantile(b, 1 - th)), float(np.nanquantile(b, th)))

    log("\n=== B. Does the signal shrink adverse selection? Liquid runners, k = 1, hold 60s, by signal quintile ===")
    log("(Q5 = signal says shorten: good for backs; Q1 = drift: good for lays)")
    tb = pd.concat([quintile_table(df, c, s) for c in SIGNALS for s in ("back", "lay")], ignore_index=True)
    tb.to_csv(os.path.join(a.out, f"B_quintiles{tag}.csv"), index=False)
    for c in SIGNALS:
        log(f"\n{c}:")
        log(tb[tb["signal"] == c].drop(columns="signal").round(3).to_string(index=False))

    log(f"\n=== C. Rule search (signal x side x k x hold x liquidity x threshold x rank): TRAIN (>= {a.min_races} "
        f"races) -> HOLDOUT ===")
    rs = rule_search(df, train, cuts, a.min_races)
    rs.to_csv(os.path.join(a.out, f"C_rules{tag}.csv"), index=False)
    log(rs.round(3).to_string(index=False))
    best = rs.iloc[0].to_dict() if len(rs) else {}
    edge = bool(best) and best["holdout_mean"] > 0 and (best["holdout_t"] or 0) > 2 and \
        best["holdout_races"] >= 20 and best["holdout_drop_best"] > 0
    log("EDGE: the best TRAIN rule holds up on HOLDOUT -> pre-register it and forward-test it like P3c" if edge else
        "NO EDGE: the best TRAIN rule does not hold up on HOLDOUT")
    verdict = dict(liq_cut=cut, edge=edge, best_rule=best, orientation=info,
                   note="rule search over many slices: a pass is a hypothesis for a forward test")
    json.dump(verdict, open(os.path.join(a.out, f"verdict{tag}.json"), "w"), indent=1, default=float)
    log(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
