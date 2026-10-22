"""P15: enter better than best available. Resting orders inside / at the spread instead of crossing it.

Crossing the spread costs ~3.9% round trip even on liquid runners (P12, P14). Instead of taking
the best price, offer a better one and wait: back k ticks ABOVE the best price available to back
(k = 1 is inside the spread when it is wider than a tick, else it joins the backers' queue at the
best lay price), or lay k ticks BELOW the best price available to lay. k = 0 = crossing now (the
baseline). How readily do these fill, and are the fills any good?

Fill model = the simulator's passive model (exchange.py), per order, $10 at risk (back: stake,
lay: liability), live one step after the decision, cancelled after --hold-s or at the off:
  * queue ahead = the size already shown at our price; volume traded AT our price burns it,
    then fills us; the queue can never be longer than what is still shown there
  * volume traded THROUGH our price fills us directly (they would have hit us first)
  * the book crossing our price fills us (a better-for-them offer appeared on our level)
  * cross-matching: Betfair also matches our resting back on runner i against the backers
    waiting on every other runner once 1/P + sum_j 1/q_j <= 1, i.e. when the virtual back price
    built from the others (P14, rounded against us) reaches P. Same for lays.

Value: every fill is greened up at BSP (P3b's exit; BSP is fair, so this is also the expected value
of holding): per $ at risk, commission on a positive green only,
    back at P: P/BSP - 1;   lay at P: (1 - P/BSP) / (P - 1)
This separates two things:
    price improvement  EV at P if EVERY order filled  vs crossing now
    adverse selection  EV at P over the orders that DID fill vs all of them: resting orders tend
                       to fill when the price is about to move against them (P10 in-play)

  A  by side and offset: fill rate by 30 / 60 / 120s, time to fill, what filled us (trades,
     book crossing, cross-matching), EV if always filled, EV given a fill, adverse selection,
     EV per attempt vs crossing now.
  B  does price, favourite rank, spread, queue position, liquidity or time to the off matter?
  C  rule search over side x offset x hold x liquidity x price x rank: picked on TRAIN days,
     scored once on HOLDOUT days (means and t are per race, so a busy race counts once;
     drop-the-best-race check). "edge" = holdout
     mean > 0, t > 2, >= 20 races and still > 0 without its best race.

    python -m ahr_rl.passive_entry_study --tapes "data/tapes/*.npz" --out runs/passive_entry
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
from .blackbox import best_usd_trailing
from .crossmatch_study import virtual_books
from .env import list_tapes, split_by_date
from .ladder import N_TICKS, PRICES
from .tape import Tape

HOLDS = (30, 60, 120)
PRICE_BINS = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 1001.0]
PRICE_LABELS = ["<2", "2-3", "3-5", "5-10", "10-20", "20+"]


def queue_fills(q0: float, shown: np.ndarray, at: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Per-step fill of a resting order (unbounded by its size).

    q0     queue ahead when the order goes live
    shown  [n] size shown at our price after each step (inf = our price is beyond the visible ladder)
    at     [n] volume traded AT our price in each step;  thr: [n] volume traded THROUGH it
    Queue: q_t = min(max(q_{t-1} - at_t, 0), shown_t)  (closed form: running minimum)."""
    A = np.cumsum(at)
    M = np.minimum(q0, np.minimum.accumulate(shown + A))
    q = np.maximum(M - A, 0.0)
    q_prev = np.concatenate([[q0], q[:-1]])
    return thr + np.maximum(at - q_prev, 0.0)


def sim_order(t: Tape, r: int, side: str, P: int, size: float, s1: int, end: int, ts, tt, tv, vsize, horizon):
    """One resting order, replaying exchange.py's realistic passive rules step by step, plus cross-matching.

    Live at s1 (crossing liquidity at or better than P is taken at once, then it joins the queue);
    each later step: forget taken liquidity that is gone or older than the horizon, fill from trades
    (queue first), cap the queue at what is shown, then take any book that has crossed our price.
    Cross-matching (vsize: virtual size at or better than P, per step) fills on top, each unit once.
    ts/tt/tv: this runner's trades (absolute step, tick, volume) in (s1, end].
    -> (cumulative filled size per step s1..end, {source: amount, "_q0": queue at start})"""
    back = side == "back"
    xt_all = (t.back_tick if back else t.lay_tick)[s1:end + 1, r].tolist()  # book we can take (crossing)
    xs_all = (t.back_size if back else t.lay_size)[s1:end + 1, r].tolist()
    qt_all = (t.lay_tick if back else t.back_tick)[s1:end + 1, r].tolist()  # book we queue in
    qs_all = (t.lay_size if back else t.back_size)[s1:end + 1, r].tolist()
    susp = t.suspended[s1:end + 1].tolist()
    vs = vsize.tolist()
    n = end - s1 + 1
    at = [0.0] * n
    thr = [0.0] * n
    for s, k, v in zip(ts.tolist(), tt.tolist(), tv.tolist()):
        j = s - s1
        if 0 < j < n:
            if k == P:
                at[j] += v
            elif (k > P) if back else (k < P):
                thr[j] += v
    rem, mem, vtaken, vmax = size, {}, 0.0, 0.0
    got = {"trades": 0.0, "book_cross": 0.0, "cross_match": 0.0}
    cum, out = 0.0, [0.0] * n

    def take(j):
        nonlocal rem, cum
        for tk, sz in zip(xt_all[j], xs_all[j]):
            if rem <= 1e-9 or tk < 0 or ((tk < P) if back else (tk > P)):
                break
            c = mem.get(tk)
            avail = max(0.0, sz - (c[0] if c else 0.0))
            if avail <= 1e-9:
                continue
            x = min(avail, rem)
            rem -= x
            cum += x
            got["book_cross"] += x
            if c:
                c[0] += x
                c[1] = j
            else:
                mem[tk] = [x, j]

    q = 0.0
    for j in range(n):
        if mem:  # exchange._decay_consumed
            lv = dict(zip(xt_all[j], xs_all[j]))
            for tk in list(mem):
                c = mem[tk]
                c[0] = min(c[0], lv.get(tk, 0.0))
                if c[0] <= 1e-9 or j - c[1] > horizon:
                    del mem[tk]
        if not susp[j]:
            if j == 0:  # goes live: take what already crosses, then join the queue
                take(0)
                lvq = qt_all[0]
                valid = [x for x in lvq if x >= 0]
                if P in lvq:
                    q = qs_all[0][lvq.index(P)]
                elif valid and len(valid) == len(lvq) and ((P > max(valid)) if back else (P < min(valid))):
                    q = 1e9  # past the visible ladder
                else:
                    q = 0.0
                got["_q0"] = q
            elif rem > 1e-9:
                burn = min(q, at[j])
                q -= burn
                x = min(rem, thr[j] + at[j] - burn)
                if x > 0:
                    rem -= x
                    cum += x
                    got["trades"] += x
                lvq = qt_all[j]
                if P in lvq:
                    q = min(q, qs_all[j][lvq.index(P)])
                elif q < 1e9:
                    q = 0.0
                take(j)
            if rem > 1e-9 and vs[j] > 0:  # cross-matching
                vmax = max(vmax, vs[j])
                x = min(rem, vmax - vtaken)
                if x > 1e-9:
                    rem -= x
                    cum += x
                    vtaken += x
                    got["cross_match"] += x
        out[j] = cum
    return np.array(out), got


def _net(g, c):
    return np.where(g > 0, g * (1.0 - c), g)  # commission on a positive market result only


def ev_back(p, bsp, c):
    """Back $1 at p, green up at BSP (lay p/BSP at BSP): the same result whoever wins."""
    return _net(p / bsp - 1.0, c)


def ev_lay(p, bsp, c):
    """Lay at p (per $1 of liability), green up at BSP."""
    return _net((1.0 - p / bsp) / (p - 1.0), c)


def sample_tape(path: str, every_s: float = 20.0, last_s: float = 600.0, hold_s: float = 120.0,
                offsets=(1, 2, 3), stake: float = 10.0):
    t = Tape.load(path)
    if not t.went_in_play:
        return None
    race = os.path.basename(path).split(".npz")[0]
    dt, T = t.dt, t.n_steps
    comm = t.base_rate / 100.0
    vb, vbs, vl, vls, live = virtual_books(t)
    rb, rl = t.back_tick[:, :, 0].astype(np.int32), t.lay_tick[:, :, 0].astype(np.int32)
    liq = best_usd_trailing(t, 10.0)
    after0 = np.where(t.t_rel >= 0)[0]
    off = int(after0[0]) if len(after0) else T - 1
    susp = t.suspended
    H = int(round(hold_s / dt))
    horizon = 10.0 / dt  # the exchange forgets liquidity we took after impact_decay_s (10s)
    by_runner = {}
    for r in range(t.n_runners):
        m = t.trade_runner == r
        by_runner[r] = (t.trade_step[m], t.trade_tick[m].astype(np.int32), t.trade_vol[m].astype(np.float64))
    rows = []
    k_dec = max(1, int(round(every_s / dt)))
    dec = np.where((t.t_rel >= -last_s) & (t.t_rel <= -5.0) & ~susp)[0][::k_dec]
    for s0 in dec:
        s1 = s0 + 1  # latency: live one step later
        end = min(s0 + H, off, T - 1)
        if s1 >= end or susp[s1]:
            continue
        ok = live[s0] & (rb[s0] >= 0) & (rl[s0] >= 0) & (t.bsp > 1.0)
        if ok.sum() < 2:
            continue
        mid = np.where(ok, np.sqrt(PRICES[np.maximum(rb[s0], 0)] * PRICES[np.maximum(rl[s0], 0)]), np.inf)
        rank = mid.argsort().argsort() + 1
        W2 = np.arange(s1, end + 1)
        for r in np.where(ok)[0]:
            ts, tt, tv = by_runner[r]
            a, b = np.searchsorted(ts, s1 + 1), np.searchsorted(ts, end, side="right")
            ws, wt, wv = ts[a:b], tt[a:b], tv[a:b]  # this runner's trades in (s1, end]
            base = dict(race=race, day=race[:8], runner=int(r), t=float(t.t_rel[s0]), rank=int(rank[r]),
                        field=int(ok.sum()), spread=int(rl[s0, r] - rb[s0, r]), best_usd=float(liq[s0, r]),
                        bsp=float(t.bsp[r]), comm=comm, rb_p=float(PRICES[rb[s0, r]]), rl_p=float(PRICES[rl[s0, r]]))
            for side in ("back", "lay"):
                # k = 0: cross now (at the best price when the order lands), the baseline
                pc = rb[s1, r] if side == "back" else rl[s1, r]
                if pc >= 0:
                    rows.append(dict(base, side=side, k=0, price=float(PRICES[pc]), position="cross", q0=0.0,
                                     f30=1.0, f60=1.0, f120=1.0, t_fill=0.0, source="cross_now"))
                for k in offsets:
                    P = rb[s0, r] + k if side == "back" else rl[s0, r] - k
                    if not (0 <= P < N_TICKS):
                        continue
                    p = float(PRICES[P])
                    size = stake if side == "back" else stake / (p - 1.0)  # backer-stake terms
                    opp_best = rl[s0, r] if side == "back" else rb[s0, r]
                    position = ("inside" if (P < opp_best if side == "back" else P > opp_best)
                                else "join" if P == opp_best else "behind")
                    if side == "back":
                        vtk, vsz = vb[W2, r], vbs[W2, r]
                        vsize = np.where((vtk >= 0) & (vtk >= P), vsz, 0.0)
                    else:
                        vtk, vsz = vl[W2, r], vls[W2, r]
                        vsize = np.where((vtk >= 0) & (vtk <= P), vsz, 0.0)
                    cum, src_amt = sim_order(t, r, side, P, size, s1, end, ws, wt, wv, vsize, horizon)
                    q0 = src_amt.pop("_q0", 0.0)
                    filled = np.minimum(cum, size)
                    full = cum >= size - 1e-9
                    i_full = int(np.argmax(full)) if full.any() else len(cum)

                    def by(h):
                        j = min(int(round(h / dt)), len(filled) - 1)
                        return float(filled[j] / size)

                    src = max(src_amt, key=src_amt.get) if cum[-1] > 1e-9 else "none"
                    rows.append(dict(base, side=side, k=k, price=p, position=position,
                                     q0=float(min(q0, 1e6)),
                                     f30=by(30), f60=by(60), f120=by(120),
                                     t_fill=float(i_full * dt) if i_full < len(cum) else np.nan, source=src))
    if not rows:
        return None
    d = pd.DataFrame(rows)
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    return d


def _one(args):
    path, kw = args
    try:
        return sample_tape(path, **kw)
    except Exception as e:
        print(f"skip {os.path.basename(path)}: {e}", flush=True)
        return None


def add_values(df: pd.DataFrame) -> pd.DataFrame:
    p, bsp, c = df["price"].to_numpy(float), df["bsp"].to_numpy(float), df["comm"].to_numpy(float)
    back = (df["side"] == "back").to_numpy()
    df["ev_fill"] = np.where(back, ev_back(p, bsp, c), ev_lay(p, bsp, c)) * 100  # % per $ at risk, if filled
    df["ev_fill_gross"] = np.where(back, ev_back(p, bsp, 0.0), ev_lay(p, bsp, 0.0)) * 100
    for h in HOLDS:
        df[f"ev_attempt_{h}"] = df["ev_fill"] * df[f"f{h}"]  # unfilled part = no bet
    df["price_band"] = pd.cut(df["price"], PRICE_BINS, labels=PRICE_LABELS, right=False)
    df["fav_rank"] = pd.cut(df["rank"], [0, 1, 3, 6, 99], labels=["fav", "2-3", "4-6", "7+"])
    df["spread_b"] = pd.cut(df["spread"], [0, 1, 2, 4, 9999], labels=["1 tick", "2", "3-4", "5+"])
    df["time"] = pd.cut(df["t"], [-1e9, -300, -120, -30, 0.1], labels=["10-5m", "5-2m", "2m-30s", "last 30s"])
    return df


def table_a(df: pd.DataFrame, hold: int = 60) -> pd.DataFrame:
    rows = []
    for (side, k), g in df.groupby(["side", "k"]):
        filled = g[f"f{hold}"] > 0
        src = g.loc[filled, "source"].value_counts(normalize=True) * 100
        rows.append({"side": side, "k": k, "attempts": len(g), "races": g["race"].nunique(),
                     "fill30_%": g["f30"].mean() * 100, "fill60_%": g["f60"].mean() * 100,
                     "fill120_%": g["f120"].mean() * 100, "t_fill_med_s": g["t_fill"].median(),
                     "by_trades_%": src.get("trades", 0.0), "by_book_cross_%": src.get("book_cross", 0.0),
                     "by_cross_match_%": src.get("cross_match", 0.0),
                     "ev_if_all_filled_%": g["ev_fill"].mean(),
                     f"ev_given_fill{hold}_%": float(np.average(g["ev_fill"], weights=g[f"f{hold}"]))
                     if g[f"f{hold}"].sum() > 0 else np.nan,
                     f"ev_attempt{hold}_%": g[f"ev_attempt_{hold}"].mean()})
    out = pd.DataFrame(rows)
    out["adverse_sel_pp"] = out[f"ev_given_fill{hold}_%"] - out["ev_if_all_filled_%"]
    return out


def table_by(df: pd.DataFrame, dim: str, hold: int = 60) -> pd.DataFrame:
    rows = []
    for (side, v), g in df[df["k"] > 0].groupby(["side", dim], observed=True):
        cr = df[(df["k"] == 0) & (df["side"] == side) & (df[dim] == v)]
        w = g[f"f{hold}"]
        rows.append({"side": side, dim: v, "attempts": len(g), "races": g["race"].nunique(),
                     f"fill{hold}_%": w.mean() * 100, "t_fill_med_s": g["t_fill"].median(),
                     "ev_if_all_filled_%": g["ev_fill"].mean(),
                     "ev_given_fill_%": float(np.average(g["ev_fill"], weights=w)) if w.sum() > 0 else np.nan,
                     "ev_attempt_%": g[f"ev_attempt_{hold}"].mean(),
                     "cross_now_ev_%": cr["ev_fill"].mean() if len(cr) else np.nan})
    out = pd.DataFrame(rows)
    out["adverse_sel_pp"] = out["ev_given_fill_%"] - out["ev_if_all_filled_%"]
    return out


def rt(v, race) -> dict:
    """Race-clustered: mean of per-race means, its t, and the number of races."""
    per = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna().groupby("race")["v"].mean()
    if len(per) < 2 or per.std(ddof=1) == 0:
        return dict(mean=float(per.mean()) if len(per) else np.nan, t=np.nan, races=len(per))
    return dict(mean=float(per.mean()), t=float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))), races=len(per))


def rule_search(df: pd.DataFrame, hold_days: set, min_races: int = 100, top: int = 10):
    tr = ~df["day"].isin(hold_days)
    dims = dict(side=["back", "lay"], k=[0, 1, 2, 3], hold=list(HOLDS), liquid=[False, True],
                price_band=[None] + PRICE_LABELS, fav_rank=[None, "fav", "2-3", "4-6", "7+"])
    res = []
    for side, k, h, lq, pb, fr in itertools.product(*dims.values()):
        m = (df["side"] == side) & (df["k"] == k)
        if lq:
            m &= df["liquid"]
        if pb is not None:
            m &= df["price_band"] == pb
        if fr is not None:
            m &= df["fav_rank"] == fr
        if k == 0 and h != HOLDS[1]:
            continue  # crossing fills at once: the hold does not matter
        g = df[m & tr]
        if g["race"].nunique() < min_races:
            continue
        col = f"ev_attempt_{h}" if k > 0 else "ev_fill"
        r_ = rt(g[col], g["race"])
        res.append(dict(side=side, k=k, hold=h, liquid=lq, price_band=pb or "all", fav_rank=fr or "all",
                        train_mean=r_["mean"], train_t=r_["t"], train_races=r_["races"], _m=m, _col=col))
    res = sorted(res, key=lambda x: -np.nan_to_num(x["train_t"], nan=-99))[:top]
    out = []
    for r in res:
        g = df[r["_m"] & ~tr]
        ht = rt(g[r["_col"]], g["race"])
        per = g.groupby("race")[r["_col"]].mean()
        drop = float(per.drop(per.idxmax()).mean()) if len(per) > 1 else np.nan  # without its best race
        out.append({k: v for k, v in r.items() if not k.startswith("_")} |
                   dict(holdout_mean=ht["mean"], holdout_t=ht["t"], holdout_races=ht["races"], holdout_drop_best=drop))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-s", type=float, default=20.0)
    ap.add_argument("--last-s", type=float, default=600.0)
    ap.add_argument("--hold-s", type=float, default=120.0, help="longest wait before the order is cancelled")
    ap.add_argument("--liq-cut", default="auto", help='"auto" = 80th pct of $ at best back + lay, TRAIN days')
    ap.add_argument("--min-races", type=int, default=100, help="rule search: min TRAIN races per slice")
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
    cache = os.path.join(a.out, f"orders{tag}.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        log(f"loaded {cache}")
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        parts = []
        kw = dict(every_s=a.every_s, last_s=a.last_s, hold_s=a.hold_s)
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            for i, d in enumerate(pool.map(_one, [(p, kw) for p in paths], chunksize=2)):
                if d is not None:
                    parts.append(d)
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(paths)} races, {time.time() - t0:.0f}s", flush=True)
        if not parts:
            raise SystemExit("no orders")
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(cache, index=False)
    df = add_values(df)
    tr = ~df["day"].isin(hold_days)
    cut = (float(np.nanquantile(df.loc[tr & (df["k"] == 0) & (df["t"] >= -300), "best_usd"], 0.8))
           if a.liq_cut == "auto" else float(a.liq_cut))
    df["liquid"] = df["best_usd"] >= cut
    df["liquidity"] = np.where(df["liquid"], f">= ${cut:.0f}", f"< ${cut:.0f}")
    log(f"{len(df)} orders ({(df['k'] > 0).sum()} resting, {(df['k'] == 0).sum()} crossing), {df['race'].nunique()} "
        f"races; $10 at risk, decisions every {a.every_s:.0f}s in the last {a.last_s / 60:.0f} min; liquid = >= "
        f"${cut:.2f} at best back + lay  [{time.time() - t0:.0f}s]")
    log("EV = per $ at risk, valued at BSP, net of commission (enter, exit at BSP). k = ticks better than the best "
        "available price; k = 0 = cross now.\n")

    log("=== A. Fill rate and value by side and offset (hold 60s for the EV columns) ===")
    for name, d in (("all runners", df), (f"liquid runners (>= ${cut:.0f})", df[df["liquid"]])):
        ta = table_a(d)
        ta.to_csv(os.path.join(a.out, f"A_{'liquid' if 'liquid' in name else 'all'}{tag}.csv"), index=False)
        log(f"\n{name}:")
        log(ta.round(3).to_string(index=False))

    log("\n=== B. Does price, rank, spread, queue position, liquidity or time matter? (k = 1..3 pooled, hold 60s) ===")
    for dim in ("price_band", "fav_rank", "spread_b", "position", "liquidity", "time"):
        tb = table_by(df, dim)
        tb.to_csv(os.path.join(a.out, f"B_{dim}{tag}.csv"), index=False)
        log(f"\nby {dim}:")
        log(tb.round(3).to_string(index=False))

    log(f"\n=== C. Rule search: picked on TRAIN days (>= {a.min_races} races), scored once on HOLDOUT ===")
    rs = rule_search(df, hold_days, a.min_races)
    rs.to_csv(os.path.join(a.out, f"C_rules{tag}.csv"), index=False)
    log(rs.round(3).to_string(index=False))
    best = rs.iloc[0].to_dict() if len(rs) else {}
    edge = bool(best) and best["holdout_mean"] > 0 and (best["holdout_t"] or 0) > 2 and \
        best["holdout_races"] >= 20 and best["holdout_drop_best"] > 0
    log("EDGE: the best TRAIN slice holds up on HOLDOUT -> forward-test it like P3c" if edge else
        "NO EDGE: the best TRAIN slice does not hold up on HOLDOUT")
    lq = table_a(df[df["liquid"]]).set_index(["side", "k"])
    verdict = dict(liq_cut=cut, edge=edge, best_rule=best,
                   liquid_k1={s: lq.loc[(s, 1)].to_dict() for s in ("back", "lay") if (s, 1) in lq.index},
                   note="EV valued at BSP, net of commission; rule search is multiple testing: a pass needs a forward test")
    json.dump(verdict, open(os.path.join(a.out, f"verdict{tag}.json"), "w"), indent=1, default=float)
    log(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
