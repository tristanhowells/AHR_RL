"""Community strategies: the pre-off rules Betfair traders actually use and
automate, tested on our recordings.

Sources (Bet Angel forum Guardian bots, Caan Berry, trading guides). Each rule is
written as an exact, mechanical spec so it can be tested honestly.

Scalping rules (every trade through the full queue-aware simulator, exchange.py:
latency, queue position, trade-through fills, commission; $10 stakes):

  S1 WOM scalp      Final 5 min, favourite or 2nd favourite. When one side of the
                    book holds > thr of the money in the top 3 levels, rest an
                    order at the best price on the reverse side ("offer at the best
                    reverse price"), 1-tick take-profit, 4-tick stop loss, no new
                    trades after T-30s, green at T-15s. Variants: thr 0.6 / 0.7,
                    trade WITH the heavy side or AGAINST it.
  S2 Pressure scalp One side has >= 2x the money at the best level and >= 1.5x at
                    the 2nd and 3rd levels ("pressure has built"). Same exits.
                    Favourite, 2nd favourite or both; WITH / AGAINST.
  S3 Gap fill       The spread has an empty tick (best lay - best back >= 2 ticks)
                    and one side holds > 0.7 of the top-3 money: fill the gap with a
                    resting order one tick inside the spread. Same exits.
  C  Control        The same scalp (join the best price, same exits) at random
                    moments on a random side: what the execution alone earns.

Signal rules (aggressive round trips as in the jump study: order lands one step
later, walks the visible book, commission; follow or fade, held 30 / 60 / 120s):

  V  Volume support / resistance   The price returns to the tick where the most
                    money has traded this session (after being >= 3 ticks away in
                    the last 60s). Fade = bet on a bounce, follow = a breakout.
  P  Spoofs         A large order (>= $100 and >= 5x the median level size) appears
                    1-3 ticks behind the best price and is pulled within 10s with
                    < 20% of it traded. At the pull, follow or fade the direction the
                    order implied (big back-side money implies a shortening).
  N  Late scratchings  After a runner with reduction factor >= 2.5% is scratched,
                    every other runner's fair price is its old price x (1 - RF).
                    Once the book reopens, runners still > 1 tick away from that
                    fair price are traded toward it.

Market-wide check:
  A  Arbitrage      How often backing (or laying) every runner at the best prices
                    locks a profit after commission, for how much, and whether it
                    survives the 0.5s latency.

Every rule/variant is picked on TRAIN days and scored once on HOLDOUT days
(race-clustered t); "edge" needs holdout mean > 0, t > 2 and >= 20 holdout races.

    python -m ahr_rl.community --tapes "data/tapes/*.npz" --out runs/community
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
from .exchange import BACK, LAY, Exchange, ExchangeConfig
from .jump_study import _round_trip
from .ladder import N_TICKS, PRICES
from .tape import Tape

STAKE = 10.0


# ----------------------------------------------------------------- simulator
def bracket(t: Tape, end: int, s: int, r: int, side: int, tick: int, s_last_entry: int, s_green: int,
            tp: int = 1, sl: int = 4, entry_wait_s: float = 10.0, max_hold_s: float = 120.0,
            fill_mode: str = "realistic") -> dict:
    """Resting entry at `tick`, resting take-profit `tp` ticks better, stop `sl` ticks
    against, forced green at s_green. Returns filled fraction, P&L per $1 matched and
    the step the trade finished."""
    dt = t.dt
    ex = Exchange(t, ExchangeConfig(fill_mode=fill_mode), start_step=s)
    o = ex.submit(r, side, int(np.clip(tick, 0, N_TICKS - 1)), STAKE)
    if o is None:
        return dict(filled=0.0, pnl=np.nan, done=s + 1)
    stop_wait = min(s + int(entry_wait_s / dt), s_last_entry, s_green)
    while ex.step < stop_wait and o.matched < STAKE - 1e-6 and ex.step < end:
        ex.advance()
    matched = o.matched
    ex.cancel_runner(r)
    if matched < 0.01:
        return dict(filled=0.0, pnl=0.0, done=ex.step + 1)
    p_in = PRICES[tick]
    hside = LAY if side == BACK else BACK
    tp_tick = int(np.clip(tick - tp if side == BACK else tick + tp, 0, N_TICKS - 1))
    ex.advance()  # let the cancel land before the exit order
    ex.submit(r, hside, tp_tick, matched * p_in / PRICES[tp_tick], is_hedge=True)
    s_exit = min(ex.step + int(max_hold_s / dt), s_green, end - 1)
    while ex.step < s_exit:
        ex.advance()
        if abs(ex.W[r] - ex.L[r]) < 1e-3:
            break
        bb, bl = ex.best(ex.step, r)
        if side == BACK and bl >= 0 and bl >= tick + sl:
            break
        if side == LAY and bb >= 0 and bb <= tick - sl:
            break
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
    pnl = ex.worst_net() if abs(ex.W[r] - ex.L[r]) < 0.05 else ex.green_value()
    return dict(filled=matched / STAKE, pnl=pnl / matched, done=ex.step + 1)


# ----------------------------------------------------------------- helpers
def _prep(t: Tape):
    T = t.n_steps
    ok = ~t.suspended.copy()
    if t.went_in_play and T > 1:
        ok[-1] = False
    if not ok.any():
        return None
    end = int(np.where(ok)[0].max())
    bt, lt = t.back_tick[:, :, 0].astype(int), t.lay_tick[:, :, 0].astype(int)
    valid = (bt >= 0) & (lt >= 0) & t.active & ok[:, None]
    valid[end + 1:] = False
    midp = np.where(valid, np.sqrt(PRICES[np.maximum(bt, 0)] * PRICES[np.maximum(lt, 0)]), np.nan)
    midt = np.where(valid, (bt + lt) / 2.0, np.nan)
    tr = t.t_rel.astype(float)
    step_of = lambda sec: int(min(max(np.searchsorted(tr, sec), 0), end))
    bad = np.zeros(T, bool)
    for s in t.removal_step:
        bad[max(0, s - 20): min(T, s + 21)] = True
    return dict(end=end, bt=bt, lt=lt, valid=valid, midp=midp, midt=midt, tr=tr, step_of=step_of, bad=bad,
                comm=t.base_rate / 100)


def _rank_at(P, s):
    live = np.where(P["valid"][s])[0]
    return live[np.argsort(P["midp"][s, live])]


def _race_t(v, race):
    x = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna()
    if len(x) < 2:
        return dict(n=len(x), races=x["race"].nunique(), mean=np.nan, t=np.nan)
    per = x.groupby("race")["v"].mean()
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 and per.std(ddof=1) > 0 else np.nan
    return dict(n=len(x), races=len(per), mean=float(x["v"].mean()), t=float(t))


# ----------------------------------------------------------------- scalping rules
def scalp_rules(t: Tape, P: dict, race: str, rng, every_s: float = 4.0, fill_mode: str = "realistic") -> list[dict]:
    end, bt, lt = P["end"], P["bt"], P["lt"]
    s0, s_last_entry, s_green = P["step_of"](-300), P["step_of"](-30), P["step_of"](-15)
    if s_green <= s0 + 10:
        return []
    k = max(1, int(every_s / t.dt))
    atb, atl = t.back_size[:, :, :3], t.lay_size[:, :, :3]
    rows = []

    def run(rule, variant, runner_set, trigger):
        """Walk the window; one open trade per runner at a time."""
        busy = {}
        for s in range(s0, s_last_entry, k):
            if P["bad"][s]:
                continue
            order = _rank_at(P, s)
            targets = {"fav": order[:1], "2nd": order[1:2], "top2": order[:2]}[runner_set]
            for r in targets:
                r = int(r)
                if busy.get(r, -1) > s or lt[s, r] - bt[s, r] > 3 or P["midp"][s, r] > 30:
                    continue
                sig = trigger(s, r)
                if sig is None:
                    continue
                side, tick = sig
                res = bracket(t, end, s, r, side, tick, s_last_entry, s_green, fill_mode=fill_mode)
                busy[r] = res["done"]
                rows.append(dict(race=race, rule=rule, variant=variant, runners=runner_set, t_rel=P["tr"][s], **res))

    def share_backers(s, r, lv=3):
        a, b = atl[s, r, :lv].sum(), atb[s, r, :lv].sum()  # atl = backers' money, atb = layers' money
        return a / (a + b) if a + b > 0 else np.nan

    # S1 WOM scalp: offer at the best reverse price on the heavy side (WITH) or the light side (AGAINST)
    for thr in (0.6, 0.7):
        for mode in ("with", "against"):
            for rs in ("fav", "2nd"):
                def trig(s, r, thr=thr, mode=mode):
                    w = share_backers(s, r)
                    if not np.isfinite(w):
                        return None
                    if w > thr:
                        heavy = BACK  # backers dominate -> expect a shortening -> back
                    elif 1 - w > thr:
                        heavy = LAY
                    else:
                        return None
                    side = heavy if mode == "with" else (LAY if heavy == BACK else BACK)
                    return (side, lt[s, r]) if side == BACK else (side, bt[s, r])
                run("S1_wom_scalp", f"thr{thr}_{mode}", rs, trig)

    # S2 pressure scalp: level ratios 2x / 1.5x / 1.5x
    for mode in ("with", "against"):
        for rs in ("fav", "2nd", "top2"):
            def trig(s, r, mode=mode):
                a, b = atl[s, r], atb[s, r]
                if (a[0] >= 2 * b[0] > 0) and a[1] >= 1.5 * b[1] and a[2] >= 1.5 * b[2]:
                    heavy = BACK
                elif (b[0] >= 2 * a[0] > 0) and b[1] >= 1.5 * a[1] and b[2] >= 1.5 * a[2]:
                    heavy = LAY
                else:
                    return None
                side = heavy if mode == "with" else (LAY if heavy == BACK else BACK)
                return (side, lt[s, r]) if side == BACK else (side, bt[s, r])
            run("S2_pressure_scalp", mode, rs, trig)

    # S3 gap fill: one tick inside a >= 2 tick spread, side chosen by WOM > 0.7
    for mode in ("with", "against"):
        for rs in ("fav", "top2"):
            def trig(s, r, mode=mode):
                if lt[s, r] - bt[s, r] < 2:
                    return None
                w = share_backers(s, r)
                if not np.isfinite(w) or max(w, 1 - w) <= 0.7:
                    return None
                heavy = BACK if w > 0.7 else LAY
                side = heavy if mode == "with" else (LAY if heavy == BACK else BACK)
                return (side, lt[s, r] - 1) if side == BACK else (side, bt[s, r] + 1)
            run("S3_gap_fill", mode, rs, trig)

    # C control: random moments, random side, join
    for rs in ("fav", "2nd"):
        def trig(s, r):
            if rng.random() > 0.15:
                return None
            side = BACK if rng.random() < 0.5 else LAY
            return (side, lt[s, r]) if side == BACK else (side, bt[s, r])
        run("C_control", "random", rs, trig)
    return rows


# ----------------------------------------------------------------- signal rules
def _outcomes(t, P, s, r, direction, horizons=(30, 60, 120)):
    """direction +1 = expect the price to lengthen, -1 = shorten. follow_H trades that way."""
    end, dt = P["end"], t.dt
    rec = {}
    for H in horizons:
        s2 = min(s + int(H / dt), end)
        m0, m1 = P["midt"][s, r], P["midt"][s2, r]
        rec[f"move_{H}"] = float(direction * (m1 - m0)) if np.isfinite(m0) and np.isfinite(m1) else np.nan
        s_in, s_out = min(s + 1, end), min(s2 + 1, end)
        if s_out <= s_in:
            rec[f"follow_{H}"] = rec[f"fade_{H}"] = np.nan
            continue
        fside, dside = ("back", "lay") if direction < 0 else ("lay", "back")
        rec[f"follow_{H}"] = _round_trip(t, r, s_in, s_out, fside, STAKE, P["comm"])[0]
        rec[f"fade_{H}"] = _round_trip(t, r, s_in, s_out, dside, STAKE, P["comm"])[0]
    return rec


def signal_rules(t: Tape, P: dict, race: str) -> list[dict]:
    end, dt, tr = P["end"], t.dt, P["tr"]
    T, R = t.n_steps, t.n_runners
    rows = []
    # V: volume profile point-of-control touches
    vol_tick = {}
    ts, rr, tk, vv = t.trade_step, t.trade_runner, t.trade_tick, t.trade_vol
    order = np.argsort(ts, kind="stable")
    ptr = 0
    prof = [dict() for _ in range(R)]
    last_touch = np.full(R, -10**9)
    for s in range(1, end - 10):
        while ptr < len(order) and ts[order[ptr]] <= s:
            j = order[ptr]
            d = prof[rr[j]]
            d[int(tk[j])] = d.get(int(tk[j]), 0.0) + float(vv[j])
            ptr += 1
        if s % 2 or P["bad"][s]:
            continue
        for r in range(R):
            m = P["midt"][s, r]
            if not np.isfinite(m) or not prof[r] or P["midp"][s, r] > 30 or P["lt"][s, r] - P["bt"][s, r] > 3:
                continue
            tot = sum(prof[r].values())
            if tot < 200:
                continue
            poc = max(prof[r], key=prof[r].get)
            if abs(m - poc) > 0.5 or s - last_touch[r] < int(60 / dt):
                continue
            past = P["midt"][max(0, s - int(60 / dt)): s, r]
            past = past[np.isfinite(past)]
            if not len(past) or np.max(np.abs(past - poc)) < 3:
                continue
            last_touch[r] = s
            came_from = np.sign(past[np.argmax(np.abs(past - poc))] - poc)  # +1: came down from longer prices
            # breakout direction = continuing the approach (came from longer -> keeps shortening)
            direction = -1 if came_from > 0 else 1
            rows.append(dict(race=race, rule="V_volume_level", variant="breakout=follow", t_rel=tr[s],
                             poc_share=prof[r][poc] / tot, **_outcomes(t, P, s, r, direction)))
    # P: spoofs (large orders behind the best price, pulled without trading)
    for r in range(R):
        for book, sizes, ticks, implied in ((BACK, t.back_size, t.back_tick, +1), (LAY, t.lay_size, t.lay_tick, -1)):
            # book BACK = atb (layers' money): a big lay offer implies a drift (+1); atl (backers) implies -1
            open_ = {}
            for s in range(1, end - 10):
                if not P["valid"][s, r]:
                    open_.clear()
                    continue
                cur = {int(ticks[s, r, j]): float(sizes[s, r, j]) for j in range(1, 4) if ticks[s, r, j] >= 0}
                prev = {int(ticks[s - 1, r, j]): float(sizes[s - 1, r, j]) for j in range(0, 8) if ticks[s - 1, r, j] >= 0}
                med = np.median(sizes[s, r][sizes[s, r] > 0]) if (sizes[s, r] > 0).any() else 0
                for tick_, sz in cur.items():
                    add = sz - prev.get(tick_, 0.0)
                    if add >= max(100.0, 5 * med) and tick_ not in open_:
                        open_[tick_] = (s, add)
                for tick_, (s_on, add) in list(open_.items()):
                    now = {int(ticks[s, r, j]): float(sizes[s, r, j]) for j in range(0, 8) if ticks[s, r, j] >= 0}
                    if s - s_on > int(10 / dt):
                        del open_[tick_]
                        continue
                    if now.get(tick_, 0.0) <= 0.2 * add:
                        sel = (rr == r) & (tk == tick_) & (ts > s_on) & (ts <= s)
                        if vv[sel].sum() < 0.2 * add and not P["bad"][s]:
                            if P["midp"][s, r] <= 30:
                                rows.append(dict(race=race, rule="P_spoof_pulled", variant="follow=implied", t_rel=tr[s],
                                                 size=add, secs_shown=(s - s_on) * dt,
                                                 **_outcomes(t, P, s, r, implied)))
                        del open_[tick_]
    # N: late scratchings
    n_at = pd.Series(t.removal_step).value_counts() if len(t.removal_step) else pd.Series(dtype=int)
    for s_rm, r_rm, rf in zip(t.removal_step, t.removal_runner, t.removal_factor):
        # a single scratching only (several runners removed at once = abandonment / resettlement)
        if rf < 2.5 or s_rm < 4 or s_rm >= end - int(60 / dt) or n_at.get(s_rm, 0) > 1:
            continue
        s_pre = s_rm - 2
        s_post = next((s for s in range(s_rm + 1, min(end, s_rm + int(60 / dt))) if not t.suspended[s]
                       and P["valid"][s].sum() >= 2), None)
        if s_post is None:
            continue
        s_post = min(s_post + int(5 / dt), end - 1)  # give the book 5s to rebuild
        for r in range(R):
            if r == r_rm or not (np.isfinite(P["midp"][s_pre, r]) and np.isfinite(P["midp"][s_post, r])):
                continue
            fair = P["midp"][s_pre, r] * (1 - rf / 100.0)
            gap = np.log(P["midp"][s_post, r] / fair) * 100
            fair_tick = int(np.abs(PRICES - fair).argmin())
            if abs(P["midt"][s_post, r] - fair_tick) <= 1 or P["midp"][s_post, r] > 30:
                continue
            direction = -1 if gap > 0 else 1  # still too long -> expect shortening
            rows.append(dict(race=race, rule="N_scratching", variant="follow=toward_fair", t_rel=tr[s_post], rf=rf,
                             gap_pct=gap, **_outcomes(t, P, s_post, r, direction)))
    return rows


def arbitrage(t: Tape, P: dict, race: str) -> dict:
    end, comm = P["end"], P["comm"]
    live = t.active[:end + 1]
    bp = np.where(P["bt"][:end + 1] >= 0, PRICES[np.maximum(P["bt"][:end + 1], 0)], np.nan)
    lp = np.where(P["lt"][:end + 1] >= 0, PRICES[np.maximum(P["lt"][:end + 1], 0)], np.nan)
    full = ~np.isnan(np.where(live, bp, 0)).any(1) & ~np.isnan(np.where(live, lp, 0)).any(1) & ~t.suspended[:end + 1]
    bb = np.nansum(np.where(live, 1 / bp, 0), 1)
    lb = np.nansum(np.where(live, 1 / lp, 0), 1)
    back_arb = full & ((1 / np.maximum(bb, 1e-9) - 1) * (1 - comm) > 0)
    lay_arb = full & ((lb - 1) * (1 - comm) > 0)
    both = lambda x: x[:-1] & x[1:]  # still there 0.5s later (latency)
    cap = []
    for s in np.where(back_arb[:-1] & back_arb[1:])[0]:
        m = live[s]
        S = np.min(t.back_size[s + 1, m, 0] * bp[s + 1, m]) * bb[s + 1]
        cap.append(S * (1 / bb[s + 1] - 1) * (1 - comm))
    return dict(race=race, rows=int(full.sum()), back_arb=int(back_arb.sum()), lay_arb=int(lay_arb.sum()),
                back_arb_persist=int(both(back_arb).sum()), lay_arb_persist=int(both(lay_arb).sum()),
                back_arb_profit=float(np.sum(cap)) if cap else 0.0, median_back_book=float(np.median(bb[full])) if full.any() else np.nan,
                median_lay_book=float(np.median(lb[full])) if full.any() else np.nan)


def study_tape(path: str, fill_mode: str = "realistic", seed: int = 0) -> dict:
    t = Tape.load(path)
    race = os.path.basename(path).split(".npz")[0]
    P = _prep(t)
    if P is None:
        return dict(scalp=[], signal=[], arb=None)
    rng = np.random.default_rng(abs(hash(race)) % (2**32) + seed)
    return dict(scalp=scalp_rules(t, P, race, rng, fill_mode=fill_mode), signal=signal_rules(t, P, race),
                arb=arbitrage(t, P, race))


def _one(args):
    path, kw = args
    try:
        return study_tape(path, **kw)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}", flush=True)
        return dict(scalp=[], signal=[], arb=None)


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-races", type=int, default=100000)
    ap.add_argument("--min-races", type=int, default=30)
    ap.add_argument("--fill-mode", default="realistic", choices=["realistic", "no_queue", "touch"])
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    t0 = time.time()
    paths = list_tapes(a.tapes)[: a.max_races]
    tr, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    os.environ["OMP_NUM_THREADS"] = "1"
    scalp, signal, arb = [], [], []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, res in enumerate(pool.map(_one, [(p, dict(fill_mode=a.fill_mode)) for p in paths], chunksize=2)):
            scalp.extend(res["scalp"])
            signal.extend(res["signal"])
            if res["arb"]:
                arb.append(res["arb"])
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(paths)} races, {len(scalp)} scalps, {len(signal)} signal events, "
                      f"{time.time() - t0:.0f}s", flush=True)
    split = lambda df: np.where(df["race"].str[:8].isin(hold_days), "holdout", "train")
    S = pd.DataFrame(scalp)
    G = pd.DataFrame(signal)
    A = pd.DataFrame(arb)
    for name, df in (("scalps", S), ("signals", G), ("arbitrage", A)):
        if len(df):
            df["split"] = split(df)
            df.to_parquet(os.path.join(a.out, f"{name}.parquet"), index=False)
    verdict = {}
    hold_min = max(20, a.min_races // 2)

    # ------------- scalps
    print("\n" + "=" * 100 + f"\nSCALPING RULES (full simulator, fill mode '{a.fill_mode}'; P&L per $1 matched, "
          "after commission)\n" + "=" * 100)
    if len(S):
        rows = []
        for (rule, var, rs), g in S.groupby(["rule", "variant", "runners"]):
            rec = dict(rule=rule, variant=var, runners=rs)
            for sp, gg in g.groupby("split"):
                f = gg[gg["filled"] > 0]
                pf, pa = _race_t(f["pnl"], f["race"]), _race_t(gg["pnl"].fillna(0) * gg["filled"], gg["race"])
                rec.update({f"{sp}_attempts": len(gg), f"{sp}_fill_%": (gg["filled"] > 0).mean() * 100,
                            f"{sp}_per_fill": pf["mean"], f"{sp}_per_attempt": pa["mean"], f"{sp}_t": pa["t"],
                            f"{sp}_races": pa["races"], f"{sp}_attempts_per_race": len(gg) / max(gg["race"].nunique(), 1)})
            rows.append(rec)
        R_ = pd.DataFrame(rows).sort_values("train_t", ascending=False)
        R_.to_csv(os.path.join(a.out, "scalp_rules.csv"), index=False)
        cols = [c for c in ["rule", "variant", "runners", "train_attempts_per_race", "train_fill_%", "train_per_fill",
                            "train_per_attempt", "train_t", "holdout_fill_%", "holdout_per_fill", "holdout_per_attempt",
                            "holdout_t", "holdout_races"] if c in R_]
        print(R_[cols].round(4).to_string(index=False))
        best = R_[R_["rule"] != "C_control"].iloc[0].to_dict()
        verdict["scalp_best"] = {k: best.get(k) for k in cols}
        verdict["scalp_edge"] = bool((best.get("holdout_per_attempt") or 0) > 0 and (best.get("holdout_t") or 0) > 2
                                     and (best.get("holdout_races") or 0) >= hold_min)

    # ------------- signals
    print("\n" + "=" * 100 + "\nSIGNAL RULES (aggressive round trips, $10; move in ticks in the expected direction)\n" + "=" * 100)
    if len(G):
        rows = []
        for (rule, var), g in G.groupby(["rule", "variant"]):
            for H in (30, 60, 120):
                for act in ("follow", "fade"):
                    rec = dict(rule=rule, variant=var, horizon=H, action=act, events=len(g),
                               races=g["race"].nunique(), move_ticks=g[f"move_{H}"].mean(),
                               pct_expected_dir=(g[f"move_{H}"] > 0).mean() * 100)
                    for sp, gg in g.groupby("split"):
                        rt = _race_t(gg[f"{act}_{H}"], gg["race"])
                        rec.update({f"{sp}_mean": rt["mean"], f"{sp}_t": rt["t"], f"{sp}_races": rt["races"]})
                    rows.append(rec)
        Q = pd.DataFrame(rows).sort_values("train_t", ascending=False)
        Q.to_csv(os.path.join(a.out, "signal_rules.csv"), index=False)
        print(Q.round(4).to_string(index=False))
        best = Q.iloc[0].to_dict()
        verdict["signal_best"] = best
        verdict["signal_edge"] = bool((best.get("holdout_mean") or 0) > 0 and (best.get("holdout_t") or 0) > 2
                                      and (best.get("holdout_races") or 0) >= hold_min)
        if "rf" not in G or not (G["rule"] == "N_scratching").any():
            print("\nscratchings: none inside the recorded windows (most happen before T-10m)")
        if "rf" in G:
            n = G[G["rule"] == "N_scratching"]
            if len(n):
                print(f"\nscratchings inside the recorded window: {n['race'].nunique()} races, {len(n)} runner events, median |gap to fair| "
                      f"{n['gap_pct'].abs().median():.1f}%")

    # ------------- arbitrage
    print("\n" + "=" * 100 + "\nARBITRAGE (back or lay every runner at the best prices; after commission)\n" + "=" * 100)
    if len(A):
        tot = A["rows"].sum()
        print(f"{len(A)} races, {tot} snapshots. Back-all arb in {A['back_arb'].sum()} snapshots "
              f"({A['back_arb'].sum() / tot * 100:.3f}%), still there 0.5s later in {A['back_arb_persist'].sum()}; "
              f"lay-all arb in {A['lay_arb'].sum()} ({A['lay_arb'].sum() / tot * 100:.3f}%), persisting "
              f"{A['lay_arb_persist'].sum()}. Races with a persisting back arb: {(A['back_arb_persist'] > 0).sum()}.")
        print(f"Profit available from persisting back arbs (visible size, once each snapshot): "
              f"${A['back_arb_profit'].sum():.2f} in total. Median back book {A['median_back_book'].median() * 100:.1f}%, "
              f"lay book {A['median_lay_book'].median() * 100:.1f}%.")
        verdict["arb"] = dict(back_arb_pct=float(A["back_arb"].sum() / tot * 100),
                              back_arb_persist=int(A["back_arb_persist"].sum()),
                              back_arb_profit_total=float(A["back_arb_profit"].sum()))

    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=str)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=str))
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
