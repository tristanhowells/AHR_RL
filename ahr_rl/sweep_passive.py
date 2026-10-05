"""Post-sweep passive test: earn the overshoot after a sweep with a RESTING order
instead of paying the spread to catch it.

The jump study found that after a >= 5 tick sweep the price comes back ~0.5-1
tick, but crossing the spread twice costs more than that. Right after a sweep
the book on one side has just been emptied, so an order resting there may sit at
the FRONT of the queue and be filled by the snap-back itself. Earlier passive
tests lost money to adverse selection (fills mostly came from informed traders);
this checks whether post-sweep fills are different.

Every trade runs through the full exchange simulator (``exchange.py``): 0.5s
latency, queue position behind the size already shown at our price (burned by
trades at that price), trade-through and book-crossing fills, commission.

Events: sweeps of >= J ticks within 5s (J = 3, 5) with >= $20 traded, tradeable
runners (spread <= 3 ticks, price <= 30), not near a scratching (see jump_study).

Strategy (per event, $10):
  entry  a resting order on the FADE side (bet on the snap-back: back after a
         drift, lay after a steam), either JOINING the best price on its side
         of the book or IMPROVING it by one tick when the spread allows. If it
         is not filled within ``W`` seconds it is cancelled: no trade.
  exit   a resting take-profit hedge ``tp`` ticks better than the entry (tp = 0:
         none), and after ``H`` seconds (or 10s before the last pre-off row)
         anything left is hedged aggressively (up to 5 ticks through).
  P&L    locked (worst-case) green after commission, per $1 of matched entry.
The same strategy is also run on the FOLLOW side, and at random moments with a
random side (control: what resting orders earn without a sweep).

Rules (J, direction, entry mode, W, tp, H, side, top-3 runners or all) are
picked on TRAIN days and scored once on HOLDOUT days; "edge" also needs >= 20
holdout races. ``--fill-modes realistic,no_queue`` repeats the
winner with "always first in the queue" as an optimistic bound.

    python -m ahr_rl.sweep_passive --tapes "data/tapes/*.npz" --out runs/sweep_passive
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

from .env import list_tapes, split_by_date
from .exchange import BACK, LAY, Exchange, ExchangeConfig
from .ladder import N_TICKS, PRICES
from .tape import Tape

GRID = dict(entry=("join", "improve"), W=(10, 30), tp=(0, 2), H=(60, 120))


def find_events(t: Tape, jumps=(3, 5), window_s=5.0, cooldown_s=15.0, max_spread=3, max_price=30.0,
                min_sweep_vol=20.0, control_every_s=180.0, rng=None, live=False):
    """Sweeps (and random control moments) per runner. ``top3`` ranks runners by
    matched volume at the off (look-ahead; what P3/P3b used) unless ``live``, which
    ranks by volume matched so far and also scans the last 40s before the off;
    ``top3_final`` is always the at-the-off ranking."""
    dt, T = t.dt, t.n_steps
    ok = ~t.suspended.copy()
    if t.went_in_play and T > 1:
        ok[-1] = False
    if not ok.any():
        return [], 0
    end = int(np.where(ok)[0].max())
    W, cool = int(window_s / dt), int(cooldown_s / dt)
    bt, lt = t.back_tick[:, :, 0].astype(float), t.lay_tick[:, :, 0].astype(float)
    valid = (bt >= 0) & (lt >= 0) & t.active & ok[:, None]
    spread = np.where(valid, lt - bt, np.inf)
    mid = np.where(valid & (spread <= max_spread), (bt + lt) / 2, np.nan)
    mid[end + 1:] = np.nan
    bad = np.zeros(T, bool)
    for s in t.removal_step:
        bad[max(0, s - int(10 / dt)): min(T, s + int(10 / dt) + 1)] = True
    vol = np.zeros((T, t.n_runners))
    np.add.at(vol, (np.clip(t.trade_step, 0, T - 1), t.trade_runner), t.trade_vol)
    cv = np.vstack([np.zeros((1, t.n_runners)), np.cumsum(vol, 0)])
    matched = t.tv[end]
    rank = np.argsort(np.argsort(-np.nan_to_num(matched)))
    rank_now = np.argsort(np.argsort(-cv[1:], axis=1), axis=1) if live else None  # [T, R], volume up to step s
    last_s = end if live else end - int(40 / dt)
    events = []
    for r in range(t.n_runners):
        m = mid[:, r]
        for J in jumps:
            last = -10**9
            for s in range(W, last_s):
                if s - last < cool or bad[s] or not np.isfinite(m[s]) or not np.isfinite(m[s - W]):
                    continue
                d = m[s] - m[s - W]
                if abs(d) < J or PRICES[int(round(m[s]))] > max_price:
                    continue
                if cv[s + 1, r] - cv[s - W + 1, r] < min_sweep_vol:
                    continue
                last = s
                events.append(dict(kind="sweep", J=J, step=s, runner=r, direction=1 if d > 0 else -1,
                                   top3=bool((rank_now[s, r] if live else rank[r]) < 3),
                                   top3_final=bool(rank[r] < 3), t_rel=float(t.t_rel[s])))
        ce = int(control_every_s / dt)
        for s in range(W + int(rng.integers(0, ce)), end - int(40 / dt), ce):
            if bad[s] or not np.isfinite(m[s]) or PRICES[int(round(m[s]))] > max_price:
                continue
            events.append(dict(kind="control", J=0, step=s, runner=r, direction=int(rng.choice([-1, 1])),
                               top3=bool(rank[r] < 3), t_rel=float(t.t_rel[s])))
    return events, end


def run_trade(t: Tape, end: int, s: int, r: int, side: int, entry: str, W: int, tp: int, H: int,
              stake: float = 10.0, fill_mode: str = "realistic") -> dict:
    """One passive round trip in the simulator. side = BACK or LAY (the entry side)."""
    dt = t.dt
    ex = Exchange(t, ExchangeConfig(fill_mode=fill_mode), start_step=s)
    bb, bl = ex.best(s, r)
    if bb < 0 or bl < 0:
        return dict(filled=0.0, pnl=np.nan)
    # a resting BACK sits on the lay side (best lay price bl); a resting LAY at the best back price bb
    if side == BACK:
        tick = bl - 1 if (entry == "improve" and bl - 1 > bb) else bl
    else:
        tick = bb + 1 if (entry == "improve" and bb + 1 < bl) else bb
    o = ex.submit(r, side, tick, stake)
    if o is None:
        return dict(filled=0.0, pnl=np.nan)
    last = end - int(10 / dt)
    s_entry_end = min(s + int(W / dt), last)
    while ex.step < s_entry_end and o.matched < stake - 1e-6:
        ex.advance()
    matched = o.matched
    if matched < 0.01:
        return dict(filled=0.0, pnl=0.0)
    ex.cancel_runner(r)
    entry_price = PRICES[tick]
    s_exit = min(ex.step + int(H / dt), last)
    # resting take-profit hedge
    if tp > 0:
        plan = ex.hedge_plan(r)
        if plan is not None:
            hside, _, hstake = plan
            tp_tick = int(np.clip(tick - tp if side == BACK else tick + tp, 0, N_TICKS - 1))
            # hedge a back with a lay at a SHORTER price (lower tick); a lay with a back at a LONGER price
            ex.submit(r, hside, tp_tick, matched * entry_price / PRICES[tp_tick], is_hedge=True)
    while ex.step < s_exit:
        ex.advance()
        if abs(ex.W[r] - ex.L[r]) < 1e-3:  # fully hedged
            break
    # hedge whatever is left, aggressively (up to 5 ticks through)
    for _ in range(6):
        ex.cancel_runner(r)
        plan = ex.hedge_plan(r)
        if plan is None or ex.step >= end:
            break
        hside, htick, hstake = plan
        lim = int(np.clip(htick + 5 if hside == LAY else htick - 5, 0, N_TICKS - 1))
        ex.submit(r, hside, lim, hstake, is_hedge=True)
        ex.advance()
        ex.advance()
    ex.cancel_runner(r)
    pnl = ex.worst_net() if abs(ex.W[r] - ex.L[r]) < 0.05 else ex.green_value()
    return dict(filled=matched / stake, pnl=pnl / matched)


def study_tape(path: str, fill_mode: str = "realistic", seed: int = 0, max_events: int = 400) -> list[dict]:
    t = Tape.load(path)
    race = os.path.basename(path).split(".npz")[0]
    rng = np.random.default_rng(abs(hash(race)) % (2**32) + seed)
    events, end = find_events(t, rng=rng)
    if len(events) > max_events:
        events = [events[i] for i in rng.choice(len(events), max_events, replace=False)]
    rows = []
    comm = t.base_rate / 100
    for e in events:
        for act in ("fade", "follow"):
            # fade: bet on the snap-back. drift (price up) -> back; steam (price down) -> lay
            side = (BACK if e["direction"] > 0 else LAY) if act == "fade" else (LAY if e["direction"] > 0 else BACK)
            for entry, W, tp, H in itertools.product(*GRID.values()):
                res = run_trade(t, end, e["step"], e["runner"], side, entry, W, tp, H, fill_mode=fill_mode)
                rows.append(dict(race=race, day=race[:8], comm=comm, **e, action=act, entry=entry, W=W, tp=tp, H=H,
                                 **res))
    return rows


def _one(args):
    path, kw = args
    try:
        return study_tape(path, **kw)
    except Exception as ex:
        print(f"  skip {os.path.basename(path)}: {ex}", flush=True)
        return []


def race_t(v, race):
    x = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna()
    if len(x) < 2:
        return dict(n=len(x), races=0, mean=np.nan, t=np.nan)
    per = x.groupby("race")["v"].mean()
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 and per.std(ddof=1) > 0 else np.nan
    return dict(n=len(x), races=len(per), mean=float(x["v"].mean()), t=float(t))


RULE = ["kind", "J", "dir", "action", "entry", "W", "tp", "H", "top3"]


def score(df: pd.DataFrame, min_races: int) -> pd.DataFrame:
    out = []
    df = df.assign(dir=np.where(df["direction"] > 0, "drift", "steam"))
    for keys, g in df.groupby(["kind", "J", "action", "entry", "W", "tp", "H"]):
        for dirn in ("drift", "steam", "both"):
            for top in ("all", "top3"):
                gg = g if dirn == "both" else g[g["dir"] == dirn]
                if top == "top3":
                    gg = gg[gg["top3"]]
                if gg["race"].nunique() < min_races:
                    continue
                f = gg[gg["filled"] > 0]
                rt = race_t(f["pnl"], f["race"])  # per filled trade
                ra = race_t(gg["pnl"].fillna(0) * gg["filled"], gg["race"])  # per attempt ($ per $10 intended)
                out.append(dict(zip(["kind", "J", "action", "entry", "W", "tp", "H"], keys), dir=dirn, top3=top,
                                attempts=len(gg), fill_rate=(gg["filled"] > 0).mean() * 100,
                                per_fill=rt["mean"], per_fill_t=rt["t"], per_attempt=ra["mean"], per_attempt_t=ra["t"],
                                races=ra["races"]))
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-races", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-races", type=int, default=100000)
    ap.add_argument("--fill-modes", default="realistic,no_queue")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    t0 = time.time()
    paths = list_tapes(a.tapes)[: a.max_races]
    tr, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    modes = a.fill_modes.split(",")
    os.environ["OMP_NUM_THREADS"] = "1"
    rows = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, res in enumerate(pool.map(_one, [(p, dict(fill_mode=modes[0])) for p in paths], chunksize=2)):
            rows.extend(res)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(paths)} races, {len(rows)} simulated trades, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["split"] = np.where(df["day"].isin(hold_days), "holdout", "train")
    df.to_parquet(os.path.join(a.out, "trades.parquet"), index=False)
    print(f"\n{len(df)} simulated trades ({df['race'].nunique()} races), fill mode {modes[0]}")

    print("\n" + "=" * 100 + "\nFILL RATES AND P&L, all days (per $1 of matched entry, after commission)\n" + "=" * 100)
    summ = (df.assign(dir=np.where(df["direction"] > 0, "drift", "steam"), fill=df["filled"] > 0)
            .groupby(["kind", "J", "dir", "action", "entry"])
            .apply(lambda g: pd.Series(dict(attempts=len(g), fill_rate=g["fill"].mean() * 100,
                                            per_fill=g.loc[g["fill"], "pnl"].mean(),
                                            per_attempt=(g["pnl"].fillna(0) * g["filled"]).mean()))))
    summ.to_csv(os.path.join(a.out, "summary.csv"))
    print(summ.round(4).to_string())

    print("\n" + "=" * 100 + "\nRULES picked on TRAIN days, scored on HOLDOUT days\n" + "=" * 100)
    st = score(df[df["split"] == "train"], a.min_races)
    sh = score(df[df["split"] == "holdout"], max(5, a.min_races // 4))
    keys = ["kind", "J", "action", "entry", "W", "tp", "H", "dir", "top3"]
    m = st.merge(sh, on=keys, how="left", suffixes=("", "_holdout"))
    m = m[m["kind"] == "sweep"].sort_values("per_attempt_t", ascending=False)
    m.to_csv(os.path.join(a.out, "rules.csv"), index=False)
    cols = keys[1:] + ["attempts", "fill_rate", "per_fill", "per_attempt", "per_attempt_t", "fill_rate_holdout",
                       "per_fill_holdout", "per_attempt_holdout", "per_attempt_t_holdout", "races_holdout"]
    print(m[cols].head(15).round(4).to_string(index=False))
    ctl = st[st["kind"] == "control"].sort_values("per_attempt_t", ascending=False)
    print("\ncontrol (random moments, random side) - best train cells:")
    print(ctl[["action", "entry", "W", "tp", "H", "attempts", "fill_rate", "per_fill", "per_attempt", "per_attempt_t"]]
          .head(5).round(4).to_string(index=False))

    best = m.iloc[0].to_dict() if len(m) else {}
    verdict = dict(best_rule={k: best.get(k) for k in keys}, train_per_attempt=best.get("per_attempt"),
                   train_t=best.get("per_attempt_t"), holdout_per_attempt=best.get("per_attempt_holdout"),
                   holdout_t=best.get("per_attempt_t_holdout"), holdout_fill_rate=best.get("fill_rate_holdout"))
    verdict["holdout_races"] = best.get("races_holdout")
    verdict["edge"] = bool(best and (verdict["holdout_per_attempt"] or 0) > 0 and (verdict["holdout_t"] or 0) > 2
                           and (best.get("races_holdout") or 0) >= max(20, a.min_races // 2))

    # optimistic bound: re-run the best rule's events with other fill modes
    if best and len(modes) > 1:
        ev = df[(df["kind"] == "sweep") & (df["J"] == best["J"]) & (df["action"] == best["action"])
                & (df["entry"] == best["entry"]) & (df["W"] == best["W"]) & (df["tp"] == best["tp"])
                & (df["H"] == best["H"])]
        if best["dir"] != "both":
            ev = ev[(ev["direction"] > 0) == (best["dir"] == "drift")]
        if best["top3"] == "top3":
            ev = ev[ev["top3"]]
        for mode in modes[1:]:
            vals, cache = [], {}
            for e in ev.itertuples():
                p = [q for q in paths if os.path.basename(q).startswith(e.race)][0]
                if p not in cache:
                    tt = Tape.load(p)
                    ok = ~tt.suspended.copy()
                    ok[-1] = False
                    cache = {p: (tt, int(np.where(ok)[0].max()))}
                tt, end = cache[p]
                side = (BACK if e.direction > 0 else LAY) if e.action == "fade" else (LAY if e.direction > 0 else BACK)
                res = run_trade(tt, end, e.step, e.runner, side, e.entry, e.W, e.tp, e.H, fill_mode=mode)
                vals.append((e.race, e.split, res["filled"], res["pnl"]))
            v = pd.DataFrame(vals, columns=["race", "split", "filled", "pnl"])
            for sp, g in v.groupby("split"):
                ra = race_t(g["pnl"].fillna(0) * g["filled"], g["race"])
                print(f"best rule under fill mode '{mode}' ({sp}): fill rate {(g['filled'] > 0).mean() * 100:.0f}%, "
                      f"per attempt {ra['mean']:+.4f} (t {ra['t']:.2f})")
                verdict[f"{mode}_{sp}_per_attempt"] = ra["mean"]
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=str)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=str))
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
