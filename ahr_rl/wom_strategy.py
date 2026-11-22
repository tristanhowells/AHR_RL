"""P24: a strategy built around the weight-of-money (WOM) change signal.

P22 found one small directional signal that keeps its size out of sample: the 30s change in weight of money
(best 3 levels, back side share) had a within-race IC of about -0.05 with the next 60-120s back return and +0.05
with the lay return, on TRAIN and TEST days alike. Far too weak to pay for crossing in and out (~3.9% on liquid
runners), so this study puts it in the cheapest vehicle the simulator has: P15's resting orders (k = 1..3 ticks
better than the best price, waiting 60 / 120s), greened at BSP, which cost ~0.2% per attempt but lose their price
improvement to adverse selection. k = 0 (cross now, green at BSP) is kept as the crossing reference. The
question: does WOM change screen out the bad fills well enough to leave a profit?

Two exits, so the vehicle matches the signal's horizon:
  bsp    P15's: every fill greened at BSP (held to the off, up to 10 min)
  cross  out by crossing the spread 60 / 120s after the decision (the signal is about the next 30-120s; held to
         the off, a 60s signal washes out - the positive control shows it)

Signals, at P15's decision rows (every 20s in the last 10 min), live (no look-ahead):
  wom          weight of money, best 3 levels: back-side $ / (back + lay)
  wom_chg_10 / _30 / _60   its change over 10 / 30 / 60s
  wom1_chg_30  the same change at the best level only
  wom_z_30     wom_chg_30 / the runner's own typical 30s WOM change so far in the race (how unusual it is)
  combo        linear blend, fitted on TRAIN days to the next 30s move
Each is z-scored and oriented on TRAIN days so high = "about to shorten" (good for a resting back).

  0  positive control: synthetic races in which money builds on one side of the book for 30s before each price
     step that way (a planted WOM-leads-price signal); the same pipeline must find it, or a null means little
  A  predictive power: within-race rank IC with the next 30 / 60 / 120s move, TRAIN vs HOLDOUT, all / liquid
  B  does it shrink adverse selection? liquid runners, k = 1, hold 60s, by signal quintile (TRAIN cut-offs)
  C  rule search (P16's): signal x side x tail x k x hold x liquidity x threshold x rank on TRAIN days, scored
     once on HOLDOUT (VAL + TEST days), each against its baseline (same slice, no signal) and the within-race
     lift. "edge" = HOLDOUT mean > 0, t > 2, >= 20 races, > 0 without its best race, positive lift.
  D  if there is an edge, the rule is frozen to strategy.json (signal, cut-off values, side, k, hold, liquidity,
     rank) for a pre-registered forward test on races recorded after today, like P3c arm C.

    python -m ahr_rl.wom_strategy --tapes "data/tapes/*.npz" --orders runs/passive_entry/orders.parquet --out runs/wom
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .env import list_tapes, split_by_date
from .ladder import PRICES
from .passive_entry_study import add_values
from .passive_entry_study import sample_tape as sample_orders
from .rebalance_study import _prob_grid
from .signal_entry_study import THRESH, _pool, orient, quintile_table, race_ic, rule_search
from .tape import Tape

RAW = ("wom", "wom_chg_10", "wom_chg_30", "wom_chg_60", "wom1_chg_30", "wom_z_30")
SIGNALS = RAW + ("combo",)
FWD = (30, 60, 120)


def wom_rows(path: str, every_s: float = 20.0, last_s: float = 600.0):
    """WOM signals at P15's decision rows (same grid), plus the next 30 / 60 / 120s moves (for orienting / IC)."""
    t = Tape.load(path)
    if not t.went_in_play:
        return None
    race = os.path.basename(path).split(".npz")[0]
    p, valid, end, bt, lt = _prob_grid(t)
    ok = ~t.suspended[:, None] & t.active
    xb = pd.DataFrame(np.where(ok & (bt >= 0), PRICES[np.maximum(bt, 0)], np.nan)).ffill(limit=10).to_numpy()
    xl = pd.DataFrame(np.where(ok & (lt >= 0), PRICES[np.maximum(lt, 0)], np.nan)).ffill(limit=10).to_numpy()
    k = lambda sec: int(round(sec / t.dt))
    bs3, ls3 = t.back_size[:, :, :3].sum(-1), t.lay_size[:, :, :3].sum(-1)
    bs1, ls1 = t.back_size[:, :, 0], t.lay_size[:, :, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        wom = np.where(bs3 + ls3 > 0, bs3 / (bs3 + ls3), np.nan)
        wom1 = np.where(bs1 + ls1 > 0, bs1 / (bs1 + ls1), np.nan)
    d30 = wom - np.vstack([np.full((k(30), t.n_runners), np.nan), wom[:-k(30)]])
    dec = np.where((t.t_rel >= -last_s) & (t.t_rel <= -5.0) & ~t.suspended)[0][::max(1, k(every_s))]
    rows = []
    for s in dec:
        live = np.where(np.isfinite(p[s]))[0]
        if len(live) < 3:
            continue
        for r in live:
            rec = dict(race=race, runner=int(r), t=np.float32(t.t_rel[s]), wom=float(wom[s, r]))
            for W in (10, 30, 60):
                s0 = s - k(W)
                rec[f"wom_chg_{W}"] = float(wom[s, r] - wom[s0, r]) if s0 >= 0 else np.nan
            s0 = s - k(30)
            rec["wom1_chg_30"] = float(wom1[s, r] - wom1[s0, r]) if s0 >= 0 else np.nan
            past = d30[:s + 1, r]
            past = past[np.isfinite(past)]
            sd = float(np.std(past)) if len(past) >= 20 else np.nan
            rec["wom_z_30"] = float(rec["wom_chg_30"] / sd) if sd and np.isfinite(sd) and sd > 0 else np.nan
            for H in (60, 120):  # crossing-out prices: decided at s + H, filled a row later
                x = min(s + k(H) + 1, end)
                rec[f"exit_back_{H}"], rec[f"exit_lay_{H}"] = float(xb[x, r]), float(xl[x, r])
            for H in FWD:
                s2 = min(s + k(H), end)
                rec[f"fwd_{H}"] = (float(100 * np.log(p[s2, r] / p[s, r]))
                                   if np.isfinite(p[s2, r]) and s2 > s else np.nan)
            rows.append(rec)
    if not rows:
        return None
    d = pd.DataFrame(rows)
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    return d


def control_tape(seed: int, n_runners: int = 6, pre_s: float = 600.0, post_s: float = 60.0, dt: float = 0.5):
    """Random-walk races where, for 30s before each tick step, the money waiting on the side the price is about
    to move to grows (back side before a shortening, lay side before a drift): WOM change leads the price."""
    from .ladder import N_TICKS

    rng = np.random.default_rng(seed)
    T, R, K = int((pre_s + post_s) / dt), n_runners, 8
    lead = int(30 / dt)
    t_rel = (-pre_s + np.arange(T) * dt).astype(np.float32)
    base = rng.integers(150, 230, size=R)
    steps = np.where(rng.random((T, R)) < 0.02, rng.choice([-1, 1], size=(T, R)), 0)  # a tick step every ~25s
    steps[:lead] = 0
    mid = base + np.cumsum(steps, 0)
    mid = np.clip(mid, K + 2, N_TICKS - K - 3)
    bmult, lmult = np.ones((T, R)), np.ones((T, R))
    for x, r in zip(*np.nonzero(steps)):
        ramp = np.linspace(1.0, 3.0, lead)
        if steps[x, r] < 0:  # shortening next: backers' money builds
            bmult[x - lead:x, r] *= ramp
        else:
            lmult[x - lead:x, r] *= ramp
    bt = np.stack([mid - k for k in range(K)], -1).astype(np.int16)
    lt = np.stack([mid + 1 + k for k in range(K)], -1).astype(np.int16)
    bs = (rng.uniform(0.8, 1.2, (T, R, K)) * 40 * bmult[..., None]).astype(np.float32)
    ls = (rng.uniform(0.8, 1.2, (T, R, K)) * 40 * lmult[..., None]).astype(np.float32)
    tr = []
    for s_ in range(1, T):
        for r in range(R):
            if rng.random() < 0.3:
                tr.append((s_, r, bt[s_, r, 0] if rng.random() < 0.5 else lt[s_, r, 0], float(rng.uniform(1, 10))))
    tr = np.array(tr, np.float64).reshape(-1, 4)

    fair = PRICES[mid[-1]]
    bsp = (fair * (1.0 / fair).sum()).astype(np.float32)  # a 100% book at the last mid (sum of 1 / bsp = 1)
    return Tape(market_id=f"ctl.{seed}", name="", dt=dt, t_rel=t_rel, back_tick=bt, back_size=bs, lay_tick=lt,
                lay_size=ls, ltp_tick=mid.astype(np.int16), tv=np.cumsum(np.ones((T, R), np.float32), 0),
                spn=np.zeros((T, R), np.float32), active=np.ones((T, R), bool), suspended=np.zeros(T, bool),
                total_matched=np.arange(T, dtype=np.float32) * 10, trade_step=tr[:, 0].astype(np.int32),
                trade_runner=tr[:, 1].astype(np.int16), trade_tick=tr[:, 2].astype(np.int16),
                trade_vol=tr[:, 3].astype(np.float32), removal_step=np.zeros(0, np.int32),
                removal_runner=np.zeros(0, np.int16), removal_factor=np.zeros(0, np.float32),
                selection_ids=np.arange(R, dtype=np.int64), base_rate=8.0, went_in_play=True,
                winner=int(rng.integers(R)), bsp=bsp)


def control_paths(out_dir: str, n: int = 120) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(out_dir, f"2099{i % 30 + 1:04d}_1200_WomCtl_1_{i}.npz")
        if not os.path.exists(p):
            t = control_tape(90_000 + i)
            t.name = os.path.basename(p)[:-4]
            t.save(p)
        paths.append(p)
    return paths


def cross_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    """The same orders, each fill closed by crossing the spread 60 / 120s after the decision instead of at BSP:
    ev_attempt_H = filled share by H x the green at that exit; ev_fill = the 60s green (k = 0 crosses now)."""
    x = df.copy()
    back = (x["side"] == "back").to_numpy()
    P, c = x["price"].to_numpy(float), x["comm"].to_numpy(float)
    for H in (60, 120):
        g = np.where(back, P / x[f"exit_lay_{H}"].to_numpy(float) - 1, 1 - P / x[f"exit_back_{H}"].to_numpy(float))
        g = np.where(g > 0, g * (1 - c), g) * 100
        x[f"x_{H}"] = g
        x[f"ev_attempt_{H}"] = g * x[f"f{H}"].to_numpy(float)
    x["ev_fill"] = x["x_60"]
    return x


def sections_bc(df, train, cuts, min_races, out, vname, log):
    log("\n=== B. Does WOM change shrink adverse selection? Liquid runners, k = 1, hold 60s, by signal quintile ===")
    log("(Q5 = says shorten: good for resting backs; Q1 = drift: good for resting lays)")
    tb = pd.concat([quintile_table(df, c, s) for c in ("wom_chg_30", "wom_z_30", "combo") for s in ("back", "lay")],
                   ignore_index=True)
    tb.to_csv(os.path.join(out, f"B_quintiles_{vname}.csv"), index=False)
    for c in ("wom_chg_30", "wom_z_30", "combo"):
        log(f"\n{c}:")
        log(tb[tb["signal"] == c].drop(columns="signal").round(3).to_string(index=False))

    log(f"\n=== C. Rule search (WOM signal x side x tail x k x hold x liquidity x threshold x rank): TRAIN (>= "
        f"{min_races} races) -> HOLDOUT ===")
    rs = rule_search(df, train, cuts, min_races, signals=SIGNALS)
    rs.to_csv(os.path.join(out, f"C_rules_{vname}.csv"), index=False)
    if len(rs):
        log(rs[["signal", "side", "when", "k", "hold", "liquid", "top", "fav_rank", "train_mean", "train_t",
                "train_races", "holdout_mean", "holdout_t", "holdout_races", "holdout_drop_best"]].round(3)
            .to_string(index=False))
        log("\nthe same rules against their baseline (same slice, no WOM condition): the lift is what WOM adds")
        log(rs[["signal", "side", "when", "k", "hold", "top", "fav_rank", "train_baseline", "train_lift",
                "train_lift_t", "holdout_baseline", "holdout_lift", "holdout_lift_t"]].round(3).to_string(index=False))
    best = rs.iloc[0].to_dict() if len(rs) else {}
    edge = bool(best) and best["holdout_mean"] > 0 and (best["holdout_t"] or 0) > 2 and \
        best["holdout_races"] >= 20 and best["holdout_drop_best"] > 0 and best["holdout_lift"] > 0
    log(f"exit {vname}: " + ("EDGE on HOLDOUT, beating its baseline" if edge else "no edge on HOLDOUT"))
    return rs, best, edge


def analyse(paths, orders_path, out, a, log, t0, min_races: int) -> dict:
    """Sections A-D on one set of races; returns the verdict (strategy.json written on an edge)."""
    os.makedirs(out, exist_ok=True)
    _, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    kw = dict(every_s=a.every_s, last_s=a.last_s)
    if orders_path and os.path.exists(orders_path):
        orders = pd.read_parquet(orders_path)
        log(f"loaded {len(orders):,} orders from {orders_path}")
    else:
        orders = _pool(sample_orders, paths, kw, a.workers, t0, "orders")
        orders.to_parquet(os.path.join(out, "orders.parquet"), index=False)
    scache = os.path.join(out, "wom_signals.parquet")
    sig = pd.read_parquet(scache) if os.path.exists(scache) else None
    if sig is not None and not set(orders["race"].unique()) <= set(sig["race"].unique()):
        log(f"cached signals miss {len(set(orders['race'].unique()) - set(sig['race'].unique()))} of the orders' "
            "races (recorded since the cache was built): rebuilding them")
        sig = None
    if sig is None or any(c not in sig for c in RAW):
        sig = _pool(wom_rows, paths, kw, a.workers, t0, "signals")
        sig.to_parquet(scache, index=False)
    sig["day"] = sig["race"].str[:8]
    train_s = ~sig["day"].isin(hold_days).to_numpy()
    info = orient(sig, train_s, raw=RAW, target="fwd_30")

    df = add_values(orders.copy())
    df["t"] = df["t"].astype(np.float32)
    sig["t"] = sig["t"].astype(np.float32)
    xcols = [f"exit_{s_}_{H}" for s_ in ("back", "lay") for H in (60, 120)]
    df = df.merge(sig[["race", "runner", "t"] + [f"o_{c}" for c in SIGNALS] + xcols], on=["race", "runner", "t"],
                  how="inner")
    train = ~df["day"].isin(hold_days)
    df["is_train"] = train
    cut = (float(np.nanquantile(df.loc[train & (df["k"] == 0) & (df["t"] >= -300), "best_usd"], 0.8))
           if a.liq_cut == "auto" else float(a.liq_cut))
    df["liquid"] = df["best_usd"] >= cut
    n_tr, n_ho = df.loc[train, "race"].nunique(), df.loc[~train, "race"].nunique()
    log(f"{len(df):,} orders with WOM signals, {df['race'].nunique()} races ({n_tr} TRAIN / {n_ho} HOLDOUT); "
        f"liquid = >= ${cut:.2f} at the best prices  [{time.time() - t0:.0f}s]")
    log("signals oriented on TRAIN days: high = the runner is about to SHORTEN (good for a resting back)\n")

    log("=== A. Predictive power: within-race rank IC with the next 30 / 60 / 120s move (oriented signal) ===")
    sig_liq = sig.merge(df.loc[df["k"] == 0, ["race", "runner", "t", "liquid"]].drop_duplicates(),
                        on=["race", "runner", "t"], how="left")
    rows = []
    for c in SIGNALS:
        r = {"signal": c, "orientation": "+" if c == "combo" or info[c]["sign"] > 0 else "-"}
        for nm, m in (("train", train_s), ("holdout", ~train_s)):
            d = sig_liq[m]
            for H in FWD:
                r[f"{nm}_ic_{H}"] = race_ic(d["o_" + c], d[f"fwd_{H}"], d["race"])
            dl = d[d["liquid"] == True]  # noqa: E712
            r[f"{nm}_ic_30_liquid"] = race_ic(dl["o_" + c], dl["fwd_30"], dl["race"])
        rows.append(r)
    ta = pd.DataFrame(rows)
    ta.to_csv(os.path.join(out, "A_ic.csv"), index=False)
    log(ta.round(4).to_string(index=False))
    log(f"combo weights (oriented, TRAIN): {json.dumps({k: round(v, 4) for k, v in info['combo']['weights'].items()})}")

    cuts, base = {}, df[(df["k"] == 0) & (df["side"] == "back")]
    for c in SIGNALS:
        o = df["o_" + c]
        edges = np.unique(np.nanquantile(base.loc[train & base["liquid"], "o_" + c], [0.2, 0.4, 0.6, 0.8]))
        labels = [f"Q{i + 1}" for i in range(len(edges) + 1)]
        labels[0], labels[-1] = labels[0] + " drift", labels[-1] + " shorten"
        df["q_" + c] = pd.cut(o, [-np.inf, *edges, np.inf], labels=labels)
        for lq in (False, True):
            b = base.loc[train & (base["liquid"] if lq else True), "o_" + c]
            for th in THRESH:
                cuts[(c, lq, th)] = (float(np.nanquantile(b, 1 - th)), float(np.nanquantile(b, th)))
            cuts[(c, lq, "quiet")] = (float(np.nanquantile(b, 0.4)), float(np.nanquantile(b, 0.6)))

    vehicles = {"bsp": df, "cross": cross_vehicle(df)}
    best_by, edge_by, rs_by = {}, {}, {}
    for vname, vdf in vehicles.items():
        log(f"\n{'#' * 30} exit: {'green at BSP (held to the off)' if vname == 'bsp' else 'cross out 60 / 120s after the decision'} {'#' * 30}")
        rs, best, edge = sections_bc(vdf, train, cuts, min_races, out, vname, log)
        best_by[vname], edge_by[vname], rs_by[vname] = best, edge, rs
    vbest = max(best_by, key=lambda v: (edge_by[v], np.nan_to_num(best_by[v].get("holdout_mean", -99), nan=-99)))
    best, edge = best_by[vbest], edge_by[vbest]
    verdict = dict(liq_cut=cut, edge=edge, best_exit=vbest, best_rule=best, orientation=info,
                   by_exit={v: dict(edge=edge_by[v], best_rule=best_by[v]) for v in vehicles},
                   note="rule search over many slices: a pass is a hypothesis for a forward test")
    if edge:
        c = best["signal"]
        lq = bool(best["liquid"])
        th = {"20%": 0.20, "10%": 0.10, "5%": 0.05}.get(best["top"])
        frozen = dict(registered=time.strftime("%Y%m%d"), signal=c, side=best["side"], when=best["when"],
                      k=int(best["k"]), hold_s=int(best["hold"]), liquid_only=lq, liq_cut=cut,
                      fav_rank=best["fav_rank"], orientation=info.get(c),
                      cut_values=cuts[(c, lq, th)] if th else cuts[(c, lq, "quiet")],
                      exit="green at BSP" if vbest == "bsp" else f"cross out {int(best['hold'])}s after the decision",
                      stake=10.0,
                      decide=("forward test on races recorded after the registration day; >= 100 races and 300 "
                              "attempts before judging, as P3c arm C"))
        json.dump(frozen, open(os.path.join(out, "strategy.json"), "w"), indent=1, default=float)
        verdict["frozen_strategy"] = frozen
    log("\n=== D. Verdict ===")
    if edge:
        log("EDGE: the best TRAIN rule holds up on HOLDOUT and beats its baseline -> frozen to strategy.json for a "
            "pre-registered forward test")
    else:
        log("NO EDGE: the best WOM rule does not hold up on HOLDOUT (or does not beat its baseline), even in the "
            "cheapest vehicle (resting entry, green at BSP)")
    return verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orders", default=None, help="P15's orders.parquet (else simulated here, ~20 min)")
    ap.add_argument("--every-s", type=float, default=20.0, help="must match the P15 run")
    ap.add_argument("--last-s", type=float, default=600.0, help="must match the P15 run")
    ap.add_argument("--liq-cut", default="auto")
    ap.add_argument("--min-races", type=int, default=100)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    logf = open(os.path.join(a.out, "log.txt"), "w")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(str(msg) + "\n")
        logf.flush()  # so a run that dies part-way leaves its log on Drive

    verdict = {}
    if not a.skip_control:
        log("=== 0. Positive control: synthetic races where WOM change leads each price step ===")
        cp = control_paths(os.path.join(a.out, "control_tapes"))
        cv = analyse(cp, None, os.path.join(a.out, "control"), a, log, t0, min_races=10)
        verdict["control_found_signal"] = bool(cv["edge"])
        log("control: the pipeline FINDS the planted WOM signal" if cv["edge"] else
            "control: NOT found -> a null result on real races means little")
        log("\n" + "=" * 100 + "\nREAL RACES\n" + "=" * 100)
    verdict.update(analyse(list_tapes(a.tapes), a.orders, a.out, a, log, t0, min_races=a.min_races))
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    log(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
