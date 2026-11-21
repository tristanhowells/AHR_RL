"""P25: green by trading the FIELD instead of the runner - back (or lay) every runner so every outcome pays the
same, and try to complete the book below 100% (backs) / above 100% (lays) before the start.

A back on every runner, stake_i = Q / P_i, returns Q whichever runner wins; it is green when the stakes add up to
less than Q, i.e. when the back book (sum of 1 / P_i over the prices actually got) is under 100%. Laying every
runner for backer's stake Q / L_i costs Q whichever runner wins and collects the stakes: green when the lay book is
over 100%. Betfair's own cross-matching (P14) already closes ready-made arbitrage at the best prices, so the
question is whether the book can be FORCED by resting every leg k ticks better than the best price and waiting.

  A  the book as quoted: back book (sum of 1 / best back) and lay book (sum of 1 / best lay) every 20s in the last
     10 min: how often an instant arbitrage exists (back book < 100%, lay book > 100%), by time to the start and
     field size
  B  forced dutch: at a decision, every runner's leg rests k = 0..3 ticks better (k = 0: cross them all now), stakes
     sized for an equal payout Q = $100; fills from P15's passive-fill simulation (queue, traded volume, the book
     crossing, cross-matching); after H = 60 / 120s every unfilled part is completed by crossing at the best price
     then. Green % = (Q - stakes) / Q for backs, (stakes - Q) / Q for lays, commission on a profit; and how often
     every leg filled. The unfilled legs are where adverse selection bites: the legs that do NOT fill are the ones
     whose price moved away, completed at a worse price.
  C  the hindsight ceiling: every runner's leg at its best price within the next 120s (on the 20s grid), crossing
  D  rule search on TRAIN days (side x k x H x time to the start x field size x quoted book), the best TRAIN rule
     scored once on HOLDOUT days (race-clustered); "edge" = HOLDOUT mean > 0, t > 2, >= 20 races, > 0 without
     its best race
  0  positive control first: synthetic races with a fair 100% book whose runners wobble +-1 tick independently;
     resting every leg above the fair price completes a sub-100% book, which B / D must find

Inputs: P15-style simulated orders and P24's signal file (it holds the best prices 60 / 120s after each
decision). Pass the P24 run's files (--orders / --signals); missing ones are rebuilt here.

    python -m ahr_rl.dutch_study --tapes "data/tapes/*.npz" --orders runs/wom/orders.parquet \\
        --signals runs/wom/wom_signals.parquet --out runs/dutch
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from .env import list_tapes, split_by_date
from .passive_entry_study import rt
from .passive_entry_study import sample_tape as sample_orders
from .signal_entry_study import _pool
from .wom_strategy import wom_rows

Q = 100.0
HS = (60, 120)
KS = (0, 1, 2, 3)


def book_table(orders: pd.DataFrame) -> pd.DataFrame:
    """Per (race, t): the field, the quoted back and lay books (sum of 1 / best price at the decision)."""
    b = orders[(orders["side"] == "back") & (orders["k"] == 1)]  # one row per runner and decision
    g = b.groupby(["race", "t"], observed=True)
    out = g.agg(field=("field", "first"), legs=("runner", "size"), comm=("comm", "first"),
                back_book=("rb_p", lambda x: float((1.0 / x).sum())), lay_book=("rl_p", lambda x: float((1.0 / x).sum())),
                min_best_usd=("best_usd", "min")).reset_index()
    out["complete"] = out["legs"] == out["field"]
    return out


def forced_dutch(orders: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """Every (race, t, side, k, H): the green of a full-field equal-payout book, legs resting k ticks better,
    unfilled parts completed by crossing H seconds after the decision."""
    x = orders.merge(sig[["race", "runner", "t", "exit_back_60", "exit_lay_60", "exit_back_120", "exit_lay_120"]],
                     on=["race", "runner", "t"], how="inner")
    rows = []
    for (side, k), g in x.groupby(["side", "k"], observed=True):
        P = g["price"].to_numpy(float)
        for H in HS:
            f = np.ones(len(g)) if k == 0 else np.clip(g[f"f{H}"].to_numpy(float), 0, 1)
            X = g[f"exit_{'back' if side == 'back' else 'lay'}_{H}"].to_numpy(float)  # completion price
            need = f < 1 - 1e-9
            stake = f * Q / P + np.where(need, (1 - f) * Q / X, 0.0)  # NaN if a leg cannot be completed
            d = pd.DataFrame({"race": g["race"].to_numpy(), "t": g["t"].to_numpy(), "stake": stake,
                              "filled": ~need, "comm": g["comm"].to_numpy()})
            agg = d.groupby(["race", "t"], observed=True).agg(stakes=("stake", lambda s: s.sum(min_count=len(s))),
                                                              legs=("stake", "size"), all_filled=("filled", "all"),
                                                              unfilled=("filled", lambda s: int((~s).sum())),
                                                              comm=("comm", "first")).reset_index()
            gross = (Q - agg["stakes"]) if side == "back" else (agg["stakes"] - Q)
            agg["green_pct"] = np.where(gross > 0, gross * (1 - agg["comm"]), gross) / Q * 100
            agg["side"], agg["k"], agg["H"] = side, int(k), int(H)
            rows.append(agg.drop(columns=["stakes", "comm"]))
    return pd.concat(rows, ignore_index=True)


def hindsight_ceiling(orders: pd.DataFrame, window_s: float = 120.0) -> pd.DataFrame:
    """Each leg at its best quoted price over the next window_s (on the decision grid), crossing: back book uses
    each runner's highest best back, lay book its lowest best lay."""
    b = orders[(orders["side"] == "back") & (orders["k"] == 1)][["race", "t", "runner", "rb_p", "rl_p", "field", "comm"]]
    rows = []
    for race, g in b.groupby("race", observed=True):
        ts = np.sort(g["t"].unique())
        piv_b = g.pivot_table(index="t", columns="runner", values="rb_p")
        piv_l = g.pivot_table(index="t", columns="runner", values="rl_p")
        for t0 in ts:
            w = (piv_b.index >= t0) & (piv_b.index <= t0 + window_s)
            first = g[g["t"] == t0]
            if len(first) != int(first["field"].iloc[0]):
                continue
            cols = first["runner"].to_numpy()
            bb = piv_b.loc[w, cols].max().to_numpy()
            ll = piv_l.loc[w, cols].min().to_numpy()
            c = float(first["comm"].iloc[0])
            gb = Q - (Q / bb).sum()
            gl = (Q / ll).sum() - Q
            rows.append(dict(race=race, t=t0, back_green_pct=(gb * (1 - c) if gb > 0 else gb) / Q * 100,
                             lay_green_pct=(gl * (1 - c) if gl > 0 else gl) / Q * 100))
    return pd.DataFrame(rows)


def add_buckets(d: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    d = d.merge(books[["race", "t", "field", "complete", "back_book", "lay_book", "min_best_usd"]], on=["race", "t"],
                how="left")
    d = d[d["complete"] == True].copy()  # noqa: E712 - every active runner has a leg
    d["time_b"] = pd.cut(-d["t"], [0, 60, 180, 360, 601], labels=["<1m", "1-3m", "3-6m", "6-10m"])
    d["field_b"] = pd.cut(d["field"], [0, 6, 9, 12, 40], labels=["<=6", "7-9", "10-12", "13+"])
    gap = np.where(d["side"] == "back", d["back_book"] - 1, 1 - d["lay_book"])  # how far the quote is from arbitrage
    d["gap_b"] = pd.cut(gap * 100, [-100, 0, 1, 2, 4, 1e9], labels=["arb", "<1%", "1-2%", "2-4%", "4%+"])
    return d


def rule_search(d: pd.DataFrame, train_races: set, min_races: int, top: int = 15) -> pd.DataFrame:
    rows = []
    tr = d[d["race"].isin(train_races)]
    ho = d[~d["race"].isin(train_races)]
    dims = dict(time_b=[None, "<1m", "1-3m", "3-6m", "6-10m"], field_b=[None, "<=6", "7-9", "10-12", "13+"],
                gap_b=[None, "arb", "<1%", "1-2%", "2-4%"])
    for (side, k, H), g in tr.groupby(["side", "k", "H"], observed=True):
        h = ho[(ho["side"] == side) & (ho["k"] == k) & (ho["H"] == H)]
        for tb, fb, gb in itertools.product(*dims.values()):
            m, mh = np.ones(len(g), bool), np.ones(len(h), bool)
            for col, v in (("time_b", tb), ("field_b", fb), ("gap_b", gb)):
                if v is not None:
                    m &= (g[col] == v).to_numpy()
                    mh &= (h[col] == v).to_numpy()
            s = rt(g.loc[m, "green_pct"], g.loc[m, "race"])
            if s["races"] < min_races:
                continue
            rows.append(dict(side=side, k=k, H=H, time=tb or "all", field=fb or "all", quoted_gap=gb or "all",
                             attempts=int(m.sum()), all_filled_pct=float(g.loc[m, "all_filled"].mean() * 100),
                             train_mean=s["mean"], train_t=s["t"], train_races=s["races"], _h=h[mh]))
    rows = sorted(rows, key=lambda r: -np.nan_to_num(r["train_t"], nan=-99))
    seen, uniq = set(), []
    for r in rows:  # a filter that keeps the same attempts as a wider one is the same rule
        key = (r["side"], r["k"], r["H"], r["attempts"], round(r["train_mean"], 9))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    rows = uniq[:top]
    for r in rows:
        h = r.pop("_h")
        s = rt(h["green_pct"], h["race"]) if len(h) else dict(mean=np.nan, t=np.nan, races=0)
        per = h.groupby("race")["green_pct"].mean() if len(h) else pd.Series(dtype=float)
        r.update(holdout_mean=s["mean"], holdout_t=s["t"], holdout_races=s["races"],
                 holdout_drop_best=float(per.drop(per.idxmax()).mean()) if len(per) > 1 else np.nan)
    return pd.DataFrame(rows)


def analyse(paths, orders_path, signals_path, out, workers, log, t0, min_races) -> dict:
    os.makedirs(out, exist_ok=True)
    kw = dict(every_s=20.0, last_s=600.0)
    if orders_path and os.path.exists(orders_path):
        orders = pd.read_parquet(orders_path)
        log(f"loaded {len(orders):,} orders from {orders_path}")
    else:
        orders = _pool(sample_orders, paths, kw, workers, t0, "orders")
        orders.to_parquet(os.path.join(out, "orders.parquet"), index=False)
    if signals_path and os.path.exists(signals_path):
        sig = pd.read_parquet(signals_path)
        log(f"loaded {len(sig):,} signal rows from {signals_path}")
    else:
        sig = _pool(wom_rows, paths, kw, workers, t0, "signals")
        sig.to_parquet(os.path.join(out, "wom_signals.parquet"), index=False)
    for d_ in (orders, sig):
        d_["t"] = d_["t"].astype(np.float32)
    books = book_table(orders)
    comp = books[books["complete"]]
    log(f"{books['race'].nunique()} races, {len(books):,} decisions; every active runner has a leg in "
        f"{books['complete'].mean() * 100:.0f}%  [{time.time() - t0:.0f}s]")

    log("\n=== A. The book as quoted (complete fields): how often is there an instant arbitrage? ===")
    comp = comp.assign(time_b=pd.cut(-comp["t"], [0, 60, 180, 360, 601], labels=["<1m", "1-3m", "3-6m", "6-10m"]),
                       field_b=pd.cut(comp["field"], [0, 6, 9, 12, 40], labels=["<=6", "7-9", "10-12", "13+"]))
    ta = []
    for col in ("time_b", "field_b"):
        for lv, g in comp.groupby(col, observed=True):
            ta.append(dict(by=col, level=str(lv), decisions=len(g), races=g["race"].nunique(),
                           back_book_median_pct=g["back_book"].median() * 100,
                           back_arb_pct=(g["back_book"] < 1).mean() * 100,
                           lay_book_median_pct=g["lay_book"].median() * 100,
                           lay_arb_pct=(g["lay_book"] > 1).mean() * 100))
    ta = pd.DataFrame(ta)
    ta.to_csv(os.path.join(out, "A_books.csv"), index=False)
    log(ta.round(3).to_string(index=False))

    log("\n=== B. Forced dutch: every leg rests k ticks better, unfilled parts completed by crossing after H ===")
    fd = add_buckets(forced_dutch(orders, sig), books).dropna(subset=["green_pct"])
    tb = []
    for (side, k, H), g in fd.groupby(["side", "k", "H"], observed=True):
        s = rt(g["green_pct"], g["race"])
        tb.append(dict(side=side, k=k, H=H, attempts=len(g), races=s["races"], mean_green_pct=s["mean"], t=s["t"],
                       pct_green=(g["green_pct"] > 0).mean() * 100, all_legs_filled_pct=g["all_filled"].mean() * 100,
                       green_when_all_filled=g.loc[g["all_filled"], "green_pct"].mean(),
                       green_when_completed=g.loc[~g["all_filled"], "green_pct"].mean(),
                       mean_unfilled_legs=g["unfilled"].mean()))
    tb = pd.DataFrame(tb)
    tb.to_csv(os.path.join(out, "B_forced_dutch.csv"), index=False)
    log("(green % of the $100 payout; 'when completed' = at least one leg had to be finished by crossing)")
    log(tb.round(3).to_string(index=False))

    log("\n=== C. Hindsight ceiling: every leg at its best quoted price within the next 120s, crossing ===")
    hc = hindsight_ceiling(orders)
    if len(hc):
        tc = pd.DataFrame([dict(side=s_, decisions=len(hc), mean_green_pct=hc[f"{s_}_green_pct"].mean(),
                                pct_green=(hc[f"{s_}_green_pct"] > 0).mean() * 100) for s_ in ("back", "lay")])
        tc.to_csv(os.path.join(out, "C_hindsight.csv"), index=False)
        log(tc.round(3).to_string(index=False))

    days = sorted(fd["race"].str[:8].unique())
    tr_r, va_r, te_r = split_by_date(sorted(fd["race"].unique()))
    log(f"\n=== D. Rule search (side x k x H x time x field x quoted gap to arbitrage): TRAIN {len(tr_r)} races -> "
        f"HOLDOUT {len(va_r) + len(te_r)} races ===")
    rs = rule_search(fd, set(tr_r), min_races)
    rs.to_csv(os.path.join(out, "D_rules.csv"), index=False)
    if len(rs):
        log(rs.round(3).to_string(index=False))
    best = rs.iloc[0].to_dict() if len(rs) else {}
    edge = bool(best) and best["holdout_mean"] > 0 and (best["holdout_t"] or 0) > 2 and \
        best["holdout_races"] >= 20 and (best["holdout_drop_best"] or -1) > 0
    log("EDGE: the best TRAIN rule completes a green book on HOLDOUT days" if edge else
        "NO EDGE: no forced field book completes green on HOLDOUT days")
    return dict(edge=edge, best_rule=best, days=len(days),
                instant_back_arb_pct=float((comp["back_book"] < 1).mean() * 100),
                instant_lay_arb_pct=float((comp["lay_book"] > 1).mean() * 100))


def control_tape(seed: int):
    """A bounce race (each runner wobbles +-1 tick around its fair tick independently, best back / lay one tick
    either side) whose fair ticks form a ~100% book: a back resting above the fair price, or a lay below it, fills
    on the wobble, and a full set of them is a sub-100% back book / over-100% lay book."""
    from .ladder import N_TICKS, PRICES, price_to_tick
    from .synthetic import make_bounce_tape

    rng = np.random.default_rng(seed)
    t = make_bounce_tape(seed)
    R = t.n_runners
    q = rng.dirichlet(np.full(R, 3.0))
    fair = np.array([price_to_tick(float(np.clip(1 / x, 1.6, 40))) for x in q])
    for _ in range(50):  # nudge the tick-rounded fair book to within a fraction of a percent of 100%
        book = (1 / PRICES[fair]).sum()
        if abs(book - 1) < 0.004:
            break
        r = int(rng.integers(R))
        fair[r] += 1 if book > 1 else -1
    fair = np.clip(fair, 10, N_TICKS - 12)
    centre = np.median(t.ltp_tick, 0).astype(int)  # the bounce tape's own fair ticks
    shift = fair - centre
    t.back_tick = (t.back_tick + shift[None, :, None]).astype(np.int16)
    t.lay_tick = (t.lay_tick + shift[None, :, None]).astype(np.int16)
    t.ltp_tick = (t.ltp_tick + shift[None, :]).astype(np.int16)
    t.trade_tick = (t.trade_tick + shift[t.trade_runner.astype(int)]).astype(np.int16)
    fp = PRICES[fair]
    t.bsp = (fp * (1.0 / fp).sum()).astype(np.float32)  # sum of 1 / BSP = 1
    t.winner = int(rng.choice(R, p=(1 / t.bsp) / (1 / t.bsp).sum()))
    return t


def control_paths(out_dir: str, n: int = 120) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(out_dir, f"2099{i % 30 + 1:04d}_1200_DutchCtl_1_{i}.npz")
        if not os.path.exists(p):
            t = control_tape(60_000 + i)
            t.name = os.path.basename(p)[:-4]
            t.save(p)
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orders", default=None, help="P24's (or P15's) orders.parquet")
    ap.add_argument("--signals", default=None, help="P24's wom_signals.parquet (best prices after 60 / 120s)")
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

    verdict = {}
    if not a.skip_control:
        log("=== 0. Positive control: fair 100% books whose runners wobble +-1 tick independently ===")
        cv = analyse(control_paths(os.path.join(a.out, "control_tapes")), None, None, os.path.join(a.out, "control"),
                     a.workers, log, t0, min_races=10)
        verdict["control_found_edge"] = cv["edge"]
        log("control: a forced book above the fair prices IS found green" if cv["edge"] else
            "control: NOT found -> a null result on real races means little")
        log("\n" + "=" * 100 + "\nREAL RACES\n" + "=" * 100)
    verdict.update(analyse(list_tapes(a.tapes), a.orders, a.signals, a.out, a.workers, log, t0, a.min_races))
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    log(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
