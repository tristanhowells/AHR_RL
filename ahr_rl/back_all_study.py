"""P21: back (or lay) EVERY runner N minutes before the advertised start, then green each one up just before the start.

For each race and each entry time (default 9, 5, 4, 3, 2 and 1 minutes before the advertised start):
  entry  the first live row at or after the entry time: every active runner with a two-sided book is backed
         (or laid) for --stake (backer's stake), crossing the spread and walking the visible ladder (VWAP)
  hedge  the last live row at or before --hedge-s seconds before the advertised start: each runner gets the
         opposite bet sized so it pays the same whether that runner wins or loses (an unbiased green-up),
             back S at P_in  ->  lay  S * P_in / P_out at P_out   (green = S * (P_in / P_out - 1))
             lay  S at P_in  ->  back S * P_in / P_out at P_out   (green = S * (1 - P_in / P_out))
         again walking the visible ladder
Settlement: the market's P&L in every outcome (each runner winning), commission (the race's base rate) on a
positive net market result only. "green" = the worst outcome after commission (the profit locked in);
"realised" = the actual winner's outcome where the result is known.

green_touch = every green-up filled in full at the best hedge price: the theory itself, separated from whether
the visible ladder 10s out can take the green-up (on long shots it often cannot: a $10 lay of a 200 shot is a
$1,990 liability, and what the ladder can't take stays open, so "green" - the worst outcome - is dominated by it).
Split of green_touch, per runner and summed per race:
  drift   what the mid price move alone paid: back S * (mid_in / mid_out - 1), lay S * (1 - mid_in / mid_out)
  spread  the rest: the cost of crossing the spread twice (and walking the entry ladder) - always negative
Backing everything wins when prices shorten on average between entry and hedge and loses when they drift; laying
everything is the mirror image, so "back+lay" (both at once) is the pure round-trip cost.

Races with a scratching between entry and hedge are skipped for that entry time (reduction factors would
change the matched prices). Means and t are per race (each race counts once).

    python -m ahr_rl.back_all_study --tapes "data/tapes/*.npz" --out runs/back_all
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .env import list_tapes
from .ladder import PRICES
from .tape import Tape

ENTRIES_S = (540, 300, 240, 180, 120, 60)


def rt(v, race) -> dict:
    """Race-clustered: mean of per-race means, its t, and the number of races (as passive_entry_study.rt)."""
    per = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna().groupby("race")["v"].mean()
    if len(per) < 2 or per.std(ddof=1) == 0:
        return dict(mean=float(per.mean()) if len(per) else np.nan, t=np.nan, races=len(per))
    return dict(mean=float(per.mean()), t=float(per.mean() / (per.std(ddof=1) / np.sqrt(len(per)))), races=len(per))


def walk(ticks: np.ndarray, sizes: np.ndarray, stake: float) -> tuple[float, float]:
    """Take up to `stake` (backer's stake) from one side of the ladder, best level first -> (filled, VWAP odds)."""
    got, cost = 0.0, 0.0
    for tk, sz in zip(ticks, sizes):
        if tk < 0 or sz <= 0 or got >= stake - 1e-9:
            break
        x = min(float(sz), stake - got)
        got += x
        cost += x * PRICES[int(tk)]
    return got, (cost / got if got > 0 else np.nan)


def runner_trade(t: Tape, r: int, s_in: int, s_out: int, side: str, stake: float) -> dict | None:
    """One runner: enter at row s_in, unbiased green-up at row s_out. P&L if it wins / loses."""
    bt_in, bs_in, lt_in, ls_in = t.back_tick[s_in, r], t.back_size[s_in, r], t.lay_tick[s_in, r], t.lay_size[s_in, r]
    bt_out, bs_out, lt_out, ls_out = t.back_tick[s_out, r], t.back_size[s_out, r], t.lay_tick[s_out, r], t.lay_size[s_out, r]
    if min(bt_in[0], lt_in[0], bt_out[0], lt_out[0]) < 0:
        return None
    mid_in = np.sqrt(PRICES[bt_in[0]] * PRICES[lt_in[0]])
    mid_out = np.sqrt(PRICES[bt_out[0]] * PRICES[lt_out[0]])
    if side == "back":  # back now at the best back prices, lay later at the best lay prices
        s, p_in = walk(bt_in, bs_in, stake)
        o_ticks, o_sizes = lt_out, ls_out
    else:  # lay now at the best lay prices, back later at the best back prices
        s, p_in = walk(lt_in, ls_in, stake)
        o_ticks, o_sizes = bt_out, bs_out
    if s <= 0:
        return None
    p_touch = PRICES[o_ticks[0]]  # best hedge price, ignoring how much is there
    p_out, want, h = p_touch, s * p_in / p_touch, 0.0
    for _ in range(6):  # the hedge's own VWAP depends on its size
        want = s * p_in / p_out
        h, p2 = walk(o_ticks, o_sizes, want)
        if h <= 0 or not np.isfinite(p2):
            break
        if h < want - 1e-9:  # not enough on the ladder: a partial green-up, the rest stays open
            p_out = p2
            break
        if abs(p2 - p_out) < 1e-9:
            break
        p_out = p2
    want = s * p_in / p_out
    h = min(h, want)
    if side == "back":
        win, lose = s * (p_in - 1) - h * (p_out - 1), -s + h
        drift = s * (mid_in / mid_out - 1)
        touch = s * (p_in / p_touch - 1)
    else:
        win, lose = -s * (p_in - 1) + h * (p_out - 1), s - h
        drift = s * (1 - mid_in / mid_out)
        touch = s * (1 - p_in / p_touch)
    return dict(runner=r, stake=s, p_in=p_in, p_out=p_out, hedge=h, win=win, lose=lose, drift=drift,
                touch=touch, hedge_frac=float(h / want) if want > 0 else 0.0, mid_in=mid_in, mid_out=mid_out,
                full_hedge=bool(h >= want * (1 - 1e-6)))


def race_rows(path: str, entries=ENTRIES_S, hedge_s: float = 10.0, stake: float = 10.0) -> pd.DataFrame | None:
    t = Tape.load(path)
    if not t.went_in_play:
        return None
    race = os.path.basename(path).split(".npz")[0]
    live = ~t.suspended
    comm = t.base_rate / 100.0
    out_rows = np.where(live & (t.t_rel <= -hedge_s))[0]
    if not len(out_rows):
        return None
    s_out = int(out_rows[-1])
    rows, runner_rows = [], []
    for e in entries:
        cand = np.where(live & (t.t_rel >= -e))[0]
        if not len(cand) or cand[0] >= s_out or t.t_rel[cand[0]] > -e + 5.0:
            continue
        s_in = int(cand[0])
        if ((t.removal_step > s_in) & (t.removal_step <= s_out)).any():
            continue  # a scratching in between: reduction factors would change the matched prices
        act = t.active[s_in] & t.active[s_out]
        rank = np.full(t.n_runners, 99)
        bt0, lt0 = t.back_tick[s_in, :, 0], t.lay_tick[s_in, :, 0]
        ok = act & (bt0 >= 0) & (lt0 >= 0)
        if ok.sum() < 2:
            continue
        mid = np.where(ok, np.sqrt(PRICES[np.maximum(bt0, 0)] * PRICES[np.maximum(lt0, 0)]), np.inf)
        rank[ok] = mid[ok].argsort().argsort() + 1
        for side in ("back", "lay"):
            tr = [x for x in (runner_trade(t, r, s_in, s_out, side, stake) for r in np.where(ok)[0]) if x]
            if len(tr) < 2:
                continue
            d = pd.DataFrame(tr)
            W, L = d["win"].to_numpy(), d["lose"].to_numpy()
            # market P&L if runner i wins: its win P&L plus every other runner's lose P&L; plus the outcome where a
            # runner we hold nothing on wins (an active runner without a two-sided book)
            outcomes = list(L.sum() - L + W)
            if (t.active[s_out] & ~np.isin(np.arange(t.n_runners), d["runner"])).any():
                outcomes.append(L.sum())
            net = lambda x: x * (1 - comm) if x > 0 else x
            green = min(net(x) for x in outcomes)
            realised = np.nan
            if 0 <= t.winner < t.n_runners:
                w = d.index[d["runner"] == t.winner]
                realised = net(L.sum() - L[w[0]] + W[w[0]]) if len(w) else net(L.sum())
            full = d["full_hedge"].to_numpy()
            rows.append(dict(race=race, day=race[:8], side=side, entry_min=e / 60, runners=len(d),
                             green_touch=net(float(d["touch"].sum())),
                             green_hedged=net(float(((d["win"] + d["lose"]) / 2)[full].sum())),
                             staked_hedged=float(d["stake"][full].sum()),
                             open_liability=float(-np.minimum(d["win"], d["lose"])[~full].sum()),
                             staked=float(d["stake"].sum()), green=green, realised=realised,
                             drift=float(d["drift"].sum()), spread=float(d["touch"].sum() - d["drift"].sum()),
                             full_hedges=float(d["full_hedge"].mean()), sub_min_hedges=int((d["hedge"] < 5.0).sum()),
                             t_in=float(t.t_rel[s_in]), t_out=float(t.t_rel[s_out])))
            d["green"] = (d["win"] + d["lose"]) / 2  # per runner, before commission (win == lose when fully hedged)
            d["rank"] = rank[d["runner"].to_numpy()]
            d["race"], d["side"], d["entry_min"] = race, side, e / 60
            runner_rows.append(d[["race", "side", "entry_min", "runner", "rank", "stake", "p_in", "p_out", "green",
                                  "touch", "drift", "hedge_frac", "full_hedge"]])
    if not rows:
        return None
    return pd.DataFrame(rows), pd.concat(runner_rows, ignore_index=True)


def _one(args):
    path, kw = args
    try:
        return race_rows(path, **kw)
    except Exception as e:
        print(f"skip {os.path.basename(path)}: {e}", flush=True)
        return None


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (side, e), g in df.groupby(["side", "entry_min"]):
        r = rt(g["green"], g["race"])
        rt_ = rt(g["green_touch"], g["race"])
        rows.append(dict(side=side, entry_min=e, races=r["races"], runners_per_race=g["runners"].mean(),
                         green_touch_per_race=rt_["mean"], t_touch=rt_["t"],
                         touch_return_pct=g["green_touch"].sum() / g["staked"].sum() * 100,
                         hedged_return_pct=g["green_hedged"].sum() / max(g["staked_hedged"].sum(), 1e-9) * 100,
                         open_liability_per_race=g["open_liability"].mean(),
                         staked_per_race=g["staked"].mean(), green_per_race=r["mean"], t=r["t"],
                         pct_races_green=(g["green_touch"] > 0).mean() * 100, return_per_staked_pct=g["green"].sum() / g["staked"].sum() * 100,
                         drift_per_race=g["drift"].mean(), spread_per_race=g["spread"].mean(),
                         realised_per_race=g["realised"].mean(), full_hedge_pct=g["full_hedges"].mean() * 100,
                         sub_min_hedges_per_race=g["sub_min_hedges"].mean()))
    return pd.DataFrame(rows)


def by_group(rr: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per runner, % of stake: green if the whole green-up filled at the best price (touch), the mid-price
    drift part, and how much of the green-up the visible ladder could actually take."""
    rows = []
    for (side, e, g_), g in rr.groupby(["side", "entry_min", col], observed=True):
        r = rt(g["touch"] / g["stake"] * 100, g["race"])
        rows.append({"side": side, "entry_min": e, col: g_, "runners": len(g), "touch_green_pct": r["mean"],
                     "t": r["t"], "drift_pct": float((g["drift"] / g["stake"]).mean() * 100),
                     "fully_greened_pct": float(g["full_hedge"].mean() * 100),
                     "hedge_filled_pct": float(g["hedge_frac"].mean() * 100)})
    return pd.DataFrame(rows)


def by_rank(rr: pd.DataFrame) -> pd.DataFrame:
    rr = rr.copy()
    rr["rank"] = pd.cut(rr["rank"], [0, 1, 3, 6, 99], labels=["fav", "2-3", "4-6", "7+"])
    return by_group(rr, "rank")


def by_price(rr: pd.DataFrame) -> pd.DataFrame:
    rr = rr.copy()
    rr["odds"] = pd.cut(rr["p_in"], [1, 3, 6, 11, 21, 51, 1001], labels=["<3", "3-6", "6-11", "11-21", "21-51", "51+"])
    return by_group(rr, "odds")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--entries", default=",".join(str(x) for x in ENTRIES_S), help="seconds before the advertised start")
    ap.add_argument("--hedge-s", type=float, default=10.0, help="green up at the last live row this many s before the start")
    ap.add_argument("--stake", type=float, default=10.0, help="backer's stake per runner")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()
    paths = list_tapes(a.tapes)
    kw = dict(entries=tuple(float(x) for x in a.entries.split(",")), hedge_s=a.hedge_s, stake=a.stake)
    os.environ["OMP_NUM_THREADS"] = "1"
    races, runners = [], []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, res in enumerate(pool.map(_one, [(p, kw) for p in paths], chunksize=4)):
            if res is not None:
                races.append(res[0])
                runners.append(res[1])
            if (i + 1) % 300 == 0:
                print(f"  {i + 1}/{len(paths)} races, {time.time() - t0:.0f}s", flush=True)
    if not races:
        raise SystemExit("no races")
    df, rr = pd.concat(races, ignore_index=True), pd.concat(runners, ignore_index=True)
    both = df.groupby(["race", "entry_min"]).filter(lambda g: len(g) == 2).groupby(["race", "day", "entry_min"],
                                                                                   as_index=False).agg(
        runners=("runners", "mean"), staked=("staked", "sum"), green=("green", "sum"), realised=("realised", "sum"),
        drift=("drift", "sum"), spread=("spread", "sum"), full_hedges=("full_hedges", "mean"),
        green_touch=("green_touch", "sum"), green_hedged=("green_hedged", "sum"), staked_hedged=("staked_hedged", "sum"),
        open_liability=("open_liability", "sum"),
        sub_min_hedges=("sub_min_hedges", "sum"))
    both["side"] = "back+lay"
    df_all = pd.concat([df, both], ignore_index=True)
    df_all.to_csv(os.path.join(a.out, "races.csv"), index=False)
    rr.to_csv(os.path.join(a.out, "runners.csv"), index=False)
    s = summary(df_all)
    s.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    rk = by_rank(rr)
    rk.to_csv(os.path.join(a.out, "by_rank.csv"), index=False)
    pr = by_price(rr)
    pr.to_csv(os.path.join(a.out, "by_odds.csv"), index=False)
    print(f"=== P21: back / lay every runner, unbiased green-up {a.hedge_s:.0f}s before the advertised start; "
          f"${a.stake:.0f} per runner; {df['race'].nunique()} races ===")
    print("$ per race. green_touch = every green-up filled at the best price (the theory itself); green = the worst "
          "outcome with the visible ladder (a green-up the ladder can't take leaves the rest open)")
    cols = ["side", "entry_min", "races", "runners_per_race", "staked_per_race", "green_touch_per_race", "t_touch",
            "touch_return_pct", "pct_races_green", "drift_per_race", "full_hedge_pct", "hedged_return_pct",
            "open_liability_per_race", "green_per_race", "realised_per_race"]
    print(s[cols].round(3).to_string(index=False))
    print("\n=== By favourite rank at entry, % of stake per runner (race-clustered t) ===")
    print(rk.round(3).to_string(index=False))
    print("\n=== By odds at entry ===")
    print(pr.round(3).to_string(index=False))
    print(f"\nsaved to {a.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
