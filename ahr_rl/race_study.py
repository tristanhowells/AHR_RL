"""Race study: do race-level categories (track, state, metro, distance, race type,
class, day of week, time of day, race number, field size, commission) change how
tradeable a market is?

Race metadata comes from the market catalogues (``catalogues`` folder on Drive):
marketName ("R2 1400m CL2" -> race 2, 1400m, class "CL2"), raceType, venue,
timezone (-> state), start time (-> local day of week and hour). Races without a
catalogue fall back to the recording file name (date, time, venue).

Race-level tradeability is aggregated from the market study samples (cell 2g
writes ``samples.parquet``; if it is missing it is rebuilt here with the same
code). "Main runners" are those with >= 10% market share, the only ones that are
realistically tradeable (see the market study):

  matched_$          total matched at the off (single-counted)
  late_share_%       share of it matched in the last 2 min before the scheduled start or later
  off_delay_s        actual off minus scheduled start
  cost_%             median cost of an instant $10 round trip, main runners, final 2 min
  best_$             median $ at the best back + lay price, main runners, final 5 min
  tradeable_%        share of time main runners are tradeable (spread <= 2 ticks, $20+ at
                     the best prices, price <= 30), final 5 min
  move/cost          typical 2-min move / cost, main runners, final 5 min (> 1 = moves beat costs)
  random_trade_%     mean P&L of a random-side $10 round trip held 2 min, main runners,
                     final 5 min (what trading without an edge costs)
  hindsight_%        mean P&L of the same round trip on the side that turned out right
                     (the value of perfect direction knowledge, after real costs)
  wom_ic             within-race rank correlation of WoM with the next 30s move (signal strength)

Analyses (printed and saved in --out):
  A  coverage and category counts
  B  each metric by each category (median over races; mean for P&L columns)
  C  all categories together: OLS of each metric on every category at once, so
     e.g. "metro" and "Saturday" are separated (coefficients vs a baseline level)
  D  venue league table (venues with >= --min-venue races)
  E  crosses: race type x metro, day x metro, time of day x race type
  F  charts

    python -m ahr_rl.race_study --tapes "data/tapes/*.npz" --catalogues "<drive>/catalogues" \\
        --samples runs/market/samples.parquet --out runs/races
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .catalogue import index_catalogues, load_catalogue
from .env import list_tapes
from .tape import Tape

TZ_STATE = {"Australia/Sydney": "NSW", "Australia/Melbourne": "VIC", "Australia/Brisbane": "QLD",
            "Australia/Adelaide": "SA", "Australia/Perth": "WA", "Australia/Hobart": "TAS",
            "Australia/Darwin": "NT", "Australia/Canberra": "ACT"}
# metropolitan tracks (thoroughbred + harness); everything else is provincial / country
METRO = {"randwick", "royal randwick", "rosehill", "warwick farm", "canterbury", "flemington", "caulfield",
         "moonee valley", "the valley", "sandown", "sandown hillside", "sandown lakeside", "eagle farm", "doomben",
         "morphettville", "morphettville parks", "ascot", "belmont", "menangle", "melton", "albion park",
         "gloucester park", "globe derby"}
METRICS = ["matched_$", "late_share_%", "off_delay_s", "cost_%", "best_$", "tradeable_%", "move/cost",
           "random_trade_%", "hindsight_%", "wom_ic"]
MEAN_METRICS = {"random_trade_%", "hindsight_%", "wom_ic"}
CATS = ["race_type", "metro", "state", "distance_b", "class_group", "dow", "weekend", "time_of_day", "race_no_b",
        "field_b", "commission"]


def parse_class(name: str) -> tuple[str, str]:
    """marketName -> (class group, raw class text after the distance)."""
    raw = re.sub(r"^R\d+\s+", "", name or "")
    raw = re.sub(r"^\d{3,5}m\s*", "", raw).strip()
    s = raw.lower()
    if re.search(r"\bgrp ?[123]\b|\bg[123]\b|\blisted\b|\blr\b", s):
        return "Group/Listed", raw
    if re.search(r"\bmdn\b|maiden", s):
        return "Maiden", raw
    if re.search(r"\bbm ?\d+|benchmark", s):
        return "Benchmark", raw
    if re.search(r"\bcl(ass)? ?\d|\bc\d\b|\brst\b|restricted", s):
        return "Class/Restricted", raw
    if re.search(r"hcap|hcp|handicap", s):
        return "Handicap", raw
    if re.search(r"\bpace\b|\bpc\b", s):
        return "Harness pace", raw
    if re.search(r"\btrot\b|\btr\b", s):
        return "Harness trot", raw
    if re.search(r"\bstk|stakes|qlty|quality|\bplate\b", s):
        return "Stakes/Quality", raw
    return ("Other" if raw else "Unknown"), raw


def race_meta(path: str, cat_idx: dict) -> dict:
    t = Tape.load(path)
    name = t.name or os.path.basename(path).split(".npz")[0]
    parts = os.path.basename(path).split(".npz")[0].split("_")
    venue = parts[2] if len(parts) > 2 else "?"
    local = None
    try:
        local = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M")  # recorder's local clock
    except (ValueError, IndexError):
        pass
    rec = dict(race=os.path.basename(path).split(".npz")[0], market_id=t.market_id, venue=venue, race_type="Unknown",
               state="?", market_name="", class_group="Unknown", class_raw="", race_no=np.nan, distance_m=np.nan,
               has_catalogue=False, commission=f"{t.base_rate:g}%")
    cp = cat_idx.get(t.market_id)
    if cp:
        try:
            c = load_catalogue(cp)
            ev, desc = c.get("event") or {}, c.get("description") or {}
            rec["has_catalogue"] = True
            rec["venue"] = ev.get("venue") or venue
            rec["race_type"] = desc.get("raceType") or "Unknown"
            tz = ev.get("timezone") or ""
            rec["state"] = TZ_STATE.get(tz, tz or "?")
            mn = c.get("marketName") or ""
            rec["market_name"] = mn
            rec["class_group"], rec["class_raw"] = parse_class(mn)
            m = re.match(r"R(\d+)", mn)
            rec["race_no"] = int(m.group(1)) if m else np.nan
            m = re.search(r"(\d{3,5})m", mn)
            rec["distance_m"] = float(m.group(1)) if m else np.nan
            st = c.get("marketStartTime")
            if st and tz:
                local = datetime.fromisoformat(st.replace("Z", "+00:00")).astimezone(ZoneInfo(tz)).replace(tzinfo=None)
        except Exception as e:  # metadata is best-effort
            print(f"  catalogue problem {os.path.basename(cp)}: {e}")
    if rec["race_type"] == "Unknown" and t.static is not None and len(t.static):
        from .catalogue import STATIC_NAMES

        if t.static[:, STATIC_NAMES.index("is_harness")].max() > 0:
            rec["race_type"] = "Harness"
        elif t.static[:, STATIC_NAMES.index("is_flat")].max() > 0:
            rec["race_type"] = "Flat"
    rec["local_time"] = local
    # tape-level liquidity facts
    tm = t.total_matched.astype(float)
    tr = t.t_rel
    rec["matched_$"] = float(tm[-1])
    i2 = int(np.searchsorted(tr, -120))
    rec["late_share_%"] = float((tm[-1] - tm[min(i2, len(tm) - 1)]) / tm[-1] * 100) if tm[-1] > 0 else np.nan
    rec["off_delay_s"] = float(tr[-1]) if t.went_in_play else np.nan
    rec["field"] = int(t.active[-1].sum())
    return rec


def add_categories(m: pd.DataFrame) -> pd.DataFrame:
    m["metro"] = np.where(m["venue"].str.lower().isin(METRO), "metro", "non-metro")
    m["distance_b"] = pd.cut(m["distance_m"], [0, 1100, 1300, 1600, 2000, 2400, 5000],
                             labels=["<=1100m", "1101-1300m", "1301-1600m", "1601-2000m", "2001-2400m", "2400m+"])
    lt = pd.to_datetime(m["local_time"])
    m["dow"] = pd.Categorical(lt.dt.day_name().str[:3], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    m["weekend"] = np.where(lt.dt.dayofweek >= 5, "Sat/Sun", "weekday")
    # local time: catalogue start time in the track's timezone; without a catalogue,
    # the recorder's clock from the file name
    m["time_of_day"] = pd.cut(lt.dt.hour + lt.dt.minute / 60, [0, 13, 15, 17, 19, 24],
                              labels=["<13:00", "13-15", "15-17", "17-19", "19:00+"], right=False)
    m["race_no_b"] = pd.cut(m["race_no"], [0, 2, 4, 6, 8, 20], labels=["R1-2", "R3-4", "R5-6", "R7-8", "R9+"])
    m["field_b"] = pd.cut(m["field"], [0, 7, 10, 13, 30], labels=["<=7", "8-10", "11-13", "14+"])
    return m


def race_metrics(s: pd.DataFrame) -> pd.DataFrame:
    """Race-level tradeability from market-study samples."""
    main = s[s["prob"] >= 10]
    f5 = main[(main["t"] >= -300) & (main["t"] < 0)]
    f2 = main[(main["t"] >= -120) & (main["t"] < 0)]
    g5, g2 = f5.groupby("race"), f2.groupby("race")
    out = pd.DataFrame({
        "cost_%": g2["cost10"].median(),
        "best_$": g5["best_usd"].median(),
        "tradeable_%": g5["tradeable"].mean() * 100 if "tradeable" in f5 else np.nan,
        "move_%": g5["fwd_120"].apply(lambda x: x.abs().median()),
        "cost5_%": g5["cost10"].median(),
        "random_trade_%": g5.apply(lambda d: np.nanmean(np.r_[d["back_120"].values, d["lay_120"].values]) * 100),
        "hindsight_%": g5.apply(lambda d: np.nanmean(np.fmax(d["back_120"].values, d["lay_120"].values)) * 100),
    })
    out["move/cost"] = out["move_%"] / out["cost5_%"]

    def wic(d):
        d = d[["wom", "fwd_30"]].dropna()
        return d["wom"].rank().corr(d["fwd_30"].rank()) if len(d) > 30 else np.nan

    out["wom_ic"] = s.groupby("race").apply(wic)
    return out.drop(columns=["move_%", "cost5_%"])


def by_category(r: pd.DataFrame, cat: str, min_n: int = 1) -> pd.DataFrame:
    g = r.groupby(cat, observed=True)
    out = pd.DataFrame({"races": g.size()})
    for mtr in METRICS:
        out[mtr] = g[mtr].mean() if mtr in MEAN_METRICS else g[mtr].median()
    return out[out["races"] >= min_n]


def ols_table(r: pd.DataFrame, y: str, cats, min_level: int = 10) -> pd.DataFrame | None:
    """OLS of y on one-hot categories (most common level = baseline), HC1 standard errors.
    Levels with fewer than ``min_level`` races are pooled into "other" (their
    coefficients would be noise, and single-race levels make the fit degenerate)."""
    d = r[[y] + cats].dropna(subset=[y]).copy()
    if len(d) < 50:
        return None
    for c in cats:
        lv = d[c].astype(str).replace({"nan": "unknown"})
        vc = lv.value_counts()
        d[c] = lv.where(lv.map(vc) >= min_level, "other")
    cats = [c for c in cats if d[c].nunique() > 1]
    X = [np.ones(len(d))]
    names = ["(baseline)"]
    for c in cats:
        levels = d[c].astype(str)
        base = levels.value_counts().idxmax()
        for lv in sorted(levels.unique()):
            if lv == base:
                continue
            X.append((levels == lv).values.astype(float))
            names.append(f"{c}={lv} (vs {base})")
    X = np.column_stack(X)
    yv = d[y].values.astype(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    meat = X.T @ (X * (resid ** 2)[:, None])
    cov = xtx_inv @ meat @ xtx_inv * n / max(n - k, 1)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return pd.DataFrame({"coef": beta, "se": se, "t": beta / np.where(se > 0, se, np.nan)}, index=names)


def plots(r: pd.DataFrame, out_dir: str, min_n: int, min_venue: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["race_type", "metro", "state", "distance_b", "class_group", "dow", "time_of_day", "race_no_b", "field_b"]
    for metric, fname, ref in (("move/cost", "races_move_cost.png", 1.0), ("tradeable_%", "races_tradeable.png", None),
                               ("matched_$", "races_matched.png", None)):
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        for ax, cat in zip(axes.flat, show):
            t = by_category(r, cat, min_n)
            if t.empty:
                ax.set_visible(False)
                continue
            ax.bar(range(len(t)), t[metric].values, color="#4c72b0")
            ax.set_xticks(range(len(t)), [f"{i}\n(n={n})" for i, n in zip(t.index.astype(str), t["races"])],
                          fontsize=7, rotation=0 if len(t) <= 6 else 45, ha="center" if len(t) <= 6 else "right")
            if ref is not None:
                ax.axhline(ref, color="k", lw=0.8, ls="--")
            ax.set_title(f"{metric} by {cat}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=100)
        plt.close(fig)
    v = by_category(r, "venue", min_venue).sort_values("move/cost")
    if len(v):
        fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(v))))
        ax.barh(range(len(v)), v["move/cost"].values, color=np.where(v.index.str.lower().isin(METRO), "#c44e52", "#4c72b0"))
        ax.set_yticks(range(len(v)), [f"{i} (n={n})" for i, n in zip(v.index, v["races"])], fontsize=7)
        ax.axvline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("typical 2-min move / cost, runners with >= 10% market share, final 5 min (red = metro)")
        ax.set_title("Venues by tradeability")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "races_venues.png"), dpi=100)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--catalogues", default="")
    ap.add_argument("--samples", default="", help="samples.parquet from the market study (cell 2g)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=10, help="hide category levels with fewer races")
    ap.add_argument("--min-venue", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)

    paths = list_tapes(a.tapes)
    cat_idx = index_catalogues(a.catalogues) if a.catalogues and os.path.isdir(a.catalogues) else {}
    print(f"{len(paths)} tapes, {len(cat_idx)} catalogues indexed", flush=True)
    meta = pd.DataFrame([race_meta(p, cat_idx) for p in paths])

    if a.samples and os.path.exists(a.samples):
        s = pd.read_parquet(a.samples)
        print(f"loaded {len(s)} market-study samples from {a.samples}")
    else:
        print("no market-study samples found: building them (same code as cell 2g)...", flush=True)
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        from .market_study import _one

        os.environ["OMP_NUM_THREADS"] = "1"
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            parts = [d for d in pool.map(_one, [(p, dict(every_s=10.0, stake=10.0)) for p in paths], chunksize=4)
                     if d is not None]
        s = pd.concat(parts, ignore_index=True)
        s.to_parquet(os.path.join(a.out, "samples.parquet"), index=False)
    if "tradeable" not in s:
        s["tradeable"] = (s["spread"] <= 2) & (s["best_usd"] >= 20) & (s["price"] <= 30)
    r = add_categories(meta.merge(race_metrics(s), left_on="race", right_index=True, how="left"))
    r.to_csv(os.path.join(a.out, "races.csv"), index=False)

    # ---------------- A: coverage
    print("\n" + "=" * 100 + "\nA. COVERAGE\n" + "=" * 100)
    print(f"{len(r)} races, {int(r['has_catalogue'].sum())} with a catalogue "
          f"({r['has_catalogue'].mean() * 100:.0f}%); metrics available for {int(r['cost_%'].notna().sum())}")
    for c in CATS:
        print(f"  {c}: " + ", ".join(f"{k}={v}" for k, v in r[c].astype(str).value_counts().items()))
    print("\nmost common class texts (after race no. and distance) and how they were grouped:")
    print(r.groupby(["class_group", "class_raw"]).size().sort_values(ascending=False).head(40).to_string())
    print("\nmetric medians over all races:")
    print(r[METRICS].median().round(3).to_string())

    # ---------------- B: by category
    print("\n" + "=" * 100 + "\nB. TRADEABILITY BY CATEGORY (median over races; mean for random_trade_%, hindsight_%, "
          f"wom_ic; levels with >= {a.min_n} races)\n" + "=" * 100)
    for c in CATS:
        t = by_category(r, c, a.min_n)
        t.to_csv(os.path.join(a.out, f"B_{c}.csv"))
        print(f"\n--- {c} ---")
        print(t.round(2).to_string())

    # ---------------- C: joint OLS
    print("\n" + "=" * 100 + "\nC. ALL CATEGORIES AT ONCE (OLS, HC1 standard errors; coefficient = difference from the "
          "baseline level with everything else held fixed)\n" + "=" * 100)
    jc = ["race_type", "metro", "state", "distance_b", "dow", "time_of_day", "field_b"]
    r["log_matched"] = np.log10(r["matched_$"].clip(lower=1))
    coefs = {}
    for y in ("log_matched", "cost_%", "tradeable_%", "move/cost", "random_trade_%", "hindsight_%", "wom_ic"):
        tab = ols_table(r, y, jc, max(a.min_n, 10))
        if tab is None:
            continue
        tab.to_csv(os.path.join(a.out, f"C_ols_{y.replace('/', '_per_')}.csv"))
        coefs[y] = tab
        sig = tab[(tab["t"].abs() >= 2) | (tab.index == "(baseline)")]
        print(f"\n--- {y}: {len(tab) - 1} terms, {int((tab['t'].abs() >= 2).sum())} with |t| >= 2 (shown) ---")
        print(sig.round(3).to_string())

    # ---------------- D: venues
    print("\n" + "=" * 100 + f"\nD. VENUES (>= {a.min_venue} races), sorted by move/cost\n" + "=" * 100)
    v = by_category(r, "venue", a.min_venue).sort_values("move/cost", ascending=False)
    v.insert(1, "metro", np.where(v.index.str.lower().isin(METRO), "metro", ""))
    v.to_csv(os.path.join(a.out, "D_venues.csv"))
    print(v.round(2).to_string())

    # ---------------- E: crosses
    print("\n" + "=" * 100 + "\nE. CROSSES (median move/cost; races in brackets)\n" + "=" * 100)
    for rows_, cols_ in (("race_type", "metro"), ("dow", "metro"), ("time_of_day", "race_type"),
                         ("distance_b", "race_type"), ("class_group", "metro")):
        val = r.pivot_table(index=rows_, columns=cols_, values="move/cost", aggfunc="median", observed=True)
        cnt = r.pivot_table(index=rows_, columns=cols_, values="race", aggfunc="count", observed=True)
        cell = val.round(2).astype(str) + " (" + cnt.fillna(0).astype(int).astype(str) + ")"
        cell.to_csv(os.path.join(a.out, f"E_{rows_}_x_{cols_}.csv"))
        print(f"\n--- {rows_} x {cols_} ---")
        print(cell.to_string())

    plots(r, a.out, a.min_n, a.min_venue)
    best = v.head(5)[["races", "move/cost", "tradeable_%", "random_trade_%", "hindsight_%"]].round(2)
    summary = {
        "races": int(len(r)), "with_catalogue": int(r["has_catalogue"].sum()),
        "overall_median_move_per_cost": float(r["move/cost"].median()),
        "overall_mean_random_trade_%": float(r["random_trade_%"].mean()),
        "overall_mean_hindsight_%": float(r["hindsight_%"].mean()),
        "top_venues_by_move_per_cost": best.reset_index().to_dict("records"),
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=1, default=str))
    print(f"\nsaved to {a.out}")


if __name__ == "__main__":
    main()
