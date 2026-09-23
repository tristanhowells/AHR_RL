"""Step 1: review and analyse the Betfair AUS win price data.

Run from the repo root:  python analysis/01_data_review.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from racenet.data import implied_probs, load  # noqa: E402

COMMISSION = 0.05  # matches commission_rate recorded in the user's own market captures
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def flat_back_pnl(df, price_col="bsp", commission=COMMISSION):
    """1-unit back bet at `price_col`, commission on net winnings."""
    return np.where(df["win_lose"] == 1, (df[price_col] - 1) * (1 - commission), -1.0)


def roi_table(df, by):
    g = df.groupby(by, observed=True)["pnl_bsp"]
    t = pd.DataFrame({"n": g.size(), "win": df.groupby(by, observed=True)["win_lose"].mean(),
                      "roi": g.mean(), "roi_se": g.std() / np.sqrt(g.size())})
    return t


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def race_logloss(df, prob_col):
    """Mean -log(p_winner) over races (multinomial log loss), single-winner races only."""
    w = df[(df["win_lose"] == 1) & (df["n_winners"] == 1)]
    return -np.log(np.clip(w[prob_col], 1e-9, 1)).mean()


df = load()
df["n_winners"] = df.groupby("event_id")["win_lose"].transform("sum")

# ---------------------------------------------------------------------------
section("1. Coverage")
mk = df.drop_duplicates("event_id")
print(f"files: {df.source_file.nunique()}  runners: {len(df):,}  markets: {len(mk):,}")
print(f"race dates: {df.race_date.min().date()} -> {df.race_date.max().date()} "
      f"({df.race_date.nunique()} distinct days)")
print(f"venues: {df.venue.nunique()}  selection_ids: {df.selection_id.nunique():,}  runner keys (code|name): {df.runner_key.nunique():,}")
print("\nmarkets by code:\n", mk.race_code.value_counts().to_string())
print("\nmarkets by class:\n", mk.race_class.value_counts().to_string())
print("\ntop venues:\n", mk.venue.value_counts().head(15).to_string())
print("\ndistance (m) by code:\n", mk.groupby("race_code").distance_m.describe().round(0).to_string())
print("\nfield size:\n", mk.field_size.describe().round(2).to_string())

# ---------------------------------------------------------------------------
section("2. Data quality")
print("nulls per column:\n", df.isna().sum()[lambda s: s > 0].to_string() or "none")
print("\nwinners per market:\n", mk.n_winners.value_counts().to_string())
print("\nselection_id -> multiple names:",
      (df.groupby("selection_id").runner_name.nunique() > 1).sum())
print("runner_name -> multiple selection_ids:",
      (df.groupby("runner_name").selection_id.nunique() > 1).sum())
for c in ["bsp", "ppwap", "morningwap", "ppmax", "ppmin"]:
    print(f"{c:11s} min={df[c].min():7.2f}  max={df[c].max():8.2f}  "
          f"==1.0: {(df[c] == 1.0).sum():5d}  >=1000: {(df[c] >= 1000).sum():5d}")
print("\nzero pre-play volume runners:", (df.pptradedvol == 0).sum(),
      " zero morning volume:", (df.morningtradedvol == 0).sum())
print("runners with any in-play trade:", f"{df.ip_traded.mean():.1%}")
print("ppmin > ppmax (inverted / sentinel):", (df.ppmin > df.ppmax).sum())

# ---------------------------------------------------------------------------
section("3. Market books (sum of 1/price per market)")
books = pd.DataFrame({c: (1 / df[c]).groupby(df.event_id).sum()
                      for c in ["bsp", "ppwap", "morningwap"]})
print(books.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
books["field_size"] = mk.set_index("event_id").field_size
print("\nmedian book by field size:\n",
      books.groupby(pd.cut(books.field_size, [0, 6, 8, 10, 12, 20])).median().round(3).to_string())

# ---------------------------------------------------------------------------
section("4. Predictive power of each price (race-level multinomial log loss, lower=better)")
for c in ["bsp", "ppwap", "morningwap", "ppmax", "ppmin"]:
    df[f"p_{c}"] = implied_probs(df, c)
uniform = -np.log(1 / mk[mk.n_winners == 1].field_size).mean()
print(f"{'uniform (1/N)':14s} {uniform:.4f}  (all markets)")
# compare each price with BSP on the markets where that price exists for every runner,
# otherwise dropping NaN winners biases the comparison
print(f"{'price':12s} {'markets':>8s} {'LL price':>9s} {'LL bsp':>8s}")
for c in ["ppwap", "ppmax", "ppmin", "morningwap"]:
    complete = df[c].notna().groupby(df.event_id).transform("all")
    sub = df[complete].copy()
    sub["p_c"] = implied_probs(sub, c)
    sub["p_b"] = implied_probs(sub, "bsp")
    print(f"{c:12s} {sub.event_id.nunique():8d} {race_logloss(sub, 'p_c'):9.4f} {race_logloss(sub, 'p_b'):8.4f}")
print(f"{'bsp (all)':12s} {mk.shape[0]:8d} {race_logloss(df, 'p_bsp'):9.4f}")

# Leakage test: a genuinely pre-off price cannot beat BSP by this margin.
print("\nmean log(bsp/morningwap) winners vs losers:",
      np.log(df.bsp / df.morningwap).groupby(df.win_lose).mean().round(3).to_dict())

# ---------------------------------------------------------------------------
section("5. BSP calibration / favourite-longshot bias")
bins = [1, 1.5, 2, 3, 4, 6, 8, 11, 16, 21, 31, 51, 101, 1001]
df["bsp_band"] = pd.cut(df.bsp, bins, right=False)
df["pnl_bsp"] = flat_back_pnl(df)
cal = roi_table(df, "bsp_band")
cal["implied"] = df.groupby("bsp_band", observed=True).p_bsp.mean()
cal["act/impl"] = cal.win / cal.implied
print(cal.round(4).to_string())
print(f"\nall runners flat back @BSP ROI (5% comm): {df.pnl_bsp.mean():+.2%}")

# ---------------------------------------------------------------------------
section("6. By BSP rank in race (favouritism)")
df["bsp_rank"] = df.groupby("event_id").bsp.rank(method="average")
df["rank_c"] = df.bsp_rank.clip(upper=8).round().astype(int)
rk = roi_table(df, "rank_c")
rk["implied"] = df.groupby("rank_c").p_bsp.mean()
print(rk.round(4).to_string())
print("\nfavourite ROI by code:\n",
      df[df.bsp_rank == 1].groupby("race_code").pnl_bsp.agg(["size", "mean"]).round(4).to_string())

# ---------------------------------------------------------------------------
section("7. ROI by segment (flat back @BSP, 5% comm) — all runners")
for col in ["race_code", pd.cut(df.field_size, [0, 6, 8, 10, 12, 20]).rename("field"),
            "race_class"]:
    g = roi_table(df, col)
    print(g[g.n >= 300].round(4).to_string(), "\n")

# ---------------------------------------------------------------------------
section("8. Price movement ppwap -> bsp (both only final at the off)")
df["lr_pp_bsp"] = np.log(df.bsp / df.ppwap)
df["mv"] = pd.cut(df.lr_pp_bsp, [-10, -0.2, -0.05, 0.05, 0.2, 10],
                  labels=["firm>20%", "firm5-20", "steady", "drift5-20", "drift>20%"])
mv = roi_table(df, "mv")
mv["implied"] = df.groupby("mv", observed=True).p_bsp.mean()
mv["act/impl"] = mv.win / mv.implied
print(mv.round(4).to_string())
print("\ncorr(log ppwap, log bsp):", np.corrcoef(np.log(df.ppwap), np.log(df.bsp))[0, 1].round(3))
print("pp range missing (no trade in window):", f"{df.pp_range_missing.mean():.1%}")

# ---------------------------------------------------------------------------
section("9. Liquidity")
liq = mk.set_index("event_id")
vol = df.groupby("event_id")[["morningtradedvol", "pptradedvol", "iptradedvol"]].sum()
print(vol.describe(percentiles=[.1, .5, .9]).round(0).to_string())
vol["code"] = liq.race_code
print("\nmedian pre-play matched by code:\n", vol.groupby("code").pptradedvol.median().round(0).to_string())

# ---------------------------------------------------------------------------
section("10. Runner identity and history available inside this sample")
ids_per_key = df.groupby("runner_key").selection_id.nunique()
print("runner_keys (code|name):", len(ids_per_key), " with >1 selection_id:", (ids_per_key > 1).sum())
print("selection_ids reused across different names:",
      (df.groupby("selection_id").runner_key.nunique() > 1).sum())
runs = df.sort_values("event_dt").groupby("runner_key").event_dt
n_runs = runs.size()
print("runs per horse in sample:\n", n_runs.value_counts().sort_index().head(10).to_string())
gaps = runs.diff().dt.days.dropna()
print("\ndays between consecutive runs (within sample):\n",
      gaps.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
