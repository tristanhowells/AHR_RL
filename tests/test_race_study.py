import numpy as np
import pandas as pd

from ahr_rl.race_study import add_categories, ols_table, parse_class


def test_parse_class_real_market_names():
    assert parse_class("R4 1720m Trot M")[0] == "Harness trot"
    assert parse_class("R8 2185m Pace M")[0] == "Harness pace"
    assert parse_class("R1 1220m Hcap")[0] == "Handicap"
    assert parse_class("R2 1400m CL2") == ("Class/Restricted", "CL2")
    assert parse_class("R7 1200m Grp3")[0] == "Group/Listed"
    assert parse_class("R3 1000m Mdn")[0] == "Maiden"
    assert parse_class("R5 1600m BM64")[0] == "Benchmark"
    assert parse_class("R2 900m")[0] == "Unknown"


def test_ols_recovers_effect_and_pools_rare_levels():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"metro": rng.choice(["metro", "non-metro"], n),
                       "state": rng.choice(["NSW", "VIC"], n)})
    df.loc[:2, "state"] = "NT"  # 3 races: pooled into "other", not a degenerate column
    df["y"] = 1.0 + 2.0 * (df["metro"] == "metro") + rng.normal(0, 0.1, n)
    tab = ols_table(df, "y", ["metro", "state"], min_level=10)
    coef = tab.filter(like="metro=", axis=0)["coef"].iloc[0]
    assert abs(abs(coef) - 2.0) < 0.05
    assert not any("NT" in i for i in tab.index)
    assert np.isfinite(tab["t"]).all()


def test_categories():
    m = pd.DataFrame({"venue": ["Rosehill", "Kilmore"], "distance_m": [1400, 1690], "race_no": [2, 9],
                      "field": [16, 8], "local_time": pd.to_datetime(["2026-08-29 12:20", "2026-09-10 21:37"])})
    m = add_categories(m)
    assert list(m["metro"]) == ["metro", "non-metro"]
    assert list(m["dow"].astype(str)) == ["Sat", "Thu"]
    assert list(m["weekend"]) == ["Sat/Sun", "weekday"]
