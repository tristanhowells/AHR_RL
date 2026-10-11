import numpy as np

from ahr_rl.market_study import SIGNALS, add_race_ranks, add_segments, ic, sample_tape
from ahr_rl.synthetic import make_synthetic_tape


def test_sample_tape_columns_and_ranges(tmp_path):
    p = str(tmp_path / "20260101_0000_Test_1_1.npz")
    make_synthetic_tape(0).save(p)
    df = sample_tape(p, every_s=20)
    assert df is not None and len(df)
    for c in SIGNALS + ("prob", "rank", "cost10", "fwd_30", "back_30", "lay_start"):
        assert c in df.columns
    # market shares of one snapshot sum to 100, ranks start at the favourite
    snap = df[df["t"] == df["t"].iloc[0]]
    assert abs(snap["prob"].sum() - 100) < 0.5
    assert snap.loc[snap["rank"] == 1, "price"].iloc[0] == snap["price"].min()  # ties allowed
    assert (df["cost10"].dropna() > -1e-6).all()  # an instant round trip never makes money
    df = add_race_ranks(add_segments(df), list(SIGNALS) + ["fwd_30"])
    r = ic(df, "wom", "fwd_30", min_races=1)
    assert np.isnan(r["ic"]) or -1 <= r["ic"] <= 1
