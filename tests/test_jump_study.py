import numpy as np

from ahr_rl.jump_study import _green, _walk, study_tape
from ahr_rl.ladder import PRICES, price_to_tick


def test_walk_respects_limit():
    ticks = np.array([price_to_tick(5.0), price_to_tick(4.9), price_to_tick(4.5), -1])
    sizes = np.array([4.0, 4.0, 100.0, 0.0])
    p, f = _walk(ticks, sizes, 10.0, max_ticks=2)  # 4.5 is 5 ticks away: not taken
    assert abs(f - 0.8) < 1e-9 and abs(p - 4.95) < 1e-9
    p, f = _walk(ticks, sizes, 10.0, max_ticks=10)
    assert f == 1.0 and abs(p - (4 * 5.0 + 4 * 4.9 + 2 * 4.5) / 10) < 1e-9


def test_green_signs_and_commission():
    assert abs(_green("back", 20.0, 16.0, 0.1) - 0.25 * 0.9) < 1e-9  # backed 20, laid 16: +25% less 10%
    assert abs(_green("back", 16.0, 20.0, 0.1) - (-0.2)) < 1e-9  # losses pay no commission
    assert abs(_green("lay", 16.0, 20.0, 0.0) - 0.2) < 1e-9  # laid 16, backed 20
    assert np.isnan(_green("back", float("nan"), 2.0, 0.0))


def test_study_runs_on_a_tape(tmp_path):
    from ahr_rl.synthetic import make_synthetic_tape

    t = make_synthetic_tape(0)
    p = str(tmp_path / "20260101_0000_Test_1_1.npz")
    t.save(p)
    rows = study_tape(p, horizons=(5, 30, "start"))
    assert rows, "expected at least control rows"
    events = {r["event"] for r in rows}
    assert "control" in events
    for r in rows:
        assert r["price"] <= 30 and r["spread"] <= 3
        assert r["event"] == "control" or abs(r["jump_ticks"]) >= r["J"]
        # the scheduled-start exit only exists for events before the start
        assert (r["t_rel"] < 0) or np.isnan(r["cont_start"])
