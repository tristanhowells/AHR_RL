import numpy as np
import pandas as pd

from ahr_rl.bsp_study import back_ret, lay_clv, lay_ret
from ahr_rl.exchange import BACK
from ahr_rl.leadlag_study import sample_tape as leadlag_sample
from ahr_rl.sweep_passive import find_events, run_trade
from ahr_rl.synthetic import make_synthetic_tape


def test_bsp_returns():
    assert np.isclose(back_ret(5.0, True, 0.1), 3.6) and back_ret(5.0, False, 0.1) == -1
    assert lay_ret(5.0, True, 0.1) == -1 and np.isclose(lay_ret(5.0, False, 0.1), 0.9 / 4)
    # laying at the fair price has zero value; laying shorter than BSP has positive value
    assert np.isclose(lay_clv(5.0, 5.0), 0) and lay_clv(4.0, 5.0) > 0


def test_leadlag_and_sweep_run(tmp_path):
    t = make_synthetic_tape(0)
    t.spn = np.where(t.active, 5.0, 0.0).astype(np.float32)
    p = str(tmp_path / "20260101_0000_Test_1_1.npz")
    t.save(p)
    rows = leadlag_sample(p, every_s=20)
    assert rows and all(np.isfinite(r["gap"]) for r in rows)
    ev, end = find_events(t, rng=np.random.default_rng(0))
    assert end > 0
    e = ev[0] if ev else dict(step=100, runner=0)
    res = run_trade(t, end, e["step"], e["runner"], BACK, "join", 10, 2, 60)
    assert 0 <= res["filled"] <= 1
    assert res["filled"] == 0 or np.isfinite(res["pnl"])


def test_community_runs(tmp_path):
    from ahr_rl.community import _prep, arbitrage, bracket, study_tape

    t = make_synthetic_tape(1)
    p = str(tmp_path / "20260101_0000_Test_1_2.npz")
    t.save(p)
    res = study_tape(p)
    assert set(res) == {"scalp", "signal", "arb"}
    for r in res["scalp"]:
        assert 0 <= r["filled"] <= 1 and (r["filled"] == 0 or np.isfinite(r["pnl"]))
    P = _prep(t)
    a = arbitrage(t, P, "x")
    assert a["back_arb"] >= a["back_arb_persist"] >= 0


def test_sweep_bsp_trade(tmp_path):
    from ahr_rl.sweep_bsp import _score, study_tape

    # green position: same P&L either way; commission only on profit
    sc = _score(1.0, 1.0, 0.1, 0.3, True)
    assert np.isclose(sc["ev"], 0.9) and np.isclose(sc["worst"], 0.9)
    t = make_synthetic_tape(2)
    t.bsp = np.full(t.n_runners, 5.0, np.float32)
    p = str(tmp_path / "20260101_0000_Test_1_3.npz")
    t.save(p)
    for r in study_tape(p):
        assert 0 <= r["filled"] <= 1
        if r["filled"] > 0:
            assert np.isfinite(r["aggressive_ev"])
