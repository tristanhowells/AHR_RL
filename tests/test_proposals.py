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
