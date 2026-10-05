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


def test_pair_study(tmp_path):
    from ahr_rl.ladder import PRICES
    from ahr_rl.pair_study import KS, pair_pnl, sample_tape

    t = make_synthetic_tape(3)
    t.bsp = np.full(t.n_runners, 5.0, np.float32)
    t.winner, t.went_in_play = 0, True
    p = str(tmp_path / "20260101_0000_Test_1_4.npz")
    t.save(p)
    df = sample_tape(p)
    assert df is not None and len(df)
    # a k=1 ladder is a single k=1 pair; longer ladders use at least one leg
    for side in ("back", "lay"):
        for st in ("close", "bsp", "hold"):
            single = pair_pnl(df, side, 1, st, "through")[0]
            lad = df[f"lad_{side}_1_{st}"].to_numpy(float)
            m = np.isfinite(single) & np.isfinite(lad)
            assert m.any() and np.allclose(single[m], lad[m], atol=1e-4)
            assert (df[f"ladn_{side}_8_{st}"] >= df[f"ladn_{side}_1_{st}"]).all()
    for side in ("back", "lay"):
        for k in KS:
            fs = df[f"fs_{side}_{k}_through"]
            assert ((fs == -1) | (fs >= df["s"] + 3)).all()
            # 'through' needs a trade beyond the hedge price, so it can't fill before 'touch'
            ft = df[f"fs_{side}_{k}_touch"]
            assert ((fs == -1) | ((ft >= 0) & (ft <= fs))).all()
    # hand-built row: back 5.0 hedged two ticks lower (4.8) -> 5/4.8 - 1 after commission
    e0 = int(np.argmin(np.abs(PRICES - 5.0)))
    row = {c: [v] for c, v in dict(back_p=5.0, back_e0=e0, comm=0.1, s_close=50, end=60, back_gclose=-0.05,
                                   bsp=6.0, win=0, fs_back_2_through=55).items()}
    d = pd.DataFrame(row)
    g, filled, done = pair_pnl(d, "back", 2, "close", "through")
    assert filled[0] == False and np.isclose(g[0], -0.05) and done[0] == 50  # filled after the close
    g, filled, done = pair_pnl(d, "back", 2, "bsp", "through")
    assert filled[0] and np.isclose(g[0], (5.0 / PRICES[e0 - 2] - 1) * 0.9) and done[0] == 55
    d["fs_back_2_through"] = -1
    assert np.isclose(pair_pnl(d, "back", 2, "bsp", "through")[0][0], 5.0 / 6.0 - 1)
    assert np.isclose(pair_pnl(d, "back", 2, "hold", "through")[0][0], -1.0)


def test_p3b_forward(tmp_path):
    from ahr_rl.p3b_forward import decide, study_tape, summary
    from ahr_rl.sweep_passive import find_events

    base = dict(races=150, realised_per_fill=0.0, worst_per_fill=0.0, fill_pct=50.0, tp_pct=50.0)
    assert decide(dict(base, attempts=100, ev_per_attempt=0.05, t=5.0)) == "CONTINUE"
    assert decide(dict(base, attempts=300, ev_per_attempt=-0.001, t=-1.0)) == "FAIL"
    assert decide(dict(base, attempts=300, ev_per_attempt=0.01, t=2.5)) == "PASS"
    assert decide(dict(base, attempts=300, ev_per_attempt=0.01, t=1.0)) == "CONTINUE"
    assert decide(dict(base, attempts=600, ev_per_attempt=0.01, t=1.0)) == "FAIL"
    t = make_synthetic_tape(4)
    t.bsp = np.full(t.n_runners, 5.0, np.float32)
    p = str(tmp_path / "20261010_0000_Test_1_5.npz")
    t.save(p)
    # the live ranking only uses volume matched so far; the default keeps the at-the-off ranking
    ev_live, _ = find_events(t, jumps=(5,), rng=np.random.default_rng(0), control_every_s=1e9, live=True)
    ev_old, _ = find_events(t, jumps=(5,), rng=np.random.default_rng(0), control_every_s=1e9)
    assert all(e["top3"] == e["top3_final"] for e in ev_old)
    assert len(ev_live) >= len(ev_old)
    rows = study_tape(p)
    for r in rows:
        assert 0 <= r["filled"] <= 1
    if rows:
        s = summary(pd.DataFrame(rows).assign(hit_tp=lambda d: d["hit_tp"].fillna(False).astype(bool)))
        assert s["attempts"] == len(rows)
