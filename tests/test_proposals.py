import os
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


def test_seq_frames_no_lookahead(tmp_path):
    from ahr_rl.seq_study import CHANNELS, tape_frames

    t = make_synthetic_tape(5)
    p1, p2 = str(tmp_path / "a.npz"), str(tmp_path / "b.npz")
    t.save(p1)
    rows = pd.DataFrame(dict(s=[400, 600], runner=[0, 1]))
    X1, Y1 = tape_frames(p1, rows, 60.0, 2.0)
    assert X1.shape == (2, 30, len(CHANNELS)) and np.isfinite(X1.astype(np.float32)).all()
    # scramble everything after the decision step: frames must not change, targets should
    s = 400
    t.back_tick[s + 1:] = np.maximum(t.back_tick[s + 1:] - 3, 0)
    t.lay_tick[s + 1:] = np.maximum(t.lay_tick[s + 1:] - 3, 0)
    t.back_size[s + 1:] *= 3
    keep = t.trade_step <= s
    t.trade_vol = np.where(keep, t.trade_vol, t.trade_vol * 5).astype(np.float32)
    t.save(p2)
    X2, Y2 = tape_frames(p2, rows.iloc[:1], 60.0, 2.0)
    assert np.array_equal(X1[:1], X2)


def test_blackbox_explorer_and_cap():
    from ahr_rl.continuous import RandomStrategies
    from ahr_rl.env_continuous import N_CONT, ContinuousAllocEnv, ContinuousConfig
    from ahr_rl.features import R_MAX

    n, F, G = 3, 10, 5
    ex = RandomStrategies(n, F, G, "episodic", max_fraction=0.1, seed=0)
    rng = np.random.default_rng(0)
    obs = dict(runners=rng.normal(size=(n, R_MAX, F)).astype(np.float32),
               global_=None, mask=np.zeros((n, R_MAX), np.float32))
    obs["global"] = rng.normal(size=(n, G)).astype(np.float32)
    obs["mask"][:, :6] = 1
    a1 = ex.act(obs)
    assert a1.shape == (n, N_CONT) and (a1[:, 6:R_MAX] == 0).all() and (a1[:, -1] <= 0.1 + 1e-9).all()
    # an episodic strategy is a fixed function of the state within a race ...
    assert np.allclose(a1, ex.act(obs))
    # ... and a new one is drawn when the race ends
    ex.episode_done(np.array([True, False, False]))
    a2 = ex.act(obs)
    assert not np.allclose(a1[0], a2[0]) and np.allclose(a1[1:], a2[1:])
    # the wager cap holds in the env
    t = make_synthetic_tape(6)
    env = ContinuousAllocEnv([t], ContinuousConfig(max_fraction=0.02))
    o, _ = env.reset(options={"tape": t})
    a = np.zeros(N_CONT, np.float32)
    a[:6] = 1.0
    a[-1] = 1.0  # asks for the whole bank
    env.step(a)
    assert env.ex.turnover <= 0.02 * 500 + 1e-6


def test_strategy_search_pieces(tmp_path):
    from ahr_rl.strategy_search import (GLOBAL_IDX, RUNNER_IDX, _eval_race, make_policy, mutate, random_strategy,
                                        env_config)

    rng = np.random.default_rng(0)
    s = random_strategy(rng)
    assert np.isclose(np.linalg.norm(s["w"]), 1.0) and len(s["v"]) == len(GLOBAL_IDX)
    m = mutate(s, rng, 0.5)
    assert np.isclose(np.linalg.norm(m["w"]), 1.0) and 0.3 <= m["enter"] <= 5.0
    t = make_synthetic_tape(7)
    p = str(tmp_path / "20990101_syn_7.npz")
    t.save(p)
    stats = (np.zeros(len(RUNNER_IDX)), np.ones(len(RUNNER_IDX)), np.zeros(len(GLOBAL_IDX)), np.ones(len(GLOBAL_IDX)))
    # an always-enter strategy trades; "None" is do-nothing and scores exactly 0
    eager = dict(s, enter=-1e9, side="back", max_open=1)
    out = _eval_race((p, [eager, None], stats, 1))
    assert np.isfinite(out).all() and out[1] == 0.0
    pol = make_policy(eager, *stats, env_config())
    from ahr_rl.env import BetfairPreRaceEnv
    env = BetfairPreRaceEnv([p], env_config())
    obs, _ = env.reset(options={"tape": p})
    a = pol(obs, env)
    assert (a > 0).sum() == 1  # max_open = 1


def test_day_study_signals_no_lookahead():
    from ahr_rl.day_study import back_ret, lay_ret_liab, meeting_signals, stressed

    assert np.isclose(back_ret(np.array([5.0]), np.array([1]), 0.1)[0], 3.6)
    assert np.isclose(lay_ret_liab(np.array([1.05]), np.array([0]), 0.1)[0], 0.9 / 0.05)
    assert lay_ret_liab(np.array([1.05]), np.array([1]), 0.1)[0] == -1.0
    # lays at 1.05 that won 47 / 50 look profitable (+0.14 per $1 liability) on this sample,
    # but not at the 95% upper bound of the win rate
    won = np.r_[np.ones(47, int), np.zeros(3, int)]
    obs = lay_ret_liab(np.full(50, 1.05), won, np.full(50, 0.1)).mean()
    assert obs > 0 and stressed(np.full(50, 1.05), won, np.full(50, 0.1), "lay") < 0
    rows = []
    for k in range(4):  # one meeting, races in order; jockey "A" wins races 0 and 1 as an outsider
        for i, (j, draw) in enumerate([("A", 1), ("B", 2), ("C", 3)]):
            won = int(i == 0) if k < 2 else int(i == 1)
            rows.append(dict(race=f"r{k}", day="20260101", venue="X", start_ms=k * 1000.0, won=won, p_bsp=1 / 3,
                             rank=i + 1, draw_rel=i / 2, jockey=j, trainer=j))
    d = meeting_signals(pd.DataFrame(rows))
    ja = d[d["jockey"] == "A"].sort_values("order")
    assert np.isnan(ja["sig_jockey"].iloc[0])  # first race: nothing known yet
    assert np.isclose(ja["sig_jockey"].iloc[1], 2 / 3)  # only race 0 counts
    assert np.isclose(ja["sig_jockey"].iloc[3], 2 / 3 + 2 / 3 - 1 / 3)  # races 0-2, not race 3 itself
    assert d.loc[d["order"] <= 2, "sig_draw"].isna().all()  # needs two earlier races
    assert (d.loc[(d["order"] == 3) & (d["draw_rel"] == 0), "sig_draw"] > 0).all()  # inside was winning


def test_race_filter(tmp_path):
    import argparse

    from ahr_rl import race_filter

    paths = [str(tmp_path / f"20260101_1200_{v}_1_{i}.npz") for i, v in enumerate(["Flemington", "Melton", "Kilmore"])]
    csv = tmp_path / "races.csv"
    pd.DataFrame(dict(race=[os.path.basename(p)[:-4] for p in paths], race_type=["Flat", "Harness", "Flat"],
                      venue=["Flemington", "Melton", "Kilmore"])).to_csv(csv, index=False)
    ap = argparse.ArgumentParser()
    race_filter.add_args(ap)
    a = ap.parse_args(["--race-type", "flat", "--races-csv", str(csv)])
    assert race_filter.filter_paths(paths, a, log=lambda m: None) == [paths[0], paths[2]]
    a = ap.parse_args(["--race-type", "flat", "--metro-only", "--races-csv", str(csv)])
    assert race_filter.filter_paths(paths, a, log=lambda m: None) == [paths[0]]
    assert race_filter.filter_paths(paths, ap.parse_args([])) == paths  # no filter: unchanged
    assert race_filter.tag(a) == "_flat_metro"
