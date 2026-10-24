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


def test_liquidity_gate():
    from ahr_rl.blackbox import best_usd_trailing
    from ahr_rl.env_continuous import N_CONT, ContinuousAllocEnv, ContinuousConfig
    from ahr_rl.features import R_MAX

    t = make_synthetic_tape(6)
    t.back_size[:, 1:, 0] = 1.0  # runners 1.. are thin, runner 0 keeps its synthetic depth (~$40 a side)
    t.lay_size[:, 1:, 0] = 1.0
    t.back_size[:, 0, 0] = t.lay_size[:, 0, 0] = 30.0
    cfg = ContinuousConfig(min_best_usd=20.0, max_fraction=0.05, random_start_s=0.0)
    env = ContinuousAllocEnv([t], cfg)
    o, _ = env.reset(options={"tape": t})
    assert o["act_mask"][0] == 1 and o["act_mask"][1:6].sum() == 0 and o["mask"][:6].sum() == 6  # sees all
    a = np.zeros(N_CONT, np.float32)
    a[:6], a[-1] = 1.0, 0.05  # back everything
    env.step(a)
    assert {f.runner for f in env.ex.fills} <= {0}  # thin runners cannot be opened
    # a held runner that falls below the gate stays actionable, but only to reduce
    env.ex.W[2], env.ex.L[2] = 5.0, 0.0  # pretend we are long runner 2
    o = env._obs()
    assert o["act_mask"][2] == 1
    n0 = len(env.ex.fills)
    a[:] = 0
    a[2], a[-1] = 1.0, 0.05  # back more: not allowed
    env.step(a)
    assert all(f.runner != 2 for f in env.ex.fills[n0:])
    # the networks get (above cut, distance to cut) per runner; off-gate it is unchanged
    from ahr_rl.features import N_RUNNER_FEATURES
    assert o["runners"].shape[1] == N_RUNNER_FEATURES + 2
    assert o["runners"][0, -2] == 1 and o["runners"][1, -2] == 0 and o["runners"][2, -2] == 0  # held != above cut
    assert np.isclose(o["runners"][0, -1], np.log(60 / 20) / 3) and np.isclose(o["runners"][1, -1], np.log(2 / 20) / 3)
    assert (o["runners"][6:, -2:] == 0).all()  # empty slots
    o2, _ = ContinuousAllocEnv([t], ContinuousConfig()).reset(options={"tape": t})
    assert o2["runners"].shape[1] == N_RUNNER_FEATURES
    o3, _ = ContinuousAllocEnv([t], ContinuousConfig(min_best_usd=20.0, gate_inputs=False)).reset(options={"tape": t})
    assert o3["runners"].shape[1] == N_RUNNER_FEATURES
    # actor and critic accept the wider input
    import torch
    from ahr_rl.continuous import Actor, Critic, act_mask_t, obs_t
    actor, critic = Actor(16, 1, runner_extra=2), Critic(16, 1, runner_extra=2)
    r, g, m = (x[None] for x in obs_t(o, "cpu"))
    _, act, _, _, _ = actor.sample(r, g, m, act_mask=act_mask_t(o, "cpu")[None])
    assert act.shape == (1, N_CONT) and (act[0, 6:R_MAX] == 0).all() and act[0, 1] == 0 and act[0, 3:6].abs().sum() == 0
    assert critic(r, g, m, act).shape == (1,)
    # the trailing mean is live: no future rows
    b = best_usd_trailing(t, 10.0)
    assert np.isclose(b[0, 0], 60.0) and np.isclose(b[5, 1], 2.0)

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


def test_liquidity_quintiles_use_train_cutoffs():
    from ahr_rl.liquidity_study import Q_LABELS, quintiles

    x = pd.Series(np.r_[np.arange(100.0), np.arange(1000.0, 1100.0)])
    train = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    q = quintiles(x, train)
    # cut-offs come from the first 100 (train) values only: every later, larger value is in the top quintile
    assert list(q.cat.categories) == Q_LABELS
    assert (q[100:] == "Q5 deep").all() and (q[:100].value_counts() == 20).all()


def test_rebalance_signals(tmp_path):
    from ahr_rl.rebalance_study import race_ic, sample_tape

    t = make_synthetic_tape(8)
    p = str(tmp_path / "20260101_0000_Test_1_9.npz")
    t.save(p)
    df, ll = sample_tape(p)
    assert df is not None and len(df) and ll is not None
    # the signals only use the past: sig_lag = -(others' change) x share - own change, own change = mom
    w = 15
    row = df.dropna(subset=[f"sig_lag_{w}"]).iloc[0]
    assert np.isfinite(row[f"mom_{w}"])
    # a perfectly informative signal has IC 1 in every race
    d = pd.DataFrame(dict(race=np.repeat(["a", "b", "c"], 10), x=np.tile(np.arange(10.0), 3)))
    d["y"] = d["x"] * 2
    r = race_ic(d, "x", "y")
    assert np.isclose(r["ic"], 1.0) and r["races"] == 3


def test_runner_agents():
    import torch
    from ahr_rl.continuous import act_mask_t, obs_t
    from ahr_rl.features import N_RUNNER_FEATURES, R_MAX
    from ahr_rl.runner_agents import (N_ACT, RunnerActor, RunnerAgentConfig, RunnerAgentEnv, RunnerCritic,
                                      flat_action)

    t = make_synthetic_tape(6)
    t.back_size[:, 1:, 0] = t.lay_size[:, 1:, 0] = 1.0  # only runner 0 is above the gate
    t.back_size[:, 0, 0] = t.lay_size[:, 0, 0] = 30.0
    cfg = RunnerAgentConfig(min_best_usd=20.0, random_start_s=0.0, order_cap=20.0, runner_cap=30.0, market_cap=150.0)
    env = RunnerAgentEnv([t], cfg)
    o, _ = env.reset(options={"tape": t})
    assert env.action_space.shape == (N_ACT * R_MAX,)
    a = np.zeros(N_ACT * R_MAX, np.float32)
    a[:6], a[R_MAX:R_MAX + 6] = 1.0, 1.0  # back every runner at full size
    for _ in range(4):
        env.step(a)
    assert {f.runner for f in env.ex.fills} == {0}  # only the online runner trades
    loss0 = -min(env.ex.W[0], env.ex.L[0])
    assert loss0 <= 30.0 + 1e-6  # the runner cap holds over repeated adds
    # reduce-only exit on a runner below the gate, capped at flat
    env.ex.W[3], env.ex.L[3] = -40.0, 10.0  # pretend we laid runner 3
    n0 = len(env.ex.fills)
    a[:] = 0
    a[3], a[R_MAX + 3] = 1.0, 1.0  # back it: reduces
    env.step(a)
    assert any(f.runner == 3 for f in env.ex.fills[n0:])  # the exit is allowed below the gate
    assert env.ex.W[3] - env.ex.L[3] <= 1e-6  # never past flat
    # networks: per-runner outputs, masked to online runners, critic per runner
    o = env._obs()
    F = o["runners"].shape[1]
    actor, critic = RunnerActor(16, 1, F - N_RUNNER_FEATURES), RunnerCritic(16, 1, F - N_RUNNER_FEATURES)
    r, g, m = (x[None] for x in obs_t(o, "cpu"))
    am = act_mask_t(o, "cpu")[None]
    _, act, lp = actor.sample(r, g, m, am)
    assert act.shape == (1, R_MAX, N_ACT) and lp.shape == (1, R_MAX)
    assert (act[0, ~am[0].bool()] == 0).all() and (act[0, :, 1] >= 0).all()
    assert critic(r, g, m, act).shape == (1, R_MAX)
    fa = flat_action(act[0].detach().numpy())
    assert fa.shape == (N_ACT * R_MAX,) and np.allclose(fa[:R_MAX], act[0, :, 0].detach().numpy())


def test_crossmatch_virtual():
    from ahr_rl.crossmatch_study import _virtual
    from ahr_rl.ladder import PRICES, price_to_tick

    tick = np.array([[price_to_tick(2.0), price_to_tick(4.0), price_to_tick(5.0)]])
    size = np.array([[10.0, 20.0, 30.0]])
    live = np.ones((1, 3), bool)
    vt, vs = _virtual(tick, size, live, round_down=True)  # virtual price to back each runner
    # runner 0: 1 / (1 - 1/4 - 1/5) = 1.818 -> 1.81 (rounded against us); runner 1: 3.33 -> 3.30; runner 2: 4.0
    assert [float(PRICES[x]) for x in vt[0]] == [1.81, 3.3, 4.0]
    # size is limited by the thinnest other leg (size * price): runner 0 by runner 1 (80), others by runner 0 (20)
    assert np.allclose(vs[0], [80 / 1.81, 20 / 3.3, 20 / 4.0])
    vt_l, _ = _virtual(tick, size, live, round_down=False)  # lay side rounds up
    assert [float(PRICES[x]) for x in vt_l[0]] == [1.82, 3.35, 4.0]
    tick[0, 2] = -1  # one runner uncovered -> no virtual offers at all
    assert (_virtual(tick, size, live, True)[0] == -1).all()
    live[0, 2] = False  # ... unless that runner is no longer live
    assert (_virtual(tick, size, live, True)[0][0, :2] >= 0).all()


def test_passive_entry_fills_match_exchange():
    from ahr_rl.exchange import LAY, Exchange, ExchangeConfig
    from ahr_rl.ladder import PRICES
    from ahr_rl.passive_entry_study import ev_back, ev_lay, queue_fills, sim_order

    # queue: 10 ahead, 4 traded at our price, then 8 -> the queue is gone and 2 fill us
    assert np.allclose(queue_fills(10.0, np.array([20, 20, 20.]), np.array([4, 8, 0.]), np.zeros(3)), [0, 2, 0])
    # green at BSP: back 4.0 vs BSP 3.0 = +1/3 less commission; lay is the mirror image per $ of liability
    assert np.isclose(ev_back(4.0, 3.0, 0.1), (4 / 3 - 1) * 0.9) and np.isclose(ev_back(3.0, 4.0, 0.1), 3 / 4 - 1)
    assert np.isclose(ev_lay(3.0, 4.0, 0.0), (1 - 3 / 4) / 2)
    # the step-by-step replay gives the same fills as the exchange engine (cross-matching off)
    t = make_synthetic_tape(3)
    rng = np.random.default_rng(0)
    for _ in range(25):
        r, s0 = int(rng.integers(0, 6)), int(rng.integers(10, t.n_steps - 300))
        back = bool(rng.integers(0, 2))
        bb, bl = int(t.back_tick[s0, r, 0]), int(t.lay_tick[s0, r, 0])
        k = int(rng.integers(1, 4))
        P = bb + k if back else bl - k
        size = 10.0 if back else float(np.floor(10.0 / (PRICES[P] - 1) * 100) / 100)
        s1, end = s0 + 1, s0 + 240
        m = (t.trade_runner == r) & (t.trade_step > s1) & (t.trade_step <= end)
        cum, _ = sim_order(t, r, "back" if back else "lay", P, size, s1, end, t.trade_step[m],
                           t.trade_tick[m].astype(np.int32), t.trade_vol[m].astype(np.float64),
                           np.zeros(end - s1 + 1), 10.0 / t.dt)
        ex = Exchange(t, ExchangeConfig(bankroll=1e9, min_stake=0.0), start_step=s0)
        o = ex.submit(r, BACK if back else LAY, P, size)
        while ex.step < end:
            ex.advance()
        assert abs(min(cum[-1], size) - o.matched) < 1e-6, (r, s0, back, k, cum[-1], o.matched)


def test_signal_entry_no_lookahead(tmp_path):
    from ahr_rl.signal_entry_study import signal_rows

    t = make_synthetic_tape(5)
    t.went_in_play = True
    p1 = str(tmp_path / "20260901_1200_Test_1_1.npz")
    t.save(p1)
    a = signal_rows(p1, every_s=20.0, last_s=600.0)
    assert a is not None and {"sig_book_30", "sig_lag_15", "mom_30", "wom", "fwd_30"} <= set(a.columns)
    # change the book after the 400th step: signals at decision times before it must not move
    t.back_tick[400:] = np.clip(t.back_tick[400:] + 3, 0, None)
    t.lay_tick[400:] = np.clip(t.lay_tick[400:] + 3, 0, None)
    t.back_size[400:] *= 2.0
    p2 = str(tmp_path / "20260901_1200_Test_1_2.npz")
    t.save(p2)
    b = signal_rows(p2, every_s=20.0, last_s=600.0)
    cut = float(t.t_rel[399])
    sa, sb = a[a["t"] < cut - 30].reset_index(drop=True), b[b["t"] < cut - 30].reset_index(drop=True)
    cols = ["sig_book_15", "sig_book_30", "sig_lag_15", "sig_lag_30", "mom_30", "wom"]
    assert len(sa) and np.allclose(sa[cols].to_numpy(), sb[cols].to_numpy(), equal_nan=True)


def test_resting_agents():
    from ahr_rl.exchange import Exchange, ExchangeConfig
    from ahr_rl.features import R_MAX
    from ahr_rl.ladder import PRICES
    from ahr_rl.passive_agents import N_ACT3, RestingAgentConfig, RestingAgentEnv
    from ahr_rl.synthetic import make_bounce_tape

    t = make_bounce_tape(4)
    cfg = RestingAgentConfig(min_best_usd=0.0, random_start_s=0.0, order_cap=20.0, size_frac_cap=0.2)
    env = RestingAgentEnv([t], cfg)
    env.reset(options={"tape": t})
    a = np.zeros(N_ACT3 * R_MAX, np.float32)
    a[0], a[R_MAX], a[2 * R_MAX] = 1.0, 1.0, 0.9  # runner 0: back, full size, rest 3 ticks above best
    env.step(a)
    o1 = [o for o in env.ex.orders if o.runner == 0]
    assert len(o1) == 1 and o1[0].tick == t.back_tick[0, 0, 0] + 3  # decided at step 0
    # size capped at 20% of the money shown at the best level on its side (or the min stake)
    assert o1[0].size + o1[0].matched <= max(5.0, 0.2 * 1.5 * 30.0) + 1e-6
    oid = o1[0].oid
    env.step(a)  # the same order again (if the book has not moved) keeps its place
    o2 = [o for o in env.ex.orders if o.runner == 0]
    if o2 and o2[0].tick == o1[0].tick:
        assert o2[0].oid == oid
    # BSP exit: hold a position into the start, it is greened at BSP
    env2 = RestingAgentEnv([t], cfg)
    env2.reset(options={"tape": t})
    env2.ex._add_bet(1, BACK, float(t.bsp[1]) * 1.2, 10.0, passive=True)  # backed 20% above the BSP
    while True:
        _, _, done, _, info = env2.step(np.zeros(N_ACT3 * R_MAX, np.float32))
        if done:
            break
    assert abs(env2.ex.W[1] - env2.ex.L[1]) < 1e-6  # flat on runner 1
    assert np.isclose(info["worst"], 10.0 * 0.2 * (1 - 0.08))  # locked +20% of the stake, less commission
    # bounce control: crossing to back never beats fair, a back resting at the top does
    s0 = int(np.argmax(t.back_tick[:, 0, 0] == t.back_tick[:, 0, 0].max()))  # runner 0 at the top of its range
    assert PRICES[t.back_tick[s0, 0, 0]] <= t.bsp[0] + 1e-9
    # pessimistic fills: queue x2 and half the traded volume fill less than the normal model
    filled = {}
    for name, c in (("normal", ExchangeConfig()), ("pess", ExchangeConfig(queue_mult=2.0, trade_fill_frac=0.5))):
        ex = Exchange(t, replace_cfg(c), start_step=10)
        o = ex.submit(2, BACK, int(t.lay_tick[10, 2, 0]), 20.0)
        for _ in range(400):
            ex.advance()
        filled[name] = o.matched
    assert filled["pess"] <= filled["normal"]


def replace_cfg(c):
    from dataclasses import replace

    return replace(c, bankroll=1e9)


def test_top_races_selection_and_folds(tmp_path):
    from ahr_rl.top_races import folds_of, top_races

    paths = []
    for i, scale in enumerate((1.0, 5.0, 3.0)):
        t = make_synthetic_tape(i)
        t.total_matched = t.total_matched * scale
        p = str(tmp_path / f"2026090{i + 1}_1200_Test_1_{i}.npz")
        t.save(p)
        paths.append(p)
    top = top_races(paths, 2, workers=1)
    assert list(top["race"]) == ["20260902_1200_Test_1_1", "20260903_1200_Test_1_2"]  # by matched at the off
    f = folds_of([f"2026090{i}" for i in range(1, 10)] + ["20260910"], 5)
    assert len(f) == 5 and sum(len(x) for x in f) == 10 and f[0] == ["20260901", "20260902"]
    assert len({r for x in f for r in x}) == 10  # every race in exactly one fold


def test_arm_c_frozen_rule():
    from ahr_rl.arm_c_forward import ARM_C, decide, tape_rows

    assert ARM_C["registered"] == "20261024" and ARM_C["since"] == "20261025" and ARM_C["k"] == 3  # frozen
    s = dict(attempts=1000, races=100, ev_per_attempt_pct=0.5, t=2.5)
    assert decide(s) == "PASS" and decide(s | dict(t=1.0)) == "CONTINUE" and decide(s | dict(races=200, t=1.0)) == "FAIL"
    assert decide(s | dict(ev_per_attempt_pct=-0.1)) == "FAIL" and decide(s | dict(races=40)) == "CONTINUE"
    t = make_synthetic_tape(2)
    t.bsp = np.full(t.n_runners, 6.0, np.float32)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "20261026_1200_Test_1_1.npz")
        t.save(p)
        moms, _ = tape_rows(p, None)
        cut = float(np.quantile(moms["mom_30"], 0.10))
        _, att = tape_rows(p, cut)
    if att is not None:  # only 2nd-3rd favourites below the cut, backs 3 ticks above the best back price
        assert att["rank"].isin([2, 3]).all() and (att["mom_30"] <= cut + 1e-9).all()
        assert att["filled"].between(0, 1).all()
