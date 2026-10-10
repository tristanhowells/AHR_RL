import os

import numpy as np
import pandas as pd
import pytest

from single_runner import ladder
from single_runner.env import EnvConfig, SingleRunnerTradingEnv, episode_specs, obs_dim
from single_runner.features import build_race
from single_runner.matching import MatchingEngine, Position, Snapshot, green_value

PARQUET = os.path.join(os.path.dirname(__file__), "..", "2026-02-13_Newcastle_Race7_1.253949170.parquet")
NAN = np.nan


def snap_(bp, bs, lp, ls, ltp=8.4, tv=0.0, fair=8.4, **kw):
    f = lambda x: np.array(x, dtype=float)
    return Snapshot(f(bp), f(bs), f(lp), f(ls), ltp, tv, fair, **{k: f(v) for k, v in kw.items()})


BOOK = dict(bp=[8.2, 8.0, 7.8], bs=[10, 20, 30], lp=[8.8, 9.0, 9.2], ls=[10, 20, 30])


# ---------------------------------------------------------------- ladder
def test_ladder():
    assert ladder.N_TICKS == 350
    assert ladder.shift(2.0, 1) == 2.02 and ladder.shift(2.0, -1) == 1.99
    assert ladder.shift(9.8, 1) == 10.0 and ladder.shift(1.01, -5) == 1.01
    assert ladder.snap(8.33) == 8.4 and ladder.snap(8.29, "down") == 8.2


# ---------------------------------------------------------------- green-up
@pytest.mark.parametrize("c", [1.5, 3.0, 8.4, 40.0])
def test_green_value_equalises_outcomes(c):
    pos = Position()
    pos.add("B", 8.0, 20)
    pos.add("L", 7.0, 5)
    W, L = pos.win_payoff, pos.lose_payoff
    h = abs(W - L) / c
    if W > L:   # hedge by laying
        Wh, Lh = W - h * (c - 1), L + h
    else:
        Wh, Lh = W + h * (c - 1), L - h
    assert Wh == pytest.approx(Lh)
    assert green_value(W, L, c) == pytest.approx(Lh)
    # unbiased: equals expected P&L under the implied probability 1/c
    assert green_value(W, L, c) == pytest.approx(W / c + L * (1 - 1 / c))


def test_round_trip_at_same_price_is_flat():
    e = MatchingEngine()
    e.pos.add("B", 5.0, 10)
    e.pos.add("L", 5.0, 10)
    assert e.green_up(snap_(**BOOK, fair=3.0))[0] == pytest.approx(0.0)


def test_cross_greenup_pays_spread_fair_does_not():
    s = snap_(**BOOK, fair=8.5)
    fair, cross = MatchingEngine(greenup_mode="fair"), MatchingEngine(greenup_mode="cross")
    for e in (fair, cross):
        e.pos.add("B", 8.5, 10)
    assert fair.green_up(s)[0] == pytest.approx(0.0)
    g, price, side, _ = cross.green_up(s)
    assert side == "L" and price == pytest.approx(8.8) and g < 0


# ---------------------------------------------------------------- aggressive matching
def test_aggressive_back_walks_book_with_price_improvement():
    e = MatchingEngine(cross_matching=False)
    m = e.place("B", 8.0, 25, snap_(**BOOK), t=0)
    assert m == pytest.approx(25)
    assert e.pos.back_stake == pytest.approx(25)
    assert e.pos.avg_back == pytest.approx((10 * 8.2 + 15 * 8.0) / 25)
    assert not e.orders


def test_aggressive_respects_depth_and_rests_remainder():
    e = MatchingEngine(cross_matching=False)
    m = e.place("L", 9.0, 50, snap_(**BOOK), t=0)
    assert m == pytest.approx(30)                       # 10 @ 8.8 + 20 @ 9.0
    assert len(e.orders) == 1 and e.orders[0].size == pytest.approx(20)
    assert e.orders[0].queue == 0                       # 9.0 beats best ATB 8.2 -> front of queue


def test_same_snapshot_liquidity_not_reused():
    e = MatchingEngine(cross_matching=False)
    s = snap_(**BOOK)
    e.place("B", 8.2, 6, s, t=0)
    assert e.place("B", 8.2, 6, s, t=0) == pytest.approx(4)


def test_cross_matching_virtual_back():
    # others' available-to-LAY sum to 1/2 + 1/4 = 0.75 -> virtual back price 4.0 > displayed 3.0
    s = snap_([3.0, NAN, NAN], [5, 0, 0], [3.2, NAN, NAN], [5, 0, 0], ltp=3.1, fair=3.1,
              others_back_p1=[1.9, 3.8], others_back_s1=[100, 100],
              others_lay_p1=[2.0, 4.0], others_lay_s1=[100, 100])
    e = MatchingEngine(cross_matching=True)
    e.place("B", 3.5, 10, s, t=0)
    assert e.fills[0].price == pytest.approx(4.0)
    assert e.pos.back_stake == pytest.approx(10)
    off = MatchingEngine(cross_matching=False)
    off.place("B", 3.5, 10, s, t=0)
    assert off.pos.back_stake == 0


# ---------------------------------------------------------------- passive fills
def test_passive_back_trade_through_and_queue():
    e = MatchingEngine(cross_matching=False)
    s0 = snap_(**BOOK, tv=100)
    e.place("B", 8.6, 10, s0, t=0)                      # inside spread -> front of queue
    assert e.orders[0].queue == 0
    # trades at 8.2 (backers hitting ATB) must NOT fill a resting back at 8.6
    e.on_new_snapshot(s0, snap_(**BOOK, ltp=8.2, tv=150), t=1)
    assert e.pos.back_stake == 0
    # trades at 8.8 went through 8.6 -> filled at our limit, capped by traded volume
    e.on_new_snapshot(s0, snap_(**BOOK, ltp=8.8, tv=104), t=2)
    assert e.pos.back_stake == pytest.approx(4) and e.pos.avg_back == pytest.approx(8.6)


def test_queue_estimate_behind_displayed_depth():
    e = MatchingEngine(cross_matching=False)
    e.place("B", 9.4, 5, snap_(**BOOK), t=0)            # worse than ATL level 3 (9.2)
    assert e.orders[0].queue == pytest.approx(20)       # mean displayed level size


def test_passive_trade_at_price_consumes_queue_first():
    e = MatchingEngine(cross_matching=False)
    s0 = snap_(**BOOK, tv=0)
    e.place("B", 8.8, 5, s0, t=0)                       # join ATL at 8.8 behind 10
    assert e.orders[0].queue == pytest.approx(10)
    e.on_new_snapshot(s0, snap_(**BOOK, ltp=8.8, tv=12), t=1)
    assert e.pos.back_stake == pytest.approx(2)


def test_passive_lay_crossed_book():
    e = MatchingEngine(cross_matching=False)
    s0 = snap_(**BOOK, tv=0)
    e.place("L", 8.4, 10, s0, t=0)
    crossed = snap_([8.0, 7.8, 7.6], [5, 5, 5], [8.3, 8.4, 8.6], [3, 4, 50], ltp=8.4, tv=0)
    e.on_new_snapshot(s0, crossed, t=1)
    assert e.pos.lay_stake == pytest.approx(7) and e.pos.avg_lay == pytest.approx(8.4)


# ---------------------------------------------------------------- exposure
def test_max_affordable_respects_balance():
    e = MatchingEngine(cross_matching=False)
    assert e.max_affordable("B", 5.0, 100) == pytest.approx(100)
    assert e.max_affordable("L", 5.0, 100) == pytest.approx(25)
    e.pos.add("B", 5.0, 40)                             # W=+160, L=-40
    assert e.max_affordable("B", 5.0, 100) == pytest.approx(60)
    assert e.max_affordable("L", 5.0, 100) == pytest.approx(65)   # (160+100)/4


# ---------------------------------------------------------------- environment
@pytest.fixture(scope="module")
def env():
    return SingleRunnerTradingEnv([PARQUET], EnvConfig(), seed=0)


def test_episode_per_runner(env):
    specs = episode_specs(env, [PARQUET])
    assert [s["rank"] for s in specs] == list(range(1, 10))
    targets = set()
    for s in specs:
        _, info = env.reset(options=s)
        targets.add(info["target_runner"])
        assert env.race.rank[env.t0, env.target] == s["rank"]
    assert len(targets) == 9


def test_do_nothing_is_zero_and_ends_at_in_play(env):
    obs, info = env.reset(options={"race_path": PARQUET, "rank": 1})
    assert obs.shape == (obs_dim(),)
    n, done = 0, False
    while not done:
        obs, r, done, trunc, info = env.step(-np.ones(6))
        n += 1
        assert r == 0.0
    assert n == 72 and info["pnl"] == 0.0               # first in-play row is 72


@pytest.mark.parametrize("rank", [1, 4, 9])
@pytest.mark.parametrize("mode", ["fair", "cross"])
def test_reward_telescopes_and_risk_is_bounded(rank, mode):
    e = SingleRunnerTradingEnv([PARQUET], EnvConfig(greenup_mode=mode), seed=rank)
    e.reset(options={"race_path": PARQUET, "rank": rank})
    rng, ret, done = np.random.default_rng(rank), 0.0, False
    while not done:
        _, r, done, _, info = e.step(rng.uniform(-1, 1, 6))
        ret += r
        W, L = e.engine.worst_case()
        assert -min(W, L) <= e.cfg.initial_balance + 1e-6
    assert ret == pytest.approx(info["pnl"] / 100 * 10, abs=1e-6)
    assert info["pnl"] >= -100 - 1e-6


def test_no_lookahead_and_no_result_leak():
    df = pd.read_parquet(PARQUET)
    full = build_race(df)
    k = 40
    part = build_race(df.iloc[:k + 1])
    np.testing.assert_allclose(full.runner_feats[:k + 1], part.runner_feats, atol=1e-6)
    np.testing.assert_allclose(full.global_feats[:k + 1], part.global_feats, atol=1e-6)
    leaked = df.copy()
    for i in range(9):
        leaked[f"run[{i}].is_winner"] = float(i == 0)
    leaked["result_winner_idx_first"] = 0
    alt = build_race(leaked)
    np.testing.assert_array_equal(full.runner_feats, alt.runner_feats)
    np.testing.assert_array_equal(full.global_feats, alt.global_feats)


def test_rolling_quiet_runner_no_warning():
    import warnings
    from single_runner.features import _rolling
    a = np.full((60, 2), np.nan)
    a[::3, 0] = np.arange(20.0)                         # runner 0 trades, runner 1 never does
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = _rolling(a, 45, np.nanmax)
    assert out[59, 0] == 19.0 and np.isnan(out[:, 1]).all()


def test_agent_checkpoint_roundtrip(tmp_path, env):
    from single_runner.sac import SACAgent, SACConfig
    agent = SACAgent(obs_dim(), 6, SACConfig(hidden=(32, 32), device="cpu"))
    obs, _ = env.reset(options={"race_path": PARQUET, "rank": 2})
    agent.norm.update(np.stack([obs, obs * 0.5]))
    agent.save(tmp_path / "a.pt")
    agent2, _ = SACAgent.load(tmp_path / "a.pt", device="cpu")
    np.testing.assert_allclose(agent.act(obs, True), agent2.act(obs, True), atol=1e-6)
