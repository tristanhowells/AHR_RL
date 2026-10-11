"""Single-runner env on stream tapes. Fill mechanics themselves are ahr_rl's and
are covered by tests/test_exchange.py; these tests cover what single_runner adds."""
import os
import warnings
from dataclasses import replace

import numpy as np
import pytest

from ahr_rl.exchange import BACK, LAY, ExchangeConfig
from ahr_rl.ladder import PRICES, price_to_tick
from ahr_rl.synthetic import make_synthetic_tape
from single_runner.env import EnvConfig, SingleRunnerTradingEnv, episode_specs, obs_dim
from single_runner.exchange_x import CrossMatchingExchange
from single_runner.features import build_features, fair_price
from single_runner.parquet_tape import parquet_to_tape

PARQUET = os.path.join(os.path.dirname(__file__), "..", "2026-02-13_Newcastle_Race7_1.253949170.parquet")


@pytest.fixture(scope="module")
def real():
    return parquet_to_tape(PARQUET)


@pytest.fixture(scope="module")
def syn():
    return make_synthetic_tape(7)


def truncate(tape, s):
    """The tape as it would look at grid step s (rows <= s, trades <= s)."""
    keep = tape.trade_step <= s
    return replace(tape, t_rel=tape.t_rel[:s + 1], back_tick=tape.back_tick[:s + 1],
                   back_size=tape.back_size[:s + 1], lay_tick=tape.lay_tick[:s + 1],
                   lay_size=tape.lay_size[:s + 1], ltp_tick=tape.ltp_tick[:s + 1], tv=tape.tv[:s + 1],
                   spn=tape.spn[:s + 1], active=tape.active[:s + 1], suspended=tape.suspended[:s + 1],
                   total_matched=tape.total_matched[:s + 1], trade_step=tape.trade_step[keep],
                   trade_runner=tape.trade_runner[keep], trade_tick=tape.trade_tick[keep],
                   trade_vol=tape.trade_vol[keep], _trade_ptr=None)


# ---------------------------------------------------------------- features
@pytest.mark.parametrize("which", ["real", "syn"])
def test_features_are_causal(which, request):
    tape = request.getfixturevalue(which)
    full = build_features(tape, 0.08)
    s = tape.n_steps // 2
    part = build_features(truncate(tape, s), 0.08)
    np.testing.assert_allclose(full.runner[:s + 1], part.runner, atol=1e-5)
    np.testing.assert_allclose(full.glob[:s + 1], part.glob, atol=1e-5)


def test_features_ignore_result(syn):
    a = build_features(syn, 0.08)
    b = build_features(replace(syn, winner=(syn.winner + 1) % syn.n_runners, bsp=syn.bsp + 3.0), 0.08)
    np.testing.assert_array_equal(a.runner, b.runner)
    np.testing.assert_array_equal(a.glob, b.glob)


def test_features_quiet_runner_no_warnings(syn):
    quiet = syn.trade_runner != 0
    t = replace(syn, trade_step=syn.trade_step[quiet], trade_runner=syn.trade_runner[quiet],
                trade_tick=syn.trade_tick[quiet], trade_vol=syn.trade_vol[quiet], _trade_ptr=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = build_features(t, 0.08)
    assert np.isfinite(f.runner).all()


def test_fair_price_is_probability_microprice():
    b, l = price_to_tick(4.0), price_to_tick(5.0)
    # equal size: halfway in probability (1/4.5 = 0.2222...), not in price
    assert fair_price(b, 10.0, l, 10.0) == pytest.approx(1 / ((1 / 4 + 1 / 5) / 2))
    # heavy atb size (lay offers) pulls towards the atl price
    assert fair_price(b, 90.0, l, 10.0) > fair_price(b, 10.0, l, 90.0)
    assert fair_price(b, 5.0, -1, 0.0) == pytest.approx(4.0)
    assert fair_price(-1, 0.0, -1, 0.0) == 0.0


# ---------------------------------------------------------------- cross-matching
def _three_runner_tape(other_lay_prices, target_atb=3.0, target_atl=3.2):
    t = make_synthetic_tape(3, n_runners=3)
    bt, lt = t.back_tick.copy(), t.lay_tick.copy()
    bs, ls = t.back_size.copy(), t.lay_size.copy()
    bt[:, 0, :] = -1
    bt[:, 0, 0] = price_to_tick(target_atb)
    lt[:, 0, :] = -1
    lt[:, 0, 0] = price_to_tick(target_atl)
    bs[:, 0, 0] = ls[:, 0, 0] = 5.0
    for j, p in zip((1, 2), other_lay_prices):
        lt[:, j, 0] = price_to_tick(p)
        ls[:, j, 0] = 100.0
    return replace(t, back_tick=bt, lay_tick=lt, back_size=bs, lay_size=ls)


def test_cross_matching_virtual_back():
    # Backing runner 0 (displayed atb 3.0) via the other runners' atl offers:
    # 1/v = 1 - sum_j 1/p_j. Others at 2.0 and 1/(0.672 - 0.5) give v = 3.05, one tick better.
    t = _three_runner_tape((2.0, 1 / ((1 - 1 / 3.05) - 0.5)))
    ex = CrossMatchingExchange(t, ExchangeConfig(bankroll=100), 0, max_virtual_ticks=3)
    ticks, _ = ex._levels(0, 0, BACK)
    assert ticks[0] == price_to_tick(3.05) and ticks[1] == price_to_tick(3.0)
    ex.submit(0, BACK, price_to_tick(3.0), 10.0)
    ex.advance()                                          # latency: lands next step
    assert ex.fills[0].price == pytest.approx(3.05)
    off = CrossMatchingExchange(t, ExchangeConfig(bankroll=100), 0, cross_matching=False)
    assert len(off._levels(0, 0, BACK)[0]) == 1


def test_cross_matching_ignores_incoherent_or_impossible_books():
    # others at 1.62 and 4.3 imply v = 6.66, 30+ ticks past the displayed 3.0: stale book, ignored
    ex = CrossMatchingExchange(_three_runner_tape((1.62, 4.3)), ExchangeConfig(bankroll=100), 0)
    assert len(ex._levels(0, 0, BACK)[0]) == 1
    # others already sum past 100% -> no virtual price exists
    ex = CrossMatchingExchange(_three_runner_tape((1.5, 2.6)), ExchangeConfig(bankroll=100), 0)
    assert ex._virtual(0, 0, BACK) is None


# ---------------------------------------------------------------- green-up
@pytest.mark.parametrize("mode", ["fair", "cross"])
def test_green_value_matches_explicit_hedge(real, mode):
    ex = CrossMatchingExchange(real, ExchangeConfig(bankroll=100, latency_steps=0), 10)
    r = 0
    ex.W[r], ex.L[r] = 40.0, -12.0
    g, c = ex.green_value(r, 20, mode)
    if mode == "fair":
        h = (40.0 + 12.0) / c                          # lay hedge
        assert g == pytest.approx(40.0 - h * (c - 1)) == pytest.approx(-12.0 + h)
        assert g == pytest.approx(40.0 / c - 12.0 * (1 - 1 / c))   # market-implied expectation
    else:
        assert g <= ex.green_value(r, 20, "fair")[0] + 1e-9        # crossing pays the spread


# ---------------------------------------------------------------- environment
def test_episode_per_runner(real):
    env = SingleRunnerTradingEnv([real], EnvConfig(decision_every=1))
    specs = episode_specs(env, [real])
    assert [s["rank"] for s in specs] == list(range(1, 10))
    targets = {env.reset(options=s)[1]["target_runner"] for s in specs}
    assert len(targets) == 9


def test_do_nothing_is_zero(real):
    env = SingleRunnerTradingEnv([real], EnvConfig(decision_every=1))
    obs, _ = env.reset(options={"tape": real, "rank": 1})
    assert obs.shape == (obs_dim(),)
    done, n = False, 0
    while not done:
        obs, r, done, _, info = env.step(-np.ones(6))
        assert r == 0.0
        n += 1
    assert info["pnl"] == 0.0 and n == real.n_steps - 1


@pytest.mark.parametrize("which", ["real", "syn"])
@pytest.mark.parametrize("mode", ["fair", "cross"])
def test_reward_telescopes_and_risk_bounded(which, mode, request):
    tape = request.getfixturevalue(which)
    env = SingleRunnerTradingEnv([tape], EnvConfig(greenup_mode=mode, decision_every=2))
    for rank in (1, 3):
        env.reset(options={"tape": tape, "rank": rank})
        rng, ret, done = np.random.default_rng(rank), 0.0, False
        while not done:
            _, r, done, _, info = env.step(rng.uniform(-1, 1, 6))
            ret += r
            assert -env.ex.worst_case() <= env.cfg.initial_balance + 1e-6
        assert ret == pytest.approx(info["pnl"] / 100 * env.cfg.reward_scale, abs=1e-6)
        assert info["pnl"] >= -100 - 1e-6


def test_latency_no_fill_at_decision_step(real):
    env = SingleRunnerTradingEnv([real], EnvConfig(decision_every=1))
    env.reset(options={"tape": real, "rank": 1})
    s0 = env.ex.step
    env.step(np.array([1.0, -1.0, -1, 0, -1, -1]))        # aggressive back, max size
    assert all(f.step > s0 for f in env.ex.fills)


def test_min_stake_for_openers_but_not_hedges(real):
    env = SingleRunnerTradingEnv([real], EnvConfig(decision_every=1))
    env.reset(options={"tape": real, "rank": 1})
    env.step(np.array([0.02, -1.0, -1, 0, -1, -1]))       # $2 opening back -> rejected
    assert env.ex.n_rejected == 1 and not env.ex.orders and not env.ex.fills
    env.ex.W[env.target], env.ex.L[env.target] = 10.0, -6.0    # long the runner
    rej = env.ex.n_rejected
    env._place(LAY, 0.001, 0.0)                            # tiny lay that reduces the position
    assert env.ex.n_rejected == rej and env.ex.orders


def test_abandoned_race_not_eligible(syn):
    env = SingleRunnerTradingEnv([syn], EnvConfig())
    assert episode_specs(env, [replace(syn, went_in_play=False)]) == []


def test_agent_checkpoint_roundtrip(tmp_path, real):
    from single_runner.sac import SACAgent, SACConfig
    env = SingleRunnerTradingEnv([real], EnvConfig(decision_every=1))
    obs, _ = env.reset(options={"tape": real, "rank": 2})
    agent = SACAgent(obs_dim(), 6, SACConfig(hidden=(32, 32), device="cpu"))
    agent.norm.update(np.stack([obs, obs * 0.5]))
    agent.save(tmp_path / "a.pt")
    agent2, _ = SACAgent.load(tmp_path / "a.pt", device="cpu")
    np.testing.assert_allclose(agent.act(obs, True), agent2.act(obs, True), atol=1e-6)
