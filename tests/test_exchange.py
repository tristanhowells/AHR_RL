import numpy as np
import pytest

from ahr_rl.exchange import BACK, LAY, Exchange, ExchangeConfig
from ahr_rl.ladder import N_TICKS, PRICES, price_to_tick, tick_to_price
from ahr_rl.tape import Tape


def make_tape(T=20, R=2, back=((4.0, 100.0),), lay=((4.2, 100.0),), trades=(), removals=(),
              base_rate=8.0, winner=0):
    """Constant book: `back` = atb levels (best first), `lay` = atl levels."""
    K = 4
    bt = np.full((T, R, K), -1, np.int16); bs = np.zeros((T, R, K), np.float32)
    lt = np.full((T, R, K), -1, np.int16); ls = np.zeros((T, R, K), np.float32)
    for r in range(R):
        for k, (p, s) in enumerate(back):
            bt[:, r, k], bs[:, r, k] = price_to_tick(p), s
        for k, (p, s) in enumerate(lay):
            lt[:, r, k], ls[:, r, k] = price_to_tick(p), s
    tr = np.array(trades, np.float64).reshape(-1, 4)  # (step, runner, price, vol)
    rm = np.array(removals, np.float64).reshape(-1, 3)
    active = np.ones((T, R), bool)
    for s, r, _ in removals:
        active[int(s):, int(r)] = False
    return Tape(
        market_id="1.1", name="test", dt=0.5, t_rel=np.linspace(-600, 0, T).astype(np.float32),
        back_tick=bt, back_size=bs, lay_tick=lt, lay_size=ls,
        ltp_tick=np.full((T, R), -1, np.int16), tv=np.zeros((T, R), np.float32),
        spn=np.zeros((T, R), np.float32), active=active, suspended=np.zeros(T, bool),
        total_matched=np.zeros(T, np.float32),
        trade_step=tr[:, 0].astype(np.int32), trade_runner=tr[:, 1].astype(np.int16),
        trade_tick=np.array([price_to_tick(p) for p in tr[:, 2]], np.int16),
        trade_vol=tr[:, 3].astype(np.float32),
        removal_step=rm[:, 0].astype(np.int32), removal_runner=rm[:, 1].astype(np.int16),
        removal_factor=rm[:, 2].astype(np.float32),
        selection_ids=np.arange(R, dtype=np.int64), base_rate=base_rate, went_in_play=True,
        winner=winner, bsp=np.zeros(R, np.float32),
    )


def test_ladder():
    assert N_TICKS == 350
    assert PRICES[0] == 1.01 and PRICES[-1] == 1000
    for p in (1.01, 1.99, 2.0, 2.02, 3.05, 4.1, 6.2, 10.5, 21, 32, 55, 110, 1000):
        assert tick_to_price(price_to_tick(p)) == p
    assert price_to_tick(2.02) - price_to_tick(2.0) == 1


def test_aggressive_back_then_green():
    t = make_tape(back=((4.0, 100.0),), lay=((4.2, 100.0),))
    ex = Exchange(t, ExchangeConfig(commission=0.0))
    ex.submit(0, BACK, price_to_tick(4.0), 10.0)
    ex.advance()  # latency 1 step
    assert ex.W[0] == pytest.approx(30.0) and ex.L[0] == pytest.approx(-10.0)
    # hedge at 4.2 (lay): stake d/p = 40/4.2
    side, tk, stake = ex.hedge_plan(0)
    assert side == LAY and tick_to_price(tk) == 4.2
    ex.submit(0, side, tk, stake, is_hedge=True)
    ex.advance()
    pnl = ex.pnl_by_outcome()
    # equal across outcomes (to the 2dp stake rounding) and negative (crossed spread)
    assert abs(pnl[0] - pnl[1]) < 0.05
    assert pnl.max() < 0
    assert ex.green_value() == pytest.approx(pnl.min(), abs=0.05)


def test_green_value_matches_formula():
    t = make_tape(back=((3.0, 1000.0),), lay=((3.05, 1000.0),))
    ex = Exchange(t, ExchangeConfig(commission=0.0))
    # manually give a back 10 @ 4.0 on runner 0
    ex._add_bet(0, BACK, 4.0, 10.0, passive=False)
    # flatten by laying 40/3.05 at 3.05 -> locked = -10 + 40/3.05
    assert ex.green_value() == pytest.approx(-10 + 40 / 3.05, abs=1e-6)


def test_commission_only_on_profit():
    t = make_tape()
    ex = Exchange(t, ExchangeConfig(commission=0.08))
    net = ex.net_of_commission(np.array([10.0, -5.0]))
    assert net[0] == pytest.approx(9.2) and net[1] == -5.0


def test_funds_constraint_clips_lay_liability():
    t = make_tape(back=((100.0, 1e6),), lay=((110.0, 1e6),))
    ex = Exchange(t, ExchangeConfig(bankroll=500.0, min_stake=2.0))
    o = ex.submit(0, LAY, price_to_tick(110.0), 50.0)
    # liability 50*109 > 500 -> clipped to 500/109 (would be rejected at min_stake=5)
    assert o is not None and o.size == pytest.approx(np.floor(500 / 109 * 100) / 100)
    assert ex.available_funds() >= -1e-6
    # nothing left for another lay on the same runner
    assert ex.submit(0, LAY, price_to_tick(110.0), 5.0) is None


def test_clipped_below_min_stake_rejected():
    t = make_tape(back=((100.0, 1e6),), lay=((110.0, 1e6),))
    ex = Exchange(t, ExchangeConfig(bankroll=500.0, min_stake=5.0))
    assert ex.submit(0, LAY, price_to_tick(110.0), 50.0) is None


def test_funds_constraint_backs_across_runners():
    t = make_tape(R=3)
    ex = Exchange(t, ExchangeConfig(bankroll=500.0))
    ex.submit(0, BACK, price_to_tick(4.0), 300.0)
    o = ex.submit(1, BACK, price_to_tick(4.0), 300.0)
    assert o.size == pytest.approx(200.0)  # runner 2 winning loses both stakes


def test_liquidity_consumed_once():
    t = make_tape(back=((4.0, 8.0), (3.9, 100.0)))
    ex = Exchange(t, ExchangeConfig(commission=0.0))
    ex.submit(0, BACK, price_to_tick(4.0), 5.0)
    ex.submit(0, BACK, price_to_tick(4.0), 5.0)
    ex.advance()
    assert sum(f.stake for f in ex.fills) == pytest.approx(8.0)
    # remainder of 2nd order rests at 4.0 on the lay side
    assert len(ex.orders) == 1 and ex.orders[0].size == pytest.approx(2.0)


def test_passive_queue():
    # our back at 4.2 joins 10 already queued; trades at 4.2 burn the queue first
    trades = [(3, 0, 4.2, 6.0), (4, 0, 4.2, 8.0)]
    t = make_tape(back=((4.0, 100.0),), lay=((4.2, 10.0),), trades=trades)
    ex = Exchange(t, ExchangeConfig(commission=0.0))
    ex.submit(0, BACK, price_to_tick(4.2), 5.0)
    ex.advance(); ex.advance()  # step 2: live, queue 10
    assert ex.orders[0].queue_ahead == pytest.approx(10.0)
    ex.advance()  # step 3: 6 traded -> queue 4
    assert not ex.fills and ex.orders[0].queue_ahead == pytest.approx(4.0)
    ex.advance()  # step 4: 8 traded -> 4 burn queue, 4 fill us
    assert sum(f.stake for f in ex.fills) == pytest.approx(4.0)
    assert ex.fills[0].price == 4.2 and ex.fills[0].passive


def test_trade_through_fills_resting_order():
    t = make_tape(back=((4.0, 100.0),), lay=((4.3, 50.0),), trades=[(3, 0, 4.4, 3.0)])
    ex = Exchange(t, ExchangeConfig(commission=0.0))
    ex.submit(0, BACK, price_to_tick(4.2), 5.0)  # improves on best offer 4.3 -> front of queue
    for _ in range(3):
        ex.advance()
    assert sum(f.stake for f in ex.fills) == pytest.approx(3.0)


def test_unmatched_lapse_at_off():
    t = make_tape(T=5, back=((4.0, 100.0),), lay=((4.2, 100.0),))
    ex = Exchange(t, ExchangeConfig(commission=0.0))
    ex.submit(0, BACK, price_to_tick(5.0), 10.0)  # never matched
    for _ in range(4):
        ex.advance()
    res = ex.settle()
    assert res["worst"] == 0.0 and not ex.fills


def test_removal_reduction_factor_and_void():
    t = make_tape(R=3, removals=[(5, 2, 20.0)])
    ex = Exchange(t, ExchangeConfig(commission=0.0))
    ex._add_bet(0, BACK, 10.0, 10.0, passive=False)
    ex._add_bet(2, BACK, 5.0, 10.0, passive=False)
    for _ in range(5):
        ex.advance()
    assert ex.void_runner[2]
    assert ex.bet_price[0] == pytest.approx(8.0)  # 10 * (1 - 0.2)
    assert ex.W[2] == 0 and ex.L[2] == 0
    assert ex.W[0] == pytest.approx(70.0)
