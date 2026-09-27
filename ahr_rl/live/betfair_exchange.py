"""Real-money order routing via betfairlightweight.

``BetfairExchange`` is a drop-in replacement for the simulator ``Exchange`` used
by ``MarketSession``: the agent/bracket/auto-green logic is unchanged, but
submit/cancel go to Betfair and fills come from ``listCurrentOrders``.

!!! This path has NOT been exercised against the live exchange. Run the bot in
!!! paper mode first, then with tiny stakes, and watch it.
"""
from __future__ import annotations

import logging

import numpy as np

from ..exchange import BACK, Exchange, ExchangeConfig, Order
from ..ladder import PRICES

log = logging.getLogger(__name__)

try:
    from betfairlightweight import filters as bf_filters
except ImportError:  # pragma: no cover - only needed live
    bf_filters = None


class BetfairExchange(Exchange):
    def __init__(self, tape, cfg: ExchangeConfig, trading, market_id: str, customer_ref: str = "ahr_rl"):
        super().__init__(tape, cfg)
        if bf_filters is None:
            raise ImportError("pip install betfairlightweight")
        self.trading = trading
        self.market_id = market_id
        self.customer_ref = customer_ref
        self.bet_ids: dict[int, str] = {}  # our oid -> betfair betId
        self._matched_seen: dict[str, tuple[float, float]] = {}  # betId -> (size, size*avgprice)

    # ------------------------------------------------------------------ orders
    def submit(self, runner, side, tick, stake, is_hedge=False) -> Order | None:
        o = super().submit(runner, side, tick, stake, is_hedge)  # same funds / min-stake checks
        if o is None:
            return None
        o.live = True  # real orders are live as soon as Betfair accepts them
        try:
            if o.size >= self.cfg.min_stake:
                bet_id = self._place(runner, side, PRICES[o.tick], o.size)
            else:
                bet_id = self._place_below_min(runner, side, PRICES[o.tick], o.size)
        except Exception as e:  # never leave a phantom local order
            log.exception("place failed: %s", e)
            bet_id = None
        if bet_id is None:
            self.orders.remove(o)
            return None
        self.bet_ids[o.oid] = bet_id
        self._init_queue(o)
        return o

    def _instruction(self, runner, side, price, size):
        return bf_filters.place_instruction(
            order_type="LIMIT",
            selection_id=int(self.tape.selection_ids[runner]),
            side="BACK" if side == BACK else "LAY",
            limit_order=bf_filters.limit_order(size=round(size, 2), price=float(price), persistence_type="LAPSE"),
        )

    def _place(self, runner, side, price, size) -> str | None:
        r = self.trading.betting.place_orders(market_id=self.market_id,
                                              instructions=[self._instruction(runner, side, price, size)],
                                              customer_ref=None)
        rep = r.place_instruction_reports[0]
        if rep.status != "SUCCESS":
            log.warning("place rejected: %s %s", rep.status, rep.error_code)
            return None
        return rep.bet_id

    def _place_below_min(self, runner, side, price, size) -> str | None:
        """Standard workaround for sub-minimum hedges: place min stake at an
        unmatchable price, cancel it down to `size`, then move it to `price`."""
        far = 1000.0 if side == BACK else 1.01
        bet_id = self._place(runner, side, far, self.cfg.min_stake)
        if bet_id is None:
            return None
        self.trading.betting.cancel_orders(
            market_id=self.market_id,
            instructions=[bf_filters.cancel_instruction(bet_id=bet_id,
                                                        size_reduction=round(self.cfg.min_stake - size, 2))])
        r = self.trading.betting.replace_orders(
            market_id=self.market_id,
            instructions=[bf_filters.replace_instruction(bet_id=bet_id, new_price=float(price))])
        rep = r.replace_instruction_reports[0]
        if rep.status != "SUCCESS":
            log.warning("replace failed: %s", rep.status)
            return None
        return rep.place_instruction_reports.bet_id

    def _cancel(self, oids) -> None:
        ins = [bf_filters.cancel_instruction(bet_id=self.bet_ids[i]) for i in oids if i in self.bet_ids]
        if ins:
            try:
                self.trading.betting.cancel_orders(market_id=self.market_id, instructions=ins)
            except Exception as e:
                log.exception("cancel failed: %s", e)

    def cancel_runner(self, runner) -> None:
        self._cancel([o.oid for o in self.orders if o.runner == runner])

    def cancel_all(self) -> None:
        self._cancel([o.oid for o in self.orders])

    def cancel_order(self, oid) -> None:
        self._cancel([oid])

    # ------------------------------------------------------------------ time
    def advance(self) -> None:
        """New grid row: sync matched amounts from Betfair instead of simulating."""
        if self.step >= self.tape.n_steps - 1:
            return
        self.step += 1
        self._apply_removals()
        self._sync_orders()

    def _sync_orders(self) -> None:
        try:
            cur = self.trading.betting.list_current_orders(market_ids=[self.market_id])
        except Exception as e:
            log.exception("listCurrentOrders failed: %s", e)
            return
        by_bet = {co.bet_id: co for co in cur.orders}
        oid_of = {b: o for o, b in self.bet_ids.items()}
        for bet_id, co in by_bet.items():
            oid = oid_of.get(bet_id)
            if oid is None:
                continue
            m_size = float(co.size_matched or 0.0)
            m_val = m_size * float(co.average_price_matched or 0.0)
            prev_size, prev_val = self._matched_seen.get(bet_id, (0.0, 0.0))
            d_size = m_size - prev_size
            if d_size > 1e-9:
                price = (m_val - prev_val) / d_size
                runner = int(np.nonzero(self.tape.selection_ids == co.selection_id)[0][0])
                self._add_bet(runner, BACK if co.side == "BACK" else -1, price, d_size, passive=True)
                self._matched_seen[bet_id] = (m_size, m_val)
            for o in self.orders:
                if o.oid == oid:
                    o.size = float(co.size_remaining or 0.0)
                    o.matched = m_size
        live = set(by_bet)
        self.orders = [o for o in self.orders if o.size > 1e-9 and self.bet_ids.get(o.oid) in live]
