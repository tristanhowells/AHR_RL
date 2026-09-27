"""Exchange simulator: replays a Tape and matches the agent's orders against it.

Modelling choices (all conservative - it is much worse to train on a simulator that
flatters the agent than one that is slightly harsh):

Latency
    An order (or cancel) submitted after observing step ``s`` reaches the exchange
    at step ``s+1`` (``dt`` = 0.5s by default; measured stream latency ~130ms).

Aggressive orders
    Match against the historical ladder at step ``s+1`` level by level, at each
    level's price, up to the order's limit price and the size shown. Liquidity we
    take is remembered ("consumed") so we cannot take the same money twice; the
    memory expires when the level changes or after ``impact_decay_s``.

Passive orders
    Rest at the limit price with ``queue_ahead`` = size already at that price.
    Traded volume at our price first burns the queue, then fills us. Trades at a
    price *worse* for the counterparty than ours fill us directly (they would have
    hit us first). If the historical book later crosses our price, we match that
    liquidity at our price. Queue ahead can only shrink to the size still shown
    (cancellations assumed to come from ahead of us - optimistic but standard).

Funds
    Betfair exposure = worst case over outcomes of matched P&L plus, for every
    unmatched order, the worse of matched / unmatched. An order that would take
    worst-case loss beyond the bankroll is clipped (or rejected below min stake).

Settlement
    Unmatched orders lapse at the off. Commission is charged on net market profit
    for the realised outcome. Late scratchings void bets on the removed runner,
    cancel all unmatched orders and reduce matched prices on the other runners.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ladder import MIN_PRICE, N_TICKS, PRICES
from .tape import Tape

BACK, LAY = 1, -1


@dataclass
class ExchangeConfig:
    bankroll: float = 500.0
    min_stake: float = 5.0  # Betfair AU minimum stake (check your account)
    commission: float | None = None  # fraction, e.g. 0.08. None => market base rate
    commission_discount: float = 0.0  # fraction of base rate refunded
    latency_steps: int = 1
    impact_decay_s: float = 10.0
    reduction_factor_threshold: float = 2.5  # % below which RF is not applied
    # Hedges below min stake are placeable live via the standard workaround
    # (place min stake at an unmatchable price, cancel down, replace price).
    allow_sub_min_hedge: bool = True
    # Passive-fill model, for diagnosing how much results depend on execution:
    #   "realistic": queue behind the size already shown at our price; fills
    #                limited by traded volume (default, used for all real results)
    #   "no_queue":  as realistic but we are always first in the queue
    #   "touch":     first in queue AND the whole order fills on any trade at or
    #                through our price (the edge test's generous assumption)
    # In the two optimistic modes a new order can also be filled by trades in the
    # same half-second it goes live, and cancels are applied before fills.
    fill_mode: str = "realistic"


@dataclass
class Order:
    oid: int
    runner: int
    side: int  # BACK / LAY
    tick: int
    size: float  # remaining unmatched stake
    placed_step: int
    queue_ahead: float = 0.0
    matched: float = 0.0
    live: bool = False  # reached the exchange


@dataclass
class Fill:
    step: int
    runner: int
    side: int
    price: float
    stake: float
    passive: bool


class Exchange:
    def __init__(self, tape: Tape, cfg: ExchangeConfig | None = None, start_step: int = 0):
        self.tape = tape
        self.cfg = cfg or ExchangeConfig()
        self.R = tape.n_runners
        c = self.cfg.commission if self.cfg.commission is not None else tape.base_rate / 100.0
        self.commission = c * (1.0 - self.cfg.commission_discount)
        self.step = start_step
        self.orders: list[Order] = []
        self._pending_cancels: dict[int, int] = {}  # runner -> cancel orders with oid <= this
        self._pending_cancel_ids: set[int] = set()  # order ids
        self._oid = 0
        # matched bets (append-only lists; converted on demand)
        self.bet_runner: list[int] = []
        self.bet_side: list[int] = []
        self.bet_price: list[float] = []
        self.bet_stake: list[float] = []
        self.bet_step: list[int] = []
        self.void_runner = np.zeros(self.R, bool)
        self.fills: list[Fill] = []
        self._consumed: dict[tuple[int, int, int], list] = {}  # (runner, book, tick) -> [amt, step]
        self.W = np.zeros(self.R)  # P&L contribution of runner j if j wins
        self.L = np.zeros(self.R)  # P&L contribution of runner j if j loses
        self.n_rejected = 0
        self.turnover = 0.0

    # ================================================================ book views
    def _levels(self, step: int, runner: int, book: int):
        """book=BACK -> atb (what we can back into), best (highest) first.
        book=LAY -> atl (what we can lay into), best (lowest) first."""
        t = self.tape
        if book == BACK:
            ticks, sizes = t.back_tick[step, runner], t.back_size[step, runner]
        else:
            ticks, sizes = t.lay_tick[step, runner], t.lay_size[step, runner]
        n = int((ticks >= 0).sum())
        return ticks[:n], sizes[:n]

    def best(self, step: int, runner: int) -> tuple[int, int]:
        """(best back tick, best lay tick), -1 if empty."""
        t = self.tape
        return int(t.back_tick[step, runner, 0]), int(t.lay_tick[step, runner, 0])

    def _avail(self, step: int, runner: int, book: int, tick: int, shown: float) -> float:
        c = self._consumed.get((runner, book, tick))
        return max(0.0, shown - (c[0] if c else 0.0))

    def _consume(self, runner: int, book: int, tick: int, amt: float) -> None:
        key = (runner, book, tick)
        c = self._consumed.get(key)
        if c:
            c[0] += amt
            c[1] = self.step
        else:
            self._consumed[key] = [amt, self.step]

    def _decay_consumed(self) -> None:
        horizon = self.cfg.impact_decay_s / self.tape.dt
        dead = []
        for (r, book, tick), c in self._consumed.items():
            ticks, sizes = self._levels(self.step, r, book)
            hit = np.nonzero(ticks == tick)[0]
            shown = float(sizes[hit[0]]) if len(hit) else 0.0
            c[0] = min(c[0], shown)
            if c[0] <= 1e-9 or self.step - c[1] > horizon:
                dead.append((r, book, tick))
        for k in dead:
            del self._consumed[k]

    # ================================================================ accounting
    def _add_bet(self, runner: int, side: int, price: float, stake: float, passive: bool) -> None:
        if stake <= 1e-9:
            return
        self.bet_runner.append(runner)
        self.bet_side.append(side)
        self.bet_price.append(price)
        self.bet_stake.append(stake)
        self.bet_step.append(self.step)
        if side == BACK:
            self.W[runner] += stake * (price - 1.0)
            self.L[runner] -= stake
        else:
            self.W[runner] -= stake * (price - 1.0)
            self.L[runner] += stake
        self.turnover += stake
        self.fills.append(Fill(self.step, runner, side, price, stake, passive))

    def outcome_mask(self) -> np.ndarray:
        """Runners that can still win (active at the current step, not voided)."""
        return self.tape.active[self.step] & ~self.void_runner

    def pnl_by_outcome(self, W=None, L=None) -> np.ndarray:
        """Gross P&L if runner o wins, for every runner (use with outcome_mask)."""
        W = self.W if W is None else W
        L = self.L if L is None else L
        return W - L + L.sum()

    def net_of_commission(self, pnl: np.ndarray) -> np.ndarray:
        return np.where(pnl > 0, pnl * (1.0 - self.commission), pnl)

    def worst_case(self, include_unmatched: bool = True) -> float:
        """Betfair-style worst case (gross), used for the funds constraint."""
        pnl = self.pnl_by_outcome().copy()
        if include_unmatched:
            for o in self.orders:
                if o.size <= 0:
                    continue
                p = PRICES[o.tick]
                if o.side == BACK:  # loses stake unless runner wins
                    pnl -= o.size
                    pnl[o.runner] += o.size
                else:  # loses liability if runner wins
                    pnl[o.runner] -= o.size * (p - 1.0)
        m = self.outcome_mask()
        return float(pnl[m].min()) if m.any() else 0.0

    def available_funds(self) -> float:
        return self.cfg.bankroll + min(0.0, self.worst_case())

    def max_stake(self, runner: int, side: int, tick: int) -> float:
        """Largest stake that keeps worst-case loss within the bankroll."""
        B = self.cfg.bankroll
        pnl = self.pnl_by_outcome().copy()
        for o in self.orders:
            if o.size <= 0:
                continue
            if o.side == BACK:
                pnl -= o.size
                pnl[o.runner] += o.size
            else:
                pnl[o.runner] -= o.size * (PRICES[o.tick] - 1.0)
        m = self.outcome_mask()
        if side == BACK:
            others = m.copy()
            others[runner] = False
            if not others.any():
                return np.inf
            return max(0.0, float((pnl[others] + B).min()))
        p = PRICES[tick]
        return max(0.0, float(pnl[runner] + B) / max(p - 1.0, 1e-9))

    # ================================================================ order entry
    def submit(self, runner: int, side: int, tick: int, stake: float, is_hedge: bool = False) -> Order | None:
        t = self.tape
        if not (0 <= runner < self.R) or not t.active[self.step, runner] or self.void_runner[runner]:
            self.n_rejected += 1
            return None
        tick = int(np.clip(tick, 0, N_TICKS - 1))
        stake = min(stake, self.max_stake(runner, side, tick))
        stake = float(np.floor(stake * 100) / 100)  # Betfair: 2dp stakes
        floor = 0.01 if (is_hedge and self.cfg.allow_sub_min_hedge) else self.cfg.min_stake
        if stake < floor:
            self.n_rejected += 1
            return None
        self._oid += 1
        o = Order(self._oid, runner, side, tick, stake, self.step)
        self.orders.append(o)
        return o

    def cancel_runner(self, runner: int) -> None:
        self._pending_cancels[runner] = self._oid

    def cancel_all(self) -> None:
        for r in range(self.R):
            self._pending_cancels[r] = self._oid

    def cancel_order(self, oid: int) -> None:
        self._pending_cancel_ids.add(oid)

    # ================================================================ matching
    def _match_aggressive(self, o: Order, limit_tick: int, at_own_price: bool) -> None:
        """Match o against the opposite book at self.step up to limit_tick."""
        book = BACK if o.side == BACK else LAY  # BACK order consumes atb, LAY consumes atl
        ticks, sizes = self._levels(self.step, o.runner, book)
        for tk, sz in zip(ticks, sizes):
            if o.size <= 1e-9:
                break
            tk = int(tk)
            if (o.side == BACK and tk < limit_tick) or (o.side == LAY and tk > limit_tick):
                break
            avail = self._avail(self.step, o.runner, book, tk, float(sz))
            if avail <= 1e-9:
                continue
            x = min(avail, o.size)
            price = PRICES[limit_tick] if at_own_price else PRICES[tk]
            self._add_bet(o.runner, o.side, float(price), x, passive=at_own_price)
            self._consume(o.runner, book, tk, x)
            o.size -= x
            o.matched += x

    def _init_queue(self, o: Order) -> None:
        # our BACK rests among other backers' offers (atl), our LAY among layers' (atb)
        book = LAY if o.side == BACK else BACK
        ticks, sizes = self._levels(self.step, o.runner, book)
        if self.cfg.fill_mode != "realistic":
            o.queue_ahead = 0.0
            return
        hit = np.nonzero(ticks == o.tick)[0]
        if len(hit):
            o.queue_ahead = float(sizes[hit[0]])
        elif len(ticks) and ((o.side == BACK and o.tick > ticks[-1]) or (o.side == LAY and o.tick < ticks[-1])):
            o.queue_ahead = 1e9  # deeper than the visible ladder: treat as unreachable
        else:
            o.queue_ahead = 0.0

    def _passive_fills(self) -> None:
        """Fill resting orders from trades in (step-1, step] and book crossings at step."""
        tr_r, tr_t, tr_v = self.tape.trades_at(self.step)
        for o in self.orders:
            if not o.live or o.size <= 1e-9:
                continue
            sel = tr_r == o.runner
            if sel.any():
                tt, vv = tr_t[sel], tr_v[sel].astype(np.float64)
                through = float(vv[(tt > o.tick) if o.side == BACK else (tt < o.tick)].sum())
                at = float(vv[tt == o.tick].sum())
                burn = min(o.queue_ahead, at)
                o.queue_ahead -= burn
                x = min(o.size, through + at - burn)
                if self.cfg.fill_mode == "touch" and through + at > 0:
                    x = o.size  # any trade at/through our price fills us completely
                if x > 1e-9:
                    self._add_bet(o.runner, o.side, float(PRICES[o.tick]), x, passive=True)
                    o.size -= x
                    o.matched += x
            # queue can't be longer than what is still shown at our price
            book = LAY if o.side == BACK else BACK
            ticks, sizes = self._levels(self.step, o.runner, book)
            hit = np.nonzero(ticks == o.tick)[0]
            if len(hit):
                o.queue_ahead = min(o.queue_ahead, float(sizes[hit[0]]))
            elif o.queue_ahead < 1e9:
                o.queue_ahead = 0.0
            # historical book crossed our resting price -> we'd have been matched
            if o.size > 1e-9:
                self._match_aggressive(o, o.tick, at_own_price=True)

    def _apply_removals(self) -> None:
        t = self.tape
        idx = np.nonzero(t.removal_step == self.step)[0]
        if not len(idx):
            return
        for i in idx:
            j, af = int(t.removal_runner[i]), float(t.removal_factor[i])
            self.void_runner[j] = True
            if af >= self.cfg.reduction_factor_threshold:
                for b in range(len(self.bet_price)):
                    if self.bet_runner[b] != j and self.bet_step[b] <= self.step:
                        self.bet_price[b] = max(MIN_PRICE, self.bet_price[b] * (1.0 - af / 100.0))
        self.orders = []  # Betfair cancels unmatched bets on a non-runner
        self._recompute_book()

    def _recompute_book(self) -> None:
        self.W[:] = 0.0
        self.L[:] = 0.0
        for r, s, p, st in zip(self.bet_runner, self.bet_side, self.bet_price, self.bet_stake):
            if self.void_runner[r]:
                continue
            if s == BACK:
                self.W[r] += st * (p - 1.0)
                self.L[r] -= st
            else:
                self.W[r] -= st * (p - 1.0)
                self.L[r] += st

    def advance(self) -> None:
        """Move one tape step forward and run the matching engine."""
        if self.step >= self.tape.n_steps - 1:
            return
        self.step += 1
        self._decay_consumed()
        self._apply_removals()
        if self.tape.suspended[self.step]:
            return
        if self.cfg.fill_mode == "realistic":
            # conservative ordering: resting orders can fill before our cancel lands,
            # and new orders only see trades from the next interval on
            self._passive_fills()
            self._apply_cancels()
            self._activate_new()
        else:
            self._apply_cancels()
            self._activate_new()
            self._passive_fills()
        self.orders = [o for o in self.orders if o.size > 1e-9]

    def _apply_cancels(self) -> None:
        if self._pending_cancels or self._pending_cancel_ids:
            for o in self.orders:
                if o.oid <= self._pending_cancels.get(o.runner, 0) or o.oid in self._pending_cancel_ids:
                    o.size = 0.0
            self._pending_cancels.clear()
            self._pending_cancel_ids.clear()

    def _activate_new(self) -> None:
        for o in self.orders:
            if not o.live and o.size > 1e-9 and self.step - o.placed_step >= self.cfg.latency_steps:
                o.live = True
                if not self.tape.active[self.step, o.runner]:
                    o.size = 0.0
                    continue
                self._match_aggressive(o, o.tick, at_own_price=False)
                if o.size > 1e-9:
                    self._init_queue(o)

    # ================================================================ valuation
    def hedge_plan(self, runner: int, step: int | None = None):
        """(side, tick, stake) that flattens runner at the best opposite price, or None."""
        step = self.step if step is None else step
        d = self.W[runner] - self.L[runner]
        bb, bl = self.best(step, runner)
        if d > 1e-6 and bl >= 0:  # long the runner -> lay it
            return LAY, bl, d / PRICES[bl]
        if d < -1e-6 and bb >= 0:  # short the runner -> back it
            return BACK, bb, -d / PRICES[bb]
        return None

    def green_value(self, step: int | None = None) -> float:
        """Mark-to-market: worst-case net P&L if every runner were flattened now by
        crossing the spread (walking visible depth). Unhedgeable residual stays."""
        step = self.step if step is None else step
        W, L = self._flattened(step)
        return self._worst_net(W, L, step)

    def runner_values(self, step: int | None = None, w_unhedged: float = 0.0) -> np.ndarray:
        """Per-runner decomposition of the green value (used for per-runner credit
        assignment). Runner j flattened now locks min(W'_j, L'_j); the sum over
        runners is a lower bound on the portfolio worst case. ``w_unhedged``
        blends towards the un-hedged per-runner worst case min(W_j, L_j)."""
        step = self.step if step is None else step
        W, L = self._flattened(step)
        v = np.minimum(W, L)
        if w_unhedged > 0:
            v = (1 - w_unhedged) * v + w_unhedged * np.minimum(self.W, self.L)
        v[self.void_runner] = 0.0
        return v

    def _flattened(self, step: int):
        W, L = self.W.copy(), self.L.copy()
        for j in range(self.R):
            d = W[j] - L[j]
            if abs(d) < 1e-6 or self.void_runner[j]:
                continue
            book = LAY if d > 0 else BACK
            ticks, sizes = self._levels(step, j, book)
            for tk, sz in zip(ticks, sizes):
                p = PRICES[int(tk)]
                x = min(self._avail(step, j, book, int(tk), float(sz)), abs(d) / p)
                if d > 0:  # lay x at p
                    W[j] -= x * (p - 1.0)
                    L[j] += x
                else:  # back x at p
                    W[j] += x * (p - 1.0)
                    L[j] -= x
                d = W[j] - L[j]
                if abs(d) < 1e-6:
                    break
        return W, L

    def worst_net(self, step: int | None = None) -> float:
        """Worst-case net P&L of matched bets as they stand (no hedging)."""
        return self._worst_net(self.W, self.L, self.step if step is None else step)

    def _worst_net(self, W, L, step) -> float:
        m = self.tape.active[step] & ~self.void_runner
        if not m.any():
            return 0.0
        pnl = self.net_of_commission(self.pnl_by_outcome(W, L))
        return float(pnl[m].min())

    def settle(self) -> dict:
        """Called at the off: lapse unmatched orders and report results."""
        self.orders = []
        t = self.tape
        if not t.went_in_play:  # abandoned: everything void
            return dict(worst=0.0, best=0.0, realised=0.0, expected=0.0, green=True, void=True)
        step = t.n_steps - 1
        m = t.active[step] & ~self.void_runner
        net = self.net_of_commission(self.pnl_by_outcome())
        worst = float(net[m].min()) if m.any() else 0.0
        best = float(net[m].max()) if m.any() else 0.0
        realised = float(net[t.winner]) if t.winner >= 0 else float("nan")
        # market-implied expectation at the off (luck-free evaluation metric)
        mid = np.zeros(self.R)
        for j in np.nonzero(m)[0]:
            bb, bl = self.best(step, j)
            ps = [PRICES[x] for x in (bb, bl) if x >= 0]
            mid[j] = np.mean(ps) if ps else 0.0
        prob = np.where(mid > 1.0, 1.0 / np.maximum(mid, 1.01), 0.0)
        expected = float((prob * net).sum() / prob.sum()) if prob.sum() > 0 else worst
        return dict(worst=worst, best=best, realised=realised, expected=expected,
                    green=worst >= -1e-6, void=False)
