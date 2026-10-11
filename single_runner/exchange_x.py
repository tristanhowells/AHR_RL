"""ahr_rl's conservative Exchange + Betfair cross-matching + single-runner green-up.

Everything about fills (latency, queue position, trade-driven passive fills,
liquidity memory, funds check, min stake, commission, scratchings) is the
research simulator in ahr_rl/exchange.py, unchanged. This module adds:

Cross-matching
    The recorded atb/atl ladders are the raw ones (no virtual bets), so Betfair's
    cross-matcher is added on top. Backing runner r can match the other runners'
    best atl offers jointly: 1/v = 1 - sum_{j != r} 1/p_j, rounded down to the
    ladder (worse for us), size limited by the thinnest leg,
    v * stake <= min_j(size_j * p_j). Laying uses their best atb offers and rounds
    up. The virtual level is used only when it beats the displayed best price,
    and it goes through the same liquidity memory as real levels. (The research
    found this happens in 0.4-2% of snapshots, so it rarely matters.)
    A virtual price more than `max_virtual_ticks` better than the displayed one
    is ignored: on a live exchange the cross-matcher would already have matched
    such offers, so a gap that large means the snapshot is stale or incoherent
    (e.g. other runners' offers summing past 100%), and an agent would learn to
    farm it.

Green-up
    With W / L = the target's payoff if it wins / loses, hedging at price c
    equalises both outcomes at G(c) = L + (W - L) / c. At the fair price c = 1/q
    (q = probability-space microprice) G = qW + (1-q)L is the market's own
    expectation: no spread charged or gifted, and the winner is never used.
"""
import numpy as np

from ahr_rl.exchange import BACK, LAY, Exchange
from ahr_rl.ladder import N_TICKS, PRICES

from .features import fair_price


class CrossMatchingExchange(Exchange):
    def __init__(self, tape, cfg=None, start_step=0, cross_matching=True, max_virtual_ticks=3):
        super().__init__(tape, cfg, start_step)
        self.cross_matching = cross_matching
        self.max_virtual_ticks = max_virtual_ticks
        self._vcache = {}

    def _virtual(self, step, runner, book):
        key = (step, runner, book)
        if key in self._vcache:
            return self._vcache[key]
        t = self.tape
        others = t.active[step] & ~self.void_runner
        others[runner] = False
        out = None
        if others.any():
            # backing r uses the others' atl (lay side); laying r uses their atb
            ticks = (t.lay_tick if book == BACK else t.back_tick)[step, others, 0].astype(np.int64)
            sizes = (t.lay_size if book == BACK else t.back_size)[step, others, 0].astype(np.float64)
            if (ticks >= 0).all() and (sizes > 0).all():
                p = PRICES[ticks]
                rest = float((1.0 / p).sum())
                if rest < 1.0 - 1e-6:
                    v = 1.0 / (1.0 - rest)
                    if 1.01 <= v <= 1000.0:
                        if book == BACK:   # round down: worse price for the backer
                            tk = int(np.searchsorted(PRICES, v + 1e-9) - 1)
                        else:              # round up: worse price for the layer
                            tk = int(np.searchsorted(PRICES, v - 1e-9))
                        tk = int(np.clip(tk, 0, N_TICKS - 1))
                        size = float((sizes * p).min() / PRICES[tk])
                        if size > 0.01:
                            out = (tk, size)
        if len(self._vcache) > 4096:
            self._vcache.clear()
        self._vcache[key] = out
        return out

    def _levels(self, step, runner, book):
        ticks, sizes = super()._levels(step, runner, book)
        if not self.cross_matching:
            return ticks, sizes
        v = self._virtual(step, runner, book)
        if v is None:
            return ticks, sizes
        if len(ticks) == 0:
            return ticks, sizes          # no displayed price to sanity-check against
        gain = (v[0] - ticks[0]) if book == BACK else (ticks[0] - v[0])
        if not 0 < gain <= self.max_virtual_ticks:
            return ticks, sizes
        return (np.concatenate([[v[0]], ticks]).astype(ticks.dtype),
                np.concatenate([[v[1]], sizes]).astype(np.float32))

    # ------------------------------------------------------------ single-runner valuation
    def fair_at(self, step, runner):
        t = self.tape
        c = float(fair_price(t.back_tick[step, runner, 0], t.back_size[step, runner, 0],
                             t.lay_tick[step, runner, 0], t.lay_size[step, runner, 0]))
        if c <= 1.0 and t.ltp_tick[step, runner] >= 0:
            c = float(PRICES[t.ltp_tick[step, runner]])
        return c

    def green_value(self, runner, step, mode="fair"):
        """(greened gross P&L, hedge price) for `runner` at `step`."""
        W, L = float(self.W[runner]), float(self.L[runner])
        if self.void_runner[runner] or abs(W - L) < 1e-9:
            return (0.0 if self.void_runner[runner] else L), 0.0
        if mode == "cross":
            Wf, Lf = self._flattened(step)
            return float(min(Wf[runner], Lf[runner])), 0.0
        c = self.fair_at(step, runner)
        if c <= 1.0:            # no book and no trade: value un-hedged worst case
            return min(W, L), 0.0
        return L + (W - L) / c, c

    def net(self, gross):
        return gross - self.commission * max(gross, 0.0)

    def runner_bets(self, runner):
        """(back stake, avg back price, lay stake, avg lay price) of matched bets."""
        r = np.asarray(self.bet_runner)
        if not len(r):
            return 0.0, 0.0, 0.0, 0.0
        m = r == runner
        s, p, st = np.asarray(self.bet_side)[m], np.asarray(self.bet_price)[m], np.asarray(self.bet_stake)[m]
        b, l = s == BACK, s == LAY
        bs, ls = st[b].sum(), st[l].sum()
        return (float(bs), float((st[b] * p[b]).sum() / bs) if bs > 0 else 0.0,
                float(ls), float((st[l] * p[l]).sum() / ls) if ls > 0 else 0.0)
