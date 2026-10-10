"""Betfair-style order matching + cross-matching simulator for ONE target runner.

Terminology (Betfair stream convention, matches the parquet columns):
  back_p / back_s  = "available to BACK" offers (resting LAY orders by others).
                     Level 1 is the HIGHEST price. A back order matches these.
  lay_p  / lay_s   = "available to LAY" offers (resting BACK orders by others).
                     Level 1 is the LOWEST price. A lay order matches these.

Order lifecycle
---------------
1. Aggressive part (at placement, against snapshot t):
   A BACK at limit p matches every available-to-back level priced >= p, best
   price first, at the *level's* price (Betfair price improvement), limited by
   displayed size. LAY is symmetric (levels priced <= q). Cross-matched
   (virtual) liquidity built from the other runners' books is added when it
   offers a strictly better price than the displayed level 1.
2. Passive part (rests at the limit price, filled on later snapshots):
   * crossed book : the new book offers the opposite side at a price at least
                    as good as our limit -> fill against that displayed (and
                    virtual) liquidity, at our limit price.
   * trade-through: runner traded volume increased and the LTP is strictly
                    through our price (LTP > p for a resting back, LTP < q for
                    a resting lay) -> those takers would have hit us first, so
                    fill up to the traded volume.
   * trade-at     : LTP == our price -> traded volume first consumes the queue
                    ahead of us, the rest fills us.
   * queue decay  : queue ahead = min(queue ahead, displayed size at our
                    price); if our price is better than the best same-side
                    offer we are at the front (queue 0).
   Unmatched orders lapse at the in-play transition (Betfair default LAPSE).

Why "realistic but optimistic"
------------------------------
Fills are bounded by displayed liquidity and by actual traded volume, and come
with natural adverse selection (we get filled when the price moves through
us). The optimistic assumptions are: zero latency, our orders don't move the
recorded book, all of a step's traded volume is attributed to the LTP side,
and size cancelled at our price is assumed to be ahead of us in the queue.
`fill_optimism` (< 1 = less optimistic) scales the traded volume available to
passive orders.

Green-up
--------
With matched backs (stake s_i @ p_i) and lays (stake l_j @ q_j) on one runner:
    W = sum s_i (p_i - 1) - sum l_j (q_j - 1)     (payoff if runner wins)
    L = -sum s_i + sum l_j                         (payoff if runner loses)
Hedging at price c with stake |W - L| / c (lay if W > L, back otherwise)
equalises both outcomes at
    G(c) = L + (W - L) / c.
G is outcome-independent. With c = fair price (microprice), G is exactly the
expected P&L under the market-implied probability 1/c, i.e. an unbiased
valuation: it neither charges nor gifts the spread and never looks at who won.
`mode='cross'` instead walks the real book (pessimistic: pays the spread).
"""
from dataclasses import dataclass, field

import numpy as np

from .ladder import snap

EPS = 1e-9


@dataclass
class Order:
    oid: int
    side: str           # 'B' (back) or 'L' (lay)
    price: float
    size: float         # unmatched stake remaining
    queue: float        # displayed volume ahead of us at our price
    placed_t: int


@dataclass
class Fill:
    side: str
    price: float
    size: float
    t: int
    kind: str           # 'aggressive' | 'crossed' | 'trade'


@dataclass
class Position:
    back_stake: float = 0.0
    back_value: float = 0.0     # sum stake * price
    lay_stake: float = 0.0
    lay_value: float = 0.0

    def add(self, side, price, size):
        if side == "B":
            self.back_stake += size
            self.back_value += size * price
        else:
            self.lay_stake += size
            self.lay_value += size * price

    @property
    def win_payoff(self):
        return (self.back_value - self.back_stake) - (self.lay_value - self.lay_stake)

    @property
    def lose_payoff(self):
        return self.lay_stake - self.back_stake

    @property
    def avg_back(self):
        return self.back_value / self.back_stake if self.back_stake > EPS else 0.0

    @property
    def avg_lay(self):
        return self.lay_value / self.lay_stake if self.lay_stake > EPS else 0.0


def green_value(W, L, c):
    """Outcome-independent P&L after hedging the whole position at price c."""
    return L + (W - L) / c


def commission_on(pnl, rate):
    return rate * max(pnl, 0.0)


@dataclass
class Snapshot:
    """Book of the target runner + level-1 of every runner at one time step."""
    back_p: np.ndarray      # [3]
    back_s: np.ndarray      # [3]
    lay_p: np.ndarray       # [3]
    lay_s: np.ndarray       # [3]
    ltp: float
    tv: float
    fair: float
    others_back_p1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    others_back_s1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    others_lay_p1: np.ndarray = field(default_factory=lambda: np.zeros(0))
    others_lay_s1: np.ndarray = field(default_factory=lambda: np.zeros(0))


def snapshot_from_race(race, t, r):
    others = np.array([j for j in range(race.n_runners) if j != r and race.active[t, j]], dtype=int)
    return Snapshot(
        back_p=race.back_p[t, r], back_s=race.back_s[t, r],
        lay_p=race.lay_p[t, r], lay_s=race.lay_s[t, r],
        ltp=float(race.ltp[t, r]), tv=float(race.tv[t, r]), fair=float(race.fair[t, r]),
        others_back_p1=race.back_p[t, others, 0], others_back_s1=race.back_s[t, others, 0],
        others_lay_p1=race.lay_p[t, others, 0], others_lay_s1=race.lay_s[t, others, 0],
    )


def _virtual(prices, sizes, rounding):
    """Cross-matched offer generated from the other runners' level-1 offers.

    For backing the target, the other runners' available-to-LAY offers (their
    resting backers) combine with us: 1/v = 1 - sum_j 1/p_j. Laying the target
    uses their available-to-BACK offers the same way. Size is limited by the
    thinnest leg: v * stake <= min_j(size_j * p_j).
    """
    if len(prices) == 0 or np.any(np.isnan(prices)) or np.any(sizes <= 0):
        return None
    rest = float(np.sum(1.0 / prices))
    if rest >= 1.0 - 1e-6:
        return None
    v = 1.0 / (1.0 - rest)
    if v > 1000.0 or v < 1.01:
        return None
    v = snap(v, rounding)
    size = float(np.min(sizes * prices) / v)
    return (v, size) if size > 0.01 else None


class MatchingEngine:
    def __init__(self, cross_matching=True, fill_optimism=1.0, greenup_mode="fair"):
        assert greenup_mode in ("fair", "cross", "ltp")
        self.cross_matching = cross_matching
        self.fill_optimism = float(fill_optimism)
        self.greenup_mode = greenup_mode
        self.reset()

    def reset(self):
        self.pos = Position()
        self.orders = []
        self.fills = []
        self._oid = 0
        self._consumed = {}      # (t, side, price) -> size used by us at this snapshot
        self._consumed_t = None

    # ------------------------------------------------------------ liquidity
    def _opposite_levels(self, snap_, side, t):
        """Liquidity an order of `side` can take: [(price, available_size)], best first."""
        if self._consumed_t != t:
            self._consumed, self._consumed_t = {}, t
        if side == "B":
            levels = [(float(p), float(s)) for p, s in zip(snap_.back_p, snap_.back_s)
                      if not np.isnan(p) and s > 0]
            if self.cross_matching:
                v = _virtual(snap_.others_lay_p1, snap_.others_lay_s1, "down")
                if v and (not levels or v[0] > levels[0][0] + EPS):
                    levels.append(v)
            levels.sort(key=lambda x: -x[0])
        else:
            levels = [(float(p), float(s)) for p, s in zip(snap_.lay_p, snap_.lay_s)
                      if not np.isnan(p) and s > 0]
            if self.cross_matching:
                v = _virtual(snap_.others_back_p1, snap_.others_back_s1, "up")
                if v and (not levels or v[0] < levels[0][0] - EPS):
                    levels.append(v)
            levels.sort(key=lambda x: x[0])
        out = []
        for p, s in levels:
            avail = s - self._consumed.get((side, p), 0.0)
            if avail > EPS:
                out.append((p, avail))
        return out

    def _consume(self, side, price, size):
        self._consumed[(side, price)] = self._consumed.get((side, price), 0.0) + size

    @staticmethod
    def _qualifies(side, level_price, limit):
        return level_price >= limit - EPS if side == "B" else level_price <= limit + EPS

    @staticmethod
    def initial_queue(snap_, side, price):
        """Displayed same-side volume at our price when we start resting."""
        if side == "B":   # resting back sits on the available-to-LAY ladder
            ps, ss = snap_.lay_p, snap_.lay_s
            better = np.isnan(ps[0]) or price < ps[0] - EPS
        else:             # resting lay sits on the available-to-BACK ladder
            ps, ss = snap_.back_p, snap_.back_s
            better = np.isnan(ps[0]) or price > ps[0] + EPS
        if better:
            return 0.0
        for p, s in zip(ps, ss):
            if not np.isnan(p) and abs(p - price) < EPS:
                return float(s)
        valid = ss[~np.isnan(ps)]
        return float(valid.mean()) if len(valid) else 0.0   # behind displayed depth: estimate

    # ------------------------------------------------------------ exposure
    def worst_case(self):
        """(W, L) including the worst-case effect of unmatched orders (Betfair style)."""
        W, L = self.pos.win_payoff, self.pos.lose_payoff
        for o in self.orders:
            if o.side == "B":
                L -= o.size
            else:
                W -= o.size * (o.price - 1.0)
        return W, L

    def exposure(self):
        W, L = self.worst_case()
        return max(0.0, -min(W, L))

    def max_affordable(self, side, price, balance):
        """Largest stake keeping worst-case loss <= balance."""
        W, L = self.worst_case()
        if side == "B":
            return max(0.0, L + balance)
        return max(0.0, (W + balance) / max(price - 1.0, EPS))

    # ------------------------------------------------------------ orders
    def place(self, side, price, size, snap_, t, max_open=20):
        """Submit a limit order. Returns matched size (aggressive part)."""
        if size <= 0:
            return 0.0
        remaining = size
        matched = 0.0
        for p, avail in self._opposite_levels(snap_, side, t):
            if remaining <= EPS or not self._qualifies(side, p, price):
                break
            q = min(remaining, avail)
            self._consume(side, p, q)
            self.pos.add(side, p, q)
            self.fills.append(Fill(side, p, q, t, "aggressive"))
            remaining -= q
            matched += q
        n_open = sum(1 for o in self.orders if o.side == side)
        if remaining > 0.01 and n_open < max_open:
            self._oid += 1
            self.orders.append(Order(self._oid, side, price, remaining,
                                     self.initial_queue(snap_, side, price), t))
        return matched

    def cancel(self, side=None):
        n = len(self.orders)
        self.orders = [o for o in self.orders if side is not None and o.side != side]
        return n - len(self.orders)

    def lapse_all(self):
        return self.cancel(None)

    def on_new_snapshot(self, prev, snap_, t):
        """Passive fills for resting orders on the transition prev -> snap_."""
        if not self.orders:
            return 0.0, 0.0
        dvol = max(snap_.tv - prev.tv, 0.0) * self.fill_optimism
        ltp = snap_.ltp
        pools = {"B": dvol, "L": dvol}
        filled = {"B": 0.0, "L": 0.0}
        # price priority: most competitive resting orders first
        backs = sorted([o for o in self.orders if o.side == "B"], key=lambda o: (o.price, o.oid))
        lays = sorted([o for o in self.orders if o.side == "L"], key=lambda o: (-o.price, o.oid))
        for o in backs + lays:
            s = o.side
            # 1) crossed book
            for p, avail in self._opposite_levels(snap_, s, t):
                if o.size <= EPS or not self._qualifies(s, p, o.price):
                    break
                q = min(o.size, avail)
                self._consume(s, p, q)
                self._fill(o, q, t, "crossed")
                filled[s] += q
            # 2) traded volume through / at our price
            if o.size > EPS and pools[s] > EPS and not np.isnan(ltp):
                through = ltp > o.price + EPS if s == "B" else ltp < o.price - EPS
                at = abs(ltp - o.price) < EPS
                if at:
                    eat = min(o.queue, pools[s])
                    o.queue -= eat
                    pools[s] -= eat
                if through or (at and o.queue <= EPS):
                    q = min(o.size, pools[s])
                    pools[s] -= q
                    if q > EPS:
                        self._fill(o, q, t, "trade")
                        filled[s] += q
            # 3) queue decay against the new displayed book
            o.queue = min(o.queue, self.initial_queue(snap_, s, o.price))
        self.orders = [o for o in self.orders if o.size > 0.01]
        return filled["B"], filled["L"]

    def _fill(self, o, q, t, kind):
        o.size -= q
        self.pos.add(o.side, o.price, q)
        self.fills.append(Fill(o.side, o.price, q, t, kind))

    # ------------------------------------------------------------ valuation
    def green_up(self, snap_):
        """(greened P&L, average hedge price, hedge side, hedge stake). Pure: no state change."""
        W, L = self.pos.win_payoff, self.pos.lose_payoff
        if abs(W - L) < 1e-9:
            return L, 0.0, None, 0.0
        if self.greenup_mode in ("fair", "ltp"):
            c = snap_.fair if self.greenup_mode == "fair" else snap_.ltp
            c = c if (c and not np.isnan(c) and c > 1.0) else snap_.fair
            side = "L" if W > L else "B"
            return green_value(W, L, c), c, side, abs(W - L) / c
        # 'cross': walk the real book. W > L -> hedge by laying (take available-to-LAY)
        side = "L" if W > L else "B"
        if side == "L":
            levels = [(p, s) for p, s in zip(snap_.lay_p, snap_.lay_s) if not np.isnan(p) and s > 0]
        else:
            levels = [(p, s) for p, s in zip(snap_.back_p, snap_.back_s) if not np.isnan(p) and s > 0]
        if not levels:
            levels = [(snap_.fair, np.inf)]
        stake_tot = val_tot = 0.0
        for i, (p, s) in enumerate(levels):
            need = abs(W - L) / p
            h = need if i == len(levels) - 1 else min(need, s)   # remainder at the last level
            if side == "L":
                W -= h * (p - 1.0)
                L += h
            else:
                W += h * (p - 1.0)
                L -= h
            stake_tot += h
            val_tot += h * p
            if abs(W - L) < 1e-9:
                break
        return L, val_tot / max(stake_tot, EPS), side, stake_tot
