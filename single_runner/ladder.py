"""Betfair price ladder (tick) helpers.

All prices the agent submits are snapped onto the real Betfair ladder
(1.01 .. 1000), so tick offsets mean the same thing they mean on the exchange.
"""
import numpy as np

# (upper bound of band, tick increment) — standard Betfair odds ladder
_BANDS = [(2.0, 0.01), (3.0, 0.02), (4.0, 0.05), (6.0, 0.1), (10.0, 0.2),
          (20.0, 0.5), (30.0, 1.0), (50.0, 2.0), (100.0, 5.0), (1000.0, 10.0)]


def _build_ladder():
    prices = [1.01]
    for hi, inc in _BANDS:
        p = prices[-1]
        while p < hi - 1e-9:
            p = round(p + inc, 2)
            prices.append(p)
    return np.array(prices, dtype=np.float64)


LADDER = _build_ladder()
N_TICKS = len(LADDER)
MIN_PRICE = float(LADDER[0])
MAX_PRICE = float(LADDER[-1])


def price_to_tick(price, side="nearest"):
    """Index of `price` on the ladder.

    side='nearest' rounds to the closest tick, 'down' to the tick at or below,
    'up' to the tick at or above. Off-ladder prices are clipped.
    """
    p = float(np.clip(price, MIN_PRICE, MAX_PRICE))
    i = int(np.searchsorted(LADDER, p - 1e-9))  # first tick >= p
    i = min(i, N_TICKS - 1)
    if abs(LADDER[i] - p) < 1e-9:
        return i
    if side == "up":
        return i
    if side == "down":
        return max(i - 1, 0)
    lo = max(i - 1, 0)
    return lo if (p - LADDER[lo]) <= (LADDER[i] - p) else i


def tick_to_price(i):
    return float(LADDER[int(np.clip(i, 0, N_TICKS - 1))])


def snap(price, side="nearest"):
    return tick_to_price(price_to_tick(price, side))


def shift(price, ticks):
    """Move `price` by an integer number of ladder ticks (clipped to 1.01..1000)."""
    return tick_to_price(price_to_tick(price) + int(ticks))


def ticks_between(p_lo, p_hi):
    """Signed tick distance from p_lo to p_hi."""
    return price_to_tick(p_hi) - price_to_tick(p_lo)
