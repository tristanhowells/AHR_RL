"""Betfair CLASSIC price ladder utilities.

Prices on Betfair odds markets can only sit on discrete ticks. Everything in the
simulator (order prices, book levels, features) is expressed in tick indices
0..N_TICKS-1 so that "one tick better" is always +/-1 regardless of price band.
"""
from __future__ import annotations

import numpy as np

# (band_start, band_end, increment)
_BANDS = [
    (1.01, 2.0, 0.01),
    (2.0, 3.0, 0.02),
    (3.0, 4.0, 0.05),
    (4.0, 6.0, 0.1),
    (6.0, 10.0, 0.2),
    (10.0, 20.0, 0.5),
    (20.0, 30.0, 1.0),
    (30.0, 50.0, 2.0),
    (50.0, 100.0, 5.0),
    (100.0, 1000.0, 10.0),
]


def _build() -> np.ndarray:
    out = []
    for lo, hi, inc in _BANDS:
        p = lo
        while p < hi - 1e-9:
            out.append(round(p, 2))
            p += inc
    out.append(1000.0)
    return np.array(sorted(set(out)), dtype=np.float64)


PRICES: np.ndarray = _build()
N_TICKS: int = len(PRICES)  # 350
MIN_PRICE, MAX_PRICE = float(PRICES[0]), float(PRICES[-1])
_LOOKUP = {round(float(p), 2): i for i, p in enumerate(PRICES)}


def price_to_tick(price: float) -> int:
    """Exact tick for a ladder price, or nearest tick for an off-ladder value."""
    i = _LOOKUP.get(round(float(price), 2))
    if i is not None:
        return i
    return int(np.abs(PRICES - price).argmin())


def tick_to_price(tick: int) -> float:
    return float(PRICES[int(np.clip(tick, 0, N_TICKS - 1))])


def clamp_tick(tick: int) -> int:
    return int(np.clip(tick, 0, N_TICKS - 1))


def ticks_between(p1: float, p2: float) -> int:
    return price_to_tick(p2) - price_to_tick(p1)
