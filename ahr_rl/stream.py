"""Betfair Exchange Stream API (market change messages) -> in-memory order book.

The same `MarketCache` is used for
  * offline replay of the recorded ``*.ndjson.gz`` files (see ``read_recording``), and
  * live trading, where betfairlightweight hands us the identical raw ``mcm`` dicts.

Keeping one implementation guarantees the agent sees identically-built state in
training and in production.

Recording format (one JSON object per line):
  {"op": "meta", "market_id", "market_start_utc", "fields", ...}
  {"op": "mcm", "pt": <publish ms>, "mc": [{"id", "img"?, "marketDefinition"?, "rc": [...]}]}
  {"op": "trailer", "inplay_from_pt", "winners", "bsp", ...}

Runner change (rc) keys used: atb / atl (full depth, [price, size], size 0 deletes),
trd (cumulative traded volume per price), ltp, tv, spn, spf.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

try:  # orjson is ~5x faster on these files but optional
    import orjson as _json

    _loads = _json.loads
except ImportError:  # pragma: no cover
    _loads = json.loads


@dataclass
class RunnerBook:
    selection_id: int
    atb: dict = field(default_factory=dict)  # price -> size: unmatched LAYS you can back into
    atl: dict = field(default_factory=dict)  # price -> size: unmatched BACKS you can lay into
    trd: dict = field(default_factory=dict)  # price -> cumulative traded volume
    ltp: float = 0.0
    tv: float = 0.0
    spn: float = 0.0  # projected (near) BSP
    spf: float = 0.0  # projected (far) BSP
    status: str = "ACTIVE"
    adjustment_factor: float = 0.0
    sort_priority: int = 0

    def best_back(self) -> tuple[float, float]:
        """Best price you can BACK at (highest atb)."""
        if not self.atb:
            return 0.0, 0.0
        p = max(self.atb)
        return p, self.atb[p]

    def best_lay(self) -> tuple[float, float]:
        """Best price you can LAY at (lowest atl)."""
        if not self.atl:
            return 0.0, 0.0
        p = min(self.atl)
        return p, self.atl[p]

    def back_levels(self, k: int) -> list[tuple[float, float]]:
        return sorted(self.atb.items(), key=lambda x: -x[0])[:k]

    def lay_levels(self, k: int) -> list[tuple[float, float]]:
        return sorted(self.atl.items(), key=lambda x: x[0])[:k]


class MarketCache:
    """Order-book state for a single market, updated message by message."""

    def __init__(self, market_id: str):
        self.market_id = market_id
        self.runners: dict[int, RunnerBook] = {}
        self.status = "OPEN"
        self.in_play = False
        self.market_time_ms: float | None = None
        self.market_base_rate = 0.0
        self.total_matched = 0.0
        self.pt = 0  # publish time (ms) of the last applied message
        self.bet_delay = 0
        # (selection_id, price, volume_delta) trades since last drain_trades()
        self._trades: list[tuple[int, float, float]] = []
        # selection ids removed since last drain_removals(): (sid, adjustment_factor, pt)
        self._removals: list[tuple[int, float, int]] = []

    # ------------------------------------------------------------------ updates
    def _runner(self, sid: int) -> RunnerBook:
        r = self.runners.get(sid)
        if r is None:
            r = self.runners[sid] = RunnerBook(sid)
        return r

    def apply_mcm(self, msg: dict) -> None:
        pt = msg.get("pt", self.pt)
        for mc in msg.get("mc", ()):
            if mc.get("id") != self.market_id:
                continue
            self.pt = pt
            if mc.get("img"):
                # full image: replace ladders but keep cumulative traded info so
                # we don't mistake a reconnect for a burst of trading
                for r in self.runners.values():
                    r.atb.clear()
                    r.atl.clear()
            md = mc.get("marketDefinition")
            if md is not None:
                self._apply_definition(md, pt)
            if "tv" in mc:
                self.total_matched = mc["tv"]
            for rc in mc.get("rc", ()):
                self._apply_rc(rc, bool(mc.get("img")))

    def _apply_definition(self, md: dict, pt: int) -> None:
        self.status = md.get("status", self.status)
        self.in_play = bool(md.get("inPlay", self.in_play))
        self.market_base_rate = float(md.get("marketBaseRate", self.market_base_rate) or 0.0)
        self.bet_delay = int(md.get("betDelay", 0) or 0)
        mt = md.get("marketTime")
        if mt:
            self.market_time_ms = datetime.fromisoformat(mt.replace("Z", "+00:00")).timestamp() * 1000
        for rd in md.get("runners", ()):
            r = self._runner(rd["id"])
            new_status = rd.get("status", r.status)
            if new_status == "REMOVED" and r.status != "REMOVED":
                self._removals.append((r.selection_id, float(rd.get("adjustmentFactor") or 0.0), pt))
            r.status = new_status
            r.adjustment_factor = float(rd.get("adjustmentFactor") or 0.0)
            r.sort_priority = int(rd.get("sortPriority", r.sort_priority) or 0)

    def _apply_rc(self, rc: dict, img: bool) -> None:
        r = self._runner(rc["id"])
        for key in ("atb", "atl"):
            if key in rc:
                book = getattr(r, key)
                for price, size in rc[key]:
                    if size == 0:
                        book.pop(price, None)
                    else:
                        book[price] = size
        if "trd" in rc:
            for price, vol in rc["trd"]:
                prev = r.trd.get(price, 0.0)
                if vol > prev + 1e-9 and not img:
                    self._trades.append((r.selection_id, price, vol - prev))
                if vol == 0:
                    r.trd.pop(price, None)
                else:
                    r.trd[price] = vol
        if "ltp" in rc:
            r.ltp = rc["ltp"]
        if "tv" in rc:
            r.tv = rc["tv"]
        if "spn" in rc:
            r.spn = _num(rc["spn"])
        if "spf" in rc:
            r.spf = _num(rc["spf"])

    # ------------------------------------------------------------------ drains
    def drain_trades(self) -> list[tuple[int, float, float]]:
        t, self._trades = self._trades, []
        return t

    def drain_removals(self) -> list[tuple[int, float, int]]:
        t, self._removals = self._removals, []
        return t

    # ------------------------------------------------------------------ views
    def active_runners(self) -> list[RunnerBook]:
        rs = [r for r in self.runners.values() if r.status == "ACTIVE"]
        return sorted(rs, key=lambda r: r.sort_priority)


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v != v or v in (float("inf"), float("-inf")) else v


@dataclass
class Recording:
    meta: dict
    messages: list[dict]
    trailer: dict | None

    @property
    def market_id(self) -> str:
        return self.meta["market_id"]

    @property
    def market_start_ms(self) -> float:
        return datetime.fromisoformat(self.meta["market_start_utc"]).timestamp() * 1000


def read_recording(path: str) -> Recording:
    opener = gzip.open if str(path).endswith(".gz") else open
    meta, trailer, msgs = {}, None, []
    with opener(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            m = _loads(line)
            op = m.get("op")
            if op == "mcm":
                msgs.append(m)
            elif op == "meta":
                meta = m
            elif op == "trailer":
                trailer = m
    return Recording(meta, msgs, trailer)


def iter_replay(rec: Recording) -> Iterator[tuple[dict, MarketCache]]:
    cache = MarketCache(rec.market_id)
    for m in rec.messages:
        cache.apply_mcm(m)
        yield m, cache
