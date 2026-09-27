"""Recording -> fixed time-grid "tape" of the pre-race market, stored as .npz.

Replaying raw JSON every episode is slow, so each recording is compiled once into
dense numpy arrays sampled every ``dt`` seconds from the first message until the
market turns in-play (or closes / is abandoned). The tape keeps everything the
simulator needs for realistic fills:

* top-K ladder on each side (tick index + size) per runner per step,
* every trade between consecutive steps (runner, tick, single-counted volume),
* runner removals (late scratchings) with their reduction factor,
* settlement info (winner, BSP) for evaluation only - never shown to the agent.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from .ladder import N_TICKS, price_to_tick
from .stream import MarketCache, read_recording

# Betfair stream `trd` counts both sides of each match (verified on these
# recordings: traded-volume increments are exactly 2x the liquidity removed).
TRADED_VOLUME_DOUBLE_COUNTED = True


@dataclass
class Tape:
    market_id: str
    name: str
    dt: float
    t_rel: np.ndarray  # [T] seconds relative to scheduled start
    back_tick: np.ndarray  # [T,R,K] int16, best first (highest price), -1 = empty
    back_size: np.ndarray  # [T,R,K] float32
    lay_tick: np.ndarray  # [T,R,K] int16, best first (lowest price), -1 = empty
    lay_size: np.ndarray  # [T,R,K] float32
    ltp_tick: np.ndarray  # [T,R] int16, -1 = none
    tv: np.ndarray  # [T,R] float32 cumulative traded (single counted)
    spn: np.ndarray  # [T,R] float32 projected BSP, 0 = none
    active: np.ndarray  # [T,R] bool
    suspended: np.ndarray  # [T] bool
    total_matched: np.ndarray  # [T] float32
    trade_step: np.ndarray  # [M] int32: trade happened in (step-1, step]
    trade_runner: np.ndarray  # [M] int16
    trade_tick: np.ndarray  # [M] int16
    trade_vol: np.ndarray  # [M] float32 single-counted
    removal_step: np.ndarray  # [Q] int32
    removal_runner: np.ndarray  # [Q] int16
    removal_factor: np.ndarray  # [Q] float32 (percent)
    selection_ids: np.ndarray  # [R] int64
    base_rate: float  # market base commission rate (percent)
    went_in_play: bool  # False => abandoned/closed before the off: all bets void
    winner: int  # runner index of the winner, -1 unknown
    bsp: np.ndarray  # [R] float32, 0 unknown

    @property
    def n_steps(self) -> int:
        return len(self.t_rel)

    @property
    def n_runners(self) -> int:
        return len(self.selection_ids)

    # trade index: first row of trade_* for each step (built lazily)
    _trade_ptr: np.ndarray | None = None

    def trades_at(self, step: int):
        if self._trade_ptr is None:
            self._trade_ptr = np.searchsorted(self.trade_step, np.arange(self.n_steps + 1))
        a, b = self._trade_ptr[step], self._trade_ptr[step + 1]
        return self.trade_runner[a:b], self.trade_tick[a:b], self.trade_vol[a:b]

    def save(self, path: str) -> None:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        np.savez_compressed(path, **d)

    @staticmethod
    def load(path: str) -> "Tape":
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in z.files}
        for k in ("market_id", "name"):
            kw[k] = str(kw[k])
        for k in ("dt", "base_rate"):
            kw[k] = float(kw[k])
        kw["went_in_play"] = bool(kw["went_in_play"])
        kw["winner"] = int(kw["winner"])
        return Tape(**kw)


class TapeBuilder:
    """Incrementally turns stream messages into Tape rows on a fixed time grid.

    Used offline by :func:`build_tape` and live by the bot, so the agent sees
    identically constructed state in both. Row ``s`` is the market state as of
    grid time ``t0 + s*dt`` (all messages with ``pt <= t``); trades are tagged
    with the row whose interval ``(t_{s-1}, t_s]`` they fall in.
    """

    def __init__(self, market_id: str, market_start_ms: float, dt: float = 0.5, k_levels: int = 8,
                 pre_start_s: float = 600.0):
        self.market_id, self.start_ms, self.dt, self.K = market_id, market_start_ms, dt, k_levels
        self.pre_start_s = pre_start_s
        self.cache = MarketCache(market_id)
        self.sids: list[int] = []
        self.ridx: dict[int, int] = {}
        self.rows: dict[str, list] = {k: [] for k in ("t", "bt", "bs", "lt", "ls", "ltp", "tv", "spn", "act",
                                                       "susp", "tm")}
        self.trades: list[tuple[int, int, int, float]] = []
        self.removals: list[tuple[int, int, float]] = []
        self.vol_scale = 0.5 if TRADED_VOLUME_DOUBLE_COUNTED else 1.0
        self.initialised = False
        self.went_in_play = False
        self.finished = False
        self.t0 = None
        self.next_t = None
        self.step = 0

    @property
    def n_rows(self) -> int:
        return len(self.rows["t"])

    def _snapshot(self):
        R, K, cache, ridx = len(self.sids), self.K, self.cache, self.ridx
        bt = np.full((R, K), -1, np.int16)
        bs = np.zeros((R, K), np.float32)
        lt = np.full((R, K), -1, np.int16)
        ls = np.zeros((R, K), np.float32)
        ltp = np.full(R, -1, np.int16)
        tv = np.zeros(R, np.float32)
        spn = np.zeros(R, np.float32)
        act = np.zeros(R, bool)
        for sid, r in cache.runners.items():
            i = ridx.get(sid)
            if i is None:
                continue
            act[i] = r.status == "ACTIVE"
            for j, (p, sz) in enumerate(r.back_levels(K)):
                bt[i, j], bs[i, j] = price_to_tick(p), sz
            for j, (p, sz) in enumerate(r.lay_levels(K)):
                lt[i, j], ls[i, j] = price_to_tick(p), sz
            if r.ltp:
                ltp[i] = price_to_tick(r.ltp)
            tv[i] = r.tv * self.vol_scale
            spn[i] = r.spn
        rows = self.rows
        rows["bt"].append(bt); rows["bs"].append(bs); rows["lt"].append(lt); rows["ls"].append(ls)
        rows["ltp"].append(ltp); rows["tv"].append(tv); rows["spn"].append(spn); rows["act"].append(act)
        rows["susp"].append(cache.status != "OPEN")
        rows["tm"].append(cache.total_matched * self.vol_scale)
        rows["t"].append((self.next_t - self.start_ms) / 1000.0)

    def advance_to(self, now_ms: float) -> int:
        """Emit rows for every grid point strictly before now_ms. Returns rows emitted."""
        n = 0
        while self.initialised and not self.finished and now_ms > self.next_t:
            self._snapshot()
            self.step += 1
            self.next_t += self.dt * 1000
            n += 1
        return n

    def feed(self, msg: dict) -> int:
        """Apply one mcm message; returns number of grid rows emitted before it."""
        if self.finished:
            return 0
        pt = msg["pt"]
        if not self.sids:
            for mc in msg.get("mc", ()):
                md = mc.get("marketDefinition")
                if md and mc.get("id") == self.market_id:
                    self.sids = [r["id"] for r in sorted(md["runners"], key=lambda r: r.get("sortPriority", 0))]
                    self.ridx = {sid: i for i, sid in enumerate(self.sids)}
                    self.t0 = max(pt, self.start_ms - self.pre_start_s * 1000)
                    self.next_t = self.t0
        n = self.advance_to(pt)
        cache = self.cache
        cache.apply_mcm(msg)
        for sid, price, vol in cache.drain_trades():
            if sid in self.ridx and self.t0 is not None and pt >= self.t0:
                self.trades.append((self.step, self.ridx[sid], price_to_tick(price), vol * self.vol_scale))
        rem = cache.drain_removals()
        if self.initialised:
            for sid, af, _ in rem:
                if sid in self.ridx:
                    self.removals.append((self.step, self.ridx[sid], af))
        if not self.initialised and self.sids and pt >= self.t0 and cache.runners:
            self.initialised = True
            self.next_t = max(self.next_t, pt)
        if cache.in_play:
            self.went_in_play = True
            self.finish()
        elif self.initialised and (cache.status == "CLOSED" or not cache.active_runners()):
            self.finish()
        return n

    def finish(self):
        """Terminal row = state at the off / close."""
        if not self.finished and self.initialised:
            self._snapshot()
        self.finished = True

    def tape(self, name: str = "", trailer: dict | None = None) -> Tape:
        R, rows = len(self.sids), self.rows
        trailer = trailer or {}
        winners = trailer.get("winners") or []
        winner = self.ridx.get(winners[0], -1) if winners else -1
        bsp = np.zeros(R, np.float32)
        for sid, v in (trailer.get("bsp") or {}).items():
            if int(sid) in self.ridx and v:
                bsp[self.ridx[int(sid)]] = float(v)
        # only completed rows are visible (live: trades/removals for the row
        # currently being accumulated belong to the future)
        n = self.n_rows
        tr = np.array(self.trades, dtype=np.float64).reshape(-1, 4)
        tr = tr[tr[:, 0] < n]
        rm = np.array(self.removals, dtype=np.float64).reshape(-1, 3)
        rm = rm[rm[:, 0] < n]
        return Tape(
            market_id=self.market_id, name=name, dt=self.dt,
            t_rel=np.array(rows["t"], np.float32),
            back_tick=np.stack(rows["bt"]), back_size=np.stack(rows["bs"]),
            lay_tick=np.stack(rows["lt"]), lay_size=np.stack(rows["ls"]),
            ltp_tick=np.stack(rows["ltp"]), tv=np.stack(rows["tv"]), spn=np.stack(rows["spn"]),
            active=np.stack(rows["act"]), suspended=np.array(rows["susp"], bool),
            total_matched=np.array(rows["tm"], np.float32),
            trade_step=tr[:, 0].astype(np.int32), trade_runner=tr[:, 1].astype(np.int16),
            trade_tick=tr[:, 2].astype(np.int16), trade_vol=tr[:, 3].astype(np.float32),
            removal_step=rm[:, 0].astype(np.int32), removal_runner=rm[:, 1].astype(np.int16),
            removal_factor=rm[:, 2].astype(np.float32),
            selection_ids=np.array(self.sids, np.int64),
            base_rate=float(self.cache.market_base_rate or 0.0),
            went_in_play=self.went_in_play, winner=winner, bsp=bsp,
        )


def build_tape(path: str, dt: float = 0.5, k_levels: int = 8, pre_start_s: float = 600.0) -> Tape | None:
    rec = read_recording(path)
    if not rec.messages:
        return None
    b = TapeBuilder(rec.market_id, rec.market_start_ms, dt, k_levels, pre_start_s)
    for m in rec.messages:
        b.feed(m)
        if b.finished:
            break
    b.finish()
    if not b.sids or not b.n_rows:
        return None
    return b.tape(os.path.basename(path).split(".ndjson")[0], rec.trailer)


def build_all(src_dir: str, out_dir: str, dt: float = 0.5, k_levels: int = 8, workers: int = 4) -> list[str]:
    """Compile every *.ndjson.gz under src_dir into out_dir/<name>.npz (skips existing)."""
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "**", "*.ndjson.gz"), recursive=True))
    todo = [(f, os.path.join(out_dir, os.path.basename(f).split(".ndjson")[0] + ".npz")) for f in files]
    todo = [(f, o) for f, o in todo if not os.path.exists(o)]
    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as ex:
            list(ex.map(_build_one, todo, [dt] * len(todo), [k_levels] * len(todo), chunksize=4))
    else:
        for f, o in todo:
            _build_one((f, o), dt, k_levels)
    return sorted(glob.glob(os.path.join(out_dir, "*.npz")))


def _build_one(fo, dt, k_levels):
    f, o = fo
    try:
        tape = build_tape(f, dt=dt, k_levels=k_levels)
    except Exception as e:  # corrupt / truncated recordings shouldn't kill a batch
        print(f"[tape] skip {os.path.basename(f)}: {e!r}")
        return
    if tape is None or tape.n_steps < 10:
        print(f"[tape] skip {os.path.basename(f)}: empty")
        return
    tape.save(o)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Compile Betfair stream recordings into training tapes")
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    out = build_all(a.src, a.out, a.dt, a.levels, a.workers)
    print(f"{len(out)} tapes in {a.out}")
