"""One market, driven by stream messages instead of a pre-built tape.

The session reuses *exactly* the training code path:
    stream mcm -> TapeBuilder -> Tape (growing) -> features -> policy
               -> BetfairPreRaceEnv.decide (brackets, stops, auto-green) -> Exchange

``Exchange`` is either the simulator itself (paper trading against the live
book, identical to training) or ``BetfairExchange`` which sends real orders.
``tests/test_live_parity.py`` replays a recording message-by-message through a
session and checks it reproduces the offline environment step for step.
"""
from __future__ import annotations

from typing import Callable

from ..env import BetfairPreRaceEnv, EnvConfig
from ..exchange import Exchange
from ..tape import TapeBuilder


class MarketSession:
    def __init__(self, market_id: str, market_start_ms: float, policy_fn: Callable, env_cfg: EnvConfig,
                 exchange_factory: Callable[..., Exchange] | None = None, log: Callable = print):
        self.cfg = env_cfg
        self.builder = TapeBuilder(market_id, market_start_ms, dt=0.5)
        self.policy_fn = policy_fn
        self.exchange_factory = exchange_factory or (lambda tape, cfg: Exchange(tape, cfg))
        self.env = BetfairPreRaceEnv([], env_cfg)
        self.ex: Exchange | None = None
        self.result: dict | None = None
        self.log = log
        self.decisions = 0
        self.obs_trace: list = []  # (step, obs) for parity testing
        self.keep_trace = False

    @property
    def done(self) -> bool:
        return self.result is not None

    def on_message(self, msg: dict) -> None:
        if self.done:
            return
        self.builder.feed(msg)
        self._sync()

    def on_clock(self, now_ms: float) -> None:
        """Call periodically so quiet markets still advance on the grid."""
        if self.done:
            return
        self.builder.advance_to(now_ms)
        self._sync()

    def _sync(self) -> None:
        b = self.builder
        if not b.n_rows:
            return
        tape = b.tape(name=b.market_id)
        if self.ex is None:
            self.ex = self.exchange_factory(tape, self.cfg.exchange)
            self.env.attach(tape, self.ex)
            self._decide()
        self.ex.tape = tape
        self.env.attach(tape, self.ex)
        last = tape.n_steps - 1
        while self.ex.step < last:
            self.ex.advance()
            if b.finished and self.ex.step >= last:
                self.result = self.ex.settle()
                self.log(f"[{b.market_id}] off: {self.result}")
                return
            if self.ex.step % self.cfg.decision_every == 0:
                self._decide()

    def _decide(self) -> None:
        self.env._phi = self.env._potential()  # same bookkeeping as env.step
        obs = self.env.observe()
        if self.keep_trace:
            self.obs_trace.append((self.ex.step, obs))
        action = self.policy_fn(obs, self.env)
        self.env.decide(action)
        self.decisions += 1
