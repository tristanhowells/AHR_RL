"""Run a trained agent on live Betfair AU horse-racing WIN markets.

    # paper trading (default): real live prices, simulated fills - no money at risk
    python -m ahr_rl.live.bot --model runs/ppo/best.pt

    # real money (only after extensive paper trading)
    python -m ahr_rl.live.bot --model runs/ppo/best.pt --live --i-understand-this-bets-real-money

Credentials come from env vars BF_USERNAME, BF_PASSWORD, BF_APP_KEY and
BF_CERTS (directory holding the client-2048.crt/.key pair).

Every market is joined ~10 minutes before its scheduled start with a fresh
bankroll (EnvConfig.exchange.bankroll, default $500), exactly as in training.
The auto-green safety net closes everything from the scheduled start.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone

import torch

from ..env import EnvConfig
from ..evaluate import make_torch_policy
from ..exchange import ExchangeConfig
from ..policy import RunnerTransformerPolicy
from .session import MarketSession

STREAM_FIELDS = ["EX_ALL_OFFERS", "EX_TRADED", "EX_TRADED_VOL", "EX_LTP", "EX_MARKET_DEF", "SP_TRADED",
                 "SP_PROJECTED"]  # identical to the recorder that produced the training data


def load_model(path: str, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["cfg"]
    cfg = EnvConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items() if k != "exchange"})
    cfg.exchange = ExchangeConfig(**c["exchange"])
    a = ck.get("args", {})
    model = RunnerTransformerPolicy(cfg, d_model=a.get("d_model", 96), n_layers=a.get("n_layers", 2),
                                    n_runner_features=a.get("n_runner_features"))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def main():
    import betfairlightweight
    from betfairlightweight import filters
    from betfairlightweight.streaming import BaseListener

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--i-understand-this-bets-real-money", action="store_true")
    ap.add_argument("--join-before-s", type=float, default=600)
    ap.add_argument("--countries", default="AU")
    ap.add_argument("--log", default="live_results.csv")
    args = ap.parse_args()
    live = args.live and args.i_understand_this_bets_real_money
    if args.live and not live:
        raise SystemExit("--live also needs --i-understand-this-bets-real-money")

    model, cfg = load_model(args.model)
    policy = make_torch_policy(model, deterministic=True)
    trading = betfairlightweight.APIClient(os.environ["BF_USERNAME"], os.environ["BF_PASSWORD"],
                                           app_key=os.environ["BF_APP_KEY"], certs=os.environ.get("BF_CERTS"))
    trading.login()
    print(f"logged in; mode = {'LIVE MONEY' if live else 'paper'}")

    q: queue.Queue = queue.Queue()

    class RawListener(BaseListener):
        def on_data(self, raw_data):
            msg = json.loads(raw_data)
            if msg.get("op") == "mcm":
                q.put(msg)
            return True

    stream = trading.streaming.create_stream(listener=RawListener())
    threading.Thread(target=stream.start, daemon=True).start()

    sessions: dict[str, MarketSession] = {}
    subscribed: set[str] = set()
    clock_offset_ms = 0.0
    last_scan = 0.0
    logf = open(args.log, "a", newline="")
    writer = csv.writer(logf)

    def exchange_factory(market_id):
        if not live:
            return None  # simulator
        from .betfair_exchange import BetfairExchange
        return lambda tape, xcfg: BetfairExchange(tape, xcfg, trading, market_id)

    while True:
        now = time.time()
        if now - last_scan > 30:
            last_scan = now
            t0 = datetime.now(timezone.utc)
            cats = trading.betting.list_market_catalogue(
                filter=filters.market_filter(event_type_ids=["7"], market_countries=args.countries.split(","),
                                             market_type_codes=["WIN"],
                                             market_start_time={"from": t0.isoformat(),
                                                                "to": (t0 + timedelta(minutes=12)).isoformat()}),
                market_projection=["MARKET_START_TIME"], max_results=50)
            for c in cats:
                mid = c.market_id
                secs = (c.market_start_time.replace(tzinfo=timezone.utc) - t0).total_seconds()
                if mid not in sessions and secs <= args.join_before_s + 15:
                    start_ms = c.market_start_time.replace(tzinfo=timezone.utc).timestamp() * 1000
                    sessions[mid] = MarketSession(mid, start_ms, policy, cfg, exchange_factory(mid))
                    print(f"joined {mid} ({secs:.0f}s to start)")
            want = {m for m, s in sessions.items() if not s.done}
            if want and want != subscribed:
                stream.subscribe_to_markets(
                    market_filter=filters.streaming_market_filter(market_ids=sorted(want)),
                    market_data_filter=filters.streaming_market_data_filter(fields=STREAM_FIELDS))
                subscribed = want
        try:
            msg = q.get(timeout=0.1)
            if "pt" in msg:
                clock_offset_ms = msg["pt"] - time.time() * 1000
            for mc in msg.get("mc", ()):
                s = sessions.get(mc.get("id"))
                if s is not None:
                    s.on_message({"op": "mcm", "pt": msg.get("pt"), "mc": [mc]})
        except queue.Empty:
            pass
        now_ms = time.time() * 1000 + clock_offset_ms
        for mid, s in list(sessions.items()):
            s.on_clock(now_ms - 250)  # small margin for in-flight messages
            if s.done:
                writer.writerow([datetime.now(timezone.utc).isoformat(), mid, "live" if live else "paper",
                                 s.result.get("worst"), s.result.get("expected"), s.ex.turnover if s.ex else 0])
                logf.flush()
                del sessions[mid]


if __name__ == "__main__":
    main()
