# AHR_RL: pre-race greening agent for Betfair horse racing

An RL agent that trades Australian horse-racing WIN markets on Betfair in the
~10 minutes before the jump, aiming to be **green** (profit on every outcome)
when the market turns in-play. Each race starts with a fresh nominal **$500**.
The agent can back or lay any runner at any available price and stake. Every
order passes a Betfair-style funds check: worst-case loss across outcomes,
including unmatched orders, can never exceed the bankroll.

Everything here was written from scratch against the raw stream recordings in
`My Drive/betfair stream data/recordings`. The older notebooks and CSVs in the
repo root are untouched and unused.

```
ahr_rl/
  stream.py      Betfair Exchange Stream (mcm) -> order-book cache (shared with live)
  tape.py        recording -> fixed 0.5s grid "tape" (.npz); TapeBuilder shared with live
  ladder.py      Betfair price ladder (350 ticks)
  exchange.py    matching engine: latency, depth, queue position, funds, commission, scratchings
  features.py    per-runner + market features (causal; identical offline/live)
  env.py         Gymnasium env: bracket-trade actions, auto-green safety net, reward
  policy.py      runner-equivariant transformer actor-critic, factorised action head
  train.py       PPO with per-runner credit assignment, chronological train/val/test split
  evaluate.py    evaluation + baselines (do-nothing, random, rule-based scalper)
  report.py      held-out test report
  synthetic.py   races with a planted edge (sanity check for the learner)
  live/session.py          one live market: stream -> same env/policy code path
  live/bot.py              market discovery + streaming loop (paper by default)
  live/betfair_exchange.py real order routing via betfairlightweight (untested live)
notebooks/AHR_RL_train_colab.ipynb   full pipeline on Drive data
tests/                               simulator maths + live/offline parity
```

## What the data looks like (61-race sample, profiled)

* Recordings start **~600s before the scheduled off** and run to settlement. They
  include full-depth `atb`/`atl` ladders (deltas), `trd`/`ltp`/`tv`, BSP
  projections, market definitions, plus a trailer with the winner and BSP.
* **The off is late:** in-play arrives at a median of +68s after the advertised
  time, up to +363s. One race went early (−60s) and one was abandoned. The
  episode ends on the real `inPlay` flag, never on the clock.
* **`trd` volume is double-counted.** Traded-volume increments are exactly 2x the
  liquidity removed (85k trades checked), so the simulator halves it.
* **Liquidity is thin:** the median best-price size is ~$5 (harness tracks $1–3,
  metro thoroughbreds $20–150), with a 2-tick median spread. A $500 bank can
  easily outsize the book, which is why the fill model matters.
* Late scratchings happen mid-window, and commission (`marketBaseRate`) is 8–10%.
* 7 of the 61 recordings stop before settlement (winner unknown). Training
  doesn't need the winner; see the metrics below.

## Simulator (exchange.py): conservative by design

| Aspect | Model |
|---|---|
| Latency | orders and cancels land one grid step (0.5s) after the decision (measured stream latency ~130ms) |
| Aggressive fills | walk the visible ladder level by level, at each level's price; liquidity we take is remembered so it can't be taken twice |
| Passive fills | queue behind the size already at our price; traded volume burns the queue first; trades through our price, or the book crossing it, fill us |
| Funds | Betfair exposure = worst case over outcomes of matched plus the worse of each unmatched order; orders are clipped or rejected at the $500 limit |
| Min stake | $5 for opening orders; hedges may go below it (standard cancel-down/replace workaround, implemented in the live router) |
| Settlement | unmatched orders lapse at the off; commission on net market profit; non-runners void their bets, cancel unmatched and apply reduction factors |

## Agent

**Actions** (per runner, every 2s): `NOOP`, `CLOSE` (flatten at market),
`CANCEL_ENTRY`, or `OPEN(side, entry = take | join queue, stake ∈ {5,10,25,50},
take-profit ∈ {1,2,4} ticks)`. `OPEN` is a bracket trade. Once the entry
matches, an exit order sized to green that runner rests `tp` ticks better, and
a stop closes it if the price moves `2·tp` ticks against. Every leg is an
ordinary back or lay.

**Safety net:** from the scheduled start the env stops opening trades and
flattens everything each decision until the jump. The live bot runs the same
code.

**Reward:** potential-based on mark-to-market green value, blended towards the
un-hedged worst case over the final 2 minutes. The episode return equals the
guaranteed (worst-case, after commission) profit, which is exactly the
objective. Training uses the per-runner decomposition of this value, so each
runner's decisions are credited only with that runner's P&L.

**Network:** each runner is a token. A 2-layer transformer lets runners attend to
each other (money moving between horses), with factorised action heads and a
per-runner critic. It handles any field size up to 24.

## Metrics

* `mean_green_$`: average guaranteed profit per race (worst case over outcomes,
  after commission). This is the target. **Do-nothing scores exactly 0.**
* `pct_green`: share of races finishing green.
* `mean_expected_$`: P&L valued at the market's own off prices, a luck-free check.
* `mean_realised_$`: P&L given the actual winner (only where the recording has it).

## Usage

```bash
pip install -r requirements.txt
python -m ahr_rl.tape "<drive>/betfair stream data/recordings" data/tapes     # compile
python -m pytest -q                                                            # sim + parity tests
python -m ahr_rl.train --tapes "data/tapes/*.npz" --out runs/ppo --total-steps 5000000
python -m ahr_rl.report --run runs/ppo --split test
python -m ahr_rl.live.bot --model runs/ppo/best.pt                             # paper trading
```

Or open `notebooks/AHR_RL_train_colab.ipynb` in Colab to run everything on the
full Drive dataset.

## Live trading

`MarketSession` feeds raw stream messages through the *same* `TapeBuilder`,
features, policy and env decision code used in training.
`tests/test_live_parity.py` replays recordings message by message and asserts
bit-identical observations, fills and P&L versus the offline env.

Paper mode is the default. Real money needs `--live
--i-understand-this-bets-real-money`. **The real-money router
(`betfair_exchange.py`) has never been run against the live exchange.** Paper
trade first, then use minimal stakes.
