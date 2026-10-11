# Single-Runner SAC Trader (stream tapes)

The agent **sees the whole race but trades exactly one runner per episode**, and
learns from every runner of every race as separate episodes. It runs on the
Betfair **stream tapes** used by the `ahr_rl` research, compiled with
`python -m ahr_rl.tape`, and its fills come from the research simulator,
`ahr_rl/exchange.py`.

```
single_runner/
  features.py      tape -> causal per-runner + global feature arrays
  exchange_x.py    ahr_rl Exchange + cross-matching + single-runner green-up
  env.py           SingleRunnerTradingEnv (gymnasium)
  sac.py           black-box SAC (PyTorch)
  train.py         train / validate / score once on TEST
  parquet_tape.py  old scraped parquet -> Tape (lower fidelity; tests / pretraining only)
tests/test_single_runner.py
```

## Quick start

```bash
pip install -r requirements.txt
python -m ahr_rl.tape "<drive>/betfair stream data/recordings" /content/tapes          # once
python -m ahr_rl.catalogue /content/tapes "<drive>/betfair stream data/catalogues"    # optional form features
python -m single_runner.train --tapes "/content/tapes/*.npz" --out-dir runs/sr1 --total-steps 1000000
python -m single_runner.train --tapes "/content/tapes/*.npz" --out-dir runs/sr1 --eval-test runs/sr1/best.pt
```

Useful flags:

| Flag | Effect |
|---|---|
| `--ranks 1,2,3` | train only on these favouritism ranks |
| `--race-type flat` | race filter shared with the research studies |
| `--metro-only` | race filter shared with the research studies |
| `--decision-every 20` | decide every 10 s instead of every 2 s |
| `--fill-mode no_queue` | optimistic fill-sensitivity check |
| `--greenup cross` | green up by walking the book (pays the spread) |

## What changed from the parquet version

| | Scraped parquet (old) | Stream tapes (now) |
|---|---|---|
| Grid | ~1.33 s irregular snapshots | 0.5 s, built from every stream message |
| Book | top 3 levels | 8 levels |
| Passive fills | guessed from last traded price and total volume | the actual recorded trades: price tick and single-counted volume |
| Latency | none | orders and cancels land one step (0.5 s) later |
| Liquidity we take | forgotten at the next snapshot | remembered until the level changes or 10 s pass |
| Minimum stake | $1 | $5 to open; hedges may go lower |
| Commission | 5% (as recorded in those files) | market base rate from the tape (8–10% observed) |
| Scratchings | none | runner voided, unmatched orders cancelled, reduction factors applied |
| Abandoned races | n/a | excluded from training and evaluation |
| Extras | none | projected BSP, catalogue form features |

## Episodes

* **Episode = (tape, target rank).** The target is the runner holding that
  favouritism rank (by fair price) at the first step. It stays the target even
  if the market reorders.
* Training samples a random race, then a random eligible rank (or one from
  `--ranks`). Evaluation covers every (race, rank) pair.
* Episodes start `start_s` (600) seconds before the scheduled off and end when
  the market turns in-play. `--random-start-s` delays the start at random
  during training.
* The agent decides every `decision_every` tape steps, 2 s by default.
* If the target is scratched mid-race, its bets are void and the episode ends
  with a P&L of 0.

## Observation (1,709 floats)

| Block | Size | Contents |
|---|---|---|
| Global | 13 | time to scheduled off, total matched and its 30 s change, back/lay overround, entropy, favourite's probability gap, mean spread, field size, commission, suspended flag, has-form flag |
| Target runner | 65 + 4 | its runner features, plus start rank, rank change, log price move since start, elapsed time |
| All runners | 24 × 67 | every runner ordered by start favouritism, zero-padded, plus `is_target` and rank change |
| Agent state | 19 | win/lose payoff, exposure, available funds, matched back/lay stake and average price, MTM equity, unmatched stake/price/queue per side, open orders, last fills |

The 65 per-runner features are:

* **Raw book:** 3 levels of back/lay price and size, plus total 8-level depth
  on each side.
* **Price:** last traded price, fair price and normalised fair probability,
  spread in ticks, projected BSP relative to fair price.
* **Weight of money (WOM):** level 1, 3 levels and all 8 levels.
* **Weighted average price (WAP):** size-weighted book price; session and
  60 s VWAP from the exact trade prices.
* **Max/min matched price:** session and 60 s window, from the actual trades.
* **Volume:** total, 60 s and 5 s volume, trade-flow imbalance (backers
  taking vs layers taking over 30 s), volume share, time since last trade.
* **Returns:** over 5 s, 15 s, 30 s, 60 s, 2 min and 5 min.
* **Rank:** current favouritism rank and an `is_fav` flag.
* **Form:** 17 catalogue form features, when attached with `ahr_rl.catalogue`.

Everything is causal. A test rebuilds the features from a tape truncated at
step *s* and checks they match the full build. The winner and BSP are never
used.

## Actions (6, in [-1, 1])

| idx | meaning |
|---|---|
| 0 | back size: `max(a, 0)` × max affordable stake (`<= 0` places no bet) |
| 1 | back price: `round(a × tick_range)` ticks from the best atb price (+ is more passive) |
| 2 | lay size: `max(a, 0)` × max affordable stake |
| 3 | lay price: `round(a × tick_range)` ticks from the best atl price (+ is more passive) |
| 4 / 5 | `> 0` pulls all unmatched backs / lays on the target |

The agent can back and lay in the same decision, at any size up to the $100
funds limit and any ladder price within ±`tick_range` (10) ticks of the touch.

## Matching

Fills come from `ahr_rl.exchange`, unchanged:

* Orders and cancels arrive 0.5 s late.
* Aggressive orders walk the visible ladder at each level's own price, and
  liquidity we take is remembered.
* Resting orders queue behind the size already at their price. Only recorded
  trades at that price, trades through it, or the book crossing it fill them.
* Betfair's funds check, the minimum stake, scratchings and base-rate
  commission all apply.

### Cross-matching

The recorded ladders exclude virtual bets, so `exchange_x.py` adds them.
Backing the target can use the other runners' best atl offers combined, at a
price `v` where `1/v = 1 − Σ 1/p_j`, rounded to the worse tick. The size is
limited by the thinnest of those offers. Laying works the same way from their
atb offers.

A virtual price is **ignored if it improves on the displayed best by more
than `max_virtual_ticks` (3)**. On the live exchange such offers would
already have been cross-matched, so a gap that large means the snapshot is
stale or incoherent. Without this guard, random trading on the synthetic
tapes made +$376 per race purely from farming those virtual prices. With it,
random trading loses on every runner that has no planted edge.

## Non-biased green-up at the off

At the last tradeable pre-off snapshot, the target position is hedged so that
P&L is the same whether it wins or loses:

```
G(c) = L + (W − L) / c        W / L = payoff if the target wins / loses
```

The fair price `c = 1/q` is the **probability-space microprice** from level 1.
With it, `G = qW + (1 − q)L`: the market's own expectation, which neither
charges nor gifts the spread and never looks at the winner. Commission is the
market base rate on positive net profit. `--greenup cross` walks the real book
instead (pessimistic) and is also reported by `--eval-test`.

## Reward and metrics

* **Reward:** the change in greened, post-commission equity / balance ×
  `reward_scale`. It telescopes to the final P&L, and doing nothing scores
  exactly 0.
* **`val_metrics.csv`:** mean and median P&L, `t_race` (t-stat with races as
  the unit, since a race's ranks aren't independent), win rate, trade rate,
  matched volume, passive fill share, and mean P&L for each rank.
* **`--eval-test`:** scores a checkpoint once on TEST, under both the fair and
  cross green-up rules.
* **Bar to beat:** use the research gate, mean > 0 with t > 2 on TEST.
