# Single-Runner SAC Trader

A new approach that replaces the multi-runner market maker (V43–V51). The agent
**sees the whole race but trades exactly one runner per episode**, and learns
from every runner of every race as separate episodes.

```
single_runner/
  ladder.py     Betfair tick ladder (1.01..1000, 350 ticks)
  features.py   parquet -> causal per-runner + global feature arrays
  matching.py   order matching, cross-matching, passive fills, green-up
  env.py        SingleRunnerTradingEnv (gymnasium)
  sac.py        black-box SAC (PyTorch, no SB3 dependency)
  train.py      training / evaluation CLI
tests/test_single_runner.py
```

## Quick start

```bash
pip install torch gymnasium pandas pyarrow numpy
python -m single_runner.train --data-dir /path/to/race_out --out-dir runs/sr1 \
    --total-steps 1000000 --eval-every 50000            # all runners
python -m single_runner.train ... --ranks 1,2,3          # only 1st-3rd favourites
python -m pytest -q
```

`SingleRunnerTradingEnv` is a normal `gymnasium.Env`, so SB3's `SAC("MlpPolicy", env)` also works.

## Episodes

* **Episode = (race, target rank).** The target is the runner holding that
  favouritism rank (by fair price) at the episode's first step. It stays the
  target for the whole episode even if the market reorders.
* Training samples a random race, then a random eligible rank (or from
  `--ranks`). `episode_specs()` lists every (race, rank) pair for exhaustive
  evaluation.
* Steps are the pre-race `OPEN` snapshots (~1.33 s apart, ~550 per race).
  `--max-episode-steps N` keeps only the last N snapshots before the off, and
  `--random-start` adds start-time augmentation.

## Observation (1,214 floats)

| block | size | contents |
|---|---|---|
| global | 18 | scheduled secs-to-off, market volume and its change, back/lay overround, entropy, fav gap, spread stats, field size, commission, race code, distance, dt |
| target runner | 45 + 4 | its runner features + start rank, rank change, log price move since start, elapsed steps |
| all runners | 24 x 47 | every runner (ordered by start favouritism, zero-padded) + `is_target` + rank change |
| agent state | 19 | win/lose payoffs, exposure, available, matched back/lay stake and avg price, MTM equity, unmatched stake/price/queue per side, open orders, last fills |

The 45 per-runner features are the raw book (3 levels of back/lay price and
size, LTP, traded volume, microprice, imbalance, volatility, implied prob)
plus engineered ones:

* **WOM**: level 1 and 3-level weight of money.
* **WAP**: size-weighted book price, running traded VWAP and a ~60 s VWAP.
* **Max/min matched price**: running and ~60 s window.
* **Rank**: current favouritism rank and an `is_fav` flag.
* **Other**: spread in ticks, returns over 1/5/20/60 steps, volume share,
  normalised fair probability.

All features are causal. A test rebuilds a race from a truncated dataframe and
checks they match. Result columns (`is_winner`, `result_*`) are never used.

## Actions (6, in [-1, 1])

| idx | meaning |
|---|---|
| 0 | back size: `max(a,0)` x max affordable back stake (`<=0` places no bet) |
| 1 | back price: `round(a x tick_range)` ticks from best available-to-back (+ is more passive) |
| 2 | lay size: `max(a,0)` x max affordable lay stake |
| 3 | lay price: `round(a x tick_range)` ticks from best available-to-lay (+ is more passive) |
| 4 / 5 | `> 0` cancels all unmatched backs / lays before placing |

The agent can back and lay in the same step, at any size up to the bank and at
any ladder price within ±`tick_range` ticks (default 10) of the touch.
"Max affordable" is the stake that keeps the Betfair-style worst-case
exposure, with unmatched orders included, within the $100 balance.

## Matching simulation (realistic but optimistic)

1. **Aggressive:** walks the displayed 3 levels at each level's own price
   (price improvement), limited by displayed size. Liquidity used at a
   snapshot can't be reused in that same snapshot.
2. **Cross-matching:** builds virtual offers from the other runners' level-1
   books (`1/v = 1 - sum 1/p_j`, size limited by the thinnest leg). They are
   used when they beat the displayed best price.
3. **Passive orders** rest at the limit price with a queue position taken
   from displayed size. Each new snapshot can fill them in three ways:
   * **Crossed book:** the opposite side now offers our price or better.
   * **Trade-through:** volume traded and the LTP went strictly through our
     price, so those takers would have hit us first.
   * **Trade-at:** the LTP equals our price; the traded volume eats the
     queue ahead of us first.

   Separately, size cancelled at our price is assumed to have been ahead of
   us in the queue.
4. **Optimistic assumptions:** zero latency, no market impact on the recorded
   book, all traded volume counted toward the side that could fill us, and
   cancellations taken from ahead of us. Set `fill_optimism < 1` to dial
   this down.
5. Unmatched orders **lapse at in-play**.

Fills come with natural adverse selection: a resting back fills when the price
drifts through it. Sanity check on the sample race: naive passive quoting on
both sides loses money, so the simulator doesn't hand out free spread.

## Non-biased green-up at the off

At the last pre-race snapshot the position is hedged so that the P&L is the
same whether the target wins or loses:

```
W = payoff if target wins, L = payoff if it loses
G(c) = L + (W - L) / c          (hedge stake |W-L|/c)
```

With `c = fair price` (microprice, the default `greenup_mode='fair'`),
`G = W/c + L(1 - 1/c)`, the expected P&L under the market-implied
probability. It doesn't charge or gift the spread and never looks at who won.
Commission (read from the data, 5%) applies to positive net profit.
`--greenup cross` walks the real book instead, which is pessimistic because
it pays the spread. Use it as a stress test.

## Reward

`reward_t = (equity_t - equity_{t-1}) / balance x reward_scale`. Equity is
the greened, post-commission value of the matched position, marked at the
same price rule. The episode return therefore equals
`final net P&L / balance x reward_scale` exactly, and doing nothing scores 0.

## Evaluation metrics (`val_metrics.csv`)

The deterministic policy runs on every (validation race, rank) pair. Reported:
mean, median and total P&L, win rate, trade rate, matched volume, passive fill
share, per-episode Sharpe, and mean P&L for each favouritism rank. The split is
chronological on the date in the filename (`--val-frac` or `--val-start`).
`info['ungreened_pnl_actual']` is a diagnostic only: what the book would have
paid on the real result without greening.
