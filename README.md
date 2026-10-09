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
  market_study.py tradeability x signals, crossed, with train->holdout trading tests
  race_study.py  race categories (track, class, day, time...) x tradeability
  jump_study.py  event study: follow or fade price jumps and WAP displacement
  live/session.py          one live market: stream -> same env/policy code path
  live/bot.py              market discovery + streaming loop (paper by default)
  live/betfair_exchange.py real order routing via betfairlightweight (untested live)
notebooks/AHR_RL_train_colab.ipynb   full pipeline on Drive data
notebooks/race_deep_dive.ipynb       one random race: facts, runner summary, volume/price/relative-price charts
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

## V2 agent (current recommendation)

V2 is the environment and agent rebuilt around what the tests showed: 10s
decisions, trades held for minutes or to the start, engineered order-book,
longer price-history and catalogue form features, a cross-fitted price-move
forecaster whose predictions the agent sees, and a PPO exploration floor.
Results come with a test-set report broken down by segment and a deployment
gate.

See `docs/V2.md` for the evidence behind each choice, and notebook section V2
to run it.

```bash
python -m ahr_rl.catalogue data/tapes "<drive>/betfair stream data/catalogues"   # form features
python -m ahr_rl.forecaster --samples runs/edge/samples_base.parquet --tapes "data/tapes/*.npz" --out runs/fc.pkl
python -m ahr_rl.train --env v2 --forecaster runs/fc.pkl --out runs/ppo_v2 --explore-floor 0.05 --type-ent-coef 0.01
python -m ahr_rl.report --run runs/ppo_v2            # writes gate.json
```

## Research tools

| Module | Question it answers |
|---|---|
| `edge.py` | Do the features predict price moves well enough to beat round-trip costs? |
| `feature_study.py` | Which engineered features add signal? |
| `signal_bot.py` (`--diagnose`) | Does the edge survive realistic execution, and how much is fill modelling? |
| `continuous.py`, `compare.py` | Continuous allocation spec with PPO vs SAC |
| `market_study.py` | Which runners are tradeable (market share, rank, time to off), which signals (WoM, WAP, price and volume rate of change) carry information, how they cross, and can any of it be traded? |
| `race_study.py` | Do track, state, metro, distance, race type, class, day of week and time of day change how tradeable a race is? |
| `bsp_study.py` | Value betting: does anything (exchange prices, projected BSP, a form + market model) beat BSP on who wins? |
| `leadlag_study.py` | Does an outside price (bookmaker/tote CSV; projected BSP until one exists) lead the exchange, and can the gap be traded? |
| `sweep_passive.py` | After a sweep, does a resting order on the snap-back side make money in the full queue-aware simulator? |
| `sweep_bsp.py` | The post-sweep fade, tuned (take-profit, hold) with a BSP exit and no look-ahead on the off time |
| `p3b_forward.py` | The P3b rule frozen and scored only on races recorded after it was picked (no look-ahead top 3), with a pre-registered pass / fail rule |
| `lay_inplay.py` | Lay (or back) the n-th favourite pre-off and hedge with a resting back (or lay) that persists in-play: does it pay? |
| `pair_study.py` | A paired action spec (back pair / lay pair / nothing, each bet with its own resting green-up hedge): what each action is worth, the oracle headroom, and whether a model can pick the good pairs |
| `seq_study.py` | Frame stacking: a 1D CNN on raw stacked frames vs boosted trees on hand-made features, same decisions and holdout days as P6 |
| `blackbox.py` | Black box: SAC with a long random-strategy warm-up, a planted-edge positive control, multi-seed TEST evaluation with a latency stress |
| `strategy_search.py` | Black box, part 2: random search + mutation over whole rule-based strategies, positive control, scored once on TEST with a latency stress |
| `day_study.py` | Beyond green-before-the-off: in-play resting orders (low lays / drift backs) and what earlier races at a meeting reveal (barrier, jockey, trainer, favourites), bets to the result |
| `community.py` | The pre-off rules traders automate (WOM scalp, pressure scalp, gap fill, volume levels, spoofs, scratchings, arbitrage), tested honestly |
| `jump_study.py` | After a sudden price jump, or once a price leaves its session WAP, does it keep going or come back, and can you trade it? |

## Live trading

`MarketSession` feeds raw stream messages through the *same* `TapeBuilder`,
features, policy and env decision code used in training.
`tests/test_live_parity.py` replays recordings message by message and asserts
bit-identical observations, fills and P&L versus the offline env.

Paper mode is the default. Real money also requires the checkpoint's
`gate.json` (from `ahr_rl.report`) to show it beat "do nothing" on unseen
test races (mean > 0, t > 2); `--ignore-gate` overrides this and is not
advised. Real money needs `--live
--i-understand-this-bets-real-money`. **The real-money router
(`betfair_exchange.py`) has never been run against the live exchange.** Paper
trade first, then use minimal stakes.
