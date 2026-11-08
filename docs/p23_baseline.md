# P23 baseline (run of 2026-11-08 00:41, before the WOM / WAP / volume features)

Benchmark for the rerun with order-flow features. Source: Drive `rl_runs_ahr_rl/move_detect`
(`verdict.json`, `how_much_is_left.csv`, `real_rules_*.csv`). ~1,850 races; every 5s, tradeable runners
(spread <= 3 ticks, price <= 50); returns are % of stake, net of commission, race-clustered t.

Positive control (planted gradual steamer): found (`control_found_gradual_move: true`).

## Best rule per family (picked by t on TRAIN days, scored once on VAL / TEST)

| family | rules tested | positive with t > 2 on TRAIN | pre-registered rule | TRAIN | VAL | TEST (races) |
|---|---|---|---|---|---|---|
| gradual | 1,336 | 0 | W90 k3 p1.0 nojump top3, exit 120s, follow | -8.67% (t -4.5) | -3.96% (t -3.6) | -4.81% (t -3.6, 47) |
| knock-on | 240 | 0 | lead 4.0pp lag 1.0pp top3, exit 120s, follow | -8.05% (t -4.0) | -6.50% (t -2.1) | -7.21% (t -3.4, 17) |

## How much is left at detection (persistence >= 0.8; ticks within 120s; cost = 2 x spread)

- Median further move the same way: 1.5 - 3 ticks; median against: 1 - 3 ticks.
- Further move beats the crossing cost in only 32 - 47% of detections.
- More with than against: 47.7 - 53.5% (a coin flip). Best cell: W60 k4 no-jump, 53.5% (2,419 samples, 760 races).
- Longer look-backs leave less: W30 40-44% beat cost, W60 35-47%, W90 32-38%.

## What the rerun must beat

- The confirmed family or the model: TEST mean > 0 with t > 2 over >= 30 races, and VAL > 0.
- Short of an edge, a real improvement would show as:
  - a best TRAIN rule well above -8% (the gradual / knock-on best);
  - a TEST result above -4.8%;
  - "more with than against" clearly above 53.5%.
