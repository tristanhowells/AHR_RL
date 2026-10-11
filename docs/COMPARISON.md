# Action spec and algorithm comparison

Same simulator, funds check, commission, auto-green safety net from the scheduled
start, and reward (change in mark-to-market green value) for every agent. All
agents use the same runner-equivariant transformer backbone. Results are on
held-out TEST races that no agent trained or selected checkpoints on.

**Specs**
* **Bracket (discrete):** per runner, NOOP / CLOSE / CANCEL / OPEN(back|lay,
  take|join, $5-50, take-profit 1/2/4 ticks). Exits and stops are automatic.
* **Continuous allocation:** 24 values in [-1,1] (sign = back/lay) plus a
  fraction f of available funds per 2s step, split by softmax(|a|/0.25) across
  runners above a 0.05 dead-zone. Aggressive orders only (unfilled remainder
  cancelled next step). A lay's share is its liability. No explicit exit: the
  agent greens up by sizing its own offsetting bets.

## Synthetic races (planted edge: one runner steadily shortens; 16 test races)

| Policy | Train steps | Mean green $/race | % green | Turnover $ |
|---|---|---|---|---|
| do nothing | - | 0.00 | 100 | 0 |
| random, bracket spec | - | -2.19 | 25 | 308 |
| random, continuous spec (f=0.02) | - | +16.73 | 100 | 281 |
| rule scalper, bracket spec | - | +0.06 | 62.5 | 78 |
| **PPO, bracket spec** | 205k | +0.52 | 62.5 | 135 |
| **PPO, continuous spec** | 300k | 0.00 (never bets) | 100 | 0 |
| **SAC, continuous spec** | 75k | **+222.60** | 100 | 2,477 |

Hand-coded oracles: bracket spec +$8.28/race, continuous spec +$18.74/race.

Dollar amounts are **not comparable across specs** on synthetic data. The
planted trend is very large (~100 ticks) and the synthetic books have
effectively unlimited depth, so the continuous spec's uncapped sizing
multiplies the edge (random allocation already makes +$16.73), while the
bracket spec caps each trade at $50. What the synthetic test does show is
whether each learner can *find* an edge.

## Real races (47 train / 7 val / 7 test races, split by date)

| Policy | Train steps | Mean green $/race | % green | Turnover $ | Worst race $ |
|---|---|---|---|---|---|
| do nothing | - | 0.00 | 100 | 0 | 0.00 |
| random, bracket spec | - | -6.03 | 0 | 163 | -15.87 |
| random, continuous spec (f=0.02) | - | -3.76 | 43 | 126 | -23.04 |
| rule scalper, bracket spec | - | -0.26 | 29 | 36 | -0.65 |
| **PPO, bracket spec** | ~220k | 0.00 (no trades) | 100 | 0 | 0.00 |
| **PPO, continuous spec** | 100k | 0.00 (no bets) | 100 | 0 | 0.00 |
| **SAC, continuous spec** | 100k | -0.05 | 86 | 2 | -0.32 |

With 47 training races, every learner converges to (nearly) not trading, which
is the correct response when no edge is visible. **This says nothing yet about
the full 1,171-race dataset.**

## Findings

1. **SAC vs PPO (continuous spec).** SAC is far more sample-efficient: its replay
   buffer found the synthetic edge within 25k steps, where continuous PPO never
   bet at all. But SAC was **unstable**: validation went +$85, then -$267
   (0% green, over-betting), then +$231 across 25k/50k/75k steps. It is also
   ~10x slower per env step on CPU (16-35 vs ~280 steps/s), because of a
   gradient step every 8 env steps through two critics. On real data it
   went from heavy losing (-$80/race train) to near-zero activity.
2. **Continuous spec vs bracket spec.**
   * The single shared wager fraction couples all runners: a bad bet on any
     horse pushes *f* down for all of them. That is why continuous PPO
     collapsed to f≈0.
   * There is no exit action, so greening needs precise self-sizing of
     offsetting bets. Random continuous trading at f=0.1 lost $85/race with a
     -$438 worst race, because positions accumulate across many runners faster
     than auto-green can unwind them in thin books. The bracket spec's worst
     random race is about -$24.
   * Its upside is expressiveness. It can size up aggressively when the signal
     is strong, which is where SAC's synthetic result comes from. On real,
     thin books that same property is the main risk.
3. **Recommendation.** Keep **PPO + bracket spec** as the primary agent (stable,
   bounded risk, fast to train). Run **SAC + continuous spec** on the full
   dataset as a challenger, with a per-step cap on *f* (e.g. <= 0.05) and on
   per-runner exposure, and judge both on held-out test races. Only deploy
   something that beats do-nothing there.

Reproduce: `python -m ahr_rl.continuous --algo sac|ppo ...` then
`python -m ahr_rl.compare --name real --discrete <run> --ppo-cont <run> --sac <run>`.
