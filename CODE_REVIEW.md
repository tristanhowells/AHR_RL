# V43 Green-Up Complete - Code Review

## Project Overview

This is a Reinforcement Learning system for automated horse racing market making on Betfair, using SAC (Soft Actor-Critic) with a 755-dimensional observation space. The agent trades pre-race, then "greens up" (flattens all positions) when the race goes in-play.

---

## CRITICAL BUGS

### 1. `_check_in_play_transition()` checks wrong column name (Cell 3)

```python
seconds_to_start = current_row.get('seconds_to_start', None)
if seconds_to_start is not None:
    return seconds_to_start <= 0
```

**Problem:** The parquet data contains `secs_to_off`, not `seconds_to_start`. This column is never found, so the code always falls through to the fallback (last 5% of data). This means:
- The agent doesn't green-up at the actual in-play transition point
- It uses an arbitrary 95th-percentile cutoff instead
- The `in_play` column (which exists and is accurate) is never checked

**Fix:** Use the actual `in_play` column:
```python
def _check_in_play_transition(self):
    if self.step_idx >= len(self.current_race_df):
        return False
    current_row = self.current_race_df.iloc[self.step_idx]
    return bool(current_row.get('in_play', 0))
```

### 2. Commission rate mismatch (Cell 2 config vs actual data)

```python
COMMISSION_RATE = 0.02  # 2% (training rate)
```

**Problem:** The actual data shows `commission_rate = 0.05` (5%). Training at 2% means the agent learns to make trades that are profitable at 2% commission but unprofitable at the real 5% rate. This will cause significant losses in production.

**Fix:** Use `COMMISSION_RATE = 0.05` or better, read it from the data:
```python
COMMISSION_RATE = float(self.current_race_df.iloc[0]['commission_rate'])
```

### 3. Green-up P&L formula is inverted for BACK positions (Cell 3)

```python
if net_stake > 0:
    # NET BACK position
    pnl = net_stake * (weighted_price - current_price) / current_price
```

**Problem:** For a BACK position, you profit when the price *drifts* (goes up, i.e., runner becomes less likely). The formula `(weighted_price - current_price)` gives positive P&L when the entry price is *higher* than the current price. But in betting:
- You BACK at `weighted_price` (entry)
- To green-up, you LAY at `current_price`
- Profit = `stake * (1/weighted_price - 1/current_price)` is the standard green-up formula, OR equivalently: `stake * (current_price - weighted_price) / (weighted_price * current_price) * weighted_price`

The current formula is non-standard and produces incorrect P&L magnitudes. For a BACK at 3.0 greened at 2.5 (price shortened = runner more likely = profitable back), the code gives `stake * (3.0 - 2.5) / 2.5 = +0.2 * stake`, but the correct green-up profit is `stake * (1/2.5 - 1/3.0) * stake_at_lay_price`, which depends on the lay stake.

**Fix:** Use the standard green-up formulas:
```python
if net_stake > 0:  # NET BACK position - green up by laying
    # Back at weighted_price, lay at current_price
    # Green-up profit: stake * (current_price - weighted_price) / current_price
    # Note: profit when current_price < weighted_price (price shortened)
    pnl = net_stake * (weighted_price - current_price) / weighted_price
```

### 4. LAY green-up P&L formula is also incorrect (Cell 3)

```python
else:
    # NET LAY position
    liability = abs(net_stake) * (weighted_price - 1)
    pnl = liability * (1.0 / weighted_price - 1.0 / current_price)
```

**Problem:** For a LAY position, the `net_stake` is negative (set by `pos['net_stake'] -= stake`), and `abs(net_stake)` gives the *stake* not the *liability*. But then it multiplies by `(weighted_price - 1)` to get liability - this double-counts. The actual liability was already recorded in `total_lay_liability`. Also, the P&L formula `liability * (1/wp - 1/cp)` doesn't correctly represent lay green-up economics.

**Fix:** For a LAY position greened up by backing:
```python
else:  # NET LAY position - green up by backing
    abs_stake = abs(net_stake)
    # Lay at weighted_price, back at current_price
    # Profit when current_price > weighted_price (price drifted)
    pnl = abs_stake * (current_price - weighted_price)
```

### 5. `_get_position_pnl()` inconsistent with `_calculate_green_up_pnl()` (Cell 3)

```python
def _get_position_pnl(self, runner_id, current_price):
    if net_stake > 0:
        pnl = net_stake * (current_price - entry_price)  # Simple difference
    else:
        pnl = abs_stake * (entry_price - current_price)  # Simple difference
```

**Problem:** This MTM P&L function uses a completely different formula (simple price difference) than `_calculate_green_up_pnl()` (which uses division-based formulas). The two should be consistent since MTM should approximate what you'd get by greening up. The simple difference formula is also incorrect for betting - a 1-tick move at price 2.0 has very different implied probability impact than a 1-tick move at price 50.0.

---

## SIGNIFICANT LOGIC ERRORS

### 6. Cell 5 redefines `get_runner_data()`, `to_float()`, `safe_normalize()`, and `safe_log_norm()` with DIFFERENT behavior

Cell 3 defines a 26-feature `get_runner_data()` that returns keys like `ret_std_5s`, `ret_std_20s`, `prob_implied`, `has_level_1`, etc. Cell 5 (which runs *after* Cell 3) redefines `get_runner_data()` with only 14 keys and different key names (`wom` instead of `ob_imbalance`, `spread` instead of `rel_spread`, adds `is_winner`, `ltv`).

**If Cell 5 runs after Cell 3**, it silently replaces the environment's feature extractor, and the environment (which expects 26-key dicts with keys like `has_level_1`, `ret_std_5s`, etc.) will crash with `KeyError` on every step.

Similarly, `safe_normalize()` is redefined in Cell 5 with different behavior:
- Cell 3: `safe_normalize(value, min_val, max_val, epsilon=1e-8)` returns 0.5 when range is too small
- Cell 5: `safe_normalize(val, min_val, max_val)` returns 0.0 when value is 0 (incorrect: 0 may be a valid normalized value)

And `safe_log_norm()`:
- Cell 3: `np.log1p(max(0, value) + epsilon)` - raw log value
- Cell 5: `np.log1p(val) / 10.0` - scaled by 1/10

**Impact:** Running cells in order will break the environment entirely. The notebook cell ordering is critical and fragile.

### 7. `weighted_price` tracking is broken for mixed BACK/LAY positions (Cell 3)

```python
if side == 'BACK':
    new_back = pos['total_back_stake'] + stake
    if new_back > 0:
        pos['weighted_price'] = (
            (pos['weighted_price'] * pos['total_back_stake'] + price * stake) / new_back
        )
    pos['total_back_stake'] = new_back
    pos['net_stake'] += stake
else:
    pos['total_lay_liability'] += liability
    pos['net_stake'] -= stake
    if pos['weighted_price'] == 0:
        pos['weighted_price'] = price
```

**Problem:**
- For LAY trades, `weighted_price` is only set on the *first* lay (when it's 0). Subsequent lay trades at different prices don't update the weighted average.
- When a runner has both BACK and LAY trades, `weighted_price` only reflects the back side. The `_calculate_green_up_pnl()` then uses this back-weighted price to calculate lay P&L, which is nonsensical.
- `total_back_stake` is only incremented for backs but `net_stake` is used for both sides, so `total_back_stake` and `net_stake` diverge.

### 8. The agent trades DURING in-play (not just pre-race) (Cell 3)

The `step()` method executes trades first, THEN checks for in-play transition:

```python
# Execute trades with V43 safety checks
for runner_idx in range(min(24, self.runner_count)):
    ...  # trades happen here

# AFTER trading:
reached_in_play = self._check_in_play_transition()
```

**Problem:** Since `_check_in_play_transition()` never correctly detects in-play (Bug #1), the agent trades through the entire race including in-play. Even with a fix, the current step's trades execute *before* the in-play check, so the agent makes one trade during in-play before the episode ends.

### 9. `MARKET_STATUS = 'SUSPENDED'` not checked (Cell 3)

The data shows `market_status` values of `['OPEN', 'SUSPENDED']`. Trading during a SUSPENDED market is not possible on a real exchange, but the environment doesn't check for this. The agent could learn to trade during suspended periods, which would never work in production.

---

## DATA HANDLING ISSUES

### 10. `secs_to_off` is always negative in the example data

All 290 rows have negative `secs_to_off` (range: -245 to -2.4). This means the entire dataset is captured *after* the scheduled start time. The environment doesn't account for this - it treats the data as starting pre-race and ending when `in_play` triggers.

### 11. Data validation in `load_race_files()` reads every parquet file into memory (Cell 3)

```python
for file in os.listdir(data_dir):
    if file.endswith('.parquet'):
        df = pd.read_parquet(filepath)  # Reads entire file
```

**Problem:** For thousands of race files, this loads each one fully into memory during validation. This is slow and memory-intensive. A better approach: read only metadata or the first row.

### 12. `race_files.remove()` during iteration in `reset()` (Cell 3)

```python
for attempt in range(max_attempts):
    try:
        race_file = random.choice(self.race_files)
        self.current_race_df = pd.read_parquet(race_file)
        break
    except Exception as e:
        if race_file in self.race_files:
            self.race_files.remove(race_file)
```

**Problem:** Modifying `self.race_files` (shared list) during training permanently removes files. Over time, the training set shrinks. If multiple environments share the same list, this causes race conditions.

---

## OBSERVATION SPACE ISSUES

### 13. Observation space claims 755 dims but the count depends on hardcoded 24 runners

The observation is `24 runners x 31 features + 11 global = 755`. But the data only has 9 runners. Runners 9-23 are always zero-padded. This means:
- 465 of 755 features (62%) are always zero
- The SAC policy network wastes capacity learning that these features are always zero
- With a `Box(-inf, inf)` space, the policy has no way to distinguish "no runner" from "runner with all-zero features"

### 14. `safe_normalize` uses fixed range [1.01, 1000.0] for prices

```python
back_1 = safe_normalize(runner_data['back_1'], 1.01, 1000.0)
```

**Problem:** Most racing prices are between 1.5 and 30.0. Normalizing to [1.01, 1000] means 97% of the normalized range is wasted on prices > 30. A price of 3.0 normalizes to ~0.002, and a price of 5.0 normalizes to ~0.004. The agent can barely distinguish between common price levels.

**Fix:** Use log-normalization for prices, or a tighter range like [1.01, 100.0].

---

## REWARD DESIGN ISSUES

### 15. Capital preservation bonus creates a constant positive reward signal

```python
capital_bonus = CAPITAL_PRESERVATION_BONUS if self.balance > (MAX_CAPITAL * 0.8) else 0.0
```

With `CAPITAL_PRESERVATION_BONUS = 0.1`, the agent gets +0.1 reward on almost every step (since the balance rarely drops below $800 early in training). Combined with `STEP_REWARD_NO_TRADE = -0.001`, this creates an overwhelming incentive to do nothing: the agent gets +0.099 per step by not trading. This likely explains the "no-trade" problem the diagnostic cells are trying to debug.

### 16. Sharpe reward calculation uses a short lookback window and raw P&L changes

```python
recent_pnl = list(self.pnl_history)[-20:]
mean_pnl = np.mean(recent_pnl)
std_pnl = np.std(recent_pnl)
sharpe = mean_pnl / std_pnl
```

**Problem:** The `pnl_history` stores `mtm_change` (change in unrealized P&L per step). With no positions, `mtm_change = 0` every step, so the deque fills with zeros. Then `std_pnl < 1e-8` triggers the early return. But as soon as the agent takes ONE position, it gets a large Sharpe spike (one nonzero value vs many zeros). This creates an unstable, noisy reward signal.

### 17. `NoTradeStreakWrapper` penalty grows exponentially and dominates rewards

```python
penalty = -1.0 * (2 ** exponent)  # exponent up to 5 => penalty up to -32
```

After 6 consecutive no-trade episodes, the penalty is -32. This is orders of magnitude larger than any other reward signal (MTM, Sharpe, activity rewards are all < 1.0). This extreme penalty can destabilize SAC training by creating huge gradient spikes.

---

## STRUCTURAL / CODE QUALITY ISSUES

### 18. Notebook contains dead/commented-out code and duplicate cells

- Cell 5 contains TWO versions of `get_runner_data()`: one fully commented out, one active. Both are different from Cell 3's version.
- Cells 6-15 are diagnostic/debug cells that should not be part of the production notebook. They test various aspects of the environment but are leftovers from debugging sessions.
- Cell 16 prints "V42" but this is supposed to be V43.

### 19. Signal handler for timeouts (`signal.signal`) won't work in Colab/threads

```python
import signal
def timeout_handler(signum, frame):
    raise FileLoadTimeout("File loading timed out")
```

`signal.signal` only works in the main thread. In Google Colab, notebook cells may run in a different thread context, causing `ValueError: signal only works in main thread`.

### 20. No `.gitignore` - parquet files shouldn't be in the repo

The 345KB parquet file is committed to git. As the project grows with more race data, the repo will become bloated. Binary data files should be in `.gitignore` and stored separately (e.g., Google Drive, as the notebook already expects).

### 21. `val_files` and `train_files` aren't shuffled before splitting

```python
val_files = train_files[-100:]
train_files = train_files[:-100]
```

Since `os.listdir()` returns files in arbitrary (filesystem-dependent) order, the validation set might be biased toward certain dates or venues. Should shuffle before splitting.

---

## SUMMARY OF PRIORITY

| Priority | Issue # | Description |
|----------|---------|-------------|
| P0 - Critical | 1 | In-play detection uses wrong column name |
| P0 - Critical | 2 | Commission rate 2% vs actual 5% |
| P0 - Critical | 3, 4 | Green-up P&L formulas are incorrect |
| P0 - Critical | 6 | Cell 5 redefines core functions with incompatible behavior |
| P1 - High | 5 | MTM P&L inconsistent with green-up P&L |
| P1 - High | 7 | weighted_price tracking broken for mixed positions |
| P1 - High | 8 | Agent trades during in-play |
| P1 - High | 9 | No SUSPENDED market check |
| P1 - High | 15 | Capital preservation bonus dominates, encourages inaction |
| P2 - Medium | 14 | Price normalization wastes 97% of range |
| P2 - Medium | 16 | Sharpe reward is noisy and unstable |
| P2 - Medium | 17 | No-trade penalty grows to -32, destabilizes training |
| P2 - Medium | 13 | 62% of observation space is always zero |
| P3 - Low | 10, 11, 12, 18-21 | Data handling, code quality, structural issues |
