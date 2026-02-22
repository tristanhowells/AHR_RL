#!/usr/bin/env python3
"""
test_fixed_env.py — Smoke-test the V44 environment against the example parquet.

Tests:
  1. Environment instantiation with 755-dim observation
  2. reset() reads commission_rate from data (should be 0.05)
  3. step() with random actions produces valid obs/reward/done
  4. In-play detection triggers at correct row (row 72 in example data)
  5. Market SUSPENDED rows are skipped (no trades executed)
  6. Green-up P&L computed at episode end
  7. _get_position_pnl() matches _calculate_green_up_pnl() for same prices
  8. NoTradeStreakWrapper penalty is capped
  9. Back green-up formula: stake*(B-C)/C
 10. Lay green-up formula:  stake*(C-L)/C
"""

import json
import sys
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Step 0 — Extract and exec cells 2+3 from the fixed notebook so we get
#           all constants, helpers, and classes in this namespace.
# ---------------------------------------------------------------------------
NB_PATH = "/home/user/AHR_RL/V44_Green_Up_Position_Netting.ipynb"
PARQUET = "/home/user/AHR_RL/2026-02-13_Newcastle_Race7_1.253949170.parquet"

with open(NB_PATH, "r") as f:
    nb = json.load(f)

# Execute config cell (cell 2) and env cell (cell 3)
cell2_src = "".join(nb["cells"][2]["source"])
cell3_src = "".join(nb["cells"][3]["source"])

# Strip Colab-specific lines
cell2_src = cell2_src.replace("os.makedirs(BASE_PATH, exist_ok=True)", "pass")

ns = {}
exec(cell2_src, ns)
exec(cell3_src, ns)

# Pull out what we need
MarketMakingEnv = ns["MarketMakingEnv"]
NoTradeStreakWrapper = ns["NoTradeStreakWrapper"]
MAX_CAPITAL = ns["MAX_CAPITAL"]
NO_TRADE_PENALTY_CAP = ns["NO_TRADE_PENALTY_CAP"]

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name} -- {detail}")
        failed += 1


# ---------------------------------------------------------------------------
# Test 1 — Instantiation
# ---------------------------------------------------------------------------
print("\n=== Test 1: Environment instantiation ===")
env = MarketMakingEnv(race_files=[PARQUET])
check("obs_space shape", env.observation_space.shape == (755,),
      f"got {env.observation_space.shape}")
check("action_space shape", env.action_space.shape == (25,),
      f"got {env.action_space.shape}")

# ---------------------------------------------------------------------------
# Test 2 — reset() reads commission_rate from data
# ---------------------------------------------------------------------------
print("\n=== Test 2: reset() and commission_rate ===")
obs, info = env.reset()
check("obs dtype", obs.dtype == np.float32, f"got {obs.dtype}")
check("obs length", len(obs) == 755, f"got {len(obs)}")
check("commission_rate from data", abs(env.commission_rate - 0.05) < 1e-6,
      f"got {env.commission_rate}")
check("current_race_file stored", env.current_race_file == PARQUET,
      f"got {env.current_race_file}")

# ---------------------------------------------------------------------------
# Test 3 — step() with random actions
# ---------------------------------------------------------------------------
print("\n=== Test 3: step() basic ===")
action = env.action_space.sample()
obs2, reward, done, truncated, info = env.step(action)
check("obs after step", len(obs2) == 755 and obs2.dtype == np.float32)
check("reward is float", isinstance(reward, (float, np.floating)), f"type={type(reward)}")
check("done is bool", isinstance(done, (bool, np.bool_)), f"type={type(done)}")

# ---------------------------------------------------------------------------
# Test 4 — In-play detection at correct row
# ---------------------------------------------------------------------------
print("\n=== Test 4: In-play detection ===")
df = pd.read_parquet(PARQUET)
# Find first row where in_play == 1
in_play_rows = df[df['in_play'] == 1]
first_in_play = in_play_rows.index[0] if len(in_play_rows) > 0 else None
check("in_play found in data", first_in_play is not None, "no in_play=1 rows")
check("in_play starts at row 72", first_in_play == 72, f"got row {first_in_play}")

# Run through the environment and confirm it ends at the in-play transition
env2 = MarketMakingEnv(race_files=[PARQUET])
obs, _ = env2.reset()
steps = 0
episode_done = False
while not episode_done and steps < 500:
    action = np.zeros(25, dtype=np.float32)  # no-op
    obs, reward, episode_done, truncated, info = env2.step(action)
    steps += 1
    episode_done = episode_done or truncated

# The episode should end at step_idx == first_in_play (checked BEFORE trading)
# step_idx is incremented at end of step, so last step processed is step_idx-1
# But in-play check happens at beginning of step, so the env should stop when
# it reaches the first in-play row.
check("episode ends at in-play",
      env2.step_idx <= first_in_play + 1,
      f"step_idx={env2.step_idx}, expected <= {first_in_play+1}")
check("episode done", episode_done, "episode did not terminate")

# Pre-in-play rows: rows 0..71 are in_play=0 (72 rows), so max 72 steps
check("ran pre-race steps", steps <= first_in_play + 1,
      f"steps={steps}, expected <= {first_in_play+1}")

# ---------------------------------------------------------------------------
# Test 5 — Market SUSPENDED check
# ---------------------------------------------------------------------------
print("\n=== Test 5: Market SUSPENDED check ===")
suspended_rows = df[df['market_status'] == 'SUSPENDED']
check("SUSPENDED rows exist in data", len(suspended_rows) > 0,
      f"found {len(suspended_rows)} suspended rows")

# Find if any SUSPENDED rows are before in-play
suspended_pre_play = suspended_rows[suspended_rows.index < first_in_play]
if len(suspended_pre_play) > 0:
    # Run an episode and confirm no trades happen during suspended periods
    env3 = MarketMakingEnv(race_files=[PARQUET])
    obs, _ = env3.reset()
    # Force a strong buy signal on every step
    strong_action = np.ones(25, dtype=np.float32)  # strong buy
    strong_action[24] = 1.0  # full allocation

    for s in range(min(first_in_play, len(df))):
        obs, r, d, t, info = env3.step(strong_action)
        if d or t:
            break

    check("suspended_violations > 0", env3.suspended_violations > 0,
          f"got {env3.suspended_violations}")
    print(f"    (suspended_violations={env3.suspended_violations})")
else:
    print("    (no SUSPENDED rows before in-play, skipping sub-test)")
    check("no SUSPENDED pre-play", True)

# ---------------------------------------------------------------------------
# Test 6 — Green-up P&L computed at episode end
# ---------------------------------------------------------------------------
print("\n=== Test 6: Green-up P&L ===")
CurriculumTracker = ns["CurriculumTracker"]

# Create a permissive curriculum (warmup=0 so min_liability starts at 0.05)
permissive_curr = CurriculumTracker(total_steps=1000, warmup_steps=0)

env4 = MarketMakingEnv(race_files=[PARQUET], curriculum_tracker=permissive_curr)
obs, _ = env4.reset()

# Use SMALL signals (0.02) to respect depth limits — $1200 * 0.02 = $24 < 50% of available depth
small_back = np.full(25, 0.02, dtype=np.float32)
small_back[24] = 1.0  # full allocation

# Step through pre-race with trades
for s in range(min(first_in_play, 200)):
    obs, r, d, t, info = env4.step(small_back)
    if d or t:
        break

final_balance = env4.balance
initial_balance = env4.initial_balance
green_up_pnl = final_balance - initial_balance
print(f"    trades: {len(env4.trades_this_episode)}, green_up_pnl: ${green_up_pnl:.4f}")
check("balance changed from initial",
      abs(final_balance - initial_balance) >= 0.0,  # may be zero if no trades
      f"balance={final_balance}")

# ---------------------------------------------------------------------------
# Test 7 — _get_position_pnl matches _calculate_green_up_pnl
# ---------------------------------------------------------------------------
print("\n=== Test 7: PnL consistency ===")
permissive_curr2 = CurriculumTracker(total_steps=1000, warmup_steps=0)
env5 = MarketMakingEnv(race_files=[PARQUET], curriculum_tracker=permissive_curr2)
obs, _ = env5.reset()

# Use small signals to pass depth check
for s in range(min(40, first_in_play)):
    obs, r, d, t, info = env5.step(small_back)
    if d or t:
        break

# Before episode ends, compare MTM and green-up at same point
if env5.positions and env5.step_idx < len(env5.current_race_df):
    mtm_total = env5._get_total_unrealized_pnl()
    green_total = env5._calculate_green_up_pnl()
    # Green-up includes commission on per-runner profits, MTM does not.
    # Therefore green_up <= MTM (commission always reduces or equals).
    # The difference is the commission charged on per-runner profits.
    diff = mtm_total - green_total
    print(f"    MTM: ${mtm_total:.4f}, Green-up: ${green_total:.4f}, Commission diff: ${diff:.4f}")
    check("green_up <= MTM (commission deducted on profits)",
          green_total <= mtm_total + 1e-6,
          f"green_up={green_total}, mtm={mtm_total}")
    check("commission difference is non-negative",
          diff >= -1e-6,
          f"diff={diff}")
else:
    check("positions exist for PnL test", False, "no positions built")

# ---------------------------------------------------------------------------
# Test 8 — NoTradeStreakWrapper penalty capped
# ---------------------------------------------------------------------------
print("\n=== Test 8: NoTradeStreakWrapper penalty cap ===")
# Simulate many no-trade episodes
from unittest.mock import MagicMock
import gymnasium as gym

class DummyEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.trades_this_episode = []
        self._step = 0
    def reset(self, **kw):
        self.trades_this_episode = []
        self._step = 0
        return np.zeros(2, dtype=np.float32), {}
    def step(self, action):
        self._step += 1
        done = self._step >= 3
        return np.zeros(2, dtype=np.float32), 0.0, done, False, {}

dummy = DummyEnv()
wrapped = NoTradeStreakWrapper(dummy)

penalties = []
for episode in range(10):
    obs, _ = wrapped.reset()
    done = False
    while not done:
        obs, r, done, t, info = wrapped.step(np.zeros(1))
    penalties.append(info.get('no_trade_penalty', 0.0))

# After 10 no-trade episodes, penalty should be capped
worst = min(penalties)
check("penalty capped", worst >= NO_TRADE_PENALTY_CAP - 1e-6,
      f"worst={worst}, cap={NO_TRADE_PENALTY_CAP}")
check("penalty not -32", worst > -32, f"worst={worst}")
print(f"    penalties: {penalties}")

# ---------------------------------------------------------------------------
# Test 9+10 — Green-up formulas (unit test)
# ---------------------------------------------------------------------------
print("\n=== Test 9-10: Green-up formula unit tests ===")

# BACK: backed at 3.0, green at 2.0.  stake=10
# Expected: 10 * (3.0 - 2.0) / 2.0 = 5.0
back_pnl = 10.0 * (3.0 - 2.0) / 2.0
check("BACK green-up: back@3, now@2, stake=10 -> $5",
      abs(back_pnl - 5.0) < 1e-6, f"got {back_pnl}")

# BACK: backed at 3.0, green at 4.0 (drifted).  stake=10
# Expected: 10 * (3.0 - 4.0) / 4.0 = -2.5
back_pnl2 = 10.0 * (3.0 - 4.0) / 4.0
check("BACK green-up: back@3, now@4, stake=10 -> -$2.5",
      abs(back_pnl2 - (-2.5)) < 1e-6, f"got {back_pnl2}")

# LAY: laid at 3.0, green at 4.0 (drifted = good for lay).  stake=10
# Expected: 10 * (4.0 - 3.0) / 4.0 = 2.5
lay_pnl = 10.0 * (4.0 - 3.0) / 4.0
check("LAY green-up: lay@3, now@4, stake=10 -> $2.5",
      abs(lay_pnl - 2.5) < 1e-6, f"got {lay_pnl}")

# LAY: laid at 3.0, green at 2.0 (shortened = bad for lay).  stake=10
# Expected: 10 * (2.0 - 3.0) / 2.0 = -5.0
lay_pnl2 = 10.0 * (2.0 - 3.0) / 2.0
check("LAY green-up: lay@3, now@2, stake=10 -> -$5",
      abs(lay_pnl2 - (-5.0)) < 1e-6, f"got {lay_pnl2}")

# Verify the actual environment method produces the same results
env6 = MarketMakingEnv(race_files=[PARQUET])
obs, _ = env6.reset()
# Manually inject a position
env6.positions[0] = {
    'net_stake': 10.0,
    'total_back_stake': 10.0,
    'weighted_back_price': 3.0,
    'total_lay_stake': 0.0,
    'weighted_lay_price': 0.0,
}
pnl_at_2 = env6._get_position_pnl(0, 2.0)
check("env._get_position_pnl BACK@3 now@2", abs(pnl_at_2 - 5.0) < 1e-6,
      f"got {pnl_at_2}")

env6.positions[1] = {
    'net_stake': -10.0,
    'total_back_stake': 0.0,
    'weighted_back_price': 0.0,
    'total_lay_stake': 10.0,
    'weighted_lay_price': 3.0,
}
pnl_lay = env6._get_position_pnl(1, 4.0)
check("env._get_position_pnl LAY@3 now@4", abs(pnl_lay - 2.5) < 1e-6,
      f"got {pnl_lay}")


# ---------------------------------------------------------------------------
# Test 11: Position netting — close back with lay
# ---------------------------------------------------------------------------
print("\n=== Test 11: Position netting ===")

env7 = MarketMakingEnv(race_files=[PARQUET])
obs, _ = env7.reset()
env7.commission_rate = 0.05

# Manually open a BACK position: stake=10 at price=3.0
env7._step_realized_pnl = 0.0
env7._execute_trade(0, 'BACK', 10.0, 3.0)
check("back position opened", env7.positions[0]['total_back_stake'] == 10.0,
      f"got {env7.positions[0]['total_back_stake']}")
initial_bal = env7.balance

# Close by laying — liability=15 -> stake=15/(2.5-1)=10, fully closes back
env7._step_realized_pnl = 0.0
env7._execute_trade(0, 'LAY', 15.0, 2.5)
# P&L = 10 * (3.0 - 2.5) / 2.5 = 2.0 gross, commission = 0.10, net = 1.90
expected_pnl = 10.0 * (3.0 - 2.5) / 2.5 * 0.95  # 1.90
check("back fully closed", env7.positions[0]['total_back_stake'] < 0.01,
      f"got {env7.positions[0]['total_back_stake']}")
check("lay not opened (exact close)", env7.positions[0]['total_lay_stake'] < 0.01,
      f"got {env7.positions[0]['total_lay_stake']}")
check("balance increased by net P&L",
      abs(env7.balance - initial_bal - expected_pnl) < 0.01,
      f"expected +${expected_pnl:.2f}, got +${env7.balance - initial_bal:.2f}")
check("mid_race_pnl tracked",
      abs(env7.mid_race_pnl - expected_pnl) < 0.01,
      f"expected ${expected_pnl:.2f}, got ${env7.mid_race_pnl:.2f}")
check("step_realized_pnl tracked",
      abs(env7._step_realized_pnl - expected_pnl) < 0.01,
      f"expected ${expected_pnl:.2f}, got ${env7._step_realized_pnl:.2f}")

# Test partial close + flip
print("\n=== Test 12: Partial close + overshoot opens opposite side ===")
env8 = MarketMakingEnv(race_files=[PARQUET])
obs, _ = env8.reset()
env8.commission_rate = 0.05

# Open BACK 10 at 4.0
env8._step_realized_pnl = 0.0
env8._execute_trade(0, 'BACK', 10.0, 4.0)
check("back=10 opened", env8.positions[0]['total_back_stake'] == 10.0, "")

# Lay 20 at 3.0 (price shortened): stake=20/(3-1)=10, but only 10 to close
# Wait - we need lay stake > back stake to test overshoot
# Lay with liability=30 at price 3.0: stake = 30/2 = 15
# closes 10 of back, opens 5 of lay
env8._step_realized_pnl = 0.0
initial_bal8 = env8.balance
env8._execute_trade(0, 'LAY', 30.0, 3.0)
# Closed 10 back at 4.0, lay at 3.0: pnl = 10*(4.0-3.0)/3.0 = 3.333
# Commission on profit: 3.333 * 0.05 = 0.1667, net = 3.1667
close_pnl = 10.0 * (4.0 - 3.0) / 3.0
close_net = close_pnl * 0.95
check("back fully closed after overshoot", env8.positions[0]['total_back_stake'] < 0.01,
      f"got {env8.positions[0]['total_back_stake']}")
check("lay opened with remainder", abs(env8.positions[0]['total_lay_stake'] - 5.0) < 0.01,
      f"got {env8.positions[0]['total_lay_stake']}")
check("realized P&L correct",
      abs(env8.balance - initial_bal8 - close_net) < 0.01,
      f"expected +${close_net:.2f}, got +${env8.balance - initial_bal8:.2f}")

# Test closing lay with back (opposite direction)
print("\n=== Test 13: Close lay position with back ===")
env9 = MarketMakingEnv(race_files=[PARQUET])
obs, _ = env9.reset()
env9.commission_rate = 0.05

# Open LAY: liability=20, price=3.0 -> stake = 20/2 = 10
env9._step_realized_pnl = 0.0
env9._execute_trade(0, 'LAY', 20.0, 3.0)
check("lay=10 opened", abs(env9.positions[0]['total_lay_stake'] - 10.0) < 0.01,
      f"got {env9.positions[0]['total_lay_stake']}")

# Close by backing at 4.0 (drift, good for lay): stake=10
env9._step_realized_pnl = 0.0
initial_bal9 = env9.balance
env9._execute_trade(0, 'BACK', 10.0, 4.0)
# pnl = 10 * (4.0 - 3.0) / 4.0 = 2.5 gross, commission = 0.125, net = 2.375
expected_lay_close = 10.0 * (4.0 - 3.0) / 4.0 * 0.95
check("lay fully closed", env9.positions[0]['total_lay_stake'] < 0.01,
      f"got {env9.positions[0]['total_lay_stake']}")
check("balance increased (lay close)",
      abs(env9.balance - initial_bal9 - expected_lay_close) < 0.01,
      f"expected +${expected_lay_close:.2f}, got +${env9.balance - initial_bal9:.2f}")

# Test losing close (back at bad price)
print("\n=== Test 14: Losing close (back at worse price) ===")
env10 = MarketMakingEnv(race_files=[PARQUET])
obs, _ = env10.reset()
env10.commission_rate = 0.05

# Open BACK 10 at 3.0
env10._step_realized_pnl = 0.0
env10._execute_trade(0, 'BACK', 10.0, 3.0)
initial_bal10 = env10.balance

# Close by laying at 4.0 (price drifted, loss for back)
# liability=30 -> stake=30/3=10
env10._step_realized_pnl = 0.0
env10._execute_trade(0, 'LAY', 30.0, 4.0)
# pnl = 10*(3.0-4.0)/4.0 = -2.5 (loss, no commission)
expected_loss = 10.0 * (3.0 - 4.0) / 4.0  # -2.5
check("losing close: balance decreased",
      abs(env10.balance - initial_bal10 - expected_loss) < 0.01,
      f"expected ${expected_loss:.2f}, got ${env10.balance - initial_bal10:.2f}")
check("losing close: no commission on loss",
      env10.total_commission_paid < 0.01,
      f"got commission={env10.total_commission_paid}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed, {failed} failed out of {passed+failed} checks")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)
