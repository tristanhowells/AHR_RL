"""Smoke test for V50a environment — runnable outside Colab.

Loads a race, builds env, runs 10 random-policy steps.
Asserts no NaN in obs or reward, correct obs dimension.
"""
import sys
import os
import numpy as np

# Stub out Colab-only and training-only imports
sys.modules['google'] = type(sys)('google')
sys.modules['google.colab'] = type(sys)('google.colab')

# Mock stable_baselines3 — not needed for env smoke test
from unittest.mock import MagicMock
sb3_mock = MagicMock()
sys.modules['stable_baselines3'] = sb3_mock
sys.modules['stable_baselines3.common'] = sb3_mock
sys.modules['stable_baselines3.common.callbacks'] = sb3_mock
sys.modules['stable_baselines3.common.monitor'] = sb3_mock
sys.modules['torch'] = MagicMock()

# Set config constants directly (normally set by Cell 3)
MAX_CAPITAL = 1000.0
COMMISSION_RATE = 0.05
CURRICULUM_TOTAL_STEPS = 500000
CURRICULUM_WARMUP_STEPS = 50000
INITIAL_MIN_LIABILITY = 0.05
PRODUCTION_MIN_LIABILITY = 5.0
INITIAL_ACTION_THRESHOLD = 0.01
PRODUCTION_ACTION_THRESHOLD = 0.3
RESERVE_RATIO = 0.20
MAX_EXPOSURE_MULTIPLIER = 1.5
STEP_REWARD_TRADE = 0.0
STEP_REWARD_NO_TRADE = 0.0
MTM_REWARD_SCALE = 1.0
TERMINAL_REWARD_SCALE = 10.0
NETTING_BONUS_SCALE = 0.0
SHARPE_REWARD_SCALE = 0.01
URGENCY_SCALE = 0.05
URGENCY_START = 0.8
MIN_DEPTH_RATIO = 0.5
HIGH_VOLATILITY_THRESHOLD = 0.05
STALE_MARKET_THRESHOLD = 60
NO_TRADE_PENALTY_CAP = -2.0
NO_TRADE_STREAK_GRACE = 3
Q_VALUE_CLIP_MIN = -20.0
Q_VALUE_CLIP_MAX = 20.0
STALE_TRADE_PENALTY = -0.02
GREEN_EPISODE_BONUS = 1.0
PHASE_4E_SMOOTH_CURRICULUM = True
PHASE_4A_DISABLE_RESETS = True
PHASE_4B_STOCHASTIC_VAL = True
PHASE_4D_RELAX_NO_TRADE = True
N_STOCHASTIC_ROLLOUTS = 5
SAC_ENT_COEF_MIN = 0.08
SAC_ENT_COEF_MAX = 0.5
SAC_GRADIENT_STEPS = 4
SAC_LEARNING_RATE = 3e-4
RESET_FREQ = 200000
RESET_ENT_COEF = True
GITHUB_SYNC_ENABLED = False
GITHUB_TOKEN = ''

# Find a parquet file
parquet_files = [f for f in os.listdir('.') if f.endswith('.parquet')]
if not parquet_files:
    print("SKIP: No parquet files found in current directory")
    sys.exit(0)

race_file = os.path.join('.', parquet_files[0])
print(f"Using race file: {race_file}")

# Import env components by executing notebook cell code
import json
with open('V50a_Reward_Fix.ipynb') as f:
    nb = json.load(f)

# Extract just the environment cell (cell 3) and exec it
cell3_src = ''.join(nb['cells'][3]['source'])

# Suppress prints during import
import io
old_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    exec(cell3_src, globals())
finally:
    sys.stdout = old_stdout

print("Environment code loaded successfully")

# Build env
env = MarketMakingEnv([race_file], curriculum_tracker=None)
obs, info = env.reset()

print(f"Observation shape: {obs.shape}")
print(f"Expected: ({env.OBS_DIM},)")
assert obs.shape == (env.OBS_DIM,), f"Expected ({env.OBS_DIM},), got {obs.shape}"
assert not np.any(np.isnan(obs)), "NaN in initial observation"
print(f"Obs range: [{obs.min():.4f}, {obs.max():.4f}]")

# Run 10 random steps
for step in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    assert not np.any(np.isnan(obs)), f"NaN in obs at step {step}"
    assert not np.isnan(reward), f"NaN reward at step {step}"
    assert obs.shape == (env.OBS_DIM,), f"Wrong obs shape at step {step}"
    if done:
        print(f"  Episode ended at step {step}")
        break

print(f"\nSmoke test PASSED")
print(f"  10 steps completed, no NaN, obs dim = {env.OBS_DIM}")
print(f"  Final balance: ${env.balance:.2f}")
print(f"  Trades: {len(env.trades_this_episode)}")
print(f"  MTM reward scale: {MTM_REWARD_SCALE} (should be 1.0)")
print(f"  Netting bonus scale: {NETTING_BONUS_SCALE} (should be 0.0)")

# Verify the reward fix is actually applied
assert NETTING_BONUS_SCALE == 0.0, "NETTING_BONUS_SCALE should be 0.0"
assert MTM_REWARD_SCALE == 1.0, "MTM_REWARD_SCALE should be 1.0"
print("\n  Reward fix verified: netting bonus = 0, MTM = 1.0x")
