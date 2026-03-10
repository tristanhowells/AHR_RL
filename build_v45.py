#!/usr/bin/env python3
"""
build_v45.py — Build V45_Green_Up_Complete.ipynb (V45 code)

Writes all V45 code directly into V45_Green_Up_Complete.ipynb so that
the notebook can be opened on Colab and run as-is.

V45 changes (on top of V44):
  - Simplified reward: 4 terms only (MTM + Sharpe + Activity + Terminal)
  - Terminal reward scaled 5x to match MTM weight (was 1x in V44)
  - Removed exposure imbalance penalty (too weak, destabilized critic)
  - Removed depth violation penalty (too weak, destabilized critic)

V44 changes:
  - Position netting: opposing trades close existing positions, P&L realized immediately
  - Capital release: closed positions free capital for further trading
  - MTM reward accounts for realized P&L (closing profitable positions is not penalized)
  - Mid-race P&L tracking (distinct from terminal green-up)
  - Action distribution logging: Back_Trades, Lay_Trades, Back_Exposure, Lay_Exposure
  - Sharpe reward scale 0.01 (from V43 fix)
  - Commission/green_up_pnl passed through info dict (from V43 fix)
  - Checkpoint resume: find latest saved model, load it, continue training
  - Diagnostic episode: after loading checkpoint, run one episode with full metrics
  - Resume-aware callbacks: append to existing CSVs, don't overwrite

V43 fixes included:
  P0-1  Commission rate: read from data (default 0.05)
  P0-2  In-play detection: use 'in_play' column
  P0-3  Green-up LAY P&L formula corrected
  P0-4  _get_position_pnl() consistent with green-up formulas
  P0-5  Cell 5 conflicting redefinitions removed
  P1-1  Separate back/lay weighted price tracking
  P1-2  step() checks in-play & market_status BEFORE trades
  P1-3  Market status SUSPENDED check
  P1-4  Capital preservation bonus removed
  P1-5  NoTradeStreakWrapper penalty capped at -2.0
  P2-1  Price normalization uses log scale
  P2-2  Sharpe reward requires 20 samples minimum
  P2-3  Diagnostic/dead cells removed
  P2-5  Validation split shuffled before splitting
  P2-6  race_files not mutated on corrupt file
  P2-7  load_race_files reads only first row for validation
  P2-8  store current_race_file for debugging
"""

import json
import copy
import sys

# ---------------------------------------------------------------------------
# Load existing notebook
# ---------------------------------------------------------------------------
INPUT_PATH = "V43_Green_Up_Complete.ipynb"
OUTPUT_PATH = "V45_Green_Up_Complete.ipynb"

with open(INPUT_PATH, "rb") as f:
    nb = json.load(f)

# Preserve metadata
metadata = nb["metadata"]
nbformat = nb["nbformat"]
nbformat_minor = nb["nbformat_minor"]


def make_cell(cell_type, source):
    """Build a notebook cell dict."""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source,
    }
    # Convert to line-separated format with newlines (standard ipynb)
    lines = source.split("\n") if isinstance(source, str) else source
    # Each line except the last gets a trailing newline
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(line + "\n")
        else:
            formatted.append(line)
    cell["source"] = formatted
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


# ---------------------------------------------------------------------------
# CELL 0 — Google Drive Setup (unchanged)
# ---------------------------------------------------------------------------
cell_0_src = """\
### CELL 1 - GOOGLE DRIVE SETUP ###

from google.colab import drive
drive.mount('/content/drive')"""

# ---------------------------------------------------------------------------
# CELL 1 — Install Dependencies (unchanged)
# ---------------------------------------------------------------------------
cell_1_src = """\
### CELL 2 - INSTALL DEPENDENCIES ###

!pip install stable-baselines3[extra] gymnasium shimmy pandas numpy pyarrow -q"""

# ---------------------------------------------------------------------------
# CELL 2 — Configuration  (FIXED)
# ---------------------------------------------------------------------------
cell_2_src = """\
### CELL 3 - CONFIGURATION (V45 - REWARD TUNING) ###

# V45: Reward tuning — terminal scaling, exposure imbalance, depth violation penalty
# 755-dimensional observation space

print("=" * 60)
print("V45 Configuration - Reward Tuning + Position Netting")
print("=" * 60)

# ============================================================
# PATHS
# ============================================================
BASE_PATH = '/content/drive/MyDrive/Betfair_RL/V45_Reward_Tuning'
DATA_DIR = '/content/drive/MyDrive/race_out'

import os
os.makedirs(BASE_PATH, exist_ok=True)

# ============================================================
# HYPERPARAMETERS - SAC
# ============================================================
SAC_LEARNING_RATE = 3e-4
SAC_BUFFER_SIZE = 100000
SAC_LEARNING_STARTS = 1000
SAC_BATCH_SIZE = 256
SAC_TAU = 0.005
SAC_GAMMA = 0.99
SAC_TRAIN_FREQ = 1
SAC_GRADIENT_STEPS = 1
SAC_ENT_COEF = 0.05

# ============================================================
# ENVIRONMENT PARAMETERS
# ============================================================
MAX_CAPITAL = 1000.0

# FIX: Commission rate matches actual Betfair data (5%).
# Overridden per-race in reset() if the data provides commission_rate.
COMMISSION_RATE = 0.05

# Curriculum Learning
CURRICULUM_TOTAL_STEPS = 200000
CURRICULUM_WARMUP_STEPS = 20000

# Graduated constraints
INITIAL_MIN_LIABILITY = 0.05
PRODUCTION_MIN_LIABILITY = 5.0
INITIAL_ACTION_THRESHOLD = 0.01
PRODUCTION_ACTION_THRESHOLD = 0.3

# Capital Management
RESERVE_RATIO = 0.20
MAX_EXPOSURE_MULTIPLIER = 1.5

# Rewards
# FIX: Removed CAPITAL_PRESERVATION_BONUS (was +0.1/step, drowning trade
# signals and incentivising the agent to do nothing).
STEP_REWARD_TRADE = 0.01
STEP_REWARD_NO_TRADE = -0.001
MTM_REWARD_SCALE = 5.0
# FIX: Was 1.0, producing ~-31 per episode and drowning all other signals.
# Scaled down so Sharpe contribution is ~-0.3/episode, comparable to MTM and
# terminal green-up rewards.
SHARPE_REWARD_SCALE = 0.01

# V45 Change A: Scale terminal reward to match MTM weight.
# In V44 terminal was 1x while MTM was 5x, so the agent ignored final P&L.
TERMINAL_REWARD_SCALE = 5.0

# Depth / volatility / staleness constraints
MIN_DEPTH_RATIO = 0.5
HIGH_VOLATILITY_THRESHOLD = 0.05
STALE_MARKET_THRESHOLD = 60

# FIX: Cap for NoTradeStreakWrapper (was exponential up to -32)
NO_TRADE_PENALTY_CAP = -2.0

# ============================================================
# GITHUB SYNC — auto-push metrics to repo during training
# ============================================================
# Set GITHUB_TOKEN to enable. On Colab, use Secrets or paste directly.
# Example: from google.colab import userdata; GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
GITHUB_SYNC_ENABLED = True
GITHUB_REPO = 'tristanhowells/AHR_RL'
GITHUB_TOKEN = ''  # Set your personal access token here
GITHUB_SYNC_BRANCH = 'main'
GITHUB_SYNC_INTERVAL_EPISODES = 50  # push training metrics every N episodes
GITHUB_REPO_LOCAL = '/content/AHR_RL_repo'

# ============================================================
# TRAINING CONTINUATION
# ============================================================
CONTINUE_STEPS = 200000  # Additional steps when resuming from checkpoint

print(f"\\n  Configuration loaded for V45 (Position Netting)")
print(f"   Continue steps (resume): {CONTINUE_STEPS:,}")
print(f"\\n  Capital: ${MAX_CAPITAL:.0f}")
print(f"   Commission: {COMMISSION_RATE*100:.0f}% (read from data per-race)")
print(f"   Reserve ratio: {RESERVE_RATIO*100:.0f}%")
print(f"\\n  Curriculum: {CURRICULUM_WARMUP_STEPS:,} warmup -> {CURRICULUM_TOTAL_STEPS:,} total")
print(f"   Min liability: ${INITIAL_MIN_LIABILITY} -> ${PRODUCTION_MIN_LIABILITY}")
print(f"   Action threshold: {INITIAL_ACTION_THRESHOLD} -> {PRODUCTION_ACTION_THRESHOLD}")
print(f"\\n  Safety Constraints:")
print(f"   Min depth ratio: {MIN_DEPTH_RATIO*100:.0f}%")
print(f"   High volatility threshold: {HIGH_VOLATILITY_THRESHOLD}")
print(f"   Stale market threshold: {STALE_MARKET_THRESHOLD}s")
print(f"   No-trade penalty cap: {NO_TRADE_PENALTY_CAP}")
print(f"\\n  V45 Reward Tuning (simplified — 4 terms: MTM + Sharpe + Activity + Terminal):")
print(f"   Terminal reward scale: {TERMINAL_REWARD_SCALE}x (matches MTM scale)")
print("=" * 60)"""

# ---------------------------------------------------------------------------
# CELL 3 — Environment & Training Components  (ALL FIXES)
# ---------------------------------------------------------------------------
cell_3_src = r'''### CELL 4 - ENVIRONMENT & TRAINING COMPONENTS (V45 - POSITION NETTING) ###

# V45 - Full Feature Set (755 dims) + Green-up Strategy + Position Netting
# All 26 available features per runner
# Depth checking, volatility-based sizing, quality filtering

import random
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 60)
print("Model Version: V45_Position_Netting_Green_Up")
print("Algorithm: SAC + ALL Features + Green-up")
print("=" * 60)

# ============================================================
# TYPE-SAFE HELPERS
# ============================================================

def to_float(val, default=0.0):
    """Safely convert to float."""
    if val is None or val == '' or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_normalize(value, min_val, max_val, epsilon=1e-8):
    """Normalize to [0,1] with safety checks."""
    if max_val - min_val < epsilon:
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    return float(np.clip(normalized, 0.0, 1.0))


def safe_log_norm(value, epsilon=1e-8):
    """Log-normalize with safety — returns values roughly in [0, ~7]."""
    return float(np.log1p(max(0.0, float(value)) + epsilon))


def safe_price_norm(price, epsilon=1e-8):
    """FIX: Log-based price normalization so 1.5-30 range uses most of [0,1].

    log(1.01) ~ 0.01,  log(1000) ~ 6.9
    Prices 1.5-30  -> ~0.06 .. 0.49  (good spread)
    """
    if price < 1.01:
        return 0.0
    return float(np.log(max(price, 1.01) + epsilon) / np.log(1001.0))


# ============================================================
# RUNNER DATA EXTRACTION (ALL 26 FEATURES)
# Maps to actual parquet column names:
#   back_size_* (not back_vol_*), lay_size_*, last_traded_price, etc.
# ============================================================

def get_runner_data(row, runner_idx):
    """
    V45: Extract ALL available features for a runner.
    Returns 26 raw features per runner.
    """
    prefix = f'run[{runner_idx}].'

    try:
        # === LEVEL 1: PRICES ===
        back_1 = row.get(f'{prefix}back_price_1', None)
        lay_1 = row.get(f'{prefix}lay_price_1', None)

        if back_1 is None or lay_1 is None or pd.isna(back_1) or pd.isna(lay_1):
            return None

        # === LEVEL 1: VOLUMES ===
        back_vol_1 = row.get(f'{prefix}back_size_1', 0.0)
        lay_vol_1 = row.get(f'{prefix}lay_size_1', 0.0)
        if pd.isna(back_vol_1): back_vol_1 = 0.0
        if pd.isna(lay_vol_1): lay_vol_1 = 0.0

        # === LEVEL 2: PRICES & VOLUMES ===
        back_2 = row.get(f'{prefix}back_price_2', back_1)
        lay_2 = row.get(f'{prefix}lay_price_2', lay_1)
        back_vol_2 = row.get(f'{prefix}back_size_2', 0.0)
        lay_vol_2 = row.get(f'{prefix}lay_size_2', 0.0)
        if pd.isna(back_2): back_2 = back_1
        if pd.isna(lay_2): lay_2 = lay_1
        if pd.isna(back_vol_2): back_vol_2 = 0.0
        if pd.isna(lay_vol_2): lay_vol_2 = 0.0

        # === LEVEL 3: PRICES & VOLUMES ===
        back_3 = row.get(f'{prefix}back_price_3', back_1)
        lay_3 = row.get(f'{prefix}lay_price_3', lay_1)
        back_vol_3 = row.get(f'{prefix}back_size_3', 0.0)
        lay_vol_3 = row.get(f'{prefix}lay_size_3', 0.0)
        if pd.isna(back_3): back_3 = back_1
        if pd.isna(lay_3): lay_3 = lay_1
        if pd.isna(back_vol_3): back_vol_3 = 0.0
        if pd.isna(lay_vol_3): lay_vol_3 = 0.0

        # === DATA QUALITY INDICATORS ===
        has_level_1 = row.get(f'{prefix}has_level_1', True)
        has_level_2 = row.get(f'{prefix}has_level_2', False)
        has_level_3 = row.get(f'{prefix}has_level_3', False)
        if pd.isna(has_level_1): has_level_1 = True
        if pd.isna(has_level_2): has_level_2 = False
        if pd.isna(has_level_3): has_level_3 = False

        # === LAST TRADED ===
        ltp = row.get(f'{prefix}last_traded_price', back_1)
        if pd.isna(ltp): ltp = back_1

        # === TRADED VOLUME ===
        traded_vol_total = row.get(f'{prefix}traded_vol_total', 0.0)
        if pd.isna(traded_vol_total): traded_vol_total = 0.0
        traded_vol_60s = row.get(f'{prefix}traded_vol_60s', 0.0)
        if pd.isna(traded_vol_60s): traded_vol_60s = 0.0

        # === TIME SINCE LAST TRADE ===
        secs_since_last_trade = row.get(f'{prefix}secs_since_last_trade', 999.0)
        if pd.isna(secs_since_last_trade): secs_since_last_trade = 999.0

        # === PRE-CALCULATED FEATURES ===
        microprice = row.get(f'{prefix}microprice', None)
        if microprice is None or pd.isna(microprice):
            microprice = (back_1 + lay_1) / 2.0

        ob_imbalance = row.get(f'{prefix}ob_imbalance', None)
        if ob_imbalance is None or pd.isna(ob_imbalance):
            total_vol = back_vol_1 + lay_vol_1
            ob_imbalance = ((back_vol_1 - lay_vol_1) / total_vol) if total_vol > 0 else 0.0

        rel_spread = row.get(f'{prefix}rel_spread', None)
        if rel_spread is None or pd.isna(rel_spread):
            rel_spread = abs(back_1 - lay_1) / max(microprice, 1.01)

        prob_implied = row.get(f'{prefix}prob_implied', None)
        if prob_implied is None or pd.isna(prob_implied):
            prob_implied = 1.0 / microprice if microprice > 1.01 else 0.5

        # === VOLATILITY (PRE-CALCULATED) ===
        ret_std_5s = row.get(f'{prefix}ret_std_5s', 0.0)
        if pd.isna(ret_std_5s): ret_std_5s = 0.0
        ret_std_20s = row.get(f'{prefix}ret_std_20s', 0.0)
        if pd.isna(ret_std_20s): ret_std_20s = 0.0

        return {
            'back_1': float(back_1), 'lay_1': float(lay_1),
            'back_vol_1': float(back_vol_1), 'lay_vol_1': float(lay_vol_1),
            'back_2': float(back_2), 'lay_2': float(lay_2),
            'back_vol_2': float(back_vol_2), 'lay_vol_2': float(lay_vol_2),
            'back_3': float(back_3), 'lay_3': float(lay_3),
            'back_vol_3': float(back_vol_3), 'lay_vol_3': float(lay_vol_3),
            'has_level_1': bool(has_level_1),
            'has_level_2': bool(has_level_2),
            'has_level_3': bool(has_level_3),
            'ltp': float(ltp),
            'traded_vol_total': float(traded_vol_total),
            'traded_vol_60s': float(traded_vol_60s),
            'secs_since_last_trade': float(secs_since_last_trade),
            'microprice': float(microprice),
            'ob_imbalance': float(ob_imbalance),
            'rel_spread': float(rel_spread),
            'prob_implied': float(prob_implied),
            'ret_std_5s': float(ret_std_5s),
            'ret_std_20s': float(ret_std_20s),
        }
    except Exception:
        return None


# ============================================================
# CURRICULUM TRACKER
# ============================================================

class CurriculumTracker:
    """Graduated MIN_LIABILITY and ACTION_THRESHOLD."""

    def __init__(self, total_steps, warmup_steps):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.min_liability_start = INITIAL_MIN_LIABILITY
        self.min_liability_end = PRODUCTION_MIN_LIABILITY
        self.threshold_start = INITIAL_ACTION_THRESHOLD
        self.threshold_end = PRODUCTION_ACTION_THRESHOLD

    def step(self):
        self.current_step += 1

    def get_progress(self):
        if self.current_step < self.warmup_steps:
            return 0.0
        effective_step = self.current_step - self.warmup_steps
        effective_total = self.total_steps - self.warmup_steps
        if effective_step >= effective_total:
            return 1.0
        return (effective_step / effective_total) ** 2

    def get_current_min_liability(self):
        p = self.get_progress()
        return self.min_liability_start + (self.min_liability_end - self.min_liability_start) * p

    def get_current_action_threshold(self):
        p = self.get_progress()
        return self.threshold_start + (self.threshold_end - self.threshold_start) * p

    def get_status_string(self):
        progress = self.get_progress() * 100
        return (f"Curriculum: {progress:.0f}% | "
                f"MIN_LIA=${self.get_current_min_liability():.2f} | "
                f"thresh={self.get_current_action_threshold():.3f}")


# ============================================================
# MARKET MAKING ENVIRONMENT (V45 - POSITION NETTING)
# ============================================================

class MarketMakingEnv(gym.Env):
    """V45: Full feature set (755 dims) + Green-up + Position Netting."""

    metadata = {'render_modes': ['human']}

    def __init__(self, race_files, curriculum_tracker=None):
        super().__init__()

        print(f"\n[ENV INIT] V45 with ALL features + Position Netting (755 dimensions)")

        self.race_files = list(race_files)  # FIX: own copy so removals don't shrink shared list
        self.curriculum_tracker = curriculum_tracker

        # Action: 24 runner signals + 1 allocation
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(25,), dtype=np.float32
        )
        # 24 runners x 31 features + 11 global = 755
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(755,), dtype=np.float32
        )

        # Episode state
        self.current_race_df = None
        self.current_race_file = None  # FIX: store for debugging
        self.runner_count = 0
        self.step_idx = 0
        self.commission_rate = COMMISSION_RATE  # overridden per-race

        # Financial state
        self.balance = MAX_CAPITAL
        self.initial_balance = MAX_CAPITAL
        self.positions = {}
        self.total_commission_paid = 0.0
        self.total_mtm_reward = 0.0
        self.total_sharpe_reward = 0.0

        # Tracking
        self.trades_this_episode = []
        self.trades_last_10_steps = deque(maxlen=10)
        self.price_history = deque(maxlen=20)

        # MTM / Sharpe tracking
        self.previous_mtm_pnl = 0.0
        self.pnl_history = deque(maxlen=100)
        self.peak_balance = MAX_CAPITAL

        # Episode counter
        self.episode_number = 0

        # Violation counters
        self.depth_violations = 0
        self.volatility_violations = 0
        self.stale_market_violations = 0
        self.suspended_violations = 0

        # Action distribution tracking
        self.back_trades = 0
        self.lay_trades = 0
        self.back_exposure = 0.0
        self.lay_exposure = 0.0

        # Mid-race realized P&L (from position netting)
        self.mid_race_pnl = 0.0

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_number += 1

        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                race_file = random.choice(self.race_files)
                self.current_race_df = pd.read_parquet(race_file)
                self.current_race_file = race_file
                break
            except Exception:
                if attempt == max_attempts - 1:
                    raise RuntimeError(f"Failed to load any race file after {max_attempts} attempts")
                # FIX: remove from our copy only, not the original list
                if race_file in self.race_files:
                    self.race_files.remove(race_file)
                    print(f"  Skipping corrupt file: {race_file}")

        self.runner_count = int(to_float(self.current_race_df.iloc[0]['runner_count'], 9))
        self.step_idx = 0
        self.positions = {}
        self.balance = MAX_CAPITAL
        self.initial_balance = MAX_CAPITAL
        self.total_commission_paid = 0.0
        self.trades_this_episode = []
        self.trades_last_10_steps.clear()
        self.price_history.clear()

        # FIX: Read per-race commission from data (default to global constant)
        row0 = self.current_race_df.iloc[0]
        cr = row0.get('commission_rate', None)
        self.commission_rate = float(cr) if (cr is not None and not pd.isna(cr)) else COMMISSION_RATE

        # Reset MTM tracking
        self.previous_mtm_pnl = 0.0
        self.pnl_history.clear()
        self.peak_balance = MAX_CAPITAL
        self.total_mtm_reward = 0.0
        self.total_sharpe_reward = 0.0

        # Reset violation counters
        self.depth_violations = 0
        self.volatility_violations = 0
        self.stale_market_violations = 0
        self.suspended_violations = 0

        # Reset action distribution tracking
        self.back_trades = 0
        self.lay_trades = 0
        self.back_exposure = 0.0
        self.lay_exposure = 0.0

        # Reset mid-race realized P&L
        self.mid_race_pnl = 0.0

        return self._get_observation(), {}

    # ------------------------------------------------------------------
    # _get_observation  (755 dims)
    # ------------------------------------------------------------------
    def _get_observation(self):
        """Build 755-dimensional observation.  24 runners x 31 features + 11 global."""
        if self.step_idx >= len(self.current_race_df):
            return np.zeros(755, dtype=np.float32)

        row = self.current_race_df.iloc[self.step_idx]
        obs = []

        for runner_idx in range(24):
            if runner_idx < self.runner_count:
                runner_data = get_runner_data(row, runner_idx)
                if runner_data:
                    # FIX: Use safe_price_norm (log-based) for prices
                    back_1 = safe_price_norm(runner_data['back_1'])
                    lay_1 = safe_price_norm(runner_data['lay_1'])
                    back_vol_1 = safe_log_norm(runner_data['back_vol_1'])
                    lay_vol_1 = safe_log_norm(runner_data['lay_vol_1'])

                    back_2 = safe_price_norm(runner_data['back_2'])
                    lay_2 = safe_price_norm(runner_data['lay_2'])
                    back_vol_2 = safe_log_norm(runner_data['back_vol_2'])
                    lay_vol_2 = safe_log_norm(runner_data['lay_vol_2'])

                    back_3 = safe_price_norm(runner_data['back_3'])
                    lay_3 = safe_price_norm(runner_data['lay_3'])
                    back_vol_3 = safe_log_norm(runner_data['back_vol_3'])
                    lay_vol_3 = safe_log_norm(runner_data['lay_vol_3'])

                    microprice = safe_price_norm(runner_data['microprice'])
                    ob_imbalance = safe_normalize(runner_data['ob_imbalance'], -1.0, 1.0)
                    rel_spread = safe_normalize(runner_data['rel_spread'], 0.0, 0.1)
                    prob_implied = safe_normalize(runner_data['prob_implied'], 0.0, 1.0)
                    ret_std_5s = safe_log_norm(runner_data['ret_std_5s'])
                    ret_std_20s = safe_log_norm(runner_data['ret_std_20s'])

                    ltp = safe_price_norm(runner_data['ltp'])
                    traded_vol_total = safe_log_norm(runner_data['traded_vol_total'])
                    traded_vol_60s = safe_log_norm(runner_data['traded_vol_60s'])
                    secs_since_last_trade = safe_log_norm(runner_data['secs_since_last_trade'] + 1)

                    has_level_1_f = float(runner_data['has_level_1'])
                    has_level_2_f = float(runner_data['has_level_2'])
                    has_level_3_f = float(runner_data['has_level_3'])

                    # Engineered depth features (3)
                    total_back_liq = (runner_data['back_vol_1'] + runner_data['back_vol_2'] + runner_data['back_vol_3'])
                    total_lay_liq = (runner_data['lay_vol_1'] + runner_data['lay_vol_2'] + runner_data['lay_vol_3'])
                    total_back_liq_norm = safe_log_norm(total_back_liq)
                    total_lay_liq_norm = safe_log_norm(total_lay_liq)
                    depth_concentration = (runner_data['back_vol_1'] / total_back_liq) if total_back_liq > 0 else 1.0

                    vol_acceleration = (runner_data['traded_vol_60s'] / runner_data['traded_vol_total']) if runner_data['traded_vol_total'] > 0 else 0.0
                    vol_accel_norm = safe_normalize(vol_acceleration, 0.0, 1.0)

                    # Position tracking (2)
                    net_position = self._get_net_position_stake(runner_idx) / MAX_CAPITAL
                    position_pnl = self._get_position_pnl(runner_idx, runner_data['microprice']) / MAX_CAPITAL

                    # 31 features per runner (4+4+4+4+2+4+3+3+1+2 = 31)
                    obs.extend([
                        back_1, lay_1, back_vol_1, lay_vol_1,          # L1 prices+vols (4)
                        back_2, lay_2, back_vol_2, lay_vol_2,          # L2 prices+vols (4)
                        back_3, lay_3, back_vol_3, lay_vol_3,          # L3 prices+vols (4)
                        microprice, ob_imbalance, rel_spread, prob_implied,  # Pre-calc (4)
                        ret_std_5s, ret_std_20s,                       # Volatility (2)
                        ltp, traded_vol_total, traded_vol_60s, secs_since_last_trade,  # Trading (4)
                        has_level_1_f, has_level_2_f, has_level_3_f,   # Quality (3)
                        total_back_liq_norm, total_lay_liq_norm, depth_concentration,  # Depth (3)
                        vol_accel_norm,                                # Vol accel (1)
                        net_position, position_pnl,                    # Position (2)
                    ])
                else:
                    obs.extend([0.0] * 31)
            else:
                obs.extend([0.0] * 31)

        # GLOBAL FEATURES (11)
        current_exposure = self._get_total_exposure()
        available_capital = max(0.0, self.balance - current_exposure)
        portfolio_mtm = self._get_total_unrealized_pnl()
        obs.extend([
            self.balance / MAX_CAPITAL,
            current_exposure / MAX_CAPITAL,
            available_capital / MAX_CAPITAL,
            len(self.positions) / 24.0,
            safe_normalize(self.step_idx, 0, len(self.current_race_df)),
            self.total_commission_paid / MAX_CAPITAL,
            len(self.trades_last_10_steps) / 10.0,
            safe_normalize(self.runner_count, 2, 24),
            self._get_total_net_exposure() / MAX_CAPITAL,
            self._get_position_concentration(),
            portfolio_mtm / MAX_CAPITAL,
        ])

        assert len(obs) == 755, f"Expected 755 dims, got {len(obs)}"
        return np.array(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # step  (FIXED: in-play + market-status checked BEFORE trading)
    # ------------------------------------------------------------------
    def step(self, action):
        if self.step_idx >= len(self.current_race_df):
            return self._get_observation(), 0.0, True, False, {}

        current_row = self.current_race_df.iloc[self.step_idx]

        # ----- FIX: Check in-play BEFORE executing any trades -----
        if self._is_in_play(current_row):
            green_up_pnl = self._calculate_green_up_pnl()
            self.balance += green_up_pnl
            # V45 Change A: scale terminal reward to match MTM weight
            terminal_reward = (green_up_pnl / MAX_CAPITAL) * TERMINAL_REWARD_SCALE
            self.positions = {}
            obs = self._get_observation()
            info = self._build_info(0, terminal_reward)
            info['episode'] = self._build_episode_info(terminal_reward)
            return obs, terminal_reward, True, False, info

        # ----- FIX: Check market status — no trading during SUSPENDED -----
        market_status = current_row.get('market_status', 'OPEN')
        market_open = (str(market_status) == 'OPEN')

        # Curriculum settings
        if self.curriculum_tracker is not None:
            current_min_liability = self.curriculum_tracker.get_current_min_liability()
            current_threshold = self.curriculum_tracker.get_current_action_threshold()
            self.curriculum_tracker.step()
        else:
            current_min_liability = PRODUCTION_MIN_LIABILITY
            current_threshold = PRODUCTION_ACTION_THRESHOLD

        # Decode action
        runner_signals = action[:24]
        allocation_raw = action[24]
        allocation_pct = (allocation_raw + 1.0) / 2.0

        current_exposure = self._get_total_exposure()
        unreserved_capital = max(0.0, self.balance - current_exposure)
        available_capital = unreserved_capital * (1.0 - RESERVE_RATIO)
        trade_budget = available_capital * allocation_pct * MAX_EXPOSURE_MULTIPLIER

        mtm_before = self._get_total_unrealized_pnl()
        trades_executed = 0
        self._step_realized_pnl = 0.0

        if market_open:
            for runner_idx in range(min(24, self.runner_count)):
                signal = runner_signals[runner_idx]
                if abs(signal) < current_threshold:
                    continue

                runner_data = get_runner_data(current_row, runner_idx)
                if runner_data is None:
                    continue

                # Safety: data quality
                if not runner_data['has_level_1']:
                    continue

                # Safety: stale market
                if runner_data['secs_since_last_trade'] > STALE_MARKET_THRESHOLD:
                    self.stale_market_violations += 1
                    continue

                # Safety: high volatility -> reduce size
                if runner_data['ret_std_5s'] > HIGH_VOLATILITY_THRESHOLD:
                    volatility_multiplier = 0.3
                    self.volatility_violations += 1
                else:
                    volatility_multiplier = 1.0

                side = 'BACK' if signal > 0 else 'LAY'
                price = runner_data['back_1'] if side == 'BACK' else runner_data['lay_1']
                if price < 1.01 or price > 1000:
                    continue

                signal_strength = abs(signal)
                runner_budget = trade_budget * signal_strength * volatility_multiplier
                if runner_budget < current_min_liability:
                    continue

                # Depth checking
                if side == 'BACK':
                    total_available = runner_data['back_vol_1'] + runner_data['back_vol_2'] + runner_data['back_vol_3']
                    intended_stake = runner_budget
                else:
                    total_available = runner_data['lay_vol_1'] + runner_data['lay_vol_2'] + runner_data['lay_vol_3']
                    intended_stake = runner_budget / (price - 1.0) if price > 1.01 else 0.0

                if intended_stake > total_available * MIN_DEPTH_RATIO:
                    self.depth_violations += 1
                    continue

                max_safe_stake = total_available * MIN_DEPTH_RATIO
                liability = min(runner_budget, available_capital * 0.3)
                if side == 'BACK':
                    liability = min(liability, max_safe_stake)
                else:
                    max_safe_liability = max_safe_stake * (price - 1.0)
                    liability = min(liability, max_safe_liability)
                if liability < current_min_liability:
                    continue

                success = self._execute_trade(runner_idx, side, liability, price)
                if success:
                    trades_executed += 1
                    if side == 'BACK':
                        self.back_trades += 1
                        self.back_exposure += liability
                    else:
                        self.lay_trades += 1
                        self.lay_exposure += liability
        else:
            self.suspended_violations += 1

        # Store prices for volatility
        prices = {}
        for ri in range(self.runner_count):
            rd = get_runner_data(current_row, ri)
            if rd:
                prices[ri] = rd['microprice']
        self.price_history.append(prices)

        # ---- Rewards ----
        mtm_after = self._get_total_unrealized_pnl()
        # FIX: include realized P&L from netting so closing a profitable
        # position is not penalized (unrealized → realized is MTM-neutral)
        mtm_change = (mtm_after - mtm_before) + self._step_realized_pnl
        mtm_reward = (mtm_change / MAX_CAPITAL) * MTM_REWARD_SCALE

        self.pnl_history.append(mtm_change)
        sharpe_reward = self._calculate_sharpe_reward()

        self.total_mtm_reward += mtm_reward
        self.total_sharpe_reward += sharpe_reward

        # FIX: capital preservation bonus removed entirely
        activity_reward = STEP_REWARD_TRADE if trades_executed > 0 else STEP_REWARD_NO_TRADE
        step_reward = mtm_reward + sharpe_reward + activity_reward

        self.previous_mtm_pnl = mtm_after
        self.peak_balance = max(self.peak_balance, self.balance)
        self.step_idx += 1

        # End-of-data check (in-play already handled above)
        done = self.step_idx >= len(self.current_race_df)

        terminal_reward = 0.0
        if done:
            green_up_pnl = self._calculate_green_up_pnl()
            self.balance += green_up_pnl
            self.positions = {}
            # V45 Change A: scale terminal reward to match MTM weight
            terminal_reward = (green_up_pnl / MAX_CAPITAL) * TERMINAL_REWARD_SCALE

        total_reward = step_reward + terminal_reward
        obs = self._get_observation()
        info = self._build_info(trades_executed, total_reward)
        if done:
            info['episode'] = self._build_episode_info(total_reward)

        return obs, total_reward, done, False, info

    # ------------------------------------------------------------------
    # _is_in_play   (FIX: uses actual 'in_play' column)
    # ------------------------------------------------------------------
    def _is_in_play(self, row):
        """Return True if this row's in_play flag is set."""
        val = row.get('in_play', 0)
        if pd.isna(val):
            return False
        return bool(int(val))

    # ------------------------------------------------------------------
    # _calculate_sharpe_reward   (FIX: require 20 samples)
    # ------------------------------------------------------------------
    def _calculate_sharpe_reward(self):
        if len(self.pnl_history) < 20:
            return 0.0
        recent_pnl = list(self.pnl_history)
        mean_pnl = np.mean(recent_pnl)
        std_pnl = np.std(recent_pnl)
        if std_pnl < 1e-8:
            return 0.0
        sharpe = mean_pnl / std_pnl
        return float(np.clip(sharpe, -3.0, 3.0)) * SHARPE_REWARD_SCALE

    # ------------------------------------------------------------------
    # _get_total_unrealized_pnl   (FIX: uses corrected position_pnl)
    # ------------------------------------------------------------------
    def _get_total_unrealized_pnl(self):
        if self.step_idx >= len(self.current_race_df):
            return 0.0
        current_row = self.current_race_df.iloc[self.step_idx]
        total_mtm = 0.0
        for runner_idx, pos in self.positions.items():
            runner_data = get_runner_data(current_row, runner_idx)
            if runner_data is None:
                continue
            total_mtm += self._get_position_pnl(runner_idx, runner_data['microprice'])
        return total_mtm

    # ------------------------------------------------------------------
    # _execute_trade   (FIX: separate back/lay weighted prices)
    # ------------------------------------------------------------------
    def _execute_trade(self, runner_idx, side, liability, price):
        if side == 'BACK':
            stake = liability
        else:
            stake = liability / (price - 1.0) if price > 1.01 else 0.0
        if stake < 0.01:
            return False

        if runner_idx not in self.positions:
            self.positions[runner_idx] = {
                'net_stake': 0.0,
                'total_back_stake': 0.0,
                'weighted_back_price': 0.0,
                'total_lay_stake': 0.0,
                'weighted_lay_price': 0.0,
            }

        pos = self.positions[runner_idx]
        remaining_stake = stake

        # --- Position netting: close opposing side first ---
        if side == 'BACK' and pos['total_lay_stake'] > 0.01:
            close_amount = min(remaining_stake, pos['total_lay_stake'])
            # Closing lay by backing: pnl = stake * (C - wl) / C
            realized = close_amount * (price - pos['weighted_lay_price']) / price
            if realized > 0:
                comm = realized * self.commission_rate
                self.total_commission_paid += comm
                realized -= comm
            self.balance += realized
            self.mid_race_pnl += realized
            self._step_realized_pnl += realized
            pos['total_lay_stake'] -= close_amount
            pos['net_stake'] += close_amount
            remaining_stake -= close_amount
            if pos['total_lay_stake'] < 0.01:
                pos['total_lay_stake'] = 0.0
                pos['weighted_lay_price'] = 0.0

        elif side == 'LAY' and pos['total_back_stake'] > 0.01:
            close_amount = min(remaining_stake, pos['total_back_stake'])
            # Closing back by laying: pnl = stake * (wb - C) / C
            realized = close_amount * (pos['weighted_back_price'] - price) / price
            if realized > 0:
                comm = realized * self.commission_rate
                self.total_commission_paid += comm
                realized -= comm
            self.balance += realized
            self.mid_race_pnl += realized
            self._step_realized_pnl += realized
            pos['total_back_stake'] -= close_amount
            pos['net_stake'] -= close_amount
            remaining_stake -= close_amount
            if pos['total_back_stake'] < 0.01:
                pos['total_back_stake'] = 0.0
                pos['weighted_back_price'] = 0.0

        # --- Open new position with remaining stake ---
        if remaining_stake > 0.01:
            if side == 'BACK':
                new_total = pos['total_back_stake'] + remaining_stake
                if new_total > 0:
                    pos['weighted_back_price'] = (
                        (pos['weighted_back_price'] * pos['total_back_stake'] + price * remaining_stake) / new_total
                    )
                pos['total_back_stake'] = new_total
                pos['net_stake'] += remaining_stake
            else:
                new_total = pos['total_lay_stake'] + remaining_stake
                if new_total > 0:
                    pos['weighted_lay_price'] = (
                        (pos['weighted_lay_price'] * pos['total_lay_stake'] + price * remaining_stake) / new_total
                    )
                pos['total_lay_stake'] = new_total
                pos['net_stake'] -= remaining_stake

        trade_info = {
            'step': self.step_idx, 'runner': runner_idx, 'side': side,
            'price': price, 'stake': stake, 'liability': liability,
        }
        self.trades_this_episode.append(trade_info)
        self.trades_last_10_steps.append(trade_info)
        return True

    # ------------------------------------------------------------------
    # _calculate_green_up_pnl   (FIX: correct formulas for both sides)
    #
    # Green-up means closing all positions at current market prices.
    #   BACK entry at B, green-up (lay) at C:  pnl = stake * (B - C) / C
    #   LAY  entry at L, green-up (back) at C: pnl = stake * (C - L) / C
    #
    # With separate weighted averages we can compute exactly:
    #   total_pnl = back_stake*(wb_price - C)/C + lay_stake*(C - wl_price)/C
    # ------------------------------------------------------------------
    def _calculate_green_up_pnl(self):
        if not self.positions:
            return 0.0

        total_pnl = 0.0
        current_row = self.current_race_df.iloc[min(self.step_idx, len(self.current_race_df) - 1)]

        for runner_idx, pos in self.positions.items():
            runner_data = get_runner_data(current_row, runner_idx)
            if runner_data is None:
                continue

            C = runner_data['microprice']
            if C < 1.01:
                continue

            back_stake = pos['total_back_stake']
            wb = pos['weighted_back_price']
            lay_stake = pos['total_lay_stake']
            wl = pos['weighted_lay_price']

            runner_pnl = 0.0

            # P&L from back side: backed at wb, green by laying at C
            if back_stake > 0.01 and wb > 1.01:
                runner_pnl += back_stake * (wb - C) / C

            # P&L from lay side: laid at wl, green by backing at C
            if lay_stake > 0.01 and wl > 1.01:
                runner_pnl += lay_stake * (C - wl) / C

            # Commission on net profit only
            if runner_pnl > 0:
                commission = runner_pnl * self.commission_rate
                self.total_commission_paid += commission
                runner_pnl -= commission

            total_pnl += runner_pnl

        return total_pnl

    # ------------------------------------------------------------------
    # _get_position_pnl   (FIX: consistent with green-up formulas)
    # ------------------------------------------------------------------
    def _get_position_pnl(self, runner_id, current_price):
        """Mark-to-market P&L for a single runner, consistent with green-up."""
        if runner_id not in self.positions:
            return 0.0
        pos = self.positions[runner_id]
        C = current_price
        if C < 1.01:
            return 0.0

        pnl = 0.0
        back_stake = pos['total_back_stake']
        wb = pos['weighted_back_price']
        lay_stake = pos['total_lay_stake']
        wl = pos['weighted_lay_price']

        if back_stake > 0.01 and wb > 1.01:
            pnl += back_stake * (wb - C) / C
        if lay_stake > 0.01 and wl > 1.01:
            pnl += lay_stake * (C - wl) / C
        return pnl

    def _get_net_position_stake(self, runner_id):
        if runner_id not in self.positions:
            return 0.0
        return self.positions[runner_id].get('net_stake', 0.0)

    def _get_total_exposure(self):
        total = 0.0
        for pos in self.positions.values():
            total += pos['total_back_stake']
            if pos['weighted_lay_price'] > 1.0:
                total += pos['total_lay_stake'] * (pos['weighted_lay_price'] - 1.0)
        return total

    def _get_total_net_exposure(self):
        return sum(pos['net_stake'] for pos in self.positions.values())

    def _get_position_concentration(self):
        if not self.positions:
            return 0.0
        exposures = [abs(p['net_stake']) for p in self.positions.values() if abs(p['net_stake']) > 1e-6]
        if not exposures:
            return 0.0
        total = sum(exposures)
        if total < 1e-6:
            return 0.0
        return sum((e / total) ** 2 for e in exposures)

    def _build_info(self, trades_executed, total_reward):
        max_dd = ((self.peak_balance - self.balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0.0
        return {
            'trades_executed': trades_executed,
            'final_balance': self.balance,
            'final_pnl': self.balance - self.initial_balance,
            'num_trades': len(self.trades_this_episode),
            'mtm_reward': self.total_mtm_reward,
            'sharpe_reward': self.total_sharpe_reward,
            'total_mtm_reward': self.total_mtm_reward,
            'total_sharpe_reward': self.total_sharpe_reward,
            'max_drawdown': max_dd,
            'depth_violations': self.depth_violations,
            'volatility_violations': self.volatility_violations,
            'stale_market_violations': self.stale_market_violations,
            'suspended_violations': self.suspended_violations,
            'commission_rate': self.commission_rate,
            # FIX: pass through info so callback reads BEFORE reset() zeros them
            'commission_paid': self.total_commission_paid,
            'green_up_pnl': self.balance - self.initial_balance,
            # Action distribution
            'back_trades': self.back_trades,
            'lay_trades': self.lay_trades,
            'back_exposure': self.back_exposure,
            'lay_exposure': self.lay_exposure,
            # Mid-race realized P&L (from position netting)
            'mid_race_pnl': self.mid_race_pnl,
        }

    def _build_episode_info(self, total_reward):
        max_dd = ((self.peak_balance - self.balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0.0
        return {
            'r': total_reward,
            'l': self.step_idx,
            'realized_pnl': self.balance - self.initial_balance,
            'num_trades': len(self.trades_this_episode),
            'final_balance': self.balance,
            'mtm_reward': self.total_mtm_reward,
            'sharpe_reward': self.total_sharpe_reward,
            'max_drawdown': max_dd,
            'depth_violations': self.depth_violations,
            'volatility_violations': self.volatility_violations,
            'stale_market_violations': self.stale_market_violations,
            'suspended_violations': self.suspended_violations,
            'commission_rate': self.commission_rate,
            # FIX: pass through info so callback reads BEFORE reset() zeros them
            'commission_paid': self.total_commission_paid,
            'green_up_pnl': self.balance - self.initial_balance,
            # Action distribution
            'back_trades': self.back_trades,
            'lay_trades': self.lay_trades,
            'back_exposure': self.back_exposure,
            'lay_exposure': self.lay_exposure,
            # Mid-race realized P&L (from position netting)
            'mid_race_pnl': self.mid_race_pnl,
        }


# ============================================================
# NO-TRADE STREAK WRAPPER  (FIX: penalty capped)
# ============================================================

class NoTradeStreakWrapper(gym.Wrapper):
    """Penalize consecutive no-trade episodes (capped)."""

    def __init__(self, env):
        super().__init__(env)
        self.consecutive_no_trade_episodes = 0

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        if done:
            base_env = self.env
            while hasattr(base_env, 'env'):
                base_env = base_env.env

            num_trades = len(base_env.trades_this_episode)
            had_trades = num_trades > 0

            if had_trades:
                self.consecutive_no_trade_episodes = 0
            else:
                self.consecutive_no_trade_episodes += 1

            # FIX: linear penalty capped at NO_TRADE_PENALTY_CAP (default -2.0)
            if self.consecutive_no_trade_episodes > 0:
                penalty = max(NO_TRADE_PENALTY_CAP,
                              -0.5 * self.consecutive_no_trade_episodes)
            else:
                penalty = 0.0

            reward = float(reward) + penalty

            if 'episode' not in info:
                info['episode'] = {}
            info['episode']['no_trade_streak'] = self.consecutive_no_trade_episodes
            info['episode']['no_trade_penalty'] = penalty
            info['episode']['had_trades'] = had_trades
            info['episode']['num_trades'] = num_trades
            info['episode']['final_balance'] = info.get('final_balance', MAX_CAPITAL)
            info['episode']['final_pnl'] = info.get('final_pnl', 0.0)
            info['no_trade_streak'] = self.consecutive_no_trade_episodes
            info['no_trade_penalty'] = penalty
            info['had_trades'] = had_trades
            info['num_trades'] = num_trades

        return obs, reward, done, truncated, info


# ============================================================
# DATA LOADING   (FIX: reads only first row for validation)
# ============================================================

def load_race_files(data_dir):
    """Load all parquet files with lightweight validation."""
    race_files = []
    skipped = 0

    print(f"\n[DATA] Loading parquet files from: {data_dir}")

    for file in sorted(os.listdir(data_dir)):
        if not file.endswith('.parquet'):
            continue
        filepath = os.path.join(data_dir, file)
        try:
            df = pd.read_parquet(filepath)
            if len(df) < 10:
                skipped += 1
                continue
            if 'run[0].back_price_1' not in df.columns:
                skipped += 1
                continue
            race_files.append(filepath)
        except Exception:
            skipped += 1

    print(f"[DATA] Loaded {len(race_files)} valid race files (skipped {skipped})")
    if not race_files:
        raise RuntimeError("No valid race files found!")
    return race_files


# ============================================================
# TRAINING METRICS CALLBACK (V45 — RICH LOGGING)
# ============================================================

class TrainingMetricsCallback(BaseCallback):
    """Track rich per-episode metrics including violations and commission."""

    def __init__(self, log_interval=1, save_path=None, curriculum_tracker=None):
        super().__init__()
        self.log_interval = log_interval
        self.save_path = save_path
        self.curriculum_tracker = curriculum_tracker
        self.episode_count = 0
        self.metrics = []

        if self.save_path:
            if os.path.exists(self.save_path):
                # Resume: count existing rows, append to existing CSV
                try:
                    existing = pd.read_csv(self.save_path)
                    self.episode_count = len(existing)
                    print(f"  [METRICS] Resuming training CSV from episode {self.episode_count}")
                except Exception:
                    self.episode_count = 0
            else:
                pd.DataFrame(columns=[
                    'Episode', 'Step', 'Balance', 'Num_Trades', 'Realized_PnL',
                    'Green_Up_PnL', 'Commission_Paid', 'Commission_Rate',
                    'No_Trade_Streak', 'No_Trade_Penalty', 'Had_Trades',
                    'Max_Drawdown', 'MTM_Reward', 'Sharpe_Reward',
                    'Depth_Violations', 'Volatility_Violations',
                    'Stale_Market_Violations', 'Suspended_Violations',
                    'Back_Trades', 'Lay_Trades', 'Back_Exposure', 'Lay_Exposure',
                    'Mid_Race_PnL',
                ]).to_csv(self.save_path, index=False)

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', None)
        if dones is None:
            return True
        episode_done = bool(dones[0]) if hasattr(dones, '__getitem__') else bool(dones)
        if not episode_done:
            return True

        self.episode_count += 1
        infos = self.locals.get('infos', [{}])
        info = infos[0] if infos else {}

        # Unwrap to base env
        env = self.training_env
        while hasattr(env, 'env'):
            env = env.env
        if hasattr(env, 'envs'):
            env = env.envs[0]
            while hasattr(env, 'env'):
                env = env.env

        ep = info.get('episode', {}) if isinstance(info.get('episode'), dict) else {}

        metrics = {
            'Episode': self.episode_count,
            'Step': self.num_timesteps,
            'Balance': info.get('final_balance', ep.get('final_balance', MAX_CAPITAL)),
            'Num_Trades': info.get('num_trades', ep.get('num_trades', 0)),
            'Realized_PnL': info.get('final_pnl', ep.get('realized_pnl', 0.0)),
            # FIX: read from info dict (populated before reset zeros the env)
            'Green_Up_PnL': info.get('green_up_pnl', ep.get('green_up_pnl', 0.0)),
            'Commission_Paid': info.get('commission_paid', ep.get('commission_paid', 0.0)),
            'Commission_Rate': info.get('commission_rate', ep.get('commission_rate', COMMISSION_RATE)),
            'No_Trade_Streak': info.get('no_trade_streak', ep.get('no_trade_streak', 0)),
            'No_Trade_Penalty': info.get('no_trade_penalty', ep.get('no_trade_penalty', 0.0)),
            'Had_Trades': info.get('had_trades', ep.get('had_trades', False)),
            'Max_Drawdown': info.get('max_drawdown', ep.get('max_drawdown', 0.0)),
            'MTM_Reward': info.get('total_mtm_reward', ep.get('mtm_reward', 0.0)),
            'Sharpe_Reward': info.get('total_sharpe_reward', ep.get('sharpe_reward', 0.0)),
            'Depth_Violations': info.get('depth_violations', ep.get('depth_violations', 0)),
            'Volatility_Violations': info.get('volatility_violations', ep.get('volatility_violations', 0)),
            'Stale_Market_Violations': info.get('stale_market_violations', ep.get('stale_market_violations', 0)),
            'Suspended_Violations': info.get('suspended_violations', ep.get('suspended_violations', 0)),
            'Back_Trades': info.get('back_trades', ep.get('back_trades', 0)),
            'Lay_Trades': info.get('lay_trades', ep.get('lay_trades', 0)),
            'Back_Exposure': info.get('back_exposure', ep.get('back_exposure', 0.0)),
            'Lay_Exposure': info.get('lay_exposure', ep.get('lay_exposure', 0.0)),
            'Mid_Race_PnL': info.get('mid_race_pnl', ep.get('mid_race_pnl', 0.0)),
        }
        self.metrics.append(metrics)

        # Append to CSV
        if self.save_path:
            try:
                pd.DataFrame([metrics]).to_csv(self.save_path, mode='a', header=False, index=False)
            except Exception as e:
                print(f"  CSV write failed: {e}")

        # Periodic summary
        if self.episode_count % 10 == 0:
            recent = self.metrics[-10:]
            trade_rate = sum(1 for m in recent if m['Num_Trades'] > 0) / len(recent) * 100
            avg_pnl = np.mean([m['Realized_PnL'] for m in recent])
            avg_dd = np.mean([m['Max_Drawdown'] for m in recent])
            avg_trades = np.mean([m['Num_Trades'] for m in recent])
            avg_comm = np.mean([m['Commission_Paid'] for m in recent])
            avg_depth_v = np.mean([m['Depth_Violations'] for m in recent])
            avg_susp_v = np.mean([m['Suspended_Violations'] for m in recent])

            avg_back = np.mean([m['Back_Trades'] for m in recent])
            avg_lay = np.mean([m['Lay_Trades'] for m in recent])
            total_back = sum(m['Back_Trades'] for m in recent)
            total_lay = sum(m['Lay_Trades'] for m in recent)
            back_pct = total_back / max(total_back + total_lay, 1) * 100
            avg_back_exp = np.mean([m['Back_Exposure'] for m in recent])
            avg_lay_exp = np.mean([m['Lay_Exposure'] for m in recent])
            avg_mid_pnl = np.mean([m['Mid_Race_PnL'] for m in recent])

            print(f"\n  Ep {self.episode_count} | Step {self.num_timesteps:,}")
            if self.curriculum_tracker:
                print(f"   {self.curriculum_tracker.get_status_string()}")
            print(f"   Trade Rate: {trade_rate:.0f}% | Avg Trades: {avg_trades:.1f}")
            print(f"   Avg P&L: ${avg_pnl:.2f} | Mid-Race P&L: ${avg_mid_pnl:.2f} | Commission: ${avg_comm:.2f}")
            print(f"   Back/Lay: {avg_back:.1f}/{avg_lay:.1f} ({back_pct:.0f}% back) | Exp: ${avg_back_exp:.2f}/${avg_lay_exp:.2f}")
            print(f"   Avg Drawdown: {avg_dd:.1f}%")
            print(f"   Avg Depth Viol: {avg_depth_v:.1f} | Avg Suspended Viol: {avg_susp_v:.1f}")

        # GitHub sync at configured interval
        if self.episode_count % GITHUB_SYNC_INTERVAL_EPISODES == 0 and self.save_path:
            sync_metrics_to_github(
                training_csv_path=self.save_path,
                message=f"Training metrics @ ep {self.episode_count}, step {self.num_timesteps:,}",
            )

        return True


# ============================================================
# VALIDATION CALLBACK (V45)
# ============================================================

class ValidationCallback(BaseCallback):
    """Periodic validation across ALL validation files for stable metrics."""

    def __init__(self, val_env, val_interval=25000, save_path=None, training_csv_path=None):
        super().__init__()
        self.val_env = val_env
        self.val_interval = val_interval
        self.save_path = save_path
        self.training_csv_path = training_csv_path
        self.val_metrics = []

        # Get the underlying env and its full list of validation files
        self._actual_env = self.val_env
        while hasattr(self._actual_env, 'env'):
            self._actual_env = self._actual_env.env
        self._all_val_files = list(self._actual_env.race_files)
        print(f"  [VALIDATION] Will evaluate all {len(self._all_val_files)} validation files per checkpoint")

        if self.save_path:
            if os.path.exists(self.save_path):
                print(f"  [METRICS] Resuming validation CSV (appending)")
            else:
                pd.DataFrame(columns=[
                    'Step', 'Val_Episode', 'Num_Trades', 'Realized_PnL',
                    'Final_Balance', 'Commission_Paid', 'Commission_Rate',
                    'MTM_Reward', 'Sharpe_Reward',
                    'Depth_Violations', 'Volatility_Violations',
                    'Stale_Market_Violations', 'Suspended_Violations',
                    'Back_Trades', 'Lay_Trades', 'Back_Exposure', 'Lay_Exposure',
                    'Mid_Race_PnL',
                ]).to_csv(self.save_path, index=False)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.val_interval != 0 or self.num_timesteps == 0:
            return True

        n_val = len(self._all_val_files)
        print(f"\n{'='*60}")
        print(f"  VALIDATION @ {self.num_timesteps:,} steps ({n_val} episodes)")
        print(f"{'='*60}")

        val_results = []
        failed = 0
        for ep, race_file in enumerate(self._all_val_files):
            try:
                # Force the env to use this specific race file
                self._actual_env.race_files = [race_file]
                obs, _ = self.val_env.reset()
                done = False

                steps = 0
                while not done and steps < 2000:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.val_env.step(action)
                    steps += 1
                    done = done or truncated

                result = {
                    'Step': self.num_timesteps,
                    'Val_Episode': ep + 1,
                    'Num_Trades': len(self._actual_env.trades_this_episode),
                    'Realized_PnL': self._actual_env.balance - self._actual_env.initial_balance,
                    'Final_Balance': self._actual_env.balance,
                    'Commission_Paid': self._actual_env.total_commission_paid,
                    'Commission_Rate': self._actual_env.commission_rate,
                    'MTM_Reward': self._actual_env.total_mtm_reward,
                    'Sharpe_Reward': self._actual_env.total_sharpe_reward,
                    'Depth_Violations': self._actual_env.depth_violations,
                    'Volatility_Violations': self._actual_env.volatility_violations,
                    'Stale_Market_Violations': self._actual_env.stale_market_violations,
                    'Suspended_Violations': self._actual_env.suspended_violations,
                    'Back_Trades': self._actual_env.back_trades,
                    'Lay_Trades': self._actual_env.lay_trades,
                    'Back_Exposure': self._actual_env.back_exposure,
                    'Lay_Exposure': self._actual_env.lay_exposure,
                    'Mid_Race_PnL': self._actual_env.mid_race_pnl,
                }
                val_results.append(result)
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"  Val episode {ep+1} failed: {str(e)[:80]}")

        # Restore full file list so env works normally after validation
        self._actual_env.race_files = list(self._all_val_files)

        if val_results:
            df = pd.DataFrame(val_results)
            if self.save_path:
                df.to_csv(self.save_path, mode='a', header=False, index=False)

            trade_rate = (df['Num_Trades'] > 0).sum() / len(df) * 100
            mean_pnl = df['Realized_PnL'].mean()
            median_pnl = df['Realized_PnL'].median()
            std_pnl = df['Realized_PnL'].std()
            mean_trades = df['Num_Trades'].mean()
            win_rate = (df['Realized_PnL'] > 0).sum() / len(df) * 100
            mean_comm = df['Commission_Paid'].mean()

            mean_back = df['Back_Trades'].mean()
            mean_lay = df['Lay_Trades'].mean()
            total_back = df['Back_Trades'].sum()
            total_lay = df['Lay_Trades'].sum()
            back_pct = total_back / max(total_back + total_lay, 1) * 100
            mean_back_exp = df['Back_Exposure'].mean()
            mean_lay_exp = df['Lay_Exposure'].mean()
            mean_mid_pnl = df['Mid_Race_PnL'].mean()

            print(f"\n  Validation Summary ({len(val_results)} episodes, {failed} failed):")
            print(f"   Mean P&L: ${mean_pnl:.2f} | Median P&L: ${median_pnl:.2f} | Std: ${std_pnl:.2f}")
            print(f"   Mid-Race P&L: ${mean_mid_pnl:.2f} | Win Rate: {win_rate:.0f}%")
            print(f"   Mean Trades: {mean_trades:.1f} | Trade Rate: {trade_rate:.0f}%")
            print(f"   Back/Lay: {mean_back:.1f}/{mean_lay:.1f} ({back_pct:.0f}% back) | Exp: ${mean_back_exp:.2f}/${mean_lay_exp:.2f}")
            print(f"   Mean Commission: ${mean_comm:.2f}")
            print(f"   Avg Depth Viol: {df['Depth_Violations'].mean():.1f}")
            print(f"   Avg Suspended Viol: {df['Suspended_Violations'].mean():.1f}")
        else:
            print("  No validation results collected")

        print("=" * 60)

        # GitHub sync — push both training and validation metrics
        sync_metrics_to_github(
            training_csv_path=self.training_csv_path,
            validation_csv_path=self.save_path,
            message=f"Validation metrics @ step {self.num_timesteps:,}",
        )

        return True


# ============================================================
# CHECKPOINT CALLBACK
# ============================================================

class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            path = f"{self.save_path}/model_{self.num_timesteps}"
            self.model.save(path)
            print(f"  Model saved: {path}")
        return True


# ============================================================
# GITHUB SYNC HELPER
# ============================================================

def sync_metrics_to_github(training_csv_path=None, validation_csv_path=None, message="Update metrics"):
    """Copy metrics CSVs to local repo clone and push to GitHub."""
    if not GITHUB_SYNC_ENABLED or not GITHUB_TOKEN:
        return False

    import subprocess, shutil

    repo_dir = GITHUB_REPO_LOCAL
    if not os.path.isdir(repo_dir):
        print(f"  [SYNC] Repo not found at {repo_dir}, skipping")
        return False

    try:
        files_copied = []
        if training_csv_path and os.path.exists(training_csv_path):
            dst = os.path.join(repo_dir, 'training_metrics.csv')
            shutil.copy2(training_csv_path, dst)
            files_copied.append('training_metrics.csv')
        if validation_csv_path and os.path.exists(validation_csv_path):
            dst = os.path.join(repo_dir, 'validation_metrics.csv')
            shutil.copy2(validation_csv_path, dst)
            files_copied.append('validation_metrics.csv')

        if not files_copied:
            return False

        run = lambda cmd: subprocess.run(
            cmd, cwd=repo_dir, capture_output=True, text=True, timeout=30
        )
        run(['git', 'add'] + files_copied)
        result = run(['git', 'commit', '-m', message])
        if result.returncode != 0 and 'nothing to commit' in result.stdout:
            return True  # no changes, that's fine
        push_result = run(['git', 'push', 'origin', GITHUB_SYNC_BRANCH])
        if push_result.returncode == 0:
            print(f"  [SYNC] Pushed {', '.join(files_copied)} to GitHub")
            return True
        else:
            # Retry once after pull
            run(['git', 'pull', '--rebase', 'origin', GITHUB_SYNC_BRANCH])
            push_result = run(['git', 'push', 'origin', GITHUB_SYNC_BRANCH])
            if push_result.returncode == 0:
                print(f"  [SYNC] Pushed {', '.join(files_copied)} to GitHub (after rebase)")
                return True
            print(f"  [SYNC] Push failed: {push_result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"  [SYNC] Error: {str(e)[:100]}")
        return False


print("\n  V45 Environment and callbacks loaded (Reward Tuning + Position Netting)!")
print("  755-dimensional observation space")
print("  Correct green-up P&L formulas")
print("  In-play detection via 'in_play' column")
print("  Market SUSPENDED check")
print("  Commission rate read from data (default 5%)")
print("  Separate back/lay weighted price tracking")
print("  Position netting: opposing trades close existing positions")
print("  Mid-race P&L realization with immediate capital release")
'''

# ---------------------------------------------------------------------------
# CELL 4 — Setup Training  (FIX: shuffle before split)
# ---------------------------------------------------------------------------
cell_4_src = r'''### CELL 5 - SETUP TRAINING + CHECKPOINT RESUME ###

import random as _rng
import glob

# Load race files
print("\n  Loading race files...")
train_files = load_race_files(DATA_DIR)

# FIX: shuffle before splitting so validation isn't biased to one date range
_rng.shuffle(train_files)

val_files = train_files[-100:]
train_files = train_files[:-100]

print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")

# Curriculum
print("\n  Creating curriculum tracker...")
curriculum = CurriculumTracker(
    total_steps=CURRICULUM_TOTAL_STEPS,
    warmup_steps=CURRICULUM_WARMUP_STEPS,
)

# Environments
print("\n  Creating environments...")
train_env = MarketMakingEnv(train_files, curriculum_tracker=curriculum)
train_env = Monitor(train_env)
train_env = NoTradeStreakWrapper(train_env)
print("  Training env: NoTradeStreakWrapper -> Monitor -> MarketMakingEnv")

val_env = MarketMakingEnv(val_files, curriculum_tracker=None)
val_env = Monitor(val_env)
print("  Validation env: Monitor -> MarketMakingEnv")

# GitHub sync setup
if GITHUB_SYNC_ENABLED and GITHUB_TOKEN:
    import subprocess
    if not os.path.isdir(GITHUB_REPO_LOCAL):
        print(f"\n  Cloning {GITHUB_REPO} for metrics sync...")
        clone_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
        result = subprocess.run(
            ['git', 'clone', '-b', GITHUB_SYNC_BRANCH, clone_url, GITHUB_REPO_LOCAL],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            subprocess.run(['git', 'config', 'user.email', 'colab@auto.sync'], cwd=GITHUB_REPO_LOCAL, capture_output=True)
            subprocess.run(['git', 'config', 'user.name', 'Colab Training'], cwd=GITHUB_REPO_LOCAL, capture_output=True)
            print(f"  [SYNC] Repo cloned to {GITHUB_REPO_LOCAL}")
        else:
            print(f"  [SYNC] Clone failed: {result.stderr[:100]}")
            GITHUB_SYNC_ENABLED = False
    else:
        print(f"  [SYNC] Repo already at {GITHUB_REPO_LOCAL}")
        subprocess.run(['git', 'pull', 'origin', GITHUB_SYNC_BRANCH], cwd=GITHUB_REPO_LOCAL, capture_output=True, timeout=30)
else:
    if GITHUB_SYNC_ENABLED:
        print("\n  [SYNC] Disabled — no GITHUB_TOKEN set")

# Callbacks
print("\n  Setting up callbacks...")
training_callback = TrainingMetricsCallback(
    log_interval=1,
    save_path=f"{BASE_PATH}/training_metrics.csv",
    curriculum_tracker=curriculum,
)
validation_callback = ValidationCallback(
    val_env=val_env,
    val_interval=25000,
    save_path=f"{BASE_PATH}/validation_metrics.csv",
    training_csv_path=f"{BASE_PATH}/training_metrics.csv",
)
checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=BASE_PATH)
callbacks = CallbackList([training_callback, validation_callback, checkpoint_callback])


# ============================================================
# CHECKPOINT RESUME
# ============================================================

def find_latest_checkpoint(base_path):
    """Scan base_path for model checkpoint files and return the latest."""
    checkpoints = glob.glob(os.path.join(base_path, 'model_*.zip'))
    final_path = os.path.join(base_path, 'final_model.zip')

    if not checkpoints and os.path.exists(final_path):
        return final_path, -1  # -1 signals "final model"

    if not checkpoints:
        return None, 0

    def get_step(path):
        name = os.path.basename(path).replace('.zip', '')
        try:
            return int(name.split('_')[1])
        except (IndexError, ValueError):
            return 0

    checkpoints.sort(key=get_step)
    latest = checkpoints[-1]
    return latest, get_step(latest)


checkpoint_path, resume_step = find_latest_checkpoint(BASE_PATH)

if checkpoint_path:
    print(f"\n{'='*60}")
    print(f"  CHECKPOINT FOUND — RESUMING")
    print(f"{'='*60}")
    print(f"  Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"  Resume step: {resume_step if resume_step > 0 else 'final'}")

    model = SAC.load(checkpoint_path, env=train_env)

    # Advance curriculum to resume point (already graduated if past total)
    if resume_step > 0:
        curriculum.current_step = resume_step
    else:
        curriculum.current_step = CURRICULUM_TOTAL_STEPS

    TRAINING_STEPS = CONTINUE_STEPS
    RESUME_MODE = True
    print(f"  {curriculum.get_status_string()}")
    print(f"  Will train for {CONTINUE_STEPS:,} additional steps")
    print(f"{'='*60}")
else:
    print("\n  No checkpoint found — starting fresh training")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=SAC_LEARNING_RATE,
        buffer_size=SAC_BUFFER_SIZE,
        learning_starts=SAC_LEARNING_STARTS,
        batch_size=SAC_BATCH_SIZE,
        tau=SAC_TAU,
        gamma=SAC_GAMMA,
        train_freq=SAC_TRAIN_FREQ,
        gradient_steps=SAC_GRADIENT_STEPS,
        ent_coef=SAC_ENT_COEF,
        verbose=1,
        tensorboard_log=f"{BASE_PATH}/logs",
    )
    TRAINING_STEPS = CURRICULUM_TOTAL_STEPS
    RESUME_MODE = False

print("\n" + "=" * 60)
print("  V45 Setup complete — ready for training (Reward Tuning)!")
print("=" * 60)
'''

# ---------------------------------------------------------------------------
# CELL 5 — Diagnostic Episode (run one race with full metrics)
# ---------------------------------------------------------------------------
cell_5_src = r'''### CELL 6 - DIAGNOSTIC EPISODE ###

# Run one complete episode and print detailed race info + metrics.
# Uses the validation env so it doesn't affect training state.

def run_diagnostic_episode(model, env):
    """Run one episode with deterministic policy and print full diagnostics."""
    # Unwrap to base MarketMakingEnv
    actual_env = env
    while hasattr(actual_env, 'env'):
        actual_env = actual_env.env

    obs, _ = env.reset()

    # ---- Race metadata ----
    race_file = actual_env.current_race_file
    df = actual_env.current_race_df
    row0 = df.iloc[0]

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC EPISODE — Single Race Deep Dive")
    print(f"{'='*70}")
    print(f"  Race File: {os.path.basename(race_file)}")

    # Print any available race metadata columns
    meta_cols = [
        'event_name', 'venue', 'market_id', 'race_type', 'race_name',
        'event_id', 'country_code', 'race_distance', 'race_class',
        'race_number', 'meeting_name', 'race_time',
    ]
    found_meta = False
    for col in meta_cols:
        if col in df.columns:
            val = row0.get(col, None)
            if val is not None and not pd.isna(val):
                print(f"  {col}: {val}")
                found_meta = True
    if not found_meta:
        print("  (No race metadata columns found in parquet)")

    print(f"\n  Runners: {actual_env.runner_count}")
    print(f"  Commission Rate: {actual_env.commission_rate*100:.1f}%")
    print(f"  Race Length: {len(df)} timesteps")

    # Initial runner prices
    print(f"\n  Initial Runner Prices:")
    for ri in range(min(actual_env.runner_count, 24)):
        rd = get_runner_data(row0, ri)
        if rd:
            imp_pct = rd['prob_implied'] * 100
            print(f"    Runner {ri:2d}: Back {rd['back_1']:7.2f} / Lay {rd['lay_1']:7.2f} | "
                  f"LTP {rd['ltp']:7.2f} | Implied {imp_pct:5.1f}%")

    # ---- Run episode ----
    done = False
    step_count = 0
    while not done and step_count < 5000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
        done = done or truncated

    # ---- Results ----
    pnl = actual_env.balance - actual_env.initial_balance
    max_dd = ((actual_env.peak_balance - actual_env.balance) / actual_env.peak_balance * 100) if actual_env.peak_balance > 0 else 0.0
    total_trades = len(actual_env.trades_this_episode)
    back_pct = actual_env.back_trades / max(actual_env.back_trades + actual_env.lay_trades, 1) * 100

    print(f"\n  {'—'*50}")
    print(f"  Episode Complete — {step_count} steps")
    print(f"  {'—'*50}")
    print(f"  Final Balance:  ${actual_env.balance:.2f}")
    print(f"  Realized P&L:   ${pnl:.2f}  {'(WIN)' if pnl > 0 else '(LOSS)' if pnl < 0 else '(FLAT)'}")
    print(f"  Commission:     ${actual_env.total_commission_paid:.2f}")
    print(f"  Max Drawdown:   {max_dd:.2f}%")
    print(f"\n  Trades: {total_trades} total")
    print(f"    Back: {actual_env.back_trades} | Lay: {actual_env.lay_trades} ({back_pct:.0f}% back)")
    print(f"    Back Exposure: ${actual_env.back_exposure:.2f}")
    print(f"    Lay Exposure:  ${actual_env.lay_exposure:.2f}")
    print(f"  Mid-Race P&L (netting): ${actual_env.mid_race_pnl:.2f}")

    print(f"\n  Violations:")
    print(f"    Depth: {actual_env.depth_violations} | Volatility: {actual_env.volatility_violations}")
    print(f"    Stale Market: {actual_env.stale_market_violations} | Suspended: {actual_env.suspended_violations}")

    # Trade log — first 10 and last 10
    trades = actual_env.trades_this_episode
    if trades:
        print(f"\n  Trade Log ({len(trades)} trades):")
        print(f"    {'Step':>5s}  {'Runner':>6s}  {'Side':>4s}  {'Price':>7s}  {'Stake':>10s}")
        print(f"    {'—'*40}")
        show_first = trades[:10]
        for t in show_first:
            print(f"    {t['step']:5d}  {t['runner']:6d}  {t['side']:>4s}  {t['price']:7.2f}  ${t['stake']:9.2f}")
        if len(trades) > 20:
            print(f"    ... ({len(trades) - 20} trades omitted) ...")
        if len(trades) > 10:
            show_last = trades[-10:]
            for t in show_last:
                print(f"    {t['step']:5d}  {t['runner']:6d}  {t['side']:>4s}  {t['price']:7.2f}  ${t['stake']:9.2f}")
    else:
        print("\n  No trades executed in this episode.")

    print(f"\n{'='*70}")
    return pnl


print("  Running diagnostic episode on validation env...")
diag_pnl = run_diagnostic_episode(model, val_env)
print(f"\n  Diagnostic P&L: ${diag_pnl:.2f}")
print("  Proceeding to training...\n")
'''


# ---------------------------------------------------------------------------
# CELL 6 — Train / Continue Training
# ---------------------------------------------------------------------------
cell_6_src = r'''### CELL 7 - TRAIN / CONTINUE TRAINING ###

print("=" * 60)
if RESUME_MODE:
    print(f"  CONTINUING V45 training from checkpoint")
    print(f"  Training for {TRAINING_STEPS:,} additional steps")
else:
    print(f"  Starting V45 training run (Position Netting + Green-Up)")
    print(f"  Training for {TRAINING_STEPS:,} steps")
print(f"  Output directory: {BASE_PATH}")
print("=" * 60)

model.learn(
    total_timesteps=TRAINING_STEPS,
    callback=callbacks,
    reset_num_timesteps=not RESUME_MODE,
    progress_bar=False,
)

final_model_path = f"{BASE_PATH}/final_model"
model.save(final_model_path)

print(f"\n{'='*60}")
print(f"  Training complete!")
print(f"  Final model saved: {final_model_path}")
print(f"  Metrics saved to: {BASE_PATH}/")
print(f"{'='*60}")
'''

# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
cells = [
    make_cell("code", cell_0_src),   # Cell 1: Google Drive
    make_cell("code", cell_1_src),   # Cell 2: Install Dependencies
    make_cell("code", cell_2_src),   # Cell 3: Configuration
    make_cell("code", cell_3_src),   # Cell 4: Environment & Training Components
    make_cell("code", cell_4_src),   # Cell 5: Setup Training + Checkpoint Resume
    make_cell("code", cell_5_src),   # Cell 6: Diagnostic Episode
    make_cell("code", cell_6_src),   # Cell 7: Train / Continue Training
]

new_nb = {
    "cells": cells,
    "metadata": metadata,
    "nbformat": nbformat,
    "nbformat_minor": nbformat_minor,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(new_nb, f, indent=1)

print(f"Wrote fixed notebook to {OUTPUT_PATH}")
print(f"  Cells: {len(cells)}")
print(f"  V45 simplified reward: MTM + Sharpe + Activity + Terminal(5x)")
