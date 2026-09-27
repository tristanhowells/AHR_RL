"""Evaluate a policy (trained or baseline) over a fixed set of races."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .env import BACK, JOIN, LAY, TAKE, BetfairPreRaceEnv, EnvConfig, encode_open
from .features import R_MAX


def run_episode(env: BetfairPreRaceEnv, tape, policy_fn) -> dict:
    obs, info = env.reset(options={"tape": tape})
    done = False
    while not done:
        a = policy_fn(obs, env)
        obs, _, done, _, info = env.step(a)
    return info


def evaluate(policy_fn, tapes: list, cfg: EnvConfig | None = None) -> pd.DataFrame:
    env = BetfairPreRaceEnv(tapes, cfg, cache_tapes=False)
    rows = [run_episode(env, t, policy_fn) for t in tapes]
    cols = ["market", "worst", "expected", "realised", "best", "green", "void", "turnover", "n_fills",
            "n_rejected", "n_opens", "n_closes", "n_stops"]
    return pd.DataFrame(rows)[cols]


def summarise(df: pd.DataFrame) -> dict:
    live = df[~df["void"]]
    traded = live[live["turnover"] > 0]
    return {
        "races": len(df),
        "mean_green_$": float(live["worst"].mean()),
        "median_green_$": float(live["worst"].median()),
        "pct_green": float((live["worst"] >= -0.005).mean() * 100),
        "pct_traded": float(len(traded) / max(1, len(live)) * 100),
        "pct_green_when_traded": float((traded["worst"] >= -0.005).mean() * 100) if len(traded) else float("nan"),
        "mean_expected_$": float(live["expected"].mean()),
        "mean_realised_$": float(live["realised"].mean()),
        "total_green_$": float(live["worst"].sum()),
        "mean_turnover_$": float(live["turnover"].mean()),
        "mean_trades": float(live["n_opens"].mean()),
        "worst_race_$": float(live["worst"].min()),
    }


# ---------------------------------------------------------------- baselines
def noop_policy(obs, env):
    return np.zeros(R_MAX, np.int64)


def make_random_policy(p_act: float = 0.02, seed: int = 0):
    rng = np.random.default_rng(seed)

    def f(obs, env):
        a = np.zeros(R_MAX, np.int64)
        for r in np.nonzero(obs["mask"])[0]:
            if rng.random() < p_act:
                a[r] = rng.integers(1, env.cfg.n_actions)
        return a

    return f


def make_scalper_policy(stake_idx: int = 1, tp_idx: int = 0, n_fav: int = 3):
    """Rule-based sanity baseline: repeatedly queue a 1-tick scalp (JOIN entry,
    alternating back/lay) on the favourites. Tells us whether passive spread
    capture survives the fill model."""

    def f(obs, env):
        ex = env.ex
        a = np.zeros(R_MAX, np.int64)
        favs = np.argsort(np.where(obs["mask"].astype(bool), -obs["runners"][:, 4], 9))[:n_fav]
        for r in favs:
            if obs["mask"][r] and r not in ex.brackets:
                side = BACK if (ex.step // 8) % 2 == 0 else LAY
                a[r] = encode_open(side, JOIN, stake_idx, tp_idx, env.cfg)
        return a

    return f


def make_torch_policy(model, deterministic: bool = True, device="cpu"):
    def f(obs, env):
        a, _, _ = model.act(obs, deterministic=deterministic, device=device)
        return a

    return f
