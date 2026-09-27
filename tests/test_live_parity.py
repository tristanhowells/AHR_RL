"""Replaying a recording message-by-message through the live MarketSession must
reproduce the offline training environment exactly (features, orders, P&L)."""
import glob

import numpy as np
import pytest

from ahr_rl.env import BetfairPreRaceEnv, EnvConfig
from ahr_rl.evaluate import make_scalper_policy
from ahr_rl.live.session import MarketSession
from ahr_rl.stream import read_recording
from ahr_rl.tape import build_tape

RECS = sorted(glob.glob("data/recordings/*.ndjson.gz"))


@pytest.mark.skipif(not RECS, reason="no recordings in data/recordings")
@pytest.mark.parametrize("path", RECS[:3])
def test_live_session_matches_offline_env(path):
    cfg = EnvConfig()
    # offline
    tape = build_tape(path)
    env = BetfairPreRaceEnv([tape], cfg)
    pol = make_scalper_policy()
    obs, _ = env.reset(options={"tape": tape})
    off_obs, done = [(env.ex.step, obs)], False
    while not done:
        obs, _, done, _, info = env.step(pol(obs, env))
        if not done:
            off_obs.append((env.ex.step, obs))
    # live replay
    rec = read_recording(path)
    s = MarketSession(rec.market_id, rec.market_start_ms, make_scalper_policy(), cfg, log=lambda *a: None)
    s.keep_trace = True
    for m in rec.messages:
        s.on_message(m)
        if s.done:
            break
    assert s.done
    assert len(s.obs_trace) == len(off_obs)
    for (s1, o1), (s2, o2) in zip(off_obs, s.obs_trace):
        assert s1 == s2
        for k in o1:
            np.testing.assert_allclose(o1[k], o2[k], atol=1e-6, err_msg=f"{k} @ step {s1}")
    for k in ("worst", "expected", "best"):
        assert s.result[k] == pytest.approx(info[k], abs=1e-9)
    assert len(s.ex.fills) == info["n_fills"]
