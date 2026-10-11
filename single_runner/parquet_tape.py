"""Convert an old scraped parquet race (schema 1.3.x) into an ahr_rl Tape.

LOWER FIDELITY than real stream tapes, use for tests / pretraining only:
  * ~1.33 s irregular snapshots treated as a fixed grid (dt = median gap),
  * 3 ladder levels instead of 8, no projected BSP, no scratchings,
  * no per-trade data: each step's traded volume is placed at that step's LTP,
  * volume halved by default on the assumption that, like the stream `tv`, it
    counts both sides of each match (per-runner volume sums exactly to the
    market's total matched, which is consistent with that but doesn't prove it),
  * the 5% commission recorded in these files is lower than the 8-10% base
    rate seen in the stream recordings.
"""
import numpy as np
import pandas as pd

from ahr_rl.ladder import price_to_tick
from ahr_rl.tape import Tape


def _col(df, name, default=np.nan):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(np.float64)
    return np.full(len(df), default)


def parquet_to_tape(path, vol_scale=0.5):
    df = pd.read_parquet(path)
    in_play = _col(df, "in_play", 0.0) > 0.5
    end = int(np.argmax(in_play)) if in_play.any() else len(df)
    d = df.iloc[:end].reset_index(drop=True)
    T = len(d)
    R = int(d["runner_count"].iloc[0]) if "runner_count" in d.columns else \
        1 + max(int(c[4:c.index("]")]) for c in d.columns if c.startswith("run["))
    K = 3

    def tick(a):
        out = np.full(a.shape, -1, np.int16)
        ok = ~np.isnan(a)
        out[ok] = [price_to_tick(x) for x in a[ok]]
        return out

    bt = np.full((T, R, K), -1, np.int16)
    lt = np.full((T, R, K), -1, np.int16)
    bs = np.zeros((T, R, K), np.float32)
    ls = np.zeros((T, R, K), np.float32)
    ltp = np.full((T, R), -1, np.int16)
    tv = np.zeros((T, R), np.float32)
    for r in range(R):
        for k in range(K):
            bt[:, r, k] = tick(_col(d, f"run[{r}].back_price_{k + 1}"))
            lt[:, r, k] = tick(_col(d, f"run[{r}].lay_price_{k + 1}"))
            bs[:, r, k] = np.nan_to_num(_col(d, f"run[{r}].back_size_{k + 1}"))
            ls[:, r, k] = np.nan_to_num(_col(d, f"run[{r}].lay_size_{k + 1}"))
        ltp[:, r] = tick(_col(d, f"run[{r}].last_traded_price"))
        tv[:, r] = pd.Series(_col(d, f"run[{r}].traded_vol_total")).ffill().fillna(0).to_numpy() * vol_scale
    bs[bt < 0] = 0
    ls[lt < 0] = 0

    trades = []
    dtv = np.diff(tv, axis=0, prepend=tv[:1])
    for s, r in zip(*np.nonzero(dtv > 1e-6)):
        if ltp[s, r] >= 0:
            trades.append((s, r, ltp[s, r], dtv[s, r]))
    tr = np.array(trades, np.float64).reshape(-1, 4)

    ts = _col(d, "ts_unix") / 1000.0
    dt = float(np.nanmedian(np.diff(ts))) if T > 1 else 1.33
    t_rel = -np.nan_to_num(_col(d, "secs_to_off"))
    winner = _col(df, "result_winner_idx_first", -1)[0]
    date = str(d["race_date"].iloc[0]).replace("-", "") if "race_date" in d.columns else "19700101"
    market = str(d["file_market_id"].iloc[0]) if "file_market_id" in d.columns else "?"
    comm = _col(d, "commission_rate", 0.05)[0]
    return Tape(
        market_id=market, name=f"{date}_parquet_{market}", dt=dt, t_rel=t_rel.astype(np.float32),
        back_tick=bt, back_size=bs, lay_tick=lt, lay_size=ls, ltp_tick=ltp, tv=tv,
        spn=np.zeros((T, R), np.float32), active=(bt[..., 0] >= 0) | (lt[..., 0] >= 0),
        suspended=(d["market_status"].astype(str) != "OPEN").to_numpy(),
        total_matched=(pd.Series(_col(d, "total_matched_market")).ffill().fillna(0).to_numpy()
                       * vol_scale).astype(np.float32),
        trade_step=tr[:, 0].astype(np.int32), trade_runner=tr[:, 1].astype(np.int16),
        trade_tick=tr[:, 2].astype(np.int16), trade_vol=tr[:, 3].astype(np.float32),
        removal_step=np.zeros(0, np.int32), removal_runner=np.zeros(0, np.int16),
        removal_factor=np.zeros(0, np.float32), selection_ids=np.arange(R, dtype=np.int64),
        base_rate=float(100 * (0.05 if np.isnan(comm) else comm)), went_in_play=bool(in_play.any()),
        winner=int(winner) if not np.isnan(winner) else -1, bsp=np.zeros(R, np.float32),
    )
