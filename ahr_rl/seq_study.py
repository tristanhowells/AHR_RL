"""P7: does a raw, frame-stacked view of the market add anything the hand-made features miss?

Same decisions as P6 (every 10s, every runner; pairs.parquet), three models, the
same TRAIN / HOLDOUT days:

  gbm     boosted trees on the P6 hand-made features (momentum 30/120s, volume rate,
          acceleration and share, WoM and its change, WAP gap, spread, rank, ...)
  seq     a small 1D CNN on raw stacked frames only: the runner's last --window-s
          seconds every --frame-s seconds, 9 channels per frame:
            mid (ticks, relative to now), spread, log top-3 back $, log top-3 lay $,
            WoM, log traded $ in the frame, implied-prob share, log market traded $
            in the frame, book-valid flag
  hybrid  the CNN plus the hand-made features

Targets (all known only after the decision, never seen as inputs):
  fwd_30 / fwd_60 / fwd_120   mid move in ticks over the next 30 / 60 / 120s
  back_k2 / lay_k2 / back_k8 / lay_k8   the P6 pair P&L (hedge k ticks away,
                                        'through' fills, unfilled hedge -> BSP)

Scored on HOLDOUT days:
  IC     Spearman correlation of prediction and outcome, and the CNN-minus-trees
         difference with a 95% race-bootstrap interval. "adds information" = the
         interval is above 0 on a forward-move target.
  pairs  trade the top 5% of holdout decisions by predicted pair P&L (the side with
         the higher prediction), as in P6: mean P&L per $1 at risk, race-clustered t.
         "edge" = mean > 0, t > 2, >= 20 races.

The CNN trains on the first 80% of TRAIN days and stops early on the last 20%.

    python -m ahr_rl.seq_study --pairs runs/pair_study/pairs.parquet --tapes "data/tapes/*.npz" --out runs/seq_study
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import race_filter
from .env import list_tapes, split_by_date
from .ladder import PRICES
from .pair_study import FEATURES, pair_pnl, race_t
from .tape import Tape

CHANNELS = ["mid_rel", "spread", "back3", "lay3", "wom", "vol", "prob", "mkt_vol", "valid"]
FWD_H = (30, 60, 120)
PAIR_T = [("back", 2), ("lay", 2), ("back", 8), ("lay", 8)]
TARGETS = [f"fwd_{h}" for h in FWD_H] + [f"{s}_k{k}" for s, k in PAIR_T]


# ----------------------------------------------------------------- frames
def tape_frames(path: str, rows: pd.DataFrame, window_s: float, frame_s: float):
    """Frames [n, F, C] (float16) and forward moves [n, 3] for the (s, runner) rows."""
    t = Tape.load(path)
    dt, T, R = t.dt, t.n_steps, t.n_runners
    ok = ~t.suspended.copy()
    if T > 1:
        ok[-1] = False
    end = int(np.where(ok)[0].max()) if ok.any() else T - 1
    bt, lt = t.back_tick[:, :, 0].astype(float), t.lay_tick[:, :, 0].astype(float)
    valid = (bt >= 0) & (lt >= 0) & t.active & ok[:, None]
    valid[end + 1:] = False
    midt = np.where(valid, (bt + lt) / 2.0, np.nan)
    mid_ff = pd.DataFrame(midt).ffill().to_numpy()  # carry the last valid mid
    spread = np.where(valid, lt - bt, np.nan)
    bs3, ls3 = t.back_size[:, :, :3].sum(-1), t.lay_size[:, :, :3].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        wom = np.where(bs3 + ls3 > 0, bs3 / (bs3 + ls3), 0.5)
        midp = np.where(valid, np.sqrt(PRICES[np.maximum(bt, 0).astype(int)] * PRICES[np.maximum(lt, 0).astype(int)]),
                        np.nan)
    ip = np.where(valid, 1.0 / midp, 0.0)
    share = ip / np.maximum(ip.sum(1, keepdims=True), 1e-12)
    vol = np.zeros((T, R))
    np.add.at(vol, (np.clip(t.trade_step, 0, T - 1), t.trade_runner), t.trade_vol)
    cv = np.vstack([np.zeros((1, R)), np.cumsum(vol, 0)])  # cv[s+1] = volume up to step s
    cmk = cv.sum(1)

    k = max(1, int(round(frame_s / dt)))
    F = int(round(window_s / frame_s))
    s = rows["s"].to_numpy(int)
    r = rows["runner"].to_numpy(int)
    idx = s[:, None] - (F - 1 - np.arange(F))[None, :] * k  # [n, F] step at the end of each frame
    inb = idx >= 0
    ic = np.clip(idx, 0, T - 1)
    icp = np.clip(idx - k, -1, T - 1)
    rr = np.broadcast_to(r[:, None], ic.shape)
    now = mid_ff[s, r][:, None]
    X = np.zeros(ic.shape + (len(CHANNELS),), np.float32)
    X[..., 0] = np.clip(np.nan_to_num(mid_ff[ic, rr] - now), -20, 20) / 5
    X[..., 1] = np.clip(np.nan_to_num(spread[ic, rr], nan=10), 0, 10) / 5
    X[..., 2] = np.log1p(bs3[ic, rr]) / 5
    X[..., 3] = np.log1p(ls3[ic, rr]) / 5
    X[..., 4] = wom[ic, rr] - 0.5
    X[..., 5] = np.log1p(np.maximum(cv[ic + 1, rr] - cv[icp + 1, rr], 0)) / 5
    X[..., 6] = share[ic, rr] * 5
    X[..., 7] = np.log1p(np.maximum(cmk[ic + 1] - cmk[icp + 1], 0)) / 8
    X[..., 8] = (valid[ic, rr] & inb).astype(np.float32)
    X[~inb] = 0.0

    Y = np.full((len(s), len(FWD_H)), np.nan, np.float32)
    for j, h in enumerate(FWD_H):
        s2 = np.minimum(s + int(h / dt), end)
        for i in range(len(s)):
            seg = midt[s[i]:s2[i] + 1, r[i]]
            ok_i = np.where(np.isfinite(seg))[0]
            if len(ok_i) and s2[i] > s[i] and np.isfinite(midt[s[i], r[i]]):
                Y[i, j] = np.clip(seg[ok_i[-1]] - midt[s[i], r[i]], -20, 20)
    return X.astype(np.float16), Y


def _one(args):
    path, rows, window_s, frame_s = args
    try:
        return rows.index.to_numpy(), *tape_frames(path, rows, window_s, frame_s)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}", flush=True)
        return None


# ----------------------------------------------------------------- models
def _net(n_ch: int, n_feat: int, n_out: int):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_ch, 32, 5, padding=2), nn.GELU(),
                nn.Conv1d(32, 64, 5, padding=4, dilation=2), nn.GELU(),
                nn.Conv1d(64, 64, 5, padding=8, dilation=4), nn.GELU())
            self.head = nn.Sequential(nn.Linear(128 + n_feat, 128), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(128, n_out))

        def forward(self, x, f):
            h = self.conv(x.transpose(1, 2))  # [B, 64, F]
            z = torch.cat([h.mean(-1), h[..., -1]], 1)
            if f is not None:
                z = torch.cat([z, f], 1)
            return self.head(z)

    return Net()


def train_cnn(X, Fs, Y, tr, va, use_feat: bool, device: str, epochs: int = 15, bs: int = 1024, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    n_out = Y.shape[1]
    net = _net(X.shape[2], Fs.shape[1] if use_feat else 0, n_out).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    mask = np.isfinite(Y)
    Yz = np.nan_to_num(Y)

    def batches(ix, shuffle):
        ix = np.random.permutation(ix) if shuffle else ix
        for i in range(0, len(ix), bs):
            b = ix[i:i + bs]
            yield (torch.from_numpy(X[b].astype(np.float32)).to(device),
                   torch.from_numpy(Fs[b]).to(device) if use_feat else None,
                   torch.from_numpy(Yz[b]).to(device), torch.from_numpy(mask[b]).to(device))

    def loss_on(ix):
        net.eval()
        tot, n = 0.0, 0.0
        with torch.no_grad():
            for x, f, y, m in batches(ix, False):
                e = ((net(x, f) - y) ** 2 * m).sum().item()
                tot, n = tot + e, n + m.sum().item()
        return tot / max(n, 1)

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        for x, f, y, m in batches(tr, True):
            loss = ((net(x, f) - y) ** 2 * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = loss_on(va)
        print(f"    epoch {ep + 1}: val loss {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(ix):
        out = []
        with torch.no_grad():
            for x, f, _, _ in batches(ix, False):
                out.append(net(x, f).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, n_out))

    return predict


def _standardise(Y, tr):
    lo, hi = np.nanpercentile(Y[tr], 1, axis=0), np.nanpercentile(Y[tr], 99, axis=0)
    Yc = np.clip(Y, lo, hi)
    mu, sd = np.nanmean(Yc[tr], 0), np.nanstd(Yc[tr], 0) + 1e-9
    return (Yc - mu) / sd


# ----------------------------------------------------------------- scoring
def ic(p, y):
    from scipy.stats import spearmanr

    m = np.isfinite(p) & np.isfinite(y)
    return float(spearmanr(p[m], y[m])[0]) if m.sum() > 10 else np.nan


def ic_diff_ci(pa, pb, y, race, n_boot: int = 200, seed: int = 0):
    """IC(a) - IC(b) with a 95% interval from resampling races."""
    rng = np.random.default_rng(seed)
    races = np.unique(race)
    groups = {rc: np.where(race == rc)[0] for rc in races}
    d0 = ic(pa, y) - ic(pb, y)
    ds = []
    for _ in range(n_boot):
        ix = np.concatenate([groups[rc] for rc in rng.choice(races, len(races))])
        ds.append(ic(pa[ix], y[ix]) - ic(pb[ix], y[ix]))
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    return d0, lo, hi


def top_pairs(pred: dict, g: dict, race, k: int, q: float = 0.95) -> dict:
    pb, pl = pred[f"back_k{k}"], pred[f"lay_k{k}"]
    score = np.fmax(pb, pl)
    gg = np.where(pb >= pl, g[f"back_k{k}"], g[f"lay_k{k}"])
    sel = (score >= np.nanquantile(score, q)) & np.isfinite(gg)
    rt = race_t(gg[sel] * 100, race[sel])
    return dict(mean_pct=rt["mean"], t=rt["t"], races=rt["races"], n=rt["n"])


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True, help="pairs.parquet written by P6 (pair_study)")
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-s", type=float, default=120.0)
    ap.add_argument("--frame-s", type=float, default=2.0)
    ap.add_argument("--max-rows", type=int, default=500000, help="random subsample of decisions (memory)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    race_filter.add_args(ap)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    t0 = time.time()

    paths = list_tapes(a.tapes)
    _, va_p, te_p = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va_p + te_p}  # same days as the unfiltered run
    paths = race_filter.filter_paths(paths, a)
    by_race = {os.path.basename(p).split(".npz")[0]: p for p in paths}
    df = pd.read_parquet(a.pairs)
    df = df[df["race"].isin(by_race)].reset_index(drop=True)
    if len(df) > a.max_rows:
        df = df.sample(a.max_rows, random_state=0).sort_values(["race", "s", "runner"]).reset_index(drop=True)
    print(f"{len(df)} decisions from {df['race'].nunique()} races, device {a.device}")

    # ---------------- frames
    F = int(round(a.window_s / a.frame_s))
    X = np.zeros((len(df), F, len(CHANNELS)), np.float16)
    Yf = np.full((len(df), len(FWD_H)), np.nan, np.float32)
    jobs = [(by_race[rc], g[["s", "runner"]], a.window_s, a.frame_s) for rc, g in df.groupby("race")]
    os.environ["OMP_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, res in enumerate(pool.map(_one, jobs, chunksize=4)):
            if res is not None:
                ix, x, y = res
                X[ix], Yf[ix] = x, y
            if (i + 1) % 200 == 0:
                print(f"  frames {i + 1}/{len(jobs)} races, {time.time() - t0:.0f}s", flush=True)
    print(f"frames {X.shape} ({X.nbytes / 1e9:.2f} GB) in {time.time() - t0:.0f}s")

    # ---------------- targets
    G = {f"fwd_{h}": Yf[:, j] for j, h in enumerate(FWD_H)}
    for side, k in PAIR_T:
        G[f"{side}_k{k}"] = pair_pnl(df, side, k, "bsp", "through")[0].astype(np.float32)
    Y = np.stack([G[c] for c in TARGETS], 1)

    hold = df["day"].isin(hold_days).to_numpy()
    tr_all = np.where(~hold)[0]
    ho = np.where(hold)[0]
    tdays = np.sort(df.loc[tr_all, "day"].unique())
    cut = tdays[int(len(tdays) * 0.8)] if len(tdays) > 4 else tdays[-1]
    fit = tr_all[df.loc[tr_all, "day"].to_numpy() < cut]
    val = tr_all[df.loc[tr_all, "day"].to_numpy() >= cut]
    print(f"train {len(fit)} rows (fit) + {len(val)} (early stopping), holdout {len(ho)} rows "
          f"({df.loc[ho, 'race'].nunique()} races)")

    Fs_raw = df[FEATURES].to_numpy(np.float32)
    mu, sd = np.nanmean(Fs_raw[tr_all], 0), np.nanstd(Fs_raw[tr_all], 0) + 1e-6
    Fs = np.nan_to_num((Fs_raw - mu) / sd).astype(np.float32)
    Yz = _standardise(Y, tr_all).astype(np.float32)

    preds = {}
    # ---------------- gbm
    print("\nfitting boosted trees (hand-made features)...", flush=True)
    from .pair_study import _fit

    P = np.full((len(ho), len(TARGETS)), np.nan)
    for j, c in enumerate(TARGETS):
        mdl = _fit(Fs_raw[tr_all].astype(float), Y[tr_all, j].astype(float))
        P[:, j] = mdl.predict(Fs_raw[ho].astype(float))
    preds["gbm"] = P
    # ---------------- cnn
    for name, use_feat in (("seq", False), ("hybrid", True)):
        print(f"\ntraining CNN '{name}' ({'frames + features' if use_feat else 'frames only'})...", flush=True)
        predict = train_cnn(X, Fs, Yz, fit, val, use_feat, a.device, epochs=a.epochs)
        preds[name] = predict(ho)

    # ---------------- scoring
    race = df.loc[ho, "race"].to_numpy()
    print("\n=== Holdout IC (Spearman) by target ===")
    rows = []
    for j, c in enumerate(TARGETS):
        y = Y[ho, j]
        rec = dict(target=c, **{f"ic_{m}": ic(preds[m][:, j], y) for m in preds})
        for m in ("seq", "hybrid"):
            d, lo, hi = ic_diff_ci(preds[m][:, j], preds["gbm"][:, j], y, race)
            rec.update({f"{m}-gbm": d, f"{m}-gbm_lo": lo, f"{m}-gbm_hi": hi})
        rows.append(rec)
    icd = pd.DataFrame(rows)
    icd.to_csv(os.path.join(a.out, "ic.csv"), index=False)
    print(icd.round(4).to_string(index=False))

    print("\n=== Trading the top 5% of holdout decisions by predicted pair P&L (per $1 at risk) ===")
    gh = {c: Y[ho, j] for j, c in enumerate(TARGETS)}
    rows = []
    for m in preds:
        pm = {c: preds[m][:, j] for j, c in enumerate(TARGETS)}
        for k in (2, 8):
            rows.append(dict(model=m, k=k, **top_pairs(pm, gh, race, k)))
    tp = pd.DataFrame(rows)
    tp.to_csv(os.path.join(a.out, "top_pairs.csv"), index=False)
    print(tp.round(3).to_string(index=False))

    fwd = icd[icd["target"].str.startswith("fwd")]
    adds = bool(((fwd["seq-gbm_lo"] > 0) | (fwd["hybrid-gbm_lo"] > 0)).any())
    nn = tp[tp["model"].isin(["seq", "hybrid"])]
    edge = bool(((nn["mean_pct"] > 0) & (nn["t"] > 2) & (nn["races"] >= 20)).any())
    verdict = dict(adds_information=adds, edge=edge,
                   best_ic_gain=float(np.nanmax(fwd[["seq-gbm", "hybrid-gbm"]].to_numpy())),
                   best_nn_top5=nn.sort_values("mean_pct", ascending=False).iloc[0].to_dict() if len(nn) else None)
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=float)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=float))
    print("frames ADD information beyond the hand-made features" if adds else
          "frames add NO information beyond the hand-made features (CI of the IC gain includes 0)")
    print("and the best pairs they pick make money on unseen days -> worth feeding frames to the agent" if edge else
          "and no model's top pairs make money on unseen days -> frame stacking won't give the agent an edge")
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
