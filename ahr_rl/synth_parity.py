"""P20 parity: does a synthetic race behave like a real one where it matters for trading?

tape_synth's own parity check (one bet at t-60s held to the result) only tests the result calibration. An agent
trained on synthetic races will exploit whatever the generator does differently, so this compares the dynamics a
trading agent can act on, real TRAIN races vs synthetic races, with race-clustered standard errors:

  calibrated (should pass by construction)
    vol_*           rms 0.5s change of log implied probability by time to the off
    vr60            60s variance ratio (< 1 = moves partly revert)
    trade_ic        corr(trade side, next 10s move): trades carry information
  CRITICAL, not fitted - what an agent could exploit if wrong
    mom30_ic        corr(own 30s move, next 30s move)              momentum / reversal (P16)
    mom30_ic_120    corr(own 30s move, next 120s move)
    driftfade_120   next-120s move of a 2nd/3rd favourite after a bottom-10% 30s drift (P16 / arm C), log %
    cross_ev        EV per $ of crossing now and greening at BSP (P15 k = 0): the cost of a round trip
    adverse_*       P15 adverse selection of a resting order 1 tick better: EV given a fill - EV of all orders
    fill60_*        share of those resting orders filled within 60s (too easy fills amplify any passive edge)
  information
    spread_*, l1_*, p_trade, bsp_err_sd, open_close_sd, runners

"ok" for a statistic = |synth - real| <= max(2 x combined s.e., its tolerance). parity_ok = every CRITICAL
statistic is ok.

    python -m ahr_rl.synth_parity --real "data/tapes/*.npz" --synth "runs/synth/pool/*.npz"
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .env import list_tapes
from .ladder import PRICES
from .passive_entry_study import add_values, sample_tape
from .rebalance_study import _prob_grid
from .synth_market import PB_EDGES, book_state, dyn_stats, pb_of
from .tape import Tape

PB_NAMES = ["<2", "2-3", "3-5", "5-10", "10-20", "20-50", ">50"]
TB_NAMES = ["<-480s", "-480..-300", "-300..-180", "-180..-90", "-90..-45", "-45..-15", "last 15s"]


def signal_sample(path: str, every_s: float = 20.0, last_s: float = 600.0) -> pd.DataFrame | None:
    """Own 30s move (pp of implied probability), favourite rank and the next 30s / 120s move (log %)."""
    t = Tape.load(path)
    if not t.went_in_play:
        return None
    p, valid, end, bt, lt = _prob_grid(t)
    k = lambda sec: int(round(sec / t.dt))
    dec = np.where((t.t_rel >= -last_s) & (t.t_rel <= -5.0) & ~t.suspended)[0][::max(1, k(every_s))]
    rows = []
    for s in dec:
        if s < k(30):
            continue
        live = np.where(np.isfinite(p[s]))[0]
        if len(live) < 3:
            continue
        rank = np.empty(t.n_runners, int)
        rank[live] = (-p[s, live]).argsort().argsort() + 1
        s30, s120 = min(s + k(30), end), min(s + k(120), end)
        for r in live:
            if not np.isfinite(p[s - k(30), r]):
                continue
            f30 = 100 * np.log(p[s30, r] / p[s, r]) if np.isfinite(p[s30, r]) and s30 > s else np.nan
            f120 = 100 * np.log(p[s120, r] / p[s, r]) if np.isfinite(p[s120, r]) and s120 > s else np.nan
            rows.append((int(r), int(rank[r]), 100 * (p[s, r] - p[s - k(30), r]), f30, f120))
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["runner", "rank", "mom_30", "fwd_30", "fwd_120"])
    d["race"] = os.path.basename(path)
    return d


def static_sample(path: str) -> dict | None:
    """Per-race microstructure numbers (information only)."""
    t = Tape.load(path)
    if t.n_steps < 200:
        return None
    s = book_state(t)
    ok, mid = s["ok"], s["mid"]
    pbm = pb_of(mid)
    spr = (s["l0"] - s["b0"]).astype(float)
    scale = max(float(t.total_matched[-1]), 1.0)
    out = dict(race=os.path.basename(path), runners=int(s["act"][-1].sum()))
    for i, nm in enumerate(PB_NAMES):
        m = ok & (pbm == i)
        out[f"spread_{nm}"] = float(spr[m].mean()) if m.any() else np.nan
        b1 = t.back_size[:, :, 0]
        m2 = m & (b1 > 0)
        out[f"l1_{nm}"] = float(np.median(b1[m2] / scale) * 1e3) if m2.any() else np.nan
    vol = np.zeros(s["b0"].shape)
    ts, tr = t.trade_step.astype(np.int64), t.trade_runner.astype(np.int64)
    mm = ts < s["n"]
    np.add.at(vol, (ts[mm], tr[mm]), t.trade_vol[mm])
    out["p_trade"] = float((vol[ok] > 0).mean()) if ok.any() else np.nan
    fr = np.flatnonzero(s["full"])
    if len(fr) > 50:
        x0, x1 = s["x"][fr[0]], s["x"][fr[-1]]
        d = x1 - x0
        out["open_close_sd"] = float(np.nanstd(d[np.isfinite(d)])) if np.isfinite(d).sum() > 1 else np.nan
        ml = mid[fr[-1]]
        has = s["act"][-1] & (t.bsp > 1) & np.isfinite(ml)
        out["bsp_err_sd"] = float(np.std(np.log(t.bsp[has] / ml[has]))) if has.sum() > 1 else np.nan
    return out


def _pmap(fn, paths, workers):
    if workers <= 1:
        return [fn(p) for p in paths]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, paths, chunksize=4))


def _passive(path):
    try:
        return sample_tape(path, offsets=(1,))
    except Exception:
        return None


def _jack(df: pd.DataFrame, fn, groups: int = 20) -> tuple[float, float]:
    """Statistic fn(df) and its delete-a-group jackknife s.e. over races."""
    est = fn(df)
    races = df["race"].unique()
    if len(races) < 4:
        return est, np.nan
    g = pd.Series(np.arange(len(races)) % min(groups, len(races)), index=np.random.default_rng(0).permutation(races))
    gid = df["race"].map(g).to_numpy()
    G = int(gid.max()) + 1
    vals = np.array([fn(df[gid != j]) for j in range(G)])
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return est, np.nan
    return est, float(np.sqrt((len(vals) - 1) / len(vals) * ((vals - vals.mean()) ** 2).sum()))


def _corr(x, y):
    def f(d):
        a, b = d[x].to_numpy(float), d[y].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else np.nan
    return f


def _mean(col, where=None):
    def f(d):
        d = d if where is None else d[where(d)]
        v = d[col].to_numpy(float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else np.nan
    return f


def collect(paths, workers: int = 1) -> dict:
    """Everything the comparison needs from one set of tapes (real or synthetic)."""
    paths = list(paths)
    sig = [d for d in _pmap(signal_sample, paths, workers) if d is not None]
    pas = [d for d in _pmap(_passive, paths, workers) if d is not None]
    sta = [d for d in _pmap(static_sample, paths, workers) if d is not None]
    return dict(dyn=dyn_stats(paths), sig=pd.concat(sig, ignore_index=True) if sig else pd.DataFrame(),
                pas=add_values(pd.concat(pas, ignore_index=True)) if pas else pd.DataFrame(),
                sta=pd.DataFrame(sta), n=len(paths))


def compare(real: dict, syn: dict) -> pd.DataFrame:
    rows = []

    def add(name, r, s, tol, critical, se_r=np.nan, se_s=np.nan, kind="abs"):
        se = np.sqrt(np.nansum([se_r ** 2 if np.isfinite(se_r) else 0, se_s ** 2 if np.isfinite(se_s) else 0]))
        if kind == "rel":
            lim = tol * abs(r) if np.isfinite(r) else np.nan
        else:
            lim = tol
        lim = max(lim, 2 * se) if np.isfinite(lim) else 2 * se
        ok = bool(np.isfinite(r) and np.isfinite(s) and abs(s - r) <= lim)
        rows.append(dict(stat=name, real=r, synth=s, diff=s - r, se_real=se_r, se_synth=se_s, limit=lim, ok=ok,
                         critical=critical))

    for i, nm in enumerate(TB_NAMES):
        add(f"vol_{nm}", real["dyn"]["vol"][i], syn["dyn"]["vol"][i], 0.25, False, kind="rel")
    add("vr60", real["dyn"]["vr60"], syn["dyn"]["vr60"], 0.10, False)
    add("trade_ic", real["dyn"]["trade_ic"], syn["dyn"]["trade_ic"], 0.03, False)

    rs, ss = real["sig"], syn["sig"]
    if len(rs) and len(ss):
        for y, nm in (("fwd_30", "mom30_ic"), ("fwd_120", "mom30_ic_120")):
            r, se_r = _jack(rs, _corr("mom_30", y))
            s, se_s = _jack(ss, _corr("mom_30", y))
            add(nm, r, s, 0.03, True, se_r, se_s)
        cut = float(rs["mom_30"].quantile(0.10))  # the real bottom-10% drift, applied to both
        cond = lambda d: d["rank"].isin([2, 3]) & (d["mom_30"] <= cut)
        r, se_r = _jack(rs, _mean("fwd_120", cond))
        s, se_s = _jack(ss, _mean("fwd_120", cond))
        add("driftfade_120", r, s, 0.5, True, se_r, se_s)

    rp, sp = real["pas"], syn["pas"]
    if len(rp) and len(sp):
        for side in ("back", "lay"):
            c0 = lambda d, side=side: (d["side"] == side) & (d["k"] == 0)
            r, se_r = _jack(rp, _mean("ev_fill", c0))
            s, se_s = _jack(sp, _mean("ev_fill", c0))
            add(f"cross_ev_{side}", r, s, 0.5, True, se_r, se_s)
            c1 = lambda d, side=side: (d["side"] == side) & (d["k"] == 1)

            def adverse(d, c1=c1):
                g = d[c1(d)]
                f = g["f60"] > 0
                return float(g.loc[f, "ev_fill"].mean() - g["ev_fill"].mean()) if f.sum() > 5 else np.nan

            r, se_r = _jack(rp, adverse)
            s, se_s = _jack(sp, adverse)
            add(f"adverse_{side}", r, s, 0.5, True, se_r, se_s)
            r, se_r = _jack(rp, _mean("f60", c1))
            s, se_s = _jack(sp, _mean("f60", c1))
            add(f"fill60_{side}", r, s, 0.3, True, se_r, se_s, kind="rel")

    ra, sa = real["sta"], syn["sta"]
    if len(ra) and len(sa):
        for c in [c for c in ra.columns if c != "race"]:
            if c in sa:
                # medians over races: a few thin races make per-race ratios (l1 size / volume) heavy-tailed
                add(c, float(ra[c].median()), float(sa[c].median()), 0.25, False,
                    float(1.25 * ra[c].std() / np.sqrt(ra[c].notna().sum())),
                    float(1.25 * sa[c].std() / np.sqrt(sa[c].notna().sum())), kind="rel")
    return pd.DataFrame(rows)


def parity(real_paths, synth_paths, workers: int = 1, out: str | None = None) -> tuple[pd.DataFrame, bool]:
    real, syn = collect(real_paths, workers), collect(synth_paths, workers)
    df = compare(real, syn)
    ok = bool(df.loc[df["critical"], "ok"].all()) if df["critical"].any() else False
    if out:
        os.makedirs(out, exist_ok=True)
        df.to_csv(os.path.join(out, "parity.csv"), index=False)
        json.dump(dict(parity_ok=ok, failed=df.loc[~df["ok"], "stat"].tolist(), n_real=real["n"], n_synth=syn["n"]),
                  open(os.path.join(out, "parity.json"), "w"), indent=1)
    return df, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True)
    ap.add_argument("--synth", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    rp, sp = list_tapes(a.real), list_tapes(a.synth)
    rp = sorted(rng.choice(rp, min(a.max, len(rp)), replace=False).tolist())
    sp = sp[: a.max]
    df, ok = parity(rp, sp, a.workers, a.out)
    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print("PARITY OK" if ok else "PARITY FAILED on: " + ", ".join(df.loc[df["critical"] & ~df["ok"], "stat"]))


if __name__ == "__main__":
    main()
