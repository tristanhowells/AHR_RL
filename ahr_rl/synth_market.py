"""P20 generator: synthetic pre-off WIN markets fitted on real TRAIN tapes, produced as our Tape.

Adapted from betfair-rl/tape_synth.py (fit / generate / calibrate). It keeps tape_synth's empirical market model:
templates (field size, opening book, liquidity, timing), block-bootstrapped real price moves, real spread / level
gap / depth / size (AR(1) copula) / trade-rate / trade-size / projected-SP distributions, scratchings with
suspension and reduction factors. Three changes, so an agent trained here cannot learn something the real market
does not do:

  1. NO BRIDGE TO THE CLOSE. tape_synth moves every runner a step closer to its known closing price each row
     ((x_close - x) / rows left): a steady trend an agent can learn (momentum), the opposite of the reversal P16
     measured. Here the displayed price is
         x = f + u
         f  latent fair value: a random walk of block-bootstrapped real moves (detrended) plus a per-runner
            drift (steaming / drifting spells) drawn from the real distribution of whole-race drifts and
            renewed with half-life drift_half_life_s, no target price, renormalised
         u  a transient deviation that decays back to f (OU, half-life rev_half_life_s), in units of the runner's
            own recent move size
     The MEAN drift for each price band is kept at full real strength: zero-mean moves in log probability
     would make the probability itself drift up (Jensen), so every runner would shorten on average and backing
     early would pay (a fake edge P20's parity check caught: crossing to back and greening at BSP returned
     +11% per $ on synthetic races vs -8% on real ones). Only the spells' spread around it (drift_mult) is
     calibrated, so the 30s momentum IC (own last 30s move vs next 30s, in
     points of implied probability) matches real: real markets do trend a little, and that weak momentum is
     kept, not planted or removed. The 120s IC and open-to-close dispersion are checked, not fitted, in parity.
     sigma_u is calibrated so the 60s variance ratio (60s variance / 120 x 0.5s variance) matches real. The book only re-centres once the
     latent price has moved tick_hysteresis ticks from where it was last shown: real books are sticky, and a
     continuous price re-rounded every row would flip between neighbouring ticks (a fake bounce). For the
     same reason a spread change of odd size (which moves the mid half a tick) only happens on a row where
     the book re-centres anyway.
  2. INFORMED TRADES. tape_synth draws each trade's side at random, so a resting order's fills carry no news (no
     adverse selection - P15's main finding). Here a trade lands on the side the latent price is about to move to
     (backers take the best back price just before it shortens) with strength beta, calibrated so the correlation
     between a trade's side and the next 10s move matches real.
  3. SETTLEMENT FROM BSP. BSP = the last book mid x the real (BSP / last mid) error for that price band; the winner
     is drawn from the BSP-implied probabilities with the real win-rate-vs-BSP calibration (longshot bias). Late
     money is then as informative as it is in real races, and BSP exits are priced as in real races.

Model inputs are TRAIN tapes only; VAL / TEST races are never seen. The volatility per time-to-jump bucket, the
variance ratio and the trade-side information are moment-matched by `calibrate`.

    python -m ahr_rl.synth_market fit --tapes "data/tapes/*.npz" --model runs/synth/model.pkl
    python -m ahr_rl.synth_market generate --model runs/synth/model.pkl --out runs/synth/pool --n 200
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from .env import list_tapes
from .ladder import N_TICKS, PRICES
from .tape import Tape

LOGP = np.log(PRICES)
TICKS = np.arange(N_TICKS, dtype=np.float64)

TB_EDGES = np.array([-480.0, -300.0, -180.0, -90.0, -45.0, -15.0])  # secs to the jump (end of tape)
PB_EDGES = np.array([2.0, 3.0, 5.0, 10.0, 20.0, 50.0])  # mid price
NTB, NPB = len(TB_EDGES) + 1, len(PB_EDGES) + 1
NLEV = 8  # ladder levels per side in the tapes
BLOCK = 20  # bootstrap block length (rows)
MAXS, MAXG = 40, 15  # spread / level-gap caps (ticks)
NQ = 101
QS = np.linspace(0.0, 1.0, NQ)
DT = 0.5
INFO_H = 20  # rows a trade "knows" ahead (10s), also the horizon of the trade-side statistic
RET_H = 120  # rows of the variance-ratio return (60s)
MOM_H = 60  # rows of the momentum statistic's look-back and look-ahead (30s)
CAL_EDGES = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5])  # implied-probability bins for the win calibration
NCAL = len(CAL_EDGES) + 1
BUF = 3 * BLOCK + INFO_H

DEFAULT_GEN = dict(open_noise=0.15, liq_jitter=0.3, vol_jitter=0.2, shuffle=True, gap_resample=0.1,
                   depth_resample=0.05, rev_half_life_s=5.0, tick_hysteresis=0.6,
                   drift_half_life_s=60.0)


def tb_of(tj):
    return np.digitize(tj, TB_EDGES)


def pb_of(price):
    return np.digitize(np.nan_to_num(price, nan=1000.0), PB_EDGES)


def norm_cdf(z):
    # Abramowitz-Stegun 7.1.26 erf approximation (|err| < 1.5e-7)
    x = np.abs(z) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * np.exp(-x * x)
    return 0.5 * (1.0 + np.sign(z) * y)


def _as_tape(t) -> Tape:
    return t if isinstance(t, Tape) else Tape.load(t)


def _lognorm(x):
    x = np.asarray(x, np.float64)
    fin = np.isfinite(x)
    if not fin.any():
        return x
    m = x[fin].max()
    return np.where(fin, x - (m + np.log(np.exp(x[fin] - m).sum())), -np.inf)


def book_state(t: Tape) -> dict:
    """Per-tape arrays shared by fitting, calibration and parity: book mid, normalised log implied prob x."""
    tt = t.t_rel.astype(np.float64)
    tj = tt - tt[-1]  # tapes end when the market turns in-play
    susp, act = t.suspended.astype(bool), t.active.astype(bool)
    bt, lt = t.back_tick.astype(np.int32), t.lay_tick.astype(np.int32)
    b0, l0 = bt[:, :, 0], lt[:, :, 0]
    ok = (b0 >= 0) & (l0 > b0) & act & ~susp[:, None]
    mid = np.full(b0.shape, np.nan)
    mid[ok] = np.sqrt(PRICES[b0[ok]] * PRICES[l0[ok]])
    full = (ok | ~act).all(1) & ~susp & act.any(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = 1.0 / mid
        ovr = np.nansum(q, 1)
        x = np.log(np.where(full[:, None] & ok, q / ovr[:, None], np.nan))
    return dict(t=tt, n=len(tt), tj=tj, susp=susp, act=act, bt=bt, lt=lt, b0=b0, l0=l0, ok=ok, mid=mid,
                full=full, ovr=ovr, x=x)


# ============================================================================ fitting
def _extract(args):
    path, seed, rows_per_tape = args
    try:
        z = _as_tape(path)
    except Exception:
        return None
    if z.n_steps < 200 or float(z.total_matched[-1]) <= 0:
        return None
    rng = np.random.default_rng(seed)
    s = book_state(z)
    n, R = s["b0"].shape
    ok, tj, mid, x = s["ok"], s["tj"], s["mid"], s["x"]
    scale = float(z.total_matched[-1])
    tb = tb_of(tj)
    tb2 = np.broadcast_to(tb[:, None], (n, R))
    pbm = pb_of(mid)
    out: dict = {"name": z.name}

    # detrended log-prob increment blocks
    dx = np.diff(x, axis=0)
    blocks, bkeys, dkeys, dvals = [], [], [], []
    for i in range(R):
        d = dx[:, i]
        g = np.isfinite(d)
        if g.sum() < 2 * BLOCK:
            continue
        first = int(np.flatnonzero(g)[0])
        dkeys.append(pbm[first, i])
        dvals.append(d[g].mean())
        d = d - d[g].mean()
        for st in range(0, n - 1 - BLOCK, BLOCK):
            seg = d[st:st + BLOCK]
            if np.isfinite(seg).all():
                blocks.append(seg.astype(np.float32))
                bkeys.append(tb[st] * NPB + pbm[st, i])
    out["blocks"] = np.array(blocks, np.float32).reshape(-1, BLOCK)
    out["bkeys"] = np.array(bkeys, np.int32)
    out["dkeys"] = np.array(dkeys, np.int32)
    out["dvals"] = np.array(dvals, np.float32)

    # spread
    spr = s["l0"] - s["b0"]
    sc = np.clip(spr[ok], 1, MAXS)
    out["spread_cnt"] = np.bincount((pbm[ok] * NTB + tb2[ok]) * (MAXS + 1) + sc, minlength=NPB * NTB * (MAXS + 1))
    ok2 = ok[1:] & ok[:-1]
    chg = (spr[1:] != spr[:-1]) & ok2
    ck = pbm[1:][ok2] * NTB + tb2[1:][ok2]
    out["spread_chg"] = np.bincount(ck, weights=chg[ok2], minlength=NPB * NTB)
    out["spread_chg_n"] = np.bincount(ck, minlength=NPB * NTB).astype(float)

    # ladder gaps and depth
    gap_cnt = np.zeros((2, NPB, MAXG + 1))
    nlev_cnt = np.zeros((2, NPB, NLEV + 1))
    for side, tk in ((0, s["bt"]), (1, s["lt"])):
        for k in range(min(NLEV, tk.shape[2]) - 1):
            g = tk[:, :, k] - tk[:, :, k + 1] if side == 0 else tk[:, :, k + 1] - tk[:, :, k]
            v = ok & (tk[:, :, k + 1] >= 0) & (g > 0)
            np.add.at(gap_cnt[side], (pbm[v], np.clip(g[v], 1, MAXG)), 1)
        nl = (tk >= 0).sum(2)
        np.add.at(nlev_cnt[side], (pbm[ok], np.minimum(nl[ok], NLEV)), 1)
    out["gap_cnt"], out["nlev_cnt"] = gap_cnt, nlev_cnt

    # sizes (sub-sampled rows), relative to the market's final matched volume
    rr = np.sort(rng.choice(n, min(rows_per_tape, n), replace=False))
    skeys, svals = [], []
    for side, tk, sz in ((0, s["bt"], z.back_size), (1, s["lt"], z.lay_size)):
        for k in range(min(NLEV, tk.shape[2])):
            v = ok[rr] & (tk[rr, :, k] >= 0) & (sz[rr, :, k] > 0)
            skeys.append((((side * NLEV + k) * NPB + pbm[rr][v]) * NTB + tb2[rr][v]))
            svals.append(np.log(sz[rr, :, k][v] / scale))
    out["skeys"] = np.concatenate(skeys).astype(np.int32)
    out["svals"] = np.concatenate(svals).astype(np.float32)
    ls0 = np.log(np.maximum(z.back_size[:, :, 0], 1e-6) / scale)
    v = ok[1:] & ok[:-1] & (z.back_size[1:, :, 0] > 0) & (z.back_size[:-1, :, 0] > 0)
    a, b = ls0[:-1][v], ls0[1:][v]
    out["size_ar"] = np.array([len(a), a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum()])

    # trades
    ts, tr = z.trade_step.astype(np.int64), z.trade_runner.astype(np.int64)
    tt, tvol = z.trade_tick.astype(np.int64), z.trade_vol.astype(np.float64)
    m = (ts < n) & (tvol > 0) & (tr < R)
    vol = np.zeros((n, R))
    np.add.at(vol, (ts[m], tr[m]), tvol[m])
    ev = (vol > 0) & ok
    key = tb2 * NPB + pbm
    out["trade_den"] = np.bincount(key[ok], minlength=NTB * NPB).astype(float)
    out["trade_num"] = np.bincount(key[ev], minlength=NTB * NPB).astype(float)
    out["tvkeys"] = key[ev].astype(np.int32)
    out["tvvals"] = np.log(vol[ev] / scale).astype(np.float32)
    mo = m.copy()
    mo[m] = ok[ts[m], tr[m]]
    out["at_back"] = np.array([(tt[mo] <= s["b0"][ts[mo], tr[mo]]).sum(), mo.sum()], float)

    # projected SP noise and book overround
    sp = z.spn.astype(np.float64)
    okS = ok & (sp > 1.0) & (sp <= 1000.0)
    e = np.where(okS, np.log(np.where(okS, sp, 1.0) / np.where(okS, mid, 1.0)), np.nan)
    sv = okS[rr]
    out["spnkeys"] = tb2[rr][sv].astype(np.int32)
    out["spnvals"] = e[rr][sv].astype(np.float32)
    v = okS[1:] & okS[:-1]
    a, b = e[:-1][v], e[1:][v]
    out["spn_ar"] = np.array([len(a), a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum()])
    f = s["full"]
    out["ovr_sum"] = np.bincount(tb[f], weights=np.log(s["ovr"][f]), minlength=NTB)
    out["ovr_n"] = np.bincount(tb[f], minlength=NTB).astype(float)

    # settlement: BSP vs the last book mid, and the win rate vs BSP-implied probability
    out["cal_w"], out["cal_e"] = np.zeros(NCAL), np.zeros(NCAL)
    out["bspkeys"], out["bspvals"] = np.zeros(0, np.int32), np.zeros(0, np.float32)
    fr = np.flatnonzero(f)
    bsp = z.bsp.astype(np.float64)
    if len(fr):
        ml = mid[fr[-1]]
        has = s["act"][-1] & (bsp > 1.0) & np.isfinite(ml)
        if has.any():
            out["bspkeys"] = pb_of(ml[has]).astype(np.int32)
            out["bspvals"] = np.log(bsp[has] / ml[has]).astype(np.float32)
        w = int(z.winner)
        allb = s["act"][-1] & (bsp > 1.0)
        if 0 <= w < R and allb.sum() >= 2 and allb[w]:
            qb = np.where(allb, 1.0 / np.where(allb, bsp, 1.0), 0.0)
            qb /= qb.sum()
            y = np.zeros(R)
            y[w] = 1
            cb = np.digitize(qb[allb], CAL_EDGES)
            out["cal_w"] = np.bincount(cb, weights=y[allb], minlength=NCAL)
            out["cal_e"] = np.bincount(cb, weights=qb[allb], minlength=NCAL)

    # removals
    out["removals"] = [(float(tj[min(r_, n - 1)]), int(s["susp"][r_:r_ + 40].sum()) if r_ < n else 2)
                       for r_ in z.removal_step.astype(np.int64)]

    # template
    if len(fr) >= 50:
        f0 = fr[0]
        keep = s["act"][fr[-1]] & np.isfinite(x[f0])
        if keep.sum() >= 2:
            out["template"] = dict(name=z.name, n=int(keep.sum()), x_open=_lognorm(x[f0, keep]),
                                   tv0=z.tv[0].astype(np.float64)[keep] / scale, scale=scale,
                                   t0=float(s["t"][0]), jump_t=float(s["t"][-1]), base_rate=float(z.base_rate))
    return out


def _quantiles_by_key(keys, vals, nkeys, min_n):
    Q = np.full((nkeys, NQ), np.nan)
    if len(keys) == 0:
        return Q
    order = np.argsort(keys, kind="stable")
    ks, vs = keys[order], vals[order]
    bounds = np.searchsorted(ks, np.arange(nkeys + 1))
    for k in range(nkeys):
        a, b = bounds[k], bounds[k + 1]
        if b - a >= min_n:
            Q[k] = np.quantile(vs[a:b], QS)
    return Q


def _fill(Q, *coarse):
    """Fill NaN rows of Q from successively coarser (table, key_fn) pairs."""
    for table, fn in coarse:
        miss = np.isnan(Q[:, 0])
        if not miss.any():
            break
        idx = np.flatnonzero(miss)
        Q[idx] = table[fn(idx)]
    return Q


def _probs(cnt, min_n=50):
    """Normalise counts along the last axis; sparse rows fall back to the pooled row."""
    cnt = np.asarray(cnt, float)
    flat = cnt.reshape(-1, cnt.shape[-1])
    pooled = flat.sum(0) + 1e-9
    tot = flat.sum(1, keepdims=True)
    P = np.where(tot >= min_n, flat / np.maximum(tot, 1e-9), pooled / pooled.sum())
    return P.reshape(cnt.shape)


def _ar_corr(acc):
    n, sa, sb, saa, sbb, sab = acc
    if n < 10:
        return 0.9
    cov = sab / n - (sa / n) * (sb / n)
    va, vb = saa / n - (sa / n) ** 2, sbb / n - (sb / n) ** 2
    return float(np.clip(cov / np.sqrt(max(va * vb, 1e-12)), 0.0, 0.999))


def _global_q(vals, fallback):
    return np.quantile(vals, QS)[None] if len(vals) else np.full((1, NQ), fallback)


def fit(paths, model_path: str, rows_per_tape: int = 150, max_blocks: int = 6000, threads: int = 8, seed: int = 0,
        calibrate_iters: int = 4, calib_markets: int = 64) -> dict:
    """Fit on `paths` (TRAIN tapes only), calibrate, save to model_path."""
    paths = list(paths)
    if not paths:
        raise SystemExit("no tapes to fit on")
    print(f"[synth] fitting on {len(paths)} TRAIN tapes", flush=True)
    t_start = time.time()
    parts = []
    with ThreadPoolExecutor(threads) as ex:
        for i, r in enumerate(ex.map(_extract, [(p, seed + i, rows_per_tape) for i, p in enumerate(paths)])):
            if r is not None:
                parts.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(paths)} tapes ({time.time() - t_start:.0f}s)", flush=True)
    if not parts:
        raise SystemExit("no usable tapes")
    rng = np.random.default_rng(seed)
    cat = lambda k: np.concatenate([p[k] for p in parts])
    tot = lambda k: np.sum([p[k] for p in parts], axis=0)

    blocks, bkeys = cat("blocks"), cat("bkeys")
    pools, pools_tb = {}, {}
    for key in np.unique(bkeys):
        idx = np.flatnonzero(bkeys == key)
        if len(idx) > max_blocks:
            idx = rng.choice(idx, max_blocks, replace=False)
        if len(idx) >= 30:
            pools[int(key)] = blocks[idx]
    for tbk in range(NTB):
        idx = np.flatnonzero(bkeys // NPB == tbk)
        if len(idx):
            pools_tb[tbk] = blocks[rng.choice(idx, min(len(idx), max_blocks), replace=False)]
    pool_all = blocks[rng.choice(len(blocks), min(len(blocks), max_blocks), replace=False)]

    skeys, svals = cat("skeys"), cat("svals")
    Qs = _quantiles_by_key(skeys, svals, 2 * NLEV * NPB * NTB, 200)
    Qs_c = _quantiles_by_key(skeys // NTB, svals, 2 * NLEV * NPB, 200)
    Qs_cc = _quantiles_by_key(skeys // (NTB * NPB), svals, 2 * NLEV, 50)
    Qs_cc = _fill(Qs_cc, (Qs_cc, lambda i: (i // NLEV) * NLEV), (_global_q(svals, np.log(1e-3)), lambda i: i * 0))
    Qs_c = _fill(Qs_c, (Qs_cc, lambda i: i // NPB))
    Qs = _fill(Qs, (Qs_c, lambda i: i // NTB)).reshape(2, NLEV, NPB, NTB, NQ)

    tvk, tvv = cat("tvkeys"), cat("tvvals")
    Qv_pb = _fill(_quantiles_by_key(tvk % NPB, tvv, NPB, 100), (_global_q(tvv, np.log(1e-4)), lambda i: i * 0))
    Qv = _fill(_quantiles_by_key(tvk, tvv, NTB * NPB, 100), (Qv_pb, lambda i: i % NPB))
    tden, tnum = tot("trade_den"), tot("trade_num")
    p_trade = np.where(tden > 200, tnum / np.maximum(tden, 1), tnum.sum() / max(tden.sum(), 1))

    spk, spv = cat("spnkeys"), cat("spnvals")
    Qspn = _fill(_quantiles_by_key(spk, spv, NTB, 200), (_global_q(spv, 0.0), lambda i: i * 0))
    dk, dv = cat("dkeys"), cat("dvals")
    Qdrift = _fill(_quantiles_by_key(dk, dv, NPB, 30), (_global_q(dv, 0.0), lambda i: i * 0))
    cnt = np.bincount(dk, minlength=NPB)
    drift_mu = np.where(cnt >= 30, np.bincount(dk, weights=dv, minlength=NPB) / np.maximum(cnt, 1),
                        float(dv.mean()) if len(dv) else 0.0)
    bk, bv = cat("bspkeys"), cat("bspvals")
    Qbsp = _fill(_quantiles_by_key(bk, bv, NPB, 50), (_global_q(bv, 0.0), lambda i: i * 0))
    cal_w, cal_e = tot("cal_w"), tot("cal_e")
    win_cal = (cal_w + 10.0) / (cal_e + 10.0)  # shrunk towards 1 (prior = 10 wins per bin)

    removals = [r for p in parts for r in p["removals"]]
    n_rem = [len(p["removals"]) for p in parts]
    at_back = tot("at_back")
    model = dict(
        version=2, n_source=len(parts), source=[p["name"] for p in parts],
        templates=[p["template"] for p in parts if "template" in p],
        pools=pools, pools_tb=pools_tb, pool_all=pool_all,
        spread_p=_probs(tot("spread_cnt").reshape(NPB, NTB, MAXS + 1)),
        spread_change=(tot("spread_chg") / np.maximum(tot("spread_chg_n"), 1)).reshape(NPB, NTB),
        gap_p=_probs(tot("gap_cnt")), nlev_p=_probs(tot("nlev_cnt")),
        size_q=Qs.astype(np.float32), size_rho=_ar_corr(tot("size_ar")),
        p_trade=p_trade.reshape(NTB, NPB), tvol_q=Qv.reshape(NTB, NPB, NQ).astype(np.float32),
        frac_at_back=float(at_back[0] / max(at_back[1], 1)),
        spn_q=Qspn.astype(np.float32), spn_rho=_ar_corr(tot("spn_ar")),
        drift_q=Qdrift.astype(np.float32), drift_mu=drift_mu, bsp_q=Qbsp.astype(np.float32), n_bsp=int(len(bv)), win_cal=win_cal,
        log_ovr=tot("ovr_sum") / np.maximum(tot("ovr_n"), 1),
        p_removal=float(np.mean(np.array(n_rem) > 0)),
        removal_counts=[c for c in n_rem if c > 0] or [1],
        removal_tj=[r[0] for r in removals] or [-120.0],
        removal_susp=[max(r[1], 1) for r in removals] or [4],
        # calibrated below
        inc_scale=np.ones(NTB), sigma_u=0.1, beta=0.3, drift_mult=0.5,
    )
    if not model["templates"]:
        raise SystemExit("no usable templates (need tapes with >= 50 fully quoted rows)")
    save(model, model_path)
    print(f"[synth] {len(model['templates'])} templates, {len(pools)} increment buckets, {len(bv)} BSP errors "
          f"({time.time() - t_start:.0f}s)", flush=True)
    print(f"  win rate / BSP-implied by bin: {np.round(win_cal, 2)}", flush=True)
    if calibrate_iters:
        calibrate(model, model_path, paths, calibrate_iters, n_markets=calib_markets, seed=seed)
    return model


def save(model, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load(path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ============================================================================ calibration statistics
def dyn_stats(tapes) -> dict:
    """The calibration targets, same function on real and synthetic tapes:
    vol      rms 0.5s change of x per time-to-jump bucket
    vr60     60s variance / (120 x 0.5s variance): < 1 = moves partly revert
    trade_ic corr(trade side: +1 at the best back price / -1 at the best lay, next 10s change of x)
    mom_ic   corr(last 60 rows' change of p = exp(x), next 60 rows' change), every 40 rows of the last 10 min
    oc_sd    sd over runners of x at the last fully quoted row - x at the first"""
    dx2 = np.zeros((NTB, 2))
    r2, n2, d2, nd = 0.0, 0, 0.0, 0
    sx, sy, sxx, syy, sxy, sn = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    ma, mb, oc = [], [], []
    for t in tapes:
        t = _as_tape(t)
        if t.n_steps < 200:
            continue
        s = book_state(t)
        x = s["x"]
        n, R = x.shape
        tb = tb_of(s["tj"])
        d = np.diff(x, axis=0)
        fin = np.isfinite(d)
        np.add.at(dx2, (np.broadcast_to(tb[1:, None], d.shape)[fin], 0), d[fin] ** 2)
        np.add.at(dx2, (np.broadcast_to(tb[1:, None], d.shape)[fin], 1), 1)
        d2 += float((d[fin] ** 2).sum())
        nd += int(fin.sum())
        if n > RET_H:
            r = x[RET_H:] - x[:-RET_H]
            fr = np.isfinite(r)
            r2 += float((r[fr] ** 2).sum())
            n2 += int(fr.sum())
        fr = np.flatnonzero(s["full"])
        if len(fr) > 50:
            d_ = x[fr[-1]] - x[fr[0]]
            oc.append(d_[np.isfinite(d_)])
        if n > 2 * MOM_H:
            gs = np.arange(max(MOM_H, n - 1200), n - MOM_H, 40)
            px = np.exp(x)
            a_, b_ = px[gs] - px[gs - MOM_H], px[gs + MOM_H] - px[gs]
            fm = np.isfinite(a_) & np.isfinite(b_)
            ma.append(a_[fm])
            mb.append(b_[fm])
        ts, tr, tk = t.trade_step.astype(np.int64), t.trade_runner.astype(np.int64), t.trade_tick.astype(np.int64)
        m = (ts >= 1) & (ts + INFO_H < n) & (tr < R)
        ts, tr, tk = ts[m], tr[m], tk[m]
        b0, l0 = s["b0"][ts - 1, tr], s["l0"][ts - 1, tr]
        sg = np.where(tk <= b0, 1.0, np.where(tk >= l0, -1.0, 0.0))
        fy = x[ts + INFO_H, tr] - x[ts, tr]
        g = (sg != 0) & np.isfinite(fy) & (b0 >= 0) & (l0 > b0)
        a, b = sg[g], fy[g]
        sx += a.sum(); sy += b.sum(); sxx += (a * a).sum(); syy += (b * b).sum(); sxy += (a * b).sum(); sn += len(a)
    vol = np.sqrt(dx2[:, 0] / np.maximum(dx2[:, 1], 1))
    vr = (r2 / max(n2, 1)) / max(RET_H * d2 / max(nd, 1), 1e-18)
    if sn > 10:
        cov = sxy / sn - (sx / sn) * (sy / sn)
        ic = cov / np.sqrt(max((sxx / sn - (sx / sn) ** 2) * (syy / sn - (sy / sn) ** 2), 1e-18))
    else:
        ic = 0.0
    ma, mb = (np.concatenate(ma), np.concatenate(mb)) if ma else (np.zeros(0), np.zeros(0))
    mom = float(np.corrcoef(ma, mb)[0, 1]) if len(ma) > 10 else 0.0
    oc_sd = float(np.std(np.concatenate(oc))) if oc else 0.0
    return dict(vol=vol, vr60=float(vr), trade_ic=float(ic), mom_ic=mom, oc_sd=oc_sd, n_trades=int(sn))


def _secant(hist, target, lo, hi, increasing: bool) -> float:
    """Next parameter value from (param, statistic) history: a secant step on the last two points when their
    slope has the expected sign, else a damped multiplicative step; clipped to [lo, hi]."""
    p1, s1 = hist[-1]
    step = None
    if len(hist) >= 2:
        p0, s0 = hist[-2]
        if abs(p1 - p0) > 1e-6:
            slope = (s1 - s0) / (p1 - p0)
            if (slope > 0) == increasing and abs(slope) > 1e-9:
                step = 0.8 * (target - s1) / slope
    if step is None:  # one point or a noisy slope: move by a factor towards the target
        up = (target > s1) == increasing
        step = (max(p1, 0.05) * (0.6 if up else -0.4))
    base = max(p1, 0.05)
    step = float(np.clip(step, -0.5 * base, base))  # at most halve or double per iteration (noisy estimates)
    return float(np.clip(p1 + step, lo, hi))


def _calib_loss(got, tgt) -> float:
    v = np.log(np.maximum(got["vol"], 1e-9) / np.maximum(tgt["vol"], 1e-9))
    return float(np.mean(v ** 2) / 0.1 ** 2 + ((got["vr60"] - tgt["vr60"]) / 0.05) ** 2
                 + ((got["trade_ic"] - max(tgt["trade_ic"], 0.0)) / 0.015) ** 2
                 + ((got["mom_ic"] - tgt["mom_ic"]) / 0.015) ** 2)


def calibrate(model, model_path, paths, iters: int = 4, n_markets: int = 64, seed: int = 0, n_ref: int = 150):
    """Moment-match per-bucket 0.5s vol (inc_scale), the 60s variance ratio (sigma_u) and the trade-side
    information (beta) to the real TRAIN tapes."""
    rng = np.random.default_rng(seed)
    ref = list(paths) if len(paths) <= n_ref else rng.choice(paths, n_ref, replace=False).tolist()
    tgt = dyn_stats(ref)
    print(f"[synth] calibration targets from {len(ref)} TRAIN tapes: vol {np.round(tgt['vol'], 4)} "
          f"vr60 {tgt['vr60']:.3f} trade_ic {tgt['trade_ic']:+.3f} mom_ic {tgt['mom_ic']:+.3f} oc_sd {tgt['oc_sd']:.3f}", flush=True)
    hist = {"sigma_u": [], "beta": [], "drift_mult": []}
    keys = ("inc_scale", "sigma_u", "beta", "drift_mult")
    best = (np.inf, None)
    for it in range(iters):
        g = np.random.default_rng([seed, 99, it])
        got = dyn_stats([generate_market(model, g, f"CAL_{it}_{j}") for j in range(n_markets)])
        loss = _calib_loss(got, tgt)
        if loss < best[0]:
            best = (loss, {k: np.copy(model[k]) if k == "inc_scale" else model[k] for k in keys})
        ratio = np.where(got["vol"] > 0, tgt["vol"] / np.maximum(got["vol"], 1e-12), 1.0)
        # quantisation noise doesn't scale with the latent move, so over-correct a little
        model["inc_scale"] = np.clip(model["inc_scale"] * ratio ** 1.5, 0.05, 5.0)
        hist["sigma_u"].append((model["sigma_u"], got["vr60"]))
        hist["beta"].append((model["beta"], got["trade_ic"]))
        hist["drift_mult"].append((model["drift_mult"], got["mom_ic"]))
        # more transient deviation -> lower variance ratio; more informed trades -> higher trade-side IC
        model["sigma_u"] = _secant(hist["sigma_u"], tgt["vr60"], 0.0, 3.0, increasing=False)
        model["beta"] = _secant(hist["beta"], max(tgt["trade_ic"], 0.0), 0.0, 8.0, increasing=True)
        model["drift_mult"] = _secant(hist["drift_mult"], tgt["mom_ic"], 0.0, 6.0, increasing=True)
        save(model, model_path)
        print(f"  iter {it + 1} (loss {loss:.1f}): synth vol {np.round(got['vol'], 4)} vr60 {got['vr60']:.3f} trade_ic "
              f"{got['trade_ic']:+.3f} mom_ic {got['mom_ic']:+.3f} oc_sd {got['oc_sd']:.3f} -> inc_scale {np.round(model['inc_scale'], 2)} "
              f"sigma_u {model['sigma_u']:.3f} beta {model['beta']:.2f} drift_mult {model['drift_mult']:.2f}", flush=True)
    if best[1] is not None:  # the parameters that matched best (estimates are noisy, the last need not be best)
        model.update(best[1])
        print(f"[synth] calibration: keeping the best iteration (loss {best[0]:.1f}): inc_scale "
              f"{np.round(model['inc_scale'], 2)} sigma_u {model['sigma_u']:.3f} beta {model['beta']:.2f} "
              f"drift_mult {model['drift_mult']:.2f}", flush=True)
    model["calibration_target"] = tgt
    save(model, model_path)
    return model


# ============================================================================ generation
def _book_probs_row(bt, lt, act):
    b0, l0 = bt[:, 0].astype(np.int64), lt[:, 0].astype(np.int64)
    ok = act & (b0 >= 0) & (l0 > b0)
    if not ok.any():
        return None
    q = np.zeros(len(act))
    q[ok] = 1.0 / np.sqrt(PRICES[b0[ok]] * PRICES[l0[ok]])
    return q


def _sample_cat(P, rng):
    """Row-wise categorical sampling; P is (m, K)."""
    c = P.cumsum(1)
    u = rng.random(len(P))[:, None] * c[:, -1:]
    return (c > u).argmax(1)


def _q_lookup(Q, u):
    """Q is (..., NQ); u in [0, 1] with the same leading shape."""
    pos = u * (NQ - 1)
    lo = np.clip(np.floor(pos).astype(np.int64), 0, NQ - 2)
    fr = pos - lo
    a = np.take_along_axis(Q, lo[..., None], -1)[..., 0]
    b = np.take_along_axis(Q, lo[..., None] + 1, -1)[..., 0]
    return a + (b - a) * fr


def _draw_block(M, rng, tb, pb):
    pool = M["pools"].get(int(tb * NPB + pb))
    if pool is None:
        pool = M["pools_tb"].get(int(tb), M["pool_all"])
    return pool[rng.integers(len(pool))]


def generate_market(M: dict, rng: np.random.Generator, name: str, gen: dict | None = None,
                    template: dict | None = None) -> Tape:
    g = {**DEFAULT_GEN, **(gen or {})}
    tpl = template or M["templates"][rng.integers(len(M["templates"]))]
    n = tpl["n"]
    perm = rng.permutation(n) if g["shuffle"] else np.arange(n)
    f = _lognorm(tpl["x_open"][perm] + rng.normal(0, g["open_noise"], n))  # latent fair value
    u = np.zeros(n)  # transient deviation
    t0, jump = tpl["t0"], tpl["jump_t"]
    T = int(np.floor((jump - t0) / DT)) + 1
    t = t0 + DT * np.arange(T)
    tj = t - t[-1]
    scale = tpl["scale"] * np.exp(rng.normal(0, g["liq_jitter"]))
    volmult = np.exp(rng.normal(0, g["vol_jitter"]))
    phi = 0.5 ** (DT / max(g["rev_half_life_s"], DT))
    sig_u, beta = float(M.get("sigma_u", 0.0)), float(M.get("beta", 0.0))
    dmult = float(M.get("drift_mult", 0.0)) if "drift_q" in M else 0.0
    mu_pb = M.get("drift_mu", np.zeros(NPB))
    # a spell = the deviation of a real whole-race drift from its price band's mean, scaled by drift_mult
    draw_drift = lambda pb_: (_q_lookup(M["drift_q"][pb_], rng.random(len(pb_))) - mu_pb[pb_]) * dmult
    drift = draw_drift(pb_of(np.exp(-f))) if dmult > 0 else np.zeros(n)
    phi_d = 0.5 ** (DT / max(g["drift_half_life_s"], DT))
    fab = np.clip(M["frac_at_back"], 0.02, 0.98)
    logit_fab = np.log(fab / (1 - fab))

    rem_at: set[int] = set()
    susp_until = -1
    if n > 4 and rng.random() < M["p_removal"]:
        for _ in range(min(int(rng.choice(M["removal_counts"])), n - 3)):
            st = int(np.clip(np.searchsorted(tj, rng.choice(M["removal_tj"])), 5, T - 10))
            while st in rem_at:
                st += 1
            rem_at.add(st)

    back_tick = np.full((T, n, NLEV), -1, np.int16)
    lay_tick = np.full((T, n, NLEV), -1, np.int16)
    back_size = np.zeros((T, n, NLEV), np.float32)
    lay_size = np.zeros((T, n, NLEV), np.float32)
    ltp_tick = np.full((T, n), -1, np.int16)
    tv_out = np.zeros((T, n), np.float32)
    spn_out = np.zeros((T, n), np.float32)
    active = np.ones((T, n), bool)
    suspended = np.zeros(T, bool)
    total_matched = np.zeros(T, np.float32)
    tr_step, tr_run, tr_tick, tr_vol = [], [], [], []
    rem_step, rem_run, rem_fac = [], [], []

    act = np.ones(n, bool)
    buf = np.zeros((n, BUF))  # upcoming latent increments (raw, before scaling)
    blen = np.zeros(n, np.int64)
    ew = np.full(n, np.nan)  # EWMA of squared latent increments (the runner's own move size)
    gaps = np.ones((n, 2, NLEV - 1), np.int64)
    nlev = np.full((n, 2), NLEV)
    spread = np.ones(n, np.int64)
    zc = rng.standard_normal((n, 2, NLEV))
    spn_e = rng.standard_normal(n)
    tv = tpl["tv0"][perm] * scale
    ltp = np.full(n, -1)
    rho, srho = M["size_rho"], M["spn_rho"]
    sides = np.arange(2)[None, :, None]
    levels = np.arange(NLEV)[None, None, :]
    z_info = np.zeros(n)
    f_disp = None  # latent price (in ticks) the book is currently centred on

    for k in range(T):
        tb = int(tb_of(tj[k]))
        sc = volmult * M["inc_scale"][tb]
        pb_now = pb_of(np.exp(-np.where(act, f, 0.0)))
        for i in np.flatnonzero(act & (blen <= INFO_H)):
            while blen[i] <= INFO_H:
                blk = _draw_block(M, rng, tb, pb_now[i])
                buf[i, blen[i]:blen[i] + BLOCK] = blk
                blen[i] += BLOCK
            if not np.isfinite(ew[i]):
                ew[i] = float(np.mean(buf[i, :blen[i]] ** 2))
        if k > 0:
            raw = buf[:, 0].copy()
            inc = raw * sc
            buf[:, :-1] = buf[:, 1:]
            buf[:, -1] = 0.0
            blen = np.maximum(blen - 1, 0)
            f = np.where(act, f + inc + mu_pb[pb_now] + drift * volmult, f)
            if dmult > 0:  # spells: the drift is renewed from the real distribution, half-life drift_half_life_s
                drift = phi_d * drift + np.sqrt(1 - phi_d ** 2) * draw_drift(pb_now)
            ew = np.where(act, 0.98 * ew + 0.02 * raw ** 2, ew)
            f = _lognorm(f)
            u = np.where(act, phi * u + sig_u * sc * np.sqrt(np.nan_to_num(ew)) * rng.standard_normal(n), 0.0)
        # what the latent price does over the next INFO_H rows, in units of the runner's move size
        z_info = buf[:, :INFO_H].sum(1) / (np.sqrt(np.nan_to_num(ew) * INFO_H) + 1e-12)
        if k in rem_at:
            cand = np.flatnonzero(act)
            i = int(rng.choice(cand))
            p_i = float(np.exp(_lognorm(f + u)[i]))
            rem_step.append(k)
            rem_run.append(i)
            rem_fac.append(float(np.round(100 * p_i, 2)))
            act[i] = False
            f[i] = -np.inf
            u[i] = 0.0
            f = _lognorm(f)
            susp_until = k + int(rng.choice(M["removal_susp"]))
        x = _lognorm(np.where(act, f + u, -np.inf))
        p = np.where(act, np.exp(x), 0.0)

        q = np.clip(p * np.exp(M["log_ovr"][tb]), 1 / 1000.0, 1 / 1.01)
        mid = 1.0 / q
        fi = np.interp(np.log(mid), LOGP, TICKS)
        if f_disp is None:
            f_disp = fi.copy()
            moved = np.zeros(n, bool)
        else:
            moved = np.abs(fi - f_disp) >= g["tick_hysteresis"]
            f_disp = np.where(moved, fi, f_disp)
        fi = f_disp
        pb = pb_of(mid)

        if k == 0:
            spread = _sample_cat(M["spread_p"][pb, tb], rng)
            for sd in range(2):
                gaps[:, sd, :] = _sample_cat(np.repeat(M["gap_p"][sd][pb], NLEV - 1, 0), rng).reshape(n, NLEV - 1)
                nlev[:, sd] = _sample_cat(M["nlev_p"][sd][pb], rng)
        else:
            ch = rng.random(n) < M["spread_change"][pb, tb]
            if ch.any():  # spreads move part-way towards a freshly sampled target (no big jumps)
                d = _sample_cat(M["spread_p"][pb[ch], tb], rng) - spread[ch]
                step = (np.abs(d) + 1) // 2
                # a change of parity shifts the mid by half a tick; with the price still, that shift is pure
                # noise and reverts (a fake reversal), so between moves the spread changes 2 ticks at a time
                step = np.where(moved[ch], step, step - step % 2)
                spread[ch] += np.sign(d) * step
            for sd in range(2):
                m = rng.random((n, NLEV - 1)) < g["gap_resample"]
                if m.any():
                    gaps[:, sd, :][m] = _sample_cat(M["gap_p"][sd][pb[np.nonzero(m)[0]]], rng)
                m = rng.random(n) < g["depth_resample"]
                if m.any():
                    nlev[m, sd] = _sample_cat(M["nlev_p"][sd][pb[m]], rng)
        spread = np.maximum(spread, 1)
        gaps = np.maximum(gaps, 1)
        nl = np.maximum(nlev, 1)

        bb = np.clip(np.rint(fi - spread / 2.0).astype(np.int64), 0, N_TICKS - 2)
        bl = np.minimum(bb + spread, N_TICKS - 1)
        bb = np.minimum(bb, bl - 1)
        off = np.concatenate([np.zeros((n, 2, 1), np.int64), np.cumsum(gaps, 2)], 2)
        bt = bb[:, None] - off[:, 0]
        lt = bl[:, None] + off[:, 1]
        bt[(bt < 0) | (np.arange(NLEV)[None] >= nl[:, 0:1])] = -1
        lt[(lt >= N_TICKS) | (np.arange(NLEV)[None] >= nl[:, 1:2])] = -1

        zc = rho * zc + np.sqrt(1 - rho * rho) * rng.standard_normal(zc.shape)
        Qk = M["size_q"][sides, levels, pb[:, None, None], tb]
        sizes = np.round(scale * np.exp(_q_lookup(Qk, norm_cdf(zc))), 2)

        is_susp = k <= susp_until or k == T - 1
        live = act & (not is_susp)
        tick_ok = np.stack([bt, lt], 1) >= 0
        sizes = np.where(tick_ok & live[:, None, None], sizes, 0.0)
        bt[~live] = -1
        lt[~live] = -1

        # trades at the touch; the side leans towards where the latent price is going next (informed flow)
        ev = live & (rng.random(n) < M["p_trade"][tb, pb] * volmult)
        if ev.any():
            ii = np.flatnonzero(ev)
            v = np.round(scale * np.exp(_q_lookup(M["tvol_q"][tb, pb[ii]], rng.random(len(ii)))), 2)
            p_back = 1.0 / (1.0 + np.exp(-(logit_fab + beta * np.clip(z_info[ii], -4, 4))))
            tk = np.where(rng.random(len(ii)) < p_back, bb[ii], bl[ii])
            tv[ii] += v
            ltp[ii] = tk
            tr_step += [k] * len(ii)
            tr_run += ii.tolist()
            tr_tick += tk.tolist()
            tr_vol += v.tolist()
        ltp = np.where(ltp < 0, np.rint(fi).astype(np.int64), ltp)

        spn_e = srho * spn_e + np.sqrt(1 - srho * srho) * rng.standard_normal(n)
        spn = np.clip(mid * np.exp(_q_lookup(np.broadcast_to(M["spn_q"][tb], (n, NQ)), norm_cdf(spn_e))), 1.01, 1000.0)

        back_tick[k], lay_tick[k] = bt, lt
        back_size[k], lay_size[k] = sizes[:, 0], sizes[:, 1]
        ltp_tick[k] = ltp
        tv_out[k] = tv
        spn_out[k] = np.where(act, np.round(spn, 2), 0.0)
        active[k] = act
        suspended[k] = is_susp
        total_matched[k] = tv.sum()

    # settlement: BSP = last book mid x real error for its price band; winner from BSP-implied probabilities
    q_last = _book_probs_row(back_tick[T - 2], lay_tick[T - 2], act)
    if q_last is None:
        q_last = np.where(act, np.exp(x), 0.0)
    mid_last = np.where(q_last > 0, 1.0 / np.maximum(q_last, 1e-9), 0.0)
    err = _q_lookup(M["bsp_q"][pb_of(np.where(act, mid_last, 1000.0))], rng.random(n))
    bsp = np.where(act, np.clip(np.round(mid_last * np.exp(err), 2), 1.01, 1000.0), 0.0)
    qb = np.where(act, 1.0 / np.maximum(bsp, 1.01), 0.0)
    qb /= qb.sum()
    p_win = qb * M["win_cal"][np.digitize(qb, CAL_EDGES)]
    p_win = np.where(act, p_win, 0.0)
    p_win /= p_win.sum()
    winner = int(rng.choice(n, p=p_win))
    sel = rng.choice(np.arange(10_000_000, 99_999_999), n, replace=False)
    return Tape(
        market_id=f"9.{rng.integers(10 ** 8, 10 ** 9)}", name=name, dt=DT, t_rel=t.astype(np.float32),
        back_tick=back_tick, back_size=back_size, lay_tick=lay_tick, lay_size=lay_size,
        ltp_tick=ltp_tick, tv=tv_out, spn=spn_out, active=active, suspended=suspended, total_matched=total_matched,
        trade_step=np.array(tr_step, np.int32), trade_runner=np.array(tr_run, np.int16),
        trade_tick=np.array(tr_tick, np.int16), trade_vol=np.array(tr_vol, np.float32),
        removal_step=np.array(rem_step, np.int32), removal_runner=np.array(rem_run, np.int16),
        removal_factor=np.array(rem_fac, np.float32), selection_ids=sel.astype(np.int64),
        base_rate=float(tpl["base_rate"]), went_in_play=True, winner=winner, bsp=bsp.astype(np.float32))


# ============================================================================ pools on disk
_WORKER_MODEL = None


def _init_worker(model_path):
    global _WORKER_MODEL
    _WORKER_MODEL = load(model_path)


def _gen_job(args):
    i, seed, out_dir, gen = args
    name = f"20990101_SYN_{seed}_{i:06d}"
    generate_market(_WORKER_MODEL, np.random.default_rng([seed, i]), name, gen).save(os.path.join(out_dir, name + ".npz"))
    return os.path.join(out_dir, name + ".npz")


def write_pool(model_path: str, out_dir: str, n: int, seed: int = 0, workers: int = 1, gen: dict | None = None) -> list[str]:
    """n synthetic races as .npz tapes (reused if already there)."""
    os.makedirs(out_dir, exist_ok=True)
    want = [os.path.join(out_dir, f"20990101_SYN_{seed}_{i:06d}.npz") for i in range(n)]
    jobs = [(i, seed, out_dir, gen) for i, p in enumerate(want) if not os.path.exists(p)]
    t0 = time.time()
    if jobs:
        if workers <= 1:
            _init_worker(model_path)
            for j in jobs:
                _gen_job(j)
        else:
            with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(model_path,)) as ex:
                list(ex.map(_gen_job, jobs, chunksize=4))
        print(f"[synth] wrote {len(jobs)} synthetic races -> {out_dir} ({time.time() - t0:.0f}s)", flush=True)
    return want


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit")
    a.add_argument("--tapes", required=True, help="glob of TRAIN tapes")
    a.add_argument("--model", required=True)
    a.add_argument("--threads", type=int, default=8)
    a.add_argument("--calibrate-iters", type=int, default=4)
    b = sub.add_parser("generate")
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=100)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    c = sub.add_parser("stats", help="calibration statistics of a set of tapes")
    c.add_argument("--tapes", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "fit":
        fit(list_tapes(args.tapes), args.model, threads=args.threads, calibrate_iters=args.calibrate_iters)
    elif args.cmd == "generate":
        write_pool(args.model, args.out, args.n, args.seed, args.workers)
    else:
        print(dyn_stats(list_tapes(args.tapes)))


if __name__ == "__main__":
    main()
