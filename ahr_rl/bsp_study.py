"""BSP value study: is there value betting against Betfair SP, rather than trading
price moves?

Greening needs price *moves* and pays the spread twice. A value bet needs a
better estimate of *who wins* than the price you get, and pays commission only
on winnings. Everything here uses the recorded winners and BSPs (races that went
in play with a known winner).

Per runner we record BSP, the result, the exchange prices at checkpoints before
the off (T-5m, T-2m, T-1m, scheduled start, last pre-off snapshot), Betfair's
projected BSP (spn) at each checkpoint, WoM, recent price moves and the
catalogue form features (if attached to the tapes).

Returns are per $1 at risk, after commission on winnings (market base rate):
  back $1 at price P:              win -> (P-1)(1-c), lose -> -1
  lay with $1 liability at P:      win -> -1,         lose -> (1-c)/(P-1)
"CLV" (closing-line value) is the luck-free version: if BSP is the fair price,
backing at P is worth P/BSP - 1 and a $1-liability lay at P is worth
(1 - P/BSP)/(P-1) (before commission). Realised returns are noisy (a few hundred holdout races); CLV is
not, so both are shown.

  A  Is BSP fair? Win rate vs BSP-implied probability and the return of backing /
     laying everything at BSP, by BSP range (favourite-longshot bias).
  B  Exchange price vs BSP: CLV and realised return of taking the best exchange
     price at each checkpoint instead of BSP, by market share.
  C  Exchange vs projected BSP: when the exchange price is better than Betfair's
     own BSP projection by x%, take it. Thresholds and checkpoint picked on TRAIN
     days, scored once on HOLDOUT days (CLV and realised).
  D  Can anything beat BSP's probabilities? A logistic model of the winner from
     pre-off market data + form, fitted on TRAIN days. On HOLDOUT days: log-loss
     vs BSP's own probabilities, and the return of BSP bets placed only when
     model probability x BSP > 1 + margin (a "limit on close" bet with limit =
     (1 + margin) / model probability, which Betfair supports).

    python -m ahr_rl.bsp_study --tapes "data/tapes/*.npz" --out runs/bsp
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from .env import list_tapes, split_by_date
from .ladder import PRICES
from .tape import Tape

CHECKPOINTS = {"T-5m": -300.0, "T-2m": -120.0, "T-1m": -60.0, "start": 0.0, "last": None}
PRICE_BINS = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 51.0, 1001.0]
PRICE_LABELS = ["<2", "2-3", "3-5", "5-8", "8-13", "13-21", "21-51", "51+"]


def back_ret(price, won, comm):
    return np.where(won, (price - 1) * (1 - comm), -1.0)


def lay_ret(price, won, comm):
    """per $1 of liability"""
    return np.where(won, -1.0, (1.0 - comm) / (price - 1))


def lay_clv(price, bsp):
    return (1 - price / bsp) / (price - 1)


def race_rows(path: str) -> list[dict]:
    t = Tape.load(path)
    if not t.went_in_play or t.winner < 0:
        return []
    race = os.path.basename(path).split(".npz")[0]
    ok = ~t.suspended.copy()
    ok[-1] = False
    if not ok.any():
        return []
    end = int(np.where(ok)[0].max())
    tr = t.t_rel
    R = t.n_runners
    comm = t.base_rate / 100
    bt, lt = t.back_tick[:, :, 0].astype(int), t.lay_tick[:, :, 0].astype(int)
    valid = (bt >= 0) & (lt >= 0) & t.active
    mid = np.where(valid, np.sqrt(PRICES[np.maximum(bt, 0)] * PRICES[np.maximum(lt, 0)]), np.nan)
    bs3, ls3 = t.back_size[:, :, :3].sum(-1), t.lay_size[:, :, :3].sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        wom = np.where(bs3 + ls3 > 0, bs3 / (bs3 + ls3), np.nan)
    live = t.active[end] & (t.bsp > 1.0)
    if live.sum() < 2:
        return []

    def row_at(sec):
        if sec is None:
            return end
        i = int(np.searchsorted(tr, sec))
        return min(max(i, 0), end)

    rows = []
    for r in np.where(live)[0]:
        rec = dict(race=race, day=race[:8], runner=int(r), bsp=float(t.bsp[r]), won=bool(r == t.winner), comm=comm,
                   field=int(live.sum()), matched=float(t.total_matched[end]))
        for name, sec in CHECKPOINTS.items():
            s = row_at(sec)
            ipm = np.where(valid[s] & live, 1 / mid[s], 0.0)
            rec[f"back_{name}"] = float(PRICES[bt[s, r]]) if bt[s, r] >= 0 else np.nan
            rec[f"lay_{name}"] = float(PRICES[lt[s, r]]) if lt[s, r] >= 0 else np.nan
            rec[f"backsz_{name}"] = float(t.back_size[s, r, 0])
            rec[f"laysz_{name}"] = float(t.lay_size[s, r, 0])
            rec[f"mid_{name}"] = float(mid[s, r])
            rec[f"share_{name}"] = float(ipm[r] / ipm.sum()) if ipm.sum() > 0 else np.nan
            rec[f"spn_{name}"] = float(t.spn[s, r]) if t.spn[s, r] > 1 else np.nan
            rec[f"wom_{name}"] = float(wom[s, r])
        s1, s5 = row_at(-60), row_at(-300)
        rec["mom_5to1"] = float(np.log(mid[s1, r] / mid[s5, r])) if np.isfinite(mid[s1, r] * mid[s5, r]) else np.nan
        if t.static is not None and len(t.static) == R:
            from .catalogue import STATIC_NAMES

            for j, n in enumerate(STATIC_NAMES):
                rec[f"st_{n}"] = float(t.static[r, j])
        rows.append(rec)
    return rows


def race_t(v: pd.Series, race: pd.Series) -> dict:
    x = pd.DataFrame({"v": np.asarray(v, float), "race": np.asarray(race)}).dropna()
    if len(x) < 2:
        return dict(n=len(x), races=0, mean=np.nan, t=np.nan)
    per = x.groupby("race")["v"].sum()  # P&L per race (bets in a race are not independent)
    per_bet = x["v"].mean()
    t = per.mean() / (per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 and per.std(ddof=1) > 0 else np.nan
    return dict(n=len(x), races=len(per), mean=float(per_bet), t=float(t))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-races", type=int, default=100000)
    ap.add_argument("--min-size", type=float, default=5.0, help="$ needed at the exchange price to count it as takeable")
    ap.add_argument("--min-bets", type=int, default=100, help="a rule needs this many train bets to be ranked")
    ap.add_argument("--max-bsp", type=float, default=50.0, help="model bets only at BSP <= this")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    t0 = time.time()
    paths = list_tapes(a.tapes)[: a.max_races]
    tr, va, te = split_by_date(paths)
    hold_days = {os.path.basename(p)[:8] for p in va + te}
    rows = []
    for i, p in enumerate(paths):
        rows.extend(race_rows(p))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paths)} races ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df["split"] = np.where(df["day"].isin(hold_days), "holdout", "train")
    df["bsp_b"] = pd.cut(df["bsp"], PRICE_BINS, labels=PRICE_LABELS)
    df.to_parquet(os.path.join(a.out, "bsp_rows.parquet"), index=False)
    print(f"{df['race'].nunique()} races with a winner and BSP, {len(df)} runners "
          f"({(df['split'] == 'holdout').groupby(df['race']).first().sum()} holdout races)\n")

    # ---------------- A: is BSP fair?
    print("=" * 100 + "\nA. IS BSP FAIR? back / lay everything at BSP, per $1 (after commission), by BSP range\n" + "=" * 100)
    df["back_bsp"] = back_ret(df["bsp"], df["won"], df["comm"])
    df["lay_bsp"] = lay_ret(df["bsp"], df["won"], df["comm"])
    rows_a = []
    for b, g in df.groupby("bsp_b", observed=True):
        bk, ly = race_t(g["back_bsp"], g["race"]), race_t(g["lay_bsp"], g["race"])
        rows_a.append(dict(bsp=b, runners=len(g), win_rate=g["won"].mean() * 100, implied=(1 / g["bsp"]).mean() * 100,
                           back_roi=bk["mean"] * 100, back_t=bk["t"], lay_roi=ly["mean"] * 100, lay_t=ly["t"]))
    A = pd.DataFrame(rows_a)
    A["win/implied"] = A["win_rate"] / A["implied"]
    A.to_csv(os.path.join(a.out, "A_bsp_fair.csv"), index=False)
    print(A.round(3).to_string(index=False))
    s = df.groupby("race")["bsp"].apply(lambda x: (1 / x).sum() * 100)
    print(f"\nBSP book % (sum of 1/BSP per race): median {s.median():.1f}%  (100% = fair; commission then applies on winnings)")

    # ---------------- B: exchange vs BSP
    print("\n" + "=" * 100 + "\nB. TAKE THE EXCHANGE PRICE INSTEAD OF BSP? CLV = value vs BSP (luck-free); realised = actual "
          "return. Per $1, runners with >= 10% share at the checkpoint\n" + "=" * 100)
    rows_b = []
    for cp in CHECKPOINTS:
        g = df[(df[f"share_{cp}"] >= 0.10)]
        for side in ("back", "lay"):
            px = g[f"{side}_{cp}"]
            size_ok = g[f"{side}sz_{cp}"] >= a.min_size
            gg = g[px.notna() & size_ok]
            px = gg[f"{side}_{cp}"]
            clv = px / gg["bsp"] - 1 if side == "back" else lay_clv(px, gg["bsp"])
            real = back_ret(px, gg["won"], gg["comm"]) if side == "back" else lay_ret(px, gg["won"], gg["comm"])
            c, rr = race_t(clv, gg["race"]), race_t(pd.Series(real, index=gg.index), gg["race"])
            rows_b.append(dict(checkpoint=cp, side=side, bets=c["n"], clv_pct=c["mean"] * 100, clv_t=c["t"],
                               realised_pct=rr["mean"] * 100, realised_t=rr["t"],
                               better_than_bsp_pct=(clv > 0).mean() * 100))
    B = pd.DataFrame(rows_b)
    B.to_csv(os.path.join(a.out, "B_exchange_vs_bsp.csv"), index=False)
    print(B.round(3).to_string(index=False))

    # ---------------- C: exchange vs projected BSP rule
    print("\n" + "=" * 100 + "\nC. RULE: take the exchange price when it beats Betfair's projected BSP by x% "
          "(picked on TRAIN, scored on HOLDOUT)\n" + "=" * 100)
    res = []
    for cp in [c for c in CHECKPOINTS if c != "last"]:
        for side in ("back", "lay"):
            for x in (0.0, 0.02, 0.05, 0.10, 0.20):
                for minshare in (0.0, 0.10):
                    rec = dict(checkpoint=cp, side=side, x=x, min_share=minshare)
                    for split, g in df.groupby("split"):
                        g = g[(g[f"share_{cp}"] >= minshare) & g[f"spn_{cp}"].notna() & g[f"{side}_{cp}"].notna()
                              & (g[f"{side}sz_{cp}"] >= a.min_size)]
                        px, spn = g[f"{side}_{cp}"], g[f"spn_{cp}"]
                        sel = g[(px > spn * (1 + x))] if side == "back" else g[(px < spn * (1 - x))]
                        pxs = sel[f"{side}_{cp}"]
                        clv = pxs / sel["bsp"] - 1 if side == "back" else lay_clv(pxs, sel["bsp"])
                        real = back_ret(pxs, sel["won"], sel["comm"]) if side == "back" else lay_ret(pxs, sel["won"], sel["comm"])
                        c, rr = race_t(clv, sel["race"]), race_t(pd.Series(real, index=sel.index), sel["race"])
                        rec.update({f"{split}_bets": c["n"], f"{split}_clv": c["mean"], f"{split}_clv_t": c["t"],
                                    f"{split}_real": rr["mean"], f"{split}_real_t": rr["t"]})
                    res.append(rec)
    C = pd.DataFrame(res).sort_values("train_clv_t", ascending=False)
    C.to_csv(os.path.join(a.out, "C_rules.csv"), index=False)
    C = C[C["train_bets"] >= a.min_bets]
    if C.empty:
        raise SystemExit(f"no rule has >= {a.min_bets} train bets (lower --min-bets)")
    print(f"({len(C)} rules with >= {a.min_bets} train bets; ranked by train CLV t)")
    print(C.head(12).round(4).to_string(index=False))
    bc = C.iloc[0]

    # ---------------- D: can anything beat BSP's probabilities?
    print("\n" + "=" * 100 + "\nD. A WIN MODEL FROM PRE-OFF DATA (T-1m) + FORM vs BSP'S OWN PROBABILITIES\n" + "=" * 100)
    from sklearn.linear_model import LogisticRegression

    feats = ["share_T-1m", "share_T-5m", "wom_T-1m", "mom_5to1", "field"] + [c for c in df.columns if c.startswith("st_")]
    d = df.copy()
    d["logit_share"] = np.log(d["share_T-1m"].clip(1e-4, 0.999) / (1 - d["share_T-1m"].clip(1e-4, 0.999)))
    d["spn_share"] = d.groupby("race")["spn_T-1m"].transform(lambda x: (1 / x) / (1 / x).sum())
    d["logit_spn"] = np.log(d["spn_share"].clip(1e-4, 0.999) / (1 - d["spn_share"].clip(1e-4, 0.999)))
    X_cols = ["logit_share", "logit_spn"] + feats
    d = d.dropna(subset=["logit_share"])
    d[X_cols] = d[X_cols].fillna(d[X_cols].median())
    tr_d, ho_d = d[d["split"] == "train"], d[d["split"] == "holdout"]
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(tr_d[X_cols].values, tr_d["won"].values)

    def race_norm(p, races):
        s = pd.Series(p, index=races.index).groupby(races).transform("sum")
        return (p / s.values)

    ho = ho_d.copy()
    ho["p_model"] = race_norm(m.predict_proba(ho[X_cols].values)[:, 1], ho["race"])
    ho["p_bsp"] = ho.groupby("race")["bsp"].transform(lambda x: (1 / x) / (1 / x).sum())
    ho["p_mkt"] = ho.groupby("race")["share_T-1m"].transform(lambda x: x / x.sum())

    def ll(p, y):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(np.log(p[y])).mean())  # log-loss of the winner (race-level multinomial)

    y = ho["won"].values
    lls = {"model": ll(ho["p_model"].values, y), "bsp": ll(ho["p_bsp"].values, y),
           "exchange_T-1m": ll(ho["p_mkt"].values, y)}
    print(f"HOLDOUT winner log-loss (lower = better): model {lls['model']:.4f} | BSP {lls['bsp']:.4f} | "
          f"exchange at T-1m {lls['exchange_T-1m']:.4f}")
    coef = pd.Series(m.coef_[0], index=X_cols).sort_values(key=np.abs, ascending=False)
    print("largest model coefficients:", ", ".join(f"{k} {v:+.2f}" for k, v in coef.head(8).items()))
    ho = ho[ho["bsp"] <= a.max_bsp]
    rows_d = []
    for margin in (0.0, 0.05, 0.10, 0.20, 0.30):
        for side in ("back", "lay"):
            if side == "back":
                sel = ho[ho["p_model"] * ho["bsp"] > 1 + margin]
                ret = back_ret(sel["bsp"], sel["won"], sel["comm"])
                ev = sel["p_bsp"] * sel["bsp"] - 1
            else:
                sel = ho[ho["p_model"] * ho["bsp"] < 1 - margin]
                ret = lay_ret(sel["bsp"], sel["won"], sel["comm"])
                ev = (1 - sel["p_bsp"] * sel["bsp"]) / (sel["bsp"] - 1)
            rr = race_t(pd.Series(ret, index=sel.index), sel["race"])
            rows_d.append(dict(side=side, margin=margin, bets=rr["n"], races=rr["races"], realised_pct=rr["mean"] * 100,
                               t=rr["t"], avg_bsp=sel["bsp"].mean()))
    D = pd.DataFrame(rows_d)
    D.to_csv(os.path.join(a.out, "D_model_bets.csv"), index=False)
    print(f"\nHOLDOUT BSP bets (BSP <= {a.max_bsp:g}) placed when model prob x BSP clears the margin "
          "(back: > 1 + margin; lay: < 1 - margin). Per $1 at risk, after commission:")
    print(D.round(3).to_string(index=False))

    verdict = {
        "bsp_book_pct_median": float(s.median()),
        "best_rule_C": {k: (float(bc[k]) if isinstance(bc[k], (float, np.floating, int, np.integer)) else bc[k])
                        for k in ("checkpoint", "side", "x", "min_share", "train_clv", "train_clv_t",
                                  "holdout_bets", "holdout_clv", "holdout_clv_t", "holdout_real", "holdout_real_t")},
        "model_logloss": lls,
        "model_beats_bsp": bool(lls["model"] < lls["bsp"]),
    }
    verdict["edge"] = bool(verdict["best_rule_C"]["holdout_clv"] > 0 and (verdict["best_rule_C"]["holdout_clv_t"] or 0) > 2)
    json.dump(verdict, open(os.path.join(a.out, "verdict.json"), "w"), indent=1, default=str)
    print("\n=== Verdict ===")
    print(json.dumps(verdict, indent=1, default=str))
    print(f"\nsaved to {a.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
