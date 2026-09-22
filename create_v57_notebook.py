"""Generate MarketMaking_V57_ActionSpec.ipynb"""
import json, sys

# ── helpers ──────────────────────────────────────────────────────────────────
def md(source):   return {"cell_type":"markdown","metadata":{},"source":source}
def code(source): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source}

# ── load V56a FastTrack source cells we'll reuse verbatim ────────────────────
FT = '/root/.claude/uploads/df5c4d9d-33b5-5f8b-87a1-6e9841c7e0c8/1503d646-MarketMaking_V56a_ArmB_FastTrack.ipynb'
with open(FT) as f:
    v56 = json.load(f)
def v56src(i): return ''.join(v56['cells'][i]['source'])

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# Cell 0  markdown – design doc
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
# V57 — 25-Action Directional Spec

A/B test against V56a. Everything identical **except the action space and environment**.

## Action space (25 dims, all SAC-native Box[-1,1])

| dims | range | meaning |
|------|-------|---------|
| `action[0:MAX_RUNNERS]` | [-1, 1] | Per-runner signal. >0 = back, <0 = lay, magnitude = confidence |
| `action[MAX_RUNNERS]` | [-1, 1] | Stake fraction → (x+1)/2 maps to [0,1] of available capital |

At each bar the agent chooses how much of its available bankroll to deploy (`stake_frac`)
and splits it proportionally to `|signal[r]|` across all runners.  Runners with signal ≈ 0
receive near-zero allocation naturally — no threshold gate needed.

**Key advantage over V56a's 24-dim offset spec:** the do-nothing policy is
a *single-dimension* discovery — push `action[MAX_RUNNERS]` to -1 (stake→0).
V56a required all 24 dims below -0.5 simultaneously (24-dimensional coordination).

## Execution model

| direction | fill price |
|-----------|-----------|
| back  (signal > 0) | `bl` — cross the spread, take the best lay offer |
| lay   (signal < 0) | `bb` — cross the spread, take the best back offer |

No passive orders, no queue simulation.  All fills are market orders.

## Observation (422 dims)

Identical static runner features to V56a (26 dims/runner).
Own-order features replaced by 6 position-tracking dims:

| dim | meaning |
|-----|---------|
| `pos_back` | back stake / MAX_RUNNER_EXP |
| `pos_lay`  | lay liability / MAX_RUNNER_EXP |
| `bpx_dist` | avg back price vs current bb (normalized) |
| `lpx_dist` | avg lay price vs current bb (normalized) |
| `green_r`  | per-runner green value (normalized) |
| `pnl_r`    | same as green_r (placeholder for net PnL) |

Market features: V56a's 24 + 2 new (total_risk_frac, available_frac) = 26.

`OBS_DIM = 12×32 + 26 + 12 = 422`

## Stage A gate

Same as V56a FastTrack:
- random ~ -$200
- do-nothing = $0
- trained must reach ≥ -$1 to pass
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 1  mount drive
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Cell 0 — Mount Drive"))
cells.append(code("from google.colab import drive\ndrive.mount('/content/drive')"))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2  environment check (verbatim from V56a cell 4)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Cell 0b — Environment check"))
cells.append(code(v56src(4)))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 3  config
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Cell 1 — Config"))
cells.append(code("""\
import os, sys, json, gzip, zlib, time, math, warnings, datetime as dt
from pathlib import Path
from collections import defaultdict
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ── Drive layout (identical to V56a) ─────────────────────────────────────────
DRIVE_DATA = None
if DRIVE_DATA is None:
    root = Path("/content/drive/MyDrive")
    hits = list(root.rglob("recordings/*.ndjson.gz"))
    if not hits:
        raise FileNotFoundError(
            f"No */recordings/*.ndjson.gz under {root}.\\n"
            "Set DRIVE_DATA manually.")
    parents = [p.parent.parent for p in hits]
    DRIVE_DATA = max(set(parents), key=parents.count)

print(f"drive data : {DRIVE_DATA}")
n_drive = len(list((DRIVE_DATA / "recordings").glob("*.ndjson.gz")))
assert n_drive > 0
print(f"on drive   : {n_drive} recordings")

DATA_DIR = Path("/content/bf_local"); DATA_DIR.mkdir(exist_ok=True)
REC_DIR  = DATA_DIR / "recordings";   REC_DIR.mkdir(exist_ok=True)
have = {f.name for f in REC_DIR.glob("*.ndjson.gz")}
todo = [f for f in (DRIVE_DATA/"recordings").glob("*.ndjson.gz") if f.name not in have]
if todo:
    import shutil, time as _t; _t0=_t.time()
    for f in todo: shutil.copy2(f, REC_DIR/f.name)
    print(f"copied     : {len(todo)} new in {_t.time()-_t0:.0f}s")
CAT_DIR = DATA_DIR/"catalogues"; CAT_DIR.mkdir(exist_ok=True)
_hc = {f.name for f in CAT_DIR.glob("*.json")}
_tc = [f for f in (DRIVE_DATA/"catalogues").glob("*.json") if f.name not in _hc]
if _tc:
    import shutil as _sh
    for f in _tc: _sh.copy2(f, CAT_DIR/f.name)
    print(f"catalogues : {len(_tc)} copied")
n_local = len(list(REC_DIR.glob("*.ndjson.gz")))
print(f"local      : {n_local} recordings")
assert n_local == n_drive

CACHE_DIR = DRIVE_DATA / "rl_cache"; CACHE_DIR.mkdir(exist_ok=True)
CORPUS_FREEZE = None
ARM     = "V57_action_spec"
RUN_DIR = DRIVE_DATA / f"rl_runs_{ARM}"; RUN_DIR.mkdir(exist_ok=True)

# ── Episode / bars ───────────────────────────────────────────────────────────
BAR_S          = 1.0
MAX_RUNNERS    = 12
MIN_PROB       = 0.01
PREOFF_START_S = 600
K_OFFSETS      = 4      # kept for cache-key compatibility with V56a CACHE_SCHEMA 6

# ── New action-spec parameters ───────────────────────────────────────────────
STARTING_BANK  = 1000.0   # initial bankroll ($)
MIN_BET        = 1.0      # minimum allocation per runner ($)
MAX_RUNNER_EXP = 200.0    # normalisation constant for position sizes

# ── Economics ────────────────────────────────────────────────────────────────
COMMISSION = 0.08         # default if catalogue missing (8%)

# ── SAC / entropy ────────────────────────────────────────────────────────────
GAMMA           = 0.99
# RAISED vs V56a (was effectively ~0.001 — caused entropy collapse by ep 50)
ENT_COEF_FLOOR  = 0.05
ENT_COEF_CEIL   = 2.0
# 25 action dims; fully-uniform H ≈ 25*ln(2) ≈ +17.3 nats.
# Target -8 is moderately low — policy can specialize but can't collapse.
TARGET_ENTROPY  = -8.0

# ── Training ─────────────────────────────────────────────────────────────────
SEED            = 42
FILL_MODE       = "touch"     # market orders only
FASTTRACK_N     = 5
PASSES_PER_RACE = 1000

print(f"  ARM            : {ARM}")
print(f"  MAX_RUNNERS    : {MAX_RUNNERS}")
print(f"  STARTING_BANK  : ${STARTING_BANK:.0f}")
print(f"  COMMISSION     : {COMMISSION:.0%}")
print(f"  ENT_COEF_FLOOR : {ENT_COEF_FLOOR}")
print(f"  TARGET_ENTROPY : {TARGET_ENTROPY}")
print(f"  action dims    : {MAX_RUNNERS + 1} (24 runner signals + 1 stake fraction)")
print("Config OK")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 4  catalogue audit (verbatim V56a cell 8)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Cell 1b — Catalogue audit"))
cells.append(code(v56src(8)))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 5  preprocessing — parse_to_bars (verbatim V56a cell 10)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Cell 2 — Offline preprocessing
`CACHE_SCHEMA = 6` — identical to V56a.  If V56a already built the cache,
this cell just finds existing `.npz` files and skips rebuilding."""))
cells.append(code(v56src(10)))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 6  build cache (verbatim V56a cell 11, reuse CACHE_SCHEMA=6)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(code(v56src(11)))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 7  calibrate adverse (verbatim V56a cell 13)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Cell 2b — Adverse selection calibration"))
cells.append(code(v56src(13)))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 8  observation spec  *** NEW ***
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Cell 3 — Observation spec (V57)

26 static runner features (identical to V56a) + **6 own-position features** (new).
Market features: V56a's 24 + 2 portfolio-level dims = 26.

`OBS_DIM = MAX_RUNNERS × NF_RUN + len(MKT_FEATS) + MAX_RUNNERS = 12×32 + 26 + 12 = 422`"""))
cells.append(code('''\
# ── Static runner features (identical to V56a) ───────────────────────────────
RUNNER_FEATS = [
 "spread_t","wom_l1","wom_l3","wom_near","wom_deep","shape_b","shape_l",
 "log_depth","log_tv","ofi_1s","ofi_5s","ofi_15s","trade_int",
 "vwap_dist","poc_dist","ret_5s","ret_15s","ret_60s","field_rel_ret",
 "field_rel_ofi","spread_rel",
 "prob","prob_rank","prob_gap",
 "own_back_off","own_lay_off"]   # last 2 zeros in V57 (no open passive orders)

# ── Own-position features (V57 — tracks market-order positions) ──────────────
OWN_FEATS = [
 "pos_back",   # back stake / MAX_RUNNER_EXP
 "pos_lay",    # lay liability / MAX_RUNNER_EXP
 "bpx_dist",   # (avg_back_px - current_bb) / current_bb  (+ = backed high)
 "lpx_dist",   # (avg_lay_px  - current_bb) / current_bb  (+ = laid above market)
 "green_r",    # per-runner green value / MAX_RUNNER_EXP
 "pnl_r",      # same for now (placeholder for net realized PnL)
]

# ── Market features (V56a's 24 + 2 portfolio dims) ───────────────────────────
MKT_FEATS_BASE = [
 "secs_to_off_n","log_secs_to_off","log_matched","back_or","lay_or",
 "entropy","n_run_n",
 "is_harness","liq_prior",
 "bsp_mkt","turn_inplay","persist","mkt_age",
 "d_sprint","d_mile","d_middle","d_stay",
 "cls_mdn","cls_hcap","cls_prem","cls_level"]
# V56a has 24; use the same list but store separately so the extractor knows the split
MKT_FEATS = MKT_FEATS_BASE + ["total_risk_frac", "available_frac"]

NF_STATIC = len(RUNNER_FEATS)    # 26
NF_OWN    = len(OWN_FEATS)       # 6
NF_RUN    = NF_STATIC + NF_OWN   # 32
OBS_DIM   = MAX_RUNNERS * NF_RUN + len(MKT_FEATS) + MAX_RUNNERS  # 422

print(f"NF_STATIC  = {NF_STATIC}")
print(f"NF_OWN     = {NF_OWN}")
print(f"NF_RUN     = {NF_RUN}")
print(f"MKT_FEATS  = {len(MKT_FEATS)}")
print(f"OBS_DIM    = {OBS_DIM}")

# ── precompute_features: IDENTICAL to V56a ───────────────────────────────────
# (own_back_off / own_lay_off are set to 0 since there are no passive orders;
#  the OWN_FEATS above fill those slots dynamically in _obs())
def precompute_features(F):
    """Static (market-derived) part of the observation. Own-order dims stay zero."""
    T,R=F["T"],F["R"]; eps=1e-9
    bbt,blt=F["bbt"],F["blt"]
    mid=0.5*(bbt+blt)
    X=np.zeros((T,R,NF_STATIC),dtype=np.float32)
    g=lambda n: RUNNER_FEATS.index(n)
    X[:,:,g("spread_t")]=np.clip(blt-bbt,0,40)/10.0
    for nm,(b,l) in [("wom_l1",(F["l1b"],F["l1l"])),("wom_l3",(F["t3b"],F["t3l"])),
                     ("wom_near",(F["nearb"],F["nearl"])),("wom_deep",(F["deepb"],F["deepl"]))]:
        X[:,:,g(nm)]=(b-l)/(b+l+eps)
    X[:,:,g("shape_b")]=F["t3b"]/(F["deepb"]+eps)
    X[:,:,g("shape_l")]=F["t3l"]/(F["deepl"]+eps)
    X[:,:,g("log_depth")]=np.log1p(F["deepb"]+F["deepl"])/10.0
    X[:,:,g("log_tv")]=np.log1p(F["tv"])/10.0
    ofi=F["lay_agg"]-F["back_agg"]; tot=F["lay_agg"]+F["back_agg"]
    X[:,:,g("ofi_1s")]=np.tanh(ofi/50.0)
    for w,nm in [(5,"ofi_5s"),(15,"ofi_15s")]:
        c=np.cumsum(ofi,axis=0)
        roll=c-np.vstack([np.zeros((min(w,T),R)),c[:-w]]) if T>w else c
        X[:,:,g(nm)]=np.tanh(roll/(50.0*w))
    X[:,:,g("trade_int")]=np.log1p(tot)/5.0
    X[:,:,g("vwap_dist")]=np.nan_to_num(np.clip(F["vwap_t"]-mid,-20,20))/10.0
    X[:,:,g("poc_dist")]=np.nan_to_num(np.clip(F["poc_t"]-mid,-20,20))/10.0
    for w,nm in [(5,"ret_5s"),(15,"ret_15s"),(60,"ret_60s")]:
        r=np.zeros((T,R)); r[w:]=mid[w:]-mid[:-w]
        X[:,:,g(nm)]=np.clip(r,-15,15)/5.0
    fr=X[:,:,g("ret_15s")]
    X[:,:,g("field_rel_ret")]=fr-fr.mean(axis=1,keepdims=True)
    o1=X[:,:,g("ofi_1s")]
    X[:,:,g("field_rel_ofi")]=o1-o1.mean(axis=1,keepdims=True)
    sp=np.clip(blt-bbt,0,40)
    c=np.cumsum(sp,axis=0); w=60
    roll=(c-np.vstack([np.zeros((min(w,T),R)),c[:-w]]))/min(w,T) if T>w else sp
    X[:,:,g("spread_rel")]=np.tanh(sp/np.maximum(roll,1e-9)-1.0)
    px=0.5*(LADDER[np.clip(bbt.round().astype(int),0,NT-1)]
            +LADDER[np.clip(blt.round().astype(int),0,NT-1)])
    prob=1.0/np.maximum(px,1.01); pn=prob/np.maximum(prob.sum(axis=1,keepdims=True),eps)
    X[:,:,g("prob")]=pn
    X[:,:,g("prob_rank")]=np.argsort(np.argsort(-pn,axis=1),axis=1)/max(R-1,1)
    X[:,:,g("prob_gap")]=pn.max(axis=1,keepdims=True)-pn
    # own_back_off, own_lay_off stay 0 (no passive orders in V57)
    M=np.zeros((T,len(MKT_FEATS)),dtype=np.float32)
    k=lambda n: MKT_FEATS.index(n)
    t_vec=np.arange(T-1,-1,-1,dtype=float)*BAR_S
    M[:,k("secs_to_off_n")]=np.clip(t_vec/PREOFF_START_S,0,1)
    M[:,k("log_secs_to_off")]=np.log1p(t_vec)/np.log1p(PREOFF_START_S)
    if "tv_mkt" in F: M[:,k("log_matched")]=np.log1p(F["tv_mkt"])/15.0
    if "back_or" in F: M[:,k("back_or")]=np.clip((F["back_or"]-1.0)/0.2,0,1)
    if "lay_or"  in F: M[:,k("lay_or")] =np.clip((F["lay_or"]-1.0)/0.2,0,1)
    M[:,k("entropy")]=(-pn*np.log(pn+eps)).sum(axis=1)/math.log(max(R,2))
    M[:,k("n_run_n")]=R/MAX_RUNNERS
    for kn,fk in [("is_harness","is_harness"),("liq_prior","liq_prior"),
                  ("bsp_mkt","bsp_mkt"),("turn_inplay","turn_inplay"),
                  ("persist","persist"),("mkt_age","mkt_age")]:
        if fk in F: M[:,k(kn)]=float(F[fk])
    for kn in ("d_sprint","d_mile","d_middle","d_stay"):
        if kn in F: M[:,k(kn)]=float(F[kn])
    for kn in ("cls_mdn","cls_hcap","cls_prem","cls_level"):
        if kn in F: M[:,k(kn)]=float(F[kn])
    # portfolio dims filled dynamically in _obs(); leave as zero here
    return X, M

print("precompute_features defined (V57 — static block)")
'''))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 9  load races (adapted from V56a build section)
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("## Cell 3b — Load & precompute races"))
cells.append(code("""\
import random as _rnd

def load_races(cache_paths, n=None, seed=SEED):
    rng = np.random.default_rng(seed)
    paths = list(cache_paths)
    if n is not None:
        paths = list(rng.choice(paths, min(n, len(paths)), replace=False))
    races = []
    for cp in paths:
        z = np.load(cp, allow_pickle=True)
        F = {k: z[k] for k in z.files}
        F["T"] = int(F["T"]); F["R"] = int(F["R"])
        # Build mask (T, MAX_RUNNERS)
        R = F["R"]
        mask = np.zeros((F["T"], MAX_RUNNERS), dtype=np.float32)
        mask[:, :R] = 1.0
        F["mask"] = mask
        # Precompute static features
        F["X"], F["M_static"] = precompute_features(F)
        races.append(F)
    return races

# Fast-track: load N random races only
print(f"Loading {FASTTRACK_N} random races for Stage A...")
RACES_ALL = load_races(CACHE, n=FASTTRACK_N, seed=SEED)
print(f"Loaded {len(RACES_ALL)} races")
for F in RACES_ALL:
    print(f"  {F.get('name','?')}  T={F['T']} R={F['R']} comm={F.get('commission',COMMISSION):.0%}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 10  environment  *** MAIN NEW CELL ***
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Cell 4 — Environment (V57 — 25-action directional spec)

**Action[0:MAX_RUNNERS]**: per-runner signal ∈ [-1,1].  Positive = back, negative = lay.
**Action[MAX_RUNNERS]**: stake fraction → (x+1)/2 ∈ [0,1] of available capital.

Allocation per runner = `total_wager × |signal[r]| / Σ|signal|`.
Fill: back at `bl` (cross the spread to take lay offers), lay at `bb`.

Reward: potential-based shaping Φ(s) = `net_green(book)`.
`r_t = γΦ(s') − Φ(s)`.  Terminal: forced flatten at off prices."""))
cells.append(code('''\
import gymnasium as gym
from gymnasium import spaces

# ── commission helper (identical to V56a) ────────────────────────────────────
def green_value(pos_back_stake, pos_back_px, pos_lay_liab, pos_lay_px, bb, bl):
    out = np.zeros_like(bb)
    m = pos_back_stake > 0
    if m.any(): out[m] += pos_back_stake[m] * (pos_back_px[m] / np.maximum(bl[m], 1.01) - 1.0)
    m = pos_lay_liab > 0
    if m.any(): out[m] += pos_lay_liab[m] * (1.0 - pos_lay_px[m] / np.maximum(bb[m], 1.01))
    return out

def net_green(gross_per_runner, comm):
    g = float(np.sum(gross_per_runner))
    return g * (1.0 - comm) if g > 0 else g


class DirectionalRaceEnv(gym.Env):
    """V57: 25-action directional trading environment.

    The single stake-fraction dimension solves V56a 24-dim coordination problem:
    to do nothing, the agent only needs action[MAX_RUNNERS] -> -1 (stake->0).
    """
    metadata = {"render_modes": []}

    def __init__(self, races, difficulty=1.0, null_mode=False, seed=SEED):
        super().__init__()
        self.races      = races
        self.difficulty = difficulty
        self.null_mode  = null_mode
        self.rng        = np.random.default_rng(seed)
        self.action_space      = spaces.Box(-1, 1, shape=(MAX_RUNNERS + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.F    = self.races[self.rng.integers(len(self.races))]
        self.T    = self.F["T"]; self.R = self.F["R"]
        self.comm = float(self.F.get("commission", COMMISSION))
        self.i    = 0
        self.bs   = np.zeros(MAX_RUNNERS)   # back stake
        self.bpx  = np.zeros(MAX_RUNNERS)   # avg back fill price
        self.ll   = np.zeros(MAX_RUNNERS)   # lay liability
        self.lpx  = np.zeros(MAX_RUNNERS)   # avg lay fill price
        self.prev_phi     = 0.0
        self.shaping_total= 0.0
        self.n_trades     = 0
        self.bk_px_sum    = np.zeros(MAX_RUNNERS)
        self.bk_px_n      = np.zeros(MAX_RUNNERS)
        self.ly_px_sum    = np.zeros(MAX_RUNNERS)
        self.ly_px_n      = np.zeros(MAX_RUNNERS)
        return self._obs(), {}

    # ── price helpers ─────────────────────────────────────────────────────────
    def _px(self, arr, i=None):
        i = self.i if i is None else i
        return LADDER[np.clip(np.round(arr[i]).astype(int), 0, NT-1)]

    def _px_pad(self, arr, i=None):
        """Padded price array → shape (MAX_RUNNERS,); out-of-range runners = 1.01."""
        raw = self._px(arr, i)            # shape (R,)
        out = np.full(MAX_RUNNERS, 1.01)
        out[:len(raw)] = raw
        return out

    # ── observation ──────────────────────────────────────────────────────────
    def _obs(self):
        i    = min(self.i, self.T - 1)
        X    = self.F["X"][i]             # (R, NF_STATIC)
        M_st = self.F["M_static"][i]      # (len(MKT_FEATS),) precomputed
        mask = self.F["mask"][i]          # (MAX_RUNNERS,)
        bb   = self._px_pad(self.F["bbt"], i)
        bl   = self._px_pad(self.F["blt"], i)

        # Build runner block (MAX_RUNNERS, NF_RUN)
        obs_r = np.zeros((MAX_RUNNERS, NF_RUN), dtype=np.float32)
        obs_r[:self.R, :NF_STATIC] = X   # static features

        # Own-position features (6 per runner)
        for r in range(self.R):
            obs_r[r, NF_STATIC + 0] = self.bs[r] / MAX_RUNNER_EXP
            obs_r[r, NF_STATIC + 1] = self.ll[r] / MAX_RUNNER_EXP
            if self.bs[r] > 0 and bb[r] > 1.0:
                obs_r[r, NF_STATIC + 2] = np.clip((self.bpx[r] - bb[r]) / bb[r], -1, 1)
            if self.ll[r] > 0 and bb[r] > 1.0:
                obs_r[r, NF_STATIC + 3] = np.clip((self.lpx[r] - bb[r]) / bb[r], -1, 1)
            gv = float(green_value(
                np.array([self.bs[r]]), np.array([self.bpx[r]]),
                np.array([self.ll[r]]), np.array([self.lpx[r]]),
                np.array([bb[r]]),      np.array([bl[r]]))[0])
            obs_r[r, NF_STATIC + 4] = np.clip(gv / max(MAX_RUNNER_EXP, 1), -1, 1)
            obs_r[r, NF_STATIC + 5] = obs_r[r, NF_STATIC + 4]

        obs_r *= mask[:, np.newaxis]

        # Portfolio dims (fill the last 2 slots of M_static in-place copy)
        M = M_st.copy()
        total_risk       = (self.bs.sum() + self.ll.sum()) / STARTING_BANK
        M[-2]            = np.float32(min(total_risk, 1.0))
        M[-1]            = np.float32(max(0.0, 1.0 - total_risk))

        return np.concatenate([obs_r.reshape(-1), M, mask]).astype(np.float32)

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        signals    = np.clip(action[:MAX_RUNNERS], -1.0, 1.0).astype(float)
        stake_frac = (float(action[MAX_RUNNERS]) + 1.0) / 2.0  # [-1,1] → [0,1]

        i    = min(self.i, self.T - 1)
        mask = self.F["mask"][i].astype(bool)
        bb   = self._px_pad(self.F["bbt"])   # lay-order fill prices for backs
        bl   = self._px_pad(self.F["blt"])   # back-order fill prices for lays

        # Null-control: permute signals so they carry no per-runner information
        if self.null_mode:
            signals = self.rng.permutation(signals)

        # Only trade on valid runners
        masked_sig = signals.copy()
        masked_sig[~mask] = 0.0

        # Available capital and wager
        total_risk   = self.bs.sum() + self.ll.sum()
        available    = max(0.0, STARTING_BANK - total_risk)
        total_wager  = available * stake_frac

        abs_sig  = np.abs(masked_sig)
        abs_sum  = abs_sig.sum()

        if abs_sum > 1e-8 and total_wager >= MIN_BET:
            weights = abs_sig / abs_sum
            alloc   = total_wager * weights

            for r in range(MAX_RUNNERS):
                if alloc[r] < MIN_BET or not mask[r]:
                    continue
                sig = float(masked_sig[r])
                a   = float(alloc[r])

                if sig > 0:          # ── BACK at bl[r] ──────────────────────
                    px = float(bl[r])
                    if px < 1.01: continue
                    if self.bs[r] > 0:
                        self.bpx[r] = (self.bs[r]*self.bpx[r] + a*px) / (self.bs[r]+a)
                    else:
                        self.bpx[r] = px
                    self.bs[r]      += a
                    self.n_trades   += 1
                    self.bk_px_sum[r] += px; self.bk_px_n[r] += 1

                elif sig < 0:        # ── LAY at bb[r]  ──────────────────────
                    px   = float(bb[r])
                    if px < 1.01: continue
                    liab = a
                    if self.ll[r] > 0:
                        self.lpx[r] = (self.ll[r]*self.lpx[r] + liab*px) / (self.ll[r]+liab)
                    else:
                        self.lpx[r] = px
                    self.ll[r]      += liab
                    self.n_trades   += 1
                    self.ly_px_sum[r] += px; self.ly_px_n[r] += 1

        # Potential-based shaping reward
        phi    = net_green(green_value(self.bs, self.bpx, self.ll, self.lpx, bb, bl), self.comm)
        reward = GAMMA * phi - self.prev_phi
        self.prev_phi       = phi
        self.shaping_total += reward

        self.i += 1
        done   = self.i >= self.T
        info   = {}

        if done:
            i_fin = self.T - 1
            bb_f  = self._px_pad(self.F["bbt"], i_fin)
            bl_f  = self._px_pad(self.F["blt"], i_fin)
            final_green = net_green(
                green_value(self.bs, self.bpx, self.ll, self.lpx, bb_f, bl_f), self.comm)
            reward += final_green - self.prev_phi   # terminal correction
            info = {
                "final_green"   : final_green,
                "n_trades"      : self.n_trades,
                "shaping_total" : self.shaping_total,
                "bk_px_sum"     : self.bk_px_sum.copy(),
                "bk_px_n"       : self.bk_px_n.copy(),
                "ly_px_sum"     : self.ly_px_sum.copy(),
                "ly_px_n"       : self.ly_px_n.copy(),
            }

        return self._obs(), float(reward), done, False, info

print("DirectionalRaceEnv defined")

# ── quick sanity: one random-policy episode ───────────────────────────────────
_e = DirectionalRaceEnv(RACES_ALL); obs, _ = _e.reset(); done = False; steps = 0
while not done:
    a   = _e.action_space.sample()
    obs, r, done, _, info = _e.step(a); steps += 1
print(f"  sanity: T={steps}  final_green=${info.get('final_green',0):.2f}"
      f"  n_trades={info.get('n_trades',0)}")

# ── do-nothing baseline: action[MAX_RUNNERS]=-1 → stake_frac=0 ──────────────
_dn_rewards = []
for _ in range(20):
    obs, _ = _e.reset(); done = False; ep_r = 0
    while not done:
        a = np.zeros(MAX_RUNNERS + 1, dtype=np.float32)
        a[MAX_RUNNERS] = -1.0          # stake_frac = 0
        obs, r, done, _, info = _e.step(a); ep_r += r
    _dn_rewards.append(info.get("final_green", 0))
print(f"  do-nothing check: mean=${np.mean(_dn_rewards):.4f}  (should be ~0.00)")
'''))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 11  SAC + Metrics callback  *** ADAPTED from V56a cell 21 ***
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Cell 5 — SAC with equivariant encoder (V57)

Re-uses V56a's equivariant DeepSets extractor, updated for V57's observation dimensions
(`NF_RUN=32`, `MKT_FEATS=26`).  The action space is now 25-dim.

`ENT_COEF_FLOOR` is raised to 0.05 to prevent the entropy collapse observed in V56a."""))
cells.append(code('''\
import torch, torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class DeepSetsExtractor(BaseFeaturesExtractor):
    """Equivariant per-runner embedding + broadcast field context (V57).

    Identical to V56a Arm B but updated for NF_RUN=32 and len(MKT_FEATS)=26.
    The actor still emits 25 dims: 24 runner signals + 1 stake fraction.
    """
    def __init__(self, obs_space, per_dim=32, ctx_dim=64, features_dim=None):
        super().__init__(obs_space, MAX_RUNNERS * (per_dim + ctx_dim))
        self.per_dim, self.ctx_dim = per_dim, ctx_dim
        self.per = nn.Sequential(nn.Linear(NF_RUN, 128), nn.ReLU(),
                                 nn.Linear(128, per_dim), nn.ReLU())
        self.ctx = nn.Sequential(
            nn.Linear(per_dim * 2 + len(MKT_FEATS), ctx_dim), nn.ReLU())

    def forward(self, x):
        B = x.shape[0]; n = MAX_RUNNERS * NF_RUN
        R    = x[:, :n].view(B, MAX_RUNNERS, NF_RUN)
        M    = x[:, n : n + len(MKT_FEATS)]
        mask = x[:, n + len(MKT_FEATS):].unsqueeze(-1)
        h    = self.per(R) * mask
        mean = h.sum(1) / mask.sum(1).clamp(min=1)
        mx   = h.masked_fill(mask == 0, -1e30).max(1).values
        mx   = torch.nan_to_num(mx, neginf=0.0)
        c    = self.ctx(torch.cat([mean, mx, M], dim=1))
        out  = torch.cat([h, c.unsqueeze(1).expand(-1, MAX_RUNNERS, -1)], dim=-1)
        return (out * mask).reshape(B, -1)


class Metrics(BaseCallback):
    """Enforces entropy floor + ceiling; logs episode metrics."""
    def __init__(self): super().__init__(); self.rows = []

    def _on_step(self):
        if hasattr(self.model, "log_ent_coef") and self.model.log_ent_coef is not None:
            with torch.no_grad():
                self.model.log_ent_coef.clamp_(
                    float(np.log(ENT_COEF_FLOOR)),
                    float(np.log(ENT_COEF_CEIL)))
        for info in self.locals.get("infos", []):
            if "final_green" in info:
                self.rows.append(info)
        if len(self.rows) and len(self.rows) % 25 == 0 and self.rows[-1].get("_p") != 1:
            self.rows[-1]["_p"] = 1
            d  = pd.DataFrame(self.rows[-25:])
            ec = (self.model.log_ent_coef.exp().item()
                  if getattr(self.model, "log_ent_coef", None) is not None else float("nan"))
            green_pct = int(100 * (d["final_green"] > 0).mean())
            pnl_mean  = d["final_green"].mean()
            trades    = d["n_trades"].mean() if "n_trades" in d else float("nan")
            H = getattr(self.model.actor, "_last_entropy", float("nan"))
            ep = len(self.rows)
            print(f"  ep {ep:>5} | green {green_pct:>3}% | pnl ${pnl_mean:>8.2f}"
                  f" | trades {trades:>6.1f} | ent {ec:.4f} (H {H:.1f})")
        return True


def make_model(env, seed=SEED):
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0, gamma=GAMMA)
    extractor_kwargs = dict(per_dim=32, ctx_dim=64)
    policy_kwargs = dict(
        features_extractor_class  = DeepSetsExtractor,
        features_extractor_kwargs = extractor_kwargs,
        net_arch                  = [256, 256],
        log_std_init              = -2.0,
    )
    model = SAC(
        "MlpPolicy", venv,
        learning_rate   = 3e-4,
        buffer_size     = 50_000,
        learning_starts = 500,
        batch_size      = 256,
        tau             = 0.01,
        gamma           = GAMMA,
        train_freq      = 1,
        gradient_steps  = 4,
        ent_coef        = "auto",
        target_entropy  = TARGET_ENTROPY,
        policy_kwargs   = policy_kwargs,
        verbose         = 0,
        seed            = seed,
    )
    return model, venv

print("SAC + DeepSetsExtractor defined")
print(f"  action dims  : {MAX_RUNNERS + 1}")
print(f"  OBS_DIM      : {OBS_DIM}")
print(f"  features_dim : {MAX_RUNNERS * (32 + 64)}")
'''))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 12  Stage A  *** ADAPTED from V56a FastTrack cell 24 ***
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md("""\
## Stage A — Single-race overfit test

Identical gate to V56a FastTrack:
- random  ~  −$200   (untrained flailing)
- do-nothing = $0    (floor any competent policy must reach)
- trained ≥ −$1      → pass (found no-trade optimum)
- trained > $0       → real edge detected

With the V57 action spec, the do-nothing policy requires only one action dimension
(`action[24] → −1`, stake→0) rather than V56a's 24-dimensional coordination.
"""))
cells.append(code("""\
def rollout(env, policy=None, n=20, seed=0, deterministic=True):
    rng  = np.random.default_rng(seed)
    pnls = []
    for _ in range(n):
        obs, _ = env.reset(seed=int(rng.integers(1 << 31)))
        done = False; ep_r = 0
        while not done:
            if policy is None:
                a = env.action_space.sample()
            else:
                obs_t = np.array([obs])
                a, _  = policy.predict(obs_t, deterministic=deterministic)
                a     = a[0]
            obs, r, done, _, info = env.step(a)
            ep_r += r
        pnls.append(info.get("final_green", ep_r))
    return np.array(pnls)


def direction_audit(env, policy, n=10):
    bs = ls = bn = ln = 0.0; both = []
    for _ in range(n):
        obs, _ = env.reset(); done = False
        while not done:
            a, _ = policy.predict(np.array([obs]), deterministic=True)
            obs, r, done, _, info = env.step(a[0])
        bps = info["bk_px_sum"]; bpn = info["bk_px_n"]
        lps = info["ly_px_sum"]; lpn = info["ly_px_n"]
        bs += bps.sum(); bn += bpn.sum()
        ls += lps.sum(); ln += lpn.sum()
        m = (bpn > 0) & (lpn > 0)
        if m.any():
            both.extend(((bps[m]/bpn[m]) / (lps[m]/lpn[m])).tolist())
    bk_avg = bs / max(bn, 1); lk_avg = ls / max(ln, 1)
    dir_ok = "CORRECT (back high / lay low)" if bk_avg > lk_avg else "WRONG DIRECTION"
    print(f"  direction: back fills @ {bk_avg:.3f} vs lay fills @ {lk_avg:.3f} -> {dir_ok}")
    if both:
        edge = np.mean([x - 1 for x in both])
        pos  = int(100 * np.mean([x > 1 for x in both]))
        print(f"  two-sided edge on {len(both)} runner-eps: {edge:.4f} ({pos}% positive)")


# ── fps probe ─────────────────────────────────────────────────────────────────
_probe_env = DirectionalRaceEnv(RACES_ALL[:1])
_pm, _pv   = make_model(_probe_env)
_t0        = time.time()
_pm.learn(total_timesteps=2000)
_fps       = 2000 / (time.time() - _t0)
_T         = int(np.mean([r["T"] for r in RACES_ALL]))
_tot       = PASSES_PER_RACE * _T * len(RACES_ALL)
print(f"{len(RACES_ALL)} races x {PASSES_PER_RACE} passes x T~{_T} = {_tot:,} steps")
print(f"  MEASURED {_fps:.1f} fps -> full run ~{_tot/_fps/3600:.1f} h")
if _fps < 20:
    print("  *** SLOW — check device=cuda ***")
del _pm, _pv, _probe_env

# ── mechanical reward-direction check ─────────────────────────────────────────
_chk = DirectionalRaceEnv(RACES_ALL)
def _g(bs, bpx, ll, lpx, bb, bl, comm=COMMISSION):
    return net_green(green_value(np.array([bs]), np.array([bpx]),
                                 np.array([ll]), np.array([lpx]),
                                 np.array([bb]), np.array([bl])), comm)
assert _g(5, 6.0, 0, 0, 5.0, 5.2)  > 0, "back then shorten should be +ve"
assert _g(5, 5.0, 0, 0, 6.0, 6.2)  < 0, "back then drift should be -ve"
assert _g(0, 0, 5, 5.0, 6.0, 6.2)  > 0, "lay then drift should be +ve"
assert _g(0, 0, 5, 6.0, 5.0, 5.2)  < 0, "lay then shorten should be -ve"
print("  reward direction confirmed: BACK HIGH / LAY LOW is what pays")

# ── do-nothing baseline check ─────────────────────────────────────────────────
_dn = []
for _ in range(30):
    obs, _ = _chk.reset(); done = False
    while not done:
        a = np.zeros(MAX_RUNNERS + 1, dtype=np.float32); a[-1] = -1.0
        obs, r, done, _, info = _chk.step(a)
    _dn.append(info.get("final_green", 0))
print(f"  do-nothing baseline: mean=${np.mean(_dn):.4f}  std=${np.std(_dn):.4f}")

# ── Stage A: fresh model per race ─────────────────────────────────────────────
STAGE_A_RESULTS = []

for race_idx, F in enumerate(RACES_ALL):
    name = str(F.get("name", f"race_{race_idx}")).split("/")[-1]
    print()
    print("#" * 72)
    print(f"# RACE {race_idx+1}/{len(RACES_ALL)}  {name}"
          f"  T={F['T']} R={F['R']} comm={F.get('commission',COMMISSION):.0%}")
    print("#" * 72)

    race_list = [F]
    env_r     = DirectionalRaceEnv(race_list, seed=SEED + race_idx)

    # Random baseline
    rand_pnl = rollout(env_r, policy=None, n=20, seed=SEED)
    print(f"  random:     pnl ${rand_pnl.mean():.2f}  | do-nothing: $0.00")

    # Train: fresh model, fresh buffer
    model, venv = make_model(DirectionalRaceEnv(race_list, seed=SEED + race_idx),
                             seed=SEED + race_idx)
    cb = Metrics()

    model.learn(total_timesteps=PASSES_PER_RACE * F["T"], callback=cb, reset_num_timesteps=True)

    # Eval trained
    eval_env = DirectionalRaceEnv(race_list, seed=SEED + race_idx + 1000)
    trained_pnl = rollout(eval_env, policy=model, n=20, deterministic=True)
    gap = trained_pnl.mean() - 0.0   # vs do-nothing
    status = "PASS" if trained_pnl.mean() >= -1.0 else "FAIL"

    print(f"  trained: pnl ${trained_pnl.mean():.2f}"
          f" (vs do-nothing $0.00 -> gap {gap:+.2f}) | [{status}]")
    direction_audit(eval_env, model, n=10)

    # Stake-fraction distribution
    _stakes = []
    obs, _ = eval_env.reset(); done = False
    while not done:
        a, _ = model.predict(np.array([obs]), deterministic=True)
        sf = (float(a[0, MAX_RUNNERS]) + 1.0) / 2.0
        _stakes.append(sf)
        obs, _, done, _, _ = eval_env.step(a[0])
    print(f"  stake_frac: mean={np.mean(_stakes):.3f}  p10={np.percentile(_stakes,10):.3f}"
          f"  p90={np.percentile(_stakes,90):.3f}  near-zero(<0.05): "
          f"{int(100*np.mean(np.array(_stakes)<0.05))}%")

    STAGE_A_RESULTS.append(dict(
        race=name, random=rand_pnl.mean(), trained=trained_pnl.mean(),
        gap=gap, status=status))

    del model, venv, cb, env_r, eval_env

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("  STAGE A SUMMARY")
print("=" * 72)
passes = sum(1 for r in STAGE_A_RESULTS if r["status"] == "PASS")
print(f"  Races tested : {len(STAGE_A_RESULTS)}")
print(f"  PASS         : {passes}/{len(STAGE_A_RESULTS)}")
for r in STAGE_A_RESULTS:
    print(f"  [{r['status']:4s}]  {r['race'][-30:]:30s}  "
          f"random ${r['random']:>8.2f}  trained ${r['trained']:>8.2f}  gap {r['gap']:>+8.2f}")
print()
if passes >= 1:
    print("  ✓ Stage A gate PASSED — proceed to Stage B (difficulty curriculum)")
else:
    print("  ✗ Stage A gate FAILED — diagnose before proceeding")
    print("    Checks:")
    print("    1. stake_frac near-zero% high? → agent found do-nothing (correct)")
    print("    2. stake_frac near-zero% low?  → entropy still collapsing, raise ENT_COEF_FLOOR")
    print("    3. direction wrong?            → check fill-price convention (bb vs bl)")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# assemble notebook
# ─────────────────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": cells
}

out_path = "/home/user/AHR_RL/MarketMaking_V57_ActionSpec.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")
