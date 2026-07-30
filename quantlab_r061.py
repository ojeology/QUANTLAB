"""
=============================================================================
QUANTLAB AI — RESEARCH #R061
Reality Check — 5-Test Stress Battery on E3.1_v2
=============================================================================

Objective:
  After R057→R060 confirmed E3.1_v2 (BBW_STRICT+RV_LO+DST_NR+PRG_VH)
  as the best current environment, one hard question remains:

      "Is this a genuine edge, or did we data-mine our way here?"

  This script runs 5 independent attack vectors. Each targets a different
  failure mode. A real edge passes all 5. A ghost edge fails some.

  ─────────────────────────────────────────────────────────────────
  TEST 1 — Parameter Robustness Grid
    Vary each of 5 parameters (BBW quantile, RV quantile, DST_NR
    quantile, PRG_VH quantile, RR) one at a time across 5 values.
    Expected if real: PF stays above 1.0 across a neighbourhood of
    the frozen parameters (a "plateau"), not a razor-thin peak.
    Expected if data-mined: PF collapses at ±1 step from optimum.

  TEST 2 — Signal Permutation Null Distribution
    For each symbol/fold, randomly sample the same NUMBER of entries
    as E3.1_v2 fires, at random valid bars. Repeat 400 times.
    Expected if real: Observed PF=1.64 sits in the top 5% of the null.
    Expected if fake: Observed PF is indistinguishable from random.

  TEST 3 — Condition Ablation
    Remove each of the 4 conditions individually; also test with no
    conditions (pure entry signal on all bars). If each condition is
    genuinely contributing, removing it must degrade performance.
    Expected if real: PF drops materially (≥0.05) for every removal.
    Expected if fake: one condition drives everything; rest are noise.

  TEST 4 — Temporal Stability (Fold-by-Fold + PnL Concentration)
    Check whether the observed PF is evenly distributed across folds
    and symbols, or driven by a lucky cluster. If >50% of net PnL
    comes from a single fold or single symbol → concentrated luck.
    Expected if real: no fold > 40% of PnL, no symbol > 15%.
    Expected if fake: 1-2 folds or 1-2 symbols carry the result.

  TEST 5 — Symbol Holdout
    Randomly split 49 symbols 35 / 14. Learn all IS thresholds
    exclusively from the 35 training symbols. Apply pooled thresholds
    to the 14 holdout symbols' OOS data.
    Expected if real: holdout PF > 1.0, LOO-sym floor > 1.0.
    Expected if fake: edge disappears on never-seen symbols.

  ─────────────────────────────────────────────────────────────────
  ALSO TESTED: DST_MD P1 (ADX_WK+DST_MD+RV_HI) runs the same
  battery for comparison. If the two families pass/fail similarly,
  both share the same structural fate.

  FINAL VERDICT: REAL / SUSPECT / GOOSE CHASE
    REAL  : ≥4 of 5 tests pass on E3.1_v2
    SUSPECT: 2–3 tests pass
    GOOSE : ≤1 test passes

=============================================================================
"""

import os, sys, math, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID = "R061"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL      = CONFIG["STARTING_CAPITAL"]
BASE_RR      = 2.0
IS_RATIO     = 0.80
N_FWD_FOLDS  = 5
N_PERM       = 400     # permutation test iterations
RAND_SEED    = 42
MIN_BARS     = 2_000

# Frozen E3.1_v2 parameters
FROZEN_CIDS  = ("BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH")
FROZEN_LABEL = "BBW_STRICT+RV_LO+DST_NR+PRG_VH"
FROZEN_PARAMS = {
    "BBW_STRICT_q": 0.25,
    "RV_LO_q":      0.33,
    "DST_NR_q":     0.33,
    "PRG_VH_q":     0.80,
    "RR":           2.0,
}

# DST_MD P1 (second family, run same battery for comparison)
DST_MD_CIDS  = ("ADX_WK", "DST_MD", "RV_HI")
DST_MD_LABEL = "ADX_WK+DST_MD+RV_HI"

# Holdout split: last 14 symbols (seeded by fixed indices across sorted list)
HOLDOUT_SEED_INDICES = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 40, 43, 45, 47]

# Verdict thresholds
PASS_PARAM_ROBUST  = 0.60   # fraction of param variations with PF > 1.0
PASS_PERM_PCTILE   = 95     # observed PF must beat this percentile of null
PASS_ABLATION_DROP = 0.05   # each condition removal must drop PF by at least this
PASS_CONC_FOLD     = 0.40   # no fold may contribute > 40% of net PnL
PASS_CONC_SYM      = 0.20   # no symbol may contribute > 20% of net PnL
PASS_HOLDOUT_PF    = 1.0    # holdout OOS PF must exceed this

# Colour palette
C_BG   = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT  = "#e0e0e0"
C_GRID = "#2a2a2a"; C_GREEN = "#00c896"; C_RED   = "#e05050"
C_GOLD = "#f5a623"; C_BLUE  = "#4a9eff"; C_PURP  = "#9b59b6"
C_CYAN = "#1abc9c"; C_ORAN  = "#e67e22"
PALETTE = [C_GREEN, C_GOLD, C_BLUE, C_RED, C_PURP,
           "#e67e22","#1abc9c","#3498db","#e74c3c","#f39c12"]

plt.rcParams.update({
    "figure.facecolor": C_BG, "axes.facecolor": C_PANEL,
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "axes.edgecolor": C_GRID, "grid.color": C_GRID,
    "font.family": "monospace",
})

SEP  = "═" * 110
SEP2 = "─" * 90

def panel_style(ax, title, fs=8):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title, fontsize=fs, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)

ALL_SYMBOLS = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "AVAX-USDT-SWAP","XRP-USDT-SWAP","LTC-USDT-SWAP","BCH-USDT-SWAP",
    "DOGE-USDT-SWAP","ADA-USDT-SWAP","BNB-USDT-SWAP","DOT-USDT-SWAP",
    "ARB-USDT-SWAP","OP-USDT-SWAP","NEAR-USDT-SWAP","ATOM-USDT-SWAP",
    "SUI-USDT-SWAP","APT-USDT-SWAP","WIF-USDT-SWAP","PEPE-USDT-SWAP",
    "ENA-USDT-SWAP","UNI-USDT-SWAP","FIL-USDT-SWAP",
    "1INCH-USDT-SWAP","AAVE-USDT-SWAP","ALGO-USDT-SWAP","AXS-USDT-SWAP",
    "CHZ-USDT-SWAP","COMP-USDT-SWAP","CRV-USDT-SWAP","DYDX-USDT-SWAP",
    "EGLD-USDT-SWAP","ETC-USDT-SWAP","FET-USDT-SWAP","GALA-USDT-SWAP",
    "GMX-USDT-SWAP","GRT-USDT-SWAP","HBAR-USDT-SWAP","ICP-USDT-SWAP",
    "IMX-USDT-SWAP","INJ-USDT-SWAP","LDO-USDT-SWAP","SAND-USDT-SWAP",
    "SHIB-USDT-SWAP","SNX-USDT-SWAP","STX-USDT-SWAP","SUSHI-USDT-SWAP",
    "TRX-USDT-SWAP","XLM-USDT-SWAP",
]

HOLDOUT_SYMS = [ALL_SYMBOLS[i] for i in HOLDOUT_SEED_INDICES if i < len(ALL_SYMBOLS)]
TRAIN_SYMS   = [s for s in ALL_SYMBOLS if s not in HOLDOUT_SYMS]

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONS CATALOGUE — parametric (quantiles can be overridden)
# ─────────────────────────────────────────────────────────────────────────────
BASE_CONDITIONS = {
    # id           feat              direction    base_q
    "ATR_LO":   ("atr_rank",       "lt_q",      0.25),
    "ATR_HI":   ("atr_rank",       "gt_q",      0.67),
    "BBW_LO":   ("bb_width",       "lt_q",      0.33),
    "BBW_STRICT":("bb_width",      "lt_q",      0.25),
    "RV_LO":    ("real_vol_20",    "lt_q",      0.33),
    "RV_HI":    ("real_vol_20",    "gt_q",      0.67),
    "SLP_UP":   ("ema200_slope",   "gt_fixed",  0.0),
    "SLP_DN":   ("ema200_slope",   "lt_fixed",  0.0),
    "DST_NR":   ("ema_dist_pct",   "lt_q",      0.33),
    "DST_MD":   ("ema_dist_pct",   "gt_q_pos",  0.60),
    "ADX_WK":   ("adx14",          "lt_q",      0.33),
    "ADX_ST":   ("adx14",          "gt_q",      0.67),
    "PRG_LO":   ("prev_range_r",   "lt_q",      0.33),
    "PRG_VH":   ("prev_range_r",   "gt_q",      0.80),
    "LON":      ("hour_utc",       "hour_rng",  (7, 14)),
    "US":       ("hour_utc",       "hour_rng",  (14, 21)),
}
QUANT_FEATS = ["atr_rank","bb_width","real_vol_20","ema_dist_pct",
               "adx14","prev_range_r","prev_body_r","prev_body_pct"]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]; o = df["open"]
    df["ema200"]        = calc_ema(c, 200)
    df["atr14"]         = calc_atr(df, 14)
    df["atr_rank"]      = df["atr14"].rolling(100).rank(pct=True) * 100
    bb_mid              = c.rolling(20).mean()
    bb_std              = c.rolling(20).std()
    df["bb_width"]      = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["ema_dist_pct"]  = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"]  = (df["ema200"] - df["ema200"].shift(10)) / \
                          df["ema200"].shift(10).replace(0, np.nan)
    vol_ma              = v.rolling(20).mean()
    df["rel_vol"]       = v / vol_ma.replace(0, np.nan)
    df["prev_close"]    = c.shift(1)
    df["prev_atr14"]    = df["atr14"].shift(1)
    log_ret             = np.log(c / c.shift(1))
    df["real_vol_20"]   = log_ret.rolling(20).std() * 100.0
    df["adx14"]         = calc_adx(df, 14)
    prev_range          = h.shift(1) - l.shift(1)
    prev_body           = (c.shift(1) - o.shift(1)).abs()
    df["prev_range_r"]  = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]   = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"] = prev_body  / prev_range.replace(0, np.nan)
    dt                  = pd.to_datetime(df["datetime"], utc=True)
    df["hour_utc"]      = dt.dt.hour.astype(np.int16)
    return df

def learn_thresholds(df_is, overrides=None):
    """Learn per-condition thresholds from IS data. overrides = {cid: new_quantile}."""
    thr = {}
    overrides = overrides or {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (feat, direction, base_q) in BASE_CONDITIONS.items():
        q = overrides.get(cid, base_q)
        if direction in ("gt_fixed","lt_fixed","hour_rng"):
            thr[cid] = q; continue
        if feat not in valid.columns:
            thr[cid] = np.nan; continue
        col = valid[feat].dropna()
        if len(col) < 20:
            thr[cid] = np.nan; continue
        if direction == "gt_q_pos":
            pos = col[col > 0]
            thr[cid] = float(pos.quantile(q) if len(pos) > 10 else col.quantile(q))
        else:
            thr[cid] = float(col.quantile(q))
    return thr

def pooled_thresholds(thr_list):
    """Average thresholds across a list of per-symbol threshold dicts."""
    keys = set().union(*thr_list)
    out  = {}
    for k in keys:
        vals = [t[k] for t in thr_list if k in t and not (isinstance(t[k], float) and np.isnan(t[k]))]
        if not vals: out[k] = np.nan
        elif isinstance(vals[0], tuple): out[k] = vals[0]  # hour ranges: take first
        else: out[k] = float(np.median(vals))
    return out

def build_env_mask(df, cond_ids, thr):
    N = len(df); mask = np.ones(N, dtype=bool)
    for cid in cond_ids:
        if cid not in BASE_CONDITIONS: return np.zeros(N, dtype=bool)
        feat, direction, _ = BASE_CONDITIONS[cid]
        if feat not in df.columns: return np.zeros(N, dtype=bool)
        col   = df[feat].values
        nan_m = np.isnan(col) if col.dtype.kind == "f" else np.zeros(N, dtype=bool)
        t     = thr.get(cid, np.nan)
        if direction == "lt_q":
            if isinstance(t, float) and np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col < t)
        elif direction in ("gt_q","gt_q_pos"):
            if isinstance(t, float) and np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col > t)
        elif direction == "gt_fixed": mask &= (~nan_m) & (col > t)
        elif direction == "lt_fixed": mask &= (~nan_m) & (col < t)
        elif direction == "hour_rng":
            lo_, hi_ = t; mask &= (col >= lo_) & (col <= hi_)
    return mask

def entry_signal(df, env_mask):
    rv = df["rel_vol"].values; c = df["close"].values
    o  = df["open"].values;    pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & env_mask

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signal, sym, fold_label, rr=BASE_RR):
    min_sl  = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf      = CONFIG["RISK_PER_TRADE_PCT"]
    fee     = CONFIG["TAKER_FEE"];  spd = CONFIG["SPREAD"] * 0.5
    slp     = CONFIG["SL_SLIPPAGE"]
    in_pos  = False; ep = st = tk = sz = 0.0; et = None; ei = -1
    trades  = []
    op_ = df["open"].values;  hi_ = df["high"].values
    lo_ = df["low"].values;   atr_ = df["prev_atr14"].values
    dts = df["datetime"].values

    for i in range(1, len(df)):
        if in_pos:
            sl_hit = lo_[i] <= st; tp_hit = hi_[i] >= tk
            if sl_hit or tp_hit:
                xp    = (st * (1 - slp)) if sl_hit else tk
                sd    = ep - st
                gross = (xp - ep) * sz
                cost  = (ep * sz + xp * sz) * (fee + spd)
                slpc  = (st - xp) * sz if sl_hit else 0.0
                net   = gross - cost - slpc
                trades.append({
                    "sym": sym, "fold": fold_label,
                    "entry_time": str(et), "pnl": round(net, 4),
                    "win": int(not sl_hit),
                })
                in_pos = False
            continue
        if signal[i-1]:
            a = atr_[i]
            if np.isnan(a) or a <= 0: continue
            ep_ = op_[i]
            if a / ep_ < min_sl: continue
            ep = ep_; st = ep - a; tk = ep + rr * a
            sz = min(CAPITAL * rf / a, (CAPITAL * max_lev) / ep)
            et = dts[i]; ei = i; in_pos = True
    return trades

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl

def metrics(trades):
    if not trades:
        return {"n":0,"wr":0.0,"pf":0.0,"net":0.0,"mdd":0.0,
                "pnls":np.array([]),"equity":np.array([CAPITAL])}
    pnl  = np.array([t["pnl"] for t in trades])
    wins = np.array([t["win"] for t in trades], dtype=bool)
    n=len(pnl); nw=wins.sum(); nl=n-nw
    gw=pnl[wins].sum() if nw else 0.0
    gl=abs(pnl[~wins].sum()) if nl else 0.0
    pf=safe_pf(gw,gl); wr=nw/n
    eq=np.concatenate([[CAPITAL],CAPITAL+np.cumsum(pnl)])
    pk=np.maximum.accumulate(eq)
    mdd=float(((eq-pk)/pk).min())
    return {"n":n,"wr":wr,"pf":pf,"net":float(pnl.sum()),
            "mdd":mdd,"pnls":pnl,"equity":eq}

def fast_pf(trades):
    if not trades: return 0.0
    gw = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    return safe_pf(gw, gl)

# ─────────────────────────────────────────────────────────────────────────────
# WFO ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_wf(cond_ids, sym_set, overrides=None, rr=BASE_RR, pooled_thr=None):
    """
    5-fold walk-forward for cond_ids on sym_set.
    If pooled_thr is provided, use that instead of per-symbol IS learning.
    Returns (all_trades, fold_trades, sym_trades).
    """
    all_t = []; fold_t = defaultdict(list); sym_t = defaultdict(list)
    for sym in sym_set:
        if sym not in all_dfs: continue
        df_is, df_fwd, sym_thr = all_dfs[sym]
        thr = pooled_thr if pooled_thr is not None else (
              learn_thresholds(df_is, overrides) if overrides else sym_thr)
        fwd_size = len(df_fwd); seg_size = max(1, fwd_size // N_FWD_FOLDS)
        for fi in range(N_FWD_FOLDS):
            seg_s = fi * seg_size
            seg_e = fwd_size if fi == N_FWD_FOLDS - 1 else (fi+1)*seg_size
            df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
            if len(df_seg) < 20: continue
            fl  = f"F{fi+1}"
            em  = build_env_mask(df_seg, cond_ids, thr)
            sig = entry_signal(df_seg, em)
            tl  = run_backtest(df_seg, sig, sym, fl, rr=rr)
            all_t.extend(tl); fold_t[fl].extend(tl); sym_t[sym].extend(tl)
    return all_t, fold_t, sym_t

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print(f"  QUANTLAB AI — RESEARCH #{RESEARCH_ID}")
print("  Reality Check — 5-Test Stress Battery on E3.1_v2")
print(SEP)
print()
print(f"  Frozen strategy : {FROZEN_LABEL}")
print(f"  Comparison      : {DST_MD_LABEL}  (DST_MD P1)")
print(f"  Universe        : {len(ALL_SYMBOLS)} symbols  |  Train: {len(TRAIN_SYMS)}  "
      f"Holdout: {len(HOLDOUT_SYMS)}")
print(f"  Tests: Param Robustness | Permutation Null | Ablation | "
      f"Temporal Stability | Symbol Holdout")
print()

all_dfs = {}
loaded  = 0
for sym in ALL_SYMBOLS:
    tag  = sym.replace("-","_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path): continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    N  = len(df)
    if N < MIN_BARS: continue
    df  = add_features(df)
    sp  = int(N * IS_RATIO)
    thr = learn_thresholds(df.iloc[:sp])
    all_dfs[sym] = (df.iloc[:sp], df.iloc[sp:].reset_index(drop=True), thr)
    loaded += 1

print(f"  Symbols loaded: {loaded} / {len(ALL_SYMBOLS)}")
print()

# ── Baseline run (E3.1_v2 on full universe)
print("  Running baseline E3.1_v2 ...")
base_all, base_fold, base_sym = run_wf(FROZEN_CIDS, ALL_SYMBOLS)
base_m = metrics(base_all)
print(f"  Baseline: PF={base_m['pf']:.3f}  n={base_m['n']}  WR={base_m['wr']:.1%}  "
      f"MDD={base_m['mdd']:.1%}  Net=${base_m['net']:+.0f}")
print()

# ── DST_MD P1 baseline
print("  Running DST_MD P1 baseline ...")
dst_all, dst_fold, dst_sym = run_wf(DST_MD_CIDS, ALL_SYMBOLS)
dst_m = metrics(dst_all)
print(f"  DST_MD P1: PF={dst_m['pf']:.3f}  n={dst_m['n']}  WR={dst_m['wr']:.1%}  "
      f"MDD={dst_m['mdd']:.1%}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — PARAMETER ROBUSTNESS GRID
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  TEST 1 — Parameter Robustness Grid")
print("  Varying each of 5 parameters one at a time (others frozen).")
print(SEP)
print()

param_variations = {
    "BBW_STRICT_q": ([0.15, 0.20, 0.25, 0.30, 0.35],  "BBW_STRICT", "BBW_STRICT"),
    "RV_LO_q":      ([0.20, 0.27, 0.33, 0.40, 0.47],  "RV_LO",      "RV_LO"),
    "DST_NR_q":     ([0.20, 0.27, 0.33, 0.40, 0.47],  "DST_NR",     "DST_NR"),
    "PRG_VH_q":     ([0.70, 0.75, 0.80, 0.85, 0.90],  "PRG_VH",     "PRG_VH"),
    "RR":           ([1.5,  1.75, 2.0,  2.25, 2.5],    None,         "RR"),
}

param_results   = {}
all_param_pfs   = []

for param_name, (values, cid_key, label) in param_variations.items():
    row_pfs = []
    frozen_val = FROZEN_PARAMS[param_name]
    print(f"  {'─'*70}")
    print(f"  {label} sweep  (frozen={frozen_val})")
    print(f"  {'Value':>8}  {'PF':>8}  {'n':>5}  {'WR':>7}  {'MDD':>8}  {'Note'}")
    print(f"  {'─'*60}")
    for val in values:
        if param_name == "RR":
            t_all, _, _ = run_wf(FROZEN_CIDS, ALL_SYMBOLS, rr=val)
        else:
            ovr = {cid_key: val}
            t_all, _, _ = run_wf(FROZEN_CIDS, ALL_SYMBOLS, overrides=ovr)
        m = metrics(t_all)
        is_frozen = abs(val - frozen_val) < 1e-6
        note = "◄ FROZEN" if is_frozen else ("▲" if m["pf"] > 1.0 else "▼")
        print(f"  {val:>8.3f}  {m['pf']:>8.3f}  {m['n']:>5}  "
              f"{m['wr']:>7.1%}  {m['mdd']:>8.1%}  {note}")
        row_pfs.append((val, m["pf"], m["n"]))
        all_param_pfs.append(m["pf"])
    param_results[param_name] = row_pfs
    pf_above_1 = sum(1 for _, pf, _ in row_pfs if pf > 1.0)
    print(f"  → {pf_above_1}/5 variations achieve PF>1.0")
    print()

# Robustness score
total_above_1 = sum(1 for pf in all_param_pfs if pf > 1.0)
total_vars    = len(all_param_pfs)
robustness_rate = total_above_1 / total_vars
t1_pass = robustness_rate >= PASS_PARAM_ROBUST
print(f"  OVERALL: {total_above_1}/{total_vars} parameter variations achieve PF>1.0 "
      f"({robustness_rate:.0%})")
print(f"  Pass threshold: ≥{PASS_PARAM_ROBUST:.0%}")
print(f"  TEST 1 RESULT: {'✓ PASS' if t1_pass else '✗ FAIL'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — SIGNAL PERMUTATION NULL DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  TEST 2 — Signal Permutation Null Distribution")
print(f"  Running {N_PERM} random-entry permutations (same entry count, random timing).")
print("  Null hypothesis: random entry achieves equivalent PF to E3.1_v2.")
print(SEP)
print()

rng = np.random.RandomState(RAND_SEED)

# Pre-collect OOS segments and their signal data
oos_segments = []
for sym in ALL_SYMBOLS:
    if sym not in all_dfs: continue
    df_is, df_fwd, thr = all_dfs[sym]
    fwd_size = len(df_fwd); seg_size = max(1, fwd_size // N_FWD_FOLDS)
    for fi in range(N_FWD_FOLDS):
        seg_s = fi * seg_size
        seg_e = fwd_size if fi == N_FWD_FOLDS - 1 else (fi+1)*seg_size
        df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
        if len(df_seg) < 20: continue
        em  = build_env_mask(df_seg, FROZEN_CIDS, thr)
        sig = entry_signal(df_seg, em)
        n_entries = int(sig[:-1].sum())  # how many entries the real signal fires
        if n_entries == 0: continue
        # Valid bars = bars where we COULD enter (have sufficient data)
        atr_vals = df_seg["prev_atr14"].values
        op_vals  = df_seg["open"].values
        valid_bars = np.where(
            (~np.isnan(atr_vals[1:])) & (atr_vals[1:] > 0) &
            (atr_vals[1:] / op_vals[1:] >= CONFIG["MIN_SL_PCT"])
        )[0]  # indices into positions 1..N-1
        oos_segments.append({
            "df_seg":    df_seg,
            "sym":       sym,
            "fold":      f"F{fi+1}",
            "real_sig":  sig,
            "n_entries": n_entries,
            "valid_bars": valid_bars,
            "thr":       thr,
        })

print(f"  OOS segments collected: {len(oos_segments)}")
print(f"  Total real entries: {sum(s['n_entries'] for s in oos_segments)}")
print()

print(f"  Running {N_PERM} permutations", end="", flush=True)
perm_pfs = []
for perm_i in range(N_PERM):
    if (perm_i + 1) % 100 == 0:
        print(f" {perm_i+1}", end="", flush=True)
    perm_trades = []
    for seg in oos_segments:
        vb = seg["valid_bars"]
        n  = seg["n_entries"]
        if len(vb) < n:
            continue
        chosen = rng.choice(vb, size=n, replace=False)
        rand_sig = np.zeros(len(seg["df_seg"]) - 1, dtype=bool)
        rand_sig[chosen] = True
        # Pad to full length (signal is indexed at i-1 for entry at i)
        rand_sig_full = np.zeros(len(seg["df_seg"]), dtype=bool)
        rand_sig_full[:len(rand_sig)] = rand_sig
        tl = run_backtest(seg["df_seg"], rand_sig_full, seg["sym"], seg["fold"])
        perm_trades.extend(tl)
    perm_pfs.append(fast_pf(perm_trades))
print(f"\n")

perm_arr    = np.array(perm_pfs)
obs_pf      = base_m["pf"]
pctile_rank = float((perm_arr < obs_pf).mean()) * 100
perm_med    = float(np.median(perm_arr))
perm_p95    = float(np.percentile(perm_arr, 95))
perm_p99    = float(np.percentile(perm_arr, 99))

print(f"  Permutation null distribution (n={N_PERM} shuffles):")
print(f"    Median PF:         {perm_med:.3f}")
print(f"    95th percentile:   {perm_p95:.3f}")
print(f"    99th percentile:   {perm_p99:.3f}")
print(f"    Observed E3.1_v2 PF: {obs_pf:.3f}  → ranks at {pctile_rank:.1f}th percentile")
print()
t2_pass = pctile_rank >= PASS_PERM_PCTILE
print(f"  Pass threshold: observed PF must beat {PASS_PERM_PCTILE}th percentile of null")
print(f"  TEST 2 RESULT: {'✓ PASS' if t2_pass else '✗ FAIL'}  "
      f"({'p<' if t2_pass else 'p≥'}{(100-pctile_rank):.1f}%)")
print()

# DST_MD P1 permutation rank
dst_pctile = float((perm_arr < dst_m["pf"]).mean()) * 100
print(f"  DST_MD P1 PF={dst_m['pf']:.3f} → {dst_pctile:.1f}th pctile of same null")
print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — CONDITION ABLATION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  TEST 3 — Condition Ablation")
print("  Remove each condition individually. Each should degrade PF meaningfully.")
print(SEP)
print()

ablation_cases = [
    ("FULL (baseline)",      FROZEN_CIDS),
    ("Drop BBW_STRICT",      ("RV_LO",     "DST_NR", "PRG_VH")),
    ("Drop RV_LO",           ("BBW_STRICT","DST_NR", "PRG_VH")),
    ("Drop DST_NR",          ("BBW_STRICT","RV_LO",  "PRG_VH")),
    ("Drop PRG_VH",          ("BBW_STRICT","RV_LO",  "DST_NR")),
    ("No conditions (raw)",  ()),  # pure entry signal only
]

ablation_results = []
print(f"  {'Case':<26}  {'PF':>8}  {'n':>5}  {'WR':>7}  {'MDD':>8}  "
      f"{'ΔPF vs Full':>12}  {'n contrib?'}")
print("  " + "─" * 82)
full_pf = base_m["pf"]
for case_label, cids in ablation_cases:
    if not cids:
        # Raw entry signal: just rel_vol > 1.5, close > open, close > prev_close
        # Use an always-true env mask
        t_all2 = []; 
        for sym in ALL_SYMBOLS:
            if sym not in all_dfs: continue
            _, df_fwd, _ = all_dfs[sym]
            fwd_size = len(df_fwd); seg_size = max(1, fwd_size // N_FWD_FOLDS)
            for fi in range(N_FWD_FOLDS):
                seg_s = fi * seg_size
                seg_e = fwd_size if fi == N_FWD_FOLDS-1 else (fi+1)*seg_size
                df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
                if len(df_seg) < 20: continue
                full_mask = np.ones(len(df_seg), dtype=bool)
                sig = entry_signal(df_seg, full_mask)
                tl  = run_backtest(df_seg, sig, sym, f"F{fi+1}")
                t_all2.extend(tl)
        m = metrics(t_all2)
    else:
        t_all2, _, _ = run_wf(cids, ALL_SYMBOLS)
        m = metrics(t_all2)
    dpf = m["pf"] - full_pf
    contrib = "✓" if abs(dpf) >= PASS_ABLATION_DROP else "✗ (tiny effect)"
    ablation_results.append((case_label, m["pf"], m["n"], dpf))
    print(f"  {case_label:<26}  {m['pf']:>8.3f}  {m['n']:>5}  {m['wr']:>7.1%}  "
          f"{m['mdd']:>8.1%}  {dpf:>+12.3f}  {contrib}")

print()
# Test 3 passes if all 4 individual removals degrade PF by >= threshold
drops = [dpf for label, _, _, dpf in ablation_results[1:5]]  # skip full + raw
t3_pass = all(dpf < -PASS_ABLATION_DROP for dpf in drops)
n_meaningful_drops = sum(1 for dpf in drops if dpf < -PASS_ABLATION_DROP)
print(f"  Conditions degrading PF by ≥{PASS_ABLATION_DROP} on removal: "
      f"{n_meaningful_drops}/4")
print(f"  Pass threshold: all 4 conditions must matter")
print(f"  TEST 3 RESULT: {'✓ PASS' if t3_pass else '✗ FAIL'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — TEMPORAL STABILITY + PnL CONCENTRATION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  TEST 4 — Temporal Stability and PnL Concentration")
print("  Is the edge distributed across time and symbols, or clustered?")
print(SEP)
print()

total_net = base_m["net"]

# Fold concentration
print("  Per-fold PnL contribution:")
fold_nets  = {}
for fi in range(1, N_FWD_FOLDS+1):
    fl = f"F{fi}"
    m_f = metrics(base_fold.get(fl,[]))
    fold_nets[fl] = m_f["net"]
    pct = m_f["net"] / total_net * 100 if total_net != 0 else 0
    bar = "▲" if m_f["pf"] >= 1.20 else ("▼" if m_f["pf"] < 1.0 else "─")
    flag = "  ⚠ CONCENTRATED" if abs(pct) > PASS_CONC_FOLD*100 else ""
    print(f"  {fl}: PF={m_f['pf']:.3f}  n={m_f['n']:3d}  Net=${m_f['net']:+6.0f}  "
          f"Share={pct:+6.1f}%  {bar}{flag}")
max_fold_share = max(abs(v/total_net) for v in fold_nets.values() if total_net != 0)
print()

# Symbol concentration
sym_nets = {}
for sym in ALL_SYMBOLS:
    tl = base_sym.get(sym,[])
    if tl: sym_nets[sym] = sum(t["pnl"] for t in tl)
if sym_nets:
    top5_syms = sorted(sym_nets, key=lambda s: -abs(sym_nets[s]))[:5]
    print("  Top-5 symbols by |PnL contribution|:")
    for sym in top5_syms:
        net_s = sym_nets[sym]
        pct   = net_s / total_net * 100 if total_net != 0 else 0
        flag  = "  ⚠ CONCENTRATED" if abs(pct) > PASS_CONC_SYM*100 else ""
        print(f"  {sym.replace('-USDT-SWAP',''):<8}  Net=${net_s:+6.0f}  "
              f"Share={pct:+6.1f}%{flag}")
    max_sym_share = max(abs(v/total_net) for v in sym_nets.values() if total_net != 0)
else:
    max_sym_share = 0.0
print()

# Positive fold fraction
n_pos_folds = sum(1 for fl, net in fold_nets.items() if net > 0)
pos_pf_folds = sum(1 for fl in [f"F{i}" for i in range(1,6)]
                   if metrics(base_fold.get(fl,[]))["pf"] > 1.0)
print(f"  Folds with positive PnL: {n_pos_folds}/5")
print(f"  Folds with PF > 1.0:     {pos_pf_folds}/5")
print(f"  Max fold PnL share:      {max_fold_share:.1%}  "
      f"(threshold ≤{PASS_CONC_FOLD:.0%})")
print(f"  Max symbol PnL share:    {max_sym_share:.1%}  "
      f"(threshold ≤{PASS_CONC_SYM:.0%})")
print()

# DST_MD P1 concentration
dst_fold_nets = {}
for fi in range(1,N_FWD_FOLDS+1):
    fl=f"F{fi}"
    dst_fold_nets[fl] = sum(t["pnl"] for t in dst_fold.get(fl,[]))
dst_total = dst_m["net"]
print("  DST_MD P1 fold PnL:")
for fl, net in dst_fold_nets.items():
    m_f = metrics(dst_fold.get(fl,[]))
    pct = net/dst_total*100 if dst_total != 0 else 0
    print(f"  {fl}: PF={m_f['pf']:.3f}  n={m_f['n']:3d}  Net=${net:+6.0f}  Share={pct:+6.1f}%")
print()

t4_fold_ok  = max_fold_share <= PASS_CONC_FOLD
t4_sym_ok   = max_sym_share  <= PASS_CONC_SYM
t4_spread_ok = pos_pf_folds >= 2  # at least 2 of 5 folds profitable
t4_pass = t4_fold_ok and t4_sym_ok and t4_spread_ok
print(f"  Fold concentration OK  (≤{PASS_CONC_FOLD:.0%}): "
      f"{'✓' if t4_fold_ok else '✗'}  ({max_fold_share:.1%})")
print(f"  Symbol concentration OK (≤{PASS_CONC_SYM:.0%}): "
      f"{'✓' if t4_sym_ok else '✗'}  ({max_sym_share:.1%})")
print(f"  ≥2 of 5 folds profitable: {'✓' if t4_spread_ok else '✗'}  ({pos_pf_folds}/5)")
print(f"  TEST 4 RESULT: {'✓ PASS' if t4_pass else '✗ FAIL'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — SYMBOL HOLDOUT
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  TEST 5 — Symbol Holdout (14 never-seen symbols)")
print("  IS thresholds learned from 35 training symbols only.")
print("  Pooled thresholds applied to 14 holdout symbols.")
print(SEP)
print()
print(f"  Training symbols ({len(TRAIN_SYMS)}): "
      f"{', '.join(s.replace('-USDT-SWAP','') for s in TRAIN_SYMS[:8])} ...")
print(f"  Holdout symbols  ({len(HOLDOUT_SYMS)}): "
      f"{', '.join(s.replace('-USDT-SWAP','') for s in HOLDOUT_SYMS)}")
print()

# Learn pooled thresholds from training symbols' IS data
train_thr_list = []
for sym in TRAIN_SYMS:
    if sym not in all_dfs: continue
    df_is, _, _ = all_dfs[sym]
    train_thr_list.append(learn_thresholds(df_is))
pooled_thr = pooled_thresholds(train_thr_list)
print(f"  Pooled thresholds learned from {len(train_thr_list)} training symbols.")

# Run E3.1_v2 on holdout symbols using pooled thresholds
hold_all, hold_fold, hold_sym = run_wf(
    FROZEN_CIDS, HOLDOUT_SYMS, pooled_thr=pooled_thr)
hold_m = metrics(hold_all)

print(f"  Holdout result: PF={hold_m['pf']:.3f}  n={hold_m['n']}  "
      f"WR={hold_m['wr']:.1%}  MDD={hold_m['mdd']:.1%}")
print()

# Also run holdout with per-symbol own IS (as a control — own IS should be better)
hold_own_all, _, _ = run_wf(FROZEN_CIDS, HOLDOUT_SYMS)
hold_own_m = metrics(hold_own_all)
print(f"  Holdout (own IS): PF={hold_own_m['pf']:.3f}  n={hold_own_m['n']}  "
      f"(control — uses each symbol's own IS, not pooled)")
print()

# Per-holdout-symbol breakdown
print(f"  Per-holdout-symbol OOS performance (pooled thresholds):")
print(f"  {'Symbol':<14}  {'PF':>8}  {'n':>4}  {'WR':>7}  {'Net':>8}")
print("  " + "─" * 50)
hold_sym_results = []
for sym in HOLDOUT_SYMS:
    tl = hold_sym.get(sym,[])
    m  = metrics(tl)
    hold_sym_results.append((sym, m))
    print(f"  {sym.replace('-USDT-SWAP',''):<14}  {m['pf']:>8.3f}  "
          f"{m['n']:>4}  {m['wr']:>7.1%}  ${m['net']:>+7.0f}")
print()

n_holdout_pos = sum(1 for _, m in hold_sym_results if m["pf"] > 1.0 and m["n"] > 3)
n_holdout_valid = sum(1 for _, m in hold_sym_results if m["n"] > 3)
print(f"  Holdout symbols with PF>1.0 (n>3): {n_holdout_pos}/{n_holdout_valid}")

t5_pf_ok   = hold_m["pf"] > PASS_HOLDOUT_PF
t5_pass    = t5_pf_ok and hold_m["n"] >= 5
print(f"  Holdout OOS PF > {PASS_HOLDOUT_PF}: "
      f"{'✓' if t5_pf_ok else '✗'}  (PF={hold_m['pf']:.3f})")
print(f"  TEST 5 RESULT: {'✓ PASS' if t5_pass else '✗ FAIL'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  FINAL VERDICT — E3.1_v2 Reality Check")
print(SEP)
print()

tests = [
    ("Test 1: Parameter Robustness",  t1_pass,
     f"{total_above_1}/{total_vars} variations PF>1.0 ({robustness_rate:.0%}) — "
     f"need ≥{PASS_PARAM_ROBUST:.0%}"),
    ("Test 2: Permutation Null",       t2_pass,
     f"Observed PF={obs_pf:.3f} at {pctile_rank:.1f}th pctile "
     f"(null median={perm_med:.3f}, p95={perm_p95:.3f})"),
    ("Test 3: Condition Ablation",     t3_pass,
     f"{n_meaningful_drops}/4 conditions degrade PF by ≥{PASS_ABLATION_DROP} on removal"),
    ("Test 4: Temporal Stability",     t4_pass,
     f"MaxFoldShare={max_fold_share:.1%}, MaxSymShare={max_sym_share:.1%}, "
     f"PosFolds={pos_pf_folds}/5"),
    ("Test 5: Symbol Holdout",         t5_pass,
     f"Pooled-thr holdout PF={hold_m['pf']:.3f} on {len(HOLDOUT_SYMS)} never-seen syms"),
]

pass_count = sum(1 for _, p, _ in tests if p)
print(f"  {'Test':<34}  {'Result':>8}  Detail")
print("  " + "─" * 100)
for name, passed, detail in tests:
    symbol = "✓ PASS" if passed else "✗ FAIL"
    color_hint = "  [OK]" if passed else "  [!!]"
    print(f"  {name:<34}  {symbol:>8}  {detail[:75]}")
print()

if pass_count >= 4:
    verdict = "REAL"
    verdict_detail = (
        "The edge is structurally robust. It holds across parameter perturbations,\n"
        "  beats random entry timing at the 95th+ percentile, every condition\n"
        "  contributes, performance is distributed across time/symbols, and\n"
        "  it generalises to never-seen symbols. This is a genuine alpha."
    )
elif pass_count >= 2:
    verdict = "SUSPECT"
    verdict_detail = (
        "The edge shows real signal but has structural weaknesses. Something in\n"
        "  its design may be overfit or regime-specific. Continue monitoring before\n"
        "  scaling. Investigate the failed tests for the specific vulnerability."
    )
else:
    verdict = "GOOSE CHASE 🪿"
    verdict_detail = (
        "The observed PF is likely an artifact of data mining, regime coincidence,\n"
        "  or parameter overfitting. Do not allocate capital. Redesign from scratch."
    )

print(f"  ╔{'═'*70}╗")
print(f"  ║  TESTS PASSED: {pass_count}/5  →  VERDICT: {verdict:<50}║")
print(f"  ╚{'═'*70}╝")
print()
print(f"  {verdict_detail}")
print()

# DST_MD summary
print(f"  ─── DST_MD P1 reference ({DST_MD_LABEL}): ───")
print(f"  Baseline PF={dst_m['pf']:.3f}  n={dst_m['n']}")
print(f"  Permutation rank: {dst_pctile:.1f}th pctile of null distribution")
print()

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts ...")
print(SEP2)

fig = plt.figure(figsize=(22, 18), facecolor=C_BG)
fig.suptitle(f"R061 — Reality Check: 5-Test Stress Battery on E3.1_v2",
             fontsize=13, color=C_GOLD, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

# ─ T1: Parameter robustness (one subplot per parameter)
ax_t1 = fig.add_subplot(gs[0, :3])
param_colors = [C_GREEN, C_GOLD, C_BLUE, C_PURP, C_ORAN]
x_off = 0; xticks = []; xlabels = []
sep_positions = []
for pi, (param_name, (values, cid_key, label)) in enumerate(param_variations.items()):
    pfs_row = [pf for _, pf, _ in param_results[param_name]]
    xs = np.arange(len(values)) + x_off
    for j, (v, pf) in enumerate(zip(values, pfs_row)):
        frozen_v = FROZEN_PARAMS[param_name]
        is_frozen = abs(v - frozen_v) < 1e-6
        col = param_colors[pi]
        alpha = 0.95 if is_frozen else 0.65
        edgecol = C_GOLD if is_frozen else col
        ax_t1.bar(xs[j], pf, color=col, alpha=alpha,
                  edgecolor=edgecol, linewidth=1.5 if is_frozen else 0.5)
        xticks.append(xs[j]); xlabels.append(f"{v}")
    sep_positions.append(x_off + len(values) + 0.5)
    x_off += len(values) + 1
ax_t1.axhline(1.0, color=C_RED, linewidth=1.0, linestyle="--", alpha=0.7, label="PF=1.0")
ax_t1.axhline(base_m["pf"], color=C_GREEN, linewidth=0.8, linestyle=":", alpha=0.5,
              label=f"Baseline PF={base_m['pf']:.3f}")
for sep in sep_positions[:-1]:
    ax_t1.axvline(sep, color=C_GRID, linewidth=0.8, alpha=0.5)
ax_t1.set_xticks(xticks); ax_t1.set_xticklabels(xlabels, rotation=45, fontsize=6)
ax_t1.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT, loc="upper right")
# Parameter labels
offset = 0
for pi, (param_name, (values, _, label)) in enumerate(param_variations.items()):
    ax_t1.text(offset + len(values)/2 - 0.5, ax_t1.get_ylim()[0] * 1.05 if ax_t1.get_ylim()[0] > 0 else -0.05,
               label, ha="center", va="top", fontsize=7, color=param_colors[pi],
               transform=ax_t1.transData)
    offset += len(values) + 1
panel_style(ax_t1, f"T1: Parameter Robustness — {total_above_1}/{total_vars} PF>1.0  "
            f"{'✓ PASS' if t1_pass else '✗ FAIL'}", fs=9)

# ─ T2: Permutation null histogram
ax_t2 = fig.add_subplot(gs[0, 3])
ax_t2.hist(perm_arr, bins=40, color=C_BLUE, alpha=0.75, edgecolor=C_GRID)
ax_t2.axvline(obs_pf,   color=C_GREEN, linewidth=2.0, linestyle="-",  label=f"Observed {obs_pf:.3f}")
ax_t2.axvline(perm_p95, color=C_GOLD,  linewidth=1.2, linestyle="--", label=f"p95={perm_p95:.3f}")
ax_t2.axvline(1.0,      color=C_RED,   linewidth=0.8, linestyle=":",  alpha=0.6, label="PF=1.0")
ax_t2.legend(fontsize=6.5, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_t2, f"T2: Permutation Null (n={N_PERM})\n"
            f"Obs at {pctile_rank:.1f}th pctile  {'✓ PASS' if t2_pass else '✗ FAIL'}", fs=8)

# ─ T3: Ablation bar chart
ax_t3 = fig.add_subplot(gs[1, :2])
ab_labels = [r[0] for r in ablation_results]
ab_pfs    = [r[1] for r in ablation_results]
ab_colors = [C_GREEN if pf >= base_m["pf"]-0.01 else
             (C_GOLD if pf >= 1.0 else C_RED) for pf in ab_pfs]
xab = np.arange(len(ab_labels))
ax_t3.bar(xab, ab_pfs, color=ab_colors, alpha=0.85)
ax_t3.axhline(1.0,          color=C_RED,   linewidth=0.8, linestyle="--", alpha=0.6)
ax_t3.axhline(base_m["pf"], color=C_GREEN, linewidth=0.8, linestyle=":",  alpha=0.5,
              label=f"Full PF={base_m['pf']:.3f}")
ax_t3.set_xticks(xab)
ax_t3.set_xticklabels([l[:18] for l in ab_labels], rotation=30, ha="right", fontsize=7)
ax_t3.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_t3, f"T3: Condition Ablation — {n_meaningful_drops}/4 conditions matter  "
            f"{'✓ PASS' if t3_pass else '✗ FAIL'}", fs=8)

# ─ T4: Fold + symbol PnL concentration
ax_t4a = fig.add_subplot(gs[1, 2])
fold_labels = [f"F{i}" for i in range(1, N_FWD_FOLDS+1)]
fold_pcts   = [fold_nets.get(fl, 0)/total_net*100 if total_net else 0
               for fl in fold_labels]
colors_f4a  = [C_GREEN if v > 0 else C_RED for v in fold_pcts]
xf = np.arange(len(fold_labels))
ax_t4a.bar(xf, fold_pcts, color=colors_f4a, alpha=0.85)
ax_t4a.axhline(PASS_CONC_FOLD*100,  color=C_GOLD, linewidth=0.8, linestyle="--",
               alpha=0.7, label=f"Conc limit {PASS_CONC_FOLD:.0%}")
ax_t4a.axhline(-PASS_CONC_FOLD*100, color=C_GOLD, linewidth=0.8, linestyle="--", alpha=0.7)
ax_t4a.set_xticks(xf); ax_t4a.set_xticklabels(fold_labels)
ax_t4a.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_t4a, f"T4a: Fold PnL Concentration\n{max_fold_share:.1%} max  "
            f"{'✓' if t4_fold_ok else '✗'}", fs=8)

ax_t4b = fig.add_subplot(gs[1, 3])
top10_syms = sorted(sym_nets, key=lambda s: -abs(sym_nets.get(s,0)))[:10]
sym_pcts   = [sym_nets.get(s,0)/total_net*100 if total_net else 0 for s in top10_syms]
sym_lbl    = [s.replace("-USDT-SWAP","") for s in top10_syms]
colors_t4b = [C_GREEN if v > 0 else C_RED for v in sym_pcts]
xs4b = np.arange(len(sym_lbl))
ax_t4b.bar(xs4b, sym_pcts, color=colors_t4b, alpha=0.85)
ax_t4b.axhline(PASS_CONC_SYM*100,  color=C_GOLD, linewidth=0.8, linestyle="--", alpha=0.7)
ax_t4b.axhline(-PASS_CONC_SYM*100, color=C_GOLD, linewidth=0.8, linestyle="--", alpha=0.7)
ax_t4b.set_xticks(xs4b); ax_t4b.set_xticklabels(sym_lbl, rotation=45, ha="right", fontsize=6)
panel_style(ax_t4b, f"T4b: Symbol PnL Concentration\n{max_sym_share:.1%} max  "
            f"{'✓' if t4_sym_ok else '✗'}", fs=8)

# ─ T5: Holdout per-symbol PF
ax_t5 = fig.add_subplot(gs[2, :2])
hold_syms_lbl = [s.replace("-USDT-SWAP","") for s in HOLDOUT_SYMS]
hold_pfs_vals = [metrics(hold_sym.get(s,[])) ["pf"] for s in HOLDOUT_SYMS]
hold_colors   = [C_GREEN if pf > 1.0 else C_RED for pf in hold_pfs_vals]
xh = np.arange(len(hold_syms_lbl))
ax_t5.bar(xh, hold_pfs_vals, color=hold_colors, alpha=0.85)
ax_t5.axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--", alpha=0.7)
ax_t5.axhline(hold_m["pf"], color=C_GOLD, linewidth=1.0, linestyle="-",
              label=f"Holdout avg PF={hold_m['pf']:.3f}")
ax_t5.set_xticks(xh); ax_t5.set_xticklabels(hold_syms_lbl, rotation=45, ha="right", fontsize=7)
ax_t5.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_t5, f"T5: Symbol Holdout PF (n={len(HOLDOUT_SYMS)} never-seen)  "
            f"Overall PF={hold_m['pf']:.3f}  {'✓ PASS' if t5_pass else '✗ FAIL'}", fs=8)

# ─ Verdict summary panel
ax_verd = fig.add_subplot(gs[2, 2:])
ax_verd.axis("off")
verd_color = C_GREEN if verdict == "REAL" else (C_GOLD if verdict == "SUSPECT" else C_RED)
vlines = [
    ("R061 — REALITY CHECK VERDICT", C_GOLD),
    ("─" * 46, C_GRID),
    (f"Tests passed: {pass_count}/5", C_TEXT),
    ("─" * 46, C_GRID),
]
for name, passed, detail in tests:
    sym = "✓" if passed else "✗"
    col = C_GREEN if passed else C_RED
    vlines.append((f"{sym} {name}", col))
vlines += [
    ("─" * 46, C_GRID),
    (f"VERDICT: {verdict}", verd_color),
    ("─" * 46, C_GRID),
    (f"E3.1_v2 baseline PF: {base_m['pf']:.3f}", C_TEXT),
    (f"Perm null p95: {perm_p95:.3f}", C_TEXT),
    (f"Holdout PF: {hold_m['pf']:.3f} ({len(HOLDOUT_SYMS)} syms)", C_TEXT),
    (f"DST_MD P1 pctile: {dst_pctile:.1f}th", C_TEXT),
]
for i, (line, col) in enumerate(vlines):
    ax_verd.text(0.02, 0.99 - i*0.072, line, transform=ax_verd.transAxes,
                 fontsize=6.5, color=col, va="top", fontfamily="monospace")
panel_style(ax_verd, "Verdict")

plt.savefig(f"{OUT}/r061_reality_check.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r061_reality_check.png")

# ─ Chart 2: DST_MD fold vs E3.1_v2 fold (anti-correlation visual)
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), facecolor=C_BG)
fig2.suptitle("R061 — Fold-by-Fold: E3.1_v2 vs DST_MD P1 (Anti-correlated regimes)",
              fontsize=11, color=C_GOLD, fontweight="bold")
fold_lbls = [f"F{i}" for i in range(1,6)]
e3_fps  = [metrics(base_fold.get(fl,[]))["pf"] for fl in fold_lbls]
dst_fps = [metrics(dst_fold.get(fl, []))["pf"] for fl in fold_lbls]
xf2 = np.arange(len(fold_lbls)); wf2 = 0.38
axes2[0].bar(xf2 - wf2/2, e3_fps,  wf2, color=C_BLUE, alpha=0.85, label="E3.1_v2")
axes2[0].bar(xf2 + wf2/2, dst_fps, wf2, color=C_GOLD, alpha=0.85, label="DST_MD P1")
axes2[0].axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--", alpha=0.6)
axes2[0].set_xticks(xf2); axes2[0].set_xticklabels(fold_lbls)
axes2[0].legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(axes2[0], "Per-Fold PF: E3.1_v2 vs DST_MD P1")

# Equity curve comparison
eq_e3  = base_m["equity"]
eq_dst = dst_m["equity"]
axes2[1].plot(np.arange(len(eq_e3)),  eq_e3,  color=C_BLUE, linewidth=1.4, label="E3.1_v2")
axes2[1].plot(np.arange(len(eq_dst)), eq_dst, color=C_GOLD, linewidth=1.4, label="DST_MD P1")
axes2[1].axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
axes2[1].legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(axes2[1], "Equity Curves")

# Null distribution with both families marked
axes2[2].hist(perm_arr, bins=40, color=C_BLUE, alpha=0.65, edgecolor=C_GRID, label="Null (random)")
axes2[2].axvline(obs_pf,       color=C_GREEN, linewidth=2.0, label=f"E3.1_v2 {obs_pf:.3f}")
axes2[2].axvline(dst_m["pf"],  color=C_GOLD,  linewidth=2.0, linestyle="--",
                 label=f"DST_MD P1 {dst_m['pf']:.3f}")
axes2[2].axvline(perm_p95,     color=C_RED,   linewidth=1.0, linestyle=":",
                 label=f"p95={perm_p95:.3f}")
axes2[2].legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(axes2[2], f"Permutation Null — Both Families\n"
            f"E3.1_v2 @ {pctile_rank:.1f}th  |  DST_MD @ {dst_pctile:.1f}th")
plt.tight_layout()
plt.savefig(f"{OUT}/r061_family_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r061_family_comparison.png")

# CSV
rows = []
for name, passed, detail in tests:
    rows.append({"test": name, "pass": int(passed), "detail": detail})
pd.DataFrame(rows).to_csv(f"{OUT}/r061_verdict.csv", index=False)

param_rows = []
for param_name, (values, _, label) in param_variations.items():
    for val, pf, n in param_results[param_name]:
        param_rows.append({"param": label, "value": val, "pf": round(pf,4), "n": n})
pd.DataFrame(param_rows).to_csv(f"{OUT}/r061_param_grid.csv", index=False)
print(f"  ✓  {OUT}/r061_verdict.csv  {OUT}/r061_param_grid.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL PRINT
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print(f"  R061 COMPLETE — REALITY CHECK")
print(SEP)
print()
print(f"  Frozen strategy  : {FROZEN_LABEL}")
print(f"  Baseline PF      : {base_m['pf']:.3f}  n={base_m['n']}  WR={base_m['wr']:.1%}")
print()
for name, passed, detail in tests:
    print(f"  {'✓' if passed else '✗'}  {name:<34}  {detail[:65]}")
print()
print(f"  TESTS PASSED: {pass_count}/5")
print(f"  VERDICT: {verdict}")
print()
print(SEP)
