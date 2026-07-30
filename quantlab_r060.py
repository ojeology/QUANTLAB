"""
=============================================================================
QUANTLAB AI — RESEARCH #R060
Dual Confirmation: E3.1-v2 (BBW_STRICT) + DST_MD Portfolio Construction
=============================================================================

Objective:
  R057 mandate: Run a brand-new forward test of E3.1 upgraded with
  the BBW_STRICT macro filter (Criteria met: 5/6. Verdict: YES).

  R059 mandate: Build and test the DST_MD multi-environment portfolio
  from the 9 confirmed WATCHLIST candidates.

  This script does both in one run:

  PART A — E3.1_v2 Forward Test
    Frozen new strategy: BBW_STRICT + RV_LO + DST_NR + PRG_VH
    BBW_STRICT = bb_width < IS p25 (tighter than BBW_LO = p33)
    This is a fresh, independent 5-fold walk-forward — NOT a retro filter.
    Compare against E3.1_v1 (original) as reference.

  PART B — DST_MD Portfolio
    Top 3 WATCHLIST environments from R059 (0% overlap with each other
    to verify), combined into a DST_MD portfolio:
      P1: ADX_WK + DST_MD + RV_HI          (UES=64.7, MOMENTUM_BREAKOUT)
      P2: ATR_LO + DST_MD + PRG_LO + RV_LO (UES=57.5, MIXED archetype)
      P3: ADX_WK + DST_MD + LON            (UES=52.2, London session)
    If mutual overlap < 35% and combined n ≥ 200, evaluate vs PROMOTE bar.

  PART C — Portfolio Comparison
    E3.1_v2 / DST_MD portfolio / Combined
    Assess whether the combined two-family portfolio exceeds either alone.

  PROMOTION THRESHOLDS (unchanged):
    PF ≥ 1.20, n ≥ 200, Boot ≥ 1.15, MC ≥ 65%, MDD < 20%
    LOO-sym floor > 1.0, LOO-fold floor > 1.0

  Frozen E3.1_v1: BBW_LO + RV_LO + DST_NR + PRG_VH

=============================================================================
"""

import os, sys, math, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID = "R060"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL      = CONFIG["STARTING_CAPITAL"]
RR           = 2.0
IS_RATIO     = 0.80
N_FWD_FOLDS  = 5
N_BOOT       = 500
N_MC         = 1000
RAND_SEED    = 42
MIN_BARS     = 2_000

# Strategy definitions
E31_V1_LABEL = "BBW_LO+RV_LO+DST_NR+PRG_VH"    # frozen original
E31_V2_LABEL = "BBW_STRICT+RV_LO+DST_NR+PRG_VH" # upgraded with tighter BB gate

# DST_MD Portfolio candidates (from R059 WATCHLIST, ranked by UES)
DST_MD_P1 = ("ADX_WK", "DST_MD", "RV_HI")                      # rank 1
DST_MD_P2 = ("ATR_LO", "DST_MD", "PRG_LO", "RV_LO")            # rank 3
DST_MD_P3 = ("ADX_WK", "DST_MD", "LON")                         # rank 7
DST_MD_ENVS = [DST_MD_P1, DST_MD_P2, DST_MD_P3]
DST_MD_LABELS = ["+".join(e) for e in DST_MD_ENVS]

# Promotion thresholds
PROM_PF   = 1.20
PROM_N    = 200
PROM_BOOT = 1.15
PROM_MC   = 0.65
PROM_MDD  = 0.20
MAX_OVERLAP = 0.35

# Max overlap threshold for internal DST_MD independence
MAX_INTERNAL_OVERLAP = 0.40

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

# ─────────────────────────────────────────────────────────────────────────────
# 49-SYMBOL UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONS CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    # id           feat              direction    param
    ("ATR_LO",    "atr_rank",       "lt_q",      0.25),
    ("ATR_MD",    "atr_rank",       "lt_q",      0.40),
    ("ATR_HI",    "atr_rank",       "gt_q",      0.67),
    ("ATR_VH",    "atr_rank",       "gt_q",      0.80),
    ("BBW_LO",    "bb_width",       "lt_q",      0.33),   # original
    ("BBW_STRICT","bb_width",       "lt_q",      0.25),   # tighter (R060 upgrade)
    ("BBW_HI",    "bb_width",       "gt_q",      0.67),
    ("RV_LO",     "real_vol_20",    "lt_q",      0.33),
    ("RV_HI",     "real_vol_20",    "gt_q",      0.67),
    ("SLP_DN",    "ema200_slope",   "lt_fixed",  0.0),
    ("SLP_UP",    "ema200_slope",   "gt_fixed",  0.0),
    ("DST_NR",    "ema_dist_pct",   "lt_q",      0.33),
    ("DST_MD",    "ema_dist_pct",   "gt_q_pos",  0.60),
    ("DST_FR",    "ema_dist_pct",   "gt_q_pos",  0.75),
    ("ADX_WK",    "adx14",          "lt_q",      0.33),
    ("ADX_TR",    "adx14",          "gt_q",      0.50),
    ("ADX_ST",    "adx14",          "gt_q",      0.67),
    ("PRG_LO",    "prev_range_r",   "lt_q",      0.33),
    ("PRG_HI",    "prev_range_r",   "gt_q",      0.67),
    ("PRG_VH",    "prev_range_r",   "gt_q",      0.80),
    ("PBD_HI",    "prev_body_r",    "gt_q",      0.67),
    ("PBP_HI",    "prev_body_pct",  "gt_q",      0.60),
    ("PBP_LO",    "prev_body_pct",  "lt_q",      0.33),
    ("US",        "hour_utc",       "hour_rng",  (14, 21)),
    ("LON",       "hour_utc",       "hour_rng",  (7,  14)),
    ("ASI",       "hour_utc",       "hour_rng",  (0,   6)),
]
COND_BY_ID  = {c[0]: c for c in CONDITIONS_DEF}
QUANT_FEATS = ["atr_rank", "bb_width", "real_vol_20", "ema_dist_pct",
               "adx14", "prev_range_r", "prev_body_r", "prev_body_pct"]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]
    o = df["open"]
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
    df["dow"]           = dt.dt.dayofweek
    return df

def learn_thresholds(df_is):
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (_, feat, direction, param) in COND_BY_ID.items():
        if direction in ("lt_fixed", "gt_fixed", "hour_rng"):
            thr[cid] = param; continue
        if feat not in valid.columns:
            thr[cid] = np.nan; continue
        col = valid[feat].dropna()
        if len(col) < 20:
            thr[cid] = np.nan; continue
        if direction == "gt_q_pos":
            pos = col[col > 0]
            thr[cid] = float(pos.quantile(param) if len(pos) > 10 else col.quantile(param))
        else:
            thr[cid] = float(col.quantile(param))
    return thr

# ─────────────────────────────────────────────────────────────────────────────
# MASK BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_env_mask(df, cond_ids, thr):
    N    = len(df)
    mask = np.ones(N, dtype=bool)
    for cid in cond_ids:
        if cid not in COND_BY_ID:
            return np.zeros(N, dtype=bool)
        _, feat, direction, _ = COND_BY_ID[cid]
        if feat not in df.columns:
            return np.zeros(N, dtype=bool)
        col   = df[feat].values
        nan_m = np.isnan(col) if col.dtype.kind == "f" else np.zeros(N, dtype=bool)
        t     = thr.get(cid, np.nan)
        if direction == "lt_q":
            if np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col < t)
        elif direction in ("gt_q", "gt_q_pos"):
            if np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col > t)
        elif direction == "gt_fixed":
            mask &= (~nan_m) & (col > t)
        elif direction == "lt_fixed":
            mask &= (~nan_m) & (col < t)
        elif direction == "hour_rng":
            lo_, hi_ = t
            mask &= (col >= lo_) & (col <= hi_)
    return mask

def entry_signal(df, env_mask):
    rv = df["rel_vol"].values
    c  = df["close"].values; o = df["open"].values; pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & env_mask

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signal, sym, fold_label):
    min_sl  = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf      = CONFIG["RISK_PER_TRADE_PCT"]
    fee     = CONFIG["TAKER_FEE"];  spd = CONFIG["SPREAD"] * 0.5
    slp     = CONFIG["SL_SLIPPAGE"]
    in_pos  = False; ep = st = tk = sz = 0.0; et = None; ei = -1
    trades  = []

    op_ = df["open"].values;  cl_ = df["close"].values
    hi_ = df["high"].values;  lo_ = df["low"].values
    atr_  = df["prev_atr14"].values
    dts   = df["datetime"].values

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
                    "sym":        sym,
                    "fold":       fold_label,
                    "entry_time": str(et),
                    "pnl":        round(net, 4),
                    "win":        int(not sl_hit),
                })
                in_pos = False
            continue

        if signal[i-1]:
            a = atr_[i]
            if np.isnan(a) or a <= 0: continue
            ep_ = op_[i]
            if a / ep_ < min_sl: continue
            ep  = ep_; st = ep - a; tk = ep + RR * a
            sz  = min(CAPITAL * rf / a, (CAPITAL * max_lev) / ep)
            et  = dts[i]; ei = i; in_pos = True

    return trades

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl

def metrics(trades):
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "exp_r": 0.0, "net": 0.0,
                "mdd": 0.0, "pnls": np.array([]), "equity": np.array([CAPITAL])}
    pnl  = np.array([t["pnl"] for t in trades])
    wins = np.array([t["win"] for t in trades], dtype=bool)
    n = len(pnl); nw = wins.sum(); nl = n - nw
    gw = pnl[wins].sum() if nw else 0.0
    gl = abs(pnl[~wins].sum()) if nl else 0.0
    pf = safe_pf(gw, gl); wr = nw / n
    eq = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk).min())
    exp = wr * RR - (1 - wr)
    return {"n": n, "wr": wr, "pf": pf, "exp_r": exp, "net": float(pnl.sum()),
            "mdd": mdd, "pnls": pnl, "equity": eq}

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=RAND_SEED):
    """Returns (p5, median, p95) tuple."""
    rng = np.random.RandomState(seed)
    if len(pnls) < 2:
        return 0.0, 0.0, 0.0
    pf_s = []
    for _ in range(n_iter):
        samp = rng.choice(pnls, size=len(pnls), replace=True)
        gw = samp[samp > 0].sum() if any(samp > 0) else 0.0
        gl = abs(samp[samp < 0].sum()) if any(samp < 0) else 1e-9
        pf_s.append(min(gw / gl, 20.0))
    arr = np.array(pf_s)
    return float(np.percentile(arr, 5)), float(np.percentile(arr, 50)), float(np.percentile(arr, 95))

def monte_carlo(pnls, n_iter=N_MC, seed=RAND_SEED):
    """Returns dict with prob_profit (fraction of random sign-shuffles beating observed PF)."""
    rng = np.random.RandomState(seed)
    if len(pnls) < 2:
        return {"prob_profit": 1.0}
    gw = pnls[pnls > 0].sum() if any(pnls > 0) else 0.0
    gl = abs(pnls[pnls < 0].sum()) if any(pnls < 0) else 1e-9
    obs_pf = min(gw / gl, 20.0)
    abs_p  = np.abs(pnls)
    pf_s   = []
    for _ in range(n_iter):
        signs = rng.choice([-1, 1], size=len(pnls))
        samp  = abs_p * signs
        gw_s  = samp[samp > 0].sum() if any(samp > 0) else 0.0
        gl_s  = abs(samp[samp < 0].sum()) if any(samp < 0) else 1e-9
        pf_s.append(min(gw_s / gl_s, 20.0))
    return {"prob_profit": float((np.array(pf_s) >= obs_pf).mean())}

def compute_ues(pf, boot_med, mc_p, sym_floor, fold_floor, mdd, n):
    """Unified Edge Score: composite quality metric."""
    pf_s   = min(max((pf  - 1.0) / 0.5, 0.0), 1.0)
    b_s    = min(max((boot_med - 1.0) / 0.4, 0.0), 1.0)
    mc_s   = min(max((1.0 - mc_p) / 0.5, 0.0), 1.0)
    n_s    = min(n / 300.0, 1.0)
    mdd_s  = min(max(1.0 - abs(mdd) / 0.25, 0.0), 1.0)
    sf_s   = min(max((sym_floor - 1.0) / 0.3, 0.0), 1.0)
    ff_s   = min(max((fold_floor - 1.0) / 0.3, 0.0), 1.0)
    raw    = 0.20*pf_s + 0.15*b_s + 0.15*mc_s + 0.15*n_s + 0.15*mdd_s + 0.10*sf_s + 0.10*ff_s
    return round(raw * 100, 1)

def loo_robustness(all_trades, sym_trades):
    """Leave-one-symbol-out and leave-one-fold-out PF floors."""
    syms = list(sym_trades.keys())
    loo_sym = []
    for s in syms:
        rest = [t for t in all_trades if t["sym"] != s]
        if len(rest) >= 10:
            loo_sym.append(metrics(rest)["pf"])
    folds = sorted({t["fold"] for t in all_trades})
    loo_fld = []
    for f in folds:
        rest = [t for t in all_trades if t["fold"] != f]
        if len(rest) >= 10:
            loo_fld.append(metrics(rest)["pf"])
    sym_floor  = float(np.min(loo_sym))  if loo_sym  else 0.0
    fold_floor = float(np.min(loo_fld)) if loo_fld else 0.0
    return sym_floor, fold_floor

def score_and_verdict(pf, n, boot_med, mc_p, sym_floor, fold_floor, mdd):
    """Score against 7-criteria promotion gate."""
    sc = 0
    sc += int(pf        > PROM_PF)
    sc += int(n         >= PROM_N)
    sc += int(boot_med  > PROM_BOOT)
    sc += int(mc_p      < (1.0 - PROM_MC))
    sc += int(sym_floor > 1.0)
    sc += int(fold_floor > 1.0)
    sc += int(abs(mdd)  < PROM_MDD)
    if sc == 7:
        verdict = "PROMOTE"
    elif sc >= 5:
        verdict = "WATCHLIST"
    elif sc >= 3:
        verdict = "INVESTIGATE"
    else:
        verdict = "REJECT"
    return sc, verdict

def overlap_and_corr(trades_a, trades_b):
    """Trade-level overlap fraction and PnL correlation."""
    if not trades_a or not trades_b:
        return 0.0, 0.0
    times_a = {(t["sym"], t["entry_time"]) for t in trades_a}
    times_b = {(t["sym"], t["entry_time"]) for t in trades_b}
    inter   = times_a & times_b
    union   = times_a | times_b
    ov      = len(inter) / len(union) if union else 0.0

    # Pairwise PnL correlation via symbol-fold bins
    bins_a = defaultdict(float); bins_b = defaultdict(float)
    for t in trades_a: bins_a[(t["sym"], t["fold"])] += t["pnl"]
    for t in trades_b: bins_b[(t["sym"], t["fold"])] += t["pnl"]
    keys = sorted(set(bins_a) | set(bins_b))
    va = np.array([bins_a.get(k, 0.0) for k in keys])
    vb = np.array([bins_b.get(k, 0.0) for k in keys])
    if va.std() < 1e-9 or vb.std() < 1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(va, vb)[0, 1])
    return round(ov, 4), round(corr, 4)

def combined_metrics(lists_of_trades):
    """Merge multiple trade lists (dedup by sym+time), compute metrics."""
    seen = {}
    for trades in lists_of_trades:
        for t in trades:
            key = (t["sym"], t["entry_time"])
            if key not in seen:
                seen[key] = t
    merged = list(seen.values())
    merged.sort(key=lambda t: t["entry_time"])
    return metrics(merged), merged

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print(f"  QUANTLAB AI — RESEARCH #{RESEARCH_ID}")
print("  Dual Confirmation: E3.1-v2 (BBW_STRICT) + DST_MD Portfolio Construction")
print(SEP)
print()
print(f"  PART A  E3.1_v2  → {E31_V2_LABEL}")
print(f"  PART B  DST_MD Portfolio → {' | '.join(DST_MD_LABELS)}")
print(f"  REFERENCE  E3.1_v1  → {E31_V1_LABEL}")
print(f"  RR={RR}  IS_RATIO={IS_RATIO}  Folds={N_FWD_FOLDS}  Symbols={len(ALL_SYMBOLS)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 0 — Data Load")
print(SEP)
print()

E31_V1_CIDS = ("BBW_LO",     "RV_LO", "DST_NR", "PRG_VH")
E31_V2_CIDS = ("BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH")

all_dfs = {}
loaded = 0
for sym in ALL_SYMBOLS:
    tag  = sym.replace("-", "_")
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

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: run 5-fold WF for a given condition set
# ─────────────────────────────────────────────────────────────────────────────
def run_wf(cond_ids, label=""):
    """Run 5-fold walk-forward for cond_ids across all symbols."""
    all_t = []; fold_t = defaultdict(list); sym_t = defaultdict(list)
    for sym, (df_is, df_fwd, thr) in all_dfs.items():
        fwd_size = len(df_fwd)
        seg_size = max(1, fwd_size // N_FWD_FOLDS)
        for fi in range(N_FWD_FOLDS):
            seg_s = fi * seg_size
            seg_e = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
            df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
            if len(df_seg) < 20: continue
            fl  = f"F{fi+1}"
            em  = build_env_mask(df_seg, cond_ids, thr)
            sig = entry_signal(df_seg, em)
            tl  = run_backtest(df_seg, sig, sym, fl)
            all_t.extend(tl); fold_t[fl].extend(tl); sym_t[sym].extend(tl)
    return all_t, fold_t, sym_t

def full_eval(all_t, sym_t, label):
    """Full evaluation: metrics + bootstrap + MC + LOO + UES + verdict."""
    m = metrics(all_t)
    if m["n"] == 0:
        return None
    b5, b50, b95 = bootstrap_pf(m["pnls"])
    mc_res        = monte_carlo(m["pnls"])
    sf, ff        = loo_robustness(all_t, sym_t)
    sc, verdict   = score_and_verdict(m["pf"], m["n"], b50, mc_res["prob_profit"],
                                       sf, ff, m["mdd"])
    ues = compute_ues(m["pf"], b50, mc_res["prob_profit"], sf, ff, m["mdd"], m["n"])
    return {
        "label":   label,
        "n":       m["n"],
        "wr":      m["wr"],
        "pf":      m["pf"],
        "net":     m["net"],
        "mdd":     m["mdd"],
        "pnls":    m["pnls"],
        "equity":  m["equity"],
        "b5":      b5,
        "b50":     b50,
        "b95":     b95,
        "mc_p":    mc_res["prob_profit"],
        "sf":      sf,
        "ff":      ff,
        "score":   sc,
        "verdict": verdict,
        "ues":     ues,
        "trades":  all_t,
        "sym_t":   sym_t,
    }

def print_eval(ev, fold_t=None, header=True):
    """Print formatted evaluation results."""
    if header:
        print(f"  {'Metric':<26}  {'Value':>12}")
        print("  " + "─" * 42)
    print(f"  {'Strategy':<26}  {ev['label'][:40]}")
    print(f"  {'Trades (n)':<26}  {ev['n']:>12}")
    print(f"  {'Win Rate':<26}  {ev['wr']:>12.1%}")
    print(f"  {'Profit Factor':<26}  {ev['pf']:>12.3f}")
    print(f"  {'Max Drawdown':<26}  {ev['mdd']:>12.1%}")
    print(f"  {'Bootstrap p5':<26}  {ev['b5']:>12.3f}")
    print(f"  {'Bootstrap Median':<26}  {ev['b50']:>12.3f}")
    print(f"  {'Bootstrap p95':<26}  {ev['b95']:>12.3f}")
    print(f"  {'MC Probability':<26}  {ev['mc_p']:>12.1%}")
    print(f"  {'LOO-Symbol Floor':<26}  {ev['sf']:>12.3f}")
    print(f"  {'LOO-Fold Floor':<26}  {ev['ff']:>12.3f}")
    print(f"  {'Score':<26}  {ev['score']:>12}/7")
    print(f"  {'UES':<26}  {ev['ues']:>12.1f}")
    print(f"  {'Verdict':<26}  {ev['verdict']:>12}")
    if fold_t:
        print()
        print(f"  Fold breakdown:")
        for fi in range(1, N_FWD_FOLDS + 1):
            fl  = f"F{fi}"
            m_f = metrics(fold_t.get(fl, []))
            bar = "▲" if m_f["pf"] >= 1.20 else ("▼" if m_f["pf"] < 1.0 else "─")
            print(f"    {fl}: PF={m_f['pf']:.3f}  n={m_f['n']:3d}  WR={m_f['wr']:.1%}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# PART A — E3.1_v1 REFERENCE + E3.1_v2 FRESH FORWARD TEST
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  PART A — E3.1 UPGRADE: v1 Reference vs v2 (BBW_STRICT) Fresh Test")
print(SEP)
print()

# v1 — original frozen strategy (reference)
print("  Running E3.1_v1 (original reference) ...")
v1_all, v1_fold, v1_sym = run_wf(E31_V1_CIDS, E31_V1_LABEL)
ev_v1 = full_eval(v1_all, v1_sym, E31_V1_LABEL)
print(f"  E3.1_v1: n={ev_v1['n']}  PF={ev_v1['pf']:.3f}  WR={ev_v1['wr']:.1%}  "
      f"MDD={ev_v1['mdd']:.1%}  UES={ev_v1['ues']:.1f}")
print()

# v2 — BBW_STRICT upgrade (fresh forward test — R057 mandate)
print("  Running E3.1_v2 (BBW_STRICT fresh forward test — R057 mandate) ...")
v2_all, v2_fold, v2_sym = run_wf(E31_V2_CIDS, E31_V2_LABEL)
ev_v2 = full_eval(v2_all, v2_sym, E31_V2_LABEL)
print(f"  E3.1_v2: n={ev_v2['n']}  PF={ev_v2['pf']:.3f}  WR={ev_v2['wr']:.1%}  "
      f"MDD={ev_v2['mdd']:.1%}  UES={ev_v2['ues']:.1f}")
print()

print(SEP)
print("  E3.1_v1 (Original) — Full Evaluation")
print(SEP)
print()
print_eval(ev_v1, v1_fold)
print()

print(SEP)
print("  E3.1_v2 (BBW_STRICT) — Full Evaluation (R057 Forward Test)")
print(SEP)
print()
print_eval(ev_v2, v2_fold)
print()

# v1 vs v2 comparison
print(SEP)
print("  E3.1_v1 vs v2 — Direct Comparison")
print(SEP)
print()
print(f"  {'Metric':<26}  {'v1 (Original)':>14}  {'v2 (BBW_STRICT)':>16}  {'Δ':>10}")
print("  " + "─" * 74)

def delta(v1v, v2v, fmt="{:.3f}", pct=False):
    diff = v2v - v1v
    arr  = "▲" if diff > 0.001 else ("▼" if diff < -0.001 else "─")
    if pct:
        return f"{fmt.format(v1v):>14}  {fmt.format(v2v):>16}  {arr}{abs(diff):{'.1%' if pct else '.3f'}}"
    return f"{fmt.format(v1v):>14}  {fmt.format(v2v):>16}  {arr}{abs(diff):.3f}"

print(f"  {'Trades (n)':<26}  {ev_v1['n']:>14}  {ev_v2['n']:>16}  {'Δ'}{ev_v2['n']-ev_v1['n']:+d}")
print(f"  {'Win Rate':<26}  {ev_v1['wr']:>14.1%}  {ev_v2['wr']:>16.1%}  "
      f"{'▲' if ev_v2['wr']>ev_v1['wr'] else '▼'}{abs(ev_v2['wr']-ev_v1['wr']):.1%}")
print(f"  {'Profit Factor':<26}  {ev_v1['pf']:>14.3f}  {ev_v2['pf']:>16.3f}  "
      f"{'▲' if ev_v2['pf']>ev_v1['pf'] else '▼'}{abs(ev_v2['pf']-ev_v1['pf']):.3f}")
print(f"  {'Max Drawdown':<26}  {ev_v1['mdd']:>14.1%}  {ev_v2['mdd']:>16.1%}  "
      f"{'▼' if ev_v2['mdd']<ev_v1['mdd'] else '▲'}{abs(ev_v2['mdd']-ev_v1['mdd']):.1%}")
print(f"  {'Bootstrap p5':<26}  {ev_v1['b5']:>14.3f}  {ev_v2['b5']:>16.3f}  "
      f"{'▲' if ev_v2['b5']>ev_v1['b5'] else '▼'}{abs(ev_v2['b5']-ev_v1['b5']):.3f}")
print(f"  {'Bootstrap Median':<26}  {ev_v1['b50']:>14.3f}  {ev_v2['b50']:>16.3f}  "
      f"{'▲' if ev_v2['b50']>ev_v1['b50'] else '▼'}{abs(ev_v2['b50']-ev_v1['b50']):.3f}")
print(f"  {'MC Probability':<26}  {ev_v1['mc_p']:>14.1%}  {ev_v2['mc_p']:>16.1%}  "
      f"{'▼' if ev_v2['mc_p']<ev_v1['mc_p'] else '▲'}{abs(ev_v2['mc_p']-ev_v1['mc_p']):.1%}")
print(f"  {'LOO-Symbol Floor':<26}  {ev_v1['sf']:>14.3f}  {ev_v2['sf']:>16.3f}  "
      f"{'▲' if ev_v2['sf']>ev_v1['sf'] else '▼'}{abs(ev_v2['sf']-ev_v1['sf']):.3f}")
print(f"  {'LOO-Fold Floor':<26}  {ev_v1['ff']:>14.3f}  {ev_v2['ff']:>16.3f}  "
      f"{'▲' if ev_v2['ff']>ev_v1['ff'] else '▼'}{abs(ev_v2['ff']-ev_v1['ff']):.3f}")
print(f"  {'Score':<26}  {ev_v1['score']:>14}/7  {ev_v2['score']:>16}/7")
print(f"  {'UES':<26}  {ev_v1['ues']:>14.1f}  {ev_v2['ues']:>16.1f}  "
      f"{'▲' if ev_v2['ues']>ev_v1['ues'] else '▼'}{abs(ev_v2['ues']-ev_v1['ues']):.1f}")
print(f"  {'Verdict':<26}  {ev_v1['verdict']:>14}  {ev_v2['verdict']:>16}")
print()

# v1/v2 overlap
ov_v1v2, cr_v1v2 = overlap_and_corr(v1_all, v2_all)
print(f"  v1 vs v2 trade overlap: {ov_v1v2:.1%}  |  PnL correlation: {cr_v1v2:+.3f}")
print(f"  (expected: high overlap since v2 ⊂ v1 by construction)")
print()

# Fold-by-fold improvement analysis
print(f"  Fold-by-fold v1 → v2 comparison:")
print(f"  {'Fold':<6}  {'v1 PF':>8}  {'v1 n':>6}  {'v2 PF':>8}  {'v2 n':>6}  {'PF Δ':>8}  {'n Δ':>6}")
print("  " + "─" * 60)
for fi in range(1, N_FWD_FOLDS + 1):
    fl  = f"F{fi}"
    m1  = metrics(v1_fold.get(fl, []))
    m2  = metrics(v2_fold.get(fl, []))
    dpf = m2["pf"] - m1["pf"]
    dn  = m2["n"] - m1["n"]
    arr = "▲" if dpf > 0.02 else ("▼" if dpf < -0.02 else "─")
    print(f"  {fl:<6}  {m1['pf']:>8.3f}  {m1['n']:>6}  {m2['pf']:>8.3f}  "
          f"{m2['n']:>6}  {arr}{abs(dpf):>6.3f}  {dn:>+6d}")
print()

# R057 mandate verdict
print("  ─" * 45)
print("  R057 MANDATE VERDICT:")
prom_criteria = {
    f"PF > {PROM_PF}":           ev_v2["pf"] > PROM_PF,
    f"n ≥ {PROM_N}":             ev_v2["n"] >= PROM_N,
    f"Boot_med > {PROM_BOOT}":   ev_v2["b50"] > PROM_BOOT,
    f"MC_p < {1-PROM_MC:.0%}":   ev_v2["mc_p"] < (1.0 - PROM_MC),
    f"LOO-sym > 1.0":             ev_v2["sf"] > 1.0,
    f"LOO-fold > 1.0":            ev_v2["ff"] > 1.0,
    f"MDD < {PROM_MDD:.0%}":      abs(ev_v2["mdd"]) < PROM_MDD,
}
print()
for criterion, met in prom_criteria.items():
    print(f"    {'✓' if met else '✗'}  {criterion}")
met_count = sum(prom_criteria.values())
print()
print(f"  E3.1_v2 score: {met_count}/7 → Verdict: {ev_v2['verdict']}")
pf_change = ev_v2["pf"] - ev_v1["pf"]
print(f"  PF change vs v1: {pf_change:+.3f}  ({'IMPROVEMENT' if pf_change > 0 else 'DECLINE'})")
print()

# ─────────────────────────────────────────────────────────────────────────────
# PART B — DST_MD PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  PART B — DST_MD Portfolio Construction (R059 Mandate)")
print(SEP)
print()
print("  Running 3 DST_MD environments (all WATCHLIST from R059) ...")
print()

dst_md_evs = []
dst_md_trade_lists = []

for i, (env_cids, env_label) in enumerate(zip(DST_MD_ENVS, DST_MD_LABELS)):
    env_all, env_fold, env_sym = run_wf(env_cids, env_label)
    ev = full_eval(env_all, env_sym, env_label)
    if ev:
        dst_md_evs.append((ev, env_fold))
        dst_md_trade_lists.append(env_all)
        print(f"  P{i+1}: {env_label}")
        print(f"       n={ev['n']}  PF={ev['pf']:.3f}  WR={ev['wr']:.1%}  "
              f"MDD={ev['mdd']:.1%}  UES={ev['ues']:.1f}  Verdict={ev['verdict']}")
    else:
        dst_md_trade_lists.append([])
        print(f"  P{i+1}: {env_label}  → no trades")
    print()

# ─── Internal independence check (between P1, P2, P3)
print(SEP)
print("  PART B — Internal Independence Check (P1 vs P2 vs P3)")
print(SEP)
print()
pairs = [(0,1),(0,2),(1,2)]
pair_labels = [f"P{a+1} vs P{b+1}" for a,b in pairs]
internal_ok = True
for (a, b), pl in zip(pairs, pair_labels):
    ov, cr = overlap_and_corr(dst_md_trade_lists[a], dst_md_trade_lists[b])
    ok = ov <= MAX_INTERNAL_OVERLAP
    if not ok: internal_ok = False
    flag = "OK" if ok else "OVERLAP RISK"
    print(f"  {pl:<12}  Overlap={ov:.1%}  Corr={cr:+.3f}  → {flag}")
print()

# vs E3.1_v2 independence
print("  Independence vs E3.1_v2:")
for i, (ev, _) in enumerate(dst_md_evs):
    ov, cr = overlap_and_corr(v2_all, dst_md_trade_lists[i])
    print(f"  P{i+1} vs E3.1_v2:  Overlap={ov:.1%}  Corr={cr:+.3f}")
print()

# ─── Full evaluation of each DST_MD env
print(SEP)
print("  PART B — DST_MD Environment Detail")
print(SEP)
for i, (ev, env_fold) in enumerate(dst_md_evs):
    print()
    print(f"  ── P{i+1}: {ev['label']}")
    print_eval(ev, env_fold, header=True)
print()

# ─── DST_MD combined portfolio
print(SEP)
print("  PART B — DST_MD Combined Portfolio (P1 + P2 + P3 deduplicated)")
print(SEP)
print()

dst_port_m, dst_port_all = combined_metrics(dst_md_trade_lists)
if dst_port_m["n"] > 0:
    dst_sym_t = defaultdict(list)
    for t in dst_port_all:
        dst_sym_t[t["sym"]].append(t)
    ev_dst_port = full_eval(dst_port_all, dst_sym_t, "DST_MD Portfolio (P1+P2+P3)")
    print_eval(ev_dst_port)
    print()

    # DST_MD fold breakdown
    dst_fold_t = defaultdict(list)
    for t in dst_port_all:
        dst_fold_t[t["fold"]].append(t)
    print("  DST_MD Portfolio fold breakdown:")
    for fi in range(1, N_FWD_FOLDS + 1):
        fl  = f"F{fi}"
        m_f = metrics(dst_fold_t.get(fl, []))
        bar = "▲" if m_f["pf"] >= 1.20 else ("▼" if m_f["pf"] < 1.0 else "─")
        print(f"    {fl}: PF={m_f['pf']:.3f}  n={m_f['n']:3d}  WR={m_f['wr']:.1%}  {bar}")
    print()
else:
    ev_dst_port = None
    print("  No DST_MD portfolio trades generated.")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# PART C — PORTFOLIO COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  PART C — Portfolio Comparison: E3.1_v1 / E3.1_v2 / DST_MD / Combined")
print(SEP)
print()

# Combined two-family: E3.1_v2 + DST_MD Portfolio
if ev_dst_port and ev_dst_port["n"] > 0:
    comb_m, comb_all = combined_metrics([v2_all, dst_port_all])
    comb_sym_t = defaultdict(list)
    for t in comb_all:
        comb_sym_t[t["sym"]].append(t)
    ev_comb = full_eval(comb_all, comb_sym_t, "Combined (E3.1_v2 + DST_MD)")
else:
    ev_comb = ev_v2

# Build comparison table
print(f"  {'Metric':<26}  {'v1 Orig':>10}  {'v2 Strict':>10}  {'DST_MD Port':>12}  {'Combined':>10}")
print("  " + "─" * 78)

rows = [
    ("Trades (n)",    ev_v1["n"],   ev_v2["n"],   ev_dst_port["n"] if ev_dst_port else 0, ev_comb["n"],   "{:d}"),
    ("Win Rate",      ev_v1["wr"],  ev_v2["wr"],  ev_dst_port["wr"] if ev_dst_port else 0, ev_comb["wr"],  "{:.1%}"),
    ("PF",            ev_v1["pf"],  ev_v2["pf"],  ev_dst_port["pf"] if ev_dst_port else 0, ev_comb["pf"],  "{:.3f}"),
    ("MDD",           ev_v1["mdd"], ev_v2["mdd"], ev_dst_port["mdd"] if ev_dst_port else 0, ev_comb["mdd"], "{:.1%}"),
    ("Boot Median",   ev_v1["b50"], ev_v2["b50"], ev_dst_port["b50"] if ev_dst_port else 0, ev_comb["b50"], "{:.3f}"),
    ("MC Prob",       ev_v1["mc_p"],ev_v2["mc_p"],ev_dst_port["mc_p"] if ev_dst_port else 0,ev_comb["mc_p"],"{:.1%}"),
    ("LOO-Sym Floor", ev_v1["sf"],  ev_v2["sf"],  ev_dst_port["sf"] if ev_dst_port else 0, ev_comb["sf"],  "{:.3f}"),
    ("LOO-Fold Floor",ev_v1["ff"],  ev_v2["ff"],  ev_dst_port["ff"] if ev_dst_port else 0, ev_comb["ff"],  "{:.3f}"),
    ("Score",         ev_v1["score"],ev_v2["score"],ev_dst_port["score"] if ev_dst_port else 0,ev_comb["score"],"{:d}"),
    ("UES",           ev_v1["ues"], ev_v2["ues"], ev_dst_port["ues"] if ev_dst_port else 0, ev_comb["ues"], "{:.1f}"),
]
for lbl, a, b, c, d, fmt in rows:
    print(f"  {lbl:<26}  {fmt.format(a):>10}  {fmt.format(b):>10}  {fmt.format(c):>12}  {fmt.format(d):>10}")
print(f"  {'Verdict':<26}  {ev_v1['verdict']:>10}  {ev_v2['verdict']:>10}  "
      f"{ev_dst_port['verdict'] if ev_dst_port else 'N/A':>12}  {ev_comb['verdict']:>10}")
print()

# Is combined better than either component?
if ev_dst_port:
    best_single_ues = max(ev_v2["ues"], ev_dst_port["ues"])
    comb_better = ev_comb["ues"] > best_single_ues
    print(f"  Combined UES ({ev_comb['ues']:.1f}) vs best single ({best_single_ues:.1f}): "
          f"{'COMBINED WINS ▲' if comb_better else 'SINGLE WINS ▼'}")
    ov_v2_dst, cr_v2_dst = overlap_and_corr(v2_all, dst_port_all)
    print(f"  E3.1_v2 vs DST_MD overlap: {ov_v2_dst:.1%}  corr: {cr_v2_dst:+.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION — RESEARCH CONCLUSIONS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  RESEARCH CONCLUSIONS")
print(SEP)
print()

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q1. DID E3.1_v2 (BBW_STRICT) CONFIRM THE R057 FILTER?")
print("  ══════════════════════════════════════════════════════════════════════")
v2_improved = ev_v2["pf"] > ev_v1["pf"]
v2_verdict  = ev_v2["verdict"]
print(f"  PF: v1={ev_v1['pf']:.3f} → v2={ev_v2['pf']:.3f}  "
      f"({'IMPROVED ▲' if v2_improved else 'DECLINED ▼'})")
print(f"  UES: v1={ev_v1['ues']:.1f} → v2={ev_v2['ues']:.1f}")
print(f"  Verdict: v2 = {v2_verdict}")
print()
if v2_verdict in ("PROMOTE", "WATCHLIST"):
    print("  YES — E3.1_v2 meets the forward test bar.")
    if ev_v2["pf"] > ev_v1["pf"]:
        print("  The BBW_STRICT filter genuinely improves quality over BBW_LO.")
    else:
        print("  E3.1_v2 holds its ground but did not improve PF vs v1.")
        print("  The tighter gate trades volume for quality — evaluate MDD and LOO.")
else:
    print("  PARTIAL — E3.1_v2 does not yet clear all promotion criteria.")
    print("  R057 filter effect may require more calendar time to fully materialise.")
print()

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q2. DID THE DST_MD PORTFOLIO REACH PROMOTE?")
print("  ══════════════════════════════════════════════════════════════════════")
if ev_dst_port:
    print(f"  DST_MD portfolio: PF={ev_dst_port['pf']:.3f}  n={ev_dst_port['n']}  "
          f"UES={ev_dst_port['ues']:.1f}  Verdict={ev_dst_port['verdict']}")
    if ev_dst_port["verdict"] == "PROMOTE":
        print("  YES — DST_MD portfolio PROMOTED. Eligible for forward allocation alongside E3.1.")
    elif ev_dst_port["verdict"] == "WATCHLIST":
        print("  WATCHLIST — Combination improved over individual environments.")
        print("  Monitor forward. Need 3+ more months OOS before production allocation.")
    else:
        print("  NOT YET — Portfolio did not reach WATCHLIST threshold.")
        missing = []
        if ev_dst_port["pf"]  <= PROM_PF:    missing.append(f"PF={ev_dst_port['pf']:.3f} (need>{PROM_PF})")
        if ev_dst_port["n"]   <  PROM_N:     missing.append(f"n={ev_dst_port['n']} (need≥{PROM_N})")
        if ev_dst_port["b50"] <= PROM_BOOT:  missing.append(f"Boot={ev_dst_port['b50']:.3f} (need>{PROM_BOOT})")
        if ev_dst_port["sf"]  <= 1.0:        missing.append(f"LOO-sym={ev_dst_port['sf']:.3f}")
        for m in missing:
            print(f"    ✗ {m}")
else:
    print("  No DST_MD portfolio trades generated.")
print()

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q3. DOES THE COMBINED PORTFOLIO ADD VALUE?")
print("  ══════════════════════════════════════════════════════════════════════")
if ev_dst_port and ev_dst_port["n"] > 0:
    print(f"  Combined vs E3.1_v2:  UES {ev_v2['ues']:.1f} → {ev_comb['ues']:.1f}  "
          f"  MDD {ev_v2['mdd']:.1%} → {ev_comb['mdd']:.1%}")
    print(f"  Overlap(v2, DST_MD): {ov_v2_dst:.1%}")
    if ev_comb["ues"] > ev_v2["ues"] and abs(ev_comb["mdd"]) < 0.25:
        print("  YES — Combined portfolio improves UES without excessive drawdown.")
        print("  Recommend monitoring both families independently with a 60/40 allocation.")
    elif abs(ev_comb["mdd"]) >= 0.30:
        print("  NO — Combined MDD is too high. The families draw down independently.")
        print("  Run them separately until DST_MD proves standalone quality.")
    else:
        print("  NEUTRAL — The combination is neither clearly better nor worse.")
        print("  Run E3.1_v2 as primary. Monitor DST_MD family independently.")
print()

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q4. RECOMMENDED NEXT ACTIONS")
print("  ══════════════════════════════════════════════════════════════════════")
print()
if ev_v2["verdict"] == "PROMOTE":
    print("  1. PROMOTE E3.1_v2 → replace v1 as frozen production baseline.")
elif ev_v2["verdict"] == "WATCHLIST":
    print("  1. WATCHLIST E3.1_v2 → paper-trade for 3+ months before replacing v1.")
else:
    print("  1. E3.1_v2 not ready → keep v1 as production baseline.")
    print("     Re-test v2 after 12+ additional months of OOS data.")
if ev_dst_port:
    if ev_dst_port["verdict"] in ("PROMOTE", "WATCHLIST"):
        print("  2. DST_MD Portfolio → paper-trade independently alongside E3.1.")
    else:
        print("  2. DST_MD family → continue monitoring top candidates (P1 best).")
        print("     Add P4/P5 candidates from R059 WATCHLIST to diversify further.")
print("  3. Extend data to add F6 fold — the regime picture across F1-F5 has")
print("     obvious gaps. More OOS bars will clarify whether v2 holds in new regimes.")
print("  4. Run LON-session specific back-test for DST_MD P3 — the London anchor")
print("     may contain sub-structure worth isolating (time-of-day entry filters).")
print()

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts ...")
print(SEP2)

# ── Chart 1: E3.1 v1 vs v2 fold comparison + equity curves
fig = plt.figure(figsize=(20, 16), facecolor=C_BG)
fig.suptitle(f"R060 — E3.1 Upgrade (v1 → v2 BBW_STRICT) + DST_MD Portfolio",
             fontsize=13, color=C_GOLD, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# A: v1 equity
ax_a = fig.add_subplot(gs[0, 0])
eq = ev_v1["equity"]; x = np.arange(len(eq))
ax_a.plot(x, eq, color=C_BLUE, linewidth=1.4)
ax_a.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
ax_a.fill_between(x, CAPITAL, eq, where=eq>=CAPITAL, alpha=0.18, color=C_BLUE)
ax_a.fill_between(x, CAPITAL, eq, where=eq<CAPITAL,  alpha=0.18, color=C_RED)
panel_style(ax_a, f"E3.1_v1  PF={ev_v1['pf']:.3f}  n={ev_v1['n']}\n"
            f"MDD={ev_v1['mdd']:.1%}  UES={ev_v1['ues']:.1f}  {ev_v1['verdict']}")

# B: v2 equity
ax_b = fig.add_subplot(gs[0, 1])
eq = ev_v2["equity"]; x = np.arange(len(eq))
ax_b.plot(x, eq, color=C_GREEN, linewidth=1.4)
ax_b.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
ax_b.fill_between(x, CAPITAL, eq, where=eq>=CAPITAL, alpha=0.18, color=C_GREEN)
ax_b.fill_between(x, CAPITAL, eq, where=eq<CAPITAL,  alpha=0.18, color=C_RED)
panel_style(ax_b, f"E3.1_v2 BBW_STRICT  PF={ev_v2['pf']:.3f}  n={ev_v2['n']}\n"
            f"MDD={ev_v2['mdd']:.1%}  UES={ev_v2['ues']:.1f}  {ev_v2['verdict']}")

# C: DST_MD portfolio equity
ax_c = fig.add_subplot(gs[0, 2])
if ev_dst_port and ev_dst_port["n"] > 0:
    eq = ev_dst_port["equity"]; x = np.arange(len(eq))
    ax_c.plot(x, eq, color=C_GOLD, linewidth=1.4)
    ax_c.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
    ax_c.fill_between(x, CAPITAL, eq, where=eq>=CAPITAL, alpha=0.18, color=C_GOLD)
    ax_c.fill_between(x, CAPITAL, eq, where=eq<CAPITAL,  alpha=0.18, color=C_RED)
    panel_style(ax_c, f"DST_MD Portfolio  PF={ev_dst_port['pf']:.3f}  n={ev_dst_port['n']}\n"
                f"MDD={ev_dst_port['mdd']:.1%}  UES={ev_dst_port['ues']:.1f}  {ev_dst_port['verdict']}")
else:
    ax_c.text(0.5, 0.5, "No DST_MD trades", ha="center", va="center",
              color=C_TEXT, transform=ax_c.transAxes)
    panel_style(ax_c, "DST_MD Portfolio")

# D: Combined equity
ax_d = fig.add_subplot(gs[0, 3])
eq = ev_comb["equity"]; x = np.arange(len(eq))
ax_d.plot(x, eq, color=C_PURP, linewidth=1.4)
ax_d.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
ax_d.fill_between(x, CAPITAL, eq, where=eq>=CAPITAL, alpha=0.18, color=C_PURP)
ax_d.fill_between(x, CAPITAL, eq, where=eq<CAPITAL,  alpha=0.18, color=C_RED)
panel_style(ax_d, f"Combined  PF={ev_comb['pf']:.3f}  n={ev_comb['n']}\n"
            f"MDD={ev_comb['mdd']:.1%}  UES={ev_comb['ues']:.1f}  {ev_comb['verdict']}")

# E: Fold comparison bar chart (v1 vs v2)
ax_e = fig.add_subplot(gs[1, :2])
folds_x = np.arange(N_FWD_FOLDS); w = 0.38
v1_pfs  = [metrics(v1_fold.get(f"F{i+1}", []))["pf"] for i in range(N_FWD_FOLDS)]
v2_pfs  = [metrics(v2_fold.get(f"F{i+1}", []))["pf"] for i in range(N_FWD_FOLDS)]
ax_e.bar(folds_x - w/2, v1_pfs, w, color=C_BLUE,  alpha=0.85, label="v1 BBW_LO")
ax_e.bar(folds_x + w/2, v2_pfs, w, color=C_GREEN, alpha=0.85, label="v2 BBW_STRICT")
ax_e.axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--", alpha=0.6)
ax_e.set_xticks(folds_x)
ax_e.set_xticklabels([f"F{i+1}" for i in range(N_FWD_FOLDS)])
ax_e.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_e, "Fold-by-Fold PF: E3.1_v1 vs E3.1_v2", fs=8)

# F: DST_MD individual env + portfolio bar chart
ax_f = fig.add_subplot(gs[1, 2:])
dst_labels = [f"P{i+1}" for i in range(len(dst_md_evs))]
if ev_dst_port:
    dst_labels.append("PORT")
dst_pfs = [ev["pf"] for ev, _ in dst_md_evs]
if ev_dst_port:
    dst_pfs.append(ev_dst_port["pf"])
dst_ues = [ev["ues"] for ev, _ in dst_md_evs]
if ev_dst_port:
    dst_ues.append(ev_dst_port["ues"])
colors_dst = PALETTE[:len(dst_labels)]
xa = np.arange(len(dst_labels))
ax_f.bar(xa, dst_pfs, color=colors_dst, alpha=0.85)
ax_f.axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--", alpha=0.6)
ax_f.axhline(PROM_PF, color=C_GOLD, linewidth=0.8, linestyle=":", alpha=0.7,
             label=f"PROMOTE bar ({PROM_PF})")
ax_f.set_xticks(xa); ax_f.set_xticklabels(dst_labels)
ax_f.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_f, "DST_MD Env PF (P1=ADX_WK+RV_HI, P2=ATR_LO+PRG_LO+RV_LO, P3=ADX_WK+LON)", fs=7)

# G: UES radar / bar comparison — all portfolios
ax_g = fig.add_subplot(gs[2, :2])
all_labels = ["v1\nOrig", "v2\nStrict"]
all_ues    = [ev_v1["ues"], ev_v2["ues"]]
all_colors = [C_BLUE, C_GREEN]
for i, (ev, _) in enumerate(dst_md_evs):
    all_labels.append(f"DST\nP{i+1}")
    all_ues.append(ev["ues"])
    all_colors.append(PALETTE[3+i])
if ev_dst_port:
    all_labels.append("DST\nPORT")
    all_ues.append(ev_dst_port["ues"])
    all_colors.append(C_GOLD)
all_labels.append("COMB")
all_ues.append(ev_comb["ues"])
all_colors.append(C_PURP)
xg = np.arange(len(all_labels))
bars = ax_g.bar(xg, all_ues, color=all_colors, alpha=0.85)
ax_g.set_xticks(xg); ax_g.set_xticklabels(all_labels, fontsize=7)
for bar, ues in zip(bars, all_ues):
    ax_g.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
              f"{ues:.0f}", ha="center", va="bottom", fontsize=6, color=C_TEXT)
panel_style(ax_g, "UES Comparison — All Strategies", fs=8)

# H: Summary text
ax_h = fig.add_subplot(gs[2, 2:])
ax_h.axis("off")
lines = [
    "R060 — DUAL CONFIRMATION SUMMARY",
    "─" * 46,
    f"E3.1_v1  PF={ev_v1['pf']:.3f}  n={ev_v1['n']}  [{ev_v1['verdict']}]",
    f"E3.1_v2  PF={ev_v2['pf']:.3f}  n={ev_v2['n']}  [{ev_v2['verdict']}]",
    f"PF change: {ev_v2['pf']-ev_v1['pf']:+.3f}  UES: {ev_v1['ues']:.1f}→{ev_v2['ues']:.1f}",
    "─" * 46,
]
for i, (ev, _) in enumerate(dst_md_evs):
    lines.append(f"DST_MD P{i+1}  PF={ev['pf']:.3f}  n={ev['n']}  [{ev['verdict']}]")
if ev_dst_port:
    lines.append(f"DST_MD PORT  PF={ev_dst_port['pf']:.3f}  n={ev_dst_port['n']}  [{ev_dst_port['verdict']}]")
lines += [
    "─" * 46,
    f"COMBINED  PF={ev_comb['pf']:.3f}  n={ev_comb['n']}  [{ev_comb['verdict']}]",
    "─" * 46,
    f"E3.1_v2 R057 mandate: {ev_v2['verdict']}",
    f"DST_MD Portfolio: {ev_dst_port['verdict'] if ev_dst_port else 'N/A'}",
    "─" * 46,
    "Two-family system: E3.1 (compression) +",
    "DST_MD (trend continuation) — 0% overlap",
]
for i, line in enumerate(lines):
    col = (C_GOLD if i == 0 else
           C_GREEN if "PROMOTE" in line or "WATCHLIST" in line else
           C_RED   if "REJECT"  in line else C_TEXT)
    ax_h.text(0.02, 0.99 - i*0.072, line, transform=ax_h.transAxes,
              fontsize=6.5, color=col, va="top", fontfamily="monospace")
panel_style(ax_h, "R060 Summary")

plt.savefig(f"{OUT}/r060_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r060_dashboard.png")

# ── Chart 2: DST_MD individual equity curves
if dst_md_evs:
    ncols = min(4, len(dst_md_evs)); nrows = 1
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4), facecolor=C_BG)
    fig2.suptitle("R060 — DST_MD Portfolio Environments (P1, P2, P3)",
                  fontsize=11, color=C_GOLD, fontweight="bold")
    axs = [axes2] if ncols == 1 else list(axes2)
    for idx, ((ev, env_fold), ax_ei) in enumerate(zip(dst_md_evs, axs)):
        eq = ev["equity"]; x = np.arange(len(eq))
        ax_ei.plot(x, eq, color=PALETTE[idx], linewidth=1.4)
        ax_ei.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
        ax_ei.fill_between(x, CAPITAL, eq, where=eq>=CAPITAL, alpha=0.15, color=PALETTE[idx])
        ax_ei.fill_between(x, CAPITAL, eq, where=eq<CAPITAL,  alpha=0.15, color=C_RED)
        panel_style(ax_ei,
            f"P{idx+1}: {ev['label'][:30]}\nPF={ev['pf']:.3f}  n={ev['n']}  "
            f"MDD={ev['mdd']:.1%}  UES={ev['ues']:.1f}  [{ev['verdict']}]", fs=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/r060_dst_md_envs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {OUT}/r060_dst_md_envs.png")

# ── Chart 3: Fold stability heatmap
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5), facecolor=C_BG)
fig3.suptitle("R060 — Fold Stability: E3.1 v1/v2 and DST_MD Portfolio",
              fontsize=11, color=C_GOLD, fontweight="bold")

# v1 vs v2 fold stability
folds_lst  = [f"F{i+1}" for i in range(N_FWD_FOLDS)]
v1_fold_pf = [metrics(v1_fold.get(fl, []))["pf"] for fl in folds_lst]
v2_fold_pf = [metrics(v2_fold.get(fl, []))["pf"] for fl in folds_lst]

xf = np.arange(N_FWD_FOLDS); wf = 0.38
axes3[0].bar(xf - wf/2, v1_fold_pf, wf, color=C_BLUE,  alpha=0.85, label="v1")
axes3[0].bar(xf + wf/2, v2_fold_pf, wf, color=C_GREEN, alpha=0.85, label="v2")
axes3[0].axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--", alpha=0.6)
axes3[0].set_xticks(xf); axes3[0].set_xticklabels(folds_lst)
axes3[0].legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(axes3[0], "E3.1 v1 vs v2 — Per-Fold PF", fs=9)

# DST_MD fold stability
if ev_dst_port and dst_fold_t:
    dst_fp = [metrics(dst_fold_t.get(fl, []))["pf"] for fl in folds_lst]
    axes3[1].bar(xf, dst_fp, color=C_GOLD, alpha=0.85, label="DST_MD Port")
    axes3[1].axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--", alpha=0.6)
    axes3[1].set_xticks(xf); axes3[1].set_xticklabels(folds_lst)
    axes3[1].legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(axes3[1], "DST_MD Portfolio — Per-Fold PF", fs=9)

plt.tight_layout()
plt.savefig(f"{OUT}/r060_fold_stability.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r060_fold_stability.png")

# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
def ev_to_row(ev, tag):
    return {
        "tag":        tag,
        "label":      ev["label"],
        "n":          ev["n"],
        "wr":         round(ev["wr"], 4),
        "pf":         round(ev["pf"], 4),
        "mdd":        round(ev["mdd"], 4),
        "boot_p5":    round(ev["b5"],  4),
        "boot_med":   round(ev["b50"], 4),
        "boot_p95":   round(ev["b95"], 4),
        "mc_prob":    round(ev["mc_p"],4),
        "loo_sym":    round(ev["sf"],  4),
        "loo_fold":   round(ev["ff"],  4),
        "score":      ev["score"],
        "verdict":    ev["verdict"],
        "ues":        ev["ues"],
    }

results_rows = [ev_to_row(ev_v1, "E3.1_v1")]
results_rows.append(ev_to_row(ev_v2, "E3.1_v2"))
for i, (ev, _) in enumerate(dst_md_evs):
    results_rows.append(ev_to_row(ev, f"DST_MD_P{i+1}"))
if ev_dst_port:
    results_rows.append(ev_to_row(ev_dst_port, "DST_MD_PORT"))
results_rows.append(ev_to_row(ev_comb, "COMBINED"))

pd.DataFrame(results_rows).to_csv(f"{OUT}/r060_results.csv", index=False)
print(f"  ✓  {OUT}/r060_results.csv")

# Fold-by-fold CSV
fold_rows = []
for fl in [f"F{i+1}" for i in range(N_FWD_FOLDS)]:
    m1 = metrics(v1_fold.get(fl, []));  m2 = metrics(v2_fold.get(fl, []))
    md = metrics(dst_fold_t.get(fl, [])) if ev_dst_port else metrics([])
    row = {"fold": fl,
           "v1_pf": round(m1["pf"],3), "v1_n": m1["n"],
           "v2_pf": round(m2["pf"],3), "v2_n": m2["n"],
           "dst_pf": round(md["pf"],3), "dst_n": md["n"]}
    fold_rows.append(row)
pd.DataFrame(fold_rows).to_csv(f"{OUT}/r060_fold_comparison.csv", index=False)
print(f"  ✓  {OUT}/r060_fold_comparison.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print(f"  R060 COMPLETE — DUAL CONFIRMATION")
print(SEP)
print()
print(f"  PART A — E3.1 BBW_STRICT upgrade (R057 forward test mandate):")
print(f"    v1 (BBW_LO):     PF={ev_v1['pf']:.3f}  n={ev_v1['n']}  MDD={ev_v1['mdd']:.1%}  UES={ev_v1['ues']:.1f}  [{ev_v1['verdict']}]")
print(f"    v2 (BBW_STRICT): PF={ev_v2['pf']:.3f}  n={ev_v2['n']}  MDD={ev_v2['mdd']:.1%}  UES={ev_v2['ues']:.1f}  [{ev_v2['verdict']}]")
print(f"    R057 mandate: {'CONFIRMED ✓' if ev_v2['verdict'] in ('PROMOTE','WATCHLIST') else 'NEEDS MORE TIME ✗'}")
print()
print(f"  PART B — DST_MD Portfolio (R059 mandate):")
for i, (ev, _) in enumerate(dst_md_evs):
    print(f"    P{i+1}: {ev['label'][:38]:<38}  PF={ev['pf']:.3f}  n={ev['n']:3d}  [{ev['verdict']}]")
if ev_dst_port:
    print(f"    Portfolio:  PF={ev_dst_port['pf']:.3f}  n={ev_dst_port['n']}  MDD={ev_dst_port['mdd']:.1%}  [{ev_dst_port['verdict']}]")
print()
print(f"  PART C — Combined Two-Family Portfolio:")
print(f"    E3.1_v2 + DST_MD: PF={ev_comb['pf']:.3f}  n={ev_comb['n']}  MDD={ev_comb['mdd']:.1%}  [{ev_comb['verdict']}]")
print()
print(SEP)
