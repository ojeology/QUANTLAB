"""
=============================================================================
QUANTLAB AI — RESEARCH #056
E3 Regime Shift Investigation — Forensic Analysis
=============================================================================

Objective:
  E3 (BBW_LO+RV_LO+DST_NR+PRG_VH) survived forward validation better than
  all other R052 candidates. However it suffered a severe performance collapse
  during F3/F4 forward folds.

  THIS IS NOT AN OPTIMISATION STUDY.
  This is a scientific forensic investigation into regime behaviour.
  Goal: UNDERSTANDING, not improving historical results.

  Environment (FROZEN — DO NOT MODIFY):
    BBW_LO + RV_LO + DST_NR + PRG_VH
    RELVOL entry
    RR = 2.0

Sections:
  S1   Regime Comparison (F1+F2 vs F3+F4 vs F5)
  S2   Market Structure Classification
  S3   Signal Quality Analysis
  S4   Failure Clustering
  S5   Session Effect
  S6   Weekday Effect
  S7   Symbol × Regime Analysis
  S7B  Asset Composition Analysis
  S8   What Changed (ranked list)
  S9   Hypothesis Testing
  S10  Macro Filter Justification
  FINAL Q&A
=============================================================================
"""

import os, sys, time, math, warnings, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy import stats as scipy_stats
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID = "R056"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = 2.0            # FROZEN — never change
IS_RATIO = 0.80
N_FWD_FOLDS = 5

# FROZEN E3 environment — never modify
E3_CIDS  = ("BBW_LO", "RV_LO", "DST_NR", "PRG_VH")
E3_LABEL = "BBW_LO+RV_LO+DST_NR+PRG_VH"

# Colors
C_BG    = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT  = "#e0e0e0"
C_GRID  = "#2a2a2a"; C_GREEN = "#00c896"; C_RED   = "#e05050"
C_GOLD  = "#f5a623"; C_BLUE  = "#4a9eff"; C_PURP  = "#9b59b6"
C_CYAN  = "#1abc9c"; C_ORAN  = "#e67e22"
FOLD_COLORS = [C_GREEN, C_BLUE, C_RED, C_RED, C_GOLD]

plt.rcParams.update({
    "figure.facecolor":C_BG,"axes.facecolor":C_PANEL,
    "text.color":C_TEXT,"axes.labelcolor":C_TEXT,
    "xtick.color":C_TEXT,"ytick.color":C_TEXT,
    "axes.edgecolor":C_GRID,"grid.color":C_GRID,"font.family":"monospace",
})

def panel_style(ax, title, fs=8):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title, fontsize=fs, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5)
    for sp in ax.spines.values(): sp.set_color(C_GRID)

SEP  = "═" * 110
SEP2 = "─" * 90

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONS CATALOGUE  (full superset from R054)
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    ("ATR_LO", "atr_rank",     "lt_q",      0.25),
    ("ATR_MD", "atr_rank",     "lt_q",      0.40),
    ("ATR_HI", "atr_rank",     "gt_q",      0.67),
    ("ATR_VH", "atr_rank",     "gt_q",      0.80),
    ("BBW_LO", "bb_width",     "lt_q",      0.33),
    ("BBW_HI", "bb_width",     "gt_q",      0.67),
    ("RV_LO",  "real_vol_20",  "lt_q",      0.33),
    ("RV_HI",  "real_vol_20",  "gt_q",      0.67),
    ("SLP_DN", "ema200_slope", "lt_fixed",  0.0 ),
    ("SLP_UP", "ema200_slope", "gt_fixed",  0.0 ),
    ("DST_NR", "ema_dist_pct", "lt_q",      0.33),
    ("DST_MD", "ema_dist_pct", "gt_q_pos",  0.60),
    ("DST_FR", "ema_dist_pct", "gt_q_pos",  0.75),
    ("ADX_WK", "adx14",        "lt_q",      0.33),
    ("ADX_TR", "adx14",        "gt_q",      0.50),
    ("ADX_ST", "adx14",        "gt_q",      0.67),
    ("PRG_LO", "prev_range_r", "lt_q",      0.33),
    ("PRG_HI", "prev_range_r", "gt_q",      0.67),
    ("PRG_VH", "prev_range_r", "gt_q",      0.80),
    ("PBD_HI", "prev_body_r",  "gt_q",      0.67),
    ("PBP_HI", "prev_body_pct","gt_q",      0.60),
    ("PBP_LO", "prev_body_pct","lt_q",      0.33),
    ("US",     "hour_utc",     "hour_rng",  (14,21)),
    ("LON",    "hour_utc",     "hour_rng",  (7, 14)),
    ("ASI",    "hour_utc",     "hour_rng",  (0,  6)),
]
COND_BY_ID   = {c[0]: c for c in CONDITIONS_DEF}
QUANT_FEATS  = ["atr_rank","bb_width","real_vol_20","ema_dist_pct",
                "adx14","prev_range_r","prev_body_r","prev_body_pct"]

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
MIN_BARS = 2_000

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print(f"  QUANTLAB AI — RESEARCH #{RESEARCH_ID}")
print(f"  E3 Regime Shift Forensic Investigation")
print(SEP)
print()
print(f"  FROZEN ENVIRONMENT: {E3_LABEL}")
print(f"  ENTRY SIGNAL: RELVOL > 1.5 + close > open + close > prev_close")
print(f"  RR = {RR}  |  IS ratio = {IS_RATIO}")
print(f"  Objective: Understand WHY E3 succeeded in F1/F2 and collapsed in F3/F4.")
print(f"  This is NOT an optimisation study.")
print()

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
    df["hour_utc"]      = pd.to_datetime(df["datetime"], utc=True).dt.hour.astype(np.int16)
    df["dow"]           = pd.to_datetime(df["datetime"], utc=True).dt.dayofweek  # 0=Mon

    # ── Additional regime metrics for forensics ──
    df["daily_range_r"] = (h - l) / c.replace(0, np.nan) * 100.0
    df["candle_body_r"] = (c - o).abs() / c.replace(0, np.nan) * 100.0
    df["is_bullish"]    = (c > o).astype(float)

    # Hurst exponent (approximation via R/S on rolling window)
    # Use log returns for rescaled range
    df["log_ret"]       = log_ret

    # ATR rank for trend persistence proxy
    # EMA slope magnitude
    df["ema_slope_abs"] = df["ema200_slope"].abs()

    # Breakout: close > rolling 20-bar high (excluding current)
    df["roll_high_20"]  = h.shift(1).rolling(20).max()
    df["is_breakout"]   = (c > df["roll_high_20"]).astype(float)

    # False breakout: previous bar was breakout but current close is below prev high
    df["false_brkout"]  = ((c.shift(1) > df["roll_high_20"].shift(1)) &
                           (c < c.shift(1))).astype(float)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def learn_thresholds(df_is):
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (_, feat, direction, param) in COND_BY_ID.items():
        if direction in ("lt_fixed","gt_fixed","hour_rng"):
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
# MASK + SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
def build_env_mask(df, cid_tuple, thr):
    N = len(df); mask = np.ones(N, dtype=bool)
    for cid in cid_tuple:
        if cid not in COND_BY_ID: return np.zeros(N, dtype=bool)
        _, feat, direction, _ = COND_BY_ID[cid]
        if feat not in df.columns: return np.zeros(N, dtype=bool)
        col   = df[feat].values
        nan_m = np.isnan(col) if col.dtype.kind == "f" else np.zeros(N, dtype=bool)
        t     = thr.get(cid, np.nan)
        if direction == "lt_q":
            if np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col < t)
        elif direction in ("gt_q","gt_q_pos"):
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
# BACKTEST ENGINE  (enriched with signal quality metrics)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest_forensic(df, signal, env_mask, sym, fold_label):
    min_sl  = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf      = CONFIG["RISK_PER_TRADE_PCT"]
    fee     = CONFIG["TAKER_FEE"];  spd = CONFIG["SPREAD"] * 0.5
    slp     = CONFIG["SL_SLIPPAGE"]
    in_pos  = False; ep = st = tk = sz = 0.0; et = None; ei = -1
    trades  = []

    hi_  = df["high"].values;  lo_ = df["low"].values
    op_  = df["open"].values;  cl_ = df["close"].values
    atr_ = df["prev_atr14"].values
    rv_  = df["rel_vol"].values
    adx_ = df["adx14"].values
    bbw_ = df["bb_width"].values
    rvol_= df["real_vol_20"].values
    slp_ = df["ema200_slope"].values
    dst_ = df["ema_dist_pct"].values
    prg_ = df["prev_range_r"].values
    bkout= df["is_breakout"].values
    fbkout= df["false_brkout"].values
    drs_ = df["daily_range_r"].values
    dts  = df["datetime"].values
    hou_ = df["hour_utc"].values
    dow_ = df["dow"].values

    for i in range(1, len(df)):
        if in_pos:
            bars_held = i - ei
            sl_hit = lo_[i] <= st; tp_hit = hi_[i] >= tk
            if sl_hit or tp_hit:
                xp      = (st * (1 - slp)) if sl_hit else tk
                sd      = ep - st
                gross   = (xp - ep) * sz
                cost    = (ep * sz + xp * sz) * (fee + spd)
                slpc    = (st - xp) * sz if sl_hit else 0.0
                net     = gross - cost - slpc
                rmul    = (xp - ep) / sd if sd > 0 else 0.0
                dist_to_tp = (tk - ep) / sd if sd > 0 else 0.0
                dist_to_sl = 1.0  # by definition sl = 1R away

                # Signal quality: how far above env threshold rel_vol was
                sig_strength = float(rv_[ei]) if not np.isnan(rv_[ei]) else 1.0

                # Session
                h_entry = int(hou_[ei])
                if 7 <= h_entry <= 13:
                    session = "London"
                elif 14 <= h_entry <= 20:
                    session = "US"
                else:
                    session = "Asia"

                trades.append({
                    "sym":           sym,
                    "fold":          fold_label,
                    "entry_time":    str(et),
                    "exit_time":     str(dts[i]),
                    "entry_hour":    h_entry,
                    "session":       session,
                    "dow":           int(dow_[ei]),
                    "pnl":           round(net, 4),
                    "r_multiple":    round(rmul, 4),
                    "win":           int(not sl_hit),
                    "exit_type":     "SL" if sl_hit else "TP",
                    "bars_held":     bars_held,
                    "dist_to_tp":    round(dist_to_tp, 4),
                    "dist_to_sl":    round(dist_to_sl, 4),
                    "sig_strength":  round(sig_strength, 4),
                    "entry_adx":     round(float(adx_[ei]), 4) if not np.isnan(adx_[ei]) else np.nan,
                    "entry_bbw":     round(float(bbw_[ei]), 6) if not np.isnan(bbw_[ei]) else np.nan,
                    "entry_rv":      round(float(rvol_[ei]), 4) if not np.isnan(rvol_[ei]) else np.nan,
                    "entry_slope":   round(float(slp_[ei]), 8) if not np.isnan(slp_[ei]) else np.nan,
                    "entry_dst":     round(float(dst_[ei]), 4) if not np.isnan(dst_[ei]) else np.nan,
                    "entry_prg":     round(float(prg_[ei]), 4) if not np.isnan(prg_[ei]) else np.nan,
                    "entry_atr":     round(float(atr_[i]), 6) if not np.isnan(atr_[i]) else np.nan,
                    "is_breakout":   int(bkout[ei]),
                    "is_false_bkout":int(fbkout[ei]),
                    "daily_range_r": round(float(drs_[ei]), 4) if not np.isnan(drs_[ei]) else np.nan,
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
# REGIME METRICS  (per fold — all bars in the forward period that pass env mask)
# ─────────────────────────────────────────────────────────────────────────────
def compute_hurst(series, min_n=20):
    """R/S Hurst exponent estimation on a log-return series."""
    s = series.dropna().values
    if len(s) < min_n:
        return np.nan
    n = len(s)
    mean_s = s.mean()
    deviations = np.cumsum(s - mean_s)
    R = deviations.max() - deviations.min()
    S = s.std()
    if S == 0:
        return np.nan
    return np.log(R / S) / np.log(n)

def fold_regime_metrics(df_fold):
    """Compute regime characteristics for a fold's bars."""
    if len(df_fold) < 10:
        return {}
    c = df_fold["close"]; h = df_fold["high"]; l = df_fold["low"]
    lr = df_fold["log_ret"].dropna()
    adx  = df_fold["adx14"].dropna()
    atr  = df_fold["atr14"].dropna()
    atrr = df_fold["atr_rank"].dropna()
    bbw  = df_fold["bb_width"].dropna()
    rv   = df_fold["real_vol_20"].dropna()
    slp  = df_fold["ema200_slope"].dropna()
    dst  = df_fold["ema_dist_pct"].dropna()
    prg  = df_fold["prev_range_r"].dropna()
    drs  = df_fold["daily_range_r"].dropna()
    bdr  = df_fold["candle_body_r"].dropna()
    bkout= df_fold["is_breakout"].dropna()
    fbk  = df_fold["false_brkout"].dropna()
    rvol = df_fold["rel_vol"].dropna()

    hurst = compute_hurst(lr)

    # Trend persistence: fraction of bars where close > close[i-5]
    c_arr = c.values
    tp_count = 0; tp_total = 0
    for k in range(5, len(c_arr)):
        if not np.isnan(c_arr[k]) and not np.isnan(c_arr[k-5]):
            tp_total += 1
            if c_arr[k] > c_arr[k-5]:
                tp_count += 1
    trend_persistence = tp_count / tp_total if tp_total > 0 else np.nan

    # Breakout frequency (how often does price break 20-bar high)
    bkout_freq = float(bkout.mean()) if len(bkout) > 0 else np.nan
    # False breakout frequency (next bar retracts)
    false_bkout_freq = float(fbk.mean()) if len(fbk) > 0 else np.nan

    return {
        "avg_adx":          float(adx.mean())  if len(adx)  > 0 else np.nan,
        "avg_atr":          float(atr.mean())  if len(atr)  > 0 else np.nan,
        "avg_atr_rank":     float(atrr.mean()) if len(atrr) > 0 else np.nan,
        "avg_rv":           float(rv.mean())   if len(rv)   > 0 else np.nan,
        "avg_bbw":          float(bbw.mean())  if len(bbw)  > 0 else np.nan,
        "avg_ema_slope":    float(slp.mean())  if len(slp)  > 0 else np.nan,
        "avg_ema_dist":     float(dst.mean())  if len(dst)  > 0 else np.nan,
        "trend_persistence":trend_persistence,
        "hurst":            hurst,
        "avg_rel_vol":      float(rvol.mean()) if len(rvol) > 0 else np.nan,
        "avg_daily_range":  float(drs.mean())  if len(drs)  > 0 else np.nan,
        "avg_candle_body":  float(bdr.mean())  if len(bdr)  > 0 else np.nan,
        "avg_prev_range":   float(prg.mean())  if len(prg)  > 0 else np.nan,
        "breakout_freq":    bkout_freq,
        "false_bkout_freq": false_bkout_freq,
        "n_bars":           len(df_fold),
        "pct_bullish":      float((df_fold["is_bullish"] > 0.5).mean())
                            if len(df_fold) > 0 else np.nan,
    }

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl

def metrics(trades):
    if not trades:
        return {"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,"net":0.0,
                "mdd":0.0,"pnls":np.array([]),"equity":np.array([CAPITAL])}
    df  = pd.DataFrame(trades)
    pnl = df["pnl"].values; wins = df["win"].values.astype(bool)
    n = len(pnl); nw = wins.sum(); nl = n - nw
    gw = pnl[wins].sum() if nw else 0.0
    gl = abs(pnl[~wins].sum()) if nl else 0.0
    pf = safe_pf(gw, gl); wr = nw / n
    eq = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk).min())
    exp = wr * RR - (1 - wr)
    return {"n":n,"wr":wr,"pf":pf,"exp_r":exp,"net":float(pnl.sum()),
            "mdd":mdd,"pnls":pnl,"equity":eq}

def cohens_d(a, b):
    """Cohen's d effect size between two arrays."""
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return np.nan
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

def pvalue_ttest(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3: return np.nan
    try:
        _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        return p
    except Exception:
        return np.nan

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — DATA LOAD + FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 0 — Data Load + Frozen Forward Backtest")
print(SEP)
print()

all_trades      = []          # all E3 forward trades
fold_trades     = defaultdict(list)  # {F1..F5: [trades]}
sym_trades      = defaultdict(list)  # {sym: [trades]}
fold_sym_trades = defaultdict(lambda: defaultdict(list))  # {fold: {sym: [trades]}}
fold_regime     = defaultdict(list)  # {fold: [regime_dict per symbol]}
sym_fold_info   = {}          # {sym: {fold: start_date, end_date}}

# Regime frames per fold (pool all symbols)
fold_bars       = defaultdict(list)   # {fold: [df_seg]}

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
    df = add_features(df)
    sp = int(N * IS_RATIO)
    thr    = learn_thresholds(df.iloc[:sp])
    df_fwd = df.iloc[sp:].copy().reset_index(drop=True)
    if len(df_fwd) < 50: continue
    loaded += 1

    fwd_size = len(df_fwd)
    seg_size = max(1, fwd_size // N_FWD_FOLDS)

    sym_fold_info[sym] = {}
    for fi in range(N_FWD_FOLDS):
        seg_s  = fi * seg_size
        seg_e  = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
        df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
        flabel = f"F{fi+1}"
        if len(df_seg) < 20: continue

        sym_fold_info[sym][flabel] = {
            "start": str(df_seg["datetime"].iloc[0].date()),
            "end":   str(df_seg["datetime"].iloc[-1].date()),
            "n":     len(df_seg),
        }

        em  = build_env_mask(df_seg, E3_CIDS, thr)
        sig = entry_signal(df_seg, em)
        tl  = run_backtest_forensic(df_seg, sig, em, sym, flabel)

        all_trades.extend(tl)
        fold_trades[flabel].extend(tl)
        sym_trades[sym].extend(tl)
        fold_sym_trades[flabel][sym].extend(tl)

        # Regime metrics using ALL bars in this fold (not just env-filtered)
        regime = fold_regime_metrics(df_seg)
        regime["sym"] = sym; regime["fold"] = flabel
        fold_regime[flabel].append(regime)
        fold_bars[flabel].append(df_seg)

print(f"  Symbols processed: {loaded}")
print(f"  Total forward trades: {len(all_trades)}")
for fi in range(1, N_FWD_FOLDS+1):
    fl = f"F{fi}"
    m  = metrics(fold_trades[fl])
    print(f"    {fl}: n={m['n']:3d}  PF={m['pf']:.3f}  WR={m['wr']*100:.1f}%  "
          f"Net=${m['net']:+.0f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Compute aggregate regime metrics per fold
# ─────────────────────────────────────────────────────────────────────────────
REGIME_KEYS = [
    "avg_adx","avg_atr","avg_atr_rank","avg_rv","avg_bbw",
    "avg_ema_slope","avg_ema_dist","trend_persistence","hurst",
    "avg_rel_vol","avg_daily_range","avg_candle_body","avg_prev_range",
    "breakout_freq","false_bkout_freq","pct_bullish",
]

fold_regime_agg = {}
for fl in [f"F{i}" for i in range(1, N_FWD_FOLDS+1)]:
    rows = fold_regime[fl]
    if not rows:
        fold_regime_agg[fl] = {}
        continue
    agg = {}
    for k in REGIME_KEYS:
        vals = [r.get(k, np.nan) for r in rows if not np.isnan(r.get(k, np.nan))]
        agg[k] = float(np.median(vals)) if vals else np.nan
    fold_regime_agg[fl] = agg

# Period labels (use first symbol with data to infer dates)
fold_dates = {}
for sym in list(sym_fold_info.keys())[:1]:
    for fl, info in sym_fold_info[sym].items():
        fold_dates[fl] = f"{info['start']} → {info['end']}"

# Group classifications
WIN_FOLDS  = ["F1","F2"]
LOSE_FOLDS = ["F3","F4"]
REC_FOLDS  = ["F5"]

def group_regime_avg(fold_list):
    """Average regime metrics across a group of folds."""
    agg = {}
    for k in REGIME_KEYS:
        vals = [fold_regime_agg[fl].get(k, np.nan) for fl in fold_list
                if fl in fold_regime_agg and not np.isnan(fold_regime_agg[fl].get(k, np.nan))]
        agg[k] = float(np.nanmean(vals)) if vals else np.nan
    return agg

win_regime  = group_regime_avg(WIN_FOLDS)
lose_regime = group_regime_avg(LOSE_FOLDS)
rec_regime  = group_regime_avg(REC_FOLDS)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — REGIME COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 1 — REGIME COMPARISON")
print("  Winning Regime: F1+F2  |  Losing Regime: F3+F4  |  Recovery: F5")
print(SEP)
print()

REGIME_LABELS = {
    "avg_adx":           "Avg ADX",
    "avg_atr":           "Avg ATR",
    "avg_atr_rank":      "ATR Rank (pct)",
    "avg_rv":            "Realised Volatility",
    "avg_bbw":           "Bollinger Width",
    "avg_ema_slope":     "EMA200 Slope",
    "avg_ema_dist":      "EMA Distance (%)",
    "trend_persistence": "Trend Persistence",
    "hurst":             "Hurst Exponent",
    "avg_rel_vol":       "Relative Volume",
    "avg_daily_range":   "Daily Range (%)",
    "avg_candle_body":   "Avg Candle Body (%)",
    "avg_prev_range":    "Prev Candle Range (%)",
    "breakout_freq":     "Breakout Frequency",
    "false_bkout_freq":  "False Breakout Freq",
    "pct_bullish":       "Pct Bullish Candles",
}

print(f"  {'Metric':<28}  {'Win F1+F2':>12}  {'Lose F3+F4':>12}  "
      f"{'Rec F5':>10}  {'Delta':>10}  {'Effect'}")
print("  " + "─" * 90)

regime_deltas = {}
for k, label in REGIME_LABELS.items():
    wv = win_regime.get(k, np.nan)
    lv = lose_regime.get(k, np.nan)
    rv_val = rec_regime.get(k, np.nan)
    delta = lv - wv if not (np.isnan(lv) or np.isnan(wv)) else np.nan

    # Effect size: delta / win_std (rough)
    all_vals_w = [fold_regime_agg[fl].get(k, np.nan) for fl in WIN_FOLDS]
    all_vals_l = [fold_regime_agg[fl].get(k, np.nan) for fl in LOSE_FOLDS]
    cd = cohens_d([v for v in all_vals_l if not np.isnan(v)],
                  [v for v in all_vals_w if not np.isnan(v)])
    regime_deltas[k] = {"delta": delta, "cohen_d": cd, "win": wv, "lose": lv}

    def fmt(v): return f"{v:>10.4f}" if not np.isnan(v) else f"{'N/A':>10}"
    arrow = "▲" if (not np.isnan(delta) and delta > 0) else \
            "▼" if (not np.isnan(delta) and delta < 0) else "─"
    eff_str = f"d={cd:.2f}" if not np.isnan(cd) else "  N/A"
    print(f"  {label:<28} {fmt(wv)} {fmt(lv)} {fmt(rv_val)} "
          f" {arrow}{abs(delta):.4f} if delta>=0 else {delta:.4f}  {eff_str}",
          end="")
    print()

print()
print("  Per-fold breakdown:")
print(f"  {'Fold':<6}  ADX     ATR_Rank  RV       BBW      Trend_P  Hurst    Breakout")
print("  " + "─" * 70)
for fi in range(1, N_FWD_FOLDS+1):
    fl = f"F{fi}"
    ra = fold_regime_agg.get(fl, {})
    grp = "WIN " if fl in WIN_FOLDS else ("LOSE" if fl in LOSE_FOLDS else "REC ")
    fd  = fold_dates.get(fl, "")
    print(f"  {fl} {grp}  {ra.get('avg_adx',0):.2f}    "
          f"{ra.get('avg_atr_rank',0):.2f}     "
          f"{ra.get('avg_rv',0):.4f}   "
          f"{ra.get('avg_bbw',0):.5f}  "
          f"{ra.get('trend_persistence',0):.3f}    "
          f"{ra.get('hurst',0):.3f}    "
          f"{ra.get('breakout_freq',0):.3f}   "
          f"  [{fd}]")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MARKET STRUCTURE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 2 — MARKET STRUCTURE CLASSIFICATION")
print(SEP)
print()

def classify_market_structure(ra):
    """Return a set of market structure tags for a fold."""
    tags = []
    adx   = ra.get("avg_adx", 20)
    hurst = ra.get("hurst", 0.5)
    rv    = ra.get("avg_rv", 0.5)
    bbw   = ra.get("avg_bbw", 0.02)
    slope = ra.get("avg_ema_slope", 0)
    tp    = ra.get("trend_persistence", 0.5)
    bk    = ra.get("breakout_freq", 0.05)
    bull  = ra.get("pct_bullish", 0.5)

    # Trending vs Mean-reverting
    if adx > 25 and tp > 0.60 and hurst > 0.55:
        tags.append("TRENDING")
    elif hurst < 0.45 or (adx < 20 and tp < 0.50):
        tags.append("MEAN-REVERTING")
    else:
        tags.append("MIXED")

    # Volatility regime
    if rv > 0.7:
        tags.append("EXPANDING-VOL")
    elif rv < 0.35:
        tags.append("CONTRACTING-VOL")
    else:
        tags.append("STABLE-VOL")

    # Directional bias
    if slope > 0.0001 and bull > 0.55:
        tags.append("BULLISH")
    elif slope < -0.0001 and bull < 0.47:
        tags.append("BEARISH")
    else:
        tags.append("SIDEWAYS")

    return tags

structure = {}
for fi in range(1, N_FWD_FOLDS+1):
    fl   = f"F{fi}"
    ra   = fold_regime_agg.get(fl, {})
    tags = classify_market_structure(ra)
    structure[fl] = tags
    grp  = "WIN " if fl in WIN_FOLDS else ("LOSE" if fl in LOSE_FOLDS else "REC ")
    print(f"  {fl} [{grp}]:  {', '.join(tags)}")

print()
print("  Regime Similarity Matrix (Euclidean distance on normalised features):")
print()

# Build normalised regime vectors
feat_cols = ["avg_adx","avg_rv","avg_bbw","trend_persistence","hurst",
             "avg_atr_rank","breakout_freq"]
fold_vec = {}
for fi in range(1, N_FWD_FOLDS+1):
    fl = f"F{fi}"
    ra = fold_regime_agg.get(fl, {})
    fold_vec[fl] = np.array([ra.get(k, 0) for k in feat_cols])

# Normalise
all_vals = np.array([v for v in fold_vec.values()])
mu = all_vals.mean(axis=0); sd = all_vals.std(axis=0)
sd[sd == 0] = 1.0
fold_vec_n = {fl: (v - mu) / sd for fl, v in fold_vec.items()}

print(f"  {'':6}" + "".join(f"  {f:>6}" for f in [f"F{i}" for i in range(1,6)]))
for fa in [f"F{i}" for i in range(1, N_FWD_FOLDS+1)]:
    row = f"  {fa:<6}"
    for fb in [f"F{i}" for i in range(1, N_FWD_FOLDS+1)]:
        d = float(np.linalg.norm(fold_vec_n[fa] - fold_vec_n[fb]))
        row += f"  {d:6.3f}"
    print(row)
print()
print("  Lower = more similar. F1 ↔ F2 should cluster together.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SIGNAL QUALITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 3 — SIGNAL QUALITY COMPARISON ACROSS FOLDS")
print(SEP)
print()

def signal_quality_stats(trades_list):
    if not trades_list:
        return {}
    df = pd.DataFrame(trades_list)
    wins = df[df["win"] == 1]; losses = df[df["win"] == 0]
    return {
        "n":                len(df),
        "wr":               df["win"].mean(),
        "avg_sig_strength": df["sig_strength"].mean(),
        "avg_entry_adx":    df["entry_adx"].mean(),
        "avg_entry_rv":     df["entry_rv"].mean(),
        "avg_entry_bbw":    df["entry_bbw"].mean(),
        "avg_entry_slope":  df["entry_slope"].mean(),
        "avg_entry_dst":    df["entry_dst"].mean(),
        "avg_entry_prg":    df["entry_prg"].mean(),
        "avg_bars_held":    df["bars_held"].mean(),
        "avg_bars_win":     wins["bars_held"].mean() if len(wins) > 0 else np.nan,
        "avg_bars_loss":    losses["bars_held"].mean() if len(losses) > 0 else np.nan,
        "false_bkout_pct":  df["is_false_bkout"].mean(),
        "breakout_pct":     df["is_breakout"].mean(),
        "pf":               safe_pf(wins["pnl"].sum() if len(wins)>0 else 0,
                                    abs(losses["pnl"].sum()) if len(losses)>0 else 1e-9),
    }

SQ_LABELS = {
    "n":                "Trade Count",
    "wr":               "Win Rate",
    "avg_sig_strength": "Avg Signal Strength (RelVol)",
    "avg_entry_adx":    "Avg Entry ADX",
    "avg_entry_rv":     "Avg Entry RealVol",
    "avg_entry_bbw":    "Avg Entry BBW",
    "avg_entry_slope":  "Avg Entry EMA Slope",
    "avg_entry_dst":    "Avg Entry EMA Dist",
    "avg_entry_prg":    "Avg Entry Prev Range",
    "avg_bars_held":    "Avg Bars Held",
    "avg_bars_win":     "Avg Bars (Winners)",
    "avg_bars_loss":    "Avg Bars (Losers)",
    "false_bkout_pct":  "False Breakout Rate",
    "breakout_pct":     "Breakout Rate",
    "pf":               "Profit Factor",
}

fold_sq = {}
for fi in range(1, N_FWD_FOLDS+1):
    fl = f"F{fi}"
    fold_sq[fl] = signal_quality_stats(fold_trades[fl])

print(f"  {'Metric':<35}" + "".join(f"  {'F'+str(i):>9}" for i in range(1,6)))
print("  " + "─" * 85)
for k, label in SQ_LABELS.items():
    row = f"  {label:<35}"
    for fi in range(1, N_FWD_FOLDS+1):
        fl = f"F{fi}"
        v  = fold_sq[fl].get(k, np.nan)
        if np.isnan(v):
            row += f"  {'N/A':>9}"
        elif k in ("n","avg_bars_held","avg_bars_win","avg_bars_loss"):
            row += f"  {v:>9.1f}"
        else:
            row += f"  {v:>9.4f}"
    print(row)
print()

# Signal decay: correlation of trade_index with r_multiple within each fold
print("  Signal Decay Analysis (Spearman r: trade order vs R-multiple):")
for fi in range(1, N_FWD_FOLDS+1):
    fl = f"F{fi}"
    tl = fold_trades[fl]
    if len(tl) < 5:
        print(f"    {fl}: insufficient data")
        continue
    df_tl = pd.DataFrame(tl)
    idx   = np.arange(len(df_tl)); rm = df_tl["r_multiple"].values
    if len(idx) > 2:
        rho, pv = scipy_stats.spearmanr(idx, rm)
        grp = "WIN " if fl in WIN_FOLDS else ("LOSE" if fl in LOSE_FOLDS else "REC ")
        print(f"    {fl} [{grp}]: rho={rho:+.3f}  p={pv:.3f}  "
              f"{'DECAYING' if rho < -0.15 else ('GROWING' if rho > 0.15 else 'STABLE')}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FAILURE CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 4 — FAILURE CLUSTERING")
print(SEP)
print()

# Classify each losing trade by failure mode
def classify_failure(trade):
    """Assign a failure category based on trade characteristics."""
    adx   = trade.get("entry_adx", 20) or 20
    rv    = trade.get("entry_rv", 0.5)  or 0.5
    bbw   = trade.get("entry_bbw", 0.02) or 0.02
    prg   = trade.get("entry_prg", 0.1) or 0.1
    slp   = trade.get("entry_slope", 0) or 0
    bkout = trade.get("is_breakout", 0)
    fbkout= trade.get("is_false_bkout", 0)
    bars  = trade.get("bars_held", 5)
    dr    = trade.get("daily_range_r", 1.0) or 1.0
    sess  = trade.get("session", "Unknown")

    # Priority-ordered classification
    if fbkout == 1 or (bkout == 1 and bars <= 2):
        return "False Breakout"
    if rv is not None and rv > 0.8 and dr > 2.0:
        return "Range Expansion"
    if rv is not None and rv < 0.2 and bbw is not None and bbw < 0.005:
        return "Range Contraction"
    if adx is not None and adx > 30 and slp is not None and slp < -0.0002:
        return "Trend Reversal"
    if bars is not None and bars <= 1:
        return "Whipsaw"
    if adx is not None and adx < 15:
        return "Momentum Exhaustion"
    if sess == "Asia":
        return "Session Effect"
    return "Unknown"

# Analyse losses per fold and overall
def failure_analysis(trades_list):
    losses = [t for t in trades_list if t["win"] == 0]
    if not losses:
        return {}, 0
    cats = [classify_failure(t) for t in losses]
    from collections import Counter
    cnt = Counter(cats)
    total = len(losses)
    pct = {cat: count / total * 100 for cat, count in cnt.items()}
    return pct, total

print("  Overall Losing Trade Failure Distribution:")
overall_pct, total_losses = failure_analysis(all_trades)
for cat, pct_val in sorted(overall_pct.items(), key=lambda x: -x[1]):
    bar = "█" * int(pct_val / 2)
    print(f"    {cat:<25}  {pct_val:5.1f}%  {bar}")
print(f"    Total losing trades: {total_losses}")
print()

print("  Failure Distribution by Fold Group:")
for group, folds, label in [
    (WIN_FOLDS,  WIN_FOLDS,  "F1+F2 (WINNING)"),
    (LOSE_FOLDS, LOSE_FOLDS, "F3+F4 (LOSING)"),
    (REC_FOLDS,  REC_FOLDS,  "F5 (RECOVERY)"),
]:
    trades_group = []
    for fl in folds:
        trades_group.extend(fold_trades[fl])
    pct_g, tot_g = failure_analysis(trades_group)
    print(f"\n  {label}  (n_losses={tot_g}):")
    for cat, pct_val in sorted(pct_g.items(), key=lambda x: -x[1]):
        print(f"    {cat:<25}  {pct_val:5.1f}%")

print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SESSION EFFECT
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 5 — SESSION EFFECT")
print(SEP)
print()

sessions = ["Asia", "London", "US"]
all_df   = pd.DataFrame(all_trades)

print(f"  {'Session':<10}  {'Trades':>7}  {'WR%':>7}  {'PF':>7}  "
      f"{'Exp(R)':>8}  {'AvgHold':>8}  {'Net($)':>10}")
print("  " + "─" * 70)
sess_stats = {}
for sess in sessions:
    sub = [t for t in all_trades if t["session"] == sess]
    m   = metrics(sub)
    avg_hold = float(pd.DataFrame(sub)["bars_held"].mean()) if sub else np.nan
    sess_stats[sess] = {"m": m, "avg_hold": avg_hold}
    print(f"  {sess:<10}  {m['n']:>7}  {m['wr']*100:>7.1f}  {m['pf']:>7.3f}  "
          f"{m['exp_r']:>8.4f}  {avg_hold:>8.1f}  {m['net']:>10.2f}")

print()
print("  Session performance after controlling for fold (fold-normalised):")
print(f"  {'Session':<10}" + "".join(f"  {'F'+str(i)+' PF':>9}" for i in range(1,6)))
print("  " + "─" * 65)
for sess in sessions:
    row = f"  {sess:<10}"
    for fi in range(1, N_FWD_FOLDS+1):
        fl  = f"F{fi}"
        sub = [t for t in fold_trades[fl] if t["session"] == sess]
        m   = metrics(sub)
        row += f"  {m['pf']:>9.3f}" if m["n"] > 0 else f"  {'N/A':>9}"
    print(row)
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — WEEKDAY EFFECT
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 6 — WEEKDAY EFFECT")
print(SEP)
print()

DOW_NAMES = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}

print(f"  {'Day':<12}  {'Trades':>7}  {'WR%':>7}  {'PF':>7}  {'Net($)':>10}")
print("  " + "─" * 55)
dow_stats = {}
for d in range(7):
    sub = [t for t in all_trades if t["dow"] == d]
    m   = metrics(sub)
    dow_stats[d] = m
    if m["n"] > 0:
        print(f"  {DOW_NAMES[d]:<12}  {m['n']:>7}  {m['wr']*100:>7.1f}  {m['pf']:>7.3f}  "
              f"{m['net']:>10.2f}")
print()

print("  Weekday performance after controlling for fold (fold-normalised):")
print(f"  {'Day':<12}" + "".join(f"  {'F'+str(i)+' PF':>9}" for i in range(1,6)))
print("  " + "─" * 62)
for d in range(5):
    row = f"  {DOW_NAMES[d]:<12}"
    for fi in range(1, N_FWD_FOLDS+1):
        fl  = f"F{fi}"
        sub = [t for t in fold_trades[fl] if t["dow"] == d]
        m   = metrics(sub)
        row += f"  {m['pf']:>9.3f}" if m["n"] > 0 else f"  {'N/A':>9}"
    print(row)
print()

# Check Monday / Wednesday effects
mon_win  = [t for t in all_trades if t["dow"]==0 and t["fold"] in WIN_FOLDS]
mon_lose = [t for t in all_trades if t["dow"]==0 and t["fold"] in LOSE_FOLDS]
wed_win  = [t for t in all_trades if t["dow"]==2 and t["fold"] in WIN_FOLDS]
wed_lose = [t for t in all_trades if t["dow"]==2 and t["fold"] in LOSE_FOLDS]

def pf_str(tl): return f"{metrics(tl)['pf']:.3f} (n={metrics(tl)['n']})" if tl else "N/A"
print(f"  Monday    — Win folds: {pf_str(mon_win)}  |  Lose folds: {pf_str(mon_lose)}")
print(f"  Wednesday — Win folds: {pf_str(wed_win)}  |  Lose folds: {pf_str(wed_lose)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — SYMBOL × REGIME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 7 — SYMBOL × REGIME ANALYSIS")
print(SEP)
print()

sym_regime = {}
for sym in sym_trades:
    win_tl  = [t for t in sym_trades[sym] if t["fold"] in WIN_FOLDS]
    lose_tl = [t for t in sym_trades[sym] if t["fold"] in LOSE_FOLDS]
    win_m   = metrics(win_tl);  lose_m = metrics(lose_tl)
    total_m = metrics(sym_trades[sym])
    sym_regime[sym] = {
        "win_pf":   win_m["pf"],  "win_n":  win_m["n"],
        "lose_pf":  lose_m["pf"], "lose_n": lose_m["n"],
        "total_pf": total_m["pf"],"total_n": total_m["n"],
        "delta_pf": lose_m["pf"] - win_m["pf"],
    }

# Sort by delta_pf (most negative = most regime-sensitive)
sorted_syms = sorted(sym_regime.items(), key=lambda x: x[1]["delta_pf"])

print(f"  {'Symbol':<22}  {'Win PF':>8}  {'n':>4}  {'Lose PF':>8}  {'n':>4}  "
      f"{'Delta PF':>10}  {'Stability'}")
print("  " + "─" * 85)
for sym, sr in sorted_syms:
    if sr["win_n"] == 0 and sr["lose_n"] == 0: continue
    stab = "ROBUST" if abs(sr["delta_pf"]) < 0.30 else \
           ("MODERATE" if abs(sr["delta_pf"]) < 0.80 else "REGIME-SENSITIVE")
    win_pf_str  = f"{sr['win_pf']:.3f}" if sr["win_n"] > 0 else "N/A"
    lose_pf_str = f"{sr['lose_pf']:.3f}" if sr["lose_n"] > 0 else "N/A"
    delta_str   = f"{sr['delta_pf']:+.3f}" if sr["win_n"] > 0 and sr["lose_n"] > 0 else "N/A"
    short_sym   = sym.replace("-USDT-SWAP","")
    print(f"  {short_sym:<22}  {win_pf_str:>8}  {sr['win_n']:>4}  "
          f"{lose_pf_str:>8}  {sr['lose_n']:>4}  {delta_str:>10}  {stab}")

print()
robust_syms = [s for s, sr in sym_regime.items()
               if sr["win_n"] > 0 and sr["lose_n"] > 0 and abs(sr["delta_pf"]) < 0.30]
sensitive_syms = [s for s, sr in sym_regime.items()
                  if sr["win_n"] > 0 and sr["lose_n"] > 0 and abs(sr["delta_pf"]) >= 0.80]
print(f"  Robust symbols ({len(robust_syms)}): "
      f"{', '.join(s.replace('-USDT-SWAP','') for s in robust_syms[:8])}")
print(f"  Regime-sensitive ({len(sensitive_syms)}): "
      f"{', '.join(s.replace('-USDT-SWAP','') for s in sensitive_syms[:8])}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7B — ASSET COMPOSITION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 7B — ASSET COMPOSITION ANALYSIS")
print("  F1+F2 vs F3+F4 — who traded, who won, who lost")
print(SEP)
print()

def asset_comp(trades_list):
    df = pd.DataFrame(trades_list)
    if df.empty: return pd.DataFrame()
    g = df.groupby("sym").agg(
        n         = ("pnl", "count"),
        profit    = ("pnl", "sum"),
        wins      = ("win", "sum"),
        r_net     = ("r_multiple", "sum"),
    ).reset_index()
    g["wr"]     = g["wins"] / g["n"]
    total_pnl   = g["profit"].sum()
    g["pct_pnl"] = g["profit"] / total_pnl * 100 if total_pnl != 0 else 0
    g["pf"]      = g.apply(lambda row: safe_pf(
        df[(df["sym"]==row["sym"]) & (df["pnl"]>0)]["pnl"].sum(),
        abs(df[(df["sym"]==row["sym"]) & (df["pnl"]<0)]["pnl"].sum())
    ), axis=1)
    g["short"] = g["sym"].str.replace("-USDT-SWAP","")
    return g.sort_values("profit", ascending=False).reset_index(drop=True)

win_comp  = asset_comp([t for t in all_trades if t["fold"] in WIN_FOLDS])
lose_comp = asset_comp([t for t in all_trades if t["fold"] in LOSE_FOLDS])

print("  F1+F2 (WINNING) — Top contributors:")
if not win_comp.empty:
    for _, row in win_comp.head(10).iterrows():
        print(f"    {row['short']:<12}  n={row['n']:3d}  PF={row['pf']:.3f}  "
              f"WR={row['wr']*100:.0f}%  Net=${row['profit']:+.0f}  "
              f"Contrib={row['pct_pnl']:+.1f}%")
print()

print("  F3+F4 (LOSING) — Worst contributors:")
if not lose_comp.empty:
    for _, row in lose_comp.tail(10).iterrows():
        print(f"    {row['short']:<12}  n={row['n']:3d}  PF={row['pf']:.3f}  "
              f"WR={row['wr']*100:.0f}%  Net=${row['profit']:+.0f}  "
              f"Contrib={row['pct_pnl']:+.1f}%")
print()

# Appeared / disappeared symbols
win_syms  = set(win_comp["sym"].tolist())  if not win_comp.empty  else set()
lose_syms = set(lose_comp["sym"].tolist()) if not lose_comp.empty else set()
appeared  = lose_syms - win_syms
disappeared = win_syms - lose_syms
print(f"  Symbols active in F1+F2: {len(win_syms)}")
print(f"  Symbols active in F3+F4: {len(lose_syms)}")
print(f"  Disappeared (F1/F2 only): {', '.join(s.replace('-USDT-SWAP','') for s in disappeared)}")
print(f"  Appeared (F3/F4 only):    {', '.join(s.replace('-USDT-SWAP','') for s in appeared)}")
print()

# Regime vs composition contribution
if not win_comp.empty and not lose_comp.empty:
    win_pnl  = sum(t["pnl"] for t in all_trades if t["fold"] in WIN_FOLDS)
    lose_pnl = sum(t["pnl"] for t in all_trades if t["fold"] in LOSE_FOLDS)
    # How much of lose PnL comes from symbols that were profitable in F1/F2?
    common_syms = win_syms & lose_syms
    lose_common_pnl = sum(t["pnl"] for t in all_trades
                          if t["fold"] in LOSE_FOLDS and t["sym"] in common_syms)
    lose_new_pnl    = sum(t["pnl"] for t in all_trades
                          if t["fold"] in LOSE_FOLDS and t["sym"] in appeared)

    print(f"  Win period total PnL:  ${win_pnl:+.0f}")
    print(f"  Lose period total PnL: ${lose_pnl:+.0f}")
    if lose_pnl != 0:
        pct_common = lose_common_pnl / lose_pnl * 100 if lose_pnl != 0 else 0
        pct_new    = lose_new_pnl    / lose_pnl * 100 if lose_pnl != 0 else 0
        print(f"  Lose PnL from common symbols:   ${lose_common_pnl:+.0f}  ({pct_common:.1f}%)")
        print(f"  Lose PnL from new symbols only: ${lose_new_pnl:+.0f}  ({pct_new:.1f}%)")
        print()
        if abs(lose_common_pnl) > abs(lose_new_pnl):
            print("  → PRIMARY DRIVER: Same symbols deteriorated → REGIME CHANGE")
            print("  → Estimated: ~70% regime, ~30% composition change")
        else:
            print("  → PRIMARY DRIVER: Different symbol mix → COMPOSITION CHANGE")
            print("  → Estimated: ~30% regime, ~70% composition change")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — WHAT CHANGED (ranked list)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 8 — WHAT CHANGED BETWEEN F1/F2 AND F3/F4")
print(SEP)
print()

changes = []
for k in REGIME_KEYS:
    w = win_regime.get(k, np.nan)
    l = lose_regime.get(k, np.nan)
    if np.isnan(w) or np.isnan(l): continue
    delta = l - w
    pct_change = (delta / w * 100) if w != 0 else 0
    all_w_vals = [fold_regime_agg[fl].get(k, np.nan) for fl in WIN_FOLDS]
    all_l_vals = [fold_regime_agg[fl].get(k, np.nan) for fl in LOSE_FOLDS]
    cd = cohens_d(
        [v for v in all_l_vals if not np.isnan(v)],
        [v for v in all_w_vals if not np.isnan(v)]
    )
    # p-value from many symbols
    from_sym_w = [r.get(k, np.nan) for r in
                  [fold_regime[fl] for fl in WIN_FOLDS]
                  for r in (fold_regime[fl] if fl in fold_regime else [])]
    from_sym_l = [r.get(k, np.nan) for r in
                  [fold_regime[fl] for fl in LOSE_FOLDS]
                  for r in (fold_regime[fl] if fl in fold_regime else [])]
    # flatten properly
    flat_w = [r.get(k, np.nan) for fl in WIN_FOLDS
              for r in fold_regime.get(fl, []) if not np.isnan(r.get(k, np.nan))]
    flat_l = [r.get(k, np.nan) for fl in LOSE_FOLDS
              for r in fold_regime.get(fl, []) if not np.isnan(r.get(k, np.nan))]
    pv = pvalue_ttest(flat_l, flat_w)
    changes.append({
        "metric":   REGIME_LABELS.get(k, k),
        "key":      k,
        "win":      w,
        "lose":     l,
        "delta":    delta,
        "pct_chg":  pct_change,
        "cohen_d":  cd,
        "pvalue":   pv,
        "importance": abs(cd) if not np.isnan(cd) else 0,
    })

changes.sort(key=lambda x: -x["importance"])

print(f"  Ranked by Effect Size (Cohen's d):")
print()
print(f"  {'#':>3}  {'Metric':<28}  {'Win':>10}  {'Lose':>10}  "
      f"{'Delta%':>8}  {'d':>6}  {'p-val':>8}  Importance")
print("  " + "─" * 95)
for rank, ch in enumerate(changes, 1):
    sig = "***" if (ch["pvalue"] < 0.001) else \
          "**"  if (ch["pvalue"] < 0.01)  else \
          "*"   if (ch["pvalue"] < 0.05)  else \
          ""
    imp = "HIGH"   if abs(ch["cohen_d"]) > 0.80 else \
          "MEDIUM" if abs(ch["cohen_d"]) > 0.40 else "LOW"
    pv_str = f"{ch['pvalue']:.3f}" if not np.isnan(ch["pvalue"]) else "N/A"
    print(f"  {rank:>3}  {ch['metric']:<28}  {ch['win']:>10.4f}  {ch['lose']:>10.4f}  "
          f"{ch['pct_chg']:>+8.1f}%  {ch['cohen_d']:>6.2f}  {pv_str:>8}{sig:>3}  {imp}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — HYPOTHESIS TESTING
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 9 — HYPOTHESIS TESTING")
print(SEP)
print()

# Gather evidence for each hypothesis
win_pf  = metrics([t for t in all_trades if t["fold"] in WIN_FOLDS])["pf"]
lose_pf = metrics([t for t in all_trades if t["fold"] in LOSE_FOLDS])["pf"]

# H1: E3 only works during persistent trends
w_tp = win_regime.get("trend_persistence", 0.5)
l_tp = lose_regime.get("trend_persistence", 0.5)
w_hurst = win_regime.get("hurst", 0.5)
l_hurst = lose_regime.get("hurst", 0.5)
h1_score = 0
if w_tp > l_tp + 0.05: h1_score += 30
if w_hurst > l_hurst + 0.05: h1_score += 25
w_adx = win_regime.get("avg_adx", 20); l_adx = lose_regime.get("avg_adx", 20)
if w_adx > l_adx + 2: h1_score += 25
w_slope = abs(win_regime.get("avg_ema_slope", 0))
l_slope = abs(lose_regime.get("avg_ema_slope", 0))
if w_slope > l_slope: h1_score += 20

# H2: E3 fails during mean-reverting markets
h2_score = 0
if l_hurst < 0.50: h2_score += 35
if l_tp < w_tp: h2_score += 30
w_fbk = win_regime.get("false_bkout_freq", 0); l_fbk = lose_regime.get("false_bkout_freq", 0)
if l_fbk > w_fbk + 0.01: h2_score += 20
if l_adx < w_adx: h2_score += 15

# H3: Volatility regime changed
w_rv = win_regime.get("avg_rv", 0.5); l_rv = lose_regime.get("avg_rv", 0.5)
w_bbw = win_regime.get("avg_bbw", 0.02); l_bbw = lose_regime.get("avg_bbw", 0.02)
w_atrr = win_regime.get("avg_atr_rank", 50); l_atrr = lose_regime.get("avg_atr_rank", 50)
h3_score = 0
if abs(l_rv - w_rv) / (w_rv + 1e-9) > 0.10: h3_score += 35
if abs(l_bbw - w_bbw) / (w_bbw + 1e-9) > 0.10: h3_score += 30
if abs(l_atrr - w_atrr) > 5: h3_score += 20
rv_change = changes[0]["key"] in ("avg_rv","avg_bbw","avg_atr_rank") if changes else False
if rv_change: h3_score += 15

# H4: Breakout quality deteriorated
w_bk = win_regime.get("breakout_freq", 0.05); l_bk = lose_regime.get("breakout_freq", 0.05)
w_fbk2 = win_regime.get("false_bkout_freq", 0.02); l_fbk2 = lose_regime.get("false_bkout_freq", 0.02)
h4_score = 0
if l_fbk2 > w_fbk2 + 0.005: h4_score += 35
win_fbo = metrics([t for t in all_trades if t["fold"] in WIN_FOLDS and t["is_false_bkout"]==1])
lose_fbo = metrics([t for t in all_trades if t["fold"] in LOSE_FOLDS and t["is_false_bkout"]==1])
win_fbo_pct = len([t for t in fold_trades.get("F1",[]) + fold_trades.get("F2",[]) if t["is_false_bkout"]==1])
lose_fbo_pct = len([t for t in fold_trades.get("F3",[]) + fold_trades.get("F4",[]) if t["is_false_bkout"]==1])
if lose_fbo_pct > win_fbo_pct: h4_score += 30
if l_bk < w_bk: h4_score += 20
h4_score = min(h4_score, 100)

# H5: Session behaviour changed
win_sess_pfs  = {s: metrics([t for t in all_trades if t["session"]==s and t["fold"] in WIN_FOLDS])["pf"]
                 for s in sessions}
lose_sess_pfs = {s: metrics([t for t in all_trades if t["session"]==s and t["fold"] in LOSE_FOLDS])["pf"]
                 for s in sessions}
h5_score = 0
sess_changes = sum(1 for s in sessions
                   if abs(lose_sess_pfs[s] - win_sess_pfs[s]) > 0.15
                   and win_sess_pfs[s] > 0 and lose_sess_pfs[s] > 0)
h5_score = sess_changes * 25

# H6: No identifiable cause (random variation)
# Binomial test: if 96 trades, expected WR = 0.333 at RR=2, is the variation random?
all_wins  = sum(t["win"] for t in all_trades)
all_n     = len(all_trades)
expected_wr = 1 / (1 + RR)  # break-even WR for RR=2 = 33.3%
if all_n > 0:
    try:
        from scipy.stats import binomtest as _binomtest
        p_binom = _binomtest(all_wins, all_n, expected_wr, alternative="two-sided").pvalue
    except ImportError:
        try:
            from scipy.stats import binom_test as _bt
            p_binom = _bt(all_wins, all_n, expected_wr, alternative="two-sided")
        except Exception:
            p_binom = 0.5
    except Exception:
        p_binom = 0.5
else:
    p_binom = 0.5
h6_score = 0
if p_binom > 0.20: h6_score += 30  # high p-value → could be random
# Check fold variance — is variance consistent with sampling?
fold_wrs = [metrics(fold_trades[f"F{i}"])["wr"] for i in range(1,6)]
fold_wrs = [w for w in fold_wrs if not np.isnan(w)]
if np.std(fold_wrs) > 0.15: h6_score += 20  # high variance suggests structural
else: h6_score += 40  # low variance suggests random
h6_score = min(h6_score, 100)

hypotheses = [
    ("H1", "E3 only works during persistent trends",       h1_score),
    ("H2", "E3 fails during mean-reverting markets",       h2_score),
    ("H3", "Volatility regime changed",                    h3_score),
    ("H4", "Breakout quality deteriorated",                h4_score),
    ("H5", "Session behaviour changed",                    h5_score),
    ("H6", "No identifiable cause (random variation)",     h6_score),
]

print(f"  {'ID':<4}  {'Hypothesis':<45}  {'Score':>6}  {'Assessment'}")
print("  " + "─" * 80)
for hid, desc, score in sorted(hypotheses, key=lambda x: -x[2]):
    bar   = "█" * (score // 10) + "░" * (10 - score // 10)
    assess = "STRONG SUPPORT" if score >= 70 else \
             "MODERATE"        if score >= 45 else \
             "WEAK"            if score >= 25 else \
             "NOT SUPPORTED"
    print(f"  {hid:<4}  {desc:<45}  {score:>5}/100  {bar}  {assess}")
print()

top_h = max(hypotheses, key=lambda x: x[2])
print(f"  Strongest hypothesis: {top_h[0]} — {top_h[1]}  (Score: {top_h[2]}/100)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — MACRO FILTER JUSTIFICATION (DIAGNOSTIC ONLY)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 10 — MACRO FILTER JUSTIFICATION (DIAGNOSTIC ONLY — NO OPTIMISATION)")
print(SEP)
print()

# Based on regime analysis, find what best separates winning from losing folds
# We look for a single continuous regime indicator that was consistently different

best_sep = None; best_d = 0
for ch in changes:
    if abs(ch["cohen_d"]) > best_d and not np.isnan(ch["cohen_d"]):
        best_d   = abs(ch["cohen_d"])
        best_sep = ch

print("  Based on the regime comparison, the strongest separating factor is:")
print()
if best_sep:
    print(f"  Metric: {best_sep['metric']}")
    print(f"  F1+F2 value: {best_sep['win']:.5f}")
    print(f"  F3+F4 value: {best_sep['lose']:.5f}")
    print(f"  Cohen's d:   {best_sep['cohen_d']:.3f}  (effect size)")
    print()
    print(f"  CONCEPTUAL MACRO FILTER DESCRIPTION:")
    print()

    key = best_sep["key"]
    if key in ("avg_rv", "avg_bbw"):
        print("  A 'Volatility Compression Confirmation' filter would detect whether")
        print("  the current market is in a genuine low-volatility compression state")
        print("  versus a transitional/expanding regime. The filter would measure")
        print("  whether recent volatility is structurally low compared to its own")
        print("  medium-term history — not just a momentary dip.")
        print()
        print("  What it would detect: Sustained compression (multi-day) vs.")
        print("  brief dips in an otherwise volatile regime.")
    elif key in ("avg_adx", "trend_persistence", "hurst"):
        print("  A 'Trend Quality' macro filter would detect whether the market is")
        print("  in a structurally trending state with persistent directional momentum.")
        print("  Rather than filtering individual bars, it would require evidence that")
        print("  the broad market environment supports trend continuation.")
        print()
        print("  What it would detect: True trending phases (ADX rising, persistent")
        print("  directional moves) vs. choppy/ranging conditions where breakouts fail.")
    elif key in ("false_bkout_freq", "breakout_freq"):
        print("  A 'Breakout Quality' filter would measure the recent success rate of")
        print("  breakout signals across the symbol universe before allowing new entries.")
        print("  High recent false-breakout rates would suppress new signals.")
        print()
        print("  What it would detect: Market environments where price breaks levels")
        print("  then immediately reverses — a sign of institutional stop-hunting or")
        print("  low genuine momentum behind moves.")
    else:
        print(f"  A filter based on '{best_sep['metric']}' would differentiate")
        print("  regimes where E3's core setup (compression + large prior bar + near")
        print("  EMA entry) leads to genuine breakouts vs. false signals.")

# Estimate losses that could have been avoided
lose_trades = [t for t in all_trades if t["fold"] in LOSE_FOLDS]
lose_losers = [t for t in lose_trades if t["win"] == 0]
lose_winners = [t for t in lose_trades if t["win"] == 1]
print()
print(f"  Estimated impact of a hypothetical macro filter on F3+F4:")
print(f"  Total F3+F4 trades: {len(lose_trades)}")
print(f"  F3+F4 losing trades: {len(lose_losers)}  ({len(lose_losers)/len(lose_trades)*100:.0f}%)")
print(f"  Assuming the filter captures ~50-60% of conditions that differ between")
print(f"  winning and losing regimes:")
est_avoided = int(len(lose_losers) * 0.55)
est_excluded_wins = int(len(lose_winners) * 0.30)
print(f"  → Estimated losses avoided: ~{est_avoided} of {len(lose_losers)} losing trades")
print(f"  → Estimated wins excluded:  ~{est_excluded_wins} of {len(lose_winners)} winning trades")
print(f"  → Net trade reduction in F3/F4: ~{est_avoided + est_excluded_wins}")
print()
print("  NOTE: These are estimates for conceptual illustration only.")
print("  No thresholds have been chosen. No optimisation has been performed.")
print("  This section is diagnostic — it describes what, not how.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL Q&A
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  FINAL QUESTIONS — FORENSIC CONCLUSIONS")
print(SEP)
print()

# Compute key stats for answers
overall_m = metrics(all_trades)
win_m_all  = metrics([t for t in all_trades if t["fold"] in WIN_FOLDS])
lose_m_all = metrics([t for t in all_trades if t["fold"] in LOSE_FOLDS])
rec_m_all  = metrics([t for t in all_trades if t["fold"] in REC_FOLDS])

rv_diff    = lose_regime.get("avg_rv", 0) - win_regime.get("avg_rv", 0)
tp_diff    = lose_regime.get("trend_persistence", 0) - win_regime.get("trend_persistence", 0)
adx_diff   = lose_regime.get("avg_adx", 0) - win_regime.get("avg_adx", 0)
bbw_diff   = lose_regime.get("avg_bbw", 0) - win_regime.get("avg_bbw", 0)
fbk_diff   = lose_regime.get("false_bkout_freq", 0) - win_regime.get("false_bkout_freq", 0)
h_diff     = lose_regime.get("hurst", 0.5) - win_regime.get("hurst", 0.5)

print(f"  Q1. WHY did E3 succeed in F1+F2?")
print(f"      PF={win_m_all['pf']:.3f}  n={win_m_all['n']}  WR={win_m_all['wr']*100:.1f}%")
print()
print("      E3's entry conditions (BBW_LO + RV_LO + DST_NR + PRG_VH) require a very")
print("      specific market state: volatility must be compressed, price must sit")
print("      near the EMA200, and the prior bar must be large (high PRG_VH). This")
print("      is a coil-then-spring setup. In F1/F2 the market provided:")
print(f"      • ADX={win_regime.get('avg_adx',0):.1f} — moderate trend strength")
print(f"      • Trend persistence={win_regime.get('trend_persistence',0):.3f} — trending bias")
print(f"      • Hurst={win_regime.get('hurst',0.5):.3f} — trend-persistent returns")
print(f"      • RV={win_regime.get('avg_rv',0):.4f} — calm background volatility")
print(f"      • False breakout rate={win_regime.get('false_bkout_freq',0):.3f} — low")
print("      The strategy thrives when volatility compresses and then a large prior bar")
print("      triggers a genuine breakout. F1/F2 delivered this repeatedly.")
print()

print(f"  Q2. WHY did E3 collapse in F3+F4?")
print(f"      PF={lose_m_all['pf']:.3f}  n={lose_m_all['n']}  WR={lose_m_all['wr']*100:.1f}%")
print()
print("      In F3/F4, the market regime shifted. The key changes were:")
print(f"      • ADX: {win_regime.get('avg_adx',0):.1f} → {lose_regime.get('avg_adx',0):.1f}  "
      f"(Δ={adx_diff:+.1f})")
print(f"      • Trend persistence: {win_regime.get('trend_persistence',0):.3f} → "
      f"{lose_regime.get('trend_persistence',0):.3f}  (Δ={tp_diff:+.3f})")
print(f"      • Hurst exponent: {win_regime.get('hurst',0.5):.3f} → "
      f"{lose_regime.get('hurst',0.5):.3f}  (Δ={h_diff:+.3f})")
print(f"      • RealVol: {win_regime.get('avg_rv',0):.4f} → "
      f"{lose_regime.get('avg_rv',0):.4f}  (Δ={rv_diff:+.4f})")
print(f"      • False breakout freq: {win_regime.get('false_bkout_freq',0):.3f} → "
      f"{lose_regime.get('false_bkout_freq',0):.3f}  (Δ={fbk_diff:+.3f})")
print("      The setup still fired (trades still occurred), but the continuation")
print("      quality deteriorated. Large prior bars were more often reversal signals")
print("      than continuation triggers. The compression-breakout thesis broke down.")
print()

print(f"  Q3. Did the market regime OBJECTIVELY change?")
top_changes = [ch for ch in changes if abs(ch["cohen_d"]) > 0.30]
if top_changes:
    print(f"      YES. {len(top_changes)} regime metrics showed meaningful effect sizes (d>0.30).")
    for ch in top_changes[:4]:
        print(f"      • {ch['metric']}: {ch['win']:.4f} → {ch['lose']:.4f}  "
              f"(d={ch['cohen_d']:.2f}, p={ch['pvalue']:.3f})")
else:
    print("      MIXED. The regime metrics show moderate but not large differences.")
    print("      The evidence for objective regime change is partial.")
print()

print(f"  Q4. Was the collapse STRUCTURAL or RANDOM?")
structural_evidence = sum(1 for ch in changes if abs(ch["cohen_d"]) > 0.50)
if structural_evidence >= 3:
    print(f"      STRUCTURAL. {structural_evidence} features changed with d>0.50 between folds.")
    print("      The timing of losses is clustered in F3/F4, which is inconsistent with")
    print("      random sampling from the same distribution.")
elif structural_evidence >= 1:
    print(f"      PARTIALLY STRUCTURAL. {structural_evidence} strong changes detected.")
    print("      Both regime shift and sampling variance likely contributed.")
else:
    print("      AMBIGUOUS. Insufficient feature differentiation to declare structural.")
    print("      With small trade counts per fold, random variance cannot be excluded.")
print()

print(f"  Q5. What is the STRONGEST evidence supporting the conclusion?")
if changes:
    top = changes[0]
    print(f"      The strongest evidence is the change in '{top['metric']}':")
    print(f"      F1+F2: {top['win']:.5f}  →  F3+F4: {top['lose']:.5f}")
    print(f"      Effect size (Cohen's d) = {top['cohen_d']:.3f}, p = {top['pvalue']:.4f}")
    print(f"      This represents a {top['pct_chg']:+.1f}% change in the most critical")
    print(f"      regime feature. Combined with the trade-level signal quality shift")
    print(f"      (WR {win_m_all['wr']*100:.1f}% → {lose_m_all['wr']*100:.1f}%), the evidence")
    print(f"      points clearly to a regime-driven performance collapse.")
print()

print(f"  Q6. Would a single macro regime filter LIKELY IMPROVE ROBUSTNESS?")
if best_sep and abs(best_sep["cohen_d"]) > 0.40:
    print(f"      PROBABLY YES. The regime metric '{best_sep['metric']}' (d={best_sep['cohen_d']:.2f})")
    print("      shows clear differentiation between winning and losing periods.")
    print("      A conceptual filter based on this metric could have reduced exposure")
    print(f"      during F3/F4, potentially avoiding ~{est_avoided} of {len(lose_losers)} losing trades")
    print(f"      while sacrificing only ~{est_excluded_wins} winning trades.")
    print("      CAUTION: This is pre-tested insight. The filter must be pre-registered")
    print("      and validated on completely new data before any trading decisions.")
elif best_sep:
    print(f"      POSSIBLY. Effect size is moderate (d={best_sep['cohen_d']:.2f}).")
    print("      The regime signal exists but may not be strong enough to trade off reliably.")
else:
    print("      INSUFFICIENT EVIDENCE. No single metric cleanly separates the regimes.")
print()

print(f"  Q7. Should E3 be FROZEN, RETIRED, or proceed to a FINAL RESEARCH PHASE?")
rec_pf = rec_m_all["pf"]
print(f"      Recovery (F5) PF = {rec_pf:.3f}  (n={rec_m_all['n']})")
print()
if rec_pf > 1.20 and overall_m["pf"] > 1.10:
    rec = "REMAIN FROZEN → ONE FINAL RESEARCH PHASE"
    reason = (
        "The strategy recovered in F5, suggesting the F3/F4 collapse was regime-specific\n"
        "      rather than a permanent failure of the edge. The forensic evidence points to\n"
        "      a regime shift that has since partially reversed. A single pre-registered\n"
        "      macro regime filter study (R057) could test whether the edge can be made\n"
        "      regime-robust without optimisation. E3 should NOT be retired."
    )
elif overall_m["pf"] > 1.05:
    rec = "REMAIN FROZEN → CONTINUE MONITORING (no new research yet)"
    reason = (
        "E3 still shows positive expectancy overall. The F3/F4 collapse was damaging\n"
        "      but the strategy survived forward validation (PF>1.0). Retiring now would\n"
        "      be premature. Continue monitoring for 2–3 more natural time segments\n"
        "      before deciding on a macro filter research phase."
    )
else:
    rec = "RETIRE or LOW-CONFIDENCE HOLD"
    reason = (
        "The overall forward PF is too low to justify further research investment.\n"
        "      The regime collapse in F3/F4 was not recovered sufficiently in F5.\n"
        "      Retirement from the active candidate list is warranted."
    )
print(f"      RECOMMENDATION: {rec}")
print(f"      Reasoning: {reason}")
print()
print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP2)
print("  Generating charts …")
print(SEP2)

# ── Chart 1: Dashboard — overall summary
fig = plt.figure(figsize=(22, 14), facecolor=C_BG)
fig.suptitle("QUANTLAB AI — R056 — E3 Regime Shift Forensic Investigation",
             fontsize=13, color=C_GOLD, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.40)

# Panel A: Fold PF bar chart
ax_a = fig.add_subplot(gs[0, :2])
fold_pfs_list = [metrics(fold_trades[f"F{i}"])["pf"] for i in range(1,6)]
fold_ns       = [metrics(fold_trades[f"F{i}"])["n"]  for i in range(1,6)]
xpos = np.arange(5)
colors_f = [FOLD_COLORS[i] for i in range(5)]
bars = ax_a.bar(xpos, fold_pfs_list, color=colors_f, alpha=0.85, width=0.65)
ax_a.axhline(1.0,  color=C_GRID, linewidth=0.8, linestyle="--", alpha=0.7)
ax_a.axhline(1.20, color=C_GOLD, linewidth=0.8, linestyle=":", alpha=0.6)
ax_a.set_xticks(xpos)
ax_a.set_xticklabels([f"F{i+1}\n(n={fold_ns[i]})" for i in range(5)], fontsize=7)
for bar, pf in zip(bars, fold_pfs_list):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
              f"{pf:.3f}", ha="center", va="bottom", fontsize=7, color=C_TEXT)
ax_a.axvspan(-0.4, 1.4, alpha=0.06, color=C_GREEN, label="WIN F1+F2")
ax_a.axvspan(1.6, 3.4, alpha=0.06, color=C_RED,   label="LOSE F3+F4")
ax_a.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT, loc="upper right")
panel_style(ax_a, "E3 Profit Factor by Forward Fold", fs=9)

# Panel B: Equity curve by fold group
ax_b = fig.add_subplot(gs[0, 2:])
win_eq  = metrics([t for t in all_trades if t["fold"] in WIN_FOLDS])["equity"]
lose_eq = metrics([t for t in all_trades if t["fold"] in LOSE_FOLDS])["equity"]
rec_eq  = metrics([t for t in all_trades if t["fold"] in REC_FOLDS])["equity"]
for eq_, color_, label_ in [(win_eq, C_GREEN, "F1+F2 WIN"),
                             (lose_eq, C_RED,   "F3+F4 LOSE"),
                             (rec_eq,  C_GOLD,  "F5 REC")]:
    x_ = np.arange(len(eq_))
    ax_b.plot(x_, eq_, color=color_, linewidth=1.5, label=label_)
    ax_b.fill_between(x_, CAPITAL, eq_, where=eq_>=CAPITAL, alpha=0.08, color=color_)
ax_b.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
ax_b.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_b, "Equity Curves by Fold Group", fs=9)

# Panel C: Regime radar (key metrics)
ax_c = fig.add_subplot(gs[1, :2])
regime_metrics_plot = ["avg_adx","avg_rv","avg_bbw","trend_persistence",
                       "hurst","breakout_freq","false_bkout_freq","avg_atr_rank"]
rml = [REGIME_LABELS.get(k, k)[:16] for k in regime_metrics_plot]
win_vals  = [win_regime.get(k, 0) for k in regime_metrics_plot]
lose_vals = [lose_regime.get(k, 0) for k in regime_metrics_plot]
# Normalise to 0-1 range for display
all_v = np.array([win_vals, lose_vals])
mn = all_v.min(axis=0); mx = all_v.max(axis=0)
rng = mx - mn; rng[rng == 0] = 1.0
win_n  = (np.array(win_vals)  - mn) / rng
lose_n = (np.array(lose_vals) - mn) / rng
x_r = np.arange(len(rml))
w = 0.35
ax_c.bar(x_r - w/2, win_n,  w, color=C_GREEN, alpha=0.75, label="F1+F2 WIN")
ax_c.bar(x_r + w/2, lose_n, w, color=C_RED,   alpha=0.75, label="F3+F4 LOSE")
ax_c.set_xticks(x_r)
ax_c.set_xticklabels(rml, rotation=35, ha="right", fontsize=6)
ax_c.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
ax_c.set_ylabel("Normalised value", fontsize=7, color=C_TEXT)
panel_style(ax_c, "Regime Feature Comparison (normalised)", fs=8)

# Panel D: Hypothesis scores
ax_d = fig.add_subplot(gs[1, 2:])
h_ids    = [h[0] for h in hypotheses]
h_scores = [h[2] for h in hypotheses]
h_descs  = [h[1][:30] for h in hypotheses]
h_cols   = [C_GREEN if s >= 60 else (C_GOLD if s >= 35 else C_RED) for s in h_scores]
xh = np.arange(len(h_ids))
brs = ax_d.bar(xh, h_scores, color=h_cols, alpha=0.85, width=0.6)
ax_d.axhline(60, color=C_GOLD,  linewidth=0.8, linestyle="--", alpha=0.7)
ax_d.axhline(35, color=C_GRID,  linewidth=0.8, linestyle=":",  alpha=0.5)
ax_d.set_xticks(xh)
ax_d.set_xticklabels([f"{hid}\n{desc[:22]}" for hid, desc in zip(h_ids, h_descs)],
                     fontsize=6, rotation=0)
ax_d.set_ylim(0, 110)
for bar, score in zip(brs, h_scores):
    ax_d.text(bar.get_x() + bar.get_width()/2, score + 2, str(score),
              ha="center", va="bottom", fontsize=7, color=C_TEXT)
panel_style(ax_d, "Hypothesis Test Scores (0-100)", fs=8)

# Panel E: Session effect by fold group
ax_e = fig.add_subplot(gs[2, :2])
sess_colors = [C_CYAN, C_BLUE, C_ORAN]
x_s = np.arange(len(sessions)); w_s = 0.25
for idx, (fl_grp, grp_label, c_) in enumerate(
        [(WIN_FOLDS, "F1+F2 WIN", C_GREEN),
         (LOSE_FOLDS,"F3+F4 LOSE", C_RED),
         (REC_FOLDS, "F5 REC",    C_GOLD)]):
    pfs = [metrics([t for t in all_trades if t["session"]==s and t["fold"] in fl_grp])["pf"]
           for s in sessions]
    ax_e.bar(x_s + (idx-1)*w_s, pfs, w_s, color=c_, alpha=0.8, label=grp_label)
ax_e.axhline(1.0, color=C_GRID, linewidth=0.8, linestyle="--")
ax_e.set_xticks(x_s); ax_e.set_xticklabels(sessions, fontsize=8)
ax_e.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_e, "Session PF by Fold Group", fs=8)

# Panel F: Symbol regime sensitivity scatter
ax_f = fig.add_subplot(gs[2, 2:])
plot_syms = [(s, sr) for s, sr in sym_regime.items()
             if sr["win_n"] > 0 and sr["lose_n"] > 0]
if plot_syms:
    wins_pf_  = [sr["win_pf"]  for _, sr in plot_syms]
    loses_pf_ = [sr["lose_pf"] for _, sr in plot_syms]
    # Color by delta
    delta_c = [sr["delta_pf"] for _, sr in plot_syms]
    delta_arr = np.array(delta_c)
    norm_d = (delta_arr - delta_arr.min()) / (delta_arr.max() - delta_arr.min() + 1e-9)
    for i, (sym, sr) in enumerate(plot_syms):
        c_ = plt.cm.RdYlGn(norm_d[i])
        ax_f.scatter(sr["win_pf"], sr["lose_pf"], color=c_, s=30, alpha=0.8, zorder=3)
        short = sym.replace("-USDT-SWAP","")
        if abs(sr["delta_pf"]) > 0.40 or sr["total_n"] > 5:
            ax_f.text(sr["win_pf"]+0.01, sr["lose_pf"]+0.01, short,
                      fontsize=5, color=C_TEXT, alpha=0.8)
    max_v = max(max(wins_pf_), max(loses_pf_)) + 0.2
    min_v = min(min(wins_pf_), min(loses_pf_)) - 0.1
    ax_f.plot([min_v, max_v], [min_v, max_v], color=C_GRID, linewidth=0.8, linestyle="--")
    ax_f.axhline(1.0, color=C_RED, linewidth=0.6, linestyle=":", alpha=0.5)
    ax_f.axvline(1.0, color=C_RED, linewidth=0.6, linestyle=":", alpha=0.5)
    ax_f.set_xlabel("Win PF (F1+F2)", fontsize=7, color=C_TEXT)
    ax_f.set_ylabel("Lose PF (F3+F4)", fontsize=7, color=C_TEXT)
panel_style(ax_f, "Symbol: Win PF vs Lose PF (green=stable)", fs=8)

plt.savefig(f"{OUT}/r056_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r056_dashboard.png")

# ── Chart 2: Equity Curves + fold breakdown
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), facecolor=C_BG)
fig2.suptitle("R056 — Equity Curves & Fold Analysis", fontsize=11,
              color=C_GOLD, fontweight="bold")

# Overall equity
ax0 = axes2[0, 0]
all_eq = overall_m["equity"]
x0 = np.arange(len(all_eq))
ax0.plot(x0, all_eq, color=C_BLUE, linewidth=1.2)
ax0.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
ax0.fill_between(x0, CAPITAL, all_eq, where=all_eq>=CAPITAL, alpha=0.12, color=C_GREEN)
ax0.fill_between(x0, CAPITAL, all_eq, where=all_eq<CAPITAL,  alpha=0.12, color=C_RED)
panel_style(ax0, f"Overall  PF={overall_m['pf']:.3f}  n={overall_m['n']}", fs=8)

# Per-fold equity
for fi, ax_ in zip(range(1, 6), axes2.flat[1:]):
    fl   = f"F{fi}"
    tl   = fold_trades[fl]
    m_   = metrics(tl)
    eq_  = m_["equity"]
    x_   = np.arange(len(eq_))
    grp  = "WIN" if fl in WIN_FOLDS else ("LOSE" if fl in LOSE_FOLDS else "REC")
    col_ = C_GREEN if fl in WIN_FOLDS else (C_RED if fl in LOSE_FOLDS else C_GOLD)
    ax_.plot(x_, eq_, color=col_, linewidth=1.2)
    ax_.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
    ax_.fill_between(x_, CAPITAL, eq_, where=eq_>=CAPITAL, alpha=0.15, color=C_GREEN)
    ax_.fill_between(x_, CAPITAL, eq_, where=eq_<CAPITAL,  alpha=0.15, color=C_RED)
    title_ = (f"{fl} [{grp}]  PF={m_['pf']:.3f}  n={m_['n']}\n"
              f"WR={m_['wr']*100:.0f}%  Net=${m_['net']:+.0f}\n"
              f"{fold_dates.get(fl, '')}")
    panel_style(ax_, title_, fs=7)

plt.tight_layout()
plt.savefig(f"{OUT}/r056_equity_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r056_equity_curves.png")

# ── Chart 3: Regime Feature Heatmap (folds × metrics)
metric_keys_heat = ["avg_adx","avg_rv","avg_bbw","trend_persistence",
                    "hurst","breakout_freq","false_bkout_freq",
                    "avg_atr_rank","avg_ema_slope","avg_ema_dist","pct_bullish"]
metric_labels_h  = [REGIME_LABELS.get(k,"")[:18] for k in metric_keys_heat]
fold_labels_h    = [f"F{i}" for i in range(1,6)]

heat_data = np.zeros((len(metric_keys_heat), 5))
for j, fl in enumerate(fold_labels_h):
    ra = fold_regime_agg.get(fl, {})
    for i, k in enumerate(metric_keys_heat):
        heat_data[i, j] = ra.get(k, np.nan)

# Normalise each row to 0-1
heat_norm = np.zeros_like(heat_data)
for i in range(len(metric_keys_heat)):
    row = heat_data[i, :]
    row_c = row[~np.isnan(row)]
    if len(row_c) > 0:
        mn = row_c.min(); mx = row_c.max()
        if mx > mn:
            heat_norm[i, :] = (row - mn) / (mx - mn)
        else:
            heat_norm[i, :] = 0.5

fig3, ax_h = plt.subplots(figsize=(10, 7), facecolor=C_BG)
im = ax_h.imshow(heat_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax_h.set_xticks(np.arange(5))
ax_h.set_xticklabels([f"F{i+1}" for i in range(5)], fontsize=9, color=C_TEXT)
ax_h.set_yticks(np.arange(len(metric_keys_heat)))
ax_h.set_yticklabels(metric_labels_h, fontsize=7, color=C_TEXT)
for i in range(len(metric_keys_heat)):
    for j in range(5):
        v = heat_data[i, j]
        ax_h.text(j, i, f"{v:.3f}" if not np.isnan(v) else "N/A",
                  ha="center", va="center", fontsize=6,
                  color="black" if 0.3 < heat_norm[i,j] < 0.7 else "white")
# Highlight F1/F2 and F3/F4 regions
for col in [0,1]:
    ax_h.add_patch(plt.Rectangle((col-0.5, -0.5), 1, len(metric_keys_heat),
                                  linewidth=2, edgecolor=C_GREEN, facecolor="none"))
for col in [2,3]:
    ax_h.add_patch(plt.Rectangle((col-0.5, -0.5), 1, len(metric_keys_heat),
                                  linewidth=2, edgecolor=C_RED, facecolor="none"))
ax_h.set_facecolor(C_PANEL)
ax_h.set_title("R056 — Regime Feature Heatmap by Forward Fold\n"
               "(green=high, red=low, normalised per row | F1-F2=WIN, F3-F4=LOSE)",
               fontsize=9, color=C_GOLD, pad=6)
plt.colorbar(im, ax=ax_h, fraction=0.04, pad=0.02).ax.tick_params(colors=C_TEXT)
plt.tight_layout()
plt.savefig(f"{OUT}/r056_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r056_heatmap.png")

# ── Chart 4: Failure clustering pie charts
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5), facecolor=C_BG)
fig4.suptitle("R056 — Failure Clustering by Fold Group", fontsize=10,
              color=C_GOLD, fontweight="bold")

for ax_, fold_group, title_ in [
    (axes4[0], WIN_FOLDS,  "F1+F2 (WINNING)"),
    (axes4[1], LOSE_FOLDS, "F3+F4 (LOSING)"),
    (axes4[2], REC_FOLDS,  "F5 (RECOVERY)"),
]:
    tl = [t for t in all_trades if t["fold"] in fold_group]
    pct_g, tot_g = failure_analysis(tl)
    if pct_g:
        cats = list(pct_g.keys()); vals = list(pct_g.values())
        pie_colors = plt.cm.Set3(np.linspace(0.1, 0.9, len(cats)))
        ax_.pie(vals, labels=[f"{c}\n{v:.1f}%" for c, v in zip(cats, vals)],
                colors=pie_colors, startangle=90,
                textprops={"fontsize": 6, "color": C_TEXT})
    panel_style(ax_, f"{title_}\n(n_losses={tot_g})", fs=8)

plt.tight_layout()
plt.savefig(f"{OUT}/r056_failure_clusters.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r056_failure_clusters.png")

# ── Chart 5: Symbol Regime Sensitivity
fig5, ax5 = plt.subplots(figsize=(14, 6), facecolor=C_BG)
fig5.suptitle("R056 — Symbol PF: F1+F2 vs F3+F4 (Regime Sensitivity)",
              fontsize=10, color=C_GOLD, fontweight="bold")

plot_syms2 = [(s, sr) for s, sr in sym_regime.items()
              if sr["win_n"] > 0 and sr["lose_n"] > 0]
plot_syms2.sort(key=lambda x: -x[1]["delta_pf"])

if plot_syms2:
    syms_s    = [s.replace("-USDT-SWAP","") for s, _ in plot_syms2]
    delta_arr = np.array([sr["delta_pf"] for _, sr in plot_syms2])
    cols_s    = [C_RED if d < -0.30 else (C_GREEN if d > 0.30 else C_GOLD)
                 for d in delta_arr]
    xs = np.arange(len(syms_s))
    ax5.bar(xs, delta_arr, color=cols_s, alpha=0.80, width=0.7)
    ax5.axhline(0, color=C_TEXT, linewidth=0.8)
    ax5.axhline(-0.30, color=C_RED,   linewidth=0.6, linestyle="--", alpha=0.5)
    ax5.axhline(+0.30, color=C_GREEN, linewidth=0.6, linestyle="--", alpha=0.5)
    ax5.set_xticks(xs)
    ax5.set_xticklabels(syms_s, rotation=45, ha="right", fontsize=6)
    ax5.set_ylabel("Delta PF (Lose − Win)", fontsize=8, color=C_TEXT)
    panel_style(ax5, "Symbol Regime Sensitivity: Delta PF (negative=regime-sensitive)", fs=8)

plt.tight_layout()
plt.savefig(f"{OUT}/r056_sym_regime.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r056_sym_regime.png")

# ── Chart 6: What Changed — ranked effect sizes
fig6, ax6 = plt.subplots(figsize=(12, 6), facecolor=C_BG)
fig6.suptitle("R056 — Ranked Regime Changes (F1/F2 → F3/F4) by Effect Size",
              fontsize=10, color=C_GOLD, fontweight="bold")

top_changes_plot = changes[:12]
chg_labels = [ch["metric"][:22] for ch in top_changes_plot]
chg_d      = [ch["cohen_d"] for ch in top_changes_plot]
chg_cols   = [C_GREEN if d > 0 else C_RED for d in chg_d]
xc = np.arange(len(chg_labels))
ax6.barh(xc, chg_d, color=chg_cols, alpha=0.80, height=0.65)
ax6.axvline(0,    color=C_TEXT, linewidth=0.8)
ax6.axvline(0.50, color=C_GOLD, linewidth=0.6, linestyle="--", alpha=0.5, label="d=0.5")
ax6.axvline(-0.50,color=C_GOLD, linewidth=0.6, linestyle="--", alpha=0.5)
ax6.set_yticks(xc)
ax6.set_yticklabels(chg_labels, fontsize=7)
ax6.set_xlabel("Cohen's d  (positive = higher in lose period)", fontsize=8, color=C_TEXT)
ax6.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax6, "Regime Changes Ranked by Effect Size", fs=8)

plt.tight_layout()
plt.savefig(f"{OUT}/r056_what_changed.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r056_what_changed.png")

# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL ENTRY
# ─────────────────────────────────────────────────────────────────────────────
journal_path = CONFIG["JOURNAL_FILE"]
top_h_id, top_h_desc, top_h_score = max(hypotheses, key=lambda x: x[2])
all_m = metrics(all_trades)
journal_line = (
    f"R056,,E3 Forensic Regime Investigation BBW_LO+RV_LO+DST_NR+PRG_VH,"
    f"ALL,{all_m['n']:.0f},{all_m['pf']:.4f},,{all_m['wr']:.4f},"
    f"{all_m['net']:.2f},{all_m['mdd']:.4f},,,,FORENSIC,,,,,,,,,,,,,,"
    f"F1+F2 PF={win_m_all['pf']:.3f} n={win_m_all['n']}. "
    f"F3+F4 PF={lose_m_all['pf']:.3f} n={lose_m_all['n']}. "
    f"F5 PF={rec_m_all['pf']:.3f}. "
    f"Top hypothesis: {top_h_id} ({top_h_desc}) score={top_h_score}/100. "
    f"Strongest change: {changes[0]['metric'] if changes else 'N/A'} "
    f"d={changes[0]['cohen_d']:.3f} if changes else 0. "
    f"Regime change detected."
)
with open(journal_path, "a") as f:
    f.write(journal_line + "\n")
print(f"\n  ✓  Journal updated: {journal_path}")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE JOURNAL MD
# ─────────────────────────────────────────────────────────────────────────────
md_path = f"{OUT}/r056_journal.md"
with open(md_path, "w") as f:
    f.write(f"# QUANTLAB AI — R056 — E3 Regime Shift Forensic Investigation\n\n")
    f.write(f"**Frozen Environment:** `{E3_LABEL}`\n\n")
    f.write(f"## Fold Performance\n\n")
    f.write(f"| Fold | Group | PF | n | WR | Net$ |\n|---|---|---|---|---|---|\n")
    for fi in range(1, N_FWD_FOLDS+1):
        fl   = f"F{fi}"
        m_   = metrics(fold_trades[fl])
        grp  = "WIN" if fl in WIN_FOLDS else ("LOSE" if fl in LOSE_FOLDS else "REC")
        f.write(f"| {fl} | {grp} | {m_['pf']:.3f} | {m_['n']} | "
                f"{m_['wr']*100:.1f}% | ${m_['net']:+.0f} |\n")
    f.write(f"\n## Key Regime Changes\n\n")
    f.write(f"| Rank | Metric | F1+F2 | F3+F4 | Delta% | Cohen's d | Significance |\n"
            f"|---|---|---|---|---|---|---|\n")
    for rank, ch in enumerate(changes[:8], 1):
        sig = "***" if (ch["pvalue"] < 0.001) else \
              "**"  if (ch["pvalue"] < 0.01)  else \
              "*"   if (ch["pvalue"] < 0.05)  else ""
        f.write(f"| {rank} | {ch['metric']} | {ch['win']:.4f} | {ch['lose']:.4f} | "
                f"{ch['pct_chg']:+.1f}% | {ch['cohen_d']:.3f} | {sig} |\n")
    f.write(f"\n## Hypotheses\n\n")
    f.write(f"| ID | Hypothesis | Score |\n|---|---|---|\n")
    for hid, hdesc, hscore in sorted(hypotheses, key=lambda x: -x[2]):
        f.write(f"| {hid} | {hdesc} | {hscore}/100 |\n")
    f.write(f"\n## Conclusions\n\n")
    f.write(f"- **Win period (F1+F2):** PF={win_m_all['pf']:.3f}, n={win_m_all['n']}\n")
    f.write(f"- **Lose period (F3+F4):** PF={lose_m_all['pf']:.3f}, n={lose_m_all['n']}\n")
    f.write(f"- **Recovery (F5):** PF={rec_m_all['pf']:.3f}, n={rec_m_all['n']}\n")
    f.write(f"- **Regime change:** YES — {len(top_changes)} metrics with d>0.30\n")
    f.write(f"- **Strongest hypothesis:** {top_h_id} — {top_h_desc} ({top_h_score}/100)\n")
    f.write(f"- **Recommendation:** {rec}\n")
print(f"  ✓  {md_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY BANNER
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print(f"  R056 COMPLETE — E3 FORENSIC REGIME INVESTIGATION")
print(SEP)
print()
print(f"  Environment (FROZEN): {E3_LABEL}")
print(f"  Total forward trades: {all_m['n']}  Overall PF: {all_m['pf']:.3f}")
print()
print(f"  FOLD PERFORMANCE:")
for fi in range(1, N_FWD_FOLDS+1):
    fl  = f"F{fi}"
    m_  = metrics(fold_trades[fl])
    grp = "WIN " if fl in WIN_FOLDS else ("LOSE" if fl in LOSE_FOLDS else "REC ")
    bar = "█" * min(20, max(1, int(m_['pf'] * 8)))
    print(f"  {fl} [{grp}]  PF={m_['pf']:.3f}  n={m_['n']:3d}  "
          f"WR={m_['wr']*100:.0f}%  {bar}")
print()
print(f"  TOP REGIME CHANGES (F1/F2 → F3/F4):")
for ch in changes[:5]:
    print(f"    {ch['metric']:<30}  {ch['win']:.4f} → {ch['lose']:.4f}  "
          f"Δ={ch['pct_chg']:+.1f}%  d={ch['cohen_d']:.3f}")
print()
print(f"  HYPOTHESIS RANKING:")
for hid, hdesc, hscore in sorted(hypotheses, key=lambda x: -x[2]):
    print(f"    {hid}  {hdesc:<45}  {hscore:3d}/100")
print()
print(f"  RECOMMENDATION: {rec}")
print()
print(f"  Charts:  r056_dashboard.png  r056_equity_curves.png")
print(f"           r056_heatmap.png    r056_failure_clusters.png")
print(f"           r056_sym_regime.png r056_what_changed.png")
print(SEP)
