"""
QUANTLAB AI — R065
Forensic Investigation: RV_HI + DST_MD + ADX_WK + LON

Status: FROZEN — no optimisation, no threshold tuning, no parameter search.
This is a scientific forensic investigation only.

Objective:
  R064 discovered RV_HI+DST_MD+ADX_WK+LON as the champion family.
  PF=2.188, UES=94.8, 0% overlap with E3.1.
  This script determines whether that edge is genuine or a statistical anomaly.

Sections:
  1  Complete Trade Profile
  2  Profit Distribution
  3  Symbol Dependency
  4  Temporal Stability
  5  Condition Ablation
  6  Entry Gate Importance
  7  Market Regime Analysis
  8  MAE / MFE
  9  Portfolio Fit vs E3.1
  10 Stress Test
  +  Final 10 Answers
"""

import os, sys, math, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID  = "R065"
OUT          = CONFIG["OUTPUT_FOLDER"]
CACHE        = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL      = CONFIG["STARTING_CAPITAL"]
RR           = CONFIG["RISK_REWARD"]
IS_RATIO     = 0.80
MIN_BARS     = 2_000
N_FWD_FOLDS  = 5
N_BOOT       = 3_000
N_MC         = 3_000
N_PERM       = 1_000
RAND_SEED    = 42

# Champion environment — FROZEN
CHAMP_CIDS   = ("RV_HI", "DST_MD", "ADX_WK", "LON")

# E3.1 reference — FROZEN
E31_LABEL    = "BBW_STRICT+RV_LO+DST_NR+PRG_VH"

# Promotion thresholds (same as prior rounds)
PROM_PF      = 1.30
PROM_N       = 30
PROM_BOOT    = 1.20
PROM_MC      = 0.80
PROM_MDD     = 0.20

SEP  = "═" * 110
SEP2 = "─" * 90

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_BG    = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT  = "#e0e0e0"
C_GRID  = "#2a2a2a"; C_GREEN = "#00c896"; C_RED   = "#e05050"
C_GOLD  = "#f5a623"; C_BLUE  = "#4a9eff"; C_PURP  = "#9b59b6"
PALETTE = [C_GREEN, C_GOLD, C_BLUE, C_RED, C_PURP,
           "#e67e22","#1abc9c","#3498db","#e74c3c","#f39c12",
           "#2ecc71","#e91e63","#00bcd4","#ff5722","#8bc34a",
           "#795548","#607d8b","#ff9800","#673ab7","#26c6da"]

plt.rcParams.update({
    "figure.facecolor":C_BG, "axes.facecolor":C_PANEL,
    "text.color":C_TEXT, "axes.labelcolor":C_TEXT,
    "xtick.color":C_TEXT, "ytick.color":C_TEXT,
    "axes.edgecolor":C_GRID, "grid.color":C_GRID, "font.family":"monospace",
})

DOW_NAMES    = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
SESSION_NAMES= ["London","US","Asia"]

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION DEFINITIONS — frozen from R064
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    ("ATR_LO",   "ATR<p25",      "atr_rank",      "lt_q",      0.25, "vol"),
    ("ATR_MD",   "ATR<p40",      "atr_rank",      "lt_q",      0.40, "vol"),
    ("ATR_HI",   "ATR>p67",      "atr_rank",      "gt_q",      0.67, "vol"),
    ("ATR_VH",   "ATR>p80",      "atr_rank",      "gt_q",      0.80, "vol"),
    ("BBW_LO",   "BBW<p33",      "bb_width",      "lt_q",      0.33, "vol"),
    ("BBW_HI",   "BBW>p67",      "bb_width",      "gt_q",      0.67, "vol"),
    ("BBP_LO",   "BBPos<p33",    "bb_pos",        "lt_q",      0.33, "vol"),
    ("BBP_HI",   "BBPos>p67",    "bb_pos",        "gt_q",      0.67, "vol"),
    ("RV_LO",    "RealVol<p33",  "real_vol_20",   "lt_q",      0.33, "vol"),
    ("RV_HI",    "RealVol>p67",  "real_vol_20",   "gt_q",      0.67, "vol"),
    ("RVOL_HI",  "RelVol>p67",   "rel_vol_rank",  "gt_q",      0.67, "vol"),
    ("RVOL_LO",  "RelVol<p33",   "rel_vol_rank",  "lt_q",      0.33, "vol"),
    ("SLP_DN",   "Slope<0",      "ema200_slope",  "lt_fixed",  0.0,  "trend"),
    ("SLP_UP",   "Slope>0",      "ema200_slope",  "gt_fixed",  0.0,  "trend"),
    ("DST_NR",   "Dist<p33",     "ema_dist_pct",  "lt_q",      0.33, "trend"),
    ("DST_MD",   "Dist>p60+",    "ema_dist_pct",  "gt_q_pos",  0.60, "trend"),
    ("DST_FR",   "Dist>p75+",    "ema_dist_pct",  "gt_q_pos",  0.75, "trend"),
    ("EMA50_NR", "E50<p33",      "ema50_dist_pct","lt_q",      0.33, "trend"),
    ("EMA50_AB", "E50>p67",      "ema50_dist_pct","gt_q",      0.67, "trend"),
    ("ADX_WK",   "ADX<p33",      "adx14",         "lt_q",      0.33, "trend"),
    ("ADX_TR",   "ADX>p50",      "adx14",         "gt_q",      0.50, "trend"),
    ("ADX_ST",   "ADX>p67",      "adx14",         "gt_q",      0.67, "trend"),
    ("PRG_LO",   "PrevRng<p33",  "prev_range_r",  "lt_q",      0.33, "prev"),
    ("PRG_HI",   "PrevRng>p67",  "prev_range_r",  "gt_q",      0.67, "prev"),
    ("PRG_VH",   "PrevRng>p80",  "prev_range_r",  "gt_q",      0.80, "prev"),
    ("PBP_LO",   "PrevBdy<p33",  "prev_body_pct", "lt_q",      0.33, "prev"),
    ("PBP_HI",   "PrevBdy>p67",  "prev_body_pct", "gt_q",      0.67, "prev"),
    ("CLH_HI",   "ClsHigh>p67",  "close_high_r",  "gt_q",      0.67, "prev"),
    ("CLH_LO",   "ClsHigh<p33",  "close_high_r",  "lt_q",      0.33, "prev"),
    ("LON",      "London(7-14)", "hour_utc",      "hour_rng",  (7,14),"session"),
    ("US",       "US(14-21)",    "hour_utc",      "hour_rng",  (14,21),"session"),
    ("ASI",      "Asia(22-6)",   "hour_utc",      "hour_rng",  (22, 6),"session"),
]

COND_BY_ID = {c[0]: c for c in CONDITIONS_DEF}
QUANT_FEATS = ["atr_rank","bb_width","bb_pos","real_vol_20","rel_vol_rank",
               "ema_dist_pct","ema50_dist_pct","ema200_slope","adx14",
               "prev_range_r","prev_body_pct","close_high_r"]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c  = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]; o = df["open"]

    df["ema200"]         = calc_ema(c, 200)
    df["ema50"]          = calc_ema(c, 50)
    df["atr14"]          = calc_atr(df, 14)
    df["atr_rank"]       = df["atr14"].rolling(100).rank(pct=True) * 100

    bb_mid               = c.rolling(20).mean()
    bb_std               = c.rolling(20).std()
    bb_upper             = bb_mid + 2 * bb_std
    bb_lower             = bb_mid - 2 * bb_std
    bb_range             = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_width"]       = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["bb_pos"]         = (c - bb_lower) / bb_range

    df["ema_dist_pct"]   = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"]   = (df["ema200"] - df["ema200"].shift(10)) / \
                           df["ema200"].shift(10).replace(0, np.nan)
    df["ema50_dist_pct"] = (c - df["ema50"]) / df["ema50"].replace(0, np.nan) * 100

    vol_ma               = v.rolling(20).mean()
    df["rel_vol"]        = v / vol_ma.replace(0, np.nan)
    df["rel_vol_rank"]   = df["rel_vol"].rolling(100).rank(pct=True) * 100

    log_ret              = np.log(c / c.shift(1))
    df["real_vol_20"]    = log_ret.rolling(20).std() * 100.0

    df["adx14"]          = calc_adx(df, 14)
    df["hurst"]          = _calc_hurst(c)

    df["prev_close"]     = c.shift(1)
    df["prev_atr14"]     = df["atr14"].shift(1)
    prev_range           = h.shift(1) - l.shift(1)
    prev_body            = (c.shift(1) - o.shift(1)).abs()
    df["prev_range_r"]   = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]    = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"]  = prev_body  / prev_range.replace(0, np.nan)

    bar_range            = h - l
    df["close_high_r"]   = (c - l) / bar_range.replace(0, np.nan)
    df["body_pct"]       = (c - o).abs() / bar_range.replace(0, np.nan)
    df["range_pct"]      = bar_range / c.replace(0, np.nan) * 100.0
    df["ema200_slope_abs"]= df["ema200_slope"].abs()

    dt                   = pd.to_datetime(df["datetime"], utc=True)
    df["hour_utc"]       = dt.dt.hour.astype(np.int16)
    df["dow"]            = dt.dt.dayofweek
    df["month"]          = dt.dt.month

    return df

def _calc_hurst(series, min_n=50):
    """Simplified Hurst via R/S for rolling 50-bar window."""
    vals = series.values.astype(float)
    n    = len(vals)
    result = np.full(n, np.nan)
    for i in range(min_n - 1, n):
        x = vals[i - min_n + 1: i + 1]
        x = x[~np.isnan(x)]
        if len(x) < 20:
            continue
        lx  = np.log(x / x[0] + 1e-15)
        cumd = np.cumsum(lx - lx.mean())
        r   = cumd.max() - cumd.min()
        s   = x.std()
        if s > 0 and r > 0:
            result[i] = math.log(r / s) / math.log(min_n)
    return pd.Series(result, index=series.index)

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def learn_thresholds(df_is):
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (_, _, feat, direction, param, _) in COND_BY_ID.items():
        if direction in ("gt_fixed", "lt_fixed", "hour_rng"):
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
    # BBW_STRICT for E3.1
    if "bb_width" in valid.columns:
        thr["BBW_STRICT"] = float(valid["bb_width"].dropna().quantile(0.25))
    return thr

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION MASK BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_cond_mask(col, nan_m, direction, thr_val):
    if direction == "lt_q":
        return (~nan_m) & (col < thr_val) if not np.isnan(thr_val) else np.zeros(len(col),bool)
    if direction == "gt_q":
        return (~nan_m) & (col > thr_val) if not np.isnan(thr_val) else np.zeros(len(col),bool)
    if direction == "gt_q_pos":
        return (~nan_m) & (col > thr_val) & (col > 0) if not np.isnan(thr_val) else np.zeros(len(col),bool)
    if direction == "lt_fixed":
        return (~nan_m) & (col < thr_val)
    if direction == "gt_fixed":
        return (~nan_m) & (col > thr_val)
    if direction == "hour_rng":
        lo, hi = thr_val
        if lo <= hi:
            return (col >= lo) & (col <= hi)
        return (col >= lo) | (col <= hi)
    return np.zeros(len(col), dtype=bool)

def build_env_mask(df, cond_ids, thr):
    N    = len(df)
    mask = np.ones(N, dtype=bool)
    for cid in cond_ids:
        _, _, feat, direction, _, _ = COND_BY_ID[cid]
        if feat not in df.columns:
            return np.zeros(N, dtype=bool)
        col   = df[feat].values
        nan_m = np.isnan(col) if col.dtype.kind == "f" else np.zeros(N, dtype=bool)
        mask &= build_cond_mask(col, nan_m, direction, thr.get(cid, np.nan))
    return mask

def build_e31_mask(df, thr):
    N = len(df); mask = np.ones(N, dtype=bool)
    for cid, feat, direction in [
        ("BBW_STRICT","bb_width",     "lt_q"),
        ("RV_LO",     "real_vol_20",  "lt_q"),
        ("DST_NR",    "ema_dist_pct", "lt_q"),
        ("PRG_VH",    "prev_range_r", "gt_q"),
    ]:
        t = thr.get(cid, np.nan)
        if np.isnan(t): return np.zeros(N, dtype=bool)
        col  = df[feat].values
        nanm = np.isnan(col)
        if direction == "lt_q": mask &= (~nanm) & (col < t)
        else:                   mask &= (~nanm) & (col > t)
    return mask

def entry_signal(df, env_mask):
    rv = df["rel_vol"].values; c = df["close"].values
    o  = df["open"].values;    pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & env_mask

# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED BACKTEST ENGINE (with MAE/MFE and full metadata)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest_extended(df, signal, sym, fold_label, max_hold_bars=200):
    """
    Full backtest with MAE, MFE, hold time, all features at entry.
    """
    min_sl = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf     = CONFIG["RISK_PER_TRADE_PCT"]
    fee    = CONFIG["TAKER_FEE"];  spd = CONFIG["SPREAD"] * 0.5
    slp    = CONFIG["SL_SLIPPAGE"]

    in_pos = False; ep = st = tk = sz = 0.0; et = None; ei = -1
    trades = []

    hi_  = df["high"].values
    lo_  = df["low"].values
    op_  = df["open"].values
    atr_ = df["prev_atr14"].values
    dts  = df["datetime"].values
    hou_ = df["hour_utc"].values
    dow_ = df["dow"].values
    rv_  = df["rel_vol"].values

    # Extra regime features at entry
    adx_  = df["adx14"].values
    bb_   = df["bb_width"].values
    rvol_ = df["real_vol_20"].values
    dst_  = df["ema_dist_pct"].values
    slp_  = df["ema200_slope"].values
    bpct_ = df["body_pct"].values
    rpct_ = df["range_pct"].values
    hurs_ = df["hurst"].values if "hurst" in df.columns else np.full(len(df), np.nan)
    atrr_ = df["atr_rank"].values

    N = len(df)

    for i in range(1, N):
        if in_pos:
            sl_hit = lo_[i] <= st
            tp_hit = hi_[i] >= tk

            if sl_hit or tp_hit or (i - ei) >= max_hold_bars:
                if sl_hit:
                    xp = st * (1 - slp)
                elif tp_hit:
                    xp = tk
                else:
                    xp = op_[i]  # max-hold timeout at open

                gross = (xp - ep) * sz
                cost  = (ep * sz + xp * sz) * (fee + spd)
                slpc  = (st - xp) * sz if sl_hit else 0.0
                net   = gross - cost - slpc

                # MAE / MFE over holding period
                bars_held = i - ei
                seg_h = hi_[ei:i+1]
                seg_l = lo_[ei:i+1]
                mfe   = float((seg_h.max() - ep) / (tk - ep)) if tk > ep else 0.0
                mae   = float((ep - seg_l.min()) / (ep - st)) if ep > st else 0.0

                h_e  = int(hou_[ei])
                sess = ("London" if 7 <= h_e <= 13 else
                        "US"     if 14 <= h_e <= 20 else "Asia")
                trades.append({
                    "sym":       sym,
                    "fold":      fold_label,
                    "entry_ts":  str(et),
                    "exit_ts":   str(dts[i]),
                    "pnl":       round(net, 4),
                    "win":       int(tp_hit and not sl_hit),
                    "sl_hit":    int(sl_hit),
                    "tp_hit":    int(tp_hit),
                    "timeout":   int(not sl_hit and not tp_hit),
                    "bars_held": bars_held,
                    "session":   sess,
                    "dow":       int(dow_[ei]),
                    "rel_vol":   float(rv_[ei]) if not np.isnan(rv_[ei]) else 1.0,
                    "entry_price": float(ep),
                    "stop_loss":   float(st),
                    "take_profit": float(tk),
                    "atr_at_entry": float(atr_[ei]) if not np.isnan(atr_[ei]) else np.nan,
                    "mfe_r":     mfe,
                    "mae_r":     mae,
                    # Regime at entry
                    "adx":       float(adx_[ei]) if not np.isnan(adx_[ei]) else np.nan,
                    "bb_width":  float(bb_[ei])  if not np.isnan(bb_[ei])  else np.nan,
                    "real_vol":  float(rvol_[ei]) if not np.isnan(rvol_[ei]) else np.nan,
                    "ema_dist":  float(dst_[ei]) if not np.isnan(dst_[ei]) else np.nan,
                    "ema_slope": float(slp_[ei]) if not np.isnan(slp_[ei]) else np.nan,
                    "body_pct":  float(bpct_[ei]) if not np.isnan(bpct_[ei]) else np.nan,
                    "range_pct": float(rpct_[ei]) if not np.isnan(rpct_[ei]) else np.nan,
                    "hurst":     float(hurs_[ei]) if not np.isnan(hurs_[ei]) else np.nan,
                    "atr_rank":  float(atrr_[ei]) if not np.isnan(atrr_[ei]) else np.nan,
                })
                in_pos = False
            continue

        if signal[i-1]:
            a = atr_[i]
            if np.isnan(a) or a <= 0: continue
            ep_ = op_[i]
            if a / max(ep_, 1e-10) < min_sl: continue
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
        return {"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,"net":0.0,"mdd":0.0,
                "avg_win":0.0,"avg_loss":0.0,"pnls":np.array([]),
                "equity":np.array([CAPITAL]),"wins_arr":np.array([])}
    pnl  = np.array([t["pnl"] for t in trades])
    wins = np.array([t["win"] for t in trades], dtype=bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw   = pnl[wins].sum()       if nw else 0.0
    gl   = abs(pnl[~wins].sum()) if nl else 0.0
    eq   = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak = np.maximum.accumulate(eq)
    mdd  = float(((eq - peak) / peak).min())
    wr   = float(nw / n); exp = wr * RR - (1 - wr)
    return {"n":n,"wr":wr,"pf":safe_pf(gw,gl),"exp_r":exp,
            "net":float(pnl.sum()),"mdd":mdd,
            "avg_win": float(pnl[wins].mean()) if nw else 0.0,
            "avg_loss":float(pnl[~wins].mean()) if nl else 0.0,
            "pnls":pnl,"equity":eq,"wins_arr":wins}

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=RAND_SEED):
    if len(pnls) < 5: return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed); pfs = []
    for _ in range(n_iter):
        s = rng.choice(pnls, len(pnls), replace=True)
        pfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
    return (float(np.percentile(pfs, 5)),
            float(np.percentile(pfs, 50)),
            float(np.percentile(pfs, 95)))

def monte_carlo(pnls, n_iter=N_MC, seed=RAND_SEED):
    if len(pnls) < 5:
        return {"prob_profit":0.0,"finals":np.array([CAPITAL]),"mdd_95":1.0,
                "p5":CAPITAL,"median":CAPITAL,"p95":CAPITAL}
    rng    = np.random.default_rng(seed)
    finals = []; mdds = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        eq = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(s)])
        pk = np.maximum.accumulate(eq)
        finals.append(float(eq[-1]))
        mdds.append(float(((eq - pk) / pk).min()))
    fa = np.array(finals)
    return {"prob_profit": float((fa > CAPITAL).mean()),
            "finals": fa, "mdd_95": float(np.percentile(mdds, 95)),
            "p5": float(np.percentile(fa, 5)),
            "median": float(np.median(fa)),
            "p95": float(np.percentile(fa, 95))}

def permutation_test(pnls, n_iter=N_PERM, seed=RAND_SEED+1):
    if len(pnls) < 10: return 1.0, 0.0
    real_pf  = safe_pf(pnls[pnls>0].sum(), abs(pnls[pnls<0].sum()))
    rng      = np.random.default_rng(seed)
    null_pfs = []
    for _ in range(n_iter):
        s = rng.permutation(pnls)
        null_pfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
    null_pfs = np.array(null_pfs)
    p_val    = float((null_pfs >= real_pf).mean())
    return p_val, float(np.percentile(null_pfs, 95))

def loo_sym(sym_trades_d):
    active = {s:tl for s,tl in sym_trades_d.items() if tl}
    if not active: return {}, 0.0
    ls = {omit: metrics([t for s,tl in active.items()
                          if s != omit for t in tl])["pf"]
          for omit in active}
    return ls, min(ls.values()) if ls else 0.0

def loo_fld(all_trades):
    folds = sorted({t["fold"] for t in all_trades})
    if not folds: return {}, 0.0
    lf = {f: metrics([t for t in all_trades if t["fold"]!=f])["pf"] for f in folds}
    return lf, min(lf.values()) if lf else 0.0

def compute_ues(pf, b50, mc_p, sf, ff, mdd, n):
    pf_pts   = min(25.0, max(0.0, (pf - 1.0) * 25.0))
    mc_pts   = min(20.0, max(0.0, mc_p * 20.0))
    boot_pts = min(15.0, max(0.0, (b50 - 1.0) / 0.5 * 15.0))
    loos_pts = min(15.0, max(0.0, (sf - 0.8)  / 0.5 * 15.0))
    loof_pts = min(10.0, max(0.0, (ff - 0.8)  / 0.5 * 10.0))
    mdd_pts  = min(10.0, max(0.0, (1.0 - abs(mdd) / 0.30) * 10.0))
    n_pts    = min(5.0,  max(0.0, (n / PROM_N) * 2.5))
    return round(pf_pts + mc_pts + boot_pts + loos_pts + loof_pts + mdd_pts + n_pts, 1)

def generalisation_score(b50, mc_p, p_val, sf, ff, pf):
    """0-100 score measuring how well the edge generalises OOS."""
    b_pts  = min(20.0, max(0.0, (b50 - 1.0) / 1.5 * 20.0))
    mc_pts = min(20.0, max(0.0, (mc_p - 0.5) / 0.5 * 20.0))
    pv_pts = min(20.0, max(0.0, (1.0 - min(p_val, 1.0)) * 20.0))
    sf_pts = min(20.0, max(0.0, (sf - 1.0) / 1.5 * 20.0))
    ff_pts = min(20.0, max(0.0, (ff - 1.0) / 1.5 * 20.0))
    return round(b_pts + mc_pts + pv_pts + sf_pts + ff_pts, 1)

# ─────────────────────────────────────────────────────────────────────────────
# INDEPENDENCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def compute_independence(ref_trades, cand_trades):
    if not ref_trades or not cand_trades:
        return {"trade_overlap":0.0,"pnl_corr":0.0,"sym_overlap":0.0,
                "session_overlap":0.0,"dd_overlap":0.0}
    ref_keys  = set((t["sym"], t["entry_ts"]) for t in ref_trades)
    cand_keys = set((t["sym"], t["entry_ts"]) for t in cand_trades)
    to        = len(ref_keys & cand_keys) / len(cand_keys) if cand_keys else 0.0

    ref_syms  = set(t["sym"] for t in ref_trades)
    cand_syms = set(t["sym"] for t in cand_trades)
    union_s   = ref_syms | cand_syms
    so        = len(ref_syms & cand_syms) / len(union_s) if union_s else 0.0

    ref_sess  = set(t.get("session","") for t in ref_trades)
    cand_sess = set(t.get("session","") for t in cand_trades)
    union_se  = ref_sess | cand_sess
    sesso     = len(ref_sess & cand_sess) / len(union_se) if union_se else 0.0

    def to_daily(trades):
        d = defaultdict(float)
        for t in trades:
            key = str(t["entry_ts"])[:10]
            d[key] += t["pnl"]
        return d

    d1 = to_daily(ref_trades); d2 = to_daily(cand_trades)
    common = sorted(set(d1) & set(d2))
    if len(common) >= 10:
        v1 = np.array([d1[d] for d in common])
        v2 = np.array([d2[d] for d in common])
        corr = float(np.corrcoef(v1,v2)[0,1]) if v1.std()>0 and v2.std()>0 else 0.0
    else:
        corr = 0.0

    return {"trade_overlap":round(to,4),"pnl_corr":round(corr,4),
            "sym_overlap":round(so,4),"session_overlap":round(sesso,4)}

# ─────────────────────────────────────────────────────────────────────────────
# WFO RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_wfo(all_dfs, cond_ids, extended=False):
    all_trades = []; sym_trades = defaultdict(list)
    fold_trades = defaultdict(list)
    for sym, (df_is, df_fwd, thr) in all_dfs.items():
        fwd_size = len(df_fwd); seg_size = max(1, fwd_size // N_FWD_FOLDS)
        for fi in range(N_FWD_FOLDS):
            seg_s  = fi * seg_size
            seg_e  = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
            df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
            if len(df_seg) < 20: continue
            em  = build_env_mask(df_seg, cond_ids, thr)
            sig = entry_signal(df_seg, em)
            fl  = f"F{fi+1}"
            if extended:
                tl = run_backtest_extended(df_seg, sig, sym, fl)
            else:
                tl = run_backtest_extended(df_seg, sig, sym, fl)
            all_trades.extend(tl)
            sym_trades[sym].extend(tl)
            fold_trades[fl].extend(tl)
    return all_trades, dict(sym_trades), dict(fold_trades)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def style_ax(ax, grid=True):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_GRID)
    if grid:
        ax.grid(True, alpha=0.25, color=C_GRID, linewidth=0.5)

def save_fig(fig, name, dpi=150):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    return p

# ─────────────────────────────────────────────────────────────────────────────
#  ══════════════════════════════════════════════
#  MAIN RESEARCH BODY
#  ══════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  QUANTLAB AI — R065")
print("  FORENSIC INVESTIGATION: RV_HI + DST_MD + ADX_WK + LON")
print(f"  Frozen environment — no optimisation — pure diagnostic")
print(SEP)
print()

t_start = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 0 — Data Load")
print(SEP)
print()

all_dfs  = {}
excluded = []
included = []

for fname in sorted(os.listdir(CACHE)):
    if not fname.endswith("_1H.parquet"): continue
    sym  = fname.replace("_1H.parquet","").replace("_","-")
    path = os.path.join(CACHE, fname)
    try:
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        n_raw = len(df)
        if n_raw < MIN_BARS:
            excluded.append((sym, n_raw, f"< {MIN_BARS} bars"))
            continue
        df = add_features(df)
        sp = int(len(df) * IS_RATIO)
        df_is  = df.iloc[:sp]
        df_fwd = df.iloc[sp:].copy().reset_index(drop=True)
        if len(df_fwd) < 50:
            excluded.append((sym, n_raw, "< 50 OOS bars"))
            continue
        thr = learn_thresholds(df_is)
        all_dfs[sym] = (df_is, df_fwd, thr)
        included.append((sym, n_raw, len(df_is), len(df_fwd)))
    except Exception as e:
        excluded.append((sym, 0, str(e)[:60]))

SYMS       = list(all_dfs.keys())
total_oos  = sum(len(v[1]) for v in all_dfs.values())
total_is   = sum(len(v[0]) for v in all_dfs.values())
oos_days   = total_oos / 24.0
oos_years  = oos_days / 365.25

print(f"  Symbols loaded          : {len(SYMS)}")
print(f"  Total IS bars           : {total_is:,}")
print(f"  Total OOS bars          : {total_oos:,}  ({oos_days:.0f} days, {oos_years:.2f} yr)")
print(f"  IS / OOS split          : {IS_RATIO:.0%} / {1-IS_RATIO:.0%}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0b — E3.1 BASELINE (reference)
# ─────────────────────────────────────────────────────────────────────────────
print("  Building E3.1 baseline ...")
e31_trades  = []; sym_e31 = defaultdict(list)
for sym, (df_is, df_fwd, thr) in all_dfs.items():
    fwd_size = len(df_fwd); seg_size = max(1, fwd_size // N_FWD_FOLDS)
    for fi in range(N_FWD_FOLDS):
        seg_s  = fi * seg_size
        seg_e  = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
        df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
        if len(df_seg) < 20: continue
        em31 = build_e31_mask(df_seg, thr)
        s31  = entry_signal(df_seg, em31)
        tl31 = run_backtest_extended(df_seg, s31, sym, f"F{fi+1}")
        e31_trades.extend(tl31); sym_e31[sym].extend(tl31)

m31     = metrics(e31_trades)
b5_31, b50_31, b95_31 = bootstrap_pf(m31["pnls"])
print(f"  E3.1 reference: PF={m31['pf']:.3f}  WR={m31['wr']:.1%}  "
      f"n={m31['n']}  MDD={m31['mdd']:.1%}  Boot50={b50_31:.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# CHAMPION RUN — full WFO
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print(f"  CHAMPION ENVIRONMENT: {'+'.join(CHAMP_CIDS)}")
print(SEP)
print()
print("  Running full 5-fold WFO across all symbols ...")

champ_trades, sym_champ, fold_champ = run_wfo(all_dfs, CHAMP_CIDS, extended=True)
m_all = metrics(champ_trades)

print(f"  Full backtest: PF={m_all['pf']:.3f}  WR={m_all['wr']:.1%}  "
      f"n={m_all['n']}  Net=${m_all['net']:,.0f}  MDD={m_all['mdd']:.1%}")
print()

if not champ_trades:
    print("  [ERROR] No trades generated. Exiting.")
    sys.exit(1)

# Convert to DataFrame for easier analysis
df_trades = pd.DataFrame(champ_trades)
df_trades["entry_ts"] = pd.to_datetime(df_trades["entry_ts"], utc=True)
try:
    df_trades["exit_ts"] = pd.to_datetime(df_trades["exit_ts"], utc=True)
except:
    pass
df_trades["month_label"] = df_trades["entry_ts"].dt.to_period("M").astype(str)
df_trades["year_label"]  = df_trades["entry_ts"].dt.year.astype(str)

pnls = m_all["pnls"]
wins = m_all["wins_arr"]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — COMPLETE TRADE PROFILE
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 1 — COMPLETE TRADE PROFILE")
print(SEP)
print()

n_total  = m_all["n"]
oos_months = oos_days / 30.44
trades_pm  = n_total / max(oos_months, 1)

print(f"  ┌─ FREQUENCY ─────────────────────────────────────────────────────")
print(f"  │  Total trades      : {n_total}")
print(f"  │  Trades / month    : {trades_pm:.1f}")
print(f"  │  Trades / year     : {trades_pm * 12:.1f}")
print(f"  │  OOS period        : {oos_days:.0f} days ({oos_years:.2f} yr)")
print(f"  │")

# Trades per symbol
print(f"  │  Trades per symbol:")
sym_counts = df_trades.groupby("sym").size().sort_values(ascending=False)
for s, cnt in sym_counts.items():
    pct = cnt / n_total * 100
    bar = "█" * int(pct / 2)
    print(f"  │    {s:<28} {cnt:>5}  ({pct:>5.1f}%)  {bar}")
print(f"  │")

# Average hold time
avg_hold_h = df_trades["bars_held"].mean()
med_hold_h = df_trades["bars_held"].median()
print(f"  │  Average hold      : {avg_hold_h:.1f} bars ({avg_hold_h:.1f}h)")
print(f"  │  Median hold       : {med_hold_h:.0f} bars ({med_hold_h:.0f}h)")
print(f"  │")

# Session distribution
sess_dist = df_trades.groupby("session").size()
print(f"  │  Session distribution:")
for sess in ["London","US","Asia"]:
    cnt = sess_dist.get(sess, 0)
    pct = cnt / n_total * 100
    print(f"  │    {sess:<10} {cnt:>5}  ({pct:>5.1f}%)")
print(f"  │")

# Weekday distribution
print(f"  │  Weekday distribution:")
dow_dist = df_trades.groupby("dow").size()
for d in range(7):
    cnt = dow_dist.get(d, 0)
    pct = cnt / n_total * 100
    print(f"  │    {DOW_NAMES[d]:<10} {cnt:>5}  ({pct:>5.1f}%)")
print(f"  │")

# Fold distribution
print(f"  │  Fold distribution:")
fold_dist = df_trades.groupby("fold").size().sort_index()
for fl, cnt in fold_dist.items():
    pct = cnt / n_total * 100
    print(f"  │    {fl:<10} {cnt:>5}  ({pct:>5.1f}%)")

freq_ok = trades_pm >= 2.0
print(f"  │")
print(f"  │  Frequency assessment: {'✓ SUFFICIENT' if freq_ok else '⚠ LOW'} "
      f"({trades_pm:.1f} trades/mo — min 2.0 for deployment)")
print(f"  └──────────────────────────────────────────────────────────────────")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PROFIT DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 2 — PROFIT DISTRIBUTION")
print(SEP)
print()

win_pnls  = pnls[wins]
loss_pnls = pnls[~wins]
n_win     = int(wins.sum())
n_los     = int((~wins).sum())
gw        = win_pnls.sum() if n_win else 0.0
gl        = abs(loss_pnls.sum()) if n_los else 0.0

# Expectancy
expect_dollar = pnls.mean()
expect_r      = m_all["wr"] * RR - (1 - m_all["wr"])

# Top-10% concentration
top10_pct  = np.percentile(win_pnls, 90) if n_win else 0.0
top10_wins = win_pnls[win_pnls >= top10_pct].sum() if n_win else 0.0
conc_top10 = top10_wins / gw if gw > 0 else 0.0

print(f"  ┌─ DISTRIBUTION ──────────────────────────────────────────────────")
print(f"  │  Profit Factor     : {m_all['pf']:.3f}")
print(f"  │  Win Rate          : {m_all['wr']:.1%}  ({n_win}W / {n_los}L)")
print(f"  │  Expectancy (R)    : {expect_r:+.3f}R")
print(f"  │  Expectancy ($)    : ${expect_dollar:+.2f} / trade")
print(f"  │  Net Profit        : ${m_all['net']:,.2f}")
print(f"  │")
print(f"  │  Average Winner    : ${win_pnls.mean():+.2f}"  if n_win else "  │  Average Winner    : n/a")
print(f"  │  Average Loser     : ${loss_pnls.mean():+.2f}" if n_los else "  │  Average Loser     : n/a")
if n_win:
    print(f"  │  Median Winner     : ${np.median(win_pnls):+.2f}")
if n_los:
    print(f"  │  Median Loser      : ${np.median(loss_pnls):+.2f}")
if n_win:
    print(f"  │  Largest Winner    : ${win_pnls.max():+.2f}")
if n_los:
    print(f"  │  Largest Loser     : ${loss_pnls.min():+.2f}")
print(f"  │")
print(f"  │  Gross Profit      : ${gw:,.2f}")
print(f"  │  Gross Loss        : ${gl:,.2f}")
print(f"  │  Top-10% win conc  : {conc_top10:.1%}  (top 10% of winners = {conc_top10*100:.0f}% of gross profit)")
print(f"  │")

# Outlier check: how much does the largest trade matter?
if n_win:
    max_win_pct = win_pnls.max() / gw * 100
    print(f"  │  Largest win / gross profit : {max_win_pct:.1f}%")
if n_los:
    max_los_pct = abs(loss_pnls.min()) / gl * 100
    print(f"  │  Largest loss / gross loss  : {max_los_pct:.1f}%")

outlier_driven = conc_top10 > 0.70
print(f"  │")
print(f"  │  Profit source: {'⚠ OUTLIER-DRIVEN (top-10% wins >70% of gross)' if outlier_driven else '✓ DISTRIBUTED (top-10% wins ≤70% of gross)'}")
print(f"  └──────────────────────────────────────────────────────────────────")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SYMBOL DEPENDENCY
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 3 — SYMBOL DEPENDENCY")
print(SEP)
print()

loo_sym_d, sym_floor = loo_sym(sym_champ)
sym_rows = []
for sym in sorted(sym_champ.keys(), key=lambda s: -metrics(sym_champ[s])["net"]):
    tl = sym_champ[sym]
    if not tl: continue
    ms = metrics(tl)
    loo_pf = loo_sym_d.get(sym, 0.0)
    sym_rows.append({
        "sym":    sym,
        "n":      ms["n"],
        "wr":     ms["wr"],
        "pf":     ms["pf"],
        "net":    ms["net"],
        "loo_pf": loo_pf,
    })

if sym_rows:
    df_sym = pd.DataFrame(sym_rows)
    print(f"  {'Symbol':<28} {'n':>5} {'WR':>7} {'PF':>7} {'Net$':>10} {'LOO-PF':>8}  Note")
    print(f"  {'─'*28} {'─'*5} {'─'*7} {'─'*7} {'─'*10} {'─'*8}")
    for _, r in df_sym.iterrows():
        note = ("★ key" if r["pf"] > 2.5 else
                "✗ drag" if r["pf"] < 1.0 and r["n"] >= 3 else "")
        print(f"  {r['sym']:<28} {int(r['n']):>5} {r['wr']:>7.1%} {r['pf']:>7.3f} "
              f"${r['net']:>9,.0f} {r['loo_pf']:>8.3f}  {note}")
    print()
    print(f"  Symbol floor PF (worst LOO) : {sym_floor:.3f}")
    n_pos = int((df_sym["pf"] > 1.0).sum())
    n_total_sym = len(df_sym)
    print(f"  Positive symbols            : {n_pos}/{n_total_sym}")
    top_sym_net = df_sym["net"].max()
    total_net   = df_sym["net"].sum()
    top_sym_contrib = top_sym_net / total_net if total_net > 0 else 0.0
    print(f"  Max single-symbol net share : {top_sym_contrib:.1%}")
    sym_diversified = sym_floor > 1.0 and top_sym_contrib < 0.60
    print(f"  Diversification verdict     : {'✓ DIVERSIFIED' if sym_diversified else '⚠ CONCENTRATED'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TEMPORAL STABILITY
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 4 — TEMPORAL STABILITY")
print(SEP)
print()

loo_fld_d, fold_floor = loo_fld(champ_trades)

fold_rows = []
for fl in sorted(fold_champ.keys()):
    tl = fold_champ[fl]
    if not tl: continue
    mf = metrics(tl)
    # Date range for this fold
    dates = [str(t["entry_ts"])[:10] for t in tl]
    d_start = min(dates); d_end = max(dates)
    fold_rows.append({
        "fold": fl, "n": mf["n"], "wr": mf["wr"],
        "pf": mf["pf"], "net": mf["net"], "mdd": mf["mdd"],
        "start": d_start, "end": d_end,
        "loo_pf": loo_fld_d.get(fl, 0.0),
    })

if fold_rows:
    df_folds = pd.DataFrame(fold_rows)
    print(f"  {'Fold':<6} {'Start':<12} {'End':<12} {'n':>5} {'WR':>7} "
          f"{'PF':>7} {'Net$':>10} {'MDD':>7}  Regime")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*5} {'─'*7} {'─'*7} {'─'*10} {'─'*7}")
    for _, r in df_folds.iterrows():
        regime = ("✓ Win" if r["pf"] >= 1.5 else
                  "~ Edge" if r["pf"] >= 1.0 else
                  "✗ Loss")
        print(f"  {r['fold']:<6} {r['start']:<12} {r['end']:<12} {int(r['n']):>5} "
              f"{r['wr']:>7.1%} {r['pf']:>7.3f} ${r['net']:>9,.0f} "
              f"{r['mdd']:>6.1%}  {regime}")
    print()
    winning_folds = int((df_folds["pf"] >= 1.0).sum())
    total_folds   = len(df_folds)
    print(f"  Winning folds     : {winning_folds}/{total_folds}")
    print(f"  Fold floor PF     : {fold_floor:.3f}")
    pf_std = df_folds["pf"].std()
    print(f"  PF std-dev        : {pf_std:.3f}  ({'✓ STABLE' if pf_std < 0.8 else '⚠ VOLATILE'})")
    temporal_ok = fold_floor > 1.0 and winning_folds >= int(total_folds * 0.6)
    print(f"  Temporal verdict  : {'✓ STABLE' if temporal_ok else '⚠ UNSTABLE'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CONDITION ABLATION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 5 — CONDITION ABLATION")
print(SEP)
print()
print("  Removing ONE condition at a time. FROZEN — no new thresholds.")
print()

ablation_results = {}

# Baseline (all 4 conditions)
abl_base_trades, _, _ = run_wfo(all_dfs, CHAMP_CIDS)
abl_base = metrics(abl_base_trades)
b5_base, b50_base, b95_base = bootstrap_pf(abl_base["pnls"])
mc_base = monte_carlo(abl_base["pnls"])

ablation_results["FULL"] = {
    "trades": abl_base_trades,
    "n": abl_base["n"], "wr": abl_base["wr"], "pf": abl_base["pf"],
    "mdd": abl_base["mdd"], "net": abl_base["net"],
    "b50": b50_base, "mc_p": mc_base["prob_profit"]
}

print(f"  BASELINE ({'+'.join(CHAMP_CIDS)}):")
print(f"    PF={abl_base['pf']:.3f}  WR={abl_base['wr']:.1%}  n={abl_base['n']}  "
      f"MDD={abl_base['mdd']:.1%}  Boot50={b50_base:.3f}  MC%={mc_base['prob_profit']:.1%}")
print()

print(f"  {'Dropped':<12} {'n':>5} {'WR':>7} {'PF':>7} {'Boot50':>8} "
      f"{'MC%':>7} {'MDD':>7} {'PF Δ':>8}  Contribution")
print(f"  {'─'*12} {'─'*5} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*7} {'─'*8}")

for drop_cid in CHAMP_CIDS:
    remaining = tuple(c for c in CHAMP_CIDS if c != drop_cid)
    t, _, _   = run_wfo(all_dfs, remaining)
    m         = metrics(t)
    b5, b50, b95 = bootstrap_pf(m["pnls"])
    mc        = monte_carlo(m["pnls"])
    delta_pf  = m["pf"] - abl_base["pf"]
    ablation_results[f"drop_{drop_cid}"] = {
        "trades": t,
        "n": m["n"], "wr": m["wr"], "pf": m["pf"],
        "mdd": m["mdd"], "net": m["net"],
        "b50": b50, "mc_p": mc["prob_profit"],
        "delta_pf": delta_pf,
    }
    contrib = ("★ CRITICAL" if delta_pf < -0.5 else
               "↑ important" if delta_pf < -0.1 else
               "→ moderate" if delta_pf < 0.1 else
               "↓ redundant")
    print(f"  -{drop_cid:<11} {m['n']:>5} {m['wr']:>7.1%} {m['pf']:>7.3f} "
          f"{b50:>8.3f} {mc['prob_profit']:>7.1%} {m['mdd']:>7.1%} "
          f"{delta_pf:>+8.3f}  {contrib}")

print()

# Rank by contribution (most critical first = biggest PF drop when removed)
ranked_abl = sorted(CHAMP_CIDS,
                    key=lambda c: ablation_results[f"drop_{c}"]["delta_pf"])
print(f"  Condition ranking by contribution (most → least critical):")
for i, cid in enumerate(ranked_abl, 1):
    d = ablation_results[f"drop_{cid}"]["delta_pf"]
    print(f"    {i}. {cid:<12} → PF drops {d:+.3f} when removed")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — ENTRY GATE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 6 — ENTRY GATE IMPORTANCE")
print(SEP)
print()
print("  Entry gates: RELVOL>1.5  |  Bull Candle (close>open)  |  PrevClose confirm (close>prev_close)")
print()

def run_with_custom_signal(all_dfs, cond_ids, use_rv=True, use_bull=True, use_prev=True):
    """Run WFO with selectable entry gates."""
    all_trades = []; sym_trades = defaultdict(list)
    for sym, (df_is, df_fwd, thr) in all_dfs.items():
        fwd_size = len(df_fwd); seg_size = max(1, fwd_size // N_FWD_FOLDS)
        for fi in range(N_FWD_FOLDS):
            seg_s  = fi * seg_size
            seg_e  = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
            df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
            if len(df_seg) < 20: continue
            em  = build_env_mask(df_seg, cond_ids, thr)
            rv  = df_seg["rel_vol"].values
            c   = df_seg["close"].values
            o   = df_seg["open"].values
            pc  = df_seg["prev_close"].values
            ok  = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
            sig = ok.copy()
            if use_rv:   sig &= (rv > 1.5)
            if use_bull: sig &= (c > o)
            if use_prev: sig &= (c > pc)
            sig &= em
            tl = run_backtest_extended(df_seg, sig, sym, f"F{fi+1}")
            all_trades.extend(tl); sym_trades[sym].extend(tl)
    return all_trades

gate_configs = [
    ("FULL (all gates)",          True,  True,  True),
    ("- RELVOL  (no rel-vol)",    False, True,  True),
    ("- BullCandle (no c>o)",     True,  False, True),
    ("- PrevClose (no c>pc)",     True,  True,  False),
    ("- BullCandle - PrevClose",  True,  False, False),
    ("- All gates (env only)",    False, False, False),
]

gate_rows = []
for label, rv, bull, prev in gate_configs:
    t   = run_with_custom_signal(all_dfs, CHAMP_CIDS, rv, bull, prev)
    m   = metrics(t)
    b5, b50, b95 = bootstrap_pf(m["pnls"])
    gate_rows.append({
        "label": label, "n": m["n"], "wr": m["wr"],
        "pf": m["pf"], "mdd": m["mdd"], "b50": b50,
    })

print(f"  {'Configuration':<38} {'n':>6} {'WR':>7} {'PF':>7} {'Boot50':>8} {'MDD':>7}")
print(f"  {'─'*38} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*7}")
base_pf = gate_rows[0]["pf"]
for r in gate_rows:
    delta = "" if r["label"].startswith("FULL") else f" ({r['pf']-base_pf:+.3f})"
    print(f"  {r['label']:<38} {r['n']:>6} {r['wr']:>7.1%} "
          f"{r['pf']:>7.3f}{delta:<9} {r['b50']:>8.3f} {r['mdd']:>7.1%}")
print()

print(f"  Gate interpretation:")
print(f"    RELVOL filter   : drop = {gate_rows[1]['pf']-base_pf:+.3f} PF  →  "
      f"{'★ critical' if gate_rows[1]['pf'] < base_pf * 0.85 else '→ moderate'}")
print(f"    Bull Candle     : drop = {gate_rows[2]['pf']-base_pf:+.3f} PF  →  "
      f"{'★ critical' if gate_rows[2]['pf'] < base_pf * 0.85 else '→ moderate'}")
print(f"    PrevClose       : drop = {gate_rows[3]['pf']-base_pf:+.3f} PF  →  "
      f"{'★ critical' if gate_rows[3]['pf'] < base_pf * 0.85 else '→ moderate'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MARKET REGIME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 7 — MARKET REGIME ANALYSIS")
print(SEP)
print()

# Split into winning and losing folds
winning_fold_labels = set(fl for fl, tl in fold_champ.items()
                          if tl and metrics(tl)["pf"] >= 1.0)
losing_fold_labels  = set(fl for fl, tl in fold_champ.items()
                          if tl and metrics(tl)["pf"] < 1.0)

df_win_folds  = df_trades[df_trades["fold"].isin(winning_fold_labels)]
df_loss_folds = df_trades[df_trades["fold"].isin(losing_fold_labels)]

regime_features = [
    ("atr_rank",  "ATR Rank"),
    ("real_vol",  "Realised Vol"),
    ("bb_width",  "BBW"),
    ("adx",       "ADX"),
    ("ema_slope", "EMA Slope"),
    ("hurst",     "Hurst"),
    ("body_pct",  "Body %"),
    ("range_pct", "Range %"),
]

print(f"  Winning folds: {sorted(winning_fold_labels)}  "
      f"({len(df_win_folds)} trades)")
print(f"  Losing folds : {sorted(losing_fold_labels)}  "
      f"({len(df_loss_folds)} trades)")
print()
print(f"  Regime feature comparison (mean at entry):")
print(f"  {'Feature':<20} {'Win folds':>12} {'Loss folds':>12} {'Difference':>12}  Signal")
print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12}")

for col, label in regime_features:
    if col not in df_trades.columns: continue
    w_mean = df_win_folds[col].dropna().mean()   if len(df_win_folds)  else np.nan
    l_mean = df_loss_folds[col].dropna().mean()  if len(df_loss_folds) else np.nan
    diff   = w_mean - l_mean if not np.isnan(w_mean) and not np.isnan(l_mean) else np.nan
    signal = ""
    if not np.isnan(diff):
        pct = abs(diff) / (abs(l_mean) + 1e-9) * 100
        signal = f"↑ Win higher ({pct:.0f}%)" if diff > 0 else f"↓ Win lower ({pct:.0f}%)"
    w_str  = f"{w_mean:.3f}" if not np.isnan(w_mean) else "n/a"
    l_str  = f"{l_mean:.3f}" if not np.isnan(l_mean) else "n/a"
    d_str  = f"{diff:+.3f}"  if not np.isnan(diff)  else "n/a"
    print(f"  {label:<20} {w_str:>12} {l_str:>12} {d_str:>12}  {signal}")

# Win/loss regime breakdown
print()
print(f"  Win rate by session:")
for sess in ["London","US","Asia"]:
    sess_t = df_trades[df_trades["session"] == sess]
    if len(sess_t) == 0: continue
    wr_s = sess_t["win"].mean()
    pf_s = metrics(sess_t.to_dict("records"))["pf"]
    print(f"    {sess:<10}  n={len(sess_t):>5}  WR={wr_s:.1%}  PF={pf_s:.3f}")

print()
print(f"  Win rate by day-of-week:")
for d in range(7):
    dow_t = df_trades[df_trades["dow"] == d]
    if len(dow_t) == 0: continue
    wr_d = dow_t["win"].mean()
    pf_d = metrics(dow_t.to_dict("records"))["pf"]
    print(f"    {DOW_NAMES[d]:<10}  n={len(dow_t):>5}  WR={wr_d:.1%}  PF={pf_d:.3f}")

print()
print(f"  MECHANISM HYPOTHESIS:")
print(f"    RV_HI  → elevated background volatility = momentum persistence")
print(f"    DST_MD → price extended above EMA200 = trend in place")
print(f"    ADX_WK → low directional strength = choppy, mean-reverting context")
print(f"    LON    → London session entry = institutional liquidity / momentum")
print(f"    Combined: entering a trending, extended market during London when")
print(f"    volatility is elevated but the ADX reads choppy — this captures")
print(f"    early-session momentum bursts in bull-continuation environments")
print(f"    before ADX catches up to the trend.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MAE / MFE
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 8 — MAE / MFE ANALYSIS")
print(SEP)
print()

mae_arr = df_trades["mae_r"].dropna().values
mfe_arr = df_trades["mfe_r"].dropna().values
win_mae = df_trades[df_trades["win"]==1]["mae_r"].dropna().values
win_mfe = df_trades[df_trades["win"]==1]["mfe_r"].dropna().values
los_mae = df_trades[df_trades["win"]==0]["mae_r"].dropna().values
los_mfe = df_trades[df_trades["win"]==0]["mfe_r"].dropna().values

print(f"  MAE = Maximum Adverse Excursion (fraction of ATR stop)")
print(f"  MFE = Maximum Favourable Excursion (fraction of target distance)")
print()
print(f"  {'Metric':<30} {'All':>10} {'Winners':>10} {'Losers':>10}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")

def pct_str(arr, pct):
    return f"{np.percentile(arr, pct):.2f}" if len(arr) else "n/a"

for pct, label in [(25,"P25 MAE"),(50,"P50 MAE (median)"),(75,"P75 MAE"),(90,"P90 MAE")]:
    print(f"  {label:<30} {pct_str(mae_arr,pct):>10} "
          f"{pct_str(win_mae,pct):>10} {pct_str(los_mae,pct):>10}")

print()
for pct, label in [(25,"P25 MFE"),(50,"P50 MFE (median)"),(75,"P75 MFE"),(90,"P90 MFE")]:
    print(f"  {label:<30} {pct_str(mfe_arr,pct):>10} "
          f"{pct_str(win_mfe,pct):>10} {pct_str(los_mfe,pct):>10}")

print()
# RR alternative analysis
print(f"  Alternative RR diagnostics (current RR = {RR:.1f}):")
for test_rr, label in [(2.5,"RR 2.5"),(3.0,"RR 3.0")]:
    # Count trades where MFE >= test_rr (would have hit higher target)
    hit = int((mfe_arr >= test_rr).sum()) if len(mfe_arr) else 0
    pct = hit / len(mfe_arr) * 100 if len(mfe_arr) else 0
    print(f"    {label}: {hit}/{len(mfe_arr)} trades ({pct:.1f}%) had MFE ≥ {test_rr:.1f}×ATR")

# Trailing stop simulation
print()
print(f"  Trailing stop note:")
p50_win_mfe = np.median(win_mfe) if len(win_mfe) else 0
p50_win_mae = np.median(win_mae) if len(win_mae) else 0
print(f"    Median winner MFE = {p50_win_mfe:.2f}×  (target is at 1.0×)")
print(f"    Median winner MAE = {p50_win_mae:.2f}×  (stop is at 1.0×)")
if p50_win_mfe > 1.5:
    print(f"    Winners often exceed target before reversal → trail could help")
else:
    print(f"    Winners do not regularly extend beyond target → trail unlikely to help")
print(f"    RECOMMENDATION: diagnostic only. No threshold change advised.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PORTFOLIO FIT vs E3.1
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 9 — PORTFOLIO FIT vs E3.1")
print(SEP)
print()

indep = compute_independence(e31_trades, champ_trades)

print(f"  Independence Analysis (RV_HI+DST_MD+ADX_WK+LON vs E3.1):")
print(f"    Trade overlap        : {indep['trade_overlap']:.1%}")
print(f"    PnL correlation      : {indep['pnl_corr']:+.3f}")
print(f"    Symbol overlap       : {indep['sym_overlap']:.1%}")
print(f"    Session overlap      : {indep['session_overlap']:.1%}")
print()

# Combined portfolio
combined_trades = champ_trades + e31_trades
m_comb   = metrics(combined_trades)
b5c, b50c, b95c = bootstrap_pf(m_comb["pnls"])
mc_comb  = monte_carlo(m_comb["pnls"])

print(f"  Portfolio composition:")
print(f"  {'Strategy':<35} {'n':>6} {'PF':>7} {'Net$':>10} {'MDD':>7}")
print(f"  {'─'*35} {'─'*6} {'─'*7} {'─'*10} {'─'*7}")
print(f"  {'E3.1 (BBW+RV_LO+DST_NR+PRG_VH)':<35} {m31['n']:>6} {m31['pf']:>7.3f} "
      f"${m31['net']:>9,.0f} {m31['mdd']:>7.1%}")
print(f"  {'Champion (RV_HI+DST_MD+ADX_WK+LON)':<35} {m_all['n']:>6} {m_all['pf']:>7.3f} "
      f"${m_all['net']:>9,.0f} {m_all['mdd']:>7.1%}")
print(f"  {'Combined Portfolio':<35} {m_comb['n']:>6} {m_comb['pf']:>7.3f} "
      f"${m_comb['net']:>9,.0f} {m_comb['mdd']:>7.1%}")
print()
print(f"  Combined portfolio robustness:")
print(f"    Bootstrap P5/P50/P95 : {b5c:.3f} / {b50c:.3f} / {b95c:.3f}")
print(f"    MC P(profit)         : {mc_comb['prob_profit']:.1%}")
print(f"    MC MDD (P95)         : {mc_comb['mdd_95']:.1%}")
print()

# Drawdown overlap
def drawdown_periods(trades, threshold=-0.05):
    """Return set of months that had drawdown > threshold."""
    if not trades: return set()
    df_t = pd.DataFrame(trades)
    df_t["entry_ts"] = pd.to_datetime(df_t["entry_ts"], utc=True)
    df_t = df_t.sort_values("entry_ts")
    pnl  = df_t["pnl"].values
    eq   = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak = np.maximum.accumulate(eq)
    dd   = (eq[1:] - peak[1:]) / peak[1:]
    bad  = df_t["entry_ts"].dt.to_period("M").astype(str).values[dd < threshold]
    return set(bad)

dd_e31   = drawdown_periods(e31_trades)
dd_champ = drawdown_periods(champ_trades)
dd_both  = dd_e31 & dd_champ
dd_either= dd_e31 | dd_champ
dd_overlap = len(dd_both) / len(dd_either) if dd_either else 0.0

print(f"  Drawdown overlap (months both in DD>5%): {dd_overlap:.1%}")
hedges = (dd_overlap < 0.40 and abs(indep["pnl_corr"]) < 0.30)
print(f"  Portfolio verdict: {'✓ COMPLEMENTARY — low DD overlap and low correlation' if hedges else '~ DILUTIVE — overlapping drawdowns or correlated PnL'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — STRESS TEST
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 10 — STRESS TEST")
print(SEP)
print()

b5, b50, b95 = bootstrap_pf(pnls)
mc           = monte_carlo(pnls)
p_val, null95= permutation_test(pnls)
loo_sym_d2, sf = loo_sym(sym_champ)
loo_fld_d2, ff = loo_fld(champ_trades)

ues  = compute_ues(m_all["pf"], b50, mc["prob_profit"], sf, ff, m_all["mdd"], m_all["n"])
gen  = generalisation_score(b50, mc["prob_profit"], p_val, sf, ff, m_all["pf"])

print(f"  ┌─ BOOTSTRAP ({N_BOOT:,} iterations) ─────────────────────────────────")
print(f"  │  PF P5  : {b5:.3f}")
print(f"  │  PF P50 : {b50:.3f}  (median resampled PF)")
print(f"  │  PF P95 : {b95:.3f}")
robust_boot = b50 >= PROM_BOOT
print(f"  │  Result : {'✓ ROBUST' if robust_boot else '⚠ FRAGILE'}  (threshold {PROM_BOOT:.2f})")
print(f"  └──────────────────────────────────────────────────────────────────")
print()

print(f"  ┌─ MONTE CARLO ({N_MC:,} iterations) ─────────────────────────────────")
print(f"  │  P(profit)   : {mc['prob_profit']:.1%}")
print(f"  │  Equity P5   : ${mc['p5']:,.0f}")
print(f"  │  Equity med  : ${mc['median']:,.0f}")
print(f"  │  Equity P95  : ${mc['p95']:,.0f}")
print(f"  │  MDD (P95)   : {mc['mdd_95']:.1%}")
robust_mc = mc["prob_profit"] >= PROM_MC
print(f"  │  Result : {'✓ ROBUST' if robust_mc else '⚠ FRAGILE'}  (threshold {PROM_MC:.1%})")
print(f"  └──────────────────────────────────────────────────────────────────")
print()

print(f"  ┌─ PERMUTATION TEST ({N_PERM:,} null shuffles) ─────────────────────")
print(f"  │  Real PF       : {m_all['pf']:.3f}")
print(f"  │  Null P95 PF   : {null95:.3f}")
print(f"  │  p-value       : {p_val:.4f}")
perm_sig = p_val < 0.05
print(f"  │  Result : {'✓ SIGNIFICANT (p<0.05)' if perm_sig else '⚠ NOT SIGNIFICANT (p≥0.05)'}")
print(f"  └──────────────────────────────────────────────────────────────────")
print()

print(f"  ┌─ LEAVE-ONE-SYMBOL-OUT ──────────────────────────────────────────")
print(f"  │  Symbol floor PF : {sf:.3f}")
print(f"  │  Full PF         : {m_all['pf']:.3f}")
for sym, pf_loo in sorted(loo_sym_d2.items(), key=lambda x: x[1]):
    flag = " ← min" if pf_loo == sf else ""
    print(f"  │  Drop {sym:<26} → PF {pf_loo:.3f}{flag}")
robust_loo_sym = sf >= 1.0
print(f"  │  Result : {'✓ ROBUST' if robust_loo_sym else '⚠ SYMBOL-DEPENDENT'}")
print(f"  └──────────────────────────────────────────────────────────────────")
print()

print(f"  ┌─ LEAVE-ONE-FOLD-OUT ────────────────────────────────────────────")
print(f"  │  Fold floor PF : {ff:.3f}")
print(f"  │  Full PF       : {m_all['pf']:.3f}")
for fl, pf_loo in sorted(loo_fld_d2.items(), key=lambda x: x[1]):
    flag = " ← min" if pf_loo == ff else ""
    print(f"  │  Drop {fl:<5} → PF {pf_loo:.3f}{flag}")
robust_loo_fld = ff >= 1.0
print(f"  │  Result : {'✓ ROBUST' if robust_loo_fld else '⚠ FOLD-DEPENDENT'}")
print(f"  └──────────────────────────────────────────────────────────────────")
print()

# Parameter robustness: test adjacent RV threshold (p60 instead of p67)
# We freeze quantile thresholds but test a structural perturbation
print(f"  ┌─ PARAMETER ROBUSTNESS ─────────────────────────────────────────")
print(f"  │  Structural perturbation: modify the RV_HI condition threshold")
print(f"  │  (Diagnostic only — not an optimisation step)")

for alt_q, note in [(0.60,"p60 (looser)"),(0.67,"p67 (frozen baseline)"),(0.75,"p75 (tighter)")]:
    class AltEnv:
        def __init__(self, q): self.q = q
    def run_with_alt_rv(all_dfs_local, alt_q_local):
        at = []
        for sym, (df_is, df_fwd, thr) in all_dfs_local.items():
            fwd_size = len(df_fwd); seg_size = max(1, fwd_size // N_FWD_FOLDS)
            for fi in range(N_FWD_FOLDS):
                seg_s = fi * seg_size
                seg_e = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
                df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
                if len(df_seg) < 20: continue
                # Build alt threshold for RV_HI
                alt_thr = dict(thr)
                col = df_seg["real_vol_20"].dropna()
                if len(col) > 10:
                    alt_thr["RV_HI"] = float(col.quantile(alt_q_local))
                em = build_env_mask(df_seg, CHAMP_CIDS, alt_thr)
                sig = entry_signal(df_seg, em)
                tl = run_backtest_extended(df_seg, sig, sym, f"F{fi+1}")
                at.extend(tl)
        return at
    alt_t = run_with_alt_rv(all_dfs, alt_q)
    alt_m = metrics(alt_t)
    flag  = " ◄ FROZEN" if alt_q == 0.67 else ""
    print(f"  │  RV_HI threshold {note:<22}: PF={alt_m['pf']:.3f}  n={alt_m['n']}{flag}")

print(f"  │")
print(f"  │  Interpretation: if PF is stable across ±1 threshold step → robust")
print(f"  └──────────────────────────────────────────────────────────────────")
print()

print(f"  ┌─ UNIVERSAL EDGE SCORE ─────────────────────────────────────────")
print(f"  │  UES = {ues:.1f} / 100")
print(f"  │")
print(f"  │  Components:")
print(f"  │    PF component   : {min(25.0,max(0.0,(m_all['pf']-1.0)*25.0)):>6.1f} / 25")
print(f"  │    MC component   : {min(20.0,max(0.0,mc['prob_profit']*20.0)):>6.1f} / 20")
print(f"  │    Boot component : {min(15.0,max(0.0,(b50-1.0)/0.5*15.0)):>6.1f} / 15")
print(f"  │    LOO-sym        : {min(15.0,max(0.0,(sf-0.8)/0.5*15.0)):>6.1f} / 15")
print(f"  │    LOO-fold       : {min(10.0,max(0.0,(ff-0.8)/0.5*10.0)):>6.1f} / 10")
print(f"  │    MDD component  : {min(10.0,max(0.0,(1.0-abs(m_all['mdd'])/0.30)*10.0)):>6.1f} / 10")
print(f"  │    N component    : {min(5.0,max(0.0,(m_all['n']/PROM_N)*2.5)):>6.1f} /  5")
print(f"  │")
print(f"  │  UES benchmark: >80 = PROMOTE | 60-80 = WATCHLIST | <60 = REJECT")
ues_verdict = "PROMOTE" if ues >= 80 else "WATCHLIST" if ues >= 60 else "REJECT"
print(f"  │  UES verdict: {ues_verdict} ({ues:.1f})")
print(f"  └──────────────────────────────────────────────────────────────────")
print()

print(f"  ┌─ GENERALISATION SCORE ────────────────────────────────────────")
print(f"  │  Gen Score = {gen:.1f} / 100")
print(f"  │  (Measures how well edge generalises to unseen data)")
gen_verdict = ("STRONG" if gen >= 70 else "MODERATE" if gen >= 50 else "WEAK")
print(f"  │  Verdict: {gen_verdict}")
print(f"  └──────────────────────────────────────────────────────────────────")
print()

# Promotion criteria checklist
criteria = {
    "PF > 1.30"        : m_all["pf"]            > PROM_PF,
    "n ≥ 30"           : m_all["n"]              >= PROM_N,
    "Boot P50 > 1.20"  : b50                     > PROM_BOOT,
    "MC P(profit) > 80%": mc["prob_profit"]       > PROM_MC,
    "LOO-sym PF > 1.0" : sf                      > 1.0,
    "LOO-fold PF > 1.0": ff                      > 1.0,
    "MDD < 20%"        : abs(m_all["mdd"])        < PROM_MDD,
}
print(f"  Promotion checklist (7/7 = PROMOTE, 5-6 = WATCHLIST):")
score_sum = 0
for crit, passed in criteria.items():
    check = "✓" if passed else "✗"
    print(f"    {check} {crit}")
    if passed: score_sum += 1
print(f"  Score: {score_sum}/7")
overall_verdict = ("PROMOTE" if score_sum == 7 else
                   "WATCHLIST" if score_sum >= 5 and m_all["pf"] > PROM_PF else
                   "REJECT")
print(f"  Verdict: {overall_verdict}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL 10 ANSWERS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  FINAL FORENSIC ANSWERS")
print(SEP)
print()

genuine_edge = (m_all["pf"] > PROM_PF and perm_sig and b50 > PROM_BOOT
                and mc["prob_profit"] > PROM_MC)
profit_diversified = (sf > 1.0 and top_sym_contrib < 0.60)
time_robust        = (fold_floor > 1.0 and winning_folds >= int(total_folds * 0.6)) if fold_rows else False
complements_e31    = hedges
stronger_than_e31  = m_all["pf"] > m31["pf"] and b50 > b50_31

print(f"  Q1. Is this edge genuine?")
q1_ans = ("YES — edge is statistically genuine."
          if genuine_edge else
          "UNCERTAIN — one or more robustness tests failed.")
print(f"      {q1_ans}")
print(f"      PF={m_all['pf']:.3f}  Boot50={b50:.3f}  MC={mc['prob_profit']:.1%}  "
      f"p-value={p_val:.4f}  UES={ues:.1f}")
print()

print(f"  Q2. Why does it work?")
print(f"      The combination creates a specific market fingerprint:")
print(f"      • RV_HI:  elevated realized volatility → markets are moving")
print(f"      • DST_MD: price is moderately extended above EMA200 → uptrend in place")
print(f"      • ADX_WK: ADX reads low/choppy → trend is early, not exhausted")
print(f"      • LON:    London session → institutional momentum, higher volume")
print(f"      Together: an uptrending market where volatility has picked up but")
print(f"      directional strength (ADX) hasn't caught up yet — the 'early trend'")
print(f"      window. Entering during London captures the opening momentum burst.")
print()

print(f"  Q3. Why does it fail?")
worst_fold = min(fold_rows, key=lambda r: r["pf"])["fold"] if fold_rows else "?"
print(f"      Failure modes identified:")
print(f"      • Losing fold: {worst_fold} — regime shift likely (ADX caught up,")
print(f"        or volatility collapsed, or trend reversed)")
print(f"      • Small n: {m_all['n']} trades total → high variance, luck plays a role")
print(f"      • Concentration: if a few large winners drive PF, luck matters")
conc_risk = conc_top10 > 0.60
print(f"      • Outlier risk: {'HIGH' if conc_risk else 'MODERATE'} (top-10% wins = {conc_top10:.0%} of gross)")
print()

print(f"  Q4. Is profit diversified?")
q4_ans = ("YES — distributed across multiple symbols and sessions."
          if profit_diversified else
          "NO — concentrated; removing one symbol may eliminate the edge.")
print(f"      {q4_ans}")
print(f"      Symbol floor PF: {sf:.3f}  |  Top symbol net share: {top_sym_contrib:.1%}")
print()

print(f"  Q5. Is time robustness acceptable?")
q5_ans = ("YES — edge persists across multiple folds."
          if time_robust else
          "MARGINAL — not all folds are profitable.")
print(f"      {q5_ans}")
if fold_rows:
    print(f"      Winning folds: {winning_folds}/{total_folds}  |  Fold floor PF: {fold_floor:.3f}")
print()

print(f"  Q6. Does it complement E3.1?")
q6_ans = ("YES — low correlation and low drawdown overlap → genuine diversification."
          if complements_e31 else
          "PARTIALLY — some overlap in drawdown periods; not a perfect hedge.")
print(f"      {q6_ans}")
print(f"      Trade overlap: {indep['trade_overlap']:.1%}  |  PnL corr: {indep['pnl_corr']:+.3f}  "
      f"|  DD overlap: {dd_overlap:.1%}")
print()

print(f"  Q7. Would you deploy it alone?")
deploy_alone = (score_sum >= 6 and m_all["n"] >= 30)
q7_ans = ("YES — standalone deployment is viable given edge quality and trade frequency."
          if deploy_alone else
          "NOT YET — trade count too low or one promotion criterion failed.")
print(f"      {q7_ans}")
print(f"      Checklist: {score_sum}/7  |  n={m_all['n']}  |  Trades/mo: {trades_pm:.1f}")
print()

print(f"  Q8. Would you combine it with E3.1 today?")
combine_now = (complements_e31 and score_sum >= 5 and m_all["pf"] > 1.2)
q8_ans = ("YES — the combination is complementary, increasing n while preserving PF."
          if combine_now else
          "CAUTION — verify DD overlap does not compound losses before combining.")
print(f"      {q8_ans}")
print(f"      Combined PF: {m_comb['pf']:.3f}  |  Combined n: {m_comb['n']}  "
      f"|  Combined MDD: {m_comb['mdd']:.1%}")
print()

print(f"  Q9. Is it stronger than E3.1?")
q9_ans = ("YES — champion outperforms E3.1 on PF and Bootstrap P50."
          if stronger_than_e31 else
          "MARGINAL — comparable performance; neither dominates clearly.")
print(f"      {q9_ans}")
print(f"      Champion PF: {m_all['pf']:.3f}  vs  E3.1 PF: {m31['pf']:.3f}")
print(f"      Champion Boot50: {b50:.3f}  vs  E3.1 Boot50: {b50_31:.3f}")
print()

print(f"  Q10. Should this become the new production candidate?")
new_prod = (score_sum >= 7 and m_all["n"] >= PROM_N and stronger_than_e31)
q10_ans  = ("YES — PROMOTE: meets all 7 criteria, stronger than E3.1, sufficient n."
            if new_prod else
            ("WATCHLIST: meets criteria but needs more OOS data or E3.1 is still stronger."
             if score_sum >= 5 else
             "NO — insufficient edge quality for production."))
print(f"      {q10_ans}")
print(f"      UES={ues:.1f}  |  Gen={gen:.1f}  |  Score={score_sum}/7  |  "
      f"Verdict={overall_verdict}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS — Dashboard
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts ...")
print(SEP2)
print()

saved_charts = []

# CHART 1: Master Dashboard
fig = plt.figure(figsize=(22, 28))
fig.suptitle(f"QuantLab AI — R065 Forensic Investigation\n"
             f"RV_HI + DST_MD + ADX_WK + LON   |   "
             f"PF={m_all['pf']:.3f}  UES={ues:.1f}  n={m_all['n']}  "
             f"Verdict={overall_verdict}",
             fontsize=12, fontweight="bold", color=C_TEXT, y=0.995)

gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.35)

# P1: Equity curve
ax_eq = fig.add_subplot(gs[0, :2])
style_ax(ax_eq)
eq = m_all["equity"]
ax_eq.plot(eq, color=C_GREEN, lw=1.5)
ax_eq.axhline(CAPITAL, color=C_GRID, lw=0.8, ls="--")
ax_eq.fill_between(range(len(eq)), eq, CAPITAL, where=np.array(eq)>CAPITAL,
                   color=C_GREEN, alpha=0.12)
ax_eq.fill_between(range(len(eq)), eq, CAPITAL, where=np.array(eq)<CAPITAL,
                   color=C_RED, alpha=0.18)
ax_eq.set_title("Equity Curve", fontsize=9, color=C_TEXT)
ax_eq.set_ylabel("Portfolio $", color=C_TEXT, fontsize=8)

# P2: Drawdown
ax_dd = fig.add_subplot(gs[0, 2:])
style_ax(ax_dd)
eq_arr = np.array(eq)
peak   = np.maximum.accumulate(eq_arr)
dd_arr = (eq_arr - peak) / peak * 100
ax_dd.fill_between(range(len(dd_arr)), dd_arr, 0, color=C_RED, alpha=0.6)
ax_dd.axhline(0, color=C_GRID, lw=0.5)
ax_dd.set_title("Drawdown %", fontsize=9, color=C_TEXT)
ax_dd.set_ylabel("DD %", color=C_TEXT, fontsize=8)

# P3: PnL distribution
ax_pnl = fig.add_subplot(gs[1, :2])
style_ax(ax_pnl)
bins = max(10, min(40, len(pnls) // 3))
ax_pnl.hist(pnls[pnls > 0], bins=bins, color=C_GREEN, alpha=0.75, label="Winners")
ax_pnl.hist(pnls[pnls < 0], bins=bins, color=C_RED,   alpha=0.75, label="Losers")
ax_pnl.axvline(0, color=C_GOLD, lw=0.8)
ax_pnl.set_title("PnL Distribution", fontsize=9, color=C_TEXT)
ax_pnl.set_xlabel("PnL ($)", color=C_TEXT, fontsize=7)
ax_pnl.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

# P4: Monte Carlo
ax_mc = fig.add_subplot(gs[1, 2:])
style_ax(ax_mc)
bins_mc = max(5, min(50, len(mc["finals"]) // 20))
ax_mc.hist(mc["finals"], bins=bins_mc, color=C_BLUE, alpha=0.7)
ax_mc.axvline(CAPITAL, color=C_GOLD,  lw=1.5, ls="--", label="Start")
ax_mc.axvline(mc["median"], color=C_GREEN, lw=1.5, label=f"Med ${mc['median']:,.0f}")
ax_mc.set_title(f"Monte Carlo ({N_MC:,} iter)  P(profit)={mc['prob_profit']:.0%}",
                fontsize=9, color=C_TEXT)
ax_mc.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

# P5: Per-symbol PF
ax_sym = fig.add_subplot(gs[2, :2])
style_ax(ax_sym)
if sym_rows:
    df_sym_plot = pd.DataFrame(sym_rows).sort_values("pf", ascending=True).tail(15)
    cols_sym = [C_GREEN if v > 1.0 else C_RED for v in df_sym_plot["pf"]]
    bars = ax_sym.barh(df_sym_plot["sym"].str.replace("-USDT-SWAP",""),
                       df_sym_plot["pf"], color=cols_sym, edgecolor=C_BG)
    ax_sym.axvline(1.0, color=C_GOLD, lw=0.8, ls="--")
    ax_sym.set_title("Per-Symbol Profit Factor", fontsize=9, color=C_TEXT)
    ax_sym.set_xlabel("PF", color=C_TEXT, fontsize=7)

# P6: Fold PF
ax_fold = fig.add_subplot(gs[2, 2:])
style_ax(ax_fold)
if fold_rows:
    df_folds_plot = pd.DataFrame(fold_rows)
    cols_fold = [C_GREEN if v > 1.0 else C_RED for v in df_folds_plot["pf"]]
    ax_fold.bar(df_folds_plot["fold"], df_folds_plot["pf"],
                color=cols_fold, edgecolor=C_BG, width=0.6)
    ax_fold.axhline(1.0, color=C_GOLD, lw=0.8, ls="--")
    ax_fold.set_title("Profit Factor by Fold (Temporal Stability)", fontsize=9, color=C_TEXT)
    ax_fold.set_ylabel("PF", color=C_TEXT, fontsize=8)
    for bar, r in zip(ax_fold.patches, df_folds_plot.to_dict("records")):
        ax_fold.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.02,
                     f"{r['pf']:.2f}", ha="center", va="bottom",
                     fontsize=7, color=C_TEXT)

# P7: Session / DOW distribution
ax_sess = fig.add_subplot(gs[3, :2])
style_ax(ax_sess)
sess_data = {s: sess_dist.get(s, 0) for s in ["London","US","Asia"]}
ax_sess.bar(list(sess_data.keys()), list(sess_data.values()),
            color=[C_GREEN, C_GOLD, C_BLUE], edgecolor=C_BG, width=0.6)
ax_sess.set_title("Trade Count by Session", fontsize=9, color=C_TEXT)

ax_dow = fig.add_subplot(gs[3, 2:])
style_ax(ax_dow)
dow_vals = [dow_dist.get(d, 0) for d in range(7)]
ax_dow.bar(DOW_NAMES, dow_vals, color=PALETTE[:7], edgecolor=C_BG, width=0.6)
ax_dow.set_title("Trade Count by Day of Week", fontsize=9, color=C_TEXT)

# P8: MAE/MFE scatter
ax_mf = fig.add_subplot(gs[4, :2])
style_ax(ax_mf)
col_scatter = [C_GREEN if w else C_RED for w in df_trades["win"].values]
ax_mf.scatter(df_trades["mae_r"].values, df_trades["mfe_r"].values,
              c=col_scatter, alpha=0.5, s=15)
ax_mf.axhline(1.0, color=C_GOLD, lw=0.8, ls="--", label="TP @ 1.0×")
ax_mf.axvline(1.0, color=C_GOLD, lw=0.8, ls="--", label="SL @ 1.0×")
ax_mf.set_xlabel("MAE (fraction of SL)", color=C_TEXT, fontsize=7)
ax_mf.set_ylabel("MFE (fraction of TP dist)", color=C_TEXT, fontsize=7)
ax_mf.set_title("MAE vs MFE  (green=win, red=loss)", fontsize=9, color=C_TEXT)
ax_mf.legend(fontsize=6, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

# P9: Ablation bar chart
ax_abl = fig.add_subplot(gs[4, 2:])
style_ax(ax_abl)
abl_labels = ["FULL"] + [f"-{c}" for c in CHAMP_CIDS]
abl_pfs    = ([ablation_results["FULL"]["pf"]] +
              [ablation_results[f"drop_{c}"]["pf"] for c in CHAMP_CIDS])
abl_cols   = [C_GREEN if i == 0 else C_GOLD for i in range(len(abl_labels))]
ax_abl.bar(abl_labels, abl_pfs, color=abl_cols, edgecolor=C_BG, width=0.6)
ax_abl.axhline(1.0, color=C_RED, lw=0.8, ls="--")
for bar, v in zip(ax_abl.patches, abl_pfs):
    ax_abl.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=C_TEXT)
ax_abl.set_title("Condition Ablation — PF Impact", fontsize=9, color=C_TEXT)
ax_abl.set_ylabel("PF", color=C_TEXT, fontsize=8)

saved_charts.append(save_fig(fig, "r065_dashboard.png"))
print(f"  → r065_dashboard.png")

# CHART 2: Equity Curves Comparison (Champion vs E3.1 vs Combined)
fig2, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(16, 10))
fig2.suptitle("R065 — Champion vs E3.1 vs Combined Portfolio", fontsize=11,
              fontweight="bold", color=C_TEXT)

for ax in (ax_a, ax_b):
    style_ax(ax)

curves = [
    ("Champion (RV_HI+DST_MD+ADX_WK+LON)", m_all["equity"],   C_GREEN),
    (f"E3.1 ({E31_LABEL})",                  m31["equity"],     C_GOLD),
    ("Combined Portfolio",                    m_comb["equity"],  C_BLUE),
]

for label, eq, col in curves:
    eq_arr2 = np.array(eq)
    m_lbl   = metrics(champ_trades if col==C_GREEN else (e31_trades if col==C_GOLD else combined_trades))
    ax_a.plot(eq_arr2, color=col, lw=1.5,
              label=f"{label[:45]}  PF={m_lbl['pf']:.3f}  n={m_lbl['n']}")
    peak2 = np.maximum.accumulate(eq_arr2)
    dd2   = (eq_arr2 - peak2) / peak2 * 100
    ax_b.fill_between(range(len(dd2)), dd2, 0, color=col, alpha=0.35, label=label[:35])

ax_a.axhline(CAPITAL, color=C_GRID, lw=0.6, ls=":")
ax_a.set_title("Equity Curves", fontsize=10, color=C_TEXT)
ax_a.set_ylabel("Portfolio $", color=C_TEXT)
ax_a.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

ax_b.axhline(0, color=C_GRID, lw=0.5)
ax_b.set_title("Drawdown Comparison (%)", fontsize=10, color=C_TEXT)
ax_b.set_ylabel("Drawdown %", color=C_TEXT)
ax_b.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

plt.tight_layout()
saved_charts.append(save_fig(fig2, "r065_equity_curves.png"))
print(f"  → r065_equity_curves.png")

# CHART 3: Symbol Breakdown
if sym_rows:
    n_syms = min(len(sym_rows), 20)
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, max(6, n_syms * 0.35 + 2)))
    fig3.suptitle("R065 — Symbol Dependency Analysis", fontsize=11,
                  fontweight="bold", color=C_TEXT)
    df_sym3 = pd.DataFrame(sym_rows).sort_values("net", ascending=True).tail(n_syms)
    sym_labels = df_sym3["sym"].str.replace("-USDT-SWAP","")
    for ax3 in axes3: style_ax(ax3)

    # Net contribution
    cols3 = [C_GREEN if v > 0 else C_RED for v in df_sym3["net"]]
    axes3[0].barh(sym_labels, df_sym3["net"], color=cols3, edgecolor=C_BG)
    axes3[0].axvline(0, color=C_GOLD, lw=0.8, ls="--")
    axes3[0].set_title("Net Contribution ($)", fontsize=9, color=C_TEXT)

    # PF per symbol
    cols3b = [C_GREEN if v > 1.0 else C_RED for v in df_sym3["pf"]]
    axes3[1].barh(sym_labels, df_sym3["pf"], color=cols3b, edgecolor=C_BG)
    axes3[1].axvline(1.0, color=C_GOLD, lw=0.8, ls="--")
    axes3[1].set_title("Profit Factor per Symbol", fontsize=9, color=C_TEXT)

    # LOO PF
    cols3c = [C_GREEN if v > 1.0 else C_RED for v in df_sym3["loo_pf"]]
    axes3[2].barh(sym_labels, df_sym3["loo_pf"], color=cols3c, edgecolor=C_BG)
    axes3[2].axvline(1.0, color=C_GOLD, lw=0.8, ls="--")
    axes3[2].set_title(f"Leave-One-Symbol-Out PF\n(Floor={sym_floor:.3f})", fontsize=9, color=C_TEXT)

    plt.tight_layout()
    saved_charts.append(save_fig(fig3, "r065_symbol_breakdown.png"))
    print(f"  → r065_symbol_breakdown.png")

# CHART 4: Regime heatmap — winning vs losing folds
regime_feats = ["adx","bb_width","real_vol","ema_dist","hurst","body_pct","atr_rank"]
regime_labels = ["ADX","BBW","RealVol","EMA Dist","Hurst","Body%","ATR Rank"]
if len(df_win_folds) > 0 or len(df_loss_folds) > 0:
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    style_ax(ax4)
    w_means = [df_win_folds[c].dropna().mean() if c in df_win_folds.columns and len(df_win_folds) else np.nan
               for c in regime_feats]
    l_means = [df_loss_folds[c].dropna().mean() if c in df_loss_folds.columns and len(df_loss_folds) else np.nan
               for c in regime_feats]
    x4 = np.arange(len(regime_feats))
    w4 = 0.35
    # Normalise to mean=1 for each feature
    all_means = [(w + l) / 2 if not np.isnan(w) and not np.isnan(l) else 1.0
                 for w, l in zip(w_means, l_means)]
    wn = [w / a if a != 0 and not np.isnan(w) else 0.0 for w, a in zip(w_means, all_means)]
    ln = [l / a if a != 0 and not np.isnan(l) else 0.0 for l, a in zip(l_means, all_means)]
    ax4.bar(x4 - w4/2, wn, w4, label=f"Winning folds ({winning_folds})", color=C_GREEN, alpha=0.8, edgecolor=C_BG)
    ax4.bar(x4 + w4/2, ln, w4, label=f"Losing folds ({total_folds - winning_folds})", color=C_RED, alpha=0.8, edgecolor=C_BG)
    ax4.axhline(1.0, color=C_GOLD, lw=0.8, ls="--", label="Neutral")
    ax4.set_xticks(x4); ax4.set_xticklabels(regime_labels, fontsize=8)
    ax4.set_title("Regime Profile: Winning vs Losing Folds (normalised to overall mean)",
                  fontsize=9, color=C_TEXT)
    ax4.set_ylabel("Relative Level", color=C_TEXT, fontsize=8)
    ax4.legend(fontsize=8, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
    plt.tight_layout()
    saved_charts.append(save_fig(fig4, "r065_regime_heatmap.png"))
    print(f"  → r065_regime_heatmap.png")

# CHART 5: Bootstrap distribution
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 5))
fig5.suptitle("R065 — Robustness Tests", fontsize=11, fontweight="bold", color=C_TEXT)
for ax5 in axes5: style_ax(ax5)

# Bootstrap PF histogram
rng5 = np.random.default_rng(RAND_SEED)
boot_pfs = []
for _ in range(N_BOOT):
    s = rng5.choice(pnls, len(pnls), replace=True)
    boot_pfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
boot_pfs = np.array(boot_pfs)
bins_b = max(5, min(50, N_BOOT // 20))
axes5[0].hist(boot_pfs, bins=bins_b, color=C_BLUE, alpha=0.75)
axes5[0].axvline(1.0, color=C_RED, lw=1.5, ls="--", label="Break-even")
axes5[0].axvline(float(np.percentile(boot_pfs, 5)),  color=C_GOLD, lw=1.2, ls=":", label=f"P5={b5:.3f}")
axes5[0].axvline(float(np.percentile(boot_pfs, 50)), color=C_GREEN, lw=1.2, label=f"P50={b50:.3f}")
axes5[0].set_title(f"Bootstrap PF Distribution ({N_BOOT:,} iter)", fontsize=9, color=C_TEXT)
axes5[0].set_xlabel("Profit Factor", color=C_TEXT, fontsize=7)
axes5[0].legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

# Monte Carlo equity distribution
bins_mc2 = max(5, min(50, N_MC // 20))
axes5[1].hist(mc["finals"], bins=bins_mc2, color=C_PURP, alpha=0.75)
axes5[1].axvline(CAPITAL, color=C_GOLD, lw=1.5, ls="--", label=f"Start ${CAPITAL:,.0f}")
axes5[1].axvline(mc["median"], color=C_GREEN, lw=1.2, label=f"Median ${mc['median']:,.0f}")
axes5[1].set_title(f"Monte Carlo Final Equity ({N_MC:,} iter)  "
                   f"P(profit)={mc['prob_profit']:.0%}", fontsize=9, color=C_TEXT)
axes5[1].set_xlabel("Final Equity ($)", color=C_TEXT, fontsize=7)
axes5[1].legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

plt.tight_layout()
saved_charts.append(save_fig(fig5, "r065_bootstrap.png"))
print(f"  → r065_bootstrap.png")

# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
# Trade log
df_trades_out = df_trades.copy()
trade_path = os.path.join(OUT, "r065_trades.csv")
df_trades_out.to_csv(trade_path, index=False)
print(f"  → r065_trades.csv  ({len(df_trades_out)} trades)")

# Per-symbol summary
if sym_rows:
    sym_path = os.path.join(OUT, "r065_symbol_summary.csv")
    pd.DataFrame(sym_rows).to_csv(sym_path, index=False)
    print(f"  → r065_symbol_summary.csv")

# Fold summary
if fold_rows:
    fold_path = os.path.join(OUT, "r065_fold_summary.csv")
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    print(f"  → r065_fold_summary.csv")

# Ablation summary
abl_rows = []
for key, v in ablation_results.items():
    abl_rows.append({
        "config": key,
        "n": v["n"], "wr": round(v["wr"], 4), "pf": round(v["pf"], 4),
        "mdd": round(v["mdd"], 4), "b50": round(v.get("b50", 0.0), 4),
        "mc_p": round(v.get("mc_p", 0.0), 4),
        "delta_pf": round(v.get("delta_pf", 0.0), 4),
    })
abl_path = os.path.join(OUT, "r065_ablation.csv")
pd.DataFrame(abl_rows).to_csv(abl_path, index=False)
print(f"  → r065_ablation.csv")

# Journal entry
journal_path = CONFIG["JOURNAL_FILE"]
os.makedirs(os.path.dirname(journal_path), exist_ok=True)
new_file = not os.path.exists(journal_path)
import csv
with open(journal_path, "a", newline="") as jf:
    cols = ["research_id","run_date","strategy_name","symbol","n_trades",
            "profit_factor","expectancy_r","win_rate","net_profit","max_drawdown",
            "sharpe","mc_prob_profit","avg_hold_minutes","verdict"]
    writer = csv.DictWriter(jf, fieldnames=cols, extrasaction="ignore")
    if new_file: writer.writeheader()
    from datetime import datetime, timezone
    writer.writerow({
        "research_id": RESEARCH_ID,
        "run_date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "strategy_name": "+".join(CHAMP_CIDS),
        "symbol": "ALL",
        "n_trades": m_all["n"],
        "profit_factor": round(m_all["pf"], 4),
        "expectancy_r": round(expect_r, 4),
        "win_rate": round(m_all["wr"], 4),
        "net_profit": round(m_all["net"], 2),
        "max_drawdown": round(m_all["mdd"], 4),
        "sharpe": 0.0,
        "mc_prob_profit": round(mc["prob_profit"], 4),
        "avg_hold_minutes": round(df_trades["bars_held"].mean() * 60, 1),
        "verdict": overall_verdict,
    })

# Research journal markdown
md_path = os.path.join(OUT, "r065_journal.md")
elapsed = time.time() - t_start
with open(md_path, "w") as mf:
    mf.write(f"# R065 — Forensic Investigation: RV_HI + DST_MD + ADX_WK + LON\n\n")
    mf.write(f"**Date:** July 2026  \n")
    mf.write(f"**Duration:** {elapsed:.0f}s  \n")
    mf.write(f"**Symbols:** {len(SYMS)}  \n")
    mf.write(f"**OOS Bars:** {total_oos:,}  \n\n")
    mf.write(f"## Verdict: {overall_verdict}\n\n")
    mf.write(f"## Core Metrics\n")
    mf.write(f"- **PF:** {m_all['pf']:.3f}  \n")
    mf.write(f"- **WR:** {m_all['wr']:.1%}  \n")
    mf.write(f"- **n:** {m_all['n']}  \n")
    mf.write(f"- **MDD:** {m_all['mdd']:.1%}  \n")
    mf.write(f"- **Bootstrap P50:** {b50:.3f}  \n")
    mf.write(f"- **MC P(profit):** {mc['prob_profit']:.1%}  \n")
    mf.write(f"- **Permutation p-value:** {p_val:.4f}  \n")
    mf.write(f"- **UES:** {ues:.1f}  \n")
    mf.write(f"- **Generalisation Score:** {gen:.1f}  \n")
    mf.write(f"- **LOO-sym floor PF:** {sf:.3f}  \n")
    mf.write(f"- **LOO-fold floor PF:** {ff:.3f}  \n\n")
    mf.write(f"## Final Answers\n")
    mf.write(f"- **Q1 Genuine edge?** {q1_ans}\n")
    mf.write(f"- **Q2 Why works?** RV_HI+DST_MD+ADX_WK+LON = early London trend burst\n")
    mf.write(f"- **Q4 Diversified?** {q4_ans}\n")
    mf.write(f"- **Q5 Time robust?** {q5_ans}\n")
    mf.write(f"- **Q6 Complements E3.1?** {q6_ans}\n")
    mf.write(f"- **Q7 Deploy alone?** {q7_ans}\n")
    mf.write(f"- **Q8 Combine with E3.1?** {q8_ans}\n")
    mf.write(f"- **Q9 Stronger than E3.1?** {q9_ans}\n")
    mf.write(f"- **Q10 New production candidate?** {q10_ans}\n\n")
    mf.write(f"## Promotion Checklist ({score_sum}/7)\n")
    for crit, passed in criteria.items():
        mf.write(f"- {'✓' if passed else '✗'} {crit}\n")
    mf.write(f"\n## Outputs\n")
    for p in saved_charts + [trade_path, sym_path if sym_rows else "", fold_path if fold_rows else "", abl_path]:
        if p:
            mf.write(f"- `{os.path.basename(p)}`\n")

print(f"  → r065_journal.md")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print(f"  R065 COMPLETE — {elapsed:.0f}s")
print(SEP)
print()
print(f"  ╔════════════════════════════════════════════════════════════╗")
print(f"  ║  FORENSIC VERDICT: {overall_verdict:<42}║")
print(f"  ║  Environment: {'+'.join(CHAMP_CIDS):<46}║")
print(f"  ║  PF={m_all['pf']:.3f}  WR={m_all['wr']:.1%}  n={m_all['n']}  "
      f"MDD={m_all['mdd']:.1%}  UES={ues:.1f}               ║")
print(f"  ║  Checklist: {score_sum}/7  Gen={gen:.1f}  p-val={p_val:.4f}  "
      f"Boot50={b50:.3f}{'   ' if b50 < 10 else '  '}             ║")
print(f"  ╚════════════════════════════════════════════════════════════╝")
print()
print(f"  Files saved to {OUT}/:")
for p in saved_charts:
    print(f"    {os.path.basename(p)}")
print(f"    r065_trades.csv")
print(f"    r065_symbol_summary.csv")
print(f"    r065_fold_summary.csv")
print(f"    r065_ablation.csv")
print(f"    r065_journal.md")
print()
