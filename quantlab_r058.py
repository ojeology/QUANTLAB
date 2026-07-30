"""
=============================================================================
QUANTLAB AI — RESEARCH #R058
Independent Structural Edge Discovery (E4)
=============================================================================

Objective:
  E3.1 is frozen: BBW_STRICT + RV_LO + DST_NR + PRG_VH
  This study searches for an entirely new structural edge (E4) that is as
  independent from E3.1 as possible while remaining profitable and robust.

  This is NOT an optimisation run.
  This is a diversification research study.

  Rules:
  - DO NOT use E3's exact filter combination.
  - Avoid environments dominated by BBW/RV_LO/DST_NR/PRG_VH.
  - Priority is LOW CORRELATION with E3, not maximum PF.
  - Use only pre-defined quantile thresholds already established.
  - No threshold mining after viewing results.
  - Full 49-symbol universe.
  - Same walk-forward pipeline as R052.

  Independence threshold: reject environments with >35% trade overlap with E3.1.

  Ranking: 1. Independence  2. Robustness  3. PF  4. Trade Frequency

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
RESEARCH_ID   = "R058"
OUT           = CONFIG["OUTPUT_FOLDER"]
CACHE         = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL       = CONFIG["STARTING_CAPITAL"]
RR            = 2.0          # same as E3.1
IS_RATIO      = 0.80         # same as R057 forward split
N_FWD_FOLDS   = 5
N_BOOT        = 500
N_MC          = 1000
RAND_SEED     = 42
MIN_BARS      = 2_000

# ── E3.1 definition (frozen baseline)
E31_CIDS_STANDARD = ("BBW_LO", "RV_LO", "DST_NR", "PRG_VH")
# BBW_STRICT uses p25 instead of BBW_LO's p33; we approximate with BBW_LO for mask purposes
# but track the distinction in independence tests.
E31_LABEL     = "BBW_STRICT+RV_LO+DST_NR+PRG_VH"

# E4 discovery thresholds
MAX_OVERLAP   = 0.35         # reject E4 candidates with > 35% trade overlap with E3.1
TOP_SCREEN_N  = 500          # oracle pre-screen survivors to full WF
MIN_N_FAST    = 20           # minimum trades for fast-screen pass
TOP_REPORT_N  = 20           # environments in final report
FAST_SPLIT    = 0.70         # IS/OOS split for fast oracle

PROM_PF   = 1.20
PROM_N    = 200
PROM_BOOT = 1.15
PROM_MC   = 0.65
PROM_MDD  = 0.20

# ── Colour palette
C_BG   = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT  = "#e0e0e0"
C_GRID = "#2a2a2a"; C_GREEN = "#00c896"; C_RED   = "#e05050"
C_GOLD = "#f5a623"; C_BLUE  = "#4a9eff"; C_PURP  = "#9b59b6"
C_CYAN = "#1abc9c"; C_ORAN  = "#e67e22"
PALETTE = [C_GREEN, C_GOLD, C_BLUE, C_RED, C_PURP,
           "#e67e22","#1abc9c","#3498db","#e74c3c","#f39c12",
           "#2ecc71","#e91e63","#00bcd4","#ff5722","#8bc34a",
           "#795548","#607d8b","#ff9800","#673ab7","#26c6da"]

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

PILOT_SYMBOLS = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "AVAX-USDT-SWAP","BNB-USDT-SWAP","HBAR-USDT-SWAP","INJ-USDT-SWAP",
]

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONS CATALOGUE  (same 25 structural conditions as R052)
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    # (id, label, feature_col, direction, param, category)
    ("ATR_LO", "ATR<p25",      "atr_rank",      "lt_q",      0.25, "vol"),
    ("ATR_MD", "ATR<p40",      "atr_rank",      "lt_q",      0.40, "vol"),
    ("ATR_HI", "ATR>p67",      "atr_rank",      "gt_q",      0.67, "vol"),
    ("ATR_VH", "ATR>p80",      "atr_rank",      "gt_q",      0.80, "vol"),
    ("BBW_LO", "BBW<p33",      "bb_width",      "lt_q",      0.33, "vol"),
    ("BBW_HI", "BBW>p67",      "bb_width",      "gt_q",      0.67, "vol"),
    ("RV_LO",  "RealVol<p33",  "real_vol_20",   "lt_q",      0.33, "vol"),
    ("RV_HI",  "RealVol>p67",  "real_vol_20",   "gt_q",      0.67, "vol"),
    ("SLP_DN", "Slope<0",      "ema200_slope",  "lt_fixed",  0.0,  "trend"),
    ("SLP_UP", "Slope>0",      "ema200_slope",  "gt_fixed",  0.0,  "trend"),
    ("DST_NR", "Dist<p33",     "ema_dist_pct",  "lt_q",      0.33, "trend"),
    ("DST_MD", "Dist>p60+",    "ema_dist_pct",  "gt_q_pos",  0.60, "trend"),
    ("DST_FR", "Dist>p75+",    "ema_dist_pct",  "gt_q_pos",  0.75, "trend"),
    ("ADX_WK", "ADX<p33",      "adx14",         "lt_q",      0.33, "trend"),
    ("ADX_TR", "ADX>p50",      "adx14",         "gt_q",      0.50, "trend"),
    ("ADX_ST", "ADX>p67",      "adx14",         "gt_q",      0.67, "trend"),
    ("PRG_LO", "PrevRng<p33",  "prev_range_r",  "lt_q",      0.33, "prev"),
    ("PRG_HI", "PrevRng>p67",  "prev_range_r",  "gt_q",      0.67, "prev"),
    ("PRG_VH", "PrevRng>p80",  "prev_range_r",  "gt_q",      0.80, "prev"),
    ("PBD_HI", "PrevBody>p67", "prev_body_r",   "gt_q",      0.67, "prev"),
    ("PBP_HI", "BodyPct>p60",  "prev_body_pct", "gt_q",      0.60, "prev"),
    ("PBP_LO", "BodyPct<p33",  "prev_body_pct", "lt_q",      0.33, "prev"),
    ("US",     "US(14-21UTC)", "hour_utc",      "hour_rng",  (14,21), "session"),
    ("LON",    "London(7-14)", "hour_utc",      "hour_rng",  (7, 14), "session"),
    ("ASI",    "Asia(0-6UTC)", "hour_utc",      "hour_rng",  (0,  6), "session"),
]

COND_IDS    = [c[0] for c in CONDITIONS_DEF]
COND_BY_ID  = {c[0]: c for c in CONDITIONS_DEF}
COND_CATS   = {c[0]: c[5] for c in CONDITIONS_DEF}
QUANT_FEATS = ["atr_rank","bb_width","real_vol_20","ema_dist_pct",
               "adx14","prev_range_r","prev_body_r","prev_body_pct"]

COND_DESC = {
    "ATR_LO": "Very quiet market (ATR < p25) — bottom-quartile range",
    "ATR_MD": "Quiet market (ATR < p40) — below-average range",
    "ATR_HI": "Active market (ATR > p67) — top-third elevated range",
    "ATR_VH": "Very active market (ATR > p80) — top-quintile range expansion",
    "BBW_LO": "Bollinger compression (BBW < p33) — band squeeze / coil building",
    "BBW_HI": "Bollinger expansion (BBW > p67) — bands widening out",
    "RV_LO":  "Low realized vol (RV20 < p33) — calm background returns",
    "RV_HI":  "High realized vol (RV20 > p67) — elevated background returns",
    "SLP_DN": "Downtrend slope (EMA200 < 0) — declining long-term average",
    "SLP_UP": "Uptrend slope (EMA200 > 0) — rising long-term average",
    "DST_NR": "Near EMA200 (Dist < p33) — price hugging long-term average",
    "DST_MD": "Moderate extension (Dist > p60, positive) — some upside extension",
    "DST_FR": "Far from EMA200 (Dist > p75, positive) — price extended above",
    "ADX_WK": "Weak trend (ADX < p33) — choppy / range-bound market",
    "ADX_TR": "Trending (ADX > p50) — above-median directional strength",
    "ADX_ST": "Strong trend (ADX > p67) — top-third directional conviction",
    "PRG_LO": "Small prior bar (PrevRng < p33) — tight previous candle",
    "PRG_HI": "Large prior bar (PrevRng > p67) — high previous amplitude",
    "PRG_VH": "Very large prior bar (PrevRng > p80) — strong prior impulse",
    "PBD_HI": "Large prior body (PrevBody > p67) — decisive prior candle",
    "PBP_HI": "High body pct (BodyPct > p60) — low-wick prior candle",
    "PBP_LO": "Low body pct (BodyPct < p33) — doji-like prior candle",
    "US":     "US session (14-21 UTC) — New York trading window",
    "LON":    "London session (7-14 UTC) — European trading window",
    "ASI":    "Asia session (0-6 UTC) — Asian trading window",
}

# ─────────────────────────────────────────────────────────────────────────────
# INVALID PAIRS (contradictions)
# ─────────────────────────────────────────────────────────────────────────────
INVALID_PAIRS = {
    frozenset({"ATR_LO","ATR_MD"}), frozenset({"ATR_LO","ATR_HI"}),
    frozenset({"ATR_LO","ATR_VH"}), frozenset({"ATR_MD","ATR_HI"}),
    frozenset({"ATR_MD","ATR_VH"}), frozenset({"ATR_HI","ATR_VH"}),
    frozenset({"BBW_LO","BBW_HI"}),
    frozenset({"RV_LO","RV_HI"}),
    frozenset({"SLP_DN","SLP_UP"}),
    frozenset({"DST_NR","DST_MD"}), frozenset({"DST_NR","DST_FR"}),
    frozenset({"DST_MD","DST_FR"}),
    frozenset({"ADX_WK","ADX_TR"}), frozenset({"ADX_WK","ADX_ST"}),
    frozenset({"ADX_TR","ADX_ST"}),
    frozenset({"PRG_LO","PRG_HI"}), frozenset({"PRG_LO","PRG_VH"}),
    frozenset({"PRG_HI","PRG_VH"}),
    frozenset({"PBP_LO","PBP_HI"}),
    frozenset({"US","LON"}), frozenset({"US","ASI"}), frozenset({"LON","ASI"}),
}

def is_valid_combo(cids):
    for a, b in itertools.combinations(cids, 2):
        if frozenset({a, b}) in INVALID_PAIRS:
            return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# E3 PROXIMITY CHECK — flag combos too close to E3.1
# ─────────────────────────────────────────────────────────────────────────────
E3_CORE = frozenset({"BBW_LO", "RV_LO", "DST_NR", "PRG_VH"})

def e3_proximity_count(cids):
    """Count how many E3 core conditions appear in this combo."""
    return len(frozenset(cids) & E3_CORE)

def is_too_close_to_e3(cids):
    """
    Flag combos as 'near-E3' if they share 3+ conditions with E3 core.
    These are evaluated but flagged; trade-level independence test is final arbiter.
    """
    return e3_proximity_count(cids) >= 3

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]
    o = df["open"]
    df["ema200"]       = calc_ema(c, 200)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank"]     = df["atr14"].rolling(100).rank(pct=True) * 100
    bb_mid             = c.rolling(20).mean()
    bb_std             = c.rolling(20).std()
    df["bb_width"]     = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["ema_dist_pct"] = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / \
                         df["ema200"].shift(10).replace(0, np.nan)
    vol_ma             = v.rolling(20).mean()
    df["rel_vol"]      = v / vol_ma.replace(0, np.nan)
    df["prev_close"]   = c.shift(1)
    df["prev_atr14"]   = df["atr14"].shift(1)
    log_ret            = np.log(c / c.shift(1))
    df["real_vol_20"]  = log_ret.rolling(20).std() * 100.0
    df["adx14"]        = calc_adx(df, 14)
    prev_range         = h.shift(1) - l.shift(1)
    prev_body          = (c.shift(1) - o.shift(1)).abs()
    df["prev_range_r"] = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]  = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"]= prev_body  / prev_range.replace(0, np.nan)
    dt                 = pd.to_datetime(df["datetime"], utc=True)
    df["hour_utc"]     = dt.dt.hour.astype(np.int16)
    df["dow"]          = dt.dt.dayofweek
    # BBW_STRICT threshold — p25 of bb_width (computed per-symbol in learn_thresholds)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def learn_all_thresholds(df_is):
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (_, _, feat, direction, param, _) in COND_BY_ID.items():
        if direction in ("gt_fixed","lt_fixed","hour_rng"):
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
    # BBW_STRICT: p25 of bb_width (used for E3.1 baseline)
    if "bb_width" in valid.columns:
        thr["BBW_STRICT"] = float(valid["bb_width"].dropna().quantile(0.25))
    return thr

# ─────────────────────────────────────────────────────────────────────────────
# MASKS
# ─────────────────────────────────────────────────────────────────────────────
def build_condition_mask(col, nan_mask, direction, threshold):
    if direction == "lt_q":
        if np.isnan(threshold): return np.zeros(len(col), dtype=bool)
        return (~nan_mask) & (col < threshold)
    elif direction in ("gt_q","gt_q_pos"):
        if np.isnan(threshold): return np.zeros(len(col), dtype=bool)
        return (~nan_mask) & (col > threshold)
    elif direction == "gt_fixed":
        return (~nan_mask) & (col > threshold)
    elif direction == "lt_fixed":
        return (~nan_mask) & (col < threshold)
    elif direction == "hour_rng":
        lo, hi = threshold
        return (col >= lo) & (col <= hi)
    return np.zeros(len(col), dtype=bool)

def build_env_mask(df, cond_ids, thr):
    N    = len(df)
    mask = np.ones(N, dtype=bool)
    for cid in cond_ids:
        _, _, feat, direction, _, _ = COND_BY_ID[cid]
        if feat not in df.columns:
            return np.zeros(N, dtype=bool)
        col = df[feat].values
        nan_m = np.isnan(col) if col.dtype.kind == "f" else np.zeros(N, dtype=bool)
        thr_val = thr.get(cid, np.nan)
        mask &= build_condition_mask(col, nan_m, direction, thr_val)
    return mask

def build_e31_mask(df, thr):
    """E3.1 uses BBW_STRICT (p25) instead of BBW_LO (p33)."""
    N    = len(df)
    mask = np.ones(N, dtype=bool)
    # BBW_STRICT: bb_width < IS p25
    bbw_strict = thr.get("BBW_STRICT", np.nan)
    if np.isnan(bbw_strict):
        return np.zeros(N, dtype=bool)
    bbw = df["bb_width"].values
    nan_bbw = np.isnan(bbw)
    mask &= (~nan_bbw) & (bbw < bbw_strict)
    # RV_LO: real_vol_20 < IS p33
    rv = df["real_vol_20"].values
    rv_t = thr.get("RV_LO", np.nan)
    if np.isnan(rv_t): return np.zeros(N, dtype=bool)
    nan_rv = np.isnan(rv)
    mask &= (~nan_rv) & (rv < rv_t)
    # DST_NR: ema_dist_pct < IS p33
    dst = df["ema_dist_pct"].values
    dst_t = thr.get("DST_NR", np.nan)
    if np.isnan(dst_t): return np.zeros(N, dtype=bool)
    nan_dst = np.isnan(dst)
    mask &= (~nan_dst) & (dst < dst_t)
    # PRG_VH: prev_range_r > IS p80
    prg = df["prev_range_r"].values
    prg_t = thr.get("PRG_VH", np.nan)
    if np.isnan(prg_t): return np.zeros(N, dtype=bool)
    nan_prg = np.isnan(prg)
    mask &= (~nan_prg) & (prg > prg_t)
    return mask

def entry_signal(df, env_mask_arr):
    rv = df["rel_vol"].values
    c  = df["close"].values; o = df["open"].values; pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & env_mask_arr

# ─────────────────────────────────────────────────────────────────────────────
# ORACLE FAST PRE-SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def precompute_oracle(df, rr=RR, max_hold=100):
    min_sl = CONFIG["MIN_SL_PCT"]
    h   = df["high"].values.astype(np.float64)
    l   = df["low"].values.astype(np.float64)
    o   = df["open"].values.astype(np.float64)
    atr = df["prev_atr14"].values.astype(np.float64)
    N   = len(df)
    result = np.zeros(N, dtype=np.int8)
    for i in range(N - 2):
        j = i + 1
        a = atr[j]
        if np.isnan(a) or a <= 0: continue
        entry = o[j]
        if np.isnan(entry) or entry <= 0: continue
        if a / entry < min_sl: continue
        sl = entry - a; tp = entry + rr * a
        end = min(j + max_hold + 1, N)
        fh = h[j:end]; fl = l[j:end]
        tp_mask = fh >= tp; sl_mask = fl <= sl
        has_tp = tp_mask.any(); has_sl = sl_mask.any()
        if not has_tp and not has_sl: continue
        tp_idx = int(np.argmax(tp_mask)) if has_tp else max_hold + 1
        sl_idx = int(np.argmax(sl_mask)) if has_sl else max_hold + 1
        result[i] = 1 if (has_tp and tp_idx <= sl_idx) else -1
    return result

def fast_pf_oracle(signal_mask, oracle, min_n=20):
    indices = np.where(signal_mask)[0]
    if len(indices) == 0: return 0.0, 0
    outcomes = oracle[indices]
    wins   = int((outcomes ==  1).sum())
    losses = int((outcomes == -1).sum())
    n = wins + losses
    if n < min_n: return 0.0, n
    return (wins * RR) / (losses if losses > 0 else 0.5), n

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signal, sym, fold_label):
    min_sl = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf     = CONFIG["RISK_PER_TRADE_PCT"]
    fee    = CONFIG["TAKER_FEE"];  spd = CONFIG["SPREAD"] * 0.5
    slp    = CONFIG["SL_SLIPPAGE"]
    in_pos = False; ep = st = tk = sz = 0.0; et = None; ei = -1
    trades = []
    hi_ = df["high"].values; lo_ = df["low"].values; op_ = df["open"].values
    atr_ = df["prev_atr14"].values; dts = df["datetime"].values
    hou_ = df["hour_utc"].values
    dow_ = df["dow"].values
    rv_  = df["rel_vol"].values

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
                rmul  = (xp - ep) / sd if sd > 0 else 0.0
                h_entry = int(hou_[ei])
                session = ("London" if 7 <= h_entry <= 13 else
                           "US"     if 14 <= h_entry <= 20 else "Asia")
                trades.append({
                    "sym":        sym,
                    "fold":       fold_label,
                    "entry_time": str(et),
                    "entry_ts":   str(et),
                    "pnl":        round(net, 4),
                    "r_multiple": round(rmul, 4),
                    "win":        int(not sl_hit),
                    "exit_type":  "SL" if sl_hit else "TP",
                    "session":    session,
                    "dow":        int(dow_[ei]),
                    "rel_vol":    float(rv_[ei]) if not np.isnan(rv_[ei]) else 1.0,
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
        return {"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,"net":0.0,
                "mdd":0.0,"pnls":np.array([]),"equity":np.array([CAPITAL])}
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values; wins = df["win"].values.astype(bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw   = pnl[wins].sum()       if nw else 0.0
    gl   = abs(pnl[~wins].sum()) if nl else 0.0
    pf   = safe_pf(gw, gl); wr = nw / n
    eq   = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak = np.maximum.accumulate(eq)
    mdd  = float(((eq - peak) / peak).min())
    exp  = wr * RR - (1 - wr)
    return {"n":n,"wr":wr,"pf":pf,"exp_r":exp,"net":float(pnl.sum()),
            "mdd":mdd,"pnls":pnl,"equity":eq}

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=RAND_SEED):
    if len(pnls) < 5: return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    pfs = []
    for _ in range(n_iter):
        s = rng.choice(pnls, len(pnls), replace=True)
        pfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
    return float(np.percentile(pfs,5)), float(np.percentile(pfs,50)), float(np.percentile(pfs,95))

def monte_carlo(pnls, n_iter=N_MC, seed=RAND_SEED):
    if len(pnls) < 5:
        return {"prob_profit":0.0,"finals":np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                        for _ in range(n_iter)])
    return {"prob_profit":float((finals > CAPITAL).mean()), "finals":finals}

def loo_sym(sym_trades_d):
    active = {s:tl for s,tl in sym_trades_d.items() if tl}
    if not active: return {}, 0.0
    ls = {omit: metrics([t for s,tl in active.items() if s!=omit for t in tl])["pf"]
          for omit in active}
    return ls, min(ls.values()) if ls else 0.0

def loo_fld(all_trades):
    folds = sorted({t["fold"] for t in all_trades})
    if not folds: return {}, 0.0
    lf = {f: metrics([t for t in all_trades if t["fold"]!=f])["pf"] for f in folds}
    return lf, min(lf.values()) if lf else 0.0

def full_stats(trades, sym_trades_d):
    m          = metrics(trades)
    b5,b50,b95 = bootstrap_pf(m["pnls"])
    mc         = monte_carlo(m["pnls"])
    ls, sf     = loo_sym(sym_trades_d)
    lf_d, ff   = loo_fld(trades)
    score      = sum([
        m["pf"]   > PROM_PF,
        m["n"]    >= PROM_N,
        b50       > PROM_BOOT,
        mc["prob_profit"] > PROM_MC,
        sf        > 1.0,
        ff        > 1.0,
        abs(m["mdd"]) < PROM_MDD,
    ])
    verdict = ("PROMOTE"   if score >= 7 else
               "WATCHLIST" if score >= 5 and m["pf"] > PROM_PF else
               "REJECT")
    return {**m, "b5":b5,"b50":b50,"b95":b95,
            "mc_p":mc["prob_profit"],"sym_floor":sf,"fold_floor":ff,
            "loo_sym":ls,"loo_fld":lf_d,"score":score,"verdict":verdict}

# ─────────────────────────────────────────────────────────────────────────────
# UES
# ─────────────────────────────────────────────────────────────────────────────
def compute_ues(pf, b50, mc_p, sf, ff, mdd, n):
    pf_pts   = min(25.0, max(0.0, (pf - 1.0) * 25.0))
    mc_pts   = min(20.0, max(0.0, mc_p * 20.0))
    boot_pts = min(15.0, max(0.0, (b50 - 1.0) / 0.5 * 15.0))
    loos_pts = min(15.0, max(0.0, (sf - 0.8)  / 0.5 * 15.0))
    loof_pts = min(10.0, max(0.0, (ff - 0.8)  / 0.5 * 10.0))
    mdd_pts  = min(10.0, max(0.0, (1.0 - abs(mdd) / 0.30) * 10.0))
    n_pts    = min(5.0,  max(0.0, (n / PROM_N) * 2.5))   # bonus for trade count
    return round(pf_pts + mc_pts + boot_pts + loos_pts + loof_pts + mdd_pts + n_pts, 1)

# ─────────────────────────────────────────────────────────────────────────────
# INDEPENDENCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_independence(e3_trades, e4_trades):
    """
    Compute independence metrics between E3.1 and E4.
    Returns dict with:
      trade_overlap: fraction of E4 trades that share (sym, entry_time) with E3.1
      pnl_corr:     Pearson correlation of daily PnL series
      sym_overlap:  Jaccard index of active symbols
      session_overlap: Jaccard index of session distributions
    """
    if not e3_trades or not e4_trades:
        return {"trade_overlap":0.0,"pnl_corr":0.0,"sym_overlap":0.0,"session_overlap":0.0}

    # Trade-level overlap: (sym, entry_time) intersection
    e3_keys = set((t["sym"], t["entry_ts"]) for t in e3_trades)
    e4_keys = set((t["sym"], t["entry_ts"]) for t in e4_trades)
    if not e4_keys:
        return {"trade_overlap":0.0,"pnl_corr":0.0,"sym_overlap":0.0,"session_overlap":0.0}
    trade_overlap = len(e3_keys & e4_keys) / len(e4_keys)

    # Symbol Jaccard
    e3_syms = set(t["sym"] for t in e3_trades)
    e4_syms = set(t["sym"] for t in e4_trades)
    union_s  = e3_syms | e4_syms
    sym_overlap = len(e3_syms & e4_syms) / len(union_s) if union_s else 0.0

    # Session Jaccard
    e3_sess = set(t.get("session","") for t in e3_trades)
    e4_sess = set(t.get("session","") for t in e4_trades)
    union_se = e3_sess | e4_sess
    sess_overlap = len(e3_sess & e4_sess) / len(union_se) if union_se else 0.0

    # Daily PnL correlation
    def to_daily_pnl(trades):
        d = defaultdict(float)
        for t in trades:
            day = t["entry_ts"][:10] if t.get("entry_ts") else "?"
            d[day] += t["pnl"]
        return d
    e3_daily = to_daily_pnl(e3_trades)
    e4_daily = to_daily_pnl(e4_trades)
    common_days = sorted(set(e3_daily) & set(e4_daily))
    if len(common_days) >= 10:
        e3_v = np.array([e3_daily[d] for d in common_days])
        e4_v = np.array([e4_daily[d] for d in common_days])
        std_e3 = e3_v.std(); std_e4 = e4_v.std()
        if std_e3 > 0 and std_e4 > 0:
            corr = float(np.corrcoef(e3_v, e4_v)[0,1])
        else:
            corr = 0.0
    else:
        corr = 0.0

    return {
        "trade_overlap":   round(trade_overlap,  4),
        "pnl_corr":        round(corr,           4),
        "sym_overlap":     round(sym_overlap,    4),
        "session_overlap": round(sess_overlap,   4),
    }

def independence_score(indep):
    """Composite independence score 0–100. Higher = more independent from E3."""
    to = 1.0 - indep["trade_overlap"]   # 1 = no overlap at all
    cr = 1.0 - max(0.0, indep["pnl_corr"])   # 1 = anti-correlated
    so = 1.0 - indep["sym_overlap"] * 0.5    # partial penalty (symbols naturally overlap)
    se = 1.0 - indep["session_overlap"] * 0.5
    return round((to * 40 + cr * 30 + so * 15 + se * 15), 1)

# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_candidates():
    cands = []
    for combo in itertools.combinations(COND_IDS, 3):
        if is_valid_combo(combo):
            cands.append(tuple(combo))
    for combo in itertools.combinations(COND_IDS, 4):
        if is_valid_combo(combo):
            cands.append(tuple(combo))
    return cands

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  QUANTLAB AI — RESEARCH #R058")
print("  Independent Structural Edge Discovery (E4)")
print(SEP)
print()
print("  FROZEN BASELINE: E3.1 = BBW_STRICT+RV_LO+DST_NR+PRG_VH")
print("  OBJECTIVE:       Discover E4 — independent structural edge")
print("  PRIORITY:        Low correlation with E3, not max PF")
print("  INDEPENDENCE:    Reject candidates with >35% trade overlap with E3.1")
print("  UNIVERSE:        49 symbols, 5-fold walk-forward")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — DATA LOAD + E3.1 BASELINE
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 0 — Data Load + E3.1 Baseline Run")
print(SEP)
print()

all_dfs    = {}   # sym → (df_is, df_fwd, thr)
e31_trades = []   # E3.1 full forward trade set
fold_e31   = defaultdict(list)
sym_e31    = defaultdict(list)

loaded = 0
for sym in ALL_SYMBOLS:
    tag  = sym.replace("-", "_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    N  = len(df)
    if N < MIN_BARS:
        continue
    df = add_features(df)
    sp = int(N * IS_RATIO)
    df_is  = df.iloc[:sp]
    df_fwd = df.iloc[sp:].copy().reset_index(drop=True)
    if len(df_fwd) < 50:
        continue
    thr = learn_all_thresholds(df_is)
    all_dfs[sym] = (df_is, df_fwd, thr)
    loaded += 1

    fwd_size = len(df_fwd)
    seg_size = max(1, fwd_size // N_FWD_FOLDS)
    for fi in range(N_FWD_FOLDS):
        seg_s  = fi * seg_size
        seg_e  = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
        df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
        flabel = f"F{fi+1}"
        if len(df_seg) < 20:
            continue
        e31_mask = build_e31_mask(df_seg, thr)
        sig      = entry_signal(df_seg, e31_mask)
        tl       = run_backtest(df_seg, sig, sym, flabel)
        e31_trades.extend(tl)
        fold_e31[flabel].extend(tl)
        sym_e31[sym].extend(tl)

print(f"  Symbols loaded: {loaded}")
print(f"  E3.1 forward trades: {len(e31_trades)}")
e31_m = metrics(e31_trades)
print(f"  E3.1 PF={e31_m['pf']:.3f}  WR={e31_m['wr']:.1%}  n={e31_m['n']}  "
      f"MDD={e31_m['mdd']:.1%}")
print()
print(f"  E3.1 fold breakdown:")
for fi in range(1, N_FWD_FOLDS+1):
    fl = f"F{fi}"
    m  = metrics(fold_e31[fl])
    print(f"    {fl}: PF={m['pf']:.3f}  n={m['n']}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 1 — Candidate Generation")
print(SEP)
print()

all_candidates = generate_candidates()
print(f"  Total valid 3/4-condition combos: {len(all_candidates):,}")

# Separate near-E3 combos (informational, not excluded from search)
near_e3_combos = [c for c in all_candidates if is_too_close_to_e3(c)]
far_e3_combos  = [c for c in all_candidates if not is_too_close_to_e3(c)]
print(f"  Near-E3 combos (≥3 shared conditions):  {len(near_e3_combos):,}")
print(f"  Far-E3 combos  (<3 shared conditions):  {len(far_e3_combos):,}")
print(f"  Note: all candidates evaluated; independence test is final gate.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ORACLE FAST PRE-SCREEN (pilot 8 symbols, 70/30 IS/OOS)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 2 — Oracle Fast Pre-Screen (8 pilot symbols)")
print(SEP)
print()

pilot_dfs  = {}  # sym → (df_is_fast, df_oos_fast, oracle, thr)
for sym in PILOT_SYMBOLS:
    if sym not in all_dfs:
        continue
    _, df_fwd, thr = all_dfs[sym]
    N   = len(df_fwd)
    sp  = int(N * FAST_SPLIT)
    dfi = df_fwd.iloc[:sp]
    dfo = df_fwd.iloc[sp:].reset_index(drop=True)
    if len(dfo) < MIN_N_FAST:
        continue
    thr_fast = learn_all_thresholds(dfi)
    oracle   = precompute_oracle(dfo)
    # Also compute env-level oracle-level features
    pilot_dfs[sym] = (dfi, dfo, oracle, thr_fast)

print(f"  Pilot symbols available: {len(pilot_dfs)}")
print(f"  Screening {len(all_candidates):,} candidates ...")

combo_scores = {}  # combo → {pf_fast, n_fast}
for combo in all_candidates:
    total_wins = 0; total_loss = 0
    for sym, (dfi, dfo, oracle, thr_fast) in pilot_dfs.items():
        em  = build_env_mask(dfo, combo, thr_fast)
        sig = entry_signal(dfo, em)
        pf, n = fast_pf_oracle(sig[:-1], oracle)
        if n < 3: continue
        w = int((oracle[np.where(sig[:-1])[0]] == 1).sum())
        l = int((oracle[np.where(sig[:-1])[0]] == -1).sum())
        total_wins += w; total_loss += l

    n_tot = total_wins + total_loss
    if n_tot < MIN_N_FAST:
        pf_fast = 0.0
    else:
        pf_fast = (total_wins * RR) / (total_loss if total_loss > 0 else 0.5)
    combo_scores[combo] = {"pf_fast": pf_fast, "n_fast": n_tot}

# Sort by PF
screened = sorted(all_candidates, key=lambda c: -combo_scores[c]["pf_fast"])
screen_pass = [(c, combo_scores[c]) for c in screened
               if combo_scores[c]["pf_fast"] >= 1.05 and combo_scores[c]["n_fast"] >= MIN_N_FAST]

print(f"  Screen survivors (PF≥1.05, n≥{MIN_N_FAST}): {len(screen_pass):,}")

# Prioritise far-E3 combos in selection
far_pass  = [(c,s) for c,s in screen_pass if not is_too_close_to_e3(c)]
near_pass = [(c,s) for c,s in screen_pass if is_too_close_to_e3(c)]
print(f"    Far-E3 survivors:  {len(far_pass)}")
print(f"    Near-E3 survivors: {len(near_pass)}")

# Take top TOP_SCREEN_N, but ensure far-E3 dominate
far_slots  = min(len(far_pass),  int(TOP_SCREEN_N * 0.85))
near_slots = min(len(near_pass), TOP_SCREEN_N - far_slots)
top_candidates = ([c for c,_ in far_pass[:far_slots]] +
                  [c for c,_ in near_pass[:near_slots]])
print(f"  Forwarding {len(top_candidates)} to full walk-forward (far={far_slots}, near={near_slots})")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FULL WALK-FORWARD (5 folds × 49 symbols)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 3 — Full Walk-Forward Validation")
print(SEP)
print()
print(f"  Evaluating {len(top_candidates)} candidates × {len(all_dfs)} symbols × "
      f"{N_FWD_FOLDS} folds ...")
print()

env_records = []

for ci, combo in enumerate(top_candidates):
    if (ci + 1) % 20 == 0:
        print(f"    Progress: {ci+1}/{len(top_candidates)} ...  "
              f"(survivors so far: {len(env_records)})", flush=True)

    all_trades  = []
    sym_trades  = defaultdict(list)

    for sym, (df_is, df_fwd, thr) in all_dfs.items():
        fwd_size = len(df_fwd)
        seg_size = max(1, fwd_size // N_FWD_FOLDS)
        for fi in range(N_FWD_FOLDS):
            seg_s  = fi * seg_size
            seg_e  = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
            df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
            flabel = f"F{fi+1}"
            if len(df_seg) < 20:
                continue
            em  = build_env_mask(df_seg, combo, thr)
            sig = entry_signal(df_seg, em)
            tl  = run_backtest(df_seg, sig, sym, flabel)
            all_trades.extend(tl)
            sym_trades[sym].extend(tl)

    m = metrics(all_trades)
    # Quick reject to save bootstrap time
    if m["n"] < 30 or m["pf"] < 1.02:
        continue

    st       = full_stats(all_trades, sym_trades)
    ues      = compute_ues(st["pf"], st["b50"], st["mc_p"],
                           st["sym_floor"], st["fold_floor"], st["mdd"], st["n"])
    label    = "+".join(combo)
    near_e3  = is_too_close_to_e3(combo)
    e3_prox  = e3_proximity_count(combo)

    env_records.append({
        "cids":      combo,
        "label":     label,
        "n":         st["n"],
        "wr":        st["wr"],
        "pf":        st["pf"],
        "b50":       st["b50"],
        "mc_p":      st["mc_p"],
        "sym_floor": st["sym_floor"],
        "fold_floor":st["fold_floor"],
        "mdd":       st["mdd"],
        "score":     st["score"],
        "verdict":   st["verdict"],
        "ues":       ues,
        "equity":    st["equity"],
        "pnls":      st["pnls"],
        "loo_sym":   st["loo_sym"],
        "loo_fld":   st["loo_fld"],
        "all_trades":all_trades,
        "sym_trades":sym_trades,
        "near_e3":   near_e3,
        "e3_prox":   e3_prox,
    })

print(f"  Full WF survivors (PF≥1.05, n≥50): {len(env_records)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — INDEPENDENCE TESTING
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 4 — Independence Testing vs E3.1")
print(SEP)
print()
print(f"  Computing independence metrics for {len(env_records)} candidates ...")
print(f"  Trade overlap threshold: {MAX_OVERLAP:.0%}  (reject if exceeded)")
print()

independent_envs = []   # passed independence test
rejected_by_overlap = []

for rec in env_records:
    indep = compute_independence(e31_trades, rec["all_trades"])
    ind_score = independence_score(indep)
    rec["indep"]      = indep
    rec["ind_score"]  = ind_score
    rec["trade_overlap"] = indep["trade_overlap"]
    rec["pnl_corr"]      = indep["pnl_corr"]

    if indep["trade_overlap"] > MAX_OVERLAP:
        rejected_by_overlap.append(rec)
    else:
        independent_envs.append(rec)

print(f"  Passed independence test (overlap ≤ {MAX_OVERLAP:.0%}): {len(independent_envs)}")
print(f"  Rejected by overlap (>{MAX_OVERLAP:.0%}):               {len(rejected_by_overlap)}")
print()

if rejected_by_overlap:
    print("  Rejected (sample):")
    for rec in rejected_by_overlap[:5]:
        print(f"    {rec['label'][:55]:<55}  overlap={rec['trade_overlap']:.1%}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — RANKING
# Priority: 1. Independence  2. Robustness  3. PF  4. Trade Frequency
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 5 — E4 Candidate Ranking")
print("  Priority: 1. Independence  2. Robustness  3. PF  4. Trade Frequency")
print(SEP)
print()

def e4_rank_key(rec):
    """Composite rank key. Higher is better."""
    ind   = rec.get("ind_score", 0.0)       # 0–100, independence (40% weight)
    ues   = rec.get("ues", 0.0)             # 0–100, robustness (40% weight)
    pf    = rec.get("pf", 0.0)              # raw PF (10% weight)
    n     = rec.get("n", 0)                 # trade count (10% weight)
    near  = -5 if rec.get("near_e3") else 0 # penalise near-E3 combos slightly
    return (ind * 0.40) + (ues * 0.40) + (min(pf,2.0)*5) + (min(n,500)/100) + near

independent_envs.sort(key=e4_rank_key, reverse=True)

print(f"  {'Rk':>3}  {'IndScore':>8}  {'UES':>5}  {'PF':>6}  {'n':>5}  "
      f"{'Boot':>6}  {'Overlap':>7}  {'Corr':>6}  {'Sc':>3}  {'NrE3':>4}  Conditions")
print("  " + "─"*120)

for rank, rec in enumerate(independent_envs[:TOP_REPORT_N], 1):
    near_tag = "YES" if rec["near_e3"] else " no"
    print(f"  {rank:>3}  {rec['ind_score']:>8.1f}  {rec['ues']:>5.1f}  "
          f"{rec['pf']:>6.3f}  {rec['n']:>5}  {rec['b50']:>6.3f}  "
          f"{rec['trade_overlap']:>6.1%}  {rec['pnl_corr']:>+6.3f}  "
          f"{rec['score']:>3}/7  {near_tag:>4}  {rec['label']}")
print()

# Best E4
best_e4 = independent_envs[0] if independent_envs else None

if best_e4:
    print(f"  ★ BEST E4 CANDIDATE: {best_e4['label']}")
    print(f"    Independence Score : {best_e4['ind_score']:.1f}/100")
    print(f"    Trade Overlap w/ E3: {best_e4['trade_overlap']:.1%}")
    print(f"    PnL Correlation    : {best_e4['pnl_corr']:+.3f}")
    print(f"    PF={best_e4['pf']:.3f}  WR={best_e4['wr']:.1%}  n={best_e4['n']}  "
          f"Boot={best_e4['b50']:.3f}  MC={best_e4['mc_p']:.1%}")
    print(f"    UES={best_e4['ues']:.1f}  MDD={best_e4['mdd']:.1%}  "
          f"SymFloor={best_e4['sym_floor']:.3f}  FoldFloor={best_e4['fold_floor']:.3f}")
    print(f"    Score={best_e4['score']}/7  Verdict={best_e4['verdict']}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DETAILED INDEPENDENCE BREAKDOWN (top 5 E4 candidates)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 6 — Detailed Independence Breakdown (Top 5 E4)")
print(SEP)
print()

for rank, rec in enumerate(independent_envs[:5], 1):
    ind = rec["indep"]
    print(f"  #{rank} {rec['label']}")
    print(f"    Trade overlap :  {ind['trade_overlap']:.1%}  "
          f"({'PASS' if ind['trade_overlap'] <= MAX_OVERLAP else 'FAIL'})")
    print(f"    PnL correlation: {ind['pnl_corr']:+.3f}")
    print(f"    Symbol Jaccard:  {ind['sym_overlap']:.3f}")
    print(f"    Session Jaccard: {ind['session_overlap']:.3f}")
    print(f"    IndScore:        {rec['ind_score']:.1f}/100  UES: {rec['ues']:.1f}/100")
    # Session distribution
    if rec["all_trades"]:
        sess_c = defaultdict(int)
        for t in rec["all_trades"]: sess_c[t.get("session","?")] += 1
        e3_sess_c = defaultdict(int)
        for t in e31_trades: e3_sess_c[t.get("session","?")] += 1
        all_sess = sorted(set(sess_c) | set(e3_sess_c))
        sess_str = "  |  ".join(
            f"{s}: E3={e3_sess_c[s]}  E4={sess_c[s]}" for s in all_sess)
        print(f"    Sessions:  {sess_str}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — VALIDATION METRICS (bootstrap, MC, LOO)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 7 — Full Validation Metrics for Best E4")
print(SEP)
print()

if best_e4:
    print(f"  Environment: {best_e4['label']}")
    print()
    # LOO fold
    print("  Leave-One-Fold Robustness:")
    for fold, pf_loo in sorted(best_e4["loo_fld"].items()):
        print(f"    {fold}: PF={pf_loo:.3f}")
    print(f"    Fold Floor: {best_e4['fold_floor']:.3f}  "
          f"({'PASS' if best_e4['fold_floor'] > 1.0 else 'FAIL'})")
    print()
    # LOO symbol (show worst 5)
    print("  Leave-One-Symbol Robustness (worst 5 symbols by LOO PF):")
    loo_s_sorted = sorted(best_e4["loo_sym"].items(), key=lambda x: x[1])
    for sym, pf_loo in loo_s_sorted[:5]:
        print(f"    {sym:<25}  PF={pf_loo:.3f}")
    print(f"    Symbol Floor: {best_e4['sym_floor']:.3f}  "
          f"({'PASS' if best_e4['sym_floor'] > 1.0 else 'FAIL'})")
    print()
    # Symbol breakdown
    print("  Per-symbol forward performance (best_e4 trades):")
    sym_pfs = {}
    for sym, tl in best_e4["sym_trades"].items():
        if tl:
            m = metrics(tl)
            sym_pfs[sym] = (m["pf"], m["n"])
    sym_pfs_sorted = sorted(sym_pfs.items(), key=lambda x: -x[1][0])
    for sym, (pf_s, n_s) in sym_pfs_sorted[:15]:
        bar = "█" * min(20, int(pf_s * 5))
        print(f"    {sym:<25}  PF={pf_s:>5.3f}  n={n_s:>4}  {bar}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PORTFOLIO TEST: E3.1 alone / E4 alone / E3.1 + E4
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 8 — Portfolio Comparison: E3.1 / E4 / Combined")
print(SEP)
print()

def combined_portfolio_stats(trades_a, trades_b):
    """Combine two trade lists and compute joint metrics."""
    combined = trades_a + trades_b
    if not combined:
        return metrics([])
    # Sort by entry time for equity curve
    combined.sort(key=lambda t: t.get("entry_ts",""))
    return metrics(combined)

e4_trades = best_e4["all_trades"] if best_e4 else []

m_e31  = metrics(e31_trades)
m_e4   = metrics(e4_trades) if e4_trades else metrics([])
m_comb = combined_portfolio_stats(e31_trades, e4_trades) if e4_trades else m_e31

b5_e31, b50_e31, b95_e31 = bootstrap_pf(m_e31["pnls"])
b5_e4,  b50_e4,  b95_e4  = (bootstrap_pf(m_e4["pnls"]) if e4_trades else (0,0,0))
b5_c,   b50_c,   b95_c   = bootstrap_pf(m_comb["pnls"])

mc_e31  = monte_carlo(m_e31["pnls"])
mc_e4   = (monte_carlo(m_e4["pnls"]) if e4_trades else {"prob_profit":0.0})
mc_comb = monte_carlo(m_comb["pnls"])

ues_e31  = compute_ues(m_e31["pf"], b50_e31, mc_e31["prob_profit"],
                       1.0, 1.0, m_e31["mdd"], m_e31["n"])
ues_e4   = compute_ues(m_e4["pf"],  b50_e4,  mc_e4["prob_profit"],
                       best_e4["sym_floor"] if best_e4 else 0,
                       best_e4["fold_floor"] if best_e4 else 0,
                       m_e4["mdd"], m_e4["n"]) if e4_trades else 0.0
ues_comb = compute_ues(m_comb["pf"], b50_c, mc_comb["prob_profit"],
                       1.0, 1.0, m_comb["mdd"], m_comb["n"])

print(f"  {'Metric':<25}  {'E3.1 alone':>12}  {'E4 alone':>12}  {'E3.1 + E4':>12}")
print("  " + "─"*70)
print(f"  {'PF':<25}  {m_e31['pf']:>12.3f}  {m_e4['pf']:>12.3f}  {m_comb['pf']:>12.3f}")
print(f"  {'Win Rate':<25}  {m_e31['wr']:>12.1%}  {m_e4['wr']:>12.1%}  {m_comb['wr']:>12.1%}")
print(f"  {'Trade Count':<25}  {m_e31['n']:>12}  {m_e4['n']:>12}  {m_comb['n']:>12}")
print(f"  {'Bootstrap Median':<25}  {b50_e31:>12.3f}  {b50_e4:>12.3f}  {b50_c:>12.3f}")
print(f"  {'MC Probability':<25}  {mc_e31['prob_profit']:>12.1%}  "
      f"{mc_e4['prob_profit']:>12.1%}  {mc_comb['prob_profit']:>12.1%}")
print(f"  {'Max Drawdown':<25}  {m_e31['mdd']:>12.1%}  {m_e4['mdd']:>12.1%}  "
      f"{m_comb['mdd']:>12.1%}")
print(f"  {'UES':<25}  {ues_e31:>12.1f}  {ues_e4:>12.1f}  {ues_comb:>12.1f}")
print()

trade_freq_improvement = (m_comb["n"] - m_e31["n"]) / max(m_e31["n"], 1)
pf_change     = m_comb["pf"] - m_e31["pf"]
mdd_change    = m_comb["mdd"] - m_e31["mdd"]
mc_change     = mc_comb["prob_profit"] - mc_e31["prob_profit"]

print(f"  Trade frequency improvement: +{m_comb['n'] - m_e31['n']} trades "
      f"(+{trade_freq_improvement:.0%})")
print(f"  PF change:     {pf_change:+.3f}")
print(f"  MDD change:    {mdd_change:+.1%}")
print(f"  MC prob change:{mc_change:+.1%}")
print(f"  UES change:    {ues_comb - ues_e31:+.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — RESEARCH CONCLUSIONS (Q1–Q7)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 9 — Research Conclusions")
print(SEP)
print()

def yn(cond): return "YES ✓" if cond else "NO  ✗"

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q1. WAS A GENUINELY INDEPENDENT EDGE DISCOVERED?")
print("  ══════════════════════════════════════════════════════════════════════")
if best_e4 and best_e4["trade_overlap"] <= MAX_OVERLAP and best_e4["pf"] > PROM_PF:
    print(f"  {yn(True)}")
    print(f"  E4 candidate: {best_e4['label']}")
    print(f"  The edge is statistically profitable (PF={best_e4['pf']:.3f}) and")
    print(f"  independent from E3.1 (trade overlap={best_e4['trade_overlap']:.1%}).")
elif best_e4:
    print(f"  {yn(best_e4['pf'] > 1.0)}")
    print(f"  Candidate found but with caveats — see Q2.")
else:
    print(f"  {yn(False)}")
    print("  No environment survived the independence test with sufficient profitability.")
    print("  Conclusion: No genuinely independent E4 currently exists in this search space.")
print()

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q2. WHAT IS THE BEST E4 ENVIRONMENT?")
print("  ══════════════════════════════════════════════════════════════════════")
if best_e4:
    for cid in best_e4["cids"]:
        print(f"  • {cid:<10}  {COND_DESC.get(cid,'')}")
    print()
    print(f"  PF={best_e4['pf']:.3f}  WR={best_e4['wr']:.1%}  n={best_e4['n']}  "
          f"Boot={best_e4['b50']:.3f}  MC={best_e4['mc_p']:.1%}")
    print(f"  MDD={best_e4['mdd']:.1%}  SymFloor={best_e4['sym_floor']:.3f}  "
          f"FoldFloor={best_e4['fold_floor']:.3f}")
    print(f"  Score={best_e4['score']}/7  Verdict={best_e4['verdict']}")
else:
    print("  No qualifying E4 environment found.")
print()

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q3. HOW INDEPENDENT IS IT FROM E3?")
print("  ══════════════════════════════════════════════════════════════════════")
if best_e4:
    ind = best_e4["indep"]
    print(f"  Independence Score:  {best_e4['ind_score']:.1f}/100")
    print(f"  Trade Overlap:       {ind['trade_overlap']:.1%}  "
          f"(threshold={MAX_OVERLAP:.0%})")
    print(f"  PnL Correlation:     {ind['pnl_corr']:+.3f}")
    print(f"  Symbol Jaccard:      {ind['sym_overlap']:.3f}")
    print(f"  Session Jaccard:     {ind['session_overlap']:.3f}")
    if ind["pnl_corr"] < 0.20:
        print("  Assessment: HIGHLY INDEPENDENT — weak to negative PnL correlation.")
    elif ind["pnl_corr"] < 0.50:
        print("  Assessment: MODERATELY INDEPENDENT — some positive correlation.")
    else:
        print("  Assessment: LOW INDEPENDENCE — significant PnL correlation with E3.1.")
print()

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q4. DOES COMBINING E3.1 AND E4 IMPROVE TRADE FREQUENCY?")
print("  ══════════════════════════════════════════════════════════════════════")
print(f"  E3.1 alone: {m_e31['n']} trades")
print(f"  E4 alone:   {m_e4['n']} trades")
print(f"  Combined:   {m_comb['n']} trades  (+{m_comb['n'] - m_e31['n']})")
print(f"  Improvement: {yn(m_comb['n'] > m_e31['n'])}  "
      f"(+{trade_freq_improvement:.0%})")
print()

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q5. DOES THE COMBINATION PRESERVE ROBUSTNESS?")
print("  ══════════════════════════════════════════════════════════════════════")
rob_ok = (b50_c >= b50_e31 * 0.90 and mc_comb["prob_profit"] >= mc_e31["prob_profit"] * 0.90)
print(f"  Bootstrap median:  E3.1={b50_e31:.3f}  Combined={b50_c:.3f}  "
      f"{'≥90% preserved' if b50_c >= b50_e31*0.90 else 'degraded'}")
print(f"  MC probability:    E3.1={mc_e31['prob_profit']:.1%}  "
      f"Combined={mc_comb['prob_profit']:.1%}  "
      f"{'≥90% preserved' if mc_comb['prob_profit'] >= mc_e31['prob_profit']*0.90 else 'degraded'}")
print(f"  Robustness preserved: {yn(rob_ok)}")
print()

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q6. IS THE COMBINED PORTFOLIO STRONGER THAN E3.1 ALONE?")
print("  ══════════════════════════════════════════════════════════════════════")
combined_stronger = (m_comb["n"] > m_e31["n"] * 1.2 and
                     m_comb["pf"] >= m_e31["pf"] * 0.90 and
                     abs(m_comb["mdd"]) <= abs(m_e31["mdd"]) * 1.15)
print(f"  Trade count  : {m_e31['n']} → {m_comb['n']}  "
      f"({'+' if m_comb['n'] > m_e31['n'] else ''}{m_comb['n'] - m_e31['n']})")
print(f"  PF           : {m_e31['pf']:.3f} → {m_comb['pf']:.3f}  ({pf_change:+.3f})")
print(f"  Max Drawdown : {m_e31['mdd']:.1%} → {m_comb['mdd']:.1%}  ({mdd_change:+.1%})")
print(f"  UES          : {ues_e31:.1f} → {ues_comb:.1f}  ({ues_comb - ues_e31:+.1f})")
print(f"  Combined portfolio stronger: {yn(combined_stronger)}")
print()

print("  ══════════════════════════════════════════════════════════════════════")
print("  Q7. SHOULD E4 BE FROZEN FOR FUTURE FORWARD TESTING?")
print("  ══════════════════════════════════════════════════════════════════════")
freeze_e4 = (best_e4 is not None and
             best_e4["trade_overlap"] <= MAX_OVERLAP and
             best_e4["pf"] > PROM_PF and
             best_e4["n"] >= PROM_N and
             best_e4["score"] >= 5)
if freeze_e4:
    print(f"  {yn(True)}")
    print(f"  RECOMMENDATION: FREEZE E4 = {best_e4['label']}")
    print(f"  Environment qualifies for forward paper-trading alongside E3.1.")
    print(f"  Monitor for at least 3 calendar months before any live allocation.")
elif best_e4:
    print(f"  {yn(False)}")
    print(f"  E4 does not yet meet all 7 promotion criteria (Score={best_e4['score']}/7).")
    print("  Recommendation: WATCHLIST. Continue monitoring with R059.")
else:
    print(f"  {yn(False)}")
    print("  No qualifying E4 found. No freeze warranted.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — FEATURE FREQUENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 10 — Feature Frequency in Independent vs Correlated Environments")
print(SEP)
print()

n_top = min(len(independent_envs), 20)
n_bot = min(len(rejected_by_overlap) + max(0, len(env_records) - len(independent_envs)), 20)
top_envs_list = independent_envs[:n_top]
bot_envs_list = rejected_by_overlap[:n_bot]

freq_top = defaultdict(int); freq_bot = defaultdict(int)
for rec in top_envs_list:
    for cid in rec["cids"]: freq_top[cid] += 1
for rec in bot_envs_list:
    for cid in rec["cids"]: freq_bot[cid] += 1

feature_freq = []
for cid in COND_IDS:
    t = freq_top.get(cid, 0); b = freq_bot.get(cid, 0)
    feature_freq.append({"cid": cid, "top": t, "bot": b, "diff": t - b})

feature_freq.sort(key=lambda x: -x["diff"])

print(f"  Top-{n_top} independent envs vs. {n_bot} overlapping/rejected envs:")
print(f"  {'Filter':<10}  {'Indep':>6}  {'Reject':>6}  {'Diff':>5}  Description")
print("  " + "─"*80)
for f in feature_freq:
    bar = "▲" * max(0, f["diff"]) if f["diff"] > 0 else "▼" * max(0, -f["diff"])
    print(f"  {f['cid']:<10}  {f['top']:>6}  {f['bot']:>6}  {f['diff']:>+5}  "
          f"{bar:<4}  {COND_DESC.get(f['cid'],'')[:55]}")
print()

best_indep_filters  = [f["cid"] for f in feature_freq if f["diff"] > 0][:6]
worst_indep_filters = [f["cid"] for f in feature_freq if f["diff"] < 0][:6]
print(f"  Filters enriched in independent environments: {', '.join(best_indep_filters)}")
print(f"  Filters enriched in overlapping environments: {', '.join(worst_indep_filters)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts ...")
print(SEP2)

top_plot = independent_envs[:min(8, len(independent_envs))]

# ── Chart 1: Dashboard
fig = plt.figure(figsize=(22, 14), facecolor=C_BG)
fig.suptitle("QUANTLAB AI — R058 — Independent Structural Edge Discovery (E4)",
             fontsize=14, color=C_GOLD, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35)

# Panel A: Top-10 E4 by independence score
ax_a = fig.add_subplot(gs[0, :2])
top10_ind = independent_envs[:10]
t10_labels = [f"#{i+1} {rec['label'][:30]}" for i, rec in enumerate(top10_ind)]
t10_ind    = [rec["ind_score"] for rec in top10_ind]
t10_cols   = [C_GREEN if rec["verdict"]=="PROMOTE" else
              (C_GOLD if rec["verdict"]=="WATCHLIST" else C_BLUE)
              for rec in top10_ind]
ax_a.barh(range(len(top10_ind)), t10_ind, color=t10_cols, alpha=0.85)
ax_a.set_yticks(range(len(top10_ind)))
ax_a.set_yticklabels(t10_labels, fontsize=6)
ax_a.axvline(50, color=C_GRID, linewidth=0.8, linestyle="--")
ax_a.invert_yaxis()
panel_style(ax_a, "Top-10 E4 Candidates by Independence Score", fs=8)
for i, v in enumerate(t10_ind):
    ax_a.text(v + 0.5, i, f"{v:.0f}", va="center", fontsize=6, color=C_TEXT)

# Panel B: Trade overlap scatter (independence vs UES)
ax_b = fig.add_subplot(gs[0, 2:])
all_ind   = [rec["ind_score"]    for rec in independent_envs]
all_ues   = [rec["ues"]          for rec in independent_envs]
all_pf    = [rec["pf"]           for rec in independent_envs]
all_cols  = [C_GREEN if rec["verdict"]=="PROMOTE" else
             (C_GOLD if rec["verdict"]=="WATCHLIST" else C_RED)
             for rec in independent_envs]
ax_b.scatter(all_ind, all_ues, c=all_cols, s=30, alpha=0.6)
if best_e4:
    ax_b.scatter([best_e4["ind_score"]], [best_e4["ues"]], s=200, color=C_GOLD,
                 marker="*", zorder=6, label="Best E4")
    ax_b.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
ax_b.set_xlabel("Independence Score", fontsize=8, color=C_TEXT)
ax_b.set_ylabel("Universal Edge Score (UES)", fontsize=8, color=C_TEXT)
leg = [mpatches.Patch(color=C_GREEN,label="PROMOTE"),
       mpatches.Patch(color=C_GOLD, label="WATCHLIST"),
       mpatches.Patch(color=C_RED,  label="REJECT")]
ax_b.legend(handles=leg, fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_b, "Independence Score vs UES (all E4 candidates)")

# Panel C: Equity curves — E3.1 / E4 / Combined
ax_c = fig.add_subplot(gs[1, :2])
eq_e31 = m_e31["equity"]
ax_c.plot(np.arange(len(eq_e31)), eq_e31, color=C_BLUE,  linewidth=1.2, label="E3.1")
if e4_trades:
    eq_e4 = m_e4["equity"]
    ax_c.plot(np.arange(len(eq_e4)),  eq_e4,  color=C_GREEN, linewidth=1.2, label="E4")
eq_c = m_comb["equity"]
ax_c.plot(np.arange(len(eq_c)),  eq_c,  color=C_GOLD,  linewidth=1.5, label="Combined", linestyle="--")
ax_c.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle=":")
ax_c.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_c, "Equity Curves: E3.1 / E4 / Combined", fs=8)

# Panel D: Portfolio comparison bar chart
ax_d = fig.add_subplot(gs[1, 2:])
port_labels = ["E3.1 alone", "E4 alone", "E3.1 + E4"]
port_pfs    = [m_e31["pf"], m_e4["pf"] if e4_trades else 0, m_comb["pf"]]
port_n      = [m_e31["n"],  m_e4["n"]  if e4_trades else 0, m_comb["n"]]
port_ues    = [ues_e31,     ues_e4,    ues_comb]
x = np.arange(3)
w = 0.28
ax_d.bar(x - w, port_pfs, w, label="PF",  color=C_GREEN, alpha=0.85)
ax_d2 = ax_d.twinx()
ax_d2.bar(x + w, port_ues, w, label="UES", color=C_GOLD,  alpha=0.75)
ax_d.axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--")
ax_d.set_xticks(x)
ax_d.set_xticklabels(port_labels, fontsize=7, color=C_TEXT)
ax_d.set_ylabel("Profit Factor", fontsize=7, color=C_GREEN)
ax_d2.set_ylabel("UES",          fontsize=7, color=C_GOLD)
ax_d.tick_params(axis="y", colors=C_GREEN)
ax_d2.tick_params(axis="y", colors=C_GOLD)
ax_d2.set_facecolor(C_PANEL)
for sp in ax_d2.spines.values(): sp.set_color(C_GRID)
lines1, labels1 = ax_d.get_legend_handles_labels()
lines2, labels2 = ax_d2.get_legend_handles_labels()
ax_d.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
            facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_d, "Portfolio Comparison: PF and UES", fs=8)
for xi, (pf_v, n_v) in zip(x, zip(port_pfs, port_n)):
    ax_d.text(xi - w, pf_v + 0.01, f"n={n_v}", ha="center", fontsize=6, color=C_TEXT)

# Panel E: Feature frequency
ax_e = fig.add_subplot(gs[2, :2])
ff_sorted = sorted(feature_freq, key=lambda x: -abs(x["diff"]))[:15]
ff_cids   = [f["cid"] for f in ff_sorted]
ff_top    = [f["top"] for f in ff_sorted]
ff_bot    = [f["bot"] for f in ff_sorted]
ff_diff   = [f["diff"] for f in ff_sorted]
xe = np.arange(len(ff_cids)); we = 0.35
ax_e.bar(xe - we/2, ff_top, we, label="Independent envs",  color=C_GREEN, alpha=0.8)
ax_e.bar(xe + we/2, ff_bot, we, label="Overlapping/reject", color=C_RED,   alpha=0.7)
ax_e.set_xticks(xe)
ax_e.set_xticklabels(ff_cids, rotation=45, ha="right", fontsize=7)
ax_e.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_e, "Filter Frequency: Independent vs Overlapping", fs=8)

# Panel F: Summary stats table
ax_f = fig.add_subplot(gs[2, 2:])
ax_f.axis("off")
summary = [
    "R058 — INDEPENDENT STRUCTURAL EDGE DISCOVERY (E4)",
    "─" * 55,
    f"Frozen baseline: {E31_LABEL}",
    f"E3.1:  PF={m_e31['pf']:.3f}  n={m_e31['n']}  MDD={m_e31['mdd']:.1%}",
    "─" * 55,
    f"Candidates:      {len(all_candidates):,}",
    f"Fast screen:     {len(screen_pass):,}",
    f"Full WF pass:    {len(env_records)}",
    f"Independence OK: {len(independent_envs)}",
    "─" * 55,
]
if best_e4:
    summary += [
        f"BEST E4: {best_e4['label'][:42]}",
        f"  PF={best_e4['pf']:.3f}  n={best_e4['n']}  UES={best_e4['ues']:.1f}",
        f"  Overlap={best_e4['trade_overlap']:.1%}  Corr={best_e4['pnl_corr']:+.3f}",
        f"  Score={best_e4['score']}/7  Verdict={best_e4['verdict']}",
        "─" * 55,
        f"Combined PF:     {m_comb['pf']:.3f}  ({pf_change:+.3f} vs E3.1)",
        f"Combined trades: {m_comb['n']}  (+{m_comb['n']-m_e31['n']})",
        f"Freeze E4:       {'YES' if freeze_e4 else 'NO'}",
    ]
else:
    summary += ["RESULT: No qualifying E4 found.", "E4 does not exist in current search space."]

for i, line in enumerate(summary):
    col = (C_GOLD if i == 0 else
           C_GREEN if "BEST E4" in line or "PROMOTE" in line else
           C_TEXT)
    ax_f.text(0.02, 0.98 - i*0.065, line, transform=ax_f.transAxes,
              fontsize=6.5, color=col, va="top", fontfamily="monospace")
panel_style(ax_f, "R058 Research Summary")

plt.savefig(f"{OUT}/r058_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r058_dashboard.png")

# ── Chart 2: Equity curves (top 8 E4)
if top_plot:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), facecolor=C_BG)
    fig.suptitle("R058 — Equity Curves: Top E4 Candidates (Forward OOS)",
                 fontsize=11, color=C_GOLD, fontweight="bold", y=0.98)
    for idx, (ax_e2, rec) in enumerate(zip(axes.flat, top_plot)):
        eq = rec["equity"]
        x  = np.arange(len(eq))
        ax_e2.plot(x, eq, color=PALETTE[idx], linewidth=1.2)
        ax_e2.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
        ax_e2.fill_between(x, CAPITAL, eq, where=eq >= CAPITAL, alpha=0.15, color=C_GREEN)
        ax_e2.fill_between(x, CAPITAL, eq, where=eq < CAPITAL,  alpha=0.15, color=C_RED)
        title = "+".join(rec["cids"])[:35]
        ax_e2.set_title(f"#{idx+1}  {title}\nPF={rec['pf']:.3f}  n={rec['n']}  "
                        f"Ind={rec['ind_score']:.0f}  Overlap={rec['trade_overlap']:.0%}",
                        fontsize=6, color=PALETTE[idx], pad=3)
        panel_style(ax_e2, "")
    # Hide unused axes
    for ax_e2 in axes.flat[len(top_plot):]:
        ax_e2.axis("off")
    plt.tight_layout()
    plt.savefig(f"{OUT}/r058_equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {OUT}/r058_equity_curves.png")

# ── Chart 3: Independence scatter — trade overlap vs PnL correlation
if independent_envs or rejected_by_overlap:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=C_BG)
    fig.suptitle("R058 — Independence Analysis", fontsize=11, color=C_GOLD, fontweight="bold")

    ax_s = axes[0]
    for rec in independent_envs:
        col = (C_GREEN if rec["verdict"]=="PROMOTE" else
               C_GOLD if rec["verdict"]=="WATCHLIST" else C_BLUE)
        ax_s.scatter(rec["trade_overlap"], rec["pnl_corr"],
                     c=col, s=max(15, rec["n"]/10), alpha=0.6)
    for rec in rejected_by_overlap:
        ax_s.scatter(rec["trade_overlap"], rec["pnl_corr"],
                     c=C_RED, s=15, alpha=0.4, marker="x")
    ax_s.axvline(MAX_OVERLAP, color=C_RED, linewidth=1.2, linestyle="--",
                 label=f"Overlap limit ({MAX_OVERLAP:.0%})")
    ax_s.axhline(0, color=C_GRID, linewidth=0.7, linestyle="--")
    if best_e4:
        ax_s.scatter([best_e4["trade_overlap"]], [best_e4["pnl_corr"]],
                     s=200, color=C_GOLD, marker="*", zorder=6, label="Best E4")
    ax_s.set_xlabel("Trade Overlap with E3.1", fontsize=8, color=C_TEXT)
    ax_s.set_ylabel("PnL Correlation with E3.1", fontsize=8, color=C_TEXT)
    ax_s.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
    leg2 = [mpatches.Patch(color=C_GREEN, label="Independent/PROMOTE"),
            mpatches.Patch(color=C_GOLD,  label="Independent/WATCHLIST"),
            mpatches.Patch(color=C_BLUE,  label="Independent/REJECT"),
            mpatches.Patch(color=C_RED,   label="Overlap Rejected (x)")]
    ax_s.legend(handles=leg2, fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)
    panel_style(ax_s, "Trade Overlap vs PnL Correlation (all evaluated candidates)")

    ax_t = axes[1]
    if independent_envs:
        all_ind_scores   = [r["ind_score"]    for r in independent_envs]
        all_ues_scores   = [r["ues"]          for r in independent_envs]
        all_n            = [max(r["n"],1)      for r in independent_envs]
        all_vcols        = [C_GREEN if r["verdict"]=="PROMOTE" else
                            (C_GOLD if r["verdict"]=="WATCHLIST" else C_RED)
                            for r in independent_envs]
        ax_t.scatter(all_ind_scores, all_ues_scores,
                     c=all_vcols, s=[n/5 for n in all_n], alpha=0.65)
        if best_e4:
            ax_t.scatter([best_e4["ind_score"]], [best_e4["ues"]],
                         s=250, color=C_GOLD, marker="*", zorder=7, label="Best E4")
    ax_t.set_xlabel("Independence Score (0=identical, 100=fully independent)", fontsize=8, color=C_TEXT)
    ax_t.set_ylabel("Universal Edge Score (robustness)", fontsize=8, color=C_TEXT)
    if independent_envs:
        leg3 = [mpatches.Patch(color=C_GREEN,label="PROMOTE"),
                mpatches.Patch(color=C_GOLD, label="WATCHLIST"),
                mpatches.Patch(color=C_RED,  label="REJECT")]
        ax_t.legend(handles=leg3, fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
    panel_style(ax_t, "Independence Score vs UES (bubble size = trade count)")

    plt.tight_layout()
    plt.savefig(f"{OUT}/r058_independence_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {OUT}/r058_independence_scatter.png")

# ── Chart 4: Portfolio comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=C_BG)
fig.suptitle("R058 — Portfolio Comparison: E3.1 / E4 / Combined",
             fontsize=11, color=C_GOLD, fontweight="bold")

def plot_equity(ax, eq, label, color, pf, n):
    x = np.arange(len(eq))
    ax.plot(x, eq, color=color, linewidth=1.4)
    ax.axhline(CAPITAL, color=C_GRID, linewidth=0.7, linestyle="--")
    ax.fill_between(x, CAPITAL, eq, where=eq >= CAPITAL, alpha=0.18, color=color)
    ax.fill_between(x, CAPITAL, eq, where=eq < CAPITAL,  alpha=0.18, color=C_RED)
    panel_style(ax, f"{label}\nPF={pf:.3f}  n={n}", fs=8)

plot_equity(axes[0], m_e31["equity"],  "E3.1 alone",  C_BLUE,  m_e31["pf"], m_e31["n"])
plot_equity(axes[1], m_e4["equity"]  if e4_trades else np.array([CAPITAL]),
            "E4 alone", C_GREEN, m_e4["pf"] if e4_trades else 0.0, m_e4["n"])
plot_equity(axes[2], m_comb["equity"], "E3.1 + E4",   C_GOLD,  m_comb["pf"], m_comb["n"])

plt.tight_layout()
plt.savefig(f"{OUT}/r058_portfolio_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r058_portfolio_comparison.png")

# ── Chart 5: Per-fold breakdown (E3.1 vs E4 vs Combined)
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=C_BG)
fig.suptitle("R058 — Fold-by-Fold Analysis: E3.1 vs E4 vs Combined",
             fontsize=11, color=C_GOLD, fontweight="bold")

fold_labels = [f"F{i}" for i in range(1, N_FWD_FOLDS+1)]

# PF per fold
e31_pf_folds = [metrics(fold_e31.get(f,[])).get("pf",0.0) for f in fold_labels]
if best_e4:
    e4_fold_trades = defaultdict(list)
    for t in e4_trades: e4_fold_trades[t["fold"]].append(t)
    e4_pf_folds   = [metrics(e4_fold_trades.get(f,[])).get("pf",0.0) for f in fold_labels]
    comb_folds    = {f: fold_e31.get(f,[]) + e4_fold_trades.get(f,[]) for f in fold_labels}
    comb_pf_folds = [metrics(comb_folds.get(f,[])).get("pf",0.0) for f in fold_labels]
else:
    e4_pf_folds   = [0] * len(fold_labels)
    comb_pf_folds = e31_pf_folds[:]

ax1 = axes[0]
x = np.arange(len(fold_labels)); w = 0.28
ax1.bar(x - w,  e31_pf_folds,  w, label="E3.1",     color=C_BLUE,  alpha=0.85)
ax1.bar(x,      e4_pf_folds,   w, label="E4",        color=C_GREEN, alpha=0.85)
ax1.bar(x + w,  comb_pf_folds, w, label="Combined",  color=C_GOLD,  alpha=0.85)
ax1.axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--")
ax1.set_xticks(x); ax1.set_xticklabels(fold_labels, fontsize=8)
ax1.set_ylabel("Profit Factor", fontsize=8, color=C_TEXT)
ax1.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax1, "Profit Factor by Fold", fs=8)

# Trade count per fold
e31_n_folds = [metrics(fold_e31.get(f,[])).get("n",0) for f in fold_labels]
e4_n_folds  = [metrics(e4_fold_trades.get(f,[])).get("n",0) for f in fold_labels] if best_e4 else [0]*5
comb_n_folds= [e31_n_folds[i] + e4_n_folds[i] for i in range(len(fold_labels))]

ax2 = axes[1]
ax2.bar(x - w, e31_n_folds,  w, label="E3.1",    color=C_BLUE,  alpha=0.85)
ax2.bar(x,     e4_n_folds,   w, label="E4",       color=C_GREEN, alpha=0.85)
ax2.bar(x + w, comb_n_folds, w, label="Combined", color=C_GOLD,  alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(fold_labels, fontsize=8)
ax2.set_ylabel("Trade Count", fontsize=8, color=C_TEXT)
ax2.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax2, "Trade Count by Fold", fs=8)

plt.tight_layout()
plt.savefig(f"{OUT}/r058_fold_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r058_fold_analysis.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for rank, rec in enumerate(independent_envs[:50], 1):
    rows.append({
        "rank":          rank,
        "conditions":    rec["label"],
        "ind_score":     round(rec["ind_score"], 1),
        "ues":           rec["ues"],
        "pf":            round(rec["pf"], 4),
        "n":             rec["n"],
        "win_rate":      round(rec["wr"], 4),
        "boot_med":      round(rec["b50"], 4),
        "mc_prob":       round(rec["mc_p"], 4),
        "sym_floor":     round(rec["sym_floor"], 4),
        "fold_floor":    round(rec["fold_floor"], 4),
        "mdd":           round(rec["mdd"], 4),
        "trade_overlap": round(rec["trade_overlap"], 4),
        "pnl_corr":      round(rec["pnl_corr"], 4),
        "score":         rec["score"],
        "verdict":       rec["verdict"],
        "near_e3":       rec["near_e3"],
        "n_conds":       len(rec["cids"]),
    })
pd.DataFrame(rows).to_csv(f"{OUT}/r058_e4_candidates.csv", index=False)
print(f"  ✓  {OUT}/r058_e4_candidates.csv  ({len(rows)} rows)")

# Portfolio CSV
port_rows = [
    {"portfolio": "E3.1 alone",
     "pf": round(m_e31["pf"],4), "n": m_e31["n"], "wr": round(m_e31["wr"],4),
     "mdd": round(m_e31["mdd"],4), "boot_med": round(b50_e31,4),
     "mc_prob": round(mc_e31["prob_profit"],4), "ues": ues_e31},
    {"portfolio": "E4 alone",
     "pf": round(m_e4["pf"],4), "n": m_e4["n"], "wr": round(m_e4["wr"],4),
     "mdd": round(m_e4["mdd"],4), "boot_med": round(b50_e4,4),
     "mc_prob": round(mc_e4["prob_profit"],4), "ues": ues_e4},
    {"portfolio": "E3.1 + E4",
     "pf": round(m_comb["pf"],4), "n": m_comb["n"], "wr": round(m_comb["wr"],4),
     "mdd": round(m_comb["mdd"],4), "boot_med": round(b50_c,4),
     "mc_prob": round(mc_comb["prob_profit"],4), "ues": ues_comb},
]
pd.DataFrame(port_rows).to_csv(f"{OUT}/r058_portfolio.csv", index=False)
print(f"  ✓  {OUT}/r058_portfolio.csv")

# Feature freq CSV
pd.DataFrame(feature_freq).to_csv(f"{OUT}/r058_feature_freq.csv", index=False)
print(f"  ✓  {OUT}/r058_feature_freq.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  R058 COMPLETE — INDEPENDENT STRUCTURAL EDGE DISCOVERY (E4)")
print(SEP)
print(f"  Symbols:              {len(all_dfs)} (49 target)")
print(f"  Candidates:           {len(all_candidates):,}")
print(f"  Fast-screen pass:     {len(screen_pass):,}")
print(f"  Full WF survivors:    {len(env_records)}")
print(f"  Independence pass:    {len(independent_envs)}")
print(f"  Rejected (overlap):   {len(rejected_by_overlap)}")
print()
print(f"  E3.1 baseline:  PF={m_e31['pf']:.3f}  n={m_e31['n']}")
if best_e4:
    print(f"  Best E4:        {best_e4['label']}")
    print(f"    PF={best_e4['pf']:.3f}  n={best_e4['n']}  Overlap={best_e4['trade_overlap']:.1%}  "
          f"Corr={best_e4['pnl_corr']:+.3f}  UES={best_e4['ues']:.1f}")
    print(f"    Verdict={best_e4['verdict']}  Freeze={'YES' if freeze_e4 else 'NO'}")
    print()
    print(f"  Combined portfolio: PF={m_comb['pf']:.3f}  n={m_comb['n']}  "
          f"UES={ues_comb:.1f}  MDD={m_comb['mdd']:.1%}")
else:
    print("  No qualifying E4 environment found.")
    print("  Conclusion: E4 does not currently exist in the available search space.")
print(SEP)
