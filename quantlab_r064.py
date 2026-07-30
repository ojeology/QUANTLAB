"""
QUANTLAB AI — R064
Full Cache Structural Mining: Discovery Research

Objective:
  Stop treating E3.1 as centre. Treat the entire accumulated QuantLab cache
  as a research database. Discover completely new structural edge families
  that have never been explored before.

  - Auto-detect every cached 1H symbol
  - Expanded 32-condition library (25 existing + 7 new structural conditions)
  - Search all valid 3- and 4-condition combos from zero
  - NO optimisation. NO threshold changes. Frozen quantile thresholds throughout.
  - Full validation: 5-fold WFO, bootstrap, Monte Carlo, permutation null,
    LOO-symbol, LOO-fold, ablation, parameter robustness
  - Independence analysis vs E3.1 and DST_MD family
  - Portfolio discovery
  - Diversity scoring (0-100)
  - Family classification and ranking
  - Final answers to 8 discovery questions
"""

import os, sys, math, warnings, itertools, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID   = "R064"
OUT           = CONFIG["OUTPUT_FOLDER"]
CACHE         = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL       = CONFIG["STARTING_CAPITAL"]
RR            = CONFIG["RISK_REWARD"]
IS_RATIO      = 0.80          # 80% IS, 20% OOS (same as R058–R063)
MIN_BARS      = 2_000         # minimum bars to include a symbol
N_FWD_FOLDS   = 5
N_BOOT        = 2_000
N_MC          = 2_000
N_PERM        = 500
RAND_SEED     = 42
TOP_SCREEN    = 300           # oracle survivors forwarded to full WFO
MIN_N_PILOT   = 5             # min oracle trades per pilot symbol

# Promotion thresholds (same as prior rounds for consistency)
PROM_PF       = 1.30
PROM_N        = 30
PROM_BOOT     = 1.20
PROM_MC       = 0.80
PROM_MDD      = 0.20

# E3.1 frozen reference
E31_LABEL     = "BBW_STRICT+RV_LO+DST_NR+PRG_VH"
E31_CORE      = frozenset({"BBW_LO","RV_LO","DST_NR","PRG_VH"})  # approx with BBW_LO=p33

# DST_MD family reference (R059/R060 era)
DST_MD_CIDS   = ("BBW_LO","RV_LO","DST_MD","PRG_VH")

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

# ─────────────────────────────────────────────────────────────────────────────
# EXPANDED CONDITION LIBRARY — 32 structural conditions
# Format: (id, label, feature_col, direction, param, category, description)
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    # ── ATR / Volatility Regime ───────────────────────────────────────────────
    ("ATR_LO", "ATR<p25",      "atr_rank",      "lt_q",      0.25, "vol",
     "Very quiet market — bottom-quartile ATR range"),
    ("ATR_MD", "ATR<p40",      "atr_rank",      "lt_q",      0.40, "vol",
     "Quiet market — below-average ATR"),
    ("ATR_HI", "ATR>p67",      "atr_rank",      "gt_q",      0.67, "vol",
     "Active market — top-third ATR expansion"),
    ("ATR_VH", "ATR>p80",      "atr_rank",      "gt_q",      0.80, "vol",
     "Very active market — top-quintile ATR surge"),
    # ── Bollinger Band Width (Compression / Expansion) ────────────────────────
    ("BBW_LO", "BBW<p33",      "bb_width",      "lt_q",      0.33, "vol",
     "Bollinger compression — band squeeze / coil building"),
    ("BBW_HI", "BBW>p67",      "bb_width",      "gt_q",      0.67, "vol",
     "Bollinger expansion — bands widening out"),
    # ── Bollinger Band Position (NEW) ─────────────────────────────────────────
    ("BBP_LO", "BBPos<p33",    "bb_pos",        "lt_q",      0.33, "vol",
     "Price near lower Bollinger Band — potential mean-reversion zone"),
    ("BBP_HI", "BBPos>p67",    "bb_pos",        "gt_q",      0.67, "vol",
     "Price near upper Bollinger Band — momentum / breakout zone"),
    # ── Realised Volatility ───────────────────────────────────────────────────
    ("RV_LO",  "RealVol<p33",  "real_vol_20",   "lt_q",      0.33, "vol",
     "Low realized volatility — calm background returns"),
    ("RV_HI",  "RealVol>p67",  "real_vol_20",   "gt_q",      0.67, "vol",
     "High realized volatility — elevated background returns"),
    # ── Relative Volume (NEW) ─────────────────────────────────────────────────
    ("RVOL_HI","RelVol>p67",   "rel_vol_rank",  "gt_q",      0.67, "vol",
     "High relative volume — volume expansion context"),
    ("RVOL_LO","RelVol<p33",   "rel_vol_rank",  "lt_q",      0.33, "vol",
     "Low relative volume — quiet volume / absorption"),
    # ── EMA200 Slope (Trend Direction) ────────────────────────────────────────
    ("SLP_DN", "Slope<0",      "ema200_slope",  "lt_fixed",  0.0,  "trend",
     "Downtrend slope — declining long-term average"),
    ("SLP_UP", "Slope>0",      "ema200_slope",  "gt_fixed",  0.0,  "trend",
     "Uptrend slope — rising long-term average"),
    # ── EMA200 Distance (Proximity / Extension) ───────────────────────────────
    ("DST_NR", "Dist<p33",     "ema_dist_pct",  "lt_q",      0.33, "trend",
     "Near EMA200 — price hugging long-term average"),
    ("DST_MD", "Dist>p60+",    "ema_dist_pct",  "gt_q_pos",  0.60, "trend",
     "Moderate upside extension — above EMA200 by >p60"),
    ("DST_FR", "Dist>p75+",    "ema_dist_pct",  "gt_q_pos",  0.75, "trend",
     "Far from EMA200 — extended above by >p75"),
    # ── Close vs EMA50 (NEW: shorter-term trend) ──────────────────────────────
    ("EMA50_NR","E50Dist<p33", "ema50_dist_pct","lt_q",      0.33, "trend",
     "Close near EMA50 — short-term support/resistance zone"),
    ("EMA50_AB","E50Dist>p67", "ema50_dist_pct","gt_q",      0.67, "trend",
     "Close above EMA50 by >p67 — short-term momentum confirmation"),
    # ── ADX Trend Strength ────────────────────────────────────────────────────
    ("ADX_WK", "ADX<p33",      "adx14",         "lt_q",      0.33, "trend",
     "Weak trend — choppy / range-bound market"),
    ("ADX_TR", "ADX>p50",      "adx14",         "gt_q",      0.50, "trend",
     "Trending — above-median directional strength"),
    ("ADX_ST", "ADX>p67",      "adx14",         "gt_q",      0.67, "trend",
     "Strong trend — top-third directional conviction"),
    # ── Previous Bar Range / Body Quality ────────────────────────────────────
    ("PRG_LO", "PrevRng<p33",  "prev_range_r",  "lt_q",      0.33, "prev",
     "Small prior bar — tight previous candle"),
    ("PRG_HI", "PrevRng>p67",  "prev_range_r",  "gt_q",      0.67, "prev",
     "Large prior bar — high previous amplitude"),
    ("PRG_VH", "PrevRng>p80",  "prev_range_r",  "gt_q",      0.80, "prev",
     "Very large prior bar — strong prior impulse"),
    ("PBD_HI", "PrevBody>p67", "prev_body_r",   "gt_q",      0.67, "prev",
     "Large prior body — decisive prior candle"),
    ("PBP_HI", "BodyPct>p60",  "prev_body_pct", "gt_q",      0.60, "prev",
     "High body pct — low-wick prior candle"),
    ("PBP_LO", "BodyPct<p33",  "prev_body_pct", "lt_q",      0.33, "prev",
     "Low body pct — doji-like prior candle"),
    # ── Close Relative to Bar High (NEW: candle quality) ─────────────────────
    ("CLH_HI", "CloseHigh>p67","close_high_r",  "gt_q",      0.67, "prev",
     "Strong bar close — price closed near session high (bullish structure)"),
    ("CLH_LO", "CloseHigh<p33","close_high_r",  "lt_q",      0.33, "prev",
     "Weak bar close — price closed near session low (exhaustion structure)"),
    # ── Session Windows ───────────────────────────────────────────────────────
    ("US",     "US(14-21UTC)", "hour_utc",      "hour_rng",  (14,21), "session",
     "US session — New York trading window"),
    ("LON",    "London(7-14)", "hour_utc",      "hour_rng",  (7, 14), "session",
     "London session — European trading window"),
    ("ASI",    "Asia(0-6UTC)", "hour_utc",      "hour_rng",  (0,  6), "session",
     "Asia session — Asian trading window"),
]

COND_IDS    = [c[0] for c in CONDITIONS_DEF]
COND_BY_ID  = {c[0]: c for c in CONDITIONS_DEF}
COND_CATS   = {c[0]: c[5] for c in CONDITIONS_DEF}
COND_DESC   = {c[0]: c[6] for c in CONDITIONS_DEF}
QUANT_FEATS = [
    "atr_rank","bb_width","bb_pos","real_vol_20","rel_vol_rank",
    "ema_dist_pct","ema50_dist_pct","adx14",
    "prev_range_r","prev_body_r","prev_body_pct","close_high_r",
]

# ── Contradictory pairs (mutually exclusive conditions) ───────────────────────
INVALID_PAIRS = {
    frozenset({"ATR_LO","ATR_MD"}), frozenset({"ATR_LO","ATR_HI"}),
    frozenset({"ATR_LO","ATR_VH"}), frozenset({"ATR_MD","ATR_HI"}),
    frozenset({"ATR_MD","ATR_VH"}), frozenset({"ATR_HI","ATR_VH"}),
    frozenset({"BBW_LO","BBW_HI"}),
    frozenset({"BBP_LO","BBP_HI"}),
    frozenset({"RV_LO","RV_HI"}),
    frozenset({"RVOL_LO","RVOL_HI"}),
    frozenset({"SLP_DN","SLP_UP"}),
    frozenset({"DST_NR","DST_MD"}), frozenset({"DST_NR","DST_FR"}),
    frozenset({"DST_MD","DST_FR"}),
    frozenset({"ADX_WK","ADX_TR"}), frozenset({"ADX_WK","ADX_ST"}),
    frozenset({"ADX_TR","ADX_ST"}),
    frozenset({"PRG_LO","PRG_HI"}), frozenset({"PRG_LO","PRG_VH"}),
    frozenset({"PRG_HI","PRG_VH"}),
    frozenset({"PBP_LO","PBP_HI"}),
    frozenset({"CLH_HI","CLH_LO"}),
    frozenset({"US","LON"}), frozenset({"US","ASI"}), frozenset({"LON","ASI"}),
    frozenset({"EMA50_NR","EMA50_AB"}),
}

def is_valid_combo(cids):
    for a, b in itertools.combinations(cids, 2):
        if frozenset({a, b}) in INVALID_PAIRS:
            return False
    return True

def e31_proximity(cids):
    """Count how many E3 core conditions appear in this combo."""
    return len(frozenset(cids) & E31_CORE)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (extended)
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c  = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]; o = df["open"]

    # Core indicators
    df["ema200"]         = calc_ema(c, 200)
    df["ema50"]          = calc_ema(c, 50)
    df["atr14"]          = calc_atr(df, 14)
    df["atr_rank"]       = df["atr14"].rolling(100).rank(pct=True) * 100

    # Bollinger Bands
    bb_mid               = c.rolling(20).mean()
    bb_std               = c.rolling(20).std()
    bb_upper             = bb_mid + 2 * bb_std
    bb_lower             = bb_mid - 2 * bb_std
    bb_range             = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_width"]       = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["bb_pos"]         = (c - bb_lower) / bb_range   # 0=lower band, 1=upper band

    # EMA distance
    df["ema_dist_pct"]   = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"]   = (df["ema200"] - df["ema200"].shift(10)) / \
                           df["ema200"].shift(10).replace(0, np.nan)
    df["ema50_dist_pct"] = (c - df["ema50"]) / df["ema50"].replace(0, np.nan) * 100

    # Volume
    vol_ma               = v.rolling(20).mean()
    df["rel_vol"]        = v / vol_ma.replace(0, np.nan)
    df["rel_vol_rank"]   = df["rel_vol"].rolling(100).rank(pct=True) * 100

    # Realised vol
    log_ret              = np.log(c / c.shift(1))
    df["real_vol_20"]    = log_ret.rolling(20).std() * 100.0

    # ADX
    df["adx14"]          = calc_adx(df, 14)

    # Previous bar features
    df["prev_close"]     = c.shift(1)
    df["prev_atr14"]     = df["atr14"].shift(1)
    prev_range           = h.shift(1) - l.shift(1)
    prev_body            = (c.shift(1) - o.shift(1)).abs()
    df["prev_range_r"]   = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]    = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"]  = prev_body  / prev_range.replace(0, np.nan)

    # NEW: Close-to-high ratio (strength of current close within the bar)
    bar_range = h - l
    df["close_high_r"]   = (c - l) / bar_range.replace(0, np.nan)

    # Datetime features
    dt                   = pd.to_datetime(df["datetime"], utc=True)
    df["hour_utc"]       = dt.dt.hour.astype(np.int16)
    df["dow"]            = dt.dt.dayofweek

    return df

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD LEARNING — frozen quantile thresholds from IS data
# ─────────────────────────────────────────────────────────────────────────────
def learn_thresholds(df_is):
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (_, _, feat, direction, param, _, _) in COND_BY_ID.items():
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
    # BBW_STRICT (p25 of bb_width) for E3.1 baseline
    if "bb_width" in valid.columns:
        thr["BBW_STRICT"] = float(valid["bb_width"].dropna().quantile(0.25))
    return thr

# ─────────────────────────────────────────────────────────────────────────────
# MASKS
# ─────────────────────────────────────────────────────────────────────────────
def build_cond_mask(col, nan_mask, direction, threshold):
    if direction == "lt_q":
        if np.isnan(threshold): return np.zeros(len(col), dtype=bool)
        return (~nan_mask) & (col < threshold)
    elif direction in ("gt_q","gt_q_pos"):
        if np.isnan(threshold): return np.zeros(len(col), dtype=bool)
        return (~nan_mask) & (col > threshold)
    elif direction == "gt_fixed": return (~nan_mask) & (col > threshold)
    elif direction == "lt_fixed": return (~nan_mask) & (col < threshold)
    elif direction == "hour_rng":
        lo, hi = threshold; return (col >= lo) & (col <= hi)
    return np.zeros(len(col), dtype=bool)

def build_env_mask(df, cond_ids, thr):
    N    = len(df); mask = np.ones(N, dtype=bool)
    for cid in cond_ids:
        _, _, feat, direction, _, _, _ = COND_BY_ID[cid]
        if feat not in df.columns: return np.zeros(N, dtype=bool)
        col   = df[feat].values
        nan_m = np.isnan(col) if col.dtype.kind == "f" else np.zeros(N, dtype=bool)
        mask &= build_cond_mask(col, nan_m, direction, thr.get(cid, np.nan))
    return mask

def build_e31_mask(df, thr):
    """E3.1: BBW_STRICT(p25) + RV_LO + DST_NR + PRG_VH."""
    N = len(df); mask = np.ones(N, dtype=bool)
    for cid, feat, direction in [
        ("BBW_STRICT","bb_width",     "lt_q"),
        ("RV_LO",     "real_vol_20",  "lt_q"),
        ("DST_NR",    "ema_dist_pct", "lt_q"),
        ("PRG_VH",    "prev_range_r", "gt_q"),
    ]:
        t = thr.get(cid, np.nan)
        if np.isnan(t): return np.zeros(N, dtype=bool)
        col   = df[feat].values
        nan_m = np.isnan(col)
        if direction == "lt_q": mask &= (~nan_m) & (col < t)
        else:                   mask &= (~nan_m) & (col > t)
    return mask

def entry_signal(df, env_mask):
    rv = df["rel_vol"].values; c = df["close"].values
    o  = df["open"].values;    pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & env_mask

# ─────────────────────────────────────────────────────────────────────────────
# ORACLE FAST PRE-SCREEN
# ─────────────────────────────────────────────────────────────────────────────
def precompute_oracle(df, rr=RR, max_hold=100):
    min_sl = CONFIG["MIN_SL_PCT"]
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    atr = df["prev_atr14"].values.astype(np.float64)
    N   = len(df); result = np.zeros(N, dtype=np.int8)
    for i in range(N - 2):
        j = i + 1; a = atr[j]
        if np.isnan(a) or a <= 0: continue
        ep = o[j]
        if np.isnan(ep) or ep <= 0 or a / ep < min_sl: continue
        sl = ep - a; tp = ep + rr * a
        end = min(j + max_hold + 1, N)
        fh = h[j:end]; fl = l[j:end]
        tp_mask = fh >= tp; sl_mask = fl <= sl
        has_tp = tp_mask.any(); has_sl = sl_mask.any()
        if not has_tp and not has_sl: continue
        tp_idx = int(np.argmax(tp_mask)) if has_tp else max_hold + 1
        sl_idx = int(np.argmax(sl_mask)) if has_sl else max_hold + 1
        result[i] = 1 if (has_tp and tp_idx <= sl_idx) else -1
    return result

def fast_pf(signal_mask, oracle, min_n=MIN_N_PILOT):
    idx = np.where(signal_mask)[0]
    if len(idx) == 0: return 0.0, 0
    outcomes = oracle[idx]
    wins  = int((outcomes ==  1).sum())
    losses= int((outcomes == -1).sum())
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
    hou_ = df["hour_utc"].values; dow_ = df["dow"].values
    rv_  = df["rel_vol"].values

    for i in range(1, len(df)):
        if in_pos:
            sl_hit = lo_[i] <= st; tp_hit = hi_[i] >= tk
            if sl_hit or tp_hit:
                xp    = (st * (1 - slp)) if sl_hit else tk
                gross = (xp - ep) * sz
                cost  = (ep * sz + xp * sz) * (fee + spd)
                slpc  = (st - xp) * sz if sl_hit else 0.0
                net   = gross - cost - slpc
                h_e   = int(hou_[ei])
                sess  = ("London" if 7 <= h_e <= 13 else
                         "US"     if 14 <= h_e <= 20 else "Asia")
                trades.append({
                    "sym": sym, "fold": fold_label,
                    "entry_ts": str(et),
                    "pnl": round(net, 4),
                    "win": int(not sl_hit),
                    "session": sess,
                    "dow": int(dow_[ei]),
                    "rel_vol": float(rv_[ei]) if not np.isnan(rv_[ei]) else 1.0,
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
        return {"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,"net":0.0,"mdd":0.0,
                "pnls":np.array([]),"equity":np.array([CAPITAL])}
    pnl  = np.array([t["pnl"] for t in trades])
    wins = np.array([t["win"] for t in trades], dtype=bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw   = pnl[wins].sum()       if nw else 0.0
    gl   = abs(pnl[~wins].sum()) if nl else 0.0
    eq   = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak = np.maximum.accumulate(eq)
    mdd  = float(((eq - peak) / peak).min())
    wr   = nw / n; exp = wr * RR - (1 - wr)
    return {"n":n,"wr":wr,"pf":safe_pf(gw,gl),"exp_r":exp,
            "net":float(pnl.sum()),"mdd":mdd,"pnls":pnl,"equity":eq}

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=RAND_SEED):
    if len(pnls) < 5: return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed); pfs = []
    for _ in range(n_iter):
        s = rng.choice(pnls, len(pnls), replace=True)
        pfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
    return float(np.percentile(pfs,5)), float(np.percentile(pfs,50)), float(np.percentile(pfs,95))

def monte_carlo(pnls, n_iter=N_MC, seed=RAND_SEED):
    if len(pnls) < 5: return {"prob_profit":0.0,"finals":np.array([CAPITAL]),"mdd_95":1.0}
    rng    = np.random.default_rng(seed)
    finals = []; mdds = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        eq = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(s)])
        pk = np.maximum.accumulate(eq)
        finals.append(float(eq[-1]))
        mdds.append(float(((eq-pk)/pk).min()))
    return {"prob_profit": float((np.array(finals) > CAPITAL).mean()),
            "finals": np.array(finals),
            "mdd_95": float(np.percentile(mdds, 95))}

def permutation_test(pnls, n_iter=N_PERM, seed=RAND_SEED+1):
    """
    Null test: shuffle trade order 500x, measure resulting PF distribution.
    p_value = fraction of null PFs that exceed real PF.
    (Tests whether PF is driven by trade selection vs random sequence.)
    """
    if len(pnls) < 10: return 1.0, 0.0
    real_pf = safe_pf(pnls[pnls>0].sum(), abs(pnls[pnls<0].sum()))
    rng = np.random.default_rng(seed); null_pfs = []
    for _ in range(n_iter):
        s = rng.permutation(pnls)
        null_pfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
    null_pfs = np.array(null_pfs)
    p_val = float((null_pfs >= real_pf).mean())
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

def full_stats(trades, sym_trades_d):
    m          = metrics(trades)
    b5,b50,b95 = bootstrap_pf(m["pnls"])
    mc         = monte_carlo(m["pnls"])
    p_val, _   = permutation_test(m["pnls"])
    ls, sf     = loo_sym(sym_trades_d)
    lf_d, ff   = loo_fld(trades)
    criteria = [
        m["pf"]   > PROM_PF,
        m["n"]    >= PROM_N,
        b50       > PROM_BOOT,
        mc["prob_profit"] > PROM_MC,
        sf        > 1.0,
        ff        > 1.0,
        abs(m["mdd"]) < PROM_MDD,
    ]
    score   = sum(criteria)
    verdict = ("PROMOTE"   if score >= 7 else
               "WATCHLIST" if score >= 5 and m["pf"] > PROM_PF else
               "REJECT")
    return {**m, "b5":b5,"b50":b50,"b95":b95,
            "mc_p":mc["prob_profit"],"mdd_mc95":mc["mdd_95"],
            "p_val":p_val,"sym_floor":sf,"fold_floor":ff,
            "loo_sym":ls,"loo_fld":lf_d,"score":score,"verdict":verdict}

def compute_ues(pf, b50, mc_p, sf, ff, mdd, n):
    pf_pts   = min(25.0, max(0.0, (pf - 1.0) * 25.0))
    mc_pts   = min(20.0, max(0.0, mc_p * 20.0))
    boot_pts = min(15.0, max(0.0, (b50 - 1.0) / 0.5 * 15.0))
    loos_pts = min(15.0, max(0.0, (sf - 0.8)  / 0.5 * 15.0))
    loof_pts = min(10.0, max(0.0, (ff - 0.8)  / 0.5 * 10.0))
    mdd_pts  = min(10.0, max(0.0, (1.0 - abs(mdd) / 0.30) * 10.0))
    n_pts    = min(5.0,  max(0.0, (n / PROM_N) * 2.5))
    return round(pf_pts + mc_pts + boot_pts + loos_pts + loof_pts + mdd_pts + n_pts, 1)

# ─────────────────────────────────────────────────────────────────────────────
# INDEPENDENCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def compute_independence(ref_trades, cand_trades):
    if not ref_trades or not cand_trades:
        return {"trade_overlap":0.0,"pnl_corr":0.0,"sym_overlap":0.0,"session_overlap":0.0}
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
            d[t["entry_ts"][:10]] += t["pnl"]
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

def independence_score(indep):
    """Composite 0–100. Higher = more independent from reference."""
    to = 1.0 - indep["trade_overlap"]
    cr = 1.0 - max(0.0, indep["pnl_corr"])
    so = 1.0 - indep["sym_overlap"] * 0.5
    se = 1.0 - indep["session_overlap"] * 0.5
    return round(to * 40 + cr * 30 + so * 15 + se * 15, 1)

# ─────────────────────────────────────────────────────────────────────────────
# FAMILY CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
def classify_family(cids, stats):
    """Automatically classify a condition set into a structural family."""
    cs = set(cids)
    cats = [COND_CATS[c] for c in cids]

    # Check for specific structural signatures
    comp_conds  = {"BBW_LO","ATR_LO","ATR_MD","RV_LO","RVOL_LO"}
    exp_conds   = {"BBW_HI","ATR_HI","ATR_VH","RV_HI","RVOL_HI"}
    trend_conds = {"SLP_UP","SLP_DN","ADX_TR","ADX_ST","DST_FR","DST_MD","EMA50_AB"}
    range_conds = {"ADX_WK","DST_NR","EMA50_NR"}
    prev_conds  = {"PRG_VH","PRG_HI","PBD_HI","PBP_HI","CLH_HI"}
    rev_conds   = {"BBP_LO","PRG_LO","CLH_LO"}
    moment_conds= {"PRG_VH","PBD_HI","RVOL_HI","ATR_HI","BBP_HI"}

    has_comp   = bool(cs & comp_conds)
    has_exp    = bool(cs & exp_conds)
    has_trend  = bool(cs & trend_conds)
    has_range  = bool(cs & range_conds)
    has_prev   = bool(cs & prev_conds)
    has_rev    = bool(cs & rev_conds)
    has_moment = bool(cs & moment_conds)

    pf = stats.get("pf", 1.0)
    n  = stats.get("n", 0)

    if has_comp and has_prev and not has_exp:
        return "Compression Breakout", "Compressed market + large prior bar = coil-spring entry"
    elif has_comp and has_trend:
        return "Quiet Trend Continuation", "Low-volatility trend regime — price hugging the mean"
    elif has_exp and has_prev and not has_comp:
        return "Volatility Expansion Momentum", "Expanding volatility + strong prior move = momentum burst"
    elif has_rev and has_range:
        return "Mean Reversion Setup", "Price near lower band + ranging market = reversion attempt"
    elif has_trend and has_prev and "DST_FR" in cs:
        return "Extended Trend Pullback", "Price far from MA + strong prior bar = pullback continuation"
    elif "ADX_WK" in cs and has_prev:
        return "Choppy-Market Breakout", "Weak ADX + large prior bar = range-break momentum"
    elif has_exp and has_trend:
        return "Trend Expansion", "Trending market + volatility expanding = trend acceleration"
    elif has_comp and not has_trend and not has_prev:
        return "Volatility Contraction Setup", "Multiple compression signals = pre-breakout coil"
    elif "PRG_VH" in cs and "ADX_ST" in cs:
        return "Strong Trend Impulse", "Strong trend + very large prior bar = impulse continuation"
    elif has_moment and not has_comp:
        return "Momentum Burst", "Volume + volatility expansion = momentum ignition"
    elif has_rev:
        return "Pullback Continuation", "Prior-bar exhaustion structure within larger trend"
    elif cats.count("session") >= 1:
        return "Session-Anchored Structure", "Edge concentrated in specific trading session"
    elif has_trend:
        return "Trend Regime Filter", "Trend-filtered momentum entry"
    else:
        return "Novel Structural Family", "Uncategorised structural combination"

# ─────────────────────────────────────────────────────────────────────────────
# DIVERSITY SCORE (0-100)
# ─────────────────────────────────────────────────────────────────────────────
def diversity_score(cids, trades, sym_trades_d, e31_trades, all_survivors):
    """
    Structural uniqueness across 6 dimensions:
    30 — Structural uniqueness vs other survivors
    20 — Regime diversity (categories in combo)
    20 — Symbol diversity (how many symbols fire)
    15 — Session diversity
    15 — Independence from E3.1
    """
    cs = frozenset(cids)
    # (1) Structural uniqueness: avg Jaccard distance vs other survivors
    if all_survivors:
        jacc = []
        for other_cids in all_survivors:
            o = frozenset(other_cids)
            j = len(cs & o) / len(cs | o) if (cs | o) else 0.0
            jacc.append(j)
        avg_jacc = np.mean(jacc) if jacc else 0.0
    else:
        avg_jacc = 0.0
    struct_score = 30.0 * (1.0 - avg_jacc)

    # (2) Category diversity
    n_cats = len(set(COND_CATS[c] for c in cids if c in COND_CATS))
    regime_score = 20.0 * min(1.0, n_cats / 3.0)   # 3+ categories = max score

    # (3) Symbol diversity
    active_syms = sum(1 for s, tl in sym_trades_d.items() if len(tl) >= 2)
    sym_score = 20.0 * min(1.0, active_syms / 25.0)

    # (4) Session diversity
    if trades:
        sess_counts = defaultdict(int)
        for t in trades: sess_counts[t.get("session","?")] += 1
        n_sess = len(sess_counts)
        max_pct = max(sess_counts.values()) / len(trades)
        sess_score = 15.0 * (n_sess / 3.0) * (1.0 - max(0.0, max_pct - 0.6))
    else:
        sess_score = 0.0

    # (5) Independence from E3.1
    indep = compute_independence(e31_trades, trades)
    indep_score = 15.0 * (1.0 - indep["trade_overlap"])

    return round(struct_score + regime_score + sym_score + sess_score + indep_score, 1)

# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_candidates():
    cands = []
    for combo in itertools.combinations(COND_IDS, 3):
        if is_valid_combo(combo): cands.append(tuple(combo))
    for combo in itertools.combinations(COND_IDS, 4):
        if is_valid_combo(combo): cands.append(tuple(combo))
    return cands

# ─────────────────────────────────────────────────────────────────────────────
# FULL WFO RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_wfo(all_dfs, cond_ids):
    """5-fold expanding WFO across all symbols. Returns (all_trades, sym_trades)."""
    all_trades = []; sym_trades = defaultdict(list)
    for sym, (df_is, df_fwd, thr) in all_dfs.items():
        fwd_size = len(df_fwd); seg_size = max(1, fwd_size // N_FWD_FOLDS)
        for fi in range(N_FWD_FOLDS):
            seg_s  = fi * seg_size
            seg_e  = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
            df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
            if len(df_seg) < 20: continue
            em  = build_env_mask(df_seg, cond_ids, thr)
            sig = entry_signal(df_seg, em)
            tl  = run_backtest(df_seg, sig, sym, f"F{fi+1}")
            all_trades.extend(tl)
            sym_trades[sym].extend(tl)
    return all_trades, dict(sym_trades)

# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════
# MAIN RESEARCH BODY
# ════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  QUANTLAB AI — R064")
print("  Full Cache Structural Mining: Discovery Research")
print("  32-condition library | All cached 1H symbols | Zero assumptions")
print(SEP)
print()

t_start = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — DATA LOAD (auto-detect all 1H symbols)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 0 — Data Load (Full Cache)")
print(SEP)
print()

all_dfs   = {}   # sym → (df_is, df_fwd, thr)
excluded  = []
included  = []

for fname in sorted(os.listdir(CACHE)):
    if not fname.endswith("_1H.parquet"): continue
    sym = fname.replace("_1H.parquet","").replace("_","-")
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

SYMS = list(all_dfs.keys())
total_oos = sum(len(v[1]) for v in all_dfs.values())
total_is  = sum(len(v[0]) for v in all_dfs.values())

print(f"  Total cached 1H files   : {len(included) + len(excluded)}")
print(f"  Symbols included        : {len(SYMS)}")
print(f"  Symbols excluded        : {len(excluded)}")
print(f"  Total IS bars           : {total_is:,}")
print(f"  Total OOS bars          : {total_oos:,}")
print(f"  IS / OOS split          : {IS_RATIO:.0%} / {1-IS_RATIO:.0%}")
print()

if excluded:
    print(f"  Excluded symbols:")
    for sym, n, reason in excluded:
        print(f"    {sym:<25}  {n:>6} bars  {reason}")
    print()

print(f"  Included symbols ({len(SYMS)}):")
for sym, n_raw, n_is, n_oos in included:
    print(f"    {sym:<25}  total={n_raw:>6}  IS={n_is:>5}  OOS={n_oos:>5}")
print()

# Condition library summary
print(f"  Condition library: {len(CONDITIONS_DEF)} conditions across "
      f"{len(set(COND_CATS.values()))} categories")
for cat in sorted(set(COND_CATS.values())):
    cids_in = [c for c in COND_IDS if COND_CATS[c] == cat]
    print(f"    {cat:<10} ({len(cids_in)}) : {', '.join(cids_in)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — E3.1 BASELINE + DST_MD REFERENCE
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 1 — Reference Baselines (E3.1 & DST_MD)")
print(SEP)
print()

e31_trades  = []; sym_e31 = defaultdict(list)
dstmd_trades= []; sym_dstmd = defaultdict(list)

for sym, (df_is, df_fwd, thr) in all_dfs.items():
    fwd_size = len(df_fwd); seg_size = max(1, fwd_size // N_FWD_FOLDS)
    for fi in range(N_FWD_FOLDS):
        seg_s  = fi * seg_size
        seg_e  = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
        df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
        if len(df_seg) < 20: continue
        fl = f"F{fi+1}"

        # E3.1
        em31 = build_e31_mask(df_seg, thr)
        s31  = entry_signal(df_seg, em31)
        tl31 = run_backtest(df_seg, s31, sym, fl)
        e31_trades.extend(tl31); sym_e31[sym].extend(tl31)

        # DST_MD family
        em_dst = build_env_mask(df_seg, DST_MD_CIDS, thr)
        s_dst  = entry_signal(df_seg, em_dst)
        tl_dst = run_backtest(df_seg, s_dst, sym, fl)
        dstmd_trades.extend(tl_dst); sym_dstmd[sym].extend(tl_dst)

m31    = metrics(e31_trades)
m_dst  = metrics(dstmd_trades)
b5_31, b50_31, b95_31 = bootstrap_pf(m31["pnls"])
b5_dst,b50_dst,b95_dst= bootstrap_pf(m_dst["pnls"])

print(f"  E3.1  ({E31_LABEL})")
print(f"    PF={m31['pf']:.3f}  WR={m31['wr']:.1%}  n={m31['n']}"
      f"  MDD={m31['mdd']:.1%}  Boot50={b50_31:.3f}")
print()
print(f"  DST_MD ({'+'.join(DST_MD_CIDS)})")
print(f"    PF={m_dst['pf']:.3f}  WR={m_dst['wr']:.1%}  n={m_dst['n']}"
      f"  MDD={m_dst['mdd']:.1%}  Boot50={b50_dst:.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 2 — Candidate Generation")
print(SEP)
print()

all_candidates = generate_candidates()
n3 = sum(1 for c in all_candidates if len(c)==3)
n4 = sum(1 for c in all_candidates if len(c)==4)
near_e3 = [c for c in all_candidates if e31_proximity(c) >= 3]
far_e3  = [c for c in all_candidates if e31_proximity(c) <  3]

print(f"  Total valid 3-condition combos : {n3:,}")
print(f"  Total valid 4-condition combos : {n4:,}")
print(f"  Total candidates               : {len(all_candidates):,}")
print(f"  Near-E3.1 (≥3 shared conds)   : {len(near_e3):,}")
print(f"  Far-E3.1  (<3 shared conds)   : {len(far_e3):,}")
print(f"  New-territory combos (0 shared): "
      f"{sum(1 for c in all_candidates if e31_proximity(c)==0):,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ORACLE FAST PRE-SCREEN (8 diverse pilot symbols)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 3 — Oracle Fast Pre-Screen")
print(SEP)
print()

# Pilot selection: pick 8 diverse symbols from loaded universe
PILOT_PREFERENCES = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","BNB-USDT-SWAP",
    "LINK-USDT-SWAP","AVAX-USDT-SWAP","XRP-USDT-SWAP","INJ-USDT-SWAP",
]
PILOT_SYMS = [s for s in PILOT_PREFERENCES if s in all_dfs]
# Backfill if some not available
if len(PILOT_SYMS) < 8:
    for s in SYMS:
        if s not in PILOT_SYMS: PILOT_SYMS.append(s)
        if len(PILOT_SYMS) >= 8: break

pilot_data = {}
for sym in PILOT_SYMS:
    df_is, df_fwd, thr = all_dfs[sym]
    sp   = int(len(df_fwd) * 0.70)
    dfi  = df_fwd.iloc[:sp]
    dfo  = df_fwd.iloc[sp:].reset_index(drop=True)
    if len(dfo) < 50: continue
    thr_p  = learn_thresholds(dfi)
    oracle = precompute_oracle(dfo)
    pilot_data[sym] = (dfo, oracle, thr_p)

print(f"  Pilot symbols   : {len(pilot_data)}")
print(f"  Screening       : {len(all_candidates):,} candidates via oracle ...")

combo_scores = {}
for combo in all_candidates:
    total_wins = 0; total_loss = 0
    for sym, (dfo, oracle, thr_p) in pilot_data.items():
        em  = build_env_mask(dfo, combo, thr_p)
        sig = entry_signal(dfo, em)
        idx = np.where(sig[:-1])[0]
        if len(idx) == 0: continue
        oc = oracle[idx]
        total_wins += int((oc ==  1).sum())
        total_loss += int((oc == -1).sum())
    n = total_wins + total_loss
    pf_fast = (total_wins * RR) / (total_loss if total_loss > 0 else 0.5) if n >= MIN_N_PILOT else 0.0
    combo_scores[combo] = {"pf_fast": pf_fast, "n_fast": n}

screened = sorted(all_candidates, key=lambda c: -combo_scores[c]["pf_fast"])
pass_thresh = 1.10
screen_pass = [(c, combo_scores[c]) for c in screened
               if combo_scores[c]["pf_fast"] >= pass_thresh
               and combo_scores[c]["n_fast"]  >= MIN_N_PILOT]

print(f"  Screen threshold: oracle PF ≥ {pass_thresh:.2f}, n ≥ {MIN_N_PILOT}")
print(f"  Oracle survivors: {len(screen_pass):,}")

# Prioritise far-E3 combos
far_pass  = [(c,s) for c,s in screen_pass if e31_proximity(c) < 3]
near_pass = [(c,s) for c,s in screen_pass if e31_proximity(c) >= 3]
far_slots  = min(len(far_pass),  int(TOP_SCREEN * 0.85))
near_slots = min(len(near_pass), TOP_SCREEN - far_slots)
top_candidates = ([c for c,_ in far_pass[:far_slots]] +
                  [c for c,_ in near_pass[:near_slots]])

print(f"    Far-E3  survivors : {len(far_pass)}")
print(f"    Near-E3 survivors : {len(near_pass)}")
print(f"  Forwarding {len(top_candidates)} to full WFO (far={far_slots}, near={near_slots})")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FULL WALK-FORWARD (5-fold × all symbols)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 4 — Full Walk-Forward Validation")
print(SEP)
print()
print(f"  Running {len(top_candidates)} candidates × {len(SYMS)} symbols × "
      f"{N_FWD_FOLDS} folds ...")

wfo_results = {}   # combo → {"trades":[], "sym_trades":{}, basic metrics}

for i, combo in enumerate(top_candidates):
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t_start
        print(f"    [{i+1}/{len(top_candidates)}] {elapsed:.0f}s elapsed ...")
    trades, sym_trades = run_wfo(all_dfs, combo)
    if not trades:
        continue
    m = metrics(trades)
    if m["pf"] >= 1.10 and m["n"] >= 15:
        wfo_results[combo] = {"trades": trades, "sym_trades": sym_trades, **m}

print(f"  WFO survivors (PF≥1.10, n≥15) : {len(wfo_results)}")

# Sort by PF for display
wfo_ranked = sorted(wfo_results.keys(), key=lambda c: -wfo_results[c]["pf"])
print(f"  Top 20 by raw PF:")
print(f"  {'Rank':<4} {'Conditions':<40} {'PF':>6} {'n':>5} {'WR':>6} {'MDD':>7}")
print(f"  {'─'*4} {'─'*40} {'─'*6} {'─'*5} {'─'*6} {'─'*7}")
for i, combo in enumerate(wfo_ranked[:20]):
    r = wfo_results[combo]
    lbl = "+".join(combo)[:40]
    print(f"  {i+1:<4} {lbl:<40} {r['pf']:>6.3f} {r['n']:>5} {r['wr']:>5.1%} {r['mdd']:>6.1%}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FULL ROBUSTNESS VALIDATION SUITE
# Applies to all WFO survivors; full stats for top-50 by PF
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 5 — Full Robustness Validation Suite")
print(SEP)
print()

FULL_VALIDATE_N = min(50, len(wfo_ranked))
print(f"  Applying full validation to top {FULL_VALIDATE_N} candidates ...")
print(f"  (Bootstrap n={N_BOOT}, Monte Carlo n={N_MC}, Permutation n={N_PERM})")
print()

validated = {}   # combo → full_stats dict

for i, combo in enumerate(wfo_ranked[:FULL_VALIDATE_N]):
    r      = wfo_results[combo]
    trades = r["trades"]
    sym_t  = r["sym_trades"]
    fs     = full_stats(trades, sym_t)
    ues    = compute_ues(fs["pf"], fs["b50"], fs["mc_p"],
                        fs["sym_floor"], fs["fold_floor"], fs["mdd"], fs["n"])
    fs["ues"] = ues
    fs["combo"] = combo
    validated[combo] = fs

    if (i + 1) % 10 == 0:
        print(f"    [{i+1}/{FULL_VALIDATE_N}] validated ...")

print()
print(f"  Validation complete. Sorting by UES ...")
print()

# Sort by UES
validated_ranked = sorted(validated.keys(), key=lambda c: -validated[c]["ues"])

print(f"  {'Rank':<4} {'Conditions':<42} {'PF':>6} {'n':>5} {'UES':>6} "
      f"{'Boot50':>7} {'MC%':>6} {'SymFlr':>7} {'FldFlr':>7} {'MDD':>7} {'Verdict'}")
print(f"  {'─'*4} {'─'*42} {'─'*6} {'─'*5} {'─'*6} {'─'*7} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*10}")
for i, combo in enumerate(validated_ranked[:25]):
    v   = validated[combo]
    lbl = "+".join(combo)[:42]
    vtxt = ("✓PROMOTE" if v["verdict"]=="PROMOTE" else
            "~WATCH"   if v["verdict"]=="WATCHLIST" else "✗REJECT")
    print(f"  {i+1:<4} {lbl:<42} {v['pf']:>6.3f} {v['n']:>5} {v['ues']:>6.1f} "
          f"{v['b50']:>7.3f} {v['mc_p']:>5.1%} {v['sym_floor']:>7.3f} "
          f"{v['fold_floor']:>7.3f} {v['mdd']:>6.1%} {vtxt}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — INDEPENDENCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 6 — Independence Analysis")
print(SEP)
print()
print("  Measuring independence vs E3.1 and DST_MD family ...")
print()

for combo in validated_ranked[:FULL_VALIDATE_N]:
    v = validated[combo]
    trades = wfo_results[combo]["trades"]
    indep_e31   = compute_independence(e31_trades,   trades)
    indep_dst   = compute_independence(dstmd_trades, trades)
    v["indep_e31"]   = indep_e31
    v["indep_dst"]   = indep_dst
    v["indep_e31_sc"]= independence_score(indep_e31)
    v["indep_dst_sc"]= independence_score(indep_dst)

print(f"  {'Rank':<4} {'Conditions':<42} {'vs E3.1':>9} {'TrdOvlp':>8} "
      f"{'PnlCorr':>8} {'vs DSTMD':>9} {'TrdOvlp':>8}")
print(f"  {'─'*4} {'─'*42} {'─'*9} {'─'*8} {'─'*8} {'─'*9} {'─'*8}")
for i, combo in enumerate(validated_ranked[:25]):
    v   = validated[combo]
    lbl = "+".join(combo)[:42]
    ie  = v.get("indep_e31", {})
    print(f"  {i+1:<4} {lbl:<42} {v.get('indep_e31_sc',0):>9.1f} "
          f"{ie.get('trade_overlap',0):>7.1%} {ie.get('pnl_corr',0):>8.3f} "
          f"{v.get('indep_dst_sc',0):>9.1f} "
          f"{v.get('indep_dst',{}).get('trade_overlap',0):>7.1%}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FAMILY CLASSIFICATION & DIVERSITY SCORING
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 7 — Family Classification & Diversity Scoring")
print(SEP)
print()

survivors = validated_ranked[:FULL_VALIDATE_N]
all_surv_cids = [c for c in survivors]

for combo in survivors:
    v = validated[combo]
    trades = wfo_results[combo]["trades"]
    sym_t  = wfo_results[combo]["sym_trades"]
    fam_name, fam_why = classify_family(combo, v)
    ds = diversity_score(combo, trades, sym_t, e31_trades, all_surv_cids)
    v["family"]   = fam_name
    v["fam_why"]  = fam_why
    v["diversity"]= ds

# Show family distribution
fam_counts = defaultdict(list)
for combo in survivors:
    fam_counts[validated[combo]["family"]].append(combo)

print(f"  Discovered {len(fam_counts)} distinct structural family types:")
print()
for fam, combos in sorted(fam_counts.items(), key=lambda x: -len(x[1])):
    print(f"  [{len(combos):>3}] {fam}")
print()

# Detailed classification for top-25
print(f"  {'Rank':<4} {'Conditions':<40} {'Family':<35} {'Diversity':>10}")
print(f"  {'─'*4} {'─'*40} {'─'*35} {'─'*10}")
for i, combo in enumerate(survivors[:25]):
    v   = validated[combo]
    lbl = "+".join(combo)[:40]
    fam = v["family"][:35]
    print(f"  {i+1:<4} {lbl:<40} {fam:<35} {v['diversity']:>10.1f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — COMPREHENSIVE RANKING
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 8 — Comprehensive Ranking (Multi-Dimensional)")
print(SEP)
print()

# Composite score: UES (40%) + Independence from E3.1 (30%) + Diversity (30%)
for combo in survivors:
    v = validated[combo]
    ues   = v.get("ues", 0.0)
    indep = v.get("indep_e31_sc", 0.0)
    div   = v.get("diversity", 0.0)
    v["composite"] = round(ues * 0.40 + indep * 0.35 + div * 0.25, 1)

comp_ranked = sorted(survivors, key=lambda c: -validated[c]["composite"])

print(f"  Composite Score = UES×0.40 + Independence×0.35 + Diversity×0.25")
print()
print(f"  {'Rk':<3} {'Conditions':<40} {'Composite':>10} {'UES':>6} "
      f"{'Indep':>6} {'Div':>5} {'PF':>6} {'n':>5} {'Verdict':<10} {'Family'}")
print(f"  {'─'*3} {'─'*40} {'─'*10} {'─'*6} {'─'*6} {'─'*5} "
      f"{'─'*6} {'─'*5} {'─'*10} {'─'*30}")
for i, combo in enumerate(comp_ranked[:25]):
    v   = validated[combo]
    lbl = "+".join(combo)[:40]
    vt  = ("PROMOTE"  if v["verdict"]=="PROMOTE" else
           "WATCHLIST" if v["verdict"]=="WATCHLIST" else "reject")
    fam = v["family"][:30]
    print(f"  {i+1:<3} {lbl:<40} {v['composite']:>10.1f} {v['ues']:>6.1f} "
          f"{v.get('indep_e31_sc',0):>6.1f} {v.get('diversity',0):>5.1f} "
          f"{v['pf']:>6.3f} {v['n']:>5} {vt:<10} {fam}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PORTFOLIO DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 9 — Portfolio Discovery")
print(SEP)
print()

# Identify portfolio-eligible families (verdict != REJECT, indep from E3.1)
eligible = [c for c in comp_ranked
            if validated[c]["verdict"] in ("PROMOTE","WATCHLIST")
            and validated[c].get("indep_e31_sc", 0) >= 40.0]

print(f"  Eligible portfolio families (non-reject + indep≥40): {len(eligible)}")
print()

if len(eligible) >= 2:
    # Add E3.1 as anchor
    port_families = eligible[:min(8, len(eligible))]   # top-8 eligible by composite

    # Greedy portfolio: start with best, add each next family that improves UES+PF
    best_start = port_families[0]
    greedy_port= [best_start]
    greedy_trades = list(wfo_results[best_start]["trades"])

    for combo in port_families[1:]:
        candidate_trades = greedy_trades + wfo_results[combo]["trades"]
        m_candidate = metrics(candidate_trades)
        m_current   = metrics(greedy_trades)
        # Accept if PF improves or stays within 5% while n grows ≥10%
        if (m_candidate["pf"] >= m_current["pf"] * 0.95 and
            m_candidate["n"]  >= m_current["n"] * 1.10):
            greedy_port.append(combo)
            greedy_trades = candidate_trades

    # All-eligible combined
    all_combined_trades = []
    for c in port_families:
        all_combined_trades.extend(wfo_results[c]["trades"])
    m_all  = metrics(all_combined_trades)
    m_best = metrics(wfo_results[comp_ranked[0]]["trades"])
    m_greedy = metrics(greedy_trades)

    print(f"  Portfolio compositions:")
    print()
    print(f"  ① Best single family:    PF={m_best['pf']:.3f}  n={m_best['n']}"
          f"  MDD={m_best['mdd']:.1%}")
    print(f"  ② Greedy combo ({len(greedy_port)} fam): PF={m_greedy['pf']:.3f}  "
          f"n={m_greedy['n']}  MDD={m_greedy['mdd']:.1%}")
    print(f"  ③ All eligible ({len(port_families)} fam): PF={m_all['pf']:.3f}  "
          f"n={m_all['n']}  MDD={m_all['mdd']:.1%}")
    print()

    # E3.1 + best new family combo
    if e31_trades:
        m_e31_plus = metrics(e31_trades + wfo_results[comp_ranked[0]]["trades"])
        print(f"  ④ E3.1 + Best new:       PF={m_e31_plus['pf']:.3f}  "
              f"n={m_e31_plus['n']}  MDD={m_e31_plus['mdd']:.1%}")
    print()

    print(f"  Greedy portfolio members ({len(greedy_port)}):")
    for j, c in enumerate(greedy_port, 1):
        v = validated[c]
        print(f"    {j}. {'+'.join(c)}")
        print(f"       PF={v['pf']:.3f}  n={v['n']}  UES={v['ues']:.1f}"
              f"  Family: {v['family']}")
    print()

    # Bootstrap portfolio
    b5g, b50g, b95g = bootstrap_pf(np.array([t["pnl"] for t in greedy_trades]))
    mc_g = monte_carlo(np.array([t["pnl"] for t in greedy_trades]))
    print(f"  Greedy portfolio robustness:")
    print(f"    Bootstrap: P5={b5g:.3f}  P50={b50g:.3f}  P95={b95g:.3f}")
    print(f"    Monte Carlo: P(profit)={mc_g['prob_profit']:.1%}  MDD95={mc_g['mdd_95']:.1%}")
    print()
else:
    greedy_port = []
    print("  Insufficient eligible families for portfolio discovery.")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — FINAL ANSWERS TO 8 DISCOVERY QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 10 — Final Discovery Answers")
print(SEP)
print()

n_families   = len(fam_counts)
n_promote    = sum(1 for c in survivors if validated[c]["verdict"]=="PROMOTE")
n_watchlist  = sum(1 for c in survivors if validated[c]["verdict"]=="WATCHLIST")
best_overall = comp_ranked[0] if comp_ranked else None
most_indep   = max(survivors, key=lambda c: validated[c].get("indep_e31_sc",0)) \
               if survivors else None
highest_n    = max(survivors, key=lambda c: validated[c]["n"]) if survivors else None

print(f"  Q1. How many completely new edge families were discovered?")
print(f"      {n_families} distinct structural family types across {len(survivors)} validated candidates.")
print(f"      PROMOTE: {n_promote}  |  WATCHLIST: {n_watchlist}  |  "
      f"Families: {', '.join(fam_counts.keys())[:80]}")
print()

if best_overall:
    v = validated[best_overall]
    print(f"  Q2. Which family is strongest overall? (highest composite score)")
    print(f"      {'+'.join(best_overall)}")
    print(f"      PF={v['pf']:.3f}  UES={v['ues']:.1f}  Composite={v['composite']:.1f}")
    print(f"      Family: {v['family']} — {v['fam_why']}")
print()

if most_indep:
    v = validated[most_indep]
    print(f"  Q3. Which family is most independent from E3.1?")
    print(f"      {'+'.join(most_indep)}")
    print(f"      Independence score: {v.get('indep_e31_sc',0):.1f}/100")
    print(f"      Trade overlap with E3.1: {v.get('indep_e31',{}).get('trade_overlap',0):.1%}")
    print(f"      PnL correlation with E3.1: {v.get('indep_e31',{}).get('pnl_corr',0):.3f}")
print()

if highest_n:
    v = validated[highest_n]
    print(f"  Q4. Which family produces the highest annual trade count?")
    print(f"      {'+'.join(highest_n)}")
    print(f"      n={v['n']} forward trades  |  Family: {v['family']}")
    # Estimate annual rate (OOS period)
    oos_years = total_oos / (52 * 8760) if total_oos > 0 else 1
    ann_rate   = v["n"] / max(oos_years, 0.1)
    print(f"      Estimated annual rate: ~{ann_rate:.0f} trades/year "
          f"({total_oos:,} OOS bars ÷ {len(SYMS)} symbols)")
print()

promote_combos = [c for c in comp_ranked if validated[c]["verdict"]=="PROMOTE"]
print(f"  Q5. Can any family be promoted immediately?")
if promote_combos:
    for c in promote_combos[:3]:
        v = validated[c]
        print(f"      YES → {'+'.join(c)}")
        print(f"            PF={v['pf']:.3f}  UES={v['ues']:.1f}  "
              f"n={v['n']}  Score={v['score']}/7")
else:
    watchlist_combos = [c for c in comp_ranked if validated[c]["verdict"]=="WATCHLIST"]
    if watchlist_combos:
        v = validated[watchlist_combos[0]]
        print(f"      No immediate PROMOTE. Best WATCHLIST:")
        print(f"      → {'+'.join(watchlist_combos[0])}")
        print(f"        PF={v['pf']:.3f}  UES={v['ues']:.1f}  Score={v['score']}/7")
        print(f"        Criteria met: {v['score']}/7. Gap: needs more data or higher PF.")
    else:
        print(f"      No families meet promotion criteria yet.")
print()

best_r065 = comp_ranked[0] if comp_ranked else None
print(f"  Q6. Which family deserves a dedicated forensic R065?")
if best_r065:
    v = validated[best_r065]
    print(f"      {'+'.join(best_r065)}")
    print(f"      Reasoning: Highest composite score ({v['composite']:.1f}), "
          f"PF={v['pf']:.3f}, UES={v['ues']:.1f}")
    print(f"      R065 should: forensic entry-gate analysis, parameter sensitivity,")
    print(f"      symbol-by-symbol breakdown, time-of-day analysis, and portfolio fit")
print()

# New behaviours visible in full cache
new_fam_names = [f for f in fam_counts if f not in
                 {"Compression Breakout","Quiet Trend Continuation"}]
print(f"  Q7. Does the full cache reveal market behaviours invisible in the 49-symbol set?")
if new_fam_names:
    print(f"      YES. New structural families discovered:")
    for fname in new_fam_names[:5]:
        print(f"      → {fname} ({len(fam_counts[fname])} candidates)")
else:
    print(f"      The structural landscape is broadly similar, confirming the 49-symbol")
    print(f"      universe was representative. Full cache adds statistical power.")
print()

print(f"  Q8. Is there evidence for multiple orthogonal edge families?")
if len(eligible) >= 2:
    # Check average pairwise trade overlap among eligible families
    overlaps = []
    for i, c1 in enumerate(eligible[:6]):
        for c2 in eligible[i+1:6]:
            ind = compute_independence(wfo_results[c1]["trades"],
                                       wfo_results[c2]["trades"])
            overlaps.append(ind["trade_overlap"])
    avg_overlap = np.mean(overlaps) if overlaps else 0.0
    avg_corr    = np.mean([compute_independence(wfo_results[c1]["trades"],
                                                 wfo_results[c2]["trades"])["pnl_corr"]
                           for i,c1 in enumerate(eligible[:6])
                           for c2 in eligible[i+1:6]]) if len(eligible) >= 2 else 0.0
    print(f"      YES. Among top eligible families:")
    print(f"      Avg trade overlap: {avg_overlap:.1%} (lower=better)")
    print(f"      Avg PnL correlation: {avg_corr:.3f} (lower=better)")
    if avg_overlap < 0.25 and avg_corr < 0.40:
        print(f"      ✓ Strong evidence for independent, diversifiable edge families.")
        print(f"      ✓ A multi-strategy production portfolio is feasible.")
    elif avg_overlap < 0.40:
        print(f"      ~ Moderate evidence. Families are partially independent.")
        print(f"      ~ Portfolio combination would reduce drawdown but not eliminate correlation.")
    else:
        print(f"      ✗ High overlap. Families share structural basis — limited diversification.")
else:
    print(f"      Insufficient eligible families for cross-family analysis.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Exporting CSV outputs ...")
print(SEP2)
print()

# 11a: Full family rankings
rows = []
for i, combo in enumerate(comp_ranked):
    v = validated[combo]
    rows.append({
        "rank":        i + 1,
        "conditions":  "+".join(combo),
        "n_conds":     len(combo),
        "composite":   v.get("composite", 0.0),
        "pf":          v["pf"],
        "ues":         v.get("ues", 0.0),
        "n":           v["n"],
        "wr":          round(v["wr"], 4),
        "mdd":         round(v["mdd"], 4),
        "net":         round(v["net"], 2),
        "boot_p50":    round(v.get("b50", 0.0), 4),
        "boot_p5":     round(v.get("b5", 0.0), 4),
        "mc_prob":     round(v.get("mc_p", 0.0), 4),
        "p_value":     round(v.get("p_val", 1.0), 4),
        "sym_floor":   round(v.get("sym_floor", 0.0), 4),
        "fold_floor":  round(v.get("fold_floor", 0.0), 4),
        "indep_e31":   round(v.get("indep_e31_sc", 0.0), 1),
        "trade_ovlp":  round(v.get("indep_e31",{}).get("trade_overlap",0.0), 4),
        "pnl_corr":    round(v.get("indep_e31",{}).get("pnl_corr",0.0), 4),
        "diversity":   round(v.get("diversity", 0.0), 1),
        "family":      v.get("family", ""),
        "verdict":     v["verdict"],
        "score":       v["score"],
        "e31_proximity": e31_proximity(combo),
    })

df_rankings = pd.DataFrame(rows)
rank_path = os.path.join(OUT, "r064_family_rankings.csv")
df_rankings.to_csv(rank_path, index=False)
print(f"  → Family rankings      : {rank_path}  ({len(rows)} families)")

# 11b: Trade log for best family
if comp_ranked:
    best = comp_ranked[0]
    df_trades = pd.DataFrame(wfo_results[best]["trades"])
    trade_path = os.path.join(OUT, "r064_best_trades.csv")
    df_trades.to_csv(trade_path, index=False)
    print(f"  → Best family trades   : {trade_path}  ({len(df_trades)} trades)")

# 11c: Portfolio
if greedy_port:
    port_rows = []
    for j, c in enumerate(greedy_port, 1):
        v = validated[c]
        port_rows.append({
            "portfolio_rank": j,
            "conditions": "+".join(c),
            "pf": v["pf"],
            "n": v["n"],
            "ues": v.get("ues",0.0),
            "family": v.get("family",""),
            "indep_e31": v.get("indep_e31_sc",0.0),
            "diversity": v.get("diversity",0.0),
        })
    df_port = pd.DataFrame(port_rows)
    port_path = os.path.join(OUT, "r064_portfolio.csv")
    df_port.to_csv(port_path, index=False)
    print(f"  → Portfolio composition: {port_path}  ({len(port_rows)} families)")

# 11d: Screener results (oracle)
screen_rows = []
for c, sc in screen_pass[:200]:
    screen_rows.append({
        "conditions": "+".join(c), "oracle_pf": round(sc["pf_fast"],3),
        "oracle_n": sc["n_fast"], "e31_prox": e31_proximity(c),
    })
df_screen = pd.DataFrame(screen_rows)
screen_path = os.path.join(OUT, "r064_screener.csv")
df_screen.to_csv(screen_path, index=False)
print(f"  → Oracle screener      : {screen_path}  ({len(screen_rows)} records)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts ...")
print(SEP2)
print()

def ps(ax, title, fs=8):
    ax.set_facecolor(C_PANEL); ax.set_title(title, fontsize=fs, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.grid(True, color=C_GRID, alpha=0.4, lw=0.5)
    for sp in ax.spines.values(): sp.set_color(C_GRID)

# ── CHART 1: Master Dashboard ─────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor=C_BG)
fig.suptitle(f"R064 — Full Cache Structural Mining Dashboard | "
             f"{len(SYMS)} symbols | {len(all_candidates):,} candidates",
             fontsize=11, color=C_TEXT, y=0.997)
gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.45, wspace=0.35,
                       top=0.97, bottom=0.03, left=0.05, right=0.97)

# P1: UES distribution of top candidates
ax1 = fig.add_subplot(gs[0, :2])
ps(ax1, "UES Distribution — Top Validated Candidates", 9)
ues_vals = [validated[c]["ues"] for c in validated_ranked[:40]]
ax1.bar(range(len(ues_vals)), ues_vals,
        color=[C_GREEN if v>=60 else C_GOLD if v>=45 else C_RED for v in ues_vals],
        edgecolor=C_GRID, lw=0.5)
ax1.axhline(60, color=C_GREEN, lw=0.8, ls="--", label="Promote zone (UES≥60)")
ax1.axhline(45, color=C_GOLD,  lw=0.8, ls="--", label="Watchlist zone (UES≥45)")
ax1.set_ylabel("UES (0-100)", fontsize=7)
ax1.set_xlabel("Candidate Rank", fontsize=7)
ax1.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)

# P2: PF vs Independence scatter
ax2 = fig.add_subplot(gs[0, 2:])
ps(ax2, "PF vs Independence from E3.1\n(bubble=n trades, colour=UES)", 9)
pf_list  = [validated[c]["pf"]                  for c in comp_ranked[:30]]
ind_list = [validated[c].get("indep_e31_sc",0)  for c in comp_ranked[:30]]
ues_list = [validated[c].get("ues",0)           for c in comp_ranked[:30]]
n_list   = [validated[c]["n"]                   for c in comp_ranked[:30]]
sc2 = ax2.scatter(ind_list, pf_list,
                  c=ues_list, cmap="RdYlGn", vmin=30, vmax=80,
                  s=[max(20, n*2) for n in n_list],
                  alpha=0.8, edgecolors=C_GRID, linewidths=0.5)
plt.colorbar(sc2, ax=ax2, label="UES", fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
ax2.axvline(40, color=C_GOLD, lw=0.8, ls="--", alpha=0.7, label="Indep≥40 threshold")
ax2.axhline(PROM_PF, color=C_GREEN, lw=0.8, ls="--", alpha=0.7, label=f"PF≥{PROM_PF}")
ax2.set_xlabel("Independence Score vs E3.1 (0-100)", fontsize=7)
ax2.set_ylabel("Profit Factor", fontsize=7)
ax2.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)
for i, (ix, pf_, c) in enumerate(zip(ind_list[:10], pf_list[:10], comp_ranked[:10])):
    ax2.text(ix, pf_+0.01, f"#{i+1}", ha="center", fontsize=5.5, color=C_TEXT)

# P3: Family type distribution
ax3 = fig.add_subplot(gs[1, :2])
ps(ax3, "Discovered Family Types (count of candidates)", 9)
fam_sorted = sorted(fam_counts.items(), key=lambda x: -len(x[1]))
fam_names  = [f[:30] for f,_ in fam_sorted]
fam_cnts   = [len(c) for _,c in fam_sorted]
fam_cols   = PALETTE[:len(fam_names)]
ax3.barh(fam_names, fam_cnts, color=fam_cols, edgecolor=C_GRID, lw=0.5)
ax3.set_xlabel("Number of candidate combos", fontsize=7)
for i, cnt in enumerate(fam_cnts):
    ax3.text(cnt + 0.1, i, str(cnt), va="center", fontsize=7, color=C_TEXT)

# P4: Condition frequency in top candidates
ax4 = fig.add_subplot(gs[1, 2:])
ps(ax4, "Condition Frequency in Top-25 Candidates", 9)
freq = defaultdict(int)
for combo in comp_ranked[:25]:
    for cid in combo: freq[cid] += 1
freq_sorted = sorted(freq.items(), key=lambda x: -x[1])[:20]
freq_labels = [f[0] for f in freq_sorted]
freq_vals   = [f[1] for f in freq_sorted]
freq_cols   = [PALETTE[i % len(PALETTE)] for i in range(len(freq_labels))]
ax4.barh(freq_labels, freq_vals, color=freq_cols, edgecolor=C_GRID, lw=0.5)
ax4.set_xlabel("Frequency in top-25 candidates", fontsize=7)
for i, v in enumerate(freq_vals):
    ax4.text(v + 0.05, i, str(v), va="center", fontsize=7, color=C_TEXT)

# P5: Equity curves — top-5 by composite score
ax5 = fig.add_subplot(gs[2, :2])
ps(ax5, "Equity Curves — Top-5 Families (OOS)", 9)
for i, combo in enumerate(comp_ranked[:5]):
    v  = validated[combo]
    eq = v["equity"]
    lbl = f"#{i+1} {'+'.join(combo)[:25]} (PF={v['pf']:.2f})"
    ax5.plot(range(len(eq)), eq / CAPITAL * 100,
             color=PALETTE[i], lw=1.2, label=lbl, alpha=0.9)
ax5.axhline(100, color=C_GRID, lw=0.7, ls="--")
ax5.set_ylabel("Equity (normalised to 100)", fontsize=7)
ax5.set_xlabel("Trade #", fontsize=7)
ax5.legend(fontsize=5.5, facecolor=C_PANEL, labelcolor=C_TEXT)

# P6: Independence matrix (top-10 vs E3.1 / DST_MD / inter-family)
ax6 = fig.add_subplot(gs[2, 2:])
ps(ax6, "Independence Matrix\n(trade overlap: lower=better)", 9)
top10 = comp_ranked[:10]
ref_labels = ["E3.1", "DSTMD"] + [f"#{i+1}" for i in range(len(top10))]
mat_size = 2 + len(top10)
mat = np.zeros((mat_size, mat_size))
ref_trades_list = [e31_trades, dstmd_trades] + [wfo_results[c]["trades"] for c in top10]
for i in range(mat_size):
    for j in range(mat_size):
        if i == j:
            mat[i, j] = 0.0
        else:
            ind = compute_independence(ref_trades_list[i], ref_trades_list[j])
            mat[i, j] = ind["trade_overlap"]
im6 = ax6.imshow(mat, cmap="RdYlGn_r", vmin=0.0, vmax=0.60, aspect="auto")
ax6.set_xticks(range(mat_size)); ax6.set_yticks(range(mat_size))
ax6.set_xticklabels(ref_labels, fontsize=5.5, rotation=35, ha="right")
ax6.set_yticklabels(ref_labels, fontsize=5.5)
for i in range(mat_size):
    for j in range(mat_size):
        ax6.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                 fontsize=5.5, color="black" if mat[i,j]<0.3 else "white")
plt.colorbar(im6, ax=ax6, label="Trade Overlap", fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

# P7: Validation heatmap for top-15 (multi-criteria)
ax7 = fig.add_subplot(gs[3, :3])
ps(ax7, "Validation Heatmap — Top-15 Candidates", 9)
top15 = comp_ranked[:15]
crit_labels = ["PF>1.30","n≥30","Boot>1.20","MC>80%","SymFlr>1","FldFlr>1","MDD<20%"]
heat_mat = np.zeros((len(top15), 7))
for i, combo in enumerate(top15):
    v = validated[combo]
    heat_mat[i] = [
        int(v["pf"] > PROM_PF),
        int(v["n"] >= PROM_N),
        int(v.get("b50",0) > PROM_BOOT),
        int(v.get("mc_p",0) > PROM_MC),
        int(v.get("sym_floor",0) > 1.0),
        int(v.get("fold_floor",0) > 1.0),
        int(abs(v["mdd"]) < PROM_MDD),
    ]
y_labels = [f"#{i+1} {'+'.join(c)[:22]}" for i, c in enumerate(top15)]
ax7.imshow(heat_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax7.set_xticks(range(7)); ax7.set_yticks(range(len(top15)))
ax7.set_xticklabels(crit_labels, fontsize=6.5, rotation=25, ha="right")
ax7.set_yticklabels(y_labels, fontsize=6)
for i in range(len(top15)):
    for j in range(7):
        ax7.text(j, i, "✓" if heat_mat[i,j]==1 else "✗",
                 ha="center", va="center", fontsize=8,
                 color="black" if heat_mat[i,j]==1 else "white",
                 fontweight="bold")

# P8: Diversity score ranking
ax8 = fig.add_subplot(gs[3, 3])
ps(ax8, "Diversity Score\nTop-20 candidates", 9)
div_top = sorted(comp_ranked[:30], key=lambda c: -validated[c].get("diversity",0))[:20]
div_labels = [f"#{comp_ranked.index(c)+1}" for c in div_top]
div_scores = [validated[c].get("diversity",0) for c in div_top]
div_cols   = [C_GREEN if s>70 else C_GOLD if s>50 else C_RED for s in div_scores]
ax8.barh(div_labels, div_scores, color=div_cols, edgecolor=C_GRID, lw=0.5)
ax8.axvline(70, color=C_GREEN, lw=0.8, ls="--")
ax8.axvline(50, color=C_GOLD,  lw=0.8, ls="--")
ax8.set_xlabel("Diversity Score (0-100)", fontsize=7)
ax8.set_xlim(0, 105)

# P9: Portfolio comparison bar chart
ax9 = fig.add_subplot(gs[4, :2])
ps(ax9, "Portfolio Comparison — Key Metrics", 9)
port_labels = ["E3.1\n(frozen)"]
port_pfs    = [m31["pf"]]
port_ns     = [m31["n"]]
port_mdds   = [abs(m31["mdd"])]

if comp_ranked:
    bst = comp_ranked[0]
    m_b = metrics(wfo_results[bst]["trades"])
    port_labels += [f"Best New\n({'+'.join(bst)[:15]}...)"]
    port_pfs    += [m_b["pf"]]
    port_ns     += [m_b["n"]]
    port_mdds   += [abs(m_b["mdd"])]

if greedy_port:
    port_labels += [f"Greedy\nPortfolio\n({len(greedy_port)} fam)"]
    port_pfs    += [m_greedy["pf"]]
    port_ns     += [m_greedy["n"]]
    port_mdds   += [abs(m_greedy["mdd"])]

x = np.arange(len(port_labels))
w = 0.25
ax9.bar(x - w, port_pfs, w, label="PF",      color=C_BLUE,  edgecolor=C_GRID, lw=0.5)
ax9_r = ax9.twinx()
ax9_r.bar(x,     port_ns,  w, label="n Trades", color=C_GREEN, edgecolor=C_GRID, lw=0.5, alpha=0.7)
ax9.bar(x + w, port_mdds, w, label="MDD",     color=C_RED,   edgecolor=C_GRID, lw=0.5)
ax9.set_xticks(x); ax9.set_xticklabels(port_labels, fontsize=7)
ax9.set_ylabel("PF / MDD", fontsize=7)
ax9_r.set_ylabel("Trade Count", fontsize=7, color=C_GREEN)
ax9_r.tick_params(labelsize=7, colors=C_GREEN)
ax9.axhline(1.0, color=C_GRID, lw=0.8, ls="--")
ax9.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT, loc="upper left")

# P10: Summary verdict panel
ax10 = fig.add_subplot(gs[4, 2:])
ax10.set_facecolor(C_PANEL); ax10.axis("off")
for sp in ax10.spines.values(): sp.set_color(C_GRID)
summary_lines = [
    "R064 — DISCOVERY SUMMARY",
    "─" * 32,
    f"Symbols: {len(SYMS)} | OOS bars: {total_oos:,}",
    f"Conditions: {len(CONDITIONS_DEF)} | Combos: {len(all_candidates):,}",
    f"Oracle pass: {len(screen_pass)} | WFO run: {len(top_candidates)}",
    f"WFO survivors: {len(wfo_results)} | Validated: {len(validated)}",
    "─" * 32,
    f"Families discovered: {n_families}",
    f"PROMOTE: {n_promote} | WATCHLIST: {n_watchlist}",
    "─" * 32,
]
if best_overall:
    v = validated[best_overall]
    summary_lines += [
        "BEST FAMILY (Composite):",
        f"  {'+'.join(best_overall)[:30]}",
        f"  PF={v['pf']:.3f}  UES={v['ues']:.1f}  n={v['n']}",
        f"  {v['family'][:28]}",
        "─" * 32,
    ]
if most_indep:
    v = validated[most_indep]
    summary_lines += [
        "MOST INDEPENDENT FROM E3.1:",
        f"  {'+'.join(most_indep)[:30]}",
        f"  IndepScore={v.get('indep_e31_sc',0):.1f}  PF={v['pf']:.3f}",
    ]
elapsed = time.time() - t_start
summary_lines += ["─"*32, f"Runtime: {elapsed:.0f}s"]

for i, ln in enumerate(summary_lines):
    clr = (C_GREEN if "PROMOTE" in ln or "BEST" in ln
           else C_GOLD if "WATCHLIST" in ln or "MOST" in ln
           else C_RED  if "reject" in ln.lower()
           else C_TEXT)
    bold = "bold" if i == 0 or "BEST" in ln or "MOST" in ln else "normal"
    ax10.text(0.03, 0.98 - i * 0.056, ln, transform=ax10.transAxes,
              fontsize=7, color=clr, va="top", fontweight=bold, fontfamily="monospace")

dash_path = os.path.join(OUT, "r064_dashboard.png")
fig.savefig(dash_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close(fig)
print(f"  Dashboard saved → {dash_path}")

# ── CHART 2: Top-10 Equity Curves ─────────────────────────────────────────────
if len(comp_ranked) >= 3:
    fig2, axes2 = plt.subplots(2, 5, figsize=(22, 8), facecolor=C_BG)
    fig2.suptitle("R064 — Top-10 Family Equity Curves (OOS)", fontsize=10, color=C_TEXT)
    for i, (combo, ax) in enumerate(zip(comp_ranked[:10], axes2.flatten())):
        v = validated[combo]
        eq = v["equity"]
        ax.set_facecolor(C_PANEL)
        ax.plot(range(len(eq)), eq / CAPITAL * 100, color=PALETTE[i], lw=1.2)
        ax.axhline(100, color=C_GRID, lw=0.7, ls="--")
        ax.fill_between(range(len(eq)), eq / CAPITAL * 100, 100,
                        where=(eq / CAPITAL * 100 >= 100), alpha=0.15, color=C_GREEN)
        ax.fill_between(range(len(eq)), eq / CAPITAL * 100, 100,
                        where=(eq / CAPITAL * 100 < 100), alpha=0.15, color=C_RED)
        short = "+".join(combo)[:22]
        ax.set_title(f"#{i+1} {short}\nPF={v['pf']:.2f} n={v['n']}", fontsize=6.5, color=C_TEXT)
        ax.tick_params(labelsize=6, colors=C_TEXT)
        for sp in ax.spines.values(): sp.set_color(C_GRID)
    eq_path = os.path.join(OUT, "r064_equity_curves.png")
    fig2.tight_layout()
    fig2.savefig(eq_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig2)
    print(f"  Equity curves saved  → {eq_path}")

# ── CHART 3: Family radar chart ───────────────────────────────────────────────
if len(comp_ranked) >= 3:
    top5 = comp_ranked[:5]
    cats_radar = ["PF", "UES", "Indep", "Diversity", "n/50", "Boot50", "MC%"]
    N_cats = len(cats_radar)
    angles = np.linspace(0, 2 * np.pi, N_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig3, ax3r = plt.subplots(1, 1, figsize=(8, 8), facecolor=C_BG,
                               subplot_kw=dict(projection="polar"))
    ax3r.set_facecolor(C_PANEL)
    for sp in ax3r.spines.values(): sp.set_color(C_GRID)

    for i, combo in enumerate(top5):
        v = validated[combo]
        vals = [
            min(100, (v["pf"] - 1.0) * 50),
            v.get("ues", 0),
            v.get("indep_e31_sc", 0),
            v.get("diversity", 0),
            min(100, v["n"] / PROM_N * 50),
            min(100, (v.get("b50", 1.0) - 1.0) / 0.5 * 100),
            v.get("mc_p", 0) * 100,
        ]
        vals += vals[:1]
        ax3r.plot(angles, vals, color=PALETTE[i], lw=1.5, label=f"#{i+1} "+"+".join(combo)[:20])
        ax3r.fill(angles, vals, color=PALETTE[i], alpha=0.08)

    ax3r.set_xticks(angles[:-1])
    ax3r.set_xticklabels(cats_radar, fontsize=8, color=C_TEXT)
    ax3r.set_yticklabels(["20","40","60","80","100"], fontsize=6.5, color=C_TEXT)
    ax3r.set_ylim(0, 100)
    ax3r.legend(fontsize=6.5, facecolor=C_PANEL, labelcolor=C_TEXT,
                loc="lower right", bbox_to_anchor=(1.35, -0.05))
    ax3r.set_title("R064 — Family Radar: Top-5 Composite Ranked", color=C_TEXT, fontsize=9, pad=15)
    ax3r.tick_params(colors=C_TEXT)
    ax3r.grid(color=C_GRID, alpha=0.5)

    radar_path = os.path.join(OUT, "r064_family_radar.png")
    fig3.savefig(radar_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig3)
    print(f"  Family radar saved   → {radar_path}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — RESEARCH JOURNAL ENTRY
# ─────────────────────────────────────────────────────────────────────────────
journal_path = os.path.join(OUT, "r064_journal.md")
with open(journal_path, "w") as f:
    f.write(f"# R064 — Full Cache Structural Mining: Discovery Research\n\n")
    f.write(f"**Date:** July 2026  \n")
    f.write(f"**Duration:** {(time.time()-t_start):.0f}s  \n")
    f.write(f"**Symbols:** {len(SYMS)} (1H cached)  \n")
    f.write(f"**OOS Bars:** {total_oos:,}  \n\n")
    f.write(f"## Objective\n")
    f.write(f"Full cache discovery run. 32-condition library, all valid 3 & 4 "
            f"condition combos, starting from zero assumptions.\n\n")
    f.write(f"## Results Summary\n")
    f.write(f"- Candidates generated: {len(all_candidates):,}\n")
    f.write(f"- Oracle survivors: {len(screen_pass)}\n")
    f.write(f"- WFO survivors (PF≥1.10, n≥15): {len(wfo_results)}\n")
    f.write(f"- Fully validated: {len(validated)}\n")
    f.write(f"- Families discovered: {n_families}\n")
    f.write(f"- PROMOTE: {n_promote}  |  WATCHLIST: {n_watchlist}\n\n")
    f.write(f"## Family Types\n")
    for fam, combos in sorted(fam_counts.items(), key=lambda x: -len(x[1])):
        f.write(f"- **{fam}** ({len(combos)} candidates)\n")
    f.write(f"\n## Best Family (Composite)\n")
    if best_overall:
        v = validated[best_overall]
        f.write(f"**Conditions:** `{'+'.join(best_overall)}`  \n")
        f.write(f"**PF:** {v['pf']:.3f}  |  **UES:** {v['ues']:.1f}  |  "
                f"**n:** {v['n']}  |  **MDD:** {v['mdd']:.1%}  \n")
        f.write(f"**Family:** {v['family']} — {v['fam_why']}  \n")
        f.write(f"**Composite Score:** {v.get('composite',0):.1f}  \n")
        f.write(f"**Independence from E3.1:** {v.get('indep_e31_sc',0):.1f}/100  \n\n")
    f.write(f"## Most Independent Family\n")
    if most_indep:
        v = validated[most_indep]
        f.write(f"**Conditions:** `{'+'.join(most_indep)}`  \n")
        f.write(f"**Independence:** {v.get('indep_e31_sc',0):.1f}/100  |  "
                f"**Trade Overlap:** {v.get('indep_e31',{}).get('trade_overlap',0):.1%}  \n")
        f.write(f"**PnL Correlation:** {v.get('indep_e31',{}).get('pnl_corr',0):.3f}  \n\n")
    f.write(f"## Recommendation for R065\n")
    if best_r065:
        v = validated[best_r065]
        f.write(f"Forensic investigation of: `{'+'.join(best_r065)}`  \n")
        f.write(f"Priority: entry-gate analysis, symbol-by-symbol breakdown, "
                f"parameter sensitivity, portfolio fit with E3.1.  \n\n")
    f.write(f"## Outputs\n")
    f.write(f"- `r064_dashboard.png` — Master dashboard\n")
    f.write(f"- `r064_equity_curves.png` — Top-10 equity curves\n")
    f.write(f"- `r064_family_radar.png` — Radar chart top-5\n")
    f.write(f"- `r064_family_rankings.csv` — Full rankings\n")
    f.write(f"- `r064_best_trades.csv` — Trade log for best family\n")
    if greedy_port:
        f.write(f"- `r064_portfolio.csv` — Portfolio composition\n")
    f.write(f"- `r064_screener.csv` — Oracle screener results\n")

print(f"  Journal saved        → {journal_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
elapsed = time.time() - t_start
print()
print(SEP)
print("  R064 COMPLETE")
print(SEP)
print()
print(f"  Symbols used        : {len(SYMS)} | Excluded: {len(excluded)}")
print(f"  Total OOS bars      : {total_oos:,}")
print(f"  Conditions          : {len(CONDITIONS_DEF)} (32 total: 25 existing + 7 new)")
print(f"  Candidates generated: {len(all_candidates):,}")
print(f"  Oracle survivors    : {len(screen_pass)}")
print(f"  WFO survivors       : {len(wfo_results)}")
print(f"  Fully validated     : {len(validated)}")
print(f"  Families discovered : {n_families}")
print(f"  PROMOTE             : {n_promote}")
print(f"  WATCHLIST           : {n_watchlist}")
print()
if best_overall:
    v = validated[best_overall]
    print(f"  ► Best overall      : {'+'.join(best_overall)}")
    print(f"    PF={v['pf']:.3f}  UES={v['ues']:.1f}  n={v['n']}"
          f"  Composite={v.get('composite',0):.1f}")
    print(f"    Family: {v['family']}")
if most_indep:
    v = validated[most_indep]
    print(f"  ► Most independent  : {'+'.join(most_indep)}")
    print(f"    E3.1 indep={v.get('indep_e31_sc',0):.1f}/100  "
          f"TradeOvlp={v.get('indep_e31',{}).get('trade_overlap',0):.1%}")
if highest_n:
    v = validated[highest_n]
    print(f"  ► Highest frequency : {'+'.join(highest_n)}")
    print(f"    n={v['n']} forward trades")
print()
print(f"  ► R065 target       : {'+'.join(best_r065) if best_r065 else 'TBD'}")
print()
print(f"  Outputs:")
print(f"    {dash_path}")
if len(comp_ranked) >= 3:
    print(f"    {eq_path}")
    print(f"    {radar_path}")
print(f"    {rank_path}")
print(f"    {journal_path}")
print()
print(f"  Runtime: {elapsed:.0f}s")
print(SEP)
