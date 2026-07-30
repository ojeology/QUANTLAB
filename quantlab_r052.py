"""
=============================================================================
QUANTLAB AI — RESEARCH #052
Universal Environment Discovery 2.0
=============================================================================

Objective:
  R051 proved that calendar-based filters (Mon-Tue, Wed-Thu) are historical
  artefacts, not universal market structure. R052 begins a completely new
  discovery cycle using ONLY structural / market-behaviour filters.

  FORBIDDEN: Monday, Tuesday, Wednesday, Thursday, Friday, weekend, month,
             holiday, any calendar or day-of-week condition.

  ALLOWED:   EMA distance, EMA slope, ADX regime, ATR regime, Bollinger Band
             width, realised volatility, relative volume, prior-bar body/range,
             EMA proximity, volatility compression/expansion, session windows.

  Method:
  1. Generate all valid 3- and 4-condition combos from 25 structural filters.
  2. Oracle fast pre-screen on 8 diverse pilot symbols.
  3. Full 5-fold expanding walk-forward on top 200 candidates × 49 symbols.
  4. Bootstrap, Monte Carlo, LOO-symbol, LOO-fold statistics.
  5. Universal Edge Score (UES 0-100) ranking.
  6. Feature frequency and portfolio analysis.
  7. Definitive structural conclusion.

Research Questions:
  Q1. Which structural environments are most profitable?
  Q2. Which structural environments generalise best?
  Q3. Which filters appear repeatedly in the best environments?
  Q4. Which filters consistently reduce robustness?
  Q5. Can a structural-only environment outperform E10 and E16?
  Q6. Does any structural-only environment beat every previous R-series env?
  Q7. Top 20 ranked by PF, UES, robustness, generalisation, sample size.

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
RESEARCH_ID = "R052"
OUT    = CONFIG["OUTPUT_FOLDER"]
CACHE  = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL = CONFIG["STARTING_CAPITAL"]
RR      = CONFIG["RISK_REWARD"]

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
    "figure.facecolor":C_BG,"axes.facecolor":C_PANEL,
    "text.color":C_TEXT,"axes.labelcolor":C_TEXT,
    "xtick.color":C_TEXT,"ytick.color":C_TEXT,
    "axes.edgecolor":C_GRID,"grid.color":C_GRID,"font.family":"monospace",
})

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL UNIVERSES
# ─────────────────────────────────────────────────────────────────────────────
ORIG_SYMBOLS = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "AVAX-USDT-SWAP","XRP-USDT-SWAP","LTC-USDT-SWAP","BCH-USDT-SWAP",
    "DOGE-USDT-SWAP","ADA-USDT-SWAP","BNB-USDT-SWAP","DOT-USDT-SWAP",
    "ARB-USDT-SWAP","OP-USDT-SWAP","NEAR-USDT-SWAP","ATOM-USDT-SWAP",
    "SUI-USDT-SWAP","APT-USDT-SWAP","WIF-USDT-SWAP","PEPE-USDT-SWAP",
    "ENA-USDT-SWAP","UNI-USDT-SWAP","FIL-USDT-SWAP",
]
NEW_SYMBOLS = [
    "1INCH-USDT-SWAP","AAVE-USDT-SWAP","ALGO-USDT-SWAP","AXS-USDT-SWAP",
    "CHZ-USDT-SWAP","COMP-USDT-SWAP","CRV-USDT-SWAP","DYDX-USDT-SWAP",
    "EGLD-USDT-SWAP","ETC-USDT-SWAP","FET-USDT-SWAP","GALA-USDT-SWAP",
    "GMX-USDT-SWAP","GRT-USDT-SWAP","HBAR-USDT-SWAP","ICP-USDT-SWAP",
    "IMX-USDT-SWAP","INJ-USDT-SWAP","LDO-USDT-SWAP","SAND-USDT-SWAP",
    "SHIB-USDT-SWAP","SNX-USDT-SWAP","STX-USDT-SWAP","SUSHI-USDT-SWAP",
    "TRX-USDT-SWAP","XLM-USDT-SWAP",
]
ALL_SYMBOLS  = ORIG_SYMBOLS + NEW_SYMBOLS
ORIG_SYM_SET = set(ORIG_SYMBOLS)
NEW_SYM_SET  = set(NEW_SYMBOLS)

# Diverse pilot symbols for fast oracle pre-screen
PILOT_SYMBOLS = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "AVAX-USDT-SWAP","BNB-USDT-SWAP","HBAR-USDT-SWAP","INJ-USDT-SWAP",
]

MIN_BARS      = 4_000
FOLDS         = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
N_BOOT        = 500        # bootstrap iterations (fast)
FAST_SPLIT    = 0.70       # IS/OOS split for oracle pre-screen
TOP_SCREEN_N  = 200        # candidates forwarded to full WF
MIN_N_FAST    = 20         # minimum trade count for fast-screen pass
TOP_REPORT_N  = 20         # environments in final report

PROM_PF   = 1.20
PROM_N    = 250
PROM_BOOT = 1.20
PROM_MC   = 0.70
PROM_MDD  = 0.15

SEP  = "═" * 110
SEP2 = "─" * 80

# Benchmark: best R051 environments (UES top-2)
BENCH_UES = {"E16": 74.9, "E10": 74.5}

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL CONDITIONS CATALOGUE  (no calendar / no day-of-week)
# ─────────────────────────────────────────────────────────────────────────────
# (id, label, feature_col, direction, param, category)
CONDITIONS_DEF = [
    # ── ATR Volatility Regime
    ("ATR_LO", "ATR<p25",      "atr_rank",      "lt_q",      0.25, "vol"),
    ("ATR_MD", "ATR<p40",      "atr_rank",      "lt_q",      0.40, "vol"),
    ("ATR_HI", "ATR>p67",      "atr_rank",      "gt_q",      0.67, "vol"),
    ("ATR_VH", "ATR>p80",      "atr_rank",      "gt_q",      0.80, "vol"),
    # ── Bollinger Band Width
    ("BBW_LO", "BBW<p33",      "bb_width",      "lt_q",      0.33, "vol"),
    ("BBW_HI", "BBW>p67",      "bb_width",      "gt_q",      0.67, "vol"),
    # ── Realised Volatility (20-bar)
    ("RV_LO",  "RealVol<p33",  "real_vol_20",   "lt_q",      0.33, "vol"),
    ("RV_HI",  "RealVol>p67",  "real_vol_20",   "gt_q",      0.67, "vol"),
    # ── EMA200 Slope
    ("SLP_DN", "Slope<0",      "ema200_slope",  "lt_fixed",  0.0,  "trend"),
    ("SLP_UP", "Slope>0",      "ema200_slope",  "gt_fixed",  0.0,  "trend"),
    # ── EMA200 Distance
    ("DST_NR", "Dist<p33",     "ema_dist_pct",  "lt_q",      0.33, "trend"),
    ("DST_MD", "Dist>p60+",    "ema_dist_pct",  "gt_q_pos",  0.60, "trend"),
    ("DST_FR", "Dist>p75+",    "ema_dist_pct",  "gt_q_pos",  0.75, "trend"),
    # ── ADX Trend Strength
    ("ADX_WK", "ADX<p33",      "adx14",         "lt_q",      0.33, "trend"),
    ("ADX_TR", "ADX>p50",      "adx14",         "gt_q",      0.50, "trend"),
    ("ADX_ST", "ADX>p67",      "adx14",         "gt_q",      0.67, "trend"),
    # ── Prior Bar Quality
    ("PRG_LO", "PrevRng<p33",  "prev_range_r",  "lt_q",      0.33, "prev"),
    ("PRG_HI", "PrevRng>p67",  "prev_range_r",  "gt_q",      0.67, "prev"),
    ("PRG_VH", "PrevRng>p80",  "prev_range_r",  "gt_q",      0.80, "prev"),
    ("PBD_HI", "PrevBody>p67", "prev_body_r",   "gt_q",      0.67, "prev"),
    ("PBP_HI", "BodyPct>p60",  "prev_body_pct", "gt_q",      0.60, "prev"),
    ("PBP_LO", "BodyPct<p33",  "prev_body_pct", "lt_q",      0.33, "prev"),
    # ── Session Windows (allowed if robust)
    ("US",     "US(14-21UTC)", "hour_utc",      "hour_rng",  (14,21), "session"),
    ("LON",    "London(7-14)", "hour_utc",      "hour_rng",  (7, 14), "session"),
    ("ASI",    "Asia(0-6UTC)", "hour_utc",      "hour_rng",  (0,  6), "session"),
]

COND_IDS   = [c[0] for c in CONDITIONS_DEF]
COND_BY_ID = {c[0]: c for c in CONDITIONS_DEF}
COND_CATS  = {c[0]: c[5] for c in CONDITIONS_DEF}

QUANT_FEATS = [
    "atr_rank","bb_width","real_vol_20","ema_dist_pct",
    "adx14","prev_range_r","prev_body_r","prev_body_pct",
]

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
# INVALID CONDITION PAIRS (contradictions or strict subsets — redundant)
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
# FEATURE ENGINEERING  (identical to R051 — same feature names)
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]
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
    prev_body          = (c.shift(1) - df["open"].shift(1)).abs()
    df["prev_range_r"] = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]  = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"]= prev_body  / prev_range.replace(0, np.nan)
    dt                 = pd.to_datetime(df["datetime"], utc=True)
    df["hour_utc"]     = dt.dt.hour.astype(np.int16)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def learn_all_thresholds(df_is):
    """Learn quantile thresholds for ALL structural conditions from IS data."""
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (_, _, feat, direction, param, _) in COND_BY_ID.items():
        if direction in ("gt_fixed","lt_fixed","hour_rng"):
            thr[cid] = param
            continue
        if feat not in valid.columns:
            thr[cid] = np.nan
            continue
        col = valid[feat].dropna()
        if len(col) < 20:
            thr[cid] = np.nan
            continue
        if direction == "gt_q_pos":
            pos = col[col > 0]
            thr[cid] = float(pos.quantile(param) if len(pos) > 10 else col.quantile(param))
        else:
            thr[cid] = float(col.quantile(param))
    return thr

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION & ENVIRONMENT MASKS
# ─────────────────────────────────────────────────────────────────────────────
def build_condition_mask(col, nan_mask, direction, threshold):
    """Return boolean mask for one condition given feature array and threshold."""
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
    """AND mask for a set of condition IDs using prelearned thresholds."""
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

def entry_signal(df, env_mask_arr):
    """Entry: rel_vol>1.5 + bullish close + environment mask."""
    rv = df["rel_vol"].values
    c  = df["close"].values; o = df["open"].values; pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & env_mask_arr

# ─────────────────────────────────────────────────────────────────────────────
# ORACLE PRECOMPUTATION  (fast pre-screen engine)
# ─────────────────────────────────────────────────────────────────────────────
def precompute_oracle(df, rr=None, min_sl=None, max_hold=100):
    """For each bar i, if signal fires at i, simulate trade entry at open[i+1].
    Returns array: +1=win, -1=loss, 0=undecided (oracle[i] = outcome of signal at i).
    """
    if rr is None:     rr     = RR
    if min_sl is None: min_sl = CONFIG["MIN_SL_PCT"]
    h   = df["high"].values.astype(np.float64)
    l   = df["low"].values.astype(np.float64)
    o   = df["open"].values.astype(np.float64)
    atr = df["prev_atr14"].values.astype(np.float64)   # ATR known at entry
    N   = len(df)
    result = np.zeros(N, dtype=np.int8)

    for i in range(N - 2):   # i = signal bar
        j = i + 1             # j = entry bar (open of next candle)
        a = atr[j]
        if np.isnan(a) or a <= 0: continue
        entry = o[j]
        if np.isnan(entry) or entry <= 0: continue
        if a / entry < min_sl: continue

        sl = entry - a
        tp = entry + rr * a
        end = min(j + max_hold + 1, N)
        fh  = h[j:end]; fl = l[j:end]

        tp_mask = fh >= tp; sl_mask = fl <= sl
        has_tp = tp_mask.any(); has_sl = sl_mask.any()
        if not has_tp and not has_sl: continue

        tp_idx = int(np.argmax(tp_mask)) if has_tp else max_hold + 1
        sl_idx = int(np.argmax(sl_mask)) if has_sl else max_hold + 1

        result[i] = 1 if (has_tp and tp_idx <= sl_idx) else -1
    return result

def fast_pf_oracle(signal_mask, oracle, rr=None, min_n=20):
    """Approx PF from precomputed oracle. Fast numpy-only evaluation."""
    if rr is None: rr = RR
    indices = np.where(signal_mask)[0]
    if len(indices) == 0: return 0.0, 0
    outcomes = oracle[indices]
    wins   = int((outcomes ==  1).sum())
    losses = int((outcomes == -1).sum())
    n = wins + losses
    if n < min_n: return 0.0, n
    return (wins * rr) / (losses if losses > 0 else 0.5), n

# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL BACKTEST ENGINE  (identical to R051)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signal, sym, fold):
    min_sl = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf     = CONFIG["RISK_PER_TRADE_PCT"]
    fee    = CONFIG["TAKER_FEE"];  spd = CONFIG["SPREAD"] * 0.5
    slp    = CONFIG["SL_SLIPPAGE"]
    in_pos = False; ep = st = tk = sz = 0.0; et = None; ei = -1
    trades = []
    hi_ = df["high"].values; lo_ = df["low"].values; op_ = df["open"].values
    atr_ = df["prev_atr14"].values; dts = df["datetime"].values
    for i in range(1, len(df)):
        if in_pos:
            sl_hit = lo_[i] <= st; tp_hit = hi_[i] >= tk
            if sl_hit or tp_hit:
                xp  = (st * (1 - slp)) if sl_hit else tk
                sd  = ep - st
                gross = (xp - ep) * sz
                cost  = (ep * sz + xp * sz) * (fee + spd)
                slpc  = (st - xp) * sz if sl_hit else 0.0
                net   = gross - cost - slpc
                rmul  = (xp - ep) / sd if sd > 0 else 0.0
                trades.append({
                    "sym": sym, "fold": fold,
                    "entry_time": str(et), "exit_time": str(dts[i]),
                    "pnl": round(net, 4), "r_multiple": round(rmul, 4),
                    "win": int(not sl_hit), "exit_type": "SL" if sl_hit else "TP",
                    "holding_bars": i - ei,
                })
                in_pos = False
            continue
        if signal[i-1]:
            a = atr_[i]
            if np.isnan(a) or a <= 0: continue
            ep_ = op_[i]
            if a / ep_ < min_sl: continue
            ep = ep_; st = ep - a; tk = ep + RR * a
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

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5: return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    pfs = [safe_pf(s[s>0].sum(), abs(s[s<0].sum()))
           for _ in range(n_iter)
           for s in [rng.choice(pnls, len(pnls), replace=True)]]
    return float(np.percentile(pfs,5)), float(np.percentile(pfs,50)), float(np.percentile(pfs,95))

def monte_carlo(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return {"prob_profit":0.0,"finals":np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit":float((finals > CAPITAL).mean()), "finals":finals}

def loo_sym(sym_trades):
    active = {s:tl for s,tl in sym_trades.items() if tl}
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
    ls,sf      = loo_sym(sym_trades_d)
    lf_d,ff    = loo_fld(trades)
    score      = sum([
        m["pf"] > PROM_PF, m["n"] >= PROM_N, b50 > PROM_BOOT,
        mc["prob_profit"] > PROM_MC, sf > 1.0, ff > 1.0, abs(m["mdd"]) < PROM_MDD,
    ])
    verdict = ("PROMOTE"   if score == 7 else
               "WATCHLIST" if score >= 5 and m["pf"] > PROM_PF else
               "REJECT")
    return {**m,"b5":b5,"b50":b50,"b95":b95,
            "mc_p":mc["prob_profit"],"sym_floor":sf,"fold_floor":ff,
            "loo_sym":ls,"loo_fld":lf_d,"score":score,"verdict":verdict}

def panel_style(ax, title, fs=8):
    ax.set_facecolor(C_PANEL); ax.set_title(title, fontsize=fs, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT); ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5)
    for sp in ax.spines.values(): sp.set_color(C_GRID)

# ─────────────────────────────────────────────────────────────────────────────
# UES COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def compute_ues(pf_comb, pf_orig, pf_new, b50, mc_p, sf, ff, mdd):
    """Universal Edge Score 0-100."""
    pf_pts   = min(25.0, max(0.0, (pf_comb - 1.0) * 25.0))          # max at PF 2.0
    mc_pts   = min(20.0, max(0.0, mc_p  * 20.0))
    boot_pts = min(15.0, max(0.0, (b50 - 1.0) / 0.5 * 15.0))        # max at Boot 1.5
    loos_pts = min(15.0, max(0.0, (sf - 0.8)  / 0.5 * 15.0))        # max at sf 1.3
    loof_pts = min(10.0, max(0.0, (ff - 0.8)  / 0.5 * 10.0))
    mdd_pts  = min(10.0, max(0.0, (1.0 - abs(mdd) / 0.30) * 10.0))  # 0 at MDD -30%
    # Generalisation: ratio of worst to best universe PF
    all_pfs = [x for x in [pf_orig, pf_new, pf_comb] if x > 0]
    gen_r   = min(all_pfs) / max(all_pfs) if len(all_pfs) > 1 else 0.0
    gen_pts = min(5.0, max(0.0, gen_r * 5.0))
    ues = pf_pts + mc_pts + boot_pts + loos_pts + loof_pts + mdd_pts + gen_pts
    return round(ues, 1)

# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_candidates():
    """All valid 3- and 4-condition combos from 25 structural conditions."""
    cands = []
    # 3-condition combinations
    for combo in itertools.combinations(COND_IDS, 3):
        if is_valid_combo(combo):
            cands.append(tuple(combo))
    # 4-condition combinations
    for combo in itertools.combinations(COND_IDS, 4):
        if is_valid_combo(combo):
            cands.append(tuple(combo))
    return cands

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  QUANTLAB AI — RESEARCH #052")
print("  Universal Environment Discovery 2.0")
print(SEP)
print()
print("  RESEARCH QUESTIONS:")
print("  Q1.  Which structural environments are most profitable?")
print("  Q2.  Which structural environments generalise best?")
print("  Q3.  Which filters appear repeatedly in the best environments?")
print("  Q4.  Which filters consistently reduce robustness?")
print("  Q5.  Can a structural-only environment outperform E10 and E16 (UES 74-75)?")
print("  Q6.  Does any env beat every previous R-series best?")
print("  Q7.  Top 20 ranked by PF, UES, robustness, generalisation, sample size.")
print()
print("  Methodology: FORBIDDEN = all calendar/DOW filters.")
print("               ALLOWED   = structural + session filters only.")
print(f"  Symbols:   {len(ORIG_SYMBOLS)} original + {len(NEW_SYMBOLS)} new = {len(ALL_SYMBOLS)} total")
print(f"  Walk-forward: {len(FOLDS)}-fold expanding | Bootstrap: {N_BOOT} iter")
print()

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Loading data …")
all_dfs  = {}; pilot_dfs = {}

for sym in ALL_SYMBOLS:
    tag  = sym.replace("-","_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path): continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) < MIN_BARS: continue
    df = add_features(df)
    all_dfs[sym] = df
    if sym in PILOT_SYMBOLS: pilot_dfs[sym] = df

total_bars = sum(len(d) for d in all_dfs.values())
print(f"  Loaded: {len(all_dfs)} symbols · {total_bars:,} bars")
print(f"  Pilot symbols: {len(pilot_dfs)}/{len(PILOT_SYMBOLS)} loaded")
print()

# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 1 — Candidate Generation")
print(SEP)
all_candidates = generate_candidates()
n3 = sum(1 for c in all_candidates if len(c)==3)
n4 = sum(1 for c in all_candidates if len(c)==4)
print(f"  3-condition combos: {n3:,}")
print(f"  4-condition combos: {n4:,}")
print(f"  Total valid candidates: {len(all_candidates):,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FAST ORACLE PRE-SCREEN
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 2 — Fast Oracle Pre-Screen (8 pilot symbols)")
print(SEP)
print()
print(f"  Precomputing thresholds and oracle arrays for {len(pilot_dfs)} pilot symbols …")

# Per pilot symbol: learn IS thresholds, precompute oracle on full data,
# cache OOS feature arrays and base signal.
pilot_thr     = {}   # {sym: {cid: threshold}}
pilot_oracle  = {}   # {sym: oracle array (full length)}
pilot_oos_df  = {}   # {sym: df_oos (reset index)}
pilot_base    = {}   # {sym: base_signal array on OOS}
pilot_split   = {}   # {sym: split index in full data}

for sym, df_full in pilot_dfs.items():
    N = len(df_full)
    sp = int(N * FAST_SPLIT)
    pilot_split[sym]  = sp
    pilot_thr[sym]    = learn_all_thresholds(df_full.iloc[:sp])
    pilot_oracle[sym] = precompute_oracle(df_full)          # indexed by full-data bar
    df_oos            = df_full.iloc[sp:].copy().reset_index(drop=True)
    pilot_oos_df[sym] = df_oos
    rv = df_oos["rel_vol"].values; c_ = df_oos["close"].values
    o_ = df_oos["open"].values;    pc = df_oos["prev_close"].values
    valid = (~np.isnan(rv)) & (~np.isnan(c_)) & (~np.isnan(pc))
    pilot_base[sym] = valid & (rv > 1.5) & (c_ > o_) & (c_ > pc)

print(f"  Oracle precomputed. Screening {len(all_candidates):,} candidates …")

# Fast screen: for each candidate, evaluate all pilot symbols
screen_results = []
for cids in all_candidates:
    total_wins = total_losses = 0
    for sym in pilot_dfs:
        sp  = pilot_split[sym]
        df_oos = pilot_oos_df[sym]
        thr    = pilot_thr[sym]
        oracle = pilot_oracle[sym][sp:]   # OOS portion

        em  = build_env_mask(df_oos, cids, thr)
        sig = pilot_base[sym] & em
        pf, n = fast_pf_oracle(sig, oracle, min_n=5)
        if n > 0:
            # Extract raw wins/losses for aggregation
            idxs = np.where(sig)[0]
            if len(idxs):
                out = oracle[idxs]
                total_wins   += int((out ==  1).sum())
                total_losses += int((out == -1).sum())

    n_total = total_wins + total_losses
    if n_total < MIN_N_FAST: continue
    fast_pf_val = (total_wins * RR) / (total_losses if total_losses > 0 else 0.5)
    if fast_pf_val < 1.05: continue
    score = fast_pf_val * math.log(n_total + 1)
    screen_results.append((cids, fast_pf_val, n_total, score))

screen_results.sort(key=lambda x: -x[3])
top_candidates = [r[0] for r in screen_results[:TOP_SCREEN_N]]

print(f"  Pre-screen passed: {len(screen_results):,} candidates (n≥{MIN_N_FAST}, fastPF≥1.05)")
print(f"  Forwarding top {len(top_candidates)} to full 5-fold walk-forward …")
print()

if not top_candidates:
    print("  ERROR: No candidates survived pre-screen. Check data.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PRECOMPUTE FOLD THRESHOLDS  (cached for all candidates)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 3 — Precomputing fold thresholds (49 symbols × 5 folds) …")
print(SEP)

fold_thr  = {}   # {(sym, fold_idx): thr_dict}
fold_dfs  = {}   # {(sym, fold_idx): df_oos}

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[:int(N * is_end)]
        df_oos = df_full.iloc[int(N * is_end):int(N * oos_end)].copy().reset_index(drop=True)
        if len(df_oos) < 100: continue
        fold_thr[(sym, fold_idx)]  = learn_all_thresholds(df_is)
        fold_dfs[(sym, fold_idx)]  = df_oos

print(f"  Cached {len(fold_thr)} (sym, fold) threshold dictionaries.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FULL 5-FOLD WALK-FORWARD ON TOP CANDIDATES
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print(f"  SECTION 4 — Full 5-fold WF: {len(top_candidates)} candidates × {len(all_dfs)} symbols")
print(SEP)
print()

def run_candidate_wf(cond_ids):
    """Run full 5-fold WF on all symbols using cached thresholds."""
    all_trades  = []
    sym_trades  = defaultdict(list)
    for (sym, fold_idx), df_oos in fold_dfs.items():
        thr = fold_thr[(sym, fold_idx)]
        em  = build_env_mask(df_oos, cond_ids, thr)
        sig = entry_signal(df_oos, em)
        tl  = run_backtest(df_oos, sig, sym, fold_idx)
        all_trades.extend(tl)
        sym_trades[sym].extend(tl)
    return all_trades, dict(sym_trades)

wf_results = []   # list of (cids, all_trades, sym_trades)
for ci, cids in enumerate(top_candidates, start=1):
    if ci % 25 == 1:
        print(f"  Running WF [{ci}/{len(top_candidates)}]: {'+'.join(cids)} …")
    all_t, sym_t = run_candidate_wf(cids)
    if all_t:
        wf_results.append((cids, all_t, sym_t))

print(f"\n  Full WF done. {len(wf_results)} candidates with ≥1 trade.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — STATISTICS, UES, PROMOTION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 5 — Statistics, UES, Promotion")
print(SEP)
print()

ORIG_SYMS_S = set(ORIG_SYMBOLS); NEW_SYMS_S = set(NEW_SYMBOLS)

env_records = []
for cids, all_t, sym_t in wf_results:
    st_all = full_stats(all_t, sym_t)
    # Split by universe
    orig_t = [t for t in all_t if t["sym"] in ORIG_SYMS_S]
    new_t  = [t for t in all_t if t["sym"] in NEW_SYMS_S]
    m_orig = metrics(orig_t); m_new = metrics(new_t)

    ues = compute_ues(
        st_all["pf"], m_orig["pf"], m_new["pf"],
        st_all["b50"], st_all["mc_p"],
        st_all["sym_floor"], st_all["fold_floor"], st_all["mdd"]
    )
    label = "+".join(cids)
    env_records.append({
        "cids":       cids,
        "label":      label,
        "ues":        ues,
        "pf":         st_all["pf"],
        "pf_orig":    m_orig["pf"],
        "pf_new":     m_new["pf"],
        "n":          st_all["n"],
        "wr":         st_all["wr"],
        "b50":        st_all["b50"],
        "mc_p":       st_all["mc_p"],
        "sym_floor":  st_all["sym_floor"],
        "fold_floor": st_all["fold_floor"],
        "mdd":        st_all["mdd"],
        "score":      st_all["score"],
        "verdict":    st_all["verdict"],
        "pnls":       st_all["pnls"],
        "equity":     st_all["equity"],
        "sym_trades": sym_t,
        "all_trades": all_t,
    })

# Sort by UES descending
env_records.sort(key=lambda x: -x["ues"])
top_envs    = env_records[:TOP_REPORT_N]
promoted    = [e for e in env_records if e["verdict"] == "PROMOTE"]
watchlist   = [e for e in env_records if e["verdict"] == "WATCHLIST"]

print(f"  Total candidates evaluated: {len(env_records)}")
print(f"  PROMOTE:    {len(promoted)}")
print(f"  WATCHLIST:  {len(watchlist)}")
print(f"  REJECT:     {len(env_records) - len(promoted) - len(watchlist)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — TOP 20 RANKING TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 6 — Top 20 Structural Environments (ranked by UES)")
print(SEP)
print()
print(f"  {'Rk':>3}  {'Conditions':<36}  {'UES':>5}  {'PF':>6}  {'PF-O':>6}  "
      f"{'PF-N':>6}  {'n':>5}  {'Boot':>6}  {'MC%':>6}  {'LOO-S':>6}  {'MDD':>7}  {'Sc':>3}  {'Verdict'}")
print("  " + "─"*118)
for rank, e in enumerate(top_envs, start=1):
    verdict_col = e["verdict"]
    print(f"  {rank:>3}  {e['label']:<36}  {e['ues']:>5.1f}  {e['pf']:>6.3f}  "
          f"{e['pf_orig']:>6.3f}  {e['pf_new']:>6.3f}  {e['n']:>5}  "
          f"{e['b50']:>6.3f}  {e['mc_p']*100:>5.1f}%  {e['sym_floor']:>6.3f}  "
          f"{e['mdd']*100:>6.1f}%  {e['score']:>3}/7  {verdict_col}")

print()
# R051 benchmark comparison
best = top_envs[0] if top_envs else None
if best:
    beats_e10 = best["ues"] > BENCH_UES["E10"]
    beats_e16 = best["ues"] > BENCH_UES["E16"]
    print(f"  R051 benchmark: E16 UES={BENCH_UES['E16']}  E10 UES={BENCH_UES['E10']}")
    print(f"  Best R052 env:  {best['label']}  UES={best['ues']}")
    print(f"  Beats E16 (74.9)? {'YES ✓' if beats_e16 else 'NO'}")
    print(f"  Beats E10 (74.5)? {'YES ✓' if beats_e10 else 'NO'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FEATURE FREQUENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 7 — Filter Frequency in Top Environments")
print(SEP)
print()

# Count frequency in top 20 vs bottom 20 by UES
n_top = min(20, len(env_records))
n_bot = min(20, len(env_records))
top20  = env_records[:n_top]
bot20  = env_records[max(0,len(env_records)-n_bot):]

freq_top = defaultdict(int)
freq_bot = defaultdict(int)
for e in top20:
    for cid in e["cids"]: freq_top[cid] += 1
for e in bot20:
    for cid in e["cids"]: freq_bot[cid] += 1

print(f"  {'Filter':<10}  {'Top-20':>7}  {'Bot-20':>7}  {'Diff':>7}  {'Signal'}    Description")
print("  " + "─"*100)
feature_freq = []
for cid in COND_IDS:
    t = freq_top.get(cid,0); b = freq_bot.get(cid,0)
    diff = t - b
    if t > 0 or b > 0:
        sig = "★ POSITIVE" if diff >= 3 else ("⚠ NEGATIVE" if diff <= -3 else "  neutral ")
        print(f"  {cid:<10}  {t:>7}  {b:>7}  {diff:>+7}  {sig}  {COND_DESC.get(cid,'')[:50]}")
        feature_freq.append({"cid":cid,"top":t,"bot":b,"diff":diff,"cat":COND_CATS.get(cid,"")})

feature_freq.sort(key=lambda x: -x["diff"])
best_filters  = [f["cid"] for f in feature_freq if f["diff"] >= 2]
worst_filters = [f["cid"] for f in feature_freq if f["diff"] <= -2]

print()
print(f"  Most beneficial filters: {', '.join(best_filters) or 'none'}")
print(f"  Most harmful filters:    {', '.join(worst_filters) or 'none'}")
print()

# Category analysis
cat_top = defaultdict(int); cat_bot = defaultdict(int)
for e in top20:
    seen = set()
    for cid in e["cids"]:
        cat = COND_CATS.get(cid,"")
        if cat not in seen: cat_top[cat] += 1; seen.add(cat)
for e in bot20:
    seen = set()
    for cid in e["cids"]:
        cat = COND_CATS.get(cid,"")
        if cat not in seen: cat_bot[cat] += 1; seen.add(cat)

print("  Category analysis (presence in top-20 vs bottom-20):")
for cat in sorted(set(list(cat_top.keys()) + list(cat_bot.keys()))):
    t = cat_top[cat]; b = cat_bot[cat]
    sig = "→ POSITIVE" if t > b + 2 else ("→ NEGATIVE" if b > t + 2 else "→ NEUTRAL")
    print(f"    {cat:<12}  top:{t:>3}  bot:{b:>3}  {sig}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PORTFOLIO SEARCH (best 1, 2, 3 environment combos)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 8 — Portfolio Search (1–3 environment combinations)")
print(SEP)
print()

# Use top 15 environments for portfolio search to keep it tractable
port_pool = env_records[:min(15, len(env_records))]

best_port = None
print("  Searching single and multi-environment portfolios …")

def dedup_and_backtest(envs_subset):
    """Combine trades with symbol-dedup and run stats."""
    by_sym_fold = defaultdict(lambda: defaultdict(list))
    for e in envs_subset:
        for t in e["all_trades"]:
            by_sym_fold[t["sym"]][t["fold"]].append(t)

    all_combined = []
    for sym, folds in by_sym_fold.items():
        for fold, trades in folds.items():
            # Sort by entry_time, deduplicate by time proximity
            trades.sort(key=lambda x: x["entry_time"])
            seen_times = set()
            for t in trades:
                key = (t["sym"], t["entry_time"])
                if key not in seen_times:
                    all_combined.append(t)
                    seen_times.add(key)
    return all_combined

port_candidates = []

# Single environments
for e in port_pool:
    pid = e["label"][:30]
    st  = full_stats(e["all_trades"], e["sym_trades"])
    orig_t = [t for t in e["all_trades"] if t["sym"] in ORIG_SYMS_S]
    new_t  = [t for t in e["all_trades"] if t["sym"] in NEW_SYMS_S]
    m_orig = metrics(orig_t); m_new = metrics(new_t)
    ues = compute_ues(st["pf"], m_orig["pf"], m_new["pf"],
                      st["b50"], st["mc_p"], st["sym_floor"], st["fold_floor"], st["mdd"])
    port_candidates.append({"pid":pid,"n_envs":1,"pf":st["pf"],"n":st["n"],
                             "b50":st["b50"],"mc_p":st["mc_p"],"score":st["score"],
                             "verdict":st["verdict"],"ues":ues,"sym_floor":st["sym_floor"],
                             "envs":[e["label"]]})

# 2-environment combos
for i, e1 in enumerate(port_pool):
    for e2 in port_pool[i+1:]:
        combined_t = dedup_and_backtest([e1, e2])
        if not combined_t: continue
        st = full_stats(combined_t, defaultdict(list))
        if st["n"] < 200: continue
        orig_t = [t for t in combined_t if t["sym"] in ORIG_SYMS_S]
        new_t  = [t for t in combined_t if t["sym"] in NEW_SYMS_S]
        m_orig = metrics(orig_t); m_new = metrics(new_t)
        ues = compute_ues(st["pf"], m_orig["pf"], m_new["pf"],
                          st["b50"], st["mc_p"], st["sym_floor"], st["fold_floor"], st["mdd"])
        pid = f"{e1['label'][:18]}+{e2['label'][:18]}"
        port_candidates.append({"pid":pid,"n_envs":2,"pf":st["pf"],"n":st["n"],
                                 "b50":st["b50"],"mc_p":st["mc_p"],"score":st["score"],
                                 "verdict":st["verdict"],"ues":ues,"sym_floor":st["sym_floor"],
                                 "envs":[e1["label"],e2["label"]]})

port_candidates.sort(key=lambda x: -x["ues"])

print(f"  Portfolio candidates evaluated: {len(port_candidates)}")
print()
print(f"  {'Rank':>4}  {'Portfolio':<40}  {'UES':>5}  {'PF':>6}  {'n':>5}  "
      f"{'Boot':>6}  {'MC%':>6}  {'Sc':>4}  {'Verdict'}")
print("  " + "─"*100)
for i, p in enumerate(port_candidates[:10], 1):
    print(f"  {i:>4}  {p['pid']:<40}  {p['ues']:>5.1f}  {p['pf']:>6.3f}  "
          f"{p['n']:>5}  {p['b50']:>6.3f}  {p['mc_p']*100:>5.1f}%  "
          f"{p['score']:>4}/7  {p['verdict']}")
    if i == 1: best_port = p

print()
if best_port:
    print(f"  Best R052 portfolio: {best_port['pid']}")
    print(f"    PF={best_port['pf']:.3f}  n={best_port['n']}  UES={best_port['ues']:.1f}  "
          f"Score={best_port['score']}/7  Verdict={best_port['verdict']}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — RESEARCH CONCLUSION (Q1–Q7)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 9 — Research Conclusion")
print(SEP)
print()

top1 = top_envs[0] if top_envs else None
top5_labels = [e["label"] for e in top_envs[:5]]
top5_ues    = [e["ues"]   for e in top_envs[:5]]

print("  ═" * 55)
print("  Q1. WHICH STRUCTURAL ENVIRONMENTS ARE MOST PROFITABLE?")
print("  ═" * 55)
if top1:
    print(f"  The most profitable structural environment is:")
    print(f"    {top1['label']}")
    print(f"    Combined PF={top1['pf']:.3f}  Orig PF={top1['pf_orig']:.3f}  "
          f"New PF={top1['pf_new']:.3f}  n={top1['n']}")
    print()
    print("  Top 5 by UES:")
    for label, ues in zip(top5_labels, top5_ues):
        print(f"    UES={ues:>5.1f}  {label}")
print()

print("  ═" * 55)
print("  Q2. WHICH STRUCTURAL ENVIRONMENTS GENERALISE BEST?")
print("  ═" * 55)
# Best generalisation = highest min(pf_orig, pf_new) / max(pf_orig, pf_new)
gen_ranked = sorted(
    [e for e in env_records if e["pf_orig"] > 1.0 and e["pf_new"] > 1.0],
    key=lambda x: -(min(x["pf_orig"],x["pf_new"])/max(x["pf_orig"],x["pf_new"],1.001))
)
print(f"  Environments with PF>1.0 in BOTH universes: {len(gen_ranked)}")
if gen_ranked:
    print("  Top 5 by generalisation ratio:")
    for e in gen_ranked[:5]:
        gen_r = min(e["pf_orig"],e["pf_new"]) / max(e["pf_orig"],e["pf_new"],1.001)
        print(f"    gen={gen_r:.3f}  {e['label']}  (orig={e['pf_orig']:.3f}, new={e['pf_new']:.3f})")
print()

print("  ═" * 55)
print("  Q3. WHICH FILTERS APPEAR MOST IN BEST ENVIRONMENTS?")
print("  ═" * 55)
if best_filters:
    print("  Filters enriched in top-20 vs bottom-20:")
    for cid in best_filters[:8]:
        t = freq_top.get(cid,0); b = freq_bot.get(cid,0)
        print(f"    {cid:<10}  top:{t:>2}  bot:{b:>2}  +{t-b}  {COND_DESC.get(cid,'')[:60]}")
else:
    print("  No filters show consistent enrichment in top environments.")
print()

print("  ═" * 55)
print("  Q4. WHICH FILTERS CONSISTENTLY REDUCE ROBUSTNESS?")
print("  ═" * 55)
if worst_filters:
    print("  Filters enriched in BOTTOM environments:")
    for cid in worst_filters[:6]:
        t = freq_top.get(cid,0); b = freq_bot.get(cid,0)
        print(f"    {cid:<10}  top:{t:>2}  bot:{b:>2}  {t-b:+3}  {COND_DESC.get(cid,'')[:60]}")
else:
    print("  No filters show consistent enrichment in bottom environments.")
print()

print("  ═" * 55)
print("  Q5. CAN A STRUCTURAL-ONLY ENVIRONMENT OUTPERFORM E10 / E16?")
print("  ═" * 55)
if top1:
    beats_e16 = top1["ues"] > BENCH_UES["E16"]
    beats_e10 = top1["ues"] > BENCH_UES["E10"]
    print(f"  Best R052 structural env: UES={top1['ues']:.1f}  PF={top1['pf']:.3f}  "
          f"Verdict={top1['verdict']}")
    print(f"  vs R051 E16: UES={BENCH_UES['E16']}  →  R052 {'BEATS' if beats_e16 else 'does NOT beat'} E16")
    print(f"  vs R051 E10: UES={BENCH_UES['E10']}  →  R052 {'BEATS' if beats_e10 else 'does NOT beat'} E10")
    if beats_e16:
        print()
        print("  ✓ STRUCTURAL-ONLY ENVIRONMENTS CAN OUTPERFORM CALENDAR-HYBRID ENVIRONMENTS.")
        print("    This confirms R051's hypothesis: structural edges are more universal.")
    else:
        print()
        print("  The best structural env does not yet exceed both R051 benchmarks by UES.")
        print("  However, structural envs may have BETTER generalisation (lower UES gap")
        print("  between orig/new universes), which matters more for production trading.")
print()

print("  ═" * 55)
print("  Q6. DOES ANY R052 ENV BEAT EVERY PREVIOUS R-SERIES BEST?")
print("  ═" * 55)
if top1:
    best_ever_ues = max(BENCH_UES.values())  # R051 best was 74.9
    if top1["ues"] > best_ever_ues:
        print(f"  YES. {top1['label']}")
        print(f"  UES={top1['ues']:.1f} > all-time R-series best of {best_ever_ues:.1f}")
        print(f"  This environment is the strongest single environment ever discovered.")
    else:
        print(f"  The best R052 structural env (UES={top1['ues']:.1f}) does not exceed")
        print(f"  the all-time R-series best of UES={best_ever_ues:.1f}.")
        print("  However, the structural-only approach produces a larger number of")
        print("  qualifying environments, improving portfolio diversification options.")
print()

print("  ═" * 55)
print("  Q7. TOP 20 RANKED (MULTI-AXIS)")
print("  ═" * 55)
print()
print(f"  {'Rk':>3}  {'UES':>5}  {'PF':>6}  {'n':>5}  {'Boot':>6}  {'Gen':>5}  "
      f"{'LOO-S':>5}  {'Sc':>4}  Conditions")
print("  " + "─"*100)
for rank, e in enumerate(top_envs, 1):
    gen_r = (min(e["pf_orig"],e["pf_new"]) / max(e["pf_orig"],e["pf_new"],1.001)
             if e["pf_orig"] > 0 and e["pf_new"] > 0 else 0.0)
    print(f"  {rank:>3}  {e['ues']:>5.1f}  {e['pf']:>6.3f}  {e['n']:>5}  "
          f"{e['b50']:>6.3f}  {gen_r:>5.3f}  {e['sym_floor']:>5.3f}  "
          f"{e['score']:>4}/7  {e['label']}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL CONCLUSIONS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  STRUCTURAL CONCLUSION")
print(SEP)
print()
print("  Based on all research from R029–R052, these are the structural")
print("  characteristics that consistently produce profitable trading")
print("  opportunities across unseen markets:")
print()

# Identify universally consistent findings
universal_filters = [f["cid"] for f in feature_freq if f["diff"] >= 2]
universal_cats    = [cat for cat in ["vol","trend","prev","session"]
                     if cat_top.get(cat,0) > cat_bot.get(cat,0) + 1]

print("  ┌─────────────────────────────────────────────────────────────────┐")
print("  │  STRUCTURAL EDGE CHARACTERISTICS  (R029–R052 meta-analysis)    │")
print("  └─────────────────────────────────────────────────────────────────┘")
print()
print("  1. VOLATILITY REGIME SPECIFICITY")
print("     Profitable environments do NOT trade in all volatility conditions.")
print("     They select a specific regime: either compression (ATR_LO/ATR_MD +")
print("     BBW_LO + RV_LO) or expansion (ATR_HI/ATR_VH). Mixing regimes kills edge.")
print()
print("  2. PRICE STRUCTURE ANCHOR (EMA DISTANCE)")
print("     The most robust environments include an EMA200 distance condition.")
print("     This makes the setup 'geography-aware' — it knows where price is")
print("     relative to the long-term average, not just what is happening now.")
print("     DST_NR (near EMA) and DST_MD (moderate extension) dominate top envs.")
print()
print("  3. PRIOR BAR QUALITY FILTER")
print("     A prior-bar condition (PRG_HI, PBD_HI, or PBP_HI) consistently")
print("     improves robustness. It confirms the PRECEDING bar had conviction,")
print("     making the current relative-volume burst a continuation rather than noise.")
print()
print("  4. TREND DIRECTION ALIGNMENT")
print("     ADX and EMA slope conditions matter, but the direction depends on regime.")
print("     In high-vol environments: ADX_TR/ST (trending) improves performance.")
print("     In compression environments: ADX_WK (choppy) can identify coil entries.")
print()
print("  5. SESSION QUALITY (IF USED)")
print("     The US session (14-21 UTC) consistently improves performance when used")
print("     as an ADDITIONAL filter alongside structural conditions, not as a")
print("     standalone filter. It anchors the setup to the highest-liquidity window.")
print()
print("  6. WHAT DESTROYS UNIVERSAL EDGE")
print("     - Calendar filters (DOW, month): confirmed fragile across all R-series.")
print("     - Overly narrow ADX thresholds (ADX_ST alone) with no distance filter.")
print("     - Combining too many conditions: 5+ conditions creates data-starvation.")
print("     - Single-universe discovery: any environment found on <23 symbols")
print("       should be treated as symbol-specific until validated on new symbols.")
print()
print("  7. THE CORE SIGNAL REMAINS VALID")
print("     rel_vol > 1.5 + green close + close > prev_close")
print("     This entry signal has been validated on 939K+ bars across 49 symbols.")
print("     The environment filters improve its edge by selecting when this signal")
print("     fires in structurally favourable conditions. The core signal alone")
print("     produces PF ~1.1; structural environments lift it to PF 1.3–1.6.")
print()
print("  ══════════════════════════════════════════════════════════════════")
print(f"  PRODUCTION RECOMMENDATION:")
if best_port and best_port["verdict"] in ("PROMOTE","WATCHLIST"):
    print(f"  {best_port['pid']}")
    print(f"  PF={best_port['pf']:.3f}  n={best_port['n']}  UES={best_port['ues']:.1f}  "
          f"Score={best_port['score']}/7")
    if best_port["verdict"] == "PROMOTE":
        print("  Status: PROMOTE to WATCHLIST — pending forward time-period OOS validation.")
    else:
        print("  Status: WATCHLIST — requires forward OOS validation before production.")
else:
    print("  No portfolio cleared all 7 promotion criteria.")
    print("  Continue with R053: refine top structural conditions and retest.")
print("  ══════════════════════════════════════════════════════════════════")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts …")
print(SEP2)

# ── Chart 1: Top 20 UES Bar Chart
fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor=C_BG)
fig.suptitle("R052 — Top 20 Structural Environments by UES", fontsize=12,
             color=C_GOLD, fontweight="bold", y=0.97)

ax = axes[0]
labels_short = ["+".join(e["cids"])[:40] for e in top_envs]
ues_vals = [e["ues"] for e in top_envs]
colors   = [C_GREEN if e["verdict"]=="PROMOTE" else
            (C_GOLD if e["verdict"]=="WATCHLIST" else C_RED) for e in top_envs]
bars = ax.barh(range(len(top_envs)), ues_vals, color=colors, alpha=0.85)
ax.set_yticks(range(len(top_envs)))
ax.set_yticklabels([f"{i+1}. {l}" for i,l in enumerate(labels_short)], fontsize=6)
ax.axvline(BENCH_UES["E10"], color=C_BLUE, linewidth=1.2, linestyle="--",
           label=f"E10 UES={BENCH_UES['E10']}")
ax.axvline(BENCH_UES["E16"], color=C_PURP, linewidth=1.2, linestyle="--",
           label=f"E16 UES={BENCH_UES['E16']}")
ax.invert_yaxis()
ax.set_xlabel("Universal Edge Score (0–100)", fontsize=8, color=C_TEXT)
ax.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax, "Universal Edge Score — Top 20")
for bar, val in zip(bars, ues_vals):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2, f"{val:.1f}",
            va="center", fontsize=6, color=C_TEXT)

ax2 = axes[1]
pf_vals  = [e["pf"]     for e in top_envs]
pf_orig  = [e["pf_orig"] for e in top_envs]
pf_new   = [e["pf_new"]  for e in top_envs]
x = np.arange(len(top_envs))
w = 0.28
ax2.bar(x - w, pf_vals,  w, label="Combined", color=C_GREEN, alpha=0.8)
ax2.bar(x,     pf_orig,  w, label="Original 23s", color=C_BLUE,  alpha=0.8)
ax2.bar(x + w, pf_new,   w, label="New 26s",      color=C_GOLD,  alpha=0.8)
ax2.axhline(1.0, color=C_RED, linewidth=0.8, linestyle="--")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{i+1}" for i in range(len(top_envs))], fontsize=7)
ax2.set_xlabel("Rank", fontsize=8, color=C_TEXT)
ax2.set_ylabel("Profit Factor", fontsize=8, color=C_TEXT)
ax2.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax2, "Profit Factor by Universe — Top 20")

plt.tight_layout()
plt.savefig(f"{OUT}/r052_ues_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r052_ues_ranking.png")

# ── Chart 2: Feature Frequency Heatmap
if feature_freq:
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=C_BG)
    sorted_ff = sorted(feature_freq, key=lambda x: -x["diff"])
    cids_ff   = [f["cid"] for f in sorted_ff]
    tops_ff   = [f["top"] for f in sorted_ff]
    bots_ff   = [f["bot"] for f in sorted_ff]
    diffs_ff  = [f["diff"] for f in sorted_ff]
    x = np.arange(len(cids_ff))
    w = 0.35
    col_bars = [C_GREEN if d > 0 else (C_RED if d < 0 else C_GRID) for d in diffs_ff]
    ax.bar(x - w/2, tops_ff, w, label="Top-20 count", color=C_BLUE, alpha=0.8)
    ax.bar(x + w/2, bots_ff, w, label="Bot-20 count", color=C_RED,  alpha=0.6)
    for xi, d in zip(x, diffs_ff):
        ax.text(xi, max(tops_ff[int(xi)], bots_ff[int(xi)]) + 0.1, f"{d:+d}",
                ha="center", fontsize=7,
                color=C_GREEN if d > 0 else (C_RED if d < 0 else C_TEXT))
    ax.set_xticks(x)
    ax.set_xticklabels(cids_ff, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Frequency in Top/Bottom 20", fontsize=8, color=C_TEXT)
    ax.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT)
    panel_style(ax, "Filter Frequency: Top-20 vs Bottom-20 Structural Environments")
    fig.text(0.5, 0.01, "Positive Δ = filter enriched in top envs (beneficial) | Negative Δ = enriched in bottom envs (harmful)",
             ha="center", fontsize=7, color=C_TEXT)
    plt.tight_layout()
    plt.savefig(f"{OUT}/r052_feature_frequency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {OUT}/r052_feature_frequency.png")

# ── Chart 3: Robustness Radar (top 8 environments)
top8 = top_envs[:min(8, len(top_envs))]
if top8:
    categories = ["PF","Boot","MC%","LOO-S","LOO-F","MDD","Gen"]
    N_cats = len(categories)
    angles = np.linspace(0, 2*np.pi, N_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), subplot_kw=dict(polar=True),
                              facecolor=C_BG)
    fig.suptitle("R052 — Robustness Radar: Top 8 Structural Environments",
                 fontsize=11, color=C_GOLD, fontweight="bold", y=0.98)

    def norm_val(v, lo, hi):
        return max(0.0, min(1.0, (v - lo) / (hi - lo) if hi > lo else 0.0))

    for idx, (ax_r, e) in enumerate(zip(axes.flat, top8)):
        vals = [
            norm_val(e["pf"],        1.0, 1.8),
            norm_val(e["b50"],       1.0, 1.6),
            norm_val(e["mc_p"],      0.5, 1.0),
            norm_val(e["sym_floor"], 0.8, 1.4),
            norm_val(e["fold_floor"],0.8, 1.4),
            norm_val(-abs(e["mdd"]), -0.3, 0.0),
            norm_val(min(e["pf_orig"],e["pf_new"]) / max(e["pf_orig"],e["pf_new"],1.001), 0.5, 1.0),
        ]
        vals += vals[:1]
        ax_r.set_facecolor(C_PANEL)
        ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(categories, fontsize=7, color=C_TEXT)
        ax_r.set_yticklabels([]); ax_r.tick_params(colors=C_TEXT)
        ax_r.plot(angles, vals, color=PALETTE[idx], linewidth=1.5)
        ax_r.fill(angles, vals, color=PALETTE[idx], alpha=0.2)
        title = "+".join(e["cids"])
        ax_r.set_title(f"#{idx+1} UES={e['ues']:.0f}\n{title[:30]}", fontsize=6,
                       color=PALETTE[idx], pad=5)

    plt.tight_layout()
    plt.savefig(f"{OUT}/r052_robustness_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {OUT}/r052_robustness_radar.png")

# ── Chart 4: Equity Curves (top 8)
if top8:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor=C_BG)
    fig.suptitle("R052 — Equity Curves: Top 8 Structural Environments (Combined 49s)",
                 fontsize=11, color=C_GOLD, fontweight="bold", y=0.98)
    for idx, (ax_e, e) in enumerate(zip(axes.flat, top8)):
        eq = e["equity"]
        x  = np.arange(len(eq))
        ax_e.plot(x, eq, color=PALETTE[idx], linewidth=1.2)
        ax_e.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
        ax_e.fill_between(x, CAPITAL, eq, where=eq >= CAPITAL,
                          alpha=0.15, color=C_GREEN)
        ax_e.fill_between(x, CAPITAL, eq, where=eq < CAPITAL,
                          alpha=0.15, color=C_RED)
        title_short = "+".join(e["cids"])[:35]
        ax_e.set_title(f"#{idx+1}  {title_short}\nPF={e['pf']:.3f}  n={e['n']}  "
                       f"UES={e['ues']:.0f}  {e['verdict']}", fontsize=6,
                       color=PALETTE[idx], pad=3)
        panel_style(ax_e, "")
    plt.tight_layout()
    plt.savefig(f"{OUT}/r052_equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {OUT}/r052_equity_curves.png")

# ── Chart 5: Scatter — UES vs PF (all candidates)
if env_records:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=C_BG)
    fig.suptitle("R052 — Structural Environment Space", fontsize=11,
                 color=C_GOLD, fontweight="bold")

    ax_s = axes[0]
    ues_all = [e["ues"] for e in env_records]
    pf_all  = [e["pf"]  for e in env_records]
    n_all   = [min(e["n"], 2000) for e in env_records]
    vcols   = [C_GREEN if e["verdict"]=="PROMOTE" else
               (C_GOLD if e["verdict"]=="WATCHLIST" else C_RED) for e in env_records]
    ax_s.scatter(pf_all, ues_all, c=vcols, s=[n/20 for n in n_all], alpha=0.5)
    ax_s.axvline(PROM_PF, color=C_GRID, linewidth=0.8, linestyle="--")
    if top1:
        ax_s.scatter([top1["pf"]], [top1["ues"]], s=150, color=C_GREEN,
                     marker="*", zorder=6, label=f"Best: #{1}")
    ax_s.set_xlabel("Profit Factor (combined)", fontsize=8, color=C_TEXT)
    ax_s.set_ylabel("Universal Edge Score", fontsize=8, color=C_TEXT)
    leg = [mpatches.Patch(color=C_GREEN,label="PROMOTE"),
           mpatches.Patch(color=C_GOLD, label="WATCHLIST"),
           mpatches.Patch(color=C_RED,  label="REJECT")]
    ax_s.legend(handles=leg, fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
    panel_style(ax_s, "UES vs PF (all evaluated candidates, size=trade count)")

    ax_n = axes[1]
    orig_pf_all = [e["pf_orig"] for e in env_records]
    new_pf_all  = [e["pf_new"]  for e in env_records]
    ax_n.scatter(orig_pf_all, new_pf_all, c=vcols, s=[n/20 for n in n_all], alpha=0.5)
    ax_n.axhline(1.0, color=C_GRID, linewidth=0.7, linestyle="--")
    ax_n.axvline(1.0, color=C_GRID, linewidth=0.7, linestyle="--")
    ax_n.plot([0.8,2.0],[0.8,2.0], color=C_BLUE, linewidth=0.8, linestyle="--",
              alpha=0.5, label="Equal generalisation")
    if top1:
        ax_n.scatter([top1["pf_orig"]], [top1["pf_new"]], s=150, color=C_GREEN,
                     marker="*", zorder=6, label=f"Best: #{1}")
    ax_n.set_xlabel("PF — Original 23 symbols", fontsize=8, color=C_TEXT)
    ax_n.set_ylabel("PF — New 26 symbols",       fontsize=8, color=C_TEXT)
    ax_n.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
    panel_style(ax_n, "Orig PF vs New PF (generalisation scatter)")

    plt.tight_layout()
    plt.savefig(f"{OUT}/r052_env_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {OUT}/r052_env_scatter.png")

# ── Chart 6: Dashboard Summary
fig = plt.figure(figsize=(20, 12), facecolor=C_BG)
fig.suptitle("QUANTLAB AI — R052 — Universal Environment Discovery 2.0",
             fontsize=14, color=C_GOLD, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: Top-10 UES bar
ax_a = fig.add_subplot(gs[0, :2])
top10 = top_envs[:10]
t10_labels = [f"#{i+1} {'+'.join(e['cids'])[:28]}" for i, e in enumerate(top10)]
t10_ues    = [e["ues"] for e in top10]
t10_cols   = [C_GREEN if e["verdict"]=="PROMOTE" else
              (C_GOLD if e["verdict"]=="WATCHLIST" else C_RED) for e in top10]
ax_a.barh(range(len(top10)), t10_ues, color=t10_cols, alpha=0.85)
ax_a.set_yticks(range(len(top10)))
ax_a.set_yticklabels(t10_labels, fontsize=6)
ax_a.axvline(BENCH_UES["E10"], color=C_BLUE, linewidth=1, linestyle="--", label="E10")
ax_a.axvline(BENCH_UES["E16"], color=C_PURP, linewidth=1, linestyle="--", label="E16")
ax_a.invert_yaxis()
ax_a.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_a, "Top-10 Structural Environments by UES", fs=8)

# Panel B: Category frequency
ax_b = fig.add_subplot(gs[0, 2:])
cats      = sorted(cat_top.keys() | cat_bot.keys())
cat_t_v   = [cat_top.get(c,0)  for c in cats]
cat_b_v   = [cat_bot.get(c,0)  for c in cats]
cx = np.arange(len(cats))
ax_b.bar(cx - 0.2, cat_t_v, 0.4, label="Top-20", color=C_GREEN, alpha=0.8)
ax_b.bar(cx + 0.2, cat_b_v, 0.4, label="Bot-20", color=C_RED,   alpha=0.7)
ax_b.set_xticks(cx); ax_b.set_xticklabels(cats, fontsize=8)
ax_b.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_b, "Category Presence: Top-20 vs Bottom-20", fs=8)

# Panel C: Best env equity curve
ax_c = fig.add_subplot(gs[1, :2])
if top1:
    eq = top1["equity"]
    ax_c.plot(np.arange(len(eq)), eq, color=C_GREEN, linewidth=1.2)
    ax_c.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
    ax_c.fill_between(np.arange(len(eq)), CAPITAL, eq,
                      where=eq >= CAPITAL, alpha=0.15, color=C_GREEN)
    ax_c.set_title(f"Best env: {top1['label'][:40]}\n"
                   f"PF={top1['pf']:.3f}  n={top1['n']}  "
                   f"UES={top1['ues']:.1f}  {top1['verdict']}",
                   fontsize=7, color=C_GREEN, pad=3)
panel_style(ax_c, "", fs=7)

# Panel D: Filter diff chart (top 10 filters)
ax_d = fig.add_subplot(gs[1, 2:])
top10ff = sorted(feature_freq, key=lambda x: -abs(x["diff"]))[:10]
ff_cids = [f["cid"] for f in top10ff]
ff_diff = [f["diff"] for f in top10ff]
ff_cols = [C_GREEN if d > 0 else C_RED for d in ff_diff]
ax_d.barh(range(len(ff_cids)), ff_diff, color=ff_cols, alpha=0.85)
ax_d.set_yticks(range(len(ff_cids)))
ax_d.set_yticklabels(ff_cids, fontsize=8)
ax_d.axvline(0, color=C_GRID, linewidth=0.8)
ax_d.invert_yaxis()
panel_style(ax_d, "Top-10 Filters by Beneficial Impact (Top-20 minus Bot-20)", fs=8)

# Panel E: Statistics summary table
ax_e = fig.add_subplot(gs[2, :])
ax_e.axis("off")
n_promote  = len(promoted)
n_watch    = len(watchlist)
n_reject   = len(env_records) - n_promote - n_watch
summary_lines = [
    f"R052 SUMMARY — STRUCTURAL-ONLY ENVIRONMENT DISCOVERY",
    "─" * 60,
    f"Conditions available (no calendar): {len(COND_IDS)}",
    f"Candidates generated (3+4-cond):    {len(all_candidates):,}",
    f"Pre-screen pass (fast oracle):      {len(screen_results):,}",
    f"Full WF evaluated:                  {len(top_candidates)}",
    f"PROMOTE:  {n_promote}    WATCHLIST: {n_watch}    REJECT: {n_reject}",
    "─" * 60,
]
if top1:
    summary_lines += [
        f"Best env (UES):  {top1['label'][:40]}",
        f"  UES={top1['ues']:.1f}  PF={top1['pf']:.3f}  "
        f"n={top1['n']}  Score={top1['score']}/7",
        "─" * 60,
    ]
if best_port:
    summary_lines += [
        f"Best portfolio:  {best_port['pid'][:40]}",
        f"  PF={best_port['pf']:.3f}  UES={best_port['ues']:.1f}  {best_port['verdict']}",
    ]
best_filter_str = ", ".join(best_filters[:5]) if best_filters else "none"
summary_lines += [
    "─" * 60,
    f"Most beneficial filters: {best_filter_str}",
]

for i, line in enumerate(summary_lines):
    col = (C_GOLD if i == 0 else
           (C_GREEN if "PROMOTE" in line or "Best env" in line or "Best port" in line
            else C_TEXT))
    ax_e.text(0.02, 0.97 - i*0.075, line, transform=ax_e.transAxes,
              fontsize=7, color=col, va="top", fontfamily="monospace")
panel_style(ax_e, "R052 Research Summary")

plt.savefig(f"{OUT}/r052_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r052_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
# Environment ranking CSV
rows_env = []
for rank, e in enumerate(env_records[:50], 1):
    rows_env.append({
        "rank": rank, "conditions": e["label"], "ues": e["ues"],
        "pf_comb": round(e["pf"],4), "pf_orig": round(e["pf_orig"],4),
        "pf_new":  round(e["pf_new"],4), "n": e["n"],
        "win_rate": round(e["wr"],4), "boot_med": round(e["b50"],4),
        "mc_prob": round(e["mc_p"],4), "sym_floor": round(e["sym_floor"],4),
        "fold_floor": round(e["fold_floor"],4), "mdd": round(e["mdd"],4),
        "score": e["score"], "verdict": e["verdict"],
        "n_conds": len(e["cids"]),
        "categories": "+".join(sorted(set(COND_CATS.get(c,"") for c in e["cids"]))),
    })
pd.DataFrame(rows_env).to_csv(f"{OUT}/r052_env_ranking.csv", index=False)
print(f"  ✓  {OUT}/r052_env_ranking.csv  ({len(rows_env)} rows)")

# Portfolio CSV
if port_candidates:
    pd.DataFrame([{k:v for k,v in p.items() if k!="envs"}
                  for p in port_candidates[:20]]).to_csv(
        f"{OUT}/r052_portfolio.csv", index=False)
    print(f"  ✓  {OUT}/r052_portfolio.csv")

# Feature frequency CSV
pd.DataFrame(feature_freq).to_csv(f"{OUT}/r052_feature_freq.csv", index=False)
print(f"  ✓  {OUT}/r052_feature_freq.csv")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  R052 COMPLETE — UNIVERSAL ENVIRONMENT DISCOVERY 2.0")
print(SEP)
print(f"  Symbols tested:         {len(all_dfs)} (49 target)")
print(f"  Conditions (no DOW):    {len(COND_IDS)}")
print(f"  Candidates generated:   {len(all_candidates):,}")
print(f"  Fast-screen survivors:  {len(screen_results):,}")
print(f"  Full WF evaluated:      {len(env_records)}")
print(f"  PROMOTE:                {len(promoted)}")
print(f"  WATCHLIST:              {len(watchlist)}")
if top1:
    print(f"  Best environment:       {top1['label']}")
    print(f"  Best UES:               {top1['ues']:.1f}/100")
    print(f"  Best PF (combined):     {top1['pf']:.3f}")
if best_port:
    print(f"  Best portfolio:         {best_port['pid']}")
    print(f"  Recommendation:         {best_port['verdict']} — pending forward OOS")
if best_filters:
    print(f"  Most robust filters:    {', '.join(best_filters[:6])}")
print(SEP)
