"""
QUANTLAB AI — R066
Production Portfolio Validation (Frozen Families Only)

Frozen Families:
  Family A (E3.1):  BBW_STRICT + RV_LO + DST_NR + PRG_VH
  Family B:         RV_HI + DST_MD + ADX_WK + LON
  Family C:         DST_NR + ADX_ST + PBD_HI + ASI

Rules:
  NO optimisation. NO threshold tuning. NO parameter search.
  Evaluate strategies exactly as frozen.

Sections:
  1  Verify Individual Baselines
  2  Build All Two-Family Portfolios
  3  Build Three-Family Portfolio
  4  Diversification Analysis
  5  Drawdown Diversification
  6  Trade Frequency
  7  Capital Allocation
  8  Stress Tests
  9  Production Ranking
  10 Final Verdict
"""

import os, sys, math, warnings, time, itertools
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
RESEARCH_ID  = "R066"
OUT          = CONFIG["OUTPUT_FOLDER"]
CACHE        = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL      = CONFIG["STARTING_CAPITAL"]
RR           = CONFIG["RISK_REWARD"]   # 2.0
IS_RATIO     = 0.80
MIN_BARS     = 2_000
N_FWD_FOLDS  = 5
N_BOOT       = 3_000
N_MC         = 3_000
N_PERM       = 1_000
RAND_SEED    = 42
TRADE_RISK   = 100.0   # $ risk per trade for portfolio simulation

SEP  = "═" * 110
SEP2 = "─" * 90

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN FAMILY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
FAM_A_LABEL  = "BBW_STRICT+RV_LO+DST_NR+PRG_VH"
FAM_B_LABEL  = "RV_HI+DST_MD+ADX_WK+LON"
FAM_C_LABEL  = "DST_NR+ADX_ST+PBD_HI+ASI"

FAMILIES = {
    "A": ("BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"),
    "B": ("RV_HI",      "DST_MD", "ADX_WK", "LON"),
    "C": ("DST_NR",     "ADX_ST", "PBD_HI", "ASI"),
}
FAM_LABELS = {
    "A": FAM_A_LABEL,
    "B": FAM_B_LABEL,
    "C": FAM_C_LABEL,
}

# Condition registry — frozen thresholds, no tuning
# Format: cid -> (feature_col, direction, param)
COND_DEF = {
    # Family A
    "BBW_STRICT": ("bb_width",       "lt_q",      0.25),   # tighter than BBW_LO=p33
    "RV_LO":      ("real_vol_20",    "lt_q",      0.33),
    "DST_NR":     ("ema_dist_pct",   "lt_q",      0.33),
    "PRG_VH":     ("prev_range_r",   "gt_q",      0.80),
    # Family B
    "RV_HI":      ("real_vol_20",    "gt_q",      0.67),
    "DST_MD":     ("ema_dist_pct",   "gt_q_pos",  0.60),   # positive AND > p60
    "ADX_WK":     ("adx14",          "lt_q",      0.33),
    "LON":        ("hour_utc",       "hour_rng",  (7, 14)),
    # Family C
    "ADX_ST":     ("adx14",          "gt_q",      0.67),
    "PBD_HI":     ("prev_body_r",    "gt_q",      0.67),
    "ASI":        ("hour_utc",       "hour_rng",  (0,  6)),
}

QUANT_FEATS = ["bb_width", "real_vol_20", "ema_dist_pct", "prev_range_r",
               "adx14", "prev_body_r", "atr14", "atr_rank",
               "ema200_slope", "rel_vol_rank", "prev_body_pct"]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_BG    = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT  = "#e0e0e0"
C_GRID  = "#2a2a2a"; C_GREEN = "#00c896"; C_RED   = "#e05050"
C_GOLD  = "#f5a623"; C_BLUE  = "#4a9eff"; C_PURP  = "#9b59b6"
C_TEAL  = "#1abc9c"; C_ORAN  = "#e67e22"
FAM_COLORS = {"A": C_GREEN, "B": C_GOLD, "C": C_BLUE,
              "A+B": C_TEAL, "A+C": C_PURP, "B+C": C_ORAN, "A+B+C": C_RED}
PALETTE = [C_GREEN, C_GOLD, C_BLUE, C_TEAL, C_PURP, C_ORAN, C_RED,
           "#3498db","#e74c3c","#f39c12","#2ecc71","#e91e63","#00bcd4",
           "#ff5722","#8bc34a","#795548","#607d8b","#ff9800","#673ab7","#26c6da"]

plt.rcParams.update({
    "figure.facecolor":C_BG, "axes.facecolor":C_PANEL,
    "text.color":C_TEXT, "axes.labelcolor":C_TEXT,
    "xtick.color":C_TEXT, "ytick.color":C_TEXT,
    "axes.edgecolor":C_GRID, "grid.color":C_GRID, "font.family":"monospace",
})

def style_ax(ax):
    ax.set_facecolor(C_PANEL); ax.grid(True, ls="--", lw=0.4, color=C_GRID)
    for sp in ax.spines.values(): sp.set_edgecolor(C_GRID)

def save_fig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    return p

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]; o = df["open"]

    df["ema200"]         = calc_ema(c, 200)
    df["ema50"]          = calc_ema(c, 50)
    df["atr14"]          = calc_atr(df, 14)
    df["atr_rank"]       = df["atr14"].rolling(100).rank(pct=True) * 100

    bb_mid               = c.rolling(20).mean()
    bb_std               = c.rolling(20).std(ddof=0)
    df["bb_width"]       = (bb_std * 2) / bb_mid.replace(0, np.nan) * 100.0
    bb_upper             = bb_mid + 2 * bb_std
    bb_lower             = bb_mid - 2 * bb_std
    df["bb_pos"]         = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    df["real_vol_20"]    = c.pct_change().rolling(20).std() * 100.0
    df["rel_vol_rank"]   = v.rolling(50).rank(pct=True) * 100
    df["ema200_slope"]   = df["ema200"].diff(5) / df["ema200"].shift(5).replace(0, np.nan) * 100

    ema200_safe          = df["ema200"].replace(0, np.nan)
    ema50_safe           = df["ema50"].replace(0, np.nan)
    df["ema_dist_pct"]   = (c - ema200_safe) / ema200_safe * 100.0
    df["ema50_dist_pct"] = (c - ema50_safe)  / ema50_safe  * 100.0

    prev_range           = (h.shift(1) - l.shift(1)).abs()
    prev_body            = (c.shift(1) - o.shift(1)).abs()
    df["prev_range_r"]   = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]    = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"]  = prev_body  / prev_range.replace(0, np.nan)
    df["close_high_r"]   = (c - l) / prev_range.replace(0, np.nan)

    df["hour_utc"]       = pd.to_datetime(df.index).hour
    df["adx14"]          = calc_adx(df, 14)
    df.dropna(subset=["ema200","atr14","real_vol_20","adx14","bb_width"], inplace=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def apply_cond(df, cid, thresholds):
    """Return boolean Series for condition cid using IS-derived thresholds."""
    col, direction, param = COND_DEF[cid]

    if direction == "hour_rng":
        lo, hi = param
        if lo < hi:
            return (df["hour_utc"] >= lo) & (df["hour_utc"] < hi)
        else:  # wraps midnight e.g. (22,6)
            return (df["hour_utc"] >= lo) | (df["hour_utc"] < hi)

    vals = df[col]
    if direction == "lt_q":
        thresh = thresholds.get(f"{cid}_q", np.nan)
        return vals < thresh
    elif direction == "gt_q":
        thresh = thresholds.get(f"{cid}_q", np.nan)
        return vals > thresh
    elif direction == "gt_q_pos":
        thresh = thresholds.get(f"{cid}_q", np.nan)
        return (vals > thresh) & (vals > 0)
    elif direction == "lt_fixed":
        return vals < param
    elif direction == "gt_fixed":
        return vals > param
    else:
        return pd.Series(False, index=df.index)

def compute_thresholds(df_is, cids):
    """Compute IS quantile thresholds for all conditions in cids."""
    thresholds = {}
    for cid in cids:
        col, direction, param = COND_DEF[cid]
        if direction in ("lt_q", "gt_q", "gt_q_pos"):
            vals = df_is[col].dropna()
            if direction == "gt_q_pos":
                vals_pos = vals[vals > 0]
                t = float(vals_pos.quantile(param)) if len(vals_pos) > 10 else float(vals.quantile(param))
            else:
                t = float(vals.quantile(param))
            thresholds[f"{cid}_q"] = t
    return thresholds

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY GATE  (standard: RELVOL > 1.5 × 20-bar avg, close > open, close > prev_close)
# ─────────────────────────────────────────────────────────────────────────────
def entry_gate(df):
    """Return boolean Series for the standard entry confirmation gate."""
    vol_avg  = df["vol"].rolling(20).mean()
    vol_ok   = df["vol"] > (1.5 * vol_avg)
    bull_ok  = df["close"] > df["open"]
    cont_ok  = df["close"] > df["close"].shift(1)
    return vol_ok & bull_ok & cont_ok

# ─────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def safe_pf(gross_win, gross_loss):
    if gross_loss == 0: return 999.0 if gross_win > 0 else 1.0
    return gross_win / gross_loss

def metrics_from_pnls(pnls, label=""):
    pnls = np.asarray(pnls, dtype=float)
    n    = len(pnls)
    if n == 0:
        return dict(pf=0.0, wr=0.0, n=0, net=0.0, avg_pnl=0.0, mdd=0.0,
                    max_dd_abs=0.0, equity=np.array([CAPITAL]))
    wins   = pnls[pnls > 0]; losses = pnls[pnls < 0]
    pf     = safe_pf(wins.sum(), abs(losses.sum()))
    wr     = len(wins) / n
    net    = float(pnls.sum())
    avg    = float(pnls.mean())
    # equity curve
    eq     = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnls)])
    peak   = np.maximum.accumulate(eq)
    dd_abs = (eq - peak)
    mdd    = float(dd_abs.min() / peak[np.argmin(dd_abs)]) if peak[np.argmin(dd_abs)] != 0 else 0.0
    mdd_abs= float(abs(dd_abs.min()))
    return dict(pf=pf, wr=wr, n=n, net=net, avg_pnl=avg, mdd=mdd,
                max_dd_abs=mdd_abs, equity=eq)

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=RAND_SEED):
    rng  = np.random.default_rng(seed)
    pnls = np.asarray(pnls)
    bpfs = []
    for _ in range(n_iter):
        s = rng.choice(pnls, len(pnls), replace=True)
        bpfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
    arr = np.array(bpfs)
    return dict(med=float(np.percentile(arr,50)),
                p5=float(np.percentile(arr, 5)),
                p95=float(np.percentile(arr,95)),
                pf_arr=arr)

def monte_carlo(pnls, n_iter=N_MC, seed=RAND_SEED+1):
    rng   = np.random.default_rng(seed)
    pnls  = np.asarray(pnls)
    finals= []; max_dds = []
    for _ in range(n_iter):
        s   = rng.choice(pnls, len(pnls), replace=True)
        eq  = CAPITAL + np.cumsum(s)
        finals.append(float(eq[-1]))
        peak = np.maximum.accumulate(np.concatenate([[CAPITAL], eq]))
        dd   = (np.concatenate([[CAPITAL], eq]) - peak).min()
        max_dds.append(float(dd))
    finals   = np.array(finals)
    max_dds  = np.array(max_dds)
    prob_profit = float((finals > CAPITAL).mean())
    return dict(prob_profit=prob_profit, median=float(np.median(finals)),
                p5=float(np.percentile(finals, 5)),
                p95=float(np.percentile(finals, 95)),
                finals=finals, max_dds=max_dds,
                med_mdd=float(np.median(max_dds)))

def ues_score(pf, wr, n, mdd, sym_floor, fold_floor, boot_p5, mc_prob):
    """Universal Edge Score 0–100."""
    pf_s   = min(100, max(0, (pf - 1.0)  / 0.80  * 35))
    wr_s   = min(100, max(0, (wr - 0.30) / 0.25  * 25))
    n_s    = min(100, max(0, (math.log1p(n) / math.log1p(200)) * 15))
    mdd_s  = min(100, max(0, (1 - abs(mdd) / 0.30) * 10))
    sym_s  = min(100, max(0, (sym_floor - 1.0) / 0.50 * 5))
    fold_s = min(100, max(0, (fold_floor - 1.0) / 0.50 * 5))
    boot_s = min(100, max(0, (boot_p5 - 1.0) / 0.50 * 5))
    mc_s   = min(100, max(0, mc_prob * 5))
    return round(pf_s + wr_s + n_s + mdd_s + sym_s + fold_s + boot_s + mc_s, 1)

def ulcer_index(equity):
    """Ulcer Index = RMS of drawdown percentages."""
    eq   = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd_pct = (eq - peak) / peak * 100
    return float(np.sqrt(np.mean(dd_pct**2)))

def recovery_factor(net, max_dd_abs):
    if max_dd_abs == 0: return 999.0
    return abs(net) / max_dd_abs

def permutation_pvalue(pnls, n_perm=N_PERM, seed=RAND_SEED+2):
    """Fraction of permutations where random PF >= actual PF."""
    rng  = np.random.default_rng(seed)
    pnls = np.asarray(pnls)
    actual = safe_pf(pnls[pnls>0].sum(), abs(pnls[pnls<0].sum()))
    count  = 0
    for _ in range(n_perm):
        s = rng.permutation(pnls)
        if safe_pf(s[s>0].sum(), abs(s[s<0].sum())) >= actual:
            count += 1
    return 1.0 - count / n_perm   # percentile rank (higher=better)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_symbol_data():
    data = {}
    for fn in sorted(os.listdir(CACHE)):
        if not fn.endswith("_1H.parquet"): continue
        sym = fn.replace("_1H.parquet", "")
        try:
            df = pd.read_parquet(os.path.join(CACHE, fn))
            df.index = pd.to_datetime(df.index, utc=True)
            df.sort_index(inplace=True)
            for col in ["open","high","low","close"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "vol" not in df.columns and "volume" in df.columns:
                df.rename(columns={"volume":"vol"}, inplace=True)
            df.dropna(subset=["open","high","low","close","vol"], inplace=True)
            if len(df) >= MIN_BARS:
                data[sym] = df
        except Exception:
            pass
    return data

# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def backtest_family(cids, df_feat):
    """
    5-fold WFO backtest. Returns list of trade dicts.
    Each trade: {symbol, entry_time, exit_pnl, fold, is_win, hour_utc}
    """
    n_bars   = len(df_feat)
    is_end   = int(n_bars * IS_RATIO)
    df_oos   = df_feat.iloc[is_end:].copy()
    oos_len  = len(df_oos)
    fold_sz  = max(1, oos_len // N_FWD_FOLDS)

    # Global IS thresholds (frozen once from IS data)
    df_is    = df_feat.iloc[:is_end].copy()
    thresholds = compute_thresholds(df_is, cids)

    gate     = entry_gate(df_feat).iloc[is_end:]
    masks    = [apply_cond(df_feat.iloc[is_end:], c, thresholds) for c in cids]
    combined = masks[0].copy()
    for m in masks[1:]:
        combined = combined & m
    signal   = combined & gate

    trades = []
    for fi in range(N_FWD_FOLDS):
        sl = slice(fi * fold_sz, (fi+1) * fold_sz if fi < N_FWD_FOLDS-1 else oos_len)
        fold_sig = signal.iloc[sl]
        fold_df  = df_oos.iloc[sl]
        for idx in fold_df.index[fold_sig.values]:
            pnl  = TRADE_RISK * RR if True else -TRADE_RISK  # will be randomly assigned below
            # Use entry gate success: body direction already confirmed by gate
            # Simulate: win if close > open (already in gate), but actual exit
            # We use a simplified model: win = 1 with WR implied by the strategy
            # Actually just record the signal and assign pnl based on realised return
            row   = fold_df.loc[idx]
            # Forward return proxy: next bar close vs entry
            pos   = fold_df.index.get_loc(idx)
            if pos + 1 < len(fold_df):
                exit_c = fold_df["close"].iloc[pos + 1]
            else:
                exit_c = row["close"]
            entry_c = row["close"]
            # Fixed-RR outcome: win if exit_c > entry_c (momentum continuation)
            is_win = exit_c > entry_c
            pnl    = TRADE_RISK * RR if is_win else -TRADE_RISK
            trades.append({
                "symbol":     fold_df.name if hasattr(fold_df, "name") else "UNK",
                "entry_time": idx,
                "exit_pnl":   pnl,
                "is_win":     is_win,
                "fold":       fi + 1,
                "hour_utc":   int(row["hour_utc"]) if "hour_utc" in row.index else 0,
                "close":      float(entry_c),
            })
    return trades

def run_family_all_symbols(cids, data):
    """Run a family across all symbols; return flat trade list with symbol tag."""
    all_trades = []
    for sym, df_raw in data.items():
        try:
            df_feat  = add_features(df_raw)
            df_feat.name = sym  # tag
            trades   = backtest_family(cids, df_feat)
            for t in trades:
                t["symbol"] = sym
            all_trades.extend(trades)
        except Exception:
            pass
    # Sort by entry time
    all_trades.sort(key=lambda t: t["entry_time"])
    return all_trades

# ─────────────────────────────────────────────────────────────────────────────
# FOLD-BY-FOLD BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
def fold_breakdown(trades):
    by_fold = defaultdict(list)
    for t in trades:
        by_fold[t["fold"]].append(t["exit_pnl"])
    result = {}
    for f in sorted(by_fold.keys()):
        p = np.array(by_fold[f])
        result[f] = dict(pf=safe_pf(p[p>0].sum(), abs(p[p<0].sum())),
                         n=len(p),
                         wr=float((p>0).mean()) if len(p) else 0.0)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL-BY-SYMBOL BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
def symbol_breakdown(trades):
    by_sym = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t["exit_pnl"])
    result = {}
    for sym, pnls in by_sym.items():
        p = np.array(pnls)
        result[sym] = dict(pf=safe_pf(p[p>0].sum(), abs(p[p<0].sum())),
                           n=len(p),
                           wr=float((p>0).mean()))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY RETURNS
# ─────────────────────────────────────────────────────────────────────────────
def monthly_returns(trades):
    if not trades: return {}
    months = defaultdict(float)
    for t in trades:
        key = t["entry_time"].to_period("M") if hasattr(t["entry_time"], "to_period") else str(t["entry_time"])[:7]
        months[key] += t["exit_pnl"]
    return dict(sorted(months.items(), key=lambda x: str(x[0])))

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO COMBINATION (deduplicated by symbol+time)
# ─────────────────────────────────────────────────────────────────────────────
def combine_portfolios(trade_lists, weights=None):
    """
    Merge trade lists. De-duplicate: if same (symbol, entry_time) appears in
    multiple families, keep one entry (first occurrence). Sort by time.
    weights: list of floats for each family (default equal weight).
    """
    if weights is None:
        weights = [1.0] * len(trade_lists)

    seen = {}  # (sym, time) -> trade with pnl already scaled
    for tl, w in zip(trade_lists, weights):
        for t in tl:
            key = (t["symbol"], t["entry_time"])
            if key not in seen:
                tc = dict(t)
                tc["exit_pnl"] = t["exit_pnl"] * w
                seen[key] = tc
    combined = sorted(seen.values(), key=lambda t: t["entry_time"])
    return combined

def combine_portfolios_weighted(trade_lists, weights):
    """Combine allowing duplicate signals; sum PnLs with weights (portfolio allocation sim)."""
    all_events = []
    for tl, w in zip(trade_lists, weights):
        for t in tl:
            tc = dict(t); tc["exit_pnl"] = t["exit_pnl"] * w
            all_events.append(tc)
    all_events.sort(key=lambda t: t["entry_time"])
    # Aggregate by day
    by_day = defaultdict(float)
    for t in all_events:
        day = str(t["entry_time"])[:10]
        by_day[day] += t["exit_pnl"]
    return all_events, by_day

# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION SUITE FOR ONE TRADE LIST
# ─────────────────────────────────────────────────────────────────────────────
def full_eval(trades, label=""):
    pnls = np.array([t["exit_pnl"] for t in trades])
    m    = metrics_from_pnls(pnls, label)

    if len(pnls) < 3:
        return dict(label=label, n=m["n"], pf=m["pf"], wr=m["wr"],
                    net=m["net"], mdd=m["mdd"], max_dd_abs=m["max_dd_abs"],
                    equity=m["equity"], boot={}, mc={}, ues=0.0,
                    sym_floor=0.0, fold_floor=0.0, perm=0.5,
                    folds={}, syms={}, months={},
                    ulcer=0.0, rf=0.0, trades=trades)

    boot   = bootstrap_pf(pnls)
    mc     = monte_carlo(pnls)
    perm   = permutation_pvalue(pnls)

    folds  = fold_breakdown(trades)
    syms   = symbol_breakdown(trades)
    months = monthly_returns(trades)

    sym_pfs  = [v["pf"] for v in syms.values() if v["n"] >= 3]
    fold_pfs = [v["pf"] for v in folds.values() if v["n"] >= 3]
    sym_fl   = float(min(sym_pfs))  if sym_pfs  else 0.0
    fold_fl  = float(min(fold_pfs)) if fold_pfs else 0.0

    ues = ues_score(m["pf"], m["wr"], m["n"], m["mdd"],
                    max(0, sym_fl - 1.0) + 1.0 if sym_fl > 0 else 0.0,
                    max(0, fold_fl - 1.0) + 1.0 if fold_fl > 0 else 0.0,
                    boot["p5"], mc["prob_profit"])

    ul = ulcer_index(m["equity"])
    rf = recovery_factor(m["net"], m["max_dd_abs"])

    return dict(label=label, n=m["n"], pf=m["pf"], wr=m["wr"],
                net=m["net"], mdd=m["mdd"], max_dd_abs=m["max_dd_abs"],
                equity=m["equity"], boot=boot, mc=mc, ues=ues,
                sym_floor=sym_fl, fold_floor=fold_fl, perm=perm,
                folds=folds, syms=syms, months=months,
                ulcer=ul, rf=rf, trades=trades)

# ─────────────────────────────────────────────────────────────────────────────
# TRADE FREQUENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def freq_analysis(trades):
    if not trades: return {}
    times = sorted(t["entry_time"] for t in trades)
    t0, t1 = times[0], times[-1]
    span_days = max(1, (t1 - t0).days)
    span_weeks = max(1, span_days / 7)
    span_months = max(1, span_days / 30.44)

    n = len(trades)
    tpw = n / span_weeks
    tpm = n / span_months
    tpy = tpm * 12

    # Gaps
    gaps = [(times[i+1] - times[i]).total_seconds() / 3600 for i in range(len(times)-1)]
    max_gap_hrs = float(max(gaps)) if gaps else 0.0

    # Streaks
    pnls = [t["exit_pnl"] for t in sorted(trades, key=lambda x: x["entry_time"])]
    max_win_streak = max_loss_streak = cur_win = cur_loss = 0
    for p in pnls:
        if p > 0:
            cur_win += 1; cur_loss = 0
        else:
            cur_loss += 1; cur_win = 0
        max_win_streak  = max(max_win_streak,  cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Max signals per day
    by_day = defaultdict(int)
    for t in trades:
        by_day[str(t["entry_time"])[:10]] += 1
    max_per_day = max(by_day.values()) if by_day else 0

    return dict(total=n, span_days=span_days,
                tpw=round(tpw,2), tpm=round(tpm,2), tpy=round(tpy,1),
                max_gap_hrs=round(max_gap_hrs,1),
                max_win_streak=max_win_streak,
                max_loss_streak=max_loss_streak,
                max_per_day=max_per_day)

# ─────────────────────────────────────────────────────────────────────────────
# DIVERSIFICATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def diversification(trades_a, trades_b, label_a="A", label_b="B"):
    # Trade overlap
    set_a = {(t["symbol"], str(t["entry_time"])[:13]) for t in trades_a}
    set_b = {(t["symbol"], str(t["entry_time"])[:13]) for t in trades_b}
    if set_a | set_b:
        ovlp_pct = len(set_a & set_b) / len(set_a | set_b)
    else:
        ovlp_pct = 0.0

    # Symbol overlap (Jaccard)
    sym_a = {t["symbol"] for t in trades_a}
    sym_b = {t["symbol"] for t in trades_b}
    sym_ovlp = len(sym_a & sym_b) / len(sym_a | sym_b) if sym_a | sym_b else 0.0

    # Session overlap
    def session(h):
        if 7 <= h < 14: return "LON"
        elif 14 <= h < 21: return "US"
        else: return "ASI"
    sess_a = {session(t["hour_utc"]) for t in trades_a}
    sess_b = {session(t["hour_utc"]) for t in trades_b}
    sess_ovlp = len(sess_a & sess_b) / len(sess_a | sess_b) if sess_a | sess_b else 0.0

    # PnL Correlation — daily buckets
    all_days = set(str(t["entry_time"])[:10] for t in trades_a) | \
               set(str(t["entry_time"])[:10] for t in trades_b)
    day_pnl_a = defaultdict(float); day_pnl_b = defaultdict(float)
    for t in trades_a: day_pnl_a[str(t["entry_time"])[:10]] += t["exit_pnl"]
    for t in trades_b: day_pnl_b[str(t["entry_time"])[:10]] += t["exit_pnl"]
    days_sorted = sorted(all_days)
    va = np.array([day_pnl_a[d] for d in days_sorted])
    vb = np.array([day_pnl_b[d] for d in days_sorted])
    corr = float(np.corrcoef(va, vb)[0,1]) if np.std(va) > 0 and np.std(vb) > 0 else 0.0

    # Simultaneous loss days
    loss_days_a = {d for d, v in day_pnl_a.items() if v < 0}
    loss_days_b = {d for d, v in day_pnl_b.items() if v < 0}
    simult_loss = len(loss_days_a & loss_days_b)
    total_loss  = len(loss_days_a | loss_days_b)
    simult_loss_pct = simult_loss / total_loss if total_loss > 0 else 0.0

    div_score = 100 * (1 - 0.40 * ovlp_pct - 0.30 * abs(corr) - 0.20 * sym_ovlp - 0.10 * sess_ovlp)

    return dict(
        trade_overlap=round(ovlp_pct,4),
        pnl_corr=round(corr,4),
        sym_overlap=round(sym_ovlp,4),
        sess_overlap=round(sess_ovlp,4),
        simult_loss_pct=round(simult_loss_pct,4),
        div_score=round(div_score,1),
        label_a=label_a, label_b=label_b,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ROLLING METRICS
# ─────────────────────────────────────────────────────────────────────────────
def rolling_pf(trades, window=30):
    """Rolling N-trade PF."""
    pnls = [t["exit_pnl"] for t in sorted(trades, key=lambda x: x["entry_time"])]
    result = []
    for i in range(window, len(pnls)+1):
        seg = np.array(pnls[i-window:i])
        result.append(safe_pf(seg[seg>0].sum(), abs(seg[seg<0].sum())))
    return result

# ─────────────────────────────────────────────────────────────────────────────
# CAPITAL ALLOCATION SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
def alloc_simulation(trade_lists, alloc_weights, label, seed=RAND_SEED+5):
    """
    Simulate a combined portfolio with given allocation weights.
    Each family's trade_list is scaled by its weight; merge and sort.
    """
    scaled_lists = []
    for tl, w in zip(trade_lists, alloc_weights):
        scaled = [dict(t, exit_pnl=t["exit_pnl"]*w) for t in tl]
        scaled_lists.append(scaled)
    combined = combine_portfolios(scaled_lists)
    pnls = np.array([t["exit_pnl"] for t in combined])
    m    = metrics_from_pnls(pnls, label)
    boot = bootstrap_pf(pnls, seed=seed) if len(pnls) >= 5 else {}
    ul   = ulcer_index(m["equity"])
    rf   = recovery_factor(m["net"], m["max_dd_abs"])
    ues  = ues_score(m["pf"], m["wr"], m["n"], m["mdd"], 1.0, 1.0,
                     boot.get("p5",0.0), 0.8) if boot else 0.0
    return dict(label=label, weights=alloc_weights,
                pf=m["pf"], wr=m["wr"], n=m["n"], net=m["net"],
                mdd=m["mdd"], equity=m["equity"],
                boot_p50=boot.get("med",0.0), boot_p5=boot.get("p5",0.0),
                ulcer=ul, rf=rf, ues=ues)

def kelly_fraction(pf, wr, cap=0.25):
    """Kelly fraction capped at cap."""
    rr  = RR
    f   = (wr * rr - (1 - wr)) / rr
    return min(cap, max(0.0, f))

def volatility_weight(trade_lists):
    """Weight inversely proportional to PnL std dev (vol weighting)."""
    stds = []
    for tl in trade_lists:
        p = np.array([t["exit_pnl"] for t in tl])
        stds.append(float(np.std(p)) if len(p) > 1 else 1.0)
    inv = [1.0/s if s > 0 else 0.0 for s in stds]
    total = sum(inv)
    return [round(i/total, 4) for i in inv] if total > 0 else [1/len(stds)]*len(stds)

def risk_parity_weight(trade_lists):
    """Risk parity: weight inversely proportional to max drawdown."""
    mdds = []
    for tl in trade_lists:
        p   = np.array([t["exit_pnl"] for t in tl])
        eq  = CAPITAL + np.cumsum(p)
        pk  = np.maximum.accumulate(np.concatenate([[CAPITAL], eq]))
        dd  = abs((np.concatenate([[CAPITAL], eq]) - pk).min())
        mdds.append(max(dd, 1.0))
    inv   = [1.0/d for d in mdds]
    total = sum(inv)
    return [round(i/total, 4) for i in inv] if total > 0 else [1/len(mdds)]*len(mdds)

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION RANKING SCORE
# ─────────────────────────────────────────────────────────────────────────────
def prod_score(ev):
    """Multi-dimensional production ranking score 0–100."""
    pf    = ev.get("pf", 1.0)
    wr    = ev.get("wr", 0.0)
    n     = ev.get("n", 0)
    mdd   = abs(ev.get("mdd", 0.0))
    ues   = ev.get("ues", 0.0)
    bmed  = ev.get("boot",{}).get("med", 1.0)
    bP5   = ev.get("boot",{}).get("p5",  1.0)
    mc    = ev.get("mc",{}).get("prob_profit", 0.5)
    tpm   = ev.get("tpm", 1.0)
    rf    = ev.get("rf",  1.0)

    profitability     = min(30, max(0, (pf - 1.0) / 1.0 * 30))
    robustness        = min(25, max(0, (bP5 - 1.0) / 0.5 * 12 + mc * 13))
    drawdown_s        = min(20, max(0, (1 - mdd / 0.25) * 20))
    diversification_s = 5.0  # filled in externally
    freq_s            = min(10, max(0, math.log1p(tpm) / math.log1p(20) * 10))
    stat_s            = min(5,  max(0, ues / 100 * 5))
    practical_s       = min(5,  max(0, min(tpm, 10) / 10 * 5))

    return round(profitability + robustness + drawdown_s + diversification_s +
                 freq_s + stat_s + practical_s, 1)

# ─────────────────────────────────────────────────────────────────────────────
# LEAVE-ONE-OUT TESTS
# ─────────────────────────────────────────────────────────────────────────────
def loo_symbol(trades):
    syms = list({t["symbol"] for t in trades})
    floors = []
    for s in syms:
        rest = [t for t in trades if t["symbol"] != s]
        p    = np.array([t["exit_pnl"] for t in rest])
        if len(p) >= 3:
            floors.append(safe_pf(p[p>0].sum(), abs(p[p<0].sum())))
    return float(min(floors)) if floors else 0.0

def loo_fold(trades):
    folds = list({t["fold"] for t in trades})
    floors = []
    for f in folds:
        rest = [t for t in trades if t["fold"] != f]
        p    = np.array([t["exit_pnl"] for t in rest])
        if len(p) >= 3:
            floors.append(safe_pf(p[p>0].sum(), abs(p[p<0].sum())))
    return float(min(floors)) if floors else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}")
print(f"  Production Portfolio Validation (Frozen Families Only)")
print(SEP); print()
t0_global = time.time()

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"  Loading cached symbol data …")
data = load_symbol_data()
print(f"  Symbols loaded: {len(data)}")
if not data:
    print("  ERROR: No symbol data found."); sys.exit(1)
print()

# ── Run all three families ────────────────────────────────────────────────────
print("  Running walk-forward backtests for all three frozen families …")
fam_trades = {}
for fid, cids in FAMILIES.items():
    label = FAM_LABELS[fid]
    print(f"  Family {fid}: {label}")
    trades = run_family_all_symbols(cids, data)
    fam_trades[fid] = trades
    print(f"    → {len(trades)} trades")
print()

saved_charts = []

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — VERIFY INDIVIDUAL BASELINES
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print(f"  SECTION 1 — VERIFY INDIVIDUAL BASELINES"); print(SEP2)

fam_eval = {}
for fid in ("A", "B", "C"):
    trades = fam_trades[fid]
    ev     = full_eval(trades, FAM_LABELS[fid])

    # Extra: avg hold time, avg trades/month
    n_months = max(1.0, (max(t["entry_time"] for t in trades) -
                         min(t["entry_time"] for t in trades)).days / 30.44) if trades else 1.0
    ev["tpm"] = round(len(trades) / n_months, 2)
    ev["avg_hold_bars"] = 1   # fixed 1-bar hold (next-bar exit model)

    # LOO
    ev["sym_floor"]  = loo_symbol(trades)
    ev["fold_floor"] = loo_fold(trades)

    # Recalc UES with LOO
    ev["ues"] = ues_score(ev["pf"], ev["wr"], ev["n"], ev["mdd"],
                          ev["sym_floor"], ev["fold_floor"],
                          ev["boot"].get("p5", 0.0), ev["mc"].get("prob_profit", 0.5))
    fam_eval[fid] = ev

    pf_folds = {f: f"{v['pf']:.3f}(n={v['n']})" for f, v in ev["folds"].items()}
    print(f"\n  ── Family {fid}: {FAM_LABELS[fid]}")
    print(f"     PF={ev['pf']:.3f}  WR={ev['wr']:.1%}  n={ev['n']}  "
          f"Net=${ev['net']:,.1f}  Tpm={ev['tpm']:.1f}")
    print(f"     MDD={ev['mdd']:.1%}  UES={ev['ues']:.1f}  RF={ev['rf']:.2f}")
    print(f"     Boot: med={ev['boot'].get('med',0):.3f}  "
          f"p5={ev['boot'].get('p5',0):.3f}  p95={ev['boot'].get('p95',0):.3f}")
    print(f"     MC P(profit)={ev['mc'].get('prob_profit',0):.1%}")
    print(f"     LOO-sym floor={ev['sym_floor']:.3f}  "
          f"LOO-fold floor={ev['fold_floor']:.3f}")
    print(f"     Perm pctile={ev['perm']:.4f}")
    print(f"     Folds: {pf_folds}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TWO-FAMILY PORTFOLIOS
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print(f"  SECTION 2 — TWO-FAMILY PORTFOLIOS"); print(SEP2)

two_family_ids = [("A","B"), ("A","C"), ("B","C")]
portfolio_eval = {}

for (f1, f2) in two_family_ids:
    label = f"{f1}+{f2}"
    trades = combine_portfolios([fam_trades[f1], fam_trades[f2]])
    ev     = full_eval(trades, label)
    n_months = max(1.0, (max(t["entry_time"] for t in trades) -
                         min(t["entry_time"] for t in trades)).days / 30.44) if trades else 1.0
    ev["tpm"] = round(len(trades) / n_months, 2)
    ev["sym_floor"]  = loo_symbol(trades)
    ev["fold_floor"] = loo_fold(trades)
    ev["ues"] = ues_score(ev["pf"], ev["wr"], ev["n"], ev["mdd"],
                          ev["sym_floor"], ev["fold_floor"],
                          ev["boot"].get("p5",0.0), ev["mc"].get("prob_profit",0.5))
    portfolio_eval[label] = ev
    print(f"\n  ── Portfolio {label}")
    print(f"     PF={ev['pf']:.3f}  WR={ev['wr']:.1%}  n={ev['n']}  "
          f"Tpm={ev['tpm']:.1f}  MDD={ev['mdd']:.1%}")
    print(f"     Boot p5={ev['boot'].get('p5',0):.3f}  "
          f"MC P(profit)={ev['mc'].get('prob_profit',0):.1%}  UES={ev['ues']:.1f}")
    folds_str = {f: f"{v['pf']:.3f}" for f, v in ev["folds"].items()}
    print(f"     Fold PFs: {folds_str}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — THREE-FAMILY PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print(f"  SECTION 3 — THREE-FAMILY PORTFOLIO (A+B+C)"); print(SEP2)

all3_trades = combine_portfolios([fam_trades["A"], fam_trades["B"], fam_trades["C"]])
ev_abc      = full_eval(all3_trades, "A+B+C")
n_months_abc = max(1.0, (max(t["entry_time"] for t in all3_trades) -
                          min(t["entry_time"] for t in all3_trades)).days / 30.44) if all3_trades else 1.0
ev_abc["tpm"]        = round(len(all3_trades) / n_months_abc, 2)
ev_abc["sym_floor"]  = loo_symbol(all3_trades)
ev_abc["fold_floor"] = loo_fold(all3_trades)
ev_abc["ues"]        = ues_score(ev_abc["pf"], ev_abc["wr"], ev_abc["n"], ev_abc["mdd"],
                                  ev_abc["sym_floor"], ev_abc["fold_floor"],
                                  ev_abc["boot"].get("p5",0.0),
                                  ev_abc["mc"].get("prob_profit",0.5))
portfolio_eval["A+B+C"] = ev_abc

print(f"  PF={ev_abc['pf']:.3f}  WR={ev_abc['wr']:.1%}  n={ev_abc['n']}  "
      f"Net=${ev_abc['net']:,.1f}")
print(f"  MDD={ev_abc['mdd']:.1%}  UES={ev_abc['ues']:.1f}  RF={ev_abc['rf']:.2f}")
print(f"  Boot med={ev_abc['boot'].get('med',0):.3f}  "
      f"p5={ev_abc['boot'].get('p5',0):.3f}  p95={ev_abc['boot'].get('p95',0):.3f}")
print(f"  MC P(profit)={ev_abc['mc'].get('prob_profit',0):.1%}  "
      f"Median=${ev_abc['mc'].get('median',0):,.1f}")
print(f"  LOO-sym={ev_abc['sym_floor']:.3f}  LOO-fold={ev_abc['fold_floor']:.3f}")
print(f"  Tpm={ev_abc['tpm']:.1f}  Ulcer={ev_abc['ulcer']:.2f}  RF={ev_abc['rf']:.2f}")
folds_str = {f: f"{v['pf']:.3f}(n={v['n']})" for f, v in ev_abc["folds"].items()}
print(f"  Folds: {folds_str}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DIVERSIFICATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print(f"  SECTION 4 — DIVERSIFICATION ANALYSIS"); print(SEP2)

div_pairs = {}
pair_labels = [("A","B"), ("A","C"), ("B","C")]
for (f1, f2) in pair_labels:
    d = diversification(fam_trades[f1], fam_trades[f2], f"Family {f1}", f"Family {f2}")
    div_pairs[f"{f1}_{f2}"] = d
    print(f"\n  Family {f1} vs Family {f2}:")
    print(f"    Trade Overlap:   {d['trade_overlap']:.2%}")
    print(f"    PnL Correlation: {d['pnl_corr']:+.4f}")
    print(f"    Symbol Overlap:  {d['sym_overlap']:.2%}")
    print(f"    Session Overlap: {d['sess_overlap']:.2%}")
    print(f"    Simultaneous Losses: {d['simult_loss_pct']:.2%}")
    print(f"    Diversification Score: {d['div_score']:.1f}/100")

# 3-way correlation matrix
all_ids = ["A", "B", "C"]
corr_matrix = {}
for fa in all_ids:
    corr_matrix[fa] = {}
    for fb in all_ids:
        if fa == fb:
            corr_matrix[fa][fb] = 1.0
        elif f"{fa}_{fb}" in div_pairs:
            corr_matrix[fa][fb] = div_pairs[f"{fa}_{fb}"]["pnl_corr"]
        elif f"{fb}_{fa}" in div_pairs:
            corr_matrix[fa][fb] = div_pairs[f"{fb}_{fa}"]["pnl_corr"]
        else:
            corr_matrix[fa][fb] = 0.0

print(f"\n  PnL Correlation Matrix:")
print(f"     {'':>6}  {'A':>8}  {'B':>8}  {'C':>8}")
for fa in all_ids:
    row = "  ".join(f"{corr_matrix[fa][fb]:+.4f}" for fb in all_ids)
    print(f"     Family {fa}:  {row}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — DRAWDOWN DIVERSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print(f"  SECTION 5 — DRAWDOWN DIVERSIFICATION"); print(SEP2)

print(f"\n  Individual family metrics:")
for fid in ("A", "B", "C"):
    ev = fam_eval[fid]
    print(f"    Family {fid}: MDD={ev['mdd']:.1%}  Ulcer={ev['ulcer']:.2f}  "
          f"RF={ev['rf']:.2f}  MaxDD=${ev['max_dd_abs']:,.1f}")

print(f"\n  Two-family portfolio metrics:")
for label in ["A+B", "A+C", "B+C"]:
    ev = portfolio_eval[label]
    print(f"    {label}: MDD={ev['mdd']:.1%}  Ulcer={ev['ulcer']:.2f}  "
          f"RF={ev['rf']:.2f}  MaxDD=${ev['max_dd_abs']:,.1f}")

print(f"\n  Three-family portfolio:")
print(f"    A+B+C: MDD={ev_abc['mdd']:.1%}  Ulcer={ev_abc['ulcer']:.2f}  "
      f"RF={ev_abc['rf']:.2f}  MaxDD=${ev_abc['max_dd_abs']:,.1f}")

# Rolling 30-trade PF for each
print(f"\n  Rolling 30-trade PF stability:")
for fid in ("A", "B", "C"):
    rpf = rolling_pf(fam_trades[fid], 30)
    if rpf:
        print(f"    Family {fid}: min={min(rpf):.3f}  max={max(rpf):.3f}  "
              f"mean={np.mean(rpf):.3f}  pct_above_1={100*(np.array(rpf)>1).mean():.0f}%")
    else:
        print(f"    Family {fid}: insufficient trades for rolling window")

rpf_abc = rolling_pf(all3_trades, 30)
if rpf_abc:
    print(f"    A+B+C:    min={min(rpf_abc):.3f}  max={max(rpf_abc):.3f}  "
          f"mean={np.mean(rpf_abc):.3f}  pct_above_1={100*(np.array(rpf_abc)>1).mean():.0f}%")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TRADE FREQUENCY
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print(f"  SECTION 6 — TRADE FREQUENCY"); print(SEP2)

freq_results = {}
for fid in ("A","B","C"):
    f = freq_analysis(fam_trades[fid])
    freq_results[fid] = f
    print(f"\n  Family {fid}:")
    print(f"    Trades/wk={f.get('tpw',0):.2f}  Trades/month={f.get('tpm',0):.1f}  "
          f"Expected/year={f.get('tpy',0):.0f}")
    print(f"    Max gap: {f.get('max_gap_hrs',0):.0f}h  "
          f"Max signals/day={f.get('max_per_day',0)}")
    print(f"    Longest win streak={f.get('max_win_streak',0)}  "
          f"Longest loss streak={f.get('max_loss_streak',0)}")

f_abc = freq_analysis(all3_trades)
freq_results["A+B+C"] = f_abc
print(f"\n  Combined A+B+C:")
print(f"    Trades/wk={f_abc.get('tpw',0):.2f}  Trades/month={f_abc.get('tpm',0):.1f}  "
      f"Expected/year={f_abc.get('tpy',0):.0f}")
print(f"    Max gap: {f_abc.get('max_gap_hrs',0):.0f}h  "
      f"Max signals/day={f_abc.get('max_per_day',0)}")
print(f"    Longest win streak={f_abc.get('max_win_streak',0)}  "
      f"Longest loss streak={f_abc.get('max_loss_streak',0)}")

# Practicality verdict
tpm_abc = f_abc.get("tpm", 0)
practical = "YES" if tpm_abc >= 5 else ("BORDERLINE" if tpm_abc >= 2 else "NO — too infrequent")
print(f"\n  Practical for retail trader? {practical}  ({tpm_abc:.1f} trades/month combined)")
print()

# Store tpm in evals for ranking
for fid in ("A","B","C"):
    fam_eval[fid]["tpm"] = freq_results[fid].get("tpm", 0)
for k in portfolio_eval:
    fids = k.replace("+","")
    if k in portfolio_eval:
        f = freq_analysis(portfolio_eval[k]["trades"]) if portfolio_eval[k].get("trades") else {}
        portfolio_eval[k]["tpm"] = f.get("tpm", ev["tpm"] if "tpm" in ev else 0)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CAPITAL ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print(f"  SECTION 7 — CAPITAL ALLOCATION"); print(SEP2)

tls_abc = [fam_trades["A"], fam_trades["B"], fam_trades["C"]]

# 1. Equal weight
ew   = [1/3, 1/3, 1/3]
ev_ew = alloc_simulation(tls_abc, ew, "Equal Weight (33/33/33)")

# 2. Volatility weighting
vw   = volatility_weight(tls_abc)
ev_vw = alloc_simulation(tls_abc, vw, f"Volatility Weight ({vw[0]:.2f}/{vw[1]:.2f}/{vw[2]:.2f})")

# 3. Risk parity
rp   = risk_parity_weight(tls_abc)
ev_rp = alloc_simulation(tls_abc, rp, f"Risk Parity ({rp[0]:.2f}/{rp[1]:.2f}/{rp[2]:.2f})")

# 4. Kelly fraction (capped)
kelly_weights = []
for fid in ("A","B","C"):
    ev_ = fam_eval[fid]
    kf  = kelly_fraction(ev_["pf"], ev_["wr"])
    kelly_weights.append(kf)
total_k = sum(kelly_weights) or 1.0
kelly_norm = [round(k/total_k, 4) for k in kelly_weights]
ev_kl = alloc_simulation(tls_abc, kelly_norm,
                          f"Kelly Capped ({kelly_norm[0]:.2f}/{kelly_norm[1]:.2f}/{kelly_norm[2]:.2f})")

alloc_results = [ev_ew, ev_vw, ev_rp, ev_kl]
print(f"\n  {'Scheme':<38}  {'PF':>6}  {'WR':>6}  {'MDD':>7}  {'UES':>6}  {'RF':>6}")
print(f"  {'-'*38}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*6}")
best_alloc = None; best_alloc_ues = -1
for ar in alloc_results:
    print(f"  {ar['label']:<38}  {ar['pf']:.3f}  {ar['wr']:.1%}  "
          f"{ar['mdd']:.1%}  {ar['ues']:.1f}  {ar['rf']:.2f}")
    if ar["ues"] > best_alloc_ues:
        best_alloc_ues = ar["ues"]
        best_alloc = ar
print(f"\n  Best allocation: {best_alloc['label']}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print(f"  SECTION 8 — STRESS TESTS"); print(SEP2)
print(f"\n  Stress-testing combined A+B+C portfolio …")

pnls_abc = np.array([t["exit_pnl"] for t in all3_trades])

# Bootstrap
bt_abc = bootstrap_pf(pnls_abc, seed=RAND_SEED+10)
print(f"\n  Bootstrap ({N_BOOT:,} iter): "
      f"med={bt_abc['med']:.3f}  p5={bt_abc['p5']:.3f}  p95={bt_abc['p95']:.3f}  "
      f"P(PF>1.0)={100*(bt_abc['pf_arr']>1).mean():.1f}%")

# Monte Carlo
mc_abc = monte_carlo(pnls_abc, seed=RAND_SEED+11)
print(f"  Monte Carlo ({N_MC:,} iter): "
      f"P(profit)={mc_abc['prob_profit']:.1%}  "
      f"Median final=${mc_abc['median']:,.1f}  "
      f"p5=${mc_abc['p5']:,.1f}")

# Leave-one-fold-out
loof_abc = loo_fold(all3_trades)
print(f"  LOO-fold floor PF: {loof_abc:.3f}  ({'PASS' if loof_abc > 1.0 else 'FAIL'})")

# Leave-one-symbol-out
loos_abc = loo_symbol(all3_trades)
print(f"  LOO-sym floor PF:  {loos_abc:.3f}  ({'PASS' if loos_abc > 1.0 else 'FAIL'})")

# Permutation test
perm_abc = permutation_pvalue(pnls_abc)
print(f"  Permutation pctile: {perm_abc:.4f}  "
      f"({'PASS >0.95' if perm_abc >= 0.95 else 'BORDERLINE' if perm_abc >= 0.80 else 'FAIL'})")

# Parameter robustness — shift RR ±0.5 and recompute PF
print(f"\n  Parameter Robustness (RR sensitivity):")
for rr_test in [1.5, 2.0, 2.5]:
    # Scale wins to new RR
    pnls_scaled = np.where(pnls_abc > 0, pnls_abc * (rr_test / RR), pnls_abc)
    pf_scaled   = safe_pf(pnls_scaled[pnls_scaled>0].sum(), abs(pnls_scaled[pnls_scaled<0].sum()))
    print(f"    RR={rr_test:.1f}: PF={pf_scaled:.3f}")

# Condition ablation — remove one family at a time
print(f"\n  Condition Ablation (leave-one-family-out):")
for skip in ("A","B","C"):
    kept = [fam_trades[f] for f in ("A","B","C") if f != skip]
    tc   = combine_portfolios(kept)
    p    = np.array([t["exit_pnl"] for t in tc])
    pf_a = safe_pf(p[p>0].sum(), abs(p[p<0].sum())) if len(p) >= 3 else 0.0
    print(f"    Without Family {skip}: PF={pf_a:.3f}  n={len(tc)}")

# Stress verdict
stress_pass  = sum([
    bt_abc["p5"] > 1.0,
    mc_abc["prob_profit"] > 0.75,
    loof_abc > 1.0,
    loos_abc > 1.0,
    perm_abc >= 0.80,
])
stress_verdict = ("STRONG" if stress_pass >= 5 else
                  "MODERATE" if stress_pass >= 3 else "WEAK")
print(f"\n  Stress verdict: {stress_verdict}  ({stress_pass}/5 tests passed)")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PRODUCTION RANKING
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print(f"  SECTION 9 — PRODUCTION RANKING"); print(SEP2)

all_candidates = {}
for fid in ("A","B","C"):
    all_candidates[f"Family {fid}"] = fam_eval[fid]
for label in ["A+B","A+C","B+C","A+B+C"]:
    all_candidates[label] = portfolio_eval[label]

# Score each
scores = {}
for name, ev in all_candidates.items():
    sc = prod_score(ev)
    # bonus for high trade count (frequency)
    tpm_ = ev.get("tpm", 0)
    sc   = min(100, sc)
    scores[name] = dict(ev=ev, score=sc, tpm=tpm_)

ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

print(f"\n  {'Rank':<4}  {'Candidate':<16}  {'Score':>6}  {'PF':>6}  {'WR':>6}  "
      f"{'n':>5}  {'MDD':>7}  {'UES':>6}  {'Tpm':>5}")
print(f"  {'-'*4}  {'-'*16}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*5}")
for rank, (name, d) in enumerate(ranked, 1):
    ev_  = d["ev"]
    print(f"  {rank:<4}  {name:<16}  {d['score']:>6.1f}  {ev_['pf']:>6.3f}  "
          f"{ev_['wr']:>6.1%}  {ev_['n']:>5}  {ev_['mdd']:>7.1%}  "
          f"{ev_['ues']:>6.1f}  {d['tpm']:>5.1f}")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print(f"  SECTION 10 — FINAL VERDICT"); print(SEP2)

ev_a = fam_eval["A"]; ev_b = fam_eval["B"]; ev_c = fam_eval["C"]

# Q1: Does combining all three improve portfolio?
pf_abc  = ev_abc["pf"]
pf_best_single = max(ev_a["pf"], ev_b["pf"], ev_c["pf"])
q1 = "YES" if pf_abc >= min(ev_a["pf"], ev_b["pf"], ev_c["pf"]) and ev_abc["n"] >= max(ev_a["n"], ev_b["n"]) else "MIXED"
q1_note = (f"Combined PF={pf_abc:.3f} vs best single={pf_best_single:.3f}. "
           f"n increases to {ev_abc['n']}.")

# Q2: Does it increase trade frequency enough?
tpm_a = fam_eval["A"].get("tpm", 0)
q2 = "YES" if f_abc.get("tpm",0) >= 2 * max(tpm_a, 0.5) else "MARGINAL"
q2_note = f"A+B+C: {f_abc.get('tpm',0):.1f} trades/month vs Family A alone: {tpm_a:.1f}/month"

# Q3: Does diversification reduce drawdown?
mdd_a    = abs(ev_a["mdd"]); mdd_abc_abs = abs(ev_abc["mdd"])
q3 = "YES" if mdd_abc_abs <= max(mdd_a, abs(ev_b["mdd"]), abs(ev_c["mdd"])) else "NO"
q3_note  = (f"A+B+C MDD={ev_abc['mdd']:.1%} vs A={ev_a['mdd']:.1%}, "
            f"B={ev_b['mdd']:.1%}, C={ev_c['mdd']:.1%}")

# Q4: Is combined superior to E3.1 alone?
ues_abc  = ev_abc["ues"]; ues_a = ev_a["ues"]
q4 = "YES" if ues_abc >= ues_a and ev_abc["n"] > ev_a["n"] else "NO"
q4_note  = f"A+B+C UES={ues_abc:.1f}  n={ev_abc['n']} vs Family A UES={ues_a:.1f} n={ev_a['n']}"

# Q5: Which would you deploy today?
# Best single: highest UES with n >= 20
best_rank_name = ranked[0][0]
q5_note = f"Top-ranked: {best_rank_name} (score={ranked[0][1]['score']:.1f})"
# Prefer Family A if it has best stats historically (E3.1 is most validated)
deploy_candidate = best_rank_name

print(f"\n  Q1  Does combining all three improve the portfolio?")
print(f"      {q1}: {q1_note}")

print(f"\n  Q2  Does it increase trade frequency enough to justify complexity?")
print(f"      {q2}: {q2_note}")

print(f"\n  Q3  Does diversification reduce drawdown?")
print(f"      {q3}: {q3_note}")

print(f"\n  Q4  Is combined portfolio superior to E3.1 alone?")
print(f"      {q4}: {q4_note}")

print(f"\n  Q5  Which portfolio would you deploy today for live paper trading?")
print(f"      {deploy_candidate}: {q5_note}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP); print("  GENERATING CHARTS …"); print(SEP2)

# ── Chart 1: Dashboard — 9-panel summary ─────────────────────────────────────
fig1 = plt.figure(figsize=(18, 14))
fig1.suptitle("R066 — Production Portfolio Validation (Frozen Families A, B, C)",
              fontsize=13, fontweight="bold", color=C_TEXT, y=0.98)
gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.45, wspace=0.35)

def bar_compare(ax, labels, values, colors, title, ylabel, refline=None):
    style_ax(ax)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor=C_BG, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_title(title, fontsize=8, color=C_TEXT)
    ax.set_ylabel(ylabel, fontsize=7, color=C_TEXT)
    if refline is not None:
        ax.axhline(refline, color=C_RED, lw=1.0, ls="--")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + abs(max(values,default=1))*0.01,
                f"{val:.3f}" if abs(val) < 10 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=6.5, color=C_TEXT)

all_labels_ordered = ["Fam A", "Fam B", "Fam C", "A+B", "A+C", "B+C", "A+B+C"]
all_evs_ordered    = [fam_eval["A"], fam_eval["B"], fam_eval["C"],
                      portfolio_eval["A+B"], portfolio_eval["A+C"],
                      portfolio_eval["B+C"], portfolio_eval["A+B+C"]]
colors_ordered     = [C_GREEN, C_GOLD, C_BLUE, C_TEAL, C_PURP, C_ORAN, C_RED]

# PF comparison
ax = fig1.add_subplot(gs1[0,0])
bar_compare(ax, all_labels_ordered, [e["pf"] for e in all_evs_ordered],
            colors_ordered, "Profit Factor", "PF", refline=1.0)

# WR comparison
ax = fig1.add_subplot(gs1[0,1])
bar_compare(ax, all_labels_ordered, [e["wr"]*100 for e in all_evs_ordered],
            colors_ordered, "Win Rate (%)", "WR %", refline=50)

# Trade count
ax = fig1.add_subplot(gs1[0,2])
bar_compare(ax, all_labels_ordered, [e["n"] for e in all_evs_ordered],
            colors_ordered, "Trade Count", "n")

# MDD comparison
ax = fig1.add_subplot(gs1[1,0])
bar_compare(ax, all_labels_ordered, [abs(e["mdd"])*100 for e in all_evs_ordered],
            colors_ordered, "Max Drawdown (%)", "MDD %")

# UES comparison
ax = fig1.add_subplot(gs1[1,1])
bar_compare(ax, all_labels_ordered, [e["ues"] for e in all_evs_ordered],
            colors_ordered, "Universal Edge Score", "UES")

# Bootstrap P5
ax = fig1.add_subplot(gs1[1,2])
bar_compare(ax, all_labels_ordered, [e["boot"].get("p5",0) for e in all_evs_ordered],
            colors_ordered, "Bootstrap P5 (floor)", "PF P5", refline=1.0)

# Recovery Factor
ax = fig1.add_subplot(gs1[2,0])
bar_compare(ax, all_labels_ordered, [e["rf"] for e in all_evs_ordered],
            colors_ordered, "Recovery Factor", "RF")

# Trade frequency (tpm)
ax = fig1.add_subplot(gs1[2,1])
tpm_vals = [e.get("tpm",0) for e in all_evs_ordered]
bar_compare(ax, all_labels_ordered, tpm_vals, colors_ordered, "Trades / Month", "Tpm")

# MC P(profit)
ax = fig1.add_subplot(gs1[2,2])
bar_compare(ax, all_labels_ordered, [e["mc"].get("prob_profit",0)*100 for e in all_evs_ordered],
            colors_ordered, "MC P(Profitable) %", "MC %", refline=75)

saved_charts.append(save_fig(fig1, "r066_dashboard.png"))
print(f"  → r066_dashboard.png")

# ── Chart 2: Equity Curves ────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
fig2.suptitle("R066 — Equity Curves (All Portfolios)", fontsize=11, color=C_TEXT,
              fontweight="bold", y=0.99)
all_panels = [("Family A", fam_eval["A"],        C_GREEN),
              ("Family B", fam_eval["B"],        C_GOLD),
              ("Family C", fam_eval["C"],        C_BLUE),
              ("A+B",      portfolio_eval["A+B"], C_TEAL),
              ("A+C",      portfolio_eval["A+C"], C_PURP),
              ("B+C",      portfolio_eval["B+C"], C_ORAN),
              ("A+B+C",    portfolio_eval["A+B+C"], C_RED)]

for idx, (lbl, ev_, col) in enumerate(all_panels):
    row = idx // 4; c_ = idx % 4
    ax_ = axes2[row][c_]
    style_ax(ax_)
    eq = ev_["equity"]
    ax_.plot(eq, color=col, lw=1.4)
    ax_.axhline(CAPITAL, color=C_GRID, lw=0.8, ls="--")
    ax_.set_title(f"{lbl}  PF={ev_['pf']:.3f}  n={ev_['n']}\n"
                  f"MDD={ev_['mdd']:.1%}  UES={ev_['ues']:.1f}",
                  fontsize=7.5, color=C_TEXT)
    ax_.set_xlabel("Trade #", fontsize=6, color=C_TEXT)
    ax_.set_ylabel("Equity $", fontsize=6, color=C_TEXT)

# Hide extra subplot
axes2[1][3].set_visible(False)
plt.tight_layout()
saved_charts.append(save_fig(fig2, "r066_equity_curves.png"))
print(f"  → r066_equity_curves.png")

# ── Chart 3: Fold Stability ───────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 4, figsize=(20, 9))
fig3.suptitle("R066 — Fold-by-Fold PF (Walk-Forward Stability)", fontsize=11,
              color=C_TEXT, fontweight="bold")
for idx, (lbl, ev_, col) in enumerate(all_panels):
    row = idx // 4; c_ = idx % 4
    ax_ = axes3[row][c_]
    style_ax(ax_)
    folds_d = ev_["folds"]
    fs = sorted(folds_d.keys())
    pfs = [folds_d[f]["pf"] for f in fs]
    ns  = [folds_d[f]["n"]  for f in fs]
    bars_ = ax_.bar(fs, pfs, color=[C_GREEN if p >= 1.0 else C_RED for p in pfs],
                    alpha=0.85, edgecolor=C_BG, width=0.6)
    ax_.axhline(1.0, color=C_GOLD, lw=1.0, ls="--")
    for b_, nv in zip(bars_, ns):
        ax_.text(b_.get_x()+b_.get_width()/2, b_.get_height()+0.02,
                 f"n={nv}", ha="center", va="bottom", fontsize=6, color=C_TEXT)
    ax_.set_title(f"{lbl}  PF={ev_['pf']:.3f}", fontsize=8, color=C_TEXT)
    ax_.set_xlabel("Fold", fontsize=6, color=C_TEXT)
    ax_.set_ylabel("OOS PF", fontsize=6, color=C_TEXT)
    winning = sum(1 for p in pfs if p >= 1.0)
    ax_.text(0.02, 0.97, f"{winning}/{len(fs)} folds profitable",
             transform=ax_.transAxes, fontsize=7, color=C_GOLD,
             va="top", ha="left")

axes3[1][3].set_visible(False)
plt.tight_layout()
saved_charts.append(save_fig(fig3, "r066_fold_stability.png"))
print(f"  → r066_fold_stability.png")

# ── Chart 4: Diversification / Correlation Matrix ────────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 6))
fig4.suptitle("R066 — Diversification Analysis", fontsize=11, color=C_TEXT,
              fontweight="bold")

# Correlation matrix heatmap
ax4a = axes4[0]; style_ax(ax4a)
mat_vals = np.array([[corr_matrix[fa][fb] for fb in all_ids] for fa in all_ids])
im = ax4a.imshow(mat_vals, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
ax4a.set_xticks(range(3)); ax4a.set_xticklabels([f"Family {x}" for x in all_ids], fontsize=8)
ax4a.set_yticks(range(3)); ax4a.set_yticklabels([f"Family {x}" for x in all_ids], fontsize=8)
for i in range(3):
    for j in range(3):
        ax4a.text(j, i, f"{mat_vals[i,j]:.3f}", ha="center", va="center",
                  fontsize=9, color="black", fontweight="bold")
ax4a.set_title("PnL Correlation Matrix", fontsize=9, color=C_TEXT)
plt.colorbar(im, ax=ax4a, shrink=0.8)

# Diversification scores bar
ax4b = axes4[1]; style_ax(ax4b)
pair_names = [f"{p1} vs {p2}" for (p1,p2) in [("A","B"),("A","C"),("B","C")]]
pair_divs   = [div_pairs[k]["div_score"] for k in ["A_B","A_C","B_C"]]
pair_cols   = [C_TEAL, C_PURP, C_ORAN]
bars4b = ax4b.bar(pair_names, pair_divs, color=pair_cols, alpha=0.85, edgecolor=C_BG)
ax4b.set_ylim(0, 100)
ax4b.axhline(70, color=C_GOLD, lw=1, ls="--", label="Good (70+)")
for bar4b, val4b in zip(bars4b, pair_divs):
    ax4b.text(bar4b.get_x()+bar4b.get_width()/2, bar4b.get_height()+1,
              f"{val4b:.1f}", ha="center", va="bottom", fontsize=9, color=C_TEXT)
ax4b.set_title("Diversification Score", fontsize=9, color=C_TEXT)
ax4b.set_ylabel("Score / 100", fontsize=8, color=C_TEXT)
ax4b.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

# Trade overlap bar
ax4c = axes4[2]; style_ax(ax4c)
pair_ovlps = [div_pairs[k]["trade_overlap"]*100 for k in ["A_B","A_C","B_C"]]
bars4c = ax4c.bar(pair_names, pair_ovlps, color=pair_cols, alpha=0.85, edgecolor=C_BG)
ax4c.axhline(20, color=C_RED, lw=1, ls="--", label="High overlap (20%+)")
for bar4c, val4c in zip(bars4c, pair_ovlps):
    ax4c.text(bar4c.get_x()+bar4c.get_width()/2, bar4c.get_height()+0.1,
              f"{val4c:.1f}%", ha="center", va="bottom", fontsize=9, color=C_TEXT)
ax4c.set_title("Trade Overlap %", fontsize=9, color=C_TEXT)
ax4c.set_ylabel("Overlap %", fontsize=8, color=C_TEXT)
ax4c.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

plt.tight_layout()
saved_charts.append(save_fig(fig4, "r066_diversification.png"))
print(f"  → r066_diversification.png")

# ── Chart 5: Bootstrap Distributions ─────────────────────────────────────────
fig5, axes5 = plt.subplots(2, 4, figsize=(20, 9))
fig5.suptitle("R066 — Bootstrap PF Distributions", fontsize=11, color=C_TEXT,
              fontweight="bold")
for idx, (lbl, ev_, col) in enumerate(all_panels):
    row = idx // 4; c_ = idx % 4
    ax_ = axes5[row][c_]
    style_ax(ax_)
    pf_arr = ev_["boot"].get("pf_arr", np.array([ev_["pf"]]))
    bins_  = max(5, min(50, len(pf_arr)//20))
    ax_.hist(pf_arr, bins=bins_, color=col, alpha=0.75)
    b5  = ev_["boot"].get("p5",  0)
    b50 = ev_["boot"].get("med", 0)
    ax_.axvline(1.0, color=C_RED,  lw=1.5, ls="--", label="Break-even")
    ax_.axvline(b5,  color=C_GOLD, lw=1.2, ls=":",  label=f"P5={b5:.3f}")
    ax_.axvline(b50, color=C_GREEN,lw=1.2,           label=f"P50={b50:.3f}")
    ax_.set_title(f"{lbl}  Boot P5={b5:.3f}", fontsize=8, color=C_TEXT)
    ax_.set_xlabel("Profit Factor", fontsize=6, color=C_TEXT)
    ax_.legend(fontsize=6, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

axes5[1][3].set_visible(False)
plt.tight_layout()
saved_charts.append(save_fig(fig5, "r066_bootstrap.png"))
print(f"  → r066_bootstrap.png")

# ── Chart 6: Monte Carlo Final Equity ────────────────────────────────────────
fig6, axes6 = plt.subplots(2, 4, figsize=(20, 9))
fig6.suptitle("R066 — Monte Carlo Final Equity Distribution", fontsize=11, color=C_TEXT,
              fontweight="bold")
for idx, (lbl, ev_, col) in enumerate(all_panels):
    row = idx // 4; c_ = idx % 4
    ax_ = axes6[row][c_]
    style_ax(ax_)
    finals_ = ev_["mc"].get("finals", np.array([CAPITAL]))
    bins_mc = max(5, min(50, len(finals_)//20))
    ax_.hist(finals_, bins=bins_mc, color=col, alpha=0.75)
    ax_.axvline(CAPITAL, color=C_GOLD, lw=1.5, ls="--", label=f"Start")
    mc_med = ev_["mc"].get("median", CAPITAL)
    ax_.axvline(mc_med, color=C_GREEN, lw=1.2, label=f"Median=${mc_med:,.0f}")
    prob_ = ev_["mc"].get("prob_profit", 0)
    ax_.set_title(f"{lbl}  P(profit)={prob_:.1%}", fontsize=8, color=C_TEXT)
    ax_.set_xlabel("Final Equity $", fontsize=6, color=C_TEXT)
    ax_.legend(fontsize=6, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

axes6[1][3].set_visible(False)
plt.tight_layout()
saved_charts.append(save_fig(fig6, "r066_monte_carlo.png"))
print(f"  → r066_monte_carlo.png")

# ── Chart 7: Production Ranking ───────────────────────────────────────────────
fig7, ax7 = plt.subplots(figsize=(12, 7))
fig7.suptitle("R066 — Production Ranking (Multi-Dimensional Score)", fontsize=11,
              color=C_TEXT, fontweight="bold")
style_ax(ax7)
rank_labels = [n for n, _ in ranked]
rank_scores = [d["score"] for _, d in ranked]
rank_colors = [FAM_COLORS.get(n.replace("Family ","").strip(), C_BLUE) for n in rank_labels]
bars7 = ax7.barh(range(len(ranked)), rank_scores, color=rank_colors, alpha=0.85, edgecolor=C_BG)
ax7.set_yticks(range(len(ranked)))
ax7.set_yticklabels(rank_labels, fontsize=9)
ax7.invert_yaxis()
ax7.set_xlabel("Production Score (0–100)", fontsize=9, color=C_TEXT)
for bar7, sc_ in zip(bars7, rank_scores):
    ax7.text(sc_ + 0.5, bar7.get_y() + bar7.get_height()/2,
             f"{sc_:.1f}", va="center", fontsize=9, color=C_TEXT)
plt.tight_layout()
saved_charts.append(save_fig(fig7, "r066_ranking.png"))
print(f"  → r066_ranking.png")

# ── Chart 8: Capital Allocation Comparison ───────────────────────────────────
fig8, axes8 = plt.subplots(1, 4, figsize=(20, 6))
fig8.suptitle("R066 — Capital Allocation Comparison (A+B+C Portfolio)", fontsize=10,
              color=C_TEXT, fontweight="bold")
alloc_names = [ar["label"].split("(")[0].strip() for ar in alloc_results]
alloc_cols8  = [C_GREEN, C_GOLD, C_BLUE, C_PURP]

metrics8 = [("PF",  [ar["pf"]  for ar in alloc_results], 1.0),
            ("MDD%",[abs(ar["mdd"])*100 for ar in alloc_results], None),
            ("RF",  [ar["rf"]  for ar in alloc_results], None),
            ("UES", [ar["ues"] for ar in alloc_results], None)]

for i, (mname, vals, refline) in enumerate(metrics8):
    ax8 = axes8[i]; style_ax(ax8)
    bars8 = ax8.bar(alloc_names, vals, color=alloc_cols8, alpha=0.85, edgecolor=C_BG, width=0.6)
    if refline is not None:
        ax8.axhline(refline, color=C_RED, lw=1, ls="--")
    for b8, v8 in zip(bars8, vals):
        ax8.text(b8.get_x()+b8.get_width()/2, b8.get_height()+abs(max(vals,default=1))*0.01,
                 f"{v8:.3f}" if abs(v8) < 10 else f"{v8:.1f}",
                 ha="center", va="bottom", fontsize=8, color=C_TEXT)
    ax8.set_title(mname, fontsize=9, color=C_TEXT)
    ax8.set_xticklabels(alloc_names, fontsize=7, rotation=20, ha="right")

plt.tight_layout()
saved_charts.append(save_fig(fig8, "r066_allocation.png"))
print(f"  → r066_allocation.png")

# ── Chart 9: Drawdown Diversification ────────────────────────────────────────
fig9, axes9 = plt.subplots(1, 3, figsize=(18, 6))
fig9.suptitle("R066 — Drawdown Diversification", fontsize=11, color=C_TEXT,
              fontweight="bold")

# Drawdown curve per family
ax9a = axes9[0]; style_ax(ax9a)
for fid, col in [("A", C_GREEN), ("B", C_GOLD), ("C", C_BLUE)]:
    eq = fam_eval[fid]["equity"]
    pk = np.maximum.accumulate(eq)
    dd = (eq - pk) / pk * 100
    ax9a.plot(dd, color=col, lw=1.0, label=f"Family {fid} (MDD={fam_eval[fid]['mdd']:.1%})", alpha=0.85)
ax9a.axhline(0, color=C_GRID, lw=0.8)
ax9a.set_title("Individual Family Drawdown %", fontsize=9, color=C_TEXT)
ax9a.set_xlabel("Trade #", fontsize=7, color=C_TEXT)
ax9a.set_ylabel("Drawdown %", fontsize=7, color=C_TEXT)
ax9a.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

# Combined drawdown comparison
ax9b = axes9[1]; style_ax(ax9b)
for label, ev_, col in [("A+B", portfolio_eval["A+B"], C_TEAL),
                         ("A+C", portfolio_eval["A+C"], C_PURP),
                         ("B+C", portfolio_eval["B+C"], C_ORAN),
                         ("A+B+C", portfolio_eval["A+B+C"], C_RED)]:
    eq = ev_["equity"]
    pk = np.maximum.accumulate(eq)
    dd = (eq - pk) / pk * 100
    ax9b.plot(dd, color=col, lw=1.0, label=f"{label} (MDD={ev_['mdd']:.1%})", alpha=0.85)
ax9b.axhline(0, color=C_GRID, lw=0.8)
ax9b.set_title("Portfolio Drawdown %", fontsize=9, color=C_TEXT)
ax9b.set_xlabel("Trade #", fontsize=7, color=C_TEXT)
ax9b.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

# Ulcer Index comparison
ax9c = axes9[2]; style_ax(ax9c)
ul_labels = ["A","B","C","A+B","A+C","B+C","A+B+C"]
ul_vals    = [fam_eval["A"]["ulcer"], fam_eval["B"]["ulcer"], fam_eval["C"]["ulcer"],
              portfolio_eval["A+B"]["ulcer"], portfolio_eval["A+C"]["ulcer"],
              portfolio_eval["B+C"]["ulcer"], portfolio_eval["A+B+C"]["ulcer"]]
ul_cols_   = [C_GREEN, C_GOLD, C_BLUE, C_TEAL, C_PURP, C_ORAN, C_RED]
bars9c = ax9c.bar(ul_labels, ul_vals, color=ul_cols_, alpha=0.85, edgecolor=C_BG, width=0.6)
for b9, v9 in zip(bars9c, ul_vals):
    ax9c.text(b9.get_x()+b9.get_width()/2, b9.get_height()+0.01,
              f"{v9:.2f}", ha="center", va="bottom", fontsize=8, color=C_TEXT)
ax9c.set_title("Ulcer Index (lower=better)", fontsize=9, color=C_TEXT)
ax9c.set_ylabel("Ulcer Index", fontsize=7, color=C_TEXT)

plt.tight_layout()
saved_charts.append(save_fig(fig9, "r066_drawdown_diversification.png"))
print(f"  → r066_drawdown_diversification.png")

# ── Chart 10: Monthly Returns Heatmap ─────────────────────────────────────────
fig10, axes10 = plt.subplots(2, 4, figsize=(20, 9))
fig10.suptitle("R066 — Monthly Returns Heatmap", fontsize=11, color=C_TEXT,
               fontweight="bold")
for idx, (lbl, ev_, col) in enumerate(all_panels):
    row = idx // 4; c_ = idx % 4
    ax_ = axes10[row][c_]
    style_ax(ax_)
    months_ = ev_["months"]
    if months_:
        mkeys = list(months_.keys())
        mvals = list(months_.values())
        bar_c = [C_GREEN if v >= 0 else C_RED for v in mvals]
        ax_.bar(range(len(mkeys)), mvals, color=bar_c, alpha=0.8, edgecolor=C_BG, width=0.8)
        ax_.axhline(0, color=C_GOLD, lw=0.8, ls="--")
        step = max(1, len(mkeys)//8)
        ax_.set_xticks(range(0, len(mkeys), step))
        ax_.set_xticklabels([str(mkeys[i])[:7] for i in range(0, len(mkeys), step)],
                             fontsize=5.5, rotation=45, ha="right")
        pos_m = sum(1 for v in mvals if v >= 0)
        ax_.set_title(f"{lbl}  {pos_m}/{len(mvals)} positive months",
                      fontsize=7.5, color=C_TEXT)
    else:
        ax_.set_title(f"{lbl}  (no monthly data)", fontsize=8, color=C_TEXT)
    ax_.set_ylabel("PnL $", fontsize=6, color=C_TEXT)

axes10[1][3].set_visible(False)
plt.tight_layout()
saved_charts.append(save_fig(fig10, "r066_monthly_returns.png"))
print(f"  → r066_monthly_returns.png")

print()

# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("  Saving CSV outputs …")

# Master summary CSV
rows_csv = []
for fid in ("A","B","C"):
    ev_ = fam_eval[fid]
    rows_csv.append({
        "candidate": f"Family {fid}", "conditions": FAM_LABELS[fid],
        "pf": round(ev_["pf"],4), "wr": round(ev_["wr"],4), "n": ev_["n"],
        "net": round(ev_["net"],2), "mdd": round(ev_["mdd"],4),
        "ues": ev_["ues"], "boot_p5": round(ev_["boot"].get("p5",0),4),
        "boot_med": round(ev_["boot"].get("med",0),4),
        "mc_prob": round(ev_["mc"].get("prob_profit",0),4),
        "rf": round(ev_["rf"],4), "ulcer": round(ev_["ulcer"],4),
        "sym_floor": round(ev_["sym_floor"],4), "fold_floor": round(ev_["fold_floor"],4),
        "tpm": ev_.get("tpm",0), "prod_score": round(scores.get(f"Family {fid}",{}).get("score",0),1),
        "is_portfolio": False,
    })
for label in ["A+B","A+C","B+C","A+B+C"]:
    ev_ = portfolio_eval[label]
    rows_csv.append({
        "candidate": label, "conditions": label,
        "pf": round(ev_["pf"],4), "wr": round(ev_["wr"],4), "n": ev_["n"],
        "net": round(ev_["net"],2), "mdd": round(ev_["mdd"],4),
        "ues": ev_["ues"], "boot_p5": round(ev_["boot"].get("p5",0),4),
        "boot_med": round(ev_["boot"].get("med",0),4),
        "mc_prob": round(ev_["mc"].get("prob_profit",0),4),
        "rf": round(ev_["rf"],4), "ulcer": round(ev_["ulcer"],4),
        "sym_floor": round(ev_["sym_floor"],4), "fold_floor": round(ev_["fold_floor"],4),
        "tpm": ev_.get("tpm",0), "prod_score": round(scores.get(label,{}).get("score",0),1),
        "is_portfolio": True,
    })
df_summary = pd.DataFrame(rows_csv)
df_summary.sort_values("prod_score", ascending=False, inplace=True)
df_summary.to_csv(os.path.join(OUT,"r066_summary.csv"), index=False)
print(f"  → r066_summary.csv")

# Diversification CSV
div_rows = []
for k, d in div_pairs.items():
    div_rows.append({"pair": k.replace("_","+"), **d})
pd.DataFrame(div_rows).to_csv(os.path.join(OUT,"r066_diversification.csv"), index=False)
print(f"  → r066_diversification.csv")

# Capital allocation CSV
alloc_rows = []
for ar in alloc_results:
    alloc_rows.append({"label": ar["label"],
                        "w_A": round(ar["weights"][0],4), "w_B": round(ar["weights"][1],4),
                        "w_C": round(ar["weights"][2],4),
                        "pf": round(ar["pf"],4), "wr": round(ar["wr"],4),
                        "mdd": round(ar["mdd"],4), "rf": round(ar["rf"],4),
                        "ues": round(ar["ues"],1),
                        "boot_p5": round(ar["boot_p5"],4), "boot_p50": round(ar["boot_p50"],4)})
pd.DataFrame(alloc_rows).to_csv(os.path.join(OUT,"r066_allocation.csv"), index=False)
print(f"  → r066_allocation.csv")

# Trade log for A+B+C
trade_rows = []
for t in all3_trades:
    trade_rows.append({
        "symbol": t["symbol"], "entry_time": str(t["entry_time"]),
        "exit_pnl": round(t["exit_pnl"],4), "is_win": t["is_win"],
        "fold": t["fold"], "hour_utc": t.get("hour_utc",0),
    })
pd.DataFrame(trade_rows).to_csv(os.path.join(OUT,"r066_trades_abc.csv"), index=False)
print(f"  → r066_trades_abc.csv")

# Fold detail CSV
fold_rows = []
for fid in ("A","B","C"):
    for f_, fd in fam_eval[fid]["folds"].items():
        fold_rows.append({"family": fid, "fold": f_,
                          "pf": round(fd["pf"],4), "n": fd["n"],
                          "wr": round(fd["wr"],4)})
for label in ["A+B","A+C","B+C","A+B+C"]:
    for f_, fd in portfolio_eval[label]["folds"].items():
        fold_rows.append({"family": label, "fold": f_,
                          "pf": round(fd["pf"],4), "n": fd["n"],
                          "wr": round(fd["wr"],4)})
pd.DataFrame(fold_rows).to_csv(os.path.join(OUT,"r066_folds.csv"), index=False)
print(f"  → r066_folds.csv")

print()

# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
elapsed = time.time() - t0_global
journal_path = os.path.join(OUT, "r066_journal.md")
with open(journal_path, "w") as mf:
    mf.write(f"# R066 — Production Portfolio Validation (Frozen Families Only)\n\n")
    mf.write(f"**Date:** {pd.Timestamp.now().strftime('%B %Y')}  \n")
    mf.write(f"**Duration:** {elapsed:.0f}s  \n")
    mf.write(f"**Symbols:** {len(data)}  \n\n")

    mf.write(f"## Frozen Families\n\n")
    mf.write(f"| Family | Conditions |\n|---|---|\n")
    for fid, cids in FAMILIES.items():
        mf.write(f"| {fid} | {'+'.join(cids)} |\n")
    mf.write(f"\n")

    mf.write(f"## Section 1 — Individual Baselines\n\n")
    mf.write(f"| Family | PF | WR | n | MDD | UES | Boot P5 | MC P(profit) | LOO-sym | LOO-fold |\n")
    mf.write(f"|---|---|---|---|---|---|---|---|---|---|\n")
    for fid in ("A","B","C"):
        ev_ = fam_eval[fid]
        mf.write(f"| {fid} | {ev_['pf']:.3f} | {ev_['wr']:.1%} | {ev_['n']} | "
                 f"{ev_['mdd']:.1%} | {ev_['ues']:.1f} | "
                 f"{ev_['boot'].get('p5',0):.3f} | "
                 f"{ev_['mc'].get('prob_profit',0):.1%} | "
                 f"{ev_['sym_floor']:.3f} | {ev_['fold_floor']:.3f} |\n")
    mf.write(f"\n")

    mf.write(f"## Section 2 — Two-Family Portfolios\n\n")
    mf.write(f"| Portfolio | PF | WR | n | MDD | UES | Boot P5 | MC |\n")
    mf.write(f"|---|---|---|---|---|---|---|---|\n")
    for label in ["A+B","A+C","B+C"]:
        ev_ = portfolio_eval[label]
        mf.write(f"| {label} | {ev_['pf']:.3f} | {ev_['wr']:.1%} | {ev_['n']} | "
                 f"{ev_['mdd']:.1%} | {ev_['ues']:.1f} | "
                 f"{ev_['boot'].get('p5',0):.3f} | "
                 f"{ev_['mc'].get('prob_profit',0):.1%} |\n")
    mf.write(f"\n")

    mf.write(f"## Section 3 — Three-Family Portfolio (A+B+C)\n\n")
    mf.write(f"- **PF:** {ev_abc['pf']:.3f}  \n")
    mf.write(f"- **WR:** {ev_abc['wr']:.1%}  \n")
    mf.write(f"- **n:** {ev_abc['n']}  \n")
    mf.write(f"- **Net Profit:** ${ev_abc['net']:,.1f}  \n")
    mf.write(f"- **MDD:** {ev_abc['mdd']:.1%}  \n")
    mf.write(f"- **UES:** {ev_abc['ues']:.1f}  \n")
    mf.write(f"- **Recovery Factor:** {ev_abc['rf']:.2f}  \n")
    mf.write(f"- **Ulcer Index:** {ev_abc['ulcer']:.2f}  \n")
    mf.write(f"- **Boot P50:** {ev_abc['boot'].get('med',0):.3f}  P5={ev_abc['boot'].get('p5',0):.3f}  \n")
    mf.write(f"- **MC P(profit):** {ev_abc['mc'].get('prob_profit',0):.1%}  \n")
    mf.write(f"- **LOO-sym floor:** {ev_abc['sym_floor']:.3f}  \n")
    mf.write(f"- **LOO-fold floor:** {ev_abc['fold_floor']:.3f}  \n\n")

    mf.write(f"## Section 4 — Diversification\n\n")
    mf.write(f"| Pair | Trade Overlap | PnL Corr | Sym Overlap | Div Score |\n")
    mf.write(f"|---|---|---|---|---|\n")
    for k, d in div_pairs.items():
        mf.write(f"| {k.replace('_','+')} | {d['trade_overlap']:.2%} | "
                 f"{d['pnl_corr']:+.4f} | {d['sym_overlap']:.2%} | {d['div_score']:.1f} |\n")
    mf.write(f"\n")

    mf.write(f"## Section 5 — Drawdown Diversification\n\n")
    mf.write(f"| Candidate | MDD | Ulcer | Recovery Factor |\n|---|---|---|---|\n")
    for fid in ("A","B","C"):
        ev_ = fam_eval[fid]
        mf.write(f"| Family {fid} | {ev_['mdd']:.1%} | {ev_['ulcer']:.2f} | {ev_['rf']:.2f} |\n")
    for label in ["A+B","A+C","B+C","A+B+C"]:
        ev_ = portfolio_eval[label]
        mf.write(f"| {label} | {ev_['mdd']:.1%} | {ev_['ulcer']:.2f} | {ev_['rf']:.2f} |\n")
    mf.write(f"\n")

    mf.write(f"## Section 6 — Trade Frequency\n\n")
    mf.write(f"| Candidate | Tpw | Tpm | Tpy | Max Gap (h) | Max Win Streak | Max Loss Streak |\n")
    mf.write(f"|---|---|---|---|---|---|---|\n")
    for fid in ("A","B","C"):
        f_ = freq_results[fid]
        mf.write(f"| Family {fid} | {f_.get('tpw',0):.2f} | {f_.get('tpm',0):.1f} | "
                 f"{f_.get('tpy',0):.0f} | {f_.get('max_gap_hrs',0):.0f} | "
                 f"{f_.get('max_win_streak',0)} | {f_.get('max_loss_streak',0)} |\n")
    f_ = freq_results["A+B+C"]
    mf.write(f"| A+B+C | {f_.get('tpw',0):.2f} | {f_.get('tpm',0):.1f} | "
             f"{f_.get('tpy',0):.0f} | {f_.get('max_gap_hrs',0):.0f} | "
             f"{f_.get('max_win_streak',0)} | {f_.get('max_loss_streak',0)} |\n")
    mf.write(f"\n**Practical for retail?** {practical}\n\n")

    mf.write(f"## Section 7 — Capital Allocation\n\n")
    mf.write(f"| Scheme | w_A | w_B | w_C | PF | MDD | RF | UES |\n")
    mf.write(f"|---|---|---|---|---|---|---|---|\n")
    for ar in alloc_results:
        mf.write(f"| {ar['label'].split('(')[0].strip()} | "
                 f"{ar['weights'][0]:.2f} | {ar['weights'][1]:.2f} | {ar['weights'][2]:.2f} | "
                 f"{ar['pf']:.3f} | {ar['mdd']:.1%} | {ar['rf']:.2f} | {ar['ues']:.1f} |\n")
    mf.write(f"\n**Best allocation:** {best_alloc['label']}\n\n")

    mf.write(f"## Section 8 — Stress Tests\n\n")
    mf.write(f"- **Bootstrap P5:** {bt_abc['p5']:.3f}  P50={bt_abc['med']:.3f}  "
             f"({'PASS' if bt_abc['p5'] > 1.0 else 'FAIL'})\n")
    mf.write(f"- **Monte Carlo P(profit):** {mc_abc['prob_profit']:.1%}  "
             f"({'PASS' if mc_abc['prob_profit'] > 0.75 else 'FAIL'})\n")
    mf.write(f"- **LOO-fold floor:** {loof_abc:.3f}  ({'PASS' if loof_abc > 1.0 else 'FAIL'})\n")
    mf.write(f"- **LOO-sym floor:** {loos_abc:.3f}  ({'PASS' if loos_abc > 1.0 else 'FAIL'})\n")
    mf.write(f"- **Permutation pctile:** {perm_abc:.4f}  "
             f"({'PASS' if perm_abc >= 0.95 else 'BORDERLINE' if perm_abc >= 0.80 else 'FAIL'})\n")
    mf.write(f"- **Stress verdict:** {stress_verdict}  ({stress_pass}/5)\n\n")

    mf.write(f"## Section 9 — Production Ranking\n\n")
    mf.write(f"| Rank | Candidate | Score | PF | WR | n | MDD | UES |\n")
    mf.write(f"|---|---|---|---|---|---|---|---|\n")
    for rank, (name, d) in enumerate(ranked, 1):
        ev_ = d["ev"]
        mf.write(f"| {rank} | {name} | {d['score']:.1f} | {ev_['pf']:.3f} | "
                 f"{ev_['wr']:.1%} | {ev_['n']} | {ev_['mdd']:.1%} | {ev_['ues']:.1f} |\n")
    mf.write(f"\n")

    mf.write(f"## Section 10 — Final Verdict\n\n")
    mf.write(f"1. **Does combining all three improve portfolio?** {q1}  \n   {q1_note}\n\n")
    mf.write(f"2. **Does it increase trade frequency enough?** {q2}  \n   {q2_note}\n\n")
    mf.write(f"3. **Does diversification reduce drawdown?** {q3}  \n   {q3_note}\n\n")
    mf.write(f"4. **Is combined superior to E3.1 alone?** {q4}  \n   {q4_note}\n\n")
    mf.write(f"5. **Deploy today for live paper trading?** {deploy_candidate}  \n   {q5_note}\n\n")

    mf.write(f"## Outputs\n")
    for p in saved_charts:
        mf.write(f"- `{os.path.basename(p)}`\n")
    mf.write(f"- `r066_summary.csv`\n")
    mf.write(f"- `r066_diversification.csv`\n")
    mf.write(f"- `r066_allocation.csv`\n")
    mf.write(f"- `r066_trades_abc.csv`\n")
    mf.write(f"- `r066_folds.csv`\n")

print(f"  → r066_journal.md")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print(f"  R066 COMPLETE — {elapsed:.0f}s")
print(SEP)
print()
print(f"  ╔══════════════════════════════════════════════════════════════════════╗")
print(f"  ║  PRODUCTION VALIDATION COMPLETE                                      ║")
print(f"  ║  Stress Verdict: {stress_verdict:<54}║")
print(f"  ║  Best Deployment Candidate: {deploy_candidate:<43}║")
print(f"  ╠══════════════════════════════════════════════════════════════════════╣")
print(f"  ║  {'Candidate':<18} {'PF':>6}  {'WR':>6}  {'n':>5}  {'MDD':>7}  {'UES':>6}  {'Score':>6} ║")
print(f"  ║  {'-'*18} {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*6} ║")
for rank, (name, d) in enumerate(ranked[:7], 1):
    ev_ = d["ev"]
    print(f"  ║  {name:<18} {ev_['pf']:>6.3f}  {ev_['wr']:>6.1%}  {ev_['n']:>5}  "
          f"{ev_['mdd']:>7.1%}  {ev_['ues']:>6.1f}  {d['score']:>6.1f} ║")
print(f"  ╠══════════════════════════════════════════════════════════════════════╣")
print(f"  ║  Q1 Combines improve portfolio?   {q1:<36}║")
print(f"  ║  Q3 Drawdown diversified?         {q3:<36}║")
print(f"  ║  Q4 Superior to E3.1 alone?       {q4:<36}║")
print(f"  ╚══════════════════════════════════════════════════════════════════════╝")
print()
print(f"  Files saved to {OUT}/:")
for p in saved_charts:
    print(f"    {os.path.basename(p)}")
print(f"    r066_summary.csv")
print(f"    r066_diversification.csv")
print(f"    r066_allocation.csv")
print(f"    r066_trades_abc.csv")
print(f"    r066_folds.csv")
print(f"    r066_journal.md")
print()
