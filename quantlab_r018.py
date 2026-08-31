"""
=============================================================================
QUANTLAB AI — RESEARCH #018
Objective : Regime Conditioning of Liquidity Sweep Reversal

NOT a new strategy. NOT a parameter optimisation.

Research Question:
  Does the Liquidity Sweep Reversal strategy have an edge only in specific
  market regimes identified in Research #017?

Regime source  : Research #017 KMeans k=2 (replicated here per timeframe)
  Regime 0 : Low volatility / Trending (majority regime, ~80% of bars)
  Regime 1 : High volatility / Bearish  (minority regime, ~20% of bars)

LSR strategy   : Exact entry/exit rules from Research #010 (quantlab_ai.py)
  - Sweep:   low < prior_5_bar_low  (wick below range)
  - Reclaim: close > prior_5_bar_low (close back above)
  - Bullish: close > open
  - Trend:   close > EMA200
  - Stop:    low of signal bar
  - Target:  entry + 2 × stop_dist  (RR = 2:1)
  - All fees, spread, slippage LOCKED to CONFIG

Data           : Existing cache — no new downloads
Symbols        : BTC, ETH, LINK, XRP, DOGE, LTC, AVAX, BCH (OKX perps)
Timeframe      : 1H (canonical for LSR; also 15m for comparison)
Split          : 70% train (ignored) / 30% OOS (tested)
=============================================================================
"""

import os, sys, math, warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantlab_ai import (
    CONFIG, compute_metrics, monte_carlo,
    append_journal, _journal_row, _verdict_from_metrics,
    calc_ema, calc_atr, calc_adx,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — all execution costs locked to CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID   = "R018"
OUTPUT_FOLDER = CONFIG["OUTPUT_FOLDER"]
CACHE_FOLDER  = CONFIG["CACHE_FOLDER"]
TRAIN_RATIO   = CONFIG["TRAIN_RATIO"]

TAKER_FEE    = CONFIG["TAKER_FEE"]
SPREAD       = CONFIG["SPREAD"] * 0.5
SL_SLIPPAGE  = CONFIG["SL_SLIPPAGE"]
MIN_SL_PCT   = CONFIG["MIN_SL_PCT"]
RR           = CONFIG["RISK_REWARD"]
MAX_LEV      = CONFIG["MAX_LEVERAGE"]
STARTING_CAP = CONFIG["STARTING_CAPITAL"]
RISK_PCT     = CONFIG["RISK_PER_TRADE_PCT"]
LSR_LOOKBACK = CONFIG["LSR_LOOKBACK"]      # 5 bars
EMA_LEN      = CONFIG["EMA_LENGTH"]        # 200
MC_ITER      = CONFIG["MC_ITERATIONS"]     # 1000

SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "LINK-USDT-SWAP", "AVAX-USDT-SWAP",
    "XRP-USDT-SWAP",  "DOGE-USDT-SWAP", "LTC-USDT-SWAP",  "BCH-USDT-SWAP",
]

TIMEFRAMES = [
    {"bar": "1H",  "minutes": 60,  "label": "1-hour"},
    {"bar": "15m", "minutes": 15,  "label": "15-minute"},
]

# R018 regime labels (must match R017 interpretation)
R0_LABEL = "Regime 0 (Low Vol / Trending)"
R1_LABEL = "Regime 1 (High Vol / Bearish)"
MX_LABEL = "Mixed (regime changed)"
OV_LABEL = "Overall"

BG      = "#0F1117"
C_R0    = "#4A90D9"   # Regime 0 colour
C_R1    = "#FF4560"   # Regime 1 colour
C_MX    = "#FFB347"   # Mixed colour
C_OV    = "#00C49A"   # Overall colour
PALETTE = [C_R0, C_R1, C_MX, C_OV]

PROMOTE_PF    = 1.20
PROMOTE_MDD   = 0.30
PROMOTE_MC_PP = 0.60
MIN_TRADES    = 30


# =============================================================================
# SECTION 1 — DATA LOADING (cache only — no new downloads)
# =============================================================================

def _cache_path(symbol: str, bar: str) -> str:
    safe = symbol.replace("-", "_") + f"_{bar}"
    return os.path.join(CACHE_FOLDER, f"{safe}.parquet")


def load_symbol(symbol: str, bar: str) -> pd.DataFrame:
    path = _cache_path(symbol, bar)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache missing: {path}")
    df = pd.read_parquet(path)
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def load_all(bar: str) -> dict:
    data = {}
    for sym in SYMBOLS:
        try:
            data[sym] = load_symbol(sym, bar)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}", flush=True)
    return data


# =============================================================================
# SECTION 2 — R017 REGIME FEATURES (identical to Research #017)
# =============================================================================

FEAT_COLS = [
    "atr_pct", "adx", "ema200_slope", "rv20",
    "ret20", "ret50", "roll_std20", "roll_skew50", "roll_kurt50",
    "bb_width", "dist_ema200", "vol_pct", "momentum", "rsi",
]


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_r017_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exact feature engineering from Research #017."""
    df    = df.copy().reset_index(drop=True)
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["vol"].replace(0, np.nan)

    atr     = calc_atr(df, 14)
    atr_pct = close.rolling(50, min_periods=25).rank(pct=True)  # proxy via close rank
    # Use proper ATR percentile rank
    atr_pct = atr.rolling(50, min_periods=25).rank(pct=True)

    adx      = calc_adx(df, 14)
    ema200   = calc_ema(close, 200)
    ema200_l = ema200.shift(10)
    ema200_slope = (ema200 - ema200_l) / ema200_l.replace(0, np.nan) * 100.0

    log_ret  = np.log(close / close.shift(1))
    rv20     = log_ret.rolling(20).std() * np.sqrt(365 * 24)
    ret20    = (close / close.shift(20) - 1.0) * 100.0
    ret50    = (close / close.shift(50) - 1.0) * 100.0
    roll_std = log_ret.rolling(20).std() * 100.0
    roll_skw = log_ret.rolling(50).skew()
    roll_krt = log_ret.rolling(50).kurt()

    sma20   = close.rolling(20).mean()
    std20   = close.rolling(20).std(ddof=1)
    bb_w    = (4 * std20) / sma20.replace(0, np.nan) * 100.0
    dist_e  = (close - ema200) / ema200.replace(0, np.nan) * 100.0

    log_vol = np.log1p(vol)
    vol_pct = log_vol.rolling(50, min_periods=25).rank(pct=True)
    mom     = (close / close.shift(10) - 1.0) * 100.0
    rsi     = _calc_rsi(close, 14)

    return pd.DataFrame({
        "datetime":    df["datetime"],
        "atr_pct":     atr_pct,
        "adx":         adx,
        "ema200_slope":ema200_slope,
        "rv20":        rv20,
        "ret20":       ret20,
        "ret50":       ret50,
        "roll_std20":  roll_std,
        "roll_skew50": roll_skw,
        "roll_kurt50": roll_krt,
        "bb_width":    bb_w,
        "dist_ema200": dist_e,
        "vol_pct":     vol_pct,
        "momentum":    mom,
        "rsi":         rsi,
    })


def assign_regimes(all_data: dict, bar: str) -> dict:
    """
    Pool all symbol feature matrices, fit KMeans k=2 (matching R017),
    sort clusters by ADX so Regime 0 = low ADX / low vol, Regime 1 = high ADX / high vol.
    Returns {symbol: pd.Series(regime_label, index=datetime)}.
    """
    print(f"\n  Computing R017 regime features for [{bar}]...", flush=True)

    feat_frames = []
    for sym, df in all_data.items():
        fdf = compute_r017_features(df)
        fdf["symbol"] = sym
        feat_frames.append(fdf)

    combined = pd.concat(feat_frames, ignore_index=True)
    clean    = combined.dropna(subset=FEAT_COLS)

    X        = clean[FEAT_COLS].values
    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(X)

    print(f"  Fitting KMeans k=2 on {len(X_sc):,} bars...", flush=True)
    km  = KMeans(n_clusters=2, n_init=10, random_state=42)
    raw_labels = km.fit_predict(X_sc)

    # Sort so that Regime 0 = lower ADX (calm), Regime 1 = higher ADX (volatile)
    adx_col = FEAT_COLS.index("adx")
    mean_adx = [X[raw_labels == k, adx_col].mean() for k in range(2)]
    # cluster with lower mean ADX → Regime 0
    if mean_adx[0] <= mean_adx[1]:
        label_map = {0: 0, 1: 1}
    else:
        label_map = {0: 1, 1: 0}

    clean["_raw_label"] = raw_labels
    clean["regime"]     = clean["_raw_label"].map(label_map)

    # Unpack per symbol: datetime→regime
    regime_maps = {}
    for sym in all_data:
        sub = clean[clean["symbol"] == sym][["datetime", "regime"]]
        regime_maps[sym] = sub.set_index("datetime")["regime"]

    # Print cluster info
    for k in range(2):
        mapped = label_map[k]
        adx_m  = X[raw_labels == k, adx_col].mean()
        rv_m   = X[raw_labels == k, FEAT_COLS.index("rv20")].mean()
        freq   = (raw_labels == k).sum() / len(raw_labels) * 100
        print(f"  Cluster {k} → Regime {mapped}:  ADX={adx_m:.1f}  RV={rv_m:.4f}  freq={freq:.1f}%")

    return regime_maps


# =============================================================================
# SECTION 3 — LSR INDICATORS (exact from quantlab_ai.py add_indicators)
# =============================================================================

def add_lsr_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute only the indicators needed for LSR:
      - ema200
      - lsr_prior_low (min of previous LSR_LOOKBACK bar lows)
    Identical to quantlab_ai.add_indicators() for LSR columns.
    """
    df = df.copy()
    df["ema200"]       = calc_ema(df["close"], EMA_LEN)
    df["lsr_prior_low"] = df["low"].shift(1).rolling(LSR_LOOKBACK).min()
    return df


def signal_lsr(df: pd.DataFrame) -> pd.Series:
    """
    Exact Liquidity Sweep Reversal signal from quantlab_ai.strategy_lsr.
    No modifications.
    """
    sweep   = df["low"]   < df["lsr_prior_low"]
    reclaim = df["close"] > df["lsr_prior_low"]
    bullish = df["close"] > df["open"]
    trend   = df["close"] > df["ema200"]
    return sweep & reclaim & bullish & trend


# =============================================================================
# SECTION 4 — REGIME-TAGGED BACKTEST ENGINE
# Identical execution to quantlab_ai.run_backtest; adds regime fields per trade.
# =============================================================================

def attach_regime_labels(df_oos: pd.DataFrame, regime_map: pd.Series) -> np.ndarray:
    """
    Tag every OOS bar with its regime label using int64 UTC nanosecond timestamps
    as the join key.  This is fully timezone-agnostic and works identically for
    1H, 15m, or any other timeframe regardless of how the parquet stores timestamps.

    Returns int array of shape (len(df_oos),) with values 0, 1, or -1 (unknown).
    """
    if regime_map is None or len(regime_map) == 0:
        return np.full(len(df_oos), -1, dtype=int)

    def _to_utc_ns(dt_col: pd.Series) -> np.ndarray:
        """Convert any datetime series to int64 UTC nanoseconds (tz-agnostic)."""
        s = pd.to_datetime(dt_col)
        if s.dt.tz is not None:
            # tz-aware → convert to UTC, then strip tz so .astype(int64) is safe
            s = s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s.values.astype(np.int64)

    reg_df = regime_map.reset_index()
    reg_df.columns = ["datetime", "regime"]

    reg_ns = _to_utc_ns(reg_df["datetime"])
    oos_ns = _to_utc_ns(df_oos["datetime"])

    # O(1) lookup: utc_nanoseconds → regime label
    lookup: dict = dict(zip(reg_ns.tolist(), reg_df["regime"].values.astype(int).tolist()))

    result = np.array([lookup.get(int(ns), -1) for ns in oos_ns], dtype=int)
    return result


def run_lsr_backtest_regime_tagged(
    df: pd.DataFrame,
    regime_arr: np.ndarray,
    label: str,
    bar_minutes: int,
) -> list:
    """
    Event-driven backtest of LSR strategy.
    ALL execution parameters locked to CONFIG.
    Adds per-trade: entry_regime, exit_regime, regime_changed.
    regime_arr: int array aligned to df rows (0, 1, or -1)
    """
    signals = signal_lsr(df)
    n       = len(df)

    in_pos     = False
    ep         = 0.0
    sl         = 0.0
    tp         = 0.0
    entry_time = None
    entry_idx  = -1
    pos_size   = 0.0
    entry_reg  = -1
    capital    = STARTING_CAP
    trades     = []

    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    open_arr  = df["open"].values
    dt_arr    = df["datetime"].values

    for i in range(1, n):
        hi = high_arr[i]
        lo = low_arr[i]

        if in_pos:
            sl_hit = lo <= sl
            tp_hit = hi >= tp

            if sl_hit or tp_hit:
                if sl_hit:
                    exit_price = sl * (1.0 - SL_SLIPPAGE)
                    exit_type  = "SL"
                else:
                    exit_price = tp
                    exit_type  = "TP"

                sl_dist   = ep - sl
                gross_pnl = (exit_price - ep) * pos_size
                ne        = ep * pos_size
                nx        = exit_price * pos_size
                cost_fee  = (ne + nx) * TAKER_FEE
                cost_spd  = (ne + nx) * SPREAD
                cost_slip = (sl - exit_price) * pos_size if exit_type == "SL" else 0.0
                net_pnl   = gross_pnl - cost_fee - cost_spd - cost_slip
                r_mult    = (exit_price - ep) / sl_dist if sl_dist > 0 else 0.0
                hold_min  = (i - entry_idx) * bar_minutes

                exit_reg   = int(regime_arr[i])
                regime_chg = (exit_reg != entry_reg) and (exit_reg != -1) and (entry_reg != -1)

                trades.append({
                    "label":           label,
                    "entry_time":      entry_time,
                    "exit_time":       pd.Timestamp(dt_arr[i]),
                    "entry_price":     ep,
                    "exit_price":      exit_price,
                    "stop_loss":       sl,
                    "take_profit":     tp,
                    "pnl":             net_pnl,
                    "r_multiple":      r_mult,
                    "fees":            cost_fee,
                    "spread_cost":     cost_spd,
                    "sl_slippage":     cost_slip,
                    "holding_minutes": hold_min,
                    "funding_windows_crossed": int(hold_min / 480),
                    "win":             exit_type == "TP",
                    "exit_type":       exit_type,
                    "entry_regime":    entry_reg,
                    "exit_regime":     exit_reg,
                    "regime_changed":  regime_chg,
                })
                capital  += net_pnl
                in_pos    = False
            continue

        # New signal on previous bar
        if not signals.iloc[i - 1]:
            continue

        entry_ep  = open_arr[i]
        sl_price  = df.iloc[i - 1]["low"]
        sl_dist_v = entry_ep - sl_price

        if sl_dist_v <= 0 or sl_dist_v / entry_ep < MIN_SL_PCT:
            continue

        tp_price     = entry_ep + RR * sl_dist_v
        risk_dollars = capital * RISK_PCT
        size         = min(risk_dollars / sl_dist_v, (capital * MAX_LEV) / entry_ep)

        ep         = entry_ep
        sl         = sl_price
        tp         = tp_price
        pos_size   = size
        entry_time = pd.Timestamp(dt_arr[i])
        entry_idx  = i
        in_pos     = True
        entry_reg  = int(regime_arr[i])

    return trades


# =============================================================================
# SECTION 5 — TRADE SPLITTING BY REGIME
# =============================================================================

def split_by_regime(trades: list) -> dict:
    """
    Split trade list by entry_regime and transition status.
    Categories:
      r0      : entered in Regime 0 (actionable filter — you know entry regime)
      r1      : entered in Regime 1
      r0_pure : entered AND exited in Regime 0 (analysis only)
      r1_pure : entered AND exited in Regime 1
      mixed   : regime changed during trade
      overall : all trades
    """
    all_t   = trades
    r0      = [t for t in trades if t["entry_regime"] == 0]
    r1      = [t for t in trades if t["entry_regime"] == 1]
    r0_pure = [t for t in trades if t["entry_regime"] == 0 and t["exit_regime"] == 0]
    r1_pure = [t for t in trades if t["entry_regime"] == 1 and t["exit_regime"] == 1]
    mixed   = [t for t in trades if t["regime_changed"]]

    return {
        "overall": all_t,
        "r0":      r0,
        "r1":      r1,
        "r0_pure": r0_pure,
        "r1_pure": r1_pure,
        "mixed":   mixed,
    }


# =============================================================================
# SECTION 6 — PER-REGIME METRICS
# =============================================================================

def regime_metrics(trades: list, label: str, bar_minutes: int) -> dict:
    """
    Compute full metrics for a trade slice.
    Extended version of compute_metrics supporting variable bar_minutes.
    """
    if not trades:
        return {
            "label": label, "n_trades": 0, "net_profit": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0,
            "avg_r": 0.0, "expectancy_r": 0.0,
            "max_drawdown": 0.0, "sharpe": 0.0,
            "avg_hold_minutes": 0.0,
            "equity":    np.array([STARTING_CAP]),
            "drawdown":  np.array([0.0]),
            "pnls":      np.array([]),
            "r_multiples": np.array([]),
            "trades_df": pd.DataFrame(),
        }

    df   = pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)
    pnls = df["pnl"].values
    wins = df["win"].values.astype(bool)
    rmul = df["r_multiple"].values
    n    = len(pnls)
    n_win = int(wins.sum())
    n_los = n - n_win

    gw = pnls[wins].sum() if n_win else 0.0
    gl = abs(pnls[~wins].sum()) if n_los else 1e-9
    pf = gw / gl if gl > 0 else float("inf")
    wr = n_win / n

    equity = STARTING_CAP + np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    mdd    = dd.min()

    std    = np.std(pnls, ddof=1) if n > 1 else 0.0
    bpy    = 365 * 24 * 60 / bar_minutes
    sharpe = pnls.mean() / std * math.sqrt(bpy) if std > 0 else 0.0
    exp_r  = wr * RR - (1.0 - wr)

    return {
        "label":            label,
        "n_trades":         n,
        "net_profit":       float(pnls.sum()),
        "profit_factor":    pf,
        "win_rate":         wr,
        "avg_r":            float(rmul.mean()),
        "expectancy_r":     exp_r,
        "max_drawdown":     mdd,
        "sharpe":           sharpe,
        "avg_hold_minutes": df["holding_minutes"].mean(),
        "equity":           equity,
        "drawdown":         dd,
        "pnls":             pnls,
        "r_multiples":      rmul,
        "trades_df":        df,
    }


# =============================================================================
# SECTION 7 — STATISTICAL TESTS
# =============================================================================

def statistical_tests(trades_r0: list, trades_r1: list) -> dict:
    """
    Compare Regime 0 vs Regime 1 trade R-multiples.
    Tests: Mann-Whitney U, KS, Bootstrap CI, Cohen's d
    """
    rmul_r0 = np.array([t["r_multiple"] for t in trades_r0]) if trades_r0 else np.array([])
    rmul_r1 = np.array([t["r_multiple"] for t in trades_r1]) if trades_r1 else np.array([])

    result = {
        "n_r0": len(rmul_r0),
        "n_r1": len(rmul_r1),
        "mean_r0": float(rmul_r0.mean()) if len(rmul_r0) else np.nan,
        "mean_r1": float(rmul_r1.mean()) if len(rmul_r1) else np.nan,
        "mwu_stat": np.nan, "mwu_p": np.nan,
        "ks_stat":  np.nan, "ks_p":  np.nan,
        "cohens_d": np.nan,
        "boot_ci_r0": (np.nan, np.nan),
        "boot_ci_r1": (np.nan, np.nan),
        "significant_mwu":  False,
        "significant_ks":   False,
        "significant_any":  False,
    }

    # Mann-Whitney U
    if len(rmul_r0) >= 5 and len(rmul_r1) >= 5:
        mwu_stat, mwu_p = scipy_stats.mannwhitneyu(rmul_r0, rmul_r1, alternative="two-sided")
        result["mwu_stat"] = float(mwu_stat)
        result["mwu_p"]    = float(mwu_p)
        result["significant_mwu"] = mwu_p < 0.05

    # KS Test
    if len(rmul_r0) >= 5 and len(rmul_r1) >= 5:
        ks_stat, ks_p = scipy_stats.ks_2samp(rmul_r0, rmul_r1)
        result["ks_stat"] = float(ks_stat)
        result["ks_p"]    = float(ks_p)
        result["significant_ks"] = ks_p < 0.05

    # Cohen's d
    if len(rmul_r0) >= 2 and len(rmul_r1) >= 2:
        pool_std = math.sqrt((rmul_r0.std(ddof=1)**2 + rmul_r1.std(ddof=1)**2) / 2 + 1e-9)
        result["cohens_d"] = float((rmul_r0.mean() - rmul_r1.mean()) / pool_std)

    # Bootstrap CI (mean R-multiple, 95%)
    def _boot_ci(arr, n_boot=5000):
        if len(arr) < 3:
            return (np.nan, np.nan)
        rng   = np.random.RandomState(42)
        means = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
        return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    result["boot_ci_r0"] = _boot_ci(rmul_r0)
    result["boot_ci_r1"] = _boot_ci(rmul_r1)

    result["significant_any"] = result["significant_mwu"] or result["significant_ks"]
    return result


# =============================================================================
# SECTION 8 — TRANSITION ANALYSIS (trade-level)
# =============================================================================

def transition_analysis(all_trades: list) -> dict:
    """
    Trade-level regime transition matrix:
      row=entry_regime, col=exit_regime
    Also compute win rates split by: stayed vs changed.
    """
    if not all_trades:
        return {}

    df = pd.DataFrame(all_trades)
    valid = df[(df["entry_regime"] != -1) & (df["exit_regime"] != -1)]

    trans_counts = np.zeros((2, 2), dtype=float)
    for _, row in valid.iterrows():
        er = int(row["entry_regime"])
        xr = int(row["exit_regime"])
        if 0 <= er <= 1 and 0 <= xr <= 1:
            trans_counts[er, xr] += 1

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_probs = trans_counts / row_sums

    stayed  = valid[~valid["regime_changed"]]
    changed = valid[valid["regime_changed"]]

    def _wr(sub): return sub["win"].mean() if len(sub) else np.nan
    def _pf(sub):
        pw = sub[sub["win"]]["pnl"].sum()
        pl = abs(sub[~sub["win"]]["pnl"].sum()) or 1e-9
        return pw / pl if len(sub) else np.nan

    return {
        "counts":      trans_counts,
        "probs":       trans_probs,
        "n_stayed":    len(stayed),
        "n_changed":   len(changed),
        "wr_stayed":   _wr(stayed),
        "wr_changed":  _wr(changed),
        "pf_stayed":   _pf(stayed),
        "pf_changed":  _pf(changed),
        "r00": int(trans_counts[0, 0]),
        "r01": int(trans_counts[0, 1]),
        "r10": int(trans_counts[1, 0]),
        "r11": int(trans_counts[1, 1]),
    }


# =============================================================================
# SECTION 9 — VISUALISATIONS
# =============================================================================

def _ax(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.tick_params(colors="#AAA", labelsize=8)
    ax.xaxis.label.set_color("#AAA")
    ax.yaxis.label.set_color("#AAA")
    ax.title.set_color("#EEE")
    return ax


def _save(fig, name: str):
    path = os.path.join(OUTPUT_FOLDER, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


# ── Chart 1: Equity curves by regime ─────────────────────────────────────────

def plot_equity_by_regime(metrics: dict, tf_bar: str):
    slices = ["overall", "r0", "r1", "mixed"]
    labels = [OV_LABEL, R0_LABEL, R1_LABEL, MX_LABEL]
    colors = [C_OV, C_R0, C_R1, C_MX]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    _ax(ax)

    for slc, lbl, col in zip(slices, labels, colors):
        m = metrics.get(slc)
        if m and m["n_trades"] > 0:
            eq = m["equity"]
            ax.plot(np.arange(len(eq)), eq, color=col, lw=1.8,
                    label=f"{lbl}  n={m['n_trades']}  PF={m['profit_factor']:.3f}")

    ax.axhline(STARTING_CAP, color="#555", lw=0.7, ls="--")
    ax.set_title(f"LSR Equity Curve by Regime [{tf_bar}]", fontsize=11)
    ax.set_xlabel("Trade #"); ax.set_ylabel("Capital ($)")
    ax.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="#EEE")
    _save(fig, f"r018_equity_{tf_bar}.png")


# ── Chart 2: Drawdown by regime ───────────────────────────────────────────────

def plot_drawdown_by_regime(metrics: dict, tf_bar: str):
    slices = ["overall", "r0", "r1", "mixed"]
    labels = ["Overall", "Regime 0", "Regime 1", "Mixed"]
    colors = [C_OV, C_R0, C_R1, C_MX]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, slc, lbl, col in zip(axes.flat, slices, labels, colors):
        _ax(ax)
        m = metrics.get(slc)
        if m and m["n_trades"] > 0:
            dd = m["drawdown"]
            ax.fill_between(range(len(dd)), dd * 100, 0, color=col, alpha=0.55)
            ax.plot(dd * 100, color=col, lw=1)
            ax.axhline(-20, color="#FFF", lw=0.6, ls="--", alpha=0.5)
        ax.set_title(f"{lbl}  MDD={m['max_drawdown']*100:.1f}%" if m else lbl, fontsize=9)
        ax.set_ylabel("DD %")

    fig.suptitle(f"LSR Drawdown by Regime [{tf_bar}]", color="#EEE", fontsize=11, y=1.01)
    _save(fig, f"r018_drawdown_{tf_bar}.png")


# ── Chart 3 & 4: Profit Factor + Win Rate comparison ─────────────────────────

def plot_pf_winrate(sym_regime_metrics: dict, tf_bar: str):
    """Bar chart of PF and WR per regime per symbol."""
    symbols = list(sym_regime_metrics.keys())
    short   = [s.split("-")[0] for s in symbols]
    x       = np.arange(len(symbols))
    w       = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, metric, title, refline in [
        (axes[0], "profit_factor", "Profit Factor by Regime", PROMOTE_PF),
        (axes[1], "win_rate",      "Win Rate by Regime",       0.333),
    ]:
        _ax(ax)
        for shift, slc, col, lbl in [
            (-1.5*w, "overall", C_OV, "Overall"),
            (-0.5*w, "r0",      C_R0, "Regime 0"),
            ( 0.5*w, "r1",      C_R1, "Regime 1"),
            ( 1.5*w, "mixed",   C_MX, "Mixed"),
        ]:
            vals = []
            for sym in symbols:
                m = sym_regime_metrics[sym].get(slc)
                v = m[metric] if m and m["n_trades"] > 0 else 0.0
                if not np.isfinite(v): v = 0.0
                vals.append(v * (100 if metric == "win_rate" else 1))
            ax.bar(x + shift, vals, w, color=col, alpha=0.85, label=lbl)

        ax.axhline(refline * (100 if metric == "win_rate" else 1),
                   color="#FFF", lw=0.8, ls="--", label="Threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=9, color="#EEE")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="#EEE")

    fig.suptitle(f"LSR PF & Win Rate by Regime [{tf_bar}]", color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"r018_pf_winrate_{tf_bar}.png")


# ── Chart 5: Regime transition matrix (trade-level) ──────────────────────────

def plot_trade_transition_matrix(trans: dict, tf_bar: str):
    if not trans or "probs" not in trans:
        return
    probs  = trans["probs"]
    counts = trans["counts"]
    labels = ["Regime 0", "Regime 1"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # Probability matrix
    ax = axes[0]; _ax(ax)
    im = ax.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="#EEE")
    ax.set_yticklabels(labels, color="#EEE")
    ax.set_xlabel("Exit Regime"); ax.set_ylabel("Entry Regime")
    ax.set_title("Trade Regime Transition Probabilities")
    plt.colorbar(im, ax=ax, fraction=0.05)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{probs[i,j]:.2f}\n(n={int(counts[i,j])})",
                    ha="center", va="center", fontsize=10,
                    color="white" if probs[i,j] > 0.5 else "#333")

    # WR: stayed vs changed
    ax2 = axes[1]; _ax(ax2)
    cats   = ["Regime stayed\n(n={})".format(trans["n_stayed"]),
              "Regime changed\n(n={})".format(trans["n_changed"])]
    wr_vals = [trans.get("wr_stayed", 0) or 0, trans.get("wr_changed", 0) or 0]
    pf_vals = [trans.get("pf_stayed", 0) or 0, trans.get("pf_changed", 0) or 0]
    x = np.arange(2)
    ax2.bar(x - 0.2, [v * 100 for v in wr_vals], 0.35, color=[C_R0, C_R1], alpha=0.85, label="Win Rate %")
    ax2.bar(x + 0.2, pf_vals, 0.35, color=[C_OV, C_MX], alpha=0.85, label="Profit Factor")
    ax2.axhline(33.3, color="#4A90D9", lw=0.7, ls="--")
    ax2.axhline(PROMOTE_PF, color="#00C49A", lw=0.7, ls="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cats, fontsize=9, color="#EEE")
    ax2.set_title("Win Rate & PF: Stayed vs Changed Regime")
    ax2.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="#EEE")

    fig.suptitle(f"LSR Trade-Level Regime Transitions [{tf_bar}]",
                 color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"r018_transitions_{tf_bar}.png")


# ── Chart 6: Trade distribution by regime per symbol ─────────────────────────

def plot_trade_distribution(sym_splits: dict, tf_bar: str):
    symbols = list(sym_splits.keys())
    short   = [s.split("-")[0] for s in symbols]
    x       = np.arange(len(symbols))

    r0_n = [len(sym_splits[s]["r0"]) for s in symbols]
    r1_n = [len(sym_splits[s]["r1"]) for s in symbols]
    mx_n = [len(sym_splits[s]["mixed"]) for s in symbols]

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    _ax(ax)

    bot = np.zeros(len(symbols))
    for vals, col, lbl in [(r0_n, C_R0, "Regime 0"), (r1_n, C_R1, "Regime 1"), (mx_n, C_MX, "Mixed")]:
        ax.bar(x, vals, bottom=bot, color=col, alpha=0.85, label=lbl, width=0.6)
        for xi, v in enumerate(vals):
            if v > 0:
                ax.text(xi, bot[xi] + v / 2, str(v), ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        bot += np.array(vals, dtype=float)

    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9, color="#EEE")
    ax.set_ylabel("Number of Trades")
    ax.set_title(f"LSR Trade Count Distribution by Regime per Symbol [{tf_bar}]", fontsize=10)
    ax.legend(fontsize=9, facecolor="#1A1D24", edgecolor="#444", labelcolor="#EEE")
    _save(fig, f"r018_trade_dist_{tf_bar}.png")


# ── Chart 7: Monte Carlo by regime ───────────────────────────────────────────

def plot_monte_carlo_by_regime(metrics: dict, tf_bar: str):
    slices = ["r0", "r1", "mixed", "overall"]
    labels = ["Regime 0", "Regime 1", "Mixed", "Overall"]
    colors = [C_R0, C_R1, C_MX, C_OV]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, slc, lbl, col in zip(axes.flat, slices, labels, colors):
        _ax(ax)
        m = metrics.get(slc)
        if m and m["n_trades"] >= 5:
            mc     = monte_carlo(m["pnls"], MC_ITER)
            finals = mc["final_equities"]
            if len(np.unique(finals)) > 1:
                ax.hist(finals, bins=min(40, len(np.unique(finals))),
                        color=col, alpha=0.75, density=True)
            else:
                ax.axvline(finals[0], color=col, lw=3)
            ax.axvline(STARTING_CAP, color="#FFF", lw=1.2, ls="--", label="Start")
            ax.axvline(mc["median"],  color="#FFD700", lw=1, ls="-",
                       label=f"Median ${mc['median']:,.0f}")
            ax.set_title(f"{lbl}  PP={mc['prob_profit']*100:.1f}%  n={m['n_trades']}", fontsize=9)
            ax.set_xlabel("Final Equity ($)")
            ax.legend(fontsize=7, facecolor="#1A1D24", edgecolor="#444", labelcolor="#EEE")
        else:
            ax.text(0.5, 0.5, f"{lbl}\nInsufficient data",
                    ha="center", va="center", color="#AAA", transform=ax.transAxes)

    fig.suptitle(f"LSR Monte Carlo by Regime [{tf_bar}]  (n={MC_ITER:,} permutations)",
                 color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"r018_monte_carlo_{tf_bar}.png")


# ── Chart 8: Bootstrap confidence intervals ───────────────────────────────────

def plot_bootstrap_ci(stat_result: dict, metrics: dict, tf_bar: str):
    slices = ["overall", "r0", "r1", "mixed"]
    labels = ["Overall", "Regime 0", "Regime 1", "Mixed"]
    colors = [C_OV, C_R0, C_R1, C_MX]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # Panel 1: Mean R-multiple with 95% CI per regime
    ax = axes[0]; _ax(ax)
    y_means, y_lo, y_hi, tick_labels, tick_colors = [], [], [], [], []

    # Overall
    m_ov = metrics.get("overall")
    if m_ov and m_ov["n_trades"] > 0:
        rmul = m_ov["r_multiples"]
        rng  = np.random.RandomState(42)
        boot = [rng.choice(rmul, len(rmul), replace=True).mean() for _ in range(5000)]
        y_means.append(rmul.mean()); y_lo.append(np.percentile(boot, 2.5))
        y_hi.append(np.percentile(boot, 97.5))
        tick_labels.append("Overall"); tick_colors.append(C_OV)

    for slc, lbl, col in [("r0", "Regime 0", C_R0), ("r1", "Regime 1", C_R1), ("mixed", "Mixed", C_MX)]:
        m = metrics.get(slc)
        if m and m["n_trades"] >= 5:
            rmul = m["r_multiples"]
            rng  = np.random.RandomState(42)
            boot = [rng.choice(rmul, len(rmul), replace=True).mean() for _ in range(5000)]
            y_means.append(rmul.mean()); y_lo.append(np.percentile(boot, 2.5))
            y_hi.append(np.percentile(boot, 97.5))
            tick_labels.append(lbl); tick_colors.append(col)

    if y_means:
        x_pos = np.arange(len(y_means))
        for xi, (m_v, lo, hi, col) in enumerate(zip(y_means, y_lo, y_hi, tick_colors)):
            ax.plot([xi, xi], [lo, hi], color=col, lw=3, alpha=0.7)
            ax.scatter([xi], [m_v], color=col, s=80, zorder=5)
        ax.axhline(0, color="#FFF", lw=0.8, ls="--")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tick_labels, color="#EEE", fontsize=9)
        ax.set_ylabel("Mean R-Multiple")
        ax.set_title("Bootstrap 95% CI — Mean R-Multiple by Regime", fontsize=9)

    # Panel 2: statistical test results
    ax2 = axes[1]; _ax(ax2)
    ax2.axis("off")

    lines = [
        "Statistical Tests — Regime 0 vs Regime 1 R-Multiples",
        "",
        f"Regime 0 trades : n={stat_result.get('n_r0', 0)}  mean={stat_result.get('mean_r0', np.nan):.3f}",
        f"Regime 1 trades : n={stat_result.get('n_r1', 0)}  mean={stat_result.get('mean_r1', np.nan):.3f}",
        "",
        f"Mann-Whitney U  : stat={stat_result.get('mwu_stat', np.nan):.1f}  p={stat_result.get('mwu_p', np.nan):.4f}"
          + ("  *" if stat_result.get("significant_mwu") else ""),
        f"KS Test         : stat={stat_result.get('ks_stat', np.nan):.3f}  p={stat_result.get('ks_p', np.nan):.4f}"
          + ("  *" if stat_result.get("significant_ks") else ""),
        f"Cohen's d       : {stat_result.get('cohens_d', np.nan):.3f}",
        "",
        f"Bootstrap CI R0 : [{stat_result['boot_ci_r0'][0]:.3f}, {stat_result['boot_ci_r0'][1]:.3f}]",
        f"Bootstrap CI R1 : [{stat_result['boot_ci_r1'][0]:.3f}, {stat_result['boot_ci_r1'][1]:.3f}]",
        "",
        "* p < 0.05 — statistically significant",
        "" if not stat_result.get("significant_any") else "SIGNIFICANT DIFFERENCE DETECTED",
    ]

    for i, line in enumerate(lines):
        color = "#FFD700" if "SIGNIFICANT" in line else ("#EEE" if i == 0 else "#AAA")
        ax2.text(0.05, 0.95 - i * 0.065, line, transform=ax2.transAxes,
                 fontsize=9, color=color, va="top", fontfamily="monospace")

    fig.suptitle(f"LSR Bootstrap CI & Statistical Tests [{tf_bar}]",
                 color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"r018_bootstrap_ci_{tf_bar}.png")


# =============================================================================
# SECTION 10 — CONSOLE REPORT
# =============================================================================

def print_report(metrics: dict, stat_result: dict, trans: dict,
                 sym_regime_metrics: dict, tf_cfg: dict):
    bar   = tf_cfg["bar"]
    label = tf_cfg["label"]
    W     = 110

    print()
    print("=" * W)
    print(f"  QUANTLAB AI — RESEARCH #018  [{label}]")
    print(f"  LSR Regime Conditioning — {label} candles")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * W)
    print()

    slices = [
        ("overall", "Overall (no filter)",       C_OV),
        ("r0",      "Entry in Regime 0",          C_R0),
        ("r1",      "Entry in Regime 1",          C_R1),
        ("r0_pure", "Pure Regime 0 (in+out R0)", "#87CEEB"),
        ("r1_pure", "Pure Regime 1 (in+out R1)", "#FF9999"),
        ("mixed",   "Mixed (regime changed)",     C_MX),
    ]

    hdr = f"  {'Slice':<32} {'n':>5} {'WR%':>7} {'PF':>8} {'ExpR':>8} {'NetP':>10} {'MDD%':>7} {'Sharpe':>8}"
    print(hdr)
    print("  " + "-" * (W - 2))

    for slc, lbl, _ in slices:
        m = metrics.get(slc)
        if not m:
            continue
        n      = m["n_trades"]
        flag   = "" if n >= MIN_TRADES else " (low n)"
        if n == 0:
            print(f"  {lbl:<32} {'0':>5}  {'—':>6}  {'—':>7}  {'—':>7}  {'—':>9}  {'—':>6}  {'—':>7}")
        else:
            print(f"  {lbl:<32} {n:>5}  "
                  f"{m['win_rate']*100:>6.1f}%  "
                  f"{m['profit_factor']:>7.3f}  "
                  f"{m['expectancy_r']:>+7.3f}  "
                  f"${m['net_profit']:>9,.0f}  "
                  f"{m['max_drawdown']*100:>6.1f}%  "
                  f"{m['sharpe']:>7.2f}{flag}")
    print()

    if trans:
        print(f"  ── Transition Analysis ───────────────────────────────────────────")
        print(f"  R0→R0 (stayed R0): {trans['r00']:4d}  "
              f"R0→R1 (left R0):   {trans['r01']:4d}")
        print(f"  R1→R1 (stayed R1): {trans['r11']:4d}  "
              f"R1→R0 (left R1):   {trans['r10']:4d}")
        wr_st = trans.get("wr_stayed", np.nan)
        wr_ch = trans.get("wr_changed", np.nan)
        pf_st = trans.get("pf_stayed", np.nan)
        pf_ch = trans.get("pf_changed", np.nan)
        if not math.isnan(wr_st):
            print(f"  Stayed in regime   : WR={wr_st*100:.1f}%  PF={pf_st:.3f}  n={trans['n_stayed']}")
        if not math.isnan(wr_ch):
            print(f"  Regime changed     : WR={wr_ch*100:.1f}%  PF={pf_ch:.3f}  n={trans['n_changed']}")
        print()

    if stat_result.get("n_r0", 0) + stat_result.get("n_r1", 0) >= 10:
        print(f"  ── Statistical Tests (Regime 0 vs Regime 1 R-multiples) ─────────")
        print(f"  Regime 0 mean R : {stat_result.get('mean_r0', np.nan):.3f}  "
              f"(n={stat_result.get('n_r0', 0)})")
        print(f"  Regime 1 mean R : {stat_result.get('mean_r1', np.nan):.3f}  "
              f"(n={stat_result.get('n_r1', 0)})")
        print(f"  Mann-Whitney U  : p={stat_result.get('mwu_p', np.nan):.4f}"
              + ("  ✓ significant" if stat_result.get("significant_mwu") else "  ✗ not significant"))
        print(f"  KS Test         : p={stat_result.get('ks_p', np.nan):.4f}"
              + ("  ✓ significant" if stat_result.get("significant_ks") else "  ✗ not significant"))
        print(f"  Cohen's d       : {stat_result.get('cohens_d', np.nan):.3f}"
              " (|d|>0.5=medium, |d|>0.8=large)")
        print(f"  Bootstrap CI R0 : [{stat_result['boot_ci_r0'][0]:.3f}, {stat_result['boot_ci_r0'][1]:.3f}]")
        print(f"  Bootstrap CI R1 : [{stat_result['boot_ci_r1'][0]:.3f}, {stat_result['boot_ci_r1'][1]:.3f}]")
        print()


def print_eight_questions(all_tf: dict):
    W = 110
    print()
    print("=" * W)
    print("  R018 — EIGHT FINAL QUESTIONS")
    print("  LSR Regime Conditioning Analysis")
    print("=" * W)
    print()

    def _yn(c): return "YES ✓" if c else "NO  ✗"
    def _safe(m, k, default=0.0):
        if not m or m["n_trades"] == 0: return default
        v = m.get(k, default)
        return v if np.isfinite(v) else default

    for bar, res in all_tf.items():
        m     = res["metrics"]
        trans = res["trans"]
        stat  = res["stat"]
        label = res["label"]

        m_ov = m.get("overall")
        m_r0 = m.get("r0")
        m_r1 = m.get("r1")
        m_mx = m.get("mixed")

        pf_ov = _safe(m_ov, "profit_factor")
        pf_r0 = _safe(m_r0, "profit_factor")
        pf_r1 = _safe(m_r1, "profit_factor")
        pf_mx = _safe(m_mx, "profit_factor")

        wr_ov = _safe(m_ov, "win_rate")
        wr_r0 = _safe(m_r0, "win_rate")
        wr_r1 = _safe(m_r1, "win_rate")

        mdd_ov = _safe(m_ov, "max_drawdown")
        mdd_r0 = _safe(m_r0, "max_drawdown")
        mdd_r1 = _safe(m_r1, "max_drawdown")

        exp_ov = _safe(m_ov, "expectancy_r")
        exp_r0 = _safe(m_r0, "expectancy_r")
        exp_r1 = _safe(m_r1, "expectancy_r")

        n_r0   = m_r0["n_trades"] if m_r0 else 0
        n_r1   = m_r1["n_trades"] if m_r1 else 0

        best_regime     = "Regime 0" if pf_r0 >= pf_r1 else "Regime 1"
        best_pf         = max(pf_r0, pf_r1)
        best_exp        = exp_r0 if pf_r0 >= pf_r1 else exp_r1
        best_wr         = wr_r0  if pf_r0 >= pf_r1 else wr_r1
        best_mdd        = mdd_r0 if pf_r0 >= pf_r1 else mdd_r1
        best_n          = n_r0   if pf_r0 >= pf_r1 else n_r1
        worst_regime    = "Regime 1" if pf_r0 >= pf_r1 else "Regime 0"
        worst_pf        = min(pf_r0, pf_r1)

        wr_stayed  = trans.get("wr_stayed",  0) or 0
        wr_changed = trans.get("wr_changed", 0) or 0
        pf_stayed  = trans.get("pf_stayed",  0) or 0
        pf_changed = trans.get("pf_changed", 0) or 0

        regime_diff_meaningful = abs(pf_r0 - pf_r1) > 0.20
        best_above_threshold   = best_pf   >= PROMOTE_PF
        best_exp_positive      = best_exp  >  0
        overall_negative       = pf_ov     <  1.0

        transitions_hurt  = (pf_changed < pf_stayed) or (wr_changed < wr_stayed - 0.05)
        mixing_hurts      = overall_negative and (best_pf > PROMOTE_PF)
        promote_if_filter = best_pf >= PROMOTE_PF and best_n >= MIN_TRADES and best_exp > 0

        mc_r0 = monte_carlo(m_r0["pnls"] if m_r0 and m_r0["n_trades"] > 0 else np.array([]))
        mc_r1 = monte_carlo(m_r1["pnls"] if m_r1 and m_r1["n_trades"] > 0 else np.array([]))
        mc_best = mc_r0 if pf_r0 >= pf_r1 else mc_r1

        print(f"  ══ [{label}] ══════════════════════════════════════════════════════")
        print()

        print(f"  Q1. Which regime has the highest Profit Factor?")
        print(f"      Regime 0 : PF={pf_r0:.3f}  (n={n_r0})")
        print(f"      Regime 1 : PF={pf_r1:.3f}  (n={n_r1})")
        print(f"      Mixed    : PF={pf_mx:.3f}")
        print(f"      → {best_regime}  (PF={best_pf:.3f})")
        print()

        print(f"  Q2. Which regime has the highest expectancy?")
        print(f"      Regime 0 : ExpR={exp_r0:+.3f}")
        print(f"      Regime 1 : ExpR={exp_r1:+.3f}")
        print(f"      → {best_regime}  (ExpR={best_exp:+.3f}  {'positive ✓' if best_exp > 0 else 'negative ✗'})")
        print()

        print(f"  Q3. Which regime has the lowest drawdown?")
        print(f"      Regime 0 : MDD={mdd_r0*100:.1f}%")
        print(f"      Regime 1 : MDD={mdd_r1*100:.1f}%")
        print(f"      Overall  : MDD={mdd_ov*100:.1f}%")
        best_dd_reg = "Regime 0" if abs(mdd_r0) <= abs(mdd_r1) else "Regime 1"
        print(f"      → {best_dd_reg}")
        print()

        print(f"  Q4. Which regime has the best win rate?")
        print(f"      Regime 0 : WR={wr_r0*100:.1f}%")
        print(f"      Regime 1 : WR={wr_r1*100:.1f}%")
        print(f"      Overall  : WR={wr_ov*100:.1f}%")
        print(f"      → {best_regime}  (WR={best_wr*100:.1f}%)")
        print()

        print(f"  Q5. Does LSR only work in one regime?")
        print(f"      {best_regime} PF={best_pf:.3f}  |  {worst_regime} PF={worst_pf:.3f}")
        print(f"      Difference: {abs(pf_r0-pf_r1):.3f}")
        print(f"      Statistical significance: {_yn(stat.get('significant_any', False))}")
        only_one = regime_diff_meaningful and (best_pf > 1.0) and (worst_pf < 1.0)
        print(f"      → {_yn(only_one)}")
        print()

        print(f"  Q6. Do regime transitions increase losses?")
        print(f"      Stayed in regime : WR={wr_stayed*100:.1f}%  PF={pf_stayed:.3f}  n={trans.get('n_stayed',0)}")
        print(f"      Regime changed   : WR={wr_changed*100:.1f}%  PF={pf_changed:.3f}  n={trans.get('n_changed',0)}")
        print(f"      → {_yn(transitions_hurt)}")
        print()

        print(f"  Q7. Is overall poor performance caused by mixing regimes?")
        print(f"      Overall PF={pf_ov:.3f}  |  Best-regime PF={best_pf:.3f}")
        print(f"      → {_yn(mixing_hurts)}  (overall negative, best regime positive: {_yn(best_pf > 1.0)})")
        print()

        print(f"  Q8. Would trading only the best regime produce PF > 1.20 with ≥ {MIN_TRADES} trades?")
        print(f"      Best regime: {best_regime}  PF={best_pf:.3f}  n={best_n}")
        print(f"      MC prob profit: {mc_best['prob_profit']*100:.1f}%")
        print(f"      → {_yn(promote_if_filter)}")
        print()

        # ── Verdict ──────────────────────────────────────────────────────────
        mc_pp = mc_best["prob_profit"]
        if promote_if_filter and mc_pp >= PROMOTE_MC_PP and abs(best_mdd) < PROMOTE_MDD:
            verdict = "PROMOTE — Best-regime filter produces a viable edge"
        elif best_pf >= 1.0 and best_exp > 0 and best_n >= 15:
            verdict = "WATCHLIST — Positive edge in best regime but insufficient trades or borderline metrics"
        else:
            verdict = "REJECT — No robust edge found even after regime conditioning"

        print(f"  ── VERDICT [{label}] " + "─" * 60)
        print(f"  {verdict}")
        print()
        print("  " + "─" * (W - 2))
        print()

    print("=" * W)


# =============================================================================
# SECTION 11 — MAIN PIPELINE
# =============================================================================

def run_timeframe(tf_cfg: dict) -> dict:
    bar        = tf_cfg["bar"]
    bar_min    = tf_cfg["minutes"]
    label      = tf_cfg["label"]

    print(f"\n{'='*72}")
    print(f"  TIMEFRAME: {label}  [{bar}]")
    print(f"{'='*72}")

    # ── 1. Load from cache ───────────────────────────────────────────────────
    print(f"\n  Loading {bar} data from cache...", flush=True)
    raw_data = load_all(bar)
    if not raw_data:
        raise RuntimeError(f"No cached data found for {bar}")
    for sym, df in raw_data.items():
        print(f"  {sym:20s}  {len(df):>8,} candles  "
              f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})")

    # ── 2. Assign R017 regimes ───────────────────────────────────────────────
    regime_maps = assign_regimes(raw_data, bar)

    # ── 3. LSR indicators + OOS split + backtest ─────────────────────────────
    print(f"\n  Running LSR backtest with regime tagging...", flush=True)

    all_trades      = []
    sym_splits      = {}
    sym_regime_metrics = {}
    oos_dates       = {}

    for sym, df_raw in raw_data.items():
        short = sym.split("-")[0]

        # Compute LSR indicators on full data, then OOS split
        df_ind   = add_lsr_indicators(df_raw)
        df_ind   = df_ind.dropna(subset=["ema200", "lsr_prior_low"]).reset_index(drop=True)

        if len(df_ind) < 300:
            print(f"  {short:6s}  SKIP — insufficient data")
            continue

        split_idx = int(len(df_ind) * TRAIN_RATIO)
        df_oos    = df_ind.iloc[split_idx:].reset_index(drop=True)
        oos_dates[sym] = (str(df_oos["datetime"].iloc[0].date()),
                          str(df_oos["datetime"].iloc[-1].date()))

        reg_map_sym = regime_maps.get(sym, pd.Series(dtype=int))
        regime_arr  = attach_regime_labels(df_oos, reg_map_sym)

        # Diagnostic: how many OOS bars got a valid regime label?
        valid_pct = (regime_arr != -1).mean() * 100

        trades = run_lsr_backtest_regime_tagged(
            df_oos, regime_arr, sym, bar_min)

        n_tot  = len(trades)
        n_sigs = signal_lsr(df_oos).sum()

        splits  = split_by_regime(trades)
        sym_splits[sym] = splits

        all_trades.extend(trades)

        # Per-symbol per-regime metrics
        sym_regime_metrics[sym] = {
            slc: regime_metrics(splits[slc], f"{short}/{slc}", bar_min)
            for slc in ["overall", "r0", "r1", "mixed"]
        }

        m_ov = sym_regime_metrics[sym]["overall"]
        m_r0 = sym_regime_metrics[sym]["r0"]
        m_r1 = sym_regime_metrics[sym]["r1"]

        reg_coverage = (regime_arr != -1).mean() * 100
        print(f"  {short:6s}  sigs={n_sigs:4d}  trades={n_tot:3d}  "
              f"reg_cov={reg_coverage:.0f}%  "
              f"OV: PF={m_ov['profit_factor']:.3f}  "
              f"R0(n={m_r0['n_trades']}): PF={m_r0['profit_factor']:.3f}  "
              f"R1(n={m_r1['n_trades']}): PF={m_r1['profit_factor']:.3f}")

    # ── 4. Portfolio-level splits ────────────────────────────────────────────
    portfolio_splits = split_by_regime(all_trades)
    portfolio_metrics = {
        slc: regime_metrics(portfolio_splits[slc], slc.upper(), bar_min)
        for slc in ["overall", "r0", "r1", "r0_pure", "r1_pure", "mixed"]
    }

    m_ov = portfolio_metrics["overall"]
    m_r0 = portfolio_metrics["r0"]
    m_r1 = portfolio_metrics["r1"]
    m_mx = portfolio_metrics["mixed"]

    print(f"\n  Portfolio summary [{bar}]:")
    print(f"    Overall : n={m_ov['n_trades']:4d}  PF={m_ov['profit_factor']:.3f}  "
          f"WR={m_ov['win_rate']*100:.1f}%  MDD={m_ov['max_drawdown']*100:.1f}%")
    print(f"    Regime 0: n={m_r0['n_trades']:4d}  PF={m_r0['profit_factor']:.3f}  "
          f"WR={m_r0['win_rate']*100:.1f}%  MDD={m_r0['max_drawdown']*100:.1f}%")
    print(f"    Regime 1: n={m_r1['n_trades']:4d}  PF={m_r1['profit_factor']:.3f}  "
          f"WR={m_r1['win_rate']*100:.1f}%  MDD={m_r1['max_drawdown']*100:.1f}%")
    print(f"    Mixed   : n={m_mx['n_trades']:4d}  PF={m_mx['profit_factor']:.3f}  "
          f"WR={m_mx['win_rate']*100:.1f}%  MDD={m_mx['max_drawdown']*100:.1f}%")

    # ── 5. Statistical tests ─────────────────────────────────────────────────
    stat_result = statistical_tests(
        portfolio_splits["r0"], portfolio_splits["r1"])

    # ── 6. Transition analysis ───────────────────────────────────────────────
    trans = transition_analysis(all_trades)

    # ── 7. Charts ────────────────────────────────────────────────────────────
    print(f"\n  Generating charts for [{bar}]...", flush=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    plot_equity_by_regime(portfolio_metrics, bar)
    plot_drawdown_by_regime(portfolio_metrics, bar)
    plot_pf_winrate(sym_regime_metrics, bar)
    plot_trade_transition_matrix(trans, bar)
    plot_trade_distribution(sym_splits, bar)
    plot_monte_carlo_by_regime(portfolio_metrics, bar)
    plot_bootstrap_ci(stat_result, portfolio_metrics, bar)

    # ── 8. Console report ────────────────────────────────────────────────────
    print_report(portfolio_metrics, stat_result, trans, sym_regime_metrics, tf_cfg)

    return {
        "metrics":            portfolio_metrics,
        "stat":               stat_result,
        "trans":              trans,
        "sym_splits":         sym_splits,
        "sym_regime_metrics": sym_regime_metrics,
        "oos_dates":          oos_dates,
        "label":              label,
    }


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print()
    print("╔" + "═" * 79 + "╗")
    print("║  QUANTLAB AI — RESEARCH #018" + " " * 50 + "║")
    print("║  LSR Regime Conditioning (Research #017 Regimes)" + " " * 30 + "║")
    print("╚" + "═" * 79 + "╝")
    print()
    print("  Objective  : Does LSR have an edge only in specific market regimes?")
    print("  Strategy   : Exact LSR from Research #010 (quantlab_ai.strategy_lsr)")
    print("  Regimes    : R017 KMeans k=2  (Regime 0 = low vol, Regime 1 = high vol)")
    print("  Data       : Existing cache — no new downloads")
    print("  Symbols    : BTC ETH LINK XRP DOGE LTC AVAX BCH")
    print("  Split      : 70/30 chronological train/OOS")
    print()

    all_tf = {}
    journal_rows = []

    for tf_cfg in TIMEFRAMES:
        bar = tf_cfg["bar"]
        try:
            result = run_timeframe(tf_cfg)
            all_tf[bar] = result
            result["label"] = tf_cfg["label"]

            # Journal entries for key slices
            for slc, lbl in [("overall", "LSR_Overall"),
                              ("r0",      "LSR_Regime0"),
                              ("r1",      "LSR_Regime1")]:
                m  = result["metrics"].get(slc)
                if m and m["n_trades"] > 0:
                    mc = monte_carlo(m["pnls"], MC_ITER)
                    v  = _verdict_from_metrics(m, mc)
                    journal_rows.append({
                        "research_id":    RESEARCH_ID,
                        "run_date":       datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                        "strategy_name":  f"{lbl}_{bar}",
                        "symbol":         "PORTFOLIO",
                        "n_trades":       m["n_trades"],
                        "profit_factor":  round(m["profit_factor"], 4),
                        "expectancy_r":   round(m["expectancy_r"],  4),
                        "win_rate":       round(m["win_rate"],       4),
                        "net_profit":     round(m["net_profit"],     2),
                        "max_drawdown":   round(m["max_drawdown"],   4),
                        "sharpe":         round(m["sharpe"],         4),
                        "mc_prob_profit": round(mc["prob_profit"],   4),
                        "avg_hold_minutes": round(m["avg_hold_minutes"], 1),
                        "verdict":        v,
                    })

        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
        except Exception as e:
            import traceback; traceback.print_exc()

    # ── Eight final questions ─────────────────────────────────────────────────
    if all_tf:
        print_eight_questions(all_tf)

    # ── Journal ───────────────────────────────────────────────────────────────
    if journal_rows:
        print(f"\n  Writing journal ({len(journal_rows)} rows)...")
        from quantlab_ai import JOURNAL_COLS
        import csv
        path     = CONFIG["JOURNAL_FILE"]
        new_file = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_COLS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerows(journal_rows)
        print(f"  Journal updated → {path}")

    print()
    print(f"  All outputs → {OUTPUT_FOLDER}/r018_*")
    print(f"  Research #018 complete.")
    print()


if __name__ == "__main__":
    main()
