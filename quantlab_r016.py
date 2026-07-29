"""
=============================================================================
QUANTLAB AI — RESEARCH #016
Hypothesis : Volatility Compression → Expansion
Timeframes : 15-minute  |  1-hour
Symbols    : BTC, ETH, LINK, AVAX, XRP, DOGE, LTC, BCH  (OKX perps)

Entry rules:
  LONG  : ATR(14) < rolling-20-bar median(ATR)
           AND BollingerBandWidth(20,2) < rolling-20-bar median(BBW)
           AND close > highest high of previous 20 bars
  SHORT : ATR(14) < rolling-20-bar median(ATR)
           AND BollingerBandWidth(20,2) < rolling-20-bar median(BBW)
           AND close < lowest low of previous 20 bars

Stop  : 1 × ATR(14) at entry bar
Target: 2 × ATR(14) at entry bar   (RR = 2)

This is a completely new hypothesis. No relation to any prior research.
=============================================================================
"""

import os, sys, csv, math, time, random, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantlab_ai import (
    CONFIG,
    compute_metrics, monte_carlo,
    append_journal, _journal_row, _verdict_from_metrics,
    calc_ema, calc_atr, calc_adx,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID   = "R016"
OUTPUT_FOLDER = CONFIG["OUTPUT_FOLDER"]
CACHE_FOLDER  = CONFIG["CACHE_FOLDER"]
TRAIN_RATIO   = CONFIG["TRAIN_RATIO"]

SYMBOLS = [
    "BTC-USDT-SWAP",  "ETH-USDT-SWAP",  "LINK-USDT-SWAP", "AVAX-USDT-SWAP",
    "XRP-USDT-SWAP",  "DOGE-USDT-SWAP", "LTC-USDT-SWAP",  "BCH-USDT-SWAP",
]

TIMEFRAMES = [
    {"bar": "15m", "minutes": 15,  "months": 6, "min_cache": 5_000,  "label": "15-minute"},
    {"bar": "1H",  "minutes": 60,  "months": 6, "min_cache":   500,  "label": "1-hour"},
]

MIN_OOS_TRADES = 30   # lower threshold at 1H

# Execution costs — locked (same as all research)
TAKER_FEE    = CONFIG["TAKER_FEE"]
SPREAD       = CONFIG["SPREAD"] * 0.5
SL_SLIPPAGE  = CONFIG["SL_SLIPPAGE"]
MIN_SL_PCT   = CONFIG["MIN_SL_PCT"]
RR           = 2.0      # hard-coded: TP = 2×ATR, SL = 1×ATR
MAX_LEV      = CONFIG["MAX_LEVERAGE"]
STARTING_CAP = CONFIG["STARTING_CAPITAL"]
RISK_PCT     = CONFIG["RISK_PER_TRADE_PCT"]

PROMOTE_PF    = 1.20
PROMOTE_MDD   = 0.30
PROMOTE_MC_PP = 0.60

BG = "#0F1117"

OKX_HIST_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_LIVE_URL = "https://www.okx.com/api/v5/market/candles"
CANDLE_COLS  = ["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"]
PAGE_LIMIT   = 300
API_DELAY    = 0.05
DL_WORKERS   = 4
MAX_RETRIES  = 5


# =============================================================================
# SECTION 1 — DATA DOWNLOAD (reuses R015 pattern)
# =============================================================================

def _parse_candles(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return (df[["datetime","open","high","low","close","vol"]]
            .sort_values("datetime").reset_index(drop=True))


def _fetch_page(symbol: str, bar: str, after_ms=None, use_history=True) -> list:
    url    = OKX_HIST_URL if use_history else OKX_LIVE_URL
    params = {"instId": symbol, "bar": bar, "limit": str(PAGE_LIMIT)}
    if after_ms is not None:
        params["after"] = str(after_ms)
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(2 ** attempt); continue
            d    = r.json()
            code = d.get("code", "-1")
            if code == "0":
                return d.get("data", [])
            if code in ("50011", "50013"):
                return []
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return []


def _cache_path(symbol: str, bar: str) -> str:
    safe = symbol.replace("-", "_") + f"_{bar}"
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    return os.path.join(CACHE_FOLDER, f"{safe}.parquet")


def _load_cache(symbol: str, bar: str):
    path = _cache_path(symbol, bar)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def _save_cache(df: pd.DataFrame, symbol: str, bar: str):
    df.to_parquet(_cache_path(symbol, bar), index=False)


def _download_symbol(symbol: str, bar: str, months: int) -> pd.DataFrame:
    now_ms    = int(time.time() * 1000)
    cutoff_ms = now_ms - int(months * 30.44 * 24 * 3600 * 1000)
    all_rows, after_ms_cursor, pages = [], None, 0
    while True:
        raw = _fetch_page(symbol, bar, after_ms=after_ms_cursor, use_history=True)
        if not raw:
            if pages == 0:
                raw = _fetch_page(symbol, bar, after_ms=None, use_history=False)
                if not raw: break
            else: break
        all_rows.extend(raw)
        pages += 1
        oldest_ts = int(raw[-1][0])
        if pages % 20 == 0:
            oldest_dt = datetime.fromtimestamp(oldest_ts/1000, tz=timezone.utc).date()
            print(f"    {symbol.split('-')[0]:6s}[{bar}] page {pages:3d} | oldest {oldest_dt}", flush=True)
        after_ms_cursor = oldest_ts
        if oldest_ts <= cutoff_ms: break
        time.sleep(API_DELAY)
    if not all_rows:
        raise RuntimeError(f"No data for {symbol} [{bar}]")
    df        = _parse_candles(all_rows)
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df        = df[df["datetime"] >= cutoff_dt]
    return df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def _refresh_symbol(symbol: str, bar: str, cached: pd.DataFrame) -> pd.DataFrame:
    last_ts       = cached["datetime"].iloc[-1]
    since_ms      = int(last_ts.timestamp() * 1000)
    all_rows, cursor = [], None
    for _ in range(20):
        raw = _fetch_page(symbol, bar, after_ms=cursor, use_history=False)
        if not raw:
            raw = _fetch_page(symbol, bar, after_ms=cursor, use_history=True)
        if not raw: break
        all_rows.extend(raw)
        oldest_ts = int(raw[-1][0])
        if oldest_ts <= since_ms: break
        cursor = oldest_ts
        time.sleep(API_DELAY)
    if not all_rows:
        return cached
    new_df = _parse_candles(all_rows)
    new_df = new_df[new_df["datetime"] > last_ts]
    if len(new_df) == 0:
        return cached
    combined = (pd.concat([cached, new_df], ignore_index=True)
                .drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True))
    _save_cache(combined, symbol, bar)
    return combined


def get_data(symbol: str, bar: str, months: int, min_cache: int) -> pd.DataFrame:
    cached = _load_cache(symbol, bar)
    if cached is not None and len(cached) >= min_cache:
        last_ts = cached["datetime"].iloc[-1]
        gap_min = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 60
        bar_min = int(bar.rstrip("mH")) * (60 if bar.endswith("H") else 1)
        if gap_min < bar_min * 2:
            return cached
        print(f"  {symbol.split('-')[0]:6s}[{bar}] refreshing ({len(cached):,} cached)...", flush=True)
        return _refresh_symbol(symbol, bar, cached)
    print(f"  {symbol.split('-')[0]:6s}[{bar}] full download...", flush=True)
    df = _download_symbol(symbol, bar, months)
    _save_cache(df, symbol, bar)
    span = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds() / 86400
    print(f"  {symbol.split('-')[0]:6s}[{bar}] {len(df):,} candles "
          f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}, {span:.0f}d)", flush=True)
    return df


def download_all_parallel(bar: str, months: int, min_cache: int) -> dict:
    results = {}
    def _worker(sym):
        try:    return sym, get_data(sym, bar, months, min_cache), None
        except Exception as e: return sym, None, str(e)
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as pool:
        futures = {pool.submit(_worker, s): s for s in SYMBOLS}
        for fut in as_completed(futures):
            sym, df, err = fut.result()
            if err: print(f"  [WARN] {sym}[{bar}] failed: {err}", flush=True)
            else:   results[sym] = df
    return results


# =============================================================================
# SECTION 2 — INDICATORS
# =============================================================================

def add_r016_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ATR(14)
    df["atr"] = calc_atr(df, 14)

    # ADX(14)
    df["adx"] = calc_adx(df, 14)

    # Bollinger Band Width (20, 2)
    sma20          = df["close"].rolling(20).mean()
    std20          = df["close"].rolling(20).std(ddof=1)
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_mid"]   = sma20
    # Normalised width: (upper - lower) / mid
    df["bbw"]      = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # Rolling 20-bar median of ATR and BBW  (includes current bar — standard)
    df["atr_med20"] = df["atr"].rolling(20).median()
    df["bbw_med20"] = df["bbw"].rolling(20).median()

    # ATR and BBW percentile rank within rolling 50-bar window (for attribution)
    def rolling_pct_rank(series, window=50):
        """Returns 0-1 percentile of current value within rolling window."""
        out = np.full(len(series), np.nan)
        arr = series.values
        for i in range(window - 1, len(arr)):
            win = arr[i - window + 1: i + 1]
            if np.isnan(arr[i]):
                continue
            out[i] = np.sum(win <= arr[i]) / window
        return pd.Series(out, index=series.index)

    df["atr_pct50"]  = rolling_pct_rank(df["atr"], 50)
    df["bbw_pct50"]  = rolling_pct_rank(df["bbw"], 50)

    # 20-bar rolling highest high and lowest low of PREVIOUS bars (no lookahead)
    df["hh20"] = df["high"].shift(1).rolling(20).max()
    df["ll20"]  = df["low"].shift(1).rolling(20).min()

    return df


# =============================================================================
# SECTION 3 — SIGNAL DETECTION
# =============================================================================

def compute_r016_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised signal detection.

    compression : ATR < atr_med20  AND  BBW < bbw_med20
    long_signal : compression AND close > hh20
    short_signal: compression AND close < ll20

    Signals are mutually exclusive per bar (long takes priority on ties,
    which are extremely rare since breakout above and below on same bar is
    essentially impossible).

    Returns df with added columns:
      signal_long  : bool
      signal_short : bool
      breakout_dist_pct : float  (|close - hh20 or ll20| / reference)
    """
    df = df.copy()

    compressed = (df["atr"] < df["atr_med20"]) & (df["bbw"] < df["bbw_med20"])

    long_bo  = df["close"] > df["hh20"]
    short_bo = df["close"] < df["ll20"]

    df["signal_long"]  = compressed & long_bo
    df["signal_short"] = compressed & short_bo

    # If somehow both fire (shouldn't happen), prefer long
    df.loc[df["signal_long"], "signal_short"] = False

    # Breakout distance (%)
    df["breakout_dist_pct"] = np.where(
        df["signal_long"],
        (df["close"] - df["hh20"]) / df["hh20"],
        np.where(
            df["signal_short"],
            (df["ll20"] - df["close"]) / df["ll20"],
            np.nan,
        ),
    )

    return df


# =============================================================================
# SECTION 4 — BACKTEST ENGINE (ATR-based SL/TP, both directions)
# =============================================================================

def run_r016_backtest(df: pd.DataFrame, label: str, minutes_per_bar: int) -> dict:
    """
    Event-driven simulator.
    SL  = entry_price ± 1×ATR(14) at signal bar
    TP  = entry_price ± 2×ATR(14) at signal bar
    Handles both long and short.
    Tracks MFE, MAE, direction, breakout_dist, expansion.
    """
    n           = len(df)
    in_pos      = False
    direction   = 0      # +1 long, -1 short
    entry_price = 0.0
    sl          = 0.0
    tp          = 0.0
    entry_time  = None
    entry_idx   = -1
    pos_size    = 0.0
    capital     = STARTING_CAP
    mfe_track   = 0.0
    mae_track   = 0.0

    trades = []

    close_arr  = df["close"].values
    high_arr   = df["high"].values
    low_arr    = df["low"].values
    open_arr   = df["open"].values
    dt_arr     = df["datetime"].values
    sig_l_arr  = df["signal_long"].values
    sig_s_arr  = df["signal_short"].values
    atr_arr    = df["atr"].values
    adx_arr    = df["adx"].values
    atr_p_arr  = df["atr_pct50"].values
    bbw_p_arr  = df["bbw_pct50"].values
    bd_arr     = df["breakout_dist_pct"].values
    bbw_arr    = df["bbw"].values

    for i in range(1, n):
        hi = high_arr[i]
        lo = low_arr[i]

        if in_pos:
            # Track MFE / MAE in R units
            if direction == 1:
                pnl_hi = (hi - entry_price) / (entry_price - sl)
                pnl_lo = (lo - entry_price) / (entry_price - sl)
            else:
                pnl_hi = (entry_price - lo)  / (sl - entry_price)
                pnl_lo = (entry_price - hi)  / (sl - entry_price)

            mfe_track = max(mfe_track, pnl_hi)
            mae_track = min(mae_track, pnl_lo)

            if direction == 1:
                sl_hit = lo <= sl
                tp_hit = hi >= tp
            else:
                sl_hit = hi >= sl
                tp_hit = lo <= tp

            if sl_hit or tp_hit:
                if sl_hit:
                    exit_price = sl * (1.0 - SL_SLIPPAGE) if direction == 1 else sl * (1.0 + SL_SLIPPAGE)
                    exit_type  = "SL"
                else:
                    exit_price = tp
                    exit_type  = "TP"

                sl_dist   = abs(entry_price - sl)
                if direction == 1:
                    gross_pnl = (exit_price - entry_price) * pos_size
                    r_mult    = (exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0
                else:
                    gross_pnl = (entry_price - exit_price) * pos_size
                    r_mult    = (entry_price - exit_price) / sl_dist if sl_dist > 0 else 0.0

                ne        = entry_price * pos_size
                nx        = exit_price  * pos_size
                cost_fee  = (ne + nx) * TAKER_FEE
                cost_spd  = (ne + nx) * SPREAD
                cost_slip = abs(sl - exit_price) * pos_size if exit_type == "SL" else 0.0
                net_pnl   = gross_pnl - cost_fee - cost_spd - cost_slip
                hold_bars = i - entry_idx
                hold_min  = hold_bars * minutes_per_bar

                # Expansion: how far price moved in trade direction from entry (in ATR)
                if direction == 1:
                    expansion_atr = (hi - entry_price) / sl_dist  # best R seen
                else:
                    expansion_atr = (entry_price - lo) / sl_dist

                trades.append({
                    "label":             label,
                    "direction":         "long" if direction == 1 else "short",
                    "entry_time":        entry_time,
                    "exit_time":         pd.Timestamp(dt_arr[i]),
                    "entry_price":       entry_price,
                    "exit_price":        exit_price,
                    "stop_loss":         sl,
                    "take_profit":       tp,
                    "pnl":               net_pnl,
                    "r_multiple":        r_mult,
                    "fees":              cost_fee,
                    "spread_cost":       cost_spd,
                    "sl_slippage":       cost_slip,
                    "holding_minutes":   hold_min,
                    "funding_windows_crossed": int(hold_min / 480),
                    "win":               exit_type == "TP",
                    "exit_type":         exit_type,
                    "mfe_r":             mfe_track,
                    "mae_r":             mae_track,
                    "expansion_atr":     expansion_atr,
                    # attribution from signal bar (i-1)
                    "atr_at_entry":      atr_arr[i - 1],
                    "adx_at_entry":      adx_arr[i - 1],
                    "atr_pct":           atr_p_arr[i - 1],
                    "bbw_pct":           bbw_p_arr[i - 1],
                    "breakout_dist_pct": bd_arr[i - 1],
                    "bbw_at_entry":      bbw_arr[i - 1],
                })
                capital += net_pnl
                in_pos    = False
            continue

        # ── Check for new signal on previous bar ──────────────────────────────
        go_long  = sig_l_arr[i - 1]
        go_short = sig_s_arr[i - 1]

        if not go_long and not go_short:
            continue

        ep_raw  = open_arr[i]
        atr_val = atr_arr[i - 1]

        if np.isnan(atr_val) or atr_val <= 0:
            continue

        sl_dist = atr_val  # 1 ATR stop

        if sl_dist / ep_raw < MIN_SL_PCT:
            continue

        if go_long:
            sl_price = ep_raw - sl_dist
            tp_price = ep_raw + RR * sl_dist
            dir_val  = 1
        else:
            sl_price = ep_raw + sl_dist
            tp_price = ep_raw - RR * sl_dist
            dir_val  = -1

        risk_dollars = capital * RISK_PCT
        size         = min(risk_dollars / sl_dist, (capital * MAX_LEV) / ep_raw)

        entry_price = ep_raw
        sl          = sl_price
        tp          = tp_price
        pos_size    = size
        entry_time  = pd.Timestamp(dt_arr[i])
        entry_idx   = i
        direction   = dir_val
        in_pos      = True
        mfe_track   = 0.0
        mae_track   = 0.0

    return {"trades": trades}


# =============================================================================
# SECTION 5 — PORTFOLIO METRICS
# =============================================================================

def compute_portfolio_metrics(all_trades: list, minutes_per_bar: int) -> dict:
    if not all_trades:
        return None
    df    = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
    pnls  = df["pnl"].values
    wins  = df["win"].values.astype(bool)
    rmul  = df["r_multiple"].values
    n     = len(pnls)
    n_win = int(wins.sum())
    n_los = n - n_win

    gw = pnls[wins].sum() if n_win else 0.0
    gl = abs(pnls[~wins].sum()) if n_los else 1e-9
    pf = gw / gl if gl > 0 else float("inf")
    wr = n_win / n

    equity = STARTING_CAP + np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    max_dd = dd.min()

    std    = np.std(pnls, ddof=1) if n > 1 else 0.0
    bpy    = 365 * 24 * 60 / minutes_per_bar
    sharpe = pnls.mean() / std * math.sqrt(bpy) if std > 0 else 0.0
    exp_r  = wr * RR - (1.0 - wr)

    return {
        "label":            "PORTFOLIO",
        "n_trades":         n,
        "net_profit":       float(pnls.sum()),
        "profit_factor":    pf,
        "win_rate":         wr,
        "expectancy_r":     exp_r,
        "avg_r":            float(rmul.mean()),
        "max_drawdown":     max_dd,
        "sharpe":           sharpe,
        "avg_hold_minutes": df["holding_minutes"].mean(),
        "equity":           equity,
        "drawdown":         dd,
        "pnls":             pnls,
        "r_multiples":      rmul,
        "trades_df":        df,
    }


# =============================================================================
# SECTION 6 — ATTRIBUTION
# =============================================================================

def attribution_analysis(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}

    wins  = trades_df[trades_df["win"]]
    loses = trades_df[~trades_df["win"]]
    longs = trades_df[trades_df["direction"] == "long"]
    shrts = trades_df[trades_df["direction"] == "short"]

    def _m(col):
        return trades_df[col].dropna().mean() if col in trades_df.columns else np.nan
    def _wm(col):
        return wins[col].dropna().mean() if len(wins) and col in wins.columns else np.nan
    def _lm(col):
        return loses[col].dropna().mean() if len(loses) and col in loses.columns else np.nan

    # Direction split
    n_long  = len(longs)
    n_short = len(shrts)
    wr_long  = longs["win"].mean() if n_long  else np.nan
    wr_short = shrts["win"].mean() if n_short else np.nan
    pf_long  = (longs[longs["win"]]["pnl"].sum() /
                abs(longs[~longs["win"]]["pnl"].sum() or 1e-9)) if n_long else np.nan
    pf_short = (shrts[shrts["win"]]["pnl"].sum() /
                abs(shrts[~shrts["win"]]["pnl"].sum() or 1e-9)) if n_short else np.nan

    # False breakout: trades where MAE touched -1R (stopped out immediately)
    false_bo = (trades_df["mae_r"] <= -0.95).sum() if "mae_r" in trades_df.columns else 0

    return {
        "avg_atr_pct":          _m("atr_pct"),
        "avg_bbw_pct":          _m("bbw_pct"),
        "avg_breakout_dist_pct":_m("breakout_dist_pct"),
        "avg_hold_min":         _m("holding_minutes"),
        "avg_mfe_r":            _m("mfe_r"),
        "avg_mae_r":            _m("mae_r"),
        "avg_expansion_atr":    _m("expansion_atr"),
        "avg_adx_entry":        _m("adx_at_entry"),
        "win_avg_mfe_r":        _wm("mfe_r"),
        "loss_avg_mfe_r":       _lm("mfe_r"),
        "win_avg_mae_r":        _wm("mae_r"),
        "loss_avg_mae_r":       _lm("mae_r"),
        "win_avg_hold_min":     _wm("holding_minutes"),
        "loss_avg_hold_min":    _lm("holding_minutes"),
        "n_long":               n_long,
        "n_short":              n_short,
        "wr_long":              wr_long,
        "wr_short":             wr_short,
        "pf_long":              pf_long,
        "pf_short":             pf_short,
        "false_breakout_n":     int(false_bo),
        "false_breakout_pct":   float(false_bo / max(len(trades_df), 1)),
    }


# =============================================================================
# SECTION 7 — VISUALISATIONS
# =============================================================================

PALETTE = [
    "#4A90D9","#FFB347","#00C49A","#FF4560",
    "#E040FB","#FFD700","#00D4FF","#FF6B6B",
]

def _ax(fig, gs_cell):
    ax = fig.add_subplot(gs_cell)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.tick_params(colors="#AAA", labelsize=8)
    ax.xaxis.label.set_color("#AAA")
    ax.yaxis.label.set_color("#AAA")
    ax.title.set_color("#EEE")
    return ax


def plot_equity_curves(sym_metrics: dict, port_m: dict, tf_bar: str, tf_label: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    syms = list(sym_metrics.keys())
    for k, sym in enumerate(syms):
        m  = sym_metrics[sym]["metrics"]
        eq = m["equity"]
        row, col = divmod(k, 4)
        ax = _ax(fig, gs[row, col])
        short = sym.split("-")[0]
        ax.plot(eq, color=PALETTE[k % len(PALETTE)], lw=1)
        ax.axhline(STARTING_CAP, color="#555", lw=0.5, ls="--")
        ax.set_title(f"{short}  PF={m['profit_factor']:.3f}  n={m['n_trades']}", fontsize=9)
        ax.set_ylabel("Equity $", fontsize=7)

    ax_p = _ax(fig, gs[1, 3])
    if port_m:
        ax_p.plot(port_m["equity"], color="#FFD700", lw=1.5)
        ax_p.axhline(STARTING_CAP, color="#555", lw=0.5, ls="--")
        ax_p.set_title(f"PORTFOLIO  PF={port_m['profit_factor']:.3f}  n={port_m['n_trades']}", fontsize=9)
        ax_p.set_ylabel("Equity $", fontsize=7)

    fig.suptitle(f"R016 Equity — Volatility Compression → Expansion [{tf_label}]",
                 color="#EEE", fontsize=12, y=1.01)
    path = f"{OUTPUT_FOLDER}/r016_equity_{tf_bar}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_drawdown_curves(sym_metrics: dict, port_m: dict, tf_bar: str, tf_label: str):
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)
    syms = list(sym_metrics.keys())
    for k, sym in enumerate(syms):
        m  = sym_metrics[sym]["metrics"]
        dd = m["drawdown"]
        row, col = divmod(k, 4)
        ax = _ax(fig, gs[row, col])
        short = sym.split("-")[0]
        ax.fill_between(range(len(dd)), dd * 100, 0,
                        color=PALETTE[k % len(PALETTE)], alpha=0.6)
        ax.set_title(f"{short}  MDD={m['max_drawdown']*100:.1f}%", fontsize=9)
        ax.set_ylabel("DD %", fontsize=7)

    ax_p = _ax(fig, gs[1, 3])
    if port_m:
        dd = port_m["drawdown"]
        ax_p.fill_between(range(len(dd)), dd * 100, 0, color="#FFD700", alpha=0.6)
        ax_p.set_title(f"PORTFOLIO  MDD={port_m['max_drawdown']*100:.1f}%", fontsize=9)
        ax_p.set_ylabel("DD %", fontsize=7)

    fig.suptitle(f"R016 Drawdown [{tf_label}]", color="#EEE", fontsize=12, y=1.01)
    path = f"{OUTPUT_FOLDER}/r016_drawdown_{tf_bar}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_atr_bbw_distributions(trades_df: pd.DataFrame, df_oos_all: dict,
                                tf_bar: str, tf_label: str):
    """
    Histogram of ATR percentile and BBW percentile at entry,
    split by win/loss.  Also shows overall OOS distribution.
    """
    if trades_df is None or trades_df.empty:
        return
    wins  = trades_df[trades_df["win"]]
    loses = trades_df[~trades_df["win"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)

    def _setup(ax, title):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.tick_params(colors="#AAA", labelsize=8)
        ax.set_title(title, color="#EEE", fontsize=10)

    # ATR percentile at entry
    _setup(axes[0,0], "ATR Percentile at Entry (Win vs Loss)")
    axes[0,0].hist(loses["atr_pct"].dropna(), bins=20, color="#FF4560", alpha=0.7, label="Loss")
    axes[0,0].hist(wins["atr_pct"].dropna(),  bins=20, color="#00C49A", alpha=0.7, label="Win")
    axes[0,0].axvline(0.5, color="#FFD700", lw=0.8, ls="--", label="Median")
    axes[0,0].set_xlabel("ATR pct rank (0=lowest)", color="#AAA")
    axes[0,0].legend(fontsize=8)

    # BBW percentile at entry
    _setup(axes[0,1], "BBW Percentile at Entry (Win vs Loss)")
    axes[0,1].hist(loses["bbw_pct"].dropna(), bins=20, color="#FF4560", alpha=0.7, label="Loss")
    axes[0,1].hist(wins["bbw_pct"].dropna(),  bins=20, color="#00C49A", alpha=0.7, label="Win")
    axes[0,1].axvline(0.5, color="#FFD700", lw=0.8, ls="--", label="Median")
    axes[0,1].set_xlabel("BBW pct rank (0=lowest)", color="#AAA")
    axes[0,1].legend(fontsize=8)

    # Breakout distance distribution
    _setup(axes[1,0], "Breakout Distance (% above/below channel)")
    axes[1,0].hist(loses["breakout_dist_pct"].dropna() * 100,
                   bins=40, color="#FF4560", alpha=0.7, label="Loss")
    axes[1,0].hist(wins["breakout_dist_pct"].dropna() * 100,
                   bins=40, color="#00C49A", alpha=0.7, label="Win")
    axes[1,0].set_xlabel("Breakout distance (%)", color="#AAA")
    axes[1,0].legend(fontsize=8)

    # Long vs Short trade distribution
    _setup(axes[1,1], "Trades by Direction")
    dirs   = trades_df["direction"].value_counts()
    colors = ["#4A90D9", "#FF6B6B"]
    axes[1,1].bar(dirs.index, dirs.values, color=colors[:len(dirs)], alpha=0.85)
    wr_long  = trades_df[trades_df["direction"]=="long"]["win"].mean()
    wr_short = trades_df[trades_df["direction"]=="short"]["win"].mean()
    axes[1,1].set_ylabel("Count", color="#AAA")
    axes[1,1].tick_params(colors="#AAA")
    # Annotate WR
    for i, (direction, cnt) in enumerate(dirs.items()):
        wr = wr_long if direction == "long" else wr_short
        if not np.isnan(wr):
            axes[1,1].text(i, cnt + 1, f"WR={wr*100:.1f}%", ha="center",
                           color="#EEE", fontsize=9)

    fig.suptitle(f"R016 ATR / BBW Distribution [{tf_label}]",
                 color="#EEE", fontsize=12)
    path = f"{OUTPUT_FOLDER}/r016_distributions_{tf_bar}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_mfe_mae(trades_df: pd.DataFrame, tf_bar: str, tf_label: str):
    if trades_df is None or trades_df.empty:
        return
    wins  = trades_df[trades_df["win"]]
    loses = trades_df[~trades_df["win"]]
    longs = trades_df[trades_df["direction"] == "long"]
    shrts = trades_df[trades_df["direction"] == "short"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    def _setup(ax):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.tick_params(colors="#AAA", labelsize=8)

    # MFE vs MAE — win/loss split
    _setup(axes[0])
    axes[0].scatter(wins["mae_r"],  wins["mfe_r"],  c="#00C49A", alpha=0.5, s=15, label="Win")
    axes[0].scatter(loses["mae_r"], loses["mfe_r"], c="#FF4560", alpha=0.5, s=15, label="Loss")
    axes[0].axhline(RR,   color="#FFD700", lw=0.8, ls="--", label=f"{RR}R TP")
    axes[0].axvline(-1.0, color="#FF8C00", lw=0.8, ls="--", label="1R SL")
    axes[0].set_xlabel("MAE (R)")
    axes[0].set_ylabel("MFE (R)")
    axes[0].set_title("MFE vs MAE (Win/Loss)", color="#EEE", fontsize=10)
    axes[0].legend(fontsize=8)

    # MFE vs MAE — direction split
    _setup(axes[1])
    axes[1].scatter(longs["mae_r"], longs["mfe_r"], c="#4A90D9", alpha=0.5, s=15, label="Long")
    axes[1].scatter(shrts["mae_r"], shrts["mfe_r"], c="#FF6B6B", alpha=0.5, s=15, label="Short")
    axes[1].axhline(RR,   color="#FFD700", lw=0.8, ls="--")
    axes[1].axvline(-1.0, color="#FF8C00", lw=0.8, ls="--")
    axes[1].set_xlabel("MAE (R)")
    axes[1].set_ylabel("MFE (R)")
    axes[1].set_title("MFE vs MAE (Direction)", color="#EEE", fontsize=10)
    axes[1].legend(fontsize=8)

    # R-multiple distribution
    _setup(axes[2])
    rmul = trades_df["r_multiple"].dropna().values
    axes[2].hist(rmul[rmul > 0],  bins=40, color="#00C49A", alpha=0.7, label="Win")
    axes[2].hist(rmul[rmul <= 0], bins=40, color="#FF4560", alpha=0.7, label="Loss")
    axes[2].axvline(0, color="#FFF", lw=0.5)
    axes[2].set_xlabel("R Multiple")
    axes[2].set_title("R-Multiple Distribution", color="#EEE", fontsize=10)
    axes[2].legend(fontsize=8)

    fig.suptitle(f"R016 MFE / MAE Analysis [{tf_label}]", color="#EEE", fontsize=12)
    path = f"{OUTPUT_FOLDER}/r016_mfe_mae_{tf_bar}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_portfolio_dashboard(sym_metrics: dict, tf_bar: str, tf_label: str):
    valid = {s: v for s, v in sym_metrics.items() if v["metrics"]["n_trades"] > 0}
    if not valid:
        return
    syms    = list(valid.keys())
    short   = [s.split("-")[0] for s in syms]
    pfs     = [valid[s]["metrics"]["profit_factor"] for s in syms]
    wrs     = [valid[s]["metrics"]["win_rate"] * 100 for s in syms]
    mdds    = [abs(valid[s]["metrics"]["max_drawdown"]) * 100 for s in syms]
    ntrades = [valid[s]["metrics"]["n_trades"] for s in syms]
    net_p   = [valid[s]["metrics"]["net_profit"] for s in syms]
    exp_r   = [valid[s]["metrics"]["expectancy_r"] for s in syms]

    fig    = plt.figure(figsize=(16, 8), facecolor=BG)
    gs     = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    colors = PALETTE[:len(syms)]

    def _bar(ax, vals, title, ylabel, refline=None):
        ax.bar(short, vals, color=colors, alpha=0.85)
        if refline is not None:
            ax.axhline(refline, color="#FFF", lw=0.7, ls="--")
        ax.set_title(title, fontsize=10, color="#EEE")
        ax.set_ylabel(ylabel, fontsize=8, color="#AAA")
        ax.tick_params(colors="#AAA", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.set_facecolor(BG)

    _bar(_ax(fig, gs[0, 0]), pfs,     "Profit Factor",    "PF",  refline=1.2)
    _bar(_ax(fig, gs[0, 1]), wrs,     "Win Rate (%)",     "%",   refline=50)
    _bar(_ax(fig, gs[0, 2]), mdds,    "Max Drawdown (%)", "%",   refline=30)
    _bar(_ax(fig, gs[1, 0]), ntrades, "# Trades",         "n",   refline=MIN_OOS_TRADES)
    _bar(_ax(fig, gs[1, 1]), net_p,   "Net P&L ($)",      "$")
    _bar(_ax(fig, gs[1, 2]), exp_r,   "Expectancy R",     "R",   refline=0)

    fig.suptitle(f"R016 Portfolio Dashboard [{tf_label}]", color="#EEE", fontsize=12, y=1.01)
    path = f"{OUTPUT_FOLDER}/r016_dashboard_{tf_bar}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# SECTION 8 — REPORT
# =============================================================================

def print_report(sym_metrics: dict, port_m: dict, attr: dict,
                 oos_dates: dict, tf_cfg: dict):
    W   = 110
    lbl = tf_cfg["label"]
    print()
    print("=" * W)
    print(f"  QUANTLAB AI — RESEARCH #016  [{lbl}]")
    print(f"  Volatility Compression → Expansion — {lbl} candles")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * W)
    print()
    print("  Entry  : ATR < ATR_median(20) AND BBW < BBW_median(20)")
    print("  Long   : compression AND close > HH(previous 20 bars)")
    print("  Short  : compression AND close < LL(previous 20 bars)")
    print("  Stop   : 1 × ATR at signal bar")
    print("  Target : 2 × ATR at signal bar")
    print()

    for sym, dates in oos_dates.items():
        short = sym.split("-")[0]
        print(f"  {short:6s}  OOS: {dates[0]} → {dates[1]}")
    print()

    header_syms = [s.split("-")[0] for s in sym_metrics.keys()]
    col_w = max(14, max(len(h) for h in header_syms) + 2)

    def _row(lbl_str, vals):
        print(f"  {lbl_str:<28s}" + "".join(f"{v:>{col_w}}" for v in vals))

    print("  " + "-" * (28 + col_w * len(header_syms)))
    _row("Metric", header_syms)
    print("  " + "-" * (28 + col_w * len(header_syms)))

    _row("Trades",         [f"{v['metrics']['n_trades']:>{col_w}d}"           for v in sym_metrics.values()])
    _row("Win Rate",       [f"{v['metrics']['win_rate']*100:>{col_w-1}.1f}%"  for v in sym_metrics.values()])
    _row("Profit Factor",  [f"{v['metrics']['profit_factor']:>{col_w}.3f}"    for v in sym_metrics.values()])
    _row("Expectancy R",   [f"{v['metrics']['expectancy_r']:>+{col_w}.3f}"    for v in sym_metrics.values()])
    _row("Net P&L ($)",    [f"{v['metrics']['net_profit']:>{col_w}.0f}"       for v in sym_metrics.values()])
    _row("Max Drawdown",   [f"{v['metrics']['max_drawdown']*100:>{col_w-1}.1f}%" for v in sym_metrics.values()])
    _row("Sharpe",         [f"{v['metrics']['sharpe']:>{col_w}.2f}"           for v in sym_metrics.values()])
    _row("MC Prob Profit", [f"{v['mc']['prob_profit']*100:>{col_w-1}.1f}%"    for v in sym_metrics.values()])
    _row("Verdict",        [f"{v['verdict']:>{col_w}}"                        for v in sym_metrics.values()])
    print("  " + "-" * (28 + col_w * len(header_syms)))
    print()

    if port_m:
        print(f"  ── PORTFOLIO [{lbl}] ─────────────────────────────────────")
        print(f"  Trades          : {port_m['n_trades']}")
        print(f"  Win Rate        : {port_m['win_rate']*100:.1f}%")
        print(f"  Profit Factor   : {port_m['profit_factor']:.3f}")
        print(f"  Expectancy R    : {port_m['expectancy_r']:+.3f}")
        print(f"  Net P&L         : ${port_m['net_profit']:,.0f}")
        print(f"  Max Drawdown    : {port_m['max_drawdown']*100:.1f}%")
        print(f"  Sharpe          : {port_m['sharpe']:.2f}")
        mc_port = monte_carlo(port_m["pnls"])
        print(f"  MC Prob Profit  : {mc_port['prob_profit']*100:.1f}%")
        print(f"  MC Median Equity: ${mc_port['median']:,.0f}")
        print()

    if attr:
        n_l  = attr.get("n_long", 0)
        n_s  = attr.get("n_short", 0)
        wrl  = attr.get("wr_long",  np.nan)
        wrs_ = attr.get("wr_short", np.nan)
        pfl  = attr.get("pf_long",  np.nan)
        pfs_ = attr.get("pf_short", np.nan)
        fbo  = attr.get("false_breakout_n",  0)
        fbp  = attr.get("false_breakout_pct", 0)

        print(f"  ── ATTRIBUTION [{lbl}] ────────────────────────────────────")
        print(f"  Avg ATR percentile    : {attr.get('avg_atr_pct', 0)*100:.1f}th pct")
        print(f"  Avg BBW percentile    : {attr.get('avg_bbw_pct', 0)*100:.1f}th pct")
        print(f"  Avg breakout distance : {attr.get('avg_breakout_dist_pct', 0)*100:.3f}%")
        print(f"  Avg ADX at entry      : {attr.get('avg_adx_entry', 0):.1f}")
        print(f"  Avg holding time      : {attr.get('avg_hold_min', 0):.0f} min")
        print(f"  Avg expansion (R)     : {attr.get('avg_expansion_atr', 0):+.3f}R")
        print(f"  Avg MFE               : {attr.get('avg_mfe_r', 0):+.3f}R")
        print(f"  Avg MAE               : {attr.get('avg_mae_r', 0):+.3f}R")
        print(f"  Win avg hold          : {attr.get('win_avg_hold_min', 0):.0f} min")
        print(f"  Loss avg hold         : {attr.get('loss_avg_hold_min', 0):.0f} min")
        print()
        print(f"  ── DIRECTION SPLIT ──────────────────────────────────────────")
        print(f"  Long  trades : {n_l:4d}  WR={wrl*100:.1f}%  PF={pfl:.3f}" if not math.isnan(pfl) else f"  Long  trades : {n_l}")
        print(f"  Short trades : {n_s:4d}  WR={wrs_*100:.1f}%  PF={pfs_:.3f}" if not math.isnan(pfs_) else f"  Short trades : {n_s}")
        print(f"  False breakouts : {fbo} ({fbp*100:.1f}% of trades — stopped out ≥ −0.95R immediately)")
        print()

    if port_m:
        pf   = port_m["profit_factor"]
        exp  = port_m["expectancy_r"]
        mdd  = abs(port_m["max_drawdown"])
        n    = port_m["n_trades"]
        n_sp = sum(1 for v in sym_metrics.values()
                   if v["metrics"]["profit_factor"] >= PROMOTE_PF
                   and v["metrics"]["net_profit"] > 0)
        mc_port = monte_carlo(port_m["pnls"])
        mc_pp   = mc_port["prob_profit"]

        pass_pf   = pf   > PROMOTE_PF
        pass_exp  = exp  > 0
        pass_mdd  = mdd  < PROMOTE_MDD
        pass_n    = n    >= 100
        pass_syms = n_sp >= 2
        pass_mc   = mc_pp >= PROMOTE_MC_PP

        final_pass  = all([pass_pf, pass_exp, pass_mdd, pass_n, pass_syms, pass_mc])
        verdict_str = "✅  PASS — SUITABLE FOR FORWARD DEMO TESTING" if final_pass else "❌  REJECT"

        print(f"  ── FINAL VERDICT [{lbl}] ─────────────────────────────────")
        print(f"  Combined PF > {PROMOTE_PF:.2f}    : {'✓' if pass_pf  else '✗'}  ({pf:.3f})")
        print(f"  Positive expectancy     : {'✓' if pass_exp else '✗'}  ({exp:+.3f}R)")
        print(f"  MDD < {PROMOTE_MDD*100:.0f}%           : {'✓' if pass_mdd else '✗'}  ({mdd*100:.1f}%)")
        print(f"  ≥ 100 OOS trades        : {'✓' if pass_n  else '✗'}  ({n})")
        print(f"  ≥ 2 profitable symbols  : {'✓' if pass_syms else '✗'}  ({n_sp}/{len(sym_metrics)})")
        print(f"  MC Prob Profit ≥ {PROMOTE_MC_PP*100:.0f}%  : {'✓' if pass_mc else '✗'}  ({mc_pp*100:.1f}%)")
        print()
        print(f"  {verdict_str}")
        print()
    print("=" * W)


def print_six_questions(all_tf: dict):
    """
    Answer the six final questions using data from both timeframes.
    all_tf: {bar: {"port": port_m, "attr": attr_dict, "sym_metrics": ...}}
    """
    W = 110
    print()
    print("=" * W)
    print("  R016 — SIX DIAGNOSTIC QUESTIONS")
    print("  Volatility Compression → Expansion")
    print("=" * W)
    print()

    def _yn(cond): return "YES ✓" if cond else "NO  ✗"

    for bar, tf_label in [("15m", "15-minute"), ("1H", "1-hour")]:
        if bar not in all_tf:
            continue
        port_m     = all_tf[bar]["port"]
        attr       = all_tf[bar]["attr"]
        sym_m      = all_tf[bar]["sym_metrics"]
        if not port_m:
            continue

        pf   = port_m["profit_factor"]
        exp  = port_m["expectancy_r"]
        n_profitable = sum(1 for v in sym_m.values()
                           if v["metrics"]["profit_factor"] >= PROMOTE_PF
                           and v["metrics"]["net_profit"] > 0)
        wr_l   = attr.get("wr_long",  0) or 0
        wr_s   = attr.get("wr_short", 0) or 0
        pf_l   = attr.get("pf_long",  0) or 0
        pf_s   = attr.get("pf_short", 0) or 0
        fbo_pct= attr.get("false_breakout_pct", 0) or 0

        # Best symbol
        best_sym = max(sym_m, key=lambda s: sym_m[s]["metrics"]["profit_factor"])
        best_pf  = sym_m[best_sym]["metrics"]["profit_factor"]

        print(f"  [{tf_label}]")
        print()
        print(f"  Q1. Does volatility compression produce an edge?")
        print(f"      Portfolio PF={pf:.3f}  Expectancy={exp:+.3f}R")
        print(f"      → {_yn(pf > 1.0 and exp > 0)}")
        print()
        print(f"  Q2. Are breakouts after compression profitable?")
        print(f"      False breakout rate: {fbo_pct*100:.1f}%  |  Avg expansion: {attr.get('avg_expansion_atr',0):+.3f}R")
        print(f"      → {_yn(pf > 1.0)}")
        print()
        print(f"  Q3. Which symbols respond best?")
        sym_pf_sorted = sorted(sym_m.items(), key=lambda x: x[1]["metrics"]["profit_factor"], reverse=True)
        for sym, v in sym_pf_sorted[:4]:
            short = sym.split("-")[0]
            m     = v["metrics"]
            print(f"      {short:6s}  PF={m['profit_factor']:.3f}  WR={m['win_rate']*100:.1f}%  n={m['n_trades']}")
        print()
        print(f"  Q4. Long or short performs better?")
        print(f"      Long   : n={attr.get('n_long',0):4d}  WR={wr_l*100:.1f}%  PF={pf_l:.3f}")
        print(f"      Short  : n={attr.get('n_short',0):4d}  WR={wr_s*100:.1f}%  PF={pf_s:.3f}")
        print(f"      → Better: {'Long' if pf_l > pf_s else 'Short'}  (PF {max(pf_l,pf_s):.3f} vs {min(pf_l,pf_s):.3f})")
        print()
        print(f"  Q5. Does PF exceed 1.20?")
        print(f"      Portfolio PF={pf:.3f}  |  Best symbol={best_sym.split('-')[0]} PF={best_pf:.3f}")
        print(f"      → {_yn(pf > PROMOTE_PF)}")
        print()
        print(f"  Q6. Is the edge consistent across multiple symbols?")
        print(f"      Symbols with PF > 1.20: {n_profitable}/8")
        print(f"      → {_yn(n_profitable >= 4)}")
        print()
        print("  " + "─" * (W - 2))
        print()
    print("=" * W)


# =============================================================================
# SECTION 9 — MAIN
# =============================================================================

def run_timeframe(tf_cfg: dict) -> tuple:
    bar         = tf_cfg["bar"]
    minutes     = tf_cfg["minutes"]
    months      = tf_cfg["months"]
    min_cache   = tf_cfg["min_cache"]
    label       = tf_cfg["label"]

    print(f"\n{'='*70}")
    print(f"  TIMEFRAME: {label}  [{bar}]")
    print(f"{'='*70}")

    # ── Download ──────────────────────────────────────────────────────────────
    print(f"\n  Downloading {bar} data...", flush=True)
    t0       = time.time()
    raw_data = download_all_parallel(bar, months, min_cache)
    elapsed  = time.time() - t0
    for sym, df in raw_data.items():
        print(f"  {sym:20s}  {len(df):>8,} candles  "
              f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})")
    print(f"  Download complete in {elapsed:.0f}s")

    # ── Backtest ──────────────────────────────────────────────────────────────
    print(f"\n  Computing indicators + running backtests...", flush=True)

    sym_metrics = {}
    all_trades  = []
    oos_dates   = {}

    for sym, df_raw in raw_data.items():
        short = sym.split("-")[0]

        df = add_r016_indicators(df_raw)
        df = df.dropna(subset=["atr","bbw","hh20","ll20"]).reset_index(drop=True)

        if len(df) < 300:
            print(f"  {short:6s}  SKIP — insufficient data ({len(df)} bars)")
            continue

        split_idx = int(len(df) * TRAIN_RATIO)
        df_oos    = df.iloc[split_idx:].reset_index(drop=True)
        oos_dates[sym] = (
            str(df_oos["datetime"].iloc[0].date()),
            str(df_oos["datetime"].iloc[-1].date()),
        )

        if len(df_oos) < 50:
            print(f"  {short:6s}  SKIP — OOS too short ({len(df_oos)} bars)")
            continue

        df_oos    = compute_r016_signals(df_oos)
        sig_long  = df_oos["signal_long"].sum()
        sig_short = df_oos["signal_short"].sum()

        bt  = run_r016_backtest(df_oos, label=sym, minutes_per_bar=minutes)
        m   = compute_metrics(bt["trades"], sym)
        mc  = monte_carlo(m["pnls"])
        v   = _verdict_from_metrics(m, mc)

        sym_metrics[sym] = {"metrics": m, "mc": mc, "verdict": v}
        all_trades.extend(bt["trades"])

        print(f"  {short:6s}  sig_L={sig_long:4d} sig_S={sig_short:4d}  "
              f"trades={m['n_trades']:4d}  WR={m['win_rate']*100:5.1f}%  "
              f"PF={m['profit_factor']:.3f}  [{v}]")

    # ── Portfolio ─────────────────────────────────────────────────────────────
    port_m = compute_portfolio_metrics(all_trades, minutes) if all_trades else None

    attr = {}
    if all_trades:
        td_all = pd.DataFrame(all_trades)
        attr   = attribution_analysis(td_all)

    if port_m:
        print(f"\n  Portfolio  trades={port_m['n_trades']}  "
              f"WR={port_m['win_rate']*100:.1f}%  "
              f"PF={port_m['profit_factor']:.3f}  "
              f"MDD={port_m['max_drawdown']*100:.1f}%")

    return sym_metrics, port_m, attr, oos_dates, all_trades


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CACHE_FOLDER, exist_ok=True)

    print()
    print("╔" + "═" * 79 + "╗")
    print("║  QUANTLAB AI — RESEARCH #016" + " " * 50 + "║")
    print("║  Volatility Compression → Expansion" + " " * 43 + "║")
    print("╚" + "═" * 79 + "╝")
    print()
    print("  Hypothesis : Periods of unusually low volatility precede large moves.")
    print("  Entry      : Compression (ATR + BBW < median) + channel breakout")
    print("  Long/Short : Both directions tested independently")
    print("  SL / TP    : 1 ATR / 2 ATR  (identical cost structure to all research)")
    print()

    all_tf         = {}   # {bar: {port, attr, sym_metrics}}
    all_journal    = []

    for tf_cfg in TIMEFRAMES:
        bar   = tf_cfg["bar"]

        sym_metrics, port_m, attr, oos_dates, all_trades = run_timeframe(tf_cfg)

        if not sym_metrics:
            print(f"  [WARN] No results for {tf_cfg['label']}")
            continue

        # Charts
        print(f"\n  Generating charts for {tf_cfg['label']}...")
        td_all = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

        plot_equity_curves(sym_metrics, port_m, bar, tf_cfg["label"])
        plot_drawdown_curves(sym_metrics, port_m, bar, tf_cfg["label"])
        plot_atr_bbw_distributions(td_all, {}, bar, tf_cfg["label"])
        plot_mfe_mae(td_all, bar, tf_cfg["label"])
        plot_portfolio_dashboard(sym_metrics, bar, tf_cfg["label"])

        # Report
        print_report(sym_metrics, port_m, attr, oos_dates, tf_cfg)

        all_tf[bar] = {"port": port_m, "attr": attr, "sym_metrics": sym_metrics}

        # Journal
        for sym, data in sym_metrics.items():
            m  = data["metrics"]
            mc = data["mc"]
            v  = data["verdict"]
            all_journal.append(_journal_row(
                strategy_name = f"VolCompression_{bar}_{sym.split('-')[0]}",
                symbol        = sym,
                m             = m,
                mc            = mc,
                verdict       = v,
            ))

    # ── Six diagnostic questions ──────────────────────────────────────────────
    print_six_questions(all_tf)

    # ── Journal ───────────────────────────────────────────────────────────────
    if all_journal:
        print(f"\n{'='*70}")
        print(f"  Writing journal")
        print(f"{'='*70}")
        append_journal(all_journal)
        print(f"  Journal updated → {OUTPUT_FOLDER}/research_journal.csv")

    print()
    print(f"  All outputs → {OUTPUT_FOLDER}/r016_*")
    print(f"  Research #016 complete.")
    print()


if __name__ == "__main__":
    main()
