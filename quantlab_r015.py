"""
=============================================================================
QUANTLAB AI — RESEARCH #015
Hypothesis : Trend Continuation After Pullback — Higher Timeframes
Timeframes : 15-minute candles  |  1-hour candles
Symbols    : BTC, ETH, LINK, AVAX, XRP, DOGE, LTC, BCH  (OKX perps)

Identical strategy to R014.  Only the timeframe changes.

Entry rules (LONG only):
  1. close > EMA200
  2. EMA200 slope positive  (EMA200[i] > EMA200[i-10])
  3. EMA20 > EMA50
  4. Price pulls back: at least one bar closes below EMA20, above EMA50
  5. Pullback must NOT close below EMA50
  6. Enter on first bullish candle closing back above EMA20

Stop  : pullback swing low (lowest low during the pullback sequence)
Target: 2R  (no trailing, no optimisation)

Purpose : Test whether timeframe alone changes statistical validity.
          No filters, no threshold changes, no optimisation.
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
RESEARCH_ID    = "R015"
OUTPUT_FOLDER  = CONFIG["OUTPUT_FOLDER"]
CACHE_FOLDER   = CONFIG["CACHE_FOLDER"]
TRAIN_RATIO    = CONFIG["TRAIN_RATIO"]

SYMBOLS = [
    "BTC-USDT-SWAP",  "ETH-USDT-SWAP",  "LINK-USDT-SWAP", "AVAX-USDT-SWAP",
    "XRP-USDT-SWAP",  "DOGE-USDT-SWAP", "LTC-USDT-SWAP",  "BCH-USDT-SWAP",
]

# Timeframes to test — (bar string, minutes per bar, history months, min cache bars)
TIMEFRAMES = [
    {"bar": "15m", "minutes": 15,  "months": 6, "min_cache": 5_000,  "label": "15-minute"},
    {"bar": "1H",  "minutes": 60,  "months": 6, "min_cache":   500,  "label": "1-hour"},
]

MIN_OOS_TRADES = 30   # lower threshold at 1H (fewer bars available)

# Execution costs — locked (same as all research)
TAKER_FEE    = CONFIG["TAKER_FEE"]
SPREAD       = CONFIG["SPREAD"] * 0.5
SL_SLIPPAGE  = CONFIG["SL_SLIPPAGE"]
MIN_SL_PCT   = CONFIG["MIN_SL_PCT"]
RR           = CONFIG["RISK_REWARD"]
MAX_LEV      = CONFIG["MAX_LEVERAGE"]
STARTING_CAP = CONFIG["STARTING_CAPITAL"]
RISK_PCT     = CONFIG["RISK_PER_TRADE_PCT"]

# Promote criteria
PROMOTE_PF    = 1.20
PROMOTE_MDD   = 0.30
PROMOTE_MC_PP = 0.60

BG = "#0F1117"

OKX_HIST_URL  = "https://www.okx.com/api/v5/market/history-candles"
OKX_LIVE_URL  = "https://www.okx.com/api/v5/market/candles"
CANDLE_COLS   = ["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"]
PAGE_LIMIT    = 300
API_DELAY     = 0.05
DL_WORKERS    = 4
MAX_RETRIES   = 5

# R014 results for cross-timeframe comparison (from last run)
R014_RESULTS = {
    "BTC-USDT-SWAP":  {"trades": 101, "win_rate": 0.297, "pf": 0.239, "exp_r": -0.109, "mdd": -0.583, "avg_pb": 0.0003, "avg_trend_bars": 2, "avg_adx": 22.8},
    "ETH-USDT-SWAP":  {"trades": 163, "win_rate": 0.399, "pf": 0.379, "exp_r": +0.196, "mdd": -0.642, "avg_pb": 0.0003, "avg_trend_bars": 2, "avg_adx": 22.8},
    "LINK-USDT-SWAP": {"trades": 190, "win_rate": 0.342, "pf": 0.283, "exp_r": +0.026, "mdd": -0.774, "avg_pb": 0.0003, "avg_trend_bars": 2, "avg_adx": 22.8},
    "AVAX-USDT-SWAP": {"trades": 244, "win_rate": 0.291, "pf": 0.284, "exp_r": -0.127, "mdd": -0.882, "avg_pb": 0.0003, "avg_trend_bars": 2, "avg_adx": 22.8},
    "XRP-USDT-SWAP":  {"trades": 181, "win_rate": 0.298, "pf": 0.276, "exp_r": -0.105, "mdd": -0.790, "avg_pb": 0.0003, "avg_trend_bars": 2, "avg_adx": 22.8},
    "DOGE-USDT-SWAP": {"trades": 201, "win_rate": 0.279, "pf": 0.246, "exp_r": -0.164, "mdd": -0.827, "avg_pb": 0.0003, "avg_trend_bars": 2, "avg_adx": 22.8},
    "LTC-USDT-SWAP":  {"trades": 168, "win_rate": 0.369, "pf": 0.333, "exp_r": +0.107, "mdd": -0.688, "avg_pb": 0.0003, "avg_trend_bars": 2, "avg_adx": 22.8},
    "BCH-USDT-SWAP":  {"trades": 249, "win_rate": 0.281, "pf": 0.273, "exp_r": -0.157, "mdd": -0.893, "avg_pb": 0.0003, "avg_trend_bars": 2, "avg_adx": 22.8},
}
R014_PORTFOLIO = {"trades": 1497, "win_rate": 0.316, "pf": 0.289, "exp_r": -0.052, "mdd": -6.050, "avg_adx": 22.8, "avg_pb": 0.0003, "avg_trend_bars": 2}


# =============================================================================
# SECTION 1 — GENERIC DOWNLOADER (any OKX bar size)
# =============================================================================

def _parse_candles(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open", "high", "low", "close", "vol"]:
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
                time.sleep(2 ** attempt)
                continue
            d = r.json()
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
    """Download all available history for symbol at given bar size."""
    now_ms    = int(time.time() * 1000)
    target_ms = int(months * 30.44 * 24 * 3600 * 1000)
    cutoff_ms = now_ms - target_ms

    all_rows, after_ms_cursor, pages = [], None, 0

    while True:
        raw = _fetch_page(symbol, bar, after_ms=after_ms_cursor, use_history=True)

        if not raw:
            if pages == 0:
                raw = _fetch_page(symbol, bar, after_ms=None, use_history=False)
                if not raw:
                    break
            else:
                break

        all_rows.extend(raw)
        pages += 1
        oldest_ts = int(raw[-1][0])

        if pages % 20 == 0:
            oldest_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc).date()
            print(f"    {symbol.split('-')[0]:6s}[{bar}] page {pages:3d} | "
                  f"oldest {oldest_dt} | {len(all_rows):,} candles", flush=True)

        after_ms_cursor = oldest_ts

        if oldest_ts <= cutoff_ms:
            break

        time.sleep(API_DELAY)

    if not all_rows:
        raise RuntimeError(f"No data for {symbol} [{bar}]")

    df = _parse_candles(all_rows)
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df = df[df["datetime"] >= cutoff_dt]
    return df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def _refresh_symbol(symbol: str, bar: str, cached: pd.DataFrame) -> pd.DataFrame:
    """Append candles newer than the last cached bar."""
    last_ts  = cached["datetime"].iloc[-1]
    since_ms = int(last_ts.timestamp() * 1000)
    all_rows, after_ms_cursor = [], None

    for _ in range(20):
        raw = _fetch_page(symbol, bar, after_ms=after_ms_cursor, use_history=False)
        if not raw:
            raw = _fetch_page(symbol, bar, after_ms=after_ms_cursor, use_history=True)
        if not raw:
            break
        all_rows.extend(raw)
        oldest_ts = int(raw[-1][0])
        if oldest_ts <= since_ms:
            break
        after_ms_cursor = oldest_ts
        time.sleep(API_DELAY)

    if not all_rows:
        return cached

    new_df = _parse_candles(all_rows)
    new_df = new_df[new_df["datetime"] > last_ts]
    if len(new_df) == 0:
        return cached

    combined = (pd.concat([cached, new_df], ignore_index=True)
                .drop_duplicates("datetime")
                .sort_values("datetime")
                .reset_index(drop=True))
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
        print(f"  {symbol.split('-')[0]:6s}[{bar}] refreshing "
              f"({len(cached):,} cached, gap={gap_min:.0f} min)...", flush=True)
        return _refresh_symbol(symbol, bar, cached)

    print(f"  {symbol.split('-')[0]:6s}[{bar}] full download...", flush=True)
    df = _download_symbol(symbol, bar, months)
    _save_cache(df, symbol, bar)
    span_days = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds() / 86400
    print(f"  {symbol.split('-')[0]:6s}[{bar}] {len(df):,} candles  "
          f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}, "
          f"{span_days:.0f} days)", flush=True)
    return df


def download_all_parallel(bar: str, months: int, min_cache: int) -> dict:
    results, errors = {}, {}

    def _worker(sym):
        try:
            df = get_data(sym, bar, months, min_cache)
            return sym, df, None
        except Exception as e:
            return sym, None, str(e)

    with ThreadPoolExecutor(max_workers=DL_WORKERS) as pool:
        futures = {pool.submit(_worker, s): s for s in SYMBOLS}
        for fut in as_completed(futures):
            sym, df, err = fut.result()
            if err:
                print(f"  [WARN] {sym}[{bar}] failed: {err}", flush=True)
            else:
                results[sym] = df

    return results


# =============================================================================
# SECTION 2 — INDICATORS (identical to R014)
# =============================================================================

EMA_SLOPE_BARS = 10

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"]         = calc_ema(df["close"], 20)
    df["ema50"]         = calc_ema(df["close"], 50)
    df["ema200"]        = calc_ema(df["close"], 200)
    df["ema200_lag"]    = df["ema200"].shift(EMA_SLOPE_BARS)
    df["ema200_slope"]  = df["ema200"] > df["ema200_lag"]
    df["atr"]           = calc_atr(df, 14)
    df["adx"]           = calc_adx(df, 14)
    return df


# =============================================================================
# SECTION 3 — PULLBACK SIGNAL DETECTION (identical to R014)
# =============================================================================

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n  = len(df)

    close  = df["close"].values
    low    = df["low"].values
    ema20  = df["ema20"].values
    ema50  = df["ema50"].values
    ema200 = df["ema200"].values
    slope  = df["ema200_slope"].values.astype(bool)

    signal             = np.zeros(n, dtype=bool)
    pb_swing_low       = np.full(n, np.nan)
    pb_depth_pct       = np.full(n, np.nan)
    dist_ema50_pct_arr = np.full(n, np.nan)
    trend_bars_arr     = np.zeros(n, dtype=np.int32)

    in_pullback   = False
    pb_low        = np.nan
    pb_max_depth  = 0.0
    trend_running = 0

    for i in range(n):
        if np.isnan(ema200[i]) or np.isnan(ema50[i]) or np.isnan(ema20[i]):
            in_pullback   = False
            pb_low        = np.nan
            pb_max_depth  = 0.0
            trend_running = 0
            continue

        trend_ok = (
            close[i]  > ema200[i] and
            slope[i]  and
            ema20[i]  > ema50[i]
        )

        if not trend_ok:
            in_pullback   = False
            pb_low        = np.nan
            pb_max_depth  = 0.0
            trend_running = 0
            continue

        trend_running += 1

        in_pb_zone  = (close[i] < ema20[i]) and (close[i] > ema50[i])
        below_ema50 = close[i] <= ema50[i]

        if below_ema50:
            in_pullback  = False
            pb_low       = np.nan
            pb_max_depth = 0.0

        elif in_pb_zone:
            in_pullback = True
            cur_low = low[i]
            pb_low  = cur_low if np.isnan(pb_low) else min(pb_low, cur_low)
            depth   = (ema20[i] - close[i]) / ema20[i]
            pb_max_depth = max(pb_max_depth, depth)

        elif in_pullback and close[i] > ema20[i]:
            signal[i]               = True
            pb_swing_low[i]         = pb_low
            pb_depth_pct[i]         = pb_max_depth
            dist_ema50_pct_arr[i]   = (close[i] - ema50[i]) / ema50[i]
            trend_bars_arr[i]       = trend_running
            in_pullback   = False
            pb_low        = np.nan
            pb_max_depth  = 0.0

    df["signal"]             = signal
    df["pullback_swing_low"] = pb_swing_low
    df["pullback_depth_pct"] = pb_depth_pct
    df["dist_ema50_pct"]     = dist_ema50_pct_arr
    df["trend_bars_before"]  = trend_bars_arr
    return df


# =============================================================================
# SECTION 4 — BACKTEST ENGINE (identical to R014, minutes_per_bar for hold time)
# =============================================================================

def run_backtest(df: pd.DataFrame, label: str, minutes_per_bar: int) -> dict:
    n           = len(df)
    in_pos      = False
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
    sig_arr    = df["signal"].values
    psl_arr    = df["pullback_swing_low"].values
    pd_arr     = df["pullback_depth_pct"].values
    de50_arr   = df["dist_ema50_pct"].values
    tb_arr     = df["trend_bars_before"].values
    atr_arr    = df["atr"].values
    adx_arr    = df["adx"].values
    ema200_arr = df["ema200"].values
    ema200_lag = df["ema200_lag"].values

    for i in range(1, n):
        hi = high_arr[i]
        lo = low_arr[i]

        if in_pos:
            pnl_hi = (hi - entry_price) / (entry_price - sl)
            pnl_lo = (lo - entry_price) / (entry_price - sl)
            mfe_track = max(mfe_track, pnl_hi)
            mae_track = min(mae_track, pnl_lo)

            sl_hit = lo <= sl
            tp_hit = hi >= tp

            if sl_hit or tp_hit:
                exit_price = (sl * (1.0 - SL_SLIPPAGE)) if sl_hit else tp
                exit_type  = "SL" if sl_hit else "TP"

                sl_dist   = entry_price - sl
                gross_pnl = (exit_price - entry_price) * pos_size
                ne        = entry_price * pos_size
                nx        = exit_price  * pos_size
                cost_fee  = (ne + nx) * TAKER_FEE
                cost_spd  = (ne + nx) * SPREAD
                cost_slip = (sl - exit_price) * pos_size if exit_type == "SL" else 0.0
                net_pnl   = gross_pnl - cost_fee - cost_spd - cost_slip
                r_mult    = (exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0
                hold_bars = i - entry_idx
                hold_min  = hold_bars * minutes_per_bar

                trades.append({
                    "label":           label,
                    "entry_time":      entry_time,
                    "exit_time":       pd.Timestamp(dt_arr[i]),
                    "entry_price":     entry_price,
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
                    "mfe_r":           mfe_track,
                    "mae_r":           mae_track,
                    "pb_depth_pct":    pd_arr[i - 1],
                    "dist_ema50_pct":  de50_arr[i - 1],
                    "trend_bars":      tb_arr[i - 1],
                    "atr_at_entry":    atr_arr[i - 1],
                    "adx_at_entry":    adx_arr[i - 1],
                    "ema200_slope_val": (
                        (ema200_arr[i-1] - ema200_lag[i-1]) / ema200_arr[i-1]
                        if not np.isnan(ema200_lag[i-1]) and ema200_arr[i-1] > 0
                        else np.nan
                    ),
                })
                capital += net_pnl
                in_pos   = False
            continue

        if sig_arr[i - 1]:
            ep_raw = open_arr[i]
            sl_raw = psl_arr[i - 1]

            if np.isnan(sl_raw):
                continue

            sl_dist = ep_raw - sl_raw
            if sl_dist <= 0 or sl_dist / ep_raw < MIN_SL_PCT:
                continue

            tp_price     = ep_raw + RR * sl_dist
            risk_dollars = capital * RISK_PCT
            size         = min(risk_dollars / sl_dist, (capital * MAX_LEV) / ep_raw)

            entry_price = ep_raw
            sl          = sl_raw
            tp          = tp_price
            pos_size    = size
            entry_time  = pd.Timestamp(dt_arr[i])
            entry_idx   = i
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
    bpy    = 365 * 24 * 60 / minutes_per_bar   # bars per year at this timeframe
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

    def _mean(col):
        return trades_df[col].dropna().mean() if col in trades_df.columns else np.nan
    def _wmean(col):
        return wins[col].dropna().mean() if col in wins.columns and len(wins) else np.nan
    def _lmean(col):
        return loses[col].dropna().mean() if col in loses.columns and len(loses) else np.nan

    return {
        "avg_pb_depth_pct":   _mean("pb_depth_pct"),
        "avg_hold_min":       _mean("holding_minutes"),
        "avg_atr_entry":      _mean("atr_at_entry"),
        "avg_adx_entry":      _mean("adx_at_entry"),
        "avg_ema200_slope":   _mean("ema200_slope_val"),
        "avg_dist_ema50_pct": _mean("dist_ema50_pct"),
        "avg_trend_bars":     _mean("trend_bars"),
        "avg_mfe_r":          _mean("mfe_r"),
        "avg_mae_r":          _mean("mae_r"),
        "win_avg_hold_min":   _wmean("holding_minutes"),
        "loss_avg_hold_min":  _lmean("holding_minutes"),
        "win_avg_adx":        _wmean("adx_at_entry"),
        "loss_avg_adx":       _lmean("adx_at_entry"),
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
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="#AAA", labelsize=8)
    ax.xaxis.label.set_color("#AAA")
    ax.yaxis.label.set_color("#AAA")
    ax.title.set_color("#EEE")
    return ax


def plot_equity_curves(sym_metrics: dict, port_m: dict, tf_label: str, tf_bar: str):
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
        color = PALETTE[k % len(PALETTE)]
        ax.plot(eq, color=color, lw=1)
        ax.axhline(STARTING_CAP, color="#555", lw=0.5, ls="--")
        ax.set_title(f"{short}  PF={m['profit_factor']:.3f}  n={m['n_trades']}", fontsize=9)
        ax.set_ylabel("Equity $", fontsize=7)

    ax_p = _ax(fig, gs[1, 3])
    if port_m:
        ax_p.plot(port_m["equity"], color="#FFD700", lw=1.5)
        ax_p.axhline(STARTING_CAP, color="#555", lw=0.5, ls="--")
        ax_p.set_title(f"PORTFOLIO  PF={port_m['profit_factor']:.3f}  "
                       f"n={port_m['n_trades']}", fontsize=9)
        ax_p.set_ylabel("Equity $", fontsize=7)

    fig.suptitle(f"R015 Equity Curves — Trend Continuation After Pullback [{tf_label}]",
                 color="#EEE", fontsize=12, y=1.01)
    path = f"{OUTPUT_FOLDER}/r015_equity_{tf_bar}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_drawdown_curves(sym_metrics: dict, port_m: dict, tf_label: str, tf_bar: str):
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

    fig.suptitle(f"R015 Drawdown Curves [{tf_label}]", color="#EEE", fontsize=12, y=1.01)
    path = f"{OUTPUT_FOLDER}/r015_drawdown_{tf_bar}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_symbol_comparison(sym_metrics: dict, tf_label: str, tf_bar: str):
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

    fig.suptitle(f"R015 Symbol Comparison [{tf_label}]", color="#EEE", fontsize=12, y=1.01)
    path = f"{OUTPUT_FOLDER}/r015_symbols_{tf_bar}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_cross_timeframe_comparison(tf_results: dict):
    """
    Side-by-side bar chart comparing 1m (R014), 15m, and 1H across key metrics.
    tf_results: {tf_label: {metric: value, ...}}
    """
    tfs = ["1m (R014)", "15m", "1H"]
    colors_tf = ["#4A90D9", "#FFB347", "#00C49A"]

    syms_short = [s.split("-")[0] for s in SYMBOLS]
    n_syms = len(syms_short)

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    metrics_plot = [
        ("pf",   "Profit Factor",    1.20, False),
        ("wr",   "Win Rate (%)",     50,   False),
        ("mdd",  "Max Drawdown (%)", 30,   True ),
        ("exp",  "Expectancy R",     0,    False),
        ("trades","Trade Count",     MIN_OOS_TRADES, False),
        ("adx",  "Avg ADX at Entry", 25,   False),
        ("pb",   "Avg Pullback Depth (%)", None, False),
        ("trend_bars", "Avg Trend Bars Before", None, False),
    ]

    x = np.arange(n_syms)
    width = 0.27

    for plot_idx, (metric_key, title, refline, abs_val) in enumerate(metrics_plot[:9]):
        row, col = divmod(plot_idx, 3)
        ax = _ax(fig, gs[row, col])

        for tf_idx, tf_key in enumerate(["1m", "15m", "1H"]):
            if tf_key not in tf_results:
                continue
            vals = []
            for sym in SYMBOLS:
                s_res = tf_results[tf_key].get(sym, {})
                v = s_res.get(metric_key, 0) or 0
                if abs_val:
                    v = abs(v)
                vals.append(v)
            offset = (tf_idx - 1) * width
            ax.bar(x + offset, vals, width,
                   color=colors_tf[tf_idx], alpha=0.85, label=tfs[tf_idx])

        if refline is not None:
            ax.axhline(refline, color="#FFF", lw=0.7, ls="--")
        ax.set_title(title, fontsize=9, color="#EEE")
        ax.set_xticks(x)
        ax.set_xticklabels(syms_short, fontsize=7)
        if plot_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("R015 Cross-Timeframe Comparison: 1m vs 15m vs 1H\n"
                 "Identical Strategy — Trend Continuation After Pullback",
                 color="#EEE", fontsize=13, y=1.01)
    path = f"{OUTPUT_FOLDER}/r015_cross_timeframe.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_portfolio_comparison(portfolio_summary: dict):
    """Bar chart of portfolio-level PF, WR, MDD, Expectancy across timeframes."""
    tfs = list(portfolio_summary.keys())
    colors_tf = {"1m (R014)": "#4A90D9", "15m": "#FFB347", "1H": "#00C49A"}

    metrics = [
        ("pf",  "Profit Factor", 1.20),
        ("wr",  "Win Rate (%)",  50),
        ("mdd", "Max DD (%)",    30),
        ("exp", "Expectancy R",  0),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, (key, title, ref) in zip(axes, metrics):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.tick_params(colors="#AAA", labelsize=8)

        vals   = [abs(portfolio_summary[tf].get(key, 0)) if key == "mdd"
                  else portfolio_summary[tf].get(key, 0)
                  for tf in tfs]
        clrs   = [colors_tf.get(tf, "#888") for tf in tfs]
        ax.bar(tfs, vals, color=clrs, alpha=0.85)
        ax.axhline(ref, color="#FFF", lw=0.7, ls="--")
        ax.set_title(title, color="#EEE", fontsize=10)
        ax.set_ylabel(key.upper(), fontsize=8, color="#AAA")

    fig.suptitle("R015 Portfolio Summary — 1m vs 15m vs 1H",
                 color="#EEE", fontsize=13)
    path = f"{OUTPUT_FOLDER}/r015_portfolio_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# SECTION 8 — REPORT
# =============================================================================

def print_tf_report(sym_metrics: dict, port_m: dict, attr: dict,
                    oos_dates: dict, tf_cfg: dict):
    W   = 110
    bar = tf_cfg["bar"]
    lbl = tf_cfg["label"]
    print()
    print("=" * W)
    print(f"  QUANTLAB AI — RESEARCH #015  [{lbl}]")
    print(f"  Trend Continuation After Pullback — {lbl} candles")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * W)
    print()
    print("  Entry  : close > EMA200 (rising) | EMA20 > EMA50 | "
          "pullback below EMA20 (above EMA50) | reclaim EMA20")
    print("  Stop   : pullback swing low")
    print("  Target : 2R")
    print()

    for sym, dates in oos_dates.items():
        short = sym.split("-")[0]
        print(f"  {short:6s}  OOS: {dates[0]} → {dates[1]}")
    print()

    header_syms = [s.split("-")[0] for s in sym_metrics.keys()]
    col_w = max(14, max(len(h) for h in header_syms) + 2)

    def _row(label, vals):
        print(f"  {label:<28s}" + "".join(f"{v:>{col_w}}" for v in vals))

    print("  " + "-" * (28 + col_w * len(header_syms)))
    _row("Metric", header_syms)
    print("  " + "-" * (28 + col_w * len(header_syms)))

    _row("Trades",         [f"{v['metrics']['n_trades']:>{col_w}d}"         for v in sym_metrics.values()])
    _row("Win Rate",       [f"{v['metrics']['win_rate']*100:>{col_w-1}.1f}%" for v in sym_metrics.values()])
    _row("Profit Factor",  [f"{v['metrics']['profit_factor']:>{col_w}.3f}"  for v in sym_metrics.values()])
    _row("Expectancy R",   [f"{v['metrics']['expectancy_r']:>+{col_w}.3f}"  for v in sym_metrics.values()])
    _row("Net P&L ($)",    [f"{v['metrics']['net_profit']:>{col_w}.0f}"     for v in sym_metrics.values()])
    _row("Max Drawdown",   [f"{v['metrics']['max_drawdown']*100:>{col_w-1}.1f}%" for v in sym_metrics.values()])
    _row("Sharpe",         [f"{v['metrics']['sharpe']:>{col_w}.2f}"         for v in sym_metrics.values()])
    _row("MC Prob Profit", [f"{v['mc']['prob_profit']*100:>{col_w-1}.1f}%"  for v in sym_metrics.values()])
    _row("Verdict",        [f"{v['verdict']:>{col_w}}"                      for v in sym_metrics.values()])
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
        print(f"  ── ATTRIBUTION [{lbl}] ────────────────────────────────────")
        print(f"  Avg pullback depth    : {attr.get('avg_pb_depth_pct', 0)*100:.3f}%  (vs EMA20)")
        print(f"  Avg holding time      : {attr.get('avg_hold_min', 0):.0f} min")
        print(f"  Avg ATR at entry      : {attr.get('avg_atr_entry', 0):.6f}")
        print(f"  Avg ADX at entry      : {attr.get('avg_adx_entry', 0):.1f}")
        print(f"  Avg EMA200 slope      : {attr.get('avg_ema200_slope', 0)*100:.4f}%")
        print(f"  Avg dist EMA50        : {attr.get('avg_dist_ema50_pct', 0)*100:.3f}%")
        print(f"  Avg trend bars before : {attr.get('avg_trend_bars', 0):.1f}")
        print(f"  Avg MFE               : {attr.get('avg_mfe_r', 0):+.3f}R")
        print(f"  Avg MAE               : {attr.get('avg_mae_r', 0):+.3f}R")
        print(f"  Win avg hold          : {attr.get('win_avg_hold_min', 0):.0f} min")
        print(f"  Loss avg hold         : {attr.get('loss_avg_hold_min', 0):.0f} min")
        print(f"  Win avg ADX           : {attr.get('win_avg_adx', 0):.1f}")
        print(f"  Loss avg ADX          : {attr.get('loss_avg_adx', 0):.1f}")
        print()

    if port_m:
        pf          = port_m["profit_factor"]
        exp         = port_m["expectancy_r"]
        mdd         = abs(port_m["max_drawdown"])
        n           = port_m["n_trades"]
        n_sym_pass  = sum(
            1 for v in sym_metrics.values()
            if v["metrics"]["profit_factor"] >= PROMOTE_PF
               and v["metrics"]["net_profit"] > 0
        )
        mc_port = monte_carlo(port_m["pnls"])
        mc_pp   = mc_port["prob_profit"]

        pass_pf   = pf > PROMOTE_PF
        pass_exp  = exp > 0
        pass_mdd  = mdd < PROMOTE_MDD
        pass_n    = n >= MIN_OOS_TRADES
        pass_syms = n_sym_pass >= 2
        pass_mc   = mc_pp >= PROMOTE_MC_PP

        final_pass  = all([pass_pf, pass_exp, pass_mdd, pass_n, pass_syms, pass_mc])
        verdict_str = "✅  PASS — SUITABLE FOR FORWARD DEMO TESTING" if final_pass else "❌  REJECT"

        print(f"  ── FINAL VERDICT [{lbl}] ─────────────────────────────────")
        print(f"  Combined PF > {PROMOTE_PF:.2f}    : {'✓' if pass_pf  else '✗'}  ({pf:.3f})")
        print(f"  Positive expectancy     : {'✓' if pass_exp else '✗'}  ({exp:+.3f}R)")
        print(f"  MDD < {PROMOTE_MDD*100:.0f}%           : {'✓' if pass_mdd else '✗'}  ({mdd*100:.1f}%)")
        print(f"  ≥ {MIN_OOS_TRADES} OOS trades        : {'✓' if pass_n   else '✗'}  ({n})")
        print(f"  ≥ 2 profitable symbols  : {'✓' if pass_syms else '✗'}  ({n_sym_pass}/{len(sym_metrics)})")
        print(f"  MC Prob Profit ≥ {PROMOTE_MC_PP*100:.0f}%  : {'✓' if pass_mc else '✗'}  ({mc_pp*100:.1f}%)")
        print()
        print(f"  {verdict_str}")
        print()

    print("=" * W)


def print_cross_timeframe_comparison(tf_summaries: dict, tf_attrs: dict):
    """Print the 4-question cross-timeframe analysis."""
    W = 110
    print()
    print("=" * W)
    print("  CROSS-TIMEFRAME COMPARISON")
    print("  1m (R014)  vs  15m  vs  1H")
    print("=" * W)
    print()

    # ── Portfolio table ──────────────────────────────────────────────────────
    headers = ["1m (R014)", "15m", "1H"]
    col_w   = 18

    def _row(label, vals):
        print(f"  {label:<30s}" + "".join(f"{v:>{col_w}}" for v in vals))

    print("  " + "-" * (30 + col_w * 3))
    _row("Metric", headers)
    print("  " + "-" * (30 + col_w * 3))

    def _port(key, fmt, tf_key):
        v = tf_summaries.get(tf_key, {}).get(key, None)
        if v is None:
            return "  —  "
        return format(v, fmt)

    # Build rows from stored summaries
    def _row_vals(label, key, fmt):
        vals = [_port(key, fmt, tf) for tf in ["1m", "15m", "1H"]]
        _row(label, vals)

    _row_vals("Portfolio PF",       "pf",      ".3f")
    _row_vals("Portfolio Win Rate", "wr_pct",  ".1f")
    _row_vals("Portfolio Expectancy R", "exp", "+.3f")
    _row_vals("Portfolio MDD",      "mdd_pct", ".1f")
    _row_vals("Portfolio Trades",   "trades",  "d")
    print("  " + "-" * (30 + col_w * 3))
    print()

    # ── Attribution table ────────────────────────────────────────────────────
    print("  ── ATTRIBUTION COMPARISON ──────────────────────────────────────────")
    print()

    def _attr_row(label, key, mult=1, fmt=".3f", suffix=""):
        vals = []
        for tf in ["1m", "15m", "1H"]:
            a = tf_attrs.get(tf, {})
            v = a.get(key, None)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                vals.append("  —  ")
            else:
                vals.append(f"{v * mult:{fmt}}{suffix}")
        _row(label, vals)

    _attr_row("Avg pullback depth",    "avg_pb_depth_pct",   100, ".3f", "%")
    _attr_row("Avg ADX at entry",      "avg_adx_entry",       1,  ".1f")
    _attr_row("Avg trend bars before", "avg_trend_bars",      1,  ".1f",  " bars")
    _attr_row("Avg hold time",         "avg_hold_min",        1,  ".0f",  " min")
    _attr_row("Avg MFE",               "avg_mfe_r",           1,  "+.3f", "R")
    _attr_row("Avg MAE",               "avg_mae_r",           1,  "+.3f", "R")
    print()

    # ── 5 Questions ──────────────────────────────────────────────────────────
    pf_1m  = tf_summaries.get("1m",  {}).get("pf", 0) or 0
    pf_15m = tf_summaries.get("15m", {}).get("pf", 0) or 0
    pf_1h  = tf_summaries.get("1H",  {}).get("pf", 0) or 0

    pb_1m  = (tf_attrs.get("1m",  {}).get("avg_pb_depth_pct", 0) or 0) * 100
    pb_15m = (tf_attrs.get("15m", {}).get("avg_pb_depth_pct", 0) or 0) * 100
    pb_1h  = (tf_attrs.get("1H",  {}).get("avg_pb_depth_pct", 0) or 0) * 100

    tb_1m  = tf_attrs.get("1m",  {}).get("avg_trend_bars", 0) or 0
    tb_15m = tf_attrs.get("15m", {}).get("avg_trend_bars", 0) or 0
    tb_1h  = tf_attrs.get("1H",  {}).get("avg_trend_bars", 0) or 0

    adx_1m  = tf_attrs.get("1m",  {}).get("avg_adx_entry", 0) or 0
    adx_15m = tf_attrs.get("15m", {}).get("avg_adx_entry", 0) or 0
    adx_1h  = tf_attrs.get("1H",  {}).get("avg_adx_entry", 0) or 0

    def _yn(cond): return "YES ✓" if cond else "NO  ✗"
    def _best(v1m, v15m, v1h, higher_better=True):
        vals = {"1m": v1m, "15m": v15m, "1H": v1h}
        best = max(vals, key=lambda k: vals[k]) if higher_better else min(vals, key=lambda k: vals[k])
        return f"Best: {best} ({vals[best]:.3f})" if isinstance(vals[best], float) else f"Best: {best}"

    print("  ── FIVE DIAGNOSTIC QUESTIONS ────────────────────────────────────────")
    print()
    print(f"  Q1. Does increasing timeframe improve Profit Factor?")
    print(f"      1m={pf_1m:.3f}  15m={pf_15m:.3f}  1H={pf_1h:.3f}")
    pf_improves = pf_15m > pf_1m or pf_1h > pf_1m
    print(f"      → {_yn(pf_improves)}  ({_best(pf_1m, pf_15m, pf_1h)})")
    print()

    print(f"  Q2. Does trend duration become significantly longer (bars)?")
    print(f"      1m={tb_1m:.1f} bars  15m={tb_15m:.1f} bars  1H={tb_1h:.1f} bars")
    tb_improves = tb_15m > tb_1m * 2 or tb_1h > tb_1m * 2
    print(f"      → {_yn(tb_improves)}")
    print()

    print(f"  Q3. Does pullback depth become structurally meaningful (>0.5%)?")
    print(f"      1m={pb_1m:.3f}%  15m={pb_15m:.3f}%  1H={pb_1h:.3f}%")
    pb_meaningful = pb_15m > 0.5 or pb_1h > 0.5
    print(f"      → {_yn(pb_meaningful)}")
    print()

    print(f"  Q4. Does trading cost become less important relative to target size?")
    # At 2R, cost importance scales inversely with SL size
    # Larger TF pullbacks = larger SL = cost is smaller % of gross P&L
    print(f"      (Higher pullback depth → larger SL → cost/gross-pnl ratio shrinks)")
    cost_matters_less = pb_15m > pb_1m * 5 or pb_1h > pb_1m * 5
    print(f"      1m pullback: {pb_1m:.3f}% → 15m: {pb_15m:.3f}% → 1H: {pb_1h:.3f}%")
    print(f"      → {_yn(cost_matters_less)}")
    print()

    print(f"  Q5. Does any timeframe exceed PF > 1.20?")
    any_pass = pf_15m > PROMOTE_PF or pf_1h > PROMOTE_PF
    print(f"      1m={pf_1m:.3f}  15m={pf_15m:.3f}  1H={pf_1h:.3f}  (threshold={PROMOTE_PF})")
    print(f"      → {_yn(any_pass)}")
    print()

    print("=" * W)


# =============================================================================
# SECTION 9 — MAIN
# =============================================================================

def run_timeframe(tf_cfg: dict) -> tuple:
    """
    Download data, compute indicators, run backtest for one timeframe.
    Returns (sym_metrics, port_m, attr, oos_dates, tf_sym_results).
    """
    bar          = tf_cfg["bar"]
    minutes      = tf_cfg["minutes"]
    months       = tf_cfg["months"]
    min_cache    = tf_cfg["min_cache"]
    label        = tf_cfg["label"]

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

    sym_metrics   = {}
    all_trades    = []
    oos_dates     = {}
    tf_sym_results = {}

    for sym, df_raw in raw_data.items():
        short = sym.split("-")[0]

        df = add_indicators(df_raw)
        df = df.dropna(subset=["ema200", "ema50", "ema20"]).reset_index(drop=True)

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

        df_oos    = compute_signals(df_oos)
        sig_count = df_oos["signal"].sum()

        bt  = run_backtest(df_oos, label=sym, minutes_per_bar=minutes)
        m   = compute_metrics(bt["trades"], sym)
        mc  = monte_carlo(m["pnls"])
        v   = _verdict_from_metrics(m, mc)

        sym_metrics[sym] = {"metrics": m, "mc": mc, "verdict": v}
        all_trades.extend(bt["trades"])

        # Store per-symbol summary for cross-TF chart
        trades_df_sym = pd.DataFrame(bt["trades"])
        avg_pb  = trades_df_sym["pb_depth_pct"].mean()  if not trades_df_sym.empty else 0
        avg_adx = trades_df_sym["adx_at_entry"].mean()  if not trades_df_sym.empty else 0
        avg_tb  = trades_df_sym["trend_bars"].mean()    if not trades_df_sym.empty else 0

        tf_sym_results[sym] = {
            "trades": m["n_trades"],
            "wr":     m["win_rate"] * 100,
            "pf":     m["profit_factor"],
            "exp":    m["expectancy_r"],
            "mdd":    abs(m["max_drawdown"]) * 100,
            "adx":    avg_adx,
            "pb":     avg_pb * 100,
            "trend_bars": avg_tb,
        }

        print(f"  {short:6s}  signals={sig_count:4d}  "
              f"trades={m['n_trades']:4d}  "
              f"WR={m['win_rate']*100:5.1f}%  "
              f"PF={m['profit_factor']:.3f}  "
              f"[{v}]")

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

    return sym_metrics, port_m, attr, oos_dates, tf_sym_results


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CACHE_FOLDER, exist_ok=True)

    print()
    print("╔" + "═" * 79 + "╗")
    print("║  QUANTLAB AI — RESEARCH #015" + " " * 50 + "║")
    print("║  Trend Continuation After Pullback — Higher Timeframes" + " " * 24 + "║")
    print("╚" + "═" * 79 + "╝")
    print()
    print("  Hypothesis : Does timeframe alone change statistical validity?")
    print("  Strategy   : Identical to R014 — no filters, no optimisation")
    print("  Timeframes : 15-minute  |  1-hour")
    print("  Symbols    : BTC ETH LINK AVAX XRP DOGE LTC BCH")
    print()

    # ── Run each timeframe ────────────────────────────────────────────────────
    tf_results_all  = {}   # {bar: sym_results_dict}
    tf_port_summary = {}   # {bar_label: {pf, wr, exp, mdd, trades}}
    tf_attrs        = {}   # {bar_label: attr_dict}

    # Seed with R014 1m data for comparison
    # Use the pre-computed R014 portfolio attribution
    tf_attrs["1m"] = {
        "avg_pb_depth_pct":   R014_PORTFOLIO["avg_pb"],
        "avg_adx_entry":      R014_PORTFOLIO["avg_adx"],
        "avg_trend_bars":     R014_PORTFOLIO["avg_trend_bars"],
        "avg_hold_min":       17,
        "avg_mfe_r":          1.205,
        "avg_mae_r":          -0.993,
        "win_avg_hold_min":   23,
        "loss_avg_hold_min":  14,
        "win_avg_adx":        25.7,
        "loss_avg_adx":       21.5,
    }
    tf_port_summary["1m"] = {
        "pf":      R014_PORTFOLIO["pf"],
        "wr_pct":  R014_PORTFOLIO["win_rate"] * 100,
        "exp":     R014_PORTFOLIO["exp_r"],
        "mdd_pct": abs(R014_PORTFOLIO["mdd"]) * 100,
        "trades":  R014_PORTFOLIO["trades"],
    }
    # Seed R014 per-symbol results
    tf_results_all["1m"] = {
        sym: {
            "trades": v["trades"],
            "wr":     v["win_rate"] * 100,
            "pf":     v["pf"],
            "exp":    v["exp_r"],
            "mdd":    abs(v["mdd"]) * 100,
            "adx":    v["avg_adx"],
            "pb":     v["avg_pb"] * 100,
            "trend_bars": v["avg_trend_bars"],
        }
        for sym, v in R014_RESULTS.items()
    }

    all_journal_rows = []

    for tf_cfg in TIMEFRAMES:
        bar   = tf_cfg["bar"]
        label = tf_cfg["label"]

        sym_metrics, port_m, attr, oos_dates, tf_sym_res = run_timeframe(tf_cfg)

        if not sym_metrics:
            print(f"  [WARN] No results for {label} — skipping charts/report")
            continue

        # Charts for this timeframe
        print(f"\n  Generating charts for {label}...")
        plot_equity_curves(sym_metrics, port_m, label, bar)
        plot_drawdown_curves(sym_metrics, port_m, label, bar)
        plot_symbol_comparison(sym_metrics, label, bar)

        # Report
        print_tf_report(sym_metrics, port_m, attr, oos_dates, tf_cfg)

        # Collect for cross-TF comparison
        tf_results_all[bar] = tf_sym_res

        if port_m:
            mc_port = monte_carlo(port_m["pnls"])
            tf_port_summary[bar] = {
                "pf":      port_m["profit_factor"],
                "wr_pct":  port_m["win_rate"] * 100,
                "exp":     port_m["expectancy_r"],
                "mdd_pct": abs(port_m["max_drawdown"]) * 100,
                "trades":  port_m["n_trades"],
            }

        tf_attrs[bar] = attr

        # Journal
        for sym, data in sym_metrics.items():
            m  = data["metrics"]
            mc = data["mc"]
            v  = data["verdict"]
            all_journal_rows.append(_journal_row(
                strategy_name = f"TrendPullback_{bar}_{sym.split('-')[0]}",
                symbol        = sym,
                m             = m,
                mc            = mc,
                verdict       = v,
            ))

    # ── Cross-timeframe comparison ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CROSS-TIMEFRAME COMPARISON CHARTS")
    print(f"{'='*70}")

    plot_cross_timeframe_comparison(tf_results_all)
    plot_portfolio_comparison(tf_port_summary)

    print_cross_timeframe_comparison(tf_port_summary, tf_attrs)

    # ── Journal ───────────────────────────────────────────────────────────────
    if all_journal_rows:
        print(f"\n{'='*70}")
        print(f"  Writing journal")
        print(f"{'='*70}")
        append_journal(all_journal_rows)
        print(f"  Journal updated → {OUTPUT_FOLDER}/research_journal.csv")

    print()
    print(f"  All outputs → {OUTPUT_FOLDER}/r015_*")
    print(f"  Research #015 complete.")
    print()


if __name__ == "__main__":
    main()
