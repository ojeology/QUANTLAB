"""
=============================================================================
QUANTLAB AI — RESEARCH #014
Hypothesis : Trend Continuation After Pullback
Timeframe  : 1-minute candles
Symbols    : BTC, ETH, LINK, AVAX, XRP, DOGE, LTC, BCH  (OKX perps)

Entry rules (LONG only):
  1. close > EMA200
  2. EMA200 slope positive  (EMA200[i] > EMA200[i-10])
  3. EMA20 > EMA50
  4. Price pulls back: at least one bar closes below EMA20, above EMA50
  5. Pullback must NOT close below EMA50
  6. Enter on first bullish candle closing back above EMA20

Stop  : pullback swing low (lowest low during the pullback sequence)
Target: 2R  (no trailing, no optimisation)

New hypothesis — not related to any previous research.
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
RESEARCH_ID    = "R014"
OUTPUT_FOLDER  = CONFIG["OUTPUT_FOLDER"]
CACHE_FOLDER   = CONFIG["CACHE_FOLDER"]
TIMEFRAME      = "1m"
MONTHS_HISTORY = 3          # OKX history-candles provides ~3 months for 1m; ~131k candles/symbol
TRAIN_RATIO    = CONFIG["TRAIN_RATIO"]

SYMBOLS = [
    "BTC-USDT-SWAP",  "ETH-USDT-SWAP",  "LINK-USDT-SWAP", "AVAX-USDT-SWAP",
    "XRP-USDT-SWAP",  "DOGE-USDT-SWAP", "LTC-USDT-SWAP",  "BCH-USDT-SWAP",
]
MIN_OOS_TRADES = 100

# Execution costs — locked (same as all previous research)
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
PAGE_LIMIT    = 300    # OKX max per page (history-candles supports 300)
API_DELAY     = 0.05   # seconds between pages per worker
DL_WORKERS    = 4      # parallel workers (4 × ~440 pages × 0.05s ≈ 22s per batch → ~50s total)
MAX_RETRIES   = 5      # per page retry limit


# =============================================================================
# SECTION 1 — FAST PARALLEL 1-MINUTE DATA DOWNLOAD
# =============================================================================

def _parse_candles_1m(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return (df[["datetime","open","high","low","close","vol"]]
            .sort_values("datetime").reset_index(drop=True))


def _fetch_page_1m(symbol: str, after_ms=None, use_history=True) -> list:
    """Fetch one page with retry + exponential backoff on failure / rate limit."""
    url    = OKX_HIST_URL if use_history else OKX_LIVE_URL
    params = {"instId": symbol, "bar": TIMEFRAME, "limit": str(PAGE_LIMIT)}
    if after_ms is not None:
        params["after"] = str(after_ms)

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            d = r.json()
            code = d.get("code", "-1")
            if code == "0":
                data = d.get("data", [])
                return data
            # OKX error codes that mean "no more data" (end of history)
            if code in ("50011", "50013"):
                return []
            # Other codes: backoff and retry
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return []


def _cache_path_1m(symbol: str) -> str:
    safe = symbol.replace("-", "_") + f"_{TIMEFRAME}"
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    return os.path.join(CACHE_FOLDER, f"{safe}.parquet")


def _load_cache_1m(symbol: str):
    path = _cache_path_1m(symbol)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def _save_cache_1m(df: pd.DataFrame, symbol: str):
    df.to_parquet(_cache_path_1m(symbol), index=False)


def _download_symbol_1m(symbol: str, print_fn=print) -> pd.DataFrame:
    """
    Download all available 1m history for symbol.
    OKX history-candles for 1m bars goes back ~3 months.
    Paginates backwards using the `after` cursor until the API
    returns empty (no more historical data available).
    """
    now_ms    = int(time.time() * 1000)
    target_ms = int(MONTHS_HISTORY * 30.44 * 24 * 3600 * 1000)
    cutoff_ms = now_ms - target_ms  # hard cutoff (we take what API gives)

    all_rows, after_ms_cursor, pages = [], None, 0
    consecutive_empty = 0

    while True:
        raw = _fetch_page_1m(symbol, after_ms=after_ms_cursor, use_history=True)

        if not raw:
            # history endpoint exhausted — try live endpoint once to catch
            # any very recent candles not yet in history
            if pages == 0:
                raw = _fetch_page_1m(symbol, after_ms=None, use_history=False)
                if not raw:
                    break
            else:
                break

        all_rows.extend(raw)
        pages += 1
        oldest_ts = int(raw[-1][0])
        newest_ts = int(raw[0][0])

        # Progress every 50 pages
        if pages % 50 == 0:
            oldest_dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc).date()
            print_fn(f"    {symbol.split('-')[0]:6s} page {pages:4d} | "
                     f"oldest {oldest_dt} | {len(all_rows):,} candles", flush=True)

        after_ms_cursor = oldest_ts

        if oldest_ts <= cutoff_ms:
            break

        time.sleep(API_DELAY)

    if not all_rows:
        raise RuntimeError(f"No data for {symbol}")

    df        = _parse_candles_1m(all_rows)
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df        = df[df["datetime"] >= cutoff_dt]
    return (df.drop_duplicates("datetime")
              .sort_values("datetime")
              .reset_index(drop=True))


def _refresh_symbol_1m(symbol: str, cached: pd.DataFrame) -> pd.DataFrame:
    """Append candles newer than the last cached bar."""
    last_ts  = cached["datetime"].iloc[-1]
    since_ms = int(last_ts.timestamp() * 1000)
    all_rows, after_ms_cursor = [], None

    for _ in range(40):
        raw = _fetch_page_1m(symbol, after_ms=after_ms_cursor, use_history=False)
        if not raw:
            # fall back to history endpoint
            raw = _fetch_page_1m(symbol, after_ms=after_ms_cursor, use_history=True)
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

    new_df = _parse_candles_1m(all_rows)
    new_df = new_df[new_df["datetime"] > last_ts]
    if len(new_df) == 0:
        return cached

    combined = (pd.concat([cached, new_df], ignore_index=True)
                .drop_duplicates("datetime")
                .sort_values("datetime")
                .reset_index(drop=True))
    _save_cache_1m(combined, symbol)
    return combined


MIN_CACHE_CANDLES = 20_000  # require at least ~2 weeks of 1m bars before trusting cache

def get_data_1m(symbol: str) -> pd.DataFrame:
    """Load from cache or download, then refresh to present."""
    cached = _load_cache_1m(symbol)
    if cached is not None and len(cached) >= MIN_CACHE_CANDLES:
        last_ts = cached["datetime"].iloc[-1]
        gap_min = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 60
        if gap_min < 5:
            return cached
        print(f"  {symbol.split('-')[0]:6s} refreshing ({len(cached):,} cached, "
              f"gap={gap_min:.0f} min)...", flush=True)
        return _refresh_symbol_1m(symbol, cached)

    # Full download
    print(f"  {symbol.split('-')[0]:6s} full download...", flush=True)
    df = _download_symbol_1m(symbol)
    _save_cache_1m(df, symbol)
    span_days = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds() / 86400
    print(f"  {symbol.split('-')[0]:6s} {len(df):,} candles  "
          f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}, "
          f"{span_days:.0f} days)", flush=True)
    return df


def download_all_symbols_parallel() -> dict:
    """Download / refresh all symbols using a small thread pool."""
    results = {}
    errors  = {}
    lock    = __import__("threading").Lock()

    def _worker(sym):
        try:
            df = get_data_1m(sym)
            return sym, df, None
        except Exception as e:
            return sym, None, str(e)

    with ThreadPoolExecutor(max_workers=DL_WORKERS) as pool:
        futures = {pool.submit(_worker, s): s for s in SYMBOLS}
        for fut in as_completed(futures):
            sym, df, err = fut.result()
            if err:
                errors[sym] = err
                print(f"  [WARN] {sym} failed: {err}", flush=True)
            else:
                results[sym] = df

    return results


# =============================================================================
# SECTION 2 — INDICATORS (EMA20, EMA50, EMA200, ATR, ADX)
# =============================================================================

EMA_SLOPE_BARS = 10   # bars for EMA200 slope check

def add_r014_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
# SECTION 3 — PULLBACK SIGNAL DETECTION (state machine)
# =============================================================================

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Trend-Continuation-After-Pullback entries.
    Returns df with new columns:
      signal             : bool — entry signal on bar i (enter next open)
      pullback_swing_low : float — SL level (lowest low during pullback)
      pullback_depth_pct : float — max close below EMA20 / EMA20 during pullback
      trend_bars_before  : int   — bars trend conditions held before entry
      dist_ema50_pct     : float — (close - EMA50) / EMA50 at signal bar
    """
    df = df.copy()
    n  = len(df)

    close   = df["close"].values
    low     = df["low"].values
    ema20   = df["ema20"].values
    ema50   = df["ema50"].values
    ema200  = df["ema200"].values
    slope   = df["ema200_slope"].values.astype(bool)

    signal             = np.zeros(n, dtype=bool)
    pb_swing_low       = np.full(n, np.nan)
    pb_depth_pct       = np.full(n, np.nan)
    dist_ema50_pct_arr = np.full(n, np.nan)
    trend_bars_arr     = np.zeros(n, dtype=np.int32)

    in_pullback        = False
    pb_low             = np.nan
    pb_max_depth       = 0.0   # max (ema20 - close) / ema20 during pullback
    trend_count        = 0
    trend_running      = 0

    for i in range(n):
        # Warmup guard
        if np.isnan(ema200[i]) or np.isnan(ema50[i]) or np.isnan(ema20[i]):
            in_pullback  = False
            pb_low       = np.nan
            pb_max_depth = 0.0
            trend_count  = 0
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

        # Check pullback zone: close < EMA20 AND close > EMA50
        in_pb_zone = (close[i] < ema20[i]) and (close[i] > ema50[i])
        below_ema50 = close[i] <= ema50[i]

        if below_ema50:
            # Pullback violated — reset
            in_pullback   = False
            pb_low        = np.nan
            pb_max_depth  = 0.0

        elif in_pb_zone:
            in_pullback = True
            cur_low = low[i]
            if np.isnan(pb_low):
                pb_low = cur_low
            else:
                pb_low = min(pb_low, cur_low)
            depth = (ema20[i] - close[i]) / ema20[i]
            pb_max_depth = max(pb_max_depth, depth)

        elif in_pullback and close[i] > ema20[i]:
            # Pullback reclaim — bullish close above EMA20
            # Check it's a bullish candle (close > open implied by close > ema20 + was below)
            signal[i]           = True
            pb_swing_low[i]     = pb_low
            pb_depth_pct[i]     = pb_max_depth
            dist_ema50_pct_arr[i] = (close[i] - ema50[i]) / ema50[i]
            trend_bars_arr[i]   = trend_running
            # Reset state
            in_pullback   = False
            pb_low        = np.nan
            pb_max_depth  = 0.0
            # Don't reset trend_running — may re-enter

    df["signal"]             = signal
    df["pullback_swing_low"] = pb_swing_low
    df["pullback_depth_pct"] = pb_depth_pct
    df["dist_ema50_pct"]     = dist_ema50_pct_arr
    df["trend_bars_before"]  = trend_bars_arr
    return df


# =============================================================================
# SECTION 4 — CUSTOM BACKTEST ENGINE  (uses pullback_swing_low as SL)
# =============================================================================

def run_r014_backtest(df: pd.DataFrame, label: str) -> dict:
    """
    Event-driven simulator.  Identical costs to all previous research.
    SL = pullback_swing_low of signal bar (not prev bar low).
    Tracks MFE and MAE per trade.
    """
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

    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    open_arr  = df["open"].values
    dt_arr    = df["datetime"].values
    sig_arr   = df["signal"].values
    psl_arr   = df["pullback_swing_low"].values
    pd_arr    = df["pullback_depth_pct"].values
    de50_arr  = df["dist_ema50_pct"].values
    tb_arr    = df["trend_bars_before"].values
    atr_arr   = df["atr"].values
    adx_arr   = df["adx"].values
    ema200_arr= df["ema200"].values
    ema200_lag= df["ema200_lag"].values

    for i in range(1, n):
        hi = high_arr[i]
        lo = low_arr[i]

        if in_pos:
            # Track MFE / MAE
            pnl_hi = (hi - entry_price) / (entry_price - sl)  # in R
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
                hold_min  = i - entry_idx

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
                    # attribution (from entry bar i-1)
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

        # Check signal from previous bar
        if sig_arr[i - 1]:
            ep_raw = open_arr[i]
            sl_raw = psl_arr[i - 1]

            if np.isnan(sl_raw):
                continue

            sl_dist = ep_raw - sl_raw
            if sl_dist <= 0 or sl_dist / ep_raw < MIN_SL_PCT:
                continue

            tp_price = ep_raw + RR * sl_dist
            risk_dollars = capital * RISK_PCT
            size = min(risk_dollars / sl_dist, (capital * MAX_LEV) / ep_raw)

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
# SECTION 5 — PORTFOLIO AGGREGATION
# =============================================================================

def compute_portfolio_metrics(all_trades: list) -> dict:
    """Combine all individual symbol trades into one portfolio."""
    if not all_trades:
        return None
    df = pd.DataFrame(all_trades).sort_values("entry_time").reset_index(drop=True)
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
    max_dd = dd.min()

    std    = np.std(pnls, ddof=1) if n > 1 else 0.0
    bpy    = 365 * 24 * 60   # 1-minute bars per year
    sharpe = pnls.mean() / std * math.sqrt(bpy) if std > 0 else 0.0

    exp_r  = wr * RR - (1.0 - wr)

    return {
        "label":          "PORTFOLIO",
        "n_trades":       n,
        "net_profit":     float(pnls.sum()),
        "profit_factor":  pf,
        "win_rate":       wr,
        "expectancy_r":   exp_r,
        "avg_r":          float(rmul.mean()),
        "max_drawdown":   max_dd,
        "sharpe":         sharpe,
        "avg_hold_minutes": df["holding_minutes"].mean(),
        "equity":         equity,
        "drawdown":       dd,
        "pnls":           pnls,
        "r_multiples":    rmul,
        "trades_df":      df,
    }


# =============================================================================
# SECTION 6 — ATTRIBUTION ANALYSIS
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
        "win_avg_mfe_r":      _wmean("mfe_r"),
        "loss_avg_mfe_r":     _lmean("mfe_r"),
        "win_avg_mae_r":      _wmean("mae_r"),
        "loss_avg_mae_r":     _lmean("mae_r"),
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

def _ax(fig, gs_cell, bg=BG):
    ax = fig.add_subplot(gs_cell)
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="#AAA", labelsize=8)
    ax.xaxis.label.set_color("#AAA")
    ax.yaxis.label.set_color("#AAA")
    ax.title.set_color("#EEE")
    return ax


def plot_equity_curves(sym_metrics: dict, port_m: dict):
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
        pf_str = f"{m['profit_factor']:.2f}"
        ax.set_title(f"{short}  PF={pf_str}  n={m['n_trades']}", fontsize=9)
        ax.set_ylabel("Equity $", fontsize=7)

    # Portfolio
    ax_p = _ax(fig, gs[1, 3])
    if port_m:
        ax_p.plot(port_m["equity"], color="#FFD700", lw=1.5)
        ax_p.axhline(STARTING_CAP, color="#555", lw=0.5, ls="--")
        ax_p.set_title(f"PORTFOLIO  PF={port_m['profit_factor']:.2f}  "
                       f"n={port_m['n_trades']}", fontsize=9)
        ax_p.set_ylabel("Equity $", fontsize=7)

    fig.suptitle("R014 Equity Curves — Trend Continuation After Pullback",
                 color="#EEE", fontsize=13, y=1.01)
    path = f"{OUTPUT_FOLDER}/r014_equity_curves.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_drawdown_curves(sym_metrics: dict, port_m: dict):
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

    fig.suptitle("R014 Drawdown Curves", color="#EEE", fontsize=13, y=1.01)
    path = f"{OUTPUT_FOLDER}/r014_drawdown_curves.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_trade_distribution(port_m: dict):
    if port_m is None or port_m["n_trades"] == 0:
        return
    fig = plt.figure(figsize=(14, 5), facecolor=BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    rmul = port_m["r_multiples"]
    # R distribution
    ax1 = _ax(fig, gs[0])
    wins_r  = rmul[rmul > 0]
    loses_r = rmul[rmul <= 0]
    ax1.hist(loses_r, bins=40, color="#FF4560", alpha=0.7, label="Loss")
    ax1.hist(wins_r,  bins=40, color="#00C49A", alpha=0.7, label="Win")
    ax1.axvline(0, color="#FFF", lw=0.5)
    ax1.set_title("R-Multiple Distribution", fontsize=10)
    ax1.set_xlabel("R Multiple")
    ax1.legend(fontsize=8)

    # P&L histogram
    ax2 = _ax(fig, gs[1])
    pnls = port_m["pnls"]
    ax2.hist(pnls, bins=50, color="#4A90D9", alpha=0.8)
    ax2.axvline(0, color="#FFF", lw=0.8)
    ax2.set_title("P&L Distribution ($)", fontsize=10)
    ax2.set_xlabel("P&L per trade $")

    # Win/loss per symbol
    ax3 = _ax(fig, gs[2])
    td  = port_m["trades_df"]
    syms = [s.split("-")[0] for s in td["label"].values]
    td2  = td.copy()
    td2["sym"] = syms
    sym_wr = td2.groupby("sym")["win"].mean()
    sym_n  = td2.groupby("sym").size()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(sym_wr))]
    ax3.bar(sym_wr.index, sym_wr.values * 100, color=colors, alpha=0.85)
    ax3.axhline(50, color="#FFF", lw=0.5, ls="--")
    ax3.set_title("Win Rate by Symbol (%)", fontsize=10)
    ax3.set_xlabel("Symbol")
    ax3.set_ylabel("%")

    fig.suptitle("R014 Trade Distribution", color="#EEE", fontsize=12, y=1.02)
    path = f"{OUTPUT_FOLDER}/r014_trade_distribution.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_mfe_mae(port_m: dict):
    if port_m is None or port_m["n_trades"] == 0:
        return
    td   = port_m["trades_df"]
    if "mfe_r" not in td.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    ax1 = axes[0]
    ax1.set_facecolor(BG)
    wins  = td[td["win"]]
    loses = td[~td["win"]]
    ax1.scatter(wins["mae_r"],  wins["mfe_r"],  c="#00C49A", alpha=0.5, s=15, label="Win")
    ax1.scatter(loses["mae_r"], loses["mfe_r"], c="#FF4560", alpha=0.5, s=15, label="Loss")
    ax1.axhline(2.0,  color="#FFD700", lw=0.8, ls="--", label="2R TP")
    ax1.axvline(-1.0, color="#FF8C00", lw=0.8, ls="--", label="1R SL")
    ax1.set_xlabel("MAE (R)")
    ax1.set_ylabel("MFE (R)")
    ax1.set_title("MFE vs MAE scatter", fontsize=10, color="#EEE")
    ax1.legend(fontsize=8)
    for sp in ax1.spines.values():
        sp.set_edgecolor("#333")
    ax1.tick_params(colors="#AAA")

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.hist(td["mfe_r"].dropna(), bins=40, color="#4A90D9", alpha=0.8, label="MFE")
    ax2.hist(td["mae_r"].dropna(), bins=40, color="#FF4560", alpha=0.6, label="MAE")
    ax2.axvline(2.0, color="#FFD700", lw=0.8, ls="--", label="2R TP")
    ax2.set_xlabel("R")
    ax2.set_title("MFE / MAE Distribution", fontsize=10, color="#EEE")
    ax2.legend(fontsize=8)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333")
    ax2.tick_params(colors="#AAA")

    fig.suptitle("R014 MFE vs MAE Analysis", color="#EEE", fontsize=12)
    path = f"{OUTPUT_FOLDER}/r014_mfe_mae.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_holding_time_histogram(port_m: dict):
    if port_m is None or port_m["n_trades"] == 0:
        return
    td   = port_m["trades_df"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax in axes:
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.tick_params(colors="#AAA")

    wins  = td[td["win"]]["holding_minutes"]
    loses = td[~td["win"]]["holding_minutes"]

    axes[0].hist(wins,  bins=50, color="#00C49A", alpha=0.7, label="Win")
    axes[0].hist(loses, bins=50, color="#FF4560", alpha=0.7, label="Loss")
    axes[0].set_xlabel("Holding Time (minutes)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Holding Time: Wins vs Losses", color="#EEE", fontsize=10)
    axes[0].legend(fontsize=9)

    # Cumulative by holding time
    combined = td[["holding_minutes","win"]].copy()
    combined = combined.sort_values("holding_minutes")
    combined["cum_pnl"] = (combined["win"].astype(float) * RR - (1 - combined["win"].astype(float))).cumsum()
    axes[1].plot(combined["holding_minutes"].values, combined["cum_pnl"].values,
                 color="#FFD700", lw=1.5)
    axes[1].axhline(0, color="#555", lw=0.5)
    axes[1].set_xlabel("Holding Time (minutes)")
    axes[1].set_ylabel("Cumulative R")
    axes[1].set_title("Cumulative R vs Holding Time", color="#EEE", fontsize=10)

    fig.suptitle("R014 Holding-Time Analysis", color="#EEE", fontsize=12)
    path = f"{OUTPUT_FOLDER}/r014_holding_time.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_symbol_comparison(sym_metrics: dict):
    valid = {s: v for s, v in sym_metrics.items() if v["metrics"]["n_trades"] > 0}
    if not valid:
        return
    syms  = list(valid.keys())
    short = [s.split("-")[0] for s in syms]
    pfs   = [valid[s]["metrics"]["profit_factor"] for s in syms]
    wrs   = [valid[s]["metrics"]["win_rate"] * 100 for s in syms]
    mdds  = [abs(valid[s]["metrics"]["max_drawdown"]) * 100 for s in syms]
    ntrades = [valid[s]["metrics"]["n_trades"] for s in syms]
    net_p = [valid[s]["metrics"]["net_profit"] for s in syms]
    exp_r = [valid[s]["metrics"]["expectancy_r"] for s in syms]

    fig = plt.figure(figsize=(16, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
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

    _bar(_ax(fig, gs[0, 0]), pfs,     "Profit Factor",    "PF",    refline=1.2)
    _bar(_ax(fig, gs[0, 1]), wrs,     "Win Rate (%)",     "%",     refline=50)
    _bar(_ax(fig, gs[0, 2]), mdds,    "Max Drawdown (%)", "%",     refline=30)
    _bar(_ax(fig, gs[1, 0]), ntrades, "# Trades",         "n",     refline=MIN_OOS_TRADES)
    _bar(_ax(fig, gs[1, 1]), net_p,   "Net P&L ($)",      "$")
    _bar(_ax(fig, gs[1, 2]), exp_r,   "Expectancy R",     "R",     refline=0)

    fig.suptitle("R014 Symbol Comparison Dashboard", color="#EEE", fontsize=13, y=1.01)
    path = f"{OUTPUT_FOLDER}/r014_symbol_comparison.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# SECTION 8 — REPORT
# =============================================================================

def _fmt(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  —  "
    return format(v, fmt)


def print_report(sym_metrics: dict, port_m: dict, attr: dict,
                 oos_dates: dict):
    W = 110
    print()
    print("=" * W)
    print(f"  QUANTLAB AI — RESEARCH #014")
    print(f"  Trend Continuation After Pullback — 1-minute candles")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * W)
    print()
    print("  Entry  : close > EMA200 (rising) | EMA20 > EMA50 | pullback below EMA20 (above EMA50) | reclaim EMA20")
    print("  Stop   : pullback swing low")
    print("  Target : 2R")
    print()

    # OOS windows
    for sym, dates in oos_dates.items():
        short = sym.split("-")[0]
        print(f"  {short:6s}  OOS: {dates[0]} → {dates[1]}")
    print()

    # Per-symbol table
    header_syms = [s.split("-")[0] for s in sym_metrics.keys()]
    col_w = max(14, max(len(h) for h in header_syms) + 2)

    def _row(label, vals):
        print(f"  {label:<28s}" + "".join(f"{v:>{col_w}}" for v in vals))

    print("  " + "-" * (28 + col_w * len(header_syms)))
    _row("Metric", header_syms)
    print("  " + "-" * (28 + col_w * len(header_syms)))

    def _vals(fn):
        return [fn(sym_metrics[s]) for s in sym_metrics]

    _row("Trades",          [f"{v['metrics']['n_trades']:>{col_w}d}" for v in sym_metrics.values()])
    _row("Win Rate",        [f"{v['metrics']['win_rate']*100:>{col_w-1}.1f}%" for v in sym_metrics.values()])
    _row("Profit Factor",   [f"{v['metrics']['profit_factor']:>{col_w}.3f}" for v in sym_metrics.values()])
    _row("Expectancy R",    [f"{v['metrics']['expectancy_r']:>+{col_w}.3f}" for v in sym_metrics.values()])
    _row("Net P&L ($)",     [f"{v['metrics']['net_profit']:>{col_w}.0f}" for v in sym_metrics.values()])
    _row("Max Drawdown",    [f"{v['metrics']['max_drawdown']*100:>{col_w-1}.1f}%" for v in sym_metrics.values()])
    _row("Sharpe",          [f"{v['metrics']['sharpe']:>{col_w}.2f}" for v in sym_metrics.values()])
    _row("MC Prob Profit",  [f"{v['mc']['prob_profit']*100:>{col_w-1}.1f}%" for v in sym_metrics.values()])
    _row("Verdict",         [f"{v['verdict']:>{col_w}}" for v in sym_metrics.values()])
    print("  " + "-" * (28 + col_w * len(header_syms)))

    print()
    if port_m:
        print(f"  ── PORTFOLIO ────────────────────────────────────────────")
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
        print(f"  ── ATTRIBUTION (portfolio level) ────────────────────────")
        print(f"  Avg pullback depth    : {attr.get('avg_pb_depth_pct', 0)*100:.2f}%  (vs EMA20)")
        print(f"  Avg holding time      : {attr.get('avg_hold_min', 0):.0f} min")
        print(f"  Avg ATR at entry      : {attr.get('avg_atr_entry', 0):.6f}")
        print(f"  Avg ADX at entry      : {attr.get('avg_adx_entry', 0):.1f}")
        print(f"  Avg EMA200 slope      : {attr.get('avg_ema200_slope', 0)*100:.4f}%")
        print(f"  Avg dist EMA50        : {attr.get('avg_dist_ema50_pct', 0)*100:.2f}%")
        print(f"  Avg trend bars before : {attr.get('avg_trend_bars', 0):.0f}")
        print(f"  Avg MFE               : {attr.get('avg_mfe_r', 0):+.3f}R")
        print(f"  Avg MAE               : {attr.get('avg_mae_r', 0):+.3f}R")
        print(f"  Win avg hold          : {attr.get('win_avg_hold_min', 0):.0f} min")
        print(f"  Loss avg hold         : {attr.get('loss_avg_hold_min', 0):.0f} min")
        print(f"  Win avg ADX           : {attr.get('win_avg_adx', 0):.1f}")
        print(f"  Loss avg ADX          : {attr.get('loss_avg_adx', 0):.1f}")
        print()

    # Final verdict
    if port_m:
        pf   = port_m["profit_factor"]
        exp  = port_m["expectancy_r"]
        mdd  = abs(port_m["max_drawdown"])
        n    = port_m["n_trades"]
        n_sym_pass = sum(
            1 for v in sym_metrics.values()
            if v["metrics"]["profit_factor"] >= PROMOTE_PF
               and v["metrics"]["net_profit"] > 0
        )
        mc_port = monte_carlo(port_m["pnls"])
        mc_pp   = mc_port["prob_profit"]

        pass_pf    = pf > PROMOTE_PF
        pass_exp   = exp > 0
        pass_mdd   = mdd < PROMOTE_MDD
        pass_n     = n >= MIN_OOS_TRADES
        pass_syms  = n_sym_pass >= 2
        pass_mc    = mc_pp >= PROMOTE_MC_PP

        final_pass = all([pass_pf, pass_exp, pass_mdd, pass_n, pass_syms, pass_mc])
        verdict_str = "✅  PASS — SUITABLE FOR FORWARD DEMO TESTING" if final_pass else "❌  REJECT"

        print(f"  ── FINAL VERDICT ────────────────────────────────────────")
        print(f"  Combined PF > {PROMOTE_PF:.2f}    : {'✓' if pass_pf  else '✗'}  ({pf:.3f})")
        print(f"  Positive expectancy     : {'✓' if pass_exp else '✗'}  ({exp:+.3f}R)")
        print(f"  MDD < {PROMOTE_MDD*100:.0f}%           : {'✓' if pass_mdd else '✗'}  ({mdd*100:.1f}%)")
        print(f"  ≥ {MIN_OOS_TRADES} OOS trades       : {'✓' if pass_n   else '✗'}  ({n})")
        print(f"  ≥ 2 profitable symbols  : {'✓' if pass_syms else '✗'}  ({n_sym_pass}/{len(sym_metrics)})")
        print(f"  MC Prob Profit ≥ {PROMOTE_MC_PP*100:.0f}%  : {'✓' if pass_mc else '✗'}  ({mc_pp*100:.1f}%)")
        print()
        print(f"  {verdict_str}")
        print()

    print("=" * W)


# =============================================================================
# SECTION 9 — MAIN
# =============================================================================

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CACHE_FOLDER, exist_ok=True)

    print()
    print("╔" + "═" * 79 + "╗")
    print("║  QUANTLAB AI — RESEARCH #014" + " " * 50 + "║")
    print("║  Trend Continuation After Pullback" + " " * 44 + "║")
    print("╚" + "═" * 79 + "╝")
    print()
    print(f"  Hypothesis: Strong trends continue after shallow pullbacks.")
    print(f"  Entry: reclaim EMA20 after pullback (above EMA50), uptrend confirmed.")
    print(f"  SL: pullback swing low | TP: 2R | Timeframe: 1-minute")
    print()

    # ─── STEP 1: Download / refresh data ─────────────────────────────────────
    print("=" * 70)
    print("  STEP 1: Downloading 1-minute data (parallel, up to 4 symbols at once)")
    print("=" * 70)
    t0 = time.time()
    raw_data = download_all_symbols_parallel()
    elapsed  = time.time() - t0
    for sym, df in raw_data.items():
        print(f"  {sym:20s}  {len(df):>8,} candles  "
              f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})")
    print(f"  Download complete in {elapsed:.0f}s")
    print()

    # ─── STEP 2: Run strategy on each symbol ─────────────────────────────────
    print("=" * 70)
    print("  STEP 2: Computing indicators + running backtests")
    print("=" * 70)

    sym_metrics = {}
    all_trades  = []
    oos_dates   = {}
    skipped     = []

    for sym, df_raw in raw_data.items():
        short = sym.split("-")[0]

        # Add indicators
        df = add_r014_indicators(df_raw)
        df = df.dropna(subset=["ema200", "ema50", "ema20"]).reset_index(drop=True)

        if len(df) < 500:
            print(f"  {short:6s}  SKIP — insufficient data ({len(df)} bars after warmup)")
            skipped.append(sym)
            continue

        # Train / OOS split
        split_idx = int(len(df) * TRAIN_RATIO)
        df_oos    = df.iloc[split_idx:].reset_index(drop=True)
        oos_dates[sym] = (
            str(df_oos["datetime"].iloc[0].date()),
            str(df_oos["datetime"].iloc[-1].date()),
        )

        if len(df_oos) < 200:
            print(f"  {short:6s}  SKIP — OOS too short ({len(df_oos)} bars)")
            skipped.append(sym)
            continue

        # Compute signals on OOS slice only
        df_oos = compute_signals(df_oos)
        sig_count = df_oos["signal"].sum()

        # Backtest
        bt    = run_r014_backtest(df_oos, label=sym)
        m     = compute_metrics(bt["trades"], sym)
        mc    = monte_carlo(m["pnls"])
        v     = _verdict_from_metrics(m, mc)

        sym_metrics[sym] = {"metrics": m, "mc": mc, "verdict": v}
        all_trades.extend(bt["trades"])

        print(f"  {short:6s}  signals={sig_count:4d}  "
              f"trades={m['n_trades']:4d}  "
              f"WR={m['win_rate']*100:5.1f}%  "
              f"PF={m['profit_factor']:.3f}  "
              f"[{v}]")

    if not sym_metrics:
        print("  ERROR: No symbols produced results. Check data / API.")
        return

    # ─── STEP 3: Portfolio ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  STEP 3: Portfolio aggregation")
    print("=" * 70)
    port_m = compute_portfolio_metrics(all_trades) if all_trades else None
    if port_m:
        print(f"  Portfolio  trades={port_m['n_trades']}  "
              f"WR={port_m['win_rate']*100:.1f}%  "
              f"PF={port_m['profit_factor']:.3f}  "
              f"MDD={port_m['max_drawdown']*100:.1f}%")

    # Attribution
    attr = {}
    if all_trades:
        td_all = pd.DataFrame(all_trades)
        attr   = attribution_analysis(td_all)

    # ─── STEP 4: Charts ───────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  STEP 4: Generating charts")
    print("=" * 70)
    plot_equity_curves(sym_metrics, port_m)
    plot_drawdown_curves(sym_metrics, port_m)
    plot_trade_distribution(port_m)
    plot_mfe_mae(port_m)
    plot_holding_time_histogram(port_m)
    plot_symbol_comparison(sym_metrics)

    # ─── STEP 5: Report ───────────────────────────────────────────────────────
    print_report(sym_metrics, port_m, attr, oos_dates)

    # ─── STEP 6: Journal ─────────────────────────────────────────────────────
    print("=" * 70)
    print("  STEP 6: Writing journal")
    print("=" * 70)
    journal_rows = []
    for sym, data in sym_metrics.items():
        m  = data["metrics"]
        mc = data["mc"]
        v  = data["verdict"]
        journal_rows.append(_journal_row(
            strategy_name = f"TrendPullback_{sym.split('-')[0]}",
            symbol        = sym,
            m             = m,
            mc            = mc,
            verdict       = v,
        ))
    append_journal(journal_rows)
    print(f"  Journal updated → {OUTPUT_FOLDER}/research_journal.csv")
    print()
    print(f"  All outputs → {OUTPUT_FOLDER}/r014_*")
    print(f"  Research #014 complete.")
    print()


if __name__ == "__main__":
    main()
