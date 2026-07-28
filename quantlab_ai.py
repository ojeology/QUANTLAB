"""
=============================================================================
QUANTLAB AI – RESEARCH MVP v1.0
Institutional-Style Strategy Hypothesis Tester

Purpose:
  Objectively determine whether a trading hypothesis demonstrates a
  measurable edge against a control benchmark.

Truth is more important than profitability.
Evidence is more important than opinion.

Note on timeframe:
  Default is 1H (hourly candles). 1-minute candles are available but
  produce 0 FVG signals on 24/7 perpetual futures because continuous
  crypto markets have no true between-candle price gaps — a valid and
  important finding in itself. Hourly candles exhibit real FVG patterns
  during impulse moves and produce sensible stop distances (~0.3–1% of
  price vs. ~0.008% on 1m).
=============================================================================
"""

import os
import time
import math
import random
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# =============================================================================
# CONFIGURATION — Edit these settings without touching strategy logic
# =============================================================================

CONFIG = {
    # Instruments to test (OKX perpetual futures)
    "SYMBOLS": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],

    # Candle timeframe for strategy execution
    # Use "1H" for meaningful FVG signals on 24/7 crypto perpetuals.
    # "1m" produces 0 FVG trades — continuous markets have no session gaps.
    "TIMEFRAME": "1H",

    # Target download months (history depth)
    "MONTHS_HISTORY": 9,

    # Strategy parameters
    "EMA_LENGTH": 200,
    "RISK_REWARD": 2.0,

    # FVG gap multiplier: candle[i].low > candle[i-2].high × FVG_MULT
    # Uses the proper 3-candle FVG definition (gap between candle i-2 and i)
    "FVG_MULT": 1.0001,

    # Execution costs (expressed as decimals, per side)
    "TAKER_FEE":    0.0005,   # 0.05%
    "SPREAD":       0.0002,   # 0.02%
    "SL_SLIPPAGE":  0.0003,   # 0.03% fixed stop slippage

    # Minimum SL distance as % of entry price (filter degenerate signals)
    "MIN_SL_PCT": 0.001,      # 0.1% — skips signals with implausibly tight stops

    # Maximum leverage multiplier (caps position size relative to capital)
    # Prevents runaway losses from abnormally tight stops.
    "MAX_LEVERAGE": 5.0,

    # Capital model
    "STARTING_CAPITAL":    10_000.0,
    "RISK_PER_TRADE_PCT":  0.01,      # 1% of capital risked per trade

    # Train / Out-of-Sample split
    "TRAIN_RATIO": 0.70,

    # Monte Carlo
    "MC_ITERATIONS": 1000,

    # Output paths
    "CACHE_FOLDER":  "quantlab_cache",
    "OUTPUT_FOLDER": "quantlab_output",

    # API rate-limiting: seconds between requests
    "API_DELAY": 0.2,

    # Max candles per OKX API page
    "OKX_PAGE_LIMIT": 100,
}


# =============================================================================
# SECTION 1 — DATA DOWNLOAD (OKX Public REST API)
# =============================================================================

OKX_HISTORY_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"

CANDLE_COLS = ["ts", "open", "high", "low", "close", "vol",
               "volCcy", "volCcyQuote", "confirm"]


def _parse_candles(raw: list) -> pd.DataFrame:
    """Convert raw OKX candle list to a clean DataFrame."""
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = pd.to_numeric(df[col])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[["datetime", "open", "high", "low", "close", "vol"]].copy()
    return df.sort_values("datetime").reset_index(drop=True)


def _fetch_page(symbol: str, bar: str,
                after_ms: int = None, before_ms: int = None,
                use_history: bool = True) -> list:
    """Fetch one page of candles from OKX. Returns list of raw rows."""
    url = OKX_HISTORY_URL if use_history else OKX_CANDLES_URL
    params = {"instId": symbol, "bar": bar, "limit": CONFIG["OKX_PAGE_LIMIT"]}
    if after_ms is not None:
        params["after"] = str(after_ms)
    if before_ms is not None:
        params["before"] = str(before_ms)
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get("code") == "0":
            return data.get("data", [])
    except Exception as e:
        print(f"  [WARN] API error: {e}")
    return []


def download_symbol(symbol: str, bar: str, months: int,
                    since_ms: int = None) -> pd.DataFrame:
    """
    Download candle history for a symbol by paginating backwards from now.
    If since_ms is provided, stops paging when reaching that timestamp.
    Returns a DataFrame sorted ascending.
    """
    now_ms = int(time.time() * 1000)
    target_ms = int(months * 30.44 * 24 * 3600 * 1000)
    cutoff_ms = since_ms if since_ms else now_ms - target_ms

    if since_ms:
        print(f"  Fetching new candles for {symbol} since "
              f"{datetime.fromtimestamp(since_ms/1000, tz=timezone.utc).date()}...")
    else:
        print(f"  Downloading {symbol} ({bar}) — target {months} months...")

    all_rows = []
    after_ms_cursor = None   # OKX: 'after' means fetch candles with ts < after
    pages = 0

    while True:
        # Try history endpoint first (goes further back); fall back to candles
        raw = _fetch_page(symbol, bar, after_ms=after_ms_cursor, use_history=True)
        if not raw and pages == 0:
            raw = _fetch_page(symbol, bar, use_history=False)
        if not raw:
            break

        all_rows.extend(raw)
        pages += 1

        oldest_ts = int(raw[-1][0])
        newest_ts = int(raw[0][0])
        after_ms_cursor = oldest_ts  # next page: candles older than this

        if not since_ms:
            pct = max(0.0, 100.0 * (1.0 - (oldest_ts - cutoff_ms) / target_ms))
            print(f"    Page {pages:4d} | oldest "
                  f"{datetime.fromtimestamp(oldest_ts/1000, tz=timezone.utc).date()} "
                  f"| {pct:.0f}%", end="\r")

        # Stop once we've gone past the cutoff
        if oldest_ts <= cutoff_ms:
            break

        time.sleep(CONFIG["API_DELAY"])

    if not since_ms:
        print()  # newline after \r

    if not all_rows:
        raise RuntimeError(f"No data received for {symbol}")

    df = _parse_candles(all_rows)
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df = df[df["datetime"] >= cutoff_dt]
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    return df


# =============================================================================
# SECTION 2 — LOCAL CACHE (Parquet with CSV fallback)
# =============================================================================

def _cache_path(symbol: str, bar: str) -> str:
    safe = symbol.replace("-", "_") + "_" + bar.replace("/", "")
    folder = CONFIG["CACHE_FOLDER"]
    os.makedirs(folder, exist_ok=True)
    try:
        import pyarrow  # noqa
        return os.path.join(folder, f"{safe}.parquet")
    except ImportError:
        return os.path.join(folder, f"{safe}.csv")


def _is_parquet(path: str) -> bool:
    return path.endswith(".parquet")


def save_cache(df: pd.DataFrame, symbol: str, bar: str) -> None:
    path = _cache_path(symbol, bar)
    if _is_parquet(path):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_cache(symbol: str, bar: str) -> pd.DataFrame | None:
    path = _cache_path(symbol, bar)
    if not os.path.exists(path):
        return None
    if _is_parquet(path):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def get_data(symbol: str) -> pd.DataFrame:
    """
    Load cached data and append any missing candles since last cache write.
    Falls back to a full download when no cache exists.
    """
    bar = CONFIG["TIMEFRAME"]
    print(f"\n[DATA] {symbol} ({bar})")
    cached = load_cache(symbol, bar)

    if cached is not None and len(cached) > 0:
        last_ts = cached["datetime"].iloc[-1]
        now_utc = pd.Timestamp.now(tz="UTC")

        # Determine expected candle interval in minutes
        bar_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1H": 60, "2H": 120, "4H": 240, "6H": 360, "12H": 720,
            "1D": 1440,
        }.get(bar, 60)

        gap_candles = (now_utc - last_ts).total_seconds() / 60 / bar_minutes

        if gap_candles < 2:
            print(f"  Cache is current ({len(cached):,} candles). Skipping download.")
            return cached

        print(f"  Cache found ({len(cached):,} candles, last: {last_ts.date()}). "
              f"Fetching ~{gap_candles:.0f} missing candles...")

        since_ms = int(last_ts.timestamp() * 1000)
        new_df = download_symbol(symbol, bar, months=0, since_ms=since_ms)

        if len(new_df) > 0:
            new_df = new_df[new_df["datetime"] > last_ts]

        if len(new_df) > 0:
            combined = pd.concat([cached, new_df], ignore_index=True)
            combined = (combined.drop_duplicates("datetime")
                        .sort_values("datetime")
                        .reset_index(drop=True))
            save_cache(combined, symbol, bar)
            print(f"  Appended {len(new_df):,} new candles → {len(combined):,} total.")
            return combined

        print(f"  No new candles. Using cached data.")
        return cached

    # No cache: full download
    df = download_symbol(symbol, bar, months=CONFIG["MONTHS_HISTORY"])
    save_cache(df, symbol, bar)
    print(f"  → {len(df):,} candles "
          f"({df['datetime'].iloc[0].date()} – {df['datetime'].iloc[-1].date()})")
    return df


# =============================================================================
# SECTION 3 — INDICATOR CALCULATIONS
# =============================================================================

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all required indicators. Preserves original column order."""
    df = df.copy()
    df["ema200"]     = calc_ema(df["close"], CONFIG["EMA_LENGTH"])
    # Lagged columns for signal detection
    df["prev_high"]  = df["high"].shift(1)
    df["prev_low"]   = df["low"].shift(1)
    df["prev_close"] = df["close"].shift(1)
    # Two bars back (used in 3-candle FVG definition)
    df["high_2"]     = df["high"].shift(2)
    df["ema200_prev"]= df["ema200"].shift(1)
    return df


# =============================================================================
# SECTION 4 — STRATEGY FUNCTIONS
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  SWAP THIS BLOCK TO TEST A NEW HYPOTHESIS.                             │
# │  The backtest engine, metrics, Monte Carlo, and report are unchanged.  │
# └─────────────────────────────────────────────────────────────────────────┘
# =============================================================================

def strategy_fvg_ema(df: pd.DataFrame) -> pd.Series:
    """
    STRATEGY UNDER TEST: Bullish Fair Value Gap + EMA200

    Uses the authentic 3-candle FVG definition:
      • Candle i-2: the pre-impulse candle (its HIGH defines the gap bottom)
      • Candle i-1: the impulse candle (creates the gap)
      • Candle i  : current candle (its LOW must be above candle i-2 HIGH)

    Signal conditions (all must be true on bar close):
      1. Bullish 3-candle FVG: close[i].low > high[i-2] × FVG_MULT
      2. Trend filter: close[i] > EMA200[i]

    Entry: next candle open.
    Stop:  low of current bar (candle i).
    TP:    entry + 2 × stop_distance.

    ─── REPLACE THIS FUNCTION TO TEST A NEW HYPOTHESIS ───
    """
    fvg = df["low"] > df["high_2"] * CONFIG["FVG_MULT"]
    trend = df["close"] > df["ema200"]
    return fvg & trend


def strategy_ema_only(df: pd.DataFrame) -> pd.Series:
    """
    CONTROL / BENCHMARK: EMA200 Bullish Crossover

    Signal: price closes above EMA200 after being below it the prior bar.
    Same execution, stops, costs, and RR as the FVG strategy.

    Isolates whether FVG adds edge beyond the trend filter alone.
    """
    cross_up = (df["close"] > df["ema200"]) & (df["prev_close"] <= df["ema200_prev"])
    return cross_up


# =============================================================================
# SECTION 5 — BACKTEST ENGINE (Event-Driven Simulator)
# =============================================================================

def run_backtest(df: pd.DataFrame, signal_fn, label: str) -> dict:
    """
    Event-driven position simulator.

    Execution rules:
      • Signal on bar i  →  entry at OPEN of bar i+1  (no look-ahead)
      • Stop Loss  = low of the signal bar
      • Take Profit = entry + RR × stop_distance
      • If both SL and TP fall inside the same bar range → SL assumed hit first
      • One open position at a time; new signals ignored while in position

    Costs (per trade):
      • Taker fee × 2 (entry + exit)
      • Half-spread × 2 (entry + exit)
      • SL slippage on stop exits only

    Position sizing:
      • Risk per trade = STARTING_CAPITAL × RISK_PER_TRADE_PCT
      • position_size  = risk_dollars / sl_dist  (units of price)
      • Capped at MAX_LEVERAGE × STARTING_CAPITAL / entry_price
    """
    signals   = signal_fn(df)
    min_sl    = CONFIG["MIN_SL_PCT"]
    rr        = CONFIG["RISK_REWARD"]
    max_lev   = CONFIG["MAX_LEVERAGE"]
    capital   = CONFIG["STARTING_CAPITAL"]
    risk_frac = CONFIG["RISK_PER_TRADE_PCT"]
    fee_rate  = CONFIG["TAKER_FEE"]
    spd_rate  = CONFIG["SPREAD"] * 0.5   # half spread per side
    slp_rate  = CONFIG["SL_SLIPPAGE"]

    in_position  = False
    entry_price  = 0.0
    stop_loss    = 0.0
    take_profit  = 0.0
    entry_time   = None
    entry_idx    = -1
    position_size = 0.0

    trades = []

    for i in range(1, len(df)):
        bar = df.iloc[i]

        # ── Manage open position ──────────────────────────────────────────
        if in_position:
            hi, lo = bar["high"], bar["low"]
            sl_hit = lo <= stop_loss
            tp_hit = hi >= take_profit

            if sl_hit or tp_hit:
                if sl_hit:                       # conservative: SL first
                    exit_price = stop_loss * (1.0 - slp_rate)
                    exit_type  = "SL"
                else:
                    exit_price = take_profit
                    exit_type  = "TP"

                # PnL
                gross_pnl = (exit_price - entry_price) * position_size

                # Costs (absolute dollars)
                notional_entry = entry_price * position_size
                notional_exit  = exit_price  * position_size
                cost_fee    = (notional_entry + notional_exit) * fee_rate
                cost_spread = (notional_entry + notional_exit) * spd_rate
                cost_slip   = (stop_loss - exit_price) * position_size if exit_type == "SL" else 0.0
                total_cost  = cost_fee + cost_spread + cost_slip

                net_pnl     = gross_pnl - total_cost
                sl_dist     = entry_price - stop_loss
                r_multiple  = (exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0

                holding_bars   = i - entry_idx
                holding_minutes = holding_bars * _bar_minutes()
                funding_windows = int(holding_minutes / 480)

                trades.append({
                    "label": label,
                    "entry_time": entry_time,
                    "exit_time":  bar["datetime"],
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "stop_loss":   stop_loss,
                    "take_profit": take_profit,
                    "pnl":         net_pnl,
                    "r_multiple":  r_multiple,
                    "fees":        cost_fee,
                    "spread_cost": cost_spread,
                    "sl_slippage": cost_slip,
                    "holding_minutes": holding_minutes,
                    "funding_windows_crossed": funding_windows,
                    "win":      exit_type == "TP",
                    "exit_type": exit_type,
                })
                in_position = False
            continue

        # ── Check for new entry signal ────────────────────────────────────
        if signals.iloc[i - 1]:
            prev_bar = df.iloc[i - 1]
            ep = bar["open"]
            sl = prev_bar["low"]
            sl_dist = ep - sl

            # Filter: skip degenerate signals
            if sl_dist <= 0:
                continue
            if sl_dist / ep < min_sl:
                continue

            tp = ep + rr * sl_dist

            # Position size with leverage cap
            risk_dollars   = capital * risk_frac
            raw_size       = risk_dollars / sl_dist
            max_size       = (capital * max_lev) / ep
            pos_size       = min(raw_size, max_size)

            entry_price    = ep
            stop_loss      = sl
            take_profit    = tp
            position_size  = pos_size
            entry_time     = bar["datetime"]
            entry_idx      = i
            in_position    = True

    return {"trades": trades}


def _bar_minutes() -> float:
    """Minutes per candle based on configured timeframe."""
    return {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1H": 60, "2H": 120, "4H": 240, "6H": 360, "12H": 720, "1D": 1440,
    }.get(CONFIG["TIMEFRAME"], 60)


# =============================================================================
# SECTION 6 — PERFORMANCE METRICS
# =============================================================================

def compute_metrics(trades: list, label: str) -> dict:
    """Full performance metric suite for a trade list."""
    if not trades:
        return _empty_metrics(label)

    df   = pd.DataFrame(trades)
    pnls = df["pnl"].values
    wins = df["win"].values.astype(bool)
    rmul = df["r_multiple"].values

    n     = len(pnls)
    n_win = int(wins.sum())
    n_los = n - n_win

    gross_wins  = pnls[wins].sum()   if n_win else 0.0
    gross_loss  = abs(pnls[~wins].sum()) if n_los else 1e-9
    profit_factor = gross_wins / gross_loss if gross_loss > 0 else float("inf")

    win_rate  = n_win / n
    avg_win   = pnls[wins].mean()   if n_win else 0.0
    avg_loss  = pnls[~wins].mean()  if n_los else 0.0
    avg_trade = pnls.mean()
    avg_r     = rmul.mean()

    # Expectancy in R units
    expectancy_r = (win_rate * CONFIG["RISK_REWARD"]) - ((1.0 - win_rate) * 1.0)

    largest_win  = pnls[wins].max()  if n_win else 0.0
    largest_loss = pnls[~wins].min() if n_los else 0.0

    # Equity curve (fixed starting capital, no compounding)
    equity = CONFIG["STARTING_CAPITAL"] + np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    max_dd = dd.min()

    # Approximate Sharpe (annualised, trade-basis)
    trade_std = np.std(pnls, ddof=1) if n > 1 else 0.0
    bars_per_year = (365 * 24 * 60) / _bar_minutes()
    trades_per_bar = n / max(len(pnls), 1)
    sharpe = (avg_trade / trade_std * math.sqrt(bars_per_year * trades_per_bar)
              if trade_std > 0 else 0.0)

    avg_hold = df["holding_minutes"].mean()
    total_funding = int(df["funding_windows_crossed"].sum())
    net_profit = float(pnls.sum())

    return {
        "label":           label,
        "n_trades":        n,
        "net_profit":      net_profit,
        "profit_factor":   profit_factor,
        "win_rate":        win_rate,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "avg_trade":       avg_trade,
        "avg_r":           avg_r,
        "expectancy_r":    expectancy_r,
        "largest_win":     largest_win,
        "largest_loss":    largest_loss,
        "max_drawdown":    max_dd,
        "sharpe":          sharpe,
        "avg_hold_minutes":      avg_hold,
        "total_funding_windows": total_funding,
        "equity":     equity,
        "drawdown":   dd,
        "pnls":       pnls,
        "r_multiples": rmul,
        "trades_df":  df,
    }


def _empty_metrics(label: str) -> dict:
    return {
        "label": label, "n_trades": 0, "net_profit": 0.0,
        "profit_factor": 0.0, "win_rate": 0.0,
        "avg_win": 0.0, "avg_loss": 0.0, "avg_trade": 0.0,
        "avg_r": 0.0, "expectancy_r": 0.0,
        "largest_win": 0.0, "largest_loss": 0.0,
        "max_drawdown": 0.0, "sharpe": 0.0,
        "avg_hold_minutes": 0.0, "total_funding_windows": 0,
        "equity":     np.array([CONFIG["STARTING_CAPITAL"]]),
        "drawdown":   np.array([0.0]),
        "pnls":       np.array([]),
        "r_multiples": np.array([]),
        "trades_df":  pd.DataFrame(),
    }


# =============================================================================
# SECTION 7 — MONTE CARLO ROBUSTNESS TEST
# =============================================================================

def monte_carlo(pnls: np.ndarray, n_iter: int = 1000) -> dict:
    """
    Randomly reshuffle the trade return sequence n_iter times.
    Reports the distribution of final equity outcomes.
    """
    if len(pnls) == 0:
        return {"median": 0.0, "p5": 0.0, "p95": 0.0,
                "prob_profit": 0.0, "final_equities": np.array([])}

    start = CONFIG["STARTING_CAPITAL"]
    finals = np.empty(n_iter)
    for k in range(n_iter):
        shuffled = np.random.permutation(pnls)
        finals[k] = start + shuffled.sum()

    return {
        "median":       float(np.median(finals)),
        "p5":           float(np.percentile(finals, 5)),
        "p95":          float(np.percentile(finals, 95)),
        "prob_profit":  float((finals > start).mean()),
        "final_equities": finals,
    }


# =============================================================================
# SECTION 8 — VISUALISATIONS
# =============================================================================

def plot_results(m_fvg: dict, m_bench: dict, mc_fvg: dict,
                 symbol: str, oos_start: str, oos_end: str) -> list:
    """Generate and save all charts. Returns list of saved file paths."""
    os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)
    safe = symbol.replace("-", "_")
    saved = []

    BG = "#0F1117"
    G1 = "#00C49A"
    G2 = "#4A90D9"
    RD = "#FF4560"
    OR = "#FF9500"

    def _style(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors="white", labelsize=8)
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
        ax.grid(True, alpha=0.2, color="#444")

    # ── Chart 1: Equity / Drawdown / Distribution ─────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"QuantLab AI — {symbol}  |  OOS: {oos_start} → {oos_end}",
                 fontsize=12, fontweight="bold", color="white", y=0.99)

    ax1, ax2, ax3 = axes
    for ax in axes:
        _style(ax)

    # Equity
    if len(m_fvg["equity"]) > 1:
        ax1.plot(m_fvg["equity"], color=G1, lw=1.5,
                 label=f"FVG + EMA200  (PF {m_fvg['profit_factor']:.2f})")
    if len(m_bench["equity"]) > 1:
        ax1.plot(m_bench["equity"], color=G2, lw=1.5, ls="--",
                 label=f"EMA200 Only  (PF {m_bench['profit_factor']:.2f})")
    ax1.axhline(CONFIG["STARTING_CAPITAL"], color="gray", lw=0.7, ls=":")
    ax1.set_title("Equity Curve", fontsize=10)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444",
               labelcolor="white")

    # Drawdown
    if len(m_fvg["drawdown"]) > 1:
        ax2.fill_between(range(len(m_fvg["drawdown"])),
                         m_fvg["drawdown"] * 100, 0,
                         color=RD, alpha=0.7, label="FVG + EMA200")
    if len(m_bench["drawdown"]) > 1:
        ax2.fill_between(range(len(m_bench["drawdown"])),
                         m_bench["drawdown"] * 100, 0,
                         color=OR, alpha=0.4, label="EMA200 Only")
    ax2.set_title("Drawdown (%)", fontsize=10)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444",
               labelcolor="white")

    # R-multiple distribution
    if len(m_fvg["r_multiples"]) > 0:
        ax3.hist(m_fvg["r_multiples"], bins=40, color=G1,
                 alpha=0.75, edgecolor="#000", label="FVG + EMA200")
    if len(m_bench["r_multiples"]) > 0:
        ax3.hist(m_bench["r_multiples"], bins=40, color=G2,
                 alpha=0.5, edgecolor="#000", label="EMA200 Only")
    ax3.axvline(0, color="white", lw=0.8, ls="--")
    ax3.set_title("Trade R-Multiple Distribution", fontsize=10)
    ax3.set_xlabel("R Multiple")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444",
               labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = os.path.join(CONFIG["OUTPUT_FOLDER"],
                      f"{safe}_equity_drawdown_distribution.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    saved.append(p1)

    # ── Chart 2: Monte Carlo distribution ────────────────────────────────
    if len(mc_fvg["final_equities"]) > 0:
        fig2, ax = plt.subplots(figsize=(12, 5))
        fig2.patch.set_facecolor(BG)
        _style(ax)
        fe = mc_fvg["final_equities"]
        n_unique = len(np.unique(fe))
        mc_bins = max(1, min(60, n_unique - 1)) if n_unique > 1 else 1
        ax.hist(fe, bins=mc_bins, color=G1, alpha=0.75, edgecolor="#000")
        ax.axvline(CONFIG["STARTING_CAPITAL"], color="white", lw=1.5,
                   ls="--", label="Starting Capital")
        ax.axvline(mc_fvg["p5"],    color=RD,       lw=1.5, ls=":",
                   label=f"5th pct  ${mc_fvg['p5']:,.0f}")
        ax.axvline(mc_fvg["median"], color="#FFD700", lw=1.5,
                   label=f"Median   ${mc_fvg['median']:,.0f}")
        ax.axvline(mc_fvg["p95"],   color="#00D4FF", lw=1.5, ls=":",
                   label=f"95th pct ${mc_fvg['p95']:,.0f}")
        ax.set_title(
            f"Monte Carlo Final Equity — {symbol}\n"
            f"{CONFIG['MC_ITERATIONS']:,} iterations  |  "
            f"Prob. Profitable: {mc_fvg['prob_profit']:.1%}",
            fontsize=11, color="white",
        )
        ax.set_xlabel("Final Portfolio Value ($)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9, facecolor="#1A1D24", edgecolor="#444",
                  labelcolor="white")
        plt.tight_layout()
        p2 = os.path.join(CONFIG["OUTPUT_FOLDER"], f"{safe}_monte_carlo.png")
        fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
        saved.append(p2)

    return saved


# =============================================================================
# SECTION 9 — TRADE LOG EXPORT
# =============================================================================

def save_trade_log(fvg_trades: list, bench_trades: list, symbol: str) -> str:
    all_trades = fvg_trades + bench_trades
    if not all_trades:
        return ""
    safe = symbol.replace("-", "_")
    df = pd.DataFrame(all_trades)
    cols = ["label", "entry_time", "exit_time", "entry_price", "exit_price",
            "stop_loss", "take_profit", "pnl", "r_multiple", "fees",
            "spread_cost", "sl_slippage", "holding_minutes",
            "funding_windows_crossed", "win", "exit_type"]
    df = df[[c for c in cols if c in df.columns]]
    os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)
    path = os.path.join(CONFIG["OUTPUT_FOLDER"], f"{safe}_trade_log.csv")
    df.to_csv(path, index=False)
    return path


# =============================================================================
# SECTION 10 — REPORT GENERATION
# =============================================================================

def _verdict(m_fvg: dict, m_bench: dict, mc: dict) -> tuple[str, list[str]]:
    pros, cons = [], []

    if m_fvg["profit_factor"] >= 1.2:
        pros.append(f"Profit Factor ≥ 1.2 ({m_fvg['profit_factor']:.2f})")
    else:
        cons.append(f"Profit Factor below 1.2 ({m_fvg['profit_factor']:.2f})")

    if m_fvg["expectancy_r"] > 0:
        pros.append(f"Positive expectancy ({m_fvg['expectancy_r']:+.2f}R)")
    else:
        cons.append(f"Negative expectancy ({m_fvg['expectancy_r']:+.2f}R)")

    if m_fvg["profit_factor"] > m_bench["profit_factor"]:
        pros.append(f"Outperformed benchmark (PF {m_fvg['profit_factor']:.2f} "
                    f"vs {m_bench['profit_factor']:.2f})")
    else:
        cons.append(f"Underperformed benchmark (PF {m_fvg['profit_factor']:.2f} "
                    f"vs {m_bench['profit_factor']:.2f})")

    if mc["prob_profit"] >= 0.60:
        pros.append(f"Monte Carlo prob. of profit: {mc['prob_profit']:.1%}")
    else:
        cons.append(f"Weak Monte Carlo robustness ({mc['prob_profit']:.1%})")

    if m_fvg["max_drawdown"] > -0.30:
        pros.append(f"Drawdown within acceptable range "
                    f"({m_fvg['max_drawdown']:.1%})")
    else:
        cons.append(f"Excessive drawdown ({m_fvg['max_drawdown']:.1%})")

    verdict = "PROMOTE" if len(pros) >= len(cons) else "REJECT"
    return verdict, (pros if verdict == "PROMOTE" else cons)


def _metrics_block(m: dict) -> None:
    print(f"  Strategy         : {m['label']}")
    print(f"  Trades           : {m['n_trades']}")
    print(f"  Net Profit       : ${m['net_profit']:>12,.2f}")
    print(f"  Profit Factor    : {m['profit_factor']:>12.3f}")
    print(f"  Win Rate         : {m['win_rate']:>12.1%}")
    print(f"  Avg Win          : ${m['avg_win']:>12,.2f}")
    print(f"  Avg Loss         : ${m['avg_loss']:>12,.2f}")
    print(f"  Avg Trade        : ${m['avg_trade']:>12,.2f}")
    print(f"  Avg R            : {m['avg_r']:>12.3f}")
    print(f"  Expectancy       : {m['expectancy_r']:>+12.3f}R")
    print(f"  Largest Win      : ${m['largest_win']:>12,.2f}")
    print(f"  Largest Loss     : ${m['largest_loss']:>12,.2f}")
    print(f"  Max Drawdown     : {m['max_drawdown']:>12.2%}")
    print(f"  Sharpe (approx)  : {m['sharpe']:>12.3f}")
    print(f"  Avg Hold Time    : {m['avg_hold_minutes']:>10.0f} min")
    print(f"  Funding Windows  : {m['total_funding_windows']:>12,}")


def print_report(symbol: str, m_fvg: dict, m_bench: dict,
                 mc: dict, oos_start: str, oos_end: str,
                 n_days: int, chart_paths: list) -> None:
    verdict, reasons = _verdict(m_fvg, m_bench, mc)
    stars = "★★★★★" if verdict == "PROMOTE" else "★☆☆☆☆"
    S = "=" * 67

    print(f"\n{S}")
    print("  QUANTLAB AI RESEARCH REPORT")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(S)
    print(f"  Symbol           : {symbol}")
    print(f"  Strategy         : Bullish FVG (3-candle) + EMA{CONFIG['EMA_LENGTH']}")
    print(f"  Benchmark        : EMA{CONFIG['EMA_LENGTH']} Cross Only")
    print(f"  Timeframe        : {CONFIG['TIMEFRAME']}")
    print(f"  EMA Length       : {CONFIG['EMA_LENGTH']}")
    print(f"  FVG Multiplier   : {CONFIG['FVG_MULT']}")
    print(f"  Risk:Reward      : 1:{CONFIG['RISK_REWARD']:.1f}")
    print(f"  Taker Fee        : {CONFIG['TAKER_FEE']:.3%}")
    print(f"  Spread           : {CONFIG['SPREAD']:.3%}")
    print(f"  SL Slippage      : {CONFIG['SL_SLIPPAGE']:.3%}")
    print(f"  Max Leverage     : {CONFIG['MAX_LEVERAGE']:.0f}×")
    print(S)
    print("  OUT-OF-SAMPLE WINDOW")
    print(f"  Start            : {oos_start}")
    print(f"  End              : {oos_end}")
    print(f"  Calendar Days    : {n_days}")
    print(f"  Trades (FVG)     : {m_fvg['n_trades']}")
    print(f"  Trades (Bench)   : {m_bench['n_trades']}")
    if m_fvg["n_trades"] < 30:
        print("  ⚠  WARNING: < 30 FVG trades — limited statistical significance")
    print(S)
    print("  FVG + EMA200 PERFORMANCE")
    _metrics_block(m_fvg)
    print(S)
    print("  EMA200-ONLY BENCHMARK")
    _metrics_block(m_bench)
    print(S)
    print("  MONTE CARLO ROBUSTNESS  (FVG + EMA200)")
    print(f"  Iterations       : {CONFIG['MC_ITERATIONS']:,}")
    if len(mc["final_equities"]) > 0:
        print(f"  Median Equity    : ${mc['median']:>12,.2f}")
        print(f"  Worst 5%         : ${mc['p5']:>12,.2f}")
        print(f"  Best  5%         : ${mc['p95']:>12,.2f}")
        print(f"  Prob. Profitable : {mc['prob_profit']:>12.1%}")
    else:
        print("  Insufficient trades for Monte Carlo.")
    print(S)
    print("  MODELLING ASSUMPTIONS")
    print("  ✓ Next-candle execution  (no look-ahead bias)")
    print("  ✓ Conservative SL-first  (if SL & TP both within same bar)")
    print("  ✓ Taker fees included    (both sides)")
    print("  ✓ Spread cost included   (both sides)")
    print("  ✓ SL slippage included   (fixed %)")
    print(f"  ✓ Leverage capped at     {CONFIG['MAX_LEVERAGE']:.0f}×")
    print("  ✗ Funding rate costs     NOT included in PnL")
    print(f"  ↳ FVG funding windows  : {m_fvg['total_funding_windows']:,}")
    print(f"  ↳ Bench funding windows: {m_bench['total_funding_windows']:,}")
    print(S)
    print(f"  RESEARCH VERDICT:  {stars}  {verdict}")
    print()
    bullet = "  ✓" if verdict == "PROMOTE" else "  ✗"
    for r in reasons:
        print(f"{bullet} {r}")
    print()
    if chart_paths:
        print("  OUTPUT FILES")
        for p in chart_paths:
            print(f"  → {p}")
    print(S)


# =============================================================================
# SECTION 11 — MAIN PIPELINE
# =============================================================================

def process_symbol(symbol: str) -> None:
    """Full research pipeline for one symbol."""
    sep = "─" * 67
    print(f"\n{sep}\n  PROCESSING: {symbol}\n{sep}")

    # 1. Data
    df = get_data(symbol)
    n  = len(df)
    print(f"  Total candles : {n:,}")

    warm_up = CONFIG["EMA_LENGTH"] * 3
    if n < warm_up:
        print(f"  [SKIP] Need ≥ {warm_up:,} candles for EMA warm-up. Got {n:,}.")
        return

    # 2. Indicators
    df = add_indicators(df)

    # 3. Train / OOS split
    split = int(n * CONFIG["TRAIN_RATIO"])
    df_oos = df.iloc[split:].reset_index(drop=True)

    oos_start = str(df_oos["datetime"].iloc[0].date())
    oos_end   = str(df_oos["datetime"].iloc[-1].date())
    n_days    = (df_oos["datetime"].iloc[-1] - df_oos["datetime"].iloc[0]).days

    print(f"  Train      : {df['datetime'].iloc[0].date()} → "
          f"{df['datetime'].iloc[split-1].date()} ({split:,} bars)")
    print(f"  OOS        : {oos_start} → {oos_end} "
          f"({len(df_oos):,} bars / {n_days} days)")

    # Quick diagnostic: count raw FVG signals across entire dataset
    all_fvg = strategy_fvg_ema(df).sum()
    oos_fvg = strategy_fvg_ema(df_oos).sum()
    print(f"  FVG signals: {all_fvg:,} total / {oos_fvg:,} in OOS")

    # 4. Backtests (OOS only)
    print("\n  Running FVG + EMA200 backtest...")
    res_fvg = run_backtest(df_oos, strategy_fvg_ema, "FVG + EMA200")

    print("  Running EMA200-only benchmark...")
    res_bench = run_backtest(df_oos, strategy_ema_only, "EMA200 Only")

    print(f"  Trades executed: {len(res_fvg['trades'])} FVG "
          f"| {len(res_bench['trades'])} Benchmark")

    # 5. Metrics
    m_fvg   = compute_metrics(res_fvg["trades"],   "FVG + EMA200")
    m_bench = compute_metrics(res_bench["trades"], "EMA200 Only")

    # 6. Monte Carlo
    print(f"  Running Monte Carlo ({CONFIG['MC_ITERATIONS']:,} iterations)...")
    mc = monte_carlo(m_fvg["pnls"], CONFIG["MC_ITERATIONS"])

    # 7. Trade log
    log = save_trade_log(res_fvg["trades"], res_bench["trades"], symbol)
    if log:
        print(f"  Trade log → {log}")

    # 8. Charts
    print("  Generating charts...")
    paths = plot_results(m_fvg, m_bench, mc, symbol, oos_start, oos_end)
    if log:
        paths.append(log)

    # 9. Report
    print_report(symbol, m_fvg, m_bench, mc,
                 oos_start, oos_end, n_days, paths)


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║              QUANTLAB AI — RESEARCH MVP v1.0                      ║
║         Institutional-Style Strategy Hypothesis Tester            ║
╚═══════════════════════════════════════════════════════════════════╝

  Strategy Under Test : Bullish FVG (3-candle) + EMA200
  Control Benchmark   : EMA200 Bullish Crossover
  Data Source         : OKX Public REST API (no authentication)
  Execution Model     : Event-driven | Next-candle entry | SL-first
  Evaluation Window   : Out-of-Sample only (last 30%%)
""")

    print("  ACTIVE CONFIGURATION")
    print(f"  {'─' * 44}")
    for sym in CONFIG["SYMBOLS"]:
        print(f"  Symbol        : {sym}")
    print(f"  Timeframe     : {CONFIG['TIMEFRAME']}")
    print(f"  History       : {CONFIG['MONTHS_HISTORY']} months")
    print(f"  EMA Length    : {CONFIG['EMA_LENGTH']}")
    print(f"  Risk:Reward   : 1:{CONFIG['RISK_REWARD']}")
    print(f"  Taker Fee     : {CONFIG['TAKER_FEE']:.3%}")
    print(f"  Spread        : {CONFIG['SPREAD']:.3%}")
    print(f"  SL Slippage   : {CONFIG['SL_SLIPPAGE']:.3%}")
    print(f"  Max Leverage  : {CONFIG['MAX_LEVERAGE']:.0f}×")
    print(f"  Capital       : ${CONFIG['STARTING_CAPITAL']:,.0f}")
    print(f"  Risk/Trade    : {CONFIG['RISK_PER_TRADE_PCT']:.1%}")
    print(f"  Cache         : {CONFIG['CACHE_FOLDER']}/")
    print(f"  Output        : {CONFIG['OUTPUT_FOLDER']}/")

    random.seed(42)
    np.random.seed(42)

    for sym in CONFIG["SYMBOLS"]:
        try:
            process_symbol(sym)
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {sym}: {exc}")
            traceback.print_exc()

    print(f"\n  All symbols processed.")
    print(f"  Results saved to: {CONFIG['OUTPUT_FOLDER']}/\n")


if __name__ == "__main__":
    main()
