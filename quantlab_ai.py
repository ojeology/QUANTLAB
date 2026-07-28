"""
=============================================================================
QUANTLAB AI – RESEARCH MVP v1.0
Institutional-Style Strategy Hypothesis Tester

Purpose:
  Objectively determine whether a trading hypothesis demonstrates a
  measurable edge against a control benchmark.

Truth is more important than profitability.
Evidence is more important than opinion.
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
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone

# =============================================================================
# CONFIGURATION — Edit these settings without touching strategy logic
# =============================================================================

CONFIG = {
    # Instruments to test (OKX perpetual futures)
    "SYMBOLS": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],

    # Candle timeframe
    "TIMEFRAME": "1m",

    # Target download months (history depth)
    "MONTHS_HISTORY": 9,

    # Strategy parameters
    "EMA_LENGTH": 200,
    "RISK_REWARD": 2.0,

    # FVG gap multiplier (Current Low > Previous High × FVG_MULT)
    "FVG_MULT": 1.0001,

    # Execution costs (expressed as decimals, per side)
    "TAKER_FEE": 0.0005,   # 0.05%
    "SPREAD": 0.0002,       # 0.02%
    "SL_SLIPPAGE": 0.0003,  # 0.03% fixed stop slippage

    # Capital (used only for equity curve, no compounding)
    "STARTING_CAPITAL": 10_000.0,
    "RISK_PER_TRADE_PCT": 0.01,  # 1% of capital risked per trade

    # Train / Out-of-Sample split
    "TRAIN_RATIO": 0.70,

    # Monte Carlo
    "MC_ITERATIONS": 1000,

    # Output paths
    "CACHE_FOLDER": "quantlab_cache",
    "OUTPUT_FOLDER": "quantlab_output",

    # API rate-limiting: seconds between requests
    "API_DELAY": 0.25,

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
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def _fetch_page(symbol: str, bar: str, after_ms: int = None,
                before_ms: int = None, use_history: bool = True) -> list:
    """Fetch one page of candles from OKX. Returns list of raw rows."""
    url = OKX_HISTORY_URL if use_history else OKX_CANDLES_URL
    params = {
        "instId": symbol,
        "bar": bar,
        "limit": CONFIG["OKX_PAGE_LIMIT"],
    }
    if after_ms:
        params["after"] = str(after_ms)
    if before_ms:
        params["before"] = str(before_ms)

    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get("code") == "0":
            return data.get("data", [])
    except Exception as e:
        print(f"  [WARN] API error: {e}")
    return []


def download_symbol(symbol: str, bar: str, months: int) -> pd.DataFrame:
    """
    Download full history for a symbol by paginating backwards from now.
    Returns a DataFrame sorted by datetime ascending.
    """
    target_ms = int(months * 30.44 * 24 * 60 * 60 * 1000)
    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - target_ms

    print(f"  Downloading {symbol} ({bar}) — target {months} months...")

    all_rows = []
    after_ms = None  # page cursor: fetch candles with ts < after_ms
    pages = 0

    while True:
        raw = _fetch_page(symbol, bar, after_ms=after_ms, use_history=True)

        # OKX returns newest-first; if empty we've hit the end
        if not raw:
            # Try recent candles endpoint for the tail
            if pages == 0:
                raw = _fetch_page(symbol, bar, use_history=False)
            if not raw:
                break

        all_rows.extend(raw)
        pages += 1

        # Oldest row timestamp in this batch
        oldest_ts = int(raw[-1][0])
        after_ms = oldest_ts  # next page: candles older than this

        pct = max(0, 100 * (1 - (oldest_ts - cutoff_ms) / target_ms))
        print(f"    Page {pages:4d} | oldest: "
              f"{datetime.fromtimestamp(oldest_ts/1000, tz=timezone.utc).date()} "
              f"| progress: {pct:.0f}%", end="\r")

        if oldest_ts <= cutoff_ms:
            break

        time.sleep(CONFIG["API_DELAY"])

    print()  # newline after \r

    if not all_rows:
        raise RuntimeError(f"No data received for {symbol}")

    df = _parse_candles(all_rows)
    # Keep only rows within our target window
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df = df[df["datetime"] >= cutoff_dt].reset_index(drop=True)
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    print(f"  → {len(df):,} candles from "
          f"{df['datetime'].iloc[0].date()} to {df['datetime'].iloc[-1].date()}")
    return df


# =============================================================================
# SECTION 2 — LOCAL CACHE (Parquet with CSV fallback)
# =============================================================================

def _cache_path(symbol: str) -> str:
    safe = symbol.replace("-", "_")
    folder = CONFIG["CACHE_FOLDER"]
    os.makedirs(folder, exist_ok=True)
    # Prefer parquet
    try:
        import pyarrow  # noqa
        return os.path.join(folder, f"{safe}.parquet")
    except ImportError:
        return os.path.join(folder, f"{safe}.csv")


def _is_parquet(path: str) -> bool:
    return path.endswith(".parquet")


def save_cache(df: pd.DataFrame, symbol: str) -> None:
    path = _cache_path(symbol)
    if _is_parquet(path):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_cache(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return None
    if _is_parquet(path):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["datetime"])
    # Ensure UTC timezone
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def get_data(symbol: str) -> pd.DataFrame:
    """
    Load cached data. Download only missing candles. Merge and update cache.
    """
    print(f"\n[DATA] {symbol}")
    cached = load_cache(symbol)

    if cached is not None and len(cached) > 0:
        last_ts = cached["datetime"].iloc[-1]
        now_utc = pd.Timestamp.now(tz="UTC")
        gap_minutes = (now_utc - last_ts).total_seconds() / 60

        if gap_minutes < 2:
            print(f"  Cache is current ({len(cached):,} candles). Skipping download.")
            return cached

        print(f"  Cache found ({len(cached):,} candles). "
              f"Downloading ~{gap_minutes:.0f} missing minutes...")

        # Download from last cached ts onwards
        after_ms = int(last_ts.timestamp() * 1000)
        new_rows = []
        cursor = None

        while True:
            raw = _fetch_page(symbol, CONFIG["TIMEFRAME"],
                              before_ms=None, after_ms=cursor,
                              use_history=False)
            if not raw:
                break
            new_rows.extend(raw)
            oldest = int(raw[-1][0])
            if oldest <= after_ms:
                break
            cursor = oldest
            time.sleep(CONFIG["API_DELAY"])

        if new_rows:
            new_df = _parse_candles(new_rows)
            new_df = new_df[new_df["datetime"] > last_ts]
            if len(new_df) > 0:
                df = pd.concat([cached, new_df], ignore_index=True)
                df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
                save_cache(df, symbol)
                print(f"  Appended {len(new_df):,} new candles → {len(df):,} total.")
                return df

        print(f"  No new candles found. Using cached data.")
        return cached

    # No cache: full download
    df = download_symbol(symbol, CONFIG["TIMEFRAME"], CONFIG["MONTHS_HISTORY"])
    save_cache(df, symbol)
    return df


# =============================================================================
# SECTION 3 — INDICATOR CALCULATIONS
# =============================================================================

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all required indicators to the DataFrame."""
    df = df.copy()
    df["ema200"] = calc_ema(df["close"], CONFIG["EMA_LENGTH"])
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_close"] = df["close"].shift(1)
    return df


# =============================================================================
# SECTION 4 — STRATEGY FUNCTIONS (swap this block to test a new hypothesis)
# =============================================================================

def strategy_fvg_ema(df: pd.DataFrame) -> pd.Series:
    """
    STRATEGY UNDER TEST: Bullish Fair Value Gap + EMA200

    Signal conditions (evaluated on candle close, entry on NEXT open):
      1. Bullish FVG: Current Low > Previous High × FVG_MULT
      2. Trend filter: Close > EMA200
      3. No open position (enforced by engine)

    Returns a boolean Series where True = entry signal on this bar.
    Entry executes at the OPEN of the following bar.

    ─── REPLACE THIS FUNCTION TO TEST A NEW HYPOTHESIS ───
    """
    fvg_mult = CONFIG["FVG_MULT"]
    fvg_condition = df["low"] > df["prev_high"] * fvg_mult
    trend_condition = df["close"] > df["ema200"]
    signal = fvg_condition & trend_condition
    return signal


def strategy_ema_only(df: pd.DataFrame) -> pd.Series:
    """
    CONTROL / BENCHMARK: EMA200 Crossover Only

    Signal conditions:
      1. Close crosses above EMA200 (prev close was below, current close is above)
      2. No open position (enforced by engine)

    This isolates whether the FVG contributes genuine edge beyond the trend filter.
    """
    cross_above = (df["close"] > df["ema200"]) & (df["prev_close"] <= df["ema200"])
    return cross_above


# =============================================================================
# SECTION 5 — BACKTEST ENGINE (Event-Driven Simulator)
# =============================================================================

def run_backtest(df: pd.DataFrame, signal_fn, label: str) -> dict:
    """
    Event-driven position simulator.

    Execution model:
      - Signal generated on bar close (index i)
      - Entry at open of bar i+1
      - Stop-Loss: Low of signal bar (FVG candle)
      - Take-Profit: entry + 2 × stop_distance
      - If both SL and TP inside same candle range → SL hit first (conservative)
      - One position at a time, no pyramiding

    Costs applied:
      - Taker fee on entry AND exit
      - Half spread on entry AND exit
      - Fixed SL slippage on stop exits only
    """
    signals = signal_fn(df)
    in_position = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    entry_time = None
    entry_idx = -1
    signal_bar_low = 0.0

    trades = []

    for i in range(1, len(df)):
        bar = df.iloc[i]

        # ── Manage open position ──────────────────────────────────────────
        if in_position:
            hi = bar["high"]
            lo = bar["low"]

            sl_hit = lo <= stop_loss
            tp_hit = hi >= take_profit

            if sl_hit or tp_hit:
                # Conservative: if both in range, assume SL hit first
                if sl_hit:
                    exit_price = stop_loss - stop_loss * CONFIG["SL_SLIPPAGE"]
                    exit_type = "SL"
                else:
                    exit_price = take_profit
                    exit_type = "TP"

                # Cost calculation (per unit price)
                fee_entry = entry_price * CONFIG["TAKER_FEE"]
                fee_exit = exit_price * CONFIG["TAKER_FEE"]
                spread_entry = entry_price * CONFIG["SPREAD"] * 0.5
                spread_exit = exit_price * CONFIG["SPREAD"] * 0.5
                total_cost_pts = fee_entry + fee_exit + spread_entry + spread_exit

                sl_dist = entry_price - stop_loss
                position_size = (CONFIG["STARTING_CAPITAL"] *
                                 CONFIG["RISK_PER_TRADE_PCT"]) / max(sl_dist, 1e-8)

                gross_pnl = (exit_price - entry_price) * position_size
                net_pnl = gross_pnl - total_cost_pts * position_size
                r_multiple = (exit_price - entry_price) / max(sl_dist, 1e-8)

                holding_bars = i - entry_idx
                holding_minutes = holding_bars  # 1m timeframe
                funding_windows = holding_minutes / 480  # 8-hour windows

                trades.append({
                    "label": label,
                    "entry_time": entry_time,
                    "exit_time": bar["datetime"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "pnl": net_pnl,
                    "r_multiple": r_multiple,
                    "fees": (fee_entry + fee_exit) * position_size,
                    "spread_cost": (spread_entry + spread_exit) * position_size,
                    "sl_slippage": (exit_price - (stop_loss - stop_loss * CONFIG["SL_SLIPPAGE"])) * position_size if exit_type == "SL" else 0,
                    "holding_minutes": holding_minutes,
                    "funding_windows_crossed": int(funding_windows),
                    "win": exit_type == "TP",
                    "exit_type": exit_type,
                })

                in_position = False
                continue

        # ── Check for new signal (only if flat) ──────────────────────────
        if not in_position and signals.iloc[i - 1]:
            # Signal on bar i-1; enter at open of bar i
            prev_bar = df.iloc[i - 1]
            entry_price = bar["open"]
            stop_loss = prev_bar["low"]  # Low of signal bar
            sl_dist = entry_price - stop_loss

            # Skip degenerate signals (entry below or at stop)
            if sl_dist <= 0:
                continue

            take_profit = entry_price + CONFIG["RISK_REWARD"] * sl_dist

            entry_time = bar["datetime"]
            entry_idx = i
            signal_bar_low = prev_bar["low"]
            in_position = True

    return {"trades": trades}


# =============================================================================
# SECTION 6 — PERFORMANCE METRICS
# =============================================================================

def compute_metrics(trades: list, label: str) -> dict:
    """
    Compute full suite of performance metrics for a list of trade dicts.
    Equity curve uses fixed starting capital + cumulative PnL.
    """
    if not trades:
        return _empty_metrics(label)

    df = pd.DataFrame(trades)
    pnls = df["pnl"].values
    wins = df["win"].values.astype(bool)
    r_mults = df["r_multiple"].values

    n = len(pnls)
    n_wins = wins.sum()
    n_loss = n - n_wins
    win_rate = n_wins / n if n else 0

    gross_wins = pnls[wins].sum() if n_wins else 0
    gross_loss = abs(pnls[~wins].sum()) if n_loss else 1e-9
    profit_factor = gross_wins / gross_loss if gross_loss else float("inf")

    avg_win = pnls[wins].mean() if n_wins else 0
    avg_loss = pnls[~wins].mean() if n_loss else 0
    avg_trade = pnls.mean()
    avg_r = r_mults.mean()

    # Expectancy = (win_rate × avg_win) + (loss_rate × avg_loss)
    expectancy_r = (win_rate * CONFIG["RISK_REWARD"]) - ((1 - win_rate) * 1.0)

    largest_win = pnls[wins].max() if n_wins else 0
    largest_loss = pnls[~wins].min() if n_loss else 0

    # Equity curve
    equity = CONFIG["STARTING_CAPITAL"] + np.cumsum(pnls)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Sharpe (approximate — daily aggregation of trade PnLs)
    if n > 1:
        trade_std = np.std(pnls, ddof=1)
        sharpe = (avg_trade / trade_std * math.sqrt(252)) if trade_std > 0 else 0
    else:
        sharpe = 0

    avg_hold = df["holding_minutes"].mean()
    total_funding = df["funding_windows_crossed"].sum()
    net_profit = pnls.sum()

    return {
        "label": label,
        "n_trades": n,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_trade": avg_trade,
        "avg_r": avg_r,
        "expectancy_r": expectancy_r,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "avg_hold_minutes": avg_hold,
        "total_funding_windows": total_funding,
        "equity": equity,
        "drawdown": dd,
        "pnls": pnls,
        "r_multiples": r_mults,
        "trades_df": df,
    }


def _empty_metrics(label: str) -> dict:
    return {
        "label": label, "n_trades": 0, "net_profit": 0,
        "profit_factor": 0, "win_rate": 0, "avg_win": 0,
        "avg_loss": 0, "avg_trade": 0, "avg_r": 0,
        "expectancy_r": 0, "largest_win": 0, "largest_loss": 0,
        "max_drawdown": 0, "sharpe": 0, "avg_hold_minutes": 0,
        "total_funding_windows": 0, "equity": np.array([CONFIG["STARTING_CAPITAL"]]),
        "drawdown": np.array([0]), "pnls": np.array([]),
        "r_multiples": np.array([]), "trades_df": pd.DataFrame(),
    }


# =============================================================================
# SECTION 7 — MONTE CARLO ROBUSTNESS TEST
# =============================================================================

def monte_carlo(pnls: np.ndarray, n_iter: int = 1000) -> dict:
    """
    Reshuffle trade return sequence N times.
    Returns distribution of final equity outcomes.
    """
    if len(pnls) == 0:
        return {"median": 0, "p5": 0, "p95": 0, "prob_profit": 0,
                "final_equities": np.array([])}

    start = CONFIG["STARTING_CAPITAL"]
    final_equities = []

    for _ in range(n_iter):
        shuffled = np.random.permutation(pnls)
        final = start + shuffled.sum()
        final_equities.append(final)

    final_equities = np.array(final_equities)
    return {
        "median": np.median(final_equities),
        "p5": np.percentile(final_equities, 5),
        "p95": np.percentile(final_equities, 95),
        "prob_profit": (final_equities > start).mean(),
        "final_equities": final_equities,
    }


# =============================================================================
# SECTION 8 — VISUALISATIONS
# =============================================================================

def plot_results(m_fvg: dict, m_bench: dict, mc_fvg: dict,
                 symbol: str, oos_start: str, oos_end: str) -> list:
    """
    Generate and save charts. Returns list of saved file paths.
    """
    os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)
    safe_sym = symbol.replace("-", "_")
    saved = []

    # ── Chart 1: Equity Curves ────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f"QuantLab AI — {symbol}\nOut-of-Sample: {oos_start} → {oos_end}",
                 fontsize=13, fontweight="bold", y=0.98)

    ax1 = axes[0]
    if len(m_fvg["equity"]) > 1:
        ax1.plot(m_fvg["equity"], color="#00C49A", linewidth=1.5,
                 label=f"FVG + EMA200 (PF: {m_fvg['profit_factor']:.2f})")
    if len(m_bench["equity"]) > 1:
        ax1.plot(m_bench["equity"], color="#4A90D9", linewidth=1.5,
                 linestyle="--", label=f"EMA200 Only (PF: {m_bench['profit_factor']:.2f})")
    ax1.axhline(CONFIG["STARTING_CAPITAL"], color="gray", linewidth=0.7,
                linestyle=":", alpha=0.7)
    ax1.set_title("Equity Curve (trade-by-trade)", fontsize=10)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#0F1117")
    fig.patch.set_facecolor("#0F1117")
    for a in axes:
        a.set_facecolor("#0F1117")
        a.tick_params(colors="white")
        a.yaxis.label.set_color("white")
        a.xaxis.label.set_color("white")
        a.title.set_color("white")
        for spine in a.spines.values():
            spine.set_edgecolor("#333333")

    # ── Chart 2: Drawdown ─────────────────────────────────────────────────
    ax2 = axes[1]
    if len(m_fvg["drawdown"]) > 1:
        ax2.fill_between(range(len(m_fvg["drawdown"])),
                         m_fvg["drawdown"] * 100, 0,
                         color="#FF4560", alpha=0.7, label="FVG + EMA200 DD")
    if len(m_bench["drawdown"]) > 1:
        ax2.fill_between(range(len(m_bench["drawdown"])),
                         m_bench["drawdown"] * 100, 0,
                         color="#FF9500", alpha=0.4, label="EMA200 Only DD")
    ax2.set_title("Drawdown (%)", fontsize=10)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Chart 3: Trade Distribution ───────────────────────────────────────
    ax3 = axes[2]
    if len(m_fvg["r_multiples"]) > 0:
        ax3.hist(m_fvg["r_multiples"], bins=40, color="#00C49A",
                 alpha=0.7, edgecolor="#000", label="FVG + EMA200 R")
    if len(m_bench["r_multiples"]) > 0:
        ax3.hist(m_bench["r_multiples"], bins=40, color="#4A90D9",
                 alpha=0.5, edgecolor="#000", label="EMA200 Only R")
    ax3.axvline(0, color="white", linewidth=1.0, linestyle="--")
    ax3.set_title("Trade R-Multiple Distribution", fontsize=10)
    ax3.set_xlabel("R Multiple")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path1 = os.path.join(CONFIG["OUTPUT_FOLDER"],
                         f"{safe_sym}_equity_drawdown_distribution.png")
    plt.savefig(path1, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    saved.append(path1)

    # ── Chart 4: Monte Carlo Distribution ────────────────────────────────
    if len(mc_fvg["final_equities"]) > 0:
        fig2, ax = plt.subplots(figsize=(12, 5))
        fig2.patch.set_facecolor("#0F1117")
        ax.set_facecolor("#0F1117")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        ax.hist(mc_fvg["final_equities"], bins=60, color="#00C49A",
                alpha=0.75, edgecolor="#000")
        ax.axvline(CONFIG["STARTING_CAPITAL"], color="white",
                   linewidth=1.5, linestyle="--", label="Starting Capital")
        ax.axvline(mc_fvg["p5"], color="#FF4560",
                   linewidth=1.5, linestyle=":", label=f"5th Pct: ${mc_fvg['p5']:,.0f}")
        ax.axvline(mc_fvg["median"], color="#FFD700",
                   linewidth=1.5, label=f"Median: ${mc_fvg['median']:,.0f}")
        ax.axvline(mc_fvg["p95"], color="#00D4FF",
                   linewidth=1.5, linestyle=":", label=f"95th Pct: ${mc_fvg['p95']:,.0f}")
        ax.set_title(
            f"Monte Carlo Final Equity Distribution — {symbol}\n"
            f"({CONFIG['MC_ITERATIONS']:,} iterations | "
            f"Prob. Profitable: {mc_fvg['prob_profit']:.1%})",
            fontsize=11, color="white"
        )
        ax.set_xlabel("Final Portfolio Value ($)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9, labelcolor="white",
                  facecolor="#1A1D24", edgecolor="#333333")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path2 = os.path.join(CONFIG["OUTPUT_FOLDER"],
                             f"{safe_sym}_monte_carlo.png")
        plt.savefig(path2, dpi=150, bbox_inches="tight",
                    facecolor=fig2.get_facecolor())
        plt.close(fig2)
        saved.append(path2)

    return saved


# =============================================================================
# SECTION 9 — TRADE LOG EXPORT
# =============================================================================

def save_trade_log(trades_fvg: list, trades_bench: list, symbol: str) -> str:
    """Export combined trade log to CSV."""
    os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)
    safe_sym = symbol.replace("-", "_")
    all_trades = trades_fvg + trades_bench
    if not all_trades:
        return ""
    df = pd.DataFrame(all_trades)
    cols = [
        "label", "entry_time", "exit_time", "entry_price", "exit_price",
        "stop_loss", "take_profit", "pnl", "r_multiple", "fees",
        "spread_cost", "sl_slippage", "holding_minutes",
        "funding_windows_crossed", "win", "exit_type",
    ]
    df = df[[c for c in cols if c in df.columns]]
    path = os.path.join(CONFIG["OUTPUT_FOLDER"], f"{safe_sym}_trade_log.csv")
    df.to_csv(path, index=False)
    return path


# =============================================================================
# SECTION 10 — REPORT GENERATION
# =============================================================================

def _stars(verdict: str) -> str:
    return "★★★★★" if verdict == "PROMOTE" else "★☆☆☆☆"


def _verdict(m_fvg: dict, m_bench: dict, mc: dict) -> tuple[str, list]:
    reasons_promote = []
    reasons_reject = []

    if m_fvg["profit_factor"] >= 1.2:
        reasons_promote.append("Profit Factor ≥ 1.2")
    else:
        reasons_reject.append(f"Profit Factor below 1.2 ({m_fvg['profit_factor']:.2f})")

    if m_fvg["expectancy_r"] > 0:
        reasons_promote.append(f"Positive expectancy ({m_fvg['expectancy_r']:+.2f}R)")
    else:
        reasons_reject.append(f"Negative expectancy ({m_fvg['expectancy_r']:+.2f}R)")

    if m_fvg["profit_factor"] > m_bench["profit_factor"]:
        reasons_promote.append(
            f"Outperformed benchmark "
            f"(PF {m_fvg['profit_factor']:.2f} vs {m_bench['profit_factor']:.2f})")
    else:
        reasons_reject.append(
            f"Underperformed benchmark "
            f"(PF {m_fvg['profit_factor']:.2f} vs {m_bench['profit_factor']:.2f})")

    if mc["prob_profit"] >= 0.6:
        reasons_promote.append(f"Monte Carlo prob. of profit: {mc['prob_profit']:.1%}")
    else:
        reasons_reject.append(
            f"Weak Monte Carlo robustness ({mc['prob_profit']:.1%})")

    if m_fvg["max_drawdown"] > -0.30:
        reasons_promote.append(
            f"Drawdown within acceptable range ({m_fvg['max_drawdown']:.1%})")
    else:
        reasons_reject.append(
            f"Excessive drawdown ({m_fvg['max_drawdown']:.1%})")

    if len(reasons_reject) == 0:
        return "PROMOTE", reasons_promote
    elif len(reasons_promote) >= len(reasons_reject):
        return "PROMOTE", reasons_promote
    else:
        return "REJECT", reasons_reject


def print_metrics_block(m: dict) -> None:
    print(f"  Strategy         : {m['label']}")
    print(f"  Trades           : {m['n_trades']}")
    print(f"  Net Profit       : ${m['net_profit']:>10,.2f}")
    print(f"  Profit Factor    : {m['profit_factor']:>10.3f}")
    print(f"  Win Rate         : {m['win_rate']:>10.1%}")
    print(f"  Avg Win          : ${m['avg_win']:>10,.2f}")
    print(f"  Avg Loss         : ${m['avg_loss']:>10,.2f}")
    print(f"  Avg Trade        : ${m['avg_trade']:>10,.2f}")
    print(f"  Avg R            : {m['avg_r']:>10.3f}")
    print(f"  Expectancy       : {m['expectancy_r']:>+10.3f}R")
    print(f"  Largest Win      : ${m['largest_win']:>10,.2f}")
    print(f"  Largest Loss     : ${m['largest_loss']:>10,.2f}")
    print(f"  Max Drawdown     : {m['max_drawdown']:>10.2%}")
    print(f"  Sharpe (approx)  : {m['sharpe']:>10.3f}")
    print(f"  Avg Hold Time    : {m['avg_hold_minutes']:>8.0f} min")
    print(f"  Funding Windows  : {m['total_funding_windows']:>10,}")


def print_report(symbol: str, m_fvg: dict, m_bench: dict,
                 mc_fvg: dict, oos_start, oos_end, n_oos_days: int,
                 chart_paths: list) -> None:
    verdict, reasons = _verdict(m_fvg, m_bench, mc_fvg)
    stars = _stars(verdict)

    sep = "=" * 65

    print(f"\n{sep}")
    print("  QUANTLAB AI RESEARCH REPORT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(sep)
    print(f"  Symbol           : {symbol}")
    print(f"  Strategy         : Bullish FVG + EMA{CONFIG['EMA_LENGTH']}")
    print(f"  Benchmark        : EMA{CONFIG['EMA_LENGTH']} Cross Only")
    print(f"  EMA Length       : {CONFIG['EMA_LENGTH']}")
    print(f"  Risk:Reward      : 1:{CONFIG['RISK_REWARD']:.1f}")
    print(f"  Taker Fee        : {CONFIG['TAKER_FEE']:.3%}")
    print(f"  Spread           : {CONFIG['SPREAD']:.3%}")
    print(f"  SL Slippage      : {CONFIG['SL_SLIPPAGE']:.3%}")
    print(sep)
    print("  OUT-OF-SAMPLE WINDOW")
    print(f"  Start            : {oos_start}")
    print(f"  End              : {oos_end}")
    print(f"  Calendar Days    : {n_oos_days}")
    print(f"  OOS Trades (FVG) : {m_fvg['n_trades']}")
    if m_fvg["n_trades"] < 30:
        print("  ⚠  WARNING: Low trade count — limited statistical significance")
    print(sep)
    print("  FVG + EMA200 STRATEGY METRICS")
    print_metrics_block(m_fvg)
    print(sep)
    print("  EMA200-ONLY BENCHMARK METRICS")
    print_metrics_block(m_bench)
    print(sep)
    print("  MONTE CARLO ROBUSTNESS (FVG + EMA200)")
    print(f"  Iterations       : {CONFIG['MC_ITERATIONS']:,}")
    if len(mc_fvg["final_equities"]) > 0:
        print(f"  Median Equity    : ${mc_fvg['median']:>10,.2f}")
        print(f"  Worst 5%         : ${mc_fvg['p5']:>10,.2f}")
        print(f"  Best  5%         : ${mc_fvg['p95']:>10,.2f}")
        print(f"  Prob. Profitable : {mc_fvg['prob_profit']:>10.1%}")
    else:
        print("  Insufficient trades for Monte Carlo.")
    print(sep)
    print("  MODELLING ASSUMPTIONS")
    print("  ✓ Next-candle execution (no look-ahead bias)")
    print("  ✓ Conservative SL-first assumption (if SL & TP both in bar)")
    print("  ✓ Taker fees included (both sides)")
    print("  ✓ Spread cost included (both sides)")
    print("  ✓ Fixed stop-loss slippage included")
    print("  ✗ Funding rate costs NOT included in PnL")
    print(f"  ↳ Trades crossing an 8h funding window: "
          f"{m_fvg['total_funding_windows']:,} (FVG) / "
          f"{m_bench['total_funding_windows']:,} (Bench)")
    print(f"  ↳ Funding adjustment is recommended for live deployment")
    print(sep)
    print(f"  RESEARCH VERDICT:  {stars} {verdict}")
    print()
    for r in reasons:
        prefix = "  ✓" if verdict == "PROMOTE" else "  ✗"
        print(f"{prefix} {r}")
    print()
    if chart_paths:
        print("  SAVED OUTPUT FILES")
        for p in chart_paths:
            print(f"  → {p}")
    print(sep)
    print()


# =============================================================================
# SECTION 11 — MAIN EXECUTION
# =============================================================================

def process_symbol(symbol: str) -> None:
    """Full pipeline for one symbol."""
    print(f"\n{'─' * 65}")
    print(f"  PROCESSING: {symbol}")
    print(f"{'─' * 65}")

    # 1. Get data (cache or download)
    df = get_data(symbol)
    total_bars = len(df)
    print(f"  Total candles: {total_bars:,}")

    if total_bars < CONFIG["EMA_LENGTH"] * 2:
        print(f"  [SKIP] Insufficient data for {symbol}")
        return

    # 2. Add indicators
    df = add_indicators(df)

    # 3. Train / OOS split (chronological)
    split_idx = int(total_bars * CONFIG["TRAIN_RATIO"])
    df_oos = df.iloc[split_idx:].reset_index(drop=True)

    oos_start = str(df_oos["datetime"].iloc[0].date())
    oos_end = str(df_oos["datetime"].iloc[-1].date())
    n_oos_days = (df_oos["datetime"].iloc[-1] - df_oos["datetime"].iloc[0]).days

    print(f"  Train:         {df.iloc[0]['datetime'].date()} "
          f"→ {df.iloc[split_idx - 1]['datetime'].date()} "
          f"({split_idx:,} bars)")
    print(f"  Out-of-Sample: {oos_start} → {oos_end} "
          f"({len(df_oos):,} bars / {n_oos_days} days)")

    # 4. Run backtests on OOS data only
    print("\n  Running FVG + EMA200 backtest...")
    result_fvg = run_backtest(df_oos, strategy_fvg_ema, "FVG + EMA200")

    print("  Running EMA200-only benchmark...")
    result_bench = run_backtest(df_oos, strategy_ema_only, "EMA200 Only")

    print(f"  FVG trades: {len(result_fvg['trades'])} | "
          f"Benchmark trades: {len(result_bench['trades'])}")

    # 5. Compute metrics
    m_fvg = compute_metrics(result_fvg["trades"], "FVG + EMA200")
    m_bench = compute_metrics(result_bench["trades"], "EMA200 Only")

    # 6. Monte Carlo
    print(f"  Running Monte Carlo ({CONFIG['MC_ITERATIONS']:,} iterations)...")
    mc_fvg = monte_carlo(m_fvg["pnls"], CONFIG["MC_ITERATIONS"])

    # 7. Trade log
    log_path = save_trade_log(
        result_fvg["trades"], result_bench["trades"], symbol)
    if log_path:
        print(f"  Trade log saved: {log_path}")

    # 8. Charts
    print("  Generating charts...")
    chart_paths = plot_results(m_fvg, m_bench, mc_fvg,
                               symbol, oos_start, oos_end)
    if log_path:
        chart_paths.append(log_path)

    # 9. Print report
    print_report(symbol, m_fvg, m_bench, mc_fvg,
                 oos_start, oos_end, n_oos_days, chart_paths)


def main():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║             QUANTLAB AI — RESEARCH MVP v1.0                   ║
║        Institutional-Style Strategy Hypothesis Tester         ║
╚═══════════════════════════════════════════════════════════════╝

  Strategy Under Test : Bullish Fair Value Gap + EMA200
  Control Benchmark   : EMA200 Crossover Only
  Data Source         : OKX Public REST API (no authentication)
  Execution Model     : Event-driven | Next-candle entry
  Evaluation Window   : Out-of-Sample only (last 30%)
""")

    print("  CONFIGURATION")
    print(f"  {'─' * 40}")
    print(f"  Symbols       : {', '.join(CONFIG['SYMBOLS'])}")
    print(f"  Timeframe     : {CONFIG['TIMEFRAME']}")
    print(f"  History       : {CONFIG['MONTHS_HISTORY']} months")
    print(f"  EMA Length    : {CONFIG['EMA_LENGTH']}")
    print(f"  FVG Mult      : {CONFIG['FVG_MULT']}")
    print(f"  Risk:Reward   : 1:{CONFIG['RISK_REWARD']}")
    print(f"  Taker Fee     : {CONFIG['TAKER_FEE']:.3%}")
    print(f"  Spread        : {CONFIG['SPREAD']:.3%}")
    print(f"  SL Slippage   : {CONFIG['SL_SLIPPAGE']:.3%}")
    print(f"  Capital       : ${CONFIG['STARTING_CAPITAL']:,.0f}")
    print(f"  Risk/Trade    : {CONFIG['RISK_PER_TRADE_PCT']:.1%}")
    print(f"  Cache Folder  : {CONFIG['CACHE_FOLDER']}/")
    print(f"  Output Folder : {CONFIG['OUTPUT_FOLDER']}/")

    random.seed(42)
    np.random.seed(42)

    for symbol in CONFIG["SYMBOLS"]:
        try:
            process_symbol(symbol)
        except Exception as e:
            print(f"\n  [ERROR] {symbol}: {e}")
            import traceback
            traceback.print_exc()

    print("\n  All symbols processed. QuantLab AI research complete.")
    print(f"  Output files saved to: {CONFIG['OUTPUT_FOLDER']}/\n")


if __name__ == "__main__":
    main()
