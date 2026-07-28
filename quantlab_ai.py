"""
=============================================================================
QUANTLAB AI – RESEARCH #003
Hypothesis: Does the FVG + EMA200 Slope strategy have edge specifically in
            trending market regimes, and does that edge disappear in ranging
            conditions?

Research Question:
  Does Strategy C (EMA200 + Positive Slope + Bullish FVG) produce positive
  expectancy only during objectively trending regimes (ADX ≥ 25)?  Are all
  losses concentrated in ranging markets (ADX < 20)?

Method:
  • Run Strategy C (Research #002 winner) unchanged on identical OOS data.
  • Tag every trade with the ADX-based regime at time of entry — do not
    reject any trades.
  • Group results by regime.  Report performance metrics per regime.
  • Run a what-if attribution: remove ranging-regime trades and compare
    the filtered result against the unfiltered baseline.

Three strategies still reported side-by-side for continuity:
  A  EMA200 Bullish Crossover Only        (benchmark)
  B  EMA200 + Bullish FVG                 (Research #001)
  C  EMA200 + FVG + Positive Slope        (Research #002 — regime-tagged here)

Backtest engine, fees, spread, SL, TP, slope logic, train/test split: UNCHANGED.
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

    # Candle timeframe
    # 1H used — Research #001 established 1m gives 0 FVG signals on 24/7 crypto.
    "TIMEFRAME": "1H",

    # Target download months (history depth)
    "MONTHS_HISTORY": 9,

    # Strategy parameters
    "EMA_LENGTH": 200,
    "RISK_REWARD": 2.0,

    # FVG gap multiplier: candle[i].low > candle[i-2].high × FVG_MULT
    "FVG_MULT": 1.0001,

    # Execution costs — UNCHANGED from Research #001/#002
    "TAKER_FEE":    0.0005,   # 0.05%
    "SPREAD":       0.0002,   # 0.02%
    "SL_SLIPPAGE":  0.0003,   # 0.03% fixed stop slippage

    # Minimum SL distance as % of entry price
    "MIN_SL_PCT": 0.001,

    # Leverage cap
    "MAX_LEVERAGE": 5.0,

    # Capital model
    "STARTING_CAPITAL":   10_000.0,
    "RISK_PER_TRADE_PCT": 0.01,

    # Train / Out-of-Sample split — UNCHANGED
    "TRAIN_RATIO": 0.70,

    # Monte Carlo
    "MC_ITERATIONS": 1000,

    # Output paths
    "CACHE_FOLDER":  "quantlab_cache",
    "OUTPUT_FOLDER": "quantlab_output",

    # API rate-limiting
    "API_DELAY": 0.2,

    # Max candles per OKX API page
    "OKX_PAGE_LIMIT": 100,

    # ── Research #002 parameters (UNCHANGED) ─────────────────────────────
    # EMA200 slope lookback: slope is positive when EMA200[now] > EMA200[N bars ago]
    "SLOPE_LOOKBACK": 10,

    # ── Research #003 parameters ──────────────────────────────────────────
    # ADX-based market regime classification.
    # ADX is computed with Wilder smoothing (standard definition).
    # Thresholds are configurable — do not optimise these.
    "ADX_LENGTH":    14,      # Wilder period for ADX calculation
    "ADX_TRENDING":  25,      # ADX ≥ this  → "Trending"
    "ADX_WEAK":      20,      # ADX ≥ this and < ADX_TRENDING → "Weak Trend"
                              # ADX <  ADX_WEAK               → "Ranging"
}


# =============================================================================
# SECTION 1 — DATA DOWNLOAD (OKX Public REST API)  — UNCHANGED
# =============================================================================

OKX_HISTORY_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"

CANDLE_COLS = ["ts", "open", "high", "low", "close", "vol",
               "volCcy", "volCcyQuote", "confirm"]


def _parse_candles(raw: list) -> pd.DataFrame:
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
    now_ms    = int(time.time() * 1000)
    target_ms = int(months * 30.44 * 24 * 3600 * 1000)
    cutoff_ms = since_ms if since_ms else now_ms - target_ms

    if since_ms:
        print(f"  Fetching new candles for {symbol} since "
              f"{datetime.fromtimestamp(since_ms/1000, tz=timezone.utc).date()}...")
    else:
        print(f"  Downloading {symbol} ({bar}) — target {months} months...")

    all_rows = []
    after_ms_cursor = None
    pages = 0

    while True:
        raw = _fetch_page(symbol, bar, after_ms=after_ms_cursor, use_history=True)
        if not raw and pages == 0:
            raw = _fetch_page(symbol, bar, use_history=False)
        if not raw:
            break

        all_rows.extend(raw)
        pages += 1

        oldest_ts = int(raw[-1][0])
        after_ms_cursor = oldest_ts

        if not since_ms:
            pct = max(0.0, 100.0 * (1.0 - (oldest_ts - cutoff_ms) / target_ms))
            print(f"    Page {pages:4d} | oldest "
                  f"{datetime.fromtimestamp(oldest_ts/1000, tz=timezone.utc).date()} "
                  f"| {pct:.0f}%", end="\r")

        if oldest_ts <= cutoff_ms:
            break

        time.sleep(CONFIG["API_DELAY"])

    if not since_ms:
        print()

    if not all_rows:
        raise RuntimeError(f"No data received for {symbol}")

    df = _parse_candles(all_rows)
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df = df[df["datetime"] >= cutoff_dt]
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)
    return df


# =============================================================================
# SECTION 2 — LOCAL CACHE  — UNCHANGED
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
    bar = CONFIG["TIMEFRAME"]
    print(f"\n[DATA] {symbol} ({bar})")
    cached = load_cache(symbol, bar)

    if cached is not None and len(cached) > 0:
        last_ts = cached["datetime"].iloc[-1]
        now_utc = pd.Timestamp.now(tz="UTC")

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
        new_df   = download_symbol(symbol, bar, months=0, since_ms=since_ms)
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

    df = download_symbol(symbol, bar, months=CONFIG["MONTHS_HISTORY"])
    save_cache(df, symbol, bar)
    print(f"  → {len(df):,} candles "
          f"({df['datetime'].iloc[0].date()} – {df['datetime'].iloc[-1].date()})")
    return df


# =============================================================================
# SECTION 3 — INDICATOR CALCULATIONS
# =============================================================================

def calc_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def calc_adx(df: pd.DataFrame, length: int) -> pd.Series:
    """
    Wilder's Average Directional Index (ADX).

    Uses Wilder smoothing (alpha = 1/length) for TR, +DM, -DM, and DX.
    This is the original Wilder definition — identical to most charting
    platforms.  No look-ahead.  Returns ADX as a Series aligned to df.index.
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Raw directional movement
    up   = high - high.shift(1)
    down = low.shift(1) - low

    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down,  0.0)

    plus_dm  = pd.Series(plus_dm,  index=df.index, dtype=float)
    minus_dm = pd.Series(minus_dm, index=df.index, dtype=float)

    # Wilder smoothing (equivalent to EMA with alpha=1/length)
    alpha = 1.0 / length
    sm_tr      = tr.ewm(alpha=alpha,       adjust=False).mean()
    sm_plus    = plus_dm.ewm(alpha=alpha,  adjust=False).mean()
    sm_minus   = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di  = 100.0 * sm_plus  / sm_tr.replace(0, np.nan)
    minus_di = 100.0 * sm_minus / sm_tr.replace(0, np.nan)

    di_sum  = (plus_di + minus_di).replace(0, np.nan)
    dx      = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx     = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx.fillna(0.0)


def _regime_label(adx_val: float) -> str:
    """Map a scalar ADX value to a regime string using CONFIG thresholds."""
    if adx_val >= CONFIG["ADX_TRENDING"]:
        return "Trending"
    if adx_val >= CONFIG["ADX_WEAK"]:
        return "Weak Trend"
    return "Ranging"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators including ADX regime column."""
    df = df.copy()
    df["ema200"]      = calc_ema(df["close"], CONFIG["EMA_LENGTH"])
    df["prev_high"]   = df["high"].shift(1)
    df["prev_low"]    = df["low"].shift(1)
    df["prev_close"]  = df["close"].shift(1)
    df["high_2"]      = df["high"].shift(2)
    df["ema200_prev"] = df["ema200"].shift(1)

    # Research #002: slope filter
    n = CONFIG["SLOPE_LOOKBACK"]
    df["ema200_lag"]    = df["ema200"].shift(n)
    df["ema200_rising"] = df["ema200"] > df["ema200_lag"]

    # Research #003: ADX regime tagging
    adx = calc_adx(df, CONFIG["ADX_LENGTH"])
    df["adx"]    = adx
    df["regime"] = adx.apply(_regime_label)

    return df


# =============================================================================
# SECTION 4 — STRATEGY FUNCTIONS  — UNCHANGED from Research #002
# =============================================================================

def strategy_fvg_ema(df: pd.DataFrame) -> pd.Series:
    """Strategy B — Bullish FVG + EMA200 (Research #001)."""
    fvg   = df["low"] > df["high_2"] * CONFIG["FVG_MULT"]
    trend = df["close"] > df["ema200"]
    return fvg & trend


def strategy_ema_only(df: pd.DataFrame) -> pd.Series:
    """Strategy A — EMA200 Bullish Crossover benchmark."""
    cross_up = (df["close"] > df["ema200"]) & (df["prev_close"] <= df["ema200_prev"])
    return cross_up


def strategy_fvg_ema_slope(df: pd.DataFrame) -> pd.Series:
    """Strategy C — Bullish FVG + EMA200 + Positive Slope (Research #002)."""
    fvg   = df["low"] > df["high_2"] * CONFIG["FVG_MULT"]
    trend = df["close"] > df["ema200"]
    slope = df["ema200_rising"]
    return fvg & trend & slope


# =============================================================================
# SECTION 5 — BACKTEST ENGINE  — UNCHANGED except regime tag on trade record
# =============================================================================

def run_backtest(df: pd.DataFrame, signal_fn, label: str) -> dict:
    """
    Event-driven position simulator.  Identical to Research #002 engine.

    Research #003 addition: each trade record gains a "regime" field
    containing the ADX regime string at the signal bar (entry bar - 1).
    No trades are rejected — this is attribution tagging only.
    """
    signals   = signal_fn(df)
    min_sl    = CONFIG["MIN_SL_PCT"]
    rr        = CONFIG["RISK_REWARD"]
    max_lev   = CONFIG["MAX_LEVERAGE"]
    capital   = CONFIG["STARTING_CAPITAL"]
    risk_frac = CONFIG["RISK_PER_TRADE_PCT"]
    fee_rate  = CONFIG["TAKER_FEE"]
    spd_rate  = CONFIG["SPREAD"] * 0.5
    slp_rate  = CONFIG["SL_SLIPPAGE"]

    in_position   = False
    entry_price   = 0.0
    stop_loss     = 0.0
    take_profit   = 0.0
    entry_time    = None
    entry_idx     = -1
    position_size = 0.0
    trade_regime  = "Unknown"

    trades = []

    for i in range(1, len(df)):
        bar = df.iloc[i]

        # ── Manage open position ──────────────────────────────────────────
        if in_position:
            hi, lo = bar["high"], bar["low"]
            sl_hit = lo <= stop_loss
            tp_hit = hi >= take_profit

            if sl_hit or tp_hit:
                if sl_hit:
                    exit_price = stop_loss * (1.0 - slp_rate)
                    exit_type  = "SL"
                else:
                    exit_price = take_profit
                    exit_type  = "TP"

                gross_pnl      = (exit_price - entry_price) * position_size
                notional_entry = entry_price * position_size
                notional_exit  = exit_price  * position_size
                cost_fee   = (notional_entry + notional_exit) * fee_rate
                cost_spread = (notional_entry + notional_exit) * spd_rate
                cost_slip  = (stop_loss - exit_price) * position_size if exit_type == "SL" else 0.0
                total_cost = cost_fee + cost_spread + cost_slip
                net_pnl    = gross_pnl - total_cost

                sl_dist    = entry_price - stop_loss
                r_multiple = (exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0

                holding_bars    = i - entry_idx
                holding_minutes = holding_bars * _bar_minutes()
                funding_windows = int(holding_minutes / 480)

                trades.append({
                    "label":        label,
                    "regime":       trade_regime,   # ← Research #003 addition
                    "entry_time":   entry_time,
                    "exit_time":    bar["datetime"],
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "stop_loss":    stop_loss,
                    "take_profit":  take_profit,
                    "pnl":          net_pnl,
                    "r_multiple":   r_multiple,
                    "fees":         cost_fee,
                    "spread_cost":  cost_spread,
                    "sl_slippage":  cost_slip,
                    "holding_minutes":         holding_minutes,
                    "funding_windows_crossed": funding_windows,
                    "win":       exit_type == "TP",
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

            if sl_dist <= 0:
                continue
            if sl_dist / ep < min_sl:
                continue

            tp = ep + rr * sl_dist

            risk_dollars  = capital * risk_frac
            raw_size      = risk_dollars / sl_dist
            max_size      = (capital * max_lev) / ep
            pos_size      = min(raw_size, max_size)

            entry_price   = ep
            stop_loss     = sl
            take_profit   = tp
            position_size = pos_size
            entry_time    = bar["datetime"]
            entry_idx     = i
            in_position   = True
            # Tag regime from signal bar (prev_bar) — not entry bar
            trade_regime  = prev_bar["regime"]

    return {"trades": trades}


def _bar_minutes() -> float:
    return {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1H": 60, "2H": 120, "4H": 240, "6H": 360, "12H": 720, "1D": 1440,
    }.get(CONFIG["TIMEFRAME"], 60)


# =============================================================================
# SECTION 6 — PERFORMANCE METRICS  — UNCHANGED
# =============================================================================

def compute_metrics(trades: list, label: str) -> dict:
    if not trades:
        return _empty_metrics(label)

    df   = pd.DataFrame(trades)
    pnls = df["pnl"].values
    wins = df["win"].values.astype(bool)
    rmul = df["r_multiple"].values

    n     = len(pnls)
    n_win = int(wins.sum())
    n_los = n - n_win

    gross_wins    = pnls[wins].sum()         if n_win else 0.0
    gross_loss    = abs(pnls[~wins].sum())   if n_los else 1e-9
    profit_factor = gross_wins / gross_loss  if gross_loss > 0 else float("inf")

    win_rate   = n_win / n
    avg_win    = pnls[wins].mean()  if n_win else 0.0
    avg_loss   = pnls[~wins].mean() if n_los else 0.0
    avg_trade  = pnls.mean()
    avg_r      = rmul.mean()

    expectancy_r = (win_rate * CONFIG["RISK_REWARD"]) - ((1.0 - win_rate) * 1.0)

    largest_win  = pnls[wins].max()  if n_win else 0.0
    largest_loss = pnls[~wins].min() if n_los else 0.0

    equity = CONFIG["STARTING_CAPITAL"] + np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    max_dd = dd.min()

    trade_std     = np.std(pnls, ddof=1) if n > 1 else 0.0
    bars_per_year = (365 * 24 * 60) / _bar_minutes()
    trades_per_bar = n / max(len(pnls), 1)
    sharpe = (avg_trade / trade_std * math.sqrt(bars_per_year * trades_per_bar)
              if trade_std > 0 else 0.0)

    avg_hold      = df["holding_minutes"].mean()
    total_funding = int(df["funding_windows_crossed"].sum())
    net_profit    = float(pnls.sum())

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
        "equity":    equity,
        "drawdown":  dd,
        "pnls":      pnls,
        "r_multiples": rmul,
        "trades_df": df,
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
# SECTION 7 — REGIME ANALYSIS  (Research #003 addition)
# =============================================================================

# Ordered for consistent table display
REGIME_ORDER = ["Trending", "Weak Trend", "Ranging"]
REGIME_COLORS = {
    "Trending":   "#00C49A",   # teal
    "Weak Trend": "#FFB347",   # amber
    "Ranging":    "#FF4560",   # red
}


def compute_regime_breakdown(trades: list) -> dict[str, dict]:
    """
    Split Strategy C trades by regime and compute per-regime metrics.
    Returns {regime_label: metrics_dict}.
    """
    if not trades:
        return {r: _empty_metrics(r) for r in REGIME_ORDER}

    df = pd.DataFrame(trades)
    result = {}
    for regime in REGIME_ORDER:
        subset = df[df["regime"] == regime].to_dict("records")
        result[regime] = compute_metrics(subset, regime)
    return result


def compute_whatif(trades: list) -> dict:
    """
    What-if attribution: keep only trades taken in Trending or Weak Trend
    regimes (i.e. remove all Ranging trades).  No parameter changes — pure
    attribution.
    """
    trending_trades = [t for t in trades
                       if t["regime"] in ("Trending", "Weak Trend")]
    return compute_metrics(trending_trades, "C (Non-Ranging Only)")


# =============================================================================
# SECTION 8 — MONTE CARLO  — UNCHANGED
# =============================================================================

def monte_carlo(pnls: np.ndarray, n_iter: int = 1000) -> dict:
    if len(pnls) == 0:
        return {"median": 0.0, "p5": 0.0, "p95": 0.0,
                "prob_profit": 0.0, "final_equities": np.array([])}

    start  = CONFIG["STARTING_CAPITAL"]
    finals = np.empty(n_iter)
    for k in range(n_iter):
        shuffled = np.random.permutation(pnls)
        finals[k] = start + shuffled.sum()

    return {
        "median":      float(np.median(finals)),
        "p5":          float(np.percentile(finals, 5)),
        "p95":         float(np.percentile(finals, 95)),
        "prob_profit": float((finals > start).mean()),
        "final_equities": finals,
    }


# =============================================================================
# SECTION 9 — VISUALISATIONS
# =============================================================================

def plot_results(m_a: dict, m_b: dict, m_c: dict,
                 mc_b: dict, mc_c: dict,
                 regime_metrics: dict[str, dict],
                 m_whatif: dict,
                 symbol: str, oos_start: str, oos_end: str) -> list:
    """
    Research #003 charts:
      Chart 1 — Equity / Drawdown / R-distribution (3-strategy comparison)
      Chart 2 — Monte Carlo B vs C (unchanged from #002)
      Chart 3 — Regime breakdown bar charts + regime equity curves
    """
    os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)
    safe  = symbol.replace("-", "_")
    saved = []

    BG = "#0F1117"
    CA = "#4A90D9"
    CB = "#FFB347"
    CC = "#00C49A"
    RD = "#FF4560"

    def _style(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors="white", labelsize=8)
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
        ax.grid(True, alpha=0.2, color="#444")

    leg_kw = dict(fontsize=8, facecolor="#1A1D24", edgecolor="#444",
                  labelcolor="white")

    # ── Chart 1: Equity / Drawdown / R-distribution ───────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"QuantLab AI Research #003 — {symbol}\n"
        f"OOS: {oos_start} → {oos_end}  |  "
        f"ADX({CONFIG['ADX_LENGTH']}) thresholds: "
        f"Trending≥{CONFIG['ADX_TRENDING']}  Weak≥{CONFIG['ADX_WEAK']}",
        fontsize=11, fontweight="bold", color="white", y=0.995,
    )
    ax1, ax2, ax3 = axes
    for ax in axes:
        _style(ax)

    for m, col, ls, lbl in [
        (m_a, CA, ":",  f"A  EMA200 Only       (PF {m_a['profit_factor']:.2f})"),
        (m_b, CB, "--", f"B  FVG + EMA200      (PF {m_b['profit_factor']:.2f})"),
        (m_c, CC, "-",  f"C  FVG+EMA+Slope     (PF {m_c['profit_factor']:.2f})"),
    ]:
        if len(m["equity"]) > 1:
            ax1.plot(m["equity"], color=col, lw=1.6, ls=ls, label=lbl)

    # Also overlay what-if on equity chart
    if len(m_whatif["equity"]) > 1:
        ax1.plot(m_whatif["equity"], color="#E040FB", lw=1.4, ls="-.",
                 label=f"C (Non-Ranging)      (PF {m_whatif['profit_factor']:.2f})")

    ax1.axhline(CONFIG["STARTING_CAPITAL"], color="gray", lw=0.6, ls=":")
    ax1.set_title("Equity Curve — All Strategies + What-If", fontsize=10)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(**leg_kw)

    for m, col, alpha, lbl in [
        (m_a, CA, 0.35, "A  EMA200 Only"),
        (m_b, CB, 0.45, "B  FVG + EMA200"),
        (m_c, CC, 0.70, "C  FVG + EMA + Slope"),
    ]:
        if len(m["drawdown"]) > 1:
            ax2.fill_between(range(len(m["drawdown"])),
                             m["drawdown"] * 100, 0,
                             color=col, alpha=alpha, label=lbl)
    ax2.set_title("Drawdown (%)", fontsize=10)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(**leg_kw)

    for m, col, alpha, lbl in [
        (m_b, CB, 0.55, "B  FVG + EMA200"),
        (m_c, CC, 0.75, "C  FVG + EMA + Slope"),
    ]:
        if len(m["r_multiples"]) > 0:
            ax3.hist(m["r_multiples"], bins=35, color=col,
                     alpha=alpha, edgecolor="#111", label=lbl)
    ax3.axvline(0, color="white", lw=0.8, ls="--")
    ax3.set_title("R-Multiple Distribution  (B vs C)", fontsize=10)
    ax3.set_xlabel("R Multiple")
    ax3.set_ylabel("Count")
    ax3.legend(**leg_kw)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = os.path.join(CONFIG["OUTPUT_FOLDER"],
                      f"{safe}_r003_equity_drawdown_distribution.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    saved.append(p1)

    # ── Chart 2: Monte Carlo  ─────────────────────────────────────────────
    fig2, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.patch.set_facecolor(BG)
    fig2.suptitle(
        f"Monte Carlo Final Equity — {symbol}  "
        f"({CONFIG['MC_ITERATIONS']:,} iterations)",
        fontsize=11, fontweight="bold", color="white",
    )
    for ax, mc, col, strat_lbl in [
        (ax_b, mc_b, CB, "B  FVG + EMA200"),
        (ax_c, mc_c, CC, "C  FVG + EMA + Slope"),
    ]:
        _style(ax)
        fe = mc["final_equities"]
        if len(fe) > 0:
            n_u  = len(np.unique(fe))
            bins = max(1, min(60, n_u - 1)) if n_u > 1 else 1
            ax.hist(fe, bins=bins, color=col, alpha=0.75, edgecolor="#111")
        ax.axvline(CONFIG["STARTING_CAPITAL"], color="white", lw=1.4, ls="--",
                   label="Start")
        if mc["p5"] != mc["median"]:
            ax.axvline(mc["p5"], color=RD, lw=1.3, ls=":",
                       label=f"5th  ${mc['p5']:,.0f}")
        ax.axvline(mc["median"], color="#FFD700", lw=1.4,
                   label=f"Med  ${mc['median']:,.0f}")
        if mc["p95"] != mc["median"]:
            ax.axvline(mc["p95"], color="#00D4FF", lw=1.3, ls=":",
                       label=f"95th ${mc['p95']:,.0f}")
        ax.set_title(
            f"{strat_lbl}\nProb. Profitable: {mc['prob_profit']:.1%}",
            fontsize=10, color="white",
        )
        ax.set_xlabel("Final Equity ($)")
        ax.set_ylabel("Frequency")
        ax.legend(**leg_kw)

    plt.tight_layout()
    p2 = os.path.join(CONFIG["OUTPUT_FOLDER"], f"{safe}_r003_monte_carlo.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    saved.append(p2)

    # ── Chart 3: Regime breakdown ─────────────────────────────────────────
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.patch.set_facecolor(BG)
    fig3.suptitle(
        f"Strategy C — Regime Breakdown  |  {symbol}",
        fontsize=12, fontweight="bold", color="white",
    )
    ax_cnt, ax_pf, ax_eq, ax_net = (axes3[0, 0], axes3[0, 1],
                                     axes3[1, 0], axes3[1, 1])
    for ax in axes3.flat:
        _style(ax)

    regimes      = [r for r in REGIME_ORDER if r in regime_metrics]
    r_colors     = [REGIME_COLORS[r] for r in regimes]
    trade_counts = [regime_metrics[r]["n_trades"] for r in regimes]
    pf_vals      = [regime_metrics[r]["profit_factor"] for r in regimes]
    net_vals     = [regime_metrics[r]["net_profit"] for r in regimes]
    x            = np.arange(len(regimes))

    # Trade count
    bars = ax_cnt.bar(x, trade_counts, color=r_colors, edgecolor="#111", width=0.55)
    ax_cnt.set_xticks(x); ax_cnt.set_xticklabels(regimes, color="white")
    ax_cnt.set_title("Trade Count by Regime", fontsize=10)
    ax_cnt.set_ylabel("# Trades")
    for bar, v in zip(bars, trade_counts):
        ax_cnt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(v), ha="center", va="bottom", color="white", fontsize=9)

    # Profit Factor
    bars = ax_pf.bar(x, pf_vals, color=r_colors, edgecolor="#111", width=0.55)
    ax_pf.axhline(1.0, color="white", lw=0.8, ls="--", label="Break-even (PF=1)")
    ax_pf.axhline(1.2, color="#FFD700", lw=0.8, ls=":", label="Target PF=1.2")
    ax_pf.set_xticks(x); ax_pf.set_xticklabels(regimes, color="white")
    ax_pf.set_title("Profit Factor by Regime", fontsize=10)
    ax_pf.set_ylabel("Profit Factor")
    ax_pf.legend(**leg_kw)
    for bar, v in zip(bars, pf_vals):
        ax_pf.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{v:.2f}", ha="center", va="bottom", color="white", fontsize=9)

    # Equity curves by regime
    for regime in regimes:
        m = regime_metrics[regime]
        if len(m["equity"]) > 1:
            ax_eq.plot(m["equity"], color=REGIME_COLORS[regime],
                       lw=1.5, label=f"{regime}  (n={m['n_trades']})")
    # Overlay what-if
    if len(m_whatif["equity"]) > 1:
        ax_eq.plot(m_whatif["equity"], color="#E040FB", lw=1.8, ls="-.",
                   label=f"Non-Ranging Only  (n={m_whatif['n_trades']})")
    ax_eq.axhline(CONFIG["STARTING_CAPITAL"], color="gray", lw=0.6, ls=":")
    ax_eq.set_title("Equity Curve by Regime  (+ What-If)", fontsize=10)
    ax_eq.set_ylabel("Portfolio Value ($)")
    ax_eq.legend(**leg_kw)

    # Net profit per regime
    net_colors = [REGIME_COLORS[r] for r in regimes]
    bars = ax_net.bar(x, net_vals, color=net_colors, edgecolor="#111", width=0.55)
    ax_net.axhline(0, color="white", lw=0.8, ls="--")
    ax_net.set_xticks(x); ax_net.set_xticklabels(regimes, color="white")
    ax_net.set_title("Net Profit ($) by Regime", fontsize=10)
    ax_net.set_ylabel("Net Profit ($)")
    for bar, v in zip(bars, net_vals):
        ypos = max(v, 0) + abs(max(net_vals, default=1) * 0.01)
        ax_net.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"${v:,.0f}", ha="center", va="bottom", color="white", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p3 = os.path.join(CONFIG["OUTPUT_FOLDER"], f"{safe}_r003_regime_breakdown.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig3)
    saved.append(p3)

    return saved


# =============================================================================
# SECTION 10 — TRADE LOG EXPORT
# =============================================================================

def save_trade_log(trades_a: list, trades_b: list, trades_c: list,
                   symbol: str) -> str:
    all_trades = trades_a + trades_b + trades_c
    if not all_trades:
        return ""
    safe = symbol.replace("-", "_")
    df   = pd.DataFrame(all_trades)
    cols = ["label", "regime", "entry_time", "exit_time",
            "entry_price", "exit_price", "stop_loss", "take_profit",
            "pnl", "r_multiple", "fees", "spread_cost", "sl_slippage",
            "holding_minutes", "funding_windows_crossed", "win", "exit_type"]
    df = df[[c for c in cols if c in df.columns]]
    os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)
    path = os.path.join(CONFIG["OUTPUT_FOLDER"], f"{safe}_r003_trade_log.csv")
    df.to_csv(path, index=False)
    return path


# =============================================================================
# SECTION 11 — REPORT GENERATION
# =============================================================================

def _row(label: str, a, b, c, fmt="") -> None:
    def _f(v):
        if isinstance(v, float):
            if fmt == "$":   return f"${v:>10,.2f}"
            if fmt == "%":   return f"{v:>10.2%}"
            if fmt == "r":   return f"{v:>+10.3f}R"
            if fmt == "pf":  return f"{v:>10.3f}"
            if fmt == "min": return f"{v:>8.0f} min"
            return f"{v:>10.3f}"
        return f"{str(v):>11}"
    print(f"  {label:<26} {_f(a)}  {_f(b)}  {_f(c)}")


def _row4(label: str, vals: list, fmts: list) -> None:
    """Print a row with a variable number of columns."""
    def _f(v, fmt):
        if isinstance(v, (int, float)):
            v = float(v)
            if fmt == "$":   return f"${v:>9,.2f}"
            if fmt == "%":   return f"{v:>9.2%}"
            if fmt == "r":   return f"{v:>+9.3f}R"
            if fmt == "pf":  return f"{v:>9.3f}"
            if fmt == "n":   return f"{int(v):>9d}"
            return f"{v:>9.3f}"
        return f"{str(v):>10}"
    cells = "  ".join(_f(v, f) for v, f in zip(vals, fmts))
    print(f"  {label:<24}  {cells}")


def print_report(symbol: str,
                 m_a: dict, m_b: dict, m_c: dict,
                 mc_b: dict, mc_c: dict,
                 regime_metrics: dict[str, dict],
                 m_whatif: dict,
                 oos_start: str, oos_end: str,
                 n_days: int, chart_paths: list) -> None:

    S  = "=" * 78
    S2 = "-" * 78

    print(f"\n{S}")
    print("  QUANTLAB AI — RESEARCH #003")
    print("  Hypothesis: Does FVG+Slope have edge only in trending regimes?")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(S)
    print(f"  Symbol          : {symbol}")
    print(f"  Timeframe       : {CONFIG['TIMEFRAME']}")
    print(f"  EMA Length      : {CONFIG['EMA_LENGTH']}")
    print(f"  Slope Lookback  : {CONFIG['SLOPE_LOOKBACK']} bars")
    print(f"  ADX Length      : {CONFIG['ADX_LENGTH']}")
    print(f"  ADX Trending ≥  : {CONFIG['ADX_TRENDING']}")
    print(f"  ADX Weak Trend ≥: {CONFIG['ADX_WEAK']}")
    print(f"  Risk:Reward     : 1:{CONFIG['RISK_REWARD']:.1f}")
    print(f"  Taker Fee       : {CONFIG['TAKER_FEE']:.3%}  |  "
          f"Spread: {CONFIG['SPREAD']:.3%}  |  "
          f"SL Slip: {CONFIG['SL_SLIPPAGE']:.3%}")
    print(S)
    print(f"  OOS: {oos_start} → {oos_end}  ({n_days} calendar days)")
    print(S)

    # ── Three-strategy comparison (continuity from #002) ──────────────────
    print(f"\n  {'─' * 70}")
    print("  STRATEGY COMPARISON (unchanged from Research #002)")
    print(f"  {'─' * 70}")
    H = f"  {'Metric':<26} {'A  EMA Only':>11}  {'B  FVG+EMA':>11}  {'C  FVG+Slope':>11}"
    print(H)
    print(f"  {S2[2:]}")
    _row("Trades",         float(m_a["n_trades"]),  float(m_b["n_trades"]),  float(m_c["n_trades"]))
    _row("Win Rate",       m_a["win_rate"],          m_b["win_rate"],          m_c["win_rate"],         "%")
    _row("Profit Factor",  m_a["profit_factor"],     m_b["profit_factor"],     m_c["profit_factor"],    "pf")
    _row("Expectancy",     m_a["expectancy_r"],      m_b["expectancy_r"],      m_c["expectancy_r"],     "r")
    _row("Net Profit",     m_a["net_profit"],        m_b["net_profit"],        m_c["net_profit"],       "$")
    _row("Max Drawdown",   m_a["max_drawdown"],      m_b["max_drawdown"],      m_c["max_drawdown"],     "%")
    _row("Sharpe (approx)",m_a["sharpe"],            m_b["sharpe"],            m_c["sharpe"],           "pf")
    _row("MC Prob Profit",
         mc_b.get("prob_profit", 0.0),
         mc_b["prob_profit"],
         mc_c["prob_profit"], "%")
    print(f"  {S2[2:]}")

    # ── Regime breakdown table (Strategy C trades only) ───────────────────
    print(f"\n  {'─' * 70}")
    print("  REGIME BREAKDOWN — STRATEGY C TRADES")
    print(f"  ADX({CONFIG['ADX_LENGTH']}): "
          f"Trending≥{CONFIG['ADX_TRENDING']}  "
          f"Weak Trend≥{CONFIG['ADX_WEAK']}  "
          f"Ranging<{CONFIG['ADX_WEAK']}")
    print(f"  {'─' * 70}")

    reg_header = (f"  {'Metric':<24}  "
                  f"{'Trending':>10}  "
                  f"{'Weak Trend':>10}  "
                  f"{'Ranging':>10}  "
                  f"{'ALL (C)':>10}")
    print(reg_header)
    print(f"  {'-' * 70}")

    def _rval(key, fmt):
        vals = [regime_metrics[r][key] if r in regime_metrics else 0.0
                for r in REGIME_ORDER]
        vals.append(m_c[key])
        _row4(key.replace("_", " ").title(), vals,
              [fmt, fmt, fmt, fmt])

    _row4("Trades", [regime_metrics[r]["n_trades"] if r in regime_metrics else 0
                     for r in REGIME_ORDER] + [m_c["n_trades"]],
          ["n", "n", "n", "n"])
    _row4("Win Rate",
          [regime_metrics[r]["win_rate"] if r in regime_metrics else 0.0
           for r in REGIME_ORDER] + [m_c["win_rate"]],
          ["%", "%", "%", "%"])
    _row4("Profit Factor",
          [regime_metrics[r]["profit_factor"] if r in regime_metrics else 0.0
           for r in REGIME_ORDER] + [m_c["profit_factor"]],
          ["pf", "pf", "pf", "pf"])
    _row4("Expectancy",
          [regime_metrics[r]["expectancy_r"] if r in regime_metrics else 0.0
           for r in REGIME_ORDER] + [m_c["expectancy_r"]],
          ["r", "r", "r", "r"])
    _row4("Net Profit",
          [regime_metrics[r]["net_profit"] if r in regime_metrics else 0.0
           for r in REGIME_ORDER] + [m_c["net_profit"]],
          ["$", "$", "$", "$"])
    _row4("Max Drawdown",
          [regime_metrics[r]["max_drawdown"] if r in regime_metrics else 0.0
           for r in REGIME_ORDER] + [m_c["max_drawdown"]],
          ["%", "%", "%", "%"])
    print(f"  {'-' * 70}")

    # Regime distribution of bars in OOS
    # (computed separately — see process_symbol)
    print()

    # ── What-if analysis ──────────────────────────────────────────────────
    print(f"  {'─' * 70}")
    print("  WHAT-IF ATTRIBUTION — Remove Ranging Trades")
    print(f"  {'─' * 70}")
    wh = m_whatif
    print(f"  Original Strategy C  :  {m_c['n_trades']:>4} trades  "
          f"PF {m_c['profit_factor']:.3f}  "
          f"Exp {m_c['expectancy_r']:+.3f}R  "
          f"Net ${m_c['net_profit']:,.2f}  "
          f"MDD {m_c['max_drawdown']:.2%}")
    print(f"  Non-Ranging Only     :  {wh['n_trades']:>4} trades  "
          f"PF {wh['profit_factor']:.3f}  "
          f"Exp {wh['expectancy_r']:+.3f}R  "
          f"Net ${wh['net_profit']:,.2f}  "
          f"MDD {wh['max_drawdown']:.2%}")
    ranging_n = regime_metrics.get("Ranging", _empty_metrics("Ranging"))["n_trades"]
    print(f"  Trades removed (Ranging): {ranging_n}  "
          f"({ranging_n / max(m_c['n_trades'], 1):.0%} of all C trades)")

    # ── Monte Carlo ───────────────────────────────────────────────────────
    print(f"\n  MONTE CARLO  ({CONFIG['MC_ITERATIONS']:,} iterations)")
    for lbl, mc in [("B  FVG + EMA200     ", mc_b),
                    ("C  FVG + EMA + Slope", mc_c)]:
        if len(mc["final_equities"]) > 0:
            print(f"    {lbl}  Med ${mc['median']:>8,.0f}  "
                  f"Worst5% ${mc['p5']:>8,.0f}  "
                  f"Best5% ${mc['p95']:>8,.0f}  "
                  f"ProbProfit {mc['prob_profit']:.1%}")
        else:
            print(f"    {lbl}  Insufficient trades.")

    # ── Research Questions ─────────────────────────────────────────────────
    print(f"\n{S}")
    print("  RESEARCH CONCLUSIONS — Research #003")
    print(S)

    rm_t  = regime_metrics.get("Trending",   _empty_metrics("Trending"))
    rm_wt = regime_metrics.get("Weak Trend", _empty_metrics("Weak Trend"))
    rm_r  = regime_metrics.get("Ranging",    _empty_metrics("Ranging"))

    def _ans(num: str, result: str, detail: str) -> None:
        icon = "✓" if result == "YES" else ("✗" if result == "NO" else "~")
        print(f"  {icon} Q{num}: {detail}")

    # Q1: Highest PF regime
    all_r = [(r, regime_metrics[r]["profit_factor"])
             for r in REGIME_ORDER if r in regime_metrics and
             regime_metrics[r]["n_trades"] > 0]
    if all_r:
        best_r = max(all_r, key=lambda x: x[1])
        _ans("1", "YES",
             f"Highest PF regime: {best_r[0]}  (PF {best_r[1]:.3f})")
    else:
        _ans("1", "NO", "Insufficient trades to rank regimes.")

    # Q2: Worst regime
    if all_r:
        worst_r = min(all_r, key=lambda x: x[1])
        _ans("2", "YES",
             f"Worst PF regime: {worst_r[0]}  "
             f"(PF {worst_r[1]:.3f}  Net ${regime_metrics[worst_r[0]]['net_profit']:,.0f})")

    # Q3: Are losses concentrated in ranging?
    ranging_net = rm_r["net_profit"]
    total_loss  = sum(regime_metrics[r]["net_profit"]
                      for r in REGIME_ORDER if r in regime_metrics
                      and regime_metrics[r]["net_profit"] < 0)
    pct_from_ranging = ranging_net / total_loss if total_loss < 0 else 0.0
    q3_yes = ranging_net < 0 and pct_from_ranging > 0.40
    _ans("3", "YES" if q3_yes else "NO",
         f"Losses concentrated in Ranging?  "
         f"Ranging net ${ranging_net:,.0f} = "
         f"{pct_from_ranging:.0%} of total losses.")

    # Q4: Positive expectancy in trending?
    q4_yes = rm_t["expectancy_r"] > 0 and rm_t["n_trades"] >= 5
    _ans("4", "YES" if q4_yes else "NO",
         f"Positive expectancy in Trending?  "
         f"Exp {rm_t['expectancy_r']:+.3f}R  "
         f"PF {rm_t['profit_factor']:.3f}  "
         f"n={rm_t['n_trades']} trades")

    # Q5: What-if result vs original
    whatif_better = (wh["profit_factor"] > m_c["profit_factor"] + 0.05 or
                     wh["expectancy_r"]  > m_c["expectancy_r"]  + 0.05)
    q5_yes = whatif_better and wh["n_trades"] >= 10
    _ans("5", "YES" if q5_yes else "NO",
         f"Non-Ranging filter improves results?  "
         f"C: PF {m_c['profit_factor']:.3f} Exp {m_c['expectancy_r']:+.3f}R  →  "
         f"Non-Ranging: PF {wh['profit_factor']:.3f} Exp {wh['expectancy_r']:+.3f}R")

    # ── Final verdict ──────────────────────────────────────────────────────
    print()
    trending_pf_above_target  = rm_t["profit_factor"] > 1.2 and rm_t["n_trades"] >= 10
    ranging_pf_below_breakeven = rm_r["profit_factor"] < 1.0
    whatif_positive_exp        = wh["expectancy_r"] > 0.0 and wh["n_trades"] >= 10

    if trending_pf_above_target and ranging_pf_below_breakeven:
        verdict = "REGIME-DEPENDENT EDGE CONFIRMED"
        print(f"  HYPOTHESIS VERDICT:  ★★★★★  {verdict}")
        print()
        print("  ✓ Trending regime produces PF > 1.2 — genuine measurable edge")
        print("  ✓ Ranging regime destroys profitability as hypothesised")
        print("  ✓ Future research should implement an objective ADX regime filter")
        print("  → NEXT: Research #004 — Add ADX regime filter as a hard entry gate")
        print("    (reject entries when ADX < threshold, test if it improves OOS PF)")
    elif q4_yes and not trending_pf_above_target:
        verdict = "WEAK SIGNAL — INSUFFICIENT EDGE"
        print(f"  HYPOTHESIS VERDICT:  ★★★☆☆  {verdict}")
        print()
        print("  ~ Trending regime shows positive expectancy but PF below 1.2")
        print("  ~ Edge exists directionally but is not statistically robust")
        print("  ~ Regime separation is real but magnitude is insufficient")
        print("  → NEXT: Research #004 — Test on wider history (18 months)")
        print("    to determine whether edge is real or a small-sample artifact")
    else:
        verdict = "NO REGIME-SPECIFIC EDGE DETECTED"
        print(f"  HYPOTHESIS VERDICT:  ★☆☆☆☆  {verdict}")
        print()
        print("  ✗ No regime produced consistent PF > 1.2")
        print("  ✗ Strategy lacks edge across all measured conditions")
        print("  ✗ FVG + EMA200 concept does not demonstrate statistically")
        print("    meaningful edge on 1H crypto perpetuals in this study period")
        print()
        print("  RECOMMENDATION: Reject FVG-based strategies for this asset class.")
        print("  → NEXT CONCEPT: Test momentum breakout (price closes above")
        print("    N-bar high in trending regime) — different entry logic,")
        print("    same objective regime filter and risk framework.")

    # ── Modelling assumptions ──────────────────────────────────────────────
    print()
    print(f"  {'─' * 50}")
    print("  MODELLING ASSUMPTIONS  (unchanged from Research #001)")
    print("  ✓ Next-candle execution  (no look-ahead bias)")
    print("  ✓ Conservative SL-first  (if SL & TP both within same bar)")
    print("  ✓ Taker fees, spread, SL slippage all included")
    print(f"  ✓ Leverage capped at {CONFIG['MAX_LEVERAGE']:.0f}×")
    print("  ✗ Funding rate NOT included in PnL")

    if chart_paths:
        print()
        print("  OUTPUT FILES")
        for p in chart_paths:
            print(f"  → {p}")
    print(S)


# =============================================================================
# SECTION 12 — MAIN PIPELINE
# =============================================================================

def process_symbol(symbol: str) -> None:
    sep = "─" * 78
    print(f"\n{sep}\n  PROCESSING: {symbol}\n{sep}")

    # 1. Data (reuses Research #002 cache)
    df = get_data(symbol)
    n  = len(df)
    print(f"  Total candles : {n:,}")

    warm_up = CONFIG["EMA_LENGTH"] * 3 + CONFIG["SLOPE_LOOKBACK"] + CONFIG["ADX_LENGTH"] * 3
    if n < warm_up:
        print(f"  [SKIP] Need ≥ {warm_up:,} candles. Got {n:,}.")
        return

    # 2. Indicators (adds EMA200, slope, ADX, regime column)
    df = add_indicators(df)

    # 3. Train / OOS split — UNCHANGED
    split  = int(n * CONFIG["TRAIN_RATIO"])
    df_oos = df.iloc[split:].reset_index(drop=True)

    oos_start = str(df_oos["datetime"].iloc[0].date())
    oos_end   = str(df_oos["datetime"].iloc[-1].date())
    n_days    = (df_oos["datetime"].iloc[-1] - df_oos["datetime"].iloc[0]).days

    print(f"  Train : {df['datetime'].iloc[0].date()} → "
          f"{df['datetime'].iloc[split-1].date()} ({split:,} bars)")
    print(f"  OOS   : {oos_start} → {oos_end} "
          f"({len(df_oos):,} bars / {n_days} days)")

    # Regime distribution of OOS bars
    regime_bar_counts = df_oos["regime"].value_counts()
    total_oos = len(df_oos)
    print(f"  OOS regime distribution:")
    for r in REGIME_ORDER:
        cnt = regime_bar_counts.get(r, 0)
        print(f"    {r:<12}: {cnt:>5} bars  ({cnt/total_oos:.0%})")

    # Signal diagnostics
    sig_c = strategy_fvg_ema_slope(df_oos).sum()
    print(f"  OOS Strategy C signals : {sig_c:,}")

    # 4. Backtests — UNCHANGED engine; Strategy C now tags regime per trade
    print("\n  Running Strategy A (EMA200 Only)...")
    res_a = run_backtest(df_oos, strategy_ema_only,      "A  EMA200 Only")

    print("  Running Strategy B (FVG + EMA200)...")
    res_b = run_backtest(df_oos, strategy_fvg_ema,       "B  FVG + EMA200")

    print("  Running Strategy C (FVG + EMA200 + Slope + Regime Tag)...")
    res_c = run_backtest(df_oos, strategy_fvg_ema_slope, "C  FVG+EMA+Slope")

    print(f"  Trades: A={len(res_a['trades'])}  "
          f"B={len(res_b['trades'])}  C={len(res_c['trades'])}")

    # 5. Strategy-level metrics
    m_a = compute_metrics(res_a["trades"], "A  EMA200 Only")
    m_b = compute_metrics(res_b["trades"], "B  FVG + EMA200")
    m_c = compute_metrics(res_c["trades"], "C  FVG+EMA+Slope")

    # 6. Regime breakdown — Strategy C only
    regime_metrics = compute_regime_breakdown(res_c["trades"])
    for r in REGIME_ORDER:
        rm = regime_metrics[r]
        print(f"  Regime [{r:<11}]: "
              f"{rm['n_trades']:>3} trades  "
              f"PF {rm['profit_factor']:.2f}  "
              f"Exp {rm['expectancy_r']:+.2f}R  "
              f"Net ${rm['net_profit']:>8,.0f}")

    # 7. What-if — remove ranging trades
    m_whatif = compute_whatif(res_c["trades"])
    print(f"  What-if (Non-Ranging):  "
          f"{m_whatif['n_trades']:>3} trades  "
          f"PF {m_whatif['profit_factor']:.2f}  "
          f"Exp {m_whatif['expectancy_r']:+.2f}R  "
          f"Net ${m_whatif['net_profit']:>8,.0f}")

    # 8. Monte Carlo (B and C)
    print(f"  Running Monte Carlo ({CONFIG['MC_ITERATIONS']:,} iterations × 2)...")
    mc_b = monte_carlo(m_b["pnls"], CONFIG["MC_ITERATIONS"])
    mc_c = monte_carlo(m_c["pnls"], CONFIG["MC_ITERATIONS"])

    # 9. Trade log
    log = save_trade_log(res_a["trades"], res_b["trades"], res_c["trades"], symbol)
    if log:
        print(f"  Trade log → {log}")

    # 10. Charts
    print("  Generating charts...")
    paths = plot_results(m_a, m_b, m_c, mc_b, mc_c,
                         regime_metrics, m_whatif,
                         symbol, oos_start, oos_end)
    if log:
        paths.append(log)

    # 11. Report
    print_report(symbol, m_a, m_b, m_c, mc_b, mc_c,
                 regime_metrics, m_whatif,
                 oos_start, oos_end, n_days, paths)


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║              QUANTLAB AI — RESEARCH #003                                  ║
║   Hypothesis: FVG + EMA200 Slope has edge only in trending regimes        ║
╚═══════════════════════════════════════════════════════════════════════════╝

  Strategy A : EMA200 Bullish Crossover Only          (benchmark)
  Strategy B : EMA200 + Bullish FVG                   (Research #001)
  Strategy C : EMA200 + Positive Slope + Bullish FVG  (Research #002)

  NEW in #003:
    Every Strategy C trade is tagged with ADX-based market regime.
    Results are grouped by regime (Trending / Weak Trend / Ranging).
    A what-if analysis removes ranging trades — attribution only, not optimisation.

  Engine, fees, spread, SL, TP, slope logic, train/test split : UNCHANGED
  Data Source : OKX Public REST API (cached from Research #001/002)
  Evaluation  : Out-of-Sample only (last 30%%)
""")

    print("  ACTIVE CONFIGURATION")
    print(f"  {'─' * 50}")
    for sym in CONFIG["SYMBOLS"]:
        print(f"  Symbol         : {sym}")
    print(f"  Timeframe      : {CONFIG['TIMEFRAME']}")
    print(f"  History        : {CONFIG['MONTHS_HISTORY']} months")
    print(f"  EMA Length     : {CONFIG['EMA_LENGTH']}")
    print(f"  Slope Lookback : {CONFIG['SLOPE_LOOKBACK']} bars")
    print(f"  ADX Length     : {CONFIG['ADX_LENGTH']}")
    print(f"  ADX Trending ≥ : {CONFIG['ADX_TRENDING']}")
    print(f"  ADX Weak Trend≥: {CONFIG['ADX_WEAK']}")
    print(f"  Risk:Reward    : 1:{CONFIG['RISK_REWARD']}")
    print(f"  Taker Fee      : {CONFIG['TAKER_FEE']:.3%}")
    print(f"  Spread         : {CONFIG['SPREAD']:.3%}")
    print(f"  SL Slippage    : {CONFIG['SL_SLIPPAGE']:.3%}")
    print(f"  Max Leverage   : {CONFIG['MAX_LEVERAGE']:.0f}×")
    print(f"  Capital        : ${CONFIG['STARTING_CAPITAL']:,.0f}")
    print(f"  Risk/Trade     : {CONFIG['RISK_PER_TRADE_PCT']:.1%}")
    print(f"  Cache          : {CONFIG['CACHE_FOLDER']}/")
    print(f"  Output         : {CONFIG['OUTPUT_FOLDER']}/")

    random.seed(42)
    np.random.seed(42)

    for sym in CONFIG["SYMBOLS"]:
        try:
            process_symbol(sym)
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {sym}: {exc}")
            traceback.print_exc()

    print(f"\n  Research #003 complete.")
    print(f"  Results saved to: {CONFIG['OUTPUT_FOLDER']}/\n")


if __name__ == "__main__":
    main()
