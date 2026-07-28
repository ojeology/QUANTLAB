"""
=============================================================================
QUANTLAB AI – RESEARCH #004
Objective: Multi-hypothesis tester comparing six independent trading concepts
           on identical data, engine, costs, and reporting.

Framework design:
  • Only the strategy signal function changes between hypotheses.
  • Backtest engine, fees, spread, SL, TP, position sizing, train/test split
    are locked and identical for every strategy.
  • Results are ranked in a leaderboard and appended to a persistent journal.

Strategies tested:
  1  FVG + EMA200 + Slope        Research #002/003 baseline
  2  Liquidity Sweep Reversal    Price sweeps a prior low then reclaims it
  3  Break of Structure          N-bar high breakout above prior resistance
  4  VWAP Pullback               Price tags rolling VWAP from above
  5  Opening Range Breakout      UTC-day first-4-hour range breakout
  6  Volatility Compression      ATR compression followed by range breakout

All strategies tested on BTC, ETH, SOL perpetuals, 1H candles, OOS only.
=============================================================================
"""

import os
import csv
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
# CONFIGURATION — locked; do not change between strategies
# =============================================================================

CONFIG = {
    "SYMBOLS":   ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],
    "TIMEFRAME": "1H",
    "MONTHS_HISTORY": 9,

    # Core strategy parameters (shared, unchanged)
    "EMA_LENGTH":   200,
    "RISK_REWARD":  2.0,
    "FVG_MULT":     1.0001,

    # Execution costs — NEVER changed
    "TAKER_FEE":   0.0005,
    "SPREAD":      0.0002,
    "SL_SLIPPAGE": 0.0003,
    "MIN_SL_PCT":  0.001,
    "MAX_LEVERAGE": 5.0,

    # Capital model — NEVER changed
    "STARTING_CAPITAL":   10_000.0,
    "RISK_PER_TRADE_PCT": 0.01,

    # Train / OOS split — NEVER changed
    "TRAIN_RATIO": 0.70,

    # Monte Carlo
    "MC_ITERATIONS": 1000,

    # Output
    "CACHE_FOLDER":   "quantlab_cache",
    "OUTPUT_FOLDER":  "quantlab_output",
    "JOURNAL_FILE":   "quantlab_output/research_journal.csv",

    # API
    "API_DELAY":     0.2,
    "OKX_PAGE_LIMIT": 100,

    # ── Research #002: EMA slope ──────────────────────────────────────────
    "SLOPE_LOOKBACK": 10,

    # ── Research #003: ADX regime ────────────────────────────────────────
    "ADX_LENGTH":   14,
    "ADX_TRENDING": 25,
    "ADX_WEAK":     20,

    # ── Research #004: strategy-specific indicator parameters ────────────
    # Liquidity Sweep Reversal
    "LSR_LOOKBACK": 5,      # bars to look back for prior swing lows

    # Break of Structure
    "BOS_LOOKBACK": 20,     # N-bar high defines the structure level

    # VWAP Pullback
    "VWAP_BARS": 24,        # rolling bar window for VWAP (24 × 1H ≈ 1 day)

    # Opening Range Breakout
    "ORB_HOURS": 4,         # first N UTC hours define the daily opening range

    # Volatility Compression Breakout
    "VCB_ATR_LENGTH":   14,  # ATR period
    "VCB_ATR_WINDOW":   50,  # rolling window to define "compressed" ATR
    "VCB_ATR_PCTILE":   30,  # ATR is "compressed" when below this percentile
    "VCB_BREAK_BARS":   10,  # N-bar high for compression breakout
}

RESEARCH_ID = "R004"


# =============================================================================
# SECTION 1 — DATA DOWNLOAD  (unchanged)
# =============================================================================

OKX_HISTORY_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
CANDLE_COLS     = ["ts", "open", "high", "low", "close", "vol",
                   "volCcy", "volCcyQuote", "confirm"]


def _parse_candles(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = pd.to_numeric(df[col])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return (df[["datetime", "open", "high", "low", "close", "vol"]]
            .sort_values("datetime").reset_index(drop=True))


def _fetch_page(symbol, bar, after_ms=None, use_history=True):
    url    = OKX_HISTORY_URL if use_history else OKX_CANDLES_URL
    params = {"instId": symbol, "bar": bar, "limit": CONFIG["OKX_PAGE_LIMIT"]}
    if after_ms is not None:
        params["after"] = str(after_ms)
    try:
        r    = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get("code") == "0":
            return data.get("data", [])
    except Exception as e:
        print(f"  [WARN] API error: {e}")
    return []


def download_symbol(symbol, bar, months, since_ms=None):
    now_ms    = int(time.time() * 1000)
    target_ms = int(months * 30.44 * 24 * 3600 * 1000)
    cutoff_ms = since_ms if since_ms else now_ms - target_ms

    if since_ms:
        print(f"  Fetching new candles for {symbol} since "
              f"{datetime.fromtimestamp(since_ms/1000, tz=timezone.utc).date()}...")
    else:
        print(f"  Downloading {symbol} ({bar}) — {months} months...")

    all_rows, after_ms_cursor, pages = [], None, 0
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

    df        = _parse_candles(all_rows)
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df        = df[df["datetime"] >= cutoff_dt]
    return (df.drop_duplicates("datetime")
              .sort_values("datetime")
              .reset_index(drop=True))


# =============================================================================
# SECTION 2 — LOCAL CACHE  (unchanged)
# =============================================================================

def _cache_path(symbol, bar):
    safe   = symbol.replace("-", "_") + "_" + bar.replace("/", "")
    folder = CONFIG["CACHE_FOLDER"]
    os.makedirs(folder, exist_ok=True)
    try:
        import pyarrow  # noqa
        return os.path.join(folder, f"{safe}.parquet")
    except ImportError:
        return os.path.join(folder, f"{safe}.csv")


def save_cache(df, symbol, bar):
    path = _cache_path(symbol, bar)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def load_cache(symbol, bar):
    path = _cache_path(symbol, bar)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path) if path.endswith(".parquet") else \
         pd.read_csv(path, parse_dates=["datetime"])
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def get_data(symbol):
    bar    = CONFIG["TIMEFRAME"]
    print(f"\n[DATA] {symbol} ({bar})")
    cached = load_cache(symbol, bar)

    if cached is not None and len(cached) > 0:
        last_ts     = cached["datetime"].iloc[-1]
        bar_minutes = {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,
                       "1H":60,"2H":120,"4H":240,"6H":360,"12H":720,"1D":1440
                      }.get(bar, 60)
        gap = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 60 / bar_minutes
        if gap < 2:
            print(f"  Cache current ({len(cached):,} candles).")
            return cached
        print(f"  Cache found ({len(cached):,} candles). Fetching ~{gap:.0f} new...")
        since_ms = int(last_ts.timestamp() * 1000)
        new_df   = download_symbol(symbol, bar, months=0, since_ms=since_ms)
        if len(new_df) > 0:
            new_df = new_df[new_df["datetime"] > last_ts]
        if len(new_df) > 0:
            combined = (pd.concat([cached, new_df], ignore_index=True)
                        .drop_duplicates("datetime")
                        .sort_values("datetime")
                        .reset_index(drop=True))
            save_cache(combined, symbol, bar)
            print(f"  Appended {len(new_df):,} → {len(combined):,} total.")
            return combined
        print("  No new candles.")
        return cached

    df = download_symbol(symbol, bar, months=CONFIG["MONTHS_HISTORY"])
    save_cache(df, symbol, bar)
    print(f"  → {len(df):,} candles "
          f"({df['datetime'].iloc[0].date()} – {df['datetime'].iloc[-1].date()})")
    return df


# =============================================================================
# SECTION 3 — INDICATORS
# =============================================================================

def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def calc_atr(df, length):
    """Average True Range (Wilder smoothing)."""
    prev_c = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/length, adjust=False).mean()


def calc_adx(df, length):
    """Wilder's ADX."""
    prev_c   = df["close"].shift(1)
    tr       = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    up       = df["high"] - df["high"].shift(1)
    down     = df["low"].shift(1) - df["low"]
    plus_dm  = pd.Series(np.where((up > down) & (up > 0),   up,   0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    alpha    = 1.0 / length
    sm_tr    = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di  = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / sm_tr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / sm_tr.replace(0, np.nan)
    di_sum   = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100.0 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)


def _calc_orb(df):
    """
    Opening Range Breakout indicators.
    For each UTC calendar day, compute the high and low of the first ORB_HOURS
    hours (hours 0 … ORB_HOURS-1).  These are forward-filled onto all later
    bars of the same day so the engine can use them without look-ahead.
    Returns (orb_high Series, orb_low Series) aligned to df.index.
    """
    hours    = CONFIG["ORB_HOURS"]
    dts      = df["datetime"].dt
    df_tmp   = df.copy()
    df_tmp["_date"] = dts.date
    df_tmp["_hour"] = dts.hour

    # Build day-level ORB from the first ORB_HOURS candles of each UTC day
    orb_mask  = df_tmp["_hour"] < hours
    orb_candles = df_tmp[orb_mask].groupby("_date").agg(
        orb_high=("high", "max"),
        orb_low=("low",  "min"),
    )

    # Merge back — NaN for days without a full ORB (start of data)
    df_tmp = df_tmp.join(orb_candles, on="_date")

    # ORB values should only be visible AFTER the opening range period
    # (i.e. on bars where hour >= ORB_HOURS).  Blank out bars inside the range.
    in_range = df_tmp["_hour"] < hours
    df_tmp.loc[in_range, "orb_high"] = np.nan
    df_tmp.loc[in_range, "orb_low"]  = np.nan

    return df_tmp["orb_high"].values, df_tmp["orb_low"].values


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute every indicator needed across all six strategies."""
    df = df.copy()

    # ── Shared ────────────────────────────────────────────────────────────
    df["ema200"]      = calc_ema(df["close"], CONFIG["EMA_LENGTH"])
    df["prev_high"]   = df["high"].shift(1)
    df["prev_low"]    = df["low"].shift(1)
    df["prev_close"]  = df["close"].shift(1)
    df["high_2"]      = df["high"].shift(2)
    df["ema200_prev"] = df["ema200"].shift(1)

    # ── Strategy 1: EMA slope (Research #002) ─────────────────────────────
    n = CONFIG["SLOPE_LOOKBACK"]
    df["ema200_lag"]    = df["ema200"].shift(n)
    df["ema200_rising"] = df["ema200"] > df["ema200_lag"]

    # ── ADX (Research #003 regime tag) ────────────────────────────────────
    df["adx"] = calc_adx(df, CONFIG["ADX_LENGTH"])

    # ── Strategy 2: Liquidity Sweep Reversal ─────────────────────────────
    lb = CONFIG["LSR_LOOKBACK"]
    # Minimum low of the previous LSR_LOOKBACK bars (not including current)
    df["lsr_prior_low"] = df["low"].shift(1).rolling(lb).min()

    # ── Strategy 3: Break of Structure ────────────────────────────────────
    bos_lb = CONFIG["BOS_LOOKBACK"]
    # Highest high of the previous BOS_LOOKBACK bars (not including current)
    df["bos_prior_high"] = df["high"].shift(1).rolling(bos_lb).max()

    # ── Strategy 4: VWAP Pullback ─────────────────────────────────────────
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol     = df["vol"].replace(0, np.nan)
    vwap_w  = CONFIG["VWAP_BARS"]
    df["vwap"] = (
        (typical * vol).rolling(vwap_w).sum() /
         vol.rolling(vwap_w).sum()
    )
    df["prev_vwap"] = df["vwap"].shift(1)

    # ── Strategy 5: Opening Range Breakout ────────────────────────────────
    orb_h, orb_l          = _calc_orb(df)
    df["orb_high"]        = orb_h
    df["orb_low"]         = orb_l
    df["prev_orb_high"]   = pd.Series(orb_h, index=df.index).shift(1)
    df["orb_triggered"]   = False   # reset daily — managed in signal fn

    # ── Strategy 6: Volatility Compression Breakout ───────────────────────
    atr_len  = CONFIG["VCB_ATR_LENGTH"]
    atr_win  = CONFIG["VCB_ATR_WINDOW"]
    pctile   = CONFIG["VCB_ATR_PCTILE"]
    brk_bars = CONFIG["VCB_BREAK_BARS"]

    df["atr"]         = calc_atr(df, atr_len)
    df["atr_pctile"]  = df["atr"].rolling(atr_win).quantile(pctile / 100.0)
    df["compressed"]  = df["atr"] < df["atr_pctile"]
    df["vcb_range_h"] = df["high"].shift(1).rolling(brk_bars).max()

    return df


# =============================================================================
# SECTION 4 — STRATEGY SIGNAL FUNCTIONS
# Each function receives the indicators DataFrame and returns a boolean Series.
# True on bar i = signal fires; entry on bar i+1 open.
# Stop = low of bar i.  TP = entry + 2 × stop_dist.
# No look-ahead allowed.
# =============================================================================

def strategy_fvg_ema_slope(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 1 — Bullish FVG + EMA200 + Positive Slope  [Research #002 baseline]

    Conditions (all on bar close):
      1. 3-candle Bullish FVG:  low[i] > high[i-2] × FVG_MULT
      2. Trend filter:          close[i] > EMA200[i]
      3. Slope filter:          EMA200[i] > EMA200[i - SLOPE_LOOKBACK]
    """
    fvg   = df["low"]   > df["high_2"] * CONFIG["FVG_MULT"]
    trend = df["close"] > df["ema200"]
    slope = df["ema200_rising"]
    return fvg & trend & slope


def strategy_lsr(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 2 — Liquidity Sweep Reversal (LSR)

    Hypothesis: smart money hunts stops below prior swing lows then reverses.
    A bullish sweep occurs when the candle pierces below the recent range low
    but closes back above it — a failed breakdown / stop-hunt reversal.

    Conditions (all on bar close):
      1. Sweep:    low[i]   < lsr_prior_low[i]   (wick below prior N-bar low)
      2. Reclaim:  close[i] > lsr_prior_low[i]   (closes back above the swept level)
      3. Bullish:  close[i] > open[i]             (bar closes green — rejection candle)
      4. Trend:    close[i] > EMA200[i]           (in uptrend)

    Stop: low of signal bar (the sweep wick).
    Entry: next bar open.
    """
    sweep   = df["low"]   < df["lsr_prior_low"]
    reclaim = df["close"] > df["lsr_prior_low"]
    bullish = df["close"] > df["open"]
    trend   = df["close"] > df["ema200"]
    return sweep & reclaim & bullish & trend


def strategy_bos(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 3 — Break of Structure (BoS)

    Hypothesis: closing above the prior N-bar high signals a structural break
    with continuation potential.

    Conditions (all on bar close):
      1. Structure break: close[i] > bos_prior_high[i]   (closes above N-bar resistance)
      2. Trend:           close[i] > EMA200[i]
      3. Slope:           ema200_rising (avoid flat EMA breakouts)

    Stop:  low of signal bar.
    Entry: next bar open.
    """
    structure_break = df["close"] > df["bos_prior_high"]
    trend           = df["close"] > df["ema200"]
    slope           = df["ema200_rising"]
    return structure_break & trend & slope


def strategy_vwap_pull(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 4 — VWAP Pullback

    Hypothesis: in an uptrend, a pullback to rolling VWAP followed by a close
    above it marks a high-probability continuation entry.

    Conditions (all on bar close):
      1. Touch:        low[i]  <= VWAP[i]           (bar reaches VWAP)
      2. Rejection:    close[i] > VWAP[i]           (closes back above VWAP)
      3. Prior bar above: prev_close[i] > prev_VWAP (was already above VWAP)
      4. Trend:        close[i] > EMA200[i]

    Stop: low of signal bar.
    Entry: next bar open.
    """
    touch    = df["low"]        <= df["vwap"]
    reject   = df["close"]       > df["vwap"]
    was_above = df["prev_close"] > df["prev_vwap"]
    trend    = df["close"]       > df["ema200"]
    valid    = df["vwap"].notna() & df["prev_vwap"].notna()
    return touch & reject & was_above & trend & valid


def strategy_orb(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 5 — Opening Range Breakout (ORB)

    Hypothesis: the first N UTC hours of each day define a range; a close above
    that range signals a directional move for the remainder of the session.

    Conditions (all on bar close):
      1. Breakout: close[i] > orb_high[i]   (closes above today's opening range high)
      2. Trend:    close[i] > EMA200[i]
      3. orb_high must be valid (not NaN — first day of data has no prior ORB)
      4. One signal per UTC day (tracked via orb_triggered; see engine logic)

    Stop: low of signal bar.
    Entry: next bar open.

    Note: the one-per-day filter is enforced via a stateful flag in the signal
    computation below to avoid re-triggering on the same breakout.
    """
    breakout = df["close"] > df["orb_high"]
    trend    = df["close"] > df["ema200"]
    valid    = df["orb_high"].notna()

    # One-signal-per-day filter: suppress subsequent bars if already triggered today
    raw   = breakout & trend & valid
    dates = df["datetime"].dt.date

    triggered   = pd.Series(False, index=df.index)
    last_sig_dt = None

    for idx in df.index:
        if raw.iloc[idx]:
            d = dates.iloc[idx]
            if d != last_sig_dt:
                triggered.iloc[idx] = True
                last_sig_dt = d

    return triggered


def strategy_vcb(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 6 — Volatility Compression Breakout (VCB)

    Hypothesis: periods of ATR compression (low volatility) precede explosive
    breakout moves; entering on the first close above the compression range high
    captures the expansion.

    Conditions (all on bar close):
      1. Compressed:  ATR[i] < ATR percentile-30 of last VCB_ATR_WINDOW bars
      2. Breakout:    close[i] > vcb_range_h[i]  (closes above prior VCB_BREAK_BARS high)
      3. Trend:       close[i] > EMA200[i]

    Stop: low of signal bar.
    Entry: next bar open.
    """
    compressed = df["compressed"]
    breakout   = df["close"] > df["vcb_range_h"]
    trend      = df["close"] > df["ema200"]
    valid      = df["atr_pctile"].notna() & df["vcb_range_h"].notna()
    return compressed & breakout & trend & valid


# =============================================================================
# Strategy registry — add new hypotheses here; nothing else changes
# =============================================================================

STRATEGIES = {
    "FVG+Slope":       strategy_fvg_ema_slope,
    "Liq.Sweep":       strategy_lsr,
    "Brk.Structure":   strategy_bos,
    "VWAP.Pullback":   strategy_vwap_pull,
    "ORB":             strategy_orb,
    "Volatility.Comp": strategy_vcb,
}


# =============================================================================
# SECTION 5 — BACKTEST ENGINE  (locked — identical to Research #001-003)
# =============================================================================

def run_backtest(df: pd.DataFrame, signal_fn, label: str) -> dict:
    """
    Event-driven position simulator.  Engine is identical across all strategies.
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
    trades        = []

    for i in range(1, len(df)):
        bar = df.iloc[i]

        if in_position:
            hi, lo = bar["high"], bar["low"]
            sl_hit = lo <= stop_loss
            tp_hit = hi >= take_profit

            if sl_hit or tp_hit:
                exit_price = (stop_loss * (1.0 - slp_rate)) if sl_hit else take_profit
                exit_type  = "SL" if sl_hit else "TP"

                gross_pnl  = (exit_price - entry_price) * position_size
                ne         = entry_price * position_size
                nx         = exit_price  * position_size
                cost_fee   = (ne + nx) * fee_rate
                cost_spd   = (ne + nx) * spd_rate
                cost_slip  = (stop_loss - exit_price) * position_size if exit_type == "SL" else 0.0
                net_pnl    = gross_pnl - cost_fee - cost_spd - cost_slip

                sl_dist    = entry_price - stop_loss
                r_mult     = (exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0
                hold_mins  = (i - entry_idx) * _bar_minutes()

                trades.append({
                    "label":       label,
                    "entry_time":  entry_time,
                    "exit_time":   bar["datetime"],
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "stop_loss":   stop_loss,
                    "take_profit": take_profit,
                    "pnl":         net_pnl,
                    "r_multiple":  r_mult,
                    "fees":        cost_fee,
                    "spread_cost": cost_spd,
                    "sl_slippage": cost_slip,
                    "holding_minutes": hold_mins,
                    "funding_windows_crossed": int(hold_mins / 480),
                    "win":       exit_type == "TP",
                    "exit_type": exit_type,
                })
                in_position = False
            continue

        if signals.iloc[i - 1]:
            prev_bar = df.iloc[i - 1]
            ep       = bar["open"]
            sl       = prev_bar["low"]
            sl_dist  = ep - sl

            if sl_dist <= 0 or sl_dist / ep < min_sl:
                continue

            tp           = ep + rr * sl_dist
            risk_dollars = capital * risk_frac
            pos_size     = min(risk_dollars / sl_dist, (capital * max_lev) / ep)

            entry_price   = ep
            stop_loss     = sl
            take_profit   = tp
            position_size = pos_size
            entry_time    = bar["datetime"]
            entry_idx     = i
            in_position   = True

    return {"trades": trades}


def _bar_minutes():
    return {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,
            "1H":60,"2H":120,"4H":240,"6H":360,"12H":720,"1D":1440
           }.get(CONFIG["TIMEFRAME"], 60)


# =============================================================================
# SECTION 6 — PERFORMANCE METRICS  (locked)
# =============================================================================

def compute_metrics(trades: list, label: str) -> dict:
    if not trades:
        return _empty_metrics(label)

    df   = pd.DataFrame(trades)
    pnls = df["pnl"].values
    wins = df["win"].values.astype(bool)
    rmul = df["r_multiple"].values

    n, n_win = len(pnls), int(wins.sum())
    n_los    = n - n_win

    gw = pnls[wins].sum()          if n_win else 0.0
    gl = abs(pnls[~wins].sum())    if n_los else 1e-9
    pf = gw / gl                   if gl > 0 else float("inf")
    wr = n_win / n

    exp_r  = wr * CONFIG["RISK_REWARD"] - (1.0 - wr)
    equity = CONFIG["STARTING_CAPITAL"] + np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    max_dd = dd.min()

    std    = np.std(pnls, ddof=1) if n > 1 else 0.0
    bpy    = (365 * 24 * 60) / _bar_minutes()
    sharpe = (pnls.mean() / std * math.sqrt(bpy * n / max(n, 1))
              if std > 0 else 0.0)

    return {
        "label":          label,
        "n_trades":       n,
        "net_profit":     float(pnls.sum()),
        "profit_factor":  pf,
        "win_rate":       wr,
        "avg_win":        pnls[wins].mean()  if n_win else 0.0,
        "avg_loss":       pnls[~wins].mean() if n_los else 0.0,
        "avg_trade":      pnls.mean(),
        "avg_r":          rmul.mean(),
        "expectancy_r":   exp_r,
        "largest_win":    pnls[wins].max()  if n_win else 0.0,
        "largest_loss":   pnls[~wins].min() if n_los else 0.0,
        "max_drawdown":   max_dd,
        "sharpe":         sharpe,
        "avg_hold_minutes":      df["holding_minutes"].mean(),
        "total_funding_windows": int(df["funding_windows_crossed"].sum()),
        "equity":    equity,
        "drawdown":  dd,
        "pnls":      pnls,
        "r_multiples": rmul,
        "trades_df": df,
    }


def _empty_metrics(label):
    return {
        "label": label, "n_trades": 0, "net_profit": 0.0,
        "profit_factor": 0.0, "win_rate": 0.0, "avg_win": 0.0,
        "avg_loss": 0.0, "avg_trade": 0.0, "avg_r": 0.0,
        "expectancy_r": 0.0, "largest_win": 0.0, "largest_loss": 0.0,
        "max_drawdown": 0.0, "sharpe": 0.0, "avg_hold_minutes": 0.0,
        "total_funding_windows": 0,
        "equity":     np.array([CONFIG["STARTING_CAPITAL"]]),
        "drawdown":   np.array([0.0]),
        "pnls":       np.array([]),
        "r_multiples": np.array([]),
        "trades_df":  pd.DataFrame(),
    }


# =============================================================================
# SECTION 7 — MONTE CARLO  (locked)
# =============================================================================

def monte_carlo(pnls: np.ndarray, n_iter: int = 1000) -> dict:
    if len(pnls) == 0:
        return {"median": 0.0, "p5": 0.0, "p95": 0.0,
                "prob_profit": 0.0, "final_equities": np.array([])}
    start  = CONFIG["STARTING_CAPITAL"]
    finals = np.array([start + np.random.permutation(pnls).sum()
                       for _ in range(n_iter)])
    return {
        "median":      float(np.median(finals)),
        "p5":          float(np.percentile(finals, 5)),
        "p95":         float(np.percentile(finals, 95)),
        "prob_profit": float((finals > start).mean()),
        "final_equities": finals,
    }


# =============================================================================
# SECTION 8 — LEADERBOARD  (Research #004)
# =============================================================================

def _verdict_from_metrics(m: dict, mc: dict) -> str:
    """Single-strategy verdict used in the leaderboard."""
    if m["n_trades"] < 10:
        return "INSUFFICIENT"
    pf   = m["profit_factor"]
    er   = m["expectancy_r"]
    mpp  = mc["prob_profit"]
    mdd  = abs(m["max_drawdown"])
    if pf >= 1.2 and er > 0.0 and mpp >= 0.55 and mdd < 0.30:
        return "PROMOTE"
    if pf >= 1.05 and er > 0.0:
        return "WEAK"
    return "REJECT"


def build_leaderboard(results: list[dict]) -> list[dict]:
    """
    Sort strategy results by a composite score:
      score = PF × 0.4 + (1 + expectancy_r) × 0.3 + MC_prob × 0.2 + (1 + MDD) × 0.1
    Higher = better.
    """
    def _score(r):
        pf  = min(r["profit_factor"], 5.0)         # cap outliers
        er  = max(r["expectancy_r"], -2.0)
        mpp = r["mc_prob_profit"]
        mdd = max(r["max_drawdown"], -1.0)          # already negative
        return pf * 0.40 + (1.0 + er) * 0.30 + mpp * 0.20 + (1.0 + mdd) * 0.10

    ranked = sorted(results, key=_score, reverse=True)
    for i, r in enumerate(ranked):
        r["rank"] = i + 1
    return ranked


# =============================================================================
# SECTION 9 — RESEARCH JOURNAL
# =============================================================================

JOURNAL_COLS = [
    "research_id", "run_date", "strategy_name", "symbol",
    "n_trades", "profit_factor", "expectancy_r", "win_rate",
    "net_profit", "max_drawdown", "sharpe", "mc_prob_profit",
    "avg_hold_minutes", "verdict",
]


def append_journal(rows: list[dict]) -> None:
    """Append experiment rows to the persistent CSV journal."""
    path     = CONFIG["JOURNAL_FILE"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_COLS,
                                extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def _journal_row(strategy_name: str, symbol: str,
                 m: dict, mc: dict, verdict: str) -> dict:
    return {
        "research_id":    RESEARCH_ID,
        "run_date":       datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "strategy_name":  strategy_name,
        "symbol":         symbol,
        "n_trades":       m["n_trades"],
        "profit_factor":  round(m["profit_factor"],   4),
        "expectancy_r":   round(m["expectancy_r"],    4),
        "win_rate":       round(m["win_rate"],         4),
        "net_profit":     round(m["net_profit"],       2),
        "max_drawdown":   round(m["max_drawdown"],     4),
        "sharpe":         round(m["sharpe"],           4),
        "mc_prob_profit": round(mc["prob_profit"],     4),
        "avg_hold_minutes": round(m["avg_hold_minutes"], 1),
        "verdict":        verdict,
    }


# =============================================================================
# SECTION 10 — VISUALISATIONS
# =============================================================================

_PALETTE = [
    "#4A90D9", "#FFB347", "#00C49A", "#FF4560",
    "#E040FB", "#FFD700", "#00D4FF", "#FF6B6B",
]


def plot_all_strategies(metrics_by_strat: dict,
                        symbol: str,
                        oos_start: str, oos_end: str) -> list:
    """
    Chart 1: Equity curves for all strategies on one plot.
    Chart 2: Leaderboard bar chart (PF, Expectancy, Net Profit).
    """
    os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)
    safe  = symbol.replace("-", "_")
    saved = []
    BG    = "#0F1117"

    def _style(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors="white", labelsize=8)
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
        ax.grid(True, alpha=0.2, color="#444")

    leg_kw = dict(fontsize=7, facecolor="#1A1D24",
                  edgecolor="#444", labelcolor="white")

    names = list(metrics_by_strat.keys())
    cols  = _PALETTE[:len(names)]

    # ── Chart 1: Equity Curves ────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"QuantLab AI Research #004 — {symbol}\n"
        f"All Strategies  |  OOS: {oos_start} → {oos_end}",
        fontsize=11, fontweight="bold", color="white", y=0.997,
    )
    for ax in (ax1, ax2):
        _style(ax)

    for name, col in zip(names, cols):
        m = metrics_by_strat[name]["metrics"]
        if len(m["equity"]) > 1:
            lbl = (f"{name:<20} PF {m['profit_factor']:.2f}  "
                   f"Exp {m['expectancy_r']:+.2f}R  "
                   f"n={m['n_trades']}")
            ax1.plot(m["equity"], color=col, lw=1.5, label=lbl)

    ax1.axhline(CONFIG["STARTING_CAPITAL"], color="gray", lw=0.6, ls=":")
    ax1.set_title("Equity Curves — All Strategies", fontsize=10)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(**leg_kw)

    for name, col in zip(names, cols):
        m = metrics_by_strat[name]["metrics"]
        if len(m["drawdown"]) > 1:
            ax2.fill_between(range(len(m["drawdown"])),
                             m["drawdown"] * 100, 0,
                             color=col, alpha=0.55, label=name)
    ax2.set_title("Drawdown (%)", fontsize=10)
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(**leg_kw)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = os.path.join(CONFIG["OUTPUT_FOLDER"],
                      f"{safe}_r004_equity_all.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    saved.append(p1)

    # ── Chart 2: Leaderboard bar chart ────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.patch.set_facecolor(BG)
    fig2.suptitle(f"Strategy Comparison — {symbol}", fontsize=11,
                  fontweight="bold", color="white")
    ax_pf, ax_exp, ax_net = axes2
    for ax in axes2:
        _style(ax)

    x  = np.arange(len(names))
    pf_vals  = [metrics_by_strat[n]["metrics"]["profit_factor"] for n in names]
    exp_vals = [metrics_by_strat[n]["metrics"]["expectancy_r"]  for n in names]
    net_vals = [metrics_by_strat[n]["metrics"]["net_profit"]     for n in names]

    def _bar_chart(ax, vals, title, ylabel, ref=None, ref_label=""):
        bar_colors = [cols[i] for i in range(len(names))]
        bars = ax.bar(x, vals, color=bar_colors, edgecolor="#111", width=0.6)
        if ref is not None:
            ax.axhline(ref, color="white", lw=0.8, ls="--", label=ref_label)
            ax.legend(**leg_kw)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(".", "\n") for n in names],
                           color="white", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
        for bar, v in zip(bars, vals):
            yoff = max(abs(v) * 0.02, 0.005)
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (yoff if v >= 0 else -yoff * 3),
                    f"{v:.2f}", ha="center", va="bottom",
                    color="white", fontsize=7)

    _bar_chart(ax_pf,  pf_vals,  "Profit Factor", "PF",
               ref=1.0, ref_label="Break-even")
    _bar_chart(ax_exp, exp_vals, "Expectancy (R)", "R",
               ref=0.0, ref_label="Zero expectancy")
    _bar_chart(ax_net, net_vals, "Net Profit ($)", "$")

    plt.tight_layout()
    p2 = os.path.join(CONFIG["OUTPUT_FOLDER"], f"{safe}_r004_leaderboard.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig2)
    saved.append(p2)

    return saved


def plot_monte_carlo_grid(metrics_by_strat: dict,
                          mc_by_strat: dict,
                          symbol: str) -> str:
    """Monte Carlo final-equity distribution for all strategies."""
    BG      = "#0F1117"
    names   = list(metrics_by_strat.keys())
    cols    = _PALETTE[:len(names)]
    n_strat = len(names)
    ncols   = 3
    nrows   = math.ceil(n_strat / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Monte Carlo — {symbol}  ({CONFIG['MC_ITERATIONS']:,} iter.)",
                 fontsize=11, fontweight="bold", color="white")
    axes_flat = np.array(axes).flatten() if nrows > 1 else np.array(axes)

    def _style(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors="white", labelsize=7)
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.grid(True, alpha=0.15, color="#444")

    for i, (name, col) in enumerate(zip(names, cols)):
        ax  = axes_flat[i]
        _style(ax)
        mc  = mc_by_strat[name]
        m   = metrics_by_strat[name]["metrics"]
        fe  = mc["final_equities"]
        if len(fe) > 1:
            nu   = len(np.unique(fe))
            bins = max(1, min(50, nu - 1))
            ax.hist(fe, bins=bins, color=col, alpha=0.75, edgecolor="#111")
        ax.axvline(CONFIG["STARTING_CAPITAL"], color="white", lw=1.2,
                   ls="--", label="Start")
        ax.axvline(mc["median"], color="#FFD700", lw=1.2,
                   label=f"Med ${mc['median']:,.0f}")
        ax.set_title(f"{name}  |  ProbProfit {mc['prob_profit']:.0%}  "
                     f"n={m['n_trades']}", fontsize=8, color="white")
        ax.legend(fontsize=6, facecolor="#1A1D24",
                  edgecolor="#444", labelcolor="white")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    safe = symbol.replace("-", "_")
    path = os.path.join(CONFIG["OUTPUT_FOLDER"], f"{safe}_r004_mc_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


# =============================================================================
# SECTION 11 — TRADE LOG
# =============================================================================

def save_trade_log(all_trades: list, symbol: str) -> str:
    if not all_trades:
        return ""
    safe = symbol.replace("-", "_")
    df   = pd.DataFrame(all_trades)
    cols = ["label", "entry_time", "exit_time", "entry_price", "exit_price",
            "stop_loss", "take_profit", "pnl", "r_multiple", "fees",
            "spread_cost", "sl_slippage", "holding_minutes",
            "funding_windows_crossed", "win", "exit_type"]
    df = df[[c for c in cols if c in df.columns]]
    os.makedirs(CONFIG["OUTPUT_FOLDER"], exist_ok=True)
    path = os.path.join(CONFIG["OUTPUT_FOLDER"], f"{safe}_r004_trade_log.csv")
    df.to_csv(path, index=False)
    return path


# =============================================================================
# SECTION 12 — REPORT
# =============================================================================

def print_leaderboard(ranked: list[dict], symbol: str,
                      oos_start: str, oos_end: str) -> None:
    S  = "=" * 100
    S2 = "-" * 100

    print(f"\n{S}")
    print(f"  QUANTLAB AI — RESEARCH #004   |   {symbol}")
    print(f"  Multi-Hypothesis Leaderboard  |  OOS: {oos_start} → {oos_end}")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(S)

    H = (f"  {'Rank':<5} {'Strategy':<22} {'Trades':>7} {'Win%':>7} "
         f"{'PF':>7} {'Exp(R)':>8} {'Net$':>10} {'MDD':>8} "
         f"{'Sharpe':>7} {'MC%':>7} {'Hold':>6} {'Verdict':<14}")
    print(H)
    print(f"  {S2[2:]}")

    for r in ranked:
        verdict = r["verdict"]
        star    = "★" if verdict == "PROMOTE" else ("·" if verdict == "WEAK" else " ")
        print(
            f"  {r['rank']:<5} "
            f"{r['strategy']:<22} "
            f"{r['n_trades']:>7} "
            f"{r['win_rate']:>7.1%} "
            f"{r['profit_factor']:>7.3f} "
            f"{r['expectancy_r']:>+8.3f}R "
            f"${r['net_profit']:>9,.0f} "
            f"{r['max_drawdown']:>8.2%} "
            f"{r['sharpe']:>7.2f} "
            f"{r['mc_prob_profit']:>7.1%} "
            f"{r['avg_hold_minutes']:>5.0f}h "
            f"{star} {verdict}"
        )

    print(f"  {S2[2:]}")
    print()

    # ── Research conclusions ───────────────────────────────────────────────
    promoted = [r for r in ranked if r["verdict"] == "PROMOTE"]
    weak     = [r for r in ranked if r["verdict"] == "WEAK"]
    rejected = [r for r in ranked if r["verdict"] in ("REJECT", "INSUFFICIENT")]
    best     = ranked[0] if ranked else None

    print(f"  RESEARCH CONCLUSIONS")
    print(f"  {'─' * 60}")

    if promoted:
        print(f"  PROMOTED strategies (PF≥1.2, positive expectancy, MC≥55%):")
        for r in promoted:
            print(f"    ★ {r['strategy']:<22}  PF {r['profit_factor']:.3f}  "
                  f"Exp {r['expectancy_r']:+.3f}R  "
                  f"MC {r['mc_prob_profit']:.1%}")
        print()
        top = promoted[0]
        print(f"  → NEXT: Research #005 — Run {top['strategy']} on 18-month history")
        print(f"    to confirm edge is persistent beyond this study window.")
    elif weak:
        print(f"  WEAK SIGNAL strategies (PF≥1.05, positive expectancy):")
        for r in weak:
            print(f"    · {r['strategy']:<22}  PF {r['profit_factor']:.3f}  "
                  f"Exp {r['expectancy_r']:+.3f}R")
        print()
        top = weak[0]
        print(f"  → NEXT: Research #005 — Combine {top['strategy']} with ADX regime filter")
        print(f"    (entry only when ADX≥25) to determine if trending-regime restriction")
        print(f"    upgrades WEAK to PROMOTE.")
    else:
        print(f"  All strategies REJECTED across {symbol}.")
        if best:
            print(f"  Best performer: {best['strategy']}  PF {best['profit_factor']:.3f}  "
                  f"Exp {best['expectancy_r']:+.3f}R")
        print()
        print(f"  → CONCLUSION: The tested entry concepts do not demonstrate edge on")
        print(f"    {symbol} 1H perpetuals in the current study period.")
        print(f"    Consider: (a) longer history, (b) different timeframe,")
        print(f"    (c) fundamentally different concept (e.g. mean-reversion).")

    print(S)


def print_strategy_detail(name: str, m: dict, mc: dict,
                          symbol: str) -> None:
    """Compact per-strategy detail block."""
    S2 = "─" * 70
    print(f"\n  ── {name} ──────────────────────────────────────")
    print(f"  Trades: {m['n_trades']}  WR: {m['win_rate']:.1%}  "
          f"PF: {m['profit_factor']:.3f}  Exp: {m['expectancy_r']:+.3f}R  "
          f"Net: ${m['net_profit']:,.2f}  MDD: {m['max_drawdown']:.2%}")
    print(f"  Sharpe: {m['sharpe']:.2f}  AvgHold: {m['avg_hold_minutes']:.0f}min  "
          f"MC-ProbProfit: {mc['prob_profit']:.1%}  "
          f"MC-Med: ${mc['median']:,.0f}")


# =============================================================================
# SECTION 13 — MAIN PIPELINE
# =============================================================================

def process_symbol(symbol: str) -> tuple[list, list]:
    """
    Run all strategies on one symbol.
    Returns (leaderboard_rows, journal_rows).
    """
    sep = "─" * 90
    print(f"\n{sep}\n  PROCESSING: {symbol}\n{sep}")

    df = get_data(symbol)
    n  = len(df)
    print(f"  Total candles : {n:,}")

    warm_up = CONFIG["EMA_LENGTH"] * 3 + max(
        CONFIG["SLOPE_LOOKBACK"],
        CONFIG["BOS_LOOKBACK"],
        CONFIG["VWAP_BARS"],
        CONFIG["VCB_ATR_WINDOW"] + CONFIG["VCB_ATR_LENGTH"],
    ) + CONFIG["ADX_LENGTH"] * 3
    if n < warm_up:
        print(f"  [SKIP] Need ≥ {warm_up:,} candles. Got {n:,}.")
        return [], []

    # Indicators — single pass, all strategies share the same df
    df = add_indicators(df)

    # Train / OOS split — UNCHANGED
    split  = int(n * CONFIG["TRAIN_RATIO"])
    df_oos = df.iloc[split:].reset_index(drop=True)

    oos_start = str(df_oos["datetime"].iloc[0].date())
    oos_end   = str(df_oos["datetime"].iloc[-1].date())
    n_days    = (df_oos["datetime"].iloc[-1] - df_oos["datetime"].iloc[0]).days

    print(f"  Train: {df['datetime'].iloc[0].date()} → "
          f"{df['datetime'].iloc[split-1].date()} ({split:,} bars)")
    print(f"  OOS  : {oos_start} → {oos_end}  ({len(df_oos):,} bars / {n_days}d)")

    # Run all strategies
    metrics_by_strat = {}
    mc_by_strat      = {}
    all_trades       = []
    leaderboard_rows = []
    journal_rows     = []

    for name, fn in STRATEGIES.items():
        res    = run_backtest(df_oos, fn, name)
        m      = compute_metrics(res["trades"], name)
        mc     = monte_carlo(m["pnls"], CONFIG["MC_ITERATIONS"])
        verdict = _verdict_from_metrics(m, mc)

        metrics_by_strat[name] = {"metrics": m, "verdict": verdict}
        mc_by_strat[name]      = mc
        all_trades.extend(res["trades"])

        print(f"  {name:<22}  n={m['n_trades']:>4}  "
              f"PF {m['profit_factor']:>6.3f}  "
              f"Exp {m['expectancy_r']:>+6.3f}R  "
              f"Net ${m['net_profit']:>8,.0f}  "
              f"MDD {m['max_drawdown']:.2%}  "
              f"→ {verdict}")

        leaderboard_rows.append({
            "strategy":          name,
            "symbol":            symbol,
            "n_trades":          m["n_trades"],
            "win_rate":          m["win_rate"],
            "profit_factor":     m["profit_factor"],
            "expectancy_r":      m["expectancy_r"],
            "net_profit":        m["net_profit"],
            "max_drawdown":      m["max_drawdown"],
            "sharpe":            m["sharpe"],
            "mc_prob_profit":    mc["prob_profit"],
            "avg_hold_minutes":  m["avg_hold_minutes"],
            "verdict":           verdict,
        })

        journal_rows.append(
            _journal_row(name, symbol, m, mc, verdict)
        )

    # Rank
    ranked = build_leaderboard(leaderboard_rows)

    # Charts
    print("  Generating charts...")
    chart_paths = plot_all_strategies(metrics_by_strat, symbol, oos_start, oos_end)
    mc_path     = plot_monte_carlo_grid(metrics_by_strat, mc_by_strat, symbol)
    chart_paths.append(mc_path)

    # Trade log
    log = save_trade_log(all_trades, symbol)
    if log:
        chart_paths.append(log)
        print(f"  Trade log → {log}")

    # Leaderboard report
    print_leaderboard(ranked, symbol, oos_start, oos_end)

    # Per-strategy detail
    for name in STRATEGIES:
        m  = metrics_by_strat[name]["metrics"]
        mc = mc_by_strat[name]
        print_strategy_detail(name, m, mc, symbol)

    print(f"\n  Charts saved:")
    for p in chart_paths:
        print(f"  → {p}")

    return leaderboard_rows, journal_rows


def print_cross_symbol_summary(all_lb_rows: list[dict]) -> None:
    """Aggregate leaderboard rows across all symbols; print overall ranking."""
    if not all_lb_rows:
        return

    S = "=" * 90
    print(f"\n{S}")
    print("  QUANTLAB AI — RESEARCH #004   |   CROSS-SYMBOL AGGREGATE LEADERBOARD")
    print(S)

    # Group by strategy name; average key metrics
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_lb_rows:
        grouped[r["strategy"]].append(r)

    agg = []
    for name, rows in grouped.items():
        n       = len(rows)
        avg_pf  = sum(r["profit_factor"]  for r in rows) / n
        avg_exp = sum(r["expectancy_r"]   for r in rows) / n
        avg_mdd = sum(r["max_drawdown"]   for r in rows) / n
        avg_mcp = sum(r["mc_prob_profit"] for r in rows) / n
        avg_sh  = sum(r["sharpe"]         for r in rows) / n
        tot_tr  = sum(r["n_trades"]       for r in rows)
        tot_net = sum(r["net_profit"]     for r in rows)
        n_prom  = sum(1 for r in rows if r["verdict"] == "PROMOTE")
        n_weak  = sum(1 for r in rows if r["verdict"] == "WEAK")
        agg.append({
            "strategy":         name,
            "avg_pf":           avg_pf,
            "avg_exp":          avg_exp,
            "avg_mdd":          avg_mdd,
            "avg_mc":           avg_mcp,
            "avg_sharpe":       avg_sh,
            "total_trades":     tot_tr,
            "total_net":        tot_net,
            "symbols_tested":   n,
            "promote_count":    n_prom,
            "weak_count":       n_weak,
        })

    # Score: same formula as build_leaderboard
    def _score(r):
        pf  = min(r["avg_pf"],  5.0)
        er  = max(r["avg_exp"], -2.0)
        mpp = r["avg_mc"]
        mdd = max(r["avg_mdd"], -1.0)
        return pf * 0.40 + (1.0 + er) * 0.30 + mpp * 0.20 + (1.0 + mdd) * 0.10

    agg.sort(key=_score, reverse=True)

    H = (f"  {'Rank':<5} {'Strategy':<22} {'Avg PF':>8} {'Avg Exp':>9} "
         f"{'Avg MDD':>8} {'Avg MC':>8} {'Tot Net$':>11} {'Trades':>7} "
         f"{'Promoted':>9}")
    print(H)
    print(f"  {'-'*88}")
    for i, r in enumerate(agg, 1):
        prom_str = f"{r['promote_count']}/{r['symbols_tested']} syms"
        print(
            f"  {i:<5} {r['strategy']:<22} {r['avg_pf']:>8.3f} "
            f"{r['avg_exp']:>+9.3f}R {r['avg_mdd']:>8.2%} "
            f"{r['avg_mc']:>8.1%} ${r['total_net']:>10,.0f} "
            f"{r['total_trades']:>7} {prom_str:>9}"
        )

    print(f"  {'-'*88}")
    best = agg[0]
    print(f"\n  OVERALL BEST CONCEPT:  {best['strategy']}")
    print(f"    Avg PF {best['avg_pf']:.3f}  |  "
          f"Avg Expectancy {best['avg_exp']:+.3f}R  |  "
          f"Promoted on {best['promote_count']}/{best['symbols_tested']} symbols")

    # Verdict
    promoted_all = [r for r in agg if r["promote_count"] == r["symbols_tested"]]
    promoted_any = [r for r in agg if r["promote_count"] > 0]

    print()
    if promoted_all:
        print(f"  Strategies promoted on ALL symbols (highest confidence):")
        for r in promoted_all:
            print(f"    ★ {r['strategy']}")
        print(f"  → NEXT: Research #005 — Validate on 18-month dataset")
    elif promoted_any:
        print(f"  Strategies promoted on ≥1 symbol:")
        for r in promoted_any:
            print(f"    ★ {r['strategy']}  ({r['promote_count']}/{r['symbols_tested']} symbols)")
        print(f"  → NEXT: Research #005 — Determine if promotion is symbol-specific")
        print(f"    or reflects a general edge; test on wider history.")
    else:
        weak_any = [r for r in agg if r["weak_count"] > 0]
        if weak_any:
            best_weak = weak_any[0]
            print(f"  No strategy promoted on any symbol.")
            print(f"  Strongest WEAK signal: {best_weak['strategy']}  "
                  f"Avg PF {best_weak['avg_pf']:.3f}")
            print(f"  → NEXT: Research #005 — Combine {best_weak['strategy']}")
            print(f"    with ADX regime filter (Trending only) on wider history.")
        else:
            print(f"  No strategy shows edge on 1H crypto perpetuals in this period.")
            print(f"  → RECOMMENDATION: Pivot to a fundamentally different concept.")
            print(f"    Suggested: mean-reversion on RSI extremes (RSI < 30, close > EMA200)")

    print(S)


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              QUANTLAB AI — RESEARCH #004                                      ║
║   Multi-Hypothesis Tester: Six Trading Concepts on Identical Conditions       ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  1  FVG + EMA200 + Slope        (Research #002/003 baseline)
  2  Liquidity Sweep Reversal    (sweep prior low, close above)
  3  Break of Structure          (N-bar high breakout)
  4  VWAP Pullback               (tag rolling VWAP from above)
  5  Opening Range Breakout      (UTC-day first-4-hour range)
  6  Volatility Compression      (ATR compression then breakout)

  Engine, fees, costs, RR, position sizing, train/test: LOCKED — IDENTICAL FOR ALL
""")

    print("  LOCKED PARAMETERS")
    print(f"  {'─' * 50}")
    for sym in CONFIG["SYMBOLS"]:
        print(f"  Symbol       : {sym}")
    print(f"  Timeframe    : {CONFIG['TIMEFRAME']}")
    print(f"  Risk:Reward  : 1:{CONFIG['RISK_REWARD']}")
    print(f"  Taker Fee    : {CONFIG['TAKER_FEE']:.3%}")
    print(f"  Spread       : {CONFIG['SPREAD']:.3%}")
    print(f"  SL Slippage  : {CONFIG['SL_SLIPPAGE']:.3%}")
    print(f"  Max Leverage : {CONFIG['MAX_LEVERAGE']:.0f}×")
    print(f"  Capital      : ${CONFIG['STARTING_CAPITAL']:,.0f}")
    print(f"  Risk/Trade   : {CONFIG['RISK_PER_TRADE_PCT']:.1%}")
    print(f"  Train/OOS    : {CONFIG['TRAIN_RATIO']:.0%} / {1-CONFIG['TRAIN_RATIO']:.0%}")
    print(f"  Journal      : {CONFIG['JOURNAL_FILE']}")
    print()

    random.seed(42)
    np.random.seed(42)

    all_lb_rows  = []
    all_jnl_rows = []

    for sym in CONFIG["SYMBOLS"]:
        try:
            lb_rows, jnl_rows = process_symbol(sym)
            all_lb_rows.extend(lb_rows)
            all_jnl_rows.extend(jnl_rows)
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {sym}: {exc}")
            traceback.print_exc()

    # Cross-symbol aggregate
    print_cross_symbol_summary(all_lb_rows)

    # Append to research journal
    if all_jnl_rows:
        append_journal(all_jnl_rows)
        print(f"\n  Research journal updated → {CONFIG['JOURNAL_FILE']}")
        print(f"  ({len(all_jnl_rows)} rows appended)")

    print(f"\n  Research #004 complete.")
    print(f"  Results saved to: {CONFIG['OUTPUT_FOLDER']}/\n")


if __name__ == "__main__":
    main()
