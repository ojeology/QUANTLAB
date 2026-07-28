"""
=============================================================================
QUANTLAB AI – RESEARCH #005
Trade Attribution & Edge Discovery — Liquidity Sweep Reversal

Objective:
  Run the exact same Liquidity Sweep Reversal backtest (engine locked, zero
  changes to execution, fees, RR, sizing, or train/OOS split).
  After every completed trade, capture detailed market-context statistics at
  the moment the signal fired.  Separate wins from losses and rigorously
  quantify which features best explain the outcome.

No strategy modifications.  No filters added.  Descriptive analysis only.
=============================================================================
"""

import os
import sys
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
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Import locked engine from Research #004 (nothing modified)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import (
    CONFIG,
    get_data,
    add_indicators,
    run_backtest,
    compute_metrics,
    monte_carlo,
    strategy_lsr,
    append_journal,
    _journal_row,
    _bar_minutes,
    _verdict_from_metrics,
)

RESEARCH_ID  = "R005"
OUTPUT_FOLDER = CONFIG["OUTPUT_FOLDER"]
BG = "#0F1117"

# =============================================================================
# SECTION 1 — FUNDING RATE CACHE
# =============================================================================

OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate-history"


def _funding_cache_path(symbol: str) -> str:
    safe   = symbol.replace("-", "_")
    folder = CONFIG["CACHE_FOLDER"]
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{safe}_funding.parquet")


def _fetch_funding_page(symbol: str, after_ms: int = None) -> list:
    params = {"instId": symbol, "limit": 100}
    if after_ms:
        params["after"] = str(after_ms)
    try:
        r    = requests.get(OKX_FUNDING_URL, params=params, timeout=15)
        data = r.json()
        if data.get("code") == "0":
            return data.get("data", [])
    except Exception as e:
        print(f"  [WARN] Funding API error: {e}")
    return []


def get_funding_rates(symbol: str) -> pd.DataFrame:
    """
    Fetch or load cached 8-hour funding rates for `symbol`.
    Returns DataFrame with columns [datetime (UTC), funding_rate].
    """
    path = _funding_cache_path(symbol)

    # Load existing cache
    cached_df = None
    if os.path.exists(path):
        cached_df = pd.read_parquet(path)
        if "datetime" in cached_df.columns and len(cached_df) > 0:
            if cached_df["datetime"].dt.tz is None:
                cached_df["datetime"] = cached_df["datetime"].dt.tz_localize("UTC")

    # Determine fetch window
    months    = CONFIG["MONTHS_HISTORY"]
    now_ms    = int(time.time() * 1000)
    cutoff_ms = now_ms - int(months * 30.44 * 24 * 3600 * 1000)

    if cached_df is not None and len(cached_df) > 0:
        last_ts_ms = int(cached_df["datetime"].max().timestamp() * 1000)
        if now_ms - last_ts_ms < 8 * 3600 * 1000:
            return cached_df  # fully current
        # Incremental update
        since_ms    = last_ts_ms
        print(f"  [Funding] Updating {symbol} since "
              f"{datetime.fromtimestamp(since_ms/1000, tz=timezone.utc).date()}…")
    else:
        since_ms = None
        print(f"  [Funding] Downloading {symbol} ({months}m of funding rates)…")

    all_rows   = []
    cursor     = None
    pages      = 0
    while True:
        raw = _fetch_funding_page(symbol, after_ms=cursor)
        if not raw:
            break
        all_rows.extend(raw)
        pages += 1
        oldest_ms = int(raw[-1]["fundingTime"])
        cursor    = oldest_ms
        stop_at   = since_ms if since_ms else cutoff_ms
        if oldest_ms <= stop_at:
            break
        time.sleep(0.2)

    if not all_rows:
        print(f"  [Funding] No data for {symbol}. Funding rates will be NaN.")
        return pd.DataFrame(columns=["datetime", "funding_rate"])

    new_df = pd.DataFrame(all_rows)
    new_df["datetime"]     = pd.to_datetime(
        pd.to_numeric(new_df["fundingTime"]), unit="ms", utc=True
    )
    new_df["funding_rate"] = pd.to_numeric(new_df["fundingRate"])
    new_df = new_df[["datetime", "funding_rate"]].drop_duplicates("datetime")

    # Merge with cache
    if cached_df is not None and len(cached_df) > 0:
        if since_ms:
            new_only = new_df[new_df["datetime"] > cached_df["datetime"].max()]
        else:
            new_only = new_df
        combined = (pd.concat([cached_df, new_only], ignore_index=True)
                    .drop_duplicates("datetime")
                    .sort_values("datetime")
                    .reset_index(drop=True))
    else:
        combined = new_df.sort_values("datetime").reset_index(drop=True)
        cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
        combined  = combined[combined["datetime"] >= cutoff_dt].reset_index(drop=True)

    combined.to_parquet(path, index=False)
    print(f"  [Funding] {len(combined):,} records cached → {path}")
    return combined


def attach_funding_rate(trades: list, funding_df: pd.DataFrame) -> None:
    """
    For each trade, find the most recent funding rate before entry_time.
    Mutates the trade dicts in-place, adding key 'funding_rate'.
    """
    if funding_df is None or len(funding_df) == 0:
        for t in trades:
            t["funding_rate"] = float("nan")
        return

    fts = funding_df["funding_rate"].values
    fdt = funding_df["datetime"].values  # numpy datetime64

    for t in trades:
        et  = np.datetime64(t["entry_time"].to_pydatetime().replace(tzinfo=None), "ns")
        idx = np.searchsorted(fdt, et, side="right") - 1
        t["funding_rate"] = float(fts[idx]) if 0 <= idx < len(fts) else float("nan")


# =============================================================================
# SECTION 2 — RESEARCH #005 ADDITIONAL INDICATORS
# =============================================================================

def add_r005_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add R005-specific context columns on top of the R004 indicator set.
    All computations are purely backward-looking (no look-ahead).
    """
    df = df.copy()

    # ── Relative Volume ───────────────────────────────────────────────────
    # vol / 48-bar rolling mean (≈ 2-day average on 1H)
    df["rel_vol"] = df["vol"] / df["vol"].rolling(48, min_periods=1).mean()

    # ── 5-bar price range (% of close) ───────────────────────────────────
    df["range_5_pct"] = (
        (df["high"].rolling(5).max() - df["low"].rolling(5).min())
        / df["close"]
    ) * 100.0

    # ── 10-bar return volatility (annualised-ish, kept as % std) ─────────
    df["ret_vol_10"] = df["close"].pct_change().rolling(10).std() * 100.0

    # ── 20-bar highest high & lowest low ─────────────────────────────────
    df["hh_20"] = df["high"].rolling(20).max()
    df["ll_20"] = df["low"].rolling(20).min()

    # ── Distance from swing levels (% of close) ──────────────────────────
    df["dist_from_hh_pct"] = (df["close"] - df["hh_20"]) / df["close"] * 100.0
    df["dist_from_ll_pct"] = (df["close"] - df["ll_20"]) / df["close"] * 100.0

    # ── Distance from EMA200 (% of close) ────────────────────────────────
    df["dist_from_ema_pct"] = (df["close"] - df["ema200"]) / df["close"] * 100.0

    # ── EMA200 slope (% change over SLOPE_LOOKBACK bars) ─────────────────
    df["ema200_slope_pct"] = (
        (df["ema200"] - df["ema200_lag"]) / df["ema200_lag"].replace(0, np.nan)
    ) * 100.0

    # ── Normalised candle range (signal bar, % of close) ─────────────────
    df["candle_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100.0

    # ── ATR as % of close ─────────────────────────────────────────────────
    df["atr_pct"] = df["atr"] / df["close"] * 100.0

    # ── ATR percentile (0-100, already in df as atr_pctile-ratio; recompute) ─
    # We want the rolling rank of ATR: where does current ATR sit in last 50 bars?
    atr_win = CONFIG["VCB_ATR_WINDOW"]
    df["atr_rank_pct"] = (
        df["atr"]
        .rolling(atr_win)
        .apply(lambda x: (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100
               if len(x) > 1 else 50.0, raw=True)
    )

    return df


# =============================================================================
# SECTION 3 — TRADE CONTEXT CAPTURE
# =============================================================================

def _session(hour: int) -> str:
    if 0 <= hour < 8:
        return "Asia"
    elif 8 <= hour < 16:
        return "London"
    else:
        return "New York"


DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def enrich_trades_with_context(trades: list, df_oos: pd.DataFrame) -> pd.DataFrame:
    """
    For each completed trade, look up the signal bar (bar before entry) in
    df_oos and attach market context features.

    The engine sets entry_time = df.iloc[i]["datetime"] and enters on bar i.
    The signal fires on bar i-1.  All context is from bar i-1.

    Returns a DataFrame with one row per trade plus all context columns.
    """
    if not trades:
        return pd.DataFrame()

    # Build datetime → positional index lookup
    dt_to_pos = {dt: pos for pos, dt in enumerate(df_oos["datetime"])}

    rows = []
    for t in trades:
        entry_bar_pos = dt_to_pos.get(t["entry_time"])
        if entry_bar_pos is None or entry_bar_pos < 1:
            continue  # can't find signal bar; skip

        sig = df_oos.iloc[entry_bar_pos - 1]   # signal bar
        et  = t["entry_time"]
        hour = et.hour
        dow  = et.dayofweek   # 0=Mon

        row = {
            # ── Core trade fields ─────────────────────────────────
            "entry_time":        t["entry_time"],
            "exit_time":         t["exit_time"],
            "win":               int(t["win"]),
            "pnl":               t["pnl"],
            "r_multiple":        t["r_multiple"],
            "holding_minutes":   t["holding_minutes"],
            "exit_type":         t["exit_type"],

            # ── Time context ──────────────────────────────────────
            "hour_utc":          hour,
            "day_of_week":       dow,
            "day_name":          DAY_NAMES[dow],
            "session":           _session(hour),

            # ── Trend / structure ─────────────────────────────────
            "adx":               float(sig["adx"]),
            "ema200":            float(sig["ema200"]),
            "dist_from_ema_pct": float(sig["dist_from_ema_pct"]),
            "ema200_slope_pct":  float(sig["ema200_slope_pct"]),

            # ── Volatility ────────────────────────────────────────
            "atr":               float(sig["atr"]),
            "atr_pct":           float(sig["atr_pct"]),
            "atr_rank_pct":      float(sig["atr_rank_pct"]),
            "range_5_pct":       float(sig["range_5_pct"]),
            "ret_vol_10":        float(sig["ret_vol_10"]),
            "candle_range_pct":  float(sig["candle_range_pct"]),

            # ── Volume ────────────────────────────────────────────
            "volume":            float(sig["vol"]),
            "rel_vol":           float(sig["rel_vol"]),

            # ── Swing levels ──────────────────────────────────────
            "hh_20":             float(sig["hh_20"]),
            "ll_20":             float(sig["ll_20"]),
            "dist_from_hh_pct":  float(sig["dist_from_hh_pct"]),
            "dist_from_ll_pct":  float(sig["dist_from_ll_pct"]),

            # ── Funding rate ─────────────────────────────────────
            "funding_rate":      t.get("funding_rate", float("nan")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Clip extreme outliers (winsorise at 1st / 99th percentile) for analysis
    num_cols = [c for c in df.columns
                if df[c].dtype.kind in "fi" and c not in ("win", "hour_utc", "day_of_week")]
    for col in num_cols:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    return df


# =============================================================================
# SECTION 4 — FEATURE IMPORTANCE  (no ML)
# =============================================================================

# Features used in attribution analysis
CONTEXT_FEATURES = [
    ("adx",               "ADX"),
    ("dist_from_ema_pct", "Dist from EMA200 (%)"),
    ("ema200_slope_pct",  "EMA200 Slope (%)"),
    ("atr_pct",           "ATR (% of price)"),
    ("atr_rank_pct",      "ATR Rank Percentile"),
    ("range_5_pct",       "5-Bar Range (%)"),
    ("ret_vol_10",        "10-Bar Return Vol (%)"),
    ("candle_range_pct",  "Signal Bar Range (%)"),
    ("rel_vol",           "Relative Volume"),
    ("dist_from_hh_pct",  "Dist from 20-Bar HH (%)"),
    ("dist_from_ll_pct",  "Dist from 20-Bar LL (%)"),
    ("hour_utc",          "Hour of Day (UTC)"),
    ("day_of_week",       "Day of Week"),
    ("funding_rate",      "Funding Rate"),
    ("holding_minutes",   "Holding Time (min)"),
]

FEATURE_COLS  = [c for c, _ in CONTEXT_FEATURES]
FEATURE_NAMES = {c: n for c, n in CONTEXT_FEATURES}


def compute_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every feature, compute win/loss means, difference, and Cohen's d.
    Returns a DataFrame sorted by |Cohen's d| descending.
    """
    wins = df[df["win"] == 1]
    loss = df[df["win"] == 0]
    rows = []

    for col, label in CONTEXT_FEATURES:
        if col not in df.columns:
            continue
        w = wins[col].dropna()
        l = loss[col].dropna()
        if len(w) < 2 or len(l) < 2:
            continue

        mean_w = float(w.mean())
        mean_l = float(l.mean())
        diff   = mean_w - mean_l
        pct_d  = (diff / (abs(mean_l) + 1e-12)) * 100.0

        # Pooled standard deviation (Cohen's d)
        pooled_std = math.sqrt(
            ((len(w) - 1) * float(w.var(ddof=1)) +
             (len(l) - 1) * float(l.var(ddof=1)))
            / (len(w) + len(l) - 2)
        )
        cohens_d = diff / pooled_std if pooled_std > 0 else 0.0

        # Pearson / point-biserial correlation with outcome
        all_vals = df[[col, "win"]].dropna()
        if len(all_vals) > 2:
            corr = float(np.corrcoef(
                all_vals[col].values.astype(float),
                all_vals["win"].values.astype(float)
            )[0, 1])
        else:
            corr = float("nan")

        rows.append({
            "feature":     col,
            "label":       label,
            "n_wins":      len(w),
            "n_losses":    len(l),
            "mean_wins":   mean_w,
            "mean_losses": mean_l,
            "diff":        diff,
            "pct_diff":    pct_d,
            "cohens_d":    cohens_d,
            "abs_d":       abs(cohens_d),
            "correlation": corr,
        })

    result = pd.DataFrame(rows).sort_values("abs_d", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


def compute_group_metrics(df: pd.DataFrame,
                          group_col: str,
                          groups: list) -> pd.DataFrame:
    """
    Compute win rate, profit factor, expectancy, and trade count for each group.
    `groups` is a list of (label, mask) tuples.
    """
    rows = []
    rr   = CONFIG["RISK_REWARD"]
    for label, mask in groups:
        sub = df[mask]
        n   = len(sub)
        if n == 0:
            rows.append({"group": label, "n": 0, "win_rate": 0.0,
                         "profit_factor": 0.0, "expectancy_r": 0.0,
                         "avg_r": 0.0, "net_pnl": 0.0})
            continue
        n_win = int(sub["win"].sum())
        n_los = n - n_win
        wr    = n_win / n
        gw    = sub.loc[sub["win"] == 1, "pnl"].sum()
        gl    = abs(sub.loc[sub["win"] == 0, "pnl"].sum())
        pf    = gw / gl if gl > 0 else float("inf") if gw > 0 else 0.0
        exp_r = wr * rr - (1.0 - wr)
        rows.append({
            "group":          label,
            "n":              n,
            "win_rate":       wr,
            "profit_factor":  pf,
            "expectancy_r":   exp_r,
            "avg_r":          float(sub["r_multiple"].mean()),
            "net_pnl":        float(sub["pnl"].sum()),
        })
    return pd.DataFrame(rows)


def compute_time_analysis(df: pd.DataFrame) -> tuple:
    """
    Returns (hourly_df, daily_df, session_df) DataFrames.
    Each has columns: group, n, win_rate, profit_factor, expectancy_r.
    """
    # ── Hourly ────────────────────────────────────────────────────────────
    hourly_rows = []
    for h in range(24):
        mask = df["hour_utc"] == h
        sub  = df[mask]
        if len(sub) == 0:
            hourly_rows.append({"hour": h, "n": 0, "win_rate": 0.0,
                                 "profit_factor": 0.0, "expectancy_r": 0.0})
            continue
        wr   = float(sub["win"].mean())
        gw   = sub.loc[sub["win"]==1, "pnl"].sum()
        gl   = abs(sub.loc[sub["win"]==0, "pnl"].sum())
        pf   = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
        er   = wr * CONFIG["RISK_REWARD"] - (1.0 - wr)
        hourly_rows.append({"hour": h, "n": len(sub), "win_rate": wr,
                             "profit_factor": pf, "expectancy_r": er})
    hourly = pd.DataFrame(hourly_rows)

    # ── Daily ─────────────────────────────────────────────────────────────
    daily_rows = []
    for d in range(7):
        mask = df["day_of_week"] == d
        sub  = df[mask]
        if len(sub) == 0:
            daily_rows.append({"day": DAY_NAMES[d], "n": 0, "win_rate": 0.0,
                                "profit_factor": 0.0, "expectancy_r": 0.0})
            continue
        wr   = float(sub["win"].mean())
        gw   = sub.loc[sub["win"]==1, "pnl"].sum()
        gl   = abs(sub.loc[sub["win"]==0, "pnl"].sum())
        pf   = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
        er   = wr * CONFIG["RISK_REWARD"] - (1.0 - wr)
        daily_rows.append({"day": DAY_NAMES[d], "n": len(sub), "win_rate": wr,
                            "profit_factor": pf, "expectancy_r": er})
    daily = pd.DataFrame(daily_rows)

    # ── Session ───────────────────────────────────────────────────────────
    sessions = ["Asia", "London", "New York"]
    session_groups = [(s, df["session"] == s) for s in sessions]
    session = compute_group_metrics(df, "session", session_groups)
    session = session.rename(columns={"group": "session"})

    return hourly, daily, session


def compute_volatility_analysis(df: pd.DataFrame) -> pd.DataFrame:
    p33  = df["atr_rank_pct"].quantile(0.333)
    p67  = df["atr_rank_pct"].quantile(0.667)
    groups = [
        ("Low Vol",    df["atr_rank_pct"] <= p33),
        ("Medium Vol", (df["atr_rank_pct"] > p33) & (df["atr_rank_pct"] <= p67)),
        ("High Vol",   df["atr_rank_pct"] > p67),
    ]
    return compute_group_metrics(df, "vol_group", groups)


def compute_adx_analysis(df: pd.DataFrame) -> pd.DataFrame:
    low  = CONFIG["ADX_WEAK"]
    high = CONFIG["ADX_TRENDING"]
    groups = [
        ("Ranging (ADX<20)",    df["adx"] < low),
        ("Weak Trend (20-25)",  (df["adx"] >= low) & (df["adx"] < high)),
        ("Trending (ADX≥25)",   df["adx"] >= high),
    ]
    return compute_group_metrics(df, "adx_group", groups)


# =============================================================================
# SECTION 5 — VISUALISATIONS
# =============================================================================

def _ax_style(ax, facecolor=None):
    ax.set_facecolor(facecolor or BG)
    ax.tick_params(colors="white", labelsize=8)
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333333")
    ax.grid(True, alpha=0.18, color="#444")


def _save(fig, filename: str) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


# ── Chart 1: Feature Comparison — wins vs losses bar chart ─────────────────

def plot_feature_comparison(attr: pd.DataFrame, symbol: str, n_trades: int) -> str:
    """
    Horizontal bar chart showing mean(wins) and mean(losses) for every feature,
    ranked by |Cohen's d|.
    """
    # Keep only features with data; already sorted by abs_d
    top = attr[attr["abs_d"] > 0].copy()
    n   = len(top)
    if n == 0:
        return ""

    fig_h = max(6, n * 0.55 + 2)
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_h))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Feature Comparison — Wins vs Losses  |  {symbol}  "
        f"({n_trades} trades, R005 — Liq.Sweep)",
        fontsize=10, fontweight="bold", color="white", y=1.01,
    )

    labels   = [FEATURE_NAMES.get(r["feature"], r["feature"]) for _, r in top.iterrows()]
    mean_w   = top["mean_wins"].values
    mean_l   = top["mean_losses"].values
    cohens   = top["cohens_d"].values
    y_pos    = np.arange(n)

    # Panel A: win vs loss means
    ax = axes[0]
    _ax_style(ax)
    bar_h = 0.38
    ax.barh(y_pos + bar_h/2, mean_w, height=bar_h,
            color="#00C49A", alpha=0.85, label="Wins")
    ax.barh(y_pos - bar_h/2, mean_l, height=bar_h,
            color="#FF4560", alpha=0.85, label="Losses")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="white", fontsize=8)
    ax.set_xlabel("Mean Value", color="white")
    ax.set_title("Mean at Entry — Wins vs Losses", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444",
              labelcolor="white", loc="lower right")
    ax.axvline(0, color="#666", lw=0.8)

    # Panel B: Cohen's d (effect size)
    ax2 = axes[1]
    _ax_style(ax2)
    bar_colors = ["#00C49A" if v > 0 else "#FF4560" for v in cohens]
    bars = ax2.barh(y_pos, cohens, height=0.6, color=bar_colors, alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, color="white", fontsize=8)
    ax2.set_xlabel("Cohen's d  (positive = higher in wins)", color="white")
    ax2.set_title("Effect Size (Cohen's d)", fontsize=9)
    ax2.axvline(0,    color="#666", lw=0.8)
    ax2.axvline(0.2,  color="#FFB347", lw=0.8, ls="--", alpha=0.6, label="Small (0.2)")
    ax2.axvline(0.5,  color="#FFD700", lw=0.8, ls="--", alpha=0.6, label="Medium (0.5)")
    ax2.axvline(-0.2, color="#FFB347", lw=0.8, ls="--", alpha=0.6)
    ax2.axvline(-0.5, color="#FFD700", lw=0.8, ls="--", alpha=0.6)
    ax2.legend(fontsize=7, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    plt.tight_layout()
    safe = symbol.replace("-", "_")
    return _save(fig, f"{safe}_r005_feature_comparison.png")


# ── Chart 2: Correlation Matrix ─────────────────────────────────────────────

def plot_correlation_matrix(combined_df: pd.DataFrame) -> str:
    cols = [c for c in FEATURE_COLS if c in combined_df.columns]
    cols = [c for c in cols if combined_df[c].dropna().nunique() > 1]

    mat = combined_df[cols + ["win"]].dropna().corr()
    n   = len(mat)
    if n < 2:
        return ""

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), max(9, n * 0.65)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cmap   = plt.cm.RdYlGn
    im     = ax.imshow(mat.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    all_labels = [FEATURE_NAMES.get(c, c) for c in mat.columns]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_labels, rotation=45, ha="right",
                       color="white", fontsize=7)
    ax.set_yticklabels(all_labels, color="white", fontsize=7)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            v = mat.values[i, j]
            c = "black" if abs(v) > 0.5 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color=c)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors="white", labelsize=7)
    cbar.set_label("Pearson r", color="white", fontsize=8)

    ax.set_title("Correlation Matrix — Market Context Features + Outcome (Win)\n"
                 "All symbols combined",
                 fontsize=10, fontweight="bold", color="white")
    fig.tight_layout()
    return _save(fig, "r005_correlation_matrix.png")


# ── Chart 3: Time Analysis ───────────────────────────────────────────────────

def plot_time_analysis(hourly: pd.DataFrame,
                       daily: pd.DataFrame,
                       session: pd.DataFrame) -> str:
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Time Analysis — Performance by Hour / Day / Session\n"
                 "Liq.Sweep Reversal  |  All Symbols Combined  |  R005",
                 fontsize=11, fontweight="bold", color="white", y=1.002)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    # Row 0: Hourly
    ax_hn  = fig.add_subplot(gs[0, 0])  # trade count
    ax_hwr = fig.add_subplot(gs[0, 1])  # win rate
    ax_hpf = fig.add_subplot(gs[0, 2])  # profit factor

    for ax in (ax_hn, ax_hwr, ax_hpf):
        _ax_style(ax)

    hours = hourly["hour"].values
    # shade sessions
    session_bands = [("Asia", 0, 8, "#1a3a4a"), ("London", 8, 16, "#1a4a2a"),
                     ("New York", 16, 24, "#3a2a1a")]

    for ax, col, title, ref, ref_lbl in [
        (ax_hn,  "n",             "Trade Count by Hour", None, ""),
        (ax_hwr, "win_rate",      "Win Rate by Hour",    0.333, "Break-even WR"),
        (ax_hpf, "profit_factor", "PF by Hour",          1.0,   "Break-even PF"),
    ]:
        for _, h0, h1, col_bg in session_bands:
            ax.axvspan(h0 - 0.5, h1 - 0.5, color=col_bg, alpha=0.4, zorder=0)
        bar_colors = ["#4A90D9" if v > 0 else "#FF4560"
                      for v in hourly[col].values]
        ax.bar(hours, hourly[col].values, color=bar_colors, width=0.7, zorder=2)
        if ref is not None:
            ax.axhline(ref, color="#FFD700", lw=1.0, ls="--",
                       label=ref_lbl, zorder=3)
            ax.legend(fontsize=7, facecolor="#1A1D24", edgecolor="#444",
                      labelcolor="white")
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hour UTC", color="white")
        ax.set_title(title, fontsize=9)

    # Add session labels on trade count chart
    for sess, h0, h1, _ in session_bands:
        mid = (h0 + h1) / 2 - 0.5
        ax_hn.text(mid, ax_hn.get_ylim()[1] * 0.92,
                   sess, ha="center", color="white", fontsize=7, alpha=0.9)

    # Row 1: Daily
    ax_dn  = fig.add_subplot(gs[1, 0])
    ax_dwr = fig.add_subplot(gs[1, 1])
    ax_dpf = fig.add_subplot(gs[1, 2])

    for ax in (ax_dn, ax_dwr, ax_dpf):
        _ax_style(ax)

    day_labels = daily["day"].values
    x = np.arange(7)
    for ax, col, title, ref in [
        (ax_dn,  "n",             "Trade Count by Day", None),
        (ax_dwr, "win_rate",      "Win Rate by Day",    0.333),
        (ax_dpf, "profit_factor", "PF by Day",          1.0),
    ]:
        bc = ["#00C49A" if v > (ref or 0) else "#FF4560"
              for v in daily[col].values]
        ax.bar(x, daily[col].values, color=bc, width=0.6)
        if ref:
            ax.axhline(ref, color="#FFD700", lw=1.0, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(day_labels, color="white", fontsize=8)
        ax.set_title(title, fontsize=9)

    # Row 2: Session (3 metrics as grouped bars)
    ax_s = fig.add_subplot(gs[2, :])
    _ax_style(ax_s)

    sess_names = session["session"].values if "session" in session.columns \
        else session["group"].values
    x_s    = np.arange(len(sess_names))
    width  = 0.25
    wr_vals = session["win_rate"].values
    pf_vals = session["profit_factor"].values
    er_vals = session["expectancy_r"].values
    n_vals  = session["n"].values

    b1 = ax_s.bar(x_s - width, wr_vals,  width, color="#4A90D9",  label="Win Rate",
                  alpha=0.9)
    b2 = ax_s.bar(x_s,          pf_vals,  width, color="#FFB347",  label="Profit Factor",
                  alpha=0.9)
    b3 = ax_s.bar(x_s + width,  er_vals,  width, color="#00C49A",  label="Expectancy R",
                  alpha=0.9)

    ax_s.axhline(1.0, color="#FFD700", lw=0.9, ls="--", label="PF=1 / WR≈0.33")
    ax_s.axhline(0.0, color="#888",    lw=0.7, ls=":")

    ax_s.set_xticks(x_s)
    ax_s.set_xticklabels(
        [f"{s}\n(n={n_vals[i]})" for i, s in enumerate(sess_names)],
        color="white", fontsize=9
    )
    ax_s.set_title("Performance by Trading Session", fontsize=9)
    ax_s.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white",
                loc="upper right")

    # Annotate bars with values
    for bars, vals, fmt in [(b1, wr_vals, "{:.1%}"),
                             (b2, pf_vals, "{:.2f}"),
                             (b3, er_vals, "{:+.2f}R")]:
        for bar, v in zip(bars, vals):
            ax_s.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.02,
                      fmt.format(v), ha="center", va="bottom",
                      color="white", fontsize=7)

    return _save(fig, "r005_time_analysis.png")


# ── Chart 4: Volatility Analysis ─────────────────────────────────────────────

def plot_group_analysis(group_df: pd.DataFrame,
                        group_col: str,
                        title: str,
                        filename: str,
                        subtitle: str = "") -> str:
    """
    Generic 4-panel group analysis: trade count, win rate, PF, expectancy.
    """
    groups  = group_df[group_col].values if group_col in group_df.columns \
        else group_df["group"].values
    n_g     = len(groups)
    x       = np.arange(n_g)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=10,
                 fontweight="bold", color="white")

    panels = [
        (axes[0, 0], "n",             "Trade Count",        None,  "#4A90D9"),
        (axes[0, 1], "win_rate",      "Win Rate",           0.333, "#00C49A"),
        (axes[1, 0], "profit_factor", "Profit Factor",      1.0,   "#FFB347"),
        (axes[1, 1], "expectancy_r",  "Expectancy (R)",     0.0,   "#E040FB"),
    ]

    for ax, col, panel_title, ref, color in panels:
        _ax_style(ax)
        vals = group_df[col].values
        bc   = [color if (ref is None or v > ref) else "#FF4560" for v in vals]
        ax.bar(x, vals, color=bc, width=0.6, alpha=0.9)
        if ref is not None:
            ax.axhline(ref, color="#FFD700", lw=1.0, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{g}\n(n={group_df['n'].values[i]})" for i, g in enumerate(groups)],
            color="white", fontsize=8
        )
        ax.set_title(panel_title, fontsize=9)
        for i, v in enumerate(vals):
            fmt = f"{v:.1%}" if col == "win_rate" else \
                  f"{v:.2f}" if col in ("profit_factor",) else \
                  f"{v:+.2f}R"
            ax.text(x[i], v + abs(v) * 0.03 + 0.01,
                    fmt, ha="center", va="bottom",
                    color="white", fontsize=8)

    plt.tight_layout()
    return _save(fig, filename)


# ── Chart 5: Feature Importance Ranking ─────────────────────────────────────

def plot_feature_importance(attr: pd.DataFrame) -> str:
    """
    Ranked horizontal bar chart of |Cohen's d| with correlation annotated.
    """
    top = attr.head(15).copy()
    n   = len(top)
    if n == 0:
        return ""

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.55 + 1.5)))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)

    y_pos  = np.arange(n)
    labels = [FEATURE_NAMES.get(r["feature"], r["feature"])
              for _, r in top.iterrows()]
    vals   = top["abs_d"].values
    corrs  = top["correlation"].values

    bar_colors = ["#00C49A" if top.iloc[i]["cohens_d"] > 0 else "#FF4560"
                  for i in range(n)]
    bars = ax.barh(y_pos, vals, height=0.6, color=bar_colors, alpha=0.85)

    # Reference lines
    ax.axvline(0.2, color="#FFB347", lw=1.0, ls="--", alpha=0.7, label="|d|=0.2 Small")
    ax.axvline(0.5, color="#FFD700", lw=1.0, ls="--", alpha=0.7, label="|d|=0.5 Medium")
    ax.axvline(0.8, color="#FF6B6B", lw=1.0, ls="--", alpha=0.7, label="|d|=0.8 Large")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.set_xlabel("Effect Size  |Cohen's d|", color="white", fontsize=9)
    ax.set_title("Feature Importance — Separation Between Wins and Losses\n"
                 "(All symbols combined  |  Liq.Sweep Reversal  |  R005)",
                 fontsize=9, color="white")
    ax.legend(fontsize=7, facecolor="#1A1D24", edgecolor="#444",
              labelcolor="white", loc="lower right")

    # Annotate with |d| and r
    for i, (bar, v, r) in enumerate(zip(bars, vals, corrs)):
        rstr = f"  r={r:+.2f}" if not math.isnan(r) else ""
        ax.text(bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"d={v:.2f}{rstr}",
                va="center", ha="left",
                color="white", fontsize=7)

    ax.invert_yaxis()  # highest importance at top
    plt.tight_layout()
    return _save(fig, "r005_feature_importance.png")


# ── Chart 6: Scatter — R-multiple vs key features ───────────────────────────

def plot_r_multiple_scatters(combined_df: pd.DataFrame,
                             top_features: list) -> str:
    """
    Scatter plots: R-multiple (y) vs top features (x).
    Points coloured green=win / red=loss.
    """
    feats = [f for f in top_features if f in combined_df.columns][:6]
    n_f   = len(feats)
    if n_f == 0:
        return ""

    ncols = 3
    nrows = math.ceil(n_f / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(15, 4.5 * nrows))
    fig.patch.set_facecolor(BG)
    fig.suptitle("R-Multiple vs Top Features  |  Liq.Sweep  |  R005",
                 fontsize=10, fontweight="bold", color="white")

    flat_axes = np.array(axes).flatten()
    c_win = "#00C49A"
    c_los = "#FF4560"

    for i, feat in enumerate(feats):
        ax = flat_axes[i]
        _ax_style(ax)

        sub  = combined_df[[feat, "r_multiple", "win"]].dropna()
        wins = sub[sub["win"] == 1]
        loss = sub[sub["win"] == 0]

        ax.scatter(loss[feat], loss["r_multiple"], color=c_los,
                   alpha=0.45, s=18, label="Loss", zorder=2)
        ax.scatter(wins[feat], wins["r_multiple"], color=c_win,
                   alpha=0.55, s=18, label="Win",  zorder=3)

        # Trend line
        try:
            x_all = sub[feat].values.astype(float)
            y_all = sub["r_multiple"].values.astype(float)
            z     = np.polyfit(x_all, y_all, 1)
            xr    = np.linspace(x_all.min(), x_all.max(), 100)
            ax.plot(xr, np.polyval(z, xr), color="#FFD700",
                    lw=1.2, alpha=0.8, zorder=4)
        except Exception:
            pass

        ax.axhline(0, color="#666", lw=0.7, ls=":")
        ax.set_xlabel(FEATURE_NAMES.get(feat, feat), color="white", fontsize=8)
        ax.set_ylabel("R-Multiple", color="white", fontsize=8)
        r = float(np.corrcoef(sub[feat].values.astype(float),
                              sub["r_multiple"].values.astype(float))[0, 1])
        ax.set_title(f"{FEATURE_NAMES.get(feat, feat)}  r={r:+.3f}", fontsize=8)
        ax.legend(fontsize=7, facecolor="#1A1D24", edgecolor="#444",
                  labelcolor="white")

    for j in range(n_f, len(flat_axes)):
        flat_axes[j].set_visible(False)

    plt.tight_layout()
    return _save(fig, "r005_r_vs_features.png")


# ── Chart 7: Win-rate heatmap hour × day ─────────────────────────────────────

def plot_time_heatmap(combined_df: pd.DataFrame) -> str:
    """
    24 × 7 win-rate heatmap: rows = hour UTC, cols = day of week.
    Cells with < MIN_N trades are greyed out.
    """
    MIN_N  = 2
    matrix = np.full((24, 7), np.nan)
    counts = np.zeros((24, 7), dtype=int)

    for h in range(24):
        for d in range(7):
            sub = combined_df[(combined_df["hour_utc"] == h) &
                              (combined_df["day_of_week"] == d)]
            if len(sub) >= MIN_N:
                matrix[h, d] = float(sub["win"].mean())
                counts[h, d] = len(sub)

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Mask NaN → grey
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#222222")
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1,
                   aspect="auto", origin="upper")

    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_NAMES, color="white", fontsize=9)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)],
                       color="white", fontsize=8)

    # Session bands (left side)
    for label, h0, h1, _ in [("Asia", 0, 8, ""), ("London", 8, 16, ""),
                               ("NY", 16, 24, "")]:
        ax.axhline(h0 - 0.5, color="#444", lw=0.6)
        ax.text(-0.7, (h0 + h1) / 2, label, ha="right", va="center",
                color="#aaa", fontsize=8)

    # Annotate cells with n
    for h in range(24):
        for d in range(7):
            n = counts[h, d]
            if n > 0:
                v  = matrix[h, d]
                tc = "black" if v > 0.6 or v < 0.3 else "white"
                ax.text(d, h, f"{n}", ha="center", va="center",
                        color=tc, fontsize=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Win Rate", color="white", fontsize=9)
    cbar.ax.tick_params(colors="white")

    ax.set_title("Win-Rate Heatmap  |  Hour UTC (row) × Day (col)\n"
                 "Liq.Sweep Reversal — All Symbols  |  R005  "
                 f"(cell = n trades, grey = <{MIN_N} trades)",
                 fontsize=9, fontweight="bold", color="white")

    plt.tight_layout()
    return _save(fig, "r005_time_heatmap.png")


# =============================================================================
# SECTION 6 — REPORT PRINTING
# =============================================================================

def print_attribution_report(attr: pd.DataFrame,
                              hourly: pd.DataFrame,
                              daily: pd.DataFrame,
                              session: pd.DataFrame,
                              vol_df: pd.DataFrame,
                              adx_df: pd.DataFrame,
                              combined_df: pd.DataFrame,
                              symbol_results: dict) -> None:
    S  = "=" * 100
    S2 = "─" * 100
    BLK = " " * 2

    print(f"\n{S}")
    print(f"{BLK}QUANTLAB AI — RESEARCH #005")
    print(f"{BLK}Trade Attribution & Edge Discovery — Liquidity Sweep Reversal")
    print(f"{BLK}{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(S)

    # ── Per-symbol summary ─────────────────────────────────────────────────
    print(f"\n{BLK}STRATEGY PERFORMANCE SUMMARY  (OOS — Locked Engine)")
    print(f"{BLK}{S2[2:]}")
    print(f"{BLK}{'Symbol':<24} {'Trades':>7} {'WR':>7} {'PF':>8} "
          f"{'Exp(R)':>9} {'Net$':>10} {'MDD':>8}")
    print(f"{BLK}{S2[2:]}")
    for sym, res in symbol_results.items():
        m = res["metrics"]
        v = res["verdict"]
        star = "★" if v == "PROMOTE" else ("·" if v == "WEAK" else " ")
        print(f"{BLK}{sym:<24} {m['n_trades']:>7} "
              f"{m['win_rate']:>7.1%} {m['profit_factor']:>8.3f} "
              f"{m['expectancy_r']:>+9.3f}R ${m['net_profit']:>9,.0f} "
              f"{m['max_drawdown']:>8.2%}  {star}{v}")
    print(f"{BLK}{S2[2:]}")

    n_total = len(combined_df)
    n_wins  = int(combined_df["win"].sum())
    n_loss  = n_total - n_wins
    wr_all  = n_wins / n_total if n_total else 0
    print(f"\n{BLK}Combined OOS: {n_total} trades  "
          f"({n_wins} wins / {n_loss} losses)  "
          f"Overall WR {wr_all:.1%}\n")

    # ── Feature Importance Table ───────────────────────────────────────────
    print(f"{BLK}FEATURE IMPORTANCE — Ranked by Effect Size  (Cohen's d)  "
          f"[All Symbols Combined]")
    print(f"{BLK}{S2[2:]}")
    print(f"{BLK}{'Rank':<5} {'Feature':<32} {'Mean(W)':>10} {'Mean(L)':>10} "
          f"{'Diff':>10} {'Diff%':>8} {'Cohen d':>9} {'Corr':>8}")
    print(f"{BLK}{S2[2:]}")
    for _, row in attr.iterrows():
        flag = ("  ▲▲▲" if abs(row["cohens_d"]) >= 0.8 else
                "  ▲▲ " if abs(row["cohens_d"]) >= 0.5 else
                "  ▲  " if abs(row["cohens_d"]) >= 0.2 else "     ")
        cr = f"{row['correlation']:+.3f}" if not math.isnan(row["correlation"]) else "  N/A"
        print(f"{BLK}{int(row['rank']):<5} "
              f"{FEATURE_NAMES.get(row['feature'], row['feature']):<32} "
              f"{row['mean_wins']:>10.3f} "
              f"{row['mean_losses']:>10.3f} "
              f"{row['diff']:>+10.3f} "
              f"{row['pct_diff']:>+8.1f}% "
              f"{row['cohens_d']:>+9.3f} "
              f"{cr:>8}{flag}")
    print(f"{BLK}{S2[2:]}")
    print(f"{BLK}Effect sizes: ▲ small (≥0.2)  ▲▲ medium (≥0.5)  ▲▲▲ large (≥0.8)")

    # ── Session Analysis ───────────────────────────────────────────────────
    sess_col = "session" if "session" in session.columns else "group"
    print(f"\n{BLK}SESSION ANALYSIS  (All Symbols Combined)")
    print(f"{BLK}{S2[2:]}")
    print(f"{BLK}{'Session':<18} {'Trades':>7} {'WR':>8} {'PF':>8} "
          f"{'Exp(R)':>9} {'Net$':>10}")
    print(f"{BLK}{S2[2:]}")
    for _, row in session.iterrows():
        sname = row.get("session", row.get("group", "?"))
        print(f"{BLK}{sname:<18} {int(row['n']):>7} "
              f"{row['win_rate']:>8.1%} {row['profit_factor']:>8.3f} "
              f"{row['expectancy_r']:>+9.3f}R ${row['net_pnl']:>9,.0f}")
    print(f"{BLK}{S2[2:]}")

    # Best hour
    best_hr = hourly.loc[hourly["profit_factor"].idxmax()]
    print(f"\n{BLK}Best hour by PF : {int(best_hr['hour']):02d}:00 UTC  "
          f"PF {best_hr['profit_factor']:.3f}  "
          f"WR {best_hr['win_rate']:.1%}  "
          f"n={int(best_hr['n'])}")

    # Best day
    best_dy = daily.loc[daily["profit_factor"].idxmax()]
    print(f"{BLK}Best day by PF  : {best_dy['day']}  "
          f"PF {best_dy['profit_factor']:.3f}  "
          f"WR {best_dy['win_rate']:.1%}  "
          f"n={int(best_dy['n'])}")

    # ── Volatility Analysis ────────────────────────────────────────────────
    print(f"\n{BLK}VOLATILITY ANALYSIS (ATR Rank Percentile Tertiles)")
    print(f"{BLK}{S2[2:]}")
    print(f"{BLK}{'Regime':<22} {'Trades':>7} {'WR':>8} {'PF':>8} "
          f"{'Exp(R)':>9} {'Net$':>10}")
    print(f"{BLK}{S2[2:]}")
    for _, row in vol_df.iterrows():
        gname = row.get("group", "?")
        print(f"{BLK}{gname:<22} {int(row['n']):>7} "
              f"{row['win_rate']:>8.1%} {row['profit_factor']:>8.3f} "
              f"{row['expectancy_r']:>+9.3f}R ${row['net_pnl']:>9,.0f}")
    print(f"{BLK}{S2[2:]}")

    # ── ADX Analysis ──────────────────────────────────────────────────────
    print(f"\n{BLK}TREND ANALYSIS (ADX Regime Groups)")
    print(f"{BLK}{S2[2:]}")
    print(f"{BLK}{'Regime':<28} {'Trades':>7} {'WR':>8} {'PF':>8} "
          f"{'Exp(R)':>9} {'Net$':>10}")
    print(f"{BLK}{S2[2:]}")
    for _, row in adx_df.iterrows():
        gname = row.get("group", "?")
        print(f"{BLK}{gname:<28} {int(row['n']):>7} "
              f"{row['win_rate']:>8.1%} {row['profit_factor']:>8.3f} "
              f"{row['expectancy_r']:>+9.3f}R ${row['net_pnl']:>9,.0f}")
    print(f"{BLK}{S2[2:]}")

    # ── Research Questions — answered objectively ──────────────────────────
    print(f"\n{BLK}RESEARCH QUESTIONS — OBJECTIVE ANSWERS")
    print(f"{BLK}{'─' * 80}")

    def _top_feat(rank=1):
        r = attr[attr["rank"] == rank]
        if len(r) == 0:
            return "N/A"
        return FEATURE_NAMES.get(r.iloc[0]["feature"], r.iloc[0]["feature"])

    def _top_n(n=3):
        return [FEATURE_NAMES.get(r["feature"], r["feature"])
                for _, r in attr.head(n).iterrows()]

    def _feat_d(col):
        r = attr[attr["feature"] == col]
        if len(r) == 0:
            return float("nan")
        return float(r.iloc[0]["cohens_d"])

    def _group_best(grp_df, col="profit_factor"):
        if len(grp_df) == 0:
            return "N/A"
        best_idx = grp_df[col].idxmax()
        g = grp_df.iloc[best_idx]
        gname = g.get("group", g.get("session", "?"))
        return f"{gname} (PF {g[col]:.2f}, n={int(g['n'])})"

    # Q1
    w_means = {row["feature"]: row["mean_wins"]  for _, row in attr.iterrows()}
    l_means = {row["feature"]: row["mean_losses"] for _, row in attr.iterrows()}

    print(f"\n{BLK}Q1  Which market characteristics are most common BEFORE WINNING trades?")
    for _, row in attr.head(5).iterrows():
        if row["cohens_d"] > 0:
            print(f"{BLK}    › {FEATURE_NAMES.get(row['feature'], row['feature']):<36} "
                  f"Mean {row['mean_wins']:>9.3f}  (vs loss: {row['mean_losses']:>9.3f})")

    print(f"\n{BLK}Q2  Which characteristics are most common BEFORE LOSING trades?")
    for _, row in attr.head(5).iterrows():
        if row["cohens_d"] < 0:
            print(f"{BLK}    › {FEATURE_NAMES.get(row['feature'], row['feature']):<36} "
                  f"Mean {row['mean_losses']:>9.3f}  (vs win: {row['mean_wins']:>9.3f})")

    top3 = _top_n(3)
    print(f"\n{BLK}Q3  Which feature shows the STRONGEST SEPARATION?")
    print(f"{BLK}    → {_top_feat(1)}  "
          f"(|d|={abs(attr.iloc[0]['cohens_d']):.3f})")
    print(f"{BLK}    Top 3: {', '.join(top3)}")

    adx_d  = _feat_d("adx")
    adx_ans = ("YES — large effect" if abs(adx_d) >= 0.5 else
               "MODERATE — small-medium effect" if abs(adx_d) >= 0.2 else
               "NO — negligible effect")
    print(f"\n{BLK}Q4  Is ADX actually important?")
    print(f"{BLK}    → {adx_ans}  (d={adx_d:+.3f})")
    best_adx = _group_best(adx_df)
    print(f"{BLK}    Best regime: {best_adx}")

    atr_d   = _feat_d("atr_rank_pct")
    atr_ans = ("YES — large effect" if abs(atr_d) >= 0.5 else
               "MODERATE — small-medium effect" if abs(atr_d) >= 0.2 else
               "NO — negligible effect")
    print(f"\n{BLK}Q5  Is volatility important?")
    print(f"{BLK}    → {atr_ans}  (ATR rank d={atr_d:+.3f})")
    best_vol = _group_best(vol_df)
    print(f"{BLK}    Best regime: {best_vol}")

    slp_d   = _feat_d("ema200_slope_pct")
    slp_ans = ("YES — large effect" if abs(slp_d) >= 0.5 else
               "MODERATE — small-medium effect" if abs(slp_d) >= 0.2 else
               "NO — negligible effect")
    print(f"\n{BLK}Q6  Is EMA slope important?")
    print(f"{BLK}    → {slp_ans}  (d={slp_d:+.3f})")
    if slp_d > 0:
        print(f"{BLK}    Steeper positive slope associated with wins.")
    else:
        print(f"{BLK}    Slope direction does not strongly separate outcomes.")

    # Time analysis answer
    bh     = hourly.loc[hourly["n"] >= 3, "profit_factor"].max() if len(hourly[hourly["n"]>=3])>0 else 0
    wh_pf  = hourly[hourly["n"] >= 3]["profit_factor"].max() if len(hourly[hourly["n"]>=3])>0 else 0
    bh_hr  = int(hourly.loc[(hourly["profit_factor"] == wh_pf) & (hourly["n"] >= 3), "hour"].values[0]) \
             if len(hourly[(hourly["profit_factor"] == wh_pf) & (hourly["n"] >= 3)]) > 0 else -1

    print(f"\n{BLK}Q7  Is TIME OF DAY important?")
    best_sess = _group_best(session, "profit_factor")
    print(f"{BLK}    → Best session: {best_sess}")
    if bh_hr >= 0:
        print(f"{BLK}    Best hour (n≥3): {bh_hr:02d}:00 UTC  PF {wh_pf:.3f}")

    print(f"\n{BLK}Q8  Is there evidence that ONE market condition consistently produces better trades?")
    top_feat_name = _top_feat(1)
    top_d         = float(attr.iloc[0]["cohens_d"])
    if abs(top_d) >= 0.5:
        direction = "higher" if top_d > 0 else "lower"
        print(f"{BLK}    → YES.  '{top_feat_name}' is {direction} in winning trades  "
              f"(|d|={abs(top_d):.3f}, medium-large effect).")
        print(f"{BLK}    This is the single strongest discriminating condition found.")
    elif abs(top_d) >= 0.2:
        print(f"{BLK}    → POSSIBLE.  '{top_feat_name}' shows a small but real effect  "
              f"(|d|={abs(top_d):.3f}).")
        print(f"{BLK}    Consistent pattern but not a dominant single factor.")
    else:
        print(f"{BLK}    → NO strong single condition found.  "
              f"Best feature |d|={abs(top_d):.3f} (negligible).")
        print(f"{BLK}    Win/loss outcomes appear largely driven by random execution noise.")

    # ── Final ranked summary ───────────────────────────────────────────────
    print(f"\n{BLK}FINAL RANKED LIST — Market Characteristics Explaining Liq.Sweep Outcomes")
    print(f"{BLK}{'─' * 80}")
    for _, row in attr.iterrows():
        strength = ("★★★ LARGE   " if abs(row["cohens_d"]) >= 0.8 else
                    "★★  MEDIUM  " if abs(row["cohens_d"]) >= 0.5 else
                    "★   SMALL   " if abs(row["cohens_d"]) >= 0.2 else
                    "·   MINIMAL ")
        direction = ("↑ WIN-FAVOURED " if row["cohens_d"] > 0 else
                     "↓ LOSS-FAVOURED" if row["cohens_d"] < 0 else "   NEUTRAL   ")
        cr = f"r={row['correlation']:+.3f}" if not math.isnan(row["correlation"]) else ""
        print(f"{BLK}  {int(row['rank']):>2}. {strength} "
              f"{direction}  "
              f"{FEATURE_NAMES.get(row['feature'], row['feature']):<34}  "
              f"|d|={abs(row['cohens_d']):.3f}  {cr}")

    print(f"\n{BLK}NOTE: No filters have been added.  "
          f"This is purely descriptive attribution.\n"
          f"{BLK}These findings define the hypothesis for Research #006.")
    print(S)


# =============================================================================
# SECTION 7 — SAVE OUTPUTS
# =============================================================================

def save_attribution_csv(combined_df: pd.DataFrame) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, "r005_attribution_trades.csv")
    combined_df.to_csv(path, index=False)
    return path


def save_feature_importance_csv(attr: pd.DataFrame) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, "r005_feature_importance.csv")
    out  = attr.copy()
    out["feature_label"] = out["feature"].map(
        lambda c: FEATURE_NAMES.get(c, c)
    )
    cols = ["rank", "feature_label", "n_wins", "n_losses",
            "mean_wins", "mean_losses", "diff", "pct_diff",
            "cohens_d", "abs_d", "correlation"]
    out[[c for c in cols if c in out.columns]].to_csv(path, index=False)
    return path


# =============================================================================
# SECTION 8 — MAIN PIPELINE
# =============================================================================

def process_symbol_r005(symbol: str, funding_df: pd.DataFrame) -> dict:
    """
    Run Liq.Sweep backtest on one symbol, enrich trades, return enriched DataFrame.
    """
    sep = "─" * 90
    print(f"\n{sep}\n  PROCESSING: {symbol}\n{sep}")

    df = get_data(symbol)
    n  = len(df)
    print(f"  Total candles : {n:,}")

    warm_up = CONFIG["EMA_LENGTH"] * 3 + 100  # generous warm-up
    if n < warm_up:
        print(f"  [SKIP] Insufficient data.")
        return {}

    # Add R004 indicators, then R005 extras
    df = add_indicators(df)
    df = add_r005_indicators(df)

    # Train / OOS split — UNCHANGED
    split  = int(n * CONFIG["TRAIN_RATIO"])
    df_oos = df.iloc[split:].reset_index(drop=True)

    oos_start = str(df_oos["datetime"].iloc[0].date())
    oos_end   = str(df_oos["datetime"].iloc[-1].date())
    n_days    = (df_oos["datetime"].iloc[-1] - df_oos["datetime"].iloc[0]).days
    print(f"  Train : {df['datetime'].iloc[0].date()} → "
          f"{df['datetime'].iloc[split-1].date()} ({split:,} bars)")
    print(f"  OOS   : {oos_start} → {oos_end} ({len(df_oos):,} bars / {n_days}d)")

    # ── Run backtest — LOCKED ENGINE UNCHANGED ────────────────────────────
    res  = run_backtest(df_oos, strategy_lsr, "Liq.Sweep")
    m    = compute_metrics(res["trades"], "Liq.Sweep")
    mc   = monte_carlo(m["pnls"], CONFIG["MC_ITERATIONS"])
    verdict = _verdict_from_metrics(m, mc)

    print(f"  Liq.Sweep  n={m['n_trades']:>4}  "
          f"PF {m['profit_factor']:>6.3f}  "
          f"Exp {m['expectancy_r']:>+6.3f}R  "
          f"Net ${m['net_profit']:>8,.0f}  "
          f"MDD {m['max_drawdown']:.2%}  → {verdict}")

    # ── Attach funding rate ────────────────────────────────────────────────
    attach_funding_rate(res["trades"], funding_df)

    # ── Enrich trades with market context ────────────────────────────────
    enriched = enrich_trades_with_context(res["trades"], df_oos)
    print(f"  Enriched {len(enriched)} trade records with market context.")

    # ── Per-symbol feature comparison chart ──────────────────────────────
    if len(enriched) >= 4:
        attr_sym = compute_attribution(enriched)
        p = plot_feature_comparison(attr_sym, symbol, len(enriched))
        if p:
            print(f"  → {p}")

    return {
        "metrics":  m,
        "mc":       mc,
        "verdict":  verdict,
        "enriched": enriched,
        "oos_start": oos_start,
        "oos_end":   oos_end,
    }


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              QUANTLAB AI — RESEARCH #005                                      ║
║   Trade Attribution & Edge Discovery — Liquidity Sweep Reversal               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  Strategy tested : Liquidity Sweep Reversal  (identical to R004)
  Engine          : LOCKED — zero changes to execution, fees, RR, sizing
  Purpose         : Understand WHY trades win and WHY they lose
  Method          : Market context capture → attribution → feature importance
""")

    random.seed(42)
    np.random.seed(42)

    symbols      = CONFIG["SYMBOLS"]
    symbol_results = {}
    all_enriched   = []

    for sym in symbols:
        print(f"\n[Funding Rates] {sym}")
        try:
            funding_df = get_funding_rates(sym)
        except Exception as e:
            print(f"  [WARN] Could not fetch funding rates: {e}")
            funding_df = pd.DataFrame(columns=["datetime", "funding_rate"])

        try:
            res = process_symbol_r005(sym, funding_df)
            if res:
                symbol_results[sym] = res
                if len(res["enriched"]) > 0:
                    res["enriched"]["symbol"] = sym
                    all_enriched.append(res["enriched"])
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {sym}: {exc}")
            traceback.print_exc()

    if not all_enriched:
        print("\n  No trades collected.  Cannot run attribution analysis.")
        return

    # ── Combine all symbols ───────────────────────────────────────────────
    combined_df = pd.concat(all_enriched, ignore_index=True)
    n_total     = len(combined_df)
    print(f"\n  Combined trade set: {n_total} trades across {len(all_enriched)} symbols")

    if n_total < 10:
        print("  [WARN] Too few trades for reliable attribution. Proceeding anyway.")

    # ── Attribution analysis (combined) ──────────────────────────────────
    print("\n  Running attribution analysis…")
    attr      = compute_attribution(combined_df)
    hourly, daily, session = compute_time_analysis(combined_df)
    vol_df    = compute_volatility_analysis(combined_df)
    adx_df    = compute_adx_analysis(combined_df)

    # ── Charts ────────────────────────────────────────────────────────────
    print("  Generating charts…")
    charts = []

    p = plot_correlation_matrix(combined_df)
    if p:
        charts.append(p)
        print(f"  → {p}")

    p = plot_time_analysis(hourly, daily, session)
    if p:
        charts.append(p)
        print(f"  → {p}")

    p = plot_time_heatmap(combined_df)
    if p:
        charts.append(p)
        print(f"  → {p}")

    p = plot_group_analysis(
        vol_df, "group",
        "Volatility Analysis — Performance by ATR Regime",
        "r005_volatility_analysis.png",
        subtitle="Liq.Sweep Reversal  |  All Symbols Combined  |  R005",
    )
    if p:
        charts.append(p)
        print(f"  → {p}")

    p = plot_group_analysis(
        adx_df, "group",
        "Trend Analysis — Performance by ADX Regime",
        "r005_adx_analysis.png",
        subtitle="Liq.Sweep Reversal  |  All Symbols Combined  |  R005",
    )
    if p:
        charts.append(p)
        print(f"  → {p}")

    p = plot_feature_importance(attr)
    if p:
        charts.append(p)
        print(f"  → {p}")

    # Scatter: top 6 features
    top_feat_cols = [r["feature"] for _, r in attr.head(6).iterrows()]
    p = plot_r_multiple_scatters(combined_df, top_feat_cols)
    if p:
        charts.append(p)
        print(f"  → {p}")

    # ── Save data files ───────────────────────────────────────────────────
    print("  Saving data files…")
    p1 = save_attribution_csv(combined_df)
    p2 = save_feature_importance_csv(attr)
    print(f"  → {p1}")
    print(f"  → {p2}")

    # ── Print report ──────────────────────────────────────────────────────
    print_attribution_report(
        attr, hourly, daily, session,
        vol_df, adx_df, combined_df, symbol_results,
    )

    # ── Append to research journal ────────────────────────────────────────
    jnl_rows = []
    for sym, res in symbol_results.items():
        row       = _journal_row("Liq.Sweep", sym, res["metrics"],
                                  res["mc"], res["verdict"])
        row["research_id"] = RESEARCH_ID
        jnl_rows.append(row)
    if jnl_rows:
        append_journal(jnl_rows)
        print(f"\n  Research journal updated → {CONFIG['JOURNAL_FILE']}")
        print(f"  ({len(jnl_rows)} rows appended)\n")

    print(f"  All outputs → {OUTPUT_FOLDER}/")
    print(f"  Research #005 complete.\n")


if __name__ == "__main__":
    main()
