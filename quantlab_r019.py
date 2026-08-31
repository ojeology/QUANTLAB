"""
=============================================================================
QUANTLAB AI — RESEARCH #019
Objective : Volatility Compression Breakout — Full Validation

Deep-dive of the VCB strategy flagged in Research #004.
R004 flagged:
  VolCompression/DOGE/1H : PF=2.58  WR=91%  n=11  → PROMOTE (thin sample)
  VolCompression/XRP/1H  : PF=1.145 WR=45%  n=20  → WEAK
  All others             : REJECT

Research Questions:
  1. Is the DOGE result genuine edge or small-sample luck?
  2. Does VCB have a consistent edge across all 8 symbols?
  3. What ATR-percentile / breakout-lookback parameters are robust?
  4. Does longer compression duration → better outcomes?
  5. Does 15m confirm 1H findings?

Strategy (EXACT R004 definition — LOCKED):
  Compression : ATR(14) < rolling 50-bar ATR percentile-30
  Breakout    : close > prior N-bar high (N=VCB_BREAK_BARS)
  Trend       : close > EMA(200)
  Stop        : low of signal bar
  Entry       : next bar open
  RR          : 2:1
  All execution costs locked to CONFIG.

Parameter sensitivity grid (analysis only — not live optimisation):
  ATR_PCTILE  ∈ {20, 25, 30, 35, 40}
  BREAK_BARS  ∈ {5, 10, 15, 20}

Data    : Existing cache — no new downloads
Symbols : BTC, ETH, LINK, XRP, DOGE, LTC, AVAX, BCH  (OKX perps)
Split   : 70/30 chronological train/OOS
=============================================================================
"""

import os, sys, math, warnings, itertools
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantlab_ai import (
    CONFIG, compute_metrics, monte_carlo,
    append_journal, _journal_row, _verdict_from_metrics,
    calc_ema, calc_atr, calc_adx,
    JOURNAL_COLS,
)

# =============================================================================
# CONSTANTS
# =============================================================================
RESEARCH_ID   = "R019"
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
EMA_LEN      = CONFIG["EMA_LENGTH"]
MC_ITER      = CONFIG["MC_ITERATIONS"]

# R004 baseline parameters (locked for the main backtest)
BASE_ATR_LEN   = CONFIG["VCB_ATR_LENGTH"]   # 14
BASE_ATR_WIN   = CONFIG["VCB_ATR_WINDOW"]   # 50
BASE_ATR_PCT   = CONFIG["VCB_ATR_PCTILE"]   # 30
BASE_BREAK     = CONFIG["VCB_BREAK_BARS"]   # 10

# Parameter grid for sensitivity analysis
GRID_PCTILE    = [20, 25, 30, 35, 40]
GRID_BREAK     = [5, 10, 15, 20]

PROMOTE_PF     = 1.20
PROMOTE_MC_PP  = 0.55
MIN_TRADES     = 30
WATCHLIST_PF   = 1.05

SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "LINK-USDT-SWAP", "AVAX-USDT-SWAP",
    "XRP-USDT-SWAP",  "DOGE-USDT-SWAP", "LTC-USDT-SWAP",  "BCH-USDT-SWAP",
]

TIMEFRAMES = [
    {"bar": "1H",  "minutes": 60,  "label": "1-hour"},
    {"bar": "15m", "minutes": 15,  "label": "15-minute"},
]

BG       = "#0F1117"
PALETTE  = ["#4A90D9","#FFB347","#00C49A","#FF4560",
            "#E040FB","#FFD700","#00D4FF","#FF6B6B"]


# =============================================================================
# SECTION 1 — DATA LOADING
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
# SECTION 2 — VCB INDICATORS
# =============================================================================

def add_vcb_indicators(df: pd.DataFrame, atr_pctile: int, break_bars: int) -> pd.DataFrame:
    """
    Compute VCB indicators.  Exact R004 definition with parameterised thresholds.
    """
    df = df.copy()
    df["ema200"]     = calc_ema(df["close"], EMA_LEN)
    df["atr"]        = calc_atr(df, BASE_ATR_LEN)
    df["atr_pctile"] = df["atr"].rolling(BASE_ATR_WIN).quantile(atr_pctile / 100.0)
    df["compressed"] = df["atr"] < df["atr_pctile"]
    df["vcb_range_h"]= df["high"].shift(1).rolling(break_bars).max()

    # Compression duration: how many consecutive compressed bars ending on signal bar
    # (used for quality analysis — computed via run-length encoding)
    comp_arr = df["compressed"].values.astype(int)
    durations = np.zeros(len(comp_arr), dtype=int)
    run = 0
    for i in range(len(comp_arr)):
        if comp_arr[i]:
            run += 1
        else:
            run = 0
        durations[i] = run
    df["comp_duration"] = durations

    return df


def signal_vcb(df: pd.DataFrame) -> pd.Series:
    """
    Exact R004 VCB signal.
    All conditions evaluated on the current bar close; entry on next bar open.
    """
    compressed = df["compressed"]
    breakout   = df["close"] > df["vcb_range_h"]
    trend      = df["close"] > df["ema200"]
    valid      = df["atr_pctile"].notna() & df["vcb_range_h"].notna()
    return compressed & breakout & trend & valid


# =============================================================================
# SECTION 3 — BACKTEST ENGINE (locked to CONFIG)
# =============================================================================

def run_vcb_backtest(df: pd.DataFrame, label: str, bar_minutes: int) -> list:
    """
    Event-driven backtest of VCB strategy.
    Execution parameters locked to CONFIG.
    Returns list of trade dicts with extended VCB-specific fields.
    """
    signals = signal_vcb(df)
    n       = len(df)

    in_pos     = False
    ep         = 0.0
    sl         = 0.0
    tp         = 0.0
    entry_idx  = -1
    entry_time = None
    pos_size   = 0.0
    capital    = STARTING_CAP
    trades     = []

    close_arr    = df["close"].values
    high_arr     = df["high"].values
    low_arr      = df["low"].values
    open_arr     = df["open"].values
    dt_arr       = df["datetime"].values
    comp_dur_arr = df["comp_duration"].values
    atr_arr      = df["atr"].values
    vcb_rh_arr   = df["vcb_range_h"].values

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
                    # VCB-specific
                    "comp_duration":   int(comp_dur_arr[i - 1]),
                    "signal_atr":      float(atr_arr[i - 1]),
                    "breakout_dist_pct": float((ep - vcb_rh_arr[i - 1]) / vcb_rh_arr[i - 1] * 100)
                                        if vcb_rh_arr[i - 1] > 0 else 0.0,
                })
                capital  += net_pnl
                in_pos    = False
            continue

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

    return trades


# =============================================================================
# SECTION 4 — PER-SYMBOL METRICS (extended)
# =============================================================================

def extended_metrics(trades: list, label: str, bar_minutes: int) -> dict:
    """Compute metrics + VCB-specific compression quality fields."""
    base = compute_metrics(trades, label) if trades else {
        "label": label, "n_trades": 0, "net_profit": 0.0,
        "profit_factor": 0.0, "win_rate": 0.0, "avg_r": 0.0,
        "expectancy_r": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
        "avg_hold_minutes": 0.0,
        "equity": np.array([STARTING_CAP]), "drawdown": np.array([0.0]),
        "pnls": np.array([]), "r_multiples": np.array([]),
    }

    if not trades:
        base.update({
            "avg_comp_duration": 0.0,
            "med_comp_duration": 0.0,
            "avg_breakout_dist": 0.0,
            "comp_dur_corr_r":   np.nan,
        })
        return base

    df = pd.DataFrame(trades)
    base["avg_comp_duration"] = df["comp_duration"].mean()
    base["med_comp_duration"] = df["comp_duration"].median()
    base["avg_breakout_dist"] = df["breakout_dist_pct"].mean()

    # Correlation: compression duration → R-multiple
    if len(df) >= 5 and df["comp_duration"].std() > 0:
        r, _ = scipy_stats.pearsonr(df["comp_duration"], df["r_multiple"])
        base["comp_dur_corr_r"] = r
    else:
        base["comp_dur_corr_r"] = np.nan

    # Recompute Sharpe with correct bar_minutes (compute_metrics uses CONFIG["TIMEFRAME"])
    pnls = base["pnls"]
    if len(pnls) > 1:
        std    = np.std(pnls, ddof=1)
        bpy    = 365 * 24 * 60 / bar_minutes
        base["sharpe"] = pnls.mean() / std * math.sqrt(bpy) if std > 0 else 0.0

    return base


# =============================================================================
# SECTION 5 — PARAMETER GRID
# =============================================================================

def run_param_grid(df_oos: pd.DataFrame, bar_minutes: int) -> dict:
    """
    Run all (ATR_PCTILE × BREAK_BARS) combinations on OOS data.
    Returns: {(pctile, break_bars): metrics_dict}
    """
    results = {}
    for pctile, brk in itertools.product(GRID_PCTILE, GRID_BREAK):
        df_ind = add_vcb_indicators(df_oos, pctile, brk)
        df_ind = df_ind.dropna(subset=["ema200", "atr_pctile", "vcb_range_h"]).reset_index(drop=True)
        trades = run_vcb_backtest(df_ind, f"vcb_p{pctile}_b{brk}", bar_minutes)
        m      = extended_metrics(trades, f"p{pctile}_b{brk}", bar_minutes)
        results[(pctile, brk)] = m
    return results


# =============================================================================
# SECTION 6 — STATISTICAL VALIDATION
# =============================================================================

def validate_trades(trades: list) -> dict:
    """
    Bootstrap CI on mean R-multiple.
    Mann-Whitney U vs zero-mean null (trades vs shuffled zeros).
    """
    if len(trades) < 5:
        return {
            "n": len(trades), "mean_r": np.nan,
            "boot_ci": (np.nan, np.nan),
            "mwu_vs_zero_p": np.nan, "significant": False,
        }

    rmul = np.array([t["r_multiple"] for t in trades])
    rng  = np.random.RandomState(42)

    # Bootstrap 95% CI on mean R
    boots = [rng.choice(rmul, len(rmul), replace=True).mean() for _ in range(5000)]
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    # Mann-Whitney U: trades vs same number of zero outcomes
    zeros = np.zeros(len(rmul))
    u_stat, p_val = scipy_stats.mannwhitneyu(rmul, zeros, alternative="greater")

    return {
        "n":             len(trades),
        "mean_r":        float(rmul.mean()),
        "boot_ci":       (ci_lo, ci_hi),
        "mwu_vs_zero_p": float(p_val),
        "significant":   p_val < 0.05 and ci_lo > 0,
    }


# =============================================================================
# SECTION 7 — VISUALISATIONS
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


# ── Chart 1: Equity curves per symbol ────────────────────────────────────────

def plot_equity_curves(sym_metrics: dict, tf_bar: str):
    symbols = list(sym_metrics.keys())
    ncols   = 4
    nrows   = math.ceil(len(symbols) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), facecolor=BG)
    fig.patch.set_facecolor(BG)
    axes_flat = np.array(axes).flatten() if nrows > 1 else np.array(axes)

    for ax, sym in zip(axes_flat, symbols):
        _ax(ax)
        m   = sym_metrics[sym]
        col = PALETTE[symbols.index(sym) % len(PALETTE)]
        if m["n_trades"] > 0:
            eq = m["equity"]
            ax.plot(np.arange(len(eq)), eq, color=col, lw=1.8)
            ax.fill_between(np.arange(len(eq)), eq, STARTING_CAP,
                            where=eq >= STARTING_CAP, alpha=0.15, color=col)
            ax.fill_between(np.arange(len(eq)), eq, STARTING_CAP,
                            where=eq < STARTING_CAP, alpha=0.15, color="#FF4560")
        ax.axhline(STARTING_CAP, color="#555", lw=0.7, ls="--")
        short = sym.split("-")[0]
        pf    = m["profit_factor"]
        wr    = m["win_rate"] * 100
        ax.set_title(f"{short}  n={m['n_trades']}  PF={pf:.3f}  WR={wr:.0f}%",
                     fontsize=8)
        ax.set_ylabel("Capital ($)", fontsize=7)

    for ax in axes_flat[len(symbols):]:
        ax.set_visible(False)

    fig.suptitle(f"R019 VCB Equity Curves [{tf_bar}]  (base params: ATR%<{BASE_ATR_PCT}, break={BASE_BREAK}b)",
                 color="#EEE", fontsize=11, y=1.01)
    _save(fig, f"r019_equity_curves_{tf_bar}.png")


# ── Chart 2: Drawdown per symbol ─────────────────────────────────────────────

def plot_drawdown_curves(sym_metrics: dict, tf_bar: str):
    symbols = list(sym_metrics.keys())
    ncols   = 4
    nrows   = math.ceil(len(symbols) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), facecolor=BG)
    fig.patch.set_facecolor(BG)
    axes_flat = np.array(axes).flatten() if nrows > 1 else np.array(axes)

    for ax, sym in zip(axes_flat, symbols):
        _ax(ax)
        m   = sym_metrics[sym]
        col = PALETTE[symbols.index(sym) % len(PALETTE)]
        if m["n_trades"] > 0:
            dd = m["drawdown"]
            ax.fill_between(range(len(dd)), dd * 100, 0, color=col, alpha=0.6)
            ax.plot(dd * 100, color=col, lw=1)
            ax.axhline(-20, color="#FFF", lw=0.5, ls="--", alpha=0.4)
        short = sym.split("-")[0]
        mdd   = m["max_drawdown"] * 100
        ax.set_title(f"{short}  MDD={mdd:.1f}%", fontsize=8)
        ax.set_ylabel("DD %", fontsize=7)

    for ax in axes_flat[len(symbols):]:
        ax.set_visible(False)

    fig.suptitle(f"R019 VCB Drawdown [{tf_bar}]", color="#EEE", fontsize=11, y=1.01)
    _save(fig, f"r019_drawdown_{tf_bar}.png")


# ── Chart 3: Symbol comparison bar chart ─────────────────────────────────────

def plot_symbol_comparison(sym_metrics: dict, sym_validation: dict, tf_bar: str):
    symbols = list(sym_metrics.keys())
    short   = [s.split("-")[0] for s in symbols]
    x       = np.arange(len(symbols))
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(symbols))]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=BG)
    fig.patch.set_facecolor(BG)

    metrics_to_plot = [
        ("profit_factor", "Profit Factor", PROMOTE_PF, "PF"),
        ("win_rate",      "Win Rate (%)",  1/3,        "WR"),
        ("expectancy_r",  "Expectancy (R)", 0.0,       "ExpR"),
    ]

    for ax, (key, title, ref, ylabel) in zip(axes, metrics_to_plot):
        _ax(ax)
        vals = []
        edge_colors = []
        for sym in symbols:
            m = sym_metrics[sym]
            v = m.get(key, 0.0)
            if not np.isfinite(v):
                v = 0.0
            if key == "win_rate":
                v *= 100
            vals.append(v)
            # Gold border if statistically significant
            sv = sym_validation.get(sym, {})
            edge_colors.append("#FFD700" if sv.get("significant", False) else "#222")

        bars = ax.bar(x, vals, color=colors, alpha=0.85,
                      edgecolor=edge_colors, linewidth=1.5)
        ref_v = ref * 100 if key == "win_rate" else ref
        ax.axhline(ref_v, color="#FFF", lw=0.8, ls="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=9, color="#EEE")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)

        for bar, v in zip(bars, vals):
            offset = abs(v) * 0.04 + 0.01
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (offset if v >= 0 else -offset * 3),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#EEE")

    fig.suptitle(f"R019 VCB Symbol Comparison [{tf_bar}]  (gold border = significant)",
                 color="#EEE", fontsize=11, y=1.01)
    _save(fig, f"r019_symbol_comparison_{tf_bar}.png")


# ── Chart 4: Parameter sensitivity heatmap ───────────────────────────────────

def plot_param_heatmap(portfolio_grid: dict, tf_bar: str):
    """
    Heatmap of portfolio PF across (ATR_PCTILE, BREAK_BARS) grid.
    """
    p_arr = sorted(set(p for p, _ in portfolio_grid.keys()))
    b_arr = sorted(set(b for _, b in portfolio_grid.keys()))
    n_arr = np.zeros((len(p_arr), len(b_arr)))
    pf_arr = np.zeros_like(n_arr)
    wr_arr = np.zeros_like(n_arr)

    for i, p in enumerate(p_arr):
        for j, b in enumerate(b_arr):
            m = portfolio_grid.get((p, b), {})
            pf_arr[i, j] = m.get("profit_factor", 0.0) if np.isfinite(m.get("profit_factor", 0.0)) else 0.0
            n_arr[i, j]  = m.get("n_trades", 0)
            wr_arr[i, j] = m.get("win_rate", 0.0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, data, title, fmt in [
        (axes[0], pf_arr, "Profit Factor", ".3f"),
        (axes[1], wr_arr, "Win Rate (%)",  ".1f"),
    ]:
        _ax(ax)
        vmax = max(data.max(), PROMOTE_PF * 1.2)
        im   = ax.imshow(data, cmap="RdYlGn", vmin=0.7, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(b_arr)))
        ax.set_xticklabels([f"{b}b" for b in b_arr], color="#EEE", fontsize=9)
        ax.set_yticks(range(len(p_arr)))
        ax.set_yticklabels([f"ATR<{p}%" for p in p_arr], color="#EEE", fontsize=9)
        ax.set_xlabel("Breakout lookback (bars)")
        ax.set_ylabel("ATR compression threshold")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.045)

        for i in range(len(p_arr)):
            for j in range(len(b_arr)):
                n = int(n_arr[i, j])
                v = data[i, j]
                ax.text(j, i, f"{v:{fmt}}\nn={n}",
                        ha="center", va="center", fontsize=7,
                        color="white" if data[i, j] > vmax * 0.6 else "#333",
                        fontweight="bold" if n >= MIN_TRADES else "normal")

    # Mark base params
    base_i = p_arr.index(BASE_ATR_PCT) if BASE_ATR_PCT in p_arr else -1
    base_j = b_arr.index(BASE_BREAK)   if BASE_BREAK   in b_arr else -1
    if base_i >= 0 and base_j >= 0:
        for ax in axes:
            ax.add_patch(plt.Rectangle((base_j - 0.5, base_i - 0.5), 1, 1,
                                        fill=False, edgecolor="#FFD700", lw=2.5,
                                        label="R004 base"))

    fig.suptitle(f"R019 VCB Parameter Sensitivity [{tf_bar}]  (portfolio, all 8 symbols combined)\n"
                 f"Gold border = R004 base params  |  Bold n = sample ≥ {MIN_TRADES}",
                 color="#EEE", fontsize=10, y=1.01)
    _save(fig, f"r019_param_heatmap_{tf_bar}.png")


# ── Chart 5: Monte Carlo (top symbols) ───────────────────────────────────────

def plot_monte_carlo(sym_metrics: dict, tf_bar: str):
    symbols = list(sym_metrics.keys())
    ncols = 4
    nrows = math.ceil(len(symbols) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows), facecolor=BG)
    fig.patch.set_facecolor(BG)
    axes_flat = np.array(axes).flatten() if nrows > 1 else np.array(axes)

    for ax, sym in zip(axes_flat, symbols):
        _ax(ax)
        m   = sym_metrics[sym]
        col = PALETTE[symbols.index(sym) % len(PALETTE)]
        short = sym.split("-")[0]

        if m["n_trades"] >= 5:
            mc     = monte_carlo(m["pnls"], MC_ITER)
            finals = mc["final_equities"]
            if len(np.unique(finals)) > 1:
                ax.hist(finals, bins=min(40, len(np.unique(finals))),
                        color=col, alpha=0.75, density=True)
            else:
                ax.axvline(finals[0], color=col, lw=3)
            ax.axvline(STARTING_CAP, color="#FFF", lw=1.2, ls="--")
            ax.axvline(mc["median"],  color="#FFD700", lw=1.0,
                       label=f"Med ${mc['median']:,.0f}")
            pp = mc["prob_profit"] * 100
            ax.set_title(f"{short}  PP={pp:.0f}%  n={m['n_trades']}", fontsize=8)
            ax.legend(fontsize=7, facecolor="#1A1D24", edgecolor="#444", labelcolor="#EEE")
        else:
            ax.text(0.5, 0.5, f"{short}\nInsufficient trades",
                    ha="center", va="center", color="#AAA", transform=ax.transAxes)

    for ax in axes_flat[len(symbols):]:
        ax.set_visible(False)

    fig.suptitle(f"R019 VCB Monte Carlo by Symbol [{tf_bar}]  ({MC_ITER:,} permutations)",
                 color="#EEE", fontsize=11, y=1.01)
    _save(fig, f"r019_monte_carlo_{tf_bar}.png")


# ── Chart 6: Bootstrap CI by symbol ──────────────────────────────────────────

def plot_bootstrap_ci(sym_metrics: dict, sym_validation: dict, tf_bar: str):
    symbols  = list(sym_metrics.keys())
    short    = [s.split("-")[0] for s in symbols]

    means, lo_ci, hi_ci, sig_flags = [], [], [], []
    for sym in symbols:
        v  = sym_validation.get(sym, {})
        ci = v.get("boot_ci", (np.nan, np.nan))
        means.append(v.get("mean_r", np.nan))
        lo_ci.append(ci[0])
        hi_ci.append(ci[1])
        sig_flags.append(v.get("significant", False))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # Left: CI plot
    ax = axes[0]; _ax(ax)
    x_pos = np.arange(len(symbols))
    for xi, (m_v, lo, hi, col, sig) in enumerate(
            zip(means, lo_ci, hi_ci,
                [PALETTE[i % len(PALETTE)] for i in range(len(symbols))],
                sig_flags)):
        lw  = 3.5 if sig else 1.5
        al  = 1.0 if sig else 0.5
        ax.plot([xi, xi], [lo, hi], color=col, lw=lw, alpha=al)
        ax.scatter([xi], [m_v], color=col, s=80, zorder=5, alpha=al)

    ax.axhline(0, color="#FFF", lw=0.8, ls="--", label="Zero R")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short, color="#EEE", fontsize=9)
    ax.set_ylabel("Mean R-Multiple")
    ax.set_title("Bootstrap 95% CI — Mean R-Multiple by Symbol\n(bright = statistically significant)", fontsize=9)
    ax.legend(fontsize=8)

    # Right: statistical summary table
    ax2 = axes[1]; _ax(ax2); ax2.axis("off")
    lines = ["Statistical Validation — VCB per Symbol", ""]
    hdr   = f"{'Symbol':<8} {'n':>5}  {'MeanR':>7}  {'CI_lo':>7}  {'CI_hi':>7}  {'p_MWU':>8}  {'Sig':>4}"
    lines.append(hdr)
    lines.append("─" * 55)
    for sym in symbols:
        short_s = sym.split("-")[0]
        v  = sym_validation.get(sym, {})
        n  = v.get("n", 0)
        mr = v.get("mean_r", np.nan)
        ci = v.get("boot_ci", (np.nan, np.nan))
        pv = v.get("mwu_vs_zero_p", np.nan)
        sg = "✓" if v.get("significant", False) else " "
        if np.isfinite(mr):
            lines.append(f"{short_s:<8} {n:>5}  {mr:>+7.3f}  "
                         f"{ci[0]:>7.3f}  {ci[1]:>7.3f}  "
                         f"{pv:>8.4f}  {sg:>4}")
        else:
            lines.append(f"{short_s:<8} {n:>5}  {'–':>7}  {'–':>7}  {'–':>7}  {'–':>8}  {'–':>4}")

    for i, line in enumerate(lines):
        is_sig_line = i > 3 and "✓" in line
        color = "#FFD700" if is_sig_line else ("#EEE" if i == 0 else "#AAA")
        ax2.text(0.02, 0.96 - i * 0.055, line, transform=ax2.transAxes,
                 fontsize=8, color=color, va="top", fontfamily="monospace")

    fig.suptitle(f"R019 VCB Bootstrap CI & Statistical Tests [{tf_bar}]",
                 color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"r019_bootstrap_ci_{tf_bar}.png")


# ── Chart 7: Compression quality analysis ────────────────────────────────────

def plot_compression_analysis(all_trades: list, tf_bar: str):
    if not all_trades:
        return

    df = pd.DataFrame(all_trades)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BG)
    fig.patch.set_facecolor(BG)

    wins = df[df["win"]]
    loss = df[~df["win"]]

    # Panel 1: R-multiple distribution
    ax = axes[0, 0]; _ax(ax)
    rmul = df["r_multiple"].values
    ax.hist(rmul, bins=30, color="#4A90D9", alpha=0.75, edgecolor="#222")
    ax.axvline(0, color="#FFF", lw=0.8, ls="--")
    ax.axvline(rmul.mean(), color="#FFD700", lw=1.5, ls="-",
               label=f"Mean R={rmul.mean():.3f}")
    ax.set_title(f"R-Multiple Distribution  (n={len(df)})", fontsize=9)
    ax.set_xlabel("R-Multiple"); ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Panel 2: Compression duration vs outcome
    ax2 = axes[0, 1]; _ax(ax2)
    if len(wins) > 0:
        ax2.scatter(wins["comp_duration"], wins["r_multiple"],
                    color="#00C49A", s=30, alpha=0.6, label="Win (TP)")
    if len(loss) > 0:
        ax2.scatter(loss["comp_duration"], loss["r_multiple"],
                    color="#FF4560", s=30, alpha=0.6, label="Loss (SL)")
    ax2.axhline(0, color="#FFF", lw=0.6, ls="--")

    # Regression line
    if len(df) >= 5:
        slope, intercept, r_val, _, _ = scipy_stats.linregress(
            df["comp_duration"], df["r_multiple"])
        x_line = np.linspace(df["comp_duration"].min(), df["comp_duration"].max(), 50)
        ax2.plot(x_line, slope * x_line + intercept, color="#FFD700", lw=1.5,
                 label=f"r={r_val:.3f}")

    ax2.set_title("Compression Duration vs R-Multiple", fontsize=9)
    ax2.set_xlabel("Consecutive compressed bars before signal")
    ax2.set_ylabel("R-Multiple")
    ax2.legend(fontsize=8)

    # Panel 3: Holding time distribution
    ax3 = axes[1, 0]; _ax(ax3)
    ax3.hist(df["holding_minutes"] / 60, bins=25, color="#FFB347", alpha=0.75,
             edgecolor="#222", label="All trades")
    if len(wins) > 0:
        ax3.hist(wins["holding_minutes"] / 60, bins=20, color="#00C49A", alpha=0.5,
                 label="Wins", histtype="step", lw=2)
    ax3.set_title("Holding Time Distribution", fontsize=9)
    ax3.set_xlabel("Holding time (hours)")
    ax3.set_ylabel("Count")
    ax3.legend(fontsize=8)

    # Panel 4: Win rate by compression duration bucket
    ax4 = axes[1, 1]; _ax(ax4)
    bins      = [0, 3, 6, 10, 20, df["comp_duration"].max() + 1]
    labels_b  = ["1-3", "4-6", "7-10", "11-20", "20+"]
    wr_by_bin = []
    n_by_bin  = []
    for lo_b, hi_b, lbl in zip(bins[:-1], bins[1:], labels_b):
        sub = df[(df["comp_duration"] >= lo_b) & (df["comp_duration"] < hi_b)]
        wr_by_bin.append(sub["win"].mean() * 100 if len(sub) > 0 else 0)
        n_by_bin.append(len(sub))

    bars = ax4.bar(labels_b, wr_by_bin, color="#E040FB", alpha=0.85, edgecolor="#222")
    ax4.axhline(33.3, color="#FFF", lw=0.8, ls="--", alpha=0.6, label="33.3% (RR=2:1 break-even)")
    for bar, n_b, wr in zip(bars, n_by_bin, wr_by_bin):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 wr + 1, f"n={n_b}", ha="center", va="bottom", fontsize=8, color="#EEE")
    ax4.set_title("Win Rate by Compression Duration Bucket", fontsize=9)
    ax4.set_xlabel("Compressed bars before signal")
    ax4.set_ylabel("Win Rate (%)")
    ax4.legend(fontsize=8)

    fig.suptitle(f"R019 VCB Compression Quality Analysis [{tf_bar}]",
                 color="#EEE", fontsize=11, y=1.01)
    _save(fig, f"r019_compression_analysis_{tf_bar}.png")


# ── Chart 8: Portfolio summary dashboard ─────────────────────────────────────

def plot_portfolio_dashboard(sym_metrics: dict, sym_mc: dict, tf_bar: str):
    """Single-page summary: n_trades, PF, WR, MC prob-profit per symbol."""
    symbols = list(sym_metrics.keys())
    short   = [s.split("-")[0] for s in symbols]
    x       = np.arange(len(symbols))
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(symbols))]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor=BG)
    fig.patch.set_facecolor(BG)

    metrics_cfg = [
        ("n_trades",      "Trade Count",         None,        False),
        ("profit_factor", "Profit Factor",        PROMOTE_PF,  False),
        ("win_rate",      "Win Rate (%)",         1/3,         True),
        ("expectancy_r",  "Expectancy (R)",       0.0,         False),
    ]

    for ax, (key, title, ref, as_pct) in zip(axes.flat, metrics_cfg):
        _ax(ax)
        vals = []
        for sym in symbols:
            m = sym_metrics[sym]
            v = m.get(key, 0.0)
            if not np.isfinite(v):
                v = 0.0
            if as_pct:
                v *= 100
            vals.append(v)

        bar_objs = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="#222")
        if ref is not None:
            rv = ref * 100 if as_pct else ref
            ax.axhline(rv, color="#FFF", lw=0.8, ls="--", alpha=0.7)

        # Add MC prob-profit annotation on win-rate panel
        if key == "win_rate":
            for xi, sym in enumerate(symbols):
                mc = sym_mc.get(sym)
                if mc:
                    pp = mc["prob_profit"] * 100
                    ax.text(xi, vals[xi] + 1, f"MC:{pp:.0f}%",
                            ha="center", va="bottom", fontsize=6.5, color="#FFD700")

        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=9, color="#EEE")
        ax.set_title(title, fontsize=10)

        for bar, v in zip(bar_objs, vals):
            offset = abs(v) * 0.04 + 0.5
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (offset if v >= 0 else -offset * 3),
                    f"{v:.1f}" if key != "n_trades" else str(int(v)),
                    ha="center", va="bottom", fontsize=7.5, color="#EEE")

    # Promote threshold annotations
    fig.suptitle(
        f"R019 VCB Portfolio Dashboard [{tf_bar}]  —  "
        f"PROMOTE criteria: PF≥{PROMOTE_PF}  WR≥33%  MC≥{PROMOTE_MC_PP*100:.0f}%  n≥{MIN_TRADES}",
        color="#EEE", fontsize=11, y=1.01)
    _save(fig, f"r019_dashboard_{tf_bar}.png")


# =============================================================================
# SECTION 8 — CONSOLE REPORT
# =============================================================================

def print_report(sym_metrics: dict, sym_validation: dict,
                 sym_mc: dict, tf_cfg: dict):
    bar   = tf_cfg["bar"]
    label = tf_cfg["label"]
    W     = 120

    print()
    print("=" * W)
    print(f"  QUANTLAB AI — RESEARCH #019  [{label}]")
    print(f"  Volatility Compression Breakout — Full Validation")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * W)
    print()

    hdr = (f"  {'Symbol':<12} {'n':>5} {'WR%':>7} {'PF':>8} {'ExpR':>8} "
           f"{'NetP':>10} {'MDD%':>7} {'Sharpe':>8} {'AvgHold':>8} {'Sig':>4}")
    print(hdr)
    print("  " + "-" * (W - 2))

    totals = {"n": 0, "pnl_sum": 0.0, "wins": 0, "gross_w": 0.0, "gross_l": 0.0}
    for sym in sym_metrics:
        m     = sym_metrics[sym]
        v     = sym_validation.get(sym, {})
        short = sym.split("-")[0]
        flag  = "" if m["n_trades"] >= MIN_TRADES else " ⚠"
        sig   = "✓" if v.get("significant", False) else " "

        if m["n_trades"] == 0:
            print(f"  {short:<12} {'0':>5}  {'—':>6}  {'—':>7}  {'—':>7}  {'—':>9}  {'—':>6}  {'—':>7}  {'—':>7}  {'—':>3}")
        else:
            pf_s = f"{m['profit_factor']:.3f}" if np.isfinite(m['profit_factor']) else " ∞"
            print(f"  {short:<12} {m['n_trades']:>5}  "
                  f"{m['win_rate']*100:>6.1f}%  "
                  f"{pf_s:>7}  "
                  f"{m['expectancy_r']:>+7.3f}  "
                  f"${m['net_profit']:>9,.0f}  "
                  f"{m['max_drawdown']*100:>6.1f}%  "
                  f"{m['sharpe']:>7.2f}  "
                  f"{m['avg_hold_minutes']/60:>7.1f}h  "
                  f"{sig:>3}{flag}")
            totals["n"]       += m["n_trades"]
            totals["pnl_sum"] += m["net_profit"]

    print()


def print_final_questions(all_tf: dict, all_sym_metrics: dict):
    W = 120
    print()
    print("=" * W)
    print("  R019 — FINAL ASSESSMENT")
    print("  Volatility Compression Breakout — Deep Dive Verdict")
    print("=" * W)
    print()

    def _yn(c): return "YES ✓" if c else "NO  ✗"

    for bar, tf_data in all_tf.items():
        sym_metrics  = tf_data["sym_metrics"]
        sym_mc       = tf_data["sym_mc"]
        sym_val      = tf_data["sym_validation"]
        grid         = tf_data["portfolio_grid"]
        label        = tf_data["label"]

        all_trades   = tf_data["all_trades"]
        n_total      = sum(m["n_trades"] for m in sym_metrics.values())

        promoted     = [s for s, m in sym_metrics.items()
                        if m["n_trades"] >= MIN_TRADES
                        and m["profit_factor"] >= PROMOTE_PF
                        and m["expectancy_r"] > 0
                        and sym_mc.get(s, {}).get("prob_profit", 0) >= PROMOTE_MC_PP]

        watchlist    = [s for s, m in sym_metrics.items()
                        if s not in promoted
                        and m["n_trades"] >= 15
                        and WATCHLIST_PF <= m["profit_factor"] < PROMOTE_PF
                        and m["expectancy_r"] > 0]

        sig_symbols  = [s for s, v in sym_val.items() if v.get("significant", False)]

        best_sym     = max(sym_metrics, key=lambda s: (
            sym_metrics[s]["profit_factor"] if np.isfinite(sym_metrics[s]["profit_factor"]) else 0))
        best_m       = sym_metrics[best_sym]
        best_mc      = sym_mc.get(best_sym, {"prob_profit": 0.0})

        # Find best grid params (portfolio-level)
        best_grid_key = max(
            grid.keys(),
            key=lambda k: grid[k].get("profit_factor", 0.0)
                          if np.isfinite(grid[k].get("profit_factor", 0.0)) else 0.0)
        best_grid_m   = grid[best_grid_key]
        best_pctile, best_break = best_grid_key

        # Portfolio aggregate PF (all symbols combined)
        all_pnls = np.concatenate([m["pnls"] for m in sym_metrics.values() if len(m["pnls"])])
        if len(all_pnls) > 0:
            all_wins_sum  = all_pnls[all_pnls > 0].sum() if (all_pnls > 0).any() else 0.0
            all_loss_sum  = abs(all_pnls[all_pnls < 0].sum()) if (all_pnls < 0).any() else 1e-9
            portfolio_pf  = all_wins_sum / all_loss_sum
        else:
            portfolio_pf  = 0.0

        doge_m = sym_metrics.get("DOGE-USDT-SWAP", {})
        doge_n = doge_m.get("n_trades", 0)
        doge_pf= doge_m.get("profit_factor", 0.0)

        print(f"  ══ [{label}] ══════════════════════════════════════════════════════")
        print()

        print(f"  Q1. Is the R004 DOGE result (PF=2.58, n=11) genuine or artefact?")
        print(f"      DOGE this study : PF={doge_pf:.3f}  n={doge_n}  "
              f"Sig={_yn(sym_val.get('DOGE-USDT-SWAP',{}).get('significant',False))}")
        doge_confirms = doge_pf >= PROMOTE_PF and doge_n >= MIN_TRADES
        print(f"      → {_yn(doge_confirms)}  ({'confirmed at n≥30' if doge_confirms else 'not confirmed at n≥30'})")
        print()

        print(f"  Q2. Is edge consistent across symbols?")
        print(f"      PROMOTE symbols : {len(promoted)} / {len(sym_metrics)}  "
              f"({[s.split('-')[0] for s in promoted]})")
        print(f"      Watchlist       : {len(watchlist)} / {len(sym_metrics)}  "
              f"({[s.split('-')[0] for s in watchlist]})")
        print(f"      Statistically significant : {len(sig_symbols)} / {len(sym_metrics)}  "
              f"({[s.split('-')[0] for s in sig_symbols]})")
        consistent = len(promoted) >= 3
        print(f"      → {_yn(consistent)}  (≥3 promoted symbols)")
        print()

        print(f"  Q3. What parameters are most robust?")
        print(f"      R004 base (ATR<{BASE_ATR_PCT}%, break={BASE_BREAK}b) : "
              f"Portfolio PF={portfolio_pf:.3f}  n={n_total}")
        print(f"      Best grid params : ATR<{best_pctile}%  break={best_break}b  "
              f"PF={best_grid_m.get('profit_factor',0):.3f}  "
              f"n={best_grid_m.get('n_trades',0)}")
        print()

        print(f"  Q4. Does compression duration predict outcome?")
        if all_trades:
            df_all = pd.DataFrame(all_trades)
            if len(df_all) >= 5 and df_all["comp_duration"].std() > 0:
                corr, p_corr = scipy_stats.pearsonr(
                    df_all["comp_duration"], df_all["r_multiple"])
                print(f"      Pearson r(comp_duration, R-multiple) = {corr:.3f}  p={p_corr:.4f}")
                print(f"      → {_yn(p_corr < 0.05 and abs(corr) > 0.1)}")
            else:
                print(f"      → Insufficient data")
        print()

        print(f"  Q5. Does 15m confirm 1H findings?  (see below)")
        print()

        print(f"  Q6. Portfolio-level verdict [{label}]:")
        print(f"      Portfolio PF (all 8 symbols)  : {portfolio_pf:.3f}")
        print(f"      Total trades                  : {n_total}")
        print(f"      Promoted symbols              : {len(promoted)}")

        if len(promoted) >= 3 and portfolio_pf >= PROMOTE_PF:
            verdict = "PROMOTE — VCB shows consistent, statistically significant edge across ≥3 symbols"
        elif len(promoted) >= 1 or portfolio_pf >= WATCHLIST_PF:
            verdict = "WATCHLIST — Partial edge detected; edge is symbol-specific or below threshold"
        else:
            verdict = "REJECT — No consistent VCB edge found across the full 8-symbol universe"

        print(f"      → {verdict}")
        print()
        print("  " + "─" * (W - 2))
        print()

    print("=" * W)


# =============================================================================
# SECTION 9 — MAIN PIPELINE
# =============================================================================

def run_timeframe(tf_cfg: dict) -> dict:
    bar     = tf_cfg["bar"]
    bar_min = tf_cfg["minutes"]
    label   = tf_cfg["label"]

    print(f"\n{'='*72}")
    print(f"  TIMEFRAME: {label}  [{bar}]")
    print(f"{'='*72}")

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print(f"\n  Loading {bar} data from cache...", flush=True)
    raw_data = load_all(bar)
    if not raw_data:
        raise RuntimeError(f"No cached data for {bar}")
    for sym, df in raw_data.items():
        print(f"  {sym:20s}  {len(df):>8,} candles  "
              f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})")

    # ── 2. Base-parameter backtest per symbol ────────────────────────────────
    print(f"\n  Running base-param VCB backtest (ATR<{BASE_ATR_PCT}%, break={BASE_BREAK}b)...",
          flush=True)

    sym_metrics    = {}
    sym_mc         = {}
    sym_validation = {}
    sym_oos_dates  = {}
    all_trades     = []

    for sym, df_raw in raw_data.items():
        short = sym.split("-")[0]

        df_ind  = add_vcb_indicators(df_raw, BASE_ATR_PCT, BASE_BREAK)
        df_ind  = df_ind.dropna(subset=["ema200", "atr_pctile", "vcb_range_h"]).reset_index(drop=True)

        if len(df_ind) < 300:
            print(f"  {short:6s}  SKIP — insufficient data ({len(df_ind)} bars)")
            continue

        split_idx = int(len(df_ind) * TRAIN_RATIO)
        df_oos    = df_ind.iloc[split_idx:].reset_index(drop=True)
        sym_oos_dates[sym] = (str(df_oos["datetime"].iloc[0].date()),
                              str(df_oos["datetime"].iloc[-1].date()))

        trades = run_vcb_backtest(df_oos, sym, bar_min)
        m      = extended_metrics(trades, short, bar_min)
        mc     = monte_carlo(m["pnls"], MC_ITER) if m["n_trades"] >= 5 else \
                 {"median": 0.0, "p5": 0.0, "p95": 0.0, "prob_profit": 0.0, "final_equities": np.array([])}
        vstat  = validate_trades(trades)

        sym_metrics[sym]    = m
        sym_mc[sym]         = mc
        sym_validation[sym] = vstat
        all_trades.extend(trades)

        verdict_str = ("PROMOTE" if m["n_trades"] >= MIN_TRADES
                       and m["profit_factor"] >= PROMOTE_PF
                       and m["expectancy_r"] > 0
                       and mc["prob_profit"] >= PROMOTE_MC_PP
                       else ("WATCHLIST" if m.get("profit_factor", 0) >= WATCHLIST_PF
                             and m["expectancy_r"] > 0 and m["n_trades"] >= 15
                             else "REJECT"))

        print(f"  {short:6s}  n={m['n_trades']:4d}  "
              f"PF={m['profit_factor'] if np.isfinite(m['profit_factor']) else 999:.3f}  "
              f"WR={m['win_rate']*100:.1f}%  "
              f"ExpR={m['expectancy_r']:+.3f}  "
              f"MDD={m['max_drawdown']*100:.1f}%  "
              f"Sig={'✓' if vstat['significant'] else '✗'}  "
              f"→ {verdict_str}")

    # ── 3. Parameter grid (portfolio-level: sum pnls across symbols) ─────────
    print(f"\n  Running parameter grid ({len(GRID_PCTILE)}×{len(GRID_BREAK)} = "
          f"{len(GRID_PCTILE)*len(GRID_BREAK)} combinations)...", flush=True)

    # Build per-(pctile,break) portfolio
    portfolio_grid: dict = {}
    for pctile, brk in itertools.product(GRID_PCTILE, GRID_BREAK):
        combo_trades = []
        for sym, df_raw in raw_data.items():
            df_ind = add_vcb_indicators(df_raw, pctile, brk)
            df_ind = df_ind.dropna(subset=["ema200", "atr_pctile", "vcb_range_h"]).reset_index(drop=True)
            if len(df_ind) < 300:
                continue
            split_idx = int(len(df_ind) * TRAIN_RATIO)
            df_oos    = df_ind.iloc[split_idx:].reset_index(drop=True)
            trades    = run_vcb_backtest(df_oos, sym, bar_min)
            combo_trades.extend(trades)

        if combo_trades:
            pnls = np.array([t["pnl"] for t in combo_trades])
            wins = [t["win"] for t in combo_trades]
            n    = len(pnls)
            gw   = pnls[[w for w in wins]].sum() if any(wins) else 0.0
            gw   = sum(t["pnl"] for t in combo_trades if t["win"])
            gl   = abs(sum(t["pnl"] for t in combo_trades if not t["win"])) or 1e-9
            wr   = sum(1 for t in combo_trades if t["win"]) / n
            portfolio_grid[(pctile, brk)] = {
                "n_trades":      n,
                "profit_factor": gw / gl,
                "win_rate":      wr,
                "expectancy_r":  wr * RR - (1.0 - wr),
                "pnls":          pnls,
            }
        else:
            portfolio_grid[(pctile, brk)] = {"n_trades": 0, "profit_factor": 0.0, "win_rate": 0.0}

    best_grid = max(portfolio_grid.keys(),
                    key=lambda k: portfolio_grid[k].get("profit_factor", 0.0)
                                  if np.isfinite(portfolio_grid[k].get("profit_factor", 0.0)) else 0.0)
    print(f"  Best grid params: ATR<{best_grid[0]}%  break={best_grid[1]}b  "
          f"PF={portfolio_grid[best_grid]['profit_factor']:.3f}  "
          f"n={portfolio_grid[best_grid]['n_trades']}")

    # ── 4. Charts ────────────────────────────────────────────────────────────
    print(f"\n  Generating charts for [{bar}]...", flush=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    plot_equity_curves(sym_metrics, bar)
    plot_drawdown_curves(sym_metrics, bar)
    plot_symbol_comparison(sym_metrics, sym_validation, bar)
    plot_param_heatmap(portfolio_grid, bar)
    plot_monte_carlo(sym_metrics, bar)
    plot_bootstrap_ci(sym_metrics, sym_validation, bar)
    plot_compression_analysis(all_trades, bar)
    plot_portfolio_dashboard(sym_metrics, sym_mc, bar)

    # ── 5. Console report ────────────────────────────────────────────────────
    print_report(sym_metrics, sym_validation, sym_mc, tf_cfg)

    return {
        "sym_metrics":    sym_metrics,
        "sym_mc":         sym_mc,
        "sym_validation": sym_validation,
        "sym_oos_dates":  sym_oos_dates,
        "portfolio_grid": portfolio_grid,
        "all_trades":     all_trades,
        "label":          label,
    }


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print()
    print("╔" + "═" * 79 + "╗")
    print("║  QUANTLAB AI — RESEARCH #019" + " " * 50 + "║")
    print("║  Volatility Compression Breakout — Full Validation" + " " * 28 + "║")
    print("╚" + "═" * 79 + "╝")
    print()
    print("  Objective  : Deep-dive validation of the R004 VCB signal")
    print(f"  Base params: ATR(14) < {BASE_ATR_PCT}th percentile / {BASE_ATR_WIN}b  |  breakout > {BASE_BREAK}b high")
    print("  Grid       : ATR_PCTILE ∈ {20,25,30,35,40} × BREAK_BARS ∈ {5,10,15,20}")
    print("  Data       : Existing cache — no new downloads")
    print("  Symbols    : BTC ETH LINK XRP DOGE LTC AVAX BCH")
    print("  Split      : 70/30 chronological train/OOS")
    print()

    all_tf       = {}
    journal_rows = []

    for tf_cfg in TIMEFRAMES:
        bar = tf_cfg["bar"]
        try:
            result      = run_timeframe(tf_cfg)
            all_tf[bar] = result

            # Journal entries
            for sym, m in result["sym_metrics"].items():
                if m["n_trades"] > 0:
                    mc = result["sym_mc"].get(sym, {"prob_profit": 0.0})
                    v  = _verdict_from_metrics(m, mc)
                    journal_rows.append({
                        "research_id":    RESEARCH_ID,
                        "run_date":       datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                        "strategy_name":  f"VCB_{bar}",
                        "symbol":         sym.split("-")[0],
                        "n_trades":       m["n_trades"],
                        "profit_factor":  round(m["profit_factor"], 4)
                                          if np.isfinite(m["profit_factor"]) else 0.0,
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

    # ── Final assessment ──────────────────────────────────────────────────────
    if all_tf:
        print_final_questions(all_tf, {})

    # ── Journal ───────────────────────────────────────────────────────────────
    if journal_rows:
        print(f"\n  Writing journal ({len(journal_rows)} rows)...")
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
    print(f"  All outputs → {OUTPUT_FOLDER}/r019_*")
    print(f"  Research #019 complete.")
    print()


if __name__ == "__main__":
    main()
