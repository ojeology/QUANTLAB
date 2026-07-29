"""
=============================================================================
QUANTLAB AI – RESEARCH #012
Fresh Hypothesis — Three Mean-Reversion Concepts

Objective:
  Break completely from the Liquidity Sweep Reversal path (R005–R011).
  Test three independent mean-reversion hypotheses from scratch.

Strategies:
  1. BB.Bounce   — Bollinger Band lower-band pierce + reclaim in uptrend
  2. RSI.Rev     — RSI(14) crosses back above 30 from below in uptrend
  3. 3Bar.Rev    — Three consecutive bearish bars + bullish engulfing in uptrend

Rationale:
  All three target the same market mechanism: a short-term overextension
  (to the downside) that snaps back.  The entry condition differs — price
  structure (BB), momentum oscillator (RSI), and candle pattern (3Bar).
  Testing three at once maximises information per research cycle.

Parameters:
  BB: period=20, std_mult=2.0   — industry-standard Bollinger Bands.
  RSI: period=14                — Wilder's RSI, standard period.
  3Bar: engulf_mult=1.0         — engulfing close must exceed prior open.
  ALL parameters are industry defaults.  None are optimised.

Locked: engine, fees, spread, slippage, SL, TP, sizing, split, EMA200.
=============================================================================
"""

import os
import sys
import csv
import math
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantlab_ai import (
    CONFIG, get_data, add_indicators, run_backtest,
    compute_metrics, monte_carlo,
    append_journal, _journal_row, _verdict_from_metrics,
)

RESEARCH_ID   = "R012"
OUTPUT_FOLDER = CONFIG["OUTPUT_FOLDER"]
BG            = "#0F1117"

# ── Strategy parameters (industry defaults, not optimised) ───────────────────
BB_PERIOD   = 20
BB_STD_MULT = 2.0
RSI_PERIOD  = 14
RSI_OVERSOLD = 30.0


# =============================================================================
# NEW INDICATORS
# =============================================================================

def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI using EWM smoothing (standard implementation)."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def add_r012_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add R012-specific indicators on top of the R004 base set.
    All look-ahead free.
    """
    df = df.copy()

    # ── Bollinger Bands ───────────────────────────────────────────────────
    df["bb_mid"]   = df["close"].rolling(BB_PERIOD).mean()
    bb_std         = df["close"].rolling(BB_PERIOD).std(ddof=1)
    df["bb_upper"] = df["bb_mid"] + BB_STD_MULT * bb_std
    df["bb_lower"] = df["bb_mid"] - BB_STD_MULT * bb_std

    # Previous bar's BB lower (for signal detection without look-ahead)
    df["prev_bb_lower"] = df["bb_lower"].shift(1)
    df["prev_close"]    = df["close"].shift(1)
    df["prev_low"]      = df["low"].shift(1)

    # ── RSI ───────────────────────────────────────────────────────────────
    df["rsi"] = calc_rsi(df["close"], RSI_PERIOD)
    df["prev_rsi"] = df["rsi"].shift(1)

    # ── 3-bar candle context ─────────────────────────────────────────────
    df["open_1"] = df["open"].shift(1)
    df["open_2"] = df["open"].shift(2)
    df["open_3"] = df["open"].shift(3)
    df["close_1"] = df["close"].shift(1)
    df["close_2"] = df["close"].shift(2)
    df["close_3"] = df["close"].shift(3)

    return df


# =============================================================================
# STRATEGY SIGNAL FUNCTIONS
# =============================================================================

def strategy_bb_bounce(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 1 — Bollinger Band Bounce (BB.Bounce)

    Hypothesis: when price briefly pierces below the lower Bollinger Band
    (a statistical overextension) and then closes back above it, the
    momentum exhaustion sets up a mean-reversion long opportunity.

    Conditions (all on signal-bar close, no look-ahead):
      1. bb_pierce: close[i-1] < bb_lower[i-1]   (prior bar broke below band)
               OR  low[i]    < bb_lower[i]        (current bar dipped below)
      2. bb_reclaim: close[i] > bb_lower[i]       (current bar closes back inside)
      3. bullish: close[i] > open[i]              (rejection candle)
      4. trend:   close[i] > ema200[i]            (in uptrend)

    Stop:  low of signal bar.
    Entry: next bar open.
    """
    # Previous bar closed below lower band
    prev_broke = df["prev_close"] < df["prev_bb_lower"]
    # OR current bar's low dipped below but close is back inside
    curr_dip   = df["low"] < df["bb_lower"]

    pierce  = prev_broke | curr_dip
    reclaim = df["close"] > df["bb_lower"]
    bullish = df["close"] > df["open"]
    trend   = df["close"] > df["ema200"]
    valid   = df["bb_lower"].notna() & df["prev_bb_lower"].notna()

    return pierce & reclaim & bullish & trend & valid


def strategy_rsi_rev(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 2 — RSI Oversold Reversal (RSI.Rev)

    Hypothesis: when RSI(14) drops into oversold territory (< 30) and then
    crosses back above 30, momentum has bottomed and price is likely to
    recover in the direction of the prevailing trend.

    Conditions (all on signal-bar close, no look-ahead):
      1. was_oversold: prev_rsi[i] < RSI_OVERSOLD    (RSI was below 30)
      2. cross_up:     rsi[i] >= RSI_OVERSOLD         (RSI now at/above 30)
      3. trend:        close[i] > ema200[i]           (in uptrend)

    Stop:  low of signal bar.
    Entry: next bar open.
    Note:  No bullish-bar requirement — RSI cross is the primary signal.
    """
    was_oversold = df["prev_rsi"] < RSI_OVERSOLD
    cross_up     = df["rsi"] >= RSI_OVERSOLD
    trend        = df["close"] > df["ema200"]
    valid        = df["rsi"].notna() & df["prev_rsi"].notna()

    return was_oversold & cross_up & trend & valid


def strategy_3bar_rev(df: pd.DataFrame) -> pd.Series:
    """
    Strategy 3 — Three-Bar Reversal (3Bar.Rev)

    Hypothesis: three consecutive bearish candles represent short-term selling
    exhaustion; when the fourth candle is a bullish engulfing bar (closes
    above the open of the first bearish candle), it signals a reversal with
    conviction.

    Conditions (all on signal-bar close, no look-ahead):
      1. bar_3_bear: close[i-3] < open[i-3]   (3 bars ago was bearish)
      2. bar_2_bear: close[i-2] < open[i-2]   (2 bars ago was bearish)
      3. bar_1_bear: close[i-1] < open[i-1]   (prior bar was bearish)
      4. engulf:     close[i] > open[i]        (current bar closes bullish)
                 AND close[i] > open[i-3]      (engulfs the first bearish bar's open)
      5. trend:      close[i] > ema200[i]      (in uptrend)

    Stop:  low of signal bar.
    Entry: next bar open.
    """
    bar3_bear = df["close_3"] < df["open_3"]
    bar2_bear = df["close_2"] < df["open_2"]
    bar1_bear = df["close_1"] < df["open_1"]
    engulf    = (df["close"] > df["open"]) & (df["close"] > df["open_3"])
    trend     = df["close"] > df["ema200"]
    valid     = df["open_3"].notna() & df["close_3"].notna()

    return bar3_bear & bar2_bear & bar1_bear & engulf & trend & valid


# Strategy registry
STRATEGIES_R012 = {
    "BB.Bounce": strategy_bb_bounce,
    "RSI.Rev":   strategy_rsi_rev,
    "3Bar.Rev":  strategy_3bar_rev,
}


# =============================================================================
# DATA PIPELINE
# =============================================================================

def prepare_oos_df(symbol: str) -> pd.DataFrame:
    df    = get_data(symbol)
    n     = len(df)
    df    = add_indicators(df)
    df    = add_r012_indicators(df)
    split = int(n * CONFIG["TRAIN_RATIO"])
    return df.iloc[split:].reset_index(drop=True)


def run_all_strategies(symbol: str) -> dict:
    df_oos  = prepare_oos_df(symbol)
    results = {}

    for name, fn in STRATEGIES_R012.items():
        res     = run_backtest(df_oos, fn, name)
        metrics = compute_metrics(res["trades"], name)
        mc      = monte_carlo(metrics["pnls"], CONFIG["MC_ITERATIONS"])
        verdict = _verdict_from_metrics(metrics, mc)
        results[name] = {
            "trades": res["trades"],
            "m":      metrics,
            "mc":     mc,
            "verdict": verdict,
        }

    return {"symbol": symbol, "df_oos": df_oos, "strategies": results}


# =============================================================================
# VERDICT — ABSOLUTE CRITERIA (no A/B comparison; standalone hypothesis test)
# =============================================================================

PROMOTE_CRITERIA = {
    "min_n_trades":      10,    # at least 10 OOS trades per symbol
    "min_profit_factor": 1.30,  # PF >= 1.30
    "min_win_rate":      0.34,  # slightly above break-even for 2R
    "max_drawdown":     -0.25,  # max drawdown no worse than -25%
    "mc_prob_profit":    0.60,  # MC prob of ending profitable > 60%
}


def strategy_verdict(metrics: dict, mc: dict) -> str:
    """Absolute performance-based verdict (not A/B comparison)."""
    n  = metrics.get("n_trades", 0)
    pf = metrics.get("profit_factor", 0.0)
    wr = metrics.get("win_rate", 0.0)
    dd = metrics.get("max_drawdown", -1.0)
    pp = mc.get("prob_profit", 0.0)

    if n < PROMOTE_CRITERIA["min_n_trades"]:
        return "INSUFFICIENT"
    if (pf >= PROMOTE_CRITERIA["min_profit_factor"]
            and wr >= PROMOTE_CRITERIA["min_win_rate"]
            and dd >= PROMOTE_CRITERIA["max_drawdown"]
            and pp >= PROMOTE_CRITERIA["mc_prob_profit"]):
        return "PROMOTE"
    return "REJECT"


def aggregate_strategy(symbol_results: list, strat_name: str) -> dict:
    all_trades = []
    for r in symbol_results:
        all_trades.extend(r["strategies"][strat_name]["trades"])
    if not all_trades:
        from quantlab_ai import _empty_metrics
        return _empty_metrics(strat_name)
    return compute_metrics(all_trades, strat_name)


def compute_leaderboard(symbol_results: list) -> list:
    """
    Compute aggregate metrics for each strategy and rank by profit factor.
    Returns sorted list of dicts.
    """
    rows = []
    for name in STRATEGIES_R012:
        agg = aggregate_strategy(symbol_results, name)
        mc  = monte_carlo(agg["pnls"], CONFIG["MC_ITERATIONS"])
        vd  = strategy_verdict(agg, mc)

        # Count how many individual symbol/strategy combos PROMOTE
        n_promote = sum(
            1 for r in symbol_results
            if r["strategies"][name]["verdict"] == "PROMOTE"
        )

        rows.append({
            "name":           name,
            "agg":            agg,
            "mc":             mc,
            "verdict":        vd,
            "n_promote":      n_promote,
            "n_symbols":      len(symbol_results),
            "profit_factor":  agg["profit_factor"],
            "win_rate":       agg["win_rate"],
            "expectancy_r":   agg["expectancy_r"],
            "net_profit":     agg["net_profit"],
            "max_drawdown":   agg["max_drawdown"],
            "sharpe":         agg["sharpe"],
            "n_trades":       agg["n_trades"],
            "mc_prob_profit": mc["prob_profit"],
        })

    rows.sort(key=lambda x: x["profit_factor"], reverse=True)
    return rows


# =============================================================================
# VISUALISATIONS
# =============================================================================

C_STRAT = {
    "BB.Bounce": "#4A90D9",
    "RSI.Rev":   "#FFB347",
    "3Bar.Rev":  "#00C49A",
}
C_PROMOTE = "#00C49A"
C_REJECT  = "#FF4560"
C_INSUF   = "#888888"


def _ax_style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=8)
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.grid(True, alpha=0.15, color="#444")


def _save(fig, fname: str) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


# ── Chart 1: Leaderboard overview ────────────────────────────────────────────

def plot_leaderboard(leaderboard: list) -> str:
    metrics = [
        ("profit_factor",   "Profit Factor",   False, 1.0),
        ("win_rate",        "Win Rate",        True,  1/3),
        ("expectancy_r",    "Expectancy (R)",  False, 0.0),
        ("net_profit",      "Net Profit ($)",  False, 0.0),
        ("max_drawdown",    "Max Drawdown",    False, -0.25),
        ("sharpe",          "Sharpe Ratio",    False, 0.0),
        ("n_trades",        "Trade Count",     False, None),
        ("mc_prob_profit",  "MC Prob Profit",  True,  0.60),
    ]
    n_met = len(metrics)
    n_str = len(leaderboard)

    fig, axes = plt.subplots(n_met, 1, figsize=(12, 3.0 * n_met))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "R012 — Three Fresh Mean-Reversion Strategies — Aggregate Leaderboard\n"
        "Tested on BTC, ETH, SOL (1H) OOS only  |  Engine, costs, sizing LOCKED",
        fontsize=11, fontweight="bold", color="white",
    )

    x      = np.arange(n_str)
    names  = [r["name"] for r in leaderboard]
    colors = [C_STRAT.get(n, "#aaaaaa") for n in names]

    for idx, (mkey, mlabel, pct, ref) in enumerate(metrics):
        ax = axes[idx]
        _ax_style(ax)

        vals = [r[mkey] for r in leaderboard]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.55)

        if ref is not None:
            ax.axhline(ref, color="#FF4560", lw=0.9, ls="--", alpha=0.7,
                       label=f"ref={ref}")

        def _fmt(v):
            if pct:
                return f"{v:.1%}"
            if mkey == "net_profit":
                return f"${v:+,.0f}"
            if mkey == "n_trades":
                return str(int(v))
            return f"{v:+.3f}" if v < 0 else f"{v:.3f}"

        for bar, val in zip(bars, vals):
            yp = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    yp + (abs(yp) * 0.04 + 0.001),
                    _fmt(val), ha="center", va="bottom",
                    color="white", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(names, color="white", fontsize=10)
        ax.set_ylabel(mlabel, color="white", fontsize=8)
        if ref is not None and idx == 0:
            ax.legend(fontsize=7, facecolor="#1A1D24", edgecolor="#444",
                      labelcolor="white", loc="upper right")

    # Add verdict labels below strategy names
    for ax in axes:
        ax.set_xticks(x)
        labels = []
        for r in leaderboard:
            v = r["verdict"]
            labels.append(f"{r['name']}\n[{v}]")
        ax.set_xticklabels(labels, color="white", fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r012_leaderboard.png")


# ── Chart 2: Equity curves — per strategy per symbol ─────────────────────────

def plot_equity_curves(symbol_results: list, leaderboard: list) -> str:
    symbols   = [r["symbol"].replace("-USDT-SWAP", "") for r in symbol_results]
    strat_names = [r["name"] for r in leaderboard]
    n_sym     = len(symbols)
    n_str     = len(strat_names)

    fig, axes = plt.subplots(n_sym + 1, 1, figsize=(14, 5.0 * (n_sym + 1)))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Equity Curves — R012 — Three Mean-Reversion Strategies\n"
        "Per symbol + aggregate  |  1% risk per trade, $10k starting capital",
        fontsize=9, fontweight="bold", color="white",
    )

    start = CONFIG["STARTING_CAPITAL"]

    def _plot_sym(ax, r_sym, sym_label):
        _ax_style(ax)
        for name in strat_names:
            eq  = r_sym["strategies"][name]["m"]["equity"]
            n_t = r_sym["strategies"][name]["m"]["n_trades"]
            pf  = r_sym["strategies"][name]["m"]["profit_factor"]
            ax.plot(np.arange(len(eq)), eq,
                    color=C_STRAT.get(name, "#aaa"), lw=1.8,
                    label=f"{name}  n={n_t}  PF={pf:.3f}", zorder=3)
        ax.axhline(start, color="#555", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(sym_label, fontsize=9)
        ax.set_xlabel("Trade #", color="white", fontsize=8)
        ax.set_ylabel("Capital ($)", color="white", fontsize=8)
        ax.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444",
                  labelcolor="white", loc="upper left")

    for i, r in enumerate(symbol_results):
        _plot_sym(axes[i], r, r["symbol"].replace("-USDT-SWAP", ""))

    # Aggregate equity curves (concatenated, ordered by trade entry)
    ax_agg = axes[n_sym]
    _ax_style(ax_agg)
    for name in strat_names:
        all_trades = []
        for r in symbol_results:
            all_trades.extend(r["strategies"][name]["trades"])
        if not all_trades:
            continue
        agg_m = compute_metrics(all_trades, name)
        eq    = agg_m["equity"]
        pf    = agg_m["profit_factor"]
        n_t   = agg_m["n_trades"]
        ax_agg.plot(np.arange(len(eq)), eq,
                    color=C_STRAT.get(name, "#aaa"), lw=1.8,
                    label=f"{name}  n={n_t}  PF={pf:.3f}", zorder=3)
    ax_agg.axhline(start, color="#555", lw=0.8, ls="--", alpha=0.6)
    ax_agg.set_title("AGGREGATE (all symbols)", fontsize=9)
    ax_agg.set_xlabel("Trade #", color="white", fontsize=8)
    ax_agg.set_ylabel("Capital ($)", color="white", fontsize=8)
    ax_agg.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444",
                  labelcolor="white", loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r012_equity_curves.png")


# ── Chart 3: Monte Carlo distributions ──────────────────────────────────────

def plot_monte_carlo(leaderboard: list) -> str:
    n_str = len(leaderboard)
    fig, axes = plt.subplots(1, n_str, figsize=(6 * n_str, 6))
    if n_str == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Monte Carlo Final Equity — R012  |  N={CONFIG['MC_ITERATIONS']:,} permutations",
        fontsize=10, fontweight="bold", color="white",
    )

    start = CONFIG["STARTING_CAPITAL"]

    for i, row in enumerate(leaderboard):
        ax  = axes[i]
        _ax_style(ax)
        fe  = row["mc"]["final_equities"]

        fe_min, fe_max = float(np.min(fe)), float(np.max(fe))
        fe_range = fe_max - fe_min

        if fe_range < 1e-6 or len(np.unique(fe)) < 3:
            # Not enough variance — just show a text note
            ax.text(0.5, 0.5,
                    f"Insufficient trades\nfor MC histogram\n(n={row['n_trades']})",
                    transform=ax.transAxes, ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
        else:
            bns = min(max(5, len(np.unique(fe))), 50)
            ax.hist(fe, bins=bns, color=C_STRAT.get(row["name"], "#4A90D9"),
                    alpha=0.75, density=True)
            ax.axvline(start, color="#FF4560", lw=1.5, ls="--", label="Start")
            ax.axvline(row["mc"]["median"], color="#FFD700", lw=1.2, ls=":",
                       label=f"Median={row['mc']['median']:,.0f}")
            ax.legend(fontsize=7.5, facecolor="#1A1D24", edgecolor="#444",
                      labelcolor="white")

        ax.set_title(
            f"{row['name']}  [{row['verdict']}]\n"
            f"PP={row['mc']['prob_profit']:.1%}  PF={row['profit_factor']:.3f}",
            fontsize=9, color="white",
        )
        ax.set_xlabel("Final Equity ($)", color="white", fontsize=8)
        ax.set_ylabel("Density", color="white", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "r012_monte_carlo.png")


# ── Chart 4: Per-symbol metric grid ──────────────────────────────────────────

def plot_symbol_grid(symbol_results: list) -> str:
    symbols    = [r["symbol"].replace("-USDT-SWAP", "") for r in symbol_results]
    strat_names = list(STRATEGIES_R012.keys())
    metrics    = [
        ("profit_factor", "Profit Factor", False),
        ("win_rate",      "Win Rate",      True),
        ("expectancy_r",  "Expectancy R",  False),
        ("max_drawdown",  "Max Drawdown",  False),
    ]
    n_met = len(metrics)
    n_sym = len(symbols)

    fig, axes = plt.subplots(n_met, n_sym, figsize=(6 * n_sym, 4.5 * n_met))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Per-Symbol Performance — R012 — Three Mean-Reversion Strategies",
        fontsize=10, fontweight="bold", color="white",
    )

    x      = np.arange(len(strat_names))
    colors = [C_STRAT.get(n, "#aaa") for n in strat_names]

    for col, r in enumerate(symbol_results):
        for row, (mkey, mlabel, pct) in enumerate(metrics):
            ax = axes[row][col]
            _ax_style(ax)

            vals = [r["strategies"][n]["m"].get(mkey, 0.0) for n in strat_names]
            bars = ax.bar(x, vals, color=colors, alpha=0.85, width=0.55)

            if mkey == "profit_factor":
                ax.axhline(1.0, color="#FF4560", lw=0.8, ls="--", alpha=0.7)
            elif mkey == "win_rate":
                ax.axhline(1/3, color="#FF4560", lw=0.8, ls="--", alpha=0.7)
            elif mkey == "expectancy_r":
                ax.axhline(0.0, color="#FF4560", lw=0.8, ls="--", alpha=0.7)
            elif mkey == "max_drawdown":
                ax.axhline(-0.25, color="#FF4560", lw=0.8, ls="--", alpha=0.7)

            for bar, val in zip(bars, vals):
                yp = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2,
                        yp + (abs(yp) * 0.04 + 0.001),
                        f"{val:.1%}" if pct else f"{val:.3f}",
                        ha="center", va="bottom",
                        color="white", fontsize=7.5, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(strat_names, color="white", fontsize=8)
            ax.set_ylabel(mlabel, color="white", fontsize=8)
            if row == 0:
                sym = r["symbol"].replace("-USDT-SWAP", "")
                ax.set_title(sym, fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r012_symbol_grid.png")


# ── Chart 5: Trade distribution (win/loss R-multiples) ──────────────────────

def plot_r_distribution(symbol_results: list, leaderboard: list) -> str:
    n_str = len(leaderboard)
    fig, axes = plt.subplots(1, n_str, figsize=(6 * n_str, 6))
    if n_str == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "R-Multiple Distribution — R012  |  All symbols combined\n"
        "TP hit = +2R  |  SL hit = −1R  |  Distribution reveals edge consistency",
        fontsize=9.5, fontweight="bold", color="white",
    )

    for i, row in enumerate(leaderboard):
        ax   = axes[i]
        _ax_style(ax)
        name = row["name"]

        all_r = []
        for r_sym in symbol_results:
            for t in r_sym["strategies"][name]["trades"]:
                all_r.append(t["r_multiple"])

        if not all_r:
            ax.set_visible(False)
            continue

        r_arr = np.array(all_r)
        wins  = r_arr[r_arr > 0]
        loss  = r_arr[r_arr <= 0]

        bns = max(10, min(30, len(all_r) // 3))
        rng = (float(r_arr.min()) - 0.1, float(r_arr.max()) + 0.1)

        ax.hist(loss, bins=bns, range=rng, color="#FF4560", alpha=0.7,
                label=f"Loss (n={len(loss)})")
        ax.hist(wins, bins=bns, range=rng, color="#00C49A", alpha=0.7,
                label=f"Win  (n={len(wins)})")
        ax.axvline(0, color="white", lw=1.0, ls="-", alpha=0.4)
        ax.axvline(np.mean(all_r), color="#FFD700", lw=1.5, ls="--",
                   label=f"Mean R={np.mean(all_r):+.3f}")

        wr = len(wins) / len(all_r) if all_r else 0.0
        ax.set_title(
            f"{name}  [{row['verdict']}]\n"
            f"n={len(all_r)}  WR={wr:.1%}  Avg R={np.mean(all_r):+.3f}",
            fontsize=9, color="white",
        )
        ax.set_xlabel("R-Multiple", color="white", fontsize=8)
        ax.set_ylabel("Count", color="white", fontsize=8)
        ax.legend(fontsize=7.5, facecolor="#1A1D24", edgecolor="#444",
                  labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "r012_r_distribution.png")


# =============================================================================
# REPORT
# =============================================================================

def print_r012_report(symbol_results: list, leaderboard: list) -> None:
    S  = "=" * 108
    S2 = "─" * 108
    BL = "  "

    print(f"\n{S}")
    print(f"{BL}QUANTLAB AI — RESEARCH #012")
    print(f"{BL}Fresh Hypothesis — Three Mean-Reversion Strategies")
    print(f"{BL}{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(S)

    print(f"""
{BL}  Strategies tested (all on OOS slice, engine locked):
{BL}  1. BB.Bounce  — Lower Bollinger Band pierce + reclaim + bullish close + EMA200
{BL}               BB: period={BB_PERIOD}, std={BB_STD_MULT}  (industry standard — not optimised)
{BL}  2. RSI.Rev    — RSI({RSI_PERIOD}) crosses above {RSI_OVERSOLD:.0f} from below + EMA200 trend
{BL}               RSI period={RSI_PERIOD}  (industry standard — not optimised)
{BL}  3. 3Bar.Rev   — 3 consecutive bearish bars + bullish engulfing close + EMA200
{BL}               Engulf: close > first bearish bar's open  (no size threshold)
""")

    print(f"{BL}PROMOTE criteria (absolute, not A/B comparison):")
    print(f"{BL}  PF ≥ {PROMOTE_CRITERIA['min_profit_factor']}  "
          f"|  WR ≥ {PROMOTE_CRITERIA['min_win_rate']:.0%}  "
          f"|  MDD ≥ {PROMOTE_CRITERIA['max_drawdown']:.0%}  "
          f"|  MC prob profit ≥ {PROMOTE_CRITERIA['mc_prob_profit']:.0%}  "
          f"|  n trades ≥ {PROMOTE_CRITERIA['min_n_trades']} per symbol")
    print()

    # Per-symbol results table
    strat_names = list(STRATEGIES_R012.keys())
    for r in symbol_results:
        sym = r["symbol"]
        print(f"{BL}{'─'*80}")
        print(f"{BL}{sym}")
        print(f"{BL}{'─'*80}")
        print(f"{BL}  {'Metric':<22} " +
              "  ".join(f"{n:>16}" for n in strat_names))
        print(f"{BL}  {'─'*22} " +
              "  ".join(f"{'─'*16}" for _ in strat_names))

        metric_rows = [
            ("n_trades",        "Trades",       lambda v: str(int(v))),
            ("win_rate",        "Win Rate",     lambda v: f"{v:.1%}"),
            ("profit_factor",   "PF",           lambda v: f"{v:.3f}"),
            ("expectancy_r",    "Expectancy R", lambda v: f"{v:+.3f}"),
            ("net_profit",      "Net P&L ($)",  lambda v: f"${v:+,.0f}"),
            ("max_drawdown",    "Max DD",       lambda v: f"{v:.1%}"),
            ("sharpe",          "Sharpe",       lambda v: f"{v:.3f}"),
        ]
        for key, lbl, fmt in metric_rows:
            vals = [r["strategies"][n]["m"].get(key, 0.0) for n in strat_names]
            print(f"{BL}  {lbl:<22} " +
                  "  ".join(f"{fmt(v):>16}" for v in vals))

        # MC prob
        mc_vals = [r["strategies"][n]["mc"]["prob_profit"] for n in strat_names]
        print(f"{BL}  {'MC Prob Profit':<22} " +
              "  ".join(f"{v:>16.1%}" for v in mc_vals))

        # Verdicts
        vds = [r["strategies"][n]["verdict"] for n in strat_names]
        print(f"{BL}  {'Verdict':<22} " +
              "  ".join(f"{v:>16}" for v in vds))

    # Aggregate leaderboard
    print(f"\n{S}")
    print(f"{BL}AGGREGATE LEADERBOARD  (ranked by Profit Factor, all symbols combined)")
    print(f"{BL}{S2}")
    print(f"\n{BL}  {'Rank':<6} {'Strategy':<14} {'Trades':>8} {'Win Rate':>10} "
          f"{'PF':>8} {'Exp R':>8} {'Net P&L':>12} {'Max DD':>10} "
          f"{'MC PP':>8} {'Verdict':<14}")
    print(f"{BL}  {'─'*6} {'─'*14} {'─'*8} {'─'*10} "
          f"{'─'*8} {'─'*8} {'─'*12} {'─'*10} {'─'*8} {'─'*14}")

    for rank, row in enumerate(leaderboard, 1):
        n_sym_promote = row["n_promote"]
        v_color = ("★ " if row["verdict"] == "PROMOTE" else "  ")
        print(f"{BL}  {rank:<6} {v_color+row['name']:<14} "
              f"{row['n_trades']:>8} {row['win_rate']:>10.1%} "
              f"{row['profit_factor']:>8.3f} {row['expectancy_r']:>8.3f} "
              f"${row['net_profit']:>10,.0f} {row['max_drawdown']:>10.1%} "
              f"{row['mc_prob_profit']:>8.1%} {row['verdict']:<14}")

    # Best strategy detail
    best = leaderboard[0]
    print(f"\n{S}")
    print(f"{BL}BEST STRATEGY: {best['name']}  |  Verdict: {best['verdict']}")
    print(f"{BL}{S2}")
    print(f"\n{BL}  Aggregate PF = {best['profit_factor']:.3f}  "
          f"|  WR = {best['win_rate']:.1%}  "
          f"|  ExpR = {best['expectancy_r']:+.3f}")
    print(f"{BL}  Max Drawdown = {best['max_drawdown']:.1%}  "
          f"|  Sharpe = {best['agg']['sharpe']:.3f}  "
          f"|  MC Prob Profit = {best['mc_prob_profit']:.1%}")
    print(f"{BL}  Per-symbol PROMOTE count: {best['n_promote']}/{best['n_symbols']}")

    if best["verdict"] == "PROMOTE":
        print(f"\n{BL}  ╔{'═'*66}╗")
        print(f"{BL}  ║{'PROMOTE — Warrants further development in R013':^66}║")
        print(f"{BL}  ╚{'═'*66}╝")
        print(f"\n{BL}  {best['name']} meets the absolute performance threshold.")
        print(f"{BL}  R013 should:")
        print(f"{BL}    1. Deep-dive into the {best['name']} trade context (like R005 did for LSR)")
        print(f"{BL}    2. Identify which market conditions drive wins vs losses")
        print(f"{BL}    3. Test any emerging filter hypotheses in R014")
    else:
        # Find if any PROMOTE
        promoted = [r for r in leaderboard if r["verdict"] == "PROMOTE"]
        if promoted:
            p = promoted[0]
            print(f"\n{BL}  Note: {p['name']} shows the strongest signal "
                  f"(PF={p['profit_factor']:.3f}).")
        else:
            print(f"\n{BL}  All three strategies REJECTED on aggregate.")
            print(f"\n{BL}  Interpretation:")
            print(f"{BL}    The mean-reversion family (BB, RSI, 3Bar) does not")
            print(f"{BL}    appear to have a reliable edge on crypto perpetuals")
            print(f"{BL}    at the 1H timeframe with current parameters.")
            print(f"\n{BL}  Suggested next directions for R013:")
            print(f"{BL}    A) Increase timeframe (4H or 1D) — fewer signals, cleaner structure")
            print(f"{BL}    B) Test SHORT-SIDE mean reversion (sell overbought)")
            print(f"{BL}    C) Test cross-asset momentum (lead/lag between BTC and alts)")
            print(f"{BL}    D) Test funding-rate extremes as a directional signal")

    print(f"\n{BL}  IMPORTANT: Engine, fees, spread, slippage, SL, TP, sizing — UNCHANGED.")
    print(S)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              QUANTLAB AI — RESEARCH #012                                      ║
║   Fresh Hypothesis — Three Mean-Reversion Strategies                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  Strategies:
    1. BB.Bounce  — Bollinger Band lower pierce + reclaim (period=20, 2σ)
    2. RSI.Rev    — RSI(14) crosses above 30 from below
    3. 3Bar.Rev   — 3 bearish bars + bullish engulfing close

  All parameters are industry defaults.  No optimisation.
  Engine, fees, spread, slippage, split: LOCKED.
""")

    random.seed(42)
    np.random.seed(42)

    # ── 1. Run all strategies ─────────────────────────────────────────────
    print("=" * 70)
    print("  STEP 1: Running all strategies on all symbols")
    print("=" * 70)

    symbol_results = []
    for sym in CONFIG["SYMBOLS"]:
        print(f"\n  {sym}")
        r = run_all_strategies(sym)
        symbol_results.append(r)

        for name in STRATEGIES_R012:
            m  = r["strategies"][name]["m"]
            vd = r["strategies"][name]["verdict"]
            print(f"    {name:<12} n={m['n_trades']:>3}  WR={m['win_rate']:.1%}  "
                  f"PF={m['profit_factor']:.3f}  ExpR={m['expectancy_r']:+.3f}  [{vd}]")

    # ── 2. Leaderboard ────────────────────────────────────────────────────
    print("\n  STEP 2: Building aggregate leaderboard")
    leaderboard = compute_leaderboard(symbol_results)

    for rank, row in enumerate(leaderboard, 1):
        print(f"    #{rank}  {row['name']:<14}  PF={row['profit_factor']:.3f}  "
              f"WR={row['win_rate']:.1%}  ExpR={row['expectancy_r']:+.3f}  "
              f"[{row['verdict']}]")

    # ── 3. Charts ─────────────────────────────────────────────────────────
    print("\n  STEP 3: Generating charts")
    charts = []

    p = plot_leaderboard(leaderboard)
    charts.append(p);  print(f"  → {p}")

    p = plot_equity_curves(symbol_results, leaderboard)
    charts.append(p);  print(f"  → {p}")

    p = plot_monte_carlo(leaderboard)
    charts.append(p);  print(f"  → {p}")

    p = plot_symbol_grid(symbol_results)
    charts.append(p);  print(f"  → {p}")

    p = plot_r_distribution(symbol_results, leaderboard)
    charts.append(p);  print(f"  → {p}")

    # ── 4. Full report ────────────────────────────────────────────────────
    print_r012_report(symbol_results, leaderboard)

    # ── 5. Journal ────────────────────────────────────────────────────────
    print("  STEP 4: Writing journal")
    jnl_rows = []
    for r in symbol_results:
        for name in STRATEGIES_R012:
            m  = r["strategies"][name]["m"]
            mc = r["strategies"][name]["mc"]
            vd = r["strategies"][name]["verdict"]
            row = _journal_row(name, r["symbol"], m, mc, vd)
            row["research_id"] = RESEARCH_ID
            jnl_rows.append(row)
    if jnl_rows:
        append_journal(jnl_rows)
        print(f"  Journal updated → {CONFIG['JOURNAL_FILE']}")

    print(f"\n  All outputs → {OUTPUT_FOLDER}/")
    print("  Research #012 complete.\n")


if __name__ == "__main__":
    main()
