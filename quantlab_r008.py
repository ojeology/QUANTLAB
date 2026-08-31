"""
=============================================================================
QUANTLAB AI – RESEARCH #008
Confirmation Study — Hypothesis Validation

Objective:
  Test whether the strongest interaction discovered in R007
  (High 10-Bar Realised Volatility AND price within 1.64% of 20-Bar High)
  reflects a genuine structural edge or is a random artefact of the small
  R007 sample.

Method:
  Strategy A — Liquidity Sweep Reversal, unchanged.
  Strategy B — Same strategy + TWO additional pre-entry conditions
               derived directly from the R007 quadrant medians.
               No optimisation.  No threshold search.

Locked: engine, fees, spread, slippage, SL, TP, sizing, split.
=============================================================================

SEMANTIC CLARIFICATION — READ BEFORE MODIFYING
───────────────────────────────────────────────
dist_from_hh_pct = (close − 20-bar high) / close × 100
This is ALWAYS NEGATIVE (price is below its 20-bar high).

R007 quadrant analysis found:
  High dist_from_hh_pct (>= median −1.6435) → price CLOSE to the 20-bar high
  Low  dist_from_hh_pct (<  median −1.6435) → price FAR from the 20-bar high

The R007 "High+High" quadrant (PF 2.83) is:
  High ret_vol_10         — elevated short-term volatility
  High dist_from_hh_pct  — price near the top of the 20-bar range
                            (i.e. sweeping a low while still close to the range top)

This is the ONLY condition tested.  No alternative is explored.
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
    compute_metrics, monte_carlo, strategy_lsr,
    append_journal, _journal_row, _verdict_from_metrics,
)
from quantlab_r005 import (
    get_funding_rates, add_r005_indicators,
    attach_funding_rate, enrich_trades_with_context,
)


# =============================================================================
# CONSTANTS
# =============================================================================

RESEARCH_ID = "R008"
OUTPUT_FOLDER = CONFIG["OUTPUT_FOLDER"]
BG = "#0F1117"

# ── R007 Thresholds (LOCKED — median of combined 86 OOS trades) ──────────────
# These are the ONLY values used.  No search.  No optimisation.
RV_THRESHOLD  = 0.356350     # ret_vol_10  ≥ this → "high realised vol"
HH_THRESHOLD  = -1.643503    # dist_from_hh_pct ≥ this → "price within 1.64% of 20-bar high"

LABEL_A = "Liq.Sweep (A)"
LABEL_B = "Liq.Sweep+Filter (B)"

METRIC_LABELS = {
    "n_trades":       "Trades",
    "win_rate":       "Win Rate",
    "profit_factor":  "Profit Factor",
    "expectancy_r":   "Expectancy (R)",
    "net_profit":     "Net Profit ($)",
    "max_drawdown":   "Max Drawdown",
    "sharpe":         "Sharpe Ratio",
    "avg_hold_minutes": "Avg Hold (min)",
    "mc_prob_profit": "MC Prob Profit",
}


# =============================================================================
# STRATEGY B — LSR + R007 Filter
# =============================================================================

def strategy_lsr_b(df: pd.DataFrame) -> pd.Series:
    """
    Strategy B: Liquidity Sweep Reversal + R007 interaction filter.

    Entry conditions (all on bar close, no look-ahead):
      1. sweep:    low < lsr_prior_low           (wick below prior 5-bar low)
      2. reclaim:  close > lsr_prior_low         (close back above swept level)
      3. bullish:  close > open                  (rejection candle)
      4. trend:    close > ema200                (uptrend)
      ── R007 additions ──────────────────────────────────────────────────
      5. high_vol: ret_vol_10 ≥ 0.356350         (10-bar realised vol above median)
      6. near_hh:  dist_from_hh_pct ≥ −1.6435   (price within 1.64% of 20-bar high)

    Conditions 5 & 6 are the ONLY change from Strategy A.
    Stop, entry, TP, sizing, fees: unchanged.
    """
    # Base LSR conditions (identical to strategy_lsr)
    sweep   = df["low"]   < df["lsr_prior_low"]
    reclaim = df["close"] > df["lsr_prior_low"]
    bullish = df["close"] > df["open"]
    trend   = df["close"] > df["ema200"]

    # R007 filter (pre-entry values at signal-bar close — no look-ahead)
    high_vol = df["ret_vol_10"]      >= RV_THRESHOLD
    near_hh  = df["dist_from_hh_pct"] >= HH_THRESHOLD

    return sweep & reclaim & bullish & trend & high_vol & near_hh


# =============================================================================
# DATA PIPELINE
# =============================================================================

def prepare_oos_df(symbol: str) -> pd.DataFrame:
    """
    Load price data, compute ALL indicators (base + R005 context),
    return the OOS slice.  Identical train/test split to all prior research.
    """
    df   = get_data(symbol)
    n    = len(df)
    df   = add_indicators(df)     # EMA200, lsr_prior_low, ATR, etc.
    df   = add_r005_indicators(df)  # ret_vol_10, dist_from_hh_pct, etc.
    split = int(n * CONFIG["TRAIN_RATIO"])
    return df.iloc[split:].reset_index(drop=True)


def run_both_strategies(symbol: str) -> dict:
    """
    Run Strategy A and Strategy B on the same OOS slice.
    Return per-symbol results dict.
    """
    df_oos = prepare_oos_df(symbol)

    # Strategy A
    res_a    = run_backtest(df_oos, strategy_lsr,   LABEL_A)
    metrics_a = compute_metrics(res_a["trades"], LABEL_A)
    mc_a      = monte_carlo(metrics_a["pnls"], CONFIG["MC_ITERATIONS"])
    verdict_a = _verdict_from_metrics(metrics_a, mc_a)

    # Strategy B
    res_b    = run_backtest(df_oos, strategy_lsr_b, LABEL_B)
    metrics_b = compute_metrics(res_b["trades"], LABEL_B)
    mc_b      = monte_carlo(metrics_b["pnls"], CONFIG["MC_ITERATIONS"])
    verdict_b = _verdict_from_metrics(metrics_b, mc_b)

    return {
        "symbol":     symbol,
        "df_oos":     df_oos,
        "A": {"trades": res_a["trades"], "m": metrics_a,
              "mc": mc_a, "verdict": verdict_a},
        "B": {"trades": res_b["trades"], "m": metrics_b,
              "mc": mc_b, "verdict": verdict_b},
    }


def aggregate_metrics(symbol_results: list, label: str) -> dict:
    """
    Pool all trades across symbols, re-compute combined metrics.
    """
    all_trades = []
    for r in symbol_results:
        all_trades.extend(r[label[0]]["trades"])   # label[0] = 'A' or 'B'

    if not all_trades:
        return {"label": label, "n_trades": 0, "net_profit": 0.0,
                "profit_factor": 0.0, "win_rate": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "avg_trade": 0.0,
                "avg_r": 0.0, "expectancy_r": 0.0,
                "largest_win": 0.0, "largest_loss": 0.0,
                "max_drawdown": 0.0, "sharpe": 0.0,
                "avg_hold_minutes": 0.0, "total_funding_windows": 0,
                "equity": np.array([CONFIG["STARTING_CAPITAL"]]),
                "drawdown": np.array([0.0]),
                "pnls": np.array([]), "r_multiples": np.array([]),
                "trades_df": pd.DataFrame()}

    return compute_metrics(all_trades, label)


# =============================================================================
# FILTER ANALYSIS — how many signals pass / are rejected per symbol
# =============================================================================

def filter_analysis(df_oos: pd.DataFrame) -> dict:
    """
    Break down signal counts: how many base LSR signals fire,
    how many pass the filter, how many are blocked by each condition.
    """
    base_sig  = strategy_lsr(df_oos)
    n_base    = int(base_sig.sum())

    high_vol  = df_oos["ret_vol_10"]      >= RV_THRESHOLD
    near_hh   = df_oos["dist_from_hh_pct"] >= HH_THRESHOLD
    both      = high_vol & near_hh

    pass_both = int((base_sig & both).sum())
    pass_vol_only = int((base_sig & high_vol & ~near_hh).sum())
    pass_hh_only  = int((base_sig & ~high_vol & near_hh).sum())
    pass_neither  = int((base_sig & ~high_vol & ~near_hh).sum())

    return {
        "n_base":         n_base,
        "n_pass":         pass_both,
        "n_vol_only":     pass_vol_only,
        "n_hh_only":      pass_hh_only,
        "n_neither":      pass_neither,
        "pass_rate":      pass_both / n_base if n_base > 0 else 0.0,
    }


# =============================================================================
# VISUALISATIONS
# =============================================================================

def _ax_style(ax, bg=None):
    ax.set_facecolor(bg or BG)
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


# ── Chart 1: Per-symbol metric comparison grid ──────────────────────────────

def plot_metric_grid(symbol_results: list,
                     agg_a: dict, agg_b: dict) -> str:
    symbols = [r["symbol"].replace("-USDT-SWAP", "") for r in symbol_results]
    labels  = symbols + ["AGGREGATE"]
    n_sym   = len(symbols)

    metrics = [
        ("profit_factor",  "Profit Factor",   False),
        ("win_rate",       "Win Rate",        True),
        ("expectancy_r",   "Expectancy (R)",  False),
        ("net_profit",     "Net Profit ($)",  False),
        ("max_drawdown",   "Max Drawdown",    False),
        ("sharpe",         "Sharpe Ratio",    False),
        ("n_trades",       "Trade Count",     False),
        ("mc_prob_profit", "MC Prob Profit",  True),
    ]

    n_met = len(metrics)
    fig, axes = plt.subplots(n_met, 1, figsize=(14, 3.0 * n_met))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Strategy A vs Strategy B — All Metrics  |  R008  |  Liq.Sweep Confirmation Study\n"
        f"Filter: ret_vol_10 ≥ {RV_THRESHOLD:.4f}  AND  dist_from_hh_pct ≥ {HH_THRESHOLD:.4f}\n"
        "Blue = Strategy A (unfiltered)  |  Orange = Strategy B (filtered)",
        fontsize=10, fontweight="bold", color="white",
    )

    x     = np.arange(len(labels))
    width = 0.35
    C_A   = "#4A90D9"
    C_B   = "#FFB347"

    def _get(res, key):
        """Pull from metrics dict; mc_prob_profit from mc dict."""
        if key == "mc_prob_profit":
            return res["mc"]["prob_profit"]
        v = res["m"].get(key, 0.0)
        if not (np.isfinite(v) if isinstance(v, float) else True):
            return 0.0
        return v

    def _agg(m, mc, key):
        if key == "mc_prob_profit":
            return mc["prob_profit"]
        v = m.get(key, 0.0)
        return v if np.isfinite(v) else 0.0

    for idx, (mkey, mlabel, pct) in enumerate(metrics):
        ax = axes[idx]
        _ax_style(ax)

        vals_a = [_get(r["A"], mkey) for r in symbol_results] + \
                 [_agg(agg_a, agg_b, mkey)]   # last = aggregate A
        vals_b = [_get(r["B"], mkey) for r in symbol_results] + \
                 [_agg(agg_b, agg_b, mkey)]   # last = aggregate B

        # Separate aggregate MC — always re-run from aggregate
        if mkey == "mc_prob_profit":
            vals_a[-1] = _agg(agg_a, monte_carlo(agg_a["pnls"],
                                                   CONFIG["MC_ITERATIONS"]), mkey)
            vals_b[-1] = _agg(agg_b, monte_carlo(agg_b["pnls"],
                                                   CONFIG["MC_ITERATIONS"]), mkey)

        bars_a = ax.bar(x - width/2, vals_a, width, color=C_A, alpha=0.85, label=LABEL_A)
        bars_b = ax.bar(x + width/2, vals_b, width, color=C_B, alpha=0.85, label=LABEL_B)

        # Reference lines
        if mkey == "profit_factor":
            ax.axhline(1.0, color="#FF4560", lw=0.8, ls="--", alpha=0.7, label="Break-even PF=1")
        elif mkey == "win_rate":
            ax.axhline(1/3, color="#FF4560", lw=0.8, ls="--", alpha=0.7, label="BE win rate")
        elif mkey == "expectancy_r":
            ax.axhline(0.0, color="#FF4560", lw=0.8, ls="--", alpha=0.7)
        elif mkey == "max_drawdown":
            ax.axhline(-0.20, color="#FF4560", lw=0.8, ls="--", alpha=0.7, label="-20% caution")

        def _fmt(v, is_pct):
            if is_pct:
                return f"{v:.1%}"
            if mkey == "net_profit":
                return f"${v:+,.0f}"
            if mkey == "n_trades":
                return str(int(v))
            return f"{v:+.3f}" if v < 0 else f"{v:.3f}"

        for bar, val in zip(bars_a, vals_a):
            yp = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    yp + (abs(yp) * 0.03 + 0.001),
                    _fmt(val, pct), ha="center", va="bottom",
                    color="white", fontsize=6.5, fontweight="bold")
        for bar, val in zip(bars_b, vals_b):
            yp = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2,
                    yp + (abs(yp) * 0.03 + 0.001),
                    _fmt(val, pct), ha="center", va="bottom",
                    color="white", fontsize=6.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color="white", fontsize=9)
        ax.set_ylabel(mlabel, color="white", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7, facecolor="#1A1D24",
                      edgecolor="#444", labelcolor="white",
                      loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r008_metric_grid.png")


# ── Chart 2: Equity curves per symbol + aggregate ───────────────────────────

def plot_equity_curves(symbol_results: list,
                       agg_a: dict, agg_b: dict) -> str:
    n_sym = len(symbol_results)
    n_col = 2
    n_row = math.ceil((n_sym + 1) / n_col)  # +1 for aggregate

    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(14, 5.5 * n_row))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Equity Curves — Strategy A vs B  |  R008  |  Liq.Sweep Confirmation Study\n"
        "Blue = Strategy A (unfiltered)  |  Orange = Strategy B (filtered)  |  "
        "Dashed = max drawdown reference",
        fontsize=9, fontweight="bold", color="white",
    )

    flat = np.array(axes).flatten()

    def _plot_pair(ax, eq_a, eq_b, title, n_a, n_b):
        _ax_style(ax)
        x_a = np.arange(len(eq_a))
        x_b = np.arange(len(eq_b))
        ax.plot(x_a, eq_a, color="#4A90D9", lw=1.8, label=f"A (n={n_a})", zorder=3)
        ax.plot(x_b, eq_b, color="#FFB347", lw=1.8, label=f"B (n={n_b})", zorder=3)
        ax.axhline(CONFIG["STARTING_CAPITAL"], color="#555",
                   lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Trade #", color="white", fontsize=8)
        ax.set_ylabel("Capital ($)", color="white", fontsize=8)
        ax.legend(fontsize=7.5, facecolor="#1A1D24",
                  edgecolor="#444", labelcolor="white")

    for i, r in enumerate(symbol_results):
        sym_short = r["symbol"].replace("-USDT-SWAP", "")
        eq_a = r["A"]["m"]["equity"]
        eq_b = r["B"]["m"]["equity"]
        n_a  = r["A"]["m"]["n_trades"]
        n_b  = r["B"]["m"]["n_trades"]
        pf_a = r["A"]["m"]["profit_factor"]
        pf_b = r["B"]["m"]["profit_factor"]
        _plot_pair(flat[i], eq_a, eq_b,
                   f"{sym_short} — PF_A={pf_a:.3f}  PF_B={pf_b:.3f}",
                   n_a, n_b)

    # Aggregate
    eq_a = agg_a["equity"]
    eq_b = agg_b["equity"]
    n_a  = agg_a["n_trades"]
    n_b  = agg_b["n_trades"]
    pf_a = agg_a["profit_factor"]
    pf_b = agg_b["profit_factor"]
    _plot_pair(flat[n_sym], eq_a, eq_b,
               f"AGGREGATE — PF_A={pf_a:.3f}  PF_B={pf_b:.3f}",
               n_a, n_b)

    for j in range(n_sym + 1, len(flat)):
        flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r008_equity_curves.png")


# ── Chart 3: Drawdown comparison ─────────────────────────────────────────────

def plot_drawdown(symbol_results: list,
                  agg_a: dict, agg_b: dict) -> str:
    n_sym = len(symbol_results)
    n_col = 2
    n_row = math.ceil((n_sym + 1) / n_col)

    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(14, 4.0 * n_row))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Drawdown Profile — Strategy A vs B  |  R008  |  Liq.Sweep Confirmation Study\n"
        "Blue = Strategy A  |  Orange = Strategy B  (lower = smaller drawdown = better)",
        fontsize=9, fontweight="bold", color="white",
    )

    flat = np.array(axes).flatten()

    def _dd_pair(ax, dd_a, dd_b, title, mdd_a, mdd_b):
        _ax_style(ax)
        ax.fill_between(np.arange(len(dd_a)), dd_a * 100,
                        alpha=0.35, color="#4A90D9", zorder=2)
        ax.plot(dd_a * 100, color="#4A90D9", lw=1.2,
                label=f"A  MDD={mdd_a:.1%}", zorder=3)
        ax.fill_between(np.arange(len(dd_b)), dd_b * 100,
                        alpha=0.35, color="#FFB347", zorder=2)
        ax.plot(dd_b * 100, color="#FFB347", lw=1.8,
                label=f"B  MDD={mdd_b:.1%}", zorder=3)
        ax.axhline(-20, color="#FF4560", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylim(top=2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Trade #", color="white", fontsize=8)
        ax.set_ylabel("Drawdown (%)", color="white", fontsize=8)
        ax.legend(fontsize=7.5, facecolor="#1A1D24",
                  edgecolor="#444", labelcolor="white")

    for i, r in enumerate(symbol_results):
        sym_short = r["symbol"].replace("-USDT-SWAP", "")
        _dd_pair(flat[i],
                 r["A"]["m"]["drawdown"], r["B"]["m"]["drawdown"],
                 sym_short,
                 r["A"]["m"]["max_drawdown"], r["B"]["m"]["max_drawdown"])

    _dd_pair(flat[n_sym],
             agg_a["drawdown"], agg_b["drawdown"],
             "AGGREGATE",
             agg_a["max_drawdown"], agg_b["max_drawdown"])

    for j in range(n_sym + 1, len(flat)):
        flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r008_drawdown.png")


# ── Chart 4: Monte Carlo distributions ──────────────────────────────────────

def plot_monte_carlo(symbol_results: list,
                     agg_a: dict, agg_b: dict) -> str:
    n_sym = len(symbol_results)
    n_col = 2
    n_row = math.ceil((n_sym + 1) / n_col)

    fig, axes = plt.subplots(n_row, n_col,
                             figsize=(14, 4.0 * n_row))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Monte Carlo Final Equity Distribution  |  R008  |  Liq.Sweep Confirmation Study\n"
        f"N={CONFIG['MC_ITERATIONS']:,} random permutations  |  "
        "Blue = A (unfiltered)  |  Orange = B (filtered)",
        fontsize=9, fontweight="bold", color="white",
    )

    flat = np.array(axes).flatten()
    start = CONFIG["STARTING_CAPITAL"]

    def _mc_pair(ax, mc_a, mc_b, title):
        _ax_style(ax)
        fe_a = mc_a["final_equities"]
        fe_b = mc_b["final_equities"]
        bns  = min(50, max(10, len(fe_a) // 10))
        all_v = np.concatenate([fe_a, fe_b])
        rng   = (float(all_v.min()), float(all_v.max()))
        if rng[0] >= rng[1]:
            rng = (rng[0] - 1, rng[1] + 1)
        ax.hist(fe_a, bins=bns, range=rng, color="#4A90D9", alpha=0.6,
                label=f"A  PP={mc_a['prob_profit']:.1%}  M={mc_a['median']:,.0f}", density=True)
        ax.hist(fe_b, bins=bns, range=rng, color="#FFB347", alpha=0.6,
                label=f"B  PP={mc_b['prob_profit']:.1%}  M={mc_b['median']:,.0f}", density=True)
        ax.axvline(start, color="#FF4560", lw=1.2, ls="--", alpha=0.8, label="Start capital")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Final Equity ($)", color="white", fontsize=8)
        ax.set_ylabel("Density", color="white", fontsize=8)
        ax.legend(fontsize=7, facecolor="#1A1D24",
                  edgecolor="#444", labelcolor="white")

    for i, r in enumerate(symbol_results):
        sym_short = r["symbol"].replace("-USDT-SWAP", "")
        _mc_pair(flat[i], r["A"]["mc"], r["B"]["mc"], sym_short)

    mc_agg_a = monte_carlo(agg_a["pnls"], CONFIG["MC_ITERATIONS"])
    mc_agg_b = monte_carlo(agg_b["pnls"], CONFIG["MC_ITERATIONS"])
    _mc_pair(flat[n_sym], mc_agg_a, mc_agg_b, "AGGREGATE")

    for j in range(n_sym + 1, len(flat)):
        flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r008_monte_carlo.png")


# ── Chart 5: Filter breakdown — signal pass/reject analysis ─────────────────

def plot_filter_breakdown(symbol_results: list,
                           filter_stats: dict) -> str:
    symbols = [r["symbol"].replace("-USDT-SWAP", "") for r in symbol_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Filter Analysis  |  R008  |  How Many LSR Signals Pass the R007 Filter\n"
        f"Filter: ret_vol_10 ≥ {RV_THRESHOLD:.4f}  AND  dist_from_hh_pct ≥ {HH_THRESHOLD:.4f}",
        fontsize=9, fontweight="bold", color="white",
    )

    # Panel 1: stacked bar — base vs filtered signals
    ax = axes[0]
    _ax_style(ax)
    x   = np.arange(len(symbols))
    n_base = [filter_stats[s]["n_base"] for s in
              [r["symbol"] for r in symbol_results]]
    n_pass = [filter_stats[s]["n_pass"] for s in
              [r["symbol"] for r in symbol_results]]
    n_rej  = [b - p for b, p in zip(n_base, n_pass)]

    ax.bar(x, n_pass, color="#00C49A", alpha=0.85, label="Pass filter (Strategy B)")
    ax.bar(x, n_rej, bottom=n_pass, color="#FF4560", alpha=0.70, label="Blocked by filter")

    for xi, (nb, np_, nr) in enumerate(zip(n_base, n_pass, n_rej)):
        pr = np_ / nb if nb > 0 else 0.0
        ax.text(xi, nb + 0.3, f"Total={nb}\n{pr:.0%} pass",
                ha="center", va="bottom", color="white",
                fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, color="white", fontsize=10)
    ax.set_ylabel("Signal Count", color="white")
    ax.set_title("Signals: Base LSR vs Filter-Approved", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    # Panel 2: breakdown of blocked signals
    ax2 = axes[1]
    _ax_style(ax2)
    cat_labels = ["Both pass", "High vol only", "Near HH only", "Neither"]
    cat_colors = ["#00C49A", "#FFB347", "#4A90D9", "#FF4560"]

    totals_by_cat = [0, 0, 0, 0]
    for r in symbol_results:
        s = r["symbol"]
        fs = filter_stats[s]
        totals_by_cat[0] += fs["n_pass"]
        totals_by_cat[1] += fs["n_vol_only"]
        totals_by_cat[2] += fs["n_hh_only"]
        totals_by_cat[3] += fs["n_neither"]

    total = sum(totals_by_cat)
    bars = ax2.bar(cat_labels, totals_by_cat,
                   color=cat_colors, alpha=0.85)
    for bar, v in zip(bars, totals_by_cat):
        pct = v / total if total > 0 else 0.0
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f"{v}  ({pct:.0%})",
                 ha="center", va="bottom",
                 color="white", fontsize=9, fontweight="bold")

    ax2.set_ylabel("Count (all symbols combined)", color="white")
    ax2.set_title("Filter Breakdown by Condition", fontsize=9)
    ax2.tick_params(axis="x", colors="white", labelsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "r008_filter_breakdown.png")


# ── Chart 6: Win-rate / PF scatter with delta annotations ───────────────────

def plot_ab_delta_summary(symbol_results: list,
                           agg_a: dict, agg_b: dict) -> str:
    symbols    = [r["symbol"].replace("-USDT-SWAP", "") for r in symbol_results]
    sym_labels = symbols + ["AGGREGATE"]
    n          = len(sym_labels)

    metrics_pairs = [
        ("profit_factor", "Profit Factor",  "higher is better"),
        ("win_rate",      "Win Rate",       "higher is better"),
        ("expectancy_r",  "Expectancy (R)", "higher is better"),
        ("max_drawdown",  "Max Drawdown",   "less negative = better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Strategy B − Strategy A  Delta per Symbol  |  R008\n"
        "Positive (green) = B better than A  |  Negative (red) = A better",
        fontsize=9.5, fontweight="bold", color="white",
    )

    flat = np.array(axes).flatten()

    def _get_val(res_dict, key):
        if key == "mc_prob_profit":
            return res_dict["mc"]["prob_profit"]
        v = res_dict["m"].get(key, 0.0)
        return v if np.isfinite(v) else 0.0

    def _agg_val(m, key):
        v = m.get(key, 0.0)
        return v if np.isfinite(v) else 0.0

    x = np.arange(n)
    for idx, (key, lbl, note) in enumerate(metrics_pairs):
        ax   = flat[idx]
        _ax_style(ax)

        a_vals = [_get_val(r["A"], key) for r in symbol_results] + \
                 [_agg_val(agg_a, key)]
        b_vals = [_get_val(r["B"], key) for r in symbol_results] + \
                 [_agg_val(agg_b, key)]
        deltas = [b - a for a, b in zip(a_vals, b_vals)]

        bar_c = ["#00C49A" if d > 0 else "#FF4560" for d in deltas]
        bars  = ax.bar(x, deltas, color=bar_c, alpha=0.85, width=0.55)

        ax.axhline(0, color="white", lw=0.9, ls="-", alpha=0.4)

        for bar, d, av, bv in zip(bars, deltas, a_vals, b_vals):
            yp = bar.get_height()
            sign_offset = (abs(yp) * 0.06 + 0.001) * (1 if yp >= 0 else -1)
            if key == "win_rate":
                txt = f"Δ{d:+.1%}\nA={av:.1%} B={bv:.1%}"
            elif key == "net_profit":
                txt = f"Δ${d:+,.0f}"
            else:
                txt = f"Δ{d:+.3f}\nA={av:.3f} B={bv:.3f}"
            va = "bottom" if yp >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2,
                    yp + sign_offset, txt,
                    ha="center", va=va, color="white",
                    fontsize=6.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(sym_labels, color="white", fontsize=9)
        ax.set_ylabel(f"Δ {lbl}", color="white", fontsize=8.5)
        ax.set_title(f"{lbl}  ({note})", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, "r008_ab_delta.png")


# ── Chart 7: Trade-by-trade scatter — which trades pass the filter ───────────

def plot_trade_scatter(symbol_results: list) -> str:
    n_sym = len(symbol_results)
    fig, axes = plt.subplots(1, n_sym, figsize=(6 * n_sym, 7))
    if n_sym == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "R007 Filter in Feature Space  |  R008  |  Each dot = one Strategy A trade\n"
        "X: 10-Bar Ret Vol  |  Y: Dist from 20-Bar HH (%)  |  "
        "Green = passes filter (Strategy B)  |  Red = blocked",
        fontsize=9, fontweight="bold", color="white",
    )

    for i, r in enumerate(symbol_results):
        ax = axes[i]
        _ax_style(ax)
        sym_short = r["symbol"].replace("-USDT-SWAP", "")

        trades_a = r["A"]["trades"]
        if not trades_a:
            ax.set_visible(False)
            continue

        # Enrich with pre-entry context from OOS df
        df_oos = r["df_oos"]
        rv_vals, hh_vals, win_vals = [], [], []

        for t in trades_a:
            et = t["entry_time"]
            # Entry on bar i; signal fired on bar i-1.  Find bar i-1 in df_oos.
            idx_matches = df_oos.index[df_oos["datetime"] == et].tolist()
            if not idx_matches:
                continue
            bar_entry = idx_matches[0]
            if bar_entry < 1:
                continue
            sig_bar = df_oos.iloc[bar_entry - 1]
            rv_vals.append(float(sig_bar.get("ret_vol_10", np.nan)))
            hh_vals.append(float(sig_bar.get("dist_from_hh_pct", np.nan)))
            win_vals.append(bool(t["win"]))

        if not rv_vals:
            ax.set_visible(False)
            continue

        rv_arr = np.array(rv_vals)
        hh_arr = np.array(hh_vals)
        win_arr = np.array(win_vals)
        pass_f  = (rv_arr >= RV_THRESHOLD) & (hh_arr >= HH_THRESHOLD)

        # Plot blocked (red/orange)
        m_block = ~pass_f
        ax.scatter(rv_arr[m_block & ~win_arr],
                   hh_arr[m_block & ~win_arr],
                   s=50, c="#FF4560", alpha=0.5,
                   marker="x", linewidths=1.5,
                   label="Blocked | Loss")
        ax.scatter(rv_arr[m_block & win_arr],
                   hh_arr[m_block & win_arr],
                   s=60, c="#FFB347", alpha=0.5,
                   marker="x", linewidths=1.5,
                   label="Blocked | Win")

        # Plot passing (green)
        ax.scatter(rv_arr[pass_f & ~win_arr],
                   hh_arr[pass_f & ~win_arr],
                   s=60, c="#FF4560", alpha=0.85,
                   marker="o", edgecolors="white", linewidths=0.5,
                   label="Pass | Loss")
        ax.scatter(rv_arr[pass_f & win_arr],
                   hh_arr[pass_f & win_arr],
                   s=80, c="#00C49A", alpha=0.9,
                   marker="o", edgecolors="white", linewidths=0.5,
                   label="Pass | Win")

        # Filter boundary lines
        ax.axvline(RV_THRESHOLD, color="#FFD700", lw=1.2, ls="--", alpha=0.8,
                   label=f"RV thresh={RV_THRESHOLD:.3f}")
        ax.axhline(HH_THRESHOLD, color="#FFD700", lw=1.2, ls="--", alpha=0.8,
                   label=f"HH thresh={HH_THRESHOLD:.3f}")

        # Shade accepted quadrant
        xl, xr = ax.get_xlim()
        yb, yt = ax.get_ylim()
        # Re-evaluate limits after scatter
        xr2 = max(rv_arr.max() * 1.1, RV_THRESHOLD * 1.2) if len(rv_arr) else RV_THRESHOLD * 2
        yb2 = min(hh_arr.min() * 1.1, HH_THRESHOLD * 1.2) if len(hh_arr) else HH_THRESHOLD * 2
        ax.add_patch(plt.Rectangle(
            (RV_THRESHOLD, HH_THRESHOLD),
            xr2 - RV_THRESHOLD, 2.0 - HH_THRESHOLD,
            color="#00C49A", alpha=0.07, zorder=0,
        ))

        # Win rate inside vs outside
        n_p   = int(pass_f.sum())
        n_b   = int((~pass_f).sum())
        wr_p  = float(win_arr[pass_f].mean()) if n_p > 0 else float("nan")
        wr_b  = float(win_arr[~pass_f].mean()) if n_b > 0 else float("nan")

        ax.set_title(
            f"{sym_short}  (n={len(rv_arr)})\n"
            f"Inside filter: n={n_p} WR={wr_p:.0%}  |  "
            f"Outside: n={n_b} WR={wr_b:.0%}",
            fontsize=8.5, color="white",
        )
        ax.set_xlabel("ret_vol_10", color="white", fontsize=8)
        ax.set_ylabel("dist_from_hh_pct (%)", color="white", fontsize=8)
        ax.legend(fontsize=6.5, facecolor="#1A1D24",
                  edgecolor="#444", labelcolor="white",
                  loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "r008_trade_scatter.png")


# =============================================================================
# FINAL VERDICT LOGIC
# =============================================================================

def compute_verdict(symbol_results: list,
                    agg_a: dict, agg_b: dict) -> dict:
    """
    Objective, pre-specified verdict criteria.
    PROMOTE only if ALL of:
      1. Aggregate PF_B > PF_A
      2. Aggregate WR_B > WR_A  OR  Aggregate ExpR_B > ExpR_A
      3. Aggregate MDD_B >= MDD_A (less negative = smaller drawdown)
      4. Strategy B is CONSISTENT: PF_B > PF_A in ≥ 2/3 symbols
      5. Trade count not excessive: B retains ≥ 25% of A's trades

    Any failure = REJECT.
    """
    n_sym = len(symbol_results)

    pf_a  = agg_a["profit_factor"]
    pf_b  = agg_b["profit_factor"]
    wr_a  = agg_a["win_rate"]
    wr_b  = agg_b["win_rate"]
    er_a  = agg_a["expectancy_r"]
    er_b  = agg_b["expectancy_r"]
    dd_a  = agg_a["max_drawdown"]
    dd_b  = agg_b["max_drawdown"]
    nt_a  = agg_a["n_trades"]
    nt_b  = agg_b["n_trades"]

    n_consistent = sum(
        1 for r in symbol_results
        if r["B"]["m"]["profit_factor"] > r["A"]["m"]["profit_factor"]
    )
    trade_retention = nt_b / nt_a if nt_a > 0 else 0.0

    c1 = pf_b > pf_a
    c2 = (wr_b > wr_a) or (er_b > er_a)
    c3 = dd_b >= dd_a                        # less negative = better
    c4 = n_consistent >= math.ceil(n_sym * 2/3)
    c5 = trade_retention >= 0.25

    all_pass = c1 and c2 and c3 and c4 and c5

    return {
        "verdict":          "PROMOTE" if all_pass else "REJECT",
        "c1_pf_improve":    c1,
        "c2_wr_or_er":      c2,
        "c3_dd_reduce":     c3,
        "c4_consistent":    c4,
        "c5_trade_count":   c5,
        "n_consistent":     n_consistent,
        "n_sym":            n_sym,
        "trade_retention":  trade_retention,
        "pf_a":             pf_a,
        "pf_b":             pf_b,
        "pf_delta":         pf_b - pf_a,
        "wr_a":             wr_a,
        "wr_b":             wr_b,
        "er_a":             er_a,
        "er_b":             er_b,
        "dd_a":             dd_a,
        "dd_b":             dd_b,
        "nt_a":             nt_a,
        "nt_b":             nt_b,
    }


# =============================================================================
# REPORT
# =============================================================================

def print_r008_report(symbol_results: list,
                       agg_a: dict, agg_b: dict,
                       filter_stats: dict,
                       verdict_info: dict) -> None:
    S  = "=" * 108
    S2 = "─" * 108
    BL = "  "

    print(f"\n{S}")
    print(f"{BL}QUANTLAB AI — RESEARCH #008")
    print(f"{BL}Confirmation Study — Liquidity Sweep Reversal Interaction Hypothesis")
    print(f"{BL}{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(S)

    # ── Semantic clarification ────────────────────────────────────────────────
    print(f"""
{BL}╔══════════════════════════════════════════════════════════════════════╗
{BL}║  SEMANTIC NOTE  (important for interpreting results)                 ║
{BL}║                                                                      ║
{BL}║  The R007 prompt described the filter as "price far from 20-bar      ║
{BL}║  high."  The actual R007 data shows the OPPOSITE sign convention:    ║
{BL}║                                                                      ║
{BL}║  dist_from_hh_pct = (close − 20-bar high) / close × 100             ║
{BL}║  This is ALWAYS NEGATIVE.  High (≥ median −1.6435) means price       ║
{BL}║  is CLOSE to the 20-bar high (within ~1.6%), not far from it.        ║
{BL}║                                                                      ║
{BL}║  Strategy B therefore tests:                                          ║
{BL}║    • High short-term volatility  (ret_vol_10 ≥ 0.356350)             ║
{BL}║    • Price near the 20-bar high  (dist_from_hh_pct ≥ −1.6435)       ║
{BL}║                                                                      ║
{BL}║  This is the EXACT condition found optimal in R007.                  ║
{BL}║  It was NOT changed or corrected.                                    ║
{BL}╚══════════════════════════════════════════════════════════════════════╝
""")

    # ── Filter thresholds ─────────────────────────────────────────────────────
    print(f"{BL}FILTER THRESHOLDS  (R007 medians, locked)")
    print(f"{BL}  ret_vol_10      ≥  {RV_THRESHOLD:.6f}  (10-bar realised vol above OOS median)")
    print(f"{BL}  dist_from_hh_pct ≥ {HH_THRESHOLD:.6f}  (price within 1.64% of 20-bar high)")
    print(f"{BL}  Source: median of all 86 OOS trades (combined across BTC, ETH, SOL)")

    # ── Signal pass rate ──────────────────────────────────────────────────────
    print(f"\n{BL}FILTER PASS RATE  (Strategy A signals → Strategy B signals)")
    print(f"{BL}{'Symbol':<24} {'A Signals':>11} {'B Signals':>11} "
          f"{'Pass Rate':>11} {'Blocked':>8}")
    print(f"{BL}{S2[2:70]}")
    total_a_sigs = total_b_sigs = 0
    for r in symbol_results:
        s  = r["symbol"]
        fs = filter_stats[s]
        total_a_sigs += fs["n_base"]
        total_b_sigs += fs["n_pass"]
        rate_str = f"{fs['pass_rate']:.1%}"
        block_str = str(fs["n_base"] - fs["n_pass"])
        print(f"{BL}{s:<24} {fs['n_base']:>11} {fs['n_pass']:>11} "
              f"{rate_str:>11} {block_str:>8}")
    total_rate = total_b_sigs / total_a_sigs if total_a_sigs > 0 else 0.0
    print(f"{BL}{'TOTAL':<24} {total_a_sigs:>11} {total_b_sigs:>11} "
          f"{total_rate:>11.1%} {total_a_sigs - total_b_sigs:>8}")

    # ── Per-symbol results ────────────────────────────────────────────────────
    metric_display = [
        ("n_trades",       "Trades",           lambda v: str(int(v))),
        ("win_rate",       "Win Rate",          lambda v: f"{v:.1%}"),
        ("profit_factor",  "Profit Factor",     lambda v: f"{v:.3f}"),
        ("expectancy_r",   "Expectancy (R)",    lambda v: f"{v:+.3f}"),
        ("net_profit",     "Net Profit ($)",    lambda v: f"${v:+,.0f}"),
        ("max_drawdown",   "Max Drawdown",      lambda v: f"{v:.1%}"),
        ("sharpe",         "Sharpe",            lambda v: f"{v:.3f}"),
        ("avg_hold_minutes","Avg Hold (min)",   lambda v: f"{v:.0f}"),
    ]

    for r in symbol_results:
        sym   = r["symbol"]
        ma    = r["A"]["m"]
        mb    = r["B"]["m"]
        mc_a  = r["A"]["mc"]
        mc_b  = r["B"]["mc"]
        va    = r["A"]["verdict"]
        vb    = r["B"]["verdict"]

        print(f"\n{BL}{'─'*70}")
        print(f"{BL}{sym}")
        print(f"{BL}{'─'*70}")
        print(f"{BL}{'Metric':<26} {'Strategy A':>14} {'Strategy B':>14}  {'Δ (B−A)':>12}  Better?")
        print(f"{BL}{'─'*80}")

        for key, lbl, fmt in metric_display:
            va_v = ma.get(key, 0.0)
            vb_v = mb.get(key, 0.0)
            delta = vb_v - va_v
            # For drawdown and trade-count, direction differs
            if key == "max_drawdown":
                better = "B ✓" if delta > 0 else ("=" if delta == 0 else "A")
            elif key == "n_trades":
                better = "—"
            else:
                better = "B ✓" if delta > 0 else ("=" if delta == 0 else "A")
            delta_str = fmt(delta).replace("$+", "Δ$+").replace("$-", "Δ$-")
            print(f"{BL}  {lbl:<24} {fmt(va_v):>14} {fmt(vb_v):>14}  "
                  f"{delta_str:>12}  {better}")

        # MC prob
        print(f"{BL}  {'MC Prob Profit':<24} {mc_a['prob_profit']:>14.1%} "
              f"{mc_b['prob_profit']:>14.1%}  "
              f"{'':>12}  {'B ✓' if mc_b['prob_profit'] > mc_a['prob_profit'] else 'A'}")
        print(f"{BL}  Verdict A: {va:<12}  Verdict B: {vb}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print(f"\n{BL}{'─'*70}")
    print(f"{BL}AGGREGATE  (all symbols combined)")
    print(f"{BL}{'─'*70}")
    print(f"{BL}{'Metric':<26} {'Strategy A':>14} {'Strategy B':>14}  {'Δ (B−A)':>12}  Better?")
    print(f"{BL}{'─'*80}")

    mc_agg_a = monte_carlo(agg_a["pnls"], CONFIG["MC_ITERATIONS"])
    mc_agg_b = monte_carlo(agg_b["pnls"], CONFIG["MC_ITERATIONS"])

    for key, lbl, fmt in metric_display:
        va_v = agg_a.get(key, 0.0)
        vb_v = agg_b.get(key, 0.0)
        delta = vb_v - va_v
        if key == "max_drawdown":
            better = "B ✓" if delta > 0 else ("=" if delta == 0 else "A")
        elif key == "n_trades":
            better = "—"
        else:
            better = "B ✓" if delta > 0 else ("=" if delta == 0 else "A")
        delta_str = fmt(delta)
        print(f"{BL}  {lbl:<24} {fmt(va_v):>14} {fmt(vb_v):>14}  "
              f"{delta_str:>12}  {better}")
    print(f"{BL}  {'MC Prob Profit':<24} {mc_agg_a['prob_profit']:>14.1%} "
          f"{mc_agg_b['prob_profit']:>14.1%}  "
          f"{'':>12}  "
          f"{'B ✓' if mc_agg_b['prob_profit'] > mc_agg_a['prob_profit'] else 'A'}")

    # ── Final Questions ───────────────────────────────────────────────────────
    vi = verdict_info
    print(f"\n{S}")
    print(f"{BL}FINAL QUESTIONS — OBJECTIVE ANSWERS")
    print(f"{BL}{S2[2:]}")

    # Q1: Does the interaction improve Profit Factor?
    pf_delta_pct = (vi["pf_b"] / vi["pf_a"] - 1.0) * 100 \
                   if vi["pf_a"] > 0 else 0.0
    print(f"\n{BL}Q1  Does the filter improve Profit Factor?")
    if vi["c1_pf_improve"]:
        print(f"{BL}    → YES  (aggregate PF: A={vi['pf_a']:.3f} → B={vi['pf_b']:.3f}  "
              f"Δ={vi['pf_delta']:+.3f}  +{pf_delta_pct:.1f}%)")
    else:
        print(f"{BL}    → NO   (aggregate PF: A={vi['pf_a']:.3f} → B={vi['pf_b']:.3f}  "
              f"Δ={vi['pf_delta']:+.3f}  {pf_delta_pct:.1f}%)")

    # Q2: Does it reduce drawdown?
    print(f"\n{BL}Q2  Does the filter reduce drawdown?")
    if vi["c3_dd_reduce"]:
        print(f"{BL}    → YES  (aggregate MDD: A={vi['dd_a']:.1%} → B={vi['dd_b']:.1%}  "
              f"Δ={vi['dd_b']-vi['dd_a']:+.1%}  smaller drawdown)")
    else:
        print(f"{BL}    → NO   (aggregate MDD: A={vi['dd_a']:.1%} → B={vi['dd_b']:.1%}  "
              f"Δ={vi['dd_b']-vi['dd_a']:+.1%})")

    # Q3: Does it reduce trade count excessively?
    tr_str = f"{vi['trade_retention']:.1%}"
    print(f"\n{BL}Q3  Does the filter reduce trade count excessively?")
    if vi["c5_trade_count"]:
        print(f"{BL}    → NO   (B retains {tr_str} of A's trades  "
              f"A={vi['nt_a']} → B={vi['nt_b']})  threshold: ≥25%  ✓")
    else:
        print(f"{BL}    → YES  (B retains only {tr_str} of A's trades  "
              f"A={vi['nt_a']} → B={vi['nt_b']})  threshold: ≥25%  ✗")
        print(f"{BL}    Filter removes too many trades for a reliable signal.")

    # Q4: Consistent across symbols?
    print(f"\n{BL}Q4  Is the improvement consistent across symbols?")
    sym_lines = []
    for r in symbol_results:
        sym_s = r["symbol"].replace("-USDT-SWAP", "")
        pf_a_ = r["A"]["m"]["profit_factor"]
        pf_b_ = r["B"]["m"]["profit_factor"]
        arrow = "↑" if pf_b_ > pf_a_ else ("→" if pf_b_ == pf_a_ else "↓")
        sym_lines.append(f"{sym_s}  A={pf_a_:.3f}→B={pf_b_:.3f} {arrow}")
    consistency_str = "  |  ".join(sym_lines)
    if vi["c4_consistent"]:
        print(f"{BL}    → YES  ({vi['n_consistent']}/{vi['n_sym']} symbols show B > A)")
        print(f"{BL}    {consistency_str}")
    else:
        print(f"{BL}    → NO   (only {vi['n_consistent']}/{vi['n_sym']} symbols show B > A, "
              f"need ≥ {math.ceil(vi['n_sym'] * 2/3)}/{vi['n_sym']})")
        print(f"{BL}    {consistency_str}")

    # Q5: Large enough to justify future research?
    print(f"\n{BL}Q5  Is the improvement large enough to justify further research?")
    worth_pursuing = (
        vi["pf_delta"] > 0.05
        and vi["c4_consistent"]
        and vi["c5_trade_count"]
    )
    if worth_pursuing:
        print(f"{BL}    → YES  — ΔPF={vi['pf_delta']:+.3f} with consistent cross-symbol evidence.")
        print(f"{BL}    The filter isolates a denser cluster of wins without eliminating")
        print(f"{BL}    trade frequency.  R009 may explore alternative formulations.")
    elif not vi["c5_trade_count"]:
        print(f"{BL}    → CONDITIONAL  — ΔPF={vi['pf_delta']:+.3f} but trade count too thin.")
        print(f"{BL}    Small-N risk: any observed improvement may be variance.")
        print(f"{BL}    Not ready for further development.")
    elif vi["pf_delta"] > 0 and not vi["c4_consistent"]:
        print(f"{BL}    → MARGINAL  — ΔPF={vi['pf_delta']:+.3f} but inconsistent across symbols.")
        print(f"{BL}    Symbol-specific patterns, not a structural edge.")
    else:
        print(f"{BL}    → NO  — ΔPF={vi['pf_delta']:+.3f}  Improvement too small or negative.")
        print(f"{BL}    Hypothesis does not justify further development as-is.")

    # ── Verdict block ─────────────────────────────────────────────────────────
    v   = vi["verdict"]
    box = "╔" + "═" * 66 + "╗"
    row = f"║{'VERDICT':^66}║"
    v_r = f"║{v:^66}║"
    end = "╚" + "═" * 66 + "╝"
    c1s = "✓" if vi["c1_pf_improve"]  else "✗"
    c2s = "✓" if vi["c2_wr_or_er"]    else "✗"
    c3s = "✓" if vi["c3_dd_reduce"]   else "✗"
    c4s = "✓" if vi["c4_consistent"]  else "✗"
    c5s = "✓" if vi["c5_trade_count"] else "✗"

    print(f"\n{BL}{box}")
    print(f"{BL}{row}")
    print(f"{BL}{v_r}")
    print(f"{BL}{end}")
    print(f"\n{BL}  Criteria checklist:")
    print(f"{BL}  [{c1s}] Aggregate PF improves          "
          f"A={vi['pf_a']:.3f} → B={vi['pf_b']:.3f}")
    print(f"{BL}  [{c2s}] WR or Expectancy improves      "
          f"WR A={vi['wr_a']:.1%}→B={vi['wr_b']:.1%}  "
          f"ExpR A={vi['er_a']:+.3f}→B={vi['er_b']:+.3f}")
    print(f"{BL}  [{c3s}] Drawdown reduces or holds      "
          f"A={vi['dd_a']:.1%} → B={vi['dd_b']:.1%}")
    print(f"{BL}  [{c4s}] Consistent across ≥2/3 symbols "
          f"{vi['n_consistent']}/{vi['n_sym']} symbols agree")
    print(f"{BL}  [{c5s}] Trade count ≥25% retained      "
          f"A={vi['nt_a']} → B={vi['nt_b']} ({vi['trade_retention']:.0%})")

    if v == "PROMOTE":
        print(f"\n{BL}  All criteria met.  The R007 interaction hypothesis is CONFIRMED.")
        print(f"{BL}  Hypothesis: elevated short-term volatility + price near the 20-bar")
        print(f"{BL}  range top is associated with a cleaner Liquidity Sweep entry.")
        print(f"{BL}  This filter warrants formal development in R009.")
    else:
        failed = [
            "PF improvement" if not vi["c1_pf_improve"] else None,
            "WR/Expectancy improvement" if not vi["c2_wr_or_er"] else None,
            "Drawdown reduction" if not vi["c3_dd_reduce"] else None,
            f"cross-symbol consistency ({vi['n_consistent']}/{vi['n_sym']})"
            if not vi["c4_consistent"] else None,
            f"trade count ({vi['trade_retention']:.0%} retained)"
            if not vi["c5_trade_count"] else None,
        ]
        failed = [f for f in failed if f]
        print(f"\n{BL}  Hypothesis REJECTED.  Failed criteria: {', '.join(failed)}.")
        print(f"{BL}  The observed R007 interaction does not hold as an independent filter.")
        print(f"{BL}  Do not attempt to rescue it.  Evidence comes before optimisation.")
        print(f"{BL}  R009 (if any) must be a different hypothesis from a clean start.")

    print(f"\n{BL}  IMPORTANT: Engine, fees, spread, slippage, SL, TP, sizing — unchanged.")
    print(S)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              QUANTLAB AI — RESEARCH #008                                      ║
║   Confirmation Study — Liq.Sweep R007 Interaction Hypothesis                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  Strategy A: Liquidity Sweep Reversal, unchanged.
  Strategy B: Same + [ret_vol_10 ≥ 0.356350] AND [dist_from_hh_pct ≥ −1.6435]

  Thresholds: R007 OOS medians.  No optimisation.  No threshold search.
  Engine, fees, spread, slippage, split: LOCKED.
""")

    random.seed(42)
    np.random.seed(42)

    # ── 1. Run both strategies on all symbols ─────────────────────────────
    print("=" * 70)
    print("  STEP 1: Running Strategy A and B on all symbols")
    print("=" * 70)

    symbol_results = []
    filter_stats   = {}

    for sym in CONFIG["SYMBOLS"]:
        print(f"\n  {sym}")
        r = run_both_strategies(sym)
        symbol_results.append(r)

        # Filter analysis
        fs = filter_analysis(r["df_oos"])
        filter_stats[sym] = fs

        ma = r["A"]["m"]
        mb = r["B"]["m"]
        print(f"    A: n={ma['n_trades']:>3}  WR={ma['win_rate']:.1%}  "
              f"PF={ma['profit_factor']:.3f}  "
              f"ExpR={ma['expectancy_r']:+.3f}  {r['A']['verdict']}")
        print(f"    B: n={mb['n_trades']:>3}  WR={mb['win_rate']:.1%}  "
              f"PF={mb['profit_factor']:.3f}  "
              f"ExpR={mb['expectancy_r']:+.3f}  {r['B']['verdict']}")
        print(f"    Filter pass: {fs['n_pass']}/{fs['n_base']} signals "
              f"({fs['pass_rate']:.1%})")

    # ── 2. Aggregate ──────────────────────────────────────────────────────
    print("\n  STEP 2: Computing aggregate metrics")
    all_trades_a = []
    all_trades_b = []
    for r in symbol_results:
        all_trades_a.extend(r["A"]["trades"])
        all_trades_b.extend(r["B"]["trades"])

    agg_a = compute_metrics(all_trades_a, LABEL_A)
    agg_b = compute_metrics(all_trades_b, LABEL_B)
    mc_agg_a = monte_carlo(agg_a["pnls"], CONFIG["MC_ITERATIONS"])
    mc_agg_b = monte_carlo(agg_b["pnls"], CONFIG["MC_ITERATIONS"])

    print(f"    A aggregate: n={agg_a['n_trades']}  WR={agg_a['win_rate']:.1%}  "
          f"PF={agg_a['profit_factor']:.3f}  ExpR={agg_a['expectancy_r']:+.3f}")
    print(f"    B aggregate: n={agg_b['n_trades']}  WR={agg_b['win_rate']:.1%}  "
          f"PF={agg_b['profit_factor']:.3f}  ExpR={agg_b['expectancy_r']:+.3f}")

    # ── 3. Verdict ────────────────────────────────────────────────────────
    print("\n  STEP 3: Computing verdict")
    verdict_info = compute_verdict(symbol_results, agg_a, agg_b)
    print(f"  → {verdict_info['verdict']}")

    # ── 4. Charts ─────────────────────────────────────────────────────────
    print("\n  STEP 4: Generating charts")
    charts = []

    p = plot_metric_grid(symbol_results, agg_a, agg_b)
    charts.append(p); print(f"  → {p}")

    p = plot_equity_curves(symbol_results, agg_a, agg_b)
    charts.append(p); print(f"  → {p}")

    p = plot_drawdown(symbol_results, agg_a, agg_b)
    charts.append(p); print(f"  → {p}")

    p = plot_monte_carlo(symbol_results, agg_a, agg_b)
    charts.append(p); print(f"  → {p}")

    p = plot_filter_breakdown(symbol_results, filter_stats)
    charts.append(p); print(f"  → {p}")

    p = plot_ab_delta_summary(symbol_results, agg_a, agg_b)
    charts.append(p); print(f"  → {p}")

    p = plot_trade_scatter(symbol_results)
    charts.append(p); print(f"  → {p}")

    # ── 5. Full report ────────────────────────────────────────────────────
    print_r008_report(symbol_results, agg_a, agg_b,
                       filter_stats, verdict_info)

    # ── 6. Journal ────────────────────────────────────────────────────────
    jnl_rows = []
    for r in symbol_results:
        for strat_key, strat_label in [("A", LABEL_A), ("B", LABEL_B)]:
            m  = r[strat_key]["m"]
            mc = r[strat_key]["mc"]
            vd = r[strat_key]["verdict"]
            row = _journal_row(strat_label, r["symbol"], m, mc, vd)
            row["research_id"] = RESEARCH_ID
            jnl_rows.append(row)
    if jnl_rows:
        append_journal(jnl_rows)
        print(f"\n  Journal updated → {CONFIG['JOURNAL_FILE']}")

    print(f"\n  All outputs → {OUTPUT_FOLDER}/")
    print("  Research #008 complete.\n")


if __name__ == "__main__":
    main()
