"""
=============================================================================
QUANTLAB AI – RESEARCH #009
Session-Time + Relative-Volume Filter for Liquidity Sweep Reversal

Objective:
  Test whether restricting LSR entries to the liquid market hours
  (08:00–21:59 UTC, covering London + New York sessions) combined with
  elevated relative volume (rel_vol ≥ REL_VOL_THRESHOLD) improves the
  baseline Liquidity Sweep Reversal strategy.

Hypothesis:
  Smart-money liquidity sweeps that happen during peak-volume market hours
  with above-average volume participation are structurally different from
  off-hours sweeps.  Low-volume, Asia-session sweeps may be noise; high-volume
  London/NY sweeps are more likely to be genuine institutional stop-hunts that
  reverse cleanly.

Method:
  Strategy A — Liquidity Sweep Reversal, unchanged (same as all prior research).
  Strategy B — Same strategy + TWO pre-entry conditions:
               1. hour_utc in [8, 21]   (London + New York session window)
               2. rel_vol ≥ 1.2         (relative volume ≥ 120% of 48-bar mean)

  Thresholds:
    Hour window: 08–21 UTC — a round, interpretable cut.  NOT optimised.
    rel_vol ≥ 1.2 — round number, NOT a searched value.
    Both chosen BEFORE running any backtest.  No threshold search.

  Motivation:
    R005 captured session and volume data for every trade.  The R005
    time-heatmap and feature-importance outputs motivated this hypothesis.
    This research tests the directional prediction; it does NOT use any
    R005 OOS outcome labels to pick the thresholds.

Locked: engine, fees, spread, slippage, SL, TP, sizing, split.
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

RESEARCH_ID   = "R009"
OUTPUT_FOLDER = CONFIG["OUTPUT_FOLDER"]
BG            = "#0F1117"

# ── Filter thresholds (pre-specified, not optimised) ─────────────────────────
SESSION_START    = 8      # inclusive  (08:00 UTC — London open)
SESSION_END      = 21     # inclusive  (21:59 UTC — NY session)
REL_VOL_THRESHOLD = 1.2   # rel_vol ≥ this → above-average participation

LABEL_A = "Liq.Sweep (A)"
LABEL_B = "Liq.Sweep+SessVol (B)"

METRIC_LABELS = {
    "n_trades":         "Trades",
    "win_rate":         "Win Rate",
    "profit_factor":    "Profit Factor",
    "expectancy_r":     "Expectancy (R)",
    "net_profit":       "Net Profit ($)",
    "max_drawdown":     "Max Drawdown",
    "sharpe":           "Sharpe Ratio",
    "avg_hold_minutes": "Avg Hold (min)",
    "mc_prob_profit":   "MC Prob Profit",
}


# =============================================================================
# STRATEGY B — LSR + Session/Volume filter
# =============================================================================

def strategy_lsr_b(df: pd.DataFrame) -> pd.Series:
    """
    Strategy B: Liquidity Sweep Reversal + Session-Time + Relative-Volume filter.

    Entry conditions (all on signal-bar close, no look-ahead):
      1. sweep:    low < lsr_prior_low              (wick below prior 5-bar low)
      2. reclaim:  close > lsr_prior_low            (close back above swept level)
      3. bullish:  close > open                     (rejection candle)
      4. trend:    close > ema200                   (uptrend)
      ── R009 additions ─────────────────────────────────────────────────────
      5. session:  hour_utc in [SESSION_START, SESSION_END]
                   (London + New York liquidity window)
      6. volume:   rel_vol ≥ REL_VOL_THRESHOLD      (above-average participation)

    Conditions 5 & 6 are the ONLY change from Strategy A.
    Stop, entry, TP, sizing, fees: unchanged.
    """
    # Base LSR conditions (identical to strategy_lsr)
    sweep   = df["low"]   < df["lsr_prior_low"]
    reclaim = df["close"] > df["lsr_prior_low"]
    bullish = df["close"] > df["open"]
    trend   = df["close"] > df["ema200"]

    # R009 additions
    hour_utc = df["datetime"].dt.hour
    in_session = (hour_utc >= SESSION_START) & (hour_utc <= SESSION_END)
    high_vol   = df["rel_vol"] >= REL_VOL_THRESHOLD

    return sweep & reclaim & bullish & trend & in_session & high_vol


# =============================================================================
# DATA PIPELINE
# =============================================================================

def prepare_oos_df(symbol: str) -> pd.DataFrame:
    """
    Load price data, compute ALL indicators (base + R005 context),
    return the OOS slice.  Identical train/test split to all prior research.
    """
    df    = get_data(symbol)
    n     = len(df)
    df    = add_indicators(df)
    df    = add_r005_indicators(df)
    split = int(n * CONFIG["TRAIN_RATIO"])
    return df.iloc[split:].reset_index(drop=True)


def run_both_strategies(symbol: str) -> dict:
    """
    Run Strategy A and Strategy B on the same OOS slice.
    Returns per-symbol results dict.
    """
    df_oos = prepare_oos_df(symbol)

    # Strategy A
    res_a     = run_backtest(df_oos, strategy_lsr,   LABEL_A)
    metrics_a = compute_metrics(res_a["trades"], LABEL_A)
    mc_a      = monte_carlo(metrics_a["pnls"], CONFIG["MC_ITERATIONS"])
    verdict_a = _verdict_from_metrics(metrics_a, mc_a)

    # Strategy B
    res_b     = run_backtest(df_oos, strategy_lsr_b, LABEL_B)
    metrics_b = compute_metrics(res_b["trades"], LABEL_B)
    mc_b      = monte_carlo(metrics_b["pnls"], CONFIG["MC_ITERATIONS"])
    verdict_b = _verdict_from_metrics(metrics_b, mc_b)

    return {
        "symbol": symbol,
        "df_oos": df_oos,
        "A": {"trades": res_a["trades"], "m": metrics_a,
              "mc": mc_a, "verdict": verdict_a},
        "B": {"trades": res_b["trades"], "m": metrics_b,
              "mc": mc_b, "verdict": verdict_b},
    }


def aggregate_metrics(all_trades: list, label: str) -> dict:
    if not all_trades:
        return {
            "label": label, "n_trades": 0, "net_profit": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "avg_trade": 0.0,
            "avg_r": 0.0, "expectancy_r": 0.0,
            "largest_win": 0.0, "largest_loss": 0.0,
            "max_drawdown": 0.0, "sharpe": 0.0,
            "avg_hold_minutes": 0.0, "total_funding_windows": 0,
            "equity": np.array([CONFIG["STARTING_CAPITAL"]]),
            "drawdown": np.array([0.0]),
            "pnls": np.array([]), "r_multiples": np.array([]),
            "trades_df": pd.DataFrame(),
        }
    return compute_metrics(all_trades, label)


# =============================================================================
# FILTER ANALYSIS
# =============================================================================

def filter_analysis(df_oos: pd.DataFrame) -> dict:
    """
    Break down signal counts: how many base LSR signals fire,
    how many pass each filter condition, and the combined pass rate.
    """
    base_sig = strategy_lsr(df_oos)
    n_base   = int(base_sig.sum())

    hour_utc   = df_oos["datetime"].dt.hour
    in_session = (hour_utc >= SESSION_START) & (hour_utc <= SESSION_END)
    high_vol   = df_oos["rel_vol"] >= REL_VOL_THRESHOLD
    both       = in_session & high_vol

    pass_both     = int((base_sig & both).sum())
    pass_sess_only = int((base_sig & in_session & ~high_vol).sum())
    pass_vol_only  = int((base_sig & ~in_session & high_vol).sum())
    pass_neither   = int((base_sig & ~in_session & ~high_vol).sum())

    return {
        "n_base":           n_base,
        "n_pass":           pass_both,
        "n_sess_only":      pass_sess_only,
        "n_vol_only":       pass_vol_only,
        "n_neither":        pass_neither,
        "pass_rate":        pass_both / n_base if n_base > 0 else 0.0,
    }


# =============================================================================
# SESSION BREAKDOWN — win rate by hour bucket
# =============================================================================

def session_win_rate_analysis(symbol_results: list) -> dict:
    """
    Pool all Strategy A trades and compute win rate / count by UTC hour bucket.
    Helps visualise the session pattern that motivates the filter.
    """
    all_rows = []
    for r in symbol_results:
        df_oos   = r["df_oos"]
        trades_a = r["A"]["trades"]
        dt_to_pos = {dt: pos for pos, dt in enumerate(df_oos["datetime"])}
        for t in trades_a:
            ep = t["entry_time"]
            pos = dt_to_pos.get(ep)
            if pos is None or pos < 1:
                continue
            sig = df_oos.iloc[pos - 1]
            all_rows.append({
                "hour_utc": ep.hour,
                "rel_vol":  float(sig.get("rel_vol", float("nan"))),
                "win":      int(t["win"]),
                "pnl":      t["pnl"],
            })

    if not all_rows:
        return {}

    df = pd.DataFrame(all_rows)
    by_hour = (
        df.groupby("hour_utc")
        .agg(n=("win", "count"), wins=("win", "sum"), pnl=("pnl", "sum"))
        .assign(win_rate=lambda x: x["wins"] / x["n"])
        .reset_index()
    )
    return {"by_hour": by_hour, "all_df": df}


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


# ── Chart 1: Metric comparison grid ─────────────────────────────────────────

def plot_metric_grid(symbol_results: list,
                     agg_a: dict, agg_b: dict) -> str:
    symbols = [r["symbol"].replace("-USDT-SWAP", "") for r in symbol_results]
    labels  = symbols + ["AGGREGATE"]

    metrics = [
        ("profit_factor",   "Profit Factor",   False),
        ("win_rate",        "Win Rate",        True),
        ("expectancy_r",    "Expectancy (R)",  False),
        ("net_profit",      "Net Profit ($)",  False),
        ("max_drawdown",    "Max Drawdown",    False),
        ("sharpe",          "Sharpe Ratio",    False),
        ("n_trades",        "Trade Count",     False),
        ("mc_prob_profit",  "MC Prob Profit",  True),
    ]

    n_met = len(metrics)
    fig, axes = plt.subplots(n_met, 1, figsize=(14, 3.0 * n_met))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Strategy A vs Strategy B — All Metrics  |  R009  |  Session-Time + Volume Filter\n"
        f"Filter: hour_utc in [{SESSION_START}, {SESSION_END}]  AND  rel_vol ≥ {REL_VOL_THRESHOLD}\n"
        "Blue = Strategy A (unfiltered)  |  Orange = Strategy B (filtered)",
        fontsize=10, fontweight="bold", color="white",
    )

    x     = np.arange(len(labels))
    width = 0.35
    C_A   = "#4A90D9"
    C_B   = "#FFB347"

    def _get(res, key):
        if key == "mc_prob_profit":
            return res["mc"]["prob_profit"]
        v = res["m"].get(key, 0.0)
        return v if (np.isfinite(v) if isinstance(v, float) else True) else 0.0

    def _agg_v(m, mc, key):
        if key == "mc_prob_profit":
            return mc["prob_profit"]
        v = m.get(key, 0.0)
        return v if np.isfinite(v) else 0.0

    mc_agg_a = monte_carlo(agg_a["pnls"], CONFIG["MC_ITERATIONS"])
    mc_agg_b = monte_carlo(agg_b["pnls"], CONFIG["MC_ITERATIONS"])

    for idx, (mkey, mlabel, pct) in enumerate(metrics):
        ax = axes[idx]
        _ax_style(ax)

        vals_a = [_get(r["A"], mkey) for r in symbol_results] + [_agg_v(agg_a, mc_agg_a, mkey)]
        vals_b = [_get(r["B"], mkey) for r in symbol_results] + [_agg_v(agg_b, mc_agg_b, mkey)]

        bars_a = ax.bar(x - width/2, vals_a, width, color=C_A, alpha=0.85, label=LABEL_A)
        bars_b = ax.bar(x + width/2, vals_b, width, color=C_B, alpha=0.85, label=LABEL_B)

        if mkey == "profit_factor":
            ax.axhline(1.0, color="#FF4560", lw=0.8, ls="--", alpha=0.7, label="PF=1 (break-even)")
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
                      edgecolor="#444", labelcolor="white", loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r009_metric_grid.png")


# ── Chart 2: Equity curves per symbol + aggregate ───────────────────────────

def plot_equity_curves(symbol_results: list,
                       agg_a: dict, agg_b: dict) -> str:
    n_sym = len(symbol_results)
    n_col = 2
    n_row = math.ceil((n_sym + 1) / n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(14, 5.5 * n_row))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Equity Curves — Strategy A vs B  |  R009  |  Session-Time + Volume Filter\n"
        "Blue = A (unfiltered)  |  Orange = B (filtered)",
        fontsize=9, fontweight="bold", color="white",
    )

    flat = np.array(axes).flatten()

    def _plot_pair(ax, eq_a, eq_b, title, n_a, n_b):
        _ax_style(ax)
        ax.plot(np.arange(len(eq_a)), eq_a, color="#4A90D9", lw=1.8,
                label=f"A (n={n_a})", zorder=3)
        ax.plot(np.arange(len(eq_b)), eq_b, color="#FFB347", lw=1.8,
                label=f"B (n={n_b})", zorder=3)
        ax.axhline(CONFIG["STARTING_CAPITAL"], color="#555",
                   lw=0.8, ls="--", alpha=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Trade #", color="white", fontsize=8)
        ax.set_ylabel("Capital ($)", color="white", fontsize=8)
        ax.legend(fontsize=7.5, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    for i, r in enumerate(symbol_results):
        sym = r["symbol"].replace("-USDT-SWAP", "")
        pf_a = r["A"]["m"]["profit_factor"]
        pf_b = r["B"]["m"]["profit_factor"]
        _plot_pair(flat[i],
                   r["A"]["m"]["equity"], r["B"]["m"]["equity"],
                   f"{sym} — PF_A={pf_a:.3f}  PF_B={pf_b:.3f}",
                   r["A"]["m"]["n_trades"], r["B"]["m"]["n_trades"])

    pf_a = agg_a["profit_factor"]
    pf_b = agg_b["profit_factor"]
    _plot_pair(flat[n_sym],
               agg_a["equity"], agg_b["equity"],
               f"AGGREGATE — PF_A={pf_a:.3f}  PF_B={pf_b:.3f}",
               agg_a["n_trades"], agg_b["n_trades"])

    for j in range(n_sym + 1, len(flat)):
        flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r009_equity_curves.png")


# ── Chart 3: Drawdown comparison ─────────────────────────────────────────────

def plot_drawdown(symbol_results: list,
                  agg_a: dict, agg_b: dict) -> str:
    n_sym = len(symbol_results)
    n_col = 2
    n_row = math.ceil((n_sym + 1) / n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(14, 4.0 * n_row))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Drawdown Profile — Strategy A vs B  |  R009  |  Session-Time + Volume Filter",
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
        ax.legend(fontsize=7.5, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    for i, r in enumerate(symbol_results):
        sym = r["symbol"].replace("-USDT-SWAP", "")
        _dd_pair(flat[i],
                 r["A"]["m"]["drawdown"], r["B"]["m"]["drawdown"],
                 sym,
                 r["A"]["m"]["max_drawdown"], r["B"]["m"]["max_drawdown"])

    _dd_pair(flat[n_sym],
             agg_a["drawdown"], agg_b["drawdown"],
             "AGGREGATE",
             agg_a["max_drawdown"], agg_b["max_drawdown"])

    for j in range(n_sym + 1, len(flat)):
        flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r009_drawdown.png")


# ── Chart 4: Monte Carlo distributions ──────────────────────────────────────

def plot_monte_carlo(symbol_results: list,
                     agg_a: dict, agg_b: dict) -> str:
    n_sym = len(symbol_results)
    n_col = 2
    n_row = math.ceil((n_sym + 1) / n_col)

    fig, axes = plt.subplots(n_row, n_col, figsize=(14, 4.0 * n_row))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Monte Carlo Final Equity Distribution  |  R009  |  N={CONFIG['MC_ITERATIONS']:,} permutations\n"
        "Blue = A (unfiltered)  |  Orange = B (filtered)",
        fontsize=9, fontweight="bold", color="white",
    )

    flat  = np.array(axes).flatten()
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
        ax.legend(fontsize=7, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    for i, r in enumerate(symbol_results):
        sym = r["symbol"].replace("-USDT-SWAP", "")
        _mc_pair(flat[i], r["A"]["mc"], r["B"]["mc"], sym)

    mc_agg_a = monte_carlo(agg_a["pnls"], CONFIG["MC_ITERATIONS"])
    mc_agg_b = monte_carlo(agg_b["pnls"], CONFIG["MC_ITERATIONS"])
    _mc_pair(flat[n_sym], mc_agg_a, mc_agg_b, "AGGREGATE")

    for j in range(n_sym + 1, len(flat)):
        flat[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "r009_monte_carlo.png")


# ── Chart 5: Filter breakdown ─────────────────────────────────────────────────

def plot_filter_breakdown(symbol_results: list, filter_stats: dict) -> str:
    symbols = [r["symbol"].replace("-USDT-SWAP", "") for r in symbol_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Filter Analysis  |  R009  |  How Many LSR Signals Pass the Session+Volume Filter\n"
        f"Filter: hour_utc ∈ [{SESSION_START}, {SESSION_END}]  AND  rel_vol ≥ {REL_VOL_THRESHOLD}",
        fontsize=9, fontweight="bold", color="white",
    )

    # Panel 1: base vs filtered
    ax = axes[0]
    _ax_style(ax)
    x      = np.arange(len(symbols))
    n_base = [filter_stats[r["symbol"]]["n_base"] for r in symbol_results]
    n_pass = [filter_stats[r["symbol"]]["n_pass"] for r in symbol_results]
    n_rej  = [b - p for b, p in zip(n_base, n_pass)]

    ax.bar(x, n_pass, color="#00C49A", alpha=0.85, label="Pass filter (Strategy B)")
    ax.bar(x, n_rej,  bottom=n_pass, color="#FF4560", alpha=0.70, label="Blocked by filter")

    for xi, (nb, np_, _) in enumerate(zip(n_base, n_pass, n_rej)):
        pr = np_ / nb if nb > 0 else 0.0
        ax.text(xi, nb + 0.3, f"Total={nb}\n{pr:.0%} pass",
                ha="center", va="bottom", color="white", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, color="white", fontsize=10)
    ax.set_ylabel("Signal Count", color="white")
    ax.set_title("Signals: Base LSR vs Filter-Approved", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    # Panel 2: breakdown by condition
    ax2 = axes[1]
    _ax_style(ax2)
    cat_labels = ["Both pass", "Session only", "High vol only", "Neither"]
    cat_colors = ["#00C49A", "#FFB347", "#4A90D9", "#FF4560"]
    totals = [0, 0, 0, 0]
    for r in symbol_results:
        fs = filter_stats[r["symbol"]]
        totals[0] += fs["n_pass"]
        totals[1] += fs["n_sess_only"]
        totals[2] += fs["n_vol_only"]
        totals[3] += fs["n_neither"]

    total = sum(totals)
    bars  = ax2.bar(cat_labels, totals, color=cat_colors, alpha=0.85)
    for bar, v in zip(bars, totals):
        pct = v / total if total > 0 else 0.0
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.2,
                 f"{v}  ({pct:.0%})",
                 ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")

    ax2.set_ylabel("Count (all symbols combined)", color="white")
    ax2.set_title("Filter Breakdown by Condition", fontsize=9)
    ax2.tick_params(axis="x", colors="white", labelsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "r009_filter_breakdown.png")


# ── Chart 6: Delta summary ────────────────────────────────────────────────────

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
        "Strategy B − Strategy A  Delta per Symbol  |  R009\n"
        "Positive (green) = B better than A  |  Negative (red) = A better",
        fontsize=9.5, fontweight="bold", color="white",
    )

    flat = np.array(axes).flatten()
    x    = np.arange(n)

    def _gv(res_dict, key):
        v = res_dict["m"].get(key, 0.0)
        return v if np.isfinite(v) else 0.0

    def _av(m, key):
        v = m.get(key, 0.0)
        return v if np.isfinite(v) else 0.0

    for idx, (key, lbl, note) in enumerate(metrics_pairs):
        ax  = flat[idx]
        _ax_style(ax)

        a_vals = [_gv(r["A"], key) for r in symbol_results] + [_av(agg_a, key)]
        b_vals = [_gv(r["B"], key) for r in symbol_results] + [_av(agg_b, key)]
        deltas = [b - a for a, b in zip(a_vals, b_vals)]

        bar_c = ["#00C49A" if d > 0 else "#FF4560" for d in deltas]
        bars  = ax.bar(x, deltas, color=bar_c, alpha=0.85, width=0.55)
        ax.axhline(0, color="white", lw=0.9, ls="-", alpha=0.4)

        for bar, d, av, bv in zip(bars, deltas, a_vals, b_vals):
            yp = bar.get_height()
            so = (abs(yp) * 0.06 + 0.001) * (1 if yp >= 0 else -1)
            if key == "win_rate":
                txt = f"Δ{d:+.1%}\nA={av:.1%} B={bv:.1%}"
            else:
                txt = f"Δ{d:+.3f}\nA={av:.3f} B={bv:.3f}"
            va = "bottom" if yp >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2,
                    yp + so, txt,
                    ha="center", va=va, color="white",
                    fontsize=6.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(sym_labels, color="white", fontsize=9)
        ax.set_ylabel(f"Δ {lbl}", color="white", fontsize=8.5)
        ax.set_title(f"{lbl}  ({note})", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, "r009_ab_delta.png")


# ── Chart 7: Hour-of-day win rate heatmap ────────────────────────────────────

def plot_session_analysis(symbol_results: list) -> str:
    """
    Show win rate and trade count by UTC hour for all Strategy A trades.
    Shade the session window the filter uses.
    """
    sess_data = session_win_rate_analysis(symbol_results)
    if not sess_data or "by_hour" not in sess_data:
        return ""

    by_hour = sess_data["by_hour"].set_index("hour_utc").reindex(
        np.arange(0, 24), fill_value=0
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Session Analysis — All Strategy A Trades (combined BTC+ETH+SOL)  |  R009\n"
        f"Orange shading = session window used in Filter (hours {SESSION_START}–{SESSION_END} UTC)",
        fontsize=9.5, fontweight="bold", color="white",
    )

    # Panel 1: Trade count by hour
    ax = axes[0]
    _ax_style(ax)
    hours = np.arange(24)
    counts = [int(by_hour.loc[h, "n"]) if h in by_hour.index else 0 for h in hours]
    bar_colors = ["#FFB347" if SESSION_START <= h <= SESSION_END else "#4A90D9"
                  for h in hours]
    ax.bar(hours, counts, color=bar_colors, alpha=0.85)
    ax.axvspan(SESSION_START - 0.5, SESSION_END + 0.5,
               alpha=0.08, color="#FFB347", zorder=0)
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}" for h in hours], fontsize=7)
    ax.set_xlabel("UTC Hour", color="white", fontsize=9)
    ax.set_ylabel("Trade Count", color="white", fontsize=9)
    ax.set_title("Number of LSR Signals by UTC Hour", fontsize=9)

    # Panel 2: Win rate by hour
    ax2 = axes[1]
    _ax_style(ax2)
    win_rates = []
    for h in hours:
        n = int(by_hour.loc[h, "n"]) if h in by_hour.index else 0
        w = int(by_hour.loc[h, "wins"]) if h in by_hour.index else 0
        win_rates.append(w / n if n > 0 else float("nan"))

    valid_h  = [h for h, wr in zip(hours, win_rates) if not np.isnan(wr)]
    valid_wr = [wr for wr in win_rates if not np.isnan(wr)]
    bar_c2   = ["#FFB347" if SESSION_START <= h <= SESSION_END else "#4A90D9"
                for h in valid_h]
    ax2.bar(valid_h, valid_wr, color=bar_c2, alpha=0.85)
    ax2.axhline(1/3, color="#FF4560", lw=1.0, ls="--", alpha=0.7, label="BE win rate (33%)")
    ax2.axhline(0.5, color="#00C49A", lw=0.8, ls=":", alpha=0.5, label="50% reference")
    ax2.axvspan(SESSION_START - 0.5, SESSION_END + 0.5,
                alpha=0.08, color="#FFB347", zorder=0)
    ax2.set_xticks(hours)
    ax2.set_xticklabels([f"{h:02d}" for h in hours], fontsize=7)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("UTC Hour", color="white", fontsize=9)
    ax2.set_ylabel("Win Rate", color="white", fontsize=9)
    ax2.set_title("Win Rate by UTC Hour (Strategy A trades)", fontsize=9)
    ax2.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    # Annotate count on each bar
    for h, wr in zip(valid_h, valid_wr):
        n_ = int(by_hour.loc[h, "n"]) if h in by_hour.index else 0
        ax2.text(h, wr + 0.02, f"n={n_}", ha="center", va="bottom",
                 color="white", fontsize=6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, "r009_session_analysis.png")


# ── Chart 8: Relative-volume distribution (wins vs losses) ───────────────────

def plot_rel_vol_distribution(symbol_results: list) -> str:
    """
    Histogram of rel_vol at signal bar, split by win/loss, for Strategy A trades.
    Mark the filter threshold.
    """
    rows = []
    for r in symbol_results:
        df_oos = r["df_oos"]
        dt_to_pos = {dt: pos for pos, dt in enumerate(df_oos["datetime"])}
        for t in r["A"]["trades"]:
            pos = dt_to_pos.get(t["entry_time"])
            if pos is None or pos < 1:
                continue
            sig = df_oos.iloc[pos - 1]
            rows.append({
                "rel_vol": float(sig.get("rel_vol", float("nan"))),
                "win":     int(t["win"]),
                "symbol":  r["symbol"].replace("-USDT-SWAP", ""),
            })

    if not rows:
        return ""

    df = pd.DataFrame(rows).dropna(subset=["rel_vol"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Relative Volume at Signal Bar — Wins vs Losses  |  R009\n"
        f"Orange line = rel_vol threshold ({REL_VOL_THRESHOLD})",
        fontsize=9.5, fontweight="bold", color="white",
    )

    # Panel 1: combined histogram
    ax = axes[0]
    _ax_style(ax)
    wins_rv  = df[df["win"] == 1]["rel_vol"].values
    loss_rv  = df[df["win"] == 0]["rel_vol"].values
    all_rv   = df["rel_vol"].values
    bns      = min(40, max(10, len(all_rv) // 3))
    rng      = (float(np.percentile(all_rv, 1)), float(np.percentile(all_rv, 99)))
    ax.hist(wins_rv, bins=bns, range=rng, color="#00C49A", alpha=0.65,
            label=f"Wins (n={len(wins_rv)})", density=True)
    ax.hist(loss_rv, bins=bns, range=rng, color="#FF4560", alpha=0.65,
            label=f"Losses (n={len(loss_rv)})", density=True)
    ax.axvline(REL_VOL_THRESHOLD, color="#FFB347", lw=1.8, ls="--",
               label=f"Filter threshold={REL_VOL_THRESHOLD}")
    ax.set_xlabel("Relative Volume", color="white", fontsize=9)
    ax.set_ylabel("Density", color="white", fontsize=9)
    ax.set_title("All Symbols Combined", fontsize=9)
    ax.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    # Panel 2: win rate by rel_vol quartile
    ax2 = axes[1]
    _ax_style(ax2)
    df["rv_quartile"] = pd.qcut(df["rel_vol"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    q_stats = (df.groupby("rv_quartile", observed=True)["win"]
               .agg(["count", "sum"])
               .assign(wr=lambda x: x["sum"] / x["count"])
               .reset_index())
    q_labels = q_stats["rv_quartile"].astype(str).tolist()
    q_wr     = q_stats["wr"].values
    q_n      = q_stats["count"].values

    bar_c = ["#00C49A" if wr >= 1/3 else "#FF4560" for wr in q_wr]
    bars  = ax2.bar(np.arange(len(q_labels)), q_wr, color=bar_c, alpha=0.85)
    ax2.axhline(1/3, color="#FF4560", lw=1.0, ls="--", alpha=0.7, label="BE win rate")

    for bar, wr, n_ in zip(bars, q_wr, q_n):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{wr:.1%}\nn={n_}",
                 ha="center", va="bottom", color="white",
                 fontsize=8, fontweight="bold")

    ax2.set_xticks(np.arange(len(q_labels)))
    ax2.set_xticklabels(q_labels, color="white", fontsize=8.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Win Rate", color="white", fontsize=9)
    ax2.set_title("Win Rate by Relative Volume Quartile (all symbols)", fontsize=9)
    ax2.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "r009_rel_vol_distribution.png")


# =============================================================================
# VERDICT LOGIC
# =============================================================================

def compute_verdict(symbol_results: list,
                    agg_a: dict, agg_b: dict) -> dict:
    """
    Objective, pre-specified verdict criteria.
    PROMOTE only if ALL of:
      1. Aggregate PF_B > PF_A
      2. Aggregate WR_B > WR_A  OR  Aggregate ExpR_B > ExpR_A
      3. Aggregate MDD_B >= MDD_A  (less negative = better)
      4. Consistent: PF_B > PF_A in >= 2/3 symbols
      5. Trade count: B retains >= 25% of A's trades
    """
    n_sym = len(symbol_results)

    pf_a = agg_a["profit_factor"]
    pf_b = agg_b["profit_factor"]
    wr_a = agg_a["win_rate"]
    wr_b = agg_b["win_rate"]
    er_a = agg_a["expectancy_r"]
    er_b = agg_b["expectancy_r"]
    dd_a = agg_a["max_drawdown"]
    dd_b = agg_b["max_drawdown"]
    nt_a = agg_a["n_trades"]
    nt_b = agg_b["n_trades"]

    n_consistent = sum(
        1 for r in symbol_results
        if r["B"]["m"]["profit_factor"] > r["A"]["m"]["profit_factor"]
    )
    trade_retention = nt_b / nt_a if nt_a > 0 else 0.0

    c1 = pf_b > pf_a
    c2 = (wr_b > wr_a) or (er_b > er_a)
    c3 = dd_b >= dd_a
    c4 = n_consistent >= math.ceil(n_sym * 2 / 3)
    c5 = trade_retention >= 0.25

    return {
        "verdict":         "PROMOTE" if (c1 and c2 and c3 and c4 and c5) else "REJECT",
        "c1_pf_improve":   c1,
        "c2_wr_or_er":     c2,
        "c3_dd_reduce":    c3,
        "c4_consistent":   c4,
        "c5_trade_count":  c5,
        "n_consistent":    n_consistent,
        "n_sym":           n_sym,
        "trade_retention": trade_retention,
        "pf_a":  pf_a, "pf_b":  pf_b, "pf_delta": pf_b - pf_a,
        "wr_a":  wr_a, "wr_b":  wr_b,
        "er_a":  er_a, "er_b":  er_b,
        "dd_a":  dd_a, "dd_b":  dd_b,
        "nt_a":  nt_a, "nt_b":  nt_b,
    }


# =============================================================================
# REPORT
# =============================================================================

def print_r009_report(symbol_results: list,
                      agg_a: dict, agg_b: dict,
                      filter_stats: dict,
                      verdict_info: dict) -> None:
    S  = "=" * 108
    S2 = "─" * 108
    BL = "  "

    print(f"\n{S}")
    print(f"{BL}QUANTLAB AI — RESEARCH #009")
    print(f"{BL}Session-Time + Relative-Volume Filter for Liquidity Sweep Reversal")
    print(f"{BL}{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(S)

    print(f"""
{BL}╔══════════════════════════════════════════════════════════════════════╗
{BL}║  HYPOTHESIS                                                          ║
{BL}║                                                                      ║
{BL}║  LSR signals during London + New York sessions (08–21 UTC) with      ║
{BL}║  above-average relative volume (rel_vol ≥ {REL_VOL_THRESHOLD}) are            ║
{BL}║  more reliable than signals at any hour and any volume.              ║
{BL}║                                                                      ║
{BL}║  Filter (Strategy B additions):                                      ║
{BL}║    hour_utc ∈ [{SESSION_START}, {SESSION_END}]     — London + New York session      ║
{BL}║    rel_vol  ≥ {REL_VOL_THRESHOLD}        — above-average participation            ║
{BL}║                                                                      ║
{BL}║  Both thresholds are pre-specified round numbers, NOT optimised.     ║
{BL}╚══════════════════════════════════════════════════════════════════════╝
""")

    # Filter pass rate
    print(f"{BL}FILTER PASS RATE  (Strategy A signals → Strategy B signals)")
    print(f"{BL}{'Symbol':<24} {'A Signals':>11} {'B Signals':>11} "
          f"{'Pass Rate':>11} {'Blocked':>8}")
    print(f"{BL}{S2[2:70]}")
    ta, tb = 0, 0
    for r in symbol_results:
        s  = r["symbol"]
        fs = filter_stats[s]
        ta += fs["n_base"];  tb += fs["n_pass"]
        print(f"{BL}{s:<24} {fs['n_base']:>11} {fs['n_pass']:>11} "
              f"{fs['pass_rate']:>11.1%} {fs['n_base']-fs['n_pass']:>8}")
    total_rate = tb / ta if ta > 0 else 0.0
    print(f"{BL}{'TOTAL':<24} {ta:>11} {tb:>11} "
          f"{total_rate:>11.1%} {ta-tb:>8}")

    # Per-symbol
    metric_display = [
        ("n_trades",         "Trades",           lambda v: str(int(v))),
        ("win_rate",         "Win Rate",          lambda v: f"{v:.1%}"),
        ("profit_factor",    "Profit Factor",     lambda v: f"{v:.3f}"),
        ("expectancy_r",     "Expectancy (R)",    lambda v: f"{v:+.3f}"),
        ("net_profit",       "Net Profit ($)",    lambda v: f"${v:+,.0f}"),
        ("max_drawdown",     "Max Drawdown",      lambda v: f"{v:.1%}"),
        ("sharpe",           "Sharpe",            lambda v: f"{v:.3f}"),
        ("avg_hold_minutes", "Avg Hold (min)",    lambda v: f"{v:.0f}"),
    ]

    for r in symbol_results:
        sym = r["symbol"]
        ma  = r["A"]["m"];  mb  = r["B"]["m"]
        mc_a = r["A"]["mc"]; mc_b = r["B"]["mc"]

        print(f"\n{BL}{'─'*70}")
        print(f"{BL}{sym}")
        print(f"{BL}{'─'*70}")
        print(f"{BL}{'Metric':<26} {'Strategy A':>14} {'Strategy B':>14}  {'Δ (B−A)':>12}  Better?")
        print(f"{BL}{'─'*80}")

        for key, lbl, fmt in metric_display:
            va_v = ma.get(key, 0.0)
            vb_v = mb.get(key, 0.0)
            delta = vb_v - va_v
            if key == "n_trades":
                better = "—"
            else:
                better = "B ✓" if delta > 0 else ("=" if delta == 0 else "A")
            print(f"{BL}  {lbl:<24} {fmt(va_v):>14} {fmt(vb_v):>14}  "
                  f"{fmt(delta):>12}  {better}")
        print(f"{BL}  {'MC Prob Profit':<24} {mc_a['prob_profit']:>14.1%} "
              f"{mc_b['prob_profit']:>14.1%}  "
              f"{'':>12}  {'B ✓' if mc_b['prob_profit'] > mc_a['prob_profit'] else 'A'}")

    # Aggregate
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
        if key == "n_trades":
            better = "—"
        else:
            better = "B ✓" if delta > 0 else ("=" if delta == 0 else "A")
        print(f"{BL}  {lbl:<24} {fmt(va_v):>14} {fmt(vb_v):>14}  "
              f"{fmt(delta):>12}  {better}")
    print(f"{BL}  {'MC Prob Profit':<24} {mc_agg_a['prob_profit']:>14.1%} "
          f"{mc_agg_b['prob_profit']:>14.1%}  {'':>12}  "
          f"{'B ✓' if mc_agg_b['prob_profit'] > mc_agg_a['prob_profit'] else 'A'}")

    # Verdict
    vi  = verdict_info
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

    pf_delta_pct = (vi["pf_b"] / vi["pf_a"] - 1.0) * 100 if vi["pf_a"] > 0 else 0.0

    print(f"\n{S}")
    print(f"{BL}OBJECTIVE VERDICT CRITERIA")
    print(f"{BL}{S2[2:]}")
    print(f"\n{BL}  [{c1s}] PF improvement     A={vi['pf_a']:.3f} → B={vi['pf_b']:.3f}"
          f"  Δ={vi['pf_delta']:+.3f}  ({pf_delta_pct:+.1f}%)")
    print(f"{BL}  [{c2s}] WR or ExpR improve  WR A={vi['wr_a']:.1%}→B={vi['wr_b']:.1%}"
          f"  ExpR A={vi['er_a']:+.3f}→B={vi['er_b']:+.3f}")
    print(f"{BL}  [{c3s}] Drawdown reduces    A={vi['dd_a']:.1%} → B={vi['dd_b']:.1%}")
    print(f"{BL}  [{c4s}] Cross-symbol consistency  {vi['n_consistent']}/{vi['n_sym']} symbols agree")
    print(f"{BL}  [{c5s}] Trade count ≥25%    A={vi['nt_a']} → B={vi['nt_b']}"
          f"  ({vi['trade_retention']:.0%} retained)")

    print(f"\n{BL}{box}")
    print(f"{BL}{row}")
    print(f"{BL}{v_r}")
    print(f"{BL}{end}")

    if v == "PROMOTE":
        print(f"\n{BL}  All criteria met.  Session-Time + Volume hypothesis is CONFIRMED.")
        print(f"{BL}  London+NY hours with elevated volume appear to select cleaner LSR signals.")
        print(f"{BL}  R010 may tighten the session window or layer this with other filters.")
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
        print(f"\n{BL}  Hypothesis REJECTED.  Failed: {', '.join(failed)}.")
        print(f"{BL}  The session-time + volume filter does not reliably improve LSR.")
        print(f"{BL}  R010 (if any) must start from a new hypothesis.")

    print(f"\n{BL}  Engine, fees, spread, slippage, SL, TP, sizing — UNCHANGED.")
    print(S)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              QUANTLAB AI — RESEARCH #009                                      ║
║   Session-Time + Relative-Volume Filter for Liquidity Sweep Reversal          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  Hypothesis:
    LSR signals during London + New York sessions (08–21 UTC)
    with rel_vol ≥ 1.2 are more reliable than all-hours signals.

  Strategy A: Liquidity Sweep Reversal, unchanged.
  Strategy B: Same + [hour_utc ∈ [8, 21]] AND [rel_vol ≥ 1.2]

  Thresholds: pre-specified round numbers.  No optimisation.
  Engine, fees, spread, slippage, split: LOCKED.
""")

    random.seed(42)
    np.random.seed(42)

    # ── 1. Run both strategies ─────────────────────────────────────────────
    print("=" * 70)
    print("  STEP 1: Running Strategy A and B on all symbols")
    print("=" * 70)

    symbol_results = []
    filter_stats   = {}

    for sym in CONFIG["SYMBOLS"]:
        print(f"\n  {sym}")
        r = run_both_strategies(sym)
        symbol_results.append(r)

        fs = filter_analysis(r["df_oos"])
        filter_stats[sym] = fs

        ma = r["A"]["m"]
        mb = r["B"]["m"]
        print(f"    A: n={ma['n_trades']:>3}  WR={ma['win_rate']:.1%}  "
              f"PF={ma['profit_factor']:.3f}  ExpR={ma['expectancy_r']:+.3f}  "
              f"{r['A']['verdict']}")
        print(f"    B: n={mb['n_trades']:>3}  WR={mb['win_rate']:.1%}  "
              f"PF={mb['profit_factor']:.3f}  ExpR={mb['expectancy_r']:+.3f}  "
              f"{r['B']['verdict']}")
        print(f"    Filter pass: {fs['n_pass']}/{fs['n_base']} signals ({fs['pass_rate']:.1%})")

    # ── 2. Aggregate ──────────────────────────────────────────────────────
    print("\n  STEP 2: Aggregate metrics")
    all_trades_a = []
    all_trades_b = []
    for r in symbol_results:
        all_trades_a.extend(r["A"]["trades"])
        all_trades_b.extend(r["B"]["trades"])

    agg_a = aggregate_metrics(all_trades_a, LABEL_A)
    agg_b = aggregate_metrics(all_trades_b, LABEL_B)

    print(f"    A: n={agg_a['n_trades']}  WR={agg_a['win_rate']:.1%}  "
          f"PF={agg_a['profit_factor']:.3f}  ExpR={agg_a['expectancy_r']:+.3f}")
    print(f"    B: n={agg_b['n_trades']}  WR={agg_b['win_rate']:.1%}  "
          f"PF={agg_b['profit_factor']:.3f}  ExpR={agg_b['expectancy_r']:+.3f}")

    # ── 3. Verdict ────────────────────────────────────────────────────────
    print("\n  STEP 3: Verdict")
    verdict_info = compute_verdict(symbol_results, agg_a, agg_b)
    print(f"  → {verdict_info['verdict']}")

    # ── 4. Charts ─────────────────────────────────────────────────────────
    print("\n  STEP 4: Generating charts")
    charts = []

    for fn, label in [
        (plot_metric_grid,         "Metric grid"),
        (plot_equity_curves,       "Equity curves"),
        (plot_drawdown,            "Drawdown"),
        (plot_monte_carlo,         "Monte Carlo"),
        (plot_ab_delta_summary,    "Delta summary"),
    ]:
        p = fn(symbol_results, agg_a, agg_b)
        charts.append(p);  print(f"  → {p}")

    p = plot_filter_breakdown(symbol_results, filter_stats)
    charts.append(p);  print(f"  → {p}")

    p = plot_session_analysis(symbol_results)
    if p:
        charts.append(p);  print(f"  → {p}")

    p = plot_rel_vol_distribution(symbol_results)
    if p:
        charts.append(p);  print(f"  → {p}")

    # ── 5. Report ─────────────────────────────────────────────────────────
    print_r009_report(symbol_results, agg_a, agg_b, filter_stats, verdict_info)

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
    print("  Research #009 complete.\n")


if __name__ == "__main__":
    main()
