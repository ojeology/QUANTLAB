"""
=============================================================================
QUANTLAB AI – RESEARCH #013
BTC Lead Signal — Cross-Asset Momentum

Objective:
  Test whether BTC's 1H momentum is a leading indicator for ETH and SOL.
  When BTC closes a strong bullish candle above its 200-EMA, enter ETH and
  SOL long on the *following* candle.

Hypothesis:
  BTC is the dominant asset in the crypto market.  Strong BTC momentum
  (above-trend, meaningfully bullish candle) tends to pull altcoins up with
  a 1-candle lag as capital rotates and correlated hedges unwind.

Setups tested:
  1. BTC.Self   — BTC momentum signal traded on BTC itself (control baseline)
  2. ETH.Cross  — BTC momentum signal, entered on ETH next candle
  3. SOL.Cross  — BTC momentum signal, entered on SOL next candle

BTC Momentum Signal (fires on bar i; entry on bar i+1 open):
  A. close[i] > ema200[i]                         — uptrend confirmed
  B. close[i] > open[i]                           — bullish candle
  C. (close[i] - open[i]) / open[i] >= BTC_MIN_BODY   — meaningful move
  D. atr[i] > 0                                   — valid ATR (warmup guard)

Locked: engine, fees, spread, slippage, SL (prev bar low), TP (2R), sizing.
        No parameters optimised.
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

RESEARCH_ID   = "R013"
OUTPUT_FOLDER = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BG = "#0F1117"

# ── BTC signal threshold (industry intuition: 0.5% body = meaningful candle) ─
BTC_MIN_BODY = 0.005   # 0.5% — NOT optimised; first reasonable value tested


# =============================================================================
# STEP 1 — BTC SIGNAL FUNCTION
# =============================================================================

def btc_momentum_signal(df_btc: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series aligned to df_btc index.
    True on bar i means: enter the target asset at bar i+1 open.

    Conditions (all evaluated on bar i, no look-ahead):
      A. close > ema200       — confirmed uptrend
      B. close > open         — bullish candle
      C. body_pct >= BTC_MIN_BODY  — meaningful move (not just drift)
      D. atr > 0              — indicators are warmed up
    """
    body_pct = (df_btc["close"] - df_btc["open"]) / df_btc["open"]

    trend   = df_btc["close"] > df_btc["ema200"]
    bullish = df_btc["close"] > df_btc["open"]
    strong  = body_pct >= BTC_MIN_BODY
    valid   = df_btc["atr"] > 0

    return trend & bullish & strong & valid


# =============================================================================
# STEP 2 — CROSS-ASSET SIGNAL INJECTION
# =============================================================================

def inject_btc_signal(df_alt: pd.DataFrame, btc_signal: pd.Series) -> pd.DataFrame:
    """
    Merges BTC's momentum signal (shifted 1 bar forward) into an altcoin df.

    df_alt must have a 'datetime' column.
    btc_signal must be indexed by the BTC OOS df's integer index, so we
    use the datetime alignment to be safe.
    """
    df = df_alt.copy()
    # Build a lookup: datetime → btc signal (boolean)
    btc_sig_df = btc_signal.rename("btc_sig").to_frame()
    btc_sig_df["datetime"] = df_alt["datetime"].values  # same calendar dates
    # Normalise tz: strip timezone so merge keys are compatible
    if hasattr(btc_sig_df["datetime"].dtype, "tz") and btc_sig_df["datetime"].dt.tz is not None:
        btc_sig_df["datetime"] = btc_sig_df["datetime"].dt.tz_localize(None)
    if hasattr(df["datetime"].dtype, "tz") and df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)

    # Shift: BTC signal on bar i → alt entry on bar i+1
    btc_sig_df["btc_sig_lag1"] = btc_sig_df["btc_sig"].shift(1).fillna(False)

    # Merge on datetime (safe even if lengths diverge slightly)
    df = df.merge(
        btc_sig_df[["datetime", "btc_sig_lag1"]],
        on="datetime", how="left"
    )
    df["btc_sig_lag1"] = df["btc_sig_lag1"].fillna(False)
    return df


# =============================================================================
# STEP 3 — STRATEGY SIGNAL WRAPPERS FOR run_backtest
# =============================================================================

def strategy_btc_self(df: pd.DataFrame) -> pd.Series:
    """BTC.Self — trade BTC when its own momentum signal fires."""
    return btc_momentum_signal(df)


def strategy_eth_cross(df: pd.DataFrame) -> pd.Series:
    """ETH.Cross — enter ETH when BTC's lag-1 momentum signal is True."""
    return df["btc_sig_lag1"].astype(bool)


def strategy_sol_cross(df: pd.DataFrame) -> pd.Series:
    """SOL.Cross — enter SOL when BTC's lag-1 momentum signal is True."""
    return df["btc_sig_lag1"].astype(bool)


# =============================================================================
# STEP 4 — DATA PIPELINE
# =============================================================================

PROMOTE_CRITERIA = {
    "min_n_trades":      10,
    "min_profit_factor": 1.30,
    "min_win_rate":      0.34,
    "max_drawdown":     -0.25,
    "mc_prob_profit":    0.60,
}


def strategy_verdict(metrics: dict, mc: dict) -> str:
    c = PROMOTE_CRITERIA
    n = metrics["n_trades"]
    if n < c["min_n_trades"]:
        return "INSUFFICIENT"
    fails = []
    if metrics["profit_factor"] < c["min_profit_factor"]:
        fails.append(f"PF={metrics['profit_factor']:.3f}<{c['min_profit_factor']}")
    if metrics["win_rate"] < c["min_win_rate"]:
        fails.append(f"WR={metrics['win_rate']:.1%}<{c['min_win_rate']:.0%}")
    if metrics["max_drawdown"] < c["max_drawdown"]:
        fails.append(f"MDD={metrics['max_drawdown']:.1%}<{c['max_drawdown']:.0%}")
    if mc["prob_profit"] < c["mc_prob_profit"]:
        fails.append(f"MC={mc['prob_profit']:.1%}<{c['mc_prob_profit']:.0%}")
    if fails:
        return "REJECT"
    return "PROMOTE"


def load_and_split(symbol: str):
    """Load symbol data, add indicators, return full df + OOS slice + split_idx."""
    df = get_data(symbol)
    n  = len(df)
    df = add_indicators(df)
    split = int(n * CONFIG["TRAIN_RATIO"])
    df_oos = df.iloc[split:].reset_index(drop=True)
    return df, df_oos, split


# =============================================================================
# STEP 5 — PLOTTING HELPERS
# =============================================================================

C_SETUP = {
    "BTC.Self":  "#F7931A",   # BTC orange
    "ETH.Cross": "#627EEA",   # ETH purple-blue
    "SOL.Cross": "#9945FF",   # SOL purple
}

def _ax_style(ax):
    ax.set_facecolor("#1A1D24")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")


def plot_equity_curves(results: dict, path: str):
    """One equity curve per setup."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=BG)
    fig.suptitle("R013 — Equity Curves (OOS)", color="white", fontsize=12, y=1.01)

    start = CONFIG["STARTING_CAPITAL"]

    for ax, (name, data) in zip(axes, results.items()):
        _ax_style(ax)
        trades = data["trades"]
        if not trades:
            ax.text(0.5, 0.5, "No trades", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=10)
            ax.set_title(name, fontsize=10, color="white")
            continue

        equity = [start]
        for t in trades:
            equity.append(equity[-1] + t["pnl"])

        xs = range(len(equity))
        color = C_SETUP.get(name, "#4A90D9")
        ax.plot(xs, equity, color=color, lw=1.5)
        ax.fill_between(xs, equity, start, where=[e >= start for e in equity],
                        alpha=0.15, color=color)
        ax.fill_between(xs, equity, start, where=[e < start for e in equity],
                        alpha=0.15, color="#FF4560")
        ax.axhline(start, color="#888", lw=0.8, ls="--")

        m = data["m"]
        v = data["verdict"]
        ax.set_title(
            f"{name}  [{v}]\nPF={m['profit_factor']:.3f}  WR={m['win_rate']:.1%}  n={m['n_trades']}",
            fontsize=9, color="white",
        )
        ax.set_xlabel("Trade #", color="white", fontsize=8)
        ax.set_ylabel("Equity ($)", color="white", fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_mc(results: dict, path: str):
    """Monte Carlo histogram grid — one panel per setup."""
    start = CONFIG["STARTING_CAPITAL"]
    n_setups = len(results)
    fig, axes = plt.subplots(1, n_setups, figsize=(5 * n_setups, 4), facecolor=BG)
    if n_setups == 1:
        axes = [axes]
    fig.suptitle("R013 — Monte Carlo Final Equity Distribution", color="white",
                 fontsize=12, y=1.01)

    for ax, (name, data) in zip(axes, results.items()):
        _ax_style(ax)
        fe = data["mc"]["final_equities"]
        fe_min, fe_max = float(np.min(fe)), float(np.max(fe))

        if (fe_max - fe_min) < 1e-6 or len(np.unique(fe)) < 3:
            ax.text(0.5, 0.5,
                    f"Insufficient trades\nfor MC histogram\n(n={data['m']['n_trades']})",
                    transform=ax.transAxes, ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
        else:
            bns = min(max(5, len(np.unique(fe))), 50)
            color = C_SETUP.get(name, "#4A90D9")
            ax.hist(fe, bins=bns, color=color, alpha=0.75, density=True)
            ax.axvline(start, color="#FF4560", lw=1.5, ls="--", label="Start")
            ax.axvline(data["mc"]["median"], color="#FFD700", lw=1.2, ls=":",
                       label=f"Median={data['mc']['median']:,.0f}")
            ax.legend(fontsize=7.5, facecolor="#1A1D24", edgecolor="#444",
                      labelcolor="white")

        m = data["m"]
        ax.set_title(
            f"{name}  [{data['verdict']}]\n"
            f"PP={data['mc']['prob_profit']:.1%}  PF={m['profit_factor']:.3f}",
            fontsize=9, color="white",
        )
        ax.set_xlabel("Final Equity ($)", color="white", fontsize=8)
        ax.set_ylabel("Density", color="white", fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_signal_stats(df_btc_oos: pd.DataFrame, btc_sig: pd.Series, path: str):
    """Show BTC signal frequency and body size distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)
    fig.suptitle("R013 — BTC Momentum Signal Characterisation (OOS)",
                 color="white", fontsize=12, y=1.02)

    # Panel 1: signal frequency by month
    ax = axes[0]
    _ax_style(ax)
    sig_df = df_btc_oos[["datetime"]].copy()
    sig_df["signal"] = btc_sig.values
    sig_df["month"]  = sig_df["datetime"].dt.to_period("M").astype(str)
    monthly = sig_df.groupby("month")["signal"].sum()
    monthly_total = sig_df.groupby("month")["signal"].count()
    monthly_pct = (monthly / monthly_total * 100).fillna(0)

    xs = range(len(monthly_pct))
    ax.bar(xs, monthly_pct.values, color="#F7931A", alpha=0.8, width=0.7)
    ax.set_xticks(list(xs)[::2])
    ax.set_xticklabels(list(monthly_pct.index)[::2], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Signal Rate (%)", color="white", fontsize=8)
    ax.set_title(f"BTC Signal Frequency by Month\n(≥{BTC_MIN_BODY:.1%} body, above EMA200)",
                 fontsize=9, color="white")
    ax.axhline(monthly_pct.mean(), color="#FFD700", lw=1.2, ls="--",
               label=f"Avg {monthly_pct.mean():.1f}%")
    ax.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    # Panel 2: body size distribution (all uptrend bullish candles)
    ax2 = axes[1]
    _ax_style(ax2)
    uptrend_bull = df_btc_oos[
        (df_btc_oos["close"] > df_btc_oos["ema200"]) &
        (df_btc_oos["close"] > df_btc_oos["open"])
    ].copy()
    body_pcts = ((uptrend_bull["close"] - uptrend_bull["open"]) / uptrend_bull["open"] * 100)
    ax2.hist(body_pcts.clip(0, 3), bins=40, color="#627EEA", alpha=0.8, density=True)
    ax2.axvline(BTC_MIN_BODY * 100, color="#FF4560", lw=1.5, ls="--",
                label=f"Threshold {BTC_MIN_BODY:.1%}")
    pct_above = (body_pcts >= BTC_MIN_BODY * 100).mean() * 100
    ax2.set_xlabel("Candle Body Size (%)", color="white", fontsize=8)
    ax2.set_ylabel("Density", color="white", fontsize=8)
    ax2.set_title(f"Bullish Candle Body Distribution (OOS)\n"
                  f"{pct_above:.1f}% of up-candles qualify",
                  fontsize=9, color="white")
    ax2.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_r_distribution(results: dict, path: str):
    """R-multiple distribution for each setup."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), facecolor=BG)
    if n == 1:
        axes = [axes]
    fig.suptitle("R013 — R-Multiple Distribution", color="white", fontsize=12, y=1.01)

    for ax, (name, data) in zip(axes, results.items()):
        _ax_style(ax)
        trades = data["trades"]
        if len(trades) < 2:
            ax.text(0.5, 0.5, "Insufficient trades", transform=ax.transAxes,
                    ha="center", va="center", color="white", fontsize=10)
            ax.set_title(name, fontsize=10, color="white")
            continue
        rmults = [t["r_multiple"] for t in trades]
        color  = C_SETUP.get(name, "#4A90D9")
        ax.hist(rmults, bins=min(20, len(rmults)), color=color, alpha=0.8)
        ax.axvline(0, color="#FF4560", lw=1.5, ls="--")
        ax.axvline(np.mean(rmults), color="#FFD700", lw=1.2, ls=":",
                   label=f"Mean {np.mean(rmults):.2f}R")
        m = data["m"]
        ax.set_title(
            f"{name}  [{data['verdict']}]\n"
            f"ExpR={m['expectancy_r']:+.3f}  WR={m['win_rate']:.1%}",
            fontsize=9, color="white",
        )
        ax.set_xlabel("R Multiple", color="white", fontsize=8)
        ax.set_ylabel("Count", color="white", fontsize=8)
        ax.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


# =============================================================================
# STEP 6 — REPORT
# =============================================================================

def print_report(results: dict, btc_sig_count: int, total_oos_bars: int):
    W = 108
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("=" * W)
    print(f"  QUANTLAB AI — RESEARCH #013")
    print(f"  BTC Lead Signal — Cross-Asset Momentum")
    print(f"  {now}")
    print("=" * W)
    print()
    print(f"  BTC Momentum Signal: close > EMA200 AND body ≥ {BTC_MIN_BODY:.1%}")
    print(f"  OOS bars: {total_oos_bars:,}  |  BTC signal fires: {btc_sig_count:,} "
          f"({btc_sig_count/total_oos_bars:.1%} of bars)")
    print()
    print(f"  Setups:")
    print(f"    BTC.Self  — signal fires on BTC → enter BTC next bar (control)")
    print(f"    ETH.Cross — signal fires on BTC → enter ETH next bar")
    print(f"    SOL.Cross — signal fires on BTC → enter SOL next bar")
    print()
    print(f"  PROMOTE criteria: PF≥1.30 | WR≥34% | MDD≥-25% | MC PP≥60% | n≥10")
    print()

    hdr = f"  {'Metric':<26} {'BTC.Self':>16} {'ETH.Cross':>16} {'SOL.Cross':>16}"
    sep = "  " + "─" * 26 + " " + "─" * 16 + " " + "─" * 16 + " " + "─" * 16
    print(sep)
    print(hdr)
    print(sep)

    rows = [
        ("Trades",        lambda m, mc: f"{m['n_trades']}"),
        ("Win Rate",      lambda m, mc: f"{m['win_rate']:.1%}"),
        ("Profit Factor", lambda m, mc: f"{m['profit_factor']:.3f}"),
        ("Expectancy R",  lambda m, mc: f"{m['expectancy_r']:+.3f}"),
        ("Net P&L ($)",   lambda m, mc: f"${m['net_pnl']:>+,.0f}"),
        ("Max Drawdown",  lambda m, mc: f"{m['max_drawdown']:.1%}"),
        ("Sharpe",        lambda m, mc: f"{m['sharpe_ratio']:.3f}"),
        ("MC Prob Profit",lambda m, mc: f"{mc['prob_profit']:.1%}"),
        ("Verdict",       lambda m, mc: results[n]["verdict"]),
    ]

    names = list(results.keys())
    for label, fn in rows:
        vals = []
        for n in names:
            d = results[n]
            try:
                vals.append(fn(d["m"], d["mc"]) if label != "Verdict" else d["verdict"])
            except Exception:
                vals.append("—")
        print(f"  {label:<26} {vals[0]:>16} {vals[1]:>16} {vals[2]:>16}")

    print(sep)
    print()

    # Verdict summary
    promotes = [n for n in names if results[n]["verdict"] == "PROMOTE"]
    print(f"  PROMOTE count: {len(promotes)}/{len(names)}", end="")
    if promotes:
        print(f"  → {', '.join(promotes)}")
    else:
        print()
    print()

    # Interpretation
    btc_v   = results["BTC.Self"]["verdict"]
    eth_v   = results["ETH.Cross"]["verdict"]
    sol_v   = results["SOL.Cross"]["verdict"]
    btc_pf  = results["BTC.Self"]["m"]["profit_factor"]
    eth_pf  = results["ETH.Cross"]["m"]["profit_factor"]
    sol_pf  = results["SOL.Cross"]["m"]["profit_factor"]

    print("  " + "─" * (W - 4))
    print("  INTERPRETATION")
    print("  " + "─" * (W - 4))
    print()

    if btc_v == "PROMOTE":
        print("  ✓ BTC.Self PROMOTES — BTC momentum predicts BTC continuation. Strong foundation.")
    elif btc_v == "INSUFFICIENT":
        print("  △ BTC.Self is INSUFFICIENT — too few trades to evaluate the control.")
    else:
        print(f"  ✗ BTC.Self REJECTS (PF={btc_pf:.3f}) — BTC momentum does not predict BTC continuation.")
        print("    This weakens the entire cross-asset hypothesis.")

    if eth_v == "PROMOTE":
        print("  ✓ ETH.Cross PROMOTES — BTC lead signal works on ETH.")
    elif eth_v == "INSUFFICIENT":
        print("  △ ETH.Cross INSUFFICIENT — same signal count as BTC.Self (by design).")
    else:
        print(f"  ✗ ETH.Cross REJECTS (PF={eth_pf:.3f}) — BTC momentum does not lead ETH reliably.")

    if sol_v == "PROMOTE":
        print("  ✓ SOL.Cross PROMOTES — BTC lead signal works on SOL.")
    elif sol_v == "INSUFFICIENT":
        print("  △ SOL.Cross INSUFFICIENT — same signal count as BTC.Self (by design).")
    else:
        print(f"  ✗ SOL.Cross REJECTS (PF={sol_pf:.3f}) — BTC momentum does not lead SOL reliably.")

    print()
    print("  Suggested next directions for R014:")
    print("    A) Increase BTC body threshold (0.5% → 1.0%) — fewer but stronger signals")
    print("    B) Extend lag window (enter within 2 bars of BTC signal, not just next bar)")
    print("    C) Combine BTC lead signal with altcoin EMA200 filter + pullback entry")
    print("    D) Test SHORT side: BTC bearish momentum → short ETH/SOL")
    print("    E) Funding rate extremes as directional signal (fresh hypothesis)")
    print()
    print("=" * W)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("╔" + "═" * 79 + "╗")
    print("║              QUANTLAB AI — RESEARCH #013" + " " * 38 + "║")
    print("║   BTC Lead Signal — Cross-Asset Momentum" + " " * 38 + "║")
    print("╚" + "═" * 79 + "╝")
    print()
    print("  Hypothesis: BTC 1H momentum (≥0.5% body above EMA200)")
    print("              predicts near-term upside in ETH and SOL (1-bar lag).")
    print()
    print("  Setups: BTC.Self (control) | ETH.Cross | SOL.Cross")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  STEP 1: Loading data and computing BTC signal")
    print("=" * 70)

    # Load BTC (drive the split date from BTC)
    df_btc_full, df_btc_oos, btc_split = load_and_split("BTC-USDT-SWAP")
    oos_start = df_btc_oos["datetime"].iloc[0]
    oos_end   = df_btc_oos["datetime"].iloc[-1]
    print(f"  OOS window: {oos_start.date()} → {oos_end.date()}  "
          f"({len(df_btc_oos):,} bars)")

    # Compute BTC signal on OOS slice
    btc_sig    = btc_momentum_signal(df_btc_oos)
    sig_count  = int(btc_sig.sum())
    print(f"  BTC signal fires: {sig_count:,} times "
          f"({sig_count/len(df_btc_oos):.1%} of OOS bars)")

    # Load and align ETH and SOL using the same calendar cutoff
    df_eth_full, df_eth_oos, _ = load_and_split("ETH-USDT-SWAP")
    df_sol_full, df_sol_oos, _ = load_and_split("SOL-USDT-SWAP")

    # Trim all OOS slices to the BTC OOS window for clean alignment
    def trim_to_window(df_oos, start_dt, end_dt):
        mask = (df_oos["datetime"] >= start_dt) & (df_oos["datetime"] <= end_dt)
        return df_oos[mask].reset_index(drop=True)

    df_btc_oos = trim_to_window(df_btc_oos, oos_start, oos_end)
    df_eth_oos = trim_to_window(df_eth_oos, oos_start, oos_end)
    df_sol_oos = trim_to_window(df_sol_oos, oos_start, oos_end)

    # Recompute BTC signal after trim (should be same, safety re-align)
    btc_sig = btc_momentum_signal(df_btc_oos)
    sig_count = int(btc_sig.sum())
    print(f"  After timestamp alignment:")
    print(f"    BTC: {len(df_btc_oos):,} bars  |  ETH: {len(df_eth_oos):,} bars  "
          f"|  SOL: {len(df_sol_oos):,} bars")

    # Inject BTC signal into ETH and SOL (lag = 1 candle)
    df_eth_oos = inject_btc_signal(df_eth_oos, btc_sig)
    df_sol_oos = inject_btc_signal(df_sol_oos, btc_sig)

    injected_eth = int(df_eth_oos["btc_sig_lag1"].sum())
    injected_sol = int(df_sol_oos["btc_sig_lag1"].sum())
    print(f"  Lag-1 signals injected: ETH={injected_eth}  SOL={injected_sol}")

    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  STEP 2: Running backtests")
    print("=" * 70)

    results = {}

    # BTC.Self — control
    res_btc = run_backtest(df_btc_oos, strategy_btc_self, "BTC.Self")
    m_btc   = compute_metrics(res_btc["trades"], "BTC.Self")
    mc_btc  = monte_carlo(m_btc["pnls"], CONFIG["MC_ITERATIONS"])
    v_btc   = strategy_verdict(m_btc, mc_btc)
    results["BTC.Self"] = {
        "trades": res_btc["trades"], "m": m_btc, "mc": mc_btc, "verdict": v_btc,
    }
    print(f"    BTC.Self    n={m_btc['n_trades']:>4}  WR={m_btc['win_rate']:.1%}  "
          f"PF={m_btc['profit_factor']:.3f}  [{v_btc}]")

    # ETH.Cross
    res_eth = run_backtest(df_eth_oos, strategy_eth_cross, "ETH.Cross")
    m_eth   = compute_metrics(res_eth["trades"], "ETH.Cross")
    mc_eth  = monte_carlo(m_eth["pnls"], CONFIG["MC_ITERATIONS"])
    v_eth   = strategy_verdict(m_eth, mc_eth)
    results["ETH.Cross"] = {
        "trades": res_eth["trades"], "m": m_eth, "mc": mc_eth, "verdict": v_eth,
    }
    print(f"    ETH.Cross   n={m_eth['n_trades']:>4}  WR={m_eth['win_rate']:.1%}  "
          f"PF={m_eth['profit_factor']:.3f}  [{v_eth}]")

    # SOL.Cross
    res_sol = run_backtest(df_sol_oos, strategy_sol_cross, "SOL.Cross")
    m_sol   = compute_metrics(res_sol["trades"], "SOL.Cross")
    mc_sol  = monte_carlo(m_sol["pnls"], CONFIG["MC_ITERATIONS"])
    v_sol   = strategy_verdict(m_sol, mc_sol)
    results["SOL.Cross"] = {
        "trades": res_sol["trades"], "m": m_sol, "mc": mc_sol, "verdict": v_sol,
    }
    print(f"    SOL.Cross   n={m_sol['n_trades']:>4}  WR={m_sol['win_rate']:.1%}  "
          f"PF={m_sol['profit_factor']:.3f}  [{v_sol}]")

    # ─────────────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  STEP 3: Generating charts")
    print("=" * 70)

    plot_equity_curves(results,
        os.path.join(OUTPUT_FOLDER, "r013_equity_curves.png"))

    plot_mc(results,
        os.path.join(OUTPUT_FOLDER, "r013_monte_carlo.png"))

    plot_signal_stats(df_btc_oos, btc_sig,
        os.path.join(OUTPUT_FOLDER, "r013_signal_stats.png"))

    plot_r_distribution(results,
        os.path.join(OUTPUT_FOLDER, "r013_r_distribution.png"))

    # ─────────────────────────────────────────────────────────────────────────
    print()
    print_report(results, sig_count, len(df_btc_oos))

    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  STEP 4: Writing journal")
    print("=" * 70)

    journal_rows = []
    for setup_name, data in results.items():
        m  = data["m"]
        mc = data["mc"]
        v  = data["verdict"]
        journal_rows.append(_journal_row(
            strategy_name = setup_name,
            symbol        = f"CROSS/{setup_name.split('.')[0]}",
            m             = m,
            mc            = mc,
            verdict       = v,
        ))

    append_journal(journal_rows)
    print(f"  Journal updated → {OUTPUT_FOLDER}/research_journal.csv")
    print()
    print(f"  All outputs → {OUTPUT_FOLDER}/")
    print(f"  Research #013 complete.")
    print()


if __name__ == "__main__":
    main()
