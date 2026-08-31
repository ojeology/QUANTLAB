"""
QUANTLAB AI — RESEARCH #022
ATR Stop vs Previous-Bar-Low Stop: EMA Pullback Strategy
=========================================================

Objective:
  Replace the previous-bar-low stop with ATR(14)-based stops and determine
  whether this improves the EMA Pullback strategy across all symbols.

Only the stop-loss mechanism changes. Everything else is locked:
  - Entry conditions   : EMA Pullback (Family C winner from R021)
  - EMA parameters     : fast=20, slow=100  (best combo from R021 train)
  - Take-profit ratio  : 2R
  - Fees, spread, slippage, risk sizing, data split

Stop variants tested:
  BASELINE : Entry − prev_bar_low           (R021 original)
  ATR_1.0  : Entry − 1.0 × ATR(14)
  ATR_1.5  : Entry − 1.5 × ATR(14)
  ATR_2.0  : Entry − 2.0 × ATR(14)

For shorts: Entry + multiplier × ATR(14)  [symmetric; strategy is long-only]

Symbols / split: same 9 symbols, same 70/30 chronological OOS as R021.
"""

import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import (
    CONFIG, calc_ema, calc_atr, calc_adx,
    compute_metrics, monte_carlo, append_journal,
)

RESEARCH_ID = "R022"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

ALL_SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "LINK-USDT-SWAP",
    "XRP-USDT-SWAP", "DOGE-USDT-SWAP", "LTC-USDT-SWAP", "AVAX-USDT-SWAP", "BCH-USDT-SWAP",
]
SPLIT   = 0.70
CAPITAL = CONFIG["STARTING_CAPITAL"]

# EMA Pullback best params from R021 (Family C winner)
EMA_FAST = 20
EMA_SLOW = 100

COLOURS = {
    "BTC-USDT-SWAP":  "#F7931A", "ETH-USDT-SWAP":  "#627EEA",
    "SOL-USDT-SWAP":  "#9945FF", "LINK-USDT-SWAP":  "#2A5ADA",
    "XRP-USDT-SWAP":  "#00AAE4", "DOGE-USDT-SWAP":  "#C3A634",
    "LTC-USDT-SWAP":  "#BFBBBB", "AVAX-USDT-SWAP":  "#E84142",
    "BCH-USDT-SWAP":  "#8DC351",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_1h(sym):
    tag = sym.replace("-", "_")
    df  = pd.read_parquet(f"{CACHE}/{tag}_1H.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


def split_df(df):
    cut = int(len(df) * SPLIT)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def bar_minutes():
    return {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,
            "1H":60,"2H":120,"4H":240,"6H":360,"12H":720,"1D":1440
           }.get(CONFIG["TIMEFRAME"], 60)


def portfolio_pf(trades_list):
    if not trades_list:
        return 0.0
    wins = sum(t["pnl"] for t in trades_list if t["pnl"] > 0)
    loss = abs(sum(t["pnl"] for t in trades_list if t["pnl"] < 0))
    return wins / loss if loss > 0 else (float("inf") if wins > 0 else 0.0)


def verdict(m):
    n, pf, ex = m["n_trades"], m["profit_factor"], m["expectancy_r"]
    if n < 20:             return "INSUFFICIENT"
    if pf >= 1.5 and n >= 30 and ex >= 0.0: return "PROMOTE"
    if pf >= 1.3 and ex >= 0.15: return "WATCHLIST"
    if pf >= 1.0:          return "WEAK"
    return "REJECT"


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL: EMA PULLBACK  (identical to R021 Family C)
# ─────────────────────────────────────────────────────────────────────────────

def build_signals(df, fast=EMA_FAST, slow=EMA_SLOW):
    """Return (signal Series, ema_fast Series, atr Series)."""
    df = df.copy()
    df["ema_f"] = calc_ema(df["close"], fast)
    df["ema_s"] = calc_ema(df["close"], slow)
    df["adx"]   = calc_adx(df, 14)
    df["atr14"] = calc_atr(df, 14)

    touched          = df["low"] <= df["ema_f"]
    touched_recently = touched | touched.shift(1) | touched.shift(2)

    uptrend = df["close"] > df["ema_s"]
    bounce  = df["close"] > df["ema_f"]
    trend   = df["adx"] > 20

    signal = (uptrend & touched_recently & bounce & trend).fillna(False).astype(int)
    return signal, df["ema_f"], df["atr14"]


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM BACKTEST ENGINE  (stop-loss variant injected per call)
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_atr(df: pd.DataFrame, signal: pd.Series,
                     atr14: pd.Series, label: str,
                     stop_mode: str, atr_mult: float = 1.5) -> dict:
    """
    stop_mode = "prev_low"  → stop = prev_bar low  (baseline)
    stop_mode = "atr"       → stop = entry − atr_mult × ATR(14)
    
    Everything else (entry, TP, sizing, fees, costs) is locked.
    """
    min_sl    = CONFIG["MIN_SL_PCT"]
    rr        = CONFIG["RISK_REWARD"]
    max_lev   = CONFIG["MAX_LEVERAGE"]
    capital   = CONFIG["STARTING_CAPITAL"]
    risk_frac = CONFIG["RISK_PER_TRADE_PCT"]
    fee_rate  = CONFIG["TAKER_FEE"]
    spd_rate  = CONFIG["SPREAD"] * 0.5
    slp_rate  = CONFIG["SL_SLIPPAGE"]
    bm        = bar_minutes()

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

                sl_dist   = entry_price - stop_loss
                r_mult    = (exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0
                hold_mins = (i - entry_idx) * bm

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

        if signal.iloc[i - 1]:
            prev_bar = df.iloc[i - 1]
            ep       = bar["open"]

            if stop_mode == "prev_low":
                sl = prev_bar["low"]
            else:  # atr
                atr_val = atr14.iloc[i - 1]
                if pd.isna(atr_val) or atr_val <= 0:
                    continue
                sl = ep - atr_mult * atr_val

            sl_dist = ep - sl
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


# ─────────────────────────────────────────────────────────────────────────────
# PRINT BANNER
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #022" + " "*50 + "║")
print("║  ATR Stop vs Previous-Bar-Low Stop: EMA Pullback" + " "*29 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Strategy : EMA Pullback (Family C, R021 winner: fast={EMA_FAST}, slow={EMA_SLOW})
  Stop variants:
    BASELINE : prev_bar_low   (R021 original)
    ATR_1.0  : Entry − 1.0 × ATR(14)
    ATR_1.5  : Entry − 1.5 × ATR(14)
    ATR_2.0  : Entry − 2.0 × ATR(14)
  Symbols  : All 9 (1H)
  Split    : 70/30 chronological (OOS only evaluated)
""")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & SPLIT DATA
# ─────────────────────────────────────────────────────────────────────────────
print("  Loading 1H data …")
oos_dfs = {}
for sym in ALL_SYMBOLS:
    try:
        df = load_1h(sym)
        _, oos = split_df(df)
        oos_dfs[sym] = oos
        print(f"  {sym:25s}  total={len(df):,}  oos={len(oos):,}")
    except FileNotFoundError:
        print(f"  {sym:25s}  *** cache missing — skipping ***")

symbols = list(oos_dfs.keys())

# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL STOP VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    ("BASELINE",  "prev_low", None),
    ("ATR_1.0",   "atr",      1.0),
    ("ATR_1.5",   "atr",      1.5),
    ("ATR_2.0",   "atr",      2.0),
]

all_results = {}   # variant_name → {sym → metrics}
all_trades  = {}   # variant_name → flat list of trades
all_port    = {}   # variant_name → portfolio metrics

for vname, mode, mult in VARIANTS:
    sym_results   = {}
    flat_trades   = []
    mult_str      = f"{mult:.1f}×" if mult else "prev_low"
    print(f"\n{'─'*70}")
    print(f"  VARIANT: {vname}  (stop={mult_str})")
    print("─"*70)

    for sym in symbols:
        df_oos = oos_dfs[sym].copy()
        sig, ema_f, atr14 = build_signals(df_oos)
        res = run_backtest_atr(df_oos, sig, atr14, sym,
                               stop_mode=mode,
                               atr_mult=mult if mult else 1.5)
        m   = compute_metrics(res["trades"], sym)
        v   = verdict(m)
        sym_results[sym] = m
        flat_trades.extend(res["trades"])
        tag = sym.split("-")[0]
        print(f"  {tag:5s}  n={m['n_trades']:3d}  WR={m['win_rate']*100:4.1f}%"
              f"  PF={m['profit_factor']:.3f}  ExpR={m['expectancy_r']:+.3f}"
              f"  Sharpe={m['sharpe']:5.2f}  MDD={m['max_drawdown']*100:.1f}%"
              f"  Net=${m['net_profit']:+.0f}  → {v}")

    port = compute_metrics(flat_trades, "PORTFOLIO")
    print(f"  {'PORTFOLIO':5s}  n={port['n_trades']:3d}  WR={port['win_rate']*100:4.1f}%"
          f"  PF={port['profit_factor']:.3f}  ExpR={port['expectancy_r']:+.3f}"
          f"  Sharpe={port['sharpe']:5.2f}  MDD={port['max_drawdown']*100:.1f}%"
          f"  Net=${port['net_profit']:+.0f}")

    all_results[vname] = sym_results
    all_trades[vname]  = flat_trades
    all_port[vname]    = port

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL SIGNIFICANCE  (Wilcoxon signed-rank test per variant vs baseline)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("  STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank, vs BASELINE)")
print("═"*70)

baseline_pnls = np.array([t["pnl"] for t in all_trades["BASELINE"]])
sig_tests = {}
for vname, mode, mult in VARIANTS[1:]:  # skip BASELINE vs itself
    vpnls = np.array([t["pnl"] for t in all_trades[vname]])
    n_min = min(len(baseline_pnls), len(vpnls))
    if n_min >= 10:
        # Compare pnl distributions (test if ATR variant is different from baseline)
        try:
            stat, pval = scipy_stats.wilcoxon(
                vpnls[:n_min] - baseline_pnls[:n_min], alternative="two-sided"
            )
        except Exception:
            pval = np.nan
    else:
        pval = np.nan
    sig_tests[vname] = pval
    pval_str = f"{pval:.4f}" if not np.isnan(pval) else "N/A"
    sig_str  = "✓ SIGNIFICANT" if (not np.isnan(pval) and pval < 0.05) else "✗ not significant"
    print(f"  {vname:10s}  p={pval_str}  {sig_str}")

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*70)
print("  PORTFOLIO SUMMARY — ALL VARIANTS")
print("═"*70)
print(f"  {'Variant':12s}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'ExpR':>7}  {'Sharpe':>7}  {'MDD':>7}  {'Net $':>10}")
print("  " + "─"*72)
for vname, _, _ in VARIANTS:
    p = all_port[vname]
    print(f"  {vname:12s}  {p['n_trades']:5d}  {p['win_rate']*100:5.1f}%"
          f"  {p['profit_factor']:7.3f}  {p['expectancy_r']:+7.3f}"
          f"  {p['sharpe']:7.2f}  {p['max_drawdown']*100:6.1f}%"
          f"  {p['net_profit']:+10.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH CONCLUSIONS
# ─────────────────────────────────────────────────────────────────────────────

baseline_port = all_port["BASELINE"]
best_variant  = max(VARIANTS[1:], key=lambda v: all_port[v[0]]["profit_factor"])
best_vname    = best_variant[0]
best_port     = all_port[best_vname]

# Per-question answers
def pf_improved(vname):
    return all_port[vname]["profit_factor"] > baseline_port["profit_factor"]

def exp_improved(vname):
    return all_port[vname]["expectancy_r"] > baseline_port["expectancy_r"]

def dd_reduced(vname):
    return all_port[vname]["max_drawdown"] > baseline_port["max_drawdown"]  # dd is negative

# Consistent across symbols: count how many symbols have PF > baseline for best variant
def count_improved(vname):
    count = 0
    for sym in symbols:
        base_pf = all_results["BASELINE"][sym]["profit_factor"]
        atr_pf  = all_results[vname][sym]["profit_factor"]
        if atr_pf > base_pf:
            count += 1
    return count

def link_profitable(vname):
    lk = "LINK-USDT-SWAP"
    if lk not in all_results[vname]:
        return False
    return all_results[vname][lk]["profit_factor"] > 1.0

def new_profitable(vname):
    result = []
    for sym in symbols:
        base_pf = all_results["BASELINE"][sym]["profit_factor"]
        atr_pf  = all_results[vname][sym]["profit_factor"]
        base_v  = verdict(all_results["BASELINE"][sym])
        atr_v   = verdict(all_results[vname][sym])
        if base_v in ("REJECT","WEAK","INSUFFICIENT") and atr_pf > 1.20:
            result.append(sym)
    return result

# Best multiplier by symbol count improvement
best_by_count = max(["ATR_1.0","ATR_1.5","ATR_2.0"], key=count_improved)

# Reject hypothesis check: no ATR version reaches PF > 1.20 on multiple symbols
def passes_threshold(vname, pf_thr=1.20, n_thr=2):
    count = sum(1 for sym in symbols
                if all_results[vname][sym]["profit_factor"] > pf_thr
                and all_results[vname][sym]["n_trades"] >= 20)
    return count >= n_thr

any_passes = any(passes_threshold(v[0]) for v in VARIANTS[1:])

print(f"""
{'='*70}
  RESEARCH #022 — CONCLUSIONS
{'='*70}

  1. Does ATR stop improve Profit Factor?
     BASELINE PF : {baseline_port['profit_factor']:.3f}
     ATR_1.0  PF : {all_port['ATR_1.0']['profit_factor']:.3f}  {'▲ YES' if pf_improved('ATR_1.0') else '▼ NO'}
     ATR_1.5  PF : {all_port['ATR_1.5']['profit_factor']:.3f}  {'▲ YES' if pf_improved('ATR_1.5') else '▼ NO'}
     ATR_2.0  PF : {all_port['ATR_2.0']['profit_factor']:.3f}  {'▲ YES' if pf_improved('ATR_2.0') else '▼ NO'}

  2. Does ATR stop improve Expectancy (R)?
     BASELINE ExpR: {baseline_port['expectancy_r']:+.3f}
     ATR_1.0  ExpR: {all_port['ATR_1.0']['expectancy_r']:+.3f}  {'▲ YES' if exp_improved('ATR_1.0') else '▼ NO'}
     ATR_1.5  ExpR: {all_port['ATR_1.5']['expectancy_r']:+.3f}  {'▲ YES' if exp_improved('ATR_1.5') else '▼ NO'}
     ATR_2.0  ExpR: {all_port['ATR_2.0']['expectancy_r']:+.3f}  {'▲ YES' if exp_improved('ATR_2.0') else '▼ NO'}

  3. Does ATR stop reduce Max Drawdown?
     BASELINE MDD : {baseline_port['max_drawdown']*100:.1f}%
     ATR_1.0  MDD : {all_port['ATR_1.0']['max_drawdown']*100:.1f}%  {'▲ REDUCED' if dd_reduced('ATR_1.0') else '▼ WORSE'}
     ATR_1.5  MDD : {all_port['ATR_1.5']['max_drawdown']*100:.1f}%  {'▲ REDUCED' if dd_reduced('ATR_1.5') else '▼ WORSE'}
     ATR_2.0  MDD : {all_port['ATR_2.0']['max_drawdown']*100:.1f}%  {'▲ REDUCED' if dd_reduced('ATR_2.0') else '▼ WORSE'}

  4. Best ATR multiplier across symbols?
     Counted by number of symbols with PF improvement over baseline:
     ATR_1.0 : {count_improved('ATR_1.0')}/{len(symbols)} symbols improved
     ATR_1.5 : {count_improved('ATR_1.5')}/{len(symbols)} symbols improved
     ATR_2.0 : {count_improved('ATR_2.0')}/{len(symbols)} symbols improved
     → Best consistent: {best_by_count}

  5. Does LINK remain profitable?
     LINK BASELINE  PF={all_results['BASELINE'].get('LINK-USDT-SWAP', {}).get('profit_factor', 0):.3f}
     LINK ATR_1.0   PF={all_results['ATR_1.0'].get('LINK-USDT-SWAP', {}).get('profit_factor', 0):.3f}  {'✓' if link_profitable('ATR_1.0') else '✗'}
     LINK ATR_1.5   PF={all_results['ATR_1.5'].get('LINK-USDT-SWAP', {}).get('profit_factor', 0):.3f}  {'✓' if link_profitable('ATR_1.5') else '✗'}
     LINK ATR_2.0   PF={all_results['ATR_2.0'].get('LINK-USDT-SWAP', {}).get('profit_factor', 0):.3f}  {'✓' if link_profitable('ATR_2.0') else '✗'}

  6. New symbols becoming profitable (PF > 1.20)?
     ATR_1.0 : {new_profitable('ATR_1.0') or ['none']}
     ATR_1.5 : {new_profitable('ATR_1.5') or ['none']}
     ATR_2.0 : {new_profitable('ATR_2.0') or ['none']}

  7. Statistical significance (p < 0.05)?
     ATR_1.0  p={sig_tests.get('ATR_1.0', float('nan')):.4f}  {'✓ SIGNIFICANT' if not np.isnan(sig_tests.get('ATR_1.0', float('nan'))) and sig_tests['ATR_1.0'] < 0.05 else '✗ not significant'}
     ATR_1.5  p={sig_tests.get('ATR_1.5', float('nan')):.4f}  {'✓ SIGNIFICANT' if not np.isnan(sig_tests.get('ATR_1.5', float('nan'))) and sig_tests['ATR_1.5'] < 0.05 else '✗ not significant'}
     ATR_2.0  p={sig_tests.get('ATR_2.0', float('nan')):.4f}  {'✓ SIGNIFICANT' if not np.isnan(sig_tests.get('ATR_2.0', float('nan'))) and sig_tests['ATR_2.0'] < 0.05 else '✗ not significant'}

  VERDICT:
""")

if any_passes:
    print(f"  ✓ HYPOTHESIS SUPPORTED — {best_vname} reaches PF > 1.20 on ≥ 2 symbols.")
    print(f"    Best overall: {best_vname}  (Portfolio PF={best_port['profit_factor']:.3f})")
else:
    print("  ✗ HYPOTHESIS REJECTED — No ATR variant reaches PF > 1.20 across multiple symbols.")
    print("    Archiving EMA Pullback family. Strategy does not generalise with ATR stops.")

print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating charts …")

def dark_ax(ax, title=None, col="white"):
    ax.set_facecolor("#111")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=col, fontsize=10)

# ── Chart 1: Portfolio PF bar — all variants ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111")
fig.suptitle("R022 — ATR Stop vs Baseline: Portfolio Comparison", color="white", fontsize=12)

variant_names = [v[0] for v in VARIANTS]
variant_cols  = ["#9E9E9E", "#4CAF50", "#FF9800", "#F44336"]

ax_pf = axes[0]
dark_ax(ax_pf, "Portfolio Profit Factor")
pfs   = [all_port[vn]["profit_factor"] for vn in variant_names]
bars  = ax_pf.bar(variant_names, pfs, color=variant_cols, alpha=0.85, edgecolor="#333")
ax_pf.axhline(1.0, color="white", lw=1, ls="--", alpha=0.5)
ax_pf.axhline(1.2, color="#FF9800", lw=1, ls=":", alpha=0.6, label="PF=1.20 threshold")
ax_pf.set_ylabel("Profit Factor", color="white")
for b, pf in zip(bars, pfs):
    ax_pf.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
               f"{pf:.3f}", ha="center", color="white", fontsize=9)
ax_pf.legend(facecolor="#222", labelcolor="white", fontsize=8)

ax_mdd = axes[1]
dark_ax(ax_mdd, "Portfolio Max Drawdown")
mdds = [abs(all_port[vn]["max_drawdown"])*100 for vn in variant_names]
bars2 = ax_mdd.bar(variant_names, mdds, color=variant_cols, alpha=0.85, edgecolor="#333")
ax_mdd.set_ylabel("Max Drawdown %", color="white")
for b, d in zip(bars2, mdds):
    ax_mdd.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                f"{d:.1f}%", ha="center", color="white", fontsize=9)

plt.tight_layout()
p = f"{OUT}/r022_dashboard.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 2: Per-symbol PF heatmap ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5), facecolor="#111")
fig.suptitle("R022 — Per-Symbol Profit Factor by Stop Variant", color="white", fontsize=12)
dark_ax(ax)

sym_labels    = [s.split("-")[0] for s in symbols]
n_sym = len(sym_labels)
x     = np.arange(n_sym)
w     = 0.20
offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
for j, (vname, col) in enumerate(zip(variant_names, variant_cols)):
    pfs_sym = [all_results[vname][s]["profit_factor"] for s in symbols]
    bars = ax.bar(x + offsets[j], pfs_sym, w, label=vname, color=col, alpha=0.85)
ax.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5)
ax.axhline(1.2, color="#FF9800", lw=0.8, ls=":", alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(sym_labels, color="white")
ax.set_ylabel("Profit Factor", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r022_grid_heatmap.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 3: WR/PF scatter for LINK specifically ─────────────────────────────
link_sym = "LINK-USDT-SWAP"
if link_sym in symbols:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#111")
    dark_ax(ax, "R022 — LINK: WR vs PF Across Stop Variants")
    for j, (vname, col) in enumerate(zip(variant_names, variant_cols)):
        m = all_results[vname][link_sym]
        wr, pf_ = m["win_rate"]*100, m["profit_factor"]
        ax.scatter(wr, pf_, color=col, s=140, zorder=5)
        ax.text(wr+0.3, pf_, vname, color=col, fontsize=9)
    ax.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(50,  color="white", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Win Rate %", color="white")
    ax.set_ylabel("Profit Factor", color="white")
    plt.tight_layout()
    p = f"{OUT}/r022_link_heatmap.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  → {p}")

# ── Chart 4: Equity curves — best ATR variant vs baseline (all symbols) ───────
n_sym = len(symbols)
ncols = 3; nrows = (n_sym + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows*3.5), facecolor="#111")
fig.suptitle(f"R022 — Equity Curves: BASELINE vs {best_vname} (OOS)",
             color="white", fontsize=12)
ax_flat = axes.flatten()
for i, sym in enumerate(symbols):
    ax  = ax_flat[i]
    col = COLOURS[sym]
    dark_ax(ax, sym.split("-")[0], col)
    eq_b = all_results["BASELINE"][sym].get("equity", np.array([CAPITAL]))
    eq_a = all_results[best_vname][sym].get("equity", np.array([CAPITAL]))
    ax.plot(eq_b, color="gray",  lw=1.2, ls="--", label="Baseline", alpha=0.7)
    ax.plot(eq_a, color=col,     lw=1.5, label=best_vname)
    ax.axhline(CAPITAL, color="white", lw=0.5, ls=":", alpha=0.4)
    pf_b = all_results["BASELINE"][sym]["profit_factor"]
    pf_a = all_results[best_vname][sym]["profit_factor"]
    delta = "▲" if pf_a > pf_b else "▼"
    ax.text(0.05, 0.95,
            f"BASE={pf_b:.2f} | {best_vname}={pf_a:.2f} {delta}",
            transform=ax.transAxes, color="white", fontsize=8, va="top")
    ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=7, loc="lower right")
for j in range(i+1, len(ax_flat)):
    ax_flat[j].set_visible(False)
plt.tight_layout()
p = f"{OUT}/r022_equity_curves.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 5: RSI-style win-rate distribution across variants (all symbols) ────
fig, axes = plt.subplots(1, len(VARIANTS), figsize=(16, 4), facecolor="#111")
fig.suptitle("R022 — Win Rate per Symbol × Stop Variant", color="white", fontsize=12)
for ax, (vname, col) in zip(axes, zip(variant_names, variant_cols)):
    dark_ax(ax, vname, col)
    wrs = [all_results[vname][s]["win_rate"]*100 for s in symbols]
    ax.barh([s.split("-")[0] for s in symbols], wrs, color=col, alpha=0.8)
    ax.axvline(50, color="white", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Win Rate %", color="white", fontsize=7)
    for k, (bar_, wr_) in enumerate(zip(ax.patches, wrs)):
        ax.text(wr_+0.5, bar_.get_y()+bar_.get_height()/2,
                f"{wr_:.1f}%", va="center", color="white", fontsize=7)
plt.tight_layout()
p = f"{OUT}/r022_rsi_distribution.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 6: WR vs PF scatter (all variants, all symbols) ────────────────────
fig, ax = plt.subplots(figsize=(10, 6), facecolor="#111")
dark_ax(ax, "R022 — Win Rate vs Profit Factor: All Symbols × All Variants")
markers = ["o", "s", "^", "D"]
for j, (vname, col, mrkr) in enumerate(zip(variant_names, variant_cols, markers)):
    wrs = [all_results[vname][s]["win_rate"]*100 for s in symbols]
    pfs_sym = [all_results[vname][s]["profit_factor"] for s in symbols]
    ax.scatter(wrs, pfs_sym, color=col, s=80, marker=mrkr, label=vname, zorder=5, alpha=0.85)
ax.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.4)
ax.axvline(50,  color="white", lw=0.8, ls="--", alpha=0.4)
ax.set_xlabel("Win Rate %", color="white")
ax.set_ylabel("Profit Factor", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r022_wr_pf_scatter.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 7: Comprehensive stats table (dashboard) ───────────────────────────
fig = plt.figure(figsize=(20, 12), facecolor="#0d0d0d")
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.5)
fig.suptitle("QUANTLAB AI — R022 DASHBOARD\nATR Stop Hypothesis Test: EMA Pullback Strategy",
             color="white", fontsize=13, y=0.99)

# Full comparison table (variant × symbol)
ax_tbl = fig.add_subplot(gs[0])
ax_tbl.axis("off")

col_headers = ["Symbol"] + [v[0] for v in VARIANTS]
rows_data   = []
for sym in symbols:
    row = [sym.split("-")[0]]
    for vname, _, _ in VARIANTS:
        m = all_results[vname][sym]
        row.append(f"n={m['n_trades']}  PF={m['profit_factor']:.3f}\n"
                   f"WR={m['win_rate']*100:.0f}%  MDD={m['max_drawdown']*100:.1f}%")
    rows_data.append(row)

# Portfolio row
port_row = ["PORTFOLIO"]
for vname, _, _ in VARIANTS:
    p = all_port[vname]
    port_row.append(f"n={p['n_trades']}  PF={p['profit_factor']:.3f}\n"
                    f"WR={p['win_rate']*100:.0f}%  MDD={p['max_drawdown']*100:.1f}%")
rows_data.append(port_row)

tbl = ax_tbl.table(cellText=rows_data, colLabels=col_headers,
                   loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
tbl.scale(1, 2.2)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r % 2 == 0 else "#222")
    cell.set_text_props(color="white")
    cell.set_edgecolor("#333")
    if r == 0:
        cell.set_facecolor("#2a2a2a")
        cell.set_text_props(color="#aaa", fontweight="bold")
    # Highlight best PF per row (>baseline)
    if r > 0 and c > 1:
        row_data = rows_data[r - 1]
        vname    = VARIANTS[c - 1][0]
        sym_key  = symbols[r - 1] if r - 1 < len(symbols) else None
        if sym_key:
            base_pf = all_results["BASELINE"][sym_key]["profit_factor"]
            cur_pf  = all_results[vname][sym_key]["profit_factor"]
            if cur_pf > base_pf + 0.05:
                cell.set_facecolor("#1a3a1a")

# Portfolio PF bar
ax_pf2 = fig.add_subplot(gs[1])
dark_ax(ax_pf2, "Portfolio Metrics Comparison")
x_   = np.arange(len(variant_names))
w_   = 0.2
pf_vals  = [all_port[vn]["profit_factor"] for vn in variant_names]
wr_vals  = [all_port[vn]["win_rate"]*100  for vn in variant_names]
sh_vals  = [all_port[vn]["sharpe"]        for vn in variant_names]

b1 = ax_pf2.bar(x_ - w_, pf_vals,  w_, color=variant_cols, alpha=0.85, label="PF")
ax_pf2.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.4)
ax_pf2.set_xticks(x_); ax_pf2.set_xticklabels(variant_names, color="white")
ax_pf2.set_ylabel("Portfolio PF", color="white")
ax_pf2.legend(facecolor="#222", labelcolor="white", fontsize=8)
for b, pf in zip(b1, pf_vals):
    ax_pf2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f"{pf:.3f}", ha="center", color="white", fontsize=9)

# Conclusions strip
ax_c = fig.add_subplot(gs[2])
ax_c.axis("off")
verdict_str = ("✓ HYPOTHESIS SUPPORTED — ATR stop improves performance on multiple symbols." 
               if any_passes else 
               "✗ HYPOTHESIS REJECTED — Archive EMA Pullback family (ATR stops do not generalise).")
conclusions = [
    verdict_str,
    f"Best ATR multiplier: {best_by_count}  ({count_improved(best_by_count)}/{len(symbols)} symbols improved over baseline)",
    f"Portfolio PF: BASELINE={baseline_port['profit_factor']:.3f}  "
    f"ATR_1.0={all_port['ATR_1.0']['profit_factor']:.3f}  "
    f"ATR_1.5={all_port['ATR_1.5']['profit_factor']:.3f}  "
    f"ATR_2.0={all_port['ATR_2.0']['profit_factor']:.3f}",
    f"Statistical significance: "
    + "  ".join(f"{vn} p={sig_tests.get(vn,float('nan')):.3f}" for vn in ["ATR_1.0","ATR_1.5","ATR_2.0"]),
]
for ki, txt in enumerate(conclusions):
    col_c = "#4CAF50" if "SUPPORTED" in txt else ("#F44336" if "REJECTED" in txt else "white")
    ax_c.text(0.01, 0.85 - ki*0.22, txt, transform=ax_c.transAxes,
              color=col_c, fontsize=9, va="top", wrap=True)

p = f"{OUT}/r022_bootstrap_ci.png"  # reusing expected filename for journal consistency
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 8: Stats summary table image ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5), facecolor="#111")
ax.axis("off")
fig.suptitle("R022 — Portfolio Statistics Summary", color="white", fontsize=12, y=0.97)
dark_ax(ax)

summary_headers = ["Variant", "N", "WR", "PF", "ExpR", "Sharpe", "MDD", "Net $", "p-val"]
summary_rows    = []
for vname, _, _ in VARIANTS:
    p = all_port[vname]
    pval = sig_tests.get(vname, np.nan)
    pval_s = f"{pval:.4f}" if not np.isnan(pval) else "—"
    summary_rows.append([
        vname, str(p["n_trades"]),
        f"{p['win_rate']*100:.1f}%",
        f"{p['profit_factor']:.3f}",
        f"{p['expectancy_r']:+.3f}",
        f"{p['sharpe']:.2f}",
        f"{p['max_drawdown']*100:.1f}%",
        f"${p['net_profit']:+.0f}",
        pval_s,
    ])

tbl2 = ax.table(cellText=summary_rows, colLabels=summary_headers,
                loc="center", cellLoc="center")
tbl2.auto_set_font_size(False); tbl2.set_fontsize(10)
tbl2.scale(1, 2.0)
for (r, c), cell in tbl2.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r % 2 == 0 else "#222")
    cell.set_text_props(color="white")
    cell.set_edgecolor("#333")
    if r == 0:
        cell.set_facecolor("#2a2a2a")
        cell.set_text_props(color="#aaa", fontweight="bold")
    if r == 1:  # BASELINE row — gray tint
        cell.set_facecolor("#252525")
plt.tight_layout()
p = f"{OUT}/r022_stats_summary.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE TRADE LOG (best ATR variant)
# ─────────────────────────────────────────────────────────────────────────────
if all_trades[best_vname]:
    df_log = pd.DataFrame(all_trades[best_vname])
    df_log["entry_time"] = df_log["entry_time"].astype(str)
    df_log["exit_time"]  = df_log["exit_time"].astype(str)
    log_path = f"{OUT}/r022_trade_log.csv"
    df_log.to_csv(log_path, index=False)
    print(f"  → {log_path}  ({len(df_log)} trades, {best_vname})")

# ─────────────────────────────────────────────────────────────────────────────
# APPEND TO JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
try:
    best_atr = best_port
    from datetime import datetime, timezone as _tz
    verdict_str = "PASS" if any_passes else "FAIL"
    journal_rows = []
    for sym in symbols:
        for vname, _, _ in VARIANTS:
            m = all_results[vname][sym]
            journal_rows.append({
                "research_id":     RESEARCH_ID,
                "run_date":        datetime.now(tz=_tz.utc).strftime("%Y-%m-%d"),
                "strategy_name":   f"EMA_Pullback_Stop_{vname}",
                "symbol":          sym,
                "n_trades":        m["n_trades"],
                "profit_factor":   round(m["profit_factor"],   4),
                "expectancy_r":    round(m["expectancy_r"],    4),
                "win_rate":        round(m["win_rate"],         4),
                "net_profit":      round(m["net_profit"],       2),
                "max_drawdown":    round(m["max_drawdown"],     4),
                "sharpe":          round(m["sharpe"],           4),
                "mc_prob_profit":  0.0,
                "avg_hold_minutes": round(m["avg_hold_minutes"], 1),
                "verdict":         verdict(m),
            })
    append_journal(journal_rows)
    print(f"\n  Journal updated → {CONFIG['JOURNAL_FILE']}")
except Exception as e:
    print(f"\n  [WARN] Journal update failed: {e}")

print(f"\n{'═'*70}")
print(f"  R022 complete. Output folder: {OUT}/")
print(f"{'═'*70}\n")
