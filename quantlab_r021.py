"""
QUANTLAB AI — RESEARCH #021
Strategy Tournament: Trend-Following Families
=============================================

Three signal families tested head-to-head across all 8 symbols on 1H.
Winner determined by portfolio OOS Profit Factor.

Family A — EMA Crossover + ADX Filter
  Signal : fast_EMA crosses above slow_EMA AND ADX > threshold
  Grid   : fast ∈ {10,20} × slow ∈ {50,100} × adx ∈ {20,25,30}  → 12 combos

Family B — Donchian Channel Breakout
  Signal : close > rolling_N_bar_high (no lookahead, shifted by 1)
           AND close > EMA(200)  [trend filter]
  Grid   : N ∈ {10,20,30,50}  → 4 combos

Family C — EMA Pullback in Uptrend
  Signal : close > EMA(slow)  [uptrend]
           AND low touched fast_EMA in last 3 bars (price pulled back)
           AND close > fast_EMA  [bounce confirmed]
           AND ADX > 20  [trend active]
  Grid   : fast ∈ {20,50} × slow ∈ {100,200}  → 4 combos

All backtests use the locked run_backtest engine (long-only, stop = prev-bar low).
Split: 70/30 chronological.  All 8 symbols.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import (
    CONFIG, calc_ema, calc_atr, calc_adx,
    run_backtest, compute_metrics, monte_carlo, append_journal,
)

RESEARCH_ID = "R021"
OUT         = CONFIG["OUTPUT_FOLDER"]
CACHE       = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

ALL_SYMBOLS = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "XRP-USDT-SWAP","DOGE-USDT-SWAP","LTC-USDT-SWAP","AVAX-USDT-SWAP","BCH-USDT-SWAP",
]
SPLIT = 0.70
CAPITAL = CONFIG["STARTING_CAPITAL"]

COLOURS = {
    "BTC-USDT-SWAP":"#F7931A","ETH-USDT-SWAP":"#627EEA","SOL-USDT-SWAP":"#9945FF",
    "LINK-USDT-SWAP":"#2A5ADA","XRP-USDT-SWAP":"#00AAE4","DOGE-USDT-SWAP":"#C3A634",
    "LTC-USDT-SWAP":"#BFBBBB","AVAX-USDT-SWAP":"#E84142","BCH-USDT-SWAP":"#8DC351",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_1h(sym):
    tag = sym.replace("-","_")
    df  = pd.read_parquet(f"{CACHE}/{tag}_1H.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)

def split_df(df):
    cut = int(len(df) * SPLIT)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)

def verdict(m):
    n, pf, ex = m["n_trades"], m["profit_factor"], m["expectancy_r"]
    sig = pf >= 1.5 and n >= 30
    if n < 20:             return "INSUFFICIENT"
    if pf >= 1.5 and sig:  return "PROMOTE"
    if pf >= 1.3 and ex >= 0.15: return "WATCHLIST"
    if pf >= 1.0:          return "WEAK"
    return "REJECT"

def portfolio_pf(trades_list):
    """Compute portfolio PF from a flat list of trade dicts."""
    if not trades_list: return 0.0
    wins  = sum(t["pnl"] for t in trades_list if t["pnl"] > 0)
    loss  = abs(sum(t["pnl"] for t in trades_list if t["pnl"] < 0))
    return wins / loss if loss > 0 else (float("inf") if wins > 0 else 0.0)

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def sig_ema_cross(df, fast, slow, adx_thresh):
    """Family A: EMA(fast) crosses above EMA(slow) with ADX filter."""
    df = df.copy()
    df["ema_f"] = calc_ema(df["close"], fast)
    df["ema_s"] = calc_ema(df["close"], slow)
    df["adx"]   = calc_adx(df, 14)
    cross = (df["ema_f"] > df["ema_s"]) & (df["ema_f"].shift(1) <= df["ema_s"].shift(1))
    return (cross & (df["adx"] > adx_thresh)).astype(int)

def sig_donchian(df, n):
    """Family B: Close breaks N-bar high (no lookahead) + above EMA200."""
    df = df.copy()
    df["ema200"]   = calc_ema(df["close"], 200)
    # shift(1) so the high of the prior N bars is used (no lookahead)
    df["dc_high"]  = df["high"].shift(1).rolling(n).max()
    signal = (df["close"] > df["dc_high"]) & (df["close"] > df["ema200"])
    return signal.fillna(False).astype(int)

def sig_ema_pullback(df, fast, slow):
    """
    Family C: Uptrend pullback entry.
    Uptrend  : close > EMA(slow)
    Pullback : low <= EMA(fast) on one of last 3 bars
    Bounce   : close > EMA(fast) today
    Trend    : ADX(14) > 20
    """
    df = df.copy()
    df["ema_f"] = calc_ema(df["close"], fast)
    df["ema_s"] = calc_ema(df["close"], slow)
    df["adx"]   = calc_adx(df, 14)

    touched = (df["low"] <= df["ema_f"])
    touched_recently = touched | touched.shift(1) | touched.shift(2)

    uptrend = df["close"] > df["ema_s"]
    bounce  = df["close"] > df["ema_f"]
    trend   = df["adx"] > 20

    return (uptrend & touched_recently & bounce & trend).fillna(False).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# RUN TOURNAMENT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #021" + " "*50 + "║")
print("║  Strategy Tournament: Trend-Following Families" + " "*32 + "║")
print("╚" + "═"*79 + "╝")
print("""
  Family A : EMA Crossover + ADX   fast×slow×adx  → 12 combos
  Family B : Donchian Breakout      N bars         →  4 combos
  Family C : EMA Pullback           fast×slow      →  4 combos
  Symbols  : All 9 (1H)
  Split    : 70/30 chronological
""")

# ── load & split ──────────────────────────────────────────────────────────────
print("  Loading 1H data …")
train_dfs = {}
oos_dfs   = {}
for sym in ALL_SYMBOLS:
    try:
        df = load_1h(sym)
        train_dfs[sym], oos_dfs[sym] = split_df(df)
        print(f"  {sym:25s}  total={len(df):,}  train={len(train_dfs[sym]):,}  oos={len(oos_dfs[sym]):,}")
    except FileNotFoundError:
        print(f"  {sym:25s}  *** cache missing — skipping ***")

symbols = list(train_dfs.keys())

# ─────────────────────────────────────────────────────────────────────────────
# FAMILY A  — EMA CROSSOVER + ADX
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("  FAMILY A — EMA Crossover + ADX")
print("─"*70)

A_FAST  = [10, 20]
A_SLOW  = [50, 100]
A_ADX   = [20, 25, 30]

best_a_pf    = 0.0
best_a_params = None
best_a_train  = {}

for fast, slow, adx_t in product(A_FAST, A_SLOW, A_ADX):
    combo_trades = []
    sym_results  = {}
    for sym in symbols:
        df_tr  = train_dfs[sym].copy()
        sigs   = sig_ema_cross(df_tr, fast, slow, adx_t)
        df_tr["signal_col"] = sigs
        res    = run_backtest(df_tr, lambda d: d["signal_col"], f"{sym}")
        m      = compute_metrics(res["trades"], sym)
        sym_results[sym] = m
        combo_trades.extend(res["trades"])
    pf = portfolio_pf(combo_trades)
    if pf > best_a_pf:
        best_a_pf     = pf
        best_a_params = (fast, slow, adx_t)
        best_a_train  = sym_results

print(f"  Best train params: EMA({best_a_params[0]}/{best_a_params[1]}) ADX>{best_a_params[2]}  PF={best_a_pf:.3f}")

# OOS with best A params
a_oos_results = {}
a_oos_trades  = []
fast, slow, adx_t = best_a_params
for sym in symbols:
    df_oos = oos_dfs[sym].copy()
    sigs   = sig_ema_cross(df_oos, fast, slow, adx_t)
    df_oos["signal_col"] = sigs
    res    = run_backtest(df_oos, lambda d: d["signal_col"], sym)
    m      = compute_metrics(res["trades"], sym)
    v      = verdict(m)
    a_oos_results[sym] = m
    a_oos_trades.extend(res["trades"])
    tag = sym.split("-")[0]
    print(f"  {tag:5s}  n={m['n_trades']:3d}  WR={m['win_rate']*100:4.1f}%  PF={m['profit_factor']:.3f}"
          f"  ExpR={m['expectancy_r']:+.3f}  → {v}")

a_port_oos = compute_metrics(a_oos_trades, "PORTFOLIO")
print(f"  Portfolio OOS: PF={a_port_oos['profit_factor']:.3f}  n={a_port_oos['n_trades']}")

# ─────────────────────────────────────────────────────────────────────────────
# FAMILY B  — DONCHIAN BREAKOUT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("  FAMILY B — Donchian Channel Breakout (+ EMA200 filter)")
print("─"*70)

B_PERIODS = [10, 20, 30, 50]

best_b_pf     = 0.0
best_b_params = None

for n in B_PERIODS:
    combo_trades = []
    for sym in symbols:
        df_tr = train_dfs[sym].copy()
        sigs  = sig_donchian(df_tr, n)
        df_tr["signal_col"] = sigs
        res   = run_backtest(df_tr, lambda d: d["signal_col"], sym)
        combo_trades.extend(res["trades"])
    pf = portfolio_pf(combo_trades)
    print(f"  Donchian({n:2d})  train PF={pf:.3f}  n={len(combo_trades)}")
    if pf > best_b_pf:
        best_b_pf     = pf
        best_b_params = n

print(f"\n  Best train: Donchian({best_b_params})  PF={best_b_pf:.3f}")

b_oos_results = {}
b_oos_trades  = []
for sym in symbols:
    df_oos = oos_dfs[sym].copy()
    sigs   = sig_donchian(df_oos, best_b_params)
    df_oos["signal_col"] = sigs
    res    = run_backtest(df_oos, lambda d: d["signal_col"], sym)
    m      = compute_metrics(res["trades"], sym)
    v      = verdict(m)
    b_oos_results[sym] = m
    b_oos_trades.extend(res["trades"])
    tag = sym.split("-")[0]
    print(f"  {tag:5s}  n={m['n_trades']:3d}  WR={m['win_rate']*100:4.1f}%  PF={m['profit_factor']:.3f}"
          f"  ExpR={m['expectancy_r']:+.3f}  → {v}")

b_port_oos = compute_metrics(b_oos_trades, "PORTFOLIO")
print(f"  Portfolio OOS: PF={b_port_oos['profit_factor']:.3f}  n={b_port_oos['n_trades']}")

# ─────────────────────────────────────────────────────────────────────────────
# FAMILY C  — EMA PULLBACK
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*70)
print("  FAMILY C — EMA Pullback in Uptrend")
print("─"*70)

C_FAST = [20, 50]
C_SLOW = [100, 200]

best_c_pf     = 0.0
best_c_params = None

for fast, slow in product(C_FAST, C_SLOW):
    combo_trades = []
    for sym in symbols:
        df_tr = train_dfs[sym].copy()
        sigs  = sig_ema_pullback(df_tr, fast, slow)
        df_tr["signal_col"] = sigs
        res   = run_backtest(df_tr, lambda d: d["signal_col"], sym)
        combo_trades.extend(res["trades"])
    pf = portfolio_pf(combo_trades)
    print(f"  Pullback EMA({fast}/{slow})  train PF={pf:.3f}  n={len(combo_trades)}")
    if pf > best_c_pf:
        best_c_pf     = pf
        best_c_params = (fast, slow)

print(f"\n  Best train: Pullback EMA({best_c_params[0]}/{best_c_params[1]})  PF={best_c_pf:.3f}")

c_oos_results = {}
c_oos_trades  = []
fast, slow = best_c_params
for sym in symbols:
    df_oos = oos_dfs[sym].copy()
    sigs   = sig_ema_pullback(df_oos, fast, slow)
    df_oos["signal_col"] = sigs
    res    = run_backtest(df_oos, lambda d: d["signal_col"], sym)
    m      = compute_metrics(res["trades"], sym)
    v      = verdict(m)
    c_oos_results[sym] = m
    c_oos_trades.extend(res["trades"])
    tag = sym.split("-")[0]
    print(f"  {tag:5s}  n={m['n_trades']:3d}  WR={m['win_rate']*100:4.1f}%  PF={m['profit_factor']:.3f}"
          f"  ExpR={m['expectancy_r']:+.3f}  → {v}")

c_port_oos = compute_metrics(c_oos_trades, "PORTFOLIO")
print(f"  Portfolio OOS: PF={c_port_oos['profit_factor']:.3f}  n={c_port_oos['n_trades']}")

# ─────────────────────────────────────────────────────────────────────────────
# TOURNAMENT WINNER
# ─────────────────────────────────────────────────────────────────────────────
results_map = {
    "A_EMA_Cross":  (a_port_oos["profit_factor"], a_oos_results, a_oos_trades,
                     f"EMA({best_a_params[0]}/{best_a_params[1]}) ADX>{best_a_params[2]}"),
    "B_Donchian":   (b_port_oos["profit_factor"], b_oos_results, b_oos_trades,
                     f"Donchian({best_b_params}) + EMA200"),
    "C_Pullback":   (c_port_oos["profit_factor"], c_oos_results, c_oos_trades,
                     f"Pullback EMA({best_c_params[0]}/{best_c_params[1]})"),
}

winner_key  = max(results_map, key=lambda k: results_map[k][0])
winner_pf, winner_oos, winner_trades, winner_label = results_map[winner_key]

print(f"""
{'='*80}
  TOURNAMENT RESULTS (OOS Portfolio PF)
{'='*80}
  Family A  EMA Crossover  : PF={a_port_oos['profit_factor']:.3f}  n={a_port_oos['n_trades']}
  Family B  Donchian Break : PF={b_port_oos['profit_factor']:.3f}  n={b_port_oos['n_trades']}
  Family C  EMA Pullback   : PF={c_port_oos['profit_factor']:.3f}  n={c_port_oos['n_trades']}

  ★ WINNER: {winner_key}  — {winner_label}  PF={winner_pf:.3f}
{'='*80}
""")

# Per-symbol breakdown for winner
promoted  = [s for s in symbols if verdict(winner_oos[s]) == "PROMOTE"]
watchlist = [s for s in symbols if verdict(winner_oos[s]) == "WATCHLIST"]
print("  Winner per-symbol detail:")
for sym in symbols:
    m = winner_oos[sym]
    v = verdict(m)
    print(f"  {sym.split('-')[0]:5s}  n={m['n_trades']:3d}  WR={m['win_rate']*100:4.1f}%"
          f"  PF={m['profit_factor']:.3f}  ExpR={m['expectancy_r']:+.3f}"
          f"  Sharpe={m['sharpe']:5.2f}  MDD={m['max_drawdown']*100:.1f}%  → {v}")

port_winner = compute_metrics(winner_trades, "PORTFOLIO")

# ─────────────────────────────────────────────────────────────────────────────
# DEEP-DIVE: MONTE CARLO ON WINNER
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Running Monte Carlo on winner portfolio …")
winner_pnls = np.array([t["pnl"] for t in winner_trades])
mc_result   = monte_carlo(winner_pnls, n_iter=CONFIG["MC_ITERATIONS"]) if len(winner_pnls) >= 10 else None

# Bootstrap CI per winning symbol
boot_ci = {}
for sym in symbols:
    pnls = np.array([t["pnl"] for t in winner_trades if t["label"] == sym])
    if len(pnls) >= 5:
        samps = [np.mean(np.random.choice(pnls, len(pnls), replace=True)) for _ in range(2000)]
        boot_ci[sym] = np.percentile(samps, [5, 50, 95])
    else:
        boot_ci[sym] = np.zeros(3)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating charts …")

# helpers
def dark_ax(ax, title=None, col="white"):
    ax.set_facecolor("#111")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    if title: ax.set_title(title, color=col, fontsize=10)

# ── Chart 1: Tournament OOS comparison bar chart ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#111")
fig.suptitle("R021 — Tournament OOS Comparison (Best Params per Family)",
             color="white", fontsize=12)
family_labels = ["A: EMA Cross", "B: Donchian", "C: Pullback"]
family_pfs    = [a_port_oos["profit_factor"],
                 b_port_oos["profit_factor"],
                 c_port_oos["profit_factor"]]
family_ns     = [a_port_oos["n_trades"],
                 b_port_oos["n_trades"],
                 c_port_oos["n_trades"]]
fam_data = [
    (a_oos_results, "#F7931A"),
    (b_oos_results, "#627EEA"),
    (c_oos_results, "#9945FF"),
]
for ax, (res_dict, col), flbl in zip(axes, fam_data, family_labels):
    dark_ax(ax, flbl, col)
    syms_s = [s.split("-")[0] for s in symbols]
    pfs    = [res_dict[s]["profit_factor"] for s in symbols]
    bars   = ax.bar(syms_s, pfs, color=[COLOURS[s] for s in symbols])
    ax.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.5)
    ax.set_ylabel("Profit Factor", color="white")
    ax.set_xticklabels(syms_s, rotation=30, ha="right")
    for b, p in zip(bars, pfs):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"{p:.2f}", ha="center", color="white", fontsize=7)
plt.tight_layout()
p = f"{OUT}/r021_tournament_comparison.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 2: Winner equity curves ────────────────────────────────────────────
n_sym = len(symbols)
ncols = 3; nrows = (n_sym + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows*3.5), facecolor="#111")
fig.suptitle(f"R021 — Winner [{winner_key}] Equity Curves (OOS)\n{winner_label}",
             color="white", fontsize=12)
ax_flat = axes.flatten()
for i, sym in enumerate(symbols):
    ax = ax_flat[i]
    m  = winner_oos[sym]
    eq = m.get("equity", np.array([CAPITAL]))
    col = COLOURS[sym]
    dark_ax(ax, sym.split("-")[0], col)
    ax.plot(eq, color=col, lw=1.4)
    ax.axhline(CAPITAL, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Trade #", color="white", fontsize=7)
    v = verdict(m)
    ax.text(0.05, 0.95, f"n={m['n_trades']}  PF={m['profit_factor']:.2f}  {v}",
            transform=ax.transAxes, color="white", fontsize=8, va="top")
for j in range(i+1, len(ax_flat)):
    ax_flat[j].set_visible(False)
plt.tight_layout()
p = f"{OUT}/r021_winner_equity.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 3: Winner drawdown ─────────────────────────────────────────────────
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows*3.5), facecolor="#111")
fig.suptitle(f"R021 — Winner [{winner_key}] Drawdown (OOS)", color="white", fontsize=12)
ax_flat = axes.flatten()
for i, sym in enumerate(symbols):
    ax  = ax_flat[i]
    m   = winner_oos[sym]
    eq  = m.get("equity", np.array([CAPITAL]))
    dd  = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100
    col = COLOURS[sym]
    dark_ax(ax, sym.split("-")[0], col)
    ax.fill_between(range(len(dd)), dd, 0, color=col, alpha=0.55)
    ax.set_xlabel("Trade #", color="white", fontsize=7)
    ax.set_ylabel("DD %", color="white", fontsize=7)
for j in range(i+1, len(ax_flat)):
    ax_flat[j].set_visible(False)
plt.tight_layout()
p = f"{OUT}/r021_winner_drawdown.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 4: Portfolio PF bar — all families + winner highlight ───────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#111")
dark_ax(ax, "R021 — OOS Portfolio PF by Strategy Family", "white")
bars = ax.bar(family_labels, family_pfs, color=["#F7931A","#627EEA","#9945FF"], alpha=0.85)
ax.axhline(1.0, color="white", lw=1, ls="--", alpha=0.5)
win_idx = ["A_EMA_Cross","B_Donchian","C_Pullback"].index(winner_key)
bars[win_idx].set_edgecolor("white"); bars[win_idx].set_linewidth(2.5)
for b, pf, n in zip(bars, family_pfs, family_ns):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            f"PF={pf:.3f}\nn={n}", ha="center", color="white", fontsize=10)
ax.set_ylabel("Portfolio Profit Factor", color="white")
plt.tight_layout()
p = f"{OUT}/r021_family_pf.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 5: Win rate distribution (winner, all symbols) ─────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#111")
dark_ax(ax, f"R021 — Win Rate & PF by Symbol [{winner_label}]", "white")
syms_s = [s.split("-")[0] for s in symbols]
wrs = [winner_oos[s]["win_rate"]*100 for s in symbols]
pfs = [winner_oos[s]["profit_factor"] for s in symbols]
x   = np.arange(len(symbols)); w = 0.35
ax2 = ax.twinx()
ax.bar(x - w/2, wrs, w, color=[COLOURS[s] for s in symbols], alpha=0.8, label="Win Rate %")
ax2.bar(x + w/2, pfs, w, color=[COLOURS[s] for s in symbols], alpha=0.5, label="PF")
ax.axhline(50, color="white", lw=0.7, ls="--", alpha=0.4)
ax2.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.4)
ax.set_xticks(x); ax.set_xticklabels(syms_s, color="white")
ax.tick_params(colors="white"); ax2.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.set_ylabel("Win Rate %", color="white"); ax2.set_ylabel("Profit Factor", color="white")
ax.legend(loc="upper left", facecolor="#222", labelcolor="white", fontsize=8)
ax2.legend(loc="upper right", facecolor="#222", labelcolor="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r021_winner_wr_pf.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 6: Monte Carlo (winner portfolio) ───────────────────────────────────
if mc_result:
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#111")
    dark_ax(ax, f"R021 — Monte Carlo Final Equity [{winner_label}]  ({CONFIG['MC_ITERATIONS']:,} iter.)", "white")
    finals = mc_result["final_equities"]
    _lo, _hi = np.min(finals), np.max(finals)
    if _hi > _lo:
        ax.hist(finals, bins=np.linspace(_lo, _hi, 51),
                color="#627EEA", alpha=0.75, edgecolor="none")
    else:
        ax.axvline(_lo, color="#627EEA", lw=3)
        ax.text(0.5, 0.5, f"All paths={_lo:,.0f}", transform=ax.transAxes,
                ha="center", color="white", fontsize=9)
    p5, p50, p95 = np.percentile(finals, [5, 50, 95])
    ax.axvline(p5,  color="#F44336", lw=1.5, ls="--", label=f"p5  ${p5:,.0f}")
    ax.axvline(p50, color="#4CAF50", lw=1.5, ls="--", label=f"p50 ${p50:,.0f}")
    ax.axvline(p95, color="#FF9800", lw=1.5, ls="--", label=f"p95 ${p95:,.0f}")
    ax.axvline(CAPITAL, color="white", lw=1, ls=":", alpha=0.5, label=f"Start ${CAPITAL:,}")
    ax.set_xlabel("Final Equity $", color="white"); ax.set_ylabel("Count", color="white")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
    plt.tight_layout()
    p = f"{OUT}/r021_monte_carlo.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  → {p}")

# ── Chart 7: Bootstrap CI (winner, per symbol) ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor="#111")
dark_ax(ax, f"R021 — Bootstrap 90% CI on Mean P&L / Trade [{winner_label}]", "white")
for i, sym in enumerate(symbols):
    lo, med, hi = boot_ci[sym]
    col = COLOURS[sym]
    ax.errorbar(i, med, yerr=[[med-lo],[hi-med]],
                fmt="o", color=col, capsize=8, capthick=2, ms=7)
    ax.text(i, hi + abs(hi-lo)*0.1, f"[{lo:.0f},{hi:.0f}]",
            ha="center", color=col, fontsize=7)
ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
ax.set_xticks(range(len(symbols)))
ax.set_xticklabels([s.split("-")[0] for s in symbols], color="white")
ax.set_ylabel("Mean P&L per Trade $", color="white")
plt.tight_layout()
p = f"{OUT}/r021_bootstrap_ci.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 8: Dashboard ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor="#0d0d0d")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)
fig.suptitle(f"QUANTLAB AI — R021 DASHBOARD\nStrategy Tournament  |  Winner: {winner_key} — {winner_label}",
             color="white", fontsize=13, y=0.98)

# Metric table (top full width)
ax_t = fig.add_subplot(gs[0, :])
ax_t.axis("off")
rows_data = []
for sym in symbols:
    m = winner_oos[sym]
    v = verdict(m)
    rows_data.append([
        sym.split("-")[0],
        str(m["n_trades"]),
        f"{m['win_rate']*100:.1f}%",
        f"{m['profit_factor']:.3f}",
        f"{m['expectancy_r']:+.3f}",
        f"{m['sharpe']:5.2f}",
        f"{m['max_drawdown']*100:.1f}%",
        v,
    ])
# portfolio row
pm = port_winner
rows_data.append([
    "PORTFOLIO",
    str(pm["n_trades"]),
    f"{pm['win_rate']*100:.1f}%",
    f"{pm['profit_factor']:.3f}",
    f"{pm['expectancy_r']:+.3f}",
    f"{pm['sharpe']:5.2f}",
    f"{pm['max_drawdown']*100:.1f}%",
    "PORTFOLIO",
])
headers = ["Symbol","n","WR","PF","ExpR","Sharpe","MDD","Verdict"]
tbl = ax_t.table(cellText=rows_data, colLabels=headers, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r,c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r%2==0 else "#222")
    cell.set_text_props(color="white")
    cell.set_edgecolor("#333")
    if r == 0:
        cell.set_facecolor("#2a2a2a")
        cell.set_text_props(color="#aaa", fontweight="bold")
    if r > 0 and c == 7:
        v_text = rows_data[r-1][7]
        col_map = {"PROMOTE":"#4CAF50","WATCHLIST":"#FF9800","WEAK":"#FFC107",
                   "REJECT":"#F44336","INSUFFICIENT":"#9E9E9E","PORTFOLIO":"#627EEA"}
        cell.set_facecolor(col_map.get(v_text,"#222"))

# Portfolio bar
ax_pf = fig.add_subplot(gs[1, 0])
dark_ax(ax_pf, "Portfolio PF", "white")
ax_pf.bar(["A","B","C"], family_pfs, color=["#F7931A","#627EEA","#9945FF"], alpha=0.85)
ax_pf.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.5)
ax_pf.set_ylabel("PF", color="white", fontsize=7)
bars2 = ax_pf.patches
bars2[win_idx].set_edgecolor("white"); bars2[win_idx].set_linewidth(2)
for b, pf in zip(bars2, family_pfs):
    ax_pf.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{pf:.2f}",
               ha="center", color="white", fontsize=8)

# Winner equity (first promoted/watchlist symbol or BTC)
best_sym = promoted[0] if promoted else (watchlist[0] if watchlist else "BTC-USDT-SWAP")
ax_eq = fig.add_subplot(gs[1, 1])
eq = winner_oos[best_sym].get("equity", np.array([CAPITAL]))
dark_ax(ax_eq, f"{best_sym.split('-')[0]} Equity", COLOURS[best_sym])
ax_eq.plot(eq, color=COLOURS[best_sym], lw=1.3)
ax_eq.axhline(CAPITAL, color="gray", lw=0.6, ls=":")

# MC final equity distribution
if mc_result:
    ax_mc = fig.add_subplot(gs[1, 2])
    dark_ax(ax_mc, "Monte Carlo Final $", "white")
    _fe = mc_result["final_equities"]
    _lo2, _hi2 = np.min(_fe), np.max(_fe)
    if _hi2 > _lo2:
        ax_mc.hist(_fe, bins=np.linspace(_lo2, _hi2, 31), color="#627EEA", alpha=0.7)
    else:
        ax_mc.axvline(_lo2, color="#627EEA", lw=3)
    ax_mc.axvline(CAPITAL, color="white", lw=1, ls=":", alpha=0.5)
    ax_mc.axvline(np.percentile(mc_result["final_equities"], 5),
                  color="#F44336", lw=1.5, ls="--")

# Bootstrap CI mini
ax_b = fig.add_subplot(gs[1, 3])
dark_ax(ax_b, "Boot CI (mean P&L)", "white")
for i, sym in enumerate(symbols):
    lo, med, hi = boot_ci[sym]
    ax_b.errorbar(i, med, yerr=[[med-lo],[hi-med]],
                  fmt="o", color=COLOURS[sym], capsize=5, ms=5)
ax_b.axhline(0, color="white", lw=0.6, ls="--", alpha=0.5)
ax_b.set_xticks(range(len(symbols)))
ax_b.set_xticklabels([s.split("-")[0][:3] for s in symbols], fontsize=6)

# WR by symbol
ax_wr = fig.add_subplot(gs[2, :2])
dark_ax(ax_wr, "Win Rate by Symbol (OOS)", "white")
wrs = [winner_oos[s]["win_rate"]*100 for s in symbols]
ax_wr.bar([s.split("-")[0] for s in symbols], wrs,
          color=[COLOURS[s] for s in symbols], alpha=0.85)
ax_wr.axhline(50, color="white", lw=0.7, ls="--", alpha=0.5)
ax_wr.set_ylabel("WR %", color="white", fontsize=7)

# PF by symbol
ax_pfb = fig.add_subplot(gs[2, 2:])
dark_ax(ax_pfb, "Profit Factor by Symbol (OOS)", "white")
pfs2 = [winner_oos[s]["profit_factor"] for s in symbols]
ax_pfb.bar([s.split("-")[0] for s in symbols], pfs2,
           color=[COLOURS[s] for s in symbols], alpha=0.85)
ax_pfb.axhline(1.0, color="white", lw=0.7, ls="--", alpha=0.5)
ax_pfb.set_ylabel("Profit Factor", color="white", fontsize=7)

plt.savefig(f"{OUT}/r021_dashboard.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  → {OUT}/r021_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Writing journal …")
today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
rows  = []

# All three families — portfolio row each
for key, (pf, oos_res, oos_tr, label) in results_map.items():
    pm = compute_metrics(oos_tr, "PORTFOLIO")
    rows.append({
        "research_id":    RESEARCH_ID,
        "date":           today,
        "strategy_label": key,
        "symbol":         "PORTFOLIO",
        "n_trades":       pm["n_trades"],
        "profit_factor":  round(pm["profit_factor"], 4),
        "expectancy_r":   round(pm["expectancy_r"],  4),
        "win_rate":       round(pm["win_rate"],       4),
        "net_pnl":        round(pm["net_profit"],     2),
        "max_drawdown":   round(pm["max_drawdown"],   4),
        "sharpe_ratio":   round(pm["sharpe"],         4),
        "significant":    1 if pm["profit_factor"] >= 1.5 and pm["n_trades"] >= 30 else 0,
        "avg_hold_h":     round(pm.get("avg_hold_minutes", 0) / 60, 1),
        "verdict":        "PROMOTE" if pm["profit_factor"] >= 1.5 else (
                          "WATCHLIST" if pm["profit_factor"] >= 1.3 else (
                          "WEAK" if pm["profit_factor"] >= 1.0 else "REJECT")),
    })

# Per-symbol rows for winner
for sym in symbols:
    m = winner_oos[sym]
    v = verdict(m)
    rows.append({
        "research_id":    RESEARCH_ID,
        "date":           today,
        "strategy_label": f"{winner_key}_{sym.split('-')[0]}",
        "symbol":         sym,
        "n_trades":       m["n_trades"],
        "profit_factor":  round(m["profit_factor"], 4),
        "expectancy_r":   round(m["expectancy_r"],  4),
        "win_rate":       round(m["win_rate"],       4),
        "net_pnl":        round(m["net_profit"],     2),
        "max_drawdown":   round(m["max_drawdown"],   4),
        "sharpe_ratio":   round(m["sharpe"],         4),
        "significant":    1 if m["profit_factor"] >= 1.5 and m["n_trades"] >= 30 else 0,
        "avg_hold_h":     round(m.get("avg_hold_minutes", 0) / 60, 1),
        "verdict":        v,
    })

append_journal(rows)
print(f"  Journal updated → {CONFIG['JOURNAL_FILE']}  ({len(rows)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL PRINT
# ─────────────────────────────────────────────────────────────────────────────
print(f"""
{'='*80}
  QUANTLAB AI — R021 FINAL SUMMARY
{'='*80}
  ★ WINNER : {winner_key} — {winner_label}
  Portfolio OOS PF = {winner_pf:.3f}   n = {port_winner['n_trades']}
  Promoted symbols : {len(promoted)}  {[s.split('-')[0] for s in promoted]}
  Watchlist        : {len(watchlist)}  {[s.split('-')[0] for s in watchlist]}
""")

if mc_result:
    finals = mc_result["final_equities"]
    p5, p50, p95 = np.percentile(finals, [5, 50, 95])
    ruin_rate = np.mean(finals < CAPITAL * 0.5) * 100
    print(f"  Monte Carlo ({CONFIG['MC_ITERATIONS']:,} iter.):")
    print(f"    p5=${p5:,.0f}  median=${p50:,.0f}  p95=${p95:,.0f}")
    print(f"    Ruin rate (<50% capital): {ruin_rate:.1f}%")

print(f"\n  All outputs → {OUT}/r021_*")
print("  Research #021 complete.\n")
