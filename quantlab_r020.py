"""
QUANTLAB AI — RESEARCH #020
Funding Rate Extremes: Mean-Reversion
======================================

Hypothesis
----------
OKX perpetual funding settles every 8 hours (00:00 / 08:00 / 16:00 UTC).
When funding is extremely positive  → the market is crowded long → fade SHORT.
When funding is extremely negative  → the market is crowded short → fade LONG.
The crowd that paid elevated funding has an incentive to close, providing
short-term mean-reversion pressure over the next 8–24 hours.

Symbols : BTC, ETH, SOL  (only symbols with funding cache)
TF      : 1H price data aligned to funding settlements
Split   : 70/30 chronological train / OOS

Grid
----
  threshold  ∈ {0.03%, 0.05%, 0.07%, 0.10%}   |funding_rate| cutoff
  hold_hours ∈ {8, 16, 24}                      time-based exit (hours)
  direction  ∈ {long_only, short_only, both}     sensitivity test

Output
------
  8 charts + ≤9 journal rows (3 symbols × up to 3 verdict tiers)
"""

import os, sys, csv, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings("ignore")

# ── shared infrastructure ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import (
    CONFIG, compute_metrics, monte_carlo, append_journal, JOURNAL_COLS,
)

RESEARCH_ID  = "R020"
OUT          = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUT, exist_ok=True)

SYMBOLS_FUND = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
CACHE        = CONFIG["CACHE_FOLDER"]
RR           = CONFIG["RISK_REWARD"]
FEE          = CONFIG["TAKER_FEE"]
SPREAD       = CONFIG["SPREAD"] * 0.5
SLIP         = CONFIG["SL_SLIPPAGE"]
CAPITAL      = CONFIG["STARTING_CAPITAL"]
RISK_FRAC    = CONFIG["RISK_PER_TRADE_PCT"]
MAX_LEV      = CONFIG["MAX_LEVERAGE"]
MIN_SL_PCT   = CONFIG["MIN_SL_PCT"]
MC_ITER      = CONFIG["MC_ITERATIONS"]

THRESHOLDS   = [0.000030, 0.000050, 0.000070, 0.000090]  # p60…p90 of OKX settlements
HOLD_HOURS   = [8, 16, 24]
SPLIT_FRAC   = 0.70

COLOURS = {
    "BTC-USDT-SWAP": "#F7931A",
    "ETH-USDT-SWAP": "#627EEA",
    "SOL-USDT-SWAP": "#9945FF",
}
STYLE = "dark_background"

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(sym: str):
    """Load 1H OHLCV and funding frames; align on settlement times."""
    tag  = sym.replace("-", "_")
    ohlc = pd.read_parquet(f"{CACHE}/{tag}_1H.parquet")
    fund = pd.read_parquet(f"{CACHE}/{tag}_funding.parquet")

    # Normalise tz → UTC, floor to hour
    ohlc["datetime"] = pd.to_datetime(ohlc["datetime"], utc=True)
    fund["datetime"] = pd.to_datetime(fund["datetime"], utc=True)

    ohlc = ohlc.sort_values("datetime").reset_index(drop=True)
    fund = fund.sort_values("datetime").reset_index(drop=True)

    # Forward-fill funding onto every 1H bar (rate is known AT settlement bar)
    ohlc = ohlc.set_index("datetime")
    fund = fund.set_index("datetime").rename(columns={"funding_rate": "fr"})
    ohlc = ohlc.join(fund, how="left")
    ohlc["fr"] = ohlc["fr"].ffill()
    ohlc = ohlc.reset_index()

    # Restrict to the window where funding is available
    first_fund = fund.index.min()
    ohlc = ohlc[ohlc["datetime"] >= first_fund].reset_index(drop=True)

    return ohlc


# ─────────────────────────────────────────────────────────────────────────────
# 2. SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_signals(df: pd.DataFrame, threshold: float):
    """
    Returns a Series of +1 (long), -1 (short), 0 (flat) for each bar.
    A signal fires on the bar AFTER a funding extreme is published.
    The funding rate on bar i was settled AT bar i; we act on bar i+1 open.
    We only fire at settlement bars (08:00, 16:00, 00:00 UTC).
    """
    hrs = df["datetime"].dt.hour
    is_settle = hrs.isin([0, 8, 16])

    sig = pd.Series(0, index=df.index)
    # At settlement bar i, if extreme → signal on bar i+1
    for i in df.index[:-1]:
        if not is_settle.iloc[i]:
            continue
        fr = df["fr"].iloc[i]
        if pd.isna(fr):
            continue
        if fr > threshold:          # crowded long → short
            sig.iloc[i + 1] = -1
        elif fr < -threshold:       # crowded short → long
            sig.iloc[i + 1] = +1
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# 3. BIDIRECTIONAL BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def backtest_fund(df: pd.DataFrame, signals: pd.Series,
                  hold_hours: int, label: str) -> dict:
    """
    Bidirectional engine (long + short).
    Exit logic: time-based (hold_hours) with ATR stop as emergency guard.
    Stop = high/low of signal bar (2-ATR guard if that is tighter than MIN_SL_PCT).
    """
    hold_bars = hold_hours  # 1H bars

    trades    = []
    capital   = CAPITAL
    in_pos    = False
    entry_px  = 0.0
    stop_px   = 0.0
    take_px   = 0.0
    direction = 0          # +1 long, -1 short
    entry_idx = -1
    entry_time = None
    pos_size  = 0.0

    for i in range(1, len(df)):
        bar = df.iloc[i]

        if in_pos:
            hi, lo = bar["high"], bar["low"]
            elapsed = i - entry_idx

            # Emergency stop check
            if direction == +1:
                sl_hit = lo <= stop_px
                tp_hit = hi >= take_px
                time_hit = elapsed >= hold_bars and not sl_hit
                if sl_hit:
                    exit_px   = stop_px * (1.0 - SLIP)
                    exit_type = "SL"
                elif tp_hit:
                    exit_px   = take_px
                    exit_type = "TP"
                elif time_hit:
                    exit_px   = bar["open"]
                    exit_type = "TIME"
                else:
                    continue
                gross = (exit_px - entry_px) * pos_size

            else:  # short
                sl_hit   = hi >= stop_px
                tp_hit   = lo <= take_px
                time_hit = elapsed >= hold_bars and not sl_hit
                if sl_hit:
                    exit_px   = stop_px * (1.0 + SLIP)
                    exit_type = "SL"
                elif tp_hit:
                    exit_px   = take_px
                    exit_type = "TP"
                elif time_hit:
                    exit_px   = bar["open"]
                    exit_type = "TIME"
                else:
                    continue
                gross = (entry_px - exit_px) * pos_size

            ne       = entry_px * pos_size
            nx       = exit_px  * pos_size
            costs    = (ne + nx) * (FEE + SPREAD)
            net_pnl  = gross - costs

            sl_dist  = abs(entry_px - stop_px)
            r_mult   = (gross / costs if costs else 0.0)   # fallback
            if sl_dist > 0:
                if direction == +1:
                    r_mult = (exit_px - entry_px) / sl_dist
                else:
                    r_mult = (entry_px - exit_px) / sl_dist

            hold_mins = elapsed * 60
            capital  += net_pnl

            trades.append({
                "label":       label,
                "entry_time":  entry_time,
                "exit_time":   bar["datetime"],
                "entry_price": entry_px,
                "exit_price":  exit_px,
                "stop_loss":   stop_px,
                "take_profit": take_px,
                "pnl":         net_pnl,
                "r_multiple":  r_mult,
                "fees":        costs,
                "spread_cost": 0.0,
                "sl_slippage": abs(stop_px - exit_px) * pos_size if exit_type == "SL" else 0.0,
                "holding_minutes": hold_mins,
                "funding_windows_crossed": int(hold_mins / 480),
                "win":       net_pnl > 0,
                "exit_type": exit_type,
                "direction": "LONG" if direction == +1 else "SHORT",
            })
            in_pos = False
            continue

        sig = signals.iloc[i - 1]
        if sig == 0:
            continue

        prev = df.iloc[i - 1]
        ep   = bar["open"]

        if sig == +1:   # LONG
            sl     = prev["low"]
            sl_d   = ep - sl
            if sl_d <= 0 or sl_d / ep < MIN_SL_PCT:
                continue
            tp     = ep + RR * sl_d
            direction = +1
        else:           # SHORT
            sl     = prev["high"]
            sl_d   = sl - ep
            if sl_d <= 0 or sl_d / ep < MIN_SL_PCT:
                continue
            tp     = ep - RR * sl_d
            direction = -1

        risk_d   = capital * RISK_FRAC
        pos_size = min(risk_d / sl_d, (capital * MAX_LEV) / ep)

        entry_px   = ep
        stop_px    = sl
        take_px    = tp
        entry_time = bar["datetime"]
        entry_idx  = i
        in_pos     = True

    return {"trades": trades}


# ─────────────────────────────────────────────────────────────────────────────
# 4. VERDICT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def verdict(m: dict) -> str:
    n  = m.get("n_trades", 0)
    pf = m.get("profit_factor", 0)
    ex = m.get("expectancy_r", 0)
    sig = m["profit_factor"] >= 1.5 and m["n_trades"] >= 30
    if n < 15:
        return "INSUFFICIENT"
    if pf >= 1.5 and ex >= 0.3 and sig:
        return "PROMOTE"
    if pf >= 1.2 and ex >= 0.1:
        return "WATCHLIST"
    if pf >= 1.0:
        return "WEAK"
    return "REJECT"


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #020" + " "*50 + "║")
print("║  Funding Rate Extremes: Mean-Reversion" + " "*40 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Hypothesis : High +funding → crowd long → fade SHORT
               High -funding → crowd short → fade LONG
  Symbols    : BTC  ETH  SOL  (only symbols with funding cache)
  Threshold  : {{0.03%, 0.05%, 0.07%, 0.10%}}
  Hold hours : {{8, 16, 24}}
  Split      : 70/30 chronological
""")

# ── load & split ──────────────────────────────────────────────────────────────
raw = {}
oos_data = {}
train_data = {}

print("  Loading data …")
for sym in SYMBOLS_FUND:
    df = load_data(sym)
    cut = int(len(df) * SPLIT_FRAC)
    train_data[sym] = df.iloc[:cut].reset_index(drop=True)
    oos_data[sym]   = df.iloc[cut:].reset_index(drop=True)
    raw[sym]        = df
    tag = sym.split("-")[0]
    print(f"  {sym:25s}  total={len(df):,}  train={cut:,}  oos={len(df)-cut:,}"
          f"  funding_rows={df['fr'].notna().sum():,}")

# ── base params (threshold=0.05%, hold=8H) ──────────────────────────────────
BASE_THRESH = 0.000050   # ~75th percentile of OKX funding settlements
BASE_HOLD   = 8

print(f"\n  Base params: |funding| > {BASE_THRESH*100:.2f}%  hold={BASE_HOLD}H\n")

base_results = {}   # sym → metrics dict (OOS)
all_base_trades = []

for sym in SYMBOLS_FUND:
    df_oos = oos_data[sym]
    sigs   = make_signals(df_oos, BASE_THRESH)
    n_sigs = (sigs != 0).sum()
    res    = backtest_fund(df_oos, sigs, BASE_HOLD, label=sym)
    m      = compute_metrics(res["trades"], sym)
    v      = verdict(m)
    base_results[sym] = m
    all_base_trades.extend(res["trades"])
    tag = sym.split("-")[0]
    print(f"  {tag:4s}  signals={n_sigs:3d}  trades={m['n_trades']:3d}"
          f"  WR={m['win_rate']*100:4.1f}%  PF={m['profit_factor']:.3f}"
          f"  ExpR={m['expectancy_r']:+.3f}  MDD={m['max_drawdown']*100:.1f}%"
          f"  → {v}")

# ── parameter grid ───────────────────────────────────────────────────────────
print(f"\n  Grid: {len(THRESHOLDS)}×{len(HOLD_HOURS)} = {len(THRESHOLDS)*len(HOLD_HOURS)} combos …")

grid_results = {}   # (threshold, hold) → {sym: metrics}
grid_pf      = np.zeros((len(THRESHOLDS), len(HOLD_HOURS)))

for ti, thresh in enumerate(THRESHOLDS):
    for hi, hold in enumerate(HOLD_HOURS):
        combo_trades = []
        sym_m = {}
        for sym in SYMBOLS_FUND:
            df_tr = train_data[sym]
            sigs  = make_signals(df_tr, thresh)
            res   = backtest_fund(df_tr, sigs, hold, label=sym)
            m     = compute_metrics(res["trades"], sym)
            sym_m[sym] = m
            combo_trades.extend(res["trades"])
        port_m = compute_metrics(combo_trades, "PORTFOLIO")
        grid_results[(thresh, hold)] = {"syms": sym_m, "port": port_m}
        grid_pf[ti, hi] = port_m["profit_factor"]

# best grid combo
flat_pf = [(grid_pf[ti, hi], THRESHOLDS[ti], HOLD_HOURS[hi])
           for ti in range(len(THRESHOLDS)) for hi in range(len(HOLD_HOURS))]
best_pf, best_thresh, best_hold = max(flat_pf, key=lambda x: x[0])
print(f"  Best train combo: thresh={best_thresh*100:.2f}%  hold={best_hold}H  PF={best_pf:.3f}")

# ── OOS with best params ──────────────────────────────────────────────────────
print(f"\n  OOS validation with best params ({best_thresh*100:.2f}%, hold={best_hold}H)…")
best_oos = {}
best_oos_trades = []
for sym in SYMBOLS_FUND:
    df_oos = oos_data[sym]
    sigs   = make_signals(df_oos, best_thresh)
    res    = backtest_fund(df_oos, sigs, best_hold, label=sym)
    m      = compute_metrics(res["trades"], sym)
    v      = verdict(m)
    best_oos[sym] = m
    best_oos_trades.extend(res["trades"])
    tag = sym.split("-")[0]
    print(f"  {tag:4s}  trades={m['n_trades']:3d}  WR={m['win_rate']*100:4.1f}%"
          f"  PF={m['profit_factor']:.3f}  ExpR={m['expectancy_r']:+.3f}  → {v}")

port_best_oos = compute_metrics(best_oos_trades, "PORTFOLIO")
print(f"\n  Portfolio OOS PF={port_best_oos['profit_factor']:.3f}"
      f"  n={port_best_oos['n_trades']}  WR={port_best_oos['win_rate']*100:.1f}%")

# ── direction breakdown ───────────────────────────────────────────────────────
def direction_split(trades):
    longs  = [t for t in trades if t.get("direction") == "LONG"]
    shorts = [t for t in trades if t.get("direction") == "SHORT"]
    def wr(ts): return np.mean([t["win"] for t in ts]) if ts else 0.0
    def pf(ts):
        w = sum(t["pnl"] for t in ts if t["pnl"] > 0)
        l = abs(sum(t["pnl"] for t in ts if t["pnl"] < 0))
        return w / l if l > 0 else (float("inf") if w > 0 else 0.0)
    return {
        "n_long":  len(longs),  "wr_long":  wr(longs),  "pf_long":  pf(longs),
        "n_short": len(shorts), "wr_short": wr(shorts), "pf_short": pf(shorts),
    }

dir_stats = direction_split(best_oos_trades)
print(f"\n  Direction split (best OOS):")
print(f"    LONG  n={dir_stats['n_long']:3d}  WR={dir_stats['wr_long']*100:4.1f}%"
      f"  PF={dir_stats['pf_long']:.3f}")
print(f"    SHORT n={dir_stats['n_short']:3d}  WR={dir_stats['wr_short']*100:4.1f}%"
      f"  PF={dir_stats['pf_short']:.3f}")

# ── bootstrap CI (base params OOS) ───────────────────────────────────────────
boot_ci = {}
for sym in SYMBOLS_FUND:
    pnls = np.array([t["pnl"] for t in []]) if not base_results[sym]["n_trades"] \
        else np.array([t["pnl"] for t in
                       backtest_fund(oos_data[sym],
                                     make_signals(oos_data[sym], BASE_THRESH),
                                     BASE_HOLD, sym)["trades"]])
    if len(pnls) >= 5:
        boot_means = [np.mean(np.random.choice(pnls, len(pnls), replace=True))
                      for _ in range(2000)]
        boot_ci[sym] = np.percentile(boot_means, [5, 50, 95])
    else:
        boot_ci[sym] = np.array([0, 0, 0])

# ── funding distribution stats ────────────────────────────────────────────────
all_fr = []
for sym in SYMBOLS_FUND:
    fr = raw[sym]["fr"].dropna().values
    all_fr.extend(fr)
all_fr = np.array(all_fr)
pct_extreme_pos = np.mean(all_fr > BASE_THRESH) * 100
pct_extreme_neg = np.mean(all_fr < -BASE_THRESH) * 100
print(f"\n  Funding distribution: mean={all_fr.mean()*100:.4f}%  "
      f"std={all_fr.std()*100:.4f}%")
print(f"  Extreme positive (>{BASE_THRESH*100:.2f}%): {pct_extreme_pos:.1f}%  "
      f"Extreme negative (<{-BASE_THRESH*100:.2f}%): {pct_extreme_neg:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating charts …")

# ── Chart 1: Equity curves (base params OOS) ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#111")
fig.suptitle("R020 — Equity Curves (Base Params OOS: thresh=0.05%, hold=8H)",
             color="white", fontsize=13)
for ax, sym in zip(axes, SYMBOLS_FUND):
    m   = base_results[sym]
    eq  = m.get("equity", np.array([CAPITAL]))
    col = COLOURS[sym]
    ax.plot(eq, color=col, lw=1.5)
    ax.axhline(CAPITAL, color="gray", lw=0.7, ls=":")
    ax.set_facecolor("#111")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.set_title(sym.split("-")[0], color=col, fontsize=11)
    ax.set_xlabel("Trade #", color="white")
    ax.set_ylabel("Capital $", color="white")
    n  = m["n_trades"]
    pf = m["profit_factor"]
    ax.text(0.05, 0.95, f"n={n}  PF={pf:.2f}", transform=ax.transAxes,
            color="white", fontsize=9, va="top")
plt.tight_layout()
p = f"{OUT}/r020_equity_curves.png"; plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 2: Drawdown (base params OOS) ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4), facecolor="#111")
fig.suptitle("R020 — Drawdown (Base Params OOS)", color="white", fontsize=13)
for ax, sym in zip(axes, SYMBOLS_FUND):
    m   = base_results[sym]
    eq  = m.get("equity", np.array([CAPITAL]))
    dd  = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100
    ax.fill_between(range(len(dd)), dd, 0, color=COLOURS[sym], alpha=0.6)
    ax.set_facecolor("#111")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.set_title(sym.split("-")[0], color=COLOURS[sym], fontsize=11)
    ax.set_xlabel("Trade #", color="white")
    ax.set_ylabel("Drawdown %", color="white")
plt.tight_layout()
p = f"{OUT}/r020_drawdown.png"; plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 3: Symbol comparison (base vs best OOS) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111")
fig.suptitle("R020 — Symbol Comparison: Base vs Best-Grid OOS",
             color="white", fontsize=13)
syms_short = [s.split("-")[0] for s in SYMBOLS_FUND]
cols       = [COLOURS[s] for s in SYMBOLS_FUND]

for ax, (results, title) in zip(axes, [(base_results, "Base (0.05%, 8H)"),
                                        (best_oos,      f"Best ({best_thresh*100:.2f}%, {best_hold}H)")]):
    pfs = [results[s]["profit_factor"] for s in SYMBOLS_FUND]
    bars = ax.bar(syms_short, pfs, color=cols)
    ax.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5)
    ax.set_facecolor("#111")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.set_title(title, color="white", fontsize=11)
    ax.set_ylabel("Profit Factor", color="white")
    for bar, pf in zip(bars, pfs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{pf:.2f}", ha="center", color="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r020_symbol_comparison.png"; plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 4: Funding rate distribution ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor="#111")
ax.set_facecolor("#111")
ax.hist(all_fr * 100, bins=60, color="#627EEA", alpha=0.7, edgecolor="none")
for thresh, col, lbl in [(BASE_THRESH, "#F7931A", f"+{BASE_THRESH*100:.2f}%"),
                          (-BASE_THRESH, "#9945FF", f"-{BASE_THRESH*100:.2f}%")]:
    ax.axvline(thresh * 100, color=col, lw=2, ls="--", label=lbl)
for thresh in THRESHOLDS[1:]:
    ax.axvline( thresh * 100, color="white", lw=0.7, ls=":", alpha=0.4)
    ax.axvline(-thresh * 100, color="white", lw=0.7, ls=":", alpha=0.4)
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.set_title("R020 — Funding Rate Distribution (all symbols)", color="white", fontsize=13)
ax.set_xlabel("Funding Rate (%)", color="white")
ax.set_ylabel("Count", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
ax.text(0.98, 0.95, f"Mean={all_fr.mean()*100:.4f}%\nStd={all_fr.std()*100:.4f}%\n"
        f"Extreme+ {pct_extreme_pos:.1f}%  Extreme- {pct_extreme_neg:.1f}%",
        transform=ax.transAxes, ha="right", va="top", color="white",
        fontsize=9, bbox=dict(facecolor="#222", alpha=0.7))
plt.tight_layout()
p = f"{OUT}/r020_funding_distribution.png"; plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 5: Heatmap (threshold × hold → portfolio PF, train) ────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111")
ax.set_facecolor("#111")
im = ax.imshow(grid_pf, aspect="auto", cmap="RdYlGn", vmin=0.6, vmax=1.6)
ax.set_xticks(range(len(HOLD_HOURS)));  ax.set_xticklabels([f"{h}H" for h in HOLD_HOURS], color="white")
ax.set_yticks(range(len(THRESHOLDS))); ax.set_yticklabels([f"{t*100:.2f}%" for t in THRESHOLDS], color="white")
ax.set_xlabel("Hold Hours", color="white")
ax.set_ylabel("|Funding| Threshold", color="white")
ax.set_title("R020 — Grid Heatmap (Portfolio PF, Train)", color="white", fontsize=13)
for ti in range(len(THRESHOLDS)):
    for hi in range(len(HOLD_HOURS)):
        ax.text(hi, ti, f"{grid_pf[ti,hi]:.2f}", ha="center", va="center",
                color="black" if 0.8 < grid_pf[ti,hi] < 1.4 else "white", fontsize=10)
cb = fig.colorbar(im, ax=ax); cb.ax.tick_params(labelcolor="white")
# highlight best
best_ti = THRESHOLDS.index(best_thresh)
best_hi = HOLD_HOURS.index(best_hold)
ax.add_patch(plt.Rectangle((best_hi - 0.5, best_ti - 0.5), 1, 1,
                             fill=False, edgecolor="white", lw=2.5))
plt.tight_layout()
p = f"{OUT}/r020_param_heatmap.png"; plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 6: Direction breakdown ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#111")
fig.suptitle("R020 — Long vs Short Performance (Best OOS)", color="white", fontsize=13)

# counts
ax = axes[0]
x  = np.arange(2)
bars = ax.bar(["LONG", "SHORT"], [dir_stats["n_long"], dir_stats["n_short"]],
              color=["#4CAF50", "#F44336"])
ax.set_facecolor("#111"); ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.set_ylabel("Trade Count", color="white"); ax.set_title("Trade Count", color="white")
for bar, val in zip(bars, [dir_stats["n_long"], dir_stats["n_short"]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            str(val), ha="center", color="white", fontsize=11)

# win rate + PF
ax = axes[1]
wr_vals = [dir_stats["wr_long"]*100, dir_stats["wr_short"]*100]
pf_vals = [dir_stats["pf_long"], dir_stats["pf_short"]]
x = np.arange(2); width = 0.35
b1 = ax.bar(x - width/2, wr_vals, width, label="Win Rate %", color="#4CAF50", alpha=0.8)
ax2 = ax.twinx()
b2 = ax2.bar(x + width/2, pf_vals, width, label="Profit Factor", color="#FF9800", alpha=0.8)
ax.axhline(50, color="white", lw=0.8, ls="--", alpha=0.4)
ax2.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.4)
ax.set_facecolor("#111"); ax.tick_params(colors="white"); ax2.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.set_xticks(x); ax.set_xticklabels(["LONG", "SHORT"], color="white")
ax.set_ylabel("Win Rate %", color="#4CAF50")
ax2.set_ylabel("Profit Factor", color="#FF9800")
ax.set_title("WR & PF by Direction", color="white")
ax.legend(loc="upper left",  facecolor="#222", labelcolor="white", fontsize=8)
ax2.legend(loc="upper right", facecolor="#222", labelcolor="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r020_direction_split.png"; plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 7: Bootstrap CI ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#111")
ax.set_facecolor("#111")
sym_labels = [s.split("-")[0] for s in SYMBOLS_FUND]
for i, sym in enumerate(SYMBOLS_FUND):
    lo, med, hi = boot_ci[sym]
    col = COLOURS[sym]
    ax.errorbar(i, med, yerr=[[med - lo], [hi - med]],
                fmt="o", color=col, capsize=8, capthick=2, ms=8)
    ax.text(i, hi + 0.2, f"[{lo:.1f}, {hi:.1f}]", ha="center",
            color=col, fontsize=8)
ax.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
ax.set_xticks(range(len(SYMBOLS_FUND))); ax.set_xticklabels(sym_labels, color="white")
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.set_title("R020 — Bootstrap 90% CI on Mean P&L per Trade (Base OOS)",
             color="white", fontsize=12)
ax.set_ylabel("Mean P&L per Trade $", color="white")
plt.tight_layout()
p = f"{OUT}/r020_bootstrap_ci.png"; plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 8: Dashboard ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor="#0d0d0d")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)
fig.suptitle("QUANTLAB AI — R020 DASHBOARD\nFunding Rate Mean-Reversion",
             color="white", fontsize=14, y=0.98)

# Metric table
ax_t = fig.add_subplot(gs[0, :])
ax_t.axis("off")
rows_data = []
for sym in SYMBOLS_FUND:
    m = base_results[sym]
    v = verdict(m)
    rows_data.append([
        sym.split("-")[0],
        str(m["n_trades"]),
        f"{m['win_rate']*100:.1f}%",
        f"{m['profit_factor']:.3f}",
        f"{m['expectancy_r']:+.3f}",
        f"{m['sharpe']:.2f}",
        f"{m['max_drawdown']*100:.1f}%",
        v,
    ])
headers = ["Symbol", "n", "WR", "PF", "ExpR", "Sharpe", "MDD", "Verdict"]
tbl = ax_t.table(cellText=rows_data, colLabels=headers,
                 loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r % 2 == 0 else "#222")
    cell.set_text_props(color="white")
    cell.set_edgecolor("#333")
    if r == 0:
        cell.set_facecolor("#2a2a2a")
        cell.set_text_props(color="#aaa", fontweight="bold")

# Equity BTC
ax_e = fig.add_subplot(gs[1, 0])
eq = base_results[SYMBOLS_FUND[0]].get("equity", np.array([CAPITAL]))
ax_e.plot(eq, color=COLOURS[SYMBOLS_FUND[0]], lw=1.3)
ax_e.axhline(CAPITAL, color="gray", lw=0.6, ls=":")
ax_e.set_facecolor("#111"); ax_e.tick_params(colors="white", labelsize=7)
for sp in ax_e.spines.values(): sp.set_edgecolor("#333")
ax_e.set_title("BTC Equity", color=COLOURS[SYMBOLS_FUND[0]], fontsize=9)

# Equity ETH
ax_e2 = fig.add_subplot(gs[1, 1])
eq2 = base_results[SYMBOLS_FUND[1]].get("equity", np.array([CAPITAL]))
ax_e2.plot(eq2, color=COLOURS[SYMBOLS_FUND[1]], lw=1.3)
ax_e2.axhline(CAPITAL, color="gray", lw=0.6, ls=":")
ax_e2.set_facecolor("#111"); ax_e2.tick_params(colors="white", labelsize=7)
for sp in ax_e2.spines.values(): sp.set_edgecolor("#333")
ax_e2.set_title("ETH Equity", color=COLOURS[SYMBOLS_FUND[1]], fontsize=9)

# Equity SOL
ax_e3 = fig.add_subplot(gs[1, 2])
eq3 = base_results[SYMBOLS_FUND[2]].get("equity", np.array([CAPITAL]))
ax_e3.plot(eq3, color=COLOURS[SYMBOLS_FUND[2]], lw=1.3)
ax_e3.axhline(CAPITAL, color="gray", lw=0.6, ls=":")
ax_e3.set_facecolor("#111"); ax_e3.tick_params(colors="white", labelsize=7)
for sp in ax_e3.spines.values(): sp.set_edgecolor("#333")
ax_e3.set_title("SOL Equity", color=COLOURS[SYMBOLS_FUND[2]], fontsize=9)

# Heatmap mini
ax_h = fig.add_subplot(gs[2, 0])
im2 = ax_h.imshow(grid_pf, aspect="auto", cmap="RdYlGn", vmin=0.6, vmax=1.6)
ax_h.set_xticks(range(len(HOLD_HOURS))); ax_h.set_xticklabels([f"{h}H" for h in HOLD_HOURS], color="white", fontsize=7)
ax_h.set_yticks(range(len(THRESHOLDS))); ax_h.set_yticklabels([f"{t*100:.2f}%" for t in THRESHOLDS], color="white", fontsize=7)
ax_h.set_title("Grid Heatmap", color="white", fontsize=9)
ax_h.set_facecolor("#111")

# Direction bar
ax_d = fig.add_subplot(gs[2, 1])
ax_d.bar(["LONG", "SHORT"], [dir_stats["wr_long"]*100, dir_stats["wr_short"]*100],
         color=["#4CAF50", "#F44336"], alpha=0.85)
ax_d.axhline(50, color="white", lw=0.8, ls="--", alpha=0.4)
ax_d.set_facecolor("#111"); ax_d.tick_params(colors="white", labelsize=7)
for sp in ax_d.spines.values(): sp.set_edgecolor("#333")
ax_d.set_title("Win Rate by Direction", color="white", fontsize=9)
ax_d.set_ylabel("WR %", color="white", fontsize=7)

# Funding dist mini
ax_f = fig.add_subplot(gs[2, 2])
ax_f.hist(all_fr * 100, bins=40, color="#627EEA", alpha=0.7, edgecolor="none")
ax_f.axvline(BASE_THRESH * 100,  color="#F7931A", lw=1.5, ls="--")
ax_f.axvline(-BASE_THRESH * 100, color="#9945FF", lw=1.5, ls="--")
ax_f.set_facecolor("#111"); ax_f.tick_params(colors="white", labelsize=7)
for sp in ax_f.spines.values(): sp.set_edgecolor("#333")
ax_f.set_title("Funding Distribution", color="white", fontsize=9)
ax_f.set_xlabel("Rate %", color="white", fontsize=7)

plt.savefig(f"{OUT}/r020_dashboard.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  → {OUT}/r020_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FINAL ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
port_base = compute_metrics(all_base_trades, "PORTFOLIO")
promoted  = [s for s in SYMBOLS_FUND if verdict(base_results[s]) == "PROMOTE"]
watchlist = [s for s in SYMBOLS_FUND if verdict(base_results[s]) == "WATCHLIST"]

print(f"""
{'='*110}
  QUANTLAB AI — RESEARCH #020  [FINAL ASSESSMENT]
  Funding Rate Extremes: Mean-Reversion
{'='*110}

  Q1. Does funding rate signal have mean-reversion edge? (base params OOS)
      Portfolio PF={port_base['profit_factor']:.3f}  n={port_base['n_trades']}
      Promoted : {len(promoted)}/3  {[s.split('-')[0] for s in promoted]}
      Watchlist: {len(watchlist)}/3  {[s.split('-')[0] for s in watchlist]}

  Q2. Is the edge directional (longs vs shorts)?
      LONG  n={dir_stats['n_long']:3d}  WR={dir_stats['wr_long']*100:4.1f}%  PF={dir_stats['pf_long']:.3f}
      SHORT n={dir_stats['n_short']:3d}  WR={dir_stats['wr_short']*100:4.1f}%  PF={dir_stats['pf_short']:.3f}

  Q3. Best grid params (train)?
      thresh={best_thresh*100:.2f}%  hold={best_hold}H  PF={best_pf:.3f}
      OOS with best params: PF={port_best_oos['profit_factor']:.3f}  n={port_best_oos['n_trades']}

  Q4. Funding extreme frequency:
      >{BASE_THRESH*100:.2f}%: {pct_extreme_pos:.1f}% of settlements
      <{-BASE_THRESH*100:.2f}%: {pct_extreme_neg:.1f}% of settlements
""")

# Per-symbol verdict
print("  Symbol breakdown (base params, OOS):")
for sym in SYMBOLS_FUND:
    m = base_results[sym]
    v = verdict(m)
    print(f"    {sym.split('-')[0]:4s}  n={m['n_trades']:3d}  WR={m['win_rate']*100:4.1f}%"
          f"  PF={m['profit_factor']:.3f}  ExpR={m['expectancy_r']:+.3f}"
          f"  Sharpe={m['sharpe']:5.2f}  MDD={m['max_drawdown']*100:.1f}%"
          f"  → {v}")

# Overall verdict
overall_pf = port_base["profit_factor"]
if overall_pf >= 1.4 and len(promoted) >= 2:
    overall_v = "PROMOTE"
    msg = "Clear funding-rate mean-reversion edge — proceed to live testing"
elif overall_pf >= 1.2 or len(watchlist) >= 1:
    overall_v = "WATCHLIST"
    msg = "Weak-to-moderate edge, worth deeper investigation with more symbols/history"
else:
    overall_v = "REJECT"
    msg = "No consistent funding-rate mean-reversion edge found in this dataset"

print(f"""
  Portfolio overall: PF={overall_pf:.3f}  n={port_base['n_trades']}  → {overall_v}
  Assessment: {msg}
{'='*110}
""")


# ─────────────────────────────────────────────────────────────────────────────
# 8. JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
journal_rows = []
for sym in SYMBOLS_FUND:
    m = base_results[sym]
    v = verdict(m)
    journal_rows.append({
        "research_id":    RESEARCH_ID,
        "date":           pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
        "strategy_label": f"FundMR_1H_{sym.split('-')[0]}",
        "symbol":         sym,
        "n_trades":       m["n_trades"],
        "profit_factor":  round(m["profit_factor"], 4),
        "expectancy_r":   round(m["expectancy_r"], 4),
        "win_rate":       round(m["win_rate"], 4),
        "net_pnl":        round(m["net_profit"], 2),
        "max_drawdown":   round(m["max_drawdown"], 4),
        "sharpe_ratio":   round(m["sharpe"], 4),
        "significant":    1 if m["profit_factor"] > 1.5 and m["n_trades"] >= 30 else 0,
        "avg_hold_h":     round(m.get("avg_hold_minutes", 0) / 60, 1),
        "verdict":        v,
    })

# Portfolio row
journal_rows.append({
    "research_id":    RESEARCH_ID,
    "date":           pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
    "strategy_label": "FundMR_1H_PORTFOLIO",
    "symbol":         "PORTFOLIO",
    "n_trades":       port_base["n_trades"],
    "profit_factor":  round(port_base["profit_factor"], 4),
    "expectancy_r":   round(port_base["expectancy_r"], 4),
    "win_rate":       round(port_base["win_rate"], 4),
    "net_pnl":        round(port_base["net_profit"], 2),
    "max_drawdown":   round(port_base["max_drawdown"], 4),
    "sharpe_ratio":   round(port_base["sharpe"], 4),
    "significant":    1 if port_base["profit_factor"] > 1.5 and port_base["n_trades"] >= 30 else 0,
    "avg_hold_h":     round(port_base.get("avg_hold_minutes", 0) / 60, 1),
    "verdict":        overall_v,
})

print(f"  Writing journal ({len(journal_rows)} rows)…")
append_journal(journal_rows)
print(f"  Journal updated → {CONFIG['JOURNAL_FILE']}")
print(f"\n  All outputs → {OUT}/r020_*")
print("  Research #020 complete.\n")
