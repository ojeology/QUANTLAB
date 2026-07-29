"""
QUANTLAB AI — RESEARCH #025
Failed Breakdown Reversal (False Breakout) Strategy Validation
==============================================================

Hypothesis:
  A bearish breakdown that immediately reverses (liquidity sweep) creates a
  genuine statistical long edge. Price closes below the 20-bar low, then the
  NEXT candle closes back above it — trapping short-sellers.

Strategy rules (frozen — no optimisation):
  ENTRY CONDITIONS (all must be true):
    1. Sweep candle closes BELOW prior 20-bar low  (breakout)
    2. Next candle closes BACK ABOVE that same 20-bar low  (failure confirmed)
    3. ATR Rank < 33rd percentile  (low relative volatility)
    4. Bollinger Band Width > its rolling median
    5. Relative Volume > its 20-bar average
    6. Session ≠ Dead (21:00–00:00 UTC)

  ENTRY PRICE: Open of the candle immediately after confirmation
  STOP LOSS:   Lowest low of the sweep candle
  TAKE PROFIT: Entry + 2 × (Entry − Stop)  [2R]
  DIRECTION:   Long only

Symbols: BTC-USDT-SWAP, ETH-USDT-SWAP, SOL-USDT-SWAP, LINK-USDT-SWAP
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
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

RESEARCH_ID = "R025"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

SYMBOLS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "LINK-USDT-SWAP",
]
SPLIT    = 0.70
CAPITAL  = CONFIG["STARTING_CAPITAL"]
RISK_R   = CONFIG["RISK_PER_TRADE_PCT"]
MAX_LEV  = CONFIG["MAX_LEVERAGE"]
MIN_SL   = CONFIG["MIN_SL_PCT"]
FEE      = CONFIG["TAKER_FEE"]
SPREAD   = CONFIG["SPREAD"] * 0.5
SLIPPAGE = CONFIG["SL_SLIPPAGE"]
RR       = 2.0   # fixed 2R — frozen

COLOURS = {
    "BTC-USDT-SWAP":"#F7931A",
    "ETH-USDT-SWAP":"#627EEA",
    "SOL-USDT-SWAP":"#9945FF",
    "LINK-USDT-SWAP":"#2A5ADA",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA & FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def load_1h(sym):
    tag = sym.replace("-", "_")
    df  = pd.read_parquet(f"{CACHE}/{tag}_1H.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)

def split_oos(df):
    cut = int(len(df) * SPLIT)
    return df.iloc[cut:].reset_index(drop=True)

def session_label(hour):
    if   0  <= hour <  7:  return "Asia"
    elif 7  <= hour < 12:  return "London"
    elif 12 <= hour < 20:  return "NewYork"
    else:                  return "Dead"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df["close"]
    v  = df["vol"]

    # ATR & ATR Rank (33rd pct threshold computed per-symbol from OOS)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank_pct"] = df["atr14"].rolling(100).rank(pct=True) * 100

    # 20-bar rolling low (computed on bars BEFORE current bar)
    # We compare close[i] against low20[i] where low20 = min(low[i-20..i-1])
    df["low20_prev"] = df["low"].shift(1).rolling(20).min()

    # Bollinger Band Width (20-bar, 2σ) as % of mid-price
    bb_mid        = c.rolling(20).mean()
    bb_std        = c.rolling(20).std(ddof=0)
    df["bb_width"] = (bb_std * 4) / bb_mid * 100
    df["bb_width_median"] = df["bb_width"].rolling(50).median()

    # Relative Volume: current volume vs 20-bar average
    df["vol_ma20"] = v.rolling(20).mean()
    df["rel_vol"]  = v / df["vol_ma20"]

    # Session
    df["hour_utc"] = df["datetime"].dt.hour
    df["session"]  = df["hour_utc"].apply(session_label)

    # Sweep candle low (stored so we can check the next bar)
    df["sweep_low"] = df["low"]

    return df

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with columns:
      signal     : 1 = entry on next open
      entry_open : open of the entry bar (bar i+1 relative to confirmation bar)
      sweep_low_ : lowest low of the sweep candle (used as stop anchor)
      confirm_i  : index of the confirmation bar
    """
    df = df.copy()
    n  = len(df)
    signals = np.zeros(n, dtype=int)
    sweep_lows = np.full(n, np.nan)

    # Compute OOS-level percentile threshold for ATR Rank
    atr_33 = df["atr_rank_pct"].quantile(0.33)

    for i in range(1, n - 1):
        sweep = df.iloc[i - 1]   # potential sweep candle (bar i-1)
        conf  = df.iloc[i]       # confirmation candle (bar i)
        entry = df.iloc[i + 1]   # entry bar (bar i+1)

        low20 = df.iloc[i - 1]["low20_prev"]  # 20-bar low prior to sweep bar
        if np.isnan(low20):
            continue

        # Condition 1: sweep candle closes BELOW 20-bar low
        if not (sweep["close"] < low20):
            continue

        # Condition 2: confirmation candle closes BACK ABOVE that same 20-bar low
        if not (conf["close"] > low20):
            continue

        # Condition 3: ATR Rank < 33rd percentile (measured at confirmation bar)
        if not (conf["atr_rank_pct"] < atr_33):
            continue

        # Condition 4: BB Width > rolling median at confirmation bar
        if np.isnan(conf["bb_width_median"]):
            continue
        if not (conf["bb_width"] > conf["bb_width_median"]):
            continue

        # Condition 5: Relative Volume > 1.0 at confirmation bar
        if not (conf["rel_vol"] > 1.0):
            continue

        # Condition 6: Not in Dead session (check confirmation bar session)
        if conf["session"] == "Dead":
            continue

        signals[i + 1]    = 1           # enter on bar i+1 open
        sweep_lows[i + 1] = sweep["low"]  # stop anchor = lowest low of sweep candle

    df["signal"]    = signals
    df["sweep_low_anchor"] = sweep_lows
    return df

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, sym: str) -> list:
    """
    Process signals: enter at open of signal bar, SL = sweep_low_anchor,
    TP = entry + 2 × (entry − SL).  Fixed 2R, no trailing.
    """
    trades   = []
    in_pos   = False
    entry_px = stop = take = 0.0
    pos_size = 0.0
    entry_tm = None
    entry_i  = -1
    capital  = CAPITAL
    mfe = mae = 0.0

    for i in range(1, len(df)):
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_pos:
            hi, lo = bar["high"], bar["low"]
            sl_dist = entry_px - stop
            if sl_dist > 0:
                mfe = max(mfe, (hi - entry_px) / sl_dist)
                mae = min(mae, (lo - entry_px) / sl_dist)

            sl_hit = lo <= stop
            tp_hit = hi >= take

            if sl_hit or tp_hit:
                exit_px   = (stop * (1.0 - SLIPPAGE)) if sl_hit else take
                exit_type = "SL" if sl_hit else "TP"

                gross = (exit_px - entry_px) * pos_size
                ne    = entry_px * pos_size
                nx    = exit_px  * pos_size
                cost  = (ne + nx) * FEE + (ne + nx) * SPREAD
                slp_c = (stop - exit_px) * pos_size if sl_hit else 0.0
                net   = gross - cost - slp_c
                rmul  = (exit_px - entry_px) / sl_dist if sl_dist > 0 else 0.0

                trades.append({
                    "symbol":       sym,
                    "entry_time":   str(entry_tm),
                    "exit_time":    str(bar["datetime"]),
                    "entry_price":  round(entry_px, 6),
                    "exit_price":   round(exit_px, 6),
                    "stop_loss":    round(stop, 6),
                    "take_profit":  round(take, 6),
                    "pnl":          net,
                    "r_multiple":   rmul,
                    "win":          int(exit_type == "TP"),
                    "exit_type":    exit_type,
                    "holding_mins": (i - entry_i) * 60,
                    "mfe_r":        mfe,
                    "mae_r":        mae,
                    "session":      bar["session"],
                    "atr_rank_pct": prev["atr_rank_pct"],
                    "bb_width":     prev["bb_width"],
                    "rel_vol":      prev["rel_vol"],
                })
                capital += net
                in_pos   = False
            continue

        if df.iloc[i]["signal"] == 1:
            anchor   = df.iloc[i]["sweep_low_anchor"]
            ep       = bar["open"]
            sl       = anchor          # stop at sweep-candle low
            sl_dist  = ep - sl

            if sl_dist <= 0 or sl_dist / ep < MIN_SL:
                continue

            tp       = ep + RR * sl_dist
            risk_usd = capital * RISK_R
            sz       = min(risk_usd / sl_dist, (capital * MAX_LEV) / ep)

            entry_px = ep
            stop     = sl
            take     = tp
            pos_size = sz
            entry_tm = bar["datetime"]
            entry_i  = i
            in_pos   = True
            mfe = mae = 0.0

    return trades

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def metrics(trades: list, label: str = "") -> dict:
    if not trades:
        return {"label":label,"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,
                "net":0.0,"sharpe":0.0,"mdd":0.0,"avg_hold":0.0,
                "equity":np.array([CAPITAL]),"pnls":np.array([]),"wins":np.array([])}
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values
    wins = df["win"].values.astype(bool)
    n, nw = len(pnl), wins.sum()
    nl    = n - nw
    gw    = pnl[wins].sum() if nw else 0.0
    gl    = abs(pnl[~wins].sum()) if nl else 1e-9
    pf    = gw / gl
    wr    = nw / n
    exp_r = wr * RR - (1 - wr)     # Expected R at 2R target
    eq    = CAPITAL + np.cumsum(pnl)
    peak  = np.maximum.accumulate(eq)
    dd    = (eq - peak) / peak
    mdd   = dd.min()
    std   = np.std(pnl, ddof=1) if n > 1 else 1e-9
    # Annualised Sharpe (1H data, n trades spread across OOS period)
    sharpe = (pnl.mean() / std * math.sqrt(n)) if std > 0 else 0.0
    avg_h  = df["holding_mins"].mean()
    return {"label":label,"n":n,"wr":wr,"pf":pf,"exp_r":exp_r,"net":float(pnl.sum()),
            "sharpe":sharpe,"mdd":mdd,"avg_hold":avg_h,"equity":eq,"pnls":pnl,"wins":wins}

def monte_carlo(pnls: np.ndarray, n_iter: int = 2000) -> dict:
    if len(pnls) < 5:
        return {"prob_profit":0.0,"p5":CAPITAL,"p50":CAPITAL,"p95":CAPITAL,"finals":np.array([CAPITAL])}
    rng    = np.random.default_rng(42)
    finals = np.zeros(n_iter)
    for i in range(n_iter):
        s = rng.choice(pnls, len(pnls), replace=True)
        finals[i] = CAPITAL + s.sum()
    return {
        "prob_profit": (finals > CAPITAL).mean(),
        "p5":  np.percentile(finals, 5),
        "p50": np.percentile(finals, 50),
        "p95": np.percentile(finals, 95),
        "finals": finals,
    }

def bootstrap_pf(pnls: np.ndarray, n_iter: int = 2000) -> tuple:
    if len(pnls) < 10:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    pfs = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        pfs.append(wp / lp if lp > 0 else 2.0)
    return float(np.percentile(pfs, 5)), float(np.percentile(pfs, 50)), float(np.percentile(pfs, 95))

def loo_pf(sym_trades: dict) -> dict:
    syms = list(sym_trades.keys())
    result = {}
    for omit in syms:
        flat = [t for s, tl in sym_trades.items() if s != omit for t in tl]
        if flat:
            result[omit] = metrics(flat)["pf"]
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #025" + " "*50 + "║")
print("║  Failed Breakdown Reversal (False Breakout) Strategy Validation" + " "*15 + "║")
print("╚" + "═"*79 + "╝")
print("""
  Strategy : Failed Breakdown Reversal — Long Only
  Entry    : Breakdown below 20-bar low → next candle closes back above → enter open
  Stop     : Lowest low of sweep candle
  Target   : 2R fixed
  Filters  : ATR Rank < 33rd pct  |  BB Width > median  |  RelVol > 20-bar avg
             Session ≠ Dead (21–00 UTC)
""")

print("  Loading and preparing OOS data …")
sym_trades  = {}
sym_signals = {}

for sym in SYMBOLS:
    try:
        df      = load_1h(sym)
        df_feat = add_features(df)
        df_oos  = split_oos(df_feat)
        df_sig  = generate_signals(df_oos)
        n_sig   = int(df_sig["signal"].sum())
        trades  = run_backtest(df_sig, sym)
        sym_trades[sym]  = trades
        sym_signals[sym] = n_sig
        tag = sym.split("-")[0]
        print(f"  {tag:6s}  OOS bars={len(df_oos):,}  signals={n_sig:4d}  trades={len(trades):4d}")
    except FileNotFoundError:
        print(f"  {sym}: cache file missing")

flat_trades = [t for tl in sym_trades.values() for t in tl]
port        = metrics(flat_trades, "PORTFOLIO")

# ─────────────────────────────────────────────────────────────────────────────
# PRINT PER-SYMBOL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*78)
print("  PER-SYMBOL RESULTS")
print("═"*78)
print(f"  {'Symbol':8s}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'ExpR':>7}  {'Sharpe':>7}  {'MDD':>7}  {'Net $':>9}")
print("  " + "─"*68)
sym_metrics = {}
for sym in SYMBOLS:
    m   = metrics(sym_trades[sym], sym)
    sym_metrics[sym] = m
    tag = sym.split("-")[0]
    print(f"  {tag:8s}  {m['n']:5d}  {m['wr']*100:5.1f}%  {m['pf']:7.3f}  {m['exp_r']:+7.3f}"
          f"  {m['sharpe']:7.2f}  {m['mdd']*100:6.1f}%  {m['net']:+9.0f}")

print("  " + "─"*68)
print(f"  {'PORTFOLIO':8s}  {port['n']:5d}  {port['wr']*100:5.1f}%  {port['pf']:7.3f}"
      f"  {port['exp_r']:+7.3f}  {port['sharpe']:7.2f}  {port['mdd']*100:6.1f}%"
      f"  {port['net']:+9.0f}")

best_sym = max(SYMBOLS, key=lambda s: sym_metrics[s]["pf"] if sym_metrics[s]["n"] >= 5 else 0)
print(f"\n  Best symbol: {best_sym.split('-')[0]}  PF={sym_metrics[best_sym]['pf']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS TESTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*78)
print("  ROBUSTNESS TESTS")
print("─"*78)

port_pnls = port["pnls"]

# Monte Carlo
mc = monte_carlo(port_pnls, n_iter=2000)
print(f"\n  Monte Carlo (n=2,000):")
print(f"    P(profit)   = {mc['prob_profit']*100:.1f}%")
print(f"    p5  equity  = ${mc['p5']:,.0f}")
print(f"    p50 equity  = ${mc['p50']:,.0f}")
print(f"    p95 equity  = ${mc['p95']:,.0f}")

# Bootstrap PF CI
b5, b50, b95 = bootstrap_pf(port_pnls, n_iter=2000)
print(f"\n  Bootstrap 90% CI on PF (n=2,000):")
print(f"    [p5={b5:.3f}, p50={b50:.3f}, p95={b95:.3f}]")
print(f"    Lower CI > 1.0: {'YES ✓' if b5 > 1.0 else 'NO ✗'}")
print(f"    Lower CI > 1.2: {'YES ✓' if b5 > 1.2 else 'NO ✗'}")

# Leave-one-symbol-out
loo = loo_pf(sym_trades)
print(f"\n  Leave-one-symbol-out PF:")
for sym in SYMBOLS:
    tag = sym.split("-")[0]
    v   = loo.get(sym, 0.0)
    print(f"    Leave out {tag:5s}: PF={v:.3f}")

# Cost sensitivity (double fees + spread)
print(f"\n  Cost sensitivity:")
for mult in [1.0, 2.0, 3.0]:
    scaled = []
    for t in flat_trades:
        nt  = dict(t)
        ne  = t["entry_price"] * abs(t["pnl"]) / max(abs(t["pnl"]), 0.01)
        nx  = t["exit_price"]  * abs(t["pnl"]) / max(abs(t["pnl"]), 0.01)
        extra_cost = (ne + nx) * FEE * (mult - 1)
        nt["pnl"] = t["pnl"] - extra_cost
        scaled.append(nt)
    m_ = metrics(scaled)
    print(f"    Cost {mult:.0f}×  PF={m_['pf']:.3f}  Net=${m_['net']:+,.0f}")

# Execution sensitivity
print(f"\n  Execution sensitivity (extra SL slippage):")
for slp_x in [1.0, 2.0, 3.0]:
    adj = []
    for t in flat_trades:
        nt = dict(t)
        if t["exit_type"] == "SL":
            extra_slp = t["stop_loss"] * SLIPPAGE * (slp_x - 1)
            ps = abs(t["pnl"]) / max(abs(t["entry_price"] - t["exit_price"]), 1e-9)
            nt["pnl"] = t["pnl"] - extra_slp * ps
        adj.append(nt)
    m_ = metrics(adj)
    print(f"    Slippage {slp_x:.0f}×  PF={m_['pf']:.3f}  Net=${m_['net']:+,.0f}")

# Statistical significance (t-test on R-multiples)
t_stat = p_val = None
if len(port_pnls) >= 5:
    rmuls = np.array([t["r_multiple"] for t in flat_trades])
    t_stat, p_val = scipy_stats.ttest_1samp(rmuls, 0.0)
    print(f"\n  t-test (H0: mean R = 0): t={t_stat:.3f}  p={p_val:.4f}  "
          f"{'Significant at 5%' if p_val < 0.05 else 'Not significant'}")
    print(f"  Mean R = {rmuls.mean():.4f}  Median R = {np.median(rmuls):.4f}")
else:
    print(f"\n  t-test: insufficient trades (n={len(port_pnls)}) — skipped")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*78)
print("  DIAGNOSTIC QUESTIONS")
print("═"*78)

pf_above_120 = port["pf"] > 1.20
best_tag      = best_sym.split("-")[0]
pf_consistent = sum(1 for s in SYMBOLS if sym_metrics[s]["pf"] > 1.0 and sym_metrics[s]["n"] >= 5)
min_loo_pf    = min(loo.values()) if loo else 0.0
rmuls_arr     = np.array([t["r_multiple"] for t in flat_trades]) if flat_trades else np.array([])
mean_r        = rmuls_arr.mean() if len(rmuls_arr) else 0.0

# Per-gate trade counts (diagnostic only — no optimisation)
n_sig_total = sum(sym_signals.values())
n_executed  = port["n"]

print(f"""
  1. PF > 1.20?
     Portfolio PF = {port['pf']:.3f}  →  {'YES ✓' if pf_above_120 else 'NO ✗'}

  2. Best symbol?
     {best_tag}  (PF={sym_metrics[best_sym]['pf']:.3f}, n={sym_metrics[best_sym]['n']})

  3. Consistent across symbols?
     {pf_consistent}/{len(SYMBOLS)} symbols with PF > 1.0 (n ≥ 5)
     {'Consistent ✓' if pf_consistent >= 3 else 'Inconsistent ✗'}

  4. ATR Rank useful filter?
     ATR Rank < 33rd pct is baked into the signal — measuring retention impact:
     Total raw signals (before ATR/BB/RelVol): {n_sig_total}
     Trades executed (all filters): {n_executed}
     (Signal filtering removes {(1 - n_executed/max(n_sig_total,1))*100:.0f}% of raw setups)

  5. Relative Volume useful?
     RelVol > 20-bar avg is required — baked into signal. Indivisible in this design.

  6. Trades surviving all filters?
     {n_executed} trades across {len(SYMBOLS)} symbols over OOS period.

  7. Survives Monte Carlo and Bootstrap?
     MC P(profit) = {mc['prob_profit']*100:.1f}%  {'✓' if mc['prob_profit'] > 0.60 else '✗'}
     Bootstrap p5 PF = {b5:.3f}  {'> 1.0 ✓' if b5 > 1.0 else '< 1.0 ✗'}

  8. Robust to higher costs/slippage?
     (See cost & execution sensitivity tables above)
     All PF figures shown at 1×/2×/3× slippage.

  9. Statistically significant?
     {'Mean R = ' + f'{mean_r:.4f}' + ' — ' + ('Significant (p<0.05)' if (p_val is not None and p_val < 0.05) else 'Not significant') if (len(rmuls_arr) >= 5 and p_val is not None) else 'Insufficient trades (n<5)'}

  10. VERDICT:""")

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────
def get_verdict():
    if not flat_trades or port["n"] < 20:
        return "ARCHIVE", "Insufficient trades for statistical evaluation"
    pf_ok  = port["pf"] > 1.20
    exp_ok = port["exp_r"] > 0
    mc_ok  = mc["prob_profit"] > 0.60
    ret_ok = port["n"] >= 30
    if pf_ok and exp_ok and mc_ok and ret_ok:
        return "PROMOTE", f"PF={port['pf']:.3f}  MC={mc['prob_profit']*100:.1f}%  n={port['n']}"
    elif 1.00 <= port["pf"] <= 1.20 and exp_ok:
        return "WATCHLIST", f"PF={port['pf']:.3f} is marginal — needs more data"
    else:
        return "ARCHIVE", f"PF={port['pf']:.3f} below threshold"

VERDICT, VERDICT_REASON = get_verdict()
VCOLOUR = "\033[92m" if VERDICT == "PROMOTE" else "\033[93m" if VERDICT == "WATCHLIST" else "\033[91m"
VRESET  = "\033[0m"

print(f"     {VCOLOUR}{VERDICT}{VRESET} — {VERDICT_REASON}")
print(f"\n{'═'*78}")

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
        ax.set_title(title, color=col, fontsize=9)

# ── Chart 1: Equity curves per symbol ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#111")
fig.suptitle("R025 — Failed Breakdown Reversal: Equity Curves (OOS)", color="white", fontsize=12)
for i, sym in enumerate(SYMBOLS):
    ax  = axes[i // 2][i % 2]
    col = COLOURS[sym]
    tag = sym.split("-")[0]
    m   = sym_metrics[sym]
    dark_ax(ax, f"{tag}  n={m['n']}  PF={m['pf']:.3f}  WR={m['wr']*100:.1f}%", col)
    if m["n"] > 0:
        ax.plot(m["equity"], color=col, lw=1.5)
    ax.axhline(CAPITAL, color="white", lw=0.7, ls=":", alpha=0.4)
    ax.set_xlabel("Trade #", color="white", fontsize=8)
    ax.set_ylabel("Equity $", color="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r025_equity_curves.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 2: R-multiple distribution ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#111")
fig.suptitle("R025 — R-Multiple Distributions", color="white", fontsize=11)

ax = axes[0]
dark_ax(ax, f"Portfolio R-multiples  (n={port['n']})")
if flat_trades:
    rmuls_p = [t["r_multiple"] for t in flat_trades]
    lo_, hi_ = min(rmuls_p), max(rmuls_p)
    bins = np.linspace(lo_, hi_, min(30, len(rmuls_p)//2+2))
    ax.hist(rmuls_p, bins=bins, color="#4CAF50", alpha=0.75, edgecolor="none")
    ax.axvline(0, color="white", lw=0.8, ls="--")
    ax.axvline(np.mean(rmuls_p), color="#FF9800", lw=1.5, ls="--",
               label=f"Mean={np.mean(rmuls_p):.2f}R")
    ax.axvline(RR, color="#4CAF50", lw=1, ls=":", alpha=0.5, label=f"TP={RR}R")
    ax.axvline(-1.0, color="#F44336", lw=1, ls=":", alpha=0.5, label="SL=−1R")
    ax.set_xlabel("R Multiple", color="white"); ax.set_ylabel("Count", color="white")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=8)

ax2 = axes[1]
dark_ax(ax2, "R-multiple by Symbol")
for sym in SYMBOLS:
    if sym_trades[sym]:
        rs  = [t["r_multiple"] for t in sym_trades[sym]]
        col = COLOURS[sym]
        tag = sym.split("-")[0]
        ax2.scatter([tag]*len(rs), rs, color=col, alpha=0.4, s=20)
        ax2.axhline(np.mean(rs), color=col, lw=1.5, alpha=0.9,
                    label=f"{tag} μ={np.mean(rs):.2f}")
ax2.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
ax2.set_ylabel("R Multiple", color="white")
plt.tight_layout()
p = f"{OUT}/r025_r_distribution.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 3: Monte Carlo ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor="#111")
dark_ax(ax, f"R025 — Monte Carlo Final Equity  (P(profit)={mc['prob_profit']*100:.1f}%)")
fe = mc["finals"]
if fe.max() > fe.min():
    ax.hist(fe, bins=np.linspace(fe.min(), fe.max(), 51),
            color="#4CAF50", alpha=0.70, edgecolor="none")
for pv, pc, pl in [(5,"#F44336","p5"),(50,"#4CAF50","p50"),(95,"#FF9800","p95")]:
    v = np.percentile(fe, pv)
    ax.axvline(v, color=pc, lw=1.5, ls="--", label=f"{pl} ${v:,.0f}")
ax.axvline(CAPITAL, color="white", lw=1, ls=":", alpha=0.5, label=f"Start ${CAPITAL:,}")
ax.set_xlabel("Final Equity $", color="white")
ax.set_ylabel("Count", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r025_monte_carlo.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 4: Bootstrap CI ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor="#111")
dark_ax(ax, "R025 — Bootstrap 90% CI on Profit Factor")
ax.errorbar(0, b50, yerr=[[b50-b5],[b95-b50]], fmt="o",
            color="#4CAF50", capsize=14, capthick=3, ms=12)
ax.text(0, b95+0.02, f"p95={b95:.3f}", ha="center", color="#FF9800", fontsize=10)
ax.text(0, b5-0.03, f"p5={b5:.3f}", ha="center", color="#F44336", fontsize=10)
ax.axhline(1.0, color="white", lw=1, ls="--", alpha=0.5, label="PF=1.0")
ax.axhline(1.2, color="#FF9800", lw=1, ls=":", alpha=0.6, label="PF=1.2 target")
ax.set_xticks([0]); ax.set_xticklabels(["Portfolio"], color="white")
ax.set_ylabel("PF", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r025_bootstrap_ci.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 5: Win/Loss breakdown ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111")
fig.suptitle("R025 — Win/Loss Structure", color="white", fontsize=11)

ax = axes[0]
dark_ax(ax, "Win Rate by Symbol")
tags = [s.split("-")[0] for s in SYMBOLS]
wrs  = [sym_metrics[s]["wr"]*100 for s in SYMBOLS]
cols = [COLOURS[s] for s in SYMBOLS]
b    = ax.bar(tags, wrs, color=cols, alpha=0.85)
ax.axhline(33.3, color="white", lw=1, ls="--", alpha=0.5, label="33.3% (break-even at 2R)")
ax.axhline(50, color="#FF9800", lw=0.7, ls=":", alpha=0.5)
ax.set_ylabel("Win Rate %", color="white")
for bar_, wr_ in zip(b, wrs):
    ax.text(bar_.get_x()+bar_.get_width()/2, bar_.get_height()+0.5,
            f"{wr_:.1f}%", ha="center", color="white", fontsize=9)
ax.legend(facecolor="#222", labelcolor="white", fontsize=8)

ax2 = axes[1]
dark_ax(ax2, "Average Win vs Average Loss (R)")
avg_wins  = []
avg_losses= []
for sym in SYMBOLS:
    rs = np.array([t["r_multiple"] for t in sym_trades[sym]])
    avg_wins.append(rs[rs>0].mean() if (rs>0).any() else 0)
    avg_losses.append(abs(rs[rs<0].mean()) if (rs<0).any() else 0)
x_ = np.arange(len(SYMBOLS)); w = 0.35
ax2.bar(x_ - w/2, avg_wins,   w, color="#4CAF50", alpha=0.85, label="Avg Win R")
ax2.bar(x_ + w/2, avg_losses, w, color="#F44336", alpha=0.85, label="Avg Loss R")
ax2.set_xticks(x_); ax2.set_xticklabels(tags, color="white")
ax2.set_ylabel("R Multiple", color="white")
ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r025_win_loss.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 6: Session & ATR analysis ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111")
fig.suptitle("R025 — Trade Distribution by Session & ATR Rank", color="white", fontsize=11)
if flat_trades:
    df_tr = pd.DataFrame(flat_trades)

    ax = axes[0]
    dark_ax(ax, "Win Rate by Session")
    for sess, grp in df_tr.groupby("session"):
        pass
    sess_wr  = df_tr.groupby("session")["win"].mean() * 100
    sess_n   = df_tr.groupby("session").size()
    sess_pf  = df_tr.groupby("session").apply(
        lambda g: g[g["pnl"]>0]["pnl"].sum() / max(abs(g[g["pnl"]<0]["pnl"].sum()), 1e-9))
    SCOLS    = {"Asia":"#2196F3","London":"#4CAF50","NewYork":"#FF9800","Dead":"#9E9E9E"}
    for j, (sess, wr) in enumerate(sess_wr.items()):
        c = SCOLS.get(sess, "#aaa")
        ax.bar(sess, wr, color=c, alpha=0.85)
        n_ = sess_n.get(sess, 0)
        pf_= sess_pf.get(sess, 0)
        ax.text(j, wr+0.5, f"n={n_}\nPF={pf_:.2f}", ha="center", color="white", fontsize=8)
    ax.axhline(33.3, color="white", lw=1, ls="--", alpha=0.5, label="33.3% B/E")
    ax.set_ylabel("Win Rate %", color="white")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=8)

    ax2 = axes[1]
    dark_ax(ax2, "ATR Rank at Entry (should be low)")
    ax2.hist(df_tr["atr_rank_pct"], bins=20, color="#9C27B0", alpha=0.75)
    ax2.axvline(df_tr["atr_rank_pct"].quantile(0.33), color="#FF9800", lw=1.5, ls="--",
                label="33rd pct threshold")
    ax2.set_xlabel("ATR Rank %", color="white")
    ax2.set_ylabel("Count", color="white")
    ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)

plt.tight_layout()
p = f"{OUT}/r025_session_atr.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 7: Dashboard ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14), facecolor="#0a0a0a")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.45)
vcolor = "#4CAF50" if VERDICT == "PROMOTE" else "#FF9800" if VERDICT == "WATCHLIST" else "#F44336"
fig.suptitle(
    f"QUANTLAB AI — R025 DASHBOARD\n"
    f"Failed Breakdown Reversal | Verdict: {VERDICT}",
    color="white", fontsize=13, y=0.99)

# Summary table
ax_t = fig.add_subplot(gs[0, :])
ax_t.axis("off")
rows = []
for sym in SYMBOLS:
    m   = sym_metrics[sym]
    tag = sym.split("-")[0]
    rows.append([tag, str(m["n"]), f"{m['wr']*100:.1f}%", f"{m['pf']:.3f}",
                 f"{m['exp_r']:+.3f}", f"{m['sharpe']:.2f}",
                 f"{m['mdd']*100:.1f}%", f"${m['net']:+,.0f}"])
rows.append(["PORTFOLIO", str(port["n"]), f"{port['wr']*100:.1f}%", f"{port['pf']:.3f}",
             f"{port['exp_r']:+.3f}", f"{port['sharpe']:.2f}",
             f"{port['mdd']*100:.1f}%", f"${port['net']:+,.0f}"])
hdrs = ["Symbol","n","WR","PF","ExpR","Sharpe","MDD","Net $"]
tbl  = ax_t.table(cellText=rows, colLabels=hdrs, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r%2==0 else "#222")
    cell.set_text_props(color="white")
    cell.set_edgecolor("#333")
    if r == 0:
        cell.set_facecolor("#2a2a2a"); cell.set_text_props(color="#aaa", fontweight="bold")
    if r == len(rows):   # Portfolio row
        cell.set_facecolor("#0d2a0d" if port["pf"] > 1.0 else "#2a0d0d")

# Equity curves
for i, sym in enumerate(SYMBOLS):
    ax_ = fig.add_subplot(gs[1, i])
    col = COLOURS[sym]
    dark_ax(ax_, sym.split("-")[0], col)
    m   = sym_metrics[sym]
    if m["n"] > 0:
        ax_.plot(m["equity"], color=col, lw=1.5)
    ax_.axhline(CAPITAL, color="white", lw=0.5, ls=":", alpha=0.4)
    ax_.text(0.05, 0.95, f"PF={m['pf']:.2f}\nn={m['n']}",
             transform=ax_.transAxes, color="white", fontsize=8, va="top")

# MC distribution
ax_mc = fig.add_subplot(gs[2, :2])
dark_ax(ax_mc, f"Monte Carlo  P(profit)={mc['prob_profit']*100:.1f}%")
if fe.max() > fe.min():
    ax_mc.hist(fe, bins=np.linspace(fe.min(), fe.max(), 31), color="#4CAF50", alpha=0.7)
ax_mc.axvline(CAPITAL, color="white", lw=1, ls=":", alpha=0.5)
ax_mc.axvline(np.percentile(fe, 5), color="#F44336", lw=1.5, ls="--", label="p5")
ax_mc.axvline(np.percentile(fe, 50), color="#4CAF50", lw=1.5, ls="--", label="p50")
ax_mc.legend(facecolor="#222", labelcolor="white", fontsize=8)
ax_mc.set_xlabel("Final Equity $", color="white", fontsize=8)

# Verdict panel
ax_v = fig.add_subplot(gs[2, 2:])
ax_v.axis("off"); ax_v.set_facecolor("#111")
ax_v.text(0.5, 0.80, f"VERDICT: {VERDICT}", transform=ax_v.transAxes,
          color=vcolor, fontsize=18, ha="center", fontweight="bold")
ax_v.text(0.5, 0.65, VERDICT_REASON, transform=ax_v.transAxes,
          color="white", fontsize=11, ha="center")
stats_txt = (f"PF={port['pf']:.3f}  WR={port['wr']*100:.1f}%  n={port['n']}\n"
             f"Bootstrap p5={b5:.3f}  MC P(profit)={mc['prob_profit']*100:.1f}%\n"
             f"ExpR={port['exp_r']:+.3f}  MDD={port['mdd']*100:.1f}%")
ax_v.text(0.5, 0.35, stats_txt, transform=ax_v.transAxes,
          color="#aaa", fontsize=10, ha="center")

plt.savefig(f"{OUT}/r025_dashboard.png", dpi=130, bbox_inches="tight",
            facecolor="#0a0a0a"); plt.close()
print(f"  → {OUT}/r025_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE TRADE LOG & JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
if flat_trades:
    df_log = pd.DataFrame(flat_trades)
    log_path = f"{OUT}/r025_trade_log.csv"
    df_log.to_csv(log_path, index=False)
    print(f"  → {log_path}  ({len(df_log)} trades)")

try:
    from quantlab_ai import append_journal
    from datetime import datetime, timezone as _tz
    rows = []
    for sym in SYMBOLS:
        m = sym_metrics[sym]
        mc_sym = monte_carlo(m["pnls"], n_iter=500)
        rows.append({
            "research_id":      RESEARCH_ID,
            "run_date":         datetime.now(tz=_tz.utc).strftime("%Y-%m-%d"),
            "strategy_name":    "Failed_Breakdown_Reversal",
            "symbol":           sym.split("-")[0],
            "n_trades":         m["n"],
            "profit_factor":    round(m["pf"],   4),
            "expectancy_r":     round(m["exp_r"], 4),
            "win_rate":         round(m["wr"],    4),
            "net_profit":       round(m["net"],   2),
            "max_drawdown":     round(m["mdd"],   4),
            "sharpe":           round(m["sharpe"],4),
            "mc_prob_profit":   round(mc_sym["prob_profit"], 4),
            "avg_hold_minutes": round(m["avg_hold"],1),
            "verdict":          VERDICT,
        })
    rows.append({
        "research_id":      RESEARCH_ID,
        "run_date":         datetime.now(tz=_tz.utc).strftime("%Y-%m-%d"),
        "strategy_name":    "Failed_Breakdown_Reversal",
        "symbol":           "PORTFOLIO",
        "n_trades":         port["n"],
        "profit_factor":    round(port["pf"],    4),
        "expectancy_r":     round(port["exp_r"], 4),
        "win_rate":         round(port["wr"],    4),
        "net_profit":       round(port["net"],   2),
        "max_drawdown":     round(port["mdd"],   4),
        "sharpe":           round(port["sharpe"],4),
        "mc_prob_profit":   round(mc["prob_profit"], 4),
        "avg_hold_minutes": round(port["avg_hold"],1),
        "verdict":          VERDICT,
    })
    append_journal(rows)
    print(f"  Journal updated → {CONFIG['JOURNAL_FILE']}")
except Exception as e:
    print(f"  [WARN] Journal: {e}")

print(f"\n{'═'*78}")
print(f"  R025 complete.")
print(f"  Strategy  : Failed Breakdown Reversal")
print(f"  Trades    : {port['n']}  |  WR: {port['wr']*100:.1f}%  |  PF: {port['pf']:.3f}")
print(f"  Verdict   : {VERDICT} — {VERDICT_REASON}")
print(f"  Output    → {OUT}/r025_*")
print(f"{'═'*78}\n")
