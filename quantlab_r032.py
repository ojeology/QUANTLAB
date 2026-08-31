"""
QUANTLAB AI — RESEARCH #032
BB Width Sweet Spot Validation
==============================

R031 verdict  : VALIDATE_R032
R031 finding  : BB Width ranks #1 in permutation + SHAP importance.
                Narrowest quartile (BBW < p25 of signal bars ≈ 0.013):
                WR=62.5%, PF=1.924.  Q2 zone is a dead zone (PF=0.620).

Hypothesis:
  The FVG + EMA200 Slope + Low ATR strategy only has a true edge when
  Bollinger Band Width(20,2) is in the extreme lower tail (≤ p25), not
  simply below the median.

Four variants (OOS, 9 symbols, 1H, identical engine to R029):
  A  Baseline          — no filters
  B  Low ATR           — ATR Rank < p25 (R029 reference, n=64, PF=1.205)
  C  Low ATR + BB p50  — Low ATR + BBW < pooled-OOS p50
  D  Low ATR + BB p25  — Low ATR + BBW < pooled-OOS p25  ← PRIMARY

BB Width thresholds computed from ALL pooled OOS bars (no look-ahead).

PROMOTE criteria (all must pass for D):
  PF > 1.20  |  n ≥ 80  |  boot p50 > 1.20  |  MC P(profit) > 60%
  ≥6/9 symbols improve vs B  |  Max Drawdown < 20%
"""

import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr

RESEARCH_ID = "R032"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
    "LINK-USDT-SWAP", "AVAX-USDT-SWAP", "XRP-USDT-SWAP",
    "LTC-USDT-SWAP", "BCH-USDT-SWAP", "DOGE-USDT-SWAP",
]
SPLIT   = 0.70
CAPITAL = CONFIG["STARTING_CAPITAL"]

COLOURS = {
    "BTC-USDT-SWAP":  "#F7931A", "ETH-USDT-SWAP":  "#627EEA",
    "SOL-USDT-SWAP":  "#9945FF", "LINK-USDT-SWAP": "#2A5ADA",
    "AVAX-USDT-SWAP": "#E84142", "XRP-USDT-SWAP":  "#346AA9",
    "LTC-USDT-SWAP":  "#BFBBBB", "BCH-USDT-SWAP":  "#8DC351",
    "DOGE-USDT-SWAP": "#C3A634",
}

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #032" + " "*50 + "║")
print("║  BB Width Sweet Spot Validation" + " "*47 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  R029 reference  : Low ATR   PF=1.205  n=64  (failed PROMOTE on n<80, boot p50<1.20)
  R031 finding    : BBW Q1 (< p25 of signal bars): PF=1.924, WR=62.5%
  R032 hypothesis : Low ATR + BBW tight tail (p25) recovers the edge cleanly
  Symbols         : {', '.join(s.split('-')[0] for s in SYMBOLS)}
  Engine          : Identical to R029 — no changes to signal/SL/TP/costs
""")

# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def calc_bb_width(close, length=20, mult=2.0):
    """Bollinger Band Width(20,2) = (Upper - Lower) / Middle."""
    sma  = close.rolling(length).mean()
    std  = close.rolling(length).std(ddof=0)
    upper = sma + mult * std
    lower = sma - mult * std
    bbw  = (upper - lower) / sma.replace(0, np.nan)
    return bbw

def add_features(df):
    df = df.copy()
    c  = df["close"]
    df["ema200"]        = calc_ema(c, 200)
    df["atr14"]         = calc_atr(df, 14)
    df["atr_rank_pct"]  = df["atr14"].rolling(100).rank(pct=True) * 100
    df["ema200_rising"] = df["ema200"] > df["ema200"].shift(10)
    df["high_2ago"]     = df["high"].shift(2)
    df["fvg_gap"]       = df["low"] > df["high_2ago"] * 1.0001
    df["prev_low"]      = df["low"].shift(1)
    df["bb_width"]      = calc_bb_width(c, length=20, mult=2.0)
    return df

def signal_fvg_slope(df):
    return (df["fvg_gap"] &
            (df["close"] > df["ema200"]) &
            df["ema200_rising"] &
            df["high_2ago"].notna()).fillna(False)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

print("  Loading OOS 1H data …")
oos_dfs = {}
for sym in SYMBOLS:
    tag = sym.replace("-", "_")
    try:
        df      = pd.read_parquet(f"{CACHE}/{tag}_1H.parquet")
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
        cut = int(len(df) * SPLIT)
        df_oos = add_features(df).iloc[cut:].reset_index(drop=True)
        oos_dfs[sym] = df_oos
        print(f"  {sym.split('-')[0]:5s}  OOS={len(df_oos):,} bars  "
              f"BBW range={df_oos['bb_width'].min():.4f}–{df_oos['bb_width'].max():.4f}")
    except FileNotFoundError:
        print(f"  {sym}: cache missing — skipped")

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS (pooled OOS — no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

pool = pd.concat(list(oos_dfs.values()), ignore_index=True)

atr_p25 = float(pool["atr_rank_pct"].quantile(0.25))
bbw_p25 = float(pool["bb_width"].dropna().quantile(0.25))
bbw_p50 = float(pool["bb_width"].dropna().quantile(0.50))

# Compute what fraction of bars each BBW gate retains
pct_atr_low       = (pool["atr_rank_pct"] < atr_p25).mean() * 100
pct_bbw_p50       = (pool["bb_width"] < bbw_p50).dropna().mean() * 100 if len(pool) else 0
pct_atr_bbw50     = ((pool["atr_rank_pct"] < atr_p25) & (pool["bb_width"] < bbw_p50)).mean() * 100
pct_atr_bbw25     = ((pool["atr_rank_pct"] < atr_p25) & (pool["bb_width"] < bbw_p25)).mean() * 100

print(f"""
  Pooled OOS thresholds (all bars — no look-ahead):
    ATR Rank < p25    = {atr_p25:.1f}   → {pct_atr_low:.1f}% of bars
    BB Width < p50    = {bbw_p50:.5f}   → {pct_bbw_p50:.1f}% of bars
    BB Width < p25    = {bbw_p25:.5f}
    Low ATR + BBW p50 gate covers {pct_atr_bbw50:.1f}% of bars
    Low ATR + BBW p25 gate covers {pct_atr_bbw25:.1f}% of bars

  Cross-reference with R031 signal-bar BBW:
    R031 trade p25 ≈ 0.01316  |  R031 trade p50 ≈ 0.02036
    Pooled OOS p25 = {bbw_p25:.5f}   |  Pooled OOS p50 = {bbw_p50:.5f}
""")

thresholds = {"atr_p25": atr_p25, "bbw_p50": bbw_p50, "bbw_p25": bbw_p25}

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE (identical to R029)
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df, signal, variant, thr, sym_label):
    """
    variant: "A" baseline | "B" low_atr | "C" low_atr+bbw_p50 | "D" low_atr+bbw_p25
    """
    min_sl    = CONFIG["MIN_SL_PCT"]
    rr        = CONFIG["RISK_REWARD"]
    max_lev   = CONFIG["MAX_LEVERAGE"]
    capital   = CONFIG["STARTING_CAPITAL"]
    risk_frac = CONFIG["RISK_PER_TRADE_PCT"]
    fee_rate  = CONFIG["TAKER_FEE"]
    spd_rate  = CONFIG["SPREAD"] * 0.5
    slp_rate  = CONFIG["SL_SLIPPAGE"]

    in_pos = False
    entry_px = stop = take = pos_size = 0.0
    entry_tm = None; entry_i = -1
    trades = []

    for i in range(1, len(df)):
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_pos:
            sl_hit = bar["low"]  <= stop
            tp_hit = bar["high"] >= take
            if sl_hit or tp_hit:
                exit_px   = (stop * (1.0 - slp_rate)) if sl_hit else take
                exit_type = "SL" if sl_hit else "TP"
                sl_dist   = entry_px - stop
                gross     = (exit_px - entry_px) * pos_size
                ne, nx    = entry_px * pos_size, exit_px * pos_size
                cost      = (ne + nx) * fee_rate + (ne + nx) * spd_rate
                slp_c     = (stop - exit_px) * pos_size if sl_hit else 0.0
                net       = gross - cost - slp_c
                rmul      = (exit_px - entry_px) / sl_dist if sl_dist > 0 else 0.0
                trades.append({
                    "sym":         sym_label,
                    "variant":     variant,
                    "entry_time":  str(entry_tm),
                    "exit_time":   str(bar["datetime"]),
                    "entry_price": entry_px,
                    "exit_price":  exit_px,
                    "stop_loss":   stop,
                    "take_profit": take,
                    "pnl":         net,
                    "r_multiple":  rmul,
                    "win":         int(exit_type == "TP"),
                    "exit_type":   exit_type,
                    "holding_hrs": i - entry_i,
                    "atr_rank_pct":float(prev["atr_rank_pct"]),
                    "bb_width":    float(prev["bb_width"]) if not pd.isna(prev["bb_width"]) else np.nan,
                })
                in_pos = False
            continue

        if signal.iloc[i - 1]:
            atr_pct = prev["atr_rank_pct"]
            bbw_val = prev["bb_width"]
            if pd.isna(atr_pct): continue

            # Variant gates
            if variant in ("B", "C", "D"):
                if atr_pct >= thr["atr_p25"]: continue
            if variant == "C":
                if pd.isna(bbw_val) or bbw_val >= thr["bbw_p50"]: continue
            if variant == "D":
                if pd.isna(bbw_val) or bbw_val >= thr["bbw_p25"]: continue

            ep      = bar["open"]
            sl      = prev["prev_low"]
            sl_dist = ep - sl
            if sl_dist <= 0 or sl_dist / ep < min_sl:
                sl      = prev["low"]
                sl_dist = ep - sl
                if sl_dist <= 0 or sl_dist / ep < min_sl:
                    continue

            tp       = ep + rr * sl_dist
            risk_usd = capital * risk_frac
            sz       = min(risk_usd / sl_dist, (capital * max_lev) / ep)

            entry_px = ep; stop = sl; take = tp
            pos_size = sz; entry_tm = bar["datetime"]; entry_i = i
            in_pos   = True

    return trades

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def metrics(trades, label=""):
    empty = {"label": label, "n": 0, "wr": 0.0, "pf": 0.0, "exp_r": 0.0,
             "net": 0.0, "sharpe": 0.0, "mdd": 0.0,
             "equity": np.array([CAPITAL]), "pnls": np.array([])}
    if not trades: return empty
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values
    wins = df["win"].values.astype(bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw  = pnl[wins].sum()       if nw else 0.0
    gl  = abs(pnl[~wins].sum()) if nl else 1e-9
    pf  = gw / gl
    wr  = nw / n
    eq  = CAPITAL + np.cumsum(pnl)
    mdd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    std = np.std(pnl, ddof=1) if n > 1 else 1e-9
    sh  = pnl.mean() / std * math.sqrt(n) if std > 0 else 0.0
    exp = wr * CONFIG["RISK_REWARD"] - (1 - wr)
    return {"label": label, "n": n, "wr": wr, "pf": pf, "exp_r": exp,
            "net": float(pnl.sum()), "sharpe": sh, "mdd": mdd,
            "equity": eq, "pnls": pnl}

def bootstrap_pf(pnls, n_iter=5000, seed=42):
    if len(pnls) < 5:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    pfs = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        pfs.append(wp / lp if lp > 0 else 2.0)
    return (float(np.percentile(pfs, 5)),
            float(np.percentile(pfs, 50)),
            float(np.percentile(pfs, 95)))

def monte_carlo(pnls, n_iter=5000, seed=42):
    if len(pnls) < 5:
        return {"prob_profit": 0.0, "p5": CAPITAL, "p50": CAPITAL,
                "p95": CAPITAL, "finals": np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit": (finals > CAPITAL).mean(),
            "p5":  float(np.percentile(finals, 5)),
            "p50": float(np.percentile(finals, 50)),
            "p95": float(np.percentile(finals, 95)),
            "finals": finals}

def loo_pf(sym_trades_dict):
    out = {}
    for omit in sym_trades_dict:
        flat = [t for s, tl in sym_trades_dict.items() if s != omit for t in tl]
        m    = metrics(flat)
        out[omit] = {"pf": m["pf"], "n": m["n"]}
    return out

def jackknife_pf(pnls):
    """Leave-one-trade-out PF distribution."""
    if len(pnls) < 5:
        return np.array([])
    jk = []
    for i in range(len(pnls)):
        s  = np.delete(pnls, i)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        jk.append(wp / lp if lp > 0 else 2.0)
    return np.array(jk)

def boot_ci_wr(wins_arr, n_iter=5000, seed=42):
    """Bootstrap CI on win rate."""
    rng = np.random.default_rng(seed)
    wrs = [rng.choice(wins_arr, len(wins_arr), replace=True).mean()
           for _ in range(n_iter)]
    return (float(np.percentile(wrs, 2.5)),
            float(np.percentile(wrs, 50)),
            float(np.percentile(wrs, 97.5)))

# ─────────────────────────────────────────────────────────────────────────────
# RUN BACKTESTS — all 4 variants
# ─────────────────────────────────────────────────────────────────────────────

print("─"*78)
print("  Running backtests — 4 variants × 9 symbols …")
print()

variant_labels = {"A": "Baseline", "B": "Low ATR", "C": "LowATR+BBW p50", "D": "LowATR+BBW p25"}
variant_colors = {"A": "#9E9E9E",  "B": "#4CAF50", "C": "#2196F3",        "D": "#FF9800"}

sym_trades = {v: {} for v in "ABCD"}

for sym in oos_dfs:
    df_oos = oos_dfs[sym]
    sig    = signal_fvg_slope(df_oos)
    tag    = sym.split("-")[0]
    row    = f"  {tag:5s}"
    for v in "ABCD":
        t = run_backtest(df_oos, sig, v, thresholds, sym)
        sym_trades[v][sym] = t
        m = metrics(t)
        row += f"  {v}:n={m['n']:3d} PF={m['pf']:.3f}"
    print(row)

# Portfolio aggregates
port = {v: metrics([t for s in oos_dfs for t in sym_trades[v][s]], variant_labels[v])
        for v in "ABCD"}

base_n = port["A"]["n"]
print(f"\n  {'Variant':16s}  {'n':>5}  {'Retain':>7}  {'WR':>6}  {'PF':>7}  "
      f"{'ExpR':>7}  {'Sharpe':>7}  {'MDD':>7}  {'Net $':>9}")
print("  " + "─"*74)
for v in "ABCD":
    m   = port[v]
    ret = f"{m['n']/max(base_n,1)*100:.0f}%"
    print(f"  {variant_labels[v]:16s}  {m['n']:5d}  {ret:>7}  {m['wr']*100:5.1f}%  "
          f"{m['pf']:7.3f}  {m['exp_r']:+7.3f}  {m['sharpe']:7.2f}  "
          f"{m['mdd']*100:6.1f}%  {m['net']:+9.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# PER-SYMBOL TABLE (variants B, C, D)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  PER-SYMBOL RESULTS — B / C / D vs Baseline")
print("═"*78)

sym_tags = [s.split("-")[0] for s in oos_dfs]
sym_ms   = {v: {sym: metrics(sym_trades[v][sym]) for sym in oos_dfs} for v in "ABCD"}

print(f"\n  {'Symbol':6s}  {'A:n':>5} {'A:PF':>6}  {'B:n':>5} {'B:PF':>6}  "
      f"{'C:n':>5} {'C:PF':>6}  {'D:n':>5} {'D:PF':>6}  {'D:WR':>6}  "
      f"{'δPF D-B':>8}  {'ImprD':>6}")
print("  " + "─"*80)

n_improved_D_vs_B = 0
n_improved_D_vs_A = 0
sym_improve_D_vs_B = []

for sym in oos_dfs:
    tag   = sym.split("-")[0]
    mA, mB, mC, mD = sym_ms["A"][sym], sym_ms["B"][sym], sym_ms["C"][sym], sym_ms["D"][sym]
    dpf_db = mD["pf"] - mB["pf"]
    ok_db  = dpf_db > 0 and mD["n"] > 0
    if ok_db:
        n_improved_D_vs_B += 1
        sym_improve_D_vs_B.append(tag)
    if (mD["pf"] - mA["pf"]) > 0 and mD["n"] > 0:
        n_improved_D_vs_A += 1
    flag = "✓" if ok_db else "✗"
    print(f"  {tag:6s}  {mA['n']:5d} {mA['pf']:6.3f}  {mB['n']:5d} {mB['pf']:6.3f}  "
          f"{mC['n']:5d} {mC['pf']:6.3f}  {mD['n']:5d} {mD['pf']:6.3f}  "
          f"{mD['wr']*100:5.1f}%  {dpf_db:+8.3f}  {flag:>6}")

print(f"\n  D vs B: {n_improved_D_vs_B}/9 symbols improve")
print(f"  D vs A: {n_improved_D_vs_A}/9 symbols improve")
print(f"  Symbols improved (D vs B): {', '.join(sym_improve_D_vs_B) if sym_improve_D_vs_B else 'none'}")

# ─────────────────────────────────────────────────────────────────────────────
# TRADE RETENTION TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Trade Retention Summary:")
print(f"  {'Variant':16s}  {'n':>5}  {'Retention':>10}  {'% of signals taken':>20}")
for v in "ABCD":
    n_ = port[v]["n"]
    base_n_ = port["A"]["n"]
    low_atr_n_ = port["B"]["n"]
    ref_n = base_n_ if v == "A" else (base_n_ if v == "B" else low_atr_n_)
    if v in ("C", "D"):
        ret_vs_b = f"{n_/max(port['B']['n'],1)*100:.1f}% of Low ATR"
    elif v == "B":
        ret_vs_b = f"{n_/max(base_n_,1)*100:.1f}% of Baseline"
    else:
        ret_vs_b = "100%"
    print(f"  {variant_labels[v]:16s}  {n_:5d}  {n_/max(base_n_,1)*100:9.1f}%  {ret_vs_b}")

# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS TESTS (primary: variant D)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  ROBUSTNESS TESTS — Primary Variant: D (Low ATR + BBW p25)")
print("═"*78)

N_BOOT = 5_000
pD  = port["D"]["pnls"]
pB  = port["B"]["pnls"]

# Bootstrap PF — all 4 variants
print(f"\n  Bootstrap PF ({N_BOOT:,} iterations):")
print(f"  {'Variant':16s}  {'p5':>8}  {'p50':>8}  {'p95':>8}  {'p50>1.20':>10}")
boot_results = {}
for v in "ABCD":
    b5, b50, b95 = bootstrap_pf(port[v]["pnls"], N_BOOT)
    boot_results[v] = (b5, b50, b95)
    ok = "✓" if b50 > 1.20 else "✗"
    print(f"  {variant_labels[v]:16s}  {b5:8.3f}  {b50:8.3f}  {b95:8.3f}  {ok:>10}")

b5_D, b50_D, b95_D = boot_results["D"]
b5_B, b50_B, b95_B = boot_results["B"]

# Monte Carlo — all variants
print(f"\n  Monte Carlo ({N_BOOT:,} simulations):")
print(f"  {'Variant':16s}  {'P(profit)':>10}  {'p5 equity':>12}  {'p50 equity':>12}  {'p95 equity':>12}")
mc_results = {}
for v in "ABCD":
    mc = monte_carlo(port[v]["pnls"], N_BOOT)
    mc_results[v] = mc
    ok = "✓" if mc["prob_profit"] > 0.60 else "✗"
    print(f"  {variant_labels[v]:16s}  {mc['prob_profit']*100:9.1f}% {ok}  "
          f"${mc['p5']:11,.0f}  ${mc['p50']:11,.0f}  ${mc['p95']:11,.0f}")

mc_D = mc_results["D"]

# Leave-one-symbol-out — variants B and D
print(f"\n  Leave-One-Symbol-Out:")
print(f"  {'Symbol':7s}  {'LOO PF (B)':>12}  {'LOO PF (C)':>12}  {'LOO PF (D)':>12}  {'D>1.20':>7}")
loo_B = loo_pf(sym_trades["B"])
loo_C = loo_pf(sym_trades["C"])
loo_D = loo_pf(sym_trades["D"])
loo_D_pfs = []
loo_D_robust = True
for sym in oos_dfs:
    tag = sym.split("-")[0]
    pf_B_ = loo_B[sym]["pf"]
    pf_C_ = loo_C[sym]["pf"]
    pf_D_ = loo_D[sym]["pf"]
    loo_D_pfs.append(pf_D_)
    if pf_D_ <= 1.0: loo_D_robust = False
    ok = "✓" if pf_D_ > 1.20 else ("~" if pf_D_ > 1.0 else "✗")
    print(f"  {tag:7s}  {pf_B_:12.3f}  {pf_C_:12.3f}  {pf_D_:12.3f}  {ok:>7}")
print(f"\n  D LOO floor={min(loo_D_pfs):.3f}  max={max(loo_D_pfs):.3f}  "
      f"all>1.0: {'YES ✓' if loo_D_robust else 'NO ✗'}")
print(f"  D LOO all>1.20: {'YES ✓' if min(loo_D_pfs) > 1.20 else f'NO — floor={min(loo_D_pfs):.3f}'}")

# Jackknife (leave-one-trade-out) on variant D PF
jk_D = jackknife_pf(pD)
jk_D_above_1 = (jk_D > 1.0).mean() * 100
jk_D_above_120 = (jk_D > 1.20).mean() * 100
print(f"\n  Jackknife (LOO-trade) on variant D:")
if len(jk_D) > 0:
    print(f"    mean PF={jk_D.mean():.3f}  std={jk_D.std():.3f}  "
          f"min={jk_D.min():.3f}  max={jk_D.max():.3f}")
    print(f"    % jackknife PF > 1.0  : {jk_D_above_1:.0f}%")
    print(f"    % jackknife PF > 1.20 : {jk_D_above_120:.0f}%")
else:
    print("    Insufficient trades for jackknife")

# Bootstrap CI on win rate for variant D
if len(pD) > 0:
    wins_D = [t["win"] for sym in oos_dfs for t in sym_trades["D"][sym]]
    wins_D_arr = np.array(wins_D)
    wr_lo, wr_med, wr_hi = boot_ci_wr(wins_D_arr, N_BOOT)
    print(f"\n  Bootstrap 95% CI on Win Rate (variant D):")
    print(f"    p2.5={wr_lo*100:.1f}%   p50={wr_med*100:.1f}%   p97.5={wr_hi*100:.1f}%")
    wr_above_bep = wr_lo > 1/3.0  # break-even WR at 2R
    print(f"    Entire CI above break-even ({100/3.0:.1f}%): {'YES ✓' if wr_above_bep else 'NO ✗'}")

# SL slippage sensitivity
print(f"\n  SL Slippage sensitivity (variant D):")
flat_D = [t for sym in oos_dfs for t in sym_trades["D"][sym]]
row_pfs_D = []
for mult in [1.0, 2.0, 3.0]:
    adj = []
    for t in flat_D:
        nt = dict(t)
        if t["exit_type"] == "SL":
            extra = t["stop_loss"] * CONFIG["SL_SLIPPAGE"] * (mult - 1)
            ps    = abs(t["pnl"]) / max(abs(t["entry_price"] - t["exit_price"]), 1e-9)
            nt["pnl"] = t["pnl"] - extra * ps
        adj.append(nt)
    row_pfs_D.append(metrics(adj)["pf"])
print(f"  1× = {row_pfs_D[0]:.3f}  |  2× = {row_pfs_D[1]:.3f}  |  3× = {row_pfs_D[2]:.3f}")
stays_profitable = all(p > 1.0 for p in row_pfs_D)
print(f"  Profitable at all slippage multiples: {'YES ✓' if stays_profitable else 'NO ✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  RESEARCH QUESTIONS")
print("═"*78)

mD = port["D"]; mC = port["C"]; mB = port["B"]; mA = port["A"]

q1 = mD["pf"] > mC["pf"]
q2 = mD["pf"] > mB["pf"] and mD["wr"] > mB["wr"]
q3 = mD["pf"] > 1.20 and mD["n"] >= 80
q4 = n_improved_D_vs_B >= 6
q5 = b50_D > 1.20
q6 = mc_D["prob_profit"] > 0.60
q7 = loo_D_robust

print(f"""
  Q1. Does BB Width p25 outperform BB Width median?
      D (p25) PF={mD['pf']:.3f}  WR={mD['wr']*100:.1f}%  n={mD['n']}
      C (p50) PF={mC['pf']:.3f}  WR={mC['wr']*100:.1f}%  n={mC['n']}
      D > C:  {'YES ✓' if q1 else 'NO ✗'}

  Q2. Does BB Width p25 outperform Low ATR alone?
      D (p25)   PF={mD['pf']:.3f}  WR={mD['wr']*100:.1f}%  n={mD['n']}
      B (no BB) PF={mB['pf']:.3f}  WR={mB['wr']*100:.1f}%  n={mB['n']}
      δPF={mD['pf']-mB['pf']:+.3f}  δWR={mD['wr']*100-mB['wr']*100:+.1f}pp
      D > B:    {'YES ✓' if q2 else 'NO ✗'}

  Q3. Does PF exceed 1.20 with at least 80 trades?
      D: PF={mD['pf']:.3f}  n={mD['n']}
      {'YES ✓  Both conditions met' if q3 else
       ('PF OK but n<80 ✗' if mD['pf']>1.20 else
        ('n OK but PF<1.20 ✗' if mD['n']>=80 else 'Both fail ✗'))}

  Q4. Do at least 6/9 symbols improve (D vs B)?
      {n_improved_D_vs_B}/9 symbols improve
      {'YES ✓' if q4 else 'NO ✗'}  ({', '.join(sym_improve_D_vs_B) if sym_improve_D_vs_B else 'none'})

  Q5. Is bootstrap median PF above 1.20?
      D boot p50 = {b50_D:.3f}
      {'YES ✓' if q5 else 'NO ✗'}

  Q6. Is Monte Carlo probability of profit above 60%?
      D MC P(profit) = {mc_D['prob_profit']*100:.1f}%
      {'YES ✓' if q6 else 'NO ✗'}

  Q7. Is the edge robust after leave-one-symbol-out?
      D LOO floor PF = {min(loo_D_pfs):.3f}
      All LOO PF > 1.0: {'YES ✓' if loo_D_robust else 'NO ✗'}
""")

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────

print("═"*78)
print("  PROMOTE CRITERIA & VERDICT")
print("═"*78)

CRITERIA = {
    f"PF > 1.20  (D={mD['pf']:.3f})":         mD["pf"] > 1.20,
    f"n ≥ 80  (D={mD['n']})":                  mD["n"] >= 80,
    f"Boot p50 > 1.20  (p50={b50_D:.3f})":     b50_D > 1.20,
    f"MC P(profit) > 60%  ({mc_D['prob_profit']*100:.1f}%)": mc_D["prob_profit"] > 0.60,
    f"≥6/9 symbols improve  ({n_improved_D_vs_B}/9)": n_improved_D_vs_B >= 6,
    f"Max Drawdown < 20%  ({abs(mD['mdd'])*100:.1f}%)": abs(mD["mdd"]) < 0.20,
}

n_pass = sum(CRITERIA.values())
n_total_crit = len(CRITERIA)

if all(CRITERIA.values()):
    VERDICT = "PROMOTE"
elif n_pass >= 4:
    VERDICT = "WATCHLIST"
else:
    VERDICT = "REJECT"

vmap   = {"PROMOTE": "\033[92m", "WATCHLIST": "\033[93m", "REJECT": "\033[91m"}
vreset = "\033[0m"

print(f"\n  {vmap[VERDICT]}VERDICT: {VERDICT}{vreset}  ({n_pass}/{n_total_crit} criteria met)\n")
for crit, ok in CRITERIA.items():
    print(f"    {'✓' if ok else '✗'} {crit}")

print(f"\n  Key numbers — Variant D (Low ATR + BB Width p25):")
print(f"    n={mD['n']}  PF={mD['pf']:.3f}  WR={mD['wr']*100:.1f}%  "
      f"MDD={mD['mdd']*100:.1f}%  Net=${mD['net']:+,.0f}")
print(f"    Boot: p5={b5_D:.3f}  p50={b50_D:.3f}  p95={b95_D:.3f}")
print(f"    MC: P(profit)={mc_D['prob_profit']*100:.1f}%  "
      f"p50=${mc_D['p50']:,.0f}")
print(f"    LOO floor={min(loo_D_pfs):.3f}")

if VERDICT == "PROMOTE":
    print(f"""
  RECOMMENDATION: Begin paper trading with 0.5× risk.
  ─────────────────────────────────────────────────────────────────────────
  Strategy  : FVG + EMA200 Slope + Low ATR (Rank<p25) + BB Width (< p25)
  Universe  : {', '.join(s.split('-')[0] for s in oos_dfs)}  |  1H  |  Long only
  Entry     : Open of bar after FVG signal
  Stop      : Prior bar low (fallback: current bar low)
  Target    : Entry + 2 × stop_dist
  Size      : 1% risk per trade  |  max 5× leverage
  BBW gate  : BB Width(20,2) at signal bar < {thresholds['bbw_p25']:.5f}  (pooled OOS p25)

  Paper trade for 30 days / 30 trades (whichever comes first).
  Live monitoring: daily equity mark, trade log, no parameter changes.
""")
elif VERDICT == "WATCHLIST":
    failing = [c for c, ok in CRITERIA.items() if not ok]
    print(f"""
  RECOMMENDATION: WATCHLIST — {n_pass}/{n_total_crit} criteria met.
  ─────────────────────────────────────────────────────────────────────────
  Failing criteria: {failing}
  Next step: Diagnose failing criteria before any further filter addition.
  Do NOT add more filters — each filter further reduces n.
""")
else:
    failing = [c for c, ok in CRITERIA.items() if not ok]
    print(f"""
  RECOMMENDATION: REJECT
  ─────────────────────────────────────────────────────────────────────────
  Failing criteria ({len(failing)}): {failing}
  BB Width p25 gate does not improve the strategy sufficiently at this
  sample size. Return to the attribution phase.
""")

print("═"*78)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Generating charts …")

def dark_ax(ax, title=None, col="white"):
    ax.set_facecolor("#111")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    if title: ax.set_title(title, color=col, fontsize=9)

VARIANT_ORDER = ["A", "B", "C", "D"]
VLABELS = [f"A\nBaseline", f"B\nLow ATR", f"C\nLow ATR\n+BBW p50", f"D\nLow ATR\n+BBW p25"]
VCOLORS = [variant_colors[v] for v in VARIANT_ORDER]

# ── Chart 1: PF comparison + Bootstrap CIs ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor="#111")
fig.suptitle("R032 — Variant Comparison: Profit Factor", color="white", fontsize=11)

ax1 = axes[0]; dark_ax(ax1, "Portfolio PF by Variant")
pf_vals  = [port[v]["pf"]  for v in VARIANT_ORDER]
ns_      = [port[v]["n"]   for v in VARIANT_ORDER]
bars_    = ax1.bar(VLABELS, pf_vals, color=VCOLORS, alpha=0.85)
ax1.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5)
ax1.axhline(1.2, color="#FFEB3B", lw=1.0, ls=":", label="Target PF=1.20")
ax1.set_ylabel("Profit Factor", color="white")
ax1.legend(facecolor="#222", labelcolor="white", fontsize=9)
for b_, v_, n_ in zip(bars_, pf_vals, ns_):
    ax1.text(b_.get_x()+b_.get_width()/2, v_+0.01, f"{v_:.3f}",
             ha="center", color="white", fontsize=12, fontweight="bold")
    ax1.text(b_.get_x()+b_.get_width()/2, 0.05, f"n={n_}",
             ha="center", color="white", fontsize=9, transform=ax1.get_xaxis_transform())

ax2 = axes[1]; dark_ax(ax2, "Bootstrap 90% CI on PF (5,000 iterations)")
for xi, v in enumerate(VARIANT_ORDER):
    b5_, b50_, b95_ = boot_results[v]
    ax2.errorbar(xi, b50_, yerr=[[b50_-b5_],[b95_-b50_]],
                 fmt="o", color=variant_colors[v], capsize=12, capthick=2.5, ms=9)
    ax2.text(xi, b95_+0.02, f"{b95_:.3f}", ha="center", color=variant_colors[v], fontsize=8)
    ax2.text(xi, b5_-0.06,  f"{b5_:.3f}",  ha="center", color=variant_colors[v], fontsize=8)
    ax2.text(xi, b50_+0.01, f"{b50_:.3f}", ha="center", color=variant_colors[v], fontsize=9,
             fontweight="bold")
ax2.axhline(1.0, color="white", lw=0.8, ls="--")
ax2.axhline(1.2, color="#FFEB3B", lw=0.8, ls=":")
ax2.set_xticks(range(4)); ax2.set_xticklabels(VLABELS, color="white")
ax2.set_ylabel("PF", color="white")
plt.tight_layout()
p = f"{OUT}/r032_pf_comparison.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 2: Per-symbol PF — variants A, B, D ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(28, 5), facecolor="#111")
fig.suptitle("R032 — Per-Symbol PF: Baseline / Low ATR / Low ATR+BBW p25", color="white", fontsize=11)
for ax_i, (v, col, lbl) in enumerate([("A","#9E9E9E","A: Baseline"),
                                        ("B","#4CAF50","B: Low ATR"),
                                        ("D","#FF9800","D: Low ATR+BBW p25")]):
    ax_ = axes[ax_i]; dark_ax(ax_, lbl, col)
    pfs_ = [sym_ms[v][s]["pf"] for s in oos_dfs]
    ns_  = [sym_ms[v][s]["n"]  for s in oos_dfs]
    cols_s = [COLOURS.get(s, "white") for s in oos_dfs]
    bars__ = ax_.bar(sym_tags, pfs_, color=cols_s, alpha=0.8)
    ax_.axhline(1.0, color="white", lw=0.7, ls="--")
    ax_.axhline(1.2, color="#FFEB3B", lw=0.7, ls=":")
    ax_.set_ylabel("PF", color="white")
    for b__, v_, n_ in zip(bars__, pfs_, ns_):
        ax_.text(b__.get_x()+b__.get_width()/2, v_+0.01, f"{v_:.2f}\nn={n_}",
                 ha="center", color="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r032_per_symbol_pf.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 3: δPF (D vs B) by symbol ─────────────────────────────────────────
dpf_db_sym  = [sym_ms["D"][s]["pf"] - sym_ms["B"][s]["pf"] for s in oos_dfs]
dpf_da_sym  = [sym_ms["D"][s]["pf"] - sym_ms["A"][s]["pf"] for s in oos_dfs]
x_ = np.arange(len(sym_tags)); w_ = 0.38
fig, axes = plt.subplots(1, 2, figsize=(22, 5), facecolor="#111")
fig.suptitle("R032 — δPF by Symbol", color="white", fontsize=11)

ax1 = axes[0]; dark_ax(ax1, "δPF: D − B (Low ATR+BBW p25 vs Low ATR)")
cols_db = ["#4CAF50" if d > 0 else "#F44336" for d in dpf_db_sym]
bars_ = ax1.bar(sym_tags, dpf_db_sym, color=cols_db, alpha=0.85)
ax1.axhline(0, color="white", lw=0.7, ls="--")
ax1.set_ylabel("δPF", color="white")
for b_, d in zip(bars_, dpf_db_sym):
    y_ = d + 0.02 if d >= 0 else d - 0.07
    ax1.text(b_.get_x()+b_.get_width()/2, y_, f"{d:+.3f}", ha="center", color="white", fontsize=9)

ax2 = axes[1]; dark_ax(ax2, "δPF: D − A (Low ATR+BBW p25 vs Baseline)")
cols_da = ["#4CAF50" if d > 0 else "#F44336" for d in dpf_da_sym]
bars_ = ax2.bar(sym_tags, dpf_da_sym, color=cols_da, alpha=0.85)
ax2.axhline(0, color="white", lw=0.7, ls="--")
ax2.set_ylabel("δPF", color="white")
for b_, d in zip(bars_, dpf_da_sym):
    y_ = d + 0.02 if d >= 0 else d - 0.07
    ax2.text(b_.get_x()+b_.get_width()/2, y_, f"{d:+.3f}", ha="center", color="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r032_delta_pf.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 4: Equity curves — all 4 variants ──────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(22, 12), facecolor="#111")
fig.suptitle("R032 — Equity Curves by Variant (9 Symbols, 1H OOS)", color="white", fontsize=11)
for i, v in enumerate(VARIANT_ORDER):
    ax_ = axes[i//2][i%2]
    m_  = port[v]
    col = variant_colors[v]
    dark_ax(ax_, f"{variant_labels[v]}  PF={m_['pf']:.3f}  n={m_['n']}  MDD={m_['mdd']*100:.1f}%", col)
    if m_["n"] > 0:
        ax_.plot(m_["equity"], color=col, lw=1.8)
    ax_.axhline(CAPITAL, color="white", lw=0.5, ls=":", alpha=0.4)
    ax_.text(0.05, 0.95,
             f"WR={m_['wr']*100:.1f}%  ExpR={m_['exp_r']:+.3f}\nNet=${m_['net']:+,.0f}",
             transform=ax_.transAxes, color="white", fontsize=9, va="top")
    ax_.set_xlabel("Trade #", color="white"); ax_.set_ylabel("Equity $", color="white")
plt.tight_layout()
p = f"{OUT}/r032_equity_curves.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 5: Monte Carlo — variants B and D ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 5), facecolor="#111")
fig.suptitle("R032 — Monte Carlo Simulation (5,000 runs)", color="white", fontsize=11)
for ax_i, (v, col, lbl) in enumerate([("B","#4CAF50","B: Low ATR"),
                                        ("D","#FF9800","D: Low ATR+BBW p25")]):
    ax_ = axes[ax_i]; mc_ = mc_results[v]
    dark_ax(ax_, f"{lbl}  P(profit)={mc_['prob_profit']*100:.1f}%", col)
    fe_ = mc_["finals"]
    if fe_.max() > fe_.min():
        ax_.hist(fe_, bins=np.linspace(fe_.min(), fe_.max(), 51), color=col, alpha=0.65)
    for pv_, pc_, pl_ in [(5,"#F44336","p5"),(50,col,"p50"),(95,"#FFEB3B","p95")]:
        val_ = np.percentile(fe_, pv_)
        ax_.axvline(val_, color=pc_, lw=1.5, ls="--", label=f"{pl_} ${val_:,.0f}")
    ax_.axvline(CAPITAL, color="white", lw=1, ls=":", alpha=0.5, label=f"Start ${CAPITAL:,}")
    ax_.set_xlabel("Final Equity $", color="white"); ax_.set_ylabel("Count", color="white")
    ax_.legend(facecolor="#222", labelcolor="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r032_monte_carlo.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 6: LOO-symbol robustness ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 5), facecolor="#111")
fig.suptitle("R032 — Leave-One-Symbol-Out PF", color="white", fontsize=11)
for ax_i, (v, loo_data, col, lbl) in enumerate([
        ("B", loo_B, "#4CAF50", "B: Low ATR"),
        ("D", loo_D, "#FF9800", "D: Low ATR+BBW p25")]):
    ax_  = axes[ax_i]; dark_ax(ax_, lbl, col)
    pfs_ = [loo_data[s]["pf"] for s in oos_dfs]
    bc_  = [col if v_ > 1.20 else ("#8BC34A" if v_ > 1.0 else "#F44336") for v_ in pfs_]
    ax_.bar(sym_tags, pfs_, color=bc_, alpha=0.85)
    ax_.axhline(1.0, color="white", lw=0.8, ls="--")
    ax_.axhline(1.2, color="#FFEB3B", lw=0.8, ls=":")
    for xi_, (x_, y_) in enumerate(zip(sym_tags, pfs_)):
        ax_.text(xi_, y_+0.01, f"{y_:.3f}", ha="center", color="white", fontsize=9)
    ax_.set_ylabel("Portfolio PF (symbol excluded)", color="white")
plt.tight_layout()
p = f"{OUT}/r032_loo_robustness.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 7: Win Rate & Expectancy by variant ────────────────────────────────
wr_vals  = [port[v]["wr"]*100 for v in VARIANT_ORDER]
exp_vals = [port[v]["exp_r"]  for v in VARIANT_ORDER]
mdd_vals = [abs(port[v]["mdd"])*100 for v in VARIANT_ORDER]

fig, axes = plt.subplots(1, 3, figsize=(22, 5), facecolor="#111")
fig.suptitle("R032 — Win Rate / Expectancy / Max Drawdown by Variant", color="white", fontsize=11)

ax1 = axes[0]; dark_ax(ax1, "Win Rate")
bars_ = ax1.bar(VLABELS, wr_vals, color=VCOLORS, alpha=0.85)
ax1.axhline(33.3, color="white", lw=0.8, ls="--", alpha=0.5, label="Break-even")
ax1.axhline(50.0, color="#FF9800", lw=0.8, ls=":", alpha=0.5)
ax1.set_ylabel("Win Rate %", color="white")
ax1.legend(facecolor="#222", labelcolor="white", fontsize=8)
for b_, v_ in zip(bars_, wr_vals):
    ax1.text(b_.get_x()+b_.get_width()/2, v_+0.3, f"{v_:.1f}%", ha="center", color="white", fontsize=9)

ax2 = axes[1]; dark_ax(ax2, "Expectancy R")
cols_e = ["#4CAF50" if e > 0 else "#F44336" for e in exp_vals]
bars_ = ax2.bar(VLABELS, exp_vals, color=cols_e, alpha=0.85)
ax2.axhline(0, color="white", lw=0.8, ls="--")
ax2.set_ylabel("Expectancy R", color="white")
for b_, v_ in zip(bars_, exp_vals):
    ax2.text(b_.get_x()+b_.get_width()/2, v_+0.005, f"{v_:+.3f}", ha="center", color="white", fontsize=9)

ax3 = axes[2]; dark_ax(ax3, "Max Drawdown")
cols_m = ["#4CAF50" if m < 10 else ("#FF9800" if m < 20 else "#F44336") for m in mdd_vals]
bars_ = ax3.bar(VLABELS, mdd_vals, color=cols_m, alpha=0.85)
ax3.axhline(20, color="#F44336", lw=0.8, ls="--", label="20% limit")
ax3.set_ylabel("Max Drawdown %", color="white")
ax3.legend(facecolor="#222", labelcolor="white", fontsize=8)
for b_, v_ in zip(bars_, mdd_vals):
    ax3.text(b_.get_x()+b_.get_width()/2, v_+0.2, f"{v_:.1f}%", ha="center", color="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r032_metrics_comparison.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 8: Jackknife PF distribution (variant D) ──────────────────────────
if len(jk_D) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), facecolor="#111")
    fig.suptitle("R032 — Variant D Robustness: Jackknife + BBW Gate Illustration",
                 color="white", fontsize=11)
    ax1 = axes[0]; dark_ax(ax1, "Jackknife PF (leave-one-trade-out) — Variant D")
    ax1.hist(jk_D, bins=20, color="#FF9800", alpha=0.80)
    ax1.axvline(jk_D.mean(), color="white", lw=2, ls="--", label=f"mean={jk_D.mean():.3f}")
    ax1.axvline(1.0,         color="#F44336", lw=1.2, ls=":",  label="PF=1.0")
    ax1.axvline(1.2,         color="#FFEB3B", lw=1.2, ls=":",  label="PF=1.2")
    ax1.set_xlabel("PF (one trade removed)", color="white")
    ax1.set_ylabel("Count", color="white")
    ax1.text(0.05, 0.95,
             f"{jk_D_above_1:.0f}% > 1.0\n{jk_D_above_120:.0f}% > 1.20",
             transform=ax1.transAxes, color="white", va="top", fontsize=10)
    ax1.legend(facecolor="#222", labelcolor="white", fontsize=9)

    # BBW gate illustration: BBW histogram with threshold lines
    ax2 = axes[1]; dark_ax(ax2, "BBW Distribution: Pooled OOS vs Trade Signal Bars")
    pool_bbw = pool["bb_width"].dropna().values
    trade_bbw_D = np.array([t["bb_width"] for sym in oos_dfs for t in sym_trades["D"][sym]
                             if not np.isnan(t.get("bb_width", np.nan))])
    trade_bbw_B = np.array([t["bb_width"] for sym in oos_dfs for t in sym_trades["B"][sym]
                             if not np.isnan(t.get("bb_width", np.nan))])
    bins_ = np.linspace(0, pool_bbw.clip(0, 0.10).max(), 50)
    ax2.hist(pool_bbw.clip(0, 0.10), bins=bins_, color="#607D8B", alpha=0.40,
             density=True, label="All OOS bars")
    ax2.hist(trade_bbw_B.clip(0, 0.10), bins=bins_, color="#4CAF50", alpha=0.55,
             density=True, label=f"B trades (n={len(trade_bbw_B)})")
    ax2.hist(trade_bbw_D.clip(0, 0.10), bins=bins_, color="#FF9800", alpha=0.75,
             density=True, label=f"D trades (n={len(trade_bbw_D)})")
    ax2.axvline(thresholds["bbw_p25"], color="#FF9800", lw=2, ls="--",
                label=f"p25={thresholds['bbw_p25']:.4f}")
    ax2.axvline(thresholds["bbw_p50"], color="#2196F3", lw=1.5, ls=":",
                label=f"p50={thresholds['bbw_p50']:.4f}")
    ax2.set_xlabel("BB Width(20,2)", color="white")
    ax2.set_ylabel("Density", color="white")
    ax2.legend(facecolor="#222", labelcolor="white", fontsize=9)
    plt.tight_layout()
    p = f"{OUT}/r032_jackknife_bbw.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  → {p}")

# ── Chart 9: Full Dashboard ──────────────────────────────────────────────────
vcolor_map  = {"PROMOTE": "#4CAF50", "WATCHLIST": "#FF9800", "REJECT": "#F44336"}
vcolor_dash = vcolor_map.get(VERDICT, "white")

fig = plt.figure(figsize=(30, 22), facecolor="#0a0a0a")
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.70, wspace=0.45)
fig.suptitle(
    f"QUANTLAB AI — R032 DASHBOARD\n"
    f"BB Width Sweet Spot Validation | Low ATR + BBW p25 | 1H | 9 Symbols | Verdict: {VERDICT}",
    color="white", fontsize=13, y=0.995)

# Row 0: Cross-symbol comparison table (full width)
ax_tbl = fig.add_subplot(gs[0, :]); ax_tbl.axis("off")
tbl_rows = []
for sym in oos_dfs:
    tag = sym.split("-")[0]
    mA_ = sym_ms["A"][sym]; mB_ = sym_ms["B"][sym]
    mC_ = sym_ms["C"][sym]; mD_ = sym_ms["D"][sym]
    dpf_db_ = mD_["pf"] - mB_["pf"]
    tbl_rows.append([
        tag,
        f"{mA_['n']} | {mA_['pf']:.3f}",
        f"{mB_['n']} | {mB_['pf']:.3f}",
        f"{mC_['n']} | {mC_['pf']:.3f}",
        f"{mD_['n']} | {mD_['pf']:.3f}",
        f"{mD_['wr']*100:.0f}%",
        f"{dpf_db_:+.3f}",
        "✓" if dpf_db_ > 0 and mD_["n"] > 0 else "✗",
    ])
hdrs = ["Symbol", "A: n|PF", "B: n|PF", "C: n|PF", "D: n|PF", "D:WR", "δPF D-B", "Impr?"]
tbl  = ax_tbl.table(cellText=tbl_rows, colLabels=hdrs, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r_, c_), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r_ % 2 == 0 else "#222")
    cell.set_text_props(color="white"); cell.set_edgecolor("#333")
    if r_ == 0:
        cell.set_facecolor("#2a2a2a"); cell.set_text_props(color="#aaa", fontweight="bold")

# Row 1: PF bar + Bootstrap CI
ax1 = fig.add_subplot(gs[1, :2]); dark_ax(ax1, "Portfolio PF by Variant")
bars_ = ax1.bar(VLABELS, pf_vals, color=VCOLORS, alpha=0.85)
ax1.axhline(1.0, color="white", lw=0.7, ls="--")
ax1.axhline(1.2, color="#FFEB3B", lw=0.7, ls=":")
for b_, v_, n_ in zip(bars_, pf_vals, ns_):
    ax1.text(b_.get_x()+b_.get_width()/2, v_+0.01, f"{v_:.3f}\nn={n_}",
             ha="center", color="white", fontsize=10, fontweight="bold")

ax2 = fig.add_subplot(gs[1, 2:]); dark_ax(ax2, "Bootstrap 90% CI on PF")
for xi, v in enumerate(VARIANT_ORDER):
    b5_, b50_, b95_ = boot_results[v]
    ax2.errorbar(xi, b50_, yerr=[[b50_-b5_],[b95_-b50_]],
                 fmt="o", color=variant_colors[v], capsize=12, capthick=2.5, ms=9)
    ax2.text(xi, b50_+0.01, f"{b50_:.3f}", ha="center", color=variant_colors[v],
             fontsize=9, fontweight="bold")
ax2.axhline(1.0, color="white", lw=0.7, ls="--")
ax2.axhline(1.2, color="#FFEB3B", lw=0.7, ls=":")
ax2.set_xticks(range(4)); ax2.set_xticklabels(VLABELS, color="white")
ax2.set_ylabel("PF", color="white")

# Row 2: Equity (D) + LOO (D)
ax3 = fig.add_subplot(gs[2, :2])
dark_ax(ax3, f"Equity — Variant D  PF={mD['pf']:.3f}  n={mD['n']}", "#FF9800")
if mD["n"] > 0:
    ax3.plot(mD["equity"], color="#FF9800", lw=2.0, label="D: Low ATR+BBW p25")
if mB["n"] > 0:
    ax3.plot(mB["equity"], color="#4CAF50", lw=1.2, alpha=0.6, label="B: Low ATR")
ax3.axhline(CAPITAL, color="white", lw=0.5, ls=":", alpha=0.4)
ax3.legend(facecolor="#222", labelcolor="white", fontsize=9)
ax3.set_ylabel("Equity $", color="white")

ax4 = fig.add_subplot(gs[2, 2:])
dark_ax(ax4, "Leave-One-Symbol-Out PF (Variant D)", "#FF9800")
pfs_loo_D = [loo_D[s]["pf"] for s in oos_dfs]
bc_loo = ["#FF9800" if v_ > 1.20 else ("#8BC34A" if v_ > 1.0 else "#F44336")
          for v_ in pfs_loo_D]
ax4.bar(sym_tags, pfs_loo_D, color=bc_loo, alpha=0.85)
ax4.axhline(1.0, color="white", lw=0.7, ls="--")
ax4.axhline(1.2, color="#FFEB3B", lw=0.7, ls=":")
for xi_, (x_, y_) in enumerate(zip(sym_tags, pfs_loo_D)):
    ax4.text(xi_, y_+0.01, f"{y_:.2f}", ha="center", color="white", fontsize=9)
ax4.set_ylabel("PF (symbol excluded)", color="white")

# Row 3: Criteria table + Verdict
ax5 = fig.add_subplot(gs[3, :2]); ax5.axis("off")
crit_rows = [[("✓" if ok else "✗"), crit] for crit, ok in CRITERIA.items()]
ctbl = ax5.table(cellText=crit_rows, colLabels=["Pass?", "Criterion"],
                  loc="center", cellLoc="left")
ctbl.auto_set_font_size(False); ctbl.set_fontsize(10)
for (r_, c_), cell in ctbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r_ % 2 == 0 else "#222")
    cell.set_text_props(color="white"); cell.set_edgecolor("#333")
    if r_ == 0:
        cell.set_facecolor("#2a2a2a"); cell.set_text_props(color="#aaa", fontweight="bold")

ax6 = fig.add_subplot(gs[3, 2:]); ax6.axis("off"); ax6.set_facecolor("#111")
ax6.text(0.5, 0.90, f"VERDICT: {VERDICT}", transform=ax6.transAxes,
         color=vcolor_dash, fontsize=22, ha="center", fontweight="bold")
summary = (f"Low ATR + BB Width (20,2) < p25={thresholds['bbw_p25']:.5f}\n"
           f"n={mD['n']}  PF={mD['pf']:.3f}  WR={mD['wr']*100:.1f}%  MDD={mD['mdd']*100:.1f}%\n"
           f"Boot p5/p50/p95: {b5_D:.3f}/{b50_D:.3f}/{b95_D:.3f}\n"
           f"MC P(profit)={mc_D['prob_profit']*100:.1f}%  p50=${mc_D['p50']:,.0f}\n"
           f"LOO floor={min(loo_D_pfs):.3f}  JK>{jk_D_above_120:.0f}%>1.20\n"
           f"Symbols improved (D vs B): {n_improved_D_vs_B}/9\n"
           f"{n_pass}/{n_total_crit} criteria met")
ax6.text(0.5, 0.52, summary, transform=ax6.transAxes,
         color="white", fontsize=9, ha="center", va="center")

plt.savefig(f"{OUT}/r032_dashboard.png", dpi=130, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"  → {OUT}/r032_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE TRADE LOG
# ─────────────────────────────────────────────────────────────────────────────

flat_D_all = [t for sym in oos_dfs for t in sym_trades["D"][sym]]
if flat_D_all:
    path = f"{OUT}/r032_variant_d_trades.csv"
    pd.DataFrame(flat_D_all).to_csv(path, index=False)
    print(f"  → {path}  ({len(flat_D_all)} trades)")

# All variants
all_trades_rows = []
for v in "ABCD":
    for sym in oos_dfs:
        all_trades_rows.extend(sym_trades[v][sym])
if all_trades_rows:
    path = f"{OUT}/r032_all_variants_trades.csv"
    pd.DataFrame(all_trades_rows).to_csv(path, index=False)
    print(f"  → {path}  ({len(all_trades_rows)} total trade records)")

# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────────────

try:
    from quantlab_ai import append_journal
    from datetime import datetime, timezone as _tz
    run_date = datetime.now(tz=_tz.utc).strftime("%Y-%m-%d")
    rows_j = []
    for v in "ABCD":
        m_  = port[v]
        mc_ = monte_carlo(m_["pnls"], n_iter=500)
        b5_, b50_, _ = boot_results[v]
        rows_j.append({
            "research_id":    RESEARCH_ID,
            "run_date":       run_date,
            "strategy_name":  f"FVG+Slope_{variant_labels[v].replace(' ','_')}_1H_9sym",
            "symbol":         "PORTFOLIO_9SYM",
            "n_trades":       m_["n"],
            "profit_factor":  round(m_["pf"],    4),
            "expectancy_r":   round(m_["exp_r"], 4),
            "win_rate":       round(m_["wr"],    4),
            "net_profit":     round(m_["net"],   2),
            "max_drawdown":   round(m_["mdd"],   4),
            "sharpe":         round(m_["sharpe"],4),
            "mc_prob_profit": round(mc_["prob_profit"], 4),
            "avg_hold_minutes": 0,
            "verdict":        VERDICT if v == "D" else f"ref_{variant_labels[v]}",
        })
    append_journal(rows_j)
    print(f"  Journal updated → {CONFIG['JOURNAL_FILE']}")
except Exception as e:
    print(f"  [WARN] Journal: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'═'*78}")
print(f"  R032 complete.")
print(f"  Verdict         : {VERDICT}  ({n_pass}/{n_total_crit} criteria met)")
print(f"\n  {'Variant':16s}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'Boot p50':>9}  "
      f"{'MC P%':>7}  {'MDD':>7}")
print("  " + "─"*62)
for v in "ABCD":
    m_  = port[v]
    b_  = boot_results[v][1]
    mc_ = mc_results[v]["prob_profit"]
    print(f"  {variant_labels[v]:16s}  {m_['n']:5d}  {m_['wr']*100:5.1f}%  "
          f"{m_['pf']:7.3f}  {b_:9.3f}  {mc_*100:6.1f}%  {m_['mdd']*100:6.1f}%")
print(f"\n  BB Width p25 threshold : {thresholds['bbw_p25']:.5f} (pooled OOS)")
print(f"  BB Width p50 threshold : {thresholds['bbw_p50']:.5f} (pooled OOS)")
print(f"  LOO-sym floor (D)      : {min(loo_D_pfs):.3f}")
print(f"  Jackknife > 1.20 (D)   : {jk_D_above_120:.0f}%")
print(f"  Symbols improved D>B   : {n_improved_D_vs_B}/9")
print(f"  Output                 : {OUT}/r032_*")
print(f"{'═'*78}\n")
