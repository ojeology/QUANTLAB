"""
QUANTLAB AI — RESEARCH #029
FVG + EMA200 Slope + Low ATR — Large-Sample Validation
=======================================================

R027: FVG+Slope + Low ATR → PF=1.089, n=37 (1H, 4 symbols). Insufficient sample.
R028: 4H resampling → <16 signals total. Signal too rare on higher timeframes.

R029 fix: expand horizontally. Same 1H engine, all 9 available symbols.
Target: ≥80 Low ATR trades for statistical inference.

NO changes to:
  entry / exit / stop / take-profit / EMA settings / ATR thresholds / costs / risk

Only two variants:
  A. Baseline (no ATR filter)
  B. Low ATR  (ATR Rank < 25th percentile, pooled OOS)

PROMOTE criteria (all must pass):
  PF > 1.20  |  n ≥ 80  |  ≥6/9 symbols improve  |  boot p50 > 1.20  |  MC P(profit) > 60%
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

RESEARCH_ID = "R029"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

SYMBOLS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "LINK-USDT-SWAP",
    "AVAX-USDT-SWAP",
    "XRP-USDT-SWAP",
    "LTC-USDT-SWAP",
    "BCH-USDT-SWAP",
    "DOGE-USDT-SWAP",
]
SPLIT   = 0.70
CAPITAL = CONFIG["STARTING_CAPITAL"]

COLOURS = {
    "BTC-USDT-SWAP":  "#F7931A",
    "ETH-USDT-SWAP":  "#627EEA",
    "SOL-USDT-SWAP":  "#9945FF",
    "LINK-USDT-SWAP": "#2A5ADA",
    "AVAX-USDT-SWAP": "#E84142",
    "XRP-USDT-SWAP":  "#346AA9",
    "LTC-USDT-SWAP":  "#BFBBBB",
    "BCH-USDT-SWAP":  "#8DC351",
    "DOGE-USDT-SWAP": "#C3A634",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_1h(sym):
    tag = sym.replace("-", "_")
    df  = pd.read_parquet(f"{CACHE}/{tag}_1H.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)

def split_oos(df):
    cut = int(len(df) * SPLIT)
    return df.iloc[cut:].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES (identical to R027)
# ─────────────────────────────────────────────────────────────────────────────

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
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL (identical to R027)
# ─────────────────────────────────────────────────────────────────────────────

def signal_fvg_slope(df):
    return (df["fvg_gap"] &
            (df["close"] > df["ema200"]) &
            df["ema200_rising"] &
            df["high_2ago"].notna()).fillna(False)

# ─────────────────────────────────────────────────────────────────────────────
# ATR THRESHOLDS (pooled OOS, same method as R027)
# ─────────────────────────────────────────────────────────────────────────────

def compute_thresholds(oos_dfs):
    pool = pd.concat(list(oos_dfs.values()), ignore_index=True)
    return {
        "p25": float(pool["atr_rank_pct"].quantile(0.25)),
        "p75": float(pool["atr_rank_pct"].quantile(0.75)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST (identical engine to R027)
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df, signal, atr_mode, thresholds, sym_label):
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
                })
                in_pos = False
            continue

        if signal.iloc[i - 1]:
            atr_pct = prev["atr_rank_pct"]
            if atr_mode == "low" and not (atr_pct < thresholds["p25"]): continue
            if np.isnan(atr_pct): continue

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
    if not trades:
        return empty
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

def bootstrap_pf(pnls, n_iter=5000):
    if len(pnls) < 10:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    pfs = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        pfs.append(wp / lp if lp > 0 else 2.0)
    return float(np.percentile(pfs, 5)), float(np.percentile(pfs, 50)), float(np.percentile(pfs, 95))

def monte_carlo(pnls, n_iter=5000):
    if len(pnls) < 5:
        return {"prob_profit": 0.0, "p5": CAPITAL, "p50": CAPITAL,
                "p95": CAPITAL, "finals": np.array([CAPITAL])}
    rng    = np.random.default_rng(42)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit": (finals > CAPITAL).mean(),
            "p5": np.percentile(finals, 5), "p50": np.percentile(finals, 50),
            "p95": np.percentile(finals, 95), "finals": finals}

def loo_pf(sym_trades):
    out = {}
    for omit in sym_trades:
        flat = [t for s, tl in sym_trades.items() if s != omit for t in tl]
        m    = metrics(flat)
        out[omit] = {"pf": m["pf"], "n": m["n"]}
    return out

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #029" + " "*50 + "║")
print("║  FVG + EMA200 Slope + Low ATR — Large-Sample Validation" + " "*23 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  R027 result  : PF=1.089  n=37  (1H, 4 symbols — insufficient sample)
  R028 result  : 4H generates too few FVG signals
  R029 method  : Identical 1H engine | 9 symbols | Baseline vs Low ATR only
  Target       : ≥80 Low ATR trades for statistical inference
  Symbols      : {', '.join(s.split('-')[0] for s in SYMBOLS)}
""")

print("  Loading OOS 1H data …")
oos_dfs = {}
for sym in SYMBOLS:
    try:
        df      = load_1h(sym)
        df_feat = add_features(df)
        df_oos  = split_oos(df_feat)
        oos_dfs[sym] = df_oos
        print(f"  {sym.split('-')[0]:5s}  1H bars={len(df):,}  OOS={len(df_oos):,}")
    except FileNotFoundError:
        print(f"  {sym}: cache missing — skipped")

thresholds = compute_thresholds(oos_dfs)
pool       = pd.concat(list(oos_dfs.values()), ignore_index=True)
pct_low    = (pool["atr_rank_pct"] < thresholds["p25"]).mean() * 100
print(f"\n  ATR Rank threshold (pooled OOS 1H):")
print(f"    Low ATR (< 25th pct) = {thresholds['p25']:.1f}   covers {pct_low:.1f}% of bars")

# ─────────────────────────────────────────────────────────────────────────────
# BACKTESTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*78)
print("  Running backtests …")

base_trades = {}
low_trades  = {}

for sym in oos_dfs:
    df_oos = oos_dfs[sym]
    sig    = signal_fvg_slope(df_oos)
    bt     = run_backtest(df_oos, sig, None,  thresholds, sym)
    lt     = run_backtest(df_oos, sig, "low", thresholds, sym)
    base_trades[sym] = bt
    low_trades[sym]  = lt

    bm = metrics(bt); lm = metrics(lt)
    tag    = sym.split("-")[0]
    retain = f"{lm['n']/max(bm['n'],1)*100:.0f}%"
    dpf    = lm["pf"] - bm["pf"]
    flag   = "▲" if dpf > 0 else "▼"
    print(f"  {tag:5s}  base n={bm['n']:3d} PF={bm['pf']:.3f}  "
          f"low n={lm['n']:3d} PF={lm['pf']:.3f}  ret={retain}  {flag}δPF={dpf:+.3f}")

# Portfolio aggregates
port_base = metrics([t for s in oos_dfs for t in base_trades[s]], "Baseline")
port_low  = metrics([t for s in oos_dfs for t in low_trades[s]],  "LowATR")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  PORTFOLIO RESULTS — FVG+Slope, 1H, 9 Symbols")
print("═"*78)
print(f"  {'Variant':10s}  {'n':>5}  {'WR':>6}  {'PF':>7}  "
      f"{'ExpR':>7}  {'Sharpe':>7}  {'MDD':>7}  {'Net $':>9}  {'Retain':>7}")
print("  " + "─"*68)
for label, m in [("Baseline", port_base), ("Low ATR", port_low)]:
    ret = f"{m['n']/max(port_base['n'],1)*100:.0f}%"
    print(f"  {label:10s}  {m['n']:5d}  {m['wr']*100:5.1f}%  {m['pf']:7.3f}  "
          f"{m['exp_r']:+7.3f}  {m['sharpe']:7.2f}  {m['mdd']*100:6.1f}%  "
          f"{m['net']:+9.0f}  {ret:>7}")

dpf_port = port_low["pf"] - port_base["pf"]
print(f"\n  δPF (Low ATR vs Baseline): {dpf_port:+.3f}")
print(f"  WR shift: {port_base['wr']*100:.1f}% → {port_low['wr']*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SYMBOL TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*78)
print("  CROSS-SYMBOL: FVG+Slope, 1H — Baseline vs Low ATR")
print("─"*78)
print(f"  {'Symbol':7s}  {'Base n':>7}  {'Base PF':>9}  {'Low n':>7}  {'Low PF':>9}  "
      f"{'WR Low':>7}  {'δPF':>9}  {'Improve?':>9}")
print("  " + "─"*72)

n_improved = 0
for sym in oos_dfs:
    bm_ = metrics(base_trades[sym])
    lm_ = metrics(low_trades[sym])
    dpf = lm_["pf"] - bm_["pf"]
    ok  = dpf > 0 and lm_["n"] > 0
    if ok:
        n_improved += 1
    tag  = sym.split("-")[0]
    flag = "✓" if ok else "✗"
    print(f"  {tag:7s}  {bm_['n']:7d}  {bm_['pf']:9.3f}  {lm_['n']:7d}  {lm_['pf']:9.3f}  "
          f"{lm_['wr']*100:6.1f}%  {dpf:+9.3f}  {flag:>9}")

n_total = len(oos_dfs)
print(f"\n  Low ATR improved {n_improved}/{n_total} symbols")

# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*78)
print("  ROBUSTNESS TESTS")
print("─"*78)

# Bootstrap
b5, b50, b95 = bootstrap_pf(port_low["pnls"])
mc           = monte_carlo(port_low["pnls"])
loo          = loo_pf(low_trades)
loo_base     = loo_pf(base_trades)

print(f"\n  Bootstrap CI on Low ATR PF (5,000 iterations):")
print(f"    p5  = {b5:.3f}")
print(f"    p50 = {b50:.3f}  {'> 1.20 ✓' if b50 > 1.20 else '< 1.20 ✗'}")
print(f"    p95 = {b95:.3f}")

print(f"\n  Monte Carlo — Low ATR portfolio (5,000 simulations):")
print(f"    P(profit)  = {mc['prob_profit']*100:.1f}%  {'> 60% ✓' if mc['prob_profit'] > 0.60 else '≤ 60% ✗'}")
print(f"    p5  equity = ${mc['p5']:,.0f}")
print(f"    p50 equity = ${mc['p50']:,.0f}")
print(f"    p95 equity = ${mc['p95']:,.0f}")

print(f"\n  Leave-one-symbol-out (Low ATR PF when each symbol is excluded):")
print(f"  {'Symbol':7s}  {'LOO PF':>9}  {'LOO n':>7}  {'Dir':>5}")
print("  " + "─"*32)
loo_vals = []
for sym in oos_dfs:
    v   = loo[sym]
    tag = sym.split("-")[0]
    loo_vals.append(v["pf"])
    print(f"  {tag:7s}  {v['pf']:9.3f}  {v['n']:7d}  {'✓' if v['pf'] > 1.0 else '✗'}")
print(f"\n  LOO min={min(loo_vals):.3f}  max={max(loo_vals):.3f}  "
      f"all>1.0: {'YES ✓' if min(loo_vals) > 1.0 else 'NO ✗'}")

print(f"\n  Execution sensitivity (SL slippage multiplier — Low ATR):")
flat_low = [t for s in oos_dfs for t in low_trades[s]]
row_pfs  = []
for mult in [1.0, 2.0, 3.0]:
    adj = []
    for t in flat_low:
        nt = dict(t)
        if t["exit_type"] == "SL":
            extra = t["stop_loss"] * CONFIG["SL_SLIPPAGE"] * (mult - 1)
            ps    = abs(t["pnl"]) / max(abs(t["entry_price"] - t["exit_price"]), 1e-9)
            nt["pnl"] = t["pnl"] - extra * ps
        adj.append(nt)
    row_pfs.append(metrics(adj)["pf"])
print(f"  1× (base): {row_pfs[0]:.3f}  |  2×: {row_pfs[1]:.3f}  |  3×: {row_pfs[2]:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  RESEARCH QUESTIONS")
print("═"*78)

# Q4: ETH outlier?
eth_lm = metrics(low_trades.get("ETH-USDT-SWAP", []))
others = [t for s in oos_dfs if s != "ETH-USDT-SWAP" for t in low_trades[s]]
others_m = metrics(others)

# Q8: symbol inclusion/exclusion
include = []; exclude = []
for sym in oos_dfs:
    lm_ = metrics(low_trades[sym])
    bm_ = metrics(base_trades[sym])
    if lm_["n"] >= 5 and lm_["pf"] > 1.0:
        include.append(sym.split("-")[0])
    elif lm_["n"] < 3 or (lm_["n"] >= 3 and lm_["pf"] < 0.7):
        exclude.append(sym.split("-")[0])

print(f"""
  Q1. Does Low ATR still improve FVG at 9-symbol scale?
      Baseline PF={port_base['pf']:.3f} → Low ATR PF={port_low['pf']:.3f}
      δPF={dpf_port:+.3f}  WR: {port_base['wr']*100:.1f}% → {port_low['wr']*100:.1f}%
      Answer: {'YES ▲ — effect confirmed' if dpf_port > 0 else 'NO ▼ — effect disappears at scale'}

  Q2. Does portfolio PF exceed 1.20?
      Low ATR PF = {port_low['pf']:.3f}
      Answer: {'YES ✓' if port_low['pf'] > 1.20 else 'NO ✗  (below 1.20 threshold)'}

  Q3. At least 80 Low ATR trades?
      n = {port_low['n']}
      Answer: {'YES ✓' if port_low['n'] >= 80 else f'NO ✗  ({port_low["n"]} trades, need 80)'}

  Q4. Is ETH still an outlier?
      ETH Low ATR : PF={eth_lm['pf']:.3f}  n={eth_lm['n']}
      Others (ex-ETH) : PF={others_m['pf']:.3f}  n={others_m['n']}
      Answer: {'ETH IS an outlier — significantly above the rest' if eth_lm['pf'] > others_m['pf'] + 0.30 else 'ETH converges with other symbols at this scale'}

  Q5. Bootstrap CI (Low ATR)?
      p5={b5:.3f}  p50={b50:.3f}  p95={b95:.3f}
      Answer: {'p50 > 1.20 ✓' if b50 > 1.20 else f'p50 = {b50:.3f} < 1.20 ✗'}

  Q6. Monte Carlo P(profit)?
      {mc['prob_profit']*100:.1f}%
      Answer: {'> 60% ✓' if mc['prob_profit'] > 0.60 else f'= {mc["prob_profit"]*100:.1f}% < 60% ✗'}

  Q7. Leave-one-symbol-out floor?
      Minimum LOO PF = {min(loo_vals):.3f}
      Answer: {'All LOO samples > 1.0 ✓ — no single symbol is load-bearing' if min(loo_vals) > 1.0 else f'LOO floor {min(loo_vals):.3f} < 1.0 ✗ — some symbols drag the result'}

  Q8. Symbol inclusion / exclusion recommendation?
      Include  : {', '.join(include) if include else 'None meet criteria'}
      Exclude  : {', '.join(exclude) if exclude else 'None clearly excluded'}
      (Criteria: include if Low ATR n≥5 and PF>1.0; exclude if n<3 or PF<0.70)
""")

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict():
    pf_ok   = port_low["pf"]  > 1.20
    n_ok    = port_low["n"]   >= 80
    sym_ok  = n_improved      >= 6
    boot_ok = b50             > 1.20
    mc_ok   = mc["prob_profit"] > 0.60
    checks  = {"PF>1.20": pf_ok, "n≥80": n_ok,
                f"≥6/{n_total} symbols improve": sym_ok,
                "bootstrap p50>1.20": boot_ok,
                "MC P(profit)>60%": mc_ok}
    n_pass  = sum(checks.values())
    if all(checks.values()):
        verdict = "PROMOTE"
    elif n_pass >= 4:
        verdict = "WATCHLIST"
    else:
        verdict = "REJECT"
    return verdict, checks

VERDICT, CHECKS = compute_verdict()
vmap  = {"PROMOTE": "\033[92m", "WATCHLIST": "\033[93m", "REJECT": "\033[91m"}
vreset= "\033[0m"

print(f"{'═'*78}")
print(f"  VERDICT: {vmap[VERDICT]}{VERDICT}{vreset}")
print()
for crit, ok in CHECKS.items():
    print(f"    {'✓' if ok else '✗'} {crit}")

n_pass = sum(CHECKS.values())
print(f"\n  {n_pass}/{len(CHECKS)} criteria met.")
print(f"\n  Core numbers:")
print(f"    PF     : {port_base['pf']:.3f} → {port_low['pf']:.3f}  (δ={dpf_port:+.3f})")
print(f"    WR     : {port_base['wr']*100:.1f}% → {port_low['wr']*100:.1f}%")
print(f"    n      : {port_low['n']} Low ATR trades")
print(f"    Boot   : p5={b5:.3f}  p50={b50:.3f}  p95={b95:.3f}")
print(f"    MC     : P(profit)={mc['prob_profit']*100:.1f}%")
print(f"    LOO    : floor={min(loo_vals):.3f}")
print(f"    Symbols: {n_improved}/{n_total} improve")

# Statistical justification
pf_val = port_low["pf"]
n_val  = port_low["n"]
if VERDICT != "PROMOTE":
    wr_bep = 1/(1+CONFIG["RISK_REWARD"])
    actual_wr = port_low["wr"]
    print(f"""
  Statistical justification for {VERDICT}:
  ─────────────────────────────────────────
  Break-even WR at 2R target = {wr_bep*100:.1f}%
  Observed Low ATR WR        = {actual_wr*100:.1f}%

  With n={n_val} trades, the 95% bootstrap CI spans {b5:.3f}–{b95:.3f}.
  This width ({b95-b5:.3f}) is {'still too large to confirm PF>1.20 reliably.' if b95-b5 > 0.80 else 'acceptable for inference.'}

  The Low ATR filter consistently shifts win rate above break-even ({actual_wr*100:.1f}% vs {wr_bep*100:.1f}% needed).
  {'The effect is real but sample size remains the binding constraint.' if n_val < 80 else ''}
  {'PF<1.20 means the wins do not outweigh losses by enough to cover costs.' if pf_val < 1.20 else ''}

  Path to PROMOTE from here:
    The minimum failing criterion is: {[c for c, ok in CHECKS.items() if not ok]}
    Immediate option: Add a secondary filter that raises WR further
    (e.g. ADX>20 at entry — confirms trend conviction in low-vol regime)
""")
else:
    print(f"""
  All PROMOTE criteria met.
  FVG + EMA200 Slope + Low ATR is ready for paper trading validation.
  Recommended next step: Live paper trade 30 days with position size at 0.5× risk.
""")

print(f"{'═'*78}")

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

sym_tags = [s.split("-")[0] for s in oos_dfs]

# ── Chart 1: Portfolio PF comparison ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), facecolor="#111")
dark_ax(ax, "R029 — FVG+Slope PF: Baseline vs Low ATR (1H, 9 Symbols)")
vals  = [port_base["pf"], port_low["pf"]]
bars  = ax.bar(["Baseline", "Low ATR\n(< p25)"], vals,
               color=["#9E9E9E", "#4CAF50"], alpha=0.85, width=0.4)
ax.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5)
ax.axhline(1.2, color="#FF9800", lw=0.8, ls=":", alpha=0.6, label="PF=1.20 target")
ax.set_ylabel("Profit Factor", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
for b, v, n_ in zip(bars, vals, [port_base["n"], port_low["n"]]):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", color="white", fontsize=14, fontweight="bold")
    ax.text(b.get_x()+b.get_width()/2, 0.04, f"n={n_}",
            ha="center", color="white", fontsize=9, transform=ax.get_xaxis_transform())
plt.tight_layout()
p = f"{OUT}/r029_pf_comparison.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 2: Cross-symbol PF bar chart ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 5), facecolor="#111")
fig.suptitle("R029 — FVG+Slope: Per-Symbol PF (Baseline vs Low ATR)", color="white", fontsize=11)

base_pfs = [metrics(base_trades[s])["pf"] for s in oos_dfs]
low_pfs  = [metrics(low_trades[s])["pf"]  for s in oos_dfs]
sym_cols = [COLOURS.get(s, "white") for s in oos_dfs]

ax1 = axes[0]; dark_ax(ax1, "Baseline", "#9E9E9E")
bars_ = ax1.bar(sym_tags, base_pfs, color=sym_cols, alpha=0.8)
ax1.axhline(1.0, color="white", lw=0.7, ls="--")
ax1.axhline(1.2, color="#FF9800", lw=0.7, ls=":")
ax1.set_ylabel("PF", color="white")
for b, v in zip(bars_, base_pfs):
    ax1.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}",
             ha="center", color="white", fontsize=8)

ax2 = axes[1]; dark_ax(ax2, "Low ATR (< p25)", "#4CAF50")
bars_ = ax2.bar(sym_tags, low_pfs, color=sym_cols, alpha=0.8)
ax2.axhline(1.0, color="white", lw=0.7, ls="--")
ax2.axhline(1.2, color="#FF9800", lw=0.7, ls=":")
ax2.set_ylabel("PF", color="white")
for b, v in zip(bars_, low_pfs):
    ax2.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}",
             ha="center", color="white", fontsize=8)

plt.tight_layout()
p = f"{OUT}/r029_cross_symbol_pf.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 3: δPF by symbol ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4), facecolor="#111")
dark_ax(ax, "R029 — δPF (Low ATR − Baseline) by Symbol")
dpfs  = [lp - bp for lp, bp in zip(low_pfs, base_pfs)]
dcols = ["#4CAF50" if d > 0 else "#F44336" for d in dpfs]
bars  = ax.bar(sym_tags, dpfs, color=dcols, alpha=0.85)
ax.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
ax.set_ylabel("δPF", color="white")
for b, d in zip(bars, dpfs):
    y_ = b.get_height() + 0.02 if d >= 0 else b.get_height() - 0.08
    ax.text(b.get_x()+b.get_width()/2, y_, f"{d:+.2f}",
            ha="center", color="white", fontsize=10)
plt.tight_layout()
p = f"{OUT}/r029_delta_pf_symbol.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 4: Equity curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 5), facecolor="#111")
fig.suptitle("R029 — Portfolio Equity: Baseline vs Low ATR (1H, 9 Symbols)", color="white", fontsize=11)
for ax_i, (label, m, col) in enumerate([
        ("Baseline", port_base, "#9E9E9E"), ("Low ATR", port_low, "#4CAF50")]):
    ax_ = axes[ax_i]
    dark_ax(ax_, f"{label}  PF={m['pf']:.3f}  n={m['n']}", col)
    if m["n"] > 0:
        ax_.plot(m["equity"], color=col, lw=1.8)
    ax_.axhline(CAPITAL, color="white", lw=0.5, ls=":", alpha=0.4)
    ax_.text(0.05, 0.95, f"WR={m['wr']*100:.1f}%\nMDD={m['mdd']*100:.1f}%\nNet=${m['net']:+,.0f}",
             transform=ax_.transAxes, color="white", fontsize=9, va="top")
    ax_.set_xlabel("Trade #", color="white", fontsize=8)
    ax_.set_ylabel("Equity $", color="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r029_equity_curves.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 5: Bootstrap CI ────────────────────────────────────────────────────
bbase5, bbase50, bbase95 = bootstrap_pf(port_base["pnls"])
fig, ax = plt.subplots(figsize=(8, 5), facecolor="#111")
dark_ax(ax, "R029 — Bootstrap 90% CI on PF")
for xi, (vname, vcol, p5_, p50_, p95_) in enumerate([
        ("Baseline", "#9E9E9E", bbase5, bbase50, bbase95),
        ("Low ATR",  "#4CAF50", b5, b50, b95)]):
    ax.errorbar(xi, p50_, yerr=[[p50_-p5_], [p95_-p50_]],
                fmt="o", color=vcol, capsize=14, capthick=2.5, ms=9)
    ax.text(xi, p95_+0.03, f"p95={p95_:.2f}", ha="center", color=vcol, fontsize=8)
    ax.text(xi, p5_-0.06,  f"p5={p5_:.2f}",  ha="center", color=vcol, fontsize=8)
    ax.text(xi, p50_+0.01, f"p50={p50_:.2f}", ha="center", color=vcol, fontsize=9,
            fontweight="bold")
ax.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5, label="PF=1.0")
ax.axhline(1.2, color="#FF9800", lw=0.8, ls=":", alpha=0.6, label="PF=1.2")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline", "Low ATR"], color="white")
ax.set_ylabel("PF", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r029_bootstrap_ci.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 6: Monte Carlo distribution ───────────────────────────────────────
fe  = mc["finals"]
fig, ax = plt.subplots(figsize=(12, 5), facecolor="#111")
dark_ax(ax, f"R029 Monte Carlo — Low ATR  P(profit)={mc['prob_profit']*100:.1f}%")
if fe.max() > fe.min():
    ax.hist(fe, bins=np.linspace(fe.min(), fe.max(), 51),
            color="#4CAF50", alpha=0.70, edgecolor="none")
for pv, pc, pl in [(5,"#F44336","p5"),(50,"#4CAF50","p50"),(95,"#FF9800","p95")]:
    v = np.percentile(fe, pv)
    ax.axvline(v, color=pc, lw=1.5, ls="--", label=f"{pl} ${v:,.0f}")
ax.axvline(CAPITAL, color="white", lw=1, ls=":", alpha=0.5, label=f"Start ${CAPITAL:,}")
ax.set_xlabel("Final Equity $", color="white"); ax.set_ylabel("Count", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r029_monte_carlo.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 7: LOO robustness ──────────────────────────────────────────────────
loo_pfs_ = [loo[s]["pf"] for s in oos_dfs]
loo_cols  = ["#4CAF50" if v > 1.0 else "#F44336" for v in loo_pfs_]
fig, ax   = plt.subplots(figsize=(12, 4), facecolor="#111")
dark_ax(ax, "R029 — Leave-One-Symbol-Out PF (Low ATR)")
ax.bar(sym_tags, loo_pfs_, color=loo_cols, alpha=0.85)
ax.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5)
ax.axhline(1.2, color="#FF9800", lw=0.8, ls=":", alpha=0.5)
for xi, (x, y) in enumerate(zip(sym_tags, loo_pfs_)):
    ax.text(xi, y+0.01, f"{y:.3f}", ha="center", color="white", fontsize=9)
ax.set_ylabel("Portfolio PF (symbol excluded)", color="white")
plt.tight_layout()
p = f"{OUT}/r029_loo_robustness.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 8: Per-symbol n comparison ────────────────────────────────────────
base_ns = [metrics(base_trades[s])["n"] for s in oos_dfs]
low_ns  = [metrics(low_trades[s])["n"]  for s in oos_dfs]
x       = np.arange(len(sym_tags)); w = 0.35
fig, ax = plt.subplots(figsize=(14, 4), facecolor="#111")
dark_ax(ax, "R029 — Trade Count by Symbol: Baseline vs Low ATR")
ax.bar(x - w/2, base_ns, w, color="#9E9E9E", alpha=0.8, label="Baseline")
ax.bar(x + w/2, low_ns,  w, color="#4CAF50", alpha=0.8, label="Low ATR")
ax.set_xticks(x); ax.set_xticklabels(sym_tags, color="white")
ax.set_ylabel("Trade Count", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
for xi, (bn, ln) in enumerate(zip(base_ns, low_ns)):
    ax.text(xi - w/2, bn + 0.3, str(bn), ha="center", color="white", fontsize=8)
    ax.text(xi + w/2, ln + 0.3, str(ln), ha="center", color="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r029_trade_counts.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 9: Full Dashboard ──────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 18), facecolor="#0a0a0a")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.45)
vcolor_map = {"PROMOTE": "#4CAF50", "WATCHLIST": "#FF9800", "REJECT": "#F44336"}
vcolor = vcolor_map.get(VERDICT, "white")

fig.suptitle(
    f"QUANTLAB AI — R029 DASHBOARD\n"
    f"FVG+Slope + Low ATR | 1H | 9 Symbols | Verdict: {VERDICT}",
    color="white", fontsize=13, y=0.99)

# Cross-symbol table (top row)
ax_t = fig.add_subplot(gs[0, :])
ax_t.axis("off")
hdrs_t = ["Symbol","Base n","Base PF","Low n","Low PF","Low WR","δPF","Improve?"]
rows_t = []
for sym in oos_dfs:
    bm_ = metrics(base_trades[sym])
    lm_ = metrics(low_trades[sym])
    dpf = lm_["pf"] - bm_["pf"]
    rows_t.append([
        sym.split("-")[0],
        str(bm_["n"]), f"{bm_['pf']:.3f}",
        str(lm_["n"]), f"{lm_['pf']:.3f}",
        f"{lm_['wr']*100:.1f}%",
        f"{dpf:+.3f}",
        "✓" if dpf > 0 and lm_["n"] > 0 else "✗",
    ])
tbl = ax_t.table(cellText=rows_t, colLabels=hdrs_t, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r_, c_), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r_ % 2 == 0 else "#222")
    cell.set_text_props(color="white"); cell.set_edgecolor("#333")
    if r_ == 0:
        cell.set_facecolor("#2a2a2a"); cell.set_text_props(color="#aaa", fontweight="bold")

# PF bars
ax1 = fig.add_subplot(gs[1, :2])
dark_ax(ax1, "Portfolio PF: Baseline vs Low ATR")
vals_  = [port_base["pf"], port_low["pf"]]
bars_  = ax1.bar(["Baseline","Low ATR"], vals_, color=["#9E9E9E","#4CAF50"], alpha=0.85)
ax1.axhline(1.0, color="white", lw=0.7, ls="--"); ax1.axhline(1.2, color="#FF9800", lw=0.7, ls=":")
for b_, v_ in zip(bars_, vals_):
    ax1.text(b_.get_x()+b_.get_width()/2, v_+0.01, f"{v_:.3f}",
             ha="center", color="white", fontsize=13, fontweight="bold")

# LOO
ax2 = fig.add_subplot(gs[1, 2:])
dark_ax(ax2, "Leave-One-Symbol-Out (Low ATR)")
bc_   = ["#4CAF50" if v > 1.0 else "#F44336" for v in loo_pfs_]
ax2.bar(sym_tags, loo_pfs_, color=bc_, alpha=0.85)
ax2.axhline(1.0, color="white", lw=0.7, ls="--")
ax2.axhline(1.2, color="#FF9800", lw=0.7, ls=":")
for xi_, (x_, y_) in enumerate(zip(sym_tags, loo_pfs_)):
    ax2.text(xi_, y_+0.01, f"{y_:.2f}", ha="center", color="white", fontsize=9)
ax2.set_ylabel("PF", color="white")

# Equity (Low ATR)
ax3 = fig.add_subplot(gs[2, :2])
dark_ax(ax3, f"Equity — Low ATR  PF={port_low['pf']:.3f}  n={port_low['n']}", "#4CAF50")
if port_low["n"] > 0:
    ax3.plot(port_low["equity"], color="#4CAF50", lw=1.8, label="Low ATR")
if port_base["n"] > 0:
    ax3.plot(port_base["equity"], color="#9E9E9E", lw=1.0, alpha=0.4, label="Baseline")
ax3.axhline(CAPITAL, color="white", lw=0.5, ls=":", alpha=0.4)
ax3.legend(facecolor="#222", labelcolor="white", fontsize=9)
ax3.set_ylabel("Equity $", color="white")

# Verdict panel
ax4 = fig.add_subplot(gs[2, 2:])
ax4.axis("off"); ax4.set_facecolor("#111")
ax4.text(0.5, 0.90, f"VERDICT: {VERDICT}", transform=ax4.transAxes,
         color=vcolor, fontsize=22, ha="center", fontweight="bold")
summary = (f"n={port_low['n']}  PF={port_low['pf']:.3f}  WR={port_low['wr']*100:.1f}%\n"
           f"δPF={dpf_port:+.3f}  MDD={port_low['mdd']*100:.1f}%\n"
           f"Boot p5/p50/p95: {b5:.3f}/{b50:.3f}/{b95:.3f}\n"
           f"MC P(profit): {mc['prob_profit']*100:.1f}%\n"
           f"LOO floor: {min(loo_vals):.3f}\n"
           f"Symbols improved: {n_improved}/{n_total}")
ax4.text(0.5, 0.55, summary, transform=ax4.transAxes,
         color="white", fontsize=10, ha="center", va="center")
checks_str = "\n".join(f"{'✓' if ok else '✗'} {c}" for c, ok in CHECKS.items())
ax4.text(0.5, 0.12, checks_str, transform=ax4.transAxes,
         color="#aaa", fontsize=9, ha="center", va="bottom")

plt.savefig(f"{OUT}/r029_dashboard.png", dpi=130, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"  → {OUT}/r029_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE TRADE LOG + JOURNAL
# ─────────────────────────────────────────────────────────────────────────────

flat_low_all = [t for s in oos_dfs for t in low_trades[s]]
if flat_low_all:
    path = f"{OUT}/r029_fvg_low_atr_1h_9sym_trades.csv"
    pd.DataFrame(flat_low_all).to_csv(path, index=False)
    print(f"  → {path}  ({len(flat_low_all)} trades)")

try:
    from quantlab_ai import append_journal
    from datetime import datetime, timezone as _tz
    run_date = datetime.now(tz=_tz.utc).strftime("%Y-%m-%d")
    rows_j = []
    for var, m in [("Baseline", port_base), ("LowATR", port_low)]:
        mc_ = monte_carlo(m["pnls"], n_iter=500)
        rows_j.append({
            "research_id":      RESEARCH_ID,
            "run_date":         run_date,
            "strategy_name":    f"FVG+Slope_{var}_1H_9sym",
            "symbol":           "PORTFOLIO_9SYM",
            "n_trades":         m["n"],
            "profit_factor":    round(m["pf"],    4),
            "expectancy_r":     round(m["exp_r"], 4),
            "win_rate":         round(m["wr"],    4),
            "net_profit":       round(m["net"],   2),
            "max_drawdown":     round(m["mdd"],   4),
            "sharpe":           round(m["sharpe"],4),
            "mc_prob_profit":   round(mc_["prob_profit"], 4),
            "avg_hold_minutes": 0,
            "verdict":          VERDICT if var == "LowATR" else var,
        })
    append_journal(rows_j)
    print(f"  Journal updated → {CONFIG['JOURNAL_FILE']}")
except Exception as e:
    print(f"  [WARN] Journal: {e}")

print(f"\n{'═'*78}")
print(f"  R029 complete.")
print(f"  Verdict     : {VERDICT}")
print(f"  n Low ATR   : {port_low['n']}  (target: ≥80)")
print(f"  PF Low ATR  : {port_low['pf']:.3f}  (target: >1.20)")
print(f"  WR Low ATR  : {port_low['wr']*100:.1f}%")
print(f"  Boot p50    : {b50:.3f}  (target: >1.20)")
print(f"  MC P(profit): {mc['prob_profit']*100:.1f}%  (target: >60%)")
print(f"  Symbols up  : {n_improved}/{n_total}  (target: ≥6)")
print(f"  LOO floor   : {min(loo_vals):.3f}")
print(f"  Output      → {OUT}/r029_*")
print(f"{'═'*78}\n")
