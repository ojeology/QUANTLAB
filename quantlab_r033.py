"""
QUANTLAB AI — RESEARCH #033
Low ATR Generalisation Test — Walk-Forward Validation
======================================================

R029 verdict : WATCHLIST
R029 result  : Low ATR PF=1.205, n=64 (failed PROMOTE on n<80, boot p50<1.20)
R032 result  : BBW p25 filter → REJECT (PF<1.0, n=35)

R033 objective:
  Determine whether the Low ATR edge is a genuine market edge or a
  sample-size artefact.  Same strategy, zero new filters.

  The binding constraint from R029 is statistical power (n=64, CI too wide).
  R033 fixes this by using walk-forward OOS instead of a single 70/30 split:
    • 5 expanding windows, each OOS slice = ~10% of symbol data
    • Total OOS coverage ≈ 50% of each symbol's history
    • Each fold's ATR threshold computed from its IS period only (no look-ahead)
    • Aggregate all OOS trades across all folds and symbols

Walk-forward design (per symbol):
  Fold 1  IS = 0–50%   OOS = 50–60%
  Fold 2  IS = 0–60%   OOS = 60–70%
  Fold 3  IS = 0–70%   OOS = 70–80%   ← matches R029 OOS window
  Fold 4  IS = 0–80%   OOS = 80–90%
  Fold 5  IS = 0–90%   OOS = 90–100%

PROMOTE criteria (all must pass):
  PF > 1.20  |  n ≥ 100  |  boot p50 > 1.20  |  MC P(profit) > 60%
  ≥ 6/9 symbols improve vs baseline  |  LOO floor > 1.0  |  Max DD < 25%
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

RESEARCH_ID = "R033"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
    "LINK-USDT-SWAP", "AVAX-USDT-SWAP", "XRP-USDT-SWAP",
    "LTC-USDT-SWAP", "BCH-USDT-SWAP", "DOGE-USDT-SWAP",
]
CAPITAL = CONFIG["STARTING_CAPITAL"]

COLOURS = {
    "BTC-USDT-SWAP":  "#F7931A", "ETH-USDT-SWAP":  "#627EEA",
    "SOL-USDT-SWAP":  "#9945FF", "LINK-USDT-SWAP": "#2A5ADA",
    "AVAX-USDT-SWAP": "#E84142", "XRP-USDT-SWAP":  "#346AA9",
    "LTC-USDT-SWAP":  "#BFBBBB", "BCH-USDT-SWAP":  "#8DC351",
    "DOGE-USDT-SWAP": "#C3A634",
}

# Walk-forward fold definition: (IS_end_pct, OOS_end_pct)
# Fold 3 (IS=0–70%, OOS=70–80%) reproduces the R029 OOS window exactly.
FOLDS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #033" + " "*50 + "║")
print("║  Low ATR Generalisation Test — Walk-Forward Validation" + " "*24 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Strategy    : FVG + EMA200 Slope + Low ATR (ATR Rank < IS p25)
  Symbols     : {', '.join(s.split('-')[0] for s in SYMBOLS)}
  Method      : 5-fold expanding walk-forward OOS (≈50% coverage vs R029's 30%)
  Hypothesis  : Low ATR PF=1.205 in R029 is real; larger OOS confirms or rejects it
  PROMOTE bar : PF>1.20, n≥100, boot p50>1.20, MC P>60%, ≥6/9 improve, LOO>1.0
""")

# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["ema200"]        = calc_ema(c, 200)
    df["atr14"]         = calc_atr(df, 14)
    df["atr_rank_pct"]  = df["atr14"].rolling(100).rank(pct=True) * 100
    df["ema200_rising"] = df["ema200"] > df["ema200"].shift(10)
    df["high_2ago"]     = df["high"].shift(2)
    df["fvg_gap"]       = df["low"] > df["high_2ago"] * 1.0001
    df["prev_low"]      = df["low"].shift(1)
    return df

def signal_fvg_slope(df: pd.DataFrame) -> pd.Series:
    return (df["fvg_gap"] &
            (df["close"] > df["ema200"]) &
            df["ema200_rising"] &
            df["high_2ago"].notna()).fillna(False)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

print("  Loading 1H data …")
all_dfs = {}
for sym in SYMBOLS:
    tag = sym.replace("-", "_")
    try:
        df = pd.read_parquet(f"{CACHE}/{tag}_1H.parquet")
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
        all_dfs[sym] = add_features(df)
        print(f"  {sym.split('-')[0]:5s}  bars={len(df):,}  "
              f"{df.datetime.min().date()} → {df.datetime.max().date()}")
    except FileNotFoundError:
        print(f"  {sym}: cache missing — skipped")

print()

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE (identical to R029)
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, signal: pd.Series,
                 atr_threshold: float | None, sym_label: str,
                 fold: int) -> list:
    """
    atr_threshold: IS-period ATR rank p25.  None = baseline (no filter).
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
                    "sym":          sym_label,
                    "fold":         fold,
                    "entry_time":   str(entry_tm),
                    "exit_time":    str(bar["datetime"]),
                    "entry_price":  entry_px,
                    "exit_price":   exit_px,
                    "stop_loss":    stop,
                    "take_profit":  take,
                    "pnl":          net,
                    "r_multiple":   rmul,
                    "win":          int(exit_type == "TP"),
                    "exit_type":    exit_type,
                    "holding_hrs":  i - entry_i,
                    "atr_rank_pct": float(prev["atr_rank_pct"]),
                })
                in_pos = False
            continue

        if signal.iloc[i - 1]:
            atr_pct = prev["atr_rank_pct"]
            if np.isnan(atr_pct): continue
            if atr_threshold is not None and atr_pct >= atr_threshold:
                continue

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

def metrics(trades: list, label: str = "") -> dict:
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

def bootstrap_pf(pnls: np.ndarray, n_iter: int = 5000, seed: int = 42) -> tuple:
    if len(pnls) < 10:
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

def monte_carlo(pnls: np.ndarray, n_iter: int = 5000, seed: int = 42) -> dict:
    if len(pnls) < 5:
        return {"prob_profit": 0.0, "p5": CAPITAL, "p50": CAPITAL,
                "p95": CAPITAL, "finals": np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit": float((finals > CAPITAL).mean()),
            "p5":  float(np.percentile(finals, 5)),
            "p50": float(np.percentile(finals, 50)),
            "p95": float(np.percentile(finals, 95)),
            "finals": finals}

def loo_pf(sym_trades: dict) -> dict:
    out = {}
    for omit in sym_trades:
        flat = [t for s, tl in sym_trades.items() if s != omit for t in tl]
        m    = metrics(flat)
        out[omit] = {"pf": m["pf"], "n": m["n"]}
    return out

def jackknife_pf(pnls: np.ndarray) -> np.ndarray:
    if len(pnls) < 5:
        return np.array([])
    jk = []
    for i in range(len(pnls)):
        s  = np.delete(pnls, i)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        jk.append(wp / lp if lp > 0 else 2.0)
    return np.array(jk)

def slippage_sensitivity(trades: list, multipliers=(1.0, 2.0, 3.0)) -> list:
    results = []
    for mult in multipliers:
        adj = []
        for t in trades:
            nt = dict(t)
            if t["exit_type"] == "SL":
                extra = t["stop_loss"] * CONFIG["SL_SLIPPAGE"] * (mult - 1)
                ps    = abs(t["pnl"]) / max(abs(t["entry_price"] - t["exit_price"]), 1e-9)
                nt["pnl"] = t["pnl"] - extra * ps
            adj.append(nt)
        results.append(metrics(adj)["pf"])
    return results

# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD BACKTESTS
# ─────────────────────────────────────────────────────────────────────────────

print("─"*78)
print("  Running walk-forward backtests — 5 folds × 9 symbols × 2 variants …")
print()

# sym_base_trades[sym] = list of all OOS baseline trades across folds
# sym_low_trades[sym]  = list of all OOS Low ATR trades across folds
sym_base_trades: dict[str, list] = {s: [] for s in all_dfs}
sym_low_trades:  dict[str, list] = {s: [] for s in all_dfs}

fold_summaries = []  # (fold_idx, port_base_pf, port_low_pf, n_base, n_low)

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_base = []
    fold_low  = []
    fold_thr  = []

    for sym, df_full in all_dfs.items():
        N       = len(df_full)
        is_cut  = int(N * is_end)
        oos_cut = int(N * oos_end)

        df_is  = df_full.iloc[:is_cut]
        df_oos = df_full.iloc[is_cut:oos_cut].reset_index(drop=True)

        if len(df_oos) < 100:
            continue  # too few bars to be meaningful

        # ATR threshold from IS period only (no look-ahead)
        is_atr_p25 = float(df_is["atr_rank_pct"].dropna().quantile(0.25))
        fold_thr.append(is_atr_p25)

        sig = signal_fvg_slope(df_oos)
        bt  = run_backtest(df_oos, sig, None,         sym, fold_idx)
        lt  = run_backtest(df_oos, sig, is_atr_p25,   sym, fold_idx)

        sym_base_trades[sym].extend(bt)
        sym_low_trades[sym].extend(lt)
        fold_base.extend(bt)
        fold_low.extend(lt)

    fb = metrics(fold_base); fl = metrics(fold_low)
    thr_mean = np.mean(fold_thr) if fold_thr else 0.0
    print(f"  Fold {fold_idx}  (IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%)  "
          f"thr={thr_mean:.1f}  "
          f"Base: n={fb['n']:3d} PF={fb['pf']:.3f}  "
          f"LowATR: n={fl['n']:3d} PF={fl['pf']:.3f}")
    fold_summaries.append((fold_idx, is_end, oos_end, fb["pf"], fl["pf"], fb["n"], fl["n"]))

print()

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO AGGREGATES
# ─────────────────────────────────────────────────────────────────────────────

all_base_flat = [t for s in all_dfs for t in sym_base_trades[s]]
all_low_flat  = [t for s in all_dfs for t in sym_low_trades[s]]

port_base = metrics(all_base_flat, "Baseline (WF)")
port_low  = metrics(all_low_flat,  "Low ATR (WF)")

print("═"*78)
print("  PORTFOLIO RESULTS — Walk-Forward OOS (5 folds, 9 symbols)")
print("═"*78)
print(f"\n  {'Variant':16s}  {'n':>5}  {'WR':>6}  {'PF':>7}  "
      f"{'ExpR':>7}  {'Sharpe':>7}  {'MDD':>7}  {'Net $':>9}")
print("  " + "─"*66)
for label, m in [("Baseline", port_base), ("Low ATR", port_low)]:
    print(f"  {label:16s}  {m['n']:5d}  {m['wr']*100:5.1f}%  {m['pf']:7.3f}  "
          f"{m['exp_r']:+7.3f}  {m['sharpe']:7.2f}  {m['mdd']*100:6.1f}%  "
          f"{m['net']:+9.0f}")

dpf = port_low["pf"] - port_base["pf"]
print(f"\n  δPF (Low ATR vs Baseline): {dpf:+.3f}")
print(f"  WR shift: {port_base['wr']*100:.1f}% → {port_low['wr']*100:.1f}%")
print(f"  Trade retention: {port_low['n']}/{port_base['n']} "
      f"= {port_low['n']/max(port_base['n'],1)*100:.0f}%")

# Fold-by-fold table
print(f"\n  {'Fold':>5}  {'IS%':>6}→{'OOS%':>5}  "
      f"{'Base PF':>9}  {'LowATR PF':>10}  {'n(base)':>8}  {'n(low)':>7}  {'δPF':>8}")
print("  " + "─"*64)
for fi, is_end, oos_end, bpf, lpf, nb, nl in fold_summaries:
    print(f"  {fi:5d}  {is_end*100:5.0f}%→{oos_end*100:4.0f}%  "
          f"{bpf:9.3f}  {lpf:10.3f}  {nb:8d}  {nl:7d}  {lpf-bpf:+8.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# PER-SYMBOL TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  PER-SYMBOL RESULTS — Baseline vs Low ATR (all 5 folds combined)")
print("═"*78)
print(f"\n  {'Symbol':6s}  {'Base n':>7}  {'Base PF':>8}  {'Low n':>7}  "
      f"{'Low PF':>8}  {'Low WR':>7}  {'δPF':>8}  {'Improve?':>9}")
print("  " + "─"*69)

n_improved = 0
sym_ranks = []

for sym in all_dfs:
    tag  = sym.split("-")[0]
    bm   = metrics(sym_base_trades[sym])
    lm   = metrics(sym_low_trades[sym])
    dpf_ = lm["pf"] - bm["pf"]
    ok   = dpf_ > 0 and lm["n"] > 0
    if ok:
        n_improved += 1
    flag = "✓" if ok else "✗"
    print(f"  {tag:6s}  {bm['n']:7d}  {bm['pf']:8.3f}  {lm['n']:7d}  "
          f"{lm['pf']:8.3f}  {lm['wr']*100:6.1f}%  {dpf_:+8.3f}  {flag:>9}")
    sym_ranks.append((sym, lm["n"], lm["pf"], lm["wr"], dpf_))

print(f"\n  Low ATR improved {n_improved}/{len(all_dfs)} symbols")

# Symbol ranking by PF
sym_ranks.sort(key=lambda x: x[2], reverse=True)
print(f"\n  Symbol ranking by Low ATR PF:")
print(f"  {'Rank':>4}  {'Symbol':6s}  {'n':>5}  {'PF':>7}  {'WR':>6}  {'Verdict':>10}")
print("  " + "─"*40)
for rank, (sym, n_, pf_, wr_, dpf_) in enumerate(sym_ranks, 1):
    tag = sym.split("-")[0]
    if pf_ >= 1.20 and n_ >= 5:
        verdict = "KEEP ✓"
    elif pf_ >= 1.0 and n_ >= 3:
        verdict = "MONITOR ~"
    elif n_ < 3:
        verdict = "FEW TRADES"
    else:
        verdict = "REMOVE ✗"
    print(f"  {rank:4d}  {tag:6s}  {n_:5d}  {pf_:7.3f}  {wr_*100:5.1f}%  {verdict:>10}")

# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS TESTS
# ─────────────────────────────────────────────────────────────────────────────

N_BOOT = 5_000
pnls_low  = port_low["pnls"]
pnls_base = port_base["pnls"]

print("\n" + "═"*78)
print("  ROBUSTNESS TESTS — Low ATR Walk-Forward Portfolio")
print("═"*78)

# Bootstrap PF
b5, b50, b95 = bootstrap_pf(pnls_low, N_BOOT)
print(f"\n  Bootstrap PF ({N_BOOT:,} iterations):")
print(f"    p5  = {b5:.3f}")
print(f"    p50 = {b50:.3f}  {'> 1.20 ✓' if b50 > 1.20 else '< 1.20 ✗'}")
print(f"    p95 = {b95:.3f}")
print(f"    95% CI width = {b95 - b5:.3f}")

# Bootstrap baseline for comparison
bb5, bb50, bb95 = bootstrap_pf(pnls_base, N_BOOT)
print(f"\n  Bootstrap PF — Baseline (reference):")
print(f"    p5={bb5:.3f}  p50={bb50:.3f}  p95={bb95:.3f}")

# Monte Carlo
mc = monte_carlo(pnls_low, N_BOOT)
print(f"\n  Monte Carlo ({N_BOOT:,} simulations):")
print(f"    P(profit) = {mc['prob_profit']*100:.1f}%  "
      f"{'> 60% ✓' if mc['prob_profit'] > 0.60 else '≤ 60% ✗'}")
print(f"    p5  equity = ${mc['p5']:,.0f}")
print(f"    p50 equity = ${mc['p50']:,.0f}")
print(f"    p95 equity = ${mc['p95']:,.0f}")

# Leave-one-symbol-out
loo = loo_pf(sym_low_trades)
print(f"\n  Leave-One-Symbol-Out (Low ATR PF when each symbol is removed):")
print(f"  {'Symbol':7s}  {'LOO PF':>9}  {'LOO n':>7}  {'PF>1.0':>8}  {'PF>1.20':>8}")
print("  " + "─"*43)
loo_vals = []
for sym in all_dfs:
    v   = loo[sym]
    tag = sym.split("-")[0]
    loo_vals.append(v["pf"])
    mark_1  = "✓" if v["pf"] > 1.0  else "✗"
    mark_12 = "✓" if v["pf"] > 1.20 else "✗"
    print(f"  {tag:7s}  {v['pf']:9.3f}  {v['n']:7d}  {mark_1:>8}  {mark_12:>8}")

loo_floor = min(loo_vals)
loo_all_pos = loo_floor > 1.0
loo_all_120 = loo_floor > 1.20
print(f"\n  LOO floor={loo_floor:.3f}  max={max(loo_vals):.3f}")
print(f"  All LOO > 1.0  : {'YES ✓' if loo_all_pos else 'NO ✗'}")
print(f"  All LOO > 1.20 : {'YES ✓' if loo_all_120 else 'NO ✗'}")

# Jackknife on Low ATR PF
jk = jackknife_pf(pnls_low)
if len(jk):
    print(f"\n  Jackknife (leave-one-trade-out):")
    print(f"    mean={jk.mean():.3f}  std={jk.std():.3f}  min={jk.min():.3f}  max={jk.max():.3f}")
    print(f"    % JK PF > 1.0  : {(jk > 1.0).mean()*100:.0f}%")
    print(f"    % JK PF > 1.20 : {(jk > 1.20).mean()*100:.0f}%")

# Slippage sensitivity
slip_pfs = slippage_sensitivity(all_low_flat)
print(f"\n  SL slippage sensitivity (Low ATR):")
print(f"    1× = {slip_pfs[0]:.3f}  |  2× = {slip_pfs[1]:.3f}  |  3× = {slip_pfs[2]:.3f}")
print(f"    Profitable at 3× slippage: {'YES ✓' if slip_pfs[2] > 1.0 else 'NO ✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# FOLD CONSISTENCY CHECK
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  FOLD CONSISTENCY — is the edge stable across time?")
print("═"*78)

fold_pfs_low  = [f[4] for f in fold_summaries]
fold_pfs_base = [f[3] for f in fold_summaries]
n_folds_above_1   = sum(1 for p in fold_pfs_low if p > 1.0)
n_folds_above_120 = sum(1 for p in fold_pfs_low if p > 1.20)

print(f"\n  Low ATR fold PFs: {[f'{p:.3f}' for p in fold_pfs_low]}")
print(f"  Folds with PF > 1.0  : {n_folds_above_1}/5")
print(f"  Folds with PF > 1.20 : {n_folds_above_120}/5")

if n_folds_above_1 >= 4:
    print("  Edge is time-stable: appears in most OOS windows ✓")
elif n_folds_above_1 >= 3:
    print("  Edge is moderate: positive in majority of windows ~")
else:
    print("  Edge is unstable: fails majority of OOS windows ✗")

# R029 comparison (Fold 3 is the R029 OOS window)
fold3 = next((f for f in fold_summaries if f[0] == 3), None)
if fold3:
    print(f"\n  R029 OOS window replication (Fold 3: IS=70%→OOS=80%):")
    print(f"    R029 Low ATR PF = 1.205  |  R033 Fold 3 PF = {fold3[4]:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  RESEARCH QUESTIONS")
print("═"*78)

q1 = port_low["pf"] > 1.20
q2 = b50 > 1.20
q3 = mc["prob_profit"] > 0.60
q4 = port_low["n"] >= 100
q5 = n_improved >= 6
q6_keep  = [s.split("-")[0] for s, n_, pf_, wr_, dpf_ in sym_ranks if pf_ >= 1.20 and n_ >= 5]
q7_remove = [s.split("-")[0] for s, n_, pf_, wr_, dpf_ in sym_ranks if pf_ < 1.0 and n_ >= 3]

print(f"""
  Q1. Does portfolio PF remain above 1.20?
      Low ATR PF = {port_low['pf']:.3f}
      {'YES ✓' if q1 else 'NO ✗  (failed — edge does not survive expansion)'}

  Q2. Does bootstrap median PF stay above 1.20?
      boot p50 = {b50:.3f}
      {'YES ✓' if q2 else 'NO ✗'}

  Q3. Does Monte Carlo P(profit) remain above 60%?
      P(profit) = {mc['prob_profit']*100:.1f}%
      {'YES ✓' if q3 else 'NO ✗'}

  Q4. Does trade count exceed 100?
      n = {port_low['n']}
      {'YES ✓' if q4 else 'NO ✗  (sample still insufficient for definitive verdict)'}

  Q5. Do at least 6 symbols individually improve vs baseline?
      {n_improved}/9 symbols improve
      {'YES ✓' if q5 else 'NO ✗'}

  Q6. Which symbols consistently produce the edge?
      PF ≥ 1.20 and n ≥ 5: {', '.join(q6_keep) if q6_keep else 'none'}
      Top 3 symbols: {', '.join(s.split('-')[0] for s, *_ in sym_ranks[:3])}

  Q7. Which symbols should be permanently removed?
      PF < 1.0 and n ≥ 3: {', '.join(q7_remove) if q7_remove else 'none meet remove criteria'}
""")

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────

print("═"*78)
print("  PROMOTE CRITERIA & VERDICT")
print("═"*78)

CRITERIA = {
    f"PF > 1.20                 (PF={port_low['pf']:.3f})":         port_low["pf"] > 1.20,
    f"n ≥ 100                   (n={port_low['n']})":                port_low["n"] >= 100,
    f"Boot p50 > 1.20           (p50={b50:.3f})":                   b50 > 1.20,
    f"MC P(profit) > 60%        ({mc['prob_profit']*100:.1f}%)":     mc["prob_profit"] > 0.60,
    f"≥6/9 symbols improve      ({n_improved}/9)":                   n_improved >= 6,
    f"LOO floor > 1.0           (floor={loo_floor:.3f})":            loo_floor > 1.0,
    f"Max Drawdown < 25%        ({abs(port_low['mdd'])*100:.1f}%)": abs(port_low["mdd"]) < 0.25,
}

n_pass = sum(CRITERIA.values())
n_total = len(CRITERIA)

if all(CRITERIA.values()):
    VERDICT = "PROMOTE"
elif n_pass >= 5:
    VERDICT = "WATCHLIST"
elif n_pass >= 3:
    VERDICT = "INVESTIGATE"
else:
    VERDICT = "REJECT"

vmap   = {"PROMOTE": "\033[92m", "WATCHLIST": "\033[93m",
          "INVESTIGATE": "\033[94m", "REJECT": "\033[91m"}
vreset = "\033[0m"

print(f"\n  {vmap[VERDICT]}VERDICT: {VERDICT}{vreset}  ({n_pass}/{n_total} criteria)\n")
for crit, ok in CRITERIA.items():
    print(f"    {'✓' if ok else '✗'} {crit}")

print(f"\n  Key numbers — Low ATR Walk-Forward:")
print(f"    n={port_low['n']}  PF={port_low['pf']:.3f}  WR={port_low['wr']*100:.1f}%  "
      f"MDD={port_low['mdd']*100:.1f}%  Net=${port_low['net']:+,.0f}")
print(f"    Boot: p5={b5:.3f}  p50={b50:.3f}  p95={b95:.3f}")
print(f"    MC  : P(profit)={mc['prob_profit']*100:.1f}%  p50=${mc['p50']:,.0f}")
print(f"    LOO : floor={loo_floor:.3f}")
print(f"    Folds profitable: {n_folds_above_1}/5 (>1.0)  {n_folds_above_120}/5 (>1.20)")

if VERDICT == "PROMOTE":
    print(f"""
  ═══ PROMOTION ═══════════════════════════════════════════════════════════
  Strategy  : FVG + EMA200 Slope + Low ATR (ATR Rank < IS p25)
  Universe  : {', '.join(s.split('-')[0] for s in all_dfs)}
  Timeframe : 1H  |  Long only
  Entry     : Open of bar after FVG signal
  Stop      : Prior bar low (fallback: current bar low)
  Target    : Entry + 2× stop distance
  Size      : 1% risk per trade  |  max 5× leverage
  ATR gate  : ATR(14) rolling rank < 25th pct of IS period

  NEXT STEP : Begin paper trading at 0.5× risk for 30 days / 30 trades.
  ═══════════════════════════════════════════════════════════════════════════
""")
elif VERDICT == "WATCHLIST":
    print(f"""
  ═══ WATCHLIST ═══════════════════════════════════════════════════════════
  The edge survives expansion but does not yet clear all PROMOTE bars.
  Failing criteria: {[c.split('(')[0].strip() for c, ok in CRITERIA.items() if not ok]}

  Options to reach PROMOTE:
    a) Fetch deeper history (more bars → more trades → narrower CI)
    b) Investigate symbol subset: keep high-PF symbols, remove drag symbols
    c) Accept at WATCHLIST with reduced position sizing (0.25× risk)
  ═══════════════════════════════════════════════════════════════════════════
""")
elif VERDICT == "INVESTIGATE":
    print(f"""
  ═══ INVESTIGATE ════════════════════════════════════════════════════════
  Mixed results. The edge may be regime-specific.
  Passing criteria  : {[c.split('(')[0].strip() for c, ok in CRITERIA.items() if ok]}
  Failing criteria  : {[c.split('(')[0].strip() for c, ok in CRITERIA.items() if not ok]}

  Recommended analysis:
    → Sub-sample by fold: which time periods drive the result?
    → Drop bottom-3 symbols and retest portfolio
    → Check if LOO failure is driven by one outlier symbol
""")
else:
    print(f"""
  ═══ REJECT ══════════════════════════════════════════════════════════════
  Low ATR edge does not survive walk-forward expansion.
  This strategy should be permanently removed from the research pipeline.
  The Low ATR filter does not add a reliable edge over raw FVG signals.
""")

print("═"*78)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Generating charts …")

BG  = "#111111"
FG  = "white"
ACC = "#FF9800"

def dark_ax(ax, title=None):
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=FG, fontsize=9, pad=6)

sym_tags = [s.split("-")[0] for s in all_dfs]
sym_cols = [COLOURS.get(s, "white") for s in all_dfs]

# ── 1: PF comparison bar chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
dark_ax(ax, "R033 — Portfolio PF: Baseline vs Low ATR (Walk-Forward OOS)")
vals   = [port_base["pf"], port_low["pf"]]
labels = [f"Baseline\nn={port_base['n']}", f"Low ATR\nn={port_low['n']}"]
cols   = ["#9E9E9E", "#4CAF50"]
bars   = ax.bar(labels, vals, color=cols, alpha=0.85, width=0.4)
ax.axhline(1.0, color=FG,  lw=0.8, ls="--", alpha=0.4)
ax.axhline(1.2, color=ACC, lw=0.8, ls=":",  alpha=0.7, label="PF=1.20 target")
ax.set_ylabel("Profit Factor", color=FG)
ax.legend(facecolor="#222", labelcolor=FG, fontsize=9)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", color=FG, fontsize=14, fontweight="bold")
plt.tight_layout()
p = f"{OUT}/r033_pf_comparison.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 2: Per-symbol PF ─────────────────────────────────────────────────────────
base_pfs = [metrics(sym_base_trades[s])["pf"] for s in all_dfs]
low_pfs  = [metrics(sym_low_trades[s])["pf"]  for s in all_dfs]
low_ns   = [metrics(sym_low_trades[s])["n"]   for s in all_dfs]

fig, axes = plt.subplots(1, 2, figsize=(20, 5), facecolor=BG)
fig.suptitle("R033 — Per-Symbol PF: Baseline vs Low ATR (all folds)", color=FG, fontsize=11)
for ax_, pfs, title, ref_col in [(axes[0], base_pfs, "Baseline", "#9E9E9E"),
                                  (axes[1], low_pfs,  "Low ATR", "#4CAF50")]:
    dark_ax(ax_, title)
    bs_ = ax_.bar(sym_tags, pfs, color=sym_cols, alpha=0.85)
    ax_.axhline(1.0, color=FG,  lw=0.7, ls="--", alpha=0.4)
    ax_.axhline(1.2, color=ACC, lw=0.7, ls=":",  alpha=0.6)
    ax_.set_ylabel("PF", color=FG)
    for b, v in zip(bs_, pfs):
        ax_.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}",
                 ha="center", color=FG, fontsize=8)
for b, n_ in zip(axes[1].patches, low_ns):
    axes[1].text(b.get_x()+b.get_width()/2, 0.05, f"n={n_}",
                 ha="center", color="#ccc", fontsize=7,
                 transform=axes[1].get_xaxis_transform())
plt.tight_layout()
p = f"{OUT}/r033_per_symbol_pf.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 3: δPF by symbol ─────────────────────────────────────────────────────────
dpfs  = [l - b for l, b in zip(low_pfs, base_pfs)]
dcols = ["#4CAF50" if d > 0 else "#F44336" for d in dpfs]
fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG)
dark_ax(ax, "R033 — δPF (Low ATR − Baseline) by Symbol")
bs_ = ax.bar(sym_tags, dpfs, color=dcols, alpha=0.85)
ax.axhline(0, color=FG, lw=0.7, ls="--", alpha=0.4)
ax.set_ylabel("δPF", color=FG)
for b, d in zip(bs_, dpfs):
    y = b.get_height() + 0.02 if d >= 0 else b.get_height() - 0.08
    ax.text(b.get_x()+b.get_width()/2, y, f"{d:+.2f}",
            ha="center", color=FG, fontsize=8)
plt.tight_layout()
p = f"{OUT}/r033_delta_pf.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 4: Equity curves ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 5), facecolor=BG)
fig.suptitle("R033 — Equity Curves (per-symbol Low ATR, walk-forward OOS)", color=FG, fontsize=11)
for ax_, sym_dict, title in [(axes[0], sym_base_trades, "Baseline"),
                              (axes[1], sym_low_trades,  "Low ATR")]:
    dark_ax(ax_, title)
    for sym in all_dfs:
        tl = sym_dict[sym]
        if not tl: continue
        pnls_ = [t["pnl"] for t in tl]
        eq_   = CAPITAL + np.cumsum(pnls_)
        ax_.plot(eq_, color=COLOURS.get(sym, "white"), lw=1.2, alpha=0.8,
                 label=sym.split("-")[0])
    ax_.axhline(CAPITAL, color=FG, lw=0.6, ls="--", alpha=0.3)
    ax_.set_ylabel("Equity ($)", color=FG)
    ax_.legend(facecolor="#222", labelcolor=FG, fontsize=7, ncol=2)
plt.tight_layout()
p = f"{OUT}/r033_equity_curves.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 5: Bootstrap CI ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 5), facecolor=BG)
fig.suptitle("R033 — Bootstrap PF Distribution (Low ATR vs Baseline)", color=FG, fontsize=11)

for ax_, pnls_, label_, b5_, b50_, b95_, col in [
        (axes[0], pnls_base, "Baseline",  bb5, bb50, bb95, "#9E9E9E"),
        (axes[1], pnls_low,  "Low ATR",   b5,  b50,  b95,  "#4CAF50")]:
    rng = np.random.default_rng(42)
    pfs_ = []
    for _ in range(N_BOOT):
        s  = rng.choice(pnls_, len(pnls_), replace=True)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        pfs_.append(wp / lp if lp > 0 else 2.0)
    dark_ax(ax_, f"{label_}: p5={b5_:.3f}  p50={b50_:.3f}  p95={b95_:.3f}")
    ax_.hist(pfs_, bins=60, color=col, alpha=0.75, edgecolor="none")
    ax_.axvline(1.0,  color=FG,  lw=0.8, ls="--", alpha=0.5, label="PF=1.0")
    ax_.axvline(1.2,  color=ACC, lw=0.8, ls=":",  alpha=0.7, label="PF=1.20")
    ax_.axvline(b50_, color=col, lw=1.5, ls="-",  alpha=0.9, label=f"p50={b50_:.3f}")
    ax_.set_xlabel("PF", color=FG); ax_.set_ylabel("Count", color=FG)
    ax_.legend(facecolor="#222", labelcolor=FG, fontsize=8)
plt.tight_layout()
p = f"{OUT}/r033_bootstrap_ci.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 6: Monte Carlo ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
dark_ax(ax, f"R033 — Monte Carlo ({N_BOOT:,} sims)  P(profit)={mc['prob_profit']*100:.1f}%")
ax.hist(mc["finals"], bins=80, color="#4CAF50", alpha=0.7, edgecolor="none")
ax.axvline(CAPITAL,   color=FG,  lw=1.0, ls="--", label=f"Start ${CAPITAL:,.0f}")
ax.axvline(mc["p5"],  color="#F44336", lw=1.0, ls=":", label=f"p5 ${mc['p5']:,.0f}")
ax.axvline(mc["p50"], color="#4CAF50", lw=1.5, ls="-", label=f"p50 ${mc['p50']:,.0f}")
ax.axvline(mc["p95"], color="#2196F3", lw=1.0, ls=":", label=f"p95 ${mc['p95']:,.0f}")
ax.set_xlabel("Final Equity ($)", color=FG)
ax.set_ylabel("Count", color=FG)
ax.legend(facecolor="#222", labelcolor=FG, fontsize=9)
plt.tight_layout()
p = f"{OUT}/r033_monte_carlo.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 7: LOO robustness ────────────────────────────────────────────────────────
loo_syms = [s.split("-")[0] for s in all_dfs]
loo_vals_list = [loo[s]["pf"] for s in all_dfs]
fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG)
dark_ax(ax, "R033 — Leave-One-Symbol-Out PF (Low ATR)")
cols_loo = ["#4CAF50" if v > 1.2 else ("#FF9800" if v > 1.0 else "#F44336")
            for v in loo_vals_list]
bs_ = ax.bar(loo_syms, loo_vals_list, color=cols_loo, alpha=0.85)
ax.axhline(1.0, color=FG,  lw=0.8, ls="--", alpha=0.4, label="PF=1.0")
ax.axhline(1.2, color=ACC, lw=0.8, ls=":",  alpha=0.7, label="PF=1.20")
ax.set_ylabel("LOO PF", color=FG)
ax.legend(facecolor="#222", labelcolor=FG, fontsize=9)
for b, v in zip(bs_, loo_vals_list):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", color=FG, fontsize=8)
plt.tight_layout()
p = f"{OUT}/r033_loo_robustness.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 8: Fold PF timeline ──────────────────────────────────────────────────────
f_idxs    = [f[0] for f in fold_summaries]
f_bpfs    = [f[3] for f in fold_summaries]
f_lpfs    = [f[4] for f in fold_summaries]
f_labels  = [f"F{f[0]}\n{int(f[1]*100)}→{int(f[2]*100)}%" for f in fold_summaries]

fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
dark_ax(ax, "R033 — PF by Fold (edge stability across time)")
x = np.arange(len(f_idxs))
w = 0.35
bars_b = ax.bar(x - w/2, f_bpfs, w, color="#9E9E9E", alpha=0.75, label="Baseline")
bars_l = ax.bar(x + w/2, f_lpfs, w, color="#4CAF50", alpha=0.85, label="Low ATR")
ax.axhline(1.0, color=FG,  lw=0.7, ls="--", alpha=0.4)
ax.axhline(1.2, color=ACC, lw=0.7, ls=":",  alpha=0.6)
ax.set_xticks(x); ax.set_xticklabels(f_labels, color=FG)
ax.set_ylabel("PF", color=FG)
ax.legend(facecolor="#222", labelcolor=FG, fontsize=9)
# Highlight Fold 3 (R029 equivalent)
ax.axvspan(2-0.5, 2+0.5, color="#FF9800", alpha=0.08, label="Fold 3 = R029 OOS")
ax.text(2, max(max(f_bpfs), max(f_lpfs))*0.95, "R029\nOOS",
        ha="center", color=ACC, fontsize=8)
plt.tight_layout()
p = f"{OUT}/r033_fold_pf_timeline.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 9: Dashboard ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14), facecolor=BG)
fig.suptitle("QUANTLAB AI — R033 | Low ATR Generalisation Test | Walk-Forward OOS",
             color=FG, fontsize=14, y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# PF comparison
ax1 = fig.add_subplot(gs[0, 0])
dark_ax(ax1, "Portfolio PF")
vals_  = [port_base["pf"], port_low["pf"]]
cols_  = ["#9E9E9E", "#4CAF50"]
bs_    = ax1.bar(["Base", "LowATR"], vals_, color=cols_, alpha=0.85)
ax1.axhline(1.2, color=ACC, lw=0.8, ls=":")
for b, v in zip(bs_, vals_):
    ax1.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
             ha="center", color=FG, fontsize=10, fontweight="bold")
ax1.set_ylabel("PF", color=FG)

# δPF by symbol
ax2 = fig.add_subplot(gs[0, 1:3])
dark_ax(ax2, "δPF by Symbol (Low ATR − Baseline)")
ax2.bar(sym_tags, dpfs,
        color=["#4CAF50" if d > 0 else "#F44336" for d in dpfs], alpha=0.85)
ax2.axhline(0, color=FG, lw=0.7, ls="--", alpha=0.4)
ax2.set_ylabel("δPF", color=FG)

# Fold timeline
ax3 = fig.add_subplot(gs[0, 3])
dark_ax(ax3, "PF by Fold")
x_ = np.arange(len(f_idxs))
ax3.bar(x_ - 0.2, f_bpfs, 0.35, color="#9E9E9E", alpha=0.7, label="Base")
ax3.bar(x_ + 0.2, f_lpfs, 0.35, color="#4CAF50", alpha=0.85, label="LowATR")
ax3.axhline(1.2, color=ACC, lw=0.8, ls=":")
ax3.set_xticks(x_); ax3.set_xticklabels([f"F{i}" for i in f_idxs], color=FG)
ax3.legend(facecolor="#222", labelcolor=FG, fontsize=7)
ax3.set_ylabel("PF", color=FG)

# Equity curves Low ATR
ax4 = fig.add_subplot(gs[1, :2])
dark_ax(ax4, "Low ATR — Equity Curves (per symbol)")
for sym in all_dfs:
    tl = sym_low_trades[sym]
    if not tl: continue
    eq_ = CAPITAL + np.cumsum([t["pnl"] for t in tl])
    ax4.plot(eq_, color=COLOURS.get(sym, "white"), lw=1.2, alpha=0.8,
             label=sym.split("-")[0])
ax4.axhline(CAPITAL, color=FG, lw=0.6, ls="--", alpha=0.3)
ax4.set_ylabel("Equity ($)", color=FG)
ax4.legend(facecolor="#222", labelcolor=FG, fontsize=7, ncol=3)

# Bootstrap histogram
ax5 = fig.add_subplot(gs[1, 2:])
dark_ax(ax5, f"Bootstrap PF — Low ATR  p50={b50:.3f}")
rng2 = np.random.default_rng(42)
pfs_boot = []
for _ in range(N_BOOT):
    s = rng2.choice(pnls_low, len(pnls_low), replace=True)
    wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
    pfs_boot.append(wp / lp if lp > 0 else 2.0)
ax5.hist(pfs_boot, bins=60, color="#4CAF50", alpha=0.7, edgecolor="none")
ax5.axvline(1.0,  color=FG,  lw=0.8, ls="--", alpha=0.5)
ax5.axvline(1.2,  color=ACC, lw=0.8, ls=":")
ax5.axvline(b50,  color="#4CAF50", lw=1.5)
ax5.set_xlabel("PF", color=FG); ax5.set_ylabel("Count", color=FG)

# LOO
ax6 = fig.add_subplot(gs[2, :2])
dark_ax(ax6, "Leave-One-Symbol-Out PF")
cols_loo_ = ["#4CAF50" if v > 1.2 else ("#FF9800" if v > 1.0 else "#F44336")
             for v in loo_vals_list]
bs_loo = ax6.bar(loo_syms, loo_vals_list, color=cols_loo_, alpha=0.85)
ax6.axhline(1.0, color=FG,  lw=0.7, ls="--", alpha=0.4)
ax6.axhline(1.2, color=ACC, lw=0.7, ls=":")
ax6.set_ylabel("LOO PF", color=FG)
for b, v in zip(bs_loo, loo_vals_list):
    ax6.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}",
             ha="center", color=FG, fontsize=7)

# Monte Carlo
ax7 = fig.add_subplot(gs[2, 2:])
dark_ax(ax7, f"Monte Carlo — P(profit)={mc['prob_profit']*100:.1f}%")
ax7.hist(mc["finals"], bins=60, color="#4CAF50", alpha=0.7, edgecolor="none")
ax7.axvline(CAPITAL,   color=FG,       lw=1.0, ls="--")
ax7.axvline(mc["p50"], color="#4CAF50", lw=1.5)
ax7.axvline(mc["p5"],  color="#F44336", lw=0.9, ls=":")
ax7.set_xlabel("Final Equity ($)", color=FG); ax7.set_ylabel("Count", color=FG)

# Verdict text box
criteria_text = "\n".join(
    f"  {'✓' if ok else '✗'} {crit.split('(')[0].strip()}"
    for crit, ok in CRITERIA.items()
)
verdict_text = (f"VERDICT: {VERDICT}  ({n_pass}/{n_total} criteria)\n\n"
                f"{criteria_text}\n\n"
                f"n={port_low['n']}  PF={port_low['pf']:.3f}\n"
                f"WR={port_low['wr']*100:.1f}%  MDD={port_low['mdd']*100:.1f}%\n"
                f"Boot p50={b50:.3f}  MC P={mc['prob_profit']*100:.0f}%")
fig.text(0.01, 0.01, verdict_text, color=FG, fontsize=8,
         fontfamily="monospace", va="bottom",
         bbox=dict(boxstyle="round", facecolor="#1a1a1a", edgecolor="#444"))

plt.savefig(f"{OUT}/r033_dashboard.png", dpi=130, bbox_inches="tight")
plt.close()
print(f"  → {OUT}/r033_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# TRADE LOG & JOURNAL
# ─────────────────────────────────────────────────────────────────────────────

trade_log_path = f"{OUT}/r033_trade_log.csv"
pd.DataFrame(all_low_flat).to_csv(trade_log_path, index=False)
print(f"  → {trade_log_path}  ({len(all_low_flat)} trades)")

# Journal update
journal_path = CONFIG["JOURNAL_FILE"]
journal_row  = {
    "research_id":  RESEARCH_ID,
    "date":         pd.Timestamp.now().strftime("%Y-%m-%d"),
    "strategy":     "FVG+EMA200Slope+LowATR",
    "timeframe":    "1H",
    "symbols":      ",".join(s.split("-")[0] for s in all_dfs),
    "method":       "walk-forward-5fold",
    "n_oos":        port_low["n"],
    "wr":           round(port_low["wr"], 4),
    "pf":           round(port_low["pf"], 4),
    "sharpe":       round(port_low["sharpe"], 4),
    "mdd":          round(port_low["mdd"], 4),
    "net":          round(port_low["net"], 2),
    "boot_p50":     round(b50, 4),
    "mc_prob":      round(mc["prob_profit"], 4),
    "loo_floor":    round(loo_floor, 4),
    "verdict":      VERDICT,
}
journal_df = pd.read_csv(journal_path) if os.path.exists(journal_path) else pd.DataFrame()
journal_df = pd.concat([journal_df, pd.DataFrame([journal_row])], ignore_index=True)
journal_df.to_csv(journal_path, index=False)
print(f"  Journal updated → {journal_path}")

print("\n" + "═"*78)
print(f"  R033 complete.")
print(f"  Verdict   : {VERDICT}  ({n_pass}/{n_total} criteria)")
print(f"  Strategy  : FVG + EMA200 Slope + Low ATR (walk-forward OOS)")
print(f"  n         : {port_low['n']}  (Baseline: {port_base['n']})")
print(f"  PF        : Base={port_base['pf']:.3f}  LowATR={port_low['pf']:.3f}  δ={dpf:+.3f}")
print(f"  WR        : Base={port_base['wr']*100:.1f}%  LowATR={port_low['wr']*100:.1f}%")
print(f"  MDD       : {port_low['mdd']*100:.1f}%")
print(f"  Boot p50  : {b50:.3f}")
print(f"  MC P(profit): {mc['prob_profit']*100:.1f}%")
print(f"  LOO floor : {loo_floor:.3f}")
print(f"  Folds >1.0: {n_folds_above_1}/5  Folds >1.20: {n_folds_above_120}/5")
print(f"  Output    : {OUT}/r033_*")
print("═"*78)
