"""
QUANTLAB AI — RESEARCH #026
Universal Environment Validation
=================================

Hypothesis:
  The R023 environment (price near 20-bar low + wide Bollinger Bands) contains
  strategy-independent edge. If true, it should improve multiple entry concepts,
  not just EMA Pullback.

Environment Filter (applied identically to every strategy):
  Gate A: Price within 5% of 20-bar low  (dist_from_low20_pct <= 5.0)
  Gate B: Bollinger Band Width >= 67th percentile (upper tertile)

Strategies:
  1. Liquidity Sweep Reversal
  2. FVG + EMA200 Slope
  3. Break of Structure
  4. Volatility Compression Breakout
  5. Donchian Breakout
  6. EMA Pullback  (control — previously tested in R024)

Symbols: BTC, ETH, SOL, LINK, AVAX, LTC  (1H, OOS-only, 70/30 split)
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

RESEARCH_ID = "R026"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

SYMBOLS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP",
    "LINK-USDT-SWAP",
    "AVAX-USDT-SWAP",
    "LTC-USDT-SWAP",
]
SPLIT    = 0.70
CAPITAL  = CONFIG["STARTING_CAPITAL"]
EMA_FAST = 20
EMA_SLOW = 100

COLOURS = {
    "BTC-USDT-SWAP":  "#F7931A",
    "ETH-USDT-SWAP":  "#627EEA",
    "SOL-USDT-SWAP":  "#9945FF",
    "LINK-USDT-SWAP": "#2A5ADA",
    "AVAX-USDT-SWAP": "#E84142",
    "LTC-USDT-SWAP":  "#BFBBBB",
}
STRAT_COLOURS = {
    "LiqSweep":    "#4CAF50",
    "FVG+Slope":   "#FF9800",
    "BreakStruct": "#9C27B0",
    "VolComp":     "#F44336",
    "Donchian":    "#00BCD4",
    "EMA_Pullback":"#2196F3",
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

def session_label(hour):
    if   0  <= hour <  7:  return "Asia"
    elif 7  <= hour < 12:  return "London"
    elif 12 <= hour < 20:  return "NewYork"
    else:                  return "Dead"

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df["close"]

    # EMAs
    df["ema_fast"] = calc_ema(c, EMA_FAST)
    df["ema_slow"] = calc_ema(c, EMA_SLOW)
    df["ema200"]   = calc_ema(c, 200)
    df["adx14"]    = calc_adx(df, 14)
    df["atr14"]    = calc_atr(df, 14)

    # ATR Rank percentile (100-bar window)
    df["atr_rank_pct"] = df["atr14"].rolling(100).rank(pct=True) * 100

    # EMA200 slope (% over 10 bars)
    df["ema200_slope_pct"] = (df["ema200"] - df["ema200"].shift(10)) / df["ema200"].shift(10) * 100

    # 20-bar structure
    df["low20"]  = df["low"].rolling(20).min()
    df["high20"] = df["high"].rolling(20).max()
    df["dist_from_low20_pct"]  = (c - df["low20"])  / df["low20"]  * 100
    df["dist_from_high20_pct"] = (c - df["high20"]) / c * 100

    # Prior structure (shifted for look-ahead safety)
    df["low20_prev"]  = df["low"].shift(1).rolling(20).min()
    df["high20_prev"] = df["high"].shift(1).rolling(20).max()

    # Bollinger Band Width (20-bar, 4σ equivalent = upper-lower band / mid × 100)
    bb_mid        = c.rolling(20).mean()
    bb_std        = c.rolling(20).std(ddof=0)
    df["bb_width"] = (bb_std * 4) / bb_mid * 100

    # Volume
    vol            = df["vol"]
    df["vol_ma20"] = vol.rolling(20).mean()
    df["rel_vol"]  = vol / df["vol_ma20"]

    # ATR compression (for VCB)
    df["atr_50th"] = df["atr14"].rolling(50).quantile(0.30)
    df["compressed"] = df["atr14"] < df["atr_50th"]

    # VWAP rolling
    typical      = (df["high"] + df["low"] + c) / 3.0
    df["vwap24"] = (typical * vol).rolling(24).sum() / vol.rolling(24).sum()

    # Session
    df["hour_utc"] = df["datetime"].dt.hour
    df["session"]  = df["hour_utc"].apply(session_label)

    # Prior bars
    df["prev_close"] = c.shift(1)
    df["prev_low"]   = df["low"].shift(1)
    df["prev_high"]  = df["high"].shift(1)
    df["high_2ago"]  = df["high"].shift(2)

    # EMA200 rising flag
    df["ema200_rising"] = df["ema200"] > df["ema200"].shift(10)

    # FVG gap
    df["fvg_gap"] = df["low"] > df["high_2ago"] * 1.0001

    return df

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT FILTER THRESHOLDS (computed from pooled OOS data)
# ─────────────────────────────────────────────────────────────────────────────

def compute_env_thresholds(oos_dfs: dict) -> dict:
    pool = pd.concat(list(oos_dfs.values()), ignore_index=True)
    return {
        "bb_width_67": float(pool["bb_width"].quantile(0.67)),
    }

def env_filter_passes(row: pd.Series, thresholds: dict) -> bool:
    """Gate A: dist_from_low20_pct <= 5.0  AND  Gate B: bb_width >= 67th pct"""
    return (row["dist_from_low20_pct"] <= 5.0 and
            row["bb_width"] >= thresholds["bb_width_67"])

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL FUNCTIONS (vectorised, return boolean Series — no look-ahead)
# ─────────────────────────────────────────────────────────────────────────────

def signal_liq_sweep(df: pd.DataFrame) -> pd.Series:
    """Liquidity Sweep Reversal: wick below prior 5-bar low, closes above, bullish, above EMA200."""
    prior_low = df["low"].shift(1).rolling(5).min()
    sweep   = df["low"]   < prior_low
    reclaim = df["close"] > prior_low
    bullish = df["close"] > df["open"]
    trend   = df["close"] > df["ema200"]
    return (sweep & reclaim & bullish & trend).fillna(False)

def signal_fvg_slope(df: pd.DataFrame) -> pd.Series:
    """Bullish FVG + EMA200 + Positive Slope."""
    fvg   = df["fvg_gap"]
    trend = df["close"] > df["ema200"]
    slope = df["ema200_rising"]
    return (fvg & trend & slope).fillna(False)

def signal_break_struct(df: pd.DataFrame) -> pd.Series:
    """Break of Structure: close above prior 20-bar high."""
    structure_break = df["close"] > df["high20_prev"]
    trend           = df["close"] > df["ema200"]
    slope           = df["ema200_rising"]
    valid           = df["high20_prev"].notna()
    return (structure_break & trend & slope & valid).fillna(False)

def signal_vol_comp(df: pd.DataFrame) -> pd.Series:
    """Volatility Compression Breakout: ATR compressed, then close above 10-bar high."""
    high10_prev = df["high"].shift(1).rolling(10).max()
    breakout    = df["close"] > high10_prev
    trend       = df["close"] > df["ema200"]
    valid       = df["atr_50th"].notna() & high10_prev.notna()
    return (df["compressed"] & breakout & trend & valid).fillna(False)

def signal_donchian(df: pd.DataFrame) -> pd.Series:
    """Donchian Breakout: close above prior 20-bar high, above EMA200, rising EMA200."""
    breakout = df["close"] > df["high20_prev"]
    trend    = df["close"] > df["ema200"]
    slope    = df["ema200_rising"]
    valid    = df["high20_prev"].notna()
    return (breakout & trend & slope & valid).fillna(False)

def signal_ema_pullback(df: pd.DataFrame) -> pd.Series:
    """EMA Pullback (R024 control): price touches EMA_FAST in uptrend, closes above, ADX > 20."""
    touched = (
        (df["low"] <= df["ema_fast"]) |
        (df["low"].shift(1) <= df["ema_fast"].shift(1)) |
        (df["low"].shift(2) <= df["ema_fast"].shift(2))
    )
    uptrend = df["close"] > df["ema_slow"]
    bounce  = df["close"] > df["ema_fast"]
    trend   = df["adx14"] > 20
    return (uptrend & touched & bounce & trend).fillna(False)

STRATEGIES = {
    "LiqSweep":    signal_liq_sweep,
    "FVG+Slope":   signal_fvg_slope,
    "BreakStruct": signal_break_struct,
    "VolComp":     signal_vol_comp,
    "Donchian":    signal_donchian,
    "EMA_Pullback":signal_ema_pullback,
}

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE (identical to prior research — locked)
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, signal: pd.Series,
                 thresholds: dict, use_env_filter: bool,
                 label: str) -> list:
    min_sl    = CONFIG["MIN_SL_PCT"]
    rr        = CONFIG["RISK_REWARD"]
    max_lev   = CONFIG["MAX_LEVERAGE"]
    capital   = CONFIG["STARTING_CAPITAL"]
    risk_frac = CONFIG["RISK_PER_TRADE_PCT"]
    fee_rate  = CONFIG["TAKER_FEE"]
    spd_rate  = CONFIG["SPREAD"] * 0.5
    slp_rate  = CONFIG["SL_SLIPPAGE"]

    in_pos    = False
    entry_px  = stop = take = 0.0
    pos_size  = 0.0
    entry_tm  = None
    entry_i   = -1
    trades    = []

    for i in range(1, len(df)):
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_pos:
            hi, lo  = bar["high"], bar["low"]
            sl_dist = entry_px - stop
            sl_hit  = lo <= stop
            tp_hit  = hi >= take

            if sl_hit or tp_hit:
                exit_px   = (stop * (1.0 - slp_rate)) if sl_hit else take
                exit_type = "SL" if sl_hit else "TP"
                gross     = (exit_px - entry_px) * pos_size
                ne, nx    = entry_px * pos_size, exit_px * pos_size
                cost      = (ne + nx) * fee_rate + (ne + nx) * spd_rate
                slp_c     = (stop - exit_px) * pos_size if sl_hit else 0.0
                net       = gross - cost - slp_c
                rmul      = (exit_px - entry_px) / sl_dist if sl_dist > 0 else 0.0

                trades.append({
                    "label":       label,
                    "symbol":      label,
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
                    "holding_mins":(i - entry_i) * 60,
                    "dist_from_low20_pct": float(prev.get("dist_from_low20_pct", np.nan)),
                    "bb_width":    float(prev.get("bb_width", np.nan)),
                    "session":     prev.get("session", ""),
                })
                in_pos = False
            continue

        if signal.iloc[i - 1]:
            if use_env_filter and not env_filter_passes(prev, thresholds):
                continue

            ep      = bar["open"]
            sl      = prev["prev_low"]
            sl_dist = ep - sl

            if sl_dist <= 0 or sl_dist / ep < min_sl:
                sl      = prev["low"]
                sl_dist = ep - sl
                if sl_dist <= 0 or sl_dist / ep < min_sl:
                    continue

            tp          = ep + rr * sl_dist
            risk_usd    = capital * risk_frac
            sz          = min(risk_usd / sl_dist, (capital * max_lev) / ep)

            entry_px = ep; stop = sl; take = tp
            pos_size = sz; entry_tm = bar["datetime"]; entry_i = i
            in_pos   = True

    return trades

# ─────────────────────────────────────────────────────────────────────────────
# METRICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def metrics(trades: list, label: str = "") -> dict:
    if not trades:
        return {
            "label": label, "n": 0, "wr": 0.0, "pf": 0.0, "exp_r": 0.0,
            "net": 0.0, "sharpe": 0.0, "mdd": 0.0, "avg_hold": 0.0,
            "equity": np.array([CAPITAL]), "pnls": np.array([]), "wins": np.array([]),
        }
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values
    wins = df["win"].values.astype(bool)
    n, nw = len(pnl), wins.sum()
    nl    = n - nw
    gw    = pnl[wins].sum() if nw  else 0.0
    gl    = abs(pnl[~wins].sum()) if nl else 1e-9
    pf    = gw / gl
    wr    = nw / n
    rr    = CONFIG["RISK_REWARD"]
    exp_r = wr * rr - (1 - wr)
    eq    = CAPITAL + np.cumsum(pnl)
    peak  = np.maximum.accumulate(eq)
    dd    = (eq - peak) / peak
    mdd   = dd.min()
    std   = np.std(pnl, ddof=1) if n > 1 else 1e-9
    sharpe = (pnl.mean() / std * math.sqrt(n)) if std > 0 else 0.0
    avg_h  = df["holding_mins"].mean()
    return {
        "label": label, "n": n, "wr": wr, "pf": pf, "exp_r": exp_r,
        "net": float(pnl.sum()), "sharpe": sharpe, "mdd": mdd,
        "avg_hold": avg_h, "equity": eq, "pnls": pnl, "wins": wins,
    }

def monte_carlo(pnls: np.ndarray, n_iter: int = 2000) -> dict:
    if len(pnls) < 5:
        return {"prob_profit": 0.0, "p5": CAPITAL, "p50": CAPITAL, "p95": CAPITAL,
                "finals": np.array([CAPITAL])}
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
    rng  = np.random.default_rng(42)
    pfs  = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        pfs.append(wp / lp if lp > 0 else 2.0)
    return float(np.percentile(pfs, 5)), float(np.percentile(pfs, 50)), float(np.percentile(pfs, 95))

def loo_pf(sym_trades: dict) -> dict:
    result = {}
    for omit in sym_trades:
        flat = [t for s, tl in sym_trades.items() if s != omit for t in tl]
        if flat:
            result[omit] = metrics(flat)["pf"]
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #026" + " "*50 + "║")
print("║  Universal Environment Validation" + " "*45 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Hypothesis  : R023 environment (price near 20-bar low + wide BB) contains
                strategy-independent edge.
  Environment : Gate A: dist_from_low20 ≤ 5%  |  Gate B: BB Width ≥ 67th pct
  Strategies  : {', '.join(STRATEGIES.keys())}
  Symbols     : {', '.join(s.split('-')[0] for s in SYMBOLS)}
  Split       : 70/30 train/OOS  |  Engine: identical to R004/R024
""")

# Load OOS data
print("  Loading 1H OOS data …")
oos_dfs = {}
for sym in SYMBOLS:
    try:
        df      = load_1h(sym)
        df_feat = add_features(df)
        df_oos  = split_oos(df_feat)
        oos_dfs[sym] = df_oos
        print(f"  {sym.split('-')[0]:5s}  OOS bars={len(df_oos):,}")
    except FileNotFoundError:
        print(f"  {sym}: cache missing — skipped")

# Environment thresholds from pooled OOS data
pool_df    = pd.concat(list(oos_dfs.values()), ignore_index=True)
thresholds = compute_env_thresholds(oos_dfs)
print(f"\n  Environment thresholds (pooled OOS):")
print(f"    BB Width 67th pct = {thresholds['bb_width_67']:.4f}")
print(f"    Price near low    = dist_from_low20_pct <= 5.0 (fixed)")

# Check what fraction of bars satisfy each gate
gate_a_pct = (pool_df["dist_from_low20_pct"] <= 5.0).mean() * 100
gate_b_pct = (pool_df["bb_width"] >= thresholds["bb_width_67"]).mean() * 100
both_pct   = ((pool_df["dist_from_low20_pct"] <= 5.0) &
              (pool_df["bb_width"] >= thresholds["bb_width_67"])).mean() * 100
print(f"\n  Gate A alone (dist ≤ 5%):  {gate_a_pct:.1f}% of all OOS bars")
print(f"  Gate B alone (BB ≥ 67th):  {gate_b_pct:.1f}% of all OOS bars")
print(f"  Both gates combined:       {both_pct:.1f}% of all OOS bars (theoretical signal window)")

# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL STRATEGY × (BASELINE / FILTERED) COMBINATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*78)
print("  Running backtests …")

# results[strat_name][sym]["baseline"|"filtered"] = trades list
results     = {s: {} for s in STRATEGIES}
port_metrics = {}  # (strat, variant) -> metrics dict

for strat_name, sig_fn in STRATEGIES.items():
    print(f"\n  {strat_name}:")
    for sym in oos_dfs:
        df_oos = oos_dfs[sym]
        sig    = sig_fn(df_oos)
        base_t = run_backtest(df_oos, sig, thresholds, False, sym)
        filt_t = run_backtest(df_oos, sig, thresholds, True,  sym)
        results[strat_name][sym] = {"baseline": base_t, "filtered": filt_t}
        tag = sym.split("-")[0]
        bm  = metrics(base_t)
        fm  = metrics(filt_t)
        ret = f"{fm['n']/max(bm['n'],1)*100:.0f}%"
        print(f"    {tag:5s}  base n={bm['n']:4d} PF={bm['pf']:.3f}  "
              f"filt n={fm['n']:4d} PF={fm['pf']:.3f}  δPF={fm['pf']-bm['pf']:+.3f}  retain={ret}")

    # Portfolio-level
    flat_base = [t for s in oos_dfs for t in results[strat_name][s]["baseline"]]
    flat_filt = [t for s in oos_dfs for t in results[strat_name][s]["filtered"]]
    port_metrics[(strat_name, "baseline")] = metrics(flat_base, f"{strat_name}_base")
    port_metrics[(strat_name, "filtered")] = metrics(flat_filt, f"{strat_name}_filt")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  PORTFOLIO RESULTS: BASELINE vs ENV-FILTERED")
print("═"*78)
print(f"  {'Strategy':12s}  {'Variant':10s}  {'n':>5}  {'WR':>6}  {'PF':>7}  "
      f"{'ExpR':>7}  {'Sharpe':>7}  {'MDD':>7}  {'Net $':>9}  {'Retain':>7}")
print("  " + "─"*78)

for strat_name in STRATEGIES:
    bm = port_metrics[(strat_name, "baseline")]
    fm = port_metrics[(strat_name, "filtered")]
    ret = f"{fm['n']/max(bm['n'],1)*100:.0f}%"
    dpf = fm["pf"] - bm["pf"]
    arrow = "▲" if dpf > 0 else "▼"
    print(f"  {strat_name:12s}  {'BASE':10s}  {bm['n']:5d}  {bm['wr']*100:5.1f}%  "
          f"{bm['pf']:7.3f}  {bm['exp_r']:+7.3f}  {bm['sharpe']:7.2f}  "
          f"{bm['mdd']*100:6.1f}%  {bm['net']:+9.0f}  {'100%':>7}")
    print(f"  {strat_name:12s}  {'ENV-FILTER':10s}  {fm['n']:5d}  {fm['wr']*100:5.1f}%  "
          f"{fm['pf']:7.3f}  {fm['exp_r']:+7.3f}  {fm['sharpe']:7.2f}  "
          f"{fm['mdd']*100:6.1f}%  {fm['net']:+9.0f}  {ret:>7}  {arrow} δPF={dpf:+.3f}")
    print("  " + "·"*50)

# ─────────────────────────────────────────────────────────────────────────────
# DELTA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*78)
print("  DELTA ANALYSIS (Filtered vs Baseline)")
print("─"*78)
print(f"  {'Strategy':14s}  {'δPF':>8}  {'δWR':>7}  {'δMDD':>8}  {'δNet $':>9}  "
      f"{'PF_filt':>8}  {'n_filt':>7}  {'Verdict'}")
print("  " + "─"*74)

delta_results = {}
for strat_name in STRATEGIES:
    bm = port_metrics[(strat_name, "baseline")]
    fm = port_metrics[(strat_name, "filtered")]
    dpf  = fm["pf"]    - bm["pf"]
    dwr  = (fm["wr"]   - bm["wr"])   * 100
    dmdd = (fm["mdd"]  - bm["mdd"])  * 100   # more negative = worse drawdown
    dnet = fm["net"]   - bm["net"]
    ret  = fm["n"] / max(bm["n"], 1)
    ok_pf  = fm["pf"]  > 1.20
    ok_n   = fm["n"]   >= 30
    ok_ret = ret       >= 0.30
    ok_exp = fm["exp_r"] > 0
    verdict = "PROMOTE" if (ok_pf and ok_n and ok_ret and ok_exp) else \
              "WATCHLIST" if (fm["pf"] >= 1.0 and ok_exp and ok_n) else "REJECT"
    delta_results[strat_name] = {
        "dpf": dpf, "dwr": dwr, "dmdd": dmdd, "dnet": dnet,
        "ret": ret, "verdict": verdict, "fm": fm, "bm": bm,
    }
    v_sym = {"PROMOTE": "✓ PROMOTE", "WATCHLIST": "~ WATCHLIST", "REJECT": "✗ REJECT"}
    print(f"  {strat_name:14s}  {dpf:+8.3f}  {dwr:+6.1f}%  {dmdd:+7.1f}%  {dnet:+9.0f}  "
          f"{fm['pf']:8.3f}  {fm['n']:7d}  {v_sym[verdict]}")

best_strat   = max(STRATEGIES, key=lambda s: delta_results[s]["dpf"])
best_abs_pf  = max(STRATEGIES, key=lambda s: delta_results[s]["fm"]["pf"])
print(f"\n  Largest δPF:   {best_strat}  ({delta_results[best_strat]['dpf']:+.3f})")
print(f"  Best abs PF:   {best_abs_pf}  (PF={delta_results[best_abs_pf]['fm']['pf']:.3f}  "
      f"n={delta_results[best_abs_pf]['fm']['n']})")

n_improved   = sum(1 for s in STRATEGIES if delta_results[s]["dpf"] > 0)
n_promoted   = sum(1 for s in STRATEGIES if delta_results[s]["verdict"] == "PROMOTE")
n_watchlist  = sum(1 for s in STRATEGIES if delta_results[s]["verdict"] == "WATCHLIST")
print(f"  Strategies improved: {n_improved}/{len(STRATEGIES)}")
print(f"  Strategies promoted: {n_promoted}  watchlisted: {n_watchlist}")

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SYMBOL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*78)
print("  CROSS-SYMBOL ANALYSIS  (Filtered PF per symbol per strategy)")
print("─"*78)
header = f"  {'Symbol':7s}  " + "  ".join(f"{s[:10]:>10s}" for s in STRATEGIES)
print(header)
print("  " + "─"*72)
for sym in oos_dfs:
    tag   = sym.split("-")[0]
    line  = f"  {tag:7s}  "
    for strat_name in STRATEGIES:
        t  = results[strat_name][sym]["filtered"]
        m  = metrics(t)
        pf = m["pf"]
        line += f"  {pf:>10.3f}"
    print(line)

# Symbols where filtered PF > 1.0 for each strategy
print()
for strat_name in STRATEGIES:
    syms_above = [s.split("-")[0] for s in oos_dfs
                  if metrics(results[strat_name][s]["filtered"])["pf"] > 1.0 and
                     metrics(results[strat_name][s]["filtered"])["n"] >= 5]
    print(f"  {strat_name:14s}: {len(syms_above)}/6 symbols PF>1.0  "
          f"({', '.join(syms_above) if syms_above else 'none'})")

# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS — BEST FILTERED STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*78)
print(f"  ROBUSTNESS TESTS — Best filtered strategy: {best_abs_pf}")
print("─"*78)

flat_best = [t for s in oos_dfs for t in results[best_abs_pf][s]["filtered"]]
mc_best   = monte_carlo(np.array([t["pnl"] for t in flat_best]))
b5, b50, b95 = bootstrap_pf(np.array([t["pnl"] for t in flat_best]))

print(f"\n  Monte Carlo (2,000 iter):")
print(f"    P(profit) = {mc_best['prob_profit']*100:.1f}%")
print(f"    p5/p50/p95 equity: ${mc_best['p5']:,.0f} / ${mc_best['p50']:,.0f} / ${mc_best['p95']:,.0f}")

print(f"\n  Bootstrap 90% CI on PF:")
print(f"    [p5={b5:.3f}, p50={b50:.3f}, p95={b95:.3f}]")
print(f"    Lower CI (p5) > 1.0: {'YES ✓' if b5 > 1.0 else 'NO ✗'}")
print(f"    Lower CI (p5) > 1.2: {'YES ✓' if b5 > 1.2 else 'NO ✗'}")

loo = loo_pf({s: results[best_abs_pf][s]["filtered"] for s in oos_dfs})
print(f"\n  Leave-one-symbol-out PF ({best_abs_pf} filtered):")
for sym in oos_dfs:
    print(f"    Leave out {sym.split('-')[0]:5s}: PF={loo.get(sym, 0.0):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  DIAGNOSTIC QUESTIONS")
print("═"*78)

# Q1: Does the environment improve every strategy?
improves_all = n_improved == len(STRATEGIES)
print(f"""
  Q1. Does the environment improve every strategy?
      {n_improved}/{len(STRATEGIES)} strategies show δPF > 0
      Answer: {'YES — environment is universal ✓' if improves_all else f'NO — only {n_improved} of {len(STRATEGIES)} improved'}

  Q2. Which strategy benefits most?
      {best_strat}  δPF={delta_results[best_strat]['dpf']:+.3f}
      Baseline PF={delta_results[best_strat]['bm']['pf']:.3f}  →  Filtered PF={delta_results[best_strat]['fm']['pf']:.3f}

  Q3. Does any strategy exceed PF 1.20 after filtering?
      {f"YES — {best_abs_pf} reaches PF={delta_results[best_abs_pf]['fm']['pf']:.3f} ✓" if delta_results[best_abs_pf]['fm']['pf'] > 1.20 else f"NO — best is {best_abs_pf} at PF={delta_results[best_abs_pf]['fm']['pf']:.3f} ✗"}

  Q4. Is improvement consistent across symbols?
""")

for strat_name in STRATEGIES:
    syms_above = sum(1 for s in oos_dfs
                     if metrics(results[strat_name][s]["filtered"])["pf"] > 1.0 and
                        metrics(results[strat_name][s]["filtered"])["n"] >= 5)
    print(f"      {strat_name:14s}: {syms_above}/6 symbols filtered PF>1.0")

print(f"""
  Q5. Demo-worthy strategy?
      {'YES: ' + next((s for s in STRATEGIES if delta_results[s]['verdict'] == 'PROMOTE'), 'none') if n_promoted > 0 else 'NO — no strategy meets all PROMOTE criteria'}

  Q6. Is PF≈2 from R023 reproducible or descriptive?
      R023 PF≈2 was from a feature attribution table, not a live backtest.
      This study tests whether the environment translates to a live filtered PF.
      Result: Best filtered PF = {delta_results[best_abs_pf]['fm']['pf']:.3f}
      Answer: {'Partially reproducible — edge exists but below R023 attribution level' if delta_results[best_abs_pf]['fm']['pf'] > 1.0 else 'NOT reproducible — filter does not lift PF in live backtest'}
""")

# ─────────────────────────────────────────────────────────────────────────────
# OVERALL VERDICT
# ─────────────────────────────────────────────────────────────────────────────

def get_overall_verdict():
    best_fm = delta_results[best_abs_pf]["fm"]
    best_bm = delta_results[best_abs_pf]["bm"]
    ret     = best_fm["n"] / max(best_bm["n"], 1)
    syms_ok = sum(1 for s in oos_dfs
                  if metrics(results[best_abs_pf][s]["filtered"])["pf"] > 1.0 and
                     metrics(results[best_abs_pf][s]["filtered"])["n"] >= 5)
    ok_pf   = best_fm["pf"] > 1.20
    ok_n    = best_fm["n"]  >= 30
    ok_sym  = syms_ok       >= 4
    ok_ret  = ret           >= 0.30
    ok_exp  = best_fm["exp_r"] > 0

    if ok_pf and ok_n and ok_sym and ok_ret and ok_exp:
        return "PROMOTE", best_abs_pf, (f"PF={best_fm['pf']:.3f}  n={best_fm['n']}  "
                                         f"{syms_ok}/6 symbols")
    elif best_fm["pf"] >= 1.0 and ok_exp and n_improved >= 4:
        return "WATCHLIST", best_abs_pf, (f"Environment shows consistent improvement "
                                           f"but PF={best_fm['pf']:.3f} below 1.20 threshold")
    elif n_improved >= 4:
        return "WATCHLIST", best_abs_pf, f"Edge directionally positive in {n_improved}/6 strategies"
    else:
        return "REJECT", best_abs_pf, f"Environment does not consistently improve strategies"

VERDICT, VERDICT_STRAT, VERDICT_REASON = get_overall_verdict()
VCOLOUR = "\033[92m" if VERDICT == "PROMOTE" else "\033[93m" if VERDICT == "WATCHLIST" else "\033[91m"
VRESET  = "\033[0m"

print(f"{'═'*78}")
print(f"  OVERALL VERDICT: {VCOLOUR}{VERDICT}{VRESET}")
print(f"  Best strategy:   {VERDICT_STRAT}")
print(f"  Reason:          {VERDICT_REASON}")
print(f"{'═'*78}")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def dark_ax(ax, title=None, col="white"):
    ax.set_facecolor("#111")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=col, fontsize=9)

print("\n  Generating charts …")

strat_names = list(STRATEGIES.keys())
n_strats    = len(strat_names)

# ── Chart 1: PF comparison — baseline vs filtered for all strategies ──────────
fig, axes = plt.subplots(1, 3, figsize=(22, 6), facecolor="#111")
fig.suptitle("R026 — Universal Environment: Baseline vs Filtered PF", color="white", fontsize=12)

ax = axes[0]
dark_ax(ax, "Profit Factor — Baseline (grey) vs Filtered (colour)")
x_   = np.arange(n_strats); w = 0.38
base_pfs = [port_metrics[(s, "baseline")]["pf"] for s in strat_names]
filt_pfs = [port_metrics[(s, "filtered")]["pf"] for s in strat_names]
cols = [STRAT_COLOURS[s] for s in strat_names]
ax.bar(x_ - w/2, base_pfs, w, color="#555", alpha=0.85, label="Baseline")
ax.bar(x_ + w/2, filt_pfs, w, color=cols,  alpha=0.85, label="Filtered")
ax.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5)
ax.axhline(1.2, color="#FF9800", lw=0.8, ls=":", alpha=0.6, label="PF=1.20 target")
ax.set_xticks(x_); ax.set_xticklabels(strat_names, rotation=35, ha="right", fontsize=7, color="white")
ax.set_ylabel("Profit Factor", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=7)
for b, p_ in zip(ax.patches[n_strats:], filt_pfs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            f"{p_:.2f}", ha="center", color="white", fontsize=7)

ax2 = axes[1]
dark_ax(ax2, "δPF: Filtered minus Baseline")
dpfs   = [delta_results[s]["dpf"] for s in strat_names]
d_cols = ["#4CAF50" if d > 0 else "#F44336" for d in dpfs]
bars   = ax2.bar(strat_names, dpfs, color=d_cols, alpha=0.85)
ax2.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
ax2.set_xticklabels(strat_names, rotation=35, ha="right", fontsize=7, color="white")
ax2.set_ylabel("δPF", color="white")
for b, d in zip(bars, dpfs):
    y_ = b.get_height() + 0.005 if d >= 0 else b.get_height() - 0.02
    ax2.text(b.get_x()+b.get_width()/2, y_, f"{d:+.3f}",
             ha="center", color="white", fontsize=8)

ax3 = axes[2]
dark_ax(ax3, "Trade Retention % (Filtered/Baseline)")
rets_ = [delta_results[s]["ret"]*100 for s in strat_names]
rc    = ["#4CAF50" if r >= 30 else "#FF9800" if r >= 15 else "#F44336" for r in rets_]
ax3.bar(strat_names, rets_, color=rc, alpha=0.85)
ax3.axhline(30, color="#FF9800", lw=1, ls="--", alpha=0.6, label="30% floor")
ax3.set_xticklabels(strat_names, rotation=35, ha="right", fontsize=7, color="white")
ax3.set_ylabel("Retention %", color="white")
ax3.legend(facecolor="#222", labelcolor="white", fontsize=7)
for b, r in zip(ax3.patches, rets_):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
             f"{r:.0f}%", ha="center", color="white", fontsize=8)

plt.tight_layout()
p = f"{OUT}/r026_pf_overview.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 2: Cross-symbol heatmap ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 6), facecolor="#111")
fig.suptitle("R026 — Cross-Symbol Filtered PF Heatmap", color="white", fontsize=12)

for ax_i, (variant, var_label) in enumerate([("baseline", "Baseline"), ("filtered", "Filtered")]):
    ax = axes[ax_i]
    ax.set_facecolor("#111")
    sym_tags = [s.split("-")[0] for s in oos_dfs]
    matrix   = np.zeros((n_strats, len(sym_tags)))
    for si, strat_name in enumerate(strat_names):
        for syi, sym in enumerate(oos_dfs):
            m = metrics(results[strat_name][sym][variant])
            matrix[si, syi] = m["pf"]
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=2.0)
    ax.set_xticks(range(len(sym_tags))); ax.set_xticklabels(sym_tags, color="white", fontsize=9)
    ax.set_yticks(range(n_strats));      ax.set_yticklabels(strat_names, color="white", fontsize=9)
    ax.set_title(f"PF Heatmap — {var_label}", color="white", fontsize=10)
    for si in range(n_strats):
        for syi in range(len(sym_tags)):
            v = matrix[si, syi]
            ax.text(syi, si, f"{v:.2f}", ha="center", va="center",
                    color="black" if 0.8 < v < 1.5 else "white", fontsize=8, fontweight="bold")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
p = f"{OUT}/r026_symbol_heatmap.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 3: Equity curves — filtered strategies ──────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(22, 10), facecolor="#111")
fig.suptitle("R026 — Equity Curves: Filtered Strategies (Portfolio)", color="white", fontsize=12)
ax_flat = axes.flatten()

for si, strat_name in enumerate(strat_names):
    ax   = ax_flat[si]
    col  = STRAT_COLOURS[strat_name]
    bm   = port_metrics[(strat_name, "baseline")]
    fm   = port_metrics[(strat_name, "filtered")]
    dark_ax(ax, f"{strat_name}  base PF={bm['pf']:.2f}  filt PF={fm['pf']:.2f}", col)
    if bm["n"] > 0:
        ax.plot(bm["equity"], color="#555", lw=1.2, ls="--", alpha=0.7, label=f"Base n={bm['n']}")
    if fm["n"] > 0:
        ax.plot(fm["equity"], color=col, lw=1.5, label=f"Filt n={fm['n']}")
    ax.axhline(CAPITAL, color="white", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlabel("Trade #", color="white", fontsize=7)
    ax.set_ylabel("Equity $", color="white", fontsize=7)
    ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=7, loc="upper left")

plt.tight_layout()
p = f"{OUT}/r026_equity_curves.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 4: Cross-symbol δPF per strategy ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(22, 10), facecolor="#111")
fig.suptitle("R026 — δPF by Symbol (Filtered − Baseline)", color="white", fontsize=12)
ax_flat = axes.flatten()
sym_tags = [s.split("-")[0] for s in oos_dfs]

for si, strat_name in enumerate(strat_names):
    ax  = ax_flat[si]
    col = STRAT_COLOURS[strat_name]
    dark_ax(ax, f"{strat_name}", col)
    dpfs_sym = []
    for sym in oos_dfs:
        bm_ = metrics(results[strat_name][sym]["baseline"])
        fm_ = metrics(results[strat_name][sym]["filtered"])
        dpfs_sym.append(fm_["pf"] - bm_["pf"])
    d_cols = ["#4CAF50" if d > 0 else "#F44336" for d in dpfs_sym]
    bars = ax.bar(sym_tags, dpfs_sym, color=d_cols, alpha=0.85)
    ax.axhline(0, color="white", lw=0.7, ls="--", alpha=0.5)
    ax.set_ylabel("δPF", color="white", fontsize=7)
    for b, d in zip(bars, dpfs_sym):
        y_ = b.get_height() + 0.01 if d >= 0 else b.get_height() - 0.04
        ax.text(b.get_x()+b.get_width()/2, y_, f"{d:+.2f}",
                ha="center", color="white", fontsize=8)

plt.tight_layout()
p = f"{OUT}/r026_delta_pf_by_symbol.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 5: Monte Carlo — best filtered strategy ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#111")
fig.suptitle(f"R026 — Robustness: {best_abs_pf} (Filtered)", color="white", fontsize=12)

ax = axes[0]
dark_ax(ax, f"Monte Carlo  P(profit)={mc_best['prob_profit']*100:.1f}%")
fe = mc_best["finals"]
if fe.max() > fe.min():
    ax.hist(fe, bins=np.linspace(fe.min(), fe.max(), 51), color="#4CAF50", alpha=0.70, edgecolor="none")
for pv, pc, pl in [(5, "#F44336", "p5"), (50, "#4CAF50", "p50"), (95, "#FF9800", "p95")]:
    v = np.percentile(fe, pv)
    ax.axvline(v, color=pc, lw=1.5, ls="--", label=f"{pl} ${v:,.0f}")
ax.axvline(CAPITAL, color="white", lw=1, ls=":", alpha=0.5, label=f"Start ${CAPITAL:,}")
ax.set_xlabel("Final Equity $", color="white"); ax.set_ylabel("Count", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=8)

ax2 = axes[1]
dark_ax(ax2, f"Bootstrap 90% CI on PF — {best_abs_pf}")
ax2.errorbar(0, b50, yerr=[[b50-b5], [b95-b50]], fmt="o",
             color="#4CAF50", capsize=14, capthick=3, ms=12)
ax2.text(0, b95+0.02, f"p95={b95:.3f}", ha="center", color="#FF9800", fontsize=10)
ax2.text(0, b5-0.04, f"p5={b5:.3f}", ha="center", color="#F44336", fontsize=10)
ax2.axhline(1.0, color="white", lw=1, ls="--", alpha=0.5, label="PF=1.0")
ax2.axhline(1.2, color="#FF9800", lw=1, ls=":", alpha=0.6, label="PF=1.2 target")
ax2.set_xticks([0]); ax2.set_xticklabels([f"{best_abs_pf}\n(Filtered)"], color="white")
ax2.set_ylabel("PF", color="white")
ax2.legend(facecolor="#222", labelcolor="white", fontsize=9)

plt.tight_layout()
p = f"{OUT}/r026_robustness.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 6: Full Dashboard ───────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 16), facecolor="#0a0a0a")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.65, wspace=0.45)
vcolor = "#4CAF50" if VERDICT == "PROMOTE" else "#FF9800" if VERDICT == "WATCHLIST" else "#F44336"
fig.suptitle(
    f"QUANTLAB AI — R026 DASHBOARD\n"
    f"Universal Environment Validation | Verdict: {VERDICT}",
    color="white", fontsize=13, y=0.99)

# Summary table
ax_t = fig.add_subplot(gs[0, :])
ax_t.axis("off")
rows = []
for strat_name in strat_names:
    bm = port_metrics[(strat_name, "baseline")]
    fm = port_metrics[(strat_name, "filtered")]
    ret = f"{fm['n']/max(bm['n'],1)*100:.0f}%"
    dpf = fm["pf"] - bm["pf"]
    rows.append([
        strat_name,
        f"{bm['n']}/{fm['n']}", f"{bm['wr']*100:.1f}%/{fm['wr']*100:.1f}%",
        f"{bm['pf']:.3f}/{fm['pf']:.3f}", f"{dpf:+.3f}",
        f"{fm['exp_r']:+.3f}", f"{fm['sharpe']:.2f}", f"{fm['mdd']*100:.1f}%",
        f"${fm['net']:+,.0f}", ret, delta_results[strat_name]["verdict"],
    ])
hdrs = ["Strategy","n (B/F)","WR (B/F)","PF (B/F)","δPF","ExpR","Sharpe","MDD","Net $","Ret","Verdict"]
tbl  = ax_t.table(cellText=rows, colLabels=hdrs, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r % 2 == 0 else "#222")
    cell.set_text_props(color="white"); cell.set_edgecolor("#333")
    if r == 0:
        cell.set_facecolor("#2a2a2a"); cell.set_text_props(color="#aaa", fontweight="bold")

# PF comparison
ax_pf = fig.add_subplot(gs[1, 0])
dark_ax(ax_pf, "PF: Baseline vs Filtered")
x_ = np.arange(n_strats); w = 0.4
ax_pf.bar(x_ - w/2, base_pfs, w, color="#555", alpha=0.8)
ax_pf.bar(x_ + w/2, filt_pfs, w, color=[STRAT_COLOURS[s] for s in strat_names], alpha=0.85)
ax_pf.axhline(1.0, color="white", lw=0.7, ls="--")
ax_pf.axhline(1.2, color="#FF9800", lw=0.7, ls=":")
ax_pf.set_xticks(x_); ax_pf.set_xticklabels(strat_names, rotation=40, ha="right", fontsize=6, color="white")
ax_pf.set_ylabel("PF", color="white", fontsize=7)

# δPF
ax_dpf = fig.add_subplot(gs[1, 1])
dark_ax(ax_dpf, "δPF (Filtered − Baseline)")
ax_dpf.bar(strat_names, [delta_results[s]["dpf"] for s in strat_names],
           color=["#4CAF50" if delta_results[s]["dpf"] > 0 else "#F44336" for s in strat_names],
           alpha=0.85)
ax_dpf.axhline(0, color="white", lw=0.7, ls="--")
ax_dpf.set_xticks(range(n_strats)); ax_dpf.set_xticklabels(strat_names, rotation=40, ha="right", fontsize=6, color="white")
ax_dpf.set_ylabel("δPF", color="white", fontsize=7)

# Symbol heatmap (filtered)
ax_hm = fig.add_subplot(gs[1, 2:])
ax_hm.set_facecolor("#111")
matrix = np.zeros((n_strats, len(sym_tags)))
for si, sn in enumerate(strat_names):
    for syi, sym in enumerate(oos_dfs):
        matrix[si, syi] = metrics(results[sn][sym]["filtered"])["pf"]
im = ax_hm.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=2.0)
ax_hm.set_xticks(range(len(sym_tags))); ax_hm.set_xticklabels(sym_tags, color="white", fontsize=8)
ax_hm.set_yticks(range(n_strats));      ax_hm.set_yticklabels(strat_names, color="white", fontsize=8)
ax_hm.set_title("Filtered PF Heatmap", color="white", fontsize=9)
for si in range(n_strats):
    for syi in range(len(sym_tags)):
        v = matrix[si, syi]
        ax_hm.text(syi, si, f"{v:.2f}", ha="center", va="center",
                   color="black" if 0.8 < v < 1.5 else "white", fontsize=7, fontweight="bold")

# Equity of best strategy
ax_eq = fig.add_subplot(gs[2, :2])
dark_ax(ax_eq, f"{best_abs_pf} — Equity (Base grey, Filtered colour)")
col = STRAT_COLOURS[best_abs_pf]
bm_ = port_metrics[(best_abs_pf, "baseline")]
fm_ = port_metrics[(best_abs_pf, "filtered")]
if bm_["n"] > 0:
    ax_eq.plot(bm_["equity"], color="#555", lw=1.2, ls="--", alpha=0.7, label=f"Baseline n={bm_['n']}")
if fm_["n"] > 0:
    ax_eq.plot(fm_["equity"], color=col, lw=1.5, label=f"Filtered n={fm_['n']} PF={fm_['pf']:.2f}")
ax_eq.axhline(CAPITAL, color="white", lw=0.5, ls=":", alpha=0.4)
ax_eq.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
ax_eq.set_xlabel("Trade #", color="white", fontsize=7)
ax_eq.set_ylabel("Equity $", color="white", fontsize=7)

# Verdict panel
ax_v = fig.add_subplot(gs[2, 2:])
ax_v.axis("off"); ax_v.set_facecolor("#111")
ax_v.text(0.5, 0.82, f"VERDICT: {VERDICT}", transform=ax_v.transAxes,
          color=vcolor, fontsize=20, ha="center", fontweight="bold")
ax_v.text(0.5, 0.68, VERDICT_REASON, transform=ax_v.transAxes,
          color="white", fontsize=10, ha="center")
summary_txt = (
    f"Best strategy: {best_abs_pf}\n"
    f"Filtered PF={delta_results[best_abs_pf]['fm']['pf']:.3f}  n={delta_results[best_abs_pf]['fm']['n']}\n"
    f"Bootstrap p5={b5:.3f}  MC P(profit)={mc_best['prob_profit']*100:.1f}%\n"
    f"Strategies improved: {n_improved}/{len(STRATEGIES)}\n"
    f"Environment bars: {both_pct:.1f}% of OOS data"
)
ax_v.text(0.5, 0.38, summary_txt, transform=ax_v.transAxes,
          color="#aaa", fontsize=10, ha="center", va="center")

plt.savefig(f"{OUT}/r026_dashboard.png", dpi=130, bbox_inches="tight",
            facecolor="#0a0a0a"); plt.close()
print(f"  → {OUT}/r026_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE TRADE LOG + JOURNAL
# ─────────────────────────────────────────────────────────────────────────────

# Save best filtered strategy trades
best_trades = [t for s in oos_dfs for t in results[best_abs_pf][s]["filtered"]]
if best_trades:
    df_log = pd.DataFrame(best_trades)
    log_path = f"{OUT}/r026_trade_log_{best_abs_pf.lower()}.csv"
    df_log.to_csv(log_path, index=False)
    print(f"  → {log_path}  ({len(df_log)} trades)")

# Journal
try:
    from quantlab_ai import append_journal
    from datetime import datetime, timezone as _tz
    journal_rows = []
    run_date = datetime.now(tz=_tz.utc).strftime("%Y-%m-%d")
    for strat_name in strat_names:
        for var in ("baseline", "filtered"):
            m   = port_metrics[(strat_name, var)]
            mc_ = monte_carlo(m["pnls"], n_iter=500)
            journal_rows.append({
                "research_id":      RESEARCH_ID,
                "run_date":         run_date,
                "strategy_name":    f"{strat_name}_{var}",
                "symbol":           "PORTFOLIO",
                "n_trades":         m["n"],
                "profit_factor":    round(m["pf"],    4),
                "expectancy_r":     round(m["exp_r"], 4),
                "win_rate":         round(m["wr"],    4),
                "net_profit":       round(m["net"],   2),
                "max_drawdown":     round(m["mdd"],   4),
                "sharpe":           round(m["sharpe"],4),
                "mc_prob_profit":   round(mc_["prob_profit"], 4),
                "avg_hold_minutes": round(m["avg_hold"], 1),
                "verdict":          delta_results[strat_name]["verdict"] if var == "filtered" else "BASELINE",
            })
    append_journal(journal_rows)
    print(f"  Journal updated → {CONFIG['JOURNAL_FILE']}")
except Exception as e:
    print(f"  [WARN] Journal: {e}")

print(f"\n{'═'*78}")
print(f"  R026 complete.")
print(f"  Verdict  : {VERDICT}")
print(f"  Best     : {best_abs_pf}  PF={delta_results[best_abs_pf]['fm']['pf']:.3f}")
print(f"  Improved : {n_improved}/{len(STRATEGIES)} strategies")
print(f"  Output   → {OUT}/r026_*")
print(f"{'═'*78}\n")
