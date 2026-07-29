"""
=============================================================================
QUANTLAB AI — RESEARCH #036
Environment-Native Strategy Research
=============================================================================

R035 discovered:  ATR Rank LOW + EMA200 Slope POSITIVE +
                  Distance above EMA200 HIGH + BB Width NARROW
  → n=190  WR=49.5%  PF=1.356  (pooled, all entries)

R036 asks: Can a strategy designed FOR this environment achieve PF>1.20 OOS?

Three entry families tested ONLY inside the qualifying environment:

  A — Mean Reversion   : enter long when price pulls back to EMA20 (wicks EMA20
                         from above, closes above)
  B — Momentum         : enter long when close > prev bar high (breakout)
  C — RelVol Breakout  : enter long when vol spike (>1.5× 20-bar avg) and
                         bullish candle (close > open, close > prev close)

Shared exit:  Stop = 1×ATR14,  Target = 2×ATR14  (fixed 2R)

Environment gate (all 4 must be true — thresholds from IS period only):
  1. ATR Rank  < IS p25
  2. EMA200 slope > 0  (ema200 > ema200.shift(10))
  3. EMA Distance above EMA200 > IS p75
  4. BB Width  < IS p33

Walk-forward: 5-fold expanding, same as R033/R034.
Dataset: 23 symbols × 27 months 1H.

PROMOTE: PF>1.20 · n≥100 · Boot p50>1.20 · MC P>60%
         LOO-sym floor>1.0 · LOO-fold floor>1.0 · MDD<25%
=============================================================================
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
from quantlab_ai import CONFIG, calc_ema, calc_atr

RESEARCH_ID = "R036"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = CONFIG["RISK_REWARD"]   # 2.0
BEP_WR   = 1.0 / (1.0 + RR)       # 33.33%

SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
    "LINK-USDT-SWAP", "AVAX-USDT-SWAP", "XRP-USDT-SWAP",
    "LTC-USDT-SWAP", "BCH-USDT-SWAP", "DOGE-USDT-SWAP",
    "ADA-USDT-SWAP", "BNB-USDT-SWAP", "DOT-USDT-SWAP",
    "ARB-USDT-SWAP", "OP-USDT-SWAP", "NEAR-USDT-SWAP",
    "ATOM-USDT-SWAP", "SUI-USDT-SWAP", "APT-USDT-SWAP",
    "WIF-USDT-SWAP", "PEPE-USDT-SWAP", "ENA-USDT-SWAP",
    "UNI-USDT-SWAP", "FIL-USDT-SWAP",
]
MIN_BARS = 4_000

FOLDS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]
N_BOOT = 5_000

COLOURS = {
    "BTC-USDT-SWAP": "#F7931A", "ETH-USDT-SWAP": "#627EEA",
    "SOL-USDT-SWAP": "#9945FF", "LINK-USDT-SWAP": "#2A5ADA",
    "AVAX-USDT-SWAP":"#E84142", "XRP-USDT-SWAP":  "#346AA9",
    "LTC-USDT-SWAP": "#BFBBBB", "BCH-USDT-SWAP":  "#8DC351",
    "DOGE-USDT-SWAP":"#C3A634", "ADA-USDT-SWAP":  "#0033AD",
    "BNB-USDT-SWAP": "#F3BA2F", "DOT-USDT-SWAP":  "#E6007A",
    "ARB-USDT-SWAP": "#28A0F0", "OP-USDT-SWAP":   "#FF0420",
    "NEAR-USDT-SWAP":"#00C08B", "ATOM-USDT-SWAP": "#6F4CFF",
    "SUI-USDT-SWAP": "#6FBCF0", "APT-USDT-SWAP":  "#00B4D8",
    "WIF-USDT-SWAP": "#A67C52", "PEPE-USDT-SWAP": "#4CAF50",
    "ENA-USDT-SWAP": "#8B0000", "UNI-USDT-SWAP":  "#FF007A",
    "FIL-USDT-SWAP": "#0090FF",
}

STRAT_COLS = {"A": "#4CAF50", "B": "#2196F3", "C": "#FF9800"}

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #036" + " "*50 + "║")
print("║  Environment-Native Strategy Research" + " "*41 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Environment  : ATR_Rank<p25 · EMA200_slope>0 · EMA_dist>p75 · BB_width<p33
  Entry A      : Mean Reversion — pullback to EMA20 (wick, close above)
  Entry B      : Momentum       — close > prev bar high
  Entry C      : RelVol Breakout— vol > 1.5× avg AND bullish candle
  Exit         : Stop=1×ATR14  Target=2×ATR14  (fixed 2R)
  Method       : 5-fold expanding walk-forward, IS thresholds only
  PROMOTE bar  : PF>1.20 · n≥100 · boot_p50>1.20 · MC P>60%
                 LOO-sym>1.0 · LOO-fold>1.0 · MDD<25%
""")

# =============================================================================
# INDICATORS
# =============================================================================

def _calc_rsi(close, length=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag    = gain.ewm(alpha=1/length, adjust=False).mean()
    al    = loss.ewm(alpha=1/length, adjust=False).mean()
    rs    = ag / al.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = df["close"]; h = df["high"]; l = df["low"]
    v  = df["vol"]

    df["ema200"]       = calc_ema(c, 200)
    df["ema20"]        = calc_ema(c, 20)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank"]     = df["atr14"].rolling(100).rank(pct=True) * 100

    bb_mid  = c.rolling(20).mean()
    bb_std  = c.rolling(20).std()
    df["bb_upper"]     = bb_mid + 2 * bb_std
    df["bb_lower"]     = bb_mid - 2 * bb_std
    df["bb_width"]     = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

    df["ema_dist_pct"] = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / df["ema200"].shift(10).replace(0, np.nan)

    vol_ma             = v.rolling(20).mean()
    df["rel_vol"]      = v / vol_ma.replace(0, np.nan)

    df["prev_high"]    = h.shift(1)
    df["prev_low"]     = l.shift(1)
    df["prev_close"]   = c.shift(1)
    df["prev_open"]    = df["open"].shift(1)
    df["ema20_prev"]   = df["ema20"].shift(1)
    df["prev_atr14"]   = df["atr14"].shift(1)
    df["prev_rel_vol"] = df["rel_vol"].shift(1)

    return df

# =============================================================================
# ENVIRONMENT GATE (thresholds from IS, applied to OOS)
# =============================================================================

def learn_env_thresholds(df_is: pd.DataFrame) -> dict:
    """Learn p25/p75/p33 from IS period (no look-ahead)."""
    valid = df_is.dropna(subset=["atr_rank","ema_dist_pct","bb_width"])
    atr_p25       = float(valid["atr_rank"].quantile(0.25))
    ema_dist_p75  = float(valid[valid["ema_dist_pct"] > 0]["ema_dist_pct"].quantile(0.75)
                          if (valid["ema_dist_pct"] > 0).any()
                          else valid["ema_dist_pct"].quantile(0.75))
    bb_width_p33  = float(valid["bb_width"].quantile(0.33))
    return {
        "atr_p25":      atr_p25,
        "ema_dist_p75": ema_dist_p75,
        "bb_p33":       bb_width_p33,
    }

def in_environment(df: pd.DataFrame, thr: dict) -> pd.Series:
    """Boolean mask: all 4 environment conditions met at each bar."""
    return (
        (df["atr_rank"]     < thr["atr_p25"])      &
        (df["ema200_slope"] > 0)                    &
        (df["ema_dist_pct"] > thr["ema_dist_p75"])  &
        (df["bb_width"]     < thr["bb_p33"])
    ).fillna(False)

# =============================================================================
# SIGNALS  (each returns boolean Series; requires env to already be computed)
# =============================================================================

def signal_A_mean_reversion(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """
    Mean Reversion: prev bar's low ≤ EMA20 AND prev bar's close > EMA20.
    (Wick touched or briefly breached EMA20, closed back above = bounce.)
    Only valid when env is True on the signal bar.
    """
    bounce = (
        (df["prev_low"]   <= df["ema20_prev"]) &
        (df["prev_close"] >  df["ema20_prev"]) &
        (df["ema20_prev"].notna())
    )
    return (bounce & env).fillna(False)

def signal_B_momentum(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """
    Momentum Continuation: close > prev bar high (bar closed above prior high).
    """
    breakout = (
        (df["close"] > df["prev_high"]) &
        df["prev_high"].notna()
    )
    return (breakout & env).fillna(False)

def signal_C_relvol_breakout(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """
    Relative Volume Breakout: vol > 1.5× 20-bar average AND bullish candle
    (close > open AND close > prev close).
    """
    vol_spike  = df["rel_vol"] > 1.5
    bullish    = (df["close"] > df["open"]) & (df["close"] > df["prev_close"])
    return (vol_spike & bullish & env).fillna(False)

# =============================================================================
# BACKTEST ENGINE  (ATR-based stops)
# =============================================================================

def run_backtest(df: pd.DataFrame, signal: pd.Series,
                 sym_label: str, fold: int, strat: str) -> list:
    """
    Entry:    open of bar after signal
    Stop:     entry - atr14 of signal bar
    Target:   entry + 2 × atr14 of signal bar  (2R)
    Min SL:   CONFIG["MIN_SL_PCT"]
    Sizing:   1% risk per trade, max 5× leverage
    """
    min_sl    = CONFIG["MIN_SL_PCT"]
    max_lev   = CONFIG["MAX_LEVERAGE"]
    risk_frac = CONFIG["RISK_PER_TRADE_PCT"]
    fee_rate  = CONFIG["TAKER_FEE"]
    spd_rate  = CONFIG["SPREAD"] * 0.5
    slp_rate  = CONFIG["SL_SLIPPAGE"]
    rr        = RR

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
                    "fold":        fold,
                    "strat":       strat,
                    "entry_time":  str(entry_tm),
                    "exit_time":   str(bar["datetime"]),
                    "entry_price": round(entry_px, 6),
                    "exit_price":  round(exit_px, 6),
                    "stop_loss":   round(stop, 6),
                    "take_profit": round(take, 6),
                    "pnl":         round(net, 4),
                    "r_multiple":  round(rmul, 4),
                    "win":         int(exit_type == "TP"),
                    "exit_type":   exit_type,
                    "holding_hrs": i - entry_i,
                })
                in_pos = False
            continue

        if signal.iloc[i - 1]:
            ep      = bar["open"]
            atr_val = prev["atr14"]
            if pd.isna(atr_val) or atr_val <= 0:
                continue

            sl_dist = atr_val           # 1×ATR stop
            if sl_dist / ep < min_sl:
                continue                # skip if stop is unrealistically tight

            stop = ep - sl_dist
            take = ep + rr * sl_dist    # 2×ATR target

            risk_usd = CAPITAL * risk_frac
            sz       = min(risk_usd / sl_dist, (CAPITAL * max_lev) / ep)

            entry_px = ep; pos_size = sz
            entry_tm = bar["datetime"]; entry_i = i
            in_pos   = True

    return trades

# =============================================================================
# STATISTICS
# =============================================================================

def metrics(trades: list, label: str = "") -> dict:
    empty = {"label": label, "n": 0, "wr": 0.0, "pf": 0.0,
             "exp_r": 0.0, "net": 0.0, "sharpe": 0.0, "mdd": 0.0,
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
    exp_r = wr * RR - (1 - wr)
    equity = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak   = np.maximum.accumulate(equity)
    mdd    = float(((equity - peak) / peak).min())
    bpy    = 365 * 24
    ann    = (equity[-1] / CAPITAL) ** (bpy / max(n, 1)) - 1
    vol    = pnl.std() * math.sqrt(bpy) if n > 1 else 1e-9
    sharpe = ann / vol if vol > 0 else 0.0
    return {"label": label, "n": n, "wr": wr, "pf": pf, "exp_r": exp_r,
            "net": float(pnl.sum()), "sharpe": sharpe, "mdd": mdd,
            "equity": equity, "pnls": pnl}

def bootstrap_pf(pnls: np.ndarray, n_iter=N_BOOT, seed=42) -> tuple:
    if len(pnls) < 10:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    pfs = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        pfs.append(wp / max(lp, 1e-9))
    return float(np.percentile(pfs, 5)), float(np.percentile(pfs, 50)), float(np.percentile(pfs, 95))

def monte_carlo(pnls: np.ndarray, n_iter=N_BOOT, seed=42) -> dict:
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

def loo_symbols(sym_trades: dict) -> dict:
    out = {}
    for omit in sym_trades:
        flat = [t for s, tl in sym_trades.items() if s != omit for t in tl]
        m    = metrics(flat)
        out[omit] = {"pf": m["pf"], "n": m["n"]}
    return out

def loo_folds(all_trades: list) -> dict:
    out = {}
    fold_ids = sorted({t["fold"] for t in all_trades})
    for omit in fold_ids:
        flat = [t for t in all_trades if t["fold"] != omit]
        m    = metrics(flat)
        out[omit] = {"pf": m["pf"], "n": m["n"]}
    return out

# =============================================================================
# DATA LOAD
# =============================================================================

print("─"*78)
print("  Loading 1H data …")
all_dfs: dict[str, pd.DataFrame] = {}
skipped_syms = []
for sym in SYMBOLS:
    tag  = sym.replace("-", "_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path):
        skipped_syms.append((sym, "cache missing"))
        continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) < MIN_BARS:
        skipped_syms.append((sym, f"only {len(df)} bars"))
        continue
    all_dfs[sym] = add_features(df)
    print(f"  {sym.split('-')[0]:5s}  bars={len(df):,}  "
          f"{df.datetime.min().date()} → {df.datetime.max().date()}")

if skipped_syms:
    print(f"\n  Skipped: {[s for s,_ in skipped_syms]}")

SYMBOLS = list(all_dfs.keys())
print(f"\n  {len(SYMBOLS)} symbols qualified  "
      f"({sum(len(d) for d in all_dfs.values()):,} total bars)\n")

# =============================================================================
# WALK-FORWARD BACKTEST
# =============================================================================

STRATS  = ["A", "B", "C"]
NAMES   = {"A": "Mean Reversion", "B": "Momentum", "C": "RelVol Breakout"}

# sym_trades[strat][sym] = list of all OOS trades
sym_trades = {s: {sym: [] for sym in SYMBOLS} for s in STRATS}
fold_summaries = []   # (fold_idx, is_end, oos_end, metrics_A, metrics_B, metrics_C)
env_stats = []        # (fold_idx, sym, env_bars_count, env_pct)

print("─"*78)
print(f"  Walk-forward: {len(FOLDS)} folds × {len(SYMBOLS)} symbols × {len(STRATS)} strategies")
print()

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_t = {s: [] for s in STRATS}
    fold_env_count = 0
    fold_env_pct_sum = 0.0
    fold_sym_count   = 0

    for sym, df_full in all_dfs.items():
        N       = len(df_full)
        is_cut  = int(N * is_end)
        oos_cut = int(N * oos_end)

        df_is  = df_full.iloc[:is_cut]
        df_oos = df_full.iloc[is_cut:oos_cut].reset_index(drop=True)

        if len(df_oos) < 100:
            continue

        # Learn IS thresholds
        thr = learn_env_thresholds(df_is)

        # Apply environment gate to OOS
        env = in_environment(df_oos, thr)
        n_env = int(env.sum())
        fold_env_count += n_env
        fold_env_pct_sum += n_env / max(len(df_oos), 1) * 100
        fold_sym_count += 1
        env_stats.append((fold_idx, sym, n_env, n_env/max(len(df_oos),1)*100))

        # Generate signals (only inside environment)
        sig_A = signal_A_mean_reversion(df_oos, env)
        sig_B = signal_B_momentum(df_oos, env)
        sig_C = signal_C_relvol_breakout(df_oos, env)

        for strat, sig in [("A", sig_A), ("B", sig_B), ("C", sig_C)]:
            tl = run_backtest(df_oos, sig, sym, fold_idx, strat)
            sym_trades[strat][sym].extend(tl)
            fold_t[strat].extend(tl)

    env_pct_avg = fold_env_pct_sum / max(fold_sym_count, 1)
    mA = metrics(fold_t["A"]); mB = metrics(fold_t["B"]); mC = metrics(fold_t["C"])

    print(f"  Fold {fold_idx}  (IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%)  "
          f"env_bars={fold_env_count:,} ({env_pct_avg:.1f}%)  "
          f"A: n={mA['n']:4d} PF={mA['pf']:.3f}  "
          f"B: n={mB['n']:4d} PF={mB['pf']:.3f}  "
          f"C: n={mC['n']:4d} PF={mC['pf']:.3f}")
    fold_summaries.append((fold_idx, is_end, oos_end, mA, mB, mC))

print()

# =============================================================================
# PORTFOLIO AGGREGATES
# =============================================================================

all_flat  = {s: [t for sym in SYMBOLS for t in sym_trades[s][sym]] for s in STRATS}
port      = {s: metrics(all_flat[s], f"Strategy {s} — {NAMES[s]}") for s in STRATS}

print("═"*78)
print(f"  PORTFOLIO RESULTS — Walk-Forward OOS (5 folds, {len(SYMBOLS)} symbols)")
print("═"*78)
print(f"\n  {'Strategy':22s}  {'n':>5}  {'WR':>6}  {'PF':>7}  "
      f"{'ExpR':>7}  {'Sharpe':>7}  {'MDD':>7}  {'Net $':>9}")
print("  " + "─"*70)
for s in STRATS:
    m = port[s]
    promote_flag = "★" if m["pf"] > 1.20 and m["n"] >= 100 else " "
    print(f"  {promote_flag} {m['label']:20s}  {m['n']:5d}  {m['wr']*100:5.1f}%  "
          f"{m['pf']:7.3f}  {m['exp_r']:+7.3f}  {m['sharpe']:7.2f}  "
          f"{m['mdd']*100:6.1f}%  {m['net']:+9.0f}")

# Best strategy
best_s = max(STRATS, key=lambda s: port[s]["pf"])
print(f"\n  Best entry: Strategy {best_s} ({NAMES[best_s]})  "
      f"PF={port[best_s]['pf']:.3f}  n={port[best_s]['n']}")

# =============================================================================
# FOLD-BY-FOLD TABLE
# =============================================================================

print(f"\n  Fold-by-Fold PF:")
print(f"  {'Fold':>5}  {'IS%':>5}→{'OOS%':>4}  "
      + "  ".join(f"{'PF_'+s:>8}" for s in STRATS)
      + "  " + "  ".join(f"{'n_'+s:>6}" for s in STRATS))
print("  " + "─"*64)
for fi, is_e, oos_e, mA, mB, mC in fold_summaries:
    pfs = [mA["pf"], mB["pf"], mC["pf"]]
    best_fold = STRATS[pfs.index(max(pfs))]
    print(f"  {fi:5d}  {is_e*100:4.0f}%→{oos_e*100:3.0f}%  "
          + "  ".join(f"{p:8.3f}" for p in pfs)
          + "  " + "  ".join(f"{m['n']:6d}" for m in [mA, mB, mC]))

# =============================================================================
# PER-SYMBOL TABLE
# =============================================================================

print("\n" + "═"*78)
print("  PER-SYMBOL RESULTS (Q4)")
print("═"*78)
print(f"\n  {'Symbol':6s}  " + "  ".join(f"{'PF_'+s:>7}  {'n_'+s:>5}" for s in STRATS))
print("  " + "─"*60)

sym_results = {}
for sym in SYMBOLS:
    tag = sym.split("-")[0]
    row = {"sym": sym}
    for s in STRATS:
        m = metrics(sym_trades[s][sym])
        row[f"pf_{s}"] = m["pf"]
        row[f"n_{s}"]  = m["n"]
        row[f"wr_{s}"] = m["wr"]
    sym_results[sym] = row
    print(f"  {tag:6s}  " + "  ".join(
        f"{row[f'pf_{s}']:7.3f}  {row[f'n_{s}']:5d}" for s in STRATS))

# =============================================================================
# STATISTICAL TESTS — best strategy
# =============================================================================

print("\n" + "═"*78)
print(f"  STATISTICAL TESTS — All strategies")
print("═"*78)

boot_results = {}
mc_results   = {}
binom_results = {}

for s in STRATS:
    pnls = port[s]["pnls"]
    n    = port[s]["n"]
    wr   = port[s]["wr"]
    b5, b50, b95 = bootstrap_pf(pnls)
    mc   = monte_carlo(pnls)
    binom = scipy_stats.binomtest(int(round(wr * n)), n, BEP_WR,
                                  alternative="greater").pvalue if n >= 5 else 1.0
    boot_results[s]  = (b5, b50, b95)
    mc_results[s]    = mc
    binom_results[s] = binom

    print(f"\n  Strategy {s} — {NAMES[s]}")
    print(f"    Bootstrap PF: p5={b5:.3f}  p50={b50:.3f}  p95={b95:.3f}")
    print(f"    MC P(profit): {mc['prob_profit']*100:.1f}%  "
          f"p50=${mc['p50']:,.0f}  p5=${mc['p5']:,.0f}  p95=${mc['p95']:,.0f}")
    print(f"    Binomial WR test: p={binom:.6f}  "
          f"{'SIGNIFICANT ✓' if binom < 0.01 else ('marginal ~' if binom < 0.05 else 'NOT significant ✗')}")

# =============================================================================
# LEAVE-ONE-SYMBOL & LEAVE-ONE-FOLD
# =============================================================================

print("\n" + "─"*78)
print("  Q4 / Q5 — Leave-One-Out Analysis (best strategy by PF)")
print("─"*78)

loo_all = {}
for s in STRATS:
    loo_s  = loo_symbols(sym_trades[s])
    loo_f  = loo_folds(all_flat[s])
    s_vals = [v["pf"] for v in loo_s.values()]
    f_vals = [v["pf"] for v in loo_f.values()]
    loo_all[s] = {
        "sym_floor": min(s_vals) if s_vals else 0.0,
        "sym_ceil":  max(s_vals) if s_vals else 0.0,
        "fold_floor": min(f_vals) if f_vals else 0.0,
        "fold_ceil":  max(f_vals) if f_vals else 0.0,
        "sym_all_pos":  all(v > 1.0 for v in s_vals),
        "fold_all_pos": all(v > 1.0 for v in f_vals),
    }

print(f"\n  {'Strat':>6}  {'LOO-sym floor':>14}  {'LOO-sym >1.0':>13}  "
      f"{'LOO-fold floor':>15}  {'LOO-fold >1.0':>14}")
print("  " + "─"*67)
for s in STRATS:
    l = loo_all[s]
    print(f"  {s:>6}  {l['sym_floor']:14.3f}  "
          f"{'YES ✓' if l['sym_all_pos'] else 'NO ✗':>13}  "
          f"{l['fold_floor']:15.3f}  "
          f"{'YES ✓' if l['fold_all_pos'] else 'NO ✗':>14}")

# Detailed LOO-symbol for best strategy
print(f"\n  LOO-symbol detail for Strategy {best_s} ({NAMES[best_s]}):")
loo_b = loo_symbols(sym_trades[best_s])
print(f"  {'Symbol':7s}  {'LOO PF':>9}  {'n':>5}")
print("  " + "─"*25)
for sym, v in sorted(loo_b.items(), key=lambda x: x[1]["pf"]):
    tag = sym.split("-")[0]
    mark = "✓" if v["pf"] > 1.0 else "✗"
    print(f"  {tag:7s}  {v['pf']:9.3f}  {v['n']:5d}  {mark}")

# =============================================================================
# RESEARCH QUESTIONS
# =============================================================================

print("\n" + "═"*78)
print("  RESEARCH QUESTIONS")
print("═"*78)

any_above_120 = any(port[s]["pf"] > 1.20 for s in STRATS)
best_pf       = max(port[s]["pf"] for s in STRATS)
best_n        = max(port[s]["n"]  for s in STRATS)
syms_above_1  = {s: sum(1 for sym in SYMBOLS
                        if metrics(sym_trades[s][sym])["pf"] > 1.0)
                 for s in STRATS}

print(f"""
  Q1. Does any environment-native strategy achieve PF > 1.20?
      {'YES ✓' if any_above_120 else 'NO ✗'}  Best PF = {best_pf:.3f} (Strategy {best_s})

  Q2. Which entry family performs best?
      A ({NAMES['A']}):   PF={port['A']['pf']:.3f}  n={port['A']['n']}
      B ({NAMES['B']}):   PF={port['B']['pf']:.3f}  n={port['B']['n']}
      C ({NAMES['C']}):   PF={port['C']['pf']:.3f}  n={port['C']['n']}
      Winner: Strategy {best_s} — {NAMES[best_s]}

  Q3. Does the environment provide edge for multiple entries?
      Strategies above PF=1.0: {sum(1 for s in STRATS if port[s]['pf'] > 1.0)}/3
      {'YES — environment is real ✓' if sum(1 for s in STRATS if port[s]['pf'] > 1.0) >= 2
       else 'PARTIAL — only one entry captures the edge'
       if sum(1 for s in STRATS if port[s]['pf'] > 1.0) == 1
       else 'NO — environment edge not captured by any entry ✗'}

  Q4. Edge distributed across symbols or concentrated?
      Strategy {best_s} symbols with PF>1.0: {syms_above_1[best_s]}/{len(SYMBOLS)}
      LOO-sym floor = {loo_all[best_s]['sym_floor']:.3f}
      {'Distributed ✓' if syms_above_1[best_s] >= len(SYMBOLS)//2 else 'Concentrated ✗'}

  Q5. Does any strategy survive every walk-forward fold?
      {'YES ✓' if any(loo_all[s]['fold_all_pos'] for s in STRATS) else 'NO ✗'}
      Strategy {best_s} LOO-fold floor = {loo_all[best_s]['fold_floor']:.3f}
""")

# =============================================================================
# VERDICT
# =============================================================================

print("═"*78)
print("  PROMOTE CRITERIA")
print("═"*78)

verdict_per_strat = {}
for s in STRATS:
    m  = port[s]
    b5, b50, b95 = boot_results[s]
    mc = mc_results[s]
    la = loo_all[s]

    criteria = {
        f"PF > 1.20          (PF={m['pf']:.3f})":      m["pf"] > 1.20,
        f"n ≥ 100            (n={m['n']})":             m["n"]  >= 100,
        f"Boot p50 > 1.20    ({b50:.3f})":              b50 > 1.20,
        f"MC P(profit)>60%   ({mc['prob_profit']*100:.1f}%)": mc["prob_profit"] > 0.60,
        f"LOO-sym floor>1.0  ({la['sym_floor']:.3f})":  la["sym_all_pos"],
        f"LOO-fold floor>1.0 ({la['fold_floor']:.3f})": la["fold_all_pos"],
        f"MDD < 25%          ({abs(m['mdd'])*100:.1f}%)": abs(m["mdd"]) < 0.25,
    }
    n_pass  = sum(criteria.values())
    n_total = len(criteria)

    if all(criteria.values()):
        verdict = "PROMOTE"
    elif m["pf"] > 1.0 and n_pass >= n_total - 2:
        verdict = "WATCHLIST"
    elif n_pass >= 3:
        verdict = "INVESTIGATE"
    else:
        verdict = "REJECT"

    verdict_per_strat[s] = {"verdict": verdict, "n_pass": n_pass,
                             "n_total": n_total, "criteria": criteria}

    vmap   = {"PROMOTE": "\033[92m", "WATCHLIST": "\033[93m",
              "INVESTIGATE": "\033[94m", "REJECT": "\033[91m"}
    vreset = "\033[0m"
    print(f"\n  ── Strategy {s}: {NAMES[s]} ──")
    print(f"  {vmap[verdict]}{verdict}{vreset}  ({n_pass}/{n_total})\n")
    for crit, ok in criteria.items():
        print(f"    {'✓' if ok else '✗'} {crit}")

# Overall verdict
best_verdict = max(STRATS, key=lambda s: verdict_per_strat[s]["n_pass"])
print(f"\n{'═'*78}")
ov = verdict_per_strat[best_verdict]["verdict"]
vmap2 = {"PROMOTE":"92","WATCHLIST":"93","INVESTIGATE":"94","REJECT":"91"}
print(f"  OVERALL BEST: Strategy {best_verdict} ({NAMES[best_verdict]}) — "
      f"\033[{vmap2[ov]}m{ov}\033[0m  "
      f"({verdict_per_strat[best_verdict]['n_pass']}/{verdict_per_strat[best_verdict]['n_total']} criteria)")

# Strategy conclusion
if any(verdict_per_strat[s]["verdict"] == "PROMOTE" for s in STRATS):
    print("""
  ═══ CONCLUSION ══════════════════════════════════════════════════════════
  The environment and entry together form a deployable strategy candidate.
  The environment (Low ATR + Positive EMA200 slope + Extended price +
  Narrow BB) provides genuine out-of-sample edge.
  ═══════════════════════════════════════════════════════════════════════════
""")
elif any_above_120:
    print("""
  ═══ CONCLUSION ══════════════════════════════════════════════════════════
  One or more entries achieve PF > 1.20 but do not meet all PROMOTE bars.
  The environment is real. Entry refinement may close the gap.
  ═══════════════════════════════════════════════════════════════════════════
""")
else:
    print(f"""
  ═══ CONCLUSION ══════════════════════════════════════════════════════════
  Best PF = {best_pf:.3f} — The environment has detectable edge (R035 confirmed),
  but none of the three entry families captured it cleanly OOS.
  The environment is necessary but not sufficient. A more precise entry
  within the environment is required for R037.
  ═══════════════════════════════════════════════════════════════════════════
""")

print("═"*78)

# =============================================================================
# CHARTS
# =============================================================================

print("\n  Generating charts …")

BG    = "#0d1117"
PANEL = "#161b22"
TXT   = "#e0e0e0"
AMBER = "#f0c040"
GREEN = "#2ea043"
RED   = "#cf222e"

def _style(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=TXT, fontsize=8, pad=4)

# ── 1: Portfolio PF comparison ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
_style(ax, "R036 — Portfolio PF: Three Environment-Native Entries")
pf_vals = [port[s]["pf"] for s in STRATS]
n_vals  = [port[s]["n"]  for s in STRATS]
labels  = [f"{s}\n{NAMES[s]}\nn={n}" for s, n in zip(STRATS, n_vals)]
cols    = [STRAT_COLS[s] for s in STRATS]
bars    = ax.bar(labels, pf_vals, color=cols, alpha=0.85, width=0.5)
ax.axhline(1.0,  color="#888",  lw=0.8, ls="--", alpha=0.6)
ax.axhline(1.20, color=AMBER,   lw=1.0, ls=":",  alpha=0.8, label="PF=1.20 PROMOTE bar")
ax.set_ylabel("Profit Factor", color="#888", fontsize=9)
ax.legend(facecolor=PANEL, labelcolor=TXT, fontsize=9)
for b, v in zip(bars, pf_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", color=TXT, fontsize=13, fontweight="bold")
plt.tight_layout()
p = f"{OUT}/r036_pf_comparison.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 2: Fold PF timeline ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)
_style(ax, "R036 — PF by Fold (edge stability across time)")
x = np.arange(len(fold_summaries))
w = 0.28
for i, s in enumerate(STRATS):
    pfs = [f[3+i]["pf"] for f in fold_summaries]
    ax.bar(x + (i-1)*w, pfs, w, color=STRAT_COLS[s], alpha=0.85, label=f"Strat {s}")
ax.axhline(1.0,  color="#888", lw=0.7, ls="--", alpha=0.4)
ax.axhline(1.20, color=AMBER,  lw=0.8, ls=":",  alpha=0.7, label="1.20 target")
xlabels = [f"F{f[0]}\n{int(f[1]*100)}→{int(f[2]*100)}%" for f in fold_summaries]
ax.set_xticks(x); ax.set_xticklabels(xlabels, color=TXT, fontsize=8)
ax.set_ylabel("Profit Factor", color="#888", fontsize=8)
ax.legend(facecolor=PANEL, labelcolor=TXT, fontsize=8)
plt.tight_layout()
p = f"{OUT}/r036_fold_timeline.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 3: Equity curves ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 5), facecolor=BG)
fig.suptitle("R036 — Portfolio Equity Curves (per-symbol, Walk-Forward OOS)", color=TXT, fontsize=11)
for ax_, s in zip(axes, STRATS):
    _style(ax_, f"Strategy {s}: {NAMES[s]}  PF={port[s]['pf']:.3f}  n={port[s]['n']}")
    for sym in SYMBOLS:
        tl = sym_trades[s][sym]
        if not tl: continue
        pnls_ = [t["pnl"] for t in tl]
        eq_   = CAPITAL + np.cumsum(pnls_)
        ax_.plot(eq_, color=COLOURS.get(sym,"#888"), lw=1.0, alpha=0.75,
                 label=sym.split("-")[0])
    ax_.axhline(CAPITAL, color="#444", lw=0.6, ls="--")
    ax_.set_ylabel("Equity ($)", color="#888", fontsize=7)
    ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=5, ncol=3)
plt.tight_layout()
p = f"{OUT}/r036_equity_curves.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 4: Per-symbol PF heatmap ──────────────────────────────────────────────────
sym_tags = [s.split("-")[0] for s in SYMBOLS]
hm_data  = np.array([[sym_results[sym][f"pf_{s}"] for s in STRATS]
                      for sym in SYMBOLS])
fig, ax = plt.subplots(figsize=(8, max(8, len(SYMBOLS)*0.35+2)), facecolor=BG)
_style(ax, "R036 — Per-Symbol PF Heatmap (A / B / C)")
from matplotlib.colors import LinearSegmentedColormap
cmap_rb = LinearSegmentedColormap.from_list("rb", [RED, "#444", GREEN])
im = ax.imshow(hm_data, aspect="auto", cmap=cmap_rb, vmin=0.5, vmax=2.0)
ax.set_xticks([0,1,2])
ax.set_xticklabels([f"Strat {s}\n{NAMES[s]}" for s in STRATS], color=TXT, fontsize=8)
ax.set_yticks(range(len(sym_tags)))
ax.set_yticklabels(sym_tags, color=TXT, fontsize=7)
for i, sym in enumerate(SYMBOLS):
    for j, s in enumerate(STRATS):
        v = hm_data[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white", fontsize=6, fontweight="bold")
plt.colorbar(im, ax=ax, label="Profit Factor")
plt.tight_layout()
p = f"{OUT}/r036_symbol_heatmap.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 5: Bootstrap CI ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
fig.suptitle("R036 — Bootstrap PF Distribution (5,000 iterations)", color=TXT, fontsize=11)
for ax_, s in zip(axes, STRATS):
    b5, b50, b95 = boot_results[s]
    pnls = port[s]["pnls"]
    if len(pnls) < 10:
        continue
    rng = np.random.default_rng(42)
    pfs_boot = []
    for _ in range(N_BOOT):
        samp = rng.choice(pnls, len(pnls), replace=True)
        wp = samp[samp>0].sum(); lp = abs(samp[samp<0].sum())
        pfs_boot.append(wp / max(lp, 1e-9))
    _style(ax_, f"Strategy {s}: {NAMES[s]}\np5={b5:.3f}  p50={b50:.3f}  p95={b95:.3f}")
    ax_.hist(pfs_boot, bins=60, color=STRAT_COLS[s], alpha=0.75, edgecolor="none")
    ax_.axvline(1.0,  color="#888",          lw=0.8, ls="--")
    ax_.axvline(1.20, color=AMBER,           lw=0.8, ls=":")
    ax_.axvline(b50,  color=STRAT_COLS[s],   lw=1.5)
    ax_.set_xlabel("PF", color="#888"); ax_.set_ylabel("Count", color="#888")
plt.tight_layout()
p = f"{OUT}/r036_bootstrap_ci.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 6: Monte Carlo ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
fig.suptitle("R036 — Monte Carlo Final Equity Distribution (5,000 sims)", color=TXT, fontsize=11)
for ax_, s in zip(axes, STRATS):
    mc = mc_results[s]
    _style(ax_, f"Strategy {s}: {NAMES[s]}\nP(profit)={mc['prob_profit']*100:.1f}%")
    ax_.hist(mc["finals"], bins=60, color=STRAT_COLS[s], alpha=0.7, edgecolor="none")
    ax_.axvline(CAPITAL,      color="#888",        lw=1.0, ls="--")
    ax_.axvline(mc["p50"],    color=STRAT_COLS[s], lw=1.5, label=f"p50=${mc['p50']:,.0f}")
    ax_.axvline(mc["p5"],     color=RED,           lw=0.8, ls=":", label=f"p5=${mc['p5']:,.0f}")
    ax_.set_xlabel("Final Equity ($)", color="#888"); ax_.set_ylabel("Count", color="#888")
    ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)
plt.tight_layout()
p = f"{OUT}/r036_monte_carlo.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 7: LOO-symbol bars ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 5), facecolor=BG)
fig.suptitle("R036 — Leave-One-Symbol-Out PF", color=TXT, fontsize=11)
for ax_, s in zip(axes, STRATS):
    _style(ax_, f"Strategy {s}: {NAMES[s]}")
    loo_s  = loo_symbols(sym_trades[s])
    tags   = [sym.split("-")[0] for sym in loo_s]
    pfs_l  = [loo_s[sym]["pf"] for sym in loo_s]
    col_l  = [GREEN if v > 1.2 else (AMBER if v > 1.0 else RED) for v in pfs_l]
    bars_l = ax_.bar(tags, pfs_l, color=col_l, alpha=0.85)
    ax_.axhline(1.0,  color="#888", lw=0.7, ls="--")
    ax_.axhline(1.20, color=AMBER,  lw=0.7, ls=":")
    ax_.set_ylabel("LOO PF", color="#888", fontsize=7)
    ax_.tick_params(axis="x", rotation=45)
    for b, v in zip(bars_l, pfs_l):
        ax_.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}",
                 ha="center", color=TXT, fontsize=6)
plt.tight_layout()
p = f"{OUT}/r036_loo_robustness.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 8: Environment coverage ───────────────────────────────────────────────────
env_df = pd.DataFrame(env_stats, columns=["fold","sym","n_env","env_pct"])
sym_env = env_df.groupby("sym")["env_pct"].mean().sort_values(ascending=False)
sym_tags_env = [s.split("-")[0] for s in sym_env.index]

fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)
fig.suptitle("R036 — Environment Coverage", color=TXT, fontsize=11)

ax_ = axes[0]
_style(ax_, "% Bars in Environment by Symbol (avg across folds)")
ax_.barh(sym_tags_env[::-1], sym_env.values[::-1], color=AMBER, alpha=0.8)
ax_.set_xlabel("% Bars in Environment", color="#888", fontsize=8)

ax_ = axes[1]
_style(ax_, "% Bars in Environment by Fold (avg across symbols)")
fold_env_pct = env_df.groupby("fold")["env_pct"].mean()
ax_.bar([f"Fold {f}" for f in fold_env_pct.index], fold_env_pct.values,
        color=AMBER, alpha=0.8)
ax_.set_ylabel("% in Environment", color="#888", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r036_env_coverage.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 9: Main dashboard ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 18), facecolor=BG)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.33,
                        top=0.93, bottom=0.05, left=0.05, right=0.97)

summary_lines = [
    "R036 ENVIRONMENT-NATIVE",
    "",
    f"Environment: Low ATR + EMA slope>0",
    f"+ Extended above EMA200 + Narrow BB",
    "",
]
for s in STRATS:
    m = port[s]; b5, b50, b95 = boot_results[s]
    v = verdict_per_strat[s]["verdict"]
    summary_lines += [f"Strat {s} ({NAMES[s][:12]}):"]
    summary_lines += [f"  n={m['n']}  PF={m['pf']:.3f}  WR={m['wr']*100:.1f}%"]
    summary_lines += [f"  Boot p50={b50:.3f}  [{v}]", ""]

# PF comparison
ax0 = fig.add_subplot(gs[0, 0])
_style(ax0, "Portfolio PF")
bars_ = ax0.bar([f"Strat {s}" for s in STRATS],
                [port[s]["pf"] for s in STRATS],
                color=[STRAT_COLS[s] for s in STRATS], alpha=0.85)
ax0.axhline(1.2, color=AMBER, lw=0.8, ls=":")
for b, s in zip(bars_, STRATS):
    ax0.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             f"{port[s]['pf']:.3f}", ha="center", color=TXT, fontsize=9, fontweight="bold")
ax0.set_ylabel("PF", color="#888", fontsize=7)

# Fold timeline
ax1 = fig.add_subplot(gs[0, 1:3])
_style(ax1, "PF by Walk-Forward Fold")
x = np.arange(len(fold_summaries))
w = 0.28
for i, s in enumerate(STRATS):
    pfs_ = [f[3+i]["pf"] for f in fold_summaries]
    ax1.bar(x + (i-1)*w, pfs_, w, color=STRAT_COLS[s], alpha=0.85, label=f"Strat {s}")
ax1.axhline(1.2, color=AMBER, lw=0.8, ls=":")
ax1.set_xticks(x)
ax1.set_xticklabels([f"F{f[0]}" for f in fold_summaries], color=TXT, fontsize=8)
ax1.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# Symbol heatmap (compact)
ax2 = fig.add_subplot(gs[0, 3])
_style(ax2, "Symbol PF Heatmap")
im_ = ax2.imshow(hm_data, aspect="auto", cmap=cmap_rb, vmin=0.5, vmax=2.0)
ax2.set_xticks([0,1,2]); ax2.set_xticklabels(["A","B","C"], color=TXT, fontsize=7)
ax2.set_yticks(range(len(sym_tags))); ax2.set_yticklabels(sym_tags, color=TXT, fontsize=6)

# Equity curves best strategy
for col_idx, s in enumerate(STRATS):
    ax3 = fig.add_subplot(gs[1, col_idx])
    _style(ax3, f"Strat {s} Equity  PF={port[s]['pf']:.3f}")
    for sym in SYMBOLS:
        tl = sym_trades[s][sym]
        if not tl: continue
        eq_ = CAPITAL + np.cumsum([t["pnl"] for t in tl])
        ax3.plot(eq_, color=COLOURS.get(sym,"#888"), lw=0.8, alpha=0.7)
    ax3.axhline(CAPITAL, color="#444", lw=0.6, ls="--")
    ax3.set_ylabel("Equity ($)", color="#888", fontsize=7)

# Summary text
ax_txt = fig.add_subplot(gs[1, 3])
ax_txt.set_facecolor(PANEL)
for sp in ax_txt.spines.values(): sp.set_visible(False)
ax_txt.set_xticks([]); ax_txt.set_yticks([])
ax_txt.text(0.05, 0.95, "\n".join(summary_lines),
            transform=ax_txt.transAxes, color=TXT, fontsize=7,
            fontfamily="monospace", va="top")

# Bootstrap histograms
for col_idx, s in enumerate(STRATS):
    ax_b = fig.add_subplot(gs[2, col_idx])
    b5, b50, b95 = boot_results[s]
    mc = mc_results[s]
    pnls = port[s]["pnls"]
    if len(pnls) >= 10:
        rng_ = np.random.default_rng(42)
        pfs_b = []
        for _ in range(N_BOOT):
            samp = rng_.choice(pnls, len(pnls), replace=True)
            wp = samp[samp>0].sum(); lp = abs(samp[samp<0].sum())
            pfs_b.append(wp / max(lp, 1e-9))
        _style(ax_b, f"Strat {s} Bootstrap  p50={b50:.3f}  MC={mc['prob_profit']*100:.0f}%")
        ax_b.hist(pfs_b, bins=50, color=STRAT_COLS[s], alpha=0.75, edgecolor="none")
        ax_b.axvline(1.2, color=AMBER, lw=0.8, ls=":")
        ax_b.axvline(b50, color=STRAT_COLS[s], lw=1.5)
        ax_b.set_xlabel("PF", color="#888", fontsize=7)

# Verdict text
ax_v = fig.add_subplot(gs[2, 3])
ax_v.set_facecolor(PANEL)
for sp in ax_v.spines.values(): sp.set_visible(False)
ax_v.set_xticks([]); ax_v.set_yticks([])
v_text = "VERDICT SUMMARY\n\n"
for s in STRATS:
    vd = verdict_per_strat[s]
    v_text += f"Strat {s}: {vd['verdict']}  ({vd['n_pass']}/{vd['n_total']})\n"
v_text += f"\nBest: Strat {best_verdict}\n"
v_text += f"PF={port[best_verdict]['pf']:.3f}  n={port[best_verdict]['n']}\n"
v_text += f"Boot p50={boot_results[best_verdict][1]:.3f}\n"
v_text += f"MC P={mc_results[best_verdict]['prob_profit']*100:.1f}%"
ax_v.text(0.05, 0.95, v_text, transform=ax_v.transAxes,
          color=TXT, fontsize=9, fontfamily="monospace", va="top",
          bbox=dict(boxstyle="round", facecolor="#0d1117", edgecolor="#444"))

fig.suptitle(
    f"QUANTLAB AI — R036 | Environment-Native Strategy Research\n"
    f"Environment: Low ATR · EMA200 slope>0 · Extended · Narrow BB  |  "
    f"{len(SYMBOLS)} symbols · 5-fold WF · Best={best_s}: PF={port[best_s]['pf']:.3f}",
    color=TXT, fontsize=12, y=0.975
)
dash_path = f"{OUT}/r036_dashboard.png"
fig.savefig(dash_path, dpi=120, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"  → {dash_path}")

# =============================================================================
# TRADE LOG & JOURNAL
# =============================================================================

for s in STRATS:
    path_ = f"{OUT}/r036_strat{s}_trades.csv"
    pd.DataFrame(all_flat[s]).to_csv(path_, index=False)
    print(f"  → {path_}  ({len(all_flat[s])} trades)")

journal_path = CONFIG["JOURNAL_FILE"]
for s in STRATS:
    m   = port[s]
    b5, b50, b95 = boot_results[s]
    mc  = mc_results[s]
    row = {
        "research_id": f"{RESEARCH_ID}-{s}",
        "date":        pd.Timestamp.now().strftime("%Y-%m-%d"),
        "strategy":    f"ENV_NATIVE_STRAT_{s}",
        "timeframe":   "1H",
        "symbols":     ",".join(sym.split("-")[0] for sym in SYMBOLS),
        "method":      "env-gate-walk-forward-5fold",
        "n_oos":       m["n"],
        "wr":          round(m["wr"], 4),
        "pf":          round(m["pf"], 4),
        "sharpe":      round(m["sharpe"], 4),
        "mdd":         round(m["mdd"], 4),
        "net":         round(m["net"], 2),
        "boot_p50":    round(b50, 4),
        "mc_prob":     round(mc["prob_profit"], 4),
        "loo_floor":   round(loo_all[s]["sym_floor"], 4),
        "verdict":     verdict_per_strat[s]["verdict"],
    }
    jdf = pd.read_csv(journal_path) if os.path.exists(journal_path) else pd.DataFrame()
    jdf = pd.concat([jdf, pd.DataFrame([row])], ignore_index=True)
    jdf.to_csv(journal_path, index=False)
print(f"  Journal updated → {journal_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "═"*78)
print(f"  R036 COMPLETE — Environment-Native Strategy Research")
print("═"*78)
print(f"  Environment : ATR_Rank<p25 · EMA200_slope>0 · EMA_dist>p75 · BB_width<p33")
print(f"  Dataset     : {len(SYMBOLS)} symbols · 27 months · 5-fold WF")
print()
print(f"  {'Strategy':30s}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'p50':>7}  {'MC%':>6}  {'Verdict':>12}")
print("  " + "─"*72)
for s in STRATS:
    m   = port[s]
    b5, b50, b95 = boot_results[s]
    mc  = mc_results[s]
    v   = verdict_per_strat[s]["verdict"]
    print(f"  {f'Strat {s}: {NAMES[s]}':30s}  {m['n']:5d}  {m['wr']*100:5.1f}%  "
          f"{m['pf']:7.3f}  {b50:7.3f}  {mc['prob_profit']*100:5.1f}%  {v:>12}")
print()
print(f"  Best: Strategy {best_verdict} ({NAMES[best_verdict]})  "
      f"PF={port[best_verdict]['pf']:.3f}  n={port[best_verdict]['n']}")
print(f"  Overall verdict: {verdict_per_strat[best_verdict]['verdict']}")
print(f"  Output : {OUT}/r036_*")
print("═"*78)
