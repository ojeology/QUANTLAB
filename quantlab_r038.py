"""
=============================================================================
QUANTLAB AI — RESEARCH #038
Entry Family Tournament Inside the Proven Environment
=============================================================================

Background:
  R036: environment PF 1.26–1.82 across three unrelated entries but n too low.
  R037: sensitivity analysis — removed one gate at a time to recover frequency.
        Best variant: BASELINE (all 4 gates, highest score+PF, robust edge).

Objective:
  Determine whether the edge belongs to the environment or to one specific
  entry family.  Test 9 entry families inside ONLY the R037 best environment.

Environment — R037 BASELINE (all 4 gates, IS thresholds):
    • ATR Rank     < IS p25
    • EMA200 slope > 0
    • EMA Distance > IS p75
    • BB Width     < IS p33

Entry Families:
  1  FVG            – Fair Value Gap (3-bar imbalance retracement)
  2  LIQ_SWEEP      – Liquidity Sweep (wick below swing low, close recovers)
  3  BOS             – Break of Structure (close above 5-bar swing high)
  4  EMA_PULLBACK   – EMA20 pullback bounce
  5  DONCHIAN       – Donchian channel breakout (close > 20-bar highest high)
  6  RELVOL         – Relative Volume spike + bullish close (= R036 Strat C)
  7  MOMENTUM       – Momentum close (close > prev bar high = R036 Strat B)
  8  MEAN_REV       – Mean Reversion (EMA20 bounce = R036 Strat A)
  9  ORB             – Opening Range Breakout (00–04 UTC session)

Shared Rules (identical across all strategies):
  Stop  = 1×ATR14    Target = 2×ATR14    (2R fixed)
  Risk  = 1% of capital per trade
  Max leverage = 5×
  Walk-forward = 5-fold expanding (IS thresholds only)
  OOS only — no IS trades counted

PROMOTE: PF>1.20 · n≥100 · Boot_p50>1.20 · MC_P>60%
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
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr

RESEARCH_ID = "R038"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = CONFIG["RISK_REWARD"]
BEP_WR   = 1.0 / (1.0 + RR)

SYMBOLS = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "AVAX-USDT-SWAP","XRP-USDT-SWAP","LTC-USDT-SWAP","BCH-USDT-SWAP",
    "DOGE-USDT-SWAP","ADA-USDT-SWAP","BNB-USDT-SWAP","DOT-USDT-SWAP",
    "ARB-USDT-SWAP","OP-USDT-SWAP","NEAR-USDT-SWAP","ATOM-USDT-SWAP",
    "SUI-USDT-SWAP","APT-USDT-SWAP","WIF-USDT-SWAP","PEPE-USDT-SWAP",
    "ENA-USDT-SWAP","UNI-USDT-SWAP","FIL-USDT-SWAP",
]
MIN_BARS = 4_000
FOLDS    = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
N_BOOT   = 2_000     # sufficient for median/CI estimates; keeps runtime manageable

# R037 BASELINE environment (all 4 gates) — validated best variant
BEST_VID   = "BASELINE"
BEST_GATES = {"atr": True, "slope": True, "dist": True, "bb": True}

# Entry colours
STRAT_COLOURS = {
    "FVG":          "#9b59b6",
    "LIQ_SWEEP":    "#e67e22",
    "BOS":          "#1abc9c",
    "EMA_PULLBACK": "#3498db",
    "DONCHIAN":     "#f39c12",
    "RELVOL":       "#FF9800",
    "MOMENTUM":     "#2196F3",
    "MEAN_REV":     "#4CAF50",
    "ORB":          "#e74c3c",
}
STRAT_NAMES = {
    "FVG":          "Fair Value Gap",
    "LIQ_SWEEP":    "Liquidity Sweep",
    "BOS":          "Break of Structure",
    "EMA_PULLBACK": "EMA Pullback",
    "DONCHIAN":     "Donchian Breakout",
    "RELVOL":       "RelVol Breakout",
    "MOMENTUM":     "Momentum Close",
    "MEAN_REV":     "Mean Reversion",
    "ORB":          "Opening Range Breakout",
}
STRATS = list(STRAT_NAMES.keys())

SYM_COLS = {
    "BTC-USDT-SWAP":"#F7931A","ETH-USDT-SWAP":"#627EEA","SOL-USDT-SWAP":"#9945FF",
    "LINK-USDT-SWAP":"#2A5ADA","AVAX-USDT-SWAP":"#E84142","XRP-USDT-SWAP":"#346AA9",
    "LTC-USDT-SWAP":"#BFBBBB","BCH-USDT-SWAP":"#8DC351","DOGE-USDT-SWAP":"#C3A634",
    "ADA-USDT-SWAP":"#0033AD","BNB-USDT-SWAP":"#F3BA2F","DOT-USDT-SWAP":"#E6007A",
    "ARB-USDT-SWAP":"#28A0F0","OP-USDT-SWAP":"#FF0420","NEAR-USDT-SWAP":"#00C08B",
    "ATOM-USDT-SWAP":"#6F4CFF","SUI-USDT-SWAP":"#6FBCF0","APT-USDT-SWAP":"#00B4D8",
    "WIF-USDT-SWAP":"#A67C52","PEPE-USDT-SWAP":"#4CAF50","ENA-USDT-SWAP":"#8B0000",
    "UNI-USDT-SWAP":"#FF007A","FIL-USDT-SWAP":"#0090FF",
}

BG    = "#0d1117"
PANEL = "#161b22"
TXT   = "#e0e0e0"
AMBER = "#f0c040"
GREEN = "#2ea043"
RED   = "#cf222e"

TARGET_PF   = 1.20
TARGET_N    = 100
TARGET_BOOT = 1.20
TARGET_MC   = 0.60

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #038" + " "*50 + "║")
print("║  Entry Family Tournament — Best R037 Environment" + " "*29 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  R037 best environment: {BEST_VID}
  Gates: ATR<p25 · EMA200_slope>0 · EMA_dist>p75 · BB_width<p33
  9 entry families tested | Stop=1×ATR14 | Target=2×ATR14
  Method: 5-fold expanding WF, IS thresholds only
""")

# =============================================================================
# INDICATORS
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]

    df["ema200"]       = calc_ema(c, 200)
    df["ema20"]        = calc_ema(c, 20)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank"]     = df["atr14"].rolling(100).rank(pct=True) * 100

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"]     = bb_mid + 2 * bb_std
    df["bb_lower"]     = bb_mid - 2 * bb_std
    df["bb_width"]     = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)

    df["ema_dist_pct"] = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / df["ema200"].shift(10).replace(0, np.nan)

    vol_ma             = v.rolling(20).mean()
    df["rel_vol"]      = v / vol_ma.replace(0, np.nan)

    # Donchian 20-bar (excluding current bar → shift)
    df["donch_high"]   = h.shift(1).rolling(20).max()

    # Shifted bars
    df["prev_high"]    = h.shift(1)
    df["prev_low"]     = l.shift(1)
    df["prev_close"]   = c.shift(1)
    df["prev_open"]    = df["open"].shift(1)
    df["prev_atr14"]   = df["atr14"].shift(1)
    df["ema20_prev"]   = df["ema20"].shift(1)
    df["prev2_high"]   = h.shift(2)
    df["prev2_low"]    = l.shift(2)

    # Swing high/low: 5-bar rolling (excluding current)
    df["swing_high_5"] = h.rolling(5).max().shift(1)
    df["swing_low_5"]  = l.rolling(5).min().shift(1)

    # Opening Range (00:00–03:59 UTC high/low per date)
    df["hour_utc"] = df["datetime"].dt.hour
    df["date"]     = df["datetime"].dt.date
    or_mask = df["hour_utc"] < 4
    or_df   = df[or_mask].groupby("date").agg(or_h=("high","max"), or_l=("low","min"))
    or_h_map = or_df["or_h"].to_dict()
    or_l_map = or_df["or_l"].to_dict()
    df["or_high"] = df["date"].map(or_h_map)
    df["or_low"]  = df["date"].map(or_l_map)

    return df

# =============================================================================
# ENVIRONMENT GATE
# =============================================================================

def learn_thresholds(df_is: pd.DataFrame) -> dict:
    valid = df_is.dropna(subset=["atr_rank","ema_dist_pct","bb_width"])
    atr_p25 = float(valid["atr_rank"].quantile(0.25))
    pos_dist = valid[valid["ema_dist_pct"] > 0]["ema_dist_pct"]
    ema_dist_p75 = float(pos_dist.quantile(0.75) if len(pos_dist) > 10
                         else valid["ema_dist_pct"].quantile(0.75))
    bb_p33 = float(valid["bb_width"].quantile(0.33))
    return {"atr_p25": atr_p25, "ema_dist_p75": ema_dist_p75, "bb_p33": bb_p33}

def in_environment(df: pd.DataFrame, thr: dict) -> pd.Series:
    return (
        (df["atr_rank"]     < thr["atr_p25"])      &
        (df["ema200_slope"] > 0)                    &
        (df["ema_dist_pct"] > thr["ema_dist_p75"])  &
        (df["bb_width"]     < thr["bb_p33"])
    ).fillna(False)

# =============================================================================
# SIGNAL FUNCTIONS  (env already applied — fire only inside environment)
# =============================================================================

def sig_fvg(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """
    Bullish FVG: bar[i-2].high < bar[i].low (gap up, price above gap).
    Entry: price retraces into gap — prev bar's low ≤ prev2 high AND
    prev bar's close > prev2 high (bounce off FVG).
    """
    retraced = (
        (df["prev_low"]   <= df["prev2_high"]) &
        (df["prev_close"] >  df["prev2_high"]) &
        df["prev2_high"].notna()
    )
    return (retraced & env).fillna(False)

def sig_liq_sweep(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """
    Liquidity Sweep: prev bar's low < 5-bar swing low AND prev bar's
    close > swing low (wick swept stops, price recovered = bullish rejection).
    """
    sweep = (
        (df["prev_low"]   < df["swing_low_5"]) &
        (df["prev_close"] > df["swing_low_5"]) &
        df["swing_low_5"].notna()
    )
    return (sweep & env).fillna(False)

def sig_bos(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """Break of Structure: close > 5-bar swing high (bullish BOS)."""
    bos = (df["close"] > df["swing_high_5"]) & df["swing_high_5"].notna()
    return (bos & env).fillna(False)

def sig_ema_pullback(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """
    EMA20 Pullback: prev bar's low ≤ EMA20 AND close > EMA20
    (wick touched EMA20, closed back above — identical to R036 Strat A).
    """
    bounce = (
        (df["prev_low"]   <= df["ema20_prev"]) &
        (df["prev_close"] >  df["ema20_prev"]) &
        df["ema20_prev"].notna()
    )
    return (bounce & env).fillna(False)

def sig_donchian(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """Donchian Breakout: close > 20-bar highest high (prior 20 bars)."""
    brk = (df["close"] > df["donch_high"]) & df["donch_high"].notna()
    return (brk & env).fillna(False)

def sig_relvol(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """RelVol Breakout: vol > 1.5× avg AND bullish candle. (R036 Strat C)"""
    return (
        (df["rel_vol"]    > 1.5) &
        (df["close"]      > df["open"]) &
        (df["close"]      > df["prev_close"]) &
        env
    ).fillna(False)

def sig_momentum(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """Momentum Close: close > prev bar high. (R036 Strat B)"""
    return ((df["close"] > df["prev_high"]) & env).fillna(False)

def sig_mean_rev(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """Mean Reversion EMA20 bounce. (R036 Strat A — same as EMA Pullback)"""
    bounce = (
        (df["prev_low"]   <= df["ema20_prev"]) &
        (df["prev_close"] >  df["ema20_prev"]) &
        df["ema20_prev"].notna()
    )
    return (bounce & env).fillna(False)

def sig_orb(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """
    Opening Range Breakout: hour >= 4 UTC, close > OR high (00–03 UTC),
    bullish candle, valid OR exists.
    """
    in_session = df["hour_utc"] >= 4
    above_or   = df["close"] > df["or_high"]
    bullish    = df["close"] > df["open"]
    valid_or   = df["or_high"].notna()
    return (in_session & above_or & bullish & valid_or & env).fillna(False)

SIGNAL_FUNCS = {
    "FVG":          sig_fvg,
    "LIQ_SWEEP":    sig_liq_sweep,
    "BOS":          sig_bos,
    "EMA_PULLBACK": sig_ema_pullback,
    "DONCHIAN":     sig_donchian,
    "RELVOL":       sig_relvol,
    "MOMENTUM":     sig_momentum,
    "MEAN_REV":     sig_mean_rev,
    "ORB":          sig_orb,
}

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, signal: pd.Series,
                 sym: str, fold: int, strat: str) -> list:
    min_sl = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf     = CONFIG["RISK_PER_TRADE_PCT"]; fee = CONFIG["TAKER_FEE"]
    spd    = CONFIG["SPREAD"] * 0.5; slp = CONFIG["SL_SLIPPAGE"]

    in_pos = False
    ep = st = tk = sz = 0.0; et = None; ei = -1
    trades = []

    for i in range(1, len(df)):
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_pos:
            sl_hit = bar["low"]  <= st
            tp_hit = bar["high"] >= tk
            if sl_hit or tp_hit:
                xp   = (st * (1 - slp)) if sl_hit else tk
                xt   = "SL" if sl_hit else "TP"
                sd   = ep - st
                gross = (xp - ep) * sz
                cost  = (ep * sz + xp * sz) * (fee + spd)
                slpc  = (st - xp) * sz if sl_hit else 0.0
                net   = gross - cost - slpc
                rmul  = (xp - ep) / sd if sd > 0 else 0.0
                trades.append({
                    "sym": sym, "fold": fold, "strat": strat,
                    "entry_time": str(et), "exit_time": str(bar["datetime"]),
                    "pnl": round(net, 4), "r_multiple": round(rmul, 4),
                    "win": int(xt == "TP"), "exit_type": xt,
                    "holding_bars": i - ei,
                })
                in_pos = False
            continue

        if signal.iloc[i - 1]:
            atr_ = prev["atr14"]
            if pd.isna(atr_) or atr_ <= 0:
                continue
            sd = atr_
            if sd / bar["open"] < min_sl:
                continue
            ep = bar["open"]; st = ep - sd; tk = ep + RR * sd
            sz = min(CAPITAL * rf / sd, (CAPITAL * max_lev) / ep)
            et = bar["datetime"]; ei = i
            in_pos = True

    return trades

# =============================================================================
# STATISTICS
# =============================================================================

def safe_pf(wins_sum, loss_sum):
    """Cap PF at 10 when losses are zero (prevents inf / statistical artifacts)."""
    if loss_sum <= 0:
        return min(wins_sum / 1e-9, 10.0)
    return wins_sum / loss_sum

def metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "exp_r": 0.0, "net": 0.0,
                "sharpe": 0.0, "mdd": 0.0, "pnls": np.array([]),
                "equity": np.array([CAPITAL])}
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values
    wins = df["win"].values.astype(bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw  = pnl[wins].sum()       if nw else 0.0
    gl  = abs(pnl[~wins].sum()) if nl else 0.0
    pf  = safe_pf(gw, gl)
    wr  = nw / n
    exp_r = wr * RR - (1 - wr)
    equity = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak   = np.maximum.accumulate(equity)
    mdd    = float(((equity - peak) / peak).min())
    bpy    = 365 * 24
    ann    = (equity[-1] / CAPITAL) ** (bpy / max(n, 1)) - 1
    vol    = pnl.std() * math.sqrt(bpy) if n > 1 else 1e-9
    sharpe = ann / vol if vol > 0 else 0.0
    return {"n": n, "wr": wr, "pf": pf, "exp_r": exp_r,
            "net": float(pnl.sum()), "sharpe": sharpe, "mdd": mdd,
            "pnls": pnl, "equity": equity}

def bootstrap_pf(pnls: np.ndarray, n_iter=N_BOOT, seed=42) -> tuple:
    if len(pnls) < 5:
        return 0.0, 0.0, 0.0
    rng  = np.random.default_rng(seed)
    pfs  = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        wp = s[s > 0].sum()
        lp = abs(s[s < 0].sum())
        pfs.append(safe_pf(wp, lp))
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

def loo_sym(sym_trades_dict: dict) -> dict:
    out = {}
    for omit in sym_trades_dict:
        flat = [t for s, tl in sym_trades_dict.items() if s != omit for t in tl]
        out[omit] = metrics(flat)["pf"]
    return out

def loo_fld(all_trades: list) -> dict:
    out = {}
    for omit in sorted({t["fold"] for t in all_trades}):
        flat = [t for t in all_trades if t["fold"] != omit]
        out[omit] = metrics(flat)["pf"]
    return out

# =============================================================================
# DATA LOAD
# =============================================================================

print("─" * 78)
print("  Loading 1H data …")
all_dfs: dict[str, pd.DataFrame] = {}
for sym in SYMBOLS:
    tag  = sym.replace("-", "_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) < MIN_BARS:
        continue
    all_dfs[sym] = add_features(df)

SYMBOLS = list(all_dfs.keys())
print(f"  {len(SYMBOLS)} symbols  ({sum(len(d) for d in all_dfs.values()):,} bars)")

# =============================================================================
# WALK-FORWARD BACKTEST
# =============================================================================

print()
print(f"  Walk-forward: {len(FOLDS)} folds × {len(SYMBOLS)} symbols × "
      f"{len(STRATS)} entry families")
print(f"  Environment: {BEST_VID} (ATR<p25 · slope>0 · dist>p75 · BB<p33)")
print()

sym_trades   = {s: {sym: [] for sym in SYMBOLS} for s in STRATS}
fold_summaries = []

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_t   = {s: [] for s in STRATS}
    env_bars = 0

    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[:int(N * is_end)]
        df_oos = df_full.iloc[int(N * is_end):int(N * oos_end)].reset_index(drop=True)
        if len(df_oos) < 100:
            continue

        thr = learn_thresholds(df_is)
        env = in_environment(df_oos, thr)
        env_bars += int(env.sum())

        for s in STRATS:
            sig = SIGNAL_FUNCS[s](df_oos, env)
            tl  = run_backtest(df_oos, sig, sym, fold_idx, s)
            sym_trades[s][sym].extend(tl)
            fold_t[s].extend(tl)

    ms = {s: metrics(fold_t[s]) for s in STRATS}
    fold_summaries.append((fold_idx, is_end, oos_end, ms, env_bars))

    print(f"  Fold {fold_idx}  (IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%)  "
          f"env_bars={env_bars:,}")
    for s in STRATS:
        m = ms[s]
        print(f"    {s:14s}: n={m['n']:3d}  WR={m['wr']*100:4.0f}%  PF={m['pf']:.3f}")
    print()

# =============================================================================
# AGGREGATE
# =============================================================================

print("─" * 78)
print("  Computing statistics …")

all_flat = {s: [t for sym in SYMBOLS for t in sym_trades[s][sym]] for s in STRATS}
results  = {}

for s in STRATS:
    m          = metrics(all_flat[s])
    b5, b50, b95 = bootstrap_pf(m["pnls"])
    mc         = monte_carlo(m["pnls"])
    ls         = loo_sym(sym_trades[s])
    lf         = loo_fld(all_flat[s])
    sf         = min(ls.values()) if ls else 0.0
    ff         = min(lf.values()) if lf else 0.0

    score = sum([
        m["pf"] > TARGET_PF,
        m["n"]  >= TARGET_N,
        b50     > TARGET_BOOT,
        mc["prob_profit"] > TARGET_MC,
        sf > 1.0,
        ff > 1.0,
        abs(m["mdd"]) < 0.25,
    ])

    results[s] = {
        "n": m["n"], "wr": m["wr"], "pf": m["pf"], "exp_r": m["exp_r"],
        "net": m["net"], "sharpe": m["sharpe"], "mdd": m["mdd"],
        "b5": b5, "b50": b50, "b95": b95,
        "mc_p": mc["prob_profit"], "mc_p5": mc["p5"],
        "mc_p50": mc["p50"], "mc_p95": mc["p95"],
        "sym_floor": sf, "fold_floor": ff,
        "pnls": m["pnls"], "equity": m["equity"],
        "loo_sym": ls, "loo_fld": lf,
        "mc_finals": mc["finals"],
        "score": score,
    }
    print(f"  {s:14s}: n={m['n']:3d}  PF={m['pf']:.3f}  p50={b50:.3f}  "
          f"MC={mc['prob_profit']*100:.1f}%  LOO-S={sf:.3f}  LOO-F={ff:.3f}  "
          f"score={score}/7")

# =============================================================================
# PROMOTE CRITERIA
# =============================================================================

CRITERIA = [
    ("PF > 1.20",           lambda r: r["pf"] > TARGET_PF),
    ("n ≥ 100",             lambda r: r["n"]  >= TARGET_N),
    ("Boot p50 > 1.20",     lambda r: r["b50"] > TARGET_BOOT),
    ("MC P > 60%",          lambda r: r["mc_p"] > TARGET_MC),
    ("LOO-sym floor > 1.0", lambda r: r["sym_floor"] > 1.0),
    ("LOO-fold floor > 1.0",lambda r: r["fold_floor"] > 1.0),
    ("MDD < 25%",           lambda r: abs(r["mdd"]) < 0.25),
]

verdicts = {}
for s in STRATS:
    r  = results[s]
    sc = sum(fn(r) for _, fn in CRITERIA)
    if sc == 7:
        vdct = "PROMOTE"
    elif sc >= 5 and r["pf"] > 1.0:
        vdct = "WATCHLIST"
    elif sc >= 3:
        vdct = "INVESTIGATE"
    else:
        vdct = "REJECT"
    verdicts[s] = {"verdict": vdct, "score": sc}

strats_by_pf = sorted(STRATS, key=lambda x: -results[x]["pf"])
promote_list = [s for s in STRATS if verdicts[s]["verdict"] == "PROMOTE"]
watchlist    = [s for s in STRATS if verdicts[s]["verdict"] == "WATCHLIST"]
profitable   = [s for s in strats_by_pf if results[s]["pf"] > 1.0]
above_120    = [s for s in strats_by_pf if results[s]["pf"] > TARGET_PF]

q1_best = strats_by_pf[0]
q2_best = sorted(STRATS, key=lambda x: -results[x]["score"])[0]
best_candidate = promote_list[0] if promote_list else (
    watchlist[0] if watchlist else strats_by_pf[0])

# =============================================================================
# RESULTS TABLE
# =============================================================================

print("\n" + "═" * 100)
print("  R038 — ENTRY FAMILY TOURNAMENT (Best R037 Environment: BASELINE)")
print("═" * 100)
hdr = (f"  {'':1s}{'Strat':14s}  {'Name':22s}  {'n':>5}  {'WR':>6}  "
       f"{'PF':>7}  {'Exp_R':>6}  {'Sharpe':>7}  {'MDD':>7}  "
       f"{'p50':>7}  {'MC%':>6}  {'LOO-S':>6}  {'LOO-F':>6}  {'Score':>5}  Verdict")
print(hdr)
print("  " + "─" * 115)

for s in strats_by_pf:
    r  = results[s]
    vd = verdicts[s]
    flag = "★" if vd["score"] >= 6 else ("↑" if r["pf"] > TARGET_PF else " ")
    print(f"  {flag}{s:13s}  {STRAT_NAMES[s]:22s}  {r['n']:5d}  "
          f"{r['wr']*100:5.1f}%  {r['pf']:7.3f}  {r['exp_r']:+6.3f}  "
          f"{r['sharpe']:7.2f}  {abs(r['mdd'])*100:6.1f}%  "
          f"{r['b50']:7.3f}  {r['mc_p']*100:5.1f}%  "
          f"{r['sym_floor']:6.3f}  {r['fold_floor']:6.3f}  "
          f"{vd['score']:3d}/7  {vd['verdict']}")

print()
print("  Criteria:  PF>1.20  n≥100  Boot_p50>1.20  MC_P>60%  LOO-S>1.0  LOO-F>1.0  MDD<25%")
print()
print(f"  {'Strat':14s}  " +
      "  ".join(f"{'C'+str(i+1):>6}" for i in range(7)) + "  Score  Verdict")
print("  " + "─" * 75)
for s in strats_by_pf:
    r  = results[s]
    vd = verdicts[s]
    checks = ["✓" if fn(r) else "✗" for _, fn in CRITERIA]
    print(f"  {s:14s}  " +
          "  ".join(f"{c:>6}" for c in checks) +
          f"  {vd['score']:3d}/7  {vd['verdict']}")

# =============================================================================
# RESEARCH QUESTIONS
# =============================================================================

print("\n" + "═" * 100)
print("  RESEARCH QUESTIONS")
print("═" * 100)

q4_text = ("Environment-dominant" if len(above_120) >= len(STRATS) // 2
           else "Mixed — env + entry both matter" if len(above_120) >= 2
           else "Entry-specific" if len(above_120) == 1
           else "Unclear — no entry reaches PF=1.20")

recommend_text = (
    f"PROMOTE → {promote_list[0]} ({STRAT_NAMES[promote_list[0]]})" if promote_list
    else f"WATCHLIST → {watchlist[0]} ({STRAT_NAMES[watchlist[0]]})" if watchlist
    else f"INVESTIGATE → {strats_by_pf[0]} ({STRAT_NAMES[strats_by_pf[0]]})"
)

print(f"""
  Q1. Which entry achieves the highest PF?
      → {q1_best}: {STRAT_NAMES[q1_best]}
        PF={results[q1_best]['pf']:.3f}  WR={results[q1_best]['wr']*100:.1f}%  n={results[q1_best]['n']}
        Boot_p50={results[q1_best]['b50']:.3f}  MC={results[q1_best]['mc_p']*100:.1f}%

  Q2. Which entry is most robust?
      → {q2_best}: {STRAT_NAMES[q2_best]}
        Score={results[q2_best]['score']}/7
        LOO-sym floor={results[q2_best]['sym_floor']:.3f}
        LOO-fold floor={results[q2_best]['fold_floor']:.3f}

  Q3. Do multiple entry families remain profitable (PF>1.0)?
      {len(profitable)}/{len(STRATS)} entries profitable:
      {', '.join(f"{s}({results[s]['pf']:.3f})" for s in profitable) or 'None'}
      {len(above_120)}/{len(STRATS)} entries exceed PF=1.20:
      {', '.join(f"{s}({results[s]['pf']:.3f})" for s in above_120) or 'None'}

  Q4. Is there one dominant entry, or does the environment make almost everything profitable?
      → {q4_text}

  Q5. Which entry should become the production candidate?
      → {recommend_text}
""")

# =============================================================================
# FOLD TABLE
# =============================================================================

print("─" * 100)
print("  Fold-by-Fold PF:")
print(f"  {'Fold':>5}  {'IS':>5}→{'OOS':>4}  " +
      "  ".join(f"{s[:8]:>9}" for s in STRATS))
print("  " + "─" * 85)
for fold_idx, is_end, oos_end, ms, eb in fold_summaries:
    print(f"  {fold_idx:5d}  {is_end*100:4.0f}%→{oos_end*100:3.0f}%  " +
          "  ".join(f"{ms[s]['pf']:9.3f}" for s in STRATS))

# =============================================================================
# CHARTS
# =============================================================================

print("\n  Generating charts …")

def _style(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=TXT, fontsize=8, pad=4)

# ── 1: Rankings ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(26, 10), facecolor=BG)
fig.suptitle(f"R038 — Entry Rankings | Env={BEST_VID} | 9 Entry Families",
             color=TXT, fontsize=12)

metric_panels = [
    ("n",          "Trade Count",      TARGET_N,    False),
    ("pf",         "Profit Factor",    TARGET_PF,   False),
    ("b50",        "Bootstrap p50",    TARGET_BOOT, False),
    ("mc_p",       "MC P(profit)",     TARGET_MC,   True),
    ("wr",         "Win Rate",         BEP_WR,      True),
    ("sym_floor",  "LOO-sym Floor",    1.0,         False),
    ("fold_floor", "LOO-fold Floor",   1.0,         False),
    ("sharpe",     "Sharpe Ratio",     0.5,         False),
]
for ax_, (key, title, tgt, pct) in zip(axes.flat, metric_panels):
    _style(ax_, title)
    vals  = [results[s][key] * (100 if pct else 1) for s in strats_by_pf]
    tgt_  = tgt * (100 if pct else 1)
    cols  = [STRAT_COLOURS[s] for s in strats_by_pf]
    bars_ = ax_.bar(range(len(strats_by_pf)), vals, color=cols, alpha=0.85)
    ax_.axhline(tgt_, color=AMBER, lw=0.9, ls=":", alpha=0.8)
    ax_.set_xticks(range(len(strats_by_pf)))
    ax_.set_xticklabels([s[:7] for s in strats_by_pf], rotation=35, ha="right",
                         color=TXT, fontsize=7)
    for b, val in zip(bars_, vals):
        ax_.text(b.get_x() + b.get_width() / 2, max(val, 0) + abs(tgt_) * 0.01,
                 f"{val:.1f}", ha="center", color=TXT, fontsize=6, fontweight="bold")

plt.tight_layout()
p = f"{OUT}/r038_rankings.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 2: Equity Curves ──────────────────────────────────────────────────────────
n_cols = 3; n_rows = math.ceil(len(STRATS) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 5 * n_rows), facecolor=BG)
fig.suptitle("R038 — OOS Equity Curves by Entry Family (per symbol)", color=TXT, fontsize=11)
axes_flat = list(axes.flat)
for ax_, s in zip(axes_flat, STRATS):
    r = results[s]
    _style(ax_, f"{s}  {STRAT_NAMES[s]}  PF={r['pf']:.3f}  n={r['n']}")
    for sym in SYMBOLS:
        tl = sym_trades[s][sym]
        if not tl:
            continue
        eq_ = CAPITAL + np.cumsum([t["pnl"] for t in tl])
        ax_.plot(eq_, color=SYM_COLS.get(sym, "#888"), lw=0.9, alpha=0.7)
    ax_.axhline(CAPITAL, color="#444", lw=0.5, ls="--")
    ax_.set_ylabel("Equity ($)", color="#888", fontsize=7)
for ax_ in axes_flat[len(STRATS):]:
    ax_.set_visible(False)
plt.tight_layout()
p = f"{OUT}/r038_equity_curves.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 3: Bootstrap CI ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6), facecolor=BG)
_style(ax, f"R038 — Bootstrap PF: Median ± 95% CI  ({N_BOOT:,} iterations)")
x     = np.arange(len(strats_by_pf))
cols_ = [STRAT_COLOURS[s] for s in strats_by_pf]
ax.bar(x, [results[s]["pf"] for s in strats_by_pf],
       color=cols_, alpha=0.4, width=0.55, label="Observed PF")
ax.errorbar(x, [results[s]["b50"] for s in strats_by_pf],
            yerr=[[results[s]["b50"] - results[s]["b5"]  for s in strats_by_pf],
                  [results[s]["b95"] - results[s]["b50"] for s in strats_by_pf]],
            fmt="o", color="white", capsize=5, ms=5, label="Boot p50 ± CI95")
ax.axhline(TARGET_PF, color=AMBER, lw=1.0, ls=":", alpha=0.9, label="1.20 target")
ax.axhline(1.00, color="#555", lw=0.6, ls="--")
ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n{STRAT_NAMES[s][:16]}" for s in strats_by_pf],
                    color=TXT, fontsize=7)
ax.set_ylabel("Profit Factor", color="#888", fontsize=9)
ax.legend(facecolor=PANEL, labelcolor=TXT, fontsize=8)
plt.tight_layout()
p = f"{OUT}/r038_bootstrap.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 4: Monte Carlo ────────────────────────────────────────────────────────────
n_cols_mc = 3; n_rows_mc = math.ceil(len(STRATS) / n_cols_mc)
fig, axes = plt.subplots(n_rows_mc, n_cols_mc,
                          figsize=(21, 5 * n_rows_mc), facecolor=BG)
fig.suptitle(f"R038 — Monte Carlo Final Equity ({N_BOOT:,} simulations)",
             color=TXT, fontsize=11)
axes_mc = list(axes.flat)
for ax_, s in zip(axes_mc, STRATS):
    r = results[s]
    _style(ax_, f"{s}  {STRAT_NAMES[s]}  P(profit)={r['mc_p']*100:.1f}%")
    if len(r["pnls"]) >= 5:
        ax_.hist(r["mc_finals"], bins=60, color=STRAT_COLOURS[s], alpha=0.75, edgecolor="none")
        ax_.axvline(CAPITAL,     color="#888",              lw=0.8, ls="--")
        ax_.axvline(r["mc_p50"], color=STRAT_COLOURS[s],   lw=1.5, label=f"p50=${r['mc_p50']:,.0f}")
        ax_.axvline(r["mc_p5"],  color=RED,                lw=0.8, ls=":", label=f"p5=${r['mc_p5']:,.0f}")
        ax_.set_xlabel("Final Equity ($)", color="#888", fontsize=7)
        ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=6)
for ax_ in axes_mc[len(STRATS):]:
    ax_.set_visible(False)
plt.tight_layout()
p = f"{OUT}/r038_mc.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 5: Strategy Comparison CSV ────────────────────────────────────────────────
csv_rows = []
for i, s in enumerate(strats_by_pf, 1):
    r  = results[s]
    vd = verdicts[s]
    csv_rows.append({
        "rank":           i,
        "strategy":       s,
        "name":           STRAT_NAMES[s],
        "environment":    BEST_VID,
        "n_trades":       r["n"],
        "win_rate":       round(r["wr"], 4),
        "profit_factor":  round(r["pf"], 4),
        "expectancy_r":   round(r["exp_r"], 4),
        "sharpe":         round(r["sharpe"], 4),
        "max_drawdown":   round(r["mdd"], 4),
        "boot_p5":        round(r["b5"], 4),
        "boot_p50":       round(r["b50"], 4),
        "boot_p95":       round(r["b95"], 4),
        "mc_prob_profit": round(r["mc_p"], 4),
        "loo_sym_floor":  round(r["sym_floor"], 4),
        "loo_fold_floor": round(r["fold_floor"], 4),
        "criteria_score": vd["score"],
        "verdict":        vd["verdict"],
    })
csv_path = f"{OUT}/r038_strategy_comparison.csv"
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"  → {csv_path}")

# ── 6: Dashboard ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 22), facecolor=BG)
gs  = gridspec.GridSpec(4, 5, figure=fig, hspace=0.55, wspace=0.35,
                         top=0.94, bottom=0.04, left=0.04, right=0.98)

# Row 0: PF, n, Bootstrap p50, MC%, LOO-sym
row0_panels = [("pf","Profit Factor",TARGET_PF,False),
               ("n","Trade Count",TARGET_N,False),
               ("b50","Bootstrap p50",TARGET_BOOT,False),
               ("mc_p","MC P(profit)",TARGET_MC,True),
               ("sym_floor","LOO-sym Floor",1.0,False)]
for col, (key, title, tgt, pct) in enumerate(row0_panels):
    ax_ = fig.add_subplot(gs[0, col])
    _style(ax_, title)
    vals_ = [results[s][key] * (100 if pct else 1) for s in strats_by_pf]
    tgt_  = tgt * (100 if pct else 1)
    bars__ = ax_.bar(range(len(strats_by_pf)), vals_,
                     color=[STRAT_COLOURS[s] for s in strats_by_pf], alpha=0.85)
    ax_.axhline(tgt_, color=AMBER, lw=0.8, ls=":", alpha=0.8)
    for b, val in zip(bars__, vals_):
        ax_.text(b.get_x() + b.get_width() / 2, max(val, 0) + abs(tgt_) * 0.01,
                 f"{val:.1f}", ha="center", color=TXT, fontsize=6, fontweight="bold")
    ax_.set_xticks(range(len(strats_by_pf)))
    ax_.set_xticklabels([s[:7] for s in strats_by_pf], rotation=35, ha="right",
                         color=TXT, fontsize=6)

# Row 1: Bootstrap CI (full width) + equity of top-2 strategies
ax_bc = fig.add_subplot(gs[1, 0:3])
_style(ax_bc, "Bootstrap PF  Median ± 95% CI")
x_ = np.arange(len(strats_by_pf))
ax_bc.bar(x_, [results[s]["pf"] for s in strats_by_pf],
          color=[STRAT_COLOURS[s] for s in strats_by_pf], alpha=0.35, width=0.5)
ax_bc.errorbar(x_, [results[s]["b50"] for s in strats_by_pf],
               yerr=[[results[s]["b50"] - results[s]["b5"]  for s in strats_by_pf],
                     [results[s]["b95"] - results[s]["b50"] for s in strats_by_pf]],
               fmt="o", color="white", capsize=4, ms=4)
ax_bc.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":")
ax_bc.axhline(1.0, color="#555", lw=0.5, ls="--")
ax_bc.set_xticks(x_)
ax_bc.set_xticklabels([s[:7] for s in strats_by_pf], rotation=30, color=TXT, fontsize=6)
ax_bc.set_ylabel("PF", color="#888", fontsize=8)

for col_off, s in enumerate(strats_by_pf[:2]):
    ax_ = fig.add_subplot(gs[1, 3 + col_off])
    r   = results[s]
    _style(ax_, f"{s}  PF={r['pf']:.3f}  n={r['n']}")
    for sym in SYMBOLS:
        tl = sym_trades[s][sym]
        if not tl:
            continue
        ax_.plot(CAPITAL + np.cumsum([t["pnl"] for t in tl]),
                 color=SYM_COLS.get(sym, "#888"), lw=0.9, alpha=0.7)
    ax_.axhline(CAPITAL, color="#444", lw=0.5, ls="--")
    ax_.set_ylabel("Equity ($)", color="#888", fontsize=7)

# Row 2: Fold stability PF + fold n
ax_fp = fig.add_subplot(gs[2, 0:3])
_style(ax_fp, "PF by Fold — All Entry Families")
for s in strats_by_pf:
    fold_pfs_ = [fold_summaries[fi][3][s]["pf"] for fi in range(len(FOLDS))]
    ax_fp.plot(range(1, 6), fold_pfs_, marker="o", ms=4, lw=1.3,
               color=STRAT_COLOURS[s], label=s[:7], alpha=0.9)
ax_fp.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":")
ax_fp.axhline(1.0, color="#555", lw=0.5, ls="--")
ax_fp.set_xticks(range(1, 6))
ax_fp.set_ylabel("PF", color="#888", fontsize=8)
ax_fp.legend(facecolor=PANEL, labelcolor=TXT, fontsize=6, ncol=3)

ax_fn = fig.add_subplot(gs[2, 3:5])
_style(ax_fn, "Trades by Fold — All Entry Families")
for s in strats_by_pf:
    fold_ns_ = [fold_summaries[fi][3][s]["n"] for fi in range(len(FOLDS))]
    ax_fn.plot(range(1, 6), fold_ns_, marker="o", ms=4, lw=1.3,
               color=STRAT_COLOURS[s], label=s[:7], alpha=0.9)
ax_fn.set_xticks(range(1, 6))
ax_fn.set_ylabel("n trades", color="#888", fontsize=8)
ax_fn.legend(facecolor=PANEL, labelcolor=TXT, fontsize=6, ncol=3)

# Row 3: Ranking table + answers panel
ax_tbl = fig.add_subplot(gs[3, 0:3])
ax_tbl.set_facecolor(PANEL)
for sp in ax_tbl.spines.values():
    sp.set_visible(False)
ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
tbl = [f"{'Rank':>4}  {'Strat':14s}  {'Name':22s}  {'n':>5}  {'PF':>7}  "
       f"{'p50':>7}  {'MC%':>6}  {'LOO-S':>6}  {'LOO-F':>6}  {'Score':>5}  Verdict",
       "─" * 100]
for i, s in enumerate(strats_by_pf, 1):
    r  = results[s]; vd = verdicts[s]
    tbl.append(f"{i:4d}  {s:14s}  {STRAT_NAMES[s]:22s}  {r['n']:5d}  "
               f"{r['pf']:7.3f}  {r['b50']:7.3f}  {r['mc_p']*100:5.1f}%  "
               f"{r['sym_floor']:6.3f}  {r['fold_floor']:6.3f}  "
               f"{vd['score']:3d}/7  {vd['verdict']}")
ax_tbl.text(0.01, 0.98, "\n".join(tbl),
            transform=ax_tbl.transAxes, color=TXT, fontsize=7,
            fontfamily="monospace", va="top")

ax_ans = fig.add_subplot(gs[3, 3:5])
ax_ans.set_facecolor(PANEL)
for sp in ax_ans.spines.values():
    sp.set_visible(False)
ax_ans.set_xticks([]); ax_ans.set_yticks([])
ans_text = (
    f"RESEARCH ANSWERS\n\n"
    f"Env: {BEST_VID} (all 4 gates)\n"
    f"  ATR<p25 · slope>0 · dist>p75 · BB<p33\n\n"
    f"Q1 Highest PF:    {q1_best}\n"
    f"   {STRAT_NAMES[q1_best]}\n"
    f"   PF={results[q1_best]['pf']:.3f}  n={results[q1_best]['n']}\n\n"
    f"Q2 Most Robust:   {q2_best}\n"
    f"   score={results[q2_best]['score']}/7\n\n"
    f"Q3 Profitable:    {len(profitable)}/{len(STRATS)} > PF=1.0\n"
    f"   {len(above_120)}/{len(STRATS)} > PF=1.20\n\n"
    f"Q4 Edge source:   {q4_text}\n\n"
    f"Q5 Candidate:     {best_candidate}\n"
    f"   PF={results[best_candidate]['pf']:.3f}  "
    f"n={results[best_candidate]['n']}\n"
    f"   Boot={results[best_candidate]['b50']:.3f}  "
    f"MC={results[best_candidate]['mc_p']*100:.1f}%\n\n"
    f"VERDICT: {verdicts[best_candidate]['verdict']}"
)
ax_ans.text(0.05, 0.97, ans_text, transform=ax_ans.transAxes,
            color=TXT, fontsize=8, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#0d1117", edgecolor="#444"))

fig.suptitle(
    f"QUANTLAB AI — R038 | Entry Family Tournament | Env={BEST_VID} "
    f"(ATR<p25 · slope>0 · dist>p75 · BB<p33)\n"
    f"9 entries | 5-fold WF | {len(SYMBOLS)} symbols | Stop=1×ATR Target=2×ATR | "
    f"Best: {q1_best} PF={results[q1_best]['pf']:.3f} — {verdicts[q1_best]['verdict']}",
    color=TXT, fontsize=11, y=0.975
)
dash_path = f"{OUT}/r038_dashboard.png"
fig.savefig(dash_path, dpi=120, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"  → {dash_path}")

# =============================================================================
# JOURNAL (MARKDOWN)
# =============================================================================

def ck(cond): return "✓" if cond else "✗"

md = [
    f"# QUANTLAB AI — R038: Entry Family Tournament\n",
    f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
    f"**Environment:** R037 {BEST_VID} — ATR<p25 · EMA200_slope>0 · EMA_dist>p75 · BB_width<p33  ",
    f"**Setup:** Stop=1×ATR14 · Target=2×ATR14 (2R) · Risk=1% · Max_lev=5× · 5-fold WF  ",
    f"**Universe:** {len(SYMBOLS)} symbols · 27mo · 1H data  \n",
    "## Tournament Results\n",
    "| Rank | Strategy | Name | n | WR | PF | Exp_R | Sharpe | MDD | Boot_p50 | MC% | LOO-S | LOO-F | Score | Verdict |",
    "|------|----------|------|---|----|----|-------|--------|-----|----------|-----|-------|-------|-------|---------|",
]
for i, s in enumerate(strats_by_pf, 1):
    r  = results[s]; vd = verdicts[s]
    md.append(
        f"| {i} | {s} | {STRAT_NAMES[s]} | {r['n']} | {r['wr']*100:.1f}% | "
        f"{r['pf']:.3f} | {r['exp_r']:+.3f} | {r['sharpe']:.2f} | "
        f"{abs(r['mdd'])*100:.1f}% | {r['b50']:.3f} | {r['mc_p']*100:.1f}% | "
        f"{r['sym_floor']:.3f} | {r['fold_floor']:.3f} | {vd['score']}/7 | **{vd['verdict']}** |"
    )

md += [
    "\n## Promote Criteria\n",
    "| Strategy | PF>1.20 | n≥100 | Bt_p50>1.20 | MC_P>60% | LOO-S>1 | LOO-F>1 | MDD<25% | Score | Verdict |",
    "|----------|---------|-------|-------------|----------|---------|---------|---------|-------|---------|",
]
for s in strats_by_pf:
    r  = results[s]; vd = verdicts[s]
    md.append(
        f"| {s} | {ck(r['pf']>1.20)} | {ck(r['n']>=100)} | {ck(r['b50']>1.20)} | "
        f"{ck(r['mc_p']>0.60)} | {ck(r['sym_floor']>1.0)} | {ck(r['fold_floor']>1.0)} | "
        f"{ck(abs(r['mdd'])<0.25)} | {vd['score']}/7 | **{vd['verdict']}** |"
    )

md += [
    "\n## Research Questions\n",
    f"**Q1. Highest PF:** {q1_best} — {STRAT_NAMES[q1_best]}  ",
    f"PF={results[q1_best]['pf']:.3f} · WR={results[q1_best]['wr']*100:.1f}% · n={results[q1_best]['n']} · Boot_p50={results[q1_best]['b50']:.3f} · MC={results[q1_best]['mc_p']*100:.1f}%\n",
    f"**Q2. Most Robust:** {q2_best} — {STRAT_NAMES[q2_best]}  ",
    f"Score={results[q2_best]['score']}/7 · LOO-sym={results[q2_best]['sym_floor']:.3f} · LOO-fold={results[q2_best]['fold_floor']:.3f}\n",
    f"**Q3. Multiple profitable entries:**  ",
    f"{len(profitable)}/{len(STRATS)} entries PF>1.0: {', '.join(profitable) or 'None'}  ",
    f"{len(above_120)}/{len(STRATS)} entries PF>1.20: {', '.join(above_120) or 'None'}\n",
    f"**Q4. Edge source:** {q4_text}  ",
    f"{len(above_120)}/{len(STRATS)} entries exceed PF=1.20\n",
    f"**Q5. Production candidate:** {recommend_text}\n",
    "## Verdicts\n",
]
for s in strats_by_pf:
    vd = verdicts[s]
    md.append(f"- **{s}** ({STRAT_NAMES[s]}): **{vd['verdict']}** ({vd['score']}/7)")

md += [
    f"\n## Key Finding\n",
    f"The R037 {BEST_VID} environment (all 4 gates: ATR<p25 · slope>0 · EMA_dist>p75 · BB_width<p33)",
    f"identifies very low trade frequency but high PF when activated.",
    f"",
    f"- Total OOS environment trades across all 9 families combined: "
    f"{sum(results[s]['n'] for s in STRATS)}",
    f"- BASELINE env activates infrequently (sparse qualifying bars) — this is",
    f"  by design: the environment selects only the cleanest setups.",
    f"",
    f"**Overall recommendation:** {recommend_text}",
    f"\n---",
    f"*Generated by QUANTLAB AI R038 — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC*",
]

jmd_path = f"{OUT}/r038_journal.md"
with open(jmd_path, "w") as f:
    f.write("\n".join(md) + "\n")
print(f"  → {jmd_path}")

# =============================================================================
# JOURNAL CSV
# =============================================================================

journal_path = CONFIG["JOURNAL_FILE"]
for s in strats_by_pf:
    r  = results[s]; vd = verdicts[s]
    row = {
        "research_id": f"{RESEARCH_ID}-{s}",
        "date":        pd.Timestamp.now().strftime("%Y-%m-%d"),
        "strategy":    f"ENTRY_TOURNEY_{s}",
        "timeframe":   "1H",
        "symbols":     str(len(SYMBOLS)),
        "method":      f"env-{BEST_VID}-5fold-WF",
        "n_oos":       r["n"],
        "wr":          round(r["wr"], 4),
        "pf":          round(r["pf"], 4),
        "sharpe":      round(r["sharpe"], 4),
        "mdd":         round(r["mdd"], 4),
        "net":         round(r["net"], 2),
        "boot_p50":    round(r["b50"], 4),
        "mc_prob":     round(r["mc_p"], 4),
        "loo_floor":   round(r["sym_floor"], 4),
        "verdict":     vd["verdict"],
    }
    jdf = pd.read_csv(journal_path) if os.path.exists(journal_path) else pd.DataFrame()
    jdf = pd.concat([jdf, pd.DataFrame([row])], ignore_index=True)
    jdf.to_csv(journal_path, index=False)
print(f"  → Journal: {journal_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "═" * 100)
print(f"  R038 COMPLETE — Entry Family Tournament")
print("═" * 100)
print(f"  Environment : {BEST_VID}  (ATR<p25 · slope>0 · dist>p75 · BB<p33)")
print(f"  Dataset     : {len(SYMBOLS)} symbols · 5-fold WF · 2R (stop=1×ATR target=2×ATR)")
print()
print(f"  {'Rank':>4}  {'Strat':14s}  {'Name':22s}  {'n':>5}  {'PF':>7}  "
      f"{'p50':>7}  {'MC%':>6}  {'LOO-S':>6}  {'LOO-F':>6}  {'Score':>5}  Verdict")
print("  " + "─" * 95)
for i, s in enumerate(strats_by_pf, 1):
    r  = results[s]; vd = verdicts[s]
    mark = "★" if vd["verdict"] == "PROMOTE" else ("→" if vd["verdict"] == "WATCHLIST" else " ")
    print(f"  {mark}{i:3d}  {s:14s}  {STRAT_NAMES[s]:22s}  {r['n']:5d}  "
          f"{r['pf']:7.3f}  {r['b50']:7.3f}  {r['mc_p']*100:5.1f}%  "
          f"{r['sym_floor']:6.3f}  {r['fold_floor']:6.3f}  "
          f"{vd['score']:3d}/7  {vd['verdict']}")

print()
print(f"  Profitable (PF>1.0):  {len(profitable)}/{len(STRATS)}: "
      f"{', '.join(s for s in strats_by_pf if results[s]['pf'] > 1.0) or 'None'}")
print(f"  Above PF=1.20:        {len(above_120)}/{len(STRATS)}: "
      f"{', '.join(s for s in strats_by_pf if results[s]['pf'] > TARGET_PF) or 'None'}")
print(f"  PROMOTE:              {len(promote_list)}: "
      f"{', '.join(promote_list) or 'None'}")
print(f"  WATCHLIST:            {len(watchlist)}: "
      f"{', '.join(watchlist) or 'None'}")

print(f"""
  Q1: Highest PF       → {q1_best} ({STRAT_NAMES[q1_best]})  PF={results[q1_best]['pf']:.3f}
  Q2: Most Robust      → {q2_best} ({STRAT_NAMES[q2_best]})  score={results[q2_best]['score']}/7
  Q3: Multiple profit  → {len(profitable)}/{len(STRATS)} profitable  {len(above_120)}/{len(STRATS)} above 1.20
  Q4: Edge source      → {q4_text}
  Q5: Production cand. → {recommend_text}

  Output:
    {OUT}/r038_dashboard.png
    {OUT}/r038_strategy_comparison.csv
    {OUT}/r038_equity_curves.png
    {OUT}/r038_rankings.png
    {OUT}/r038_bootstrap.png
    {OUT}/r038_mc.png
    {OUT}/r038_journal.md
""" + "═" * 100)
