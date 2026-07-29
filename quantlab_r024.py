"""
QUANTLAB AI — RESEARCH #024
Blueprint Validation: Falsifying the QuantLab Edge Blueprint
=============================================================

Objective:
  Validate whether the 5 gates from R023 genuinely improve an existing
  strategy out-of-sample — or whether the blueprint is curve-fitted noise.

Strategy selected: EMA Pullback (fast=20, slow=100)
  Rationale: highest trade count (345), most stable behaviour, largest OOS sample.
  Not chosen for PF — chosen for statistical reliability.

Blueprint Gates (exactly as specified in R023):
  Gate 1: ATR Rank < 50th percentile
  Gate 2: EMA(200) slope positive
  Gate 3: Price within 5% of 20-bar low  (dist_from_low20_pct ≤ 5.0)
  Gate 4: Session = Asia or London
  Gate 5: Bollinger Width in upper tertile OR ATR Rank in lower tertile (< 33rd pct)

Symbols: LINK, ETH, SOL, BTC, LTC, AVAX (6 retained; XRP/DOGE/BCH removed by blueprint)

This is an attempt to FALSIFY the blueprint, not to confirm it.
"""

import os, sys, math, warnings, itertools, random
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

RESEARCH_ID = "R024"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

# Blueprint symbols (XRP, DOGE, BCH removed)
SYMBOLS = [
    "LINK-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
    "BTC-USDT-SWAP",  "LTC-USDT-SWAP", "AVAX-USDT-SWAP",
]
SPLIT   = 0.70
CAPITAL = CONFIG["STARTING_CAPITAL"]

# EMA Pullback params (frozen from R021/R023)
EMA_FAST = 20
EMA_SLOW = 100

COLOURS = {
    "BTC-USDT-SWAP":"#F7931A","ETH-USDT-SWAP":"#627EEA","SOL-USDT-SWAP":"#9945FF",
    "LINK-USDT-SWAP":"#2A5ADA","LTC-USDT-SWAP":"#BFBBBB","AVAX-USDT-SWAP":"#E84142",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def load_1h(sym):
    tag = sym.replace("-","_")
    df  = pd.read_parquet(f"{CACHE}/{tag}_1H.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)

def load_funding(sym):
    tag  = sym.replace("-","_")
    path = f"{CACHE}/{tag}_funding.parquet"
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)

def split_oos(df):
    cut = int(len(df) * SPLIT)
    return df.iloc[cut:].reset_index(drop=True)

def calc_rsi(series, length=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/length, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def session_label(hour):
    if   0  <= hour < 7:   return "Asia"
    elif 7  <= hour < 12:  return "London"
    elif 12 <= hour < 20:  return "NewYork"
    else:                  return "Dead"

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all blueprint features."""
    df = df.copy()
    c  = df["close"]

    df["ema_fast"]   = calc_ema(c, EMA_FAST)
    df["ema_slow"]   = calc_ema(c, EMA_SLOW)
    df["ema200"]     = calc_ema(c, 200)
    df["adx14"]      = calc_adx(df, 14)
    df["atr14"]      = calc_atr(df, 14)

    # ATR Rank: percentile over 100-bar rolling window
    df["atr_rank_pct"] = df["atr14"].rolling(100).rank(pct=True) * 100

    # EMA200 slope (% over 10 bars)
    df["ema200_slope_pct"] = (df["ema200"] - df["ema200"].shift(10)) / df["ema200"].shift(10) * 100

    # 20-bar structure
    df["low20"]  = df["low"].rolling(20).min()
    df["high20"] = df["high"].rolling(20).max()
    df["dist_from_low20_pct"]  = (c - df["low20"])  / df["low20"]  * 100
    df["dist_from_high20_pct"] = (c - df["high20"]) / c * 100

    # Bollinger Width (20, 2σ, as % of price)
    bb_mid   = c.rolling(20).mean()
    bb_std   = c.rolling(20).std(ddof=0)
    df["bb_width"] = (bb_std * 4) / bb_mid * 100

    # Realised volatility
    log_ret  = np.log(c / c.shift(1))
    df["realized_vol"] = log_ret.rolling(20).std() * math.sqrt(24) * 100

    # Session
    df["hour_utc"] = df["datetime"].dt.hour
    df["session"]  = df["hour_utc"].apply(session_label)

    # Prev bar
    df["prev_low"]   = df["low"].shift(1)
    df["prev_high"]  = df["high"].shift(1)
    df["prev_close"] = df["close"].shift(1)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EMA PULLBACK SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def signal_ema_pullback(df: pd.DataFrame) -> pd.Series:
    """EMA Pullback — identical to R021/R023. Frozen."""
    touched_recently = (
        (df["low"] <= df["ema_fast"]) |
        (df["low"].shift(1) <= df["ema_fast"].shift(1)) |
        (df["low"].shift(2) <= df["ema_fast"].shift(2))
    )
    uptrend = df["close"] > df["ema_slow"]
    bounce  = df["close"] > df["ema_fast"]
    trend   = df["adx14"] > 20
    return (uptrend & touched_recently & bounce & trend).fillna(False).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — BLUEPRINT GATES (computed from OOS data thresholds)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gate_thresholds(df_oos: pd.DataFrame) -> dict:
    """
    Compute percentile thresholds from the OOS data itself.
    Gates use fixed rules (50th/33rd pct, 67th pct) — not optimised.
    """
    atr_rank_50   = df_oos["atr_rank_pct"].quantile(0.50)
    atr_rank_33   = df_oos["atr_rank_pct"].quantile(0.33)
    bb_width_67   = df_oos["bb_width"].quantile(0.67)
    return {
        "atr_rank_50": atr_rank_50,
        "atr_rank_33": atr_rank_33,
        "bb_width_67": bb_width_67,
    }

def apply_gates(df_row: pd.Series, thresholds: dict, gates: list) -> bool:
    """
    Return True if all requested gates pass.
    gates: list of gate names to apply (subset of all 5).
    """
    if "gate1_atr_rank" in gates:
        if not (df_row["atr_rank_pct"] < thresholds["atr_rank_50"]):
            return False
    if "gate2_ema_slope" in gates:
        if not (df_row["ema200_slope_pct"] > 0):
            return False
    if "gate3_price_loc" in gates:
        if not (df_row["dist_from_low20_pct"] <= 5.0):
            return False
    if "gate4_session" in gates:
        if df_row["session"] not in ("Asia", "London"):
            return False
    if "gate5_vol_regime" in gates:
        bb_ok  = df_row["bb_width"]     >= thresholds["bb_width_67"]
        atr_ok = df_row["atr_rank_pct"] <  thresholds["atr_rank_33"]
        if not (bb_ok or atr_ok):
            return False
    return True

ALL_GATES = ["gate1_atr_rank","gate2_ema_slope","gate3_price_loc",
             "gate4_session","gate5_vol_regime"]
GATE_LABELS = {
    "gate1_atr_rank":  "Gate 1: ATR Rank < 50th pct",
    "gate2_ema_slope": "Gate 2: EMA200 slope > 0",
    "gate3_price_loc": "Gate 3: Price within 5% of 20-bar low",
    "gate4_session":   "Gate 4: Session = Asia or London",
    "gate5_vol_regime":"Gate 5: BB Width ≥ 67th pct OR ATR Rank < 33rd pct",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — FEATURE-COLLECTING BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest_gated(df: pd.DataFrame, signal: pd.Series,
                       thresholds: dict, active_gates: list,
                       label: str) -> list:
    """
    Full backtest with optional blueprint gates applied at signal bar.
    active_gates=[] → baseline (no filtering).
    """
    min_sl    = CONFIG["MIN_SL_PCT"]
    rr        = CONFIG["RISK_REWARD"]
    max_lev   = CONFIG["MAX_LEVERAGE"]
    capital   = CONFIG["STARTING_CAPITAL"]
    risk_frac = CONFIG["RISK_PER_TRADE_PCT"]
    fee_rate  = CONFIG["TAKER_FEE"]
    spd_rate  = CONFIG["SPREAD"] * 0.5
    slp_rate  = CONFIG["SL_SLIPPAGE"]

    in_pos        = False
    entry_price   = 0.0
    stop_loss     = 0.0
    take_profit   = 0.0
    entry_time    = None
    entry_idx     = -1
    position_size = 0.0
    mfe = mae = 0.0
    trades = []

    for i in range(1, len(df)):
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_pos:
            hi, lo = bar["high"], bar["low"]
            sl_dist = entry_price - stop_loss
            if sl_dist > 0:
                mfe = max(mfe, (hi - entry_price) / sl_dist)
                mae = min(mae, (lo - entry_price) / sl_dist)

            sl_hit = lo <= stop_loss
            tp_hit = hi >= take_profit

            if sl_hit or tp_hit:
                exit_price = (stop_loss * (1.0 - slp_rate)) if sl_hit else take_profit
                exit_type  = "SL" if sl_hit else "TP"

                gross = (exit_price - entry_price) * position_size
                ne, nx = entry_price * position_size, exit_price * position_size
                c_fee  = (ne + nx) * fee_rate
                c_spd  = (ne + nx) * spd_rate
                c_slp  = (stop_loss - exit_price) * position_size if sl_hit else 0.0
                net    = gross - c_fee - c_spd - c_slp
                r_mult = (exit_price - entry_price) / sl_dist if sl_dist > 0 else 0.0

                trades.append({
                    "label": label, "symbol": label,
                    "entry_time":  str(entry_time),
                    "exit_time":   str(bar["datetime"]),
                    "entry_price": entry_price, "exit_price": exit_price,
                    "stop_loss":   stop_loss,   "take_profit": take_profit,
                    "pnl": net,   "r_multiple": r_mult,
                    "win": int(exit_type == "TP"), "exit_type": exit_type,
                    "holding_mins": (i - entry_idx) * 60,
                    "mfe_r": mfe,  "mae_r": mae,
                    # Features at entry
                    "atr_rank_pct":       prev["atr_rank_pct"],
                    "ema200_slope_pct":   prev["ema200_slope_pct"],
                    "dist_from_low20_pct":prev["dist_from_low20_pct"],
                    "session":            prev["session"],
                    "bb_width":           prev["bb_width"],
                    "adx14":              prev["adx14"],
                    "atr14":              prev["atr14"],
                })
                in_pos = False
            continue

        if signal.iloc[i - 1]:
            # Check blueprint gates (pass prev bar features)
            if active_gates and not apply_gates(prev, thresholds, active_gates):
                continue

            ep   = bar["open"]
            sl   = prev["low"]
            sl_d = ep - sl

            if sl_d <= 0 or sl_d / ep < min_sl:
                continue

            tp            = ep + rr * sl_d
            risk_dollars  = capital * risk_frac
            pos_size      = min(risk_dollars / sl_d, (capital * max_lev) / ep)

            entry_price   = ep
            stop_loss     = sl
            take_profit   = tp
            position_size = pos_size
            entry_time    = bar["datetime"]
            entry_idx     = i
            in_pos        = True
            mfe = mae = 0.0

    return trades

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — METRICS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def metrics(trades: list, label: str = "") -> dict:
    if not trades:
        return {"label": label, "n": 0, "wr": 0.0, "pf": 0.0, "exp_r": 0.0,
                "net": 0.0, "sharpe": 0.0, "mdd": 0.0, "avg_hold": 0.0, "equity": np.array([CAPITAL])}
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values
    wins = df["win"].values.astype(bool)
    n, nw = len(pnl), wins.sum()
    nl    = n - nw
    gw    = pnl[wins].sum() if nw else 0.0
    gl    = abs(pnl[~wins].sum()) if nl else 1e-9
    pf    = gw / gl
    wr    = nw / n
    exp_r = wr * 2.0 - (1.0 - wr)
    eq    = CAPITAL + np.cumsum(pnl)
    peak  = np.maximum.accumulate(eq)
    dd    = (eq - peak) / peak
    mdd   = dd.min()
    std   = np.std(pnl, ddof=1) if n > 1 else 1e-9
    bpy   = 365 * 24  # bars/year for 1H
    sharpe= (pnl.mean() / std * math.sqrt(bpy * n / max(n, 1))) if std > 0 else 0.0
    avg_h = df["holding_mins"].mean()
    return {"label": label, "n": n, "wr": wr, "pf": pf, "exp_r": exp_r,
            "net": float(pnl.sum()), "sharpe": sharpe, "mdd": mdd,
            "avg_hold": avg_h, "equity": eq, "pnls": pnl, "wins": wins}

def monte_carlo(pnls: np.ndarray, n_iter: int = 2000, seed: int = 42) -> dict:
    if len(pnls) < 5:
        return {"prob_profit": 0.0, "p5": CAPITAL, "p50": CAPITAL, "p95": CAPITAL, "finals": np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.zeros(n_iter)
    for i in range(n_iter):
        sample  = rng.choice(pnls, len(pnls), replace=True)
        finals[i] = CAPITAL + sample.sum()
    return {
        "prob_profit": (finals > CAPITAL).mean(),
        "p5":   np.percentile(finals, 5),
        "p50":  np.percentile(finals, 50),
        "p95":  np.percentile(finals, 95),
        "finals": finals,
    }

def bootstrap_pf(pnls: np.ndarray, n_iter: int = 2000) -> tuple:
    """Bootstrap 90% CI on PF."""
    if len(pnls) < 10:
        return 0.0, 0.0, 0.0
    rng  = np.random.default_rng(42)
    pfs  = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        pfs.append(wp / lp if lp > 0 else 2.0)
    return float(np.percentile(pfs, 5)), float(np.percentile(pfs, 50)), float(np.percentile(pfs, 95))

def jackknife_pf(trades: list) -> tuple:
    """Jackknife: remove one trade at a time, report mean/std of PF."""
    pnls = np.array([t["pnl"] for t in trades])
    pfs  = []
    for i in range(len(pnls)):
        s  = np.delete(pnls, i)
        wp = s[s>0].sum(); lp = abs(s[s<0].sum())
        pfs.append(wp/lp if lp>0 else 2.0)
    pfs = np.array(pfs)
    return float(pfs.mean()), float(pfs.std())

def loo_pf(sym_trades: dict) -> dict:
    """Leave-one-symbol-out PF."""
    result = {}
    symbols = list(sym_trades.keys())
    for omit in symbols:
        flat = []
        for s, tl in sym_trades.items():
            if s != omit:
                flat.extend(tl)
        if flat:
            m = metrics(flat)
            result[omit] = m["pf"]
    return result

def exec_sensitivity(trades: list, slippage_mult: float) -> dict:
    """Re-price all SL exits with extra slippage multiplier."""
    slp_rate = CONFIG["SL_SLIPPAGE"] * slippage_mult
    new_trades = []
    for t in trades:
        nt = dict(t)
        if t["exit_type"] == "SL":
            orig_sl   = t["stop_loss"]
            orig_ep   = t["entry_price"]
            new_ep_out = orig_sl * (1.0 - slp_rate)
            sl_dist   = orig_ep - orig_sl
            pos_size  = abs(t["pnl"] + (orig_ep - t["exit_price"]) * 0)
            # Re-compute PnL with worse exit
            pos = (t["pnl"] / (t["exit_price"] - orig_ep)) if (t["exit_price"] - orig_ep) != 0 else 0
            # Approximate: adjust pnl by extra slippage
            extra_slp = orig_sl * slp_rate
            nt["pnl"] = t["pnl"] - abs(extra_slp * abs(t["pnl"]) / max(abs(t["pnl"]), 0.01))
        new_trades.append(nt)
    return metrics(new_trades)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — RUN BASELINE + ALL GATE VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #024" + " "*50 + "║")
print("║  Blueprint Validation: Falsifying the QuantLab Edge Blueprint" + " "*17 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Strategy : EMA Pullback (fast={EMA_FAST}/slow={EMA_SLOW})
  Symbols  : {', '.join(s.split('-')[0] for s in SYMBOLS)}
  Blueprint Gates:
    Gate 1: ATR Rank < 50th percentile
    Gate 2: EMA200 slope > 0
    Gate 3: Price within 5% of 20-bar low
    Gate 4: Session = Asia or London
    Gate 5: BB Width ≥ 67th pct OR ATR Rank < 33rd pct
""")

# Load data
print("  Loading OOS 1H data …")
oos_dfs = {}
for sym in SYMBOLS:
    try:
        df      = load_1h(sym)
        df_feat = add_features(df)
        df_oos  = split_oos(df_feat)
        oos_dfs[sym] = df_oos
        print(f"  {sym.split('-')[0]:5s}  OOS bars={len(df_oos):,}")
    except FileNotFoundError:
        print(f"  {sym}: cache missing")

# Compute global thresholds from all OOS data pooled
pool_df    = pd.concat(list(oos_dfs.values()), ignore_index=True)
thresholds = compute_gate_thresholds(pool_df)
print(f"\n  Blueprint thresholds (from pooled OOS data):")
print(f"    ATR Rank 50th pct  : {thresholds['atr_rank_50']:.1f}")
print(f"    ATR Rank 33rd pct  : {thresholds['atr_rank_33']:.1f}")
print(f"    BB Width  67th pct : {thresholds['bb_width_67']:.4f}")

# ── Run Baseline + Blueprint + Individual gates ──────────────────────────────
VARIANTS = [
    ("BASELINE",   []),
    ("BLUEPRINT",  ALL_GATES),
    ("Gate1_only", ["gate1_atr_rank"]),
    ("Gate2_only", ["gate2_ema_slope"]),
    ("Gate3_only", ["gate3_price_loc"]),
    ("Gate4_only", ["gate4_session"]),
    ("Gate5_only", ["gate5_vol_regime"]),
]

variant_sym_trades = {}   # variant → {sym → trades}
variant_all_trades = {}   # variant → flat list
variant_metrics    = {}   # variant → portfolio metrics

for vname, gates in VARIANTS:
    sym_trd = {}
    flat    = []
    for sym in SYMBOLS:
        df_oos = oos_dfs[sym]
        sig    = signal_ema_pullback(df_oos)
        trd    = run_backtest_gated(df_oos, sig, thresholds, gates, sym)
        sym_trd[sym] = trd
        flat.extend(trd)
    variant_sym_trades[vname] = sym_trd
    variant_all_trades[vname] = flat
    variant_metrics[vname]    = metrics(flat, vname)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  RESULTS: BASELINE vs BLUEPRINT (all variants)")
print("═"*78)
print(f"  {'Variant':15s}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'ExpR':>7}  {'Sharpe':>7}"
      f"  {'MDD':>7}  {'Net $':>9}  {'Retain':>7}")
print("  " + "─"*72)
baseline_n = variant_metrics["BASELINE"]["n"]
for vname, _ in VARIANTS:
    m   = variant_metrics[vname]
    ret = f"{m['n']/baseline_n*100:.0f}%" if baseline_n > 0 else "—"
    print(f"  {vname:15s}  {m['n']:5d}  {m['wr']*100:5.1f}%"
          f"  {m['pf']:7.3f}  {m['exp_r']:+7.3f}  {m['sharpe']:7.2f}"
          f"  {m['mdd']*100:6.1f}%  {m['net']:+9.0f}  {ret:>7}")

blueprint_m  = variant_metrics["BLUEPRINT"]
baseline_m   = variant_metrics["BASELINE"]
retention_pct = blueprint_m["n"] / baseline_n * 100 if baseline_n > 0 else 0.0
removed_pct   = 100.0 - retention_pct
pf_improved   = blueprint_m["pf"] > baseline_m["pf"]
exp_improved  = blueprint_m["exp_r"] > baseline_m["exp_r"]

print(f"""
  Trade retention: {blueprint_m['n']}/{baseline_n}  ({retention_pct:.1f}% kept, {removed_pct:.1f}% removed)
  PF improvement:  {baseline_m['pf']:.3f} → {blueprint_m['pf']:.3f}  {'▲ IMPROVED' if pf_improved else '▼ NO IMPROVEMENT'}
  ExpR shift:      {baseline_m['exp_r']:+.3f} → {blueprint_m['exp_r']:+.3f}  {'▲ IMPROVED' if exp_improved else '▼ NO IMPROVEMENT'}
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — CROSS-SYMBOL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("─"*78)
print("  CROSS-SYMBOL ANALYSIS")
print("─"*78)
print(f"  {'Symbol':8s}  {'Base n':>6}  {'Base PF':>8}  {'BP n':>6}  {'BP PF':>8}  {'Δ PF':>7}  {'Retain':>7}  {'Verdict'}")
print("  " + "─"*68)

sym_verdicts = {}
for sym in SYMBOLS:
    base_t = variant_sym_trades["BASELINE"][sym]
    bp_t   = variant_sym_trades["BLUEPRINT"][sym]
    bm     = metrics(base_t)
    bpm    = metrics(bp_t)
    delta  = bpm["pf"] - bm["pf"]
    ret    = bpm["n"] / bm["n"] * 100 if bm["n"] > 0 else 0.0
    improved = "▲" if delta > 0 else "▼"
    profitable = bpm["pf"] > 1.20 and bpm["n"] >= 10
    verdict = "PROMOTE" if (bpm["pf"] > 1.20 and bpm["n"] >= 10 and bpm["exp_r"] > 0) else \
              "WATCHLIST" if (bpm["pf"] >= 1.00 and bpm["exp_r"] >= 0) else "REJECT"
    sym_verdicts[sym] = verdict
    tag = sym.split("-")[0]
    print(f"  {tag:8s}  {bm['n']:6d}  {bm['pf']:8.3f}  {bpm['n']:6d}  {bpm['pf']:8.3f}"
          f"  {delta:+7.3f}  {ret:6.1f}%  {improved} {verdict}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — GATE ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*78)
print("  GATE ATTRIBUTION: Individual contribution of each gate")
print("─"*78)
print(f"  {'Variant':18s}  {'n':>5}  {'PF':>7}  {'ΔPF vs Base':>12}  {'Retain':>7}  {'Net $':>9}")
print("  " + "─"*65)

gate_attrs = {}
for vname in ["BASELINE","Gate1_only","Gate2_only","Gate3_only","Gate4_only","Gate5_only","BLUEPRINT"]:
    m   = variant_metrics[vname]
    delta_pf = m["pf"] - baseline_m["pf"]
    ret  = m["n"] / baseline_n * 100 if baseline_n > 0 else 0.0
    gate_attrs[vname] = {"pf": m["pf"], "delta": delta_pf, "n": m["n"], "retain": ret}
    label_map = {
        "BASELINE":    "BASELINE",
        "Gate1_only":  "Gate1 ATR Rank",
        "Gate2_only":  "Gate2 EMA Slope",
        "Gate3_only":  "Gate3 Price Loc",
        "Gate4_only":  "Gate4 Session",
        "Gate5_only":  "Gate5 Vol Regime",
        "BLUEPRINT":   "ALL GATES",
    }
    lbl = label_map[vname]
    print(f"  {lbl:18s}  {m['n']:5d}  {m['pf']:7.3f}  {delta_pf:+12.3f}  {ret:6.1f}%  {m['net']:+9.0f}")

best_gate   = max(["Gate1_only","Gate2_only","Gate3_only","Gate4_only","Gate5_only"],
                  key=lambda v: gate_attrs[v]["delta"])
weakest_gate= min(["Gate1_only","Gate2_only","Gate3_only","Gate4_only","Gate5_only"],
                  key=lambda v: gate_attrs[v]["delta"])
print(f"\n  Best gate:    {best_gate}  (ΔPF = {gate_attrs[best_gate]['delta']:+.3f})")
print(f"  Weakest gate: {weakest_gate}  (ΔPF = {gate_attrs[weakest_gate]['delta']:+.3f})")
all_combined = blueprint_m["pf"] > max(gate_attrs[g]["pf"] for g in
                ["Gate1_only","Gate2_only","Gate3_only","Gate4_only","Gate5_only"])
print(f"  All gates combined > any single gate: {'YES ✓' if all_combined else 'NO ✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — ROBUSTNESS TESTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─"*78)
print("  ROBUSTNESS TESTS")
print("─"*78)

bp_pnls    = variant_metrics["BLUEPRINT"]["pnls"]
base_pnls  = variant_metrics["BASELINE"]["pnls"]
bp_trades  = variant_all_trades["BLUEPRINT"]
base_trades= variant_all_trades["BASELINE"]

# Monte Carlo
print("\n  Monte Carlo (2,000 iterations):")
mc_base = monte_carlo(base_pnls,  n_iter=2000)
mc_bp   = monte_carlo(bp_pnls,    n_iter=2000)
print(f"  Baseline  : P(profit)={mc_base['prob_profit']*100:.1f}%  p5=${mc_base['p5']:,.0f}  p50=${mc_base['p50']:,.0f}  p95=${mc_base['p95']:,.0f}")
print(f"  Blueprint : P(profit)={mc_bp['prob_profit']*100:.1f}%   p5=${mc_bp['p5']:,.0f}  p50=${mc_bp['p50']:,.0f}  p95=${mc_bp['p95']:,.0f}")

# Bootstrap
print("\n  Bootstrap 90% CI on PF (2,000 resample):")
b5, b50, b95 = bootstrap_pf(base_pnls)
p5, p50, p95 = bootstrap_pf(bp_pnls)
print(f"  Baseline  : [{b5:.3f}, {b50:.3f}, {b95:.3f}]  (p5, p50, p95)")
print(f"  Blueprint : [{p5:.3f}, {p50:.3f}, {p95:.3f}]  (p5, p50, p95)")
bp_ci_beats_1 = p5 > 1.0
print(f"  Blueprint lower CI (p5) > 1.0: {'YES ✓' if bp_ci_beats_1 else 'NO ✗'}")

# Jackknife
print("\n  Jackknife (leave-one-trade-out):")
jk_mean_base, jk_std_base = jackknife_pf(base_trades)  if base_trades else (0.0, 0.0)
jk_mean_bp,   jk_std_bp   = jackknife_pf(bp_trades)    if bp_trades   else (0.0, 0.0)
print(f"  Baseline  : PF_jk = {jk_mean_base:.3f} ± {jk_std_base:.3f}")
print(f"  Blueprint : PF_jk = {jk_mean_bp:.3f} ± {jk_std_bp:.3f}")

# Leave-one-symbol-out
print("\n  Leave-one-symbol-out:")
loo_base = loo_pf(variant_sym_trades["BASELINE"])
loo_bp   = loo_pf(variant_sym_trades["BLUEPRINT"])
for sym in SYMBOLS:
    tag = sym.split("-")[0]
    lb  = loo_base.get(sym, 0.0)
    lbp = loo_bp.get(sym, 0.0)
    print(f"  Leave out {tag:5s}: Base PF={lb:.3f}  BP PF={lbp:.3f}")

# Execution sensitivity
print("\n  Execution sensitivity (extra slippage on SL exits):")
for mult in [1.0, 2.0, 3.0]:
    m_base = exec_sensitivity(base_trades, mult)
    m_bp   = exec_sensitivity(bp_trades,   mult)
    print(f"  Slippage {mult:.0f}×  Base PF={m_base['pf']:.3f}  BP PF={m_bp['pf']:.3f}")

# Statistical test: Blueprint vs Baseline PnL distributions
print("\n  Mann-Whitney test (Blueprint PnL vs Baseline PnL):")
if len(bp_pnls) >= 5 and len(base_pnls) >= 5:
    mw_stat, mw_p = scipy_stats.mannwhitneyu(bp_pnls, base_pnls, alternative="greater")
    print(f"  H0: blueprint PnL distribution ≤ baseline")
    print(f"  p-value = {mw_p:.4f}  → {'REJECT H0 (blueprint significantly better)' if mw_p < 0.05 else 'FAIL TO REJECT H0 (not significant)'}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — FAILURE ANALYSIS (if PF doesn't improve)
# ─────────────────────────────────────────────────────────────────────────────
if not pf_improved:
    print("\n" + "─"*78)
    print("  FAILURE ANALYSIS: which gate rejects the most profitable trades")
    print("─"*78)
    base_df = pd.DataFrame(base_trades)
    for gate in ALL_GATES:
        # Trades that pass signal but fail this gate
        def passes_gate(row, g, thr):
            if g == "gate1_atr_rank": return row.get("atr_rank_pct", 50) < thr["atr_rank_50"]
            if g == "gate2_ema_slope": return row.get("ema200_slope_pct", 0) > 0
            if g == "gate3_price_loc": return row.get("dist_from_low20_pct", 99) <= 5.0
            if g == "gate4_session": return row.get("session","Dead") in ("Asia","London")
            if g == "gate5_vol_regime":
                return (row.get("bb_width",0) >= thr["bb_width_67"] or
                        row.get("atr_rank_pct",99) < thr["atr_rank_33"])
            return True
        fails = [t for t in base_trades if not passes_gate(t, gate, thresholds)]
        wins_rejected  = sum(1 for t in fails if t["win"])
        loses_rejected = sum(1 for t in fails if not t["win"])
        pnl_rejected   = sum(t["pnl"] for t in fails)
        print(f"  {GATE_LABELS[gate][:45]:45s}  rejected={len(fails):4d}"
              f"  wins_lost={wins_rejected:3d}  losses_saved={loses_rejected:3d}"
              f"  net PnL={pnl_rejected:+8.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  FINAL QUESTIONS & VERDICT")
print("═"*78)

# Q1-7 automated answers
bp_pf          = blueprint_m["pf"]
bp_mc_prob     = mc_bp["prob_profit"]
bp_min_trades  = blueprint_m["n"] >= 30
bp_retention   = retention_pct >= 30.0
bp_exp_pos     = blueprint_m["exp_r"] > 0
symbols_improved = sum(1 for s in SYMBOLS
                       if (metrics(variant_sym_trades["BLUEPRINT"][s])["pf"] >
                           metrics(variant_sym_trades["BASELINE"][s])["pf"]))
sym_cross = symbols_improved / len(SYMBOLS)

# Verdict criteria
def get_verdict():
    if (bp_pf > 1.20 and bp_exp_pos and
        bp_mc_prob > 0.60 and bp_min_trades and bp_retention):
        return "PROMOTE"
    elif (1.00 <= bp_pf <= 1.20 and bp_exp_pos):
        return "WATCHLIST"
    elif not bp_retention:
        return "REJECT — insufficient trade retention"
    elif bp_pf < 1.00:
        return "REJECT — PF below 1.0"
    else:
        return "REJECT"

blueprint_verdict = get_verdict()

print(f"""
  Q1. Does the complete blueprint improve Profit Factor?
      Baseline PF={baseline_m['pf']:.3f}  →  Blueprint PF={bp_pf:.3f}
      Answer: {'YES ▲' if pf_improved else 'NO ▼'}

  Q2. Which individual gate contributes most?
      {best_gate}  ΔPF = {gate_attrs[best_gate]['delta']:+.3f}

  Q3. Which gate contributes least?
      {weakest_gate}  ΔPF = {gate_attrs[weakest_gate]['delta']:+.3f}

  Q4. Does combining all gates outperform every individual gate?
      Best single gate PF: {max(gate_attrs[g]['pf'] for g in ['Gate1_only','Gate2_only','Gate3_only','Gate4_only','Gate5_only']):.3f}
      All gates combined:  {bp_pf:.3f}
      Answer: {'YES ▲' if all_combined else 'NO — combining dilutes rather than compounds'}

  Q5. Does blueprint generalise across symbols?
      Improved: {symbols_improved}/{len(SYMBOLS)} symbols
      Answer: {'YES ▲' if sym_cross > 0.5 else 'NO ▼'} ({sym_cross*100:.0f}% of symbols improved)

  Q6. Does trade retention remain above 30%?
      Retention: {retention_pct:.1f}%  ({blueprint_m['n']}/{baseline_n} trades)
      Answer: {'YES ▲' if bp_retention else 'NO ▼'}

  Q7. Is PF greater than 1.20 after all validation tests?
      PF = {bp_pf:.3f}  MC P(profit) = {bp_mc_prob*100:.1f}%  Bootstrap p5 = {p5:.3f}
      Answer: {'YES ▲' if bp_pf > 1.20 else 'NO ▼'}  Bootstrap CI lower: {'> 1.0 ✓' if bp_ci_beats_1 else '< 1.0 ✗'}
""")

print(f"{'═'*78}")
print(f"  BLUEPRINT VERDICT: {blueprint_verdict}")
print(f"{'═'*78}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n  Generating charts …")

def dark_ax(ax, title=None, col="white"):
    ax.set_facecolor("#111")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=col, fontsize=9)

VCOLS = {"BASELINE":"#9E9E9E","BLUEPRINT":"#4CAF50",
         "Gate1_only":"#2196F3","Gate2_only":"#FF9800","Gate3_only":"#9C27B0",
         "Gate4_only":"#F44336","Gate5_only":"#00BCD4"}
vnames_all = [v[0] for v in VARIANTS]

# ── Chart 1: PF & retention bar chart ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#111")
fig.suptitle("R024 — Blueprint Validation: EMA Pullback Strategy", color="white", fontsize=12)

ax = axes[0]
dark_ax(ax, "Profit Factor by Variant")
pfs_  = [variant_metrics[v]["pf"] for v in vnames_all]
cols_ = [VCOLS[v] for v in vnames_all]
bars  = ax.bar(vnames_all, pfs_, color=cols_, alpha=0.85)
ax.axhline(1.0, color="white", lw=1, ls="--", alpha=0.5)
ax.axhline(1.2, color="#FF9800", lw=1, ls=":", alpha=0.6, label="PF=1.20 target")
ax.set_xticklabels(vnames_all, rotation=40, ha="right", fontsize=7)
ax.set_ylabel("Profit Factor", color="white")
for b, p in zip(bars, pfs_):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{p:.3f}",
            ha="center", color="white", fontsize=7)
ax.legend(facecolor="#222", labelcolor="white", fontsize=7)

ax2 = axes[1]
dark_ax(ax2, "Trade Count & Retention")
ns_  = [variant_metrics[v]["n"] for v in vnames_all]
ax2.bar(vnames_all, ns_, color=cols_, alpha=0.85)
ax2.axhline(baseline_n * 0.30, color="#F44336", lw=1.5, ls="--",
            label=f"30% floor = {int(baseline_n*0.3)}")
ax2.set_xticklabels(vnames_all, rotation=40, ha="right", fontsize=7)
ax2.set_ylabel("Trade Count", color="white")
ax2.legend(facecolor="#222", labelcolor="white", fontsize=7)
for b, n in zip(ax2.patches, ns_):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1, str(n),
             ha="center", color="white", fontsize=7)

ax3 = axes[2]
dark_ax(ax3, "Expectancy (R) by Variant")
exps = [variant_metrics[v]["exp_r"] for v in vnames_all]
ax3.bar(vnames_all, exps, color=cols_, alpha=0.85)
ax3.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
ax3.set_xticklabels(vnames_all, rotation=40, ha="right", fontsize=7)
ax3.set_ylabel("Expectancy (R)", color="white")
for b, e in zip(ax3.patches, exps):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
             f"{e:+.3f}", ha="center", color="white", fontsize=7)

plt.tight_layout()
p = f"{OUT}/r024_pf_retention.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 2: Equity curves — Baseline vs Blueprint ────────────────────────────
n_sym = len(SYMBOLS)
ncols = 3; nrows = (n_sym + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows*3.5), facecolor="#111")
fig.suptitle("R024 — Equity Curves: Baseline (grey) vs Blueprint (colour)",
             color="white", fontsize=12)
ax_flat = axes.flatten()
for i, sym in enumerate(SYMBOLS):
    ax  = ax_flat[i]
    col = COLOURS[sym]
    dark_ax(ax, sym.split("-")[0], col)
    eq_b = metrics(variant_sym_trades["BASELINE"][sym])["equity"]
    eq_p = metrics(variant_sym_trades["BLUEPRINT"][sym])["equity"]
    ax.plot(eq_b, color="gray", lw=1.2, ls="--", alpha=0.7, label="Baseline")
    ax.plot(eq_p, color=col,    lw=1.5,            label="Blueprint")
    ax.axhline(CAPITAL, color="white", lw=0.5, ls=":", alpha=0.4)
    bm_ = metrics(variant_sym_trades["BASELINE"][sym])
    pm_ = metrics(variant_sym_trades["BLUEPRINT"][sym])
    ax.text(0.05, 0.95,
            f"Base n={bm_['n']} PF={bm_['pf']:.2f}\nBP  n={pm_['n']} PF={pm_['pf']:.2f}",
            transform=ax.transAxes, color="white", fontsize=7, va="top")
    ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=7, loc="lower right")
for j in range(i+1, len(ax_flat)):
    ax_flat[j].set_visible(False)
plt.tight_layout()
p = f"{OUT}/r024_equity_curves.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 3: Monte Carlo distributions ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#111")
fig.suptitle("R024 — Monte Carlo Final Equity (2,000 iterations)", color="white", fontsize=12)
for ax, (mc_r, label, col) in zip(axes, [(mc_base,"Baseline","#9E9E9E"),(mc_bp,"Blueprint","#4CAF50")]):
    dark_ax(ax, f"{label}  P(profit)={mc_r['prob_profit']*100:.1f}%", col)
    finals = mc_r["finals"]
    lo, hi = np.min(finals), np.max(finals)
    if hi > lo:
        ax.hist(finals, bins=np.linspace(lo, hi, 51), color=col, alpha=0.7, edgecolor="none")
    else:
        ax.axvline(lo, color=col, lw=3)
    for pv, pc, pl in [(5,"#F44336","p5"),(50,"#4CAF50" if label=="Baseline" else "#F7931A","p50"),(95,"#FF9800","p95")]:
        v = np.percentile(finals, pv)
        ax.axvline(v, color=pc, lw=1.5, ls="--", label=f"{pl} ${v:,.0f}")
    ax.axvline(CAPITAL, color="white", lw=1, ls=":", alpha=0.5, label=f"Start ${CAPITAL:,}")
    ax.set_xlabel("Final Equity $", color="white")
    ax.set_ylabel("Count", color="white")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r024_monte_carlo.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 4: Bootstrap CI ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#111")
dark_ax(ax, "R024 — Bootstrap 90% CI on PF: Baseline vs Blueprint")
for xi, (label, lo_, mid_, hi_) in enumerate(
        [("Baseline", b5, b50, b95), ("Blueprint", p5, p50, p95)]):
    col = "#9E9E9E" if xi == 0 else "#4CAF50"
    ax.errorbar(xi, mid_, yerr=[[mid_-lo_],[hi_-mid_]], fmt="o",
                color=col, capsize=12, capthick=2.5, ms=10)
    ax.text(xi, hi_+0.01, f"[{lo_:.2f}, {hi_:.2f}]", ha="center", color=col, fontsize=9)
ax.axhline(1.0, color="white", lw=1, ls="--", alpha=0.5, label="PF=1.0")
ax.axhline(1.2, color="#FF9800", lw=1, ls=":", alpha=0.6, label="PF=1.2")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Baseline","Blueprint"], color="white")
ax.set_ylabel("Profit Factor", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r024_bootstrap_ci.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 5: Cross-symbol PF grid ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5), facecolor="#111")
dark_ax(ax, "R024 — Cross-Symbol PF: Baseline vs Blueprint")
sym_tags = [s.split("-")[0] for s in SYMBOLS]
x_ = np.arange(len(SYMBOLS)); w = 0.35
base_pfs_s = [metrics(variant_sym_trades["BASELINE"][s])["pf"] for s in SYMBOLS]
bp_pfs_s   = [metrics(variant_sym_trades["BLUEPRINT"][s])["pf"] for s in SYMBOLS]
b1 = ax.bar(x_ - w/2, base_pfs_s, w, color="#9E9E9E", alpha=0.8, label="Baseline")
b2 = ax.bar(x_ + w/2, bp_pfs_s,   w,
            color=[COLOURS[s] for s in SYMBOLS], alpha=0.85, label="Blueprint")
ax.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5)
ax.axhline(1.2, color="#FF9800", lw=0.8, ls=":", alpha=0.6)
ax.set_xticks(x_); ax.set_xticklabels(sym_tags, color="white")
ax.set_ylabel("Profit Factor", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
for b, p_ in zip(b2, bp_pfs_s):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            f"{p_:.2f}", ha="center", color="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r024_symbol_comparison.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 6: Gate attribution delta PF ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor="#111")
dark_ax(ax, "R024 — Gate Attribution: ΔPF vs Baseline")
g_labels = ["Gate1\nATR Rank","Gate2\nEMA Slope","Gate3\nPrice Loc",
            "Gate4\nSession","Gate5\nVol Regime","ALL\nGATES"]
g_vnames = ["Gate1_only","Gate2_only","Gate3_only","Gate4_only","Gate5_only","BLUEPRINT"]
g_deltas = [gate_attrs[v]["delta"] for v in g_vnames]
g_cols   = ["#4CAF50" if d > 0 else "#F44336" for d in g_deltas]
g_cols[-1] = "#FF9800"
bars = ax.bar(g_labels, g_deltas, color=g_cols, alpha=0.85)
ax.axhline(0, color="white", lw=1, ls="--", alpha=0.5)
ax.set_ylabel("ΔPF vs Baseline", color="white")
for b, d in zip(bars, g_deltas):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005 if d >= 0 else b.get_height()-0.02,
            f"{d:+.3f}", ha="center", color="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r024_gate_attribution.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 7: Drawdown comparison ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 4), facecolor="#111")
fig.suptitle("R024 — Drawdown: Baseline vs Blueprint", color="white", fontsize=11)
for ax, (label, trades_list, col) in zip(axes, [
    ("Baseline", base_trades, "#9E9E9E"),
    ("Blueprint", bp_trades,  "#4CAF50")]):
    dark_ax(ax, f"{label}  MDD={metrics(trades_list)['mdd']*100:.1f}%", col)
    if trades_list:
        eq  = CAPITAL + np.cumsum([t["pnl"] for t in trades_list])
        dd  = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100
        ax.fill_between(range(len(dd)), dd, 0, color=col, alpha=0.5)
        ax.set_xlabel("Trade #", color="white", fontsize=8)
        ax.set_ylabel("DD %", color="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r024_drawdown.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 8: Full Dashboard ───────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14), facecolor="#0a0a0a")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.45)
fig.suptitle(
    f"QUANTLAB AI — R024 DASHBOARD\n"
    f"Blueprint Validation: EMA Pullback | Verdict: {blueprint_verdict}",
    color="white", fontsize=13, y=0.99)

# Summary table
ax_t = fig.add_subplot(gs[0, :])
ax_t.axis("off")
tbl_rows = []
for vname, _ in VARIANTS:
    m   = variant_metrics[vname]
    ret = f"{m['n']/baseline_n*100:.0f}%"
    tbl_rows.append([
        vname, str(m["n"]), f"{m['wr']*100:.1f}%",
        f"{m['pf']:.3f}", f"{m['exp_r']:+.3f}",
        f"{m['sharpe']:.2f}", f"{m['mdd']*100:.1f}%",
        f"${m['net']:+,.0f}", ret,
    ])
hdrs = ["Variant","n","WR","PF","ExpR","Sharpe","MDD","Net $","Retain"]
tbl  = ax_t.table(cellText=tbl_rows, colLabels=hdrs, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r%2==0 else "#222")
    cell.set_text_props(color="white")
    cell.set_edgecolor("#333")
    if r == 0:
        cell.set_facecolor("#2a2a2a")
        cell.set_text_props(color="#aaa", fontweight="bold")
    if r == 2:  # Blueprint row
        cell.set_facecolor("#0d2a0d")

# PF bar
ax_pf = fig.add_subplot(gs[1, 0])
dark_ax(ax_pf, "PF by Variant")
ax_pf.bar(vnames_all, [variant_metrics[v]["pf"] for v in vnames_all],
          color=[VCOLS[v] for v in vnames_all], alpha=0.85)
ax_pf.axhline(1.0, color="white", lw=0.7, ls="--")
ax_pf.axhline(1.2, color="#FF9800", lw=0.7, ls=":")
ax_pf.set_xticklabels(vnames_all, rotation=45, ha="right", fontsize=6)
ax_pf.set_ylabel("PF", color="white", fontsize=7)

# MC final equity
ax_mc = fig.add_subplot(gs[1, 1])
dark_ax(ax_mc, f"MC Blueprint P(profit)={mc_bp['prob_profit']*100:.1f}%")
fe = mc_bp["finals"]
ax_mc.hist(fe, bins=np.linspace(fe.min(), fe.max(), 31), color="#4CAF50", alpha=0.7)
ax_mc.axvline(CAPITAL, color="white", lw=1, ls=":", alpha=0.5)
ax_mc.axvline(np.percentile(fe, 5), color="#F44336", lw=1.5, ls="--")
ax_mc.axvline(np.percentile(fe, 50), color="#4CAF50", lw=1.5, ls="--")

# Symbol comparison
ax_sym = fig.add_subplot(gs[1, 2:])
dark_ax(ax_sym, "Cross-Symbol PF")
x_ = np.arange(len(SYMBOLS)); w = 0.35
ax_sym.bar(x_ - w/2, [metrics(variant_sym_trades["BASELINE"][s])["pf"] for s in SYMBOLS],
           w, color="#9E9E9E", alpha=0.8, label="Baseline")
ax_sym.bar(x_ + w/2, [metrics(variant_sym_trades["BLUEPRINT"][s])["pf"] for s in SYMBOLS],
           w, color=[COLOURS[s] for s in SYMBOLS], alpha=0.85, label="Blueprint")
ax_sym.axhline(1.0, color="white", lw=0.7, ls="--")
ax_sym.axhline(1.2, color="#FF9800", lw=0.7, ls=":")
ax_sym.set_xticks(x_); ax_sym.set_xticklabels([s.split("-")[0] for s in SYMBOLS], color="white", fontsize=8)
ax_sym.legend(facecolor="#222", labelcolor="white", fontsize=7)

# Gate attribution
ax_gate = fig.add_subplot(gs[2, :2])
dark_ax(ax_gate, "Gate Attribution: ΔPF vs Baseline")
ax_gate.bar(g_labels, g_deltas, color=g_cols, alpha=0.85)
ax_gate.axhline(0, color="white", lw=0.7, ls="--")
ax_gate.set_ylabel("ΔPF", color="white", fontsize=8)
for b, d in zip(ax_gate.patches, g_deltas):
    ax_gate.text(b.get_x()+b.get_width()/2, b.get_height()+0.003 if d >= 0 else b.get_height()-0.015,
                 f"{d:+.3f}", ha="center", color="white", fontsize=8)

# Verdict panel
ax_v = fig.add_subplot(gs[2, 2:])
ax_v.axis("off")
ax_v.set_facecolor("#111")
verdict_col = "#4CAF50" if blueprint_verdict == "PROMOTE" else \
              "#FF9800" if "WATCHLIST" in blueprint_verdict else "#F44336"
ax_v.text(0.5, 0.75, f"VERDICT: {blueprint_verdict}",
          transform=ax_v.transAxes, color=verdict_col, fontsize=16,
          ha="center", va="center", fontweight="bold")
summary_lines = [
    f"PF: {baseline_m['pf']:.3f} → {bp_pf:.3f}  ({'▲' if pf_improved else '▼'})",
    f"Retention: {retention_pct:.0f}%  Trades: {blueprint_m['n']}",
    f"MC P(profit): {mc_bp['prob_profit']*100:.1f}%",
    f"Bootstrap p5: {p5:.3f}  {'> 1.0 ✓' if bp_ci_beats_1 else '< 1.0 ✗'}",
    f"Symbols improved: {symbols_improved}/{len(SYMBOLS)}",
]
for ki, line in enumerate(summary_lines):
    ax_v.text(0.5, 0.50 - ki*0.12, line,
              transform=ax_v.transAxes, color="white", fontsize=10, ha="center")

plt.savefig(f"{OUT}/r024_dashboard.png", dpi=130, bbox_inches="tight",
            facecolor="#0a0a0a"); plt.close()
print(f"  → {OUT}/r024_dashboard.png")

# ── Chart 9: R-multiple distributions ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111")
fig.suptitle("R024 — R-Multiple Distribution: Baseline vs Blueprint", color="white", fontsize=11)
for ax, (label, trades_list, col) in zip(axes, [
        ("Baseline", base_trades, "#9E9E9E"), ("Blueprint", bp_trades, "#4CAF50")]):
    if not trades_list: continue
    dark_ax(ax, f"{label}  (n={len(trades_list)})", col)
    rmuls = [t["r_multiple"] for t in trades_list]
    ax.hist(rmuls, bins=30, color=col, alpha=0.75, edgecolor="none")
    ax.axvline(0, color="white", lw=0.8, ls="--")
    ax.axvline(np.mean(rmuls), color="#FF9800", lw=1.5, ls="--",
               label=f"Mean={np.mean(rmuls):.2f}R")
    ax.set_xlabel("R Multiple", color="white")
    ax.set_ylabel("Count", color="white")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r024_r_distribution.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — SAVE TRADE LOG & JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
if bp_trades:
    df_log = pd.DataFrame(bp_trades)
    log_path = f"{OUT}/r024_trade_log.csv"
    df_log.to_csv(log_path, index=False)
    print(f"  → {log_path}  ({len(df_log)} Blueprint trades)")

# Append to journal
try:
    from quantlab_ai import append_journal
    from datetime import datetime, timezone as _tz
    journal_rows = []
    for vname, _ in VARIANTS:
        m = variant_metrics[vname]
        journal_rows.append({
            "research_id":    RESEARCH_ID,
            "run_date":       datetime.now(tz=_tz.utc).strftime("%Y-%m-%d"),
            "strategy_name":  f"EMA_Pullback_{vname}",
            "symbol":         "PORTFOLIO",
            "n_trades":       m["n"],
            "profit_factor":  round(m["pf"],   4),
            "expectancy_r":   round(m["exp_r"], 4),
            "win_rate":       round(m["wr"],    4),
            "net_profit":     round(m["net"],   2),
            "max_drawdown":   round(m["mdd"],   4),
            "sharpe":         round(m["sharpe"],4),
            "mc_prob_profit": round(mc_bp["prob_profit"] if vname=="BLUEPRINT" else mc_base["prob_profit"], 4),
            "avg_hold_minutes": round(m["avg_hold"],1),
            "verdict":        blueprint_verdict if vname=="BLUEPRINT" else "BASELINE",
        })
    append_journal(journal_rows)
    print(f"  Journal updated → {CONFIG['JOURNAL_FILE']}")
except Exception as e:
    print(f"  [WARN] Journal: {e}")

print(f"\n{'═'*78}")
print(f"  R024 complete. Verdict: {blueprint_verdict}")
print(f"  Output → {OUT}/r024_*")
print(f"{'═'*78}\n")
