"""
=============================================================================
QUANTLAB AI — RESEARCH #037
Environment Sensitivity Analysis
=============================================================================

R036 found PF 1.4–1.8 inside a 4-gate environment, but n=22–49 (too thin).
R037 removes ONE gate at a time to find which gates suppress frequency vs
which gates drive profitability.

Baseline (all 4 gates) — R036 Strategy C:
  1. ATR Rank  < p25
  2. EMA200 slope > 0
  3. EMA Distance > p75
  4. BB Width  < p33

Variant A — remove ATR Rank
Variant B — remove EMA Distance
Variant C — remove EMA200 Slope
Variant D — remove BB Width

Entry: Strategy C (RelVol Breakout) — vol>1.5× 20-bar avg + bullish candle
Exit:  Stop=1×ATR14  Target=2×ATR14  (fixed 2R)
Method: 5-fold expanding walk-forward, IS thresholds only
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

RESEARCH_ID = "R037"
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

FOLDS = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
N_BOOT = 5_000

# Variant definitions: (label, description, gates included)
# gates: atr_rank, ema_slope, ema_dist, bb_width
VARIANTS = [
    ("BASELINE", "All 4 gates",             {"atr":True,  "slope":True,  "dist":True,  "bb":True}),
    ("VAR_A",    "Remove ATR Rank",          {"atr":False, "slope":True,  "dist":True,  "bb":True}),
    ("VAR_B",    "Remove EMA Distance",      {"atr":True,  "slope":True,  "dist":False, "bb":True}),
    ("VAR_C",    "Remove EMA200 Slope",      {"atr":True,  "slope":False, "dist":True,  "bb":True}),
    ("VAR_D",    "Remove BB Width",          {"atr":True,  "slope":True,  "dist":True,  "bb":False}),
]
VAR_IDS   = [v[0] for v in VARIANTS]
VAR_NAMES = {v[0]: v[1] for v in VARIANTS}
VAR_GATES = {v[0]: v[2] for v in VARIANTS}

COLOURS_V = {
    "BASELINE": "#888888",
    "VAR_A":    "#4CAF50",
    "VAR_B":    "#2196F3",
    "VAR_C":    "#FF9800",
    "VAR_D":    "#E91E63",
}

BG    = "#0d1117"
PANEL = "#161b22"
TXT   = "#e0e0e0"
AMBER = "#f0c040"
GREEN = "#2ea043"
RED   = "#cf222e"

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #037" + " "*50 + "║")
print("║  Environment Sensitivity Analysis" + " "*44 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Entry      : Strategy C (RelVol Breakout) — vol>1.5× avg + bullish candle
  Exit       : Stop=1×ATR14  Target=2×ATR14  (2R)
  Method     : 5-fold expanding WF · IS thresholds only
  BASELINE   : ATR<p25 + EMA200_slope>0 + EMA_dist>p75 + BB<p33
  VAR_A      : EMA200_slope>0 + EMA_dist>p75 + BB<p33           (–ATR Rank)
  VAR_B      : ATR<p25 + EMA200_slope>0 + BB<p33                (–EMA Distance)
  VAR_C      : ATR<p25 + EMA_dist>p75 + BB<p33                  (–EMA200 Slope)
  VAR_D      : ATR<p25 + EMA200_slope>0 + EMA_dist>p75          (–BB Width)
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

    vol_ma          = v.rolling(20).mean()
    df["rel_vol"]   = v / vol_ma.replace(0, np.nan)

    df["prev_high"]  = h.shift(1)
    df["prev_low"]   = l.shift(1)
    df["prev_close"] = c.shift(1)
    df["prev_open"]  = df["open"].shift(1)
    df["prev_atr14"] = df["atr14"].shift(1)
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

def make_env(df: pd.DataFrame, thr: dict, gates: dict) -> pd.Series:
    """Build environment mask with exactly the gates specified in `gates`."""
    cond = pd.Series(True, index=df.index)
    if gates["atr"]:
        cond &= (df["atr_rank"] < thr["atr_p25"]).fillna(False)
    if gates["slope"]:
        cond &= (df["ema200_slope"] > 0).fillna(False)
    if gates["dist"]:
        cond &= (df["ema_dist_pct"] > thr["ema_dist_p75"]).fillna(False)
    if gates["bb"]:
        cond &= (df["bb_width"] < thr["bb_p33"]).fillna(False)
    return cond

# =============================================================================
# SIGNAL — Strategy C (RelVol Breakout)
# =============================================================================

def signal_C(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    """vol > 1.5× 20-bar avg AND close > open AND close > prev_close."""
    vol_spike = df["rel_vol"] > 1.5
    bullish   = (df["close"] > df["open"]) & (df["close"] > df["prev_close"])
    return (vol_spike & bullish & env).fillna(False)

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, signal: pd.Series,
                 sym: str, fold: int, variant: str) -> list:
    min_sl   = CONFIG["MIN_SL_PCT"]
    max_lev  = CONFIG["MAX_LEVERAGE"]
    risk_frac= CONFIG["RISK_PER_TRADE_PCT"]
    fee_rate = CONFIG["TAKER_FEE"]
    spd_rate = CONFIG["SPREAD"] * 0.5
    slp_rate = CONFIG["SL_SLIPPAGE"]
    rr       = RR

    in_pos   = False
    entry_px = stop = take = pos_size = 0.0
    entry_tm = None; entry_i = -1
    trades   = []

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
                    "sym":        sym, "fold": fold, "variant": variant,
                    "entry_time": str(entry_tm), "exit_time": str(bar["datetime"]),
                    "pnl":        round(net, 4),
                    "r_multiple": round(rmul, 4),
                    "win":        int(exit_type == "TP"),
                    "exit_type":  exit_type,
                })
                in_pos = False
            continue

        if signal.iloc[i - 1]:
            ep      = bar["open"]
            atr_val = prev["atr14"]
            if pd.isna(atr_val) or atr_val <= 0:
                continue
            sl_dist = atr_val
            if sl_dist / ep < min_sl:
                continue
            stop = ep - sl_dist
            take = ep + rr * sl_dist
            sz   = min(CAPITAL * risk_frac / sl_dist,
                       (CAPITAL * max_lev) / ep)
            entry_px = ep; pos_size = sz
            entry_tm = bar["datetime"]; entry_i = i
            in_pos   = True

    return trades

# =============================================================================
# STATISTICS
# =============================================================================

def metrics(trades: list) -> dict:
    if not trades:
        return {"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,"net":0.0,
                "mdd":0.0,"pnls":np.array([]),"equity":np.array([CAPITAL])}
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values
    wins = df["win"].values.astype(bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw  = pnl[wins].sum()       if nw else 0.0
    gl  = abs(pnl[~wins].sum()) if nl else 1e-9
    pf  = gw / gl
    wr  = nw / n
    equity = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak   = np.maximum.accumulate(equity)
    mdd    = float(((equity - peak) / peak).min())
    exp_r  = wr * RR - (1 - wr)
    return {"n":n,"wr":wr,"pf":pf,"exp_r":exp_r,"net":float(pnl.sum()),
            "mdd":mdd,"pnls":pnl,"equity":equity}

def bootstrap_pf(pnls: np.ndarray, n_iter=N_BOOT, seed=42) -> tuple:
    if len(pnls) < 5: return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    pfs = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        wp = s[s>0].sum(); lp = abs(s[s<0].sum())
        pfs.append(wp / max(lp, 1e-9))
    return float(np.percentile(pfs, 5)), float(np.percentile(pfs, 50)), float(np.percentile(pfs, 95))

def monte_carlo(pnls: np.ndarray, n_iter=N_BOOT, seed=42) -> dict:
    if len(pnls) < 5:
        return {"prob_profit":0.0,"p5":CAPITAL,"p50":CAPITAL,"p95":CAPITAL}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit":float((finals>CAPITAL).mean()),
            "p5":float(np.percentile(finals,5)),
            "p50":float(np.percentile(finals,50)),
            "p95":float(np.percentile(finals,95))}

def loo_sym(sym_trades: dict) -> dict:
    out = {}
    for omit in sym_trades:
        flat = [t for s,tl in sym_trades.items() if s!=omit for t in tl]
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

print("─"*78)
print("  Loading 1H data …")
all_dfs: dict[str, pd.DataFrame] = {}
for sym in SYMBOLS:
    tag  = sym.replace("-","_")
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
# WALK-FORWARD — all variants in one pass
# =============================================================================

# Storage: var_sym_trades[variant][sym] = list of trades
var_sym_trades = {v[0]: {sym: [] for sym in SYMBOLS} for v in VARIANTS}
# env_bar counts per (variant, fold)
env_count_tbl  = {v[0]: [] for v in VARIANTS}   # per fold: total env bars

print(f"\n  Walk-forward: {len(FOLDS)} folds × {len(SYMBOLS)} symbols × {len(VARIANTS)} variants")
print()

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_trades = {v[0]: [] for v in VARIANTS}
    fold_env    = {v[0]: 0  for v in VARIANTS}

    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        is_cut = int(N * is_end)
        oo_cut = int(N * oos_end)
        df_is  = df_full.iloc[:is_cut]
        df_oos = df_full.iloc[is_cut:oo_cut].reset_index(drop=True)
        if len(df_oos) < 100:
            continue

        thr = learn_thresholds(df_is)

        for vid, vname, gates in VARIANTS:
            env = make_env(df_oos, thr, gates)
            fold_env[vid] += int(env.sum())
            sig  = signal_C(df_oos, env)
            tl   = run_backtest(df_oos, sig, sym, fold_idx, vid)
            var_sym_trades[vid][sym].extend(tl)
            fold_trades[vid].extend(tl)

    for vid, _, _ in VARIANTS:
        env_count_tbl[vid].append(fold_env[vid])

    # Fold summary line
    parts = [f"F{fold_idx} (IS={is_end*100:.0f}%→{oos_end*100:.0f}%)"]
    for vid, _, _ in VARIANTS:
        m = metrics(fold_trades[vid])
        parts.append(f"{vid[:6]}:n={m['n']},PF={m['pf']:.2f}")
    print("  " + "  ".join(parts))

print()

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

results = {}
for vid, vname, gates in VARIANTS:
    all_flat = [t for sym in SYMBOLS for t in var_sym_trades[vid][sym]]
    m   = metrics(all_flat)
    b5, b50, b95 = bootstrap_pf(m["pnls"])
    mc  = monte_carlo(m["pnls"])
    ls  = loo_sym(var_sym_trades[vid])
    lf  = loo_fld(all_flat)
    sym_floor  = min(ls.values()) if ls else 0.0
    fold_floor = min(lf.values()) if lf else 0.0
    results[vid] = {
        "name":       vname,
        "gates":      gates,
        "n":          m["n"],
        "wr":         m["wr"],
        "pf":         m["pf"],
        "exp_r":      m["exp_r"],
        "net":        m["net"],
        "mdd":        m["mdd"],
        "b5":         b5,
        "b50":        b50,
        "b95":        b95,
        "mc_p":       mc["prob_profit"],
        "sym_floor":  sym_floor,
        "fold_floor": fold_floor,
        "all_trades": all_flat,
        "pnls":       m["pnls"],
        "env_total":  sum(env_count_tbl[vid]),
        "loo_sym":    ls,
        "loo_fld":    lf,
    }

# =============================================================================
# RESULTS TABLE
# =============================================================================

baseline_n = results["BASELINE"]["n"]

print("═"*78)
print("  SENSITIVITY RESULTS — RelVol Breakout Entry (Strategy C)")
print("═"*78)
header = (f"  {'Variant':10s}  {'Gates':30s}  {'n':>5}  {'ΔTrades':>8}  "
          f"{'WR':>6}  {'PF':>7}  {'p50':>7}  {'MC%':>5}  {'MDD':>7}  "
          f"{'LOO-S':>6}  {'LOO-F':>6}  {'env_bars':>9}")
print(header)
print("  " + "─"*108)

for vid, vname, _ in VARIANTS:
    r   = results[vid]
    dn  = r["n"] - baseline_n
    dn_s = f"+{dn}" if dn >= 0 else str(dn)
    flag  = ""
    if r["pf"] > 1.20 and r["n"] >= 100: flag = "★"
    elif r["pf"] > 1.20:                  flag = "↑"
    print(f"  {flag:1s}{vid:10s}  {vname:30s}  {r['n']:5d}  {dn_s:>8s}  "
          f"{r['wr']*100:5.1f}%  {r['pf']:7.3f}  {r['b50']:7.3f}  "
          f"{r['mc_p']*100:5.1f}%  {abs(r['mdd'])*100:6.1f}%  "
          f"{r['sym_floor']:6.3f}  {r['fold_floor']:6.3f}  {r['env_total']:9,}")

# =============================================================================
# GATE IMPORTANCE ANALYSIS
# =============================================================================

print("\n" + "═"*78)
print("  GATE IMPORTANCE — Removing each gate vs Baseline")
print("═"*78)

# For each removal variant:
# • Frequency lift  = n(variant) - n(BASELINE)
# • PF cost         = pf(BASELINE) - pf(variant)  (positive = gate was helping)
# • Env bar lift    = env_bars(variant) - env_bars(BASELINE)

gate_map = {
    "VAR_A": "ATR Rank < p25",
    "VAR_B": "EMA Distance > p75",
    "VAR_C": "EMA200 Slope > 0",
    "VAR_D": "BB Width < p33",
}
gate_importance = []
for vid in ["VAR_A","VAR_B","VAR_C","VAR_D"]:
    r  = results[vid]
    rb = results["BASELINE"]
    freq_lift = r["n"] - rb["n"]
    pf_cost   = rb["pf"] - r["pf"]
    env_lift  = r["env_total"] - rb["env_total"]
    gate_importance.append({
        "gate":       gate_map[vid],
        "variant":    vid,
        "freq_lift":  freq_lift,
        "pf_cost":    pf_cost,
        "env_lift":   env_lift,
        "pf_after":   r["pf"],
        "n_after":    r["n"],
        "n_pct_gain": freq_lift / max(rb["n"],1) * 100,
        "env_total":  r["env_total"],
    })

# Sort by frequency lift (desc) for Q1
gate_importance_freq = sorted(gate_importance, key=lambda x: -x["freq_lift"])
# Sort by PF cost (desc) for Q2/Q4
gate_importance_pf   = sorted(gate_importance, key=lambda x: -x["pf_cost"])

print(f"\n  Ranked by Frequency Lift (removing gate → most new trades):")
print(f"  {'Gate':25s}  {'n_after':>7}  {'Δn':>6}  {'Δn%':>6}  "
      f"{'PF_after':>9}  {'ΔPF':>7}  {'env_bars':>9}")
print("  " + "─"*72)
for g in gate_importance_freq:
    dpf_s = f"{-g['pf_cost']:+.3f}"
    print(f"  {g['gate']:25s}  {g['n_after']:7d}  {g['freq_lift']:+6d}  "
          f"{g['n_pct_gain']:+5.0f}%  {g['pf_after']:9.3f}  {dpf_s:>7}  "
          f"{g['env_total']:9,}")

print(f"\n  Ranked by PF Cost (removing gate → biggest PF drop = gate is important):")
print(f"  {'Gate':25s}  {'PF_after':>9}  {'ΔPF':>7}  {'Importance':>12}")
print("  " + "─"*62)
for g in gate_importance_pf:
    if g["pf_cost"] > 0:
        importance = "HIGH" if g["pf_cost"] > 0.3 else "MEDIUM" if g["pf_cost"] > 0.1 else "LOW"
    else:
        importance = "REDUNDANT"
    print(f"  {g['gate']:25s}  {g['pf_after']:9.3f}  {-g['pf_cost']:+7.3f}  {importance:>12}")

# =============================================================================
# RESEARCH QUESTIONS
# =============================================================================

print("\n" + "═"*78)
print("  RESEARCH QUESTIONS")
print("═"*78)

# Q1: Which gate removes most trades?
q1 = gate_importance_freq[0]
print(f"""
  Q1. Which gate suppresses trade frequency the most?
      Removing {q1['gate']} adds the most trades: Δn=+{q1['freq_lift']} ({q1['n_pct_gain']:+.0f}%)
      → This gate is the biggest frequency suppressor.

  Q2. Which gate contributes most to profitability?
      Removing {gate_importance_pf[0]['gate']} drops PF by {gate_importance_pf[0]['pf_cost']:.3f}
      (from {results['BASELINE']['pf']:.3f} to {gate_importance_pf[0]['pf_after']:.3f})
      → This is the most important gate for edge.
""")

q3_candidates = [(vid, results[vid]) for vid in ["VAR_A","VAR_B","VAR_C","VAR_D"]
                 if results[vid]["n"] >= 100 and results[vid]["pf"] > 1.20]
if q3_candidates:
    q3_str = ", ".join(f"{v} (n={r['n']}, PF={r['pf']:.3f})" for v,r in q3_candidates)
    print(f"  Q3. Can removing one gate reach n≥100 AND PF>1.20?")
    print(f"      YES — {q3_str}")
else:
    best_q3 = max(["VAR_A","VAR_B","VAR_C","VAR_D"], key=lambda v: results[v]["n"])
    print(f"  Q3. Can removing one gate reach n≥100 AND PF>1.20?")
    print(f"      NO — best single removal: {best_q3} (n={results[best_q3]['n']}, "
          f"PF={results[best_q3]['pf']:.3f})")

q4 = gate_importance_pf[0]
print(f"""
  Q4. Which gate is essential (removing it most hurts PF)?
      {q4['gate']}  (ΔPF={-q4['pf_cost']:+.3f})

  Q5. Which gate is redundant (removing it barely changes PF)?
      {gate_importance_pf[-1]['gate']}  (ΔPF={-gate_importance_pf[-1]['pf_cost']:+.3f})
""")

# Q6: Full ranked importance table
print("  Q6. Full gate importance ranking:")
print(f"  {'Rank':>4}  {'Gate':25s}  {'PF Cost':>8}  {'Freq Gain':>10}  {'Importance':>12}")
print("  " + "─"*65)
all_ranked = sorted(gate_importance, key=lambda x: -abs(x["pf_cost"]))
for i, g in enumerate(all_ranked, 1):
    if g["pf_cost"] > 0.3:    imp = "ESSENTIAL"
    elif g["pf_cost"] > 0.1:  imp = "HIGH"
    elif g["pf_cost"] > 0.0:  imp = "MEDIUM"
    else:                      imp = "REDUNDANT"
    print(f"  {i:4d}  {g['gate']:25s}  {g['pf_cost']:+8.3f}  {g['freq_lift']:+10d}  {imp:>12}")

# =============================================================================
# MINIMUM VIABLE ENVIRONMENT RECOMMENDATION
# =============================================================================

print("\n" + "═"*78)
print("  MINIMUM VIABLE ENVIRONMENT — Final Recommendation")
print("═"*78)

# Score each variant on 4 criteria
TARGET_PF = 1.20; TARGET_N = 100; TARGET_BOOT = 1.20; TARGET_MC = 0.60

for vid, vname, _ in VARIANTS:
    r = results[vid]
    score = sum([r["pf"] > TARGET_PF, r["n"] >= TARGET_N,
                 r["b50"] > TARGET_BOOT, r["mc_p"] > TARGET_MC])
    results[vid]["score"] = score

best_vid = max(VAR_IDS, key=lambda v: (results[v]["score"], results[v]["pf"]))
best_r   = results[best_vid]

print(f"\n  Evaluation (PF>1.20, n≥100, Boot_p50>1.20, MC_P>60%):")
print(f"\n  {'Variant':10s}  {'Gates':30s}  {'✓/4':>4}  {'n':>5}  {'PF':>7}  "
      f"{'Boot_p50':>9}  {'MC%':>5}  {'Verdict':>12}")
print("  " + "─"*90)

for vid, vname, _ in VARIANTS:
    r = results[vid]
    checks = [
        "✓" if r["pf"] > TARGET_PF else "✗",
        "✓" if r["n"] >= TARGET_N  else "✗",
        "✓" if r["b50"] > TARGET_BOOT else "✗",
        "✓" if r["mc_p"] > TARGET_MC  else "✗",
    ]
    s = r["score"]
    if s == 4: vdct = "VIABLE ★"
    elif s == 3: vdct = "CANDIDATE"
    elif s >= 2: vdct = "PARTIAL"
    else: vdct = "REJECT"
    mark = "→" if vid == best_vid else " "
    print(f"  {mark}{vid:10s}  {vname:30s}  {s}/4  {r['n']:5d}  {r['pf']:7.3f}  "
          f"{r['b50']:9.3f}  {r['mc_p']*100:5.1f}%  {vdct:>12}")

print(f"""
  ══ RECOMMENDED MINIMUM VIABLE ENVIRONMENT ══════════════════════════════
  Variant : {best_vid} — {best_r['name']}
  Gates kept :""")
for gname, active in results[best_vid]["gates"].items():
    full = {"atr":"ATR Rank < p25","slope":"EMA200 Slope > 0",
            "dist":"EMA Distance > p75","bb":"BB Width < p33"}[gname]
    print(f"    {'✓' if active else '✗ (removed)'} {full}")

print(f"""
  Results : n={best_r['n']}  PF={best_r['pf']:.3f}  WR={best_r['wr']*100:.1f}%
            Boot p50={best_r['b50']:.3f} [{best_r['b5']:.3f},{best_r['b95']:.3f}]
            MC P(profit)={best_r['mc_p']*100:.1f}%  MDD={abs(best_r['mdd'])*100:.1f}%
            LOO-sym floor={best_r['sym_floor']:.3f}  LOO-fold floor={best_r['fold_floor']:.3f}
  ═══════════════════════════════════════════════════════════════════════════
""")

# =============================================================================
# CHARTS
# =============================================================================

print("  Generating charts …")

def _style(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=TXT, fontsize=8, pad=4)

# ── 1: Summary bar chart ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5), facecolor=BG)
fig.suptitle("R037 — Environment Sensitivity: removing one gate at a time",
             color=TXT, fontsize=11)

metrics_plot = [
    ("n",     "Trade Count",    TARGET_N,    False),
    ("pf",    "Profit Factor",  TARGET_PF,   False),
    ("b50",   "Bootstrap p50",  TARGET_BOOT, False),
    ("mc_p",  "MC P(profit)",   TARGET_MC,   True),
]
for ax_, (key, title, target, pct) in zip(axes, metrics_plot):
    _style(ax_, title)
    vals = [results[v][key] * (100 if pct else 1) for v, *_ in VARIANTS]
    tgt  = target * (100 if pct else 1)
    cols = [COLOURS_V[v] for v, *_ in VARIANTS]
    xlabels = [v for v, *_ in VARIANTS]
    bars = ax_.bar(xlabels, vals, color=cols, alpha=0.85)
    ax_.axhline(tgt, color=AMBER, lw=0.9, ls=":", alpha=0.8, label=f"Target={tgt}")
    for b, val in zip(bars, vals):
        ax_.text(b.get_x()+b.get_width()/2, val+tgt*0.01, f"{val:.1f}",
                 ha="center", color=TXT, fontsize=8, fontweight="bold")
    ax_.tick_params(axis="x", rotation=30)
    ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

plt.tight_layout()
p = f"{OUT}/r037_sensitivity_bars.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 2: PF vs N scatter ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG)
_style(ax, "R037 — PF vs Trade Count: Gate Sensitivity Space")
for vid, vname, _ in VARIANTS:
    r = results[vid]
    ax.scatter(r["n"], r["pf"], s=180, color=COLOURS_V[vid], zorder=5,
               label=f"{vid}: {vname}")
    ax.annotate(vid, (r["n"], r["pf"]), xytext=(6, 3),
                textcoords="offset points", color=COLOURS_V[vid], fontsize=8)
ax.axhline(1.20,    color=AMBER,  lw=0.8, ls=":", alpha=0.8, label="PF=1.20 target")
ax.axvline(100,     color=AMBER,  lw=0.8, ls=":", alpha=0.8, label="n=100 target")
ax.fill_between([100, ax.get_xlim()[1] if ax.get_xlim()[1] > 100 else 200],
                1.20, 3.0, alpha=0.06, color=GREEN, label="VIABLE zone")
ax.set_xlabel("OOS Trade Count", color="#888", fontsize=10)
ax.set_ylabel("Profit Factor",   color="#888", fontsize=10)
ax.legend(facecolor=PANEL, labelcolor=TXT, fontsize=8)
plt.tight_layout()
p = f"{OUT}/r037_pf_vs_n.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 3: Fold stability lines ───────────────────────────────────────────────────
fold_pfs = {vid: [] for vid, *_ in VARIANTS}
fold_ns  = {vid: [] for vid, *_ in VARIANTS}
for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    for vid, _, gates in VARIANTS:
        tl = [t for sym in SYMBOLS
              for t in var_sym_trades[vid][sym] if t["fold"] == fold_idx]
        m  = metrics(tl)
        fold_pfs[vid].append(m["pf"])
        fold_ns[vid].append(m["n"])

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.suptitle("R037 — Fold Stability", color=TXT, fontsize=11)

ax_ = axes[0]
_style(ax_, "PF by Fold")
for vid, *_ in VARIANTS:
    ax_.plot(range(1, 6), fold_pfs[vid], marker="o", color=COLOURS_V[vid],
             lw=1.5, ms=5, label=vid)
ax_.axhline(1.20, color=AMBER, lw=0.7, ls=":", alpha=0.7)
ax_.axhline(1.0,  color="#555", lw=0.5, ls="--")
ax_.set_xlabel("Fold", color="#888"); ax_.set_ylabel("PF", color="#888")
ax_.set_xticks(range(1,6))
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

ax_ = axes[1]
_style(ax_, "Trade Count by Fold")
for vid, *_ in VARIANTS:
    ax_.plot(range(1, 6), fold_ns[vid], marker="o", color=COLOURS_V[vid],
             lw=1.5, ms=5, label=vid)
ax_.axhline(20, color=AMBER, lw=0.7, ls=":", alpha=0.7, label="n=20/fold reference")
ax_.set_xlabel("Fold", color="#888"); ax_.set_ylabel("n", color="#888")
ax_.set_xticks(range(1,6))
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)
plt.tight_layout()
p = f"{OUT}/r037_fold_stability.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 4: LOO-symbol floor bars ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(VARIANTS), figsize=(24, 5), facecolor=BG)
fig.suptitle("R037 — Leave-One-Symbol-Out PF", color=TXT, fontsize=11)
for ax_, (vid, vname, _) in zip(axes, VARIANTS):
    _style(ax_, f"{vid}\n{vname}")
    ls  = results[vid]["loo_sym"]
    tags  = [s.split("-")[0] for s in ls]
    pfls  = list(ls.values())
    cols  = [GREEN if v>1.2 else (AMBER if v>1.0 else RED) for v in pfls]
    bars_ = ax_.bar(tags, pfls, color=cols, alpha=0.85)
    ax_.axhline(1.0,  color="#888", lw=0.7, ls="--")
    ax_.axhline(1.20, color=AMBER,  lw=0.7, ls=":")
    ax_.tick_params(axis="x", rotation=45, labelsize=5)
    ax_.set_ylabel("LOO PF", color="#888", fontsize=6)
    for b, v in zip(bars_, pfls):
        ax_.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}",
                 ha="center", color=TXT, fontsize=5)
plt.tight_layout()
p = f"{OUT}/r037_loo_symbol.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 5: Bootstrap CI comparison ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
_style(ax, "R037 — Bootstrap PF: 95% CI per Variant")
x = np.arange(len(VARIANTS))
ax.bar(x, [results[v]["pf"]  for v,*_ in VARIANTS],
       color=[COLOURS_V[v] for v,*_ in VARIANTS], alpha=0.5, width=0.5, label="PF")
ax.errorbar(x, [results[v]["b50"] for v,*_ in VARIANTS],
            yerr=[[results[v]["b50"]-results[v]["b5"]  for v,*_ in VARIANTS],
                  [results[v]["b95"]-results[v]["b50"] for v,*_ in VARIANTS]],
            fmt="o", color="white", capsize=5, ms=4, label="Boot median ± CI95")
ax.axhline(1.20, color=AMBER, lw=0.8, ls=":", label="1.20 target")
ax.set_xticks(x)
ax.set_xticklabels([f"{v}\n{VAR_NAMES[v]}" for v,*_ in VARIANTS], color=TXT, fontsize=7)
ax.set_ylabel("Profit Factor", color="#888", fontsize=9)
ax.legend(facecolor=PANEL, labelcolor=TXT, fontsize=8)
plt.tight_layout()
p = f"{OUT}/r037_bootstrap_ci.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 6: Equity curves best variant ────────────────────────────────────────────
COLOURS_SYM = {
    "BTC-USDT-SWAP":"#F7931A","ETH-USDT-SWAP":"#627EEA","SOL-USDT-SWAP":"#9945FF",
    "LINK-USDT-SWAP":"#2A5ADA","AVAX-USDT-SWAP":"#E84142","XRP-USDT-SWAP":"#346AA9",
    "LTC-USDT-SWAP":"#BFBBBB","BCH-USDT-SWAP":"#8DC351","DOGE-USDT-SWAP":"#C3A634",
    "ADA-USDT-SWAP":"#0033AD","BNB-USDT-SWAP":"#F3BA2F","DOT-USDT-SWAP":"#E6007A",
    "ARB-USDT-SWAP":"#28A0F0","OP-USDT-SWAP":"#FF0420","NEAR-USDT-SWAP":"#00C08B",
    "ATOM-USDT-SWAP":"#6F4CFF","SUI-USDT-SWAP":"#6FBCF0","APT-USDT-SWAP":"#00B4D8",
    "WIF-USDT-SWAP":"#A67C52","PEPE-USDT-SWAP":"#4CAF50","ENA-USDT-SWAP":"#8B0000",
    "UNI-USDT-SWAP":"#FF007A","FIL-USDT-SWAP":"#0090FF",
}

fig, axes = plt.subplots(1, len(VARIANTS), figsize=(28, 5), facecolor=BG)
fig.suptitle("R037 — Equity Curves (per symbol)", color=TXT, fontsize=11)
for ax_, (vid, vname, _) in zip(axes, VARIANTS):
    _style(ax_, f"{vid} PF={results[vid]['pf']:.3f} n={results[vid]['n']}")
    for sym in SYMBOLS:
        tl = var_sym_trades[vid][sym]
        if not tl: continue
        eq_ = CAPITAL + np.cumsum([t["pnl"] for t in tl])
        ax_.plot(eq_, color=COLOURS_SYM.get(sym,"#888"), lw=0.9, alpha=0.75)
    ax_.axhline(CAPITAL, color="#444", lw=0.5, ls="--")
    ax_.set_ylabel("Equity ($)", color="#888", fontsize=7)
plt.tight_layout()
p = f"{OUT}/r037_equity_curves.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 7: Dashboard ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(26, 16), facecolor=BG)
gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.52, wspace=0.35,
                        top=0.93, bottom=0.05, left=0.04, right=0.98)

# Row 0: PF, n, Boot p50, MC% bars — one per gate
for col, (key, title, target, pct) in enumerate(metrics_plot):
    ax_ = fig.add_subplot(gs[0, col])
    _style(ax_, title)
    vals = [results[v][key] * (100 if pct else 1) for v,*_ in VARIANTS]
    tgt  = target * (100 if pct else 1)
    cols = [COLOURS_V[v] for v,*_ in VARIANTS]
    bars_ = ax_.bar([v for v,*_ in VARIANTS], vals, color=cols, alpha=0.85)
    ax_.axhline(tgt, color=AMBER, lw=0.9, ls=":", alpha=0.8)
    for b, val in zip(bars_, vals):
        ax_.text(b.get_x()+b.get_width()/2, val+tgt*0.01,
                 f"{val:.1f}", ha="center", color=TXT, fontsize=7, fontweight="bold")
    ax_.tick_params(axis="x", rotation=30, labelsize=6)

# PF vs n scatter
ax_ = fig.add_subplot(gs[0, 4])
_style(ax_, "PF vs n")
for vid, *_ in VARIANTS:
    r = results[vid]
    ax_.scatter(r["n"], r["pf"], s=120, color=COLOURS_V[vid], zorder=5)
    ax_.annotate(vid, (r["n"], r["pf"]), xytext=(4,2),
                 textcoords="offset points", color=COLOURS_V[vid], fontsize=6)
ax_.axhline(1.20, color=AMBER, lw=0.6, ls=":"); ax_.axvline(100, color=AMBER, lw=0.6, ls=":")
ax_.set_xlabel("n", color="#888", fontsize=7); ax_.set_ylabel("PF", color="#888", fontsize=7)

# Row 1: Fold stability PF, fold n, Bootstrap CI, LOO summary, recommendation text
ax_ = fig.add_subplot(gs[1, 0])
_style(ax_, "PF by Fold")
for vid, *_ in VARIANTS:
    ax_.plot(range(1,6), fold_pfs[vid], marker="o", ms=4, lw=1.2,
             color=COLOURS_V[vid], label=vid)
ax_.axhline(1.20, color=AMBER, lw=0.6, ls=":"); ax_.axhline(1.0, color="#555", lw=0.5, ls="--")
ax_.set_xticks(range(1,6)); ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=5, ncol=2)
ax_.set_ylabel("PF", color="#888", fontsize=7)

ax_ = fig.add_subplot(gs[1, 1])
_style(ax_, "Trades by Fold")
for vid, *_ in VARIANTS:
    ax_.plot(range(1,6), fold_ns[vid], marker="o", ms=4, lw=1.2,
             color=COLOURS_V[vid], label=vid)
ax_.set_xticks(range(1,6)); ax_.set_ylabel("n", color="#888", fontsize=7)
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=5, ncol=2)

# Equity curves best variant
ax_ = fig.add_subplot(gs[1, 2])
_style(ax_, f"Equity {best_vid}  PF={best_r['pf']:.3f}")
for sym in SYMBOLS:
    tl = var_sym_trades[best_vid][sym]
    if not tl: continue
    eq_ = CAPITAL + np.cumsum([t["pnl"] for t in tl])
    ax_.plot(eq_, color=COLOURS_SYM.get(sym,"#888"), lw=0.8, alpha=0.7)
ax_.axhline(CAPITAL, color="#444", lw=0.5, ls="--")
ax_.set_ylabel("Equity ($)", color="#888", fontsize=7)

# Gate importance
ax_ = fig.add_subplot(gs[1, 3])
_style(ax_, "Gate PF Cost (importance)")
g_names  = [g["gate"].split()[0] for g in gate_importance_pf]
g_costs  = [g["pf_cost"] for g in gate_importance_pf]
g_cols   = [GREEN if v>0.3 else (AMBER if v>0.1 else RED) for v in g_costs]
ax_.barh(g_names[::-1], g_costs[::-1], color=g_cols[::-1], alpha=0.85)
ax_.axvline(0, color="#555", lw=0.5)
ax_.set_xlabel("PF drop when removed", color="#888", fontsize=7)

# Recommendation panel
ax_ = fig.add_subplot(gs[1, 4])
ax_.set_facecolor(PANEL)
for sp in ax_.spines.values(): sp.set_visible(False)
ax_.set_xticks([]); ax_.set_yticks([])
rec_text = (f"RECOMMENDATION\n\n"
            f"Best: {best_vid}\n"
            f"'{best_r['name']}'\n\n"
            f"n      = {best_r['n']}\n"
            f"PF     = {best_r['pf']:.3f}\n"
            f"WR     = {best_r['wr']*100:.1f}%\n"
            f"p50    = {best_r['b50']:.3f}\n"
            f"MC_P   = {best_r['mc_p']*100:.1f}%\n"
            f"MDD    = {abs(best_r['mdd'])*100:.1f}%\n"
            f"LOO-S  = {best_r['sym_floor']:.3f}\n"
            f"LOO-F  = {best_r['fold_floor']:.3f}\n\n"
            f"Score  = {best_r['score']}/4")
ax_.text(0.07, 0.95, rec_text, transform=ax_.transAxes,
         color=TXT, fontsize=8, fontfamily="monospace", va="top",
         bbox=dict(boxstyle="round", facecolor="#0d1117", edgecolor="#444"))

# Row 2: Bootstrap CI chart + LOO summary table
ax_ = fig.add_subplot(gs[2, 0:3])
_style(ax_, "Bootstrap PF — Median ± 95% CI per Variant")
x = np.arange(len(VARIANTS))
ax_.bar(x, [results[v]["pf"] for v,*_ in VARIANTS],
        color=[COLOURS_V[v] for v,*_ in VARIANTS], alpha=0.4, width=0.55)
ax_.errorbar(x, [results[v]["b50"] for v,*_ in VARIANTS],
             yerr=[[results[v]["b50"]-results[v]["b5"]  for v,*_ in VARIANTS],
                   [results[v]["b95"]-results[v]["b50"] for v,*_ in VARIANTS]],
             fmt="o", color="white", capsize=6, ms=5, label="Boot p50 ± CI95")
ax_.axhline(1.20, color=AMBER, lw=0.8, ls=":")
ax_.set_xticks(x)
ax_.set_xticklabels([f"{v}\n{VAR_NAMES[v]}" for v,*_ in VARIANTS], color=TXT, fontsize=7)
ax_.set_ylabel("Profit Factor", color="#888", fontsize=8)
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=8)

# LOO table text
ax_ = fig.add_subplot(gs[2, 3:5])
ax_.set_facecolor(PANEL)
for sp in ax_.spines.values(): sp.set_visible(False)
ax_.set_xticks([]); ax_.set_yticks([])
tbl_lines = ["LOO ROBUSTNESS\n",
             f"{'Variant':10s}  {'LOO-sym':>8}  {'LOO-fld':>8}  {'All>1.0?':>8}"]
tbl_lines += ["─"*42]
for vid, vname, _ in VARIANTS:
    r   = results[vid]
    sym_ok  = "YES ✓" if all(v>1.0 for v in r["loo_sym"].values()) else "NO ✗"
    fold_ok = "YES ✓" if all(v>1.0 for v in r["loo_fld"].values()) else "NO ✗"
    tbl_lines.append(f"{vid:10s}  {r['sym_floor']:8.3f}  {r['fold_floor']:8.3f}  {sym_ok:>8}")
ax_.text(0.03, 0.95, "\n".join(tbl_lines), transform=ax_.transAxes,
         color=TXT, fontsize=8, fontfamily="monospace", va="top")

fig.suptitle(
    f"QUANTLAB AI — R037 | Environment Sensitivity Analysis\n"
    f"Entry: Strategy C (RelVol Breakout) | 5-fold WF | "
    f"Best viable: {best_vid} ({best_r['name']}) PF={best_r['pf']:.3f} n={best_r['n']}",
    color=TXT, fontsize=11, y=0.975
)
dash_path = f"{OUT}/r037_dashboard.png"
fig.savefig(dash_path, dpi=120, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"  → {dash_path}")

# =============================================================================
# SAVE TRADES & JOURNAL
# =============================================================================

for vid, *_ in VARIANTS:
    path_ = f"{OUT}/r037_{vid.lower()}_trades.csv"
    pd.DataFrame(results[vid]["all_trades"]).to_csv(path_, index=False)

journal_path = CONFIG["JOURNAL_FILE"]
for vid, vname, _ in VARIANTS:
    r = results[vid]
    row = {
        "research_id": f"{RESEARCH_ID}-{vid}",
        "date":        pd.Timestamp.now().strftime("%Y-%m-%d"),
        "strategy":    f"ENV_SENSITIVITY_{vid}",
        "timeframe":   "1H",
        "symbols":     str(len(SYMBOLS)),
        "method":      "sensitivity-5fold-WF",
        "n_oos":       r["n"],
        "wr":          round(r["wr"], 4),
        "pf":          round(r["pf"], 4),
        "sharpe":      0.0,
        "mdd":         round(r["mdd"], 4),
        "net":         round(r["net"], 2),
        "boot_p50":    round(r["b50"], 4),
        "mc_prob":     round(r["mc_p"], 4),
        "loo_floor":   round(r["sym_floor"], 4),
        "verdict":     "VIABLE" if r["score"]==4 else ("CANDIDATE" if r["score"]==3 else "PARTIAL"),
    }
    jdf = pd.read_csv(journal_path) if os.path.exists(journal_path) else pd.DataFrame()
    jdf = pd.concat([jdf, pd.DataFrame([row])], ignore_index=True)
    jdf.to_csv(journal_path, index=False)
print(f"  Journal updated → {journal_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "═"*78)
print(f"  R037 COMPLETE — Environment Sensitivity Analysis")
print("═"*78)

print(f"\n  {'Variant':10s}  {'Description':30s}  {'n':>5}  {'PF':>7}  "
      f"{'p50':>7}  {'MC%':>6}  {'Score':>6}  {'Verdict':>10}")
print("  " + "─"*84)
for vid, vname, _ in VARIANTS:
    r = results[vid]
    s = r["score"]
    vdct = "VIABLE ★" if s==4 else ("CANDIDATE" if s==3 else "PARTIAL")
    print(f"  {vid:10s}  {vname:30s}  {r['n']:5d}  {r['pf']:7.3f}  "
          f"{r['b50']:7.3f}  {r['mc_p']*100:5.1f}%  {s:>6}/4  {vdct:>10}")

print(f"""
  ── Gate Importance Ranking ──
  {'Rank':>4}  {'Gate':25s}  {'PF Cost':>8}  {'Freq Gain':>10}  {'Role':>10}""")
for i, g in enumerate(all_ranked, 1):
    role = "ESSENTIAL" if g["pf_cost"]>0.3 else "HIGH" if g["pf_cost"]>0.1 else \
           "MEDIUM" if g["pf_cost"]>0 else "REDUNDANT"
    print(f"  {i:4d}  {g['gate']:25s}  {g['pf_cost']:+8.3f}  {g['freq_lift']:+10d}  {role:>10}")

print(f"""
  Minimum Viable Environment: {best_vid} — {best_r['name']}
  n={best_r['n']}  PF={best_r['pf']:.3f}  Boot_p50={best_r['b50']:.3f}  MC_P={best_r['mc_p']*100:.1f}%

  Output : {OUT}/r037_*
""" + "═"*78)
