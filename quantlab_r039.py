"""
=============================================================================
QUANTLAB AI — RESEARCH #039
Environment Expansion Study
=============================================================================

Background:
  R038 proved the market environment is the edge.
  6/9 entry families achieved PF>1.20 inside the BASELINE environment.
  RELVOL reached PF=1.817. Sole blocker: n<100 (environment too selective).

Objective:
  Increase trade count while preserving edge.
  Fixed entry: RELVOL Breakout (vol>1.5× avg + bullish candle).
  Relax ONE or more environment gates, testing 8 variants (A–H).

Variants:
  A — BASELINE:            ATR<p25 + slope>0 + EMA_dist>p75 + BB<p33
  B — Relax ATR:           ATR<p35 + slope>0 + EMA_dist>p75 + BB<p33
  C — Relax EMA Dist:      ATR<p25 + slope>0 + EMA_dist>p60 + BB<p33
  D — Relax BB:            ATR<p25 + slope>0 + EMA_dist>p75 + BB<p50
  E — Relax ATR + Dist:    ATR<p35 + slope>0 + EMA_dist>p60 + BB<p33
  F — Relax ATR + BB:      ATR<p35 + slope>0 + EMA_dist>p75 + BB<p50
  G — Relax Dist + BB:     ATR<p25 + slope>0 + EMA_dist>p60 + BB<p50
  H — Relax all three:     ATR<p35 + slope>0 + EMA_dist>p60 + BB<p50

EMA200 slope>0 is preserved in all variants.

Entry:  RELVOL — vol > 1.5× 20-bar avg AND close > open AND close > prev_close
Exit:   Stop = 1×ATR14   Target = 2×ATR14   (2R fixed)
Method: 5-fold expanding walk-forward, IS thresholds only
PROMOTE: PF>1.20 · n≥100 · Boot_p50>1.20 · MC_P>60%
         LOO-sym>1.0 · LOO-fold>1.0 · MDD<25%
=============================================================================
"""

import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr

RESEARCH_ID = "R039"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = CONFIG["RISK_REWARD"]   # 2.0
BEP_WR   = 1.0 / (1.0 + RR)       # 0.333

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
N_BOOT   = 2_000

TARGET_PF   = 1.20
TARGET_N    = 100
TARGET_BOOT = 1.20
TARGET_MC   = 0.60

# Variant definitions: (id, label, atr_pct, ema_dist_pct, bb_pct)
# atr_pct/ema_dist_pct/bb_pct are the quantile thresholds
# EMA200 slope>0 is always on
VARIANTS = [
    ("A", "BASELINE",           0.25, 0.75, 0.33),
    ("B", "Relax ATR p35",      0.35, 0.75, 0.33),
    ("C", "Relax EMA_Dist p60", 0.25, 0.60, 0.33),
    ("D", "Relax BB p50",       0.25, 0.75, 0.50),
    ("E", "ATR p35 + Dist p60", 0.35, 0.60, 0.33),
    ("F", "ATR p35 + BB p50",   0.35, 0.75, 0.50),
    ("G", "Dist p60 + BB p50",  0.25, 0.60, 0.50),
    ("H", "All three relaxed",  0.35, 0.60, 0.50),
]
VAR_IDS   = [v[0] for v in VARIANTS]
VAR_NAMES = {v[0]: v[1] for v in VARIANTS}

COLOURS = {
    "A": "#888888","B": "#4CAF50","C": "#2196F3","D": "#FF9800",
    "E": "#9C27B0","F": "#E91E63","G": "#00BCD4","H": "#F44336",
}
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

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #039" + " "*50 + "║")
print("║  Environment Expansion Study" + " "*50 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Entry: RELVOL Breakout (fixed)
  Gates tested: ATR Rank, EMA Distance, BB Width (each relaxed individually and combined)
  EMA200 slope > 0 kept in ALL variants
  Stop=1×ATR14  Target=2×ATR14  5-fold WF  IS thresholds only
""")

# =============================================================================
# INDICATORS
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]

    df["ema200"]       = calc_ema(c, 200)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank"]     = df["atr14"].rolling(100).rank(pct=True) * 100

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_width"]     = (bb_mid + 2*bb_std - (bb_mid - 2*bb_std)) / bb_mid.replace(0, np.nan)

    df["ema_dist_pct"] = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / df["ema200"].shift(10).replace(0, np.nan)

    vol_ma             = v.rolling(20).mean()
    df["rel_vol"]      = v / vol_ma.replace(0, np.nan)

    df["prev_close"]   = c.shift(1)
    df["prev_atr14"]   = df["atr14"].shift(1)

    return df

# =============================================================================
# THRESHOLDS  (per IS period)
# =============================================================================

def learn_thresholds(df_is: pd.DataFrame, atr_q: float, dist_q: float, bb_q: float) -> dict:
    valid = df_is.dropna(subset=["atr_rank","ema_dist_pct","bb_width"])
    atr_thr  = float(valid["atr_rank"].quantile(atr_q))
    pos_dist = valid[valid["ema_dist_pct"] > 0]["ema_dist_pct"]
    dist_thr = float(pos_dist.quantile(dist_q) if len(pos_dist) > 10
                     else valid["ema_dist_pct"].quantile(dist_q))
    bb_thr   = float(valid["bb_width"].quantile(bb_q))
    return {"atr": atr_thr, "dist": dist_thr, "bb": bb_thr}

def in_environment(df: pd.DataFrame, thr: dict) -> pd.Series:
    return (
        (df["atr_rank"]     < thr["atr"])  &
        (df["ema200_slope"] > 0)            &
        (df["ema_dist_pct"] > thr["dist"])  &
        (df["bb_width"]     < thr["bb"])
    ).fillna(False)

# =============================================================================
# SIGNAL — RELVOL Breakout (fixed)
# =============================================================================

def signal_relvol(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    return (
        (df["rel_vol"]    > 1.5) &
        (df["close"]      > df["open"]) &
        (df["close"]      > df["prev_close"]) &
        env
    ).fillna(False)

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(df: pd.DataFrame, signal: pd.Series,
                 sym: str, fold: int, vid: str) -> list:
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
                cost  = (ep*sz + xp*sz) * (fee + spd)
                slpc  = (st - xp) * sz if sl_hit else 0.0
                net   = gross - cost - slpc
                rmul  = (xp - ep) / sd if sd > 0 else 0.0
                trades.append({
                    "sym": sym, "fold": fold, "variant": vid,
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

def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl

def metrics(trades: list) -> dict:
    if not trades:
        return {"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,"net":0.0,
                "sharpe":0.0,"mdd":0.0,"pnls":np.array([]),"equity":np.array([CAPITAL])}
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values
    wins = df["win"].values.astype(bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw   = pnl[wins].sum()       if nw else 0.0
    gl   = abs(pnl[~wins].sum()) if nl else 0.0
    pf   = safe_pf(gw, gl)
    wr   = nw / n
    exp_r = wr * RR - (1 - wr)
    equity = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak   = np.maximum.accumulate(equity)
    mdd    = float(((equity - peak) / peak).min())
    bpy    = 365 * 24
    ann    = (equity[-1] / CAPITAL) ** (bpy / max(n, 1)) - 1
    vol    = pnl.std() * math.sqrt(bpy) if n > 1 else 1e-9
    sharpe = ann / vol if vol > 0 else 0.0
    return {"n":n,"wr":wr,"pf":pf,"exp_r":exp_r,"net":float(pnl.sum()),
            "sharpe":sharpe,"mdd":mdd,"pnls":pnl,"equity":equity}

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed); pfs = []
    for _ in range(n_iter):
        s = rng.choice(pnls, len(pnls), replace=True)
        pfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
    return float(np.percentile(pfs,5)), float(np.percentile(pfs,50)), float(np.percentile(pfs,95))

def monte_carlo(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return {"prob_profit":0.0,"p5":CAPITAL,"p50":CAPITAL,"p95":CAPITAL,"finals":np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit":float((finals>CAPITAL).mean()),
            "p5":float(np.percentile(finals,5)),"p50":float(np.percentile(finals,50)),
            "p95":float(np.percentile(finals,95)),"finals":finals}

def loo_sym(sym_trades_dict):
    return {omit: metrics([t for s,tl in sym_trades_dict.items() if s!=omit for t in tl])["pf"]
            for omit in sym_trades_dict}

def loo_fld(all_trades):
    return {omit: metrics([t for t in all_trades if t["fold"]!=omit])["pf"]
            for omit in sorted({t["fold"] for t in all_trades})}

# =============================================================================
# DATA LOAD
# =============================================================================

print("─"*78)
print("  Loading 1H data …")
all_dfs = {}
for sym in SYMBOLS:
    tag  = sym.replace("-","_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path): continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) < MIN_BARS: continue
    all_dfs[sym] = add_features(df)
SYMBOLS = list(all_dfs.keys())
print(f"  {len(SYMBOLS)} symbols  ({sum(len(d) for d in all_dfs.values()):,} bars)")

# =============================================================================
# WALK-FORWARD  (all variants in one pass)
# =============================================================================

# var_sym_trades[vid][sym] = list of trades
var_sym_trades = {v[0]: {sym: [] for sym in SYMBOLS} for v in VARIANTS}
var_env_bars   = {v[0]: 0 for v in VARIANTS}
fold_pf_tbl    = {v[0]: [] for v in VARIANTS}   # PF per fold
fold_n_tbl     = {v[0]: [] for v in VARIANTS}   # n per fold

print(f"\n  Walk-forward: {len(FOLDS)} folds × {len(SYMBOLS)} symbols × "
      f"{len(VARIANTS)} variants  (entry: RELVOL fixed)")
print()

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_t = {v[0]: [] for v in VARIANTS}
    fold_e = {v[0]: 0  for v in VARIANTS}

    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[:int(N*is_end)]
        df_oos = df_full.iloc[int(N*is_end):int(N*oos_end)].reset_index(drop=True)
        if len(df_oos) < 100: continue

        # Learn thresholds for each variant's specific quantiles
        for vid, vlabel, atr_q, dist_q, bb_q in VARIANTS:
            thr = learn_thresholds(df_is, atr_q, dist_q, bb_q)
            env = in_environment(df_oos, thr)
            fold_e[vid] += int(env.sum())
            sig = signal_relvol(df_oos, env)
            tl  = run_backtest(df_oos, sig, sym, fold_idx, vid)
            var_sym_trades[vid][sym].extend(tl)
            fold_t[vid].extend(tl)

    # Fold summary
    print(f"  Fold {fold_idx}  (IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%)")
    for vid, vlabel, *_ in VARIANTS:
        m = metrics(fold_t[vid])
        fold_pf_tbl[vid].append(m["pf"])
        fold_n_tbl[vid].append(m["n"])
        var_env_bars[vid] += fold_e[vid]
        print(f"    {vid} {vlabel:25s}  env_bars={fold_e[vid]:5,}  "
              f"n={m['n']:4d}  PF={m['pf']:.3f}")
    print()

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print("─"*78)
print("  Computing aggregate statistics …")
results = {}

for vid, vlabel, atr_q, dist_q, bb_q in VARIANTS:
    all_flat = [t for sym in SYMBOLS for t in var_sym_trades[vid][sym]]
    m         = metrics(all_flat)
    b5,b50,b95 = bootstrap_pf(m["pnls"])
    mc         = monte_carlo(m["pnls"])
    ls         = loo_sym(var_sym_trades[vid])
    lf         = loo_fld(all_flat)
    sf         = min(ls.values()) if ls else 0.0
    ff         = min(lf.values()) if lf else 0.0

    score = sum([
        m["pf"]         > TARGET_PF,
        m["n"]          >= TARGET_N,
        b50             > TARGET_BOOT,
        mc["prob_profit"] > TARGET_MC,
        sf > 1.0,
        ff > 1.0,
        abs(m["mdd"])   < 0.25,
    ])

    results[vid] = {
        "label": vlabel, "atr_q": atr_q, "dist_q": dist_q, "bb_q": bb_q,
        "n": m["n"], "wr": m["wr"], "pf": m["pf"], "exp_r": m["exp_r"],
        "net": m["net"], "sharpe": m["sharpe"], "mdd": m["mdd"],
        "b5": b5, "b50": b50, "b95": b95,
        "mc_p": mc["prob_profit"], "mc_p5": mc["p5"],
        "mc_p50": mc["p50"], "mc_p95": mc["p95"],
        "sym_floor": sf, "fold_floor": ff,
        "pnls": m["pnls"], "equity": m["equity"],
        "loo_sym": ls, "loo_fld": lf,
        "mc_finals": mc["finals"],
        "env_bars": var_env_bars[vid],
        "score": score,
    }
    print(f"  {vid} {vlabel:25s}: n={m['n']:4d}  PF={m['pf']:.3f}  "
          f"p50={b50:.3f}  MC={mc['prob_profit']*100:.1f}%  "
          f"LOO-S={sf:.3f}  LOO-F={ff:.3f}  score={score}/7")

# =============================================================================
# VERDICT
# =============================================================================

def get_verdict(r):
    sc = r["score"]
    if sc == 7:                        return "PROMOTE"
    elif sc >= 5 and r["pf"] > 1.0:   return "WATCHLIST"
    elif sc >= 3:                      return "INVESTIGATE"
    else:                              return "REJECT"

for vid in VAR_IDS:
    results[vid]["verdict"] = get_verdict(results[vid])

baseline_n  = results["A"]["n"]
baseline_pf = results["A"]["pf"]

promote_list = [v for v in VAR_IDS if results[v]["verdict"] == "PROMOTE"]
watchlist    = [v for v in VAR_IDS if results[v]["verdict"] == "WATCHLIST"]

# Sort by score then PF
ranked = sorted(VAR_IDS, key=lambda v: (-results[v]["score"], -results[v]["pf"]))

# =============================================================================
# RESULTS TABLE
# =============================================================================

print("\n" + "═"*105)
print("  R039 — ENVIRONMENT EXPANSION (RELVOL fixed entry)")
print("═"*105)
print(f"  {'':1s}{'V':>2}  {'Label':25s}  {'ATR':>5}  {'Dist':>5}  {'BB':>5}  "
      f"{'n':>5}  {'Δn':>6}  {'WR':>6}  {'PF':>7}  {'p50':>7}  "
      f"{'MC%':>6}  {'LOO-S':>6}  {'LOO-F':>6}  {'Score':>5}  Verdict")
print("  " + "─"*115)

for vid in ranked:
    r   = results[vid]
    dn  = r["n"] - baseline_n
    dn_s = f"+{dn}" if dn >= 0 else str(dn)
    dpf = r["pf"] - baseline_pf
    flag = "★" if r["score"] == 7 else ("↑" if r["pf"] > TARGET_PF else " ")
    print(f"  {flag}{vid:>2}  {r['label']:25s}  {r['atr_q']:.2f}  "
          f"{r['dist_q']:.2f}  {r['bb_q']:.2f}  {r['n']:5d}  {dn_s:>6}  "
          f"{r['wr']*100:5.1f}%  {r['pf']:7.3f}  {r['b50']:7.3f}  "
          f"{r['mc_p']*100:5.1f}%  {r['sym_floor']:6.3f}  "
          f"{r['fold_floor']:6.3f}  {r['score']:3d}/7  {r['verdict']}")

# =============================================================================
# GATE IMPORTANCE (single-gate removals B, C, D)
# =============================================================================

print("\n" + "═"*105)
print("  GATE IMPORTANCE ANALYSIS")
print("═"*105)
gate_map = {
    "B": ("ATR Rank",    "p25 → p35"),
    "C": ("EMA Distance","p75 → p60"),
    "D": ("BB Width",    "p33 → p50"),
}
print(f"\n  Single-gate relaxations vs BASELINE:")
print(f"  {'Gate':12s}  {'Relaxation':12s}  {'n_after':>8}  {'Δn':>6}  "
      f"{'Δn%':>6}  {'PF_after':>9}  {'ΔPF':>7}  {'env_bars':>9}")
print("  " + "─"*80)
gate_importance = []
for vid, (gname, relax) in gate_map.items():
    r   = results[vid]
    rb  = results["A"]
    dn  = r["n"] - rb["n"]
    dpf = r["pf"] - rb["pf"]
    npct = dn / max(rb["n"], 1) * 100
    gate_importance.append({"vid":vid,"gate":gname,"relax":relax,
                             "dn":dn,"dpf":dpf,"npct":npct,
                             "n":r["n"],"pf":r["pf"],"env":r["env_bars"]})
    print(f"  {gname:12s}  {relax:12s}  {r['n']:8d}  {dn:+6d}  "
          f"{npct:+5.0f}%  {r['pf']:9.3f}  {dpf:+7.3f}  {r['env_bars']:9,}")

gate_freq = sorted(gate_importance, key=lambda x: -x["dn"])
gate_pf   = sorted(gate_importance, key=lambda x: -x["pf"])
gate_imp  = sorted(gate_importance, key=lambda x: x["dpf"])  # biggest PF drop = most important

# =============================================================================
# RESEARCH QUESTIONS
# =============================================================================

print("\n" + "═"*105)
print("  RESEARCH QUESTIONS")
print("═"*105)

q1_best = gate_freq[0]  # biggest n increase
q2_best = gate_pf[0]    # best preserved PF from single relaxations
q3_gate = gate_imp[0]   # most important gate (biggest PF drop)

# Q4: any variant with n>=100 AND PF>1.20?
q4_candidates = [(v, results[v]) for v in VAR_IDS
                 if results[v]["n"] >= TARGET_N and results[v]["pf"] > TARGET_PF]
q4_cands_all  = [(v, results[v]) for v in VAR_IDS if results[v]["n"] >= TARGET_N]

# Q5: best new environment
best_by_score = ranked[0] if ranked else "A"
best_r = results[best_by_score]

# For Q5 recommendation: best balance of frequency gain + PF preservation
# Score: n>=100 AND PF>1.20
qualify = [(v, results[v]) for v in VAR_IDS
           if results[v]["pf"] > TARGET_PF]
if q4_candidates:
    q5_rec = max(q4_candidates, key=lambda x: (x[1]["score"], x[1]["pf"]))[0]
    q5_text = f"PROMOTE → {q5_rec} ({results[q5_rec]['label']})"
elif qualify:
    # Find smallest relaxation with best PF (among all above 1.20)
    q5_rec = max(qualify, key=lambda x: x[1]["score"] * 10 + x[1]["pf"])[0]
    vdct   = results[q5_rec]["verdict"]
    q5_text = f"{vdct} → {q5_rec} ({results[q5_rec]['label']})"
else:
    q5_rec  = ranked[0]
    q5_text = f"INVESTIGATE → {q5_rec} ({results[q5_rec]['label']})"

print(f"""
  Q1. Which single relaxation produces the biggest trade-count increase?
      → {q1_best['gate']} ({q1_best['relax']})
        Variant {q1_best['vid']}: Δn=+{q1_best['dn']}  (+{q1_best['npct']:.0f}%)
        n={results[q1_best['vid']]['n']}  (baseline n={baseline_n})

  Q2. Which single relaxation preserves PF best?
      → Variant {q2_best['vid']} ({q2_best['gate']} {q2_best['relax']})
        PF={q2_best['pf']:.3f}  (ΔPFF={q2_best['dpf']:+.3f} vs baseline PF={baseline_pf:.3f})

  Q3. Which gate contributes most to rarity (removing it collapses PF most)?
      → {q3_gate['gate']} ({q3_gate['relax']})
        PF drops from {baseline_pf:.3f} to {q3_gate['pf']:.3f}  (ΔPF={q3_gate['dpf']:+.3f})
        This gate is ESSENTIAL for edge preservation.

  Q4. Can trade count exceed 100 while PF remains above 1.20?""")

if q4_candidates:
    for vid, r in q4_candidates:
        print(f"      YES → Variant {vid} ({r['label']}): n={r['n']}  PF={r['pf']:.3f}")
elif q4_cands_all:
    print(f"      NOT YET — n≥100 achieved but PF falls below 1.20:")
    for vid, r in q4_cands_all:
        print(f"      Variant {vid} ({r['label']}): n={r['n']}  PF={r['pf']:.3f}")
else:
    best_n_vid = max(VAR_IDS, key=lambda v: results[v]["n"])
    print(f"      NO — Best n={results[best_n_vid]['n']} in Variant {best_n_vid} "
          f"({results[best_n_vid]['label']})")

print(f"""
  Q5. Is there a new production environment?
      → {q5_text}""")
if q4_candidates:
    qv = q4_candidates[0][0]; qr = q4_candidates[0][1]
    print(f"      n={qr['n']}  PF={qr['pf']:.3f}  "
          f"Boot_p50={qr['b50']:.3f}  MC={qr['mc_p']*100:.1f}%  "
          f"LOO-S={qr['sym_floor']:.3f}  LOO-F={qr['fold_floor']:.3f}")
elif qualify:
    qr = results[q5_rec]
    print(f"      n={qr['n']}  PF={qr['pf']:.3f}  "
          f"Boot_p50={qr['b50']:.3f}  MC={qr['mc_p']*100:.1f}%  "
          f"LOO-S={qr['sym_floor']:.3f}  LOO-F={qr['fold_floor']:.3f}")

print()

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

# ── 1: Trade-off chart (n vs PF scatter) ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
fig.suptitle("R039 — Environment Expansion: Frequency vs Quality Trade-off",
             color=TXT, fontsize=11)

ax_ = axes[0]
_style(ax_, "n vs PF (RELVOL, 8 variants)")
for vid in VAR_IDS:
    r = results[vid]
    ax_.scatter(r["n"], r["pf"], s=160, color=COLOURS[vid], zorder=5)
    ax_.annotate(f"{vid}\n{r['label'][:12]}", (r["n"], r["pf"]),
                 xytext=(5, 3), textcoords="offset points",
                 color=COLOURS[vid], fontsize=6)
ax_.axhline(TARGET_PF, color=AMBER, lw=0.9, ls=":", label="PF=1.20")
ax_.axvline(TARGET_N,  color=AMBER, lw=0.9, ls="--", label="n=100")
ax_.fill_between([TARGET_N, max(results[v]["n"] for v in VAR_IDS)*1.1],
                  TARGET_PF, ax_.get_ylim()[1] if ax_.get_ylim()[1] > TARGET_PF else 3.0,
                  alpha=0.05, color=GREEN, label="Viable zone")
ax_.set_xlabel("OOS Trade Count", color="#888", fontsize=9)
ax_.set_ylabel("Profit Factor",   color="#888", fontsize=9)
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# ── PF bar chart ──────────────────────────────────────────────────────────────
ax_ = axes[1]
_style(ax_, "Profit Factor by Variant")
x_  = np.arange(len(VAR_IDS))
bars_ = ax_.bar(x_, [results[v]["pf"] for v in VAR_IDS],
                color=[COLOURS[v] for v in VAR_IDS], alpha=0.85)
ax_.axhline(TARGET_PF, color=AMBER, lw=0.9, ls=":")
ax_.axhline(baseline_pf, color="#888", lw=0.8, ls="--", label=f"Baseline={baseline_pf:.3f}")
ax_.set_xticks(x_)
ax_.set_xticklabels([f"{v}\n{results[v]['label'][:10]}" for v in VAR_IDS],
                     rotation=30, ha="right", color=TXT, fontsize=6)
ax_.set_ylabel("PF", color="#888", fontsize=8)
for b, v in zip(bars_, VAR_IDS):
    pf_ = results[v]["pf"]
    ax_.text(b.get_x()+b.get_width()/2, pf_+0.02,
             f"{pf_:.3f}", ha="center", color=TXT, fontsize=7, fontweight="bold")
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# ── n bar chart ───────────────────────────────────────────────────────────────
ax_ = axes[2]
_style(ax_, "Trade Count by Variant")
bars_ = ax_.bar(x_, [results[v]["n"] for v in VAR_IDS],
                color=[COLOURS[v] for v in VAR_IDS], alpha=0.85)
ax_.axhline(TARGET_N, color=AMBER, lw=0.9, ls=":")
ax_.axhline(baseline_n, color="#888", lw=0.8, ls="--", label=f"Baseline={baseline_n}")
ax_.set_xticks(x_)
ax_.set_xticklabels([f"{v}\n{results[v]['label'][:10]}" for v in VAR_IDS],
                     rotation=30, ha="right", color=TXT, fontsize=6)
ax_.set_ylabel("n trades", color="#888", fontsize=8)
for b, v in zip(bars_, VAR_IDS):
    n_ = results[v]["n"]
    ax_.text(b.get_x()+b.get_width()/2, n_+0.5, str(n_),
             ha="center", color=TXT, fontsize=7, fontweight="bold")
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

plt.tight_layout()
p = f"{OUT}/r039_tradeoff_chart.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 2: Equity Curves ─────────────────────────────────────────────────────────
n_cols = 4; n_rows = math.ceil(len(VAR_IDS)/n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows), facecolor=BG)
fig.suptitle("R039 — OOS Equity Curves by Variant (per symbol, RELVOL entry)",
             color=TXT, fontsize=11)
axes_flat = list(axes.flat)
for ax_, vid in zip(axes_flat, VAR_IDS):
    r = results[vid]
    _style(ax_, f"Var {vid}: {r['label']}\nPF={r['pf']:.3f}  n={r['n']}  "
                 f"Verdict={r['verdict']}")
    for sym in SYMBOLS:
        tl = var_sym_trades[vid][sym]
        if not tl: continue
        eq_ = CAPITAL + np.cumsum([t["pnl"] for t in tl])
        ax_.plot(eq_, color=SYM_COLS.get(sym,"#888"), lw=0.9, alpha=0.75)
    ax_.axhline(CAPITAL, color="#444", lw=0.5, ls="--")
    ax_.set_ylabel("Equity ($)", color="#888", fontsize=7)
for ax_ in axes_flat[len(VAR_IDS):]:
    ax_.set_visible(False)
plt.tight_layout()
p = f"{OUT}/r039_equity_curves.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 3: Heatmap ───────────────────────────────────────────────────────────────
metrics_keys = ["n","pf","b50","mc_p","sym_floor","fold_floor","wr","sharpe"]
metrics_labels = ["n","PF","Boot_p50","MC_P","LOO-sym","LOO-fold","WR","Sharpe"]
cmap_rb = LinearSegmentedColormap.from_list("rb", [RED, "#444", GREEN])

raw_data = np.array([[results[v][k]*(100 if k in ("mc_p","wr") else 1)
                      for k in metrics_keys] for v in VAR_IDS], dtype=float)

# Normalize column-wise for color
norm_data = np.zeros_like(raw_data)
for col in range(raw_data.shape[1]):
    col_vals = raw_data[:, col]
    vmin, vmax = col_vals.min(), col_vals.max()
    if vmax > vmin:
        norm_data[:, col] = (col_vals - vmin) / (vmax - vmin)
    else:
        norm_data[:, col] = 0.5

fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG)
_style(ax, "R039 — Variant Metric Heatmap (column-normalized)")
im = ax.imshow(norm_data, aspect="auto", cmap=cmap_rb, vmin=0.0, vmax=1.0)
ax.set_xticks(range(len(metrics_labels)))
ax.set_xticklabels(metrics_labels, color=TXT, fontsize=9)
ax.set_yticks(range(len(VAR_IDS)))
ax.set_yticklabels([f"{v}: {results[v]['label']}" for v in VAR_IDS],
                    color=TXT, fontsize=8)
for row, vid in enumerate(VAR_IDS):
    for col, k in enumerate(metrics_keys):
        val = raw_data[row, col]
        fmt = ".0f" if k == "n" else ".1f" if k in ("mc_p","wr") else ".3f"
        ax.text(col, row, f"{val:{fmt}}", ha="center", va="center",
                color="white", fontsize=8, fontweight="bold")
plt.colorbar(im, ax=ax, label="Normalized (0=worst 1=best)", fraction=0.02)
plt.tight_layout()
p = f"{OUT}/r039_heatmap.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 4: Dashboard ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 22), facecolor=BG)
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.35,
                         top=0.94, bottom=0.04, left=0.04, right=0.98)

# Row 0: Trade count, PF, Bootstrap p50, MC% — all variants
row0_items = [
    ("n",    "Trade Count",   TARGET_N,    False),
    ("pf",   "Profit Factor", TARGET_PF,   False),
    ("b50",  "Bootstrap p50", TARGET_BOOT, False),
    ("mc_p", "MC P(profit)",  TARGET_MC,   True),
]
for col, (key, title, tgt, pct) in enumerate(row0_items):
    ax_ = fig.add_subplot(gs[0, col])
    _style(ax_, title)
    vals_ = [results[v][key]*(100 if pct else 1) for v in VAR_IDS]
    tgt_  = tgt * (100 if pct else 1)
    bars_ = ax_.bar(range(len(VAR_IDS)), vals_,
                    color=[COLOURS[v] for v in VAR_IDS], alpha=0.85)
    ax_.axhline(tgt_, color=AMBER, lw=0.8, ls=":")
    ax_.set_xticks(range(len(VAR_IDS)))
    ax_.set_xticklabels([f"Var {v}" for v in VAR_IDS], rotation=30,
                         ha="right", color=TXT, fontsize=7)
    for b, val in zip(bars_, vals_):
        ax_.text(b.get_x()+b.get_width()/2, max(val,0)+abs(tgt_)*0.01,
                 f"{val:.1f}", ha="center", color=TXT, fontsize=7, fontweight="bold")

# Row 1: n vs PF scatter + fold stability lines
ax_sc = fig.add_subplot(gs[1, 0:2])
_style(ax_sc, "n vs PF  — Frequency vs Quality")
for vid in VAR_IDS:
    r = results[vid]
    ax_sc.scatter(r["n"], r["pf"], s=200, color=COLOURS[vid], zorder=5)
    ax_sc.annotate(f"Var {vid}", (r["n"], r["pf"]),
                   xytext=(5,3), textcoords="offset points",
                   color=COLOURS[vid], fontsize=7)
ax_sc.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":")
ax_sc.axvline(TARGET_N,  color=AMBER, lw=0.8, ls="--")
ax_sc.set_xlabel("n trades", color="#888", fontsize=8)
ax_sc.set_ylabel("PF",       color="#888", fontsize=8)

ax_fp = fig.add_subplot(gs[1, 2:4])
_style(ax_fp, "PF by Fold — All Variants")
for vid in VAR_IDS:
    ax_fp.plot(range(1,6), fold_pf_tbl[vid], marker="o", ms=4, lw=1.3,
               color=COLOURS[vid], label=f"Var {vid}", alpha=0.9)
ax_fp.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":")
ax_fp.axhline(1.0, color="#555", lw=0.5, ls="--")
ax_fp.set_xticks(range(1,6))
ax_fp.set_ylabel("PF", color="#888", fontsize=8)
ax_fp.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7, ncol=2)

# Row 2: Bootstrap CI bar chart + LOO floors
ax_bc = fig.add_subplot(gs[2, 0:2])
_style(ax_bc, "Bootstrap PF  Median ± 95% CI")
x_ = np.arange(len(VAR_IDS))
ax_bc.bar(x_, [results[v]["pf"] for v in VAR_IDS],
          color=[COLOURS[v] for v in VAR_IDS], alpha=0.35, width=0.55)
ax_bc.errorbar(x_, [results[v]["b50"] for v in VAR_IDS],
               yerr=[[results[v]["b50"]-results[v]["b5"] for v in VAR_IDS],
                     [results[v]["b95"]-results[v]["b50"] for v in VAR_IDS]],
               fmt="o", color="white", capsize=5, ms=4)
ax_bc.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":")
ax_bc.set_xticks(x_)
ax_bc.set_xticklabels([f"Var {v}" for v in VAR_IDS], rotation=30,
                       ha="right", color=TXT, fontsize=7)
ax_bc.set_ylabel("PF", color="#888", fontsize=8)

ax_loo = fig.add_subplot(gs[2, 2:4])
_style(ax_loo, "LOO Floors  (sym & fold)")
x_loo = np.arange(len(VAR_IDS)); w = 0.38
ax_loo.bar(x_loo-w/2, [results[v]["sym_floor"]  for v in VAR_IDS],
           w, color=[COLOURS[v] for v in VAR_IDS], alpha=0.85, label="LOO-sym")
ax_loo.bar(x_loo+w/2, [results[v]["fold_floor"] for v in VAR_IDS],
           w, color=[COLOURS[v] for v in VAR_IDS], alpha=0.45, label="LOO-fold")
ax_loo.axhline(1.0, color=AMBER, lw=0.8, ls=":")
ax_loo.set_xticks(x_loo)
ax_loo.set_xticklabels([f"Var {v}" for v in VAR_IDS], rotation=30,
                        ha="right", color=TXT, fontsize=7)
ax_loo.set_ylabel("LOO PF floor", color="#888", fontsize=8)
ax_loo.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# Row 3: Summary table + research answers
ax_tbl = fig.add_subplot(gs[3, 0:3])
ax_tbl.set_facecolor(PANEL)
for sp in ax_tbl.spines.values(): sp.set_visible(False)
ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
tbl_lines = [
    f"{'V':>2}  {'Label':25s}  {'ATR':>5}  {'Dist':>5}  {'BB':>5}  "
    f"{'n':>5}  {'Δn':>5}  {'PF':>7}  {'p50':>7}  {'MC%':>6}  "
    f"{'LOO-S':>6}  {'LOO-F':>6}  {'Score':>5}  Verdict",
    "─"*105,
]
for vid in ranked:
    r  = results[vid]
    dn = r["n"] - baseline_n
    dn_s = f"+{dn}" if dn >= 0 else str(dn)
    tbl_lines.append(
        f"{vid:>2}  {r['label']:25s}  {r['atr_q']:.2f}  "
        f"{r['dist_q']:.2f}  {r['bb_q']:.2f}  {r['n']:5d}  {dn_s:>5}  "
        f"{r['pf']:7.3f}  {r['b50']:7.3f}  {r['mc_p']*100:5.1f}%  "
        f"{r['sym_floor']:6.3f}  {r['fold_floor']:6.3f}  "
        f"{r['score']:3d}/7  {r['verdict']}"
    )
ax_tbl.text(0.01, 0.97, "\n".join(tbl_lines),
            transform=ax_tbl.transAxes, color=TXT, fontsize=7,
            fontfamily="monospace", va="top")

ax_ans = fig.add_subplot(gs[3, 3])
ax_ans.set_facecolor(PANEL)
for sp in ax_ans.spines.values(): sp.set_visible(False)
ax_ans.set_xticks([]); ax_ans.set_yticks([])
ans_text = (
    f"RESEARCH ANSWERS\n\n"
    f"Q1 Biggest freq gain:\n"
    f"  Var {q1_best['vid']} ({q1_best['gate']})\n"
    f"  Δn=+{q1_best['dn']}  n={results[q1_best['vid']]['n']}\n\n"
    f"Q2 Best PF preserved:\n"
    f"  Var {q2_best['vid']} ({q2_best['gate']})\n"
    f"  PF={q2_best['pf']:.3f}  ΔPF={q2_best['dpf']:+.3f}\n\n"
    f"Q3 Most essential gate:\n"
    f"  {q3_gate['gate']}\n"
    f"  ΔPF={q3_gate['dpf']:+.3f} when relaxed\n\n"
    f"Q4 n≥100 & PF>1.20?\n"
    f"  {'YES: '+', '.join(v for v,_ in q4_candidates) if q4_candidates else 'NO'}\n\n"
    f"Q5 New environment:\n"
    f"  {q5_text}\n\n"
    f"Entry: RELVOL (fixed)\n"
    f"Baseline: Var A n={baseline_n} PF={baseline_pf:.3f}"
)
ax_ans.text(0.05, 0.97, ans_text, transform=ax_ans.transAxes,
            color=TXT, fontsize=8, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#0d1117", edgecolor="#444"))

fig.suptitle(
    f"QUANTLAB AI — R039 | Environment Expansion Study | RELVOL Fixed Entry\n"
    f"8 variants (A–H) | 5-fold WF | {len(SYMBOLS)} symbols | "
    f"Best: Var {ranked[0]} ({results[ranked[0]]['label']}) "
    f"PF={results[ranked[0]]['pf']:.3f} n={results[ranked[0]]['n']} "
    f"— {results[ranked[0]]['verdict']}",
    color=TXT, fontsize=11, y=0.975
)
dash_path = f"{OUT}/r039_dashboard.png"
fig.savefig(dash_path, dpi=120, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"  → {dash_path}")

# =============================================================================
# CSV
# =============================================================================

csv_rows = []
for vid in ranked:
    r  = results[vid]
    dn = r["n"] - baseline_n
    csv_rows.append({
        "variant":          vid,
        "label":            r["label"],
        "atr_threshold":    r["atr_q"],
        "dist_threshold":   r["dist_q"],
        "bb_threshold":     r["bb_q"],
        "n_trades":         r["n"],
        "delta_n":          dn,
        "win_rate":         round(r["wr"], 4),
        "profit_factor":    round(r["pf"], 4),
        "expectancy_r":     round(r["exp_r"], 4),
        "sharpe":           round(r["sharpe"], 4),
        "max_drawdown":     round(r["mdd"], 4),
        "boot_p5":          round(r["b5"], 4),
        "boot_p50":         round(r["b50"], 4),
        "boot_p95":         round(r["b95"], 4),
        "mc_prob_profit":   round(r["mc_p"], 4),
        "loo_sym_floor":    round(r["sym_floor"], 4),
        "loo_fold_floor":   round(r["fold_floor"], 4),
        "env_bars":         r["env_bars"],
        "criteria_score":   r["score"],
        "verdict":          r["verdict"],
    })
csv_path = f"{OUT}/r039_variant_table.csv"
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"  → {csv_path}")

# =============================================================================
# JOURNAL MARKDOWN
# =============================================================================

def ck(c): return "✓" if c else "✗"

md = [
    "# QUANTLAB AI — R039: Environment Expansion Study\n",
    f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
    f"**Entry:** RELVOL Breakout (fixed — vol>1.5× avg + bullish candle)  ",
    f"**Exit:** Stop=1×ATR14 · Target=2×ATR14 (2R)  ",
    f"**Method:** 5-fold expanding WF · IS thresholds only  ",
    f"**Universe:** {len(SYMBOLS)} symbols · 27mo · 1H  ",
    f"**Baseline (Var A):** ATR<p25 · slope>0 · EMA_dist>p75 · BB<p33  \n",
    "## Variant Results\n",
    "| V | Label | ATR | Dist | BB | n | Δn | WR | PF | p50 | MC% | LOO-S | LOO-F | Score | Verdict |",
    "|---|-------|-----|------|----|----|-----|----|----|-----|-----|-------|-------|-------|---------|",
]
for vid in ranked:
    r  = results[vid]
    dn = r["n"] - baseline_n
    dn_s = f"+{dn}" if dn >= 0 else str(dn)
    md.append(
        f"| {vid} | {r['label']} | {r['atr_q']:.2f} | {r['dist_q']:.2f} | "
        f"{r['bb_q']:.2f} | {r['n']} | {dn_s} | {r['wr']*100:.1f}% | "
        f"{r['pf']:.3f} | {r['b50']:.3f} | {r['mc_p']*100:.1f}% | "
        f"{r['sym_floor']:.3f} | {r['fold_floor']:.3f} | "
        f"{r['score']}/7 | **{r['verdict']}** |"
    )

md += [
    "\n## Promote Criteria\n",
    "| V | PF>1.20 | n≥100 | Bt_p50 | MC>60% | LOO-S>1 | LOO-F>1 | MDD<25% | Score | Verdict |",
    "|---|---------|-------|--------|--------|---------|---------|---------|-------|---------|",
]
for vid in ranked:
    r  = results[vid]
    md.append(
        f"| {vid} | {ck(r['pf']>1.20)} | {ck(r['n']>=100)} | "
        f"{ck(r['b50']>1.20)} | {ck(r['mc_p']>0.60)} | "
        f"{ck(r['sym_floor']>1.0)} | {ck(r['fold_floor']>1.0)} | "
        f"{ck(abs(r['mdd'])<0.25)} | {r['score']}/7 | **{r['verdict']}** |"
    )

md += [
    "\n## Research Questions\n",
    f"**Q1. Biggest frequency increase:** Var {q1_best['vid']} — {q1_best['gate']} ({q1_best['relax']})  ",
    f"Δn=+{q1_best['dn']} (+{q1_best['npct']:.0f}%)  n_after={results[q1_best['vid']]['n']}\n",
    f"**Q2. Best PF preserved:** Var {q2_best['vid']} — {q2_best['gate']} ({q2_best['relax']})  ",
    f"PF={q2_best['pf']:.3f}  ΔPF={q2_best['dpf']:+.3f}\n",
    f"**Q3. Most essential gate:** {q3_gate['gate']}  ",
    f"Relaxing drops PF from {baseline_pf:.3f} to {q3_gate['pf']:.3f} (ΔPF={q3_gate['dpf']:+.3f})\n",
    f"**Q4. n≥100 & PF>1.20?** {'YES: '+', '.join(v+' (n='+str(results[v]['n'])+' PF='+str(round(results[v]['pf'],3))+')' for v,_ in q4_candidates) if q4_candidates else 'NO'}\n",
    f"**Q5. New production environment?** {q5_text}\n",
    "\n## Gate Importance Summary\n",
    "| Gate | Relaxation | n_after | Δn | ΔPF | Importance |",
    "|------|-----------|---------|-----|-----|------------|",
]
for g in sorted(gate_importance, key=lambda x: x["dpf"]):
    imp = "ESSENTIAL" if g["dpf"] < -0.3 else ("HIGH" if g["dpf"] < -0.1 else "MEDIUM")
    md.append(f"| {g['gate']} | {g['relax']} | {g['n']} | "
              f"+{g['dn']} | {g['dpf']:+.3f} | {imp} |")

md += [
    "\n## Key Finding\n",
    f"The RELVOL entry inside the R037 BASELINE environment produces PF={baseline_pf:.3f}",
    f"with n={baseline_n}. This study identifies which gate relaxations can lift n≥100",
    f"while maintaining PF>1.20.",
    "",
    f"**Overall recommendation:** {q5_text}",
    f"\n---\n*Generated by QUANTLAB AI R039 — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC*",
]

jmd_path = f"{OUT}/r039_journal.md"
with open(jmd_path, "w") as f:
    f.write("\n".join(md) + "\n")
print(f"  → {jmd_path}")

# =============================================================================
# JOURNAL CSV
# =============================================================================

journal_path = CONFIG["JOURNAL_FILE"]
for vid in VAR_IDS:
    r  = results[vid]
    row = {
        "research_id": f"{RESEARCH_ID}-{vid}",
        "date":        pd.Timestamp.now().strftime("%Y-%m-%d"),
        "strategy":    f"ENV_EXPAND_VAR{vid}_RELVOL",
        "timeframe":   "1H",
        "symbols":     str(len(SYMBOLS)),
        "method":      f"env-expand-var{vid}-5fold-WF",
        "n_oos":       r["n"],
        "wr":          round(r["wr"], 4),
        "pf":          round(r["pf"], 4),
        "sharpe":      round(r["sharpe"], 4),
        "mdd":         round(r["mdd"], 4),
        "net":         round(r["net"], 2),
        "boot_p50":    round(r["b50"], 4),
        "mc_prob":     round(r["mc_p"], 4),
        "loo_floor":   round(r["sym_floor"], 4),
        "verdict":     r["verdict"],
    }
    jdf = pd.read_csv(journal_path) if os.path.exists(journal_path) else pd.DataFrame()
    jdf = pd.concat([jdf, pd.DataFrame([row])], ignore_index=True)
    jdf.to_csv(journal_path, index=False)
print(f"  → Journal: {journal_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "═"*105)
print(f"  R039 COMPLETE — Environment Expansion Study (RELVOL fixed entry)")
print("═"*105)
print(f"  Dataset : {len(SYMBOLS)} symbols · 5-fold WF · 2R fixed")
print(f"  Baseline: Var A  n={baseline_n}  PF={baseline_pf:.3f}")
print()
print(f"  {'V':>2}  {'Label':25s}  {'n':>5}  {'Δn':>5}  {'PF':>7}  "
      f"{'p50':>7}  {'MC%':>6}  {'LOO-S':>6}  {'LOO-F':>6}  {'Score':>5}  Verdict")
print("  " + "─"*97)
for vid in ranked:
    r   = results[vid]
    dn  = r["n"] - baseline_n
    dn_s = f"+{dn}" if dn >= 0 else str(dn)
    mark = "★" if r["verdict"]=="PROMOTE" else ("→" if r["verdict"]=="WATCHLIST" else " ")
    print(f"  {mark}{vid:>2}  {r['label']:25s}  {r['n']:5d}  {dn_s:>5}  "
          f"{r['pf']:7.3f}  {r['b50']:7.3f}  {r['mc_p']*100:5.1f}%  "
          f"{r['sym_floor']:6.3f}  {r['fold_floor']:6.3f}  "
          f"{r['score']:3d}/7  {r['verdict']}")

print()
print(f"  Q1: Biggest freq gain  → Var {q1_best['vid']} ({q1_best['gate']} {q1_best['relax']}): "
      f"Δn=+{q1_best['dn']}")
print(f"  Q2: Best PF preserved  → Var {q2_best['vid']} ({q2_best['gate']}): PF={q2_best['pf']:.3f}")
print(f"  Q3: Essential gate     → {q3_gate['gate']}  (ΔPF={q3_gate['dpf']:+.3f} when relaxed)")
print(f"  Q4: n≥100 & PF>1.20?  → "
      f"{'YES: '+', '.join(v for v,_ in q4_candidates) if q4_candidates else 'NO'}")
print(f"  Q5: New environment    → {q5_text}")
print(f"""
  Output:
    {OUT}/r039_dashboard.png
    {OUT}/r039_variant_table.csv
    {OUT}/r039_equity_curves.png
    {OUT}/r039_tradeoff_chart.png
    {OUT}/r039_heatmap.png
    {OUT}/r039_journal.md
""" + "═"*105)
