"""
=============================================================================
QUANTLAB AI — RESEARCH #041
Environment Expansion Optimisation (1H, RELVOL fixed)
=============================================================================

Background:
  R039 Baseline (Var A): ATR<p25 · slope>0 · EMA_dist>p75 · BB<p33
    PF=1.817  n=22  — WATCHLIST (high quality, too few trades)
  R039 Var D (best):     ATR<p25 · slope>0 · EMA_dist>p75 · BB<p50
    PF=2.220  n=27  — WATCHLIST (best quality, still too few trades)
  R040: 15m transfer → REJECT (edge timeframe-dependent)

Objective:
  Systematically relax ONE or TWO conditions at a time on 1H to find the
  Pareto-optimal environment: highest PF while achieving n ≥ 100.

LOCKED (unchanged):
  • Entry:  RELVOL Breakout (vol > 1.5×avg, bullish candle)
  • Exit:   Stop=1×ATR14  Target=2×ATR14  (2R fixed)
  • Timeframe: 1H
  • Fees, slippage, risk model

BASELINE: ATR<p25 · slope>0 · EMA_dist>p75 · BB<p33

Variants (relaxing ONE or TWO conditions):
  A  EMA Distance > p60                          (loosen dist only)
  B  EMA Distance > p50                          (loosen dist more)
  C  BB Width < p50                              (loosen BB only) [= R039 Var D]
  D  EMA Distance > p60 + BB Width < p50         (loosen dist + BB)
  E  ATR Rank < p33                              (loosen ATR only)
  F  ATR Rank < p40                              (loosen ATR more)
  G  ATR Rank < p33 + BB Width < p50             (loosen ATR + BB)
  H  ATR Rank < p40 + BB Width < p50             (loosen ATR + BB more)
  I  EMA Distance > p60 + ATR Rank < p33         (loosen dist + ATR)

Additional Pareto fillers (two-condition relaxations):
  J  EMA Distance > p50 + BB Width < p50         (dist+BB further)
  K  ATR Rank < p33 + EMA Dist > p60             (already = I, skip)

Promote: PF>1.20 · n≥100 · Boot_p50>1.20 · MC_P>60%
         LOO-sym>1.00 · LOO-fold>1.00 · MDD<25%
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

RESEARCH_ID = "R041"
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

# ── VARIANT DEFINITIONS ──────────────────────────────────────────────────────
# (id, label, atr_q, dist_q, bb_q)
# EMA200 slope > 0 is enforced in ALL variants — never relaxed
VARIANTS = [
    # id    label                          atr_q  dist_q  bb_q
    ("BASE","BASELINE",                    0.25,  0.75,   0.33),
    ("A",   "Relax Dist→p60",              0.25,  0.60,   0.33),
    ("B",   "Relax Dist→p50",              0.25,  0.50,   0.33),
    ("C",   "Relax BB→p50",                0.25,  0.75,   0.50),
    ("D",   "Dist→p60 + BB→p50",           0.25,  0.60,   0.50),
    ("E",   "Relax ATR→p33",               0.33,  0.75,   0.33),
    ("F",   "Relax ATR→p40",               0.40,  0.75,   0.33),
    ("G",   "ATR→p33 + BB→p50",            0.33,  0.75,   0.50),
    ("H",   "ATR→p40 + BB→p50",            0.40,  0.75,   0.50),
    ("I",   "ATR→p33 + Dist→p60",          0.33,  0.60,   0.33),
    ("J",   "Dist→p50 + BB→p50",           0.25,  0.50,   0.50),
]
VAR_IDS   = [v[0] for v in VARIANTS]
VAR_NAMES = {v[0]: v[1] for v in VARIANTS}

# Gate changed vs BASELINE (for labelling)
def gate_changes(atr_q, dist_q, bb_q):
    changes = []
    if atr_q  != 0.25: changes.append(f"ATR<p{atr_q*100:.0f}")
    if dist_q != 0.75: changes.append(f"Dist>p{dist_q*100:.0f}")
    if bb_q   != 0.33: changes.append(f"BB<p{bb_q*100:.0f}")
    return ", ".join(changes) if changes else "—"

COLOURS_BASE = {
    "BASE":"#888888",
    "A":"#2196F3","B":"#0D47A1",
    "C":"#4CAF50",
    "D":"#00BCD4",
    "E":"#FF9800","F":"#E65100",
    "G":"#9C27B0","H":"#6A1B9A",
    "I":"#E91E63","J":"#880E4F",
}

BG    = "#0d1117"
PANEL = "#161b22"
TXT   = "#e0e0e0"
AMBER = "#f0c040"
GREEN = "#2ea043"
RED   = "#cf222e"

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

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #041" + " "*50 + "║")
print("║  Environment Expansion Optimisation (1H, RELVOL)" + " "*29 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Baseline: ATR<p25 · EMA200_slope>0 · EMA_dist>p75 · BB<p33
  Entry: RELVOL Breakout  |  Exit: Stop=1×ATR14  Target=2×ATR14  (2R fixed)
  Method: 5-fold expanding WF · IS thresholds only
  Variants: {len(VARIANTS)} (baseline + {len(VARIANTS)-1} relaxations)
  Goal: PF>1.20 · n≥100  (PROMOTE criteria)
""")

# =============================================================================
# INDICATORS  (identical to R039)
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
# THRESHOLDS  (per IS period, per variant)
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
# BACKTEST ENGINE  (identical to R039)
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
                xp    = (st * (1 - slp)) if sl_hit else tk
                xt    = "SL" if sl_hit else "TP"
                sd    = ep - st
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
            if pd.isna(atr_) or atr_ <= 0: continue
            sd = atr_
            if sd / bar["open"] < min_sl: continue
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
print(f"  {len(SYMBOLS)} symbols  ({sum(len(d) for d in all_dfs.values()):,} bars)\n")

# =============================================================================
# WALK-FORWARD  (all variants in one pass)
# =============================================================================

var_sym_trades = {v[0]: {sym: [] for sym in SYMBOLS} for v in VARIANTS}
var_env_bars   = {v[0]: 0 for v in VARIANTS}
fold_pf_tbl    = {v[0]: [] for v in VARIANTS}
fold_n_tbl     = {v[0]: [] for v in VARIANTS}

print(f"  Walk-forward: {len(FOLDS)} folds × {len(SYMBOLS)} symbols × "
      f"{len(VARIANTS)} variants  (entry: RELVOL fixed)\n")

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_t = {v[0]: [] for v in VARIANTS}
    fold_e = {v[0]: 0  for v in VARIANTS}

    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[:int(N*is_end)]
        df_oos = df_full.iloc[int(N*is_end):int(N*oos_end)].reset_index(drop=True)
        if len(df_oos) < 100: continue

        for vid, vlabel, atr_q, dist_q, bb_q in VARIANTS:
            thr = learn_thresholds(df_is, atr_q, dist_q, bb_q)
            env = in_environment(df_oos, thr)
            fold_e[vid] += int(env.sum())
            sig = signal_relvol(df_oos, env)
            tl  = run_backtest(df_oos, sig, sym, fold_idx, vid)
            var_sym_trades[vid][sym].extend(tl)
            fold_t[vid].extend(tl)

    print(f"  Fold {fold_idx}  (IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%)")
    for vid, vlabel, *_ in VARIANTS:
        m = metrics(fold_t[vid])
        fold_pf_tbl[vid].append(m["pf"])
        fold_n_tbl[vid].append(m["n"])
        var_env_bars[vid] += fold_e[vid]
        print(f"    {vid:4s} {vlabel:30s}  env={fold_e[vid]:5,}  "
              f"n={m['n']:4d}  PF={m['pf']:.3f}")
    print()

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print("─"*78)
print("  Computing aggregate statistics …")
results = {}

for vid, vlabel, atr_q, dist_q, bb_q in VARIANTS:
    all_flat  = [t for sym in SYMBOLS for t in var_sym_trades[vid][sym]]
    m         = metrics(all_flat)
    b5,b50,b95 = bootstrap_pf(m["pnls"])
    mc         = monte_carlo(m["pnls"])
    ls         = loo_sym(var_sym_trades[vid])
    lf         = loo_fld(all_flat)
    sf         = min(ls.values()) if ls else 0.0
    ff         = min(lf.values()) if lf else 0.0

    score = sum([
        m["pf"]           > TARGET_PF,
        m["n"]            >= TARGET_N,
        b50               > TARGET_BOOT,
        mc["prob_profit"] > TARGET_MC,
        sf > 1.0,
        ff > 1.0,
        abs(m["mdd"])     < 0.25,
    ])

    changes = gate_changes(atr_q, dist_q, bb_q)
    results[vid] = {
        "label": vlabel, "atr_q": atr_q, "dist_q": dist_q, "bb_q": bb_q,
        "changes": changes,
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

def get_verdict(r):
    sc = r["score"]
    if sc == 7:                        return "PROMOTE"
    elif sc >= 5 and r["pf"] > 1.0:   return "WATCHLIST"
    elif sc >= 3:                      return "INVESTIGATE"
    else:                              return "REJECT"

for vid in VAR_IDS:
    results[vid]["verdict"] = get_verdict(results[vid])

baseline_n  = results["BASE"]["n"]
baseline_pf = results["BASE"]["pf"]

# Sort: score desc, then PF desc
ranked = sorted(VAR_IDS, key=lambda v: (-results[v]["score"], -results[v]["pf"]))
promote_list  = [v for v in VAR_IDS if results[v]["verdict"] == "PROMOTE"]
watchlist     = [v for v in VAR_IDS if results[v]["verdict"] == "WATCHLIST"]

# =============================================================================
# RESULTS TABLE
# =============================================================================

print("\n" + "═"*115)
print("  R041 — ENVIRONMENT EXPANSION OPTIMISATION  (RELVOL fixed, 1H)")
print("═"*115)
print(f"\n  Baseline: PF={baseline_pf:.3f}  n={baseline_n}\n")
print(f"  {'':1s}{'V':>4}  {'Label':30s}  {'Changes':22s}  "
      f"{'n':>5}  {'Δn':>6}  {'WR':>6}  {'PF':>7}  {'p50':>7}  "
      f"{'MC%':>6}  {'LOO-S':>6}  {'LOO-F':>6}  {'MDD':>7}  Score  Verdict")
print("  " + "─"*120)

for vid in ranked:
    r   = results[vid]
    dn  = r["n"] - baseline_n
    dn_s = f"+{dn}" if dn >= 0 else str(dn)
    flag = "★" if r["score"] == 7 else ("↑" if r["pf"] > TARGET_PF else " ")
    print(f"  {flag}{vid:>4}  {r['label']:30s}  {r['changes']:22s}  "
          f"{r['n']:5d}  {dn_s:>6}  {r['wr']*100:5.1f}%  {r['pf']:7.3f}  "
          f"{r['b50']:7.3f}  {r['mc_p']*100:5.1f}%  {r['sym_floor']:6.3f}  "
          f"{r['fold_floor']:6.3f}  {r['mdd']:6.1%}  {r['score']:3d}/7  {r['verdict']}")

# =============================================================================
# RESEARCH QUESTIONS
# =============================================================================

print("\n" + "═"*115)
print("  RESEARCH QUESTIONS")
print("═"*115)

# Q1: Which relaxation increases trades most (single-condition variants only)?
single_variants = ["A","B","C","E","F"]
q1_best = max(single_variants, key=lambda v: results[v]["n"])
q1_r = results[q1_best]

# Q2: Which relaxation preserves PF best (among single-condition variants with n>baseline)?
q2_candidates = [(v, results[v]) for v in single_variants
                 if results[v]["n"] > baseline_n and results[v]["pf"] > 0]
q2_best_vid = max(q2_candidates, key=lambda x: x[1]["pf"])[0] if q2_candidates else q1_best
q2_r = results[q2_best_vid]

# Q3: n>100 AND PF>1.20?
q3_hit = [(v, results[v]) for v in VAR_IDS
          if results[v]["n"] >= TARGET_N and results[v]["pf"] > TARGET_PF]
q3_n_only = [(v, results[v]) for v in VAR_IDS if results[v]["n"] >= TARGET_N]

# Q4: Pareto-optimal — score based trade-off
# Pareto: a variant is dominated if there exists another with higher PF and higher n
def is_pareto_dominated(vid, all_vids):
    r = results[vid]
    return any(results[v]["pf"] >= r["pf"] and results[v]["n"] >= r["n"]
               and (results[v]["pf"] > r["pf"] or results[v]["n"] > r["n"])
               for v in all_vids if v != vid)

pareto_front = [v for v in VAR_IDS if not is_pareto_dominated(v, VAR_IDS)]
pareto_front_sorted = sorted(pareto_front, key=lambda v: results[v]["n"])

# Q5: Best single production environment
# Prioritise: score == 7 (PROMOTE), then WATCHLIST with PF>1.20, then highest score+PF
if promote_list:
    q5_rec = max(promote_list, key=lambda v: (results[v]["score"], results[v]["pf"]))
    q5_text = f"PROMOTE → {q5_rec} ({results[q5_rec]['label']})"
elif watchlist:
    q5_cands = [(v, results[v]) for v in watchlist if results[v]["pf"] > TARGET_PF]
    if q5_cands:
        q5_rec  = max(q5_cands, key=lambda x: (x[1]["score"], x[1]["pf"]))[0]
    else:
        q5_rec  = max(watchlist, key=lambda v: (results[v]["score"], results[v]["pf"]))
    q5_text = f"WATCHLIST → {q5_rec} ({results[q5_rec]['label']})"
else:
    q5_rec  = ranked[0]
    q5_text = f"INVESTIGATE → {q5_rec} ({results[q5_rec]['label']})"

q5_r = results[q5_rec]

print(f"""
  Q1. Which single relaxation increases trades the most?
      → Variant {q1_best}: {q1_r['label']}  ({q1_r['changes']})
        n={q1_r['n']}  Δn=+{q1_r['n']-baseline_n}  (+{(q1_r['n']-baseline_n)/max(baseline_n,1)*100:.0f}%)
        PF={q1_r['pf']:.3f}  (baseline PF={baseline_pf:.3f})

  Q2. Which single relaxation preserves PF best while adding trades?
      → Variant {q2_best_vid}: {q2_r['label']}  ({q2_r['changes']})
        PF={q2_r['pf']:.3f}  n={q2_r['n']}  Δn=+{q2_r['n']-baseline_n}  Boot_p50={q2_r['b50']:.3f}

  Q3. Can any relaxed environment reach PF>1.20 AND n≥100?""")

if q3_hit:
    for v, r in sorted(q3_hit, key=lambda x: -x[1]["score"]):
        print(f"      YES ★  Variant {v} ({r['label']}): "
              f"n={r['n']}  PF={r['pf']:.3f}  p50={r['b50']:.3f}  MC={r['mc_p']*100:.1f}%  "
              f"LOO-S={r['sym_floor']:.3f}  LOO-F={r['fold_floor']:.3f}  Score={r['score']}/7")
elif q3_n_only:
    print(f"      NOT YET — n≥100 achieved but PF below 1.20:")
    for v, r in q3_n_only:
        print(f"        Variant {v} ({r['label']}): n={r['n']}  PF={r['pf']:.3f}")
else:
    best_n_vid = max(VAR_IDS, key=lambda v: results[v]["n"])
    print(f"      NO — Best n={results[best_n_vid]['n']} in Variant {best_n_vid}")

print(f"""
  Q4. Pareto-optimal environments (n vs PF front):""")
for v in pareto_front_sorted:
    r   = results[v]
    dn_ = r["n"] - baseline_n
    print(f"      Var {v:4s} {r['label']:30s}  n={r['n']:4d}  "
          f"PF={r['pf']:.3f}  score={r['score']}/7  {r['verdict']}")

print(f"""
  Q5. Recommended production environment:
      → {q5_text}
        n={q5_r['n']}  WR={q5_r['wr']*100:.1f}%  PF={q5_r['pf']:.3f}  "
        Boot_p50={q5_r['b50']:.3f}  MC={q5_r['mc_p']*100:.1f}%  "
        LOO-S={q5_r['sym_floor']:.3f}  LOO-F={q5_r['fold_floor']:.3f}  "
        MDD={q5_r['mdd']:.1%}  Score={q5_r['score']}/7
""")

# =============================================================================
# CHARTS
# =============================================================================

print("─"*78)
print("  Generating charts …")

def _style(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=TXT, fontsize=8, pad=4)

# ── 1: Pareto scatter (n vs PF) with Pareto front highlighted ────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor=BG)
fig.suptitle("R041 — Environment Expansion: Pareto Frontier (n vs PF)",
             color=TXT, fontsize=11)

ax_ = axes[0]
_style(ax_, "n vs PF — All Variants")
for vid in VAR_IDS:
    r = results[vid]
    on_front = vid in pareto_front
    ec = "white" if on_front else "none"
    ms = 200 if on_front else 130
    ax_.scatter(r["n"], r["pf"], s=ms, color=COLOURS_BASE[vid],
                edgecolors=ec, linewidths=1.5, zorder=5)
    ax_.annotate(f"{vid}\n{r['label'][:10]}", (r["n"], r["pf"]),
                 xytext=(5, 3), textcoords="offset points",
                 color=COLOURS_BASE[vid], fontsize=6)

# Draw Pareto front line
px = [results[v]["n"]  for v in pareto_front_sorted]
py = [results[v]["pf"] for v in pareto_front_sorted]
ax_.plot(px, py, color=AMBER, lw=1.2, ls="--", alpha=0.7, label="Pareto front")
ax_.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":", label="PF=1.20")
ax_.axvline(TARGET_N,  color=AMBER, lw=0.8, ls="--", label="n=100")
ax_.set_xlabel("OOS Trade Count", color="#888", fontsize=9)
ax_.set_ylabel("Profit Factor",   color="#888", fontsize=9)
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# ── PF bar ────────────────────────────────────────────────────────────────────
ax_ = axes[1]
_style(ax_, "Profit Factor by Variant")
x_ = np.arange(len(VAR_IDS))
bars_ = ax_.bar(x_, [results[v]["pf"] for v in VAR_IDS],
                color=[COLOURS_BASE[v] for v in VAR_IDS], alpha=0.85)
ax_.axhline(TARGET_PF, color=AMBER, lw=0.9, ls=":")
ax_.axhline(baseline_pf, color="#888", lw=0.8, ls="--",
            label=f"Baseline={baseline_pf:.3f}")
ax_.set_xticks(x_)
ax_.set_xticklabels([f"{v}\n{results[v]['label'][:9]}" for v in VAR_IDS],
                     rotation=30, ha="right", color=TXT, fontsize=6)
ax_.set_ylabel("PF", color="#888", fontsize=8)
for b, v in zip(bars_, VAR_IDS):
    pf_ = results[v]["pf"]
    ax_.text(b.get_x()+b.get_width()/2, pf_+0.02,
             f"{pf_:.3f}", ha="center", color=TXT, fontsize=6, fontweight="bold")
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# ── n bar ─────────────────────────────────────────────────────────────────────
ax_ = axes[2]
_style(ax_, "Trade Count by Variant")
bars_ = ax_.bar(x_, [results[v]["n"] for v in VAR_IDS],
                color=[COLOURS_BASE[v] for v in VAR_IDS], alpha=0.85)
ax_.axhline(TARGET_N,   color=AMBER, lw=0.9, ls=":", label="n=100")
ax_.axhline(baseline_n, color="#888", lw=0.8, ls="--",
            label=f"Baseline={baseline_n}")
ax_.set_xticks(x_)
ax_.set_xticklabels([f"{v}\n{results[v]['label'][:9]}" for v in VAR_IDS],
                     rotation=30, ha="right", color=TXT, fontsize=6)
ax_.set_ylabel("n trades", color="#888", fontsize=8)
for b, v in zip(bars_, VAR_IDS):
    n_ = results[v]["n"]
    ax_.text(b.get_x()+b.get_width()/2, n_+0.3,
             str(n_), ha="center", color=TXT, fontsize=6, fontweight="bold")
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

plt.tight_layout()
p = f"{OUT}/r041_pareto.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 2: Equity Curves ─────────────────────────────────────────────────────────
n_cols = 4; n_rows = math.ceil(len(VAR_IDS) / n_cols)
fig, axes2 = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows), facecolor=BG)
fig.suptitle("R041 — OOS Equity Curves per Variant (RELVOL, 1H)",
             color=TXT, fontsize=11)
axes_flat = list(axes2.flat)
for ax_, vid in zip(axes_flat, VAR_IDS):
    r = results[vid]
    _style(ax_, f"Var {vid}: {r['label']}\n"
                f"PF={r['pf']:.3f}  n={r['n']}  {r['verdict']}")
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
p = f"{OUT}/r041_equity_curves.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 3: Heatmap ────────────────────────────────────────────────────────────────
metrics_keys   = ["n","pf","b50","mc_p","sym_floor","fold_floor","wr","mdd"]
metrics_labels = ["n","PF","Boot_p50","MC_P","LOO-sym","LOO-fold","WR","MDD"]
cmap_rb = LinearSegmentedColormap.from_list("rb", [RED, "#444", GREEN])

raw_data = np.array([
    [results[v]["n"],
     results[v]["pf"],
     results[v]["b50"],
     results[v]["mc_p"]*100,
     results[v]["sym_floor"],
     results[v]["fold_floor"],
     results[v]["wr"]*100,
     abs(results[v]["mdd"])*100]
    for v in VAR_IDS
], dtype=float)

# MDD: lower = better, invert for colormap
mdd_col = metrics_keys.index("mdd")
norm_data = np.zeros_like(raw_data)
for col in range(raw_data.shape[1]):
    col_vals = raw_data[:, col]
    vmin, vmax = col_vals.min(), col_vals.max()
    if vmax > vmin:
        norm_data[:, col] = (col_vals - vmin) / (vmax - vmin)
    else:
        norm_data[:, col] = 0.5
norm_data[:, mdd_col] = 1.0 - norm_data[:, mdd_col]  # invert MDD

fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
_style(ax, "R041 — Variant Metric Heatmap (column-normalized, green=better)")
im = ax.imshow(norm_data, aspect="auto", cmap=cmap_rb, vmin=0.0, vmax=1.0)
ax.set_xticks(range(len(metrics_labels)))
ax.set_xticklabels(metrics_labels, color=TXT, fontsize=9)
ax.set_yticks(range(len(VAR_IDS)))
ax.set_yticklabels([f"{v}: {results[v]['label']}" for v in VAR_IDS],
                    color=TXT, fontsize=8)
for row, vid in enumerate(VAR_IDS):
    for col in range(len(metrics_keys)):
        val = raw_data[row, col]
        fmt = ".0f" if metrics_keys[col] in ("n","mc_p","wr","mdd") else ".3f"
        ax.text(col, row, f"{val:{fmt}}", ha="center", va="center",
                color="white", fontsize=8, fontweight="bold")
plt.colorbar(im, ax=ax, label="Normalized (0=worst 1=best)", fraction=0.02)
plt.tight_layout()
p = f"{OUT}/r041_heatmap.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 4: Fold stability ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 7), facecolor=BG)
_style(ax, "R041 — PF by Fold — All Variants")
for vid in VAR_IDS:
    ax.plot(range(1, 6), fold_pf_tbl[vid], marker="o", ms=4, lw=1.3,
            color=COLOURS_BASE[vid], label=f"{vid}", alpha=0.9)
ax.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":")
ax.axhline(1.0, color="#555", lw=0.5, ls="--")
ax.set_xticks(range(1, 6))
ax.set_xlabel("Fold", color="#888", fontsize=9)
ax.set_ylabel("PF", color="#888", fontsize=9)
ax.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7, ncol=4)
plt.tight_layout()
p = f"{OUT}/r041_fold_stability.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 5: Bootstrap CI bars (all variants) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 7), facecolor=BG)
_style(ax, "R041 — Bootstrap PF  Median ± 90% CI")
x_ = np.arange(len(VAR_IDS))
ax.bar(x_, [results[v]["pf"] for v in VAR_IDS],
       color=[COLOURS_BASE[v] for v in VAR_IDS], alpha=0.30, width=0.6)
ax.errorbar(x_, [results[v]["b50"] for v in VAR_IDS],
            yerr=[[results[v]["b50"]-results[v]["b5"] for v in VAR_IDS],
                  [results[v]["b95"]-results[v]["b50"] for v in VAR_IDS]],
            fmt="o", color="white", capsize=6, ms=5, lw=1.5)
ax.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":", label="PF=1.20")
ax.set_xticks(x_)
ax.set_xticklabels([f"{v}\n{results[v]['label'][:10]}" for v in VAR_IDS],
                    rotation=30, ha="right", color=TXT, fontsize=7)
ax.set_ylabel("PF", color="#888", fontsize=9)
ax.legend(facecolor=PANEL, labelcolor=TXT, fontsize=8)
plt.tight_layout()
p = f"{OUT}/r041_bootstrap_ci.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 6: Dashboard ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(30, 24), facecolor=BG)
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.35,
                         top=0.93, bottom=0.04, left=0.04, right=0.98)

row0_items = [
    ("n", "Trade Count", TARGET_N, False),
    ("pf", "Profit Factor", TARGET_PF, False),
    ("b50", "Bootstrap p50", TARGET_BOOT, False),
    ("mc_p", "MC P(profit)", TARGET_MC, True),
]
for col, (key, title, tgt, pct) in enumerate(row0_items):
    ax_ = fig.add_subplot(gs[0, col])
    _style(ax_, title)
    vals_ = [results[v][key]*(100 if pct else 1) for v in VAR_IDS]
    tgt_  = tgt * (100 if pct else 1)
    bars_ = ax_.bar(range(len(VAR_IDS)), vals_,
                    color=[COLOURS_BASE[v] for v in VAR_IDS], alpha=0.85)
    ax_.axhline(tgt_, color=AMBER, lw=0.8, ls=":")
    ax_.set_xticks(range(len(VAR_IDS)))
    ax_.set_xticklabels([f"{v}" for v in VAR_IDS], color=TXT, fontsize=8)
    for b, val in zip(bars_, vals_):
        ax_.text(b.get_x()+b.get_width()/2, max(val,0)+abs(tgt_)*0.01,
                 f"{val:.1f}", ha="center", color=TXT, fontsize=6, fontweight="bold")

ax_sc = fig.add_subplot(gs[1, 0:2])
_style(ax_sc, "Pareto Frontier — n vs PF")
for vid in VAR_IDS:
    r = results[vid]
    on_front = vid in pareto_front
    ax_sc.scatter(r["n"], r["pf"], s=200 if on_front else 120,
                  color=COLOURS_BASE[vid],
                  edgecolors="white" if on_front else "none",
                  linewidths=1.5, zorder=5)
    ax_sc.annotate(f"{vid}", (r["n"], r["pf"]),
                   xytext=(5, 3), textcoords="offset points",
                   color=COLOURS_BASE[vid], fontsize=7)
if pareto_front_sorted:
    px = [results[v]["n"]  for v in pareto_front_sorted]
    py = [results[v]["pf"] for v in pareto_front_sorted]
    ax_sc.plot(px, py, color=AMBER, lw=1.2, ls="--", alpha=0.7, label="Pareto front")
ax_sc.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":")
ax_sc.axvline(TARGET_N,  color=AMBER, lw=0.8, ls="--")
ax_sc.set_xlabel("n trades", color="#888", fontsize=8)
ax_sc.set_ylabel("PF", color="#888", fontsize=8)
ax_sc.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

ax_fp = fig.add_subplot(gs[1, 2:4])
_style(ax_fp, "PF by Fold — All Variants")
for vid in VAR_IDS:
    ax_fp.plot(range(1, 6), fold_pf_tbl[vid], marker="o", ms=4, lw=1.3,
               color=COLOURS_BASE[vid], label=f"{vid}", alpha=0.9)
ax_fp.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":")
ax_fp.axhline(1.0, color="#555", lw=0.5, ls="--")
ax_fp.set_xticks(range(1, 6))
ax_fp.set_ylabel("PF", color="#888", fontsize=8)
ax_fp.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7, ncol=3)

ax_bc = fig.add_subplot(gs[2, 0:2])
_style(ax_bc, "Bootstrap PF  Median ± 90% CI")
x2_ = np.arange(len(VAR_IDS))
ax_bc.bar(x2_, [results[v]["pf"] for v in VAR_IDS],
          color=[COLOURS_BASE[v] for v in VAR_IDS], alpha=0.30, width=0.6)
ax_bc.errorbar(x2_, [results[v]["b50"] for v in VAR_IDS],
               yerr=[[results[v]["b50"]-results[v]["b5"] for v in VAR_IDS],
                     [results[v]["b95"]-results[v]["b50"] for v in VAR_IDS]],
               fmt="o", color="white", capsize=5, ms=4)
ax_bc.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":")
ax_bc.set_xticks(x2_)
ax_bc.set_xticklabels([f"{v}" for v in VAR_IDS], color=TXT, fontsize=8)
ax_bc.set_ylabel("PF", color="#888", fontsize=8)

ax_loo = fig.add_subplot(gs[2, 2:4])
_style(ax_loo, "LOO Floors (sym & fold)")
w = 0.38
x_loo = np.arange(len(VAR_IDS))
ax_loo.bar(x_loo-w/2, [results[v]["sym_floor"]  for v in VAR_IDS],
           w, color=[COLOURS_BASE[v] for v in VAR_IDS], alpha=0.85, label="LOO-sym")
ax_loo.bar(x_loo+w/2, [results[v]["fold_floor"] for v in VAR_IDS],
           w, color=[COLOURS_BASE[v] for v in VAR_IDS], alpha=0.45, label="LOO-fold")
ax_loo.axhline(1.0, color=AMBER, lw=0.8, ls=":")
ax_loo.set_xticks(x_loo)
ax_loo.set_xticklabels([f"{v}" for v in VAR_IDS], color=TXT, fontsize=8)
ax_loo.set_ylabel("LOO PF floor", color="#888", fontsize=8)
ax_loo.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# Row 3: Summary table + Q5 box
ax_tbl = fig.add_subplot(gs[3, 0:3])
ax_tbl.set_facecolor(PANEL)
for sp in ax_tbl.spines.values(): sp.set_visible(False)
ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
tbl_lines = [
    f"{'V':>4}  {'Label':28s}  {'Changes':22s}  "
    f"{'n':>5}  {'Δn':>5}  {'PF':>7}  {'p50':>7}  {'MC%':>6}  "
    f"{'LOO-S':>6}  {'LOO-F':>6}  {'MDD':>6}  {'Score':>5}  Verdict",
    "─"*118,
]
for vid in ranked:
    r  = results[vid]
    dn = r["n"] - baseline_n
    dn_s = f"+{dn}" if dn >= 0 else str(dn)
    tbl_lines.append(
        f"{vid:>4}  {r['label']:28s}  {r['changes']:22s}  "
        f"{r['n']:5d}  {dn_s:>5}  {r['pf']:7.3f}  {r['b50']:7.3f}  "
        f"{r['mc_p']*100:5.1f}%  {r['sym_floor']:6.3f}  {r['fold_floor']:6.3f}  "
        f"{r['mdd']:5.1%}  {r['score']:3d}/7  {r['verdict']}"
    )
ax_tbl.text(0.01, 0.97, "\n".join(tbl_lines),
            transform=ax_tbl.transAxes, color=TXT, fontsize=7,
            fontfamily="monospace", va="top")

ax_ans = fig.add_subplot(gs[3, 3])
ax_ans.set_facecolor(PANEL)
for sp in ax_ans.spines.values(): sp.set_visible(False)
ax_ans.set_xticks([]); ax_ans.set_yticks([])
ck = lambda c: "✓" if c else "✗"
ans_text = (
    f"RESEARCH ANSWERS\n\n"
    f"Q1 Biggest freq gain:\n"
    f"  Var {q1_best} ({q1_r['label'][:16]})\n"
    f"  n={q1_r['n']}  Δn=+{q1_r['n']-baseline_n}\n\n"
    f"Q2 Best PF preserved:\n"
    f"  Var {q2_best_vid} ({q2_r['label'][:16]})\n"
    f"  PF={q2_r['pf']:.3f}  n={q2_r['n']}\n\n"
    f"Q3 PF>1.20 & n≥100?\n"
    f"  {'YES: '+', '.join(v for v,_ in q3_hit) if q3_hit else 'NO'}\n\n"
    f"Q4 Pareto front:\n"
    f"  {', '.join(pareto_front_sorted)}\n\n"
    f"Q5 Recommendation:\n"
    f"  {q5_text[:30]}\n"
    f"  n={q5_r['n']} PF={q5_r['pf']:.3f}\n"
    f"  Score={q5_r['score']}/7"
)
ax_ans.text(0.05, 0.97, ans_text, transform=ax_ans.transAxes,
            color=TXT, fontsize=8, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#0d1117", edgecolor="#444"))

best_vid = ranked[0]
fig.suptitle(
    f"QUANTLAB AI — R041 | Environment Expansion Optimisation | RELVOL Fixed, 1H\n"
    f"{len(VARIANTS)} variants | 5-fold WF | {len(SYMBOLS)} symbols | "
    f"Best: Var {best_vid} ({results[best_vid]['label']}) "
    f"PF={results[best_vid]['pf']:.3f} n={results[best_vid]['n']} "
    f"— {results[best_vid]['verdict']}",
    color=TXT, fontsize=10, y=0.975
)
dash_path = f"{OUT}/r041_dashboard.png"
fig.savefig(dash_path, dpi=120, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"  → {dash_path}")

# ── 7: Per-symbol PF heatmap for best variant ─────────────────────────────────
best_scored = ranked[0]
sym_pf_base = {sym: metrics(var_sym_trades["BASE"][sym])["pf"] for sym in SYMBOLS}
sym_pf_best = {sym: metrics(var_sym_trades[best_scored][sym])["pf"] for sym in SYMBOLS}
syms_sorted = sorted(SYMBOLS, key=lambda s: -sym_pf_best[s])

fig, axes = plt.subplots(1, 2, figsize=(22, 8), facecolor=BG)
fig.suptitle(f"R041 — Per-Symbol Performance: Baseline vs Var {best_scored} (best)",
             color=TXT, fontsize=10)
for ax_, (label, sym_dict) in zip(axes, [("BASELINE", sym_pf_base), (f"Var {best_scored}", sym_pf_best)]):
    _style(ax_, label)
    pfs_ = [sym_dict.get(s, 0) for s in syms_sorted]
    clrs_ = [GREEN if p > 1.20 else (AMBER if p > 1.0 else RED) for p in pfs_]
    ax_.bar(range(len(syms_sorted)), pfs_, color=clrs_, alpha=0.85)
    ax_.axhline(1.20, color=AMBER, lw=0.8, ls=":", label="PF=1.20")
    ax_.axhline(1.00, color=RED,   lw=0.6, ls="--")
    ax_.set_xticks(range(len(syms_sorted)))
    ax_.set_xticklabels([s.replace("-USDT-SWAP","") for s in syms_sorted],
                         rotation=45, ha="right", color=TXT, fontsize=7)
    ax_.set_ylabel("PF", color="#888", fontsize=8)
    ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)
plt.tight_layout()
p = f"{OUT}/r041_per_symbol_pf.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# =============================================================================
# CSV OUTPUT
# =============================================================================

csv_rows = []
for vid in ranked:
    r  = results[vid]
    dn = r["n"] - baseline_n
    csv_rows.append({
        "variant":          vid,
        "label":            r["label"],
        "changes":          r["changes"],
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
        "on_pareto_front":  int(vid in pareto_front),
        "criteria_score":   r["score"],
        "verdict":          r["verdict"],
    })
csv_path = f"{OUT}/r041_variant_table.csv"
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"  → {csv_path}")

# =============================================================================
# JOURNAL MARKDOWN
# =============================================================================

ck2 = lambda c: "✓" if c else "✗"
md = [
    "# QUANTLAB AI — R041: Environment Expansion Optimisation\n",
    f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
    f"**Entry:** RELVOL Breakout (locked)  ",
    f"**Exit:** Stop=1×ATR14 · Target=2×ATR14 (2R fixed)  ",
    f"**Timeframe:** 1H  ",
    f"**Method:** 5-fold expanding WF · IS thresholds only  ",
    f"**Universe:** {len(SYMBOLS)} symbols  ",
    f"**Baseline:** ATR<p25 · EMA200_slope>0 · EMA_dist>p75 · BB<p33 → n={baseline_n} PF={baseline_pf:.3f}  \n",
    "## Variant Results\n",
    "| V | Label | Changes | n | Δn | WR | PF | p50 | MC% | LOO-S | LOO-F | MDD | Score | Verdict | Pareto? |",
    "|---|-------|---------|---|-----|----|----|-----|-----|-------|-------|-----|-------|---------|---------|",
]
for vid in ranked:
    r  = results[vid]
    dn = r["n"] - baseline_n
    dn_s = f"+{dn}" if dn >= 0 else str(dn)
    on_p = "★" if vid in pareto_front else ""
    md.append(
        f"| {vid} | {r['label']} | {r['changes']} | {r['n']} | {dn_s} | "
        f"{r['wr']*100:.1f}% | {r['pf']:.3f} | {r['b50']:.3f} | "
        f"{r['mc_p']*100:.1f}% | {r['sym_floor']:.3f} | {r['fold_floor']:.3f} | "
        f"{r['mdd']:.1%} | {r['score']}/7 | **{r['verdict']}** | {on_p} |"
    )

md += [
    "\n## Promotion Criteria\n",
    "| V | PF>1.20 | n≥100 | Bt_p50 | MC>60% | LOO-S>1 | LOO-F>1 | MDD<25% | Score |",
    "|---|---------|-------|--------|--------|---------|---------|---------|-------|",
]
for vid in ranked:
    r  = results[vid]
    md.append(
        f"| {vid} | {ck2(r['pf']>1.20)} | {ck2(r['n']>=100)} | "
        f"{ck2(r['b50']>1.20)} | {ck2(r['mc_p']>0.60)} | "
        f"{ck2(r['sym_floor']>1.0)} | {ck2(r['fold_floor']>1.0)} | "
        f"{ck2(abs(r['mdd'])<0.25)} | {r['score']}/7 |"
    )

md += [
    "\n## Research Questions\n",
    f"**Q1. Biggest frequency increase (single-condition):** Var {q1_best} ({q1_r['label']})  ",
    f"n={q1_r['n']}  Δn=+{q1_r['n']-baseline_n}  (+{(q1_r['n']-baseline_n)/max(baseline_n,1)*100:.0f}%)  "
    f"PF={q1_r['pf']:.3f}\n",
    f"**Q2. Best PF preserved while adding trades:** Var {q2_best_vid} ({q2_r['label']})  ",
    f"PF={q2_r['pf']:.3f}  n={q2_r['n']}  Boot_p50={q2_r['b50']:.3f}\n",
    f"**Q3. n≥100 AND PF>1.20?**  ",
    ("YES: " + ", ".join(f"{v} (n={r['n']}, PF={r['pf']:.3f})" for v,r in q3_hit) + "\n"
     if q3_hit else "NO — no variant achieves both simultaneously in this run.\n"),
    "\n**Q4. Pareto-optimal front (n vs PF):**\n",
    "| V | Label | n | PF | Score | Verdict |",
    "|---|-------|---|----|-------|---------|",
]
for v in pareto_front_sorted:
    r = results[v]
    md.append(f"| {v} | {r['label']} | {r['n']} | {r['pf']:.3f} | {r['score']}/7 | **{r['verdict']}** |")

md += [
    f"\n**Q5. Recommended production environment:** {q5_text}  ",
    f"n={q5_r['n']}  PF={q5_r['pf']:.3f}  Boot_p50={q5_r['b50']:.3f}  "
    f"MC={q5_r['mc_p']*100:.1f}%  LOO-S={q5_r['sym_floor']:.3f}  "
    f"LOO-F={q5_r['fold_floor']:.3f}  MDD={q5_r['mdd']:.1%}  Score={q5_r['score']}/7\n",
    "\n## Key Findings\n",
    f"- Baseline: PF={baseline_pf:.3f} n={baseline_n}",
    f"- Best trade count: Var {q1_best} (n={q1_r['n']}, PF={q1_r['pf']:.3f})",
    f"- Best PF preserved: Var {q2_best_vid} (PF={q2_r['pf']:.3f}, n={q2_r['n']})",
    f"- Q3 result: {'ACHIEVED — ' + str(len(q3_hit)) + ' variant(s) qualify' if q3_hit else 'NOT achieved — no single variant hits both PF>1.20 AND n≥100'}",
    f"- Pareto front: {', '.join(pareto_front_sorted)}",
    f"\n---\n*Generated by QUANTLAB AI R041 — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC*",
]

jmd_path = f"{OUT}/r041_journal.md"
with open(jmd_path, "w") as f:
    f.write("\n".join(md) + "\n")
print(f"  → {jmd_path}")

# =============================================================================
# JOURNAL CSV
# =============================================================================

journal_path = CONFIG["JOURNAL_FILE"]
for vid in VAR_IDS:
    r   = results[vid]
    row = {
        "research_id": f"{RESEARCH_ID}-{vid}",
        "date":        pd.Timestamp.now().strftime("%Y-%m-%d"),
        "strategy":    f"ENV_OPT_VAR{vid}_RELVOL",
        "timeframe":   "1H",
        "symbols":     str(len(SYMBOLS)),
        "method":      f"env-opt-var{vid}-5fold-WF",
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

print("\n" + "═"*115)
print(f"  R041 COMPLETE — Environment Expansion Optimisation (RELVOL fixed, 1H)")
print("═"*115)
print(f"  Dataset : {len(SYMBOLS)} symbols · 1H · 5-fold WF · 2R fixed")
print(f"  Baseline: n={baseline_n}  PF={baseline_pf:.3f}")
print()
print(f"  {'':1s}{'V':>4}  {'Label':30s}  {'Changes':22s}  "
      f"{'n':>5}  {'Δn':>6}  {'PF':>7}  {'p50':>7}  "
      f"{'MC%':>6}  {'LOO-S':>6}  {'LOO-F':>6}  {'Score':>5}  Verdict  Pareto?")
print("  " + "─"*125)
for vid in ranked:
    r    = results[vid]
    dn   = r["n"] - baseline_n
    dn_s = f"+{dn}" if dn >= 0 else str(dn)
    mark = "★" if r["verdict"] == "PROMOTE" else ("→" if r["verdict"] == "WATCHLIST" else " ")
    pareto_m = "★" if vid in pareto_front else ""
    print(f"  {mark}{vid:>4}  {r['label']:30s}  {r['changes']:22s}  "
          f"{r['n']:5d}  {dn_s:>6}  {r['pf']:7.3f}  {r['b50']:7.3f}  "
          f"{r['mc_p']*100:5.1f}%  {r['sym_floor']:6.3f}  {r['fold_floor']:6.3f}  "
          f"{r['score']:3d}/7  {r['verdict']:12s}  {pareto_m}")
print()
print(f"  Q1: Biggest freq (single) → Var {q1_best} ({q1_r['label']}): "
      f"n={q1_r['n']}  Δn=+{q1_r['n']-baseline_n}")
print(f"  Q2: Best PF preserved    → Var {q2_best_vid} ({q2_r['label']}): "
      f"PF={q2_r['pf']:.3f}  n={q2_r['n']}")
if q3_hit:
    print(f"  Q3: PF>1.20 & n≥100    → YES: {', '.join(v for v,_ in q3_hit)}")
else:
    best_n_v = max(VAR_IDS, key=lambda v: results[v]["n"])
    print(f"  Q3: PF>1.20 & n≥100    → NO  (best n={results[best_n_v]['n']}, best PF among n≥100 candidates below 1.20)")
print(f"  Q4: Pareto front         → {', '.join(pareto_front_sorted)}")
print(f"  Q5: Recommendation       → {q5_text}")
print(f"""
  Output:
    {OUT}/r041_dashboard.png
    {OUT}/r041_pareto.png
    {OUT}/r041_heatmap.png
    {OUT}/r041_equity_curves.png
    {OUT}/r041_fold_stability.png
    {OUT}/r041_bootstrap_ci.png
    {OUT}/r041_per_symbol_pf.png
    {OUT}/r041_variant_table.csv
    {OUT}/r041_journal.md
""" + "═"*115)
