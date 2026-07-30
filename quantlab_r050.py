"""
=============================================================================
QUANTLAB AI — RESEARCH #050
Universe Robustness Scan — All 9 Environments on New Symbol Universe
=============================================================================

Objective:
  R049 showed E06+E11 fails on the 26 untested symbols.
  R050 asks the harder question: does ANY of the 9 R046 survivor environments
  retain its edge on a completely different symbol universe?

  Each environment is run independently (no portfolio combination) through the
  identical 5-fold expanding walk-forward used in R042–R047 — but applied
  to the 26 symbols that never appeared in any prior research.

  Nothing in the environments is changed.  Thresholds are always learned from
  each symbol's own IS data (same mechanism as R042–R047).

  Verdict per environment: PROMOTE / WATCHLIST / REJECT
  Verdict language: "universe-robust" vs "symbol-specific"

R046 survivors tested:
  E05  ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0
  E06  ATR<p25 · Mon-Tue · BodyPct>p60 · Slope<0
  E07  ATR>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)
  E08  Dist>p60 · Wed-Thu · BodyPct>p60 · US(14-21UTC)
  E09  ADX>p67 · Dist>p75 · BodyPct>p60 · US(14-21UTC)
  E10  ATR<p40 · Dist<p33 · PrevRng>p67 · RealVol<p33
  E11  ADX>p50 · Dist>p75 · Wed-Thu · US(14-21UTC)
  E15  ADX>p67 · Dist>p75 · Wed-Thu · BodyPct>p60
  E16  Dist>p60 · Wed-Thu · PrevBody>p67 · US(14-21UTC)

Promotion criteria: identical to R047
  PF > 1.20  n ≥ 250  Boot > 1.20  MC > 80%  LOO-S > 1.0  LOO-F > 1.0  MDD < 15%

=============================================================================
"""

import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

RESEARCH_ID = "R050"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL = CONFIG["STARTING_CAPITAL"]
RR      = CONFIG["RISK_REWARD"]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_GOLD   = "#F5A623"; C_TEAL   = "#00C4CC"; C_RED    = "#E84545"
C_GREEN  = "#4BB543"; C_PURPLE = "#9B59B6"; C_BLUE   = "#2E86AB"
C_GREY   = "#888888"; C_BG     = "#0D1117"; C_PANEL  = "#161B22"
C_TEXT   = "#E6EDF3"; C_GRID   = "#21262D"

ENV_COLOURS = {
    "E05": "#F5A623", "E06": "#00C4CC", "E07": "#E84545", "E08": "#4BB543",
    "E09": "#9B59B6", "E10": "#2E86AB", "E11": "#FF6B6B", "E15": "#45B7D1",
    "E16": "#96CEB4",
}

plt.rcParams.update({
    "figure.facecolor": C_BG,  "axes.facecolor":  C_PANEL,
    "axes.edgecolor":   C_GRID,"axes.labelcolor": C_TEXT,
    "xtick.color":      C_TEXT,"ytick.color":     C_TEXT,
    "text.color":       C_TEXT,"grid.color":      C_GRID,
    "grid.alpha":       0.4,   "axes.titlecolor": C_TEXT,
    "font.family":      "monospace",
})

# ─────────────────────────────────────────────────────────────────────────────
# ALL 9 R046 SURVIVOR ENVIRONMENTS  (frozen)
# ─────────────────────────────────────────────────────────────────────────────
R046_ENVS = [
    ("E05", "ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0",   ("ATR_MD","PRG_VH","RV_LO","SLP_DN")),
    ("E06", "ATR<p25 · Mon-Tue · BodyPct>p60 · Slope<0",       ("ATR_LO","EARLY","PBP_HI","SLP_DN")),
    ("E07", "ATR>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)",     ("ATR_HI","DST_FR","MIDWK","US")),
    ("E08", "Dist>p60 · Wed-Thu · BodyPct>p60 · US(14-21UTC)", ("DST_MD","MIDWK","PBP_HI","US")),
    ("E09", "ADX>p67 · Dist>p75 · BodyPct>p60 · US(14-21UTC)",("ADX_ST","DST_FR","PBP_HI","US")),
    ("E10", "ATR<p40 · Dist<p33 · PrevRng>p67 · RealVol<p33",  ("ATR_MD","DST_NR","PRG_HI","RV_LO")),
    ("E11", "ADX>p50 · Dist>p75 · Wed-Thu · US(14-21UTC)",     ("ADX_TR","DST_FR","MIDWK","US")),
    ("E15", "ADX>p67 · Dist>p75 · Wed-Thu · BodyPct>p60",      ("ADX_ST","DST_FR","MIDWK","PBP_HI")),
    ("E16", "Dist>p60 · Wed-Thu · PrevBody>p67 · US(14-21UTC)",("DST_MD","MIDWK","PBD_HI","US")),
]
ENV_IDS   = [e[0] for e in R046_ENVS]
ENV_LABEL = {e[0]: e[1] for e in R046_ENVS}
ENV_CONDS = {e[0]: e[2] for e in R046_ENVS}

# R047 benchmark results on original 23-symbol universe
R047_BENCH = {
    "E05": {"pf":1.416,"n":213,"wr":0.459,"b50":1.422,"mc":0.995,"loo_s":1.311,"loo_f":1.174,"mdd":-0.076,"score":6},
    "E06": {"pf":1.491,"n":107,"wr":0.477,"b50":1.499,"mc":0.995,"loo_s":1.363,"loo_f":1.291,"mdd":-0.075,"score":6},
    "E07": {"pf":1.509,"n":175,"wr":0.469,"b50":1.514,"mc":0.998,"loo_s":1.377,"loo_f":1.322,"mdd":-0.075,"score":6},
    "E08": {"pf":1.453,"n":209,"wr":0.469,"b50":1.459,"mc":0.996,"loo_s":1.338,"loo_f":1.269,"mdd":-0.075,"score":6},
    "E09": {"pf":1.432,"n":118,"wr":0.458,"b50":1.440,"mc":0.993,"loo_s":1.310,"loo_f":1.271,"mdd":-0.079,"score":6},
    "E10": {"pf":1.387,"n":291,"wr":0.459,"b50":1.393,"mc":0.990,"loo_s":1.280,"loo_f":1.228,"mdd":-0.059,"score":6},
    "E11": {"pf":1.483,"n":156,"wr":0.468,"b50":1.490,"mc":0.996,"loo_s":1.362,"loo_f":1.298,"mdd":-0.073,"score":6},
    "E15": {"pf":1.533,"n":103,"wr":0.476,"b50":1.543,"mc":0.996,"loo_s":1.406,"loo_f":1.342,"mdd":-0.069,"score":6},
    "E16": {"pf":1.427,"n":154,"wr":0.461,"b50":1.433,"mc":0.993,"loo_s":1.316,"loo_f":1.258,"mdd":-0.079,"score":6},
}

# ─────────────────────────────────────────────────────────────────────────────
# NEW SYMBOL UNIVERSE  (26 symbols — zero overlap with R042–R047)
# ─────────────────────────────────────────────────────────────────────────────
SYMBOLS = [
    "1INCH-USDT-SWAP","AAVE-USDT-SWAP","ALGO-USDT-SWAP","AXS-USDT-SWAP",
    "CHZ-USDT-SWAP","COMP-USDT-SWAP","CRV-USDT-SWAP","DYDX-USDT-SWAP",
    "EGLD-USDT-SWAP","ETC-USDT-SWAP","FET-USDT-SWAP","GALA-USDT-SWAP",
    "GMX-USDT-SWAP","GRT-USDT-SWAP","HBAR-USDT-SWAP","ICP-USDT-SWAP",
    "IMX-USDT-SWAP","INJ-USDT-SWAP","LDO-USDT-SWAP","SAND-USDT-SWAP",
    "SHIB-USDT-SWAP","SNX-USDT-SWAP","STX-USDT-SWAP","SUSHI-USDT-SWAP",
    "TRX-USDT-SWAP","XLM-USDT-SWAP",
]

MIN_BARS = 4_000
FOLDS    = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
N_BOOT   = 2_000

PROM_PF   = 1.20
PROM_N    = 250
PROM_BOOT = 1.20
PROM_MC   = 0.80
PROM_MDD  = 0.15

SEP  = "═" * 110
SEP2 = "─" * 80

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION CATALOGUE  (full — frozen from R047)
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    ("ATR_LO", "ATR<p25",      "atr_rank",      "lt_q",      0.25, "vol"),
    ("ATR_MD", "ATR<p40",      "atr_rank",      "lt_q",      0.40, "vol"),
    ("ATR_HI", "ATR>p67",      "atr_rank",      "gt_q",      0.67, "vol"),
    ("RV_LO",  "RealVol<p33",  "real_vol_20",   "lt_q",      0.33, "vol"),
    ("SLP_DN", "Slope<0",      "ema200_slope",  "lt_fixed",  0.0,  "trend"),
    ("DST_FR", "Dist>p75",     "ema_dist_pct",  "gt_q_pos",  0.75, "trend"),
    ("DST_MD", "Dist>p60",     "ema_dist_pct",  "gt_q_pos",  0.60, "trend"),
    ("DST_NR", "Dist<p33",     "ema_dist_pct",  "lt_q",      0.33, "trend"),
    ("ADX_TR", "ADX>p50",      "adx14",         "gt_q",      0.50, "trend"),
    ("ADX_ST", "ADX>p67",      "adx14",         "gt_q",      0.67, "trend"),
    ("PRG_HI", "PrevRng>p67",  "prev_range_r",  "gt_q",      0.67, "part"),
    ("PRG_VH", "PrevRng>p80",  "prev_range_r",  "gt_q",      0.80, "part"),
    ("PBD_HI", "PrevBody>p67", "prev_body_r",   "gt_q",      0.67, "part"),
    ("PBP_HI", "BodyPct>p60",  "prev_body_pct", "gt_q",      0.60, "part"),
    ("US",     "US(14-21UTC)", "hour_utc",      "hour_rng",  (14,21), "time"),
    ("MIDWK",  "Wed-Thu",      "day_of_week",   "isin",      [2,3],   "time"),
    ("EARLY",  "Mon-Tue",      "day_of_week",   "isin",      [0,1],   "time"),
]
COND_BY_ID    = {c[0]: c for c in CONDITIONS_DEF}
NEEDED_CONDS  = sorted({cid for e in R046_ENVS for cid in e[2]})
QUANT_FEATS   = [
    "atr_rank","real_vol_20","bb_width","ema_dist_pct",
    "adx14","prev_range_r","prev_body_r","prev_body_pct",
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (frozen)
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]
    df["ema200"]       = calc_ema(c, 200)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank"]     = df["atr14"].rolling(100).rank(pct=True) * 100
    bb_mid             = c.rolling(20).mean()
    bb_std             = c.rolling(20).std()
    df["bb_width"]     = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["ema_dist_pct"] = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / \
                          df["ema200"].shift(10).replace(0, np.nan)
    vol_ma             = v.rolling(20).mean()
    df["rel_vol"]      = v / vol_ma.replace(0, np.nan)
    df["prev_close"]   = c.shift(1)
    df["prev_atr14"]   = df["atr14"].shift(1)
    log_ret            = np.log(c / c.shift(1))
    df["real_vol_20"]  = log_ret.rolling(20).std() * 100.0
    df["adx14"]        = calc_adx(df, 14)
    prev_range         = h.shift(1) - l.shift(1)
    prev_body          = (c.shift(1) - df["open"].shift(1)).abs()
    df["prev_range_r"] = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]  = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"]= prev_body  / prev_range.replace(0, np.nan)
    dt                 = pd.to_datetime(df["datetime"], utc=True)
    df["hour_utc"]     = dt.dt.hour.astype(np.int16)
    df["day_of_week"]  = dt.dt.dayofweek.astype(np.int16)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS & MASKS  (frozen)
# ─────────────────────────────────────────────────────────────────────────────
def learn_thresholds(df_is):
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid in NEEDED_CONDS:
        _, _, feat, direction, param, _ = COND_BY_ID[cid]
        if direction in ("gt_fixed","lt_fixed","hour_rng","isin"):
            thr[cid] = param; continue
        col = valid[feat].dropna() if feat in valid.columns else pd.Series(dtype=float)
        if len(col) < 20:
            thr[cid] = np.nan; continue
        if direction == "gt_q_pos":
            pos = col[col > 0]
            thr[cid] = float(pos.quantile(param) if len(pos) > 10 else col.quantile(param))
        else:
            thr[cid] = float(col.quantile(param))
    return thr

def condition_mask(df, cid, thr):
    _, _, feat, direction, _, _ = COND_BY_ID[cid]
    threshold = thr.get(cid, np.nan)
    n = len(df)
    if feat not in df.columns:
        return np.zeros(n, dtype=bool)
    col = df[feat].values
    nan_mask = np.isnan(col) if col.dtype.kind == "f" else np.zeros(n, dtype=bool)
    if direction == "lt_q":
        return (~nan_mask & (col < threshold)
                if not (isinstance(threshold, float) and np.isnan(threshold))
                else np.zeros(n, dtype=bool))
    elif direction in ("gt_q","gt_q_pos"):
        return (~nan_mask & (col > threshold)
                if not (isinstance(threshold, float) and np.isnan(threshold))
                else np.zeros(n, dtype=bool))
    elif direction == "gt_fixed":
        return ~nan_mask & (col > threshold)
    elif direction == "lt_fixed":
        return ~nan_mask & (col < threshold)
    elif direction == "hour_rng":
        lo, hi = threshold
        return (col >= lo) & (col <= hi)
    elif direction == "isin":
        return np.isin(col, threshold)
    return np.zeros(n, dtype=bool)

def env_mask(df, eid, thr):
    conds = ENV_CONDS[eid]
    out   = condition_mask(df, conds[0], thr)
    for cid in conds[1:]:
        out &= condition_mask(df, cid, thr)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL + BACKTEST  (frozen)
# ─────────────────────────────────────────────────────────────────────────────
def signal_relvol(df, emask):
    rv = df["rel_vol"].values
    c  = df["close"].values; o = df["open"].values; pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & emask

def run_backtest(df, signal, sym, fold):
    min_sl  = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf      = CONFIG["RISK_PER_TRADE_PCT"]
    fee     = CONFIG["TAKER_FEE"];  spd = CONFIG["SPREAD"] * 0.5
    slp     = CONFIG["SL_SLIPPAGE"]
    in_pos  = False; ep = st = tk = sz = 0.0; et = None; ei = -1
    trades  = []
    hi_ = df["high"].values; lo_ = df["low"].values; op_ = df["open"].values
    atr_= df["prev_atr14"].values; dts = df["datetime"].values
    for i in range(1, len(df)):
        if in_pos:
            sl_hit = lo_[i] <= st; tp_hit = hi_[i] >= tk
            if sl_hit or tp_hit:
                xp    = (st * (1 - slp)) if sl_hit else tk
                sd    = ep - st
                gross = (xp - ep) * sz
                cost  = (ep * sz + xp * sz) * (fee + spd)
                slpc  = (st - xp) * sz if sl_hit else 0.0
                net   = gross - cost - slpc
                rmul  = (xp - ep) / sd if sd > 0 else 0.0
                trades.append({
                    "sym": sym, "fold": fold,
                    "entry_time": str(et), "exit_time": str(dts[i]),
                    "pnl": round(net, 4), "r_multiple": round(rmul, 4),
                    "win": int(not sl_hit), "exit_type": "SL" if sl_hit else "TP",
                    "holding_bars": i - ei,
                })
                in_pos = False
            continue
        if signal[i-1]:
            a = atr_[i]
            if np.isnan(a) or a <= 0: continue
            ep_ = op_[i]
            if a / ep_ < min_sl: continue
            ep = ep_; st = ep - a; tk = ep + RR * a
            sz = min(CAPITAL * rf / a, (CAPITAL * max_lev) / ep)
            et = dts[i]; ei = i; in_pos = True
    return trades

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS  (frozen)
# ─────────────────────────────────────────────────────────────────────────────
def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl

def metrics(trades):
    if not trades:
        return {"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,"net":0.0,
                "sharpe":0.0,"mdd":0.0,"pnls":np.array([]),
                "equity":np.array([CAPITAL])}
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values; wins = df["win"].values.astype(bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw   = pnl[wins].sum()       if nw else 0.0
    gl   = abs(pnl[~wins].sum()) if nl else 0.0
    pf   = safe_pf(gw, gl); wr = nw / n
    eq   = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak = np.maximum.accumulate(eq)
    mdd  = float(((eq - peak) / peak).min())
    bpy  = 365 * 24
    ann  = (eq[-1] / CAPITAL) ** (bpy / max(n, 1)) - 1
    vol  = pnl.std() * math.sqrt(bpy) if n > 1 else 1e-9
    sha  = ann / vol if vol > 0 else 0.0
    exp  = wr * RR - (1 - wr)
    return {"n":n,"wr":wr,"pf":pf,"exp_r":exp,"net":float(pnl.sum()),
            "sharpe":sha,"mdd":mdd,"pnls":pnl,"equity":eq}

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5: return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    pfs = [safe_pf(s[s>0].sum(), abs(s[s<0].sum()))
           for _ in range(n_iter)
           for s in [rng.choice(pnls, len(pnls), replace=True)]]
    return float(np.percentile(pfs,5)), float(np.percentile(pfs,50)), float(np.percentile(pfs,95))

def monte_carlo(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return {"prob_profit":0.0,"p5":CAPITAL,"p50":CAPITAL,"p95":CAPITAL,
                "finals":np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls,len(pnls),replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit":float((finals>CAPITAL).mean()),
            "p5":float(np.percentile(finals,5)),
            "p50":float(np.percentile(finals,50)),
            "p95":float(np.percentile(finals,95)),
            "finals":finals}

def loo_sym(sym_trades):
    active = {s:tl for s,tl in sym_trades.items() if tl}
    if not active: return {}, 0.0
    ls = {omit: metrics([t for s,tl in active.items() if s!=omit for t in tl])["pf"]
          for omit in active}
    return ls, min(ls.values()) if ls else 0.0

def loo_fld(all_trades):
    folds = sorted({t["fold"] for t in all_trades})
    lf = {f: metrics([t for t in all_trades if t["fold"]!=f])["pf"] for f in folds}
    return lf, min(lf.values()) if lf else 0.0

def score_env(m, b50, mc_p, sf, ff):
    return sum([
        m["pf"] > PROM_PF, m["n"] >= PROM_N, b50 > PROM_BOOT,
        mc_p > PROM_MC, sf > 1.0, ff > 1.0, abs(m["mdd"]) < PROM_MDD,
    ])

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  QUANTLAB AI — RESEARCH #050")
print("  Universe Robustness Scan — All 9 Environments on New Symbol Universe")
print(SEP)
print()
print(f"  Testing: {len(ENV_IDS)} environments  ×  26 new symbols  ×  5-fold WF")
print(f"  Same frozen environments, same logic, completely different symbol set")
print(f"  Promotion criteria: identical to R047 (7/7 for PROMOTE)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
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

SYMBOLS    = list(all_dfs.keys())
total_bars = sum(len(d) for d in all_dfs.values())
print(f"  Loaded {len(SYMBOLS)} symbols · {total_bars:,} bars")
print()

# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD  —  one pass per environment
# ─────────────────────────────────────────────────────────────────────────────
print(f"  Running 5-fold WF for each environment …")
print(SEP2)

env_all_trades  = {eid: []                     for eid in ENV_IDS}
env_sym_trades  = {eid: defaultdict(list)      for eid in ENV_IDS}
env_fold_counts = {eid: []                     for eid in ENV_IDS}

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_n = {eid: 0 for eid in ENV_IDS}
    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[:int(N * is_end)]
        df_oos = df_full.iloc[int(N * is_end):int(N * oos_end)].copy().reset_index(drop=True)
        if len(df_oos) < 100: continue
        thr = learn_thresholds(df_is)
        for eid in ENV_IDS:
            em  = env_mask(df_oos, eid, thr)
            sig = signal_relvol(df_oos, em)
            tl  = run_backtest(df_oos, sig, sym, fold_idx)
            env_all_trades[eid].extend(tl)
            env_sym_trades[eid][sym].extend(tl)
            fold_n[eid] += len(tl)
    for eid in ENV_IDS:
        env_fold_counts[eid].append(fold_n[eid])
    counts = "  ".join(f"{e}={fold_n[e]:3d}" for e in ENV_IDS)
    print(f"  Fold {fold_idx}  IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%  {counts}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# STATS PER ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
print("  Computing stats …")
results = {}
for eid in ENV_IDS:
    tl = env_all_trades[eid]
    m  = metrics(tl)
    b5,b50,b95 = bootstrap_pf(m["pnls"])
    mc = monte_carlo(m["pnls"])
    ls,sf = loo_sym(dict(env_sym_trades[eid]))
    lf,ff = loo_fld(tl)
    sc = score_env(m, b50, mc["prob_profit"], sf, ff)
    verdict = ("PROMOTE"   if sc == 7 else
               "WATCHLIST" if sc >= 5 and m["pf"] > PROM_PF else
               "REJECT")
    results[eid] = {
        "eid":eid, "label":ENV_LABEL[eid],
        "n":m["n"], "wr":m["wr"], "pf":m["pf"],
        "b5":b5, "b50":b50, "b95":b95,
        "mc_p":mc["prob_profit"],
        "sym_floor":sf, "fold_floor":ff,
        "mdd":m["mdd"], "equity":m["equity"], "pnls":m["pnls"],
        "mc_finals":mc["finals"],
        "score":sc, "verdict":verdict,
        "loo_sym":ls, "loo_fld":lf,
    }

# Sort by score desc, then PF desc
ranked = sorted(results.values(), key=lambda x: (-x["score"], -x["pf"]))

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  ENVIRONMENT RANKINGS — NEW 26-SYMBOL UNIVERSE")
print(SEP)
print()
hdr = (f"  {'Rank':>4}  {'Env':>4}  {'Conditions':<45}  "
       f"{'n':>5}  {'WR':>6}  {'PF':>6}  {'Boot':>6}  "
       f"{'MC':>6}  {'LOO-S':>6}  {'LOO-F':>6}  {'MDD':>7}  "
       f"{'Sc':>3}  {'Verdict':<10}")
print(hdr)
print("  " + "─" * (len(hdr)-2))

for rank, r in enumerate(ranked, 1):
    vc = ("✓ " if r["verdict"]=="PROMOTE" else
          "~ " if r["verdict"]=="WATCHLIST" else
          "✗ ")
    print(f"  {rank:>4}  {r['eid']:>4}  {r['label']:<45}  "
          f"{r['n']:>5}  {r['wr']:>6.1%}  {r['pf']:>6.3f}  {r['b50']:>6.3f}  "
          f"{r['mc_p']:>6.1%}  {r['sym_floor']:>6.3f}  {r['fold_floor']:>6.3f}  "
          f"{r['mdd']:>7.2%}  {r['score']:>3}/7  {vc}{r['verdict']:<10}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# SIDE-BY-SIDE: R047 vs R050 per environment
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  COMPARISON: R047 (original 23 syms) vs R050 (new 26 syms)")
print(SEP2)
print(f"  {'Env':>4}  {'R047 PF':>8}  {'R050 PF':>8}  {'ΔPF':>8}  "
      f"{'R047 WR':>8}  {'R050 WR':>8}  {'R047 Sc':>8}  {'R050 Sc':>8}  {'R050 Verdict'}")
print("  " + "─" * 90)
for eid in ENV_IDS:
    r  = results[eid]
    b  = R047_BENCH.get(eid, {})
    dpf = r["pf"] - b.get("pf", 0)
    arrow = "▲" if dpf >= 0 else "▼"
    print(f"  {eid:>4}  {b.get('pf',0):>8.3f}  {r['pf']:>8.3f}  "
          f"{arrow}{abs(dpf):>7.3f}  "
          f"{b.get('wr',0):>8.1%}  {r['wr']:>8.1%}  "
          f"{b.get('score',0):>6}/7  {r['score']:>6}/7  {r['verdict']}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SURVIVORS
# ─────────────────────────────────────────────────────────────────────────────
promotes   = [r for r in ranked if r["verdict"] == "PROMOTE"]
watchlists = [r for r in ranked if r["verdict"] == "WATCHLIST"]
rejects    = [r for r in ranked if r["verdict"] == "REJECT"]

print(SEP)
print("  UNIVERSE ROBUSTNESS VERDICT")
print(SEP)
print()
print(f"  PROMOTE   (universe-robust):   {len(promotes)}")
for r in promotes:
    print(f"    ✓  {r['eid']}  {r['label']}  PF={r['pf']:.3f}  Score={r['score']}/7")
print()
print(f"  WATCHLIST (partial transfer):  {len(watchlists)}")
for r in watchlists:
    print(f"    ~  {r['eid']}  {r['label']}  PF={r['pf']:.3f}  Score={r['score']}/7")
print()
print(f"  REJECT    (symbol-specific):   {len(rejects)}")
for r in rejects:
    print(f"    ✗  {r['eid']}  {r['label']}  PF={r['pf']:.3f}  Score={r['score']}/7")
print()

if promotes:
    print("  ─── UNIVERSE-ROBUST ENVIRONMENTS FOUND ───")
    for r in promotes:
        print(f"\n  {r['eid']}: {r['label']}")
        print(f"    PF={r['pf']:.3f}  WR={r['wr']:.1%}  n={r['n']}"
              f"  Boot={r['b50']:.3f}  MC={r['mc_p']:.1%}"
              f"  LOO-S={r['sym_floor']:.3f}  LOO-F={r['fold_floor']:.3f}"
              f"  MDD={r['mdd']:.2%}")
else:
    print("  No environment achieves PROMOTE on the new universe.")
    print()
    if watchlists:
        print("  The best partial transfers:")
        for r in watchlists:
            passed  = []
            failed  = []
            for k,v in [("PF>1.2",r["pf"]>PROM_PF),("n≥250",r["n"]>=PROM_N),
                         ("Boot>1.2",r["b50"]>PROM_BOOT),("MC>80%",r["mc_p"]>PROM_MC),
                         ("LOO-S>1",r["sym_floor"]>1.0),("LOO-F>1",r["fold_floor"]>1.0),
                         ("MDD<15%",abs(r["mdd"])<PROM_MDD)]:
                (passed if v else failed).append(k)
            print(f"    {r['eid']}  PF={r['pf']:.3f}  Score={r['score']}/7")
            print(f"       Pass: {', '.join(passed)}")
            print(f"       Fail: {', '.join(failed)}")
    print()
    best = ranked[0]
    print(f"  Best performer on new universe: {best['eid']}  "
          f"PF={best['pf']:.3f}  Score={best['score']}/7  {best['verdict']}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for r in ranked:
    b = R047_BENCH.get(r["eid"], {})
    rows.append({
        "rank": ranked.index(r)+1,
        "eid": r["eid"], "label": r["label"],
        "n_new": r["n"], "pf_new": r["pf"], "wr_new": r["wr"],
        "boot_new": r["b50"], "mc_new": r["mc_p"],
        "loo_s_new": r["sym_floor"], "loo_f_new": r["fold_floor"],
        "mdd_new": r["mdd"], "score_new": r["score"], "verdict": r["verdict"],
        "pf_r047": b.get("pf",None), "score_r047": b.get("score",None),
        "delta_pf": r["pf"] - b.get("pf",0),
    })
pd.DataFrame(rows).to_csv(f"{OUT}/r050_results.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts …")
print(SEP2)

def panel_style(ax, title, fs=8):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title, fontsize=fs, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5)
    for sp in ax.spines.values(): sp.set_color(C_GRID)

verdict_map = {"PROMOTE":C_GREEN, "WATCHLIST":C_GOLD, "REJECT":C_RED}

# ── Chart 1: PF Comparison (R047 vs R050) ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(C_BG)
fig.suptitle("R050 — All 9 Environments: Original 23-sym (R047) vs New 26-sym (R050)",
             fontsize=11, color=C_TEXT, y=1.01)

eids  = ENV_IDS
x     = np.arange(len(eids))
w     = 0.38

# Panel A — PF
ax0 = axes[0]
pf_r047 = [R047_BENCH[e]["pf"] for e in eids]
pf_r050 = [results[e]["pf"]    for e in eids]
ax0.bar(x - w/2, pf_r047, w, color=C_GOLD, alpha=0.85, label="R047 (23 syms)")
ax0.bar(x + w/2, pf_r050, w,
        color=[verdict_map[results[e]["verdict"]] for e in eids], alpha=0.85,
        label="R050 (26 new syms)")
ax0.axhline(PROM_PF, color=C_PURPLE, linewidth=0.9, linestyle=":", alpha=0.8,
            label=f"Threshold {PROM_PF}")
ax0.axhline(1.0,     color=C_GREY,   linewidth=0.7, linestyle="--", alpha=0.5)
ax0.set_xticks(x); ax0.set_xticklabels(eids, fontsize=8)
panel_style(ax0, "Profit Factor: R047 vs R050")
ax0.legend(fontsize=6, loc="upper right")

# Panel B — Score
ax1 = axes[1]
sc_r047 = [R047_BENCH[e]["score"] for e in eids]
sc_r050 = [results[e]["score"]    for e in eids]
ax1.bar(x - w/2, sc_r047, w, color=C_GOLD, alpha=0.85, label="R047")
ax1.bar(x + w/2, sc_r050, w,
        color=[verdict_map[results[e]["verdict"]] for e in eids], alpha=0.85,
        label="R050")
ax1.axhline(7, color=C_GREY, linewidth=0.7, linestyle="--", alpha=0.5, label="Max (7)")
ax1.set_ylim(0, 8); ax1.set_xticks(x); ax1.set_xticklabels(eids, fontsize=8)
panel_style(ax1, "Promotion Score: R047 vs R050")
ax1.legend(fontsize=6)
for xi, sv, so in zip(x, sc_r047, sc_r050):
    ax1.text(xi+w/2, so+0.1, str(so), ha="center", va="bottom", fontsize=7,
             color=verdict_map[results[eids[xi]]["verdict"]])

# Panel C — WR comparison
ax2 = axes[2]
wr_r047 = [R047_BENCH[e]["wr"] for e in eids]
wr_r050 = [results[e]["wr"]    for e in eids]
ax2.bar(x - w/2, wr_r047, w, color=C_GOLD, alpha=0.85, label="R047")
ax2.bar(x + w/2, wr_r050, w,
        color=[verdict_map[results[e]["verdict"]] for e in eids], alpha=0.85,
        label="R050")
ax2.axhline(1/(1+RR), color=C_PURPLE, linewidth=0.9, linestyle=":", alpha=0.8,
            label=f"BEP {1/(1+RR):.1%}")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
ax2.set_xticks(x); ax2.set_xticklabels(eids, fontsize=8)
panel_style(ax2, "Win Rate: R047 vs R050")
ax2.legend(fontsize=6)

legend_patches = [mpatches.Patch(color=c, label=v)
                  for v, c in verdict_map.items()]
fig.legend(handles=legend_patches, loc="lower center", ncol=3,
           fontsize=8, framealpha=0.3, facecolor=C_PANEL)

plt.tight_layout()
plt.savefig(f"{OUT}/r050_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 2: Heatmap — 9 envs × 7 criteria ───────────────────────────────────
criteria_keys = ["PF>1.2","n≥250","Boot>1.2","MC>80%","LOO-S>1","LOO-F>1","MDD<15%"]
heat = np.zeros((len(ENV_IDS), len(criteria_keys)))
for i, eid in enumerate(ENV_IDS):
    r = results[eid]
    vals = [r["pf"]>PROM_PF, r["n"]>=PROM_N, r["b50"]>PROM_BOOT,
            r["mc_p"]>PROM_MC, r["sym_floor"]>1.0, r["fold_floor"]>1.0,
            abs(r["mdd"])<PROM_MDD]
    heat[i] = [float(v) for v in vals]

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(C_BG)
im = ax.imshow(heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(criteria_keys)))
ax.set_xticklabels(criteria_keys, fontsize=8, color=C_TEXT)
ax.set_yticks(range(len(ENV_IDS)))
yticklabels = [f"{e}  {ENV_LABEL[e][:38]}" for e in ENV_IDS]
ax.set_yticklabels(yticklabels, fontsize=7, color=C_TEXT)
for i in range(len(ENV_IDS)):
    for j in range(len(criteria_keys)):
        ax.text(j, i, "✓" if heat[i,j] else "✗",
                ha="center", va="center", fontsize=11,
                color="white" if heat[i,j] else "#FF6B6B")
# Add score + verdict on right
ax2r = ax.twinx()
ax2r.set_ylim(ax.get_ylim())
ax2r.set_yticks(range(len(ENV_IDS)))
ax2r.set_yticklabels(
    [f"{results[e]['score']}/7  {results[e]['verdict']}" for e in ENV_IDS],
    fontsize=7)
for tick, eid in zip(ax2r.get_yticklabels(), ENV_IDS):
    tick.set_color(verdict_map[results[eid]["verdict"]])
ax2r.tick_params(colors=C_TEXT)
for sp in ax2r.spines.values(): sp.set_color(C_GRID)

ax.set_facecolor(C_PANEL)
ax.set_title("R050 · 9 Environments × 7 Criteria — New 26-Symbol Universe",
             fontsize=9, color=C_TEXT, pad=6)
ax.tick_params(colors=C_TEXT)
for sp in ax.spines.values(): sp.set_color(C_GRID)
plt.tight_layout()
plt.savefig(f"{OUT}/r050_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 3: PF retention scatter (R047 PF vs R050 PF, each env a dot) ────────
fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor(C_BG)
for eid in ENV_IDS:
    r  = results[eid]
    b  = R047_BENCH[eid]
    col = verdict_map[r["verdict"]]
    ax.scatter(b["pf"], r["pf"], s=160, color=col, zorder=4,
               edgecolors=C_TEXT, linewidths=0.5)
    ax.annotate(eid, (b["pf"], r["pf"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, color=col)
# y=x reference: if edge fully transfers, R050 PF = R047 PF
xlim = ax.get_xlim(); ylim = ax.get_ylim()
mn = min(xlim[0], ylim[0]); mx = max(xlim[1], ylim[1])
ax.plot([mn,mx],[mn,mx], color=C_GREY, linewidth=0.8, linestyle="--",
        alpha=0.5, label="Perfect transfer (y=x)")
ax.axhline(PROM_PF, color=C_PURPLE, linewidth=0.8, linestyle=":", alpha=0.8,
           label=f"PF threshold {PROM_PF}")
ax.axvline(PROM_PF, color=C_PURPLE, linewidth=0.8, linestyle=":", alpha=0.8)
# Shade "transfer zone" (above threshold on both axes)
ax.fill_between([PROM_PF, mx], PROM_PF, mx, alpha=0.06, color=C_GREEN,
                label="Both PROMOTE zone")
legend_patches2 = [mpatches.Patch(color=c, label=v) for v,c in verdict_map.items()]
ax.legend(handles=legend_patches2 + ax.get_legend_handles_labels()[0],
          fontsize=7, loc="upper left")
panel_style(ax, "PF Retention: R047 PF (x) vs R050 PF (y) — Does the edge transfer?")
ax.set_xlabel("R047 PF  (original 23 syms)", fontsize=9)
ax.set_ylabel("R050 PF  (new 26 syms)", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/r050_pf_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 4: Equity curves — all 9 envs on new universe ──────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 13))
fig.patch.set_facecolor(C_BG)
fig.suptitle("R050 · Equity Curves — 9 Environments on New 26-Symbol Universe",
             fontsize=11, color=C_TEXT)

for idx, eid in enumerate(ENV_IDS):
    r   = results[eid]
    ax  = axes[idx//3][idx%3]
    col = verdict_map[r["verdict"]]
    if len(r["equity"]) > 1:
        eqi = np.arange(len(r["equity"]))
        ax.plot(eqi, r["equity"], color=col, linewidth=1.4)
        pk = np.maximum.accumulate(r["equity"])
        ax.fill_between(eqi, r["equity"], pk, alpha=0.2, color=C_RED)
        ax.axhline(CAPITAL, color=C_GREY, linewidth=0.7, linestyle="--", alpha=0.5)
    panel_style(ax, f"{eid}  PF={r['pf']:.3f}  n={r['n']}  Sc={r['score']}/7  {r['verdict']}")
    ax.set_xlabel("Trade #", fontsize=7)
    ax.set_ylabel("Capital ($)", fontsize=7)
    # R047 reference line
    b   = R047_BENCH[eid]
    ax.text(0.98, 0.06, f"R047: PF={b['pf']:.3f}",
            transform=ax.transAxes, fontsize=6, ha="right", color=C_GOLD, alpha=0.8)

plt.tight_layout()
plt.savefig(f"{OUT}/r050_equity_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 5: Dashboard summary ────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor(C_BG)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35,
                         left=0.05, right=0.97, top=0.93, bottom=0.06)

n_promo = len(promotes); n_watch = len(watchlists); n_reject = len(rejects)
summary_col = C_GREEN if n_promo > 0 else (C_GOLD if n_watch > 0 else C_RED)
fig.text(0.5, 0.965,
         "QUANTLAB AI — R050: Universe Robustness Scan", 
         ha="center", fontsize=13, color=C_TEXT, weight="bold")
fig.text(0.5, 0.945,
         f"9 environments × 26 new symbols  ·  "
         f"PROMOTE: {n_promo}  WATCHLIST: {n_watch}  REJECT: {n_reject}",
         ha="center", fontsize=9, color=summary_col)

# A — PF grouped bar
ax_a = fig.add_subplot(gs[0, 0])
pf_r047 = [R047_BENCH[e]["pf"] for e in ENV_IDS]
pf_r050 = [results[e]["pf"]    for e in ENV_IDS]
x  = np.arange(len(ENV_IDS)); w = 0.38
ax_a.bar(x-w/2, pf_r047, w, color=C_GOLD,  alpha=0.85, label="R047")
ax_a.bar(x+w/2, pf_r050, w,
         color=[verdict_map[results[e]["verdict"]] for e in ENV_IDS],
         alpha=0.85, label="R050 (new)")
ax_a.axhline(PROM_PF, color=C_PURPLE, linewidth=0.9, linestyle=":", alpha=0.8)
ax_a.axhline(1.0,     color=C_GREY,   linewidth=0.7, linestyle="--", alpha=0.4)
ax_a.set_xticks(x); ax_a.set_xticklabels(ENV_IDS, fontsize=8)
panel_style(ax_a, "PF — R047 vs R050 (new universe)")
ax_a.legend(fontsize=7)

# B — score grouped bar
ax_b = fig.add_subplot(gs[0, 1])
sc_r047 = [R047_BENCH[e]["score"] for e in ENV_IDS]
sc_r050 = [results[e]["score"]    for e in ENV_IDS]
ax_b.bar(x-w/2, sc_r047, w, color=C_GOLD,  alpha=0.85, label="R047")
ax_b.bar(x+w/2, sc_r050, w,
         color=[verdict_map[results[e]["verdict"]] for e in ENV_IDS],
         alpha=0.85, label="R050 (new)")
ax_b.axhline(7, color=C_GREY, linewidth=0.7, linestyle="--", alpha=0.4, label="Max")
ax_b.set_ylim(0,8); ax_b.set_xticks(x); ax_b.set_xticklabels(ENV_IDS, fontsize=8)
panel_style(ax_b, "Score/7 — R047 vs R050 (new universe)")
ax_b.legend(fontsize=7)

# C — scatter
ax_c = fig.add_subplot(gs[1, 0])
for eid in ENV_IDS:
    r  = results[eid]; b = R047_BENCH[eid]
    ax_c.scatter(b["pf"], r["pf"], s=120,
                 color=verdict_map[r["verdict"]], zorder=4,
                 edgecolors=C_TEXT, linewidths=0.5)
    ax_c.annotate(eid, (b["pf"], r["pf"]),
                  textcoords="offset points", xytext=(5, 3),
                  fontsize=7, color=verdict_map[r["verdict"]])
xlim2 = ax_c.get_xlim(); ylim2 = ax_c.get_ylim()
mn2 = min(xlim2[0], ylim2[0]); mx2 = max(xlim2[1], ylim2[1])
ax_c.plot([mn2,mx2],[mn2,mx2], color=C_GREY, linewidth=0.7, linestyle="--", alpha=0.5)
ax_c.axhline(PROM_PF, color=C_PURPLE, linewidth=0.7, linestyle=":", alpha=0.7)
ax_c.axvline(PROM_PF, color=C_PURPLE, linewidth=0.7, linestyle=":", alpha=0.7)
panel_style(ax_c, "PF Transfer: R047 PF (x) → R050 PF (y)")
ax_c.set_xlabel("R047 PF", fontsize=8); ax_c.set_ylabel("R050 PF", fontsize=8)

# D — verdict table
ax_d = fig.add_subplot(gs[1, 1])
ax_d.set_facecolor(C_PANEL); ax_d.axis("off")
tbl_hdr = ["Env","Conditions (abbr)","R047 PF","R050 PF","ΔPF","Score","Verdict"]
tbl_dat = []
for r in ranked:
    b   = R047_BENCH[r["eid"]]
    dpf = r["pf"] - b["pf"]
    tbl_dat.append([
        r["eid"],
        r["label"][:35],
        f"{b['pf']:.3f}",
        f"{r['pf']:.3f}",
        f"{dpf:+.3f}",
        f"{r['score']}/7",
        r["verdict"],
    ])
tbl = ax_d.table(cellText=tbl_dat, colLabels=tbl_hdr,
                 cellLoc="center", loc="center",
                 bbox=[0.0, 0.0, 1.0, 1.0])
tbl.auto_set_font_size(False); tbl.set_fontsize(6)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(C_GRID)
    cell.set_facecolor(C_PANEL if row > 0 else "#1C2128")
    cell.set_text_props(color=C_TEXT)
    if row > 0 and col == 6:
        v = tbl_dat[row-1][6]
        cell.set_text_props(color=verdict_map.get(v, C_TEXT))
    if row > 0 and col == 4:
        dpf_v = float(tbl_dat[row-1][4])
        cell.set_text_props(color=C_GREEN if dpf_v >= 0 else C_RED)
panel_style(ax_d, "Full Results Table")

plt.savefig(f"{OUT}/r050_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  OUTPUT FILES")
print(SEP)
for fname in ["r050_dashboard.png","r050_comparison.png","r050_heatmap.png",
              "r050_pf_scatter.png","r050_equity_curves.png","r050_results.csv"]:
    if os.path.exists(f"{OUT}/{fname}"):
        print(f"    {OUT}/{fname}")
print()
print(SEP)
print("  R050 COMPLETE — UNIVERSE ROBUSTNESS SCAN")
print(SEP)
print(f"  Symbols tested:  {len(SYMBOLS)} (zero overlap with R042–R047)")
print(f"  Environments:    {len(ENV_IDS)}")
print(f"  PROMOTE:         {n_promo}")
print(f"  WATCHLIST:       {n_watch}")
print(f"  REJECT:          {n_reject}")
print()
best = ranked[0]
print(f"  Best on new universe: {best['eid']}  PF={best['pf']:.3f}  "
      f"Score={best['score']}/7  {best['verdict']}")
print(SEP)
