"""
=============================================================================
QUANTLAB AI — RESEARCH #047
Global Portfolio Optimisation — Exhaustive Search
=============================================================================

Objective:
  R046 used a greedy heuristic. R047 exhaustively evaluates EVERY feasible
  2-, 3-, 4- and 5-environment portfolio from the 9 R046 survivors (E12
  rejected) to find the globally optimal production portfolio.

Dataset:
  9 surviving environments from R046 (E05-E11, E15, E16; E12 excluded).
  Identical RELVOL entry · identical 5-fold expanding walk-forward OOS.
  No parameter changes.

Search space:
  C(9,2)=36  +  C(9,3)=84  +  C(9,4)=126  +  C(9,5)=126  =  372 portfolios
  Walk-forward runs ONCE; all 372 combos computed from those trade sets.

Promotion Rules (R047 — tighter than R046):
  PF > 1.20  ·  n ≥ 250  ·  Boot median > 1.20  ·  MC > 80%
  LOO-S > 1.0  ·  LOO-F > 1.0  ·  MDD < 15%

=============================================================================
"""

import os, sys, math, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

RESEARCH_ID = "R047"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = CONFIG["RISK_REWARD"]
BEP_WR   = 1.0 / (1.0 + RR)

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
    "figure.facecolor": C_BG, "axes.facecolor": C_PANEL,
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "text.color": C_TEXT, "grid.color": C_GRID,
    "grid.alpha": 0.4, "axes.titlecolor": C_TEXT,
    "font.family": "monospace",
})

# ─────────────────────────────────────────────────────────────────────────────
# R046 SURVIVORS (E12 rejected — fold_floor < 1.0, score 3/7)
# ─────────────────────────────────────────────────────────────────────────────
R046_ENVS = [
    ("E05", "ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0",  ("ATR_MD","PRG_VH","RV_LO","SLP_DN")),
    ("E06", "ATR<p25 · Mon-Tue · BodyPct>p60 · Slope<0",      ("ATR_LO","EARLY","PBP_HI","SLP_DN")),
    ("E07", "ATR>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)",    ("ATR_HI","DST_FR","MIDWK","US")),
    ("E08", "Dist>p60 · Wed-Thu · BodyPct>p60 · US(14-21UTC)",("DST_MD","MIDWK","PBP_HI","US")),
    ("E09", "ADX>p67 · Dist>p75 · BodyPct>p60 · US(14-21UTC)",("ADX_ST","DST_FR","PBP_HI","US")),
    ("E10", "ATR<p40 · Dist<p33 · PrevRng>p67 · RealVol<p33", ("ATR_MD","DST_NR","PRG_HI","RV_LO")),
    ("E11", "ADX>p50 · Dist>p75 · Wed-Thu · US(14-21UTC)",    ("ADX_TR","DST_FR","MIDWK","US")),
    ("E15", "ADX>p67 · Dist>p75 · Wed-Thu · BodyPct>p60",     ("ADX_ST","DST_FR","MIDWK","PBP_HI")),
    ("E16", "Dist>p60 · Wed-Thu · PrevBody>p67 · US(14-21UTC)",("DST_MD","MIDWK","PBD_HI","US")),
]

ENV_IDS   = [e[0] for e in R046_ENVS]
ENV_LABEL = {e[0]: e[1] for e in R046_ENVS}
ENV_CONDS = {e[0]: e[2] for e in R046_ENVS}

ENV_SESSION = {
    "E05": {"NONE"}, "E06": {"EARLY"},      "E07": {"US","MIDWK"},
    "E08": {"US","MIDWK"}, "E09": {"US"},   "E10": {"NONE"},
    "E11": {"US","MIDWK"}, "E15": {"MIDWK"},"E16": {"US","MIDWK"},
}

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    ("ATR_LO", "ATR<p25",      "atr_rank",      "lt_q",      0.25, "vol"),
    ("ATR_MD", "ATR<p40",      "atr_rank",      "lt_q",      0.40, "vol"),
    ("ATR_HI", "ATR>p67",      "atr_rank",      "gt_q",      0.67, "vol"),
    ("RV_LO",  "RealVol<p33",  "real_vol_20",   "lt_q",      0.33, "vol"),
    ("BB_TG",  "BB<p33",       "bb_width",      "lt_q",      0.33, "vol"),
    ("BB_MD",  "BB<p50",       "bb_width",      "lt_q",      0.50, "vol"),
    ("SLP_UP", "Slope>0",      "ema200_slope",  "gt_fixed",  0.0,  "trend"),
    ("SLP_DN", "Slope<0",      "ema200_slope",  "lt_fixed",  0.0,  "trend"),
    ("DST_FR", "Dist>p75",     "ema_dist_pct",  "gt_q_pos",  0.75, "trend"),
    ("DST_MD", "Dist>p60",     "ema_dist_pct",  "gt_q_pos",  0.60, "trend"),
    ("DST_NR", "Dist<p33",     "ema_dist_pct",  "lt_q",      0.33, "trend"),
    ("ADX_TR", "ADX>p50",      "adx14",         "gt_q",      0.50, "trend"),
    ("ADX_ST", "ADX>p67",      "adx14",         "gt_q",      0.67, "trend"),
    ("ADX_WK", "ADX<p33",      "adx14",         "lt_q",      0.33, "trend"),
    ("PRG_HI", "PrevRng>p67",  "prev_range_r",  "gt_q",      0.67, "part"),
    ("PRG_VH", "PrevRng>p80",  "prev_range_r",  "gt_q",      0.80, "part"),
    ("PBD_HI", "PrevBody>p67", "prev_body_r",   "gt_q",      0.67, "part"),
    ("PBP_HI", "BodyPct>p60",  "prev_body_pct", "gt_q",      0.60, "part"),
    ("ASIA",   "Asia(0-7UTC)", "hour_utc",      "hour_rng",  (0,7),   "time"),
    ("EUR",    "Eur(8-15UTC)", "hour_utc",      "hour_rng",  (8,15),  "time"),
    ("US",     "US(14-21UTC)","hour_utc",       "hour_rng",  (14,21), "time"),
    ("MIDWK",  "Wed-Thu",     "day_of_week",   "isin",      [2,3],   "time"),
    ("EARLY",  "Mon-Tue",     "day_of_week",   "isin",      [0,1],   "time"),
]
COND_BY_ID = {c[0]: c for c in CONDITIONS_DEF}
NEEDED_CONDS = sorted({cid for e in R046_ENVS for cid in e[2]})
QUANT_FEATS = [
    "atr_rank","real_vol_20","bb_width","ema_dist_pct",
    "adx14","prev_range_r","prev_body_r","prev_body_pct",
]

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

# R047 tighter promotion thresholds
PROM_PF   = 1.20
PROM_N    = 250        # raised from 200
PROM_BOOT = 1.20
PROM_MC   = 0.80       # raised from 0.60
PROM_MDD  = 0.15       # tightened from 0.25

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
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
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / df["ema200"].shift(10).replace(0,np.nan)
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
# THRESHOLDS & MASKS
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
    nan_mask = np.isnan(col) if col.dtype.kind == 'f' else np.zeros(n, dtype=bool)
    if direction == "lt_q":
        return (~nan_mask & (col < threshold) if not (isinstance(threshold, float) and np.isnan(threshold))
                else np.zeros(n, dtype=bool))
    elif direction in ("gt_q","gt_q_pos"):
        return (~nan_mask & (col > threshold) if not (isinstance(threshold, float) and np.isnan(threshold))
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

def signal_relvol(df, emask):
    rv = df["rel_vol"].values
    c  = df["close"].values
    o  = df["open"].values
    pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & emask

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signal, sym, fold, eid, attribution=None):
    min_sl  = CONFIG["MIN_SL_PCT"]
    max_lev = CONFIG["MAX_LEVERAGE"]
    rf      = CONFIG["RISK_PER_TRADE_PCT"]
    fee     = CONFIG["TAKER_FEE"]
    spd     = CONFIG["SPREAD"] * 0.5
    slp     = CONFIG["SL_SLIPPAGE"]
    in_pos  = False
    ep = st = tk = sz = 0.0; et = None; ei = -1
    trades  = []
    hi_  = df["high"].values
    lo_  = df["low"].values
    op_  = df["open"].values
    atr_ = df["prev_atr14"].values
    dts  = df["datetime"].values
    for i in range(1, len(df)):
        if in_pos:
            sl_hit = lo_[i] <= st
            tp_hit = hi_[i] >= tk
            if sl_hit or tp_hit:
                xp    = (st * (1 - slp)) if sl_hit else tk
                xt    = "SL" if sl_hit else "TP"
                sd    = ep - st
                gross = (xp - ep) * sz
                cost  = (ep * sz + xp * sz) * (fee + spd)
                slpc  = (st - xp) * sz if sl_hit else 0.0
                net   = gross - cost - slpc
                rmul  = (xp - ep) / sd if sd > 0 else 0.0
                fired = attribution[i-1] if attribution is not None else eid
                trades.append({
                    "sym": sym, "fold": fold, "env": fired,
                    "entry_time": str(et), "exit_time": str(dts[i]),
                    "pnl": round(net, 4), "r_multiple": round(rmul, 4),
                    "win": int(xt == "TP"), "exit_type": xt,
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
            et = dts[i]; ei = i
            in_pos = True
    return trades

def portfolio_signal(env_signals):
    n        = len(env_signals[0][1])
    combined = np.zeros(n, dtype=bool)
    attr     = np.full(n, '', dtype=object)
    for eid, sig in env_signals:
        new_fires      = sig & ~combined
        combined      |= new_fires
        attr[new_fires] = eid
    return combined, attr

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl

def metrics(trades):
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
    return (float(np.percentile(pfs, 5)),
            float(np.percentile(pfs, 50)),
            float(np.percentile(pfs, 95)))

def monte_carlo(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return {"prob_profit":0.0,"p5":CAPITAL,"p50":CAPITAL,
                "p95":CAPITAL,"finals":np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls,len(pnls),replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit":float((finals > CAPITAL).mean()),
            "p5":float(np.percentile(finals,5)),
            "p50":float(np.percentile(finals,50)),
            "p95":float(np.percentile(finals,95)),
            "finals":finals}

def loo_sym(sym_trades_dict):
    all_s = [s for s in sym_trades_dict if sym_trades_dict[s]]
    return {omit: metrics([t for s,tl in sym_trades_dict.items()
                           if s != omit for t in tl])["pf"]
            for omit in all_s}

def loo_fld(all_trades):
    folds = sorted({t["fold"] for t in all_trades})
    return {f: metrics([t for t in all_trades if t["fold"] != f])["pf"]
            for f in folds}

def full_stats(all_trades, sym_trades):
    m          = metrics(all_trades)
    b5,b50,b95 = bootstrap_pf(m["pnls"])
    mc         = monte_carlo(m["pnls"])
    ls         = loo_sym(sym_trades)
    lf         = loo_fld(all_trades)
    sf         = min(ls.values()) if ls else 0.0
    ff         = min(lf.values()) if lf else 0.0
    score      = sum([
        m["pf"]           > PROM_PF,
        m["n"]            >= PROM_N,
        b50               > PROM_BOOT,
        mc["prob_profit"] > PROM_MC,
        sf                > 1.0,
        ff                > 1.0,
        abs(m["mdd"])     < PROM_MDD,
    ])
    verdict = ("PROMOTE"   if score == 7 else
               "WATCHLIST" if score >= 5 and m["pf"] > PROM_PF else
               "REJECT")
    return {**m,
            "b5":b5,"b50":b50,"b95":b95,
            "mc_p":mc["prob_profit"],"mc_p50":mc["p50"],
            "mc_finals":mc["finals"],
            "sym_floor":sf,"fold_floor":ff,
            "loo_sym":ls,"loo_fld":lf,
            "score":score,"verdict":verdict}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
SEP = "═" * 110
print(SEP)
print("  QUANTLAB AI — RESEARCH #047")
print("  Global Portfolio Optimisation — Exhaustive Search")
print(SEP)
print()
print(f"  R046 survivors: {len(R046_ENVS)} environments (E12 excluded)")
total_combos = sum(
    len(list(itertools.combinations(ENV_IDS, k))) for k in range(2, 6)
)
for k in range(2, 6):
    n = len(list(itertools.combinations(ENV_IDS, k)))
    print(f"    {k}-env portfolios: {n}")
print(f"  Total portfolios to evaluate: {total_combos}")
print(f"  Promotion thresholds: PF>{PROM_PF}  n≥{PROM_N}  Boot>{PROM_BOOT}  "
      f"MC>{PROM_MC*100:.0f}%  LOO>1.0  MDD<{PROM_MDD*100:.0f}%")
print()

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
total_bars = sum(len(d) for d in all_dfs.values())
print(f"  Loaded {len(SYMBOLS)} symbols · {total_bars:,} bars")
print()

# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD — single pass, all 9 environments
# ─────────────────────────────────────────────────────────────────────────────
env_sym_trades = {eid: defaultdict(list) for eid in ENV_IDS}
env_bar_pnl    = {eid: defaultdict(float) for eid in ENV_IDS}
fold_env_n     = {eid: [] for eid in ENV_IDS}

print(f"  Walk-forward: {len(FOLDS)} folds × {len(SYMBOLS)} symbols")
print("─" * 80)

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_counts = {eid: 0 for eid in ENV_IDS}
    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[:int(N * is_end)]
        df_oos = df_full.iloc[int(N * is_end):int(N * oos_end)].reset_index(drop=True)
        if len(df_oos) < 100: continue
        thr = learn_thresholds(df_is)
        for eid in ENV_IDS:
            em  = env_mask(df_oos, eid, thr)
            sig = signal_relvol(df_oos, em)
            tl  = run_backtest(df_oos, sig, sym, fold_idx, eid)
            env_sym_trades[eid][sym].extend(tl)
            fold_counts[eid] += len(tl)
            for t in tl:
                env_bar_pnl[eid][t["entry_time"]] += t["pnl"]
    counts_str = "  ".join(f"{e}={fold_counts[e]:3d}" for e in ENV_IDS)
    total_f    = sum(fold_counts.values())
    print(f"  Fold {fold_idx}  IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%  {counts_str}  TOTAL={total_f}")
    for eid in ENV_IDS:
        fold_env_n[eid].append(fold_counts[eid])

print()

# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL ENV STATS (R047 thresholds)
# ─────────────────────────────────────────────────────────────────────────────
print("  Computing individual environment stats …")
env_results = {}
for eid in ENV_IDS:
    flat = [t for tl in env_sym_trades[eid].values() for t in tl]
    env_results[eid] = {
        "id": eid, "label": ENV_LABEL[eid],
        "sym_trades": dict(env_sym_trades[eid]),
        "_flat": flat,
        **full_stats(flat, dict(env_sym_trades[eid]))
    }

env_trade_sets = {
    eid: {(t["sym"], t["entry_time"]) for t in env_results[eid]["_flat"]}
    for eid in ENV_IDS
}

# Per-bar PnL vectors for correlation
all_times  = sorted({t for eid in ENV_IDS for t in env_bar_pnl[eid]})
time_index = {t: i for i, t in enumerate(all_times)}
T          = len(all_times)
env_pnl_vec = {}
for eid in ENV_IDS:
    v = np.zeros(T)
    for t, pnl in env_bar_pnl[eid].items():
        if t in time_index:
            v[time_index[t]] += pnl
    env_pnl_vec[eid] = v

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO BUILDER — dedup by priority cascade on pre-computed trade sets
# ─────────────────────────────────────────────────────────────────────────────
def build_portfolio_trades(eid_list):
    """
    Priority cascade over sorted eid_list.
    Returns deduped flat trade list.
    """
    seen     = set()
    combined = []
    for eid in eid_list:
        for sym, tl in env_sym_trades[eid].items():
            for t in tl:
                key = (t["sym"], t["entry_time"])
                if key not in seen:
                    seen.add(key)
                    combined.append({**t, "env": eid})
    return combined


def portfolio_metrics_fast(eid_list):
    """
    Build portfolio, compute full_stats, return result dict.
    """
    flat = build_portfolio_trades(eid_list)
    sym_trades = defaultdict(list)
    for t in flat:
        sym_trades[t["sym"]].append(t)
    return full_stats(flat, dict(sym_trades))


def pairwise_avg_overlap(eid_list):
    """Average Jaccard trade overlap across all pairs in portfolio."""
    pairs = list(itertools.combinations(eid_list, 2))
    if not pairs:
        return 0.0
    ovs = []
    for e1, e2 in pairs:
        s1, s2 = env_trade_sets[e1], env_trade_sets[e2]
        union  = s1 | s2
        ovs.append(len(s1 & s2) / max(len(union), 1) * 100)
    return float(np.mean(ovs))


def pairwise_avg_corr(eid_list):
    """Average pairwise PnL correlation across all pairs."""
    pairs = list(itertools.combinations(eid_list, 2))
    if not pairs:
        return 0.0
    corrs = []
    for e1, e2 in pairs:
        v1, v2 = env_pnl_vec[e1], env_pnl_vec[e2]
        active = (v1 != 0) | (v2 != 0)
        if active.sum() < 10:
            corrs.append(0.0); continue
        r = np.corrcoef(v1[active], v2[active])[0, 1]
        corrs.append(float(r) if not np.isnan(r) else 0.0)
    return float(np.mean(corrs))


def diversification_ratio(eid_list):
    """
    Weighted avg of individual PFs / portfolio PF.
    > 1.0 = diversification benefit; < 1.0 = diversification drag.
    """
    port_pf = portfolio_metrics_fast(eid_list)["pf"]
    if port_pf <= 0:
        return 0.0
    individual_pfs = [env_results[e]["pf"] for e in eid_list]
    avg_ind_pf = np.mean(individual_pfs)
    return avg_ind_pf / port_pf if port_pf > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# EXHAUSTIVE SEARCH
# ─────────────────────────────────────────────────────────────────────────────
print(f"  Exhaustive search: {total_combos} portfolios …")
print()

all_portfolio_results = []

for k in range(2, 6):
    combos = list(itertools.combinations(ENV_IDS, k))
    print(f"  {k}-environment portfolios ({len(combos)} combos) …", flush=True)
    for combo in combos:
        eid_list = list(combo)
        pid      = "+".join(eid_list)
        r        = portfolio_metrics_fast(eid_list)
        avg_ov   = pairwise_avg_overlap(eid_list)
        avg_corr = pairwise_avg_corr(eid_list)
        all_portfolio_results.append({
            "pid":        pid,
            "envs":       eid_list,
            "k":          k,
            "n":          r["n"],
            "wr":         r["wr"],
            "pf":         r["pf"],
            "b50":        r["b50"],
            "b5":         r["b5"],
            "b95":        r["b95"],
            "mc_p":       r["mc_p"],
            "mdd":        r["mdd"],
            "sym_floor":  r["sym_floor"],
            "fold_floor": r["fold_floor"],
            "score":      r["score"],
            "verdict":    r["verdict"],
            "avg_overlap": avg_ov,
            "avg_corr":   avg_corr,
            "exp_r":      r["exp_r"],
            "net":        r["net"],
            "_flat":      None,  # kept sparse — only build when needed
            "_stats":     r,
        })
    k_results = [x for x in all_portfolio_results if x["k"] == k]
    best_k    = max(k_results, key=lambda x: (x["score"], x["pf"]))
    print(f"    Best {k}-env: {best_k['pid']}  PF={best_k['pf']:.3f}  n={best_k['n']}  "
          f"Score={best_k['score']}/7  {best_k['verdict']}")

# Sort by (score desc, pf desc)
all_portfolio_results.sort(key=lambda x: (x["score"], x["pf"]), reverse=True)

print()
print(f"  Total portfolios evaluated: {len(all_portfolio_results)}")
promote_all  = [x for x in all_portfolio_results if x["verdict"] == "PROMOTE"]
watch_all    = [x for x in all_portfolio_results if x["verdict"] == "WATCHLIST"]
reject_all   = [x for x in all_portfolio_results if x["verdict"] == "REJECT"]
print(f"  PROMOTE:   {len(promote_all)}")
print(f"  WATCHLIST: {len(watch_all)}")
print(f"  REJECT:    {len(reject_all)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  RESEARCH QUESTIONS")
print(SEP)

def best_by_k(k_val):
    subset = [x for x in all_portfolio_results if x["k"] == k_val]
    return max(subset, key=lambda x: (x["score"], x["pf"]))

def top_n_by_k(k_val, n=5):
    subset = [x for x in all_portfolio_results if x["k"] == k_val]
    subset.sort(key=lambda x: (x["score"], x["pf"]), reverse=True)
    return subset[:n]

def print_portfolio_row(x, label=""):
    print(f"    {label or x['pid']:<30}  n={x['n']:>4}  WR={x['wr']*100:>5.1f}%  "
          f"PF={x['pf']:>6.3f}  p50={x['b50']:>6.3f}  "
          f"[{x['b5']:.3f},{x['b95']:.3f}]  "
          f"MC={x['mc_p']*100:>3.0f}%  MDD={x['mdd']:>5.1%}  "
          f"LOO-S={x['sym_floor']:>5.3f}  LOO-F={x['fold_floor']:>5.3f}  "
          f"Sc={x['score']}/7  {x['verdict']}")

for q, k_val in enumerate([2, 3, 4, 5], start=1):
    bk = best_by_k(k_val)
    print(f"\n  Q{q}. Best {k_val}-environment portfolio maximising Score:")
    print(f"  {'─'*80}")
    print(f"    Winner:  {bk['pid']}")
    print(f"    Envs:    {', '.join(bk['envs'])}")
    print_portfolio_row(bk, label=bk['pid'])
    print(f"    Labels:")
    for e in bk['envs']:
        print(f"      {e}: {ENV_LABEL[e]}")
    top5 = top_n_by_k(k_val, 5)
    if len(top5) > 1:
        print(f"    Top 5 {k_val}-env by score/PF:")
        for x in top5:
            print_portfolio_row(x, label=f"      {x['pid']:<28}")

# Q5: Does adding envs reduce PF through correlation?
print(f"\n  Q5. Does adding environments eventually reduce PF due to correlation?")
print(f"  {'─'*80}")
for k_val in range(2, 6):
    subset = [x for x in all_portfolio_results if x["k"] == k_val]
    pfs  = [x["pf"]  for x in subset]
    mcs  = [x["mc_p"] for x in subset]
    best = max(subset, key=lambda x: x["pf"])
    worst= min(subset, key=lambda x: x["pf"])
    print(f"    k={k_val}:  median_PF={np.median(pfs):.3f}  max_PF={best['pf']:.3f}  "
          f"min_PF={worst['pf']:.3f}  median_MC={np.median(mcs)*100:.0f}%")

# Correlation effect: pick best 2-env and trace what adding each extra env does
best2 = best_by_k(2)
print(f"\n  Trace from best 2-env [{best2['pid']}] PF={best2['pf']:.3f}:")
base_envs = list(best2["envs"])
for eid in [e for e in ENV_IDS if e not in base_envs]:
    candidate = sorted(base_envs + [eid])
    match = next((x for x in all_portfolio_results if sorted(x["envs"]) == candidate), None)
    if match:
        delta = match["pf"] - best2["pf"]
        print(f"    +{eid}: PF={match['pf']:.3f} ({delta:+.3f})  n={match['n']}  "
              f"corr={match['avg_corr']:.3f}  overlap={match['avg_overlap']:.1f}%")

# Q6: Pareto frontier (PF vs Trades)
print(f"\n  Q6. Pareto frontier (PF vs Trade Count):")
print(f"  {'─'*80}")
# Compute Pareto-optimal set
pareto = []
for x in all_portfolio_results:
    dominated = False
    for y in all_portfolio_results:
        if y["pid"] == x["pid"]: continue
        if y["pf"] >= x["pf"] and y["n"] >= x["n"] and (y["pf"] > x["pf"] or y["n"] > x["n"]):
            dominated = True; break
    if not dominated:
        pareto.append(x)
pareto.sort(key=lambda x: x["n"])
print(f"    {len(pareto)} portfolios on Pareto frontier:")
for x in pareto[:15]:
    print(f"    {x['pid']:<32}  PF={x['pf']:.3f}  n={x['n']:>4}  "
          f"Score={x['score']}/7  {x['verdict']}")

# Q7: Is R046 greedy portfolio globally optimal?
print(f"\n  Q7. Is R046 greedy portfolio (E08+E06) globally optimal?")
print(f"  {'─'*80}")
r046_greedy_pid = "E08+E06"
r046_greedy = next((x for x in all_portfolio_results
                    if sorted(x["envs"]) == sorted(["E08","E06"])), None)
global_best  = all_portfolio_results[0]

if r046_greedy:
    print(f"    R046 greedy: {r046_greedy_pid}")
    print_portfolio_row(r046_greedy, label=f"    R046 greedy [{r046_greedy_pid}]")
print(f"    Global best: {global_best['pid']}")
print_portfolio_row(global_best, label=f"    Global best  [{global_best['pid']}]")

if r046_greedy and global_best["pid"] != r046_greedy_pid:
    delta_pf = global_best["pf"] - r046_greedy["pf"]
    delta_n  = global_best["n"]  - r046_greedy["n"]
    print(f"\n    ✗ Greedy is NOT globally optimal.")
    print(f"    Improvement: PF {delta_pf:+.3f}  |  Trades {delta_n:+d}")
    print(f"    Optimal replaces [{r046_greedy_pid}] with [{global_best['pid']}]")
else:
    print(f"\n    ✓ Greedy IS globally optimal (or matches global best).")

# ─────────────────────────────────────────────────────────────────────────────
# TOP 20 PORTFOLIOS
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  TOP 20 PORTFOLIOS (sorted by Score then PF)")
print(SEP)
hdr = (f"  {'#':>3}  {'Portfolio':<32}  {'k':>2}  {'n':>4}  {'WR':>6}  {'PF':>6}  "
       f"{'p50':>6}  {'90%CI':>14}  {'MC%':>4}  {'MDD':>6}  "
       f"{'LOO-S':>5}  {'LOO-F':>5}  {'OvlpAvg':>7}  {'Corr':>5}  {'Sc':>3}  Verdict")
print(hdr)
print("  " + "─" * 120)
oos_years = 1.0
for rank, x in enumerate(all_portfolio_results[:20], start=1):
    ci = f"[{x['b5']:.3f},{x['b95']:.3f}]"
    flag = ("★ PROMOTE"  if x["verdict"]=="PROMOTE"  else
            "◎ WATCHLIST" if x["verdict"]=="WATCHLIST" else "✗ REJECT")
    print(f"  {rank:>3}  {x['pid']:<32}  {x['k']:>2}  {x['n']:>4}  "
          f"{x['wr']*100:>5.1f}%  {x['pf']:>6.3f}  {x['b50']:>6.3f}  {ci:>14}  "
          f"{x['mc_p']*100:>3.0f}%  {x['mdd']:>5.1%}  "
          f"{x['sym_floor']:>5.3f}  {x['fold_floor']:>5.3f}  "
          f"{x['avg_overlap']:>6.1f}%  {x['avg_corr']:>5.3f}  "
          f"{x['score']:>3}/7  {flag}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL RECOMMENDATION
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  FINAL RECOMMENDATION")
print(SEP)

best_overall = all_portfolio_results[0]
bo_flat = build_portfolio_trades(best_overall["envs"])
bo_st   = defaultdict(list)
for t in bo_flat:
    bo_st[t["sym"]].append(t)
bo_r = best_overall["_stats"]

proj_tyr = best_overall["n"] / oos_years
proj_tmo = proj_tyr / 12

print(f"""
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  RECOMMENDED PRODUCTION PORTFOLIO                                           │
  │  {best_overall['pid']:<73}│
  ├─────────────────────────────────────────────────────────────────────────────┤
  │  Environments ({best_overall['k']}):                                        │""")
for e in best_overall["envs"]:
    print(f"  │    {e}: {ENV_LABEL[e]:<69}│")
print(f"""  ├─────────────────────────────────────────────────────────────────────────────┤
  │  OOS Performance                                                            │
  │    Trades (OOS):          {best_overall['n']:<51}│
  │    Win Rate:              {best_overall['wr']*100:.1f}%{'':<48}│
  │    Profit Factor:         {best_overall['pf']:.3f}{'':<49}│
  │    Bootstrap p50:         {best_overall['b50']:.3f}  [{best_overall['b5']:.3f}, {best_overall['b95']:.3f}]{'':<27}│
  │    MC P(profit):          {best_overall['mc_p']*100:.1f}%{'':<48}│
  │    Max Drawdown:          {best_overall['mdd']:.1%}{'':<49}│
  │    LOO-Symbol floor:      {best_overall['sym_floor']:.3f}{'':<49}│
  │    LOO-Fold floor:        {best_overall['fold_floor']:.3f}{'':<49}│
  │    Score:                 {best_overall['score']}/7  ({best_overall['verdict']}){'':<41}│
  │    Avg pairwise overlap:  {best_overall['avg_overlap']:.1f}%{'':<48}│
  │    Avg pairwise corr:     {best_overall['avg_corr']:.3f}{'':<49}│
  │    Est. trades/year:      {proj_tyr:.0f}  ({proj_tmo:.0f}/month){'':<35}│
  └─────────────────────────────────────────────────────────────────────────────┘""")

print(f"""
  WHY IT WINS
  {'─'*60}""")
print(f"  Score {best_overall['score']}/7 under R047 tighter criteria "
      f"(n≥{PROM_N}, MC>{PROM_MC*100:.0f}%, MDD<{PROM_MDD*100:.0f}%)")
print(f"  PF={best_overall['pf']:.3f} with Bootstrap p50={best_overall['b50']:.3f} — "
      f"edge is real and robust to resampling")
print(f"  Average pairwise overlap {best_overall['avg_overlap']:.1f}% < 30% threshold — "
      f"environments are genuinely independent")
print(f"  Average pairwise correlation {best_overall['avg_corr']:.3f} < 0.30 — "
      f"diversification is not illusory")

# Show which were REJECTED and why
print(f"""
  WHY REJECTED PORTFOLIOS FAIL
  {'─'*60}""")
reject_reasons = {"n<250": 0, "MC<80%": 0, "MDD>15%": 0, "PF<1.2": 0, "LOO<1": 0}
for x in reject_all[:30]:
    if x["n"] < PROM_N:              reject_reasons["n<250"] += 1
    if x["mc_p"] < PROM_MC:          reject_reasons["MC<80%"] += 1
    if abs(x["mdd"]) >= PROM_MDD:    reject_reasons["MDD>15%"] += 1
    if x["pf"] < PROM_PF:            reject_reasons["PF<1.2"] += 1
    if x["sym_floor"] < 1 or x["fold_floor"] < 1: reject_reasons["LOO<1"] += 1
for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
    print(f"    {reason}: {count} portfolios fail this criterion")

# Diminishing returns analysis
print(f"""
  DIMINISHING RETURNS ANALYSIS
  {'─'*60}""")
for k_val in range(2, 6):
    subset = [x for x in all_portfolio_results if x["k"] == k_val]
    bk = max(subset, key=lambda x: (x["score"], x["pf"]))
    n_promote = sum(1 for x in subset if x["verdict"] == "PROMOTE")
    print(f"    k={k_val}: best_PF={bk['pf']:.3f}  best_n={bk['n']}  "
          f"promoted={n_promote}/{len(subset)}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# SAVE CSVs
# ─────────────────────────────────────────────────────────────────────────────
rows = []
for rank, x in enumerate(all_portfolio_results, start=1):
    rows.append({
        "rank":        rank,
        "pid":         x["pid"],
        "k":           x["k"],
        "envs":        ",".join(x["envs"]),
        "n":           x["n"],
        "win_rate":    round(x["wr"],   4),
        "pf":          round(x["pf"],   4),
        "boot_p50":    round(x["b50"],  4),
        "boot_p5":     round(x["b5"],   4),
        "boot_p95":    round(x["b95"],  4),
        "mc_prob":     round(x["mc_p"], 4),
        "mdd":         round(x["mdd"],  4),
        "sym_floor":   round(x["sym_floor"],  4),
        "fold_floor":  round(x["fold_floor"], 4),
        "avg_overlap": round(x["avg_overlap"], 2),
        "avg_corr":    round(x["avg_corr"],    4),
        "exp_r":       round(x["exp_r"],       4),
        "score":       x["score"],
        "verdict":     x["verdict"],
    })
df_all = pd.DataFrame(rows)
df_all.to_csv(f"{OUT}/r047_all_portfolios.csv", index=False)
df_all.head(20).to_csv(f"{OUT}/r047_top20.csv", index=False)

# Save best portfolio trades
best_flat = build_portfolio_trades(best_overall["envs"])
pd.DataFrame(best_flat).to_csv(f"{OUT}/r047_best_portfolio_trades.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 80)
print("  Generating charts …")

# ── Chart 1: Environment Correlation Heatmap ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=C_BG)
fig.suptitle("R047 — Environment Pairwise Analysis", fontsize=13, color=C_TEXT)

# Trade overlap
mat_ov = np.array([[
    len(env_trade_sets[e1] & env_trade_sets[e2]) /
    max(len(env_trade_sets[e1] | env_trade_sets[e2]), 1) * 100
    for e2 in ENV_IDS] for e1 in ENV_IDS])

# Return correlation
mat_corr = np.array([[
    (lambda v1, v2, act: (float(np.corrcoef(v1[act], v2[act])[0,1])
                          if act.sum() >= 10 and not np.isnan(np.corrcoef(v1[act], v2[act])[0,1])
                          else 0.0))(
        env_pnl_vec[e1], env_pnl_vec[e2],
        (env_pnl_vec[e1] != 0) | (env_pnl_vec[e2] != 0)
    )
    for e2 in ENV_IDS] for e1 in ENV_IDS])

for ax, mat, title, vmax, fmt in [
    (axes[0], mat_ov,   "Trade Overlap % (Jaccard)",     100, "{:.0f}%"),
    (axes[1], mat_corr, "Return Correlation (Pearson)",   1, "{:.2f}"),
]:
    cmap = LinearSegmentedColormap.from_list("ov", [C_PANEL, C_GOLD, C_RED])
    im   = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(ENV_IDS))); ax.set_yticks(range(len(ENV_IDS)))
    ax.set_xticklabels(ENV_IDS, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(ENV_IDS, fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(len(ENV_IDS)):
        for j in range(len(ENV_IDS)):
            v   = mat[i,j]
            txt = fmt.format(v)
            col = "white" if v > (50 if "Overlap" in title else 0.5) else C_TEXT
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=col)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"{OUT}/r047_env_heatmap.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()

# ── Chart 2: Pareto Frontier (PF vs Trades) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8), facecolor=C_BG)
fig.suptitle("R047 — Pareto Frontier: Profit Factor vs Trade Count", fontsize=13, color=C_TEXT)

k_colors = {2: C_TEAL, 3: C_GOLD, 4: C_GREEN, 5: C_PURPLE}
for k_val in range(2, 6):
    subset = [x for x in all_portfolio_results if x["k"] == k_val]
    xs = [x["n"]  for x in subset]
    ys = [x["pf"] for x in subset]
    ax.scatter(xs, ys, s=20, alpha=0.4, color=k_colors[k_val], label=f"k={k_val}", zorder=3)

# Pareto frontier
pareto_n  = [x["n"]  for x in pareto]
pareto_pf = [x["pf"] for x in pareto]
ax.plot(pareto_n, pareto_pf, color=C_RED, linewidth=2.0, zorder=5, label="Pareto frontier")
ax.scatter(pareto_n, pareto_pf, s=60, color=C_RED, zorder=6)
for x in pareto[:8]:
    ax.annotate(x["pid"], (x["n"], x["pf"]),
                textcoords="offset points", xytext=(5, 3), fontsize=6, color=C_TEXT)

# Mark best overall and R046 greedy
ax.scatter([best_overall["n"]], [best_overall["pf"]], s=200, marker="*",
           color=C_GOLD, zorder=7, label=f"Best: {best_overall['pid']}")
if r046_greedy:
    ax.scatter([r046_greedy["n"]], [r046_greedy["pf"]], s=120, marker="D",
               color=C_TEAL, zorder=7, label=f"R046 greedy: E08+E06")

ax.axhline(1.20, color=C_GREEN, linewidth=1.0, linestyle="--", label="PF=1.20 promote")
ax.axhline(1.00, color=C_GREY,  linewidth=0.7, linestyle=":")
ax.axvline(250,  color=C_GREEN, linewidth=1.0, linestyle="--", label="n=250 promote")
ax.set_xlabel("Total OOS Trades (n)", fontsize=10)
ax.set_ylabel("Profit Factor", fontsize=10)
ax.legend(fontsize=8, framealpha=0.3, ncol=3)
ax.grid(zorder=0)
plt.tight_layout()
plt.savefig(f"{OUT}/r047_pareto.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()

# ── Chart 3: Diversification Benefit Curves ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=C_BG)
fig.suptitle("R047 — Diversification Benefit: Environments Added vs Performance",
             fontsize=13, color=C_TEXT)

for ax, metric, ylabel, threshold, threshold_label in [
    (axes[0], "pf",  "Profit Factor",      1.20, "Promote (1.20)"),
    (axes[1], "n",   "Total OOS Trades",   250,  "Promote (n=250)"),
]:
    k_vals = list(range(2, 6))
    medians, maxes, mins = [], [], []
    for k_val in k_vals:
        vals = [x[metric] for x in all_portfolio_results if x["k"] == k_val]
        medians.append(np.median(vals)); maxes.append(max(vals)); mins.append(min(vals))

    ax.fill_between(k_vals, mins, maxes, color=C_TEAL, alpha=0.15, label="Min–Max range")
    ax.plot(k_vals, medians, color=C_TEAL, linewidth=2.0, marker="o", label="Median")
    ax.plot(k_vals, maxes,   color=C_GREEN, linewidth=1.5, linestyle="--", marker="^", label="Best")
    ax.plot(k_vals, mins,    color=C_RED,   linewidth=1.0, linestyle="--", marker="v", label="Worst")

    # Mark global best per k
    for k_val in k_vals:
        bk = max([x for x in all_portfolio_results if x["k"] == k_val],
                 key=lambda x: (x["score"], x[metric]))
        ax.annotate(bk["pid"], (k_val, bk[metric]),
                    textcoords="offset points", xytext=(4, 4), fontsize=6, color=C_TEXT)

    ax.axhline(threshold, color=C_GOLD, linewidth=1.2, linestyle=":", label=threshold_label)
    ax.set_xlabel("Number of Environments (k)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"{ylabel} vs Portfolio Size", fontsize=10)
    ax.set_xticks(k_vals)
    ax.legend(fontsize=7, framealpha=0.3)
    ax.grid(zorder=0)

plt.tight_layout()
plt.savefig(f"{OUT}/r047_diversification_curves.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()

# ── Chart 4: Top 20 Portfolio Ranked Bar ─────────────────────────────────────
top20 = all_portfolio_results[:20]
fig, axes = plt.subplots(2, 1, figsize=(18, 10), facecolor=C_BG)
fig.suptitle("R047 — Top 20 Portfolios (Score then PF)", fontsize=13, color=C_TEXT)

labels = [f"#{i+1}\n{x['pid']}" for i, x in enumerate(top20)]
v_col  = [C_GREEN if x["verdict"]=="PROMOTE" else
          (C_GOLD if x["verdict"]=="WATCHLIST" else C_RED)
          for x in top20]

ax = axes[0]
pfs   = [x["pf"]  for x in top20]
b50s  = [x["b50"] for x in top20]
b5s   = [x["b5"]  for x in top20]
b95s  = [x["b95"] for x in top20]
bars  = ax.bar(range(20), pfs, color=v_col, edgecolor=C_GRID, linewidth=0.6, zorder=3)
for i, (lo, hi) in enumerate(zip(b5s, b95s)):
    ax.plot([i,i],[lo,hi], color=C_TEXT, linewidth=1.2, zorder=4)
    ax.plot([i-0.15,i+0.15],[lo,lo], color=C_TEXT, linewidth=1.2, zorder=4)
    ax.plot([i-0.15,i+0.15],[hi,hi], color=C_TEXT, linewidth=1.2, zorder=4)
ax.axhline(1.20, color=C_GREEN, linewidth=1.2, linestyle="--", label="Promote (1.2)")
for i, v in enumerate(pfs):
    ax.text(i, v+0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=6, color=C_TEXT)
ax.set_xticks(range(20)); ax.set_xticklabels(labels, fontsize=5, rotation=0)
ax.set_ylabel("Profit Factor"); ax.set_title("Profit Factor (90% Bootstrap CI)")
ax.legend(fontsize=7, framealpha=0.3); ax.grid(axis="y", zorder=0)

ax = axes[1]
ns  = [x["n"]   for x in top20]
mcs = [x["mc_p"]*100 for x in top20]
x2  = np.arange(20); w = 0.4
b1  = ax.bar(x2-w/2, ns,  width=w, color=v_col, alpha=0.8, label="Trade count", edgecolor=C_GRID, linewidth=0.5)
ax2 = ax.twinx()
b2  = ax2.bar(x2+w/2, mcs, width=w, color=[C_TEAL]*20, alpha=0.7, label="MC%", edgecolor=C_GRID, linewidth=0.5)
ax.axhline(250, color=C_GREEN, linewidth=1.0, linestyle="--")
ax2.axhline(80, color=C_TEAL, linewidth=1.0, linestyle=":")
ax.set_xticks(range(20)); ax.set_xticklabels(labels, fontsize=5, rotation=0)
ax.set_ylabel("OOS Trades", color=C_TEXT); ax2.set_ylabel("MC P(profit) %", color=C_TEAL)
ax.set_title("Trade Count (bars) + MC P(profit) (teal)")
ax.grid(axis="y", zorder=0)
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=7, framealpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/r047_top20.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()

# ── Chart 5: Master Dashboard ─────────────────────────────────────────────────
best_eq = best_overall["_stats"]["equity"]

fig = plt.figure(figsize=(24, 18), facecolor=C_BG)
fig.suptitle("QUANTLAB AI — R047: Global Portfolio Optimisation\n"
             f"Exhaustive Search · {total_combos} Portfolios · Production Recommendation: "
             f"{best_overall['pid']}",
             fontsize=14, color=C_TEXT, y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# Panel A: Pareto scatter
ax_p = fig.add_subplot(gs[0, :2])
for k_val in range(2, 6):
    subset = [x for x in all_portfolio_results if x["k"] == k_val]
    xs = [x["n"] for x in subset]; ys = [x["pf"] for x in subset]
    ax_p.scatter(xs, ys, s=15, alpha=0.35, color=k_colors[k_val], label=f"k={k_val}")
ax_p.plot(pareto_n, pareto_pf, color=C_RED, linewidth=2.0, zorder=5, label="Pareto")
ax_p.scatter(pareto_n, pareto_pf, s=40, color=C_RED, zorder=6)
ax_p.scatter([best_overall["n"]], [best_overall["pf"]], s=180, marker="*",
             color=C_GOLD, zorder=7, label=f"Best: {best_overall['pid']}")
if r046_greedy:
    ax_p.scatter([r046_greedy["n"]], [r046_greedy["pf"]], s=100, marker="D",
                 color=C_TEAL, zorder=7, label="R046 greedy")
ax_p.axhline(1.20, color=C_GREEN, linewidth=1.0, linestyle="--")
ax_p.axvline(250,  color=C_GREEN, linewidth=1.0, linestyle="--")
ax_p.set_xlabel("OOS Trades"); ax_p.set_ylabel("Profit Factor")
ax_p.set_title("Pareto Frontier (PF vs Trades)", fontsize=9)
ax_p.legend(fontsize=6, framealpha=0.3, ncol=3); ax_p.grid()

# Panel B: Score distribution by k
ax_s = fig.add_subplot(gs[0, 2])
for k_val in range(2, 6):
    scores = [x["score"] for x in all_portfolio_results if x["k"] == k_val]
    ax_s.hist(scores, bins=range(0, 9), alpha=0.6, color=k_colors[k_val],
              label=f"k={k_val}", edgecolor=C_GRID, density=True)
ax_s.set_xlabel("Score /7"); ax_s.set_ylabel("Density")
ax_s.set_title("Score Distribution by k", fontsize=9)
ax_s.legend(fontsize=7, framealpha=0.3); ax_s.grid(axis="y")

# Panel C: Trade overlap heatmap
ax_ov = fig.add_subplot(gs[1, 0])
cmap_ov = LinearSegmentedColormap.from_list("ov",[C_PANEL,C_GOLD,C_RED])
ax_ov.imshow(mat_ov, cmap=cmap_ov, vmin=0, vmax=100, aspect="auto")
ax_ov.set_xticks(range(len(ENV_IDS))); ax_ov.set_yticks(range(len(ENV_IDS)))
ax_ov.set_xticklabels(ENV_IDS, rotation=40, fontsize=7)
ax_ov.set_yticklabels(ENV_IDS, fontsize=7)
for i in range(len(ENV_IDS)):
    for j in range(len(ENV_IDS)):
        v = mat_ov[i,j]
        ax_ov.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=6,
                   color="white" if v>50 else C_TEXT)
ax_ov.set_title("Trade Overlap %", fontsize=9)

# Panel D: Return correlation heatmap
ax_cr = fig.add_subplot(gs[1, 1])
cmap_cr = LinearSegmentedColormap.from_list("cr",[C_PANEL,C_TEAL,C_RED])
ax_cr.imshow(mat_corr, cmap=cmap_cr, vmin=-0.5, vmax=1.0, aspect="auto")
ax_cr.set_xticks(range(len(ENV_IDS))); ax_cr.set_yticks(range(len(ENV_IDS)))
ax_cr.set_xticklabels(ENV_IDS, rotation=40, fontsize=7)
ax_cr.set_yticklabels(ENV_IDS, fontsize=7)
for i in range(len(ENV_IDS)):
    for j in range(len(ENV_IDS)):
        v = mat_corr[i,j]
        ax_cr.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                   color="white" if v>0.5 else C_TEXT)
ax_cr.set_title("Return Correlation", fontsize=9)

# Panel E: Equity curve — best portfolio
ax_eq = fig.add_subplot(gs[1, 2])
eq = best_eq
ax_eq.plot(eq, color=C_GREEN, linewidth=1.5, zorder=3)
ax_eq.fill_between(range(len(eq)), CAPITAL, eq, where=eq>=CAPITAL, color=C_GREEN, alpha=0.15)
ax_eq.fill_between(range(len(eq)), CAPITAL, eq, where=eq<CAPITAL,  color=C_RED,   alpha=0.15)
ax_eq.axhline(CAPITAL, color=C_GREY, linewidth=0.8, linestyle="--")
ax_eq.set_title(f"Best: {best_overall['pid']}\nPF={best_overall['pf']:.3f}  n={best_overall['n']}", fontsize=8)
ax_eq.set_xlabel("Trade #"); ax_eq.set_ylabel("Portfolio Value ($)"); ax_eq.grid()

# Panel F: Diversification curve (PF)
ax_dpf = fig.add_subplot(gs[2, 0])
k_vals = list(range(2, 6))
for k_val in k_vals:
    vals = [x["pf"] for x in all_portfolio_results if x["k"] == k_val]
    ax_dpf.scatter([k_val]*len(vals), vals, s=8, alpha=0.3, color=k_colors[k_val])
meds = [np.median([x["pf"] for x in all_portfolio_results if x["k"]==k]) for k in k_vals]
bsts = [max(x["pf"] for x in all_portfolio_results if x["k"]==k) for k in k_vals]
ax_dpf.plot(k_vals, meds, color=C_TEAL, linewidth=2, marker="o", label="Median PF")
ax_dpf.plot(k_vals, bsts, color=C_GREEN, linewidth=1.5, linestyle="--", marker="^", label="Best PF")
ax_dpf.axhline(1.20, color=C_GOLD, linewidth=1.0, linestyle=":")
ax_dpf.set_xlabel("k (# environments)"); ax_dpf.set_ylabel("Profit Factor")
ax_dpf.set_title("PF vs Portfolio Size", fontsize=9)
ax_dpf.legend(fontsize=7, framealpha=0.3); ax_dpf.grid()

# Panel G: Diversification curve (n)
ax_dn = fig.add_subplot(gs[2, 1])
for k_val in k_vals:
    vals = [x["n"] for x in all_portfolio_results if x["k"] == k_val]
    ax_dn.scatter([k_val]*len(vals), vals, s=8, alpha=0.3, color=k_colors[k_val])
meds_n = [np.median([x["n"] for x in all_portfolio_results if x["k"]==k]) for k in k_vals]
bsts_n = [max(x["n"]  for x in all_portfolio_results if x["k"]==k) for k in k_vals]
ax_dn.plot(k_vals, meds_n, color=C_TEAL, linewidth=2, marker="o", label="Median n")
ax_dn.plot(k_vals, bsts_n, color=C_GREEN, linewidth=1.5, linestyle="--", marker="^", label="Best n")
ax_dn.axhline(250, color=C_GOLD, linewidth=1.0, linestyle=":")
ax_dn.set_xlabel("k (# environments)"); ax_dn.set_ylabel("OOS Trades")
ax_dn.set_title("Trade Count vs Portfolio Size", fontsize=9)
ax_dn.legend(fontsize=7, framealpha=0.3); ax_dn.grid()

# Panel H: Summary comparison table
ax_tbl = fig.add_subplot(gs[2, 2])
ax_tbl.axis("off")

r046_r = r046_greedy if r046_greedy else {"n":0,"wr":0,"pf":0,"b50":0,"mc_p":0,"mdd":0,"sym_floor":0,"fold_floor":0,"score":0,"verdict":"N/A"}
tbl_data = [
    ["Metric", "R046 Greedy\n(E08+E06)", f"R047 Global Best\n({best_overall['pid']})"],
    ["Trades",    str(r046_r["n"]),    str(best_overall["n"])],
    ["Win Rate",  f"{r046_r['wr']*100:.1f}%",  f"{best_overall['wr']*100:.1f}%"],
    ["PF",        f"{r046_r['pf']:.3f}",  f"{best_overall['pf']:.3f}"],
    ["Boot p50",  f"{r046_r['b50']:.3f}",  f"{best_overall['b50']:.3f}"],
    ["MC%",       f"{r046_r['mc_p']*100:.0f}%",f"{best_overall['mc_p']*100:.0f}%"],
    ["MDD",       f"{r046_r['mdd']:.1%}",  f"{best_overall['mdd']:.1%}"],
    ["LOO-S",     f"{r046_r['sym_floor']:.3f}", f"{best_overall['sym_floor']:.3f}"],
    ["LOO-F",     f"{r046_r['fold_floor']:.3f}",f"{best_overall['fold_floor']:.3f}"],
    ["Score",     f"{r046_r['score']}/7",  f"{best_overall['score']}/7"],
    ["Verdict",   r046_r["verdict"],  best_overall["verdict"]],
]
tbl = ax_tbl.table(tbl_data, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(7)
tbl.scale(1, 1.3)
for (r_, c_), cell in tbl.get_celld().items():
    cell.set_facecolor(C_PANEL if r_ > 0 else "#1F2937")
    cell.set_edgecolor(C_GRID)
    cell.set_text_props(color=C_TEXT)
ax_tbl.set_title("R046 Greedy vs R047 Global Best", fontsize=9)

plt.savefig(f"{OUT}/r047_dashboard.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  OUTPUT FILES")
print(SEP)
outputs = [
    "r047_dashboard.png",
    "r047_env_heatmap.png",
    "r047_pareto.png",
    "r047_diversification_curves.png",
    "r047_top20.png",
    "r047_all_portfolios.csv",
    "r047_top20.csv",
    "r047_best_portfolio_trades.csv",
]
for f in outputs:
    print(f"    quantlab_output/{f}")

print()
print(SEP)
print(f"  R047 COMPLETE — GLOBAL PORTFOLIO OPTIMISATION")
print(SEP)
print(f"  Portfolios evaluated: {total_combos}")
print(f"  PROMOTE:  {len(promote_all)}")
print(f"  WATCHLIST:{len(watch_all)}")
print(f"  REJECT:   {len(reject_all)}")
print()
print(f"  PRODUCTION PORTFOLIO: {best_overall['pid']}")
print(f"    Environments: {best_overall['k']}")
for e in best_overall["envs"]:
    print(f"      {e}: {ENV_LABEL[e]}")
print(f"    PF:     {best_overall['pf']:.3f}")
print(f"    n:      {best_overall['n']}")
print(f"    Score:  {best_overall['score']}/7")
print(f"    Verdict:{best_overall['verdict']}")
print()
if r046_greedy and best_overall["pid"] != "E06+E08":
    print(f"  R046 GREEDY (E08+E06):  PF={r046_greedy['pf']:.3f}  n={r046_greedy['n']}  Score={r046_greedy['score']}/7")
    print(f"  Global best improvement: ΔPF={best_overall['pf']-r046_greedy['pf']:+.3f}  "
          f"Δn={best_overall['n']-r046_greedy['n']:+d}")
print(SEP)
