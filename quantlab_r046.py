"""
=============================================================================
QUANTLAB AI — RESEARCH #046
Independent Environment Portfolio Validation
=============================================================================

Objective:
  Determine whether a portfolio of independent environments from R042 achieves
  materially higher annual trade frequency while preserving PF > 1.20 and
  statistical robustness.

Rules:
  • Environments used EXACTLY as discovered in R042 (no threshold changes)
  • Identical RELVOL entry: rel_vol > 1.5, bullish candle, close > prev_close
  • Identical exits and risk management (R043/R044 spec)
  • 5-fold expanding walk-forward · OOS only · 23 R043 symbols · 1H
  • Priority cascade de-duplication within each candidate portfolio
  • No look-ahead, no parameter tuning

Research questions:
  Q1  Top 10 independent environments ranked by R042 PF (excl. Portfolio C)
  Q2  Individual environment full validation (RELVOL entry, unchanged)
  Q3  7-point scorecard per environment
  Q4  Pairwise overlap matrix (trade overlap %, return correlation, session overlap)
  Q5  Greedy portfolio construction (overlap<30%, corr<0.30, PF↑, n↑)
  Q6  Best single env vs multi-environment portfolio
  Q7  Final recommendations: PROMOTE / WATCHLIST / REJECT per env

Environments tested (R042 ranks 5–16, excluding Port C E1-E4):
  E05: ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0      (ATR_MD|PRG_VH|RV_LO|SLP_DN)
  E06: ATR<p25 · Mon-Tue · BodyPct>p60 · Slope<0          (ATR_LO|EARLY|PBP_HI|SLP_DN)
  E07: ATR>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)        (ATR_HI|DST_FR|MIDWK|US)
  E08: Dist>p60 · Wed-Thu · BodyPct>p60 · US(14-21UTC)    (DST_MD|MIDWK|PBP_HI|US)
  E09: ADX>p67 · Dist>p75 · BodyPct>p60 · US(14-21UTC)   (ADX_ST|DST_FR|PBP_HI|US)
  E10: ATR<p40 · Dist<p33 · PrevRng>p67 · RealVol<p33    (ATR_MD|DST_NR|PRG_HI|RV_LO)
  E11: ADX>p50 · Dist>p75 · Wed-Thu · US(14-21UTC)        (ADX_TR|DST_FR|MIDWK|US)
  E12: Dist<p33 · PrevRng>p80 · RealVol<p33 · US(14-21) (DST_NR|PRG_VH|RV_LO|US)
  E15: ADX>p67 · Dist>p75 · Wed-Thu · BodyPct>p60        (ADX_ST|DST_FR|MIDWK|PBP_HI)
  E16: Dist>p60 · Wed-Thu · PrevBody>p67 · US(14-21UTC)  (DST_MD|MIDWK|PBD_HI|US)
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

RESEARCH_ID = "R046"
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
    "E05": "#F5A623","E06": "#00C4CC","E07": "#E84545","E08": "#4BB543",
    "E09": "#9B59B6","E10": "#2E86AB","E11": "#FF6B6B","E12": "#4ECDC4",
    "E15": "#45B7D1","E16": "#96CEB4",
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
# R043 SYMBOL UNIVERSE (23 symbols — same as original research)
# ─────────────────────────────────────────────────────────────────────────────
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

PROM_PF   = 1.20;  PROM_N    = 200;  PROM_BOOT = 1.20
PROM_MC   = 0.60;  PROM_MDD  = 0.25

# ─────────────────────────────────────────────────────────────────────────────
# Q1 — TOP 10 INDEPENDENT ENVIRONMENTS (R042 ranks 5–16, excl. Port C)
#       Ranks 13 & 14 are identical envs (same n/PF); rank 14 skipped.
# ─────────────────────────────────────────────────────────────────────────────
# Portfolio C environments (excluded):
PORT_C_COND_IDS = {
    "DST_FR|MIDWK|PBP_HI|US",   # E1
    "ADX_ST|DST_FR|MIDWK|US",   # E2
    "DST_FR|MIDWK|PBD_HI|US",   # E3
    "ADX_ST|DST_MD|MIDWK|US",   # E4
}

R046_ENVS = [
    # (id, R042_rank, label, cond_ids_tuple)
    ("E05", 5,  "ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0",
     ("ATR_MD","PRG_VH","RV_LO","SLP_DN")),
    ("E06", 6,  "ATR<p25 · Mon-Tue · BodyPct>p60 · Slope<0",
     ("ATR_LO","EARLY","PBP_HI","SLP_DN")),
    ("E07", 7,  "ATR>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)",
     ("ATR_HI","DST_FR","MIDWK","US")),
    ("E08", 8,  "Dist>p60 · Wed-Thu · BodyPct>p60 · US(14-21UTC)",
     ("DST_MD","MIDWK","PBP_HI","US")),
    ("E09", 9,  "ADX>p67 · Dist>p75 · BodyPct>p60 · US(14-21UTC)",
     ("ADX_ST","DST_FR","PBP_HI","US")),
    ("E10", 10, "ATR<p40 · Dist<p33 · PrevRng>p67 · RealVol<p33",
     ("ATR_MD","DST_NR","PRG_HI","RV_LO")),
    ("E11", 11, "ADX>p50 · Dist>p75 · Wed-Thu · US(14-21UTC)",
     ("ADX_TR","DST_FR","MIDWK","US")),
    ("E12", 12, "Dist<p33 · PrevRng>p80 · RealVol<p33 · US(14-21UTC)",
     ("DST_NR","PRG_VH","RV_LO","US")),
    ("E15", 15, "ADX>p67 · Dist>p75 · Wed-Thu · BodyPct>p60",
     ("ADX_ST","DST_FR","MIDWK","PBP_HI")),
    ("E16", 16, "Dist>p60 · Wed-Thu · PrevBody>p67 · US(14-21UTC)",
     ("DST_MD","MIDWK","PBD_HI","US")),
]

ENV_IDS   = [e[0] for e in R046_ENVS]
ENV_RANK  = {e[0]: e[1] for e in R046_ENVS}
ENV_LABEL = {e[0]: e[2] for e in R046_ENVS}
ENV_CONDS = {e[0]: e[3] for e in R046_ENVS}

# Session/day tags for session-overlap reporting
ENV_SESSION = {
    "E05": {"NONE"},     "E06": {"EARLY"},      "E07": {"US","MIDWK"},
    "E08": {"US","MIDWK"},"E09": {"US"},         "E10": {"NONE"},
    "E11": {"US","MIDWK"},"E12": {"US"},          "E15": {"MIDWK"},
    "E16": {"US","MIDWK"},
}

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION CATALOGUE  (identical to R042/R043/R044)
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

NEEDED_CONDS = sorted({cid for e in R046_ENVS for cid in e[3]})
QUANT_FEATS = [
    "atr_rank","real_vol_20","bb_width","ema_dist_pct",
    "adx14","prev_range_r","prev_body_r","prev_body_pct",
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (identical to R042/R043/R044)
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c  = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]
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
# THRESHOLD LEARNING & CONDITION MASKS  (identical to R043)
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
# RELVOL SIGNAL  (identical to R043 — do not change)
# ─────────────────────────────────────────────────────────────────────────────
def signal_relvol(df, emask):
    rv = df["rel_vol"].values
    c  = df["close"].values
    o  = df["open"].values
    pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & emask

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY CASCADE  (identical to R043)
# ─────────────────────────────────────────────────────────────────────────────
def portfolio_signal(env_signals):
    n        = len(env_signals[0][1])
    combined = np.zeros(n, dtype=bool)
    attr     = np.full(n, '', dtype=object)
    for eid, sig in env_signals:
        new_fires       = sig & ~combined
        combined       |= new_fires
        attr[new_fires] = eid
    return combined, attr

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE  (identical to R043)
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

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS  (identical to R043)
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
    verdict = ("PROMOTE"     if score == 7 else
               "WATCHLIST"   if score >= 5 and m["pf"] > PROM_PF else
               "INVESTIGATE" if score >= 3 else "REJECT")
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
print("  QUANTLAB AI — RESEARCH #046")
print("  Independent Environment Portfolio Validation")
print(SEP)
print()
print(f"  {len(R046_ENVS)} environments under test (R042 ranks 5–16, excl. Port C)")
print(f"  Symbols: {len(SYMBOLS)} (identical to R043)")
print(f"  Folds: {len(FOLDS)}  Entry: RELVOL (unchanged)")
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
# WALK-FORWARD  (single pass — all 10 environments computed simultaneously)
# ─────────────────────────────────────────────────────────────────────────────
env_sym_trades = {eid: defaultdict(list) for eid in ENV_IDS}

# Store per-bar PnL series for return correlation (indexed by entry_time)
env_bar_pnl = {eid: defaultdict(float) for eid in ENV_IDS}   # entry_time → pnl

fold_env_n = {eid: [] for eid in ENV_IDS}

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

        # Compute signals for all 10 environments
        env_sigs = {}
        for eid in ENV_IDS:
            em  = env_mask(df_oos, eid, thr)
            sig = signal_relvol(df_oos, em)
            env_sigs[eid] = sig

        # Individual backtests
        for eid in ENV_IDS:
            tl = run_backtest(df_oos, env_sigs[eid], sym, fold_idx, eid)
            env_sym_trades[eid][sym].extend(tl)
            fold_counts[eid] += len(tl)
            for t in tl:
                env_bar_pnl[eid][t["entry_time"]] += t["pnl"]

    counts_str = "  ".join(f"{e}={fold_counts[e]:3d}" for e in ENV_IDS)
    total_f    = sum(fold_counts.values())
    print(f"  Fold {fold_idx}  IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%   {counts_str}   TOTAL={total_f}")
    for eid in ENV_IDS:
        fold_env_n[eid].append(fold_counts[eid])

print()

# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL ENVIRONMENT STATISTICS  (Q2 & Q3)
# ─────────────────────────────────────────────────────────────────────────────
print("  Computing statistics …")
env_results = {}
for eid in ENV_IDS:
    flat = [t for tl in env_sym_trades[eid].values() for t in tl]
    env_results[eid] = {
        "id": eid, "label": ENV_LABEL[eid],
        "r042_rank": ENV_RANK[eid],
        "sym_trades": dict(env_sym_trades[eid]),
        "_flat": flat,
        **full_stats(flat, dict(env_sym_trades[eid]))
    }

# Trade sets (for overlap)
env_trade_sets = {
    eid: {(t["sym"], t["entry_time"]) for t in env_results[eid]["_flat"]}
    for eid in ENV_IDS
}

# ─────────────────────────────────────────────────────────────────────────────
# PRINT Q2/Q3: INDIVIDUAL SCORECARD
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  Q2 & Q3 — INDIVIDUAL ENVIRONMENT SCORECARD")
print(SEP)

hdr = (f"  {'ID':<5} {'R042Rk':>6}  {'n':>4}  {'WR':>6}  {'PF':>6}  "
       f"{'p50':>6}  {'90%CI':>14}  {'MC%':>5}  {'MDD':>6}  "
       f"{'LOO-S':>6}  {'LOO-F':>6}  {'Sc':>3}  Verdict")
print(hdr)
print("  " + "─" * 100)
for eid in ENV_IDS:
    r  = env_results[eid]
    ci = f"[{r['b5']:.3f},{r['b95']:.3f}]"
    flag = ("★ PROMOTE"  if r["verdict"] == "PROMOTE"  else
            "◎ WATCHLIST" if r["verdict"] == "WATCHLIST" else
            "✗ REJECT")
    print(f"  {eid:<5} {r['r042_rank']:>6}  {r['n']:>4}  {r['wr']*100:>5.1f}%  {r['pf']:>6.3f}  "
          f"{r['b50']:>6.3f}  {ci:>14}  {r['mc_p']*100:>4.0f}%  {r['mdd']:>5.1%}  "
          f"{r['sym_floor']:>6.3f}  {r['fold_floor']:>6.3f}  {r['score']:>3}/7  {flag}")
    print(f"         {ENV_LABEL[eid][:80]}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Q4 — PAIRWISE OVERLAP MATRIX
# ─────────────────────────────────────────────────────────────────────────────

# Build common time-indexed PnL series for correlation
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

# Trade overlap (Jaccard)
overlap_trade = {}
for e1 in ENV_IDS:
    for e2 in ENV_IDS:
        s1, s2  = env_trade_sets[e1], env_trade_sets[e2]
        union   = s1 | s2
        jaccard = len(s1 & s2) / max(len(union), 1)
        overlap_trade[(e1, e2)] = round(jaccard * 100, 1)

# Return correlation (Pearson on PnL vectors — only where at least one fires)
overlap_corr = {}
for e1 in ENV_IDS:
    for e2 in ENV_IDS:
        v1, v2 = env_pnl_vec[e1], env_pnl_vec[e2]
        active = (v1 != 0) | (v2 != 0)
        if active.sum() < 10:
            overlap_corr[(e1, e2)] = 0.0
            continue
        r = np.corrcoef(v1[active], v2[active])[0, 1]
        overlap_corr[(e1, e2)] = round(float(r) if not np.isnan(r) else 0.0, 3)

# Session overlap
def session_overlap(e1, e2):
    return len(ENV_SESSION.get(e1, set()) & ENV_SESSION.get(e2, set()))

# Profitable symbol overlap
def prof_sym_overlap(e1, e2):
    def prof_syms(eid):
        out = set()
        for sym, tl in env_sym_trades[eid].items():
            if tl:
                wins = sum(t["win"] for t in tl)
                gw   = sum(t["pnl"] for t in tl if t["win"])
                gl   = abs(sum(t["pnl"] for t in tl if not t["win"]))
                if safe_pf(gw, gl) > 1.0:
                    out.add(sym)
        return out
    return len(prof_syms(e1) & prof_syms(e2))

print()
print(SEP)
print("  Q4 — PAIRWISE OVERLAP MATRIX")
print(SEP)
print()
print("  Trade Overlap % (Jaccard — lower is better for diversification):")
print(f"  {'':6}", end="")
for eid in ENV_IDS:
    print(f"  {eid:>6}", end="")
print()
print("  " + "─" * 80)
for e1 in ENV_IDS:
    print(f"  {e1:<6}", end="")
    for e2 in ENV_IDS:
        v = overlap_trade[(e1,e2)]
        print(f"  {v:>5.1f}%", end="")
    print()

print()
print("  Return Correlation (Pearson on per-bar PnL — lower = more diversification):")
print(f"  {'':6}", end="")
for eid in ENV_IDS:
    print(f"  {eid:>6}", end="")
print()
print("  " + "─" * 80)
for e1 in ENV_IDS:
    print(f"  {e1:<6}", end="")
    for e2 in ENV_IDS:
        v = overlap_corr[(e1,e2)]
        flag = " *" if (e1 != e2 and v > 0.30) else "  "
        print(f"  {v:>5.3f}{flag}", end="")
    print()

print("\n  (* marks pairs with return correlation > 0.30 — potential redundancy)")

print()
print("  Session overlap (number of shared session tags):")
print(f"  {'':6}", end="")
for eid in ENV_IDS:
    print(f"  {eid:>6}", end="")
print()
print("  " + "─" * 80)
for e1 in ENV_IDS:
    print(f"  {e1:<6}", end="")
    for e2 in ENV_IDS:
        v = session_overlap(e1, e2)
        print(f"  {v:>6}", end="")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Q5 — GREEDY PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

OVERLAP_THRESH = 30.0   # max trade overlap % to add env
CORR_THRESH    = 0.30   # max return correlation to add env

def dedup_trades(trade_lists_by_env_priority):
    """
    Priority-cascade dedup over a list of (eid, trades).
    Higher priority = earlier in list. Same (sym, entry_time) pair → first env wins.
    """
    seen    = set()
    combined = []
    for eid, tl in trade_lists_by_env_priority:
        for t in tl:
            key = (t["sym"], t["entry_time"])
            if key not in seen:
                seen.add(key)
                combined.append({**t, "env": eid})
    return combined


def portfolio_overlap_with_env(port_trade_set, candidate_trade_set):
    union = port_trade_set | candidate_trade_set
    return len(port_trade_set & candidate_trade_set) / max(len(union), 1) * 100


def portfolio_corr_with_env(port_pnl_vec, cand_pnl_vec):
    active = (port_pnl_vec != 0) | (cand_pnl_vec != 0)
    if active.sum() < 10: return 0.0
    r = np.corrcoef(port_pnl_vec[active], cand_pnl_vec[active])[0, 1]
    return float(r) if not np.isnan(r) else 0.0


print()
print(SEP)
print("  Q5 — GREEDY PORTFOLIO CONSTRUCTION")
print(f"  Rules: trade_overlap < {OVERLAP_THRESH}%  ·  return_corr < {CORR_THRESH}  ·  PF↑  ·  n↑")
print(SEP)
print()

# Sort candidate envs by score desc, then PF desc
candidates = sorted(ENV_IDS, key=lambda e: (env_results[e]["score"], env_results[e]["pf"]), reverse=True)

selected   = []
rejected   = []
port_flat  = []
port_trade_set = set()
port_pnl_vec   = np.zeros(T)
current_pf     = 0.0
current_n      = 0

for eid in candidates:
    r = env_results[eid]
    if r["n"] == 0:
        rejected.append((eid, "No trades"))
        continue

    if not selected:
        # Seed with best single env
        selected.append(eid)
        port_flat      = list(r["_flat"])
        port_trade_set = set(env_trade_sets[eid])
        port_pnl_vec   = env_pnl_vec[eid].copy()
        current_pf     = r["pf"]
        current_n      = r["n"]
        print(f"  SEED  {eid}  PF={r['pf']:.3f}  n={r['n']}  {ENV_LABEL[eid]}")
        continue

    # Check overlap with current portfolio
    tov  = portfolio_overlap_with_env(port_trade_set, env_trade_sets[eid])
    corr = portfolio_corr_with_env(port_pnl_vec, env_pnl_vec[eid])

    # Simulate adding this env (dedup, priority = current order + new at end)
    candidate_trades   = [(e, env_sym_trades[e]) for e in selected] + [(eid, env_sym_trades[eid])]
    sim_flat           = []
    seen_keys          = set()
    for e, sym_trd in candidate_trades:
        for sym, tl in sym_trd.items():
            for t in tl:
                key = (t["sym"], t["entry_time"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    sim_flat.append({**t, "env": e})
    sim_m   = metrics(sim_flat)
    new_pf  = sim_m["pf"]
    new_n   = sim_m["n"]

    reasons  = []
    if tov   >= OVERLAP_THRESH: reasons.append(f"trade_overlap={tov:.1f}%≥{OVERLAP_THRESH}%")
    if corr  >= CORR_THRESH:    reasons.append(f"corr={corr:.3f}≥{CORR_THRESH}")
    if new_pf < current_pf - 0.005: reasons.append(f"PF↓ ({new_pf:.3f}<{current_pf:.3f})")
    if new_n  <= current_n:     reasons.append(f"n does not increase ({new_n}≤{current_n})")

    if reasons:
        rejected.append((eid, "; ".join(reasons)))
        print(f"  SKIP  {eid}  overlap={tov:.1f}%  corr={corr:.3f}  PF_sim={new_pf:.3f}  n_sim={new_n}")
        print(f"        Reason: {'; '.join(reasons)}")
    else:
        selected.append(eid)
        port_trade_set = seen_keys  # already computed above
        port_pnl_vec   = port_pnl_vec + env_pnl_vec[eid]
        port_flat      = sim_flat
        current_pf     = new_pf
        current_n      = new_n
        print(f"  ADD   {eid}  overlap={tov:.1f}%  corr={corr:.3f}  PF={new_pf:.3f}  n={new_n}  ✓")

print()
print(f"  Greedy portfolio: {' + '.join(selected)}")
print(f"  Rejected: {[e for e,_ in rejected]}")

# Final portfolio full stats
port_sym_trades_g = defaultdict(list)
for t in port_flat:
    port_sym_trades_g[t["sym"]].append(t)
greedy_stats = full_stats(port_flat, dict(port_sym_trades_g))
greedy_stats["envs"] = selected
greedy_stats["label"] = "Greedy Port [" + "+".join(selected) + "]"

# ─────────────────────────────────────────────────────────────────────────────
# Q6 — BEST SINGLE ENV vs MULTI-ENVIRONMENT PORTFOLIO
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  Q6 — BEST SINGLE ENVIRONMENT vs MULTI-ENVIRONMENT PORTFOLIO")
print(SEP)

# Best single env by score then PF
best_single_id = max(ENV_IDS, key=lambda e: (env_results[e]["score"], env_results[e]["pf"]))
single = env_results[best_single_id]
greedy = greedy_stats

# Average monthly / yearly trades
# OOS period spans ~50% of data (folds 1-5 progressively).
# Each symbol has ~17k-21k bars; at 1H cadence ≈ 2yr history.
# OOS portion of data ≈ 1 year per symbol across 23 symbols combined.
oos_years = 1.0   # approximate total OOS time span per symbol
n_syms    = len(SYMBOLS)

def freq_stats(n):
    trades_per_sym_yr  = n / n_syms / oos_years
    trades_per_sym_mo  = trades_per_sym_yr / 12
    portfolio_yr       = n / oos_years
    portfolio_mo       = portfolio_yr / 12
    return trades_per_sym_yr, trades_per_sym_mo, portfolio_yr, portfolio_mo

s_psyr, s_psmo, s_pyr, s_pmo = freq_stats(single["n"])
g_psyr, g_psmo, g_pyr, g_pmo = freq_stats(greedy["n"])

def fmt_comparison(label, r, sym_yr, sym_mo, port_yr, port_mo):
    print(f"  {label}")
    print(f"  {'─'*60}")
    print(f"    Trades (OOS):       {r['n']}")
    print(f"    Win Rate:           {r['wr']*100:.1f}%")
    print(f"    Profit Factor:      {r['pf']:.3f}")
    print(f"    Bootstrap Median:   {r['b50']:.3f}  [{r['b5']:.3f}, {r['b95']:.3f}]")
    print(f"    MC P(profit):       {r['mc_p']*100:.1f}%")
    print(f"    Max Drawdown:       {r['mdd']:.1%}")
    print(f"    LOO-Symbol floor:   {r['sym_floor']:.3f}")
    print(f"    LOO-Fold floor:     {r['fold_floor']:.3f}")
    print(f"    Score:              {r['score']}/7  ({r['verdict']})")
    print(f"    Trades/sym/yr:      {sym_yr:.1f}   ({sym_mo:.1f}/month)")
    print(f"    Portfolio/yr:       {port_yr:.0f}  ({port_mo:.0f}/month)")

print()
fmt_comparison(f"A) BEST SINGLE ENVIRONMENT: {best_single_id} — {ENV_LABEL[best_single_id]}",
               single, s_psyr, s_psmo, s_pyr, s_pmo)
print()
fmt_comparison(f"B) GREEDY MULTI-ENV PORTFOLIO: {greedy['label']}",
               greedy, g_psyr, g_psmo, g_pyr, g_pmo)

print()
freq_gain = ((greedy["n"] / max(single["n"], 1)) - 1) * 100
pf_delta  = greedy["pf"] - single["pf"]
print(f"  Frequency gain:      +{freq_gain:.0f}%  ({single['n']} → {greedy['n']} trades)")
print(f"  PF delta:            {pf_delta:+.3f}  ({single['pf']:.3f} → {greedy['pf']:.3f})")
print(f"  MDD delta:           {(greedy['mdd']-single['mdd']):+.1%}")
print(f"  Score delta:         {greedy['score']-single['score']:+d}/7")

# ─────────────────────────────────────────────────────────────────────────────
# Q7 — FINAL RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  Q7 — FINAL RECOMMENDATIONS")
print(SEP)

promote_envs   = [(eid, env_results[eid]) for eid in ENV_IDS if env_results[eid]["verdict"] == "PROMOTE"]
watchlist_envs = [(eid, env_results[eid]) for eid in ENV_IDS if env_results[eid]["verdict"] == "WATCHLIST"]
reject_envs    = [(eid, env_results[eid]) for eid in ENV_IDS if env_results[eid]["verdict"] in ("REJECT","INVESTIGATE")]

promote_envs.sort(key=lambda x: x[1]["pf"], reverse=True)

print(f"\n  PROMOTE ({len(promote_envs)} environments):")
for eid, r in promote_envs:
    print(f"    ★ {eid}  PF={r['pf']:.3f}  n={r['n']}  Boot={r['b50']:.3f}  "
          f"MC={r['mc_p']*100:.0f}%  {ENV_LABEL[eid]}")

print(f"\n  WATCHLIST ({len(watchlist_envs)} environments):")
for eid, r in watchlist_envs:
    print(f"    ◎ {eid}  PF={r['pf']:.3f}  n={r['n']}  Score={r['score']}/7  {ENV_LABEL[eid]}")

print(f"\n  REJECT ({len(reject_envs)} environments):")
for eid, r in reject_envs:
    print(f"    ✗ {eid}  PF={r['pf']:.3f}  n={r['n']}  Score={r['score']}/7")

top3 = [e for e,_ in promote_envs[:3]] if len(promote_envs) >= 3 else [e for e,_ in (promote_envs+watchlist_envs)[:3]]
print(f"\n  Top 3 production environments:  {', '.join(top3)}")
print(f"  Final production portfolio:     {greedy['label']}")

# Projection on production portfolio
proj_pf   = greedy["pf"]
proj_wr   = greedy["wr"] * 100
proj_mdd  = greedy["mdd"]
proj_mc   = greedy["mc_p"] * 100
proj_tyr  = g_pyr
proj_tmo  = g_pmo

print(f"""
  Expected annual trade frequency:  {proj_tyr:.0f} trades/year  ({proj_tmo:.0f}/month)
  Expected portfolio PF:            {proj_pf:.3f}
  Expected win rate:                {proj_wr:.1f}%
  Expected max drawdown:            {proj_mdd:.1%}
  MC P(profitable):                 {proj_mc:.1f}%

  Is edge stronger as diversified portfolio?
  {"YES" if greedy['n'] > single['n'] * 1.2 and greedy['pf'] > PROM_PF else "MARGINAL"}
  → Frequency gain: +{freq_gain:.0f}%  PF delta: {pf_delta:+.3f}
""")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS (CSVs)
# ─────────────────────────────────────────────────────────────────────────────
scorecard_rows = []
for eid in ENV_IDS:
    r = env_results[eid]
    scorecard_rows.append({
        "env": eid, "r042_rank": ENV_RANK[eid],
        "label": ENV_LABEL[eid],
        "n": r["n"], "win_rate": round(r["wr"],4),
        "profit_factor": round(r["pf"],4),
        "boot_p50": round(r["b50"],4), "boot_p5": round(r["b5"],4),
        "boot_p95": round(r["b95"],4), "mc_prob": round(r["mc_p"],4),
        "mdd": round(r["mdd"],4), "sym_floor": round(r["sym_floor"],4),
        "fold_floor": round(r["fold_floor"],4),
        "score": r["score"], "verdict": r["verdict"],
    })
pd.DataFrame(scorecard_rows).to_csv(f"{OUT}/r046_scorecard.csv", index=False)

overlap_rows = []
for e1 in ENV_IDS:
    for e2 in ENV_IDS:
        overlap_rows.append({
            "env_a": e1, "env_b": e2,
            "trade_overlap_pct": overlap_trade[(e1,e2)],
            "return_corr": overlap_corr[(e1,e2)],
            "session_overlap": session_overlap(e1,e2),
        })
pd.DataFrame(overlap_rows).to_csv(f"{OUT}/r046_overlap_matrix.csv", index=False)

pd.DataFrame(greedy["_flat"] if "_flat" in greedy else port_flat).to_csv(
    f"{OUT}/r046_portfolio_trades.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Chart 1: Individual Scorecard (PF + Bootstrap CI) ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=C_BG)
fig.suptitle("R046 — Individual Environment Scorecards", fontsize=13, color=C_TEXT)

for ax, metric, ylabel in zip(axes, ["pf","wr","mc_p"],
                               ["Profit Factor","Win Rate","MC P(profit)"]):
    vals   = [env_results[e][metric] * (100 if metric in ("wr","mc_p") else 1) for e in ENV_IDS]
    colors = [C_GREEN if v >= (1.2 if metric=="pf" else (50 if metric=="wr" else 60))
              else C_GOLD if v >= (1.0 if metric=="pf" else 40)
              else C_RED for v in vals]
    bars   = ax.bar(ENV_IDS, vals, color=colors, edgecolor=C_GRID, linewidth=0.7, zorder=3)
    if metric == "pf":
        lo = [env_results[e]["b5"]  for e in ENV_IDS]
        hi = [env_results[e]["b95"] for e in ENV_IDS]
        for i, (l, h) in enumerate(zip(lo, hi)):
            ax.plot([i,i],[l,h], color=C_TEXT, linewidth=1.5, zorder=4)
            ax.plot([i-0.1,i+0.1],[l,l], color=C_TEXT, linewidth=1.5, zorder=4)
            ax.plot([i-0.1,i+0.1],[h,h], color=C_TEXT, linewidth=1.5, zorder=4)
        ax.axhline(1.2, color=C_GREEN, linewidth=1.2, linestyle="--", label="Promote (1.2)")
        ax.axhline(1.0, color=C_GOLD,  linewidth=0.8, linestyle=":")
        ax.legend(fontsize=7, framealpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01*max(vals),
                f"{v:.2f}" if metric=="pf" else f"{v:.0f}%",
                ha="center", va="bottom", fontsize=7, color=C_TEXT)
    ax.set_title(ylabel, fontsize=10); ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(bottom=0); ax.grid(axis="y", zorder=0)
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(f"{OUT}/r046_scorecard_chart.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()


# ── Chart 2: Trade Overlap Heatmap ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=C_BG)
fig.suptitle("R046 — Pairwise Overlap Analysis", fontsize=13, color=C_TEXT)

for ax, data_fn, title, fmt in [
    (axes[0], lambda e1,e2: overlap_trade[(e1,e2)],  "Trade Overlap % (Jaccard)", "{:.0f}%"),
    (axes[1], lambda e1,e2: overlap_corr[(e1,e2)],   "Return Correlation (Pearson)", "{:.2f}"),
]:
    mat = np.array([[data_fn(e1,e2) for e2 in ENV_IDS] for e1 in ENV_IDS])
    cmap = LinearSegmentedColormap.from_list("ov", [C_PANEL, C_GOLD, C_RED])
    im   = ax.imshow(mat, cmap=cmap, vmin=0,
                      vmax=100 if title.startswith("Trade") else 1)
    ax.set_xticks(range(len(ENV_IDS))); ax.set_yticks(range(len(ENV_IDS)))
    ax.set_xticklabels(ENV_IDS, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(ENV_IDS, fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(len(ENV_IDS)):
        for j in range(len(ENV_IDS)):
            v  = mat[i,j]
            txt = fmt.format(v)
            col = "white" if v > (50 if title.startswith("Trade") else 0.5) else C_TEXT
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=col)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"{OUT}/r046_overlap_heatmap.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()


# ── Chart 3: Greedy Portfolio Build Waterfall ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=C_BG)
fig.suptitle("R046 — Greedy Portfolio Construction", fontsize=13, color=C_TEXT)

# Simulate cumulative PF and n as envs are added
cumulative_pf = []
cumulative_n  = []
for k in range(1, len(selected)+1):
    subset = selected[:k]
    seen_k = set()
    flat_k = []
    for eid in subset:
        for sym, tl in env_sym_trades[eid].items():
            for t in tl:
                key = (t["sym"], t["entry_time"])
                if key not in seen_k:
                    seen_k.add(key)
                    flat_k.append(t)
    m_k = metrics(flat_k)
    cumulative_pf.append(m_k["pf"])
    cumulative_n.append(m_k["n"])

x     = range(len(selected))
xlbls = [f"+{e}" for e in selected]

ax = axes[0]
ax.bar(x, cumulative_pf, color=[ENV_COLOURS.get(e, C_GREY) for e in selected],
       edgecolor=C_GRID, linewidth=0.7, zorder=3)
ax.axhline(1.2, color=C_GREEN, linewidth=1.2, linestyle="--", label="Promote (1.2)")
ax.axhline(1.0, color=C_GOLD,  linewidth=0.8, linestyle=":")
for i, v in enumerate(cumulative_pf):
    ax.text(i, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=C_TEXT)
ax.set_xticks(x); ax.set_xticklabels(xlbls, rotation=15)
ax.set_ylabel("Portfolio PF"); ax.set_title("Profit Factor Growth")
ax.grid(axis="y"); ax.legend(fontsize=7, framealpha=0.3)

ax = axes[1]
ax.bar(x, cumulative_n, color=[ENV_COLOURS.get(e, C_GREY) for e in selected],
       edgecolor=C_GRID, linewidth=0.7, zorder=3)
ax.axhline(200, color=C_GREEN, linewidth=1.2, linestyle="--", label="n=200 threshold")
for i, v in enumerate(cumulative_n):
    ax.text(i, v+1, str(v), ha="center", va="bottom", fontsize=9, color=C_TEXT)
ax.set_xticks(x); ax.set_xticklabels(xlbls, rotation=15)
ax.set_ylabel("Total OOS Trades"); ax.set_title("Trade Count Growth")
ax.grid(axis="y"); ax.legend(fontsize=7, framealpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/r046_greedy_waterfall.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()


# ── Chart 4: Equity Curves — Single vs Portfolio ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=C_BG)
fig.suptitle("R046 — Equity Curves: Best Single vs Multi-Environment Portfolio",
             fontsize=13, color=C_TEXT)

for ax, r, title, col in [
    (axes[0], single, f"Best Single: {best_single_id}", ENV_COLOURS.get(best_single_id, C_TEAL)),
    (axes[1], greedy, f"Greedy Portfolio [{'+'.join(selected)}]", C_GREEN),
]:
    eq = r["equity"]
    ax.plot(eq, color=col, linewidth=1.5, zorder=3)
    ax.fill_between(range(len(eq)), CAPITAL, eq,
                    where=(eq >= CAPITAL), color=col, alpha=0.15)
    ax.fill_between(range(len(eq)), CAPITAL, eq,
                    where=(eq < CAPITAL), color=C_RED, alpha=0.15)
    ax.axhline(CAPITAL, color=C_GREY, linewidth=0.8, linestyle="--")
    ax.set_title(f"{title}\nPF={r['pf']:.3f}  n={r['n']}  MDD={r['mdd']:.1%}", fontsize=9)
    ax.set_xlabel("Trade #"); ax.set_ylabel("Portfolio Value ($)")
    ax.grid(zorder=0)

plt.tight_layout()
plt.savefig(f"{OUT}/r046_equity_curves.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()


# ── Chart 5: Fold Stability ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=C_BG)
fig.suptitle("R046 — Fold Stability", fontsize=13, color=C_TEXT)

for ax, eid_list, title in [
    (axes[0], ENV_IDS, "Per-Environment Fold Trade Counts"),
    (axes[1], selected, "Greedy Portfolio Fold Allocation"),
]:
    fold_lbls = [f"F{i}" for i in range(1, len(FOLDS)+1)]
    x  = np.arange(len(FOLDS))
    w  = 0.8 / max(len(eid_list), 1)
    for k, eid in enumerate(eid_list):
        vals = fold_env_n[eid]
        ax.bar(x + k*w - 0.4 + w/2, vals, width=w,
               color=ENV_COLOURS.get(eid, C_GREY), label=eid,
               edgecolor=C_GRID, linewidth=0.4, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(fold_lbls)
    ax.set_ylabel("Trades"); ax.set_title(title)
    ax.grid(axis="y"); ax.legend(fontsize=7, framealpha=0.3, ncol=3)

plt.tight_layout()
plt.savefig(f"{OUT}/r046_fold_stability.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()


# ── Chart 6: Verdict Summary Bar ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7), facecolor=C_BG)
sorted_envs = sorted(ENV_IDS, key=lambda e: env_results[e]["pf"], reverse=True)
pfs   = [env_results[e]["pf"]  for e in sorted_envs]
boots = [env_results[e]["b50"] for e in sorted_envs]
lo_ci = [env_results[e]["b5"]  for e in sorted_envs]
hi_ci = [env_results[e]["b95"] for e in sorted_envs]
v_col = [C_GREEN if env_results[e]["verdict"]=="PROMOTE"
         else (C_GOLD if env_results[e]["verdict"]=="WATCHLIST" else C_RED)
         for e in sorted_envs]
bars  = ax.bar(sorted_envs, pfs, color=v_col, edgecolor=C_GRID, linewidth=0.7, zorder=3)
for i, (lo, hi) in enumerate(zip(lo_ci, hi_ci)):
    ax.plot([i,i],[lo,hi], color=C_TEXT, linewidth=1.5, zorder=4)
    ax.plot([i-0.1,i+0.1],[lo,lo], color=C_TEXT, linewidth=1.5, zorder=4)
    ax.plot([i-0.1,i+0.1],[hi,hi], color=C_TEXT, linewidth=1.5, zorder=4)
# Overlay portfolio line
ax.axhline(greedy["pf"], color=C_TEAL, linewidth=2.0, linestyle="-.",
           label=f"Greedy Portfolio PF={greedy['pf']:.3f}")
ax.axhline(single["pf"], color=C_PURPLE, linewidth=1.5, linestyle="--",
           label=f"Best Single PF={single['pf']:.3f}")
ax.axhline(1.20, color=C_GREEN, linewidth=1.0, linestyle=":", label="Promote (1.2)")
for bar, v in zip(bars, pfs):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=8, color=C_TEXT)
legend_patches = [
    mpatches.Patch(color=C_GREEN, label="PROMOTE"),
    mpatches.Patch(color=C_GOLD,  label="WATCHLIST"),
    mpatches.Patch(color=C_RED,   label="REJECT/INVESTIGATE"),
]
ax.legend(handles=legend_patches + [
    plt.Line2D([],[],color=C_TEAL,   linestyle="-.",  label=f"Greedy Portfolio PF={greedy['pf']:.3f}"),
    plt.Line2D([],[],color=C_PURPLE, linestyle="--", label=f"Best Single PF={single['pf']:.3f}"),
    plt.Line2D([],[],color=C_GREEN,  linestyle=":",   label="Promote (1.2)"),
], fontsize=8, framealpha=0.3)
ax.set_ylim(bottom=0)
ax.set_ylabel("Profit Factor (OOS)", fontsize=9)
ax.set_title("R046 — Environment Verdict Summary (sorted by PF)", fontsize=11)
ax.grid(axis="y", zorder=0)
plt.tight_layout()
plt.savefig(f"{OUT}/r046_verdict_summary.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()


# ── Chart 7: Master Dashboard ─────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16), facecolor=C_BG)
fig.suptitle("QUANTLAB AI — R046: Independent Environment Portfolio Validation\n"
             "10 Environments · Greedy Diversification · Full Walk-Forward OOS",
             fontsize=14, color=C_TEXT, y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: PF per env
ax_pf = fig.add_subplot(gs[0, :2])
pf_vals = [env_results[e]["pf"] for e in sorted_envs]
b5_vals = [env_results[e]["b5"]  for e in sorted_envs]
b95_vals= [env_results[e]["b95"] for e in sorted_envs]
bars = ax_pf.bar(sorted_envs, pf_vals, color=v_col, edgecolor=C_GRID, linewidth=0.5, zorder=3)
for i, (lo, hi) in enumerate(zip(b5_vals, b95_vals)):
    ax_pf.plot([i,i],[lo,hi], color=C_TEXT, linewidth=1.2, zorder=4)
ax_pf.axhline(1.2, color=C_GREEN, linewidth=1.0, linestyle="--")
ax_pf.axhline(greedy["pf"], color=C_TEAL, linewidth=1.5, linestyle="-.")
for bar, v in zip(bars, pf_vals):
    ax_pf.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}",
               ha="center", va="bottom", fontsize=7, color=C_TEXT)
ax_pf.set_title("Individual PF (90% Bootstrap CI)", fontsize=9)
ax_pf.grid(axis="y"); ax_pf.set_ylim(bottom=0)

# Panel B: Trade counts
ax_n = fig.add_subplot(gs[0, 2])
n_vals = [env_results[e]["n"] for e in sorted_envs]
ax_n.barh(sorted_envs, n_vals, color=v_col, edgecolor=C_GRID, linewidth=0.5)
ax_n.axvline(200, color=C_GREEN, linewidth=1.0, linestyle="--")
ax_n.set_title("Trade Count", fontsize=9); ax_n.grid(axis="x")

# Panel C: Equity curve — single
ax_eq1 = fig.add_subplot(gs[1, 0])
eq = single["equity"]
ax_eq1.plot(eq, color=C_TEAL, linewidth=1.2)
ax_eq1.fill_between(range(len(eq)), CAPITAL, eq, where=eq>=CAPITAL, color=C_TEAL, alpha=0.15)
ax_eq1.fill_between(range(len(eq)), CAPITAL, eq, where=eq<CAPITAL, color=C_RED,  alpha=0.15)
ax_eq1.axhline(CAPITAL, color=C_GREY, linewidth=0.7, linestyle="--")
ax_eq1.set_title(f"Single: {best_single_id}\nPF={single['pf']:.3f}  n={single['n']}", fontsize=8)
ax_eq1.grid()

# Panel D: Equity curve — portfolio
ax_eq2 = fig.add_subplot(gs[1, 1])
eq2 = greedy["equity"]
ax_eq2.plot(eq2, color=C_GREEN, linewidth=1.2)
ax_eq2.fill_between(range(len(eq2)), CAPITAL, eq2, where=eq2>=CAPITAL, color=C_GREEN, alpha=0.15)
ax_eq2.fill_between(range(len(eq2)), CAPITAL, eq2, where=eq2<CAPITAL,  color=C_RED,   alpha=0.15)
ax_eq2.axhline(CAPITAL, color=C_GREY, linewidth=0.7, linestyle="--")
ax_eq2.set_title(f"Portfolio: {'+'.join(selected)}\nPF={greedy['pf']:.3f}  n={greedy['n']}", fontsize=8)
ax_eq2.grid()

# Panel E: Overlap heatmap (trade)
ax_ov = fig.add_subplot(gs[1, 2])
mat_ov = np.array([[overlap_trade[(e1,e2)] for e2 in ENV_IDS] for e1 in ENV_IDS])
cmap_ov = LinearSegmentedColormap.from_list("ov",[C_PANEL,C_GOLD,C_RED])
ax_ov.imshow(mat_ov, cmap=cmap_ov, vmin=0, vmax=100, aspect="auto")
ax_ov.set_xticks(range(len(ENV_IDS))); ax_ov.set_yticks(range(len(ENV_IDS)))
ax_ov.set_xticklabels(ENV_IDS, rotation=40, fontsize=7)
ax_ov.set_yticklabels(ENV_IDS, fontsize=7)
ax_ov.set_title("Trade Overlap %", fontsize=9)

# Panel F: Greedy waterfall
ax_wf = fig.add_subplot(gs[2, :2])
ax_wf.bar(range(len(cumulative_pf)), cumulative_pf,
          color=[ENV_COLOURS.get(e, C_GREY) for e in selected],
          edgecolor=C_GRID, linewidth=0.5, zorder=3)
ax_wf.axhline(1.2, color=C_GREEN, linewidth=1.0, linestyle="--")
for i, v in enumerate(cumulative_pf):
    ax_wf.text(i, v+0.005, f"{v:.3f}  (n={cumulative_n[i]})",
               ha="center", va="bottom", fontsize=7, color=C_TEXT)
ax_wf.set_xticks(range(len(selected)))
ax_wf.set_xticklabels([f"+{e}" for e in selected], rotation=10)
ax_wf.set_title("Greedy Build: Portfolio PF as Envs Are Added", fontsize=9)
ax_wf.grid(axis="y")

# Panel G: Single vs Portfolio comparison table
ax_tbl = fig.add_subplot(gs[2, 2])
ax_tbl.axis("off")
tbl_data = [
    ["Metric", "Single", "Portfolio"],
    ["Trades", str(single["n"]), str(greedy["n"])],
    ["Win Rate", f"{single['wr']*100:.1f}%", f"{greedy['wr']*100:.1f}%"],
    ["PF", f"{single['pf']:.3f}", f"{greedy['pf']:.3f}"],
    ["Boot p50", f"{single['b50']:.3f}", f"{greedy['b50']:.3f}"],
    ["MC%", f"{single['mc_p']*100:.0f}%", f"{greedy['mc_p']*100:.0f}%"],
    ["MDD", f"{single['mdd']:.1%}", f"{greedy['mdd']:.1%}"],
    ["LOO-S", f"{single['sym_floor']:.3f}", f"{greedy['sym_floor']:.3f}"],
    ["LOO-F", f"{single['fold_floor']:.3f}", f"{greedy['fold_floor']:.3f}"],
    ["Score", f"{single['score']}/7", f"{greedy['score']}/7"],
    ["Verdict", single["verdict"], greedy["verdict"]],
]
tbl = ax_tbl.table(tbl_data, loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
tbl.scale(1, 1.4)
for (r_, c_), cell in tbl.get_celld().items():
    cell.set_facecolor(C_PANEL); cell.set_edgecolor(C_GRID)
    cell.set_text_props(color=C_TEXT)
    if r_ == 0: cell.set_facecolor("#1F2937")
ax_tbl.set_title("Single vs Portfolio", fontsize=9)

plt.savefig(f"{OUT}/r046_dashboard.png", dpi=130, bbox_inches="tight", facecolor=C_BG)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  OUTPUT FILES")
print(SEP)
outputs = [
    "r046_dashboard.png",
    "r046_scorecard_chart.png",
    "r046_overlap_heatmap.png",
    "r046_greedy_waterfall.png",
    "r046_equity_curves.png",
    "r046_fold_stability.png",
    "r046_verdict_summary.png",
    "r046_scorecard.csv",
    "r046_overlap_matrix.csv",
    "r046_portfolio_trades.csv",
]
for f in outputs:
    print(f"    quantlab_output/{f}")
print()
print(SEP)
print(f"  R046 COMPLETE")
print(f"  Environments tested:  {len(ENV_IDS)}")
print(f"  PROMOTE:              {len(promote_envs)}")
print(f"  WATCHLIST:            {len(watchlist_envs)}")
print(f"  Greedy portfolio:     {greedy['label']}")
print(f"  Portfolio PF:         {greedy['pf']:.3f}")
print(f"  Portfolio trades:     {greedy['n']}")
print(f"  Freq gain vs single:  +{freq_gain:.0f}%")
print(SEP)
