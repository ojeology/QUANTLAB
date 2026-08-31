"""
=============================================================================
QUANTLAB AI — RESEARCH #049
Universe Out-of-Sample Validation — New Symbol Universe
=============================================================================

Objective:
  R047 validated E06+E11 on 23 specific perpetual swap symbols.
  R048 tested forward in time (insufficient data yet).

  R049 tests ACROSS SPACE: run the frozen E06+E11 portfolio on 26 symbols
  that have NEVER appeared in any research from R042 to R047.

  Nothing is changed:
    • Environment E06  unchanged
    • Environment E11  unchanged
    • Entry / exit / risk  unchanged
    • Walk-forward structure  identical to R047  (5-fold expanding)
    • Thresholds learned per symbol per fold from that symbol's IS data
      (same mechanism as R047 — quantile thresholds are always re-learned
       from IS; the CONDITIONS themselves are what is frozen)

  New universe (26 symbols, never touched in R042–R047):
    1INCH AAVE ALGO AXS CHZ COMP CRV DYDX EGLD ETC
    FET GALA GMX GRT HBAR ICP IMX INJ LDO SAND
    SHIB SNX STX SUSHI TRX XLM

Research questions:
  Q1. Does E06+E11 achieve PF > 1.20 on the new universe?
  Q2. Win Rate
  Q3. Trade count
  Q4. Bootstrap PF
  Q5. Monte Carlo probability
  Q6. LOO-symbol floor
  Q7. LOO-fold floor
  Q8. Max drawdown
  Q9. Equity curve
  Q10. Compare with R047 benchmark (original 23 symbols)

Promotion criteria:  identical to R047
  PF > 1.20  ·  n ≥ 250  ·  Boot > 1.20  ·  MC > 80%
  LOO-S > 1.0  ·  LOO-F > 1.0  ·  MDD < 15%
  Score 7/7 = PROMOTE

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

RESEARCH_ID = "R049"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = CONFIG["RISK_REWARD"]
BEP_WR   = 1.0 / (1.0 + RR)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (identical to R047/R048)
# ─────────────────────────────────────────────────────────────────────────────
C_GOLD   = "#F5A623"; C_TEAL   = "#00C4CC"; C_RED    = "#E84545"
C_GREEN  = "#4BB543"; C_PURPLE = "#9B59B6"; C_BLUE   = "#2E86AB"
C_GREY   = "#888888"; C_BG     = "#0D1117"; C_PANEL  = "#161B22"
C_TEXT   = "#E6EDF3"; C_GRID   = "#21262D"

ENV_COLOURS = {"E06": "#00C4CC", "E11": "#FF6B6B"}

plt.rcParams.update({
    "figure.facecolor": C_BG,  "axes.facecolor":  C_PANEL,
    "axes.edgecolor":   C_GRID,"axes.labelcolor": C_TEXT,
    "xtick.color":      C_TEXT,"ytick.color":     C_TEXT,
    "text.color":       C_TEXT,"grid.color":      C_GRID,
    "grid.alpha":       0.4,   "axes.titlecolor": C_TEXT,
    "font.family":      "monospace",
})

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION PORTFOLIO — FROZEN FROM R047  (do not modify)
# ─────────────────────────────────────────────────────────────────────────────
PROD_ENVS = [
    ("E06", "ATR<p25 · Mon-Tue · BodyPct>p60 · Slope<0",    ("ATR_LO","EARLY","PBP_HI","SLP_DN")),
    ("E11", "ADX>p50 · Dist>p75 · Wed-Thu · US(14-21UTC)",  ("ADX_TR","DST_FR","MIDWK","US")),
]
PROD_IDS   = [e[0] for e in PROD_ENVS]
PROD_LABEL = {e[0]: e[1] for e in PROD_ENVS}
PROD_CONDS = {e[0]: e[2] for e in PROD_ENVS}

# ─────────────────────────────────────────────────────────────────────────────
# R047 BENCHMARK  (original 23-symbol universe)
# ─────────────────────────────────────────────────────────────────────────────
R047_BENCH = {
    "pid": "E06+E11", "n": 253, "pf": 1.601, "wr": 0.490,
    "boot_p50": 1.603, "mc": 1.00, "mdd": -0.053,
    "loo_s": 1.508, "loo_f": 1.358, "score": 7,
}

# ─────────────────────────────────────────────────────────────────────────────
# NEW SYMBOL UNIVERSE — never touched in R042–R047
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

# R047 promotion thresholds — kept identical
PROM_PF   = 1.20
PROM_N    = 250
PROM_BOOT = 1.20
PROM_MC   = 0.80
PROM_MDD  = 0.15

SEP  = "═" * 110
SEP2 = "─" * 80

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION CATALOGUE  (frozen)
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    ("ATR_LO", "ATR<p25",      "atr_rank",      "lt_q",      0.25, "vol"),
    ("ATR_MD", "ATR<p40",      "atr_rank",      "lt_q",      0.40, "vol"),
    ("ATR_HI", "ATR>p67",      "atr_rank",      "gt_q",      0.67, "vol"),
    ("RV_LO",  "RealVol<p33",  "real_vol_20",   "lt_q",      0.33, "vol"),
    ("SLP_UP", "Slope>0",      "ema200_slope",  "gt_fixed",  0.0,  "trend"),
    ("SLP_DN", "Slope<0",      "ema200_slope",  "lt_fixed",  0.0,  "trend"),
    ("DST_FR", "Dist>p75",     "ema_dist_pct",  "gt_q_pos",  0.75, "trend"),
    ("DST_MD", "Dist>p60",     "ema_dist_pct",  "gt_q_pos",  0.60, "trend"),
    ("ADX_TR", "ADX>p50",      "adx14",         "gt_q",      0.50, "trend"),
    ("PBP_HI", "BodyPct>p60",  "prev_body_pct", "gt_q",      0.60, "part"),
    ("US",     "US(14-21UTC)", "hour_utc",      "hour_rng",  (14,21), "time"),
    ("MIDWK",  "Wed-Thu",      "day_of_week",   "isin",      [2,3],   "time"),
    ("EARLY",  "Mon-Tue",      "day_of_week",   "isin",      [0,1],   "time"),
]
COND_BY_ID = {c[0]: c for c in CONDITIONS_DEF}

NEEDED_CONDS = sorted({cid for e in PROD_ENVS for cid in e[2]})
QUANT_FEATS  = [
    "atr_rank","real_vol_20","bb_width","ema_dist_pct",
    "adx14","prev_range_r","prev_body_r","prev_body_pct",
]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (frozen — identical to R047)
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
    conds = PROD_CONDS[eid]
    out   = condition_mask(df, conds[0], thr)
    for cid in conds[1:]:
        out &= condition_mask(df, cid, thr)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL  (frozen)
# ─────────────────────────────────────────────────────────────────────────────
def signal_relvol(df, emask):
    rv = df["rel_vol"].values
    c  = df["close"].values
    o  = df["open"].values
    pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & emask

def portfolio_signal(env_signals):
    n        = len(env_signals[0][1])
    combined = np.zeros(n, dtype=bool)
    attr     = np.full(n, "", dtype=object)
    for eid, sig in env_signals:
        new_fires       = sig & ~combined
        combined       |= new_fires
        attr[new_fires] = eid
    return combined, attr

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE  (frozen)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signal, sym, fold, attribution=None):
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
                fired = attribution[i-1] if attribution is not None else "PORT"
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
    return (float(np.percentile(pfs,  5)),
            float(np.percentile(pfs, 50)),
            float(np.percentile(pfs, 95)))

def monte_carlo(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return {"prob_profit":0.0,"p5":CAPITAL,"p50":CAPITAL,
                "p95":CAPITAL,"finals":np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit": float((finals > CAPITAL).mean()),
            "p5":  float(np.percentile(finals,  5)),
            "p50": float(np.percentile(finals, 50)),
            "p95": float(np.percentile(finals, 95)),
            "finals": finals}

def loo_sym(sym_trades_dict):
    active = {s: tl for s, tl in sym_trades_dict.items() if tl}
    if not active: return {}
    return {omit: metrics([t for s, tl in active.items()
                            if s != omit for t in tl])["pf"]
            for omit in active}

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
    score = sum([
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
            "mc_p":mc["prob_profit"],"mc_p50":mc["p50"],"mc_finals":mc["finals"],
            "sym_floor":sf,"fold_floor":ff,
            "loo_sym":ls,"loo_fld":lf,
            "score":score,"verdict":verdict}

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  QUANTLAB AI — RESEARCH #049")
print("  Universe Out-of-Sample Validation — New Symbol Universe")
print(SEP)
print()
print("  FROZEN PORTFOLIO  (unchanged from R047):")
for eid, label, _ in PROD_ENVS:
    print(f"    {eid}: {label}")
print()
print(f"  New universe:  {len(SYMBOLS)} symbols  (never used in R042–R047)")
print(f"  Walk-forward:  {len(FOLDS)}-fold expanding  (identical to R047)")
print()
print("  ⚠  NO OPTIMISATION · NO TUNING · EXACTLY ONE RUN")
print()

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOAD
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Loading new universe …")
print(SEP2)

all_dfs = {}
for sym in SYMBOLS:
    tag  = sym.replace("-","_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path):
        print(f"  [SKIP] {sym}: no cache file")
        continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) < MIN_BARS:
        print(f"  [SKIP] {sym}: only {len(df)} bars < {MIN_BARS}")
        continue
    all_dfs[sym] = add_features(df)

SYMBOLS     = list(all_dfs.keys())
total_bars  = sum(len(d) for d in all_dfs.values())
print(f"  Loaded {len(SYMBOLS)} symbols · {total_bars:,} bars")
print()

# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD  (5-fold expanding, identical structure to R047)
# ─────────────────────────────────────────────────────────────────────────────
sym_trades_dict = defaultdict(list)
fold_env_n      = {eid: [] for eid in PROD_IDS}

print(f"  Walk-forward: {len(FOLDS)} folds × {len(SYMBOLS)} symbols")
print(SEP2)

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_counts = {eid: 0 for eid in PROD_IDS}
    fold_port_n = 0

    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[:int(N * is_end)]
        df_oos = df_full.iloc[int(N * is_end):int(N * oos_end)].copy().reset_index(drop=True)
        if len(df_oos) < 100: continue

        thr = learn_thresholds(df_is)

        # Build per-environment signals
        env_signals = []
        for eid in PROD_IDS:
            em  = env_mask(df_oos, eid, thr)
            sig = signal_relvol(df_oos, em)
            env_signals.append((eid, sig))
            fold_counts[eid] += int(sig.sum())   # signal fires (pre-dedup)

        # Combine via priority cascade
        port_sig, attr = portfolio_signal(env_signals)

        # Run frozen backtest
        tl = run_backtest(df_oos, port_sig, sym, fold_idx, attribution=attr)
        sym_trades_dict[sym].extend(tl)
        fold_port_n += len(tl)

    for eid in PROD_IDS:
        fold_env_n[eid].append(fold_counts[eid])

    counts_str = "  ".join(f"{e}={fold_counts[e]:3d}" for e in PROD_IDS)
    print(f"  Fold {fold_idx}  IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%  "
          f"{counts_str}  PORT={fold_port_n:3d}")

print()

all_trades = [t for tl in sym_trades_dict.values() for t in tl]

# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL ENVIRONMENT STATS
# ─────────────────────────────────────────────────────────────────────────────
print("  Individual environment stats on new universe:")
env_solo_trades = defaultdict(list)
# Re-run single-env for per-env stats
for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[:int(N * is_end)]
        df_oos = df_full.iloc[int(N * is_end):int(N * oos_end)].copy().reset_index(drop=True)
        if len(df_oos) < 100: continue
        thr = learn_thresholds(df_is)
        for eid in PROD_IDS:
            em  = env_mask(df_oos, eid, thr)
            sig = signal_relvol(df_oos, em)
            tl  = run_backtest(df_oos, sig, sym, fold_idx)
            env_solo_trades[eid].extend(tl)

env_solo_stats = {}
for eid in PROD_IDS:
    tl   = env_solo_trades[eid]
    ms   = metrics(tl)
    b5s,b50s,_ = bootstrap_pf(ms["pnls"])
    mcs  = monte_carlo(ms["pnls"])
    env_solo_stats[eid] = {**ms, "b50": b50s, "mc_p": mcs["prob_profit"]}
    print(f"    {eid}  ({PROD_LABEL[eid][:45]:45s})"
          f"  n={ms['n']:4d}  WR={ms['wr']:.1%}  PF={ms['pf']:.3f}"
          f"  Boot={b50s:.3f}  MC={mcs['prob_profit']:.1%}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO STATS
# ─────────────────────────────────────────────────────────────────────────────
print("  Computing portfolio stats …")
r = full_stats(all_trades, dict(sym_trades_dict))

# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  RESEARCH QUESTIONS  —  UNIVERSE OUT-OF-SAMPLE")
print(SEP)
print()
print(f"  Portfolio:   E06 + E11  (frozen from R047)")
print(f"  Universe:    {len(SYMBOLS)} new symbols  (never in R042–R047)")
print(f"  Data range:  ~24 months per symbol")
print()

pf_ok   = r["pf"]           > PROM_PF
n_ok    = r["n"]            >= PROM_N
boot_ok = r["b50"]          > PROM_BOOT
mc_ok   = r["mc_p"]         > PROM_MC
loos_ok = r["sym_floor"]    > 1.0
loof_ok = r["fold_floor"]   > 1.0
mdd_ok  = abs(r["mdd"])     < PROM_MDD

print(f"  Q1.  PF on new universe      : {r['pf']:.4f}  "
      f"{'✓ ABOVE 1.20' if pf_ok else '✗ BELOW 1.20'}")
print(f"  Q2.  Win Rate                : {r['wr']:.1%}")
print(f"  Q3.  Trade count             : {r['n']}  "
      f"{'✓' if n_ok else '⚠ below ' + str(PROM_N)}")
print(f"  Q4.  Bootstrap PF            : p5={r['b5']:.3f}  median={r['b50']:.3f}  p95={r['b95']:.3f}  "
      f"{'✓' if boot_ok else '✗'}")
print(f"  Q5.  Monte Carlo prob profit : {r['mc_p']:.1%}  "
      f"{'✓' if mc_ok else '✗'}")
print(f"  Q6.  LOO-symbol floor        : {r['sym_floor']:.4f}  "
      f"{'✓' if loos_ok else '✗'}")
print(f"  Q7.  LOO-fold floor          : {r['fold_floor']:.4f}  "
      f"{'✓' if loof_ok else '✗'}")
print(f"  Q8.  Max Drawdown            : {r['mdd']:.2%}  "
      f"{'✓' if mdd_ok else '✗'}")
print()
print(f"  Q9.  Equity Curve:")
print(f"       Start : ${CAPITAL:,.0f}   End : ${CAPITAL + r['net']:,.2f}")
print(f"       Net   : ${r['net']:+,.2f}  ({r['net'] / CAPITAL:+.2%})")
print(f"       Sharpe: {r['sharpe']:.3f}   AvgR: {r['exp_r']:.4f}")
print()
print("  Q10. Comparison — R047 (original 23 syms) vs R049 (new 26 syms):")
print(f"       {'Metric':<22}  {'R047 orig':>10}  {'R049 new':>10}  {'Delta':>12}")
print(f"       {'─'*22}  {'─'*10}  {'─'*10}  {'─'*12}")
def fmt_delta(v049, v047, higher_better=True):
    d = v049 - v047
    arrow = ("▲" if d >= 0 else "▼") if higher_better else ("▲" if d <= 0 else "▼")
    return f"{arrow} {abs(d):.4f}"
print(f"       {'PF':<22}  {R047_BENCH['pf']:>10.4f}  {r['pf']:>10.4f}  {fmt_delta(r['pf'], R047_BENCH['pf']):>12}")
print(f"       {'Win Rate':<22}  {R047_BENCH['wr']:>10.1%}  {r['wr']:>10.1%}  {r['wr']-R047_BENCH['wr']:>+11.1%}")
print(f"       {'n trades':<22}  {R047_BENCH['n']:>10d}  {r['n']:>10d}  {r['n']-R047_BENCH['n']:>+10d}")
print(f"       {'Boot median':<22}  {R047_BENCH['boot_p50']:>10.4f}  {r['b50']:>10.4f}  {fmt_delta(r['b50'], R047_BENCH['boot_p50']):>12}")
print(f"       {'MC prob':<22}  {R047_BENCH['mc']:>10.1%}  {r['mc_p']:>10.1%}  {r['mc_p']-R047_BENCH['mc']:>+11.1%}")
print(f"       {'LOO-S floor':<22}  {R047_BENCH['loo_s']:>10.4f}  {r['sym_floor']:>10.4f}  {fmt_delta(r['sym_floor'], R047_BENCH['loo_s']):>12}")
print(f"       {'LOO-F floor':<22}  {R047_BENCH['loo_f']:>10.4f}  {r['fold_floor']:>10.4f}  {fmt_delta(r['fold_floor'], R047_BENCH['loo_f']):>12}")
print(f"       {'MDD':<22}  {R047_BENCH['mdd']:>10.2%}  {r['mdd']:>10.2%}  {fmt_delta(abs(r['mdd']), abs(R047_BENCH['mdd']), higher_better=False):>12}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SCORECARD
# ─────────────────────────────────────────────────────────────────────────────
criteria = {
    "PF > 1.20":           pf_ok,
    "n ≥ 250":             n_ok,
    "Boot median > 1.20":  boot_ok,
    "MC > 80%":            mc_ok,
    "LOO-S > 1.0":         loos_ok,
    "LOO-F > 1.0":         loof_ok,
    "MDD < 15%":           mdd_ok,
}
print(SEP2)
print("  SCORECARD  (R049 — identical thresholds to R047, 7/7)")
print(SEP2)
for crit, passed in criteria.items():
    print(f"    {'✓' if passed else '✗'}  {crit}")
print(f"\n    Score: {r['score']}/7")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
print("  " + "▓" * 60)
print(f"  FINAL VERDICT:  {r['verdict']}")
print("  " + "▓" * 60)
print()
if r["verdict"] == "PROMOTE":
    print("  E06+E11 HOLDS on a completely new symbol universe.")
    print("  The edge is not symbol-specific — it transfers.")
elif r["verdict"] == "WATCHLIST":
    failed = [k for k, v in criteria.items() if not v]
    print(f"  Conditional pass. Failed: {', '.join(failed)}")
    print("  Edge partially transfers but not fully robust across universe.")
else:
    failed = [k for k, v in criteria.items() if not v]
    print(f"  FAILED on new universe. Failed: {', '.join(failed)}")
    print("  Edge appears symbol-specific. Review environment definitions.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# LOO-SYMBOL TABLE
# ─────────────────────────────────────────────────────────────────────────────
if r["loo_sym"]:
    print("  LOO-Symbol PF table:")
    print(f"    {'Symbol':<25}  {'LOO PF':>8}  {'n own':>6}")
    for sym in sorted(r["loo_sym"], key=lambda s: r["loo_sym"][s]):
        icon = "✓" if r["loo_sym"][sym] > 1.0 else "✗"
        n_sym = len(sym_trades_dict[sym])
        print(f"    {icon}  {sym:<23}  {r['loo_sym'][sym]:>8.4f}  {n_sym:>6}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# LOO-FOLD TABLE
# ─────────────────────────────────────────────────────────────────────────────
if r["loo_fld"]:
    print("  LOO-Fold PF table:")
    for f_idx, pf_f in sorted(r["loo_fld"].items()):
        icon = "✓" if pf_f > 1.0 else "✗"
        print(f"    {icon}  Fold {f_idx} LOO  PF={pf_f:.4f}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SAVE TRADE LOG
# ─────────────────────────────────────────────────────────────────────────────
if all_trades:
    pd.DataFrame(all_trades).to_csv(f"{OUT}/r049_trades.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts …")
print(SEP2)

verdict_col = C_GREEN if r["verdict"] == "PROMOTE" else \
              (C_GOLD if r["verdict"] == "WATCHLIST" else C_RED)

def panel_style(ax, title, fontsize=8):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title, fontsize=fontsize, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)

# ── Chart 1: Equity Curve ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(C_BG)
if len(r["equity"]) > 1:
    eqi = np.arange(len(r["equity"]))
    ax.plot(eqi, r["equity"], color=C_TEAL, linewidth=1.6,
            label=f"E06+E11 new universe  PF={r['pf']:.3f}  n={r['n']}")
    peak_e = np.maximum.accumulate(r["equity"])
    ax.fill_between(eqi, r["equity"], peak_e, alpha=0.22, color=C_RED, label="Drawdown")
    ax.axhline(CAPITAL, color=C_GREY, linewidth=0.8, linestyle="--", alpha=0.6,
               label="Breakeven")
panel_style(ax, f"Q9 · Equity Curve — E06+E11 on New Universe  ({r['n']} trades  PF={r['pf']:.3f})")
ax.set_xlabel("Trade #", fontsize=8)
ax.set_ylabel("Capital ($)", fontsize=8)
ax.legend(fontsize=7, loc="upper left")
verdict_txt = (f"New universe: {len(SYMBOLS)} symbols (never tested)\n"
               f"Verdict: {r['verdict']}  Score: {r['score']}/7")
ax.text(0.98, 0.04, verdict_txt, transform=ax.transAxes,
        fontsize=7, ha="right", va="bottom", color=verdict_col)
plt.tight_layout()
plt.savefig(f"{OUT}/r049_equity_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 2: Per-Symbol Trade Count + PF ─────────────────────────────────────
syms_with_trades = [(s, sym_trades_dict[s]) for s in SYMBOLS if sym_trades_dict[s]]
syms_with_trades.sort(key=lambda x: -metrics(x[1])["pf"])

if syms_with_trades:
    fig, (ax_pf, ax_n) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.patch.set_facecolor(C_BG)
    labels    = [s.replace("-USDT-SWAP","") for s, _ in syms_with_trades]
    pf_vals   = [metrics(tl)["pf"]  for _, tl in syms_with_trades]
    n_vals    = [len(tl)             for _, tl in syms_with_trades]
    x         = np.arange(len(labels))

    c_pf = [C_GREEN if p > 1.0 else C_RED for p in pf_vals]
    ax_pf.bar(x, pf_vals, color=c_pf, alpha=0.85, width=0.7)
    ax_pf.axhline(1.0,    color=C_GREY,   linewidth=0.8, linestyle="--", alpha=0.6)
    ax_pf.axhline(PROM_PF,color=C_PURPLE, linewidth=0.8, linestyle=":",  alpha=0.8,
                  label=f"Threshold {PROM_PF}")
    for xi, pv in zip(x, pf_vals):
        ax_pf.text(xi, pv + 0.02, f"{pv:.2f}", ha="center", va="bottom", fontsize=6, color=C_TEXT)
    panel_style(ax_pf, "PF by Symbol — New Universe  (green=profitable)")
    ax_pf.set_ylabel("Profit Factor", fontsize=8)
    ax_pf.legend(fontsize=7, loc="upper right")

    c_n = [C_TEAL if n >= 5 else C_GREY for n in n_vals]
    ax_n.bar(x, n_vals, color=c_n, alpha=0.85, width=0.7)
    for xi, nv in zip(x, n_vals):
        ax_n.text(xi, nv + 0.2, str(nv), ha="center", va="bottom", fontsize=6, color=C_TEXT)
    panel_style(ax_n, "Trade Count by Symbol — New Universe")
    ax_n.set_ylabel("Trades", fontsize=8)
    ax_n.set_xticks(x)
    ax_n.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUT}/r049_symbol_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()

# ── Chart 3: Bootstrap Distribution ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(C_BG)
if len(r["pnls"]) >= 5:
    rng  = np.random.default_rng(42)
    bpfs = [safe_pf(s[s>0].sum(), abs(s[s<0].sum()))
            for _ in range(N_BOOT)
            for s in [rng.choice(r["pnls"], len(r["pnls"]), replace=True)]]
    ax.hist(bpfs, bins=60, color=C_TEAL, alpha=0.75, edgecolor="none",
            label=f"Bootstrap  n={N_BOOT:,}")
    ax.axvline(r["b5"],  color=C_RED,    linewidth=1.2, linestyle="--",
               label=f"p5={r['b5']:.3f}")
    ax.axvline(r["b50"], color=C_GOLD,   linewidth=1.8, linestyle="-",
               label=f"p50={r['b50']:.3f}")
    ax.axvline(r["b95"], color=C_GREEN,  linewidth=1.2, linestyle="--",
               label=f"p95={r['b95']:.3f}")
    ax.axvline(1.0,       color=C_GREY,   linewidth=1.0, linestyle=":", alpha=0.6)
    ax.axvline(PROM_PF,   color=C_PURPLE, linewidth=1.0, linestyle=":", alpha=0.8)
    ax.axvline(R047_BENCH["boot_p50"], color=C_GOLD, linewidth=1.0, linestyle="-.",
               alpha=0.7, label=f"R047 p50={R047_BENCH['boot_p50']:.3f}")
panel_style(ax, f"Q4 · Bootstrap PF — New Universe  (median={r['b50']:.3f})")
ax.set_xlabel("Profit Factor", fontsize=8)
ax.set_ylabel("Frequency", fontsize=8)
ax.legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT}/r049_bootstrap.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 4: Fold Consistency (per-fold PF bar chart) ────────────────────────
fold_pfs = {}
for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    ftrades = [t for t in all_trades if t["fold"] == fold_idx]
    fold_pfs[fold_idx] = metrics(ftrades)["pf"] if ftrades else 0.0

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(C_BG)
fx = list(fold_pfs.keys())
fy = [fold_pfs[f] for f in fx]
fc = [C_GREEN if p > 1.0 else C_RED for p in fy]
xlabels = [f"Fold {f}\n{int(FOLDS[f-1][0]*100)}–{int(FOLDS[f-1][1]*100)}%" for f in fx]
bars = ax.bar(fx, fy, color=fc, alpha=0.85, width=0.6)
ax.axhline(1.0,    color=C_GREY,   linewidth=0.8, linestyle="--", alpha=0.6)
ax.axhline(PROM_PF,color=C_PURPLE, linewidth=0.8, linestyle=":", alpha=0.8,
           label=f"Threshold {PROM_PF}")
for xi, yi in zip(fx, fy):
    ax.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=8, color=C_TEXT)
ax.set_xticks(fx)
ax.set_xticklabels(xlabels, fontsize=8)
panel_style(ax, f"Q7 (LOO-F={r['fold_floor']:.3f}) · Per-Fold PF — New Universe")
ax.set_ylabel("Profit Factor", fontsize=8)
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT}/r049_fold_consistency.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 5: R047 vs R049 Side-by-Side Comparison ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(C_BG)
fig.suptitle("Q10 · R047 (Original 23 syms) vs R049 (New 26 syms) — Frozen E06+E11",
             fontsize=10, color=C_TEXT)

metrics_cmp = [
    ("PF",          R047_BENCH["pf"],       r["pf"],        1.2),
    ("Boot p50",    R047_BENCH["boot_p50"], r["b50"],        1.2),
    ("LOO-S",       R047_BENCH["loo_s"],    r["sym_floor"],  1.0),
    ("LOO-F",       R047_BENCH["loo_f"],    r["fold_floor"], 1.0),
    ("MDD (abs)",   abs(R047_BENCH["mdd"]), abs(r["mdd"]),   None),
]
keys = [m[0] for m in metrics_cmp]
v047 = [m[1] for m in metrics_cmp]
v049 = [m[2] for m in metrics_cmp]
thresh = [m[3] for m in metrics_cmp]

ax0 = axes[0]
xv  = np.arange(len(keys))
w   = 0.35
ax0.bar(xv - w/2, v047, w, color=C_GOLD,  alpha=0.85, label="R047 (23 syms)")
ax0.bar(xv + w/2, v049, w, color=C_TEAL,  alpha=0.85, label="R049 (26 syms)")
for i, thr in enumerate(thresh):
    if thr is not None:
        ax0.hlines(thr, i-w, i+w, colors=C_PURPLE, linewidths=0.8, linestyles=":")
ax0.set_xticks(xv)
ax0.set_xticklabels(keys, fontsize=7)
panel_style(ax0, "Key Metrics Comparison")
ax0.legend(fontsize=7)

# Win rate + MC
ax1 = axes[1]
wr_labels = ["Win Rate", "MC Prob"]
wr_047    = [R047_BENCH["wr"],  R047_BENCH["mc"]]
wr_049    = [r["wr"],           r["mc_p"]]
xw = np.arange(2)
ax1.bar(xw - w/2, wr_047, w, color=C_GOLD,  alpha=0.85, label="R047")
ax1.bar(xw + w/2, wr_049, w, color=C_TEAL,  alpha=0.85, label="R049")
ax1.axhline(0.80, color=C_PURPLE, linewidth=0.8, linestyle=":")
ax1.set_xticks(xw)
ax1.set_xticklabels(wr_labels, fontsize=8)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
panel_style(ax1, "Win Rate & MC")
ax1.legend(fontsize=7)

# Score comparison
ax2 = axes[2]
score_labels = ["R047\n(23 syms)", "R049\n(26 syms)"]
score_vals   = [R047_BENCH["score"], r["score"]]
score_cols   = [C_GOLD, verdict_col]
ax2.bar([0,1], score_vals, color=score_cols, alpha=0.85, width=0.5)
ax2.axhline(7, color=C_GREY, linewidth=0.8, linestyle="--", alpha=0.6, label="Max (7)")
for xi, sv in enumerate(score_vals):
    ax2.text(xi, sv + 0.1, f"{sv}/7", ha="center", va="bottom", fontsize=10,
             color=C_TEXT, weight="bold")
ax2.set_xticks([0,1])
ax2.set_xticklabels(score_labels, fontsize=8)
ax2.set_ylim(0, 8)
panel_style(ax2, "Promotion Score")
ax2.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT}/r049_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 6: Dashboard ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor(C_BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35,
                         left=0.05, right=0.97, top=0.93, bottom=0.06)

fig.text(0.5, 0.965, "QUANTLAB AI — R049: UNIVERSE OUT-OF-SAMPLE VALIDATION",
         ha="center", fontsize=13, color=C_TEXT, weight="bold")
fig.text(0.5, 0.945,
         f"E06+E11 (frozen)  ·  {len(SYMBOLS)} new symbols never tested  ·  "
         f"Verdict: {r['verdict']}  ({r['score']}/7)",
         ha="center", fontsize=9, color=verdict_col)

# A — equity curve
ax_a = fig.add_subplot(gs[0, :2])
if len(r["equity"]) > 1:
    eqi = np.arange(len(r["equity"]))
    ax_a.plot(eqi, r["equity"], color=C_TEAL, linewidth=1.5)
    pk_a = np.maximum.accumulate(r["equity"])
    ax_a.fill_between(eqi, r["equity"], pk_a, alpha=0.22, color=C_RED)
    ax_a.axhline(CAPITAL, color=C_GREY, linewidth=0.8, linestyle="--", alpha=0.5)
panel_style(ax_a, f"Equity Curve  n={r['n']}  PF={r['pf']:.3f}  MDD={r['mdd']:.2%}")
ax_a.set_ylabel("Capital ($)", fontsize=7)

# B — bootstrap
ax_b = fig.add_subplot(gs[0, 2])
if len(r["pnls"]) >= 5:
    rng2 = np.random.default_rng(42)
    bpfs2 = [safe_pf(s[s>0].sum(), abs(s[s<0].sum()))
             for _ in range(N_BOOT)
             for s in [rng2.choice(r["pnls"], len(r["pnls"]), replace=True)]]
    ax_b.hist(bpfs2, bins=40, color=C_TEAL, alpha=0.75, edgecolor="none")
    ax_b.axvline(r["b50"], color=C_GOLD,   linewidth=1.6)
    ax_b.axvline(PROM_PF,  color=C_PURPLE, linewidth=1.0, linestyle=":")
panel_style(ax_b, f"Bootstrap  p50={r['b50']:.3f}")
ax_b.set_xlabel("PF", fontsize=7)

# C — fold consistency
ax_c = fig.add_subplot(gs[1, 0])
fx2  = list(fold_pfs.keys())
fy2  = [fold_pfs[f] for f in fx2]
fc2  = [C_GREEN if p > 1.0 else C_RED for p in fy2]
ax_c.bar(fx2, fy2, color=fc2, alpha=0.85, width=0.6)
ax_c.axhline(1.0, color=C_GREY, linewidth=0.7, linestyle="--", alpha=0.5)
for xi2, yi2 in zip(fx2, fy2):
    ax_c.text(xi2, yi2 + 0.01, f"{yi2:.2f}", ha="center", va="bottom",
              fontsize=6, color=C_TEXT)
panel_style(ax_c, f"Fold PF  floor={r['fold_floor']:.3f}")
ax_c.set_xticks(fx2)
ax_c.set_xticklabels([f"F{f}" for f in fx2], fontsize=7)

# D — LOO-symbol
ax_d = fig.add_subplot(gs[1, 1:])
if r["loo_sym"]:
    sl   = sorted(r["loo_sym"], key=lambda s: r["loo_sym"][s])
    slbl = [s.replace("-USDT-SWAP","") for s in sl]
    sval = [r["loo_sym"][s] for s in sl]
    scol = [C_GREEN if v > 1.0 else C_RED for v in sval]
    yy   = np.arange(len(slbl))
    ax_d.barh(yy, sval, color=scol, alpha=0.8, height=0.7)
    ax_d.axvline(1.0, color=C_GREY, linewidth=0.7, linestyle="--", alpha=0.5)
    ax_d.set_yticks(yy)
    ax_d.set_yticklabels(slbl, fontsize=5)
panel_style(ax_d, f"LOO-Symbol  floor={r['sym_floor']:.3f}")
ax_d.set_xlabel("PF (leave one sym out)", fontsize=7)

# E — comparison table
ax_e = fig.add_subplot(gs[2, :2])
ax_e.set_facecolor(C_PANEL)
ax_e.axis("off")
tbl_rows = [
    ["Metric",      "R047 (23 syms)", "R049 (26 syms)", "Pass R049?"],
    ["PF",          f"{R047_BENCH['pf']:.3f}",      f"{r['pf']:.3f}",
     "✓" if pf_ok else "✗"],
    ["WR",          f"{R047_BENCH['wr']:.1%}",       f"{r['wr']:.1%}", "—"],
    ["n",           str(R047_BENCH["n"]),             str(r["n"]),
     "✓" if n_ok else "✗"],
    ["Boot p50",    f"{R047_BENCH['boot_p50']:.3f}", f"{r['b50']:.3f}",
     "✓" if boot_ok else "✗"],
    ["MC prob",     f"{R047_BENCH['mc']:.1%}",       f"{r['mc_p']:.1%}",
     "✓" if mc_ok else "✗"],
    ["LOO-S",       f"{R047_BENCH['loo_s']:.3f}",    f"{r['sym_floor']:.3f}",
     "✓" if loos_ok else "✗"],
    ["LOO-F",       f"{R047_BENCH['loo_f']:.3f}",    f"{r['fold_floor']:.3f}",
     "✓" if loof_ok else "✗"],
    ["MDD",         f"{R047_BENCH['mdd']:.2%}",      f"{r['mdd']:.2%}",
     "✓" if mdd_ok else "✗"],
    ["Score",       f"{R047_BENCH['score']}/7",       f"{r['score']}/7",
     r["verdict"]],
]
tbl = ax_e.table(cellText=tbl_rows[1:], colLabels=tbl_rows[0],
                 cellLoc="center", loc="center",
                 bbox=[0.0, 0.0, 1.0, 1.0])
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(C_GRID)
    cell.set_facecolor(C_PANEL if row > 0 else "#1C2128")
    cell.set_text_props(color=C_TEXT)
    if row > 0 and col == 3:
        txt = tbl_rows[row][3]
        cell.set_text_props(
            color=(C_GREEN if txt == "✓" else
                   C_RED   if txt == "✗" else
                   verdict_col))
panel_style(ax_e, "Q10 · R047 vs R049 Comparison")

# F — scorecard
ax_f = fig.add_subplot(gs[2, 2])
ax_f.set_facecolor(C_PANEL)
ax_f.axis("off")
ax_f.text(0.5, 0.97, f"VERDICT: {r['verdict']}", transform=ax_f.transAxes,
          ha="center", va="top", fontsize=12, color=verdict_col, weight="bold")
ax_f.text(0.5, 0.87, f"Score: {r['score']}/7", transform=ax_f.transAxes,
          ha="center", va="top", fontsize=10, color=C_TEXT)
y = 0.76
for crit, passed in criteria.items():
    ax_f.text(0.06, y, f"{'✓' if passed else '✗'}  {crit}",
              transform=ax_f.transAxes, ha="left", va="top",
              fontsize=7, color=C_GREEN if passed else C_RED)
    y -= 0.11
ax_f.text(0.5, 0.03,
          f"New universe: {len(SYMBOLS)} syms\nn={r['n']}  PF={r['pf']:.3f}\n"
          f"MDD={r['mdd']:.2%}",
          transform=ax_f.transAxes, ha="center", va="bottom", fontsize=7, color=C_TEXT)
for sp in ax_f.spines.values():
    sp.set_color(C_GRID)

plt.savefig(f"{OUT}/r049_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  OUTPUT FILES")
print(SEP)
for fname in ["r049_dashboard.png","r049_equity_curve.png","r049_symbol_breakdown.png",
              "r049_bootstrap.png","r049_fold_consistency.png",
              "r049_comparison.png","r049_trades.csv"]:
    path = f"{OUT}/{fname}"
    if os.path.exists(path):
        print(f"    {path}")
print()

print(SEP)
print("  R049 COMPLETE — UNIVERSE OUT-OF-SAMPLE VALIDATION")
print(SEP)
print(f"  Symbols tested:  {len(SYMBOLS)}  (0 overlap with R042–R047)")
print(f"  n trades:        {r['n']}")
print(f"  PF:              {r['pf']:.4f}")
print(f"  Win Rate:        {r['wr']:.1%}")
print(f"  MDD:             {r['mdd']:.2%}")
print(f"  Boot p50:        {r['b50']:.4f}")
print(f"  MC prob:         {r['mc_p']:.1%}")
print(f"  LOO-S floor:     {r['sym_floor']:.4f}")
print(f"  LOO-F floor:     {r['fold_floor']:.4f}")
print(f"  Score:           {r['score']}/7")
print()
print(f"  ╔══════════════════════════════════════════════╗")
print(f"  ║  FINAL VERDICT:  {r['verdict']:<28}║")
print(f"  ╚══════════════════════════════════════════════╝")
print(SEP)
