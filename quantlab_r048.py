"""
=============================================================================
QUANTLAB AI — RESEARCH #048
Frozen Blind Forward Validation
=============================================================================

Objective:
  The research phase is complete.  R047 selected the production portfolio
  (E06 + E11) after evaluating 372 portfolios.  No further optimisation is
  allowed.

  This script runs the frozen portfolio ONCE on a completely untouched
  forward period: every bar whose timestamp is strictly after the last bar
  that existed in the cache when R047 completed.

  Nothing is changed:
    • Environment E06  unchanged
    • Environment E11  unchanged
    • Entry / exit / risk  unchanged
    • Thresholds learned on IS data  (all historical data ≤ BLIND_CUTOFF)
    • Blind OOS  =  all bars  >  BLIND_CUTOFF

  BLIND_CUTOFF  =  2026-07-29 15:00:00 UTC
    (= min(max datetime across all 23 symbols) at end of R047)

Research questions answered:
  Q1. Blind Forward PF
  Q2. Blind Forward Win Rate
  Q3. Blind Forward trade count
  Q4. Bootstrap PF  (p5 / median / p95)
  Q5. Monte Carlo probability of profit
  Q6. LOO-symbol floor PF
  Q7. Max drawdown
  Q8. Equity curve
  Q9. Compare with R047 walk-forward benchmark

Promotion criteria (R048 — single blind period, n floor lowered):
  PF > 1.20  ·  n ≥ 30  ·  Boot median > 1.20  ·  MC > 80%
  LOO-S > 1.0  ·  MDD < 15%
  Score out of 6  (fold-LOO not applicable for a single forward period)

=============================================================================
"""

import os, sys, math, warnings, time
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
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx, get_data

RESEARCH_ID = "R048"
OUT   = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL = CONFIG["STARTING_CAPITAL"]
RR      = CONFIG["RISK_REWARD"]

# ─────────────────────────────────────────────────────────────────────────────
# BLIND CUTOFF — hardcoded at the last bar present when R047 completed
# ALL data  ≤  BLIND_CUTOFF  →  IS  (threshold learning only, never tested)
# ALL data  >  BLIND_CUTOFF  →  Blind OOS  (tested exactly once, right now)
# ─────────────────────────────────────────────────────────────────────────────
BLIND_CUTOFF = pd.Timestamp("2026-07-29 15:00:00", tz="UTC")

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (identical to R047)
# ─────────────────────────────────────────────────────────────────────────────
C_GOLD   = "#F5A623"; C_TEAL   = "#00C4CC"; C_RED    = "#E84545"
C_GREEN  = "#4BB543"; C_PURPLE = "#9B59B6"; C_BLUE   = "#2E86AB"
C_GREY   = "#888888"; C_BG     = "#0D1117"; C_PANEL  = "#161B22"
C_TEXT   = "#E6EDF3"; C_GRID   = "#21262D"

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
    ("E06", "ATR<p25 · Mon-Tue · BodyPct>p60 · Slope<0",     ("ATR_LO","EARLY","PBP_HI","SLP_DN")),
    ("E11", "ADX>p50 · Dist>p75 · Wed-Thu · US(14-21UTC)",   ("ADX_TR","DST_FR","MIDWK","US")),
]
PROD_IDS   = [e[0] for e in PROD_ENVS]
PROD_LABEL = {e[0]: e[1] for e in PROD_ENVS}
PROD_CONDS = {e[0]: e[2] for e in PROD_ENVS}

# ─────────────────────────────────────────────────────────────────────────────
# R047 BENCHMARK (for Q9 comparison)
# ─────────────────────────────────────────────────────────────────────────────
R047_BENCH = {
    "pid": "E06+E11", "pf": 1.601, "n": 253, "wr": 0.490,
    "boot_p50": 1.603, "mc": 1.00, "mdd": -0.053, "loo_s": 1.508,
    "score": 7, "verdict": "PROMOTE",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION CATALOGUE  (frozen copy — not modified)
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

NEEDED_CONDS = sorted({cid for e in PROD_ENVS for cid in e[2]})
QUANT_FEATS  = [
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
MIN_IS_BARS = 2_000   # minimum IS bars per symbol to learn thresholds
N_BOOT      = 2_000
N_MC_PATHS  = 200

# R048 promotion thresholds — single blind forward period
PROM_PF   = 1.20
PROM_N    = 30       # lower floor: one period, not 5 OOS folds
PROM_BOOT = 1.20
PROM_MC   = 0.80
PROM_MDD  = 0.15
# Note: fold-LOO is not applicable for a single period → score out of 6

SEP  = "═" * 110
SEP2 = "─" * 80

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
# THRESHOLDS & MASKS  (frozen — identical to R047)
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
# SIGNAL FUNCTION  (frozen — identical to R047)
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
# BACKTEST ENGINE  (frozen — identical to R047)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signal, sym, attribution=None):
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
                    "sym":        sym,
                    "env":        fired,
                    "entry_time": str(et),
                    "exit_time":  str(dts[i]),
                    "pnl":        round(net, 4),
                    "r_multiple": round(rmul, 4),
                    "win":        int(xt == "TP"),
                    "exit_type":  xt,
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
# STATISTICS  (frozen — identical to R047)
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
    if not active:
        return {}
    return {omit: metrics([t for s, tl in active.items()
                            if s != omit for t in tl])["pf"]
            for omit in active}

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  QUANTLAB AI — RESEARCH #048")
print("  Frozen Blind Forward Validation")
print(SEP)
print()
print("  PRODUCTION PORTFOLIO  (frozen — no changes from R047):")
for eid, label, _ in PROD_ENVS:
    print(f"    {eid}: {label}")
print()
print(f"  BLIND_CUTOFF  =  {BLIND_CUTOFF}")
print(f"  All bars after this timestamp are unseen forward data.")
print()
print("  ⚠  NO OPTIMISATION · NO TUNING · NO SEARCH · EXACTLY ONE RUN")
print()

# ─────────────────────────────────────────────────────────────────────────────
# DATA — force-refresh all 23 symbols
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Refreshing data …")
print(SEP2)

all_dfs   = {}
blind_bar_counts = {}

for sym in SYMBOLS:
    try:
        df_raw = get_data(sym)
    except Exception as exc:
        print(f"  [SKIP] {sym}: {exc}")
        continue

    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], utc=True)
    df_raw = df_raw.sort_values("datetime").reset_index(drop=True)

    df_full = add_features(df_raw)

    # IS  =  everything up to and including the cutoff
    df_is   = df_full[df_full["datetime"] <= BLIND_CUTOFF].copy()
    # Blind OOS  =  strictly after the cutoff
    df_oos  = df_full[df_full["datetime"] >  BLIND_CUTOFF].copy().reset_index(drop=True)

    if len(df_is) < MIN_IS_BARS:
        print(f"  [SKIP] {sym}: insufficient IS bars ({len(df_is)} < {MIN_IS_BARS})")
        continue

    all_dfs[sym]        = {"is": df_is, "oos": df_oos, "full": df_full}
    blind_bar_counts[sym] = len(df_oos)

print()

SYMBOLS = list(all_dfs.keys())
total_is_bars    = sum(len(v["is"])  for v in all_dfs.values())
total_oos_bars   = sum(len(v["oos"]) for v in all_dfs.values())

# Determine blind forward date range
all_oos_dfs = [v["oos"] for v in all_dfs.values() if len(v["oos"]) > 0]
if all_oos_dfs:
    blind_start = min(d["datetime"].min() for d in all_oos_dfs)
    blind_end   = max(d["datetime"].max() for d in all_oos_dfs)
    blind_days  = (blind_end - blind_start).total_seconds() / 86400
else:
    blind_start = blind_end = BLIND_CUTOFF
    blind_days  = 0.0

print(f"  Symbols loaded:   {len(SYMBOLS)}")
print(f"  IS bars total:    {total_is_bars:,}  (threshold learning only, never tested)")
print(f"  Blind OOS bars:   {total_oos_bars:,}  ({blind_days:.1f} calendar days)")
if all_oos_dfs:
    print(f"  Blind OOS range:  {blind_start.strftime('%Y-%m-%d %H:%M')} UTC"
          f"  →  {blind_end.strftime('%Y-%m-%d %H:%M')} UTC")
print()

# ─────────────────────────────────────────────────────────────────────────────
# INSUFFICIENT DATA WARNING
# ─────────────────────────────────────────────────────────────────────────────
MIN_OOS_BARS_WARN = 168   # < 1 week of hourly bars
if total_oos_bars < MIN_OOS_BARS_WARN:
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  ⚠  BLIND FORWARD PERIOD IS SHORT  ({blind_days:.1f} days)           ║")
    print("  ║  The cache was refreshed very recently.  Statistical power  ║")
    print(f"  ║  is limited with only {total_oos_bars:,} OOS bars.                   ║")
    print("  ║  Consider re-running in 4–8 weeks to accumulate trades.    ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN BACKTEST — single pass per symbol
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Running frozen portfolio  E06 + E11  on blind forward period …")
print(SEP2)

all_trades      = []
sym_trades_dict = defaultdict(list)
env_trade_counts = {eid: 0 for eid in PROD_IDS}

for sym in SYMBOLS:
    df_is  = all_dfs[sym]["is"]
    df_oos = all_dfs[sym]["oos"]

    if len(df_oos) < 2:
        continue

    # Learn thresholds on IS only
    thr = learn_thresholds(df_is)

    # Build per-environment signals on blind OOS
    env_signals = []
    for eid in PROD_IDS:
        em  = env_mask(df_oos, eid, thr)
        sig = signal_relvol(df_oos, em)
        env_signals.append((eid, sig))

    # Combine via priority cascade (E06 first, E11 second)
    port_sig, attr = portfolio_signal(env_signals)

    # Run the frozen backtest
    tl = run_backtest(df_oos, port_sig, sym, attribution=attr)

    all_trades.extend(tl)
    sym_trades_dict[sym].extend(tl)
    for t in tl:
        env_trade_counts[t["env"]] += 1

    n_sym = len(tl)
    if n_sym > 0:
        wr_sym = sum(t["win"] for t in tl) / n_sym
        pnl_sym = sum(t["pnl"] for t in tl)
        print(f"  {sym:25s}  n={n_sym:3d}  WR={wr_sym:.1%}  PnL={pnl_sym:+.2f}")

print()
print(f"  E06 attributed trades: {env_trade_counts['E06']}")
print(f"  E11 attributed trades: {env_trade_counts['E11']}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
m = metrics(all_trades)

if m["n"] < 5:
    print("  ⚠  Fewer than 5 trades in the blind forward period.")
    print("     Statistical analysis requires more data.")
    print("     Re-run R048 in 4–8 weeks when the blind period has grown.")
    b5 = b50 = b95 = 0.0
    mc = {"prob_profit": 0.0, "p5": CAPITAL, "p50": CAPITAL, "p95": CAPITAL,
          "finals": np.array([CAPITAL])}
    ls = {}
    sf = 0.0
else:
    b5, b50, b95 = bootstrap_pf(m["pnls"])
    mc            = monte_carlo(m["pnls"])
    ls            = loo_sym(sym_trades_dict)
    sf            = min(ls.values()) if ls else 0.0

# Score (6 criteria — no fold-LOO for single forward period)
criteria = {
    "PF > 1.20":          m["pf"]           > PROM_PF,
    "n ≥ 30":             m["n"]            >= PROM_N,
    "Boot median > 1.20": b50               > PROM_BOOT,
    "MC > 80%":           mc["prob_profit"] > PROM_MC,
    "LOO-S > 1.0":        sf                > 1.0,
    "MDD < 15%":          abs(m["mdd"])     < PROM_MDD,
}
score   = sum(criteria.values())
verdict = ("PROMOTE"   if score == 6 else
           "WATCHLIST" if score >= 4 and m["pf"] > PROM_PF else
           "REJECT")

# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  RESEARCH QUESTIONS  —  BLIND FORWARD VALIDATION")
print(SEP)
print()
print(f"  Portfolio:    E06 + E11  (frozen from R047)")
print(f"  Period:       {blind_start.strftime('%Y-%m-%d')} → {blind_end.strftime('%Y-%m-%d')}"
      f"  ({blind_days:.1f} days)")
print()

print(f"  Q1. Blind Forward PF         : {m['pf']:.4f}  "
      f"{'✓ ABOVE 1.20' if m['pf'] > 1.20 else '✗ BELOW 1.20'}")
print(f"  Q2. Blind Forward Win Rate   : {m['wr']:.1%}")
print(f"  Q3. Blind Forward n trades   : {m['n']}  "
      f"{'✓' if m['n'] >= PROM_N else '⚠ below floor of ' + str(PROM_N)}")
print(f"  Q4. Bootstrap PF             : p5={b5:.3f}  median={b50:.3f}  p95={b95:.3f}  "
      f"{'✓' if b50 > 1.20 else '✗'}")
print(f"  Q5. Monte Carlo prob profit  : {mc['prob_profit']:.1%}  "
      f"{'✓' if mc['prob_profit'] > 0.80 else '✗'}")
print(f"  Q6. LOO-symbol floor PF      : {sf:.4f}  "
      f"{'✓' if sf > 1.0 else '✗'}")
print(f"  Q7. Max Drawdown             : {m['mdd']:.2%}  "
      f"{'✓' if abs(m['mdd']) < 0.15 else '✗'}")
print()

print("  Q8. Equity Curve:")
print(f"      Start capital : ${CAPITAL:,.0f}")
print(f"      End capital   : ${CAPITAL + m['net']:,.2f}")
print(f"      Net PnL       : ${m['net']:+,.2f}  ({m['net'] / CAPITAL:+.2%})")
print(f"      Sharpe        : {m['sharpe']:.3f}")
print(f"      Avg R/trade   : {m['exp_r']:.4f}")
print()

print("  Q9. Comparison with R047 Walk-Forward Benchmark:")
print(f"      {'Metric':<25}  {'R047 WF':>10}  {'R048 Blind':>10}  {'Delta':>10}")
print(f"      {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}")
def delta_str(v048, v047, higher_is_better=True):
    d = v048 - v047
    if higher_is_better:
        return f"{'▲' if d >= 0 else '▼'} {abs(d):.4f}"
    else:
        return f"{'▲' if d <= 0 else '▼'} {abs(d):.4f}"

print(f"      {'PF':<25}  {R047_BENCH['pf']:>10.4f}  {m['pf']:>10.4f}  "
      f"{delta_str(m['pf'], R047_BENCH['pf']):>12}")
print(f"      {'Win Rate':<25}  {R047_BENCH['wr']:>10.1%}  {m['wr']:>10.1%}  "
      f"{m['wr'] - R047_BENCH['wr']:>+11.1%}")
print(f"      {'n trades':<25}  {R047_BENCH['n']:>10d}  {m['n']:>10d}  "
      f"{m['n'] - R047_BENCH['n']:>+10d}")
print(f"      {'Bootstrap median':<25}  {R047_BENCH['boot_p50']:>10.4f}  {b50:>10.4f}  "
      f"{delta_str(b50, R047_BENCH['boot_p50']):>12}")
print(f"      {'MC prob profit':<25}  {R047_BENCH['mc']:>10.1%}  {mc['prob_profit']:>10.1%}  "
      f"{mc['prob_profit'] - R047_BENCH['mc']:>+11.1%}")
print(f"      {'LOO-S floor':<25}  {R047_BENCH['loo_s']:>10.4f}  {sf:>10.4f}  "
      f"{delta_str(sf, R047_BENCH['loo_s']):>12}")
print(f"      {'MDD':<25}  {R047_BENCH['mdd']:>10.2%}  {m['mdd']:>10.2%}  "
      f"{delta_str(abs(m['mdd']), abs(R047_BENCH['mdd']), higher_is_better=False):>12}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SCORECARD
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  SCORECARD  (R048 — single blind period, 6 criteria, no fold-LOO)")
print(SEP2)
for crit, passed in criteria.items():
    icon = "✓" if passed else "✗"
    print(f"    {icon}  {crit}")
print(f"\n    Score: {score}/6")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
VERDICT_COLOUR = {"PROMOTE": "GREEN", "WATCHLIST": "YELLOW", "REJECT": "RED"}
print("  " + "▓" * 60)
print(f"  FINAL VERDICT:  {verdict}")
print("  " + "▓" * 60)
print()
if verdict == "PROMOTE":
    print("  The frozen portfolio E06+E11 HOLDS on completely unseen data.")
    print("  All 6 validation criteria passed on blind forward data.")
    print("  The walk-forward edge is real and transferable.")
elif verdict == "WATCHLIST":
    failed = [k for k, v in criteria.items() if not v]
    print(f"  Conditional pass.  Failed criteria: {', '.join(failed)}")
    print("  Monitor with additional forward data before live deployment.")
else:
    failed = [k for k, v in criteria.items() if not v]
    print(f"  The blind forward test FAILED.  Failed criteria: {', '.join(failed)}")
    print("  Do NOT deploy.  Return to R047 analysis.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# LOO-SYMBOL TABLE
# ─────────────────────────────────────────────────────────────────────────────
if ls:
    print("  LOO-Symbol PF table:")
    print(f"    {'Symbol':<25}  {'LOO PF':>8}  {'n own':>6}")
    for sym in sorted(ls, key=lambda s: ls[s]):
        n_own = len(sym_trades_dict[sym])
        icon  = "✓" if ls[sym] > 1.0 else "✗"
        print(f"    {icon}  {sym:<23}  {ls[sym]:>8.4f}  {n_own:>6}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# TRADE LOG — save CSV
# ─────────────────────────────────────────────────────────────────────────────
if all_trades:
    tdf = pd.DataFrame(all_trades)
    tdf.to_csv(f"{OUT}/r048_blind_trades.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts …")
print(SEP2)

# ── helpers ──────────────────────────────────────────────────────────────────
def panel_style(ax, title):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title, fontsize=8, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)

# ── Chart 1: Equity Curve ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(C_BG)

if len(m["equity"]) > 1:
    eq_idx = np.arange(len(m["equity"]))
    ax.plot(eq_idx, m["equity"], color=C_TEAL, linewidth=1.6, zorder=3,
            label=f"E06+E11  PF={m['pf']:.3f}  n={m['n']}")
    peak_eq = np.maximum.accumulate(m["equity"])
    ax.fill_between(eq_idx, m["equity"], peak_eq,
                    alpha=0.25, color=C_RED, label="Drawdown")
    ax.axhline(CAPITAL, color=C_GREY, linewidth=0.8, linestyle="--", alpha=0.6)

    # R047 slope reference (expected rate from WF)
    expected_rate = (R047_BENCH["pf"] - 1) / (R047_BENCH["n"]) * m["n"]
    wf_final      = CAPITAL * (1 + expected_rate * 0.01)  # approximate
    ax.axhline(wf_final, color=C_GOLD, linewidth=0.8, linestyle=":",
               alpha=0.7, label=f"R047 WF reference  PF={R047_BENCH['pf']:.3f}")

panel_style(ax, f"Q8 · Blind Forward Equity Curve  —  E06+E11  ({m['n']} trades)")
ax.set_xlabel("Trade #", fontsize=8)
ax.set_ylabel("Capital ($)", fontsize=8)
ax.legend(fontsize=7, loc="upper left")

period_str = (f"{blind_start.strftime('%Y-%m-%d')} → {blind_end.strftime('%Y-%m-%d')}"
              if blind_days > 0 else "No blind data yet")
verdict_col = C_GREEN if verdict == "PROMOTE" else (C_GOLD if verdict == "WATCHLIST" else C_RED)
ax.text(0.98, 0.04, f"BLIND FORWARD: {period_str}\nVerdict: {verdict}  Score: {score}/6",
        transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
        color=verdict_col)

plt.tight_layout()
plt.savefig(f"{OUT}/r048_equity_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 2: Bootstrap PF Distribution ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(C_BG)

if len(m["pnls"]) >= 5:
    rng  = np.random.default_rng(42)
    boot_pfs = [safe_pf(s[s>0].sum(), abs(s[s<0].sum()))
                for _ in range(N_BOOT)
                for s in [rng.choice(m["pnls"], len(m["pnls"]), replace=True)]]
    ax.hist(boot_pfs, bins=60, color=C_TEAL, alpha=0.75, edgecolor="none",
            label=f"Bootstrap PF  (n={N_BOOT:,} resamples)")
    ax.axvline(b5,     color=C_RED,    linewidth=1.2, linestyle="--", label=f"p5  = {b5:.3f}")
    ax.axvline(b50,    color=C_GOLD,   linewidth=1.8, linestyle="-",  label=f"p50 = {b50:.3f}")
    ax.axvline(b95,    color=C_GREEN,  linewidth=1.2, linestyle="--", label=f"p95 = {b95:.3f}")
    ax.axvline(1.0,    color=C_GREY,   linewidth=1.0, linestyle=":",  alpha=0.6, label="PF=1.0")
    ax.axvline(PROM_PF,color=C_PURPLE, linewidth=1.0, linestyle=":",  alpha=0.8,
               label=f"Threshold {PROM_PF}")
    ax.axvline(R047_BENCH["boot_p50"], color=C_GOLD, linewidth=1.0, linestyle="-.",
               alpha=0.7, label=f"R047 p50={R047_BENCH['boot_p50']:.3f}")
else:
    ax.text(0.5, 0.5, f"Insufficient trades for bootstrap\n(n={m['n']})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=C_TEXT)

panel_style(ax, f"Q4 · Bootstrap PF Distribution  —  E06+E11  Blind Forward")
ax.set_xlabel("Profit Factor", fontsize=8)
ax.set_ylabel("Frequency", fontsize=8)
ax.legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT}/r048_bootstrap.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 3: Monte Carlo Equity Fan ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(C_BG)

if len(m["pnls"]) >= 5:
    rng2 = np.random.default_rng(99)
    paths   = np.array([np.concatenate([[CAPITAL],
                         CAPITAL + np.cumsum(rng2.choice(m["pnls"], len(m["pnls"]), replace=True))])
                         for _ in range(N_MC_PATHS)])

    p5_path  = np.percentile(paths, 5,  axis=0)
    p50_path = np.percentile(paths, 50, axis=0)
    p95_path = np.percentile(paths, 95, axis=0)

    x = np.arange(len(p50_path))
    for path in paths[:50]:
        ax.plot(x, path, color=C_TEAL, alpha=0.06, linewidth=0.5)
    ax.fill_between(x, p5_path, p95_path, alpha=0.18, color=C_TEAL, label="p5–p95 band")
    ax.plot(x, p50_path, color=C_GOLD, linewidth=1.8, label=f"Median  p50={mc['p50']:,.0f}")
    ax.plot(x, p5_path,  color=C_RED,  linewidth=1.0, linestyle="--", label=f"p5={mc['p5']:,.0f}")
    ax.plot(x, p95_path, color=C_GREEN,linewidth=1.0, linestyle="--", label=f"p95={mc['p95']:,.0f}")
    if len(m["equity"]) > 1:
        ax.plot(np.arange(len(m["equity"])), m["equity"],
                color=C_TEXT, linewidth=2.0, zorder=5, label="Actual equity")
    ax.axhline(CAPITAL, color=C_GREY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(0.98, 0.06, f"MC prob profit: {mc['prob_profit']:.1%}",
            transform=ax.transAxes, fontsize=8, ha="right", color=C_GOLD)
else:
    ax.text(0.5, 0.5, f"Insufficient trades for MC\n(n={m['n']})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, color=C_TEXT)

panel_style(ax, f"Q5 · Monte Carlo Equity Fan  —  E06+E11  Blind Forward  ({N_MC_PATHS} paths)")
ax.set_xlabel("Trade #", fontsize=8)
ax.set_ylabel("Capital ($)", fontsize=8)
ax.legend(fontsize=7, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/r048_monte_carlo.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 4: Per-Symbol Breakdown ────────────────────────────────────────────
syms_with_trades = [(s, sym_trades_dict[s]) for s in SYMBOLS if sym_trades_dict[s]]
syms_with_trades.sort(key=lambda x: -len(x[1]))

if syms_with_trades:
    fig, (ax_n, ax_pf) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor(C_BG)

    labels = [s.replace("-USDT-SWAP","") for s, _ in syms_with_trades]
    ns     = [len(tl)                    for _, tl in syms_with_trades]
    pfs    = []
    for _, tl in syms_with_trades:
        mm = metrics(tl)
        pfs.append(mm["pf"])

    x   = np.arange(len(labels))
    colours_n  = [C_TEAL if n >= 3 else C_GREY for n in ns]
    colours_pf = [C_GREEN if p > 1.0 else C_RED for p in pfs]

    ax_n.bar(x, ns, color=colours_n, alpha=0.85, width=0.7)
    for xi, ni in zip(x, ns):
        ax_n.text(xi, ni + 0.1, str(ni), ha="center", va="bottom",
                  fontsize=6, color=C_TEXT)
    panel_style(ax_n, "Trade Count by Symbol — Blind Forward")
    ax_n.set_ylabel("Trades", fontsize=8)

    ax_pf.bar(x, pfs, color=colours_pf, alpha=0.85, width=0.7)
    ax_pf.axhline(1.0, color=C_GREY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax_pf.axhline(PROM_PF, color=C_PURPLE, linewidth=0.8, linestyle=":", alpha=0.8)
    for xi, pf_v in zip(x, pfs):
        ax_pf.text(xi, pf_v + 0.01, f"{pf_v:.2f}", ha="center", va="bottom",
                   fontsize=6, color=C_TEXT)
    panel_style(ax_pf, "PF by Symbol — Blind Forward  (green = profitable)")
    ax_pf.set_ylabel("Profit Factor", fontsize=8)
    ax_pf.set_xticks(x)
    ax_pf.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUT}/r048_symbol_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()

# ── Chart 5: Dashboard ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(C_BG)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35,
                         left=0.05, right=0.97, top=0.93, bottom=0.06)

verdict_col = C_GREEN if verdict == "PROMOTE" else (C_GOLD if verdict == "WATCHLIST" else C_RED)

# Title
fig.text(0.5, 0.965, "QUANTLAB AI — R048: FROZEN BLIND FORWARD VALIDATION",
         ha="center", fontsize=13, color=C_TEXT, weight="bold")
fig.text(0.5, 0.945,
         f"E06+E11  ·  Blind Period: {blind_start.strftime('%Y-%m-%d')} → "
         f"{blind_end.strftime('%Y-%m-%d')}  ({blind_days:.0f} days)  ·  "
         f"Verdict: {verdict}  ({score}/6)",
         ha="center", fontsize=9, color=verdict_col)

# Panel A — equity curve
ax_a = fig.add_subplot(gs[0, :2])
if len(m["equity"]) > 1:
    eqi = np.arange(len(m["equity"]))
    ax_a.plot(eqi, m["equity"], color=C_TEAL, linewidth=1.6, label="Blind OOS equity")
    pk_a = np.maximum.accumulate(m["equity"])
    ax_a.fill_between(eqi, m["equity"], pk_a, alpha=0.25, color=C_RED)
    ax_a.axhline(CAPITAL, color=C_GREY, linewidth=0.8, linestyle="--", alpha=0.6)
else:
    ax_a.text(0.5, 0.5, "No trades in blind period", ha="center", va="center",
              transform=ax_a.transAxes, fontsize=10, color=C_GREY)
panel_style(ax_a, "Equity Curve — Blind Forward")
ax_a.set_xlabel("Trade #", fontsize=7)
ax_a.set_ylabel("Capital ($)", fontsize=7)

# Panel B — bootstrap
ax_b = fig.add_subplot(gs[0, 2])
if len(m["pnls"]) >= 5:
    rng3 = np.random.default_rng(42)
    bpfs = [safe_pf(s[s>0].sum(), abs(s[s<0].sum()))
            for _ in range(N_BOOT)
            for s in [rng3.choice(m["pnls"], len(m["pnls"]), replace=True)]]
    ax_b.hist(bpfs, bins=40, color=C_TEAL, alpha=0.75, edgecolor="none", orientation="vertical")
    ax_b.axvline(b50,    color=C_GOLD,   linewidth=1.6)
    ax_b.axvline(PROM_PF,color=C_PURPLE, linewidth=1.0, linestyle=":")
else:
    ax_b.text(0.5, 0.5, f"n={m['n']}\n(need ≥5)", ha="center", va="center",
              transform=ax_b.transAxes, fontsize=9, color=C_GREY)
panel_style(ax_b, f"Bootstrap PF  median={b50:.3f}")
ax_b.set_xlabel("PF", fontsize=7)

# Panel C — MC fan
ax_c = fig.add_subplot(gs[1, :2])
if len(m["pnls"]) >= 5:
    rng4 = np.random.default_rng(77)
    mc_paths = np.array([np.concatenate([[CAPITAL],
                          CAPITAL + np.cumsum(rng4.choice(m["pnls"], len(m["pnls"]), replace=True))])
                          for _ in range(N_MC_PATHS)])
    xc = np.arange(mc_paths.shape[1])
    for pp in mc_paths[:40]:
        ax_c.plot(xc, pp, color=C_TEAL, alpha=0.05, linewidth=0.5)
    ax_c.plot(xc, np.percentile(mc_paths,50,axis=0), color=C_GOLD, linewidth=1.5)
    ax_c.fill_between(xc, np.percentile(mc_paths,5,axis=0),
                      np.percentile(mc_paths,95,axis=0), alpha=0.15, color=C_TEAL)
    if len(m["equity"]) > 1:
        ax_c.plot(np.arange(len(m["equity"])), m["equity"],
                  color=C_TEXT, linewidth=2.0, zorder=5)
    ax_c.axhline(CAPITAL, color=C_GREY, linewidth=0.8, linestyle="--", alpha=0.5)
panel_style(ax_c, f"Monte Carlo Fan  —  MC={mc['prob_profit']:.1%}")
ax_c.set_xlabel("Trade #", fontsize=7)
ax_c.set_ylabel("Capital ($)", fontsize=7)

# Panel D — LOO-symbol
ax_d = fig.add_subplot(gs[1, 2])
if ls:
    sorted_syms  = sorted(ls, key=lambda s: ls[s])
    short_labels = [s.replace("-USDT-SWAP","") for s in sorted_syms]
    loo_vals     = [ls[s] for s in sorted_syms]
    cols_loo     = [C_GREEN if v > 1.0 else C_RED for v in loo_vals]
    yy = np.arange(len(short_labels))
    ax_d.barh(yy, loo_vals, color=cols_loo, alpha=0.8, height=0.7)
    ax_d.axvline(1.0, color=C_GREY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax_d.set_yticks(yy)
    ax_d.set_yticklabels(short_labels, fontsize=5)
else:
    ax_d.text(0.5, 0.5, "Insufficient trades\nfor LOO-symbol",
              ha="center", va="center", transform=ax_d.transAxes,
              fontsize=9, color=C_GREY)
panel_style(ax_d, f"LOO-Symbol  floor={sf:.3f}")
ax_d.set_xlabel("PF (leave one symbol out)", fontsize=7)

# Panel E — Q9 comparison table
ax_e = fig.add_subplot(gs[2, :2])
ax_e.set_facecolor(C_PANEL)
ax_e.axis("off")

tbl_data = [
    ["Metric",        "R047 WF\n(E06+E11)", "R048 Blind\n(E06+E11)", "Pass?"],
    ["PF",            f"{R047_BENCH['pf']:.4f}",  f"{m['pf']:.4f}",
     "✓" if m["pf"] > PROM_PF else "✗"],
    ["Win Rate",      f"{R047_BENCH['wr']:.1%}",  f"{m['wr']:.1%}",  "—"],
    ["n trades",      str(R047_BENCH["n"]),        str(m["n"]),
     "✓" if m["n"] >= PROM_N else "✗"],
    ["Boot p50",      f"{R047_BENCH['boot_p50']:.4f}", f"{b50:.4f}",
     "✓" if b50 > PROM_BOOT else "✗"],
    ["MC prob",       f"{R047_BENCH['mc']:.1%}",  f"{mc['prob_profit']:.1%}",
     "✓" if mc["prob_profit"] > PROM_MC else "✗"],
    ["LOO-S floor",   f"{R047_BENCH['loo_s']:.4f}", f"{sf:.4f}",
     "✓" if sf > 1.0 else "✗"],
    ["MDD",           f"{R047_BENCH['mdd']:.2%}",  f"{m['mdd']:.2%}",
     "✓" if abs(m["mdd"]) < PROM_MDD else "✗"],
    ["Score",         f"{R047_BENCH['score']}/7",  f"{score}/6",      verdict],
]

tbl = ax_e.table(cellText=tbl_data[1:], colLabels=tbl_data[0],
                 cellLoc="center", loc="center",
                 bbox=[0.0, 0.0, 1.0, 1.0])
tbl.auto_set_font_size(False)
tbl.set_fontsize(7)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor(C_GRID)
    cell.set_facecolor(C_PANEL if row > 0 else "#1C2128")
    cell.set_text_props(color=C_TEXT)
    if row > 0 and col == 3:
        txt = tbl_data[row][3]
        cell.set_text_props(
            color=(C_GREEN if txt == "✓" else
                   C_RED   if txt == "✗" else
                   verdict_col))
panel_style(ax_e, "Q9 · Comparison: R047 Walk-Forward vs R048 Blind Forward")

# Panel F — scorecard
ax_f = fig.add_subplot(gs[2, 2])
ax_f.set_facecolor(C_PANEL)
ax_f.axis("off")

y_pos  = 0.92
ax_f.text(0.5, 0.98, f"VERDICT: {verdict}", transform=ax_f.transAxes,
          ha="center", va="top", fontsize=11, color=verdict_col, weight="bold")
ax_f.text(0.5, 0.88, f"Score: {score}/6", transform=ax_f.transAxes,
          ha="center", va="top", fontsize=9, color=C_TEXT)

y_pos = 0.76
for crit, passed in criteria.items():
    col = C_GREEN if passed else C_RED
    icon = "✓" if passed else "✗"
    ax_f.text(0.08, y_pos, f"{icon}  {crit}", transform=ax_f.transAxes,
              ha="left", va="top", fontsize=7, color=col)
    y_pos -= 0.11

ax_f.text(0.5, 0.04,
          f"Period: {blind_days:.0f} days\n"
          f"n={m['n']}  PF={m['pf']:.4f}\n"
          f"MDD={m['mdd']:.2%}",
          transform=ax_f.transAxes, ha="center", va="bottom",
          fontsize=7, color=C_TEXT)
for sp in ax_f.spines.values():
    sp.set_color(C_GRID)

plt.savefig(f"{OUT}/r048_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  OUTPUT FILES")
print(SEP)
print(f"    {OUT}/r048_dashboard.png")
print(f"    {OUT}/r048_equity_curve.png")
print(f"    {OUT}/r048_bootstrap.png")
print(f"    {OUT}/r048_monte_carlo.png")
if syms_with_trades:
    print(f"    {OUT}/r048_symbol_breakdown.png")
if all_trades:
    print(f"    {OUT}/r048_blind_trades.csv")
print()

print(SEP)
print("  R048 COMPLETE — FROZEN BLIND FORWARD VALIDATION")
print(SEP)
print(f"  Portfolio:    E06 + E11  (frozen)")
print(f"  Blind period: {blind_start.strftime('%Y-%m-%d')} → {blind_end.strftime('%Y-%m-%d')}"
      f"  ({blind_days:.0f} calendar days)")
print(f"  n trades:     {m['n']}")
print(f"  PF:           {m['pf']:.4f}")
print(f"  Win Rate:     {m['wr']:.1%}")
print(f"  MDD:          {m['mdd']:.2%}")
print(f"  Boot p50:     {b50:.4f}")
print(f"  MC prob:      {mc['prob_profit']:.1%}")
print(f"  LOO-S floor:  {sf:.4f}")
print(f"  Score:        {score}/6")
print()
print(f"  ╔══════════════════════════════════════════════╗")
print(f"  ║  FINAL VERDICT:  {verdict:<28}║")
print(f"  ╚══════════════════════════════════════════════╝")
print(SEP)
