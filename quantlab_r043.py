"""
=============================================================================
QUANTLAB AI — RESEARCH #043
Independent Environment Portfolio Validation
=============================================================================

Objective:
  Determine whether combining the top-5 independent environments from R042
  increases trade frequency while preserving edge quality.

Rules:
  • No new environments — use only R042 top-5 by rank
  • De-duplication: if multiple envs trigger on the same bar×symbol,
    highest-ranked environment captures the trade (priority cascade)
  • Identical risk management throughout
  • 5-fold expanding walk-forward · OOS only · 23 symbols · 1H

Environments (from r042_environment_library.csv):
  E1  Dist>p75 · Wed-Thu · BodyPct>p60 · US(14-21UTC)       PF=2.003  n=68
  E2  ADX>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)           PF=1.933  n=137
  E3  Dist>p75 · Wed-Thu · PrevBody>p67 · US(14-21UTC)      PF=1.867  n=88
  E4  ADX>p67 · Dist>p60 · Wed-Thu · US(14-21UTC)           PF=1.781  n=234
  E5  ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0         PF=1.735  n=78

Portfolios:
  A = E1 + E2
  B = E1 + E2 + E3
  C = E1 + E2 + E3 + E4
  D = E1 + E2 + E3 + E4 + E5

PROMOTE criteria: PF>1.20 · n≥200 · Boot>1.20 · MC>60%
                  LOO-S>1.00 · LOO-F>1.00 · MDD<25%
=============================================================================
"""

import os, sys, math, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

RESEARCH_ID = "R043"
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
N_BOOT   = 2_000

# PROMOTE thresholds
PROM_PF   = 1.20
PROM_N    = 200
PROM_BOOT = 1.20
PROM_MC   = 0.60
PROM_MDD  = 0.25

# =============================================================================
# R042 TOP-5 ENVIRONMENTS  (loaded from CSV, no hard-coding of thresholds)
# =============================================================================

R042_ENVS = [
    # (id, label, cond_ids_tuple)
    ("E1", "Dist>p75 · Wed-Thu · BodyPct>p60 · US(14-21UTC)",
     ("DST_FR", "MIDWK", "PBP_HI", "US")),
    ("E2", "ADX>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)",
     ("ADX_ST", "DST_FR", "MIDWK", "US")),
    ("E3", "Dist>p75 · Wed-Thu · PrevBody>p67 · US(14-21UTC)",
     ("DST_FR", "MIDWK", "PBD_HI", "US")),
    ("E4", "ADX>p67 · Dist>p60 · Wed-Thu · US(14-21UTC)",
     ("ADX_ST", "DST_MD", "MIDWK", "US")),
    ("E5", "ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0",
     ("ATR_MD", "PRG_VH", "RV_LO", "SLP_DN")),
]

ENV_IDS    = [e[0] for e in R042_ENVS]
ENV_LABEL  = {e[0]: e[1] for e in R042_ENVS}
ENV_CONDS  = {e[0]: e[2] for e in R042_ENVS}

PORTFOLIOS = [
    ("A", ["E1", "E2"]),
    ("B", ["E1", "E2", "E3"]),
    ("C", ["E1", "E2", "E3", "E4"]),
    ("D", ["E1", "E2", "E3", "E4", "E5"]),
]
PORT_IDS   = [p[0] for p in PORTFOLIOS]
PORT_ENVS  = {p[0]: p[1] for p in PORTFOLIOS}

# =============================================================================
# FEATURE ENGINEERING  (identical to R042)
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]
    df["ema200"]        = calc_ema(c, 200)
    df["atr14"]         = calc_atr(df, 14)
    df["atr_rank"]      = df["atr14"].rolling(100).rank(pct=True) * 100
    bb_mid              = c.rolling(20).mean()
    bb_std              = c.rolling(20).std()
    df["bb_width"]      = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["ema_dist_pct"]  = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"]  = (df["ema200"] - df["ema200"].shift(10)) / df["ema200"].shift(10).replace(0, np.nan)
    vol_ma              = v.rolling(20).mean()
    df["rel_vol"]       = v / vol_ma.replace(0, np.nan)
    df["prev_close"]    = c.shift(1)
    df["prev_atr14"]    = df["atr14"].shift(1)
    log_ret             = np.log(c / c.shift(1))
    df["real_vol_20"]   = log_ret.rolling(20).std() * 100.0
    df["adx14"]         = calc_adx(df, 14)
    prev_range          = h.shift(1) - l.shift(1)
    prev_body           = (c.shift(1) - df["open"].shift(1)).abs()
    df["prev_range_r"]  = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]   = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"] = prev_body  / prev_range.replace(0, np.nan)
    dt = pd.to_datetime(df["datetime"], utc=True)
    df["hour_utc"]      = dt.dt.hour.astype(np.int16)
    df["day_of_week"]   = dt.dt.dayofweek.astype(np.int16)
    return df

# =============================================================================
# CONDITION CATALOGUE  (identical to R042)
# =============================================================================

CONDITIONS_DEF = [
    ("ATR_LO",  "ATR<p25",       "atr_rank",      "lt_q",     0.25,  "vol"),
    ("ATR_MD",  "ATR<p40",       "atr_rank",       "lt_q",    0.40,  "vol"),
    ("ATR_HI",  "ATR>p67",       "atr_rank",       "gt_q",    0.67,  "vol"),
    ("RV_LO",   "RealVol<p33",   "real_vol_20",    "lt_q",    0.33,  "vol"),
    ("BB_TG",   "BB<p33",        "bb_width",       "lt_q",    0.33,  "vol"),
    ("BB_MD",   "BB<p50",        "bb_width",       "lt_q",    0.50,  "vol"),
    ("SLP_UP",  "Slope>0",       "ema200_slope",   "gt_fixed", 0.0,  "trend"),
    ("SLP_DN",  "Slope<0",       "ema200_slope",   "lt_fixed", 0.0,  "trend"),
    ("DST_FR",  "Dist>p75",      "ema_dist_pct",   "gt_q_pos", 0.75, "trend"),
    ("DST_MD",  "Dist>p60",      "ema_dist_pct",   "gt_q_pos", 0.60, "trend"),
    ("DST_NR",  "Dist<p33",      "ema_dist_pct",   "lt_q",    0.33,  "trend"),
    ("ADX_TR",  "ADX>p50",       "adx14",          "gt_q",    0.50,  "trend"),
    ("ADX_ST",  "ADX>p67",       "adx14",          "gt_q",    0.67,  "trend"),
    ("ADX_WK",  "ADX<p33",       "adx14",          "lt_q",    0.33,  "trend"),
    ("PRG_HI",  "PrevRng>p67",   "prev_range_r",   "gt_q",    0.67,  "part"),
    ("PRG_VH",  "PrevRng>p80",   "prev_range_r",   "gt_q",    0.80,  "part"),
    ("PBD_HI",  "PrevBody>p67",  "prev_body_r",    "gt_q",    0.67,  "part"),
    ("PBP_HI",  "BodyPct>p60",   "prev_body_pct",  "gt_q",    0.60,  "part"),
    ("ASIA",    "Asia(0-7UTC)",  "hour_utc",       "hour_rng", (0,7),   "time"),
    ("EUR",     "Eur(8-15UTC)", "hour_utc",        "hour_rng", (8,15),  "time"),
    ("US",      "US(14-21UTC)", "hour_utc",        "hour_rng", (14,21), "time"),
    ("MIDWK",   "Wed-Thu",      "day_of_week",     "isin",    [2, 3],   "time"),
    ("EARLY",   "Mon-Tue",      "day_of_week",     "isin",    [0, 1],   "time"),
]
COND_BY_ID = {c[0]: c for c in CONDITIONS_DEF}
COND_IDS   = [c[0] for c in CONDITIONS_DEF]

# Collect all unique condition IDs needed by the 5 environments
NEEDED_CONDS = sorted({cid for e in R042_ENVS for cid in e[2]})

QUANT_FEATS = [
    "atr_rank", "real_vol_20", "bb_width", "ema_dist_pct",
    "adx14", "prev_range_r", "prev_body_r", "prev_body_pct",
]

# =============================================================================
# THRESHOLD LEARNING
# =============================================================================

def learn_thresholds(df_is: pd.DataFrame) -> dict:
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid in NEEDED_CONDS:
        _, _, feat, direction, param, _ = COND_BY_ID[cid]
        if direction in ("gt_fixed", "lt_fixed", "hour_rng", "isin"):
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

# =============================================================================
# CONDITION & ENVIRONMENT MASKS
# =============================================================================

def condition_mask(df: pd.DataFrame, cid: str, thr: dict) -> np.ndarray:
    _, _, feat, direction, _, _ = COND_BY_ID[cid]
    threshold = thr.get(cid, np.nan)
    n = len(df)
    if feat not in df.columns:
        return np.zeros(n, dtype=bool)
    col = df[feat].values
    nan_mask = np.isnan(col) if col.dtype.kind == 'f' else np.zeros(n, dtype=bool)
    if direction == "lt_q":
        return ~nan_mask & (col < threshold) if not (isinstance(threshold, float) and np.isnan(threshold)) else np.zeros(n, dtype=bool)
    elif direction in ("gt_q", "gt_q_pos"):
        return ~nan_mask & (col > threshold) if not (isinstance(threshold, float) and np.isnan(threshold)) else np.zeros(n, dtype=bool)
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


def env_mask(df: pd.DataFrame, eid: str, thr: dict) -> np.ndarray:
    conds = ENV_CONDS[eid]
    out   = condition_mask(df, conds[0], thr)
    for cid in conds[1:]:
        out &= condition_mask(df, cid, thr)
    return out

# =============================================================================
# RELVOL SIGNAL
# =============================================================================

def signal_relvol(df: pd.DataFrame, emask: np.ndarray) -> np.ndarray:
    rv  = df["rel_vol"].values
    c   = df["close"].values
    o   = df["open"].values
    pc  = df["prev_close"].values
    ok  = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & emask

# =============================================================================
# PRIORITY CASCADE  (de-duplicated portfolio signal)
# =============================================================================

def portfolio_signal(env_signals: list) -> tuple:
    """
    env_signals: list of (eid, signal_np_array) in priority order (highest first).
    Returns (combined_signal, attribution_array)
    attribution[i] = eid of highest-priority environment that fires on bar i,
                     or '' if none fires.
    """
    n       = len(env_signals[0][1])
    combined = np.zeros(n, dtype=bool)
    attr     = np.full(n, '', dtype=object)
    for eid, sig in env_signals:
        new_fires         = sig & ~combined
        combined         |= new_fires
        attr[new_fires]   = eid
    return combined, attr

# =============================================================================
# BACKTEST ENGINE  (identical to R041/R042, adds attribution field)
# =============================================================================

def run_backtest(df: pd.DataFrame, signal: np.ndarray,
                 sym: str, fold: int, eid: str,
                 attribution: np.ndarray | None = None) -> list:
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
                fired_eid = attribution[i - 1] if attribution is not None else eid
                trades.append({
                    "sym":        sym,
                    "fold":       fold,
                    "env":        fired_eid,
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

        if signal[i - 1]:
            a = atr_[i]
            if np.isnan(a) or a <= 0: continue
            ep_ = op_[i]
            if a / ep_ < min_sl: continue
            ep = ep_; st = ep - a; tk = ep + RR * a
            sz = min(CAPITAL * rf / a, (CAPITAL * max_lev) / ep)
            et = dts[i]; ei = i
            in_pos = True
    return trades

# =============================================================================
# STATISTICS
# =============================================================================

def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl


def metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "exp_r": 0.0, "net": 0.0,
                "sharpe": 0.0, "mdd": 0.0, "pnls": np.array([]), "equity": np.array([CAPITAL])}
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values
    wins = df["win"].values.astype(bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw   = pnl[wins].sum()       if nw else 0.0
    gl   = abs(pnl[~wins].sum()) if nl else 0.0
    pf   = safe_pf(gw, gl)
    wr   = nw / n
    equity = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak   = np.maximum.accumulate(equity)
    mdd    = float(((equity - peak) / peak).min())
    bpy    = 365 * 24
    ann    = (equity[-1] / CAPITAL) ** (bpy / max(n, 1)) - 1
    vol    = pnl.std() * math.sqrt(bpy) if n > 1 else 1e-9
    sharpe = ann / vol if vol > 0 else 0.0
    exp_r  = wr * RR - (1 - wr)
    return {"n": n, "wr": wr, "pf": pf, "exp_r": exp_r,
            "net": float(pnl.sum()), "sharpe": sharpe, "mdd": mdd,
            "pnls": pnl, "equity": equity}


def bootstrap_pf(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    pfs = [safe_pf(s[s > 0].sum(), abs(s[s < 0].sum()))
           for _ in range(n_iter)
           for s in [rng.choice(pnls, len(pnls), replace=True)]]
    return float(np.percentile(pfs, 5)), float(np.percentile(pfs, 50)), float(np.percentile(pfs, 95))


def monte_carlo(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return {"prob_profit": 0.0, "p5": CAPITAL, "p50": CAPITAL, "p95": CAPITAL,
                "finals": np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit": float((finals > CAPITAL).mean()),
            "p5":  float(np.percentile(finals,  5)),
            "p50": float(np.percentile(finals, 50)),
            "p95": float(np.percentile(finals, 95)),
            "finals": finals}


def loo_sym(sym_trades_dict: dict) -> dict:
    all_syms = [s for s in sym_trades_dict if sym_trades_dict[s]]
    return {omit: metrics([t for s, tl in sym_trades_dict.items() if s != omit for t in tl])["pf"]
            for omit in all_syms}


def loo_fld(all_trades: list) -> dict:
    folds = sorted({t["fold"] for t in all_trades})
    return {f: metrics([t for t in all_trades if t["fold"] != f])["pf"] for f in folds}


def full_stats(all_trades: list, sym_trades: dict) -> dict:
    m           = metrics(all_trades)
    b5, b50, b95 = bootstrap_pf(m["pnls"])
    mc          = monte_carlo(m["pnls"])
    ls          = loo_sym(sym_trades)
    lf          = loo_fld(all_trades)
    sf          = min(ls.values()) if ls else 0.0
    ff          = min(lf.values()) if lf else 0.0
    score       = sum([
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
            "b5": b5, "b50": b50, "b95": b95,
            "mc_p": mc["prob_profit"], "mc_p50": mc["p50"],
            "mc_finals": mc["finals"],
            "sym_floor": sf, "fold_floor": ff,
            "loo_sym": ls, "loo_fld": lf,
            "score": score, "verdict": verdict}

# =============================================================================
# DATA LOAD
# =============================================================================

print("=" * 80)
print("  QUANTLAB AI — RESEARCH #043")
print("  Independent Environment Portfolio Validation")
print("=" * 80)
print()
print("  Loading 1H data …")
all_dfs = {}
for sym in SYMBOLS:
    tag  = sym.replace("-", "_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path): continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) < MIN_BARS: continue
    all_dfs[sym] = add_features(df)
SYMBOLS = list(all_dfs.keys())
total_bars = sum(len(d) for d in all_dfs.values())
print(f"  {len(SYMBOLS)} symbols  ({total_bars:,} bars)")
print()
print("  Environments under test:")
for eid, label, conds in R042_ENVS:
    print(f"    {eid}: {label}")
print()

# =============================================================================
# WALK-FORWARD  (single pass, all environments + portfolios)
# =============================================================================

# Stores: env_sym_trades[eid][sym] = [trade_dicts]  (individual env attribution)
# Stores: port_sym_trades[pid][sym] = [trade_dicts]  (deduped portfolio trades)
env_sym_trades  = {eid: defaultdict(list) for eid in ENV_IDS}
port_sym_trades = {pid: defaultdict(list) for pid in PORT_IDS}

fold_env_n    = {eid: [] for eid in ENV_IDS}
fold_port_n   = {pid: [] for pid in PORT_IDS}

print("─" * 80)
print(f"  Walk-forward: {len(FOLDS)} folds × {len(SYMBOLS)} symbols")
print("─" * 80)

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_counts_env  = {eid: 0 for eid in ENV_IDS}
    fold_counts_port = {pid: 0 for pid in PORT_IDS}

    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[: int(N * is_end)]
        df_oos = df_full.iloc[int(N * is_end): int(N * oos_end)].reset_index(drop=True)
        if len(df_oos) < 100:
            continue

        thr = learn_thresholds(df_is)

        # Compute per-environment masks and signals
        env_signals = {}   # eid -> signal np.array
        env_emasks  = {}   # eid -> env mask
        for eid in ENV_IDS:
            em  = env_mask(df_oos, eid, thr)
            sig = signal_relvol(df_oos, em)
            env_signals[eid] = sig
            env_emasks[eid]  = em

        # ── Individual environment backtests ──────────────────────────────────
        for eid in ENV_IDS:
            tl = run_backtest(df_oos, env_signals[eid], sym, fold_idx, eid)
            env_sym_trades[eid][sym].extend(tl)
            fold_counts_env[eid] += len(tl)

        # ── Portfolio backtests (priority-cascaded, de-duplicated) ─────────────
        for pid, port_env_ids in PORTFOLIOS:
            ordered = [(eid, env_signals[eid]) for eid in port_env_ids]
            combined_sig, attr = portfolio_signal(ordered)
            tl = run_backtest(df_oos, combined_sig, sym, fold_idx, pid, attribution=attr)
            port_sym_trades[pid][sym].extend(tl)
            fold_counts_port[pid] += len(tl)

    print(f"  Fold {fold_idx}  (IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%)")
    print(f"    Individual:  " +
          "  ".join(f"{eid}={fold_counts_env[eid]:3d}" for eid in ENV_IDS))
    print(f"    Portfolios:  " +
          "  ".join(f"P{pid}={fold_counts_port[pid]:3d}" for pid in PORT_IDS))

    for eid in ENV_IDS:
        fold_env_n[eid].append(fold_counts_env[eid])
    for pid in PORT_IDS:
        fold_port_n[pid].append(fold_counts_port[pid])
    print()

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print("  Computing statistics …")

# Individual results
env_results = {}
for eid in ENV_IDS:
    flat = [t for tl in env_sym_trades[eid].values() for t in tl]
    env_results[eid] = {
        "id": eid, "label": ENV_LABEL[eid],
        "sym_trades": dict(env_sym_trades[eid]),
        "_flat": flat,
        **full_stats(flat, dict(env_sym_trades[eid]))
    }

# Portfolio results
port_results = {}
for pid in PORT_IDS:
    flat = [t for tl in port_sym_trades[pid].values() for t in tl]
    port_results[pid] = {
        "id": pid,
        "label": "Port " + pid + ": " + " + ".join(PORT_ENVS[pid]),
        "envs": PORT_ENVS[pid],
        "sym_trades": dict(port_sym_trades[pid]),
        "_flat": flat,
        **full_stats(flat, dict(port_sym_trades[pid]))
    }

# =============================================================================
# OVERLAP / CORRELATION MATRIX
# =============================================================================

def trade_set(trades): return {(t["sym"], t["entry_time"]) for t in trades}

env_trade_sets = {eid: trade_set(env_results[eid]["_flat"]) for eid in ENV_IDS}
overlap_matrix = {}
for e1 in ENV_IDS:
    for e2 in ENV_IDS:
        s1, s2 = env_trade_sets[e1], env_trade_sets[e2]
        union  = s1 | s2
        overlap_matrix[(e1, e2)] = len(s1 & s2) / max(len(union), 1)

# =============================================================================
# INCREMENTAL CONTRIBUTION ANALYSIS  (Q4, Q5)
# =============================================================================

# For each portfolio, compute PF if that environment is REMOVED
incr = {}
for pid in PORT_IDS:
    envs = PORT_ENVS[pid]
    incr[pid] = {}
    for eid in envs:
        remaining = [e for e in envs if e != eid]
        if not remaining:
            incr[pid][eid] = {"n": 0, "pf": 0.0, "delta_pf": 0.0}
            continue
        # Reconstruct trades without this env using attribution
        flat_without = [t for t in port_results[pid]["_flat"] if t["env"] != eid]
        sym_without  = defaultdict(list)
        for t in flat_without:
            sym_without[t["sym"]].append(t)
        m_wo = metrics(flat_without)
        delta = port_results[pid]["pf"] - m_wo["pf"]
        incr[pid][eid] = {
            "n_contrib":  sum(1 for t in port_results[pid]["_flat"] if t["env"] == eid),
            "pf_without": m_wo["pf"],
            "delta_pf":   delta,  # positive = env adds PF; negative = env hurts PF
        }

# =============================================================================
# RESULTS TABLES
# =============================================================================

SEP = "═" * 110

def fmt_row(label, r, *, indent=2):
    pad = " " * indent
    return (f"{pad}{label:35s}  "
            f"n={r['n']:4d}  WR={r['wr']*100:4.1f}%  PF={r['pf']:6.3f}  "
            f"p50={r['b50']:6.3f}  [{r['b5']:.3f},{r['b95']:.3f}]  "
            f"MC={r['mc_p']*100:4.0f}%  MDD={r['mdd']:5.1%}  "
            f"ExpR={r['exp_r']:+.3f}  LOO-S={r['sym_floor']:.3f}  "
            f"LOO-F={r['fold_floor']:.3f}  {r['score']}/7  {r['verdict']}")

print()
print(SEP)
print("  R043 — INDIVIDUAL ENVIRONMENT SCORECARD")
print(SEP)
hdr = (f"  {'Label':35s}  {'n':>4}  {'WR':>6}  {'PF':>6}  "
       f"{'p50':>6}  {'Boot CI':>14}  {'MC%':>5}  {'MDD':>6}  "
       f"{'ExpR':>7}  {'LOO-S':>6}  {'LOO-F':>6}  {'Sc':>4}  Verdict")
print(hdr)
print("  " + "─" * 106)
for eid in ENV_IDS:
    r = env_results[eid]
    print(fmt_row(f"{eid}: {ENV_LABEL[eid][:30]}", r))

print()
print(SEP)
print("  R043 — PORTFOLIO COMPARISON TABLE")
print(SEP)
print(hdr)
print("  " + "─" * 106)
for pid in PORT_IDS:
    r = port_results[pid]
    print(fmt_row(f"Port {pid} ({', '.join(PORT_ENVS[pid])})", r))

# =============================================================================
# RESEARCH QUESTIONS
# =============================================================================

print()
print(SEP)
print("  RESEARCH QUESTIONS")
print(SEP)

# Q1: Does combining increase frequency without materially reducing PF?
print("\n  Q1.  Does combining environments increase frequency without material PF loss?")
print("  " + "─" * 80)
base_pf = env_results["E1"]["pf"]
base_n  = env_results["E1"]["n"]
for pid in PORT_IDS:
    pr    = port_results[pid]
    n_raw = sum(env_results[e]["n"] for e in PORT_ENVS[pid])
    dedup_loss = n_raw - pr["n"]
    pf_delta   = pr["pf"] - base_pf
    print(f"  Port {pid} ({', '.join(PORT_ENVS[pid])}): "
          f"n={pr['n']} (raw={n_raw}, dedup_removed={dedup_loss})  "
          f"PF={pr['pf']:.3f} ({pf_delta:+.3f} vs E1)  "
          f"{'✓ FREQ+QUAL' if pr['n'] > base_n and pr['pf'] > PROM_PF else '△ CHECK'}")

# Q2: Best frequency/quality balance
print("\n  Q2.  Which portfolio achieves the best balance between frequency and quality?")
print("  " + "─" * 80)
# Score = PF × sqrt(n) / sqrt(E1_n)  — harmonic of quality and frequency
q2_scored = sorted(PORT_IDS, key=lambda p: port_results[p]["pf"] * (port_results[p]["n"] ** 0.5),
                   reverse=True)
for pid in q2_scored:
    pr = port_results[pid]
    score_val = pr["pf"] * (pr["n"] ** 0.5)
    print(f"  Port {pid}: PF×√n = {score_val:.2f}  (PF={pr['pf']:.3f}  n={pr['n']}  p50={pr['b50']:.3f})")
best_balance = q2_scored[0]
print(f"\n  → Best balance: Portfolio {best_balance}")

# Q3: PROMOTE criteria
print(f"\n  Q3.  Which portfolio satisfies all PROMOTE criteria?"
      f"  (PF>{PROM_PF}, n≥{PROM_N}, Boot>{PROM_BOOT}, MC>{PROM_MC*100:.0f}%, "
      f"LOO-S>1.0, LOO-F>1.0, MDD<{PROM_MDD*100:.0f}%)")
print("  " + "─" * 80)
promote_ports = [pid for pid in PORT_IDS if port_results[pid]["verdict"] == "PROMOTE"]
if promote_ports:
    for pid in promote_ports:
        pr = port_results[pid]
        print(f"  ★ PROMOTE: Portfolio {pid}  PF={pr['pf']:.3f}  n={pr['n']}  "
              f"Score={pr['score']}/7  MDD={pr['mdd']:.1%}")
else:
    print("  No portfolio reaches PROMOTE on all 7 criteria.")
    for pid in PORT_IDS:
        pr = port_results[pid]
        failing = []
        if pr["pf"]         <= PROM_PF:    failing.append(f"PF={pr['pf']:.3f}≤{PROM_PF}")
        if pr["n"]          <  PROM_N:     failing.append(f"n={pr['n']}<{PROM_N}")
        if pr["b50"]        <= PROM_BOOT:  failing.append(f"Boot={pr['b50']:.3f}≤{PROM_BOOT}")
        if pr["mc_p"]       <= PROM_MC:    failing.append(f"MC={pr['mc_p']*100:.0f}%")
        if pr["sym_floor"]  <= 1.0:        failing.append(f"LOO-S={pr['sym_floor']:.3f}")
        if pr["fold_floor"] <= 1.0:        failing.append(f"LOO-F={pr['fold_floor']:.3f}")
        if abs(pr["mdd"])   >= PROM_MDD:   failing.append(f"MDD={pr['mdd']:.1%}")
        status = "✓ PASS" if not failing else f"✗ FAIL [{', '.join(failing)}]"
        print(f"  Port {pid}: {status}")

# Q4: Incremental contribution
print(f"\n  Q4.  Which environment contributes the most incremental edge?")
print("  " + "─" * 80)
best_port = max(PORT_IDS, key=lambda p: port_results[p]["score"] * 10 + port_results[p]["pf"])
for eid in PORT_ENVS[best_port]:
    ic = incr[best_port][eid]
    direction = "ADDS" if ic["delta_pf"] > 0 else "HURTS"
    print(f"  {eid} in Port {best_port}: n_contrib={ic['n_contrib']:3d}  "
          f"PF_without={ic['pf_without']:.3f}  ΔPF={ic['delta_pf']:+.3f}  → {direction}")

# Q5: Does any env reduce quality?
print(f"\n  Q5.  Does any environment reduce portfolio quality?")
print("  " + "─" * 80)
for pid in PORT_IDS:
    for eid in PORT_ENVS[pid]:
        ic = incr[pid][eid]
        if ic["delta_pf"] < -0.05:
            print(f"  ⚠ {eid} in Port {pid}: removing it improves PF by {-ic['delta_pf']:+.3f} "
                  f"(Port PF={port_results[pid]['pf']:.3f} → PF_without={ic['pf_without']:.3f})")
        elif ic["delta_pf"] < 0:
            print(f"  △ {eid} in Port {pid}: marginal negative impact ΔPF={ic['delta_pf']:+.3f} "
                  f"(within tolerance)")
        else:
            print(f"  ✓ {eid} in Port {pid}: additive ΔPF={ic['delta_pf']:+.3f}")

# Q6: Symbol and time diversification
print(f"\n  Q6.  Is the recommended portfolio diversified?")
print("  " + "─" * 80)
rec_port = best_balance
pr_flat  = port_results[rec_port]["_flat"]

# Symbol distribution
sym_counts = defaultdict(int)
for t in pr_flat:
    sym_counts[t["sym"]] += 1
total_t = max(len(pr_flat), 1)
sym_shares = {s: sym_counts[s] / total_t for s in SYMBOLS if sym_counts[s] > 0}
hhi = sum(v ** 2 for v in sym_shares.values())
print(f"  Recommended portfolio: {rec_port}  (n={len(pr_flat)})")
print(f"  Symbol HHI (0=perfect spread, 1=one symbol): {hhi:.3f}")
print(f"  Active symbols: {len(sym_shares)}/{len(SYMBOLS)}")
top5_syms = sorted(sym_shares, key=lambda s: -sym_shares[s])[:5]
for s in top5_syms:
    bar = "█" * int(sym_shares[s] * 30)
    print(f"    {s:20s}  {sym_counts[s]:4d} trades  {sym_shares[s]*100:4.1f}%  {bar}")

# Environment attribution in portfolio
env_attr_counts = defaultdict(int)
for t in pr_flat:
    env_attr_counts[t["env"]] += 1
print(f"\n  Attribution in Port {rec_port}:")
for eid in PORT_ENVS[rec_port]:
    cnt  = env_attr_counts.get(eid, 0)
    pct  = cnt / total_t * 100
    bar  = "█" * int(pct / 2)
    print(f"    {eid}: {cnt:4d} trades  {pct:4.1f}%  {bar}")

# Fold distribution
fold_counts_rec = defaultdict(int)
for t in pr_flat:
    fold_counts_rec[t["fold"]] += 1
print(f"\n  Fold distribution (Port {rec_port}):")
for f in sorted(fold_counts_rec):
    cnt = fold_counts_rec[f]
    bar = "█" * int(cnt / max(fold_counts_rec.values()) * 20)
    print(f"    Fold {f}: {cnt:4d} trades  {bar}")

print()

# =============================================================================
# OVERLAP MATRIX
# =============================================================================

print(SEP)
print("  TRADE OVERLAP MATRIX  (Jaccard similarity between environments)")
print(SEP)
print()
print("  " + "    ".join(f"  {eid:4s}" for eid in ENV_IDS))
print("  " + "─" * 45)
for e1 in ENV_IDS:
    row = f"  {e1}"
    for e2 in ENV_IDS:
        ov = overlap_matrix[(e1, e2)]
        row += f"  {ov:5.1%}"
    print(row)
print()

# =============================================================================
# FINAL VERDICT
# =============================================================================

# Best portfolio overall
best_score_port = max(PORT_IDS, key=lambda p: (port_results[p]["score"], port_results[p]["pf"]))
br = port_results[best_score_port]

print(SEP)
print("  FINAL VERDICT")
print(SEP)
print(f"\n  Recommended production portfolio: Portfolio {best_score_port}")
print(f"  Environments: {' + '.join(PORT_ENVS[best_score_port])}")
for eid in PORT_ENVS[best_score_port]:
    print(f"    {eid}: {ENV_LABEL[eid]}")
print()
print(f"  {'Metric':<30} {'Value':>12}  {'Criterion':>12}  {'Pass?':>6}")
print(f"  {'─'*30} {'─'*12}  {'─'*12}  {'─'*6}")
criteria = [
    ("Profit Factor",      br["pf"],         f">{PROM_PF}",    br["pf"] > PROM_PF),
    ("Trade Count",        br["n"],           f"≥{PROM_N}",     br["n"] >= PROM_N),
    ("Bootstrap p50",      br["b50"],         f">{PROM_BOOT}",  br["b50"] > PROM_BOOT),
    ("Monte Carlo P%",     br["mc_p"]*100,    f">{PROM_MC*100:.0f}%", br["mc_p"] > PROM_MC),
    ("LOO Symbol Floor",   br["sym_floor"],   ">1.00",          br["sym_floor"] > 1.0),
    ("LOO Fold Floor",     br["fold_floor"],  ">1.00",          br["fold_floor"] > 1.0),
    ("Max Drawdown",       abs(br["mdd"])*100,f"<{PROM_MDD*100:.0f}%", abs(br["mdd"]) < PROM_MDD),
]
for name, val, crit, passes in criteria:
    tick = "✓" if passes else "✗"
    print(f"  {name:<30} {val:>12.3f}  {crit:>12}  {tick:>6}")
print()
print(f"  Score:   {br['score']}/7")
print(f"  Verdict: {br['verdict']}")
print()

# =============================================================================
# CHARTS
# =============================================================================

print("─" * 80)
print("  Generating charts …")

DARK_BG  = "#0e1117"
GRID_CLR = "#1e2430"
TEXT_CLR = "#e0e0e0"
ACCENT   = "#4fc3f7"
GREEN    = "#69f0ae"
RED      = "#ef5350"
GOLD     = "#ffd54f"
PURPLE   = "#ce93d8"
ORANGE   = "#ffb74d"

ENV_COLORS  = {"E1": GREEN, "E2": ACCENT, "E3": GOLD, "E4": ORANGE, "E5": PURPLE}
PORT_COLORS = {"A": "#64b5f6", "B": "#4db6ac", "C": "#aed581", "D": "#fff176"}

def _style(ax, title=""):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    for sp in ax.spines.values(): sp.set_color(GRID_CLR)
    ax.xaxis.label.set_color(TEXT_CLR); ax.yaxis.label.set_color(TEXT_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5)
    if title: ax.set_title(title, color=TEXT_CLR, fontsize=8, pad=4)


# ── Chart 1: Equity curves — individual envs + portfolios ────────────────────
fig, axes = plt.subplots(3, 3, figsize=(14, 9), facecolor=DARK_BG)
axes = axes.flatten()
for i, eid in enumerate(ENV_IDS):
    ax = axes[i]; _style(ax, f"{eid}  PF={env_results[eid]['pf']:.3f}  n={env_results[eid]['n']}")
    eq = env_results[eid]["equity"]
    x  = np.arange(len(eq))
    ax.plot(x, eq, color=ENV_COLORS[eid], lw=1.2)
    ax.fill_between(x, CAPITAL, eq, where=eq >= CAPITAL, alpha=0.15, color=GREEN)
    ax.fill_between(x, CAPITAL, eq, where=eq <  CAPITAL, alpha=0.15, color=RED)
    ax.axhline(CAPITAL, color=TEXT_CLR, lw=0.5, ls="--")
for j, pid in enumerate(PORT_IDS):
    ax = axes[5 + j]; pr = port_results[pid]
    _style(ax, f"Port {pid}  PF={pr['pf']:.3f}  n={pr['n']}")
    eq = pr["equity"]; x = np.arange(len(eq))
    ax.plot(x, eq, color=PORT_COLORS[pid], lw=1.5)
    ax.fill_between(x, CAPITAL, eq, where=eq >= CAPITAL, alpha=0.15, color=GREEN)
    ax.fill_between(x, CAPITAL, eq, where=eq <  CAPITAL, alpha=0.15, color=RED)
    ax.axhline(CAPITAL, color=TEXT_CLR, lw=0.5, ls="--")
plt.suptitle("R043 — Equity Curves: Individual Environments & Portfolios",
             color=TEXT_CLR, fontsize=10)
plt.tight_layout()
p1 = f"{OUT}/r043_equity_curves.png"
plt.savefig(p1, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p1}")

# ── Chart 2: Portfolio comparison — PF vs n ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)

ax = axes[0]; _style(ax, "PF vs Trade Count  (Env=circle  Port=diamond)")
for eid in ENV_IDS:
    r = env_results[eid]
    ax.scatter(r["n"], r["pf"], color=ENV_COLORS[eid], s=80, zorder=5, marker="o",
               label=eid)
    ax.annotate(eid, (r["n"], r["pf"]), fontsize=7, color=TEXT_CLR,
                xytext=(4, 2), textcoords="offset points")
for pid in PORT_IDS:
    pr = port_results[pid]
    ax.scatter(pr["n"], pr["pf"], color=PORT_COLORS[pid], s=120, zorder=5, marker="D",
               label=f"P{pid}")
    ax.annotate(f"P{pid}", (pr["n"], pr["pf"]), fontsize=7, color=TEXT_CLR,
                xytext=(4, 2), textcoords="offset points")
ax.axhline(PROM_PF, color=RED, lw=0.8, ls="--", label=f"PF={PROM_PF}")
ax.axvline(PROM_N,  color=GOLD, lw=0.8, ls="--", label=f"n={PROM_N}")
ax.set_xlabel("Trade Count (n)"); ax.set_ylabel("Profit Factor")
ax.legend(fontsize=6, facecolor=DARK_BG, labelcolor=TEXT_CLR, ncol=2)

ax = axes[1]; _style(ax, "Bootstrap PF CI  (p5—p95 bar, ◆=p50, ●=actual)")
items = ([(eid, env_results[eid], ENV_COLORS[eid])  for eid in ENV_IDS] +
         [(f"P{pid}", port_results[pid], PORT_COLORS[pid]) for pid in PORT_IDS])
ys = np.arange(len(items))
for i, (lbl, r, clr) in enumerate(items):
    ax.barh(i, r["b95"] - r["b5"], left=r["b5"], color=clr, alpha=0.35, height=0.55)
    ax.plot([r["b50"]], [i], "D", color=clr,      ms=6, zorder=5)
    ax.plot([r["pf"]],  [i], "o", color=TEXT_CLR, ms=4, zorder=5)
ax.axvline(PROM_PF, color=RED, lw=0.8, ls="--")
ax.axvline(1.0,     color=GRID_CLR, lw=0.6, ls=":")
ax.set_yticks(ys); ax.set_yticklabels([lbl for lbl, _, _ in items], fontsize=7)
ax.set_xlabel("Profit Factor"); ax.invert_yaxis()
plt.tight_layout()
p2 = f"{OUT}/r043_pf_comparison.png"
plt.savefig(p2, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p2}")

# ── Chart 3: Overlap/correlation matrix ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
_style(ax, "Trade Overlap Matrix (Jaccard)")
mat = np.array([[overlap_matrix[(e1, e2)] for e2 in ENV_IDS] for e1 in ENV_IDS])
cmap_ov = LinearSegmentedColormap.from_list("ov", [DARK_BG, ACCENT, GREEN])
im = ax.imshow(mat, cmap=cmap_ov, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(ENV_IDS))); ax.set_xticklabels(ENV_IDS, fontsize=8, color=TEXT_CLR)
ax.set_yticks(range(len(ENV_IDS))); ax.set_yticklabels(ENV_IDS, fontsize=8, color=TEXT_CLR)
for i in range(len(ENV_IDS)):
    for j in range(len(ENV_IDS)):
        ax.text(j, i, f"{mat[i,j]:.0%}", ha="center", va="center",
                fontsize=8, color="white" if mat[i,j] < 0.5 else DARK_BG)
cb = plt.colorbar(im, ax=ax); cb.ax.yaxis.label.set_color(TEXT_CLR)
cb.ax.tick_params(colors=TEXT_CLR)
plt.tight_layout()
p3 = f"{OUT}/r043_overlap_matrix.png"
plt.savefig(p3, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p3}")

# ── Chart 4: Incremental contribution ────────────────────────────────────────
fig, axes = plt.subplots(1, len(PORT_IDS), figsize=(13, 4), facecolor=DARK_BG, sharey=False)
for j, pid in enumerate(PORT_IDS):
    ax = axes[j]; _style(ax, f"Port {pid}: ΔPF if env removed")
    envs_in = PORT_ENVS[pid]
    deltas  = [incr[pid][eid]["delta_pf"] for eid in envs_in]
    colors  = [GREEN if d >= 0 else RED for d in deltas]
    bars    = ax.barh(envs_in, deltas, color=colors, alpha=0.8)
    ax.axvline(0, color=TEXT_CLR, lw=0.7)
    ax.set_xlabel("ΔPF"); ax.tick_params(axis="y", labelsize=7)
plt.suptitle("Incremental PF Contribution (positive = env adds quality)",
             color=TEXT_CLR, fontsize=9)
plt.tight_layout()
p4 = f"{OUT}/r043_incremental.png"
plt.savefig(p4, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p4}")

# ── Chart 5: Per-symbol PF heatmap for best portfolio ────────────────────────
pr_best = port_results[best_balance]
sym_list = sorted(SYMBOLS)
fig, ax = plt.subplots(figsize=(max(8, len(sym_list) * 0.55), 3.5), facecolor=DARK_BG)
_style(ax, f"Per-Symbol PF — Portfolio {best_balance}")
sym_pf = []
for sym in sym_list:
    tl = pr_best["sym_trades"].get(sym, [])
    sym_pf.append(metrics(tl)["pf"] if tl else 0.0)
sym_pf = np.array(sym_pf).reshape(1, -1)
cmap_sym = LinearSegmentedColormap.from_list("pf", [RED, DARK_BG, GREEN])
im = ax.imshow(sym_pf, aspect="auto", cmap=cmap_sym, vmin=0, vmax=3)
ax.set_xticks(range(len(sym_list)))
ax.set_xticklabels([s.split("-")[0] for s in sym_list], rotation=55, fontsize=6.5, color=TEXT_CLR)
ax.set_yticks([]); ax.set_ylabel(f"P{best_balance}", color=TEXT_CLR)
for j, pf_val in enumerate(sym_pf[0]):
    if pf_val > 0:
        ax.text(j, 0, f"{pf_val:.2f}", ha="center", va="center", fontsize=6, color="white")
plt.colorbar(im, ax=ax, orientation="vertical", label="PF").ax.yaxis.label.set_color(TEXT_CLR)
plt.tight_layout()
p5 = f"{OUT}/r043_per_symbol_pf.png"
plt.savefig(p5, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p5}")

# ── Chart 6: Fold stability ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK_BG)
fold_labels = [f"F{i+1}" for i in range(len(FOLDS))]

ax = axes[0]; _style(ax, "Fold-by-Fold Trade Count")
x = np.arange(len(FOLDS)); width = 0.15
for k, eid in enumerate(ENV_IDS):
    ax.bar(x + k * width, fold_env_n[eid], width, label=eid, color=ENV_COLORS[eid], alpha=0.8)
ax.set_xticks(x + width * 2); ax.set_xticklabels(fold_labels, color=TEXT_CLR)
ax.legend(fontsize=6, facecolor=DARK_BG, labelcolor=TEXT_CLR)
ax.set_ylabel("Trades")

ax = axes[1]; _style(ax, "Fold-by-Fold Trade Count — Portfolios")
for k, pid in enumerate(PORT_IDS):
    ax.plot(range(len(FOLDS)), fold_port_n[pid], marker="o", color=PORT_COLORS[pid],
            label=f"P{pid}", lw=1.5)
ax.set_xticks(range(len(FOLDS))); ax.set_xticklabels(fold_labels, color=TEXT_CLR)
ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR)
ax.set_ylabel("Trades")
plt.tight_layout()
p6 = f"{OUT}/r043_fold_stability.png"
plt.savefig(p6, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p6}")

# ── Chart 7: Monte Carlo fan for recommended portfolio ────────────────────────
pr_rec  = port_results[best_score_port]
fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
_style(ax, f"Monte Carlo Distribution — Portfolio {best_score_port} (n_iter={N_BOOT})")
finals = pr_rec["mc_finals"]
ax.hist(finals, bins=60, color=PORT_COLORS.get(best_score_port, ACCENT), alpha=0.7, edgecolor="none")
ax.axvline(CAPITAL, color=RED, lw=1, ls="--", label="Breakeven")
ax.axvline(np.percentile(finals, 5),  color=GOLD, lw=1, ls=":",  label="p5")
ax.axvline(np.percentile(finals, 50), color=GREEN, lw=1.5,       label="p50")
ax.axvline(np.percentile(finals, 95), color=ACCENT, lw=1, ls=":", label="p95")
ax.set_xlabel("Terminal Capital ($)"); ax.set_ylabel("Frequency")
ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR)
pwin = (finals > CAPITAL).mean() * 100
ax.set_title(f"Portfolio {best_score_port} — P(profit)={pwin:.1f}%",
             color=TEXT_CLR, fontsize=9)
plt.tight_layout()
p7 = f"{OUT}/r043_monte_carlo.png"
plt.savefig(p7, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p7}")

# ── Chart 8: Dashboard ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

# A: PF vs n
ax_a = fig.add_subplot(gs[0, 0:2])
_style(ax_a, "PF vs Trade Count")
for eid in ENV_IDS:
    r = env_results[eid]
    ax_a.scatter(r["n"], r["pf"], color=ENV_COLORS[eid], s=90, marker="o", label=eid, zorder=5)
    ax_a.annotate(eid, (r["n"], r["pf"]), fontsize=7, color=TEXT_CLR, xytext=(3,2), textcoords="offset points")
for pid in PORT_IDS:
    pr = port_results[pid]
    ax_a.scatter(pr["n"], pr["pf"], color=PORT_COLORS[pid], s=130, marker="D", label=f"P{pid}", zorder=5)
    ax_a.annotate(f"P{pid}", (pr["n"], pr["pf"]), fontsize=7, color=TEXT_CLR, xytext=(3,2), textcoords="offset points")
ax_a.axhline(PROM_PF, color=RED, lw=0.8, ls="--"); ax_a.axvline(PROM_N, color=GOLD, lw=0.8, ls="--")
ax_a.legend(fontsize=6, facecolor=DARK_BG, labelcolor=TEXT_CLR, ncol=3)
ax_a.set_xlabel("n"); ax_a.set_ylabel("PF")

# B: Overlap matrix
ax_b = fig.add_subplot(gs[0, 2:4])
_style(ax_b, "Overlap Matrix")
im_b = ax_b.imshow(mat, cmap=cmap_ov, vmin=0, vmax=1)
ax_b.set_xticks(range(len(ENV_IDS))); ax_b.set_xticklabels(ENV_IDS, fontsize=8, color=TEXT_CLR)
ax_b.set_yticks(range(len(ENV_IDS))); ax_b.set_yticklabels(ENV_IDS, fontsize=8, color=TEXT_CLR)
for i in range(len(ENV_IDS)):
    for j in range(len(ENV_IDS)):
        ax_b.text(j, i, f"{mat[i,j]:.0%}", ha="center", va="center",
                  fontsize=8, color="white" if mat[i,j] < 0.5 else DARK_BG)

# C-F: Portfolio equity curves
for k, pid in enumerate(PORT_IDS):
    ax_c = fig.add_subplot(gs[1, k])
    pr   = port_results[pid]
    _style(ax_c, f"P{pid}  PF={pr['pf']:.3f}  n={pr['n']}")
    eq = pr["equity"]; x = np.arange(len(eq))
    ax_c.plot(x, eq, color=PORT_COLORS[pid], lw=1.4)
    ax_c.fill_between(x, CAPITAL, eq, where=eq >= CAPITAL, alpha=0.15, color=GREEN)
    ax_c.fill_between(x, CAPITAL, eq, where=eq <  CAPITAL, alpha=0.15, color=RED)
    ax_c.axhline(CAPITAL, color=TEXT_CLR, lw=0.5, ls="--")
    ax_c.tick_params(labelsize=6)

# G: Incremental PF deltas for best portfolio
ax_g = fig.add_subplot(gs[2, 0:2])
_style(ax_g, f"Incremental ΔPF — Port {best_score_port}")
envs_bp  = PORT_ENVS[best_score_port]
deltas_bp = [incr[best_score_port][eid]["delta_pf"] for eid in envs_bp]
clrs_bp   = [GREEN if d >= 0 else RED for d in deltas_bp]
ax_g.barh(envs_bp, deltas_bp, color=clrs_bp, alpha=0.8)
ax_g.axvline(0, color=TEXT_CLR, lw=0.7)
ax_g.set_xlabel("ΔPF (positive = env adds edge)")

# H: Summary text
ax_h = fig.add_subplot(gs[2, 2:4])
ax_h.set_facecolor(DARK_BG); ax_h.axis("off")
lines = [
    "R043  PORTFOLIO VALIDATION SUMMARY",
    "",
    f"Dataset: {len(SYMBOLS)} syms · 1H · 5-fold WF · OOS only",
    "",
    f"{'':4s}{'Label':<35} {'n':>5}  {'PF':>6}  {'p50':>6}  {'Sc':>4}  Verdict",
    "─" * 64,
]
for eid in ENV_IDS:
    r = env_results[eid]
    lines.append(f"{eid:4s} {ENV_LABEL[eid][:33]:<35} {r['n']:>5}  {r['pf']:>6.3f}  {r['b50']:>6.3f}  {r['score']:>2}/7  {r['verdict']}")
lines += ["", "Portfolios:"]
for pid in PORT_IDS:
    pr = port_results[pid]
    lines.append(f"P{pid}  ({', '.join(PORT_ENVS[pid]):<14})  n={pr['n']:4d}  PF={pr['pf']:.3f}  {pr['score']}/7  {pr['verdict']}")
lines += ["", f"Recommended: Portfolio {best_score_port}  → {br['verdict']}"]
ax_h.text(0.02, 0.98, "\n".join(lines),
          transform=ax_h.transAxes, fontsize=6.5, color=TEXT_CLR,
          va="top", ha="left", fontfamily="monospace",
          bbox=dict(facecolor=GRID_CLR, alpha=0.5, edgecolor="none", pad=5))

plt.suptitle("QUANTLAB AI — R043  Portfolio Validation Dashboard",
             color=TEXT_CLR, fontsize=11, y=1.005)
p8 = f"{OUT}/r043_dashboard.png"
plt.savefig(p8, dpi=130, facecolor=DARK_BG, bbox_inches="tight"); plt.close()
print(f"  → {p8}")

# =============================================================================
# CSV OUTPUTS
# =============================================================================

# Scorecard CSV (individual + portfolios)
sc_rows = []
for eid in ENV_IDS:
    r = env_results[eid]
    sc_rows.append({
        "type": "individual", "id": eid, "label": ENV_LABEL[eid],
        "n": r["n"], "win_rate": round(r["wr"], 4), "profit_factor": round(r["pf"], 4),
        "expectancy": round(r["exp_r"], 4), "boot_p50": round(r["b50"], 4),
        "boot_p5": round(r["b5"], 4), "boot_p95": round(r["b95"], 4),
        "mc_prob": round(r["mc_p"], 4), "mdd": round(r["mdd"], 4),
        "net_pnl": round(r["net"], 2), "sharpe": round(r["sharpe"], 4),
        "sym_floor": round(r["sym_floor"], 4), "fold_floor": round(r["fold_floor"], 4),
        "score": r["score"], "verdict": r["verdict"],
    })
for pid in PORT_IDS:
    pr = port_results[pid]
    sc_rows.append({
        "type": "portfolio", "id": f"Port_{pid}", "label": pr["label"],
        "n": pr["n"], "win_rate": round(pr["wr"], 4), "profit_factor": round(pr["pf"], 4),
        "expectancy": round(pr["exp_r"], 4), "boot_p50": round(pr["b50"], 4),
        "boot_p5": round(pr["b5"], 4), "boot_p95": round(pr["b95"], 4),
        "mc_prob": round(pr["mc_p"], 4), "mdd": round(pr["mdd"], 4),
        "net_pnl": round(pr["net"], 2), "sharpe": round(pr["sharpe"], 4),
        "sym_floor": round(pr["sym_floor"], 4), "fold_floor": round(pr["fold_floor"], 4),
        "score": pr["score"], "verdict": pr["verdict"],
    })
csv1 = f"{OUT}/r043_scorecard.csv"
pd.DataFrame(sc_rows).to_csv(csv1, index=False)
print(f"  → {csv1}")

# Incremental contribution CSV
ic_rows = []
for pid in PORT_IDS:
    for eid in PORT_ENVS[pid]:
        ic = incr[pid][eid]
        ic_rows.append({
            "portfolio": pid, "env": eid,
            "n_attributed": ic["n_contrib"],
            "pf_without": round(ic["pf_without"], 4),
            "delta_pf": round(ic["delta_pf"], 4),
            "additive": ic["delta_pf"] >= 0,
        })
csv2 = f"{OUT}/r043_incremental.csv"
pd.DataFrame(ic_rows).to_csv(csv2, index=False)
print(f"  → {csv2}")

# =============================================================================
# JOURNAL MARKDOWN
# =============================================================================

def make_table(rows, cols):
    hdr = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [hdr, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c[1], "")) for c in cols) + " |")
    return "\n".join(lines)

sc_cols = [
    ("ID","id"),("n","n"),("WR","win_rate"),("PF","profit_factor"),
    ("p50","boot_p50"),("MC%","mc_prob"),("MDD","mdd"),
    ("LOO-S","sym_floor"),("LOO-F","fold_floor"),("Score","score"),("Verdict","verdict"),
]

jmd = [
    f"# QUANTLAB AI — R043 Research Journal",
    f"",
    f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
    f"**Research ID:** R043  ",
    f"**Title:** Independent Environment Portfolio Validation  ",
    f"**Dataset:** 1H · {len(SYMBOLS)} symbols · {total_bars:,} bars · 5-fold WF  ",
    f"",
    f"---",
    f"",
    f"## Environments Under Test",
    f"",
]
for eid, label, conds in R042_ENVS:
    jmd.append(f"- **{eid}:** {label}")
jmd += [
    f"",
    f"## Individual Scorecard",
    f"",
    make_table(sc_rows[:5], sc_cols),
    f"",
    f"## Portfolio Results",
    f"",
    make_table(sc_rows[5:], sc_cols),
    f"",
    f"## Overlap Matrix",
    f"",
    "| | " + " | ".join(ENV_IDS) + " |",
    "| --- |" + " --- |" * len(ENV_IDS),
]
for e1 in ENV_IDS:
    row = f"| **{e1}** |"
    for e2 in ENV_IDS:
        row += f" {overlap_matrix[(e1,e2)]:.0%} |"
    jmd.append(row)

jmd += [
    f"",
    f"## Research Questions",
    f"",
    f"**Q1 — Frequency vs Quality:**",
]
for pid in PORT_IDS:
    pr = port_results[pid]
    n_raw = sum(env_results[e]["n"] for e in PORT_ENVS[pid])
    jmd.append(f"- Port {pid}: n={pr['n']} (raw={n_raw})  PF={pr['pf']:.3f}  "
               f"{'✓' if pr['n'] > env_results['E1']['n'] and pr['pf'] > PROM_PF else '△'}")

jmd += [
    f"",
    f"**Q2 — Best balance:** Portfolio {best_balance}",
    f"",
    f"**Q3 — PROMOTE criteria:**",
]
for pid in PORT_IDS:
    pr = port_results[pid]
    jmd.append(f"- Port {pid}: Score={pr['score']}/7  **{pr['verdict']}**")

jmd += [
    f"",
    f"**Q4 — Incremental contribution (Port {best_score_port}):**",
]
for eid in PORT_ENVS[best_score_port]:
    ic = incr[best_score_port][eid]
    jmd.append(f"- {eid}: n_attr={ic['n_contrib']}  ΔPF={ic['delta_pf']:+.3f}  "
               f"{'ADDS' if ic['delta_pf'] >= 0 else 'HURTS'}")

jmd += [
    f"",
    f"**Q5 — Any env reduces quality?**",
]
detractors = [(pid, eid, incr[pid][eid]["delta_pf"])
              for pid in PORT_IDS for eid in PORT_ENVS[pid]
              if incr[pid][eid]["delta_pf"] < -0.05]
if detractors:
    for pid, eid, d in detractors:
        jmd.append(f"- ⚠ {eid} in Port {pid}: ΔPF={d:+.3f}")
else:
    jmd.append("- No environment materially reduces portfolio quality.")

jmd += [
    f"",
    f"**Q6 — Diversification (Port {rec_port}):**",
    f"- Active symbols: {len(sym_shares)}/{len(SYMBOLS)}",
    f"- HHI: {hhi:.3f} ({'concentrated' if hhi > 0.25 else 'diversified'})",
    f"",
    f"---",
    f"",
    f"## Final Verdict",
    f"",
    f"**Recommended portfolio: Portfolio {best_score_port}**",
    f"",
    f"Environments: {' + '.join(PORT_ENVS[best_score_port])}",
    f"",
    f"| Metric | Value | Criterion | Pass |",
    f"|--------|-------|-----------|------|",
]
for name, val, crit, passes in criteria:
    jmd.append(f"| {name} | {val:.3f} | {crit} | {'✓' if passes else '✗'} |")

jmd += [
    f"",
    f"**Score: {br['score']}/7**  ",
    f"",
    f"## VERDICT: {br['verdict']}",
]
jmd_path = f"{OUT}/r043_journal.md"
with open(jmd_path, "w") as fh:
    fh.write("\n".join(jmd))
print(f"  → {jmd_path}")

# Append to master journal
try:
    jrow = {
        "research_id":   RESEARCH_ID,
        "strategy_name": f"Portfolio {best_score_port}: {'+'.join(PORT_ENVS[best_score_port])}",
        "symbol":        "ALL",
        "n_trades":      br["n"],
        "profit_factor": round(br["pf"], 4),
        "win_rate":      round(br["wr"], 4),
        "net_profit":    round(br["net"], 2),
        "max_drawdown":  round(br["mdd"], 4),
        "sharpe":        round(br["sharpe"], 4),
        "verdict":       br["verdict"],
        "notes":         (f"5 envs × 4 portfolios. Best={best_score_port} "
                          f"PF={br['pf']:.3f} n={br['n']} Score={br['score']}/7")
    }
    jp = CONFIG["JOURNAL_FILE"]
    jdf = pd.read_csv(jp) if os.path.exists(jp) else pd.DataFrame()
    jdf = pd.concat([jdf, pd.DataFrame([jrow])], ignore_index=True)
    jdf.to_csv(jp, index=False)
    print(f"  → Journal: {jp}")
except Exception as ex:
    print(f"  ⚠  Journal write failed: {ex}")

# =============================================================================
# FINAL CONSOLE SUMMARY
# =============================================================================

print()
print(SEP)
print(f"  R043 COMPLETE — Portfolio Validation (RELVOL fixed · 1H · 5-fold WF)")
print(SEP)
print()
print(f"  Dataset: {len(SYMBOLS)} symbols · {total_bars:,} bars · OOS only")
print()
print(f"  {'ID':>8}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'p50':>7}  {'MC%':>6}  "
      f"{'MDD':>7}  {'LOO-S':>6}  {'LOO-F':>6}  {'Sc':>4}  Verdict")
print("  " + "─" * 100)
for eid in ENV_IDS:
    r = env_results[eid]
    print(f"  {eid:>8}  {r['n']:>5}  {r['wr']*100:5.1f}%  {r['pf']:7.3f}  "
          f"{r['b50']:7.3f}  {r['mc_p']*100:5.0f}%  {r['mdd']:6.1%}  "
          f"{r['sym_floor']:6.3f}  {r['fold_floor']:6.3f}  {r['score']:>2}/7  {r['verdict']}")
    print(f"            {ENV_LABEL[eid]}")
print()
for pid in PORT_IDS:
    pr = port_results[pid]
    print(f"  {'Port '+pid:>8}  {pr['n']:>5}  {pr['wr']*100:5.1f}%  {pr['pf']:7.3f}  "
          f"{pr['b50']:7.3f}  {pr['mc_p']*100:5.0f}%  {pr['mdd']:6.1%}  "
          f"{pr['sym_floor']:6.3f}  {pr['fold_floor']:6.3f}  {pr['score']:>2}/7  {pr['verdict']}")
    print(f"            {' + '.join(PORT_ENVS[pid])}")
print()
print(f"  Q1: Frequency+Quality   → {'YES' if port_results[best_balance]['n'] > env_results['E1']['n'] else 'NO'}")
print(f"  Q2: Best balance        → Portfolio {best_balance}  "
      f"(PF={port_results[best_balance]['pf']:.3f}  n={port_results[best_balance]['n']})")
print(f"  Q3: All PROMOTE crit.   → {'YES: ' + ', '.join(promote_ports) if promote_ports else 'NO — see failing criteria above'}")
print(f"  Q4: Most incremental    → see ΔPF table above")
print(f"  Q5: Quality detractors  → {'NONE' if not detractors else str([(p,e) for p,e,_ in detractors])}")
print(f"  Q6: Diversification     → {len(sym_shares)}/{len(SYMBOLS)} symbols active  HHI={hhi:.3f}")
print()
print(f"  ╔══════════════════════════════════════════════════════════╗")
print(f"  ║  FINAL VERDICT: Portfolio {best_score_port}  →  {br['verdict']:12s}         ║")
print(f"  ║  PF={br['pf']:.3f}  n={br['n']}  Boot={br['b50']:.3f}  Score={br['score']}/7  MDD={br['mdd']:.1%}  ║")
print(f"  ╚══════════════════════════════════════════════════════════╝")
print()
print(f"  Output files:")
for p in [p1, p2, p3, p4, p5, p6, p7, p8, csv1, csv2, jmd_path]:
    if p: print(f"    {p}")
print(SEP)
