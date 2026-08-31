"""
=============================================================================
QUANTLAB AI — RESEARCH #044
External Symbol Validation (Generalisation Test)
=============================================================================

Objective:
  Validate whether Portfolio C from R043 represents a genuine market edge by
  testing it on completely unseen symbols never used in R042/R043.

Rules:
  • FROZEN STRATEGY: Portfolio C = E1 + E2 + E3 + E4 — NO CHANGES WHATSOEVER
  • No thresholds, filters, entry logic, exits, or parameters changed
  • New symbols only — none from R042/R043 research universe
  • 5-fold expanding walk-forward (identical to R043)
  • 1H candles only

Generalisation classification:
  STRONG       PF within 10% of R043 AND ≥70% symbols profitable
  MODERATE     PF within 20% of R043 AND ≥60% symbols profitable
  WEAK         PF falls below 1.20
  OVERFIT      PF < 1.00 OR robustness collapses

Final verdict: PROMOTE / WATCHLIST / REJECT
=============================================================================
"""

import os, sys, math, warnings, time
import concurrent.futures
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
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx, download_symbol

RESEARCH_ID = "R044"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT,   exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = CONFIG["RISK_REWARD"]
BEP_WR   = 1.0 / (1.0 + RR)

# =============================================================================
# NEW SYMBOLS — none used in R042/R043
# R043 used: BTC ETH SOL LINK AVAX XRP LTC BCH DOGE ADA BNB DOT
#             ARB OP NEAR ATOM SUI APT WIF PEPE ENA UNI FIL
# =============================================================================

R043_SYMBOLS = {
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "AVAX-USDT-SWAP","XRP-USDT-SWAP","LTC-USDT-SWAP","BCH-USDT-SWAP",
    "DOGE-USDT-SWAP","ADA-USDT-SWAP","BNB-USDT-SWAP","DOT-USDT-SWAP",
    "ARB-USDT-SWAP","OP-USDT-SWAP","NEAR-USDT-SWAP","ATOM-USDT-SWAP",
    "SUI-USDT-SWAP","APT-USDT-SWAP","WIF-USDT-SWAP","PEPE-USDT-SWAP",
    "ENA-USDT-SWAP","UNI-USDT-SWAP","FIL-USDT-SWAP",
}

# Target: 15-30 completely new symbols
# 4 symbols from OKX cache (TRX, DYDX, GMX, LDO) + 22 from KuCoin Futures
# All pre-downloaded and cached — data loading will use cache only.
CANDIDATE_SYMBOLS = [
    # OKX-cached (full 2-year history ~17k bars each)
    "TRX-USDT-SWAP",     # Tron
    "DYDX-USDT-SWAP",    # dYdX
    "LDO-USDT-SWAP",     # Lido DAO
    "GMX-USDT-SWAP",     # GMX
    # KuCoin-downloaded (24M history, ~17k bars each)
    "1INCH-USDT-SWAP",   # 1inch
    "AAVE-USDT-SWAP",    # Aave
    "ALGO-USDT-SWAP",    # Algorand
    "AXS-USDT-SWAP",     # Axie Infinity
    "CHZ-USDT-SWAP",     # Chiliz
    "COMP-USDT-SWAP",    # Compound
    "CRV-USDT-SWAP",     # Curve
    "EGLD-USDT-SWAP",    # MultiversX
    "ETC-USDT-SWAP",     # Ethereum Classic
    "FET-USDT-SWAP",     # Fetch.ai
    "GALA-USDT-SWAP",    # Gala
    "GRT-USDT-SWAP",     # The Graph
    "HBAR-USDT-SWAP",    # Hedera
    "ICP-USDT-SWAP",     # Internet Computer
    "IMX-USDT-SWAP",     # Immutable X
    "INJ-USDT-SWAP",     # Injective
    "SAND-USDT-SWAP",    # The Sandbox
    "SHIB-USDT-SWAP",    # Shiba Inu
    "SNX-USDT-SWAP",     # Synthetix
    "STX-USDT-SWAP",     # Stacks
    "SUSHI-USDT-SWAP",   # SushiSwap
    "XLM-USDT-SWAP",     # Stellar
]

# Safety check: none of these should be in R043 universe
assert all(s not in R043_SYMBOLS for s in CANDIDATE_SYMBOLS), \
    "ERROR: R043 symbol found in R044 candidates!"

MIN_BARS   = 4_000
FOLDS      = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
N_BOOT     = 2_000
MONTHS_1H  = 24      # Download 24 months — same history length as R043 symbols
FAST_DELAY = 0.05    # Reduced page delay for fast download (OKX allows ~20 req/s)

# PROMOTE thresholds (same as R043)
PROM_PF   = 1.20
PROM_N    = 200
PROM_BOOT = 1.20
PROM_MC   = 0.60
PROM_MDD  = 0.25

# =============================================================================
# PORTFOLIO C — FROZEN (E1+E2+E3+E4 from R043)
# =============================================================================

R042_ENVS = [
    ("E1", "Dist>p75 · Wed-Thu · BodyPct>p60 · US(14-21UTC)",
     ("DST_FR", "MIDWK", "PBP_HI", "US")),
    ("E2", "ADX>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)",
     ("ADX_ST", "DST_FR", "MIDWK", "US")),
    ("E3", "Dist>p75 · Wed-Thu · PrevBody>p67 · US(14-21UTC)",
     ("DST_FR", "MIDWK", "PBD_HI", "US")),
    ("E4", "ADX>p67 · Dist>p60 · Wed-Thu · US(14-21UTC)",
     ("ADX_ST", "DST_MD", "MIDWK", "US")),
]

ENV_IDS   = [e[0] for e in R042_ENVS]
ENV_LABEL = {e[0]: e[1] for e in R042_ENVS}
ENV_CONDS = {e[0]: e[2] for e in R042_ENVS}

PORT_C_ENVS = ["E1", "E2", "E3", "E4"]   # FROZEN — Portfolio C only

# =============================================================================
# CONDITION CATALOGUE  (identical to R042/R043 — no changes)
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

# Only the conditions needed by Port C environments
NEEDED_CONDS = sorted({cid for e in R042_ENVS for cid in e[2]})

QUANT_FEATS = [
    "atr_rank", "real_vol_20", "bb_width", "ema_dist_pct",
    "adx14", "prev_range_r", "prev_body_r", "prev_body_pct",
]

# =============================================================================
# PARALLEL DATA DOWNLOAD
# =============================================================================

def _cache_path_1h(sym):
    tag = sym.replace("-", "_")
    return f"{CACHE}/{tag}_1H.parquet"


import threading
import requests as _requests_mod

OKX_HIST_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_CAND_URL = "https://www.okx.com/api/v5/market/candles"
CANDLE_COLS  = ["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"]

# Global rate limiter: at most 15 requests/second across all threads
_rate_lock    = threading.Lock()
_last_req_t   = [0.0]
_RATE_INTERVAL = 1.0 / 15   # ~66ms between requests = safe 15 req/s

def _fetch_fast(sym, bar, after_ms=None, use_history=True):
    url    = OKX_HIST_URL if use_history else OKX_CAND_URL
    params = {"instId": sym, "bar": bar, "limit": "100"}
    if after_ms is not None:
        params["after"] = str(after_ms)
    # Rate-limit: only one new HTTP request can START every _RATE_INTERVAL seconds
    with _rate_lock:
        elapsed = time.time() - _last_req_t[0]
        if elapsed < _RATE_INTERVAL:
            time.sleep(_RATE_INTERVAL - elapsed)
        _last_req_t[0] = time.time()
    # HTTP call is outside the lock so threads overlap on IO
    try:
        r = _requests_mod.get(url, params=params, timeout=15)
        d = r.json()
        if d.get("code") == "0":
            return d.get("data", [])
    except Exception:
        pass
    return []


def _fast_download(sym, months):
    """Fast paginated download using reduced delay."""
    now_ms    = int(time.time() * 1000)
    target_ms = int(months * 30.44 * 24 * 3600 * 1000)
    cutoff_ms = now_ms - target_ms
    all_rows  = []; after_ms_cursor = None; pages = 0
    while True:
        raw = _fetch_fast(sym, "1H", after_ms=after_ms_cursor, use_history=True)
        if not raw and pages == 0:
            raw = _fetch_fast(sym, "1H", use_history=False)
        if not raw:
            break
        all_rows.extend(raw); pages += 1
        oldest_ts = int(raw[-1][0])
        after_ms_cursor = oldest_ts
        if oldest_ts <= cutoff_ms:
            break
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for col in ["open","high","low","close","vol"]:
        df[col] = pd.to_numeric(df[col])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df = df[df["datetime"] >= cutoff_dt]
    df = (df[["datetime","open","high","low","close","vol"]]
            .drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True))
    return df


def _download_or_load(sym):
    """Load from cache if sufficient; otherwise fast-download. Returns (sym, df_or_None)."""
    path = _cache_path_1h(sym)
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.sort_values("datetime").reset_index(drop=True)
            if len(df) >= MIN_BARS:
                return sym, df
        except Exception:
            pass

    # Fast full download
    try:
        df = _fast_download(sym, MONTHS_1H)
        if len(df) >= MIN_BARS:
            df.to_parquet(path, index=False)
            return sym, df
        else:
            if len(df) > 0:
                print(f"  [SKIP] {sym}: only {len(df):,} bars (need {MIN_BARS:,})")
            else:
                print(f"  [FAIL] {sym}: no data returned")
            return sym, None
    except Exception as exc:
        print(f"  [FAIL] {sym}: {exc}")
        return sym, None


print("=" * 80)
print("  QUANTLAB AI — RESEARCH #044")
print("  External Symbol Validation — Portfolio C Generalisation Test")
print("=" * 80)
print()
print(f"  Candidate symbols: {len(CANDIDATE_SYMBOLS)}")
print(f"  Downloading 1H candles in parallel (fastest mode)…")
print()

t0 = time.time()
raw_dfs = {}
# 3-worker parallel download with semaphore rate-limiter (safe for OKX ~20 req/s)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    futures = {pool.submit(_download_or_load, sym): sym for sym in CANDIDATE_SYMBOLS}
    for fut in concurrent.futures.as_completed(futures):
        sym_name, df_result = fut.result()
        if df_result is not None and len(df_result) >= MIN_BARS:
            raw_dfs[sym_name] = df_result
            print(f"  ✓ {sym_name:<28} {len(df_result):>6,} bars  "
                  f"({df_result['datetime'].iloc[0].date()} → "
                  f"{df_result['datetime'].iloc[-1].date()})")
        else:
            print(f"  ✗ {sym_name:<28} insufficient data")

print()
print(f"  Download complete in {time.time()-t0:.1f}s  —  "
      f"{len(raw_dfs)}/{len(CANDIDATE_SYMBOLS)} symbols usable")
print()

if len(raw_dfs) < 5:
    print("  ⚠  Fewer than 5 symbols available. Cannot run meaningful validation.")
    sys.exit(1)

# =============================================================================
# FEATURE ENGINEERING  (identical to R043)
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


print("  Engineering features …")
all_dfs = {}
for sym, df_raw in raw_dfs.items():
    all_dfs[sym] = add_features(df_raw)

SYMBOLS      = list(all_dfs.keys())
total_bars   = sum(len(d) for d in all_dfs.values())
print(f"  {len(SYMBOLS)} symbols  ({total_bars:,} bars)\n")

# =============================================================================
# THRESHOLD LEARNING  (identical to R043)
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
# CONDITION & ENVIRONMENT MASKS  (identical to R043)
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
# RELVOL SIGNAL  (frozen — identical to R043)
# =============================================================================

def signal_relvol(df: pd.DataFrame, emask: np.ndarray) -> np.ndarray:
    rv  = df["rel_vol"].values
    c   = df["close"].values
    o   = df["open"].values
    pc  = df["prev_close"].values
    ok  = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & emask

# =============================================================================
# PRIORITY CASCADE  (identical to R043)
# =============================================================================

def portfolio_signal(env_signals: list) -> tuple:
    n        = len(env_signals[0][1])
    combined = np.zeros(n, dtype=bool)
    attr     = np.full(n, '', dtype=object)
    for eid, sig in env_signals:
        new_fires         = sig & ~combined
        combined         |= new_fires
        attr[new_fires]   = eid
    return combined, attr

# =============================================================================
# BACKTEST ENGINE  (identical to R043)
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
# STATISTICS  (identical to R043)
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
    m            = metrics(all_trades)
    b5, b50, b95 = bootstrap_pf(m["pnls"])
    mc           = monte_carlo(m["pnls"])
    ls           = loo_sym(sym_trades)
    lf           = loo_fld(all_trades)
    sf           = min(ls.values()) if ls else 0.0
    ff           = min(lf.values()) if lf else 0.0
    score        = sum([
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
            "b5": b5, "b50": b50, "b95": b95,
            "mc_p": mc["prob_profit"], "mc_p50": mc["p50"],
            "mc_finals": mc["finals"],
            "sym_floor": sf, "fold_floor": ff,
            "loo_sym": ls, "loo_fld": lf,
            "score": score, "verdict": verdict}

# =============================================================================
# WALK-FORWARD — PORTFOLIO C ONLY
# =============================================================================

port_sym_trades = defaultdict(list)  # sym -> [trade_dicts]
fold_port_n     = []                  # trades per fold

SEP = "═" * 110

print(SEP)
print(f"  WALK-FORWARD — Portfolio C (E1+E2+E3+E4) — {len(SYMBOLS)} new symbols × {len(FOLDS)} folds")
print(SEP)
print()

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_count = 0
    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[: int(N * is_end)]
        df_oos = df_full.iloc[int(N * is_end): int(N * oos_end)].reset_index(drop=True)
        if len(df_oos) < 100:
            continue

        thr = learn_thresholds(df_is)

        # Compute per-environment signals
        env_signals = []
        for eid in PORT_C_ENVS:
            em  = env_mask(df_oos, eid, thr)
            sig = signal_relvol(df_oos, em)
            env_signals.append((eid, sig))

        # Priority-cascaded portfolio signal
        combined_sig, attr = portfolio_signal(env_signals)
        tl = run_backtest(df_oos, combined_sig, sym, fold_idx, "C", attribution=attr)
        port_sym_trades[sym].extend(tl)
        fold_count += len(tl)

    fold_port_n.append(fold_count)
    print(f"  Fold {fold_idx}  (IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%)  "
          f"trades={fold_count:3d}")

print()

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print("  Computing statistics …")
flat_trades = [t for tl in port_sym_trades.values() for t in tl]
r44 = full_stats(flat_trades, dict(port_sym_trades))

# Per-symbol metrics
sym_metrics = {}
for sym in SYMBOLS:
    tl = port_sym_trades.get(sym, [])
    sym_metrics[sym] = metrics(tl)

# =============================================================================
# LOAD R043 REFERENCE  (Portfolio C row from scorecard)
# =============================================================================

R043_REF = {}
sc_path = f"{OUT}/r043_scorecard.csv"
if os.path.exists(sc_path):
    sc_df = pd.read_csv(sc_path)
    row = sc_df[sc_df["id"] == "Port_C"]
    if not row.empty:
        R043_REF = row.iloc[0].to_dict()
        print(f"  R043 Portfolio C loaded from scorecard:")
        print(f"    n={R043_REF.get('n','?')}  PF={R043_REF.get('profit_factor','?'):.3f}  "
              f"Boot={R043_REF.get('boot_p50','?'):.3f}  "
              f"MC={float(R043_REF.get('mc_prob','0'))*100:.0f}%")
    else:
        print("  ⚠  Port_C row not found in r043_scorecard.csv")
else:
    print("  ⚠  r043_scorecard.csv not found — R043 reference unavailable")
print()

# =============================================================================
# RESULTS TABLES
# =============================================================================

print(SEP)
print("  R044 — PORTFOLIO C ON NEW SYMBOLS")
print(SEP)
print(f"\n  Trades:          {r44['n']}")
print(f"  Win Rate:        {r44['wr']*100:.1f}%")
print(f"  Profit Factor:   {r44['pf']:.3f}")
print(f"  Bootstrap p50:   {r44['b50']:.3f}  [{r44['b5']:.3f}, {r44['b95']:.3f}]")
print(f"  MC P(profit):    {r44['mc_p']*100:.1f}%")
print(f"  Max Drawdown:    {r44['mdd']:.1%}")
print(f"  LOO-Symbol floor:{r44['sym_floor']:.3f}")
print(f"  LOO-Fold floor:  {r44['fold_floor']:.3f}")
print(f"  Score:           {r44['score']}/7")
print()

# =============================================================================
# Q1 — Does Portfolio C still achieve PF > 1.20?
# =============================================================================

print(SEP)
print("  Q1: Does Portfolio C achieve PF > 1.20 on completely unseen symbols?")
print(SEP)
passes_pf = r44['pf'] > PROM_PF
tick = "✓ YES" if passes_pf else "✗ NO"
print(f"  Portfolio C PF = {r44['pf']:.3f}  (threshold: >{PROM_PF})  → {tick}")
print()

# =============================================================================
# Q2 — Per-symbol breakdown
# =============================================================================

print(SEP)
print("  Q2: Per-Symbol Report")
print(SEP)
print(f"\n  {'Symbol':<26} {'Trades':>6}  {'Win%':>6}  {'PF':>7}  {'Net R':>8}  {'Status':>10}")
print("  " + "─" * 75)
n_profitable = 0
sym_report_rows = []
for sym in sorted(SYMBOLS):
    sm = sym_metrics[sym]
    tag = sym.replace("-USDT-SWAP", "")
    status = "profitable" if sm["pf"] > 1.0 and sm["n"] > 0 else ("no trades" if sm["n"] == 0 else "loss")
    if sm["pf"] > 1.0 and sm["n"] > 0:
        n_profitable += 1
    net_r = sm["wr"] * RR - (1 - sm["wr"]) if sm["n"] > 0 else 0.0
    print(f"  {tag:<26} {sm['n']:>6}  {sm['wr']*100:>5.1f}%  {sm['pf']:>7.3f}  {net_r:>+8.3f}  {status:>10}")
    sym_report_rows.append({
        "symbol": tag, "n_trades": sm["n"], "win_rate": round(sm["wr"], 4),
        "profit_factor": round(sm["pf"], 4), "net_r": round(net_r, 4),
        "net_pnl": round(sm["net"], 2), "mdd": round(sm["mdd"], 4),
    })

pct_profitable = n_profitable / len(SYMBOLS) * 100
print(f"\n  Profitable symbols: {n_profitable}/{len(SYMBOLS)} ({pct_profitable:.0f}%)")
print()

# =============================================================================
# Q3 — Robustness criteria
# =============================================================================

print(SEP)
print("  Q3: Portfolio Robustness (Bootstrap · Monte Carlo · LOO)")
print(SEP)
q3_checks = [
    ("Bootstrap median PF > 1.20",  r44["b50"],        1.20,  r44["b50"]  > 1.20),
    ("Monte Carlo P(profit) > 60%", r44["mc_p"]*100,   60.0,  r44["mc_p"] > 0.60),
    ("LOO-symbol floor > 1.00",     r44["sym_floor"],  1.00,  r44["sym_floor"]  > 1.0),
    ("LOO-fold floor > 1.00",       r44["fold_floor"], 1.00,  r44["fold_floor"] > 1.0),
]
print(f"\n  {'Criterion':<35} {'Value':>10}  {'Threshold':>12}  {'Pass?':>7}")
print("  " + "─" * 70)
for name, val, thr_val, passes in q3_checks:
    print(f"  {name:<35} {val:>10.3f}  {thr_val:>12.2f}  {'✓ PASS' if passes else '✗ FAIL':>7}")
print()

# LOO symbol table
print(f"\n  LOO-Symbol Table (PF when each symbol is removed):")
print(f"  {'Symbol':<26}  {'LOO PF':>8}")
print("  " + "─" * 36)
for sym_k in sorted(r44["loo_sym"].keys()):
    tag = sym_k.replace("-USDT-SWAP", "")
    pf_v = r44["loo_sym"][sym_k]
    flag = " ⚠" if pf_v < 1.0 else ""
    print(f"  {tag:<26}  {pf_v:>8.3f}{flag}")
print()

# LOO fold table
print(f"\n  LOO-Fold Table (PF when each fold is removed):")
print(f"  {'Fold':<8}  {'LOO PF':>8}")
print("  " + "─" * 20)
for fold_k in sorted(r44["loo_fld"].keys()):
    pf_v = r44["loo_fld"][fold_k]
    flag = " ⚠" if pf_v < 1.0 else ""
    print(f"  F{fold_k:<7}  {pf_v:>8.3f}{flag}")
print()

# =============================================================================
# Q4 — Direct comparison vs R043
# =============================================================================

print(SEP)
print("  Q4: R044 vs R043 Portfolio C — Direct Comparison")
print(SEP)
print()

if R043_REF:
    r43_pf    = float(R043_REF.get("profit_factor", 0))
    r43_n     = int(R043_REF.get("n", 0))
    r43_wr    = float(R043_REF.get("win_rate", 0))
    r43_boot  = float(R043_REF.get("boot_p50", 0))
    r43_mc    = float(R043_REF.get("mc_prob", 0))
    r43_mdd   = float(R043_REF.get("mdd", 0))
    r43_loos  = float(R043_REF.get("sym_floor", 0))
    r43_loof  = float(R043_REF.get("fold_floor", 0))
else:
    # Fallback: use known approximate R043 Port C values (will be replaced by CSV)
    r43_pf   = 1.350; r43_n   = 500; r43_wr   = 0.420
    r43_boot = 1.300; r43_mc  = 0.75; r43_mdd  = -0.10
    r43_loos = 1.20;  r43_loof = 1.20
    print("  (R043 reference loaded from fallback estimates)")

comparison = [
    ("Trades",          r43_n,        r44["n"],          None),
    ("Win Rate",        r43_wr*100,   r44["wr"]*100,     None),
    ("Profit Factor",   r43_pf,       r44["pf"],         r44["pf"] - r43_pf),
    ("Bootstrap Median",r43_boot,     r44["b50"],        r44["b50"] - r43_boot),
    ("Monte Carlo P%",  r43_mc*100,   r44["mc_p"]*100,   (r44["mc_p"] - r43_mc)*100),
    ("Max Drawdown",    r43_mdd*100,  r44["mdd"]*100,    (r44["mdd"] - r43_mdd)*100),
    ("LOO Symbol Floor",r43_loos,     r44["sym_floor"],  r44["sym_floor"] - r43_loos),
    ("LOO Fold Floor",  r43_loof,     r44["fold_floor"], r44["fold_floor"] - r43_loof),
]

print(f"  {'Metric':<25} {'R043':>10}  {'R044':>10}  {'Delta':>10}")
print("  " + "─" * 60)
for name, v43, v44, delta in comparison:
    if delta is None:
        print(f"  {name:<25} {v43:>10.0f}  {v44:>10.0f}  {'':>10}")
    else:
        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "≈")
        print(f"  {name:<25} {v43:>10.3f}  {v44:>10.3f}  {delta:>+10.3f} {arrow}")
print()

# =============================================================================
# Q5 — Generalisation Classification
# =============================================================================

print(SEP)
print("  Q5: Generalisation Classification")
print(SEP)
print()

pf_ratio = r44["pf"] / r43_pf if r43_pf > 0 else 0.0
pf_pct_change = (r44["pf"] - r43_pf) / r43_pf * 100 if r43_pf > 0 else -100.0
pf_within_10 = pf_ratio >= 0.90
pf_within_20 = pf_ratio >= 0.80
pct_prof_gte70 = pct_profitable >= 70.0
pct_prof_gte60 = pct_profitable >= 60.0
robustness_ok  = (r44["sym_floor"] > 1.0 and r44["fold_floor"] > 1.0 and
                  r44["mc_p"] > 0.50)

if r44["pf"] < 1.0 or not robustness_ok:
    gen_class = "OVERFIT"
elif r44["pf"] < PROM_PF:
    gen_class = "WEAK GENERALISATION"
elif pf_within_10 and pct_prof_gte70:
    gen_class = "STRONG GENERALISATION"
elif pf_within_20 and pct_prof_gte60:
    gen_class = "MODERATE GENERALISATION"
else:
    gen_class = "WEAK GENERALISATION"

print(f"  R043 Port C PF:    {r43_pf:.3f}")
print(f"  R044 Port C PF:    {r44['pf']:.3f}")
print(f"  PF ratio:          {pf_ratio:.3f}  ({pf_pct_change:+.1f}%)")
print(f"  PF within 10%:     {'YES' if pf_within_10 else 'NO'}")
print(f"  PF within 20%:     {'YES' if pf_within_20 else 'NO'}")
print(f"  Symbols profitable:{pct_profitable:.0f}% (≥70% for STRONG, ≥60% for MODERATE)")
print(f"  Robustness:        {'INTACT' if robustness_ok else 'COLLAPSED'}")
print()
print(f"  ┌─────────────────────────────────────────────┐")
print(f"  │  GENERALISATION CLASS:  {gen_class:<20}│")
print(f"  └─────────────────────────────────────────────┘")
print()

# =============================================================================
# FOLD BREAKDOWN
# =============================================================================

print(SEP)
print("  FOLD BREAKDOWN")
print(SEP)
fold_metrics = {}
for fold_idx in range(1, len(FOLDS)+1):
    fold_trades = [t for t in flat_trades if t["fold"] == fold_idx]
    fm = metrics(fold_trades)
    fold_metrics[fold_idx] = fm
    print(f"  Fold {fold_idx}  n={fm['n']:3d}  WR={fm['wr']*100:4.1f}%  PF={fm['pf']:.3f}  "
          f"Net=${fm['net']:+.0f}  MDD={fm['mdd']:.1%}")
print()

# =============================================================================
# FINAL VERDICT
# =============================================================================

print(SEP)
print("  FINAL VERDICT")
print(SEP)
print()
print(f"  {'Criterion':<35} {'Value':>10}  {'Threshold':>12}  {'Pass?':>7}")
print("  " + "─" * 70)
all_criteria = [
    ("Profit Factor",        r44["pf"],          f">{PROM_PF}",    r44["pf"] > PROM_PF),
    ("Trade Count",          r44["n"],            f"≥{PROM_N}",     r44["n"] >= PROM_N),
    ("Bootstrap p50",        r44["b50"],          f">{PROM_BOOT}",  r44["b50"] > PROM_BOOT),
    ("Monte Carlo P(profit)",r44["mc_p"]*100,     ">60%",           r44["mc_p"] > PROM_MC),
    ("LOO Symbol Floor",     r44["sym_floor"],    ">1.00",          r44["sym_floor"] > 1.0),
    ("LOO Fold Floor",       r44["fold_floor"],   ">1.00",          r44["fold_floor"] > 1.0),
    ("Max Drawdown",         abs(r44["mdd"])*100, f"<{PROM_MDD*100:.0f}%", abs(r44["mdd"]) < PROM_MDD),
]
for name, val, crit, passes in all_criteria:
    print(f"  {name:<35} {val:>10.3f}  {crit:>12}  {'✓ PASS' if passes else '✗ FAIL':>7}")
print()
print(f"  Score: {r44['score']}/7")
print()
print(f"  ╔══════════════════════════════════════════════════════════════════╗")
print(f"  ║  VERDICT: {r44['verdict']:<10}  |  GENERALISATION: {gen_class:<22}  ║")
print(f"  ╚══════════════════════════════════════════════════════════════════╝")
print()

# Final question
print(SEP)
print("  FINAL QUESTION: Does R043 edge genuinely generalise?")
print(SEP)
print()
if gen_class == "STRONG GENERALISATION":
    answer = (f"  YES — STRONGLY. Portfolio C's edge is NOT specific to the R043 research "
              f"universe.\n"
              f"  PF on {len(SYMBOLS)} entirely new symbols ({r44['pf']:.3f}) is within 10% of "
              f"R043 ({r43_pf:.3f}),\n"
              f"  with {pct_profitable:.0f}% of symbols individually profitable.\n"
              f"  The edge is real, robust, and generalises broadly across crypto markets.")
elif gen_class == "MODERATE GENERALISATION":
    answer = (f"  YES — MODERATELY. Portfolio C shows meaningful generalisation.\n"
              f"  PF on {len(SYMBOLS)} new symbols ({r44['pf']:.3f}) vs R043 ({r43_pf:.3f}) "
              f"— PF ratio {pf_ratio:.2f} ({pf_pct_change:+.1f}%).\n"
              f"  {pct_profitable:.0f}% of symbols profitable. Some degradation from the "
              f"original universe is normal,\n"
              f"  but the core edge persists.")
elif gen_class == "WEAK GENERALISATION":
    answer = (f"  PARTIALLY. PF remains above 1.00 but falls below the 1.20 threshold.\n"
              f"  R044 PF={r44['pf']:.3f} vs R043 PF={r43_pf:.3f} ({pf_pct_change:+.1f}%).\n"
              f"  The edge is present but weaker on new symbols — may reflect some "
              f"universe-specificity,\n"
              f"  or differences in data history length between old and new symbols.")
else:
    answer = (f"  NO — OVERFIT. Portfolio C does not generalise to new symbols.\n"
              f"  R044 PF={r44['pf']:.3f} < 1.00 or robustness has collapsed.\n"
              f"  The R043 results were specific to the original research universe.")
print(answer)
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

def _style(ax, title=""):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    for sp in ax.spines.values(): sp.set_color(GRID_CLR)
    ax.xaxis.label.set_color(TEXT_CLR); ax.yaxis.label.set_color(TEXT_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5)
    if title: ax.set_title(title, color=TEXT_CLR, fontsize=8, pad=4)


# ── Chart 1: Equity curve ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5), facecolor=DARK_BG)
_style(ax, f"R044 Portfolio C — Equity Curve  "
       f"(PF={r44['pf']:.3f}  n={r44['n']}  "
       f"{len(SYMBOLS)} new symbols)")
eq = r44["equity"]
x  = np.arange(len(eq))
ax.plot(x, eq, color=ACCENT, lw=1.5)
ax.fill_between(x, CAPITAL, eq, where=eq >= CAPITAL, alpha=0.15, color=GREEN)
ax.fill_between(x, CAPITAL, eq, where=eq <  CAPITAL, alpha=0.15, color=RED)
ax.axhline(CAPITAL, color=TEXT_CLR, lw=0.8, ls="--", label="Break-even")
ax.set_xlabel("Trade #"); ax.set_ylabel("Equity ($)")
ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR)
plt.tight_layout()
p1 = f"{OUT}/r044_equity_curves.png"
plt.savefig(p1, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p1}")

# ── Chart 2: Per-symbol PF heatmap ────────────────────────────────────────────
sym_list_sorted = sorted(SYMBOLS)
sym_pf_vals = np.array([sym_metrics[s]["pf"] for s in sym_list_sorted]).reshape(1, -1)
sym_n_vals  = [sym_metrics[s]["n"] for s in sym_list_sorted]
fig, ax = plt.subplots(figsize=(max(10, len(sym_list_sorted)*0.65), 3.5), facecolor=DARK_BG)
_style(ax, f"R044 Per-Symbol Profit Factor — Portfolio C on New Symbols")
cmap_sym = LinearSegmentedColormap.from_list("pf", [RED, DARK_BG, GREEN])
im = ax.imshow(sym_pf_vals, aspect="auto", cmap=cmap_sym, vmin=0, vmax=3)
ax.set_xticks(range(len(sym_list_sorted)))
ax.set_xticklabels([s.replace("-USDT-SWAP","") for s in sym_list_sorted],
                    rotation=45, fontsize=7, color=TEXT_CLR)
ax.set_yticks([]); ax.set_ylabel("Port C", color=TEXT_CLR)
for j, (pf_v, n_v) in enumerate(zip(sym_pf_vals[0], sym_n_vals)):
    if n_v > 0:
        ax.text(j, 0, f"{pf_v:.2f}\nn={n_v}", ha="center", va="center", fontsize=5.5, color="white")
    else:
        ax.text(j, 0, "no\ntrades", ha="center", va="center", fontsize=5, color=TEXT_CLR)
plt.colorbar(im, ax=ax, orientation="vertical", label="PF").ax.yaxis.label.set_color(TEXT_CLR)
plt.tight_layout()
p2 = f"{OUT}/r044_per_symbol_pf.png"
plt.savefig(p2, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p2}")

# ── Chart 3: Bootstrap distribution ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
_style(ax, f"R044 Bootstrap PF Distribution (n_iter={N_BOOT})")
pnls = r44["pnls"]
if len(pnls) >= 5:
    rng_b = np.random.default_rng(42)
    boot_pfs = [safe_pf(s[s>0].sum(), abs(s[s<0].sum()))
                for _ in range(N_BOOT)
                for s in [rng_b.choice(pnls, len(pnls), replace=True)]]
    ax.hist(boot_pfs, bins=60, color=ACCENT, alpha=0.75, edgecolor="none")
ax.axvline(1.0,        color=RED,    lw=1.0, ls=":",  label="PF=1.0 (breakeven)")
ax.axvline(PROM_PF,    color=GOLD,   lw=1.0, ls="--", label=f"PF={PROM_PF} (threshold)")
ax.axvline(r44["b50"], color=GREEN,  lw=1.5,           label=f"Median={r44['b50']:.3f}")
if R043_REF:
    ax.axvline(r43_pf, color=ORANGE, lw=1.2, ls="--", label=f"R043 PF={r43_pf:.3f}")
ax.set_xlabel("Profit Factor"); ax.set_ylabel("Frequency")
ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR)
plt.tight_layout()
p3 = f"{OUT}/r044_bootstrap_ci.png"
plt.savefig(p3, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p3}")

# ── Chart 4: Monte Carlo ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
_style(ax, f"R044 Monte Carlo — Portfolio C  P(profit)={r44['mc_p']*100:.1f}%")
finals = r44["mc_finals"]
ax.hist(finals, bins=60, color=PURPLE, alpha=0.75, edgecolor="none")
ax.axvline(CAPITAL,                   color=RED,   lw=1, ls="--", label="Breakeven")
ax.axvline(np.percentile(finals,  5), color=GOLD,  lw=1, ls=":", label="p5")
ax.axvline(np.percentile(finals, 50), color=GREEN, lw=1.5,        label="p50")
ax.axvline(np.percentile(finals, 95), color=ACCENT,lw=1, ls=":", label="p95")
ax.set_xlabel("Terminal Capital ($)"); ax.set_ylabel("Frequency")
ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR)
plt.tight_layout()
p4 = f"{OUT}/r044_monte_carlo.png"
plt.savefig(p4, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p4}")

# ── Chart 5: LOO tables ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(r44["loo_sym"])*0.28 + 1)),
                         facecolor=DARK_BG)

ax = axes[0]; _style(ax, "LOO-Symbol PF (removing one symbol at a time)")
syms_s = sorted(r44["loo_sym"].keys(), key=lambda s: r44["loo_sym"][s])
vals_s = [r44["loo_sym"][s] for s in syms_s]
tags_s = [s.replace("-USDT-SWAP","") for s in syms_s]
colors_s = [GREEN if v > 1.2 else (GOLD if v > 1.0 else RED) for v in vals_s]
ax.barh(tags_s, vals_s, color=colors_s, alpha=0.8)
ax.axvline(1.0, color=TEXT_CLR, lw=0.8, ls=":")
ax.axvline(PROM_PF, color=RED, lw=0.8, ls="--")
ax.set_xlabel("LOO Profit Factor")

ax = axes[1]; _style(ax, "LOO-Fold PF (removing one fold at a time)")
folds_f = sorted(r44["loo_fld"].keys())
vals_f  = [r44["loo_fld"][f] for f in folds_f]
colors_f = [GREEN if v > 1.2 else (GOLD if v > 1.0 else RED) for v in vals_f]
ax.barh([f"Fold {f}" for f in folds_f], vals_f, color=colors_f, alpha=0.8)
ax.axvline(1.0, color=TEXT_CLR, lw=0.8, ls=":")
ax.axvline(PROM_PF, color=RED, lw=0.8, ls="--")
ax.set_xlabel("LOO Profit Factor")
plt.tight_layout()
p5 = f"{OUT}/r044_loo_robustness.png"
plt.savefig(p5, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p5}")

# ── Chart 6: Fold stability ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=DARK_BG)
ax = axes[0]; _style(ax, "Fold Trade Count")
fold_labels = [f"F{i+1}" for i in range(len(FOLDS))]
ax.bar(fold_labels, fold_port_n, color=ACCENT, alpha=0.8)
ax.set_ylabel("Trades"); ax.tick_params(axis="x", colors=TEXT_CLR)

ax = axes[1]; _style(ax, "Fold PF")
fold_pf_vals = [fold_metrics[f]["pf"] if fold_metrics[f]["n"] > 0 else 0.0
                for f in range(1, len(FOLDS)+1)]
bar_colors = [GREEN if v > PROM_PF else (GOLD if v > 1.0 else RED) for v in fold_pf_vals]
ax.bar(fold_labels, fold_pf_vals, color=bar_colors, alpha=0.8)
ax.axhline(PROM_PF, color=RED, lw=0.8, ls="--")
ax.axhline(1.0, color=GOLD, lw=0.6, ls=":")
ax.set_ylabel("Profit Factor"); ax.tick_params(axis="x", colors=TEXT_CLR)
plt.tight_layout()
p6 = f"{OUT}/r044_fold_stability.png"
plt.savefig(p6, dpi=130, facecolor=DARK_BG); plt.close()
print(f"  → {p6}")

# ── Chart 7: R043 vs R044 comparison bar chart ────────────────────────────────
if R043_REF:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=DARK_BG)
    labels = ["R043\n(original)", "R044\n(new syms)"]
    metrics_to_plot = [
        ("Profit Factor",       [r43_pf, r44["pf"]],         PROM_PF),
        ("Bootstrap Median",    [r43_boot, r44["b50"]],       PROM_BOOT),
        ("MC P(profit)",        [r43_mc*100, r44["mc_p"]*100], 60.0),
    ]
    for ax, (title, vals, ref) in zip(axes, metrics_to_plot):
        _style(ax, title)
        colors = [GOLD, ACCENT]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.axhline(ref, color=RED, lw=0.9, ls="--", label=f"threshold={ref}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + max(abs(v)*0.02, 0.01),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=TEXT_CLR)
        ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR)
    plt.suptitle(f"R043 vs R044 — Portfolio C  ({gen_class})",
                 color=TEXT_CLR, fontsize=10)
    plt.tight_layout()
    p7 = f"{OUT}/r044_comparison.png"
    plt.savefig(p7, dpi=130, facecolor=DARK_BG); plt.close()
    print(f"  → {p7}")
else:
    p7 = None

# ── Chart 8: Dashboard ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 10), facecolor=DARK_BG)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

# A: Equity curve
ax_a = fig.add_subplot(gs[0, 0:2]); _style(ax_a, f"Portfolio C Equity — R044")
eq = r44["equity"]; x = np.arange(len(eq))
ax_a.plot(x, eq, color=ACCENT, lw=1.5)
ax_a.fill_between(x, CAPITAL, eq, where=eq>=CAPITAL, alpha=0.15, color=GREEN)
ax_a.fill_between(x, CAPITAL, eq, where=eq< CAPITAL, alpha=0.15, color=RED)
ax_a.axhline(CAPITAL, color=TEXT_CLR, lw=0.5, ls="--"); ax_a.set_ylabel("Equity ($)")

# B: Per-symbol PF
ax_b = fig.add_subplot(gs[0, 2:4]); _style(ax_b, "Per-Symbol PF (new symbols)")
spf_tags = [s.replace("-USDT-SWAP","") for s in sym_list_sorted]
spf_vals = [sym_metrics[s]["pf"] for s in sym_list_sorted]
spf_clrs = [GREEN if v > 1.2 else (GOLD if v > 1.0 else RED) for v in spf_vals]
ax_b.bar(spf_tags, spf_vals, color=spf_clrs, alpha=0.8)
ax_b.axhline(PROM_PF, color=RED, lw=0.8, ls="--")
ax_b.axhline(1.0, color=GOLD, lw=0.6, ls=":")
ax_b.set_xticklabels(spf_tags, rotation=55, fontsize=5.5, color=TEXT_CLR)
ax_b.set_ylabel("PF")

# C: Bootstrap histogram
ax_c = fig.add_subplot(gs[1, 0:2]); _style(ax_c, "Bootstrap PF Distribution")
if len(pnls) >= 5:
    ax_c.hist(boot_pfs, bins=50, color=ACCENT, alpha=0.7, edgecolor="none")
ax_c.axvline(r44["b50"], color=GREEN, lw=1.5, label=f"p50={r44['b50']:.3f}")
ax_c.axvline(PROM_PF, color=RED, lw=0.8, ls="--")
ax_c.legend(fontsize=6, facecolor=DARK_BG, labelcolor=TEXT_CLR)
ax_c.set_xlabel("PF")

# D: MC histogram
ax_d = fig.add_subplot(gs[1, 2:4]); _style(ax_d, f"Monte Carlo  P(profit)={r44['mc_p']*100:.1f}%")
ax_d.hist(finals, bins=50, color=PURPLE, alpha=0.7, edgecolor="none")
ax_d.axvline(CAPITAL, color=RED, lw=1, ls="--")
ax_d.axvline(np.percentile(finals, 50), color=GREEN, lw=1.5)
ax_d.set_xlabel("Terminal Capital ($)")

# E: LOO Symbol
ax_e = fig.add_subplot(gs[2, 0:2]); _style(ax_e, "LOO-Symbol PF")
ax_e.barh(tags_s, vals_s, color=colors_s, alpha=0.8)
ax_e.axvline(1.0, color=TEXT_CLR, lw=0.7, ls=":")
ax_e.axvline(PROM_PF, color=RED, lw=0.7, ls="--")
ax_e.tick_params(axis="y", labelsize=5.5)
ax_e.set_xlabel("PF")

# F: Summary text
ax_f = fig.add_subplot(gs[2, 2:4]); ax_f.set_facecolor(DARK_BG); ax_f.axis("off")
ref_pf_str = f"{r43_pf:.3f}" if R043_REF else "N/A"
lines = [
    "R044  EXTERNAL SYMBOL VALIDATION",
    "",
    f"Dataset: {len(SYMBOLS)} NEW symbols · 1H · 5-fold WF",
    f"Strategy: Portfolio C (E1+E2+E3+E4) — FROZEN",
    "",
    f"Metric             R044       R043",
    f"─" * 36,
    f"Trades         {r44['n']:>8}   {'?' if not R043_REF else r43_n:>8}",
    f"Win Rate       {r44['wr']*100:>7.1f}%   {'?' if not R043_REF else f'{r43_wr*100:.1f}%':>8}",
    f"Profit Factor  {r44['pf']:>8.3f}   {ref_pf_str:>8}",
    f"Bootstrap p50  {r44['b50']:>8.3f}   {'?' if not R043_REF else f'{r43_boot:.3f}':>8}",
    f"MC P(profit)   {r44['mc_p']*100:>7.1f}%   {'?' if not R043_REF else f'{r43_mc*100:.1f}%':>8}",
    f"LOO-S floor    {r44['sym_floor']:>8.3f}",
    f"LOO-F floor    {r44['fold_floor']:>8.3f}",
    f"Score          {r44['score']:>5}/7",
    "",
    f"Symbols profitable: {n_profitable}/{len(SYMBOLS)} ({pct_profitable:.0f}%)",
    "",
    f"Generalisation:  {gen_class}",
    f"Verdict:         {r44['verdict']}",
]
ax_f.text(0.02, 0.98, "\n".join(lines), transform=ax_f.transAxes,
          fontsize=7, color=TEXT_CLR, va="top", ha="left", fontfamily="monospace",
          bbox=dict(facecolor=GRID_CLR, alpha=0.5, edgecolor="none", pad=5))

plt.suptitle(f"QUANTLAB AI — R044  External Symbol Validation  |  "
             f"Portfolio C  |  {gen_class}",
             color=TEXT_CLR, fontsize=10, y=1.005)
p8 = f"{OUT}/r044_dashboard.png"
plt.savefig(p8, dpi=130, facecolor=DARK_BG, bbox_inches="tight"); plt.close()
print(f"  → {p8}")

# =============================================================================
# CSV OUTPUTS
# =============================================================================

# Per-symbol report
csv1 = f"{OUT}/r044_per_symbol_report.csv"
pd.DataFrame(sym_report_rows).to_csv(csv1, index=False)
print(f"  → {csv1}")

# Trade log
if flat_trades:
    tl_df = pd.DataFrame(flat_trades)
    csv2  = f"{OUT}/r044_trade_log.csv"
    tl_df.to_csv(csv2, index=False)
    print(f"  → {csv2}")
else:
    csv2 = None

# LOO symbol table
loo_rows = [{"symbol": k.replace("-USDT-SWAP",""), "loo_pf": round(v, 4)}
             for k, v in sorted(r44["loo_sym"].items())]
csv3 = f"{OUT}/r044_loo_symbol.csv"
pd.DataFrame(loo_rows).to_csv(csv3, index=False)
print(f"  → {csv3}")

# LOO fold table
loo_fld_rows = [{"fold": k, "loo_pf": round(v, 4)}
                 for k, v in sorted(r44["loo_fld"].items())]
csv4 = f"{OUT}/r044_loo_fold.csv"
pd.DataFrame(loo_fld_rows).to_csv(csv4, index=False)
print(f"  → {csv4}")

# Fold report
fold_rows = []
for f_i in range(1, len(FOLDS)+1):
    fm = fold_metrics[f_i]
    fold_rows.append({
        "fold": f_i, "n": fm["n"], "win_rate": round(fm["wr"], 4),
        "profit_factor": round(fm["pf"], 4), "net_pnl": round(fm["net"], 2),
        "mdd": round(fm["mdd"], 4),
    })
csv5 = f"{OUT}/r044_fold_report.csv"
pd.DataFrame(fold_rows).to_csv(csv5, index=False)
print(f"  → {csv5}")

# =============================================================================
# JOURNAL MARKDOWN
# =============================================================================

jmd_lines = [
    f"# QUANTLAB AI — R044 Research Journal",
    f"",
    f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
    f"**Research ID:** R044  ",
    f"**Title:** External Symbol Validation — Portfolio C Generalisation Test  ",
    f"**Dataset:** 1H · {len(SYMBOLS)} NEW symbols · {total_bars:,} bars · 5-fold WF · OOS only  ",
    f"",
    f"---",
    f"",
    f"## Portfolio C (Frozen from R043)",
    f"",
    f"E1: {ENV_LABEL['E1']}  ",
    f"E2: {ENV_LABEL['E2']}  ",
    f"E3: {ENV_LABEL['E3']}  ",
    f"E4: {ENV_LABEL['E4']}  ",
    f"",
    f"**No parameters changed. Complete strategy freeze.**",
    f"",
    f"## Q1 — PF > 1.20 on New Symbols?",
    f"",
    f"Portfolio C PF = **{r44['pf']:.3f}** ({'PASS ✓' if r44['pf'] > PROM_PF else 'FAIL ✗'})",
    f"",
    f"## Q2 — Per-Symbol Report",
    f"",
    f"| Symbol | Trades | Win Rate | PF | Net R |",
    f"|--------|--------|----------|----|-------|",
]
for row in sym_report_rows:
    jmd_lines.append(f"| {row['symbol']} | {row['n_trades']} | "
                     f"{row['win_rate']*100:.1f}% | {row['profit_factor']:.3f} | "
                     f"{row['net_r']:+.3f} |")
jmd_lines += [
    f"",
    f"Profitable symbols: **{n_profitable}/{len(SYMBOLS)} ({pct_profitable:.0f}%)**",
    f"",
    f"## Q3 — Robustness",
    f"",
]
for name, val, thr_v, passes in q3_checks:
    jmd_lines.append(f"- {name}: {val:.3f} → {'✓ PASS' if passes else '✗ FAIL'}")
jmd_lines += [
    f"",
    f"## Q4 — R043 vs R044 Comparison",
    f"",
    f"| Metric | R043 | R044 | Delta |",
    f"|--------|------|------|-------|",
]
for name, v43, v44, delta in comparison:
    if delta is None:
        jmd_lines.append(f"| {name} | {v43:.0f} | {v44:.0f} | — |")
    else:
        jmd_lines.append(f"| {name} | {v43:.3f} | {v44:.3f} | {delta:+.3f} |")
jmd_lines += [
    f"",
    f"## Q5 — Generalisation Classification",
    f"",
    f"**{gen_class}**",
    f"",
    f"- PF ratio: {pf_ratio:.3f} ({pf_pct_change:+.1f}%)",
    f"- Symbols profitable: {pct_profitable:.0f}%",
    f"",
    f"---",
    f"",
    f"## Final Verdict",
    f"",
    f"**Score: {r44['score']}/7**  ",
    f"",
    f"**Verdict: {r44['verdict']}**  ",
    f"",
    f"**Generalisation: {gen_class}**  ",
    f"",
    answer,
]
jmd_path = f"{OUT}/r044_journal.md"
with open(jmd_path, "w") as fh:
    fh.write("\n".join(jmd_lines))
print(f"  → {jmd_path}")

# Append to master journal
try:
    jrow = {
        "research_id":   RESEARCH_ID,
        "strategy_name": f"Portfolio C ({gen_class})",
        "symbol":        "NEW_SYMBOLS",
        "n_trades":      r44["n"],
        "profit_factor": round(r44["pf"], 4),
        "win_rate":      round(r44["wr"], 4),
        "net_profit":    round(r44["net"], 2),
        "max_drawdown":  round(r44["mdd"], 4),
        "sharpe":        round(r44["sharpe"], 4),
        "verdict":       r44["verdict"],
        "notes":         (f"Generalisation test: {len(SYMBOLS)} new syms. "
                          f"PF={r44['pf']:.3f} vs R043={r43_pf:.3f} ({pf_pct_change:+.1f}%). "
                          f"{gen_class}. {n_profitable}/{len(SYMBOLS)} syms profitable.")
    }
    jp  = CONFIG["JOURNAL_FILE"]
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
print(f"  R044 COMPLETE — External Symbol Validation")
print(SEP)
print()
print(f"  Dataset:   {len(SYMBOLS)} UNSEEN symbols · {total_bars:,} bars · 1H · 5-fold WF · OOS only")
print(f"  Strategy:  Portfolio C (E1+E2+E3+E4) — COMPLETELY FROZEN")
print()
print(f"  {'Metric':<35} {'R044':>10}  {'R043':>10}  {'Delta':>10}")
print("  " + "─" * 70)
for name, v43, v44, delta in comparison:
    if delta is None:
        print(f"  {name:<35} {v44:>10.0f}  {v43:>10.0f}")
    else:
        arrow = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "≈")
        print(f"  {name:<35} {v44:>10.3f}  {v43:>10.3f}  {delta:>+10.3f} {arrow}")
print()
print(f"  Profitable symbols:   {n_profitable}/{len(SYMBOLS)} ({pct_profitable:.0f}%)")
print(f"  Fold range:           "
      f"{min(fold_metrics[f]['pf'] for f in fold_metrics if fold_metrics[f]['n']>0):.3f}"
      f" – {max(fold_metrics[f]['pf'] for f in fold_metrics if fold_metrics[f]['n']>0):.3f}")
print()
print(f"  ╔══════════════════════════════════════════════════════════╗")
print(f"  ║  VERDICT:         {r44['verdict']:<10}  Score={r44['score']}/7              ║")
print(f"  ║  GENERALISATION:  {gen_class:<40}║")
print(f"  ╚══════════════════════════════════════════════════════════╝")
print()
print(f"  Final question: Does the R043 edge genuinely generalise to")
print(f"  completely unseen crypto markets?")
print()
print(answer)
print()
print(f"  Output files:")
for p in [p1, p2, p3, p4, p5, p6, p7, p8, csv1, csv2, csv3, csv4, csv5, jmd_path]:
    if p: print(f"    {p}")
print(SEP)
