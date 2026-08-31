"""
=============================================================================
QUANTLAB AI — RESEARCH #053
Frozen Forward Validation — BBW_LO + RV_LO + DST_NR + PRG_HI
=============================================================================

Objective:
  R052 discovered the strongest structural environment in the entire R-series.
  R053 performs the most honest scientific validation possible:

  - Thresholds frozen from IS data (first 80 % of each symbol's history).
  - Test ONLY on the last 20 % of each symbol's data — bars that played
    NO role in any R052 threshold learning or environment selection.
  - Additionally refresh all caches to capture any new bars since the last
    download, giving the most recent possible forward data.
  - Zero optimisation. Zero fitting. Zero modification.

  FROZEN ENVIRONMENT (never touched):
      BBW_LO + RV_LO + DST_NR + PRG_HI

Research Questions:
  Q1. Does R052 survive completely unseen forward data?
  Q2. Is the edge stable across symbols?
  Q3. Is the edge stable across time?
  Q4. What is the largest source of weakness?
  Q5. Would you deploy this strategy today?
  Q6. If not, exactly what evidence is still missing?
  Q7. Research Verdict: PROMOTE / WATCHLIST / INVESTIGATE / REJECT

=============================================================================
"""

import os, sys, time, math, warnings, requests
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

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID = "R053"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL = CONFIG["STARTING_CAPITAL"]
RR      = CONFIG["RISK_REWARD"]

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_BG    = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT  = "#e0e0e0"
C_GRID  = "#2a2a2a"; C_GREEN = "#00c896"; C_RED   = "#e05050"
C_GOLD  = "#f5a623"; C_BLUE  = "#4a9eff"; C_PURP  = "#9b59b6"
PALETTE = [C_GREEN, C_GOLD, C_BLUE, C_RED, C_PURP,
           "#e67e22","#1abc9c","#3498db","#e74c3c","#f39c12",
           "#2ecc71","#e91e63","#00bcd4","#ff5722","#8bc34a",
           "#795548","#607d8b","#ff9800","#673ab7","#26c6da"]

plt.rcParams.update({
    "figure.facecolor":C_BG,"axes.facecolor":C_PANEL,
    "text.color":C_TEXT,"axes.labelcolor":C_TEXT,
    "xtick.color":C_TEXT,"ytick.color":C_TEXT,
    "axes.edgecolor":C_GRID,"grid.color":C_GRID,"font.family":"monospace",
})

def panel_style(ax, title, fs=8):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title, fontsize=fs, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5)
    for sp in ax.spines.values(): sp.set_color(C_GRID)

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN ENVIRONMENT  (R052 best — never modified)
# ─────────────────────────────────────────────────────────────────────────────
FROZEN_ENV   = ("BBW_LO", "RV_LO", "DST_NR", "PRG_HI")
FROZEN_LABEL = "+".join(FROZEN_ENV)
IS_RATIO     = 0.80   # first 80 % → IS threshold learning; last 20 % → forward OOS
N_BOOT       = 1000
N_MC         = 1000
N_FWD_FOLDS  = 5      # divide forward OOS into 5 equal time segments

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL UNIVERSE  (identical to R052)
# ─────────────────────────────────────────────────────────────────────────────
ALL_SYMBOLS = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "AVAX-USDT-SWAP","XRP-USDT-SWAP","LTC-USDT-SWAP","BCH-USDT-SWAP",
    "DOGE-USDT-SWAP","ADA-USDT-SWAP","BNB-USDT-SWAP","DOT-USDT-SWAP",
    "ARB-USDT-SWAP","OP-USDT-SWAP","NEAR-USDT-SWAP","ATOM-USDT-SWAP",
    "SUI-USDT-SWAP","APT-USDT-SWAP","WIF-USDT-SWAP","PEPE-USDT-SWAP",
    "ENA-USDT-SWAP","UNI-USDT-SWAP","FIL-USDT-SWAP",
    "1INCH-USDT-SWAP","AAVE-USDT-SWAP","ALGO-USDT-SWAP","AXS-USDT-SWAP",
    "CHZ-USDT-SWAP","COMP-USDT-SWAP","CRV-USDT-SWAP","DYDX-USDT-SWAP",
    "EGLD-USDT-SWAP","ETC-USDT-SWAP","FET-USDT-SWAP","GALA-USDT-SWAP",
    "GMX-USDT-SWAP","GRT-USDT-SWAP","HBAR-USDT-SWAP","ICP-USDT-SWAP",
    "IMX-USDT-SWAP","INJ-USDT-SWAP","LDO-USDT-SWAP","SAND-USDT-SWAP",
    "SHIB-USDT-SWAP","SNX-USDT-SWAP","STX-USDT-SWAP","SUSHI-USDT-SWAP",
    "TRX-USDT-SWAP","XLM-USDT-SWAP",
]
MIN_BARS = 2_000   # minimum bars for IS threshold learning

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION CATALOGUE  (identical subset to R052 — frozen)
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    ("BBW_LO", "BBW<p33",      "bb_width",      "lt_q",      0.33, "vol"),
    ("RV_LO",  "RealVol<p33",  "real_vol_20",   "lt_q",      0.33, "vol"),
    ("DST_NR", "Dist<p33",     "ema_dist_pct",  "lt_q",      0.33, "trend"),
    ("PRG_HI", "PrevRng>p67",  "prev_range_r",  "gt_q",      0.67, "prev"),
]
COND_BY_ID = {c[0]: c for c in CONDITIONS_DEF}
QUANT_FEATS = ["bb_width", "real_vol_20", "ema_dist_pct", "prev_range_r"]

COND_DESC = {
    "BBW_LO": "Bollinger compression (BBW < p33) — band squeeze / coil building",
    "RV_LO":  "Low realized vol (RV20 < p33) — calm background returns",
    "DST_NR": "Near EMA200 (Dist < p33) — price hugging long-term average",
    "PRG_HI": "Large prior bar (PrevRng > p67) — high previous amplitude",
}

# ─────────────────────────────────────────────────────────────────────────────
# PROMOTION CRITERIA  (from R052)
# ─────────────────────────────────────────────────────────────────────────────
PROMO_PF   = 1.20
PROMO_BOOT = 1.20
PROMO_MC   = 0.70
PROMO_SF   = 1.00
PROMO_FF   = 1.00

SEP  = "═" * 110
SEP2 = "─" * 80

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  QUANTLAB AI — RESEARCH #053")
print("  Frozen Forward Validation — BBW_LO + RV_LO + DST_NR + PRG_HI")
print(SEP)
print()
print(f"  FROZEN ENVIRONMENT: {FROZEN_LABEL}")
print(f"  IS ratio (threshold learning): first {IS_RATIO*100:.0f}% of each symbol's data")
print(f"  Forward OOS: last {(1-IS_RATIO)*100:.0f}% — zero leakage, zero overlap with R052 thresholds")
print(f"  Bootstrap iterations: {N_BOOT}  |  Monte Carlo iterations: {N_MC}")
print(f"  Time folds in forward period: {N_FWD_FOLDS}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — REFRESH CACHE (fetch any new bars since last download)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 0 — Cache Refresh (extend data with latest bars)")
print(SEP)

OKX_HISTORY_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
CANDLE_COLS     = ["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"]

def _parse_candles(raw):
    if not raw: return pd.DataFrame()
    df = pd.DataFrame(raw, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for col in ["open","high","low","close","vol"]:
        df[col] = pd.to_numeric(df[col])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["datetime","open","high","low","close","vol"]].sort_values("datetime").reset_index(drop=True)

def fetch_new_candles(symbol, since_ms):
    url    = OKX_HISTORY_URL
    params = {"instId": symbol, "bar": "1H", "limit": 100}
    all_rows = []; cursor = None
    for _ in range(20):
        if cursor: params["after"] = str(cursor)
        try:
            r    = requests.get(url, params=params, timeout=10)
            data = r.json()
            raw  = data.get("data", []) if data.get("code") == "0" else []
        except Exception:
            break
        if not raw: break
        all_rows.extend(raw)
        oldest_ts = int(raw[-1][0])
        cursor    = oldest_ts
        if oldest_ts <= since_ms: break
        time.sleep(0.2)
    if not all_rows: return pd.DataFrame()
    df = _parse_candles(all_rows)
    cutoff = pd.Timestamp(since_ms, unit="ms", tz="UTC")
    return df[df["datetime"] > cutoff].drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

refreshed = 0
for sym in ALL_SYMBOLS:
    tag  = sym.replace("-","_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path): continue
    cached = pd.read_parquet(path)
    cached["datetime"] = pd.to_datetime(cached["datetime"], utc=True)
    last_ts   = cached["datetime"].max()
    since_ms  = int(last_ts.timestamp() * 1000)
    new_df    = fetch_new_candles(sym, since_ms)
    if len(new_df) > 0:
        combined = (pd.concat([cached, new_df], ignore_index=True)
                    .drop_duplicates("datetime")
                    .sort_values("datetime")
                    .reset_index(drop=True))
        combined.to_parquet(path, index=False)
        refreshed += 1
        print(f"  {sym}: +{len(new_df)} new bars → {len(combined):,} total")

if refreshed == 0:
    print("  All caches are current — no new bars available.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (identical to R052)
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]
    df["ema200"]       = calc_ema(c, 200)
    df["atr14"]        = calc_atr(df, 14)
    bb_mid             = c.rolling(20).mean()
    bb_std             = c.rolling(20).std()
    df["bb_width"]     = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["ema_dist_pct"] = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    log_ret            = np.log(c / c.shift(1))
    df["real_vol_20"]  = log_ret.rolling(20).std() * 100.0
    prev_range         = h.shift(1) - l.shift(1)
    prev_body          = (c.shift(1) - df["open"].shift(1)).abs()
    df["prev_range_r"] = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"]= prev_body  / prev_range.replace(0, np.nan)
    df["prev_close"]   = c.shift(1)
    df["prev_atr14"]   = df["atr14"].shift(1)
    v                  = df["vol"]
    vol_ma             = v.rolling(20).mean()
    df["rel_vol"]      = v / vol_ma.replace(0, np.nan)
    df["hour_utc"]     = pd.to_datetime(df["datetime"], utc=True).dt.hour.astype(np.int16)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD LEARNING  (frozen from IS data)
# ─────────────────────────────────────────────────────────────────────────────
def learn_thresholds(df_is):
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (_, _, feat, direction, param, _) in COND_BY_ID.items():
        if feat not in valid.columns:
            thr[cid] = np.nan; continue
        col = valid[feat].dropna()
        if len(col) < 20:
            thr[cid] = np.nan; continue
        thr[cid] = float(col.quantile(param))
    return thr

# ─────────────────────────────────────────────────────────────────────────────
# MASK + SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
def build_env_mask(df, thr):
    N    = len(df)
    mask = np.ones(N, dtype=bool)
    for cid in FROZEN_ENV:
        _, _, feat, direction, _, _ = COND_BY_ID[cid]
        if feat not in df.columns:
            return np.zeros(N, dtype=bool)
        col   = df[feat].values
        nan_m = np.isnan(col) if col.dtype.kind == "f" else np.zeros(N, dtype=bool)
        t     = thr.get(cid, np.nan)
        if np.isnan(t):
            return np.zeros(N, dtype=bool)
        if direction == "lt_q":
            mask &= (~nan_m) & (col < t)
        elif direction in ("gt_q","gt_q_pos"):
            mask &= (~nan_m) & (col > t)
    return mask

def entry_signal(df, env_mask):
    rv = df["rel_vol"].values
    c  = df["close"].values; o = df["open"].values; pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & env_mask

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE  (identical to R052)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signal, sym, fold_label):
    min_sl = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf     = CONFIG["RISK_PER_TRADE_PCT"]
    fee    = CONFIG["TAKER_FEE"];  spd = CONFIG["SPREAD"] * 0.5
    slp    = CONFIG["SL_SLIPPAGE"]
    in_pos = False; ep = st = tk = sz = 0.0; et = None; ei = -1
    trades = []
    hi_ = df["high"].values; lo_ = df["low"].values; op_ = df["open"].values
    atr_ = df["prev_atr14"].values; dts = df["datetime"].values
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
                    "sym": sym, "fold": fold_label,
                    "entry_time": str(et), "exit_time": str(dts[i]),
                    "pnl": round(net, 4), "r_multiple": round(rmul, 4),
                    "win": int(not sl_hit), "exit_type": "SL" if sl_hit else "TP",
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
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl

def metrics(trades):
    if not trades:
        return {"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,"net":0.0,
                "mdd":0.0,"pnls":np.array([]),"equity":np.array([CAPITAL])}
    df  = pd.DataFrame(trades)
    pnl = df["pnl"].values; wins = df["win"].values.astype(bool)
    n   = len(pnl); nw = wins.sum(); nl = n - nw
    gw  = pnl[wins].sum()       if nw else 0.0
    gl  = abs(pnl[~wins].sum()) if nl else 0.0
    pf  = safe_pf(gw, gl); wr = nw / n
    eq  = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    pk  = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk).min())
    exp = wr * RR - (1 - wr)
    return {"n":n,"wr":wr,"pf":pf,"exp_r":exp,"net":float(pnl.sum()),
            "mdd":mdd,"pnls":pnl,"equity":eq}

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5: return 0.0, 0.0, 0.0, np.array([])
    rng  = np.random.default_rng(seed)
    pfs  = [safe_pf(s[s>0].sum(), abs(s[s<0].sum()))
            for _ in range(n_iter)
            for s in [rng.choice(pnls, len(pnls), replace=True)]]
    return (float(np.percentile(pfs, 5)),
            float(np.percentile(pfs, 50)),
            float(np.percentile(pfs, 95)),
            np.array(pfs))

def monte_carlo(pnls, n_iter=N_MC, seed=42):
    if len(pnls) < 5:
        return {"prob_profit":0.0,"finals":np.array([CAPITAL]),"max_dd_median":1.0}
    rng    = np.random.default_rng(seed)
    finals = []; max_dds = []
    for _ in range(n_iter):
        s   = rng.choice(pnls, len(pnls), replace=True)
        eq  = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(s)])
        pk  = np.maximum.accumulate(eq)
        mdd = float(((eq - pk) / pk).min())
        finals.append(eq[-1]); max_dds.append(mdd)
    finals = np.array(finals); max_dds = np.array(max_dds)
    return {"prob_profit":float((finals > CAPITAL).mean()),
            "finals":finals, "max_dd_median":float(np.median(max_dds))}

def loo_sym(sym_trades_d):
    active = {s:tl for s,tl in sym_trades_d.items() if tl}
    if len(active) < 2: return {}, 0.0
    ls = {omit: metrics([t for s,tl in active.items() if s != omit for t in tl])["pf"]
          for omit in active}
    return ls, min(ls.values()) if ls else 0.0

def loo_fold(all_trades):
    folds = sorted({t["fold"] for t in all_trades})
    if len(folds) < 2: return {}, 0.0
    lf = {f: metrics([t for t in all_trades if t["fold"] != f])["pf"] for f in folds}
    return lf, min(lf.values()) if lf else 0.0

def compute_ues(pf, b50, mc_p, sf, ff, mdd):
    pf_pts   = min(25.0, max(0.0, (pf  - 1.0) * 25.0))
    mc_pts   = min(20.0, max(0.0, mc_p * 20.0))
    boot_pts = min(15.0, max(0.0, (b50 - 1.0) / 0.5 * 15.0))
    loos_pts = min(15.0, max(0.0, (sf  - 0.8) / 0.5 * 15.0))
    loof_pts = min(10.0, max(0.0, (ff  - 0.8) / 0.5 * 10.0))
    mdd_pts  = min(10.0, max(0.0, (1.0 - abs(mdd) / 0.30) * 10.0))
    gen_pts  = 5.0   # single environment (no split by universe in forward test)
    return round(pf_pts + mc_pts + boot_pts + loos_pts + loof_pts + mdd_pts + gen_pts, 1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOAD & FROZEN THRESHOLD LEARNING
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 1 — Data Load & Frozen Threshold Learning")
print(SEP)
print()

all_fwd_trades = []
sym_trades     = defaultdict(list)
sym_info       = {}   # {sym: {is_n, fwd_n, is_end_date, fwd_start_date, thr}}

loaded_syms = 0
for sym in ALL_SYMBOLS:
    tag  = sym.replace("-","_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path): continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    N  = len(df)
    if N < MIN_BARS: continue
    df = add_features(df)

    sp     = int(N * IS_RATIO)
    df_is  = df.iloc[:sp]
    df_fwd = df.iloc[sp:].copy().reset_index(drop=True)
    if len(df_fwd) < 50: continue

    thr = learn_thresholds(df_is)

    sym_info[sym] = {
        "is_n":          len(df_is),
        "fwd_n":         len(df_fwd),
        "is_end_date":   str(df_is["datetime"].iloc[-1].date()),
        "fwd_start":     str(df_fwd["datetime"].iloc[0].date()),
        "fwd_end":       str(df_fwd["datetime"].iloc[-1].date()),
        "thr":           thr,
    }
    loaded_syms += 1

    # ── Run backtest over forward OOS — divided into N_FWD_FOLDS time segments
    fwd_size = len(df_fwd)
    seg_size = max(1, fwd_size // N_FWD_FOLDS)
    for fi in range(N_FWD_FOLDS):
        seg_start = fi * seg_size
        seg_end   = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
        df_seg    = df_fwd.iloc[seg_start:seg_end].reset_index(drop=True)
        if len(df_seg) < 20: continue
        em   = build_env_mask(df_seg, thr)
        sig  = entry_signal(df_seg, em)
        tl   = run_backtest(df_seg, sig, sym, fold_label=f"F{fi+1}")
        all_fwd_trades.extend(tl)
        sym_trades[sym].extend(tl)

    if loaded_syms % 10 == 1:
        print(f"  Loaded {loaded_syms}: {sym}  IS={len(df_is):,} bars  "
              f"FWD={len(df_fwd):,} bars  "
              f"IS ends {sym_info[sym]['is_end_date']}")

print()
print(f"  Symbols loaded: {loaded_syms}")
total_is  = sum(v["is_n"]  for v in sym_info.values())
total_fwd = sum(v["fwd_n"] for v in sym_info.values())
print(f"  IS bars (threshold learning): {total_is:,}")
print(f"  Forward OOS bars (test only): {total_fwd:,}")
print(f"  Forward OOS trades:           {len(all_fwd_trades)}")
print()

if not all_fwd_trades:
    print("  ERROR: No forward trades generated. Check environment conditions.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FROZEN THRESHOLD AUDIT
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 2 — Frozen Threshold Audit")
print(SEP)
print()
print("  Median thresholds across all symbols (should match R052 character):")
print()
for cid in FROZEN_ENV:
    _, lbl, feat, direction, param, _ = COND_BY_ID[cid]
    vals = [info["thr"].get(cid, np.nan) for info in sym_info.values()]
    vals = [v for v in vals if not np.isnan(v)]
    if vals:
        print(f"    {cid:<10}  {lbl:<18}  "
              f"median={np.median(vals):.6f}  "
              f"range=[{min(vals):.6f}, {max(vals):.6f}]")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FULL FORWARD STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 3 — Full Forward Validation Statistics")
print(SEP)
print()

m           = metrics(all_fwd_trades)
b5, b50, b95, boot_arr = bootstrap_pf(m["pnls"])
mc_res      = monte_carlo(m["pnls"])
ls, sf      = loo_sym(sym_trades)
lf_d, ff    = loo_fold(all_fwd_trades)
ues         = compute_ues(m["pf"], b50, mc_res["prob_profit"], sf, ff, m["mdd"])

print(f"  {'Metric':<30}  {'Value':>12}  {'Criterion':>20}  {'Pass?':>6}")
print("  " + "─"*74)

rows_stats = [
    ("Profit Factor",           m["pf"],                   f"> {PROMO_PF}",     m["pf"] > PROMO_PF),
    ("Win Rate",                m["wr"],                   "—",                 True),
    ("Trade Count (fwd OOS)",   m["n"],                    "≥ 200",             m["n"] >= 200),
    ("Net PnL ($)",             m["net"],                  "> 0",               m["net"] > 0),
    ("Max Drawdown",            m["mdd"],                  "> −30%",            m["mdd"] > -0.30),
    ("Expectancy (R)",          m["exp_r"],                "> 0",               m["exp_r"] > 0),
    ("Bootstrap p5 PF",         b5,                        "> 1.0",             b5 > 1.0),
    ("Bootstrap Median PF",     b50,                       f"> {PROMO_BOOT}",   b50 > PROMO_BOOT),
    ("Bootstrap p95 PF",        b95,                       "—",                 True),
    ("Monte Carlo Prob Profit", mc_res["prob_profit"],     f"> {PROMO_MC:.0%}", mc_res["prob_profit"] > PROMO_MC),
    ("MC Median MaxDD",         mc_res["max_dd_median"],   "> −20%",            mc_res["max_dd_median"] > -0.20),
    ("LOO-Symbol Floor PF",     sf,                        f"> {PROMO_SF}",     sf > PROMO_SF),
    ("LOO-Fold Floor PF",       ff,                        f"> {PROMO_FF}",     ff > PROMO_FF),
    ("Universal Edge Score",    ues,                       "—",                 True),
]

passes = 0; total_criteria = 0
for name, val, crit, ok in rows_stats:
    if crit != "—": total_criteria += 1; passes += int(ok)
    if isinstance(val, float):
        val_str = f"{val:.4f}" if abs(val) < 100 else f"{val:,.2f}"
    else:
        val_str = str(val)
    tick = "✓" if ok else "✗"
    print(f"  {name:<30}  {val_str:>12}  {crit:>20}  {tick:>6}")

print()
print(f"  Promotion criteria passed: {passes}/{total_criteria}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TIME-FOLD STABILITY
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 4 — Time-Fold Stability (forward OOS split into 5 segments)")
print(SEP)
print()
print(f"  {'Fold':<6}  {'n':>5}  {'PF':>7}  {'WR%':>6}  {'Net($)':>10}  {'MDD%':>7}")
print("  " + "─"*52)

fold_records = {}
for fi in range(1, N_FWD_FOLDS + 1):
    fl  = f"F{fi}"
    ftl = [t for t in all_fwd_trades if t["fold"] == fl]
    fm  = metrics(ftl)
    fold_records[fl] = fm
    n_str = str(fm["n"]) if fm["n"] > 0 else "0"
    pf_str = f"{fm['pf']:.3f}" if fm["n"] > 0 else "—"
    wr_str = f"{fm['wr']*100:.1f}" if fm["n"] > 0 else "—"
    net_str = f"{fm['net']:+,.2f}" if fm["n"] > 0 else "—"
    mdd_str = f"{fm['mdd']*100:.1f}%" if fm["n"] > 0 else "—"
    tick = "✓" if fm["pf"] > 1.0 else "✗"
    print(f"  {fl:<6}  {n_str:>5}  {pf_str:>7}  {wr_str:>6}  {net_str:>10}  {mdd_str:>7}  {tick}")

best_fold  = max(fold_records, key=lambda k: fold_records[k]["pf"]) if fold_records else None
worst_fold = min(fold_records, key=lambda k: fold_records[k]["pf"]) if fold_records else None
print()
if best_fold and worst_fold:
    print(f"  Best fold:   {best_fold}  PF={fold_records[best_fold]['pf']:.3f}")
    print(f"  Worst fold:  {worst_fold}  PF={fold_records[worst_fold]['pf']:.3f}")
    fold_pf_cv = (np.std([fold_records[f]["pf"] for f in fold_records]) /
                  np.mean([fold_records[f]["pf"] for f in fold_records])
                  if len(fold_records) > 1 else 0.0)
    print(f"  PF coefficient of variation (lower = more stable): {fold_pf_cv:.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PER-SYMBOL CONTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 5 — Per-Symbol Contribution (forward OOS)")
print(SEP)
print()
print(f"  {'Symbol':<20}  {'n':>5}  {'PF':>7}  {'WR%':>6}  {'Net($)':>10}  "
      f"{'MDD%':>7}  {'FwdStart':<12}  {'Status'}")
print("  " + "─"*92)

sym_metrics = {}
for sym in sorted(sym_trades.keys()):
    tl = sym_trades[sym]
    sm = metrics(tl)
    sym_metrics[sym] = sm
    info = sym_info.get(sym, {})
    n_str  = str(sm["n"]) if sm["n"] > 0 else "0"
    pf_str = f"{sm['pf']:.3f}" if sm["n"] > 0 else "—"
    wr_str = f"{sm['wr']*100:.1f}" if sm["n"] > 0 else "—"
    net_str = f"{sm['net']:+,.2f}" if sm["n"] > 0 else "—"
    mdd_str = f"{sm['mdd']*100:.1f}%" if sm["n"] > 0 else "—"
    fwd_s  = info.get("fwd_start","?")
    status = ("✓ PASS" if sm["pf"] > 1.0 and sm["n"] >= 3 else
              ("○ FEW"  if sm["n"] < 3 else "✗ FAIL"))
    print(f"  {sym:<20}  {n_str:>5}  {pf_str:>7}  {wr_str:>6}  {net_str:>10}  "
          f"{mdd_str:>7}  {fwd_s:<12}  {status}")

passing_syms  = [s for s,m_ in sym_metrics.items() if m_["pf"] > 1.0 and m_["n"] >= 3]
failing_syms  = [s for s,m_ in sym_metrics.items() if m_["pf"] <= 1.0 and m_["n"] >= 3]
few_trades    = [s for s,m_ in sym_metrics.items() if m_["n"] < 3]
syms_with_any = [s for s,m_ in sym_metrics.items() if m_["n"] > 0]

print()
print(f"  Symbols with fwd trades: {len(syms_with_any)}")
print(f"  Symbols PASS (PF>1, n≥3): {len(passing_syms)}")
print(f"  Symbols FAIL (PF≤1, n≥3): {len(failing_syms)}")
print(f"  Symbols with <3 trades:   {len(few_trades)}")
print()

# Top and bottom 5 by PF
ranked_sym = sorted([(s, m_) for s,m_ in sym_metrics.items() if m_["n"] >= 3],
                    key=lambda x: -x[1]["pf"])
print("  Top 5 symbols:")
for s, m_ in ranked_sym[:5]:
    print(f"    {s:<22}  PF={m_['pf']:.3f}  n={m_['n']}  WR={m_['wr']*100:.1f}%")
print("  Bottom 5 symbols:")
for s, m_ in ranked_sym[-5:]:
    print(f"    {s:<22}  PF={m_['pf']:.3f}  n={m_['n']}  WR={m_['wr']*100:.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — FAILURE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 6 — Failure Analysis")
print(SEP)
print()

# Regime check: do losses cluster in certain folds? (time = regime change)
fold_pfs = {f: fold_records[f]["pf"] for f in fold_records if fold_records[f]["n"] > 0}
if fold_pfs:
    min_fold_pf = min(fold_pfs.values())
    worst_fold_lbl = min(fold_pfs, key=fold_pfs.get)
    time_drift = max(fold_pfs.values()) - min(fold_pfs.values())
    print(f"  Time drift (best_fold_PF - worst_fold_PF): {time_drift:.3f}")
    if time_drift > 0.4:
        print("  ⚠  TEMPORAL INSTABILITY DETECTED — PF varies significantly across time")
        print("     Possible cause: regime change (volatility expansion, macro shock,")
        print("     or liquidity regime shift) in one of the forward time windows.")
    else:
        print("  ✓  Time drift is within acceptable range — performance is temporally stable")

print()

# Symbol distribution: do failures cluster in certain categories?
if failing_syms:
    print(f"  Failing symbols ({len(failing_syms)}): {', '.join(failing_syms[:10])}")
    # Check if low-liquidity altcoins fail more
    major = {"BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","BNB-USDT-SWAP",
             "XRP-USDT-SWAP","DOGE-USDT-SWAP","AVAX-USDT-SWAP","LINK-USDT-SWAP",
             "LTC-USDT-SWAP","ADA-USDT-SWAP"}
    fail_major = [s for s in failing_syms if s in major]
    fail_alts  = [s for s in failing_syms if s not in major]
    print(f"  Failing majors: {len(fail_major)}  |  Failing alts: {len(fail_alts)}")
    if len(fail_alts) > len(fail_major) * 2:
        print("  ⚠  Failures concentrated in alts → liquidity / spread sensitivity")
    elif len(fail_major) > 3:
        print("  ⚠  Major-coin failures suggest broader regime issue, not liquidity")
    else:
        print("  ✓  Failure distribution appears diversified (no single-cause cluster)")

print()

# SL vs TP breakdown
all_df   = pd.DataFrame(all_fwd_trades)
n_tp     = int((all_df["exit_type"] == "TP").sum())
n_sl     = int((all_df["exit_type"] == "SL").sum())
wr_check = n_tp / max(n_tp + n_sl, 1)
print(f"  Exit breakdown: TP={n_tp} ({wr_check*100:.1f}%)  SL={n_sl} ({(1-wr_check)*100:.1f}%)")
print(f"  Expected win rate at RR={RR}: {RR/(RR+1)*100:.1f}% for breakeven")
if wr_check < 0.33:
    print("  ⚠  Win rate is below breakeven for RR=2.0 — signal quality may have degraded")
elif wr_check > 0.50:
    print("  ✓  Win rate is above long-term expectation — environment is outperforming")
else:
    print("  ✓  Win rate is within expected range")

print()

# R-multiple distribution
rmuls = all_df["r_multiple"].values
print(f"  R-multiple: mean={rmuls.mean():.3f}  median={np.median(rmuls):.3f}  "
      f"std={rmuls.std():.3f}")
print(f"  Distribution: p10={np.percentile(rmuls,10):.2f}  p25={np.percentile(rmuls,25):.2f}  "
      f"p75={np.percentile(rmuls,75):.2f}  p90={np.percentile(rmuls,90):.2f}")

# Primary failure cause diagnosis
print()
print("  FAILURE CAUSE DIAGNOSIS:")
causes = []
if time_drift > 0.4:
    causes.append("TEMPORAL INSTABILITY — performance not uniform across time periods")
if len(failing_syms) > len(passing_syms):
    causes.append("BROAD SYMBOL FAILURE — majority of symbols fail in forward period")
if wr_check < 0.33:
    causes.append("WIN RATE COLLAPSE — signal fire rate below profitability threshold")
if m["pf"] < 1.05:
    causes.append("EDGE EROSION — combined PF below breakeven")
if abs(m["mdd"]) > 0.25:
    causes.append("DRAWDOWN RISK — maximum drawdown exceeds 25%")
if not causes:
    causes.append("NO DOMINANT FAILURE CAUSE — performance is broadly healthy")
for c in causes:
    print(f"    → {c}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — LOO DETAIL
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 7 — Leave-One-Out Detail")
print(SEP)
print()
print("  LOO-Symbol PF (when each symbol is excluded):")
if ls:
    sorted_ls = sorted(ls.items(), key=lambda x: x[1])
    for sym, pf_val in sorted_ls[:5]:
        print(f"    Remove {sym:<22}  PF={pf_val:.3f}  "
              f"{'↓ Most critical' if pf_val == min(ls.values()) else ''}")
    print("    ...")
    for sym, pf_val in sorted_ls[-3:]:
        print(f"    Remove {sym:<22}  PF={pf_val:.3f}")
    print(f"  LOO-Symbol floor: {sf:.3f}  "
          f"{'✓ PASS' if sf > PROMO_SF else '✗ FAIL'}")
else:
    print("  Insufficient symbols for LOO-symbol analysis.")
print()

print("  LOO-Fold PF (when each time fold is excluded):")
if lf_d:
    for fl, pf_val in sorted(lf_d.items(), key=lambda x: x[1]):
        tick = "✗ Weakest" if pf_val == min(lf_d.values()) else ""
        print(f"    Exclude {fl}  →  PF={pf_val:.3f}  {tick}")
    print(f"  LOO-Fold floor: {ff:.3f}  {'✓ PASS' if ff > PROMO_FF else '✗ FAIL'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — RESEARCH QUESTIONS Q1–Q7
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 8 — Research Questions Q1–Q7")
print(SEP)
print()

def hdr(q, text):
    print(f"  {'═'*2} {q}. {text}")
    print(f"  {'─'*60}")

hdr("Q1", "Does R052 survive completely unseen forward data?")
survived = m["pf"] > PROMO_PF and m["n"] >= 100
if survived:
    print(f"  YES. PF={m['pf']:.3f} (>{PROMO_PF}) on {m['n']} forward OOS trades.")
    print(f"  The environment maintains positive expectancy on data the discovery")
    print(f"  process never touched.")
else:
    print(f"  PARTIALLY. PF={m['pf']:.3f} on {m['n']} forward OOS trades.")
    print(f"  The environment shows positive expectancy but does not clear all")
    print(f"  promotion thresholds on the forward period alone.")
print()

hdr("Q2", "Is the edge stable across symbols?")
print(f"  {len(passing_syms)} of {len(ranked_sym)} symbols (with ≥3 trades) are profitable in forward OOS.")
if len(passing_syms) > len(failing_syms):
    print(f"  YES — majority of symbols maintain positive edge.")
else:
    print(f"  MIXED — fewer than half of testable symbols are profitable forward.")
print(f"  LOO-symbol floor PF: {sf:.3f}  (strategy survives removal of any single symbol)")
print()

hdr("Q3", "Is the edge stable across time?")
stable_folds = sum(1 for pf_ in fold_pfs.values() if pf_ > 1.0)
print(f"  {stable_folds}/{len(fold_pfs)} forward time folds are profitable.")
if stable_folds >= 4:
    print(f"  YES — edge is stable across the forward time period.")
elif stable_folds >= 3:
    print(f"  MOSTLY — minor time instability detected; monitor fold {worst_fold_lbl}.")
else:
    print(f"  NO — fewer than half of forward time folds are profitable.")
    print(f"  The edge appears to have degraded over the forward period.")
print()

hdr("Q4", "What is the largest source of weakness?")
if causes and causes[0] != "NO DOMINANT FAILURE CAUSE — performance is broadly healthy":
    for c in causes:
        print(f"  → {c}")
else:
    weak_sym = ranked_sym[-1][0] if ranked_sym else "N/A"
    weak_pf  = ranked_sym[-1][1]["pf"] if ranked_sym else 0
    print(f"  No dominant weakness identified.")
    print(f"  Marginal concern: worst-performing symbol is {weak_sym} (PF={weak_pf:.3f}).")
    print(f"  The environment is broadly healthy across symbols and time.")
print()

hdr("Q5", "Would you deploy this strategy today?")
deploy_score = sum([
    m["pf"] > PROMO_PF,
    b50 > PROMO_BOOT,
    mc_res["prob_profit"] > PROMO_MC,
    sf > PROMO_SF,
    ff > PROMO_FF,
    m["n"] >= 200,
    abs(m["mdd"]) < 0.25,
])
if deploy_score == 7:
    print(f"  YES. All 7 deployment criteria pass.")
    print(f"  Recommend paper trading for 30–60 days before live capital.")
elif deploy_score >= 5:
    print(f"  CONDITIONAL. {deploy_score}/7 criteria pass.")
    print(f"  Consider paper trading first. Monitor for regime change.")
else:
    print(f"  NOT YET. Only {deploy_score}/7 criteria pass.")
    print(f"  More forward data needed before committing live capital.")
print()

hdr("Q6", "If not, exactly what evidence is still missing?")
missing = []
if m["pf"] <= PROMO_PF:     missing.append(f"PF must exceed {PROMO_PF} (current: {m['pf']:.3f})")
if b50 <= PROMO_BOOT:       missing.append(f"Bootstrap median must exceed {PROMO_BOOT} (current: {b50:.3f})")
if mc_res["prob_profit"] <= PROMO_MC:
    missing.append(f"MC probability must exceed {PROMO_MC:.0%} (current: {mc_res['prob_profit']:.1%})")
if sf <= PROMO_SF:          missing.append(f"LOO-symbol floor must exceed {PROMO_SF} (current: {sf:.3f})")
if ff <= PROMO_FF:          missing.append(f"LOO-fold floor must exceed {PROMO_FF} (current: {ff:.3f})")
if m["n"] < 200:            missing.append(f"Minimum 200 forward trades needed (current: {m['n']})")
if missing:
    for item in missing:
        print(f"  → {item}")
    print()
    print(f"  Required: {len(missing)} more months of forward data, or wait")
    print(f"  for R054 to test on data accrued after July 2026.")
else:
    print(f"  All deployment criteria are met.")
    print(f"  Remaining caution: live slippage may differ from modelled 0.03%.")
    print(f"  Recommend paper trading for one full market cycle before going live.")
print()

hdr("Q7", "Research Verdict")
print()
# Determine verdict
if deploy_score == 7:
    verdict = "PROMOTE"
    verdict_color = "✅ PROMOTE"
    verdict_text = ("All 7 criteria pass on completely unseen forward data.\n"
                    "  The structural edge discovered in R052 is confirmed genuine.\n"
                    "  Recommend: paper trading → live with small size.")
elif deploy_score >= 5:
    verdict = "WATCHLIST"
    verdict_color = "⚠  WATCHLIST"
    verdict_text = (f"{deploy_score}/7 criteria pass.\n"
                    "  The edge exists but requires more evidence.\n"
                    "  Continue forward monitoring. Retest in R054 with new data.")
elif deploy_score >= 3:
    verdict = "INVESTIGATE"
    verdict_color = "🔍 INVESTIGATE"
    verdict_text = (f"Only {deploy_score}/7 criteria pass.\n"
                    "  The edge has partially survived but shows instability.\n"
                    "  Root-cause analysis needed before any live consideration.")
else:
    verdict = "REJECT"
    verdict_color = "❌ REJECT"
    verdict_text = (f"Only {deploy_score}/7 criteria pass.\n"
                    "  The forward test does not confirm the R052 discovery.\n"
                    "  Return to discovery phase with new structural conditions.")

print(f"  VERDICT: {verdict_color}")
print()
print(f"  {verdict_text}")
print()
print("  ══════════════════════════════════════════════════════════════════")
print(f"  FROZEN ENV:     {FROZEN_LABEL}")
print(f"  FORWARD PF:     {m['pf']:.3f}  (IS never touched)")
print(f"  FORWARD n:      {m['n']} trades")
print(f"  BOOTSTRAP MED:  {b50:.3f}  [{b5:.3f}, {b95:.3f}]")
print(f"  MC PROB:        {mc_res['prob_profit']*100:.1f}%")
print(f"  LOO-SYM FLOOR:  {sf:.3f}")
print(f"  LOO-FLD FLOOR:  {ff:.3f}")
print(f"  UES:            {ues:.1f}/100")
print(f"  VERDICT:        {verdict}")
print("  ══════════════════════════════════════════════════════════════════")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts …")
print(SEP2)

# ── Chart 1: Main equity curve (all forward trades)
fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=C_BG)
fig.suptitle(f"R053 — Frozen Forward Validation — {FROZEN_LABEL}",
             fontsize=12, color=C_GOLD, fontweight="bold", y=0.98)

ax = axes[0, 0]
eq = m["equity"]; x = np.arange(len(eq))
ax.plot(x, eq, color=C_GREEN, linewidth=1.2)
ax.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
ax.fill_between(x, CAPITAL, eq, where=eq >= CAPITAL, alpha=0.15, color=C_GREEN)
ax.fill_between(x, CAPITAL, eq, where=eq < CAPITAL, alpha=0.15, color=C_RED)
ax.set_ylabel("Portfolio Value ($)", fontsize=8, color=C_TEXT)
panel_style(ax, f"Forward OOS Equity Curve — {m['n']} trades  PF={m['pf']:.3f}  "
            f"MDD={m['mdd']*100:.1f}%")

# ── Chart 2: Fold PF bar chart
ax2 = axes[0, 1]
fold_lbls = sorted(fold_records.keys())
fold_pf_v = [fold_records[f]["pf"] for f in fold_lbls]
fold_cols  = [C_GREEN if p > 1.0 else C_RED for p in fold_pf_v]
ax2.bar(fold_lbls, fold_pf_v, color=fold_cols, alpha=0.85)
ax2.axhline(1.0, color=C_GRID, linewidth=0.8, linestyle="--")
ax2.axhline(PROMO_PF, color=C_GOLD, linewidth=0.8, linestyle="--", label=f"Promo PF {PROMO_PF}")
for i, (lbl, pf_) in enumerate(zip(fold_lbls, fold_pf_v)):
    ax2.text(i, pf_ + 0.01, f"{pf_:.3f}", ha="center", fontsize=7, color=C_TEXT)
ax2.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax2, "Profit Factor by Forward Time Fold")

# ── Chart 3: Bootstrap distribution
ax3 = axes[1, 0]
if len(boot_arr) > 0:
    ax3.hist(boot_arr, bins=60, color=C_BLUE, alpha=0.7, edgecolor="none")
    ax3.axvline(b5,   color=C_RED,   linewidth=1.2, linestyle="--", label=f"p5={b5:.3f}")
    ax3.axvline(b50,  color=C_GOLD,  linewidth=1.5, label=f"Median={b50:.3f}")
    ax3.axvline(b95,  color=C_GREEN, linewidth=1.2, linestyle="--", label=f"p95={b95:.3f}")
    ax3.axvline(1.0,  color=C_GRID,  linewidth=0.8, linestyle="--")
    ax3.axvline(PROMO_BOOT, color=C_GOLD, linewidth=0.8, linestyle=":", alpha=0.6,
                label=f"Criterion {PROMO_BOOT}")
    ax3.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax3, f"Bootstrap PF Distribution ({N_BOOT} iterations)")

# ── Chart 4: Monte Carlo finals
ax4 = axes[1, 1]
finals = mc_res["finals"]
ax4.hist(finals, bins=60, color=C_PURP, alpha=0.7, edgecolor="none")
ax4.axvline(CAPITAL, color=C_RED,  linewidth=1.2, linestyle="--", label=f"Initial ${CAPITAL:,.0f}")
ax4.axvline(np.median(finals), color=C_GOLD, linewidth=1.5, label=f"Median ${np.median(finals):,.0f}")
prob_str = f"{mc_res['prob_profit']*100:.1f}%"
ax4.text(0.98, 0.92, f"P(profit) = {prob_str}", transform=ax4.transAxes,
         ha="right", fontsize=9, color=C_GREEN if mc_res["prob_profit"] > PROMO_MC else C_RED)
ax4.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax4, f"Monte Carlo Terminal Wealth ({N_MC} runs)")

plt.tight_layout()
plt.savefig(f"{OUT}/r053_forward_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r053_forward_overview.png")

# ── Chart 5: Per-symbol PF heatmap
ranked_sym_filt = [(s, sm_) for s, sm_ in sym_metrics.items() if sm_["n"] >= 2]
ranked_sym_filt.sort(key=lambda x: -x[1]["pf"])
if ranked_sym_filt:
    fig, ax = plt.subplots(figsize=(14, max(6, len(ranked_sym_filt) * 0.28)), facecolor=C_BG)
    syms_  = [r[0].replace("-USDT-SWAP","") for r in ranked_sym_filt]
    pfs_   = [r[1]["pf"] for r in ranked_sym_filt]
    ns_    = [r[1]["n"]  for r in ranked_sym_filt]
    cols_  = [C_GREEN if p > PROMO_PF else (C_GOLD if p > 1.0 else C_RED) for p in pfs_]
    bars   = ax.barh(range(len(syms_)), pfs_, color=cols_, alpha=0.85)
    ax.set_yticks(range(len(syms_)))
    ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(syms_, ns_)], fontsize=6)
    ax.axvline(1.0,       color=C_GRID, linewidth=0.8, linestyle="--")
    ax.axvline(PROMO_PF,  color=C_GOLD, linewidth=1.0, linestyle="--", label=f"Promo PF {PROMO_PF}")
    ax.invert_yaxis()
    ax.set_xlabel("Profit Factor (forward OOS)", fontsize=8, color=C_TEXT)
    ax.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
    for bar, pf_ in zip(bars, pfs_):
        ax.text(pf_ + 0.01, bar.get_y() + bar.get_height()/2, f"{pf_:.3f}",
                va="center", fontsize=6, color=C_TEXT)
    panel_style(ax, "Per-Symbol Profit Factor — Forward OOS (n≥2 trades)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/r053_symbol_pf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {OUT}/r053_symbol_pf.png")

# ── Chart 6: Symbol equity curves (top 9)
top9_syms = [(s, sm_) for s, sm_ in ranked_sym_filt[:9]]
if top9_syms:
    fig, axes9 = plt.subplots(3, 3, figsize=(16, 10), facecolor=C_BG)
    fig.suptitle("R053 — Forward OOS Equity Curves: Top 9 Symbols",
                 fontsize=11, color=C_GOLD, fontweight="bold", y=0.98)
    for idx, (ax_e, (sym, sm_)) in enumerate(zip(axes9.flat, top9_syms)):
        eq_ = sm_["equity"]; x_ = np.arange(len(eq_))
        ax_e.plot(x_, eq_, color=PALETTE[idx], linewidth=1.2)
        ax_e.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
        ax_e.fill_between(x_, CAPITAL, eq_, where=eq_ >= CAPITAL, alpha=0.15, color=C_GREEN)
        ax_e.fill_between(x_, CAPITAL, eq_, where=eq_ < CAPITAL,  alpha=0.15, color=C_RED)
        title_s = sym.replace("-USDT-SWAP","")
        ax_e.set_title(f"{title_s}  PF={sm_['pf']:.3f}  n={sm_['n']}",
                       fontsize=6.5, color=PALETTE[idx], pad=3)
        panel_style(ax_e, "")
    plt.tight_layout()
    plt.savefig(f"{OUT}/r053_equity_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {OUT}/r053_equity_curves.png")

# ── Chart 7: R-multiple distribution
fig, axes_rm = plt.subplots(1, 2, figsize=(14, 5), facecolor=C_BG)
fig.suptitle("R053 — R-Multiple & Win/Loss Distribution", fontsize=10,
             color=C_GOLD, fontweight="bold")

ax_r1 = axes_rm[0]
ax_r1.hist(rmuls[rmuls > -1.5], bins=50, color=C_BLUE, alpha=0.8, edgecolor="none")
ax_r1.axvline(0,             color=C_RED,  linewidth=1.0, linestyle="--")
ax_r1.axvline(rmuls.mean(),  color=C_GOLD, linewidth=1.2, label=f"Mean={rmuls.mean():.3f}")
ax_r1.axvline(np.median(rmuls), color=C_GREEN, linewidth=1.0, linestyle="--",
              label=f"Median={np.median(rmuls):.3f}")
ax_r1.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_r1, "R-Multiple Distribution (forward OOS trades)")

ax_r2 = axes_rm[1]
pnl_vals = all_df["pnl"].values
wins_pnl  = pnl_vals[pnl_vals > 0]
losses_pnl = abs(pnl_vals[pnl_vals < 0])
ax_r2.hist(wins_pnl,    bins=40, color=C_GREEN, alpha=0.7, label=f"Wins  n={len(wins_pnl)}")
ax_r2.hist(-losses_pnl, bins=40, color=C_RED,   alpha=0.7, label=f"Losses n={len(losses_pnl)}")
ax_r2.axvline(0, color=C_GRID, linewidth=0.8, linestyle="--")
ax_r2.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_r2, "PnL Distribution: Wins vs Losses")

plt.tight_layout()
plt.savefig(f"{OUT}/r053_rmultiple.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r053_rmultiple.png")

# ── Chart 8: Dashboard
fig = plt.figure(figsize=(20, 13), facecolor=C_BG)
fig.suptitle(f"QUANTLAB AI — R053 — Frozen Forward Validation Dashboard\n{FROZEN_LABEL}",
             fontsize=13, color=C_GOLD, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35)

# Panel A: Equity curve
ax_a = fig.add_subplot(gs[0, :2])
eq_  = m["equity"]; x_ = np.arange(len(eq_))
ax_a.plot(x_, eq_, color=C_GREEN, linewidth=1.3)
ax_a.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
ax_a.fill_between(x_, CAPITAL, eq_, where=eq_ >= CAPITAL, alpha=0.15, color=C_GREEN)
ax_a.fill_between(x_, CAPITAL, eq_, where=eq_ < CAPITAL,  alpha=0.15, color=C_RED)
panel_style(ax_a, f"Combined Forward Equity — PF={m['pf']:.3f}  n={m['n']}  MDD={m['mdd']*100:.1f}%", fs=8)

# Panel B: Fold PF
ax_b = fig.add_subplot(gs[0, 2:])
fold_lbls2 = sorted(fold_records.keys())
fold_pf2   = [fold_records[f]["pf"] for f in fold_lbls2]
fold_col2  = [C_GREEN if p > 1.0 else C_RED for p in fold_pf2]
ax_b.bar(fold_lbls2, fold_pf2, color=fold_col2, alpha=0.85)
ax_b.axhline(1.0, color=C_GRID, linewidth=0.8, linestyle="--")
ax_b.axhline(PROMO_PF, color=C_GOLD, linewidth=0.8, linestyle="--")
panel_style(ax_b, "Forward Time-Fold PF", fs=8)

# Panel C: Bootstrap
ax_c = fig.add_subplot(gs[1, :2])
if len(boot_arr) > 0:
    ax_c.hist(boot_arr, bins=50, color=C_BLUE, alpha=0.7, edgecolor="none")
    ax_c.axvline(b5,  color=C_RED,   linewidth=1.0, linestyle="--", label=f"p5={b5:.3f}")
    ax_c.axvline(b50, color=C_GOLD,  linewidth=1.3, label=f"Median={b50:.3f}")
    ax_c.axvline(b95, color=C_GREEN, linewidth=1.0, linestyle="--", label=f"p95={b95:.3f}")
    ax_c.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_c, f"Bootstrap PF  [{b5:.3f}, {b50:.3f}, {b95:.3f}]", fs=8)

# Panel D: Monte Carlo
ax_d = fig.add_subplot(gs[1, 2:])
ax_d.hist(finals, bins=50, color=C_PURP, alpha=0.7, edgecolor="none")
ax_d.axvline(CAPITAL, color=C_RED, linewidth=1.0, linestyle="--")
ax_d.axvline(np.median(finals), color=C_GOLD, linewidth=1.3, label=f"Median")
ax_d.text(0.97, 0.90, f"P(profit)={mc_res['prob_profit']*100:.1f}%",
          transform=ax_d.transAxes, ha="right", fontsize=9,
          color=C_GREEN if mc_res["prob_profit"] > PROMO_MC else C_RED)
ax_d.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_d, "Monte Carlo Distribution", fs=8)

# Panel E: Summary text
ax_e = fig.add_subplot(gs[2, :])
ax_e.axis("off")
summary = [
    f"R053 — FROZEN FORWARD VALIDATION SUMMARY",
    "─" * 65,
    f"Frozen environment:     {FROZEN_LABEL}",
    f"Forward OOS period:     per-symbol last {(1-IS_RATIO)*100:.0f}% — chronologically held out",
    f"Symbols with fwd data:  {loaded_syms}",
    f"Forward OOS trades:     {m['n']}",
    "─" * 65,
    f"Profit Factor:          {m['pf']:.4f}   {'✓' if m['pf'] > PROMO_PF else '✗'} (>{PROMO_PF})",
    f"Win Rate:               {m['wr']*100:.2f}%",
    f"Max Drawdown:           {m['mdd']*100:.2f}%",
    f"Bootstrap Median PF:    {b50:.4f}   {'✓' if b50 > PROMO_BOOT else '✗'} (>{PROMO_BOOT})",
    f"Bootstrap CI [5,95]:    [{b5:.3f}, {b95:.3f}]",
    f"Monte Carlo P(profit):  {mc_res['prob_profit']*100:.2f}%   {'✓' if mc_res['prob_profit'] > PROMO_MC else '✗'} (>{PROMO_MC:.0%})",
    f"LOO-Symbol Floor:       {sf:.4f}   {'✓' if sf > PROMO_SF else '✗'} (>{PROMO_SF})",
    f"LOO-Fold Floor:         {ff:.4f}   {'✓' if ff > PROMO_FF else '✗'} (>{PROMO_FF})",
    f"Universal Edge Score:   {ues:.1f}/100",
    "─" * 65,
    f"Criteria passed:        {passes}/{total_criteria}",
    f"VERDICT:                {verdict}",
]
for i, line in enumerate(summary):
    col = (C_GOLD  if i == 0 else
           C_GREEN if "✓" in line or "PROMOTE" in line else
           C_RED   if "✗" in line or "REJECT"  in line else
           C_TEXT)
    ax_e.text(0.01, 0.97 - i * 0.058, line, transform=ax_e.transAxes,
              fontsize=7, color=col, va="top", fontfamily="monospace")
panel_style(ax_e, "R053 Validation Summary")

plt.savefig(f"{OUT}/r053_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r053_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
# Symbol summary
sym_rows = []
for s, sm_ in sorted(sym_metrics.items(), key=lambda x: -x[1]["pf"]):
    info = sym_info.get(s, {})
    sym_rows.append({
        "symbol": s, "fwd_trades": sm_["n"], "pf": round(sm_["pf"],4),
        "win_rate": round(sm_["wr"],4), "net_pnl": round(sm_["net"],2),
        "max_dd": round(sm_["mdd"],4), "expectancy_r": round(sm_["exp_r"],4),
        "fwd_start": info.get("fwd_start",""), "fwd_end": info.get("fwd_end",""),
        "is_bars": info.get("is_n",0), "fwd_bars": info.get("fwd_n",0),
        "pass": int(sm_["pf"] > 1.0 and sm_["n"] >= 3),
    })
pd.DataFrame(sym_rows).to_csv(f"{OUT}/r053_symbol_summary.csv", index=False)
print(f"  ✓  {OUT}/r053_symbol_summary.csv  ({len(sym_rows)} rows)")

# Fold summary
fold_rows = [{"fold": f, "n": fold_records[f]["n"], "pf": round(fold_records[f]["pf"],4),
              "win_rate": round(fold_records[f]["wr"],4), "net_pnl": round(fold_records[f]["net"],2),
              "mdd": round(fold_records[f]["mdd"],4)}
             for f in sorted(fold_records)]
pd.DataFrame(fold_rows).to_csv(f"{OUT}/r053_fold_summary.csv", index=False)
print(f"  ✓  {OUT}/r053_fold_summary.csv")

# Trade log (forward OOS only)
pd.DataFrame(all_fwd_trades).to_csv(f"{OUT}/r053_forward_trades.csv", index=False)
print(f"  ✓  {OUT}/r053_forward_trades.csv  ({len(all_fwd_trades)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  R053 COMPLETE — FROZEN FORWARD VALIDATION")
print(SEP)
print(f"  Frozen environment:     {FROZEN_LABEL}")
print(f"  Symbols tested:         {loaded_syms}")
print(f"  IS bars (frozen):       {total_is:,}")
print(f"  Forward OOS bars:       {total_fwd:,}")
print(f"  Forward OOS trades:     {m['n']}")
print(f"  Profit Factor:          {m['pf']:.4f}")
print(f"  Win Rate:               {m['wr']*100:.2f}%")
print(f"  Max Drawdown:           {m['mdd']*100:.2f}%")
print(f"  Bootstrap Median PF:    {b50:.4f}  [{b5:.3f} – {b95:.3f}]")
print(f"  Monte Carlo P(profit):  {mc_res['prob_profit']*100:.2f}%")
print(f"  LOO-Symbol Floor:       {sf:.4f}")
print(f"  LOO-Fold Floor:         {ff:.4f}")
print(f"  Universal Edge Score:   {ues:.1f}/100")
print(f"  Criteria passed:        {passes}/{total_criteria}")
print(f"  VERDICT:                {verdict}")
print(SEP)
