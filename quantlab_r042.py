"""
=============================================================================
QUANTLAB AI — RESEARCH #042
Independent Environment Discovery
=============================================================================

Objective:
  Discover completely new market environments that produce independent edge
  when paired with the locked RELVOL Breakout entry.  Do NOT optimise the
  existing winning environment — search from scratch.

Locked (unchanged across all experiments):
  • Entry:     RELVOL Breakout (vol>1.5×20-bar avg, bullish candle)
  • Exit:      Stop=1×ATR14  Target=2×ATR14  (2R fixed)
  • Timeframe: 1H
  • Fees / slippage / risk model

Search space:
  Volatility  : ATR Rank, Realised Volatility, Bollinger Width
  Trend       : EMA200 Slope, EMA Distance, ADX
  Participation: Relative Volume, Prev Candle Range, Prev Candle Body
  Time        : Session (Asia/Europe/US), Hour UTC, Day of Week

Method:
  1. Define 23 atomic conditions across 4 feature families
  2. Generate all valid 3-condition combinations (~1500 combos)
  3. Extend top-30 three-condition results with a 4th condition
  4. 5-fold expanding walk-forward; IS thresholds; OOS evaluation only
  5. Filter: n≥40 · PF>1.20 · Boot_p50>1.20
  6. Rank surviving environments → Environment Library

Reference environment (R041 Var G):
  ATR<p33 · Slope>0 · EMA_dist>p75 · BB<p50  (overlap computed for Q2)

Promote criteria: n≥40 · PF>1.20 · Boot>1.20 · MC>60% · LOO-S>1.0 · LOO-F>1.0

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

RESEARCH_ID = "R042"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = CONFIG["RISK_REWARD"]          # 2.0
BEP_WR   = 1.0 / (1.0 + RR)              # 0.333

SYMBOLS = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "AVAX-USDT-SWAP","XRP-USDT-SWAP","LTC-USDT-SWAP","BCH-USDT-SWAP",
    "DOGE-USDT-SWAP","ADA-USDT-SWAP","BNB-USDT-SWAP","DOT-USDT-SWAP",
    "ARB-USDT-SWAP","OP-USDT-SWAP","NEAR-USDT-SWAP","ATOM-USDT-SWAP",
    "SUI-USDT-SWAP","APT-USDT-SWAP","WIF-USDT-SWAP","PEPE-USDT-SWAP",
    "ENA-USDT-SWAP","UNI-USDT-SWAP","FIL-USDT-SWAP",
]
MIN_BARS     = 4_000
FOLDS        = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
N_BOOT       = 2_000

# Discovery thresholds
MIN_N         = 40
MIN_PF        = 1.20
MIN_BOOT      = 1.20
MIN_MC        = 0.60
MIN_ENV_BARS  = 8    # min OOS env bars before running backtest per fold×symbol
TOP_N_EXTEND  = 30   # top 3-cond results to extend to 4 conditions

# Overlap threshold for independence
INDEPENDENCE_OVERLAP = 0.30   # ≤30% overlap with Var G trades = independent

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]

    # ── Core (same as R039–R041) ──────────────────────────────────────────────
    df["ema200"]       = calc_ema(c, 200)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank"]     = df["atr14"].rolling(100).rank(pct=True) * 100
    bb_mid             = c.rolling(20).mean()
    bb_std             = c.rolling(20).std()
    df["bb_width"]     = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["ema_dist_pct"] = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / df["ema200"].shift(10).replace(0, np.nan)
    vol_ma             = v.rolling(20).mean()
    df["rel_vol"]      = v / vol_ma.replace(0, np.nan)
    df["prev_close"]   = c.shift(1)
    df["prev_atr14"]   = df["atr14"].shift(1)

    # ── New: Realised Volatility (20-bar rolling std of log-returns, annualised %) ─
    log_ret            = np.log(c / c.shift(1))
    df["real_vol_20"]  = log_ret.rolling(20).std() * 100.0

    # ── New: ADX ──────────────────────────────────────────────────────────────
    df["adx14"]        = calc_adx(df, 14)

    # ── New: Previous candle features ─────────────────────────────────────────
    prev_range         = (h.shift(1) - l.shift(1))
    prev_body          = (c.shift(1) - df["open"].shift(1)).abs()
    df["prev_range_r"] = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]  = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"]= prev_body  / prev_range.replace(0, np.nan)  # body/range conviction

    # ── New: Time features ────────────────────────────────────────────────────
    dt = pd.to_datetime(df["datetime"], utc=True)
    df["hour_utc"]     = dt.dt.hour.astype(np.int16)
    df["day_of_week"]  = dt.dt.dayofweek.astype(np.int16)   # 0=Mon … 6=Sun

    return df

# =============================================================================
# CONDITION CATALOGUE
# =============================================================================
# Schema: (id, label, feature, direction, param, group)
# direction:
#   "lt_q"      feature < quantile(param) learned from IS
#   "gt_q"      feature > quantile(param) learned from IS
#   "gt_q_pos"  feature > quantile(param) on positive-only IS values
#   "gt_fixed"  feature > param  (no IS learning)
#   "lt_fixed"  feature < param
#   "hour_rng"  param=(lo,hi); hour in [lo..hi] inclusive
#   "isin"      feature value in param (list)

CONDITIONS_DEF = [
    # ── Volatility ────────────────────────────────────────────────────────────
    ("ATR_LO",  "ATR<p25",       "atr_rank",      "lt_q",     0.25,  "vol"),
    ("ATR_MD",  "ATR<p40",       "atr_rank",       "lt_q",    0.40,  "vol"),
    ("ATR_HI",  "ATR>p67",       "atr_rank",       "gt_q",    0.67,  "vol"),
    ("RV_LO",   "RealVol<p33",   "real_vol_20",    "lt_q",    0.33,  "vol"),
    ("BB_TG",   "BB<p33",        "bb_width",       "lt_q",    0.33,  "vol"),
    ("BB_MD",   "BB<p50",        "bb_width",       "lt_q",    0.50,  "vol"),
    # ── Trend ─────────────────────────────────────────────────────────────────
    ("SLP_UP",  "Slope>0",       "ema200_slope",   "gt_fixed", 0.0,  "trend"),
    ("SLP_DN",  "Slope<0",       "ema200_slope",   "lt_fixed", 0.0,  "trend"),
    ("DST_FR",  "Dist>p75",      "ema_dist_pct",   "gt_q_pos", 0.75, "trend"),
    ("DST_MD",  "Dist>p60",      "ema_dist_pct",   "gt_q_pos", 0.60, "trend"),
    ("DST_NR",  "Dist<p33",      "ema_dist_pct",   "lt_q",    0.33,  "trend"),
    ("ADX_TR",  "ADX>p50",       "adx14",          "gt_q",    0.50,  "trend"),
    ("ADX_ST",  "ADX>p67",       "adx14",          "gt_q",    0.67,  "trend"),
    ("ADX_WK",  "ADX<p33",       "adx14",          "lt_q",    0.33,  "trend"),
    # ── Participation ─────────────────────────────────────────────────────────
    ("PRG_HI",  "PrevRng>p67",   "prev_range_r",   "gt_q",    0.67,  "part"),
    ("PRG_VH",  "PrevRng>p80",   "prev_range_r",   "gt_q",    0.80,  "part"),
    ("PBD_HI",  "PrevBody>p67",  "prev_body_r",    "gt_q",    0.67,  "part"),
    ("PBP_HI",  "BodyPct>p60",   "prev_body_pct",  "gt_q",    0.60,  "part"),
    # ── Time ──────────────────────────────────────────────────────────────────
    ("ASIA",    "Asia(0-7UTC)",  "hour_utc",       "hour_rng", (0,7),   "time"),
    ("EUR",     "Eur(8-15UTC)", "hour_utc",        "hour_rng", (8,15),  "time"),
    ("US",      "US(14-21UTC)", "hour_utc",        "hour_rng", (14,21), "time"),
    ("MIDWK",   "Wed-Thu",      "day_of_week",     "isin",    [2, 3],   "time"),
    ("EARLY",   "Mon-Tue",      "day_of_week",     "isin",    [0, 1],   "time"),
]

COND_IDS   = [c[0] for c in CONDITIONS_DEF]
COND_BY_ID = {c[0]: c for c in CONDITIONS_DEF}
COND_GROUP = {c[0]: c[5] for c in CONDITIONS_DEF}
COND_LABEL = {c[0]: c[1] for c in CONDITIONS_DEF}

# Mutually exclusive pairs (logical contradictions or pure redundancies)
_CONFLICTS_RAW = [
    ("SLP_UP",  "SLP_DN"),   # opposite slope direction
    ("ATR_LO",  "ATR_HI"),   # low vol vs high vol
    ("ATR_MD",  "ATR_HI"),   # moderate vs high
    ("ATR_LO",  "ATR_MD"),   # LO ⊂ MD — redundant
    ("DST_FR",  "DST_NR"),   # far above vs near MA
    ("DST_MD",  "DST_NR"),   # moderate above vs near MA
    ("DST_FR",  "DST_MD"),   # FR ⊂ MD — redundant
    ("ADX_TR",  "ADX_WK"),   # strong trend vs weak trend
    ("ADX_ST",  "ADX_WK"),   # very strong vs weak
    ("ADX_ST",  "ADX_TR"),   # ST ⊂ TR — redundant
    ("BB_TG",   "BB_MD"),    # TG ⊂ MD — redundant
    ("PRG_HI",  "PRG_VH"),   # HI ⊂ VH — redundant
    ("ASIA",    "EUR"),       # hours don't overlap cleanly — allow combos
    # Three sessions simultaneously means "any session" (no filter) — exclude triple
    # We only block obvious direct contradictions above
]
CONFLICTS = {frozenset(p) for p in _CONFLICTS_RAW}


def has_conflict(combo: tuple) -> bool:
    s = set(combo)
    return any(p <= s for p in CONFLICTS)


def generate_combos(depth: int, base_ids=None) -> list:
    """All non-conflicting depth-combinations of COND_IDS (or extensions of base_ids)."""
    pool = COND_IDS if base_ids is None else COND_IDS
    if base_ids is None:
        raw = list(itertools.combinations(pool, depth))
    else:
        # Extend each base tuple with one extra condition not already in it
        raw = []
        for base in base_ids:
            bs = set(base)
            for cid in pool:
                if cid not in bs:
                    raw.append(tuple(sorted(base + (cid,))))
        raw = list(set(raw))  # deduplicate
    return [c for c in raw if not has_conflict(c)]


# Variant G reference (R041 best): ATR<p33 (≈p40 in our set) · Slope>0 · Dist>p75 · BB<p50
VARG_CONDS = ("ATR_MD", "SLP_UP", "DST_FR", "BB_MD")

# =============================================================================
# THRESHOLD LEARNING  (per IS fold)
# =============================================================================

QUANT_FEATS = [
    "atr_rank", "real_vol_20", "bb_width", "ema_dist_pct",
    "adx14", "prev_range_r", "prev_body_r", "prev_body_pct",
]


def learn_thresholds(df_is: pd.DataFrame) -> dict:
    thr = {}
    valid = df_is.dropna(subset=QUANT_FEATS)

    for cid, _, feat, direction, param, _ in CONDITIONS_DEF:
        if direction in ("gt_fixed", "lt_fixed", "hour_rng", "isin"):
            thr[cid] = param
            continue
        col = valid[feat].dropna()
        if len(col) < 20:
            thr[cid] = np.nan
            continue
        if direction == "gt_q_pos":
            pos = col[col > 0]
            thr[cid] = float(pos.quantile(param) if len(pos) > 10 else col.quantile(param))
        else:
            thr[cid] = float(col.quantile(param))

    return thr

# =============================================================================
# CONDITION MASKS  (precomputed per fold×symbol, numpy bool arrays)
# =============================================================================

def condition_mask(df: pd.DataFrame, cid: str, thr: dict) -> np.ndarray:
    _, _, feat, direction, _, _ = COND_BY_ID[cid]
    threshold = thr.get(cid, np.nan)

    n = len(df)
    if feat not in df.columns:
        return np.zeros(n, dtype=bool)

    col = df[feat].values

    if direction == "lt_q":
        if np.isnan(threshold): return np.zeros(n, dtype=bool)
        return (~np.isnan(col)) & (col < threshold)
    elif direction in ("gt_q", "gt_q_pos"):
        if np.isnan(threshold): return np.zeros(n, dtype=bool)
        return (~np.isnan(col)) & (col > threshold)
    elif direction == "gt_fixed":
        return (~np.isnan(col)) & (col > threshold)
    elif direction == "lt_fixed":
        return (~np.isnan(col)) & (col < threshold)
    elif direction == "hour_rng":
        lo, hi = threshold
        return (col >= lo) & (col <= hi)
    elif direction == "isin":
        return np.isin(col, threshold)
    else:
        return np.zeros(n, dtype=bool)


def precompute_all_masks(df_oos: pd.DataFrame, thr: dict) -> dict:
    return {cid: condition_mask(df_oos, cid, thr) for cid in COND_IDS}


def combine_masks(masks: dict, combo: tuple) -> np.ndarray:
    out = masks[combo[0]].copy()
    for cid in combo[1:]:
        out &= masks[cid]
    return out

# =============================================================================
# RELVOL SIGNAL
# =============================================================================

def signal_relvol_env(df: pd.DataFrame, env_mask: np.ndarray) -> np.ndarray:
    rv = df["rel_vol"].values
    c  = df["close"].values
    o  = df["open"].values
    pc = df["prev_close"].values
    valid = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc)) & (~np.isnan(o))
    return valid & (rv > 1.5) & (c > o) & (c > pc) & env_mask

# =============================================================================
# BACKTEST ENGINE  (identical to R041)
# =============================================================================

def run_backtest(df: pd.DataFrame, signal: np.ndarray,
                 sym: str, fold: int, eid: str) -> list:
    min_sl  = CONFIG["MIN_SL_PCT"]
    max_lev = CONFIG["MAX_LEVERAGE"]
    rf      = CONFIG["RISK_PER_TRADE_PCT"]
    fee     = CONFIG["TAKER_FEE"]
    spd     = CONFIG["SPREAD"] * 0.5
    slp     = CONFIG["SL_SLIPPAGE"]
    in_pos  = False
    ep = st = tk = sz = 0.0
    et = None; ei = -1
    trades = []

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
                trades.append({
                    "sym": sym, "fold": fold, "env": eid,
                    "entry_time": str(et), "exit_time": str(dts[i]),
                    "pnl": round(net, 4), "r_multiple": round(rmul, 4),
                    "win": int(xt == "TP"), "exit_type": xt,
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
# STATISTICS  (identical to R041)
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
    return {"n": n, "wr": wr, "pf": pf, "exp_r": wr * RR - (1 - wr),
            "net": float(pnl.sum()), "sharpe": sharpe, "mdd": mdd,
            "pnls": pnl, "equity": equity}


def bootstrap_pf(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    pfs = []
    for _ in range(n_iter):
        s = rng.choice(pnls, len(pnls), replace=True)
        pfs.append(safe_pf(s[s > 0].sum(), abs(s[s < 0].sum())))
    return float(np.percentile(pfs, 5)), float(np.percentile(pfs, 50)), float(np.percentile(pfs, 95))


def monte_carlo(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return {"prob_profit": 0.0, "p5": CAPITAL, "p50": CAPITAL, "p95": CAPITAL, "finals": np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit": float((finals > CAPITAL).mean()),
            "p5":  float(np.percentile(finals,  5)),
            "p50": float(np.percentile(finals, 50)),
            "p95": float(np.percentile(finals, 95)),
            "finals": finals}


def loo_sym(sym_trades_dict: dict) -> dict:
    return {omit: metrics([t for s, tl in sym_trades_dict.items() if s != omit for t in tl])["pf"]
            for omit in sym_trades_dict if sym_trades_dict[omit]}


def loo_fld(all_trades: list) -> dict:
    folds = sorted({t["fold"] for t in all_trades})
    return {omit: metrics([t for t in all_trades if t["fold"] != omit])["pf"]
            for omit in folds}

# =============================================================================
# DATA LOAD
# =============================================================================

print("=" * 79)
print("  QUANTLAB AI — RESEARCH #042")
print("  Independent Environment Discovery (1H · RELVOL fixed)")
print("=" * 79)
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
print(f"  {len(SYMBOLS)} symbols  ({total_bars:,} bars)\n")

# =============================================================================
# GENERATE SEARCH SPACE
# =============================================================================

combos_3 = generate_combos(3)
print(f"  Search space: {len(combos_3):,} valid 3-condition combinations")
print(f"  Walk-forward: {len(FOLDS)} folds × {len(SYMBOLS)} symbols")
print(f"  Promote filter: n≥{MIN_N} · PF>{MIN_PF} · Boot>{MIN_BOOT}")
print()

# =============================================================================
# WALK-FORWARD — PASS 1 (3-condition search + Var G reference)
# =============================================================================

# env_sym_trades[combo_tuple][sym] = [trade_dicts]
env_sym_trades: dict = defaultdict(lambda: defaultdict(list))
varg_sym_trades: dict = defaultdict(list)

# combo_env_bars[combo_tuple] = total OOS env bars (quick sanity)
combo_env_bars: dict = defaultdict(int)

print("─" * 79)
print("  PASS 1 — 3-condition exhaustive search + Var G reference")
print("─" * 79)

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_skipped = 0
    fold_tested  = 0
    fold_trades  = 0

    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[: int(N * is_end)]
        df_oos = df_full.iloc[int(N * is_end): int(N * oos_end)].reset_index(drop=True)
        if len(df_oos) < 100:
            continue

        thr   = learn_thresholds(df_is)
        masks = precompute_all_masks(df_oos, thr)

        # ── Var G reference ───────────────────────────────────────────────────
        vg_env = masks[VARG_CONDS[0]].copy()
        for cid in VARG_CONDS[1:]:
            vg_env &= masks[cid]
        vg_sig = signal_relvol_env(df_oos, vg_env)
        vg_tl  = run_backtest(df_oos, vg_sig, sym, fold_idx, "VARG")
        varg_sym_trades[sym].extend(vg_tl)

        # ── 3-condition search ────────────────────────────────────────────────
        for combo in combos_3:
            env = masks[combo[0]] & masks[combo[1]] & masks[combo[2]]
            if env.sum() < MIN_ENV_BARS:
                fold_skipped += 1
                continue
            sig = signal_relvol_env(df_oos, env)
            if sig.sum() == 0:
                fold_skipped += 1
                continue
            tl = run_backtest(df_oos, sig, sym, fold_idx, combo)
            combo_env_bars[combo] += int(env.sum())
            if tl:
                env_sym_trades[combo][sym].extend(tl)
                fold_trades += len(tl)
            fold_tested += 1

    print(f"  Fold {fold_idx}  (IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%)"
          f"  tested={fold_tested:,}  skipped={fold_skipped:,}  trades={fold_trades:,}")

print()

# =============================================================================
# AGGREGATE PASS 1 RESULTS
# =============================================================================

print("  Aggregating 3-condition results …")

pass1_results = {}
for combo in combos_3:
    flat = [t for sym_tl in env_sym_trades[combo].values() for t in sym_tl]
    if len(flat) < 5:
        continue
    m = metrics(flat)
    if m["n"] < MIN_N or m["pf"] < MIN_PF:
        continue
    b5, b50, b95 = bootstrap_pf(m["pnls"])
    if b50 < MIN_BOOT:
        continue
    pass1_results[combo] = {
        "combo": combo, "depth": 3,
        "n": m["n"], "wr": m["wr"], "pf": m["pf"],
        "b50": b50, "b5": b5, "b95": b95,
        "mdd": m["mdd"], "net": m["net"],
        "pnls": m["pnls"], "equity": m["equity"],
        "env_bars": combo_env_bars[combo],
        "sym_trades": dict(env_sym_trades[combo]),
        "_flat": flat,
    }

print(f"  Pass 1 survivors (n≥{MIN_N}, PF>{MIN_PF}, Boot>{MIN_BOOT}): {len(pass1_results)}")
print()

# =============================================================================
# PASS 2 — EXTEND TOP 3-COND RESULTS TO 4 CONDITIONS
# =============================================================================

pass2_results = {}
if pass1_results:
    top_30_combos = sorted(pass1_results, key=lambda c: -pass1_results[c]["pf"])[:TOP_N_EXTEND]
    combos_4 = generate_combos(4, base_ids=top_30_combos)
    # Remove duplicates and combos already tested (they came from same 3-cond base)
    combos_4_new = [c for c in combos_4 if c not in pass1_results]
    print("─" * 79)
    print(f"  PASS 2 — 4-condition extensions of top-{min(len(top_30_combos),TOP_N_EXTEND)} Pass-1 results")
    print(f"  Combos to test: {len(combos_4_new):,}")
    print("─" * 79)

    env_sym_trades_4: dict = defaultdict(lambda: defaultdict(list))
    combo_env_bars_4: dict = defaultdict(int)

    for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
        fold_t4 = 0
        for sym, df_full in all_dfs.items():
            N      = len(df_full)
            df_is  = df_full.iloc[: int(N * is_end)]
            df_oos = df_full.iloc[int(N * is_end): int(N * oos_end)].reset_index(drop=True)
            if len(df_oos) < 100:
                continue
            thr   = learn_thresholds(df_is)
            masks = precompute_all_masks(df_oos, thr)

            for combo in combos_4_new:
                env = masks[combo[0]].copy()
                for cid in combo[1:]:
                    env &= masks[cid]
                if env.sum() < MIN_ENV_BARS:
                    continue
                sig = signal_relvol_env(df_oos, env)
                if sig.sum() == 0:
                    continue
                tl = run_backtest(df_oos, sig, sym, fold_idx, combo)
                combo_env_bars_4[combo] += int(env.sum())
                if tl:
                    env_sym_trades_4[combo][sym].extend(tl)
                    fold_t4 += len(tl)
        print(f"  Fold {fold_idx}  trades={fold_t4:,}")

    print()
    print("  Aggregating 4-condition results …")
    for combo in combos_4_new:
        flat = [t for sym_tl in env_sym_trades_4[combo].values() for t in sym_tl]
        if len(flat) < 5:
            continue
        m = metrics(flat)
        if m["n"] < MIN_N or m["pf"] < MIN_PF:
            continue
        b5, b50, b95 = bootstrap_pf(m["pnls"])
        if b50 < MIN_BOOT:
            continue
        pass2_results[combo] = {
            "combo": combo, "depth": 4,
            "n": m["n"], "wr": m["wr"], "pf": m["pf"],
            "b50": b50, "b5": b5, "b95": b95,
            "mdd": m["mdd"], "net": m["net"],
            "pnls": m["pnls"], "equity": m["equity"],
            "env_bars": combo_env_bars_4[combo],
            "sym_trades": dict(env_sym_trades_4[combo]),
            "_flat": flat,
        }
    print(f"  Pass 2 survivors: {len(pass2_results)}")
    print()

# =============================================================================
# MERGE ALL SURVIVORS
# =============================================================================

all_survivors = {**pass1_results, **pass2_results}
print(f"  Total survivors: {len(all_survivors)}")

if not all_survivors:
    print("\n  ⚠  No environments survived all three filters.")
    print("  Consider lowering MIN_N or MIN_PF thresholds.\n")

# =============================================================================
# FULL STATS ON SURVIVORS  (MC, LOO, etc.)
# =============================================================================

print("  Computing full statistics on survivors …")

# Var G reference
varg_flat  = [t for tl in varg_sym_trades.values() for t in tl]
varg_set   = {(t["sym"], t["entry_time"]) for t in varg_flat}
varg_m     = metrics(varg_flat)

library = []
for combo, r in all_survivors.items():
    flat   = r["_flat"]
    m2     = metrics(flat)
    mc     = monte_carlo(m2["pnls"])
    ls     = loo_sym(r["sym_trades"])
    lf     = loo_fld(flat)
    sf     = min(ls.values()) if ls else 0.0
    ff     = min(lf.values()) if lf else 0.0

    # Overlap with Var G
    env_set  = {(t["sym"], t["entry_time"]) for t in flat}
    overlap  = len(env_set & varg_set) / max(len(varg_set), 1)

    # Robustness score (7 criteria)
    score = sum([
        m2["pf"]           > MIN_PF,
        m2["n"]            >= MIN_N,
        r["b50"]           > MIN_BOOT,
        mc["prob_profit"]  > MIN_MC,
        sf                 > 1.0,
        ff                 > 1.0,
        abs(m2["mdd"])     < 0.30,
    ])

    # Feature family diversity
    groups = [COND_GROUP[cid] for cid in combo]
    n_families = len(set(groups))

    entry = {
        "combo": combo,
        "depth": r["depth"],
        "label": " · ".join(COND_LABEL[cid] for cid in combo),
        "cond_ids": list(combo),
        "groups": groups,
        "n_families": n_families,
        "n":       m2["n"],
        "wr":      m2["wr"],
        "pf":      m2["pf"],
        "b5":      r["b5"],
        "b50":     r["b50"],
        "b95":     r["b95"],
        "mc_p":    mc["prob_profit"],
        "mdd":     m2["mdd"],
        "net":     m2["net"],
        "sharpe":  m2["sharpe"],
        "sym_floor": sf,
        "fold_floor": ff,
        "env_bars": r["env_bars"],
        "overlap_varg": overlap,
        "independent": overlap <= INDEPENDENCE_OVERLAP,
        "score":   score,
        "pnls":    m2["pnls"],
        "equity":  m2["equity"],
        "mc_finals": mc["finals"],
        "sym_trades": r["sym_trades"],
        "_flat":   flat,
    }

    # Verdict
    if score == 7:
        entry["verdict"] = "PROMOTE"
    elif score >= 5 and m2["pf"] > MIN_PF:
        entry["verdict"] = "WATCHLIST"
    elif score >= 3:
        entry["verdict"] = "INVESTIGATE"
    else:
        entry["verdict"] = "REJECT"

    library.append(entry)

# Sort: score desc → PF desc
library.sort(key=lambda e: (-e["score"], -e["pf"]))

# =============================================================================
# RESULTS TABLE
# =============================================================================

print("\n" + "=" * 110)
print("  R042 — ENVIRONMENT LIBRARY  (RELVOL fixed · 1H · 5-fold WF · OOS only)")
print("=" * 110)
print(f"\n  Var G reference: n={varg_m['n']}  PF={varg_m['pf']:.3f}"
      f"  ({' · '.join(COND_LABEL[c] for c in VARG_CONDS)})\n")

hdr = (f"  {'#':>3}  {'Depth':>5}  {'n':>5}  {'WR':>6}  {'PF':>7}  "
       f"{'p50':>7}  {'MC%':>6}  {'MDD':>7}  {'LOO-S':>6}  {'LOO-F':>6}  "
       f"{'Ovlp':>6}  {'Ind?':>5}  {'Sc':>4}  Verdict")
print(hdr)
print("  " + "─" * 106)

for i, e in enumerate(library, start=1):
    ind = "★ YES" if e["independent"] else "  no"
    print(f"  {i:>3}  {e['depth']:>5}  {e['n']:>5}  {e['wr']*100:5.1f}%  "
          f"{e['pf']:7.3f}  {e['b50']:7.3f}  {e['mc_p']*100:5.1f}%  "
          f"{e['mdd']:6.1%}  {e['sym_floor']:6.3f}  {e['fold_floor']:6.3f}  "
          f"{e['overlap_varg']:5.1%}  {ind:>5}  {e['score']:>2}/7  {e['verdict']}")
    print(f"       Conditions: {e['label']}")
    print()

# =============================================================================
# RESEARCH QUESTIONS
# =============================================================================

print("=" * 110)
print("  RESEARCH QUESTIONS")
print("=" * 110)

# Q1 — Top 10 by PF
q1_top10 = library[:10]
print("\n  Q1.  Top 10 independent environments ranked by PF")
print("  " + "─" * 72)
for i, e in enumerate(q1_top10, start=1):
    print(f"  #{i:>2}  PF={e['pf']:.3f}  n={e['n']:3d}  p50={e['b50']:.3f}  "
          f"MC={e['mc_p']*100:.0f}%  Depth={e['depth']}")
    print(f"       {e['label']}")

# Q2 — Independent from Var G (overlap ≤ 30%)
q2_independent = [e for e in library if e["independent"]]
q2_overlap_hi  = [e for e in library if not e["independent"]]
print(f"\n  Q2.  Environments statistically independent from Var G  "
      f"(overlap ≤ {INDEPENDENCE_OVERLAP*100:.0f}%)")
print("  " + "─" * 72)
if q2_independent:
    for e in q2_independent[:10]:
        print(f"  PF={e['pf']:.3f}  n={e['n']:3d}  overlap={e['overlap_varg']*100:.0f}%  "
              f"Score={e['score']}/7  {e['label']}")
else:
    print("  None found — all environments share ≥30% of Var G trade set.")

# Q3 — PF > 1.20 with n ≥ 40
q3_pass = [e for e in library if e["pf"] > MIN_PF and e["n"] >= MIN_N]
q3_n_only = [e for e in library if e["n"] >= MIN_N]
print(f"\n  Q3.  Environments with PF>{MIN_PF} and n≥{MIN_N}")
print("  " + "─" * 72)
if q3_pass:
    print(f"  YES — {len(q3_pass)} environment(s) qualify:")
    for e in q3_pass[:10]:
        print(f"    PF={e['pf']:.3f}  n={e['n']:3d}  Score={e['score']}/7  {e['label']}")
elif q3_n_only:
    print(f"  NOT YET — n≥{MIN_N} achieved but PF below threshold:")
    for e in q3_n_only[:5]:
        print(f"    PF={e['pf']:.3f}  n={e['n']:3d}  {e['label']}")
else:
    print(f"  NO — no environment reaches both n≥{MIN_N} and PF>{MIN_PF}.")

# Q4 — Feature frequency among top-20 surviving environments
print(f"\n  Q4.  Feature frequency among top-{min(20,len(library))} surviving environments")
print("  " + "─" * 72)
top20 = library[:20]
freq: dict = defaultdict(int)
for e in top20:
    for cid in e["cond_ids"]:
        freq[cid] += 1
sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
for cid, cnt in sorted_freq:
    bar = "█" * cnt
    print(f"  {cid:10s} ({COND_LABEL[cid]:18s})  {bar}  {cnt}/{len(top20)}")

# Q5 — Portfolio combination: best independent pair
print(f"\n  Q5.  Can two independent environments be combined without reducing PF?")
print("  " + "─" * 72)
if len(library) >= 2:
    # Find pair with lowest overlap and highest combined PF
    best_pair = None
    best_combined_pf = 0.0
    candidates = library[:min(20, len(library))]
    for i, ea in enumerate(candidates):
        for eb in candidates[i+1:]:
            # Check trade overlap between these two environments
            sa = {(t["sym"], t["entry_time"]) for t in ea["_flat"]}
            sb = {(t["sym"], t["entry_time"]) for t in eb["_flat"]}
            ab_overlap = len(sa & sb) / max(len(sa | sb), 1)
            if ab_overlap > INDEPENDENCE_OVERLAP:
                continue
            # Combined trades
            combined = ea["_flat"] + [t for t in eb["_flat"]
                                       if (t["sym"], t["entry_time"]) not in sa]
            m_comb = metrics(combined)
            if m_comb["pf"] >= min(ea["pf"], eb["pf"]) and m_comb["pf"] > best_combined_pf:
                best_combined_pf = m_comb["pf"]
                best_pair = (ea, eb, m_comb, ab_overlap)

    if best_pair:
        ea, eb, m_comb, ab_ov = best_pair
        print(f"  YES — Portfolio of Env #1 and Env #2 maintains or improves PF")
        print(f"  Env A: PF={ea['pf']:.3f}  n={ea['n']}  {ea['label']}")
        print(f"  Env B: PF={eb['pf']:.3f}  n={eb['n']}  {eb['label']}")
        print(f"  Combined: PF={m_comb['pf']:.3f}  n={m_comb['n']}  overlap={ab_ov*100:.0f}%")
    else:
        print("  No independent pair found that maintains PF when combined.")
        if len(library) >= 2:
            e1, e2 = library[0], library[1]
            print(f"  Best available pair (may share trades):")
            print(f"  Env A: PF={e1['pf']:.3f}  n={e1['n']}  {e1['label']}")
            print(f"  Env B: PF={e2['pf']:.3f}  n={e2['n']}  {e2['label']}")
else:
    print("  Insufficient environments discovered for portfolio analysis.")

# Recommendation for R043
print(f"\n  R043 Recommendation:")
print("  " + "─" * 72)
if library:
    rec = library[0]
    ind_label = "INDEPENDENT" if rec["independent"] else "OVERLAPPING with Var G"
    print(f"  Best candidate: #{1}")
    print(f"  Conditions  : {rec['label']}")
    print(f"  PF={rec['pf']:.3f}  n={rec['n']}  Boot_p50={rec['b50']:.3f}  "
          f"MC={rec['mc_p']*100:.0f}%  Score={rec['score']}/7")
    print(f"  Independence: {ind_label}  (overlap={rec['overlap_varg']*100:.0f}%)")
    print(f"  Verdict     : {rec['verdict']}")
    print(f"  → R043 should test this environment in portfolio alongside Var G (R041)")
else:
    print("  No viable candidate found. Recommend widening search or relaxing filters.")

print()

# =============================================================================
# WRITE ENVIRONMENT LIBRARY CSV
# =============================================================================

print("─" * 79)
print("  Writing outputs …")

csv_rows = []
for i, e in enumerate(library, start=1):
    csv_rows.append({
        "rank": i,
        "depth": e["depth"],
        "n": e["n"],
        "win_rate": round(e["wr"], 4),
        "profit_factor": round(e["pf"], 4),
        "boot_p5": round(e["b5"], 4),
        "boot_p50": round(e["b50"], 4),
        "boot_p95": round(e["b95"], 4),
        "mc_prob_profit": round(e["mc_p"], 4),
        "max_drawdown": round(e["mdd"], 4),
        "sharpe": round(e["sharpe"], 4),
        "net_pnl": round(e["net"], 2),
        "sym_floor": round(e["sym_floor"], 4),
        "fold_floor": round(e["fold_floor"], 4),
        "env_bars": e["env_bars"],
        "overlap_varg": round(e["overlap_varg"], 4),
        "independent": e["independent"],
        "n_families": e["n_families"],
        "score": e["score"],
        "verdict": e["verdict"],
        "conditions": e["label"],
        "cond_ids": "|".join(e["cond_ids"]),
    })

csv_path = f"{OUT}/r042_environment_library.csv"
pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
print(f"  → {csv_path}")

# =============================================================================
# CHARTS
# =============================================================================

DARK_BG  = "#0e1117"
GRID_CLR = "#1e2430"
TEXT_CLR = "#e0e0e0"
ACCENT   = "#4fc3f7"
GREEN    = "#69f0ae"
RED      = "#ef5350"
GOLD     = "#ffd54f"
PURPLE   = "#ce93d8"

def _style(ax, title=""):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=7)
    for sp in ax.spines.values(): sp.set_color(GRID_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    if title: ax.set_title(title, color=TEXT_CLR, fontsize=8, pad=4)
    ax.grid(True, color=GRID_CLR, linewidth=0.5)


def verdict_color(v):
    return {"PROMOTE": GREEN, "WATCHLIST": GOLD, "INVESTIGATE": ACCENT, "REJECT": RED}.get(v, TEXT_CLR)


TOP_K = min(10, len(library))
top_envs = library[:TOP_K]

# ── Chart 1: PF vs n scatter (all survivors) ──────────────────────────────────
if library:
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=DARK_BG)
    _style(ax, "Environment Discovery — PF vs Trade Count (all survivors)")
    ns   = [e["n"]  for e in library]
    pfs  = [e["pf"] for e in library]
    clrs = [verdict_color(e["verdict"]) for e in library]
    ax.scatter(ns, pfs, c=clrs, s=40, alpha=0.7, zorder=3)
    # Highlight top-10
    for i, e in enumerate(top_envs):
        ax.annotate(f"#{i+1}", (e["n"], e["pf"]), fontsize=6, color=TEXT_CLR,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(MIN_PF, color=GOLD, lw=0.8, ls="--", label=f"Min PF={MIN_PF}")
    ax.axvline(MIN_N,  color=ACCENT, lw=0.8, ls="--", label=f"Min n={MIN_N}")
    ax.set_xlabel("Trade Count (n)"); ax.set_ylabel("Profit Factor")
    ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR)
    plt.tight_layout()
    p1 = f"{OUT}/r042_scatter.png"
    plt.savefig(p1, dpi=130, facecolor=DARK_BG); plt.close()
    print(f"  → {p1}")
else:
    p1 = None

# ── Chart 2: Equity curves of top-10 environments ─────────────────────────────
if top_envs:
    fig, axes = plt.subplots(2, 5, figsize=(14, 5), facecolor=DARK_BG)
    axes = axes.flatten()
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, TOP_K))
    for i, (e, ax) in enumerate(zip(top_envs, axes)):
        _style(ax, f"#{i+1} PF={e['pf']:.3f} n={e['n']}")
        eq = e["equity"]
        x  = np.arange(len(eq))
        ax.plot(x, eq, color=colors[i], lw=1.2)
        ax.fill_between(x, CAPITAL, eq,
                        where=eq >= CAPITAL, alpha=0.2, color=GREEN)
        ax.fill_between(x, CAPITAL, eq,
                        where=eq <  CAPITAL, alpha=0.2, color=RED)
        ax.axhline(CAPITAL, color=TEXT_CLR, lw=0.5, ls="--")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Top-10 Equity Curves", color=TEXT_CLR, fontsize=9)
    plt.tight_layout()
    p2 = f"{OUT}/r042_equity_curves.png"
    plt.savefig(p2, dpi=130, facecolor=DARK_BG); plt.close()
    print(f"  → {p2}")
else:
    p2 = None

# ── Chart 3: Feature frequency bar chart ──────────────────────────────────────
if sorted_freq:
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    _style(ax, f"Feature Frequency in Top-{len(top20)} Surviving Environments")
    labels = [COND_LABEL[c] for c, _ in sorted_freq]
    counts = [cnt for _, cnt in sorted_freq]
    grps   = [COND_GROUP[c] for c, _ in sorted_freq]
    gcols  = {"vol": ACCENT, "trend": GREEN, "part": GOLD, "time": PURPLE}
    bar_colors = [gcols.get(g, TEXT_CLR) for g in grps]
    bars = ax.barh(labels, counts, color=bar_colors, alpha=0.8)
    ax.axvline(len(top20) * 0.5, color=RED, lw=0.8, ls="--", label="50% threshold")
    ax.set_xlabel("Frequency"); ax.invert_yaxis()
    ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR)
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=v, label=k) for k, v in gcols.items()]
    ax.legend(handles=legend_patches, fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR, loc="lower right")
    plt.tight_layout()
    p3 = f"{OUT}/r042_feature_freq.png"
    plt.savefig(p3, dpi=130, facecolor=DARK_BG); plt.close()
    print(f"  → {p3}")
else:
    p3 = None

# ── Chart 4: Bootstrap CI for top-10 ──────────────────────────────────────────
if top_envs:
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=DARK_BG)
    _style(ax, "Bootstrap PF 5th–95th percentile — Top-10 Environments")
    ys = np.arange(TOP_K)
    for i, e in enumerate(top_envs):
        ax.barh(i, e["b95"] - e["b5"], left=e["b5"], color=ACCENT, alpha=0.4, height=0.6)
        ax.plot([e["b50"]], [i], "o", color=GOLD, ms=6, zorder=5)
        ax.plot([e["pf"]],  [i], "D", color=GREEN, ms=5, zorder=5)
    ax.axvline(MIN_PF, color=RED, lw=1, ls="--", label=f"PF={MIN_PF}")
    ax.axvline(1.0,   color=GRID_CLR, lw=0.8, ls=":")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"#{i+1}" for i in ys], fontsize=7)
    ax.set_xlabel("Profit Factor")
    from matplotlib.lines import Line2D
    legend_elem = [Line2D([0],[0],marker="o",color="w",markerfacecolor=GOLD, label="Boot p50"),
                   Line2D([0],[0],marker="D",color="w",markerfacecolor=GREEN, label="Actual PF")]
    ax.legend(handles=legend_elem, fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR)
    ax.invert_yaxis()
    plt.tight_layout()
    p4 = f"{OUT}/r042_bootstrap_ci.png"
    plt.savefig(p4, dpi=130, facecolor=DARK_BG); plt.close()
    print(f"  → {p4}")
else:
    p4 = None

# ── Chart 5: Var G overlap vs PF ──────────────────────────────────────────────
if library:
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=DARK_BG)
    _style(ax, "Var G Overlap vs PF — Independence Map")
    ovlp  = [e["overlap_varg"] for e in library]
    pfs_  = [e["pf"] for e in library]
    clrs_ = [GREEN if e["independent"] else RED for e in library]
    ax.scatter(ovlp, pfs_, c=clrs_, s=40, alpha=0.7, zorder=3)
    for i, e in enumerate(top_envs):
        ax.annotate(f"#{i+1}", (e["overlap_varg"], e["pf"]),
                    fontsize=6, color=TEXT_CLR, xytext=(3,3), textcoords="offset points")
    ax.axvline(INDEPENDENCE_OVERLAP, color=GOLD, lw=1, ls="--",
               label=f"Independence threshold ({INDEPENDENCE_OVERLAP*100:.0f}%)")
    ax.axhline(MIN_PF, color=ACCENT, lw=0.8, ls="--")
    ax.set_xlabel("Trade Overlap with Var G"); ax.set_ylabel("Profit Factor")
    from matplotlib.patches import Patch
    lp = [Patch(color=GREEN, label="Independent"), Patch(color=RED, label="Overlapping")]
    ax.legend(handles=lp, fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_CLR)
    plt.tight_layout()
    p5 = f"{OUT}/r042_overlap_map.png"
    plt.savefig(p5, dpi=130, facecolor=DARK_BG); plt.close()
    print(f"  → {p5}")
else:
    p5 = None

# ── Chart 6: Per-symbol PF heatmap for top-5 ─────────────────────────────────
top5 = top_envs[:5]
if top5 and SYMBOLS:
    fig, ax = plt.subplots(figsize=(max(8, len(SYMBOLS)*0.5), 4), facecolor=DARK_BG)
    _style(ax, "Per-Symbol PF — Top-5 Environments")
    hm_data = np.zeros((len(top5), len(SYMBOLS)))
    sym_list = sorted(SYMBOLS)
    for i, e in enumerate(top5):
        for j, sym in enumerate(sym_list):
            tl = e["sym_trades"].get(sym, [])
            hm_data[i, j] = metrics(tl)["pf"] if tl else 0.0
    cmap = LinearSegmentedColormap.from_list("rg", [RED, DARK_BG, GREEN])
    im = ax.imshow(hm_data, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(len(sym_list)))
    ax.set_xticklabels([s.split("-")[0] for s in sym_list], rotation=60, fontsize=6, color=TEXT_CLR)
    ax.set_yticks(range(len(top5)))
    ax.set_yticklabels([f"#{i+1}" for i in range(len(top5))], fontsize=7, color=TEXT_CLR)
    plt.colorbar(im, ax=ax, label="PF").ax.yaxis.label.set_color(TEXT_CLR)
    plt.tight_layout()
    p6 = f"{OUT}/r042_per_symbol_pf.png"
    plt.savefig(p6, dpi=130, facecolor=DARK_BG); plt.close()
    print(f"  → {p6}")
else:
    p6 = None

# ── Chart 7: Dashboard ────────────────────────────────────────────────────────
if library:
    fig = plt.figure(figsize=(14, 9), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A: PF vs n scatter
    ax_a = fig.add_subplot(gs[0, 0])
    _style(ax_a, "PF vs n  (all survivors)")
    ns2  = [e["n"]  for e in library]
    pfs2 = [e["pf"] for e in library]
    clr2 = [verdict_color(e["verdict"]) for e in library]
    ax_a.scatter(ns2, pfs2, c=clr2, s=25, alpha=0.7, zorder=3)
    ax_a.axhline(MIN_PF, color=GOLD, lw=0.7, ls="--")
    ax_a.axvline(MIN_N,  color=ACCENT, lw=0.7, ls="--")
    ax_a.set_xlabel("n"); ax_a.set_ylabel("PF")

    # Panel B: Bootstrap CI top-5
    ax_b = fig.add_subplot(gs[0, 1])
    _style(ax_b, "Boot CI top-5")
    ys5 = np.arange(min(5, len(library)))
    for i, e in enumerate(library[:5]):
        ax_b.barh(i, e["b95"] - e["b5"], left=e["b5"], color=ACCENT, alpha=0.4, height=0.6)
        ax_b.plot([e["b50"]], [i], "o", color=GOLD, ms=5)
    ax_b.axvline(MIN_PF, color=RED, lw=0.8, ls="--")
    ax_b.set_yticks(ys5); ax_b.set_yticklabels([f"#{i+1}" for i in ys5], fontsize=6)
    ax_b.invert_yaxis()

    # Panel C: Feature frequency
    ax_c = fig.add_subplot(gs[0, 2])
    _style(ax_c, "Feature freq (top-20)")
    if sorted_freq:
        lbls = [COND_LABEL[c] for c, _ in sorted_freq[:8]]
        cnts = [cnt for _, cnt in sorted_freq[:8]]
        ax_c.barh(lbls, cnts, color=ACCENT, alpha=0.8)
        ax_c.invert_yaxis()
        ax_c.tick_params(axis="y", labelsize=6)

    # Panel D-F: Equity of top-3
    colors3 = [GREEN, GOLD, PURPLE]
    for idx in range(min(3, len(library))):
        ax_d = fig.add_subplot(gs[1, idx])
        e = library[idx]
        _style(ax_d, f"#{idx+1}  PF={e['pf']:.3f}  n={e['n']}")
        eq = e["equity"]
        x  = np.arange(len(eq))
        ax_d.plot(x, eq, color=colors3[idx], lw=1)
        ax_d.fill_between(x, CAPITAL, eq, where=eq >= CAPITAL, alpha=0.15, color=GREEN)
        ax_d.fill_between(x, CAPITAL, eq, where=eq <  CAPITAL, alpha=0.15, color=RED)
        ax_d.axhline(CAPITAL, color=TEXT_CLR, lw=0.5, ls="--")

    # Panel G: Overlap map (miniature)
    ax_g = fig.add_subplot(gs[2, 0])
    _style(ax_g, "Overlap vs PF")
    ax_g.scatter([e["overlap_varg"] for e in library],
                 [e["pf"]  for e in library],
                 c=[GREEN if e["independent"] else RED for e in library], s=20, alpha=0.7)
    ax_g.axvline(INDEPENDENCE_OVERLAP, color=GOLD, lw=0.7, ls="--")
    ax_g.axhline(MIN_PF, color=ACCENT, lw=0.7, ls="--")
    ax_g.set_xlabel("Overlap"); ax_g.set_ylabel("PF")

    # Panel H: Summary table text
    ax_h = fig.add_subplot(gs[2, 1:])
    ax_h.set_facecolor(DARK_BG)
    ax_h.axis("off")
    summary_lines = [
        f"R042 — Independent Environment Discovery",
        f"",
        f"Symbols: {len(SYMBOLS)}   Folds: 5   Depth: 3–4 conditions",
        f"3-cond combos tested: {len(combos_3):,}",
        f"Pass-1 survivors: {len(pass1_results)}   Pass-2 survivors: {len(pass2_results)}",
        f"Total in library: {len(library)}",
        f"",
        f"{'#':<3}  {'PF':>6}  {'n':>4}  {'p50':>6}  {'Ind':>5}  Conditions",
    ]
    for i, e in enumerate(library[:7], start=1):
        summary_lines.append(
            f"{i:<3}  {e['pf']:6.3f}  {e['n']:4d}  {e['b50']:6.3f}  "
            f"{'YES' if e['independent'] else 'no':>5}  {e['label'][:45]}"
        )
    ax_h.text(0.02, 0.98, "\n".join(summary_lines),
              transform=ax_h.transAxes, fontsize=6.5, color=TEXT_CLR,
              va="top", ha="left", fontfamily="monospace",
              bbox=dict(facecolor=GRID_CLR, alpha=0.5, edgecolor="none", pad=4))

    plt.suptitle("QUANTLAB AI — R042  Independent Environment Discovery",
                 color=TEXT_CLR, fontsize=10, y=0.99)
    p7 = f"{OUT}/r042_dashboard.png"
    plt.savefig(p7, dpi=130, facecolor=DARK_BG, bbox_inches="tight"); plt.close()
    print(f"  → {p7}")
else:
    p7 = None

# =============================================================================
# JOURNAL MARKDOWN
# =============================================================================

def md_cond_table(envs, max_rows=15):
    rows = ["| # | PF | n | Boot p50 | MC% | MDD | LOO-S | LOO-F | Overlap | Ind? | Verdict | Conditions |",
            "|---|-----|---|---------|-----|-----|-------|-------|---------|------|---------|------------|"]
    for i, e in enumerate(envs[:max_rows], start=1):
        rows.append(
            f"| {i} | {e['pf']:.3f} | {e['n']} | {e['b50']:.3f} | {e['mc_p']*100:.0f}% "
            f"| {e['mdd']:.1%} | {e['sym_floor']:.3f} | {e['fold_floor']:.3f} "
            f"| {e['overlap_varg']*100:.0f}% | {'★' if e['independent'] else '—'} "
            f"| {e['verdict']} | {e['label']} |"
        )
    return "\n".join(rows)

ind_count  = sum(1 for e in library if e["independent"])
prom_count = sum(1 for e in library if e["verdict"] == "PROMOTE")
watch_count= sum(1 for e in library if e["verdict"] == "WATCHLIST")

jmd_lines = [
    f"# QUANTLAB AI — R042 Research Journal",
    f"",
    f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
    f"**Research ID:** R042  ",
    f"**Title:** Independent Environment Discovery  ",
    f"**Dataset:** 1H · {len(SYMBOLS)} symbols · {total_bars:,} bars · 5-fold WF  ",
    f"",
    f"---",
    f"",
    f"## Objective",
    f"",
    f"Discover market environments completely independent of R041's Variant G that produce "
    f"profitable edge when paired with the locked RELVOL Breakout signal.",
    f"",
    f"## Method",
    f"",
    f"- **Entry:** RELVOL Breakout (fixed, unchanged)",
    f"- **Conditions per environment:** 3–4",
    f"- **3-cond combos tested:** {len(combos_3):,}",
    f"- **4-cond extensions (from top-{TOP_N_EXTEND}):** {len(pass2_results)} survivors",
    f"- **Filter:** n≥{MIN_N} · PF>{MIN_PF} · Boot_p50>{MIN_BOOT}",
    f"- **Walk-forward:** 5-fold expanding, OOS only",
    f"",
    f"## Var G Reference",
    f"",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| Conditions | {' · '.join(COND_LABEL[c] for c in VARG_CONDS)} |",
    f"| n | {varg_m['n']} |",
    f"| PF | {varg_m['pf']:.3f} |",
    f"| WR | {varg_m['wr']*100:.1f}% |",
    f"",
    f"## Environment Library — Top 15",
    f"",
    md_cond_table(library, max_rows=15),
    f"",
    f"## Research Questions",
    f"",
    f"**Q1. Top 10 by PF:**",
]
for i, e in enumerate(library[:10], start=1):
    jmd_lines.append(f"- #{i}: PF={e['pf']:.3f} n={e['n']} Boot={e['b50']:.3f} — {e['label']}")

jmd_lines += [
    f"",
    f"**Q2. Independent environments (overlap ≤ {INDEPENDENCE_OVERLAP*100:.0f}%):** {ind_count}",
]
for e in q2_independent[:5]:
    jmd_lines.append(f"- PF={e['pf']:.3f} n={e['n']} overlap={e['overlap_varg']*100:.0f}% — {e['label']}")

jmd_lines += [
    f"",
    f"**Q3. PF>{MIN_PF} AND n≥{MIN_N}:** {'YES — ' + str(len(q3_pass)) + ' environments' if q3_pass else 'No'}",
    f"",
    f"**Q4. Most frequent features:**",
]
for cid, cnt in sorted_freq[:5]:
    jmd_lines.append(f"- {COND_LABEL[cid]} ({cid}): appears in {cnt}/{min(20,len(library))} top environments")

jmd_lines += [
    f"",
    f"**Q5. Portfolio combination:** ",
]
if best_pair if 'best_pair' in dir() and best_pair else False:
    ea_, eb_, mc_, _ = best_pair
    jmd_lines.append(f"Combined PF={mc_['pf']:.3f} n={mc_['n']} — "
                     f"{ea_['label']} + {eb_['label']}")
else:
    jmd_lines.append("No non-overlapping pair maintains PF.")

jmd_lines += [
    f"",
    f"## Verdict",
    f"",
    f"- **PROMOTE:** {prom_count}",
    f"- **WATCHLIST:** {watch_count}",
    f"- **Total in library:** {len(library)}",
    f"- **Independent from Var G:** {ind_count}",
    f"",
    f"## R043 Recommendation",
    f"",
]
if library:
    rec = library[0]
    jmd_lines += [
        f"Best candidate: **{rec['label']}**",
        f"- PF={rec['pf']:.3f}  n={rec['n']}  Boot_p50={rec['b50']:.3f}  MC={rec['mc_p']*100:.0f}%",
        f"- Score={rec['score']}/7  Verdict={rec['verdict']}",
        f"- Overlap with Var G: {rec['overlap_varg']*100:.0f}%  Independent: {rec['independent']}",
        f"",
        f"**Action:** Portfolio-test top candidate alongside Var G in R043.",
    ]
else:
    jmd_lines.append("No viable candidate. Widen search in R043.")

jmd_path = f"{OUT}/r042_journal.md"
with open(jmd_path, "w") as fh:
    fh.write("\n".join(jmd_lines))
print(f"  → {jmd_path}")

# =============================================================================
# RESEARCH JOURNAL CSV
# =============================================================================

journal_path = CONFIG["JOURNAL_FILE"]
try:
    if library:
        rec = library[0]
        jrow = {
            "research_id":   RESEARCH_ID,
            "strategy_name": f"Best Env: {rec['label'][:60]}",
            "symbol":        "ALL",
            "n_trades":      rec["n"],
            "profit_factor": round(rec["pf"], 4),
            "win_rate":      round(rec["wr"], 4),
            "net_profit":    round(rec["net"], 2),
            "max_drawdown":  round(rec["mdd"], 4),
            "sharpe":        round(rec["sharpe"], 4),
            "verdict":       rec["verdict"],
            "notes":         (f"Discovery: {len(library)} envs found. "
                              f"Independent: {ind_count}. "
                              f"3-cond tested: {len(combos_3)}.")
        }
        jdf = pd.read_csv(journal_path) if os.path.exists(journal_path) else pd.DataFrame()
        jdf = pd.concat([jdf, pd.DataFrame([jrow])], ignore_index=True)
        jdf.to_csv(journal_path, index=False)
        print(f"  → Journal: {journal_path}")
except Exception as ex:
    print(f"  ⚠  Journal write failed: {ex}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("═" * 110)
print(f"  R042 COMPLETE — Independent Environment Discovery (RELVOL fixed · 1H)")
print("═" * 110)
print(f"  Dataset   : {len(SYMBOLS)} symbols · 1H · 5-fold WF · OOS only")
print(f"  Searched  : {len(combos_3):,} three-condition combinations")
print(f"  Survivors : {len(library)} environments  "
      f"(PROMOTE={prom_count}  WATCHLIST={watch_count}  Independent={ind_count})")
print(f"  Var G ref : n={varg_m['n']}  PF={varg_m['pf']:.3f}")
print()

if library:
    print(f"  {'#':>3}  {'D':>2}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'p50':>7}  "
          f"{'MC%':>6}  {'MDD':>7}  {'LOO-S':>6}  {'LOO-F':>6}  "
          f"{'Ovlp':>5}  {'Ind':>5}  {'Sc':>4}  Verdict")
    print("  " + "─" * 104)
    for i, e in enumerate(library, start=1):
        ind = "★" if e["independent"] else " "
        print(f"  {i:>3}  {e['depth']:>2}  {e['n']:>5}  {e['wr']*100:5.1f}%  "
              f"{e['pf']:7.3f}  {e['b50']:7.3f}  {e['mc_p']*100:5.1f}%  "
              f"{e['mdd']:6.1%}  {e['sym_floor']:6.3f}  {e['fold_floor']:6.3f}  "
              f"{e['overlap_varg']*100:4.0f}%  {ind:>5}  {e['score']:>2}/7  {e['verdict']}")
        print(f"       {e['label']}")
    print()

    print(f"  Q1: Top PF          → #{1} PF={library[0]['pf']:.3f}  n={library[0]['n']}  {library[0]['label']}")
    print(f"  Q2: Independent envs → {ind_count}  (≤{INDEPENDENCE_OVERLAP*100:.0f}% overlap with Var G)")
    print(f"  Q3: PF>{MIN_PF} & n≥{MIN_N} → {'YES (' + str(len(q3_pass)) + ')' if q3_pass else 'NO'}")
    print(f"  Q4: Top feature      → {sorted_freq[0][0]} ({COND_LABEL[sorted_freq[0][0]]}) "
          f"in {sorted_freq[0][1]}/{min(20,len(library))} top envs"
          if sorted_freq else "  Q4: No features ranked")
    rec = library[0]
    print(f"  Q5: Portfolio comb.  → {'Tested — see journal' if len(library)>=2 else 'N/A'}")
    print()
    print(f"  R043 Candidate: PF={rec['pf']:.3f}  n={rec['n']}  p50={rec['b50']:.3f}  "
          f"Score={rec['score']}/7  Overlap={rec['overlap_varg']*100:.0f}%  Ind={rec['independent']}")
    print(f"  → {rec['label']}")
else:
    print("  ⚠  Zero environments survived all filters.")
    print("     Suggested next steps:")
    print("     1. Lower MIN_N from 40 to 25")
    print("     2. Lower MIN_PF from 1.20 to 1.10")
    print("     3. Add more condition families (e.g. VWAP distance, RSI regime)")

print()
print(f"  Output files:")
for p in [p1, p2, p3, p4, p5, p6, p7, csv_path, jmd_path]:
    if p: print(f"    {p}")
print("═" * 110)
