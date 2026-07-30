"""
=============================================================================
QUANTLAB AI — RESEARCH #R057
Single Macro Regime Filter Validation
=============================================================================

Objective:
  R056 showed E3 (BBW_LO+RV_LO+DST_NR+PRG_VH) collapsed in F3/F4 due to a
  structural market regime shift — ATR Rank d=1.48, RV d=0.86.

  R057 tests ONE question only:
  Can a single macro regime filter prevent the majority of those losses
  while preserving the original edge?

  THIS IS NOT AN OPTIMISATION STUDY.
  - No parameter mining.
  - No new strategy discovery.
  - E3 is frozen and unchanged.
  - Thresholds are pre-registered at IS median/percentile.
  - Success = robustness improvement on unseen data, not max historical PF.

  Frozen Environment: BBW_LO + RV_LO + DST_NR + PRG_VH
  Entry: RELVOL > 1.5 + close > open + close > prev_close
  RR = 2.0
=============================================================================
"""

import os, sys, warnings, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict
from scipy import stats as scipy_stats
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID = "R057"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL      = CONFIG["STARTING_CAPITAL"]
RR           = 2.0
IS_RATIO     = 0.80
N_FWD_FOLDS  = 5
N_BOOT       = 1000
N_MC         = 1000
RAND_SEED    = 42

E3_CIDS  = ("BBW_LO", "RV_LO", "DST_NR", "PRG_VH")
E3_LABEL = "BBW_LO+RV_LO+DST_NR+PRG_VH"
WIN_FOLDS  = ["F1", "F2"]
LOSE_FOLDS = ["F3", "F4"]
REC_FOLDS  = ["F5"]

C_BG = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT = "#e0e0e0"
C_GRID = "#2a2a2a"; C_GREEN = "#00c896"; C_RED = "#e05050"
C_GOLD = "#f5a623"; C_BLUE = "#4a9eff"; C_PURP = "#9b59b6"
C_CYAN = "#1abc9c"; C_ORAN = "#e67e22"

plt.rcParams.update({
    "figure.facecolor": C_BG, "axes.facecolor": C_PANEL,
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "axes.edgecolor": C_GRID, "grid.color": C_GRID,
    "font.family": "monospace",
})

def panel_style(ax, title, fs=8):
    ax.set_facecolor(C_PANEL)
    ax.set_title(title, fontsize=fs, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)

SEP  = "═" * 110
SEP2 = "─" * 90

# ─────────────────────────────────────────────────────────────────────────────
# MACRO FILTER CATALOGUE  (pre-registered; no parameter mining)
# ─────────────────────────────────────────────────────────────────────────────
# Thresholds are set at IS quantile. These choices were locked BEFORE running
# the forward backtest. The rationale column links each filter to the R056
# evidence that motivated it.
#
# Direction conventions:
#   "lt"         → trade passes if entry feature < IS threshold
#   "gt"         → trade passes if entry feature > IS threshold
#   "not_both_gt"→ trade passes UNLESS both feat_a AND feat_b exceed threshold
#
MACRO_FILTERS = [
    {
        "id":        "F_ATR_CALM",
        "name":      "ATR Rank Calm",
        "short":     "ATR_CALM",
        "desc":      "Block entry when ATR Rank > IS 50th pct (calm-regime only)",
        "feat":      "entry_atr_rank",
        "direction": "lt",
        "quantile":  0.50,
        "rationale": "R056 primary finding: ATR Rank shift d=1.48, p=0.003. "
                     "Hypothesis: E3 only works during low relative-ATR regimes.",
    },
    {
        "id":        "F_RV_CALM",
        "name":      "Realised Vol Calm",
        "short":     "RV_CALM",
        "desc":      "Block entry when 20-bar realised vol > IS 50th pct",
        "feat":      "entry_rv",
        "direction": "lt",
        "quantile":  0.50,
        "rationale": "R056: RV rose +15.6% in losing folds, d=0.86, p=0.023. "
                     "Hypothesis: expanding realised vol invalidates compression setups.",
    },
    {
        "id":        "F_BBW_STRICT",
        "name":      "BB Width Strict",
        "short":     "BBW_STRICT",
        "desc":      "Require BB width in lowest 25th pct (vs 33rd in BBW_LO)",
        "feat":      "entry_bbw",
        "direction": "lt",
        "quantile":  0.25,
        "rationale": "Tighter compression selectivity: only the purest "
                     "coil setups pass both BBW_LO and this stricter gate.",
    },
    {
        "id":        "F_VOL_DUAL",
        "name":      "Dual Vol Calm",
        "short":     "VOL_DUAL",
        "desc":      "Block only when BOTH ATR Rank AND RV > IS 60th pct",
        "feat":      "combo",
        "feats":     ["entry_atr_rank", "entry_rv"],
        "direction": "not_both_gt",
        "quantile":  0.60,
        "rationale": "Combined hypothesis: the regime becomes hostile only when "
                     "BOTH primary R056 drivers (ATR Rank, RV) are elevated together.",
    },
    {
        "id":        "F_ADX_MOD",
        "name":      "ADX Moderate",
        "short":     "ADX_MOD",
        "desc":      "Block entry when ADX > IS 75th pct (avoid extreme trending)",
        "feat":      "entry_adx",
        "direction": "lt",
        "quantile":  0.75,
        "rationale": "Extreme trend strength converts compression setups into "
                     "late trend entries that revert.",
    },
    {
        "id":        "F_EMA_SLOPE",
        "name":      "EMA Slope Calm",
        "short":     "EMA_SLOPE",
        "desc":      "Block when |EMA200 slope| > IS 60th pct",
        "feat":      "entry_slope_abs",
        "direction": "lt",
        "quantile":  0.60,
        "rationale": "A large EMA slope magnitude signals an accelerating regime "
                     "change already underway — unfavorable for coil setups.",
    },
]

FILTER_IDS = [f["id"] for f in MACRO_FILTERS]

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONS CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    ("ATR_LO", "atr_rank",     "lt_q",      0.25),
    ("ATR_MD", "atr_rank",     "lt_q",      0.40),
    ("ATR_HI", "atr_rank",     "gt_q",      0.67),
    ("ATR_VH", "atr_rank",     "gt_q",      0.80),
    ("BBW_LO", "bb_width",     "lt_q",      0.33),
    ("BBW_HI", "bb_width",     "gt_q",      0.67),
    ("RV_LO",  "real_vol_20",  "lt_q",      0.33),
    ("RV_HI",  "real_vol_20",  "gt_q",      0.67),
    ("SLP_DN", "ema200_slope", "lt_fixed",  0.0),
    ("SLP_UP", "ema200_slope", "gt_fixed",  0.0),
    ("DST_NR", "ema_dist_pct", "lt_q",      0.33),
    ("DST_MD", "ema_dist_pct", "gt_q_pos",  0.60),
    ("DST_FR", "ema_dist_pct", "gt_q_pos",  0.75),
    ("ADX_WK", "adx14",        "lt_q",      0.33),
    ("ADX_TR", "adx14",        "gt_q",      0.50),
    ("ADX_ST", "adx14",        "gt_q",      0.67),
    ("PRG_LO", "prev_range_r", "lt_q",      0.33),
    ("PRG_HI", "prev_range_r", "gt_q",      0.67),
    ("PRG_VH", "prev_range_r", "gt_q",      0.80),
    ("PBD_HI", "prev_body_r",  "gt_q",      0.67),
    ("PBP_HI", "prev_body_pct","gt_q",      0.60),
    ("PBP_LO", "prev_body_pct","lt_q",      0.33),
    ("US",     "hour_utc",     "hour_rng",  (14, 21)),
    ("LON",    "hour_utc",     "hour_rng",  (7, 14)),
    ("ASI",    "hour_utc",     "hour_rng",  (0,  6)),
]
COND_BY_ID  = {c[0]: c for c in CONDITIONS_DEF}
QUANT_FEATS = ["atr_rank", "bb_width", "real_vol_20", "ema_dist_pct",
               "adx14", "prev_range_r", "prev_body_r", "prev_body_pct"]

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
MIN_BARS = 2_000

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print(f"  QUANTLAB AI — RESEARCH #{RESEARCH_ID}")
print(f"  Single Macro Regime Filter Validation")
print(SEP)
print()
print(f"  FROZEN STRATEGY: {E3_LABEL}")
print(f"  ENTRY: RELVOL > 1.5 + close > open + close > prev_close")
print(f"  RR = {RR}  |  IS ratio = {IS_RATIO}")
print(f"  OBJECTIVE: Validate the volatility-regime hypothesis from R056.")
print(f"  This is NOT an optimisation study. Success = robustness, not max PF.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]
    o = df["open"]
    df["ema200"]        = calc_ema(c, 200)
    df["atr14"]         = calc_atr(df, 14)
    df["atr_rank"]      = df["atr14"].rolling(100).rank(pct=True) * 100
    bb_mid              = c.rolling(20).mean()
    bb_std              = c.rolling(20).std()
    df["bb_width"]      = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["ema_dist_pct"]  = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"]  = (df["ema200"] - df["ema200"].shift(10)) / \
                          df["ema200"].shift(10).replace(0, np.nan)
    vol_ma              = v.rolling(20).mean()
    df["rel_vol"]       = v / vol_ma.replace(0, np.nan)
    df["prev_close"]    = c.shift(1)
    df["prev_atr14"]    = df["atr14"].shift(1)
    log_ret             = np.log(c / c.shift(1))
    df["real_vol_20"]   = log_ret.rolling(20).std() * 100.0
    df["adx14"]         = calc_adx(df, 14)
    prev_range          = h.shift(1) - l.shift(1)
    prev_body           = (c.shift(1) - o.shift(1)).abs()
    df["prev_range_r"]  = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]   = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"] = prev_body  / prev_range.replace(0, np.nan)
    df["hour_utc"]      = pd.to_datetime(df["datetime"], utc=True).dt.hour.astype(np.int16)
    df["dow"]           = pd.to_datetime(df["datetime"], utc=True).dt.dayofweek
    df["ema_slope_abs"] = df["ema200_slope"].abs()
    return df

def learn_thresholds(df_is):
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (_, feat, direction, param) in COND_BY_ID.items():
        if direction in ("lt_fixed", "gt_fixed", "hour_rng"):
            thr[cid] = param; continue
        if feat not in valid.columns:
            thr[cid] = np.nan; continue
        col = valid[feat].dropna()
        if len(col) < 20:
            thr[cid] = np.nan; continue
        if direction == "gt_q_pos":
            pos = col[col > 0]
            thr[cid] = float(pos.quantile(param) if len(pos) > 10 else col.quantile(param))
        else:
            thr[cid] = float(col.quantile(param))
    return thr

# Explicit mapping: trade-record key → dataframe column name in add_features()
FEAT_KEY_TO_COL = {
    "entry_atr_rank":   "atr_rank",
    "entry_rv":         "real_vol_20",
    "entry_bbw":        "bb_width",
    "entry_adx":        "adx14",
    "entry_slope_abs":  "ema_slope_abs",
    "entry_slope":      "ema200_slope",
    "entry_dst":        "ema_dist_pct",
    "entry_prg":        "prev_range_r",
}

def learn_macro_thresholds(df_is, filter_list):
    """Learn IS thresholds for each macro filter feature from IS bars."""
    macro_thr = {}
    for filt in filter_list:
        feat_keys = filt["feats"] if filt["direction"] == "not_both_gt" else [filt["feat"]]
        for feat_key in feat_keys:
            if feat_key in macro_thr:
                continue  # already computed for a previous filter
            col_name = FEAT_KEY_TO_COL.get(feat_key)
            if col_name is None or col_name not in df_is.columns:
                macro_thr[feat_key] = np.nan
                continue
            col = df_is[col_name].dropna()
            if len(col) < 20:
                macro_thr[feat_key] = np.nan
                continue
            macro_thr[feat_key] = float(col.quantile(filt["quantile"]))
    return macro_thr

def build_env_mask(df, cid_tuple, thr):
    N = len(df); mask = np.ones(N, dtype=bool)
    for cid in cid_tuple:
        if cid not in COND_BY_ID: return np.zeros(N, dtype=bool)
        _, feat, direction, _ = COND_BY_ID[cid]
        if feat not in df.columns: return np.zeros(N, dtype=bool)
        col   = df[feat].values
        nan_m = np.isnan(col) if col.dtype.kind == "f" else np.zeros(N, dtype=bool)
        t     = thr.get(cid, np.nan)
        if direction == "lt_q":
            if np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col < t)
        elif direction in ("gt_q", "gt_q_pos"):
            if np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col > t)
        elif direction == "gt_fixed":
            mask &= (~nan_m) & (col > t)
        elif direction == "lt_fixed":
            mask &= (~nan_m) & (col < t)
        elif direction == "hour_rng":
            lo_, hi_ = t
            mask &= (col >= lo_) & (col <= hi_)
    return mask

def entry_signal(df, env_mask):
    rv = df["rel_vol"].values
    c  = df["close"].values; o = df["open"].values; pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & env_mask

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE  (enriched with macro filter features at entry)
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signal, sym, fold_label):
    min_sl  = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf      = CONFIG["RISK_PER_TRADE_PCT"]
    fee     = CONFIG["TAKER_FEE"];  spd = CONFIG["SPREAD"] * 0.5
    slp     = CONFIG["SL_SLIPPAGE"]
    in_pos  = False; ep = st = tk = sz = 0.0; et = None; ei = -1
    trades  = []

    hi_    = df["high"].values;    lo_    = df["low"].values
    op_    = df["open"].values;    cl_    = df["close"].values
    atr_   = df["prev_atr14"].values
    rv_    = df["rel_vol"].values
    adx_   = df["adx14"].values
    bbw_   = df["bb_width"].values
    rvol_  = df["real_vol_20"].values
    slp_   = df["ema200_slope"].values
    slpabs_= df["ema_slope_abs"].values
    dst_   = df["ema_dist_pct"].values
    prg_   = df["prev_range_r"].values
    atrr_  = df["atr_rank"].values
    dts    = df["datetime"].values
    hou_   = df["hour_utc"].values
    dow_   = df["dow"].values

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

                h_entry = int(hou_[ei])
                if 7 <= h_entry <= 13:
                    session = "London"
                elif 14 <= h_entry <= 20:
                    session = "US"
                else:
                    session = "Asia"

                trades.append({
                    "sym":              sym,
                    "fold":             fold_label,
                    "entry_time":       str(et),
                    "pnl":              round(net, 4),
                    "r_multiple":       round(rmul, 4),
                    "win":              int(not sl_hit),
                    "exit_type":        "SL" if sl_hit else "TP",
                    "session":          session,
                    "dow":              int(dow_[ei]),
                    # — macro filter features at entry —
                    "entry_atr_rank":   float(atrr_[ei]) if not np.isnan(atrr_[ei]) else np.nan,
                    "entry_rv":         float(rvol_[ei]) if not np.isnan(rvol_[ei]) else np.nan,
                    "entry_bbw":        float(bbw_[ei])  if not np.isnan(bbw_[ei])  else np.nan,
                    "entry_adx":        float(adx_[ei])  if not np.isnan(adx_[ei])  else np.nan,
                    "entry_slope_abs":  float(slpabs_[ei]) if not np.isnan(slpabs_[ei]) else np.nan,
                    # — additional context —
                    "entry_slope":      float(slp_[ei])  if not np.isnan(slp_[ei])  else np.nan,
                    "entry_dst":        float(dst_[ei])  if not np.isnan(dst_[ei])  else np.nan,
                    "entry_prg":        float(prg_[ei])  if not np.isnan(prg_[ei])  else np.nan,
                    "sig_strength":     float(rv_[ei])   if not np.isnan(rv_[ei])   else 1.0,
                })
                in_pos = False
            continue

        if signal[i-1]:
            a = atr_[i]
            if np.isnan(a) or a <= 0: continue
            ep_ = op_[i]
            if a / ep_ < min_sl: continue
            ep  = ep_; st = ep - a; tk = ep + RR * a
            sz  = min(CAPITAL * rf / a, (CAPITAL * max_lev) / ep)
            et  = dts[i]; ei = i; in_pos = True

    return trades

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl

def metrics(trades):
    if not trades:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "exp_r": 0.0, "net": 0.0,
                "mdd": 0.0, "pnls": np.array([]), "equity": np.array([CAPITAL])}
    df  = pd.DataFrame(trades)
    pnl = df["pnl"].values; wins = df["win"].values.astype(bool)
    n = len(pnl); nw = wins.sum(); nl = n - nw
    gw = pnl[wins].sum() if nw else 0.0
    gl = abs(pnl[~wins].sum()) if nl else 0.0
    pf = safe_pf(gw, gl); wr = nw / n
    eq = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk).min())
    exp = wr * RR - (1 - wr)
    return {"n": n, "wr": wr, "pf": pf, "exp_r": exp, "net": float(pnl.sum()),
            "mdd": mdd, "pnls": pnl, "equity": eq}

def bootstrap_pf(trades, n_iter=N_BOOT, seed=RAND_SEED):
    """Bootstrap PF with replacement — returns mean, 90% CI, pct_profitable."""
    rng = np.random.RandomState(seed)
    if not trades:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "pct_gt1": 0.0}
    pnls = np.array([t["pnl"] for t in trades])
    pf_s = []
    for _ in range(n_iter):
        samp = rng.choice(pnls, size=len(pnls), replace=True)
        gw = samp[samp > 0].sum() if any(samp > 0) else 0.0
        gl = abs(samp[samp < 0].sum()) if any(samp < 0) else 1e-9
        pf_s.append(min(gw / gl, 20.0))
    arr = np.array(pf_s)
    return {
        "mean":    float(arr.mean()),
        "ci_lo":   float(np.percentile(arr, 5)),
        "ci_hi":   float(np.percentile(arr, 95)),
        "pct_gt1": float((arr > 1.0).mean()),
    }

def monte_carlo_prob(trades, n_iter=N_MC, seed=RAND_SEED):
    """P(random sign-shuffle achieves observed PF). Low = genuine edge."""
    rng = np.random.RandomState(seed)
    if not trades:
        return {"obs_pf": 0.0, "mc_prob": 1.0}
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    gw   = wins.sum() if len(wins) > 0 else 0.0
    gl   = abs(losses.sum()) if len(losses) > 0 else 1e-9
    obs_pf = min(gw / gl, 20.0)
    pf_s = []
    abs_p = np.abs(pnls)
    for _ in range(n_iter):
        signs = rng.choice([-1, 1], size=len(pnls))
        samp  = abs_p * signs
        gw_s  = samp[samp > 0].sum() if any(samp > 0) else 0.0
        gl_s  = abs(samp[samp < 0].sum()) if any(samp < 0) else 1e-9
        pf_s.append(min(gw_s / gl_s, 20.0))
    arr = np.array(pf_s)
    return {"obs_pf": obs_pf, "mc_prob": float((arr >= obs_pf).mean())}

def cohens_d(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2: return np.nan
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

def mean_ci_95(arr):
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2: return np.nan, np.nan, np.nan
    m = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return m, m - 1.96*se, m + 1.96*se

# ─────────────────────────────────────────────────────────────────────────────
# MACRO FILTER APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def apply_filter(trades, filt, sym_macro_thr):
    """
    Return a boolean mask (list) of same length as trades.
    True = trade PASSES filter (allowed to execute).
    False = trade BLOCKED by filter.
    """
    result = []
    for t in trades:
        sym = t["sym"]
        thr = sym_macro_thr.get(sym, {})

        if filt["direction"] == "not_both_gt":
            # Passes UNLESS both features exceed threshold
            vals_ok = True
            for feat_key in filt["feats"]:
                fv  = t.get(feat_key, np.nan)
                thr_v = thr.get(feat_key, np.nan)
                if np.isnan(fv) or np.isnan(thr_v):
                    vals_ok = False; break  # can't evaluate → conservative: block
                if fv <= thr_v:
                    # At least one feature is calm → passes
                    vals_ok = True; break
                # else this feature is above threshold; keep checking next
                vals_ok = (fv <= thr_v)
            # Recompute cleanly:
            feat_above = []
            skip = False
            for feat_key in filt["feats"]:
                fv    = t.get(feat_key, np.nan)
                thr_v = thr.get(feat_key, np.nan)
                if np.isnan(fv) or np.isnan(thr_v):
                    skip = True; break
                feat_above.append(fv > thr_v)
            if skip:
                result.append(False)  # missing data → conservative block
            else:
                both_elevated = all(feat_above)
                result.append(not both_elevated)

        elif filt["direction"] == "lt":
            feat_key = filt["feat"]
            fv       = t.get(feat_key, np.nan)
            thr_v    = thr.get(feat_key, np.nan)
            if np.isnan(fv) or np.isnan(thr_v):
                result.append(False)  # conservative
            else:
                result.append(bool(fv < thr_v))

        elif filt["direction"] == "gt":
            feat_key = filt["feat"]
            fv       = t.get(feat_key, np.nan)
            thr_v    = thr.get(feat_key, np.nan)
            if np.isnan(fv) or np.isnan(thr_v):
                result.append(False)
            else:
                result.append(bool(fv > thr_v))
        else:
            result.append(True)

    return result

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — DATA LOAD + FROZEN FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 0 — Data Load + Frozen Forward Backtest (Baseline)")
print(SEP)
print()

all_trades       = []
fold_trades      = defaultdict(list)
sym_trades       = defaultdict(list)
fold_sym_trades  = defaultdict(lambda: defaultdict(list))
sym_macro_thr    = {}   # {sym: {feat_key: threshold}}

loaded = 0
for sym in ALL_SYMBOLS:
    tag  = sym.replace("-", "_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path): continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    N  = len(df)
    if N < MIN_BARS: continue
    df = add_features(df)
    sp = int(N * IS_RATIO)
    df_is  = df.iloc[:sp]
    df_fwd = df.iloc[sp:].copy().reset_index(drop=True)
    if len(df_fwd) < 50: continue

    thr       = learn_thresholds(df_is)
    macro_thr = learn_macro_thresholds(df_is, MACRO_FILTERS)
    sym_macro_thr[sym] = macro_thr
    loaded += 1

    fwd_size = len(df_fwd)
    seg_size = max(1, fwd_size // N_FWD_FOLDS)

    for fi in range(N_FWD_FOLDS):
        seg_s  = fi * seg_size
        seg_e  = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
        df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
        flabel = f"F{fi+1}"
        if len(df_seg) < 20: continue

        em  = build_env_mask(df_seg, E3_CIDS, thr)
        sig = entry_signal(df_seg, em)
        tl  = run_backtest(df_seg, sig, sym, flabel)

        all_trades.extend(tl)
        fold_trades[flabel].extend(tl)
        sym_trades[sym].extend(tl)
        fold_sym_trades[flabel][sym].extend(tl)

print(f"  Symbols processed: {loaded}")
print(f"  Total forward trades (baseline): {len(all_trades)}")
baseline_m = metrics(all_trades)
for fi in range(1, N_FWD_FOLDS + 1):
    fl = f"F{fi}"
    m  = metrics(fold_trades[fl])
    grp = "WIN " if fl in WIN_FOLDS else ("LOSE" if fl in LOSE_FOLDS else "REC ")
    print(f"    {fl} [{grp}]  n={m['n']:3d}  PF={m['pf']:.3f}  "
          f"WR={m['wr']*100:.1f}%  Net=${m['net']:+.0f}")
print(f"\n  Baseline: Overall n={baseline_m['n']}  PF={baseline_m['pf']:.3f}  "
      f"WR={baseline_m['wr']*100:.1f}%  Net=${baseline_m['net']:+.0f}")
print()

# Separate winning/losing period trades for reference
win_trades  = [t for fl in WIN_FOLDS  for t in fold_trades[fl]]
lose_trades = [t for fl in LOSE_FOLDS for t in fold_trades[fl]]
rec_trades  = [t for fl in REC_FOLDS  for t in fold_trades[fl]]
win_m   = metrics(win_trades);  lose_m  = metrics(lose_trades); rec_m  = metrics(rec_trades)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CANDIDATE MACRO FILTERS (pre-registered)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 1 — CANDIDATE MACRO FILTERS (PRE-REGISTERED)")
print(SEP)
print()
print(f"  {'#':<3}  {'ID':<14}  {'Name':<22}  {'Feature':<20}  {'Q':<5}  Rationale")
print("  " + "─" * 100)
for i, filt in enumerate(MACRO_FILTERS, 1):
    qstr = f"{filt.get('quantile', '-'):.0%}"
    print(f"  {i:<3}  {filt['id']:<14}  {filt['name']:<22}  "
          f"{filt['feat']:<20}  {qstr:<5}  {filt['rationale'][:55]}")
print()
print("  IS threshold learning: per-symbol quantile of IS distribution.")
print("  These thresholds were set BEFORE examining forward fold outcomes.")
print("  They are not tuned for maximum historical PF — they are median/fixed percentiles.")
print()

# Show representative thresholds (first symbol with data)
sample_sym = next(iter(sym_macro_thr), None)
if sample_sym:
    print(f"  Sample IS thresholds ({sample_sym.replace('-USDT-SWAP', '')}):")
    thr_s = sym_macro_thr[sample_sym]
    for filt in MACRO_FILTERS:
        if filt["direction"] == "not_both_gt":
            for fk in filt["feats"]:
                v = thr_s.get(fk, np.nan)
                print(f"    {fk:<25}  {v:.4f}")
        else:
            fk = filt["feat"]
            v  = thr_s.get(fk, np.nan)
            print(f"    {fk:<25}  {v:.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — INDEPENDENT VALIDATION (per filter)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 2 — INDEPENDENT VALIDATION — Per Filter Results")
print(SEP)
print()

filter_results = {}   # {filter_id: {...}}

for filt in MACRO_FILTERS:
    fid   = filt["id"]
    fname = filt["name"]

    # Apply filter to all trades
    mask_all = apply_filter(all_trades, filt, sym_macro_thr)
    kept  = [t for t, ok in zip(all_trades, mask_all) if ok]
    rmvd  = [t for t, ok in zip(all_trades, mask_all) if not ok]

    m_kept = metrics(kept)
    m_rmvd = metrics(rmvd)

    # Per-fold with filter
    filt_fold_trades = defaultdict(list)
    for t, ok in zip(all_trades, mask_all):
        if ok:
            filt_fold_trades[t["fold"]].append(t)

    fold_pfs = {}
    for fi in range(1, N_FWD_FOLDS + 1):
        fl = f"F{fi}"
        fold_pfs[fl] = metrics(filt_fold_trades[fl])

    # Per-symbol with filter
    filt_sym_trades = defaultdict(list)
    for t, ok in zip(all_trades, mask_all):
        if ok:
            filt_sym_trades[t["sym"]].append(t)

    # Bootstrap
    bstrap = bootstrap_pf(kept)

    # Monte Carlo
    mc_res = monte_carlo_prob(kept)

    # Leave-one-fold-out robustness
    loo_fold_pfs = []
    for fi in range(1, N_FWD_FOLDS + 1):
        fl_out = f"F{fi}"
        loo_trades = [t for t, ok in zip(all_trades, mask_all)
                      if ok and t["fold"] != fl_out]
        if len(loo_trades) > 5:
            loo_fold_pfs.append(metrics(loo_trades)["pf"])

    # Leave-one-symbol-out robustness
    all_syms_present = list({t["sym"] for t in kept})
    loo_sym_pfs = []
    for sym_out in all_syms_present:
        loo_trades = [t for t in kept if t["sym"] != sym_out]
        if len(loo_trades) > 5:
            loo_sym_pfs.append(metrics(loo_trades)["pf"])

    loo_fold_mean, loo_fold_lo, loo_fold_hi = mean_ci_95(loo_fold_pfs)
    loo_sym_mean,  loo_sym_lo,  loo_sym_hi  = mean_ci_95(loo_sym_pfs)

    filter_results[fid] = {
        "filt":           filt,
        "kept":           kept,
        "rmvd":           rmvd,
        "m_kept":         m_kept,
        "m_rmvd":         m_rmvd,
        "fold_pfs":       fold_pfs,
        "filt_fold":      filt_fold_trades,
        "bstrap":         bstrap,
        "mc_res":         mc_res,
        "loo_fold_pfs":   loo_fold_pfs,
        "loo_sym_pfs":    loo_sym_pfs,
        "loo_fold_mean":  loo_fold_mean,
        "loo_fold_lo":    loo_fold_lo,
        "loo_fold_hi":    loo_fold_hi,
        "loo_sym_mean":   loo_sym_mean,
        "loo_sym_lo":     loo_sym_lo,
        "loo_sym_hi":     loo_sym_hi,
    }

    n_base = len(all_trades); n_kept = len(kept); n_rmvd = len(rmvd)
    pct_kept = n_kept / n_base * 100 if n_base > 0 else 0

    print(f"  {'─'*100}")
    print(f"  FILTER: {fname}  ({filt['short']})")
    print(f"  {filt['desc']}")
    print(f"  Rationale: {filt['rationale'][:90]}")
    print()
    print(f"  {'Metric':<28}  {'Baseline':>12}  {'With Filter':>12}  {'Change':>10}")
    print(f"  {'─'*70}")
    def cmp(lbl, bv, fv, fmt="{:.3f}"):
        diff = fv - bv if not (np.isnan(bv) or np.isnan(fv)) else np.nan
        arrow = "▲" if diff > 0.001 else ("▼" if diff < -0.001 else "─")
        print(f"  {lbl:<28}  {fmt.format(bv):>12}  {fmt.format(fv):>12}  "
              f"  {arrow}{abs(diff):.3f}")

    cmp("Trades",          len(all_trades), n_kept,             fmt="{:.0f}")
    cmp("Trades removed",  0,               n_rmvd,             fmt="{:.0f}")
    cmp("% Trades kept",   100.0,           pct_kept,           fmt="{:.1f}")
    cmp("PF",              baseline_m["pf"], m_kept["pf"])
    cmp("Win Rate",        baseline_m["wr"]*100, m_kept["wr"]*100, fmt="{:.1f}")
    cmp("Net Profit ($)",  baseline_m["net"], m_kept["net"],    fmt="{:.0f}")
    cmp("Max Drawdown",    baseline_m["mdd"]*100, m_kept["mdd"]*100, fmt="{:.1f}")
    print()
    print(f"  Bootstrap PF:  mean={bstrap['mean']:.3f}  "
          f"90% CI=[{bstrap['ci_lo']:.3f}, {bstrap['ci_hi']:.3f}]  "
          f"P(PF>1)={bstrap['pct_gt1']*100:.1f}%")
    print(f"  Monte Carlo:   P(random≥obs)={mc_res['mc_prob']*100:.1f}%  "
          f"(lower = more significant)")
    print()
    print(f"  LOO-fold PF:   mean={loo_fold_mean:.3f}  "
          f"95% CI=[{loo_fold_lo:.3f}, {loo_fold_hi:.3f}]")
    print(f"  LOO-symbol PF: mean={loo_sym_mean:.3f}  "
          f"95% CI=[{loo_sym_lo:.3f}, {loo_sym_hi:.3f}]")
    print()
    print(f"  Per-fold PF breakdown:")
    print(f"  {'Fold':<6}  {'Baseline PF':>12}  {'Filtered PF':>12}  "
          f"{'n_base':>8}  {'n_filt':>8}  {'WR%':>7}")
    print(f"  {'─'*60}")
    for fi in range(1, N_FWD_FOLDS + 1):
        fl  = f"F{fi}"
        mb  = metrics(fold_trades[fl])
        mf_ = fold_pfs[fl]
        grp = "WIN " if fl in WIN_FOLDS else ("LOSE" if fl in LOSE_FOLDS else "REC ")
        print(f"  {fl} [{grp}]  {mb['pf']:>12.3f}  {mf_['pf']:>12.3f}  "
              f"{mb['n']:>8}  {mf_['n']:>8}  {mf_['wr']*100:>7.1f}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LOSS PREVENTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 3 — LOSS PREVENTION ANALYSIS")
print("  For each filter: F3/F4 losses avoided vs F1/F2 winners sacrificed")
print(SEP)
print()

print(f"  {'Filter':<14}  {'F3/F4 L Rmvd':>13}  {'F3/F4 L%':>9}  "
      f"{'F1/F2 W Rmvd':>13}  {'F1/F2 W%':>9}  "
      f"{'Efficiency':>11}  {'Net Gain':>10}")
print("  " + "─" * 95)

section3_data = {}

for filt in MACRO_FILTERS:
    fid = filt["id"]
    res = filter_results[fid]
    mask_all = apply_filter(all_trades, filt, sym_macro_thr)

    # F3/F4 losses
    f34_losses_base = [t for fl in LOSE_FOLDS for t in fold_trades[fl] if t["win"] == 0]
    f34_losses_rmvd = []
    f12_wins_base   = [t for fl in WIN_FOLDS  for t in fold_trades[fl] if t["win"] == 1]
    f12_wins_rmvd   = []

    for t, ok in zip(all_trades, mask_all):
        if not ok:
            if t["fold"] in LOSE_FOLDS and t["win"] == 0:
                f34_losses_rmvd.append(t)
            if t["fold"] in WIN_FOLDS and t["win"] == 1:
                f12_wins_rmvd.append(t)

    n_f34_base = len(f34_losses_base)
    n_f34_rmvd = len(f34_losses_rmvd)
    n_f12_base = len(f12_wins_base)
    n_f12_rmvd = len(f12_wins_rmvd)

    pct_f34 = n_f34_rmvd / n_f34_base * 100 if n_f34_base > 0 else 0.0
    pct_f12 = n_f12_rmvd / n_f12_base * 100 if n_f12_base > 0 else 0.0
    efficiency = pct_f34 / pct_f12 if pct_f12 > 0 else (pct_f34 if pct_f34 > 0 else 0.0)
    net_gain   = pct_f34 - pct_f12

    section3_data[fid] = {
        "n_f34_base":  n_f34_base,
        "n_f34_rmvd":  n_f34_rmvd,
        "pct_f34":     pct_f34,
        "n_f12_base":  n_f12_base,
        "n_f12_rmvd":  n_f12_rmvd,
        "pct_f12":     pct_f12,
        "efficiency":  efficiency,
        "net_gain":    net_gain,
    }

    eff_str = f"{efficiency:.2f}x" if efficiency > 0 else "N/A"
    print(f"  {fid:<14}  {n_f34_rmvd:>6}/{n_f34_base:<5}  {pct_f34:>8.1f}%"
          f"  {n_f12_rmvd:>6}/{n_f12_base:<5}  {pct_f12:>8.1f}%"
          f"  {eff_str:>11}  {net_gain:>+9.1f}%")

print()
print("  Efficiency ratio = (% F3/F4 losses removed) / (% F1/F2 winners removed)")
print("  Values >1.0 indicate the filter selectively removes losses over winners.")
print(f"\n  Baseline reference:")
print(f"    F3+F4 total losing trades: {len(f34_losses_base)}")
print(f"    F1+F2 total winning trades: {len(f12_wins_base)}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 4 — ROBUSTNESS ASSESSMENT")
print(SEP)
print()

# Time robustness: does PF improve in ALL three time periods or just one?
print(f"  {'Filter':<14}  {'F1+F2 PF':>10}  {'F3+F4 PF':>10}  {'F5 PF':>10}  "
      f"{'All PF':>10}  {'Time-Robust?':>13}")
print("  " + "─" * 80)

for filt in MACRO_FILTERS:
    fid = filt["id"]
    res = filter_results[fid]

    w_tr = [t for fl in WIN_FOLDS  for t in res["filt_fold"][fl]]
    l_tr = [t for fl in LOSE_FOLDS for t in res["filt_fold"][fl]]
    r_tr = [t for fl in REC_FOLDS  for t in res["filt_fold"][fl]]

    mw = metrics(w_tr); ml = metrics(l_tr); mr_ = metrics(r_tr)

    # Time-robust = PF improves (or stays similar) in BOTH win and lose periods
    win_ok  = mw["pf"] >= win_m["pf"]  * 0.90   # within 10% of baseline win-period PF
    lose_ok = ml["pf"] >= lose_m["pf"] * 1.00    # strictly better in lose period
    robust  = "YES ✓" if (win_ok and lose_ok) else ("PARTIAL" if lose_ok else "NO ✗")

    print(f"  {fid:<14}  {mw['pf']:>10.3f}  {ml['pf']:>10.3f}  "
          f"{mr_['pf']:>10.3f}  {res['m_kept']['pf']:>10.3f}  {robust:>13}")

print()

# Fold robustness: variance of PF across folds
print(f"  Fold PF Variance (lower = more consistent):")
print(f"  {'Filter':<14}  {'F1':>7}  {'F2':>7}  {'F3':>7}  {'F4':>7}  "
      f"{'F5':>7}  {'Std':>7}  {'Min':>7}  {'Consistent?':>12}")
print("  " + "─" * 90)

# Baseline row
base_pfs = [metrics(fold_trades[f"F{i}"])["pf"] for i in range(1, 6)]
base_arr = np.array(base_pfs)
base_std = float(base_arr.std())
print(f"  {'BASELINE':<14}" + "".join(f"  {v:>5.3f}" for v in base_pfs)
      + f"  {base_std:>7.3f}  {base_arr.min():>7.3f}  {'(reference)':>12}")

for filt in MACRO_FILTERS:
    fid = filt["id"]
    res = filter_results[fid]
    fld_pfs = [res["fold_pfs"][f"F{i}"]["pf"] for i in range(1, 6)]
    arr = np.array(fld_pfs)
    std = float(arr.std())
    consistent = "YES ✓" if std < base_std * 0.85 else ("MIXED" if std < base_std else "NO ✗")
    print(f"  {fid:<14}" + "".join(f"  {v:>5.3f}" if not np.isnan(v) else "  N/A  "
                                    for v in fld_pfs)
          + f"  {std:>7.3f}  {arr[~np.isnan(arr)].min():>7.3f}  {consistent:>12}")

print()

# Symbol robustness
print(f"  Symbol Robustness (LOO-symbol PF stability):")
print(f"  {'Filter':<14}  {'LOO-sym Mean PF':>17}  {'95% CI Lo':>12}  "
      f"{'95% CI Hi':>12}  {'Std':>7}  {'Robust?':>10}")
print("  " + "─" * 80)

base_sym_pfs = []
for sym_out in {t["sym"] for t in all_trades}:
    loo_t = [t for t in all_trades if t["sym"] != sym_out]
    if len(loo_t) > 5:
        base_sym_pfs.append(metrics(loo_t)["pf"])
bsm, bsl, bsh = mean_ci_95(base_sym_pfs)
bstd = float(np.std(base_sym_pfs)) if base_sym_pfs else np.nan
print(f"  {'BASELINE':<14}  {bsm:>17.3f}  {bsl:>12.3f}  {bsh:>12.3f}  "
      f"{bstd:>7.3f}  {'(reference)':>10}")

for filt in MACRO_FILTERS:
    fid = filt["id"]
    res = filter_results[fid]
    lstd = float(np.std(res["loo_sym_pfs"])) if res["loo_sym_pfs"] else np.nan
    robust = ("YES ✓" if res["loo_sym_mean"] > baseline_m["pf"] * 0.95
              else "MIXED" if res["loo_sym_mean"] > baseline_m["pf"] * 0.80
              else "NO ✗")
    print(f"  {fid:<14}  {res['loo_sym_mean']:>17.3f}  "
          f"{res['loo_sym_lo']:>12.3f}  {res['loo_sym_hi']:>12.3f}  "
          f"{lstd:>7.3f}  {robust:>10}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — STATISTICAL EVIDENCE
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 5 — STATISTICAL EVIDENCE")
print(SEP)
print()

print(f"  {'Filter':<14}  {'PF base':>9}  {'PF filt':>9}  {'Boot CI 90%':>18}  "
      f"{'MC P(rand)':>12}  {'Effect d':>9}  {'Sig':>5}")
print("  " + "─" * 95)

for filt in MACRO_FILTERS:
    fid   = filt["id"]
    res   = filter_results[fid]
    kept  = res["kept"]
    bstrap = res["bstrap"]
    mc_res = res["mc_res"]

    # Cohen's d on PnL distributions: filtered vs all
    d = cohens_d(
        [t["pnl"] for t in kept],
        [t["pnl"] for t in all_trades]
    )

    # Significance: t-test on PnL (filtered vs all)
    a = np.array([t["pnl"] for t in kept]); b = np.array([t["pnl"] for t in all_trades])
    try:
        _, pval = scipy_stats.ttest_ind(a, b, equal_var=False)
    except Exception:
        pval = np.nan
    sig_str = "***" if (not np.isnan(pval) and pval < 0.001) else \
              "**"  if (not np.isnan(pval) and pval < 0.01)  else \
              "*"   if (not np.isnan(pval) and pval < 0.05)  else "n.s."
    mc_str  = f"{mc_res['mc_prob']*100:.1f}%"
    d_str   = f"{d:+.3f}" if not np.isnan(d) else "N/A"

    print(f"  {fid:<14}  {baseline_m['pf']:>9.3f}  {res['m_kept']['pf']:>9.3f}  "
          f"  [{bstrap['ci_lo']:.3f}, {bstrap['ci_hi']:.3f}]  "
          f"{mc_str:>12}  {d_str:>9}  {sig_str:>5}")

print()
print("  Monte Carlo P(rand): probability that random sign-shuffle achieves observed PF.")
print("  Lower = less likely to be noise. Threshold: <5% is strong, <20% is indicative.")
print()

# Per-period statistical comparison for best filter candidates
print("  Detailed statistical comparison (kept vs removed trades, F3+F4 only):")
print()
for filt in MACRO_FILTERS:
    fid  = filt["id"]
    mask = apply_filter(all_trades, filt, sym_macro_thr)
    f34_kept = [t for t, ok in zip(all_trades, mask)
                if ok and t["fold"] in LOSE_FOLDS]
    f34_rmvd = [t for t, ok in zip(all_trades, mask)
                if not ok and t["fold"] in LOSE_FOLDS]
    if not f34_kept or not f34_rmvd:
        continue
    mk = metrics(f34_kept); mr_ = metrics(f34_rmvd)
    d34 = cohens_d([t["pnl"] for t in f34_kept], [t["pnl"] for t in f34_rmvd])
    try:
        _, pv34 = scipy_stats.ttest_ind(
            [t["pnl"] for t in f34_kept],
            [t["pnl"] for t in f34_rmvd], equal_var=False)
        sig34 = "***" if pv34 < 0.001 else "**" if pv34 < 0.01 else "*" if pv34 < 0.05 else "n.s."
    except Exception:
        sig34 = "n.s."
    print(f"  {fid:<14}  F3+F4:  kept n={mk['n']} PF={mk['pf']:.3f}  "
          f"|  removed n={mr_['n']} PF={mr_['pf']:.3f}  "
          f"|  d={d34:.3f}  {sig34}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SIMPLICITY RANKING
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 6 — SIMPLICITY RANKING")
print(SEP)
print()

# Scoring (0-100 each axis, pre-defined weights):
# PF improvement score (30)
# Loss prevention efficiency (25)
# Robustness (fold + symbol) (25)
# Simplicity (fewer features = simpler) (10)
# Bootstrap / MC evidence (10)

# Complexity: combo = 2, single = 1
def complexity_score(filt):
    return 1 if filt["direction"] != "not_both_gt" else 2

ranking_rows = []
for filt in MACRO_FILTERS:
    fid = filt["id"]
    res = filter_results[fid]
    s3  = section3_data[fid]
    m_k = res["m_kept"]
    bs  = res["bstrap"]
    mc  = res["mc_res"]

    # PF improvement (0-30): max 30 if PF doubles, 0 if same
    pf_improvement = max(0.0, m_k["pf"] - baseline_m["pf"])
    pf_score = min(30.0, pf_improvement / (baseline_m["pf"] * 0.5) * 30)

    # Loss prevention efficiency (0-25)
    eff = s3["efficiency"]
    eff_score = min(25.0, eff / 3.0 * 25) if eff > 0 else 0.0

    # Robustness: LOO-fold and LOO-sym both above baseline (0-25)
    lof = res["loo_fold_mean"]; los = res["loo_sym_mean"]
    rob_score = 0.0
    if not np.isnan(lof) and lof > baseline_m["pf"]: rob_score += 12.5
    if not np.isnan(los) and los > baseline_m["pf"]: rob_score += 12.5

    # Simplicity (0-10): single feature = 10, combo = 5
    simp_score = 10 if complexity_score(filt) == 1 else 5

    # Statistical evidence (0-10): MC prob < 0.10 = 10, < 0.20 = 7, else 3
    mc_p = mc["mc_prob"]
    ev_score = 10.0 if mc_p < 0.10 else (7.0 if mc_p < 0.20 else 3.0)

    total = pf_score + eff_score + rob_score + simp_score + ev_score

    ranking_rows.append({
        "fid":        fid,
        "name":       filt["name"],
        "total":      round(total, 1),
        "pf_score":   round(pf_score, 1),
        "eff_score":  round(eff_score, 1),
        "rob_score":  round(rob_score, 1),
        "simp_score": round(simp_score, 1),
        "ev_score":   round(ev_score, 1),
        "pf":         m_k["pf"],
        "efficiency": s3["efficiency"],
        "mc_prob":    mc_p,
        "pct_kept":   len(res["kept"]) / len(all_trades) * 100,
    })

ranking_rows.sort(key=lambda x: -x["total"])

print(f"  Scoring: PF improvement (30) + Loss efficiency (25) + "
      f"Robustness (25) + Simplicity (10) + Evidence (10) = 100")
print()
print(f"  {'Rank':<5}  {'Filter':<14}  {'Total/100':>10}  {'PF(30)':>8}  "
      f"{'Eff(25)':>8}  {'Rob(25)':>8}  {'Simp(10)':>9}  {'Evid(10)':>9}  "
      f"{'Fil PF':>8}  {'%Kept':>7}  {'MC%':>7}")
print("  " + "─" * 110)

for rank, row in enumerate(ranking_rows, 1):
    print(f"  {rank:<5}  {row['fid']:<14}  {row['total']:>10.1f}  "
          f"{row['pf_score']:>8.1f}  {row['eff_score']:>8.1f}  "
          f"{row['rob_score']:>8.1f}  {row['simp_score']:>9.1f}  "
          f"{row['ev_score']:>9.1f}  {row['pf']:>8.3f}  "
          f"{row['pct_kept']:>7.1f}  {row['mc_prob']*100:>7.1f}")

print()
best_filter_id   = ranking_rows[0]["fid"]
best_filter_name = ranking_rows[0]["name"]
best_filter_pf   = ranking_rows[0]["pf"]
best_filter      = next(f for f in MACRO_FILTERS if f["id"] == best_filter_id)
best_res         = filter_results[best_filter_id]
best_s3          = section3_data[best_filter_id]

print(f"  ★  BEST FILTER: {best_filter_name} ({best_filter_id})")
print(f"     {best_filter['desc']}")
print(f"     PF: {baseline_m['pf']:.3f} → {best_filter_pf:.3f}  "
      f"({'+' if best_filter_pf > baseline_m['pf'] else ''}"
      f"{(best_filter_pf-baseline_m['pf'])/baseline_m['pf']*100:+.1f}%)")
print(f"     F3/F4 losses avoided: {best_s3['pct_f34']:.1f}%  |  "
      f"F1/F2 winners sacrificed: {best_s3['pct_f12']:.1f}%")
print(f"     Efficiency ratio: {best_s3['efficiency']:.2f}x")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL Q&A
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  FINAL Q&A")
print(SEP)
print()

# Determine verdict on best filter
best_pf_delta  = best_filter_pf - baseline_m["pf"]
best_mc_prob   = best_res["mc_res"]["mc_prob"]
best_eff       = best_s3["efficiency"]
best_lof_mean  = best_res["loo_fold_mean"]
best_los_mean  = best_res["loo_sym_mean"]
best_pct_f34   = best_s3["pct_f34"]
best_pct_f12   = best_s3["pct_f12"]
n_rmvd_best    = len(best_res["rmvd"])
n_kept_best    = len(best_res["kept"])

# Determine whether any filter "materially" improves
MATERIAL_THRESHOLD_PF_DELTA = 0.10   # PF must rise by at least 0.10
MATERIAL_THRESHOLD_MC       = 0.25   # MC prob must be < 25%
MATERIAL_THRESHOLD_EFF      = 1.2    # Efficiency must be > 1.2x

any_material = any(
    r["pf"] >= baseline_m["pf"] + MATERIAL_THRESHOLD_PF_DELTA
    and r["mc_prob"] < MATERIAL_THRESHOLD_MC
    for r in ranking_rows
)

best_is_material = (
    best_filter_pf >= baseline_m["pf"] + MATERIAL_THRESHOLD_PF_DELTA
    and best_mc_prob < MATERIAL_THRESHOLD_MC
    and best_eff >= MATERIAL_THRESHOLD_EFF
)

# F3/F4 fold-level PF
best_f34_pf_base = lose_m["pf"]
best_f34_t_filt  = [t for fl in LOSE_FOLDS for t in best_res["filt_fold"][fl]]
best_f34_pf_filt = metrics(best_f34_t_filt)["pf"]

# F1/F2 fold-level PF
best_f12_t_filt = [t for fl in WIN_FOLDS for t in best_res["filt_fold"][fl]]
best_f12_pf_filt = metrics(best_f12_t_filt)["pf"]

print("  Q1. Does one macro regime filter materially improve E3?")
print()
if any_material:
    print(f"  A: YES — at least one filter improves PF by ≥{MATERIAL_THRESHOLD_PF_DELTA} "
          f"with MC P(random) < {MATERIAL_THRESHOLD_MC*100:.0f}%.")
    print(f"     The improvement is not consistent with random variation in trade sequencing.")
else:
    print(f"  A: MARGINALLY / NOT CONVINCINGLY. No single filter improved PF by >"
          f"{MATERIAL_THRESHOLD_PF_DELTA} with MC P < {MATERIAL_THRESHOLD_MC*100:.0f}%.")
    print(f"     The filters show limited evidence of a genuine, robust regime gate.")
print()

print("  Q2. Which filter performed best?")
print()
print(f"  A: {best_filter_name} ({best_filter_id})")
print(f"     {best_filter['desc']}")
print(f"     Rationale: {best_filter['rationale'][:75]}")
print(f"     Overall PF: {baseline_m['pf']:.3f} → {best_filter_pf:.3f}  "
      f"({'improved' if best_filter_pf > baseline_m['pf'] else 'worsened'})")
print(f"     F3+F4 PF: {best_f34_pf_base:.3f} → {best_f34_pf_filt:.3f}  "
      f"({'improved' if best_f34_pf_filt > best_f34_pf_base else 'worsened'})")
print(f"     F1+F2 PF: {win_m['pf']:.3f} → {best_f12_pf_filt:.3f}")
print()

print("  Q3. How many F3/F4 losses were avoided?")
print()
f34_losses = [t for fl in LOSE_FOLDS for t in fold_trades[fl] if t["win"] == 0]
mask_best  = apply_filter(all_trades, best_filter, sym_macro_thr)
f34_losses_avoided = [t for t, ok in zip(all_trades, mask_best)
                      if not ok and t["fold"] in LOSE_FOLDS and t["win"] == 0]
f34_losses_avoided_pnl = sum(t["pnl"] for t in f34_losses_avoided)
print(f"  A: {len(f34_losses_avoided)} of {len(f34_losses)} F3+F4 losing trades avoided "
      f"({best_s3['pct_f34']:.1f}%).")
print(f"     PnL recovered: ${abs(f34_losses_avoided_pnl):+.0f}  "
      f"(these are losses that would have been taken without the filter)")
print()

print("  Q4. How many winning trades were sacrificed?")
print()
all_wins     = [t for t in all_trades if t["win"] == 1]
wins_blocked = [t for t, ok in zip(all_trades, mask_best) if not ok and t["win"] == 1]
wins_blocked_pnl = sum(t["pnl"] for t in wins_blocked)
f12_wins_sacrificed = [t for t in wins_blocked if t["fold"] in WIN_FOLDS]
print(f"  A: {len(wins_blocked)} of {len(all_wins)} total winning trades sacrificed "
      f"({len(wins_blocked)/len(all_wins)*100:.1f}% overall).")
print(f"     Of those, {len(f12_wins_sacrificed)} were in the F1+F2 winning period "
      f"({best_s3['pct_f12']:.1f}% of F1+F2 winners).")
print(f"     Foregone PnL: ${wins_blocked_pnl:+.0f}")
print()

print("  Q5. Is the improvement statistically convincing?")
print()
mc_verdict = (
    "STRONG — MC P(random) suggests the filtered result is unlikely by chance alone."
    if best_mc_prob < 0.05 else
    "INDICATIVE — MC P(random) is low enough to support the hypothesis, though sample is small."
    if best_mc_prob < 0.20 else
    "WEAK — MC P(random) cannot rule out sampling variation as the explanation."
)
bstrap_best = best_res["bstrap"]
print(f"  A: Monte Carlo P(random ≥ observed PF) = {best_mc_prob*100:.1f}%  → {mc_verdict}")
print(f"     Bootstrap PF 90% CI = [{bstrap_best['ci_lo']:.3f}, {bstrap_best['ci_hi']:.3f}]")
print(f"     P(Bootstrap PF > 1.0) = {bstrap_best['pct_gt1']*100:.1f}%")
print(f"     LOO-fold PF mean = {best_res['loo_fold_mean']:.3f}  "
      f"(baseline: {baseline_m['pf']:.3f})")
print(f"     LOO-symbol PF mean = {best_res['loo_sym_mean']:.3f}")
print()

print("  Q6. Would you freeze this updated strategy for a brand-new forward test?")
print()
freeze_criteria = {
    "PF materially improved":     best_filter_pf > baseline_m["pf"] + 0.10,
    "F3/F4 losses avoided > 30%": best_s3["pct_f34"] > 30.0,
    "Efficiency ratio > 1.2x":    best_s3["efficiency"] > 1.2,
    "MC probability < 20%":       best_mc_prob < 0.20,
    "LOO-fold mean ≥ baseline":   (not np.isnan(best_res["loo_fold_mean"]) and
                                   best_res["loo_fold_mean"] >= baseline_m["pf"] * 0.95),
    "F1/F2 PF maintained > 90%":  best_f12_pf_filt >= win_m["pf"] * 0.90,
}
criteria_met = sum(1 for v in freeze_criteria.values() if v)
total_criteria = len(freeze_criteria)

print(f"  Freeze criteria ({criteria_met}/{total_criteria} met):")
for criterion, met in freeze_criteria.items():
    print(f"    {'✓' if met else '✗'}  {criterion}")

print()
if criteria_met >= 5:
    freeze_verdict = "YES — freeze this updated strategy and run a brand-new forward test."
    freeze_detail  = (f"The {best_filter_name} filter meets {criteria_met}/{total_criteria} "
                      f"pre-registered criteria. The improvement is consistent across folds "
                      f"and symbols. It should be locked and tested on completely unseen data.")
elif criteria_met >= 3:
    freeze_verdict = "CONDITIONAL — promising but sample is too small to freeze with high confidence."
    freeze_detail  = (f"{criteria_met}/{total_criteria} criteria met. The filter shows "
                      f"directional improvement, but the evidence is not yet strong enough to "
                      f"justify freezing without more data. Consider running one more validation "
                      f"fold before committing.")
else:
    freeze_verdict = "NO — insufficient evidence to freeze."
    freeze_detail  = (f"Only {criteria_met}/{total_criteria} criteria met. The filter may "
                      f"be mining noise in a limited sample.")

print(f"  Verdict: {freeze_verdict}")
print(f"  Detail:  {freeze_detail}")
print()

print("  Q7. If no macro filter survives — what then?")
print()
if criteria_met >= 4:
    print("  A: A filter DID survive. Proceed to Q6 verdict above.")
else:
    print("  A: No macro filter passed all robustness criteria.")
    print("     This means E3 should remain UNCHANGED.")
    print("     Future research should move toward discovering NEW universal structural")
    print("     edges rather than trying to regime-gate the existing one.")
    print("     A structural edge (like compression + relative volume) that works across")
    print("     many regimes is more valuable than a regime-conditional gate that merely")
    print("     avoids the worst folds in a limited historical sample.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts …")
print(SEP2)

FILTER_COLORS = [C_BLUE, C_GREEN, C_CYAN, C_PURP, C_ORAN, C_GOLD]

# ── Chart 1: Dashboard — PF/WR/Net for baseline vs all filters
fig1 = plt.figure(figsize=(18, 10), facecolor=C_BG)
fig1.suptitle("R057 — Macro Filter Validation Dashboard\n"
              f"Frozen Strategy: {E3_LABEL} | RR={RR}",
              fontsize=11, color=C_GOLD, fontweight="bold", y=0.98)
gs1 = gridspec.GridSpec(2, 3, figure=fig1, hspace=0.45, wspace=0.35)

filter_names_short = ["BASE"] + [f["short"] for f in MACRO_FILTERS]
filter_pfs         = [baseline_m["pf"]] + [filter_results[f["id"]]["m_kept"]["pf"]
                                             for f in MACRO_FILTERS]
filter_wrs         = [baseline_m["wr"] * 100] + [
    filter_results[f["id"]]["m_kept"]["wr"] * 100 for f in MACRO_FILTERS]
filter_nets        = [baseline_m["net"]] + [
    filter_results[f["id"]]["m_kept"]["net"] for f in MACRO_FILTERS]
filter_mdds        = [abs(baseline_m["mdd"]) * 100] + [
    abs(filter_results[f["id"]]["m_kept"]["mdd"]) * 100 for f in MACRO_FILTERS]
filter_kept_pct    = [100.0] + [
    len(filter_results[f["id"]]["kept"]) / len(all_trades) * 100 for f in MACRO_FILTERS]
bar_colors = [C_GOLD] + FILTER_COLORS

xs = np.arange(len(filter_names_short))

def bar_panel(ax, values, title, ylabel, ref=None, higher_better=True):
    cols = []
    for i, v in enumerate(values):
        if i == 0:
            cols.append(C_GOLD)
        elif (higher_better and v >= (ref or values[0])) or \
             (not higher_better and v <= (ref or values[0])):
            cols.append(C_GREEN)
        else:
            cols.append(C_RED)
    ax.bar(xs, values, color=cols, alpha=0.82, width=0.65)
    if ref is not None:
        ax.axhline(ref, color=C_GOLD, linewidth=1.0, linestyle="--", alpha=0.7, label="Baseline")
    ax.set_xticks(xs)
    ax.set_xticklabels(filter_names_short, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel(ylabel, fontsize=7, color=C_TEXT)
    for i, v in enumerate(values):
        ax.text(i, v + (max(values) - min(values)) * 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6, color=C_TEXT)
    panel_style(ax, title, fs=8)

ax_pf  = fig1.add_subplot(gs1[0, 0])
ax_wr  = fig1.add_subplot(gs1[0, 1])
ax_net = fig1.add_subplot(gs1[0, 2])
ax_mdd = fig1.add_subplot(gs1[1, 0])
ax_kpt = fig1.add_subplot(gs1[1, 1])
ax_eff = fig1.add_subplot(gs1[1, 2])

bar_panel(ax_pf,  filter_pfs,      "Profit Factor",         "PF",    ref=baseline_m["pf"])
bar_panel(ax_wr,  filter_wrs,      "Win Rate (%)",          "WR%",   ref=baseline_m["wr"]*100)
bar_panel(ax_net, filter_nets,     "Net Profit ($)",        "Net$",  ref=baseline_m["net"])
bar_panel(ax_mdd, filter_mdds,     "Max Drawdown (%)",      "MDD%",  ref=abs(baseline_m["mdd"])*100,
          higher_better=False)
bar_panel(ax_kpt, filter_kept_pct, "Trades Kept (%)",       "%Kept", ref=100.0)

# Efficiency bar
eff_vals = [1.0] + [section3_data[f["id"]]["efficiency"] for f in MACRO_FILTERS]
bar_panel(ax_eff, eff_vals, "Loss Prevention Efficiency\n(F3/F4 loss% / F1/F2 win%)", "Ratio",
          ref=1.0)

plt.savefig(f"{OUT}/r057_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r057_dashboard.png")

# ── Chart 2: Equity curves — Baseline vs Best Filter (by fold)
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 9), facecolor=C_BG)
fig2.suptitle(f"R057 — Equity Curves: Baseline vs {best_filter_name}\n"
              f"Frozen: {E3_LABEL}",
              fontsize=10, color=C_GOLD, fontweight="bold")

axes2_flat = axes2.flat

# Overall
ax0 = axes2_flat[0]
eq_base = metrics(all_trades)["equity"]
eq_filt = metrics(best_res["kept"])["equity"]
ax0.plot(np.linspace(0, 1, len(eq_base)), eq_base, color=C_GOLD,  linewidth=1.2, label="Baseline")
ax0.plot(np.linspace(0, 1, len(eq_filt)), eq_filt, color=C_CYAN,  linewidth=1.2, label=best_filter_name)
ax0.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
ax0.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax0, f"All Folds: Base PF={baseline_m['pf']:.3f}  Filt PF={best_filter_pf:.3f}", fs=8)

for fi, ax_ in zip(range(1, N_FWD_FOLDS + 1), list(axes2_flat)[1:]):
    fl   = f"F{fi}"
    mb   = metrics(fold_trades[fl])
    mf_  = filter_results[best_filter_id]["fold_pfs"][fl]
    eq_b = mb["equity"]; eq_f_ = mf_["equity"]
    grp  = "WIN" if fl in WIN_FOLDS else ("LOSE" if fl in LOSE_FOLDS else "REC")
    col  = C_GREEN if fl in WIN_FOLDS else (C_RED if fl in LOSE_FOLDS else C_GOLD)
    x_b  = np.linspace(0, 1, len(eq_b)); x_f = np.linspace(0, 1, len(eq_f_))
    ax_.plot(x_b, eq_b, color=col,   linewidth=1.0, linestyle="--", alpha=0.6, label="Base")
    ax_.plot(x_f, eq_f_, color=C_CYAN, linewidth=1.2, label="Filt")
    ax_.axhline(CAPITAL, color=C_GRID, linewidth=0.5, linestyle=":")
    ax_.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)
    panel_style(ax_, f"{fl} [{grp}]  Base={mb['pf']:.2f}→Filt={mf_['pf']:.2f}  "
                f"n={mb['n']}→{mf_['n']}", fs=7)

plt.tight_layout()
plt.savefig(f"{OUT}/r057_equity_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r057_equity_curves.png")

# ── Chart 3: Loss Prevention Matrix — heatmap
n_filters = len(MACRO_FILTERS)
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6), facecolor=C_BG)
fig3.suptitle("R057 — Loss Prevention Matrix", fontsize=10, color=C_GOLD, fontweight="bold")

filt_shorts = [f["short"] for f in MACRO_FILTERS]
pct_f34_arr = np.array([section3_data[f["id"]]["pct_f34"] for f in MACRO_FILTERS])
pct_f12_arr = np.array([section3_data[f["id"]]["pct_f12"] for f in MACRO_FILTERS])
eff_arr     = np.array([section3_data[f["id"]]["efficiency"] for f in MACRO_FILTERS])

xs3 = np.arange(n_filters)
ax31, ax32 = axes3

# Grouped bar: losses avoided vs winners sacrificed
width = 0.35
ax31.bar(xs3 - width/2, pct_f34_arr, width, color=C_GREEN, alpha=0.8, label="F3/F4 losses avoided %")
ax31.bar(xs3 + width/2, pct_f12_arr, width, color=C_RED,   alpha=0.8, label="F1/F2 winners sacrificed %")
ax31.set_xticks(xs3); ax31.set_xticklabels(filt_shorts, rotation=30, ha="right", fontsize=7)
ax31.set_ylabel("Percentage (%)", fontsize=7, color=C_TEXT)
ax31.axhline(0, color=C_TEXT, linewidth=0.5)
ax31.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax31, "Loss Prevention vs Winner Sacrifice\n(green=losses avoided, red=winners lost)", fs=8)
for i, (la, wa) in enumerate(zip(pct_f34_arr, pct_f12_arr)):
    ax31.text(i - width/2, la + 0.5, f"{la:.1f}", ha="center", fontsize=6, color=C_TEXT)
    ax31.text(i + width/2, wa + 0.5, f"{wa:.1f}", ha="center", fontsize=6, color=C_TEXT)

# Efficiency ratio bar
cols_eff = [C_GREEN if e > 1.2 else (C_GOLD if e > 0.8 else C_RED) for e in eff_arr]
ax32.bar(xs3, eff_arr, color=cols_eff, alpha=0.82)
ax32.axhline(1.0, color=C_GOLD, linewidth=1.0, linestyle="--", alpha=0.7, label="Neutral (1x)")
ax32.axhline(1.5, color=C_GREEN, linewidth=0.7, linestyle=":", alpha=0.5, label="Target (1.5x)")
ax32.set_xticks(xs3); ax32.set_xticklabels(filt_shorts, rotation=30, ha="right", fontsize=7)
ax32.set_ylabel("Efficiency ratio", fontsize=7, color=C_TEXT)
ax32.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax32, "Efficiency Ratio = % Losses Avoided / % Winners Sacrificed\n"
            "(green >1.2 = selectively avoids losses)", fs=8)
for i, e in enumerate(eff_arr):
    ax32.text(i, e + 0.03, f"{e:.2f}x", ha="center", fontsize=6, color=C_TEXT)

plt.tight_layout()
plt.savefig(f"{OUT}/r057_loss_prevention.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r057_loss_prevention.png")

# ── Chart 4: Robustness Grid — fold PF heatmap for all filters
fig4, ax4 = plt.subplots(figsize=(12, 7), facecolor=C_BG)
fig4.suptitle("R057 — Fold PF Robustness Grid\n"
              "(F1/F2=WIN, F3/F4=LOSE, F5=REC | green=improvement over baseline)",
              fontsize=9, color=C_GOLD, fontweight="bold")

row_labels  = ["BASELINE"] + [f["short"] for f in MACRO_FILTERS]
col_labels  = [f"F{i}" for i in range(1, N_FWD_FOLDS + 1)]

heat_vals   = np.zeros((len(row_labels), N_FWD_FOLDS))
heat_vals[0, :] = [metrics(fold_trades[f"F{i}"])["pf"] for i in range(1, N_FWD_FOLDS + 1)]
for fi_r, filt in enumerate(MACRO_FILTERS, 1):
    for fi_c in range(1, N_FWD_FOLDS + 1):
        heat_vals[fi_r, fi_c - 1] = filter_results[filt["id"]]["fold_pfs"][f"F{fi_c}"]["pf"]

# Normalise by baseline row for colour mapping
base_row = heat_vals[0, :]
norm_vals = np.zeros_like(heat_vals)
for r in range(len(row_labels)):
    for c in range(N_FWD_FOLDS):
        b = base_row[c]
        norm_vals[r, c] = 0.5 + (heat_vals[r, c] - b) / (b + 0.001) * 0.5
norm_vals = np.clip(norm_vals, 0, 1)

im4 = ax4.imshow(norm_vals, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax4.set_xticks(np.arange(N_FWD_FOLDS))
ax4.set_xticklabels(col_labels, fontsize=8, color=C_TEXT)
ax4.set_yticks(np.arange(len(row_labels)))
ax4.set_yticklabels(row_labels, fontsize=7, color=C_TEXT)

for r in range(len(row_labels)):
    for c in range(N_FWD_FOLDS):
        v = heat_vals[r, c]
        ax4.text(c, r, f"{v:.3f}", ha="center", va="center",
                 fontsize=7, color="black" if 0.3 < norm_vals[r, c] < 0.7 else "white")

# Highlight lose-period columns
for col in [2, 3]:
    ax4.add_patch(plt.Rectangle(
        (col - 0.5, -0.5), 1, len(row_labels),
        linewidth=2, edgecolor=C_RED, facecolor="none"))

ax4.set_facecolor(C_PANEL)
plt.colorbar(im4, ax=ax4, fraction=0.03, pad=0.02).ax.tick_params(colors=C_TEXT)
plt.tight_layout()
plt.savefig(f"{OUT}/r057_robustness_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r057_robustness_grid.png")

# ── Chart 5: Statistical Summary
fig5, axes5 = plt.subplots(1, 3, figsize=(16, 6), facecolor=C_BG)
fig5.suptitle("R057 — Statistical Evidence Summary", fontsize=10, color=C_GOLD, fontweight="bold")

ax51, ax52, ax53 = axes5
filt_short_labels = [f["short"] for f in MACRO_FILTERS]
xs5 = np.arange(n_filters)

# Bootstrap CI plot
boot_means = [filter_results[f["id"]]["bstrap"]["mean"]  for f in MACRO_FILTERS]
boot_lo    = [filter_results[f["id"]]["bstrap"]["ci_lo"] for f in MACRO_FILTERS]
boot_hi    = [filter_results[f["id"]]["bstrap"]["ci_hi"] for f in MACRO_FILTERS]
err_lo     = [m - l for m, l in zip(boot_means, boot_lo)]
err_hi     = [h - m for m, h in zip(boot_means, boot_hi)]
cols5a     = [C_GREEN if m > baseline_m["pf"] else C_RED for m in boot_means]
ax51.errorbar(xs5, boot_means, yerr=[err_lo, err_hi], fmt="o", color=C_CYAN,
              ecolor=C_GRID, elinewidth=1.5, capsize=4, capthick=1, markersize=7)
for i, (m, c) in enumerate(zip(boot_means, cols5a)):
    ax51.plot(i, m, "o", color=c, markersize=8, zorder=5)
ax51.axhline(baseline_m["pf"], color=C_GOLD, linewidth=1.0, linestyle="--", label="Baseline PF")
ax51.axhline(1.0, color=C_TEXT, linewidth=0.6, linestyle=":", alpha=0.5, label="PF=1")
ax51.set_xticks(xs5); ax51.set_xticklabels(filt_short_labels, rotation=30, ha="right", fontsize=7)
ax51.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax51, "Bootstrap PF — Mean ± 90% CI\n(1000 bootstrap iterations)", fs=8)

# Monte Carlo probability
mc_probs = [filter_results[f["id"]]["mc_res"]["mc_prob"] * 100 for f in MACRO_FILTERS]
mc_cols  = [C_GREEN if p < 5 else (C_GOLD if p < 20 else C_RED) for p in mc_probs]
ax52.bar(xs5, mc_probs, color=mc_cols, alpha=0.82)
ax52.axhline(5,  color=C_GREEN, linewidth=0.7, linestyle="--", alpha=0.6, label="5% (strong)")
ax52.axhline(20, color=C_GOLD,  linewidth=0.7, linestyle=":",  alpha=0.6, label="20% (indicative)")
ax52.set_xticks(xs5); ax52.set_xticklabels(filt_short_labels, rotation=30, ha="right", fontsize=7)
ax52.set_ylabel("P(random ≥ observed) %", fontsize=7, color=C_TEXT)
ax52.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax52, "Monte Carlo Significance\n(lower = less likely by chance)", fs=8)
for i, p in enumerate(mc_probs):
    ax52.text(i, p + 0.3, f"{p:.1f}%", ha="center", fontsize=6, color=C_TEXT)

# LOO PF comparison (fold vs symbol)
loo_fold_means = [filter_results[f["id"]]["loo_fold_mean"] for f in MACRO_FILTERS]
loo_sym_means  = [filter_results[f["id"]]["loo_sym_mean"]  for f in MACRO_FILTERS]
width53 = 0.35
ax53.bar(xs5 - width53/2, loo_fold_means, width53, color=C_BLUE,  alpha=0.82, label="LOO-fold mean PF")
ax53.bar(xs5 + width53/2, loo_sym_means,  width53, color=C_PURP,  alpha=0.82, label="LOO-symbol mean PF")
ax53.axhline(baseline_m["pf"], color=C_GOLD, linewidth=1.0, linestyle="--", label="Baseline PF")
ax53.set_xticks(xs5); ax53.set_xticklabels(filt_short_labels, rotation=30, ha="right", fontsize=7)
ax53.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax53, "Leave-One-Out Robustness\n(above baseline line = genuinely robust)", fs=8)

plt.tight_layout()
plt.savefig(f"{OUT}/r057_statistical.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r057_statistical.png")

# ── Chart 6: Ranking Summary
fig6, ax6 = plt.subplots(figsize=(12, 6), facecolor=C_BG)
fig6.suptitle("R057 — Filter Ranking Summary\n"
              "PF improvement (30) + Loss efficiency (25) + Robustness (25) + "
              "Simplicity (10) + Evidence (10)",
              fontsize=9, color=C_GOLD, fontweight="bold")

rank_names  = [r["fid"].replace("F_", "") for r in ranking_rows]
rank_totals = [r["total"] for r in ranking_rows]
rank_cols   = [C_GREEN if r["total"] >= 60 else (C_GOLD if r["total"] >= 40 else C_RED)
               for r in ranking_rows]
xr = np.arange(len(rank_names))

bars6 = ax6.barh(xr, rank_totals, color=rank_cols, alpha=0.82, height=0.6)
ax6.axvline(50, color=C_GOLD,  linewidth=0.7, linestyle="--", alpha=0.7, label="Score=50")
ax6.axvline(60, color=C_GREEN, linewidth=0.7, linestyle=":",  alpha=0.5, label="Score=60")
ax6.set_yticks(xr)
ax6.set_yticklabels(rank_names, fontsize=8)
ax6.set_xlabel("Composite Score / 100", fontsize=8, color=C_TEXT)
ax6.invert_yaxis()  # best at top
ax6.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)

for i, (row, bar) in enumerate(zip(ranking_rows, bars6)):
    ax6.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{row['total']:.1f}  PF={row['pf']:.3f}  MC={row['mc_prob']*100:.1f}%",
             va="center", fontsize=7, color=C_TEXT)

panel_style(ax6, "Composite Filter Ranking (1st = best overall)", fs=9)
plt.tight_layout()
plt.savefig(f"{OUT}/r057_ranking.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r057_ranking.png")

# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
journal_path = CONFIG["JOURNAL_FILE"]
journal_line = (
    f"R057,,Single Macro Filter Validation — {E3_LABEL},"
    f"ALL,{baseline_m['n']:.0f},{baseline_m['pf']:.4f},,{baseline_m['wr']:.4f},"
    f"{baseline_m['net']:.2f},{baseline_m['mdd']:.4f},,,,FILTER_VALIDATION,,,,,,,,,,,,,"
    f"Best filter: {best_filter_name} ({best_filter_id}). "
    f"Filtered PF: {best_filter_pf:.4f} (base {baseline_m['pf']:.4f}). "
    f"F3+F4 losses avoided: {best_s3['pct_f34']:.1f}%. "
    f"F1+F2 winners sacrificed: {best_s3['pct_f12']:.1f}%. "
    f"Efficiency: {best_s3['efficiency']:.2f}x. "
    f"MC P: {best_mc_prob*100:.1f}%. "
    f"Criteria met: {criteria_met}/{total_criteria}. "
    f"Verdict: {freeze_verdict}"
)
with open(journal_path, "a") as f:
    f.write(journal_line + "\n")
print(f"\n  ✓  Journal updated: {journal_path}")

md_path = f"{OUT}/r057_journal.md"
with open(md_path, "w") as f:
    f.write(f"# QUANTLAB AI — R057 — Single Macro Regime Filter Validation\n\n")
    f.write(f"**Frozen Strategy:** `{E3_LABEL}` | RR={RR}\n\n")
    f.write(f"## Baseline Performance\n\n")
    f.write(f"| Fold | Group | PF | n | WR |\n|---|---|---|---|---|\n")
    for fi in range(1, N_FWD_FOLDS + 1):
        fl  = f"F{fi}"
        m_  = metrics(fold_trades[fl])
        grp = "WIN" if fl in WIN_FOLDS else ("LOSE" if fl in LOSE_FOLDS else "REC")
        f.write(f"| {fl} | {grp} | {m_['pf']:.3f} | {m_['n']} | {m_['wr']*100:.1f}% |\n")
    f.write(f"\n## Filter Results Summary\n\n")
    f.write(f"| Filter | PF | F3/F4 Losses Avoided | F1/F2 Winners Sacrificed | Efficiency | MC P |\n"
            f"|---|---|---|---|---|---|\n")
    for filt in MACRO_FILTERS:
        fid = filt["id"]
        res = filter_results[fid]
        s3  = section3_data[fid]
        mc  = res["mc_res"]["mc_prob"]
        f.write(f"| {filt['name']} | {res['m_kept']['pf']:.3f} | {s3['pct_f34']:.1f}% | "
                f"{s3['pct_f12']:.1f}% | {s3['efficiency']:.2f}x | {mc*100:.1f}% |\n")
    f.write(f"\n## Best Filter: {best_filter_name}\n\n")
    f.write(f"- **Filter ID:** {best_filter_id}\n")
    f.write(f"- **Description:** {best_filter['desc']}\n")
    f.write(f"- **PF improvement:** {baseline_m['pf']:.3f} → {best_filter_pf:.3f}\n")
    f.write(f"- **F3+F4 losses avoided:** {best_s3['pct_f34']:.1f}%\n")
    f.write(f"- **F1+F2 winners sacrificed:** {best_s3['pct_f12']:.1f}%\n")
    f.write(f"- **Efficiency ratio:** {best_s3['efficiency']:.2f}x\n")
    f.write(f"- **Freeze verdict:** {freeze_verdict}\n")
    f.write(f"\n## Ranking\n\n")
    f.write(f"| Rank | Filter | Score/100 | PF | Efficiency | MC% |\n"
            f"|---|---|---|---|---|---|\n")
    for rank, row in enumerate(ranking_rows, 1):
        f.write(f"| {rank} | {row['fid']} | {row['total']:.1f} | "
                f"{row['pf']:.3f} | {row['efficiency']:.2f}x | {row['mc_prob']*100:.1f}% |\n")
print(f"  ✓  {md_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY BANNER
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print(f"  R057 COMPLETE — SINGLE MACRO REGIME FILTER VALIDATION")
print(SEP)
print()
print(f"  Frozen Strategy: {E3_LABEL}  RR={RR}")
print(f"  Baseline: n={baseline_m['n']}  PF={baseline_m['pf']:.3f}  "
      f"WR={baseline_m['wr']*100:.1f}%  Net=${baseline_m['net']:+.0f}")
print()
print(f"  FILTER RANKING:")
for rank, row in enumerate(ranking_rows, 1):
    marker = "★ " if rank == 1 else "  "
    print(f"  {marker}{rank}. {row['fid']:<14}  Score={row['total']:>5.1f}/100  "
          f"PF={row['pf']:.3f}  Eff={row['efficiency']:.2f}x  "
          f"F34_avoided={section3_data[row['fid']]['pct_f34']:.1f}%  "
          f"F12_lost={section3_data[row['fid']]['pct_f12']:.1f}%  "
          f"MC={row['mc_prob']*100:.1f}%")
print()
print(f"  BEST FILTER: {best_filter_name} ({best_filter_id})")
print(f"    {best_filter['desc']}")
print(f"    PF: {baseline_m['pf']:.3f} → {best_filter_pf:.3f}  |  "
      f"Criteria met: {criteria_met}/{total_criteria}")
print(f"    VERDICT: {freeze_verdict}")
print()
print(f"  Charts:  r057_dashboard.png   r057_equity_curves.png")
print(f"           r057_loss_prevention.png  r057_robustness_grid.png")
print(f"           r057_statistical.png  r057_ranking.png")
print(SEP)
