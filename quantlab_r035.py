"""
=============================================================================
QUANTLAB AI — RESEARCH #035
Universal Edge Discovery — What Separates Winners From Losers?
=============================================================================

Objective: Ignore strategy names. Pool every available trade (R002–R034).
           Enrich each trade with 15+ market-condition features at entry time.
           Search for the market environments that consistently produce wins.

Methods:
  1  Data assembly & feature enrichment (from 1H cache)
  2  Single-feature bucket analysis (quintiles, bootstrap PF)
  3  Two-feature interaction grids
  4  Three-feature combination search
  5  Decision tree rules (max_depth=4)
  6  Random Forest feature importance + permutation importance
  7  SHAP explanations (TreeExplainer)
  8  K-means clustering (k=8)
  9  Association rule mining (manual, no dependencies)
  10 Top-20 environments ranked by PF × n × bootstrap stability

Report only conditions with: n ≥ 30 (per-cell), bootstrap PF > 1.10,
statistically significant (binomial p < 0.05).
=============================================================================
"""

import os, sys, math, warnings, glob, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import mutual_info_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("  [WARN] sklearn not available — tree/RF/clustering sections skipped")

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

RESEARCH_ID = "R035"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = CONFIG["RISK_REWARD"]
BEP_WR   = 1.0 / (1.0 + RR)          # 33.33%

# Minimum n per condition to report
MIN_N     = 30
BOOT_ITER = 3_000

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #035" + " "*50 + "║")
print("║  Universal Edge Discovery — What Separates Winners From Losers?" + " "*14 + "║")
print("╚" + "═"*79 + "╝")

# =============================================================================
# SECTION 1 — LOAD & STANDARDISE ALL TRADE LOGS
# =============================================================================

print("\n" + "─"*78)
print("  SECTION 1 — Loading & standardising trade logs")
print("─"*78)

def _parse_time(s):
    if pd.isna(s):
        return pd.NaT
    try:
        ts = pd.to_datetime(str(s), utc=True)
        return ts
    except Exception:
        return pd.NaT

def _load_log(path, sym_override=None):
    """Load one CSV → standardised df with [sym, entry_time, pnl, win, source]."""
    df = pd.read_csv(path)
    out = pd.DataFrame()

    # Resolve symbol
    if sym_override:
        out["sym"] = sym_override
    elif "sym" in df.columns:
        out["sym"] = df["sym"].astype(str)
    elif "symbol" in df.columns:
        out["sym"] = df["symbol"].astype(str)
    else:
        return None  # can't determine symbol

    # entry_time
    et_col = next((c for c in ["entry_time","signal_time"] if c in df.columns), None)
    if et_col is None:
        return None
    out["entry_time"] = df[et_col].apply(_parse_time)

    # pnl / win
    if "pnl" not in df.columns or "win" not in df.columns:
        return None
    out["pnl"]  = pd.to_numeric(df["pnl"],  errors="coerce")
    out["win"]  = pd.to_numeric(df["win"],  errors="coerce").astype(int)

    # optional passthrough fields (will be re-enriched but useful for validation)
    for col in ["r_multiple","atr_rank_pct","adx14","rsi14","ema200_slope_pct",
                "dist_from_ema200_pct","bb_width","realized_vol","rel_vol",
                "hour_utc","day_of_week","session","regime","strategy",
                "exit_type","holding_hrs","holding_mins","fold"]:
        if col in df.columns:
            out[col] = df[col].values

    out["source"] = os.path.basename(path)
    return out.dropna(subset=["entry_time","pnl","win"])

# Per-symbol early logs (R002–R004)
EARLY_SYM_MAP = {
    "BTC_USDT_SWAP_r002_trade_log.csv": "BTC-USDT-SWAP",
    "BTC_USDT_SWAP_r003_trade_log.csv": "BTC-USDT-SWAP",
    "BTC_USDT_SWAP_r004_trade_log.csv": "BTC-USDT-SWAP",
    "BTC_USDT_SWAP_trade_log.csv":      "BTC-USDT-SWAP",
    "ETH_USDT_SWAP_r002_trade_log.csv": "ETH-USDT-SWAP",
    "ETH_USDT_SWAP_r003_trade_log.csv": "ETH-USDT-SWAP",
    "ETH_USDT_SWAP_r004_trade_log.csv": "ETH-USDT-SWAP",
    "ETH_USDT_SWAP_trade_log.csv":      "ETH-USDT-SWAP",
    "SOL_USDT_SWAP_r002_trade_log.csv": "SOL-USDT-SWAP",
    "SOL_USDT_SWAP_r003_trade_log.csv": "SOL-USDT-SWAP",
    "SOL_USDT_SWAP_r004_trade_log.csv": "SOL-USDT-SWAP",
    "SOL_USDT_SWAP_trade_log.csv":      "SOL-USDT-SWAP",
}

# Multi-symbol / rich logs (include all)
MULTI_LOGS = [
    "r005_attribution_trades.csv",
    "r022_trade_log.csv",
    "r023_master_trades.csv",
    "r024_trade_log.csv",
    "r026_trade_log_liqsweep.csv",
    "r027_fvg_slope_low_atr_trades.csv",
    "r027_liqsweep_low_atr_trades.csv",
    "r029_fvg_low_atr_1h_9sym_trades.csv",
    "r030_enriched_trades.csv",
    "r032_all_variants_trades.csv",
    "r033_trade_log.csv",
    "r034_trade_log.csv",
]

all_parts = []

for fname, sym in EARLY_SYM_MAP.items():
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        part = _load_log(path, sym_override=sym)
        if part is not None and len(part):
            all_parts.append(part)

for fname in MULTI_LOGS:
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        part = _load_log(path)
        if part is not None and len(part):
            all_parts.append(part)

raw = pd.concat(all_parts, ignore_index=True)

# De-duplicate: same sym + entry_time + win
raw = (raw
       .drop_duplicates(subset=["sym","entry_time","win"])
       .sort_values("entry_time")
       .reset_index(drop=True))

# Only keep symbols whose 1H cache exists
cache_syms = set(
    f.replace(CACHE+"/","").replace("_1H.parquet","").replace("_","-",2)
    for f in glob.glob(f"{CACHE}/*_1H.parquet")
)
raw = raw[raw["sym"].isin(cache_syms)].reset_index(drop=True)

print(f"\n  Loaded {len(raw):,} unique trades across {raw['sym'].nunique()} symbols")
print(f"  Win rate: {raw['win'].mean()*100:.1f}%  |  "
      f"PF: {raw[raw.pnl>0].pnl.sum()/max(abs(raw[raw.pnl<0].pnl.sum()),1e-9):.3f}")
print(f"  Sources: {raw['source'].nunique()} files")
print(f"  Date range: {raw.entry_time.min().date()} → {raw.entry_time.max().date()}")

# =============================================================================
# SECTION 2 — LOAD CACHE & COMPUTE FEATURE MATRIX
# =============================================================================

print("\n" + "─"*78)
print("  SECTION 2 — Loading cache & computing feature matrices")
print("─"*78)

def _calc_rsi(close, length=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/length, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

def _calc_adx(df, length=14):
    prev_c   = df["close"].shift(1)
    tr       = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    up   = df["high"] - df["high"].shift(1)
    down = df["low"].shift(1) - df["low"]
    plus_dm  = np.where((up > down) & (up > 0),   up,   0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    alpha = 1.0 / length
    sm_tr    = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di  = 100.0 * pd.Series(plus_dm,  index=df.index).ewm(alpha=alpha, adjust=False).mean() / sm_tr.replace(0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / sm_tr.replace(0, np.nan)
    di_sum   = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100.0 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)

def build_feature_index(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features on full price series. Returns df indexed by datetime."""
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]

    ema200        = calc_ema(c, 200)
    atr14         = calc_atr(df, 14)
    atr_rank      = atr14.rolling(100).rank(pct=True) * 100

    log_ret       = np.log(c / c.shift(1))
    real_vol      = log_ret.rolling(20).std() * math.sqrt(24 * 365)  # annualised

    vol_ma        = v.rolling(20).mean()
    rel_vol       = v / vol_ma.replace(0, np.nan)

    bb_mid        = c.rolling(20).mean()
    bb_std        = c.rolling(20).std()
    bb_upper      = bb_mid + 2 * bb_std
    bb_lower      = bb_mid - 2 * bb_std
    bb_width      = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    ema_dist_pct  = (c - ema200) / ema200.replace(0, np.nan) * 100
    ema_slope     = (ema200 - ema200.shift(10)) / ema200.shift(10).replace(0, np.nan) * 100

    rsi14         = _calc_rsi(c, 14)
    adx14         = _calc_adx(df, 14)

    prev_range_pct = (h.shift(1) - l.shift(1)) / c.shift(1) * 100
    body           = (c.shift(1) - df["open"].shift(1)).abs()
    rng            = (h.shift(1) - l.shift(1)).replace(0, np.nan)
    prev_body_ratio = body / rng

    # Rolling 20-bar high/low for pullback
    hh20 = h.rolling(20).max()
    ll20 = l.rolling(20).min()
    rng20 = (hh20 - ll20).replace(0, np.nan)
    pullback_pct  = (hh20 - c) / rng20 * 100   # 0=at top, 100=at bottom

    feat = pd.DataFrame({
        "datetime":       df["datetime"],
        "atr_rank":       atr_rank,
        "real_vol":       real_vol,
        "rel_vol":        rel_vol,
        "bb_width":       bb_width,
        "ema_dist_pct":   ema_dist_pct,
        "ema_slope":      ema_slope,
        "rsi14":          rsi14,
        "adx14":          adx14,
        "prev_range_pct": prev_range_pct,
        "prev_body_ratio":prev_body_ratio,
        "pullback_pct":   pullback_pct,
        "close":          c,
        "volume":         v,
    })
    feat["hour_utc"]  = pd.to_datetime(feat["datetime"]).dt.hour
    feat["dow"]       = pd.to_datetime(feat["datetime"]).dt.dayofweek  # 0=Mon
    feat["session"]   = feat["hour_utc"].map(
        lambda h: "Asia" if h < 8 else ("London" if h < 16 else "NY"))
    feat["trend"]     = feat["ema_slope"].map(
        lambda s: "bull" if s > 0.02 else ("bear" if s < -0.02 else "flat"))
    feat["regime"]    = feat["adx14"].map(
        lambda a: "trending" if a > 25 else ("ranging" if a < 18 else "mixed"))

    return feat.set_index("datetime")

# Load & index all caches
print()
feature_index: dict[str, pd.DataFrame] = {}
for f in sorted(glob.glob(f"{CACHE}/*_1H.parquet")):
    tag = os.path.basename(f).replace("_1H.parquet","")
    sym = tag.replace("_","-",2)
    if sym not in raw["sym"].values:
        continue
    df = pd.read_parquet(f)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    feat = build_feature_index(df)
    feature_index[sym] = feat
    print(f"  {sym:22s}  features computed  n={len(feat):,}")

# =============================================================================
# SECTION 3 — ENRICH EACH TRADE
# =============================================================================

print("\n" + "─"*78)
print("  SECTION 3 — Enriching trades with market features at entry bar")
print("─"*78)

FEATURE_COLS = ["atr_rank","real_vol","rel_vol","bb_width","ema_dist_pct",
                "ema_slope","rsi14","adx14","prev_range_pct","prev_body_ratio",
                "pullback_pct","hour_utc","dow","session","trend","regime"]

enriched_rows = []
skipped = 0

for _, row in raw.iterrows():
    sym = row["sym"]
    if sym not in feature_index:
        skipped += 1
        continue
    fidx = feature_index[sym]

    # Find the signal bar = entry_time - 1H  (features must be pre-entry)
    et = row["entry_time"]
    signal_t = et - pd.Timedelta("1H")

    # Look up in index — try exact match then nearest
    if signal_t in fidx.index:
        feat_row = fidx.loc[signal_t]
    else:
        # Find nearest bar ≤ signal_t
        valid = fidx.index[fidx.index <= signal_t]
        if len(valid) == 0:
            skipped += 1
            continue
        feat_row = fidx.loc[valid[-1]]

    rec = {
        "sym":        sym,
        "entry_time": et,
        "pnl":        row["pnl"],
        "win":        int(row["win"]),
        "source":     row.get("source", "?"),
    }
    for fc in FEATURE_COLS:
        rec[fc] = feat_row.get(fc, np.nan) if isinstance(feat_row, pd.Series) else feat_row[fc]
    enriched_rows.append(rec)

master = pd.DataFrame(enriched_rows)
master = master.dropna(subset=[c for c in FEATURE_COLS if c not in ("session","trend","regime")])
master = master.reset_index(drop=True)

print(f"\n  Enriched: {len(master):,} trades  (skipped: {skipped})")
print(f"  Win rate: {master.win.mean()*100:.1f}%")
wins_pnl = master[master.win==1].pnl.sum()
loss_pnl = abs(master[master.win==0].pnl.sum())
port_pf  = wins_pnl / max(loss_pnl, 1e-9)
print(f"  Portfolio PF: {port_pf:.3f}")
print(f"  Symbols: {master.sym.nunique()}  |  Sources: {master.source.nunique()}")

NUMERIC_FEATURES = ["atr_rank","real_vol","rel_vol","bb_width","ema_dist_pct",
                    "ema_slope","rsi14","adx14","prev_range_pct","prev_body_ratio",
                    "pullback_pct","hour_utc","dow"]
CATEG_FEATURES   = ["session","trend","regime","sym"]

# =============================================================================
# HELPERS
# =============================================================================

def pf_of(trades_df):
    pnl = trades_df["pnl"].values
    gw = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    return gw / max(gl, 1e-9)

def bootstrap_pf_arr(pnl_arr, n_iter=BOOT_ITER, seed=42):
    if len(pnl_arr) < 5:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    pfs = []
    for _ in range(n_iter):
        s = rng.choice(pnl_arr, len(pnl_arr), replace=True)
        gw = s[s > 0].sum(); gl = abs(s[s < 0].sum())
        pfs.append(gw / max(gl, 1e-9))
    return float(np.percentile(pfs, 5)), float(np.percentile(pfs, 50)), float(np.percentile(pfs, 95))

def binomial_p(n_wins, n_total):
    if n_total < 5:
        return 1.0
    return scipy_stats.binomtest(int(n_wins), n_total, BEP_WR, alternative="greater").pvalue

def condition_stats(subset_df, label=""):
    n  = len(subset_df)
    if n == 0:
        return None
    wr = subset_df.win.mean()
    pf = pf_of(subset_df)
    exp_r = wr * RR - (1.0 - wr)
    b5, b50, b95 = bootstrap_pf_arr(subset_df.pnl.values)
    p_binom = binomial_p(int(subset_df.win.sum()), n)
    nsyms = subset_df.sym.nunique() if "sym" in subset_df.columns else 1
    return {
        "label":   label,
        "n":       n,
        "wr":      wr,
        "pf":      pf,
        "exp_r":   exp_r,
        "boot_p50": b50,
        "boot_p5":  b5,
        "boot_p95": b95,
        "p_binom": p_binom,
        "n_syms":  nsyms,
    }

# =============================================================================
# SECTION 4 — SINGLE-FEATURE BUCKET ANALYSIS
# =============================================================================

print("\n" + "═"*78)
print("  SECTION 4 — Single-Feature Bucket Analysis")
print("═"*78)

single_results = []

# Numeric: quintile buckets
for feat in NUMERIC_FEATURES:
    col = master[feat].dropna()
    if len(col) < 50:
        continue
    q_labels = ["Q1 (low)","Q2","Q3","Q4","Q5 (high)"]
    try:
        master[f"_q_{feat}"] = pd.qcut(master[feat], q=5, labels=q_labels, duplicates="drop")
    except Exception:
        continue
    for ql in q_labels:
        sub = master[master[f"_q_{feat}"] == ql]
        if len(sub) < MIN_N:
            continue
        s = condition_stats(sub, f"{feat} = {ql}")
        if s:
            s["feature"] = feat
            s["bucket"]  = ql
            s["type"]    = "numeric_quintile"
            single_results.append(s)

# Categorical: one-hot
for feat in CATEG_FEATURES:
    if feat not in master.columns:
        continue
    for val in master[feat].dropna().unique():
        sub = master[master[feat] == val]
        if len(sub) < MIN_N:
            continue
        s = condition_stats(sub, f"{feat} = {val}")
        if s:
            s["feature"] = feat
            s["bucket"]  = str(val)
            s["type"]    = "categorical"
            single_results.append(s)

sf_df = pd.DataFrame(single_results).sort_values("pf", ascending=False)
print(f"\n  Top 20 single-feature conditions (by PF):")
print(f"  {'Condition':45s}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'Boot p50':>9}  {'Syms':>5}  {'p':>8}")
print("  " + "─"*86)
for _, r in sf_df.head(20).iterrows():
    sig = "✓" if (r["pf"] > 1.10 and r["p_binom"] < 0.05 and r["n"] >= MIN_N) else " "
    print(f"  {sig} {r['label']:43s}  {r['n']:5d}  {r['wr']*100:5.1f}%  {r['pf']:7.3f}  "
          f"{r['boot_p50']:9.3f}  {r['n_syms']:5d}  {r['p_binom']:8.4f}")

print(f"\n  Bottom 10 single-feature conditions (worst environments to avoid):")
for _, r in sf_df.tail(10).sort_values("pf").iterrows():
    print(f"    ✗ {r['label']:43s}  n={r['n']:4d}  PF={r['pf']:.3f}  WR={r['wr']*100:.1f}%")

# =============================================================================
# SECTION 5 — TWO-FEATURE INTERACTION ANALYSIS
# =============================================================================

print("\n" + "═"*78)
print("  SECTION 5 — Two-Feature Interaction Analysis")
print("═"*78)

# Pick top features by PF variance across buckets (most predictive)
feat_variance = {}
for feat in NUMERIC_FEATURES + CATEG_FEATURES:
    sub = sf_df[sf_df["feature"] == feat]
    if len(sub) >= 2:
        feat_variance[feat] = sub["pf"].std()

top_feats = sorted(feat_variance, key=feat_variance.get, reverse=True)[:8]
print(f"\n  Top 8 most predictive features: {top_feats}")

# Discretise top numeric features into 3 buckets for interaction
def discretize(col, feat, n_bins=3):
    if feat in CATEG_FEATURES:
        return col.astype(str)
    labels = ["low","mid","high"]
    try:
        return pd.qcut(col, q=n_bins, labels=labels, duplicates="drop").astype(str)
    except Exception:
        return pd.cut(col, bins=n_bins, labels=labels).astype(str)

for feat in NUMERIC_FEATURES:
    if feat in top_feats:
        master[f"_d_{feat}"] = discretize(master[feat], feat)

def get_d(feat):
    if feat in CATEG_FEATURES:
        return master[feat].astype(str)
    return master.get(f"_d_{feat}", master[feat].astype(str))

two_results = []
pairs = list(itertools.combinations(top_feats[:8], 2))
print(f"\n  Testing {len(pairs)} feature pairs …")

for fa, fb in pairs:
    da = get_d(fa); db = get_d(fb)
    for va in da.dropna().unique():
        for vb in db.dropna().unique():
            mask = (da == va) & (db == vb)
            sub  = master[mask]
            if len(sub) < MIN_N:
                continue
            s = condition_stats(sub, f"{fa}={va} ∧ {fb}={vb}")
            if s:
                s["fa"] = fa; s["va"] = va
                s["fb"] = fb; s["vb"] = vb
                two_results.append(s)

two_df = pd.DataFrame(two_results).sort_values("pf", ascending=False)
print(f"\n  Top 15 two-feature environments (by PF, n≥{MIN_N}):")
print(f"  {'Condition':55s}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'Boot p50':>9}")
print("  " + "─"*87)
for _, r in two_df.head(15).iterrows():
    sig = "✓" if (r["pf"] > 1.10 and r["p_binom"] < 0.05) else " "
    print(f"  {sig} {r['label']:53s}  {r['n']:5d}  {r['wr']*100:5.1f}%  "
          f"{r['pf']:7.3f}  {r['boot_p50']:9.3f}")

# =============================================================================
# SECTION 6 — THREE-FEATURE COMBINATION SEARCH
# =============================================================================

print("\n" + "═"*78)
print("  SECTION 6 — Three-Feature Combination Search")
print("═"*78)

# Use top 6 features only for 3-way combinations (manageable)
top6 = top_feats[:6]
triples = list(itertools.combinations(top6, 3))
print(f"\n  Testing {len(triples)} three-way combinations ({top6}) …")

three_results = []
for fa, fb, fc in triples:
    da = get_d(fa); db = get_d(fb); dc = get_d(fc)
    for va in da.dropna().unique():
        for vb in db.dropna().unique():
            for vc in dc.dropna().unique():
                mask = (da == va) & (db == vb) & (dc == vc)
                sub  = master[mask]
                if len(sub) < MIN_N:
                    continue
                s = condition_stats(sub, f"{fa}={va} ∧ {fb}={vb} ∧ {fc}={vc}")
                if s:
                    s["fa"] = fa; s["va"] = va
                    s["fb"] = fb; s["vb"] = vb
                    s["fc"] = fc; s["vc"] = vc
                    three_results.append(s)

three_df = pd.DataFrame(three_results).sort_values("pf", ascending=False) if three_results else pd.DataFrame()
print(f"\n  Top 12 three-feature environments (by PF, n≥{MIN_N}):")
print(f"  {'Condition':65s}  {'n':>5}  {'WR':>6}  {'PF':>7}")
print("  " + "─"*89)
if len(three_df):
    for _, r in three_df.head(12).iterrows():
        sig = "✓" if (r["pf"] > 1.10 and r["p_binom"] < 0.05) else " "
        print(f"  {sig} {r['label']:63s}  {r['n']:5d}  {r['wr']*100:5.1f}%  {r['pf']:7.3f}")
else:
    print("  (no three-way cells met n threshold)")

# =============================================================================
# SECTION 7 — DECISION TREE RULES
# =============================================================================

print("\n" + "═"*78)
print("  SECTION 7 — Decision Tree Rules")
print("═"*78)

if SKLEARN_OK:
    X_num  = master[NUMERIC_FEATURES].fillna(master[NUMERIC_FEATURES].median())
    y      = master["win"].values

    dt = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=40, random_state=42,
        class_weight="balanced"
    )
    dt.fit(X_num, y)
    tree_text = export_text(dt, feature_names=NUMERIC_FEATURES, max_depth=4)
    print("\n  Decision Tree Rules (max_depth=4, min_leaf=40):")
    print("  " + tree_text.replace("\n", "\n  "))

    # Extract leaf node statistics
    leaf_ids = dt.apply(X_num)
    leaf_stats = []
    for leaf in np.unique(leaf_ids):
        mask = leaf_ids == leaf
        sub  = master[mask]
        s = condition_stats(sub, f"DT-leaf-{leaf}")
        if s and s["n"] >= MIN_N:
            leaf_stats.append(s)
    leaf_df = pd.DataFrame(leaf_stats).sort_values("pf", ascending=False)
    print(f"\n  Decision Tree Leaf Summary (n≥{MIN_N}):")
    print(f"  {'Leaf':10s}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'Exp R':>7}")
    for _, r in leaf_df.iterrows():
        print(f"  {r['label']:10s}  {r['n']:5d}  {r['wr']*100:5.1f}%  "
              f"{r['pf']:7.3f}  {r['exp_r']:+7.3f}")
else:
    leaf_df = pd.DataFrame()

# =============================================================================
# SECTION 8 — RANDOM FOREST + PERMUTATION IMPORTANCE
# =============================================================================

print("\n" + "═"*78)
print("  SECTION 8 — Random Forest Feature Importance")
print("═"*78)

if SKLEARN_OK:
    rf = RandomForestClassifier(n_estimators=400, max_depth=8,
                                min_samples_leaf=20, random_state=42,
                                n_jobs=-1, class_weight="balanced")
    rf.fit(X_num, y)
    imp = pd.Series(rf.feature_importances_, index=NUMERIC_FEATURES).sort_values(ascending=False)
    print("\n  RF Feature Importances (Gini):")
    for feat, val in imp.items():
        bar = "█" * int(val * 40)
        print(f"    {feat:22s}  {val:.4f}  {bar}")

    # Permutation importance (more reliable)
    perm = permutation_importance(rf, X_num, y, n_repeats=10, random_state=42, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=NUMERIC_FEATURES).sort_values(ascending=False)
    print("\n  Permutation Feature Importances:")
    for feat, val in perm_imp.items():
        bar = "█" * max(0, int(val * 200))
        print(f"    {feat:22s}  {val:+.4f}  {bar}")

    # RF-predicted probability for each trade
    master["rf_win_prob"] = rf.predict_proba(X_num)[:, 1]
else:
    imp = pd.Series(dtype=float)
    perm_imp = pd.Series(dtype=float)

# =============================================================================
# SECTION 9 — SHAP EXPLANATIONS
# =============================================================================

print("\n" + "═"*78)
print("  SECTION 9 — SHAP Explanations")
print("═"*78)

shap_vals = None
if SKLEARN_OK and SHAP_OK:
    explainer  = shap.TreeExplainer(rf)
    shap_raw   = explainer.shap_values(X_num)
    # Handle multiple API shapes:
    #  - list of 2 arrays (old shap): take index 1 → (n_samples, n_features)
    #  - 3D array (new shap ≥0.42):  shape (n_samples, n_features, n_classes) → take [:, :, 1]
    #  - 2D array already:            (n_samples, n_features)
    if isinstance(shap_raw, list):
        shap_vals = shap_raw[1]
    elif isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        shap_vals = shap_raw[:, :, 1]
    else:
        shap_vals = shap_raw
    mean_abs   = np.abs(shap_vals).mean(axis=0)
    shap_imp   = pd.Series(mean_abs, index=NUMERIC_FEATURES).sort_values(ascending=False)
    print("\n  SHAP Mean |value| (importance):")
    for feat, val in shap_imp.items():
        bar = "█" * int(val * 60)
        print(f"    {feat:22s}  {val:.4f}  {bar}")
elif SKLEARN_OK:
    print("  [SHAP not installed — skipping. Install with: pip install shap]")
    shap_imp = perm_imp.copy() if len(perm_imp) else pd.Series(dtype=float)
else:
    shap_imp = pd.Series(dtype=float)

# =============================================================================
# SECTION 10 — K-MEANS CLUSTERING
# =============================================================================

print("\n" + "═"*78)
print("  SECTION 10 — K-Means Clustering (k=8)")
print("═"*78)

cluster_results = []
if SKLEARN_OK:
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    km       = KMeans(n_clusters=8, random_state=42, n_init=20)
    master["cluster"] = km.fit_predict(X_scaled)

    print(f"\n  {'Cluster':>9}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'Exp R':>7}  "
          f"{'Boot p50':>9}  {'Top feature (centroid)':>26}")
    print("  " + "─"*74)
    for k in range(8):
        sub = master[master["cluster"] == k]
        s   = condition_stats(sub, f"Cluster-{k}")
        if s:
            centroid = km.cluster_centers_[k]
            top_dim  = NUMERIC_FEATURES[np.argmax(np.abs(centroid))]
            top_val  = centroid[np.argmax(np.abs(centroid))]
            s["top_dim"] = f"{top_dim}={top_val:+.2f}"
            cluster_results.append(s)
            sig = "✓" if s["pf"] > 1.10 else " "
            print(f"  {sig} {s['label']:9s}  {s['n']:5d}  {s['wr']*100:5.1f}%  "
                  f"{s['pf']:7.3f}  {s['exp_r']:+7.3f}  {s['boot_p50']:9.3f}  "
                  f"{s['top_dim']:>26}")

# =============================================================================
# SECTION 11 — ASSOCIATION RULES (manual, no mlxtend)
# =============================================================================

print("\n" + "═"*78)
print("  SECTION 11 — Association Rule Mining")
print("═"*78)

# Discretise all features into binary flags: top-third vs rest
def flag_col(col, feat, high=True):
    """Return boolean: col in top (or bottom) tertile."""
    q = col.quantile(0.67 if high else 0.33)
    return (col >= q) if high else (col <= q)

# Build flag matrix
flags = {}
for feat in ["atr_rank","rsi14","adx14","ema_dist_pct","ema_slope",
             "rel_vol","bb_width","prev_range_pct","pullback_pct"]:
    flags[f"{feat}_HIGH"] = flag_col(master[feat], feat, high=True)
    flags[f"{feat}_LOW"]  = flag_col(master[feat], feat, high=False)

for feat in CATEG_FEATURES:
    if feat in master.columns:
        for val in master[feat].dropna().unique():
            flags[f"{feat}_{val}"] = (master[feat] == val)

flag_df = pd.DataFrame(flags)

# Mine two-item rules
rule_results = []
flag_cols = list(flag_df.columns)
for i, f1 in enumerate(flag_cols):
    for f2 in flag_cols[i+1:]:
        if f1.split("_")[0] == f2.split("_")[0]:
            continue  # skip same-feature pairs (HIGH+LOW of same)
        mask = flag_df[f1] & flag_df[f2]
        sub  = master[mask]
        if len(sub) < MIN_N:
            continue
        s = condition_stats(sub, f"{f1} ∧ {f2}")
        if s and s["pf"] > 1.10 and s["p_binom"] < 0.1:
            rule_results.append(s)

rule_df = pd.DataFrame(rule_results).sort_values("pf", ascending=False) if rule_results else pd.DataFrame()
print(f"\n  High-PF association rules (n≥{MIN_N}, PF>1.10):")
if len(rule_df):
    print(f"  {'Rule':55s}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'Syms':>5}")
    print("  " + "─"*78)
    for _, r in rule_df.head(20).iterrows():
        print(f"  {r['label']:55s}  {r['n']:5d}  {r['wr']*100:5.1f}%  "
              f"{r['pf']:7.3f}  {r['n_syms']:5d}")
else:
    print("  (no rules met threshold)")

# =============================================================================
# SECTION 12 — TOP-20 ENVIRONMENTS
# =============================================================================

print("\n" + "═"*78)
print("  SECTION 12 — TOP-20 MARKET ENVIRONMENTS")
print("═"*78)

def _score(r):
    """Composite score: PF × log(n) × bootstrap_stability."""
    stab = r["boot_p50"] / max(r["pf"], 0.01)  # how stable is PF estimate
    sig  = 1.0 if r["p_binom"] < 0.05 else 0.5
    return r["pf"] * math.log(max(r["n"], 2)) * stab * sig

# Pool all conditions
all_conds = []
for df_, method in [
    (sf_df,    "1-feat"),
    (two_df,   "2-feat"),
    (three_df, "3-feat"),
    (rule_df,  "assoc"),
]:
    if len(df_) == 0:
        continue
    for _, r in df_.iterrows():
        if r["n"] < MIN_N or r["pf"] <= 0:
            continue
        r2 = r.to_dict()
        r2["method"] = method
        r2["score"]  = _score(r2)
        all_conds.append(r2)

if SKLEARN_OK and len(cluster_results):
    for r in cluster_results:
        if r["n"] < MIN_N:
            continue
        r2 = dict(r)
        r2["method"] = "cluster"
        r2["score"]  = _score(r2)
        all_conds.append(r2)

top_conds = (pd.DataFrame(all_conds)
             .sort_values("score", ascending=False)
             .drop_duplicates(subset=["label"])
             .head(20)
             .reset_index(drop=True))

print(f"\n  {'Rank':>4}  {'Method':>7}  {'Condition':55s}  {'n':>5}  "
      f"{'WR':>6}  {'PF':>7}  {'Boot p50':>9}  {'Exp R':>7}  {'Score':>7}")
print("  " + "─"*105)
for rank, (_, r) in enumerate(top_conds.iterrows(), 1):
    sig = "★" if r["pf"] > 1.20 and r["p_binom"] < 0.05 else " "
    print(f"  {sig} {rank:3d}  {r['method']:>7}  {r['label']:55s}  "
          f"{r['n']:5.0f}  {r['wr']*100:5.1f}%  {r['pf']:7.3f}  "
          f"{r['boot_p50']:9.3f}  {r['exp_r']:+7.3f}  {r['score']:7.2f}")

# Worst environments
print(f"\n  Bottom-10 environments (worst — avoid these conditions):")
worst = (pd.DataFrame(all_conds)
         .sort_values("pf")
         .drop_duplicates(subset=["label"])
         .head(10))
for _, r in worst.iterrows():
    print(f"    ✗ [{r['method']}] {r['label']:55s}  n={r['n']:.0f}  PF={r['pf']:.3f}  WR={r['wr']*100:.1f}%")

# =============================================================================
# SECTION 13 — CHARTS
# =============================================================================

print("\n" + "─"*78)
print("  Generating charts …")
print("─"*78)

BG     = "#0d1117"
PANEL  = "#161b22"
TXT    = "#e0e0e0"
GREEN  = "#2ea043"
RED    = "#cf222e"
BLUE   = "#58a6ff"
AMBER  = "#f0c040"

def _style(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=TXT, fontsize=8, pad=4)

# ── Chart 1: Feature importance comparison ────────────────────────────────────
if SKLEARN_OK and len(imp):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
    fig.suptitle("R035 — Feature Importances: RF Gini vs Permutation", color=TXT, fontsize=11)

    for ax_, series, title_, col in [
        (axes[0], imp,      "RF Gini Importance",         BLUE),
        (axes[1], perm_imp, "Permutation Importance",     GREEN),
    ]:
        _style(ax_, title_)
        idx = list(range(len(series)))
        bars = ax_.barh(series.index[::-1], series.values[::-1], color=col, alpha=0.8)
        ax_.set_xlabel("Importance", color="#888", fontsize=8)

    plt.tight_layout()
    p = f"{OUT}/r035_feature_importance.png"
    plt.savefig(p, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")

# ── Chart 2: Single-feature PF heatmap ───────────────────────────────────────
feat_q_pivot = []
Q_LABELS = ["Q1 (low)","Q2","Q3","Q4","Q5 (high)"]
for feat in NUMERIC_FEATURES:
    row = {"feature": feat}
    sub = sf_df[sf_df["feature"] == feat]
    for ql in Q_LABELS:
        match = sub[sub["bucket"] == ql]
        row[ql] = match["pf"].values[0] if len(match) else np.nan
    feat_q_pivot.append(row)
piv_df = pd.DataFrame(feat_q_pivot).set_index("feature")[Q_LABELS]

fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
_style(ax, "R035 — PF by Feature Quintile (red < 1.0 → green > 1.2)")
cmap = LinearSegmentedColormap.from_list("rg", [RED, "#555", GREEN])
im   = ax.imshow(piv_df.values, aspect="auto", cmap=cmap, vmin=0.7, vmax=1.4)
ax.set_xticks(range(len(Q_LABELS))); ax.set_xticklabels(Q_LABELS, color=TXT, fontsize=8)
ax.set_yticks(range(len(piv_df))); ax.set_yticklabels(piv_df.index, color=TXT, fontsize=8)
for i in range(len(piv_df)):
    for j in range(len(Q_LABELS)):
        v = piv_df.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold")
plt.colorbar(im, ax=ax, label="Profit Factor")
plt.tight_layout()
p = f"{OUT}/r035_feature_quintile_heatmap.png"
plt.savefig(p, dpi=120, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"  → {p}")

# ── Chart 3: Top-20 environments bar chart ───────────────────────────────────
if len(top_conds):
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG)
    _style(ax, "R035 — Top-20 Market Environments by Composite Score")
    labels20 = [f"#{i+1} {r['label'][:50]}" for i, (_, r) in enumerate(top_conds.iterrows())]
    pfs20    = top_conds["pf"].values
    ns20     = top_conds["n"].values
    cols20   = [GREEN if p > 1.20 else BLUE for p in pfs20]
    bars = ax.barh(labels20[::-1], pfs20[::-1], color=cols20[::-1], alpha=0.85)
    ax.axvline(1.0,  color="#888", lw=0.8, ls="--")
    ax.axvline(1.20, color=AMBER,  lw=1.0, ls=":", label="PF=1.20")
    ax.set_xlabel("Profit Factor", color="#888", fontsize=8)
    for bar_, n_, pf_ in zip(bars, ns20[::-1], pfs20[::-1]):
        ax.text(bar_.get_width() + 0.01, bar_.get_y() + bar_.get_height()/2,
                f"n={n_:.0f}", va="center", color="#aaa", fontsize=7)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TXT)
    plt.tight_layout()
    p = f"{OUT}/r035_top20_environments.png"
    plt.savefig(p, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")

# ── Chart 4: Interaction heatmaps for top pairs ───────────────────────────────
if len(top_feats) >= 4 and len(two_df):
    # Pick the two pairs with highest max-PF
    best_pairs = (two_df.groupby(["fa","fb"])["pf"].max()
                  .sort_values(ascending=False).head(4).index.tolist())
    n_pairs = len(best_pairs)
    if n_pairs:
        fig, axes = plt.subplots(1, n_pairs, figsize=(5*n_pairs, 5), facecolor=BG)
        if n_pairs == 1:
            axes = [axes]
        fig.suptitle("R035 — Two-Feature Interaction Heatmaps (PF)", color=TXT, fontsize=11)
        for ax_, (fa, fb) in zip(axes, best_pairs):
            sub = two_df[(two_df["fa"] == fa) & (two_df["fb"] == fb)]
            vals_a = sorted(sub["va"].unique())
            vals_b = sorted(sub["vb"].unique())
            mat = np.full((len(vals_a), len(vals_b)), np.nan)
            for _, row in sub.iterrows():
                i_ = vals_a.index(row["va"])
                j_ = vals_b.index(row["vb"])
                mat[i_, j_] = row["pf"]
            _style(ax_, f"{fa} × {fb}")
            cmap2 = LinearSegmentedColormap.from_list("rg", [RED, "#444", GREEN])
            im2 = ax_.imshow(mat, aspect="auto", cmap=cmap2, vmin=0.6, vmax=1.5)
            ax_.set_xticks(range(len(vals_b))); ax_.set_xticklabels(vals_b, color=TXT, fontsize=7, rotation=30)
            ax_.set_yticks(range(len(vals_a))); ax_.set_yticklabels(vals_a, color=TXT, fontsize=7)
            for i_ in range(len(vals_a)):
                for j_ in range(len(vals_b)):
                    v = mat[i_, j_]
                    if not np.isnan(v):
                        ax_.text(j_, i_, f"{v:.2f}", ha="center", va="center",
                                 color="white", fontsize=8, fontweight="bold")
        plt.tight_layout()
        p = f"{OUT}/r035_interaction_heatmaps.png"
        plt.savefig(p, dpi=120, facecolor=BG, bbox_inches="tight")
        plt.close()
        print(f"  → {p}")

# ── Chart 5: SHAP summary plot ────────────────────────────────────────────────
if SKLEARN_OK and SHAP_OK and shap_vals is not None:
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(PANEL)
    shap.summary_plot(shap_vals, X_num, feature_names=NUMERIC_FEATURES,
                      show=False, plot_type="bar", color=BLUE)
    plt.savefig(f"{OUT}/r035_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  → {OUT}/r035_shap_summary.png")

# ── Chart 6: Cluster PF bars ──────────────────────────────────────────────────
if SKLEARN_OK and len(cluster_results):
    cr_df  = pd.DataFrame(cluster_results).sort_values("pf", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    fig.suptitle("R035 — K-Means Cluster Analysis (k=8)", color=TXT, fontsize=11)

    ax_ = axes[0]
    _style(ax_, "PF by Cluster")
    clr = [GREEN if r["pf"] > 1.10 else (AMBER if r["pf"] > 1.0 else RED) for r in cluster_results]
    clr_sorted = [GREEN if r > 1.10 else (AMBER if r > 1.0 else RED) for r in cr_df["pf"]]
    bars = ax_.bar(cr_df["label"], cr_df["pf"], color=clr_sorted, alpha=0.85)
    ax_.axhline(1.0,  color="#888", lw=0.8, ls="--")
    ax_.axhline(1.10, color=AMBER,  lw=0.8, ls=":")
    ax_.set_ylabel("PF", color="#888", fontsize=8)
    for b, n_ in zip(bars, cr_df["n"]):
        ax_.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"n={n_:.0f}", ha="center", color="#aaa", fontsize=7)

    ax2 = axes[1]
    _style(ax2, "Cluster Centroid — Feature Values")
    centres = pd.DataFrame(km.cluster_centers_, columns=NUMERIC_FEATURES)
    centres_n = (centres - centres.mean()) / (centres.std() + 1e-9)
    im3 = ax2.imshow(centres_n.T.values, aspect="auto",
                     cmap="RdYlGn", vmin=-2, vmax=2)
    ax2.set_xticks(range(8)); ax2.set_xticklabels([f"C{i}" for i in range(8)], color=TXT, fontsize=7)
    ax2.set_yticks(range(len(NUMERIC_FEATURES)))
    ax2.set_yticklabels(NUMERIC_FEATURES, color=TXT, fontsize=7)
    plt.colorbar(im3, ax=ax2, label="Normalized Value")
    plt.tight_layout()
    p = f"{OUT}/r035_clusters.png"
    plt.savefig(p, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")

# ── Chart 7: Decision tree leaf PF ────────────────────────────────────────────
if SKLEARN_OK and len(leaf_df):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    _style(ax, "R035 — Decision Tree Leaf PF")
    leaf_sorted = leaf_df.sort_values("pf", ascending=False)
    cols_l = [GREEN if p > 1.10 else (AMBER if p > 1.0 else RED) for p in leaf_sorted["pf"]]
    bars = ax.bar(leaf_sorted["label"], leaf_sorted["pf"], color=cols_l, alpha=0.85)
    ax.axhline(1.0,  color="#888", lw=0.8, ls="--")
    ax.axhline(1.10, color=AMBER,  lw=0.8, ls=":")
    ax.set_ylabel("PF", color="#888", fontsize=8)
    for b, r in zip(bars, leaf_sorted.itertuples()):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                f"n={r.n}", ha="center", color="#aaa", fontsize=7)
    plt.tight_layout()
    p = f"{OUT}/r035_dt_leaves.png"
    plt.savefig(p, dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"  → {p}")

# ── Chart 8: Win distribution by session / dow / hour ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG)
fig.suptitle("R035 — Win Rate & Trade Density by Session / Day / Hour", color=TXT, fontsize=11)

for ax_, groupby_, labels_, title_ in [
    (axes[0], "session", ["Asia","London","NY"],               "Session"),
    (axes[1], "dow",     ["Mon","Tue","Wed","Thu","Fri","Sat","Sat"], "Day of Week"),
    (axes[2], "hour_utc",range(24),                            "Hour UTC"),
]:
    _style(ax_, title_)
    grp = master.groupby(groupby_).agg(wr=("win","mean"), n=("win","count")).reset_index()
    grp = grp.sort_values(groupby_)
    xcol = grp[groupby_].astype(str)
    cols_wr = [GREEN if w > BEP_WR else RED for w in grp.wr]
    bars_ = ax_.bar(xcol, grp.wr, color=cols_wr, alpha=0.75)
    ax_.axhline(BEP_WR, color=AMBER, lw=0.8, ls="--", label=f"BEP {BEP_WR*100:.0f}%")
    ax_.set_ylabel("Win Rate", color="#888", fontsize=8)
    ax2_ = ax_.twinx()
    ax2_.plot(xcol, grp.n, color=BLUE, lw=1.2, marker="o", markersize=3, label="n trades")
    ax2_.set_ylabel("n trades", color="#888", fontsize=8)
    ax2_.tick_params(colors="#888", labelsize=6)
    ax_.legend(fontsize=7, facecolor=PANEL, labelcolor=TXT, loc="upper left")

plt.tight_layout()
p = f"{OUT}/r035_session_dow_hour.png"
plt.savefig(p, dpi=120, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"  → {p}")

# ── Chart 9: Main dashboard ───────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 18), facecolor=BG)
fig.suptitle(
    f"QUANTLAB AI — R035 | Universal Edge Discovery | "
    f"{len(master):,} trades · {master.sym.nunique()} symbols · "
    f"WR={master.win.mean()*100:.1f}% · Portfolio PF={port_pf:.3f}",
    color=TXT, fontsize=13, y=0.98
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35,
                       top=0.93, bottom=0.05, left=0.05, right=0.97)

# 1: Feature importance
if SKLEARN_OK and len(imp):
    ax1 = fig.add_subplot(gs[0, :2])
    _style(ax1, "Feature Importance (RF Gini)")
    imp_s = imp.sort_values()
    ax1.barh(imp_s.index, imp_s.values, color=BLUE, alpha=0.8)
    ax1.set_xlabel("Importance", color="#888", fontsize=7)

# 2: Quintile heatmap (compact)
ax2 = fig.add_subplot(gs[0, 2:])
_style(ax2, "PF by Feature Quintile")
cmap_rg = LinearSegmentedColormap.from_list("rg", [RED, "#555", GREEN])
im_ = ax2.imshow(piv_df.values, aspect="auto", cmap=cmap_rg, vmin=0.7, vmax=1.4)
ax2.set_xticks(range(len(Q_LABELS))); ax2.set_xticklabels(["Q1","Q2","Q3","Q4","Q5"],
               color=TXT, fontsize=7)
ax2.set_yticks(range(len(piv_df))); ax2.set_yticklabels(piv_df.index, color=TXT, fontsize=7)
for i in range(len(piv_df)):
    for j in range(len(Q_LABELS)):
        v = piv_df.values[i, j]
        if not np.isnan(v):
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white", fontsize=6)

# 3: Session WR
ax3 = fig.add_subplot(gs[1, 0])
_style(ax3, "WR by Session")
sess_g = master.groupby("session")["win"].agg(["mean","count"]).reset_index()
sess_pf = master.groupby("session").apply(lambda x: pf_of(x)).reset_index(name="pf")
sess_g = sess_g.merge(sess_pf, on="session")
cols_s = [GREEN if w > BEP_WR else RED for w in sess_g["mean"]]
ax3.bar(sess_g["session"], sess_g["mean"], color=cols_s, alpha=0.8)
ax3.axhline(BEP_WR, color=AMBER, lw=0.8, ls="--")
ax3.set_ylabel("Win Rate", color="#888", fontsize=7)

# 4: Regime PF
ax4 = fig.add_subplot(gs[1, 1])
_style(ax4, "PF by Regime")
reg_pf = master.groupby("regime").apply(lambda x: pf_of(x)).sort_values(ascending=False)
cols_r = [GREEN if v > 1.1 else (AMBER if v > 1.0 else RED) for v in reg_pf]
ax4.bar(reg_pf.index, reg_pf.values, color=cols_r, alpha=0.8)
ax4.axhline(1.0, color="#888", lw=0.8, ls="--")
ax4.set_ylabel("PF", color="#888", fontsize=7)

# 5: Trend PF
ax5 = fig.add_subplot(gs[1, 2])
_style(ax5, "PF by Trend")
tr_pf = master.groupby("trend").apply(lambda x: pf_of(x)).sort_values(ascending=False)
cols_t = [GREEN if v > 1.1 else (AMBER if v > 1.0 else RED) for v in tr_pf]
ax5.bar(tr_pf.index, tr_pf.values, color=cols_t, alpha=0.8)
ax5.axhline(1.0, color="#888", lw=0.8, ls="--")
ax5.set_ylabel("PF", color="#888", fontsize=7)

# 6: Symbol PF
ax6 = fig.add_subplot(gs[1, 3])
_style(ax6, "PF by Symbol")
sym_pf = (master.groupby("sym")
          .apply(lambda x: pf_of(x))
          .sort_values(ascending=False))
sym_tags = [s.split("-")[0] for s in sym_pf.index]
cols_sy  = [GREEN if v > 1.1 else (AMBER if v > 1.0 else RED) for v in sym_pf]
ax6.barh(sym_tags[::-1], sym_pf.values[::-1], color=cols_sy[::-1], alpha=0.8)
ax6.axvline(1.0, color="#888", lw=0.8, ls="--")
ax6.set_xlabel("PF", color="#888", fontsize=7)

# 7: Top-20 environments
if len(top_conds):
    ax7 = fig.add_subplot(gs[2, :3])
    _style(ax7, "Top-20 Market Environments (PF)")
    labs  = [f"#{i+1} {r['label'][:40]}" for i, (_, r) in enumerate(top_conds.iterrows())]
    pfvals = top_conds["pf"].values
    c20   = [GREEN if p > 1.20 else (BLUE if p > 1.10 else AMBER) for p in pfvals]
    ax7.barh(labs[::-1], pfvals[::-1], color=c20[::-1], alpha=0.85)
    ax7.axvline(1.0,  color="#888", lw=0.8, ls="--")
    ax7.axvline(1.20, color=AMBER,  lw=0.8, ls=":")
    ax7.set_xlabel("Profit Factor", color="#888", fontsize=7)

# 8: Summary text
ax8 = fig.add_subplot(gs[2, 3])
ax8.set_facecolor(PANEL)
for sp in ax8.spines.values(): sp.set_visible(False)
ax8.set_xticks([]); ax8.set_yticks([])
top3_conds = top_conds.head(3)
summary_lines = [
    f"R035 EDGE DISCOVERY SUMMARY",
    f"",
    f"Total trades: {len(master):,}",
    f"Symbols:      {master.sym.nunique()}",
    f"Portfolio PF: {port_pf:.3f}",
    f"Win rate:     {master.win.mean()*100:.1f}%",
    f"",
    f"Top environment:",
]
if len(top_conds):
    r0 = top_conds.iloc[0]
    summary_lines += [
        f"  {r0['label'][:35]}",
        f"  PF={r0['pf']:.3f}  n={r0['n']:.0f}  WR={r0['wr']*100:.1f}%",
        f"",
        f"Key finding: see",
        f"r035_top20_environments",
        f"r035_feature_quintile_heatmap",
        f"r035_interaction_heatmaps",
    ]
ax8.text(0.05, 0.95, "\n".join(summary_lines),
         transform=ax8.transAxes, color=TXT, fontsize=8,
         fontfamily="monospace", va="top",
         bbox=dict(boxstyle="round", facecolor="#0d1117", edgecolor="#333"))

plt.savefig(f"{OUT}/r035_dashboard.png", dpi=120, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"  → {OUT}/r035_dashboard.png")

# =============================================================================
# SECTION 14 — WRITE EDGE BLUEPRINT
# =============================================================================

blueprint_path = f"{OUT}/r035_edge_blueprint.txt"
with open(blueprint_path, "w") as bpf_:
    bpf_.write("=" * 80 + "\n")
    bpf_.write("QUANTLAB AI — R035 EDGE BLUEPRINT\n")
    bpf_.write("Universal Edge Discovery from Pooled Trades\n")
    bpf_.write("=" * 80 + "\n\n")

    bpf_.write(f"Total trades analysed : {len(master):,}\n")
    bpf_.write(f"Symbols               : {master.sym.nunique()}\n")
    bpf_.write(f"Portfolio WR          : {master.win.mean()*100:.1f}%\n")
    bpf_.write(f"Portfolio PF          : {port_pf:.3f}\n\n")

    bpf_.write("TOP-20 WINNING MARKET ENVIRONMENTS\n")
    bpf_.write("-" * 80 + "\n")
    for rank, (_, r) in enumerate(top_conds.iterrows(), 1):
        bpf_.write(f"  #{rank:2d}  [{r['method']:7s}]  {r['label']}\n")
        bpf_.write(f"        n={r['n']:.0f}  WR={r['wr']*100:.1f}%  PF={r['pf']:.3f}  "
                   f"Boot_p50={r['boot_p50']:.3f}  Exp_R={r['exp_r']:+.3f}  "
                   f"p_binom={r['p_binom']:.4f}\n\n")

    bpf_.write("SINGLE-FEATURE RANKINGS (top 5 per feature)\n")
    bpf_.write("-" * 80 + "\n")
    for feat in top_feats[:8]:
        sub = sf_df[sf_df["feature"] == feat].head(5)
        if len(sub) == 0:
            continue
        bpf_.write(f"\n  {feat}:\n")
        for _, r in sub.iterrows():
            bpf_.write(f"    {r['bucket']:15s}  n={r['n']:.0f}  PF={r['pf']:.3f}  WR={r['wr']*100:.1f}%\n")

    bpf_.write("\nWORST ENVIRONMENTS (avoid)\n")
    bpf_.write("-" * 80 + "\n")
    for _, r in worst.iterrows():
        bpf_.write(f"  [{r['method']:7s}]  {r['label']}\n")
        bpf_.write(f"    n={r['n']:.0f}  PF={r['pf']:.3f}  WR={r['wr']*100:.1f}%\n")

    if SKLEARN_OK and len(imp):
        bpf_.write("\nFEATURE IMPORTANCES (RF Gini)\n")
        bpf_.write("-" * 80 + "\n")
        for feat, val in imp.items():
            bpf_.write(f"  {feat:22s}  {val:.4f}\n")

    bpf_.write("\nNEXT STEPS — STRATEGY CONSTRUCTION\n")
    bpf_.write("-" * 80 + "\n")
    bpf_.write("  1. Select the top environment(s) with n≥200 and PF>1.20\n")
    bpf_.write("  2. These define the ENTRY FILTER for R036\n")
    bpf_.write("  3. Design a new entry signal that triggers ONLY in this environment\n")
    bpf_.write("  4. The entry signal should be independent of the filter features\n")
    bpf_.write("  5. Validate with 5-fold walk-forward on all 23 symbols\n")
    bpf_.write("=" * 80 + "\n")

print(f"  → {blueprint_path}")

# =============================================================================
# SECTION 15 — MASTER TRADE LOG
# =============================================================================

master_path = f"{OUT}/r035_master_trades.csv"
save_cols = ["sym","entry_time","pnl","win","source"] + FEATURE_COLS
if "rf_win_prob" in master.columns:
    save_cols.append("rf_win_prob")
if "cluster" in master.columns:
    save_cols.append("cluster")
master[[c for c in save_cols if c in master.columns]].to_csv(master_path, index=False)
print(f"  → {master_path}  ({len(master):,} trades)")

# Journal update
journal_path = CONFIG["JOURNAL_FILE"]
journal_row  = {
    "research_id": RESEARCH_ID,
    "date":        pd.Timestamp.now().strftime("%Y-%m-%d"),
    "strategy":    "UniversalEdgeDiscovery",
    "timeframe":   "1H",
    "symbols":     ",".join(sorted(master.sym.unique())),
    "method":      "pool+enrich+ML+assoc",
    "n_oos":       len(master),
    "wr":          round(master.win.mean(), 4),
    "pf":          round(port_pf, 4),
    "sharpe":      0.0,
    "mdd":         0.0,
    "net":         round(float(master.pnl.sum()), 2),
    "boot_p50":    round(top_conds.iloc[0]["boot_p50"] if len(top_conds) else 0, 4),
    "mc_prob":     0.0,
    "loo_floor":   0.0,
    "verdict":     "DISCOVERY",
}
jdf = pd.read_csv(journal_path) if os.path.exists(journal_path) else pd.DataFrame()
jdf = pd.concat([jdf, pd.DataFrame([journal_row])], ignore_index=True)
jdf.to_csv(journal_path, index=False)
print(f"  Journal updated → {journal_path}")

# =============================================================================
# SECTION 16 — FINAL SUMMARY
# =============================================================================

print("\n" + "═"*78)
print(f"  R035 COMPLETE — UNIVERSAL EDGE DISCOVERY")
print("═"*78)
print(f"\n  Trades analysed : {len(master):,}")
print(f"  Symbols         : {master.sym.nunique()}")
print(f"  Portfolio PF    : {port_pf:.3f}  WR={master.win.mean()*100:.1f}%")
print(f"\n  ──────── TOP 5 MARKET ENVIRONMENTS ────────")
for rank, (_, r) in enumerate(top_conds.head(5).iterrows(), 1):
    print(f"\n  #{rank}  {r['label']}")
    print(f"     n={r['n']:.0f}  WR={r['wr']*100:.1f}%  PF={r['pf']:.3f}  "
          f"Boot_p50={r['boot_p50']:.3f}  Exp_R={r['exp_r']:+.3f}")

if SKLEARN_OK and len(imp):
    print(f"\n  ──────── TOP 5 FEATURES BY IMPORTANCE ────────")
    for feat, val in imp.head(5).items():
        print(f"    {feat:22s}  {val:.4f}")

print(f"\n  ──────── CATEGORICAL INSIGHTS ────────")
for feat, label in [("session","Session"), ("trend","Trend"), ("regime","Regime")]:
    grp = master.groupby(feat).apply(lambda x: pf_of(x)).sort_values(ascending=False)
    print(f"    {label}: " + "  ".join(f"{k}={v:.3f}" for k, v in grp.items()))

print(f"\n  Output: {OUT}/r035_*")
print(f"  Blueprint: {blueprint_path}")
print("═"*78)
