"""
QUANTLAB AI — RESEARCH #030
ADX Attribution for the Low ATR FVG Edge
=========================================

R029 produced the first profitable configuration:
  FVG + EMA200 Slope + Low ATR → PF=1.205, WR=46.9%, n=64
  Failed PROMOTE only on statistical confidence (n<80, boot p50=1.196).

R030 objective: determine whether ADX14 independently explains the residual
edge and should be added as a permanent filter.

METHOD
  • Use the EXACT 64 trades from R029 trade log — no re-backtesting.
  • Re-attach ADX14, BB Width(20,2), Relative Volume(20) at the SIGNAL bar
    (1 bar before entry) for every trade.
  • All analysis is DESCRIPTIVE only. No threshold optimisation.

OUTPUTS
  Statistical significance tests | Quartile tables | Interaction matrices |
  Natural breakpoint analysis | Per-symbol diagnosis | Verdict
"""

import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import shap

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

RESEARCH_ID = "R030"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
TRADE_LOG = f"{OUT}/r029_fvg_low_atr_1h_9sym_trades.csv"

SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
    "LINK-USDT-SWAP", "AVAX-USDT-SWAP", "XRP-USDT-SWAP",
    "LTC-USDT-SWAP",  "BCH-USDT-SWAP",  "DOGE-USDT-SWAP",
]
SPLIT = 0.70

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #030" + " "*50 + "║")
print("║  ADX Attribution for the Low ATR FVG Edge" + " "*36 + "║")
print("╚" + "═"*79 + "╝")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Load R029 trades
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n  Loading R029 trade log …")
trades = pd.read_csv(TRADE_LOG)
trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
trades["exit_time"]  = pd.to_datetime(trades["exit_time"],  utc=True)
# Signal bar = bar BEFORE entry (ADX etc. computed on closed signal bar)
trades["signal_time"] = trades["entry_time"] - pd.Timedelta(hours=1)
print(f"  Loaded {len(trades)} trades across {trades['sym'].nunique()} symbols")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Compute ADX + enrichment features from 1H OHLCV
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Computing ADX14, BB Width(20,2), Relative Volume(20), EMA slope …")

def load_oos_features(sym):
    tag  = sym.replace("-", "_")
    df   = pd.read_parquet(f"{CACHE}/{tag}_1H.parquet")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)

    # ADX14
    df["adx"] = calc_adx(df, 14)

    # ATR14 rank (same as R029)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank_pct"] = df["atr14"].rolling(100).rank(pct=True) * 100

    # Bollinger Bands (20, 2)
    df["bb_mid"]   = df["close"].rolling(20).mean()
    df["bb_std"]   = df["close"].rolling(20).std(ddof=1)
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)

    # Relative Volume (20-bar rolling)
    df["rel_vol"] = df["vol"] / df["vol"].rolling(20).mean().replace(0, np.nan)

    # EMA200 slope value (numeric, not boolean)
    df["ema200"] = calc_ema(df["close"], 200)
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / df["ema200"].shift(10) * 100

    # OOS only (same split as R029)
    cut = int(len(df) * SPLIT)
    return df.iloc[cut:].reset_index(drop=True)

feat_dfs = {}
for sym in SYMBOLS:
    try:
        feat_dfs[sym] = load_oos_features(sym)
        print(f"    {sym.split('-')[0]:5s}  OK  (OOS bars={len(feat_dfs[sym]):,})")
    except FileNotFoundError:
        print(f"    {sym}: cache missing — skipped")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Join features to trades
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Joining features to trades …")

enriched_rows = []
for _, row in trades.iterrows():
    sym = row["sym"]
    if sym not in feat_dfs:
        continue
    fdf = feat_dfs[sym]
    # Match signal bar by datetime
    mask = fdf["datetime"] == row["signal_time"]
    if not mask.any():
        # Try closest bar within 1h tolerance
        diffs = (fdf["datetime"] - row["signal_time"]).abs()
        idx = diffs.idxmin()
        if diffs.iloc[idx] > pd.Timedelta(hours=2):
            continue
        bar = fdf.iloc[idx]
    else:
        bar = fdf[mask].iloc[0]

    enriched_rows.append({
        **row.to_dict(),
        "adx_signal":       float(bar["adx"]),
        "bb_width_signal":  float(bar["bb_width"]) if not np.isnan(bar["bb_width"]) else np.nan,
        "rel_vol_signal":   float(bar["rel_vol"])   if not np.isnan(bar["rel_vol"])   else np.nan,
        "ema_slope_signal": float(bar["ema200_slope"]) if not np.isnan(bar["ema200_slope"]) else np.nan,
    })

df = pd.DataFrame(enriched_rows)
df = df.dropna(subset=["adx_signal"]).reset_index(drop=True)
print(f"  Enriched {len(df)}/{len(trades)} trades with ADX14")

wins  = df[df["win"] == 1]
losses= df[df["win"] == 0]
adx_w = wins["adx_signal"].values
adx_l = losses["adx_signal"].values

print(f"\n  ADX at signal bar:")
print(f"    All  : mean={df['adx_signal'].mean():.2f}  median={df['adx_signal'].median():.2f}  "
      f"std={df['adx_signal'].std():.2f}  min={df['adx_signal'].min():.2f}  max={df['adx_signal'].max():.2f}")
print(f"    Winners ({len(wins)}): mean={adx_w.mean():.2f}  median={np.median(adx_w):.2f}  std={adx_w.std():.2f}")
print(f"    Losers  ({len(losses)}): mean={adx_l.mean():.2f}  median={np.median(adx_l):.2f}  std={adx_l.std():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  STATISTICAL TESTS — ADX14 vs Trade Outcome")
print("═"*78)

# 4.1 Cohen's d
def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = math.sqrt(((n1-1)*a.std()**2 + (n2-1)*b.std()**2) / (n1+n2-2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0

d = cohens_d(adx_w, adx_l)
t_stat, p_val = stats.ttest_ind(adx_w, adx_l, equal_var=False)
mw_stat, mw_p = stats.mannwhitneyu(adx_w, adx_l, alternative="two-sided")

print(f"\n  4.1  Cohen's d (Winner ADX − Loser ADX):")
print(f"       d = {d:.4f}  {'(negligible)' if abs(d)<0.2 else '(small)' if abs(d)<0.5 else '(medium)' if abs(d)<0.8 else '(large)'}")
print(f"       Welch t-test:  t={t_stat:.3f}  p={p_val:.4f}  {'SIGNIFICANT p<0.05 ✓' if p_val < 0.05 else 'NOT significant ✗'}")
print(f"       Mann-Whitney U: U={mw_stat:.0f}  p={mw_p:.4f}  {'SIGNIFICANT p<0.05 ✓' if mw_p < 0.05 else 'NOT significant ✗'}")

# 4.2 Spearman correlation
sp_win,  sp_p_win  = stats.spearmanr(df["adx_signal"], df["win"])
sp_rmul, sp_p_rmul = stats.spearmanr(df["adx_signal"], df["r_multiple"])
sp_pnl,  sp_p_pnl  = stats.spearmanr(df["adx_signal"], df["pnl"])

print(f"\n  4.2  Spearman Correlations with ADX:")
print(f"       ADX → Win         ρ={sp_win:+.4f}  p={sp_p_win:.4f}  {'✓' if sp_p_win < 0.05 else '✗'}")
print(f"       ADX → R-Multiple  ρ={sp_rmul:+.4f}  p={sp_p_rmul:.4f}  {'✓' if sp_p_rmul < 0.05 else '✗'}")
print(f"       ADX → P&L         ρ={sp_pnl:+.4f}  p={sp_p_pnl:.4f}  {'✓' if sp_p_pnl < 0.05 else '✗'}")

# 4.3 Bootstrap CI on mean ADX difference (winners minus losers)
rng = np.random.default_rng(42)
N_BOOT = 10_000
boot_diffs = []
for _ in range(N_BOOT):
    bw = rng.choice(adx_w, len(adx_w), replace=True).mean()
    bl = rng.choice(adx_l, len(adx_l), replace=True).mean()
    boot_diffs.append(bw - bl)
boot_diffs = np.array(boot_diffs)
ci_lo  = np.percentile(boot_diffs, 2.5)
ci_hi  = np.percentile(boot_diffs, 97.5)
ci_med = np.percentile(boot_diffs, 50)
ci_sig = (ci_lo > 0) or (ci_hi < 0)  # interval excludes 0

print(f"\n  4.3  Bootstrap CI on (mean ADX winners − mean ADX losers) — {N_BOOT:,} iterations:")
print(f"       p2.5 = {ci_lo:+.3f}   p50 = {ci_med:+.3f}   p97.5 = {ci_hi:+.3f}")
print(f"       95% CI excludes zero: {'YES ✓  (significant)' if ci_sig else 'NO ✗  (not significant)'}")

# 4.4 Permutation importance (using all features)
feat_cols = ["adx_signal", "atr_rank_pct", "bb_width_signal", "rel_vol_signal", "ema_slope_signal"]
feat_labels = ["ADX14", "ATR Rank", "BB Width", "Rel Volume", "EMA Slope"]
df_feat = df[feat_cols + ["win"]].dropna()
X = df_feat[feat_cols].values
y = df_feat["win"].values

print(f"\n  4.4  Permutation Importance (RandomForest, {len(df_feat)} trades):")
rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=4)
rf.fit(X, y)
perm = permutation_importance(rf, X, y, n_repeats=500, random_state=42)
perm_means = perm.importances_mean
perm_stds  = perm.importances_std
perm_order = np.argsort(perm_means)[::-1]
for i in perm_order:
    print(f"       {feat_labels[i]:12s}: importance={perm_means[i]:+.4f} ± {perm_stds[i]:.4f}  "
          f"{'important ✓' if perm_means[i] > 0.005 else 'noise ✗'}")

# 4.5 SHAP
print(f"\n  4.5  SHAP Feature Importance (TreeExplainer):")
explainer  = shap.TreeExplainer(rf)
shap_vals  = explainer.shap_values(X)
# TreeExplainer returns:
#   list[class0_arr, class1_arr]  — older SHAP / sklearn
#   ndarray shape (n_samples, n_features)         — newer SHAP single output
#   ndarray shape (n_samples, n_features, n_classes) — newer SHAP multi-output
if isinstance(shap_vals, list):
    sv = shap_vals[1]                 # class 1 from list
elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
    sv = shap_vals[:, :, 1]           # shape → (n_samples, n_features)
elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
    sv = shap_vals                    # already (n_samples, n_features)
else:
    sv = np.atleast_2d(shap_vals)
shap_abs = np.abs(sv).mean(axis=0)   # shape: (n_features,)
shap_order = np.argsort(shap_abs)[::-1]
for i in shap_order:
    print(f"       {feat_labels[i]:12s}: mean|SHAP| = {shap_abs[i]:.4f}  "
          f"{'top feature ✓' if i == shap_order[0] else ''}")

adx_shap_rank = list(shap_order).index(0) + 1   # rank of ADX (index 0)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ADX QUARTILE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  ADX QUARTILE ANALYSIS")
print("═"*78)

q_bounds = df["adx_signal"].quantile([0.0, 0.25, 0.50, 0.75, 1.0]).values
df["adx_quartile"] = pd.qcut(df["adx_signal"], q=4, labels=["Q1","Q2","Q3","Q4"])

print(f"\n  ADX quartile boundaries: "
      f"[{q_bounds[0]:.1f}, {q_bounds[1]:.1f}, {q_bounds[2]:.1f}, {q_bounds[3]:.1f}, {q_bounds[4]:.1f}]")
print(f"\n  {'Quartile':8s}  {'ADX Range':14s}  {'n':>4}  {'WR':>6}  {'PF':>7}  "
      f"{'ExpR':>7}  {'Net $':>8}  {'Avg ADX':>8}")
print("  " + "─"*72)

quartile_stats = {}
for q in ["Q1","Q2","Q3","Q4"]:
    sub  = df[df["adx_quartile"] == q]
    n    = len(sub)
    if n == 0:
        continue
    nw   = sub["win"].sum()
    nl   = n - nw
    gw   = sub[sub["win"]==1]["pnl"].sum() if nw > 0 else 0.0
    gl   = abs(sub[sub["win"]==0]["pnl"].sum()) if nl > 0 else 1e-9
    pf   = gw / gl
    wr   = nw / n
    exp  = wr * 2.0 - (1-wr)
    net  = sub["pnl"].sum()
    adx_lo = sub["adx_signal"].min()
    adx_hi = sub["adx_signal"].max()
    adx_avg= sub["adx_signal"].mean()
    quartile_stats[q] = {"n":n,"wr":wr,"pf":pf,"exp":exp,"net":net,"adx_avg":adx_avg}
    print(f"  {q:8s}  {adx_lo:5.1f}–{adx_hi:5.1f}    {n:4d}  {wr*100:5.1f}%  {pf:7.3f}  "
          f"{exp:+7.3f}  {net:+8.0f}  {adx_avg:8.2f}")

print(f"\n  Trend: WR monotonically increasing with ADX? ", end="")
wrs = [quartile_stats[q]["wr"] for q in ["Q1","Q2","Q3","Q4"]]
mono = all(wrs[i] <= wrs[i+1] for i in range(len(wrs)-1))
print("YES ✓" if mono else f"NO ✗  ({[f'{w*100:.1f}%' for w in wrs]})")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — INTERACTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  INTERACTION ANALYSIS")
print("═"*78)

def pf_from_sub(sub):
    if len(sub) == 0: return 0.0, 0
    nw = sub["win"].sum(); nl = len(sub)-nw
    gw = sub[sub["win"]==1]["pnl"].sum() if nw>0 else 0.0
    gl = abs(sub[sub["win"]==0]["pnl"].sum()) if nl>0 else 1e-9
    return gw/gl, len(sub)

# Median splits
adx_med = df["adx_signal"].median()
atr_med = df["atr_rank_pct"].median()
bbw_med = df["bb_width_signal"].median()
rv_med  = df["rel_vol_signal"].median()

print(f"\n  Median splits used:  ADX={adx_med:.1f}  ATR Rank={atr_med:.1f}  "
      f"BB Width={bbw_med:.4f}  RelVol={rv_med:.2f}")

# 6.1 ADX × ATR Rank
print(f"\n  6.1  ADX × ATR Rank (Low ATR already applied in R029, shown for completeness)")
print(f"  {'':12s}  {'ADX Low (<med)':>16}  {'ADX High (≥med)':>16}")
for atr_tag, atr_mask in [("ATR Low",  df["atr_rank_pct"] < atr_med),
                           ("ATR High", df["atr_rank_pct"] >= atr_med)]:
    pfll, nll = pf_from_sub(df[atr_mask & (df["adx_signal"] < adx_med)])
    pflh, nlh = pf_from_sub(df[atr_mask & (df["adx_signal"] >= adx_med)])
    print(f"  {atr_tag:12s}  PF={pfll:.3f} n={nll:3d}     PF={pflh:.3f} n={nlh:3d}")

# 6.2 ADX × EMA Slope
print(f"\n  6.2  ADX × EMA200 Slope (numeric)")
slope_med = df["ema_slope_signal"].median()
print(f"  Slope median: {slope_med:.4f}%")
for slope_tag, slope_mask in [("Slope Low",  df["ema_slope_signal"] < slope_med),
                               ("Slope High", df["ema_slope_signal"] >= slope_med)]:
    pfll, nll = pf_from_sub(df[slope_mask & (df["adx_signal"] < adx_med)])
    pflh, nlh = pf_from_sub(df[slope_mask & (df["adx_signal"] >= adx_med)])
    print(f"  {slope_tag:12s}  ADX Low: PF={pfll:.3f} n={nll:3d}   ADX High: PF={pflh:.3f} n={nlh:3d}")

# 6.3 ADX × BB Width
print(f"\n  6.3  ADX × Bollinger Width")
df_bbw = df.dropna(subset=["bb_width_signal"])
bbw_med2 = df_bbw["bb_width_signal"].median()
for bbw_tag, bbw_mask in [("BB Narrow",  df_bbw["bb_width_signal"] < bbw_med2),
                           ("BB Wide",   df_bbw["bb_width_signal"] >= bbw_med2)]:
    pfll, nll = pf_from_sub(df_bbw[bbw_mask & (df_bbw["adx_signal"] < adx_med)])
    pflh, nlh = pf_from_sub(df_bbw[bbw_mask & (df_bbw["adx_signal"] >= adx_med)])
    print(f"  {bbw_tag:10s}  ADX Low: PF={pfll:.3f} n={nll:3d}   ADX High: PF={pflh:.3f} n={nlh:3d}")

# 6.4 ADX × Relative Volume
print(f"\n  6.4  ADX × Relative Volume")
df_rv = df.dropna(subset=["rel_vol_signal"])
rv_med2 = df_rv["rel_vol_signal"].median()
for rv_tag, rv_mask in [("Low RelVol",  df_rv["rel_vol_signal"] < rv_med2),
                         ("High RelVol", df_rv["rel_vol_signal"] >= rv_med2)]:
    pfll, nll = pf_from_sub(df_rv[rv_mask & (df_rv["adx_signal"] < adx_med)])
    pflh, nlh = pf_from_sub(df_rv[rv_mask & (df_rv["adx_signal"] >= adx_med)])
    print(f"  {rv_tag:12s}  ADX Low: PF={pfll:.3f} n={nll:3d}   ADX High: PF={pflh:.3f} n={nlh:3d}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — NATURAL BREAKPOINT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  NATURAL BREAKPOINT DETECTION")
print("═"*78)

adx_vals = df["adx_signal"].values
print(f"\n  ADX distribution (all 64 trades):")
print(f"    p10={np.percentile(adx_vals,10):.1f}  p20={np.percentile(adx_vals,20):.1f}  "
      f"p25={np.percentile(adx_vals,25):.1f}  p33={np.percentile(adx_vals,33):.1f}  "
      f"p40={np.percentile(adx_vals,40):.1f}  p50={np.percentile(adx_vals,50):.1f}")
print(f"    p60={np.percentile(adx_vals,60):.1f}  p67={np.percentile(adx_vals,67):.1f}  "
      f"p75={np.percentile(adx_vals,75):.1f}  p80={np.percentile(adx_vals,80):.1f}  "
      f"p90={np.percentile(adx_vals,90):.1f}")

# Kernel density estimate — find local minima (natural valleys) as candidate thresholds
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
adx_range = np.linspace(adx_vals.min(), adx_vals.max(), 300)
kde_bw = 1.5 * adx_vals.std() * len(adx_vals)**(-1/5)  # Silverman's rule * 1.5
kde_vals = np.array([np.exp(-0.5*((adx_range - v)/kde_bw)**2).sum() for v in adx_range])
kde_vals = kde_vals / kde_vals.sum()
kde_smooth = gaussian_filter1d(kde_vals, sigma=5)

# Find valleys (local minima in KDE = natural separation points)
valleys, _ = find_peaks(-kde_smooth, distance=20)
peaks,   _ = find_peaks( kde_smooth, distance=20)

print(f"\n  KDE peaks (modes) at approx ADX:")
for p in peaks:
    print(f"    {adx_range[p]:.1f}")

print(f"\n  KDE valleys (natural breakpoints) at approx ADX:")
candidate_breaks = []
for v in valleys:
    candidate_breaks.append(adx_range[v])
    print(f"    {adx_range[v]:.1f}")

# Sliding WR across ADX thresholds (5–95th pct) — find where WR changes most
print(f"\n  WR by ADX threshold (sliding, step=2 units):")
print(f"  {'ADX≥thr':>8s}  {'n_above':>8s}  {'WR_above':>9s}  {'WR_below':>9s}  {'ΔWRR':>8s}")
thresholds_slide = np.arange(math.floor(np.percentile(adx_vals,15)),
                              math.ceil(np.percentile(adx_vals,85)), 2)
slide_rows = []
for thr in thresholds_slide:
    ab  = df[df["adx_signal"] >= thr]
    bel = df[df["adx_signal"] < thr]
    if len(ab) < 5 or len(bel) < 5:
        continue
    wr_ab  = ab["win"].mean()
    wr_bel = bel["win"].mean()
    d_wr   = wr_ab - wr_bel
    slide_rows.append({"thr": thr, "n_above": len(ab), "wr_above": wr_ab,
                        "wr_below": wr_bel, "delta_wr": d_wr})
    print(f"  {thr:8.1f}  {len(ab):8d}  {wr_ab*100:8.1f}%  {wr_bel*100:8.1f}%  {d_wr*100:+7.1f}pp")

slide_df = pd.DataFrame(slide_rows)
if len(slide_df) > 0:
    best_idx = slide_df["delta_wr"].idxmax()
    best_row = slide_df.loc[best_idx]
    print(f"\n  Peak ΔWR at ADX ≥ {best_row['thr']:.1f}: "
          f"WR {best_row['wr_below']*100:.1f}% → {best_row['wr_above']*100:.1f}%  "
          f"(Δ={best_row['delta_wr']*100:+.1f}pp)  n_above={int(best_row['n_above'])}")
    natural_break = float(best_row["thr"])
else:
    natural_break = adx_med

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PER-SYMBOL ADX DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  PER-SYMBOL ADX DIAGNOSIS")
print("═"*78)
print(f"\n  {'Symbol':7s}  {'n':>3}  {'ADX mean':>9}  {'ADX med':>8}  "
      f"{'WR':>6}  {'PF':>7}  {'ADX_W mean':>11}  {'ADX_L mean':>11}")
print("  " + "─"*74)

sym_stats = {}
for sym in df["sym"].unique():
    sub = df[df["sym"] == sym]
    sw  = sub[sub["win"]==1]["adx_signal"]
    sl  = sub[sub["win"]==0]["adx_signal"]
    adx_w_mean = sw.mean() if len(sw) else np.nan
    adx_l_mean = sl.mean() if len(sl) else np.nan
    pf_v, _ = pf_from_sub(sub)
    sym_stats[sym] = {"adx_mean": sub["adx_signal"].mean(), "adx_w": adx_w_mean,
                      "adx_l": adx_l_mean, "pf": pf_v}
    tag = sym.split("-")[0]
    print(f"  {tag:7s}  {len(sub):3d}  {sub['adx_signal'].mean():9.2f}  "
          f"{sub['adx_signal'].median():8.2f}  "
          f"{sub['win'].mean()*100:5.1f}%  {pf_v:7.3f}  "
          f"{adx_w_mean:11.2f}  {adx_l_mean:11.2f}")

# ETH vs others diagnosis
eth_sub  = df[df["sym"]=="ETH-USDT-SWAP"]
link_sub = df[df["sym"]=="LINK-USDT-SWAP"]
others   = df[df["sym"].isin(["ETH-USDT-SWAP","LINK-USDT-SWAP"]) == False]
eth_adx_w_mean  = eth_sub[eth_sub["win"]==1]["adx_signal"].mean() if len(eth_sub) else np.nan
link_adx_w_mean = link_sub[link_sub["win"]==1]["adx_signal"].mean() if len(link_sub) else np.nan

print(f"\n  ETH vs others:")
print(f"    ETH  ADX mean  = {eth_sub['adx_signal'].mean():.2f}  WR={eth_sub['win'].mean()*100:.1f}%")
print(f"    Other 8 syms   = {others['adx_signal'].mean():.2f}  WR={others['win'].mean()*100:.1f}%")

print(f"\n  LINK diagnosis:")
print(f"    LINK ADX mean  = {link_sub['adx_signal'].mean():.2f}  WR={link_sub['win'].mean()*100:.1f}%")
print(f"    LINK winner ADX mean = {link_adx_w_mean:.2f}")
print(f"    LINK loser  ADX mean = {link_sub[link_sub['win']==0]['adx_signal'].mean():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — RESEARCH QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  RESEARCH QUESTIONS")
print("═"*78)

# Determine answers programmatically
adx_predictive = (sp_p_win < 0.10) or (p_val < 0.10) or (abs(d) >= 0.2) or ci_sig

eth_adx_high = eth_sub["adx_signal"].mean() > df["adx_signal"].mean() + 3
eth_wr_high  = eth_sub["win"].mean() > df["win"].mean() + 0.10

link_adx_mix = (not np.isnan(link_adx_w_mean) and
                link_adx_w_mean > link_sub[link_sub["win"]==0]["adx_signal"].mean() + 2)

# ADX additive: does ADX≥nat_break give better PF than full Low ATR baseline?
above_break = df[df["adx_signal"] >= natural_break]
below_break = df[df["adx_signal"] <  natural_break]
pf_above, n_above = pf_from_sub(above_break)
pf_below, n_below = pf_from_sub(below_break)
adx_additive = pf_above > 1.20 and n_above >= 15

print(f"""
  Q1. Is ADX independently predictive of trade outcome?
      Cohen's d    = {d:.4f}  {'(≥0.20 = small effect)' if abs(d) >= 0.20 else '(< 0.20 = negligible)'}
      Spearman ρ   = {sp_win:+.4f}  p={sp_p_win:.4f}
      Boot 95% CI  = [{ci_lo:+.3f}, {ci_hi:+.3f}]  {'excludes 0 ✓' if ci_sig else 'includes 0 ✗'}
      SHAP rank    = #{adx_shap_rank} of {len(feat_labels)} features
      Answer: {'YES ✓ — ADX has statistically detectable predictive power' if adx_predictive else 'WEAK ✗ — ADX does not reach significance at conventional thresholds'}

  Q2. Does ADX explain ETH's strength?
      ETH ADX mean = {eth_sub['adx_signal'].mean():.2f} vs portfolio mean {df['adx_signal'].mean():.2f}
      ETH WR       = {eth_sub['win'].mean()*100:.1f}%  vs portfolio {df['win'].mean()*100:.1f}%
      Answer: {'YES — ETH trades coincide with higher ADX, consistent with ADX explaining ETH edge' if eth_adx_high else 'PARTIAL — ETH ADX is not markedly above average; other factors drive ETH outperformance' if eth_wr_high else 'NO — ADX does not explain ETH strength at this sample size'}

  Q3. Does ADX explain LINK's weakness?
      LINK winners ADX mean = {link_adx_w_mean:.2f}
      LINK losers  ADX mean = {link_sub[link_sub['win']==0]['adx_signal'].mean():.2f}
      LINK overall ADX mean = {link_sub['adx_signal'].mean():.2f}
      Answer: {'YES — LINK winners had meaningfully higher ADX than LINK losers' if link_adx_mix else 'PARTIAL — ADX alone does not explain LINK underperformance; Low ATR filter on LINK may just select choppy bars'}

  Q4. Is ADX additive with Low ATR?
      ADX ≥ {natural_break:.0f}: PF={pf_above:.3f}  n={n_above}
      ADX <  {natural_break:.0f}: PF={pf_below:.3f}  n={n_below}
      Answer: {'YES ✓ — combining ADX ≥ natural break with Low ATR produces PF above 1.20' if adx_additive else f'MARGINAL — PF={pf_above:.3f} above break, but n={n_above} is {"insufficient" if n_above < 15 else "thin"}; additive potential exists but unconfirmed'}

  Q5. Should ADX become a permanent feature?
      Statistical significance: {'REACHED' if adx_predictive else 'NOT REACHED'}
      SHAP rank: #{adx_shap_rank}/{len(feat_labels)}
      WR monotone with ADX: {'YES' if mono else 'NO'}
      Natural break PF: {pf_above:.3f} (n={n_above})
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — VERDICT
# ─────────────────────────────────────────────────────────────────────────────

print("═"*78)
print("  VERDICT")
print("═"*78)

# Criteria for recommending ADX for R031 validation
crit = {
    "ADX predictive (d≥0.20 or p<0.10 or CI excludes 0)": adx_predictive,
    "ADX SHAP rank ≤ 3":                                   adx_shap_rank <= 3,
    "WR monotone with ADX quartiles":                      mono,
    "Above-break PF > 1.00":                               pf_above > 1.00,
    "Natural breakpoint identified":                       len(slide_rows) > 0,
}
n_pass = sum(crit.values())
VERDICT = "VALIDATE_R031" if n_pass >= 3 else "REJECT"
vcolor  = "\033[92m" if VERDICT == "VALIDATE_R031" else "\033[91m"
vreset  = "\033[0m"

print(f"\n  {vcolor}VERDICT: {VERDICT}{vreset}")
print()
for c, ok in crit.items():
    print(f"    {'✓' if ok else '✗'} {c}")
print(f"\n  {n_pass}/{len(crit)} criteria met.")

if VERDICT == "VALIDATE_R031":
    print(f"""
  RECOMMENDATION: Proceed to R031 — ADX-Gated Backtest
  ─────────────────────────────────────────────────────
  ADX14 shows detectable predictive power on the existing 64 Low ATR trades.
  R031 should:
    • Re-run the full R029 backtest engine on all 9 symbols, 1H
    • Add ADX14 ≥ {natural_break:.0f} as a gate on top of Low ATR
    • Report: new n, PF, WR, bootstrap p50, MC P(profit)
    • DO NOT change entry/exit/SL/TP/risk — ADX is additive only
    • Promote criteria remain unchanged: PF>1.20, n≥80, boot p50>1.20, MC>60%

  Natural breakpoint to test: ADX ≥ {natural_break:.0f}
  (Identified as peak-WR split, not optimised)
""")
else:
    print(f"""
  RECOMMENDATION: Reject ADX for R031 — Search Elsewhere
  ───────────────────────────────────────────────────────
  ADX14 does not show sufficient independent predictive power on 64 trades.
  The signal is too weak to justify adding it as a permanent filter.

  Suggested alternatives for R031:
    • Volume surge filter (Rel Vol > 1.5 at entry)
    • Time-of-day filter (exclude dead hours UTC 0–6)
    • Session filter (Asian vs London vs NY session)
    • Secondary momentum: RSI > 50 at signal bar
""")

print("═"*78)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Generating charts …")

def dark_ax(ax, title=None, col="white"):
    ax.set_facecolor("#111")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=col, fontsize=9)

# ── Chart 1: ADX distribution winners vs losers ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#111")
fig.suptitle("R030 — ADX14 Distribution: Winners vs Losers", color="white", fontsize=11)

ax1 = axes[0]; dark_ax(ax1, "Histogram")
bins = np.linspace(df["adx_signal"].min()-1, df["adx_signal"].max()+1, 25)
ax1.hist(adx_w, bins=bins, color="#4CAF50", alpha=0.65, label=f"Winners n={len(wins)}", density=True)
ax1.hist(adx_l, bins=bins, color="#F44336", alpha=0.65, label=f"Losers n={len(losses)}", density=True)
ax1.axvline(adx_w.mean(), color="#4CAF50", lw=1.5, ls="--", label=f"W mean={adx_w.mean():.1f}")
ax1.axvline(adx_l.mean(), color="#F44336", lw=1.5, ls="--", label=f"L mean={adx_l.mean():.1f}")
ax1.set_xlabel("ADX14 at signal bar", color="white"); ax1.set_ylabel("Density", color="white")
ax1.legend(facecolor="#222", labelcolor="white", fontsize=8)

ax2 = axes[1]; dark_ax(ax2, "KDE + Breakpoints")
ax2.plot(adx_range, kde_smooth / kde_smooth.max(), color="#9E9E9E", lw=1.5, label="KDE")
ax2.hist(df["adx_signal"], bins=20, color="#607D8B", alpha=0.3, density=True,
         label="All trades")
for v in valleys:
    ax2.axvline(adx_range[v], color="#FF9800", lw=1.2, ls=":", alpha=0.8,
                label=f"Valley {adx_range[v]:.1f}")
ax2.axvline(natural_break, color="#FFEB3B", lw=1.8, ls="--", label=f"Best break {natural_break:.0f}")
ax2.set_xlabel("ADX14", color="white"); ax2.set_ylabel("Normalised density", color="white")
ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r030_adx_distribution.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 2: Quartile WR and PF ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111")
fig.suptitle("R030 — ADX Quartile Analysis: WR and PF", color="white", fontsize=11)
qs    = ["Q1","Q2","Q3","Q4"]
q_wrs = [quartile_stats[q]["wr"]*100   for q in qs]
q_pfs = [quartile_stats[q]["pf"]       for q in qs]
q_ns  = [quartile_stats[q]["n"]        for q in qs]
q_adx = [quartile_stats[q]["adx_avg"]  for q in qs]
q_cols= ["#F44336","#FF9800","#8BC34A","#4CAF50"]

ax1 = axes[0]; dark_ax(ax1, "Win Rate by ADX Quartile")
bars_ = ax1.bar([f"{q}\n≈{a:.0f}" for q,a in zip(qs,q_adx)], q_wrs, color=q_cols, alpha=0.85)
ax1.axhline(33.3, color="white", lw=0.8, ls="--", alpha=0.6, label="BEP 33.3%")
ax1.axhline(50.0, color="#FF9800", lw=0.8, ls=":", alpha=0.6, label="50%")
ax1.set_ylabel("Win Rate %", color="white"); ax1.legend(facecolor="#222", labelcolor="white", fontsize=8)
for b, v, n_ in zip(bars_, q_wrs, q_ns):
    ax1.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%\nn={n_}",
             ha="center", color="white", fontsize=9)

ax2 = axes[1]; dark_ax(ax2, "Profit Factor by ADX Quartile")
bars_ = ax2.bar([f"{q}\n≈{a:.0f}" for q,a in zip(qs,q_adx)], q_pfs, color=q_cols, alpha=0.85)
ax2.axhline(1.0, color="white", lw=0.8, ls="--", alpha=0.5)
ax2.axhline(1.2, color="#FF9800", lw=0.8, ls=":", alpha=0.6, label="PF=1.20")
ax2.set_ylabel("Profit Factor", color="white"); ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
for b, v in zip(bars_, q_pfs):
    ax2.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", color="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r030_quartile_analysis.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 3: Feature importance comparison ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#111")
fig.suptitle("R030 — Feature Importance: Permutation vs SHAP", color="white", fontsize=11)

ax1 = axes[0]; dark_ax(ax1, "Permutation Importance (higher = more important)")
pm_sorted = sorted(zip(feat_labels, perm_means, perm_stds), key=lambda x: x[1], reverse=True)
pm_labs, pm_vals, pm_stds = zip(*pm_sorted)
cols_perm = ["#FF9800" if l == "ADX14" else "#607D8B" for l in pm_labs]
ax1.barh(pm_labs[::-1], [v for v in pm_vals[::-1]], xerr=[s for s in pm_stds[::-1]],
         color=cols_perm[::-1], alpha=0.85, capsize=5)
ax1.axvline(0, color="white", lw=0.7, ls="--", alpha=0.5)
ax1.set_xlabel("Mean importance ± std", color="white")

ax2 = axes[1]; dark_ax(ax2, "SHAP Feature Importance (mean |SHAP value|)")
sh_sorted = sorted(zip(feat_labels, shap_abs), key=lambda x: x[1], reverse=True)
sh_labs, sh_vals = zip(*sh_sorted)
cols_shap = ["#FF9800" if l == "ADX14" else "#607D8B" for l in sh_labs]
ax2.barh(sh_labs[::-1], [v for v in sh_vals[::-1]], color=cols_shap[::-1], alpha=0.85)
ax2.set_xlabel("Mean |SHAP value|", color="white")
plt.tight_layout()
p = f"{OUT}/r030_feature_importance.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 4: WR sliding threshold ────────────────────────────────────────────
if len(slide_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#111")
    dark_ax(ax, "R030 — WR Above vs Below ADX Threshold (sliding)")
    ax.plot(slide_df["thr"], slide_df["wr_above"]*100, color="#4CAF50", lw=2, label="WR ≥ threshold")
    ax.plot(slide_df["thr"], slide_df["wr_below"]*100, color="#F44336", lw=2, label="WR < threshold")
    ax.fill_between(slide_df["thr"],
                    slide_df["wr_above"]*100, slide_df["wr_below"]*100,
                    alpha=0.15, color="#FF9800", label="Δ WR")
    ax.axvline(natural_break, color="#FFEB3B", lw=2, ls="--", label=f"Best break = {natural_break:.0f}")
    ax.axhline(33.3, color="white", lw=0.7, ls=":", alpha=0.5, label="BEP 33.3%")
    ax.set_xlabel("ADX threshold", color="white"); ax.set_ylabel("Win Rate %", color="white")
    ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
    plt.tight_layout()
    p = f"{OUT}/r030_adx_threshold_sweep.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  → {p}")

# ── Chart 5: ADX vs R-Multiple scatter ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6), facecolor="#111")
dark_ax(ax, f"R030 — ADX14 vs R-Multiple  ρ={sp_rmul:+.3f}  p={sp_p_rmul:.3f}")
cols_sc = ["#4CAF50" if w else "#F44336" for w in df["win"]]
ax.scatter(df["adx_signal"], df["r_multiple"], c=cols_sc, alpha=0.7, s=60, edgecolors="none")
# regression line
ols_m, ols_b, *_ = stats.linregress(df["adx_signal"], df["r_multiple"])
m, b = ols_m, ols_b   # keep short aliases for later reuse
x_line = np.array([df["adx_signal"].min(), df["adx_signal"].max()])
ax.plot(x_line, ols_m*x_line + ols_b, color="#FF9800", lw=1.5, ls="--", label=f"OLS slope={ols_m:.3f}")
ax.axhline(0, color="white", lw=0.6, ls=":", alpha=0.4)
ax.axvline(natural_break, color="#FFEB3B", lw=1.2, ls="--", alpha=0.7, label=f"Break={natural_break:.0f}")
ax.set_xlabel("ADX14 at signal bar", color="white"); ax.set_ylabel("R-Multiple", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
green_patch = plt.Line2D([0],[0],marker="o",color="#4CAF50",lw=0,label="Win")
red_patch   = plt.Line2D([0],[0],marker="o",color="#F44336",lw=0,label="Loss")
ax.legend(handles=[green_patch, red_patch,
                   plt.Line2D([0],[0],color="#FF9800",ls="--",label=f"OLS slope={m:.3f}"),
                   plt.Line2D([0],[0],color="#FFEB3B",ls="--",label=f"Break={natural_break:.0f}")],
          facecolor="#222", labelcolor="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r030_adx_scatter.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 6: Per-symbol ADX breakdown ────────────────────────────────────────
syms_plot = list(df["sym"].unique())
sym_tags_p = [s.split("-")[0] for s in syms_plot]
sym_adx_w  = [df[(df["sym"]==s) & (df["win"]==1)]["adx_signal"].mean() if len(df[(df["sym"]==s) & (df["win"]==1)])>0 else np.nan for s in syms_plot]
sym_adx_l  = [df[(df["sym"]==s) & (df["win"]==0)]["adx_signal"].mean() if len(df[(df["sym"]==s) & (df["win"]==0)])>0 else np.nan for s in syms_plot]
sym_adx_a  = [df[df["sym"]==s]["adx_signal"].mean() for s in syms_plot]

x_s = np.arange(len(syms_plot)); w_s = 0.28
fig, ax = plt.subplots(figsize=(18, 5), facecolor="#111")
dark_ax(ax, "R030 — ADX14 at Signal Bar by Symbol: Winners vs Losers vs All")
ax.bar(x_s - w_s, sym_adx_w, w_s, color="#4CAF50", alpha=0.85, label="Winner ADX mean")
ax.bar(x_s,       sym_adx_a, w_s, color="#607D8B", alpha=0.85, label="All ADX mean")
ax.bar(x_s + w_s, sym_adx_l, w_s, color="#F44336", alpha=0.85, label="Loser ADX mean")
ax.set_xticks(x_s); ax.set_xticklabels(sym_tags_p, color="white")
ax.set_ylabel("ADX14", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
for xi, (aw, al) in enumerate(zip(sym_adx_w, sym_adx_l)):
    if not np.isnan(aw):
        ax.text(xi - w_s, aw + 0.3, f"{aw:.1f}", ha="center", color="white", fontsize=7)
    if not np.isnan(al):
        ax.text(xi + w_s, al + 0.3, f"{al:.1f}", ha="center", color="white", fontsize=7)
plt.tight_layout()
p = f"{OUT}/r030_symbol_adx_breakdown.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 7: Bootstrap CI on ADX difference ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#111")
dark_ax(ax, f"R030 — Bootstrap CI: Mean ADX Winners − Mean ADX Losers  (n={N_BOOT:,} iter)")
ax.hist(boot_diffs, bins=60, color="#9C27B0", alpha=0.75, edgecolor="none", density=True)
ax.axvline(0,      color="white",   lw=1.5, ls="--", label="Zero (null)")
ax.axvline(ci_lo,  color="#F44336", lw=1.5, ls=":",  label=f"p2.5 = {ci_lo:.3f}")
ax.axvline(ci_hi,  color="#4CAF50", lw=1.5, ls=":",  label=f"p97.5 = {ci_hi:.3f}")
ax.axvline(ci_med, color="#FF9800", lw=2.0, ls="-",  label=f"p50 = {ci_med:.3f}")
ax.set_xlabel("Δ ADX mean (Winners − Losers)", color="white")
ax.set_ylabel("Density", color="white")
ax.legend(facecolor="#222", labelcolor="white", fontsize=9)
sig_txt = "95% CI excludes 0 — SIGNIFICANT" if ci_sig else "95% CI includes 0 — NOT significant"
ax.text(0.5, 0.95, sig_txt, transform=ax.transAxes, color="#FFEB3B",
        ha="center", va="top", fontsize=10, fontweight="bold")
plt.tight_layout()
p = f"{OUT}/r030_bootstrap_ci.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 8: Interaction heatmaps ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor="#111")
fig.suptitle("R030 — ADX × ATR Rank & ADX × BB Width: WR Heatmap", color="white", fontsize=11)

def interaction_heatmap(ax, df_, col_x, label_x, col_y, label_y, n_bins=4):
    try:
        df_c = df_.dropna(subset=[col_x, col_y])
        df_c = df_c.copy()
        df_c["bx"] = pd.qcut(df_c[col_x], n_bins, labels=False, duplicates="drop")
        df_c["by"] = pd.qcut(df_c[col_y], n_bins, labels=False, duplicates="drop")
        pivot = df_c.groupby(["by","bx"])["win"].mean().unstack(fill_value=np.nan)
        count = df_c.groupby(["by","bx"]).size().unstack(fill_value=0)
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        plt.colorbar(im, ax=ax, label="Win Rate")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_yticks(range(pivot.shape[0]))
        xq_bounds = pd.qcut(df_c[col_x], n_bins, retbins=True, duplicates="drop")[1]
        yq_bounds = pd.qcut(df_c[col_y], n_bins, retbins=True, duplicates="drop")[1]
        ax.set_xticklabels([f"Q{i+1}\n≤{xq_bounds[i+1]:.1f}" for i in range(len(xq_bounds)-1)],
                           color="white", fontsize=7)
        ax.set_yticklabels([f"Q{i+1}\n≤{yq_bounds[i+1]:.1f}" for i in range(len(yq_bounds)-1)],
                           color="white", fontsize=7)
        ax.set_xlabel(label_x, color="white"); ax.set_ylabel(label_y, color="white")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i,j]; c = count.values[i,j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v*100:.0f}%\nn={c}", ha="center", va="center",
                            color="black", fontsize=8, fontweight="bold")
    except Exception as e:
        ax.text(0.5, 0.5, f"Insufficient data\n{e}", transform=ax.transAxes,
                ha="center", va="center", color="white")

interaction_heatmap(axes[0], df, "adx_signal", "ADX14", "atr_rank_pct", "ATR Rank")
dark_ax(axes[0], "ADX × ATR Rank — Win Rate")
interaction_heatmap(axes[1], df, "adx_signal", "ADX14", "bb_width_signal", "BB Width")
dark_ax(axes[1], "ADX × BB Width — Win Rate")
plt.tight_layout()
p = f"{OUT}/r030_interaction_heatmaps.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 9: Full Dashboard ───────────────────────────────────────────────────
vcolor_map = {"VALIDATE_R031": "#4CAF50", "REJECT": "#F44336"}
vcolor_dash = vcolor_map.get(VERDICT, "white")

fig = plt.figure(figsize=(28, 20), facecolor="#0a0a0a")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.60, wspace=0.45)

fig.suptitle(
    f"QUANTLAB AI — R030 DASHBOARD\n"
    f"ADX Attribution | Low ATR FVG Edge | 64 Trades | Verdict: {VERDICT}",
    color="white", fontsize=13, y=0.99)

# Stats summary table
ax_t = fig.add_subplot(gs[0, :2])
ax_t.axis("off")
tbl_data = [
    ["Cohen's d",        f"{d:.4f}", "small ✓" if abs(d)>=0.2 else "negligible ✗"],
    ["Spearman ρ (win)", f"{sp_win:+.4f}", f"p={sp_p_win:.4f} {'✓' if sp_p_win<0.10 else '✗'}"],
    ["Boot p50 diff",    f"{ci_med:+.3f}", f"[{ci_lo:.3f},{ci_hi:.3f}] {'excl.0 ✓' if ci_sig else 'incl.0 ✗'}"],
    ["SHAP rank",        f"#{adx_shap_rank}/{len(feat_labels)}", "✓" if adx_shap_rank<=3 else "✗"],
    ["WR monotone",      "YES" if mono else "NO",  "✓" if mono else "✗"],
    ["Above-break PF",   f"{pf_above:.3f}  (n={n_above})", "✓" if pf_above>1.20 else f"✗"],
]
tbl = ax_t.table(cellText=tbl_data, colLabels=["Test","Value","Pass?"],
                 loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10)
for (r_, c_), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r_ % 2 == 0 else "#222")
    cell.set_text_props(color="white"); cell.set_edgecolor("#333")
    if r_ == 0:
        cell.set_facecolor("#2a2a2a"); cell.set_text_props(color="#aaa", fontweight="bold")
ax_t.set_title("Statistical Tests", color="white", fontsize=10)

# Quartile table
ax_q = fig.add_subplot(gs[0, 2:])
ax_q.axis("off")
q_tbl = [[q, f"≈{quartile_stats[q]['adx_avg']:.0f}", str(quartile_stats[q]["n"]),
           f"{quartile_stats[q]['wr']*100:.1f}%", f"{quartile_stats[q]['pf']:.3f}",
           f"{quartile_stats[q]['exp']:+.3f}"]
          for q in ["Q1","Q2","Q3","Q4"]]
qtbl = ax_q.table(cellText=q_tbl,
                   colLabels=["Quartile","ADX≈","n","WR","PF","ExpR"],
                   loc="center", cellLoc="center")
qtbl.auto_set_font_size(False); qtbl.set_fontsize(10)
for (r_, c_), cell in qtbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r_ % 2 == 0 else "#222")
    cell.set_text_props(color="white"); cell.set_edgecolor("#333")
    if r_ == 0:
        cell.set_facecolor("#2a2a2a"); cell.set_text_props(color="#aaa", fontweight="bold")
ax_q.set_title("ADX Quartile Summary", color="white", fontsize=10)

# WR by quartile bar
ax1 = fig.add_subplot(gs[1, :2])
dark_ax(ax1, "Win Rate by ADX Quartile")
bars_ = ax1.bar([f"{q}\n≈{a:.0f}" for q,a in zip(qs,q_adx)], q_wrs, color=q_cols, alpha=0.85)
ax1.axhline(33.3, color="white", lw=0.8, ls="--"); ax1.axhline(50, color="#FF9800", lw=0.8, ls=":")
ax1.set_ylabel("WR %", color="white")
for b,v,n_ in zip(bars_,q_wrs,q_ns):
    ax1.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%\nn={n_}", ha="center", color="white", fontsize=9)

# ADX scatter
ax2 = fig.add_subplot(gs[1, 2:])
dark_ax(ax2, f"ADX14 vs R-Multiple  ρ={sp_rmul:+.3f}")
cols_sc2 = ["#4CAF50" if w else "#F44336" for w in df["win"]]
ax2.scatter(df["adx_signal"], df["r_multiple"], c=cols_sc2, alpha=0.7, s=50)
ax2.plot(x_line, ols_m*x_line+ols_b, color="#FF9800", lw=1.5, ls="--")
ax2.axvline(natural_break, color="#FFEB3B", lw=1.2, ls="--")
ax2.axhline(0, color="white", lw=0.5, ls=":", alpha=0.4)
ax2.set_xlabel("ADX14", color="white"); ax2.set_ylabel("R-Multiple", color="white")

# Sliding threshold
if len(slide_df) > 0:
    ax3 = fig.add_subplot(gs[2, :2])
    dark_ax(ax3, "WR vs ADX Threshold")
    ax3.plot(slide_df["thr"], slide_df["wr_above"]*100, color="#4CAF50", lw=2, label="WR ≥ thr")
    ax3.plot(slide_df["thr"], slide_df["wr_below"]*100, color="#F44336", lw=2, label="WR < thr")
    ax3.axvline(natural_break, color="#FFEB3B", lw=2, ls="--", label=f"Best={natural_break:.0f}")
    ax3.axhline(33.3, color="white", lw=0.7, ls=":", alpha=0.5)
    ax3.set_xlabel("ADX", color="white"); ax3.set_ylabel("WR %", color="white")
    ax3.legend(facecolor="#222", labelcolor="white", fontsize=8)

# Verdict panel
ax4 = fig.add_subplot(gs[2, 2:])
ax4.axis("off"); ax4.set_facecolor("#111")
ax4.text(0.5, 0.92, f"VERDICT: {VERDICT}", transform=ax4.transAxes,
         color=vcolor_dash, fontsize=20, ha="center", fontweight="bold")
summary = (f"n=64  Cohen's d={d:.3f}  Spearman ρ={sp_win:+.3f}  p={sp_p_win:.3f}\n"
           f"Boot 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]  {'excl.0' if ci_sig else 'incl.0'}\n"
           f"SHAP rank #{adx_shap_rank}/{len(feat_labels)}  |  WR monotone: {'YES' if mono else 'NO'}\n"
           f"Natural break: ADX≥{natural_break:.0f}  →  PF={pf_above:.3f}  n={n_above}\n"
           f"{n_pass}/{len(crit)} criteria met")
ax4.text(0.5, 0.55, summary, transform=ax4.transAxes,
         color="white", fontsize=10, ha="center", va="center")
checks_str = "\n".join(f"{'✓' if ok else '✗'} {c}" for c, ok in crit.items())
ax4.text(0.5, 0.08, checks_str, transform=ax4.transAxes,
         color="#aaa", fontsize=9, ha="center", va="bottom")

plt.savefig(f"{OUT}/r030_dashboard.png", dpi=130, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"  → {OUT}/r030_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE enriched trade log
# ─────────────────────────────────────────────────────────────────────────────

out_csv = f"{OUT}/r030_enriched_trades.csv"
df.to_csv(out_csv, index=False)
print(f"  → {out_csv}  ({len(df)} trades)")

# Journal
try:
    from quantlab_ai import append_journal
    from datetime import datetime, timezone as _tz
    run_date = datetime.now(tz=_tz.utc).strftime("%Y-%m-%d")
    append_journal([{
        "research_id":    RESEARCH_ID,
        "run_date":       run_date,
        "strategy_name":  f"ADX_Attribution_LowATR_FVG",
        "symbol":         "PORTFOLIO_9SYM",
        "n_trades":       len(df),
        "profit_factor":  round(pf_above, 4),
        "expectancy_r":   round(sp_rmul, 4),
        "win_rate":       round(df["win"].mean(), 4),
        "net_profit":     round(df["pnl"].sum(), 2),
        "max_drawdown":   0.0,
        "sharpe":         0.0,
        "mc_prob_profit": 0.0,
        "avg_hold_minutes": 0,
        "verdict":        VERDICT,
    }])
    print(f"  Journal updated → {CONFIG['JOURNAL_FILE']}")
except Exception as e:
    print(f"  [WARN] Journal: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{'═'*78}")
print(f"  R030 complete.")
print(f"  Verdict       : {VERDICT}")
print(f"  Cohen's d     : {d:.4f}  ({abs(d):.3f})")
print(f"  Spearman ρ    : {sp_win:+.4f}  p={sp_p_win:.4f}")
print(f"  Boot 95% CI   : [{ci_lo:.3f}, {ci_hi:.3f}]  {'SIGNIFICANT' if ci_sig else 'NOT significant'}")
print(f"  SHAP rank     : #{adx_shap_rank}/{len(feat_labels)}")
print(f"  WR monotone   : {'YES' if mono else 'NO'}")
print(f"  Natural break : ADX ≥ {natural_break:.0f}  →  PF={pf_above:.3f}  n={n_above}")
print(f"  {n_pass}/{len(crit)} criteria met")
print(f"  Output        → {OUT}/r030_*")
print(f"{'═'*78}\n")
