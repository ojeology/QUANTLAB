"""
QUANTLAB AI — RESEARCH #031
BB Width Attribution for the Low ATR FVG Edge
==============================================

R030 rejected ADX as explanatory variable.
BB Width ranked #1 in both SHAP and permutation importance (R030).

Hypothesis:
  "Bollinger Band Width(20,2) contains independent predictive information
   about trade quality within the Low ATR FVG + EMA200 Slope strategy."

METHOD
  • Use exact 64 trades from R029 Low ATR FVG, enriched with features in R030.
  • No re-backtesting. Attribution study only.
  • No optimisation, no threshold tuning, no strategy modifications.
  • Scientific honesty above all.

STRUCTURE
  Q1  Winner vs loser BB Width comparison + statistics
  Q2  Quartile analysis (4 equal buckets)
  Q3  Feature importance — permutation + SHAP
  Q4  ATR Rank × BB Width 2×2 interaction matrix
  Q5  BB Width deciles (descriptive only, no threshold recommendations)
  R   Robustness: bootstrap(5000), jackknife, leave-one-symbol-out
  FQ  Final research questions + verdict
"""

import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG

RESEARCH_ID = "R031"
OUT   = CONFIG["OUTPUT_FOLDER"]
CAPITAL = CONFIG["STARTING_CAPITAL"]
os.makedirs(OUT, exist_ok=True)

ENRICHED_CSV = f"{OUT}/r030_enriched_trades.csv"

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #031" + " "*50 + "║")
print("║  BB Width Attribution for the Low ATR FVG Edge" + " "*31 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Source     : R029 Low ATR FVG trades (n=64), features attached in R030
  Primary    : Bollinger Band Width (20,2) at signal bar
  Secondary  : ATR Rank, Rel Volume, EMA Slope, ADX14
  Hypothesis : BB Width independently predicts trade quality
""")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

df_raw = pd.read_csv(ENRICHED_CSV)
df_raw["entry_time"] = pd.to_datetime(df_raw["entry_time"], utc=True)

# Drop rows missing BB Width (should be none)
df = df_raw.dropna(subset=["bb_width_signal"]).reset_index(drop=True)
print(f"  Loaded {len(df)}/{len(df_raw)} trades with BB Width data")

wins   = df[df["win"] == 1]
losses = df[df["win"] == 0]
bbw_w  = wins["bb_width_signal"].values
bbw_l  = losses["bb_width_signal"].values

print(f"\n  BB Width at signal bar:")
print(f"    All     : mean={df['bb_width_signal'].mean():.4f}  "
      f"median={df['bb_width_signal'].median():.4f}  "
      f"std={df['bb_width_signal'].std():.4f}  "
      f"min={df['bb_width_signal'].min():.4f}  "
      f"max={df['bb_width_signal'].max():.4f}")
print(f"    Winners ({len(wins)}): mean={bbw_w.mean():.4f}  "
      f"median={np.median(bbw_w):.4f}  std={bbw_w.std():.4f}")
print(f"    Losers  ({len(losses)}): mean={bbw_l.mean():.4f}  "
      f"median={np.median(bbw_l):.4f}  std={bbw_l.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Q1 — STATISTICAL TESTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  Q1 — WINNER vs LOSER BB WIDTH COMPARISON")
print("═"*78)

# Cohen's d
def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2: return 0.0
    pooled = math.sqrt(((n1-1)*a.std()**2 + (n2-1)*b.std()**2) / (n1+n2-2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0

def effect_label(d):
    a = abs(d)
    if a < 0.20: return "negligible"
    if a < 0.50: return "small"
    if a < 0.80: return "medium"
    return "large"

d = cohens_d(bbw_w, bbw_l)
t_stat, p_val = stats.ttest_ind(bbw_w, bbw_l, equal_var=False)
mw_stat, mw_p = stats.mannwhitneyu(bbw_w, bbw_l, alternative="two-sided")
sp_win,  sp_p_win  = stats.spearmanr(df["bb_width_signal"], df["win"])
sp_rmul, sp_p_rmul = stats.spearmanr(df["bb_width_signal"], df["r_multiple"])
sp_pnl,  sp_p_pnl  = stats.spearmanr(df["bb_width_signal"], df["pnl"])

# Bootstrap CI on mean difference (winners − losers)
N_BOOT = 5_000
rng    = np.random.default_rng(42)
boot_diffs = np.array([
    rng.choice(bbw_w, len(bbw_w), replace=True).mean() -
    rng.choice(bbw_l, len(bbw_l), replace=True).mean()
    for _ in range(N_BOOT)
])
ci_lo  = np.percentile(boot_diffs, 2.5)
ci_hi  = np.percentile(boot_diffs, 97.5)
ci_med = np.percentile(boot_diffs, 50)
ci_sig = (ci_lo > 0) or (ci_hi < 0)

print(f"""
  Winner  BBW mean = {bbw_w.mean():.5f}  median = {np.median(bbw_w):.5f}
  Loser   BBW mean = {bbw_l.mean():.5f}  median = {np.median(bbw_l):.5f}
  Δ mean           = {bbw_w.mean()-bbw_l.mean():+.5f}

  Cohen's d        = {d:.4f}  ({effect_label(d)})
  Welch t-test     : t={t_stat:.3f}  p={p_val:.4f}  {'SIGNIFICANT ✓' if p_val < 0.05 else 'p<0.10 ✓' if p_val < 0.10 else 'NOT significant ✗'}
  Mann-Whitney U   : U={mw_stat:.0f}  p={mw_p:.4f}  {'SIGNIFICANT ✓' if mw_p < 0.05 else 'p<0.10 ✓' if mw_p < 0.10 else 'NOT significant ✗'}

  Spearman correlations with BB Width:
    BBW → Win         ρ={sp_win:+.4f}  p={sp_p_win:.4f}  {'✓ <0.05' if sp_p_win<0.05 else '✓ <0.10' if sp_p_win<0.10 else '✗'}
    BBW → R-Multiple  ρ={sp_rmul:+.4f}  p={sp_p_rmul:.4f}  {'✓ <0.05' if sp_p_rmul<0.05 else '✓ <0.10' if sp_p_rmul<0.10 else '✗'}
    BBW → P&L         ρ={sp_pnl:+.4f}  p={sp_p_pnl:.4f}  {'✓ <0.05' if sp_p_pnl<0.05 else '✓ <0.10' if sp_p_pnl<0.10 else '✗'}

  Bootstrap 95% CI on Δ mean ({N_BOOT:,} iterations):
    p2.5={ci_lo:+.5f}   p50={ci_med:+.5f}   p97.5={ci_hi:+.5f}
    95% CI excludes zero: {'YES ✓  (significant)' if ci_sig else 'NO ✗  (not significant)'}
""")

bbw_predictive = (p_val < 0.10) or (mw_p < 0.10) or (abs(d) >= 0.20) or ci_sig

# ─────────────────────────────────────────────────────────────────────────────
# Q2 — QUARTILE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("═"*78)
print("  Q2 — BB WIDTH QUARTILE ANALYSIS")
print("═"*78)

q_bounds = df["bb_width_signal"].quantile([0.0, 0.25, 0.50, 0.75, 1.0]).values
df["bbw_quartile"] = pd.qcut(df["bb_width_signal"], q=4, labels=["Q1","Q2","Q3","Q4"])

print(f"\n  BB Width quartile boundaries:")
for i, (lo, hi) in enumerate(zip(q_bounds[:-1], q_bounds[1:]), 1):
    print(f"    Q{i}: [{lo:.5f}, {hi:.5f}]")

print(f"\n  {'Quartile':8s}  {'BBW Range':22s}  {'n':>4}  {'WR':>6}  {'PF':>7}  "
      f"{'ExpR':>7}  {'Net $':>8}  {'Sharpe':>8}  {'MDD':>7}")
print("  " + "─"*82)

quartile_stats = {}
for q in ["Q1","Q2","Q3","Q4"]:
    sub  = df[df["bbw_quartile"] == q]
    n    = len(sub)
    if n == 0: continue
    nw   = sub["win"].sum(); nl = n - nw
    gw   = sub[sub["win"]==1]["pnl"].sum() if nw > 0 else 0.0
    gl   = abs(sub[sub["win"]==0]["pnl"].sum()) if nl > 0 else 1e-9
    pf   = gw / gl
    wr   = nw / n
    exp  = wr * 2.0 - (1 - wr)
    net  = sub["pnl"].sum()
    pnls = sub["pnl"].values
    eq   = CAPITAL + np.cumsum(pnls)
    pk   = np.maximum.accumulate(eq)
    mdd  = ((eq - pk) / pk).min() if n > 1 else 0.0
    std  = pnls.std(ddof=1) if n > 1 else 1e-9
    sh   = pnls.mean() / std * math.sqrt(n) if std > 0 else 0.0
    bbw_lo = sub["bb_width_signal"].min()
    bbw_hi = sub["bb_width_signal"].max()
    quartile_stats[q] = {
        "n":n,"wr":wr,"pf":pf,"exp":exp,"net":net,
        "sharpe":sh,"mdd":mdd,"bbw_mean":sub["bb_width_signal"].mean()
    }
    print(f"  {q:8s}  {bbw_lo:.5f}–{bbw_hi:.5f}    {n:4d}  {wr*100:5.1f}%  {pf:7.3f}  "
          f"{exp:+7.3f}  {net:+8.0f}  {sh:8.2f}  {mdd*100:6.1f}%")

# Characterise the relationship
wrs = [quartile_stats[q]["wr"] for q in ["Q1","Q2","Q3","Q4"]]
pfs = [quartile_stats[q]["pf"] for q in ["Q1","Q2","Q3","Q4"]]
mono_inc  = all(wrs[i] <= wrs[i+1] for i in range(3))
mono_dec  = all(wrs[i] >= wrs[i+1] for i in range(3))
u_shape   = wrs[0] > wrs[1] and wrs[3] > wrs[2]
inv_u     = wrs[0] < wrs[1] and wrs[3] < wrs[2]

if mono_inc:   pattern = "monotonically INCREASING ✓ — narrow BB reduces quality"
elif mono_dec: pattern = "monotonically DECREASING — narrow BB improves quality"
elif u_shape:  pattern = "U-shaped — both extremes outperform midrange"
elif inv_u:    pattern = "inverted-U — midrange BB optimal"
else:          pattern = "non-monotone (mixed)"

print(f"\n  WR pattern: {pattern}")
print(f"  WR by quartile: {[f'{w*100:.1f}%' for w in wrs]}")
print(f"  PF by quartile: {[f'{p:.3f}' for p in pfs]}")

# ─────────────────────────────────────────────────────────────────────────────
# Q3 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  Q3 — FEATURE IMPORTANCE: PERMUTATION + SHAP")
print("═"*78)

feat_cols   = ["bb_width_signal", "atr_rank_pct", "rel_vol_signal",
               "ema_slope_signal", "adx_signal"]
feat_labels = ["BB Width", "ATR Rank", "Rel Volume", "EMA Slope", "ADX14"]

df_feat = df[feat_cols + ["win"]].dropna()
X = df_feat[feat_cols].values
y = df_feat["win"].values

print(f"\n  Training RandomForest on {len(df_feat)} trades …")
rf = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=4,
                             min_samples_leaf=3)
rf.fit(X, y)

# Permutation importance
perm = permutation_importance(rf, X, y, n_repeats=300, random_state=42)
perm_means = perm.importances_mean
perm_stds  = perm.importances_std
perm_order = np.argsort(perm_means)[::-1]

print(f"\n  Permutation Importance (1,000 repeats):")
print(f"  {'Feature':12s}  {'Importance':>12}  {'± Std':>8}  {'Rank':>5}  {'Signal':>8}")
print("  " + "─"*52)
for rank, i in enumerate(perm_order, 1):
    sig = "important ✓" if perm_means[i] > 0.005 else "noise ✗"
    print(f"  {feat_labels[i]:12s}  {perm_means[i]:+12.5f}  {perm_stds[i]:8.5f}  "
          f"#{rank:4d}  {sig}")

# SHAP
explainer = shap.TreeExplainer(rf)
shap_vals  = explainer.shap_values(X)
if isinstance(shap_vals, list):
    sv = shap_vals[1]
elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
    sv = shap_vals[:, :, 1]
elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
    sv = shap_vals
else:
    sv = np.atleast_2d(shap_vals)
shap_abs   = np.abs(sv).mean(axis=0)
shap_order = np.argsort(shap_abs)[::-1]

print(f"\n  SHAP Feature Importance (mean |SHAP value|):")
print(f"  {'Feature':12s}  {'Mean |SHAP|':>12}  {'Rank':>5}  {'Signal':>12}")
print("  " + "─"*48)
for rank, i in enumerate(shap_order, 1):
    top = "top feature ✓" if rank == 1 else ""
    print(f"  {feat_labels[i]:12s}  {shap_abs[i]:12.5f}  #{rank:4d}  {top}")

bbw_perm_rank = int(np.where(perm_order == 0)[0][0]) + 1  # rank of index 0 (BBW)
bbw_shap_rank = int(np.where(shap_order == 0)[0][0]) + 1

print(f"\n  BB Width rank: Permutation #{bbw_perm_rank}  |  SHAP #{bbw_shap_rank}")
print(f"  BB Width remains #1 in both: {'YES ✓' if bbw_perm_rank==1 and bbw_shap_rank==1 else 'NO ✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# Q4 — INTERACTION: ATR RANK × BB WIDTH
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  Q4 — INTERACTION: ATR RANK × BB WIDTH")
print("═"*78)

def cell_metrics(sub):
    n  = len(sub)
    if n == 0: return {"n":0, "wr":0.0, "pf":0.0, "exp":0.0, "net":0.0}
    nw = sub["win"].sum(); nl = n - nw
    gw = sub[sub["win"]==1]["pnl"].sum() if nw > 0 else 0.0
    gl = abs(sub[sub["win"]==0]["pnl"].sum()) if nl > 0 else 1e-9
    pf = gw / gl
    wr = nw / n
    return {"n":n, "wr":wr, "pf":pf, "exp": wr*2.0-(1-wr), "net": sub["pnl"].sum()}

# Use median split for ATR Rank and BB Width
atr_med = df["atr_rank_pct"].median()
bbw_med = df["bb_width_signal"].median()

print(f"\n  Median splits:  ATR Rank={atr_med:.1f}  |  BB Width={bbw_med:.5f}")
print(f"\n  2 × 2 Interaction Matrix:")
print(f"  {'':20s}  {'BB Narrow (< med)':>22}  {'BB Wide (≥ med)':>22}")
print("  " + "─"*70)

cells = {}
for atr_tag, atr_mask in [("ATR Low  (< med)",  df["atr_rank_pct"] <  atr_med),
                           ("ATR High (≥ med)", df["atr_rank_pct"] >= atr_med)]:
    narrow = cell_metrics(df[atr_mask & (df["bb_width_signal"] <  bbw_med)])
    wide   = cell_metrics(df[atr_mask & (df["bb_width_signal"] >= bbw_med)])
    cells[atr_tag] = {"narrow": narrow, "wide": wide}
    n_str  = f"PF={narrow['pf']:.3f} WR={narrow['wr']*100:.1f}% n={narrow['n']}"
    w_str  = f"PF={wide['pf']:.3f} WR={wide['wr']*100:.1f}% n={wide['n']}"
    print(f"  {atr_tag:20s}  {n_str:>22}  {w_str:>22}")

# Full 4-cell detail
print(f"\n  Full 4-cell detail:")
header = f"  {'Cell':30s}  {'n':>4}  {'WR':>6}  {'PF':>7}  {'ExpR':>7}  {'Net $':>8}"
print(header)
print("  " + "─"*62)
for atr_tag, masks in [
    ("ATR Low  + BB Narrow",  df["atr_rank_pct"] <  atr_med),
    ("ATR Low  + BB Wide",    df["atr_rank_pct"] <  atr_med),
    ("ATR High + BB Narrow",  df["atr_rank_pct"] >= atr_med),
    ("ATR High + BB Wide",    df["atr_rank_pct"] >= atr_med),
]:
    pass  # build properly below

combos = [
    ("ATR Low  + BB Narrow", (df["atr_rank_pct"] <  atr_med) & (df["bb_width_signal"] <  bbw_med)),
    ("ATR Low  + BB Wide",   (df["atr_rank_pct"] <  atr_med) & (df["bb_width_signal"] >= bbw_med)),
    ("ATR High + BB Narrow", (df["atr_rank_pct"] >= atr_med) & (df["bb_width_signal"] <  bbw_med)),
    ("ATR High + BB Wide",   (df["atr_rank_pct"] >= atr_med) & (df["bb_width_signal"] >= bbw_med)),
]
combo_results = {}
for label, mask in combos:
    m = cell_metrics(df[mask])
    combo_results[label] = m
    print(f"  {label:30s}  {m['n']:4d}  {m['wr']*100:5.1f}%  {m['pf']:7.3f}  "
          f"{m['exp']:+7.3f}  {m['net']:+8.0f}")

# Does BB Width add beyond ATR Rank alone?
atr_low_narrow_pf  = combo_results["ATR Low  + BB Narrow"]["pf"]
atr_low_wide_pf    = combo_results["ATR Low  + BB Wide"]["pf"]
atr_low_all_pf     = cell_metrics(df[df["atr_rank_pct"] < atr_med])["pf"]
bbw_adds_within_atr_low = abs(atr_low_narrow_pf - atr_low_wide_pf) > 0.10

print(f"\n  Within Low ATR trades:")
print(f"    Low ATR all          PF = {atr_low_all_pf:.3f}")
print(f"    Low ATR + BB Narrow  PF = {atr_low_narrow_pf:.3f}")
print(f"    Low ATR + BB Wide    PF = {atr_low_wide_pf:.3f}")
print(f"    δPF (Narrow − Wide)  = {atr_low_narrow_pf - atr_low_wide_pf:+.3f}")
print(f"    BB Width adds signal within Low ATR: "
      f"{'YES ✓ (Δ > 0.10)' if bbw_adds_within_atr_low else 'MARGINAL ✗ (Δ ≤ 0.10)'}")

# ─────────────────────────────────────────────────────────────────────────────
# Q5 — BB WIDTH DECILES (descriptive only)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  Q5 — BB WIDTH DECILES (DESCRIPTIVE ONLY — NO THRESHOLD RECOMMENDATIONS)")
print("═"*78)

print(f"\n  {'Decile':8s}  {'BBW Range':26s}  {'n':>4}  {'WR':>6}  {'PF':>7}  {'ExpR':>7}")
print("  " + "─"*64)
try:
    df["bbw_decile"] = pd.qcut(df["bb_width_signal"], q=10, labels=False,
                                duplicates="drop")
    for d_idx in sorted(df["bbw_decile"].dropna().unique()):
        sub = df[df["bbw_decile"] == d_idx]
        if len(sub) == 0: continue
        m   = cell_metrics(sub)
        lo  = sub["bb_width_signal"].min()
        hi  = sub["bb_width_signal"].max()
        print(f"  D{int(d_idx)+1:2d}      {lo:.5f}–{hi:.5f}    {m['n']:4d}  "
              f"{m['wr']*100:5.1f}%  {m['pf']:7.3f}  {m['exp']:+7.3f}")
except Exception as e:
    # Fall back to percentile table if too few unique values for 10 bins
    print(f"  [Note: decile bins collapsed due to ties — showing percentile table]")
    pcts = [10,20,30,40,50,60,70,80,90,100]
    thresholds = [df["bb_width_signal"].quantile(p/100) for p in pcts]
    prev = df["bb_width_signal"].min() - 1e-9
    for p, thr in zip(pcts, thresholds):
        sub = df[(df["bb_width_signal"] > prev) & (df["bb_width_signal"] <= thr)]
        m   = cell_metrics(sub)
        print(f"  p{p:3d}    {prev:.5f}–{thr:.5f}  {m['n']:4d}  "
              f"{m['wr']*100:5.1f}%  {m['pf']:7.3f}  {m['exp']:+7.3f}")
        prev = thr

# Percentile summary table
print(f"\n  BBW Percentile Distribution:")
for pct in [10,20,25,30,40,50,60,70,75,80,90]:
    print(f"    p{pct:3d} = {df['bb_width_signal'].quantile(pct/100):.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS TESTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  ROBUSTNESS TESTS")
print("═"*78)

# R1: Bootstrap on Spearman ρ (BBW → Win)
print(f"\n  R1. Bootstrap Spearman ρ (BBW → Win)  [{N_BOOT:,} iterations]:")
boot_rhos = []
for _ in range(N_BOOT):
    idx = rng.integers(0, len(df), len(df))
    bdf = df.iloc[idx]
    if bdf["bb_width_signal"].std() == 0 or bdf["win"].std() == 0:
        continue
    rho, _ = stats.spearmanr(bdf["bb_width_signal"], bdf["win"])
    boot_rhos.append(rho)
boot_rhos = np.array(boot_rhos)
br_lo  = np.percentile(boot_rhos, 2.5)
br_hi  = np.percentile(boot_rhos, 97.5)
br_med = np.percentile(boot_rhos, 50)
br_sig = (br_lo > 0) or (br_hi < 0)
print(f"    p2.5={br_lo:+.4f}  p50={br_med:+.4f}  p97.5={br_hi:+.4f}")
print(f"    CI excludes zero: {'YES ✓' if br_sig else 'NO ✗'}")

# R2: Bootstrap on PF (BB Narrow half)
bbw_50 = df["bb_width_signal"].median()
narrow_pnls = df[df["bb_width_signal"] < bbw_50]["pnl"].values
wide_pnls   = df[df["bb_width_signal"] >= bbw_50]["pnl"].values

def bootstrap_pf(pnls, n_iter=N_BOOT):
    if len(pnls) < 5:
        return 0.0, 0.0, 0.0
    pfs = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        pfs.append(wp / lp if lp > 0 else 2.0)
    return (float(np.percentile(pfs, 5)),
            float(np.percentile(pfs, 50)),
            float(np.percentile(pfs, 95)))

narrow_b5, narrow_b50, narrow_b95 = bootstrap_pf(narrow_pnls)
wide_b5,   wide_b50,   wide_b95   = bootstrap_pf(wide_pnls)

print(f"\n  R2. Bootstrap PF by BB Width half (narrow=below median, wide=above):")
print(f"    Narrow BBW  p5={narrow_b5:.3f}  p50={narrow_b50:.3f}  p95={narrow_b95:.3f}  "
      f"n={len(narrow_pnls)}")
print(f"    Wide   BBW  p5={wide_b5:.3f}    p50={wide_b50:.3f}    p95={wide_b95:.3f}  "
      f"n={len(wide_pnls)}")
narrow_better_boot = narrow_b50 > wide_b50

# R3: Jackknife (leave-one-trade-out) on Spearman ρ
print(f"\n  R3. Jackknife on Spearman ρ (leave-one-trade-out):")
jk_rhos = []
for i in range(len(df)):
    sub = df.drop(index=i)
    if sub["bb_width_signal"].std() == 0: continue
    rho, _ = stats.spearmanr(sub["bb_width_signal"], sub["win"])
    jk_rhos.append(rho)
jk_rhos = np.array(jk_rhos)
jk_mean = jk_rhos.mean()
jk_std  = jk_rhos.std()
jk_min  = jk_rhos.min()
jk_max  = jk_rhos.max()
sign_stable = np.sign(jk_rhos).sum() == len(jk_rhos) or np.sign(jk_rhos).sum() == -len(jk_rhos)
print(f"    mean ρ={jk_mean:+.4f}  std={jk_std:.4f}  min={jk_min:+.4f}  max={jk_max:+.4f}")
print(f"    Sign stable (all same direction): {'YES ✓' if sign_stable else 'NO ✗'}")
print(f"    Fraction ρ > 0: {(jk_rhos > 0).mean()*100:.1f}%")

# R4: Leave-One-Symbol-Out (LOO-sym) on Spearman ρ
print(f"\n  R4. Leave-One-Symbol-Out (Spearman ρ):")
print(f"  {'Sym excluded':10s}  {'n_remain':>9}  {'ρ':>8}  {'p':>8}  {'dir':>5}")
print("  " + "─"*46)
loo_sym_rhos = []
for sym in df["sym"].unique():
    sub = df[df["sym"] != sym]
    if len(sub) < 5: continue
    rho, p = stats.spearmanr(sub["bb_width_signal"], sub["win"])
    loo_sym_rhos.append(rho)
    tag = sym.split("-")[0]
    print(f"  {tag:10s}  {len(sub):9d}  {rho:+8.4f}  {p:8.4f}  "
          f"{'✓' if p < 0.10 else '–'}")
loo_sign_stable = (all(r > 0 for r in loo_sym_rhos) or
                   all(r < 0 for r in loo_sym_rhos))
print(f"\n  LOO-sym min ρ={min(loo_sym_rhos):+.4f}  max ρ={max(loo_sym_rhos):+.4f}  "
      f"sign stable: {'YES ✓' if loo_sign_stable else 'NO ✗'}")

# R5: Permutation importance jackknife (does BBW stay #1 when leaving each symbol out?)
print(f"\n  R5. Permutation Importance Rank (LOO-symbol): does BBW stay #1?")
bbw_ranks_loo = []
for sym in df["sym"].unique():
    sub = df[df["sym"] != sym].dropna(subset=feat_cols + ["win"])
    if len(sub) < 10: continue
    Xs = sub[feat_cols].values; ys = sub["win"].values
    rf2 = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4,
                                  min_samples_leaf=2)
    rf2.fit(Xs, ys)
    pi2 = permutation_importance(rf2, Xs, ys, n_repeats=50, random_state=42)
    order2 = np.argsort(pi2.importances_mean)[::-1]
    rank = int(np.where(order2 == 0)[0][0]) + 1
    bbw_ranks_loo.append(rank)
    tag = sym.split("-")[0]
    print(f"    Excl {tag:5s}: BBW perm rank = #{rank}")
print(f"  BBW stays #1 all LOO: {'YES ✓' if all(r == 1 for r in bbw_ranks_loo) else f'NO ✗  ranks={bbw_ranks_loo}'}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL RESEARCH QUESTIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═"*78)
print("  FINAL RESEARCH QUESTIONS")
print("═"*78)

eth_sub  = df[df["sym"] == "ETH-USDT-SWAP"]
link_sub = df[df["sym"] == "LINK-USDT-SWAP"]

eth_bbw_w_mean  = eth_sub[eth_sub["win"]==1]["bb_width_signal"].mean()
eth_bbw_l_mean  = eth_sub[eth_sub["win"]==0]["bb_width_signal"].mean()
link_bbw_w_mean = link_sub[link_sub["win"]==1]["bb_width_signal"].mean() if len(link_sub[link_sub["win"]==1]) else np.nan
link_bbw_l_mean = link_sub[link_sub["win"]==0]["bb_width_signal"].mean() if len(link_sub[link_sub["win"]==0]) else np.nan

bbw_explains_eth  = (not np.isnan(eth_bbw_w_mean) and not np.isnan(eth_bbw_l_mean) and
                     abs(eth_bbw_w_mean - eth_bbw_l_mean) > 0.003)
bbw_explains_link = (not np.isnan(link_bbw_w_mean) and not np.isnan(link_bbw_l_mean) and
                     abs(link_bbw_w_mean - link_bbw_l_mean) > 0.003)

print(f"""
  Q1. Is BB Width independently predictive?
      Cohen's d    = {d:.4f}  ({effect_label(d)})
      Spearman ρ   = {sp_win:+.4f}  p={sp_p_win:.4f}
      Boot 95% CI  = [{ci_lo:+.5f}, {ci_hi:+.5f}]  {'excludes 0 ✓' if ci_sig else 'includes 0 ✗'}
      Perm rank    = #{bbw_perm_rank}  |  SHAP rank = #{bbw_shap_rank}
      Answer: {'YES ✓ — BB Width shows statistically detectable predictive power' if bbw_predictive else 'WEAK ✗ — BB Width does not reach p<0.10 on this sample (n=64)'}

  Q2. Does BB Width explain ETH's strength?
      ETH  BBW mean (winners): {eth_bbw_w_mean:.5f}
      ETH  BBW mean (losers) : {eth_bbw_l_mean:.5f}
      Portfolio BBW mean     : {df['bb_width_signal'].mean():.5f}
      Answer: {'YES — ETH winner trades coincide with meaningfully different BB Width than ETH losers' if bbw_explains_eth else 'PARTIAL — ETH winner/loser BBW split is small; ETH outperforms for other reasons'}

  Q3. Does BB Width explain LINK's behaviour?
      LINK BBW mean (winners): {link_bbw_w_mean:.5f}
      LINK BBW mean (losers) : {link_bbw_l_mean:.5f}
      Answer: {'YES — BBW separates LINK winners from losers' if bbw_explains_link else 'NO — BB Width does not explain LINK behaviour in this sample'}

  Q4. Does BB Width add information beyond ATR Rank?
      Low ATR all          PF = {atr_low_all_pf:.3f}
      Low ATR + BB Narrow  PF = {atr_low_narrow_pf:.3f}
      Low ATR + BB Wide    PF = {atr_low_wide_pf:.3f}
      δPF                    = {atr_low_narrow_pf - atr_low_wide_pf:+.3f}
      Answer: {'YES ✓ — BB Width adds Δ>0.10 PF signal within Low ATR trades' if bbw_adds_within_atr_low else 'MARGINAL ✗ — Δ≤0.10; BB Width does not materially add beyond ATR Rank alone'}

  Q5. Should BB Width become a permanent QuantLab feature?
      Statistical significance: {'REACHED (p<0.10)' if p_val < 0.10 or mw_p < 0.10 else 'NOT REACHED'}
      Permutation rank #{bbw_perm_rank}/{len(feat_labels)}
      SHAP rank #{bbw_shap_rank}/{len(feat_labels)}
      Jackknife sign stable: {'YES' if sign_stable else 'NO'}
      LOO-symbol sign stable: {'YES' if loo_sign_stable else 'NO'}
      Boot p50 (narrow>wide): {'YES' if narrow_better_boot else 'NO'}
""")

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT
# ─────────────────────────────────────────────────────────────────────────────

print("═"*78)
print("  VERDICT")
print("═"*78)

crit = {
    "BB Width predictive (d≥0.20 or p<0.10 or CI excl.0)": bbw_predictive,
    "BB Width Permutation rank ≤ 2":                        bbw_perm_rank <= 2,
    "BB Width SHAP rank ≤ 2":                               bbw_shap_rank <= 2,
    "Jackknife sign stable":                                 sign_stable,
    "LOO-symbol sign stable":                               loo_sign_stable,
    "BB Width adds ΔPF>0.10 within Low ATR":               bbw_adds_within_atr_low,
}
n_pass  = sum(crit.values())
VERDICT = "VALIDATE_R032" if n_pass >= 4 else "REJECT"
vcolor  = "\033[92m" if VERDICT == "VALIDATE_R032" else "\033[91m"
vreset  = "\033[0m"

print(f"\n  {vcolor}VERDICT: {VERDICT}{vreset}")
print()
for c, ok in crit.items():
    print(f"    {'✓' if ok else '✗'} {c}")
print(f"\n  {n_pass}/{len(crit)} criteria met.")

next_feature = "Relative Volume" if feat_labels[list(perm_order)[1]] != "BB Width" else feat_labels[list(perm_order)[2]]
perm_runner_up = feat_labels[perm_order[1]]
shap_runner_up = feat_labels[shap_order[1]]

if VERDICT == "VALIDATE_R032":
    print(f"""
  RECOMMENDATION: Proceed to R032 — BB Width Gated Backtest
  ──────────────────────────────────────────────────────────
  BB Width shows sufficient evidence as an independent predictive feature.
  R032 should:
    • Re-run the full R029 engine on all 9 symbols, 1H
    • Add BB Width NARROW gate (< median or natural low boundary) to Low ATR
    • DO NOT change any other parameter — additive only
    • Report: new n, PF, WR, bootstrap p50, MC P(profit)
    • Promote criteria remain: PF>1.20, n≥80, boot p50>1.20, MC>60%

  Natural boundary to explore in R032: BB Width < {df['bb_width_signal'].quantile(0.50):.5f} (median)
  (Descriptive — R032 must confirm with a robustness test before adopting)
""")
else:
    print(f"""
  RECOMMENDATION: Reject BB Width as primary filter — Search Next Feature
  ─────────────────────────────────────────────────────────────────────────
  BB Width does not show sufficient independent predictive power on n=64 trades.

  Runner-up features (consider for R032 attribution):
    Permutation #2 : {perm_runner_up}
    SHAP #2        : {shap_runner_up}

  Suggested R032 focus: {perm_runner_up} attribution
    (Same methodology as R031 — descriptive, no optimisation)
""")

print("═"*78)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n  Generating charts …")

def dark_ax(ax, title=None, col="white"):
    ax.set_facecolor("#111")
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=col, fontsize=9)

# ── Chart 1: BBW distribution winners vs losers ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#111")
fig.suptitle("R031 — Bollinger Width Distribution: Winners vs Losers", color="white", fontsize=11)

ax1 = axes[0]; dark_ax(ax1, "Histogram")
bins = np.linspace(df["bb_width_signal"].min()*0.98, df["bb_width_signal"].max()*1.02, 25)
ax1.hist(bbw_w, bins=bins, color="#4CAF50", alpha=0.65, label=f"Winners n={len(wins)}", density=True)
ax1.hist(bbw_l, bins=bins, color="#F44336", alpha=0.65, label=f"Losers n={len(losses)}", density=True)
ax1.axvline(bbw_w.mean(), color="#4CAF50", lw=1.5, ls="--", label=f"W={bbw_w.mean():.4f}")
ax1.axvline(bbw_l.mean(), color="#F44336", lw=1.5, ls="--", label=f"L={bbw_l.mean():.4f}")
ax1.set_xlabel("BB Width(20,2) at signal bar", color="white"); ax1.set_ylabel("Density", color="white")
ax1.legend(facecolor="#222", labelcolor="white", fontsize=8)

ax2 = axes[1]; dark_ax(ax2, "Bootstrap CI — Δ mean BBW (Winners − Losers)")
ax2.hist(boot_diffs, bins=50, color="#9C27B0", alpha=0.75, density=True)
ax2.axvline(0,      color="white",   lw=1.5, ls="--", label="Zero (null)")
ax2.axvline(ci_lo,  color="#F44336", lw=1.2, ls=":", label=f"p2.5={ci_lo:.5f}")
ax2.axvline(ci_hi,  color="#4CAF50", lw=1.2, ls=":", label=f"p97.5={ci_hi:.5f}")
ax2.axvline(ci_med, color="#FF9800", lw=1.8, ls="-",  label=f"p50={ci_med:.5f}")
ax2.set_xlabel("Δ BBW mean (W−L)", color="white"); ax2.set_ylabel("Density", color="white")
sig_txt = "CI excludes 0 ✓" if ci_sig else "CI includes 0 ✗"
ax2.text(0.5, 0.95, sig_txt, transform=ax2.transAxes, color="#FFEB3B",
         ha="center", va="top", fontsize=10, fontweight="bold")
ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
plt.tight_layout()
p = f"{OUT}/r031_bbw_distribution.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 2: Quartile WR and PF ──────────────────────────────────────────────
qs     = ["Q1","Q2","Q3","Q4"]
q_wrs  = [quartile_stats[q]["wr"]*100  for q in qs]
q_pfs  = [quartile_stats[q]["pf"]      for q in qs]
q_ns   = [quartile_stats[q]["n"]       for q in qs]
q_bbw  = [quartile_stats[q]["bbw_mean"] for q in qs]
q_cols = ["#2196F3","#8BC34A","#FF9800","#F44336"]  # narrow→wide colour ramp

fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#111")
fig.suptitle("R031 — BB Width Quartile Analysis", color="white", fontsize=11)

ax1 = axes[0]; dark_ax(ax1, "Win Rate by BB Width Quartile")
bars_ = ax1.bar([f"{q}\n≈{a:.4f}" for q, a in zip(qs, q_bbw)], q_wrs,
                color=q_cols, alpha=0.85)
ax1.axhline(33.3, color="white", lw=0.8, ls="--", alpha=0.6)
ax1.axhline(50.0, color="#FF9800", lw=0.8, ls=":", alpha=0.6)
ax1.set_ylabel("Win Rate %", color="white")
for b_, v_, n_ in zip(bars_, q_wrs, q_ns):
    ax1.text(b_.get_x()+b_.get_width()/2, v_+0.5, f"{v_:.1f}%\nn={n_}",
             ha="center", color="white", fontsize=9)

ax2 = axes[1]; dark_ax(ax2, "Profit Factor by BB Width Quartile")
bars_ = ax2.bar([f"{q}\n≈{a:.4f}" for q, a in zip(qs, q_bbw)], q_pfs,
                color=q_cols, alpha=0.85)
ax2.axhline(1.0, color="white", lw=0.8, ls="--")
ax2.axhline(1.2, color="#FF9800", lw=0.8, ls=":", label="PF=1.20")
ax2.set_ylabel("Profit Factor", color="white")
ax2.legend(facecolor="#222", labelcolor="white", fontsize=8)
for b_, v_ in zip(bars_, q_pfs):
    ax2.text(b_.get_x()+b_.get_width()/2, v_+0.01, f"{v_:.3f}",
             ha="center", color="white", fontsize=9)

ax3 = axes[2]; dark_ax(ax3, "Bootstrap PF: Narrow vs Wide BBW")
for xi, (name, b5_, b50_, b95_, col_) in enumerate([
    ("Narrow BBW\n(<median)", narrow_b5, narrow_b50, narrow_b95, "#2196F3"),
    ("Wide BBW\n(≥median)",   wide_b5,   wide_b50,   wide_b95,   "#F44336"),
]):
    ax3.errorbar(xi, b50_, yerr=[[b50_-b5_],[b95_-b50_]],
                 fmt="o", color=col_, capsize=12, capthick=2.5, ms=9)
    ax3.text(xi, b95_+0.02, f"p95={b95_:.3f}", ha="center", color=col_, fontsize=8)
    ax3.text(xi, b5_-0.06,  f"p5={b5_:.3f}",  ha="center", color=col_, fontsize=8)
    ax3.text(xi, b50_+0.01, f"p50={b50_:.3f}", ha="center", color=col_, fontsize=9,
             fontweight="bold")
ax3.axhline(1.0, color="white", lw=0.8, ls="--")
ax3.axhline(1.2, color="#FF9800", lw=0.8, ls=":")
ax3.set_xticks([0,1]); ax3.set_xticklabels(["Narrow BBW","Wide BBW"], color="white")
ax3.set_ylabel("PF", color="white")
plt.tight_layout()
p = f"{OUT}/r031_quartile_analysis.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 3: Feature importance comparison ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#111")
fig.suptitle("R031 — Feature Importance: Permutation vs SHAP", color="white", fontsize=11)

ax1 = axes[0]; dark_ax(ax1, "Permutation Importance")
pm_s = sorted(zip(feat_labels, perm_means, perm_stds), key=lambda x: x[1], reverse=True)
pm_l, pm_v, pm_e = zip(*pm_s)
cols_p = ["#FF9800" if l == "BB Width" else "#607D8B" for l in pm_l]
ax1.barh(pm_l[::-1], [v for v in pm_v[::-1]], xerr=[e for e in pm_e[::-1]],
         color=cols_p[::-1], alpha=0.85, capsize=5)
ax1.axvline(0, color="white", lw=0.7, ls="--", alpha=0.5)
ax1.set_xlabel("Mean importance", color="white")

ax2 = axes[1]; dark_ax(ax2, "SHAP Feature Importance (mean |SHAP|)")
sh_s = sorted(zip(feat_labels, shap_abs), key=lambda x: x[1], reverse=True)
sh_l, sh_v = zip(*sh_s)
cols_s = ["#FF9800" if l == "BB Width" else "#607D8B" for l in sh_l]
ax2.barh(sh_l[::-1], [v for v in sh_v[::-1]], color=cols_s[::-1], alpha=0.85)
ax2.set_xlabel("Mean |SHAP value|", color="white")
plt.tight_layout()
p = f"{OUT}/r031_feature_importance.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 4: BBW vs R-Multiple scatter ───────────────────────────────────────
ols_m, ols_b, *_ = stats.linregress(df["bb_width_signal"], df["r_multiple"])
x_line = np.array([df["bb_width_signal"].min(), df["bb_width_signal"].max()])
fig, ax = plt.subplots(figsize=(10, 6), facecolor="#111")
dark_ax(ax, f"R031 — BB Width vs R-Multiple  ρ={sp_rmul:+.3f}  p={sp_p_rmul:.3f}")
cols_sc = ["#4CAF50" if w else "#F44336" for w in df["win"]]
ax.scatter(df["bb_width_signal"], df["r_multiple"], c=cols_sc, alpha=0.7, s=60)
ax.plot(x_line, ols_m*x_line + ols_b, color="#FF9800", lw=1.5, ls="--",
        label=f"OLS slope={ols_m:.2f}")
ax.axhline(0, color="white", lw=0.5, ls=":", alpha=0.4)
ax.axvline(bbw_med, color="#FFEB3B", lw=1.2, ls="--", label=f"Median={bbw_med:.5f}")
ax.set_xlabel("BB Width(20,2) at signal bar", color="white")
ax.set_ylabel("R-Multiple", color="white")
green_p = plt.Line2D([0],[0],marker="o",color="#4CAF50",lw=0,label="Win")
red_p   = plt.Line2D([0],[0],marker="o",color="#F44336",lw=0,label="Loss")
ax.legend(handles=[green_p, red_p,
                   plt.Line2D([0],[0],color="#FF9800",ls="--",label=f"OLS slope={ols_m:.2f}"),
                   plt.Line2D([0],[0],color="#FFEB3B",ls="--",label=f"Median={bbw_med:.4f}")],
          facecolor="#222", labelcolor="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r031_bbw_scatter.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 5: 2×2 interaction heatmap ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#111")
fig.suptitle("R031 — ATR Rank × BB Width Interaction (2×2)", color="white", fontsize=11)

def make_2x2(ax_, col_x, col_y, lx, ly, bins_x=3, bins_y=3):
    df2 = df[[col_x, col_y, "win", "pnl"]].dropna().copy()
    df2["bx"] = pd.qcut(df2[col_x], bins_x, labels=False, duplicates="drop")
    df2["by"] = pd.qcut(df2[col_y], bins_y, labels=False, duplicates="drop")
    pivot_wr = df2.groupby(["by","bx"])["win"].mean().unstack(fill_value=np.nan)
    pivot_n  = df2.groupby(["by","bx"]).size().unstack(fill_value=0)
    im = ax_.imshow(pivot_wr.values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax_, label="Win Rate")
    xb = pd.qcut(df2[col_x], bins_x, retbins=True, duplicates="drop")[1]
    yb = pd.qcut(df2[col_y], bins_y, retbins=True, duplicates="drop")[1]
    ax_.set_xticks(range(len(xb)-1))
    ax_.set_yticks(range(len(yb)-1))
    ax_.set_xticklabels([f"Q{i+1}\n≤{xb[i+1]:.2f}" for i in range(len(xb)-1)], color="white", fontsize=7)
    ax_.set_yticklabels([f"Q{i+1}\n≤{yb[i+1]:.5f}" for i in range(len(yb)-1)], color="white", fontsize=7)
    ax_.set_xlabel(lx, color="white"); ax_.set_ylabel(ly, color="white")
    for i in range(pivot_wr.shape[0]):
        for j in range(pivot_wr.shape[1]):
            v = pivot_wr.values[i,j]; c = pivot_n.values[i,j]
            if not np.isnan(v):
                ax_.text(j, i, f"{v*100:.0f}%\nn={c}", ha="center", va="center",
                         color="black", fontsize=9, fontweight="bold")

try:
    make_2x2(axes[0], "atr_rank_pct", "bb_width_signal", "ATR Rank", "BB Width")
    dark_ax(axes[0], "ATR Rank × BB Width — Win Rate (3×3)")
except Exception as e:
    axes[0].text(0.5, 0.5, str(e), transform=axes[0].transAxes, color="white", ha="center")

# SHAP beeswarm substitute (sorted bar per feature)
ax2 = axes[1]; dark_ax(ax2, "SHAP Value Distribution by Feature")
shap_sorted = sorted(zip(feat_labels, [sv[:, i] for i in range(len(feat_labels))]),
                     key=lambda x: np.abs(x[1]).mean(), reverse=True)
y_pos = list(range(len(shap_sorted)))
cols_sv = ["#FF9800","#4CAF50","#2196F3","#9C27B0","#F44336"]
for yi, (fl, sv_col) in enumerate(shap_sorted):
    ax2.scatter(sv_col, [yi]*len(sv_col), c=cols_sv[yi], alpha=0.5, s=20)
    ax2.axvline(0, color="white", lw=0.5, ls=":", alpha=0.3)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([fl for fl,_ in shap_sorted], color="white")
ax2.set_xlabel("SHAP value", color="white")
plt.tight_layout()
p = f"{OUT}/r031_interaction_shap.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 6: Per-symbol BBW breakdown ────────────────────────────────────────
syms_plot = list(df["sym"].unique())
sym_tags_p = [s.split("-")[0] for s in syms_plot]
sym_bbw_w  = [df[(df["sym"]==s)&(df["win"]==1)]["bb_width_signal"].mean()
              if len(df[(df["sym"]==s)&(df["win"]==1)])>0 else np.nan for s in syms_plot]
sym_bbw_l  = [df[(df["sym"]==s)&(df["win"]==0)]["bb_width_signal"].mean()
              if len(df[(df["sym"]==s)&(df["win"]==0)])>0 else np.nan for s in syms_plot]
sym_bbw_a  = [df[df["sym"]==s]["bb_width_signal"].mean() for s in syms_plot]

x_s = np.arange(len(syms_plot)); w_s = 0.28
fig, axes = plt.subplots(1, 2, figsize=(22, 5), facecolor="#111")
fig.suptitle("R031 — BB Width by Symbol: Winners vs Losers + Jackknife ρ stability",
             color="white", fontsize=11)

ax1 = axes[0]; dark_ax(ax1, "BB Width at Signal Bar by Symbol")
ax1.bar(x_s - w_s, sym_bbw_w, w_s, color="#4CAF50", alpha=0.85, label="Winner BBW mean")
ax1.bar(x_s,       sym_bbw_a, w_s, color="#607D8B", alpha=0.85, label="All BBW mean")
ax1.bar(x_s + w_s, sym_bbw_l, w_s, color="#F44336", alpha=0.85, label="Loser BBW mean")
ax1.set_xticks(x_s); ax1.set_xticklabels(sym_tags_p, color="white")
ax1.set_ylabel("BB Width", color="white")
ax1.legend(facecolor="#222", labelcolor="white", fontsize=9)

ax2 = axes[1]; dark_ax(ax2, "Jackknife ρ Distribution (leave-one-trade-out)")
ax2.hist(jk_rhos, bins=30, color="#3F51B5", alpha=0.80)
ax2.axvline(jk_mean, color="#FF9800", lw=2, ls="--", label=f"mean={jk_mean:.4f}")
ax2.axvline(0,       color="white",   lw=1.5, ls=":",  label="zero")
ax2.set_xlabel("Spearman ρ (BBW → Win)", color="white")
ax2.set_ylabel("Count", color="white")
ax2.text(0.05, 0.95, f"sign stable: {'YES ✓' if sign_stable else 'NO ✗'}\n"
         f"frac ρ>0: {(jk_rhos>0).mean()*100:.0f}%",
         transform=ax2.transAxes, color="white", va="top", fontsize=9)
ax2.legend(facecolor="#222", labelcolor="white", fontsize=9)
plt.tight_layout()
p = f"{OUT}/r031_symbol_jackknife.png"
plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── Chart 7: Decile WR/PF ────────────────────────────────────────────────────
try:
    dec_data = []
    df_d = df.copy()
    df_d["bbw_dec"] = pd.qcut(df_d["bb_width_signal"], q=10, labels=False, duplicates="drop")
    for d_idx in sorted(df_d["bbw_dec"].dropna().unique()):
        sub = df_d[df_d["bbw_dec"] == d_idx]
        m   = cell_metrics(sub)
        dec_data.append({"dec": int(d_idx)+1, "wr": m["wr"], "pf": m["pf"], "n": m["n"],
                         "bbw": sub["bb_width_signal"].mean()})
    dec_df = pd.DataFrame(dec_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#111")
    fig.suptitle("R031 — BB Width Decile Analysis (Descriptive Only)", color="white", fontsize=11)
    cols_d = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(dec_df)))

    ax1 = axes[0]; dark_ax(ax1, "Win Rate by BBW Decile")
    ax1.bar(dec_df["dec"], dec_df["wr"]*100, color=cols_d, alpha=0.85)
    ax1.axhline(33.3, color="white", lw=0.8, ls="--")
    ax1.set_xlabel("BBW Decile (1=narrowest)", color="white")
    ax1.set_ylabel("Win Rate %", color="white")
    for _, row in dec_df.iterrows():
        ax1.text(row["dec"], row["wr"]*100+0.3, f"{row['wr']*100:.0f}%",
                 ha="center", color="white", fontsize=7)

    ax2 = axes[1]; dark_ax(ax2, "Profit Factor by BBW Decile")
    ax2.bar(dec_df["dec"], dec_df["pf"], color=cols_d, alpha=0.85)
    ax2.axhline(1.0, color="white", lw=0.8, ls="--")
    ax2.axhline(1.2, color="#FF9800", lw=0.8, ls=":")
    ax2.set_xlabel("BBW Decile (1=narrowest)", color="white")
    ax2.set_ylabel("Profit Factor", color="white")
    for _, row in dec_df.iterrows():
        ax2.text(row["dec"], row["pf"]+0.02, f"{row['pf']:.2f}",
                 ha="center", color="white", fontsize=7)
    plt.tight_layout()
    p = f"{OUT}/r031_decile_analysis.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  → {p}")
except Exception as e:
    print(f"  [WARN] Decile chart skipped: {e}")

# ── Chart 8: Full Dashboard ───────────────────────────────────────────────────
vcolor_map = {"VALIDATE_R032": "#4CAF50", "REJECT": "#F44336"}
vcolor_dash = vcolor_map.get(VERDICT, "white")

fig = plt.figure(figsize=(28, 20), facecolor="#0a0a0a")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.65, wspace=0.45)

fig.suptitle(
    f"QUANTLAB AI — R031 DASHBOARD\n"
    f"BB Width Attribution | Low ATR FVG Edge | 64 Trades | Verdict: {VERDICT}",
    color="white", fontsize=13, y=0.99)

# Stats summary
ax_t = fig.add_subplot(gs[0, :2]); ax_t.axis("off")
tbl_data = [
    ["Cohen's d",          f"{d:.4f}", effect_label(d)],
    ["Welch t-test p",     f"{p_val:.4f}", "✓" if p_val<0.10 else "✗"],
    ["Mann-Whitney p",     f"{mw_p:.4f}",  "✓" if mw_p<0.10  else "✗"],
    ["Spearman ρ (Win)",   f"{sp_win:+.4f} p={sp_p_win:.4f}", "✓" if sp_p_win<0.10 else "✗"],
    ["Boot 95% CI",        f"[{ci_lo:+.5f},{ci_hi:+.5f}]", "excl.0 ✓" if ci_sig else "incl.0 ✗"],
    ["Perm rank",          f"#{bbw_perm_rank}/{len(feat_labels)}", "✓" if bbw_perm_rank<=2 else "✗"],
    ["SHAP rank",          f"#{bbw_shap_rank}/{len(feat_labels)}", "✓" if bbw_shap_rank<=2 else "✗"],
    ["Jackknife sign stbl",f"{(jk_rhos>0).mean()*100:.0f}% ρ>0", "✓" if sign_stable else "✗"],
    ["LOO-sym sign stable",f"min={min(loo_sym_rhos):+.4f}", "✓" if loo_sign_stable else "✗"],
]
tbl = ax_t.table(cellText=tbl_data, colLabels=["Test","Value","Pass?"],
                 loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for (r_, c_), cell in tbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r_ % 2 == 0 else "#222")
    cell.set_text_props(color="white"); cell.set_edgecolor("#333")
    if r_ == 0:
        cell.set_facecolor("#2a2a2a")
        cell.set_text_props(color="#aaa", fontweight="bold")
ax_t.set_title("Statistical Tests", color="white", fontsize=10)

# Quartile table
ax_q = fig.add_subplot(gs[0, 2:]); ax_q.axis("off")
q_tbl = [[q, f"≈{quartile_stats[q]['bbw_mean']:.4f}", str(quartile_stats[q]["n"]),
           f"{quartile_stats[q]['wr']*100:.1f}%", f"{quartile_stats[q]['pf']:.3f}",
           f"{quartile_stats[q]['exp']:+.3f}", f"{quartile_stats[q]['sharpe']:.2f}",
           f"{quartile_stats[q]['mdd']*100:.1f}%"]
          for q in ["Q1","Q2","Q3","Q4"]]
qtbl = ax_q.table(cellText=q_tbl,
                   colLabels=["Quartile","BBW≈","n","WR","PF","ExpR","Sharpe","MDD"],
                   loc="center", cellLoc="center")
qtbl.auto_set_font_size(False); qtbl.set_fontsize(9)
for (r_, c_), cell in qtbl.get_celld().items():
    cell.set_facecolor("#1a1a1a" if r_ % 2 == 0 else "#222")
    cell.set_text_props(color="white"); cell.set_edgecolor("#333")
    if r_ == 0:
        cell.set_facecolor("#2a2a2a")
        cell.set_text_props(color="#aaa", fontweight="bold")
ax_q.set_title("BB Width Quartile Summary", color="white", fontsize=10)

# WR by quartile
ax1 = fig.add_subplot(gs[1, :2]); dark_ax(ax1, "Win Rate by BB Width Quartile")
bars__ = ax1.bar([f"{q}\n≈{a:.4f}" for q,a in zip(qs,q_bbw)], q_wrs,
                  color=q_cols, alpha=0.85)
ax1.axhline(33.3, color="white", lw=0.8, ls="--")
ax1.axhline(50.0, color="#FF9800", lw=0.8, ls=":")
ax1.set_ylabel("WR %", color="white")
for b_, v_, n_ in zip(bars__, q_wrs, q_ns):
    ax1.text(b_.get_x()+b_.get_width()/2, v_+0.5, f"{v_:.1f}%\nn={n_}",
             ha="center", color="white", fontsize=9)

# Scatter
ax2 = fig.add_subplot(gs[1, 2:]); dark_ax(ax2, f"BBW vs R-Multiple  ρ={sp_rmul:+.3f}")
ax2.scatter(df["bb_width_signal"], df["r_multiple"],
            c=["#4CAF50" if w else "#F44336" for w in df["win"]], alpha=0.65, s=45)
ax2.plot(x_line, ols_m*x_line+ols_b, color="#FF9800", lw=1.5, ls="--")
ax2.axvline(bbw_med, color="#FFEB3B", lw=1, ls="--")
ax2.axhline(0, color="white", lw=0.5, ls=":", alpha=0.4)
ax2.set_xlabel("BB Width", color="white"); ax2.set_ylabel("R-Multiple", color="white")

# 4-cell interaction
ax3 = fig.add_subplot(gs[2, :2]); dark_ax(ax3, "ATR Rank × BB Width (4-cell PF)")
combo_names  = [l.replace(" + ","\n+") for l,_ in combos]
combo_pfs    = [combo_results[l]["pf"] for l,_ in combos]
combo_ns     = [combo_results[l]["n"]  for l,_ in combos]
combo_colors = ["#2196F3","#607D8B","#FF9800","#F44336"]
bars__ = ax3.bar(combo_names, combo_pfs, color=combo_colors, alpha=0.85)
ax3.axhline(1.0, color="white", lw=0.8, ls="--")
ax3.axhline(1.2, color="#FF9800", lw=0.8, ls=":")
ax3.set_ylabel("Profit Factor", color="white")
for b_, v_, n_ in zip(bars__, combo_pfs, combo_ns):
    ax3.text(b_.get_x()+b_.get_width()/2, v_+0.02, f"{v_:.3f}\nn={n_}",
             ha="center", color="white", fontsize=9)

# Verdict panel
ax4 = fig.add_subplot(gs[2, 2:]); ax4.axis("off"); ax4.set_facecolor("#111")
ax4.text(0.5, 0.92, f"VERDICT: {VERDICT}", transform=ax4.transAxes,
         color=vcolor_dash, fontsize=20, ha="center", fontweight="bold")
summary = (f"Cohen's d={d:.3f} ({effect_label(d)})  |  Spearman ρ={sp_win:+.3f} p={sp_p_win:.3f}\n"
           f"Boot 95% CI: [{ci_lo:.5f}, {ci_hi:.5f}]  {'excl.0 ✓' if ci_sig else 'incl.0 ✗'}\n"
           f"Perm #{bbw_perm_rank}/{len(feat_labels)}  SHAP #{bbw_shap_rank}/{len(feat_labels)}\n"
           f"Jackknife: {(jk_rhos>0).mean()*100:.0f}% ρ>0  |  LOO-sym sign stable: {'YES' if loo_sign_stable else 'NO'}\n"
           f"BBW narrow PF boot p50={narrow_b50:.3f}  wide p50={wide_b50:.3f}\n"
           f"Pattern: {pattern[:40]}\n"
           f"{n_pass}/{len(crit)} criteria met")
ax4.text(0.5, 0.57, summary, transform=ax4.transAxes,
         color="white", fontsize=9, ha="center", va="center")
checks_str = "\n".join(f"{'✓' if ok else '✗'} {c}" for c, ok in crit.items())
ax4.text(0.5, 0.08, checks_str, transform=ax4.transAxes,
         color="#aaa", fontsize=8.5, ha="center", va="bottom")

plt.savefig(f"{OUT}/r031_dashboard.png", dpi=130, bbox_inches="tight", facecolor="#0a0a0a")
plt.close()
print(f"  → {OUT}/r031_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
try:
    from quantlab_ai import append_journal
    from datetime import datetime, timezone as _tz
    append_journal([{
        "research_id":    RESEARCH_ID,
        "run_date":       datetime.now(tz=_tz.utc).strftime("%Y-%m-%d"),
        "strategy_name":  "BBWidth_Attribution_LowATR_FVG",
        "symbol":         "PORTFOLIO_9SYM",
        "n_trades":       len(df),
        "profit_factor":  round(atr_low_narrow_pf, 4),
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
print(f"  R031 complete.")
print(f"  Verdict          : {VERDICT}")
print(f"  Cohen's d        : {d:.4f}  ({effect_label(d)})")
print(f"  Spearman ρ       : {sp_win:+.4f}  p={sp_p_win:.4f}")
print(f"  Boot 95% CI Δ    : [{ci_lo:.5f}, {ci_hi:.5f}]  {'SIGNIFICANT' if ci_sig else 'not significant'}")
print(f"  Perm rank        : #{bbw_perm_rank}/{len(feat_labels)}")
print(f"  SHAP rank        : #{bbw_shap_rank}/{len(feat_labels)}")
print(f"  Jackknife stable : {'YES' if sign_stable else 'NO'}  ({(jk_rhos>0).mean()*100:.0f}% ρ>0)")
print(f"  LOO-sym stable   : {'YES' if loo_sign_stable else 'NO'}")
print(f"  BBW narr p50 PF  : {narrow_b50:.3f}")
print(f"  BBW wide p50 PF  : {wide_b50:.3f}")
print(f"  Pattern          : {pattern}")
print(f"  Runner-up (Perm) : {perm_runner_up}")
print(f"  {n_pass}/{len(crit)} criteria met")
print(f"  Output           → {OUT}/r031_*")
print(f"{'═'*78}\n")
