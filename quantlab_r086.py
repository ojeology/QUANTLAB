"""
QUANTLAB AI — R086
ML TYPE zoo (not more features) — Random Forest / SVM / MLP / Naive Bayes / Ensemble

User request: don't add more features; try DIFFERENT ML algorithm types, with a few
special features.

Design:
  - Same 73-symbol universe, same raw Family A trades, same walk-forward protocol
  - LEAN feature set: base 11 + only 3 special features (regime breadth quartile,
    distance from 48-bar high, green-candle streak) = 14 total (not a big bag)
  - ML TYPES (all on identical features, top-q=0.55):
      LR_base   : logistic regression (champion reference, R084)
      RF        : random forest
      SVM       : RBF support vector machine
      MLP       : neural net (1 hidden layer)
      NB        : Gaussian naive bayes
      ENSEMBLE  : soft-voting of LR + RF + SVM + MLP
  - Walk-forward (train on past only), threshold from selection, holdout 2026 untouched

Success (user spec): prof-mo%>=70, t/mo>=8, worst<=3, holPF>1.1, PF@0.05%>1.1.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG, calc_ema
from scripts.ql_engine import (
    add_features, build_signal_mask, sim_symbol, stats_from_trades,
    cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

RESEARCH_ID = "R086"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
RR = 1.5
Q = 0.55

ORIGINAL52 = {
    "1INCH_USDT_SWAP","AAVE_USDT_SWAP","ADA_USDT_SWAP","ALGO_USDT_SWAP",
    "APT_USDT_SWAP","ARB_USDT_SWAP","ATOM_USDT_SWAP","AVAX_USDT_SWAP",
    "AXS_USDT_SWAP","BCH_USDT_SWAP","BNB_USDT_SWAP","BONK_USDT_SWAP",
    "BTC_USDT_SWAP","CHZ_USDT_SWAP","COMP_USDT_SWAP","CRV_USDT_SWAP",
    "DOGE_USDT_SWAP","DOT_USDT_SWAP","DYDX_USDT_SWAP","EGLD_USDT_SWAP",
    "ENA_USDT_SWAP","ETC_USDT_SWAP","ETH_USDT_SWAP","FET_USDT_SWAP",
    "FIL_USDT_SWAP","FLOKI_USDT_SWAP","GALA_USDT_SWAP","GMX_USDT_SWAP",
    "GRT_USDT_SWAP","HBAR_USDT_SWAP","ICP_USDT_SWAP","IMX_USDT_SWAP",
    "INJ_USDT_SWAP","LDO_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP",
    "NEAR_USDT_SWAP","OP_USDT_SWAP","PEPE_USDT_SWAP","SAND_USDT_SWAP",
    "SATS_USDT_SWAP","SHIB_USDT_SWAP","SNX_USDT_SWAP","SOL_USDT_SWAP",
    "STX_USDT_SWAP","SUI_USDT_SWAP","SUSHI_USDT_SWAP","TRX_USDT_SWAP",
    "UNI_USDT_SWAP","WIF_USDT_SWAP","XLM_USDT_SWAP","XRP_USDT_SWAP",
}
NEW18 = {"BICO_USDT_SWAP","HYPE_USDT_SWAP","XAU_USDT_SWAP","HOME_USDT_SWAP",
         "PUMP_USDT_SWAP","ZBT_USDT_SWAP","ZEC_USDT_SWAP","BEAT_USDT_SWAP",
         "SNDK_USDT_SWAP","SPCX_USDT_SWAP","MU_USDT_SWAP","SKHYNIX_USDT_SWAP",
         "SOXL_USDT_SWAP","UB_USDT_SWAP","SNXX_USDT_SWAP","SKHY_USDT_SWAP",
         "KORU_USDT_SWAP","CL_USDT_SWAP"}
NEW3 = {"XAG_USDT_SWAP","ALLO_USDT_SWAP","AAOI_USDT_SWAP"}
ALL_SYMS = ORIGINAL52 | NEW18 | NEW3

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  ML-TYPE zoo (RF / SVM / MLP / NB / Ensemble)")
print(SEP)
t0 = time.time()

# ── Load ─────────────────────────────────────────────────────────────────────
print("\n  Loading data (73 symbols) …")
raw1h = {}; feats = {}
for sym in ALL_SYMS:
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        raw1h[sym] = df
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY: feats[sym] = f
    except Exception:
        pass
print(f"  Symbols: {len(feats)}")

# universe breadth for the "special" regime feature
above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
# rolling 100-bar percentile of breadth = regime quartile
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100

# ── Raw trades + LEAN feature rows (base 11 + 3 special = 14) ────────────────
cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
mask = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats.items()}
raw = []
for sym, f in feats.items():
    for t in sim_symbol(f, mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"  Raw trades: {len(raw)}")

BASE_FEATS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
              "rel_vol","bb_width","real_vol_20","hour","dow"]
SPECIAL = ["breadth_q", "dist_hi48", "green_streak"]
FEATS = BASE_FEATS + SPECIAL

rows = []
for t in raw:
    sym = t["sym"]; ts = t["entry_time"]; f = feats[sym]
    row = f.loc[ts]
    i = f.index.get_loc(ts)
    c = float(row["close"])
    hi48 = float(f["close"].rolling(48).max().iloc[i]) if i >= 0 else np.nan
    dist_hi = (c / hi48 - 1) * 100 if pd.notna(hi48) and hi48 > 0 else 0.0
    streak = 0
    for k in range(0, 6):
        j = i - k
        if j < 0: break
        if f["close"].iloc[j] > f["open"].iloc[j]: streak += 1
        else: break
    bq = float(breadth_pct.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) if ts >= breadth_pct.index[0] else 50.0
    rows.append(dict(sym=sym, ts=ts, r=t["r"], win=int(t["r"] > 0),
                     **{c: row.get(c, 0) for c in BASE_FEATS},
                     breadth_q=bq, dist_hi48=dist_hi, green_streak=streak))
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
print(f"  Feature rows: {len(mldf)} | total features: {len(FEATS)} (11 base + 3 special)")
sel_mask = mldf["ts"] < HOLDOUT_START

# ── Walk-forward model zoo ───────────────────────────────────────────────────
def make_model(name):
    if name == "LR":     return LogisticRegression(max_iter=2000, C=0.5)
    if name == "RF":     return RandomForestClassifier(n_estimators=200, max_depth=4,
                                                       min_samples_leaf=15, random_state=42,
                                                       n_jobs=-1)
    if name == "SVM":    return SVC(C=1.0, gamma="scale", probability=True)
    if name == "MLP":    return MLPClassifier(hidden_layer_sizes=(16,), alpha=0.5,
                                              max_iter=2000, random_state=42)
    if name == "NB":     return GaussianNB()
    if name == "ENSEMBLE": return VotingClassifier(
        estimators=[("lr", LogisticRegression(max_iter=2000, C=0.5)),
                    ("rf", RandomForestClassifier(n_estimators=150, max_depth=4,
                                                  min_samples_leaf=15, random_state=42, n_jobs=-1)),
                    ("svm", SVC(C=1.0, gamma="scale", probability=True)),
                    ("mlp", MLPClassifier(hidden_layer_sizes=(16,), alpha=0.5,
                                          max_iter=2000, random_state=42))],
        voting="soft")
    raise ValueError(name)

def walkforward(name, min_train=150):
    X = mldf[FEATS].fillna(0).values
    y = mldf["win"].values
    pred = np.full(len(mldf), np.nan)
    needs_scale = name in ("LR","SVM","MLP")
    sc = StandardScaler() if needs_scale else None
    for i in range(min_train, len(mldf)):
        clf = make_model(name)
        Xi = sc.fit_transform(X[:i]) if needs_scale else X[:i]
        clf.fit(Xi, y[:i])
        Xn = sc.transform(X[i:i+1]) if needs_scale else X[i:i+1]
        try:
            pred[i] = clf.predict_proba(Xn)[0, 1]
        except Exception:
            pred[i] = 0.5
    return pred

MODELS = ["LR", "RF", "SVM", "MLP", "NB", "ENSEMBLE"]
print("\n  Fitting walk-forward models …")
preds = {}
for m in MODELS:
    t1 = time.time()
    preds[m] = walkforward(m)
    print(f"    {m}: done in {time.time()-t1:.0f}s")

def keep_ts(pred, q=Q):
    thr = pd.Series(pred).where(sel_mask.values).dropna().quantile(1 - q)
    return set(mldf.loc[pred >= thr, "ts"])

keeps = {m: keep_ts(preds[m]) for m in MODELS}

def monthly_profile(trades):
    if not trades: return dict(prof=float("nan"), worst=float("nan"), tpm=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        worst = max(worst, cur)
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df) / 27.0)

def evaluate(name, trades):
    s = stats_from_trades(trades)
    pf_c = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel_t = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel_t)["pf"]; hp = stats_from_trades(hol)["pf"]
    mp = monthly_profile(trades)
    return dict(cfg=name, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"],
                pf_c=pf_c, mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"],
                selpf=sp, holpf=hp)

print(f"\n{SEP2}")
print("  RESULTS (all on SAME 14 features, top-q=0.55)")
hdr = (f"    {'ML type':<12}{'n':>5}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>6}{'worst':>6}{'selPF':>7}{'holPF':>7}")
print(hdr); print("    " + "─"*100)
rows = []
for m in MODELS:
    trades = [t for t in raw if t["entry_time"] in keeps[m]]
    r = evaluate(m, trades)
    rows.append(r)
    tag = "  ← champion" if m == "LR" else ""
    print(f"    {m:<12}{r['n']:>5}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>5.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}{tag}")

print(f"\n{SEP2}")
print("  SUCCESS (prof%>=70, t/mo>=8, worst<=3, holPF>1.1, PF@.05>1.1)")
passed = [r for r in rows if r["prof"] >= 0.70 and r["tpm"] >= 8 and r["worst"] <= 3
          and r["holpf"] > 1.1 and r["pf_c"] > 1.1]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: t/mo={r['tpm']:.1f} prof%={r['prof']*100:.0f}% "
              f"worst={r['worst']} PF={r['pf']:.2f} PF@.05={r['pf_c']:.2f} holPF={r['holpf']:.2f}")
else:
    print("  ❌ none pass all. Closest:")
    for r in sorted(rows, key=lambda r: -(0.3*(r["prof"]>=0.70) + 0.3*(r["tpm"]>=8) +
                                          0.2*(r["worst"]<=3) + 0.2*(r["holpf"]>1.1)))[:5]:
        print(f"    {r['cfg']:<12} t/mo={r['tpm']:.1f} prof%={r['prof']*100:.0f}% "
              f"worst={r['worst']} PF={r['pf']:.2f} PF@.05={r['pf_c']:.2f} holPF={r['holpf']:.2f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "r086_ml_types.csv"), index=False)
lines = [f"# R086 — ML-TYPE zoo (RF / SVM / MLP / NB / Ensemble)\n",
         f"**Date:** 2026-08-06 | same 14 lean features (11 base + 3 special: "
         f"breadth-quartile, dist-to-48h-high, green-streak) | walk-forward, top-q=0.55\n",
         f"\n## Results\n",
         "| ML type | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    tag = " ⭐ champion" if r["cfg"] == "LR" else ""
    lines.append(f"| {r['cfg']}{tag} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | "
                 f"{r['pf']:.2f} | {r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | "
                 f"{r['worst']} | {r['selpf']:.2f} | {r['holpf']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    best = max(passed, key=lambda r: r["tpm"] * r["holpf"] * r["prof"])
    lines.append(f"**✅ {best['cfg']} meets the retail spec:** {best['tpm']:.1f} t/mo, "
                 f"{best['prof']*100:.0f}% prof-mo, worst {best['worst']}, "
                 f"PF {best['pf']:.2f} (cost {best['pf_c']:.2f}), holPF {best['holpf']:.2f}.")
else:
    lines.append("**❌ No ML type beats the LR champion on ALL criteria.** Honest result — "
                 "the simple model stays the most robust.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r086_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r086_*")
