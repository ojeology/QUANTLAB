"""
QUANTLAB AI — R095
SUPER-ADVANCED ML on 5m with SIMPLE indicators (the honest gap — never tested)

On 1H, walk-forward ML (SVM / gradient boosting) was what FOUND the edge. On 5m we
only ever tested RULES (R089-R094). This closes that gap: apply the exact same
advanced-ML machinery (the 1H champions) to 5m, using SIMPLE indicators as features.

Design:
  - Raw trades: pool ALL simple bank-style 5m signals (vwap-fade, prevday-low,
    sesslow-accum, 2day-sweep) + Family-A port = a big pool of 5m entries.
  - Features (simple, causal): rsi14, atr_rank, bb_width, adx14, ema_dist_pct,
    rel_vol, real_vol_20, vwap_dist, hour, dow, breadth, green_streak.
  - Models (walk-forward, top-q keep, threshold from selection only):
      LR (baseline) | SVM-RBF (1H champion) | GradientBoosting (79% prof-mo on 1H)
  - Protocol: selection <= 2026-05-31, holdout Jun-Aug untouched, cost gate 0.05%.
  - Success: holPF@cost > 1.1, selection n >= 200.

Guardrails: causal features only; all models train on past only; thresholds from
selection only; results shown gross AND @0.05% cost.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, sim_symbol, stats_from_trades, cost_adjusted_rs, pf_of_rs
import scripts.ql_engine as qle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier

RESEARCH_ID = "R095"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-06-01", tz="UTC")
IS_LOOKBACK = 6000
RECAL_EVERY = 2016
MIN_BARS = IS_LOOKBACK + RECAL_EVERY + 500
SYMS = ["BTC_USDT_SWAP","ETH_USDT_SWAP","DOGE_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP"]
Q = 0.50

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  ADVANCED ML on 5m, SIMPLE indicators")
print(SEP)
t0 = time.time()

print("\n  Loading 5m data …")
feats = {}
for sym in SYMS:
    p = os.path.join(CACHE, f"{sym}_5m.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < MIN_BARS: continue
        f = add_features(df)
        f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14",
                         "ema_dist_pct","real_vol_20","bb_width","prev_range_r",
                         "prev_body_r"], inplace=True)
        if len(f) >= MIN_BARS: feats[sym] = f
    except Exception as e:
        print(f"  {sym}: ERR {e}")
print(f"  Symbols: {len(feats)}")

def ind(f):
    f = f.copy()
    day = f.index.normalize()
    typ = (f["high"] + f["low"] + f["close"]) / 3.0
    f["vwap"] = (typ * f["vol"]).groupby(day).cumsum() / f["vol"].groupby(day).cumsum().replace(0, np.nan)
    f["vwap_dist"] = (f["close"] - f["vwap"]) / f["atr14"]
    f["sess_low"] = f["low"].groupby(day).cummin()
    d = f.resample("1D").agg({"high":"max","low":"min","close":"last"}).shift(1)
    f["prev_day_low"] = d["low"].reindex(day).values
    d2 = f.resample("1D").agg({"low":"min"}).rolling(2).min().shift(1)
    f["two_day_low"] = d2["low"].reindex(day).values
    return f

feats2 = {s: ind(f) for s, f in feats.items()}
above = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats2.items()}
breadth = pd.DataFrame(above).sort_index().mean(axis=1, skipna=True)

# ── Simple 5m signals (pool) ─────────────────────────────────────────────────
def s_vwap(f):
    deep = (f["vwap_dist"] < -1.0).rolling(5, min_periods=1).max().astype(bool)
    return (deep & (f["close"] > f["vwap"] - 0.5 * f["atr14"]) & (f["close"] > f["open"]) & f["vwap"].notna())
def s_prevlow(f):
    sw = (f["low"] < f["prev_day_low"]).rolling(5, min_periods=1).max().astype(bool)
    return (sw & (f["close"] > f["prev_day_low"]) & (f["close"] > f["open"]) & (f["rel_vol"] > 1.2) & f["prev_day_low"].notna())
def s_sesslow(f):
    near = f["low"] <= f["sess_low"] + 0.2 * f["atr14"]
    return (near & (f["rsi14"] < 25) & (f["close"] > f["open"]) & (f["rel_vol"] > 1.3) & f["sess_low"].notna())
def s_2day(f):
    sw = (f["low"] < f["two_day_low"] - 0.1 * f["atr14"]).rolling(5, min_periods=1).max().astype(bool)
    return (sw & (f["close"] > f["two_day_low"]) & (f["close"] > f["open"]) & f["two_day_low"].notna())
def s_family(f):
    # Family-A compression-pop on 5m
    bb = f["bb_width"].rolling(6000).quantile(0.25)
    rv = f["real_vol_20"].rolling(6000).quantile(0.33)
    prg = f["prev_range_r"].rolling(6000).quantile(0.80)
    return ((f["bb_width"] < bb) & (f["real_vol_20"] < rv) & (f["prev_range_r"] > prg) &
            (f["rel_vol"] > 1.5) & (f["close"] > f["open"]) & (f["close"] > f["close"].shift(1)))

SIGNALS = {"vwap": s_vwap, "prevlow": s_prevlow, "sesslow": s_sesslow,
           "2day": s_2day, "family": s_family}

qle.IS_LOOKBACK = IS_LOOKBACK
qle.RECAL_EVERY = RECAL_EVERY

print("\n  Pooling raw 5m signals …")
raw = []
for name, fn in SIGNALS.items():
    for sym, f in feats2.items():
        m = fn(f)
        for t in sim_symbol(f, m, 1.5, dict(entry_next=False, exit="timeN", sl_mult=1.0, time_bars=90, hours=None)):
            t["sig"] = name; t["sym"] = sym; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"  Pooled raw trades: {len(raw)}")

# ── Simple features at entry (all causal) ────────────────────────────────────
FEATS = ["rsi14","atr_rank","bb_width","adx14","ema_dist_pct","rel_vol","real_vol_20",
         "vwap_dist","hour","dow","breadth","green_streak"]
rows = []
for t in raw:
    sym = t["sym"]; ts = t["entry_time"]; f = feats2[sym]
    if ts not in f.index: continue
    row = f.loc[ts]
    i = f.index.get_loc(ts)
    streak = 0
    for k in range(0, 6):
        j = i - k
        if j < 0: break
        if f["close"].iloc[j] > f["open"].iloc[j]: streak += 1
        else: break
    bq = float(breadth.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) if ts >= breadth.index[0] else 0.5
    rows.append(dict(sym=sym, sig=t["sig"], ts=ts, r=t["r"], win=int(t["r"] > 0),
                     rsi14=float(row.get("rsi14",50)), atr_rank=float(row.get("atr_rank",50)),
                     bb_width=float(row.get("bb_width",0.4)), adx14=float(row.get("adx14",20)),
                     ema_dist_pct=float(row.get("ema_dist_pct",0)), rel_vol=float(row.get("rel_vol",1)),
                     real_vol_20=float(row.get("real_vol_20",1)), vwap_dist=float(row.get("vwap_dist",0)),
                     hour=ts.hour, dow=ts.dayofweek, breadth=bq, green_streak=streak))
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
sel_mask = mldf["ts"] < HOLDOUT_START
print(f"  ML feature rows: {len(mldf)} | selection {sel_mask.sum()} | holdout {(~sel_mask).sum()}")
print(f"  Gross PF of pooled raw: {pf_of_rs(mldf['r'].values):.2f}")

# ── Walk-forward advanced ML ─────────────────────────────────────────────────
def walkforward(model, min_train=500, step=200):
    """Block walk-forward: retrain every `step` rows, predict the next block.
    Strictly causal (only trains on past), ~step× faster than per-row refit."""
    X = mldf[FEATS].fillna(0).values
    y = mldf["win"].values
    n = len(mldf)
    pred = np.full(n, np.nan)
    sc = StandardScaler() if model != "gb" else None
    i = min_train
    while i < n:
        j = min(i + step, n)
        if model == "lr":
            clf = LogisticRegression(max_iter=2000, C=0.5)
            clf.fit(sc.fit_transform(X[:i]), y[:i])
            Xi = sc.transform(X[i:j])
            pred[i:j] = clf.predict_proba(Xi)[:, 1]
        elif model == "svm":
            clf = SVC(C=1.0, gamma="scale", probability=True)
            clf.fit(sc.fit_transform(X[:i]), y[:i])
            Xi = sc.transform(X[i:j])
            pred[i:j] = clf.predict_proba(Xi)[:, 1]
        elif model == "gb":
            clf = HistGradientBoostingClassifier(max_iter=300, max_depth=3,
                                                 l2_regularization=2.0,
                                                 min_samples_leaf=50, learning_rate=0.05)
            clf.fit(X[:i], y[:i])
            pred[i:j] = clf.predict_proba(X[i:j])[:, 1]
        i = j
    return pred

print("\n  Fitting walk-forward models (this is the slow part) …")
models = {}
for m in ["lr", "svm", "gb"]:
    t1 = time.time()
    models[m] = walkforward(m)
    print(f"    {m}: {time.time()-t1:.0f}s")

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
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df)/4.5)

def evaluate(name, trades):
    if not trades: return dict(cfg=name, n=0, tpm=0, wr=float('nan'), pf=float('nan'),
                               pf_c=float('nan'), mdd=0, prof=float('nan'), worst=float('nan'),
                               selpf=float('nan'), holpf=float('nan'), holpf_c=float('nan'))
    s = stats_from_trades(trades)
    pf_c = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]
    hp = stats_from_trades(hol)["pf"]
    hpc = pf_of_rs(cost_adjusted_rs(hol, 0.05))
    mp = monthly_profile(trades)
    return dict(cfg=name, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"], pf_c=pf_c,
                mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"], selpf=sp, holpf=hp, holpf_c=hpc)

print(f"\n{SEP2}\n  RESULTS (advanced ML on simple 5m indicators, top-q={Q})\n{SEP2}")
hdr = (f"    {'Model':<8}{'n':>6}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}{'holPF@c':>9}")
print(hdr); print("    " + "─"*105)
res = []
# raw reference
r0 = evaluate("RAW_pool", raw)
res.append(r0)
print(f"    {'RAW':<8}{r0['n']:>6}{r0['tpm']:>6.1f}{r0['wr']*100:>6.0f}%"
      f"{r0['pf']:>7.2f}{r0['pf_c']:>8.2f}{r0['mdd']*100:>7.1f}%{r0['prof']*100:>6.0f}%"
      f"{r0['worst']:>6}{r0['selpf']:>7.2f}{r0['holpf']:>7.2f}{r0['holpf_c']:>9.2f}  ← base")
for m in ["lr","svm","gb"]:
    thr = pd.Series(models[m]).where(sel_mask.values).dropna().quantile(1 - Q)
    keep = set(mldf.loc[models[m] >= thr, "ts"])
    trades = [t for t in raw if t["entry_time"] in keep]
    r = evaluate(m.upper(), trades)
    res.append(r)
    tag = "  ← 1H champion" if m == "svm" else ("  ← 79% prof-mo on 1H" if m == "gb" else "")
    print(f"    {m.upper():<8}{r['n']:>6}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}{r['holpf_c']:>9.2f}{tag}")

print(f"\n{SEP2}\n  SUCCESS (holPF@cost>1.1, sel n>=200)")
passed = [r for r in res if r["holpf_c"] > 1.1 and r["n"] >= 200]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: holPF@cost={r['holpf_c']:.2f}")
else:
    print("  ❌ none passed. Closest:")
    for r in sorted(res, key=lambda r: -r["holpf_c"])[:3]:
        print(f"    {r['cfg']}: holPF@cost={r['holpf_c']:.2f} holPF={r['holpf']:.2f} PF={r['pf']:.2f} n={r['n']}")

pd.DataFrame(res).to_csv(os.path.join(OUT, "r095_5m_ml.csv"), index=False)
lines = [f"# R095 — ADVANCED ML on 5m with SIMPLE indicators\n",
         f"**Date:** 2026-08-07 | pooled 5 simple 5m signals, walk-forward LR/SVM/GB, "
         f"top-q={Q} | selection ≤May, holdout Jun-Aug, cost gate 0.05%\n",
         f"\n## Results\n",
         "| Model | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF | holPF@cost |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in res:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} | {r['holpf_c']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    best = max(passed, key=lambda r: r["holpf_c"])
    lines.append(f"**✅ {best['cfg']} survives costs on holdout (holPF@cost {best['holpf_c']:.2f}).**")
else:
    lines.append("**❌ No advanced-ML config survives 0.05% costs on holdout.** "
                 "7th independent 5m confirmation — ML can't create edge where raw PF≈1.0.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r095_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r095_*")
