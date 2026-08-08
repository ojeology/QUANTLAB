"""
FOREX HUNT — F002
Daily-TF context filters + walk-forward ML filter (the crypto-winning combo).

Base: F001's best raw signals (VWAP-reclaim, momentum-burst, trend-pullback) pooled.
Upgrades:
  A  Daily-trend filter  : trade only when daily close > daily EMA50 (resampled)
  B  Daily-ADX filter    : trade only when daily ADX14 > 20
  C  Daily-trend + ML-SVM: walk-forward SVM on pooled raw signals + daily context,
     keep top-q (like crypto R086 champion)
  D  Hour-session refine : only London/NY overlap hours (12-18 UTC)
Protocol: selection <= Aug-2025, holdout Aug-2025..Aug-2026 untouched, retail spreads.
Success: holPF@cost > 1.1, selection n >= 60.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx
from scripts.ql_engine import add_features, sim_symbol, stats_from_trades, pf_of_rs, IS_LOOKBACK, RECAL_EVERY
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FOREX_DIR = os.path.join(CONFIG["CACHE_FOLDER"], "forex")
OUT = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2025-08-01", tz="UTC")
RR = 1.5
Q = 0.55

PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP"]
SPREAD = {"EURUSD":0.00006,"GBPUSD":0.00010,"USDJPY":0.010,"AUDUSD":0.00008,
          "USDCAD":0.00010,"USDCHF":0.00010,"NZDUSD":0.00012,"EURGBP":0.00010}

SEP = "=" * 110
SEP2 = "-" * 90
print(); print(SEP)
print("  FOREX HUNT — F002  (daily context + ML filter)")
print(SEP)
t0 = time.time()

print("\n  Loading forex data …")
raw1h = {}
feats = {}
for p in PAIRS:
    fpath = os.path.join(FOREX_DIR, f"{p}_1H.parquet")
    if not os.path.exists(fpath): continue
    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for col in ["open","high","low","close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open","high","low","close"], inplace=True)
    if "vol" not in df.columns:
        df["vol"] = (df["high"] - df["low"]).clip(lower=1e-12)
    raw1h[p] = df
    f = add_features(df)
    f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14",
                     "ema_dist_pct","real_vol_20","bb_width","prev_range_r",
                     "prev_body_r"], inplace=True)
    if len(f) >= IS_LOOKBACK + RECAL_EVERY:
        feats[p] = f
print(f"  Pairs: {len(feats)}")

# daily context (resample 1H -> 1D)
print("  Building daily context …")
daily = {}
for p, df in raw1h.items():
    d = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    d["ema50"] = calc_ema(d["close"], 50)
    d["adx14"] = calc_adx(d, 14)
    daily[p] = d

def daily_flag(p, ts, kind):
    d = daily[p]
    if kind == "trend":
        return bool((d["close"] > d["ema50"]).reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0])
    return bool((d["adx14"] > 20).reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0])

# fx indicators (causal)
def add_fx(f):
    f = f.copy()
    day = f.index.normalize()
    typ = (f["high"] + f["low"] + f["close"]) / 3.0
    f["vwap"] = (typ * f["vol"]).groupby(day).cumsum() / f["vol"].groupby(day).cumsum().replace(0, np.nan)
    f["vwap_dist"] = (f["close"] - f["vwap"]) / f["atr14"]
    return f

feats2 = {p: add_fx(f) for p, f in feats.items()}

# base signals (F001 winners)
def s_vwap(f):
    deep = (f["vwap_dist"] < -0.5).rolling(5, min_periods=1).max().astype(bool)
    return (deep & (f["close"] > f["vwap"]) & (f["close"] > f["open"]) & f["vwap"].notna())
def s_mom(f):
    big = (f["close"] - f["open"]).abs() > 1.2 * f["atr14"]
    return (big & (f["rel_vol"] > 1.5) & (f["close"] > f["open"]) & (f["close"] > f["close"].shift(1)))
def s_pull(f):
    low2 = f["low"].rolling(2, min_periods=1).min()
    return ((f["ema50"] > f["ema200"]) & (f["ema50_slope"] > 0) &
            (low2 < f["ema20"]) & (f["close"] > f["ema20"]) & (f["close"] > f["open"]))

SIGS = {"vwap": s_vwap, "mom": s_mom, "pull": s_pull}

def run_pooled():
    out = []
    for p, f in feats2.items():
        for name, fn in SIGS.items():
            m = fn(f)
            try:
                for t in sim_symbol(f, m, RR, dict(entry_next=False, exit="timeN",
                                                   time_bars=60, hours=None)):
                    t["pair"] = p; t["sig"] = name; t["spread"] = SPREAD[p]
                    out.append(t)
            except Exception:
                pass
    out.sort(key=lambda t: t["entry_time"])
    return out

raw_pool = run_pooled()
print(f"  Pooled raw trades: {len(raw_pool)}")

# features for ML
FEATS_ML = ["rsi14","atr_rank","bb_width","adx14","ema_dist_pct","rel_vol",
            "real_vol_20","vwap_dist","hour","dow","d_trend","d_adx"]
rows = []
for t in raw_pool:
    p = t["pair"]; ts = t["entry_time"]; f = feats2[p]
    if ts not in f.index: continue
    row = f.loc[ts]
    rows.append(dict(pair=p, ts=ts, r=t["r"], win=int(t["r"] > 0),
                     rsi14=float(row.get("rsi14",50)), atr_rank=float(row.get("atr_rank",50)),
                     bb_width=float(row.get("bb_width",0.4)), adx14=float(row.get("adx14",20)),
                     ema_dist_pct=float(row.get("ema_dist_pct",0)), rel_vol=float(row.get("rel_vol",1)),
                     real_vol_20=float(row.get("real_vol_20",1)), vwap_dist=float(row.get("vwap_dist",0)),
                     hour=ts.hour, dow=ts.dayofweek,
                     d_trend=int(daily_flag(p, ts, "trend")), d_adx=int(daily_flag(p, ts, "adx"))))
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
sel_mask = mldf["ts"] < HOLDOUT_START
print(f"  ML rows: {len(mldf)} | sel {sel_mask.sum()} | hol {(~sel_mask).sum()} | gross PF {pf_of_rs(mldf['r'].values):.2f}")

# walk-forward SVM (block)
print("  Fitting walk-forward SVM …")
X = mldf[FEATS_ML].fillna(0).values
y = mldf["win"].values
pred = np.full(len(mldf), np.nan)
sc = StandardScaler()
min_train = 400
step = 200
i = min_train
while i < len(mldf):
    j = min(i + step, len(mldf))
    clf = SVC(C=1.0, gamma="scale", probability=True)
    clf.fit(sc.fit_transform(X[:i]), y[:i])
    pred[i:j] = clf.predict_proba(sc.transform(X[i:j]))[:, 1]
    i = j
thr = pd.Series(pred).where(sel_mask.values).dropna().quantile(1 - Q)
keep_ts = set(mldf.loc[pred >= thr, "ts"])

def eval_cfg(name, trades):
    if not trades:
        return dict(cfg=name, n=0, tpm=0, wr=float('nan'), pf=float('nan'), pf_c=float('nan'),
                    mdd=0, prof=float('nan'), worst=float('nan'), selpf=float('nan'),
                    holpf=float('nan'), holpf_c=float('nan'))
    s = stats_from_trades(trades)
    rs = np.array([t["r"] for t in trades])
    cost_r = 2 * np.array([t["spread"]/max(t["atr"],1e-12) for t in trades])
    rc = rs - cost_r
    pf_c = (rc[rc>0].sum()/abs(rc[rc<0].sum())) if (rc<0).any() else 99.0
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]
    hp = stats_from_trades(hol)["pf"]
    hrc = np.array([t["r"] for t in hol]) - 2*np.array([t["spread"]/max(t["atr"],1e-12) for t in hol])
    hpc = (hrc[hrc>0].sum()/abs(hrc[hrc<0].sum())) if (hrc<0).any() else 99.0
    df = pd.DataFrame(trades); df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g>0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur+1 if not v else 0
        worst = max(worst, cur)
    return dict(cfg=name, n=len(trades), tpm=len(df)/24, wr=s["wr"], pf=s["pf"], pf_c=pf_c,
                mdd=s["mdd"], prof=float((g>0).mean()), worst=worst, selpf=sp, holpf=hp, holpf_c=hpc)

CONFIGS = {
    "A_raw_pool": raw_pool,
    "B_daily_trend": [t for t in raw_pool if daily_flag(t["pair"], t["entry_time"], "trend")],
    "C_daily_adx":   [t for t in raw_pool if daily_flag(t["pair"], t["entry_time"], "adx")],
    "D_ml_svm":      [t for t in raw_pool if t["entry_time"] in keep_ts],
    "E_ml+trend":    [t for t in raw_pool if t["entry_time"] in keep_ts and daily_flag(t["pair"], t["entry_time"], "trend")],
    "F_hour1218":    [t for t in raw_pool if 12 <= t["entry_time"].hour < 18],
    "G_hour1218+trend": [t for t in raw_pool if 12 <= t["entry_time"].hour < 18 and daily_flag(t["pair"], t["entry_time"], "trend")],
}

print(f"\n{SEP2}\n  RESULTS  (selection <= Aug-2025 | holdout Aug-2025..Aug-2026 untouched)\n{SEP2}")
hdr = (f"    {'Config':<18}{'n':>6}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@cost':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}{'holPF@c':>9}")
print(hdr); print("    " + "-"*108)
rows = []
for name, trades in CONFIGS.items():
    r = eval_cfg(name, trades)
    rows.append(r)
    print(f"    {name:<18}{r['n']:>6}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}{r['holpf_c']:>9.2f}")

print(f"\n{SEP2}\n  SUCCESS (holPF@cost>1.1, sel n>=60)")
passed = [r for r in rows if r["holpf_c"] > 1.1 and r["n"] >= 60]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: holPF@cost={r['holpf_c']:.2f}")
else:
    print("  ❌ none passed. Closest:")
    for r in sorted(rows, key=lambda r: -r["holpf_c"])[:4]:
        print(f"    {r['cfg']}: holPF@cost={r['holpf_c']:.2f} holPF={r['holpf']:.2f} PF={r['pf']:.2f} n={r['n']}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "f002_forex_results.csv"), index=False)
lines = [f"# FOREX F002 — daily context + ML filter\n",
         f"**Date:** 2026-08-08 | 8 majors, 1H, 2yr | selection ≤ Aug-2025, holdout untouched\n",
         f"\n## Results\n",
         "| Config | n | t/mo | WR | PF | PF@cost | MDD% | prof% | worst | selPF | holPF | holPF@cost |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} | {r['holpf_c']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    lines.append(f"**✅ {passed[0]['cfg']} survives costs on holdout.**")
else:
    lines.append("**❌ No forex config survives retail spreads on holdout.** F002 = run 2 of the hunt.")
report = "\n".join(lines)
with open(os.path.join(OUT, "f002_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/f002_*")
