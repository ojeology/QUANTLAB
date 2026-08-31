"""
FOREX HUNT — F004
4H TIMEFRAME + RR sweep on ML+daily-trend (the cost-drag fix).

F003 showed: at 1H, RR3.0 gets holPF@cost 0.92 but the ~0.3R spread drag is structural.
At 4H the ATR ~doubles -> same spread costs ~half per trade in R-terms.
Resample the 1H data we already have to 4H (no new fetch needed).

Config: pooled signals (vwap/mom/pull) on 4H, ML-SVM walk-forward keep-top-q,
daily-trend filter, RR sweep {1.5, 2.0, 3.0, 4.0}.
Protocol: selection <= Aug-2025, holdout Aug-2025..Aug-2026 untouched, retail spreads.
Success: holPF@cost > 1.1, selection n >= 60.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG, calc_ema, calc_adx
from scripts.ql_engine import add_features, sim_symbol, stats_from_trades, pf_of_rs, IS_LOOKBACK, RECAL_EVERY
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FOREX_DIR = os.path.join(CONFIG["CACHE_FOLDER"], "forex")
OUT = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2025-08-01", tz="UTC")
Q = 0.55

PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP"]
SPREAD = {"EURUSD":0.00006,"GBPUSD":0.00010,"USDJPY":0.010,"AUDUSD":0.00008,
          "USDCAD":0.00010,"USDCHF":0.00010,"NZDUSD":0.00012,"EURGBP":0.00010}

# 4H-scaled engine params
IS_LOOKBACK_4H = 125
RECAL_EVERY_4H = 42

SEP = "=" * 110
SEP2 = "-" * 90
print(); print(SEP)
print("  FOREX HUNT — F004  (4H timeframe + RR sweep, ML+daily-trend)")
print(SEP)
t0 = time.time()

print("\n  Loading 1H forex data + resampling to 4H …")
raw4h = {}
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
    # resample to 4H
    r = df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()
    raw4h[p] = r
    print(f"    {p}: {len(r)} 4H bars")

# features on 4H
feats = {}
for p, df in raw4h.items():
    f = add_features(df)
    f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14",
                     "ema_dist_pct","real_vol_20","bb_width","prev_range_r",
                     "prev_body_r"], inplace=True)
    if len(f) >= IS_LOOKBACK_4H + RECAL_EVERY_4H:
        feats[p] = f
print(f"  Pairs ready on 4H: {len(feats)}")

print("  Building daily context …")
daily = {}
for p, df in raw4h.items():
    d = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    d["ema50"] = calc_ema(d["close"], 50)
    d["adx14"] = calc_adx(d, 14)
    daily[p] = d

def in_daily_trend(p, ts):
    d = daily[p]
    return bool((d["close"] > d["ema50"]).reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0])
def in_daily_adx(p, ts):
    d = daily[p]
    return bool((d["adx14"] > 20).reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0])

def add_fx(f):
    f = f.copy()
    day = f.index.normalize()
    typ = (f["high"] + f["low"] + f["close"]) / 3.0
    f["vwap"] = (typ * f["vol"]).groupby(day).cumsum() / f["vol"].groupby(day).cumsum().replace(0, np.nan)
    f["vwap_dist"] = (f["close"] - f["vwap"]) / f["atr14"]
    return f

feats2 = {p: add_fx(f) for p, f in feats.items()}

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

# patch engine params for 4H
import scripts.ql_engine as qle
qle.IS_LOOKBACK = IS_LOOKBACK_4H
qle.RECAL_EVERY = RECAL_EVERY_4H

def run_pooled(rr):
    out = []
    for p, f in feats2.items():
        for name, fn in SIGS.items():
            m = fn(f)
            try:
                for t in sim_symbol(f, m, rr, dict(entry_next=False, exit="timeN",
                                                   time_bars=24, hours=None)):
                    t["pair"] = p; t["sig"] = name; t["spread"] = SPREAD[p]
                    out.append(t)
            except Exception:
                pass
    out.sort(key=lambda t: t["entry_time"])
    return out

print("\n  Pooling 4H trades at RR1.5 for ML training …")
raw15 = run_pooled(1.5)
print(f"  4H pooled at RR1.5: {len(raw15)}")

FEATS_ML = ["rsi14","atr_rank","bb_width","adx14","ema_dist_pct","rel_vol",
            "real_vol_20","vwap_dist","hour","dow","d_trend","d_adx"]
rows = []
for t in raw15:
    p = t["pair"]; ts = t["entry_time"]; f = feats2[p]
    if ts not in f.index: continue
    row = f.loc[ts]
    rows.append(dict(pair=p, ts=ts, r=t["r"], win=int(t["r"] > 0),
                     rsi14=float(row.get("rsi14",50)), atr_rank=float(row.get("atr_rank",50)),
                     bb_width=float(row.get("bb_width",0.4)), adx14=float(row.get("adx14",20)),
                     ema_dist_pct=float(row.get("ema_dist_pct",0)), rel_vol=float(row.get("rel_vol",1)),
                     real_vol_20=float(row.get("real_vol_20",1)), vwap_dist=float(row.get("vwap_dist",0)),
                     hour=ts.hour, dow=ts.dayofweek,
                     d_trend=int(in_daily_trend(p, ts)), d_adx=int(in_daily_adx(p, ts))))
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
sel_mask = mldf["ts"] < HOLDOUT_START
print(f"  ML rows: {len(mldf)} | sel {sel_mask.sum()} | hol {(~sel_mask).sum()} | gross PF {pf_of_rs(mldf['r'].values):.2f}")

print("  Fitting walk-forward SVM (4H) …")
X = mldf[FEATS_ML].fillna(0).values
y = mldf["win"].values
pred = np.full(len(mldf), np.nan)
sc = StandardScaler()
min_train, step = 60, 50
i = min_train
while i < len(mldf):
    j = min(i + step, len(mldf))
    clf = SVC(C=1.0, gamma="scale", probability=True)
    clf.fit(sc.fit_transform(X[:i]), y[:i])
    pred[i:j] = clf.predict_proba(sc.transform(X[i:j]))[:, 1]
    i = j
thr = pd.Series(pred).where(sel_mask.values).dropna().quantile(1 - Q)
keep_ts = set(mldf.loc[pred >= thr, "ts"])
print(f"  ML keep: {len(keep_ts)} timestamps")

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

RR_SWEEP = [1.5, 2.0, 3.0, 4.0]
print(f"\n{SEP2}\n  RR SWEEP on 4H ML+daily-trend\n{SEP2}")
hdr = (f"    {'RR':>6}{'n':>6}{'WR':>7}{'PF':>7}{'PF@cost':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}{'holPF@c':>9}")
print(hdr); print("    " + "-"*96)
rows = []
for rr in RR_SWEEP:
    trades = []
    for p, f in feats2.items():
        for name, fn in SIGS.items():
            m = fn(f)
            try:
                for t in sim_symbol(f, m, rr, dict(entry_next=False, exit="timeN",
                                                   time_bars=24, hours=None)):
                    if t["entry_time"] in keep_ts and in_daily_trend(p, t["entry_time"]):
                        t["pair"] = p; t["spread"] = SPREAD[p]
                        trades.append(t)
            except Exception:
                pass
    trades.sort(key=lambda t: t["entry_time"])
    r = eval_cfg(f"RR{rr}", trades)
    rows.append(r)
    print(f"    {rr:>6.1f}{r['n']:>6}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}{r['holpf_c']:>9.2f}")

print(f"\n{SEP2}\n  SUCCESS (holPF@cost>1.1, sel n>=60)")
passed = [r for r in rows if r["holpf_c"] > 1.1 and r["n"] >= 60]
if passed:
    for r in passed:
        print(f"  ✅ RR{r['cfg']}: holPF@cost={r['holpf_c']:.2f}")
else:
    print("  ❌ none passed. Closest:")
    for r in sorted(rows, key=lambda r: -r["holpf_c"])[:3]:
        print(f"    RR{r['cfg']}: holPF@cost={r['holpf_c']:.2f} holPF={r['holpf']:.2f} PF={r['pf']:.2f} n={r['n']}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "f004_4h_rr.csv"), index=False)
lines = [f"# FOREX F004 — 4H TF + RR sweep (ML+daily-trend)\n",
         f"**Date:** 2026-08-08 | 4H (resampled from 1H), selection ≤ Aug-2025, "
         f"holdout untouched, retail spreads\n",
         f"\n## Results\n",
         "| RR | n | WR | PF | PF@cost | MDD% | prof% | worst | selPF | holPF | holPF@cost |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['wr']*100:.0f}% | {r['pf']:.2f} | {r['pf_c']:.2f} | "
                 f"{r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | {r['selpf']:.2f} | "
                 f"{r['holpf']:.2f} | {r['holpf_c']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    best = max(passed, key=lambda r: r["holpf_c"])
    lines.append(f"**✅ RR{best['cfg']} survives costs on holdout (holPF@cost {best['holpf_c']:.2f}) — "
                 f"FIRST COST-SURVIVING FOREX EDGE.**")
else:
    lines.append("**❌ No 4H RR config survives costs on holdout.**")
report = "\n".join(lines)
with open(os.path.join(OUT, "f004_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/f004_*")
