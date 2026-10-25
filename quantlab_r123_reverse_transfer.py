#!/usr/bin/env python3
# R123 — REVERSE TRANSFER: train the pop filter on FX pops (2016-2024), test on crypto pops.
# Prespec: docs/RESEARCH_FRONTIER_R123.md (LOCKED). One cell, one run.
import os, sys, glob, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, build_signal_mask, sim_symbol, IS_LOOKBACK, RECAL_EVERY
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CACHE = CONFIG["CACHE_FOLDER"]
RR = 1.5
Q = 0.55
TRAIN_END = pd.Timestamp("2024-12-31", tz="UTC")
EXCLUDE_NONCRYPTO = ("XAU", "XAG", "CL_", "SOXL", "MU", "SPCX", "AAOI", "UB_", "SNDK",
                     "SKHYNIX", "SKHY", "BEAT", "ALLO", "ZBT", "SNXX", "PUMP", "HOME")

# ── 1. FX training set ───────────────────────────────────────────────────────
fx = pd.read_csv("quantlab_output/r121_fx_trades.csv", parse_dates=["entry_time"])
FEATS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
         "rel_vol","bb_width","real_vol_20","hour","dow","breadth_q","dist_hi48","green_streak"]
fx_train = fx[fx.entry_time <= TRAIN_END]
print(f"FX train: {len(fx_train)} trades (2016-01 -> 2024-12), WR {(fx_train.r>0).mean():.1%}")
Xf = fx_train[FEATS].fillna(0).values
yf = (fx_train.r > 0).astype(int).values
sc_fx = StandardScaler(); clf_fx = SVC(C=1.0, gamma="scale", probability=True)
clf_fx.fit(sc_fx.fit_transform(Xf), yf)
pickle.dump(dict(model=clf_fx, scaler=sc_fx, features=FEATS,
                 trained_on=len(fx_train), period=(str(fx_train.entry_time.min()), str(fx_train.entry_time.max()))),
            open("quantlab_output/r123_fx_svm.pkl","wb"))

# ── 2. crypto raw pops rebuild (identical machinery to champion/R121 extract) ─
syms = sorted(os.path.basename(f).replace("_USDT_SWAP_1H.parquet","")
              for f in glob.glob(os.path.join(CACHE, "*_USDT_SWAP_1H.parquet")))
syms = [s for s in syms if not s.startswith(EXCLUDE_NONCRYPTO)]
feats = {}
for sym in syms:
    try:
        df = pd.read_parquet(os.path.join(CACHE, f"{sym}_USDT_SWAP_1H.parquet"))
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY: feats[sym] = f
    except Exception as e:
        print(f"  load-skip {sym}: {type(e).__name__}")
print(f"crypto universe usable: {len(feats)}")

cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
mask = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats.items()}
raw = []
for sym, f in feats.items():
    for t in sim_symbol(f, mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"crypto raw pops: {len(raw)}")

above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth_pct = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True).rolling(100, min_periods=50).rank(pct=True) * 100

rows = []
for t in raw:
    sym = t["sym"]; ts = t["entry_time"]; f = feats[sym]
    row = f.loc[ts]; i = f.index.get_loc(ts)
    c = float(row["close"])
    hi48 = float(f["close"].rolling(48).max().iloc[i])
    dist_hi = (c/hi48 - 1) * 100 if hi48 > 0 else 0.0
    streak = 0
    for k in range(0, 6):
        j = i - k
        if j < 0: break
        if f["close"].iloc[j] > f["open"].iloc[j]: streak += 1
        else: break
    bq = float(breadth_pct.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) if ts >= breadth_pct.index[0] else 50.0
    rows.append(dict(sym=sym, entry_time=ts, r=t["r"], atr=t["atr"], entry=t["entry"],
                     **{cf: row.get(cf, 0) for cf in FEATS}))
T = pd.DataFrame(rows)
T["rc"] = T.r - 2.0*(0.05/100.0)*T.entry/T.atr
Xc = T[FEATS].fillna(0).values
T["score"] = clf_fx.predict_proba(sc_fx.transform(Xc))[:, 1]

thr = T.score.quantile(1 - Q)
kept = T[T.score >= thr]
print(f"keep q={Q}: threshold {thr:.3f} -> kept {len(kept)}")

def pf(x):
    x = np.asarray(x, float); w = x[x>0].sum(); lo = abs(x[x<0].sum())
    return w/lo if lo>0 else (999.0 if w>0 else 1.0)

pf_raw = pf(T.rc.values); pf_kept = pf(kept.rc.values)
rng = np.random.default_rng(123)
rand = [pf(rng.choice(T.rc.values, len(kept), replace=False)) for _ in range(200)]
rand_q = float(np.quantile(rand, Q))
boot = [pf(rng.choice(kept.rc.values, len(kept))) for _ in range(2000)]
boot_p = float(np.mean(np.array(boot) > 1.0))
loao = {s: pf(kept[kept.sym != s].rc.values) for s in kept.sym.unique()}

print("\n" + "="*96)
print(f"  R123 — FX-BRAIN -> CRYPTO (kept 55%, RR 1.5, 0.05%/side; confirmatory-grade window)")
print("="*96)
print(f"  raw pops      : n={len(T):>3}  PF@c={pf_raw:.2f}  WR={(T.rc>0).mean()*100:.0f}%  sum={T.rc.sum():+.1f}R")
print(f"  kept (FX brain): n={len(kept):>3}  PF@c={pf_kept:.2f}  WR={(kept.rc>0).mean()*100:.0f}%  sum={kept.rc.sum():+.1f}R")
print(f"  random-55% q-quantile: {rand_q:.2f}   | reference crypto-brain book: PF@c 1.64")
print(f"  bootP(kept>1) = {boot_p:.3f}  | LOAO min = {min(loao.values()):.2f} "
      f"({min(loao, key=loao.get)})")
fx25 = fx[fx.entry_time >= pd.Timestamp("2025-01-01", tz="UTC")]
s25 = clf_fx.predict_proba(sc_fx.transform(fx25[FEATS].fillna(0).values))[:,1]
k25 = fx25[s25 >= np.quantile(s25, 1-Q)]
print(f"  FX self-test 2025+ (report-only): raw PF {pf(fx25.r.values):.2f} (n={len(fx25)}) | kept PF {pf(k25.r.values):.2f} (n={len(k25)})")

g1, g2, g3, g4 = pf_kept >= 1.10, len(kept) >= 100, boot_p >= 0.95, min(loao.values()) > 1.0
b1, b2 = pf_kept > pf_raw, pf_kept > rand_q
print(f"\n  GATES: G1 {'PASS' if g1 else 'FAIL'} | G2 {'PASS' if g2 else 'FAIL'} | G3 {'PASS' if g3 else 'FAIL'} | G4 {'PASS' if g4 else 'FAIL'}")
print(f"  MUST-BEAT: raw {'PASS' if b1 else 'FAIL'} | random {'PASS' if b2 else 'FAIL'}")
print(f"\n  VERDICT: " + ("PROMISING AS CHALLENGER FILTER (forward validation only)" if all([g1,g2,g3,g4,b1,b2]) else "REJECT"))
T.to_csv("quantlab_output/r123_crypto_scored.csv", index=False)
print("saved: r123_crypto_scored.csv")
