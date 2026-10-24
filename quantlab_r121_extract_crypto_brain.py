#!/usr/bin/env python3
# R121 step 1 — extract the CRYPTO-TRAINED pop filter (frozen artifact for FX transfer).
# Fits the champion's walk-forward SVM machinery on ALL crypto trades with entry < 2026-01-01
# (selection period) and saves model+scaler+features. No FX data is read here, ever.
import os, sys, pickle, warnings
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
import glob as _glob
ALL_SYMS = sorted(os.path.basename(f).replace("_USDT_SWAP_1H.parquet", "")
                  for f in _glob.glob(os.path.join(CACHE, "*_USDT_SWAP_1H.parquet")))
ALL_SYMS = [s for s in ALL_SYMS if not s.startswith(("XAU", "XAG", "CL_", "SOXL", "MU", "SPCX", "AAOI", "UB_", "SNDK", "SKHYNIX", "SKHY", "BEAT", "ALLO", "ZBT", "SNXX", "PUMP", "HOME"))]  # keep crypto perps only
print(f"crypto universe for transfer-fit: {len(ALL_SYMS)} symbols")

raw1h, feats = {}, {}
for sym in ALL_SYMS:
    p = os.path.join(CACHE, f"{sym}_USDT_SWAP_1H.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open", "high", "low", "close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open", "high", "low", "close", "vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        raw1h[sym] = df
        f = add_features(df)
        f.dropna(subset=["ema200", "atr14", "adx14", "ema_dist_pct", "real_vol_20",
                         "bb_width", "prev_range_r", "prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY: feats[sym] = f
    except Exception as e:
        if not globals().get("_err_shown"):
            print(f"  first load error ({sym}): {type(e).__name__}: {e}")
            globals()["_err_shown"] = True
print(f"usable: {len(feats)}")

cids = ["BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"]
mask = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats.items()}
raw = []
for sym, f in feats.items():
    for t in sim_symbol(f, mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"crypto raw pops: {len(raw)}")

above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100

BASE_FEATS = ["atr_rank", "adx14", "rsi14", "ema_dist_pct", "prev_body_r", "prev_range_r",
              "rel_vol", "bb_width", "real_vol_20", "hour", "dow"]
SPECIAL = ["breadth_q", "dist_hi48", "green_streak"]
FEATS = BASE_FEATS + SPECIAL

rows = []
for t in raw:
    sym = t["sym"]; ts = t["entry_time"]; f = feats[sym]
    row = f.loc[ts]; i = f.index.get_loc(ts)
    c = float(row["close"])
    hi48 = float(f["close"].rolling(48).max().iloc[i])
    dist_hi = (c / hi48 - 1) * 100 if hi48 > 0 else 0.0
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
train = mldf[mldf.ts < pd.Timestamp("2026-01-01", tz="UTC")]
print(f"transfer-fit set: {len(train)} crypto trades (2025-06 → 2025-12 selection period)")

X = train[FEATS].fillna(0).values
y = train["win"].values
sc = StandardScaler()
Xs = sc.fit_transform(X)
clf = SVC(C=1.0, gamma="scale", probability=True)
clf.fit(Xs, y)
art = dict(model=clf, scaler=sc, features=FEATS, trained_on=len(train),
           period=(str(train.ts.min()), str(train.ts.max())),
           win_rate=float(y.mean()))
with open("quantlab_output/r121_crypto_svm.pkl", "wb") as fh:
    pickle.dump(art, fh)
print(f"saved quantlab_output/r121_crypto_svm.pkl | train win-rate {y.mean():.2%}")
