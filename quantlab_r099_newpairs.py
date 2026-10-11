"""
QUANTLAB AI — R099
UNSEEN-UNIVERSE TRANSFER TEST: does the champion's edge hold on brand-new pairs?

Setup: 27 liquid OKX USDT swaps NEVER used in any selection decision (R099 discovery:
top-30 by 24h vol outside the 73-symbol universe; 27 with sufficient 1H history).
Pipeline: IDENTICAL to R096 champion — Family A signals (rolling 500-bar thresholds,
causal), E6 entry, RR1.5/SL1, breadth_q computed from the ORIGINAL 73 (production-
faithful), walk-forward SVM fit on ORIGINAL-universe trades only (every prediction is
out-of-sample in time), keep-threshold q0.55 calibrated on original selection period.
Nothing about the new pairs influenced any fit or threshold.

Question (R078 found PF 0.63 on 8 new syms, n=17 — inconclusive): does the edge
generalize, or is it universe-specific?
Gates: n ≥ 100 · PF@cost > 1.10 · LOO(symbol) floor > 1.00.
"""
import os, sys, time, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, build_signal_mask, sim_symbol, stats_from_trades,
    cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CACHE = CONFIG["CACHE_FOLDER"]
NEWDIR = os.path.join(CACHE, "newpairs")
OUT = CONFIG["OUTPUT_FOLDER"]; os.makedirs(OUT, exist_ok=True)
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
FRESH_START   = pd.Timestamp("2026-08-08", tz="UTC")
RR = 1.5
Q  = 0.55
RNG = np.random.default_rng(42)
SEP  = "═" * 110
SEP2 = "─" * 90

BASE_FEATS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
              "rel_vol","bb_width","real_vol_20","hour","dow"]
SPECIAL = ["breadth_q", "dist_hi48", "green_streak"]
FEATS = BASE_FEATS + SPECIAL

def load_frame(path):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for col in ["open","high","low","close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open","high","low","close","vol"], inplace=True)
    return df

def build(sym, df):
    if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: return None
    f = add_features(df)
    f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                     "bb_width","prev_range_r","prev_body_r"], inplace=True)
    if len(f) < IS_LOOKBACK + RECAL_EVERY: return None
    return f

def ml_row(sym, ts, f, breadth_pct):
    row = f.loc[ts]; i = f.index.get_loc(ts)
    c = float(row["close"])
    hi48 = float(f["close"].rolling(48).max().iloc[i]) if i >= 0 else np.nan
    dist_hi = (c/hi48 - 1)*100 if pd.notna(hi48) and hi48 > 0 else 0.0
    streak = 0
    for k in range(0, 6):
        j = i - k
        if j < 0: break
        if f["close"].iloc[j] > f["open"].iloc[j]: streak += 1
        else: break
    bq = float(breadth_pct.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) if ts >= breadth_pct.index[0] else 50.0
    return dict(sym=sym, ts=ts, **{c_: row.get(c_, 0) for c_ in BASE_FEATS},
                breadth_q=bq, dist_hi48=dist_hi, green_streak=streak)

print(); print(SEP)
print("  QUANTLAB AI — R099  unseen-universe transfer test (27 fresh pairs)")
print(SEP)
t0 = time.time()

# ── ORIGINAL 73 pipeline (champion reference) ────────────────────────────────
print("\n  [1] Original 73: features + signals + trades …")
feats73 = {}
for p in sorted(glob.glob(os.path.join(CACHE, "*_1H.parquet"))):
    sym = os.path.basename(p).replace("_1H.parquet","")
    if sym.endswith("_SWAP") is False: continue
    f = build(sym, load_frame(p))
    if f is not None: feats73[sym] = f
print(f"  symbols: {len(feats73)}")
above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats73.items()}
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100
cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
mask73 = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats73.items()}
raw73 = []
for sym, f in feats73.items():
    for t in sim_symbol(f, mask73[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw73.append(t)
raw73.sort(key=lambda t: t["entry_time"])
print(f"  original raw trades: {len(raw73)}")

rows = [ml_row(t["sym"], t["entry_time"], feats73[t["sym"]], breadth_pct) | {"r": t["r"], "win": int(t["r"]>0)} for t in raw73]
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
sel_mask = mldf["ts"] < HOLDOUT_START
X = mldf[FEATS].fillna(0).values; y = mldf["win"].values
pred = np.full(len(mldf), np.nan); sc = StandardScaler()
for i in range(150, len(mldf)):
    clf = SVC(C=1.0, gamma="scale", probability=True)
    clf.fit(sc.fit_transform(X[:i]), y[:i])
    pred[i] = clf.predict_proba(sc.transform(X[i:i+1]))[0, 1]
THR = float(pd.Series(pred).where(sel_mask.values).dropna().quantile(1 - Q))
keep_mask = pred >= THR
mldf["kept"] = np.where(np.isnan(pred), False, keep_mask)
trades73 = [t for t, k in zip(raw73, mldf.sort_values("ts")["kept"].values) if k]
print(f"  original champion trades (q{Q}): {len(trades73)}  THR={THR:.3f}")

# ── NEW PAIRS pipeline (identical, zero refit on them) ───────────────────────
print(f"\n  [2] New pairs: signals + trades …")
new_feats = {}
for p in sorted(glob.glob(os.path.join(NEWDIR, "*_1H.parquet"))):
    sym = os.path.basename(p).replace("_1H.parquet","")
    f = build(sym, load_frame(p))
    if f is not None: new_feats[sym] = f
print(f"  new symbols with enough history: {len(new_feats)}")
mask_new = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in new_feats.items()}
raw_new = []
for sym, f in new_feats.items():
    for t in sim_symbol(f, mask_new[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw_new.append(t)
raw_new.sort(key=lambda t: t["entry_time"])
print(f"  new-pair raw trades: {len(raw_new)}")
if not raw_new: sys.exit("no new-pair trades")

print("  scoring new-pair signals with the champion SVM (walk-forward in time) …")
new_rows = [ml_row(t["sym"], t["entry_time"], new_feats[t["sym"]], breadth_pct) | {"r": t["r"], "win": int(t["r"]>0), "entry": t["entry"], "atr": t["atr"]} for t in raw_new]
ndf = pd.DataFrame(new_rows).sort_values("ts").reset_index(drop=True)
Xn = ndf[FEATS].fillna(0).values
predn = np.full(len(ndf), np.nan)
scn = StandardScaler()
for i in range(len(ndf)):
    ts_i = ndf["ts"].iloc[i]
    hist = mldf[mldf["ts"] < ts_i]
    if len(hist) < 150: continue
    clf = SVC(C=1.0, gamma="scale", probability=True)
    clf.fit(scn.fit_transform(hist[FEATS].fillna(0).values), hist["win"].values)
    predn[i] = clf.predict_proba(scn.transform(Xn[i:i+1]))[0, 1]
ndf["pred"] = predn
ndf["kept"] = ndf["pred"] >= THR

# ── EVALUATION ───────────────────────────────────────────────────────────────
def pf_on(sub, cost=True):
    if len(sub) == 0: return np.nan, 0
    rs = sub["r"].values
    if cost:
        rs = rs - 2*0.0005*sub["entry"].values/sub["atr"].values
    return pf_of_rs(rs), len(sub)

print(f"\n{SEP2}\n  3) RESULTS — new pairs (never seen) vs original (reference)\n{SEP2}")
print(f"    {'window':<22}{'subset':<18}{'n':>5}{'PF':>7}{'PF@cost':>9}{'WR':>7}")
for tag, sub in [("all (2025-07→)", ndf), ("< 2026", ndf[ndf["ts"]<HOLDOUT_START]),
                 ("2026 (≥Jan)", ndf[ndf["ts"]>=HOLDOUT_START]),
                 ("fresh (≥Aug8)", ndf[ndf["ts"]>=FRESH_START])]:
    pf_raw, n = pf_on(sub, cost=False); pf_c, _ = pf_on(sub, cost=True)
    wr = (sub["r"]>0).mean()*100 if len(sub) else np.nan
    print(f"    {tag:<22}{'raw (no filter)':<18}{n:>5}{pf_raw:>7.2f}{pf_c:>9.2f}{wr:>6.0f}%")
for tag, sub in [("all (2025-07→)", ndf[ndf["kept"]]), ("< 2026", ndf[ndf["kept"] & (ndf["ts"]<HOLDOUT_START)]),
                 ("2026 (≥Jan)", ndf[ndf["kept"] & (ndf["ts"]>=HOLDOUT_START)]),
                 ("fresh (≥Aug8)", ndf[ndf["kept"] & (ndf["ts"]>=FRESH_START)])]:
    pf_raw, n = pf_on(sub, cost=False); pf_c, _ = pf_on(sub, cost=True)
    wr = (sub["r"]>0).mean()*100 if len(sub) else np.nan
    print(f"    {tag:<22}{f'champion filter':<18}{n:>5}{pf_raw:>7.2f}{pf_c:>9.2f}{wr:>6.0f}%")

# reference: original universe same windows (from this run's own pipeline)
o = mldf.copy(); o["entry"]= [t["entry"] for t in sorted(raw73, key=lambda t: t["entry_time"])]
o["atr"]   = [t["atr"] for t in sorted(raw73, key=lambda t: t["entry_time"])]
print()
for tag, sub in [("all", o), ("2026 (≥Jan)", o[o["ts"]>=HOLDOUT_START])]:
    sub_k = sub[sub["kept"]]
    pf_raw, n = pf_on(sub_k, cost=False); pf_c, _ = pf_on(sub_k, cost=True)
    print(f"    {tag:<22}{'ORIGINAL q0.55':<18}{n:>5}{pf_raw:>7.2f}{pf_c:>9.2f}{(sub_k['r']>0).mean()*100:>6.0f}%")

# LOO by symbol on kept new trades
sub = ndf[ndf["kept"]]
if len(sub) >= 20:
    print(f"\n    LOO by new symbol (kept trades, PF@cost):")
    floor = 9.9; worst=None
    for s in sub["sym"].unique():
        v = sub[sub["sym"]!=s]
        pfv,_ = pf_on(v)
        if pfv < floor: floor, worst = pfv, s
    print(f"    floor={floor:.2f} (drop {worst})")
    print(f"    per-symbol kept counts: {sub['sym'].value_counts().head(8).to_dict()}")

# bootstrap on kept new-pair 2026 trades
k26 = sub[sub["ts"]>=HOLDOUT_START]
if len(k26) >= 30:
    rs = k26["r"].values - 2*0.0005*k26["entry"].values/k26["atr"].values
    outb=np.empty(3000)
    for b in range(3000):
        x=rs[RNG.integers(0,len(rs),len(rs))]; outb[b]=pf_of_rs(x)
    print(f"\n    boot CI new-pairs 2026 PF@cost: P5={np.percentile(outb,5):.2f} med={np.median(outb):.2f} P95={np.percentile(outb,95):.2f}  P(PF>1)={(outb>1).mean()*100:.0f}%")

ndf.to_csv(os.path.join(OUT,"r099_newpair_trades.csv"), index=False)
print(f"\n  Done in {time.time()-t0:.0f}s → quantlab_output/r099_newpair_trades.csv")
