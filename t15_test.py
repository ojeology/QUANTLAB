"""
TEST 28 — 15m PROOF-OF-CONCEPT. The 12 majors have 7 months of 15m (2026-02..09).
Test the condition-aware TREND (T25 logic) on 15m: Donchian breakout + RF filter.
Mini walk-forward: train first half (<=2026-05-15), test second half. Goal: confirm
an edge exists on 15m (high trade count + PF>1) before grinding a full 3-year fetch.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from ql_engine import add_features, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import build_mldf, FEATS
import demo_bot as bot
from sklearn.ensemble import RandomForestClassifier

SAVE = "quantlab_cache_15m"
FEE = 0.0005
SPLIT = pd.Timestamp("2026-05-15", tz="UTC")


def backtest_donchian(df, N=20, Nx=20, atr_mult=2.0, adx_min=20.0):
    df = df.copy()
    hh = df["high"].rolling(N).max().shift(1); ll = df["low"].rolling(Nx).min().shift(1)
    trades = []; in_pos=False; ep=None; stop=None
    for i in range(N, len(df)):
        bar = df.iloc[i]
        if not in_pos:
            if bar["close"]>hh.iloc[i] and bar["adx14"]>adx_min and bar["close"]>bar["ema200"]:
                ep=bar["close"]; stop=ep-atr_mult*bar["atr14"]; in_pos=True
        else:
            ex=None; et=None
            if bar["low"]<=stop: ex=stop; et="SL"
            elif bar["close"]<ll.iloc[i]: ex=bar["close"]; et="BRK"
            if ex is not None:
                trades.append(dict(entry_time=df.index[i], r=(ex/ep-1.0))); in_pos=False
    return trades


print("[load] 12 majors 15m …", flush=True)
feats, above20 = {}, {}
for f in sorted(os.listdir(SAVE)):
    if not f.endswith("_15m.parquet"): continue
    sym = f[:-len("_15m.parquet")]
    try:
        df = pd.read_parquet(os.path.join(SAVE, f)); df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for c in ["open","high","low","close","vol"]:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        ftr=add_features(df); ftr.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(ftr)>=IS_LOOKBACK+RECAL_EVERY: feats[sym]=ftr; above20[sym]=(ftr["close"]>ftr["ema20"]).astype(float)
        print(f"  loaded {sym}: {len(ftr)} bars ({ftr.index[0].date()})", flush=True)
    except Exception as e: print(f"  err {sym}: {e}", flush=True)
print(f"[load] usable: {len(feats)}", flush=True)
breadth=pd.DataFrame(above20).sort_index().mean(axis=1,skipna=True)
breadth_pct=breadth.rolling(100,min_periods=50).rank(pct=True)*100

raw_trend=[]
for s in feats:
    for t in backtest_donchian(feats[s]):
        raw_trend.append(dict(sym=s, entry_time=t["entry_time"], r=t["r"]))
raw_trend.sort(key=lambda t:t["entry_time"])
mldf=build_mldf(raw_trend,feats,breadth,breadth_pct)
print(f"[signals] 15m Donchian trades: {len(raw_trend)}", flush=True)

tr=mldf[mldf.ts<SPLIT]; te=mldf[mldf.ts>=SPLIT]
print(f"[split] train={len(tr)} test={len(te)}", flush=True)
m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr[FEATS],tr["win"])
P=m.predict_proba(te[FEATS])[:,1]; q=0.65; thr=np.quantile(P,1-q)
kept=set(te[P>=thr]["ts"])
champ=[t for t in raw_trend if t["entry_time"]>=SPLIT and t["entry_time"] in kept]
for t in champ: t["adj_r"]=t["r"]-2*FEE
print(f"[bt] 15m champion (test half): {len(champ)}", flush=True)

print("\n"+"="*78)
print("TEST 28 — 15m condition-aware TREND (proof-of-concept, 2026-05-15..09 test)")
print("="*78)
eq=1.0;peak=1.0;mdd=0.0
for t in sorted(champ,key=lambda x:x["entry_time"]):
    eq*=(1+0.01*t["adj_r"]); peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
rs=[t["adj_r"] for t in champ]
wins=sum(1 for r in rs if r>0)/len(rs)
pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
msum=defaultdict(float)
for t in champ: msum[(t["entry_time"].year,t["entry_time"].month)]+=t["adj_r"]
pos=sum(1 for v in msum.values() if v>0)
print(f"n={len(champ)} win={wins:.0%} PF@c={pf:.3f} MAX DD={mdd:.1%} prof-months={pos}/{len(msum)}")
for (yy,mm) in sorted(msum):
    nr=msum[(yy,mm)]; print(f"   {yy}-{mm:02d}  netR={nr:>+7.2f}  {'OK' if nr>0 else 'LOSS'}")
# dollar sim $100, $2/trade
eq=100.0
for t in sorted(champ,key=lambda x:x["entry_time"]): eq+=2.0*t["adj_r"]
print(f"$100 @ $2/trade -> ${eq:,.2f} ({(eq/100-1):+.1%})")
print("\n[done]")
