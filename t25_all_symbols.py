"""
TEST 25c — T25 (condition-aware trend) on the FULL universe (all 73 symbols in cache),
not the 20-sym subset. Walk-forward 2025 (train 2024) and 2026 (train 2024+2025).
Reports PF@cost, MAX DD, profitable-months, # symbols profitable, and a $100/$2 dollar
sim. 2024 excluded (needs 2023 fetch for all 73; validated on 20-sym subset in T25).
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

CACHE = "quantlab_cache"
FEE = 0.0005


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


print("[load] all symbols in cache …", flush=True)
syms = sorted(f[:-len("_1H.parquet")] for f in os.listdir(CACHE) if f.endswith("_1H.parquet"))
print(f"[load] {len(syms)} symbols found", flush=True)
feats, above20 = {}, {}
for sym in syms:
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    try:
        df = pd.read_parquet(p); df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for c in ["open","high","low","close","vol"]:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        f=add_features(df); f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f)>=IS_LOOKBACK+RECAL_EVERY: feats[sym]=f; above20[sym]=(f["close"]>f["ema20"]).astype(float)
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
print(f"[signals] 1H Donchian trades 2024-2026: {len(raw_trend)} across {len(feats)} syms", flush=True)

tr_champ=[]
for Y in [2025,2026]:
    tr=mldf[mldf.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te=mldf[mldf.ts.dt.year==Y]
    if len(tr)<50 or len(te)==0: continue
    m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr[FEATS],tr["win"])
    P=m.predict_proba(te[FEATS])[:,1]; q=0.65; thr=np.quantile(P,1-q)
    kept=set(te[P>=thr]["ts"])
    for t in raw_trend:
        if t["entry_time"].year==Y and t["entry_time"] in kept: tr_champ.append(t)
for t in tr_champ: t["adj_r"]=t["r"]-2*FEE
print(f"[bt] T25 condition-aware trend champion trades (all syms): {len(tr_champ)}", flush=True)

def report_year(Y):
    yt=[t for t in tr_champ if t["entry_time"].year==Y]
    if not yt: return
    eq=1.0;peak=1.0;mdd=0.0
    for t in sorted(yt,key=lambda x:x["entry_time"]):
        eq*=(1+0.01*t["adj_r"]); peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
    rs=[t["adj_r"] for t in yt]
    wins=sum(1 for r in rs if r>0)/len(rs)
    pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
    msum=defaultdict(float); sym_net=defaultdict(float)
    for t in yt:
        msum[(t["entry_time"].year,t["entry_time"].month)]+=t["adj_r"]
        sym_net[t["sym"]]+=t["adj_r"]
    pos=sum(1 for v in msum.values() if v>0)
    nprof=sum(1 for v in sym_net.values() if v>0)
    print(f"\n[{Y}] n={len(yt)} win={wins:.0%} PF@c={pf:.3f} MAX DD={mdd:.1%} prof-months={pos}/{len(msum)}  symbols-profitable={nprof}/{len(sym_net)}")

print("\n"+"="*82)
print("TEST 25c — T25 condition-aware TREND on FULL universe (all cached symbols)")
print("="*82)
report_year(2025); report_year(2026)
# full 2-yr
yt=tr_champ
eq=1.0;peak=1.0;mdd=0.0
for t in sorted(yt,key=lambda x:x["entry_time"]):
    eq*=(1+0.01*t["adj_r"]); peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
rs=[t["adj_r"] for t in yt]
pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
print(f"\n[FULL 2025-2026] n={len(yt)} PF@c={pf:.3f} MAX DD={mdd:.1%}")

# dollar sim $100, $2/trade fixed
eq=100.0
for Y in [2025,2026]:
    yt=[t for t in tr_champ if t["entry_time"].year==Y]
    start=eq
    for t in sorted(yt,key=lambda x:x["entry_time"]): eq+=2.0*t["adj_r"]
    print(f"[{Y}] $100 start -> ${eq:,.2f} ({(eq/start-1):+.1%})")
print(f"[FULL $] $100 -> ${eq:,.2f} ({(eq/100-1):+.1%})")
print("\n[done]")
