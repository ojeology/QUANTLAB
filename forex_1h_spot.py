"""
TEST 33 — Apply CRYPTO T25/T27 methodology to FOREX 1H SPOT (Deriv). Not binary: measure
R-multiple PF like crypto. Reuse ql_engine (add_features, Donchian trend, Family-A MR) + RF
condition filter. Deriv 1H serves ~7 months/pair. Walk-forward (train first 60%, test last 40%).
Question: does our proven crypto edge TRANSFER to forex 1H spot?
"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import pandas as pd, numpy as np, websocket
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
from ql_engine import add_features, build_signal_mask, sim_symbol, cost_adjusted_rs, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import build_mldf, FEATS
import demo_bot as bot

APP_ID=1089
PAIRS=["frxEURUSD","frxGBPUSD","frxUSDCHF","frxEURGBP","frxNZDUSD","frxAUDUSD","frxUSDCAD","frxUSDJPY"]
FAM_A=["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]; FEE=0.0003


def pull(symbol, gran=3600, chunk=5000):
    end=int(pd.Timestamp.now(tz="UTC").timestamp()*1000)-1
    try:
        ws=websocket.create_connection(f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}",timeout=30)
        candles=[]; cur=end
        while len(candles)<chunk:
            start=cur-chunk*gran*1000
            ws.send(json.dumps({"ticks_history":symbol,"style":"candles","granularity":gran,
                                "start":int(start/1000),"end":int(cur/1000),"count":chunk,"req_id":1}))
            got=None
            while True:
                m=json.loads(ws.recv())
                if "candles" in m or "error" in m: got=m; break
            if "error" in got: print(f"  {symbol} err {got['error'].get('message')}"); break
            cs=got["candles"]
            if not cs: break
            candles.extend(cs); cur=int(cs[0]["epoch"])*1000-1; time.sleep(0.2)
            if len(candles)>=chunk: break
        ws.close()
        df=pd.DataFrame(candles); df["ts"]=pd.to_numeric(df["ts"] if "ts" in df else df["epoch"])
        df["ts"]=pd.to_numeric(df["epoch"])
        for c in ["open","high","low","close"]: df[c]=pd.to_numeric(df[c],errors="coerce")
        df["datetime"]=pd.to_datetime(df["ts"],unit="s",utc=True)
        df=df[["datetime","open","high","low","close"]].set_index("datetime").astype(float)
        df["vol"]=1.0
        return df.drop_duplicates().sort_index()
    except Exception as e:
        print(f"  {symbol} pull fail: {e}"); return None


def backtest_donchian(df, N=20, Nx=20, atr_mult=2.0, adx_min=20.0):
    df=df.copy(); hh=df["high"].rolling(N).max().shift(1); ll=df["low"].rolling(Nx).min().shift(1)
    trades=[]; in_pos=False; ep=None; stop=None
    for i in range(N,len(df)):
        bar=df.iloc[i]
        if not in_pos:
            if bar["close"]>hh.iloc[i] and bar["adx14"]>adx_min and bar["close"]>bar["ema200"]:
                ep=bar["close"]; stop=ep-atr_mult*bar["atr14"]; in_pos=True
        else:
            ex=None
            if bar["low"]<=stop: ex=stop
            elif bar["close"]<ll.iloc[i]: ex=bar["close"]
            if ex is not None: trades.append(dict(entry_time=df.index[i], r=(ex/ep-1.0))); in_pos=False
    return trades


print("[load] forex 1H via Deriv …", flush=True)
feats={}
for s in PAIRS:
    print(f"  {s} …", flush=True)
    df=pull(s)
    if df is None or len(df)<200: print(f"    {s}: NO DATA"); continue
    f=add_features(df); f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"],inplace=True)
    if len(f)>=IS_LOOKBACK+RECAL_EVERY: feats[s]=f
    print(f"    {s}: {len(f)} bars {f.index[0].date()}..{f.index[-1].date()}", flush=True)

# dummy breadth (single instruments, no cross-market breadth)
_allidx=pd.concat([pd.Series(f.index) for f in feats.values()]).drop_duplicates().sort_values()
breadth=pd.Series(0.5,index=_allidx)
breadth_pct=pd.Series(0.5,index=_allidx)

print("[signals] donchian trend + Family-A MR …", flush=True)
trend_raw=[]; mr_raw=[]
for s,f in feats.items():
    for t in backtest_donchian(f): trend_raw.append(dict(sym=s,entry_time=t["entry_time"],r=t["r"]))
    mask=build_signal_mask(f,FAM_A,"green",1.5)
    for t in sim_symbol(f,mask,1.5,dict(entry_next=False,exit="base",hours=None)):
        t["sym"]=s
        ts=pd.Timestamp(t["entry_time"])
        if ts.tzinfo is None: ts=ts.tz_localize("UTC")
        t["entry_time"]=ts
        mr_raw.append(t)
trend_raw.sort(key=lambda t:t["entry_time"]); mr_raw.sort(key=lambda t:t["entry_time"])
mldf_tr=build_mldf(trend_raw,feats,breadth,breadth_pct)
print(f"  MR raw signals on forex: {len(mr_raw)}", flush=True)
mldf_mr=build_mldf(mr_raw,feats,breadth,breadth_pct) if len(mr_raw)>10 else None

# walk-forward: train first 60% by date, test last 40%
def wf(mldf, label, q=0.65):
    cut=mldf["ts"].quantile(0.6)
    tr=mldf[mldf.ts<cut]; te=mldf[mldf.ts>=cut]
    if len(tr)<50 or len(te)==0:
        print(f"  {label}: insufficient (train={len(tr)} test={len(te)})"); return
    m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr[FEATS],tr["win"])
    P=m.predict_proba(te[FEATS])[:,1]; thr=np.quantile(P,1-q)
    kept=set(te[P>=thr]["ts"])
    champ=[t for t in (trend_raw if "TREND" in label else mr_raw) if t["entry_time"] in kept]
    for t in champ: t["adj_r"]=t["r"]-2*FEE  # simple round-trip fee, consistent with crypto T30
    rs=[t["adj_r"] for t in champ]; pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
    eq=1.0;peak=1.0;mdd=0.0; msum=defaultdict(float)
    for t in sorted(champ,key=lambda x:x["entry_time"]):
        eq*=(1+0.01*t["adj_r"]); peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
        msum[(t["entry_time"].year,t["entry_time"].month)]+=t["adj_r"]
    pos=sum(1 for v in msum.values() if v>0)
    print(f"  {label}: n={len(champ)} PF={pf:.3f} DD={mdd:.1%} prof-months={pos}/{len(msum)}", flush=True)
    return champ

print("\n--- FOREX 1H SPOT (T25/T27 methodology) ---")
tr_ch=wf(mldf_tr,"TREND (T25)",0.65)
mr_ch=wf(mldf_mr,"MR (T24)",0.65) if mldf_mr is not None else None
if tr_ch and mr_ch:
    # T27-style trend-weighted 70/30 portfolio
    comb=defaultdict(lambda:0.0)
    for t in tr_ch: comb[t["entry_time"]]+=0.7*t["adj_r"]
    for t in mr_ch: comb[t["entry_time"]]+=0.3*t["adj_r"]
    rs=list(comb.values()); pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
    eq=1.0;peak=1.0;mdd=0.0
    for r in sorted(comb): eq*=(1+0.01*comb[r]); peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
    print(f"  COMBINED (T27 70/30): n={len(rs)} PF={pf:.3f} DD={mdd:.1%}", flush=True)
elif tr_ch:
    print("  COMBINED: MR silent on forex 1H -> TREND-only edge (T25 transfers, MR does not)", flush=True)
print("[done]")
