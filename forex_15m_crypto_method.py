"""
TEST 32 — Apply CRYPTO trend/reversal methodology to FOREX 15m BINARIES.
Reuse ql_engine.add_features (ema_dist, rsi, adx, bb_width, vol regimes) + Donchian
breakout + Family-A MR signals (exactly T25/T24), but label each signal by its
NEXT-15m DIRECTION (binary outcome) and filter with an RF condition model. Measure
the BINARY WIN RATE — does our crypto edge transfer to forex 15m binaries?
"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import pandas as pd, numpy as np, websocket
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
from ql_engine import add_features, build_signal_mask, sim_symbol, IS_LOOKBACK, RECAL_EVERY
import demo_bot as bot

APP_ID=1089
START=int(pd.Timestamp("2024-01-01",tz="UTC").timestamp())
END=int(pd.Timestamp("2026-09-02",tz="UTC").timestamp())
PAIRS=["frxEURUSD","frxGBPUSD","frxUSDCHF","frxEURGBP","frxNZDUSD","frxAUDUSD","frxUSDCAD","frxUSDJPY"]
FAM_A=["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]


def pull(symbol,start,end,gran=900,chunk=5000):
    try:
        ws=websocket.create_connection(f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}",timeout=30)
        candles=[]; cur=start; req=0
        while cur<end:
            req+=1; e=min(cur+chunk*gran,end)
            ws.send(json.dumps({"ticks_history":symbol,"style":"candles","granularity":gran,
                                "start":cur,"end":e,"count":chunk,"req_id":req}))
            got=None
            while True:
                m=json.loads(ws.recv())
                if m.get("req_id")==req and ("candles" in m or "error" in m): got=m; break
            if "error" in got:
                print(f"    {symbol} err: {got['error'].get('message','?')}"); break
            cs=got["candles"]
            if not cs: break
            candles.extend(cs); cur=int(cs[-1]["epoch"])+gran; time.sleep(0.2)
        ws.close()
        if not candles: return None
        df=pd.DataFrame(candles); df["epoch"]=pd.to_numeric(df["epoch"])
        df=df.drop_duplicates("epoch").sort_values("epoch")
        df["datetime"]=pd.to_datetime(df["epoch"],unit="s",utc=True)
        df=df[["datetime","open","high","low","close"]].set_index("datetime").astype(float); df["vol"]=1.0  # Deriv fx candles have no vol; unit proxy for add_features
        return df
    except Exception as e:
        print(f"    {symbol} pull fail: {e}"); return None


def donchian_signals(df, N=20, Nx=20, atr_mult=2.0, adx_min=20.0):
    df=df.copy()
    hh=df["high"].rolling(N).max().shift(1); ll=df["low"].rolling(Nx).min().shift(1)
    sig=[]
    for i in range(N,len(df)):
        bar=df.iloc[i]
        if bar["close"]>hh.iloc[i] and bar["adx14"]>adx_min and bar["close"]>bar["ema200"]:
            sig.append((df.index[i], +1))
        elif bar["close"]<ll.iloc[i] and bar["adx14"]>adx_min and bar["close"]<bar["ema200"]:
            sig.append((df.index[i], -1))
    return sig


print("[load] pull + features + signals (forex 15m) …", flush=True)
feats={}
for s in PAIRS:
    print(f"  {s} …", flush=True)
    df=pull(s,START,END,900)
    if df is None or len(df)<500: print(f"    {s}: NO DATA"); continue
    f=add_features(df)
    f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
    f["hour"]=f.index.hour; f["dow"]=f.index.dayofweek
    feats[s]=f
    print(f"    {s}: {len(f)} bars", flush=True)

FEATS=["ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r","adx14","rsi14","hour","dow"]
print("[signals] donchian + MR …", flush=True)
raw=[]
for s,f in feats.items():
    # trend signals (both directions)
    for t,d in donchian_signals(f):
        if t not in f.index: continue
        row=f.loc[t]; raw.append(dict(sym=s, ts=pd.Timestamp(t), dirn=d,
            ema_dist_pct=row.ema_dist_pct, real_vol_20=row.real_vol_20, bb_width=row.bb_width,
            prev_range_r=row.prev_range_r, prev_body_r=row.prev_body_r, adx14=row.adx14,
            rsi14=row.rsi14 if "rsi14" in row else np.nan, hour=row.hour, dow=row.dow))
    # MR (Family A) long = reversion up
    mask=build_signal_mask(f, FAM_A, "green", 1.5)
    for t in sim_symbol(f, mask, 1.5, dict(entry_next=False, exit="base", hours=None)):
        if t["entry_time"] not in f.index: continue
        row=f.loc[t["entry_time"]]; raw.append(dict(sym=s, ts=pd.Timestamp(t["entry_time"]), dirn=+1,
            ema_dist_pct=row.ema_dist_pct, real_vol_20=row.real_vol_20, bb_width=row.bb_width,
            prev_range_r=row.prev_range_r, prev_body_r=row.prev_body_r, adx14=row.adx14,
            rsi14=row.rsi14 if "rsi14" in row else np.nan, hour=row.hour, dow=row.dow))
raw=[r for r in raw if pd.notna(r["rsi14"])]
# binary label: correct direction on next 15m close
for r in raw:
    f=feats[r["sym"]]; i=f.index.get_indexer([r["ts"]])[0]
    if i+1>=len(f): r["win"]=np.nan; continue
    nxt=f["close"].iloc[i+1]; cur=f["close"].iloc[i]
    correct = (nxt>cur) if r["dirn"]==+1 else (nxt<cur)
    r["win"]=1 if correct else 0
raw=[r for r in raw if pd.notna(r.get("win"))]
print(f"[signals] total raw = {len(raw)}", flush=True)

# walk-forward by year
raw.sort(key=lambda r:r["ts"])
def year(r): return r["ts"].year
champ=[]
for Y in [2024,2025,2026]:
    tr=[r for r in raw if r["ts"].year<Y]; te=[r for r in raw if r["ts"].year==Y]
    if len(tr)<200 or len(te)==0: continue
    Xtr=pd.DataFrame(tr)[FEATS]; ytr=pd.Series([r["win"] for r in tr])
    Xte=pd.DataFrame(te)[FEATS]; yte=[r["win"] for r in te]
    m=RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=100,n_jobs=-1,random_state=0).fit(Xtr,ytr)
    P=m.predict_proba(Xte)[:,1]
    for r,p in zip(te,P): r["p"]=p
    q=0.58; kept=[r for r in te if r["p"]>=q]
    n=len(kept); win=np.mean([r["win"] for r in kept]) if n else 0
    print(f"  [{Y}] signals={len(te)} traded@{q}={n}  BINARY WIN RATE={win:.1%}", flush=True)
    champ.extend(kept)

n=len(champ); win=np.mean([r["win"] for r in champ]) if n else 0
print(f"\n[COMBINED] traded signals={n}  BINARY WIN RATE={win:.1%}  "
      f"-> {'EDGE ✓ (>55%)' if (n>=300 and win>=0.55) else 'no edge'}")
print("[done]")
