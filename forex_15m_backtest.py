"""
FOREX 15m DIRECTIONAL BACKTEST on Deriv. The user wants a real shot at frx 15m binary.
Forex has real structure (sessions, trends) unlike synthetics. Pull 15m for 10 pairs over
~2.7y, build forex-suited features, RF-predict up/down, report FULL accuracy AND thresholded
win rate. The binary profit bar = >55% directional accuracy.
"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, websocket
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

APP_ID=1089
START=int(pd.Timestamp("2024-01-01",tz="UTC").timestamp())
END=int(pd.Timestamp("2026-09-02",tz="UTC").timestamp())
PAIRS=["frxEURUSD","frxGBPUSD","frxUSDJPY","frxAUDUSD","frxUSDCHF",
       "frxEURGBP","frxUSDCAD","frxEURJPY","frxGBPJPY","frxNZDUSD"]


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
        df=df[["datetime","open","high","low","close"]].set_index("datetime").astype(float)
        return df
    except Exception as e:
        print(f"    {symbol} pull fail: {e}"); return None


def feats(df):
    c=df["close"]; d=df.copy()
    d["r1"]=c.pct_change(1); d["r3"]=c.pct_change(3); d["r5"]=c.pct_change(5)
    d["r10"]=c.pct_change(10); d["r20"]=c.pct_change(20)
    d["rsi"]=100-100/(1+c.diff().clip(lower=0).rolling(14).mean()/(-c.diff().clip(upper=0).rolling(14).mean()).abs())
    d["body"]=(d["close"]-d["open"])/d["open"]; d["rng"]=(d["high"]-d["low"])/d["close"]
    d["sma20"]=c.rolling(20).mean(); d["sma50"]=c.rolling(50).mean(); d["sma96"]=c.rolling(96).mean()
    d["d20"]=c/d["sma20"]-1; d["d50"]=c/d["sma50"]-1; d["d96"]=c/d["sma96"]-1
    d["vol"]=c.rolling(20).std()
    hr=d.index.hour
    d["asia"]=((hr>=0)&(hr<8)).astype(int); d["london"]=((hr>=8)&(hr<16)).astype(int); d["ny"]=((hr>=13)&(hr<22)).astype(int)
    d["dow"]=d.index.dayofweek
    d["up"]=(c.shift(-1)>c).astype(int)
    return d.dropna()


def backtest(df):
    d=feats(df)
    if len(d)<2000: return None
    F=["r1","r3","r5","r10","r20","rsi","body","rng","d20","d50","d96","vol","asia","london","ny","dow"]
    cut=int(len(d)*0.7); tr,te=d.iloc[:cut],d.iloc[cut:]
    m=RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=200,n_jobs=-1,random_state=0).fit(tr[F],tr["up"])
    p=m.predict_proba(te[F])[:,1]
    full_acc=(m.predict(te[F])==te["up"]).mean()
    out={"full_acc":full_acc,"n_test":len(te)}
    for thr in [0.55,0.60,0.65]:
        longs=p>thr; shorts=p<(1-thr); trd=te.iloc[longs|shorts].copy()
        trd["pred"]=np.where(p[longs|shorts]>0.5,1,0)
        n=len(trd); win=(trd["pred"]==trd["up"]).mean() if n>0 else 0
        out[f"win_{thr:.2f}"]=win; out[f"n_{thr:.2f}"]=n
    return out


print(f"[forex 15m] {len(PAIRS)} pairs, 2024-01..2026-09", flush=True)
results={}
for sym in PAIRS:
    print(f"[pull] {sym} 15m …", flush=True)
    df=pull(sym,START,END,900)
    if df is None or len(df)<500:
        print(f"  {sym}: NO DATA", flush=True); continue
    r=backtest(df)
    results[sym]=r
    print(f"  {sym}: n={len(df)} full_acc={r['full_acc']:.1%}  "
          f"win@.55={r[f'win_{0.55:.2f}']:.1%}(n={r[f'n_{0.55:.2f}']})  "
          f"win@.60={r[f'win_{0.60:.2f}']:.1%}(n={r[f'n_{0.60:.2f}']})  "
          f"win@.65={r[f'win_{0.65:.2f}']:.1%}(n={r[f'n_{0.65:.2f}']})", flush=True)

print("\n"+"="*78)
print("FOREX 15m — DIRECTIONAL ACCURACY (binary needs >55%)")
print("="*78)
for sym,r in results.items():
    edge="EDGE ✓" if (r["full_acc"]>=0.55 and r["n_test"]>500) else "no"
    print(f"  {sym:12s} full_acc={r['full_acc']:.1%}  "
          f"win.55={r[f'win_{0.55:.2f}']:.1%}({r[f'n_{0.55:.2f}']})  "
          f"win.60={r[f'win_{0.60:.2f}']:.1%}({r[f'n_{0.60:.2f}']})  {edge}")
print("[done]")
