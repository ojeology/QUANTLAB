"""
DERIV SWEEP — pull 5m + 1m synthetic-index candles and 15m forex candles, run a BINARY
directional backtest (win rate) on each. Goal: find ANY instrument with a >55% directional
edge (the bar binaries need to be profitable). Synthetics 24/7; forex min granularity 15m.
Note: Deriv's finest candle granularity is 60s (1m); true 1s candles aren't served historically.
"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, websocket
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

APP_ID = 1089
START = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
END   = int(pd.Timestamp("2026-09-02", tz="UTC").timestamp())

# (symbol, granularity_seconds, label)
INSTRS = [
    ("R_10",300,"Vol-10 5m"),("R_25",300,"Vol-25 5m"),("R_50",300,"Vol-50 5m"),
    ("R_75",300,"Vol-75 5m"),("R_100",300,"Vol-100 5m"),
    ("stpRNG",300,"Step 5m"),
    ("jump10",300,"Jump-10 5m"),("jump25",300,"Jump-25 5m"),("jump50",300,"Jump-50 5m"),
    ("jump75",300,"Jump-75 5m"),("jump100",300,"Jump-100 5m"),
    ("R_75",60,"Vol-75 1m (finest)"),
    ("frxEURUSD",900,"EURUSD 15m"),("frxGBPUSD",900,"GBPUSD 15m"),("frxUSDJPY",900,"USDJPY 15m"),
]


def pull(symbol, start, end, gran, chunk=5000):
    try:
        ws = websocket.create_connection(f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}", timeout=30)
        candles=[]; cur=start; req=0
        while cur < end:
            req+=1; e=min(cur+chunk*gran, end)
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
            candles.extend(cs); cur=int(cs[-1]["epoch"])+gran; time.sleep(0.25)
        ws.close()
        if not candles: return None
        df=pd.DataFrame(candles); df["epoch"]=pd.to_numeric(df["epoch"])
        df=df.drop_duplicates("epoch").sort_values("epoch")
        df["datetime"]=pd.to_datetime(df["epoch"],unit="s",utc=True)
        df=df[["datetime","open","high","low","close"]].set_index("datetime").astype(float)
        return df
    except Exception as e:
        print(f"    {symbol} pull fail: {e}"); return None


def backtest(df, thr=0.60):
    c=df["close"]; d=df.copy()
    d["r1"]=c.pct_change(1); d["r3"]=c.pct_change(3); d["r5"]=c.pct_change(5)
    d["rsi"]=100-100/(1+c.diff().clip(lower=0).rolling(14).mean()/(-c.diff().clip(upper=0).rolling(14).mean()).abs())
    d["body"]=(d["close"]-d["open"])/d["open"]; d["rng"]=(d["high"]-d["low"])/d["close"]
    d["ma20"]=c.rolling(20).mean(); d["dist_ma"]=c/d["ma20"]-1; d["vol"]=c.rolling(20).std()
    d["hr"]=d.index.hour; d["dow"]=d.index.dayofweek
    d["up"]=(c.shift(-1)>c).astype(int)
    d=d.dropna()
    if len(d)<2000: return None
    feats=["r1","r3","r5","rsi","body","rng","dist_ma","vol","hr","dow"]
    cut=int(len(d)*0.7); tr,te=d.iloc[:cut],d.iloc[cut:]
    m=RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=100,n_jobs=-1,random_state=0).fit(tr[feats],tr["up"])
    p=m.predict_proba(te[feats])[:,1]
    longs=p>thr; shorts=p<(1-thr)
    traded=te.iloc[longs|shorts].copy(); traded["pred"]=np.where(p[longs|shorts]>0.5,1,0)
    n=len(traded)
    if n<100: return (None,n)
    win=(traded["pred"]==traded["up"]).mean()
    return (win,n)


print(f"[sweep] {len(INSTRS)} instruments, 2026-01-01..09-02", flush=True)
results=[]
for sym,gran,label in INSTRS:
    print(f"[pull] {label} …", flush=True)
    df=pull(sym,START,END,gran)
    if df is None or len(df)<500:
        print(f"  {label}: NO DATA", flush=True); continue
    res=backtest(df,0.60)
    win,n = res if res else (None,0)
    print(f"  {label}: candles={len(df)}  traded_n={n}  WIN={win:.1%}" if win is not None else f"  {label}: candles={len(df)}  traded_n={n}  WIN=n/a", flush=True)
    results.append((label,len(df),n,win))

print("\n"+"="*70)
print("DERIV SWEEP — binary directional WIN RATE (need >=55% & enough trades)")
print("="*70)
for label,ncand,nt,win in results:
    edge = "EDGE ✓" if (win is not None and win>=0.55 and nt>=200) else "no"
    print(f"  {label:18s} candles={ncand:>6}  trades={nt:>5}  win={win if win else 0:.1%}  {edge}")
print("[done]")
