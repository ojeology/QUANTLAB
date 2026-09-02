"""
TEST 30 — TEST ALL untested 1H ideas (data already cached, no fetch):
  A. Cross-sectional MOMENTUM ROTATION (rank coins, hold strongest) -- never tested.
  B. TRAILING-EXIT TREND (T25 Donchian + trailing stop instead of fixed ATR) -- holds winners longer.
  C. UNIFIED MODEL (one RF on combined MR+trend raw signals) -- routes both via learned conditions.
  D. BTC-ETH PAIRS mean-reversion (spread z-score).
Report PF/DD/months (or CAGR/DD for portfolio-style) per year + FULL, walk-forward 2024/25/26.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from ql_engine import add_features, build_signal_mask, sim_symbol, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import build_mldf, FEATS
import demo_bot as bot
from sklearn.ensemble import RandomForestClassifier

SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]
CACHE = "quantlab_cache"; FAM_A = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]; FEE = 0.0005

print("[load] 20-sym full 1H (2023-2026) …", flush=True)
feats = {}
for s in SUBSET:
    p = os.path.join(CACHE, f"{s}_1H.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p); df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for c in ["open","high","low","close","vol"]:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
        f=add_features(df); f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f)>=IS_LOOKBACK+RECAL_EVERY+100: feats[s]=f
    except Exception as e: print(f"  err {s}: {e}", flush=True)
print(f"[load] usable: {len(feats)}", flush=True)
above20={s:(f["close"]>f["ema20"]).astype(float) for s,f in feats.items()}
breadth=pd.DataFrame(above20).sort_index().mean(axis=1,skipna=True)
breadth_pct=breadth.rolling(100,min_periods=50).rank(pct=True)*100


def report_curve(step_rets, name, yearly=True):
    """step_rets: list of (timestamp, return_fraction). Builds equity, reports."""
    print(f"\n--- {name} ---")
    eq=1.0; peak=1.0; fmdd=0.0
    for Y in [2024,2025,2026]:
        yr=[(ts,r) for ts,r in step_rets if ts.year==Y]
        if not yr: continue
        e=1.0; pk=1.0; mdd=0.0; pos=0; n=0
        for ts,r in yr:
            e*=(1+r); pk=max(pk,e); mdd=min(mdd,e/pk-1); n+=1; pos+= (1 if r>0 else 0)
        cagr=(e**(365.25/len(yr)))-1 if len(yr)>0 else 0
        print(f"  [{Y}] days={len(yr)} win={pos/n:.0%} CAGR={cagr:+.1%} MAX DD={mdd:.1%}")
        eq*=e; peak=max(peak,eq); fmdd=min(fmdd,eq/peak-1)
    tot=len(step_rets); pos=sum(1 for _,r in step_rets if r>0)
    cagr_all=(eq**(365.25/tot))-1
    print(f"  [FULL] days={tot} win={pos/tot:.0%} CAGR={cagr_all:+.1%} MAX DD={fmdd:.1%}")


# ── A. CROSS-SECTIONAL MOMENTUM ROTATION ──
def strat_mom(syms, lookback=48, hold=24, topk=5):
    ref = feats["BTC_USDT_SWAP"].index[::hold]
    rets=[]
    for i in range(len(ref)-1):
        t, tnext = ref[i], ref[i+1]
        moms={}; r={}
        for s in syms:
            f=feats[s]
            if t not in f.index or tnext not in f.index: continue
            idx=f.index.get_indexer([t])[0]
            if idx<lookback: continue
            moms[s]=f["close"].iloc[idx]/f["close"].iloc[idx-lookback]-1
            r[s]=f.loc[tnext,"close"]/f.loc[t,"close"]-1
        if len(moms)<topk: continue
        top=sorted(moms,key=lambda s:moms[s],reverse=True)[:topk]
        rets.append((t, sum(r[s] for s in top)/topk - 2*FEE))
    return rets
print("[A] momentum rotation …", flush=True)
report_curve(strat_mom(list(feats.keys())), "A. CROSS-SECTIONAL MOMENTUM (top5 by 48-bar mom, daily rebal)")


# ── B. TRAILING-EXIT TREND ──
def backtest_donchian_trail(df, N=20, Nx=20, atr_mult=2.0, adx_min=20.0):
    df=df.copy()
    hh=df["high"].rolling(N).max().shift(1); ll=df["low"].rolling(Nx).min().shift(1)
    trades=[]; in_pos=False; ep=None; trail=None
    for i in range(N,len(df)):
        bar=df.iloc[i]
        if not in_pos:
            if bar["close"]>hh.iloc[i] and bar["adx14"]>adx_min and bar["close"]>bar["ema200"]:
                ep=bar["close"]; trail=ep-atr_mult*bar["atr14"]; in_pos=True
        else:
            trail=max(trail, bar["close"]-atr_mult*bar["atr14"])
            ex=None
            if bar["low"]<=trail: ex=trail; et="TRAIL"
            elif bar["close"]<ll.iloc[i]: ex=bar["close"]; et="BRK"
            if ex is not None:
                trades.append(dict(entry_time=df.index[i], r=(ex/ep-1.0))); in_pos=False
    return trades
print("[B] trailing trend …", flush=True)
rawB=[]
for s in feats:
    for t in backtest_donchian_trail(feats[s]): rawB.append(dict(sym=s, entry_time=t["entry_time"], r=t["r"]))
rawB.sort(key=lambda t:t["entry_time"])
mldfB=build_mldf(rawB,feats,breadth,breadth_pct)
chB=[]
for Y in [2024,2025,2026]:
    tr=mldfB[mldfB.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te=mldfB[mldfB.ts.dt.year==Y]
    if len(tr)<50 or len(te)==0: continue
    m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr[FEATS],tr["win"])
    P=m.predict_proba(te[FEATS])[:,1]; q=0.65; thr=np.quantile(P,1-q)
    kept=set(te[P>=thr]["ts"])
    for t in rawB:
        if t["entry_time"].year==Y and t["entry_time"] in kept: chB.append(t)
for t in chB: t["adj_r"]=t["r"]-2*FEE
print(f"  B champion trades: {len(chB)}", flush=True)
def report_trades(trades, name):
    print(f"\n--- {name} ---")
    for Y in [2024,2025,2026]:
        yt=[t for t in trades if t["entry_time"].year==Y]
        if not yt: continue
        eq=1.0;peak=1.0;mdd=0.0
        for t in sorted(yt,key=lambda x:x["entry_time"]): eq*=(1+0.01*t["adj_r"]);peak=max(peak,eq);mdd=min(mdd,eq/peak-1)
        rs=[t["adj_r"] for t in yt]; wins=sum(1 for r in rs if r>0)/len(rs)
        pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
        print(f"  [{Y}] n={len(yt)} win={wins:.0%} PF@c={pf:.3f} MAX DD={mdd:.1%}")
    rs=[t["adj_r"] for t in trades]; pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
    print(f"  [FULL] PF@c={pf:.3f}")
report_trades(chB, "B. TRAILING-EXIT TREND (Donchian + trailing stop, RF q65)")


# ── C. UNIFIED MR+TREND MODEL ──
print("[C] unified MR+trend …", flush=True)
mask={s:build_signal_mask(feats[s],FAM_A,"green",1.5) for s in feats}
mr_raw=[]
for s in feats:
    for t in sim_symbol(feats[s],mask[s],1.5,dict(entry_next=False,exit="base",hours=None)):
        t["sym"]=s; mr_raw.append(t)
tr_raw=[]
for s in feats:
    for t in backtest_donchian_trail(feats[s]): tr_raw.append(dict(sym=s, entry_time=t["entry_time"], r=t["r"]))
# normalise keys
mr_raw=[dict(sym=t["sym"], entry_time=t["entry_time"], r=t["r"]) for t in mr_raw]
rawC=mr_raw+tr_raw; rawC.sort(key=lambda t:t["entry_time"])
mldfC=build_mldf(rawC,feats,breadth,breadth_pct)
chC=[]
for Y in [2024,2025,2026]:
    tr=mldfC[mldfC.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te=mldfC[mldfC.ts.dt.year==Y]
    if len(tr)<50 or len(te)==0: continue
    m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr[FEATS],tr["win"])
    P=m.predict_proba(te[FEATS])[:,1]; q=0.65; thr=np.quantile(P,1-q)
    kept=set(te[P>=thr]["ts"])
    for t in rawC:
        if t["entry_time"].year==Y and t["entry_time"] in kept: chC.append(t)
for t in chC: t["adj_r"]=t["r"]-2*FEE
print(f"  C champion trades: {len(chC)}", flush=True)
report_trades(chC, "C. UNIFIED MODEL (one RF on combined MR+trend raw signals, q65)")


# ── D. BTC-ETH PAIRS ──
print("[D] BTC-ETH pairs …", flush=True)
btc=feats["BTC_USDT_SWAP"]; eth=feats["ETH_USDT_SWAP"]
join=pd.concat([btc["close"].rename("b"), eth["close"].rename("e")],axis=1).dropna()
join["spread"]=np.log(join["b"]/join["e"])
join["z"]=(join["spread"]-join["spread"].rolling(100,min_periods=50).mean())/join["spread"].rolling(100,min_periods=50).std()
join["bret"]=join["b"].pct_change(); join["eret"]=join["e"].pct_change()
step=[]
pos=0
for ts,row in join.iterrows():
    if pd.isna(row["z"]): continue
    if abs(row["z"])>2 and pos==0: pos=-1*np.sign(row["z"])  # long spread if z<-2 (long b short e), short spread if z>2
    elif abs(row["z"])<0.5 and pos!=0: pos=0
    if pos!=0:
        step.append((ts, pos*(row["bret"]-row["eret"]) - 2*FEE))  # pair return
print(f"  D active bars: {len(step)}", flush=True)
report_curve(step, "D. BTC-ETH PAIRS (spread z-score mean-reversion)")

print("\n[done]")
