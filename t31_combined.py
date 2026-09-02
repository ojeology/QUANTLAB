"""
TEST 31 — COMBINED MULTI-STRATEGY PORTFOLIO. Combine 3 validated 1H edges into one
diversified book (equal risk 1/3 each):
  T25  = condition-aware TREND (Donchian + RF q65)
  MR   = mean-reversion champion (SVM Q65 adaptive VolCeil)
  C    = UNIFIED model (one RF on combined MR+trend raw, q65)
Fetches 2023 1H (resume-safe, 1H not throttled) so 2024 is walk-forward validated too.
Reports combined PF/DD/months + each sub-strategy.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from ql_engine import add_features, build_signal_mask, sim_symbol, cost_adjusted_rs, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import SVMQ65Adaptive, build_mldf, FEATS
import demo_bot as bot
from sklearn.ensemble import RandomForestClassifier

SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]
CACHE = "quantlab_cache"; SAVE = "quantlab_cache_2023"; FAM_A = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]; FEE=0.0005


def fetch_before(end_ms, n_bars, inst, bar="1H", page_limit=200):
    all_rows, after, pages = [], end_ms, 0
    while len(all_rows) < n_bars and pages < page_limit:
        params={"instId":inst,"bar":bar,"limit":bot.PAGE_LIMIT,"after":str(after)}
        raw=bot._get(bot.OKX_CANDLES,params) or bot._get(bot.OKX_CANDLES_CUR,params)
        if not raw: break
        all_rows.extend(raw); pages+=1; after=int(raw[-1][0])
        if len(all_rows)>=n_bars: break
        time.sleep(bot.PAGE_DELAY)
    if not all_rows: return None
    df=pd.DataFrame(all_rows,columns=bot.CANDLE_COLS); df["ts"]=pd.to_numeric(df["ts"])
    for c in ["open","high","low","close","vol"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["datetime"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
    return df[["datetime","open","high","low","close","vol"]].sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True).set_index("datetime")


# ── fetch 2023 1H (resume-safe) ──
print("[fetch] 2023 1H for 20 subset …", flush=True)
for s in SUBSET:
    p=os.path.join(SAVE, f"{s}_1H_full.parquet")
    if os.path.exists(p):
        try:
            if len(pd.read_parquet(p))>=30000: continue
        except Exception: pass
    inst=s.replace("_","-")
    df=pd.read_parquet(os.path.join(CACHE, f"{s}_1H.parquet")); df.index=pd.to_datetime(df.index,utc=True)
    end_ms=int(df.index[0].timestamp()*1000)-1
    f2023=fetch_before(end_ms,9200,inst,"1H")
    if f2023 is not None and len(f2023):
        full=pd.concat([f2023,df]); full=full[~full.index.duplicated(keep="last")].sort_index()
        full.to_parquet(p); 
        if hasattr(os,"sync"): os.sync()
print("[fetch] 2023 done", flush=True)

# ── load ──
print("[load] 20-sym full 1H …", flush=True)
feats={}
for s in SUBSET:
    p=os.path.join(SAVE, f"{s}_1H_full.parquet")
    if not os.path.exists(p): p=os.path.join(CACHE, f"{s}_1H.parquet")
    try:
        df=pd.read_parquet(p); df.index=pd.to_datetime(df.index,utc=True); df.sort_index(inplace=True)
        for c in ["open","high","low","close","vol"]:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
        f=add_features(df); f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"],inplace=True)
        if len(f)>=IS_LOOKBACK+RECAL_EVERY+100: feats[s]=f
    except Exception as e: print(f"  err {s}: {e}", flush=True)
print(f"[load] usable: {len(feats)}", flush=True)
above20={s:(f["close"]>f["ema20"]).astype(float) for s,f in feats.items()}
breadth=pd.DataFrame(above20).sort_index().mean(axis=1,skipna=True)
breadth_pct=breadth.rolling(100,min_periods=50).rank(pct=True)*100


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
            if bar["low"]<=stop: ex=stop; et="SL"
            elif bar["close"]<ll.iloc[i]: ex=bar["close"]; et="BRK"
            if ex is not None: trades.append(dict(entry_time=df.index[i], r=(ex/ep-1.0))); in_pos=False
    return trades


# ── build 3 champion sets ──
print("[build] T25 trend + MR + unified …", flush=True)
mask={s:build_signal_mask(feats[s],FAM_A,"green",1.5) for s in feats}
mr_raw=[]
for s in feats:
    for t in sim_symbol(feats[s],mask[s],1.5,dict(entry_next=False,exit="base",hours=None)):
        t["sym"]=s; mr_raw.append(t)
tr_raw=[]
for s in feats:
    for t in backtest_donchian(feats[s]): tr_raw.append(dict(sym=s, entry_time=t["entry_time"], r=t["r"]))
rawC=mr_raw+tr_raw
for t in rawC: t["entry_time"]=t["entry_time"]
mr_raw.sort(key=lambda t:t["entry_time"]); tr_raw.sort(key=lambda t:t["entry_time"]); rawC.sort(key=lambda t:t["entry_time"])
mldf_mr=build_mldf(mr_raw,feats,breadth,breadth_pct)
mldf_tr=build_mldf(tr_raw,feats,breadth,breadth_pct)
mldfC=build_mldf(rawC,feats,breadth,breadth_pct)

mr_champ=[]; tr_champ=[]; c_champ=[]
for Y in [2024,2025,2026]:
    tr_mr=mldf_mr[mldf_mr.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te_mr=mldf_mr[mldf_mr.ts.dt.year==Y]
    tr_tr=mldf_tr[mldf_tr.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te_tr=mldf_tr[mldf_tr.ts.dt.year==Y]
    trC=mldfC[mldfC.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; teC=mldfC[mldfC.ts.dt.year==Y]
    if len(tr_mr)>=50 and len(te_mr)>0:
        m=SVMQ65Adaptive().fit_mldf(tr_mr); kept,_=m.keep_mldf(te_mr)
        for t in mr_raw:
            if t["entry_time"].year==Y and t["entry_time"] in kept: mr_champ.append(t)
    if len(tr_tr)>=50 and len(te_tr)>0:
        m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr_tr[FEATS],tr_tr["win"])
        P=m.predict_proba(te_tr[FEATS])[:,1]; thr=np.quantile(P,1-0.65)
        kept=set(te_tr[P>=thr]["ts"])
        for t in tr_raw:
            if t["entry_time"].year==Y and t["entry_time"] in kept: tr_champ.append(t)
    if len(trC)>=50 and len(teC)>0:
        m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(trC[FEATS],trC["win"])
        P=m.predict_proba(teC[FEATS])[:,1]; thr=np.quantile(P,1-0.65)
        kept=set(teC[P>=thr]["ts"])
        for t in rawC:
            if t["entry_time"].year==Y and t["entry_time"] in kept: c_champ.append(t)
mr_adj=cost_adjusted_rs(list(mr_champ),0.05)
for t,a in zip(mr_champ,mr_adj): t["adj_r"]=a
for t in tr_champ: t["adj_r"]=t["r"]-2*FEE
for t in c_champ: t["adj_r"]=t["r"]-2*FEE
print(f"[champ] MR={len(mr_champ)} TREND={len(tr_champ)} UNIFIED={len(c_champ)}", flush=True)


def report_sub(trades, name):
    for Y in [2024,2025,2026]:
        yt=[t for t in trades if t["entry_time"].year==Y]
        if not yt: continue
        eq=1.0;peak=1.0;mdd=0.0
        for t in sorted(yt,key=lambda x:x["entry_time"]): eq*=(1+0.01*t["adj_r"]);peak=max(peak,eq);mdd=min(mdd,eq/peak-1)
        rs=[t["adj_r"] for t in yt]; wins=sum(1 for r in rs if r>0)/len(rs)
        pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
        print(f"  {name} [{Y}] n={len(yt)} win={wins:.0%} PF={pf:.3f} DD={mdd:.1%}")
    rs=[t["adj_r"] for t in trades]; pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
    print(f"  {name} [FULL] PF={pf:.3f}")

print("\n--- SUB-STRATEGIES ---")
report_sub(mr_champ,"MR"); report_sub(tr_champ,"TREND"); report_sub(c_champ,"UNIFIED")


def report_combined(champs, risk_each=0.003, name="COMBINED"):
    print(f"\n--- {name} (equal risk {risk_each:.1%} x3 subs) ---")
    for Y in [2024,2025,2026]:
        subs={"MR":[t for t in champs["MR"] if t["entry_time"].year==Y],
              "TR":[t for t in champs["TR"] if t["entry_time"].year==Y],
              "C":[t for t in champs["C"] if t["entry_time"].year==Y]}
        ev={k:{t["entry_time"]:t["adj_r"] for t in v} for k,v in subs.items()}
        times=sorted(set().union(*[set(e.keys()) for e in ev.values()]))
        eq=1.0;peak=1.0;mdd=0.0; msum=defaultdict(float); rs=[]
        for ts in times:
            ret=0.0
            for k in ev:
                if ts in ev[k]: ret+=risk_each*ev[k][ts]
            eq*=(1+ret); peak=max(peak,eq); mdd=min(mdd,eq/peak-1); rs.append(ret)
            for k in ev:
                if ts in ev[k]: msum[(ts.year,ts.month)]+=ret
        wins=sum(1 for r in rs if r>0)/len(rs)
        pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
        pos=sum(1 for v in msum.values() if v>0)
        print(f"  [{Y}] n_trades={len(rs)} win={wins:.0%} PF={pf:.3f} MAX DD={mdd:.1%} prof-months={pos}/{len(msum)}")
        for (yy,mm) in sorted(msum):
            nr=msum[(yy,mm)]; print(f"     {yy}-{mm:02d} netR={nr:>+7.3f} {'OK' if nr>0 else 'LOSS'}")
    # full
    ev={"MR":{t["entry_time"]:t["adj_r"] for t in champs["MR"]},
        "TR":{t["entry_time"]:t["adj_r"] for t in champs["TR"]},
        "C":{t["entry_time"]:t["adj_r"] for t in champs["C"]}}
    times=sorted(set().union(*[set(e.keys()) for e in ev.values()]))
    eq=1.0;peak=1.0;mdd=0.0; rs=[]
    for ts in times:
        ret=sum(risk_each*ev[k][ts] for k in ev if ts in ev[k])
        eq*=(1+ret); peak=max(peak,eq); mdd=min(mdd,eq/peak-1); rs.append(ret)
    pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
    print(f"  [FULL] PF={pf:.3f} MAX DD={mdd:.1%}")

print("\n"+"="*70)
report_combined({"MR":mr_champ,"TR":tr_champ,"C":c_champ}, risk_each=0.003, name="COMBINED MULTI-STRATEGY (MR+TREND+UNIFIED)")
# dollar sim $100, $2/trade per sub (total $6 risk when all 3 trade) -> scale to $2 total
eq=100.0
ev={"MR":{t["entry_time"]:t["adj_r"] for t in mr_champ},
    "TR":{t["entry_time"]:t["adj_r"] for t in tr_champ},
    "C":{t["entry_time"]:t["adj_r"] for t in c_champ}}
times=sorted(set().union(*[set(e.keys()) for e in ev.values()]))
for ts in times:
    ret=sum(0.00067*ev[k][ts] for k in ev if ts in ev[k])  # ~$2 total risk split 3 ways
    eq+=ret
print(f"$100 @ ~$2/trade total -> ${eq:,.2f} ({(eq/100-1):+.1%})")
print("[done]")
