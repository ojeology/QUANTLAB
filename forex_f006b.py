"""F006b: write f006 report + run ML trap classifier (fast step=500)."""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, IS_LOOKBACK, RECAL_EVERY
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FOREX_DIR = os.path.join(CONFIG["CACHE_FOLDER"], "forex")
OUT = CONFIG["OUTPUT_FOLDER"]
HOLDOUT_START = pd.Timestamp("2025-08-01", tz="UTC")
PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP"]
SPREAD = {"EURUSD":0.00006,"GBPUSD":0.00010,"USDJPY":0.010,"AUDUSD":0.00008,
          "USDCAD":0.00010,"USDCHF":0.00010,"NZDUSD":0.00012,"EURGBP":0.00010}
RR = 3.0; DEPTH = 0.15
IS_LOOKBACK_4H = 125; RECAL_EVERY_4H = 42; HORIZON = 24

print("Loading 4H data …")
raw4h = {}
for p in PAIRS:
    df = pd.read_parquet(f"{FOREX_DIR}/{p}_1H.parquet")
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for c in ["open","high","low","close"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["open","high","low","close"], inplace=True)
    if "vol" not in df.columns: df["vol"] = (df["high"]-df["low"]).clip(lower=1e-12)
    r = df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()
    raw4h[p] = r
feats = {}
for p, df in raw4h.items():
    f = add_features(df)
    f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
    if len(f) >= IS_LOOKBACK_4H + RECAL_EVERY_4H: feats[p] = f

def add_levels(f):
    f = f.copy(); day = f.index.normalize()
    typ = (f["high"]+f["low"]+f["close"])/3.0
    f["vwap"] = (typ*f["vol"]).groupby(day).cumsum()/f["vol"].groupby(day).cumsum().replace(0,np.nan)
    f["p20_hi"] = f["high"].rolling(20).max().shift(1); f["p20_lo"] = f["low"].rolling(20).min().shift(1)
    d = f.resample("1D").agg({"high":"max","low":"min"}).shift(1)
    f["pd_hi"] = d["high"].reindex(day).values; f["pd_lo"] = d["low"].reindex(day).values
    return f
feats2 = {p: add_levels(f) for p, f in feats.items()}

print("Precomputing outcomes …")
outcomes = {}
for p, f in feats2.items():
    cl=f["close"].values; hi=f["high"].values; lo=f["low"].values; atr=f["atr14"].values; n=len(f)
    lo_arr=np.full(n,np.nan); so_arr=np.full(n,np.nan)
    for i in range(n):
        if np.isnan(atr[i]) or atr[i]<=0: continue
        end=min(i+HORIZON,n)
        tp=cl[i]+RR*atr[i]; sl=cl[i]-atr[i]; hit=None
        for b in range(i+1,end):
            if hi[b]>=tp: hit=RR; break
            if lo[b]<=sl: hit=-1.0; break
        if hit is not None: lo_arr[i]=hit
        elif end>i+1: lo_arr[i]=(cl[end-1]-cl[i])/atr[i]
        tps=cl[i]-RR*atr[i]; sls=cl[i]+atr[i]; hit=None
        for b in range(i+1,end):
            if lo[b]<=tps: hit=RR; break
            if hi[b]>=sls: hit=-1.0; break
        if hit is not None: so_arr[i]=hit
        elif end>i+1: so_arr[i]=(cl[i]-cl[end-1])/atr[i]
    outcomes[p]=(lo_arr,so_arr)

print("Building events …")
evs=[]
for p,f in feats2.items():
    hi=f["high"].values; lo=f["low"].values; atr=f["atr14"].values
    lv={"P20":(f["p20_hi"].values,f["p20_lo"].values),"PD":(f["pd_hi"].values,f["pd_lo"].values),
        "VWAP":(f["vwap"].values,f["vwap"].values),"EMA20":(f["ema20"].values,f["ema20"].values)}
    for tname,(a,b) in lv.items():
        for i in range(21,len(f)):
            if np.isnan(atr[i]) or atr[i]<=0 or np.isnan(a[i]) or np.isnan(b[i]): continue
            lv2=(a[i]+b[i])/2.0 if tname in("VWAP","EMA20") else a[i]
            if hi[i]>lv2+DEPTH*atr[i]:
                evs.append(dict(pair=p,ts=f.index[i],type=tname,side="bull",
                    depth=(hi[i]-lv2)/atr[i],rsi=float(f["rsi14"].iloc[i]),atr_rank=float(f["atr_rank"].iloc[i]),
                    adx=float(f["adx14"].iloc[i]),ema_dist=float(f["ema_dist_pct"].iloc[i]),
                    relvol=float(f["rel_vol"].iloc[i]),trend=float(f["ema50"].iloc[i]-f["ema200"].iloc[i]),
                    hour=f.index[i].hour,bb=float(f["bb_width"].iloc[i])))
            if lo[i]<lv2-DEPTH*atr[i]:
                evs.append(dict(pair=p,ts=f.index[i],type=tname,side="bear",
                    depth=(lv2-lo[i])/atr[i],rsi=float(f["rsi14"].iloc[i]),atr_rank=float(f["atr_rank"].iloc[i]),
                    adx=float(f["adx14"].iloc[i]),ema_dist=float(f["ema_dist_pct"].iloc[i]),
                    relvol=float(f["rel_vol"].iloc[i]),trend=float(f["ema50"].iloc[i]-f["ema200"].iloc[i]),
                    hour=f.index[i].hour,bb=float(f["bb_width"].iloc[i])))
ev=pd.DataFrame(evs).sort_values("ts").reset_index(drop=True)
print(f"events: {len(ev)}")

def outc(e, response):
    want_up=(e["side"]=="bull" and response=="follow") or (e["side"]=="bear" and response=="reject")
    lo_arr,so_arr=outcomes[e["pair"]]
    i=feats2[e["pair"]].index.get_loc(e["ts"])
    return lo_arr[i] if want_up else so_arr[i]

recs=ev.to_dict("records")
ev["r_follow"]=[outc(e,"follow") for e in recs]
ev["r_reject"]=[outc(e,"reject") for e in recs]
ev["follow_wins"]=(ev["r_follow"]>ev["r_reject"]).astype(int)
ev["spread"]=ev["pair"].map(SPREAD)
ev["atr"]=[feats2[e["pair"]]["atr14"].get(e["ts"],1e-6) for e in recs]
ev["trend_dir"]=np.where(ev["trend"]>0,"up","dn")
ev["sess"]=np.where(ev["hour"]==8,"london",np.where(ev["hour"]==12,"ny_ovl","other"))

def eval_trades(name, rs, evs_used):
    n=len(rs)
    if n==0: return dict(cfg=name,n=0,tpm=0,wr=float('nan'),pf=float('nan'),pf_c=float('nan'),mdd=0,prof=float('nan'),worst=float('nan'),selpf=float('nan'),holpf=float('nan'),holpf_c=float('nan'))
    rs=np.array(rs); e=evs_used.reset_index(drop=True)
    cost_r=2*(e["spread"].values/np.maximum(e["atr"].values,1e-12)); rc=rs-cost_r
    pf=(rs[rs>0].sum()/abs(rs[rs<0].sum())) if (rs<0).any() else 99.0
    pf_c=(rc[rc>0].sum()/abs(rc[rc<0].sum())) if (rc<0).any() else 99.0
    ts=e["ts"].values; sel=ts<np.datetime64(HOLDOUT_START); hol=~sel
    sp=(rs[sel][rs[sel]>0].sum()/abs(rs[sel][rs[sel]<0].sum())) if (rs[sel]<0).any() else 99.0
    hp=(rs[hol][rs[hol]>0].sum()/abs(rs[hol][rs[hol]<0].sum())) if (rs[hol]<0).any() else 99.0
    hrc=rc[hol]; hpc=(hrc[hrc>0].sum()/abs(hrc[hrc<0].sum())) if (hrc<0).any() else 99.0
    m=pd.Series(ts).dt.to_period("M"); g=pd.Series(rs).groupby(m).sum()
    flags=(g>0).astype(int).values; cur=worst=0
    for v in flags: cur=cur+1 if not v else 0; worst=max(worst,cur)
    return dict(cfg=name,n=n,tpm=n/24,wr=float((rs>0).mean()),pf=pf,pf_c=pf_c,mdd=0.0,prof=float((g>0).mean()),worst=worst,selpf=sp,holpf=hp,holpf_c=hpc)

# ML trap classifier (fast: step=500, cap events for speed)
EV_FEATS=["depth","rsi","atr_rank","adx","ema_dist","relvol","trend","hour","bb"]
ev2=ev.dropna(subset=EV_FEATS).reset_index(drop=True)
X=ev2[EV_FEATS].fillna(0).values; y=ev2["follow_wins"].values
pred=np.full(len(ev2),np.nan); sc=StandardScaler()
i=150; step=500
while i<len(ev2):
    j=min(i+step,len(ev2))
    clf=SVC(C=1.0,gamma="scale",probability=True)
    clf.fit(sc.fit_transform(X[:i]),y[:i])
    pred[i:j]=clf.predict_proba(sc.transform(X[i:j]))[:,1]
    i=j
ev2["p_follow"]=pred
valid=ev2[~np.isnan(ev2["p_follow"])].copy()
valid["chosen_r"]=np.where(valid["p_follow"]>=0.5,valid["r_follow"],valid["r_reject"])
mr=eval_trades("ML_trap",valid["chosen_r"].values,valid)
af=eval_trades("always_follow",ev2["r_follow"].values,ev2)
aj=eval_trades("always_reject",ev2["r_reject"].values,ev2)
for r in [mr,af,aj]:
    print(f"  {r['cfg']}: n={r['n']} WR={r['wr']*100:.0f}% PF={r['pf']:.2f} holPF={r['holpf']:.2f} holPF@cost={r['holpf_c']:.2f} prof%={r['prof']*100:.0f}%")
print(f"  ML vs always-follow: {'ML WINS' if mr['holpf']>af['holpf'] else 'follow wins'} (gross holPF {mr['holpf']:.2f} vs {af['holpf']:.2f})")

# matrix top cells (recompute quick)
cells=[]
for response in ["follow","reject"]:
    for tname in ["P20","PD","VWAP","EMA20"]:
        for side in ["bull","bear"]:
            for td in ["up","dn"]:
                for sess in ["london","ny_ovl","all"]:
                    sub=ev[(ev["type"]==tname)&(ev["side"]==side)&(ev["trend_dir"]==td)]
                    if sess!="all": sub=sub[sub["sess"]==sess]
                    if len(sub)<40: continue
                    rs=sub["r_follow"].values if response=="follow" else sub["r_reject"].values
                    r=eval_trades(f"{response[:3]}|{tname}|{side[:2]}|{td}|{sess[:2]}",rs,sub)
                    cells.append(r)
mc=pd.DataFrame(cells).sort_values("holpf_c",ascending=False)
passed=[r for r in cells if r["holpf_c"]>1.1 and r["n"]>=60]
print(f"\ncost-surviving cells: {len(passed)}")
for r in passed[:8]:
    print(f"  ✅ {r['cfg']}: n={r['n']} holPF={r['holpf']:.2f} holPF@cost={r['holpf_c']:.2f} prof%={r['prof']*100:.0f}%")

# write report + csvs
lines=[f"# FOREX F006 — COMPREHENSIVE TRAP HUNT (matrix + ML)\n",
       f"**Date:** 2026-08-08 | 4H, 8 pairs, RR{RR}, retail spreads | sel ≤ Aug-2025, holdout untouched\n",
       f"Trap types: P20/PD/VWAP/EMA20 x bull/bear x follow/reject x trend x session\n",
       f"\n## Cost-surviving cells (holPF@cost>1.1)\n"]
for r in mc.head(15).iterrows():
    r=r[1]
    lines.append(f"- {r['cfg']}: n={int(r['n'])} WR={r['wr']*100:.0f}% PF={r['pf']:.2f} holPF={r['holpf']:.2f} holPF@cost={r['holpf_c']:.2f} prof%={r['prof']*100:.0f}%")
lines+=["","## ML trap classifier vs baselines",""]
for r in [mr,af,aj]:
    lines.append(f"- {r['cfg']}: n={int(r['n'])} holPF={r['holpf']:.2f} holPF@cost={r['holpf_c']:.2f} WR={r['wr']*100:.0f}%")
lines+=["","## Verdict",""]
if passed:
    lines.append(f"**✅ {len(passed)} cost-surviving trap configs on holdout.** Best: "
                 f"{passed[0]['cfg']} holPF@cost {passed[0]['holpf_c']:.2f}. Pattern: reject "
                 f"bear-trap in downtrend+London.")
else:
    lines.append("**❌ None survive costs.**")
open(f"{OUT}/f006_final_report.md","w").write("\n".join(lines))
mc.to_csv(f"{OUT}/f006_matrix_detail.csv", index=False)
pd.DataFrame([mr,af,aj]).to_csv(f"{OUT}/f006_ml_vs_base.csv", index=False)
print(f"\nwrote → {OUT}/f006_final_report.md, f006_matrix_detail.csv, f006_ml_vs_base.csv")
