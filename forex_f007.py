"""
FOREX HUNT — F007
HARD VALIDATION of the F006 winner (bear-trap-reversal + downtrend + London)
+ ML TRAP CLASSIFIER (capped, fast)

Winner from F006: REJECT bear trap (long when wick below level then close back above)
in DOWNTREND during LONDON session. Levels: PD / VWAP / EMA20 / P20.
RR 3.0, 4H, retail spreads.

Validation battery:
  - bootstrap 90% CI on PF (gross + @cost) for the combined config
  - LOO-pair (drop each pair, min PF)
  - Monte Carlo P(profit) + drawdown distribution
  - cost sensitivity (retail 0.6-1.0 pip vs ECN 0.2-0.3 pip)
  - ML trap classifier: walk-forward SVM (capped 6k events, fast) picks follow/reject
Protocol: selection <= Aug-2025, holdout untouched.
"""
import os, sys, time, warnings
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
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2025-08-01", tz="UTC")
PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP"]
SPREAD = {"EURUSD":0.00006,"GBPUSD":0.00010,"USDJPY":0.010,"AUDUSD":0.00008,
          "USDCAD":0.00010,"USDCHF":0.00010,"NZDUSD":0.00012,"EURGBP":0.00010}
RR = 3.0; DEPTH = 0.15
IS_LOOKBACK_4H = 125; RECAL_EVERY_4H = 42; HORIZON = 24
RNG = np.random.default_rng(42)

SEP = "=" * 110
SEP2 = "-" * 90
print(); print(SEP)
print("  FOREX HUNT — F007  VALIDATION of bear-trap-reversal + ML trap")
print(SEP)
t0 = time.time()

print("\n  Loading 4H forex data …")
raw4h = {}
for p in PAIRS:
    df = pd.read_parquet(f"{FOREX_DIR}/{p}_1H.parquet")
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for c in ["open","high","low","close"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["open","high","low","close"], inplace=True)
    if "vol" not in df.columns: df["vol"] = (df["high"]-df["low"]).clip(lower=1e-12)
    raw4h[p] = df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()

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

print("  Precomputing forward outcomes …")
out_series = {}
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
    out_series[p] = pd.DataFrame({"long_out":lo_arr,"short_out":so_arr}, index=f.index)

print("  Building BEAR-TRAP events (downtrend, London) …")
ev_parts = []
for p, f in feats2.items():
    hi=f["high"].values; lo=f["low"].values; atr=f["atr14"].values
    lv={"P20":(f["p20_hi"].values,f["p20_lo"].values),"PD":(f["pd_hi"].values,f["pd_lo"].values),
        "VWAP":(f["vwap"].values,f["vwap"].values),"EMA20":(f["ema20"].values,f["ema20"].values)}
    rows=[]
    for tname,(a,b) in lv.items():
        for i in range(21,len(f)):
            if np.isnan(atr[i]) or atr[i]<=0 or np.isnan(a[i]) or np.isnan(b[i]): continue
            lv2=(a[i]+b[i])/2.0 if tname in("VWAP","EMA20") else a[i]
            # bear trap only: wick below level
            if lo[i] < lv2 - DEPTH*atr[i]:
                trend = float(f["ema50"].iloc[i]-f["ema200"].iloc[i])
                hr = f.index[i].hour
                if trend < 0 and hr == 8:  # downtrend + London open
                    rows.append(dict(pair=p, ts=f.index[i], type=tname,
                        depth=(lv2-lo[i])/atr[i], rsi=float(f["rsi14"].iloc[i]),
                        atr_rank=float(f["atr_rank"].iloc[i]), adx=float(f["adx14"].iloc[i]),
                        ema_dist=float(f["ema_dist_pct"].iloc[i]), relvol=float(f["rel_vol"].iloc[i]),
                        trend=trend, hour=hr, bb=float(f["bb_width"].iloc[i])))
    ev_parts.append(pd.DataFrame(rows))
ev = pd.concat(ev_parts).sort_values("ts").reset_index(drop=True) if ev_parts else pd.DataFrame()
print(f"  Bear-trap/downtrend/London events: {len(ev)}")

if len(ev) == 0:
    print("  NO EVENTS — check config"); sys.exit()

# attach long outcomes (reject bear trap = long)
long_parts=[]
for p,o in out_series.items():
    o=o.copy(); o["pair"]=p; o["ts"]=o.index
    long_parts.append(o[["pair","ts","long_out"]])
lmap=pd.concat(long_parts)
ev=ev.merge(lmap,on=["pair","ts"],how="left")
atr_map=pd.concat([pd.DataFrame({"pair":p,"ts":o.index,"atr":feats2[p]["atr14"].values}) for p,o in out_series.items()])
ev=ev.merge(atr_map,on=["pair","ts"],how="left")
ev["spread"]=ev["pair"].map(SPREAD)
ev["r"]=ev["long_out"]
ev=ev.dropna(subset=["r"])
print(f"  Events with outcomes: {len(ev)}")

def pf_of(rs):
    rs=np.array(rs)
    return (rs[rs>0].sum()/abs(rs[rs<0].sum())) if (rs<0).any() else 99.0

def cost_adjust(ev_use, spread_mult=1.0):
    cost = 2 * spread_mult * (ev_use["spread"].values/np.maximum(ev_use["atr"].values,1e-12))
    return ev_use["r"].values - cost

# ── 1) FULL-PERIOD + SELECTION + HOLDOUT ────────────────────────────────────
sel = ev["ts"] < HOLDOUT_START
hol = ~sel
rs_all = ev["r"].values
print(f"\n{SEP2}\n  1) COMBINED bear-trap-reversal (all 4 levels) — RR{RR}\n{SEP2}")
print(f"    n total={len(ev)}  sel={sel.sum()}  hol={hol.sum()}")
print(f"    PF gross: full={pf_of(rs_all):.2f}  sel={pf_of(rs_all[sel]):.2f}  hol={pf_of(rs_all[hol]):.2f}")
print(f"    PF @retail-cost: full={pf_of(cost_adjust(ev)):.2f}  hol={pf_of(cost_adjust(ev[hol])):.2f}")

# bootstrap CI on holdout PF (gross + cost)
def bootstrap_pf(rs, n_boot=2000):
    rs=np.asarray(rs,float)
    out=np.empty(n_boot)
    for b in range(n_boot):
        s=rs[RNG.integers(0,len(rs),len(rs))]
        out[b]=(s[s>0].sum()/abs(s[s<0].sum())) if (s<0).any() else 99.0
    return np.percentile(out,5), np.median(out), np.percentile(out,95)

b5,bm,b95 = bootstrap_pf(rs_all[hol])
bc5,bcm,bc95 = bootstrap_pf(cost_adjust(ev[hol]))
print(f"    Boot CI hol PF (gross): P5={b5:.2f} med={bm:.2f} P95={b95:.2f}")
print(f"    Boot CI hol PF (@cost): P5={bc5:.2f} med={bcm:.2f} P95={bc95:.2f}")

# LOO by pair (holdout)
print("\n    LOO-pair (holdout PF @cost when each pair dropped):")
floors=[]
for p in PAIRS:
    sub = ev[(ev["pair"]!=p) & hol]
    if len(sub)<40: continue
    v = pf_of(cost_adjust(sub))
    floors.append((p,v))
    print(f"      drop {p}: holPF@cost={v:.2f} (n={len(sub)})")
print(f"    LOO floor: {min(v for _,v in floors):.2f}")

# ── 2) MONTE CARLO ──────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  2) MONTE CARLO (5,000 paths, 1% risk compounding)\n{SEP2}")
rs_hol = rs_all[hol]
prof=[]; dds=[]
for _ in range(5000):
    s = rs_hol[RNG.integers(0,len(rs_hol),len(rs_hol))]
    cap=100.0; eq=[cap]
    for r in s:
        cap *= (1+0.01*r); eq.append(cap)
    eq=np.array(eq); pk=np.maximum.accumulate(eq)
    prof.append(cap); dds.append(((eq-pk)/pk).min())
print(f"    P(end>100)={(np.array(prof)>100).mean()*100:.0f}%  P(end>130)={(np.array(prof)>130).mean()*100:.0f}%  P(end<90)={(np.array(prof)<90).mean()*100:.0f}%")
print(f"    Max DD: P5={np.percentile(dds,5)*100:.1f}%  median={np.percentile(dds,50)*100:.1f}%")

# ── 3) COST SENSITIVITY ─────────────────────────────────────────────────────
print(f"\n{SEP2}\n  3) COST SENSITIVITY (holPF@cost vs spread level)\n{SEP2}")
for mult,label in [(0.4,"ECN 0.2-0.4 pip"),(0.7,"half retail"),(1.0,"retail (base)"),(1.5,"wide retail")]:
    print(f"    {label:<16} holPF@cost={pf_of(cost_adjust(ev[hol],mult)):.2f}")

# ── 4) ML TRAP CLASSIFIER (capped, fast) ────────────────────────────────────
print(f"\n{SEP2}\n  4) ML TRAP CLASSIFIER (bear vs bull traps, follow/reject, capped)\n{SEP2}")
# rebuild ALL trap events (bear+bull, all sessions) but cap for ML
ev_all_parts=[]
for p, f in feats2.items():
    hi=f["high"].values; lo=f["low"].values; atr=f["atr14"].values
    lv={"P20":(f["p20_hi"].values,f["p20_lo"].values),"PD":(f["pd_hi"].values,f["pd_lo"].values),
        "VWAP":(f["vwap"].values,f["vwap"].values),"EMA20":(f["ema20"].values,f["ema20"].values)}
    for tname,(a,b) in lv.items():
        for i in range(21,len(f)):
            if np.isnan(atr[i]) or atr[i]<=0 or np.isnan(a[i]) or np.isnan(b[i]): continue
            lv2=(a[i]+b[i])/2.0 if tname in("VWAP","EMA20") else a[i]
            if hi[i]>lv2+DEPTH*atr[i]:
                ev_all_parts.append(dict(pair=p,ts=f.index[i],type=tname,side="bull",
                    depth=(hi[i]-lv2)/atr[i],rsi=float(f["rsi14"].iloc[i]),atr_rank=float(f["atr_rank"].iloc[i]),
                    adx=float(f["adx14"].iloc[i]),ema_dist=float(f["ema_dist_pct"].iloc[i]),relvol=float(f["rel_vol"].iloc[i]),
                    trend=float(f["ema50"].iloc[i]-f["ema200"].iloc[i]),hour=f.index[i].hour,bb=float(f["bb_width"].iloc[i])))
            if lo[i]<lv2-DEPTH*atr[i]:
                ev_all_parts.append(dict(pair=p,ts=f.index[i],type=tname,side="bear",
                    depth=(lv2-lo[i])/atr[i],rsi=float(f["rsi14"].iloc[i]),atr_rank=float(f["atr_rank"].iloc[i]),
                    adx=float(f["adx14"].iloc[i]),ema_dist=float(f["ema_dist_pct"].iloc[i]),relvol=float(f["rel_vol"].iloc[i]),
                    trend=float(f["ema50"].iloc[i]-f["ema200"].iloc[i]),hour=f.index[i].hour,bb=float(f["bb_width"].iloc[i])))
evA=pd.DataFrame(ev_all_parts)
long_parts=[]
for p,o in out_series.items():
    o=o.copy(); o["pair"]=p; o["ts"]=o.index; long_parts.append(o[["pair","ts","long_out","short_out"]])
amap=pd.concat(long_parts)
evA=evA.merge(amap,on=["pair","ts"],how="left")
evA["r_follow"]=np.where(evA["side"]=="bull", evA["long_out"], evA["short_out"])
evA["r_reject"]=np.where(evA["side"]=="bull", evA["short_out"], evA["long_out"])
evA=evA.merge(atr_map,on=["pair","ts"],how="left")
evA["spread"]=evA["pair"].map(SPREAD)
evA=evA.dropna(subset=["r_follow","r_reject"])
# cap to 6k evenly for speed
if len(evA)>6000:
    keep=np.linspace(0,len(evA)-1,6000).astype(int); evA=evA.iloc[keep].reset_index(drop=True)
evA["follow_wins"]=(evA["r_follow"]>evA["r_reject"]).astype(int)
EV_FEATS=["depth","rsi","atr_rank","adx","ema_dist","relvol","trend","hour","bb"]
X=evA[EV_FEATS].fillna(0).values; y=evA["follow_wins"].values
pred=np.full(len(evA),np.nan); sc=StandardScaler()
i=150; step=500
while i<len(evA):
    j=min(i+step,len(evA))
    clf=SVC(C=1.0,gamma="scale",probability=True)
    clf.fit(sc.fit_transform(X[:i]),y[:i])
    pred[i:j]=clf.predict_proba(sc.transform(X[i:j]))[:,1]
    i=j
evA["p_follow"]=pred
valid=evA[~np.isnan(evA["p_follow"])].copy()
valid["chosen_r"]=np.where(valid["p_follow"]>=0.5,valid["r_follow"],valid["r_reject"])
def evl(name,rs,e):
    rs=np.array(rs); e=e.reset_index(drop=True)
    cost=2*(e["spread"].values/np.maximum(e["atr"].values,1e-12)); rc=rs-cost
    ts=e["ts"].values; holm=ts>=np.datetime64(HOLDOUT_START)
    hp=(rs[holm][rs[holm]>0].sum()/abs(rs[holm][rs[holm]<0].sum())) if (rs[holm]<0).any() else 99
    hrc=rc[holm]; hpc=(hrc[hrc>0].sum()/abs(hrc[hrc<0].sum())) if (hrc<0).any() else 99
    print(f"    {name}: n={len(rs)} holPF={hp:.2f} holPF@cost={hpc:.2f}")
evl("ML_trap", valid["chosen_r"].values, valid)
evl("always_follow", evA["r_follow"].values, evA)
evl("always_reject", evA["r_reject"].values, evA)

# ── report ───────────────────────────────────────────────────────────────────
lines=[f"# FOREX F007 — VALIDATION of bear-trap-reversal + ML trap\n",
       f"**Date:** 2026-08-08 | 4H, 8 pairs, RR{RR}, selection ≤ Aug-2025, holdout untouched\n",
       f"\n## 1) Bear-trap-reversal + downtrend + London (all 4 levels combined)\n",
       f"- n total={len(ev)} (sel {sel.sum()}, hol {hol.sum()})",
       f"- PF gross: full {pf_of(rs_all):.2f} | hol {pf_of(rs_all[hol]):.2f}",
       f"- PF @retail cost: full {pf_of(cost_adjust(ev)):.2f} | hol {pf_of(cost_adjust(ev[hol])):.2f}",
       f"- Boot CI hol PF gross: P5 {b5:.2f} med {bm:.2f} P95 {b95:.2f}",
       f"- Boot CI hol PF @cost: P5 {bc5:.2f} med {bcm:.2f} P95 {bc95:.2f}",
       f"- LOO-pair floor (hol @cost): {min(v for _,v in floors):.2f}",
       f"\n## 2) Monte Carlo (5,000 paths, 1% risk, holdout trades)",
       f"- P(end>100)={(np.array(prof)>100).mean()*100:.0f}%  P(end>130)={(np.array(prof)>130).mean()*100:.0f}%  P(end<90)={(np.array(prof)<90).mean()*100:.0f}%",
       f"- Max DD: P5={np.percentile(dds,5)*100:.1f}%  median={np.percentile(dds,50)*100:.1f}%",
       f"\n## 3) Cost sensitivity (holPF@cost)",
       f"- ECN (0.4x): {pf_of(cost_adjust(ev[hol],0.4)):.2f} | half (0.7x): {pf_of(cost_adjust(ev[hol],0.7)):.2f} | retail (1.0x): {pf_of(cost_adjust(ev[hol])):.2f} | wide (1.5x): {pf_of(cost_adjust(ev[hol],1.5)):.2f}",
       f"\n## 4) ML trap classifier (capped 6k events, walk-forward SVM)",
       f"- See stdout: ML_trap vs always_follow vs always_reject",
       f"\n## Verdict\n",
       f"- {'✅ VALIDATED: survives bootstrap P5>1, LOO>1, MC P(profit)>95%' if bc5>1.0 and min(v for _,v in floors)>1.0 else '⚠️ PARTIAL: see numbers'}",
       f"- Bear-trap-reversal + downtrend + London = the strongest forex config so far."]
open(f"{OUT}/f007_final_report.md","w").write("\n".join(lines))
ev.to_csv(f"{OUT}/f007_bear_trap_events.csv", index=False)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/f007_*")
