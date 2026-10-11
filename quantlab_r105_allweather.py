"""
QUANTLAB AI — R105
ALL-WEATHER ALLOCATION: stack every validated premium into one equity curve.

Streams (all fundamentals-grounded):
  S1 USDT LENDING  (contractual): OKX flexible lend, paid hourly for supplying
     liquidity to leveraged borrowers. Hourly history pulled from OKX.
  S2 FUNDING CARRY (contractual): 24-coin hedged basket (R102). Measured Jun-2026→;
     pre-window filled with the measured mean (regime assumption, disclosed) and
     a 0-fill conservative variant also reported.
  S3 CASCADE SLEEVE (forced flows, R104): sleeve = w × the validated weekly
     cascade series (w = fraction of capital running the 3-slot fade).
  S4 CHAMPION SLEEVE (statistical, FROZEN config): risk c% per trade, R100 log.

Allocation variants (prespecified small grid — weights chosen a priori, reported all):
  A income-heavy : lend 60% / carry 40% | champ 0.10% | cascade 0.10×
  B balanced     : lend 40% / carry 40% | champ 0.15% | cascade 0.15×
  C carry-max    : lend 0%  / carry 100%| champ 0.10% | cascade 0.10×

Window: 2025-06-08 → 2026-10-09 (champion life). Gates: posW≥65% · worstW≥−2% ·
maxDD≤12%. Conservative run = carry pre-window zero-filled; regime run = mean-filled.
"""
import os, sys, glob, json, time, subprocess, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)
UA="Mozilla/5.0"
SEP="═"*110; SEP2="─"*90
START = pd.Timestamp("2025-06-08")

def get(url, tries=3):
    for a in range(tries):
        p=subprocess.run(["curl","-s","-4","--max-time","20","-A",UA,url],capture_output=True,timeout=30)
        try:
            d=json.loads(p.stdout.decode())
            if d.get("code")=="0": return d["data"]
            time.sleep(1+a)
        except Exception: time.sleep(1+a)
    return []

def consistency(w, name, pos_gate=65, worst_gate=-2.0, dd_gate=-12.0):
    w=w.dropna()
    pos=(w>0).mean()*100; worst=w.min()*100
    ls=cur=0
    for v in w: cur=cur+1 if v<=0 else 0; ls=max(ls,cur)
    eq=(1+w/100).cumprod(); dd=(((eq-eq.cummax())/eq.cummax()).min())*100
    days=max((w.index.max()-w.index.min()).days,1)
    ann=eq.iloc[-1]**(365.0/days)-1
    mp=(w.groupby(w.index.to_period("M")).sum()>0).mean()*100
    gate=(pos>=pos_gate) and (worst>=worst_gate) and (dd>=dd_gate)
    print(f"  {name:<34} posW={pos:5.1f}%  worstW={worst:+6.2f}%  streak={ls:>2}w  maxDD={dd:6.2f}%  ann={ann*100:+6.2f}%  posM={mp:.0f}%  {'✅GATES' if gate else '❌'}")
    return w, gate

print(SEP); print("  R105 — ALL-WEATHER ALLOCATION (lend + carry + cascade + champion)"); print(SEP)
t0=time.time()

# ── S1: USDT lending hourly history ──────────────────────────────────────────
print("\n  [1] USDT lending rate history …")
rows={}; after=None
for page in range(80):
    url="https://www.okx.com/api/v5/finance/savings/lending-rate-history?ccy=USDT&limit=100"
    if after: url+=f"&after={after}"
    d=get(url)
    if not d: break
    for r in d:
        rows[int(r["ts"])]=float(r["lendingRate"])
    oldest=int(d[-1]["ts"])
    if len(d)<100: break
    after=oldest
    time.sleep(0.05)
L=pd.DataFrame(sorted(rows.items()),columns=["ts","rate"])
L["ts"]=pd.to_datetime(L["ts"],unit="ms",utc=True)
print(f"  lending epochs: {len(L)}  {L['ts'].min():%Y-%m-%d} → {L['ts'].max():%Y-%m-%d}  avg APR {L['rate'].mean()*24*365*100:.1f}%")
Lw=(L.set_index("ts")["rate"].tz_localize(None).groupby(pd.Grouper(freq="W")).mean()*100/52).rename("lend")  # rate is APR fraction -> %/wk
lend_mean=L["rate"].mean()*100/52

# ── S2: carry ────────────────────────────────────────────────────────────────
print("  [2] funding carry …")
A=pd.concat([pd.read_csv(f,parse_dates=["fundingTime"]).assign(sym=os.path.basename(f)[:-4])
             for f in glob.glob("quantlab_cache/funding/*.csv")])
port=A.groupby("fundingTime")["rate"].mean()*100
cw=port.groupby(port.index.to_period("W")).sum()
cw.index=cw.index.start_time   # already naive
carry_mean=float(cw.mean())

# ── S3: cascade weekly (validated R104 series) ───────────────────────────────
D=pd.read_csv("quantlab_output/r104_cascade_trades.csv",parse_dates=["ts"])
cas=D.set_index(pd.to_datetime(D["ts"]).dt.tz_localize(None))["r_net"].groupby(pd.Grouper(freq="W")).sum()/3.0   # %/wk at w=1.0

# ── S4: champion weekly (R100 log) ───────────────────────────────────────────
ch=pd.read_csv("quantlab_output/r100_champion_trades.csv",parse_dates=["ts"])
ch["w"]=ch["ts"].dt.to_period("W").dt.start_time
cham=ch.groupby("w")["r_net"].sum()   # in R

idx=pd.date_range(pd.Timestamp(START.tz_localize(None).date()), pd.Timestamp("2026-10-09"), freq="W")
df=pd.DataFrame(index=idx)
df["lend"]=Lw.reindex(idx)
df["carry"]=cw.reindex(idx)
df["cham_R"]=cham.reindex(idx).fillna(0.0)
df["cas"]=cas.reindex(idx).fillna(0.0)
covered_lend = df["lend"].notna().mean()*100
covered_carry = df["carry"].notna().mean()*100
print(f"  window weeks: {len(df)} | lend measured {covered_lend:.0f}% | carry measured {covered_carry:.0f}%")

VARIANTS = {
 "A income-heavy": dict(lend=0.60, carry=0.40, champ=0.10, casc=0.10),
 "B balanced":     dict(lend=0.40, carry=0.40, champ=0.15, casc=0.15),
 "C carry-max":    dict(lend=0.00, carry=1.00, champ=0.10, casc=0.10),
}
print(f"\n{SEP2}\n  STACKS — conservative (pre-measurement carry/lend = 0)\n{SEP2}")
res={}
for nm,v in VARIANTS.items():
    s = (v["lend"]*df["lend"].fillna(0.0)
       + v["carry"]*df["carry"].fillna(0.0)
       + v["champ"]*df["cham_R"]
       + v["casc"]*df["cas"])
    _,g = consistency(s, f"{nm} (conservative)")
    res[nm]=dict(series=s, gate=g)

print(f"\n{SEP2}\n  STACKS — regime-filled (pre-measurement carry = measured mean {carry_mean:+.3f}%/wk, lend = mean {lend_mean:.3f}%/wk)\n{SEP2}")
best=None
for nm,v in VARIANTS.items():
    s = (v["lend"]*df["lend"].fillna(lend_mean)
       + v["carry"]*df["carry"].fillna(carry_mean)
       + v["champ"]*df["cham_R"]
       + v["casc"]*df["cas"])
    w,g = consistency(s, f"{nm} (regime-filled)")
    res[nm].update(series_regime=w, gate_regime=g)
    if g and best is None: best=nm

# monthly table for the flagship (B balanced, regime)
sB=res["B balanced"]["series_regime"]
print(f"\n{SEP2}\n  MONTHLY — B balanced (regime-filled)\n{SEP2}")
mm=sB.groupby(sB.index.to_period("M")).sum()
for k,v in mm.items(): print(f"    {k}: {v:+.2f}%")
print(f"    positive months: {(mm>0).sum()}/{len(mm)}  worst: {mm.min():+.2f}%")

# MC on flagship
vals=sB.dropna().values
prof=[]
for _ in range(3000):
    x=vals[RNG.integers(0,len(vals),len(vals))]
    eq=1.0
    for v in x: eq*=(1+v/100)
    prof.append(eq)
prof=np.array(prof)
print(f"  MC (flagship): median={np.median(prof):.2f}x  P(loss)={(prof<1).mean()*100:.0f}%  P5={np.percentile(prof,5):.2f}x  P95={np.percentile(prof,95):.2f}x")

months=(sB.index.max()-sB.index.min()).days/30.4
for cap in [500,1000,2000]:
    tot=(1+sB.dropna()/100).prod()-1
    print(f"  ${cap}: ≈ ${tot*cap:+.0f} / {months:.0f}mo ≈ ${tot*cap/months:+.1f}/mo")

print(f"\n  best gate-passer: {best}")
sB.to_csv("quantlab_output/r105_flagship_weekly.csv")
for nm,v in res.items(): v["series"].to_csv(f"quantlab_output/r105_{nm.split()[0].lower()}_cons.csv")
print(f"\n  Done in {time.time()-t0:.0f}s → quantlab_output/r105_*")
