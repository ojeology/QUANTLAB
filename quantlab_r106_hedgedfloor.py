"""
QUANTLAB AI — R106
THE HEDGED-YIELD FLOOR: lend-the-coin + short-the-perp on the same coin.

Who pays: (1) margin borrowers rent the coins they short → lending APR;
(2) leveraged longs pay funding when long-heavy → funding APR to shorts.
The two payer crowds are OPPOSITE (short-sellers borrow coins; longs pay funding),
so the combined yield should be structurally positive across sentiment regimes —
the "always someone paying" property, now measured per coin.

Construction (all close-only, no path): coin in OKX flexible lending + 1× short perp.
Monthly reconstitution: top-12 by trailing-30d combined APR, churn 0.10% per swap
(spread cost of rotating spot+perp, amortized over the month).
Baseline: current R105 floor (USDT lend 3.5% + 24-coin carry 4%).
Output: per-coin combined yields, basket backtest over available history, new floor
$/month, and the updated R105-B projection.
"""
import os, glob, json, time, subprocess
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
UA="Mozilla/5.0"
SEP="═"*110; SEP2="─"*90
COINS = ["BTC","ETH","SOL","XRP","DOGE","BNB","ADA","LINK","LTC","AVAX",
         "DOT","NEAR","APT","SUI","ATOM","FIL","INJ","ARB","OP","ETC"]
CHURN = 0.10  # % per monthly swap (spot+perp round trip, maker-ish)

def get(url, tries=3):
    for a in range(tries):
        p=subprocess.run(["curl","-s","-4","--max-time","20","-A",UA,url],capture_output=True,timeout=30)
        try:
            d=json.loads(p.stdout.decode())
            if d.get("code")=="0": return d["data"]
            time.sleep(1+a)
        except Exception: time.sleep(1+a)
    return []

def pf(x):
    x=np.asarray(x,float); x=x[~np.isnan(x)]
    return x[x>0].sum()/abs(x[x<0].sum()) if len(x) and (x<0).any() else (99.0 if len(x) else np.nan)

print(SEP); print("  R106 — HEDGED-YIELD FLOOR (coin lending + short perp)"); print(SEP)
t0=time.time()

# ── 1) pull lending-rate history per coin ────────────────────────────────────
print("\n  [1] coin lending histories …")
lend={}
os.makedirs("quantlab_cache/lending", exist_ok=True)
for k,c in enumerate(COINS,1):
    f=f"quantlab_cache/lending/{c}.csv"
    if os.path.exists(f):
        d=pd.read_csv(f,parse_dates=["ts"]); lend[c]=d; continue
    rows={}; after=None
    for page in range(40):
        url=f"https://www.okx.com/api/v5/finance/savings/lending-rate-history?ccy={c}&limit=100"
        if after: url+=f"&after={after}"
        data=get(url)
        if not data: break
        for r in data: rows[int(r["ts"])]=float(r["lendingRate"])
        oldest=int(data[-1]["ts"])
        if len(data)<100: break
        after=oldest
        time.sleep(0.05)
    if rows:
        d=pd.DataFrame(sorted(rows.items()),columns=["ts","apr"])
        d["ts"]=pd.to_datetime(d["ts"],unit="ms",utc=True)
        d.to_csv(f,index=False); lend[c]=d
print(f"  coins with lending history: {len(lend)}  ({time.time()-t0:.0f}s)")

# ── 2) funding histories (cached) ────────────────────────────────────────────
fund={}
for f in glob.glob("quantlab_cache/funding/*.csv"):
    c=os.path.basename(f)[:-4]
    d=pd.read_csv(f,parse_dates=["fundingTime"])
    fund[c]=d.rename(columns={"fundingTime":"ts","rate":"funding"})

# ── 3) per-coin combined weekly yield ────────────────────────────────────────
print(f"\n{SEP2}\n  [2] per-coin hedged yield (lending APR/52 + funding weekly, %/wk)\n{SEP2}")
comb={}
for c in COINS:
    if c not in lend or c not in fund: continue
    L=lend[c].copy(); L["wk"]=L["ts"].dt.to_period("W").dt.start_time
    lw=L.groupby("wk")["apr"].mean()/52*100            # %/wk
    F=fund[c].copy(); F["wk"]=F["ts"].dt.to_period("W").dt.start_time
    fw=F.groupby("wk")["funding"].sum()*100            # %/wk (short earns when positive)
    j=pd.concat([lw.rename("lend"),fw.rename("fund")],axis=1).dropna()
    if len(j)<8: continue
    j["tot"]=j["lend"]+j["fund"]
    comb[c]=j
    apr = j["tot"].mean()*52
    posw = (j["tot"]>0).mean()*100
    print(f"  {c:<5} n={len(j):>3}w  lend {j['lend'].mean():.3f} + fund {j['fund'].mean():+.3f} = {j['tot'].mean():+.3f}%/wk  APR {apr:+5.1f}%  posW {posw:.0f}%")
allj=pd.concat({c:j["tot"] for c,j in comb.items()},axis=1)
allj.to_csv("quantlab_output/r106_combined_yields.csv")
print(f"  overlap window: {allj.index.min():%Y-%m-%d} → {allj.index.max():%Y-%m-%d}  ({len(allj)} weeks)")

# ── 4) baskets ───────────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  [3] baskets (equal-weight %/wk)\n{SEP2}")
def stats(w,name):
    w=w.dropna()
    pos=(w>0).mean()*100; worst=w.min()
    eq=(1+w/100).cumprod(); dd=(((eq-eq.cummax())/eq.cummax()).min())*100
    ann=eq.iloc[-1]**(52/len(w))-1
    print(f"  {name:<28} n={len(w):>3}w  mean {w.mean():+.3f}%/wk (APR {w.mean()*52:+.1f}%)  posW {pos:.0f}%  worst {worst:+.3f}%  maxDD {dd:.2f}%")
    return w
base = allj.mean(axis=1)                              # static: all coins
stats(base, "static all-coin basket")
# monthly reconstituted top-12 by trailing 4w mean
weeks=allj.index
holds=[]; cur=[]
for i,ts in enumerate(weeks):
    if i<4: continue
    trail=allj.iloc[max(0,i-4):i].mean()
    if i%4==0 or not cur:
        cur=list(trail.nlargest(12).index)
    holdings=cur
    wk=allj.loc[ts, holdings]
    # churn cost at reconstitution weeks
    if i%4==0 and holds:
        changed=len(set(holdings)^set(holds[-1]))
        wk = wk - CHURN*changed/12
    holds.append(holdings)
    wk.name=ts
    holds[-1]=holdings
    rr=pd.DataFrame([wk])
    try: reconst = pd.concat([reconst, rr])
    except NameError: reconst = rr
rc = reconst.mean(axis=1)
stats(rc, "monthly top-12 reconstituted")
rc.to_csv("quantlab_output/r106_reconstituted_weekly.csv")

# current-floor baseline for comparison
usdt_lend_apr = 3.5
carry_apr = 4.0
print(f"\n  current R105 floor: USDT-lend {usdt_lend_apr}% / carry {carry_apr}% (blended floor ≈ {(usdt_lend_apr+carry_apr)/2:.1f}% on 80% of capital)")

# ── 5) updated stack projection ──────────────────────────────────────────────
print(f"\n{SEP2}\n  [4] UPDATED R105-B with hedged-yield floor (replace lend+carry legs)\n{SEP2}")
ch=pd.read_csv("quantlab_output/r100_champion_trades.csv",parse_dates=["ts"])
ch["w"]=ch["ts"].dt.to_period("W").dt.start_time
cham=(ch.groupby("w")["r_net"].sum()*0.15).dropna()
D=pd.read_csv("quantlab_output/r104_cascade_trades.csv",parse_dates=["ts"])
cas=(D.set_index(pd.to_datetime(D["ts"]).dt.tz_localize(None))["r_net"]
      .groupby(pd.Grouper(freq="W")).sum()/3.0*0.15).dropna()
rci = rc.copy(); rci.index = pd.to_datetime(rci.index)
idx = rci.index
cham.index = pd.to_datetime(cham.index)
cas.index = pd.to_datetime(cas.index)
S = pd.DataFrame(index=idx)
S["floor"]=rc.reindex(idx)
S["champ"]=cham.reindex(idx).fillna(0.0)
S["casc"]=cas.reindex(idx).fillna(0.0)
S["tot"]=S["floor"]+S["champ"]+S["casc"]
w=S["tot"].dropna()
pos=(w>0).mean()*100; worst=w.min()
eq=(1+w/100).cumprod(); dd=(((eq-eq.cummax())/eq.cummax()).min())*100
days=max((w.index.max()-w.index.min()).days,1); ann=eq.iloc[-1]**(365.0/days)-1
mp=(w.groupby(w.index.to_period("M")).sum()>0).mean()*100
gate=(pos>=65) and (worst>=-2.0) and (dd>=-12.0)
print(f"  STACK v2 (floor+champ@0.15+cascade@0.15): n={len(w)}w  posW={pos:.0f}%  worstW={worst:+.2f}%  maxDD={dd:.2f}%  ann={ann*100:+.1f}%  posM={mp:.0f}%  {'✅GATES' if gate else '❌'}")
months=(w.index.max()-w.index.min()).days/30.4; tot=eq.iloc[-1]-1
for cap in [500,1000,2000]:
    print(f"  ${cap}: ≈ ${tot*cap:+.0f}/{months:.0f}mo ≈ ${tot*cap/months:+.2f}/mo  (floor alone ≈ ${cap*rc.mean()/100*4.33:.2f}/mo)")
S.to_csv("quantlab_output/r106_stack_v2_weekly.csv")
print(f"\n  Done in {time.time()-t0:.0f}s → quantlab_output/r106_*")
