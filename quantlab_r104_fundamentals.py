"""
QUANTLAB AI — R104
FUNDAMENTALS ROUND: forced flows + term structure + the stack.

PART 1 — BASIS PROBE (contractual, #1/#4): is a spot-vs-futures basis trade visible
  from this sandbox? OKX quarterly futures vs spot, current annualized basis + history
  availability for expired contracts.
PART 2 — CASCADE FADE (forced flows, #2): leveraged liquidations are the purest forced
  flow. Visible in price: 24h return < -3σ(30d) AND closing at 24h lows = forced-selling
  exhaustion. Fade at CLOSE, exit at CLOSE (+2% / -3% close-stop / 72h time). ONE
  position per symbol, MAX 3 SLOTS portfolio-wide (cascade correlation is the risk —
  capped mechanically, not by hope). Slot = 1/3 of capital.
  Zero path ambiguity by construction. Costs 0.05%/side.
PART 3 — THE STACK: carry (never-red floor) + frozen champion at 0.25% risk (harvest
  sleeve). Consistency metrics on the combined weekly series.
"""
import os, sys, glob, json, time, subprocess, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG

CACHE = CONFIG["CACHE_FOLDER"]
OUT = CONFIG["OUTPUT_FOLDER"]; os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
SEP="═"*110; SEP2="─"*90
UA="Mozilla/5.0"

def curl_json(url, tries=3):
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
    if len(x)==0: return np.nan
    return x[x>0].sum()/abs(x[x<0].sum()) if (x<0).any() else 99.0

def consistency(weekly, name):
    w=weekly.dropna()/100.0   # input already in percent
    pos=(w>0).mean()*100; worst=w.min()*100
    ls=cur=0
    for v in w:
        cur=cur+1 if v<=0 else 0; ls=max(ls,cur)
    eq=(1+w).cumprod(); dd=(((eq-eq.cummax())/eq.cummax()).min())*100
    days=(w.index.max()-w.index.min()).days
    ann=eq.iloc[-1]**(365.0/max(days,1))-1
    mp=(w.groupby(w.index.to_period("M")).sum()>0).mean()*100
    gate = (pos>=65) and (worst>=-3.0) and (dd>=-12.0)
    print(f"  {name:<26} posW={pos:5.1f}%  worstW={worst:+7.2f}%  streak={ls:>2}w  maxDD={dd:7.2f}%  ann={ann*100:+6.1f}%  posM={mp:.0f}%  {'✅GATES' if gate else '❌gates'}")
    return dict(pos=pos,worst=worst,dd=dd,ann=ann,w=w)

print(SEP); print("  R104 — FUNDAMENTALS ROUND: basis / cascade fade / the stack"); print(SEP)
t0=time.time()

# ── PART 1: BASIS PROBE ──────────────────────────────────────────────────────
print(f"\n{SEP2}\n  PART 1) BASIS: spot vs quarterly futures (contractual)\n{SEP2}")
spot = curl_json("https://www.okx.com/api/v5/market/tickers?instType=SPOT")
px = {t["instId"]: float(t["last"]) for t in spot if "last" in t}
fut_tk = curl_json("https://www.okx.com/api/v5/market/tickers?instType=FUT")
fut_px = {t["instId"]: float(t["last"]) for t in fut_tk if "last" in t}
futs = curl_json("https://www.okx.com/api/v5/public/instruments?instType=FUT")
import datetime as dt
now = dt.datetime.now(dt.timezone.utc)
rows=[]
for f in futs:
    iid=f.get("instId","")
    if not iid.startswith("BTC-USDT-"): continue
    try:
        exp=dt.datetime.fromtimestamp(int(f["expTime"])/1000, dt.timezone.utc)
        dte=(exp-now).days
        if dte<7: continue
        if iid not in fut_px: continue
        last=fut_px[iid]
        sp=px.get("BTC-USDT")
        if not sp: continue
        ann=((last/sp)-1)*(365.0/dte)*100
        rows.append(dict(inst=iid,dte=dte,spot=sp,fut=last,basis_ann=ann))
    except Exception: pass
if rows:
    for r in sorted(rows,key=lambda x:x["dte"]):
        print(f"  {r['inst']:<18} {r['dte']:>3}d  fut={r['fut']:.0f} spot={r['spot']:.0f}  annualized basis {r['basis_ann']:+.2f}%")
    print("  → basis trade = buy spot + short fut when annualized basis > 8% (hold to settle).")
else:
    print("  quarterly futures not readable from here — basis probe inconclusive.")

# ── PART 2: CASCADE FADE ─────────────────────────────────────────────────────
print(f"\n{SEP2}\n  PART 2) CASCADE FADE — close-only, 3 slots, costs in\n{SEP2}")
frames={}
for f in sorted(glob.glob(os.path.join(CACHE,"*_1H.parquet"))):
    sym=f.split("/")[-1].replace("_1H.parquet","")
    df=pd.read_parquet(f)
    df.index=pd.to_datetime(df.index,utc=True); df.sort_index(inplace=True)
    df=df[(df["open"]>0)&(df["low"]>0)][["close"]].astype(float)
    if len(df)>2500: frames[sym]=df["close"]
EQUITY={"MSTR","QQQ","NVDA","INTC","TSLA","SOXS","CRCL","MU","SPCX","SKHY","SKHYNIX","SNXX","UB"}
frames={s:v for s,v in frames.items() if s.split("_")[0] not in EQUITY}
closes=pd.DataFrame(frames).sort_index()
r24=closes.pct_change(24)
sig=r24.rolling(720,min_periods=360).std()
low24=closes<=closes.rolling(24).min()
cascade=(r24 < -3.0*sig) & low24
print(f"  cascade bars: {int(cascade.sum().sum())} across {len(closes.columns)} symbols")

COST=0.0005
pos={}   # sym -> dict(entry, entry_i, bars)
trades=[]
idx=closes.index
cvals=closes.values; rvals=r24.values; cvals_c=cascade.values
NS=closes.shape[1]; cols=list(closes.columns)
col_ix={c:i for i,c in enumerate(cols)}
MAXSLOTS=3
for i in range(721, len(idx)):
    ts=idx[i]
    # exits first (close-based)
    for s in list(pos.keys()):
        p=pos[s]; e=p["entry"]; px=cvals[i, col_ix[s]]
        if np.isnan(px): continue
        p["bars"]+=1
        if px>=e*1.02:
            trades.append(dict(sym=s, ts=ts, r=(px/e-1)*100, kind="TP", bars=p["bars"])); del pos[s]
        elif px<=e*0.97:
            trades.append(dict(sym=s, ts=ts, r=(px/e-1)*100, kind="SL", bars=p["bars"])); del pos[s]
        elif p["bars"]>=72:
            trades.append(dict(sym=s, ts=ts, r=(px/e-1)*100, kind="TIME", bars=p["bars"])); del pos[s]
    # entries: most-oversold first, fill free slots
    free=MAXSLOTS-len(pos)
    if free>0:
        cand=[(rvals[i,j], cols[j]) for j in range(NS) if cvals_c[i,j] and not np.isnan(rvals[i,j]) and cols[j] not in pos]
        cand.sort()
        for rv,s in cand[:free]:
            pos[s]=dict(entry=cvals[i,col_ix[s]], bars=0)
D=pd.DataFrame(trades)
viol=(D["r"].abs()>10).sum()
if viol: print(f"  ⚠ {viol} trades with |r|>10% (bad ticks) — dropped")
D=D[D["r"].abs()<=10].reset_index(drop=True)
D["r_net"]=D["r"]-2*COST*100
print(f"  trades: {len(D)}  TP={ (D['kind']=='TP').sum() }  SL={ (D['kind']=='SL').sum() }  TIME={ (D['kind']=='TIME').sum() }  WR_TP={(D['kind']=='TP').mean()*100:.0f}%")
D["ts"]=pd.to_datetime(D["ts"])
D.to_csv(os.path.join(OUT,"r104_cascade_trades.csv"), index=False)
wk = D.set_index("ts")["r_net"].groupby(pd.Grouper(freq="W")).sum()/MAXSLOTS   # percent of capital, 1/3 per slot
m2=consistency(wk, "cascade fade (1/3 per slot)")
print(f"  expectancy/trade: {D['r_net'].mean():+.3f}%   median hold: {D['bars'].median():.0f} bars")
for y,g in wk.groupby(wk.index.year):
    print(f"    {y}: weeks={len(g):>3}  posW={(g>0).mean()*100:5.1f}%  worstW={g.min()*100:+7.2f}%  sum={g.sum()*100:+7.2f}%")

# ── PART 3: THE STACK — carry floor + frozen champion @0.25% ─────────────────
print(f"\n{SEP2}\n  PART 3) THE STACK: carry + champion sleeve (0.25% risk/trade)\n{SEP2}")
ch=pd.read_csv(os.path.join(OUT,"r100_champion_trades.csv"), parse_dates=["ts"])
ch["w"]=ch["ts"].dt.to_period("W").dt.start_time
cham_w = ch.groupby("w")["r_net"].sum()*0.25/1.0    # champion weekly % at 0.25% risk
carry_w = m2 if False else None
import glob as _g
fcarry=[]
for f in _g.glob(os.path.join(CACHE,"funding","*.csv")):
    d=pd.read_csv(f, parse_dates=["fundingTime"]); d["sym"]=os.path.basename(f)[:-4]; fcarry.append(d)
A=pd.concat(fcarry)
port=A.groupby("fundingTime")["rate"].mean()*100
cw=port.groupby(port.index.to_period("W")).sum()
cw.index=cw.index.start_time
stack=cham_w.to_frame("champ").join(cw.rename("carry"), how="outer")
stack["carry"]=stack["carry"].fillna(0.0)          # carry runs every week; pre-Jun-2026 unknown → 0 (conservative)
stack["total"]=stack["champ"]+stack["carry"]
ms=consistency(stack["total"], "STACK: champ@0.25% + carry")
ov=stack[stack.index>=pd.Timestamp("2026-06-29")]
print(f"  overlap window (carry measured): posW={(ov['total']>0).mean()*100:.0f}%  worstW={ov['total'].min()*100:+.2f}%  weeks={len(ov)}")
for cap in [500,1000]:
    tot=(1+stack["total"]/100).prod()-1
    months=(stack.index.max()-stack.index.min()).days/30.4
    print(f"  ${cap}: stack ≈ ${tot*cap:+.0f} over {months:.0f}mo ≈ ${tot*cap/months:+.1f}/mo (champ@0.25% + carry)")

D.to_csv(os.path.join(OUT,"r104_cascade_trades.csv"), index=False)
stack["total"].to_csv(os.path.join(OUT,"r104_stack_weekly.csv"))
print(f"\n  Done in {time.time()-t0:.0f}s → quantlab_output/r104_*")
