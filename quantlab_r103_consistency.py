"""
QUANTLAB AI — R103
CONSISTENCY HUNT, ROUND 1: C2 low-vol basket + C3 dip-ladder (C1 carry = anchor).

LAW OF THIS HUNT: fills at bar CLOSES only. No intra-bar anything. The backtest
cannot lie about path because there is no path — every fill is a printed close.

Candidates (windows include the 2024 bad regime on purpose):
  V1 LOW-VOL BASKET (spot, weekly rebalance, 6 slots):
     each week pick the 6 symbols with the lowest trailing-30d volatility among
     those in an uptrend (close>EMA200). If <6 qualify → hold the rest in cash.
     Cost: 0.10%/side per changed slot (spot taker).
     Consistency story: low-vol names + cash cushion in storms.
  V2 DIP-LADDER (perp, close-only):
     per symbol, single position: enter at close when close>EMA200 and RSI(2)<10;
     exit at the FIRST close ≥ entry×1.015 (+1.5%) or ≤ entry×0.97 (−3%),
     whichever comes first at a close. R-normalize by the 3% stop → TP=+0.5R.
     Cost: 0.05%/side. High-WR by construction; the gates judge whether the
     fat left tail (−1R hits) breaks weekly consistency.
  C1 CARRY anchor: R102's always-on portfolio (15/15 positive weeks) quoted beside.

Gates (locked in CONSISTENCY_HUNT_BEGINS.md):
  ≥65% positive weeks · worst week ≥ −3% · maxDD ≤ 12% · costs included ·
  2024-regime survival. Metrics at 1.0× sizing on notional.
"""
import os, sys, glob, warnings
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
COST_SPOT=0.0010; COST_PERP=0.0005

print(SEP); print("  R103 — CONSISTENCY HUNT ROUND 1 (close-only fills, 2024 storm included)"); print(SEP)
t0=__import__("time").time()

# ── load hourly panel ────────────────────────────────────────────────────────
frames={}
for f in sorted(glob.glob(os.path.join(CACHE,"*_1H.parquet"))):
    sym=f.split("/")[-1].replace("_1H.parquet","")
    df=pd.read_parquet(f)
    df.index=pd.to_datetime(df.index,utc=True); df.sort_index(inplace=True)
    df=df[(df["open"]>0)&(df["low"]>0)]
    if len(df)<2500: continue
    df=df[["open","high","low","close","vol"]].astype(float)
    frames[sym]=df
print(f"symbols: {len(frames)}  span: {min(v.index.min() for v in frames.values()):%Y-%m-%d} → {max(v.index.max() for v in frames.values()):%Y-%m-%d}")

close=pd.DataFrame({s:v["close"] for s,v in frames.items()}).sort_index()
ret1=close.pct_change()
vol30=ret1.rolling(30*24).std()*np.sqrt(24*365)*100          # annualized %
ema200=close.ewm(span=200,adjust=False).mean()
uptrend=close>ema200

# weekly sampling: last bar of each ISO week (Sunday 23:00 or last available)
wk_close = close.resample("W").last()
wk_ret   = wk_close.pct_change()
wk_dates = wk_close.index

def consistency(weekly, name, sizing=1.0):
    w=weekly.dropna()*sizing
    pos=(w>0).mean()*100
    worst=w.min()*100
    # longest losing streak in weeks
    ls=cur=0
    for v in w:
        cur = cur+1 if v<=0 else 0
        ls=max(ls,cur)
    eq=(1+w).cumprod()
    dd=(((eq-eq.cummax())/eq.cummax()).min())*100
    days=(w.index.max()-w.index.min()).days
    ann=(eq.iloc[-1])**(365.0/max(days,1))-1
    calmar=ann/abs(dd/100) if dd<0 else np.nan
    mp=(w.groupby(w.index.to_period("M")).sum()>0).mean()*100
    print(f"  {name:<22} posW={pos:5.1f}%  worstW={worst:+6.2f}%  streak={ls:>2}w  maxDD={dd:6.1f}%  ann={ann*100:+6.1f}%  Calmar={calmar:5.2f}  posM={mp:.0f}%")
    return dict(name=name,pos=pos,worst=worst,streak=ls,dd=dd,ann=ann,calmar=calmar,mp=mp,w=w,eq=eq)

print(f"\n{SEP2}\n  V1) LOW-VOL BASKET — 6 slots, weekly, spot costs, cash when <6 qualify\n{SEP2}")
rows=[]; turnover=[]
dates=wk_close.index
for i in range(10, len(dates)-1):
    tsel=dates[i]; tnext=dates[i+1]
    vols=vol30.loc[:tsel].iloc[-1]
    ups =uptrend.loc[:tsel].iloc[-1]
    ok=[s for s in close.columns if pd.notna(vols.get(s)) and bool(ups.get(s, False))]
    pick=sorted(ok, key=lambda s: vols[s])[:6]
    if not pick:
        rows.append(dict(ts=tnext, r=0.0)); turnover.append(0); continue
    fwd=(wk_close.loc[tnext, pick]/wk_close.loc[tsel, pick]-1).dropna()
    # turnover vs previous pick
    prev = rows[-1].get("syms", []) if rows else []
    changed=len(set(pick)^set(prev))/max(len(pick),1)
    r=fwd.mean() - changed*COST_SPOT*2
    rows.append(dict(ts=tnext, r=r, syms=pick)); turnover.append(changed)
V1=pd.DataFrame(rows).set_index("ts")["r"]
m1=consistency(V1, "V1 low-vol basket")

print(f"\n{SEP2}\n  V2) DIP-LADDER — RSI2 dips in uptrends, TP+1.5%/SL-3% at closes only\n{SEP2}")
trades=[]
for s,df in frames.items():
    c=df["close"]
    ema=c.ewm(span=200,adjust=False).mean()
    delta=c.diff()
    up=delta.clip(lower=0).rolling(2).sum()/delta.abs().rolling(2).sum()*100
    rsi2=100-100/(1+up/(100-up).replace(0,np.nan))
    pos=None
    for i in range(210,len(df)):
        ts=df.index[i]; px=c.iloc[i]
        if pos is None:
            if c.iloc[i]>ema.iloc[i] and pd.notna(rsi2.iloc[i]) and rsi2.iloc[i]<10:
                pos=dict(entry=px, ts=ts)
            continue
        e=pos["entry"]
        if px>=e*1.015:
            trades.append(dict(sym=s, ts=ts, r=1.5, kind="TP")); pos=None      # +1.5%
        elif px<=e*0.97:
            trades.append(dict(sym=s, ts=ts, r=-3.0, kind="SL")); pos=None     # -3.0%
D=pd.DataFrame(trades)
print(f"  trades: {len(D)}  TP={(D['kind']=='TP').sum()}  SL={(D['kind']=='SL').sum()}  WR={(D['kind']=='TP').mean()*100:.0f}%")
D["r_net"]=D["r"]-2*COST_PERP*100   # net percent per trade
D["week"]=pd.to_datetime(D["ts"]).dt.to_period("W")
# weekly P&L % of notional: sum of per-trade % moves (one position per symbol; portfolio avg across syms)
D["week_ts"]=pd.to_datetime(D["ts"])
wk_pnl = D.set_index("week_ts")["r_net"].groupby(pd.Grouper(freq="W")).sum() / max(D["sym"].nunique(),1)
m2=consistency(wk_pnl, "V2 dip-ladder (avg/sym)")

# portfolio-combined weekly: 50/50 V1+carry-style? — report separately; blend V1+V2
both=pd.concat([V1.rename("V1"), wk_pnl.rename("V2")],axis=1).dropna()
blend=(both["V1"]+both["V2"])/2
m3=consistency(blend, "V1+V2 50/50 blend")

# yearly cut — the 2024 storm test
print(f"\n{SEP2}\n  YEARLY CUT (the 2024 regime test)\n{SEP2}")
for nm, s in [("V1", V1), ("V2", wk_pnl), ("BLEND", blend)]:
    print(f"  {nm}:")
    for y,g in s.groupby(s.index.year):
        pos=(g>0).mean()*100
        print(f"    {y}: weeks={len(g):>3}  posW={pos:5.1f}%  worstW={g.min()*100:+6.2f}%  sum={g.sum()*100:+7.2f}%")

# $ translation at $500 / $1000
print(f"\n{SEP2}\n  $ AT RECOMMENDED SIZING (1.0× notional for V1 spot; V2 at 1× as simulated)\n{SEP2}")
for cap in [500,1000]:
    e1=(1+V1.dropna()).cumprod().iloc[-1]
    e3=(1+blend.dropna()).cumprod().iloc[-1]
    mo1=(e1-1)*cap/max((V1.index.max()-V1.index.min()).days/30.4,1)
    mo3=(e3-1)*cap/max((blend.index.max()-blend.index.min()).days/30.4,1)
    print(f"  ${cap}: V1 ≈ ${mo1:5.1f}/mo   V1+V2 blend ≈ ${mo3:5.1f}/mo")

V1.to_csv(os.path.join(OUT,"r103_v1_weekly.csv"))
wk_pnl.to_csv(os.path.join(OUT,"r103_v2_weekly.csv"))
blend.to_csv(os.path.join(OUT,"r103_blend_weekly.csv"))
D.to_csv(os.path.join(OUT,"r103_dip_trades.csv"), index=False)
print(f"\n  Done in {__import__('time').time()-t0:.0f}s → quantlab_output/r103_*")
