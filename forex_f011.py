"""
FOREX HUNT — F011
DIRECTION ENGINE INSIDE THE BIRTH WINDOW

Carries forward F010's map: big moves are born 06:00–12:00 UTC (peak 09–12),
traps fuel them, but raw trap-fade direction is a coin flip (46–48%).
F007 found a directional edge with trend context (bear-trap fade in DOWNTREND,
London). F011 asks: which context gives the fade its direction —
  WITH-trend    (trap ends the pullback → trend continues: fade aligns WITH trend)
  COUNTER-trend (trap ends the trend → reversal: fade opposes trend, F007 story)

PRESPECIFIED hypotheses (written before any holdout run):
  H1  With-trend trap-fade has higher P(fade-direction big move) and PF than
      counter-trend trap-fade (fuel + trend both push the same way).
  H2  Restricting entries to the birth window (4H bars 04/08 UTC) improves PF
      vs the same logic at all hours (F010 timing map).
  H3  At least one prespecified config clears the repo gate:
      holdout n ≥ 200 · holPF@cost > 1.10 · LOO floor > 1.00.

Prespecified configs (locked BEFORE the run — derived from F007 + F010 only):
  CFG-A  with-trend continuation, birth window:
         bear-trap@PD/P20 (wick ≥0.30·ATR, close back inside) in UPTREND → long
         bull-trap@PD/P20 in DOWNTREND → short · 4H bar hour ∈ {04,08} · RR3/SL1
  CFG-A2 with-trend continuation, ALL hours (isolates H2 timing value)
  CFG-B  counter-trend reversal — F007 mirror on extended data:
         bear-trap (4 levels PD/P20/VWAP/EMA20, wick ≥0.15·ATR) in DOWNTREND,
         hr==8 → long; bull-trap mirror in UPTREND → short · RR3/SL1

Execution: 4H bars, signal-bar close (E6), SL 1·ATR, TP 3·ATR, horizon 24 bars,
TP-before-SL intrabar (F007 mirror). Costs: F007 retail spread table, swap as
sensitivity. Selection < 2026-04-01 · development-holdout ≥ 2026-04-01
(seen by F010 v1 — treated as development; final verdicts await fresh data).
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features

FOREX_DIR = os.path.join(CONFIG["CACHE_FOLDER"], "forex")
OUT = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUT, exist_ok=True)

PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP"]
SPREAD = {"EURUSD":0.00006,"GBPUSD":0.00010,"USDJPY":0.010,"AUDUSD":0.00008,
          "USDCAD":0.00010,"USDCHF":0.00010,"NZDUSD":0.00012,"EURGBP":0.00010}
PIP = {p:(0.01 if "JPY" in p else 0.0001) for p in PAIRS}

HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")
F007_BOUNDARY = pd.Timestamp("2025-08-01", tz="UTC")
H_STUDY   = 6
RR, SLM   = 3.0, 1.0
DEPTH_S   = 0.30   # structural traps (CFG-A/A2)
DEPTH_F   = 0.15   # F007-mirror traps (CFG-B)
BIG_Q     = 0.90
SWAP_PIP_DAY = 0.7

RNG = np.random.default_rng(42)
SEP  = "=" * 110
SEP2 = "-" * 90

def pf_of(rs):
    rs = np.asarray(rs, float); rs = rs[~np.isnan(rs)]
    if len(rs) == 0: return float("nan")
    w = rs[rs > 0].sum(); lo = abs(rs[rs < 0].sum())
    return w / lo if lo > 0 else (99.0 if w > 0 else 1.0)

def rrank(s, w=500):
    return s.rolling(w, min_periods=100).rank(pct=True)

print(); print(SEP)
print("  FOREX HUNT — F011  DIRECTION ENGINE INSIDE THE BIRTH WINDOW")
print(SEP)
t0 = time.time()

# ── 1H LOAD (for the WHY study) ──────────────────────────────────────────────
print("\n  Loading 1H data …")
frames = {}
for p in PAIRS:
    df = pd.read_parquet(f"{FOREX_DIR}/{p}_1H.parquet")
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for c in ["open","high","low","close"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["open","high","low","close"], inplace=True)
    if "vol" not in df.columns: df["vol"] = (df["high"]-df["low"]).clip(lower=1e-12)
    f = add_features(df)
    f.dropna(subset=["ema200","ema50","ema20","atr14","adx14"], inplace=True)
    day = f.index.normalize()
    f["p20_hi"] = f["high"].rolling(20).max().shift(1)
    f["p20_lo"] = f["low"].rolling(20).min().shift(1)
    d = f.resample("1D").agg({"high":"max","low":"min"}).shift(1)
    f["pd_hi"] = d["high"].reindex(day).values; f["pd_lo"] = d["low"].reindex(day).values
    a = f["atr14"]
    fwd = (f["close"].shift(-H_STUDY) - f["close"])/a
    f["fwd_abs"] = fwd.abs(); f["fwd_dir"] = np.sign(fwd)
    frames[p] = f

sel_fwd = pd.concat([f["fwd_abs"][f.index < HOLDOUT_START] for f in frames.values()]).dropna()
BIG_THR = float(sel_fwd.quantile(BIG_Q))
for p in PAIRS: frames[p]["big"] = frames[p]["fwd_abs"] >= BIG_THR

# 1H structural traps (PD/P20, 0.30·ATR) with trend + window context
print("  1H structural traps + context …")
tr_parts=[]
for p,f in frames.items():
    hi=f["high"].values; lo=f["low"].values; cl=f["close"].values; atr=f["atr14"].values
    lup=f["pd_hi"].values; ldn=f["pd_lo"].values
    u20=f["p20_hi"].values; d20=f["p20_lo"].values
    trend=(f["ema50"]-f["ema200"]).values
    rows=[]
    for i in range(21,len(f)):
        if np.isnan(atr[i]) or atr[i]<=0: continue
        dth=DEPTH_S*atr[i]
        for lvl,(Lu,Ld) in {"PD":(lup[i],ldn[i]),"P20":(u20[i],d20[i])}.items():
            if np.isnan(Lu) or np.isnan(Ld): continue
            if lo[i] < Ld-dth and cl[i] > Ld:
                rows.append(dict(pair=p,ts=f.index[i],kind="BEAR_TRAP",trend=trend[i],hour=f.index[i].hour))
            elif hi[i] > Lu+dth and cl[i] < Lu:
                rows.append(dict(pair=p,ts=f.index[i],kind="BULL_TRAP",trend=trend[i],hour=f.index[i].hour))
    tr_parts.append(pd.DataFrame(rows))
tr1 = pd.concat(tr_parts, ignore_index=True)
lab1 = pd.concat([f[["big","fwd_dir"]].assign(pair=p, ts=f.index) for p,f in frames.items()], ignore_index=True)
tr1 = tr1.merge(lab1, on=["pair","ts"], how="left").dropna(subset=["big"])
tr1["fade_side"] = np.where(tr1["kind"]=="BEAR_TRAP", 1, -1)
tr1["win"]  = tr1["hour"].between(6,11)
tr1["t_up"] = tr1["trend"] > 0
tr1["cell"] = np.where(tr1["t_up"],"UP","DN") + "/" + np.where(tr1["win"],"WIN","offW")
tr1s = tr1[tr1["ts"] < HOLDOUT_START]

print(f"\n{SEP2}\n  1) WHY: where does fade DIRECTION come from? (1H birth study, selection)\n{SEP2}")
print(f"    {'cell':<20}{'n':>7}{'P(big6h)':>10}{'%big in fade-dir':>18}")
for k in ["BEAR_TRAP","BULL_TRAP"]:
    for tup in [True, False]:
        for win in [True, False]:
            g = tr1s[(tr1s["kind"]==k)&(tr1s["t_up"]==tup)&(tr1s["win"]==win)]
            if len(g) < 30: continue
            fade_side = 1 if k=="BEAR_TRAP" else -1
            pct = ((g["fwd_dir"]==fade_side)&g["big"]).sum()/max(g["big"].sum(),1)*100
            lbl = f"{k[:4]} {'UP' if tup else 'DN'} {'WIN' if win else 'offW'}"
            print(f"    {lbl:<20}{len(g):>7}{g['big'].mean()*100:>9.1f}%{pct:>17.0f}%")

# ── 4H EXECUTION FRAMES ──────────────────────────────────────────────────────
print("\n  Building 4H frames + outcomes (SL1, TP3, 24 bars) …")
def build_4h():
    frames4={}
    for p in PAIRS:
        df = pd.read_parquet(f"{FOREX_DIR}/{p}_1H.parquet")
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        if "vol" not in df.columns: df["vol"] = (df["high"]-df["low"]).clip(lower=1e-12)
        df4 = df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()
        f4 = add_features(df4)
        f4.dropna(subset=["ema200","ema50","ema20","atr14","adx14"], inplace=True)
        day4 = f4.index.normalize()
        typ4 = (f4["high"]+f4["low"]+f4["close"])/3.0
        f4["vwap"] = (typ4*f4["vol"]).groupby(day4).cumsum()/f4["vol"].groupby(day4).cumsum().replace(0,np.nan)
        f4["p20_hi"] = f4["high"].rolling(20).max().shift(1)
        f4["p20_lo"] = f4["low"].rolling(20).min().shift(1)
        d4 = f4.resample("1D").agg({"high":"max","low":"min"}).shift(1)
        f4["pd_hi"] = d4["high"].reindex(day4).values; f4["pd_lo"] = d4["low"].reindex(day4).values
        f4["hour"]  = f4.index.hour
        f4["trend"] = f4["ema50"] - f4["ema200"]
        frames4[p]=f4
    return frames4

def outcomes(frames4, rr):
    out={}
    for p,f in frames4.items():
        cl=f["close"].values; hi=f["high"].values; lo=f["low"].values; atr=f["atr14"].values; n=len(f)
        lo_arr=np.full(n,np.nan); so_arr=np.full(n,np.nan)
        for i in range(n):
            if np.isnan(atr[i]) or atr[i]<=0: continue
            end=min(i+24,n)
            tp=cl[i]+rr*atr[i]; sl=cl[i]-SLM*atr[i]; hit=None
            for b in range(i+1,end):
                if hi[b]>=tp: hit=rr; break
                if lo[b]<=sl: hit=-1.0; break
            if hit is not None: lo_arr[i]=hit
            elif end>i+1: lo_arr[i]=(cl[end-1]-cl[i])/atr[i]
            tps=cl[i]-rr*atr[i]; sls=cl[i]+SLM*atr[i]; hit=None
            for b in range(i+1,end):
                if lo[b]<=tps: hit=rr; break
                if hi[b]>=sls: hit=-1.0; break
            if hit is not None: so_arr[i]=hit
            elif end>i+1: so_arr[i]=(cl[i]-cl[end-1])/atr[i]
        out[p]=pd.DataFrame({"long_out":lo_arr,"short_out":so_arr}, index=f.index)
    return out

def traps4(frames4, levels, depth):
    parts=[]
    for p,f in frames4.items():
        hi=f["high"].values; lo=f["low"].values; cl=f["close"].values; atr=f["atr14"].values
        lv={"PD":(f["pd_hi"].values,f["pd_lo"].values),"P20":(f["p20_hi"].values,f["p20_lo"].values),
            "VWAP":(f["vwap"].values,f["vwap"].values),"EMA20":(f["ema20"].values,f["ema20"].values)}
        rows=[]
        for tname in levels:
            lup,ldn = lv[tname]
            for i in range(21,len(f)):
                if np.isnan(atr[i]) or atr[i]<=0 or np.isnan(lup[i]) or np.isnan(ldn[i]): continue
                dth=depth*atr[i]
                if   lo[i] < ldn[i]-dth and cl[i] > ldn[i]: rows.append(dict(pair=p,ts=f.index[i],kind="BEAR_TRAP"))
                elif hi[i] > lup[i]+dth and cl[i] < lup[i]: rows.append(dict(pair=p,ts=f.index[i],kind="BULL_TRAP"))
        parts.append(pd.DataFrame(rows))
    return pd.concat(parts, ignore_index=True)

frames4 = build_4h()
out4    = outcomes(frames4, RR)
lab4 = pd.concat([f4[["trend","hour"]].assign(pair=p, ts=f4.index) for p,f4 in frames4.items()], ignore_index=True)
o4   = pd.concat([out4[p].assign(pair=p, ts=out4[p].index) for p in PAIRS], ignore_index=True)
a4m  = pd.concat([pd.DataFrame({"pair":p,"ts":frames4[p].index,"atr":frames4[p]["atr14"].values}) for p in PAIRS])

def assemble(ev):
    ev = ev.merge(lab4,on=["pair","ts"],how="left").merge(o4,on=["pair","ts"],how="left").merge(a4m,on=["pair","ts"],how="left")
    ev["spread"] = ev["pair"].map(SPREAD)
    ev["side"]   = np.where(ev["kind"]=="BEAR_TRAP", 1, -1)
    ev["r"]      = np.where(ev["side"]==1, ev["long_out"], ev["short_out"])
    ev["win"]    = ev["hour"].isin([4,8])
    ev["t_up"]   = ev["trend"] > 0
    return ev.dropna(subset=["r"]).reset_index(drop=True)

evS = assemble(traps4(frames4, ["PD","P20"], DEPTH_S))                 # structural (CFG-A/A2)
evF = assemble(traps4(frames4, ["PD","P20","VWAP","EMA20"], DEPTH_F))  # F007 mirror (CFG-B)
print(f"  4H traps with outcomes: structural={len(evS)}  mirror={len(evF)}")

def cost_r(sub, mult=1.0, swap=False):
    c = 2*mult*sub["spread"].values/np.maximum(sub["atr"].values,1e-12)
    if swap:
        c = c + (SWAP_PIP_DAY*sub["pair"].map(PIP).values)/np.maximum(sub["atr"].values,1e-12)*(24/24.0)
    return sub["r"].values - c

# ── 2) FACTORIAL MAP (selection) ─────────────────────────────────────────────
print(f"\n{SEP2}\n  2) FACTORIAL: trap x trend x window — fade PF@cost (structural 4H, selection)\n{SEP2}")
sS = evS[evS["ts"] < HOLDOUT_START]
print(f"    {'cell':<26}{'n':>6}{'PF':>7}{'PF@cost':>9}")
for kind in ["BEAR_TRAP","BULL_TRAP"]:
    for tup in [True, False]:
        for win in [True, False]:
            g = sS[(sS["kind"]==kind)&(sS["t_up"]==tup)&(sS["win"]==win)]
            if len(g) < 25: continue
            lbl = f"{kind[:4]} {'UP' if tup else 'DN'}-trend {'WIN' if win else 'offWin'}"
            print(f"    {lbl:<26}{len(g):>6}{pf_of(g['r']):>7.2f}{pf_of(cost_r(g)):>9.2f}")

# ── 3) PRESPECIFIED CONFIGS — single-shot holdout ────────────────────────────
print(f"\n{SEP2}\n  3) PRESPECIFIED CONFIGS (single-shot holdout battery)\n{SEP2}")
with_trend = (evS["kind"]=="BEAR_TRAP")&(evS["t_up"]) | (evS["kind"]=="BULL_TRAP")&(~evS["t_up"])
cfgA  = evS[with_trend & evS["win"]]
cfgA2 = evS[with_trend]
c_bear = (evF["kind"]=="BEAR_TRAP")&(~evF["t_up"])&(evF["hour"]==8)
c_bull = (evF["kind"]=="BULL_TRAP")&(evF["t_up"]) &(evF["hour"]==8)
cfgB  = evF[c_bear | c_bull]

def battery(name, sub):
    s, h = sub[sub["ts"]<HOLDOUT_START], sub[sub["ts"]>=HOLDOUT_START]
    print(f"\n  {name}")
    print(f"    selection : n={len(s):>5}  PF={pf_of(s['r']):5.2f}  PF@cost={pf_of(cost_r(s)):5.2f}")
    print(f"    holdout*  : n={len(h):>5}  PF={pf_of(h['r']):5.2f}  PF@cost={pf_of(cost_r(h)):5.2f}   (*dev holdout — seen via F010 v1)")
    pre, post = sub[sub["ts"]<F007_BOUNDARY], sub[sub["ts"]>=F007_BOUNDARY]
    print(f"    F007 bdy  : pre-Aug25 PF@cost={pf_of(cost_r(pre)):5.2f} (n={len(pre)})  post-Aug25={pf_of(cost_r(post)):5.2f} (n={len(post)})")
    if len(h) < 30:
        print("    (holdout too small — no battery)"); return
    rs_c = cost_r(h)
    outb=np.empty(2000)
    for b in range(2000):
        x=rs_c[RNG.integers(0,len(rs_c),len(rs_c))]; outb[b]=pf_of(x)
    print(f"    boot CI hol PF@cost: P5={np.percentile(outb,5):.2f} med={np.median(outb):.2f} P95={np.percentile(outb,95):.2f}")
    floor=9.9
    for p in PAIRS:
        v=pf_of(cost_r(h[h["pair"]!=p]))
        if not np.isnan(v): floor=min(floor,v)
    print(f"    LOO floor: {floor:.2f}")
    for mult,label in [(0.4,"ECN"),(1.0,"retail"),(1.5,"wide")]:
        print(f"    hol PF@cost [{label:<6}] = {pf_of(cost_r(h,mult)):.2f}  [+swap] = {pf_of(cost_r(h,mult,swap=True)):.2f}")
    prof=[]; dds=[]
    for _ in range(3000):
        x=rs_c[RNG.integers(0,len(rs_c),len(rs_c))]
        cap=100.0; eq=[cap]
        for r in x: cap*=(1+0.01*r); eq.append(cap)
        eq=np.array(eq); pk=np.maximum.accumulate(eq)
        prof.append(cap); dds.append(((eq-pk)/pk).min())
    prof=np.array(prof); dds=np.array(dds)
    print(f"    MC: P(end>100)={(prof>100).mean()*100:.0f}%  P(end<90)={(prof<90).mean()*100:.0f}%  DD med={np.percentile(dds,50)*100:.1f}% P5={np.percentile(dds,5)*100:.1f}%")
    gate=(len(h)>=200) and (pf_of(rs_c)>1.10) and (floor>1.00)
    print(f"    GATE (n≥200, holPF@cost>1.10, LOO>1.0): {'✅ PASS' if gate else '❌ FAIL'}")

battery("CFG-A  with-trend continuation, birth window (hr 04/08)", cfgA)
battery("CFG-A2 with-trend continuation, all hours", cfgA2)
battery("CFG-B  counter-trend reversal — F007 mirror (hr 8, 4 levels, 0.15)", cfgB)

# ── 4) RR SENSITIVITY on the best with-trend cell (selection only) ───────────
print(f"\n{SEP2}\n  4) RR SENSITIVITY (selection, with-trend all-hours cell)\n{SEP2}")
for rr in [2.0, 3.0, 4.0]:
    oo = outcomes(frames4, rr) if rr != RR else out4
    ooall = pd.concat([oo[p].assign(pair=p, ts=oo[p].index) for p in PAIRS], ignore_index=True)
    evX = evS[["pair","ts","kind","t_up","win"]].merge(ooall,on=["pair","ts"],how="left")
    evX["side"] = np.where(evX["kind"]=="BEAR_TRAP", 1, -1)
    evX["r"]    = np.where(evX["side"]==1, evX["long_out"], evX["short_out"])
    evX = evX.merge(a4m,on=["pair","ts"],how="left")
    evX["spread"]=evX["pair"].map(SPREAD)
    wt = evX[((evX["kind"]=="BEAR_TRAP")&(evX["t_up"])) | ((evX["kind"]=="BULL_TRAP")&(~evX["t_up"]))]
    wt = wt[wt["ts"]<HOLDOUT_START].dropna(subset=["r"])
    c  = 2*wt["spread"].values/np.maximum(wt["atr"].values,1e-12)
    print(f"    RR {rr:.1f}: n={len(wt):>5}  PF={pf_of(wt['r']):5.2f}  PF@cost={pf_of(wt['r'].values-c):5.2f}")

# ── SAVE ─────────────────────────────────────────────────────────────────────
evS.to_csv(os.path.join(OUT,"f011_structural_traps_4h.csv"), index=False)
evF.to_csv(os.path.join(OUT,"f011_mirror_traps_4h.csv"), index=False)
print(f"\n  Done in {time.time()-t0:.0f}s → quantlab_output/f011_*")
