"""
FOREX HUNT — F012
TEN-YEAR VALIDATION: does anything survive real data?

Why: F009→F011 all hit the same wall — 2y of data cannot separate a 1.0–1.1 PF
effect from noise. F012 fetches 2016→2026 (HistData ASCII M1 → 1H, UTC, vol =
bar-range proxy) and re-tests every surviving hypothesis with real power.

Design (contamination-aware — this is the point of the run):
  VIRGIN window  2016-01-01 → 2023-12-31. NO config in hunt history (R/F runs)
                 ever used pre-Oct-2024 data, so every candidate below is
                 prespecified w.r.t. this window. It acts as a pure replication
                 holdout.
  RECENT window  2024-01-01 → 2026-10-09. PARTIALLY CONTAMINATED: F007's config
                 was selected on Oct24→Aug25; F010/F011 on Oct24→Mar26. Disclosed
                 per candidate. Used only for survivors of the virgin window.

Candidates (all inherited, none new):
  C0  ALL structural trap-fades (4 levels, 0.15·ATR, all hours)   [F010 line]
  C1  F007 bear-only: bear-trap + downtrend + hr==8 → long        [F007]
  C2  C1 + bull mirror in uptrend                                 [F011 mirror]
  C3  with-trend structural fade (PD/P20, 0.30), all hours        [F011 CFG-A2]
  C4  BEAR traps in DOWNTREND, off-birth-window (structural)      [F011 hot cell]
  C5  BULL traps in UPTREND,   off-birth-window (structural)      [F011 hot cell]

Plus §1: the F010 WHEN study (birth-hour of big moves) replicated on 8y virgin
data vs the 2.7y recent era — is the 09:00–12:00 London-morning clock stable?
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features

FOREX10 = os.path.join(CONFIG["CACHE_FOLDER"], "forex10")
FOREXYF = os.path.join(CONFIG["CACHE_FOLDER"], "forex")
OUT = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUT, exist_ok=True)

PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP"]
SPREAD = {"EURUSD":0.00006,"GBPUSD":0.00010,"USDJPY":0.010,"AUDUSD":0.00008,
          "USDCAD":0.00010,"USDCHF":0.00010,"NZDUSD":0.00012,"EURGBP":0.00010}
PIP = {p:(0.01 if "JPY" in p else 0.0001) for p in PAIRS}

V_END   = pd.Timestamp("2024-01-01", tz="UTC")          # virgin window end (exclusive)
H_STUDY = 6
RR, SLM = 3.0, 1.0
BIG_Q   = 0.90
SWAP_PIP_DAY = 0.7

RNG = np.random.default_rng(42)
SEP  = "=" * 110
SEP2 = "-" * 90

def pf_of(rs):
    rs = np.asarray(rs, float); rs = rs[~np.isnan(rs)]
    if len(rs) == 0: return float("nan")
    w = rs[rs > 0].sum(); lo = abs(rs[rs < 0].sum())
    return w / lo if lo > 0 else (99.0 if w > 0 else 1.0)

def session_of(h):
    if h < 7:   return "ASIA"
    if h < 12:  return "LONDON"
    if h < 17:  return "NY_OVERLAP"
    if h < 21:  return "LATE_US"
    return "ROLLOVER"

print(); print(SEP)
print("  FOREX HUNT — F012  TEN-YEAR VALIDATION (virgin 2016-2023 vs recent 2024-2026)")
print(SEP)
t0 = time.time()

# ── LOAD 10Y 1H ──────────────────────────────────────────────────────────────
print("\n  Loading HistData 10y 1H …")
frames = {}
for p in PAIRS:
    df = pd.read_parquet(f"{FOREX10}/{p}_1H.parquet")
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    df = df[(df["open"]>0)&(df["high"]>=df["low"])&(df["low"]>0)]
    f = add_features(df)
    f.dropna(subset=["ema200","ema50","ema20","atr14","adx14"], inplace=True)
    day = f.index.normalize()
    typ = (f["high"]+f["low"]+f["close"])/3.0
    f["vwap"] = (typ*f["vol"]).groupby(day).cumsum()/f["vol"].groupby(day).cumsum().replace(0,np.nan)
    f["p20_hi"] = f["high"].rolling(20).max().shift(1)
    f["p20_lo"] = f["low"].rolling(20).min().shift(1)
    d = f.resample("1D").agg({"high":"max","low":"min"}).shift(1)
    f["pd_hi"] = d["high"].reindex(day).values; f["pd_lo"] = d["low"].reindex(day).values
    a = f["atr14"]
    fwd = (f["close"].shift(-H_STUDY) - f["close"])/a
    f["fwd_abs"] = fwd.abs(); f["fwd_dir"] = np.sign(fwd)
    f["hour"] = f.index.hour
    frames[p] = f
    print(f"    {p}: {len(f):>6} bars  {f.index.min():%Y-%m-%d} → {f.index.max():%Y-%m-%d}")

# cross-vendor sanity vs yfinance overlap (weekly closes)
try:
    cors=[]
    for p in PAIRS:
        a = pd.read_parquet(f"{FOREXYF}/{p}_1H.parquet")
        a.index = pd.to_datetime(a.index, utc=True)
        w1 = frames[p]["close"].resample("1W").last().dropna()
        w2 = a["close"].resample("1W").last().dropna()
        j = pd.concat([w1,w2],axis=1,keys=["hd","yf"]).dropna()
        if len(j)>20: cors.append(j["hd"].corr(j["yf"]))
    print(f"  vendor sanity: mean weekly-close corr HistData vs yfinance = {np.mean(cors):.5f}")
except Exception as e:
    print(f"  vendor sanity skipped: {e}")

# ── §1 WHEN — birth-hour replication ─────────────────────────────────────────
sel_fwd = pd.concat([f["fwd_abs"][f.index < V_END] for f in frames.values()]).dropna()
BIG_THR = float(sel_fwd.quantile(BIG_Q))
for p in PAIRS: frames[p]["big"] = frames[p]["fwd_abs"] >= BIG_THR

print(f"\n{SEP2}\n  1) WHEN on 10y: P(big 6h) by hour — virgin 2016-23 vs recent 24-26 (thr={BIG_THR:.2f} ATR)\n{SEP2}")
grid = pd.concat([f[["hour","big","fwd_abs"]].assign(pair=p, ts=f.index) for p,f in frames.items()], ignore_index=True)
grid["era"] = np.where(grid["ts"]<V_END, "V16-23", "R24-26")
base = grid[grid["era"]=="V16-23"]["big"].mean()
print(f"    base P(big) virgin = {base*100:.1f}%")
print(f"    {'hour':<7}{'n(V)':>7}{'P(big|V)':>9}{'lift':>7}{'P(big|R)':>10}{'liftR':>7}")
stab=[]
for h in range(24):
    gv = grid[(grid["era"]=="V16-23")&(grid["hour"]==h)]
    gr = grid[(grid["era"]=="R24-26")&(grid["hour"]==h)]
    lv, lr = gv["big"].mean()/base, gr["big"].mean()/base
    stab.append((h,lv,lr))
    print(f"    {h:02d}:00 {len(gv):>7}{gv['big'].mean()*100:>8.1f}%{lv:>6.2f}x{gr['big'].mean()*100:>9.1f}%{lr:>6.2f}x")
v_sorted = sorted(range(24), key=lambda h:-stab[h][1])[:5]
agree = sum(1 for h in v_sorted if stab[h][2] > 1.2)
top_v = ", ".join(f"{h:02d}" for h in sorted(v_sorted))
print(f"    → virgin top-5 birth hours: {top_v} UTC · replicated (>1.2x) in recent era: {agree}/5")

# ── 4H FRAMES + OUTCOMES + TRAPS ─────────────────────────────────────────────
print("\n  Building 4H frames + outcomes (SL1/TP3·ATR, 24 bars) …")
frames4={}
for p in PAIRS:
    df = frames[p][["open","high","low","close","vol"]].resample("4H").agg(
        {"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()
    f4 = add_features(df)
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

def outcomes(rr):
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

out4 = outcomes(RR)

def build_traps(levels, depth):
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
    ev = pd.concat(parts, ignore_index=True)
    lab = pd.concat([f4[["trend","hour"]].assign(pair=p, ts=f4.index) for p,f4 in frames4.items()], ignore_index=True)
    o4  = pd.concat([out4[p].assign(pair=p, ts=out4[p].index) for p in PAIRS], ignore_index=True)
    a4m = pd.concat([pd.DataFrame({"pair":p,"ts":frames4[p].index,"atr":frames4[p]["atr14"].values}) for p in PAIRS])
    ev = ev.merge(lab,on=["pair","ts"],how="left").merge(o4,on=["pair","ts"],how="left").merge(a4m,on=["pair","ts"],how="left")
    ev["spread"] = ev["pair"].map(SPREAD)
    ev["side"]   = np.where(ev["kind"]=="BEAR_TRAP", 1, -1)
    ev["r"]      = np.where(ev["side"]==1, ev["long_out"], ev["short_out"])
    ev["t_up"]   = ev["trend"] > 0
    ev["win"]    = ev["hour"].isin([4,8])
    return ev.dropna(subset=["r"]).reset_index(drop=True)

print("  Detecting traps (mirror 0.15 + structural 0.30) …")
evM = build_traps(["PD","P20","VWAP","EMA20"], 0.15)
evS = build_traps(["PD","P20"], 0.30)
print(f"  mirror traps={len(evM)}  structural traps={len(evS)}")

def cost_r(sub, mult=1.0, swap=False):
    c = 2*mult*sub["spread"].values/np.maximum(sub["atr"].values,1e-12)
    if swap:
        c = c + (SWAP_PIP_DAY*sub["pair"].map(PIP).values)/np.maximum(sub["atr"].values,1e-12)
    return sub["r"].values - c

CANDS = {
 "C0 all-trap fade (4lvl .15, all hrs)":        evM,
 "C1 F007 bear-only DN-trend hr8":              evM[(evM["kind"]=="BEAR_TRAP")&(~evM["t_up"])&(evM["hour"]==8)],
 "C2 bear+DN hr8 / bull+UP hr8 mirror":         evM[((evM["kind"]=="BEAR_TRAP")&(~evM["t_up"])&(evM["hour"]==8))
                                                   |((evM["kind"]=="BULL_TRAP")&( evM["t_up"])&(evM["hour"]==8))],
 "C3 with-trend structural fade all-hrs":       evS[((evS["kind"]=="BEAR_TRAP")&( evS["t_up"]))
                                                   |((evS["kind"]=="BULL_TRAP")&(~evS["t_up"]))],
 "C4 BEAR/DN structural off-window":            evS[(evS["kind"]=="BEAR_TRAP")&(~evS["t_up"])&(~evS["win"])],
 "C5 BULL/UP structural off-window":            evS[(evS["kind"]=="BULL_TRAP")&( evS["t_up"])&(~evS["win"])],
}

# ── §2 VIRGIN BATTERY 2016-2023 ──────────────────────────────────────────────
print(f"\n{SEP2}\n  2) VIRGIN BATTERY 2016-2023 (pure replication holdout for every candidate)\n{SEP2}")
print(f"    {'candidate':<40}{'n':>7}{'PF':>7}{'PF@cost':>9}{'WR':>7}")
survivors=[]
for nm, sub in CANDS.items():
    v = sub[sub["ts"]<V_END]
    if len(v)==0: continue
    pfc = pf_of(cost_r(v))
    print(f"    {nm:<40}{len(v):>7}{pf_of(v['r']):>7.2f}{pfc:>9.2f}{(v['r']>0).mean()*100:>6.0f}%")
    if pfc > 1.05 and len(v) >= 500:
        survivors.append((nm, sub))

# ── §3 RECENT BATTERY 2024-2026 for survivors ────────────────────────────────
print(f"\n{SEP2}\n  3) RECENT BATTERY 2024-2026 (survivors of §2 only; contamination disclosed)\n{SEP2}")
if not survivors:
    print("    NO survivors of the virgin window — nothing proceeds to the recent era.")
    print("    → Every inherited 'edge' dies on 8 years of data nobody has seen.")
else:
    for nm, sub in survivors:
        h = sub[sub["ts"]>=V_END]
        rs_c = cost_r(h)
        print(f"\n  {nm}")
        print(f"    recent   : n={len(h):>5}  PF={pf_of(h['r']):5.2f}  PF@cost={pf_of(rs_c):5.2f}")
        outb=np.empty(2000)
        for b in range(2000):
            x=rs_c[RNG.integers(0,len(rs_c),len(rs_c))]; outb[b]=pf_of(x)
        print(f"    boot CI PF@cost: P5={np.percentile(outb,5):.2f} med={np.median(outb):.2f} P95={np.percentile(outb,95):.2f}")
        floor=9.9
        for p in PAIRS:
            v=pf_of(cost_r(h[h["pair"]!=p]))
            if not np.isnan(v): floor=min(floor,v)
        print(f"    LOO floor: {floor:.2f}")
        for mult,label in [(0.4,"ECN"),(1.0,"retail"),(1.5,"wide")]:
            print(f"    PF@cost [{label:<6}] = {pf_of(cost_r(h,mult)):.2f}  [+swap] = {pf_of(cost_r(h,mult,swap=True)):.2f}")
        prof=[]; dds=[]
        for _ in range(3000):
            x=rs_c[RNG.integers(0,len(rs_c),len(rs_c))]
            cap=100.0; eq=[cap]
            for r in x: cap*=(1+0.01*r); eq.append(cap)
            eq=np.array(eq); pk=np.maximum.accumulate(eq)
            prof.append(cap); dds.append(((eq-pk)/pk).min())
        prof=np.array(prof); dds=np.array(dds)
        print(f"    MC: P(end>100)={(prof>100).mean()*100:.0f}%  DD med={np.percentile(dds,50)*100:.1f}% P5={np.percentile(dds,5)*100:.1f}%")
        gate=(len(h)>=200) and (pf_of(rs_c)>1.10) and (floor>1.00)
        print(f"    GATE (n≥200, PF@cost>1.10, LOO>1.0): {'✅ PASS' if gate else '❌ FAIL'}")
        if "F007" in nm:
            print("    ⚠ contamination: config selected on Oct24→Aug25 (yfinance era)")
        else:
            print("    ⚠ contamination: config selected on Oct24→Mar26 (F010/F011 era)")

# ── SAVE ─────────────────────────────────────────────────────────────────────
evM.to_csv(os.path.join(OUT,"f012_traps_mirror_4h.csv"), index=False)
evS.to_csv(os.path.join(OUT,"f012_traps_structural_4h.csv"), index=False)
pd.DataFrame(stab, columns=["hour","lift_virgin","lift_recent"]).to_csv(
    os.path.join(OUT,"f010_hour_lift_10y.csv"), index=False)
print(f"\n  Done in {time.time()-t0:.0f}s → quantlab_output/f012_*")
