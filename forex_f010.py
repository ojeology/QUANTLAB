"""
FOREX HUNT — F010
TRAP FORENSICS: why do big moves happen, and which conditions align when?

v1 → v2 CAUSAL AUDIT (done before any holdout conclusion was drawn):
  (a) v1 used a 24h forward label for the session study → each bar's window
      swallowed later sessions (Asia bars contain London+NY). Fixed: session
      forensics use a 6h forward label — a big move is attributed to the
      session where it is BORN.
  (b) v1 trapped on VWAP/EMA20 → they are always within 0.15·ATR of price,
      so 57k "events" ≈ every bar, lift collapses to ~1.0x. Fixed: traps are
      only swept STRUCTURAL liquidity (prior-day high/low, 20-bar extremes —
      where stops actually rest) and the wick must reach ≥ 0.30·ATR beyond.
  (c) v1 trade test ran on 1H → cost per R ≈ 3× F007's 4H execution.
      Fixed: trade battery mirrors F007 (4H bars, SL 1·ATR, TP 3·ATR,
      horizon 24×4H bars, TP-before-SL, E6 entry).
Disclosures: v1's holdout was PEEKED for two naive configs (both failed hard:
holPF@cost 0.73 / 0.75). v2 therefore reports the Apr→Oct-2026 window as a
DEVELOPMENT holdout — anything passing here is exploratory until fresh data.

Hypotheses (prespecified before the v2 run):
  H1 WHEN   — big moves are born in session opens: P(big6h) peaks in the
              first hours of London and NY, troughs in Asia mid / rollover.
  H2 WHAT   — expansion-favoring conditions (squeeze, near structural
              liquidity, ADX-active, participation surge) raise P(big6h)
              above base; the confluence score is monotone.
  H3 WHY    — structural sweep-and-fail (trap) events precede bigger moves
              than clean breaks of the same levels: trapped stops are fuel.
  H4 TRADE  — fading aligned structural traps on 4H (F007 execution) can
              clear holPF@cost > 1.10.

Protocol: 1H data 8 majors Oct-2024→Oct-2026 · selection < 2026-04-01 ·
development-holdout ≥ 2026-04-01 · big6h = top-decile |fwd6h|/ATR (threshold
from selection only) · costs = F007 retail spread table (+swap sensitivity).
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
H_STUDY  = 6      # 6h forward label — a big move belongs to the session that births it
RR, SLM  = 3.0, 1.0
DEPTH    = 0.30   # structural wick depth, ATR units (v2; was 0.15)
BIG_Q    = 0.90
LIQ_ATR  = 0.50
VOL_SURGE = 1.30
SWAP_PIP_DAY = 0.7

RNG = np.random.default_rng(42)
SEP  = "=" * 110
SEP2 = "-" * 90

def session_of(h):
    if h < 7:   return "ASIA"
    if h < 12:  return "LONDON"
    if h < 17:  return "NY_OVERLAP"
    if h < 21:  return "LATE_US"
    return "ROLLOVER"

def rrank(s, w=500):
    return s.rolling(w, min_periods=100).rank(pct=True)

def pf_of(rs):
    rs = np.asarray(rs, float); rs = rs[~np.isnan(rs)]
    if len(rs) == 0: return float("nan")
    w = rs[rs > 0].sum(); lo = abs(rs[rs < 0].sum())
    return w / lo if lo > 0 else (99.0 if w > 0 else 1.0)

print(); print(SEP)
print("  FOREX HUNT — F010 v2  TRAP FORENSICS (fixed): when / what / why of big moves")
print(SEP)
t0 = time.time()

# ── 1H FRAMES + CAUSAL FLAGS ─────────────────────────────────────────────────
print("\n  Loading 1H data + causal features …")
frames = {}
for p in PAIRS:
    df = pd.read_parquet(f"{FOREX_DIR}/{p}_1H.parquet")
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for c in ["open","high","low","close"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["open","high","low","close"], inplace=True)
    if "vol" not in df.columns: df["vol"] = (df["high"]-df["low"]).clip(lower=1e-12)

    f = add_features(df)
    f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14",
                     "ema_dist_pct","real_vol_20","bb_width"], inplace=True)
    day = f.index.normalize()
    typ = (f["high"]+f["low"]+f["close"])/3.0
    f["vwap"] = (typ*f["vol"]).groupby(day).cumsum()/f["vol"].groupby(day).cumsum().replace(0,np.nan)
    f["p20_hi"] = f["high"].rolling(20).max().shift(1)
    f["p20_lo"] = f["low"].rolling(20).min().shift(1)
    d = f.resample("1D").agg({"high":"max","low":"min"}).shift(1)
    f["pd_hi"] = d["high"].reindex(day).values; f["pd_lo"] = d["low"].reindex(day).values

    f["bb_rank"]   = rrank(f["bb_width"])
    f["squeeze"]   = f["bb_rank"] <= 0.33
    f["adx_act"]   = f["adx14"] > 20.0
    f["vol_surge"] = f["rel_vol"] >= VOL_SURGE
    a = f["atr14"]
    f["liq_p20"] = (np.minimum((f["close"]-f["p20_hi"]).abs(), (f["close"]-f["p20_lo"]).abs())/a) <= LIQ_ATR
    f["liq_pd"]  = (np.minimum((f["close"]-f["pd_hi"]).abs(),  (f["close"]-f["pd_lo"]).abs()) /a) <= LIQ_ATR
    f["liq_any"] = f["liq_p20"] | f["liq_pd"]
    f["session"] = [session_of(ts.hour) for ts in f.index]
    f["hour"]    = f.index.hour
    f["dow"]     = f.index.dayofweek

    fwd = (f["close"].shift(-H_STUDY) - f["close"])/a
    f["fwd_abs"] = fwd.abs(); f["fwd_dir"] = np.sign(fwd)
    frames[p] = f

sel_fwd = pd.concat([f["fwd_abs"][f.index < HOLDOUT_START] for f in frames.values()]).dropna()
BIG_THR = float(sel_fwd.quantile(BIG_Q))
for p in PAIRS: frames[p]["big"] = frames[p]["fwd_abs"] >= BIG_THR
print(f"  big6h threshold (P{BIG_Q:.0f}, selection): {BIG_THR:.2f} ATR")

# ── §1 WHEN — big moves by birth hour/session (selection) ────────────────────
sel_g = pd.concat([f[["session","hour","dow","big","fwd_abs","fwd_dir","squeeze","adx_act",
                      "vol_surge","liq_any","liq_p20","liq_pd"]].assign(pair=p, ts=f.index)
                   for p,f in frames.items()], ignore_index=True)
sel_g = sel_g[(sel_g["ts"] < HOLDOUT_START)].dropna(subset=["big"])

base = sel_g["big"].mean()
print(f"\n{SEP2}\n  1) WHEN: P(big 6h move) by birth session & hour (selection, base={base*100:.1f}%)\n{SEP2}")
print("    session     n      P(big)   lift   median|fwd6h|/ATR")
for s in ["ASIA","LONDON","NY_OVERLAP","LATE_US","ROLLOVER"]:
    g = sel_g[sel_g["session"]==s]
    print(f"    {s:<10} {len(g):>6}   {g['big'].mean()*100:5.1f}%   {g['big'].mean()/base:4.2f}x   {g['fwd_abs'].median():5.2f}")
print("\n    hour (UTC)  n      P(big)   lift   [peak birth hours = when]")
for h in range(24):
    g = sel_g[sel_g["hour"]==h]
    bar = '#' * int(round(g['big'].mean()/base*10))
    print(f"    {h:02d}:00     {len(g):>6}   {g['big'].mean()*100:5.1f}%   {g['big'].mean()/base:4.2f}x  {bar}")

# ── §2 WHAT — condition lift + confluence (selection) ────────────────────────
print(f"\n{SEP2}\n  2) WHAT: condition lift (selection, 6h label)\n{SEP2}")
conds = {"SQUEEZE (bb_rank≤.33)":"squeeze","ADX>20 (trend active)":"adx_act",
         "VOL SURGE (rel≥1.3)":"vol_surge","NEAR P20 pool":"liq_p20",
         "NEAR PD level":"liq_pd","NEAR any pool":"liq_any"}
rows=[]
print(f"    {'condition':<24}{'n':>8}{'P(big|c)':>10}{'lift':>7}{'P(up-big)':>11}{'P(dn-big)':>11}")
for nm,c in conds.items():
    g1 = sel_g[sel_g[c].astype(bool)]
    lift = g1["big"].mean()/base
    up = ((g1["fwd_dir"]>0)&g1["big"]).sum()/max(g1["big"].sum(),1)
    dn = ((g1["fwd_dir"]<0)&g1["big"]).sum()/max(g1["big"].sum(),1)
    rows.append(dict(condition=nm,n=len(g1),p_big=g1["big"].mean(),lift=lift,p_up=up,p_dn=dn))
    print(f"    {nm:<24}{len(g1):>8}{g1['big'].mean()*100:>9.1f}%{lift:>6.2f}x{up*100:>10.0f}%{dn*100:>10.0f}%")
pd.DataFrame(rows).to_csv(os.path.join(OUT,"f010_alignment_map.csv"), index=False)

sel_g["conf"] = (sel_g["squeeze"].astype(int)+sel_g["adx_act"].astype(int)
                +sel_g["vol_surge"].astype(int)+sel_g["liq_any"].astype(int)
                +sel_g["session"].isin(["LONDON","NY_OVERLAP"]).astype(int))
print("\n    confluence score → P(big 6h)  [H2 monotone?]")
for k,g in sel_g.groupby("conf"):
    print(f"    score {k}: n={len(g):>6}  P(big)={g['big'].mean()*100:5.1f}%  lift={g['big'].mean()/base:4.2f}x")

# ── §3 WHY — structural traps vs clean breaks (1H birth study, selection) ────
print(f"\n{SEP2}\n  3) WHY: structural sweep-and-fail vs clean break (selection)\n{SEP2}")
ev_parts=[]
for p,f in frames.items():
    hi=f["high"].values; lo=f["low"].values; cl=f["close"].values; atr=f["atr14"].values
    lup = f["pd_hi"].values; ldn = f["pd_lo"].values
    u20 = f["p20_hi"].values; d20 = f["p20_lo"].values
    rows=[]
    for i in range(21,len(f)):
        if np.isnan(atr[i]) or atr[i]<=0: continue
        dth = DEPTH*atr[i]
        for lvl,(Lu,Ld) in {"PD":(lup[i],ldn[i]),"P20":(u20[i],d20[i])}.items():
            if np.isnan(Lu) or np.isnan(Ld): continue
            bear = lo[i] < Ld-dth and cl[i] > Ld      # swept lows → failed down (long trap)
            bull = hi[i] > Lu+dth and cl[i] < Lu      # swept highs → failed up (short trap)
            brku = cl[i] > Lu+dth and lo[i] > Lu      # clean break up
            brkd = cl[i] < Ld-dth and hi[i] < Ld      # clean break down
            if bear: rows.append(dict(pair=p,ts=f.index[i],level=lvl,kind="BEAR_TRAP"))
            elif bull: rows.append(dict(pair=p,ts=f.index[i],level=lvl,kind="BULL_TRAP"))
            elif brku: rows.append(dict(pair=p,ts=f.index[i],level=lvl,kind="BRK_UP"))
            elif brkd: rows.append(dict(pair=p,ts=f.index[i],level=lvl,kind="BRK_DN"))
    ev_parts.append(pd.DataFrame(rows))
ev = pd.concat(ev_parts, ignore_index=True)
ev["lvl_ord"]=ev["level"].map({"PD":0,"P20":1})
ev = ev.sort_values(["pair","ts","lvl_ord"]).drop_duplicates(["pair","ts"]).reset_index(drop=True)
print(f"    structural events (sel+hol): {len(ev)}  {ev['kind'].value_counts().to_dict()}")

lab = pd.concat([f[["big","fwd_dir","session","squeeze","adx_act","vol_surge","liq_any","atr14"]]
                 .assign(pair=p, ts=f.index) for p,f in frames.items()], ignore_index=True)
ev = ev.merge(lab, on=["pair","ts"], how="left").dropna(subset=["big"])
ev_sel = ev[ev["ts"] < HOLDOUT_START]
print(f"    {'event':<12}{'n(sel)':>8}{'P(big6h)':>10}{'lift':>7}{'P(up-big)':>11}{'P(dn-big)':>11}")
for k in ["BEAR_TRAP","BULL_TRAP","BRK_UP","BRK_DN"]:
    g = ev_sel[ev_sel["kind"]==k]
    if len(g)==0: continue
    up=((g["fwd_dir"]>0)&g["big"]).sum()/max(g["big"].sum(),1)
    dn=((g["fwd_dir"]<0)&g["big"]).sum()/max(g["big"].sum(),1)
    print(f"    {k:<12}{len(g):>8}{g['big'].mean()*100:>9.1f}%{g['big'].mean()/base:>6.2f}x{up*100:>10.0f}%{dn*100:>10.0f}%")
# asymmetry note: direction of the move after a trap — fade direction dominance
for k,side in [("BEAR_TRAP",1),("BULL_TRAP",-1)]:
    g = ev_sel[ev_sel["kind"]==k]
    if len(g): print(f"    {k}: big moves land in fade direction {((g['fwd_dir']==side)&g['big']).sum()/max(g['big'].sum(),1)*100:.0f}% of the time (n_big={int(g['big'].sum())})")

# ── §4 TRADE — 4H execution battery (F007 mirror) ────────────────────────────
print(f"\n{SEP2}\n  4) TRADE: 4H execution (F007 mirror) — aligned structural traps\n{SEP2}")
print("  Building 4H frames …")
frames4 = {}
for p in PAIRS:
    df = pd.read_parquet(f"{FOREX_DIR}/{p}_1H.parquet")
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    if "vol" not in df.columns: df["vol"] = (df["high"]-df["low"]).clip(lower=1e-12)
    df4 = df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()
    f4 = add_features(df4)
    f4.dropna(subset=["ema200","ema50","ema20","atr14","adx14","bb_width","real_vol_20"], inplace=True)
    day4 = f4.index.normalize()
    typ4 = (f4["high"]+f4["low"]+f4["close"])/3.0
    f4["vwap"] = (typ4*f4["vol"]).groupby(day4).cumsum()/f4["vol"].groupby(day4).cumsum().replace(0,np.nan)
    f4["p20_hi"] = f4["high"].rolling(20).max().shift(1)
    f4["p20_lo"] = f4["low"].rolling(20).min().shift(1)
    d4 = f4.resample("1D").agg({"high":"max","low":"min"}).shift(1)
    f4["pd_hi"] = d4["high"].reindex(day4).values; f4["pd_lo"] = d4["low"].reindex(day4).values
    f4["bb_rank"]   = rrank(f4["bb_width"])
    f4["squeeze"]   = f4["bb_rank"] <= 0.33
    f4["adx_act"]   = f4["adx14"] > 20.0
    f4["vol_surge"] = f4["rel_vol"] >= VOL_SURGE
    a4 = f4["atr14"]
    f4["liq_any"] = ((np.minimum((f4["close"]-f4["p20_hi"]).abs(),(f4["close"]-f4["p20_lo"]).abs())/a4 <= LIQ_ATR)
                   | (np.minimum((f4["close"]-f4["pd_hi"]).abs(),(f4["close"]-f4["pd_lo"]).abs())/a4 <= LIQ_ATR))
    f4["session"] = [session_of(ts.hour) for ts in f4.index]
    frames4[p] = f4

print("  4H outcomes (SL1·TP3·ATR, 24 bars, TP-before-SL) …")
out4 = {}
for p,f in frames4.items():
    cl=f["close"].values; hi=f["high"].values; lo=f["low"].values; atr=f["atr14"].values; n=len(f)
    lo_arr=np.full(n,np.nan); so_arr=np.full(n,np.nan)
    for i in range(n):
        if np.isnan(atr[i]) or atr[i]<=0: continue
        end=min(i+24,n)
        tp=cl[i]+RR*atr[i]; sl=cl[i]-SLM*atr[i]; hit=None
        for b in range(i+1,end):
            if hi[b]>=tp: hit=RR; break
            if lo[b]<=sl: hit=-1.0; break
        if hit is not None: lo_arr[i]=hit
        elif end>i+1: lo_arr[i]=(cl[end-1]-cl[i])/atr[i]
        tps=cl[i]-RR*atr[i]; sls=cl[i]+SLM*atr[i]; hit=None
        for b in range(i+1,end):
            if lo[b]<=tps: hit=RR; break
            if hi[b]>=sls: hit=-1.0; break
        if hit is not None: so_arr[i]=hit
        elif end>i+1: so_arr[i]=(cl[i]-cl[end-1])/atr[i]
    out4[p] = pd.DataFrame({"long_out":lo_arr,"short_out":so_arr}, index=f.index)

print("  4H structural traps …")
ev4_parts=[]
for p,f in frames4.items():
    hi=f["high"].values; lo=f["low"].values; cl=f["close"].values; atr=f["atr14"].values
    lvls={"PD":(f["pd_hi"].values,f["pd_lo"].values),"P20":(f["p20_hi"].values,f["p20_lo"].values)}
    rows=[]
    for tname,(lup,ldn) in lvls.items():
        for i in range(21,len(f)):
            if np.isnan(atr[i]) or atr[i]<=0 or np.isnan(lup[i]) or np.isnan(ldn[i]): continue
            dth=DEPTH*atr[i]
            if   lo[i] < ldn[i]-dth and cl[i] > ldn[i]: rows.append(dict(pair=p,ts=f.index[i],level=tname,kind="BEAR_TRAP"))
            elif hi[i] > lup[i]+dth and cl[i] < lup[i]: rows.append(dict(pair=p,ts=f.index[i],level=tname,kind="BULL_TRAP"))
    ev4_parts.append(pd.DataFrame(rows))
ev4 = pd.concat(ev4_parts, ignore_index=True)
ev4["lvl_ord"]=ev4["level"].map({"PD":0,"P20":1})
ev4 = ev4.sort_values(["pair","ts","lvl_ord"]).drop_duplicates(["pair","ts"]).reset_index(drop=True)

lab4 = pd.concat([f[["squeeze","adx_act","vol_surge","liq_any","session"]].assign(pair=p,ts=f.index)
                  for p,f in frames4.items()], ignore_index=True)
o4   = pd.concat([out4[p].assign(pair=p,ts=out4[p].index) for p in PAIRS], ignore_index=True)
a4m  = pd.concat([pd.DataFrame({"pair":p,"ts":frames4[p].index,"atr":frames4[p]["atr14"].values}) for p in PAIRS])
ev4 = ev4.merge(lab4,on=["pair","ts"],how="left").merge(o4,on=["pair","ts"],how="left").merge(a4m,on=["pair","ts"],how="left")
ev4["spread"] = ev4["pair"].map(SPREAD)
ev4["side"]   = np.where(ev4["kind"]=="BEAR_TRAP", 1, -1)
ev4["r"]      = np.where(ev4["side"]==1, ev4["long_out"], ev4["short_out"])
ev4["conf"]   = (ev4["squeeze"].astype(int)+ev4["adx_act"].astype(int)
                +ev4["vol_surge"].astype(int)+ev4["liq_any"].astype(int)
                +ev4["session"].isin(["LONDON","NY_OVERLAP"]).astype(int))
ev4 = ev4.dropna(subset=["r"]).reset_index(drop=True)
print(f"  4H structural traps with outcomes: {len(ev4)}")

def cost_r(sub, mult=1.0, swap=False):
    c = 2*mult*sub["spread"].values/np.maximum(sub["atr"].values,1e-12)
    if swap:
        c = c + (SWAP_PIP_DAY*sub["pair"].map(PIP).values)/np.maximum(sub["atr"].values,1e-12)*(24/24.0)
    return sub["r"].values - c

def battery(name, sub):
    s, h = sub[sub["ts"]<HOLDOUT_START], sub[sub["ts"]>=HOLDOUT_START]
    print(f"\n  {name}")
    print(f"    selection : n={len(s):>5}  PF={pf_of(s['r']):5.2f}  PF@cost={pf_of(cost_r(s)):5.2f}")
    print(f"    holdout*  : n={len(h):>5}  PF={pf_of(h['r']):5.2f}  PF@cost={pf_of(cost_r(h)):5.2f}   (*development holdout — seen via v1)")
    if len(h) < 30:
        print("    (holdout too small — no battery)"); return
    rs_c = cost_r(h)
    out=np.empty(2000)
    for b in range(2000):
        x=rs_c[RNG.integers(0,len(rs_c),len(rs_c))]; out[b]=pf_of(x)
    print(f"    boot CI hol PF@cost: P5={np.percentile(out,5):.2f} med={np.median(out):.2f} P95={np.percentile(out,95):.2f}")
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

base_traps = ev4  # all structural traps, all sessions, no confluence
battery("4H ALL structural traps (fade, no filter)", base_traps)
battery("4H conf≥3 (fade)", ev4[ev4["conf"]>=3])
battery("4H conf≥4 (fade)", ev4[ev4["conf"]>=4])

print(f"\n{SEP2}\n  5) 4H holdout anatomy (descriptive)\n{SEP2}")
h4 = ev4[ev4["ts"]>=HOLDOUT_START]
for c,nmc in [("squeeze","squeeze"),("adx_act","adx>20"),("vol_surge","vol-surge"),("liq_any","near-liq")]:
    g=h4[h4[c].astype(bool)]
    if len(g)>=15: print(f"    {nmc:<11} n={len(g):>4} PF@cost={pf_of(cost_r(g)):5.2f}")
for ses in ["ASIA","LONDON","NY_OVERLAP","LATE_US","ROLLOVER"]:
    g=h4[h4["session"]==ses]
    if len(g)>=15: print(f"    {ses:<11} n={len(g):>4} PF@cost={pf_of(cost_r(g)):5.2f}")
for k in range(6):
    g=h4[h4["conf"]==k]
    if len(g)>=15: print(f"    conf={k}       n={len(g):>4} PF@cost={pf_of(cost_r(g)):5.2f}")

ev4.to_csv(os.path.join(OUT,"f010_trap_events_4h.csv"), index=False)
ev.to_csv(os.path.join(OUT,"f010_trap_events_1h.csv"), index=False)
print(f"\n  Done in {time.time()-t0:.0f}s → quantlab_output/f010_*")
