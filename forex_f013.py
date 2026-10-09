"""
FOREX HUNT — F013
REALISTIC EDGE: the two candidates with actual economic footing, on true data.

Everything at 1H/4H with SL/TP brackets died at retail costs (F009-F012, 10y).
F013 tests the two remaining realistic families, prespecified, no new mining:

  M1  DAILY TIME-SERIES MOMENTUM (20d/60d/120d, MOS-style weekly legs)
      Rationale: the most documented FX premium (trend). Costs on daily ATR are
      ~0.02R vs ~0.10R intraday. Tradeable manually, weekly routine.
      Selection 2003-2021 · holdout 2022-2026 (virgin for this family).
      Swap NOT modeled (no data) — disclosed; sensitivity ±0.5 pip/day shown.
  M2  DAILY EMA50/200 CROSSOVER — the retail classic, tested honestly.
      Segment trades: enter next 1H close after cross, exit after reverse cross.
  F1  POST-EXPANSION FADE — validated clock (F012: big moves born 02-06 UTC):
      fade the morning move at 06:00, TP/SL 1·ATR1h, time exit 15:00.
  F2  TOKYO-RANGE BREAKOUT — aligned WITH the clock: range 00-06 UTC,
      close-break entry 06-15, TP 1×range / SL 0.5×range, exit 15:00.
      Selection < 2024-04 · holdout ≥ 2024-04.

Gates: holdout PF@cost > 1.10 · LOO floor > 1.00 · n ≥ 200 (M2: n ≥ 50).
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG

F10 = os.path.join(CONFIG["CACHE_FOLDER"], "forex10")
OUT = CONFIG["OUTPUT_FOLDER"]; os.makedirs(OUT, exist_ok=True)

PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP"]
SPREAD = {"EURUSD":0.00006,"GBPUSD":0.00010,"USDJPY":0.010,"AUDUSD":0.00008,
          "USDCAD":0.00010,"USDCHF":0.00010,"NZDUSD":0.00012,"EURGBP":0.00010}
PIP = {p:(0.01 if "JPY" in p else 0.0001) for p in PAIRS}

H_START_D = pd.Timestamp("2022-01-01", tz="UTC")   # daily holdout
H_START_I = pd.Timestamp("2024-04-01", tz="UTC")   # intraday holdout (post F010 peek)

RNG = np.random.default_rng(42)
SEP  = "=" * 110
SEP2 = "-" * 90

def pf_of(rs):
    rs = np.asarray(rs, float); rs = rs[~np.isnan(rs)]
    if len(rs) == 0: return float("nan")
    w = rs[rs > 0].sum(); lo = abs(rs[rs < 0].sum())
    return w / lo if lo > 0 else (99.0 if w > 0 else 1.0)

print(); print(SEP)
print("  FOREX HUNT — F013  REALISTIC EDGE (daily momentum + validated-clock trades)")
print(SEP)
t0 = time.time()

# ── LOAD 1H, build DAILY ─────────────────────────────────────────────────────
print("\n  Loading 1H + building daily bars …")
h1, d1 = {}, {}
for p in PAIRS:
    df = pd.read_parquet(f"{F10}/{p}_1H.parquet")
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    df = df[(df["open"]>0)&(df["low"]>0)]
    df["hour"] = df.index.hour
    h1[p] = df
    dd = df.resample("1D").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    tr = np.maximum(dd["high"]-dd["low"], np.maximum((dd["high"]-dd["close"].shift()).abs(),
                                                     (dd["low"]-dd["close"].shift()).abs()))
    dd["atr14"] = tr.rolling(14).mean()
    dd["ema50"]  = dd["close"].ewm(span=50,  adjust=False).mean()
    dd["ema200"] = dd["close"].ewm(span=200, adjust=False).mean()
    d1[p] = dd
print(f"  daily bars: {min(len(v) for v in d1.values())}–{max(len(v) for v in d1.values())} per pair")

# ── M1: WEEKLY TSMOM LEGS ────────────────────────────────────────────────────
def tsmom_legs(L):
    rows=[]
    for p in PAIRS:
        dd = d1[p]
        sig = np.sign(dd["close"] - dd["close"].shift(L))          # causal, daily close
        iso = dd.index.isocalendar()
        ddf = dd.assign(yr=iso.year, wknum=iso.week, sig=sig)
        prev_sig = np.nan
        for (y,w), g in ddf.groupby(["yr","wknum"]):
            if len(g) >= 3 and np.isfinite(prev_sig) and prev_sig != 0:
                wmask = (h1[p].index >= g.index[0]) & (h1[p].index <= g.index[-1] + pd.Timedelta(hours=23))
                wh = h1[p][wmask]
                if len(wh) >= 12:
                    e, x = wh["close"].iloc[0], wh["close"].iloc[-1]
                    atr = dd["atr14"].get(g.index[0], np.nan)
                    if np.isfinite(atr) and atr > 0 and e > 0:
                        rows.append(dict(pair=p, ts=g.index[0], r=prev_sig*(x-e)/atr,
                                         cost=2*SPREAD[p]/atr,
                                         days=(g.index[-1]-g.index[0]).days+1))
            prev_sig = g["sig"].iloc[-1]
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

# ── M2: EMA CROSSOVER SEGMENTS ───────────────────────────────────────────────
def seg_legs():
    rows=[]
    for p in PAIRS:
        dd = d1[p].dropna(subset=["ema50","ema200","atr14"])
        state = np.sign(dd["ema50"] - dd["ema200"]).fillna(0)
        chg = state.diff().fillna(0)
        starts = dd.index[(chg != 0) & (state != 0)]
        for i, t0s in enumerate(starts):
            t1s = starts[i+1] if i+1 < len(starts) else dd.index[-1]
            s = state.get(t0s)
            wh_e = h1[p][h1[p].index > t0s]
            if len(wh_e) == 0: continue
            e = wh_e["close"].iloc[0]
            wh_x = h1[p][h1[p].index > t1s]
            x = wh_x["close"].iloc[0] if len(wh_x) > 0 else h1[p]["close"].iloc[-1]
            atr = dd["atr14"].get(t0s, np.nan)
            if not (np.isfinite(atr) and atr > 0 and e > 0): continue
            days = max((t1s - t0s).days, 1)
            rows.append(dict(pair=p, ts=t0s, r=s*(x-e)/atr, cost=2*SPREAD[p]/atr, days=days))
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

def battery_daily(name, legs, min_n=200):
    s, h = legs[legs["ts"]<H_START_D], legs[legs["ts"]>=H_START_D]
    rs_s = s["r"].values - s["cost"].values
    rs_h = h["r"].values - h["cost"].values
    print(f"\n  {name}")
    print(f"    selection : n={len(s):>6}  PF={pf_of(s['r']):5.2f}  PF@cost={pf_of(rs_s):5.2f}")
    print(f"    holdout   : n={len(h):>6}  PF={pf_of(h['r']):5.2f}  PF@cost={pf_of(rs_h):5.2f}")
    if len(h) < min_n or len(s) < min_n:
        print("    (insufficient n)"); return
    floor=9.9
    for p in PAIRS:
        v = h[h["pair"]!=p]
        if len(v) > 10: floor=min(floor, pf_of(v["r"].values - v["cost"].values))
    months = max((h["ts"].max()-h["ts"].min()).days/30.4, 1)
    print(f"    LOO floor: {floor:.2f}   avg net R/leg: {np.mean(rs_h):+.3f}   legs/mo: {len(h)/months:.0f}")
    outb=np.empty(2000)
    for b in range(2000):
        x=rs_h[RNG.integers(0,len(rs_h),len(rs_h))]; outb[b]=pf_of(x)
    print(f"    boot CI hol PF@cost: P5={np.percentile(outb,5):.2f} med={np.median(outb):.2f} P95={np.percentile(outb,95):.2f}")
    atr_h = np.array([d1[pp]["atr14"].asof(ts) for pp, ts in zip(h["pair"], h["ts"])])
    pip_h = np.array([PIP[pp] for pp in h["pair"]])
    for swpip in [+0.5, -0.5]:
        sw_r = swpip * pip_h * h["days"].values / atr_h
        print(f"    swap sens [{swpip:+.1f} pip/day]: hol PF(net)={pf_of(rs_h - sw_r):.2f}")
    gate = (pf_of(rs_h)>1.10) and (floor>1.00)
    print(f"    GATE (PF@cost>1.10, LOO>1.0): {'✅ PASS' if gate else '❌ FAIL'}")

for L in [20, 60, 120]:
    battery_daily(f"M1 TSMOM {L}d — weekly legs", tsmom_legs(L))
battery_daily("M2 EMA50/200 crossover segments", seg_legs(), min_n=50)

# ── F1/F2: CLOCK TRADES (1H) ─────────────────────────────────────────────────
print(f"\n{SEP2}\n  F1/F2) CLOCK TRADES (selection <2024-04, holdout ≥2024-04) — 1H, retail costs\n{SEP2}")
def atr14_1h(df):
    tr = np.maximum(df["high"]-df["low"], np.maximum((df["high"]-df["close"].shift()).abs(),
                                                     (df["low"]-df["close"].shift()).abs()))
    return tr.rolling(14).mean()

rows_f1=[]; rows_f2=[]; rows_f2sl=[]; rows_f2nb=[]; rows_f1sl=[]
for p in PAIRS:
    df = h1[p].copy(); df["atr"] = atr14_1h(df); df["ddate"] = df.index.normalize()
    m = df[df["hour"].between(0,5)].groupby("ddate").agg(
        mopen=("open","first"), mhi=("high","max"), mlo=("low","min"))
    b6 = df[df["hour"]==6].groupby("ddate")["close"].first()
    a6 = df[df["hour"]==6].groupby("ddate")["atr"].first()
    day = m.join(b6.rename("c6")).join(a6.rename("a6")).dropna()
    day["morning_move"] = (day["c6"]-day["mopen"])/day["a6"]
    day["rng"] = (day["mhi"]-day["mlo"])/day["a6"]
    sel_thr = day[day.index < H_START_I]["morning_move"].abs().quantile(0.90)
    for d0, row in day.iterrows():
        win = df[(df["ddate"]==d0) & (df["hour"]>=7) & (df["hour"]<15)]   # first FULL bar after 06:00 close entry
        if len(win) < 3 or not np.isfinite(row["a6"]) or row["a6"]<=0: continue
        a = row["a6"]; cost = 2*SPREAD[p]/a
        # F1 fade big mornings (with SL-first adversarial variant)
        if abs(row["morning_move"]) >= sel_thr:
            e = row["c6"]; s = -np.sign(row["morning_move"])
            tp, sl = e + s*a, e - s*a
            r = np.nan; r_slfirst = np.nan
            for ts, b in win.iterrows():
                if b["low"]  <= sl: r_slfirst = -1.0
                if b["high"] >= tp: r = 1.0; break
                if b["low"]  <= sl: r = -1.0; break
            if np.isnan(r): r = s*(win["close"].iloc[-1]-e)/a
            if np.isnan(r_slfirst): r_slfirst = r
            rows_f1.append(dict(pair=p, ts=d0+pd.Timedelta(hours=6), r=r, cost=cost))
            rows_f1sl.append(dict(pair=p, ts=d0+pd.Timedelta(hours=6), r=r_slfirst, cost=cost))
        # F2 range break
        entry = None
        for ts, b in win.iterrows():
            if b["close"] > row["mhi"]: entry=(ts,1,b["close"]); break
            if b["close"] < row["mlo"]: entry=(ts,-1,b["close"]); break
        if entry is None: continue
        ts, s2, e = entry
        rngp = row["rng"]*a
        tp, sl = e + s2*rngp, e - s2*0.5*rngp
        rest = win[win.index>ts]
        r = np.nan; r_slfirst = np.nan
        for t2, b in rest.iterrows():
            if b["low"]  <= sl: r_slfirst = -1.0
            if b["high"] >= tp: r = 1.0; break
            if b["low"]  <= sl: r = -1.0; break
        if np.isnan(r): r = s2*(rest["close"].iloc[-1]-e)/a if len(rest) else 0.0
        if np.isnan(r_slfirst): r_slfirst = r
        rows_f2.append(dict(pair=p, ts=ts, r=r, cost=cost))
        rows_f2sl.append(dict(pair=p, ts=ts, r=r_slfirst, cost=cost))
        rows_f2nb.append(dict(pair=p, ts=ts, r=s2*(rest["close"].iloc[-1]-e)/a if len(rest) else 0.0, cost=cost))

f1 = pd.DataFrame(rows_f1); f2 = pd.DataFrame(rows_f2)

def battery_intraday(name, legs, tpfirst=None, nobracket=None):
    s, h = legs[legs["ts"]<H_START_I], legs[legs["ts"]>=H_START_I]
    rs_h = h["r"].values - h["cost"].values
    print(f"\n  {name}")
    print(f"    selection : n={len(s):>5}  PF={pf_of(s['r']):5.2f}  PF@cost={pf_of(s['r'].values-s['cost'].values):5.2f}")
    print(f"    holdout   : n={len(h):>5}  PF={pf_of(h['r']):5.2f}  PF@cost={pf_of(rs_h):5.2f}  WR={(h['r']>0).mean()*100:.0f}%  avgR={np.mean(h['r']):+.2f}")
    if tpfirst is not None:
        hh = tpfirst[tpfirst["ts"]>=H_START_I]
        print(f"    SL-first path check: hol PF@cost={pf_of(hh['r'].values-hh['cost'].values):.2f}")
    if nobracket is not None:
        hh = nobracket[nobracket["ts"]>=H_START_I]
        print(f"    no-bracket (direction-only, exit 15:00): hol PF@cost={pf_of(hh['r'].values-hh['cost'].values):.2f} (n={len(hh)})")
    if len(h)<30: print("    (insufficient n)"); return
    floor=9.9
    for p in PAIRS:
        v=h[h["pair"]!=p]
        if len(v)>10: floor=min(floor, pf_of(v["r"].values-v["cost"].values))
    outb=np.empty(2000)
    for b in range(2000):
        x=rs_h[RNG.integers(0,len(rs_h),len(rs_h))]; outb[b]=pf_of(x)
    print(f"    boot CI hol PF@cost: P5={np.percentile(outb,5):.2f} med={np.median(outb):.2f} P95={np.percentile(outb,95):.2f}   LOO floor: {floor:.2f}")
    gate = (len(h)>=200) and (pf_of(rs_h)>1.10) and (floor>1.00)
    print(f"    GATE (n≥200, PF@cost>1.10, LOO>1.0): {'✅ PASS' if gate else '❌ FAIL'}")

f2sl = pd.DataFrame(rows_f2sl); f2nb = pd.DataFrame(rows_f2nb)
f1sl = pd.DataFrame(rows_f1sl)
battery_intraday("F1 post-expansion fade (big morning → fade 06:00→15:00)", f1, tpfirst=f1sl)
battery_intraday("F2 Tokyo-range breakout (00-06 range, break entry 06-15)", f2, tpfirst=f2sl, nobracket=f2nb)

# ── REALITY CHECK ────────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  REALITY CHECK: holdout legs at 0.5% risk/leg, compounding (median path)\n{SEP2}")
def mc_feel(name, rs, yrs):
    if len(rs)<30: return
    prof=[]; dds=[]
    for _ in range(3000):
        x=rs[RNG.integers(0,len(rs),len(rs))]
        cap=100.0; eq=[cap]
        for r in x: cap*=(1+0.005*r); eq.append(cap)
        eq=np.array(eq); pk=np.maximum.accumulate(eq)
        prof.append(cap); dds.append(((eq-pk)/pk).min())
    prof=np.array(prof); dds=np.array(dds)
    med=np.median(prof)
    ann=((med/100.0)**(1.0/yrs)-1)*100
    print(f"    {name:<34} median end={med:6.1f} (≈{ann:+.1f}%/yr at 0.5%/leg)  P(loss)={(prof<100).mean()*100:.0f}%  DD med={np.percentile(dds,50)*100:.1f}% P5={np.percentile(dds,5)*100:.1f}%")

for L in [20,60,120]:
    legs = tsmom_legs(L); h = legs[legs["ts"]>=H_START_D]
    mc_feel(f"M1 TSMOM{L}d", h["r"].values-h["cost"].values, 4.75)
legs = seg_legs(); h = legs[legs["ts"]>=H_START_D]
mc_feel("M2 EMA50/200", h["r"].values-h["cost"].values, 4.75)
h=f1[f1["ts"]>=H_START_I]; mc_feel("F1 fade", h["r"].values-h["cost"].values, 2.5)
h=f2[f2["ts"]>=H_START_I]; mc_feel("F2 breakout", h["r"].values-h["cost"].values, 2.5)

f1.to_csv(os.path.join(OUT,"f013_f1_fade.csv"), index=False)
f2.to_csv(os.path.join(OUT,"f013_f2_breakout.csv"), index=False)
print(f"\n  Done in {time.time()-t0:.0f}s")
