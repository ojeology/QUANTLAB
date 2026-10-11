"""
QUANTLAB AI — R100
CERTAINTY & SMOOTHNESS AUDIT: is the champion a lone spike or a plateau?
And can we halve the drawdown without killing the return?

PART A — ROBUSTNESS NEIGHBORHOOD (a real edge lives on a plateau, not a spike):
  RR {1.2, 1.5, 2.0} x q {0.45, 0.55, 0.65} on 2026 holdout @cost.
  Every cell gets its own walk-forward SVM (win labels change with RR).
  Also bracket-scale {0.8, 1.2} (same 1.5:1 ratio, wider/narrower stops).
  DESCRIPTIVE ONLY — the champion stays q0.55/RR1.5; this tests whether nearby
  settings also work (robustness = understanding = certainty).

PART B — ANTI-DRAWDOWN THROTTLE (mechanical, no mining):
  risk/trade = 1.00% when equity within 5% of its peak
             = 0.50% when drawdown > 5%
             = 0.25% when drawdown > 10%
  restore automatically at new equity high. Tested on 2026 + fresh vs flat 1%.
  Target: MDD cut in half, Calmar (ann.return / |MDD|) up.

PART C — STABILITY ANATOMY: monthly P&L, hour-of-day, weekday — where does the
  edge live? (understanding, in plain words)
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, build_signal_mask, sim_symbol, stats_from_trades,
    pf_of_rs, IS_LOOKBACK, RECAL_EVERY,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CACHE = CONFIG["CACHE_FOLDER"]
OUT = CONFIG["OUTPUT_FOLDER"]; os.makedirs(OUT, exist_ok=True)
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
FRESH_START   = pd.Timestamp("2026-08-08", tz="UTC")
RNG = np.random.default_rng(42)
SEP  = "═" * 110
SEP2 = "─" * 90
COST = 0.0005

BASE_FEATS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
              "rel_vol","bb_width","real_vol_20","hour","dow"]
SPECIAL = ["breadth_q", "dist_hi48", "green_streak"]
FEATS = BASE_FEATS + SPECIAL

print(); print(SEP)
print("  QUANTLAB AI — R100  certainty & smoothness audit")
print(SEP)
t0 = time.time()

# ── LOAD ─────────────────────────────────────────────────────────────────────
print("\n  Loading 73 symbols …")
feats = {}
for sym in os.listdir(CACHE):
    if not sym.endswith("_1H.parquet"): continue
    sym = sym.replace("_1H.parquet","")
    try:
        df = pd.read_parquet(os.path.join(CACHE, f"{sym}_1H.parquet"))
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY: feats[sym] = f
    except Exception: pass
print(f"  symbols: {len(feats)}")

above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100
cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
mask = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats.items()}

def ml_rows(trades):
    rows=[]
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        f = feats[t["sym"]]; ts = t["entry_time"]
        row = f.loc[ts]; i = f.index.get_loc(ts)
        c = float(row["close"])
        hi48 = float(f["close"].rolling(48).max().iloc[i])
        streak = 0
        for k in range(6):
            j = i-k
            if j < 0: break
            if f["close"].iloc[j] > f["open"].iloc[j]: streak += 1
            else: break
        bq = float(breadth_pct.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) if ts >= breadth_pct.index[0] else 50.0
        rows.append(dict(sym=t["sym"], ts=ts, r=t["r"], win=int(t["r"]>0),
                         entry=t["entry"], atr=t["atr"],
                         **{c_: row.get(c_,0) for c_ in BASE_FEATS},
                         breadth_q=bq, dist_hi48=(c/hi48-1)*100 if hi48>0 else 0.0,
                         green_streak=streak))
    return pd.DataFrame(rows)

def champion_for(rr, sl_mult=1.0):
    trs=[]
    for sym, f in feats.items():
        for t in sim_symbol(f, mask[sym], rr, dict(entry_next=False, exit="base", hours=None, sl_mult=sl_mult)):
            t["sym"]=sym; trs.append(t)
    trs.sort(key=lambda t: t["entry_time"])
    m = ml_rows(trs)
    X = m[FEATS].fillna(0).values; y = m["win"].values
    pred = np.full(len(m), np.nan); sc = StandardScaler()
    for i in range(150, len(m)):
        clf = SVC(C=1.0, gamma="scale", probability=True)
        clf.fit(sc.fit_transform(X[:i]), y[:i])
        pred[i] = clf.predict_proba(sc.transform(X[i:i+1]))[0,1]
    sel = m["ts"] < HOLDOUT_START
    thr = float(pd.Series(pred).where(sel.values).dropna().quantile(1-0.55))
    m["pred"]=pred; m["kept"]= np.where(np.isnan(pred), False, pred>=thr)
    m["r_net"] = m["r"] - 2*COST*m["entry"]/m["atr"]
    return m

def metrics(sub):
    if len(sub)==0: return dict(n=0, pf=np.nan, mdd=0.0, calmar=np.nan, tot=0.0)
    eq = (1 + 0.01*sub["r_net"]).cumprod()
    dd = ((eq-eq.cummax())/eq.cummax()).min()
    days = max((sub["ts"].max()-sub["ts"].min()).days, 1)
    ann = (eq.iloc[-1])**(365.0/days)-1 if eq.iloc[-1]>0 else -1
    return dict(n=len(sub), pf=pf_of_rs(sub["r_net"].values), mdd=dd,
                calmar=(ann/abs(dd) if dd<0 else np.nan), tot=eq.iloc[-1]-1)

# ── PART A: NEIGHBORHOOD ─────────────────────────────────────────────────────
print(f"\n{SEP2}\n  A) ROBUSTNESS NEIGHBORHOOD (2026 holdout PF@cost; own walk-forward SVM per RR)\n{SEP2}")
print(f"    {'':<8}", end="")
for q in [0.45,0.55,0.65]: print(f"q{q:<8}", end="")
print(f"{'fresh(q.55)':>12}")
plateau={}
for rr in [1.2, 1.5, 2.0]:
    m = champion_for(rr)
    row=[]
    for q in [0.45,0.55,0.65]:
        thrq = float(pd.Series(m["pred"]).where((m["ts"]<HOLDOUT_START).values).dropna().quantile(1-q))
        kept = m[np.where(np.isnan(m["pred"]), False, m["pred"]>=thrq) & (m["ts"]>=HOLDOUT_START)]
        mm = metrics(kept); row.append(mm["pf"])
    kept55 = m[m["kept"] & (m["ts"]>=FRESH_START)]
    fres = metrics(kept55)["pf"]
    plateau[rr]=row
    print(f"    RR{rr:<5}" + "".join(f"{v:<9.2f}" for v in row) + f"{fres:>12.2f}")

print(f"\n    bracket-scale (RR1.5, q0.55, own SVM):")
for sm in [0.8, 1.2]:
    m = champion_for(1.5, sl_mult=sm)
    kept = m[m["kept"] & (m["ts"]>=HOLDOUT_START)]
    mm = metrics(kept)
    print(f"    SLx{sm}: n={mm['n']:>3}  PF@c={mm['pf']:.2f}  MDD={mm['mdd']*100:.1f}%")

# ── PART B: DD THROTTLE ──────────────────────────────────────────────────────
print(f"\n{SEP2}\n  B) ANTI-DRAWDOWN THROTTLE (champion q0.55/RR1.5)\n{SEP2}")
m = champion_for(1.5)
kept = m[m["kept"]].sort_values("ts").reset_index(drop=True)

def run(throttle):
    cap=1.0; peak=1.0; out=[]; tiers=[]
    for _,t in kept.iterrows():
        dd = 1-cap/peak
        if not throttle: w=0.01
        elif dd>0.10: w=0.0025
        elif dd>0.05: w=0.005
        else: w=0.01
        tiers.append(w)
        cap *= (1 + w*t["r_net"])
        peak = max(peak, cap)
        out.append((t["ts"], cap))
    eqs = pd.Series(dict(out)).sort_index()
    dd = ((eqs-eqs.cummax())/eqs.cummax()).min()
    days = max((eqs.index.max()-eqs.index.min()).days,1)
    ann = eqs.iloc[-1]**(365.0/days)-1
    tiers=np.array(tiers)
    return dict(ret=eqs.iloc[-1]-1, mdd=dd, calmar=ann/abs(dd) if dd<0 else np.nan,
                throttle_pct=(tiers<0.01).mean()*100)

for name, thr in [("flat 1.0% (validated)", False), ("DD throttle 5%/10%", True)]:
    r = run(thr)
    print(f"    {name:<24} total={r['ret']*100:+7.1f}%  MDD={r['mdd']*100:6.1f}%  Calmar={r['calmar']:5.2f}  time-throttled={r['throttle_pct']:.0f}%")

# fresh-window comparison
hol = kept[kept["ts"]>=FRESH_START]
rs = hol["r_net"].values
for w in [0.01, 0.005]:
    cap=1.0
    for x in rs: cap*=(1+w*x)
    print(f"    fresh window @ {w*100:.1f}%/trade: {cap:.3f}x")

# ── PART C: STABILITY ANATOMY ────────────────────────────────────────────────
print(f"\n{SEP2}\n  C) WHERE THE EDGE LIVES (champion, all kept trades net of cost)\n{SEP2}")
mm = kept.copy(); mm["month"]=mm["ts"].dt.to_period("M")
g = mm.groupby("month")["r_net"].sum()
print("    monthly P&L (R):")
for k,v in g.items(): print(f"      {k}: {v:+.2f}")
print(f"    positive months: {(g>0).sum()}/{len(g)}  worst: {g.min():+.2f}R  best: {g.max():+.2f}R")
mm["hr"]=mm["ts"].dt.hour
print("    by entry hour (4H bars → 0/4/8/12/16/20 UTC):")
for h,gg in mm.groupby("hr"):
    if len(gg)>=5: print(f"      {h:02d}h: n={len(gg):>3}  PF@c={pf_of_rs(gg['r_net'].values):.2f}")
mm["dow"]=mm["ts"].dt.dayofweek
print("    by weekday:", {int(d): round(pf_of_rs(gg['r_net'].values),2) for d,gg in mm.groupby("dow") if len(gg)>=5})

kept.to_csv(os.path.join(OUT,"r100_champion_trades.csv"), index=False)
print(f"\n  Done in {time.time()-t0:.0f}s → quantlab_output/r100_*")
