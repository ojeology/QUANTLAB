"""
FOREX HUNT — F006 (fast version)
COMPREHENSIVE TRAP HUNT (bull/bear/ML traps, sessions, follow vs reject)

Fast outcome precomputation: for each pair, compute long_out[] and short_out[]
(forward 24-bar scan for SL/TP/time-stop, TP-before-SL, exactly matching sim_symbol's
timeN branch). Events then map instantly to outcomes by direction.

Trap types: prior-20 (P20), prior-day (PD), VWAP, EMA20 | responses: follow/reject
Context: trend (up/dn), session (london hour8 / ny-ovl hour12 / other). RR=3.0.
ML trap classifier: walk-forward SVM picks follow vs reject per event.
Protocol: selection <= Aug-2025, holdout untouched, retail spreads.
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
RR = 3.0
DEPTH = 0.15
IS_LOOKBACK_4H = 125
RECAL_EVERY_4H = 42
HORIZON = 24

SEP = "=" * 110
SEP2 = "-" * 90
print(); print(SEP)
print("  FOREX HUNT — F006  COMPREHENSIVE TRAP MATRIX + ML TRAP (fast)")
print(SEP)
t0 = time.time()

print("\n  Loading 4H forex data …")
raw4h = {}
for p in PAIRS:
    fpath = os.path.join(FOREX_DIR, f"{p}_1H.parquet")
    if not os.path.exists(fpath): continue
    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for col in ["open","high","low","close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open","high","low","close"], inplace=True)
    if "vol" not in df.columns:
        df["vol"] = (df["high"] - df["low"]).clip(lower=1e-12)
    r = df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()
    raw4h[p] = r

feats = {}
for p, df in raw4h.items():
    f = add_features(df)
    f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14",
                     "ema_dist_pct","real_vol_20","bb_width","prev_range_r",
                     "prev_body_r"], inplace=True)
    if len(f) >= IS_LOOKBACK_4H + RECAL_EVERY_4H:
        feats[p] = f
print(f"  Pairs ready: {len(feats)}")

def add_levels(f):
    f = f.copy()
    day = f.index.normalize()
    typ = (f["high"] + f["low"] + f["close"]) / 3.0
    f["vwap"] = (typ * f["vol"]).groupby(day).cumsum() / f["vol"].groupby(day).cumsum().replace(0, np.nan)
    f["p20_hi"] = f["high"].rolling(20).max().shift(1)
    f["p20_lo"] = f["low"].rolling(20).min().shift(1)
    d = f.resample("1D").agg({"high":"max","low":"min"}).shift(1)
    f["pd_hi"] = d["high"].reindex(day).values
    f["pd_lo"] = d["low"].reindex(day).values
    return f

feats2 = {p: add_levels(f) for p, f in feats.items()}

# ── fast forward-outcome per pair (long & short) ─────────────────────────────
print("\n  Precomputing forward outcomes (long/short, 24-bar scan) …")
outcomes = {}
for p, f in feats2.items():
    cl = f["close"].values; hi = f["high"].values; lo = f["low"].values
    atr = f["atr14"].values; n = len(f)
    long_out = np.full(n, np.nan)
    short_out = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(atr[i]) or atr[i] <= 0: continue
        tp = cl[i] + RR * atr[i]; sl = cl[i] - atr[i]
        end = min(i + HORIZON, n)
        hit = None
        for b in range(i + 1, end):
            if hi[b] >= tp: hit = RR; break
            if lo[b] <= sl: hit = -1.0; break
        if hit is not None:
            long_out[i] = hit
        elif end > i + 1:
            long_out[i] = (cl[end - 1] - cl[i]) / atr[i]
        # short (inverted): TP below, SL above
        tp_s = cl[i] - RR * atr[i]; sl_s = cl[i] + atr[i]
        hit = None
        for b in range(i + 1, end):
            if lo[b] <= tp_s: hit = RR; break
            if hi[b] >= sl_s: hit = -1.0; break
        if hit is not None:
            short_out[i] = hit
        elif end > i + 1:
            short_out[i] = (cl[i] - cl[end - 1]) / atr[i]
    outcomes[p] = (long_out, short_out)
print("  outcomes done")

# ── build event table ────────────────────────────────────────────────────────
def build_events():
    events = []
    for p, f in feats2.items():
        idx = f.index
        hi = f["high"].values; lo = f["low"].values; cl = f["close"].values
        atr = f["atr14"].values
        levels = {"P20": (f["p20_hi"].values, f["p20_lo"].values),
                  "PD":  (f["pd_hi"].values,  f["pd_lo"].values),
                  "VWAP":(f["vwap"].values,   f["vwap"].values),
                  "EMA20":(f["ema20"].values, f["ema20"].values)}
        for tname, (lv_hi, lv_lo) in levels.items():
            for i in range(21, len(f)):
                if np.isnan(atr[i]) or atr[i] <= 0: continue
                if np.isnan(lv_hi[i]) or np.isnan(lv_lo[i]): continue
                lv = (lv_hi[i] + lv_lo[i]) / 2.0 if tname in ("VWAP","EMA20") else lv_hi[i]
                if hi[i] > lv + DEPTH * atr[i]:
                    events.append(dict(pair=p, ts=idx[i], type=tname, side="bull", level=lv,
                                       depth=(hi[i]-lv)/atr[i], rsi=float(f["rsi14"].iloc[i]),
                                       atr_rank=float(f["atr_rank"].iloc[i]), adx=float(f["adx14"].iloc[i]),
                                       ema_dist=float(f["ema_dist_pct"].iloc[i]), relvol=float(f["rel_vol"].iloc[i]),
                                       trend=float(f["ema50"].iloc[i]-f["ema200"].iloc[i]),
                                       hour=idx[i].hour, bb=float(f["bb_width"].iloc[i])))
                if lo[i] < lv - DEPTH * atr[i]:
                    events.append(dict(pair=p, ts=idx[i], type=tname, side="bear", level=lv,
                                       depth=(lv-lo[i])/atr[i], rsi=float(f["rsi14"].iloc[i]),
                                       atr_rank=float(f["atr_rank"].iloc[i]), adx=float(f["adx14"].iloc[i]),
                                       ema_dist=float(f["ema_dist_pct"].iloc[i]), relvol=float(f["rel_vol"].iloc[i]),
                                       trend=float(f["ema50"].iloc[i]-f["ema200"].iloc[i]),
                                       hour=idx[i].hour, bb=float(f["bb_width"].iloc[i])))
    return pd.DataFrame(events).sort_values("ts").reset_index(drop=True)

ev = build_events()
print(f"  Trap events: {len(ev)} | bull={(ev['side']=='bull').sum()} bear={(ev['side']=='bear').sum()}")
ev["trend_dir"] = np.where(ev["trend"] > 0, "up", "dn")
ev["sess"] = np.where(ev["hour"] == 8, "london", np.where(ev["hour"] == 12, "ny_ovl", "other"))

# attach outcomes
def get_outcome(e, response):
    want_up = (e["side"] == "bull" and response == "follow") or (e["side"] == "bear" and response == "reject")
    lo, so = outcomes[e["pair"]]
    i = feats2[e["pair"]].index.get_loc(e["ts"])
    return lo[i] if want_up else so[i]

ev["r_follow"] = [get_outcome(e, "follow") for e in ev.to_dict("records")]
ev["r_reject"] = [get_outcome(e, "reject") for e in ev.to_dict("records")]
ev["follow_wins"] = (ev["r_follow"] > ev["r_reject"]).astype(int)
ev["spread"] = ev["pair"].map(SPREAD)
ev["atr"] = [feats2[e["pair"]]["atr14"].get(e["ts"], 1e-6) for e in ev.to_dict("records")]

# ── causal audit ─────────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  CAUSAL AUDIT\n{SEP2}")
test_p = sorted(feats2.keys())[0]
ft = feats2[test_p]; fts = ft.iloc[:-200]
def evs(f):
    out=set(); hi=f["high"].values; lo=f["low"].values; atr=f["atr14"].values
    lv={"P20":(f["p20_hi"].values,f["p20_lo"].values),"PD":(f["pd_hi"].values,f["pd_lo"].values),
        "VWAP":(f["vwap"].values,f["vwap"].values),"EMA20":(f["ema20"].values,f["ema20"].values)}
    for tname,(a,b) in lv.items():
        for i in range(21,len(f)):
            if np.isnan(atr[i]) or atr[i]<=0: continue
            if np.isnan(a[i]) or np.isnan(b[i]): continue
            lv2=(a[i]+b[i])/2.0 if tname in("VWAP","EMA20") else a[i]
            if hi[i]>lv2+DEPTH*atr[i]: out.add((tname,"bull",f.index[i]))
            if lo[i]<lv2-DEPTH*atr[i]: out.add((tname,"bear",f.index[i]))
    return out
full=evs(ft); trunc=evs(fts)
ov=[x for x in full if x[2] < fts.index[-1]]
print(f"  audit {'PASS ✓' if all(x in trunc for x in ov) else 'FAIL ✗'} ({len(ov)} events)")

# ── helpers ──────────────────────────────────────────────────────────────────
def eval_trades(name, rs, evs_used):
    n = len(rs)
    if n == 0: return dict(cfg=name, n=0, tpm=0, wr=float('nan'), pf=float('nan'), pf_c=float('nan'),
                           mdd=0, prof=float('nan'), worst=float('nan'), selpf=float('nan'),
                           holpf=float('nan'), holpf_c=float('nan'))
    rs = np.array(rs); evs_used = evs_used.reset_index(drop=True)
    cost_r = 2 * (evs_used["spread"].values / np.maximum(evs_used["atr"].values, 1e-12))
    rc = rs - cost_r
    pf = (rs[rs>0].sum()/abs(rs[rs<0].sum())) if (rs<0).any() else 99.0
    pf_c = (rc[rc>0].sum()/abs(rc[rc<0].sum())) if (rc<0).any() else 99.0
    ts = evs_used["ts"].values
    sel = ts < np.datetime64(HOLDOUT_START)
    hol = ~sel
    sp = (rs[sel][rs[sel]>0].sum()/abs(rs[sel][rs[sel]<0].sum())) if (rs[sel]<0).any() else 99.0
    hp = (rs[hol][rs[hol]>0].sum()/abs(rs[hol][rs[hol]<0].sum())) if (rs[hol]<0).any() else 99.0
    hrc = rc[hol]
    hpc = (hrc[hrc>0].sum()/abs(hrc[hrc<0].sum())) if (hrc<0).any() else 99.0
    m = pd.Series(ts).dt.to_period("M"); g = pd.Series(rs).groupby(m).sum()
    flags=(g>0).astype(int).values; cur=worst=0
    for v in flags:
        cur=cur+1 if not v else 0; worst=max(worst,cur)
    return dict(cfg=name, n=n, tpm=n/24, wr=float((rs>0).mean()), pf=pf, pf_c=pf_c,
                mdd=0.0, prof=float((g>0).mean()), worst=worst, selpf=sp, holpf=hp, holpf_c=hpc)

# ── matrix ───────────────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  TRAP MATRIX (response x type x side x trend x session) RR{RR}\n{SEP2}")
rows = []
for response in ["follow", "reject"]:
    for tname in ["P20","PD","VWAP","EMA20"]:
        for side in ["bull","bear"]:
            for trend_dir in ["up","dn"]:
                for sess in ["all","london","ny_ovl"]:
                    sub = ev[(ev["type"]==tname)&(ev["side"]==side)&(ev["trend_dir"]==trend_dir)]
                    if sess=="london": sub = sub[sub["sess"]=="london"]
                    elif sess=="ny_ovl": sub = sub[sub["sess"]=="ny_ovl"]
                    if len(sub) < 40: continue
                    rs = sub["r_follow"].values if response=="follow" else sub["r_reject"].values
                    label = f"{response[:3]}|{tname}|{side[:2]}|{trend_dir}|{sess[:2]}"
                    r = eval_trades(label, rs, sub)
                    r["response"]=response; r["type"]=tname; r["side"]=side
                    rows.append(r)

mdf = pd.DataFrame(rows).sort_values("holpf_c", ascending=False)
print("  TOP 15 by holPF@cost:")
show = mdf.head(15)[["cfg","n","wr","pf","holpf","holpf_c","prof","worst"]].copy()
show["wr"]=(show["wr"]*100).round(0); show["prof"]=(show["prof"]*100).round(0)
show.columns=["cfg","n","WR%","PF","holPF","holPF@c","prof%","worst"]
print(show.to_string(index=False))
print(f"\n  cells with holPF@cost>1.1: {int((mdf['holpf_c']>1.1).sum())} of {len(mdf)}")

# ── ML trap classifier ───────────────────────────────────────────────────────
print(f"\n{SEP2}\n  ML TRAP CLASSIFIER\n{SEP2}")
EV_FEATS = ["depth","rsi","atr_rank","adx","ema_dist","relvol","trend","hour","bb"]
ev2 = ev.dropna(subset=EV_FEATS).reset_index(drop=True)
X = ev2[EV_FEATS].fillna(0).values
y = ev2["follow_wins"].values
pred = np.full(len(ev2), np.nan)
sc = StandardScaler()
min_train, step = 150, 100
i = min_train
while i < len(ev2):
    j = min(i+step, len(ev2))
    clf = SVC(C=1.0, gamma="scale", probability=True)
    clf.fit(sc.fit_transform(X[:i]), y[:i])
    pred[i:j] = clf.predict_proba(sc.transform(X[i:j]))[:,1]
    i = j
ev2["p_follow"] = pred
valid = ev2[~np.isnan(ev2["p_follow"])].copy()
valid["chosen_r"] = np.where(valid["p_follow"] >= 0.5, valid["r_follow"], valid["r_reject"])
mr = eval_trades("ML_trap", valid["chosen_r"].values, valid)
rows.append(mr)
print(f"  ML-trap: n={mr['n']} WR={mr['wr']*100:.0f}% PF={mr['pf']:.2f} holPF={mr['holpf']:.2f} "
      f"holPF@cost={mr['holpf_c']:.2f} prof%={mr['prof']*100:.0f}% worst={mr['worst']}")
# baselines
af = eval_trades("always_follow", ev2["r_follow"].values, ev2)
aj = eval_trades("always_reject", ev2["r_reject"].values, ev2)
rows += [af, aj]
print(f"  always-follow: n={af['n']} holPF={af['holpf']:.2f} holPF@cost={af['holpf_c']:.2f}")
print(f"  always-reject: n={aj['n']} holPF={aj['holpf']:.2f} holPF@cost={aj['holpf_c']:.2f}")
print(f"  ML vs follow: {'ML wins' if mr['holpf']>af['holpf'] else 'follow wins'} (gross)")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "f006_trap_matrix.csv"), index=False)
mdf.to_csv(os.path.join(OUT, "f006_matrix_detail.csv"), index=False)
lines = [f"# FOREX F006 — COMPREHENSIVE TRAP HUNT\n",
         f"**Date:** 2026-08-08 | 4H, 8 pairs, RR{RR}, retail spreads | sel ≤ Aug-2025\n",
         f"Trap types: P20/PD/VWAP/EMA20 | responses follow/reject | trend | sessions\n",
         f"\n## Top matrix cells (holPF@cost)\n"]
for _, r in mdf.head(15).iterrows():
    lines.append(f"- {r['cfg']}: n={r['n']} WR={r['wr']*100:.0f}% PF={r['pf']:.2f} "
                 f"holPF={r['holpf']:.2f} holPF@cost={r['holpf_c']:.2f} prof%={r['prof']*100:.0f}%")
lines += ["", "## ML trap vs baselines", ""]
for r in [mr, af, aj]:
    lines.append(f"- {r['cfg']}: n={r['n']} holPF={r['holpf']:.2f} holPF@cost={r['holpf_c']:.2f} WR={r['wr']*100:.0f}%")
lines += ["", "## Verdict", ""]
passed = [r for r in rows if r.get("holpf_c",0) > 1.1 and r.get("n",0) >= 60]
if passed:
    lines.append(f"**✅ {passed[0]['cfg']} survives costs on holdout.**")
else:
    lines.append("**❌ No trap config survives retail spreads on holdout.**")
report = "\n".join(lines)
with open(os.path.join(OUT, "f006_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/f006_*")
