"""
TEST 29 — 15m on the FULL universe, last 7 months, fetched FAST via parallel workers.
User: "use async to collect and test faster; test the whole 70 on the last 7 months
just to know if it's worth the time." Fetches 15m (~20k bars = 2026-02..09) for all
cached symbols concurrently, then runs the condition-aware TREND (T25 logic) on all of
them. Reports aggregate PF, # symbols profitable (breadth), and $100/$2 sim. The point
is BREADTH: does the 15m edge hold across the whole market, not just 12 majors?
"""
import os, sys, time, warnings, concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from ql_engine import add_features, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import build_mldf, FEATS
import demo_bot as bot
from sklearn.ensemble import RandomForestClassifier

CACHE = "quantlab_cache"; SAVE = "quantlab_cache_15m"; FEE = 0.0005
os.makedirs(SAVE, exist_ok=True)
SPLIT = pd.Timestamp("2026-05-15", tz="UTC")
N_BARS = 20000  # ~7 months of 15m


def fetch_before(end_ms, n_bars, inst, bar="15m", page_limit=200):
    all_rows, after, pages = [], end_ms, 0
    while len(all_rows) < n_bars and pages < page_limit:
        params = {"instId": inst, "bar": bar, "limit": bot.PAGE_LIMIT, "after": str(after)}
        raw = bot._get(bot.OKX_CANDLES, params) or bot._get(bot.OKX_CANDLES_CUR, params)
        if not raw: break
        all_rows.extend(raw); pages += 1; after = int(raw[-1][0])
        if len(all_rows) >= n_bars: break
        time.sleep(bot.PAGE_DELAY)
    if not all_rows: return None
    df = pd.DataFrame(all_rows, columns=bot.CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open","high","low","close","vol"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["datetime","open","high","low","close","vol"]].sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True).set_index("datetime")


def fetch_one(sym):
    out = os.path.join(SAVE, f"{sym}_15m.parquet")
    if os.path.exists(out):
        try:
            if len(pd.read_parquet(out)) >= 15000: return None  # already have 7mo
        except Exception: pass
    try:
        inst = sym.replace("_","-")
        end_ms = int(pd.Timestamp.now(tz="UTC").timestamp()*1000) - 1
        df = fetch_before(end_ms, N_BARS, inst, "15m")
        if df is not None and len(df) > 1000:
            df.to_parquet(out); os.sync(); return sym
    except Exception: pass
    return None


syms = sorted(f[:-len("_1H.parquet")] for f in os.listdir(CACHE) if f.endswith("_1H.parquet"))
print(f"[fetch] {len(syms)} symbols; fetching 15m (7mo) in parallel …", flush=True)
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
    done = list(ex.map(fetch_one, syms))
done = [s for s in done if s]
print(f"[fetch] done {len(done)} symbols in {time.time()-t0:.0f}s", flush=True)

# ── load all 15m and run condition-aware trend ──
def backtest_donchian(df, N=20, Nx=20, atr_mult=2.0, adx_min=20.0):
    df = df.copy()
    hh = df["high"].rolling(N).max().shift(1); ll = df["low"].rolling(Nx).min().shift(1)
    trades = []; in_pos=False; ep=None; stop=None
    for i in range(N, len(df)):
        bar = df.iloc[i]
        if not in_pos:
            if bar["close"]>hh.iloc[i] and bar["adx14"]>adx_min and bar["close"]>bar["ema200"]:
                ep=bar["close"]; stop=ep-atr_mult*bar["atr14"]; in_pos=True
        else:
            ex=None; et=None
            if bar["low"]<=stop: ex=stop; et="SL"
            elif bar["close"]<ll.iloc[i]: ex=bar["close"]; et="BRK"
            if ex is not None:
                trades.append(dict(entry_time=df.index[i], r=(ex/ep-1.0))); in_pos=False
    return trades


print("[load] 15m parquets …", flush=True)
feats, above20 = {}, {}
for f in sorted(os.listdir(SAVE)):
    if not f.endswith("_15m.parquet"): continue
    sym = f[:-len("_15m.parquet")]
    try:
        df = pd.read_parquet(os.path.join(SAVE, f)); df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for c in ["open","high","low","close","vol"]:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        ftr=add_features(df); ftr.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(ftr)>=IS_LOOKBACK+RECAL_EVERY: feats[sym]=ftr; above20[sym]=(ftr["close"]>ftr["ema20"]).astype(float)
    except Exception as e: print(f"  err {sym}: {e}", flush=True)
print(f"[load] usable 15m: {len(feats)}", flush=True)
breadth=pd.DataFrame(above20).sort_index().mean(axis=1,skipna=True)
breadth_pct=breadth.rolling(100,min_periods=50).rank(pct=True)*100

raw_trend=[]
for s in feats:
    for t in backtest_donchian(feats[s]):
        raw_trend.append(dict(sym=s, entry_time=t["entry_time"], r=t["r"]))
raw_trend.sort(key=lambda t:t["entry_time"])
mldf=build_mldf(raw_trend,feats,breadth,breadth_pct)
print(f"[signals] 15m Donchian trades (7mo): {len(raw_trend)} across {len(feats)} syms", flush=True)

tr=mldf[mldf.ts<SPLIT]; te=mldf[mldf.ts>=SPLIT]
print(f"[split] train={len(tr)} test={len(te)}", flush=True)
m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr[FEATS],tr["win"])
P=m.predict_proba(te[FEATS])[:,1]; q=0.65; thr=np.quantile(P,1-q)
kept=set(te[P>=thr]["ts"])
champ=[t for t in raw_trend if t["entry_time"]>=SPLIT and t["entry_time"] in kept]
for t in champ: t["adj_r"]=t["r"]-2*FEE
print(f"[bt] 15m champion (test half, all syms): {len(champ)}", flush=True)

print("\n"+"="*80)
print("TEST 29 — 15m condition-aware TREND on FULL universe, last 7 months (test half)")
print("="*80)
eq=1.0;peak=1.0;mdd=0.0
for t in sorted(champ,key=lambda x:x["entry_time"]):
    eq*=(1+0.01*t["adj_r"]); peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
rs=[t["adj_r"] for t in champ]
wins=sum(1 for r in rs if r>0)/len(rs)
pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
msum=defaultdict(float); sym_net=defaultdict(float)
for t in champ:
    msum[(t["entry_time"].year,t["entry_time"].month)]+=t["adj_r"]; sym_net[t["sym"]]+=t["adj_r"]
pos=sum(1 for v in msum.values() if v>0); nprof=sum(1 for v in sym_net.values() if v>0)
print(f"n={len(champ)} win={wins:.0%} PF@c={pf:.3f} MAX DD={mdd:.1%} prof-months={pos}/{len(msum)} symbols-profitable={nprof}/{len(sym_net)}")
for (yy,mm) in sorted(msum):
    nr=msum[(yy,mm)]; print(f"   {yy}-{mm:02d}  netR={nr:>+7.2f}  {'OK' if nr>0 else 'LOSS'}")
eq=100.0
for t in sorted(champ,key=lambda x:x["entry_time"]): eq+=2.0*t["adj_r"]
print(f"$100 @ $2/trade -> ${eq:,.2f} ({(eq/100-1):+.1%}) over ~3.5mo (test half)")
print("\n[done]")
