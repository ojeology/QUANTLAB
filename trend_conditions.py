"""
TEST 25 — CONDITION-AWARE TREND. The user's directive: ML learns the EXACT CONDITIONS
for strong trends, and we trade trend-following only when those hold. 1H Donchian
breakout signal; RandomForest predicts trend-trade success; keep top-65% by P(win);
report feature importances = the learned conditions for strong trends. Walk-forward
2024/2025/2026. Target: PF >= 1.3 every year.
"""
import os, sys, time, warnings
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

SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]
CACHE = "quantlab_cache"
FEE = 0.0005


def fetch_before(end_ms, n_bars, inst, bar="1H", page_limit=200):
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


print("[load] fetch 2023 + cache 2024-2026 (20-sym) …", flush=True)
feats, above20 = {}, {}
for sym in SUBSET:
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p); df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for c in ["open","high","low","close","vol"]:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        inst=sym.replace("_","-"); end_ms=int(df.index[0].timestamp()*1000)-1
        f2023=fetch_before(end_ms,9200,inst)
        if f2023 is not None and len(f2023): df=pd.concat([f2023,df]); df=df[~df.index.duplicated(keep="last")].sort_index()
        if len(df)<IS_LOOKBACK+RECAL_EVERY+100: continue
        f=add_features(df); f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f)>=IS_LOOKBACK+RECAL_EVERY: feats[sym]=f; above20[sym]=(f["close"]>f["ema20"]).astype(float)
        print(f"  loaded {sym}: {len(f)} bars", flush=True)
    except Exception as e: print(f"  err {sym}: {e}", flush=True)
print(f"[load] subset: {len(feats)}", flush=True)
breadth=pd.DataFrame(above20).sort_index().mean(axis=1,skipna=True)
breadth_pct=breadth.rolling(100,min_periods=50).rank(pct=True)*100

# trend raw trades -> mldf (reuse feature builder)
raw_trend=[]
for s in feats:
    for t in backtest_donchian(feats[s]):
        raw_trend.append(dict(sym=s, ts=t["entry_time"], entry_time=t["entry_time"], r=t["r"]))
raw_trend.sort(key=lambda t:t["ts"])
mldf=build_mldf(raw_trend,feats,breadth,breadth_pct)
print(f"[signals] 1H Donchian trend trades 2023-2026: {len(raw_trend)}  mldf={len(mldf)}", flush=True)

# ── Learn the conditions for strong trends ──
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                            class_weight="balanced", n_jobs=-1, random_state=0)
full_tr = mldf[mldf.ts < pd.Timestamp("2025-01-01", tz="UTC")]
rf.fit(full_tr[FEATS], full_tr["win"])
imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\n"+"="*70)
print("CONDITIONS THE ML LEARNED for STRONG TRENDS (top drivers of trend-trade wins):")
print("="*70)
for f_,v in imp.head(12).items(): print(f"   {f_:<16} {v:.3f}")

# ── Walk-forward: keep top-65% by predicted P(win) ──
champion=[]
for Y in [2024,2025,2026]:
    tr=mldf[mldf.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te=mldf[mldf.ts.dt.year==Y]
    if len(tr)<50 or len(te)==0: continue
    m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr[FEATS],tr["win"])
    P=m.predict_proba(te[FEATS])[:,1]; q=0.65; thr=np.quantile(P,1-q)
    kept=set(te[P>=thr]["ts"])
    for t in raw_trend:
        if t["ts"].year==Y and t["ts"] in kept: champion.append(t)
for t in champion: t["adj_r"]=t["r"]-2*FEE
print(f"\n[bt] condition-aware trend trades: {len(champion)}", flush=True)


def report():
    print("\n"+"="*80)
    print("TEST 25 — CONDITION-AWARE TREND (1H Donchian + RF filter, top-65%)")
    print("="*80)
    for Y in [2024,2025,2026]:
        yt=[t for t in champion if t["ts"].year==Y]
        if not yt: print(f"[{Y}] no trades"); continue
        eq=1.0;peak=1.0;mdd=0.0
        for t in sorted(yt,key=lambda x:x["ts"]):
            eq*=(1+0.01*t["adj_r"]); peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
        rs=[t["adj_r"] for t in yt]
        wins=sum(1 for r in rs if r>0)/len(rs)
        pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
        msum=defaultdict(float)
        for t in yt: msum[(t["ts"].year,t["ts"].month)]+=t["adj_r"]
        pos=sum(1 for v in msum.values() if v>0)
        print(f"\n[{Y}] n={len(yt)} win={wins:.0%} PF@c={pf:.3f} MAX DD={mdd:.1%} prof-months={pos}/{len(msum)}")
        for (yy,mm) in sorted(msum):
            print(f"   {yy}-{mm:02d}  netR={msum[(yy,mm)]:>+7.2f}  {'OK' if msum[(yy,mm)]>0 else 'LOSS'}")
    rs=[t["adj_r"] for t in champion]
    pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
    print(f"\n[FULL 2024-2026] n={len(champion)} PF@c={pf:.3f}")
report()
print("\n[done]")
