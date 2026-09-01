"""
TEST 25b — T25 in DOLLARS. Reuse the EXACT validated T25 pipeline (1H Donchian +
RF condition-aware filter, top-65%) to get the trend champion trades, then simulate
a real account: start $100, FIXED $2 risk per trade (and also 2%-of-equity variant).
Report equity per year + max DD. No new strategy logic — only a dollar simulator.
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
    except Exception as e: print(f"  err {sym}: {e}", flush=True)
print(f"[load] subset: {len(feats)}", flush=True)
breadth=pd.DataFrame(above20).sort_index().mean(axis=1,skipna=True)
breadth_pct=breadth.rolling(100,min_periods=50).rank(pct=True)*100

raw_trend=[]
for s in feats:
    for t in backtest_donchian(feats[s]):
        raw_trend.append(dict(sym=s, entry_time=t["entry_time"], r=t["r"]))
raw_trend.sort(key=lambda t:t["entry_time"])
mldf=build_mldf(raw_trend,feats,breadth,breadth_pct)
print(f"[signals] 1H Donchian trades 2023-2026: {len(raw_trend)}", flush=True)

tr_champ=[]
for Y in [2024,2025,2026]:
    tr=mldf[mldf.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te=mldf[mldf.ts.dt.year==Y]
    if len(tr)<50 or len(te)==0: continue
    m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr[FEATS],tr["win"])
    P=m.predict_proba(te[FEATS])[:,1]; q=0.65; thr=np.quantile(P,1-q)
    kept=set(te[P>=thr]["ts"])
    for t in raw_trend:
        if t["entry_time"].year==Y and t["entry_time"] in kept: tr_champ.append(t)
for t in tr_champ: t["adj_r"]=t["r"]-2*FEE
print(f"[bt] T25 condition-aware trend champion trades: {len(tr_champ)}", flush=True)


def dollar_sim(trades, start_eq, fixed_risk=2.0, fractional=False):
    eq=start_eq; peak=start_eq; mdd=0.0; n=0; wins=0
    for t in sorted(trades, key=lambda x:x["entry_time"]):
        if fractional:
            pnl = 0.02*eq*t["adj_r"]
        else:
            pnl = fixed_risk*t["adj_r"]
        eq += pnl
        if pnl>0: wins+=1
        n+=1
        peak=max(peak,eq); mdd=min(mdd, eq/peak-1)
    return eq, mdd, n, (wins/n if n else 0)


print("\n"+"="*78)
print("T25 in DOLLARS — start $100, $2 risk per trade (fixed)")
print("="*78)
eq=100.0
for Y in [2024,2025,2026]:
    yt=[t for t in tr_champ if t["entry_time"].year==Y]
    start=eq
    end,mdd,n,wr=dollar_sim(yt,start,fixed_risk=2.0,fractional=False)
    print(f"[{Y}] start ${start:,.2f} -> end ${end:,.2f}  return {(end/start-1):+.1%}  MAX DD {mdd:.1%}  trades={n} win={wr:.0%}")
    eq=end
print(f"[FULL] start $100 -> end ${eq:,.2f}  total return {(eq/100-1):+.1%}")

print("\n"+"="*78)
print("T25 in DOLLARS — start $100, 2% of EQUITY per trade (fractional, compounding)")
print("="*78)
eq=100.0
for Y in [2024,2025,2026]:
    yt=[t for t in tr_champ if t["entry_time"].year==Y]
    start=eq
    end,mdd,n,wr=dollar_sim(yt,start,fractional=True)
    print(f"[{Y}] start ${start:,.2f} -> end ${end:,.2f}  return {(end/start-1):+.1%}  MAX DD {mdd:.1%}  trades={n} win={wr:.0%}")
    eq=end
print(f"[FULL] start $100 -> end ${eq:,.2f}  total return {(eq/100-1):+.1%}")
print("\n[done]")
