"""
MULTI-TIMEFRAME PORTFOLIO: 1H mean-reversion (SVM champion) + 1D trend (Donchian).
Different timeframes, complementary regimes: 1D trend wins trending years (2024/25),
1H MR wins choppy (2026). 20-sym, 2024-2026 (2023 fetched in-run for 1H). 50/50
capital, 0.5% risk each. Reports PF@cost, MAX DD, profitable-months per year.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from ql_engine import add_features, build_signal_mask, sim_symbol, cost_adjusted_rs, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import SVMQ65Adaptive, build_mldf
import demo_bot as bot

SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]
CACHE = "quantlab_cache"
FAM_A = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
FEE = 0.0005


def fetch_before(end_ts_ms, n_bars, inst, bar, page_limit=200):
    all_rows, after, pages = [], end_ts_ms, 0
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


def backtest_donchian(df, N=20, Nx=20, atr_mult=3.0, adx_min=25.0):
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


print("[load] 1H (fetch 2023 + cache) + 1D for 20-sym …", flush=True)
end_ms = int(pd.Timestamp.now(tz="UTC").timestamp()*1000) - 1
feats_1h, above20_1h, feats_1d = {}, {}, {}
for sym in SUBSET:
    try:
        # 1H
        p = os.path.join(CACHE, f"{sym}_1H.parquet")
        if os.path.exists(p):
            df = pd.read_parquet(p); df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
            for c in ["open","high","low","close","vol"]:
                if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
            df.dropna(subset=["open","high","low","close","vol"], inplace=True)
            inst=sym.replace("_","-"); em=int(df.index[0].timestamp()*1000)-1
            f1h=fetch_before(em,9200,inst,"1H")
            if f1h is not None and len(f1h): df=pd.concat([f1h,df]); df=df[~df.index.duplicated(keep="last")].sort_index()
            if len(df)>=IS_LOOKBACK+RECAL_EVERY+100:
                f=add_features(df); f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"],inplace=True)
                if len(f)>=IS_LOOKBACK+RECAL_EVERY: feats_1h[sym]=f; above20_1h[sym]=(f["close"]>f["ema20"]).astype(float)
        # 1D
        d1 = fetch_before(end_ms, 1400, sym.replace("_","-"), "1D")
        if d1 is not None and len(d1)>=IS_LOOKBACK+RECAL_EVERY+100:
            f=add_features(d1); f.dropna(subset=["ema200","atr14","adx14"],inplace=True)
            if len(f)>=IS_LOOKBACK+RECAL_EVERY: feats_1d[sym]=f
    except Exception as e:
        print(f"  err {sym}: {e}", flush=True)
print(f"[load] 1H={len(feats_1h)}  1D={len(feats_1d)}", flush=True)

# --- 1H MR SVM ---
breadth=pd.DataFrame(above20_1h).sort_index().mean(axis=1,skipna=True)
breadth_pct=breadth.rolling(100,min_periods=50).rank(pct=True)*100
mask={s:build_signal_mask(feats_1h[s],FAM_A,"green",1.5) for s in feats_1h}
mr_raw=[]
for s in feats_1h:
    for t in sim_symbol(feats_1h[s],mask[s],1.5,dict(entry_next=False,exit="base",hours=None)):
        t["sym"]=s; mr_raw.append(t)
mr_raw.sort(key=lambda t:t["entry_time"])
mr_mldf=build_mldf(mr_raw,feats_1h,breadth,breadth_pct)
mr_trades=[]
for Y in [2024,2025,2026]:
    tr=mr_mldf[mr_mldf.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te=mr_mldf[mr_mldf.ts.dt.year==Y]
    if len(tr)<50 or len(te)==0: continue
    m=SVMQ65Adaptive().fit_mldf(tr); kept,_=m.keep_mldf(te)
    for t in mr_raw:
        if t["entry_time"].year==Y and t["entry_time"] in kept: mr_trades.append(t)
mr_adj=cost_adjusted_rs(list(mr_trades),0.05)
for t,a in zip(mr_trades,mr_adj): t["adj_r"]=a

# --- 1D Trend ---
tr_trades=[]
for s in feats_1d:
    for t in backtest_donchian(feats_1d[s]):
        t["sym"]=s; tr_trades.append(t)
for t in tr_trades: t["adj_r"]=t["r"]-2*FEE
print(f"[bt] 1H-MR trades={len(mr_trades)}  1D-Trend trades={len(tr_trades)}", flush=True)

# --- Portfolio (50/50, 0.5% risk each) ---
print("\n"+"="*84)
print("MULTI-TIMEFRAME PORTFOLIO: 1H mean-reversion + 1D trend (50/50, 0.5% risk)")
print("="*84)
for Y in [2024,2025,2026]:
    mrs=[t for t in mr_trades if t["entry_time"].year==Y]
    trs=[t for t in tr_trades if t["entry_time"].year==Y]
    eqA=1.0;eqB=1.0;peak=1.0;ymdd=0.0;msum=defaultdict(list)
    times=sorted(set(t["entry_time"] for t in mrs)|set(t["entry_time"] for t in trs))
    evA={t["entry_time"]:t["adj_r"] for t in mrs}; evB={t["entry_time"]:t["adj_r"] for t in trs}
    for ts in times:
        if ts in evA: eqA*=(1+0.005*evA[ts])
        if ts in evB: eqB*=(1+0.005*evB[ts])
        comb=0.5*eqA+0.5*eqB; peak=max(peak,comb); ymdd=min(ymdd,comb/peak-1)
        if ts in evA: msum[(ts.year,ts.month)].append(evA[ts])
        if ts in evB: msum[(ts.year,ts.month)].append(evB[ts])
    rs=[r for v in msum.values() for r in v]
    wins=sum(1 for r in rs if r>0)/len(rs) if rs else 0
    pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0)) if rs else 0
    pos=sum(1 for v in msum.values() if sum(v)>0)
    print(f"\n[{Y}] n(MR+Tr)={len(mrs)+len(trs)} win={wins:.0%} PF@c={pf:.3f} ret(1%)={(eqA/2+eqB/2-1):+.1%} MAX DD={ymdd:.1%} prof-months={pos}/{len(msum)}")
    for (yy,mm) in sorted(msum):
        nr=sum(msum[(yy,mm)]); print(f"   {yy}-{mm:02d}  netR={nr:>+7.2f}  {'OK' if nr>0 else 'LOSS'}")
eqA=1.0;eqB=1.0;peak=1.0;fmdd=0.0
times=sorted(set(t["entry_time"] for t in mr_trades)|set(t["entry_time"] for t in tr_trades))
evA={t["entry_time"]:t["adj_r"] for t in mr_trades}; evB={t["entry_time"]:t["adj_r"] for t in tr_trades}
for ts in times:
    if ts in evA: eqA*=(1+0.005*evA[ts])
    if ts in evB: eqB*=(1+0.005*evB[ts])
    comb=0.5*eqA+0.5*eqB; peak=max(peak,comb); fmdd=min(fmdd,comb/peak-1)
rsAll=[r for v in [evA,evB] for r in v.values()]
pf=(sum(r for r in rsAll if r>0))/max(1e-9,-sum(r for r in rsAll if r<0))
print(f"\n[FULL 2024-2026] PF@c={pf:.3f} ret(1%)={(eqA/2+eqB/2-1):+.1%} MAX DD={fmdd:.1%}")
print("\n[done]")
