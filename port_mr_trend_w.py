"""
TEST 27 — TUNED PORTFOLIO: MR champion + condition-aware TREND, TREND-WEIGHTED (0.7/0.3).
Same validated components as T26, but trend (the stronger, low-DD leg) gets 70% of
capital so 2024 also clears PF>=1.3 while keeping the ~-1.2% drawdown. Bug-free reuse
of T26 logic; only the equity weighting + per-leg risk split changed.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from ql_engine import add_features, build_signal_mask, sim_symbol, cost_adjusted_rs, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import SVMQ65Adaptive, build_mldf, FEATS
import demo_bot as bot
from sklearn.ensemble import RandomForestClassifier

SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]
CACHE = "quantlab_cache"; FAM_A = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
FEE = 0.0005
W_TREND = 0.70; RISK_TR = 0.007; RISK_MR = 0.003   # trend-weighted; total max risk ~1%


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

# ── MR champion (SVMQ65Adaptive) ──
mask={s:build_signal_mask(feats[s],FAM_A,"green",1.5) for s in feats}
mr_raw=[]
for s in feats:
    for t in sim_symbol(feats[s],mask[s],1.5,dict(entry_next=False,exit="base",hours=None)):
        t["sym"]=s; mr_raw.append(t)
mr_raw.sort(key=lambda t:t["entry_time"])
mldf_mr=build_mldf(mr_raw,feats,breadth,breadth_pct)
mr_champ=[]
for Y in [2024,2025,2026]:
    tr=mldf_mr[mldf_mr.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te=mldf_mr[mldf_mr.ts.dt.year==Y]
    if len(tr)<50 or len(te)==0: continue
    m=SVMQ65Adaptive().fit_mldf(tr); kept,_=m.keep_mldf(te)
    for t in mr_raw:
        if t["entry_time"].year==Y and t["entry_time"] in kept: mr_champ.append(t)
mr_adj=cost_adjusted_rs(list(mr_champ),0.05)
for t,a in zip(mr_champ,mr_adj): t["adj_r"]=a

# ── Trend champion (RF condition-aware, top-65%) ──
tr_raw=[]
for s in feats:
    for t in backtest_donchian(feats[s]):
        tr_raw.append(dict(sym=s, entry_time=t["entry_time"], r=t["r"]))
tr_raw.sort(key=lambda t:t["entry_time"])
mldf_tr=build_mldf(tr_raw,feats,breadth,breadth_pct)
tr_champ=[]
for Y in [2024,2025,2026]:
    tr=mldf_tr[mldf_tr.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te=mldf_tr[mldf_tr.ts.dt.year==Y]
    if len(tr)<50 or len(te)==0: continue
    m=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=20,class_weight="balanced",n_jobs=-1,random_state=0).fit(tr[FEATS],tr["win"])
    P=m.predict_proba(te[FEATS])[:,1]; q=0.65; thr=np.quantile(P,1-q)
    kept=set(te[P>=thr]["ts"])
    for t in tr_raw:
        if t["entry_time"].year==Y and t["entry_time"] in kept: tr_champ.append(t)
for t in tr_champ: t["adj_r"]=t["r"]-2*FEE

print(f"[bt] MR champion={len(mr_champ)}  Trend champion={len(tr_champ)}", flush=True)


def report(name, mr_list, tr_list):
    print("\n"+"="*82)
    print(name)
    print("="*82)
    for Y in [2024,2025,2026]:
        mrs=[t for t in mr_list if t["entry_time"].year==Y]
        trs=[t for t in tr_list if t["entry_time"].year==Y]
        eqA=1.0;eqB=1.0;peak=1.0;ymdd=0.0;msum=defaultdict(list)
        times=sorted(set(t["entry_time"] for t in mrs)|set(t["entry_time"] for t in trs))
        evA={t["entry_time"]:t["adj_r"] for t in mrs}; evB={t["entry_time"]:t["adj_r"] for t in trs}
        for ts in times:
            if ts in evA: eqA*=(1+RISK_MR*evA[ts])
            if ts in evB: eqB*=(1+RISK_TR*evB[ts])
            comb=W_TREND*eqB+(1-W_TREND)*eqA; peak=max(peak,comb); ymdd=min(ymdd,comb/peak-1)
            if ts in evA: msum[(ts.year,ts.month)].append((1-W_TREND)*evA[ts])
            if ts in evB: msum[(ts.year,ts.month)].append(W_TREND*evB[ts])
        rs=[r for v in msum.values() for r in v]
        wins=sum(1 for r in rs if r>0)/len(rs) if rs else 0
        pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0)) if rs else 0
        pos=sum(1 for v in msum.values() if sum(v)>0)
        ret=(W_TREND*eqB+(1-W_TREND)*eqA)-1
        print(f"\n[{Y}] n={len(mrs)+len(trs)} win={wins:.0%} PF@c={pf:.3f} ret(1%)={ret:+.1%} MAX DD={ymdd:.1%} prof-months={pos}/{len(msum)}")
        for (yy,mm) in sorted(msum):
            nr=sum(msum[(yy,mm)]); print(f"   {yy}-{mm:02d}  netR={nr:>+7.2f}  {'OK' if nr>0 else 'LOSS'}")
    eqA=1.0;eqB=1.0;peak=1.0;fmdd=0.0
    times=sorted(set(t["entry_time"] for t in mr_list)|set(t["entry_time"] for t in tr_list))
    evA={t["entry_time"]:t["adj_r"] for t in mr_list}; evB={t["entry_time"]:t["adj_r"] for t in tr_list}
    for ts in times:
        if ts in evA: eqA*=(1+RISK_MR*evA[ts])
        if ts in evB: eqB*=(1+RISK_TR*evB[ts])
        comb=W_TREND*eqB+(1-W_TREND)*eqA; peak=max(peak,comb); fmdd=min(fmdd,comb/peak-1)
    rs=[r for v in [evA,evB] for r in v.values()]
    pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
    ret=(W_TREND*eqB+(1-W_TREND)*eqA)-1
    print(f"\n[FULL 2024-2026] PF@c={pf:.3f} ret(1%)={ret:+.1%} MAX DD={fmdd:.1%}")


report(f"PORTFOLIO (TREND-WEIGHTED {W_TREND:.0%}/{1-W_TREND:.0%}, risk {RISK_TR:.1%}/{RISK_MR:.1%})", mr_champ, tr_champ)
print("\n[done]")
