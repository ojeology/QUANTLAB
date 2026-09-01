"""
NEW HYPOTHESIS (multi-strategy) — combine mean-reversion (SVM champion) + trend
(Donchian) into ONE portfolio. They win in different regimes, so together they
should survive all 3 years with smoother equity. 1H, 20-sym, 2024-2026 (2023
fetched in-run). 50/50 capital split, each strategy risks 0.5% of total/trade.
Reports PF@cost, MAX DD, profitable-months per year + full.
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

CACHE = "quantlab_cache"
FAM_A = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]
FEE = 0.0005


def fetch_before(end_ts_ms, n_bars, inst, bar="1H", page_limit=200):
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


def backtest_donchian(df, N=20, Nx=10, atr_mult=2.0, adx_min=25.0):
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

# --- Mean-reversion (SVM champion) ---
mask={s:build_signal_mask(feats[s],FAM_A,"green",1.5) for s in feats}
mr_raw=[]
for s in feats:
    for t in sim_symbol(feats[s],mask[s],1.5,dict(entry_next=False,exit="base",hours=None)):
        t["sym"]=s; mr_raw.append(t)
mr_raw.sort(key=lambda t:t["entry_time"])
mr_mldf=build_mldf(mr_raw,feats,breadth,breadth_pct)
mr_trades=[]
for Y in [2024,2025,2026]:
    tr=mr_mldf[mr_mldf.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te=mr_mldf[mr_mldf.ts.dt.year==Y]
    if len(tr)<50 or len(te)==0: continue
    m=SVMQ65Adaptive().fit_mldf(tr); kept,_=m.keep_mldf(te)
    for t in mr_raw:
        if t["entry_time"].year==Y and t["entry_time"] in kept: mr_trades.append(t)
mr_adj=cost_adjusted_rs(list(mr_trades),0.05)
for t,a in zip(mr_trades,mr_adj): t["adj_r"]=a

# --- Trend (Donchian) ---
tr_trades=[]
for s in feats:
    for t in backtest_donchian(feats[s]):
        t["sym"]=s; tr_trades.append(t)
for t in tr_trades: t["adj_r"]=t["r"]-2*FEE

print(f"[bt] MR trades={len(mr_trades)}  Trend trades={len(tr_trades)}", flush=True)


def portfolio_metrics(mr_list, tr_list, risk_each=0.005):
    """50/50 capital; each strategy risks `risk_each` of TOTAL per trade."""
    eqA=1.0; eqB=1.0; peak=1.0; mdd=0.0; all_ev=[]
    evA={t["entry_time"]:t["adj_r"] for t in mr_list}
    evB={t["entry_time"]:t["adj_r"] for t in tr_list}
    times=sorted(set(evA)|set(evB))
    for ts in times:
        if ts in evA: eqA*=(1+risk_each*evA[ts])
        if ts in evB: eqB*=(1+risk_each*evB[ts])
        comb=0.5*eqA+0.5*eqB; peak=max(peak,comb); mdd=min(mdd,comb/peak-1)
        all_ev.append((ts,comb))
    return all_ev, mdd


def pf_of(seq):
    wins=sum(r for _,r in seq if r>0); loss=-sum(r for _,r in seq if r<0)
    return wins/max(1e-9,loss) if loss>0 else 999.0

def month_summary(trades, risk):
    eq=1.0; peak=1.0; mdd=0.0
    msum=defaultdict(list)
    for t in sorted(trades,key=lambda x:x["entry_time"]):
        eq*=(1+risk*t["adj_r"]); peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
        msum[(t["entry_time"].year,t["entry_time"].month)].append(t["adj_r"])
    return eq-1,mdd,msum

print("\n"+"="*84)
print("MULTI-STRATEGY PORTFOLIO: MR(SVM) + Trend(Donchian), 50/50, 0.5% risk each")
print("="*84)
mr_seq=[(t["entry_time"],t["adj_r"]) for t in mr_trades]
tr_seq=[(t["entry_time"],t["adj_r"]) for t in tr_trades]
all_ev,mdd=portfolio_metrics(mr_trades,tr_trades)
# combined per-year
for Y in [2024,2025,2026]:
    mrs=[t for t in mr_trades if t["entry_time"].year==Y]
    trs=[t for t in tr_trades if t["entry_time"].year==Y]
    # combined equity for the year
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
    pf=pf_of([(0,r) for r in rs]); pos=sum(1 for v in msum.values() if sum(v)>0)
    print(f"\n[{Y}] PF@cost={pf:.3f} ret(1% equiv)={(eqA/2+eqB/2-1):+.1%} MAX DD={ymdd:.1%} prof-months={pos}/{len(msum)}")
    for (yy,mm) in sorted(msum):
        nr=sum(msum[(yy,mm)]); print(f"   {yy}-{mm:02d}  netR={nr:>+7.2f}  {'OK' if nr>0 else 'LOSS'}")
# full
eqA=1.0;eqB=1.0;peak=1.0;fmdd=0.0
times=sorted(set(t["entry_time"] for t in mr_trades)|set(t["entry_time"] for t in tr_trades))
evA={t["entry_time"]:t["adj_r"] for t in mr_trades}; evB={t["entry_time"]:t["adj_r"] for t in tr_trades}
for ts in times:
    if ts in evA: eqA*=(1+0.005*evA[ts])
    if ts in evB: eqB*=(1+0.005*evB[ts])
    comb=0.5*eqA+0.5*eqB; peak=max(peak,comb); fmdd=min(fmdd,comb/peak-1)
rsAll=[r for v in [evA,evB] for r in v.values()]
print(f"\n[FULL 2024-2026] PF@cost={pf_of([(0,r) for r in rsAll]):.3f} ret(1% equiv)={(eqA/2+eqB/2-1):+.1%} MAX DD={fmdd:.1%}")
print("\n[done]")
