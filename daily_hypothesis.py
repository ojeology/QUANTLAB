"""
NEW ENVIRONMENT: DAILY (1D) crypto. Mean-reversion uses data-driven quantile
thresholds (so it fires on 1D, unlike the 1H-calibrated Family A), plus trend
Donchian. Fetch 1D 2023-2026, walk-forward 2024/2025/2026. PF@cost, MAX DD,
profitable-months per year.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from ql_engine import add_features, sim_symbol, cost_adjusted_rs, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import SVMQ65Adaptive, build_mldf
import demo_bot as bot

SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]
FEE = 0.0005
CID = {"BBW_STRICT":("bb_width","lt",0.25),"RV_LO":("real_vol_20","lt",0.33),
       "DST_NR":("ema_dist_pct","lt",0.33),"PRG_VH":("prev_range_r","gt",0.80)}
FAM_A = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]


def fetch_before(end_ts_ms, n_bars, inst, bar="1D", page_limit=200):
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


def quant_mask(f, conds, win=250, rel_vol=1.5):
    """Data-driven signal mask (timeframe-adaptive) for daily bars."""
    out = pd.Series(False, index=f.index)
    for i in range(win, len(f)):
        sub = f.iloc[i-win:i]; ok = True
        for cid in conds:
            col, d, q = CID[cid]
            if col not in f.columns: ok=False; break
            thr = float(sub[col].quantile(q)); v = f[col].iloc[i]
            if pd.isna(v): ok=False; break
            ok = (v < thr) if d=="lt" else (v > thr)
            if not ok: break
        if ok:
            ok = (f["rel_vol"].iloc[i] > rel_vol) and (f["close"].iloc[i] > f["open"].iloc[i]) \
                 and (f["close"].iloc[i] > f["close"].iloc[i-1])
        out.iloc[i] = ok
    return out


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


print("[load] fetch 1D 2023-2026 for 20-sym …", flush=True)
end_ms = int(pd.Timestamp.now(tz="UTC").timestamp()*1000) - 1
feats, above20 = {}, {}
for sym in SUBSET:
    try:
        df = fetch_before(end_ms, 1400, sym.replace("_","-"), "1D")
        if df is None or len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f; above20[sym] = (f["close"] > f["ema20"]).astype(float)
            print(f"  loaded {sym}: {len(f)} 1D bars", flush=True)
    except Exception as e:
        print(f"  err {sym}: {e}", flush=True)
print(f"[load] subset: {len(feats)}", flush=True)
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100

# MR (quantile mask) + SVM on 1D
mr_raw = []
for s in feats:
    mask = quant_mask(feats[s], FAM_A)
    for t in sim_symbol(feats[s], mask, 1.5, dict(entry_next=False, exit="base", hours=None)):
        t["sym"]=s; mr_raw.append(t)
mr_raw.sort(key=lambda t: t["entry_time"])
mr_mldf = build_mldf(mr_raw, feats, breadth, breadth_pct)
mr_trades = []
for Y in [2024,2025,2026]:
    tr=mr_mldf[mr_mldf.ts<pd.Timestamp(f"{Y}-01-01",tz="UTC")]; te=mr_mldf[mr_mldf.ts.dt.year==Y]
    if len(tr)<50 or len(te)==0: continue
    m=SVMQ65Adaptive().fit_mldf(tr); kept,_=m.keep_mldf(te)
    for t in mr_raw:
        if t["entry_time"].year==Y and t["entry_time"] in kept: mr_trades.append(t)
mr_adj = cost_adjusted_rs(list(mr_trades), 0.05)
for t,a in zip(mr_trades,mr_adj): t["adj_r"]=a

# Trend on 1D
tr_trades = []
for s in feats:
    for t in backtest_donchian(feats[s]):
        t["sym"]=s; tr_trades.append(t)
for t in tr_trades: t["adj_r"]=t["r"]-2*FEE
print(f"[bt] MR trades={len(mr_trades)}  Trend trades={len(tr_trades)}", flush=True)

def report(name, trades):
    print(f"\n--- {name} (1D) ---")
    for Y in [2024,2025,2026]:
        yt=[t for t in trades if t["entry_time"].year==Y]
        if not yt: print(f"[{Y}] no trades"); continue
        eq=1.0;peak=1.0;mdd=0.0
        for t in sorted(yt,key=lambda x:x["entry_time"]):
            eq*=(1+0.01*t["adj_r"]); peak=max(peak,eq); mdd=min(mdd,eq/peak-1)
        rs=[t["adj_r"] for t in yt]
        wins=sum(1 for r in rs if r>0)/len(rs)
        pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
        msum=defaultdict(float)
        for t in yt: msum[(t["entry_time"].year,t["entry_time"].month)]+=t["adj_r"]
        pos=sum(1 for v in msum.values() if v>0)
        print(f"[{Y}] n={len(yt)} win={wins:.0%} PF@c={pf:.3f} MAX DD={mdd:.1%} prof-months={pos}/{len(msum)}")

report("MEAN-REVERSION (quantile mask + SVM adaptive VolCeil)", mr_trades)
report("TREND (Donchian 1D)", tr_trades)
print("\n[done]")
