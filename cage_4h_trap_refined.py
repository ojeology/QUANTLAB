"""
TEST 23 — 4H-CAGE TRAP refined: ONE bounce per 4H bar. Use the 4H candle's
high/low as the cage; on 1H, when price TAGS the 4H extreme and then BOUNCES
(reversal bar), take it — but at most ONE long and ONE short per 4H bar (the real
"trap then bounce", not every micro-tag). 1H, 20-sym, 2024-2026 (2023 fetched in-run).
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from ql_engine import add_features, IS_LOOKBACK, RECAL_EVERY
import demo_bot as bot

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


def backtest_4h_trap_refined(df):
    df = df.copy()
    h4 = df.resample("4H").agg(high=("high","max"), low=("low","min"))
    bucket = df.index.floor("4H")
    cage_high = h4["high"].reindex(bucket).values
    cage_low  = h4["low"].reindex(bucket).values
    atr = df["atr14"].values
    trades = []
    in_pos=False; side=None; ep=None; stop=None; target=None
    long_taken={}; short_taken={}; pend_low={}; pend_high={}; prev_b=None
    for i in range(4, len(df)):
        b = bucket[i]
        if b != prev_b:
            long_taken[b]=False; short_taken[b]=False; pend_low[b]=False; pend_high[b]=False; prev_b=b
        o,h,l,c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
        ch = cage_high[i]; cl = cage_low[i]; a = atr[i]
        if not in_pos:
            if not long_taken[b]:
                if l <= cl:
                    if (c > cl) and (c > o) and a > 0:
                        ep=c; stop=cl-a; target=(ch+cl)/2.0; side="L"; in_pos=True; long_taken[b]=True
                    else:
                        pend_low[b]=True
                elif pend_low[b] and (c > cl) and (c > o) and a > 0:
                    ep=c; stop=cl-a; target=(ch+cl)/2.0; side="L"; in_pos=True; long_taken[b]=True; pend_low[b]=False
            if not in_pos and not short_taken[b]:
                if h >= ch:
                    if (c < ch) and (c < o) and a > 0:
                        ep=c; stop=ch+a; target=(ch+cl)/2.0; side="S"; in_pos=True; short_taken[b]=True
                    else:
                        pend_high[b]=True
                elif pend_high[b] and (c < ch) and (c < o) and a > 0:
                    ep=c; stop=ch+a; target=(ch+cl)/2.0; side="S"; in_pos=True; short_taken[b]=True; pend_high[b]=False
        else:
            ex=None; et=None
            if side=="L":
                if l <= stop: ex=stop; et="SL"
                elif c >= target: ex=target; et="REV"
            else:
                if h >= stop: ex=stop; et="SL"
                elif c <= target: ex=target; et="REV"
            if ex is not None:
                r = (ex/ep - 1.0) if side=="L" else (1.0 - ex/ep)
                trades.append(dict(entry_time=df.index[i], r=r)); in_pos=False
    return trades


print("[load] fetch 2023 + cache 2024-2026 (20-sym) …", flush=True)
feats = {}
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
        if len(df)<IS_LOOKBACK+RECAL_EVERY+4: continue
        f=add_features(df); f.dropna(subset=["atr14"], inplace=True)
        if len(f)>=IS_LOOKBACK+RECAL_EVERY: feats[sym]=f
        print(f"  loaded {sym}: {len(f)} bars", flush=True)
    except Exception as e: print(f"  err {sym}: {e}", flush=True)
print(f"[load] subset: {len(feats)}", flush=True)

all_trades = []
for s in feats:
    for t in backtest_4h_trap_refined(feats[s]):
        t["sym"]=s; all_trades.append(t)
all_trades.sort(key=lambda t: t["entry_time"])
for t in all_trades: t["adj_r"]=t["r"]-2*FEE
print(f"[bt] 4H-cage refined trades 2023-2026: {len(all_trades)}", flush=True)


def report():
    print("\n"+"="*80)
    print("TEST 23 — 4H-CAGE TRAP refined (ONE bounce per 4H bar)")
    print("="*80)
    for Y in [2024,2025,2026]:
        yt=[t for t in all_trades if t["entry_time"].year==Y]
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
        print(f"\n[{Y}] n={len(yt)} win={wins:.0%} PF@c={pf:.3f} MAX DD={mdd:.1%} prof-months={pos}/{len(msum)}")
        for (yy,mm) in sorted(msum):
            print(f"   {yy}-{mm:02d}  netR={msum[(yy,mm)]:>+7.2f}  {'OK' if msum[(yy,mm)]>0 else 'LOSS'}")
    rs=[t["adj_r"] for t in all_trades]
    pf=(sum(r for r in rs if r>0))/max(1e-9,-sum(r for r in rs if r<0))
    print(f"\n[FULL 2024-2026] n={len(all_trades)} PF@c={pf:.3f}")
report()
print("\n[done]")
