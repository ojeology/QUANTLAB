"""
TEST 20 — CAGE TRAP (false-breakout variant). Cage = EMA50 +/- 2.5*ATR band that
FOLLOWS RULES. TRAP = price breaks below the lower wall (close<L) then RECLAIMS it
(next close>=L) in a coiled/range regime -> fade the bull trap back toward center.
This is the "trap price" reading: price gets trapped outside the cage, then snaps back.
1H, 20-sym, 2024-2026 (2023 fetched in-run). PF@cost, MAX DD, profitable-months.
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


def backtest_cage_trap(df, k=2.5, adx_max=30.0, bb_q=0.30):
    df = df.copy()
    center = df["close"].ewm(span=50, adjust=False).mean()
    width = k * df["atr14"]
    L = center - width
    bbq = df["bb_width"].rolling(250).quantile(bb_q)
    trades = []; in_pos=False; ep=None; stop=None; target=None
    for i in range(250, len(df)):
        bar = df.iloc[i]; prev = df.iloc[i-1]
        if not in_pos:
            broke = prev["close"] < L.iloc[i-1]          # broke below cage
            reclaimed = bar["close"] >= L.iloc[i]          # snapped back in
            if broke and reclaimed and bar["adx14"] < adx_max and bar["bb_width"] <= bbq.iloc[i]:
                ep = bar["close"]; stop = L.iloc[i] - bar["atr14"]; target = center.iloc[i]; in_pos = True
        else:
            ex=None; et=None
            if bar["low"] <= stop: ex=stop; et="SL"
            elif bar["close"] >= target: ex=target; et="REV"
            if ex is not None:
                trades.append(dict(entry_time=df.index[i], r=(ex/ep - 1.0))); in_pos = False
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
        if len(df)<IS_LOOKBACK+RECAL_EVERY+250: continue
        f=add_features(df); f.dropna(subset=["ema200","atr14","adx14","bb_width"], inplace=True)
        if len(f)>=IS_LOOKBACK+RECAL_EVERY: feats[sym]=f
        print(f"  loaded {sym}: {len(f)} bars", flush=True)
    except Exception as e: print(f"  err {sym}: {e}", flush=True)
print(f"[load] subset: {len(feats)}", flush=True)

all_trades = []
for s in feats:
    for t in backtest_cage_trap(feats[s]):
        t["sym"]=s; all_trades.append(t)
all_trades.sort(key=lambda t: t["entry_time"])
for t in all_trades: t["adj_r"]=t["r"]-2*FEE
print(f"[bt] cage-trap trades 2023-2026: {len(all_trades)}", flush=True)

def report():
    print("\n"+"="*78)
    print("TEST 20 — CAGE TRAP (break below cage + reclaim -> fade back to center)")
    print("="*78)
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
