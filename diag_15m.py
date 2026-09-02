"""Diagnostic: fetch BTC 15m for 3 years SEQUENTIALLY (1 worker) and report how many
bars actually come back. Determines if 3yr 15m is blocked by RATE LIMIT (sequential
works -> get more) or DATA CAP (OKX only serves ~7mo of 15m -> capped)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import pandas as pd
import demo_bot as bot

CHUNK=20000; CHUNKS=6
def fetch_before(end_ms, n_bars, inst, bar="15m", page_limit=200):
    all_rows, after, pages = [], end_ms, 0
    while len(all_rows) < n_bars and pages < page_limit:
        params={"instId":inst,"bar":bar,"limit":bot.PAGE_LIMIT,"after":str(after)}
        raw=bot._get(bot.OKX_CANDLES,params) or bot._get(bot.OKX_CANDLES_CUR,params)
        if not raw:
            print("   empty response at page",pages, flush=True); break
        all_rows.extend(raw); pages+=1; after=int(raw[-1][0])
        if len(all_rows)>=n_bars: break
        time.sleep(bot.PAGE_DELAY)
    if not all_rows: return None
    df=pd.DataFrame(all_rows,columns=bot.CANDLE_COLS); df["ts"]=pd.to_numeric(df["ts"])
    for c in ["open","high","low","close","vol"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["datetime"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
    return df[["datetime","open","high","low","close","vol"]].sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True).set_index("datetime")

inst="BTC-USDT-SWAP"
end_ms=int(pd.Timestamp.now(tz="UTC").timestamp()*1000)-1
chunks=[]
for c in range(CHUNKS):
    df=fetch_before(end_ms,CHUNK,inst,"15m",page_limit=200)
    if df is None or len(df)==0:
        print(f"   chunk {c} returned NONE/empty", flush=True); break
    print(f"   chunk {c}: {len(df)} bars {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
    chunks.append(df); end_ms=int(df.index[0].timestamp()*1000)-1
    if len(chunks)*CHUNK>=105000: break
if chunks:
    full=pd.concat(chunks); full=full[~full.index.duplicated(keep="last")].sort_index()
    print(f"\nTOTAL BTC 15m: {len(full)} bars  {full.index[0].date()} -> {full.index[-1].date()}", flush=True)
else:
    print("\nBTC 15m: NO DATA at all", flush=True)
