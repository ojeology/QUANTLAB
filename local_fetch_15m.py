#!/usr/bin/env python3
"""
RUN THIS ON YOUR MACHINE (where Binance/Yahoo are reachable). It fetches 3-YEAR 15m
for the core symbols and saves parquet files. Then upload the data_15m/ folder to
GitHub (or any URL the sandbox can reach) and tell the agent to pull + validate.

Usage:  pip install pandas requests pyarrow
         python local_fetch_15m.py
Output:  data_15m/{SYMBOL}_15m.parquet   (datetime index, OHLCV)

Sources: tries Yahoo first, then Binance, for each symbol.
"""
import os, time, requests
import pandas as pd

CORE = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
        "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
        "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
        "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]
HDR = {"User-Agent":"Mozilla/5.0"}
os.makedirs("data_15m", exist_ok=True)


def from_yahoo(sym):
    y = sym.split("_")[0] + "-USD"
    rows = []
    for yr in [2023,2024,2025,2026]:
        p1=int(pd.Timestamp(f"{yr}-01-01",tz="UTC").timestamp())
        p2=int(pd.Timestamp(f"{yr+1}-01-01",tz="UTC").timestamp()) if yr<2026 else int(pd.Timestamp.now(tz="UTC").timestamp())
        r=requests.get("https://query1.finance.yahoo.com/v8/finance/chart/"+y,
                       params={"period1":p1,"period2":p2,"interval":"15m","events":"div,split"},
                       headers=HDR,timeout=30).json()
        res=(r.get("chart") or {}).get("result")
        if not res: return None
        res=res[0]; ts=res["timestamp"]; q=res["indicators"]["quote"][0]
        for i,t in enumerate(ts):
            o,h,l,c,v=q["open"][i],q["high"][i],q["low"][i],q["close"][i],q["volume"][i]
            if c is None: continue
            rows.append([int(t)*1000,o,h,l,c,v])
        time.sleep(0.3)
    return rows


def from_binance(sym):
    b = sym.split("_")[0] + "USDT"
    rows=[]; start=int(pd.Timestamp("2023-01-01",tz="UTC").timestamp()*1000)
    end=int(pd.Timestamp.now(tz="UTC").timestamp()*1000); step=900000
    while start<end:
        r=requests.get("https://api.binance.com/api/v3/klines",
                       params={"symbol":b,"interval":"15m","startTime":start,"limit":1000},timeout=30).json()
        if not r or not isinstance(r,list): break
        for k in r: rows.append([k[0],k[1],k[2],k[3],k[4],k[5]])
        start=r[-1][0]+step
        if len(r)<1000: break
        time.sleep(0.05)
    return rows


for sym in CORE:
    rows = from_yahoo(sym)
    if not rows or len(rows) < 50000:
        print(f"  {sym}: yahoo gave {len(rows) if rows else 0}, trying binance", flush=True)
        rows = from_binance(sym)
    if not rows or len(rows) < 50000:
        print(f"  {sym}: FAILED ({len(rows) if rows else 0} bars) -- skip", flush=True); continue
    df=pd.DataFrame(rows,columns=["ts","open","high","low","close","vol"])
    df["ts"]=pd.to_numeric(df["ts"]); df["datetime"]=pd.to_datetime(df["ts"],unit="ms",utc=True)
    for c in ["open","high","low","close","vol"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["close"]).set_index("datetime").sort_index()
    df.to_parquet(f"data_15m/{sym}_15m.parquet")
    print(f"  saved {sym}: {len(df)} bars {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
print("[done] upload the data_15m/ folder to GitHub and tell the agent to pull + validate.")
