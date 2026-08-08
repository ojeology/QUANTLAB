"""
Fetch Forex data from Yahoo Finance (free, no key).
  1H : ~2.9 years (2023-10 → present) — the primary hunt timeframe
  1D : ~10 years (2016 → present) — for higher-TF context filters
Saves to quantlab_cache as FOREX_<PAIR>_1H.parquet / _1D.parquet
(Yahoo symbol = PAIR=X, e.g. EURUSD=X)
"""
import os, sys, json, time
import urllib.request
import pandas as pd
import numpy as np

CACHE = "/home/user/quantlab/quantlab_cache"
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def yahoo_chart(symbol, interval, range_str):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?interval={interval}&range={range_str}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read().decode())
    res = d["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "vol": q.get("volume"),
    }, index=pd.to_datetime(ts, unit="s", utc=True))
    df = df[["open","high","low","close","vol"]].dropna()
    df.index.name = "datetime"
    return df

def main():
    print("Fetching 1H and 1D for 8 FX pairs …", flush=True)
    for pair in PAIRS:
        sym = pair + "=X"
        for interval, rng, suffix in [("1h","730d","1H"), ("1d","10y","1D")]:
            out = os.path.join(CACHE, f"FOREX_{pair}_{suffix}.parquet")
            try:
                df = yahoo_chart(sym, interval, rng)
                df.to_parquet(out)
                print(f"  ✓ {pair} {suffix}: {len(df)} bars  {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}", flush=True)
            except Exception as e:
                print(f"  ✗ {pair} {suffix}: {str(e)[:80]}", flush=True)
            time.sleep(0.4)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
