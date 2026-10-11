#!/usr/bin/env python3
"""Fetch 1H FX majors from Yahoo v8 chart API -> quantlab_cache/forex/{PAIR}_1H.parquet
Revival data source (Dukascopy TLS-dead, stooq JS-walled, HistData heavy)."""
import json, time, urllib.request, pathlib
import pandas as pd

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP"]
OUT = pathlib.Path("quantlab_cache/forex"); OUT.mkdir(parents=True, exist_ok=True)

def fetch(pair):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{pair}=X?interval=1h&range=730d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    res = d["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "ts": pd.to_datetime(res["timestamp"], unit="s"),
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
    })
    df = df.dropna(subset=["close"]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["ts"] = df["ts"].dt.tz_localize(None)  # naive UTC (session law)
    return df

for p in PAIRS:
    df = fetch(p)
    df.to_csv(OUT / f"{p}_1H.csv", index=False)  # CSV not parquet: pyarrow wiped per session
    print(f"{p}: {len(df)} bars  {df.ts.iloc[0]} -> {df.ts.iloc[-1]}")
    time.sleep(1.0)
print("DONE")
