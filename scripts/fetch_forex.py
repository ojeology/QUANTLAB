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

def robust_fetch(pair, interval, rng, tries=4):
    import urllib.error
    for k in range(tries):
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{pair}=X?interval={interval}&range={rng}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            print(f"  retry {k+1} for {pair} {interval} ({e})"); time.sleep(6 * (k + 1))
    raise RuntimeError(f"give up {pair} {interval}")

for p in PAIRS:
    f = OUT / f"{p}_1H.csv"
    if f.exists() and (time.time() - f.stat().st_mtime) < 86400:
        print(f"{p}: cached 1H"); continue
    df = fetch(p)
    df.to_csv(f, index=False)
    print(f"{p}: {len(df)} bars  {df.ts.iloc[0]} -> {df.ts.iloc[-1]}")
    time.sleep(1.5)
print("DONE")

# --- daily bars (F3 CTA-horizon needs decade lookback warmup) ---
for p in PAIRS:
    f1d = OUT / f"{p}_1D.csv"
    if f1d.exists() and (time.time() - f1d.stat().st_mtime) < 86400:
        print(f"{p} 1D: cached"); continue
    d = robust_fetch(p, "1d", "10y")
    res = d["chart"]["result"][0]; q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"ts": pd.to_datetime(res["timestamp"], unit="s").tz_localize(None),
                       "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]})
    df = df.dropna(subset=["close"]).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df.to_csv(f1d, index=False)
    print(f"{p} 1D: {len(df)} bars {df.ts.iloc[0].date()} -> {df.ts.iloc[-1].date()}")
    time.sleep(1.5)
