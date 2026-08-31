"""
Fetch additional 1H USDT-SWAP history from OKX to expand the research universe.

- Picks liquid USDT swaps NOT already in quantlab_cache (by 24h volume)
- Pages backward via the 'after' param (verified: reaches Dec 2023 for BTC)
- Saves as quantlab_cache/SYMBOL_1H.parquet matching the existing schema
  (index=datetime UTC, cols open/high/low/close/vol float64)
"""
import os, sys, time, json
import urllib.request, urllib.error
import pandas as pd
import numpy as np

sys.path.insert(0, "/home/user/quantlab")
from quantlab_ai import CONFIG

CACHE = CONFIG["CACHE_FOLDER"]
MAX_NEW = 24
MAX_BARS_PER_SYM = 17_500   # ~2 years of 1H
LIMIT = 100
SLEEP = 0.12
UA = {"User-Agent": "Mozilla/5.0"}

def get_json(url, retries=4):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 + a)
                continue
            raise
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(1 + a)
    return {}

def fetch_symbol_history(inst_id, max_bars=MAX_BARS_PER_SYM):
    """Page backward from the newest 1H candle until we have max_bars or API stops."""
    base = (f"https://www.okx.com/api/v5/market/history-candles"
            f"?instId={inst_id}&bar=1H&limit={LIMIT}")
    d = get_json(base)
    c = d.get("data", [])
    if not c:
        return None
    rows = c
    ts = min(int(x[0]) for x in c)
    # OKX returns newest-first; extend backward using after=oldest_ts
    while len(rows) < max_bars:
        d2 = get_json(base + f"&after={ts}")
        c2 = d2.get("data", [])
        if not c2:
            break
        new_min = min(int(x[0]) for x in c2)
        if new_min >= ts:
            break
        rows.extend(c2)
        ts = new_min
        time.sleep(SLEEP)
    # dedupe by ts, sort ascending
    seen = {}
    for x in rows:
        seen[int(x[0])] = x
    items = sorted(seen.values(), key=lambda x: int(x[0]))
    df = pd.DataFrame([
        dict(datetime=pd.to_datetime(int(x[0]), unit="ms", utc=True),
             open=float(x[1]), high=float(x[2]), low=float(x[3]),
             close=float(x[4]), vol=float(x[5]))
        for x in items
    ])
    df = df.set_index("datetime").sort_index()
    df.index = df.index.tz_convert("UTC")
    return df

def main():
    # current cache symbols
    have = set(f.replace("_1H.parquet", "") for f in os.listdir(CACHE)
               if f.endswith("_1H.parquet"))
    print(f"Already have {len(have)} symbols")

    # liquid USDT swaps by 24h volume
    t = get_json("https://www.okx.com/api/v5/market/tickers?instType=SWAP")
    rows = []
    for x in t.get("data", []):
        iid = x["instId"]
        if not iid.endswith("-USDT-SWAP"):
            continue
        try:
            vol24 = float(x.get("volCcy24h") or 0) * float(x.get("last") or 0)
        except Exception:
            vol24 = 0.0
        rows.append((iid, vol24))
    rows.sort(key=lambda r: -r[1])
    candidates = [iid for iid, v in rows if iid not in have and v >= 5_000_000]
    picks = candidates[:MAX_NEW]
    print(f"Liquid USDT swaps not cached: {len(candidates)} | picking {len(picks)}:")
    for p in picks:
        print("  ", p)

    fetched = 0
    for iid in picks:
        sym = iid.replace("-", "_")
        out = os.path.join(CACHE, f"{sym}_1H.parquet")
        if os.path.exists(out):
            print(f"  skip (exists): {sym}")
            continue
        try:
            df = fetch_symbol_history(iid)
            if df is None or len(df) < 500:
                print(f"  {sym}: too little data ({0 if df is None else len(df)} bars) — skip")
                continue
            df.to_parquet(out)
            fetched += 1
            print(f"  ✓ {sym}: {len(df)} bars  {df.index.min()} → {df.index.max()}")
        except Exception as e:
            print(f"  ✗ {sym}: {type(e).__name__} {str(e)[:120]}")
        time.sleep(SLEEP)

    print(f"\nDone. Fetched {fetched} new symbols → total cache now "
          f"{len([f for f in os.listdir(CACHE) if f.endswith('_1H.parquet')])}")

if __name__ == "__main__":
    main()
