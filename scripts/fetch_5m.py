"""Fetch 5m 1H-equivalent data for the 8 majors that have 15m cache (OKX serves deep history)."""
import os, sys, time, json
import urllib.request, urllib.error
import pandas as pd

CACHE = "/home/user/quantlab/quantlab_cache"
UA = {"User-Agent": "Mozilla/5.0"}
SLEEP = 0.05
LIMIT = 100
MAX_BARS = 55_000   # ~6 months of 5m

SYMS = ["BTC-USDT-SWAP","ETH-USDT-SWAP","DOGE-USDT-SWAP","LINK-USDT-SWAP",
        "LTC-USDT-SWAP","XRP-USDT-SWAP","BCH-USDT-SWAP","AVAX-USDT-SWAP"]

def get_json(url, retries=4):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 + a); continue
            raise
        except Exception:
            if a == retries - 1: raise
            time.sleep(1 + a)
    return {}

def fetch_symbol(inst_id, max_bars=MAX_BARS):
    base = (f"https://www.okx.com/api/v5/market/history-candles"
            f"?instId={inst_id}&bar=5m&limit={LIMIT}")
    d = get_json(base); c = d.get("data", [])
    if not c: return None
    rows = c; ts = min(int(x[0]) for x in c)
    while len(rows) < max_bars:
        d2 = get_json(base + f"&after={ts}"); c2 = d2.get("data", [])
        if not c2: break
        nm = min(int(x[0]) for x in c2)
        if nm >= ts: break
        rows.extend(c2); ts = nm; time.sleep(SLEEP)
    seen = {}
    for x in rows: seen[int(x[0])] = x
    items = sorted(seen.values(), key=lambda x: int(x[0]))
    df = pd.DataFrame([dict(datetime=pd.to_datetime(int(x[0]), unit="ms", utc=True),
                            open=float(x[1]), high=float(x[2]), low=float(x[3]),
                            close=float(x[4]), vol=float(x[5])) for x in items])
    df = df.set_index("datetime").sort_index(); df.index = df.index.tz_convert("UTC")
    return df

t0 = time.time()
for iid in SYMS:
    sym = iid.replace("-", "_")
    out = os.path.join(CACHE, f"{sym}_5m.parquet")
    try:
        df = fetch_symbol(iid)
        if df is None or len(df) < 10_000:
            print(f"  skip {sym} ({0 if df is None else len(df)} bars)")
            continue
        df.to_parquet(out)
        print(f"  ✓ {sym}: {len(df)} bars  {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
    except Exception as e:
        print(f"  ✗ {sym}: {str(e)[:90]}")
    time.sleep(0.1)
print(f"DONE in {time.time()-t0:.0f}s. 5m files: "
      f"{len([f for f in os.listdir(CACHE) if f.endswith('_5m.parquet')])}")
