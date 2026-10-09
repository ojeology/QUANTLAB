"""
Refresh all 1H crypto parquets in quantlab_cache forward from each file's last bar.
Source: OKX public /market/candles (newest-first, paginate 'after'=<oldest seen>).
Keeps schema: index UTC datetime, cols open/high/low/close/vol float64.
Delisted/empty symbols: kept as-is and reported.
"""
import os, time, json, glob, subprocess
import pandas as pd

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "quantlab_cache")
UA = "Mozilla/5.0"

def get(url, tries=4):
    for a in range(tries):
        p = subprocess.run(["curl","-s","-4","--max-time","20","-A",UA,url], capture_output=True, timeout=30)
        try:
            d = json.loads(p.stdout.decode())
            if d.get("code") == "0": return d["data"]
            if d.get("code") == "429": time.sleep(1+a); continue
            return []
        except Exception:
            time.sleep(1+a)
    return []

files = sorted(glob.glob(os.path.join(CACHE,"*_1H.parquet")))
print(f"{len(files)} symbols to refresh", flush=True)
t0=time.time(); done=0
for fp in files:
    sym = os.path.basename(fp).replace("_1H.parquet","")
    df = pd.read_parquet(fp)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    last = df.index.max()
    rows={}
    after=None
    last_ms = int(last.value//10**6)
    for page in range(25):
        url = f"https://www.okx.com/api/v5/market/candles?instId={sym.replace(chr(95), chr(45))}&bar=1H&limit=100"
        if after: url += f"&after={after}"
        data = get(url)
        if not data: break
        newest = int(data[0][0])
        oldest = int(data[-1][0])
        for r in data:
            ts = pd.Timestamp(int(r[0]), unit="ms", tz="UTC")
            if ts <= last: continue
            rows[ts] = [float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])]
        if oldest <= last_ms + 1 or len(data) < 100:
            break
        after = oldest
        time.sleep(0.1)
    done+=1
    if rows:
        add = pd.DataFrame.from_dict(rows, orient="index", columns=["open","high","low","close","vol"]).sort_index()
        add = add[~add.index.duplicated(keep="last")]
        full = pd.concat([df, add]); full = full[~full.index.duplicated(keep="last")].sort_index()
        full.to_parquet(fp)
        print(f"  [{done:>2}/{len(files)}] {sym}: +{len(add):>5} bars → {full.index.max():%Y-%m-%d %H:%M}  ({time.time()-t0:.0f}s)", flush=True)
    else:
        print(f"  [{done:>2}/{len(files)}] {sym}: no new data (last {last:%Y-%m-%d %H:%M})  ({time.time()-t0:.0f}s)", flush=True)
print(f"Done in {time.time()-t0:.0f}s")
