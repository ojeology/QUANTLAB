"""
Fetch ~10 years of 1H forex data for 8 majors from the Dukascopy public datafeed.

Endpoint (free, no key):
  https://datafeed.dukascopy.com/datafeed/{PAIR}/{YYYY}/{MM-1:02d}/{DD:02d}/BID_candles_min_1.bi5
  - month is 0-INDEXED (00 = January); day 1-indexed
  - file = LZMA blob of 24-byte big-endian records:
      >IIIIIf = sec-of-day, open, close, low, high  (uint32, price*point), volume (float32)
      point = 1e-5 (5-digit pairs) / 1e-3 (JPY pairs)
  - weekends/holidays return 404 or empty

Output: quantlab_cache/forex10/{PAIR}_1H.parquet  (UTC index, open/high/low/close/vol)
Method: parallel (pair, year) units; each day downloaded via curl (urllib TLS is
blocked from this sandbox); m1 bars resampled to 1H. Resumable at (pair,year) level.
"""
import os, io, lzma, struct, subprocess, sys, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

PAIRS = {"EURUSD":1e5, "GBPUSD":1e5, "USDJPY":1e3, "AUDUSD":1e5,
         "USDCAD":1e5, "USDCHF":1e5, "NZDUSD":1e5, "EURGBP":1e5}
START = dt.date(2016, 1, 1)
END   = dt.date(2026, 10, 9)
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "quantlab_cache", "forex10")
os.makedirs(OUTDIR, exist_ok=True)

BASE = "https://datafeed.dukascopy.com/datafeed/{p}/{y:04d}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}

def fetch_day(pair, day, tries=3):
    url = BASE.format(p=pair, y=day.year, m=day.month-1, d=day.day)
    for a in range(tries):
        try:
            p = subprocess.run(["curl","-s","--max-time","25","-A",UA["User-Agent"],
                                "--write-out","\n%{http_code}", url],
                               capture_output=True, timeout=30)
            out = p.stdout
            if b"\n" not in out:
                time.sleep(1+a*2); continue
            body, _, code = out.rpartition(b"\n")
            if code.strip() == b"404":
                return None
            if code.strip() == b"200" and len(body) > 0:
                try:
                    return lzma.decompress(body)
                except Exception:
                    time.sleep(1+a*2); continue
            time.sleep(1+a*2)
        except Exception:
            time.sleep(1+a*2)
    return None

REC = struct.Struct(">IIIIIf")

def parse_m1(raw, day, scale):
    if not raw: return None
    n = len(raw)//24
    if n == 0: return None
    arr = np.frombuffer(raw[:n*24], dtype=np.dtype(">u4,>u4,>u4,>u4,>u4,>f4"))
    secs  = arr["f0"].astype(np.int64)
    o = arr["f1"].astype(np.float64)/scale
    c = arr["f2"].astype(np.float64)/scale
    l = arr["f3"].astype(np.float64)/scale
    h = arr["f4"].astype(np.float64)/scale
    v = arr["f5"].astype(np.float64)
    base = pd.Timestamp(day, tz="UTC")
    idx = base + pd.to_timedelta(secs, unit="s")
    df = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"vol":v}, index=idx)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def fetch_unit(pair, year):
    scale = PAIRS[pair]
    d0, d1 = max(START, dt.date(year,1,1)), min(END, dt.date(year,12,31))
    parts=[]
    d = d0
    while d <= d1:
        raw = fetch_day(pair, d)
        if raw:
            df = parse_m1(raw, d, scale)
            if df is not None and len(df): parts.append(df)
        d += dt.timedelta(days=1)
    if not parts: return pair, year, 0
    m1 = pd.concat(parts)
    h1 = m1.resample("1H").agg({"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()
    h1 = h1[(h1["open"]>0)&(h1["high"]>=h1["low"])&(h1["low"]>0)]
    return pair, year, len(h1), h1

if __name__ == "__main__":
    only = sys.argv[1:] if len(sys.argv)>1 else None
    pairs = {k:v for k,v in PAIRS.items() if only is None or k in only}
    units = [(p,y) for p in pairs for y in range(START.year, END.year+1)]
    print(f"Fetching {len(units)} (pair,year) units "
          f"({START} → {END}) with 12 workers …", flush=True)
    results = {}
    t0=time.time()
    with ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(fetch_unit, p, y): (p,y) for p,y in units}
        done=0
        for f in as_completed(futs):
            p,y = futs[f]
            try:
                r = fetch_unit.__wrapped__ if False else f.result()
            except Exception as e:
                print(f"  ✗ {p} {y}: {type(e).__name__} {str(e)[:80]}", flush=True); continue
            pair, year, nbars = r[0], r[1], r[2]
            h1 = r[3] if len(r)>3 else None
            results[(pair,year)] = h1
            done+=1
            el=time.time()-t0
            print(f"  [{done:>3}/{len(units)}] {pair} {year}: {nbars:>6} 1H bars   ({el:.0f}s)", flush=True)
    # assemble per pair
    print("\nAssembling per-pair parquet …", flush=True)
    for p in pairs:
        dfs = [results[(p,y)] for y in range(START.year, END.year+1) if results.get((p,y)) is not None and len(results[(p,y)])]
        if not dfs:
            print(f"  ✗ {p}: NO DATA"); continue
        full = pd.concat(dfs).sort_index()
        full = full[~full.index.duplicated(keep="last")]
        out = os.path.join(OUTDIR, f"{p}_1H.parquet")
        full.to_parquet(out)
        print(f"  ✓ {p}: {len(full):>7} bars  {full.index.min():%Y-%m-%d} → {full.index.max():%Y-%m-%d}  → {out}", flush=True)
    print(f"Done in {time.time()-t0:.0f}s")
