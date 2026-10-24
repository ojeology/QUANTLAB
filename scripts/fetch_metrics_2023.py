# R120 data fetch: 2023 daily metrics (sum_open_interest) + funding extension to 2023-01.
import io, time, pathlib, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

OUT = pathlib.Path("quantlab_cache/bnmetrics"); OUT.mkdir(parents=True, exist_ok=True)
FUND = pathlib.Path("quantlab_cache/bnfut")
SYMS = ["BTC","ETH","BNB","SOL","XRP","DOGE","ADA","LINK","LTC","AVAX","DOT","NEAR",
        "APT","SUI","ATOM","FIL","INJ","ARB","OP","ETC","TRX","XLM","CRV","AAVE"]
DAYS = pd.date_range("2023-01-01","2023-12-31",freq="D").strftime("%Y-%m-%d").tolist()
FMONTHS = [(2023,m) for m in range(1,9)]

def get(url):
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as r: return r.read()
        except Exception:
            time.sleep(1.0*(k+1))
    return None

def mjob(sym, day):
    b = get(f"https://data.binance.vision/data/futures/um/daily/metrics/{sym}USDT/{sym}USDT-metrics-{day}.zip")
    if not b: return sym, day, None
    try:
        zf = zipfile.ZipFile(io.BytesIO(b)); name = zf.namelist()[0]
        df = pd.read_csv(zf.open(name))
        df.columns = [c.strip().lower() for c in df.columns]
        col_t = "create_time" if "create_time" in df.columns else df.columns[0]
        df = df.sort_values(col_t)
        last = df.iloc[-1]
        oi = pd.to_numeric(last["sum_open_interest"], errors="coerce")
        oiv = pd.to_numeric(last["sum_open_interest_value"], errors="coerce")
        return sym, day, (float(oi), float(oiv))
    except Exception:
        return sym, day, None

def fjob(sym, y, m):
    tag = f"{y}-{m:02d}"
    b = get(f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}USDT/{sym}USDT-fundingRate-{tag}.zip")
    if not b: return sym, None
    try:
        zf = zipfile.ZipFile(io.BytesIO(b)); name = zf.namelist()[0]
        df = pd.read_csv(zf.open(name), header=None)
        rows = []
        for _, r in df.iterrows():
            ts = pd.to_numeric(r.iloc[0], errors="coerce"); v = pd.to_numeric(r.iloc[-1], errors="coerce")
            if pd.notna(ts) and pd.notna(v): rows.append((float(ts), float(v)))
        return sym, rows
    except Exception:
        return sym, None

t0 = time.time()
# funding extension first (fast)
added = 0
for s in SYMS:
    fp = FUND / f"{s}_fund.csv"
    old = pd.read_csv(fp)
    acc = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fu in as_completed([ex.submit(fjob, s, y, m) for y, m in FMONTHS]):
            sym, rows = fu.result()
            if rows: acc += rows
    new = pd.DataFrame(acc, columns=["ts_ms","rate"])
    allf = pd.concat([old, new]).drop_duplicates("ts_ms").sort_values("ts_ms")
    added += len(allf) - len(old)
    allf.to_csv(fp, index=False)
print(f"FUNDING EXTENSION: +{added} epochs in {time.time()-t0:.0f}s", flush=True)

# 2023 daily metrics
t1 = time.time()
for i, s in enumerate(SYMS, 1):
    res = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fu in as_completed([ex.submit(mjob, s, d) for d in DAYS]):
            sym, day, val = fu.result()
            if val is not None: res[day] = val
    df = pd.DataFrame([(d, *res[d]) for d in sorted(res)], columns=["date","oi","oi_usd"])
    df.to_csv(OUT / f"{s}_oi_daily.csv", index=False)
    print(f"[{i}/{len(SYMS)}] {s}: {len(df)}/365 days", flush=True)
print(f"METRICS 2023 DONE in {time.time()-t1:.0f}s", flush=True)
