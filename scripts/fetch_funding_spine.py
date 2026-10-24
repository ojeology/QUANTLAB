# Funding-rate monthly spine for the R115/R116 corrected reruns + R120 OI-momentum.
# Source: Binance Vision monthly fundingRate zips. Saves quantlab_cache/bnfut/{SYM}_fund.csv
import io, time, pathlib, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

OUT = pathlib.Path("quantlab_cache/bnfut"); OUT.mkdir(parents=True, exist_ok=True)
SYMS = ["1INCH","AAVE","ADA","ALGO","APT","ARB","ATOM","AVAX","BNB","BTC","CRV","DOGE","DOT",
        "ETC","ETH","FET","FIL","GRT","HBAR","INJ","LDO","LINK","LTC","NEAR","OP","SEI","SOL",
        "SUI","TRX","UNI","XLM","XRP"]
MONTHS = [(y,m) for y in range(2023,2027) for m in range(1,13)
          if pd.Timestamp(f"{y}-{m:02d}-01") >= pd.Timestamp("2023-09-01")
          and pd.Timestamp(f"{y}-{m:02d}-01") <= pd.Timestamp("2026-09-01")]

def get(url):
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as r: return r.read()
        except Exception:
            time.sleep(1.2*(k+1))
    return None

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
for i, s in enumerate(SYMS, 1):
    acc = []; fails = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for fu in as_completed([ex.submit(fjob, s, y, m) for y, m in MONTHS]):
            sym, rows = fu.result()
            if rows is None: fails += 1
            else: acc += rows
    f = pd.DataFrame(acc, columns=["ts_ms","rate"]).drop_duplicates("ts_ms").sort_values("ts_ms")
    f.to_csv(OUT / f"{s}_fund.csv", index=False)
    print(f"[{i}/{len(SYMS)}] {s}: {len(f)} epochs ({fails} month-fails)", flush=True)
print(f"FUNDING SPINE DONE in {time.time()-t0:.0f}s", flush=True)
