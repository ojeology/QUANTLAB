"""
Fetch ~10 years of 1H forex data for 8 majors from HistData.com (free ASCII M1).

Flow (mirrors philipperemy/FX-1-Minute-Data):
  1) GET  https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair}/{year}[/{month}]
     -> HTML with hidden token input id="tk"
  2) POST https://www.histdata.com/get.php  (form: tk,date,datemonth,platform,timeframe,fxpair)
     -> zip (yearly for past years, monthly for current year). Browser UA REQUIRED.

Notes:
  - ASCII M1 = UTC timestamps "YYYYMMDD HHMMSS;O;H;L;C;V", V is always 0
    -> activity proxy: vol = high - low (documented in F012)
  - zips go to /tmp/hdzips (not workspace); parquet to quantlab_cache/forex10/
"""
import os, re, io, sys, time, zipfile, subprocess, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

PAIRS = ["eurusd","gbpusd","usdjpy","audusd","usdcad","usdchf","nzdusd","eurgbp"]
YEARS = list(range(int(os.environ.get("HD_Y0",2016)), int(os.environ.get("HD_Y1",2026))))
CUR_YEAR, CUR_MONTHS = 2026, [1,2,3,4,5,6,7,8,9,10]
ZIPDIR = "/tmp/hdzips"
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "quantlab_cache", "forex10")
os.makedirs(ZIPDIR, exist_ok=True); os.makedirs(OUTDIR, exist_ok=True)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
BASE = "https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/"

def curl(args, timeout=60):
    return subprocess.run(["curl","-s","-4","--max-time",str(timeout),"-A",UA] + args,
                          capture_output=True, timeout=timeout+10)

def download_one(pair, year, month=None, tries=4):
    tag = f"{pair.upper()}_{year}" + (f"{month:02d}" if month else "")
    zpath = os.path.join(ZIPDIR, f"{tag}.zip")
    if os.path.exists(zpath) and os.path.getsize(zpath) > 10_000:
        return zpath, "cached"
    url = f"{BASE}{pair}/{year}" + (f"/{month}" if month else "")
    referer = url
    for a in range(tries):
        r1 = curl(["--max-time","30", url, "-o","-"], timeout=40)
        if r1.returncode != 0 or len(r1.stdout) < 500:
            time.sleep(2+a*2); continue
        m = re.search(rb'id="tk"[^>]*value="([^"]+)"', r1.stdout)
        if not m:
            return None, f"no-token ({url})"
        tk = m.group(1).decode()
        data = f"tk={tk}&date={year}&datemonth={year}{('%02d'%month) if month else ''}&platform=ASCII&timeframe=M1&fxpair={pair.upper()}"
        r2 = curl(["-L","--max-time","90","-H",f"Referer: {referer}",
                   "-H","Origin: https://www.histdata.com",
                   "-H","Content-Type: application/x-www-form-urlencoded",
                   "--data", data, "https://www.histdata.com/get.php","-o","-"], timeout=110)
        body = r2.stdout
        if len(body) > 10_000 and body[:2] == b"PK":
            with open(zpath,"wb") as f: f.write(body)
            return zpath, "ok"
        time.sleep(2+a*2)
    return None, "failed"

def parse_zip(zpath):
    dfs=[]
    with zipfile.ZipFile(zpath) as z:
        for nm in z.namelist():
            if not nm.lower().endswith(".csv"): continue
            with z.open(nm) as fh:
                df = pd.read_csv(fh, sep=";", header=None,
                                 names=["ts","open","high","low","close","v"], dtype={"ts":str})
            dfs.append(df)
    if not dfs: return None
    df = pd.concat(dfs, ignore_index=True)
    idx = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S", utc=True)
    df = df[["open","high","low","close"]].astype(float)
    df.index = idx
    df["vol"] = df["high"] - df["low"]        # activity proxy (ASCII volume is 0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    h1 = df.resample("1H").agg({"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()
    return h1

if __name__ == "__main__":
    only = [p.lower() for p in sys.argv[1:]] if len(sys.argv)>1 else PAIRS
    jobs=[]
    for p in only:
        for y in YEARS: jobs.append((p,y,None))
        for m in CUR_MONTHS: jobs.append((p,CUR_YEAR,m))
    print(f"{len(jobs)} downloads queued ({only[0]}…, 2016-{CUR_MONTHS[-1]}/{CUR_YEAR}) …", flush=True)
    t0=time.time(); done=0; fails=[]
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(download_one,p,y,m): (p,y,m) for p,y,m in jobs}
        for f in as_completed(futs):
            p,y,m = futs[f]
            try: r = f.result()
            except Exception as e:
                r = (None, f"{type(e).__name__} {str(e)[:60]}")
            done+=1
            zpath, status = r
            if zpath is None:
                fails.append((p,y,m,status))
                print(f"  [{done:>3}/{len(jobs)}] ✗ {p} {y} {m}: {status}  ({time.time()-t0:.0f}s)", flush=True)
            else:
                print(f"  [{done:>3}/{len(jobs)}] ✓ {p} {y} {m}: {status} {os.path.getsize(zpath)//1024}KB  ({time.time()-t0:.0f}s)", flush=True)
    if fails:
        print(f"\n{len(fails)} failed — retrying once more sequentially …", flush=True)
        for p,y,m in [(a,b,c) for a,b,c,_ in fails]:
            zpath, status = download_one(p,y,m,tries=5)
            print(f"  retry {p} {y} {m}: {'OK' if zpath else status}", flush=True)

    print("\nParsing + resampling to 1H …", flush=True)
    for p in only:
        parts=[]
        for y in YEARS:
            zp = os.path.join(ZIPDIR, f"{p.upper()}_{y}.zip")
            if os.path.exists(zp):
                h = parse_zip(zp)
                if h is not None: parts.append(h)
        for m in CUR_MONTHS:
            zp = os.path.join(ZIPDIR, f"{p.upper()}_{CUR_YEAR}{m:02d}.zip")
            if os.path.exists(zp):
                h = parse_zip(zp)
                if h is not None: parts.append(h)
        if not parts:
            print(f"  ✗ {p}: no data"); continue
        full = pd.concat(parts).sort_index()
        full = full[~full.index.duplicated(keep="last")]
        full = full[full["open"]>0]
        out = os.path.join(OUTDIR, f"{p.upper()}_1H.parquet")
        full.to_parquet(out)
        print(f"  ✓ {p.upper()}: {len(full):>7} 1H bars  {full.index.min():%Y-%m-%d} → {full.index.max():%Y-%m-%d}", flush=True)
    print(f"Done in {time.time()-t0:.0f}s")
