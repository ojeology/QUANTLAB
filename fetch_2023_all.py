"""
Fetch 2023 (and pre-cache history) for ALL symbols in the 73-sym cache, so 2024 can be
walk-forward-validated on the full universe. RESUME-SAFE: tracks done symbols in a
manifest and saves each file with fsync, so it can be re-run safely if killed.
Saves merged 2023+cache as quantlab_cache_2023/{SYM}_1H_full.parquet.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import pandas as pd
import demo_bot as bot

CACHE = "quantlab_cache"
SAVE = "quantlab_cache_2023"
MANIFEST = os.path.join(SAVE, "_done_2023.txt")
os.makedirs(SAVE, exist_ok=True)


def fetch_before(end_ms, n_bars, inst, bar="1H", page_limit=200):
    all_rows, after, pages = [], end_ms, 0
    while len(all_rows) < n_bars and pages < page_limit:
        params = {"instId": inst, "bar": bar, "limit": bot.PAGE_LIMIT, "after": str(after)}
        raw = bot._get(bot.OKX_CANDLES, params) or bot._get(bot.OKX_CANDLES_CUR, params)
        if not raw: break
        all_rows.extend(raw); pages += 1; after = int(raw[-1][0])
        if len(all_rows) >= n_bars: break
        time.sleep(bot.PAGE_DELAY)
    if not all_rows: return None
    df = pd.DataFrame(all_rows, columns=bot.CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open","high","low","close","vol"]: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df[["datetime","open","high","low","close","vol"]].sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True).set_index("datetime")


# load manifest of already-done symbols
done = set()
if os.path.exists(MANIFEST):
    with open(MANIFEST) as f:
        done = set(l.strip() for l in f if l.strip())
    print(f"[resume] {len(done)} symbols already fetched", flush=True)

syms = sorted(f[:-len("_1H.parquet")] for f in os.listdir(CACHE) if f.endswith("_1H.parquet"))
print(f"[start] {len(syms)} symbols in cache; {len(done)} done", flush=True)

with open(MANIFEST, "a") as mf:
    for i, sym in enumerate(syms):
        if sym in done:
            continue
        p = os.path.join(CACHE, f"{sym}_1H.parquet")
        try:
            df = pd.read_parquet(p); df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
            for c in ["open","high","low","close","vol"]:
                if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
            df.dropna(subset=["open","high","low","close","vol"], inplace=True)
            inst=sym.replace("_","-"); end_ms=int(df.index[0].timestamp()*1000)-1
            f2023=fetch_before(end_ms,9200,inst)
            if f2023 is not None and len(f2023):
                df=pd.concat([f2023,df]); df=df[~df.index.duplicated(keep="last")].sort_index()
                out=os.path.join(SAVE, f"{sym}_1H_full.parquet")
                df.to_parquet(out); os.sync()
                mf.write(sym+"\n"); mf.flush()
                print(f"  saved {sym}: {len(df)} bars ({df.index[0].date()}) [{len(done)+1}/{len(syms)}]", flush=True)
                done.add(sym)
            else:
                print(f"  skip {sym} (no 2023 fetched)", flush=True)
        except Exception as e:
            print(f"  err {sym}: {e}", flush=True)
print("[done] fetch 2023 pass complete", flush=True)
print(open(MANIFEST).read().count("\n"), "symbols fetched in total", flush=True)
