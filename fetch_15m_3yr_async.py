"""
Fetch FULL 3-year 15m for a 20-symbol CORE subset (parallel, chunked, retry-with-backoff).
Full 73-sym 3yr tripped OKX rate limits under parallel load; a 20-sym core is enough
to validate the 3-year walk-forward and is feasible in one run. Resume-safe.
"""
import os, sys, time, warnings, concurrent.futures
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import pandas as pd
import demo_bot as bot

SAVE = "quantlab_cache_15m"; CHUNK = 20000; CHUNKS = 6
SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]
os.makedirs(SAVE, exist_ok=True)


def fetch_before(end_ms, n_bars, inst, bar="15m", page_limit=200, retries=3):
    for attempt in range(retries):
        try:
            all_rows, after, pages = [], end_ms, 0
            while len(all_rows) < n_bars and pages < page_limit:
                params = {"instId": inst, "bar": bar, "limit": bot.PAGE_LIMIT, "after": str(after)}
                raw = bot._get(bot.OKX_CANDLES, params) or bot._get(bot.OKX_CANDLES_CUR, params)
                if not raw:
                    time.sleep(3); break
                all_rows.extend(raw); pages += 1; after = int(raw[-1][0])
                if len(all_rows) >= n_bars: break
                time.sleep(bot.PAGE_DELAY)
            if all_rows:
                df = pd.DataFrame(all_rows, columns=bot.CANDLE_COLS)
                df["ts"] = pd.to_numeric(df["ts"])
                for c in ["open","high","low","close","vol"]: df[c] = pd.to_numeric(df[c], errors="coerce")
                df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                return df[["datetime","open","high","low","close","vol"]].sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True).set_index("datetime")
        except Exception:
            pass
        time.sleep(3)
    return None


def fetch_one(sym):
    out = os.path.join(SAVE, f"{sym}_15m.parquet")
    if os.path.exists(out):
        try:
            if len(pd.read_parquet(out)) >= 100000: return None
        except Exception: pass
    try:
        inst = sym.replace("_","-")
        end_ms = int(pd.Timestamp.now(tz="UTC").timestamp()*1000) - 1
        chunks = []
        for c in range(CHUNKS):
            df = fetch_before(end_ms, CHUNK, inst, "15m", page_limit=200)
            if df is None or len(df) == 0: break
            chunks.append(df)
            end_ms = int(df.index[0].timestamp()*1000) - 1
            if len(chunks)*CHUNK >= 105000: break
        if not chunks: return None
        full = pd.concat(chunks); full = full[~full.index.duplicated(keep="last")].sort_index()
        if len(full) < 50000: return None
        full.to_parquet(out); os.sync(); return sym
    except Exception:
        return None


print(f"[fetch] {len(SUBSET)} core symbols; 3-year 15m, chunked parallel (8 workers, retry)", flush=True)
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    done = list(ex.map(fetch_one, SUBSET))
done = [s for s in done if s]
print(f"[fetch] fetched {len(done)} this pass in {time.time()-t0:.0f}s", flush=True)
have = sum(1 for s in SUBSET if os.path.exists(os.path.join(SAVE, f"{s}_15m.parquet")) and os.path.getsize(os.path.join(SAVE, f"{s}_15m.parquet"))>0)
print(f"[fetch] core 15m files present: {have}/{len(SUBSET)}", flush=True)
print("[done]")
