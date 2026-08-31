"""
R044 — KuCoin Futures bulk downloader
Downloads 24 months of 1H data for ~22 unseen symbols from KuCoin Futures.
Saves in the same parquet format as OKX cache.
"""
import os, sys, time, math, threading, concurrent.futures
import pandas as pd
import requests

CACHE_DIR = "quantlab_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

GRANULARITY = 60          # 1H in minutes
LIMIT       = 200         # max candles per KuCoin request
MIN_BARS    = 4_000       # minimum qualifying bars
MONTHS      = 24          # history target

BASE_URL = "https://api-futures.kucoin.com/api/v1/kline/query"

# KuCoin symbol  →  OKX-style cache key (base-USDT-SWAP → BASE_USDT_SWAP)
# Only symbols NOT in R043 and NOT already cached from OKX
TARGETS = {
    "ETCUSDTM":   "ETC_USDT_SWAP",
    "AAVEUSDTM":  "AAVE_USDT_SWAP",
    "INJUSDTM":   "INJ_USDT_SWAP",
    "HBARUSDTM":  "HBAR_USDT_SWAP",
    "ICPUSDTM":   "ICP_USDT_SWAP",
    "SANDUSDTM":  "SAND_USDT_SWAP",
    "GALAUSDTM":  "GALA_USDT_SWAP",
    "SHIBUSDTM":  "SHIB_USDT_SWAP",
    "IMXUSDTM":   "IMX_USDT_SWAP",
    "AXSUSDTM":   "AXS_USDT_SWAP",
    "EGLDUSDTM":  "EGLD_USDT_SWAP",
    "CRVUSDTM":   "CRV_USDT_SWAP",
    "GRTUSDTM":   "GRT_USDT_SWAP",
    "SUSHIUSDTM": "SUSHI_USDT_SWAP",
    "XLMUSDTM":   "XLM_USDT_SWAP",
    "SNXUSDTM":   "SNX_USDT_SWAP",
    "ALGOUSDTM":  "ALGO_USDT_SWAP",
    "COMPUSDTM":  "COMP_USDT_SWAP",
    "1INCHUSDTM": "1INCH_USDT_SWAP",
    "FETUSDTM":   "FET_USDT_SWAP",
    "STXUSDTM":   "STX_USDT_SWAP",
    "CHZUSDTM":   "CHZ_USDT_SWAP",
}

# Global rate limiter: max 20 req/s
_lock = threading.Lock()
_last = [0.0]
MIN_INTERVAL = 1.0 / 20  # 50 ms between requests


def _throttle():
    with _lock:
        wait = MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def fetch_page(symbol: str, from_s: int, to_s: int) -> list:
    _throttle()
    try:
        r = requests.get(BASE_URL,
                         params={"symbol": symbol, "granularity": GRANULARITY,
                                 "from": from_s, "to": to_s},
                         timeout=15)
        d = r.json()
        if d.get("code") == "200000":
            return d.get("data") or []
    except Exception as e:
        print(f"    ⚠ {symbol} fetch error: {e}")
    return []


def download_symbol(kucoin_sym: str, cache_key: str) -> tuple:
    """Download full history and return (cache_key, df or None)."""
    path = os.path.join(CACHE_DIR, cache_key + "_1H.parquet")

    # Skip if already cached with enough bars
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            if len(df) >= MIN_BARS:
                return cache_key, df
        except Exception:
            pass

    now_ms       = int(time.time() * 1000)
    window_ms    = int(MONTHS * 30.44 * 24 * 3600 * 1000)
    start_ms     = now_ms - window_ms
    page_ms      = LIMIT * GRANULARITY * 60 * 1000   # milliseconds per page (200h)

    all_rows = []
    cur_start = start_ms

    while cur_start < now_ms:
        cur_end = min(cur_start + page_ms, now_ms)
        rows = fetch_page(kucoin_sym, cur_start, cur_end)
        if rows:
            all_rows.extend(rows)
        cur_start = cur_end
        if cur_start >= now_ms:
            break

    if not all_rows:
        return cache_key, None

    # KuCoin format: [timestamp_ms, open, high, low, close, volume, ...]
    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","vol","turnover"])
    for col in ["ts","open","high","low","close","vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = (df[["datetime","open","high","low","close","vol"]]
          .drop_duplicates("datetime")
          .sort_values("datetime")
          .reset_index(drop=True))

    if len(df) < MIN_BARS:
        return cache_key, None

    df.to_parquet(path, index=False)
    return cache_key, df


def main():
    print("=" * 70)
    print("  R044 — KuCoin Futures downloader")
    print(f"  Target: {len(TARGETS)} symbols · {MONTHS}M 1H history")
    print("=" * 70)
    print()

    t0      = time.time()
    results = {}
    failed  = []

    # 5 parallel workers — each downloads sequentially through pages
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(download_symbol, ks, ck): (ks, ck)
                for ks, ck in TARGETS.items()}
        for fut in concurrent.futures.as_completed(futs):
            ks, ck = futs[fut]
            try:
                cache_key, df = fut.result()
                if df is not None:
                    results[cache_key] = df
                    print(f"  ✓ {ks:20s} → {cache_key}  {len(df):6,} bars  "
                          f"{df['datetime'].iloc[0].date()} → "
                          f"{df['datetime'].iloc[-1].date()}")
                else:
                    failed.append(ks)
                    print(f"  ✗ {ks:20s} < {MIN_BARS} bars")
            except Exception as e:
                failed.append(ks)
                print(f"  ✗ {ks:20s} ERROR: {e}")

    elapsed = time.time() - t0
    print()
    print(f"  Downloaded: {len(results)}/{len(TARGETS)}  ({elapsed:.0f}s)")
    if failed:
        print(f"  Failed:     {', '.join(failed)}")
    print()
    print("  Qualified cache keys:")
    for k in sorted(results):
        print(f"    \"{k.replace('_USDT_SWAP','')}-USDT-SWAP\",")


if __name__ == "__main__":
    main()
