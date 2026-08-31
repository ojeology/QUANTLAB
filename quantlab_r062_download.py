"""
R062 — Parallel data downloader
Downloads 24-month 1H history for all OKX USDT perps that pass the
R062 universe filters (≥18 months old, ≥$100k/24h volume).
Uses 8 parallel threads. Skips symbols already in cache.
"""
import os, sys, time, threading, concurrent.futures
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG

CACHE_DIR   = CONFIG["CACHE_FOLDER"]
os.makedirs(CACHE_DIR, exist_ok=True)

MIN_BARS    = 2_000
MONTHS      = 24
MIN_MONTHS  = 18
MIN_VOL_USD = 100_000
MAX_WORKERS = 8
PAGE_LIMIT  = 100
PAGE_DELAY  = 0.15

OKX_HIST_URL  = "https://www.okx.com/api/v5/market/history-candles"
OKX_INSTR_URL = "https://www.okx.com/api/v5/public/instruments"
OKX_TICK_URL  = "https://www.okx.com/api/v5/market/tickers"
CANDLE_COLS   = ["ts","open","high","low","close","vol",
                 "volCcy","volCcyQuote","confirm"]

_lock = threading.Lock()
_last = [0.0]
MIN_INTERVAL = 1.0 / 20   # global rate: ≤20 req/s

def _throttle():
    with _lock:
        wait = MIN_INTERVAL - (time.time() - _last[0])
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()

def _safe_get(url, params, timeout=20):
    _throttle()
    try:
        r = requests.get(url, params=params, timeout=timeout)
        d = r.json()
        if d.get("code") == "0":
            return d.get("data", [])
    except Exception as e:
        pass
    return []

def _parse_candles(raw):
    if not raw: return pd.DataFrame()
    df = pd.DataFrame(raw, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for col in ["open","high","low","close","vol"]:
        df[col] = pd.to_numeric(df[col])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return (df[["datetime","open","high","low","close","vol"]]
            .sort_values("datetime").reset_index(drop=True))

def download_symbol(sym):
    path = os.path.join(CACHE_DIR, sym.replace("-","_") + "_1H.parquet")
    # Skip if already cached with enough bars
    if os.path.exists(path):
        try:
            df = pd.read_parquet(path)
            if len(df) >= MIN_BARS:
                return sym, len(df), "CACHED"
        except Exception:
            pass
    # Download
    now_ms    = int(time.time() * 1000)
    cutoff_ms = now_ms - int(MONTHS * 30.44 * 24 * 3600 * 1000)
    all_rows  = []; after_ms = None; pages = 0
    while True:
        params = {"instId": sym, "bar": "1H", "limit": PAGE_LIMIT}
        if after_ms: params["after"] = str(after_ms)
        raw = _safe_get(OKX_HIST_URL, params)
        if not raw:
            # Try current candles endpoint once
            if pages == 0:
                raw = _safe_get("https://www.okx.com/api/v5/market/candles", params)
            if not raw: break
        all_rows.extend(raw); pages += 1
        oldest = int(raw[-1][0]); after_ms = oldest
        if oldest <= cutoff_ms: break
        time.sleep(PAGE_DELAY)
    if not all_rows:
        return sym, 0, "NO_DATA"
    df = _parse_candles(all_rows)
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df = (df[df["datetime"] >= cutoff_dt]
          .drop_duplicates("datetime")
          .sort_values("datetime")
          .reset_index(drop=True))
    if len(df) < MIN_BARS:
        return sym, len(df), "TOO_SHORT"
    df.to_parquet(path, index=False)
    return sym, len(df), "OK"

def main():
    print("=" * 72)
    print("  R062 — OKX Universe Downloader")
    print("=" * 72)
    print()

    # Step 1: discover universe
    print("  Fetching instruments and tickers ...")
    instr_data = _safe_get(OKX_INSTR_URL, {"instType": "SWAP"})
    now_ts = int(time.time() * 1000)
    min_age_ms = MIN_MONTHS * 30.44 * 24 * 3600 * 1000

    usdt_perps = [x for x in instr_data
                  if x.get("settleCcy","") == "USDT"
                  and x.get("state","") == "live"
                  and str(x.get("instId","")).endswith("-USDT-SWAP")]
    print(f"  Found {len(usdt_perps)} live USDT perpetuals")

    time.sleep(0.4)
    ticker_data = _safe_get(OKX_TICK_URL, {"instType": "SWAP"})
    ticker_map  = {x["instId"]: x for x in ticker_data
                   if x.get("instId","").endswith("-USDT-SWAP")}
    print(f"  Fetched {len(ticker_map)} tickers")

    # Filter
    targets = []
    for inst in usdt_perps:
        iid = inst.get("instId","")
        listing_ms = int(inst.get("listTime","0") or "0")
        age_ms = now_ts - listing_ms
        if age_ms < min_age_ms: continue
        tick = ticker_map.get(iid, {})
        vol_usd = float(tick.get("volCcy24h","0") or "0")
        if vol_usd < MIN_VOL_USD: continue
        targets.append(iid)

    print(f"  Qualified: {len(targets)} symbols (≥{MIN_MONTHS}m old, ≥${MIN_VOL_USD/1e3:.0f}k vol)")

    # Check what needs downloading
    need_dl = []
    already_cached = []
    for sym in targets:
        path = os.path.join(CACHE_DIR, sym.replace("-","_") + "_1H.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                if len(df) >= MIN_BARS:
                    already_cached.append(sym)
                    continue
            except Exception:
                pass
        need_dl.append(sym)

    print(f"  Already cached: {len(already_cached)}")
    print(f"  To download:    {len(need_dl)}")
    print()

    if not need_dl:
        print("  All symbols already cached!")
        return targets, already_cached, []

    # Step 2: parallel download
    t0 = time.time()
    ok_syms  = []
    fail_syms = []
    done = [0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_symbol, sym): sym for sym in need_dl}
        for fut in concurrent.futures.as_completed(futures):
            sym = futures[fut]
            try:
                sym_, n_bars, status = fut.result()
                done[0] += 1
                if status == "OK":
                    ok_syms.append(sym_)
                    first = last = ""
                    try:
                        df_t = pd.read_parquet(os.path.join(CACHE_DIR, sym_.replace("-","_")+"_1H.parquet"))
                        first = str(df_t["datetime"].iloc[0].date())
                        last  = str(df_t["datetime"].iloc[-1].date())
                    except: pass
                    print(f"  [{done[0]:3d}/{len(need_dl)}] ✓ {sym_:<28} "
                          f"{n_bars:6,} bars  {first} → {last}")
                elif status == "CACHED":
                    ok_syms.append(sym_)
                    print(f"  [{done[0]:3d}/{len(need_dl)}] ↩ {sym_:<28} (already cached)")
                else:
                    fail_syms.append(sym_)
                    print(f"  [{done[0]:3d}/{len(need_dl)}] ✗ {sym_:<28} {status} ({n_bars} bars)")
            except Exception as e:
                done[0] += 1
                fail_syms.append(sym)
                print(f"  [{done[0]:3d}/{len(need_dl)}] ✗ {sym:<28} ERROR: {e}")

    elapsed = time.time() - t0
    print()
    print(f"  Download complete in {elapsed:.0f}s")
    print(f"  Downloaded OK : {len(ok_syms)}")
    print(f"  Failed        : {len(fail_syms)}")
    if fail_syms:
        print(f"  Failed list   : {', '.join(fail_syms[:10])}")
    print()

    # Final cache count
    final_ok = []
    for sym in targets:
        path = os.path.join(CACHE_DIR, sym.replace("-","_") + "_1H.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                if len(df) >= MIN_BARS: final_ok.append(sym)
            except Exception: pass

    print(f"  Total symbols ready for analysis: {len(final_ok)}")
    return targets, final_ok, fail_syms

if __name__ == "__main__":
    main()
