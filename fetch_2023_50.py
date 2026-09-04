"""Fetch 2022-12 -> cache-start 1H history for the 50 manifest symbols (OKX
history-candles), merge with local cache, persist _1H_full.parquet.
Single-worker + throttle + backoff (OKX rate-limits parallel page chains)."""
import os, time
import pandas as pd
import requests

CACHE = "/home/user/quantlab/quantlab_cache"
SAVE = "/home/user/bv/quantlab_cache_2023"
MANIFEST = os.path.join(SAVE, "_done_2023.txt")
PAGE = 300
HIST = "https://www.okx.com/api/v5/market/history-candles"
COL = ["ts", "open", "high", "low", "close", "vol", "volCcy", "volCcyQuote", "confirm"]
TARGET_DATE = pd.Timestamp("2022-12-20", tz="UTC")
MIN_ROWS = 1500

os.makedirs(SAVE, exist_ok=True)
syms = [l.strip() for l in open(MANIFEST) if l.strip()]


def fetch_2023(end_ms, sym):
    """Throttled page chain for one symbol."""
    inst = sym.replace("_", "-")
    rows, after, pages, fails = [], end_ms, 0, 0
    while pages < 200 and fails < 6:
        try:
            r = requests.get(HIST, params={"instId": inst, "bar": "1H", "limit": PAGE,
                                           "after": str(after)}, timeout=20)
            j = r.json()
            d = j.get("data") or []
            if j.get("code") != "0" or not d:
                fails += 1; time.sleep(1.0 + fails); continue
        except Exception:
            fails += 1; time.sleep(1.0 + fails); continue
        rows.extend(d); pages += 1; fails = 0
        after = int(d[-1][0])
        if int(d[-1][0]) <= int(TARGET_DATE.timestamp() * 1000):
            break
        time.sleep(0.22)
    return rows


def one(sym):
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    out = os.path.join(SAVE, f"{sym}_1H_full.parquet")
    if os.path.exists(out):
        return sym, "already"
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "vol"])
    end_ms = int(df.index[0].timestamp() * 1000) - 1
    rows = fetch_2023(end_ms, sym)
    if len(rows) < MIN_ROWS:
        return sym, f"short:{len(rows)}"
    f23 = pd.DataFrame(rows, columns=COL)
    f23["ts"] = pd.to_numeric(f23["ts"])
    for c in COL[1:6]:
        f23[c] = pd.to_numeric(f23[c], errors="coerce")
    f23["datetime"] = pd.to_datetime(f23["ts"], unit="ms", utc=True)
    f23 = f23[["datetime", "open", "high", "low", "close", "vol"]] \
        .sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
    full = pd.concat([f23, df]); full = full[~full.index.duplicated(keep="last")].sort_index()
    full.to_parquet(out)
    return sym, f"ok {len(f23)}->{len(full)}"


t0 = time.time()
for i, sym in enumerate(syms):
    try:
        sym, status = one(sym)
    except Exception as e:
        status = f"ERR {str(e)[:80]}"
    print(f"[{i+1}/{len(syms)}] {sym}: {status}", flush=True)
print(f"[done in {time.time()-t0:.0f}s]")
