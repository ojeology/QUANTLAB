"""
R044 fast symbol probe — downloads up to 50 pages per symbol to find
which OKX USDT perpetuals have ≥ 4000 bars of 1-hour history.
Results are cached so quantlab_r044.py picks them up directly.
"""
import os, sys, time, threading, concurrent.futures, pandas as pd, requests

sys.path.insert(0, "/home/runner/workspace")
CACHE_DIR   = "quantlab_cache"
MIN_BARS    = 4_000
MAX_PAGES   = 50          # 50 pages × 100 bars = 5 000 bars — enough to qualify
MONTHS_1H   = 24
OKX_HIST_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_CAND_URL = "https://www.okx.com/api/v5/market/candles"
CANDLE_COLS  = ["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"]

_rate_lock  = threading.Lock()
_last_req_t = [0.0]
RATE_IVL    = 1.0 / 15        # 15 req/s globally

def _fetch(sym, bar, after_ms=None, use_history=True):
    url    = OKX_HIST_URL if use_history else OKX_CAND_URL
    params = {"instId": sym, "bar": bar, "limit": "100"}
    if after_ms:
        params["after"] = str(after_ms)
    with _rate_lock:
        wait = RATE_IVL - (time.time() - _last_req_t[0])
        if wait > 0:
            time.sleep(wait)
        _last_req_t[0] = time.time()
    try:
        r = requests.get(url, params=params, timeout=15)
        d = r.json()
        if d.get("code") == "0":
            return d.get("data", [])
    except Exception:
        pass
    return []

def probe_symbol(sym):
    """Download up to MAX_PAGES pages. Return (sym, df) if ≥ MIN_BARS, else (sym, None)."""
    cache_path = os.path.join(CACHE_DIR, sym.replace("-","_") + "_1H.parquet")

    # Serve from cache if already good
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            if len(df) >= MIN_BARS:
                return sym, df
        except Exception:
            pass

    now_ms     = int(time.time() * 1000)
    target_ms  = int(MONTHS_1H * 30.44 * 24 * 3600 * 1000)
    cutoff_ms  = now_ms - target_ms
    all_rows   = []
    after_ms_c = None

    for page in range(1, MAX_PAGES + 1):
        raw = _fetch(sym, "1H", after_ms=after_ms_c, use_history=True)
        if not raw and page == 1:
            raw = _fetch(sym, "1H", use_history=False)
        if not raw:
            break
        all_rows.extend(raw)
        oldest_ts    = int(raw[-1][0])
        after_ms_c   = oldest_ts
        if oldest_ts <= cutoff_ms:
            break

    if len(all_rows) < MIN_BARS:
        return sym, None

    df = pd.DataFrame(all_rows, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for col in ["open","high","low","close","vol"]:
        df[col] = pd.to_numeric(df[col])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df = (df[df["datetime"] >= cutoff_dt]
          [["datetime","open","high","low","close","vol"]]
          .drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True))

    if len(df) < MIN_BARS:
        return sym, None

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return sym, df

# ── Candidates ──────────────────────────────────────────────────────────────
R043 = {
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","BNB-USDT-SWAP","XRP-USDT-SWAP",
    "ADA-USDT-SWAP","DOGE-USDT-SWAP","AVAX-USDT-SWAP","LINK-USDT-SWAP","DOT-USDT-SWAP",
    "LTC-USDT-SWAP","BCH-USDT-SWAP","FIL-USDT-SWAP","UNI-USDT-SWAP","NEAR-USDT-SWAP",
    "APT-USDT-SWAP","OP-USDT-SWAP","ARB-USDT-SWAP","SUI-USDT-SWAP","PEPE-USDT-SWAP",
    "WIF-USDT-SWAP","ENA-USDT-SWAP","ATOM-USDT-SWAP",
}

# All OKX USDT swaps listed ≥ 24 months ago, not in R043
CANDIDATES = [
    "ETC-USDT-SWAP","TRX-USDT-SWAP","NEO-USDT-SWAP","ALGO-USDT-SWAP","COMP-USDT-SWAP",
    "IOST-USDT-SWAP","IOTA-USDT-SWAP","ONT-USDT-SWAP","QTUM-USDT-SWAP","THETA-USDT-SWAP",
    "XLM-USDT-SWAP","XTZ-USDT-SWAP","SNX-USDT-SWAP","ZRX-USDT-SWAP","BAT-USDT-SWAP",
    "SUSHI-USDT-SWAP","YFI-USDT-SWAP","CRV-USDT-SWAP","UMA-USDT-SWAP","BAND-USDT-SWAP",
    "KSM-USDT-SWAP","TRB-USDT-SWAP","RSR-USDT-SWAP","ZIL-USDT-SWAP","AAVE-USDT-SWAP",
    "GRT-USDT-SWAP","EGLD-USDT-SWAP","1INCH-USDT-SWAP","MASK-USDT-SWAP","CFX-USDT-SWAP",
    "CHZ-USDT-SWAP","MANA-USDT-SWAP","SAND-USDT-SWAP","CRO-USDT-SWAP","LPT-USDT-SWAP",
    "RVN-USDT-SWAP","SHIB-USDT-SWAP","ICP-USDT-SWAP","MINA-USDT-SWAP","AXS-USDT-SWAP",
    "YGG-USDT-SWAP","AGLD-USDT-SWAP","DYDX-USDT-SWAP","CELO-USDT-SWAP","GALA-USDT-SWAP",
    "ENS-USDT-SWAP","IMX-USDT-SWAP","PEOPLE-USDT-SWAP","BICO-USDT-SWAP","API3-USDT-SWAP",
    "APE-USDT-SWAP","GMT-USDT-SWAP","LDO-USDT-SWAP","GMX-USDT-SWAP","MAGIC-USDT-SWAP",
    "CORE-USDT-SWAP","AR-USDT-SWAP","WOO-USDT-SWAP","BLUR-USDT-SWAP","FLOKI-USDT-SWAP",
    "STX-USDT-SWAP","ORDI-USDT-SWAP","WLD-USDT-SWAP","HBAR-USDT-SWAP","BIGTIME-USDT-SWAP",
    "GAS-USDT-SWAP","TIA-USDT-SWAP","MEME-USDT-SWAP","FLOW-USDT-SWAP","PYTH-USDT-SWAP",
    "INJ-USDT-SWAP","BONK-USDT-SWAP","JTO-USDT-SWAP","JUP-USDT-SWAP","STRK-USDT-SWAP",
    "ETHFI-USDT-SWAP","W-USDT-SWAP","GLM-USDT-SWAP","ZK-USDT-SWAP","ZRO-USDT-SWAP",
]

if __name__ == "__main__":
    t0      = time.time()
    good    = {}
    tested  = 0
    stop_at = 25   # stop as soon as we have 25 qualifying symbols

    print(f"Probing {len(CANDIDATES)} candidates for ≥ {MIN_BARS} bars (max {MAX_PAGES} pages each)…")
    print(f"Rate: 15 req/s  |  Workers: 8  |  Stop when {stop_at} qualify\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(probe_symbol, s): s for s in CANDIDATES}
        for fut in concurrent.futures.as_completed(futures):
            sym, df = fut.result()
            tested  += 1
            if df is not None:
                good[sym] = df
                print(f"  ✓ {sym:<28} {len(df):>6,} bars  "
                      f"({df['datetime'].iloc[0].date()} → "
                      f"{df['datetime'].iloc[-1].date()})")
                if len(good) >= stop_at:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
            else:
                print(f"  ✗ {sym:<28} < {MIN_BARS} bars")

    elapsed = time.time() - t0
    print(f"\n{'─'*60}")
    print(f"  Tested:     {tested}/{len(CANDIDATES)}")
    print(f"  Qualified:  {len(good)}")
    print(f"  Elapsed:    {elapsed:.1f}s")
    print(f"\nQualified symbols:")
    for s in sorted(good):
        print(f"  \"{s}\",")
