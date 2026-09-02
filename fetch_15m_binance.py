"""
Fetch 3-YEAR 15m from BINANCE (not OKX) for the core symbols, via aiohttp async.
Binance public klines allow 1000 bars/request with deep history and high rate limits --
no throttle like OKX's 15m endpoint. Maps BTC_USDT_SWAP -> BTCUSDT. Resume-safe.
"""
import os, sys, asyncio, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import aiohttp

SAVE = "quantlab_cache_15m"
os.makedirs(SAVE, exist_ok=True)
CORE = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
        "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
        "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
        "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]
BIN = {s: s.split("_")[0] + "USDT" for s in CORE}   # BTC_USDT_SWAP -> BTCUSDT
START = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp()*1000)
END = int(pd.Timestamp.now(tz="UTC").timestamp()*1000)
INT = 900000  # 15m in ms


async def fetch_sym(session, sem, sym, bin_sym):
    out = os.path.join(SAVE, f"{sym}_15m.parquet")
    if os.path.exists(out):
        try:
            if len(pd.read_parquet(out)) >= 100000: return None
        except Exception: pass
    async with sem:
        bars = []; start = START
        for attempt in range(5):
            try:
                while start < END:
                    params = {"symbol": bin_sym, "interval": "15m", "startTime": start, "limit": 1000}
                    async with session.get("https://api.binance.com/api/v3/klines", params=params, timeout=30) as r:
                        if r.status == 429:
                            await asyncio.sleep(5); continue
                        data = await r.json()
                    if not data: break
                    bars.extend(data)
                    start = data[-1][0] + INT
                    if len(data) < 1000: break
                    await asyncio.sleep(0.02)
                break
            except Exception as e:
                await asyncio.sleep(2)
        if len(bars) < 50000:
            print(f"  {sym}: only {len(bars)} bars", flush=True); return None
        df = pd.DataFrame(bars, columns=["ts","open","high","low","close","vol","closeTime","qv","trades","tbv","tqv","ignore"])
        df["ts"] = pd.to_numeric(df["ts"])
        for c in ["open","high","low","close","vol"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df[["datetime","open","high","low","close","vol"]].sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True).set_index("datetime")
        df.to_parquet(out); 
        if hasattr(os, "sync"): os.sync()
        print(f"  saved {sym}: {len(df)} 15m bars {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
        return sym


async def main():
    sem = asyncio.Semaphore(15)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_sym(session, sem, s, BIN[s]) for s in CORE]
        res = await asyncio.gather(*tasks)
    got = [r for r in res if r]
    print(f"[done] Binance 15m fetched {len(got)} core symbols", flush=True)


if __name__ == "__main__":
    print(f"[start] {len(CORE)} core symbols from Binance (3yr 15m, async)", flush=True)
    t0 = time.time()
    asyncio.run(main())
    print(f"[elapsed] {time.time()-t0:.0f}s", flush=True)
