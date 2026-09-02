"""
Fetch 3-YEAR 15m from YAHOO FINANCE (crypto as BTC-USD) for the core symbols, async.
Yahoo chart API is permissive and a different domain than the blocked exchanges.
Chunk by year (4 requests/symbol). Maps BTC_USDT_SWAP -> BTC-USD. Resume-safe.
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
YAH = {s: s.split("_")[0] + "-USD" for s in CORE}
HDR = {"User-Agent":"Mozilla/5.0"}


async def fetch_sym(session, sem, sym, ysym):
    out = os.path.join(SAVE, f"{sym}_15m.parquet")
    if os.path.exists(out):
        try:
            if len(pd.read_parquet(out)) >= 100000: return None
        except Exception: pass
    async with sem:
        rows = []
        for yr in [2023,2024,2025,2026]:
            p1 = int(pd.Timestamp(f"{yr}-01-01", tz="UTC").timestamp())
            p2 = int(pd.Timestamp(f"{yr+1}-01-01", tz="UTC").timestamp()) if yr < 2026 else int(pd.Timestamp.now(tz="UTC").timestamp())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}"
            for attempt in range(3):
                try:
                    async with session.get(url, params={"period1":p1,"period2":p2,"interval":"15m","events":"div,split"}, headers=HDR, timeout=30) as r:
                        j = await r.json()
                    if j.get("chart",{}).get("error"): break
                    res = (j.get("chart") or {}).get("result")
                    if not res: break
                    res = res[0]; ts = res.get("timestamp"); q = res["indicators"]["quote"][0]
                    for i,t in enumerate(ts):
                        o,h,l,c,v = q["open"][i],q["high"][i],q["low"][i],q["close"][i],q["volume"][i]
                        if c is None: continue
                        rows.append([int(t)*1000, o,h,l,c,v])
                    break
                except Exception:
                    await asyncio.sleep(2)
            await asyncio.sleep(0.05)
        if len(rows) < 50000:
            print(f"  {sym}: only {len(rows)} rows", flush=True); return None
        df = pd.DataFrame(rows, columns=["ts","open","high","low","close","vol"])
        df["ts"] = pd.to_numeric(df["ts"])
        for c in ["open","high","low","close","vol"]: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df[["datetime","open","high","low","close","vol"]].dropna(subset=["close"]).sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
        df.to_parquet(out)
        if hasattr(os,"sync"): os.sync()
        print(f"  saved {sym}: {len(df)} 15m bars {df.index[0].date()} -> {df.index[-1].date()}", flush=True)
        return sym


async def main():
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as session:
        res = await asyncio.gather(*[fetch_sym(session, sem, s, YAH[s]) for s in CORE])
    got = [r for r in res if r]
    print(f"[done] Yahoo 15m fetched {len(got)} core symbols", flush=True)


if __name__ == "__main__":
    print(f"[start] {len(CORE)} core symbols from Yahoo (3yr 15m, async)", flush=True)
    t0 = time.time(); asyncio.run(main()); print(f"[elapsed] {time.time()-t0:.0f}s", flush=True)
