"""
RESUME-SAFE 2023 fetcher: fetch 2023 for the 30-symbol subset, merge with the
2024-2026 cache, and SAVE each symbol's full 1H series to quantlab_cache_2023/.
Skips symbols already saved, so re-running after a kill resumes. Goal: persist
the 2023-2026 data so the adaptive sweep can run WITHOUT any live fetch (no kill).
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import pandas as pd
warnings.filterwarnings("ignore")
import sys
import demo_bot as bot

SAVE_DIR = "quantlab_cache_2023"
os.makedirs(SAVE_DIR, exist_ok=True)
CACHE = "quantlab_cache"
SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP",
          "FIL_USDT_SWAP","INJ_USDT_SWAP","AXS_USDT_SWAP","SAND_USDT_SWAP","FET_USDT_SWAP",
          "GRT_USDT_SWAP","HBAR_USDT_SWAP","IMX_USDT_SWAP","COMP_USDT_SWAP","AAVE_USDT_SWAP"]
if len(sys.argv) >= 3:
    SUBSET = SUBSET[int(sys.argv[1]):int(sys.argv[2])]


def fetch_before(end_ts_ms, n_bars, inst, bar="1H", page_limit=200):
    all_rows, after, pages = [], end_ts_ms, 0
    while len(all_rows) < n_bars and pages < page_limit:
        params = {"instId": inst, "bar": bar, "limit": bot.PAGE_LIMIT, "after": str(after)}
        raw = bot._get(bot.OKX_CANDLES, params) or bot._get(bot.OKX_CANDLES_CUR, params)
        if not raw:
            break
        all_rows.extend(raw); pages += 1; oldest = int(raw[-1][0]); after = oldest
        if len(all_rows) >= n_bars:
            break
        time.sleep(bot.PAGE_DELAY)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=bot.CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return (df[["datetime","open","high","low","close","vol"]].sort_values("datetime")
            .drop_duplicates("datetime").reset_index(drop=True).set_index("datetime"))


print(f"[fetch] saving 2023-2026 merged series to {SAVE_DIR}/ (resume-safe)", flush=True)
done = 0
for sym in SUBSET:
    out = os.path.join(SAVE_DIR, f"{sym}_1H_full.parquet")
    if os.path.exists(out):
        done += 1; continue
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    if not os.path.exists(p):
        continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for c in ["open","high","low","close","vol"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        inst = sym.replace("_","-")
        end_ms = int(df.index[0].timestamp()*1000) - 1
        f2023 = fetch_before(end_ms, 9200, inst)
        if f2023 is not None and len(f2023):
            df = pd.concat([f2023, df]); df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(out)
        os.sync()
        done += 1
        print(f"  saved {sym}: {len(df)} bars ({df.index[0].date()} -> {df.index[-1].date()})  [{done}/{len(SUBSET)}]", flush=True)
    except Exception as e:
        print(f"  err {sym}: {e}", flush=True)
print(f"[fetch] done. {done}/{len(SUBSET)} symbols saved to {SAVE_DIR}/", flush=True)
