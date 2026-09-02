"""
Fetch 15m history (3 years, ~105k bars) for a MAJOR-COIN subset, resume-safe.
15m has ~4x the bars of 1H, so this is slower; the manifest lets it resume if killed.
Saves quantlab_cache_15m/{SYM}_15m.parquet (merged full history). This is step 1 of the
15m hunt — once data exists, we test the condition-aware trend/MR on 15m.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import pandas as pd
import demo_bot as bot

SAVE = "quantlab_cache_15m"
MANIFEST = os.path.join(SAVE, "_done_15m.txt")
os.makedirs(SAVE, exist_ok=True)

MAJORS = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP"]
N_BARS = 105000  # ~3 years of 15m


def fetch_before(end_ms, n_bars, inst, bar="15m", page_limit=1200):
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


print(f"[start] {len(MAJORS)} majors (extending to 3-year 15m)", flush=True)
with open(MANIFEST, "a") as mf:
    for i, sym in enumerate(MAJORS):
        p = os.path.join(SAVE, f"{sym}_15m.parquet")
        if os.path.exists(p):
            try:
                if len(pd.read_parquet(p)) >= 100000:
                    print(f"  skip {sym} (already 3yr)", flush=True); continue
            except Exception: pass
        inst = sym.replace("_","-")
        end_ms = int(pd.Timestamp.now(tz="UTC").timestamp()*1000) - 1
        try:
            df = fetch_before(end_ms, N_BARS, inst, "15m")
            if df is not None and len(df) > 1000:
                out = os.path.join(SAVE, f"{sym}_15m.parquet")
                df.to_parquet(out); os.sync()
                mf.write(sym+"\n"); mf.flush()
                print(f"  saved {sym}: {len(df)} 15m bars ({df.index[0].date()} -> {df.index[-1].date()}) [{i+1}/{len(MAJORS)}]", flush=True)
                done.add(sym)
            else:
                print(f"  skip {sym} (empty)", flush=True)
        except Exception as e:
            print(f"  err {sym}: {e}", flush=True)
print("[done] 15m fetch pass complete", flush=True)
print(open(MANIFEST).read().count("\n"), "symbols fetched total", flush=True)
