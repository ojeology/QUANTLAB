"""
Fetch 1H forex data for 8 major pairs via yfinance (free, no key).
Normalizes to UTC, saves as quantlab_cache/forex/SYMBOL_1H.parquet.
Schema matches the crypto pipeline: datetime index (UTC), open/high/low/close.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf

CACHE = "/home/user/quantlab/quantlab_cache/forex"
os.makedirs(CACHE, exist_ok=True)

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
}

print("Fetching 8 major forex pairs (1H, ~2y each) …", flush=True)
t0 = time.time()
for sym, ticker in PAIRS.items():
    out = os.path.join(CACHE, f"{sym}_1H.parquet")
    try:
        df = yf.download(ticker, interval="1h", period="2y", progress=False,
                         auto_adjust=False)
        if df is None or df.empty:
            print(f"  ✗ {sym}: empty"); continue
        # flatten multiindex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].copy()
        df.columns = ["open", "high", "low", "close"]
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("UTC")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.dropna()
        df.to_parquet(out)
        print(f"  ✓ {sym}: {len(df)} bars  {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}")
    except Exception as e:
        print(f"  ✗ {sym}: {type(e).__name__} {str(e)[:100]}")
    time.sleep(0.3)
print(f"Done in {time.time()-t0:.0f}s → {CACHE}")
