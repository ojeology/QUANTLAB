"""
Timestamp Fix — one-time repair for all parquet files in quantlab_cache.

Root cause: download scripts saved with index=False, so datetime is a column.
Research scripts do df.index = pd.to_datetime(df.index, utc=True) on a
sequential RangeIndex → all hours = 0.

Fix: for each parquet file that has a 'datetime' column and an integer index,
set datetime as the index and re-save. Files already correctly indexed are
left untouched.
"""

import os, time
import pandas as pd

CACHE_DIR = "quantlab_cache"
t0 = time.time()

files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet"))
print(f"Found {len(files)} parquet files in {CACHE_DIR}/\n")

fixed = 0; skipped = 0; errors = 0

for fn in files:
    path = os.path.join(CACHE_DIR, fn)
    try:
        df = pd.read_parquet(path)

        # Check if already indexed correctly
        if isinstance(df.index, pd.DatetimeTZAware if hasattr(pd, 'DatetimeTZAware') else type(None)):
            skipped += 1
            continue

        # Check if datetime column exists
        if "datetime" not in df.columns:
            # Try index directly
            try:
                test = pd.to_datetime(df.index, utc=True)
                if test.hour.nunique() > 1:
                    skipped += 1
                    continue
            except Exception:
                pass
            print(f"  SKIP (no datetime col): {fn}")
            skipped += 1
            continue

        # Check if already a DatetimeIndex
        if hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index.dtype):
            # Already datetime index — check if it has timezone
            if df.index.tz is not None:
                skipped += 1
                continue

        # Fix: set datetime column as index
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()

        # Verify the fix
        sample_hours = df.index[:20].hour.tolist()
        if len(set(sample_hours)) == 1 and sample_hours[0] == 0:
            print(f"  WARN: {fn} still all-zero hours after fix — skipping")
            skipped += 1
            continue

        df.to_parquet(path)
        fixed += 1

        first = str(df.index[0])[:16]
        last  = str(df.index[-1])[:16]
        sample_h = sorted(set(df.index[:100].hour.tolist()))
        print(f"  ✓ {fn:<40}  {len(df):>6,} bars  {first} → {last}  hours={sample_h[:6]}")

    except Exception as e:
        print(f"  ✗ {fn}: {e}")
        errors += 1

elapsed = time.time() - t0
print(f"\n{'═'*70}")
print(f"  Fixed:   {fixed}")
print(f"  Skipped: {skipped}")
print(f"  Errors:  {errors}")
print(f"  Time:    {elapsed:.1f}s")
print()

# Verification
print("  Verification — checking 3 symbols for real hour distribution:")
for fn in ["BTC_USDT_SWAP_1H.parquet", "ETH_USDT_SWAP_1H.parquet", "SOL_USDT_SWAP_1H.parquet"]:
    path = os.path.join(CACHE_DIR, fn)
    if not os.path.exists(path): continue
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    unique_hours = sorted(df.index.hour.unique().tolist())
    print(f"    {fn:<35}  unique hours: {unique_hours[:6]}...  "
          f"{'OK ✓' if len(unique_hours) > 1 else 'STILL BROKEN ✗'}")
