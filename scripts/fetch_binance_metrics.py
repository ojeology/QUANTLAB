#!/usr/bin/env python3
"""Fetch Binance Vision DAILY futures metrics (OI + long/short crowding + taker flow)
-> quantlab_cache/bnmetrics/{SYM}_metrics.csv  (daily-aggregated: last OI, mean ratios)."""
import io, pathlib, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

OUT = pathlib.Path("quantlab_cache/bnmetrics"); OUT.mkdir(parents=True, exist_ok=True)
SYMS = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LINK", "LTC", "AVAX",
        "DOT", "NEAR", "APT", "SUI", "ATOM", "FIL", "INJ", "ARB", "OP", "ETC",
        "TRX", "XLM", "CRV", "AAVE"]
DAYS = pd.date_range("2024-01-01", "2026-09-30", freq="D")

def get(url):
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            time.sleep(1.2 * (k + 1))
    return None

def day_metrics(sym, day):
    tag = day.strftime("%Y-%m-%d")
    b = get(f"https://data.binance.vision/data/futures/um/daily/metrics/{sym}USDT/{sym}USDT-metrics-{tag}.zip")
    if not b:
        return sym, day, None
    try:
        zf = zipfile.ZipFile(io.BytesIO(b)); name = zf.namelist()[0]
        df = pd.read_csv(zf.open(name))
        df.columns = [c.strip().lower() for c in df.columns]
        def col(*names):
            for n in names:
                if n in df.columns:
                    return pd.to_numeric(df[n], errors="coerce")
            return pd.Series(dtype=float)
        oi = col("sum_open_interest_value", "sum_open_interest")
        lstop = col("sum_toptrader_long_short_ratio")
        lsac = col("count_long_short_ratio")
        tk = col("sum_taker_long_short_vol_ratio")
        return sym, day, dict(oi_usd=float(oi.dropna().iloc[-1]) if oi.notna().any() else None,
                              ls_top=float(lstop.mean()) if lstop.notna().any() else None,
                              ls_acct=float(lsac.mean()) if lsac.notna().any() else None,
                              taker=float(tk.mean()) if tk.notna().any() else None)
    except Exception:
        return sym, day, None

acc = {s: [] for s in SYMS}
jobs = [(s, d) for s in SYMS for d in DAYS]
done = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    for f in as_completed([ex.submit(day_metrics, s, d) for s, d in jobs]):
        s, day, row = f.result()
        if row:
            acc[s].append((day.strftime("%Y-%m-%d"), row["oi_usd"], row["ls_top"], row["ls_acct"], row["taker"]))
        done += 1
        if done % 4000 == 0:
            print(f"  {done}/{len(jobs)}")
for s in SYMS:
    if acc[s]:
        df = pd.DataFrame(acc[s], columns=["date", "oi_usd", "ls_top", "ls_acct", "taker"]).sort_values("date")
        df.to_csv(OUT / f"{s}_metrics.csv", index=False)
        print(f"{s:6s}: {len(df)} days {df.date.iloc[0]}->{df.date.iloc[-1]}")
    else:
        print(f"{s:6s}: EMPTY")
print("METRICS DONE")
