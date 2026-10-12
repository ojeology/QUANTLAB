#!/usr/bin/env python3
"""Fetch multi-year funding + 1D klines for Binance USDT-perps from data.binance.vision
(public data-dump CDN, probed reachable 2026-10-12). -> quantlab_cache/bnfut/{SYM}_fund.csv, {SYM}_1D.csv"""
import io, json, pathlib, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

OUT = pathlib.Path("quantlab_cache/bnfut"); OUT.mkdir(parents=True, exist_ok=True)
SYMS = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LINK", "LTC", "AVAX",
        "DOT", "NEAR", "APT", "SUI", "ATOM", "FIL", "INJ", "ARB", "OP", "ETC",
        "TRX", "XLM", "HBAR", "CRV", "AAVE", "UNI", "TIA", "SEI", "WIF", "PEPE",
        "SHIB", "ORDI", "JUP", "PYTH", "TON", "FET", "LDO", "GRT", "ALGO", "EOS"]
MONTHS = [(y, m) for y in (2023, 2024, 2025, 2026) for m in range(1, 13)
          if (y, m) >= (2023, 9) and (y, m) <= (2026, 9)]

def get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                return None
            time.sleep(1.5 * (k + 1))

def month_rows(sym, y, m):
    tag = f"{y}-{m:02d}"
    fr = get(f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{sym}USDT/{sym}USDT-fundingRate-{tag}.zip")
    kl = get(f"https://data.binance.vision/data/futures/um/monthly/klines/{sym}USDT/1d/{sym}USDT-1d-{tag}.zip")
    frows, krows = [], []
    if fr:
        try:
            zf = zipfile.ZipFile(io.BytesIO(fr)); name = zf.namelist()[0]
            fdf = pd.read_csv(zf.open(name), header=None)
            frows = fdf.iloc[:, [0, -1]].values.tolist()  # calc_time, rate (last col robust to format)
        except Exception:
            pass
    if kl:
        try:
            zk = zipfile.ZipFile(io.BytesIO(kl)); name = zk.namelist()[0]
            kdf = pd.read_csv(zk.open(name), header=None)
            if not str(kdf.iloc[0, 0]).replace(".", "").replace("-", "").isdigit():
                kdf = pd.read_csv(zf.open(name) if False else kdf)  # headered file
                kdf = kdf.drop(0) if str(kdf.iloc[0, 4]).lower() == "close" else kdf
            for _, r in kdf.iterrows():
                v = r.iloc[0]; s = str(v)
                d = s[:10] if "-" in s else pd.to_datetime(int(float(v)), unit="ms").strftime("%Y-%m-%d")
                krows.append([d, float(r.iloc[4])])
        except Exception:
            pass
    return sym, frows, krows

jobs = [(s, y, m) for s in SYMS for (y, m) in MONTHS]
fund_acc = {s: [] for s in SYMS}; k_acc = {s: [] for s in SYMS}
done = 0
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = {ex.submit(month_rows, s, y, m): (s,) for s, y, m in jobs}
    for f in as_completed(futs):
        s, frows, krows = f.result()
        fund_acc[s] += frows; k_acc[s] += krows
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(jobs)} month-files...")
for s in SYMS:
    if fund_acc[s]:
        fdf = pd.DataFrame(fund_acc[s], columns=["ts_ms", "rate"]).drop_duplicates("ts_ms").sort_values("ts_ms")
        fdf.to_csv(OUT / f"{s}_fund.csv", index=False)
    if k_acc[s]:
        kdf = pd.DataFrame(k_acc[s], columns=["date", "close"]).drop_duplicates("date").sort_values("date")
        kdf.to_csv(OUT / f"{s}_1D.csv", index=False)
    fr = f"{len(fund_acc[s])} fund rows, {len(k_acc[s])} klines"
    span = f"{k_acc[s][0][0]}->{k_acc[s][-1][0]}" if k_acc[s] else "EMPTY"
    print(f"{s:6s}: {fr}  {span}")
print("DONE")
