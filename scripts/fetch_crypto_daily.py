#!/usr/bin/env python3
"""Fetch daily candles for the 20 stack coins from OKX history-candles -> quantlab_cache/daily/{COIN}_1D.csv"""
import subprocess, time, pathlib, json
import pandas as pd

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK", "LTC", "AVAX",
         "DOT", "NEAR", "APT", "SUI", "ATOM", "FIL", "INJ", "ARB", "OP", "ETC"]
OUT = pathlib.Path("quantlab_cache/daily"); OUT.mkdir(parents=True, exist_ok=True)
START = "2023-06-01"  # warmup safety

def get(inst, after=""):
    url = f"https://www.okx.com/api/v5/market/history-candles?instId={inst}&bar=1D&limit=100" + (f"&after={after}" if after else "")
    r = subprocess.run(["curl", "-4", "-s", "-m", "20", url], capture_output=True, text=True)
    return json.loads(r.stdout)

for c in COINS:
    f = OUT / f"{c}_1D.csv"
    if f.exists():
        print(f"{c}: cached"); continue
    rows = []; after = ""
    for _ in range(60):
        d = get(f"{c}-USDT", after)
        if d.get("code") != "0" or not d["data"]:
            break
        rows += d["data"]
        oldest = d["data"][-1][0]
        after = oldest
        if pd.to_datetime(int(oldest), unit="ms") < pd.Timestamp(START):
            break
        time.sleep(0.15)
    if not rows:
        print(f"{c}: EMPTY"); continue
    df = pd.DataFrame(rows)
    df = df.iloc[:, :5]
    df.columns = ["ts", "o", "h", "l", "c"]
    df["ts"] = pd.to_datetime(df.ts.astype("int64"), unit="ms").dt.tz_localize(None)
    for col in ["o", "h", "l", "c"]:
        df[col] = df[col].astype(float)
    df = (df.drop_duplicates("ts").sort_values("ts")
            .query("ts >= @START")[["ts", "o", "h", "l", "c"]].reset_index(drop=True))
    df.to_csv(f, index=False)
    print(f"{c}: {len(df)} d  {df.ts.iloc[0].date()} -> {df.ts.iloc[-1].date()}")
print("DONE")
