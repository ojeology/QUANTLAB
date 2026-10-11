"""
Fetch OKX funding-rate history (8h epochs) for the champion's liquid symbols.
Output: quantlab_cache/funding/{SYM}.csv  (fundingTime UTC, fundingRate)
"""
import os, json, time, subprocess
import pandas as pd

SYMS = ["BTC","ETH","SOL","XRP","DOGE","BNB","ADA","LINK","LTC","AVAX",
        "TRX","DOT","NEAR","APT","SUI","ATOM","FIL","INJ","ARB","OP",
        "ETC","IMX","STX","HBAR"]
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "quantlab_cache", "funding")
os.makedirs(OUT, exist_ok=True)
UA = "Mozilla/5.0"

def get(url, tries=4):
    for a in range(tries):
        p = subprocess.run(["curl","-s","-4","--max-time","20","-A",UA,url], capture_output=True, timeout=30)
        try:
            d = json.loads(p.stdout.decode())
            if d.get("code") == "0": return d["data"]
            time.sleep(1+a)
        except Exception: time.sleep(1+a)
    return []

t0=time.time()
for k, s in enumerate(SYMS, 1):
    inst = f"{s}-USDT-SWAP"
    out = os.path.join(OUT, f"{s}.csv")
    if os.path.exists(out):
        print(f"[{k}/{len(SYMS)}] {s}: cached"); continue
    rows={}; after=None
    for page in range(60):
        url=f"https://www.okx.com/api/v5/public/funding-rate-history?instId={inst}&limit=100"
        if after: url += f"&after={after}"
        data=get(url)
        if not data: break
        for r in data:
            rows[int(r["fundingTime"])] = float(r["fundingRate"])
        oldest=int(data[-1]["fundingTime"])
        if len(data)<100: break
        after=oldest
        time.sleep(0.06)
    if rows:
        df=pd.DataFrame(sorted(rows.items()), columns=["fundingTime","rate"])
        df["fundingTime"]=pd.to_datetime(df["fundingTime"],unit="ms",utc=True)
        df.to_csv(out,index=False)
        print(f"[{k}/{len(SYMS)}] {s}: {len(df)} epochs  {df['fundingTime'].min():%Y-%m-%d} → {df['fundingTime'].max():%Y-%m-%d}  ({time.time()-t0:.0f}s)")
    else:
        print(f"[{k}/{len(SYMS)}] {s}: NO DATA")
print("done")
