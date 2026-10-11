"""
R099 data step — discover liquid OKX USDT perpetuals NOT in the 73-symbol universe,
fetch ~12000 1H bars each (page backward via /market/history-candles).

Output: quantlab_cache/newpairs/{SYM}_1H.parquet (same schema as main cache).
Universe: top N by est. USD 24h volume, excluding every symbol already in the
champion universe (ORIGINAL52+NEW18+NEW3) and margin-currency quirks.
"""
import os, sys, json, time, subprocess
import pandas as pd

N_TARGET = int(os.environ.get("NP_N", "30"))
MAX_BARS = int(os.environ.get("NP_BARS", "12000"))
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "quantlab_cache", "newpairs")
os.makedirs(OUTDIR, exist_ok=True)

UNIVERSE = {
    "1INCH-USDT-SWAP","AAVE-USDT-SWAP","ADA-USDT-SWAP","ALGO-USDT-SWAP","APT-USDT-SWAP",
    "ARB-USDT-SWAP","ATOM-USDT-SWAP","AVAX-USDT-SWAP","AXS-USDT-SWAP","BCH-USDT-SWAP",
    "BNB-USDT-SWAP","BONK-USDT-SWAP","BTC-USDT-SWAP","CHZ-USDT-SWAP","COMP-USDT-SWAP",
    "CRV-USDT-SWAP","DOGE-USDT-SWAP","DOT-USDT-SWAP","DYDX-USDT-SWAP","EGLD-USDT-SWAP",
    "ENA-USDT-SWAP","ETC-USDT-SWAP","ETH-USDT-SWAP","FET-USDT-SWAP","FIL-USDT-SWAP",
    "FLOKI-USDT-SWAP","GALA-USDT-SWAP","GMX-USDT-SWAP","GRT-USDT-SWAP","HBAR-USDT-SWAP",
    "ICP-USDT-SWAP","IMX-USDT-SWAP","INJ-USDT-SWAP","LDO-USDT-SWAP","LINK-USDT-SWAP",
    "LTC-USDT-SWAP","NEAR-USDT-SWAP","OP-USDT-SWAP","PEPE-USDT-SWAP","SAND-USDT-SWAP",
    "SATS-USDT-SWAP","SHIB-USDT-SWAP","SNX-USDT-SWAP","SOL-USDT-SWAP","STX-USDT-SWAP",
    "SUI-USDT-SWAP","SUSHI-USDT-SWAP","TRX-USDT-SWAP","UNI-USDT-SWAP","WIF-USDT-SWAP",
    "XLM-USDT-SWAP","XRP-USDT-SWAP",
    "BICO-USDT-SWAP","HYPE-USDT-SWAP","XAU-USDT-SWAP","HOME-USDT-SWAP","PUMP-USDT-SWAP",
    "ZBT-USDT-SWAP","ZEC-USDT-SWAP","BEAT-USDT-SWAP","SNDK-USDT-SWAP","SPCX-USDT-SWAP",
    "MU-USDT-SWAP","SKHYNIX-USDT-SWAP","SOXL-USDT-SWAP","UB-USDT-SWAP","SNXX-USDT-SWAP",
    "SKHY-USDT-SWAP","KORU-USDT-SWAP","CL-USDT-SWAP",
    "XAG-USDT-SWAP","ALLO-USDT-SWAP","AAOI-USDT-SWAP",
}

UA = "Mozilla/5.0"
def get(url, tries=4):
    for a in range(tries):
        p = subprocess.run(["curl","-s","-4","--max-time","20","-A",UA,url], capture_output=True, timeout=30)
        try:
            d = json.loads(p.stdout.decode())
            if d.get("code") == "0": return d["data"]
            time.sleep(1+a)
        except Exception:
            time.sleep(1+a)
    return []

# 1) discover
tick = get("https://www.okx.com/api/v5/market/tickers?instType=SWAP")
cands=[]
for t in tick:
    inst = t.get("instId","")
    if not inst.endswith("-USDT-SWAP"): continue
    if inst in UNIVERSE: continue
    try:
        last = float(t["last"]); vol = float(t.get("volCcy24h", 0) or 0)
        usd = last*vol
    except Exception:
        continue
    if usd <= 0: continue
    cands.append((inst, usd))
cands.sort(key=lambda x:-x[1])
picks = [c for c,_ in cands[:N_TARGET]]
print(f"{len(cands)} non-universe USDT swaps; top {len(picks)} by 24h USD vol:", flush=True)
for inst, usd in cands[:N_TARGET]:
    print(f"  {inst:<22} ${usd/1e6:8.1f}M/24h", flush=True)

# 2) fetch ~MAX_BARS 1H bars per symbol
def fetch_history(inst, max_bars):
    rows={}
    after=None
    for page in range(max_bars//100 + 5):
        url=f"https://www.okx.com/api/v5/market/history-candles?instId={inst}&bar=1H&limit=100"
        if after: url += f"&after={after}"
        data=get(url)
        if not data: break
        for r in data:
            ts=int(r[0])
            if ts not in rows:
                rows[ts]=[float(r[1]),float(r[2]),float(r[3]),float(r[4]),float(r[5])]
        oldest=int(data[-1][0])
        if len(rows)>=max_bars or len(data)<100: break
        after=oldest
        time.sleep(0.08)
    if not rows: return None
    df=pd.DataFrame.from_dict(rows,orient="index",columns=["open","high","low","close","vol"]).sort_index()
    df.index=pd.to_datetime(df.index,unit="ms",utc=True)
    df=df[~df.index.duplicated(keep="last")]
    return df

t0=time.time()
ok=0
for k,inst in enumerate(picks,1):
    sym=inst.replace("-","_")
    out=os.path.join(OUTDIR,f"{sym}_1H.parquet")
    if os.path.exists(out):
        print(f"  [{k}/{len(picks)}] {sym}: cached", flush=True); ok+=1; continue
    df=fetch_history(inst,MAX_BARS)
    if df is None or len(df)<800:
        print(f"  [{k}/{len(picks)}] {sym}: insufficient ({0 if df is None else len(df)} bars)", flush=True); continue
    df.to_parquet(out); ok+=1
    print(f"  [{k}/{len(picks)}] {sym}: {len(df)} bars  {df.index.min():%Y-%m-%d} → {df.index.max():%Y-%m-%d}  ({time.time()-t0:.0f}s)", flush=True)
print(f"Done: {ok}/{len(picks)} symbols → {OUTDIR} in {time.time()-t0:.0f}s", flush=True)
