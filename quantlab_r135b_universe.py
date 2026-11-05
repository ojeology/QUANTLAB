#!/usr/bin/env python3
# R135b — UNIVERSE EXPANSION 20 -> 73 (prespec: docs/RESEARCH_FRONTIER_R135b.md, LOCKED)
import numpy as np, pandas as pd, glob, os, pathlib

os.makedirs("quantlab_cache/daily73", exist_ok=True)
STACK20 = {"BTC","ETH","SOL","XRP","DOGE","BNB","ADA","LINK","LTC","AVAX",
           "DOT","NEAR","APT","SUI","ATOM","FIL","INJ","ARB","OP","ETC"}

# 1) build daily caches from 1H parquets (idempotent)
for f in sorted(glob.glob("quantlab_cache/*_USDT_SWAP_1H.parquet")):
    sym = os.path.basename(f).replace("_USDT_SWAP_1H.parquet", "")
    out = f"quantlab_cache/daily73/{sym}_1D.csv"
    if os.path.exists(out): continue
    d = pd.read_parquet(f)
    d.columns = [c.lower() for c in d.columns]
    d = d.rename(columns={d.columns[0] if "open" not in d.columns else "open": "open"})
    idx = d.index.tz_convert("UTC").tz_localize(None) if d.index.tz else d.index
    d.index = idx
    day = d.resample("1D").agg(open=("open", "first"), high=("high", "max"),
                               low=("low", "min"), close=("close", "last"), nbars=("close", "size"))
    day = day[day.nbars >= 12].drop(columns="nbars")
    day.to_csv(out)

# 2) locked engine (identical to R135)
def run_book(files, cost_rt=0.0010, w=5):
    rows = []
    for f in files:
        sym = os.path.basename(f).replace("_1D.csv", "")
        df = pd.read_csv(f, parse_dates=["Unnamed: 0"]).rename(columns={"Unnamed: 0": "ts"}).set_index("ts").sort_index()
        c, lo, hi = df.close.values, df.low.values, df.high.values
        n = len(c)
        if n < 140: continue
        atr = pd.Series(np.maximum(hi - lo, np.abs(hi - np.roll(c, 1)))[1:], index=df.index[1:]).rolling(14).mean()
        pl = np.zeros(n, bool); ph = np.zeros(n, bool)
        for i in range(w, n - w):
            if lo[i] == lo[i-w:i+w+1].min(): pl[i] = True
            if hi[i] == hi[i-w:i+w+1].max(): ph[i] = True
        pos = None
        for t in range(120, n):
            if pos is not None:
                d_, ep, ei, ae, lp = pos
                px = c[t]
                if (px >= lp if d_ < 0 else px <= lp) or (t - ei) >= 10:
                    r = (ep - px)/ae if d_ < 0 else (px - ep)/ae
                    rows.append(dict(sym=sym, entry_ts=df.index[ei], exit_ts=df.index[t], hold=t-ei,
                                     dir=d_, r=r - cost_rt*ep/ae, year=df.index[t].year))
                    pos = None
                continue
            idxs = [i for i in range(max(0, t-120), t-w+1) if pl[i]]
            if len(idxs) >= 2:
                i2, i1 = idxs[-1], idxs[-2]
                if lo[i2] > lo[i1]:
                    s_ = (lo[i2]-lo[i1])/(i2-i1)
                    lt = lo[i2] + s_*(t-i2); lp_ = lo[i2] + s_*(t-1-i2)
                    if c[t] < lt and c[t-1] >= lp_:
                        a = atr.iloc[t-1]
                        if not np.isnan(a) and a > 0: pos = (-1, c[t], t, a, lt); continue
            idxs = [i for i in range(max(0, t-120), t-w+1) if ph[i]]
            if len(idxs) >= 2:
                i2, i1 = idxs[-1], idxs[-2]
                if hi[i2] < hi[i1]:
                    s_ = (hi[i2]-hi[i1])/(i2-i1)
                    lt = hi[i2] + s_*(t-i2); lp_ = hi[i2] + s_*(t-1-i2)
                    if c[t] > lt and c[t-1] <= lp_:
                        a = atr.iloc[t-1]
                        if not np.isnan(a) and a > 0: pos = (1, c[t], t, a, lt)
    return pd.DataFrame(rows)

def pf(x):
    x = np.asarray(x); w_ = x[x>0].sum(); l_ = abs(x[x<0].sum())
    return w_/l_ if l_ > 0 else float("inf")

files = sorted(glob.glob("quantlab_cache/daily73/*_1D.csv"))
syms = [os.path.basename(f).replace("_1D.csv","") for f in files]
print(f"universe: {len(files)} coins")
b = run_book(files)
b["tier"] = np.where(b.sym.isin(STACK20), "R106-20", "NEW-53")
print(f"\n[w5 PRIMARY, cost 0.10% RT] n={len(b)}  WR={(b.r>0).mean()*100:.1f}%  PF@c={pf(b.r):.2f}  total={b.r.sum():+.1f}R")
for y, g in b.groupby("year"):
    print(f"    {y}: n={len(g):<5} {g.r.sum():+8.1f}R  {'OK' if g.r.sum()>=0 else 'FAIL'}  (coins={g.sym.nunique()})")
half = len(b.sort_values('entry_ts'))//2
bs = b.sort_values("entry_ts")
print(f"    halves: H1 {pf(bs.r.iloc[:half]):.2f} | H2 {pf(bs.r.iloc[half:]):.2f}")
for tier, g in b.groupby("tier"):
    print(f"    [{tier}] n={len(g)}  PF@c {pf(g.r):.2f}  {g.r.sum():+.1f}R  (coins={g.sym.nunique()})")
b2 = run_book(files, cost_rt=0.0020)
print(f"[cost stress 0.20% RT]  PF@c={pf(b2.r):.2f}  {b2.r.sum():+.1f}R")
b3 = run_book(files, w=3)
print(f"[w3 sensitivity]        n={len(b3)}  PF@c={pf(b3.r):.2f}  {b3.r.sum():+.1f}R")

g1 = pf(b.r) >= 1.15
act = b.groupby("year").sym.nunique()
g2 = all(b[b.year==y].r.sum() >= 0 for y in (2024,2025,2026) if act.get(y,0) >= 5)
g3 = pf(bs.r.iloc[:half]) >= 1.0 and pf(bs.r.iloc[half:]) >= 1.0
new53 = b[b.tier=="NEW-53"]
g4 = pf(new53.r) >= 1.0 and pf(b.r) >= 1.42*0.9
g5 = pf(b2.r) >= 1.0
print(f"\n  GATES: PF@c≥1.15 {'PASS' if g1 else 'FAIL'} | yearly {'PASS' if g2 else 'FAIL'} | halves {'PASS' if g3 else 'FAIL'} "
      f"| NEW-53 breadth {'PASS' if g4 else 'FAIL'} | 2×cost {'PASS' if g5 else 'FAIL'}")
print("  VERDICT: " + ("EDGE SCALES WITH BREADTH — cross-section law confirmed for trendline breaks"
      if all((g1,g2,g3,g4,g5)) else "BREADTH DILUTES or mixed — inspect gate detail"))
b.to_csv("quantlab_output/r135b_book.csv", index=False)
print("saved: r135b_book.csv")
