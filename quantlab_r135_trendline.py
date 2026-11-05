#!/usr/bin/env python3
# R135 — TRENDLINE BREAK, MECHANIZED (prespec: docs/RESEARCH_FRONTIER_R135.md, LOCKED)
import numpy as np, pandas as pd, pathlib

D = sorted(pathlib.Path("quantlab_cache/daily").glob("*_1D.csv"))
COST_RT = 0.0010  # 0.10% round trip

def run_variant(w):
    rows = []
    for f in D:
        sym = f.stem.replace("_1D", "")
        df = pd.read_csv(f, parse_dates=["ts"]).set_index("ts").sort_index()
        c, lo, hi = df.c.values, df.l.values, df.h.values
        n = len(c)
        atr = pd.Series(np.maximum(hi - lo, np.abs(hi - np.roll(c, 1)))[1:], index=df.index[1:]).rolling(14).mean()
        # pivots (confirmed w bars later)
        pl = np.zeros(n, bool); ph = np.zeros(n, bool)
        for i in range(w, n - w):
            if lo[i] == lo[i-w:i+w+1].min(): pl[i] = True
            if hi[i] == hi[i-w:i+w+1].max(): ph[i] = True
        pos = None  # (dir, entry_px, entry_i, atr_e, line_px_at_entry_i)
        for t in range(120, n):
            # exit first
            if pos is not None:
                d, ep, ei, ae, lp = pos
                px = c[t]
                crossed_back = px >= lp if d < 0 else px <= lp
                if crossed_back or (t - ei) >= 10:
                    r = (ep - px)/ae if d < 0 else (px - ep)/ae
                    rows.append(dict(sym=sym, entry_ts=df.index[ei], exit_ts=df.index[t],
                                     hold=t-ei, dir=d, r=r - 2*0.0005*ep/ae,
                                     year=df.index[t].year))
                    pos = None
                continue
            # uptrend support line -> SHORT on fresh breakdown
            idxs = [i for i in range(max(0, t-120), t-w+1) if pl[i]]
            if len(idxs) >= 2:
                i2, i1 = idxs[-1], idxs[-2]
                if lo[i2] > lo[i1]:
                    slope = (lo[i2]-lo[i1])/(i2-i1)
                    lt = lo[i2] + slope*(t-i2); lprev = lo[i2] + slope*(t-1-i2)
                    if c[t] < lt and c[t-1] >= lprev:
                        a = atr.iloc[t-1]
                        if not np.isnan(a) and a > 0:
                            pos = (-1, c[t], t, a, lt); continue
            idxs = [i for i in range(max(0, t-120), t-w+1) if ph[i]]
            if len(idxs) >= 2:
                i2, i1 = idxs[-1], idxs[-2]
                if hi[i2] < hi[i1]:
                    slope = (hi[i2]-hi[i1])/(i2-i1)
                    lt = hi[i2] + slope*(t-i2); lprev = hi[i2] + slope*(t-1-i2)
                    if c[t] > lt and c[t-1] <= lprev:
                        a = atr.iloc[t-1]
                        if not np.isnan(a) and a > 0:
                            pos = (1, c[t], t, a, lt)
    return pd.DataFrame(rows)

def pf(x):
    x = np.asarray(x); w = x[x>0].sum(); l = abs(x[x<0].sum())
    return w/l if l > 0 else float("inf")

print("="*100)
print("  R135 — TRENDLINE BREAK, MECHANIZED — 20 coins, dailies 2023-06 → 2026-11, costs 0.10% RT")
print("="*100)
res = {}
for w in (3, 5):
    b = run_variant(w)
    tag = f"w{w}" + (" (PRIMARY)" if w == 5 else "")
    if len(b) == 0:
        print(f"\n[{tag}] n=0"); continue
    yrs = b.groupby("year").r.agg(["count", "sum"])
    half = len(b)//2
    h1, h2 = b.r.iloc[:half], b.r.iloc[half:]
    res[w] = b
    print(f"\n[{tag}] n={len(b)}  WR={((b.r>0).mean()*100):.1f}%  PF@c={pf(b.r):.2f}  total={b.r.sum():+.1f}R")
    for y, row in yrs.iterrows():
        print(f"    {y}: n={int(row['count'])}  {row['sum']:+.1f}R {'≥0 OK' if row['sum']>=0 else '<0 FAIL'}")
    print(f"    split: H1 PF@c {pf(h1):.2f} (n={len(h1)}) | H2 PF@c {pf(h2):.2f} (n={len(h2)})")

if 5 in res:
    b = res[5]
    g1 = len(b) >= 30
    g2 = pf(b.r) >= 1.15
    g3 = all(b[b.year==y].r.sum() >= 0 for y in (2024, 2025, 2026) if len(b[b.year==y]))
    half = len(b)//2
    g4 = pf(b.r.iloc[:half]) >= 1.0 and pf(b.r.iloc[half:]) >= 1.0
    print(f"\n  GATES (w5 primary): n≥30 {'PASS' if g1 else 'FAIL'} | PF@c≥1.15 {'PASS' if g2 else 'FAIL'} "
          f"| yearly≥0R {'PASS' if g3 else 'FAIL'} | both-halves {'PASS' if g4 else 'FAIL'}")
    verd = "TRENDLINE CARRIES EDGE" if all((g1, g2, g3, g4)) else (
        "COIN-FLIP CONFIRMED — the drawn line carries no information beyond noise"
        if not g2 and not g4 else "MIXED — inspect splits")
    print(f"  VERDICT: {verd}")
    b.to_csv("quantlab_output/r135_trendline_trades.csv", index=False)
    print("saved: r135_trendline_trades.csv")
