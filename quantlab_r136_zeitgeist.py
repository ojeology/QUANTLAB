#!/usr/bin/env python3
# R136 — YOUTUBE ZEITGEIST: SUPERTREND + ICHIMOKU (prespec: docs/RESEARCH_FRONTIER_R136.md, LOCKED)
import numpy as np, pandas as pd, glob, os

COST = 0.0010

def wilder_atr(h, l, c, n=14):
    pc = np.roll(c, 1); pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1/n, adjust=False).mean().values

def cell_supertrend(files, mult=3.0, period=10, cost=COST):
    rows = []
    for f in files:
        sym = os.path.basename(f).replace("_1D.csv", "")
        df = pd.read_csv(f, parse_dates=["ts"]).set_index("ts").sort_index()
        h, l, c = df.h.values, df.l.values, df.c.values
        n = len(c)
        atr = wilder_atr(h, l, c, 14)
        hl2 = (h + l) / 2
        bu, bd = hl2 + mult*atr, hl2 - mult*atr
        fu, fl = bu.copy(), bd.copy()
        trend = np.ones(n, int)  # 1 up, -1 down
        pos = None
        for i in range(1, n):
            fu[i] = bu[i] if (bu[i] < fu[i-1] or c[i-1] > fu[i-1]) else fu[i-1]
            fl[i] = bd[i] if (bd[i] > fl[i-1] or c[i-1] < fl[i-1]) else fl[i-1]
            prev = trend[i-1]
            if prev == 1 and c[i] < fl[i]: trend[i] = -1
            elif prev == -1 and c[i] > fu[i]: trend[i] = 1
            else: trend[i] = prev
            if pos is not None and trend[i] == -1 and trend[i-1] == 1:
                a = max(atr[pos], 1e-12)
                r = (c[i] - c[pos])/a - cost*c[i]/a
                rows.append(dict(sym=sym, entry_ts=df.index[pos], exit_ts=df.index[i], hold=i-pos,
                                 r=r, year=df.index[i].year))
                pos = None
            if pos is None and trend[i] == 1 and trend[i-1] == -1 and not np.isnan(atr[i]) and atr[i] > 0:
                pos = i
    return pd.DataFrame(rows)

def cell_ichimoku(files, cost=COST):
    rows = []
    for f in files:
        sym = os.path.basename(f).replace("_1D.csv", "")
        df = pd.read_csv(f, parse_dates=["ts"]).set_index("ts").sort_index()
        h, l, c = df.h.values, df.l.values, df.c.values
        n = len(c)
        tenkan = (pd.Series(h).rolling(9).max() + pd.Series(l).rolling(9).min()).values / 2
        kijun  = (pd.Series(h).rolling(26).max() + pd.Series(l).rolling(26).min()).values / 2
        spb_raw = (pd.Series(h).rolling(52).max() + pd.Series(l).rolling(52).min()).values / 2
        spa = np.full(n, np.nan); spb = np.full(n, np.nan)
        spa[26:] = (tenkan + kijun)[26-2*n:n-26] if False else (tenkan[:n-26] + kijun[:n-26])/2
        spb[26:] = spb_raw[:n-26]
        cloud_hi = np.fmax(spa, spb); cloud_lo = np.fmin(spa, spb)
        pos = None
        for t in range(52 + 26, n):
            if pos is not None:
                p_, ei = pos
                hit = c[t] < kijun[t] or c[t] < cloud_lo[t] or (t - ei) >= 30
                if hit:
                    a = max((pd.Series(h).rolling(26).max() + pd.Series(l).rolling(26).min()).values/2)[0] if False else max(kijun[ei], 1e-12)
                    rows.append(dict(sym=sym, entry_ts=df.index[ei], exit_ts=df.index[t], hold=t-ei,
                                     r=(c[t]-c[p_])/a - cost*c[p_]/a, year=df.index[t].year))
                    pos = None
                continue
            cross = tenkan[t] > kijun[t] and tenkan[t-1] <= kijun[t-1]
            if cross and c[t] > cloud_hi[t] and not np.isnan(kijun[t]) and kijun[t] > 0:
                pos = (t, t)
    return pd.DataFrame(rows)

def pf(x):
    x = np.asarray(x); w = x[x>0].sum(); l = abs(x[x<0].sum())
    return w/l if l > 0 else float("inf")

def report(name, b, b2x):
    print(f"\n=== {name} ===")
    if len(b) < 1: print("  n=0"); return
    print(f"  n={len(b)}  WR={(b.r>0).mean()*100:.1f}%  PF@c={pf(b.r):.2f}  total={b.r.sum():+.1f}R")
    for y, g in b.groupby("year"):
        print(f"    {y}: n={len(g):<5} {g.r.sum():+8.1f}R {'OK' if g.r.sum()>=0 else 'FAIL'}")
    half = len(b)//2
    print(f"  halves: H1 {pf(b.r.iloc[:half]):.2f} | H2 {pf(b.r.iloc[half:]):.2f}  | 2x-cost PF@c {pf(b2x.r):.2f}")
    g = [len(b)>=30, pf(b.r)>=1.15,
         all(b[b.year==y].r.sum()>=0 for y in (2024,2025,2026) if len(b[b.year==y])),
         pf(b.r.iloc[:half])>=1.0 and pf(b.r.iloc[half:])>=1.0, pf(b2x.r)>=1.0]
    names = ["n≥30", "PF≥1.15", "yearly", "halves", "2×cost"]
    print("  GATES: " + " | ".join(f"{nm} {'PASS' if x else 'FAIL'}" for nm, x in zip(names, g)))
    print("  VERDICT: " + ("CANDIDATE (all gates)" if all(g) else "CLOSED — noise/dead"))

files = sorted(glob.glob("quantlab_cache/daily/*_1D.csv"))
print(f"universe: {len(files)} coins, {pd.read_csv(files[0], nrows=1).columns.tolist()}")
bA = cell_supertrend(files); bA2 = cell_supertrend(files, cost=0.0020)
report("CELL A — SUPERTREND (10, 3.0, long-only)", bA, bA2)
bB = cell_ichimoku(files); bB2 = cell_ichimoku(files, cost=0.0020)
report("CELL B — ICHIMOKU (9/26/52, TK-cross above cloud)", bB, bB2)
if len(bA): bA.to_csv("quantlab_output/r136_supertrend_book.csv", index=False)
if len(bB): bB.to_csv("quantlab_output/r136_ichimoku_book.csv", index=False)
print("\nsaved books (if nonempty)")
