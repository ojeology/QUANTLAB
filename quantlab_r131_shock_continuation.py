#!/usr/bin/env python3
# R131 — SHOCK-CONTINUATION (prespec: docs/RESEARCH_FRONTIER_R131.md, LOCKED)
import os, sys, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features

CACHE = CONFIG["CACHE_FOLDER"]
SPLIT = pd.Timestamp("2025-10-01", tz="UTC")
W = 720
SKIP = ("XAU","XAG","CL","SOXL","MU","SPCX","AAOI","UB","SNDK","SKHY","BEAT","ALLO","ZBT","SNXX","PUMP","HOME")
COST_R = 0.0014  # round trip in price units; converted to R per trade via ATR

def pf(x):
    x = np.asarray(x, float); w = x[x>0].sum(); lo = abs(x[x<0].sum())
    return w/lo if lo>0 else 999.0

books = {}
for V, hold, use_sl in [("V1", 24, False), ("V2", 72, False), ("V3", 24, True)]:
    rows = []
    for f in sorted(glob.glob(os.path.join(CACHE, "*_USDT_SWAP_1H.parquet"))):
        sym = os.path.basename(f).replace("_USDT_SWAP_1H.parquet", "")
        if sym.startswith(SKIP): continue
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        df = df[~df.index.duplicated()]
        if len(df) < W + 300: continue
        ft = add_features(df)
        ret = ft["close"].pct_change()
        sig = ret.rolling(W).std()
        rng = (ft["high"] - ft["low"])
        rel_vol = rng / rng.rolling(20).mean()
        atr = ft["atr14"]
        up_shock = (ret >= 2*sig) & (ret >= 0.01) & (rel_vol >= 1.5) & (ft["close"] > ft["open"])
        dn_shock = (ret <= -2*sig) & (ret <= -0.01) & (rel_vol >= 1.5) & (ft["close"] < ft["open"])
        c, l, h = ft["close"].values, ft["low"].values, ft["high"].values
        idx = ft.index
        for side, mask in [("LONG", up_shock)]:
            last_exit = -1
            for i in np.where(mask.fillna(False).values)[0]:
                if i <= last_exit or i + 1 >= len(idx) or i + hold >= len(idx): continue
                if not np.isfinite(atr.iloc[i]) or atr.iloc[i] <= 0: continue
                entry = c[i]; a = atr.iloc[i]
                if use_sl:
                    sl = entry - a; exit_px = None; j = i + 1
                    while j <= i + hold:
                        if l[j] <= sl: exit_px = sl; break
                        j += 1
                    if exit_px is None: exit_px = c[i + hold]
                else:
                    exit_px = c[i + hold]
                r_mult = (exit_px/entry - 1)
                r = r_mult * entry / a - COST_R * entry / a
                last_exit = i + hold
                rows.append((idx[i], sym, side, r, bool(idx[i] < SPLIT)))
    b = pd.DataFrame(rows, columns=["entry_time","sym","side","r","era_disc"])
    books[V] = b
    print(f"V{V[-1]} (hold {hold}" + (", SL 1×ATR" if use_sl else "") + f"): n={len(b)}")

print("\n" + "="*100)
print("  R131 — SHOCK-CONTINUATION (long after UP-shocks; challenger-grade seen book)")
print("="*100)
print(f"  {'var':<4}{'n':>6}{'PF@c':>7}{'sumR':>8}{'WR':>5} | {'DISC PF@c':>9}{'n':>7} | {'CONF PF@c':>9}{'n':>7} | {'maxDD@1%':>9}")
best = None
for V, b in books.items():
    if not len(b): continue
    d, c = b[b.era_disc], b[~b.era_disc]
    eq = 100 + b.r.cumsum(); dd = (eq/eq.cummax()-1).min()*100
    print(f"  {V:<4}{len(b):>6}{pf(b.r):>7.2f}{b.r.sum():>+8.1f}{(b.r>0).mean()*100:>4.0f}% | "
          f"{pf(d.r):>9.2f}{len(d):>7} | {pf(c.r):>9.2f}{len(c):>7} | {dd:>8.1f}%")
    g1, g2, g3, g4 = pf(d.r) >= 1.30, pf(c.r) >= 1.20, len(b) >= 200, abs(dd) <= 20
    ok = all([g1,g2,g3,g4])
    print(f"      gates: G1 {'PASS' if g1 else 'FAIL'} G2 {'PASS' if g2 else 'FAIL'} G3 {'PASS' if g3 else 'FAIL'} G4 {'PASS' if g4 else 'FAIL'}"
          + ("  -> PROMISING-CANDIDATE" if ok else "  -> CLOSED"))
    if ok: best = V

# report-only: short mirror on V1
rows_s = []
for f in sorted(glob.glob(os.path.join(CACHE, "*_USDT_SWAP_1H.parquet"))):
    sym = os.path.basename(f).replace("_USDT_SWAP_1H.parquet", "")
    if sym.startswith(SKIP): continue
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    df = df[~df.index.duplicated()]
    if len(df) < W + 300: continue
    ft = add_features(df)
    ret = ft["close"].pct_change(); sig = ret.rolling(W).std()
    rng = ft["high"] - ft["low"]; rel_vol = rng / rng.rolling(20).mean()
    dn = (ret <= -2*sig) & (ret <= -0.01) & (rel_vol >= 1.5) & (ft["close"] < ft["open"])
    c_, l_, h_ = ft["close"].values, ft["low"].values, ft["high"].values
    idx = ft.index; atr = ft["atr14"]; last_exit = -1
    for i in np.where(dn.fillna(False).values)[0]:
        if i <= last_exit or i + 25 >= len(idx): continue
        if not np.isfinite(atr.iloc[i]) or atr.iloc[i] <= 0: continue
        entry = c_[i]; a = atr.iloc[i]
        r = (entry/c_[i+24] - 1) * entry / a - COST_R * entry / a
        last_exit = i + 24
        rows_s.append((idx[i], sym, "SHORT", r))
s = pd.DataFrame(rows_s, columns=["entry_time","sym","side","r"])
print(f"\n  report-only SHORT mirror (V1): n={len(s)}  PF@c={pf(s.r):.2f}  sum={s.r.sum():+.1f}R")

b1 = books["V1"]
if len(b1):
    b1m = b1.copy(); b1m["mo"] = b1m.entry_time.dt.tz_localize(None).dt.to_period("M")
    print("  V1 monthly R: " + " ".join(f"{str(m)}:{v:+.0f}" for m, v in b1m.groupby("mo").r.sum().tail(8).items()))
    # overlap with champion book
    try:
        champ = pd.read_csv("quantlab_output/r096_trades_q055.csv", parse_dates=["entry_time"])
        ks = set(zip(champ.entry_time, champ.sym)); ks2 = set(zip(b1m.entry_time, b1m.sym))
        print(f"  overlap with champion kept book: {len(ks & ks2)} of V1 {len(ks2)} (champion {len(ks)})")
        loao = []
        for sym in b1m.sym.unique():
            loao.append(pf(b1m[b1m.sym != sym].r))
        print(f"  LOAO-min PF (V1): {min(loao):.2f}")
    except Exception as e:
        print(f"  overlap skipped: {e}")

print("\n  VERDICT: " + (f"V{best[-1]} PROMISING-CANDIDATE (challenger #4, forward gauntlet)" if best else
      "CLOSED — expansion cell not tradeable net of costs; R130 lifts were real, the book is not"))
for V, b in books.items():
    b.to_csv(f"quantlab_output/r131_{V}_book.csv", index=False)
print("saved: r131_V1/V2/V3 books")
