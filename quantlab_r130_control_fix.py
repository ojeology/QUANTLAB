#!/usr/bin/env python3
# R130 diagnostic — corrected positive controls (trigger-gated) after atlas control failure.
import os, sys, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, build_signal_mask
from scipy.stats import binomtest

CACHE = CONFIG["CACHE_FOLDER"]
SPLIT = pd.Timestamp("2025-10-01", tz="UTC")
W = 720
SKIP = ("XAU","XAG","CL","SOXL","MU","SPCX","AAOI","UB","SNDK","SKHY","BEAT","ALLO","ZBT","SNXX","PUMP","HOME")

coins, masks = {}, {}
for f in sorted(glob.glob(os.path.join(CACHE, "*_USDT_SWAP_1H.parquet"))):
    sym = os.path.basename(f).replace("_USDT_SWAP_1H.parquet", "")
    if sym.startswith(SKIP): continue
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    df = df[~df.index.duplicated()]
    if len(df) < W + 200: continue
    ft = add_features(df)
    ret = ft["close"].pct_change()
    ft = ft.assign(ret=ret, hsig=ret.rolling(W).std(),
                   edist_pct=ft["ema_dist_pct"].rolling(W).rank(pct=True),
                   bbw_pct=ft["bb_width"].rolling(W).rank(pct=True),
                   rv_pct=ft["real_vol_20"].rolling(W).rank(pct=True))
    coins[sym] = ft
    masks[sym] = build_signal_mask(ft, ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"], "green", 1.5)
print(f"coins: {len(coins)}")

def fwd_up_series(ft):
    bigup = (ft.ret >= 4*ft.hsig) & (ft.ret >= 0.015)
    return bigup.iloc[::-1].rolling(24, min_periods=1).max().iloc[::-1].shift(-1).fillna(0).astype(bool)

FU = {s: fwd_up_series(ft) for s, ft in coins.items()}
ERA = {s: ft.index < SPLIT for s, ft in coins.items()}

def control(name, gate_fn):
    stats = {}
    for era, sel_era in [("DISC", True), ("CONF", False)]:
        ev = n = b = bn = 0
        for sym, ft in coins.items():
            m = gate_fn(sym, ft)
            if m is None: continue
            era_ok = ERA[sym] if sel_era else ~ERA[sym]
            sel = m & era_ok
            n += int(sel.sum()); ev += int(FU[sym][sel].sum())
            w = era_ok & ft.ret.notna() & ft.hsig.notna()
            bn += int(w.sum()); b += int(FU[sym][w].sum())
        lift = (ev/n)/(b/bn) if n and bn else np.nan
        p = binomtest(ev, n, b/bn).pvalue if n else np.nan
        stats[era] = (lift, n, p)
    d, c = stats["DISC"], stats["CONF"]
    print(f"  {name:<42} DISC lift={d[0]:.2f} (n={d[1]:>7,}, p={d[2]:.1e})   CONF lift={c[0]:.2f} (n={c[1]:,})")
    return d, c

print("\nCORRECTED CONTROLS:")
print("  [1] champion FULL signal (state + trigger bar) — the real mechanism")
d1, c1 = control("champion signal mask", lambda s, ft: masks[s])
print("  [2] pctile-coil STATE ONLY (no trigger) — corrected state def")
d2, c2 = control("pctile coil state", lambda s, ft: (ft.bbw_pct <= .25) & (ft.rv_pct <= .33) & (ft.edist_pct <= .33))
print("  [3] atlas abs-coil STATE ONLY — reproduce the failure")
d3, c3 = control("abs coil state (edist<=0)", lambda s, ft: (ft.bbw_pct <= .25) & (ft.rv_pct <= .33) & (ft["ema_dist_pct"] <= 0))
print("  [4] trigger bar WITHOUT coil state (green+vol only)")
def trig_only(sym, ft):
    c, o = ft["close"], ft["open"]
    rv = ft["rel_vol"]
    return (c > o) & (c > c.shift(1)) & (rv > 1.5)
d4, c4 = control("trigger only", trig_only)

ok = d1[0] > 1.0 and d1[2] < 0.01
print(f"\n  VERDICT: corrected control {'PASSES → pipeline VALID, atlas-control premise was false' if ok else 'still fails → machinery suspect'}")
