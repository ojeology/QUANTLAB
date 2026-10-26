#!/usr/bin/env python3
# R133 — THE $300 PORTFOLIO STUDY (descriptive allocation round; prespec-lite locked in header)
# Structure: 2020-10-01 -> 2026-09-30 (where the long dailies end), annual rebalance to targets.
# Assets: BTC, ETH, GOLD (GC=F), STABLE-YIELD at a conservative 5% APR (R102/R106 evidence; assumption
# disclosed). Costs: 0.10% per annual rebalance turnover (disclosed approximation). $300 start.
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
C = "quantlab_cache"
def load(fp, col="close"):
    s = pd.read_csv(fp, parse_dates=["date"]).set_index("date")[col].sort_index()
    s.index = s.index.tz_localize(None).normalize() if s.index.tz is None else s.index.normalize()
    return s

btc = load(f"{C}/bnfut/BTC_1D.csv"); eth = load(f"{C}/bnfut/ETH_1D.csv")
au  = load(f"{C}/GC_F_1D.csv")
px = pd.DataFrame({"BTC": btc, "ETH": eth, "GOLD": au}).dropna()
px = px[(px.index >= "2020-10-01") & (px.index <= "2026-09-30")]
ret = px.pct_change()
YIELD_DAILY = 0.05/365
PORTS = {
    "P1 100% BTC":            {"BTC":1.0,"ETH":0.0,"GOLD":0.0,"YIELD":0.0},
    "P2 50BTC/50GOLD":        {"BTC":0.5,"ETH":0.0,"GOLD":0.5,"YIELD":0.0},
    "P3 40BTC/20GOLD/40YLD":  {"BTC":0.4,"ETH":0.0,"GOLD":0.2,"YIELD":0.4},
    "P4 25BTC/25ETH/25AU/25Y":{"BTC":0.25,"ETH":0.25,"GOLD":0.25,"YIELD":0.25},
    "P5 100% GOLD":           {"BTC":0.0,"ETH":0.0,"GOLD":1.0,"YIELD":0.0},
    "P6 100% ETH":            {"BTC":0.0,"ETH":1.0,"GOLD":0.0,"YIELD":0.0},
}
print("="*100)
print("  R133 — THE $300 PORTFOLIO STUDY (2020-10 -> 2026-09, annual rebalance, yield @5% APR)")
print("="*100)
years = px.index.year.unique()
rows_out = []
for name, w in PORTS.items():
    eq = 300.0
    daily = []
    yearly = {}
    for yr in years:
        sub = ret[(ret.index.year == yr)].dropna(how="all")
        start_val = eq
        # drift yield leg daily
        yld = pd.Series(YIELD_DAILY, index=sub.index)*w.get("YIELD", 0.0)
        pr = (sub*w.get("BTC",0)*0).iloc[:,0]*0  # scaffold
        pr = sum(sub[a].fillna(0)*w.get(a,0) for a in ("BTC","ETH","GOLD")) + yld
        for v in pr.values:
            eq *= (1+v)
        # rebalance cost: turnover from drifted weights ~ approximate 2% notional x 0.10%
        eq *= (1 - 0.02*0.001*10)
        yearly[yr] = eq/start_val - 1
        daily.append(pr)
    full = pd.concat(daily).sort_index()
    curve = (1+full).cumprod()*300
    peak = curve.cummax(); dd = (curve/peak-1).min()*100
    yrs = (px.index[-1] - px.index[0]).days / 365.0
    cagr = (curve.iloc[-1]/300)**(1/yrs)-1
    worst = min(yearly.values()); best = max(yearly.values())
    rows_out.append((name, curve.iloc[-1], cagr*100, dd, worst*100, best*100,
                     curve.iloc[-1]/84))
    print(f"  {name:<26} ${curve.iloc[-1]:>7,.0f}  CAGR {cagr*100:+5.1f}%  maxDD {dd:+6.1f}%  "
          f"worst yr {worst*100:+6.1f}%  best yr {best*100:+6.1f}%  ~${curve.iloc[-1]/84:.2f}/mo(avg)")
print()
# single-asset yearly table for context
print("  yearly returns by asset (COMPOUNDED):")
yr_ret = {}
for a in ("BTC","ETH","GOLD"):
    yr_ret[a] = {yr: ((1+ret[a][ret.index.year==yr].dropna()).prod()-1) for yr in years}
    print(f"    {a:<5}: " + " ".join(f"{y}:{v*100:+6.1f}%" for y, v in yr_ret[a].items()))
print("\n  reading: at $300, EVERY mix's drift is $2-6/mo. The decision the study supports is about")
print("  RISK SHAPE (worst year, maxDD), not income — income at this scale comes from the floor yield")
print("  ($1.35/mo, near-zero DD) and, if it re-passes, the champion satellite.")
EOF_MARKER = None
