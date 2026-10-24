#!/usr/bin/env python3
# R122 — COT POSITIONING -> FX DIRECTION (prespec: docs/RESEARCH_FRONTIER_R122.md, LOCKED)
# Cell A (primary): follow hedgers, z156. Cell B (secondary): fade spec extremes.
# Eras: DISCOVERY 2003->2015 (never-touched prices) | CONFIRMATORY 2016->2026 (convention-7 re-test).
import os, sys, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FXD = "quantlab_cache/forex10"
MAP = {"EUR":"EURUSD","GBP":"GBPUSD","JPY":"USDJPY","AUD":"AUDUSD","CAD":"USDCAD","CHF":"USDCHF","NZD":"NZDUSD"}
USD_BASE = {"USDJPY","USDCAD","USDCHF"}          # long "currency" = SHORT the pair
RT = {"EURUSD":0.6,"GBPUSD":0.9,"USDJPY":0.7,"AUDUSD":0.8,"USDCAD":1.0,"USDCHF":1.0,"NZDUSD":1.2}
PIP = {"USDJPY":0.01}
ZWIN, ZMIN = 156, 100
PCTWIN, PCTMIN = 260, 150
CONF_START = pd.Timestamp("2016-01-01")

def daily_closes(pair):
    p1 = pd.read_csv(f"{FXD}/{pair}_1D_2003_2015.csv", parse_dates=["date"]).set_index("date")["close"]
    p2 = pd.read_csv(f"{FXD}/{pair}_1D_2016_2026.csv", parse_dates=["date"]).set_index("date")["close"]
    s = pd.concat([p1, p2[p2.index > p1.index.max()]]).sort_index()
    return s[s > 0]

def cot_frame(cur):
    df = pd.read_csv(f"quantlab_cache/cot/{cur}.csv", parse_dates=["report_date"]).set_index("report_date").sort_index()
    df["comm_net_pct"] = (df.comm_long - df.comm_short) / df.oi
    df["spec_net_pct"] = (df.spec_long - df.spec_short) / df.oi
    return df

def build_cell(cur, cell="A"):
    df, px = cot_frame(cur), daily_closes(MAP[cur])
    idx, cl = px.index, px.values
    if cell == "A":
        z = (df.comm_net_pct - df.comm_net_pct.rolling(ZWIN, ZMIN).mean()) / df.comm_net_pct.rolling(ZWIN, ZMIN).std()
        sign = np.where(z > 0, 1, -1)
        sign = np.where(z.notna().values, sign, 0)
    else:
        pct = df.spec_net_pct.rolling(PCTWIN, PCTMIN).rank(pct=True)
        sign = np.where(pct >= 0.9, -1, np.where(pct <= 0.1, 1, 0))
    rd = df.index
    rows = []
    for i in range(1, len(rd) - 1):
        fri = rd[i] + pd.Timedelta(days=3)          # release Friday (report effective Tuesday rd[i])
        nxt = rd[i + 1] + pd.Timedelta(days=3)
        e = np.searchsorted(idx, fri, side="right") # first close STRICTLY after release
        x = np.searchsorted(idx, nxt, side="right")
        if e >= len(idx) or x >= len(idx) or x <= e: continue
        s = sign[i]
        if s == 0: continue
        direction = -s if MAP[cur] in USD_BASE else s
        r = direction * (cl[x] / cl[e] - 1)
        prev_s = sign[i - 1]
        units = abs(s - prev_s) / 2.0
        cost = units * RT[MAP[cur]] * PIP.get(MAP[cur], 0.0001) / cl[e]
        rows.append((idx[e], r - cost, r, cost))
    return pd.DataFrame(rows, columns=["entry","ret_net","ret_gross","cost"]).set_index("entry")

def pf(x):
    x = np.asarray(x, float); w = x[x>0].sum(); lo = abs(x[x<0].sum())
    return w/lo if lo>0 else (999.0 if w>0 else 1.0)

def portfolio(cell="A"):
    per = {c: build_cell(c, cell) for c in MAP}
    allw = sorted(set().union(*[set(v.index) for v in per.values()]))
    widx = pd.DatetimeIndex([d for d in allw if d.weekday() < 5])
    port, active = [], []
    for d in widx:
        vals = [per[c].ret_net.get(d, np.nan) for c in MAP]
        v = pd.Series(vals).dropna()
        if len(v) >= 3:
            port.append(v.mean()); active.append(len(v))
    return pd.Series(port, index=widx[:len(port)]), per, active

def era_stats(eq, loo_per):
    disc = eq[eq.index < CONF_START]; conf = eq[eq.index >= CONF_START]
    out = {}
    for tag, e in [("DISCOVERY", disc), ("CONFIRM", conf)]:
        if len(e) < 10: out[tag] = dict(n=0); continue
        sh = e.mean()/e.std()*np.sqrt(52) if e.std()>0 else 0
        g = (1+e).cumprod(); dd = (g/g.cummax()-1).min()*100
        rng = np.random.default_rng(122)
        bootp = float(np.mean([rng.choice(e.values, len(e), replace=True).sum() > 0 for _ in range(5000)]))
        out[tag] = dict(n=len(e), pf=pf(e.values), sharpe=sh, bootp=bootp, mdd=dd, total=(g.iloc[-1]-1)*100)
    loo = []
    for cdrop in loo_per:
        vals = []
        for c in loo_per:
            if c == cdrop: continue
            vals.append(loo_per[c].ret_net)
        m = pd.concat(vals, axis=1).dropna(how="all").mean(axis=1)
        loo.append(pf(m[m.index < CONF_START].values))
    out["loo_min"] = min(loo)
    return out

for cell, tag in [("A","PRIMARY: follow-hedgers z156"), ("B","SECONDARY: fade-spec-extremes")]:
    eq, per, active = portfolio(cell)
    st = era_stats(eq, per)
    d, c = st["DISCOVERY"], st.get("CONFIRM", {})
    print("="*96)
    print(f"  R122 CELL {cell} — {tag}")
    print("="*96)
    print(f"  DISCOVERY  2003->2015 : n={d.get('n',0):>4}  PF@c={d.get('pf',float('nan')):.2f}  Sharpe={d.get('sharpe',float('nan')):+.2f}  "
          f"bootP={d.get('bootp',float('nan')):.3f}  maxDD={d.get('mdd',float('nan')):+.1f}%  total={d.get('total',float('nan')):+.1f}%")
    print(f"  CONFIRM    2016->2026 : n={c.get('n',0):>4}  PF@c={c.get('pf',float('nan')):.2f}  Sharpe={c.get('sharpe',float('nan')):+.2f}  "
          f"maxDD={c.get('mdd',float('nan')):+.1f}%  total={c.get('total',float('nan')):+.1f}%")
    print(f"  LOO-min (discovery): {st['loo_min']:.2f}   avg pairs active/week: {np.mean(active):.1f}")
    yr = eq[eq.index < CONF_START].groupby(eq[eq.index < CONF_START].index.year).sum()*100
    print("  yearly (discovery): " + " ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))
    eq.to_csv(f"quantlab_output/r122_cell{cell}_weekly.csv", header=["ret"])
    if cell == "A":
        g1, g2, g3, g4, g5 = d.get("pf",0) >= 1.10, d.get("bootp",0) >= 0.95, d.get("sharpe",0) >= 0.5, st["loo_min"] > 0, c.get("pf",0) > 1.0
        print(f"  GATES: G1 PF>=1.10 {'PASS' if g1 else 'FAIL'} | G2 boot>=0.95 {'PASS' if g2 else 'FAIL'} | "
              f"G3 Sharpe>=0.5 {'PASS' if g3 else 'FAIL'} | G4 LOO>0 {'PASS' if g4 else 'FAIL'} | G5 conf PF>1 {'PASS' if g5 else 'FAIL'}")
        print(f"  VERDICT CELL A: " + ("PROMISING FOR FURTHER VALIDATION" if all([g1,g2,g3,g4,g5]) else
              ("INCONCLUSIVE (works old, dead modern)" if all([g1,g2,g3,g4]) and not g5 else "REJECT")))
print("\nsaved: r122_cellA_weekly.csv r122_cellB_weekly.csv")
