#!/usr/bin/env python3
# R129 — THE FLOW LAYER (prespec: docs/RESEARCH_FRONTIER_R129.md, LOCKED)
import json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BOOK = "quantlab_output/r096_trades_q055.csv"

def pf(x):
    x = np.asarray(x, float); w = x[x>0].sum(); lo = abs(x[x<0].sum())
    return w/lo if lo>0 else 999.0

# ── flow series ──────────────────────────────────────────────────────────────
st = json.load(open("/tmp/stbl.json"))
s_dates = pd.to_datetime([int(d["date"]) for d in st], unit="s").normalize()
s_vals = pd.Series([float(d["totalCirculating"]["peggedUSD"]) for d in st], index=s_dates).sort_index()
liq = (s_vals / s_vals.shift(30) - 1)                       # 30d % change

fng = json.load(open("/tmp/fng.json"))["data"]
f_dates = pd.to_datetime([int(x["timestamp"]) for x in fng], unit="s").normalize()
f_vals = pd.Series([float(x["value"]) for x in fng], index=f_dates).sort_index()
print(f"flow series: stablecoins {s_vals.index.min().date()}→{s_vals.index.max().date()} "
      f"({len(s_vals)}d) | F&G {f_vals.index.min().date()}→{f_vals.index.max().date()} ({len(f_vals)}d)")

def feat_at(series, date, minobs=90):
    """Causal: feature value at date-1 with expanding median from series start."""
    d = date - pd.Timedelta(days=1)
    hist = series[series.index <= d].dropna()
    if len(hist) < minobs: return np.nan, np.nan
    val = float(hist.iloc[-1])
    med = float(hist.median())
    return val, med

# ── champion kept book ───────────────────────────────────────────────────────
t = pd.read_csv(BOOK, parse_dates=["entry_time"])
t["rc"] = t.r - 2.0*(0.05/100.0)*t.entry/t.atr
t["date"] = t.entry_time.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
print(f"champion kept book: n={len(t)}  PF@c={pf(t.rc):.2f}  {t.date.min().date()}→{t.date.max().date()}")

PRIMARY = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-27"))
CONSIST = (pd.Timestamp("2025-06-01"), pd.Timestamp("2026-01-01"))

def evaluate(cellname, passes_fn):
    keep = []
    for _, row in t.iterrows():
        val, med = feat_at(SRC[cellname], row.date)
        keep.append(np.nan if np.isnan(val) else (1.0 if passes_fn(val, med) else 0.0))
    t["k"] = keep
    g = t[t.k == 1]
    prim = g[(g.date >= PRIMARY[0]) & (g.date < PRIMARY[1])]
    full_prim = t[(t.date >= PRIMARY[0]) & (t.date < PRIMARY[1])]
    cons = g[(g.date >= CONSIST[0]) & (g.date < CONSIST[1])]
    full_cons = t[(t.date >= CONSIST[0]) & (t.date < CONSIST[1])]
    g1 = pf(prim.rc) > pf(full_prim.rc)
    g2 = len(prim) >= 0.55 * len(full_prim)
    g3 = (len(cons) >= 10) and pf(cons.rc) > pf(full_cons.rc)
    g4 = pf(g.rc) > pf(t.rc)
    print(f"\n  CELL {cellname}:")
    print(f"    PRIMARY 2026: gated n={len(prim):>3}/{len(full_prim)} ({len(prim)/len(full_prim)*100:.0f}%)  "
          f"PF@c={pf(prim.rc):.2f} vs full {pf(full_prim.rc):.2f}  -> G1 {'PASS' if g1 else 'FAIL'} | G2 {'PASS' if g2 else 'FAIL'}")
    print(f"    CONSIST 2025: gated n={len(cons):>3}/{len(full_cons)}  PF@c={pf(cons.rc) if len(cons) else float('nan'):.2f} "
          f"vs full {pf(full_cons.rc):.2f}  -> G3 {'PASS' if g3 else 'FAIL'}")
    print(f"    FULL book: gated n={len(g):>3}  PF@c={pf(g.rc):.2f} vs incumbent {pf(t.rc):.2f}  -> G4 {'PASS' if g4 else 'FAIL'}")
    m = g.groupby(pd.Grouper(key="date", freq="MS")).rc.agg(["size","sum"])
    print("    gated monthly R: " + " ".join(f"{i.strftime('%y-%m')}:{r:+.1f}(n{n})" for i, r, n in m.itertuples(index=True) if False) or "")
    print("    " + "  ".join(f"{idx.strftime('%y%m')}:{row['sum']:+.1f}/{int(row['size'])}" for idx, row in m.iterrows()))
    ok = all([g1, g2, g3, g4])
    print(f"    VERDICT: {'PROMISING AS FLOW OVERLAY' if ok else 'CLOSED'}")
    return ok

SRC = {"A": liq, "C": f_vals}
okA = evaluate("A", lambda v, med: v > med)                       # liquidity expanding -> trade
okC = evaluate("C", lambda v, med: v <= 75)                       # not-euphoric -> trade

# descriptive interaction (report-only)
ka, kc = [], []
for _, row in t.iterrows():
    va, ma = feat_at(liq, row.date); vc, _ = feat_at(f_vals, row.date)
    ka.append(np.nan if np.isnan(va) else (1.0 if va > ma else 0.0))
    kc.append(np.nan if np.isnan(vc) else (1.0 if vc <= 75 else 0.0))
t["ka"], t["kc"] = ka, kc
both = t[(t.ka == 1) & (t.kc == 1)]
print(f"\n  report-only A×C interaction: trades passing BOTH: n={len(both)}  PF@c={pf(both.rc):.2f} "
      f"({len(both)/len(t)*100:.0f}% retention)")

print("\n" + "="*90)
verdict = []
if okA: verdict.append("A: PROMISING (challenger)")
if okC: verdict.append("C: PROMISING (challenger)")
print("  FLOW LAYER VERDICT: " + ("; ".join(verdict) if verdict else
      "CLOSED — no overlay passed; last reachable new-information axis mapped"))
print("="*90)
t.to_csv("quantlab_output/r129_flow_book.csv", index=False)
print("saved: r129_flow_book.csv")
