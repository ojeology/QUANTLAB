#!/usr/bin/env python3
# R132 — THE SEVEN FRONTIERS (prespec: docs/RESEARCH_FRONTIER_R132.md, LOCKED)
# Cells: S intraday seasonality · M1 DXY-liquidity gate (M2 real-yield: FRED blocked, disclosed) ·
# X BTC-transmission gate. Overlay template per R129. Challenger-grade, forward-only promotion.
import os, sys, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features

CACHE = CONFIG["CACHE_FOLDER"]
SPLIT = pd.Timestamp("2025-10-01", tz="UTC")
SKIP = ("XAU","XAG","CL","SOXL","MU","SPCX","AAOI","UB","SNDK","SKHY","BEAT","ALLO","ZBT","SNXX","PUMP","HOME")

def pf(x):
    x = np.asarray(x, float); w = x[x>0].sum(); lo = abs(x[x<0].sum())
    return w/lo if lo>0 else 999.0

def sharpe(x, per=24):
    x = np.asarray(x, float)
    return x.mean()/x.std()*np.sqrt(365*per) if len(x) > 10 and x.std() > 0 else 0.0

def gates(g, prim, full_prim, cons, full_cons, full):
    g1 = pf(prim.rc) > pf(full_prim.rc)
    g2 = len(prim) >= 0.55*len(full_prim)
    g3 = len(cons) >= 10 and pf(cons.rc) > pf(full_cons.rc)
    g4 = pf(g.rc) > pf(full.rc)
    return g1, g2, g3, g4

print("="*100)
print("  R132 — THE SEVEN FRONTIERS (cells S, M1, X; M2 real-yield = FRED blocked, disclosed)")
print("="*100)

# ── CELL S: intraday seasonality returns ─────────────────────────────────────
print("\n── CELL S — INTRADAY SEASONALITY (73-coin 1H returns by UTC hour) ──")
hour_rets = {h: {"DISC": [], "CONF": []} for h in range(24)}
dow_rets = {d: {"DISC": [], "CONF": []} for d in range(7)}
for f in sorted(glob.glob(os.path.join(CACHE, "*_USDT_SWAP_1H.parquet"))):
    sym = os.path.basename(f).replace("_USDT_SWAP_1H.parquet", "")
    if sym.startswith(SKIP): continue
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True)
    ret = df["close"].pct_change()
    if len(ret) < 1000: continue
    era = pd.Series(np.where(df.index < SPLIT, "DISC", "CONF"), index=df.index)
    hh = pd.Series(df.index.hour, index=df.index)
    for h in range(24):
        m = hh == h
        for e in ("DISC","CONF"):
            hour_rets[h][e].append(ret[m & (era == e)].dropna().values)
    dd = pd.Series(df.index.dayofweek, index=df.index)
    dret = df["close"].resample("1D").last().pct_change().dropna()
    dera = pd.Series(np.where(dret.index.tz_convert("UTC") < SPLIT, "DISC", "CONF"), index=dret.index)
    for d in range(7):
        m = dret.index.dayofweek == d
        for e in ("DISC","CONF"):
            dow_rets[d][e].append(dret[m & (dera == e)].dropna().values)

rows = []
for h in range(24):
    a = np.concatenate(hour_rets[h]["DISC"]) if hour_rets[h]["DISC"] else np.array([])
    b = np.concatenate(hour_rets[h]["CONF"]) if hour_rets[h]["CONF"] else np.array([])
    rows.append((h, a.mean()*100, b.mean()*100, len(a), len(b)))
H = pd.DataFrame(rows, columns=["hour","disc","conf","n_d","n_c"]).set_index("hour")
print("  hourly mean returns % (DISC | CONF):")
for h, r in H.iterrows():
    print(f"    {h:>2}h: {r.disc:+.4f} | {r.conf:+.4f}   (n={int(r.n_d):,}/{int(r.n_c):,})")
best_d = H.disc.idxmax(); worst_d = H.disc.idxmin()
print(f"  best DISC hour: {best_d}h ({H.disc.max():+.4f}%)  worst: {worst_d}h ({H.disc.min():+.4f}%)")
# strategy form (i): long the best DISC hour each day; avoid-hour variant is flat during worst (no shorting cost)
COST_H = 0.0014  # RT per daily round trip
def hour_strategy(hsel):
    out = {"DISC": [], "CONF": []}
    for f in sorted(glob.glob(os.path.join(CACHE, "*_USDT_SWAP_1H.parquet")))[:20]:  # 20-coin subsample for cost realism
        sym = os.path.basename(f).replace("_USDT_SWAP_1H.parquet", "")
        if sym.startswith(SKIP): continue
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index, utc=True)
        ret = df["close"].pct_change()
        hh = pd.Series(df.index.hour, index=df.index)
        era = pd.Series(np.where(df.index < SPLIT, "DISC", "CONF"), index=df.index)
        pos = (hh == hsel).astype(float)
        strat = pos*ret - pos.diff().abs().fillna(0)*COST_H
        for e in ("DISC","CONF"):
            out[e].append(strat[(era == e)].dropna().values)
    return {e: np.concatenate(v) for e, v in out.items()}
st_best = hour_strategy(int(best_d))
sh_d = sharpe(st_best["DISC"]); sh_c = sharpe(st_best["CONF"])
print(f"  strategy 'long best-DISC-hour daily': DISC Sharpe {sh_d:+.2f} | CONF Sharpe {sh_c:+.2f} "
      f"-> {'PASS' if (sh_d >= 0.5 and sh_c > 0) else 'DEAD'}")
# DOW daily
rows = []
for d in range(7):
    a = np.concatenate(dow_rets[d]["DISC"]) if dow_rets[d]["DISC"] else np.array([])
    b = np.concatenate(dow_rets[d]["CONF"]) if dow_rets[d]["CONF"] else np.array([])
    rows.append((d, a.mean()*100, b.mean()*100))
D = pd.DataFrame(rows, columns=["dow","disc","conf"]).set_index("dow")
print("  DOW mean daily %: " + "  ".join(f"{['Mo','Tu','We','Th','Fr','Sa','Su'][d]}:{v.disc:+.3f}/{v.conf:+.3f}" for d, v in D.iterrows()))
bd = D.disc.idxmax()
sh_d2 = D.loc[bd,"disc"]/np.nanstd([v.disc for _, v in D.iterrows()]) if True else 0
print(f"  best DOW ({['Mo','Tu','We','Th','Fr','Sa','Su'][bd]}): DISC {D.loc[bd,'disc']:+.3f}% vs CONF {D.loc[bd,'conf']:+.3f}% "
      f"-> {'same sign' if np.sign(D.loc[bd,'disc'])==np.sign(D.loc[bd,'conf']) else 'SIGN FLIP — dead'}")
print("  CELL S VERDICT: " + ("PROMISING" if (sh_d >= 0.5 and sh_c > 0) else "CLOSED (hours/days carry no net edge after costs)"))

# ── champion book + gates for M1 and X ───────────────────────────────────────
t = pd.read_csv("quantlab_output/r096_trades_q055.csv", parse_dates=["entry_time"])
t["rc"] = t.r - 2.0*(0.05/100.0)*t.entry/t.atr
t["date"] = t.entry_time.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
PRIM = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-27"))
CONS = (pd.Timestamp("2025-06-01"), pd.Timestamp("2026-01-01"))

def evaluate(name, keep_series):
    t["k"] = keep_series
    g = t[t.k == 1]
    prim = g[(g.date >= PRIM[0]) & (g.date < PRIM[1])]
    fp = t[(t.date >= PRIM[0]) & (t.date < PRIM[1])]
    cons = g[(g.date >= CONS[0]) & (g.date < CONS[1])]
    fc = t[(t.date >= CONS[0]) & (t.date < CONS[1])]
    g1, g2, g3, g4 = gates(g, prim, fp, cons, fc, t)
    print(f"\n── CELL {name} ──")
    print(f"  PRIMARY 2026: gated n={len(prim)}/{len(fp)} ({len(prim)/max(len(fp),1)*100:.0f}%)  PF@c {pf(prim.rc):.2f} vs full {pf(fp.rc):.2f}  G1 {'PASS' if g1 else 'FAIL'} G2 {'PASS' if g2 else 'FAIL'}")
    print(f"  CONSIST 2025: gated n={len(cons)}/{len(fc)}  PF@c {pf(cons.rc) if len(cons) else float('nan'):.2f} vs {pf(fc.rc):.2f}  G3 {'PASS' if g3 else 'FAIL'}")
    print(f"  FULL book: gated n={len(g)}  PF@c {pf(g.rc):.2f} vs {pf(t.rc):.2f}  G4 {'PASS' if g4 else 'FAIL'}")
    ok = all([g1,g2,g3,g4])
    print(f"  VERDICT: {'PROMISING AS OVERLAY' if ok else 'CLOSED'}")
    return ok

# ── CELL M1: DXY 30d-change gate ─────────────────────────────────────────────
print("\n── CELL M1 — DOLLAR-LIQUIDITY GATE (DXY 30d change < expanding median -> trade) ──")
dxy = pd.read_csv(f"{CACHE}/DXY_1D.csv", parse_dates=["date"]).set_index("date")["close"]
d30 = (dxy/dxy.shift(30) - 1)
med = d30.expanding(min_periods=90).median()
def dxy_soft(date):
    d = date - pd.Timedelta(days=1)
    h = d30[d30.index <= d].dropna()
    if len(h) < 90: return np.nan
    return 1.0 if h.iloc[-1] < h.median() else 0.0
evaluate("M1", t.date.apply(dxy_soft))

# ── CELL X: BTC-transmission gate ────────────────────────────────────────────
print("\n── CELL X — BTC-TRANSMISSION GATE (skip alt trades entered in a BTC -0.5% hour) ──")
btc = pd.read_parquet(f"{CACHE}/BTC_USDT_SWAP_1H.parquet")
btc.index = pd.to_datetime(btc.index, utc=True)
btc1h = btc["close"].pct_change()
def btc_state(ts):
    v = btc1h.reindex([ts], method="nearest", tolerance=pd.Timedelta("2h"))
    return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else np.nan
bt = t.entry_time.apply(btc_state)
keep = np.where(bt.isna(), np.nan, np.where(bt < -0.005, 0.0, 1.0))
evaluate("X", pd.Series(keep, index=t.index))
print(f"  report-only: entries DURING BTC-dump hour: n={int((bt < -0.005).sum())}  PF@c={pf(t[bt < -0.005].rc):.2f}"
      f" | entries during BTC-up hour (+0.5%): n={int((bt > 0.005).sum())}  PF@c={pf(t[bt > 0.005].rc):.2f}")
print("\nSeven-frontier table finalized in report. saved: r132 book columns kept in-session.")
