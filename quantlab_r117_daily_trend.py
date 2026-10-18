#!/usr/bin/env python3
"""R117 — DAILY-CANDLE TREND ON CRYPTO PERPS (the user-spotted gap). Prespec: CONSISTENCY_HUNT_BEGINS.md.
Signals: DON20/DON55 (long/flip), SMA100/SMA200 (long/flat). Universes: BTC / ETH / EQW-book.
Costs 0.07%/side; funding = actual daily epochs on held longs. Selection 2024-01->2026-03 (Sharpe);
HOLDOUT 2026-07->09 untouched. Gates: charter law. 2020-2023 context printed."""
import pathlib
import numpy as np
import pandas as pd

BC = pathlib.Path("quantlab_cache/bnfut"); OUT = pathlib.Path("quantlab_output")
COST = 0.0007
T0 = pd.Timestamp("2024-01-01"); SEL_END = pd.Timestamp("2026-03-31"); HOLD0 = pd.Timestamp("2026-07-01"); SPINE1 = pd.Timestamp("2026-09-30")
CTX0 = pd.Timestamp("2020-06-01")

def nd(v):
    v = str(v)
    if "-" in v:
        return pd.Timestamp(v[:10])
    x = int(float(v))
    return pd.Timestamp(pd.to_datetime(x, unit="ms" if x > 1e11 else "s").date())

def load():
    px, fu = {}, {}
    for f in sorted(BC.glob("*_1D.csv")):
        s = f.stem.split("_")[0]
        p = pd.read_csv(f, dtype={"date": str})
        px[s] = pd.Series(p.close.values, index=p.date.map(nd))
    P = pd.DataFrame(px).sort_index()
    P = P.mask(P.pct_change().abs() > 0.5)
    win = P.loc[T0:SPINE1]
    keep = [s for s in P.columns
            if win[s].notna().mean() >= 0.97 and P[s].first_valid_index() <= pd.Timestamp("2023-09-01")]
    P = P[keep]
    for s in keep:
        fd = pd.read_csv(BC / f"{s}_fund.csv")
        fd["ts_ms"] = pd.to_numeric(fd.ts_ms, errors="coerce"); fd["rate"] = pd.to_numeric(fd.rate, errors="coerce")
        fd = fd.dropna()
        fd["d"] = pd.to_datetime(fd.ts_ms, unit="ms").dt.normalize()
        fu[s] = fd.groupby("d").rate.sum()
    F = pd.DataFrame(fu).reindex(P.index).fillna(0.0)
    idx = P.loc[CTX0:SPINE1]
    P, F = P.loc[CTX0:SPINE1].ffill(), F.loc[idx.index]
    rets = P.pct_change().fillna(0.0).clip(-0.5, 0.5)
    print(f"universe: {len(keep)} coins (>=97% coverage, listed by 2023-09): {keep}")
    return P, F, rets, keep

P, F, RETS, UNI = load()
D = P.index

def signal_matrix(sig):
    """pos decided at close t (uses info <= t), applied to t+1 return."""
    return sig

def run(sig_name):
    if sig_name.startswith("DON"):
        N = int(sig_name[3:])
        hh = P.rolling(N).max().shift(1); ll = P.rolling(N).min().shift(1)
        pos = pd.DataFrame(0.0, index=D, columns=P.columns)
        pos[P > hh] = 1.0
        pos[P < ll] = -1.0     # flip variant
        pos = pos.replace(0.0, np.nan).ffill().fillna(0.0)
    else:
        N = int(sig_name[3:])
        sma = P.rolling(N).mean()
        pos = (P > sma).astype(float)  # long/flat
    gross = pd.DataFrame(0.0, index=D, columns=P.columns)
    cost = pd.DataFrame(0.0, index=D, columns=P.columns)
    pos_lag = pos.shift(1).fillna(0.0)
    gross = pos_lag * (RETS - F)          # price + funding paid by longs (f>0 => longs pay)
    cost = pos.diff().abs().fillna(0.0) * COST
    net = gross - cost
    return pos, net

def curve(net, universe):
    if universe == "BTC":
        return net["BTC"]
    if universe == "ETH":
        return net["ETH"]
    cols = [c for c in net.columns]
    return net[cols].mean(axis=1)

def stats(d):
    d = d[d.index >= T0]
    if len(d) < 30:
        return dict(apr=0, sharpe=0, maxdd=0, posW=0, worstW=0, yr={})
    wk = d.resample("W").sum()
    eq = (1 + d).cumprod()
    yr = d.groupby(d.index.year).sum()
    return dict(apr=d.mean() * 365 * 100, sharpe=d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else 0,
                maxdd=(eq / eq.cummax() - 1).min() * 100, posW=(wk > 0).mean() * 100, worstW=wk.min() * 100,
                yr=yr.to_dict())

print("=" * 74)
print("R117 — DAILY-CANDLE TREND, CRYPTO PERPS | 4 signals x 3 universes | no leverage")
print("=" * 74)
variants = {}
for sig in ("DON20", "DON55", "SMA100", "SMA200"):
    pos, net = run(sig)
    for uni in ("BTC", "ETH", "BOOK"):
        variants[(sig, uni)] = curve(net, uni)

print("\n--- SELECTION 2024-01 -> 2026-03 (Sharpe picks) ---")
sel_stats = {}
for k, c in variants.items():
    st = stats(c[(c.index >= T0) & (c.index <= SEL_END)])
    sel_stats[k] = st
    yr = st["yr"]
    print(f"{k[0]:7s} {k[1]:4s}: APR {st['apr']:+7.2f}%  Sharpe {st['sharpe']:+5.2f}  maxDD {st['maxdd']:+6.1f}%  "
          f"posW {st['posW']:5.1f}%  worstW {st['worstW']:+6.2f}% | R24/25/26(partial) {yr.get(2024,0):+6.1f}/{yr.get(2025,0):+6.1f}/{yr.get(2026,0):+6.1f}")

best = max(sel_stats, key=lambda k: sel_stats[k]["sharpe"])
print(f"\n>>> SELECTED: {best[0]} on {best[1]}")
c = variants[best]
full = stats(c[c.index >= T0])
print("FULL 2024-01->2026-09:", {k: (round(v, 2) if isinstance(v, (int, float)) else {kk: round(vv, 1) for kk, vv in v.items()}) for k, v in full.items()})
gates = dict(y2024=full["yr"].get(2024, 0) >= 0, y2025=full["yr"].get(2025, 0) >= 0, y2026=full["yr"].get(2026, 0) >= 0,
             maxDD=full["maxdd"] >= -12, worstW=full["worstW"] >= -3, posW=full["posW"] >= 65)
print("GATES:", {k: ("PASS" if v else "FAIL") for k, v in gates.items()}, f"-> {sum(gates.values())}/6")
hold = c[(c.index >= HOLD0)]
print(f"HOLDOUT 2026-07->09: {(hold.sum())*100:+.2f}% equity ({'PASS' if hold.sum() >= 0 else 'FAIL'})")

print("\n--- CONTEXT (not gated): yearly returns incl. the 2022 bear ---")
for k in [best, ("SMA200", "BTC"), ("DON55", "BTC"), ("SMA200", "BOOK")]:
    cc = variants[k]
    yrc = cc[cc.index >= CTX0].groupby(cc[cc.index >= CTX0].index.year).sum()
    print(f"{k[0]:7s} {k[1]:4s}: " + " ".join(f"{y}:{v*100:+7.1f}%" for y, v in yrc.items()))
bh = (P["BTC"].loc[SPINE1] / P["BTC"].loc[T0] - 1) * 100
print(f"BTC B&H 2024->2026-09 for reference: {bh:+.1f}%")

variants[best].to_csv(OUT / "r117_dailytrend_curve.csv")
pd.DataFrame({f"{k[0]}_{k[1]}": v for k, v in variants.items()}).to_csv(OUT / "r117_all_variants.csv")
GATE_PASS = sum(gates.values()) == 6 and hold.sum() >= 0
print("\n>>> VERDICT:", "SURVIVES INITIAL VALIDATION ✅" if GATE_PASS else
      ("FAILS AFTER COSTS ❌" if full["apr"] <= 0 or sum(gates.values()) <= 3 else "PROMISING BUT INCONCLUSIVE 🟡"))
