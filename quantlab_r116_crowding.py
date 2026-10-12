#!/usr/bin/env python3
"""R116 — H4: OPEN-INTEREST / CROWDED-POSITION DYNAMICS. Prespec: RESEARCH_FRONTIER_R115.md (H4 section).
16-combo grid (fam{squeeze,froth} x W{7,14} x Q{5,8} x R{3,7}); WF monthly by trailing-180d Sharpe;
holdout 2026-07..09 untouched; bootstrap + LOAO + feasibility; funding cash flows on held legs."""
import pathlib
from collections import Counter
import numpy as np
import pandas as pd

MC = pathlib.Path("quantlab_cache/bnmetrics"); BC = pathlib.Path("quantlab_cache/bnfut")
OUT = pathlib.Path("quantlab_output")
SYMS = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LINK", "LTC", "AVAX",
        "DOT", "NEAR", "APT", "SUI", "ATOM", "FIL", "INJ", "ARB", "OP", "ETC",
        "TRX", "XLM", "CRV", "AAVE"]
SPINE0, SPINE1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-09-30")
WF0 = pd.Timestamp("2024-07-01"); HOLD0 = pd.Timestamp("2026-07-01")
COST_SIDE = 0.0007; GROSS_SIDE = 0.5; LEV = 2.0
GRID = [(fam, W, Q, R) for fam in ("A", "B") for W in (7, 14) for Q in (5, 8) for R in (3, 7)]

def load():
    ms, px = {}, {}
    for s in SYMS:
        m = pd.read_csv(MC / f"{s}_metrics.csv", parse_dates=["date"]).set_index("date")
        ms[s] = m
        p = pd.read_csv(BC / f"{s}_1D.csv", dtype={"date": str})
        def nd(v):
            v = str(v)
            if "-" in v:
                return pd.Timestamp(v[:10])
            x = int(float(v))
            return pd.Timestamp(pd.to_datetime(x, unit="ms" if x > 1e11 else "s").date())
        d = p.date.map(nd)
        px[s] = pd.Series(p.close.values, index=d)
    P = pd.DataFrame(px).sort_index()
    P = P.mask(P.pct_change().abs() > 0.5)
    win = P.loc[SPINE0:SPINE1]
    keep = [s for s in SYMS if win[s].notna().mean() >= 0.97]
    P = P[keep].loc[SPINE0:SPINE1].ffill()
    ls = pd.DataFrame({s: ms[s].ls_top for s in keep}).reindex(P.index)
    oi = pd.DataFrame({s: ms[s].oi_usd for s in keep}).reindex(P.index)
    f = {}
    for s in keep:
        fd = pd.read_csv(BC / f"{s}_fund.csv")
        fd["ts_ms"] = pd.to_numeric(fd.ts_ms, errors="coerce"); fd["rate"] = pd.to_numeric(fd.rate, errors="coerce")
        fd = fd.dropna(subset=["ts_ms", "rate"])
        fd["d"] = pd.to_datetime(fd.ts_ms, unit="ms").dt.normalize()
        f[s] = fd.groupby("d").rate.sum()
    F = pd.DataFrame(f).reindex(P.index).fillna(0.0)
    cov = (ls.notna().mean() >= 0.97) & (oi.notna().mean() >= 0.97)
    keep = [s for s in keep if cov[s]]
    P, ls, oi, F = P[keep], ls[keep], oi[keep], F[keep]
    ls = ls.ffill(); oi = oi.ffill()
    rets = P.pct_change().fillna(0.0).clip(-0.5, 0.5)
    print(f"universe: {len(keep)} coins with >=97% coverage on metrics+price: {keep}")
    return P, ls, oi, F, rets

P, LS, OI, F, RETS = load()
D = P.index

def sig_matrix(fam, W):
    if fam == "A":
        return LS.rolling(W).mean()
    return OI.pct_change().rolling(W).mean()

def build(fam, W, Q, R, exclude=None):
    sig = sig_matrix(fam, W)
    rr, ff = RETS, F
    if exclude:
        keep = [s for s in sig.columns if s != exclude]
        sig, rr, ff = sig[keep], rr[keep], ff[keep]
    names = list(sig.columns)
    daily = pd.Series(0.0, index=D)
    w = pd.Series(0.0, index=names); turnover = 0.0
    for i in range(1, len(D)):
        t = D[i - 1]
        if (i - 1) % R == 0 and sig.loc[t].notna().sum() >= 2 * Q:
            s = sig.loc[t].dropna().sort_values()
            wn = pd.Series(0.0, index=names)
            wn[s.index[:Q]] = GROSS_SIDE / Q     # long least-crowded/deleveraging
            wn[s.index[-Q:]] = -GROSS_SIDE / Q   # short most-crowded/frothy
            turnover = float((wn - w).abs().sum()); w = wn
        daily.iloc[i] = float((rr.iloc[i] * w).sum()) + float(-(w * ff.iloc[i]).sum()) - COST_SIDE * turnover
    return daily

def stats(daily, a, b):
    d = daily[(daily.index >= a) & (daily.index <= b)]
    if len(d) < 10:
        return dict(apr=0.0, sharpe=0.0, maxdd=0.0)
    eq = (1 + d * LEV).cumprod()
    mu, sd = d.mean(), d.std()
    return dict(apr=mu * 365 * 100, sharpe=mu / sd * np.sqrt(365) if sd > 0 else 0,
                maxdd=(eq / eq.cummax() - 1).min() * 100)

print("=" * 74)
print("R116 — H4 CROWDING | 16 combos | WF selection | holdout 2026-07..09 untouched")
print("=" * 74)
curves = {}
for fam, W, Q, R in GRID:
    curves[(fam, W, Q, R)] = build(fam, W, Q, R)
print("\n--- GRID on SELECTION REGION 2024-07 -> 2026-06 (net, 2x lev) ---")
for g, c in curves.items():
    st = stats(c, WF0, pd.Timestamp("2026-06-30"))
    print(f"{g[0]} W={g[1]:2d} Q={g[2]} R={g[3]}: APR {st['apr']:+7.2f}%  Sharpe {st['sharpe']:+5.2f}  maxDD {st['maxdd']:+6.1f}%")

months = pd.date_range(WF0, SPINE1, freq="MS")
ens = []; chosen = []
for m0 in months:
    m1 = m0 + pd.offsets.MonthEnd(0)
    la, lb = m0 - pd.Timedelta(days=180), m0 - pd.Timedelta(days=1)
    best, bs = None, -9e9
    for g, c in curves.items():
        sh = stats(c, la, lb)["sharpe"]
        if sh > bs:
            best, bs = g, sh
    chosen.append(best)
    ens.append(curves[best][(curves[best].index >= m0) & (curves[best].index <= m1)])
ens_daily = pd.concat(ens).sort_index()
print(f"\nWF picks: {Counter([f'{a}/W{b}/Q{c}/R{d}' for a,b,c,d in chosen]).most_common()}")

wf = stats(ens_daily, WF0, pd.Timestamp("2026-06-30"))
ho = stats(ens_daily, HOLD0, SPINE1)
print(f"WF-OOS 2024-07->2026-06: APR {wf['apr']:+.2f}%  Sharpe {wf['sharpe']:+.2f}  maxDD {wf['maxdd']:+.1f}%")
print(f"HOLDOUT 2026-07->09    : APR {ho['apr']:+.2f}%  equity {(ens_daily[ens_daily.index>=HOLD0].sum()*LEV)*100:+.2f}%  maxDD {ho['maxdd']:+.1f}%")

oos = ens_daily[ens_daily.index >= WF0]
vals = oos.values; n = len(vals); B = 2000; block = 20
rng = np.random.default_rng(11)
means = np.empty(B); mins = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n - block, n // block + 1)
    smp = np.concatenate([vals[i:i + block] for i in idx])[:n]
    means[b] = smp.mean() * 365 * 100
    mins[b] = (1 + smp * LEV).cumprod().min()
lo, hi = np.percentile(means, [2.5, 97.5])
print(f"BOOTSTRAP CI95: [{lo:+.1f}%, {hi:+.1f}%]  P(mean>0)={(means>0).mean()*100:.1f}%  P(equity<60%)={(mins<0.6).mean()*100:.1f}%")

loao = []
for s in P.columns:
    c16 = {g: build(*g, exclude=s) for g in GRID}
    parts = []
    for m0 in months:
        m1 = m0 + pd.offsets.MonthEnd(0)
        la, lb = m0 - pd.Timedelta(days=180), m0 - pd.Timedelta(days=1)
        best, bs = None, -9e9
        for g, c in c16.items():
            sh = stats(c, la, lb)["sharpe"]
            if sh > bs:
                best, bs = g, sh
        parts.append(c16[best][(c16[best].index >= m0) & (c16[best].index <= m1)])
    d = pd.concat(parts).sort_index()
    loao.append((s, stats(d, WF0, pd.Timestamp("2026-06-30"))["apr"]))
loao_aprs = [v for _, v in loao]; loao.sort(key=lambda x: x[1])
print(f"LOAO APR: min {loao[0][0]} {loao[0][1]:+.1f}% .. max {loao[-1][0]} {loao[-1][1]:+.1f}%")

print("\n--- BENCHMARKS ---")
print(f"BTC B&H 2024-07->2026-09: {(P.BTC.loc[SPINE1]/P.BTC.loc[WF0]-1)*100:+.1f}% total")
print(f"cash 4% APR ref · R106 floor +5.4% APR ref · R115 funding-dispersion WF +35.5% (holdout -34%)")

print("\n--- PHASE 4: FEASIBILITY ---")
MINN = {"BTC": 20.0, "ETH": 20.0}
Qmode = Counter(chosen).most_common(1)[0][0][2]
Rmode = Counter(chosen).most_common(1)[0][0][3]
gross_min = 0.0
for m0, g in zip(months, chosen):
    sig = sig_matrix(g[0], g[1]).loc[m0].dropna().sort_values()
    gm = sum(MINN.get(c, 5.0) for c in list(sig.index[:g[2]]) + list(sig.index[-g[2]:]))
    gross_min = max(gross_min, gm)
cap_min = gross_min / LEV * 1.3
print(f"book legs={2*Qmode}  gross-notional floor ${gross_min:.0f}  MIN VIABLE CAPITAL ${cap_min:.0f}  trades/mo~{2*Qmode*30/Rmode:.0f}")
for cap in (25, 50, 100, 250):
    print(f"${cap:4d}: {'FEASIBLE' if cap >= cap_min else 'INFEASIBLE FOR SMALL ACCOUNT'} (implied gross lev {gross_min/cap:.1f}x)")

ens_daily.to_csv(OUT / "r116_crowding_ensemble.csv", index=True)
if lo > 0 and ho["apr"] >= 0 and min(loao_aprs) > 0 and wf["maxdd"] >= -20:
    V = "SURVIVES INITIAL VALIDATION"
elif wf["apr"] <= 0 or lo <= 0:
    V = "FAILS AFTER COSTS"
elif cap_min > 250:
    V = "PROFITABLE BUT INFEASIBLE FOR SMALL ACCOUNT"
else:
    V = "PROMISING BUT INCONCLUSIVE"
print(f"\n>>> VERDICT: {V}")
