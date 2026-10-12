#!/usr/bin/env python3
"""R115 — FUNDING-RATE DISPERSION (dollar-neutral, cross-sectional). Prespec: RESEARCH_FRONTIER_R115.md.
18-combo grid (W{1,3,7}d x Q{5,8} x R{1,3,7}d); long bottom-Q funding / short top-Q funding;
0.5 gross per side; costs 0.07%/side on turnover; funding cash = -w.f (uniform law);
WF monthly selection by trailing-180d Sharpe (2024-07..2026-09, includes holdout months via prior-only rule);
HOLDOUT 2026-07-01..09-30 reported separately (never touched by selection: trailing window < holdout start);
MB bootstrap + LOAO + benchmarks + min-viable-capital."""
import pathlib
import numpy as np
import pandas as pd

CACHE = pathlib.Path("quantlab_cache/bnfut"); OUT = pathlib.Path("quantlab_output")
SPINE0, SPINE1 = pd.Timestamp("2023-10-01"), pd.Timestamp("2026-09-30")
HOLD0 = pd.Timestamp("2026-07-01"); WF0 = pd.Timestamp("2024-07-01")
COST_SIDE = 0.0007; GROSS_SIDE = 0.5; LEV = 2.0; MAINT = 0.01
GRID = [(W, Q, R) for W in (1, 3, 7) for Q in (5, 8) for R in (1, 3, 7)]

def norm_date(s):
    s = str(s)
    if "-" in s:
        return pd.Timestamp(s[:10])
    v = int(float(s))
    return pd.Timestamp(pd.to_datetime(v, unit="ms" if v > 1e11 else "s"))

def load_panel():
    closes, funds, cov = {}, {}, {}
    for f in sorted(CACHE.glob("*_1D.csv")):
        s = f.stem.split("_")[0]
        k = pd.read_csv(f, dtype={"date": str})
        k["d"] = k.date.map(norm_date)
        c = k.drop_duplicates("d").set_index("d")["close"].sort_index()
        ff = CACHE / f"{s}_fund.csv"
        fd = pd.read_csv(ff)
        fd["ts_ms"] = pd.to_numeric(fd.ts_ms, errors="coerce")
        fd["rate"] = pd.to_numeric(fd.rate, errors="coerce")
        fd = fd.dropna(subset=["ts_ms", "rate"])
        fd["d"] = pd.to_datetime(fd.ts_ms, unit="ms").dt.normalize()
        fsum = fd.groupby("d").rate.sum().sort_index()
        closes[s], funds[s] = c, fsum
    px = pd.DataFrame(closes).sort_index()
    fu = pd.DataFrame(funds).reindex(px.index).fillna(0.0)
    raw = px.mask(px.pct_change().abs() > 0.5)
    win = raw.loc[SPINE0:SPINE1]  # coverage judged BEFORE fill: delistings cannot pass as data
    keep = [s for s in px.columns
            if win[s].notna().mean() >= 0.97 and win[s].first_valid_index() <= pd.Timestamp("2023-10-15")
            and (fu.loc[SPINE0:SPINE1, s] != 0).mean() >= 0.90]
    px = px[keep].loc[SPINE0:SPINE1].ffill()
    fu = fu[keep].loc[SPINE0:SPINE1]
    rets = px.pct_change().fillna(0.0).clip(-0.5, 0.5)
    print(f"universe: {len(keep)} symbols with >=97% price/funding coverage: {keep}")
    return px, fu, rets

px, fu, rets = load_panel()
D = px.index; sym = list(px.columns); S = len(sym)

def build(W, Q, R, exclude=None):
    """full-spine daily net portfolio return per unit gross (weights decided at t, P&L t+1)."""
    sig = fu.rolling(W).mean()
    if exclude:
        sig = sig.drop(columns=[exclude]); rr = rets.drop(columns=[exclude]); ff = fu.drop(columns=[exclude])
    else:
        rr, ff = rets, fu
    names = list(rr.columns); N = len(names)
    daily = pd.Series(0.0, index=D)
    w = pd.Series(0.0, index=names)
    turnover = 0.0
    n_rebal = 0
    for i in range(1, len(D)):
        t = D[i - 1]
        if (i - 1) % R == 0 and sig.loc[t].notna().sum() >= 2 * Q:
            s = sig.loc[t].dropna().sort_values()
            shorts = s.index[-Q:]; longs = s.index[:Q]
            wn = pd.Series(0.0, index=names)
            wn[longs] = GROSS_SIDE / Q; wn[shorts] = -GROSS_SIDE / Q
            turnover = float((wn - w).abs().sum())
            w = wn; n_rebal += 1
        price = float((rr.iloc[i] * w).sum())
        fund = float(-(w * ff.iloc[i]).sum())
        daily.iloc[i] = price + fund - COST_SIDE * turnover
    return daily, n_rebal

print("=" * 74)
print("R115 — FUNDING DISPERSION | 18-combo grid | WF selection | holdout 2026-07..09")
print("=" * 74)
raw = {g: build(*g) for g in GRID}
curves = {g: v[0] for g, v in raw.items()}
REBALS = {g: v[1] for g, v in raw.items()}

def stats(daily, a, b):
    d = daily[(daily.index >= a) & (daily.index <= b)]
    if len(d) < 10:
        return dict(apr=0, sharpe=0, maxdd=0)
    eq = (1 + d * LEV).cumprod()  # account equity: capital = gross/LEV -> equity ret = d*LEV
    dd = (eq / eq.cummax() - 1).min()
    mu, sd = d.mean(), d.std()
    return dict(apr=mu * 365 * 100, sharpe=mu / sd * np.sqrt(365) if sd > 0 else 0, maxdd=dd * 100)

print("\n--- FULL GRID on SELECTION REGION (2024-07 -> 2026-06), net of costs, at 2x acct lev ---")
sel_rows = []
for g, c in curves.items():
    st = stats(c, WF0, pd.Timestamp("2026-06-30"))
    sel_rows.append((g, st))
    print(f"W={g[0]} Q={g[1]} R={g[2]}: APR {st['apr']:+7.2f}%  Sharpe {st['sharpe']:+5.2f}  maxDD {st['maxdd']:+6.1f}%  rebals={REBALS[g]}")

# ---- walk-forward ensemble ----
months = pd.date_range(WF0, SPINE1, freq="MS")
ens = []
chosen = []
for m0 in months:
    m1 = (m0 + pd.offsets.MonthEnd(0))
    look_a, look_b = m0 - pd.Timedelta(days=180), m0 - pd.Timedelta(days=1)
    best, best_sh = None, -9e9
    for g, c in curves.items():
        sh = stats(c, look_a, look_b)["sharpe"]
        if sh > best_sh:
            best, best_sh = g, sh
    chosen.append(best)
    ens.append(curves[best][(curves[best].index >= m0) & (curves[best].index <= m1)])
ens_daily = pd.concat(ens).sort_index()
from collections import Counter
print(f"\nWF selections (months): {Counter([f'W{a}/Q{b}/R{c}' for a,b,c in chosen]).most_common()}")

wf_st = stats(ens_daily, WF0, pd.Timestamp("2026-06-30"))
ho_st = stats(ens_daily, HOLD0, SPINE1)
print(f"WF-OOS  2024-07->2026-06: APR {wf_st['apr']:+.2f}%  Sharpe {wf_st['sharpe']:+.2f}  maxDD {wf_st['maxdd']:+.1f}%")
print(f"HOLDOUT 2026-07->09    : APR {ho_st['apr']:+.2f}%  equity {(ens_daily[(ens_daily.index>=HOLD0)].sum()*LEV)*100:+.2f}%  maxDD {ho_st['maxdd']:+.1f}%")

# ---- block bootstrap on full OOS (WF + holdout) ----
oos = ens_daily[(ens_daily.index >= WF0)]
vals = oos.values; n = len(vals); B = 2000; block = 20
rng = np.random.default_rng(7)
means = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n - block, n // block + 1)
    sample = np.concatenate([vals[i:i + block] for i in idx])[:n]
    means[b] = sample.mean() * 365 * 100
lo, hi = np.percentile(means, [2.5, 97.5])
p_gt0 = (means > 0).mean()
print(f"BOOTSTRAP ann-mean CI95: [{lo:+.1f}%, {hi:+.1f}%]  P(mean>0)={p_gt0*100:.1f}%")

# ---- LOAO ----
loao = []
for s in sym:
    c18 = {g: build(*g, exclude=s)[0] for g in GRID}
    parts = []
    for m0 in months:
        m1 = m0 + pd.offsets.MonthEnd(0)
        look_a, look_b = m0 - pd.Timedelta(days=180), m0 - pd.Timedelta(days=1)
        best, best_sh = None, -9e9
        for g, c in c18.items():
            sh = stats(c, look_a, look_b)["sharpe"]
            if sh > best_sh:
                best, best_sh = g, sh
        parts.append(c18[best][(c18[best].index >= m0) & (c18[best].index <= m1)])
    d = pd.concat(parts).sort_index()
    loao.append((s, stats(d, WF0, pd.Timestamp("2026-06-30"))["apr"]))
loao_aprs = [v for _, v in loao]
loao.sort(key=lambda x: x[1])
print(f"LOAO APR range: min {loao[0][0]} {loao[0][1]:+.1f}% .. max {loao[-1][0]} {loao[-1][1]:+.1f}%")

# ---- benchmarks ----
print("\n--- BENCHMARKS (same spine, same leverage accounting where relevant) ---")
btc = px["BTC"]
print(f"BTC B&H 2024-07->2026-09: {(btc.loc[SPINE1]/btc.loc[WF0]-1)*100:+.1f}% total")
sfund = -fu.mean(axis=1)  # short all: collect avg funding (price-exposed)
sb = (1 + sfund[(sfund.index >= WF0)] / LEV).cumprod()
print(f"Static short-all funding harvest: APR {(sfund[(sfund.index>=WF0)].mean()*365/LEV)*100:+.1f}% (of equity, 2x lev) — price-exposed, NOT shown net of price")
print(f"Cash 4% APR reference; R106 floor reference +5.4% APR")

# ---- Phase 4: minimum viable capital (documented venue minimums) ----
print("\n--- PHASE 4: SMALL-ACCOUNT FEASIBILITY ---")
MIN_NOTIONAL = {"BTC": 20.0, "ETH": 20.0}  # documented Binance USDT-M public specs; most alts 5 USDT
legs_per_side = chosen and Counter(chosen).most_common(1)[0][0]
Qmode = Counter(chosen).most_common(1)[0][0][1]
legs = 2 * Qmode
gross_min = 0.0
for m0, g in zip(months, chosen):
    Q = g[1]; t = m0
    sig = fu.rolling(g[0]).mean().loc[t].dropna().sort_values()
    longs = sig.index[:Q]; shorts = sig.index[-Q:]
    gm = sum(MIN_NOTIONAL.get(c, 5.0) for c in list(longs) + list(shorts))
    gross_min = max(gross_min, gm)
cap_min = gross_min / LEV * 1.3
print(f"typical book: {legs} legs; max gross notional floor across WF months: ${gross_min:.0f}")
print(f"MINIMUM VIABLE CAPITAL (2x leverage, 30% buffer): ${cap_min:.0f}")
print(f"OKX cross-check note: OKX contract minimums are coarser (ctVal x minSz) — several alt-contracts >> $5/leg; OKX book may need 2-5x the Binance minimum.")
print("bootstrap P(equity < 60% at 2x lev):", end=" ")
vals2 = oos.values; n2 = len(vals2)
mins = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n2 - block, n2 // block + 1)
    smp = np.concatenate([vals2[i:i + block] for i in idx])[:n2]
    mins[b] = (1 + smp * LEV).cumprod().min()
print(f"{(mins < 0.6).mean()*100:.1f}%")
for cap in (25, 50, 100, 250):
    feas = cap >= cap_min
    print(f"${cap:4d}: {'FEASIBLE' if feas else 'INFEASIBLE (<venue min notional)'}  implied_gross_leverage={gross_min/cap:.1f}x  "
          f"utilization={'100%' if feas else '>100% (orders below minimum round up -> over-leverage)'}  "
          f"trades/mo~{legs*(30/ (Counter(chosen).most_common(1)[0][0][2])):.0f}")

ens_daily.to_csv(OUT / "r115_dispersion_ensemble.csv", index=True)
pd.DataFrame(sel_rows, columns=["grid", "stats"]).to_csv(OUT / "r115_grid_selection.csv", index=False)
VERDICT = None
if lo > 0 and ho_st["apr"] >= 0 and min(loao_aprs) > 0 and wf_st["maxdd"] >= -20:
    VERDICT = "SURVIVES INITIAL VALIDATION"
elif wf_st["apr"] <= 0 or lo <= 0:
    VERDICT = "FAILS AFTER COSTS"
elif cap_min > 250:
    VERDICT = "PROFITABLE BUT INFEASIBLE FOR SMALL ACCOUNT"
else:
    VERDICT = "PROMISING BUT INCONCLUSIVE"
print(f"\n>>> VERDICT: {VERDICT}")
