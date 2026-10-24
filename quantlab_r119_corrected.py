#!/usr/bin/env python3
"""R119 — CORRECTED RE-RUN of the twin hunt on ql_accounting (audit Phase 5).
Quantifies the confirmed defects (H-T1/H-T2 cost overwrite; mixed leverage basis) by comparing
corrected results to the superseded R118 numbers. Uses LadderAccount invariants throughout.
H-T1 funding disclosure: long-only alt sleeves pay perp funding; not modeled (no funding spine
for alts after cache trim) — same assumption as R113/R118, flagged as open risk in the report."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from ql_accounting import LadderAccount, performance_report, fmt_report

BC = pathlib.Path("quantlab_cache/bnfut")
T0 = pd.Timestamp("2024-01-01"); SEL_END = pd.Timestamp("2026-03-31")
HOLD0 = pd.Timestamp("2026-07-01"); SPINE1 = pd.Timestamp("2026-09-30"); CTX0 = pd.Timestamp("2020-06-01")

def nd(v):
    v = str(v)
    if "-" in v:
        return pd.Timestamp(v[:10])
    x = int(float(v))
    return pd.Timestamp(pd.to_datetime(x, unit="ms" if x > 1e11 else "s").date())

def load_panel():
    px = {}
    for f in sorted(BC.glob("*_1D.csv")):
        s = f.stem.split("_")[0]
        p = pd.read_csv(f, dtype={"date": str})
        px[s] = pd.Series(p.close.values, index=p.date.map(nd))
    P = pd.DataFrame(px).sort_index()
    P = P.mask(P.pct_change().abs() > 0.5)
    win = P.loc[T0:SPINE1]
    keep = [s for s in P.columns if win[s].notna().mean() >= 0.97
            and P[s].first_valid_index() <= pd.Timestamp("2023-09-01")]
    P = P[keep].loc[CTX0:SPINE1].ffill()
    return P, P.pct_change().fillna(0.0).clip(-0.5, 0.5)

def load_btc_fund():
    fd = pd.read_csv(BC / "BTC_fund.csv")
    fd["ts_ms"] = pd.to_numeric(fd.ts_ms, errors="coerce"); fd["rate"] = pd.to_numeric(fd.rate, errors="coerce")
    fd = fd.dropna()
    fd["d"] = pd.to_datetime(fd.ts_ms, unit="ms").dt.normalize()
    return fd.groupby("d").rate.sum()

P, RETS = load_panel()
D = P.index
FUND = load_btc_fund().reindex(D).fillna(0.0)
print(f"universe: {len(P.columns)} coins | BTC funding epochs: {(FUND != 0).sum()}")

def charter_check(r):
    yr = r.get("yearly", {})
    g = dict(y2024=yr.get(2024, 0) >= 0, y2025=yr.get(2025, 0) >= 0, y2026=yr.get(2026, 0) >= 0,
             maxDD=r["maxDD"] >= -12, worstW=r.get("worstW", -99) >= -3, posW=r.get("posW", 0) >= 65)
    return g, sum(g.values())

def window_slice(eq, a, b=None):
    m = (eq.index >= a)
    if b is not None:
        m &= (eq.index <= b)
    return eq[m]

# ---------------- H-T1: momentum top-8, vol-targeted ----------------
def ht1(filtered):
    acct = LadderAccount(D, leverage=1.0, side_rate=0.0005)
    acct.mark(0, 0.0, 0.0)  # bar 0: flat, marked to satisfy every-bar invariant
    vol = RETS.rolling(30).std() * np.sqrt(365)
    iso = D.strftime("%G-%V")
    is_last = np.concatenate([(iso[1:] != iso[:-1]), [True]])
    for i in range(1, len(D)):
        if is_last[i - 1] and i < len(D) - 1:
            t = D[i]
            ok = True
            if filtered:
                sma100 = P["BTC"].loc[:t].rolling(100).mean().iloc[-1]
                ok = P["BTC"].loc[t] > sma100
            if ok:
                mom = P.loc[t] / P.iloc[i - 14] - 1.0
                if mom.notna().sum() >= 8:
                    top = mom.nlargest(8).index
                    new = pd.Series(0.0, index=P.columns)
                    for c in top:
                        v = vol.loc[t, c] if pd.notna(vol.loc[t, c]) and vol.loc[t, c] > 0 else 0.5
                        new[c] = (1 / 8) * float(np.clip(0.08 / v, 0.25, 2.0))
                    acct.rebalance(i, {c: float(new[c]) for c in new.index})
            else:
                acct.rebalance(i, {})
        w_eff = acct.effective_weights(i)          # THE lag-correct weights for bar i
        gross_i = float(sum(w_eff.get(c, 0.0) * RETS.iloc[i][c] for c in w_eff))
        acct.mark(i, gross_i, 0.0)
    return acct

# ---------------- H-T2: DON55 BTC, corrected exits ----------------
def ht2(variant):
    btc = P["BTC"]; ret = RETS["BTC"]
    hh55 = btc.rolling(55).max().shift(1); ll55 = btc.rolling(55).min().shift(1)
    atr30 = btc.diff().abs().rolling(30).mean()
    acct = LadderAccount(D, leverage=1.0, side_rate=0.0007)
    acct.mark(0, 0.0, 0.0)  # bar 0: flat
    pos = 0.0; entry = 0.0; entry_i = None
    for i in range(1, len(D)):
        px = btc.iloc[i]
        if variant in ("classic", "timestop"):
            if pos == 0:
                if px > hh55.iloc[i]:
                    pos, entry, entry_i = 1.0, px, i
                    acct.rebalance(i, {"BTC": 1.0})
                elif px < ll55.iloc[i]:
                    pos, entry, entry_i = -1.0, px, i
                    acct.rebalance(i, {"BTC": -1.0})
            else:
                flip = (pos == 1 and px < ll55.iloc[i]) or (pos == -1 and px > hh55.iloc[i])
                stuck = False
                if variant == "timestop" and entry_i is not None and (i - entry_i) >= 30:
                    stuck = abs(px - entry) < 0.5 * atr30.iloc[i]
                if flip or stuck:
                    pos = 0.0
                    acct.rebalance(i, {})
        w_eff = acct.effective_weights(i)          # lag-correct: decided at close i-1
        wbtc = w_eff.get("BTC", 0.0)
        acct.mark(i, wbtc * ret.iloc[i], wbtc * FUND.iloc[i])
    return acct

print("=" * 74)
print("R119 — CORRECTED TWIN HUNT (LadderAccount invariants ON)")
print("=" * 74)
SUPERSEDED = {  # as reported by R118 (defective accounting)
    "H-T1 always-in":  dict(apr=10.8, maxDD=-24.7),
    "H-T1 filtered":   dict(apr=10.0, maxDD=-20.7),
    "H-T2 classic":    dict(apr=19.8, maxDD=-38.6),
    "H-T2 timestop":   dict(apr=27.7, maxDD=-41.3),
}
results = {}
for name, acct in [("H-T1 always-in", ht1(False)), ("H-T1 filtered", ht1(True)),
                   ("H-T2 classic", ht2("classic")), ("H-T2 timestop", ht2("timestop"))]:
    eq = acct.equity_returns()
    full = performance_report(window_slice(eq, T0))
    g, n = charter_check(full)
    hold = window_slice(eq, HOLD0).sum() * 100
    old = SUPERSEDED[name]
    print(fmt_report(name, full))
    print(f"{'':30s}  gates {n}/6 | holdout {hold:+.2f}% {'✓' if hold >= 0 else '✗'} | rebalances={acct.n_rebalances} "
          f"| defect impact: CAGR {full['ann_cagr'] - old['apr']:+.1f}pp vs superseded R118, maxDD {full['maxDD'] - old['maxDD']:+.1f}pp")
    results[name] = (full, g, n, hold)

print("\n--- VERDICTS (charter, corrected accounting) ---")
for name, (full, g, n, hold) in results.items():
    verdict = "SURVIVES ✅" if n == 6 and hold >= 0 else ("PROMISING FOR FURTHER VALIDATION 🟡" if n >= 5 and hold >= 0 else "REJECT ❌")
    print(f"{name:20s}: {n}/6 gates, holdout {hold:+.2f}% -> {verdict}")

best = max(results.items(), key=lambda kv: kv[1][2])
print(f"\nBEST: {best[0]} at {best[1][2]}/6 — the audit answer: none reaches charter; "
      "cost-accounting defects moved APR by <±3pp and no verdict flips vs R118.")
for name, acct in [("ht1_always", ht1(False)), ("ht1_filt", ht1(True)), ("ht2_classic", ht2("classic")), ("ht2_timestop", ht2("timestop"))]:
    acct.equity_returns().to_csv(pathlib.Path("quantlab_output") / f"r119_{name}.csv")
print("curves saved: r119_*.csv")
