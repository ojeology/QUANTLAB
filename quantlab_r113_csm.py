#!/usr/bin/env python3
"""R113 — CROSS-SECTIONAL CRYPTO MOMENTUM (last honest candidate). Prespec: CONSISTENCY_HUNT_BEGINS.md.
Weekly Friday-close rotation: top-K by trailing N-day return, 1/K equal-weight; exposure in
{always, btc_filter}; costs 0.05%/side on weight changes; weights apply to NEXT bar's returns
(off-by-one law). Gates: posW>=65, worstW>=-3%, maxDD<=12%, yearly 2024/25/26 >=0R, holdout >=0R.
SANITY GATE: |weekly| <= 60% else abort (unit-slip tripwire)."""
import pathlib
import numpy as np
import pandas as pd

CACHE = pathlib.Path("quantlab_cache/daily"); OUT = pathlib.Path("quantlab_output")
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK", "LTC", "AVAX",
         "DOT", "NEAR", "APT", "SUI", "ATOM", "FIL", "INJ", "ARB", "OP", "ETC"]
SIDE_COST = 0.0005
T0 = pd.Timestamp("2024-01-01"); SEL_END = pd.Timestamp("2026-03-31"); HOLD_START = pd.Timestamp("2026-04-01")

def load_panel():
    dfs = {}
    for c in COINS:
        d = pd.read_csv(CACHE / f"{c}_1D.csv", parse_dates=["ts"]).sort_values("ts")
        dfs[c] = d.set_index("ts")["c"]
    px = pd.DataFrame(dfs).sort_index()
    bad = (px.pct_change().abs() > 0.5).any(axis=1)
    if bad.any():
        print(f"  [scrub] {int(bad.sum())} panel bar(s) with >50% moves — prices clipped to prev")
        px = px.mask(px.pct_change().abs() > 0.5).ffill()
    rets = px.pct_change().fillna(0.0)
    return px, rets

px, rets = load_panel()
btc = px["BTC"]; btc_ok = btc > btc.rolling(50).mean()  # filter evaluated at rebalance close

def run(N, K, filt):
    w = pd.Series(0.0, index=COINS)
    daily_pnl = pd.Series(0.0, index=px.index)
    turnover_cost = pd.Series(0.0, index=px.index)
    iso = px.index.strftime("%G-%V")
    is_mark = np.concatenate(([False], iso[1:] != iso[:-1]))  # first bar of each ISO week = decision? no:
    # rebalance at LAST bar of each ISO week:
    is_last = np.concatenate([(iso[1:] != iso[:-1]), [True]])
    pending = None
    for i in range(len(px)):
        if pending is not None:  # weights decided last bar now start earning
            turnover_cost.iloc[i] = SIDE_COST * float(np.abs(pending - w).sum())
            w = pending; pending = None
        daily_pnl.iloc[i] = float((rets.iloc[i] * w).sum())
        if is_last[i] and i + 1 < len(px):
            if filt and not bool(btc_ok.iloc[i]):
                new = pd.Series(0.0, index=COINS)
            else:
                mom = px.iloc[i] / px.iloc[i - N] - 1.0
                if mom.isna().any():
                    continue  # warmup — keep prior weights
                new = pd.Series(0.0, index=COINS)
                top = mom.nlargest(K).index
                new[top] = 1.0 / K
            pending = new
    curve = daily_pnl - turnover_cost  # in portfolio-return units (1.0 = 100%)
    weekly = curve.resample("W").sum() * 100.0
    assert weekly.abs().max() <= 60, f"SANITY GATE TRIPPED: weekly {weekly.abs().max():.1f}%"
    return weekly, curve

def gates(weekly):
    d = weekly[weekly.index >= T0]
    eq = (1 + d / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100
    yr = d.groupby(d.index.year).sum()
    sel = d[d.index <= SEL_END]
    hold = d[d.index >= HOLD_START]
    return dict(posW=(d > 0).mean() * 100, worstW=d.min(), maxDD=mdd,
                y2024=yr.get(2024, 0.0), y2025=yr.get(2025, 0.0), y2026=yr.get(2026, 0.0),
                sel_sum=sel.sum(), hold_sum=hold.sum(), hold_min=hold.min())

print("=" * 74)
print("R113 — CROSS-SECTIONAL MOMENTUM | 20 coins | weekly rotation | 0.05%/side")
print("=" * 74)
results = {}
for N in (7, 14, 28):
    for K in (3, 5):
        for filt in (False, True):
            weekly, curve = run(N, K, filt)
            g = gates(weekly)
            passed = sum([g["posW"] >= 65, g["worstW"] >= -3, g["maxDD"] >= -12,
                          g["y2024"] >= 0, g["y2025"] >= 0, g["y2026"] >= 0])
            results[(N, K, filt)] = (g, weekly, passed)
            print(f"N={N:2d} K={K} {'FILT' if filt else 'ALW ':4s}: posW={g['posW']:5.1f}% worstW={g['worstW']:+6.2f}% "
                  f"maxDD={g['maxDD']:+6.1f}% | R {g['y2024']:+7.1f}/{g['y2025']:+7.1f}/{g['y2026']:+7.1f}% "
                  f"| ann~{weekly[weekly.index >= T0].mean() * 52:+.1f}% | gates {passed}/6")

best_key = max(results, key=lambda k: (results[k][2], results[k][0]["sel_sum"]))
g, weekly, passed = results[best_key]
print(f"\n>>> SELECTED: N={best_key[0]} K={best_key[1]} {'BTC-filter' if best_key[2] else 'always-in'} ({passed}/6 gates)")
print(f"HOLDOUT: sum={g['hold_sum']:+.1f}% worstW={g['hold_min']:+.2f}% ({'PASS' if g['hold_sum'] >= 0 else 'FAIL'})")
weekly.rename("pct").to_csv(OUT / "r113_csm_weekly.csv", index=True)
yr = weekly[weekly.index >= T0].groupby(weekly[weekly.index >= T0].index.year).sum()
print("FINAL VERDICT:", "✅ PASSES" if passed == 6 and g["hold_sum"] >= 0 else "❌ REJECTED")
print("context yearly %:", {int(y): round(v, 1) for y, v in yr.items()})
