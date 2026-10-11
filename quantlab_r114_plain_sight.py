#!/usr/bin/env python3
"""R114 — THE 'PLAIN SIGHT' SWEEP. Prespec: CONSISTENCY_HUNT_BEGINS.md (R114 section).
B1 buy-and-hold benchmarks (BTC/ETH/EQW-20 3yr; XAU/XAG partial-window context) ·
B2 $10/week DCA since 2024-01 · B3 seasonality falsification (20 pre-declared tests,
Bonferroni p<0.0025) · B4 staking probe already run: endpoints dead (no floor upgrade)."""
import pathlib
import numpy as np
import pandas as pd

CACHE = pathlib.Path("quantlab_cache/daily"); OUT = pathlib.Path("quantlab_output")
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK", "LTC", "AVAX",
         "DOT", "NEAR", "APT", "SUI", "ATOM", "FIL", "INJ", "ARB", "OP", "ETC"]
SIDE = 0.0005
T0 = pd.Timestamp("2024-01-01")

def load(c):
    d = pd.read_csv(CACHE / f"{c}_1D.csv", parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    return d.set_index("ts")["c"]

px = pd.DataFrame({c: load(c) for c in COINS}).sort_index()
px = px.mask(px.pct_change().abs() > 0.5).ffill()  # same scrub law as R113
rets = px.pct_change().fillna(0.0)
iso = px.index.strftime("%G-%V")
is_last = np.concatenate([(iso[1:] != iso[:-1]), [True]])

print("=" * 74)
print("R114 — PLAIN SIGHT SWEEP | benchmarks, DCA, seasonality (Bonferroni 0.05/20)")
print("=" * 74)

# ---------- B1: BUY & HOLD ----------
def curve_stats(vals, label, note=""):
    v = vals[vals.index >= T0]
    r = v.pct_change().fillna(0)
    weekly = (v.resample("W").last().pct_change().dropna()) * 100
    yr = (v.resample("YE").last() / v.resample("YE").first() - 1) * 100
    yr.index = yr.index.year
    dd = ((v / v.cummax() - 1).min()) * 100
    print(f"  {label:14s} {note:22s} yearly: {yr.get(2024, float('nan')):+6.1f}/{yr.get(2025, float('nan')):+6.1f}/{yr.get(2026, float('nan')):+6.1f}%"
          f" | worstWk {weekly.min():+6.1f}% | maxDD {dd:+6.1f}%")

print("\nB1 — BUY & HOLD (no gates; 'would you have survived' = worstWk vs −3 / maxDD vs −12):")
curve_stats(px.BTC, "BH BTC")
curve_stats(px.ETH, "BH ETH")
# EQW-20 weekly rebalanced
w = pd.Series(1 / len(COINS), index=COINS)
daily_eq = pd.Series(0.0, index=px.index)
costs = pd.Series(0.0, index=px.index)
for i in range(len(px)):
    daily_eq.iloc[i] = float((rets.iloc[i] * w).sum())
    if is_last[i]:
        w = w * (1 + rets.iloc[i])
        w = w / w.sum()  # drifted
        costs.iloc[i + 1 if i + 1 < len(px) else i] = SIDE * float((1.0 / len(COINS) - w).abs().sum()) * 2
        w = pd.Series(1 / len(COINS), index=COINS)
eq_curve = (1 + daily_eq - costs).cumprod() * px.BTC.iloc[0]  # normalize to BTC start for % stats
curve_stats(eq_curve, "BH EQW-20", "(weekly re-bal)")

for sym, lab in [("XAU", "BH GOLD"), ("XAG", "BH SILVER")]:
    f = CACHE / f"{sym}_1D.csv"
    if f.exists():
        s = load(sym)
        print(f"  {lab:14s} window {s.index[0].date()}->    yearly-style: total {(s.iloc[-1] / s.iloc[0] - 1) * 100:+.1f}% "
              f"| maxDD {((s / s.cummax() - 1).min()) * 100:+.1f}% (context only)")

# ---------- B2: DCA ----------
print("\nB2 — DCA $10/week since 2024-01 (the no-chart strategy):")
marks = px.index[is_last]
marks = marks[marks >= T0]
for label in ["BTC", "ETH", "EQW20"]:
    coins_held = pd.Series(0.0, index=marks)
    if label == "EQW20":
        held = pd.Series(0.0, index=COINS)
        vals = []
        for m in marks:
            alloc = 10.0 / len(COINS)
            held = held + alloc / px.loc[m]
            vals.append(float((held * px.loc[m]).sum()))
        v = pd.Series(vals, index=marks)
    else:
        held = 0.0; vals = []
        for m in marks:
            held += 10.0 / px.loc[m, label]
            vals.append(held * px.loc[m, label])
        v = pd.Series(vals, index=marks)
    invested = 10.0 * len(marks)
    mult = v.iloc[-1] / invested
    ddpath = ((v / v.cummax() - 1).min()) * 100
    print(f"  {label:6s}: invested ${invested:,.0f} -> final ${v.iloc[-1]:,.0f}  ({mult:.2f}x)  worst unrealized DD {ddpath:+.1f}%")

# ---------- B3: SEASONALITY ----------
print("\nB3 — SEASONALITY FALSIFICATION (20 tests; survive only p<0.0025):")
eq_ret = daily_eq[daily_eq.index >= T0]
btc_ret = rets.BTC[rets.index >= T0]
CPI = set(pd.to_datetime(["2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12", "2024-07-11",
                          "2024-08-14", "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11", "2025-01-15", "2025-02-12",
                          "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-11",
                          "2025-10-24", "2025-12-18", "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12",
                          "2026-06-10", "2026-07-14", "2026-08-12", "2026-09-11"]).date)
FOMC = set(pd.to_datetime(["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
                           "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17",
                           "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
                           "2026-09-16"]).date)

def welch(a, b):
    if len(a) < 5 or len(b) < 5:
        return np.nan, np.nan
    ma, mb = a.mean(), b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return np.nan, np.nan
    t = (ma - mb) / se
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))  # normal approx (n large)
    return ma - mb, p

patterns = [
    ("Monday vs rest", lambda ix: ix.weekday == 0, None),
    ("Friday vs rest", lambda ix: ix.weekday == 4, None),
    ("Weekend vs wkday", lambda ix: ix.weekday >= 5, None),
    ("Turn-of-month", lambda ix: (ix.day <= 3) | (ix.day >= 28), None),
    ("First week", lambda ix: ix.day <= 7, None),
    ("Last week", lambda ix: ix.day >= 24, None),
    ("Mid-month", lambda ix: (ix.day >= 8) & (ix.day <= 14), None),
    ("January", lambda ix: ix.month == 1, None),
    ("CPI days vs rest", lambda ix: pd.Series([d.date() in CPI for d in ix], index=ix), "flag"),
    ("FOMC days vs rest", lambda ix: pd.Series([d.date() in FOMC for d in ix], index=ix), "flag"),
]
survivors = []
minp = (1.0, "")
for sname, s in [("EQW-20", eq_ret), ("BTC", btc_ret)]:
    for pname, msk, mode in patterns:
        m = msk(s.index)
        m = pd.Series(m, index=s.index) if not isinstance(m, pd.Series) else m
        a = s[m.astype(bool).values]; b = s[~m.astype(bool).values]
        diff, p = welch(a, b)
        ann = diff * 252 * 100
        tag = f"{sname:6s} {pname:18s}"
        if p is not np.nan and p < minp[0]:
            minp = (p, tag)
        flag = "SURVIVOR" if (p == p and p < 0.0025) else ""
        if flag:
            survivors.append((tag, diff, p))
        print(f"  {tag}: ann-diff {ann:+7.2f}%  p={p if p == p else 9:.4f} {flag}")
print(f"\n  SURVIVORS: {len(survivors)}/20  | smallest p = {minp[0]:.4f} ({minp[1].strip()})  | threshold 0.0025")
print("  VERDICT:", "MIRAGE CONFIRMED — no calendar truth hiding here" if not survivors else "REAL PATTERN FOUND — deserves its own round")
