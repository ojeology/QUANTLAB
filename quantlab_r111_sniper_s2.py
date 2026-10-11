#!/usr/bin/env python3
"""R111 — GROUND ZERO SNIPER S2: macro-catalyst expansion (FOMC/CPI). Prespec: GROUND_ZERO_SNIPER.md.
Ambiguity resolution (mechanical, before any run): prespec's 'enter at eve close' clause is
unexecutable close-only without a direction; the locked direction FORMULA (event-day close vs
eve close +/- 0.5 ATR) defines the operative entry = event-day close. Disclosed in report.
Event calendar = public record: federalreserve.gov (FOMC 14:00 ET) + BLS (CPI 08:30 ET).
2025 shutdown reschedule: Sep CPI released Oct 24 2025; October 2025 CPI canceled; Nov CPI Dec 18 2025.
Events land 13:30/19:00 UTC -> inside the UTC daily bar of the calendar date (OKX venue)."""
import pathlib
import numpy as np
import pandas as pd

CACHE = pathlib.Path("quantlab_cache/daily"); OUT = pathlib.Path("quantlab_output")
COINS = ["BTC", "ETH"]
COST_PCT = 0.001; RISK = 0.01; TRIG_ATR = 0.5; STOP_ATR = 1.0; T_STOP = 2
T0 = pd.Timestamp("2024-01-01"); SEL_END = pd.Timestamp("2026-03-31"); HOLD_START = pd.Timestamp("2026-04-01")

FOMC = ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16"]
CPI = ["2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
       "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
       "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13", "2025-06-11", "2025-07-15", "2025-08-12",
       "2025-09-11", "2025-10-24", "2025-12-18",
       "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12", "2026-06-10", "2026-07-14", "2026-08-12",
       "2026-09-11"]
EVENTS = sorted(pd.to_datetime(FOMC + CPI).tz_localize(None))
print(f"events: {len(EVENTS)} ({len(FOMC)} FOMC + {len(CPI)} CPI) over {EVENTS[0].date()} -> {EVENTS[-1].date()}")

def load(c):
    df = pd.read_csv(CACHE / f"{c}_1D.csv", parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    tr = pd.concat([df.h - df.l, (df.h - df.c.shift()).abs(), (df.l - df.c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df

data = {c: load(c) for c in COINS}
trades = []
for ev in EVENTS:
    for c in COINS:
        d = data[c]
        idx = d.index[d.ts.dt.normalize() == ev]
        if len(idx) == 0:
            continue
        i = idx[0]
        if i < 16:
            continue
        eve_c = float(d.c.iloc[i - 1]); atr = float(d.atr.iloc[i - 1])
        c_ev = float(d.c.iloc[i])
        if np.isnan(atr) or atr <= 0:
            continue
        band = TRIG_ATR * atr
        if c_ev > eve_c + band:
            dirn = 1
        elif c_ev < eve_c - band:
            dirn = -1
        else:
            continue  # fizzle: no trade (sniper skips)
        entry = c_ev; atr_d = atr
        exit_i = None
        for j in range(i + 1, min(i + T_STOP + 1, len(d))):
            adverse = (float(d.c.iloc[j]) - entry) * dirn <= -STOP_ATR * atr_d
            timeout = (j - i) >= T_STOP
            if adverse or timeout:
                exit_i = j; break
        if exit_i is None:
            exit_i = len(d) - 1
        gross_r = (float(d.c.iloc[exit_i]) - entry) * dirn / atr_d
        net_r = gross_r - COST_PCT * entry / atr_d
        trades.append(dict(event=str(ev.date()), kind=("FOMC" if str(ev.date()) in FOMC else "CPI"), coin=c,
                           dir=dirn, entry_t=ev, exit_t=d.ts.iloc[exit_i], days=exit_i - i,
                           gross_r=gross_r, net_r=net_r, bucket=atr_d / entry, entry_px=entry))
tdf = pd.DataFrame(trades)
# daily MTM curve (entry+1 .. exit), cost on exit day
daily = pd.Series(0.0, index=pd.DatetimeIndex(sorted(set(data["BTC"].ts.dt.normalize()))))
for t in trades:
    c = t["coin"]; d = data[c]
    i = int(d.index[d.ts.dt.normalize() == pd.Timestamp(t["entry_t"])][0])
    x = int(d.index[d.ts.dt.normalize() == pd.Timestamp(t["exit_t"]).normalize()][0])
    for j in range(i + 1, x + 1):
        daily.loc[d.ts.iloc[j].normalize()] += (float(d.c.iloc[j]) - float(d.c.iloc[j - 1])) * t["dir"] / (t["bucket"] * t["entry_px"])
    daily.loc[d.ts.iloc[x].normalize()] -= COST_PCT * t["entry_px"] / (t["bucket"] * t["entry_px"])
gap = abs(daily.sum() - tdf.net_r.sum()); assert gap < 1e-6, f"integrity {gap}"

tdf.to_csv(OUT / "r111_event_trades.csv", index=False)
d = daily[daily.index >= T0]
weekly = d.resample("W").sum() * RISK * 100
weekly.rename("pct").to_csv(OUT / "r111_event_weekly.csv", index=True)

sel = tdf[pd.to_datetime(tdf.entry_t) <= SEL_END]
print(f"\ntrades total={len(tdf)} (sel={len(sel)}) | skipped fizzles included in count")
print(f"SELECTION: avgR={sel.net_r.mean():+.3f} (bar >= +0.40) WR={(sel.net_r > 0).mean() * 100:.0f}%")
print(tdf.groupby("kind").net_r.agg(["count", "mean", "sum"]).round(3))
print(tdf.groupby([tdf.entry_t.dt.year, "kind"]).net_r.agg(["count", "sum", "mean"]).round(2))
eq = (1 + weekly / 100).cumprod(); dd = (eq / eq.cummax() - 1).min() * 100
yr = d.groupby(d.index.year).sum()
print(f"FULL: maxDD={dd * 100:.1f}% worstW={weekly.min():+.2f}% | yearly R: " +
      " ".join(f"{y}:{v:+.1f}" for y, v in yr.items()))
hold = tdf[pd.to_datetime(tdf.exit_t) >= HOLD_START]
hpf = hold.net_r[hold.net_r > 0].sum() / max(1e-9, -hold.net_r[hold.net_r < 0].sum())
print(f"HOLDOUT: n={len(hold)} R={hold.net_r.sum():+.1f} PF={hpf:.2f}")
print("YEARLY GATES:", {int(y): ("PASS" if v >= 0 else "FAIL") for y, v in yr.items()})
print("VERDICT avgR:", "PASS" if sel.net_r.mean() >= 0.40 else "FAIL")
