#!/usr/bin/env python3
"""R112 — GROUND ZERO SNIPER S3: weekly extreme reversion. Prespec: GROUND_ZERO_SNIPER.md (locked).
Friday close with 5-day return <= thr ({-12%,-15%}) -> buy that close. Exit at closes:
reclaim (>= close 5d ago) / stop entry-1.25*ATR14 / 15d time stop. 1%/pos, max 3, deepest tiebreak,
0.10% RT. Gates: sel n>=20 & avg>=+0.40R; yearly>=0R; maxDD<=12%; worstW>=-3%; holdout >=0R & PF>=1.3."""
import pathlib
import numpy as np
import pandas as pd

CACHE = pathlib.Path("quantlab_cache/daily"); OUT = pathlib.Path("quantlab_output")
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK", "LTC", "AVAX",
         "DOT", "NEAR", "APT", "SUI", "ATOM", "FIL", "INJ", "ARB", "OP", "ETC"]
COST_PCT = 0.001; RISK = 0.01; STOP_ATR = 1.25; T_STOP = 15; WEEK = 5
T0 = pd.Timestamp("2024-01-01"); SEL_END = pd.Timestamp("2026-03-31"); HOLD_START = pd.Timestamp("2026-04-01")
MAX_CONC = 3

def load(c):
    df = pd.read_csv(CACHE / f"{c}_1D.csv", parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    tr = pd.concat([df.h - df.l, (df.h - df.c.shift()).abs(), (df.l - df.c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df

def run(thr):
    dfe = {c: load(c) for c in COINS}
    master = pd.DatetimeIndex(sorted(set().union(*[set(d.ts) for d in dfe.values()])))
    for c in COINS:
        dfe[c] = dfe[c].set_index("ts").reindex(master).rename_axis("ts").reset_index()
    dates = dfe["BTC"].ts; n_days = len(dates)
    trades = []; open_pos = {}
    for i in range(n_days):
        # exits
        for c in list(open_pos):
            d = dfe[c]; p = open_pos[c]
            if i <= p["i"]:
                continue
            ci = float(d.c.iloc[i])
            if np.isnan(ci):
                continue
            reclaim = ci >= p["reclaim"]
            stopped = ci <= p["entry"] - STOP_ATR * p["atr0"]
            timeout = (i - p["i"]) >= T_STOP
            if reclaim or stopped or timeout:
                atr_d = p["atr0"]
                gross_r = (ci - p["entry"]) / atr_d
                net_r = gross_r - COST_PCT * p["entry"] / atr_d
                trades.append(dict(coin=c, entry_t=dates.iloc[p["i"]], exit_t=dates.iloc[i], days=i - p["i"],
                                   net_r=net_r, gross_r=gross_r, bucket=atr_d / p["entry"], entry_px=p["entry"]))
                del open_pos[c]
        # entries: Friday-close weekly extreme
        if len(open_pos) < MAX_CONC:
            sigs = []
            for c in COINS:
                if c in open_pos:
                    continue
                d = dfe[c]
                if i < 21:
                    continue
                ci = float(d.c.iloc[i]); c5 = float(d.c.iloc[i - WEEK]); atr = float(d.atr.iloc[i - 1])
                if np.isnan(ci) or np.isnan(c5) or np.isnan(atr) or atr <= 0:
                    continue
                wk_ret = ci / c5 - 1
                if wk_ret <= thr:
                    sigs.append((wk_ret, c))
            for wr, c in sorted(sigs):
                if len(open_pos) >= MAX_CONC:
                    break
                d = dfe[c]; i0 = i
                open_pos[c] = dict(i=i0, entry=float(d.c.iloc[i0]), atr0=float(d.atr.iloc[i0 - 1]),
                                   reclaim=float(d.c.iloc[i0 - WEEK]))
    for c, p in open_pos.items():
        d = dfe[c]; i = n_days - 1
        gross_r = (float(d.c.iloc[i]) - p["entry"]) / p["atr0"]
        trades.append(dict(coin=c, entry_t=dates.iloc[p["i"]], exit_t=dates.iloc[i], days=i - p["i"],
                           net_r=gross_r - COST_PCT * p["entry"] / p["atr0"], gross_r=gross_r,
                           bucket=p["atr0"] / p["entry"], entry_px=p["entry"]))
    tdf = pd.DataFrame(trades)
    daily = pd.Series(0.0, index=master)
    for t in trades:
        c = t["coin"]; d = dfe[c]
        idx = d.index[d.ts == t["entry_t"]][0]
        xdx = d.index[d.ts == t["exit_t"]][0]
        atr_d = t["bucket"] * t["entry_px"]
        for j in range(idx + 1, xdx + 1):
            daily.iloc[j] += (float(d.c.iloc[j]) - float(d.c.iloc[j - 1])) / atr_d
        daily.iloc[xdx] -= COST_PCT * t["entry_px"] / atr_d
    gap = abs(daily.sum() - tdf.net_r.sum()); assert gap < 1e-6, f"integrity {gap}"
    return tdf, daily

print("=" * 74)
print("R112 — SNIPER S3: weekly extreme reversion | 20 coins | 1%/trade | max 3")
print("=" * 74)
best = None
for thr in (-0.12, -0.15):
    tr, daily = run(thr)
    sel_tr = tr[pd.to_datetime(tr.entry_t) <= SEL_END]
    d_sel = daily[(daily.index >= T0) & (daily.index <= SEL_END)]
    weekly = d_sel.resample("W").sum() * RISK * 100
    eq = (1 + weekly / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100
    yr = d_sel.groupby(d_sel.index.year).sum()
    avg = sel_tr.net_r.mean() if len(sel_tr) else -9
    core = sum([all(yr >= 0), mdd >= -12, weekly.min() >= -3])
    ok = (len(sel_tr) >= 20 and avg >= 0.40 and core == 3)
    print(f"thr={thr:.0%}: sel n={len(sel_tr):3d} avgR={avg:+.2f} (bar +0.40) worstW={weekly.min():+.2f}% "
          f"maxDD={mdd:+.1f}% | selR {yr.get(2024, 0):+.1f}/{yr.get(2025, 0):+.1f}/{yr.get(2026, 0):+.1f} | core {core}/3 {'✓CAND' if ok else ''}")
    key = (ok, len(sel_tr), avg)
    if best is None or key > best[0]:
        best = (key, thr)
_, bthr = best
print(f"\n>>> SELECTED: thr={bthr:.0%} (selection only)")
tr, daily = run(bthr)
tr.to_csv(OUT / "r112_s3_trades.csv", index=False)
d = daily[daily.index >= T0]
weekly = d.resample("W").sum() * RISK * 100
weekly.rename("pct").to_csv(OUT / "r112_s3_weekly.csv", index=True)
eq = (1 + weekly / 100).cumprod(); mdd = (eq / eq.cummax() - 1).min() * 100
yr = d.groupby(d.index.year).sum()
print(f"FULL: trades={len(tr)} WR={(tr.net_r > 0).mean() * 100:.0f}% avgR={tr.net_r.mean():+.2f}")
print(f"posW={(weekly > 0).mean() * 100:.0f}% worstW={weekly.min():+.2f}% maxDD={mdd:+.1f}%")
print("YEARLY R:", {int(y): round(v, 1) for y, v in yr.items()})
hold = tr[pd.to_datetime(tr.exit_t) >= HOLD_START]
hpf = hold.net_r[hold.net_r > 0].sum() / max(1e-9, -hold.net_r[hold.net_r < 0].sum()) if len(hold) else float("nan")
print(f"HOLDOUT: n={len(hold)} R={hold.net_r.sum():+.1f} PF={hpf:.2f}")
print("exits:", {k: int(v) for k, v in tr.days.apply(lambda x: "reclaim<15" if x < 15 else "time/forced").value_counts().items()})
print("per-coin R:", {c: round(tr[tr.coin == c].net_r.sum(), 1) for c in tr.coin.unique()})
