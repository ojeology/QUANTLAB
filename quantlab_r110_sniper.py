#!/usr/bin/env python3
"""R110 — GROUND ZERO SNIPER S1: daily cascade reversal. Prespec: GROUND_ZERO_SNIPER.md (locked).
Signal at daily close: close < SMA20-k*ATR14 AND dd-from-20d-high >= 12%. Enter at that close.
Exit at closes: >= SMA20 (target) / <= entry-1*ATR14 (stop) / T-day time stop. 1 pos/coin, max 3,
tiebreak deepest. Costs 0.10% RT in R. Risk 1%. Gates: yearly>=0R, maxDD<=12%, worstW>=-3%,
sel(2024-01->2026-03) n>=30 & avg>=+0.40R, holdout(2026-04->10) >=0R & PF>=1.3."""
import pathlib
import numpy as np
import pandas as pd

CACHE = pathlib.Path("quantlab_cache/daily"); OUT = pathlib.Path("quantlab_output")
COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "ADA", "LINK", "LTC", "AVAX",
         "DOT", "NEAR", "APT", "SUI", "ATOM", "FIL", "INJ", "ARB", "OP", "ETC"]
COST_PCT = 0.001; RISK = 0.01
T0 = pd.Timestamp("2024-01-01"); SEL_END = pd.Timestamp("2026-03-31"); HOLD_START = pd.Timestamp("2026-04-01")
MAX_CONC = 3; DD_REQ = 0.12

def load(c):
    df = pd.read_csv(CACHE / f"{c}_1D.csv", parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    tr = pd.concat([df.h - df.l, (df.h - df.c.shift()).abs(), (df.l - df.c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["sma"] = df.c.rolling(20).mean()
    df["dd"] = 1 - df.c / df.h.rolling(20).max()
    df["z"] = (df.sma - df.c) / df.atr  # depth below band (tiebreak)
    bad = df.c.diff().abs() / (df.c.shift() * df.atr / df.c) > 25
    if bad.any():
        print(f"  [scrub] {c}: {int(bad.sum())} bad bar(s)")
        prev = df.c.shift(); df.loc[bad, ["o", "h", "l", "c"]] = np.tile(prev[bad].values[:, None], (1, 4))
        df["atr"] = tr.rolling(14).mean(); df["sma"] = df.c.rolling(20).mean()
    return df

def run(k, T):
    dfe = {c: load(c) for c in COINS}
    master = sorted(set().union(*[set(d.ts) for d in dfe.values()]))
    for c in COINS:  # align every coin to the master calendar (NaN where no data)
        dfe[c] = (dfe[c].set_index("ts").reindex(pd.DatetimeIndex(master))
                    .rename_axis("ts").reset_index())
    trades = []; open_pos = {}
    n_days = len(dfe["BTC"]); dates = dfe["BTC"].ts
    for i in range(n_days):
        # exits first (all coins, at closes)
        for c in list(open_pos):
            d = dfe[c]; p = open_pos[c]
            if i <= p["i"]:  # same-bar guard
                continue
            stopped = d.c.iloc[i] <= p["entry"] - 1.0 * p["atr0"]
            target = d.c.iloc[i] >= d.sma.iloc[i]
            timeout = (i - p["i"]) >= T
            if stopped or target or timeout:
                e_i = p["i"]
                atr_d = p["atr0"]  # bucket in DOLLARS (price units)
                gross_r = (d.c.iloc[i] - d.c.iloc[e_i]) / atr_d
                px = gross_r - COST_PCT * float(d.c.iloc[e_i]) / atr_d
                trades.append(dict(coin=c, entry_t=dates.iloc[e_i], exit_t=dates.iloc[i], days=i - p["i"],
                                   net_r=px, gross_r=gross_r, bucket=atr_d / float(d.c.iloc[e_i]),
                                   entry_px=float(d.c.iloc[e_i])))
                del open_pos[c]
        # entries (capacity + tiebreak)
        sigs = []
        for c in COINS:
            if c in open_pos:
                continue
            d = dfe[c]
            if i < 21 or np.isnan(d.atr.iloc[i]) or d.atr.iloc[i] <= 0:
                continue
            if (d.c.iloc[i] < d.sma.iloc[i] - k * d.atr.iloc[i]) and (d.dd.iloc[i] >= DD_REQ):
                sigs.append((d.z.iloc[i], c))
        for z, c in sorted(sigs, reverse=True):
            if len(open_pos) >= MAX_CONC:
                break
            d = dfe[c]
            open_pos[c] = dict(i=i, entry=d.c.iloc[i], atr0=d.atr.iloc[i], bucket=d.atr.iloc[i] / d.c.iloc[i])
            d._entered = True
    # force-close stragglers at final bar
    for c, p in open_pos.items():
        d = dfe[c]; i = n_days - 1
        atr_d = p["atr0"]
        gross_r = (float(d.c.iloc[i]) - p["entry"]) / atr_d
        px = gross_r - COST_PCT * p["entry"] / atr_d
        trades.append(dict(coin=c, entry_t=dates.iloc[p["i"]], exit_t=dates.iloc[i], days=i - p["i"],
                           net_r=px, gross_r=gross_r, bucket=atr_d / p["entry"], entry_px=p["entry"]))
    # daily R from trades (MTM day-by-day, cost on exit day; off-by-one-safe: earn from entry+1)
    daily = pd.Series(0.0, index=pd.DatetimeIndex(dates))
    closes = {c: dfe[c].c.values for c in COINS}
    tmap = {c: dfe[c].ts for c in COINS}
    for t in trades:
        c = t["coin"]
        idx_e = int(np.where(tmap[c].values == np.datetime64(t["entry_t"]))[0][0])
        idx_x = int(np.where(tmap[c].values == np.datetime64(t["exit_t"]))[0][0])
        atr_d = t["bucket"] * t["entry_px"]  # bucket back to dollars
        for j in range(idx_e + 1, idx_x + 1):
            daily.iloc[j] += (closes[c][j] - closes[c][j - 1]) / atr_d
        daily.iloc[idx_x] -= COST_PCT * t["entry_px"] / atr_d
    gap = abs(daily.sum() - sum(t["net_r"] for t in trades))
    assert gap < 1e-6, f"integrity gap {gap}"
    return pd.DataFrame(trades), daily

def gates(daily, n_sel, avg_sel):
    d = daily[daily.index >= T0]
    weekly = d.resample("W").sum() * RISK * 100
    eq = (1 + weekly / 100).cumprod(); dd = (eq / eq.cummax() - 1).min() * 100
    yr = d.groupby(d.index.year).sum().to_dict()
    return dict(posW=(weekly > 0).mean() * 100, worstW=weekly.min(), maxDD=dd,
                y2024=yr.get(2024, 0.0), y2025=yr.get(2025, 0.0), y2026=yr.get(2026, 0.0))

print("=" * 74)
print("R110 — SNIPER S1: daily cascade reversal | 20 coins | 1%/trade | max 3 concurrent")
print("=" * 74)
best = None
for k in (2.0, 2.5, 3.0):
    for T in (7, 10):
        tr, daily = run(k, T)
        sel_tr = tr[pd.to_datetime(tr.entry_t) <= SEL_END]
        d_sel = daily[(daily.index >= T0) & (daily.index <= SEL_END)]
        g = gates(d_sel, len(sel_tr), sel_tr.net_r.mean() if len(sel_tr) else 0)
        avg = sel_tr.net_r.mean() if len(sel_tr) else -9
        passed = sum([g["y2024"] >= 0, g["y2025"] >= 0, g["y2026"] >= 0, g["maxDD"] >= -12, g["worstW"] >= -3])
        ok = (passed == 5) and len(sel_tr) >= 30 and avg >= 0.40
        print(f"k={k} T={T:2d}: n={len(sel_tr):3d} avgR={avg:+.2f} worstW={g['worstW']:+6.2f}% maxDD={g['maxDD']:+6.2f}% "
              f"| R {g['y2024']:+6.1f}/{g['y2025']:+6.1f}/{g['y2026']:+6.1f} | core {passed}/5 {'✓CAND' if ok else ''}")
        key = (ok, passed, avg)
        if best is None or key > best[0]:
            best = (key, k, T)
_, bk, bT = best
print(f"\n>>> SELECTED: k={bk} T={bT} (selection window only)")
tr, daily = run(bk, bT)
tr.to_csv(OUT / "r110_sniper_trades.csv", index=False)
d = daily[daily.index >= T0]
weekly = d.resample("W").sum() * RISK * 100
weekly.rename("pct").to_csv(OUT / "r110_sniper_weekly.csv", index=True)
g = gates(d, 0, 0)
yr = d.groupby(d.index.year).sum()
print("\n--- FULL 2024-01 -> 2026-10 ---")
print(f"trades={len(tr)} WR={(tr.net_r > 0).mean() * 100:.1f}% avgR={tr.net_r.mean():+.2f} medHold={tr.days.median():.0f}d")
print(f"posW={(weekly > 0).mean() * 100:.1f}% worstW={g['worstW']:+.2f}% maxDD={g['maxDD']:+.2f}%")
print(f"YEARLY R: 2024 {g['y2024']:+.1f} | 2025 {g['y2025']:+.1f} | 2026 {g['y2026']:+.1f}")
hold_tr = tr[pd.to_datetime(tr.exit_t) >= HOLD_START]
hpf = hold_tr.net_r[hold_tr.net_r > 0].sum() / max(1e-9, -hold_tr.net_r[hold_tr.net_r < 0].sum())
print(f"HOLDOUT 2026-04->10: n={len(hold_tr)} R={hold_tr.net_r.sum():+.1f} PF={hpf:.2f} "
      f"({'✓PASS' if (hold_tr.net_r.sum() >= 0 and hpf >= 1.3) else '✗FAIL'})")
print("per-coin:", {c: round(tr[tr.coin == c].net_r.sum(), 1) for c in tr.coin.unique()})
