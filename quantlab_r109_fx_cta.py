#!/usr/bin/env python3
"""R109 — F3: CTA-HORIZON TREND (FX hunt finale) — prespec in FOREX_HUNT_BEGINS.md (R109 section).
Daily close-only Donchian N in {20,60,120,252} trading days; always-in flip; 2.5xATR24d bucket;
0.5%/pair risk; 1.2 pip RT + 0.7 pip/day swap. Gates on 2024-01->2026-10; selection 2024-01->2026-03;
holdout 2026-04->10. Decade of dailies (2016-10->2026-10) = warmup + supplementary context (NOT gated)."""
import pathlib
import numpy as np
import pandas as pd

CACHE = pathlib.Path("quantlab_cache/forex"); OUT = pathlib.Path("quantlab_output")
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP"]
PIP = {p: 0.0001 for p in PAIRS}; PIP["USDJPY"] = 0.01
COST_PIPS = 1.2; SWAP_PIPS_DAY = 0.7; RISK_PCT = 0.005; RISK_ATR = 2.5; ATR_N = 24
T0 = pd.Timestamp("2024-01-01"); SEL_END = pd.Timestamp("2026-03-31"); HOLD_START = pd.Timestamp("2026-04-01")

def load(pair):
    df = pd.read_csv(CACHE / f"{pair}_1D.csv", parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
    def true_range(d):
        tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(),
                        (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
        return tr
    df["atr"] = true_range(df).rolling(ATR_N).mean()
    bad = (df.close.diff().abs() > 12 * df.atr) | ((df.high - df.low) > 12 * df.atr)
    if bad.sum():
        prev = df["close"].shift()
        df.loc[bad, ["open", "high", "low", "close"]] = np.tile(prev[bad].values[:, None], (1, 4))
        print(f"  [scrub] {pair}: {int(bad.sum())} bad-tick bar(s)")
        df["atr"] = true_range(df).rolling(ATR_N).mean()
    return df

def run_variant(pair, df, N):
    hh = df.high.rolling(N).max().shift(1)
    ll = df.low.rolling(N).min().shift(1)
    c = df.close.values; hhi = hh.values; lli = ll.values
    atr = df.atr.values; ts = df.ts.values
    pos = 0; entry_i = None
    trades = []; pos_arr = np.zeros(len(df)); risk_arr = np.zeros(len(df))
    for i in range(len(df)):
        if np.isnan(hhi[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            pos_arr[i] = pos; risk_arr[i] = risk_arr[i - 1] if i else 0; continue
        newpos = pos
        if c[i] > hhi[i]: newpos = 1
        elif c[i] < lli[i]: newpos = -1
        if newpos != pos:
            if pos != 0:
                j = i; days = (ts[j] - ts[entry_i]) / np.timedelta64(1, "D")
                pips = (c[j] - c[entry_i]) * pos / PIP[pair]
                rd = RISK_ATR * atr[entry_i] / PIP[pair]
                trades.append(dict(pair=pair, N=N, entry_time=ts[entry_i], exit_time=ts[j], dir=pos,
                                   days=days, pips=pips, risk_dist_pips=rd,
                                   net_r=(pips / rd) - (COST_PIPS + SWAP_PIPS_DAY * days) / rd))
            if newpos != 0:
                entry_i = i
            pos = newpos
        pos_arr[i] = pos
        risk_arr[i] = (RISK_ATR * atr[entry_i] if pos != 0 and entry_i is not None else 0.0)
    if pos != 0 and entry_i is not None:
        j = len(df) - 1; days = (ts[j] - ts[entry_i]) / np.timedelta64(1, "D")
        pips = (c[j] - c[entry_i]) * pos / PIP[pair]; rd = RISK_ATR * atr[entry_i] / PIP[pair]
        trades.append(dict(pair=pair, N=N, entry_time=ts[entry_i], exit_time=ts[j], dir=pos,
                           days=days, pips=pips, risk_dist_pips=rd,
                           net_r=(pips / rd) - (COST_PIPS + SWAP_PIPS_DAY * days) / rd))
    # daily R with OFF-BY-ONE GUARD (position during bar i decided at close i-1)
    pos_lag = np.concatenate(([0], pos_arr[:-1])); risk_lag = np.concatenate(([0], risk_arr[:-1]))
    r = (df.close.diff() * pos_lag / np.where(risk_lag > 0, risk_lag, np.nan)).fillna(0.0)
    daily = pd.Series(r.values, index=pd.DatetimeIndex(df.ts)).resample("D").sum()
    for t in trades:
        daily.loc[pd.Timestamp(t["exit_time"]).normalize()] -= (COST_PIPS + SWAP_PIPS_DAY * t["days"]) / t["risk_dist_pips"]
    gap = abs(daily.sum() - sum(t["net_r"] for t in trades))
    assert gap < 0.5, f"{pair}: integrity gap {gap:.2f}R"
    return trades, daily

def gates(weekly_pct, yearly_r):  # weekly_pct ALREADY percent (double-scale law)
    eq = (1 + weekly_pct / 100.0).cumprod()
    dd = (eq / eq.cummax() - 1).min() * 100
    return dict(posW=(weekly_pct > 0).mean() * 100, worstW=weekly_pct.min(), maxDD=dd,
                ann=weekly_pct.mean() * 52, y2024=yearly_r.get(2024, 0.0),
                y2025=yearly_r.get(2025, 0.0), y2026=yearly_r.get(2026, 0.0))

print("=" * 74)
print("R109 — F3 CTA-HORIZON TREND | daily close-only Donchian | 8 majors | 0.5%/pos")
print("=" * 74)
data = {p: load(p) for p in PAIRS}

sel_scores = []
for N in (20, 60, 120, 252):
    daily_sum = None; ntr = 0
    for p in PAIRS:
        tr, daily = run_variant(p, data[p], N)
        d = daily[daily.index >= T0]
        ntr += len([t for t in tr if pd.Timestamp(t["exit_time"]) >= T0])
        daily_sum = d if daily_sum is None else daily_sum.add(d, fill_value=0)
    sel = daily_sum[daily_sum.index <= SEL_END]
    weekly = sel.resample("W").sum() * RISK_PCT * 100
    yearly_sel = sel.groupby(sel.index.year).sum().to_dict()
    g = gates(weekly, yearly_sel)
    passed = sum([g["posW"] >= 65, g["worstW"] >= -3, g["maxDD"] >= -12,
                  g["y2024"] >= 0, g["y2025"] >= 0, g["y2026"] >= 0])
    sel_scores.append((passed, g["ann"], N))
    print(f"SEL N={N:3d}d: trades={ntr:4d} posW={g['posW']:5.1f}% worstW={g['worstW']:+6.2f}% "
          f"maxDD={g['maxDD']:+6.2f}% ann={g['ann']:+5.2f}% | R {g['y2024']:+6.1f}/{g['y2025']:+6.1f}/{g['y2026']:+6.1f} | gates {passed}/6")
sel_scores.sort(reverse=True)
_, _, best_N = sel_scores[0]
print(f"\n>>> SELECTED: N={best_N}d (selection window only)")

all_trades = []; daily_full = None
for p in PAIRS:
    tr, daily = run_variant(p, data[p], best_N)
    all_trades += tr
    d = daily[daily.index >= T0]
    daily_full = d if daily_full is None else daily_full.add(d, fill_value=0)
weekly = daily_full.resample("W").sum() * RISK_PCT * 100
yearly_r = daily_full.groupby(daily_full.index.year).sum().to_dict()
g = gates(weekly, yearly_r)
tdf = pd.DataFrame(all_trades); tdf.to_csv(OUT / "r109_cta_trades.csv", index=False)
weekly.rename("pct").to_csv(OUT / "r109_cta_weekly.csv", index=True)

print("\n--- FULL TEST WINDOW 2024-01 -> 2026-10 at 0.5%/pos ---")
tw = tdf[pd.to_datetime(tdf.exit_time) >= T0]
print(f"trades={len(tw)} WR={(tw.net_r > 0).mean() * 100:.1f}% avgR={tw.net_r.mean():+.3f} medHold={tw.days.median():.0f}d")
print(f"posW={g['posW']:.1f}% worstW={g['worstW']:+.2f}% maxDD={g['maxDD']:+.2f}% ann={g['ann']:+.2f}%")
print(f"YEARLY LEDGER (R): 2024 {g['y2024']:+.1f} | 2025 {g['y2025']:+.1f} | 2026 {g['y2026']:+.1f}")
gates_d = dict(posW=g["posW"] >= 65, worstW=g["worstW"] >= -3, maxDD=g["maxDD"] >= -12,
               y2024=g["y2024"] >= 0, y2025=g["y2025"] >= 0, y2026=g["y2026"] >= 0)
print("GATES:", {k: ("PASS" if v else "FAIL") for k, v in gates_d.items()}, f"-> {sum(gates_d.values())}/6")
hold = daily_full[daily_full.index >= HOLD_START]
ht = tw[pd.to_datetime(tw.exit_time) >= HOLD_START]
hpf = (ht.net_r[ht.net_r > 0].sum() / max(1e-9, -ht.net_r[ht.net_r < 0].sum())) if len(ht) else float("nan")
print(f"HOLDOUT 2026-04->10: trades={len(ht)} R={hold.sum():+.1f} PF@cost={hpf:.2f}")

# soft anchor: USDJPY net-long during Jan-Jul 2024 (documented yen slide)
uj = tdf[(tdf.pair == "USDJPY") & (tdf.N == best_N)]
days_long = sum(t["days"] * t["dir"] for t in uj.to_dict("records")
                if pd.Timestamp(t["entry_time"]) < pd.Timestamp("2024-08-01")
                and pd.Timestamp(t["exit_time"]) >= pd.Timestamp("2024-01-01"))
print(f"ANCHOR USDJPY Jan-Jul2024 net dir-days: {days_long:+.0f} ({'net LONG ✓' if days_long > 0 else 'NOT long ✗ inspect'})")

# supplementary context 2018->2023 (NOT gated)
ctx = None
for p in PAIRS:
    _, daily = run_variant(p, data[p], best_N)
    d = daily[(daily.index >= "2018-01-01") & (daily.index < "2024-01-01")]
    ctx = d if ctx is None else ctx.add(d, fill_value=0)
yc = ctx.groupby(ctx.index.year).sum()
print("CONTEXT (not gated) yearly R 2018-2023:", {int(y): round(v, 1) for y, v in yc.items()})

print("\nPER-PAIR R LEDGER (test window):")
for p in PAIRS:
    pt = tw[tw.pair == p]
    print(f"  {p}: n={len(pt):3d} R={pt.net_r.sum():+7.1f} WR={(pt.net_r > 0).mean() * 100:.0f}%")
