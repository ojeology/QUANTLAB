#!/usr/bin/env python3
"""R107 — FX REVIVAL: THE THREE-YEAR TEST (F1: Donchian trend basket)
Charter: FOREX_HUNT_BEGINS.md 'REVIVAL' section — gates locked BEFORE this run.
Close-only fills; costs 1.2 pip RT + 0.7 pip/day swap haircut; risk 0.5%/pair-position.
Selection 2024-01->2026-03 picks variant N in {80,120,200}; holdout 2026-04->10 untouchcd until verdict.
"""
import pathlib, sys
import numpy as np
import pandas as pd

CACHE = pathlib.Path("quantlab_cache/forex")
OUT = pathlib.Path("quantlab_output"); OUT.mkdir(exist_ok=True)
PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP"]
PIP = {p: 0.01 for p in PAIRS}; PIP["USDJPY"] = 0.01
for p in PAIRS:
    if p != "USDJPY": PIP[p] = 0.0001
COST_PIPS = 1.2          # round-trip, conservative
SWAP_PIPS_DAY = 0.7      # haircut per open day (direction-agnostic approximation)
RISK_PCT = 0.005         # 0.5% account risk per open pair-position
RISK_ATR = 2.5           # 1R = 2.5 * ATR24 at entry
ATR_N = 24
SEL_END = pd.Timestamp("2026-03-31")   # selection window end (inclusive)
HOLD_START = pd.Timestamp("2026-04-01")

def load(pair):
    df = pd.read_csv(CACHE / f"{pair}_1H.csv", parse_dates=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    tr = pd.concat([df.high - df.low,
                    (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_N).mean()
    # --- bad-tick scrub (Yahoo FX occasionally prints impossible bars) ---
    bad = (df.close.diff().abs() > 12 * df.atr) | ((df.high - df.low) > 12 * df.atr)
    n_bad = int(bad.sum())
    if n_bad:
        prev = df["close"].shift()
        df.loc[bad, ["open", "high", "low", "close"]] = np.tile(prev[bad].values[:, None], (1, 4))
        print(f"  [scrub] {pair}: {n_bad} bad-tick bar(s) flattened to prev close")
    tr = pd.concat([df.high - df.low,
                    (df.high - df.close.shift()).abs(),
                    (df.low - df.close.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_N).mean()
    return df

def run_variant(pair, df, N):
    """Close-only Donchian state machine. Returns trades + daily R series."""
    hh = df.high.rolling(N).max().shift(1)   # STRICTLY previous N bars (causal)
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
            if pos != 0:  # close current trade at THIS bar close (close-only law)
                j = i; days = (ts[j] - ts[entry_i]) / np.timedelta64(1, "D")
                pips = (c[j] - c[entry_i]) * pos / PIP[pair]
                rd = 2.5 * atr[entry_i] / PIP[pair]
                cost_r = (COST_PIPS + SWAP_PIPS_DAY * days) / rd
                trades.append(dict(pair=pair, N=N, entry_time=ts[entry_i], exit_time=ts[j],
                                   dir=pos, entry=c[entry_i], exit=c[j], days=days,
                                   pips=pips, risk_dist_pips=rd, net_r=(pips / rd) - cost_r))
            if newpos != 0:
                entry_i = i
            pos = newpos
        pos_arr[i] = pos
        risk_arr[i] = (2.5 * atr[entry_i] if pos != 0 and entry_i is not None else 0.0)
    # close any open trade at final bar
    if pos != 0 and entry_i is not None:
        j = len(df) - 1
        days = (ts[j] - ts[entry_i]) / np.timedelta64(1, "D")
        pips = (c[j] - c[entry_i]) * pos / PIP[pair]
        rd = 2.5 * atr[entry_i] / PIP[pair]
        trades.append(dict(pair=pair, N=N, entry_time=ts[entry_i], exit_time=ts[j], dir=pos,
                           entry=c[entry_i], exit=c[j], days=days, pips=pips,
                           risk_dist_pips=rd, net_r=(pips / rd) - (COST_PIPS + SWAP_PIPS_DAY * days) / rd))
    # daily R (MTM at closes; costs+swap charged on exit day — disclosed)
    # OFF-BY-ONE GUARD: the position held DURING bar i is the one set at close[i-1]
    pos_lag = np.concatenate(([0], pos_arr[:-1]))
    risk_lag = np.concatenate(([0], risk_arr[:-1]))
    r = (df.close.diff() * pos_lag / np.where(risk_lag > 0, risk_lag, np.nan)).fillna(0.0)
    daily = pd.Series(r.values, index=pd.DatetimeIndex(df.ts)).resample("D").sum()
    for t in trades:  # subtract full costs on exit day
        daily.loc[pd.Timestamp(t["exit_time"]).normalize()] -= (COST_PIPS + SWAP_PIPS_DAY * t["days"]) / t["risk_dist_pips"]
        daily.loc[pd.Timestamp(t["entry_time"]).normalize()] += 0.0
    # INTEGRITY LAW: daily curve must equal the trade ledger (catches any MTM drift)
    gap = abs(daily.sum() - sum(t["net_r"] for t in trades))
    assert gap < 0.5, f"{pair}: daily/trades integrity gap {gap:.2f}R"
    return trades, daily

def gates(weekly_pct, yearly_r):
    # weekly_pct is ALREADY in percent (R104/R105 lesson: never re-scale percent input)
    eq = (1 + weekly_pct / 100.0).cumprod()
    dd = (eq / eq.cummax() - 1).min() * 100
    return dict(posW=(weekly_pct > 0).mean() * 100, worstW=weekly_pct.min(),
                maxDD=dd, ann=weekly_pct.mean() * 52,
                y2024=yearly_r.get(2024, 0.0), y2025=yearly_r.get(2025, 0.0), y2026=yearly_r.get(2026, 0.0))

def evaluate(daily):
    weekly = daily.resample("W").sum() * RISK_PCT * 100
    yearly_r = daily.groupby(daily.index.year).sum().to_dict()
    return gates(weekly, yearly_r), weekly, yearly_r

print("=" * 74)
print("R107 FX REVIVAL — F1 Donchian basket | close-only | 1.2pip+0.7pip/day | 0.5%/pos")
print("=" * 74)
data = {p: load(p) for p in PAIRS}

# ---- SELECTION WINDOW ONLY: pick variant ----
sel_scores = []
for N in (80, 120, 200):
    daily_sum = None
    ntr = 0
    for p in PAIRS:
        tr, daily = run_variant(p, data[p], N)
        daily = daily[daily.index <= SEL_END]
        ntr += len([t for t in tr if pd.Timestamp(t["entry_time"]) <= SEL_END])
        daily_sum = daily if daily_sum is None else daily_sum.add(daily, fill_value=0)
    g, _, _ = evaluate(daily_sum)
    passed = sum([g["posW"] >= 65, g["worstW"] >= -3, g["maxDD"] >= -12,
                  g["y2024"] >= 0, g["y2025"] >= 0, g["y2026"] >= 0])
    sel_scores.append((passed, g["ann"], N, g, ntr))
    print(f"SEL N={N:3d}: trades={ntr:4d} posW={g['posW']:.1f}% worstW={g['worstW']:.2f}% "
          f"maxDD={g['maxDD']:.2f}% ann={g['ann']:+.2f}% | R: {g['y2024']:+.1f}/{g['y2025']:+.1f}/{g['y2026']:+.1f} | gates={passed}/6")
sel_scores.sort(reverse=True)
best_passed, _, best_N, best_sel_g, _ = sel_scores[0]
print(f"\n>>> SELECTED VARIANT: N={best_N} (selection window only, {best_passed}/6 gates)")

# ---- FULL WINDOW + HOLDOUT verdict on selected variant ----
all_trades = []; daily_full = None
for p in PAIRS:
    tr, daily = run_variant(p, data[p], best_N)
    all_trades += tr
    daily_full = daily if daily_full is None else daily_full.add(daily, fill_value=0)
g_full, weekly, yearly_r = evaluate(daily_full)
hold = daily_full[daily_full.index >= HOLD_START]
g_hold, _, _ = evaluate(hold)
hold_tr = [t for t in all_trades if pd.Timestamp(t["exit_time"]) >= HOLD_START]
hold_pf = (sum(t["net_r"] for t in hold_tr if t["net_r"] > 0) /
           max(1e-9, -sum(t["net_r"] for t in hold_tr if t["net_r"] < 0)))

tdf = pd.DataFrame(all_trades); tdf.to_csv(OUT / "r107_fx_trades.csv", index=False)
weekly.rename("pct").to_csv(OUT / "r107_fx_weekly.csv", index=True)

print("\n--- FULL WINDOW (2024-01 -> 2026-10) at 0.5%/pos ---")
print(f"trades={len(tdf)}  WR={(tdf.net_r > 0).mean() * 100:.1f}%  avgR={tdf.net_r.mean():+.3f}  medDays={tdf.days.median():.1f}")
print(f"posW={g_full['posW']:.1f}%  worstW={g_full['worstW']:.2f}%  maxDD={g_full['maxDD']:.2f}%  ann={g_full['ann']:+.2f}%")
print(f"YEARLY LEDGER (R): 2024 {g_full['y2024']:+.1f} | 2025 {g_full['y2025']:+.1f} | 2026 {g_full['y2026']:+.1f}")
gates_full = dict(posW=g_full["posW"] >= 65, worstW=g_full["worstW"] >= -3, maxDD=g_full["maxDD"] >= -12,
                  y2024=g_full["y2024"] >= 0, y2025=g_full["y2025"] >= 0, y2026=g_full["y2026"] >= 0)
print("GATES:", {k: ("PASS" if v else "FAIL") for k, v in gates_full.items()},
      f"-> {sum(gates_full.values())}/6")
print("\n--- HOLDOUT (2026-04 -> 10-09, untouched until now) ---")
print(f"trades={len(hold_tr)}  R_total={hold.sum():+.1f}  PF@cost={hold_pf:.2f}")
print("\nPER-PAIR R LEDGER (full):")
for p in PAIRS:
    pt = tdf[tdf.pair == p]
    print(f"  {p}: n={len(pt):3d} R={pt.net_r.sum():+7.1f}  WR={(pt.net_r > 0).mean() * 100:.0f}%")
