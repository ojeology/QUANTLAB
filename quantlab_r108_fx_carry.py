#!/usr/bin/env python3
"""R108 — F2: FX CARRY (contractual premium) — prespec in FOREX_HUNT_BEGINS.md (R108 section).
Dual rate spine (fresh S1 / frozen-stress S2); 21 synthetic non-USD crosses; long top-K carry;
weekly/monthly rebalance; 0.5%/pos risk; 1.5 pip RT (0.75+0.75) + 0.7 pip/day swap take.
Anchor: week of 2024-08-05 (yen-carry unwind) must be among worst weeks."""
import pathlib
import numpy as np
import pandas as pd

CACHE = pathlib.Path("quantlab_cache/forex"); OUT = pathlib.Path("quantlab_output")
PIP_DEFAULT, PIP_JPY = 0.0001, 0.01
RISK = 0.005; RT_PIPS = 1.5; SWAP_TAKE_PIPS_DAY = 0.7; ATR_N = 24
SEL_END = pd.Timestamp("2026-03-31"); HOLD_START = pd.Timestamp("2026-04-01")
CCYS = ["EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]  # non-USD legs (vs USD majors)
FRED = {"USD": "IRSTCI01USM156N", "EUR": "IRSTCI01EZM156N", "GBP": "IRSTCI01GBM156N",
        "JPY": "IRSTCI01JPM156N", "AUD": "IRSTCI01AUM156N", "CAD": "IRSTCI01CAM156N",
        "CHF": "IRSTCI01CHM156N", "NZD": "IRSTCI01NZM156N"}
DOC = {  # documented central-bank schedules for dead FRED series (audit-printed)
    "CHF": [("2024-03-01", 1.50), ("2024-06-01", 1.25), ("2024-09-01", 1.00),
            ("2024-12-01", 0.50), ("2025-03-01", 0.25), ("2025-06-01", 0.00)],
    "NZD": [("2024-08-01", 5.25), ("2024-10-01", 4.75), ("2024-11-01", 4.25),
            ("2025-02-01", 3.75), ("2025-04-01", 3.50), ("2025-05-01", 3.25),
            ("2025-07-01", 3.00), ("2025-08-01", 2.50)],
}
AUD_DOC = [("2025-02-01", 4.10), ("2025-05-01", 3.85), ("2025-08-01", 3.60)]

def fetch_rates():
    out = {}
    for c, sid in FRED.items():
        f = CACHE / f"rates_{c}.csv"
        if not f.exists():
            import subprocess
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd=2023-01-01"
            subprocess.run(["curl", "-4", "-s", "-m", "30", url, "-o", str(f)], check=True)
        df = pd.read_csv(f, names=["d", "r"], header=0)
        df["d"] = pd.to_datetime(df.d); df["r"] = pd.to_numeric(df.r, errors="coerce")
        out[c] = df.dropna().set_index("d")["r"]
        print(f"  rates {c}: {out[c].index[0].date()} -> {out[c].index[-1].date()}  last={out[c].iloc[-1]:.2f}")
    return out

def spine(raw, mode):
    daily = pd.date_range("2023-12-25", "2026-10-09", freq="D")
    sp = {}
    for c, s in raw.items():
        s = s.copy()
        if mode == "stress" and c in ("CHF", "NZD", "AUD"):
            sp[c] = pd.Series(float(s.iloc[-1] if c != "AUD" else s.iloc[0]), index=daily)
            continue
        if c in DOC:
            for d, v in DOC[c]:
                s.loc[max(pd.Timestamp(d), s.index[0]):] = v
        sp[c] = s.reindex(daily).ffill().bfill()
    if mode == "fresh":  # AUD: FRED flat 4.35 flagged suspicious -> documented cuts
        sp["AUD"] = raw["AUD"].reindex(daily).ffill().bfill()
        for d, v in AUD_DOC:
            sp["AUD"].loc[pd.Timestamp(d):] = v
    return sp

MAJORS = {"EUR": "EURUSD", "GBP": "GBPUSD", "AUD": "AUDUSD", "NZD": "NZDUSD",
          "JPY": "USDJPY", "CAD": "USDCAD", "CHF": "USDCHF"}

def load_majors():
    out = {}
    for c, pair in MAJORS.items():
        df = pd.read_csv(CACHE / f"{pair}_1H.csv", parse_dates=["ts"])[["ts", "close"]]
        if c in ("JPY", "CAD", "CHF"):
            df["close"] = 1.0 / df["close"]  # -> ccy per USD
        out[c] = df.sort_values("ts").reset_index(drop=True)
    return out

def gates(weekly_pct, yearly_r):  # weekly_pct ALREADY percent (double-scale law)
    eq = (1 + weekly_pct / 100.0).cumprod()
    dd = (eq / eq.cummax() - 1).min() * 100
    return dict(posW=(weekly_pct > 0).mean() * 100, worstW=weekly_pct.min(), maxDD=dd,
                ann=weekly_pct.mean() * 52, y2024=yearly_r.get(2024, 0.0),
                y2025=yearly_r.get(2025, 0.0), y2026=yearly_r.get(2026, 0.0))

def run(sp, maj, K, reb):
    # ---- panel: all crosses on common timestamps (inner join = close-only law) ----
    pairs = [(a, b) for i, a in enumerate(CCYS) for b in CCYS[i + 1:]]
    panel = None
    for a, b in pairs:
        m = maj[a].merge(maj[b], on="ts", suffixes=("_a", "_b"))
        n = f"{a}{b}"
        t = pd.DataFrame({"ts": m.ts, f"c_{n}": m.close_a / m.close_b})
        t[f"a_{n}"] = t[f"c_{n}"].diff().abs().rolling(ATR_N).mean() * 1.5
        panel = t if panel is None else panel.merge(t, on="ts", how="inner")
    ts_idx = pd.DatetimeIndex(panel.ts)
    names = [f"{a}{b}" for a, b in pairs]
    diffs = {n: (sp[n[:3]].reindex(ts_idx.normalize()).values
                 - sp[n[3:]].reindex(ts_idx.normalize()).values) for n in names}
    C = {n: panel[f"c_{n}"].values for n in names}
    A = {n: panel[f"a_{n}"].values for n in names}
    PIP = {n: (PIP_JPY if "JPY" in n else PIP_DEFAULT) for n in names}
    # rebalance marks = first bar of each ISO week / month (key-change rule: no set/hashing)
    keys = ts_idx.strftime("%G-%V") if reb == "W" else ts_idx.strftime("%Y-%m")
    is_mark = np.concatenate(([True], (keys.values[1:] != keys.values[:-1])))
    # ---- state machine ----
    pos = {n: 0 for n in names}
    pos_arr = {n: np.zeros(len(panel)) for n in names}
    risk_arr = {n: np.zeros(len(panel)) for n in names}
    ledger = []; open_tr = {}
    for i in range(len(panel)):
        if is_mark[i]:
            row = np.array([diffs[n][i] for n in names])
            tradable = np.array([np.isfinite(A[n][i]) and A[n][i] > 0 for n in names])
            valid = row[tradable]
            if len(valid) >= K:
                thresh = np.partition(valid, -K)[-K]
                for j, n in enumerate(names):
                    want = 1 if (tradable[j] and row[j] >= thresh) else 0
                    if want != pos[n]:
                        if pos[n] == 1:
                            e = open_tr.pop(n); b0 = e["bucket"]
                            ledger.append(dict(cross=n, entry_time=e["t"], exit_time=ts_idx[i],
                                               entry=e["px"], exit=C[n][i], bucket=b0,
                                               price_r=(C[n][i] - e["px"]) / b0,
                                               cost_r=(0.75 * PIP[n]) / b0 + (0.75 * PIP[n]) / b0))
                        if want == 1:
                            open_tr[n] = dict(t=ts_idx[i], px=C[n][i], bucket=2.5 * A[n][i])
                            risk_arr[n][i] = 2.5 * A[n][i]
                        pos[n] = want
        for n in names:
            if pos[n] == 1 and not (risk_arr[n][i] > 0):  # NaN or 0 -> carry forward
                risk_arr[n][i] = risk_arr[n][i - 1] if i else 0
            pos_arr[n][i] = pos[n]
    for n, e in open_tr.items():  # mark open trades to final close in ledger (no exit cost double-count)
        i = len(panel) - 1
        ledger.append(dict(cross=n, entry_time=e["t"], exit_time=ts_idx[i], entry=e["px"],
                           exit=C[n][i], bucket=e["bucket"], price_r=(C[n][i] - e["px"]) / e["bucket"],
                           cost_r=(0.75 * PIP[n]) / e["bucket"]))
    # ---- daily components ----
    tot = None
    for n in names:
        idx = ts_idx
        c = C[n]; rl = np.concatenate(([0], risk_arr[n][:-1])); pl = np.concatenate(([0], pos_arr[n][:-1]))
        price_r = pd.Series(np.concatenate(([0], np.diff(c))) * pl / np.where(rl > 0, rl, np.nan), index=idx).fillna(0)
        nf = pd.Series(np.where(rl > 0, RISK * c / rl, 0.0), index=idx)
        carry_r = nf * pd.Series(diffs[n], index=idx) / 100.0 / 365.0
        swap_r = nf * (SWAP_TAKE_PIPS_DAY * PIP[n]) / pd.Series(c, index=idx)
        d = (price_r + carry_r - swap_r).resample("D").sum()
        tot = d if tot is None else tot.add(d, fill_value=0)
    costs = pd.Series(0.0, index=pd.DatetimeIndex(pd.date_range("2023-12-25", "2026-10-09", freq="D")))
    for t in ledger:
        costs.loc[pd.Timestamp(t["entry_time"]).normalize()] -= (0.75 * PIP[t["cross"]]) / t["bucket"]
        costs.loc[pd.Timestamp(t["exit_time"]).normalize()] -= (0.75 * PIP[t["cross"]]) / t["bucket"]
    tot = tot.add(costs, fill_value=0)
    weekly_pct = tot.resample("W").sum() * RISK * 100
    yearly_r = tot.groupby(tot.index.year).sum().to_dict()
    return tot, weekly_pct, yearly_r, pd.DataFrame(ledger)

print("=" * 74)
print("R108 — F2 FX CARRY | 21 synthetic non-USD crosses | dual spine | top-K long-only")
print("=" * 74)
raw = fetch_rates()
maj = load_majors()

results = {}
for mode in ("fresh", "stress"):
    sp = spine(raw, mode)
    for K in (3, 5):
        for reb in ("W", "M"):
            tot, weekly_pct, yearly_r, ledger = run(sp, maj, K, reb)
            sel = weekly_pct[weekly_pct.index <= SEL_END]
            tots = tot[tot.index <= SEL_END]
            yearly_sel = tots.groupby(tots.index.year).sum().to_dict()
            g = gates(sel, yearly_sel)
            passed = sum([g["posW"] >= 65, g["worstW"] >= -3, g["maxDD"] >= -12,
                          g["y2024"] >= 0, g["y2025"] >= 0, g["y2026"] >= 0])
            results[(mode, K, reb)] = (g, weekly_pct, yearly_r, passed, ledger)
            print(f"[{mode:6s}] K={K} {reb}: posW={g['posW']:5.1f}% worstW={g['worstW']:+6.2f}% "
                  f"maxDD={g['maxDD']:+6.2f}% ann={g['ann']:+5.2f}% | R {g['y2024']:+6.1f}/{g['y2025']:+6.1f}/{g['y2026']:+6.1f} | gates {passed}/6")

print("\n--- ROBUSTNESS (selection gates, fresh vs stress) ---")
for K in (3, 5):
    for reb in ("W", "M"):
        print(f"K={K} {reb}: fresh {results[('fresh', K, reb)][3]}/6 | stress {results[('stress', K, reb)][3]}/6")

best_key = max((k for k in results if k[0] == "fresh"),
               key=lambda k: (results[k][3], results[k][0]["ann"]))
g, weekly_pct, yearly_r, passed, ledger = results[best_key]
print(f"\n>>> BEST (fresh spine): K={best_key[1]} reb={best_key[2]} ({passed}/6 selection gates)")
gf = gates(weekly_pct, yearly_r)
print("FULL WINDOW:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in gf.items()})
worst5 = weekly_pct.nsmallest(5)
print("5 worst weeks:", [(str(ix.date()), round(v, 2)) for ix, v in worst5.items()])
anchor = any(ix.month == 8 and ix.year == 2024 for ix in worst5.index)
print("ANCHOR (Aug-2024 unwind in worst-5):", "PRESENT - test sane" if anchor else "MISSING - TEST BROKEN")
hold = weekly_pct[weekly_pct.index >= HOLD_START]
print(f"HOLDOUT: posW={(hold > 0).mean() * 100:.1f}% worst={hold.min():+.2f}% sum={hold.sum():+.2f}%")
print(f"ledger: {len(ledger)} trades | avg price_r={ledger.price_r.mean():+.3f} avg cost_r={-ledger.cost_r.mean():+.3f}")
weekly_pct.rename("pct").to_csv(OUT / "r108_carry_weekly.csv", index=True)
ledger.to_csv(OUT / "r108_carry_trades.csv", index=False)
