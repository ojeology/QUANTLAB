#!/usr/bin/env python3
# R120 — OI-CHANGE MOMENTUM · 2023 DISCOVERY LEG (prespec: docs/RESEARCH_FRONTIER_R120.md, LOCKED)
# 5-day % change of open interest -> L top quartile / S bottom quartile, weekly, 2023 only.
import os, sys, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ql_accounting import LadderAccount, performance_report, SIDE_RATE_DEFAULT

BN = "quantlab_cache/bnfut"
MM = "quantlab_cache/bnmetrics"
LOOKBACK = 5                 # REGISTERED signal: 5 trading days
FREQ = 7                     # weekly formations
FIRST_FORM = pd.Timestamp("2023-01-08")
LAST_FORM  = pd.Timestamp("2023-12-24")
MIN_COINS  = 12
FRESH_DAYS = 3
MIN_HIST   = 30

syms = sorted(os.path.basename(f).split("_")[0] for f in glob.glob(f"{MM}/*_oi_daily.csv"))
print(f"R120 discovery leg — panel: {len(syms)} coins | lookback {LOOKBACK}d | weekly formations")

# ── data ─────────────────────────────────────────────────────────────────────
closes, oi_lev, fund_daily = {}, {}, {}
for s in syms:
    p = pd.read_csv(f"{BN}/{s}_1D.csv", parse_dates=["date"]).set_index("date")["close"]
    p = p[(p.index >= "2022-11-01") & (p.index <= "2023-12-31")]
    if len(p) < 100: continue
    closes[s] = p
    m = pd.read_csv(f"{MM}/{s}_oi_daily.csv", parse_dates=["date"]).set_index("date")[["oi"]]
    m["obs_date"] = m.index                                  # true observation date (pre-ffill)
    oi_lev[s] = m
    f = pd.read_csv(f"{BN}/{s}_fund.csv", parse_dates=None)
    f["date"] = pd.to_datetime(f.ts_ms, unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    fund_daily[s] = f.groupby("date").rate.sum()

PX = pd.DataFrame(closes).sort_index()                       # prices (raw calendar of perp dailies)
D = PX.index[(PX.index >= "2023-01-01") & (PX.index <= "2023-12-31")]
RETS = PX.pct_change()
print(f"window: {D[0].date()} → {D[-1].date()} | {len(D)} daily bars")

def funding_of(coin, date):
    return float(fund_daily.get(coin, pd.Series(dtype=float)).get(date, 0.0))

# ── formation builder (single registered cell + sensitivities) ───────────────
def build_weights(lookback, exclude=()):
    forms = []                                               # (i_signal_bar, {coin: w}, n_elig, quartile_map)
    form_dates = pd.date_range(FIRST_FORM, LAST_FORM, freq=f"{FREQ}D")
    for fd in form_dates:
        # map formation date to the last price bar ≤ fd (grid is daily; exact match expected)
        cand = D[D <= fd]
        if not len(cand): continue
        t = cand[-1]; i_t = D.get_loc(t)
        if i_t == 0: continue
        hist_idx = D.get_loc(t)
        rows = {}
        for c in PX.columns:
            if c in exclude: continue
            m = oi_lev[c]
            sub = m[m.index <= t]
            if not len(sub): continue
            obs_last = sub.index[-1]
            if (t - obs_last).days > FRESH_DAYS: continue     # freshness
            oi_now = float(sub.oi.iloc[-1])
            if len(sub) < lookback + 1: continue
            oi_past = float(sub.oi.iloc[-1 - lookback])
            if not np.isfinite(oi_now) or not np.isfinite(oi_past) or oi_past <= 0: continue
            if hist_idx < MIN_HIST: continue
            rows[c] = oi_now / oi_past - 1.0
        if len(rows) < MIN_COINS:
            forms.append((i_t, {}, len(rows), {})); continue
        r = pd.Series(rows).sort_values(ascending=False)
        q = max(3, len(r) // 4)
        wmap, qmap = {}, {}
        mid = list(r.index[q:len(r) - q])
        for rank, c in enumerate(r.index):
            if rank < q:
                wmap[c] = 1.0 / q; qmap[c] = 1
            elif rank >= len(r) - q:
                wmap[c] = -1.0 / q; qmap[c] = 4
            else:
                qmap[c] = 2 if mid.index(c) < len(mid) // 2 else 3
        forms.append((i_t, wmap, len(r), qmap))
    return forms

def run_backtest(lookback=LOOKBACK, exclude=(), side_rate=SIDE_RATE_DEFAULT):
    forms = build_weights(lookback, exclude)
    form_by_bar = {i_t: (w, n, qm) for i_t, w, n, qm in forms}
    acct = LadderAccount(D, leverage=1.0, side_rate=side_rate)
    prev_form_w = None
    quartile_rows = []
    for i in range(len(D)):
        if i - 1 in form_by_bar:
            w, n, qm = form_by_bar[i - 1]
            if w: acct.rebalance(i - 1, w)
            prev_form_w = (i - 1, w, qm)
        if prev_form_w and prev_form_w[0] == i - 1 and prev_form_w[2]:
            # per-coin forward weekly returns for quartile monotonicity (G3): close(i) / close(i-1+1) - 1 over 7 bars
            t0 = i
            t1 = min(i + FREQ - 1, len(D) - 1)
            if t1 > t0:
                for c, q in prev_form_w[2].items():
                    p0, p1 = PX[c].get(D[t0]), PX[c].get(D[t1])
                    if p0 and p1:
                        quartile_rows.append((D[i - 1], q, c, p1 / p0 - 1))
        w_eff = acct.effective_weights(i)
        gross = 0.0; fdrag = 0.0
        for c, w in w_eff.items():
            r = RETS[c].get(D[i], 0.0)
            r = 0.0 if pd.isna(r) else r
            gross += w * r
            fdrag += w * funding_of(c, D[i])
        if i == 0:
            acct.mark(0, 0.0, 0.0)                            # bar 0: flat (first formation later)
        else:
            acct.mark(i, gross, fdrag)
    eq = acct.equity_returns()                                # invariants verified inside
    return eq, acct, quartile_rows, forms

# ── REGISTERED CELL ──────────────────────────────────────────────────────────
eq, acct, qrows, forms = run_backtest()
rep = performance_report(eq)
wk = eq.resample("W").sum()

rng = np.random.default_rng(120)
boot = [wk.values[rng.integers(0, len(wk), len(wk))].sum() for _ in range(5000)]
p_pos = float(np.mean(np.array(boot) > 0))
sharpe = rep["sharpe"]

qm = pd.DataFrame(qrows, columns=["formation", "quartile", "coin", "fwd_ret"])
qmeans = qm.groupby("quartile").fwd_ret.mean()
mono = bool(len(qmeans) == 4 and qmeans.iloc[0] > qmeans.iloc[1] and
            qmeans.iloc[1] > qmeans.iloc[2] and qmeans.iloc[2] > qmeans.iloc[3])

loao = {}
for c in PX.columns:
    try:
        eqc, _, _, _ = run_backtest(exclude=(c,))
        loao[c] = float(eqc.sum())
    except Exception as e:
        loao[c] = np.nan
loao_min = np.nanmin([v for v in loao.values() if np.isfinite(v)])

# sensitivities (NON-selecting)
sens = {}
for lb in (3, 10):
    eqlb, _, _, _ = run_backtest(lookback=lb)
    sens[lb] = dict(total=float(eqlb.sum() * 100), sharpe=float(performance_report(eqlb)["sharpe"]))

# 0.05%/side sensitivity (prespec literal rate)
eq5, _, _, _ = run_backtest(side_rate=0.0005)
rep5 = performance_report(eq5)

print("\n" + "=" * 96)
print("  R120 — OI-CHANGE MOMENTUM · 2023 DISCOVERY (net of costs+funding, L top-ΔOI / S bottom-ΔOI)")
print("=" * 96)
print(f"  formations: {sum(1 for _, w, _, _ in forms if w)}/{len(forms)} executed | coins/formula-eligible avg "
      f"{np.mean([n for _, _, n, _ in forms]):.1f}")
print(f"  REGISTERED 5d cell @ {SIDE_RATE_DEFAULT:.4%}/side:")
print(f"    total net {rep['net_total']:+.2f}%  | ann {rep['ann_cagr']:+.1f}%  vol {rep['ann_vol']:.1f}%  "
      f"Sharpe {sharpe:+.2f}  maxDD {rep['maxDD']:+.1f}%")
print(f"    bootstrap P(net>0) = {p_pos:.3f}  (weekly blocks, 5000 draws)")
print(f"    quartile means (gross fwd wk): " + "  ".join(f"Q{q}:{qmeans.get(q, np.nan)*100:+.2f}%" for q in (1, 2, 3, 4)))
print(f"    monotonic Q1>Q2>Q3>Q4: {'YES' if mono else 'NO'}")
print(f"    LOAO min net total: {loao_min*100:+.2f}%  (worst excluded coin: "
      f"{min((v for v in loao.items() if np.isfinite(v[1])), key=lambda kv: kv[1])[0]})")
print(f"  sensitivities (non-selecting): " + " | ".join(f"{lb}d: {v['total']:+.1f}% Sh {v['sharpe']:+.2f}" for lb, v in sens.items()))
print(f"  cost sensitivity @0.05%/side: total {rep5['net_total']:+.2f}%  Sharpe {rep5['sharpe']:+.2f}")

g1 = rep["net_total"] > 0 and p_pos >= 0.95
g2 = sharpe >= 1.0
g3 = mono
g4 = loao_min > 0
print(f"\n  GATES: G1 {p_pos:.3f}/{rep['net_total']:+.1f}% -> {'PASS' if g1 else 'FAIL'} | "
      f"G2 {sharpe:.2f} -> {'PASS' if g2 else 'FAIL'} | G3 {'PASS' if g3 else 'FAIL'} | "
      f"G4 {loao_min*100:+.1f}% -> {'PASS' if g4 else 'FAIL'}")
if g1 and g2 and g3 and g4:
    verdict = "PROMISING FOR FURTHER VALIDATION"
elif not (g1 and g2 and g4) and any(sens[lb]["sharpe"] >= 1.0 for lb in sens):
    verdict = "INCONCLUSIVE (5d failed; sensitivity passed — do NOT switch lookback without a new prespec)"
else:
    verdict = "REJECT"
print(f"\n  VERDICT: {verdict}")

eq.to_csv("quantlab_output/r120_equity.csv", header=["net_spread_ret"])
qm.to_csv("quantlab_output/r120_quartiles.csv", index=False)
pd.DataFrame([(pd.Timestamp(D[i1]).date(), len(w), ",".join(sorted(w)) if w else "-")
              for i1, w, n, _ in forms], columns=["formation", "n_eligible", "longs_shorts"]) \
    .to_csv("quantlab_output/r120_formations.csv", index=False)
pd.DataFrame([{"coin": c, "loao_net_pct": v * 100} for c, v in loao.items()]) \
    .to_csv("quantlab_output/r120_loao.csv", index=False)
print("saved: r120_equity.csv r120_quartiles.csv r120_formations.csv r120_loao.csv")
