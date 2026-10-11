# R107 FINAL REPORT — FX REVIVAL, F1: Donchian Trend Basket
**The Three-Year Test · 2024 / 2025 / 2026 · verdict: ❌ REJECTED**

---

## What was tested

**Hypothesis (who pays):** Donchian breakout trend on 8 FX majors harvests the behavioral
premium — institutional slow repricing + central-bank smoothing (the documented CTA premium).
**Goal (user bar):** survive all three calendar years with no losing year.

- **Data:** Yahoo 1H, 8 majors (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP),
  2023-12-25 → 2026-10-09 = **three complete years**. 3 bad-tick bars scrubbed (USDJPY 2, NZDUSD 1).
- **Execution:** fills at bar closes only; position held during bar *i* = the one set at close *i−1*.
- **Costs:** 1.2 pip round-trip + 0.7 pip/day swap haircut (conservative for retail).
- **Risk:** 0.5% per pair-position; risk bucket = 2.5×ATR(24h) at entry.
- **Variants:** Donchian N ∈ {80h, 120h, 200h} chosen on selection window 2024-01→2026-03 ONLY;
  holdout 2026-04→10 untouched until verdict.
- **Gates:** posW ≥65% · worst week ≥−3% · maxDD ≤12% · each year ≥0R · holdout ≥0R & PF ≥1.0.

## Result: catastrophic fail — 0/6 gates (best variant)

Selected N=200 (1/6 gates on selection). Full window:

| Metric | Result | Gate | |
|---|---|---|---|
| Trades | 417 (median hold 15d) | — | |
| Win rate | 32.4% | — | |
| **Yearly ledger** | **2024 −84.9R · 2025 −349.9R · 2026 +0.7R** | no losing year | ❌ |
| Positive weeks | 37.0% | ≥65% | ❌ |
| Worst week | −21.2% | ≥−3% | ❌ |
| Max drawdown | −94.7% | ≤12% | ❌ |
| Holdout | −32.4R, PF 1.07 | ≥0R, ≥1.1 | ❌ |

**Every single pair lost** (−6.5R to −98.2R). There is no subset of majors that worked.

## Autopsy — why it died

| Component | Contribution |
|---|---|
| Gross edge (before costs) | **−211R** (−0.51R/trade) |
| Costs + swap | −223R (−0.54R/trade ≈ 18 pips vs 33-pip risk bucket) |
| **Net** | **−434R** |

Two independent killers:
1. **The premium wasn't there.** 2024–2026 majors chopped at the 3–20-day horizon — breakouts
   failed and reverted (WR 32% vs ~45% needed gross). The CTA premium lives at 1–6 MONTH
   horizons; the tested variants never reached it.
2. **Swap is a tax on patience at retail.** 19-day average holds ≈ 13 pips of swap haircut per
   trade against a 33-pip bucket — costs alone would have required a solidly positive gross
   edge just to reach zero.

## Audit trail (three bugs caught and fixed this round — the charter working as designed)

1. **Bad ticks:** Yahoo printed impossible bars (e.g. USDJPY spikes) → scrub filter (>12×ATR → prev close).
2. **Off-by-one MTM:** daily curve initially credited the entry bar's own breakout move to the new
   position (+180R phantom per pair) → lagged position array; now daily ≡ trades ledger (asserted, permanently).
3. **Double-scale:** the consistency printer multiplied already-percent weekly numbers by 100 again
   (worst week "−2118%" = −21.18% ×100; the same disease as R104/R105 — now fixed with an in-line comment law).

## Finding (recorded per charter law)

**FX breakout trend at retail horizons/costs cannot survive three years — it cannot even survive
one.** Combined with F010–F013 (session/bracket patterns, dead) and the R103 law (statistical
edges priced away at retail), the evidence now points one direction: the FX premium that CAN
survive must be **contractual, not behavioral** — i.e. **carry** (the interest-rate differential:
central banks pay you the spread between currencies; that's F2, queued next with a
policy-rate differential table as its data spine). A daily-bar CTA-horizon trend test (F3)
is queued behind it as the honest last word on trend.

## Artifacts

`quantlab_output/r107_fx_trades.csv` (417 trades) · `r107_fx_weekly.csv` · `r107_stdout.log` ·
`scripts/fetch_forex.py` · cache: `quantlab_cache/forex/{PAIR}_1H.csv` (8 pairs, refreshed).
