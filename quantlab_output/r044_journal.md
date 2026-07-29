# QUANTLAB AI — R044 Research Journal

**Date:** 2026-07-29  
**Research ID:** R044  
**Title:** External Symbol Validation — Portfolio C Generalisation Test  
**Dataset:** 1H · 26 NEW symbols · 451,069 bars · 5-fold WF · OOS only  

---

## Portfolio C (Frozen from R043)

E1: Dist>p75 · Wed-Thu · BodyPct>p60 · US(14-21UTC)  
E2: ADX>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)  
E3: Dist>p75 · Wed-Thu · PrevBody>p67 · US(14-21UTC)  
E4: ADX>p67 · Dist>p60 · Wed-Thu · US(14-21UTC)  

**No parameters changed. Complete strategy freeze.**

## Q1 — PF > 1.20 on New Symbols?

Portfolio C PF = **1.162** (FAIL ✗)

## Q2 — Per-Symbol Report

| Symbol | Trades | Win Rate | PF | Net R |
|--------|--------|----------|----|-------|
| 1INCH | 6 | 16.7% | 0.349 | -0.500 |
| AAVE | 10 | 40.0% | 1.147 | +0.200 |
| ALGO | 3 | 0.0% | 0.000 | -1.000 |
| AXS | 2 | 50.0% | 1.638 | +0.500 |
| CHZ | 8 | 50.0% | 1.762 | +0.500 |
| COMP | 5 | 60.0% | 2.490 | +0.800 |
| CRV | 3 | 0.0% | 0.000 | -1.000 |
| DYDX | 11 | 63.6% | 3.049 | +0.909 |
| EGLD | 0 | 0.0% | 0.000 | +0.000 |
| ETC | 4 | 50.0% | 1.643 | +0.500 |
| FET | 7 | 14.3% | 0.297 | -0.571 |
| GALA | 4 | 50.0% | 1.699 | +0.500 |
| GMX | 5 | 60.0% | 2.416 | +0.800 |
| GRT | 3 | 33.3% | 0.818 | -0.000 |
| HBAR | 2 | 50.0% | 1.645 | +0.500 |
| ICP | 10 | 30.0% | 0.755 | -0.100 |
| IMX | 9 | 55.6% | 2.147 | +0.667 |
| INJ | 5 | 60.0% | 2.535 | +0.800 |
| LDO | 9 | 33.3% | 0.877 | -0.000 |
| SAND | 3 | 33.3% | 0.849 | -0.000 |
| SHIB | 4 | 50.0% | 1.665 | +0.500 |
| SNX | 4 | 50.0% | 1.853 | +0.500 |
| STX | 3 | 66.7% | 3.272 | +1.000 |
| SUSHI | 5 | 20.0% | 0.435 | -0.400 |
| TRX | 3 | 0.0% | 0.000 | -1.000 |
| XLM | 5 | 40.0% | 1.150 | +0.200 |

Profitable symbols: **15/26 (58%)**

## Q3 — Robustness

- Bootstrap median PF > 1.20: 1.162 → ✗ FAIL
- Monte Carlo P(profit) > 60%: 79.900 → ✓ PASS
- LOO-symbol floor > 1.00: 1.063 → ✓ PASS
- LOO-fold floor > 1.00: 0.931 → ✗ FAIL

## Q4 — R043 vs R044 Comparison

| Metric | R043 | R044 | Delta |
|--------|------|------|-------|
| Trades | 273 | 133 | — |
| Win Rate | 48 | 41 | — |
| Profit Factor | 1.512 | 1.162 | -0.351 |
| Bootstrap Median | 1.511 | 1.162 | -0.349 |
| Monte Carlo P% | 99.950 | 79.900 | -20.050 |
| Max Drawdown | -7.260 | -7.281 | -0.021 |
| LOO Symbol Floor | 1.425 | 1.063 | -0.362 |
| LOO Fold Floor | 1.009 | 0.931 | -0.078 |

## Q5 — Generalisation Classification

**OVERFIT**

- PF ratio: 0.768 (-23.2%)
- Symbols profitable: 58%

---

## Final Verdict

**Score: 3/7**  

**Verdict: REJECT**  

**Generalisation: OVERFIT**  

  NO — OVERFIT. Portfolio C does not generalise to new symbols.
  R044 PF=1.162 < 1.00 or robustness has collapsed.
  The R043 results were specific to the original research universe.