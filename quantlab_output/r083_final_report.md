# R083 — ML-filter refinement

**Date:** 2026-08-06 | walk-forward logistic regression on Family A raw, keep top-q by P(win), threshold from selection only


## Results

| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |
|---|---|---|---|---|---|---|---|---|---|---|
| ML_q35 | 165 | 6.1 | 64% | 2.62 | 2.22 | -15.8% | 50% | 3 | 4.08 | 1.59 |
| ML_q45 | 190 | 7.0 | 62% | 2.40 | 2.03 | -17.9% | 57% | 2 | 3.62 | 1.60 |
| ML_q50_ref | 208 | 7.7 | 60% | 2.26 | 1.91 | -17.4% | 64% | 2 | 3.58 | 1.47 |
| ML_q55 | 215 | 8.0 | 61% | 2.34 | 1.97 | -17.4% | 64% | 2 | 3.75 | 1.47 |
| ML50_dt | 48 | 1.8 | 85% | 8.79 | 7.47 | -3.9% | 70% | 3 | 49.50 | 2.00 |
| ML50_dbr | 46 | 1.7 | 93% | 21.50 | 18.20 | -1.0% | 100% | 0 | 30.75 | 3.00 |
| ML40_dt | 42 | 1.6 | 88% | 11.10 | 9.47 | -2.0% | 67% | 2 | 999.00 | 1.80 |

## Verdict

**❌ Still no config meets ALL criteria**, but ML filters are the closest ever found (see table).