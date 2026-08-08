# FOREX F002 — daily context + ML filter

**Date:** 2026-08-08 | 8 majors, 1H, 2yr | selection ≤ Aug-2025, holdout untouched


## Results

| Config | n | t/mo | WR | PF | PF@cost | MDD% | prof% | worst | selPF | holPF | holPF@cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_raw_pool | 10932 | 455.5 | 39% | 0.96 | 0.72 | -98.4% | 38% | 5 | 1.00 | 0.93 | 0.68 |
| B_daily_trend | 6408 | 267.0 | 43% | 1.11 | 0.85 | -51.4% | 67% | 2 | 1.16 | 1.07 | 0.80 |
| C_daily_adx | 7330 | 305.4 | 38% | 0.94 | 0.71 | -98.2% | 29% | 4 | 0.95 | 0.92 | 0.68 |
| D_ml_svm | 6881 | 286.7 | 40% | 1.00 | 0.76 | -79.7% | 50% | 3 | 1.02 | 0.98 | 0.73 |
| E_ml+trend | 4458 | 185.8 | 43% | 1.13 | 0.87 | -59.2% | 71% | 2 | 1.15 | 1.12 | 0.84 |
| F_hour1218 | 3750 | 156.2 | 37% | 0.88 | 0.68 | -96.3% | 29% | 7 | 0.87 | 0.88 | 0.66 |
| G_hour1218+trend | 2184 | 91.0 | 41% | 1.03 | 0.80 | -54.9% | 54% | 3 | 0.99 | 1.06 | 0.80 |

## Verdict

**❌ No forex config survives retail spreads on holdout.** F002 = run 2 of the hunt.