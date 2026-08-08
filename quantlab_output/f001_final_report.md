# F001 — FOREX HUNT #1 (8 majors, 1H)

**Date:** 2026-08-08 | Yahoo 1H data 2023-10→2026-08 | selection <2025-06, holdout 2025-06→2026-08 untouched | spread costs modeled


## Results

| Hyp | n | t/mo | WR | PF | PF@spread | MDD% | prof% | worst | selPF | holPF | holPF@cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| X1_london_break | 2318 | 827.9 | 41% | 1.03 | 0.75 | -40.1% | 56% | 3 | 1.06 | 1.00 | 0.72 |
| X2_trend_pull | 4212 | 1504.3 | 39% | 0.96 | 0.71 | -84.7% | 44% | 6 | 0.99 | 0.92 | 0.66 |
| X3_ny_momentum | 0 | 0.0 | nan% | nan | nan | 0.0% | nan% | nan | nan | nan | nan |
| X4_daylow_mr | 2395 | 855.4 | 38% | 0.92 | 0.68 | -75.5% | 41% | 6 | 0.98 | 0.84 | 0.61 |
| X5_london_exp | 412 | 147.1 | 44% | 1.18 | 0.94 | -15.9% | 50% | 4 | 1.17 | 1.19 | 0.93 |
| T1_family_trans | 0 | 0.0 | nan% | nan | nan | 0.0% | nan% | nan | nan | nan | nan |

## Verdict

**❌ No forex hypothesis survives spread costs on holdout.** F001 honest negative — candidate directions for F002 noted in the run.