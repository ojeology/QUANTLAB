# R095 — ADVANCED ML on 5m with SIMPLE indicators

**Date:** 2026-08-07 | pooled 5 simple 5m signals, walk-forward LR/SVM/GB, top-q=0.5 | selection ≤May, holdout Jun-Aug, cost gate 0.05%


## Results

| Model | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF | holPF@cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RAW_pool | 6096 | 1354.7 | 40% | 1.00 | 0.39 | -88.5% | 60% | 2 | 0.98 | 1.02 | 0.42 |
| LR | 2878 | 639.6 | 40% | 0.99 | 0.39 | -84.7% | 60% | 2 | 0.89 | 1.06 | 0.46 |
| SVM | 3128 | 695.1 | 40% | 0.99 | 0.41 | -83.3% | 60% | 2 | 0.94 | 1.02 | 0.46 |
| GB | 3565 | 792.2 | 41% | 1.03 | 0.39 | -69.0% | 80% | 1 | 0.97 | 1.06 | 0.43 |

## Verdict

**❌ No advanced-ML config survives 0.05% costs on holdout.** 7th independent 5m confirmation — ML can't create edge where raw PF≈1.0.