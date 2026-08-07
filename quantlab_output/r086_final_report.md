# R086 — ML-TYPE zoo (RF / SVM / MLP / NB / Ensemble)

**Date:** 2026-08-06 | same 14 lean features (11 base + 3 special: breadth-quartile, dist-to-48h-high, green-streak) | walk-forward, top-q=0.55


## Results

| ML type | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |
|---|---|---|---|---|---|---|---|---|---|---|
| LR ⭐ champion | 233 | 8.6 | 60% | 2.22 | 1.87 | -14.1% | 64% | 3 | 3.40 | 1.55 |
| RF | 263 | 9.7 | 57% | 1.96 | 1.63 | -17.4% | 71% | 2 | 3.08 | 1.36 |
| SVM | 249 | 9.2 | 60% | 2.23 | 1.87 | -14.5% | 71% | 2 | 3.71 | 1.48 |
| MLP | 216 | 8.0 | 60% | 2.22 | 1.87 | -14.5% | 64% | 3 | 3.93 | 1.37 |
| NB | 235 | 8.7 | 58% | 2.10 | 1.74 | -17.4% | 71% | 2 | 3.11 | 1.40 |
| ENSEMBLE | 246 | 9.1 | 58% | 2.05 | 1.72 | -17.4% | 64% | 3 | 2.89 | 1.50 |

## Verdict

**✅ SVM meets the retail spec:** 9.2 t/mo, 71% prof-mo, worst 2, PF 2.23 (cost 1.87), holPF 1.48.