# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 293 | 17.6 | 59% | 2.16 | 1.81 | -15.8% | 65% | 3 | 1.53 |
| q0.65 | 325 | 19.5 | 57% | 2.01 | 1.67 | -16.3% | 59% | 2 | 1.44 |
| q0.75 | 348 | 20.9 | 55% | 1.82 | 1.52 | -18.0% | 59% | 2 | 1.39 |
| q0.85 | 360 | 21.5 | 54% | 1.73 | 1.44 | -19.4% | 59% | 2 | 1.34 |
| q0.95 | 366 | 21.8 | 54% | 1.77 | 1.47 | -19.4% | 59% | 2 | 1.34 |
| q1.0 | 519 | 16.3 | 49% | 1.45 | 1.22 | -28.6% | 47% | 3 | 1.34 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.