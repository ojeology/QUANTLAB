# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 295 | 17.8 | 62% | 2.07 | 1.71 | -12.5% | 65% | 2 | 1.59 |
| q0.65 | 329 | 19.8 | 61% | 1.99 | 1.63 | -14.2% | 71% | 1 | 1.48 |
| q0.75 | 347 | 20.8 | 60% | 1.88 | 1.54 | -18.0% | 71% | 1 | 1.49 |
| q0.85 | 354 | 21.1 | 60% | 1.88 | 1.54 | -17.0% | 71% | 1 | 1.47 |
| q0.95 | 361 | 21.6 | 59% | 1.81 | 1.48 | -17.8% | 71% | 1 | 1.44 |
| q1.0 | 515 | 16.2 | 54% | 1.48 | 1.23 | -24.1% | 56% | 3 | 1.44 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.