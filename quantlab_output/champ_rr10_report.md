# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 313 | 18.8 | 65% | 1.87 | 1.49 | -15.7% | 53% | 3 | 1.52 |
| q0.65 | 339 | 20.4 | 65% | 1.85 | 1.48 | -15.7% | 53% | 3 | 1.51 |
| q0.75 | 347 | 20.9 | 64% | 1.78 | 1.42 | -16.6% | 53% | 3 | 1.48 |
| q0.85 | 358 | 21.4 | 64% | 1.77 | 1.41 | -17.4% | 53% | 3 | 1.46 |
| q0.95 | 363 | 21.7 | 63% | 1.72 | 1.38 | -17.4% | 53% | 3 | 1.43 |
| q1.0 | 517 | 16.3 | 59% | 1.43 | 1.17 | -21.2% | 41% | 5 | 1.41 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.