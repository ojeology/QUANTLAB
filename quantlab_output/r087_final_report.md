# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 303 | 17.9 | 59% | 2.17 | 1.80 | -17.5% | 67% | 3 | 1.58 |
| q0.65 | 334 | 19.7 | 57% | 2.03 | 1.68 | -17.1% | 61% | 2 | 1.49 |
| q0.75 | 357 | 21.0 | 55% | 1.83 | 1.51 | -18.0% | 61% | 2 | 1.42 |
| q0.85 | 369 | 21.6 | 54% | 1.79 | 1.49 | -19.4% | 61% | 2 | 1.38 |
| q0.95 | 374 | 21.9 | 55% | 1.80 | 1.49 | -18.2% | 61% | 2 | 1.38 |
| q1.0 | 526 | 16.4 | 49% | 1.47 | 1.23 | -28.6% | 48% | 3 | 1.38 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.