# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 293 | 17.6 | 57% | 1.96 | 1.64 | -17.9% | 62% | 2 | 1.47 |
| q0.65 | 323 | 19.4 | 56% | 1.88 | 1.57 | -18.4% | 71% | 2 | 1.44 |
| q0.75 | 341 | 20.4 | 55% | 1.83 | 1.52 | -18.4% | 65% | 2 | 1.40 |
| q0.85 | 351 | 21.0 | 54% | 1.78 | 1.48 | -19.0% | 59% | 2 | 1.35 |
| q0.95 | 359 | 21.4 | 55% | 1.79 | 1.49 | -18.4% | 59% | 2 | 1.35 |
| q1.0 | 511 | 16.1 | 49% | 1.45 | 1.23 | -28.6% | 47% | 3 | 1.35 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.