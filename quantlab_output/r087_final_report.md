# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 313 | 17.8 | 55% | 1.85 | 1.54 | -18.2% | 71% | 2 | 1.48 |
| q0.65 | 349 | 19.8 | 56% | 1.88 | 1.56 | -18.2% | 67% | 2 | 1.46 |
| q0.75 | 370 | 21.0 | 54% | 1.75 | 1.45 | -19.2% | 56% | 2 | 1.34 |
| q0.85 | 380 | 21.5 | 53% | 1.72 | 1.43 | -19.8% | 56% | 2 | 1.33 |
| q0.95 | 386 | 21.8 | 54% | 1.73 | 1.44 | -19.2% | 56% | 2 | 1.32 |
| q1.0 | 540 | 16.5 | 49% | 1.43 | 1.21 | -29.0% | 45% | 3 | 1.32 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.