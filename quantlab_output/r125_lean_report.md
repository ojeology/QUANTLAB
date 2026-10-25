# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 309 | 18.6 | 57% | 1.95 | 1.62 | -18.8% | 69% | 2 | 1.41 |
| q0.65 | 318 | 19.1 | 56% | 1.87 | 1.56 | -19.6% | 56% | 2 | 1.37 |
| q0.75 | 331 | 19.9 | 55% | 1.84 | 1.54 | -19.6% | 59% | 2 | 1.36 |
| q0.85 | 346 | 20.8 | 54% | 1.75 | 1.46 | -19.6% | 59% | 2 | 1.36 |
| q0.95 | 354 | 21.2 | 54% | 1.77 | 1.47 | -19.6% | 59% | 2 | 1.36 |
| q1.0 | 511 | 16.1 | 49% | 1.45 | 1.23 | -28.6% | 47% | 3 | 1.35 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.