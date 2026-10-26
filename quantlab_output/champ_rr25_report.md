# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 287 | 17.2 | 46% | 2.14 | 1.83 | -21.5% | 71% | 2 | 1.52 |
| q0.65 | 306 | 18.3 | 46% | 2.12 | 1.81 | -22.3% | 71% | 2 | 1.61 |
| q0.75 | 325 | 19.5 | 44% | 1.97 | 1.69 | -23.5% | 71% | 2 | 1.60 |
| q0.85 | 343 | 20.6 | 43% | 1.88 | 1.62 | -21.6% | 59% | 2 | 1.52 |
| q0.95 | 348 | 20.9 | 43% | 1.93 | 1.65 | -21.3% | 59% | 2 | 1.54 |
| q1.0 | 503 | 15.8 | 37% | 1.47 | 1.28 | -44.7% | 41% | 7 | 1.52 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.