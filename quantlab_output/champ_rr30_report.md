# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 268 | 16.1 | 45% | 2.41 | 2.09 | -20.7% | 81% | 2 | 2.01 |
| q0.65 | 305 | 18.3 | 40% | 2.01 | 1.74 | -31.1% | 76% | 2 | 1.82 |
| q0.75 | 318 | 19.1 | 40% | 2.01 | 1.74 | -32.5% | 71% | 2 | 1.77 |
| q0.85 | 326 | 19.6 | 40% | 1.97 | 1.71 | -33.2% | 59% | 2 | 1.75 |
| q0.95 | 337 | 20.2 | 39% | 1.92 | 1.66 | -31.2% | 59% | 2 | 1.70 |
| q1.0 | 495 | 15.6 | 33% | 1.48 | 1.29 | -52.4% | 38% | 7 | 1.67 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.