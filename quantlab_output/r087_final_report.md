# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 277 | 17.3 | 59% | 2.18 | 1.82 | -15.5% | 71% | 2 | 1.59 |
| q0.65 | 309 | 19.3 | 58% | 2.04 | 1.70 | -17.1% | 71% | 2 | 1.45 |
| q0.75 | 332 | 20.7 | 55% | 1.86 | 1.55 | -18.8% | 59% | 2 | 1.40 |
| q0.85 | 342 | 21.2 | 55% | 1.83 | 1.52 | -18.8% | 59% | 2 | 1.37 |
| q0.95 | 349 | 21.7 | 55% | 1.83 | 1.53 | -18.8% | 59% | 2 | 1.37 |
| q1.0 | 502 | 16.1 | 49% | 1.46 | 1.24 | -28.6% | 47% | 3 | 1.37 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.