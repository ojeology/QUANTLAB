# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 282 | 17.0 | 56% | 2.20 | 1.86 | -13.7% | 71% | 2 | 1.57 |
| q0.65 | 320 | 19.2 | 53% | 2.01 | 1.69 | -19.0% | 59% | 2 | 1.51 |
| q0.75 | 332 | 19.9 | 53% | 1.96 | 1.65 | -19.0% | 65% | 2 | 1.52 |
| q0.85 | 340 | 20.4 | 53% | 1.93 | 1.63 | -19.8% | 65% | 2 | 1.51 |
| q0.95 | 348 | 20.9 | 52% | 1.89 | 1.59 | -18.4% | 65% | 2 | 1.45 |
| q1.0 | 504 | 15.9 | 46% | 1.51 | 1.29 | -32.0% | 53% | 3 | 1.44 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.