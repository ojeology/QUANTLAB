# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 277 | 16.6 | 53% | 2.21 | 1.87 | -18.3% | 69% | 2 | 1.60 |
| q0.65 | 315 | 18.9 | 52% | 2.13 | 1.81 | -17.5% | 65% | 2 | 1.63 |
| q0.75 | 332 | 19.9 | 51% | 2.06 | 1.75 | -18.4% | 71% | 2 | 1.60 |
| q0.85 | 343 | 20.6 | 50% | 1.98 | 1.68 | -18.4% | 65% | 2 | 1.58 |
| q0.95 | 349 | 20.9 | 50% | 1.98 | 1.68 | -19.2% | 65% | 2 | 1.57 |
| q1.0 | 502 | 15.8 | 43% | 1.52 | 1.30 | -39.0% | 44% | 7 | 1.57 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.