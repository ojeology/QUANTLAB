# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 304 | 18.2 | 52% | 2.18 | 1.84 | -18.3% | 69% | 2 | 1.65 |
| q0.65 | 331 | 19.8 | 50% | 2.05 | 1.73 | -17.5% | 59% | 2 | 1.61 |
| q0.75 | 339 | 20.3 | 50% | 2.02 | 1.71 | -18.2% | 59% | 2 | 1.60 |
| q0.85 | 349 | 20.9 | 49% | 1.95 | 1.66 | -18.3% | 65% | 2 | 1.55 |
| q0.95 | 355 | 21.2 | 49% | 1.95 | 1.66 | -18.3% | 65% | 2 | 1.53 |
| q1.0 | 510 | 16.0 | 43% | 1.50 | 1.29 | -39.0% | 44% | 7 | 1.52 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.