# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 285 | 17.1 | 42% | 2.16 | 1.87 | -24.6% | 71% | 2 | 1.81 |
| q0.65 | 315 | 18.9 | 41% | 2.06 | 1.78 | -31.1% | 71% | 2 | 1.79 |
| q0.75 | 323 | 19.4 | 40% | 1.98 | 1.71 | -31.8% | 65% | 2 | 1.71 |
| q0.85 | 336 | 20.1 | 39% | 1.95 | 1.69 | -27.7% | 59% | 2 | 1.65 |
| q0.95 | 343 | 20.6 | 39% | 1.91 | 1.65 | -27.7% | 59% | 2 | 1.65 |
| q1.0 | 503 | 15.8 | 33% | 1.46 | 1.27 | -52.4% | 38% | 7 | 1.61 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.