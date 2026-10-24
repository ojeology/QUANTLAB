# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 274 | 16.5 | 48% | 2.27 | 1.96 | -19.6% | 71% | 2 | 1.68 |
| q0.65 | 298 | 17.9 | 47% | 2.17 | 1.86 | -22.0% | 71% | 2 | 1.64 |
| q0.75 | 320 | 19.2 | 44% | 1.96 | 1.68 | -24.3% | 65% | 2 | 1.60 |
| q0.85 | 332 | 19.9 | 44% | 1.93 | 1.65 | -23.2% | 59% | 2 | 1.58 |
| q0.95 | 342 | 20.5 | 44% | 1.94 | 1.67 | -21.3% | 59% | 2 | 1.57 |
| q1.0 | 495 | 15.6 | 37% | 1.48 | 1.29 | -44.7% | 41% | 7 | 1.55 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.