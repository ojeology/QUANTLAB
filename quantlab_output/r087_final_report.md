# R087 — SVM keep-rate sweep (more-trades dial)

**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward


## Sweep (q = fraction of signals kept by SVM filter)

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 250 | 9.3 | 58% | 2.04 | 1.70 | -17.1% | 69% | 2 | 1.35 |
| q0.65 | 265 | 9.8 | 58% | 2.05 | 1.71 | -17.1% | 64% | 2 | 1.32 |
| q0.75 | 280 | 10.4 | 56% | 1.94 | 1.62 | -17.1% | 71% | 2 | 1.36 |
| q0.85 | 294 | 10.9 | 55% | 1.84 | 1.54 | -19.4% | 64% | 2 | 1.28 |
| q0.95 | 301 | 11.1 | 55% | 1.84 | 1.54 | -18.2% | 64% | 2 | 1.28 |
| q1.0 | 453 | 16.8 | 49% | 1.44 | 1.22 | -28.6% | 48% | 3 | 1.28 |

## Verdict

More trades = relax q. Each step up in q adds trades but costs profitable-months and PF. The user can pick their point on this curve.