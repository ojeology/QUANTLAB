# FOREX F003 — RR sweep on ML+daily-trend (1H)

**Date:** 2026-08-08 | selection ≤ Aug-2025, holdout untouched, retail spreads


## Results

| RR | n | WR | PF | PF@cost | MDD% | prof% | worst | selPF | holPF | holPF@cost |
|---|---|---|---|---|---|---|---|---|---|---|
| RR1.0 | 4734 | 52% | 1.09 | 0.78 | -51.4% | 54% | 3 | 1.12 | 1.06 | 0.74 |
| RR1.5 | 4418 | 43% | 1.13 | 0.87 | -59.5% | 71% | 2 | 1.15 | 1.12 | 0.84 |
| RR2.0 | 3999 | 37% | 1.17 | 0.93 | -60.8% | 67% | 2 | 1.16 | 1.18 | 0.91 |
| RR2.5 | 3725 | 32% | 1.16 | 0.93 | -62.5% | 71% | 2 | 1.18 | 1.14 | 0.90 |
| RR3.0 | 3537 | 29% | 1.19 | 0.97 | -63.5% | 79% | 2 | 1.23 | 1.15 | 0.92 |

## Verdict

**❌ No RR makes it cost-surviving at 1H.** The spread drag is structural at 1H — higher RR dilutes it but doesn't overcome it.