# FOREX F004 — 4H TF + RR sweep (ML+daily-trend)

**Date:** 2026-08-08 | 4H (resampled from 1H), selection ≤ Aug-2025, holdout untouched, retail spreads


## Results

| RR | n | WR | PF | PF@cost | MDD% | prof% | worst | selPF | holPF | holPF@cost |
|---|---|---|---|---|---|---|---|---|---|---|
| RR1.5 | 1345 | 43% | 1.11 | 0.97 | -44.5% | 54% | 6 | 1.15 | 1.07 | 0.93 |
| RR2.0 | 1234 | 36% | 1.12 | 1.00 | -58.1% | 54% | 8 | 1.20 | 1.04 | 0.92 |
| RR3.0 | 1093 | 30% | 1.14 | 1.02 | -43.2% | 62% | 2 | 1.17 | 1.11 | 0.99 |
| RR4.0 | 1024 | 27% | 1.13 | 1.02 | -49.3% | 54% | 5 | 1.23 | 1.05 | 0.94 |

## Verdict

**❌ No 4H RR config survives costs on holdout.**