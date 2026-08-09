# FOREX F005 — THE TRAP (reverse vs follow)

**Date:** 2026-08-08 | 4H, 8 pairs, selection ≤ Aug-2025, holdout untouched, retail spreads

Trap = wick through prior-20-bar level (stop-hunt); REVERSE = reclaim through it, FOLLOW = continue through it. Long & short via series inversion.


## Results

| Signal | RR | n | WR | PF | PF@cost | MDD% | prof% | worst | holPF | holPF@cost |
|---|---|---|---|---|---|---|---|---|---|---|
| T1_bear_rev_long_d0 | - | 1230 | 34% | 1.00 | 0.89 | -43.3% | 46% | 5 | 0.99 | 0.87 |
| T1_bear_rev_long_d0 | - | 1183 | 28% | 1.03 | 0.92 | -39.4% | 58% | 4 | 1.03 | 0.92 |
| T1_bear_rev_long_d2 | - | 1030 | 33% | 0.99 | 0.88 | -44.2% | 50% | 3 | 0.98 | 0.86 |
| T1_bear_rev_long_d2 | - | 1009 | 27% | 1.00 | 0.89 | -41.8% | 62% | 4 | 1.01 | 0.89 |
| T2_bear_fol_short_d0 | - | 741 | 31% | 0.89 | 0.79 | -54.1% | 42% | 4 | 0.89 | 0.78 |
| T2_bear_fol_short_d0 | - | 642 | 25% | 0.86 | 0.76 | -57.6% | 33% | 4 | 0.80 | 0.71 |
| T2_bear_fol_short_d2 | - | 721 | 31% | 0.88 | 0.78 | -54.9% | 42% | 4 | 0.87 | 0.77 |
| T2_bear_fol_short_d2 | - | 633 | 25% | 0.85 | 0.76 | -58.7% | 33% | 4 | 0.81 | 0.72 |
| T3_bull_rev_short_d0 | - | 1291 | 34% | 1.03 | 0.91 | -45.9% | 50% | 5 | 1.01 | 0.88 |
| T3_bull_rev_short_d0 | - | 1249 | 28% | 1.06 | 0.95 | -42.2% | 62% | 2 | 1.04 | 0.91 |
| T3_bull_rev_short_d2 | - | 1101 | 33% | 0.99 | 0.88 | -42.6% | 46% | 5 | 1.01 | 0.88 |
| T3_bull_rev_short_d2 | - | 1080 | 28% | 1.03 | 0.93 | -38.4% | 54% | 3 | 1.06 | 0.94 |
| T4_bull_fol_long_d0 | - | 812 | 34% | 1.00 | 0.88 | -46.0% | 42% | 4 | 1.10 | 0.96 |
| T4_bull_fol_long_d0 | - | 704 | 29% | 1.07 | 0.96 | -42.8% | 58% | 2 | 1.17 | 1.03 |
| T4_bull_fol_long_d2 | - | 786 | 33% | 0.97 | 0.86 | -43.3% | 38% | 4 | 1.05 | 0.92 |
| T4_bull_fol_long_d2 | - | 691 | 28% | 1.04 | 0.93 | -43.9% | 50% | 4 | 1.13 | 1.00 |

## Verdict

**❌ No trap config survives costs on holdout.** Honest negative — see closest above.