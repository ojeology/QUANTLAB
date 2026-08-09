# F009 — RSI2 deep-oversold fade at 1H (crypto + forex, RR 1.3)

**Date:** 2026-08-08 | 1H, ~2y, selection ≤2025, holdout 2026 untouched

F008 found the gross edge at 5m; 1H cuts cost/trade ~10x.


## Results

| Config | n | WR | PF | PF@cost | holPF | holPF@cost |
|---|---|---|---|---|---|---|
| fore|S1_rsi2fade | 2011 | 42% | 0.94 | 0.69 | 0.88 | 0.63 |
| fore|S2_rsi2v2 | 799 | 44% | 1.02 | 0.75 | 1.09 | 0.78 |
| fore|S3_bb | 240 | 42% | 0.94 | 0.70 | 0.86 | 0.62 |
| fore|S4_volceil | 1381 | 41% | 0.90 | 0.65 | 0.83 | 0.59 |
| cryp|S1_rsi2fade | 1364 | 43% | 0.99 | 0.77 | 1.03 | 0.78 |
| cryp|S2_rsi2v2 | 436 | 45% | 1.05 | 0.81 | 1.04 | 0.79 |
| cryp|S3_bb | 179 | 44% | 0.99 | 0.79 | 1.11 | 0.87 |
| cryp|S4_volceil | 904 | 44% | 1.04 | 0.78 | 1.12 | 0.82 |

## Verdict

If any config has holPF@cost > 1.1: the RSI2-fade scalp edge survives at 1H.