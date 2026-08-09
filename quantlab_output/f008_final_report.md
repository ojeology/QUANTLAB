# SCALP PROBE (crypto + forex, 5m, RR 1.3)

**Date:** 2026-08-08 | 5m candles ~60 days (yfinance) | selection=first 40d, holdout=last 20d

Crypto cost 0.05%/side; forex retail spreads. RR 1.3 (risk 1, target 1.3R).


## Results (gross vs cost)

| Config | n | WR | PF | PF@cost | holPF | holPF@cost |
|---|---|---|---|---|---|---|
| fore|S1_mom | 2035 | 41% | 0.90 | 0.12 | 0.88 | 0.13 |
| fore|S2_fade | 1418 | 43% | 0.96 | 0.15 | 0.95 | 0.14 |
| fore|S3_break | 1166 | 42% | 0.95 | 0.15 | 0.97 | 0.16 |
| fore|S4_pull | 1940 | 40% | 0.84 | 0.16 | 0.84 | 0.17 |
| cryp|S1_mom | 1527 | 44% | 1.03 | 0.19 | 1.06 | 0.19 |
| cryp|S2_fade | 800 | 46% | 1.11 | 0.21 | 1.18 | 0.20 |
| cryp|S3_break | 965 | 42% | 0.93 | 0.18 | 0.97 | 0.18 |
| cryp|S4_pull | 1677 | 43% | 0.98 | 0.15 | 0.97 | 0.15 |

## Verdict

Honest: at 5m, cost drag is huge (crypto ~0.05%/side, forex spread/ATR). Any config with holPF@cost > 1.1 would be a real scalp edge; otherwise the 5m cost wall holds in both markets.