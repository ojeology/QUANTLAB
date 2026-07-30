# QUANTLAB AI — R057 — Single Macro Regime Filter Validation

**Frozen Strategy:** `BBW_LO+RV_LO+DST_NR+PRG_VH` | RR=2.0

## Baseline Performance

| Fold | Group | PF | n | WR |
|---|---|---|---|---|
| F1 | WIN | 3.022 | 26 | 65.4% |
| F2 | WIN | 4.184 | 22 | 72.7% |
| F3 | LOSE | 0.460 | 9 | 22.2% |
| F4 | LOSE | 0.270 | 21 | 14.3% |
| F5 | REC | 1.276 | 18 | 44.4% |

## Filter Results Summary

| Filter | PF | F3/F4 Losses Avoided | F1/F2 Winners Sacrificed | Efficiency | MC P |
|---|---|---|---|---|---|
| ATR Rank Calm | 0.448 | 56.0% | 93.9% | 0.60x | 98.7% |
| Realised Vol Calm | 1.529 | 0.0% | 0.0% | 0.00x | 2.0% |
| BB Width Strict | 2.097 | 20.0% | 12.1% | 1.65x | 0.3% |
| Dual Vol Calm | 1.529 | 0.0% | 0.0% | 0.00x | 2.0% |
| ADX Moderate | 1.366 | 16.0% | 24.2% | 0.66x | 9.2% |
| EMA Slope Calm | 1.137 | 52.0% | 57.6% | 0.90x | 32.9% |

## Best Filter: BB Width Strict

- **Filter ID:** F_BBW_STRICT
- **Description:** Require BB width in lowest 25th pct (vs 33rd in BBW_LO)
- **PF improvement:** 1.467 → 2.097
- **F3+F4 losses avoided:** 20.0%
- **F1+F2 winners sacrificed:** 12.1%
- **Efficiency ratio:** 1.65x
- **Freeze verdict:** YES — freeze this updated strategy and run a brand-new forward test.

## Ranking

| Rank | Filter | Score/100 | PF | Efficiency | MC% |
|---|---|---|---|---|---|
| 1 | F_BBW_STRICT | 84.6 | 2.097 | 1.65x | 0.3% |
| 2 | F_RV_CALM | 47.6 | 1.529 | 0.00x | 2.0% |
| 3 | F_VOL_DUAL | 42.6 | 1.529 | 0.00x | 2.0% |
| 4 | F_ADX_MOD | 25.5 | 1.366 | 0.66x | 9.2% |
| 5 | F_EMA_SLOPE | 20.5 | 1.137 | 0.90x | 32.9% |
| 6 | F_ATR_CALM | 18.0 | 0.448 | 0.60x | 98.7% |
