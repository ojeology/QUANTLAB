# FOREX F001 — first forex hypotheses

**Date:** 2026-08-08 | 8 majors, 1H, Aug2024-Aug2026 | selection ≤ Aug-2025, holdout Aug-2025..Aug-2026 untouched | retail spreads modeled


## Audit

- F1_london_breakout: PASS
- F2_trend_pullback: PASS
- F3_range_meanrev: PASS
- F4_vwap_reclaim: PASS
- F5_momentum_burst: PASS
- F6_crypto_champ_transfer: PASS

## Results

| Hyp | n | t/mo | WR | PF | PF@cost | MDD% | prof% | worst | selPF | holPF | holPF@cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1_london_breakout | 4506 | 187.8 | 38% | 0.92 | 0.70 | -94.4% | 33% | 5 | 0.94 | 0.90 | 0.67 |
| F2_trend_pullback | 2948 | 122.8 | 39% | 0.94 | 0.71 | -84.8% | 38% | 6 | 0.99 | 0.90 | 0.66 |
| F3_range_meanrev | 63 | 2.6 | 40% | 0.99 | 0.75 | -13.8% | 43% | 6 | 1.41 | 0.68 | 0.50 |
| F4_vwap_reclaim | 5293 | 220.5 | 39% | 0.98 | 0.74 | -82.3% | 54% | 4 | 1.03 | 0.93 | 0.68 |
| F5_momentum_burst | 2691 | 112.1 | 39% | 0.95 | 0.72 | -70.9% | 33% | 7 | 0.93 | 0.96 | 0.71 |
| F6_crypto_champ_transfer | 207 | 8.6 | 36% | 0.83 | 0.62 | -30.7% | 43% | 4 | 1.27 | 0.56 | 0.41 |

## Verdict

**❌ No forex hypothesis survives retail spreads on holdout.** First forex run — more hypotheses needed (this is run 1 of the hunt).