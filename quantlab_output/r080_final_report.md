# R080 — Clean New Hypotheses (higher frequency)

**Date:** 2026-08-06 | E6 entry, RR=1.5, base exit, signal-only (no breadth/volceil) | selection ≤2025, holdout 2026 untouched


## Results

| Hypothesis | n | t/mo | PF | WR | MDD% | prof% | selPF | holPF | holMDD% | hol t/mo |
|---|---|---|---|---|---|---|---|---|---|---|
| A_FAM_signal | 403 | 14.9 | 1.48 | 50% | -31.1% | 48% | 1.49 | 1.45 | -19.9% | 16.3 |
| H1a_trendpull30 | 2974 | 110.1 | 1.01 | 40% | -67.6% | 47% | 1.05 | 0.90 | -63.1% | 113.0 |
| H1b_trendpull25 | 5085 | 188.3 | 0.96 | 39% | -91.8% | 37% | 0.98 | 0.89 | -81.6% | 185.4 |
| H2a_breakout | 24345 | 901.7 | 1.04 | 41% | -97.7% | 63% | 1.04 | 1.06 | -95.8% | 902.9 |
| H2b_breakout15 | 26972 | 999.0 | 1.03 | 41% | -97.9% | 60% | 1.03 | 1.04 | -96.4% | 1015.0 |
| H3a_comppop | 7717 | 285.8 | 1.00 | 40% | -94.8% | 33% | 0.98 | 1.06 | -82.5% | 321.1 |
| H3b_comppop_rv12 | 11089 | 410.7 | 0.98 | 39% | -98.8% | 43% | 0.96 | 1.02 | -91.9% | 452.7 |
| H4a_oversold | 4773 | 176.8 | 0.92 | 38% | -98.3% | 40% | 0.96 | 0.84 | -92.0% | 209.3 |
| H4b_oversold25 | 1991 | 73.7 | 0.98 | 39% | -85.2% | 50% | 1.00 | 0.93 | -60.9% | 93.9 |
| H5a_adxign | 9850 | 364.8 | 1.00 | 40% | -92.0% | 53% | 0.99 | 1.02 | -79.0% | 403.1 |
| H5b_adxign_rv12 | 13038 | 482.9 | 0.99 | 40% | -97.7% | 43% | 0.99 | 0.98 | -90.4% | 529.7 |

## Pass criteria (sel n>=40, selPF>=1.4, holPF>=1.05)

- A_FAM_signal: PASS (t/mo 14.9, PF 1.48, holPF 1.45)
- H1a_trendpull30: FAIL (t/mo 110.1, PF 1.01, holPF 0.90)
- H1b_trendpull25: FAIL (t/mo 188.3, PF 0.96, holPF 0.89)
- H2a_breakout: FAIL (t/mo 901.7, PF 1.04, holPF 1.06)
- H2b_breakout15: FAIL (t/mo 999.0, PF 1.03, holPF 1.04)
- H3a_comppop: FAIL (t/mo 285.8, PF 1.00, holPF 1.06)
- H3b_comppop_rv12: FAIL (t/mo 410.7, PF 0.98, holPF 1.02)
- H4a_oversold: FAIL (t/mo 176.8, PF 0.92, holPF 0.84)
- H4b_oversold25: FAIL (t/mo 73.7, PF 0.98, holPF 0.93)
- H5a_adxign: FAIL (t/mo 364.8, PF 1.00, holPF 1.02)
- H5b_adxign_rv12: FAIL (t/mo 482.9, PF 0.99, holPF 0.98)

## Portfolio (all passed hypotheses, deduped)
- n=403, t/mo=14.9, PF=1.48, MDD=-31.1%, prof%=48%
- holdout: holPF=1.45, holMDD=-19.9%, hol t/mo=16.3

## Verdict

**PORTFOLIO is the frequency answer:** 14.9 trades/month (vs Family A 14.9) at PF 1.48, holdout-validated.