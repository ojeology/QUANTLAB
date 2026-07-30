# R070 — Final Production Stress Test

**Duration:** 31s  |  **Symbols:** 52

## Strategies (Frozen)

- **Family A** `BBW_STRICT  +  RV_LO  +  DST_NR  +  PRG_VH`  PF=3.3529  n=91  MDD=-6.9%
- **Family C (ADX+PBD)** `ADX_ST  +  PBD_HI`  PF=1.6919  n=2049  MDD=-21.2%

## Section 1 — Monthly Stability

**Family A**: 5 months  Top-3=93.8%  FLAG ⚠

**Family C (ADX+PBD)**: 7 months  Top-3=60.8%  FLAG ⚠

## Section 3 — RR Sensitivity

**Family A**

| RR | PF | WR | Exp $ | MDD | Boot P5 |
|---|---|---|---|---|---|
| 1.0 | 1.6765 | 62.6% | 25.27 | -5.6% | 1.1667 |
| 1.25 | 2.0956 | 62.6% | 40.93 | -5.4% | 1.4583 |
| 1.5 | 2.5147 | 62.6% | 56.59 | -5.2% | 1.7500 |
| 1.75 | 2.9338 | 62.6% | 72.25 | -5.1% | 2.0417 |
| 2.0 | 3.3529 | 62.6% | 87.91 | -5.0% | 2.3333 |
| 2.25 | 3.7721 | 62.6% | 103.57 | -4.8% | 2.7439 |
| 2.5 | 4.1912 | 62.6% | 119.23 | -4.7% | 3.0488 |
| 3.0 | 5.0294 | 62.6% | 150.55 | -4.5% | 3.5000 |

**Family C (ADX+PBD)**

| RR | PF | WR | Exp $ | MDD | Boot P5 |
|---|---|---|---|---|---|
| 1.0 | 0.8459 | 45.8% | -8.35 | -175.2% | 0.7848 |
| 1.25 | 1.0574 | 45.8% | 3.11 | -23.5% | 0.9829 |
| 1.5 | 1.2689 | 45.8% | 14.57 | -13.0% | 1.1796 |
| 1.75 | 1.4804 | 45.8% | 26.02 | -6.7% | 1.3762 |
| 2.0 | 1.6919 | 45.8% | 37.48 | -5.9% | 1.5789 |
| 2.25 | 1.9034 | 45.8% | 48.94 | -5.9% | 1.7798 |
| 2.5 | 2.1149 | 45.8% | 60.40 | -5.8% | 1.9582 |
| 3.0 | 2.5378 | 45.8% | 83.31 | -5.8% | 2.3499 |

## Section 4 — Losing Streaks

**Family A**: max=12  P95=7  MC-P95=6

**Family C (ADX+PBD)**: max=25  P95=11  MC-P95=16

## Section 7 — Edge Decay

**Family A**: DECAYING ↓ ⚠

**Family C (ADX+PBD)**: DECAYING ↓ ⚠

## Section 8 — Production Scorecard

| Strategy | Monthly | RR Robust | Frequency | Conc. | Persist | Overall |
|---|---|---|---|---|---|---|
| Family A | 100 | 100 | 29 | 69 | 100 | **85** |
| Family C (ADX+PBD) | 100 | 100 | 82 | 100 | 100 | **97** |

## Final Answers

**Q1 Monthly distribution:** A: top3=94%(FLAG)  C: top3=61%(FLAG)

**Q2 Month dependency:** A:YES  C:YES

**Q3 Recommended RR:** A: RR=3.0  C: RR=3.0

**Q4 RR=3 vs RR=2:** A: 3.0>2.0  C: 3.0>2.0

**Q5 Family C RR robustness:** Min PF=0.846  BREAKS

**Q6 Family A still leads?:** A score=85  C score=97  A PF=3.353 C PF=1.692

**Q7 Deploy unchanged?:** A:YES  C:YES

**Q8 Remaining concerns:** Family A profit concentration (top-3 = 94%); Family C profit concentration (top-3 = 61%); Family A edge decay signal: DECAYING ↓ ⚠; Family C edge decay signal: DECAYING ↓ ⚠; Family A n=91 — small sample for live inference; Family C MC P95 loss streak = 16 ($1600 drawdown)

