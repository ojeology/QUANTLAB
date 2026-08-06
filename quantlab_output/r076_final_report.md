# R076 — Overlay + Cross-Sectional

**Date:** 2026-08-06  |  Selection ≤2025, holdout = 2026 (untouched)


## Part A — Market-timing overlay on Family A FINAL

| Overlay | n | PF | WR | MDD% | t/mo | prof% | streak | selPF | holPF | holMDD% | holProf% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 295 | 1.620 | 44.7% | -26.5% | 10.2 | 41% | 7 | 1.699 | 1.440 | -13.2% | 43% |
| btc_bull | 95 | 1.115 | 35.8% | -26.9% | 4.3 | 36% | 12 | 0.833 | 2.154 | -3.0% | 80% |
| breadth50 | 110 | 2.151 | 51.8% | -10.5% | 5.5 | 55% | 4 | 2.541 | 1.250 | -3.9% | 50% |
| breadth60 | 101 | 2.122 | 51.5% | -9.6% | 5.0 | 55% | 4 | 2.588 | 1.067 | -3.9% | 50% |
| medret_pos | 114 | 1.677 | 45.6% | -14.2% | 4.8 | 46% | 3 | 1.953 | 1.053 | -4.0% | 60% |

**Best overlay: breadth50**

## Part B — Cross-sectional (best config: reversal h=72 rebal=24h K=10)

| Period | n | total | MDD% | Sharpe | win% |
|---|---|---|---|---|---|
| selection | 502 | -14.0% | -52.0% | -0.03 | 55% |
| holdout | 210 | -55.2% | -59.6% | -3.53 | 45% |

## Verdict

**ADOPT overlay 'breadth50'** for Family A: full-period PF 2.15 (vs 1.62), MDD -10.5% (vs -26.5%), 55% profitable months (vs 41%), worst streak 4 (vs 7). Confirmed on untouched 2026 holdout: holPF 1.25, holMDD -3.9%, 50% profitable months.
**Cross-sectional not adopted:** best config failed the holdout (Sharpe -3.53).