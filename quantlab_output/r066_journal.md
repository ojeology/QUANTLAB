# R066 — Production Portfolio Validation (Frozen Families Only)

**Date:** July 2026  
**Duration:** 19s  
**Symbols:** 52  

## Frozen Families

| Family | Conditions |
|---|---|
| A | BBW_STRICT+RV_LO+DST_NR+PRG_VH |
| B | RV_HI+DST_MD+ADX_WK+LON |
| C | DST_NR+ADX_ST+PBD_HI+ASI |

## Section 1 — Individual Baselines

| Family | PF | WR | n | MDD | UES | Boot P5 | MC P(profit) | LOO-sym | LOO-fold |
|---|---|---|---|---|---|---|---|---|---|
| A | 3.353 | 62.6% | 91 | -4.6% | 209.1 | 2.333 | 100.0% | 3.091 | 2.600 |
| B | 0.000 | 0.0% | 0 | 0.0% | 12.5 | 0.000 | 0.0% | 0.000 | 0.000 |
| C | 1.492 | 42.7% | 721 | -15.1% | 74.5 | 1.315 | 100.0% | 1.453 | 1.401 |

## Section 2 — Two-Family Portfolios

| Portfolio | PF | WR | n | MDD | UES | Boot P5 | MC |
|---|---|---|---|---|---|---|---|
| A+B | 3.353 | 62.6% | 91 | -4.6% | 209.1 | 2.333 | 100.0% |
| A+C | 1.618 | 44.7% | 787 | -12.7% | 87.5 | 1.444 | 100.0% |
| B+C | 1.492 | 42.7% | 721 | -15.1% | 74.5 | 1.315 | 100.0% |

## Section 3 — Three-Family Portfolio (A+B+C)

- **PF:** 1.618  
- **WR:** 44.7%  
- **n:** 787  
- **Net Profit:** $26,900.0  
- **MDD:** -12.7%  
- **UES:** 87.5  
- **Recovery Factor:** 5.60  
- **Ulcer Index:** 4.66  
- **Boot P50:** 1.618  P5=1.444  
- **MC P(profit):** 100.0%  
- **LOO-sym floor:** 1.580  
- **LOO-fold floor:** 1.581  

## Section 4 — Diversification

| Pair | Trade Overlap | PnL Corr | Sym Overlap | Div Score |
|---|---|---|---|---|
| A+B | 0.00% | +0.0000 | 0.00% | 100.0 |
| A+C | 84.62% | +0.0000 | 84.62% | 39.2 |
| B+C | 0.00% | +0.0000 | 0.00% | 100.0 |

## Section 5 — Drawdown Diversification

| Candidate | MDD | Ulcer | Recovery Factor |
|---|---|---|---|
| Family A | -4.6% | 1.42 | 10.00 |
| Family B | 0.0% | 0.00 | 0.00 |
| Family C | -15.1% | 5.48 | 4.23 |
| A+B | -4.6% | 1.42 | 10.00 |
| A+C | -12.7% | 4.66 | 5.60 |
| B+C | -15.1% | 5.48 | 4.23 |
| A+B+C | -12.7% | 4.66 | 5.60 |

## Section 6 — Trade Frequency

| Candidate | Tpw | Tpm | Tpy | Max Gap (h) | Max Win Streak | Max Loss Streak |
|---|---|---|---|---|---|---|
| Family A | 91.00 | 91.0 | 1092 | 0 | 14 | 8 |
| Family B | 0.00 | 0.0 | 0 | 0 | 0 | 0 |
| Family C | 721.00 | 721.0 | 8652 | 0 | 22 | 36 |
| A+B+C | 787.00 | 787.0 | 9444 | 0 | 22 | 36 |

**Practical for retail?** YES

## Section 7 — Capital Allocation

| Scheme | w_A | w_B | w_C | PF | MDD | RF | UES |
|---|---|---|---|---|---|---|---|
| Equal Weight | 0.33 | 0.33 | 0.33 | 1.618 | -8.3% | 5.60 | 76.3 |
| Volatility Weight | 0.01 | 0.99 | 0.01 | 1.622 | -0.3% | 5.65 | 79.2 |
| Risk Parity | 0.00 | 1.00 | 0.00 | 2.106 | -0.0% | 8.36 | 103.7 |
| Kelly Capped | 0.64 | 0.00 | 0.36 | 1.718 | -8.3% | 6.57 | 81.4 |

**Best allocation:** Risk Parity (0.00/1.00/0.00)

## Section 8 — Stress Tests

- **Bootstrap P5:** 1.437  P50=1.618  (PASS)
- **Monte Carlo P(profit):** 100.0%  (PASS)
- **LOO-fold floor:** 1.581  (PASS)
- **LOO-sym floor:** 1.580  (PASS)
- **Permutation pctile:** 0.0000  (FAIL)
- **Stress verdict:** MODERATE  (4/5)

## Section 9 — Production Ranking

| Rank | Candidate | Score | PF | WR | n | MDD | UES |
|---|---|---|---|---|---|---|---|
| 1 | Family A | 96.3 | 3.353 | 62.6% | 91 | -4.6% | 209.1 |
| 2 | A+B | 96.3 | 3.353 | 62.6% | 91 | -4.6% | 209.1 |
| 3 | A+C | 76.4 | 1.618 | 44.7% | 787 | -12.7% | 87.5 |
| 4 | A+B+C | 76.4 | 1.618 | 44.7% | 787 | -12.7% | 87.5 |
| 5 | Family C | 67.0 | 1.492 | 42.7% | 721 | -15.1% | 74.5 |
| 6 | B+C | 67.0 | 1.492 | 42.7% | 721 | -15.1% | 74.5 |
| 7 | Family B | 32.1 | 0.000 | 0.0% | 0 | 0.0% | 12.5 |

## Section 10 — Final Verdict

1. **Does combining all three improve portfolio?** YES  
   Combined PF=1.618 vs best single=3.353. n increases to 787.

2. **Does it increase trade frequency enough?** YES  
   A+B+C: 787.0 trades/month vs Family A alone: 91.0/month

3. **Does diversification reduce drawdown?** YES  
   A+B+C MDD=-12.7% vs A=-4.6%, B=0.0%, C=-15.1%

4. **Is combined superior to E3.1 alone?** NO  
   A+B+C UES=87.5  n=787 vs Family A UES=209.1 n=91

5. **Deploy today for live paper trading?** Family A  
   Top-ranked: Family A (score=96.3)

## Outputs
- `r066_dashboard.png`
- `r066_equity_curves.png`
- `r066_fold_stability.png`
- `r066_diversification.png`
- `r066_bootstrap.png`
- `r066_monte_carlo.png`
- `r066_ranking.png`
- `r066_allocation.png`
- `r066_drawdown_diversification.png`
- `r066_monthly_returns.png`
- `r066_summary.csv`
- `r066_diversification.csv`
- `r066_allocation.csv`
- `r066_trades_abc.csv`
- `r066_folds.csv`
