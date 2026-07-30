# R065 — Forensic Investigation: RV_HI + DST_MD + ADX_WK + LON

**Date:** July 2026  
**Duration:** 34s  
**Symbols:** 52  
**OOS Bars:** 198,454  

## Verdict: WATCHLIST

## Core Metrics
- **PF:** 2.188  
- **WR:** 56.0%  
- **n:** 25  
- **MDD:** -5.0%  
- **Bootstrap P50:** 2.203  
- **MC P(profit):** 96.8%  
- **Permutation p-value:** 0.9960  
- **UES:** 94.8  
- **Generalisation Score:** 56.3  
- **LOO-sym floor PF:** 1.883  
- **LOO-fold floor PF:** 1.723  

## Final Answers
- **Q1 Genuine edge?** UNCERTAIN — one or more robustness tests failed.
- **Q2 Why works?** RV_HI+DST_MD+ADX_WK+LON = early London trend burst
- **Q4 Diversified?** YES — distributed across multiple symbols and sessions.
- **Q5 Time robust?** YES — edge persists across multiple folds.
- **Q6 Complements E3.1?** YES — low correlation and low drawdown overlap → genuine diversification.
- **Q7 Deploy alone?** NOT YET — trade count too low or one promotion criterion failed.
- **Q8 Combine with E3.1?** YES — the combination is complementary, increasing n while preserving PF.
- **Q9 Stronger than E3.1?** YES — champion outperforms E3.1 on PF and Bootstrap P50.
- **Q10 New production candidate?** WATCHLIST: meets criteria but needs more OOS data or E3.1 is still stronger.

## Promotion Checklist (6/7)
- ✓ PF > 1.30
- ✗ n ≥ 30
- ✓ Boot P50 > 1.20
- ✓ MC P(profit) > 80%
- ✓ LOO-sym PF > 1.0
- ✓ LOO-fold PF > 1.0
- ✓ MDD < 20%

## Outputs
- `r065_dashboard.png`
- `r065_equity_curves.png`
- `r065_symbol_breakdown.png`
- `r065_regime_heatmap.png`
- `r065_bootstrap.png`
- `r065_trades.csv`
- `r065_symbol_summary.csv`
- `r065_fold_summary.csv`
- `r065_ablation.csv`
