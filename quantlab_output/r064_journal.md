# R064 — Full Cache Structural Mining: Discovery Research

**Date:** July 2026  
**Duration:** 103s  
**Symbols:** 52 (1H cached)  
**OOS Bars:** 198,454  

## Objective
Full cache discovery run. 32-condition library, all valid 3 & 4 condition combos, starting from zero assumptions.

## Results Summary
- Candidates generated: 34,271
- Oracle survivors: 5846
- WFO survivors (PF≥1.10, n≥15): 37
- Fully validated: 37
- Families discovered: 7
- PROMOTE: 3  |  WATCHLIST: 12

## Family Types
- **Trend Expansion** (10 candidates)
- **Quiet Trend Continuation** (9 candidates)
- **Volatility Expansion Momentum** (6 candidates)
- **Compression Breakout** (5 candidates)
- **Session-Anchored Structure** (3 candidates)
- **Momentum Burst** (3 candidates)
- **Pullback Continuation** (1 candidates)

## Best Family (Composite)
**Conditions:** `RV_HI+DST_MD+ADX_WK+LON`  
**PF:** 2.188  |  **UES:** 94.8  |  **n:** 25  |  **MDD:** -5.0%  
**Family:** Trend Expansion — Trending market + volatility expanding = trend acceleration  
**Composite Score:** 89.4  
**Independence from E3.1:** 93.6/100  

## Most Independent Family
**Conditions:** `RV_HI+DST_FR+ADX_WK+LON`  
**Independence:** 93.9/100  |  **Trade Overlap:** 0.0%  
**PnL Correlation:** 0.000  

## Recommendation for R065
Forensic investigation of: `RV_HI+DST_MD+ADX_WK+LON`  
Priority: entry-gate analysis, symbol-by-symbol breakdown, parameter sensitivity, portfolio fit with E3.1.  

## Outputs
- `r064_dashboard.png` — Master dashboard
- `r064_equity_curves.png` — Top-10 equity curves
- `r064_family_radar.png` — Radar chart top-5
- `r064_family_rankings.csv` — Full rankings
- `r064_best_trades.csv` — Trade log for best family
- `r064_portfolio.csv` — Portfolio composition
- `r064_screener.csv` — Oracle screener results
