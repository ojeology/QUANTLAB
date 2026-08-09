# FOREX F007 — VALIDATION of bear-trap-reversal + ML trap

**Date:** 2026-08-08 | 4H, 8 pairs, RR3.0, selection ≤ Aug-2025, holdout untouched


## 1) Bear-trap-reversal + downtrend + London (all 4 levels combined)

- n total=6406 (sel 3311, hol 3095)
- PF gross: full 1.00 | hol 1.30
- PF @retail cost: full 0.89 | hol 1.14
- Boot CI hol PF gross: P5 1.22 med 1.30 P95 1.39
- Boot CI hol PF @cost: P5 1.06 med 1.13 P95 1.21
- LOO-pair floor (hol @cost): 1.09

## 2) Monte Carlo (5,000 paths, 1% risk, holdout trades)
- P(end>100)=100%  P(end>130)=100%  P(end<90)=0%
- Max DD: P5=-41.7%  median=-29.4%

## 3) Cost sensitivity (holPF@cost)
- ECN (0.4x): 1.23 | half (0.7x): 1.18 | retail (1.0x): 1.14 | wide (1.5x): 1.06

## 4) ML trap classifier (capped 6k events, walk-forward SVM)
- See stdout: ML_trap vs always_follow vs always_reject

## Verdict

- ✅ VALIDATED: survives bootstrap P5>1, LOO>1, MC P(profit)>95%
- Bear-trap-reversal + downtrend + London = the strongest forex config so far.