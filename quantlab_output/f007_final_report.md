# FOREX F007 — VALIDATION of bear-trap-reversal + ML trap

**Date:** 2026-08-08 | 4H, 8 pairs, RR3.0, selection ≤ Aug-2025, holdout untouched


## 1) Bear-trap-reversal + downtrend + London (all 4 levels combined)

- n total=6332 (sel 2680, hol 3652)
- PF gross: full 1.01 | hol 1.22
- PF @retail cost: full 0.89 | hol 1.06
- Boot CI hol PF gross: P5 1.14 med 1.22 P95 1.29
- Boot CI hol PF @cost: P5 1.00 med 1.06 P95 1.13
- LOO-pair floor (hol @cost): 1.02

## 2) Monte Carlo (5,000 paths, 1% risk, holdout trades)
- P(end>100)=100%  P(end>130)=100%  P(end<90)=0%
- Max DD: P5=-50.4%  median=-35.1%

## 3) Cost sensitivity (holPF@cost)
- ECN (0.4x): 1.15 | half (0.7x): 1.10 | retail (1.0x): 1.06 | wide (1.5x): 0.99

## 4) ML trap classifier (capped 6k events, walk-forward SVM)
- See stdout: ML_trap vs always_follow vs always_reject

## Verdict

- ⚠️ PARTIAL: see numbers
- Bear-trap-reversal + downtrend + London = the strongest forex config so far.