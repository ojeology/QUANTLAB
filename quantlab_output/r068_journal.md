# R068 — ADX_ST+PBD_HI Simplified Family C Validation

**Duration:** 9s  |  **Symbols:** 52

## Strategy (Frozen)

- Conditions: `ADX_ST + PBD_HI`
- Entry: RELVOL gate (unchanged)
- Exit: RR=2.0 (unchanged)

## Section 1 — Walk-Forward

| Metric | Value |
|---|---|
| PF | 1.6919 |
| WR | 45.8% |
| n | 2049 |
| MDD | -5.9% |
| Expectancy | $37.48 |
| Avg R | 0.3748R |

| Fold | PF | WR | n | MDD |
|---|---|---|---|---|
| 1 | 1.838 | 47.9% | 499 | -5.6% |
| 2 | 1.902 | 48.8% | 400 | -6.4% |
| 3 | 1.389 | 41.0% | 322 | -10.6% |
| 4 | 1.703 | 46.0% | 524 | -5.0% |
| 5 | 1.535 | 43.4% | 304 | -9.1% |

## Section 2-3 — Bootstrap & MC

- Boot Med=1.689  P5=1.573  P95=1.819
- MC P(profit)=100.0%  E[MDD]=-7.2%  Worst=-33.6%

## Section 4-5 — LOO

- LOO-sym floor=1.6737 [LTC_USDT_SWAP]
- LOO-fold floor=1.6442

## Section 10 — Production Checklist

Score: 8/8

- ✓ PF > 1.50: PF=1.6919
- ✓ Bootstrap P5 > 1.20: P5=1.5726
- ✓ MC P(profit) > 95%: P=100.0%
- ✓ LOO-sym floor > 1.0: Floor=1.6737
- ✓ LOO-fold floor > 1.0: Floor=1.6442
- ✓ All 5 folds profitable: 5/5 profitable
- ✓ MDD < 20%: MDD=-5.9%
- ✓ Practical frequency (≥20 trades/fold): avg=410/fold

## Section 11 — Comparison vs Original Family C

Bootstrap PF difference CI: [-0.1272, +0.5319]  P(simplified better)=86.9%

## Section 12 — Final Answers

**Q1. Genuine standalone edge?** → YES

**Q2. DST_NR truly redundant?** → UNCERTAIN

**Q3. Survives validation?** → YES

**Q4. Deploy on demo today?** → YES

**Q5. New official Family C?** → NO

**Q6. Stop Family C research?** → NO — paper trade first

**Q7. vs Family A — more trustworthy?** → Both real. Family A=higher edge, ADX+PBD=higher confidence (n=2049)

