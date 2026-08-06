# R075 — New Strategy Families (retail-friendly hunt)

**Date:** 2026-08-06  |  Selection ≤2025, holdout = 2026 (untouched)

**Goal:** find a profile a retail trader can survive (≥50% profitable months, max losing streak ≤4, moderate frequency, lower MDD) — even at a PF trade-off.


## Family A FINAL (R074 reference)

- PF=1.63, ~10 t/mo, 41% profitable months, worst streak 7, MDD -26% → Retail Score below


## Full-period comparison

| Family | n | WR | PF | Exp$ | MDD% | t/mo | prof% | worst | BootP5 | RetailScore |
|---|---|---|---|---|---|---|---|---|---|---|
| N1_trendpull | 17755 | 34.1% | 0.616 | -29.29 | -100.0% | 591.8 | 0% | 30 | 0.596 | 20.0 |
| N2_meanrev | 4565 | 47.6% | 0.909 | -4.74 | -94.6% | 152.2 | 37% | 8 | 0.864 | 32.2 |
| N3_breakout | 10874 | 32.3% | 0.576 | -23.39 | -100.0% | 362.5 | 0% | 30 | 0.550 | 20.0 |
| N4_orb | 38629 | 33.7% | 0.621 | -21.35 | -100.0% | 1287.6 | 0% | 30 | 0.605 | 20.0 |
| A_FINAL ⭐ Family A | 295 | 44.7% | 1.620 | +34.24 | -26.5% | 10.2 | 41% | 7 | 1.333 | 48.4 |

## Selection / holdout (new families)
| Family | sel n | sel PF | hol n | hol PF | hol MDD% | hol t/mo |
|---|---|---|---|---|---|---|
| N1_trendpull | 13178 | 0.632 | 4577 | 0.573 | -100.0% | 653.9 |
| N2_meanrev | 3208 | 0.951 | 1357 | 0.815 | -87.9% | 193.9 |
| N3_breakout | 8168 | 0.585 | 2706 | 0.551 | -99.9% | 386.6 |
| N4_orb | 27754 | 0.677 | 10875 | 0.495 | -100.0% | 1553.6 |

## Robustness of best new family (N2_meanrev)
- LOO floor PF=0.898 (drop DYDX_USDT_SWAP)
- MC: P(profit)=0.0%, net=$-8,862, DD P5=-97.4%
- Cost: 0.00%→0.909, 0.05%→0.749, 0.10%→0.615, 0.15%→0.502, 0.20%→0.407
- Monthly: 30 mo, 37% profitable, worst streak 8, best +56.0R, worst -103.3R

## Verdict

**No retail-friendly upgrade yet.** Best new family (N2_meanrev) scored 32.2 vs Family A 48.4 — Family A final remains the reference. Continue hunting.