# R093 — BANK-STYLE 5m hypotheses

**Date:** 2026-08-07 | 5m, 5 symbols | selection ≤May, holdout Jun-Aug | audit + cost gates


## Audit

- B1_prevdaylow: PASS (causal)
- B2_vwap_fade: PASS (causal)
- B3_sesslow: PASS (causal)
- B4_banker: PASS (causal)
- B5_2daysweep: PASS (causal)

## Results (gross + @0.05% cost)
| Hyp | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF | holPF@cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_prevdaylow | 877 | 194.9 | 39% | 0.98 | 0.41 | -37.9% | 40% | 3 | 1.05 | 0.93 | 0.43 |
| B2_vwap_fade | 4926 | 1094.7 | 50% | 0.99 | 0.24 | -74.4% | 60% | 1 | 1.00 | 0.99 | 0.26 |
| B3_sesslow | 283 | 62.9 | 52% | 1.08 | 0.36 | -18.7% | 40% | 3 | 0.93 | 1.19 | 0.42 |
| B4_banker | 7527 | 1672.7 | 35% | 1.07 | 0.42 | -73.5% | 80% | 1 | 1.12 | 1.04 | 0.41 |
| B5_2daysweep | 1076 | 239.1 | 40% | 0.99 | 0.46 | -53.7% | 60% | 2 | 0.94 | 1.02 | 0.51 |

## Verdict

**❌ No bank-style 5m hypothesis survives 0.05% costs on holdout.** Bank logic tested honestly; still no 5m edge after costs.