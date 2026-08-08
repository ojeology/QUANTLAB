# R094 — 5m COMBINATION SWEEP

**Date:** 2026-08-07 | combos of bank-style signals + filters | selection ≤May, holdout Jun-Aug, 0.05% cost gate


## Audit

- C1_vwap+prevlow: PASS
- C2_vwap+sesslow: PASS
- C3_sesslow+brd: PASS
- C4_prevlow+atr: PASS
- C5_2day+brd: PASS
- C6_vwap+hour: PASS
- C7_sesslow+hour: PASS
- C8_prevlow+green: PASS
- C9_2day+green: PASS
- C10_vwap+sesslow+brd: PASS

## Results

| Combo | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF | holPF@cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1_vwap+prevlow | 273 | 60.7 | 40% | 1.00 | 0.40 | -20.1% | 40% | 1 | 0.97 | 1.01 | 0.46 |
| C2_vwap+sesslow | 7 | 1.6 | 43% | 1.12 | 0.32 | -3.0% | 50% | 1 | 4.50 | 0.00 | 0.00 |
| C3_sesslow+brd | 0 | 0.0 | nan% | nan | nan | 0.0% | nan% | nan | nan | nan | nan |
| C4_prevlow+atr | 310 | 68.9 | 32% | 0.69 | 0.22 | -54.3% | 20% | 4 | 0.80 | 0.63 | 0.22 |
| C5_2day+brd | 661 | 146.9 | 41% | 1.05 | 0.46 | -29.3% | 80% | 1 | 1.11 | 1.02 | 0.49 |
| C6_vwap+hour | 992 | 220.4 | 42% | 1.09 | 0.40 | -38.2% | 60% | 1 | 0.96 | 1.19 | 0.46 |
| C7_sesslow+hour | 73 | 16.2 | 49% | 1.46 | 0.62 | -7.9% | 60% | 2 | 0.71 | 2.25 | 1.00 |
| C8_prevlow+green | 519 | 115.3 | 39% | 0.96 | 0.40 | -27.9% | 60% | 1 | 1.09 | 0.88 | 0.40 |
| C9_2day+green | 735 | 163.3 | 38% | 0.91 | 0.42 | -53.0% | 40% | 3 | 0.94 | 0.89 | 0.45 |
| C10_vwap+sesslow+brd | 0 | 0.0 | nan% | nan | nan | 0.0% | nan% | nan | nan | nan | nan |

## Verdict

**❌ No combination survives 0.05% costs on holdout.** 6th 5m confirmation.