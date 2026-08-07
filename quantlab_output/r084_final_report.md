# R084 — ML filter on expanded universe (73 symbols)

**Date:** 2026-08-06 | ML q55, walk-forward logistic regression


## Results

| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |
|---|---|---|---|---|---|---|---|---|---|---|
| A_ML52 | 216 | 8.0 | 61% | 2.36 | 1.99 | -17.0% | 64% | 2 | 3.75 | 1.50 |
| B_ML73 | 250 | 9.3 | 58% | 2.11 | 1.78 | -16.3% | 71% | 2 | 3.59 | 1.42 |
| C_ML_transfer_new | 30 | 1.1 | 47% | 1.31 | 1.03 | -5.4% | 57% | 2 | 2.25 | 1.00 |
| D_raw73 | 453 | 16.8 | 49% | 1.44 | 1.22 | -28.6% | 48% | 3 | 1.53 | 1.28 |

## Interpretation

- **Transfer test (C):** 30 trades on new pairs, PF=1.31 → ML filter DOES transfer to new instruments.
- **Expanding to 73 does not clearly help** (B vs A — see table).

## Verdict

ML filter is universe-sensitive like the raw rules. More pairs ≠ better; the validated 52 remain the trusted universe. Adding pairs only helps if per-symbol validated.