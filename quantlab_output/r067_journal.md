# R067 — Family C Dissection

**Duration:** 42s  
**Symbols:** 52

## Section 1 — Baseline

- **PF:** 1.492  **WR:** 42.7%  **n:** 721  **MDD:** -8.9%  **UES:** 70.3

## Section 2 — Ablation Results

| Variant | Conditions | PF | WR | n | MDD | UES | BootP5 | LoStr |
|---|---|---|---|---|---|---|---|---|
| C_no_DST | ADX_ST+PBD_HI+ASI | 1.692 | 45.8% | 2049 | -3.2% | 91.3 | 1.576 | 10 |
| C_no_ADX | DST_NR+PBD_HI+ASI | 1.728 | 46.3% | 1506 | -10.6% | 90.1 | 1.590 | 9 |
| C_no_PBD | DST_NR+ADX_ST+ASI | 1.573 | 44.0% | 1438 | -4.2% | 80.6 | 1.440 | 11 |
| C_no_ASI | DST_NR+ADX_ST+PBD_HI | 1.492 | 42.7% | 721 | -8.9% | 70.3 | 1.315 | 10 |
| C_FULL | DST_NR+ADX_ST+PBD_HI+ASI | 1.492 | 42.7% | 721 | -8.9% | 70.3 | 1.315 | 10 |

## Section 3 — Symbol Tiers

- **T1-Large:** PF=1.538  WR=43.5%  n=138
- **T2-Mid:** PF=1.279  WR=39.0%  n=241
- **T3-Small:** PF=1.638  WR=45.0%  n=342

## Section 4 — Intra-Session Hours

| Hour UTC | PF | WR | n |
|---|---|---|---|
| 00:00 | 1.492 | 42.7% | 721 |

**Best sub-window:** 00:00–01:00  PF=1.492  n=721

## Section 7 — Condition Contribution

| Removed | Delta PF | New PF | MDD | LoStr |
|---|---|---|---|---|
| ADX_ST | +0.236 | 1.728 | -10.6% | 9 |
| DST_NR | +0.200 | 1.692 | -3.2% | 10 |
| PBD_HI | +0.081 | 1.573 | -4.2% | 11 |
| ASI | +0.000 | 1.492 | -8.9% | 10 |

## Section 10 — Recommendation

**Verdict:** ADOPT  
**Best variant:** ADX_ST+PBD_HI+ASI  
**PF:** 1.692  **WR:** 45.8%  **n:** 2049  **MDD:** -3.2%  **Loss streak:** 10

## Outputs
- `r067_ablation.png`
- `r067_equity_curves.png`
- `r067_fold_heatmap.png`
- `r067_symbol_breakdown.png`
- `r067_intra_session.png`
- `r067_bootstrap_comparison.png`
- `r067_regime_split.png`
- `r067_ablation.csv`
- `r067_symbol_breakdown.csv`
- `r067_hours.csv`
