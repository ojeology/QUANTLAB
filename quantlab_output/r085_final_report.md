# R085 — Upgraded ML (rich features / sizing / model)

**Date:** 2026-08-06 | base = R084 ML q55 on 73 (9.3 t/mo, 71% prof-mo, PF 2.11)


## Results

| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |
|---|---|---|---|---|---|---|---|---|---|---|
| A_base_ref | 250 | 9.3 | 58% | 2.11 | 1.78 | -15.9% | 71% | 2 | 3.59 | 1.42 |
| B_rich | 193 | 7.1 | 61% | 2.36 | 2.00 | -14.1% | 50% | 3 | 3.45 | 1.50 |
| C_rich_sized | 193 | 7.1 | 61% | 2.35 | 1.96 | -11.6% | 50% | 4 | 3.36 | 1.46 |
| D_gboost | 209 | 7.7 | 62% | 2.47 | 2.06 | -15.0% | 79% | 2 | 5.39 | 1.34 |
| E_rich_dtrend | 55 | 2.0 | 76% | 4.85 | 4.09 | -3.9% | 58% | 4 | 9.90 | 1.69 |

## Feature importance (rich LR)

- d_trend: 0.802
- breadth_now: 0.707
- ema_dist_pct: 0.655
- real_vol_20: 0.604
- rank_24: 0.409
- ret_24: 0.402
- dist_high48: 0.395
- dist_ema20: 0.321
- prev_body_r: 0.308
- rel_vol: 0.307
- green5: 0.295
- prev_range_r: 0.262

## Verdict

**✅ A_base_ref beats the base:** 9.3 t/mo, 71% prof-mo, worst 2, PF 2.11 (cost 1.78), holPF 1.42.