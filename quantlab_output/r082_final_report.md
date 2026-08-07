# R082 — Multi-Timeframe / ML / Exit new levers

**Date:** 2026-08-06 | base = Family A RAW (E6, RR1.5): 14.9 t/mo, PF 1.48, MDD -31%, prof-mo 48%

**Target:** t/mo>=8, prof-mo%>=65, worst<=3, holPF>1.1, PF@0.05%>1.1


## Results

| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |
|---|---|---|---|---|---|---|---|---|---|---|
| RAW_base | 403 | 14.9 | 50% | 1.49 | 1.27 | -31.1% | 48% | 3 | 1.51 | 1.45 |
| F1_dailytrend | 127 | 4.7 | 59% | 2.16 | 1.87 | -8.3% | 57% | 3 | 2.18 | 2.00 |
| F2_4htrend | 42 | 1.6 | 43% | 1.12 | 0.96 | -8.7% | 50% | 3 | 1.03 | 1.50 |
| F3_dailyadx | 240 | 8.9 | 48% | 1.36 | 1.17 | -21.1% | 41% | 5 | 1.52 | 0.93 |
| F4_dailybr | 101 | 3.7 | 64% | 2.71 | 2.38 | -11.9% | 58% | 4 | 2.70 | 3.00 |
| F5_dt_4h | 36 | 1.3 | 47% | 1.34 | 1.16 | -8.7% | 62% | 2 | 1.08 | 6.00 |
| F6_ml | 209 | 7.7 | 60% | 2.28 | 1.93 | -17.4% | 64% | 2 | 3.63 | 1.47 |
| F7_betrail | 407 | 15.1 | 40% | 1.19 | 0.96 | -32.0% | 34% | 7 | 1.27 | 1.00 |
| F8_dt_betrail | 129 | 4.8 | 47% | 2.50 | 2.05 | -15.1% | 39% | 5 | 2.60 | 1.73 |

## Verdict

**❌ No config meets ALL criteria.** Honest result — see table. Closest configs listed above.