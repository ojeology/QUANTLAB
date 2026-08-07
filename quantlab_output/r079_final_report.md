# R079 — Trade-Frequency Expansion

**Date:** 2026-08-06 | locked baseline: Family A + E6 + RR1.5 + VolCeil70 + breadth50


## Sweep (n = full-period trades, t/mo = all-months)

| Variant | n | t/mo | PF | WR | MDD% | prof% | selPF | holPF | holMDD% |
|---|---|---|---|---|---|---|---|---|---|
| BASE | 116 | 4.3 | 2.05 | 58% | -9.2% | 65% | 2.34 | 1.40 | -3.9% |
| V01_br30 | 149 | 5.5 | 1.65 | 52% | -13.2% | 54% | 1.86 | 1.14 | -4.9% |
| V02_br35 | 143 | 5.3 | 1.65 | 52% | -14.5% | 56% | 1.80 | 1.25 | -4.9% |
| V03_br40 | 138 | 5.1 | 1.79 | 54% | -14.5% | 58% | 1.96 | 1.32 | -4.9% |
| V04_br45 | 123 | 4.6 | 1.85 | 55% | -12.7% | 61% | 2.04 | 1.41 | -3.9% |
| V05_vc80 | 131 | 4.9 | 1.83 | 55% | -12.0% | 61% | 2.09 | 1.25 | -6.3% |
| V06_vc90 | 144 | 5.3 | 1.72 | 53% | -12.0% | 57% | 1.96 | 1.21 | -9.1% |
| V07_rv13 | 156 | 5.8 | 1.62 | 52% | -10.5% | 45% | 1.94 | 0.94 | -9.7% |
| V08_rv10 | 222 | 8.2 | 1.42 | 49% | -14.1% | 42% | 1.59 | 1.00 | -14.1% |
| V09_bbwlo | 140 | 5.2 | 1.63 | 52% | -15.0% | 57% | 1.78 | 1.26 | -4.4% |
| C1_br40vc80 | 155 | 5.7 | 1.64 | 52% | -17.0% | 54% | 1.81 | 1.20 | -7.3% |
| C2_br40vc80rv | 204 | 7.6 | 1.44 | 49% | -18.7% | 50% | 1.62 | 0.98 | -11.9% |
| C3_br45vc80 | 138 | 5.1 | 1.68 | 53% | -15.4% | 56% | 1.86 | 1.26 | -6.3% |
| C4_br40bbw | 163 | 6.0 | 1.52 | 50% | -18.7% | 56% | 1.60 | 1.29 | -5.4% |

## Candidates (holPF>1, selPF>1, PF>=1.5)

**Best frequency gain: C4_br40bbw** — t/mo 6.0 (vs base 4.3), PF 1.52, MDD -18.7%, holPF 1.29

- C4_br40bbw: t/mo 6.0, PF 1.52, holPF 1.29, holMDD -5.4%
- C1_br40vc80: t/mo 5.7, PF 1.64, holPF 1.20, holMDD -7.3%
- V01_br30: t/mo 5.5, PF 1.65, holPF 1.14, holMDD -4.9%
- V06_vc90: t/mo 5.3, PF 1.72, holPF 1.21, holMDD -9.1%
- V02_br35: t/mo 5.3, PF 1.65, holPF 1.25, holMDD -4.9%
- V09_bbwlo: t/mo 5.2, PF 1.63, holPF 1.26, holMDD -4.4%
- V03_br40: t/mo 5.1, PF 1.79, holPF 1.32, holMDD -4.9%
- C3_br45vc80: t/mo 5.1, PF 1.68, holPF 1.26, holMDD -6.3%
- V05_vc80: t/mo 4.9, PF 1.83, holPF 1.25, holMDD -6.3%
- V04_br45: t/mo 4.6, PF 1.85, holPF 1.41, holMDD -3.9%
- BASE: t/mo 4.3, PF 2.05, holPF 1.40, holMDD -3.9%

## Verdict

Recommendation: adopt **C4_br40bbw** if higher frequency is worth the (small) edge trade-off; otherwise keep BASE. Every gain in t/mo costs some PF/MDD — the table shows the exact price.