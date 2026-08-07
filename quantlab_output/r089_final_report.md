# R089 — 5-Minute Edge Hunt

**Date:** 2026-08-07 | 5 symbols (BTC,ETH,DOGE,LINK,LTC), 5m candles, Jan 28 - Aug 7 2026 | selection ≤ May 31, holdout = Jun-Aug (untouched)

**Params scaled to 5m:** lookback 6000 bars (500h), recal 2016 bars (7d)


## Results

| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |
|---|---|---|---|---|---|---|---|---|---|---|
| q0.55 | 0 | 0.0 | nan% | nan | nan | 0.0% | nan% | nan | nan | nan |
| q0.75 | 0 | 0.0 | nan% | nan | nan | 0.0% | nan% | nan | nan | nan |
| q1.0 | 57 | 9.5 | 28% | 0.59 | 0.18 | -17.2% | 29% | 3 | 0.63 | 0.50 |

## Verdict

**Best 5m config: q0.55** — holPF nan, 0.0 t/mo, PF nan. 
Note: only ~6 months of 5m data (5 symbols) — wide uncertainty. A real 5m verdict needs more data/history.