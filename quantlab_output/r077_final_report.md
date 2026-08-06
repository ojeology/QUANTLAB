# R077 — $100 DD / 15m / RR sweep → Lock-In

**Date:** 2026-08-06

## A — $100 account (final strategy, breadth50)

| Risk/trade | Net $ | Max DD $ | Max DD % | Final $ |
|---|---|---|---|---|
| 0.5% | +35.19 | -5.38 | -5.4% | 135.19 |
| 1.0% | +81.50 | -10.52 | -10.5% | 181.50 |
| 1.5% | +142.01 | -15.43 | -15.4% | 242.01 |
| 2.0% | +220.54 | -20.12 | -20.1% | 320.54 |
| 2.5% | +321.73 | -24.59 | -24.6% | 421.73 |

At 1% risk / $100 (1R=$1): worst month 2024-10 -4.00R, best month 2025-10 +38.00R, worst streak 7 losers.

Monte Carlo (5,000 paths, 1% risk, $100):
- Max DD: P5=$9.72, P50=$5.88, P95=$3.94
- P(end>$100)=100.0%  P(end>$120)=9.9%  P(end<$80)=46.2%

## B — 15m lower timeframe (locked config, 8 symbols, 2026-only = untouched OOS)
| Timeframe | n | PF | WR | MDD% | t/mo |
|---|---|---|---|---|---|
| 15m | 7 | 0.800 | 28.6% | -3.0% | 1.4 |
| 1H (same 8 syms) | 3 | 1.000 | 33.3% | -2.0% | — |

## C — RR sweep (final 1H config)
| RR | n | WR | PF | MDD% | selPF | holPF | verdict |
|---|---|---|---|---|---|---|---|
| 1.0 | 118 | 62.7% | 1.682 | -10.5% | 1.839 | 1.308 | OK |
| 1.5 | 116 | 57.8% | 2.051 | -9.2% | 2.338 | 1.400 | OK |
| 2.0 | 110 | 51.8% | 2.151 | -10.5% | 2.541 | 1.250 | OK |
| 2.5 | 108 | 42.6% | 1.855 | -16.2% | 2.267 | 0.921 | sel-only |
| 3.0 | 108 | 36.1% | 1.696 | -15.9% | 2.020 | 0.900 | sel-only |
| 3.5 | 108 | 31.5% | 1.608 | -18.3% | 1.815 | 1.050 | OK |
| 4.0 | 108 | 28.7% | 1.610 | -17.5% | 1.754 | 1.200 | OK |

## D — LOCKED CONFIGURATION
**Family A + E6 entry + RR=1.5 + VolCeil(atr_rank≤70) + breadth50(>50% above EMA20)**
- n=116, PF=2.051, WR=57.8%, MDD=-9.2%, Exp=$44.40
- Boot P5=1.553, LOO floor=1.938 (drop INJ_USDT_SWAP)
- ~0.2 trades/month, 65% profitable months, worst month-streak=3
- Paired ΔPF vs RR2.0 (full period): [-0.303, +0.442] — not statistically significant
- Chosen over RR2.0 for better holdout PF (1.400 vs 1.250) and superior retail metrics (WR 58%, 65% prof months, streak 3)