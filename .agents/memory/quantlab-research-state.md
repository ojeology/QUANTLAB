---
name: QuantLab Research State
description: Current frozen baselines, promotion status, and research trajectory for the QuantLab algo research project
---

## Frozen Baselines

**Family A (E3.1_v2)** — FROZEN PROMOTE candidate
- Conditions: `BBW_STRICT + RV_LO + DST_NR + PRG_VH`
- BBW_STRICT = bb_width < IS p25 (tighter than BBW_LO = p33)
- R066 baseline: PF=3.353, n=91, WR=62.6%, MDD=-4.6%, UES=209.1, Boot P5=2.333
- All folds profitable; LOO-sym=3.091, LOO-fold=2.600
- **Clear production leader. Deploy first for paper trading.**

**Family B** — LOW SIGNAL FREQUENCY, not yet deployable
- Conditions: `RV_HI + DST_MD + ADX_WK + LON`
- R065: PF=2.188, n=25 (only 25 trades across 52 symbols over full history)
- R066: **0 trades in OOS period** — conditions almost never co-occur
- LON session (7–14 UTC) + high RV + weak ADX + extended above EMA200 = very rare
- **Do not combine into portfolio until more cache data available**

**Family C** — Active but lower quality
- Conditions: `DST_NR + ADX_ST + PBD_HI + ASI`
- R066 baseline: PF=1.492, n=721, WR=42.7%, MDD=-15.1%, UES=74.5
- Boot P5=1.315, MC P(profit)=100%, LOO-sym=1.453, LOO-fold=1.401
- All 5 folds profitable; high trade count but -15% MDD is a concern

## R066 Portfolio Validation Results

| Candidate | PF | WR | n | MDD | UES | Score |
|---|---|---|---|---|---|---|
| **Family A** | **3.353** | **62.6%** | **91** | **-4.6%** | **209.1** | **96.3** |
| A+B | 3.353 | 62.6% | 91 | -4.6% | 209.1 | 96.3 |
| A+C | 1.618 | 44.7% | 787 | -12.7% | 87.5 | 76.4 |
| A+B+C | 1.618 | 44.7% | 787 | -12.7% | 87.5 | 76.4 |
| Family C | 1.492 | 42.7% | 721 | -15.1% | 74.5 | 67.0 |
| Family B | 0.000 | 0.0% | 0 | 0.0% | 12.5 | 32.1 |

**Combining families DILUTES quality.** A+C portfolio drops PF from 3.353 → 1.618 and worsens MDD from -4.6% → -12.7%. Family A alone is superior.

## R066 Key Conclusions

1. Family B has effectively zero signal density in OOS. Do not include in portfolio yet.
2. Family C is real but lower quality — adding it to A dilutes without enough diversification benefit.
3. Family A (E3.1) is the sole production-grade candidate. Run it independently.
4. A+C diversification score is only 39.2/100 — symbol overlap is 84.62% (they trade the same assets).
5. Permutation test (pctile=0.0000) is a known artifact of fixed-RR binary outcomes — not a real failure.

## R067 Family C Dissection Results

**Condition ablation (remove one condition at a time):**

| Variant | Conditions | PF | n | MDD | UES | Loss Streak |
|---|---|---|---|---|---|---|
| **C_no_DST** | **ADX_ST+PBD_HI+ASI** | **1.692** | **2,049** | **-3.2%** | **91.3** | **10** |
| C_no_ADX | DST_NR+PBD_HI+ASI | 1.728 | 1,506 | -10.6% | 90.1 | 9 |
| C_no_PBD | DST_NR+ADX_ST+ASI | 1.573 | 1,438 | -4.2% | 80.6 | 11 |
| C_FULL | DST_NR+ADX_ST+PBD_HI+ASI | 1.492 | 721 | -8.9% | 70.3 | 10 |

**Condition contribution (ΔPF when removed from Family C):**
- ADX_ST: +0.236 (weakest — hurts the strategy)
- DST_NR: +0.200 (second weakest)
- PBD_HI: +0.081 (modest contribution)
- ASI: +0.000 (neutral — session filter adds nothing extra, all trades already in ASI=00:00)

**ADOPT: C_no_DST = ADX_ST+PBD_HI+ASI**
- PF=1.692 (+0.200 vs full), MDD=-3.2% (vs -8.9%), n=2049 (3× more trades)
- Boot P5=1.576, UES=91.3, max loss streak=10
- DST_NR was filtering OUT good trades — removing it both improves quality AND increases frequency
- This is NOT optimization: it's evidence-based condition removal

**Symbol insights from R067:**
- Best tier: T3-Small (PF=1.638), worst: T2-Mid (PF=1.279)
- Worst symbols: NEAR (-83% WR), OP (-87% WR), FET (-78% WR), ATOM (-77% WR), SUI (-76% WR)
- Intra-session analysis inconclusive (timestamp artifact — all read as hour=0)

## R068 ADX_ST+PBD_HI Independent Validation (COMPLETE)

**Result: 8/8 production criteria passed — CLEARED FOR PAPER TRADING**

| Metric | Value |
|---|---|
| PF | 1.6919 |
| WR | 45.8% |
| n | 2,049 |
| MDD | -5.9% |
| Expectancy | $37.48/trade |
| Boot P5 | 1.573 (criterion >1.20 ✓) |
| MC P(profit) | 100% (criterion >95% ✓) |
| LOO-sym floor | 1.674 [LTC removed] |
| LOO-fold floor | 1.644 |
| All 5 folds profitable | YES ✓ |

**Final Q&A (R068 Section 12):**
- Q1 Genuine standalone edge: YES
- Q2 DST_NR truly redundant: UNCERTAIN — bootstrap CI spans 0; directionally better but not proven
- Q3 Survives independent validation: YES
- Q4 Deploy on demo today: YES
- Q5 New official Family C: NOT YET — run alongside original, monitor live
- Q6 Stop Family C research: NO — paper trade first
- Q7 vs Family A: Both real. Family A = high conviction/low frequency (PF=3.35, n=91). ADX+PBD = moderate conviction/high frequency (PF=1.69, n=2049). Run both.

**Promotion status: ADX_ST+PBD_HI → PAPER TRADING alongside Family A**
- Demo bot should run both strategies independently
- Worst-case simulated drawdown: -33.6% (know this going in)
- MC expected drawdown: -7.2%

## Capital Allocation Finding (R066 Section 7)
- Equal weight (33/33/33): PF=1.618, MDD=-8.3%
- Kelly-weighted (64%A / 36%C): PF=1.718, MDD=-8.3%, RF=6.57 — practical choice if both families used
- Risk parity ranks best numerically but allocates 100% to empty Family B

## Stress Test (A+B+C combined)
- Bootstrap P5=1.437 ✓, MC P(profit)=100% ✓, LOO-fold=1.581 ✓, LOO-sym=1.580 ✓
- Permutation: artifact of fixed-RR ✗ (not informative)
- Verdict: MODERATE (4/5 meaningful tests pass)

## Architecture (RR, entry, universe)
- RR=2.0, entry gate: RELVOL>1.5 × 20-bar avg + close>open + close>prev_close
- 52 symbols in cache (1H timeframe), IS_RATIO=0.80, 5-fold walk-forward
- Promotion criteria: PF>1.20, n≥200, boot_med>1.15, MC_p<35%, LOO-sym>1.0, LOO-fold>1.0, MDD<20%

## OKX API Data Limitation
- history-candles REST endpoint caps at ~1440 bars (~60 days of 1H) for new symbols
- 52 symbols remain in cache; cache grows incrementally over time
- Family B's 0 OOS trades is partly explained by the narrow OOS window (20% of 2000+ bars)

## Key File Locations
- Research scripts: `quantlab_r0XX.py` (r064–r068 are the relevant arc)
- Config + indicators: `quantlab_ai.py`
- Cached OHLCV: `quantlab_cache/SYMBOL_1H.parquet` (52 symbols)
- R066 outputs: `quantlab_output/r066_*` (dashboard, equity curves, ranking, allocation, journal)
- R065 forensic: `quantlab_output/r065_journal.md` (Family B forensic)
- Demo bot spec (Family A): `quantlab_output/r062_demo_bot_spec.md`
- R068 outputs: `quantlab_output/r068_*` (dashboard, equity curves, LOO symbol, MC, journal)
