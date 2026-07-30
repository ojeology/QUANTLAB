---
name: QuantLab Research State
description: Current frozen baselines, promotion status, and research trajectory for the QuantLab algo research project
---

## Frozen Baselines

**E3.1_v1** (original): `BBW_LO + RV_LO + DST_NR + PRG_VH`  
- PF=1.467, n=96, WR=47.9%, MDD=-7.4%, UES=77.1, Verdict=WATCHLIST

**E3.1_v2** (R060 confirmed): `BBW_STRICT + RV_LO + DST_NR + PRG_VH`  
- PF=1.640, n=79, WR=50.6%, MDD=-7.0%, UES=80.5, Verdict=WATCHLIST  
- BBW_STRICT = bb_width < IS p25 (tighter than BBW_LO = p33)  
- **Why:** R057 validated this upgrade 5/6 criteria. R060 confirmed PF +0.173 vs v1. Not yet PROMOTE due to n<200 and bootstrap p5=1.09 (needs >1.15).

## E3.1 Fold Stability Problem (known)
- F1: PF=4.85, F2: PF=4.72 → F3: PF=0.80, F4: PF=0.30, F5: PF=0.90
- Regime change in F3/F4 is structural (ATR rank shift d=1.48). Not caused by bad parameters.

## DST_MD Second Family (R059/R060 confirmed)

All three DST_MD environments have **0% overlap with E3.1** (completely orthogonal market regimes).

| Env | Conditions | PF | n | UES | Verdict |
|-----|-----------|-----|---|-----|---------|
| P1 | ADX_WK+DST_MD+RV_HI | 1.359 | 75 | 67.6 | WATCHLIST |
| P2 | ATR_LO+DST_MD+PRG_LO+RV_LO | 1.280 | 49 | 58.5 | WATCHLIST |
| P3 | ADX_WK+DST_MD+LON | 1.225 | 94 | 50.2 | WATCHLIST |
| Portfolio | P1+P2+P3 combined | 1.216 | 183 | 56.6 | WATCHLIST |

**DST_MD Portfolio vs PROMOTE bar:**
- Failing: n=183 (need ≥200), boot_p5=0.94 (need >1.15), MC=9.6% (need <35%) — close on MC but not there yet

## Combined Two-Family Portfolio (R060)
- E3.1_v2 + DST_MD Portfolio: PF=1.331, n=262, MDD=-25.6%, UES=68.2
- 0.0% overlap between families — completely orthogonal
- Combined dilutes E3.1_v2 quality (UES 80.5→68.2) due to MDD tripling
- **Conclusion:** Run families independently until DST_MD matures

## Architecture (RR, entry, universe)
- RR=2.0, entry=RELVOL>1.5 + close>open + close>prev_close
- 49-symbol universe, 1H timeframe, IS_RATIO=0.80, 5-fold walk-forward
- Promotion criteria: PF>1.20, n≥200, boot_med>1.15, MC_p<35%, LOO-sym>1.0, LOO-fold>1.0, MDD<20%

## Open Research Questions (post-R060)
1. **F6 data extension** — F3/F4 regime question needs more calendar time to resolve
2. **DST_MD P4/P5** — Add remaining WATCHLIST candidates from R059 to grow n above 200
3. **LON session sub-structure** — P3 (ADX_WK+DST_MD+LON) has 94 trades; time-of-day precision may push PROMOTE
4. **v2 PROMOTE** — Needs ~3 more months of OOS data to accumulate n≥200 with consistent PF

## Key File Locations
- Research scripts: `quantlab_r0XX.py` (r056–r060 are the relevant arc)
- Output: `quantlab_output/r060_*`
- Config + indicators: `quantlab_ai.py`
- Cached OHLCV: `quantlab_cache/SYMBOL_1H.parquet`
