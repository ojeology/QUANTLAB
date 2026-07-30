---
name: QuantLab Research State
description: Current frozen baselines, promotion status, and research trajectory for the QuantLab algo research project
---

## Frozen Baselines

**E3.1_v1** (original): `BBW_LO + RV_LO + DST_NR + PRG_VH`  
- PF=1.467, n=96, WR=47.9%, MDD=-7.4%, UES=77.1, Verdict=WATCHLIST

**E3.1_v2** (R060 confirmed, R062 validated): `BBW_STRICT + RV_LO + DST_NR + PRG_VH`  
- PF=1.640, n=79, WR=50.6%, MDD=-7.0%, UES=80.5, Verdict=WATCHLIST (R061)  
- PF=1.561, n=79, WR=49.4%, MDD=-9.0%, UES=100.0, Verdict=STRONG REAL EDGE (R062)
- BBW_STRICT = bb_width < IS p25 (tighter than BBW_LO = p33)  
- R062 stat tests: 7/7 PASSED — bootstrap CI [1.085–2.235], MC P(profitable)=98.2%, permutation=100th pctile
- **Demo bot spec ready** → `quantlab_output/r062_demo_bot_spec.md`
- **Blocking PROMOTE:** n still only 79 (need ≥200 trades); CI narrows only with more trade history

## E3.1_v2 Key Findings from R062

- Edge is **universal** — Tier-3 (small-cap) PF=2.080 n=44; edge not confined to large-caps
- Tier distribution: 17.7% Tier1, 26.6% Tier2, 55.7% Tier3 trades
- Session split: Asia 45.6%, London 40.5%, US 13.9%
- Fold stability: F1=4.85, F2=4.72 profitable; F3/F4 losing; LOO-fold floor=1.096 (passes)
- **Core insight:** 79 trades = ~173/year rate. Need ~14 months of paper trading to reach n=200.
- **Next action:** paper trade bot as specified in `r062_demo_bot_spec.md`, collect n≥200, re-test

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
- 52-symbol cache (46 loaded for R062), 1H timeframe, IS_RATIO=0.80, 5-fold walk-forward
- Promotion criteria: PF>1.20, n≥200, boot_med>1.15, MC_p<35%, LOO-sym>1.0, LOO-fold>1.0, MDD<20%

## OKX API Data Limitation (discovered R062)
- The OKX `history-candles` REST endpoint caps at ~1440 bars (~60 days of 1H) for symbols
  without prior accumulated cache. Cannot bulk-download 24 months of 1H for new symbols.
- Workaround: let cache accumulate incrementally over months (as done for the original 49 symbols).
- 95 new symbols failed download; 52 symbols remain in cache for analysis.

## Key File Locations
- Research scripts: `quantlab_r0XX.py` (r056–r062 are the relevant arc)
- Output: `quantlab_output/r062_*`  (dashboard, equity curves, universe CSV, scorecard, bot spec)
- Config + indicators: `quantlab_ai.py`
- Cached OHLCV: `quantlab_cache/SYMBOL_1H.parquet`  (52 symbols)
- Demo bot spec: `quantlab_output/r062_demo_bot_spec.md`
