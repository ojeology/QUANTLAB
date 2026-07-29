# QUANTLAB AI — R042 Research Journal

**Date:** 2026-07-29  
**Research ID:** R042  
**Title:** Independent Environment Discovery  
**Dataset:** 1H · 23 symbols · 487,869 bars · 5-fold WF  

---

## Objective

Discover market environments completely independent of R041's Variant G that produce profitable edge when paired with the locked RELVOL Breakout signal.

## Method

- **Entry:** RELVOL Breakout (fixed, unchanged)
- **Conditions per environment:** 3–4
- **3-cond combos tested:** 1,504
- **4-cond extensions (from top-30):** 98 survivors
- **Filter:** n≥40 · PF>1.2 · Boot_p50>1.2
- **Walk-forward:** 5-fold expanding, OOS only

## Var G Reference

| Metric | Value |
|--------|-------|
| Conditions | ATR<p40 · Slope>0 · Dist>p75 · BB<p50 |
| n | 70 |
| PF | 1.174 |
| WR | 42.9% |

## Environment Library — Top 15

| # | PF | n | Boot p50 | MC% | MDD | LOO-S | LOO-F | Overlap | Ind? | Verdict | Conditions |
|---|-----|---|---------|-----|-----|-------|-------|---------|------|---------|------------|
| 1 | 2.003 | 68 | 1.998 | 100% | -3.6% | 1.839 | 1.312 | 0% | ★ | PROMOTE | Dist>p75 · Wed-Thu · BodyPct>p60 · US(14-21UTC) |
| 2 | 1.933 | 137 | 1.931 | 100% | -5.6% | 1.826 | 1.317 | 0% | ★ | PROMOTE | ADX>p67 · Dist>p75 · Wed-Thu · US(14-21UTC) |
| 3 | 1.867 | 88 | 1.873 | 100% | -5.1% | 1.705 | 1.587 | 0% | ★ | PROMOTE | Dist>p75 · Wed-Thu · PrevBody>p67 · US(14-21UTC) |
| 4 | 1.781 | 234 | 1.783 | 100% | -5.6% | 1.690 | 1.168 | 0% | ★ | PROMOTE | ADX>p67 · Dist>p60 · Wed-Thu · US(14-21UTC) |
| 5 | 1.735 | 78 | 1.732 | 99% | -3.6% | 1.648 | 1.207 | 0% | ★ | PROMOTE | ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0 |
| 6 | 1.666 | 65 | 1.669 | 98% | -4.2% | 1.461 | 1.397 | 0% | ★ | PROMOTE | ATR<p25 · Mon-Tue · BodyPct>p60 · Slope<0 |
| 7 | 1.633 | 167 | 1.637 | 100% | -6.5% | 1.549 | 1.160 | 0% | ★ | PROMOTE | ATR>p67 · Dist>p75 · Wed-Thu · US(14-21UTC) |
| 8 | 1.626 | 183 | 1.624 | 100% | -6.7% | 1.498 | 1.220 | 0% | ★ | PROMOTE | Dist>p60 · Wed-Thu · BodyPct>p60 · US(14-21UTC) |
| 9 | 1.593 | 192 | 1.591 | 100% | -7.8% | 1.480 | 1.514 | 6% | ★ | PROMOTE | ADX>p67 · Dist>p75 · BodyPct>p60 · US(14-21UTC) |
| 10 | 1.592 | 163 | 1.594 | 100% | -4.9% | 1.456 | 1.136 | 0% | ★ | PROMOTE | ATR<p40 · Dist<p33 · PrevRng>p67 · RealVol<p33 |
| 11 | 1.591 | 163 | 1.589 | 100% | -6.4% | 1.506 | 1.209 | 0% | ★ | PROMOTE | ADX>p50 · Dist>p75 · Wed-Thu · US(14-21UTC) |
| 12 | 1.577 | 59 | 1.573 | 96% | -6.3% | 1.457 | 1.107 | 0% | ★ | PROMOTE | Dist<p33 · PrevRng>p80 · RealVol<p33 · US(14-21UTC) |
| 13 | 1.572 | 41 | 1.563 | 92% | -5.7% | 1.261 | 1.340 | 13% | ★ | PROMOTE | ATR<p25 · Dist>p60 · BodyPct>p60 |
| 14 | 1.572 | 41 | 1.563 | 92% | -5.7% | 1.261 | 1.340 | 13% | ★ | PROMOTE | ATR<p25 · Dist>p60 · BodyPct>p60 · Slope>0 |
| 15 | 1.571 | 141 | 1.575 | 99% | -7.6% | 1.475 | 1.162 | 0% | ★ | PROMOTE | ADX>p67 · Dist>p75 · Wed-Thu · BodyPct>p60 |

## Research Questions

**Q1. Top 10 by PF:**
- #1: PF=2.003 n=68 Boot=1.998 — Dist>p75 · Wed-Thu · BodyPct>p60 · US(14-21UTC)
- #2: PF=1.933 n=137 Boot=1.931 — ADX>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)
- #3: PF=1.867 n=88 Boot=1.873 — Dist>p75 · Wed-Thu · PrevBody>p67 · US(14-21UTC)
- #4: PF=1.781 n=234 Boot=1.783 — ADX>p67 · Dist>p60 · Wed-Thu · US(14-21UTC)
- #5: PF=1.735 n=78 Boot=1.732 — ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0
- #6: PF=1.666 n=65 Boot=1.669 — ATR<p25 · Mon-Tue · BodyPct>p60 · Slope<0
- #7: PF=1.633 n=167 Boot=1.637 — ATR>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)
- #8: PF=1.626 n=183 Boot=1.624 — Dist>p60 · Wed-Thu · BodyPct>p60 · US(14-21UTC)
- #9: PF=1.593 n=192 Boot=1.591 — ADX>p67 · Dist>p75 · BodyPct>p60 · US(14-21UTC)
- #10: PF=1.592 n=163 Boot=1.594 — ATR<p40 · Dist<p33 · PrevRng>p67 · RealVol<p33

**Q2. Independent environments (overlap ≤ 30%):** 120
- PF=2.003 n=68 overlap=0% — Dist>p75 · Wed-Thu · BodyPct>p60 · US(14-21UTC)
- PF=1.933 n=137 overlap=0% — ADX>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)
- PF=1.867 n=88 overlap=0% — Dist>p75 · Wed-Thu · PrevBody>p67 · US(14-21UTC)
- PF=1.781 n=234 overlap=0% — ADX>p67 · Dist>p60 · Wed-Thu · US(14-21UTC)
- PF=1.735 n=78 overlap=0% — ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0

**Q3. PF>1.2 AND n≥40:** YES — 130 environments

**Q4. Most frequent features:**
- US(14-21UTC) (US): appears in 14/20 top environments
- Wed-Thu (MIDWK): appears in 10/20 top environments
- Dist>p75 (DST_FR): appears in 8/20 top environments
- BodyPct>p60 (PBP_HI): appears in 8/20 top environments
- Dist>p60 (DST_MD): appears in 6/20 top environments

**Q5. Portfolio combination:** 
Combined PF=1.858 n=215 — ADX>p67 · Dist>p75 · Wed-Thu · US(14-21UTC) + ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0

## Verdict

- **PROMOTE:** 104
- **WATCHLIST:** 26
- **Total in library:** 130
- **Independent from Var G:** 120

## R043 Recommendation

Best candidate: **Dist>p75 · Wed-Thu · BodyPct>p60 · US(14-21UTC)**
- PF=2.003  n=68  Boot_p50=1.998  MC=100%
- Score=7/7  Verdict=PROMOTE
- Overlap with Var G: 0%  Independent: True

**Action:** Portfolio-test top candidate alongside Var G in R043.