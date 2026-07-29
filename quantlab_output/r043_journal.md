# QUANTLAB AI — R043 Research Journal

**Date:** 2026-07-29  
**Research ID:** R043  
**Title:** Independent Environment Portfolio Validation  
**Dataset:** 1H · 23 symbols · 487,869 bars · 5-fold WF  

---

## Environments Under Test

- **E1:** Dist>p75 · Wed-Thu · BodyPct>p60 · US(14-21UTC)
- **E2:** ADX>p67 · Dist>p75 · Wed-Thu · US(14-21UTC)
- **E3:** Dist>p75 · Wed-Thu · PrevBody>p67 · US(14-21UTC)
- **E4:** ADX>p67 · Dist>p60 · Wed-Thu · US(14-21UTC)
- **E5:** ATR<p40 · PrevRng>p80 · RealVol<p33 · Slope<0

## Individual Scorecard

| ID | n | WR | PF | p50 | MC% | MDD | LOO-S | LOO-F | Score | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 122 | 0.5164 | 1.7954 | 1.8006 | 0.9995 | -0.062 | 1.6489 | 1.5413 | 6 | WATCHLIST |
| E2 | 159 | 0.5094 | 1.7494 | 1.7485 | 1.0 | -0.0565 | 1.6158 | 1.2384 | 6 | WATCHLIST |
| E3 | 133 | 0.4737 | 1.525 | 1.5188 | 0.989 | -0.0781 | 1.3973 | 1.2749 | 6 | WATCHLIST |
| E4 | 245 | 0.502 | 1.6738 | 1.6683 | 1.0 | -0.06 | 1.5741 | 1.0773 | 7 | PROMOTE |
| E5 | 150 | 0.4733 | 1.4356 | 1.4348 | 0.9935 | -0.0456 | 1.3703 | 1.0556 | 6 | WATCHLIST |

## Portfolio Results

| ID | n | WR | PF | p50 | MC% | MDD | LOO-S | LOO-F | Score | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Port_A | 183 | 0.4754 | 1.53 | 1.5281 | 0.997 | -0.0757 | 1.4193 | 1.1544 | 6 | WATCHLIST |
| Port_B | 187 | 0.4706 | 1.4982 | 1.5036 | 0.9945 | -0.0774 | 1.3908 | 1.0954 | 6 | WATCHLIST |
| Port_C | 273 | 0.4762 | 1.5125 | 1.511 | 0.9995 | -0.0726 | 1.4253 | 1.0093 | 7 | PROMOTE |
| Port_D | 423 | 0.4752 | 1.4845 | 1.4815 | 1.0 | -0.0805 | 1.433 | 1.2196 | 7 | PROMOTE |

## Overlap Matrix

| | E1 | E2 | E3 | E4 | E5 |
| --- | --- | --- | --- | --- | --- |
| **E1** | 100% | 42% | 66% | 25% | 0% |
| **E2** | 42% | 100% | 42% | 54% | 0% |
| **E3** | 66% | 42% | 100% | 24% | 0% |
| **E4** | 25% | 54% | 24% | 100% | 0% |
| **E5** | 0% | 0% | 0% | 0% | 100% |

## Research Questions

**Q1 — Frequency vs Quality:**
- Port A: n=183 (raw=281)  PF=1.530  ✓
- Port B: n=187 (raw=414)  PF=1.498  ✓
- Port C: n=273 (raw=659)  PF=1.513  ✓
- Port D: n=423 (raw=809)  PF=1.485  ✓

**Q2 — Best balance:** Portfolio D

**Q3 — PROMOTE criteria:**
- Port A: Score=6/7  **WATCHLIST**
- Port B: Score=6/7  **WATCHLIST**
- Port C: Score=7/7  **PROMOTE**
- Port D: Score=7/7  **PROMOTE**

**Q4 — Incremental contribution (Port C):**
- E1: n_attr=17  ΔPF=+0.199  ADDS
- E2: n_attr=13  ΔPF=+0.130  ADDS
- E3: n_attr=0  ΔPF=+0.000  ADDS
- E4: n_attr=6  ΔPF=+0.068  ADDS

**Q5 — Any env reduces quality?**
- No environment materially reduces portfolio quality.

**Q6 — Diversification (Port D):**
- Active symbols: 23/23
- HHI: 0.047 (diversified)

---

## Final Verdict

**Recommended portfolio: Portfolio C**

Environments: E1 + E2 + E3 + E4

| Metric | Value | Criterion | Pass |
|--------|-------|-----------|------|
| Profit Factor | 1.513 | >1.2 | ✓ |
| Trade Count | 273.000 | ≥200 | ✓ |
| Bootstrap p50 | 1.511 | >1.2 | ✓ |
| Monte Carlo P% | 99.950 | >60% | ✓ |
| LOO Symbol Floor | 1.425 | >1.00 | ✓ |
| LOO Fold Floor | 1.009 | >1.00 | ✓ |
| Max Drawdown | 7.256 | <25% | ✓ |

**Score: 7/7**  

## VERDICT: PROMOTE