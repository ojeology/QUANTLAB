# R128 — DISSECTION BATTERY II: SILVER + OIL — FINAL (2026-10-26)
Prespec: docs/RESEARCH_FRONTIER_R128.md (locked pre-analysis). Same battery as R127, two assets,
26y each (SI=F, CL=F dailies 2000→2026) + OKX perp rails (XAG, CL 1H 2024→26) + ML addendum
(walk-forward logistic+GBM, yearly refits 2010→2026, non-overlapping 20d steps).

## SILVER — mapped, closed
- Anatomy: +7.0%/yr (sel, Sharpe 0.37) → +13.5%/yr (conf, Sharpe 0.55) — but bears of **−72%** and
  **−53%**; 2026 YTD **−23.1%**. $300 translation: **$3.37/mo**.
- Trend grid: **0/8** (best: N250 long/flat, sel Sharpe 0.48 < 0.5 gate).
- Compression-pop: **n=1 in 33 months** — the pattern does not exist on silver 1H.
- Events: >2σ days → fwd 5d −0.25% vs +0.24% unconditional (mild exhaustion, nothing gated).
- BTC corr +0.16. Sessions: 43% of vol in London+NY.

## OIL — mapped, closed
- Anatomy: +1.0%/yr (sel) → +9.8%/yr (conf) with an **−87% bear**; 2026 YTD **+76.7%** — oil is
  the CURRENT melt-up screenshot asset, exactly as gold was in 2024-25. $300 translation: **$2.45/mo**.
- Trend grid: **0/8** (N20 long/flat's confirm Sharpe 0.52 fails on selection 0.13 — era luck).
- Compression-pop: **n=1**. Events: >2σ days → fwd 20d −1.22% vs +0.46% (strong exhaustion).
- BTC corr −0.03 (true diversifier, for whatever allocation is worth at $300).

## ML ADDENDUM — the operator's question, answered with numbers
| asset | OOS AUC (logreg / GBM) | ML-gated CAGR | buy-and-hold |
|---|---|---|---|
| GOLD | 0.469 / 0.440 | +6.4% | +8.6% |
| SILVER | 0.491 / 0.486 | +4.1% | +8.3% |
| OIL | **0.555** / 0.500 | **+7.8%** | +0.3% |

- Gold & silver: ML is AT/BELOW coin flip and UNDERPERFORMS holding — "did you try ML on it":
  **now yes; it cannot read these markets either** (consistent with R124's information law).
- Oil: the lab's first whisper of ML rank-skill (0.555) and the only ML-gated book that beat its
  underlying — consistent with short-horizon trend structure (mirrors N20 long/flat's confirm run).
  **Diagnostic only; below every promotion bar; recorded as observation, not a lead** (the COT
  confirm-era lesson applies: a whisper on seen data is not a gate pass).

## THE DISSECTION PROGRAM SCOREBOARD (all receipts pushed)
| asset | as ASSET (drift) | as TRADING substrate | verdict |
|---|---|---|---|
| GOLD | Sharpe 0.81, $3.29/mo | trend dominated, pattern n=7, ML < 0.5 | own-able, not tradeable |
| SILVER | $3.37/mo (−72% bears!) | 0/8 trend, pattern n=1, ML 0.49 | own-able (rarely), not tradeable |
| OIL | $2.45/mo (−87% bear, +77% YTD) | 0/8 trend, pattern n=1, ML whisper 0.555 | own-able, not tradeable |
| FX majors | carry fails survival | fully mapped (price+positioning) | closed |
| CRYPTO | — | the ONLY substrate with a validated (small, benched) edge | champion, Jan-1 rule |

Every retail-screenshot asset is now explained: the profits are the asset's own drift (captured
only by holding through bears most cannot stomach), prop-firm leverage, or survivorship. The lab's
one working edge remains crypto microstructure — benched, rule-gated, awaiting Jan 1.
