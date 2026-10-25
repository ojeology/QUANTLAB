# R126 — UNSUPERVISED REGIME STATES AS A GATE OVERLAY — PRESPEC (LOCKED 2026-10-24, pre-HMM)
The last untested ML paradigm (label-free). Hypothesis: an HMM fit on market-level panel features
discovers states that separate good pops from bad, adding information the 14 engineered features
lack. One cell, one run. AMENDED PRE-LOCK (data reality, disclosed): the repo's crypto 1H panel
starts 2024-01-28 (Binance-era cache wiped) → fit window = 2024-01-28→2025-05-31, NOT 2020-24.

## Fit (label-free, fit-window only)
- Panel daily features from 73-coin 1H closes: f1 = cross-sectional mean return,
  f2 = mean per-coin 20-day realized vol, f3 = cross-sectional return dispersion.
- Standardize (scaler fit on fit window). GaussianHMM, n_components=3, diagonal covariance,
  fit ONCE on 2024-01-28→2025-05-31. Frozen. Sensitivities K=2, K=4 report-only.
- Causal decode law: state at any trade date = Viterbi on the prefix ENDING at that date
  (frozen params; no future frames). Disclosed: fit-era ranking trades have 100–490-day prefixes
  (early decode wobble admitted); all test trades have ≥490-day prefixes.

## State ranking (fit-era data only)
Raw pops (unfiltered signal) 2024-09→2025-05, RR 1.5, 0.05%/side. Favorable states =
states with fit-era pop PF@c ≥ 1.1. If ALL or NO states qualify → verdict UNINFORMATIVE (REJECT):
the HMM carries no trade-relevant structure.

## OOS test (2025-06→2026-10 incumbent kept book, n=293; gates ALL required)
- G1: favorable-restricted kept book PF@c > full kept book PF@c (1.64)
- G2: favorable kept n ≥ 60
- G3: ordering — unfavorable kept PF@c < favorable kept PF@c
- G4: ordering holds inside 2026 alone (the freshest subset)
Pass all → PROMISING AS REGIME OVERLAY (validation: forward paper at re-entry, same gauntlet as
RR ladder). Any fail → REJECT: unsupervised door closed with receipts.

## Report-only (decides nothing)
Per-state 2026 table; refit filter with regime one-hots → 2026 rank-AUC vs the 0.46 incumbent
baseline; K=2/4 economics.

## RESULT (2026-10-24) - REJECT (G2 53<60; G4 no 2026 ordering)
HMM found the explosive regime cleanly (fit-era PF 1.52 vs 0.62) and its OOS slice scored 8.61 PF -
but that is rediscovery of the known vol-expansion dependence, absent in 2026 (n=6, no edge), and
unstable in K. Regime label explains the harvest; it cannot schedule the next one. Door closed.
