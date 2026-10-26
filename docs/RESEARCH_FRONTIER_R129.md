# R129 — THE FLOW LAYER — PRESPEC (LOCKED 2026-10-26, pre-pairing)
The operator-pressed hunt for NEW INFORMATION (the only axis R124 left open). Verified data recon
(receipts in-session):
- **A. Stablecoin aggregate supply** (DefiLlama `stablecoincharts/all`, totalCirculatingUSD): daily
  2017-11-29 → 2026-10-26 ✓ — genuinely exchange-independent buying-power proxy.
- **C. Fear & Greed** (alternative.me): daily 2018-02-01 → 2026-10-26 ✓ — DERIVATIVE-DATA CAVEAT
  RETAINED (partly price-composed).
- **B. On-chain activity** (Coin Metrics community + blockchain.com): **BLOCKED in this sandbox**
  (HTTP 000 / not-found) — documented, not silently skipped.
DEVIATION (pre-lock): C promoted from report-only to REGISTERED because B is blocked; caveat stands.

## Registered overlay cells (one feature, one rule, each)
Gate semantics: champion kept book (n=293, 2025-06→2026-10). A trade passes the overlay iff its
feature at (entry_date − 1 day, strictly causal) is favorable vs the feature's EXPANDING MEDIAN
(series start → entry−1, min 90 obs — no future data in the threshold).
- **A. LIQUIDITY:** stablecoin-supply 30d % change > expanding median → trade; else SKIP.
  Hypothesis: pops die when fresh buying power is contracting.
- **C. SENTIMENT:** F&G ≤ 75 at entry−1 → trade; else SKIP (one-sided: only euphoria extremes
  veto; fear/neutral always trades). Hypothesis: euphoric-extreme pops fail.

## Evaluation (locked; whole program is CHALLENGER-GRADE — book labels are seen data)
- PRIMARY slice: 2026-01-01→2026-10-26 (n=171, freshest).
- CONSISTENCY slice: 2025-06-01→2025-12-31 (n=122).
- Per cell, ALL gates → "PROMISING AS FLOW OVERLAY" (enters re-entry stack, forward-paper gauntlet):
  - G1: gated PF@c > full-book PF@c on PRIMARY
  - G2: retention ≥ 55% of PRIMARY slice n
  - G3: gated > full on CONSISTENCY slice too (ordering holds in both eras)
  - G4: gated FULL-book PF@c > incumbent full-book 1.64
- Report-only: monthly tables; A×C interaction (descriptive, post-hoc, never a third cell).

## Verdict rule
No cell passes all four → THE FLOW LAYER CLOSES: the last new-information axis reachable from this
sandbox is mapped. Any pass → challenger behind the RR ladder at re-entry. No iteration, no
threshold tuning, one run.
