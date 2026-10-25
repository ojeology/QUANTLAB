# R126 — UNSUPERVISED REGIME OVERLAY — FINAL (2026-10-24)
Prespec: docs/RESEARCH_FRONTIER_R126.md (locked pre-HMM; fit window amended to reality 2024-01→2025-05
— the repo's 1H panel starts 2024-01-28). GaussianHMM K=3, diagonal, fit ONCE on 470 days of
market-level panel features (drift, vol level, dispersion). Causal decode (prefix-Viterbi, frozen params).

## Fit-era discovery (2024-09→2025-05 raw pops, n=131)
State 2 = explosive regime (+0.4σ drift, +1.2σ vol, +1.4σ dispersion): **PF@c 1.52** (n=36)
vs states 0/1: PF@c 0.63/0.62. One favorable state — clean, exactly as the hypothesis hoped.

## OOS test (incumbent kept book 2025-06→2026-10, n=293)
| slice | n | PF@c | sum R |
|---|---|---|---|
| full kept book | 293 | 1.64 | +89.9R |
| favorable (state 2) | **53** | **8.61** | +56.9R |
| unfavorable | 240 | 1.25 | +33.0R |
| 2026 favorable alone | **6** | 1.18 | — |
| 2026 unfavorable | 180 | 1.20 | — |

GATES: G1 PASS (8.61>1.64) · **G2 FAIL (53<60)** · G3 PASS · **G4 FAIL (2026: 1.18 vs 1.20 — no ordering)**
→ **VERDICT: REJECT** per locked rules. Sensitivity (report-only): K=2 and K=4 produce completely
different state economics (K=4: 1.32/7.96/3.79/1.26) — the "favorable state" is not a robust object.

## THE READING (why this is a good reject)
The HMM did its job flawlessly: it discovered the volatility-expansion regime and showed the
champion earns essentially everything in that state (+56.9 of +89.9R from 18% of trades). But that
is REDISCOVERY, not new information — the ledger has said "harvester of volatility expansion" since
R096. As a tradeable overlay it fails three ways: (1) the favorable state fires when it fires — you
cannot manufacture it, and while it persists the ordinary gates already ride it; (2) it was absent
from 2026 (6 trades, no edge); (3) state identity is unstable in K. A regime label explains the
past harvest; it does not predict the next one.

## ML TAXONOMY — NOW FULLY DISPOSITIONED (operator's chart, complete)
supervised-classification ✓ (incumbent, kept, guarded) · ensembles ✓ (R124, memorization) ·
classical time-series ✓ (feature-equivalents: RV/ADX/EMA machinery; GARCH-type = feature swap) ·
deep/sequence ✗ (excluded by data math, R124 rationale) · reinforcement ✗ (sizing not signal;
DD-throttle exists) · **unsupervised ✓ (R126, REJECT — explains, doesn't predict)**.
Every ML door now has a receipt.
