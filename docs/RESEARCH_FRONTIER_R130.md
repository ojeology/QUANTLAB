# R130 — BIG-MOVE CENSUS & PRECURSOR ATLAS — PRESPEC (LOCKED 2026-10-26, pre-census)
Operator directive: "study every big move in crypto and the conditions that triggered it."
Design: enumerate large hourly moves across the 73-coin OKX panel (2024-01→2026-10), then measure
which ANTECEDENT states (knowable strictly before) lift the probability of a big move — vs base rate.
Anti-mining discipline: time-split validation + positive control + locked feature list.

## Event definition (locked)
Per coin-hour: |1h return| ≥ 4× that coin's trailing-720h hourly σ AND |ret| ≥ 1.5%.
Dedup: first event per coin per 24h window. Cluster flag: ≥5 coins eventing the same UTC hour.

## Splits (locked, by time)
DISCOVERY 2024-01→2025-09-30 · CONFIRM 2025-10-01→2026-10-26 (never co-analyzed until the final table).

## Antecedent features (locked list — 10, all strictly pre-event)
Per coin: bbw_pct (BB width, rolling-720h pctile), rv_pct (20-bar real vol pctile), ema_dist_pct,
prior24h drift. Panel: breadth (% of panel above EMA20), btc_1h return. Market: F&G (entry−1d),
stablecoin 30d change (entry−1d), hour-of-UTC, dow.
CONTEXT-ONLY (disclosed, NOT candidates — the vol-clustering law): recent realized vol level and
recent event history. Candidates = coil / positioning / liquidity / sentiment / timing features.
NOTE: OI + funding antecedents unavailable (bnmetrics wiped; OKX funding parquets cover 3 coins
only) — disclosed gap, the atlas stands without them.

## Positive control (pipeline validation, pre-declared)
The coil conjunction (bbw_pct ≤ 25 ∧ rv_pct ≤ 33 ∧ ema_dist_pct ≤ 33) MUST show forward-24h
big-UP-move lift > 1.0 with binomial p < 0.01 on DISCOVERY. Else: PIPELINE INVALID, no atlas claims.

## Candidate rule (all required)
For any single antecedent state (pre-binned: bottom-25 / middle-50 / top-25 pctiles; timing as-is):
DISCOVERY lift ≥ 1.5 (vs unconditional 24h big-move base rate, direction-conditional) AND
CONFIRM lift ≥ 1.3 AND discovery events ≥ 50 AND binomial p < 0.001 → PROMISING-CANDIDATE
(new signal design round). Everything else: ATLAS (descriptive). No post-hoc conjunctions.

## Follow-through module (descriptive)
After big UP events: fwd 24h/72h returns vs unconditional, split cluster/idiosyncratic —
informs continuation-vs-fade anatomy and exit design. No tradeable claim from this module.

## Verdicts
PROMISING-CANDIDATE (rare) / ATLAS COMPLETE (default). Never live-ready language. One run, no
iteration, no re-binning after seeing results.
