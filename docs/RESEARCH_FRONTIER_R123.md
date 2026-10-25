# R123 — REVERSE TRANSFER: FX-TRAINED POP FILTER → CRYPTO — PRESPEC (LOCKED 2026-10-24, pre-run)
Operator hypothesis (symmetric completion of R121): train the pop filter on FOREX pops, apply it
to CRYPTO pops. Status when locked: zero FX-trained models exist; zero FX→crypto scores computed.
One cell, one run, no iteration.

## Training set (locked)
FX raw pops 2016-01→2024-12 from the R121 pipeline (r121_fx_trades.csv, 537 trades, WR 41.7%,
8 majors, SL-first fills, locked pip costs). Model: StandardScaler + SVC(C=1.0, gamma="scale",
probability=True) — byte-identical spec to the champion's filter. Frozen to r123_fx_svm.pkl.

## Test set (locked) — and its honest grade
ALL raw crypto pops from the champion machinery, 73 crypto perps, 2025-06→2026-10 (467 trades,
same rebuild as R121's artifact). **Grade: CONFIRMATORY-STYLE, not discovery** — these labels were
seen (champion selection + rechecks). Contamination is bounded: the FX-trained model contains no
crypto information by construction (verified: fit set = FX 2016–2024 only). A pass therefore
EARNS A PLACE AS A CHALLENGER FILTER in the champion re-entry stack (subject to the standing
forward-paper gates) — it does NOT constitute discovery evidence.

## Registered cell (locked)
Score every crypto pop with the FX brain; keep the top q = 0.55 by score (global rank — scoring
is causal by construction since the model is frozen and FX-only). Trade RR 1.5, costs 0.05%/side
(engine rc convention). Single cell. NO threshold tuning, NO q sweep, NO RR sweep.

## Gates (ALL required ⇒ PROMISING AS CHALLENGER FILTER)
- G1: kept PF@c ≥ 1.10
- G2: n_kept ≥ 100
- G3: bootstrap P(net>0) ≥ 0.95 (2,000 draws)
- G4: leave-one-symbol-out min PF > 1.0
- MUST-BEAT raw: kept PF@c > raw-pop PF@c (same 467 set)
- MUST-BEAT random: kept PF@c > q-quantile of 200 random-55% draws
Report-only (decides nothing): comparison vs the crypto brain's book (PF@c 1.64); FX 2025-2026
self-test of the frozen FX brain (its own unseen window).

## Verdict rule
All gates + both must-beats → the FX brain enters the champion re-entry stack as challenger #2
(validation: forward paper only). Any failure → REJECT: reverse transfer closed, FX→crypto
direction permanently mapped, and R121+R123 together close the entire transfer program.

## RESULT (2026-10-24) - REJECT: information collapse
FX brain outputs ~uniform 0.42 to every crypto pop (59 unique scores/467 trades); kept book PF@c 1.23 vs raw 1.26 (must-beats FAIL); forced top-47 = 0.96; FX home self-test 2025+ kept 0.80 vs raw 1.15. Transfer program closed both directions - the pop edge is crypto-specific volatility microstructure.
