# CHAMPION RR SWEEP — 2026-10-24 (descriptive; NO selection while halted)
First full reward:risk landscape for the champion. Locked pipeline (q=0.55, E6, breadth50,
0.05%/side, 73 syms), fresh data through 2026-10-24 18:00 UTC. RR grid {1.0,1.25,1.5,1.75,2.0,2.5,3.0}.
Repro: `quantlab_champcheck_rr{10,125,175,25,30}.py` (+ rr2 from recheck); trades CSVs in quantlab_output/.

| RR | n | book R | book PF@c | 6-mo R | 6-mo PF@c | gate | Oct R | 2026 R | 2026 PF@c | MDD@1% |
|----|---|--------|-----------|--------|-----------|------|-------|--------|-----------|--------|
| 1.0 | 313 | +59.7 | 1.49 | +3.6 | 1.07 | PASS | −1.9 | +14.9 | 1.18 | −18.3% |
| 1.25 | 295 | +87.3 | 1.71 | +6.8 | 1.12 | PASS | −5.2 | +25.5 | 1.28 | −13.0% |
| **1.5 (traded)** | 293 | +89.9 | 1.64 | **−5.8** | **0.91** | **HALT** | −4.8 | +21.2 | 1.20 | −13.8% |
| 1.75 | 282 | +119.4 | 1.86 | +0.8 | 1.01 | PASS | −6.5 | +31.0 | 1.31 | −12.0% |
| 2.0 | 277 | +127.3 | 1.87 | +3.8 | 1.05 | PASS | −8.8 | +36.4 | 1.33 | −13.9% |
| 2.5 | 274 | +151.8 | 1.96 | +9.8 | 1.14 | PASS | −6.3 | +47.2 | 1.43 | −18.6% |
| 3.0 | 268 | +179.0 | 2.09 | +32.9 | 1.49 | PASS | +1.1 | +80.6 | 1.72 | −23.9% |

## Reading (honest)
1. The edge curve is MONOTONIC in RR on this book — wider targets pay more (expansion pops have
   fat right tails), win rate pays for it (65%→45%). RR1.5 is the local WORST 6-mo cell; every
   neighbor passes the gate. The 2026 "decay" is partly payoff-geometry-local, not necessarily
   family death: in chop, 1.5×ATR targets stopped getting reached while 3×ATR holds rode trends.
2. **Trap #1 — in-sample selection:** this table is the FULL book. Picking RR3.0 off it is
   textbook curve-fitting; it is also the cell with the LEAST validation (R098's TP-first 1m
   audit was at RR1.5 only).
3. **Trap #2 — growing optimistic bias:** the engine books TP-first when a bar touches both
   levels. The wider the target, the more likely an ambiguous bar really hit SL first — the
   assumption that cost 2% at RR1.5 is UNMEASURED at RR2.5–3.0. SL-first bounds required.
4. MDD rises with RR (−23.9% at 3.0) — the operator's survival priority caps how far up the
   curve is even admissible.

## Decision (pre-registered NOW, before the deciding data exists)
- **Halt at RR1.5 STANDS** — the traded config tripped the pre-committed rule; no outcome-switching.
- Re-entry evaluation will judge the PRE-SPECIFIED ladder {1.5 incumbent, 2.0, 2.5, 3.0} on data
  after 2026-10-24 ONLY: admissible = n≥30, SL-first intrabar bound passes (1m audit at that RR),
  6-mo PF@c≥1.0 at evaluation; among admissible pick highest forward PF@c. Declared today:
  2026-10-24. Known bias: this table was seen first (impossible to un-see); the ladder + rule +
  minimum n are still declared before the deciding window.
- Until then: paper-tracking continues; bot stays never-live; $200–300 stays out.


## ADDENDUM 2026-10-24 (e): rolling re-selection procedure test (operator's "test old data blind" idea)
Raw "pick RR from the full-book table" stays forbidden (seen data). The honest version of the
operator's question was tested instead: a MECHANICAL rule — "each quarter, hold the RR with the
best trailing-6-month PF@c among the ladder" — evaluated across the book's four decision quarters.
No human choice occurs at any checkpoint, so the historical pass is meaningful (not answer-key reuse):

| decision | rule picked | forward 3m PF@c (picked) | forward 3m PF@c (fixed RR1.5) | forward R (picked) | beat 1.5 |
|---|---|---|---|---|---|
| 2025-12-01 | 2.0 | 1.39 | 1.36 | +8.5R | YES |
| 2026-03-01 | 2.5 | 1.44 | 1.43 | +25.4R | YES |
| 2026-06-01 | 3.0 | 0.96 | 0.55 | −0.9R | YES |
| 2026-09-01 | 3.0 | 2.81 | 1.73 | +33.6R | YES |

**4/4 forward quarters beat fixed RR1.5**; the rule adapted (2.0→2.5→3.0) as the market rotated.
Caveats: only 4 checkpoints (17-month book); ladder books share signals (correlated exits —
comparison fair but not independent); forward windows now also seen, so this is supporting
evidence for the RULE, never a license to re-pick a number by hand.

### Prespec amendment (supersedes the one-shot choice in ADDENDUM (d))
At re-entry, adopt the STANDING RULE: quarterly, hold the ladder-RR with the best trailing-6-mo
PF@c (min 5 trailing trades per RR; ladder {1.5, 2.0, 2.5, 3.0}; SL-first precondition per cell).
First live pick still subject to the fresh-data forward gate. Halt at RR1.5 unchanged.
