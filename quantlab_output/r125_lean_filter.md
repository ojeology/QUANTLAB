# R125 — LEAN-FILTER CHALLENGER (linear SVM) — REJECTED (2026-10-24)
Declared pre-run: one substitution (kernel linear, same 14 features, same walk-forward, same q,
same selection-only threshold); challenger status only; promotion via standing forward gates.
Built from the frozen r096 code path (73 syms, 511 raw pops).

## Head-to-head (same costs, same fills)
| book | n | PF@c | 6-mo gate | 2026 | Oct | maxDD |
|---|---|---|---|---|---|---|
| Incumbent (RBF) | 293 | 1.64 | 0.91 HALT | +21.2R | −4.8R | −13.8% |
| Lean (linear) | 309 | 1.62 | **0.87 HALT** | +17.5R | −6.7R | −12.6% |

- Kept-set overlap: **281/293 (96%)** — the lean filter picks almost the identical book.
- **Paired bootstrap: P(lean > RBF) = 0.00** (5,000 draws, 281 common entries) — the incumbent
  dominates; the ladder's AUC hint (0.572 vs 0.536) was noise at book level, as its n=171
  caveat warned.
- The lean filter does NOT fix the halt (0.87 < 1.0) and is worse in 2026 and October.
- Its standalone battery passes (bootP 1.00, LOAO-min 1.58) — same family, equally robust, no edge.

## CONSEQUENCES
1. Challenger NOT added — the re-entry stack stays {RR1.5/2.0/2.5/3.0 + incumbent RBF filter}.
2. Upgrade path #2 (simpler model/fewer parameters) is now CLOSED at book level with a paired
   receipt, not an opinion.
3. Halt unchanged; next deciding event remains the ~Jan 1 quarterly-rule checkpoint on fresh data.
Repro: `python3 quantlab_r125_lean_filter.py` + paired analysis block (r125_lean_filter.md source).
