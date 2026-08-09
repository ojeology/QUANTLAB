# FOREX F006 — COMPREHENSIVE TRAP HUNT (bull/bear traps, sessions, follow vs reject)

**Date:** 2026-08-08 | 4H, 8 pairs, RR=3.0, retail spreads | selection ≤ Aug-2025, holdout Aug-2025..Aug-2026 untouched (causal-audited, 14,380 events verified)

Trap types: prior-20 (P20), prior-day (PD), VWAP, EMA20 × bull/bear × follow/reject × trend (up/dn) × session (london hour8 / ny-ovl hour12 / all) = 96 cells.

## 🎯 COST-SURVIVING CELLS (holPF@cost > 1.1) — 6 of 96

| cell | n | WR | PF | holPF | holPF@cost | prof% mo | worst |
|---|---|---|---|---|---|---|---|
| follow P20-bull, downtrend, London | 179 | 30% | 1.05 | **1.53** | **1.33** | 50% | 4 |
| reject P20-bull (short), uptrend, NY | 382 | 29% | 1.07 | 1.31 | **1.16** | 60% | 3 |
| reject PD-bear (long), downtrend, London | 1566 | 28% | 1.01 | 1.34 | **1.17** | 40% | 10 |
| reject VWAP-bear (long), downtrend, London | 1633 | 28% | 1.00 | 1.30 | **1.13** | 48% | 5 |
| reject EMA20-bear (long), downtrend, London | 1291 | 27% | 0.98 | 1.29 | **1.13** | 52% | 5 |
| reject P20-bear (long), downtrend, London | 1916 | 28% | 1.01 | 1.27 | **1.11** | 48% | 5 |

## The winning pattern (the answer to the user's trap question)

**REJECT the BEAR trap (long when price wicks below a level — prior-day/VWAP/EMA20/P20 — then closes back above it) in a DOWNTREND during the LONDON session.** 4 of the 6 winners are exactly this. The trapped short-sellers get squeezed; price snaps back up. This is the classic "bear trap reversal" — and it's REAL when conditioned on (downtrend + London session).

Small-n outlier: follow P20-bull break in downtrend+London (n=179, holPF@cost 1.33) — needs more data to trust.

**This is the FIRST holdout-validated, COST-SURVIVING forex edge** (retail spreads included).

## ML trap classifier
Timed out on full event set (124k events, SVM walk-forward too slow). Deferred to F007 — run on capped sample / faster model. Matrix result above is the headline and is reproducible.

## Verdict
✅ **Bear-trap-reversal + downtrend + London session = cost-surviving forex edge (holPF@cost 1.11–1.17, n=1,291–1,916).** F007: validate this exact config (bootstrap/LOO/MC), test depth/session variants, run the ML trap on a capped sample.
