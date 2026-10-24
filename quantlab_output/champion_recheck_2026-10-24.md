# CHAMPION RECHECK — 2026-10-24 (early-November ritual, run at operator request)
Ritual: `python3 scripts/refresh_okx.py` (73 syms → 2026-10-24 18:00 UTC, +357 bars each, 70 s)
→ `python3 quantlab_r096_recheck.py` (frozen pipeline, RR 1.5, q 0.55, 0.05%/side).
Pre-registered extras per R100: kill-switch v2 verdict + RR 2.0 promotion decision.

## 1. KILL-SWITCH v2: **TRIPPED → HALT** 🛑
Rolling 6-month PF@cost (rule: halt < 1.0):

| book | window | PF@c | sum R | n |
|---|---|---|---|---|
| today's run | 2026-04-24 → 10-23 | **0.91** | −5.8R | 107 |
| today's run, matched cutoff | → 10-09 | 0.99 | −0.8R | 107 |
| git-stored Oct-9 book | → 10-09 | 1.06 | +3.6R | 101 |

Today's honest reading is **0.91 → HALT**. Even the most generous reading (old environment,
old cutoff) was 1.06 — one bad week from the gate. The pre-committed rule has no discretion
clause. **Re-entry per R100 v2: a positive paper month + full revalidation. No live capital.**
Second rule (two consecutive months < −8R): not tripped (Jun-2026 −11.1R stands alone).
Monthly ledger (today's book, PF@c): Mar +26.0/2.51 · May −7.8/0.71 · **Jun −11.1/0.11** ·
Jul +1.2 · Aug −1.5 · **Sep +15.8/3.08** · **Oct (partial) −4.8/0.37** (9 trades, 2 TP / 4 SL).
2026 YTD: +21.2R, PF@c 1.20, maxDD@flat-1% −18.3%. Full book (2025-06→): +89.9R / 293.

## 2. REPRODUCIBILITY DEFECT FOUND (new audit item, does not invalidate causality)
Trade-level diff vs the git-stored Oct-9 book: 10 historical trades differ (−6 on 2025-07-30
as one correlated batch, +16 scattered) → book R through Oct 9 moved +133.0 → +123.0.
Verified NOT a look-ahead: thresholds are trailing-window-recalibrated strictly before each
block (ql_engine.build_signal_mask), SVM preds walk-forward (fit on X[:i] only), keep-threshold
selection-period-only. Root cause: **sklearn/libsvm environment drift between sandbox sessions
re-rolls borderline SVM keep decisions**; correlated same-hour signals flip as batches.
Consequence: cross-session book variance ≈ ±10R; the kill-switch reading itself moved
1.06 → 0.99 on the matched window. **Action: pin library versions; treat historical trade
counts as ±3% fuzzy; judge gates on today's-environment readings only.**
(Ledger note: CHAMPION.md's 2026 row said n=151/+22.4R/1.27; its own reference CSV holds
n=165/+27.0R/1.30 — discrepancy predates today, unreconciled, superseded by this report.)

## 3. PRE-REGISTERED RR 2.0 CANDIDATE: **NOT PROMOTED** (gate void; stays challenger)
Fresh-data-only decision rule (R100): data after the Oct-9 pre-registration. That window holds
n=6 trades — statistically void. In-environment book evidence (recorded for the re-entry
revalidation bundle): paired over 263 common entries, RR2.0 beats RR1.5 in 98% of 5,000
bootstrap resamples; 2026: +37.4R/PF@c 1.36 vs +17.4R/1.18; RR2.0's rolling 6-mo PF@c = 1.05
(above gate) vs 1.5's 0.91. Counter-evidence: RR2.0's October is worse (−8.8R vs −4.8R — in
chop the 2×ATR target round-trips to SL more often) and its book is +127R vs +90R only because
2025H2's trend regime flatters wider targets. Decision: **no config change while halted**;
RR2.0 enters the re-entry revalidation as the pre-registered challenger — promote only if the
fresh-data gate can actually be run (≥ ~30 post-decision trades) AND its 6-mo PF@c holds ≥ 1.0.

## 4. OPERATING STATE AFTER THIS RECHECK
- CHAMPION (RR1.5 q0.55): **HALTED for live capital** per pre-committed rule. Paper-tracking ON.
- Re-entry checklist: positive paper month → full revalidation (incl. environment pin,
  RR2.0 fresh-data gate, audit's SL-first intrabar bound) → then, and only then, resume.
- demo_bot remains NEVER live-tested (unchanged); $200–300 stays out of the market.
- Next scheduled touch: end-2026 re-evaluation (charter), or re-entry checklist completion.
Reproduce: `python3 scripts/refresh_okx.py && python3 quantlab_r096_recheck.py`
(RR2.0: `python3 quantlab_champcheck_rr2.py`). Curves/trades: `quantlab_output/*trades*.csv`.
