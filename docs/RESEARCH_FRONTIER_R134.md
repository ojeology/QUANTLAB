# R134 — SINGLE-PAIR ML, BLIND-YEAR TEST (BTC) — PRESPEC (LOCKED 2026-10-26, pre-run)
Operator design: study ONE pair rigorously with ML, then test BLIND on the current year.
Resolved to: BTC (deepest daily history in repo: 2020-01→2026-09 Binance perp dailies).
GRADING (locked): train 2020-01→2025-12-31 · BLIND TEST = 2026-01-01→2026-09-30 (the current year,
data already on disk but never used for BTC daily ML). In-sample-2026 is forbidden — that is the
entire point of the round. Single refit at prediction time is allowed ONLY on walk-forward terms
(no test-year labels ever enter features/training; expanding walk-forward mandated).

## Features (locked; strictly lagged: value at day t uses data ≤ t-1 close)
lags of daily return: 1,2,3,5,10,20 · momentum: 20d, 60d, 120d · vol: 10d, 30d, 60d realized ·
distance from 20d/60d/120d high & low (as %) · SMA20/50/200 distance · dow (one-hot) ·
RSI14 · MACD hist (12/26) · Bollinger %B (20,2) · ATR14/close. Total 24 features, all standard,
all computable causally.

## Targets (locked)
t1: next-day direction (up/down) — PRIMARY
t2: next-5-day direction — secondary (report)
## Models (locked, from the R124 ladder — no new model classes)
logistic L2 (C=1) · gradient boosting (300×d3) · (reference: R124's R124-law expectation)
Walk-forward protocol: expanding window, refit every 21 trading days, predict next block.

## Metrics & gates
- PRIMARY (t1, next-day): OOS-2026 accuracy, AUC, andMcNemar p vs coin-flip;
  economic check: sign-strategy daily returns × (long-if-up, cash-if-down) vs buy-and-hold, net of
  0.05%/side on position changes.
- Gates for "ML HAS INFORMATION ON THIS PAIR": OOS AUC ≥ 0.53 AND sign-strategy net CAGR ≥ B&H CAGR.
- Report-only: t2, per-month 2026 accuracy, feature importances (GBM), confusion at threshold 0.5.

## Verdicts
PASS gates → "single-pair ML carries OOS information on BTC" (then: extend design to more pairs).
FAIL → single-pair ML mapped-closed (consistent with R124's law; the sample math gets a number).
One run, no iteration, no feature addition after results.
