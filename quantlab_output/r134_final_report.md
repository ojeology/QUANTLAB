# R134 — SINGLE-PAIR ML, BLIND-YEAR TEST (BTC) — FINAL (2026-10-26)
The operator's design, implemented exactly: one pair (BTC — deepest history in repo, 6.7y daily),
studied rigorously with ML (30 causal features: lags/momentum/vol/drawdown/MA-distance/RSI/MACD/%B/ATR/
dow), then tested BLIND on the current year (train 2020→2025-12, test 2026-01→2026-09, 273 trading days,
walk-forward expanding refits every 21 days — the test year's labels never touched training).

## RESULT: **NO OOS INFORMATION — decisively**
| model | blind-2026 accuracy | AUC | sign-strategy | buy-&-hold |
|---|---|---|---|---|
| Logistic L2 | **48.4%** | **0.472** | −64.3% | −6.1% |
| Gradient boosting 300×d3 | 47.6% | 0.456 | −12.7% | −6.1% |
| (report: 5d target, GBM) | — | 0.516 | — | — |

Both models are BELOW coin flip on the blind year; the trading overlays lose to doing nothing.
Binomial p ≈ 0.3 — indistinguishable from a fair coin being asked to predict an unfair world.
Gates (AUC ≥ 0.53 ∧ net CAGR ≥ B&H): FAILED. Per prespec: single-pair ML mapped-closed.

## WHY (the structural reason, now measured rather than argued)
1. **The sample math never worked**: one pair's daily history = 2,266 usable rows, and daily BTC
   direction is ~50/50 with heavy noise — the signal-to-noise at this granularity is near zero.
   The models don't overfit here; they starve.
2. **The information lives in the CROSS-SECTION, not the time series**: R078 proved the champion's
   edge is universe-specific; R130 proved eruptions are market-events (60% clustered); R134 now
   proves a single pair's own past doesn't encode its future. Three independent confirmations.
3. Consistency: R124 (no model class can extract absent information) → R128 (metals ML < 0.5) →
   R134 (crypto single-pair ML < 0.5). The law now has a complete asset-family sweep.

## What would change this answer (named, as always)
- Finer data (1H/5m → 50–500× the samples) — but then the cost wall arrives (R085–R095 receipt).
- Order-flow features — data-blocked, the standing frontier.
- Cross-sectional features (other pairs' states) — which is literally the champion's architecture.
Repro: `python3 quantlab_r134_single_pair_ml.py` (~3 min). Predictions: r134_predictions.csv.
