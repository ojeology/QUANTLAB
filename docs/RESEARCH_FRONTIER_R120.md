# R120 — OI-CHANGE MOMENTUM · 2023 DISCOVERY LEG — PRESPEC (LOCKED 2026-10-24, pre-data)
Status when locked: ZERO 2023 OI numbers computed or seen anywhere in this lab (R116 used
2024→2026 only). Metrics cache empty; funding spine starts 2023-09. Nothing below may change
after the first 2023 byte is analyzed.

## Hypothesis (the one cell R116 left untested)
Coins whose open interest is rising over the trailing week continue to outperform coins whose
open interest is falling over the next week (crowding momentum, daily granularity).

## Data
- OI: Binance Vision daily metrics, `sum_open_interest` (contracts). Daily OI = LAST 5-min
  snapshot of each UTC day. Panel: the 24-coin metrics universe. Fetch window 2023-01-01→2023-12-31 ONLY.
- Prices: `quantlab_cache/bnfut/{SYM}_1D.csv` closes (2020→2026-09, already on disk).
- Funding: Vision monthly fundingRate, extended back to 2023-01-01 for this round.

## Signal (single registered cell)
ΔOI_5d = OI(t)/OI(t−5) − 1 (5 trading days, percentage change — size-neutral).
Formations: every 7 calendar days; FIRST formation ≥ 2023-01-08; LAST formation ≤ 2023-12-24
(so every hold ends ≤ 2023-12-31 — zero 2024 data touched, kill-switch honored literally).

## Portfolio construction
- Eligible at a formation: valid OI today and 5 days prior; last OI observation ≤ 3 days old;
  ≥ 30 days of metrics history. If < 12 eligible coins → skip formation (disclosed count).
- Rank by ΔOI_5d. QUARTILES. LONG top quartile / SHORT bottom quartile, equal weight.
  Weights: +1/q per long coin, −1/q per short coin → net 0, gross 2.0.
- Execution: signal at close t → weights effective close t+1 (LadderAccount law, no exceptions).
- Costs: 0.05%/side, charged ONCE on the execution bar. Funding: −w·f every bar held.

## Discovery gates (ALL must pass ⇒ PROMISING FOR FURTHER VALIDATION)
- G1: net spread CAGR > 0 AND bootstrap P(net return > 0) ≥ 0.95 (5,000 resamples).
- G2: net daily spread Sharpe ≥ 1.0.
- G3: quartile monotonicity — mean net weekly return Q1 > Q2 > Q3 > Q4.
- G4: leave-one-coin-out (24 leave-outs): minimum net spread > 0.
Reported but NON-selecting: ΔOI_3d and ΔOI_10d sensitivities. If 5d fails while a sensitivity
passes, the round is INCONCLUSIVE — never a lookback switch.

## Verdicts (P6 vocabulary, this round only)
REJECT / INCONCLUSIVE / PROMISING FOR FURTHER VALIDATION / READY FOR FORWARD PAPER TEST.
2023 is DISCOVERY ONLY: a pass says nothing about tradeability. Ladder after a pass:
2024–2025 walk-forward → 2026-H1 → (weakened, 5×-exposed) holdout — each only if the prior rung passed.
Charter monetization gates (posW/worstW/maxDD) apply at monetization, not here.

## Kill-switch (pre-committed)
Any gate failure (G1–G4, barring the INCONCLUSIVE note above) closes the OI direction for this
lab: no 2024+ metrics are analyzed for this hypothesis, and the negative result is reported plainly.


## RESULT (2026-10-24) — REJECT, kill-switch executed
All four gates failed on the registered 5d cell: +2.91% net / Sharpe 0.26 / bootP 0.602 /
non-monotone quartiles / LOAO-min −2.56%. 10d sensitivity reverses sign (−55.7%) — diagnosed
broad (not a glitch), factor fragile across horizons. Direction CLOSED per kill-switch; 2024+
metrics unanalyzed for this hypothesis. Full report: quantlab_output/r120_final_report.md.
