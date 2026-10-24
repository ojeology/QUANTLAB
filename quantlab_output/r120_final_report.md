# R120 — OI-CHANGE MOMENTUM · 2023 DISCOVERY LEG — FINAL (2026-10-24)
Prespec: docs/RESEARCH_FRONTIER_R120.md (LOCKED before any 2023 byte was analyzed).
Data fetched for this round: 2023 daily metrics (24 coins, 364–365/365 days; SUI 242, ARB 284 —
listing dates), funding spine extended to 2023-01-01. Signal: 5-day %ΔOI, weekly L/S quartiles,
net 0 / gross 2.0, LadderAccount law (exec t+1, costs once @0.07%/side standard, funding −w·f).

## REGISTERED CELL RESULT (5d lookback, 47/51 formations, avg 21.7 eligible coins)
- Net total **+2.91%** on the year · ann +2.9% · vol 36.8% · **Sharpe +0.26** · maxDD −32.2%
- Bootstrap P(net>0) = **0.602** (weekly blocks, 5,000 draws)
- Quartile means (gross fwd week): Q1 +1.84 · Q2 +1.56 · Q3 +1.64 · Q4 +1.52 → **non-monotone** (Q2<Q3); Q1−Q4 spread ≈ 0.3pp/wk
- LOAO min = **−2.56%** (dropping TRX flips the book negative) → coin-dependent

| Gate | Threshold | Observed | Verdict |
|---|---|---|---|
| G1 | CAGR>0 ∧ bootP≥0.95 | +2.9% ∧ 0.602 | **FAIL** |
| G2 | Sharpe ≥ 1.0 | 0.26 | **FAIL** |
| G3 | Q1>Q2>Q3>Q4 | non-monotone | **FAIL** |
| G4 | LOAO-min > 0 | −2.56% | **FAIL** |

## VERDICT: **REJECT** — all four discovery gates fail. KILL-SWITCH EXECUTED:
the OI-change-momentum direction is CLOSED. Per prespec, no 2024+ metrics are analyzed for this
hypothesis, ever. The 2023 discovery window was examined exactly once, as designed.

## Sensitivities (non-selecting, reported per prespec)
- 3d: +10.9% net, Sharpe 0.28 — still far below gates (and it is NOT a licensed switch).
- 10d: **−55.7%**, Sharpe −1.43. Diagnosed before acceptance: no single-bar anomaly (worst day −8.3%),
  losses spread across ≥5 weeks, LOAO stays ≤ −62% under every exclusion → broad, not a data glitch.
  Reading: at 10-day horizon the factor REVERSES hard — 2023's recovery was led by low-OI
  "abandoned" coins while crowded high-OI names lagged. A sign-flipping factor across horizons
  is the definition of fragile; discovery did its job.
- Costs @0.05%/side (prespec literal): +5.86% / Sharpe 0.34 — still nowhere near any gate.
  The harsher 0.07% standard (incl. slippage) is THE registered number.

## What this means for the lab
R116's "OI-change = momentum" discovery does NOT survive its honest test on untouched data —
the 4th consecutive direction (squeeze, sniper×3, momentum-DD, now OI) to die at discovery or
gates under corrected accounting. The audit-built machinery worked exactly as intended: prespec
locked pre-data, single registered cell, kill-switch executed without discretion.
Repro: `python3 quantlab_r120_oi_momentum.py` (~15 s; caches bnmetrics/ + bnfut/).
