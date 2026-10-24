# R121 — CRYPTO→FX TRANSFER — FINAL (2026-10-24)
Prespec: docs/RESEARCH_FRONTIER_R121.md (LOCKED before any FX signal was scored).
Frozen artifact: r121_crypto_svm.pkl (SVC fit on 296 crypto selection-period trades, 2025-06→12,
win-rate 50.3%). FX: 8 majors, 66.3k 1H bars each (HistData 2016→2026-10, refetched for this round).
Registered cell: champion signal machinery + crypto-brain keep-55% (expanding-window rank, warmup
100), RR 1.5, SL-first fills (5 ambiguous bars found, all booked SL), costs = locked pip table.

## PRIMARY EVAL 2021-01 → 2025-12
| book | n | PF@c | WR | sum R |
|---|---|---|---|---|
| raw FX pops | 292 | 0.96 | 39% | −7.0R |
| **crypto-brain kept 55%** | 164 | **1.15** | 43% | **+13.5R** |
| random-55% (200 draws) | — | q-quantile 0.98 | — | — |

LOO per pair: ALL eight above 1.0 (EURUSD 1.02 … AUDUSD/EURGBP 1.24).
Gates: G1 PF≥1.10 **PASS** (1.15) · G2 n≥100 **PASS** (164) · G3 bootP≥0.95 **FAIL (0.795)** ·
G4 LOO>1 **PASS** · must-beat raw **PASS** (+0.18 PF) · must-beat random **PASS**.
**VERDICT: REJECT** (G3 failed — the transfer effect is not statistically established at the
locked bar). Per prespec kill-switch: crypto→FX transfer direction CLOSES. No iteration.

## Context and honesty notes (reported, non-deciding)
- Context era 2016–2020: raw PF 1.19 (n=298) vs kept PF 1.13 (n=200) — the filter added NOTHING
  before 2021 (slightly negative). The observed transfer benefit is era-specific; that fragility
  is exactly what G3 exists to catch.
- Score sanity: model scores on FX cluster tightly (mean 0.586, sd 0.051) — the brain is only
  weakly discriminating out-of-family, yet its top half sorted winners from losers. Suggestive,
  not establishable at n=164.
- Sensitivities (non-selecting, computed per prespec): q0.70 and RR2.0 runs in r121_fx_trades.csv
  scoring columns; NOT evaluated against gates.

## Standing significance
Strongest near-miss of the audit era: the first time any filter beat BOTH its raw baseline AND
random selection on out-of-family data with full LOO robustness. Also the first REJECT issued
with a smile: the machinery worked, the effect just isn't big enough to bet on at the locked bar.
Repro: `python3 quantlab_r121_extract_crypto_brain.py && python3 quantlab_r121_fx_transfer.py`.
