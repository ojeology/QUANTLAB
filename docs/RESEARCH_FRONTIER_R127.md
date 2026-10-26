# R127 — THE GOLD MAP — PRESPEC (LOCKED 2026-10-26, pre-analysis)
Operator prompt: "real traders are making real profits on gold." Dissection battery. Status when
locked: zero gold SIGNAL tests ever in this repo (gold appears only as 1 of 73 champion symbols —
5 trades since 2025-06, +0.3R). All eras discovery-grade; structure preserved anyway:
SELECTION 2000→2015 · CONFIRMATORY 2016→2026 (the lab's standard shape).

## Data
- Long horizon: Yahoo GC=F (COMEX front-month continuous, daily, ~2000→2026-10-24). Disclosed:
  continuous-contract roll effects exist; acceptable at daily horizons for direction/dissection.
- Intraday: OKX XAU_USDT_SWAP 1H (2024-01→2026-10, on-disk).
- Costs: daily-horizon cells 0.10% round-trip per position CHANGE (conservative for gold
  spread+commission); 1H cells 0.07%/side (lab standard incl. slippage).

## Cells (multiplicity: ONLY D1 and D4 can produce verdicts; D2/D3/D5 are mapping, report-only)
- **D1 — TIME-SERIES MOMENTUM (the trend premium):** sign of past-Nd return, position next day
  (no same-bar), N ∈ {20, 60, 120, 250} — ALL FOUR registered upfront. Two shapes per N:
  long/flat and long/short. GATES per cell: selection-era (2000→2015) Sharpe ≥ 0.5 AND
  block-bootstrap P(total>0) ≥ 0.95 (monthly blocks, 5,000 draws) AND confirm-era (2016→2026)
  Sharpe > 0. All three → PROMISING-candidate (forward-paper gauntlet after that). Else DEAD.
- **D4 — COMPRESSION-POP on gold (the champion's pattern, raw):** signal machinery on OKX gold 1H
  2024-01→2026-10, RR 1.5, SL 1×ATR, costs 0.07%/side. Gates: n ≥ 30 AND PF@c ≥ 1.3 AND
  bootP ≥ 0.95 (disclosed: TP-first engine convention; no 15m gold ground truth in sandbox).
  Almost certainly n-starved — reported honestly either way.
- **D2 (mapping) — the anatomy:** drift/vol/maxDD by era and by decade; drawdown table (incl.
  2011–2015 bear); gold-vs-BTC correlation; THE $300 TRANSLATION (what each annualized figure
  means per month at operator capital); 2026 YTD state.
- **D3 (mapping) — event anatomy:** forward 5d/20d returns after |1d move| > 2σ days
  (selection era), vs unconditional; biggest single-day table.
- **D5 (mapping) — session anatomy:** UTC hour-of-day vol/return profile on gold 1H 2024–26
  (the London/NY prop-trader story), concentration index.

## Verdict vocabulary
Per registered cell: PROMISING-CANDIDATE (rare) / DEAD (default). Overall gold verdict = the
map itself: which doors gold opens. NO cell may be iterated, re-windowed, or re-costed after results.
Never live-ready language; at $300 the translation column is binding context for every number.
