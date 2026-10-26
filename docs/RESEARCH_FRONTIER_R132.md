# R132 — THE SEVEN FRONTIERS — PRESPEC (LOCKED 2026-10-26, pre-pairing)
Disposition of operator's list (all seven): 3 BLOCKED (liquidity-provision/L2, VRP/options, surprise
pure-form/consensus), 1 CLOSED (session transitions — R130 conditional lift ≈1.0, F-series, R127/128),
1 CLOSED simple-form (event response, R111). THREE OPEN CELLS REGISTERED BELOW (all challenger-grade,
R129 overlay template, forward-only promotion).

## CELL S — INTRADAY SEASONALITY RETURNS (registered battery)
73-coin 1H panel 2024-01→2026-10. For each UTC hour h: mean hourly return across all coin-hours;
split DISC 2024→2025-09 / CONF 2025-10→2026-10. Strategy forms: (i) long-at-h/flat otherwise for
the best DISC hour AND worst DISC hour (avoid), (ii) day-of-week at daily granularity (long-only).
Costs 0.07%/side per position change. Gates (per form): DISC Sharpe ≥ 0.5 ∧ CONF same sign ∧
n ≥ 100 position-changes. Default expectation: dead (prior anatomy: hour means ≈ 0.02-0.03%).

## CELL M — MACRO LIQUIDITY GATE (registered, champion overlay)
Data: Yahoo DX-Y.NYB (DXY daily) + FRED DGS10 & T10YIE (→ 10y real yield, daily). Gates evaluated
on the champion kept book (entry−1d causal values vs expanding median, min 90 obs):
- M1 dollar-soft gate: 30d ΔDXY < expanding median → trade else skip
- M2 real-yield gate: 10y real yield 30d change < expanding median → trade else skip
R129 gate template: G1 gated PF@c > full on PRIMARY (2026 slice) ∧ G2 retention ≥ 55% ∧
G3 same ordering on CONSIST (2025) slice ∧ G4 gated full-book PF@c > incumbent.

## CELL X — BTC-TRANSMISSION GATE (registered, champion overlay)
R130 atlas receipt: after a BTC −0.5% hour, alts' big-DOWN lift = 2.06 (strongest in atlas).
Hypothesis: alt pops entered during BTC-dump hours fail. Gate: skip champion trades where
BTC's 1h return at entry hour < −0.5%. Same G1–G4 template. Report-only: the mirror (trade ONLY
BTC-dump entries) and BTC-up conditioning.

## Verdict rule
Per cell: PROMISING AS OVERLAY (challenger stack, forward gauntlet) / CLOSED. One run, no iteration,
no re-windowing. The seven-frontier table (dispositions + results) is the deliverable.
