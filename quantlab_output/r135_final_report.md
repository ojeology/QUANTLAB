# R135 — TRENDLINE BREAK, MECHANIZED — FINAL (2026-11-05)
Origin: independent replication of Bryme QuantLab's published claim ("trendline breakout can't
beat a coin flip") under charter. Prespec locked pre-run (docs/RESEARCH_FRONTIER_R135.md).
Universe: 20 stack coins, OKX dailies 2023-06→2026-11 (n=1,254/coin). One-look ledger: crypto
daily panel used for the trendline cell for the first time.

## RESULT: ALL GATES PASSED — FIRST NEW CANDIDATE SINCE THE CHAMPION (candidate, not promoted)
w5 primary: **n=1,076 · WR 33.7% · PF@c 1.42 · +229.1R** — every ledger year positive
(2023 +44.5 / 2024 +81.3 / 2025 +38.1 / 2026 +65.1), halves 1.44 / 1.41.
w3 sensitivity: PF@c 1.18, FAILS yearly law (2025 −45.8, 2026 −25.1) → edge decays with line
hyper-sensitivity, exactly as a real effect would (artifact would not care).

## FORENSICS (all clean)
- R dist: mean +0.213R, median −0.387R, max +16.7R, min −6.7R — no monsters; 10 trades >8R (0.9%).
- Anatomy = design intent: winners hold 10.0d (time stop), losers 3.4d (fast re-cross); avg win
  +2.12R vs avg loss −0.76R (2.80×) — the edge IS the exit asymmetry.
- Year-block bootstrap (2,000 draws): total-R 95% CI [+165, +293] — excludes 0.
- LOAO: PF@c 1.37–1.49 across 20 leave-one-out panels; top-5 winners = 29% of net; 18/20 symbols ≥1.0.
- Where it lives: LONG breakups PF@c 1.68 (+185.2R, positive all 4 years) vs SHORT breakdowns
  1.16 (+43.8R, negative in 2023/2025). High-beta alts carry it: SOL 2.91, SUI 2.64, ADA 2.49,
  ATOM 2.23 — BTC itself 0.97 (the universe law again; the majors are the market, not the edge).
- Independent replay audit: 5 random trades re-derived from raw CSVs with a second implementation —
  pivots, rising/falling condition, fresh-cross (both bars), line projection, exit, ATR, costs:
  **5/5 EXACT.** (First build logged exit-ts only; book now carries entry/exit/hold. PnL unchanged.)

## MECHANISM (consistent with the lab's own physics)
A break of a long-standing diagonal = an expansion trigger (R130: expansion begets expansion);
the re-cross exit is a fast-fail filter that cuts the false triggers in 3.4 days and lets true
regime changes ride 10. R131 died entering after ANY 1H shock with no selection; R135 selects on
structure. The line is not the edge — the line's FAILURE MODE is the signal, the hold rules are
the edge. Reconciliation with the channel's coin-flip result: single pair + symmetric exits =
coin flip; breadth + asymmetry = PF 1.42. Both statements are true.

## STATUS & NEXT (discipline unchanged)
- CANDIDATE, NOT PROMOTED: backtest PF is never live-ready; no capital; champion stays frozen.
- Paper-tracking starts next session (signal log appended to the ritual recheck).
- Jan-1 checkpoint agenda: (a) champion 6-mo gate; (b) R135 forward window review.
- Costs note: daily bars → ~0.10% RT vs multi-% moves; the cost wall (R085–R095) is far here.
Artifacts: r135_trendline_trades.csv (entry/exit/hold/dir/r). Repro: quantlab_r135_trendline.py.
