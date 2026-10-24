# RESEARCH ROADMAP (post-audit)

## P0 — infrastructure debts (before any new numbers)
1. ✅ DONE 2026-10-24: funding spine refetched (32 coins × 3,378 epochs, 2023-09→2026-09, 0 fails) + daily panel extended to the full 24-coin metrics universe (24 × _1D.csv, listing-correct). Corrected R115/R116 reruns and R120 OI-momentum are now data-ready.
2. Cache resilience: local mirror of fetched CSVs (9 wipe events to date); checksum manifest.
3. `demo_bot.py`: full call-graph audit of risk gates (check_risk_gates/MAX_DD_LIM unresolved); SL-first intrabar bound; FIRST forward paper run (bot has never touched a live feed).

## P1 — chosen direction (one hypothesis) — **CLOSED 2026-10-24: REJECT (R120)**
R120 2023-discovery executed; all 4 gates failed (Sharpe 0.26, bootP 0.602, non-monotone,
LOAO-min<0); 10d horizon reverses sign. Kill-switch honored — direction closed, no 2024+ analysis.
Next runnable (P2): corrected R115/R116 reruns (funding spine + panel now in cache). After those,
the honest frontier is thin: order-flow remains data-blocked; consider consolidating rather than
forcing a new direction (audit directive: a defensible negative beats a forced positive).

## (HISTORICAL) P1 pre-registration
**OI-change momentum, 2023 untouched-leg design** (see audit §4). Prespec before any 2023 byte is loaded:
signal = 5-day ΔOI ranks, L top / S bottom quartile, weekly rebalance; discovery = 2023 only.
Kill-switch if 2023 fails. Monetization ladder: 2024–25 WF → 2026-H1 → (weakened) holdout.
Blocker: P0.1 funding spine. Data ready: 24 coins × 1,004d metrics.

## P2 — conditional reruns (only if a verdict sits near a boundary)
- R115 corrected rerun (post P0.1): expected shift +2–6 pp APR (conservative bias removal). Holdout −17% cannot flip; expect verdict INCONCLUSIVE→REJECT clarity.
- R116 OI-momentum full monetization only after P1 discovery passes.

## P3 — standing operations
- Champion (OKX 1H, frozen): recheck ~2026-11-01; re-eval end-2026. No config changes.
- Bot: forward paper log ONLY after P0.3. Never with the $200–300.
- Every round: prespec → run → charter verdict → push. Off-by-one law enforced by `LadderAccount`.

## Explicitly closed (do not reopen without a new lever)
Candle-indicator sweeps, Donchian families (R117/R119), squeeze/OI-level crowding (R116),
standard momentum (R113 — real but uninhabitable DD), seasonality (R114 0/20), snipers (0/3),
5m scalping (cost wall), FX arc v1 (F-series), pairs/spreads (R051-R063).
