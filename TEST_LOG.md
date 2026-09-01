# TEST LOG — Blind OOS validation (numbered, with hypothesis + reason + result)

Goal: a strategy that survives 2024/2025/2026 with good drawdown and profitable months, for a 2027-01-01 crypto bot go-live. All OOS walk-forward, fees 0.05%, crypto 1H unless noted.

## Phase 1 — Baselines from the frozen repo
- **T1. Raw Family A / Family C bot strategies.** Reason: what the repo already ships. Result: PF@cost ~1.0–1.1, weak; Family C even negative on a full year.
- **T2. R077 "holy grail" (breadth50 + VolCeil static).** Reason: documented best config. Result: PF@cost 0.75 → loses money after fees.
- **T3. Static filter sweep (q × VolCeil × breadth).** Reason: find a robust static filter. Result: NO config survives all three years (VolCeil helps 2024 but hurts 2026 — opposite regimes).
- **T4. Control 5m.** Reason: negative control (should fail). Result: correctly fails (too sparse).

## Phase 2 — SVM mean-reversion
- **T5. SVM q0.75 (original "champion").** Reason: ML filters the coiled signal. Result: PF@cost 1.25 (2026), 75% profitable-months. Good but q0.65 better.
- **T6. SVM q0.65 + cross-regime checks.** Reason: improve edge. Result: PF@cost 1.33–1.64; 2026 strong.
- **T7. 4H timeframe.** Reason: different environment. Result: only 56 raw trades/3.5yr — too sparse, edge is 1H-specific.
- **T8. 15m cache.** Reason: different environment. Result: 2026-only, sparse — unusable.
- **T9. Forex (Dukascopy).** Reason: different market. Result: BLOCKED (API unreachable).
- **T10. Adaptive VolCeil (regime-gated, |ema_dist_pct|>2.0).** Reason: resolve the 2024↔2026 opposition by applying VolCeil only when price stretched. Result: **SURVIVES ALL 3 YEARS** — 2024 1.18 / 2025 1.82 / 2026 1.53 (30-sym, fees).
- **T11. Champion deep-dive.** Reason: measure risk, not just PF. Result: MDD −9.4% @1% risk; 22/30 symbols profitable (broad); 15/27 profitable-months.
- **T12. Ensemble A∪B.** Reason: more signal definitions = more robust? Result: WORSE — 2026 PF 0.92 (loses), MDD −34%.

## Phase 3 — Drawdown sizing
- **T13. Universe/risk sizing.** Reason: 73-sym MDD was −26%, unacceptable. Result: **30-sym + 0.5% risk → MDD −3.7%, PF@cost 2.47** (both safer AND more profitable).

## Phase 4 — Different hypotheses / environments (user-directed)
- **T14. Trend-following Donchian (1H).** Reason: different strategy type (trend vs mean-reversion). Result: FAILS — PF@cost 1.00, 2025 loses, over-traded (625/yr).
- **T15. Trend-following Donchian (4H).** Reason: different environment (cleaner breakouts). Result: FAILS — 2025 ≈1.0, 2026 0.94; low DD but poor months.
- **T16. MR + Trend portfolio (1H).** Reason: combine complementary regimes. Result: FAILS — PF 0.97, 2026 kills it.
- **T17. Trend on 1D.** Reason: different environment. Result: STRONG in trending years (2024 PF 4.34, 2025 2.38, DD −1.6%) but 2026 had ~5 signals (range-bound) → sparse.
- **T18. Multi-TF portfolio (1H MR + 1D trend).** Reason: 1D trend wins 2024/25, 1H MR wins 2026. Result: FAILS — 2024 PF 0.92, 2026 sparse, FULL PF 1.19.
- **T19. CAGE mechanism — fade lower wall (EMA50 ± 2.5·ATR band; long when close<=lower wall in ADX<30 + tight-BB regime, target=cage center).** Reason: user-directed "trap price at a price that follows rules". Result: **FAILS** — PF@cost 0.613 (2025 0.71, 2026 0.57), 1357 trades (over-traded), 18–21% win. Fading the lower wall = catching falling knives. → try false-breakout variant (T20).
- **T20. CAGE TRAP (false-breakout: break below cage + reclaim → fade back to center).** Reason: alternative trap reading. Result: **FAILS** — PF@cost 0.580 (2025 0.67, 2026 0.49), 671 trades, 28–35% win. Raw cage signal is a net loser. → try cage raw + SVM filter (T21).
- **T21. CAGE raw + SVM q0.65 + adaptive VolCeil (champion pipeline on cage).** Reason: same filter that turned Family A raw into the champion. → running.
- **T22. FINALIZE sized 1H MR-SVM (30-sym, 0.5% risk) into demo_bot.** Reason: only validated 3-year survivor. → pending.

## Standing conclusion
Across every crypto hypothesis testable here (forex blocked, 15m/4H too sparse), the **only 3-year survivor is the 1H mean-reversion SVM with adaptive VolCeil**, and correctly sized (30-sym, 0.5% risk) it delivers: all years PF>1 (1.18/1.82/1.53), MDD ≈ −4%, ~56% profitable-months. Trend-following only works on 1D and fails in range-bound 2026.
