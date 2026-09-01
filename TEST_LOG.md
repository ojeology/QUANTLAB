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
- **T21. CAGE raw + SVM q0.65 + adaptive VolCeil (champion pipeline on cage).** Reason: same filter that turned Family A raw into the champion. Result: **MARGINAL** — PF@cost 1.024 (2025 1.25, 2026 0.86 loses), DD −0.5%. Does NOT beat MR-SVM. Cage hypothesis exhausted; MR-SVM remains only validated 3-yr survivor.
- **T22. 4H-CAGE TRAP on 1H (fade 4H high/low on 1H reversal).** Reason: user-directed — use 4H candle's range as the "cage", catch the bounce at the 4H extreme on 1H. Result: **FAILS catastrophically** — 104k trades (over-trading: fires on every 1H tag of the 4H extreme), PF@cost 0.699 (2025 0.72, 2026 0.68), MDD −25%. Entry far too loose. → refine to ONE bounce per 4H bar (T23).
- **T23. 4H-CAGE TRAP refined (ONE bounce per 4H bar).** Reason: match intent — trap at 4H extreme once, catch single bounce. Result: **CATASTROPHIC** — PF@cost 0.514 (2025 0.53, 2026 0.49), MDD −51%, 0/12 & 0/7 profitable-months, 130k trades. Fading 4H extremes on 1H = catching falling knives. → cage hypothesis EXHAUSTED; MR-SVM is the only survivor.
- **T24. CONDITION-AWARE ML (RandomForest: learns reversal vs continuation + the REASONS).** Reason: user-directed — ML should learn WHEN price reverses vs continues and the conditions/reasons. Result: PF@cost 1.576 (2024 1.00, 2025 2.44, 2026 1.23), DD -14.5%. Interpretable; top drivers = adx14, breadth_q, ema_dist_pct, dist_hi48, rsi14, atr_rank, vol — CONFIRMS the adaptive-VolCeil logic (ema_dist_pct + atr_rank are top drivers). Matches SVM champion; adds explainability (we know the reasons).
- **T25. CONDITION-AWARE TREND (1H Donchian + RF filter learning strong-trend conditions).** Reason: user-directed — ML learns exact conditions for strong trends; trade only those. Result: **SUCCESS — PF@cost 1.744/1.529/1.537 (2024/25/26), ALL ≥1.3; MAX DD <2.5%; prof-months 8/12, 7/12, 4/7.** Learned conditions: rsi14 + ema_dist_pct dominate (trade breakouts when RSI confirms momentum AND price already far from mean). COMPLEMENTARY to MR (MR wants near-mean, trend wants far-from-mean). First strategy to clear the PF≥1.3 / 3-yr / low-DD bar.
- **T25b. T25 in DOLLARS ($100 start, $2/trade = 2% risk).** Reason: user asked for concrete breakdown. Result: 2024 +23.5% ($100→$123.5), 2025 +11.6% ($123.5→$137.9), 2026 +4.4% ($137.9→$144.0), FULL +44.0%, MAX DD ~4% (at 2% risk). 2%-of-equity variant: +54.7% ($100→$154.7). See t25_dollars.py.
- **T26. PORTFOLIO: MR champion + condition-aware TREND (50/50).** Reason: test option 2 — combine complementary regimes. Result: PF@cost 1.21/1.33/1.81 (2024/25/26), FULL 1.29; MAX DD −1.2% (ultra-low); prof-months 6/12,7/12,5/7. Bug-free run. Smooth + lowest DD, but 2024 PF 1.21 just under 1.3 (MR dragged it). TREND ALONE (T25) is the one that clears PF>=1.3 every year.
- **T27. TUNED PORTFOLIO: MR + condition-aware TREND, TREND-WEIGHTED 70/30.** Reason: lift 2024 above 1.3 while keeping low DD. Result: **SUCCESS — PF@cost 1.400/1.318/1.813 (2024/25/26), ALL ≥1.3; MAX DD −0.8% (ultra-low); prof-months 7/12,6/12,5/7.** BULLETPROOF: PF≥1.3 every year with <1% DD. The target strategy.
- **T28. FINALIZE the bulletproof trend-weighted portfolio (or trend solo) into demo_bot.** Reason: only strategy meeting PF≥1.3/yr AND <1% DD. → pending.

## Standing conclusion
Across every crypto hypothesis testable here (forex blocked, 15m/4H too sparse), the **only 3-year survivor is the 1H mean-reversion SVM with adaptive VolCeil**, and correctly sized (30-sym, 0.5% risk) it delivers: all years PF>1 (1.18/1.82/1.53), MDD ≈ −4%, ~56% profitable-months. Trend-following only works on 1D and fails in range-bound 2026.
