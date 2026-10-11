# 🏁 QUANTLAB CLOSED → 🚀 FOREX HUNT BEGINS

**Date:** 2026-08-08

---

## PART 1 — QUANTLAB (CRYPTO): OFFICIALLY ENDED

### Final verdict after 23 research runs (R073 → R095)

| Market / TF | Verdict | Evidence |
|---|---|---|
| Crypto 5-minute | ❌ **NO EDGE** | 7 independent attempts: rules (R089), range-fade (R090-corrected), new indicators (R091), win/loss forensics (R092), bank logic (R093), combos (R094), advanced ML (R095). Every one fails after 0.05% costs. Structural: the 5m bar is too small vs retail cost. |
| Crypto 1-hour | ✅ **VALIDATED EDGE** | **SVM q0.75 on 73 symbols**: ~10.4 t/mo, PF 1.94 (1.62 @cost), ~70% profitable months, worst losing streak 2, holdout-validated on untouched 2026. |

### The ONE validated strategy (saved, not forgotten)

**FINAL LOCKED CONFIG (crypto 1H):**
- Signal: Family A compression-then-pop (BBW_STRICT + RV_LO + DST_NR + PRG_VH), rolling 500-bar thresholds recalibrated every 168 bars
- Entry: signal-bar close (E6), green gate, rel_vol > 1.5
- Filter: ML-SVM (RBF) walk-forward, keep top 75% (q=0.75), 14 features
- Universe: 73 symbols (52 original + 18 + 3)
- Exit: SL 1·ATR / TP 1.5·ATR (RR 1.5), TP before SL, no time stop
- Risk: 1% per trade
- **Re-run at end of 2026 with fresh data → decide if worth pursuing in 2027**

### What QUANTLAB taught us (the playbook for Forex)
1. Walk-forward only — never train on the future
2. Holdout untouched until the final verdict
3. Cost gates — a strategy that dies at 0.05% is not an edge
4. Causal audit on every new indicator (lookahead bugs are real — R090)
5. ML amplifies edge; it cannot create one from a fair coin
6. Edge is scarce — most hypotheses fail, and that's a finding

---

## PART 2 — FOREX HUNT: BEGINS NOW

### Why Forex is the right next arena
- **15–20 years of free deep data** for majors (Dukascopy/HistData) — solves the data scarcity that limited crypto 5m
- **Better cost-per-move** — ~0.5–1 pip spreads, commission-free options → edges survive where crypto 5m died
- **Real structure to exploit** — London/NY session opens, daily ranges, rollover rhythms
- **The full QUANTLAB pipeline transfers unchanged** — walk-forward, holdout, cost gates, causal audit, ML filters

### Forex hunt protocol (locked, non-negotiable)
1. **Universe:** 6–8 major pairs first (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP)
2. **Timeframes:** 1H and 4H first (not 5m — don't repeat the cost-wall mistake)
3. **Data:** ~10 years of 1H from Dukascopy (free), stored in quantlab_cache as parquet
4. **Costs:** spread + swap/rollover modeled explicitly in every backtest
5. **Validation:** selection ≤ 2023, holdout 2024-2025 untouched, bootstrap + LOO + MC, causal audit, cost gate
6. **Success bar:** holPF@cost > 1.1, selection n ≥ 200, survives every check

### Next steps
1. Build forex data fetcher (Dukascopy 1H/4H)
2. Fetch ~10 years for 8 pairs
3. Run session/trend/range hypotheses through the pipeline
4. Apply the ML filter if a raw edge appears

---

*"The goal isn't to find a strategy that looks good. The goal is to find one that survives every attempt to prove it wrong."*

---

## 🔁 REVIVAL — R107 (2026-10-11): THE THREE-YEAR TEST

**Data reality (disclosed):** Dukascopy TLS-dead, HistData pipeline heavy, stooq JS-walled.
**Working source: Yahoo 1H, 2023-12-25 → 2026-10-09 = three COMPLETE calendar years (2024/25/26).**
Yahoo bars are aggregated quotes — acceptable for research, costs modeled conservatively on top.

**THE HUGE HYPOTHESIS (stated before testing):** the QUANTLAB playbook transfers to majors —
a close-only, cost-honest basket of FX premia can survive three full calendar years
(2024, 2025, 2026) without a losing year.

**Who pays (locked per candidate):**
- F1 Donchian trend: institutional slow repricing + central-bank smoothing (behavioral, CTA premium)
- F2 carry (next round): rate differentials (contractual)

**GATES (prespecified, cannot change after seeing results):**
1. Fills at bar CLOSES only — zero intrabar path ambiguity
2. Costs: 1.2 pip round-trip + 0.7 pip/day swap haircut per open day
3. Positive weeks ≥ 65%
4. Worst week ≥ −3% (at 0.5% risk per pair-position)
5. Max drawdown ≤ 12%
6. **Each calendar year 2024/25/26 ≥ 0R (the user's survival bar — absolute)**
7. Variant chosen on selection window 2024-01→2026-03 ONLY; holdout 2026-04→10 untouched until verdict; holdout must itself be ≥ 0R and PF@cost ≥ 1.0
8. A failed candidate is a finding, not a failure

## ROUND R107 RESULT (2026-10-11) — F1 DONCHIAN TREND BASKET ❌ REJECTED

Yahoo 1H × 8 majors × 3 complete years (2024/25/26), close-only, 1.2pip+0.7pip/day, 0.5%/pos.
Best variant (N=200h): 2024 −84.9R · 2025 −349.9R · 2026 +0.7R → **0/6 survival gates; all 8 pairs negative; holdout PF 1.07.**
Autopsy: gross edge −211R (dead BEFORE costs) + costs −223R. Breakouts chopped at 3–20d horizon.
Audit: 3 bugs caught this round (bad ticks scrubbed; MTM off-by-one → daily≡trades assert added; double-scale printer fixed).

**LAW ACCUMULATED: behavioral FX premia are dead at retail horizons. F2 = CARRY (contractual) — queued next. F3 = CTA-horizon daily trend — last word on trend after that.**
