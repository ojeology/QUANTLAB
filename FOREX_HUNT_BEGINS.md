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

## ROUND R108 PRESPEC (2026-10-11, locked BEFORE testing) — F2: FX CARRY

**Hypothesis (who pays):** interest-rate differentials are a contractual, centuries-documented
premium (Lustig–Verdelhan): speculators borrowing low-yield currencies pay you to hold
sudden-devaluation risk of high-yield ones. Carry is CONTRACTUAL income + price risk — the
exact shape the R103 law demands.

**Data spine (dual, adversarial):**
- S1 "fresh": FRED/OECD immediate monthly rates (IRSTCI01*), ffilled daily. CH ends 2024-01,
  NZ ends 2024-10, AU suspiciously flat → supplemented by documented central-bank schedules
  (printed in report) and flagged.
- S2 "stress": supplemented currencies frozen at last FRED-verified value (CH 1.45, NZ 4.77,
  AU 4.35 as printed). Verdict must hold on BOTH spines or candidate is inconclusive.

**Universe:** all 28 synthetic crosses of the 8 majors (inner-join timestamps, close-only).
Synthetic ATR = 1.5 × mean|Δclose|24 (disclosed approximation — no synthetic H/L).
**Rule:** long-only top-K by carry (r_base − r_quote), K ∈ {3,5}, rebalance ∈ {weekly, monthly}.
**Costs:** 1.5 pip RT per position change + 0.7 pip/day swap-broker take on notional.
**Risk:** 0.5% per position, bucket 2.5×ATR (same machinery as R107, off-by-one guard + integrity assert inherited).
**Gates:** charter 6 + holdout PF≥1.1.
**VALIDATION ANCHOR (prespec'd):** week of 2024-08-05 (yen-carry unwind crash) MUST rank among
worst weeks — if it doesn't, the test is broken and no verdict is valid.

## ROUND R108 RESULT (2026-10-11) — F2 FX CARRY ❌ REJECTED AS CORE (premium validated)

21 synthetic crosses × 3 years × dual rate spine (FRED/OECD + documented CHF/NZD/AUD supplements; stress-frozen spine).
Best variant (K=5 monthly): yearly +46.9R / +39.6R / +112.2R — **NO LOSING YEAR, both spines** —
but posW 60% ❌, worst week −27.9% ❌, maxDD −47% ❌. Carry's weekly sign = drift < noise; steamroller tail structural.
**ANCHOR PASSED: 2024-08-04 unwind = worst week in sample — test independently reproduced history.**
4 audit defects caught (cache wipe, ranking bug, mark hashing, zombie positions).

**LAW: contractual premia survive retail costs ONLY when hedged. Unhedged carry = real income, unfit for consistency charter.
R106 hedged floor = carry domesticated — the two halves of the hunt now confirm each other. F3 (daily-bar CTA trend) queued as FX hunt finale.**

## ROUND R109 PRESPEC (2026-10-11, locked BEFORE testing) — F3: CTA-HORIZON TREND (finale)

**Hypothesis (who pays):** the documented medium-term trend premium (Moskowitz–Ooi–Pedersen
time-series momentum; Hurst–Ooi–Pedersen century evidence): institutions reprice slowly
(anchors, herding), central banks smooth deviations, option/reserve flows force continuation.
Payers = everyone anchored to stale prices. This is the horizon R107 never reached (it tested
1.5–8 DAY breakouts; CTA premium lives at 1–12 MONTHS).

**Data upgrade (disclosed):** Yahoo interval=1d range=max ≈ a decade of daily bars per pair
(1H cache alone cannot warm up 252-day lookbacks). **Test window unchanged: 2024-01→2026-10
(the charter three-year bar). 2016–2023 printed as supplementary context ONLY — gates unchanged.**

**Rule:** daily close-only Donchian, always-in long/short flip, N ∈ {20, 60, 120, 252} trading days.
Bucket 2.5×ATR(24d); risk 0.5%/pair; costs 1.2 pip RT + 0.7 pip/day swap (swap now material —
CTA holds are weeks-to-months; that toll is part of the question).
**Gates:** charter 6 + holdout PF≥1.1. Selection 2024-01→2026-03 only; holdout 2026-04→10.
Warmup disclosure: N=120 starts ~mid-2024, N=252 starts ~2025 — capability-adjusted yearly
ledger printed alongside raw. Soft anchor: best variant must be net-LONG USDJPY during
Jan–Jul 2024 (the documented yen slide) — if not, inspect for data bugs before verdict.

## ROUND R109 RESULT (2026-10-11) — F3 CTA-HORIZON TREND ❌ REJECTED — **FX HUNT CLOSED**

Decade of daily bars (2016-10→2026-10, 2603/pair), close-only Donchian N∈{20,60,120,252}d, always-in.
Best (N=252d): 2024 +13.3R / 2025 −2.6R / 2026 −39.8R → 1/6 gates; holdout −35R. Shorter N: 0/6, all years negative.
Anchor PASSED (net-long USDJPY Jan–Jul 2024, +1,927 dir-days; the one real trend paid +22R — the test saw it).
Decade autopsy: gross −20.9R/41 trades (premium ABSENT on majors) + toll −88R (swap −2.15R/trade at 594d holds).
**2026: premium appeared (+46R gross) → swap confiscated 37R of it.** Retail toll scales with horizon: spread kills short, swap kills long.

### ══ FINAL LAW OF THE FX HUNT (F1+F2+F3 complete) ══
**No behavioral FX premium exists for retail at any horizon. The one contractual premium (carry)
has an unhedgeable-by-charter tail unhedged. Contractual premia survive retail costs ONLY hedged —
which is exactly the R106 crypto floor.** After total famine on every other frontier,
STACK v2 stands as the only live machine — found by fundamentals, verified by adversarial process.

→ Remaining standing work: STACK v2 12-week paper trial (≥70% green, worst ≥−1.5% → size up);
champion frozen re-check end of 2026; carry-tail research (options) = the only open FX frontier.
