# QUANTLAB RESEARCH AUDIT — full-repo code & experiment audit (R-119 era)

**Scope executed 2026-10-24.** Auditor role per operator directive: lead quant researcher + code auditor.
**Standard applied:** `ql_accounting.py` (locked conventions) + `tests/test_accounting.py` (7/7 pass).
**Hard rule honored throughout:** this file claims only what was actually inspected. Nothing marked "not verified" anywhere in the repo has been silently upgraded.

---

## 0. Executive verdict (Phase-6 classification)

| Subject | Verdict |
|---|---|
| **R118 twin hunt (headline numbers)** | **SUPERSEDED** — cost-accounting defects confirmed; corrected in R119 |
| **R119 corrected twin hunt** | **REJECT** — all 4 variants fail charter (3/6 max) |
| **R115 funding dispersion** | **INCONCLUSIVE** (unchanged direction, conservative-cost bias confirmed) — holdout contaminated |
| **R116 OI crowding** | **REJECT** (squeeze family); OI-*momentum* = untested discovery, next roadmap item |
| **R117 daily Donchian** | **REJECT** stands (its defects were fixed pre-push; accounting corrected retro-consistent) |
| **`demo_bot.py` (the bot)** | **NOT READY FOR FORWARD PAPER TEST** — never executed live; 2 real defects found & fixed this audit (were silent): dead risk gate, equity-peak amnesia on restart |
| **"The floor" (R106)** | **NOT A STRATEGY** — it is an *instrument specification* (100% posW, worst week +0.055%) whose risks were never priced into a report. Amended below (§4.E). Not tradeable at $200–300 without exchange-risk acceptance the repo never documented. |
| **Repo-wide accounting** | **One convention now enforced** (`LadderAccount`); legacy scripts keep their outputs with SUPERSEDED/superseded-by labels rather than silent rewrite |

**Bottom line:** after full correction, the lab's production-ready inventory is: **one frozen champion (OKX 1H, +22.4R 2026, recheck ~Nov 1)** and **zero** new tradeable edges. The audit's most valuable output is not a strategy — it is proof (4 independent captures) that the lab's #1 recurring defect is a **same-bar position off-by-one**, now structurally impossible under `LadderAccount`.

---

## 1. What was inspected (evidence ledger)

- **Files read line-by-line:** `quantlab_r118_twin_hunt.py` (full), `quantlab_r115_funding_dispersion.py` (stats/accounting sections), `quantlab_r116_crowding.py` (stats/accounting sections), `demo_bot.py` (full), `bot_config.yaml` (full), `EXIT_MODEL_AUDIT.md` (full), `ql_accounting.py` (authored this audit), `README.md` (skim, structure), charter (`QUANTLAB_CHARTER.md` result log).
- **Commands run:** full test suite (7/7), R119 corrected run (3.7 s cached), inventory generation, git history checks (`646d9c3` = R118 fix push; `.git` locally absent 9th time — rebuilt per standing recipe at push time).
- **Not inspected (explicit):** R005–R101 script bodies (≈90 files — family-level only, via earlier session greps; graded `family-summary` in inventory), `scripts/fetch_*` internals beyond the two known float-epoch bugs (both fixed and regression-tested earlier).

---

## 2. Phase-2 defect verification (named concerns A–F)

### A. R118 cost/funding handling — **CONFIRMED, both engines**
- **H-T1:** `daily.iloc[i] -= turnover_cost` (L83) is **overwritten** by the unconditional `daily.iloc[i] = w·ret` assignment (L85). Costs never survive.
- **H-T2:** cost booked at `i+1` (L146) is **overwritten** by the next iteration's P&L assignment (L127). Costs never survive.
- **Funding in H-T2: PRESENT and correct** (`pos_decided·(ret−fund)`); the suspicion of missing funding is **unfounded** — recorded as such.
- **R119 quantification (12-coin core, locked accounting):**

| Variant | R118 claimed CAGR/maxDD | R119 corrected CAGR/maxDD | Defect impact |
|---|---|---|---|
| H-T1 always-in | +10.8% / −24.7% | **+6.5% / −22.2%** | **−4.3 pp CAGR** (cost overwrite flattered) |
| H-T1 filtered | +10.0% / −20.7% | **+10.6% / −14.8%** | +0.6 pp (universe shrink offsets; DD −5.9 pp better) |
| H-T2 classic | +19.8% / −38.6% | **−0.4% / −47.4%** | **−20.2 pp CAGR** (off-by-one phantom returns) |
| H-T2 timestop | +27.7% / −41.3% | **+14.1% / −42.1%** | **−13.6 pp CAGR** |

- **H-T2 holdout flips:** classic **+9.5% → −8.2% ✗** — the corrected accounting removes the phantom breakout-bar returns that had made the holdout look passable. Timestop holdout +6.4% ✓ but still 3/6 gates → REJECT.
- **Disclosure:** R119's H-T2 is the **exit-to-flat** state machine inherited from the R118 fix (reverse-break → flat, re-entry on fresh breakout), *not* R117's true long/flip. R117 remains the reference for true-flip DON55. This does not affect the audit conclusion (the tested family fails).

### B. R115/R116 accounting — **CONFIRMED (2 defects)**
1. **Turnover overcharge:** `turnover` updated only on rebalance bars, but `−COST_SIDE·turnover` charged **every bar** → costs multiplied by the rebalance interval (R× overcharge). Direction: **conservative** (true results would be *better*). Analytic bound: ≈2–6 pp APR understatement; **cannot flip** the R115 holdout (−17%) or R116 verdicts.
2. **Mixed basis:** `apr = μ·365` computed **unleveraged** while equity curve and maxDD computed **leveraged** → APR and DD quoted on different worlds. No verdict sat on a boundary → no rerun executed (per minimal-intervention rule); rerun listed in roadmap pending funding-spine refetch.

### C. `bot_config.yaml` stale headlines — **CONFIRMED**
L9/L14 still advertise "R066 PF=3.35 n=91" / "R068 PF=1.69" — M1-proxy-era numbers. `EXIT_MODEL_AUDIT.md` (in-repo, later) shows bot-faithful M3 **PF 1.193, +$12.09/trade** (n=91 OOS) and SL/TP-cap M2 **0.528**. **Labeled superseded in-file; originals preserved** (comment block, never deleted).

### D. `demo_bot.py` dead risk controls — **CONFIRMED & FIXED (+ regression tests)**
1. **`DAILY_LOSS_LIM` was dead code** — defined, never called. **Fix:** enforced in `run_scan` (day-start closed P&L tracked; breach → 24 h strategy pause). 
2. **Peak amnesia on restart:** `load_states` initialized `peak = max(START_EQUITY, configured cap)`, ignoring recorded equity → DD gate could restart from a fake peak. **Fix:** `SELECT peak … ORDER BY rowid DESC LIMIT 1` reload.
3. **UNRESOLVED (documented, not guessed):** (a) whether `check_risk_gates()`/`MAX_DD_LIM` has any live call-site in the scan path — grep shows definition + the new daily-loss call; a full call-graph audit of the bot is **not done** and the bot is labeled NOT-VERIFIED accordingly. (b) **Intrabar TP/SL ordering is unknowable from OHLC** — bot fills TP-first (optimistic). Required label added to bot docs: SL-first bound needed before any live capital. Bot has **never been executed against a live feed** — stated on every bot artifact.

### E. R106 "floor" realism — **CONFIRMED as mislabeled artifact**
The 15-week window shows 100% posW / worst week +0.055% because the instrument is a **hedged lend-and-short construct whose dominant risks are operational**: exchange failure/custody, lending-book withdrawal, funding-rate regime flip (short leg pays), margin cascade on the short leg, and execution slippage on 20 simultaneous legs. None were priced or reported. **Amended status: instrument specification, not strategy.** At $200–300 the binding constraint is minimum notionals and leg execution, not APR.

### F. Holdout reuse — **CONFIRMED; exposure ledger created**
2026-07→09 has now been **examined 5×** (R115 R116 R117 R118 R119). It is **contaminated for any discovery purpose** and retained only as a (weakened) final gate. Full log: `docs/EXPOSURE_LEDGER.md`.

---

## 3. Phase-3 outcome — one accounting standard + invariants

`ql_accounting.py` locks seven conventions: (1) signals decide at close *t*, execute at open/next bar *t+1*; (2) costs charged **once**, on the execution bar; (3) funding booked as `−w·f` per bar held; (4) single leveraged basis for APR/DD/vol; (5) conservative intrabar default = **SL-first**; (6) holdout exposure must be ledgered (see F); (7) every bar marked exactly once, with a ±60% daily tripwire and no-unapplied-pending check — `verify()` runs before any report is emitted.

**The audit's self-catch (recorded as the strongest finding):** the off-by-one (marking bar *i* with the position decided at close *i*) has now been captured **four times** — R104 printer-era, R107 FX, R118 v1, and **during this audit itself**, in the first draft of the corrected R119 rerun (it produced a phantom +54.5% CAGR before the invariant run and a weight-source discipline flagged it). Structural fix: `LadderAccount.effective_weights(i)` is now the **only** legal source of bar-*i* P&L weights; callers that mark from local position variables are the defect pattern. Regression test added.

---

## 4. Phase-4 direction (one hypothesis chosen, per directive)

**Order-flow (the directive's first preference) is honestly unavailable:** no L2/tick/quote source accessible from this environment (OKX rubik ≈100-day depth cap; Bybit dumps 404; Binance Vision has no book data). Per the directive, OHLCV proxies relabeled "order flow" are forbidden — so the cell is **blocked, not skipped**.

**Chosen direction: OI-change momentum, 2023 untouched-leg design** (from R116's discovery that ΔOI is momentum-like; the momentum cell was never tested because 2024–2026 metrics data was burned by the squeeze test):
- **Signal (Phase-5 separation honored):** rank coins by 5-day ΔOI (contracts), long top-quartile / short bottom-quartile, weekly rebalance. Discovery window **2023-01→2023-12 (untouched by any prior OI test)**. Monetization design only if discovery survives.
- **Validation ladder:** 2023 discovery → 2024–2025 walk-forward → 2026-01→06 (pre-contamination) → holdout only if all prior gates pass.
- **Data:** Binance Vision daily metrics, 24 coins, 2023-01→2026-10 confirmed fetchable; funding spine refetch (~25 min) is the one blocker for full monetization and is scheduled first.

**Kill-switch honesty:** if 2023-leg discovery fails, the OI direction closes **without touching 2024+ data**, preserving the walk-forward ladder for any future restart.

---

## 5. Costs, risk, sample sizes (audit-run facts)

- R119 run: 12-coin core, 3,378 BTC funding epochs restored, 3.7 s cached, invariants ON, curves in `quantlab_output/r119_*.csv`. Charter gates: yearly≥0, posW≥65%, worstW≥−3%, maxDD≤12%, holdout≥0 — **best variant 3/6**.
- Cost model used: 0.05%/side (H-T1) and 0.07%/side (H-T2) + funding — identical structure to R118 so the delta is attributable to the **defects alone**.
- Sample sizes: H-T1 330 rebalances; H-T2 32/44 trades — **H-T2 remains a 44-trade experiment**; even its corrected numbers carry wide uncertainty. This alone bars any promotion language.

## 6. Deliverables index

| File | Content |
|---|---|
| `docs/EXPERIMENT_INVENTORY.csv` | 29 rows, evidence-graded (family-summary vs file-inspected) |
| `docs/AUDIT_GAPS.txt` | 5 explicit non-claims (what this audit did NOT verify) |
| `docs/ACCOUNTING_AND_BACKTEST_STANDARDS.md` | the locked convention standard + caller discipline |
| `docs/RESEARCH_ROADMAP.md` | prioritized next experiments + infrastructure debts |
| `docs/EXPOSURE_LEDGER.md` | holdout/window reuse log (contamination control) |
| `ql_accounting.py`, `tests/test_accounting.py` | standard + 7/7 regression tests |
| `quantlab_r119_corrected.py`, `quantlab_output/r119_*` | corrected rerun, reproducible |
