# EXIT-MODEL AUDIT — QuantLab

**Date:** 2026-08-06
**Status:** ⚠️ STOP-THE-PRESSES finding — affects R066–R071 conclusions, RR decision, and demo-bot deployment expectations.

---

## 1. What was checked

The same frozen signal conditions (Family A: `BBW_STRICT+RV_LO+DST_NR+PRG_VH`, RR=2.0;
Family C: `ADX_ST+PBD_HI`, RR=3.0; IS_RATIO=0.80, 52 symbols, 1H) were run through **five
different exit simulations**. Repro script: `scripts/exit_model_audit.py`.

| Model | Description |
|---|---|
| **M1 PROXY** | R066/R068/R071 baseline engine verbatim: `win = next bar close > entry close`, pnl = ±RR. No SL/TP at all. |
| **M2 SLTP_CAP100** | R072 engine verbatim: entry next-bar close, SL=1·ATR, TP=rr·ATR, intrabar TP-first, 100-bar horizon, OPEN counted as loss. |
| **M3 SLTP_NOCAP** | **Bot-faithful:** entry next-bar close (as `demo_bot.py` does), SL/TP, no horizon, exit checks start on the bar *after* the entry bar, TP checked before SL. |
| **M4 SLTP_NOCAP_INC** | Same as M3 but entry bar included in exit checks (R072 loop style, no cap). |
| **M5 SLTP_SIGENTRY** | Entry at *signal-bar* close (most optimistic entry), SL/TP, no horizon. |

## 2. Results (OOS only)

### Family A (RR=2.0) — 91 signals

| Model | n | unres | WR | PF | Exp $/trade |
|---|---|---|---|---|---|
| M1 PROXY *(frozen baseline)* | 91 | 0 | 62.6% | **3.353** | +87.91 |
| M2 SLTP_CAP100 *(R072)* | 91 | 0 | 20.9% | 0.528 | −37.36 |
| **M3 SLTP_NOCAP *(bot-faithful)*** | 91 | 0 | 37.4% | **1.193** | +12.09 |
| M4 SLTP_NOCAP_INC | 91 | 0 | 20.9% | 0.528 | −37.36 |
| M5 SLTP_SIGENTRY | 91 | 0 | 46.2% | **1.714** | +38.46 |

### Family C (RR=3.0) — 2,044 signals

| Model | n | unres | WR | PF | Exp $/trade |
|---|---|---|---|---|---|
| M1 PROXY *(frozen baseline)* | 2,044 | 0 | 45.8% | **2.534** | +83.17 |
| M2 SLTP_CAP100 *(R072)* | 2,044 | 13 | 16.5% | 0.594 | −33.86 |
| **M3 SLTP_NOCAP *(bot-faithful)*** | 2,044 | 7 | 20.7% | **0.785** | −17.03 |
| M4 SLTP_NOCAP_INC | 2,044 | 7 | 16.6% | 0.599 | −33.46 |
| M5 SLTP_SIGENTRY | 2,044 | 4 | 22.1% | **0.852** | −11.55 |

> MDD% omitted: the ±R fixed-pnl sequence (no capital base) makes % drawdown
> meaningless across models; a proper capital/equity simulation is required for
> comparable MDD — part of the re-validation work.

## 3. Verdicts

1. **The frozen baselines are confirmed proxy artifacts.** M1 exactly reproduces the
   frozen numbers (A: PF 3.353 / WR 62.6% / n 91; C: PF 2.534 ≈ R071's 2.5378).
   R066's `backtest_family` literally computes `is_win = next_close > entry_close`.
   A "win" under the proxy only requires the next bar close to be higher — not price
   reaching +2·ATR before −1·ATR. That is the entire source of the gap.

2. **R072's realistic engine is confirmed.** M2 reproduces R072's Key Stats exactly
   (A: PF 0.528 / WR 20.9%; C: PF 0.594 / WR 16.5%). R072's **Key Stats section is
   correct**; its **Final Verdict Q&A is not** — it deploys the OLD numbers
   ("PF=3.35 with Boot P5=2.44", "46% WR compensated by RR=3.0", "fully deployable")
   against the new engine's own key stats. The verdict never reconciles that the
   realistic engine shows both families below PF 1.0. **R072's deployment
   recommendation is not supported by its own computed stats.**

3. **The horizon cap is NOT the driver.** Unresolved trades are 0–13 out of 91–2,044.
   Removing the 100-bar cap barely moves the numbers. The gap is structural: a
   1·ATR stop with a 2–3·ATR target is hit far less often than the proxy assumes.

4. **Family C is unprofitable under every realistic model** — including the most
   optimistic one (M5 PF 0.852). Its frozen "cleared for paper trading" status
   (R068) is not supported by realistic exits.

5. **Family A is borderline-positive, not exceptional.** Bot-faithful M3: PF ≈ 1.19
   (n=91 → very wide CI, needs bootstrap). Optimistic M5: PF ≈ 1.71. Entry timing
   alone swings it ~0.5 PF. It is a real but *thin* candidate — a far cry from 3.35.

6. **R072's own improvement test points at the fix direction:** "Partial TP 50% @ 1R,
   rest @ 2R" was the best variant on the realistic engine (ΔPF ≈ +1.04 for A, +0.87
   for C → ≈1.57 / ≈1.46 in-sample). In-sample, so it needs out-of-sample
   confirmation, and even so it does not restore the frozen claims.

7. **The demo bot trades the M3 model.** `demo_bot.py` enters at the next closed
   bar and closes on intrabar TP/SL (TP checked first), no time stop. Under M3,
   expected live performance is Family A ≈ PF 1.19 (thin, needs sizing discipline
   and patience given n≈91) and Family C ≈ PF 0.79 (expected loser). Running both
   unchanged will likely bleed on Family C while the research logs show "PF 1.69".

## 4. What this means for the research arc

- R066 "frozen baseline" family ranking, R068 "cleared for paper trading", and
  R071's RR decision were all computed on the proxy and are **not transferable to
  the bot's execution model**.
- The honest current state: **Family A = unproven, thin, positive**;
  **Family C = negative under realistic exits**.
- Every future run must use a realistic SL/TP engine (M3-style, matching the bot),
  or the gap between research and deployment re-opens.

## 5. Recommended next steps (candidate R073 scope)

1. **Rebuild baselines under the M3 engine** for both families: full validation
   stack (5-fold WFO, bootstrap CI, LOO-sym, LOO-fold, MC) on the corrected trade
   sets, plus a proper capital-based equity/MDD simulation.
2. **Re-run the RR sweep under M3** (R071's RR decision is proxy-based).
3. **Exit research under M3 only**: partial TP, trailing stop, BE-after-1R, wider SL,
   entry-at-signal-close — with out-of-sample confirmation.
4. **Update the demo bot** to honest expectations (likely: run Family A only until
   re-validated, or pause Family C), and update `.agents/memory/*`.
5. Optionally test entry-at-signal-close (M5) as a *deployment* improvement — the
   bot's one-bar delay is costing ~0.5 PF on Family A.

---
*Generated by `scripts/exit_model_audit.py` (reproduces M1 and M2 exactly).*
