# R112 FINAL REPORT — GROUND ZERO SNIPER, S3: Weekly Extreme Reversion (Queue Finale)
**Verdict: ❌ REJECTED — and with it, the sniper queue closes 0-for-3 with a complete law**

---

## What was tested (prespec: GROUND_ZERO_SNIPER.md)

**Who pays:** capitulating holders at weekly extremes — the documented short-horizon reversal
premium (Jegadeesh/Lehmann): after a brutal week, forced sellers are exhausted; weekend
liquidation flushes overshoot thin liquidity. The buyer at the extreme is paid the bounce.

**Rule:** Friday close with 5-day return ≤ −12%/−15% → buy that close. Exit: full reclaim of
the pre-crash close (thesis realized) / −1.25×ATR14 stop / 15-day time stop. 20 coins, 1%/trade,
max 3 concurrent (deepest pain wins), 0.10% RT, close-only, integrity-asserted.

## Results

| Variant | sel n | avg R (bar +0.40) | worstW | maxDD | Yearly R | Core gates |
|---|---|---|---|---|---|---|
| ≤ −12%/wk | 174 | **−0.05** | −12.3% | −28.8% | +14.7 / −10.1 / −15.0 | 0/3 |
| ≤ −15%/wk | 122 | **+0.08** | −8.4% | −19.7% | +21.4 / −3.4 / −11.0 | 0/3 |

Holdout (−12%): +7.6R, PF 1.34 — breathing again, exactly like S1's holdout: the premium
exists, is thin, and is nowhere near the conviction bar. 205 trades, WR 44%, avg **0.00R**.

## 🔒 THE SNIPER CAMPAIGN — CLOSED (3 candidates, 3 rejections, one complete law)

| Candidate | Payer | Result |
|---|---|---|
| S1 daily cascade | forced sellers (liquidations) | ❌ +0.07R/trade — thin |
| S2 FOMC/CPI events | event hedgers | ❌ +0.28R, decaying, dead in holdout |
| S3 weekly extremes | capitulating holders | ❌ 0.00R/trade — thin at −15% weeks |

**THE COMPLETED LAW OF THE SNIPER HUNT:** the behavioral reversal family pays retail ~+0.1R
per event at EVERY clock speed — 1H: +0.119R (n=473) · 1D: +0.07R (n=227) · 1W extremes:
+0.00…+0.08R (n=205). **Frequency and conviction are inversely related by construction: rarer,
deeper events pay the same per event — you just take fewer of them.** A "ground zero sniper"
with +0.4R conviction does not exist in this universe of payers. High per-trade conviction
at low frequency can only come from a *structural* payer — and the only structural payer
retail can actually collect from remains the hedged-contractual family (the R106 floor:
both crowds pay by contract, the price cancels).

That is 112 rounds of evidence converging on one sentence: **collect rent, don't take shots.**

## Where this leaves the project

- **Closed:** forex hunt (4 frontiers), sniper hunt (3 candidates) — all rejected with cause.
- **Alive and waiting:** STACK v2 12-week paper trial (≥70% green weeks, worst ≥−1.5% → size up);
  frozen champion recheck end-2026.
- **Open frontier (not promised):** option-based carry-tail hedging; a future S-candidate would
  need a payer that is *contractual*, not behavioral.

## Artifacts

`quantlab_output/r112_s3_trades.csv` (205) · `r112_s3_weekly.csv` · `r112_stdout.log`
· data refetched this round: `quantlab_cache/daily/` (4th snapshot wipe, 4th 20-second recovery).
