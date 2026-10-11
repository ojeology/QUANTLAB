# R111 FINAL REPORT — GROUND ZERO SNIPER, S2: Macro-Catalyst Expansion (FOMC/CPI)
**Verdict: ❌ REJECTED as prespec'd — and the FOMC-only rescue died in the holdout. The process worked.**

---

## What was tested (prespec: GROUND_ZERO_SNIPER.md)

**Who pays:** hedgers and dealers forced to reprice against scheduled macro news (8 FOMC
decisions + 12 CPI prints/year). The event-vol premium: post-news continuation on BTC/ETH.

**Disclosed ambiguity resolution (mechanical, before any run):** the prespec's "enter at eve
close" clause is unexecutable close-only without a direction; the locked direction *formula*
(event-day close vs eve ± 0.5 ATR) defines the operative entry = **event-day close**.

**Rule:** 54 public-calendar events × BTC+ETH (2024-01→2026-09; 2025 shutdown reschedules
disclosed in code). Enter at event-day close if |move| > 0.5×ATR14 (else fizzle — no trade);
exit at 48h close time-stop or 1×ATR close-stop; 1% risk; 0.10% RT.

## Results

| Segment | n | avg R | Sum | Verdict |
|---|---|---|---|---|
| **All events (prespec)** | 48 sel | **+0.275R** | — | ❌ below +0.40 bar |
| CPI only | 19 | −0.059R | −1.1R | dead (fizzles + whisper-numbered) |
| **FOMC only** | 35 | **+0.301R** | +10.5R | the premium lives here… |
| FOMC by year | | +0.38 → +0.22 → +0.25 | | …but it is DECAYING |
| FOMC-only HOLDOUT | 4 | −0.27R avg | −1.1R, PF 0.61 | ❌ dead where it matters |

Prespec gates: avg +0.275 ❌ · 2026 −2.7R ❌ · all-events holdout −3.8R (PF 0.31) ❌. MaxDD ≈ −7.7% ✓.
(One more display slip in the inline printer: maxDD printed ×100; metrics above from the CSVs.)

## The two findings that matter

1. **The event premium is real but FOMC-specific and decaying:** +0.38R/trade in 2024 → ~+0.24R
   in 2025–26 → **negative in the untouched holdout (PF 0.61)**. Crypto's macro reaction is
   being arbitraged away year by year — exactly what maturing institutionalization predicts.
2. **The holdout earned its keep.** The FOMC-only subset looked like the sniper answer on
   selection data (+0.37R, n=31, WR 50%+). Four holdout trades later it was dead. **Without
   the holdout discipline we would have shipped a decaying edge with a straight face.** This is
   the round the charter was built for.

## Sniper ledger

- S1 daily cascade: ❌ thin (+0.07R vs +0.40 bar) — clock changes frequency, not quality
- S2 macro events: ❌ decaying premium, dead on holdout
- S3 weekly extreme reversion: queued (last candidate in the charter queue)
- **Accumulating law:** every trade-frequency payer tested — forced flows, event hedgers,
  breakout herds, carry crowds — is either thin, decaying, or tail-heavy for retail. The
  hedged-contractual family (the R106 floor) remains the only structure that has ever passed
  every gate this project owns.

## Artifacts

`quantlab_output/r111_event_trades.csv` (54 trades) · `r111_event_weekly.csv` · `r111_stdout.log`
· event calendar embedded with sources (federalreserve.gov, BLS; 2025 shutdown reschedules noted).
