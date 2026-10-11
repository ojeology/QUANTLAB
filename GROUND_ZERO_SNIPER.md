# 🎯 GROUND ZERO: THE SNIPER CHARTER
**2026-10-11 · Low-frequency, high-conviction campaign · "three perfect trades per month"**

## Profile (locked)
A sniper is the opposite of the always-on stack: **few trades, each one paid to show up.**
- ≤ ~3 entries/month across the book; ≤ 3 concurrent positions; 1 position per coin.
- Every candidate must clear costs by a WIDE margin (target avg net ≥ +0.4R/trade).
- Risk 1.0% per trade (sniper sizing), hard stop mandatory, no adds, no revenge trades.
- One shot per setup signature: if the setup fails, wait for the next one. No mining.

## Who pays (candidates must answer BEFORE testing)
- **S1 Daily cascade reversal:** liquidated leveraged longs sell at panic prices; the buyer
  who provides liquidity at those extremes is paid the overshoot (forced flows — R104 found
  this premium at 1H; the sniper takes it at the daily scale where retail doesn't compete).
- **S2 Macro-catalyst expansion** (queued): post-FOMC/CPI volatility is bought by hedgers —
  documented event premium.
- **S3 Weekly extreme reversion** (queued): behavioral anchors at the weekly scale.

## S1 — Daily cascade reversal (prespec, locked BEFORE testing)
- **Universe:** the 20 R106 coins, OKX daily candles.
- **Signal (all evaluated at the daily close):**
  1. close < SMA20 − k×ATR14 (k ∈ {2.0, 2.5, 3.0}, selection picks)
  2. drawdown from rolling 20-day high ≥ 12% (real panic, not noise)
- **Entry:** at that signal bar's close (champion E6 semantics — close-only law).
- **Exit (first touch, at closes only):** close ≥ SMA20 (target) · close ≤ entry − 1.0×ATR14
  (stop) · 10 trading days (time stop) — time stop ∈ {7, 10} days, selection picks.
- **Tie-break:** more than 3 signals in a day → take the deepest (largest k-multiple).
- **Costs:** 0.10% round-trip (taker + slippage), charged in R.
- **Gates (the sniper bar):**
  1. Yearly ledger: 2024, 2025, 2026 each ≥ 0R (the survival law, unchanged)
  2. Max drawdown ≤ 12% at 1% risk
  3. Worst week ≥ −3%
  4. Selection 2024-01→2026-03: n ≥ 30 trades AND avg net ≥ +0.40R
  5. Holdout 2026-04→10 (untouched until verdict): ≥ 0R and PF ≥ 1.3
  6. Close-only fills; daily-curve ≡ trade-ledger integrity assert on every pair
- A failed candidate is a finding, not a failure. No gate moves after results exist.

## S1 RESULT (R110, 2026-10-11) — ❌ REJECTED

Daily cascade reversal, 20 coins, 1%/trade, close-only, costs in. Best grid variant (k=2.0, T=7):
avg **+0.07R/trade** (bar: ≥+0.40) · 2026 −8.6R ❌ · maxDD −26.5% ❌ · worstW −8.9% ❌ · holdout PF 1.55 ✗–✓ mixed.
**MEASURED LAW: the cascade overshoot premium ≈ +0.1R gross per event at ANY timeframe (1H: +0.119R/473 trades; 1D: +0.07R/192).
Changing the clock changes frequency, not quality. Snipers must find a different PAYER, not a different chart speed.**

## S2 PRESPEC (locked 2026-10-11, before testing) — MACRO-CATALYST EXPANSION

Who pays: hedgers and option dealers repositioning into scheduled macro news (FOMC/CPI) —
documented event-volatility premium; crypto absorbs the repricing in hours.
Rule (prespec): BTC/ETH daily. Enter at the close of the LAST bar before each scheduled
FOMC decision (14:00 ET Wed) / CPI release (08:30 ET) → direction = breakout trigger:
long if event-day close > event-eve close × (1+0.5×ATR14/c) else short (close-only, no lookahead:
event calendar is public/scheduled — causality trivially safe).
Exit: 48h time stop at close, or 1×ATR14 stop at close. 1% risk, 0.10% RT.
Gates: sniper bar (avg ≥ +0.40R needs the event premium to be LARGE) + yearly ≥ 0R + maxDD ≤ 12%.
n ≈ 8 FOMC + 12 CPI per year ≈ 60 events/3y (2 coins). Selection 2024-01→2026-03; holdout 2026-04→10.
Event calendar source: documented fixed schedule dates (public record), printed in report.

## S2 RESULT (R111, 2026-10-11) — ❌ REJECTED (and the FOMC-only rescue died in holdout)

54 public-calendar events × BTC/ETH, close-only event-day entries, 1%/trade. Prespec: avg +0.275R ❌ (< +0.40),
2026 −2.7R ❌, all-events holdout PF 0.31 ❌. Decomposition: CPI dead (−0.06R) · FOMC +0.30R but DECAYING
(+0.38 → +0.22 → +0.25 by year) → **FOMC-only HOLDOUT: 4 trades, −1.1R, PF 0.61 — dead.**
**THE HOLDOUT EARNED ITS KEEP: the post-hoc subset seduced on selection (+0.37R) and was killed before shipping.**

**SNIPER LAW ACCUMULATED: trade-frequency payers (forced flows, event hedgers, herds, carry) are thin, decaying,
or tail-heavy for retail. S3 weekly extreme reversion = last queued candidate.**

## S3 PRESPEC (locked 2026-10-11, before testing) — WEEKLY EXTREME REVERSION (queue finale)

Who pays: capitulating holders at weekly-scale extremes — documented short-horizon reversal
(Jegadeesh/Lehmann): after an extreme weekly drop, forced sellers have exhausted; weekend
liquidation flushes overshoot thin liquidity. Buyer at the extreme is paid the bounce.
Prior honesty: S1 found the same behavioral family thin at daily scale; this tests the WEEKLY
extreme (deeper pain, rarer event). Falsification value: if THIS is thin too, the behavioral
family is closed at every clock speed.

Rule: at Friday closes — 5-day return ≤ threshold (grid {−12%, −15%}) → buy that close.
Exit at closes: full reclaim (close ≥ pre-crash Friday close, i.e. 5 days earlier) = thesis
realized / stop close ≤ entry − 1.25×ATR14 / 15-day time stop. 1 pos/coin, max 3 concurrent,
deepest-drop tiebreak, 1% risk, 0.10% RT. 20 coins, OKX dailies.
Gates: sniper bar — sel(2024-01→2026-03) n≥20 & avg ≥ +0.40R; yearly ≥0R; maxDD ≤12%;
worstW ≥−3%; holdout(2026-04→10) ≥0R & PF ≥1.3. Integrity assert as always.

## S3 RESULT (R112, 2026-10-11) — ❌ REJECTED — **SNIPER QUEUE CLOSED (0/3)**

Weekly extreme reversion (≤−12%/−15% weeks): avg −0.05/+0.08R vs +0.40 bar; core gates 0/3; holdout PF 1.34 (thin-but-alive, same as S1).

### ══ FINAL LAW OF THE SNIPER HUNT ══
Behavioral reversal pays ~+0.1R/event at EVERY clock (1H +0.119 · 1D +0.07 · 1W 0.00–0.08).
Frequency and conviction are inversely related by construction — rarer events pay the same.
**A +0.4R-conviction retail sniper does not exist in this payer universe. 112 rounds converge:
collect rent (hedged-contractual), don't take shots.** → STACK v2 paper trial is the standing priority.
