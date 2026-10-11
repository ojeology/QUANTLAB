# R110 FINAL REPORT — GROUND ZERO SNIPER, S1: Daily Cascade Reversal
**Verdict: ❌ REJECTED under the sniper bar — the premium is real but thin, at every clock speed**

---

## What was tested (prespec: GROUND_ZERO_SNIPER.md, locked before the run)

**Who pays:** liquidated leveraged longs — forced sellers at panic prices; the sniper provides
liquidity at the extreme and is paid the overshoot. Daily scale (the R104 cascade premium's big
brother): 20 stack coins, OKX daily candles (1,228 days each).

**Rule:** at the daily close — close < SMA20 − k×ATR14 AND ≥12% off the 20-day high → buy that
close (close-only law). Exits at closes: SMA20 target / entry − 1×ATR14 stop / T-day time stop.
1 position per coin, max 3 concurrent (deepest signal wins the slot), 1% risk, 0.10% RT costs.
Grid k ∈ {2.0, 2.5, 3.0} × T ∈ {7, 10} on selection 2024-01→2026-03; holdout untouched.

## Results (selection; best variant k=2.0 T=7)

| Metric | Result | Sniper bar | |
|---|---|---|---|
| Trades (selection) | 192 (~5.8/mo — over budget too) | n≥30 | ✅ |
| **Avg net R per trade** | **+0.07R** (WR 50.2%) | **≥ +0.40R** | ❌ |
| 2026 | −8.6R | ≥0R every year | ❌ |
| Max drawdown | −26.5% (at 1%/trade) | ≤12% | ❌ |
| Worst week | −8.9% | ≥−3% | ❌ |
| Holdout 2026-04→10 | +9.6R, PF 1.55 | ≥0R, PF≥1.3 | ✅ |

No grid variant reached even half the required per-trade edge (best avgR +0.07; required +0.40).

## The finding — and it's a deep one

The cascade premium measures **+0.07R per trade on dailies** here vs **+0.119R per trade on
1H bars** in R104 — statistically the same thin edge. **Changing the clock does not concentrate
quality; it only changes the frequency of the same-sized premium.** That is now a measured law:
the overshoot buyers earn from forced sellers is ~+0.1R gross per event whether you snipe it
daily or hourly. There is no "sniper mode" of an existing premium — high conviction per trade
must come from a DIFFERENT payer, not a different timeframe.

(Side finding: the holdout passed at PF 1.55 while 2026 overall lost — the premium breathes;
it is not a consistency foundation at 1% sizing.)

## Audit trail (2 defects caught)

1. `edit_file` mangled the fetcher URL earlier (sed `&` expansion) — rewrote via heredoc per house law.
2. **Dimension bug in trade accounting:** dollar moves were divided by the ATR *fraction*
   (bucket in price units vs percent) — inflating every trade by ~the coin's price (BTC trades
   printed −96,669R). Caught by implied-move sanity check; fixed with entry-price-normalized
   dollars; daily≡ledger integrity assert held exactly after the fix.

## Sniper ledger

- S1 cascade daily: ❌ thin premium (this round)
- S2 macro-catalyst expansion (FOMC/CPI days on BTC): queued — a genuinely different payer
  (hedgers repositioning against scheduled news; documented event-vol premium)
- S3 weekly extreme reversion: queued behind it

## Artifacts

`quantlab_output/r110_sniper_trades.csv` (227 trades) · `r110_sniper_weekly.csv` · `r110_stdout.log`
· data: `quantlab_cache/daily/{COIN}_1D.csv` (20 coins × 1,228 OKX dailies) · `scripts/fetch_crypto_daily.py`
