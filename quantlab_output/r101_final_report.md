# R101 — SMALL-ACCOUNT PROTOCOL: the champion for $200 / $500 / $1,000

**Date:** 2026-10-09 · Premise corrected first: the champion never *needed* big capital —
percentages are size-independent. The real small-account problems are exchange minimums,
fee drag, and diversification limits. This run solves all three **without touching the
validated signal** (no inversion — the raw signal is positive, so a reversed strategy is
a guaranteed loser; no scalp-RR — R081 proved that's a cost mirage).

---

## 1) The min-order surprise that changes everything

OKX split its contract sizes (2025), so today **even BTC's smallest contract is ~$8
notional** (BTC $8.3, ZEC $12.2, MU $10.4 — and most alts under $1).

**Result: every account size ≥ $200 can trade every champion symbol.**
At 2% risk with a ~1% stop, a $200 account needs ~$400 positions — carried comfortably
by isolated 3–5× margin, with the stop (not the margin) capping every loss at $4.

## 2) Small-set performance (cheap-minimum, active symbols: HBAR, INJ, PUMP)

| Window | n | PF@cost | WR | Sum R |
|---|---|---|---|---|
| 2026 YTD | 16 | **2.18** | 62% | +7.6R |
| 17 months | 24 | 3.19 | 71% | +16.5R |
| *(all-73 reference, 2026)* | *151* | *1.27* | *51%* | *+22.4R* |

⚠️ **Honest caveat: 16–24 trades is too small to certify the subset separately.** What
makes it legitimate anyway: every one of these trades is *already* a fully-validated
champion trade (same signals, same SVM, same windows) — subsetting changes which
validated trades you take, not their quality. Expect regression toward the all-73 mean
(PF ~1.27), not a permanent 2.18.

## 3) The $ ladder — 2026 YTD, small-set, real dollars

| Sizing | $200 | $500 | $1,000 | Max DD |
|---|---|---|---|---|
| 1% per trade | +$16 | +$39 | +$78 | −3.2% |
| **2% + throttle** | **+$25** | **+$64** | **+$127** | **−6.4%** |
| **3% + throttle** | **+$37** | **+$94** | **+$187** | **−8.0%** |

Monte Carlo (3,000 resamples of 2026): at 2–3% + throttle, **P(drawdown >30%) ≈ 0%,
P(>50%) = 0%, worst-5% case DD ≤ 15%.** (Caveat: resampled from 16 trades — regime risk
is understated; the throttle exists for exactly that.)

## 4) THE SMALL-ACCOUNT PROTOCOL (final)

- **Capital:** $200+ works. Comfortable from $300.
- **Symbols:** the cheap-minimum active set (HBAR, INJ, PUMP today; keep the full
  candidate list from `r101_stdout.log`, require ≥8 champion trades to qualify).
- **Sizing:** start **2% per trade with the R100 throttle** (halve at −5% DD, quarter at
  −10%). Graduate to 3% only after 2 months live within kill-switch limits.
- **Margin:** isolated 3–5×. Loss per trade = your risk $ — always. Leverage never
  changes that here.
- **Expectation-setting (truth in advertising):** $200 at 3% ≈ **+$37 in 9.5 months**.
  Small accounts earn small dollars — the *percentage* is the same game, and the point
  of a small account is proving you (and the system) can run it, then compounding.
- **Unchanged:** kill-switch v2 (6-mo PF@cost <1.0 or two <−8R months → halt), monthly
  `refresh_okx.py` + `quantlab_r096_recheck.py`, November RR2.0 promotion test.

## 5) What we refused to do (and why)

- **"Reverse the champion"** — inverting a positive-expectancy signal yields negative
  expectancy after costs. Guaranteed loser. (F010/F011 already tested fade variants.)
- **RR<1 scalping for more action** — R081: 80% win-rate scalps die at realistic fees.
- **New filters mined for small accounts** — curve-fitting with no fresh data to verify.

*Files: `quantlab_r101.py` · `r101_stdout.log` · `r101_trades_2026.csv`*
