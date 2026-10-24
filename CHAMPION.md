# 🏆 THE CHAMPION — Complete Specification
### Family A "compression-pop" + SVM filter · 1H OKX perpetuals · validated R073→R098

**One sentence:** when a crypto market coils tight around its long-term average and then
breaks out with participation, buy the pop with a tight 1×ATR stop — and let a
walk-forward ML filter discard the pops most likely to fail.

---

## 1. THE SIGNAL (Family A — compression-then-pop)

All thresholds are **percentile ranks over the trailing 500 bars, recalibrated every 168
bars** (never fixed numbers — the strategy adapts to each symbol's own volatility regime).

| Condition | Meaning | The story |
|---|---|---|
| **BBW_STRICT** | Bollinger Band width in the bottom 25% of its 500-bar range | Volatility has squeezed to historic lows — a coil |
| **RV_LO** | 20-bar realized volatility in the bottom 33% | The market is genuinely quiet, not just narrow |
| **DST_NR** | Distance from EMA200 in the bottom 33% | Price is coiled tight around the long-term average, not trending far from it |
| **PRG_VH** | Previous bar's range in the top 20% | A big-range bar just fired — the spring is releasing |

**Entry gate (all must hold on the signal bar):**
- Green candle: `close > open` and `close > previous close` — direction confirmation
- `rel_vol > 1.5` — volume 1.5× its 20-bar average = real participation, not drift

**Why this should work (the economics):** volatility clusters — quiet periods are
followed by expansion. A tight coil around the EMA200 means positioning is compressed;
a big-range green bar with heavy volume is the release. You're buying the *start* of an
expansion wave, with a stop close enough that the risk is tiny relative to the move.

## 2. EXECUTION MECHANICS

- **Universe:** 73 OKX USDT perpetual swaps (52 original + 18 + 3, validated in R078/R084)
- **Timeframe:** 1H candles, UTC
- **Entry:** at the close of the signal bar (convention "E6" — no next-bar lag)
- **Stop loss:** entry − 1.0 × ATR(14)
- **Take profit:** entry + 1.5 × ATR(14) (risk-reward 1.5:1)
- **Exit check:** intrabar from the first full bar after entry; if one bar touches both,
  TP is assumed (validated as 98% correct against 1-minute data, R098)
- **No time stop** — winners run until TP or SL (avg hold ≈ 3 hours)
- **One position per symbol**; all symbols share the 1% risk budget per trade

## 3. THE FILTER STACK (three layers on top of the signal)

1. **Breadth50:** only trade when **more than 50% of all 73 symbols are above their
   EMA20** (percentile-gated). Logic: compression-pops pay best when the whole market is
   participating, not just one coin chopping.
2. **VolCeil:** skip entries when the symbol's ATR percentile rank is already high —
   don't buy pops when volatility has already exploded.
3. **ML filter (the champion's weapon): walk-forward SVM**
   - Model: Support Vector Machine (RBF kernel), `probability=True`, C=1.0
   - 14 features: atr_rank, adx14, rsi14, ema_dist_pct, prev_body_r, prev_range_r,
     rel_vol, bb_width, real_vol_20, hour, dow, breadth_q, dist_hi48, green_streak
   - Walk-forward: every prediction is out-of-sample. The model is re-fit on ALL
     prior trades for each new signal; the keep-threshold is calibrated on
     **selection-period data only** (< Jan 2026)
   - **Keep rate q = 0.55:** the model scores every signal P(win); only signals in the
     top 55% of scores are traded. This kills ~45% of trades — and most of the losers.

## 4. THE NUMBERS (all at 0.05%/side taker costs unless noted)

| Metric | Value | Source |
|---|---|---|
| Trades | ~15–20 / month | R096/R097 |
| Win rate (true, 1m-verified) | ~51% at RR 1.5 | R098 |
| Expectancy | ≈ +0.23R net / trade | derived |
| PF@cost, 2026 holdout | **1.30** (n=165) | R096 |
| PF@cost, FRESH Aug 8–Oct 9 2026 | **1.99** (n=31, 98% boot P(PF>1)) | R096 |
| PF@cost, 1m ground truth | **1.27 taker / 1.39 maker** | R098 |
| Profitable months | 59–71% (softening — watch it) | R096 |
| Worst losing streak | 2 months | R084/R087 |
| Max drawdown @1% risk | ≈ −15…−21% | R096/R097 |
| Monte Carlo @1% risk | 96% P(profit), median max-DD −10% | R097 |

**Decay watch:** q0.75's holdout PF@cost went 1.62 (Aug) → 1.14 (Oct). The edge is
mature, not dying — but it is not getting stronger. Respect the kill-switch.

## 5. EVERY VALIDATION IT SURVIVED

| Gate | What could have killed it | Result |
|---|---|---|
| R073–R095 research | 70+ competing ideas, 5m/crypto-15m variants | Only this family+ML stack lived |
| 2026 holdout (untouched Jan–Jul) | Overfitting to selection | PF@cost 1.42–1.30 |
| R096 fresh window (Aug 8–Oct 9) | Regime change since selection | PF@cost 1.99 |
| R097 upgrade grid | Better configs existed elsewhere | Fees only; caps/sizing/refit all rejected |
| R098 1-minute path audit | TP-before-SL illusion (killed all 5 forex strategies) | 2% flips; PF 1.27–1.39 confirmed |

No other configuration in the repo's 100+ research runs has passed all five.

## 6. OPERATING MANUAL (2027)

- **Costs:** ≤0.06% per side all-in. VIP taker 0.04% = best (PF +0.05). Standard
  account: maker limit entries (fill ≈95%) + maker TP + taker stop ≈ PF 1.27.
- **Position sizing:** 1% of equity per trade (risk = equity × 1% if SL hits). No more
  until 3+ live months inside the kill-switch limits.
- **Leverage:** 3–5× **isolated** margin per position. The 1×ATR stop (~1% away) always
  fires long before liquidation (~20% away at 5×). Leverage sets margin efficiency;
  the stop sets risk. Never cross-margin the whole account.
- **Funding:** holds average ~3 hours — funding is immaterial but log it live.
- **Kill-switch (pre-committed, no discretion):** rolling 3-month PF@cost < 1.0, OR two
  consecutive losing months → halt, re-run full validation before resuming.
- **Monthly ritual:** `python3 scripts/refresh_okx.py` → `python3 quantlab_r096_recheck.py`
  (~5 min). Fresh data only; never re-tune thresholds on recent results.
- **Scaling:** only by compounding equity, or after per-symbol validation of new symbols
  (R078: the edge is universe-specific).

## 7. WHAT THIS STRATEGY IS NOT

- Not high-frequency: ~0.7 trades/day across 73 symbols.
- Not a forex strategy: all five 2026 forex candidates failed every gate (F009–F013).
- Not risk-free: expect losing weeks, 2-month drawdowns, and eventual regime change.
- Not immune to costs: at 0.10%/side it dies. Cost discipline IS part of the strategy.

## 8. THE HONEST LIMITS OF THE EVIDENCE

- The fresh window is 31–48 trades — encouraging, not conclusive.
- The 1m path audit covers 2026 trades; earlier trades inherit the 1H assumption
  (selection-period only, so no decision ever depended on them).
- Funding rates, slippage beyond spread, and partial fills are live-trading realities
  the backtest can't see. Expect live results slightly below backtest.
- All edge decays eventually. The kill-switch exists because someday this one will too.

---
*Validation chain: R077 (locked pre-ML) → R084/R086/R087 (ML champion) → R096 (fresh
data) → R097 (upgrades) → R098 (path truth). Code: `quantlab_r096_recheck.py`,
`quantlab_r097.py`, `quantlab_r098.py`, `scripts/ql_engine.py`, `scripts/refresh_okx.py`.*

## UPDATE R100 (2026-10-09): sizing + kill-switch refinement

- **Recommended sizing is now the DD throttle:** 1.00%/trade normally → 0.50% when
  equity >5% below peak → 0.25% when >10% below → restore at new high.
  17-month sim: +127.8% vs +140.6% flat, but MDD −12.0% vs −20.9%, Calmar 7.11 vs 4.46.
- **Kill-switch (v2, pre-registered):** halt if rolling 6-month PF@cost < 1.0 OR two
  consecutive months worse than −8R; re-enter after a positive month + revalidation.
  (Rationale: the 3-month rule would have whipsawed us out of the Sep-2026 +14R month.)
- **Robustness:** 9/9 RR×q neighbors profitable (1.19–1.52) — plateau, not a spike.
- **Pre-registered candidate:** RR 2.0 (led every cell). Promotion decision at the
  November re-check on fresh data only.

## YEARLY LEDGER (added 2026-10-09, from refreshed cache)

| Year | Trades | PF@cost | Sum R | Note |
|---|---|---|---|---|
| 2024 | 105 (raw only — SVM bootstrapping) | **0.80** | **−14.0R** | losing regime; partial universe early in year |
| 2025 (Jun–Dec) | 83 | **4.12** | **+67.4R** | bumper volatility regime |
| 2026 (Jan–Oct) | 151 | **1.27** | **+22.4R** | profitable, decaying toward the gate |

Regime truth: harvester of volatility expansion. 2024-type markets = losing year is
possible → kill-switch v2 + throttle are the structural defense.


## UPDATE 2026-10-24 (early-November recheck): KILL-SWITCH v2 TRIPPED → HALT 🛑

- Rolling 6-month PF@cost = **0.91** (< 1.0 gate) on fresh data through 2026-10-23 → the
  pre-committed v2 halt rule fired. **No live capital.** Re-entry: positive paper month +
  full revalidation (see `quantlab_output/champion_recheck_2026-10-24.md`).
- 2026 YTD: +21.2R, PF@c 1.20 (n=186) — the year is still net-positive; the *rolling* edge
  is what died (May/Jun hole + soft Oct).
- New audit item: cross-environment SVM drift re-rolls borderline keeps (±10R book variance);
  pipeline verified causal; pin library versions before the next ledger claim.
- Pre-registered RR 2.0: NOT promoted (fresh-data gate void at n=6); stays the challenger for
  the re-entry revalidation (in-book paired bootstrap 0.98 in its favor, 2026 +37.4R vs +17.4R).
- The table above (added 2026-10-09) is preserved but SUPERSEDED by the recheck report for
  decision purposes: its 2026 row (n=151/+22.4R/1.27) does not reconcile with its own CSV
  (n=165/+27.0R/1.30) — pre-existing discrepancy, documented in the recheck report.
