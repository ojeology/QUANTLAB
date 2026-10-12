# R116 FINAL REPORT — H4: OPEN-INTEREST / CROWDED-POSITION DYNAMICS
**Verdict (pre-registered mapping): ❌ FAILS AFTER COSTS — squeeze premium real-ish in-sample, dead in holdout; froth-contrarian is backwards**

---

## Data spine (new capability, probed + built this round)

- OKX rubik OI-history: only ~100 days depth → **BLOCKED for 3-year testing** (disclosed).
- **Binance Vision daily futures metrics**: 5-min files daily-aggregated (OI in USD, top-trader
  long/short ratio, account L/S ratio, taker flow ratio) → **24 coins × 1,004 consecutive days
  (2024-01-01 → 2026-09-30), zero gaps** — the first multi-year crowding dataset this project has had.

## Hypotheses (prespec: RESEARCH_FRONTIER_R115.md, H4 section — locked before testing)

- **Family A — squeeze-contrarian:** short coins where top traders are most long-crowded,
  long the most short-crowded (payers: over-leveraged consensus forced out at bad prices).
- **Family B — froth-contrarian:** long deleveraging coins (falling OI), short froth (rising OI).
- Machinery identical to R115: long bottom-Q / short top-Q, 0.5 gross per side, 2× leverage,
  0.07%/side costs, real funding cash flows on legs, close-only, WF monthly by trailing Sharpe,
  untouched holdout 2026-07→09, MB-bootstrap, LOAO, feasibility.

## Results

| Family | Selection APR range | WF-OOS | Holdout |
|---|---|---|---|
| **A squeeze** | +17…+31% (all 8 positive) | **+22.4% APR, Sharpe 1.36**, maxDD −36.7% | **−34.2% APR (−17.3% equity)** ❌ |
| B froth | **−27…−55% (all 8 negative)** | — | — |

- Bootstrap CI95 on full OOS: **[−5.1%, +42.2%]**, P(mean>0) = 93.8% → CI crosses zero.
- LOAO: +10.4% (XLM) … +29.4% (XRP) — all positive, no single-asset dependence.
- Benchmarks: BTC B&H +32.9% total; cash 4%; R106 floor +5.4%; R115 dispersion +35.5% WF.
- Feasibility: identical book floor ($110 gross) → **$25/$50 INFEASIBLE FOR SMALL ACCOUNT**;
  $100/$250 mechanically feasible; ~160 rebalancing trades/mo at selected cadence.

**Mapping applied (as pre-registered):** bootstrap CI ≤ 0 OR holdout < 0 ⇒ **FAILS AFTER COSTS.**

## Findings

1. **The squeeze signal is the third consecutive "real premium, dead holdout"** — in-sample
   edge (all 8 variants positive, LOAO-robust), then −17% equity in its first untouched quarter.
   Whether decay or regime, the prespec'd law is blind: not a survivor.
2. **Family B inverted is a discovery, not a loss:** OI-change CONTRARIAN loses 27–55% APR
   consistently, meaning **rising-OI coins systematically OUTPERFORM falling-OI coins** —
   open-interest flow behaves as momentum, not froth-reversion. Recorded for a future round's
   prespec (testing the reversal NOW would be holdout-informed optimization — not permitted).
3. Crowd data (who is positioned) and flow data (who is paying funding) both contain
   cross-sectional information — **and both failed the same untouched quarter.** July–Sep 2026
   was hostile to every dollar-neutral L/S book tested so far. That quarter is becoming the
   project's most informative regime.

## Audit trail (this round)

1. OKX OI depth block (100 days) caught before design commitment.
2. Vision `metrics` path: monthly 404 → daily layout discovered via status-code sweep.
3. **Kline float-epoch bug:** headered dump files parse `open_time` as float
   (`"1717200000000.0"`) → `int()` crash, silently swallowed by a broad except in the fetcher —
   isolated by unswallowing, fixed robustly, fetcher patched for posterity.
4. Mixed-format price loader (epoch vs ISO) — normalized (R115's norm_date reused).
5. Zero-universe defect (coverage × broken prices) — resolved via component probe, not guessing.

## Standing

- Frontier queue: **H4 closed.** Remaining from the directive: #2 cross-exchange (2-venue only),
  #3 order-flow (no L2 data), #5 regime-conditioned RV (parked). **H4's B-reversal (OI momentum)
  is the only genuinely new, data-backed candidate discovered this round — queued for a future
  prespec.**
- Cores unchanged: R106 floor · frozen champion (recheck end-2026) · Opay naira pot.

## Artifacts

`quantlab_output/r116_crowding_ensemble.csv` · `r116_stdout.log` ·
`scripts/fetch_binance_metrics.py` · data: `quantlab_cache/bnmetrics/` (24 × 1,004 days) ·
`quantlab_cache/bnfut/` (38 syms funding + prices, repaired).
