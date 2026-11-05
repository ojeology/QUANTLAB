# R135b — UNIVERSE EXPANSION: 20 → 73 COINS — PRESPEC (LOCKED 2026-11-05, pre-run)
Question (operator): R135 shows signs of life — why not the whole 73 pool? The 20-coin universe
was inherited from R106's floor spec, not optimized. The lab's cross-section law (R078, R135
forensics: high-beta alts carry the engine) PREDICTS breadth should help. This round tests that
prediction. Expansion is pre-registered here, before any run — post-hoc universe shopping is
dredging and is forbidden.

## Data (locked)
All 73 OKX 1H parquets (2024-01-28 → 2026-11-05, variable starts) aggregated to daily OHLC
(open=first hourly open, high=max, low=min, close=last hourly close; days with <12 hourly bars
dropped). Dailies cached to quantlab_cache/daily73/. DISCLOSURE: the 73 pool is survivorship-biased
(today's listed coins; delisted names absent). Unfixable with this cache; named and carried.

## Mechanization (locked, IDENTICAL to R135 — no parameter may move)
w5 pivot ±5, 120-day line window, two most recent pivots, rising/falling condition, fresh close
cross, frozen-line re-cross exit, 10-day time stop, 0.10% RT costs in R, one open trade per coin.
w3 computed ONCE as report-only sensitivity (it was R135's predeclared variant).

## Gates
1. Pooled PF@c ≥ 1.15 (w5, all 73).
2. Survival law per calendar year with ≥5 active coins: 2024, 2025, 2026 each ≥ 0R.
3. Halves: first-half PF@c ≥ 1.0 AND second-half ≥ 1.0.
4. BREADTH LAW (the point of the round): the 53 coins NOT in the R106-20 must alone clear PF@c ≥ 1.0.
   Expansion ADDS if this passes and pooled PF@c ≥ R135's 1.42 × 0.9; DILUTES if the new-53 fail.
5. Cost stress: pooled PF@c at 0.20% RT (2× costs — tail-coin liquidity) ≥ 1.0.
Verdicts: PASS all → edge scales with breadth (cross-section law confirmed for trendlines);
FAIL breadth gate → edge was a top-20 phenomenon, 20-coin spec stands. One run, no iteration.
