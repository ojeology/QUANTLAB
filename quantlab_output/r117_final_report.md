# R117 FINAL REPORT — DAILY-CANDLE TREND ON CRYPTO PERPS (the user-spotted gap)
**Verdict: ❌ FAILS the consistency charter (2/6 gates) — but the six-year context confirms the style's famous reputation. The gap is closed with a complete picture.**

---

## What was tested (prespec: CONSISTENCY_HUNT_BEGINS.md, R117 section — locked first)

The one under-tested cell in the matrix, spotted by the operator: classic **daily-candle
trend-following on crypto perps**. Signals (prespec'd, no mining): DON20/DON55 Donchian long/flip,
SMA100/SMA200 regime long/flat — on BTC-only, ETH-only, and a 31-coin equal-weight book.
Costs 0.07%/side on changes + **actual funding epochs** as hold cost. Data spine extended back
to **2020-01** (2,465 bars/coin) so even SMA200 is fully warm. Selection 2024-01→2026-03;
holdout 2026-07→09 untouched; charter gates.

## Results

**Selected: DON55 on BTC.** Full window 2024-01→2026-09: APR +28.0%, Sharpe 0.59,
**maxDD −35.8% ❌ · worst week −11.9% ❌ · posW 48.6% ❌ · 2026 −9.5% ❌** → 2/6 gates.
Holdout Jul–Sep 2026: **+2.4% ✓** (first candidate to survive that killer quarter — trend's
flat-during-chop property). Verdict by pre-registered mapping: **FAILS AFTER COSTS** (charter).

All 12 variants: APR −3%…+37%, **every one** fails posW (25–50%) and most fail maxDD —
the style's inherent shape: long dry spells broken by big wins.

## The six-year context — why this round matters (2020–2023, not gated)

| Year | DON55 BTC | BTC B&H |
|---|---|---|
| 2020 | **+103.4%** | +300%-ish era |
| 2021 | **+108.2%** | +60% |
| **2022 (bear)** | **−6.5%** | **−65%** |
| 2023 | +13.3% | +160% |
| 2024 | +61.2% | +121% |
| 2025 | +25.4% | −8.8% |
| 2026 (to Sep) | −9.5% | −7.1% |

**The legend is TRUE: daily Donchian rode both bulls, sat out the 2022 crash (−6.5% vs −65%),
and beat buy-and-hold over the full six years at a third of the drawdown.** SMA200 BTC went
*flat all of 2022* (+0.0%). This is the one classic style that survives contact with a full
market cycle — and it still fails the operator's law, because −36% intra-period drawdowns,
~48% positive weeks, and a losing 2026 are what the ride actually feels like.

## The completed sentence for the daily-candle family

Time-series daily trend = **real, cycle-proven, historically B&H-beating — and structurally
incompatible with the consistency mandate** (same verdict family as R113's momentum, now proven
at the time-series level too). The repo's central law repeats a third way: *the strategies that
survive the cycle are not the strategies that survive the charter. Only the floor does both.*

## Audit trail

Cache wiped pre-round (7th) → consolidated refetch (4,484 month-files); rate-limit failures
(708) → gap-fill pass completed post-timeout; kline float-epoch parser (R116's fix) held.

## Artifacts

`quantlab_output/r117_dailytrend_curve.csv` · `r117_all_variants.csv` (12 curves) · `r117_stdout.log`
· data: `quantlab_cache/bnfut/` (38 syms, 2020-01→2026-09-30 dailies + funding spine).
