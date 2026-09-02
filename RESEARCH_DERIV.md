# Research Note — Deriv / Binary Options Exploration (2026-09)

## Why binaries
User wants fast, small-account compounding. Binaries = fixed risk, fixed payout,
high frequency (1m/5m). But profitability REQUIRES a directional win rate > break-even.
With ~90-95% payout, break-even win rate = 1/(1+0.95) ≈ 52%; a real edge needs **>55%**.
Our crypto work was R-multiple based (low win rate, big winners) — does NOT transfer.

## Deriv reachability (sandbox)
- deriv.com / api.deriv.com / ws.binaryws.com: **HTTP 200 reachable** (Binance/Bybit/Yahoo were blocked; Deriv is NOT).
- WS handshake connects; needs `app_id` (public demo `1089` works for market data, no auth).
- ticks_history (style=candles, granularity=seconds) serves historical OHLC. Min candle granularity = 60s (1m); forex min = 900s (15m). True 1s candles not served historically (ticks are live-only).

## Synthetic indices = random walk (no edge)
Backtested R_10/R_25/R_50/R_75/R_100 (5m + R_75 1m), stpRNG (step, 5m).
- RF full-test accuracy ≈ 50% (coin flip). Model finds ZERO high-confidence setups (traded_n=0).
- Volatility/step indices are engineered random walks by design → no predictable direction.
- **Conclusion: binary on synthetics is a losing game (payout <2x × 50% accuracy).**

## Forex 15m = marginal but real signal (the "more hope" case)
10 pairs, 2024-01..2026-09, 15m, forex-suited features (momentum, RSI, MA distance,
session flags, vol). Results:

| Pair | Full acc | win@0.55 | win@0.60 | n@0.55 |
|---|---|---|---|---|
| EURUSD | 50.7% | 53.2% | 0% | 312 |
| GBPUSD | 52.2% | 55.1% | 0% | 254 |
| USDJPY | 49.5% | 51.1% | 0% | 364 |
| AUDUSD | 50.4% | 56.3% | 0% | 71 |
| **USDCHF** | **55.4%** | **61.1%** | 0% | 36 |
| EURGBP | 52.7% | 56.1% | 0% (1) | 239 |
| USDCAD | 48.6% | 55.9% | 0% | 145 |
| EURJPY | 49.5% | 52.1% | 46.8% | 374 |
| GBPJPY | 52.4% | 54.7% | 0% | 391 |
| NZDUSD | 53.3% | 59.5% | 0% | 74 |

- Forex is NOT a random walk: full accuracy clusters 49-55%, with USDCHF at 55.4%.
- Several pairs clear >55% win on filtered trades (USDCHF 61%, NZDUSD 60%, AUDUSD 56%, EURGBP 56%, USDCAD 56%, GBPUSD 55%).
- **But the edge is marginal**: model confidence rarely exceeds 55-60% (no trades fire at 0.60), trade counts are modest, full accuracy barely >50% for most pairs, and USDCHF's strong number rests on only n=36 filtered trades.

## Verdict (no sugar-coating)
- Synthetics: NO binary edge (random walk).
- Forex 15m: faint, unproven edge. Better than synthetics, but NOT yet a robust >55% profit engine. USDCHF is the most promising lead.
- To make binaries viable we'd need: deeper data (multi-year, or tick-level), richer features (order-flow/session-timing/carry), and validation that the >55% holds out-of-sample with enough trade volume. Currently unproven.

## Next steps IF pursuing binaries
1. Pull 2-3yr 15m for the promising pairs (USDCHF, EURGBP, NZDUSD, GBPUSD) + more data per pair.
2. Test richer features (intraday session timing, 4H trend context, volatility-regime).
3. Walk-forward validation of the >55% win rate with adequate trade count.
4. Only after a robust, validated >55% edge → consider adding binary to demo_bot (NOT done yet).
