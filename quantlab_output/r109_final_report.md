# R109 FINAL REPORT — F3: CTA-HORIZON TREND (The FX Hunt Finale)
**Daily close-only Donchian, 8 majors, decade of data · verdict: ❌ REJECTED — and with it, the FX hunt closes**

---

## What was tested

**Hypothesis (who pays):** the documented medium-term trend premium (Moskowitz–Ooi–Pedersen;
Hurst–Ooi–Pedersen's century of evidence): institutions reprice slowly, central banks smooth,
forced flows continue moves. R107 tested 1.5–8 *day* breakouts; the academic premium lives at
1–12 *months*. This round went there: daily bars, N ∈ {20, 60, 120, 252} trading-day channels,
always-in, 2.5×ATR(24d) buckets, 0.5%/pair, 1.2 pip RT + 0.7 pip/day swap.

**Data upgrade:** Yahoo daily `range=10y` → 2,603 bars/pair (2016-10 → 2026-10). Even the
252-day lookback is fully warm before the test window — all variants judged on equal footing.
Gates unchanged: charter 6, selection 2024-01→2026-03, holdout 2026-04→10.

## Results (selection window; R in test-window scope)

| Variant | posW | worstW | maxDD | Yearly R (24/25/26) | Gates |
|---|---|---|---|---|---|
| N=20d | 39.0% | −4.3% | −25.3% | −19 / −27 / −13 | 0/6 |
| N=60d | 39.0% | −5.3% | −34.4% | −48 / −19 / −18 | 0/6 |
| N=120d | 44.1% | −4.5% | −31.0% | −35 / −9 / −12 | 0/6 |
| **N=252d** | 63.6% | −3.5% | −6.8% | **+13 / −3 / −5** | 2/6 |

Best (N=252d) full window: posW 60% ❌ · worst week −13.3% ❌ · maxDD −21.2% ❌ ·
2024 +13.3R ✓ / 2025 −2.6R ❌ / 2026 −39.8R ❌ → **1/6 gates. Holdout −35R.** REJECTED.

**Soft anchor PASSED:** USDJPY net-long Jan–Jul 2024 (+1,927 direction-days) — the system saw
the one genuine trend of the era, and that pair was the only winner (+22R). The test wasn't
broken; the premium was just absent everywhere else.

**Scope note (disclosed):** with 726-day median holds, trades straddling the 2024 boundary make
ledger-level WR/PF stats misleading (a trade's full-life R includes pre-2024 MTM). All gate
metrics come from the daily MTM curve, which is scope-correct and integrity-asserted per pair.

## The decade autopsy (41 trades, 2018→2026, best variant)

| Component | Result |
|---|---|
| Gross edge (before costs) | **−20.9R** (−0.51R/trade) |
| Retail toll (spread + swap) | **−88.0R** (−2.15R/trade; avg hold 594d → 416 swap-pips vs 206-pip bucket) |
| Net | −108.9R |
| Gross by era | 2018–2021 deeply negative; 2022–23 ≈ 0; **2026 +46.4R gross → +9.2R after 37.2R of toll** |

Two killers, both confirmed, one per horizon:
1. **The premium is absent on majors.** A decade of daily data shows no persistent gross trend
   edge on 8 FX majors (academia's results come from ~50-market baskets including commodities
   and indices — disclosed context, not an excuse).
2. **The retail toll scales with horizon.** Short holds die by spread (R107: 0.54R/trade);
   year-long holds die by swap (R109: 2.15R/trade). The sharpest single finding of the round:
   **in 2026 the trend premium finally appeared — and the swap toll confiscated 80% of it.**
   At CTA horizons, retail swap pricing is a structural tax no signal can outrun.

## 🔒 FINAL VERDICT — THE FX HUNT IS CLOSED

| Frontier | Horizon | Verdict |
|---|---|---|
| F010–F013 session/bracket patterns | intraday | ❌ dead (path artifacts, cost wall) |
| R107/F1 Donchian breakout | 1.5–8 days | ❌ dead (gross −211R, 0/6 gates) |
| R108/F2 unhedged carry | weeks | ❌ real premium, untamable tail (worst wk −27.9%) |
| R109/F3 CTA-horizon trend | 1–12 months | ❌ dead (gross −21R/decade; swap toll −2.1R/trade) |

**The law, complete:** *FX offers retail traders no behavioral premium at any horizon, and its
one contractual premium (carry) has a tail no consistency charter can carry unhedged.
Contractual premia survive retail costs only when hedged — which is precisely what the R106
crypto floor (lend + short-perp) is.* The famine across every frontier is what gives the
standing machine its value: **STACK v2 is not merely the best result of the hunt — after this
round it is the only thing standing, exactly where the fundamentals-first method said to look.**

## Artifacts

`quantlab_output/r109_cta_trades.csv` (41 trades, decade) · `r109_cta_weekly.csv` ·
`r109_stdout.log` · `quantlab_cache/forex/{PAIR}_1D.csv` (8 × 2,603 daily bars) ·
fetcher: `scripts/fetch_forex.py` (1H + 1D, cache-aware, retry-hardened).
