# R115 FINAL REPORT — NEW FRONTIER: FUNDING-RATE DISPERSION
**Verdict (Phase 5, pre-registered mapping): 🟡 PROMISING BUT INCONCLUSIVE — with $25/$50 labeled INFEASIBLE FOR SMALL ACCOUNT**

---

## Phase 1 — Coverage audit (file-inspected evidence, full table in RESEARCH_FRONTIER_R115.md)

Tested adequately: candle-indicator systems (25+ impls), Donchian (5 crypto + 2 FX impls),
SVM/ML filters (18 files), cross-sectional momentum (3 impls incl. R113), BTC-ETH-era spreads,
seasonality (R114). **CAGE: zero files in repo — flagged, not repeated.** Genuinely untested:
**funding dispersion (grep-proven zero implementations)**, OI/crowding (and a NEW daily-OI data
source was discovered today — queued). Blocked: order-flow/L2 (no tick data), cross-exchange
execution (2 venues only).

## Phase 3 — The experiment (full protocol, executed)

**Data spine (new capability):** `data.binance.vision` dumps probed reachable → **3 years of
actual funding epochs** (38 syms × 3,415 events, 2023-10→2026-09) + aligned daily closes
(30 syms pass 97% pre-fill coverage). Selection/spine code handled mixed file formats,
embedded header rows, and gap-fills.

**Construction (prespec'd):** long bottom-Q funding coins / short top-Q funding coins, 0.5
gross per side, 2× account leverage; signal = trailing mean daily funding (W days); weights at
close t, P&L from t+1; costs 0.07%/side on turnover; funding cash flows from real epochs.
Grid 18 combos (W∈{1,3,7} × Q∈{5,8} × R∈{1,3,7}), Bonferroni-aware; monthly walk-forward
selection by trailing-180d Sharpe; untouched holdout 2026-07→09; MB-block bootstrap (2,000);
leave-one-asset-out; benchmarks.

**Results:**

| Segment | APR (2× lev) | Sharpe | maxDD |
|---|---|---|---|
| Best grid on selection (W7/Q5/R1) | +59.4% | 2.84 | −20.9% |
| W7 family, all 6 variants | +21…+59% | 1.3–2.8 | −21…−43% |
| **WF-OOS ensemble 2024-07→2026-06** | **+35.5%** | **1.94** | **−21.7%** |
| **HOLDOUT 2026-07→09 (untouched)** | **−33.6%** (−16.9% equity/3mo) | — | −21.3% |
| Bootstrap ann-mean CI95 (full OOS) | [+2.9%, +61.8%], P(mean>0)=98.6% | | |
| LOAO (drop each of 30 assets) | +25.2% … +44.2% — all positive | | |

**Benchmarks:** static short-all funding harvest **−1.1% APR** → proves the edge is genuinely
*cross-sectional* (the funding LEVEL is ~zero; the SPREAD between extremes is the premium —
and is NOT captured by R102-style always-on carry). BTC B&H +32.9% total (same window, −53%-style
risk). Cash 4%. R106 floor +5.4%.

## Why INCONCLUSIVE and not SURVIVES

The premium is real, large, robust to asset removal (LOAO), and statistically significant
across 24 walk-forward months — then **it failed its first untouched holdout** (−17% equity in
3 months). Three disclosed readings: (a) the signal decays as crowds arbitrage it (same decay
shape as the FOMC premium in S2); (b) the holdout loss was driven by the *relative-price legs*
(funding spread alone cannot burn −33% APR pace) — i.e., the "neutral" portfolio's true risk
showed up exactly when it mattered; (c) a 3-month holdout is a small sample. The prespec'd
rule is blind to which: holdout < 0 ⇒ cannot be declared a survivor. **No optimization against
the holdout was performed or permitted.**

## Phase 4 — Small-account feasibility

Venue minimum notional (documented public specs, Binance USDT-M: 5 USDT most alts / 20 BTC-ETH;
OKX cross-noted as coarser via contract specs):

| Capital | Status | Notes |
|---|---|---|
| $25 / $50 | **INFEASIBLE FOR SMALL ACCOUNT** | min-notional floor $110 gross → forced 2.2–4.4× leverage, sub-minimum orders round up |
| $100 | FEASIBLE (1.1× gross) | ~300 rebalancing trades/mo at the selected R=1 cadence; fee-drag real |
| $250 | FEASIBLE (0.4× gross) | 30% buffer; bootstrap P(equity<60%) = 1.1% |

Even where feasible: maxDD −22% at 2× leverage **fails the house consistency law** (≤12%);
this is a fund-style premium, not a $20–30/month machine.

## Audit trail (defects caught this round — the protocol's cost and its proof)

1. Kline dumps: mixed epoch/string date formats + silent download gaps → gap-fill pass, normalized loader.
2. Funding CSVs: embedded header rows mid-file → numeric-coerce loader.
3. **R=1 rebalances silently never fired** (`i % 1 == 1`) → schedule law fixed, rebalance counts now printed as proof.
4. **Equity scaling divided by leverage instead of multiplying** → drawdowns were ~4× understated; fixed.
5. **EOS delisted mid-spine yet passed coverage via fill-forward** → coverage now judged pre-fill; dead leg removed; results re-run.

## Standing

- Queued: **H4 open-interest/crowding** (new OKX daily-OI source discovered today — depth audit next).
- Unchanged cores: R106 floor (dollar pot), frozen champion (recheck end-2026), Opay naira pot.
- Frontier verdict: funding dispersion joins the ledger as **real but unconfirmed out-of-sample** —
  the first candidate in the project's history to pass every robustness check and still lose its holdout.

## Artifacts

`quantlab_output/r115_dispersion_ensemble.csv` · `r115_grid_selection.csv` · `r115_stdout.log` ·
`RESEARCH_FRONTIER_R115.md` (audit + prespec) · data: `quantlab_cache/bnfut/` (38 funding × 37
months + 38 price files) · `scripts/fetch_binance_history.py`.
