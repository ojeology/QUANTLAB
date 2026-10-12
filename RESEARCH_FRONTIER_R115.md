# 🔭 NEW RESEARCH FRONTIER — R115: FUNDING-RATE DISPERSION
**Directive received 2026-10-12 · Phase 1 audit → Phase 2 selection → Phase 3 prespec (locked BEFORE testing)**

---

## PHASE 1 — COVERAGE AUDIT (evidence-based: file inspection, not log-memory)

| Avenue | Classification | Evidence |
|---|---|---|
| Candle-indicator systems (5m/1H crypto, 25+ variants) | **Already tested adequately** | R005–R095 implementations present (`grep -l` dozens); 5m killed 7× at cost wall; 1H SVM champion validated |
| Donchian / breakout | **Already tested adequately** | r021/r023/r026/r038/r091 (crypto) + R107/R109 (FX, daily CTA-horizon) — all rejected with cause |
| SVM/ML filters | **Already tested adequately** | r086–r100 series (champion = SVM q0.55 walk-forward); forex F001–F007 |
| Standard (cross-sectional) momentum | **Already tested adequately** | r076/r085 (prior implementations found by grep — pre-existing), R113 (charter-grade re-test: real premium, uninhabitable) |
| BTC-ETH spread / pairs | **Already tested adequately** | spread-term implementations present in r056/r064/r070-era scripts (5m-era relative-value probes); superseded by venue-data era |
| CAGE | **NOT FOUND in repository** | `grep -ril CAGE` → zero files. Named in the directive as tested; no implementation or log exists here. Treated as covered by the operator's account; NOT repeated. Flagged for honesty. |
| **Funding-rate dispersion (cross-sectional, dollar-neutral)** | **NOT FOUND — genuinely untested** | `grep -ril "dispersion\|low.funding\|funding.rank"` → zero files. R102 (always-on carry) and R106 (same-coin lend+short hedge) are DIFFERENT structures: neither ranks coins cross-sectionally nor holds long/short legs of different coins. |
| **Order-flow imbalance / liquidity replenishment** | **Blocked by missing data** | `grep` → zero implementations; no tick/order-book data reachable (OKX books are snapshot-only via REST; no historical L2 dump accessible from this sandbox) |
| **Open-interest / crowded-position dynamics** | **Untested; data now partially available** | zero implementations; NEW probe: OKX `rubik/stat/contracts/open-interest-history` period=1D **returns data** (depth TBD) — queued as the follow-on candidate |
| Cross-exchange funding/basis dislocations | **Blocked/incomplete** | only Binance(dumps)+OKX(API) have funding; Bybit/others unreachable → cross-venue execution modeling not honestly testable here |
| Seasonality/calendar | **Already tested (R114): 0/20 under Bonferroni** | mirage confirmed |
| Buy&hold/DCA benchmarks | **Already measured (R114)** | fail the survival law |

## PHASE 2 — PRIORITIZATION OF THE FIVE HYPOTHESES

| # | Hypothesis | Data (probed today) | Small-account feasibility | Novelty | Verdict |
|---|---|---|---|---|---|
| 1 | **Funding-rate dispersion** | ✅ **3 years** via data.binance.vision (38 syms × 3,415 epochs fetched) + aligned 1D prices (35 syms complete) | plausible (perp minimums ~5 USDT/leg) | **never tested** (grep-proven) | **SELECTED** |
| 2 | Cross-exchange funding/basis dislocations | partial (2 venues) | poor (capital spread across venues) | partial (R104 basis probe dead) | queued behind #1 |
| 3 | Order-flow imbalance / liquidity replenishment | ❌ no L2/tick dumps | N/A | untestable here | BLOCKED BY DATA |
| 4 | OI / liquidation / crowding | partial (OKX rubik daily OI works; liquidation endpoint 404) | medium | untested | **queued next** (new source discovered today) |
| 5 | Regime-conditioned relative value | buildable on existing caches | medium | overlaps tested families | parked |

## PHASE 3 — PRESPEC (LOCKED 2026-10-12, BEFORE ANY RESULT)

**Hypothesis (who pays):** traders paying extreme funding to hold crowded positions are
cross-sectionally rankable; the portfolio that shorts the most-crowded side and longs the
least-crowded side collects the funding SPREAD as (near-)contractual cash flow, bearing
relative-price risk (NOT market-neutral — different coins on each leg; disclosed).

**Data spine:** Binance USDT-perp funding (all epochs, 2023-10-01→2026-09-30) + aligned daily
closes, same venue. Symbols kept only with ≥97% coverage on BOTH series across the full spine.
**Causality law:** weights at close of day t from funding epochs ≤ t; all P&L accrues from t+1.
**Grid (pre-registered, 18 combos):** signal window W ∈ {1, 3, 7} days of trailing mean daily
funding × quartile size Q ∈ {5, 8} per side × rebalance R ∈ {1, 3, 7} days.
**Construction:** long bottom-Q funding, short top-Q funding, 0.5 gross per side (1.0 total,
2× account leverage → 50% initial margin). **Costs:** 0.07% per side notional on every weight
change (0.05% taker + 0.02% slippage); funding cash flows from actual epochs; no coin-borrow
fee (perp margin shorting); maintenance-margin requirement 1% of gross.
**Walk-forward:** monthly, 2024-07→2026-06 — each month trades the grid combo with the best
trailing-180-day net Sharpe (prior data only). **HOLDOUT (untouched): 2026-07-01→2026-09-30.**
**Multiple testing:** Bonferroni across 18 grid points; significance = block-bootstrap
(20-day blocks, 2,000 resamples) 95% CI of WF-OOS annualized mean excluding 0 AND holdout ≥ 0.
**LOAO:** drop each symbol (re-rank without it), report WF-OOS APR range.
**Benchmarks:** cash 4% APR · static short-all funding harvest (price-exposed) · BTC B&H ·
R106 floor reference (+5.4% APR line).
**Phase 4 (pre-registered):** per-leg venue minimum notional (documented public specs: most
alt-perps 5 USDT, BTC/ETH 20 USDT; OKX cross-check via instruments API) → minimum viable
capital at 2× leverage; evaluate $25/$50/$100/$250: forced leverage, utilization, trade
frequency, P(ruin) = bootstrap P(min equity < maintenance buffer). Infeasible ⇒ labeled
INFEASIBLE FOR SMALL ACCOUNT regardless of statistics.

**Pre-registered verdict mapping (Phase 5):**
- SURVIVES INITIAL VALIDATION = bootstrap CI > 0 ∧ holdout ≥ 0 ∧ LOAO-min APR > 0 ∧ maxDD ≤ 20%
- PROMISING BUT INCONCLUSIVE = positive but any of the above unconfirmed
- FAILS AFTER COSTS = WF-OOS net APR ≤ 0 or bootstrap CI ≤ 0
- BLOCKED BY DATA = spine unusable; PROFITABLE BUT INFEASIBLE = survives + capital floor > $250

## PHASE 5 — RESULT (2026-10-12, executed per prespec)

Grid (selection): W=7 dominates — +21…+59% APR, Sharpe 1.3–2.8, net of costs.
WF-OOS 2024-07→2026-06: **+35.5% APR / Sharpe 1.94 / maxDD −21.7%** (2× lev).
HOLDOUT 2026-07→09 (untouched): **−33.6% APR / −16.9% equity** ❌.
Bootstrap CI95 [+2.9%, +61.8%], P(mean>0)=98.6%. LOAO +25.2%…+44.2% (all positive).
Static short-all −1.1% ⇒ edge is genuinely cross-sectional (funding LEVEL ≈ 0; the SPREAD pays).

### VERDICT: 🟡 PROMISING BUT INCONCLUSIVE (+ $25/$50 INFEASIBLE FOR SMALL ACCOUNT; $100/$250 feasible)
Real, robust, significant — and dead in its first untouched holdout. No holdout optimization performed.
5 audit defects caught (kline formats, funding headers, R=1 schedule law, equity ×/÷ leverage, EOS coverage-defeat).
Queued next: H4 open-interest/crowding (daily-OI source discovered this round).
