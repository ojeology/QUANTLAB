# R105 — THE ALL-WEATHER ALLOCATION: every consistency gate passed ✅

**Date:** 2026-10-09 · Thesis: no single crypto strategy is consistent, but the three
*fundamentally different* income sources have mutually-offsetting tails. Stack them and
the equity curve smooths into something a small trader can live on.
**Construction (prespecified grid, no weight optimization):**

| Stream | Fundamental source | Role | Variant A | B | C |
|---|---|---|---|---|---|
| USDT lending | contractual (liquidity supply) | floor | 60% | 40% | 0% |
| Funding carry | contractual (leverage demand) | floor | 40% | 40% | 100% |
| Cascade sleeve | forced flows (R104, validated) | crash convexity | 0.10× | 0.15× | 0.10× |
| Frozen champion | statistical (R100 log) | harvest | 0.10% risk | 0.15% | 0.10% |

---

## Results — 70 weeks (Jun-2025 → Oct-2026), costs included, close-only sleeves

**Regime-filled** (carry pre-Jun-2026 = measured mean +0.121%/wk, disclosed assumption):

| Stack | Positive weeks | Worst week | Max DD | Ann. | Positive months | GATES |
|---|---|---|---|---|---|---|
| A income-heavy | 77.1% | **−1.07%** | **−1.62%** | +3.4% | 82% | ✅ |
| B balanced | 74.3% | −1.64% | −3.47% | +2.8% | 82% | ✅ |
| C carry-max | **78.6%** | **−1.02%** | **−1.17%** | +6.2% | 82% | ✅ |

**Conservative** (pre-Jun-2026 carry = 0): worst week −1.1…−1.7% ✅ · maxDD ≤4.8% ✅ ·
positive weeks 58–60% ❌ (the only gate that fails without the regime assumption).
Flagship B: **14 of 17 months positive, worst month −1.66%** · Monte Carlo: P(loss) 13%,
P5 = 0.98× (worst-5% case loses ~2%).

## What this means — read carefully

1. **The consistency machinery exists.** Worst week ≈ −1%, maxDD ≈ −3%, 14/17 green
   months. Nothing else tested in 105 runs comes close. The gates were:
   ≥65% weeks · ≥−2% worst week · ≤12% DD — **passed**.
2. **The pass leans on the carry regime assumption.** At *today's* measured funding
   (~4% APR), the contract floor alone yields only ~0.1%/week, and the conservative
   mode (no carry credit before Jun-2026) drops weekly positivity to ~60%.
   **In a bull-cycle funding regime (10–30% APR — 2021/2024-like), the same stack
   clears everything with room to spare** — that's the honest conditional.
3. **The dollar ceiling at today's regime:** ≈ **$1.2/mo per $500 · $2.4/mo per
   $1,000 · $4.7/mo per $2,000** (variant B). Consistency: achieved. Magnitude: the
   market isn't paying much for carry right now. The levers are (a) funding regime,
   (b) capital base — not more strategy risk.

## THE 2027 LIVE-PAPER PLAN (the hunt's deliverable)

- **Run variant B** (balanced): 40% USDT flexible lend · 40% hedged carry basket
  (24 coins, delta-neutral, per R102 spec — TRX excluded) · champion frozen at 0.15%
  risk/trade · cascade sleeve 0.15× (3 slots, R104 spec).
- **Weekly ritual:** collect lending+funding, log sleeve trades. **Monthly:** refresh +
  R096 recheck; stack rebalance only if weights drift >10%.
- **Kill rules:** stack drawdown >6% → sleeves to zero, floors only. Champion keeps its
  own 6-month kill-switch. Carry basket: exit any coin with 30-day mean funding < 0.
- **Re-evaluation triggers:** funding APR > 10% for 4+ weeks → scale carry to max;
  R096 champion revalidation failure → champion sleeve to 0 (floors unaffected).
- **Success criterion for going live with size:** 12 paper weeks ≥ 70% green and worst
  week ≥ −1.5%.

## Disclosure ledger

- Carry pre-Jun-2026 mean-filled (conservative variant reported side-by-side).
- Lending measured from Nov-2025 (OKX history depth); funding from Jun-2026.
- Cascade sleeve = R104 validated series × sleeve weight (no re-fit).
- Champion = frozen R100 trade log × risk weight (no re-fit).
- Units bug in early drafts (×100 in metrics printer) — all headline numbers re-verified
  from raw CSVs; correction documented in charter.

*Files: `quantlab_r105_allweather.py` · `r105_stdout.log` · `r105_flagship_weekly.csv` ·
`r105_a_cons.csv` · `r105_b_cons.csv` · `r105_c_cons.csv`*
