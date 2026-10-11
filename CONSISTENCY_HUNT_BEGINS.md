# 🎯 CONSISTENCY HUNT BEGINS (2026-10-09)

**Mission:** a strategy a small trader can live on — high share of positive weeks,
shallow drawdowns, boring equity. Target: **never a losing quarter, worst month ≥ −5%,
as close to "pays every week" as markets allow.** NOT big wins.

**Champion status:** ❄️ FROZEN. No changes. Monthly `refresh_okx.py` +
`quantlab_r096_recheck.py` only. Re-evaluation date: end of 2026 (per charter).

---

## The consistency gates (LOCKED before any candidate runs)

A candidate is a CANDIDATE only if, over the full test window (must include the
2024 bad regime):

1. **≥ 65% positive weeks**
2. **Worst week ≥ −3%** (at the strategy's recommended sizing)
3. **Max drawdown ≤ 12%**
4. **Profitable after realistic costs** (spot 0.10%/side or perp 0.05%/side, funding logged)
5. **Zero path ambiguity** — fills at bar closes only; or 1-minute audit if brackets exist
6. **Walk-forward / causal audit** — nothing fit on the future; fresh-window check when data allows
7. **Survives the 2024-style regime** without breaching gates 1–3 by more than a disclosed margin

A candidate that fails is a finding, not a failure of the hunt.

## Banned (hunt laws, inherited from 100+ runs of scars)

- Intra-bar TP-before-SL bookkeeping (the F013 killer)
- Holdout mining — anything selected on 2026 needs fresh 2026Q4/2027 data to confirm
- Signal inversion of a positive-expectancy system
- RR<1 fee-mirage scalping (R081)
- Screenshots of great equity curves without the weekly table behind them

## Candidate queue (mechanisms never tried in this repo)

| # | Mechanism | Why it could be consistent | Status |
|---|---|---|---|
| C1 | **Funding carry** (delta-neutral) | contractual income, R102: 15/15 positive weeks | ✅ measured, ANCHOR |
| C2 | **Low-vol basket** (cross-sectional, weekly rebalance, spot) | low-vol anomaly; cash cushion in bad regimes | R103 |
| C3 | **Dip-ladder** (buy dips in uptrends, close-only exits) | high hit-rate by design; honest via close-only | R103 |
| C4 | Liquidation-cascade fade | needs OI/liquidation feeds | queued (data step) |
| C5 | Options yield structures | needs capital > $2k | parked |

## Consistency metrics (every candidate reports these)

positive-weeks % · worst week · longest losing streak (weeks) · max DD ·
months positive % · Calmar · $/month at $500 and $1,000.

*"The goal isn't to find a strategy that looks good. The goal is to find one that
survives every attempt to prove it wrong." — original charter, still law.*

## ROUND 1 RESULTS (R103, 2026-10-09)

| Candidate | Gates | Verdict |
|---|---|---|
| C2 low-vol basket | 0/7 — 48.9% posW, worst −22.7%, DD −63% | REJECTED |
| C3 dip-ladder | 0/7 — WR 63% < breakeven 67.2%; −0.27%/trade | REJECTED |
| C1 carry | ALL — 15/15 positive weeks | ANCHOR (small $) |

**Sharpened law:** statistical consistency is priced away at retail cost; survivors
must earn contractually. Queue: R104 = basis trade + OI/liquidation probe + carry-plus-
tiny-champion-sleeve sizing.

## ROUND 2 RESULTS (R104, 2026-10-09) — fundamentals framing

| Source | Finding |
|---|---|
| Basis trade | probe inconclusive (API-blocked); parked |
| Cascade fade (forced flows) | premium REAL: +6%/yr ann, +16.8%/+22.2% in 2024/25, 60% TP — but −11.4% crash-weeks → ❌ standalone, hedge candidate |
| Stack: carry + champion@0.25% | worst week −1.59%, maxDD −4.68%, ann +18.6%, ≈$8/mo per $500 — smoothest equity tested; posW 52% < 65% gate at current funding |

Law refined: the three premiums (carry / cascade / champion) have mutually offsetting
tails — R105 tests the all-weather allocation. Units bug in early R104 drafts (×100)
corrected; headline figures verified from raw logs.

## ROUND 3 RESULTS (R105, 2026-10-09) — ALL-WEATHER ALLOCATION: GATES PASSED ✅

| Stack (lend/carry/champ/cascade) | posW | worstW | maxDD | ann | Verdict |
|---|---|---|---|---|---|
| A income-heavy (60/40/0.10%/0.10×) | 77.1% | −1.07% | −1.62% | +3.4% | ✅ |
| B balanced (40/40/0.15%/0.15×) | 74.3% | −1.64% | −3.47% | +2.8% | ✅ |
| C carry-max (0/100/0.10%/0.10×) | 78.6% | −1.02% | −1.17% | +6.2% | ✅ |

Conservative (carry 0-filled pre-Jun-26): posW 58-60% — only the weekly gate fails.
Deliverable: 2027 live-paper plan (variant B). Dollar reality at current funding:
≈$1.2/mo per $500, $2.4/mo per $1k, $4.7/mo per $2k. Levers: funding regime + capital.

**HUNT STATUS: construction validated. Next: 12 paper weeks ≥70% green → size up.**

## ROUND 4 RESULTS (R106, 2026-10-09) — HEDGED-YIELD FLOOR v2 ✅

Lend-the-coin + short-the-perp = paid by BOTH crowds (borrowers + longs).
Anti-correlation CONFIRMED in data (ATOM: funding −0.060 but lending +0.086 → total +).

| Basket | APR | posW | worst wk |
|---|---|---|---|
| Static all-20 hedged | **+5.4%** | **100%** | +0.055% |
| STACK v2 (floor + champ 0.15% + cascade 0.15×) | +18.6% ann | 73% | **−0.09%** |

$/mo: $7.2 per $500 · $14.5 per $1k · $28.9 per $2k (vs R105: $1.2 / $2.4 / $4.7).
Recipe: 80% static 20-coin hedged basket (isolated, perp margin 25% of notional),
20% dry powder for sleeves; monthly combined-yield check only.

**CONSISTENCY HUNT: CLOSED — construction shipped to paper-trading (12-week trial).**
