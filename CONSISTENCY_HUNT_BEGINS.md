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

## 🔁 REOPENED — ONE SEAT: R113 PRESPEC (2026-10-11, locked BEFORE testing) — CROSS-SECTIONAL MOMENTUM

The last untested payer in the building. Who pays: latecomers chasing relative winners
(documented cross-sectional momentum, Jegadeesh–Titman; strong in crypto at 1–4 week
formations). The rotation holder sells relative strength to the crowd that arrives late.

Rule: at each week's final daily close — rank all 20 coins by trailing N-day return
(N ∈ {7,14,28} days), hold top K (K ∈ {3,5}) equal-weight 1/K until next rebalance.
Exposure variants: always-in, OR cash-unless-BTC>SMA50 (beta filter — long-only crypto
momentum's known enemy is beta; the filter variant is the real candidate).
Costs: 0.05% per side on every weight change (turnover-counted, not hand-waved).
Off-by-one law: weights decided at rebalance close apply to NEXT bar's returns.
SANITY GATE (new, automatic): any weekly return outside ±60% ⇒ abort (unit-slip tripwire).
GATES (unchanged, locked Aug): posW ≥65% · worstW ≥−3% · maxDD ≤12% · 2024/25/26 each ≥0R ·
holdout (2026-04→10) ≥0R. Selection 2024-01→2026-03 picks the variant; holdout untouched.

## R113 RESULT (2026-10-11) — ❌ REJECTED (2/6) — **THE MAP IS COMPLETE**

Cross-sectional momentum (12 variants, weekly rotation, turnover-costed): the premium is REAL and
holdout-confirmed (+60.7% holdout; +46%/+46% best years) — but worstW −33%, maxDD −75%, posW ~25–50%.
**Expectancy and survivability are different axes.** Momentum is a fund's trade with hedges and 200
holdings, not a consistency account's.

### ══ FINAL MAP AFTER 113 ROUNDS ══
Behavioral signals: thin/decaying (15 strategies, 3 markets). Unhedged carry: real, tail-heavy.
Momentum: real, uninhabitable. **Hedged-contractual floor (R106): the only all-gates pass in history.**
Everything exciting fails survival; everything that survives is boring rent. Charter filter vindicated.
→ Research map CLOSED. Standing: Opay (naira pot) · STACK v2 floor (dollar pot, trial pending) · champion recheck end-2026.

## R114 PRESPEC (2026-10-11, locked) — THE "PLAIN SIGHT" SWEEP (benchmarks + mirage check)

Hypothesis: if any truth was missed, it hides in the scenarios too obvious to test.
- **B1 BENCHMARKS (no gates — baselines):** buy-and-hold yearly ledgers 2024/25/26 + maxDD +
  worst week for BTC, ETH, XAU, XAG (gold/silver sat in the cache the whole time — plain sight),
  and weekly-rebalanced EQW-20 (0.05%/side on turnover). Question: did "doing nothing" pay more
  than everything we built — and would the user have SURVIVED it (the 2025 reality)?
- **B2 DCA:** $10/week into BTC / ETH / EQW-20 since 2024-01: invested, final, multiple,
  worst unrealized drawdown. The strategy of people who never read a chart.
- **B3 SEASONALITY FALSIFICATION (gated):** 10 pre-declared calendar patterns × {EQW-20, BTC}
  (day-of-week ×5 grouped, turn-of-month, weekend, first/last week, month boundaries) on
  2024-26 dailies. **Bonferroni law: 20 tests ⇒ significance only at p < 0.0025. Prediction:
  ZERO survive. If ≥1 survives, it earns a dedicated round.**
- **B4 FLOOR UPGRADE PROBE:** retry OKX staking endpoints (dead in R106-era) for a cheap
  yield upgrade to the floor. 5-minute probe, no round spent if empty.
Disclosures: short-side seasonality untestable (borrow costs unmodeled); 2022 regime forever
out of reach; XAU/XAG via OKX SWAP dailies (venue-consistent with the coins).

## R114 RESULT (2026-10-11) — PLAIN-SIGHT SWEEP — MAP CLOSED WITH BENCHMARKS

Buy & hold: BTC +109%/−8.8%/−7.1% (2 losing years, maxDD −53%) · EQW-20 2025 −55.7% (maxDD −80%) → "doing nothing" FAILS the survival law.
DCA $10/wk: BTC 1.09× · ETH 0.95× · EQW-20 0.84× — folk wisdom lost on 2/3.
Seasonality: **0/20 survive Bonferroni (p<0.0025)**; seductive "CPI +242%/yr" = p 0.08 = noise (R111 real entries were negative).
Staking: endpoints dead twice-confirmed. **No plain-sight truth exists outside the floor.**
**FINAL: 114 rounds · every payer class, benchmark, and calendar mirage measured · hedged-contractual floor stands alone. Closed, with receipts.**

## R117 PRESPEC (2026-10-16, locked BEFORE testing) — DAILY-CANDLE TREND ON CRYPTO PERPS

The user-spotted gap: classic daily-bar trend was never given a dedicated round (components only).
Who pays: slow institutional repricing + retail anchors (behavioral trend premium at the ONE
horizon where retail costs are smallest: ~15 trades/yr, 0.07%/side vs multi-% moves).

Data: Binance Vision 1D klines, backfilled to 2020-01 for warmup (SMA200 needs it); funding
epochs (actual) as hold cost; universe = >=97% pre-fill coverage coins (the 24-30 panel).
Signals (prespec'd, classic only, no mining): DON20/DON55 Donchian long/flip;
SMA100/SMA200 regime long/flat. Universes: BTC-only, ETH-only, EQW-book (all coins, 1/N).
Costs: 0.07%/side on position changes; funding = actual daily epochs on held longs.
No leverage (1.0x); gates: charter law — yearly 2024/25/26 each >=0, maxDD <=12%, worstW >= -3%,
posW >=65%. Selection 2024-01->2026-03 (Sharpe) picks variant; HOLDOUT 2026-07->09 untouched.
2020-2023 printed as context only (incl. the 2022 bear — the regime this style is famous for).
Verdict mapping: charter standard (SURVIVES / INCONCLUSIVE / FAILS AFTER COSTS / etc).

## R117 RESULT (2026-10-16) — DAILY-CANDLE TREND ❌ FAILS CHARTER (2/6) — GAP CLOSED, LEGEND CONFIRMED

DON55/BTC selected: +28% APR but maxDD −35.8%, worstW −11.9%, posW 49%, 2026 −9.5% → 2/6 gates. Holdout +2.4% ✓ (first survivor of the killer quarter).
**Six-year context (2020-25): +103/+108/−6.5(2022 bear!)/+13/+61/+25 — the daily-trend legend is TRUE (sat out 2022, beat B&H over the cycle at 1/3 the drawdown) — and STILL uninhabitable under the consistency law.**
Law (third form): cycle-surviving strategies ≠ charter-surviving strategies. Only the floor does both.
Data spine now spans 2020-01→2026-09-30 (2,465 bars × 38 syms + funding). Sprint complete: R117 opened, tested, closed in one session.

## 🔥 REOPENED (2026-10-16) — THE HUNT FOR A LIVABLE EDGE: TWIN CAMPAIGN (H-T1 + H-T2)

Premise (from the ledger): the repo holds 3 REAL premia. Floor ✅ habitable. The other two are
real but unlivable by mechanism, and the mechanism has never been actually engineered around:
- H-T1 MOMENTUM (R113: +61% holdout, −75% DD): the pain is CONCENTRATION (top-3 = idiosyncratic
  holes). Untested lever: DIVERSIFY + DELEVERAGE (top-8, 1/8 each, vol-targeted, BTC-beta filter).
- H-T2 DAILY TREND (R117: beats B&H over 6y, −36% DD): the pain is DRAWDOWN (flip exits).
  Untested lever: TIME-STOP SOFTENING (band exits, exit-at-cross variants).

**Prespec H-T1 (locked BEFORE testing):** cross-sectional momentum, weekly Friday close,
N=14d signal (R113's), top-8 equal-weight 1/8, vol-target each sleeve to 8%/yr at formation
(clip [0.25,2]x), BTC>SMA100 cash filter variant on/off. Costs 0.05%/side on weight changes.
Selection 2024-01→2026-03 (Sharpe) picks filter on/off only. HOLDOUT 2026-07→09 untouched.
Gates: charter law (yearly ≥0 ×3, maxDD ≤12%, worstW ≥−3%, posW ≥65%) + holdout ≥0.
**Prespec H-T2 (locked):** DON55 BTC perp long/flip (R117's), exits: (a) classic N-high/low
break, (b) soft band: exit long only if close < max(SMA100, DONlow20), symmetric short,
(c) time-stop: exit if |unrealized| < 0.5×ATR30 after 30d. Selection same window picks a/b/c.
**Verdict mapping:** charter standard. 2020-2023 context for both. Zero new data needed.
**The hunt ends the day one of these passes charter + holdout — or both fail and the
"only the floor" law gets its final replication. Either outcome is a genuine edge answer.**

## TWIN HUNT RESULT (R118, 2026-10-16) — BOTH LEVERS REAL, NEITHER LIVABLE: LAW FINALLY REPLICATED

H-T1 momentum diversified+vol-targeted: DD −75%→−21/−25%, 2025 rescued (+3%), holdout ✓ — but 3/6 gates (maxDD/posW/worstW still fail).
H-T2 DON55 + time-stop: 2026 rescued (−23%→+5.4%), holdout ✓, APR +27.7% — but DD DEEPENED (−41%), 3/6 gates. Risk moved, not removed.
**Engineering can dilute or relocate charter violations; it cannot delete them. Only the contract-paid floor has nothing to engineer because it has nothing to fear.**
Audit: off-by-one resurrected in fresh state-machine code, KILLED by cross-check vs R117 (3rd appearance of this bug family — asserts now mandatory everywhere); softband variant void (degenerate prespec wording, disclosed).
**The hunt's termination condition, met: neither premia survivable ⇒ "only the floor" is now a thrice-tested law, not a hypothesis.**
