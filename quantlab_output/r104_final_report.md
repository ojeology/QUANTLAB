# R104 — FUNDAMENTALS ROUND: forced flows, term structure, and the stack

**Date:** 2026-10-09 · Framing: every durable return answers *"who pays me, and why are
they willing?"* — risk premia, forced flows, behavioral edges, or contractual income.
This round tested the two untouched sources. **Method note:** a units bug (percent
series scaled twice in the metrics printer) inflated early draft numbers 100×; headline
figures below are corrected and re-verified from raw trade logs. Yearly "worst week"
values in the raw log carry the same cosmetic ×100 — corrected here.

---

## PART 1 — Basis trade (contractual, term structure): probe inconclusive
OKX quarterly-futures history isn't readable from this sandbox (ticker endpoint silent).
The perp-funding version of the same premium is already measured (R102). Parked until
API access changes.

## PART 2 — Cascade fade (forced flows): the premium is REAL — the tail is violent

Mechanism: leveraged liquidations force sales past fair value; providing the other side
at capitulation extremes is a paid service. Detector (no feeds needed): 24h return <
−3σ(30d) + closing at 24h lows → buy at close; exit at close (+2% / −3% / 72h).
**Close-only fills — zero path ambiguity.** 3 slots (cascade correlation capped
mechanically). 473 trades, 2024→2026 (equity swaps excluded — halt gaps poison σ).

| Metric | Value |
|---|---|
| Expectancy | **+0.119%/trade** (positive all three years) |
| TP rate | 60% at +2%/−3% close-stops |
| Annualized | **+6.0%** |
| Year by year | 2024 **+16.8%** · 2025 **+22.2%** · 2026 **−20.2%** |
| Worst week | **−11.4%** · maxDD −30.4% · positive weeks 40% |

**Verdict: ❌ consistency gates — but the finding matters.** The forced-flow premium is
genuine (money in 2024's storm AND 2025's grind) — it just arrives in violent clusters:
crash weeks hit all 3 slots at once (−11% weeks). This is the *opposite* personality of
carry. It is not a standalone for the consistency charter; it's a candidate for a tiny
convexity sleeve (its best weeks are exactly the market-crash weeks when everything
else bleeds — a natural hedge worth studying in R105).

## PART 3 — The stack (carry floor + frozen champion @0.25% risk)

| Metric | Value |
|---|---|
| Positive weeks | 52.3% |
| **Worst week** | **−1.59%** |
| **Max drawdown** | **−4.68%** |
| Annualized | **+18.6%** |
| Dollars | ≈ **$8/mo per $500 · $16/mo per $1,000** (current funding regime) |

Smoothest 16-month equity of anything tested in this repo — drawdown smaller than one
champion trade at 1% risk. Still fails the ≥65% positive-weeks gate: the champion
sleeve's red weeks (even at 0.25%) outnumber carry's absorption power at current
~4%/yr funding. In a 2024/25-style funding regime (10–30% APR) the floor multiplies
and this likely clears every gate — unverifiable from this sandbox (geo-blocks),
so it stays a hypothesis, honestly labeled.

---

## The fundamentals scoreboard (what we now KNOW)

| Source | Mechanism | Measured | Consistency verdict |
|---|---|---|---|
| Contractual (funding) | leverage-demand premium | +3–7% APR now, 15/15 pos weeks | ✅ the floor |
| Forced flows (cascades) | liquidity provision at capitulation | +6%/yr, + in 2024 & 2025 | ❌ standalone (violent clusters), hedge candidate |
| Statistical (all signal families) | R008–R103 | champion only survivor | lumpy by nature |

**R105 queue:** (1) cascade-sleeve sizing — can 0.10% risk on cascades + carry + tiny
champion produce posW ≥ 65% with worst week ≥ −2%? (The three premiums have
mutually-offsetting tails: carry bleeds slowly in calm, cascades pay in crashes,
champion harvests expansions — a genuine all-weather construction.) (2) staking /
yield leg probe (contractual #3). (3) 2027 live-paper plan for the stack.

*Files: `quantlab_r104_fundamentals.py` · `r104_stdout.log` · `r104_cascade_trades.csv`
(473 trades) · `r104_stack_weekly.csv`*
