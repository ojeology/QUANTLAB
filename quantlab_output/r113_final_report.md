# R113 FINAL REPORT — CROSS-SECTIONAL CRYPTO MOMENTUM (The Last Honest Candidate)
**Verdict: ❌ REJECTED under the consistency charter — the premium is real, large, and unlivable**

---

## What was tested (prespec: CONSISTENCY_HUNT_BEGINS.md, reopened one seat)

**Who pays:** latecomers. Capital chases relative winners (documented cross-sectional
momentum, Jegadeesh–Titman; especially strong in crypto at 1–4 week formations). The rotation
holder collects from the crowd that arrives after the move is visible.

**Rule:** every week's final daily close — rank 20 coins by trailing N-day return, hold top K
at 1/K until the next rebalance; BTC>SMA50 cash-filter variant; 0.05%/side on every weight
change; weights decided at a close apply to the NEXT bar (off-by-one law); new automatic
sanity gate (any weekly |>60%| aborts — unit-slip tripwire). 1 data bar scrubbed.

## Results (12 variants, full grid 2024-01 → 2026-10)

| Variant (best of each family) | posW | worstW | maxDD | Yearly % | Gates |
|---|---|---|---|---|---|
| N=14 K=3 always-in | 49% | −33.0% | −79.6% | **+106 / −34 / −9** | 1/6 |
| N=14 K=3 BTC-filter ★ | 23% | −33.0% | −74.6% | +46 / −35 / **+24** | 2/6 |
| N=28 K=3 BTC-filter | 25% | −25.3% | −69.7% | +46 / −39 / **+46** | 2/6 |

★ selected on selection window. **Holdout 2026-04→10: +60.7% — PASS** (the premium is genuinely
there out-of-sample) …through a −20.7% week on the way.

## The finding — the perfect capstone

Cross-sectional momentum is the **first candidate since the floor with large, persistent,
holdout-confirmed expectancy** — the selected family printed +46%/+46% in its two best years
and +61% in the holdout. And it still fails: **worst week −33%, drawdown −75%, half the weeks
red, 2025 negative.** The premium is real; the *ride* is uninhabitable for a consistency
mandate. This is the textbook dichotomy of finance in one experiment: **expectancy and
survivability are different axes.** Momentum is what a diversified fund harvests with hedges
and 200 holdings; it is not what a $2,000 consistency account can hold.

## 🔒 THE MAP IS NOW COMPLETE — 113 ROUNDS

| Family | Payer | Verdict |
|---|---|---|
| Behavioral signals (14 strategies, 3 markets, every clock) | latecomers/herds/panic | thin (+~0.1R/event), decaying, or tail-heavy |
| Unhedged contractual (carry) | central banks | real, steamroller tail |
| Cross-sectional momentum | latecomers (rotation) | real & large, uninhabitable ride |
| **Hedged-contractual (R106 floor)** | **both crowds, by contract** | **✅ the only all-gates pass in project history** |

Three years of adversarial self-testing converge on one sentence: **everything with exciting
returns fails the survival gates; everything that passes the survival gates is boring rent.**
The user's original spec — "consistent gains, $20–30/month, a losing year is unacceptable" —
was never small. It was the correct filter that killed 15 strategies and kept the one truth.

Standing: **Opay for the naira life-pot · STACK v2 floor for the dollar pot (12-week trial
whenever chosen) · frozen champion recheck end-2026 · the research map is closed.**

## Artifacts

`quantlab_output/r113_csm_weekly.csv` · `r113_stdout.log` · data: `quantlab_cache/daily/` (refetched).
