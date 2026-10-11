# R108 FINAL REPORT — F2: FX CARRY (The Contractual Premium)
**Three-year test 2024/25/26 · dual rate spine · verdict: ❌ REJECTED as core — validated as a premium, unfit for the consistency charter**

---

## What was tested

**Hypothesis (who pays):** interest-rate differentials are a contractual premium centuries old
(the carry trade): speculators borrow low-yield currencies to chase high-yield ones, and the
high-yield holder is paid to bear sudden-devaluation risk. Contractual income + price tail —
the exact shape the R103 law said survives retail costs.

- **Universe:** 21 synthetic non-USD crosses built from the 8 majors (inner-join timestamps, close-only).
- **Rate spine (the hard part, disclosed):** FRED/OECD immediate rates (fresh through 2026 for
  USD/EUR/GBP/JPY/CAD/AUD). CHF dies 2024-03 and NZD dies 2024-12 in OECD's feed → supplemented
  with documented central-bank schedules (SNB 1.50→0.00 by 2025-06; RBNZ 5.25→2.50 by 2025-08;
  RBA 4.35→3.60 by 2025-08). **Stress spine:** supplemented currencies frozen at last verified
  values. Verdict required to hold on both.
- **Rule:** long-only top-K by carry (K∈{3,5}), rebalance weekly/monthly, 0.5% risk per position,
  bucket 2.5×ATR-proxy, costs 1.5 pip RT + 0.7 pip/day swap take.
- **Gates:** charter 6 + holdout. **Anchor (prespec'd):** week of 2024-08-05 must be among worst weeks.

## Result (best variant K=5 monthly, fresh spine; full window 2024→2026-10)

| Metric | Result | Gate | |
|---|---|---|---|
| Yearly ledger | **2024 +46.9R · 2025 +39.6R · 2026 +112.2R** | no losing year | ✅ |
| Positive weeks | 60.3% | ≥65% | ❌ |
| Worst week | **−27.9%** | ≥−3% | ❌ |
| Max drawdown | −47.0% | ≤12% | ❌ |
| Stress spine | all-years-positive too, gates 3/6 | robustness | ✅ (income sign) |
| ANCHOR | **2024-08-04 = worst week in sample (−27.9%)** | must be present | ✅ |

The anchor result is the crown of this round: the backtest independently reproduced the
historical **yen-carry unwind crash of August 2024** as the single worst week — across all
variants and both spines. The test reflects reality; the tail is real, not a data artifact.

## The verdict, in plain terms

Carry is a **genuine premium that made money in all three calendar years, on both rate
spines — but its weekly profile is violent**: a −28% week and a −47% drawdown. Sizing it down
to tame the tail (÷6) still fails the ≥65% positive-weeks gate, because at weekly horizon the
carry drift (~+0.05%/wk) is smaller than cross-price noise: **the sign of any given week is a
coin flip with a small positive drift, punctuated by rare steamroller weeks.** That is
structurally incompatible with the consistency charter at ANY size.

## The deep connection (why this round matters)

Unhedged FX carry fails. But we already SHIPPED its domesticated cousin: **the R106 hedged
floor IS carry with the tail cut off** — lend the coin + short the perp = collect the funding
differential while the price risk cancels. Same "who pays" (crowds paying for leverage),
premium intact, steamroller deleted. The hunt's two halves now confirm each other:
**contractual premia survive retail costs only when hedged.**

## Audit trail (4 defects caught this round)

1. FX bar cache wiped by snapshot trim AGAIN → refetched in 12s (fetch_forex.py is now a standard tool).
2. Cross-sectional ranking bug (scalar partition) → panel redesign before any results existed.
3. Rebalance marks: np.datetime64 set vs Timestamp hashing — marks silently never matched
   (7 trades/3 years, y2024=0.0) → hash-free key-change rule.
4. **Zombie positions:** day-one NaN ATR created entries with NaN buckets that squatted the
   top-5 slots producing zero P&L → untradable-at-NaN rule. (Found BECAUSE y2024=0.0 was
   physically impossible for a carry basket earning ~3%/yr.)

## Standing ledger after F2

Behavioral FX (F010–F013, R107 trend): dead. Unhedged contractual FX carry: real but unfit.
Remaining queued: **F3 — daily-bar CTA-horizon trend** (the honest last word on trend, where
the academic premium actually lives). After F3, the FX hunt closes with its law complete.

## Artifacts

`quantlab_output/r108_carry_weekly.csv` · `r108_carry_trades.csv` · `r108_stdout.log` ·
rate caches: `quantlab_cache/forex/rates_*.csv` (8 FRED series).
