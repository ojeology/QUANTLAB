# RR LADDER — TRUTH AUDIT + ROBUSTNESS BATTERY (2026-10-24)
Follow-up to `champ_rr_sweep_2026-10-24.md`. Two questions answered: (1) how much of the wide-RR
edge survives the intrabar SL-first honesty test; (2) do the cells survive the standard battery.

## 1. SL-first truth audit (ambiguous 1H bars re-scored on 15m)
Method: replicate the engine's 1H exit walk per trade; find bars touching BOTH TP and SL;
re-score those with 15m ground truth (8 covered symbols: BTC ETH XRP LINK AVAX DOGE BCH LTC,
15m refreshed to 2026-10-24 18:15 UTC). Walker verified: 38/38 exit-label agreement with the
engine book at RR1.5, all trades resolved.

| RR | n covered | ambiguous bars | SL-first | PF optimistic | PF 15m-truth | PF ultra (ambig→SL) |
|----|-----------|----------------|----------|---------------|--------------|---------------------|
| 1.0 | 43 | 0 | — | 1.97 | 1.97 | 1.97 |
| 1.25 | 41 | 0 | — | 2.38 | 2.38 | 2.38 |
| 1.5 | 38 | 0 | — | 2.06 | 2.06 | 2.06 |
| 1.75 | 37 | 0 | — | 2.35 | 2.35 | 2.35 |
| 2.0 | 37 | 0 | — | 2.15 | 2.15 | 2.15 |
| 2.5 | 36 | 0 | — | 2.63 | 2.63 | 2.63 |
| 3.0 | 34 | 0 | — | 2.85 | 2.85 | 2.85 |

**Finding: ZERO ambiguous bars on the entire ladder** in the covered sample. Mechanically
consistent: these are low-volatility coil entries; a single 1H bar must span (1+RR)×ATR to be
ambiguous (2.0× at RR1.0 … 4.0× at RR3.0) — compression pops rarely do. R098's 1m audit found
~2% flips at RR1.5 on the FULL book; expected ambiguity pool here was ≲1 trade. 
**Caveat: 15m coverage = 8/73 symbols (this sandbox holds 15m for 8 only). The bound is proven
on the covered subset, not the whole book. Standing requirement (convention #5) unchanged:
full-book 1m/15m SL-first bound before live capital.** Verdict: the wide-RR optimism trap is
EMPIRICALLY SMALL on tested ground — the sweep table stands as quoted.

## 2. Robustness battery (full book, q0.55, 0.05%/side)
| RR | boot P(PF>1) | LOAO-min PF | PF@0.03 | PF@0.075 | PF@0.10 | MC P(profit) | MC med MDD | MC p95 MDD |
|----|--------------|-------------|---------|----------|---------|--------------|------------|------------|
| 1.0 | 1.00 | 1.45 | 1.63 | 1.33 | 1.18 | 100% | −6.5% | −11.6% |
| 1.25 | 1.00 | 1.66 | 1.85 | 1.55 | 1.41 | 100% | −5.6% | −9.9% |
| 1.5 | 1.00 | 1.59 | 1.76 | 1.50 | 1.37 | 100% | −6.5% | −11.6% |
| 1.75 | 1.00 | 1.79 | 1.99 | 1.72 | 1.59 | 100% | −5.8% | −10.3% |
| 2.0 | 1.00 | 1.81 | 2.00 | 1.72 | 1.59 | 100% | −6.1% | −10.8% |
| 2.5 | 1.00 | 1.89 | 2.07 | 1.82 | 1.70 | 100% | −6.3% | −11.5% |
| 3.0 | 1.00 | 2.00 | 2.21 | 1.95 | 1.82 | 100% | −6.4% | −11.4% |

Readings: (a) plateau, not a spike — every cell LOAO-min ≥ 1.45, monotone in RR, no cliff at
0.10%/side (worst cell 1.18); (b) the historical "dies at 0.10%/side" claim was RR1.5-era
full-book — today's book tolerates it at every RR; (c) **MC DD numbers are IID-optimistic**
(resampling destroys loss clustering; realized MDD is −13…−24%) — use them for P(profit) only,
never for DD planning. Kill-gate remains the binding evidence of decay (only RR1.5 trips).

## 3. Bot risk-gate audit item CLOSED (was open from 2026-10-16 directive)
`MAX_DD_LIM` is NOT dead code: `check_risk_gates()` is called in `process_symbol()` (per-symbol,
every scan) and blocks new entries below −15% drawdown. Semantics labeled in-file: it is an
ENTRY-HALT, not a bot halt — open trades always run to TP/SL. Intrabar TP-first booking now
carries the SL-first disclosure comment. Regression test #8 added → suite 8/8.

## 4. Net effect on the re-entry plan
The pre-registered ladder {1.5, 2.0, 2.5, 3.0} judged on post-2026-10-24 data stands unchanged.
What this battery adds: none of the wide-RR cells is a robustness mirage (plateau + cost-tolerance
+ LOAO all hold); the choice among them will be decided by forward data only.
