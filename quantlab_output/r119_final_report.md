# R119 — CORRECTED TWIN HUNT (audit run) — prespec locked pre-run 2026-10-24
Purpose: re-run the R118 twins on the locked accounting standard (ql_accounting.py), quantify the
audit-confirmed defects, and emit final twin-hunt numbers. No new selection, no new variants.
Universe: 12-coin core (BTC ETH BNB SOL XRP DOGE ADA LINK LTC AVAX DOT TRX) — disclosed shrink
after the 9th cache wipe; H-T2 = exit-to-flat state machine inherited from the R118 fix
(R117 remains the true-flip DON55 reference). Costs 0.05%/side (H-T1), 0.07%/side (H-T2) + funding.
Invariants ON: every-bar marking, ±60% tripwire, no-unapplied-pending, P&L sourced ONLY from
acct.effective_weights(i) (the off-by-one law).

## RESULT (charter: yearly≥0, posW≥65%, worstW≥−3%, maxDD≤12%, holdout≥0)
| variant | CAGR | Sharpe | maxDD | posW | worstW | gates | holdout 2026-07→09 |
|---|---|---|---|---|---|---|---|
| H-T1 always-in | +6.5% | 0.50 | −22.2% | 48.6% | −4.90% | 1/6 | +10.65% ✓ |
| H-T1 filtered | +10.6% | 0.91 | −14.8% | 31.2% | −4.90% | 3/6 | +3.85% ✓ |
| H-T2 classic | −0.4% | 0.23 | −47.4% | 47.2% | −17.35% | 2/6 | −8.22% ✗ |
| H-T2 timestop | +14.1% | 0.52 | −42.1% | 45.8% | −17.35% | 3/6 | +6.43% ✓ |

**VERDICT: ❌ REJECT — all four variants fail the charter (best 3/6).**

## Defect quantification vs superseded R118 (the audit deliverable)
| variant | R118 claimed CAGR/maxDD | corrected | cause |
|---|---|---|---|
| H-T1 always-in | +10.8% / −24.7% | +6.5% / −22.2% | cost overwrite (−4.3 pp CAGR) |
| H-T1 filtered | +10.0% / −20.7% | +10.6% / −14.8% | costs minor; universe shrink ±; DD −5.9 pp |
| H-T2 classic | +19.8% / −38.6% | −0.4% / −47.4% | same-bar off-by-one phantom (−20.2 pp) |
| H-T2 timestop | +27.7% / −41.3% | +14.1% / −42.1% | same (−13.6 pp) |

H-T2 classic holdout flips POSITIVE→NEGATIVE under corrected accounting (+9.5%→−8.2%): the phantom
breakout-bar returns were carrying it. Timestop holdout +6.4% survives but the variant still fails
yearly and worst-week gates. Sample: H-T2 = 32/44 trades total — too thin to promote even if gates passed.

## Audit self-catch (recorded for the lab's permanent record)
The same-bar off-by-one was captured a FOURTH time during this audit itself — in R119's own first
draft (phantom +54.5% CAGR before invariants flagged it). Structural prevention:
`LadderAccount.effective_weights(i)` is the only legal P&L weight source (regression-tested).

Twin hunt closed under corrected accounting. "Only the floor" law stands.
Reproduce: `python3 quantlab_r119_corrected.py` (cache `quantlab_cache/bnfut/`, curves `r119_*.csv`).
