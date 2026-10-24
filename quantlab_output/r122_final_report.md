# R122 — COT POSITIONING → FX — FINAL (2026-10-24)
Prespec: docs/RESEARCH_FRONTIER_R122.md (locked pre-pairing; two data deviations amended pre-run:
Yahoo dailies both eras after HistData throttling + a sandbox cache wipe; NZD COT ends 2022-02 per
CFTC discontinuation). Data: CFTC Legacy Futures-Only (Socrata), 7 currencies, 1,368–1,940 weeks
each (1986→2026-10-20); prices Yahoo dailies 2003→2026-10. First COT bytes this repo has ever held.

## CELL A (PRIMARY) — follow the hedgers (comm net % z156, weekly, Mon→Mon, costs on flips)
| era | n wks | PF@c | Sharpe | bootP | maxDD | total |
|---|---|---|---|---|---|---|
| DISCOVERY 2003→2015 | 684 | **0.94** | **−0.14** | **0.309** | −22.1% | −14.4% |
| CONFIRM 2016→2026 | 521 | 1.14 | +0.35 | — | −10.6% | +14.2% |

Gates: G1 FAIL · G2 FAIL · G3 FAIL · G4 PASS (LOO-min 0.93) · G5 PASS.
**VERDICT: REJECT** — the primary hypothesis is flat-to-negative in the clean discovery era.
The confirmatory era's +14.2% (PF 1.14) is the era-instability trap in its purest form: the SAME
signal loses for 13 years then wins for 10 — chasing that flip is how TSMOM-style mirages are born.
Recorded as observation, never as a lead.

## CELL B (SECONDARY, reported-only) — fade spec extremes (90/10 pct, 260w)
Discovery: n=238, PF@c 0.94, −8.2% → also dead. (Confirm n=0 — extremes never triggered post-2015.)

## CLOSURE STATEMENT — THE FX MAP
With R122, both remaining FX levers are now tested with receipts: **price-derived** signals
(trend ×3 horizons, sessions, breakouts, compression, carry, crossovers: F001–F013, R107–R109)
and **positioning-derived** signals (hedger-following, spec-extreme fading: R122), plus the
cross-asset filter transfer (R121, near-miss). FX is not "out of ideas" — it is **mapped**:
nothing derivable from FX price or CFTC positioning data paid at this lab's bars, across 23 years.
Any FX revisit requires a NEW lever (new data type, e.g. true flow/L2, or a structurally different
edge class) — per charter, documented here as the closing entry.
