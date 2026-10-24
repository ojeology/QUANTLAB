# R121 — CROSS-ASSET TRANSFER: CRYPTO-TRAINED POP FILTER → FX — PRESPEC (LOCKED 2026-10-24, pre-data)
Operator hypothesis: "use what the champion learned from crypto on forex." Status when locked:
zero FX 2026-era signals computed; FX cache was wiped (refetch in flight); the crypto SVM below
is fitted ONLY on crypto trades with entry < 2026-01-01. Nothing below may change after the
first FX signal is scored.

## The one registered cell
Apply the champion's EXACT signal machinery (4 percentile conditions + green-candle + rel_vol
gate, trailing-500 recal-every-168) to 8 FX majors on 1H, then score every FX pop with the
CRYPTO-TRAINED SVM (frozen artifact, no refit on FX, ever) and keep the top q = 0.55.
Trade RR = 1.5, SL = 1×ATR(14), exits from first full bar. Question: does crypto's learned
notion of "doomed pop" transfer across asset classes?

## Prior (honest, recorded): LOW. FX arc closed 0/14 (F013); compression conditions were in 10 of
the 14 failed FX scripts (raw, unfiltered); the filter's hour/dow features may be session-specific.

## Data & deviations (locked)
- HistData ASCII M1 → 1H, 8 majors, 2016→2026-10. FX "volume" = high−low activity proxy (F012
  convention) → rel_vol = range vs 20-bar mean range. This is a documented deviation from
  crypto's tick-volume rel_vol — same code path, different volume semantics, disclosed here.
- Split: FX signals 2016–2020 = context only (reported, non-deciding). PRIMARY EVAL = 2021-01→2025-12.
  Calendar overlap with crypto training trades is permitted — the no-leak channel is that FX
  labels never enter the model (verified: fit set = crypto trades, entry < 2026-01-01).

## Costs (locked pips, round-trip, charged once at exit in R terms = pips / (ATR at entry))
EURUSD 0.6 · GBPUSD 0.9 · USDJPY 0.7 · AUDUSD 0.8 · USDCAD 1.0 · USDCHF 1.0 · NZDUSD 1.2 · EURGBP 1.2

## Fills (locked): SL-FIRST intrabar on ANY bar touching both levels (conservative convention #5;
FX has no 15m ground-truth coverage in this sandbox). Pessimistic adversarial proxy also reported:
every TP re-scored as SL.

## Gates (PRIMARY eval, ALL must pass ⇒ PROMISING FOR FURTHER VALIDATION)
- G1: PF@cost ≥ 1.10
- G2: n ≥ 100 kept trades
- G3: bootstrap P(net>0) ≥ 0.95 (2,000 draws)
- G4: leave-one-pair-out min PF > 1.0
MUST-BEAT baselines (reported, deciding): (a) raw unfiltered FX pops on the same window —
transfer is only interesting if it beats raw; (b) random-55%-keep distribution (200 draws) —
kept-set PF must exceed the random q-quantile.
Sensitivity (NON-selecting, reported): q=0.70, RR=2.0. No other variant may be computed.

## Verdict mapping
All gates + both baselines → PROMISING FOR FURTHER VALIDATION. Anything else → REJECT
(one cell, one run, no iteration). REJECT closes "crypto→FX transfer" permanently;
a pass leads to 2026 FX walk-forward as rung two. Charter monetization gates apply only at
any future monetization, not here.


## RESULT (2026-10-24) — REJECT (G3 failed: bootP 0.795 < 0.95)
Kept-set PF@c 1.15 vs raw 0.96 vs random-q 0.98; LOO all 8 pairs > 1.0; but the effect is not
established at the locked bar, and the 2016-2020 context shows NO transfer benefit (1.13 vs 1.19).
Direction CLOSED per kill-switch. Details: quantlab_output/r121_final_report.md.
