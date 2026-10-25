# R123 — REVERSE TRANSFER (FX brain → crypto) — FINAL (2026-10-24)
Prespec: docs/RESEARCH_FRONTIER_R123.md (LOCKED pre-run). Train: 537 FX pops 2016-2024 (WR 41.7%),
SVC spec byte-identical to the champion filter, frozen to r123_fx_svm.pkl. Test: all 467 crypto
raw pops (73 perps, 2025-06→2026-10) — confirmatory-grade window (labels seen; model FX-only by
construction). One cell: keep top q=0.55 by score.

## RESULT: **REJECT — total non-transfer (information collapse)**
- The FX brain assigns essentially ONE probability (~0.42, below coin-flip) to every crypto pop:
  59 unique scores across 467 trades; all score deciles ≈ 0.42; the keep-55% rule degenerates
  (97% kept due to tie mass).
- Kept book PF@c 1.23 vs raw 1.26 → MUST-BEAT raw FAIL; random-q 1.26 → MUST-BEAT random FAIL.
  (G1-G4 technically passed on the degenerate near-raw book — the must-beats are what kill it,
  exactly as designed.)
- Forced-rank check: the model's top-47 crypto picks = PF@c 0.96 (worse than raw) — no hidden
  ranking signal even at the extreme.
- FX home self-test 2025+ (report-only): kept PF 0.80 vs raw 1.15 — the FX brain no longer works
  even on its OWN market post-2024 (consistent with F013 modern-era decay and R121's era fragility).

## THE TRANSFER PROGRAM — CLOSED, BOTH DIRECTIONS (with mechanism)
| direction | result | reading |
|---|---|---|
| crypto brain → FX (R121) | weak directional help (1.15 vs 0.96), NOT established at 95% | crypto pops carry real structure; some of it rhymes with FX |
| FX brain → crypto (R123) | zero information (uniform scores) | FX pops are near-noise (PF 0.96); a filter trained on noise learns noise — you cannot distill whiskey from water |

Asymmetric transfer is itself the finding: the "compression-pop" edge is **crypto-specific
volatility microstructure**, not generic candle anatomy. The champion's crypto-trained filter
remains the only filter in this lab with proven discriminative content.

## Verdict per prespec: REJECT — reverse transfer permanently mapped. Transfer program CLOSED.
Repro: `python3 quantlab_r123_reverse_transfer.py` (~25 s; needs pyarrow).
