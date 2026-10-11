# R099 — UNSEEN-UNIVERSE TRANSFER TEST: does the edge hold on brand-new pairs?

**Date:** 2026-10-09 · 27 liquid OKX USDT swaps **never used in any selection decision**
(top-30 by 24h volume outside the 73-symbol universe; CT/XDP/QUANT too young) ·
Champion pipeline applied exactly as deployed: Family A signals (causal rolling
thresholds), E6/RR1.5/SL1, breadth_q from the original 73, walk-forward SVM fit **only
on original-universe trades**, q0.55 threshold from original selection period.
**Nothing about the new pairs touched any fit, threshold, or rule.**

---

## Results

| Window | Subset | n | PF | PF@cost | WR |
|---|---|---|---|---|---|
| all (2025-07→) | raw (no filter) | 114 | 1.55 | **1.29** | 51% |
| all | champion filter | 83 | 1.54 | **1.26** | 51% |
| < 2026 | raw | 46 | 1.95 | 1.73 | 57% |
| 2026+ | raw | 68 | 1.33 | **1.06** | 47% |
| fresh (Aug8+) | raw | 15 | 2.25 | 1.73 | 60% |
| *reference: original 73, q0.55* | *2026* | *166* | *1.65* | ***1.34*** | *52%* |

**The crypto-only cut** (excluding tokenized stock swaps):
- all: n=96, PF@cost **1.36** · 2026+: n=50, PF@cost **1.09**
- Equity swaps (MSTR/TSLA/NVDA/QQQ/INTC/SOXS/CRCL): PF@cost **0.97 — no edge, exclude permanently** (session gaps break the 24/7 compression logic).

LOO floor (drop one symbol): **1.16** — no single pair carries it.
Bootstrap on 2026 kept trades: P5 0.68 · med 1.07 · P(PF>1) = **60%** — inconclusive.

## Verdict: ⚠️ THE SIGNAL TRANSFERS — THE CERTIFIED EDGE DOES NOT (yet)

Three clear findings:

1. **✅ The compression-pop phenomenon is real beyond the universe.** Raw PF@cost
   1.29–1.36 across ~100 unseen-pair trades. Compare: forex raw pools ≈ 1.00, and
   R078's scary 0.63 on 8 symbols. The economics (coil → release with participation)
   exist on new liquid crypto too.

2. **⚠️ The magnitude is weaker, and 2026 is below the bar.** Original universe 2026:
   1.34. New crypto pairs 2026: 1.06–1.09. Formally: n-gate and PF-gate both fail.
   This is exactly what "universe-specific edge" looks like — the *idea* travels, the
   *certified numbers* don't.

3. **❌ The ML filter adds nothing on new pairs** (1.26 vs 1.29 raw) — confirming
   R084: the model's selection quality is learned from the original universe's
   losers. On unseen pairs it's just a random-ish 55% keep.

## What this means for universe expansion

- **Do NOT migrate capital to the new pairs** on this evidence.
- **Do NOT add equity swaps, ever** (0.97).
- **DO open a monitoring sleeve:** the ~22 crypto pairs (STRK, MAGIC, WLD, ONDO, TIA,
  TRUMP, TAO, OKB, PENGU, RAY, ZRO, MINA, W, BAT, MUBARAK…) tracked with the raw
  signal at paper/minimum size. Graduation rule: a pair joins the traded universe
  after it accumulates ≥ 30 raw trades with PF@cost ≥ 1.20 (per-symbol validation,
  as R078 prescribed).
- The original 73 remain the only certified home for the champion.

*Files: `scripts/fetch_newpairs.py` · `quantlab_r099_newpairs.py` · `r099_stdout.log` ·
`r099_newpair_trades.csv` · `fetch_newpairs.log`*
