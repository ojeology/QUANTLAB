# R097 — CHAMPION UPGRADES, HEAD-TO-HEAD: best safe config

**Date:** 2026-10-09 · Base: R096 champion (Family A + E6 + RR1.5 + breadth50 + walk-forward
SVM q0.55, 73 OKX swaps, data through Oct 9) · All variants are cost/portfolio/execution
changes — no signal mining · Deltas measured on 2026 + fresh (Aug8–Oct9) windows ·
Run-to-run note: SVM Platt-scaling wobbles n by ±8 and PF@c by ±0.02 between identical
runs — comparisons within a run are exact.

---

## The grid (q0.55 signal in all rows)

| Config | 2026 n | PF@cost | fresh PF | MDD | Verdict |
|---|---|---|---|---|---|
| C0 base (taker .05/.05) | ~170 | 1.25 | 2.04 | −21% | reference |
| **C1 VIP taker (.04/.04)** | ~170 | **1.30** | **2.12** | −21% | ✅ **best PF — needs fee tier** |
| C2 maker entry (.02 in, .05 out) | ~165 | 1.23 | 1.95 | −22% | ⚠️ neutral — adverse selection |
| **C3 maker-in + maker-TP (.02/.02)** | ~165 | **1.27** | 2.00 | −22% | ✅ **best at standard fees** |
| C4 base + cluster caps | ~125 | 0.97 | 1.20 | **−14%** | ❌ PF −22% for modest DD gain |
| C5–C7 (+ maker, + vol-sizing) | ~120 | 0.92–0.97 | 1.07–1.47 | −16% | ❌ caps poison everything they touch |
| q0.65 + caps | ~140 | 0.88 | 0.93 | −18% | ❌ |

Monte Carlo on C1 (2026 net Rs): **P(profit) = 96%**, median +16% per 9 months at 0.5%
risk/trade (+33% at 1%), median max-DD −5% (−10% at 1% risk).

## What won, what lost, and why

1. **FEES WON.** The single best "improvement" is paying less per trade. C1 (VIP tier)
   adds ~+0.05 PF; **C3 works at standard exchange pricing** (maker 0.02% entry via
   limit-at-close, maker 0.02% TP via limit, 0.05% taker stop) and keeps 95% of trades
   filled. No code, no signal change — pure plumbing.
2. **Cluster caps LOST.** Crypto is one risk block: the greedy correlation clustering
   merged **53 of 73 symbols into a single clique** (ρ>0.6). Capping it just amputated
   30% of trades (mostly winners) for a modest DD gain. REJECTED.
3. **Vol-scaled sizing: no effect.** It scales all trades roughly equally in this
   universe. REJECTED (keep flat 1%).
4. **Weekly SVM refit: rejected on fidelity.** Prediction corr vs per-trade refit is
   only 0.607 with 77.5% keep-set agreement — the speedup isn't worth changing the
   trades. Per-trade refit takes 23 seconds; keep it.
5. **Universe expansion: skipped by design** (R078: edge is universe-specific; nothing
   validated to add).

## ⚠️ The disclosure that matters most

Under the adversarial SL-first path convention, **every config scores PF ≈ 0.00** —
because with RR1.5/SL1 and no time-stop, every winner is a TP exit, and flipping all
TP-wins to losses leaves nothing. Reality sits between 0.00 and ~1.3 depending on how
often the 1H bar touches TP before SL. **This is the same intra-bar ambiguity that killed
the forex strategies (F013) — it caps the certainty of the champion too.** The champion's
edge has now survived holdout, freshness, and cost audits; the one audit it hasn't had
is **path resolution at 1-minute data** (~170 trades to check on OKX 1m candles).

## Final answer — best config

> **R096 champion, unchanged signal, executed at the lowest fees you can get.**
> Tier A: VIP taker 0.04% both sides → PF@c ≈ 1.30, fresh 2.12.
> Tier B (no fee tier needed): maker limit entries + maker TP + taker stop → PF@c ≈ 1.27.
> Risk 1%/trade, isolated margin 3–5×, kill-switch: rolling 3-mo PF@cost < 1.0 → halt.
> **Before scaling real money: run the 1-minute path audit.** It is the difference
> between "validated except path" and "validated."

*Files: `quantlab_r097.py` · `r097_stdout.log` · `r097_best_trades.csv`*
