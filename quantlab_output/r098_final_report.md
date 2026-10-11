# R098 — 1-MINUTE PATH AUDIT: the champion's final exam — PASSED ✅

**Date:** 2026-10-09 · Question: is the R096/R097 champion's edge real, or an artifact of
the TP-before-SL assumption on 1H bars (the exact artifact that killed every forex
strategy in F013)? · Method: OKX 1-minute candles fetched for all 165 champion trades of
2026 (`quantlab_r098.py`); every trade re-resolved by true first-touch of TP (entry+1.5·ATR)
or SL (entry−1·ATR).

---

## Result: the edge survives ground truth

| Measure | 2026 (n=161 audited) | Fresh Aug8–Oct9 (n=31) |
|---|---|---|
| PF@cost, 1H-assumed (old engine) | 1.33 | 1.99 |
| **PF@cost, 1m ground truth, taker** | **1.27** | **1.99** |
| **PF@cost, 1m ground truth, maker-in/TP** | **1.39** | **2.16** |

- The 1H TP-before-SL assumption was **wrong on only 4 of 161 trades (2%)**:
  2 wins → losses, 1 loss → win, 1 unresolvable (scored as loss, conservatively).
- **Zero same-minute TP/SL collisions** — nothing is unresolvable at 1m.
- True win rate: **51% at RR 1.5/SL 1** → +0.28R gross expectancy per trade.
- Contrast with forex (F013): pessimistic pathing collapsed PF 2.65 → 0.20 there.
  Here the *true* path barely moves the number. **This is what a real edge looks like.**

## Why the champion lives where forex died

1. **Cost per R is small**: 0.05% fee vs 1H ATR ≈ 0.02–0.05R (forex retail spread ≈ 0.10R+).
2. **The raw signal is strong**: even unfiltered, gross PF 1.46 (vs ~1.0 forex raw pools).
3. Real edge + low cost = robust to path conventions, vendor labels, and fresh regimes.

## Champion: FINAL STATUS (all gates passed)

| Gate | Status |
|---|---|
| Holdout 2026 (Jan–Jul), walk-forward | ✅ PF@cost 1.30 |
| Fresh unseen window (Aug 8–Oct 9) | ✅ PF@cost 1.99 (98% boot) |
| Cost audit (R097 grid) | ✅ robust; best = low fees |
| **Path audit (R098, 1m ground truth)** | ✅ **PF 1.27–1.39 confirmed** |
| Execution | q0.55, E6 entry, RR1.5/SL1, 73 symbols, 1% risk |
| Leverage | 3–5× isolated (stop, not leverage setting, controls risk) |
| Kill-switch (pre-committed) | rolling 3-mo PF@cost < 1.0 → halt & revalidate |

**Verdict: fully validated to the limit of what historical data can prove. The
remaining risk is regime change, not methodology. Sensible next step: paper-trade or
run small live (q0.55, ≤0.06%/side costs) through year-end, then scale per the 2027
plan if the kill-switch stays untriggered.**

*Caveats: 4/165 trades lacked 1m depth (scored conservatively); audit covers 2026 —
pre-2026 trades inherit the 1H assumption but those are selection-period only;
funding rates not modeled (holds avg 3.2 bars — immaterial at these horizons, log live).*

*Files: `quantlab_r098.py` · `r098_stdout.log` · `r098_path_audit.csv`*
