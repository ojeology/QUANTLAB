# FOREX F013 — REALISTIC EDGE: the honest end of the rope

**Date:** 2026-10-09 · 23 years (2003→2026) HistData M1→1H/1D, 8 majors, true UTC ·
all candidates prespecified · gates: holPF@cost > 1.10, LOO > 1.0, n ≥ 200 (M2: 50).

---

## Results — everything failed, and the failures are informative

### M1 — Daily time-series momentum (weekly legs, costs ≈ 0.02R)

| Variant | sel PF@cost (2003-21) | hol PF@cost (2022-26) | Verdict |
|---|---|---|---|
| TSMOM 20d | 0.92 | 0.90 | ❌ |
| TSMOM 60d (classic) | 0.95 | **0.58** | ❌ |
| TSMOM 120d | 1.03 | 0.87 | ❌ |

The most documented FX premium is **dead in the current era** — consistent with the
post-2010 decay reported in the literature. Not a implementation problem: costs are
negligible here; the signal itself just stopped paying.

### M2 — EMA50/200 crossover

Selection PF@cost 2.26 (n=167 over 19y) but only **8 trades in 4.7 holdout years** —
unvalidatable at pair level. Not promoted (that's how F007-style mirages are born).

### F1 — Fade of the big Tokyo-morning move & F2 — Tokyo-range breakout

| Test | TP-first PF@cost | SL-first PF@cost | Direction-only |
|---|---|---|---|
| F1 fade (hol) | 2.65 (WR 79%) | **0.20** | — |
| F2 breakout (hol) | 1.57 ✅ | **0.15** | **0.83** |

**Both "edges" were intra-bar path artifacts.** With 1H bars, ~1-ATR brackets, the
TP-before-SL convention decides everything: when a single bar spans both levels we
credited the win. Under the pessimistic path both are losers, and F2's direction-only
test (0.83) shows there is no directional signal at all — the bracket asymmetry plus
optimistic pathing manufactured the entire result.

**The deeper lesson: 1H-bar bracket backtests in FX are *unresolvable* — the data
cannot tell you whether TP or SL was hit first. Anyone showing you a great 1H bracket
backtest (including our own F007/F010 runs) is showing you a path assumption.**

---

## Verdict: ❌ NO VALIDATED FX EDGE — and that is the realistic answer

Five iterations (F009→F013), 23 years, ~150k events, honest costs, adversarial path
checks. What we can now say with unusual confidence:

1. **Intraday FX direction at retail cost: no edge found.** Every candidate was an
   artifact (vendor label shift, small samples, or intra-bar pathing).
2. **Daily FX momentum: dead post-2021.** The textbook premium didn't survive holdout.
3. **What IS real:** the expansion clock (big moves born 02:00–06:00 UTC, ~2× lift,
   replicated across a decade) — but timing without direction is not tradeable in
   spot FX at retail cost.
4. **The wall is structural:** ~1 pip retail spread ≈ 0.10R at 1H scale. An edge must
   clear that before it can even be measured. Most retail FX "systems" live below it.

## Where a realistic edge actually exists from here

1. **In this repo, already validated: the crypto 1H SVM strategy** — PF 1.62 @cost,
   holdout-validated on 73 symbols, locked config, due its year-end re-check with
   fresh data. It is the only config in QUANTLAB that has survived the full battery.
   Re-running it on Oct-2026 data is the highest-probability path to a live edge.
2. **Resolve the FX intra-bar question with M1 data** — the HistData pipeline is
   built and cached; a 1-minute-resolution simulator can genuinely settle whether
   Tokyo-morning brackets have path-robust edge (and at ECN vs retail costs).
3. **If staying with FX: cut the cost, not the strategy.** At ECN (0.2-0.4 pip) the
   cost wall drops ~3×; that changes which families are even testable.

*"The goal isn't to find a strategy that looks good. The goal is to find one that
survives every attempt to prove it wrong."* — the hunt's own charter. Five forex
iterations died surviving it. That's the system working, not failing.

*Files: `forex_f013.py` · `f013_stdout.log` · `f013_f1_fade.csv` · `f013_f2_breakout.csv`*
