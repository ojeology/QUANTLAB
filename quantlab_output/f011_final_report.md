# FOREX F011 — DIRECTION ENGINE INSIDE THE BIRTH WINDOW

**Date:** 2026-10-09 · 4H execution (F007 mirror: E6, SL 1·ATR, TP 3·ATR, 24-bar horizon,
TP-before-SL) · 8 majors, Oct-2024 → Oct-2026 · selection < 2026-04-01 ·
development-holdout ≥ 2026-04-01 (seen via F010 v1 — dev only) · configs prespecified
from F007 + F010 knowledge only, single-shot holdout.

**Question:** F010 proved big moves have a clock (birth window 06–12 UTC) and fuel
(traps), but no direction. F011 tested which context gives the trap-fade its direction:
**with-trend** (trap ends the pullback → continuation) vs **counter-trend** (F007's
trap-ends-the-trend → reversal), and whether the birth window improves entries.

---

## 1) WHY study — P(fade-direction big move) by cell (1H, selection)

| Cell | n | P(big6h) | %big in fade-dir |
|---|---|---|---|
| BEAR trap · UP-trend · window | 384 | 13.0% | **58%** |
| BEAR trap · UP-trend · off | 911 | 8.1% | 36% |
| BEAR trap · DN-trend · window | 447 | 16.6% | **57%** |
| BEAR trap · DN-trend · off | 1,114 | 9.8% | 47% |
| BULL trap · UP-trend · window | 445 | 14.8% | 44% |
| BULL trap · UP-trend · off | 1,052 | 9.2% | 41% |
| BULL trap · DN-trend · window | 335 | 17.3% | 43% |
| BULL trap · DN-trend · off | 840 | 10.7% | 51% |

**Read:** trend barely separates fade direction (bear-trap fade lands 57–58% in window
regardless of trend). The window raises expansion probability but does not orient it.
Note: in the window, big moves skew UP after *both* trap types — a sample-drift
artifact candidate, recorded as a curiosity, not a signal.

## 2) Factorial map — 4H fade PF@cost (selection)

Best cells: BULL/UP-trend/offWin **1.30** (n=322) and BEAR/DN-trend/offWin **1.13**
(n=345) — both *with-trend, OFF-window*. The birth window **hurt** every cell.
(Not promoted: 8 cells tested = multiple-comparison zone; they go to the notebook as
hypotheses for the 10-year dataset.)

## 3) Prespecified configs — single-shot holdout batteries

| Config | sel PF@cost | hol PF@cost | LOO floor | Verdict |
|---|---|---|---|---|
| CFG-A with-trend + birth window | 0.85 (n=331) | **0.66** (n=152) | 0.56 | ❌ FAIL |
| CFG-A2 with-trend, all hours | 0.84 (n=954) | **0.84** (n=372) | 0.77 | ❌ FAIL |
| CFG-B counter-trend (F007 mirror) | 0.88 (n=1,792) | **0.88** (n=602) | 0.83 | ❌ FAIL |

ECN costs still fail (best 0.95). MC: P(profit) ≤ 5% on all three. RR sensitivity on
the with-trend cell (selection): 0.81 / 0.84 / 0.93 at RR 2/3/4 — improves with RR but
never crosses 1.

---

## Verdict: ❌ REJECTED — cleanly, and that's informative

- **H1 (with-trend > counter-trend): REJECTED.** Trend context does not orient the
  fade (58% vs 57% in-window at 1H; both with-trend configs failed holdout hard).
- **H2 (birth window improves entries): REJECTED — inverted.** The window marks
  *expansion*, but trap-fade entries inside it pay worse fills (you're buying after
  the fuel started burning). Every window cell was worse than its off-window twin.
- **H3 (a config clears the gate): REJECTED.** 0 / 3 configs; nothing approaches
  holPF@cost > 1.10.

**Synthesis with F010:** conditions (squeeze, participation, traps, session clock)
reliably forecast *that* and *when* a big move comes — and trap-fade entries positioned
*inside* the expansion get run over. Direction remains unexplained by everything tested
so far except F007's narrow bear-trap/downtrend/London config (holPF@cost 1.06, its own
run) — and our 4-level bull+bear mirror scores lower (0.88), hinting F007's edge is
specifically the **bear-side** asymmetry, not a general trap effect.

## What the hunt should do next (F012 recommendation)

**Data before more mining.** Three consecutive iterations (F009→F011) have now hit the
same wall: 2 years of yfinance data cannot separate a 1.0–1.1 PF effect from noise
(holdout cells shrink to n≈150–600; selection cells to n≈300). The prespecified
with-trend/off-window cells (1.13–1.30 selection) are exactly the kind of hypothesis
that needs the protocol's original plan — **~10 years of Dukascopy 1H** — to be tested
with real power. Fetching that dataset is now the highest-value move; config mining on
2y data is panning the same sand twice.

*Files: `forex_f011.py` · `f011_stdout.log` · `f011_structural_traps_4h.csv` ·
`f011_mirror_traps_4h.csv`*
