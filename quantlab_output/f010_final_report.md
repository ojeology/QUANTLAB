# FOREX F010 — TRAP FORENSICS: why big moves happen + condition alignment

**Date:** 2026-10-09 · 1H data, 8 majors (Oct-2024 → Oct-2026) · selection < 2026-04-01,
development-holdout ≥ 2026-04-01 · big move = top-decile 6h move (2.83·ATR) attributed
to the session that **births** it · execution for trade tests = F007 mirror (4H, SL 1·ATR,
TP 3·ATR, E6, TP-before-SL).

---

## 1) WHEN — big moves have a birth hour (selection, n=72k bars)

| Session | P(big 6h) | lift |
|---|---|---|
| **LONDON (07–11)** | **17.3%** | **1.73x** |
| ASIA | 10.2% | 1.02x |
| NY_OVERLAP (12–16) | 8.8% | 0.88x |
| ROLLOVER (21–23) | 6.6% | 0.66x |
| LATE_US (17–20) | 4.4% | 0.44x |

Hourly profile peaks at **09:00–12:00 UTC** (1.78x → 1.97x → 1.93x → 1.63x), ramps from
06:00 (1.51x), collapses after 14:00 (0.42–0.36x in late US).

> **Big moves are born in the London morning → pre-NY handoff (09:00–12:00 UTC), not at
> the NY open.** The late-US afternoon is the quietest expansion window of the day.
> *(v1 audit note: with a 24h label this pattern is invisible — every session "contains"
> London. Label horizon is part of the physics.)*

## 2) WHAT — alignment predicts expansion, not direction (selection)

| Condition | P(big\|c) | lift |
|---|---|---|
| SQUEEZE (bb_rank ≤ 0.33) | 12.7% | **1.27x** |
| VOL SURGE (rel ≥ 1.3) | 12.4% | **1.24x** |
| ADX > 20 (trend active) | 9.3% | 0.93x |
| Near P20 / PD pool | ~10.1% | ~1.01x |

Confluence score (0→5) is **monotone**: 0.71x → 0.79x → 0.99x → 1.26x → 1.36x → **1.40x**.

> **The alignment forecasts VOLATILITY, not direction**: even under squeeze, big moves
> split 46% up / 54% down. There is no direction signal in the conditions themselves —
> the edge, if any, must come from the event+context (e.g., F007's trend+session), or
> from expressing it as a vol bet (straddle logic, breakout both ways).

## 3) WHY — trapped stops are the fuel (selection, n≈20k structural events)

| Event | n | P(big 6h) | lift |
|---|---|---|---|
| **BEAR_TRAP** (swept lows, close back above) | 2,119 | **11.0%** | **1.10x** |
| **BULL_TRAP** (swept highs, close back below) | 1,987 | **11.4%** | **1.14x** |
| BRK_UP (clean break up) | 9,045 | 8.1% | 0.81x |
| BRK_DN (clean break down) | 8,026 | 8.4% | 0.84x |

> **H3 confirmed: sweep-and-fail events precede bigger forward moves than clean breaks
> of the same levels.** Mechanism: a trap forces stop-outs + failed-breakout exits —
> fuel for the next expansion wave.
> **But the fuel burns both ways:** after a BEAR_TRAP only 46% of subsequent big moves
> land in the fade (up) direction; after BULL_TRAP 48%. Traps raise the *size* of the
> next move far more reliably than its *direction*.

## 4) TRADE — naive fade of aligned structural traps FAILS (4H battery, retail costs)

| Config | sel PF@cost | hol PF@cost | LOO floor | Gate |
|---|---|---|---|---|
| All structural traps, fade | 0.92 | 0.90 (n=661) | 0.83 | ❌ |
| conf ≥ 3, fade | 0.85 | 0.87 (n=471) | 0.82 | ❌ |
| conf ≥ 4, fade | 0.80 | 0.91 (n=215) | 0.86 | ❌ |

ECN costs still fail (best 0.98–0.99). Swap makes it worse. **H4 rejected** — the
forensics explain why: fade direction is a coin flip (§3), and RR3 asymmetric payoff
cannot survive a 50% directional edge.

---

## Verdict: ⚠️ PARTIAL — the science landed, the trade didn't

- ✅ **H1 (WHEN)** — expansion has a clock: 09:00–12:00 UTC birth window, ~2x base.
- ✅ **H2 (WHAT)** — confluence is monotone for expansion probability (0.71x → 1.40x);
  squeeze + participation are the strongest single lifts.
- ✅ **H3 (WHY)** — traps fuel bigger moves than clean breaks (1.10–1.14x vs 0.81–0.84x).
- ❌ **H4 (TRADE)** — naive trap-fade is not tradable: direction after traps ≈ 50/50,
  holPF@cost 0.87–0.91, every gate failed.

**The F010 lesson for the hunt:** what aligns before big moves forecasts *volatility*
and *timing*, not *direction*. The tradable expression needs a direction engine on top
— exactly what F007's downtrend+London context supplies (its hour-8 filter sits inside
the 09:00–12:00 birth window found here). F007's remaining edge (holPF@cost 1.06) is
consistent with this map.

## Causal audit & disclosures

- v1 artifacts (documented, fixed): 24h label swallowed sessions; VWAP/EMA20 "levels"
  generated ~57k pseudo-events (always within 0.15·ATR of price); 1H execution paid ~3x
  cost per R. v1's two naive configs were holdout-peeked (holPF@cost 0.73/0.75, failed).
- v2 therefore treats Apr→Oct-2026 as a **development** holdout: any config passing here
  is exploratory until fresh data arrives (Dukascopy 10y fetch is still the unlock).
- LATE_US 4H holdout slice showed PF@cost 1.34 (n=52) — too small to matter, not pursued
  (no cherry-picking small cells).

## Next steps (F011 candidates)

1. **Direction engine on the birth window** — F007-style trend/level context *restricted
   to 06:00–12:00 UTC* traps, tested against the 09:00–12:00 edge found here.
2. **Vol-expression** — trade the expansion itself (dual breakout brackets with cost
   gate) instead of predicting direction.
3. **Dukascopy 10y fetch** — the protocol's real data unlock; every F-iteration gets
   sharper with 15 years instead of 2.

*Files: `forex_f010.py` · `f010_stdout.log` · `f010_alignment_map.csv` ·
`f010_trap_events_1h.csv` · `f010_trap_events_4h.csv`*


---
> **⚠️ CORRECTION (F012, 2026-10-09):** the yfinance 1H FX feed used here is labeled
> **+4h off true UTC** (vendor lag test: corr 0.9999 vs HistData at +4h). All clock
> times in this report are shifted: the "09:00–12:00 birth window" is truly
> **05:00–08:00 UTC**, and the true replicated birth clock on 10-year HistData is
> **02:00–06:00 UTC** (Tokyo morning). Session labels (LONDON/NY) in this run are
> likewise misnamed. See `f012_final_report.md`.
