# FOREX F012 — TEN-YEAR VALIDATION: does anything survive real data?

**Date:** 2026-10-09 · Data: HistData ASCII M1 → 1H, UTC, 8 majors, 2016-01-03 → 2026-10-02
(~66k bars/pair; `vol` = bar-range proxy — HistData ASCII volume is 0) · vendor sanity:
weekly-close corr vs yfinance overlap = 0.9964 · Candidates inherited from F007/F010/F011,
all prespecified w.r.t. the virgin window.

**Design (contamination-aware):** *Virgin window* = 2016-2023 (8 years — no hunt config
ever used pre-Oct-2024 data). *Recent window* = 2024-2026 (partially contaminated by
F007/F010/F011 selection; used only for survivors). Nothing survived, so the recent
battery was never needed.

---

## Finding 1 — The yfinance 1H FX feed is labeled +4h off true UTC ⚠️ (methodological)

Hourly-close correlation, EURUSD overlap Oct24→Mar26: **0.9999 at lag +4h** (clear
single peak). yfinance bar "08:00" = true UTC 04:00–08:00.

Consequences for the hunt's history:
- **F010's "09:00–12:00 birth window" was actually 05:00–08:00 UTC.** F012's
  true-labeled study puts the big-move birth clock at **02:00–06:00 UTC** (Tokyo
  morning ramp), stable across BOTH eras (top-5 virgin hours replicate 5/5 in the
  recent era). The pattern was real; the clock labels were not.
- **F007's "London open, hr==8" was actually the 04:00 bar** — late-Asia/Tokyo
  morning, not London. The trading narrative was an artifact of the mislabeled feed.
- **Action: quarantine `quantlab_cache/forex/` (yfinance) from all future intraday
  FX work.** HistData/Dukascopy UTC data only.

## Finding 2 — Big moves ARE born in a clock (replicated): 02:00–06:00 UTC

| hour (true UTC) | lift 2016-23 | lift 2024-26 |
|---|---|---|
| 02:00 | 2.04x | 1.62x |
| 03:00 | 1.89x | 1.55x |
| 04:00 | 2.09x | 1.82x |
| 05:00 | 2.01x | 1.89x |
| 06:00 | 1.94x | 1.92x |
| 10:00–13:00 | 0.30–0.51x | 0.30–0.44x |

F010's qualitative conclusion stands on corrected labels: expansion has a clock.
The quietest true-UTC hours are 12:00–15:00 (not "late US" as mislabeled before).

## Finding 3 — Every inherited candidate dies on 8 virgin years ❌

| Candidate (inherited) | n (2016-23) | PF gross | PF@cost | WR |
|---|---|---|---|---|
| C0 all-trap fade (4lvl, all hrs) | 105,577 | 1.02 | 0.92 | 27% |
| **C1 F007 bear-only DN-trend hr8** | 6,517 | **1.08** | **0.98** | 29% |
| C2 C1 + bull mirror | 12,831 | 1.01 | 0.92 | 28% |
| C3 with-trend structural fade | 5,595 | 1.02 | 0.92 | 27% |
| C4 BEAR/DN structural off-window | 1,456 | 1.02 | 0.92 | 27% |
| C5 BULL/UP structural off-window | 1,398 | 1.05 | 0.95 | 27% |

Gate for the recent era was PF@cost > 1.05 & n ≥ 500 on virgin data: **0/6 pass.**

F007's config deserves the autopsy: gross PF 1.08 over 6,517 virgin trades (WR 29% vs
25% breakeven at RR3) — a faint real signal, worth ~+0.03R/trade gross, fully eaten by
retail spreads (~0.10R/trade). Hour sweep of the config on virgin data: best hour
PF@cost = 1.00 — **no hour works**. (The recent-era hr20 cell at 1.14 has a 0.94
virgin counterpart — noise, not pursued.)

---

## Verdict: ❌ NO SURVIVORS — the trap-fade family is dead on 10 years of true-UTC data

**What the hunt now knows (corrected ledger):**
1. Big-move *timing* is genuinely predictable: 02:00–06:00 UTC birth clock, ~2× base
   rate, stable across a decade. F010's finding survives with corrected labels.
2. Big-move *magnitude* is modestly predictable: traps > clean breaks, squeeze +
   participation lift expansion probability (F010/F012, monotone confluence).
3. Big-move *direction* remains unpredictable by everything tested (trend context,
   trap side, session, hour) — F007's apparent edge was mislabeled-feed artifacts +
   small-sample noise on 2024-25 data.
4. Vol-timing without direction is not tradable in spot FX at retail costs with
   SL/TP-bracket execution — 10 years, 100k+ events, every config < 1.0 @cost.

**Where the hunt can still go (F013 options):**
- **Trade the clock, not the direction** — e.g. Tokyo-morning range compression →
  breakout BOTH ways (bracket entries, small RR) exploiting the 02-06 clock; costs
  per trade are the enemy; needs ECN-grade cost assumptions to live.
- **Change instrument** — the same vol-clock maps to JPY crosses' optionality or
  index/energy futures where expansion pays asymmetrically.
- **Change horizon** — daily bars, 20+ years available; different cost/edge ratio
  (the crypto track's validated 1H edge came from a similar coarser-grain regime).

*Files: `forex_f012.py` · `scripts/fetch_forex_histdata.py` · `f012_stdout.log` ·
`f012_traps_mirror_4h.csv` · `f012_traps_structural_4h.csv` ·
`f010_hour_lift_10y.csv` · raw zips in /tmp/hdzips (not persisted)*
