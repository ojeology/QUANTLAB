# EXPOSURE LEDGER — reserved-window reuse log (convention 6)

Holdout window **2026-07-01 → 2026-09-30** is CONTAMINATED for discovery (5 exposures).
It may serve only as a final, weakened gate, and never for selection.

| # | Round | Window used | Purpose | Contamination status |
|---|---|---|---|---|
| 1 | R115 | 2026-07→09 | holdout gate (funding dispersion L/S) | first use |
| 2 | R116 | 2026-07→09 | holdout gate (squeeze L/S) | contaminated |
| 3 | R117 | 2026-07→09 | holdout gate (DON55 BTC) | contaminated |
| 4 | R118 | 2026-07→09 | holdout gate (twin hunt) | contaminated |
| 5 | R119 | 2026-07→09 | holdout gate (corrected twin hunt) | contaminated |

Other windows: **2023-01→2023-12** examined ONCE as R120 discovery (metrics+prices+funding);
now seen — any future 2023 test of an OI-adjacent signal is contaminated (this is why the
kill-switch closed the direction instead of re-trying lookbacks). 2024–2026 OKX 1D/1H heavily used by R102–R114 (selection + validation);
2023 Binance daily metrics **untouched** (reserved for P1 OI-momentum discovery);
2020-01→2023-12 Binance dailies used as context by R117/R118/R119 only.

Rule: before any new experiment reads a window, append a row here. Charter RESULT sections must cite the ledger.
