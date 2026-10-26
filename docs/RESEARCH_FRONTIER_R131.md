# R131 — SHOCK-CONTINUATION (the R130 mechanism, strategy form) — PRESPEC (LOCKED 2026-10-26)
Hypothesis from R130: big moves come FROM big moves (expansion→expansion, lift 1.58–1.68; trigger
bars 1.23/1.48; follow-through after big UP events +0.69% fwd24h / +1.12% fwd72h vs +0.10% drift).
Strategy form: trade the CONTINUATION of qualified shock bars. GRADE: challenger — all R130 lifts
are seen data; parameters are INHERITED (mechanism's own measured quantities), never swept.
Promotion path: forward paper at the Jan-1 checkpoint only (challenger #4 slot if it earns it).

## Registered variants (long after UP-shock — primary; short after DOWN-shock — reported)
Trigger (locked, mechanism-inherited): |1h ret| ≥ 2×σ(720h) AND |1h ret| ≥ 1.0% AND rel_vol ≥ 1.5
(bar range vs 20-bar mean range) AND bar direction defines side. Entry at bar close (E6).
- **V1 (PRIMARY):** exit at close +24 bars (the atlas window).
- **V2:** exit at close +72 bars (the atlas window).
- **V3:** 24h exit + SL at entry − 1×ATR(14), SL-first on ambiguous bars (convention #5).
One position per coin; overlapping coins allowed; no coil requirement (that is the point — the
champion owns the coil cell; this owns the expansion cell). Costs 0.07%/side (round trip 0.14%,
in R = 0.0014×entry/ATR). σ = trailing 720h hourly std (the atlas definition).

## Gates (on the SEEN book; consistency across the atlas halves required)
- G1: PF@c ≥ 1.30 on DISC half (2024-02→2025-09)
- G2: PF@c ≥ 1.20 on CONF half (2025-10→2026-10) — no half may be negative
- G3: n ≥ 200 (full book)
- G4: max drawdown at 1%-risk-equivalent ≤ 20% (book level; charter-grade DD is only demanded at monetization)
All four (any variant) → PROMISING-CANDIDATE (challenger #4, forward-paper gauntlet).
Any fail → the expansion cell is mapped-and-closed: R130's lifts were real but not tradeable net.

## Report-only (decides nothing)
Short-side book (mirror variants); monthly R table; overlap with champion entries (how orthogonal
is this book to the benched book); per-coin concentration (LOAO min PF).

## Verdict vocabulary
PROMISING-CANDIDATE / CLOSED. One run. No re-parameterization after results — the variants above
are the complete registered set (3 long, 3 short reported).
