# R135 — TRENDLINE BREAK, MECHANIZED — PRESPEC (LOCKED 2026-11-05, pre-run)
Purpose: independent replication of the operator's own published channel claim (Bryme QuantLab,
2026-11-02: "I Tested the Trendline Breakout — It Can't Beat a Coin Flip") under the QUANTLAB
charter — the one strategy family absent from this lab's ledger. Diagonal lines are fit to pivots,
so unlike fixed channels they adapt; that adaptation is the hypothesis being tested.

## Mechanization (the only honest way to test a drawn line — locked)
- Universe: 20 stack coins (R106), OKX dailies 2023-06-01→now. Ledger years: 2024, 2025, 2026.
- Pivot low: bar whose low is the min of ±w bars (confirmed w bars later; usable only after).
- Uptrend support line: LAST TWO pivot lows with p2.low > p1.low (rising), both within trailing
  120 days. Line through them, projected forward. SHORT on FRESH cross: close[t] < line(t) AND
  close[t-1] ≥ line(t-1). Entry at close[t] (close-only law).
- Mirror: downtrend resistance line from last two FALLING pivot highs; LONG on fresh cross above.
- Exit: close re-crosses the (frozen) line, or 10-daily time stop, whichever first.
- Costs: 0.10% round-trip, charged in R as champion convention (rc = r − 2·0.0005·entry/atr).
- One open trade per coin; signals while in-trade are skipped (no mining).

## Variants (2, predeclared): w = 3, w = 5. PRIMARY = w5. No others may be computed.

## Gates (cell-level, not sniper)
1. n ≥ 30 trades (primary variant) — else "untestable at this frequency", reported as such.
2. PF@c ≥ 1.15 pooled.
3. Survival law: 2024, 2025, 2026 each ≥ 0R net.
4. Split: first-half events vs second-half events — both PF@c ≥ 1.0 required to call it real.

## Coin-flip benchmark (the operator's own claim, quantified)
Win-rate ≈ 50% with PF ≈ 1.0 = "can't beat a coin flip" CONFIRMED. WR materially > 52% at PF@c ≥ 1.15
= the drawn line carries information beyond noise.
One run, no iteration, no re-binning after results.
