# R137 — FUNDING-RATE CARRY (THE INCOME-RELEVANT ROUND) — PRESPEC (LOCKED 2026-11-23, pre-run)
Origin: R136 zeitgeist scan disposition — the one archetype YouTube largely does NOT sell and this
lab uniquely holds the data for (32-coin × 3,378-epoch funding spine, refreshed 0 month-fails).
Operator context: the income target (~$50/mo) makes carry the most practically-relevant family in
the queue. THIS IS NOT A CHAMPION CANDIDATE — it is an income-infrastructure round.

## Mechanization (locked)
- Data: quantlab_cache/bnfut/{SYM}_fund.csv, 32 coins, 2023-09-01 → 2026-09-01, 8H epochs.
- Rule (per coin, per epoch): at settlement t the realized rate f_t is KNOWN. If f_t > θ, hold
  SHORT-perp + LONG-spot (delta-neutral) through the NEXT epoch and collect f_{t+1} (short receives
  when positive). Position persists while the condition re-fires; 0.10% notional toggle cost
  (two legs, taker+slip) charged on state changes only.
- This is the HONEST version: you predict the next epoch from the last one — the tradeable edge
  is funding persistence, adverse selection is included by construction.
- Variants θ ∈ {0.01%, 0.03%, 0.05%} per epoch. PRIMARY = 0.03%.
- Book: equal slots per coin; per-coin APR = net/span-years (calendar); book APR = mean.

## Gates
1. PRIMARY yearly net ≥ 0 for 2024, 2025, 2026 (2023 = warmup, report-only).
2. Book net APR (notional) ≥ 5%/yr — carry must beat the floor's boring yield to justify existing.
3. Sign-stability: θ=0.01% and θ=0.05% both net-positive.
4. Concentration: no single coin > 50% of book net.
5. Disclosures (mandatory in report): capital efficiency (spot+perp ≈ 2× capital → APR_capital =
   APR/2), basis drift risk, venue/exchange risk (same as floor), funding-interval drift.
Verdict: PASS 1–4 → carry is a real income candidate (next: sizing study, NOT promotion);
FAIL 1 → carry closed (persistence absent/insufficient). One run, no iteration.
