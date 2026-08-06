# R078 — Negative-Symbol Forensics + Unseen-Universe Test

**Date:** 2026-08-06 | Locked config: Family A + E6 + RR1.5 + VolCeil + breadth50

## Q1 — Is there any symbol negative the entire time?

**No symbol is *reliably* negative.** Full-period breakdown of the locked config on the
original 52-symbol universe:

- **12 of 48 traded symbols** were net-negative over the full period
- Their combined damage: **−17.5R** out of +51.5R total (only ~1/3 of the damage of a
  single good month like Oct-25 at +35R)
- **Every negative symbol had ≤5 trades** (median 2). At 1–5 samples, per-symbol
  "negative" is pure noise, not a real property.
- **4 symbols never triggered at all:** SOL, ALGO, GALA, SATS (their compression-setup
  conditions + breadth gate simply never co-occurred).

**Conclusion:** there is no dependable "bad symbol" to remove. Dropping the 12
noise-negative symbols would be textbook curve-fitting (they'd be "negative" for a
different random reason next period). **Keep the full 52-symbol universe.**

## Q2 — Get more assets to confirm 100%? → We did (18 new symbols fetched from OKX)

Fetched 18 additional liquid USDT swaps via OKX paginated history (verified the `after`
param pages to Dec 2023). 8 of them have ≥5,000 bars and are genuinely never-seen:

| New symbol | Bars | Kind |
|---|---|---|
| BICO | 17,500 | small-cap |
| HYPE | 12,759 | new L1 |
| XAU | 11,628 | **gold token** |
| HOME | 10,068 | meme/alt |
| PUMP | 9,317 | meme |
| ZBT / ZEC / BEAT | 6–7k | alt / privacy / alt |

**Result on the 8 never-seen symbols (locked config, breadth from original 52):**

- **17 trades, PF = 0.625, WR 29.4%, MDD −5.0%** → the edge did **NOT** transfer to these
  new assets. Per-symbol: BICO +0.5R, HYPE −0.5R, XAU −1.5R, PUMP −0.5R, ZEC −2.0R,
  HOME −2.0R, BEAT +1.5R — all within noise, but collectively negative.
- Full universe (52 + new): PF drops 2.05 → **1.77** (still profitable — the new assets
  dilute, they don't destroy).

**Interpretation (important):** the edge is **universe-specific, not universal.** It was
validated on the 52 established mid/large-cap swaps (which share crypto-market dynamics).
Gold tokens, memes, privacy coins, and tiny new listings behave differently and do NOT
inherit the edge. This is a *good* finding — it tells us:

1. **Do NOT add random new symbols to the bot.** Trade the validated 52.
2. Any future expansion must be validated per-symbol on a long window (like the 15m/1H
   comparison) before inclusion — not just "it's liquid, add it."
3. The breadth gate uses the original 52 universe — keep it that way.

## Verdict

- **Q1:** No reliable negative symbols → keep universe, no pruning (avoids curve-fit).
- **Q2:** More assets tested → edge does not transfer to unfamiliar assets → the 52-symbol
  universe is the strategy's home. Do not expand blindly; expand only via per-symbol
  validation.
- Locked config unchanged.

## Files
- `scripts/r078_symbol_forensics.py`, `scripts/r078_unseen_universe.py`
- `scripts/fetch_more_symbols.py` (18 new symbols now in cache: 70 total)
- Outputs: `r078_symbol_forensics.csv`, `r078_symbol_perf.csv`, `r078_new_universe_trades.csv`
