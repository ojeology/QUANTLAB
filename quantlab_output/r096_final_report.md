# R096 — YEAR-END RE-CHECK: is the crypto 1H edge worth pursuing in 2027?

**Date:** 2026-10-09 · Charter: FOREX_HUNT_BEGINS.md — *"Re-run at end of 2026 with fresh
data → decide."* · Code: exact R087 pipeline (73 OKX swaps, Family A compression-pop,
E6 entry, RR 1.5, walk-forward SVM keep-rate filter, breadth overlay) · Data refreshed
2026-08-07 → **2026-10-09 21:00 UTC** (`scripts/refresh_okx.py`) ·
**FRESH window = Aug 8 → Oct 9** — unseen by every selection decision in hunt history.

---

## Headline: ✅ THE EDGE IS ALIVE — pursue in 2027, ship q0.55

| Config | 2026 holdout (Jan→Oct) | **FRESH window (Aug 8→Oct 9)** | P(PF>1) | LOO floor |
|---|---|---|---|---|
| **q0.55 (locked champion)** | n=165 · PF@cost **1.30** | n=31 · PF@cost **1.99** (CI P5 1.15) | **98%** | 1.78 |
| q0.75 (Aug-8 champion) | n=199 · PF@cost **1.14** | n=48 · PF@cost **1.44** (P5 0.88) | 90% | 1.32 |
| q0.65 | n=176 · PF@cost ~1.5* | n=43 · PF@cost 1.53 | — | — |
| raw (no ML filter) | n=283 · PF@cost ~1.3* | n=49 · PF@cost 1.38 | — | — |

*full-period values from the R096 sweep table; holdout column = recomputed exactly from
trade logs at 0.05%/side. Fresh-window LOO = PF@cost after dropping each single symbol.

**The raw signal is positive even with NO ML filter (1.38)** — the compression-pop
phenomenon itself is working in the current regime; the SVM filter adds selection
quality on top (1.38 → 1.99 at q0.55).

## The honest caveats

1. **The decay trend is real.** q0.75's 2026 PF@cost: 1.62 at lock (Aug) → ≈1.25
   (branch re-check) → **1.14** now. It hovers at the minimum bar (1.10). The strategy
   is not dying, but the direction of travel is down — consistent with edge maturation.
2. **September carried the fresh window** (q0.75: Aug −2.4R, Sep **+17.4R**, Oct-partial
   −4.3R; q0.55: Aug +2.2R, Sep +13.0R). One strong month, n=31–48 — encouraging, not
   conclusive. The 98% bootstrap on q0.55 is the strongest single number, but it's
   bootstrap on 31 trades.
3. **Monthly profitability softened:** 71% → 59% prof-months (q0.75, full period).
   Expect losing weeks; the documented worst losing streak is still only 2 months.
4. FET had an exchange halt inside the window (894h gap) — engine skipped it correctly.

## 2027 operating plan (what "pursue" concretely means)

- **Primary: q0.55** (~15-20 t/mo across 73 symbols, PF@cost 1.30 on 2026, freshest
  window 1.99). **Alternate: q0.65** if more frequency is needed.
- **Cost discipline:** all numbers assume 0.05%/side taker. Real edge requires ≤0.06%
  all-in per side — fee tier or maker entries are part of the strategy, not an option.
- **Kill-switch (pre-committed):** if rolling 3-month PF@cost < 1.0 or two consecutive
  negative months on q0.55 → halt and re-validate. (Charter's own cost-gate logic.)
- **Monthly re-check cadence:** re-run `quantlab_r096_recheck.py` after each
  `python3 scripts/refresh_okx.py` — 5 minutes, same batteries.
- **Not pursued:** R077 static config (0.75 @cost — confirmed dead), 5m (dead), forex
  (dead per F009–F013). This 1H crypto config is the only validated edge in the repo.

## Audit trail

- R096 sweep table had a mislabeled column (`hol@c` showed full-period PF@cost);
  the correct holdout/fresh values in this report are recomputed from
  `r096_trades_q*.csv` — raw trade logs, verifiable.
- Data seam: OKX `/market/candles` depth limited the refresh to ~Aug 7 21:00; the
  5-hour stub Aug 7 16:00→21:00 is missing (immaterial). Fresh window starts Aug 8.

**Decision: PURSUE IN 2027.** The one edge that survived every attempt to kill it —
now including two more months nobody had seen.

*Files: `quantlab_r096_recheck.py` · `r096_stdout.log` · `r096_trades_q*.csv` ·
`scripts/refresh_okx.py` · `refresh_okx.log`*
