# R118 FINAL REPORT — THE TWIN HUNT: ENGINEERING THE TWO KNOWN PREMIA INTO LIVABLE FORM
**Verdict: both levers WORKED on their target — neither survives the charter. The "only the floor" law receives its final, strongest replication.**

---

## Premise (locked prespec: CONSISTENCY_HUNT_BEGINS.md)

Stop hunting new lotteries; engineer the two known real premia past their named failure modes:
- **H-T1:** momentum's pain is *concentration* (R113: +61% holdout, −75% DD) → lever: top-8 diversification + per-sleeve vol-targeting + BTC-filter.
- **H-T2:** daily trend's pain is *drawdown* (R117: −36% DD) → lever: softened exits (time-stop variant releases stuck trades).

## Results (all costs/funding in, charter gates, holdout 2026-07→09 untouched)

| Variant | APR | maxDD | worstW | posW | Yearly % | Gates | Holdout |
|---|---|---|---|---|---|---|---|
| H-T1 always-in top-8 VT | +10.8% | −24.7% | −6.5% | 50% | +38/−9/+0 | 2/6 | +11.7% ✓ |
| H-T1 BTC-filter VT | +10.0% | −20.7% | −5.7% | 29% | **+25/+3/+0** | 3/6 | +4.9% ✓ |
| H-T2 DON55 classic | +19.8% | −38.6% | −13.3% | 47% | +57/+21/−23 | 2/6 | −2.8% ✗ |
| **H-T2 time-stop** | **+27.7%** | −41.3% | −13.3% | 46% | **+50/+21/+5** | 3/6 | **+6.6% ✓** |
| Charter bar | — | ≤12% | ≥−3% | ≥65% | all ≥0 | **6/6** | ≥0 |

**What the engineering achieved — honestly:**
- **H-T1:** drawdown −75% → −21/−25%. The concentration diagnosis was correct and the fix is real — but chartered drawdown is ≤12%, and the premium's volatility is structural. Filter variant even rescued 2025 (+2.7%).
- **H-T2:** the time-stop rescued the losing year (2026: −23% → **+5.4%**) and the holdout flipped positive (+6.6%) — but the drawdown *deepened* (−41%): the risk didn't disappear, it moved from "losing years" to "losing quarters."
- Neither reached even 4/6. **Verdict per prespec: the hunt closes with the law's final replication — only the floor passes 6/6.**

## Audit trail — the round's real story

1. **The off-by-one rose from the dead — and the cross-check killed it.** First run printed DON55 at +66% APR / 2022 +59% (vs R117's +28% / −6.5%). Same rule, opposite worlds ⇒ data or bookkeeping crime. Systematic diff: `ht2` credited the breakout bar's own return to the fresh position — **the third appearance of this bug family in the repo** (R104 printer, R107 MTM, R118 state machine). Fixed per the lag law; numbers snapped back to R117's honest baseline (+19.8% APR, 2022 −11%). *Lesson now permanent: no state machine ships without a daily≡ledger assert, ever.*
2. Data spine verified clean under suspicion (0 duplicate dates, 1 real >25% move — COVID 2020-03-12; Aug–Sep 2026 +33% rally confirmed genuine).
3. **Softband variant VOID:** the prespec's exit wording (`close < max(SMA100, low20)`) is mathematically exit-always — degenerate by construction, disclosed as a design error, never run as a "result."
4. Cache wipe #8 absorbed pre-round; consolidated refetch (3,078 kline files + BTC funding) in-session.

## Standing law (now thrice-replicated at the engineering level)

> **Real premia can be tamed toward livability, but every taming trades one charter violation for another. Momentum's concentration → dilutable. Trend's drawdown → movable. Neither → removable. The floor's advantage is categorical, not engineered: it is paid by contract, so it has no concentration, no drawdown mode, and no exit to optimize.**

Remaining queues: OI-momentum (2023 virgin leg + 2026 holdout), FX residue round, champion recheck ~Nov 1. Cores unchanged.

## Artifacts

`quantlab_output/twin_ht1_filtered.csv` · `twin_ht2.csv` · `r118_stdout.log` · data refetched: `quantlab_cache/bnfut/`.
