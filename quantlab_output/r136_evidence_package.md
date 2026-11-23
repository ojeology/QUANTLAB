# R136 SUPERTREND — RESEARCH EVIDENCE PACKAGE
**Prepared 2026-11-23 for RealMarketAPI article development · audit-grade · nothing modified, nothing fabricated**

---

## 0. CHAIN OF CUSTODY (project state, verified this session)

| Item | Verified value | Method |
|---|---|---|
| Latest **pushed** commit | `103dc19` — "R136 YouTube zeitgeist: Supertrend CLOSED…" | unauthenticated `git ls-remote https://github.com/ojeology/QUANTLAB.git refs/heads/main` → `103dc19e54efcb9d4a6e5b73b46de9302d0254b0` |
| Later commits on remote | **none** — R136 is the latest pushed state; no correction, re-run, or audit has modified it | `git log 103dc19..origin/main` empty |
| Branch | `main` (remote). Local workspace git metadata was wiped by the session snapshot (recurring, documented condition) and the push token expired — **local files verified by content instead**: all five R136 artifacts are **byte-identical** to the `103dc19` blobs (diff vs fresh clone) | `diff` local ↔ clone of pushed commit |
| R136 artifact set | `docs/RESEARCH_FRONTIER_R136.md` (22 lines, prespec) · `quantlab_r136_zeitgeist.py` (99 lines, engine) · `quantlab_output/r136_final_report.md` (48 lines) · `quantlab_output/r136_supertrend_book.csv` (283 lines = header + 282 trades) · `quantlab_output/r136_ichimoku_book.csv` | file inventory |
| Post-R136 work (unpushed local content only) | R137 funding-carry round, champion retest, R135 paper tracker — **no file overlapping R136**; content identity check above proves R136 untouched | artifact-set comparison |

**Verification status legend used below:** [R] = independently recalculated this session from raw trade data; [X] = copied from a repo report after cross-checking; [D] = documented design fact (prespec/code), not a measurement.

---

## A. EXECUTIVE RESEARCH SUMMARY

QuantLab mechanized the canonical YouTube "Supertrend" strategy (period 10, multiplier 3.0, long-only) under a locked pre-specification and ran it over 20 liquid crypto perp dailies, 2023-06 → 2026-11-05, with 0.10% round-trip costs charged in risk units. The result is the cleanest illustration in the lab's ledger of why a profit factor above 2 is not an edge:

- The pooled book is spectacular: **282 trades, PF@c 2.04, +448.5R** [R].
- The profit is **78.5% attributable to 8 trades** (2.8% of the sample), 7 of which exited in 2024 [R].
- The yearly survival law (2024, 2025, 2026 each ≥ 0R — the lab's stability gate) **failed on 2026: −12.5R, PF 0.90** [R]; and 2025's nominally positive +10.7R contains the single largest jackpot (XRP +76.6R), i.e. ex-jackpot 2025 ≈ **−65.9R** [R].
- A year-block bootstrap over total return spans zero: **95% CI [−74.5, +1412.7]**, P(total-R ≤ 0) ≈ 27.8% [R].
- Under the lab's pre-declared gates the experiment was **CLOSED — not promoted, not claimed profitable, not claimed worthless**: it failed to *demonstrate reliable* profitability. The measured facts support a sharper sentence for the article: *the indicator rides real trends; the profit is a trend-regime lottery ticket; the typical trade bleeds.*

---

## B. METHODOLOGY RECONSTRUCTION (fully reproducible)

**Origin & pre-registration.** Operator directive to scan current YouTube rotation for testable strategies (R136 final report lines 1–6). Two cells (Supertrend, Ichimoku) were declared **before running**, with the rule "NO other parameters may be computed" and "One run, no iteration, no re-binning" (`docs/RESEARCH_FRONTIER_R136.md:8,22`).

**Market & instruments.** Crypto perpetual-swap **daily candles** for the 20 "stack coins" — ADA, ARB, ATOM, AVAX, BNB, BTC, DOGE, ETC, ETH, FIL, INJ, LINK, LTC, NEAR, OP, SUI, SOL, XRP, AVAX-order per `scripts/fetch_crypto_daily.py` COINS list [D]. Universe inherited from the lab's R106 floor spec; R135b (same repo, 2026-11-05) had just shown trend-type edges live on liquid coins, which is cited as the reason for 20 rather than 73 coins (prespec line 7).

**Data source & period.** OKX public REST `history-candles`, bar=1D, fetched into `quantlab_cache/daily/{COIN}_1D.csv`; window 2023-06-01 → 2026-11-05 (~1,254 bars/coin) [D; report line 4]. Candle timestamps are `16:00:00` stamps — OKX daily boundaries follow the exchange's UTC+8 convention, i.e. the "daily close" is the 16:00 UTC snapshot, not midnight UTC [D, observed in CSV]. ⚠ The raw daily cache is not itself committed to the repo (cache directory is ephemeral between sessions); the trade book and code are the persisted record. An auditor regenerates it with the committed fetch script.

**Indicator implementation** (`quantlab_r136_zeitgeist.py:7–40`):
- ATR: Wilder's — TR = max(H−L, |H−PC|, |L−PC|), smoothed with `ewm(alpha=1/14, adjust=False)` (lines 7–10) [D].
- Basic bands: (H+L)/2 ± 3.0×ATR (line 21). Final bands ratchet: upper band may only fall, or reset when the **previous close** exceeded the previous final band (line 26); mirrored for the lower band (line 27) — the standard Pine-style Supertrend reset [D].
- Trend flips when close crosses the opposite final band (lines 29–31) [D].

**Entries / exits.** LONG-ONLY. Entry: the bar where trend flips −1 → +1, at that bar's close (`pos = i`, line 38–39) — close-only execution, no lookahead (flip uses data ≤ i). Exit: the bar where trend flips +1 → −1, at that bar's close (lines 32–37). **No stop-loss, no take-profit, no time stop** — the trailing band itself is the only exit (prespec line 12: "No SL/TP adds") [D].

**Position sizing & risk.** Fixed 1 unit of notional per trade; returns expressed in **R = price-move ÷ ATR(14) at entry** (line 34). No compounding, no leverage modeling, no portfolio-level concurrency constraint (each coin runs its own single-position loop; books pooled in R units) [D].

**Costs.** `r = (c_exit − c_entry)/ATR_entry − 0.0010 × c_exit/ATR_entry` (line 34): a single 0.10% charge on exit notional — the lab's standing 0.10% round-trip convention (equivalent to 2×0.05%; entry and exit notionals differ only by price drift) [D]. A 2× cost stress (0.20%) was run as a pre-declared gate (prespec line 21) [D], [R: PF 2.0190 recomputed].

**Configurations tested.** Exactly **one** parameter set (10, 3.0) + the cost stress on the same set. **No parameter sweep, no optimization, no selection** — parameters were the canonical posted values, frozen in the prespec [D]. No development/validation/out-of-sample split exists: the full window is one pooled test, and stability is judged by the pre-declared yearly survival law and halves test, not by data partitioning [D — this is a real design limitation, disclosed, not hidden].

**What the experiment did NOT test** (declared boundaries): short side, other (period, multiplier) values, other cost models, intrabar fills, funding costs, other venues.

---

## C. VERIFIED RESULTS TABLE

| Metric | Exact value | Definition / calculation | Source | Status |
|---|---|---|---|---|
| Trades | **282** | rows of trade book (excl. header) | `r136_supertrend_book.csv` (283 lines) | [R] |
| Win rate | **34.40%** | share of r > 0 | book | [R] (report:10 said 34.4% ✓) |
| PF@c (pooled) | **2.0413** | Σ(r>0) / |Σ(r<0)| on cost-adjusted r | book; report:10 | [R] |
| Net total | **+448.55R** | Σ r | book; report:10 | [R] |
| Gross wins / losses | +879.3R / −430.8R | Σ positive / Σ negative r | book | [R] |
| Avg win / avg loss | +9.065R / −2.329R | mean of each tail | book; report:15 | [R] |
| Median trade | **−1.566R** | median r (more trades lose than win) | book; report:13 | [R] |
| Winner / loser holds | 56.5d / 23.2d (max 144d) | mean hold by outcome | book; report:14 | [R] |
| Halves PF | H1 1.65 / H2 2.48 | first 141 vs last 141 trades by book order | engine line 82–83 | [R] |
| Yearly (exit-year) | 2023: −24.77 (n=24) · 2024: **+475.08** (n=106) · 2025: +10.72 (n=76) · 2026: **−12.48** (n=76) | Σ r grouped by exit year | book; report:11 | [R] |
| 2026 detail | PF **0.899**, WR 27.6%, GP +111.15, GL −123.63, avg win +5.29, avg loss −2.25 | same formulas on 2026 slice | book | [R] |
| Top-8 share of net | **78.5%** (352.2R of 448.5R) | Σ of 8 largest r ÷ net | book; report:12 ("79%") | [R] |
| Top-8 share of *gross wins* | 40.1% | 8 largest r ÷ Σ positive r | book | [R] — sharper denominator than the report used |
| Top-8 exit-year split | **7 of 8 in 2024**, 1 in 2025 | XRP's +76.6R exited 2025-02-02 | book | [R] |
| 2× cost stress PF | **2.0190** (net +442.3R) | engine re-run, cost=0.0020, window truncated to 2026-11-05 | report:17 ("2.02") | [R] |
| Bootstrap 95% CI (total R) | **[−74.5, +1412.7]**; P(total ≤ 0) = 27.75% | year-block bootstrap, seed 42, 2000 draws, replacement | report:16 ("[−74, +1413]") | [R] — reproduced to rounding |
| Engine determinism | **exact** | re-run on truncated fresh data reproduces all 282 (r, entry_ts) | book vs regeneration | [R] |
| Verdict | **CLOSED** | gate table below | report:9, prespec:19–22 | [X] |

Caveats: (1) all R values are ATR-normalized units, not % returns — PF is scale-invariant but "+448.5R" is not directly a portfolio return; (2) yearly attribution is by **exit year** (code line 36) — a Nov-2024 entry that exits Feb-2025 counts as 2025; (3) the trade book is the only persisted record of per-trade economics (no equity curve file exists).

---

## D. PROFIT CONCENTRATION ANALYSIS (the article's center of gravity)

**The eight trades** [R, from the book]:

| # | Coin | Entry → Exit | Hold | R | Share of net |
|---|---|---|---|---|---|
| 1 | XRP | 2024-11-09 → 2025-02-02 | 85d | **+76.60** | 17.1% |
| 2 | SOL | 2023-09-29 → 2024-01-19 | 112d | **+74.25** | 16.6% |
| 3 | NEAR | 2023-10-23 → 2024-01-08 | 77d | +46.46 | 10.4% |
| 4 | AVAX | 2023-10-24 → 2024-01-05 | 73d | +46.26 | 10.3% |
| 5 | SUI | 2023-10-24 → 2024-03-05 | 133d | +36.46 | 8.1% |
| 6 | BNB | 2023-12-19 → 2024-04-15 | 118d | +25.43 | 5.7% |
| 7 | ADA | 2024-11-06 → 2024-12-19 | 43d | +24.00 | 5.4% |
| 8 | LINK | 2023-09-18 → 2024-01-05 | 109d | +22.77 | 5.1% |

These are **multi-month trend rides on the exact instruments and dates of crypto's 2023-Q4/2024 bull runs** — 7 of 8 exited in 2024, all 8 are longs held 43–133 days. The concentration is *jointly* temporal (one regime), directional (all longs), and size-driven (multi-ATR moves) [R — measured; the regime *label* is interpretation, flagged as such].

**Sensitivity table — remove the largest winners** [R]:

| Variant | n | Net (R) | PF@c | WR | Still positive? |
|---|---|---|---|---|---|
| Full book | 282 | +448.5 | 2.04 | 34.4% | — |
| Drop top 1 | 281 | +371.9 | 1.86 | 34.2% | Yes |
| Drop top 2 | 280 | +297.7 | 1.69 | 33.9% | Yes |
| Drop top 3 | 279 | +251.2 | 1.58 | 33.7% | Yes |
| Drop top 5 | 277 | +168.5 | 1.39 | 33.2% | Yes |
| Drop top 8 | 274 | +96.3 | 1.22 | 32.5% | Yes |

**What this analysis demonstrates — and what it does not.** Removing the eight jackpots does *not* push the book negative: a thin positive base (+96.3R over 274 trades, PF 1.22) survives, so the strategy is not *pure* jackpot noise. But three harder facts sit alongside it [R]: (i) the base expectancy is ~+0.35R/trade before concentration, vs −1.57R median trade — the typical trade still bleeds; (ii) 2025 ex-XRP is **−65.9R** and 2023 is −24.8R (PF 0.34) — outside trend windows the bleed is not thin at all; (iii) the survival-law gate failed on 2026 outright. Removing winners therefore does **not** by itself invalidate the strategy — the correct reading is that profitability is real but *scale and timing are lottery-dependent*, and the lab's stability gates (not the pooled PF) are what expose that. 50 trades lost more than 3R; 26 trades won more than 10R = ~6% of the sample producing the bulk of gross profit [R].

---

## E. YEAR-BY-YEAR PERFORMANCE [R]

| Year (exit) | n | Net R | PF@c | WR | Gross wins | Gross losses |
|---|---|---|---|---|---|---|
| 2023 | 24 | −24.77 | 0.34 | 29% | +12.9 | −37.6 |
| 2024 | 106 | **+475.08** | 4.21 | 42% | +623.0 | −147.9 |
| 2025 | 76 | +10.72 | 1.09 | 32% | +132.3 | −121.6 |
| 2026 | 76 | **−12.48** | 0.899 | 28% | +111.1 | −123.6 |

- 2024's +475.08R contains its own top-8 = **+297.4R (62.6% of the year)** [R].
- 2025's +10.72R contains the XRP +76.60R ride → **ex-XRP 2025 = −65.9R** [R].
- 2026 monthly pattern (exit months): **Jan −39.7R (n=16)** and **May −20.8R (n=15)** dominate the loss — both are high-trade-count whipsaw months; Jun +9.3, Sep +3.7, **Oct +37.8 (n=18)**, Nov (partial) +22.1 [R]. The negative year is therefore not a uniform bleed: two dense losing stretches + one late-year trend recovery. (No regime *explanation* is claimed here — this is the measured pattern. The lab has separately *queued*, not tested, a hypothesis that trade-frequency density marks such whipsaw periods: `docs/RESEARCH_FRONTIER_R135b.md` addendum.)

---

## F. STATISTICAL ANALYSIS — THE YEAR-BLOCK BOOTSTRAP

**Why bootstrap:** trade returns are non-normal (median −1.57R vs mean +1.59R, jackpot tail), so a parametric t-test on total return is inappropriate; resampling preserves the empirical distribution.

**Procedure (as documented in `r136_final_report.md:16` and reproduced exactly this session):** observations = per-trade R grouped into **blocks by exit-year** (2023, 2024, 2025, 2026 — 4 blocks, first-appearance order); blocks sampled **with replacement** (4 draws per replicate); **statistic = pooled total R**; **2,000** replicates; **95% CI** via the 2.5/97.5 percentiles; RNG seed 42 (`numpy default_rng`). Year blocks (rather than single trades) preserve within-year serial dependence — trend rides cluster in time, so i.i.d. trade resampling would overstate independence.

**Result** [R, reproduced to rounding from the raw book]: **CI [−74.5, +1412.7] R**; bootstrap distribution mean ≈ +451R (consistent with the observed +449R); **P(replicated total-R ≤ 0) ≈ 27.8%**.

**What "the interval includes zero" means — precisely.** It does **NOT** prove the strategy has no edge. It means: under year-block resampling, a strategy with this book's yearly structure would produce a losing pooled total about 28% of the time; zero lies comfortably inside the 95% interval, so the data cannot *rule out* "no edge" at conventional confidence. The point estimate (+449R) is strongly positive; the *uncertainty about which years the luck replicates* is enormous because the profit mass is concentrated in one block.

**Method appropriateness — honest critique.** With only 4 year-blocks, the bootstrap's percentile endpoints are coarse (each block is 25% of every replicate; the CI width is driven by whether 2024 is drawn 0–4 times). A stationary-block bootstrap on shorter blocks or a within-year trade-level test would tighten it — but any finer resampling would *understate* the regime risk, which is the actual finding. The block choice is the conservative and, in this lab's judgment, the honest one. The original bootstrap was executed in-session and documented, but **not persisted as a repo script** — this session's independent reproduction from the raw book (identical seed and procedure) is the verification of record.

---

## G. DECISION AUDIT — WHY "CLOSED"

**The gate contract** (pre-declared, `docs/RESEARCH_FRONTIER_R136.md:19–22`, applied by `quantlab_r136_zeitgeist.py:84–89`):

| Gate | Threshold | Supertrend result | Outcome |
|---|---|---|---|
| Sample size | n ≥ 30 | 282 | **PASS** [R] |
| Profit factor | PF@c ≥ 1.15 | 2.04 | **PASS** [R] |
| **Survival law** | 2024 ∧ 2025 ∧ 2026 each ≥ 0R | 2026 = −12.48R | **FAIL** [R] |
| Halves | both PF ≥ 1.0 | 1.65 / 2.48 | **PASS** [R] |
| 2× cost stress | PF ≥ 1.0 at 0.20% RT | 2.02 | **PASS** [R] |

**Decision:** the printed verdict is "CLOSED — noise/dead" (engine line 89); the report's considered language is subtler: the engine "rides trends fine — the INDICATOR was never the edge, the trend was, and the trend is not always there" (report lines 24–27) [X].

**What the decision is based on:** a combination — statistical uncertainty (CI spans zero) *and* out-of-time instability (2026 negative) *and* concentration (79%/2.8% of trades) — with performance *levels* (PF 2.04) explicitly not the basis. **What CLOSED means:** the strategy is not promoted to candidate/champion status, is not paper-tracked, and receives no capital; it is *not* a claim that the strategy loses money in every future scenario, and *not* a permanent statement about every possible Supertrend parameterization (which the prespec forbade testing). **The critical distinction for the article:** the experiment demonstrates a strategy that is *profitable in-sample yet failed to demonstrate sufficiently reliable profitability* — unprofitable ≠ failed-to-prove; this case is the latter, with a measured mechanism for why.

---

## H. TECHNICAL IMPLEMENTATION AUDIT

**Code inspected line-by-line** (`quantlab_r136_zeitgeist.py`, 99 lines). Findings:

1. **Supertrend mechanics are correct** [R]: Wilder ATR (7–10); basic bands (21); ratchet with previous-close reset (26–27); flip on close-cross of final bands (29–31). Verified behaviorally by full-book regeneration (282/282 exact).
2. **No lookahead**: flip at bar *i* uses closes ≤ *i*; entry and exit execute at the same bar's close (32–39). Close-only execution omits intrabar band touches — a trade whose band is pierced intraday but closes back inside will not exit that day. This is a disclosed modeling choice, standard for daily-close backtests; it can *delay* exits (both directions).
3. **Warm-up is safe**: `trend` initializes to +1 (line 23) and entries require a prior −1 state (38), so no trade can fire before a genuine down-flip exists; ATR warms via EMA from bar 0 (TR of bar 0 = H−L, line 8) [R].
4. **Cost model**: single 0.10% charge on *exit* notional (line 34) — the lab's RT convention; entry leg effectively free at entry price ≈ exit price relative to ATR multiples. Slippage/spread are folded into the flat 0.10%; no depth or partial-fill model [D]. Exact R formula: `r = (c_exit − c_entry)/ATR_entry − 0.0010·c_exit/ATR_entry`.
5. **Cosmetic dead code**: lines 53 and 62 contain `... if False else ...` fallback fragments (Ichimoku cell) — unreachable, no effect on Supertrend, flagged for cleanliness only [R].
6. **Year attribution by exit** (line 36) — affects yearly tables as disclosed in C/E [D].
7. **Environment note, not a code issue**: the daily-candle cache is session-ephemeral; this audit refetched it and confirmed the engine reproduces the book exactly when truncated to the original window — ruling out data-drift as a source of the published numbers [R].
8. **No material implementation issue affecting conclusions was found.** Limitations are design-level (single pooled window, no dev/val/OOS split, one parameterization, flat costs, R-unit accounting) and are disclosed in B and J, not hidden.

---

## COMPARATIVE CONTEXT (documented, verifiable, secondary)

| Experiment | Headline | Robustness outcome | Repo receipt |
|---|---|---|---|
| **R135 trendline break** | PF@c 1.42, +229.1R, n=1,076 | yearly **all positive** (2023–26), halves 1.44/1.41, bootstrap [+165, +293] excludes 0 → the contrast case: passed the same gates Supertrend failed | `quantlab_output/r135_final_report.md` |
| **R136 Ichimoku (same round)** | PF@c 2.28, net **+11.3R**, n=187 | 2025 −2.2R, 2026 −0.5R — same jackpot anatomy, less of it | `r136_final_report.md:19–21` |
| **R131 shock-continuation** | PF@c 1.04 at n=12,460 | probability lift consumed by direction risk/costs/path — "lift ≠ expectancy" | `quantlab_output/r131_final_report.md` |
| **R110 cascade sniper** | +0.07R/trade avg, n=227 | premium real but thin at every clock speed | `quantlab_output/r110_final_report.md` |
| **R085–R095 scalping battery** | — | the cost wall: per-trade edge ≪ costs kills any frequency | lab standards doc + reports |
| **Champion (R096)** | PF@c ~1.5–1.8, ~3 trades/mo | passed the survival-law family; forward-tracked, currently halted at its own gate | `CHAMPION.md`, `champion_recheck_*.md` |

The useful comparison for the article is **R135 vs R136 — same trend family, same gates, same costs**: one survived the yearly law, one didn't; the difference is exit design (fast-fail re-cross + time stop vs pure trailing flip), not indicator choice.

---

## I. ARTICLE-READY FINDINGS (defensible, each traceable to §C–§H)

1. **PF 2.04 with a losing median trade.** 34.4% of trades won; the median trade lost −1.57R; the *average* win (+9.07R) was 3.9× the average loss (−2.33R). Profit factor measures asymmetry, not typical experience.
2. **8 trades (2.8% of the sample) = 78.5% of net profit; 40.1% of gross wins.** All 8 are longs held 43–133 days, 7 of 8 exiting in 2024. A profit factor can be a lottery summary.
3. **Remove the 8 jackpots and the book stays positive (+96.3R, PF 1.22) — but 2025 ex-jackpot is −65.9R and 2026 is −12.5R.** The right conclusion is not "no edge," it is "edge that only pays in one regime, which you cannot schedule."
4. **The survival-law gate (each year ≥ 0R) is the test that caught what PF hid.** Pre-declared, trivially simple, and it failed on 2026 while four flattering metrics passed.
5. **A year-block bootstrap CI of [−74.5, +1412.7] R with P(total ≤ 0) ≈ 28%** — the honest statistical statement is "cannot rule out no-edge," not "proven no-edge." Articles that say "the bootstrap proved it's noise" are wrong; this lab's own report uses the careful phrasing.
6. **2× costs changed nothing (PF 2.04 → 2.02)** — at one trade per coin-month, costs are not the failure mode; *structure* (regime dependence) is. The inverse lesson from the same lab: at high frequency, costs kill everything (R085–R095, R137).
7. **Same family, same gates, different exits → different verdict:** R135's trendline book (fast-fail re-cross exits + 10-day cap) passed every gate; Supertrend's pure trailing flip did not. Exit design, not indicator selection, separated the candidate from the lottery ticket.
8. **The audit trail matters as much as the verdict:** the entire experiment is reproducible from four committed files (prespec, engine, book, report), and this session regenerated the book exactly, recomputed every headline figure, and reproduced the bootstrap to the decimal. That is what "we tested it" should mean in public.

---

## J. OUTSTANDING QUESTIONS (cannot be established from repository evidence)

1. **Parameter sensitivity is unknown** — the prespec froze (10, 3.0) and forbade sweeps; no evidence exists on how PF/robustness vary with period or multiplier. (Declared boundary, not an oversight.)
2. **Short-side Supertrend was never tested** (long-only by prespec line 12).
3. **Intraday max drawdown is not computable** from daily closes; only a trade-sequence R-space DD (max 137.1R, by exit order) exists [R]. No equity-curve file is persisted.
4. **The daily candles are OKX's UTC+8-aligned days** and reflect today's listed (surviving) coin set; delisted-instrument survivorship bias is unquantified in this cell.
5. **The original bootstrap ran in-session and was not persisted as a script**; its parameters are documented in the report and were reproduced exactly here, but an auditor cannot diff the original script byte-for-byte — only its output.
6. **No dev/val/OOS partition exists**; the yearly law substitutes as the stability check. Any claim of "out-of-sample" performance for this experiment would be false.
7. **Slippage beyond the flat 0.10% RT** (spread regimes, thin-coin depth) is unmodeled; the 2× stress bounds but does not simulate it.
8. **2027+ forward behavior is unknown** — the experiment was closed, so no forward tracking exists by design (unlike the champion and R135, which are paper-tracked).
9. **The exact cause of the 2026 loss is not identifiable** from this experiment alone: monthly data show two dense whipsaw stretches (Jan, May) and an October recovery, but no regime variable was measured in this cell; the frequency-density hypothesis is queued, untested (`RESEARCH_FRONTIER_R135b.md` addendum).

---

*Package prepared from: `docs/RESEARCH_FRONTIER_R136.md` · `quantlab_r136_zeitgeist.py` · `quantlab_output/r136_final_report.md` · `quantlab_output/r136_supertrend_book.csv` (+ Ichimoku book) — all verified byte-identical to pushed commit `103dc19`. Every [R] figure recalculated this session from the raw trade book with formulas shown; no QuantLab file was modified; no new experiment was run (cache refetch and in-memory recomputation only).*
