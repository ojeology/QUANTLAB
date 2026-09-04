# QuantLab
### Quantitative Strategy Research Laboratory

> A systematic research framework for discovering, validating, and stress-testing
> algorithmic **crypto → forex** trading strategies using rigorous statistical methods.
> Every hypothesis is documented — including the failures. **No run is hidden.**

---

## ═══════════════════════════════════════════════════════════════
## 📜 STATUS — 2026-09-04 · the research runs on TWO git branches
## ❄️ STRATEGY FREEZE (2026-08-09) · partly superseded by blind OOS validation
## ═══════════════════════════════════════════════════════════════

The repository contains **two research arcs**:

- **`main`** — the discovery hunt: crypto R001–R095 (2026-07-28 → 08-08; closed with
  **5m: no edge / 1H: SVM edge found**) and forex F001–F009 (2026-08-08/09;
  **bear-trap-reversal validated**), halted by the **STRATEGY FREEZE** on 2026-08-09.
  → This README documents that arc end-to-end (below), then continues with the sub-branch.
- **`blind-validation`** (sub-branch — forks exactly at the freeze commit) — the
  post-freeze **blind out-of-sample campaign** (2026-08-31 → 09-04; tests **T1–T34** +
  sub-branch probes A–E) that re-validated everything on **2023–2026 per-year holdouts
  with costs** and **revised the freeze conclusions**. Its master README, `FINDINGS.md`,
  `TEST_LOG.md` and `RESEARCH_DERIV.md` live on that branch. **T34 (2026-09-04)** audited
  implementability for a small ($100) account — see the correction block below.

### ✅ Current validated answer (branch `blind-validation`, 2026-09-03)

- **Crypto 1H is the only validated, deployable edge.** Final vehicle: **T27 — MR + trend
  portfolio (trend-weighted 70/30)** → **PF 1.40 / 1.32 / 1.81 (2024/25/26), MAX DD < 1%**.
  Strongest single strategy: **T25 — condition-aware TREND** (Donchian breakout + ADX>20 &
  above-EMA200 + RF q65 filter) → **PF 1.744 / 1.529 / 1.537**, DD ≈ −2%.
- **MR champion:** SVM q0.65 + **regime-gated VolCeil** (skip the ATR-spike rule only when
  price is stretched from its mean, `|ema_dist_pct| > 2.0`) → PF@cost **1.18 / 1.82 / 1.53**
  (2024/25/26 — the definitive T13 run; an earlier FINDINGS pass measured 2025 at 2.11),
  cumulative ≈ +71%, max DD ≈ −9.4%, 15/27 profitable months, 22/30 symbols profitable
  (top-3 only 37% of R).
- **Everything faster or on other instruments is data-blocked or edge-less** in that
  sandbox: crypto 15m (data-blocked), Deriv synthetics (random walk), forex 1H spot
  (T25 pipeline → PF 1.099 ≈ break-even, no transfer), stocks (no data).

### 🆕 2026-09-04 — T34 audit: two corrections supersede parts of the answer above

*T34 (branch `blind-validation`) re-ran the champions on the **full 50-symbol
2023→2026 universe** (2023 fetched from OKX; per-year walk-forward; fees 0.05%/side) to
size a $100 account. Files: `t34_lib.py`, `t34_validate.py`, `t34_small_account.py`,
`t34_output/`.*

1. **The T25/T27 trend numbers are exit-anchored — not implementable live.** The trend
   simulator records each trade at the *exit* bar, so the RF-q65 "condition" gate reads
   **exit-bar features** (post-entry info). Verbatim reproduction matches the log
   ($100→$198.79 vs $199.70) — but the **same champion gated at the true entry bar is
   PF@c ≈ 1.03, $100→$106 — no edge**. All TP variants (1–3R) are negative.
   → **T25 / T27 / PORT trend "PF≥1.3, DD<1%" claims are superseded** for any live or
   paper go-live until an entry-anchored trend gate is found.
2. **MR champion on the full universe: 2024 loses, drawdown is bigger than reported.**
   n=297, PF@cost **0.739 / 2.151 / 1.610**, return@1% **+71.2%** (the old +71.3%
   reproduces), **max DD @1% = −28.2%** (the 30-sym "−9.4%" missed 2024's losing regime).

**What this means for a $100 account (T34 recommendation):** trade the **MR leg only** —
FAM_A coil + SVM q0.65 adaptive VolCeil, **TP 2R / SL 1R** (R:R optimum: mean R +0.30 vs
+0.22 at the RR1.5 default), **1% risk/trade** → **$100 → ~$204 (+104%)** over
2024→Jul-2026 (~32% CAGR), MC P(profit) 99.9%, P(halve) ≈ 0%, **realized max DD −37%**
(2024→mid-2025 grind, trough ≈$63, then +49% Aug-2025). RR1.5@1% = the smoother
alternative ($173, DD −27%). Risk ≥2% → DD −50…−80% (not for small accounts). Practical
leverage: 1% risk ≈ ~1× notional (MR stop ≈1.15% of price); 3–10× exchange leverage only
for contract fit, never the risk dial.

### ⚠️ The 2026-08-09 freeze config is SUPERSEDED for a 2027 go-live

| Frozen 2026-08-09 (main) | Blind OOS re-test (branch `blind-validation`) |
|---|---|
| Crypto 1H **SVM q0.75** + static VolCeil/breadth50 · backtest PF **1.94** | **q0.75 → PF@cost ≈ 1.25 (2026)**, and **the R077 static config → PF@cost ≈ 0.75 (loses after fees)**; **no static (q × VolCeil × breadth) sweep survives all of 2024/25/26** → champion is **q0.65 + regime-gated VolCeil**, then the **T27 MR+trend portfolio** |
| Forex bear-trap (F007) · holPF@cost 1.14 | Not re-derived on that branch (T33 only tested the crypto-*trend* pipeline on forex 1H spot → no transfer). F007 stands on main pending the year-end re-check |

**Year-end protocol (Dec 2026):** fetch fresh data and re-run the full validation battery
on the blind-validation champions (T27 portfolio / SVM q0.65-adaptive) → green-light 2027
only if the edge holds year-by-year (`PF@cost ≥ ~1.2`). Details in
`.agents/memory/quantlab-research-state.md`.

**Key docs:** [`FOREX_HUNT_BEGINS.md`](FOREX_HUNT_BEGINS.md) (crypto close-out) ·
[`EXIT_MODEL_AUDIT.md`](EXIT_MODEL_AUDIT.md) (exit-engine correction) ·
`blind-validation` branch → `README.md` (master tree & T1–T33 table), `FINDINGS.md`, `TEST_LOG.md`, `RESEARCH_DERIV.md`.


---

> **The goal isn't to find a strategy that looks good. The goal is to find one that
> survives every attempt to prove it wrong.**

---

## Table of contents
1. [What this repository is](#what-this-repository-is)
2. [Research goals & philosophy](#research-goals--philosophy)
3. [The validation framework (every strategy must pass)](#the-validation-framework)
4. [Repository layout (real)](#repository-layout)
5. [How to run things](#how-to-run-things)
6. [How to read the research log](#how-to-read-the-research-log)
7. [📕 The full research log — R001 → R095 (crypto)](#-the-full-research-log--crypto)
8. [📗 The full research log — F001 → F009 (forex)](#-the-full-research-log--forex)
9. [📘 Post-freeze — the blind-validation branch (T1–T33)](#-post-freeze--the-blind-validation-branch)
10. [Key lessons learned](#key-lessons-learned)
11. [Technologies](#technologies)
12. [Disclaimer & author](#disclaimer--author)

---

# What this repository is

QuantLab is a **quantitative research laboratory**, not a strategy dump. Across **two git
branches** it spans ~95 numbered crypto research iterations (R001–R095), 9 forex iterations
(F001–F009), and — on the `blind-validation` sub-branch — a **T1–T33 blind out-of-sample
test campaign** that re-validated everything on 2023–2026 and produced the final
deployable result (T27). Every run asks one falsifiable question, answers it on
**out-of-sample data only**, gates the result behind **costs and statistical tests**, and
writes the outcome (success **or** failure) to a journal.

Notable negative results are as important as positive ones — the project's biggest
conclusions are *negative*:
- 5-minute crypto has **no cost-surviving edge** (proven 7 independent ways, R089–R095).
- The pre-audit R066–R071 "edge" was a **backtest proxy artifact** (exit-model audit).
- Most discovered edges are **universe-specific** and fail on unseen symbols (R044/R049/R078).
- The Aug-2026 "freeze config" **failed** a strict per-year blind re-test (branch
  `blind-validation`), which is what forced the better final config (T25/T27).

---

# Research goals & philosophy

- Discover statistically robust trading strategies — then try to **disprove** them.
- Minimize overfitting through **walk-forward** optimization and **untouched holdouts**.
- Validate with **multiple independent methods** (bootstrap, LOO, Monte Carlo…).
- Gate every result behind **realistic costs** (fees + spread + slippage).
- Compare competing strategy families; measure robustness, never one lucky backtest.
- Document everything — a failed experiment is a finding.

---

# The validation framework

Every candidate strategy is expected to survive most of:

- **Walk-Forward Optimization** — train only on the past, never on the future
- **Out-of-Sample / Holdout Testing** — a period (e.g. 2026) kept untouched until the verdict
- **Cost gates** — 0.05%/side crypto, retail spread + swap on forex; an edge that dies at cost is not an edge
- **Causal (lookahead) audit** — mandatory after the R090 retraction
- **Bootstrap confidence intervals** (PF p5/p50/p95, paired differences)
- **Monte Carlo simulation** — P(profit), drawdown distributions, dollar-DD
- **Leave-One-Out symbol & fold validation**
- **Monthly stability** — % profitable months, worst losing-month streak
- **Regime / drawdown / risk analysis** and **parameter robustness grids**
- **Portfolio validation** (multiple environments/families, diversification)

Promotion vocabulary used across the log: `PROMOTE` / `GO` / `VALIDATED` (passes),
`WATCHLIST` (promising, sample too thin), `REJECT` / `NO-GO` (fails), `RETRACTED` (result
withdrawn), `OVERFIT` (does not generalise).

---

# Repository layout

The layout differs from the first README draft (which pointed at `research/`, `engine/`,
`strategies/`, `docs/` — those folders never existed here). **The real layout, per branch:**

```
# — main branch (discovery hunt) —
quantlab_rXXX.py            One script per research run (R005–R095, gaps noted in log)
forex_fXXX.py               One script per forex run (F001–F009, F006b/F006c helpers)
quantlab_ai.py              RESEARCH #004 multi-strategy engine + shared CONFIG & helpers
scripts/ql_engine.py        Shared walk-forward backtest engine: feature engineering
                            (add_features, build_signal_mask, sim_symbol) — used by R073+
demo_bot.py                 Paper-trading bot (SQLite + Telegram alerts; NEVER live)
bot_config.yaml             Frozen strategy config used by the demo bot
quantlab_output/            Every artifact: *_final_report.md, *_journal.md, *.csv, *.png
quantlab_output/research_journal.csv   Master machine-readable verdict log (R004+)
quantlab_cache/             Parquet OHLCV caches (OKX crypto swaps; forex/ yfinance 1H/1D)
EXIT_MODEL_AUDIT.md         The exit-engine correction (supersedes R066–R072 verdicts)
FOREX_HUNT_BEGINS.md        Crypto close-out → forex launch document

# — blind-validation branch (post-freeze blind OOS campaign) —
README.md                   Master README: exploration tree + T1–T33 master table
FINDINGS.md                 Consolidated blind-OOS findings + champion
TEST_LOG.md                 Numbered test log T1–T27 (+ reasons)
RESEARCH_DERIV.md           Deriv / binary-options findings
scripts/svm_deploy.py       ML condition filter (RF/SVM), build_mldf + deployable module
blind_test.py, svm_crypto_blind.py, champion_showdown_2026.py, improvements.py,
split_2025_2026.py, test_2023.py, filter_sweep*.py, filter_sweep_adaptive*.py,
champion_deep_dive.py       The T-series engine scripts
t15_*.py, t25_*.py, t31_combined.py, t_all_1h.py, ml_conditions.py, trend_*.py,
mt_portfolio.py, port_mr_trend*.py, portfolio_hypothesis.py, fresh_hypothesis_1h.py,
cage_*.py, daily_hypothesis.py, ensemble_ab.py, drawdown_sizing.py, rehearsal.py  T-series tests
deriv_*.py, forex_15m_*.py, forex_1h_spot.py   Sub-branch C/D (Deriv binaries, forex spot)
fetch_15m*.py, fetch_2023_all.py, fetch_and_save_2023.py, local_fetch_15m.py  15m/2023 fetchers
*.log                       Human-readable per-test outputs
.agents/memory/             Living research-state memory (per-run summaries)
```

Each run's outputs share a prefix in `quantlab_output/`, e.g. `r077_final_report.md`,
`r077_locked_monthly.csv`, `r077_rr_sweep.csv`. Charts are PNGs; trade logs are CSVs.

---

# How to run things

```bash
# Reproduce any research run (pulls data from quantlab_cache/, writes to quantlab_output/)
python quantlab_r095.py          # or forex_f007.py, quantlab_r077.py, …

# Demo paper-trading bot (schedules hourly scans; flags below for one-shot modes)
python demo_bot.py --scan-now    # single scan & exit
python demo_bot.py --status      # open positions / equity
python demo_bot.py --report      # performance summary
```

Environment: Python + pandas/numpy/scikit-learn/matplotlib (see `pyproject.toml`,
`uv.lock`); SQLite for the bot (`demo_bot.db`). Data lives in `quantlab_cache/` and is
fetched by the scripts in `scripts/` (`fetch_5m.py`, `fetch_forex.py`, `fetch_more_symbols.py`).
> Note: 5m/1m cache indexes were at one point plain integers (the R067b timestamp bug).
> Runs from R069 onward use real UTC timestamps; see `quantlab_fix_timestamps.py`.

---

# How to read the research log

**Run numbering:** the crypto hunt is numbered R001…R095 and the forex hunt F001…F009.
Numbering is not fully dense — **R010/R011 have no surviving scripts**, **R055 was never
used**, and R088/R092 live only under `scripts/`. R090 was reported, **retracted the same
day**, then re-reported as a negative — both entries are logged. R078's engine lives in
`scripts/` too.

**Sources used to compile the log:** module docstrings (each run states its pre-registered
question), commit messages, `quantlab_output/research_journal.csv`, per-run
`*_final_report.md` / `*_journal.md`, `.agents/memory/MEMORY.md`, and the two top-level
audit documents. Metrics quoted are as reported by each run.

**Timeline of the repo itself:** the research ran Jul 28 → Aug 9 2026 on `main`, paused at
the STRATEGY FREEZE (2026-08-09), and continued **Aug 31 → Sep 3 2026** on the
`blind-validation` sub-branch (T1–T33). The GitHub history was re-initialised on
**2026-08-03** (why R001–R071-era commits sit under an older chain and R073+ under a fresh
one; the Aug 3 commit also added the first README draft). R001–R004 predate per-run
scripts — they were developed *inside* `quantlab_ai.py`, so their record lives in that
file's docstrings, git history, and the untagged `BTC/ETH/SOL_USDT_SWAP_*` artifacts in
`quantlab_output/`. This README documents `main` (R-series + F-series) and then the
`blind-validation` branch (T-series) in one combined log.

---

# 📕 The full research log — crypto

## Phase 0 — Foundations, single-strategy era (R001–R004) · 2026-07-28

No standalone scripts survived; engine = `quantlab_ai.py`. Symbols BTC/ETH/SOL 1H OKX
perps; locked engine, fees, spread, slippage, SL/TP, train/OOS split.

- **R001 — Bullish FVG + EMA200** *(in `quantlab_ai.py` git history)*
  Single-strategy tester: FVG + EMA200 long vs an EMA200-crossover benchmark.
  Established that **1m gives zero FVG signals on 24/7 crypto → moved to 1H**.
  Artifacts: untagged `*_trade_log.csv`, `*_monte_carlo.png`, `*_equity_drawdown_distribution.png`.
- **R002 — Three-way comparison** — A) EMA200 crossover (benchmark), B) EMA200+FVG
  (R001), C) EMA200+FVG+**positive slope**. Slope hypothesis supported → carried to R003.
  Artifacts: `*_r002_*`.
- **R003 — Regime tagging** — runs the R002 winner unchanged and adds per-trade **ADX
  regime tagging** + regime analysis. Artifacts: `*_r003_*` incl. `r003_regime_breakdown.png`.
- **R004 — Six-strategy tournament** (`quantlab_ai.py`, `RESEARCH_ID="R004"`)
  Six independent concepts head-to-head on an identical locked engine: FVG+Slope (baseline),
  **Liquidity Sweep Reversal (LSR)**, Break of Structure, VWAP Pullback, Opening Range
  Breakout, Volatility Compression. OOS leaderboards per symbol; journaling begins
  (`research_journal.csv`). Near-total rejection except thin per-symbol PROMOTEs
  (LSR/SOL, BOS/ETH) and a flagged VCB/DOGE (PF 2.58, n=11 — later killed in R019).

## Phase 1 — Liquidity Sweep Reversal path (R005–R011) · 2026-07-28/29

R005–R009 have scripts; **R010–R011 have no surviving scripts/artifacts** (later runs cite
"LSR entry rules from Research #010" in `quantlab_ai.py`). The path never reached
portfolio-level validity and was abandoned at R012.

- **R005 — Trade attribution** — which pre-entry context features separate LSR wins from
  losses (descriptive only). Journal: per-symbol REJECT/REJECT/PROMOTE(SOL, thin).
- **R006 — Threshold discovery** — decile analysis of the strongest R005 features
  (`atr_rank_pct`, `dist_from_ll_pct`, `dist_from_ema200_pct`, `ema200_slope_pct`,
  `funding_rate`, `hour_utc`). No strategy change.
- **R007 — Explainable ML** — Random Forest win/loss classifier (chronological 70/30),
  SHAP + interactions. Strongest lead: **high realised vol AND price near the 20-bar high**.
- **R008 — Confirmation study** — added the R007 interaction as entry conditions;
  structural edge or small-sample artefact?
- **R009 — Session + relative-volume filter** — LSR restricted to London+NY hours
  (08:00–21:59 UTC) with rel_vol ≥ threshold. Mixed per-symbol; suggested R010.
- **R010 / R011** — LSR refinement iterations (no artifacts; referenced by R012/R018).
  R012 explicitly "breaks completely from the LSR path (R005–R011)".

## Phase 2 — Fresh hypotheses & timeframes (R012–R016) · 2026-07-29

- **R012 — Three mean-reversion concepts** — BB.Bounce, RSI.Rev, 3Bar.Rev (downtrend
  oversold → reclaim, all in uptrends). **REJECT** (journal).
- **R013 — BTC lead / cross-asset momentum** — strong BTC 1H momentum above its 200-EMA as
  a 1-candle-lag long signal on ETH/SOL (BTC.Self control included).
- **R014 — Trend continuation after pullback, 1m** — EMA200-trend + EMA20/50 pullback +
  reclaim; swing-low stop, 2R target, 8 symbols.
- **R015 — Same strategy on 15m/1H** — timeframe test of R014.
- **R016 — Volatility compression → expansion** — ATR/BBW squeeze + breakout, 15m/1H,
  8 symbols. *Conceptual ancestor of the later Family-A compression-pop signal.*

## Phase 3 — Regimes & universal-edge mining (R017–R020) · 2026-07-29

- **R017 — Market regime classification** — unsupervised KMeans on 8 symbols, 15m/1H:
  Regime 0 = low-vol/trending (~80% of bars), Regime 1 = high-vol/bearish (~20%). No trades.
- **R018 — Regime-conditioned LSR** — is LSR's edge regime-specific? **REJECT** overall.
- **R019 — Volatility-Compression Breakout full validation** — deep-dive of the R004 DOGE
  lead across 8 symbols, parameters, 15m. **REJECT** — the flag was small-sample luck.
- **R020 — Funding-rate extremes mean-reversion** — fade crowded long/short when OKX 8h
  funding is extreme (threshold grid on BTC/ETH/SOL). **REJECT**.

## Phase 4 — Strategy tournaments & the Low-ATR discovery (R021–R027) · 2026-07-29

- **R021 — Trend-following family tournament** — EMA crossover + ADX (12 combos) vs
  Donchian breakout (4) vs EMA pullback, portfolio OOS PF winner. EMA pullback = most trades.
- **R022 — Stop-loss test** — ATR(14) stops vs previous-bar-low stop on the EMA-pullback
  winner (baseline & 1.0/1.5 ATR variants). REJECT rows in journal.
- **R023 — Universal "edge blueprint"** — pool *all* prior strategies' OOS trades
  (~2–4k), ML + SHAP for the universal conditions of a profitable trade.
- **R024 — Blueprint falsification** — apply R023's gates to EMA pullback out-of-sample.
- **R025 — Failed breakdown reversal** — close below 20-bar low, next bar closes back
  above it (trap the shorts), low-ATR + wide-BB context.
- **R026 — Universal environment validation** — same env (near 20-bar low + wide BB)
  applied across 6 entry concepts. LiqSweep_filtered → WATCHLIST.
- **R027 — Volatility regime validation** — Low-ATR (<p25) vs High-ATR (>p75) across
  strategies. **FVG+Slope_LowATR → WATCHLIST (PF 1.089, n=37 — too thin).**

## Phase 5 — FVG + Slope + Low-ATR: the first profitable config (R028–R034) · 2026-07-29

- **R028 — Scale validation at 4H** — resample to 4H for more independence: <16 signals.
  Signal too rare on higher TF.
- **R029 — Large-sample validation (9 symbols, 1H)** — **Low ATR → PF 1.205, n=64** —
  the first profitable configuration; missed PROMOTE only on n<80 / boot p50<1.20.
- **R030 — ADX attribution** (on the exact 64 trades) — ADX14 does **not** explain the edge. REJECT.
- **R031 — BB-Width attribution** — BBW ranks #1 (permutation + SHAP); narrowest quartile
  (BBW ≤ p25): WR 62.5%, PF 1.92.
- **R032 — BBW sweet-spot validation** — add BBW ≤ p25 as a filter: **REJECT** (PF < 1.0,
  n=35) — the quartile did not transfer as a rule.
- **R033 — Walk-forward validation** — 5 expanding OOS windows on the Low-ATR edge:
  held (PROMOTE, PF ≈ 1.22).
- **R034 — Maximum-history validation** — every symbol with ≥4k 1H bars, added LOO-fold:
  **REJECT (PF 0.87)**. The Low-ATR config died at full scale → lesson: validate on the
  *largest* honest sample, not the convenient one.

## Phase 6 — Environment-native research (R035–R044) · 2026-07-29

- **R035 — Pooled universal discovery** — pool R002–R034 trades, quintile/interaction/
  tree search → winning environment: **ATR-low + EMA200-slope>0 + far above EMA200 +
  narrow BB** (n=190, WR 49.5%, PF 1.356).
- **R036 — Environment-native strategies (A/B/C)** — mean-reversion, momentum, and
  **RelVol-breakout** entries inside the R035 env → all WATCHLIST (C: PF 1.82, n thin).
- **R037 — Gate sensitivity** — remove one of the four gates at a time: all-4-gate
  BASELINE is best; **EMA-distance is essential** (relaxing drops PF 1.82→1.07).
- **R038 — Entry-family tournament inside the env** — 9 entry families: **6/9 > PF 1.20**
  (RelVol 1.82, Donchian, Pullback…) → **the environment is the edge, not the entry**.
- **R039 — Environment expansion (recover trade count)** — relax one gate per variant:
  Var D (BB p50) PF 2.22 / n 27 → WATCHLIST; n≥100 never reached.
- **R040 — Timeframe transfer 1H→15m** — Var D on 15m: **REJECT** (PF 0.52, MC 2.6%) —
  edge is timeframe-specific.
- **R041 — 1H relaxation grid (Pareto)** — no variant achieves PF>1.20 *and* n≥100;
  WATCHLIST C/E/G (best PF 2.22 n=27).
- **R042 — Independent environment discovery** — library of 130 environments, 104 PROMOTE
  (many were **Wed-Thu + US-session calendar patterns** — later exposed as artefacts in
  R051); top: `Dist>p75 · Wed-Thu · BodyPct>p60 · US(14-21)` PF 2.00 n=68.
- **R043 — Portfolio of independent environments** — **Portfolio C (E1+E2+E3+E4):
  PF 1.51, n=273, 7/7 → PROMOTE.**
- **R044 — External-symbol validation** — the frozen Portfolio C on 26 **never-seen**
  symbols: **REJECT — OVERFIT** (PF 1.16, boot-median fail). The R043 result was
  universe-specific. *(Scripts: `quantlab_r044_download.py` / `_probe.py` fetched them.)*

## Phase 7 — Generalisation stress & Universal Discovery 2.0 (R045–R054) · 2026-07-29/30

- **R045 — Asset-archetype fit** — post-hoc: which asset classes fit Portfolio C? No params changed.
- **R046 — Greedy portfolio** over independent R042 envs (23 symbols) → 9 survivors
  (E05–E11, E15, E16; E12 rejected).
- **R047 — Exhaustive portfolio search** — all 372 feasible 2–5-environment portfolios →
  **global optimum E06+E11**.
- **R048 — Frozen blind forward validation** — run E06+E11 once on bars that appeared
  *after* R047 completed → too few new bars yet.
- **R049 — Across-space validation** — frozen E06+E11 on 26 never-researched symbols:
  **fails to transfer.**
- **R050 — Universe robustness scan** — does *any* of the 9 survivor envs hold on the new
  universe? Negative.
- **R051 — Universal edge discovery** — ablation + interaction profiling proves the
  **calendar conditions (Mon-Tue / Wed-Thu) are historical artefacts**, not market structure.
- **R052 — Universal Environment Discovery 2.0** — calendar conditions **forbidden**,
  structural/behaviour filters only → **PROMOTE (7/7): E2** `DST_NR+PRG_HI+PBP_LO+LON`
  (PF 1.34, n=303), **E3** `BBW_LO+RV_LO+DST_NR+PRG_VH` (PF 1.39, n=270), **E8**
  `SLP_DN+PRG_HI+PBP_LO+LON` (PF 1.28, n=442). **E3 is the birth of Family A.**
- **R053 — Frozen forward validation** — WATCHLIST #1 (`BBW_LO+RV_LO+DST_NR+PRG_HI`):
  **REJECT** on the last 20% (untouched) of data.
- **R054 — Forward validation of the actual PROMOTE trio** (E2/E3/E8 + portfolio):
  **E3 survives best** (others degrade) → E3 becomes the candidate.

*(R055 was not used.)*

## Phase 8 — Regime shift, E3.1, timestamp fix & families (R056–R063) · 2026-07-30

- **R056 — E3 regime-shift forensics** — why did E3 collapse in forward folds F3/F4?
  Structural **volatility regime shift** (best hypothesis 50/100): win-period PF 3.49 →
  lose-period PF 0.32. Remain frozen; one final research phase mandated.
- **R057 — Single macro-regime filter** — of 6 candidate filters, **BBW_STRICT**
  (BB-width < IS p25) is best (score 84.6, PF 2.10) → freeze **E3.1** for a forward test.
- **R058 — Independent structural edge (E4) search** — new discovery constrained to be
  uncorrelated with E3: `DST_MD` appears in 3/4 survivors; nothing promoted.
- **R059 — DST_MD trend-continuation family** — exhaustive `DST_MD`-anchored search:
  9 WATCHLIST candidates → a real trend-continuation family exists.
- **R060 — Dual mandate** — forward-test **E3.1_v2** (`BBW_STRICT+RV_LO+DST_NR+PRG_VH`;
  criteria met 5/6 → confirmed) **and** build the DST_MD portfolio.
- **R061 — Reality check** — 5-attack-vector stress battery (parameter robustness grid,
  independence, redundancy, overfit probes) on E3.1_v2.
- **R062 — Expanded universe + demo-bot spec** — screens ~140 OKX symbols
  (`r062_universe.csv`), validates E3.1_v2's edge at scale, and produces a paper-trading
  spec frozen on a **46-symbol** universe (`r062_demo_bot_spec.md`, `bot_config.yaml` —
  status: **READY FOR PAPER TRADING**); `demo_bot.py` + SQLite begin.
  *(`quantlab_r062_download.py` expanded the symbol cache.)*
- **R063 — Signal-funnel autopsy** — why E3.1_v2 only fires ~79 forward trades
  (frequency bottleneck anatomy). No optimisation.

## Phase 9 — Families A/B/C and the timestamp-bug crisis (R064–R072) · 2026-07-30 → early Aug

- **R064 — Full-cache structural mining** — 32-condition library on the entire cache →
  champion family `RV_HI+DST_MD+ADX_WK+LON` (PF 2.19, UES 94.8, **0% overlap with E3.1**).
- **R065 — Forensic of the R064 champion** — WATCHLIST (6/7, generalisation 56.3);
  E3.1 still stronger → champion not promoted.
- **R066 — Production portfolio validation** — three frozen families:
  **Family A** (E3.1), **Family B** (`RV_HI+DST_MD+ADX_WK+LON`), **Family C**
  (`DST_NR+ADX_ST+PBD_HI+ASI`) → Family A top score 96.3.
  ⚠️ *Later shown (EXIT_MODEL_AUDIT) to use a next-close proxy exit.*
- **R067 — Family C dissection** → ADOPT `ADX_ST+PBD_HI+ASI` (PF 1.69, n=2049)…
- **R067b — Overfitting audit → ⚠️ TIMESTAMP BUG.** Parquet indexes were sequential
  integers, not timestamps → `hour == 0` for every bar; **ASI/LON session conditions were
  broken**; Family B results meaningless; "3-condition" tests were actually 2-condition.
- **R068 — Simplified Family C validation** — `ADX_ST+PBD_HI` alone (walk-forward,
  bootstrap, MC 10k, LOO): **real edge; cleared for the demo bot** (8/8 checklist).
- **R069 — Full re-evaluation on real UTC timestamps** — Family A stands; Family C
  (ADX+PBD) PF 1.69 / n 2049; **Family B only 26 trades → NOT ready**. A & C cleared.
- **R070 — Production stress test** — monthly stability, RR sensitivity, losing streaks,
  edge decay: A PF 3.35 (score 85), C PF 1.69 (score 97); **edge-decay warnings on both**.
- **R071 — Full RR bootstrap** — Family A: keep **RR 2.0** (no significant gain higher);
  Family C: upgrade to **RR 3.0** (P=100%, CI [+0.59, +1.11]).
- **R072 — Structural forensics** — entry anatomy, win/loss clusters, symbol/regime/time
  forensics, exit replay, edge attribution. Final verdict said "deploy both", **but its own
  realistic Key Stats contradicted that** (A: PF 0.53 / WR 20.9%; C: PF 0.60 / WR 16.7%).
  → Resolved by the exit-model audit below.

> **2026-08-03 — repository re-initialised on GitHub** (fresh commit chain; README first
> drafted). Research resumes under the corrected engine.

## Phase 10 — Exit-model correction & the retail lock-in (R073–R081) · 2026-08-06/07

- **⚠️ `EXIT_MODEL_AUDIT.md` (2026-08-06, stop-the-presses)** — R066–R071 "wins" were
  computed as `next bar close > entry close` (**a proxy, not SL/TP**). Under the
  bot-faithful SL/TP engine: Family A ≈ PF 1.19 (bot-faithful) / 1.71 (optimistic entry),
  **Family C is unprofitable at every realistic exit**. R072's deployment verdict is not
  supported by its own stats.
- **R073 — Real-edge hunt on the corrected engine** — bot-faithful rolling walk-forward
  with real SL/TP. **Family A → GO** (E6 signal-bar-close entry, PF 1.50). **Family C →
  NO-GO** (PF < 1 at all exits/costs). Family A is real but thin; C is dead.
- **R074 — Edge refinement, strict holdout** (selection ≤ 2025 decides; 2026 untouched
  confirm) → **E6_sigentry + RR 2.0 + VolCeil(atr_rank≤70): PF 1.63, ~10 t/mo, MDD −26%** —
  real, but **not retail-friendly** (41% profitable months, 7-month losing streak).
- **R075 — Retail-friendly family hunt** — 4 new generic families (trend-pullback,
  mean-rev, breakout, ORB) all unprofitable; Family A FINAL still best (RetailScore 48.4 vs
  32.2). *Edge is scarce — no generic strategy beats it.*
- **R076 — Market-timing overlay & cross-sectional RV** — **breadth50 overlay adopted**
  (trade Family A only when >50% of the 52 symbols are above EMA20): **PF 2.15, MDD −10.5%**,
  55% profitable months, holdout PF 1.25. Cross-sectional relative value: **no edge — rejected**.
- **R077 — 🏁 FINAL LOCKED CONFIG (crypto 1H pre-ML)** — Family A + E6 + **RR 1.5** +
  VolCeil + breadth50 → **PF 2.05, WR 57.8%, MDD −9.2%, ~5.8 t/mo, 65% profitable months,
  worst month-streak 3, Boot P5 1.55, LOO 1.94**. $100 acct @1% risk: max DD ≈ −$10.50.
  15m: no edge; 1m: unusable (broken timestamps).
- **R078 — Symbol forensics & unseen universe** (`scripts/r078_*`) — no reliably negative
  symbols (don't prune). Locked edge on 8 brand-new symbols: **PF 0.63 / 17 trades → the
  edge is universe-specific**; expand only via per-symbol validation.
- **R079 — Frequency expansion sweep** — no free lunch, but a cheap one exists:
  V03 (breadth 0.40) → 5.1 t/mo (+19%) at PF 1.79, holPF 1.32; BASE (4.3 t/mo, PF 2.05)
  safest.
- **R080 — Ten clean new high-frequency hypotheses** — ALL fail, but **raw Family A**
  (E6+RR1.5, no filters) = **14.9 t/mo, PF 1.48** — the frequency dial is breadth/volceil,
  not the signal.
- **R081 — Scalp expansion (RR < 1)** — 80% WR at RR 0.4 is a **cost mirage** (breakeven
  fee ~0.06%/side; dies at realistic 0.08–0.10%); profitable-months ceiling ≈ 65% on this
  data — 70% unreachable by rules alone.
- **R081b** — mild filters (volceil/breadth) on the winning scalp config to push
  profitable-months.

## Phase 11 — ML entry filters (R082–R088) · 2026-08-07

- **R082 — New dimensions** — daily/4H/ADX/breadth gates, **walk-forward logistic-regression
  entry filter**, BE-trail exit, on raw Family A (14.9 t/mo): **ML filter is the best new
  edge** (F6 ≈ 7.7 t/mo, 64% prof-mo, holPF 1.47).
- **R083 — ML keep-rate refinement** (q 0.35/0.45/0.55 + regime combos) → **q55: 8.0 t/mo,
  PF 2.34 (1.97 @cost), 64% prof-months, holPF 1.47** on the 52-symbol universe.
- **R084 — ML on the expanded 73-symbol universe** → **🎯 hits the retail spec:
  9.3 t/mo, PF 2.11 (1.78 @cost), 71% profitable months, worst streak 2, holPF 1.42.**
  Transfer test: ML alone doesn't transfer to new pairs; gain is mostly the original 52.
- **R085 — Upgraded ML** (rich features, confidence sizing, gradient boosting) → none beats
  the base filter; **base ML q55 on 73 stays champion** (71%, 9.3/mo, holPF 1.42).
- **R086 — ML-type zoo** (LR/RF/**SVM**/MLP/NB/ensemble on identical lean features) →
  **SVM (RBF) is the new champion: 9.2 t/mo, 71% profitable months, PF 2.23 (1.87 @cost),
  holPF 1.48.**
- **R087 — SVM keep-rate sweep** → **q0.75 sweet spot: 10.4 t/mo, 71% profitable months,
  PF 1.94, holPF 1.36** (q0.85–0.95 add ~1 t/mo but drop prof-months to 64%).
- **R088 — Dollar P&L feasibility** (`scripts/r088_dollar_pnl*.py`, no hypothesis test) —
  $100 account, $2 risk/trade, SVM q0.75 vs q0.55, fixed vs 2% compounding → USD drawdown
  and PnL scenarios for the champion.

## Phase 12 — The 5-minute hunt: seven proofs of "no edge" (R089–R095) · 2026-08-07/08

5m candles for BTC/ETH/DOGE/LINK/LTC (plus 1m/15m in places), retail costs 0.05%/side,
strict selection/holdout.

- **R089 — 5m hunt #1** — raw Family-A port + SVM on 5m → **NO EDGE** (raw PF 0.59,
  holPF 0.50, WR 28%). The edge is 1H + broad-universe specific.
- **R090 — Fresh 5m hypotheses** (momentum-burst, range-fade, ORB-5M, trend-pullback,
  vol-burst) → H2 RANGE-FADE was **reported as a validated 5m edge, then RETRACTED the same
  day: lookahead bug** (day-low included future bars). Corrected: PF 0.91 → no edge.
  **Lesson: causal audit is now permanent pipeline step.**
- **R091 — New-indicator 5m hypotheses** (session VWAP, StochRSI, MACD, Keltner, Donchian,
  %B) — all causal-audited, all fail (holPF@cost ≈ 0.36). 3rd independent 5m attempt.
- **R092 — 5m win/loss forensics** (`scripts/r092_5m_forensics.py`) — no discriminating
  entry feature (max Cohen's d 0.11); hour-cluster gross edges collapse at cost. 4th proof.
- **R093 — Bank-style 5m liquidity hypotheses** (prev-day-low reclaim, VWAP deep-discount,
  session-low accumulate…) — all audit-pass, all fail the cost gate. 5th proof.
- **R094 — 5m combination sweep** — 10 pre-registered combos of bank signals + filters —
  best is exact breakeven (holPF@cost 1.00). 6th proof.
- **R095 — Advanced ML on 5m** (LR/SVM/GB on simple indicators) — raw pool PF 1.00 = no
  edge for ML to amplify; all holPF@cost 0.43–0.46. **7th proof → 5m CLOSED.**

### 🏁 CRYPTO FINAL VERDICT (2026-08-08) — as of that date

- **5-minute crypto: NO EDGE** — 7 independent proofs (later re-confirmed as a negative
  control on the `blind-validation` branch: `control_5m.py` correctly fails). The 5m bar
  is too small vs retail costs.
- **1-Hour crypto: VALIDATED EDGE** — SVM q0.75 on 73 symbols (≈10.4 t/mo, PF 1.94,
  ~70% profitable months, holdout-validated, cost-surviving). **⚠️ This exact config was
  later re-tested and refined by the blind-validation campaign (T-series, below): q0.65 +
  regime-gated VolCeil → then the T25/T27 MR+trend portfolio is the current answer.**

---

# 📗 The full research log — forex

**Universe:** 8 majors (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD, EURGBP).
**Data:** 1H ~Aug 2024–Aug 2026 + 1D ~10y (yfinance; `scripts/fetch_forex.py`), later 4H by
resampling. **Costs:** per-pair retail spreads (and swap), causal audit built in.
**Protocol:** selection ≤ 2025-08, holdout Aug 2025–2026 untouched.
**Success bar:** holPF@cost > 1.1.

- **F001 — First forex hypotheses** (London-breakout, trend-pullback, range-meanrev,
  VWAP-reclaim, momentum-burst, crypto Family-A transfer) — all audit-pass, **all fail**
  (holPF@cost 0.66–0.75). Lead: **X5 London-expansion** (gross holPF 1.19).
- **F002 — Daily-TF context + walk-forward ML** → **first holdout-validated GROSS forex
  edge: daily-trend + ML-SVM, holPF 1.12, 71% prof-months** — but retail spreads still eat
  it (0.84 @cost).
- **F003 — RR sweep** (1.0→3.0) on ML+daily-trend → higher RR helps (holPF@cost 0.74→0.92
  @RR3) but 1H spread drag is structural.
- **F004 — 4H timeframe + RR 3.0** → gross edge holdout-validated (holPF 1.11), cost-adjusted
  **0.99** — spread drag halved; ECN spreads (0.2–0.3 pip) would cross 1.1.
- **F005 — The trap hunt** — after a stop-hunt (wick through a level): FOLLOW beats REVERSE;
  best T4 bull-follow @RR3 → **holPF@cost 1.03** (first >1.0).
- **F006 — Comprehensive trap matrix** (bull/bear traps × level type × follow/reject ×
  session × trend, RR3, 4H) → **6 cost-surviving configs (holPF@cost 1.11–1.33)**. Winner
  pattern: **reject the bear trap (long: wick below level → close back above) in a
  downtrend during London.** *(F006b/F006c = fast vectorised matrix + ML trap classifier.)*
- **F007 — HARD VALIDATION of the winner** → **✅ VALIDATED: holPF@cost 1.14, boot P5 1.06,
  LOO floor 1.09, MC P(profit) 100%.** Edge is holdout-concentrated; ML trap classifier
  finds no general follow/reject edge. Strongest forex config so far.
- **F008 — 5m scalp probe (both markets)** — cost-dead everywhere (avg ~0.95R/trade of
  costs at 5m); **crypto S2 RSI2-fade shows a real GROSS edge (holPF 1.18)** → retest at 1H.
- **F009 — RSI2-fade at 1H** — cost/trade cut 6× (0.15–0.18R) but **still not
  cost-surviving** (holPF@cost 0.78–0.87). Thin edge; next steps (4H, or combine with the
  bear-trap filter) parked by the freeze.

### ❄️ STRATEGY FREEZE (2026-08-09) — end of the `main` arc

Trading paused to build capital; the freeze configs (table at top) were frozen *as of that
date*. The freeze's standing order — *don't tweak frozen strategies* — is why the next
campaign ran on a **separate branch** (`blind-validation`), which then **re-tested and
superseded** part of the freeze. See `.agents/memory/quantlab-research-state.md` and
`.agents/memory/MEMORY.md` for the standing orders.

---

# 📘 Post-freeze — the blind-validation branch (T1–T34)

> **Branch:** `blind-validation` (forks at the freeze commit `594c972`, 2026-08-31 →
> 2026-09-03). **Canonical files there:** `README.md` (master table), `FINDINGS.md`,
> `TEST_LOG.md`, `RESEARCH_DERIV.md`, `scripts/svm_deploy.py`, all `*.py`/`*.log` tests.

**Motivation:** the frozen QUANTLAB configs had *never* been re-tested year-by-year on
unseen data with costs. This campaign did exactly that — train on all prior years, test
each next year **standalone** (2023 → 2024 → 2025 → 2026), fees 0.05%/side, universe
crypto USDT perpetual swaps 1H — and hunted for a config that **survives every year**,
with a 2027-01-01 go-live in mind.

## What the blind re-test falsified

- **Raw Family A / Family C bot strategies** (as deployed): PF@cost ≈ 1.0–1.1 — weak;
  Family C even negative on a full year.
- **R077 "holy grail"** (breadth50 + static VolCeil): **PF@cost 0.75 → loses money after fees.**
- **Every static (q × VolCeil × breadth) sweep**: no config survived all of 2024/25/26 —
  a fixed VolCeil=70 fixes 2024 but breaks 2026 (opposite regimes).
- **SVM q0.75** (the Aug-8 "champion"): PF@cost ≈ 1.25 on 2026, 75% profitable months —
  good, but **q0.65 is better**.
- 4H environment: only 56 trades / 3.5 yr → edge is 1H-specific.

## The winners

| Test | Idea | Result |
|---|---|---|
| T13 | MR, **adaptive (regime-gated) VolCeil** — apply the ATR-spike skip only when abs(ema_dist_pct) > 2.0 | PF **1.18 / 1.82 / 1.53** (2024/25/26), survives every year |
| T24 | Condition-aware ML (SVM q0.65) MR champion | PF 1.58 full; learned reasons = ADX/breadth/ema_dist/atr_rank (confirms adaptive VolCeil) |
| **T25** | **Condition-aware TREND** — Donchian breakout + ADX>20 & above-EMA200, RF keeps top-65% of setups | **PF 1.744 / 1.529 / 1.537** (2024/25/26), DD ≈ −2% → bulletproof |
| T26 | MR+Trend 50/50 portfolio | PF 1.21 / 1.33 / 1.81 (2024 slightly under 1.3 — MR drag) |
| **T27** | **MR+Trend 70/30 portfolio** (trend-weighted) | **PF 1.40 / 1.32 / 1.81, MAX DD < 1% → the recommended deployable vehicle** |

**T25/T27 detail (from FINDINGS.md / branch README):** the MR leg is SVM q0.65 + VolCeil
gated by regime; cumulative ≈ **+71%** (fixed risk; +54.7% at 2% equity), max DD ≈ −9.4%
at 1% risk/trade, 15/27 profitable months, **22/30 symbols profitable** (top-3 carry only
37% of R → broad, not concentrated). T25's RF learned strong-trend conditions
(rsi14 + ema_dist_pct) and is complementary to MR → hence the portfolio.

> ⚠️ **T34 (2026-09-04) supersedes the T25/T27 trend conclusions** — the trend leg's RF
> gate is **exit-anchored** (it reads exit-bar features); re-run at the true entry bar the
> trend edge is ≈ PF 1.03 (no edge). The MR numbers above are entry-anchored and stand,
> with the full-universe corrections (2024 loses, PF 0.74; max DD @1% = −28%, not −9.4%)
> and the small-account recommendation (MR, TP 2R, 1% risk → $100→$204). See the
> correction block in the status header.

## The full master test table (T1–T34)

| # | Idea | Result |
|---|-------|--------|
| T1 | 1H MR (Family A) | 2026 loses |
| T2 | 1H MR expanded families | 2026 loses |
| T3 | 1H MR alt exits | FAIL |
| T4 | 1H MR multi-family | FAIL |
| T5 | 1H MR tight params | FAIL |
| T6 | 1H MR time-of-day | FAIL |
| T7 | 15m (too sparse) | — |
| T8–T11 | 1H MR variants | FAIL |
| T12 | 1H MR + momentum filter | 2026 loses |
| **T13** | **1H MR adaptive VolCeil** | **PF 1.18 / 1.82 / 1.53** |
| T14 | 1H Donchian trend | FAIL |
| T15 | 4H Donchian trend | FAIL |
| T16 | 1H MR+Trend portfolio | FAIL |
| T17 | 1D trend | strong but sparse |
| T18 | Multi-TF portfolio | FAIL |
| T19–T23 | CAGE false-breakout | FAIL |
| **T24** | **Condition-aware ML (SVM)** | **PF 1.58** |
| **T25** | **Condition-aware TREND (Donchian + RF q65)** | **PF 1.744 / 1.529 / 1.537 — ⚠ exit-anchored, see T34** |
| T26 | MR+Trend 50/50 | PF 1.21 / 1.33 / 1.81 |
| **T27** | **MR+Trend 70/30 portfolio** | **PF 1.40 / 1.32 / 1.81, DD < 1% — ⚠ trend leg exit-anchored, see T34** |
| T28 | 15m condition-aware | PF 1.58 / 1.75 / 0.94 (noisy) |
| T29 | 15m broad universe (73 syms) | PF 1.58, 23/38 profitable (7-mo window) |
| T30/A | Cross-sectional momentum | FAIL (80% DD) |
| T30/B | Trailing-exit trend | worse than T25 |
| T30/C | Unified MR+trend model (one RF) | PF 1.50 |
| T30/D | BTC–ETH pairs (spread z-score) | FAIL (100% DD) |
| T31 | Combined 3-strategy portfolio (MR+Trend+Unified) | PF 1.21 (< T27) |
| T32 | Deriv reach + forex-15m binary (crypto method) | synthetics = random walk; forex faint; **data-capped** |
| T33 | Forex 1H spot (T25 pipeline, via Deriv) | **PF 1.099 ≈ break-even; MR silent → crypto edge does NOT transfer** |
| T34 | Small-account leverage × R:R ($100; implementability audit) | MR TP2R @1% risk → $100→$204, DD −37%; **trend gate exit-anchored → NOT implementable** |

## Sub-branch probes (the exploration tree)

- **Main branch (crypto 1H) — the real edge:** T25 / T27 above. Validated walk-forward on
  2024/25/26, 20–73 symbols.
- **Sub-branch A — crypto 1H ideas that failed:** T1–T12 MR families; T14/T15 1H/4H
  Donchian trend; T19–T23 CAGE false-breakout (T23 refined → catastrophic PF 0.51,
  MDD −51%); T30/A momentum rotation; T30/B trailing exit; T30/D BTC–ETH pairs; T31
  combined portfolio.
- **Sub-branch B — crypto 15m (data-blocked):** T28/T29 showed a real but noisier edge
  (PF 1.58 on a 7-month window, ~6× the 1H trade count), but a full 3-yr 15m fetch failed:
  OKX throttles 15m under parallel load and Binance/Bybit/Yahoo are unreachable from the
  sandbox → 15m stays a proof-of-concept.
- **Sub-branch C — Deriv binary options:** reachable (WS `wss://ws.binaryws.com`,
  app_id 1089). Volatility/step/jump indices are engineered **random walks** (50%
  accuracy → no edge; binaries need >55%). Forex 15m showed faint >55% on ~35-day slices,
  but Deriv's free API caps 15m history → cannot validate deeply. See `RESEARCH_DERIV.md`.
- **Sub-branch D — forex 1H spot (T33):** T25/T27 pipeline on forex 1H (Deriv, ~1 yr/pair)
  → **PF 1.099 ≈ break-even; MR = 0 signals**. The crypto edge does **not** transfer to
  forex. *(Independent of the `main` F001–F009 bear-trap line, which used yfinance data.)*
- **Sub-branch E — stocks/equities:** no equity data in the sandbox (OKX is crypto-only;
  Yahoo/Binance blocked) → dead end.

## The honest bottom line (branch README, updated 2026-09-04)

Every reachable trading type was tested — crypto 1H (MR, trend, CAGE, momentum, pairs,
unified, portfolios), crypto 15m, Deriv synthetics, forex 15m binary, forex 1H spot, and
attempted stocks. **The one implementable, deployable edge is the 1H MR leg** (SVM q0.65
adaptive VolCeil). For a small account (T34, $100): **TP 2R / SL 1R, 1% risk/trade →
~+104% over ~2.6 yr**, near-zero ruin, accepting a −37% drawdown in year one-to-two. The
**T25 / T27 trend "PF≥1.3, DD<1%" numbers are exit-anchored and superseded** (entry-only
re-run ≈ PF 1.03) — do not size a live account on the trend leg as published. Everything
faster or on other instruments is **data-blocked or edge-less** in that environment.

---

# Key lessons learned

1. **Small samples lie.** R004/R019/R029/R065 repeatedly: thin-sample PROMOTEs died at scale.
2. **High PF alone is meaningless.** The pre-audit PF 3.35 (Family A) was a *proxy artifact*;
   real SL/TP exits gave ≈1.2–1.5.
3. **Exits are part of the strategy.** A different exit engine (EXIT_MODEL_AUDIT) changed
   every conclusion from R066–R072.
4. **Timestamps/data hygiene is a research risk.** The R067b integer-index bug silently
   broke all session filters; the R090 lookahead bug created a phantom "edge".
5. **Cost gates are decisive.** 5m strategies "look" profitable gross and die at 0.05% costs —
   proven 7 times.
6. **Edges are universe- and timeframe-specific.** Portfolio C overfit its symbols (R044);
   E06+E11 didn't transfer (R049); the 1H edge didn't transfer to new pairs (R078); the
   crypto edge didn't transfer to forex 1H (T33, PF 1.099).
7. **ML amplifies edge; it cannot create one from a fair coin** (R095 vs R084–R087), but a
   *condition-aware* ML filter can sharpen a real signal (T24/T25: RF/SVM learned the exact
   conditions — ADX, breadth, ema_dist, atr_rank — the human sweeps had been chasing).
8. **Robustness > optimisation.** Environment is the edge; entry family matters far less
   (R038). And a strategy is only real if it survives **every year** on per-year holdouts —
   static filters that fix one regime break the next (T13/FINDINGS: the fixed VolCeil died;
   the regime-gated VolCeil survived).
9. **Failed experiments are findings.** Most of the above negative results *are* the value of
   this repository — and the "validated" freeze config itself was later falsified by blind
   re-testing (R077 → PF@cost 0.75), which is the strongest demonstration of the method.
10. **Have a freeze — and re-test it.** The arc pauses (freeze), continues on a separate
    branch, and lets the later evidence supersede the earlier conclusions (SVM q0.75 → SVM
    q0.65-adaptive → T25/T27). Conclusions carry a date and a branch.

---

# Technologies

Python · pandas · NumPy · scikit-learn (LR/RF/SVM/GB/MLP) · Matplotlib · PyArrow/Parquet ·
SQLite · APScheduler · OKX market data · yfinance · Deriv WebSocket API ·
bootstrap methods · Monte Carlo simulation

*(Main branch: crypto OKX + forex yfinance. `blind-validation` branch: OKX crypto +
Deriv binary/forex — Binance/Bybit/Yahoo were blocked in that sandbox.)*

---

# Disclaimer

This repository is for **educational and quantitative research purposes only**. Nothing here
is financial advice or a recommendation to trade any instrument. Past performance — real or
backtested — does not guarantee future results. The demo bot never executes live trades.

---

# Author

**Ojeology** — Independent Quantitative Researcher
GitHub: https://github.com/ojeology

*"The goal isn't to find a strategy that looks good. The goal is to find one that survives
every attempt to prove it wrong."*
