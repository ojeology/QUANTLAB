---
name: QuantLab Research State
description: Current frozen baselines, promotion status, and research trajectory for the QuantLab algo research project
---

## R090 FRESH 5m HYPOTHESES (2026-08-07) — 🎯 H2 RANGE-FADE IS A VALIDATED 5m EDGE

User wanted NEW 5m hypotheses (not the 1H port). Tested 5 setups built FOR 5m:

| Hyp | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | holPF |
|---|---|---|---|---|---|---|---|---|
| **H2 RANGE-FADE** | **107** | **17.8** | **76%** | **4.67** | **2.24** | **-4.4%** | **100%** | **4.10** |
| H5 vol-burst | 1685 | 281 | 43% | 1.12 | 0.62 | -29.5 | 60% | 1.18 |
| H1 momentum-burst | 3871 | 645 | 41% | 1.02 | 0.37 | -55.3 | 60% | 1.02 |
| H3 ORB-5m | 11557 | 1926 | 39% | 0.97 | 0.35 | -98.8 | 40% | 0.97 |
| H4 trend-pullback | 6354 | 1059 | 39% | 0.97 | 0.32 | -93.2 | 40% | 0.93 |

**H2 RANGE-FADE (the winner):**
- Setup: buy when price is within 0.2*ATR of the DAY's low + RSI14<30 (oversold at the
  daily extreme) + green candle (reversal confirmed). Exit: SL 1ATR / TP 1.5ATR, 60-bar
  (5h) time stop.
- 107 trades, 76% WR, PF 4.67 (2.24 @0.05% cost), MDD -4.4%, 100% profitable months
  (5/5), worst losing streak 0.
- Holdout (Jun-Aug untouched): n=56, **holPF 4.10** — survives out-of-sample.
- Per-symbol: ALL 5 positive (LTC PF 11.25, LINK 7.88, BTC 4.25, DOGE 3.43, ETH 2.57).
- By month: all 5 months positive (Apr 9.0, May 4.67, Jun 10.0, Jul 1.75, Aug 99).
- 81 TP / 26 SL (76% hit rate).

**This is the 5m edge the user asked for — first time the 5m scale works.**
Key difference from other 5m setups: it's mean-reversion at the daily extreme
(fading the day's low), not a trend-follow. The engine was long-only — a short-side
fade of the day's high may add more (future work). Small universe (5 symbols) and
~4.5 months of data = needs more data/live confirmation, but the signal is robust
across every symbol and month.

## R089 5-Minute Edge Hunt (2026-08-07) — ❌ NO EDGE ON 5m

Fetched 5m data for 5 majors (BTC, ETH, DOGE, LINK, LTC), ~55k bars each
(2026-01-28 → 2026-08-07). Scaled the validated 1H machinery to 5m
(IS_LOOKBACK=6000 bars=500h, RECAL=2016 bars=7d).

**Result: the compression-pop signal LOSES money on 5m.**
- RAW Family A on 5m: 57 trades (9.5/mo), WR 28%, **PF 0.59**, PF@0.05% cost 0.18,
  MDD -17%, prof-mo 29%, selPF 0.63, **holPF 0.50** — negative edge, and costs destroy it.
- SVM filter couldn't be trained: only 57 trades on 5 symbols (< min_train 300).
- Consistent with 15m test (R077): the edge does NOT transfer to lower timeframes.
  It lives on 1H with a broad universe and deep history only.

**Pattern across timeframes (honest):**
- 1H, 52-73 symbols, 2.5 yrs → edge FOUND & validated (SVM q0.75: 10.4 t/mo, PF 1.94)
- 15m, 8 symbols, 6 mo → no edge (too sparse, PF 0.80 on 7 trades)
- 5m, 5 symbols, 6 mo → no edge (PF 0.59, costs kill it)
- 1m → unusable (broken timestamps in cache)

**Verdict: 5m is closed. The validated edge is a 1H, broad-universe edge.**
New data files: quantlab_cache/*_5m.parquet (5 symbols). Script: scripts/fetch_5m.py.

## R087 SVM keep-rate sweep (2026-08-06) — more-trades dial mapped

User locked SVM (R086) but wants MORE trades. Swept q (fraction of signals kept):

| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| 0.55 (locked) | 250 | 9.3 | 58% | 2.04 | 1.70 | -17.1 | 69% | 2 | 1.35 |
| 0.65 | 265 | 9.8 | 58% | 2.05 | 1.71 | -17.1 | 64% | 2 | 1.32 |
| **0.75** | **280** | **10.4** | 56% | 1.94 | 1.62 | -17.1 | **71%** | **2** | **1.36** |
| 0.85 | 294 | 10.9 | 55% | 1.84 | 1.54 | -19.4 | 64% | 2 | 1.28 |
| 0.95 | 301 | 11.1 | 55% | 1.84 | 1.54 | -18.2 | 64% | 2 | 1.28 |
| 1.0 (raw) | 453 | 16.8 | 49% | 1.44 | 1.22 | -28.6 | 48% | 3 | 1.28 |

**SWEET SPOT: q=0.75 → 10.4 t/mo (more than locked 9.3) AND 71% profitable months,
PF 1.94, worst streak 2, holPF 1.36.** Relaxing further to q0.85/0.95 adds only ~0.5-1
t/mo but prof-mo drops to 64%. Raw (no filter) = 16.8 t/mo but crashes to 48% prof-mo,
PF 1.44, MDD -28.6%.

**Recommendation: SVM q0.75 = the "more trades" version of the locked config**
(10.4 t/mo, 71% prof-mo, PF 1.94). Honest note: prof% wobbles 64-71% between q values
(small-sample noise on monthly buckets), so treat 71% as "high-60s to low-70s".

## R086 ML-TYPE zoo (2026-08-06) — SVM is the new champion (meets retail spec)

User: don't add more features, try different ML TYPES with a few special features.
Tested 6 walk-forward model types on SAME 14 lean features (11 base + 3 special:
breadth-quartile, dist-to-48h-high, green-streak), top-q=0.55, 73-symbol universe.

| ML type | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|
| LR (old champ) | 8.6 | 60% | 2.22 | 1.87 | -14.1 | 64% | 3 | 1.55 |
| **SVM (RBF)** | **9.2** | 60% | **2.23** | **1.87** | -14.5 | **71%** | **2** | **1.48** |
| RF | 9.7 | 57% | 1.96 | 1.63 | -17.4 | 71% | 2 | 1.36 |
| NB | 8.7 | 58% | 2.10 | 1.74 | -17.4 | 71% | 2 | 1.40 |
| MLP | 8.0 | 60% | 2.22 | 1.87 | -14.5 | 64% | 3 | 1.37 |
| ENSEMBLE | 9.1 | 58% | 2.05 | 1.72 | -17.4 | 64% | 3 | 1.50 |

**NEW CHAMPION: SVM (RBF) — 9.2 t/mo, 71% profitable months, worst losing streak 2,
PF 2.23 (1.87 @0.05% cost), holPF 1.48.** Beats LR on profitable-months (71 vs 64) and
trades (9.2 vs 8.6) with equal PF. RF and NB also pass the retail spec (71% prof-mo) but
with slightly lower PF/holPF than SVM.

**Takeaway: the model TYPE matters more than feature count.** SVM's RBF kernel captures
nonlinearities LR misses, converting ~64% → 71% profitable months without hurting PF or
cost-robustness. New best retail candidate: **ML-SVM q55 on 73 symbols.**

## R085 Upgraded ML (2026-08-06) — the base ML filter is robust; upgrades don't beat it

Tested 4 honest upgrades to R084's ML q55 (9.3 t/mo, 71% prof-mo, PF 2.11):
- B rich features (adds daily/4H trend, cross-sectional rank, multi-bar momentum): PF 2.36
  but prof-mo DROPPED to 50% (removed too many trades from good months)
- C rich + confidence-based sizing (risk ∝ P(win)): no improvement (prof-mo 50%)
- D gradient boosting (HistGB, rich features): **79% profitable months!** PF 2.47 (2.06@cost),
  worst streak 2, but t/mo 7.7 (just under 8 target) and selection→holdout PF drop 5.39→1.34
  (bigger overfit gap than base 3.71→1.40, holPF 1.34 < 1.42)
- E rich + daily-trend overlay: PF 4.85 but only 2.0 t/mo

Feature importance (rich LR): d_trend (0.80), breadth_now (0.71), ema_dist_pct (0.66),
real_vol_20 (0.60) — daily trend & market breadth are the strongest signals, consistent w/ R082.

**VERDICT: A_base_ref (R084 ML q55 on 73) remains the champion — it survives every upgrade
attempt.** Honest finding: your gut was right that more was possible, but the tested upgrades
either hurt monthly consistency (rich features), added overfit risk (gboost), or cut trades
too much (daily-trend). The one to WATCH is D_gboost (79% prof-mo) if user accepts 7.7 t/mo
and the larger selection→holdout gap (needs live confirmation).

## R084 ML on Expanded Universe (2026-08-06) — 🎯 HITS THE 70% PROF-MONTHS TARGET

Fetched 3 more pairs with real history (XAG, ALLO, AAOI) → cache = 73 symbols.
Ran the ML q55 filter on the expanded universe.

| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|---|
| A_ML52 (ref) | 216 | 8.0 | 61% | 2.36 | 1.99 | -17.0 | 64% | 2 | 1.50 |
| **B_ML73 (expanded)** | **250** | **9.3** | 58% | **2.11** | **1.78** | -16.3 | **71%** | **2** | **1.42** |
| C_ML transfer to new pairs | 30 | 1.1 | 47% | 1.31 | 1.03 | -5.4 | 57% | 2 | 1.00 |
| D_raw73 | 453 | 16.8 | 49% | 1.44 | 1.22 | -28.6 | 48% | 3 | 1.28 |

**KEY: B_ML73 is the FIRST config to hit the user's retail spec — ≥70% profitable
months (71%) AND ≥8 t/mo (9.3), worst losing streak 2, PF 2.11 (1.78 @0.05% cost),
holPF 1.42.** Expanding the universe from 52→73 with the ML filter improved
profitable-months 64%→71% at higher frequency.

**Honest caveats:**
- Transfer test (C): ML filter trained on 52, applied to new pairs only → PF 1.31,
  holPF 1.00 — does NOT clearly transfer. The gain in B comes mostly from the 52;
  new pairs add modest diversity (~34 extra trades), and 71% vs 64% may be partly
  small-sample noise on the extra trades.
- Universe limit reached: OKX serves deep 1H history only for older instruments;
  most of the remaining 400+ liquid swaps have <1000 bars (recent listings).

**FINAL CANDIDATE: ML q55 on the 73-symbol universe (B_ML73) = 9.3 t/mo, PF 2.11,
71% prof-months, worst streak 2, holdout-validated.**

## R082/R083 Multi-TF + ML filters (2026-08-06) — 🎯 ML FILTER IS THE BEST NEW EDGE

User wants: ≥70% prof-months, ≤2-3 bad months, MORE trades. Tested NEW dimensions:
daily/4H trend filters (resampled from 1H), daily ADX, daily breadth, ML entry filter,
BE+trail exit.

**Results (base = RAW Family A, 14.9 t/mo, PF 1.49):**

| Config | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |
|---|---|---|---|---|---|---|---|---|
| RAW base | 14.9 | 50% | 1.49 | 1.27 | -31 | 48% | 3 | 1.45 |
| F1 daily-trend | 4.7 | 59% | 2.16 | 1.87 | -8.3 | 57% | 3 | 2.00 |
| F4 daily-breadth | 3.7 | 64% | 2.71 | 2.38 | -11.9 | 58% | 4 | 3.00 |
| **F6 ML q50** | **7.7** | **60%** | **2.26** | **1.91** | -17.4 | **64%** | **2** | **1.47** |
| **ML q55 (R083)** | **8.0** | **61%** | **2.34** | **1.97** | -17.4 | **64%** | **2** | **1.47** |
| ML50+dailytrend | 1.8 | 85% | 8.79 | 7.47 | -3.9 | **70%** | 3 | 2.00 |
| ML50+dailybr | 1.7 | 93% | 21.5 | 18.2 | -1.0 | **100%** | 0 | 3.00 |

**KEY FINDING: the ML entry filter (walk-forward logistic regression on the RAW
signal, keep top-q by predicted P(win)) is the best new edge found — it nearly meets
the retail spec: 8.0 t/mo, PF 2.34 (1.97 after costs), 64% prof-months, worst losing
streak 2, holdout PF 1.47.** First config ever with ≥8 t/mo AND PF>2 AND cost-robust
AND holdout-validated.

**The 70% prof-months target remains just out of reach at high frequency** — ML+daily
filters hit 70-100% prof-months but only 1.7-1.8 t/mo (too few trades). Honest trade-off:
- **ML q55**: 8 t/mo, 64% prof-mo, worst 2 — the best high-frequency retail profile
- **ML+dailytrend**: 1.8 t/mo, 70% prof-mo, worst 3 — the best safety profile

F7 BE+trail exit on raw: no help (PF 1.19). F3 daily-ADX: fails holdout. F2 4H-trend:
too few trades, dies at cost.

**ML caveat (honest):** walk-forward, threshold from selection only, holdout-confirmed —
but 208-215 trades is modest; ML adds an overfit risk layer vs pure rule filters. Treat
as promising, needs live confirmation.



User wants: ≥70% profitable months, ≤2-3 bad months, MORE trades. Tested the untested
dimension: **sub-1R take profits (RR 0.4/0.5/0.6/0.75)** across 5 signals × {base, time6}.

**Raw results (gross, no costs) look amazing:**
- **Family A raw + RR 0.4: WR 80%, PF 1.60, MDD -8.2%, 14.3 t/mo, prof-mo 62%, worst streak 2, holPF 1.52** (holdout-validated!)
- Family A raw + RR 0.5: WR 77%, PF 1.63, MDD -11.9%, 14.3 t/mo, holPF 1.69
- Other signals (breakout/micro-scalp/RSI2/EMA20) all die or lose.

**BUT the cost wall kills it (CRITICAL for retail):**
| cost/side | rr0.4 PF | rr0.5 PF | rr0.6 PF |
|---|---|---|---|
| 0.00% | 1.63 | 1.66 | 1.66 |
| 0.05% | 1.12 | 1.21 | 1.26 |
| 0.08% | 0.86 | 0.98 | 1.06 |
| 0.10% | 0.70 | 0.84 | 0.93 |
Breakeven ≈ 0.06-0.07% per side (rr0.4), 0.07-0.08% (rr0.5), 0.08-0.09% (rr0.6).
**Realistic OKX taker+slippage ≈ 0.08-0.10% → scalps DIE.** Only viable with maker-only
fills at ≤0.05% AND minimal slippage — hard for a retail $100 trader.

**Also: profitable-months ceiling.** Best achieved = 65% (rr0.4 + breadth40 filter), NOT
70%. On this data the ceiling for a holdout-validated edge is ~60-65% profitable months;
70% is a target we could not reach in any config across R073-R081.

**Honest menu (all holdout-validated):**
- LOCKED (RR1.5+breadth50+volceil): 4.3 t/mo, PF 2.05, MDD -9%, prof-mo 65%, holPF 1.40 — survives 0.10% costs (breakeven 0.145%)
- MID (RR1.5+breadth40): 5.1 t/mo, PF 1.79, MDD -14.5%, holPF 1.32
- RAW (RR1.5, no filters): 14.9 t/mo, PF 1.48, MDD -31%, prof-mo 48%, holPF 1.45
- SCALP (RR0.4 raw): 14.3 t/mo, 80% WR, PF 1.60 gross / 1.12 @0.05% / 0.86 @0.08% — ONLY if maker fills; prof-mo 62%, worst streak 2

**Verdict for user:** 70% profitable months + high frequency + cost-surviving edge = not
found. The scalp illusion (80% WR) collapses under realistic fees. The locked config
remains the most cost-robust; RAW is the frequency option with the drawdown cost.



User wanted more trades (3-4/mo too small). Tested 10 clean NEW hypotheses (trend-pull v2,
breakout, compression-pop wide, oversold-comp bounce, ADX ignition) with strict
selection/holdout. **ALL 10 FAILED (selPF<1.4 or holPF<1.05)** — no free lunch at high
frequency, generic setups still lose.

**BUT the test surfaced the real answer:** the RAW Family A signal (E6 + RR1.5 + base exit,
NO breadth50, NO volceil) is already validated at HIGH frequency:

| Config | t/mo | PF | MDD% | prof% mo | holPF | holMDD% |
|---|---|---|---|---|---|---|
| **RAW Family A (no filters)** | **14.9** | **1.48** | **-31.1** | 48% | **1.45** | **-19.9** |
| LOCKED (breadth50+volceil) | 4.3 | 2.05 | -9.2 | 65% | 1.40 | -3.9 |

**The safety filters we added in R076/R077 (breadth50, volceil) are exactly what killed
the frequency** (15 → 4.3/mo). The raw signal itself is holdout-validated at 14.9 t/mo,
PF 1.48. Trade-off: MDD -31% (vs -9.2%), profitable months 48% (vs 65%).

**Honest menu for the user:**
- **FREQUENCY option:** RAW E6+RR1.5 = ~15 t/mo, PF 1.48, MDD -31%, holdout-validated.
  Lumpy/big drawdowns (the "professional" profile from R074, just 3.5x more often).
- **SAFETY option (locked):** ~4.3 t/mo, PF 2.05, MDD -9%, 65% prof months.
- **MIDDLE:** R079 breadths (5-6/mo, PF ~1.8).
- No clean new hypothesis beats this — the edge is Family A's, and frequency is a
  dial (filters off = more trades, more risk).

## R079 Frequency Expansion (2026-08-06) — no free lunch, but a cheap one exists

**Question: can we get more trades without killing the edge?**

Swept 14 relaxations of the locked config (breadth threshold, VolCeil, rel_vol gate,
BBW p25→p33, sensible combos). Strict selection/holdout. **Every gain in trades costs
edge — but there's a sweet spot:**

| Variant | t/mo | PF | MDD% | prof% mo | holPF |
|---|---|---|---|---|---|
| **BASE (locked)** | 4.3 | **2.05** | **-9.2** | **65%** | 1.40 |
| V04 br45 | 4.6 | 1.85 | -12.7 | 61% | 1.41 |
| V05 vc80 | 4.9 | 1.83 | -12.0 | 61% | 1.25 |
| **V03 br40** | **5.1** | **1.79** | -14.5 | 58% | 1.32 |
| V02 br35 | 5.3 | 1.65 | -14.5 | 56% | 1.25 |
| V01 br30 | 5.5 | 1.65 | -13.2 | 54% | 1.14 |
| C1 br40+vc80 | 5.7 | 1.64 | -17.0 | 54% | 1.20 |
| C4 br40+bbwlo | 6.0 | 1.52 | -18.7 | 56% | 1.29 |
| V08 rv10 | 8.2 | 1.42 | -14.1 | 42% | 1.00 ❌ |
| C2 br40+vc80+rv13 | 7.6 | 1.44 | -18.7 | 50% | 0.98 ❌ |

**Verdict:** +20–40% trades is achievable at acceptable cost (PF ~1.8, MDD ~-12..-14%,
holPF still >1.25). Best balance = **V03 br40** (breadth 0.40): 5.1 t/mo (+19%), PF 1.79,
holPF 1.32. Doubling trades (8+/mo) kills the edge (holPF ≤1.0). **BASE stays the safest;
V03_br40 is the moderate-frequency option.** User decision pending.

## R078 Symbol Forensics + Unseen Universe (2026-08-06)

**Q1 — Negative symbols?** 12/48 traded symbols net-negative full-period, but every one had
≤5 trades (median 2) → pure noise. No reliable bad symbol. 4 never triggered (SOL, ALGO,
GALA, SATS). **Do NOT prune — that would be curve-fitting.**

**Q2 — More assets?** Fetched 18 new OKX symbols (paginated history works via `after`
param; 70 symbols now cached). Ran locked config on 8 never-seen symbols (BICO/HYPE/XAU/
HOME/PUMP/ZBT/ZEC/BEAT, ≥5k bars): **17 trades, PF 0.625 — edge did NOT transfer** to
gold/meme/privacy/small-cap assets. Full universe PF 2.05 → 1.77 (diluted, still profitable).

**KEY: the edge is universe-specific, not universal.** Trade the validated 52-symbol
universe only; expand only via per-symbol validation, never by liquidity alone. Breadth
gate stays on original 52. Locked config unchanged.

## R077 Lock-In (2026-08-06) — 🏁 FINAL LOCKED CONFIGURATION

**$100-account answer (1% risk/trade): max drawdown ≈ −$10.50 (−10.5%) full-period.**
Monte Carlo (5,000 paths): DD P5=$9.72 / P50=$5.88 / P95=$3.94; P(end<$80)=0%.
Risk table: 0.5%→DD$5.38/net$35 · 1%→DD$10.52/net$82 · 1.5%→DD$15.43/net$142 ·
2%→DD$20.12/net$221 · 2.5%→DD$24.59/net$322. Worst month −4R, best +38R, 7-trade loss streak.

**15m test (8 symbols, 2026-only untouched OOS): NO — too sparse (7 trades, PF 0.80) and
no edge. 1m unusable (broken timestamps). Strategy stays on 1H.**

**RR sweep (breadth50): RR1.5 beats RR2.0 on holdout (holPF 1.400 vs 1.250) with better
retail metrics. Paired ΔPF not stat-sig (CI spans 0) but holdout+profile favor 1.5.**

### 🏁 LOCKED CONFIG (R077)
- **Strategy:** Family A conditions (BBW_STRICT+RV_LO+DST_NR+PRG_VH), rolling 500-bar
  thresholds recalibrated every 168 bars
- **Entry:** signal-bar close (E6) + gate (green candle, close>prev close, rel_vol>1.5)
- **Filters:** VolCeil (skip atr_rank>70) + **breadth50** (enter only when >50% of 52
  symbols above their EMA20)
- **Exit:** SL 1·ATR, **TP 1.5·ATR (RR=1.5)**, TP checked before SL, no time stop
- **Risk:** 1% per trade (user's choice)
- **Results:** n=116, **PF=2.051, WR=57.8%, MDD=−9.2%**, Exp $44.4/trade@$10k,
  Boot P5=1.553, LOO floor=1.938, **~5.8 t/mo, 65% profitable months, worst month-streak 3**
- **Files:** `quantlab_r077.py` + `quantlab_output/r077_*`

## R076 Overlay + Cross-Sectional (2026-08-06) — 🎯 BREADTH OVERLAY ADOPTED

**Finding: a market-breadth overlay transforms Family A into a retail-friendly strategy.**

**PART A — Overlay on Family A FINAL (E6+RR2+VolCeil):**

| Overlay | PF | MDD% | t/mo | prof% mo | streak | holPF | holMDD% | holProf% |
|---|---|---|---|---|---|---|---|---|
| none (base) | 1.62 | -26.5 | 10.2 | 41% | 7 | 1.44 | -13.2 | 43% |
| **breadth50** (only when >50% of 52 symbols above EMA20) | **2.15** | **-10.5** | 5.5 | **55%** | **4** | **1.25** | **-3.9** | **50%** |
| breadth60 | 2.12 | -9.6 | 5.0 | 55% | 4 | 1.07 | -3.9 | 50% |
| medret_pos | 1.68 | -14.2 | 4.8 | 46% | 3 | 1.05 | -4.0 | 60% |
| btc_bull | 1.12 | -26.9 | 4.3 | 36% | 12 | 2.15 | -3.0 | 80% (n small) |

**ADOPT breadth50:** PF 1.62→2.15, MDD -26.5%→-10.5%, profitable months 41%→55%, worst
streak 7→4, AND survives untouched 2026 holdout (PF 1.25, MDD -3.9%, 50% prof months).
BTC-bull overlay rejected (selection PF 0.83 <1 — unreliable signal density).

**PART B — Cross-sectional momentum/reversal (baskets):** 24-config grid (h 24/48/72,
rebal 12/24h, K 5/10, mom/rev) — ALL lose after 0.05% costs. Best selection Sharpe -0.03;
holdout worst -55%. Cross-sectional relative-value has NO edge on 1H crypto after costs.
REJECTED.

**FINAL RETAIL-FRIENDLY STRATEGY (R076):**
Family A conditions + E6 entry + RR 2.0 + VolCeil(atr_rank≤70) + **breadth50 overlay**
(enter only when >50% of universe above EMA20). ~5-6 trades/month, PF≈2.15, MDD≈-10%,
55% profitable months, worst losing streak 4, holdout-validated. This addresses the
retail-profile concern from R074/R075.

## R075 New Strategy Families (2026-08-06) — retail-friendly hunt

**All 4 pre-registered new families LOST money. Family A FINAL remains the best strategy.**

| Family | Concept | PF | t/mo | prof% months | Verdict |
|---|---|---|---|---|---|
| N1 trend pullback | EMA trend + pullback + 2ATR trail | 0.616 | 592 | 0% | NO |
| N2 range mean-reversion | ADX<20 + RSI<30 + <BB low, 1:1 | 0.909 | 152 | 37% | NO (closest, still loses) |
| N3 20-bar breakout | breakout + relvol + trail | 0.576 | 363 | 0% | NO |
| N4 London ORB | 12-14UTC range break | 0.621 | 1,288 | 0% | NO |
| **A_FINAL (R074)** | comp+pop E6+RR2+VolCeil | **1.620** | 10.2 | 41% | **reference** |

**Retail Score:** A_FINAL 48.4 vs best new (N2) 32.2. N2 also fails holdout (hol PF 0.815),
LOO 0.898, MC 0% — it's a loser, just a slower one.

**KEY INSIGHT:** generic retail strategies (trend-follow, breakout, mean-reversion, ORB)
have NO edge on this universe/timeframe at these params. The only validated edge remains
Family A's rare compression-then-pop setup — edge is scarce, not hiding in plain sight.

**Options considered for R076:** (1) N2 mean-reversion refinement (stricter oversold,
first-touch, vol filter) — last reasonable shot at retail profile, but overfitting risk;
(2) accept Family A as the project's validated strategy and shift focus to deployment
(bot config) / live validation; (3) new data sources (funding rate only exists for
BTC/ETH — too small; multi-timeframe 15m/1m only for ~7 symbols).

## R074 outcome & decision (2026-08-06)

**FINAL STRATEGY (Family A + E6 + RR2 + VolCeil)** is a real, validated edge (PF 1.63,
LOO 1.57, MC 100%) — **but judged NOT suitable for retail deployment:**
- Only 41% profitable months (12/29), worst losing streak 7 months, 2024 was −11R
- Profits ultra-concentrated: Aug–Oct 2025 = 79% of all gains
- MDD −13%..−26%, ~10 trades/month, breakeven cost 0.145%/side

**DECISION: keep hunting.** Next research target = strategies with a retail-friendly
profile: ≥50% profitable months, max losing streak ≤4, moderate frequency, lower MDD —
even at a PF trade-off. R075 = new strategy families (trend pullback, mean reversion,
breakout follow-through, ORB) with the same strict protocol (selection ≤2025, holdout
2026, bootstrap/LOO/MC/costs) + a transparent Retail Score to rank against Family A.

## R074 Edge Refinement (2026-08-06) — strict holdout (selection ≤2025, confirm 2026)

**FINAL STRATEGY: Family A + E6_sigentry (enter at signal-bar close) + RR 2.0 + VOLCEIL (skip when atr_rank > 70)**

R074 tested 11 pre-registered single-factor refinements vs the R073 winner with strict
holdout (decisions on pre-2026 data only; untouched 2026 for confirmation):

| Variant | Selection PF | ΔPF vs base [CI] | Verdict |
|---|---|---|---|
| BASE (E6, RR2) | 1.485 | — | reference |
| **V03_volceil** (skip atr_rank>70) | **1.717** | [+0.17, +1.32] SIG↑ | **ADOPT** |
| V04_breakout | 1.677 (n=57) | CI spans 0 | not significant |
| V06_vol200 | 1.604 | CI spans 0 | not significant |
| V01_time18 | 1.126 | SIG↓ | harmful |
| V10_exit_partial | 0.800 | SIG↓ | harmful (again) |
| V07/V08 RR 1.75/2.25, V09 time24, V11 trail, V05 vol175, V02 | — | CI spans 0 | no gain |

**Holdout (2026, untouched):** V03 PF=1.440 (n=86) vs base 1.524 (n=111) — PF within noise,
BUT MDD -13.2% vs -22.3% (risk-reduction confirmed out-of-sample).

**Full-period (V03):** PF=1.632 (vs base 1.496), n=296, WR=44.9%, MDD=-26.5%,
LOO floor=1.568 (vs 1.446), MC P(profit)=100% net +$17.9k DD P5=-16.6%,
breakeven cost 0.145%/side. Monthly: ~10.2/month (median 7).

**Why V03 works:** skips entries when ATR is already in the top 30% of its 100-bar range —
the compression-then-pop setup is invalid when volatility already exploded (R072 Q3 failure
mode). Removes the worst trades without removing winners.

**DEPLOYMENT CONFIG (final):** Family A only. Entry at signal-bar close (E6), RR=2.0,
atr_rank ≤ 70 filter, 1% risk, TP-before-SL intrabar, 1 pos/symbol. Family C OFF.

## ⚠️ R073 CORRECTION (2026-08-06) — supersedes all R066–R071 baselines

**The frozen baselines below were computed with a next-bar-close PROXY, not the bot's
real SL/TP execution.** R066's `backtest_family` defines a win as `next_close > entry_close`
with ±RR payout. The demo bot trades real 1·ATR SL / rr·ATR TP. The two measure different
things. Full proof: `EXIT_MODEL_AUDIT.md` (reproduces both engines exactly).

**Authoritative numbers (R073, bot-faithful rolling walk-forward, 52 symbols, 2.5 yrs, no costs):**

| | E0 (current bot) | Best variant | Best (with 0.10% costs) |
|---|---|---|---|
| **Family A** (RR=2.0) | PF=1.133, n=390, WR=36.2%, MDD=-42% | **E6_sigentry PF=1.496** (n=395, WR=42.8%, MDD=-38%) | PF≈1.13 |
| **Family C** (RR=3.0) | PF=0.926, n=10,375, WR=23.6% | E6_sigentry PF=0.938 | PF<1 at ALL costs — DEAD |

**R073 verdicts:**
1. Family A has a real but MODEST edge — NOT the exceptional PF=3.35 previously claimed.
   Best config: **E6_sigentry (enter at signal-bar close) + RR 2.0**. Boot P5=1.264,
   ΔPF vs E0 CI [+0.15,+0.55], LOO-sym floor=1.446, MC P(profit)=100%, DD -8%..-20%.
   Breakeven cost 0.145%/side. E0 (current bot) dies at ~0.04%/side — the 1-bar entry
   delay destroys the edge under costs. **E2_partial/E3_trail/E1_be1r are all significantly
   WORSE than E0** (R072's partial-TP suggestion does not survive the corrected engine).
2. Family C is unprofitable under EVERY exit and EVERY RR (0.39–1.02 PF, account ruined,
   MC P(profit)=0.0%). **Remove from the demo bot.** R068's "cleared for paper trading"
   is retracted.
3. R072's own report was internally inconsistent (key stats showed the collapse; the Q&A
   still recommended deployment on the old numbers).
4. The rolling engine (`quantlab_r073.py`) is now the project's authoritative evaluator:
   mirrors bot execution (IS_LOOKBACK=500, RECAL=168b, entry next close, TP-before-SL,
   1 pos/symbol, 1% compounding) and uses ALL data out-of-sample.

**Demo-bot action items (pending approval):** Family C → disable. Family A → evaluate
signal at latest closed bar and enter at its close (E6), keep RR=2.0, and monitor costs
(limit/maker fills or low-fee venue; breakeven 0.145%/side).

---

## Frozen Baselines (SUPERSEDED by R073 — see correction above; kept for history)

**Family A (E3.1_v2)** — FROZEN PROMOTE candidate (OLD proxy numbers)
- Conditions: `BBW_STRICT + RV_LO + DST_NR + PRG_VH`
- BBW_STRICT = bb_width < IS p25 (tighter than BBW_LO = p33)
- R066 baseline: PF=3.353, n=91, WR=62.6%, MDD=-4.6%, UES=209.1, Boot P5=2.333
- All folds profitable; LOO-sym=3.091, LOO-fold=2.600
- **Clear production leader. Deploy first for paper trading.** (OLD claim — see R073)

**Family B** — LOW SIGNAL FREQUENCY, not yet deployable
- Conditions: `RV_HI + DST_MD + ADX_WK + LON`
- R065: PF=2.188, n=25 (only 25 trades across 52 symbols over full history)
- R066: **0 trades in OOS period** — conditions almost never co-occur
- LON session (7–14 UTC) + high RV + weak ADX + extended above EMA200 = very rare
- **Do not combine into portfolio until more cache data available**

**Family C** — Active but lower quality (OLD proxy numbers)
- Conditions: `DST_NR + ADX_ST + PBD_HI + ASI`
- R066 baseline: PF=1.492, n=721, WR=42.7%, MDD=-15.1%, UES=74.5
- Boot P5=1.315, MC P(profit)=100%, LOO-sym=1.453, LOO-fold=1.401
- All 5 folds profitable; high trade count but -15% MDD is a concern
- ⚠️ R073: C = ADX_ST+PBD_HI (DST removed) is UNPROFITABLE under real SL/TP exits

## R066 Portfolio Validation Results

| Candidate | PF | WR | n | MDD | UES | Score |
|---|---|---|---|---|---|---|
| **Family A** | **3.353** | **62.6%** | **91** | **-4.6%** | **209.1** | **96.3** |
| A+B | 3.353 | 62.6% | 91 | -4.6% | 209.1 | 96.3 |
| A+C | 1.618 | 44.7% | 787 | -12.7% | 87.5 | 76.4 |
| A+B+C | 1.618 | 44.7% | 787 | -12.7% | 87.5 | 76.4 |
| Family C | 1.492 | 42.7% | 721 | -15.1% | 74.5 | 67.0 |
| Family B | 0.000 | 0.0% | 0 | 0.0% | 12.5 | 32.1 |

**Combining families DILUTES quality.** A+C portfolio drops PF from 3.353 → 1.618 and worsens MDD from -4.6% → -12.7%. Family A alone is superior. (Proxy-based — see R073.)

## R066 Key Conclusions

1. Family B has effectively zero signal density in OOS. Do not include in portfolio yet.
2. Family C is real but lower quality — adding it to A dilutes without enough diversification benefit.
3. Family A (E3.1) is the sole production-grade candidate. Run it independently.
4. A+C diversification score is only 39.2/100 — symbol overlap is 84.62% (they trade the same assets).
5. Permutation test (pctile=0.0000) is a known artifact of fixed-RR binary outcomes — not a real failure.

## R067 Family C Dissection Results

**Condition ablation (remove one condition at a time):**

| Variant | Conditions | PF | n | MDD | UES | Loss Streak |
|---|---|---|---|---|---|---|
| **C_no_DST** | **ADX_ST+PBD_HI+ASI** | **1.692** | **2,049** | **-3.2%** | **91.3** | **10** |
| C_no_ADX | DST_NR+PBD_HI+ASI | 1.728 | 1,506 | -10.6% | 90.1 | 9 |
| C_no_PBD | DST_NR+ADX_ST+ASI | 1.573 | 1,438 | -4.2% | 80.6 | 11 |
| C_FULL | DST_NR+ADX_ST+PBD_HI+ASI | 1.492 | 721 | -8.9% | 70.3 | 10 |

**Condition contribution (ΔPF when removed from Family C):**
- ADX_ST: +0.236 (weakest — hurts the strategy)
- DST_NR: +0.200 (second weakest)
- PBD_HI: +0.081 (modest contribution)
- ASI: +0.000 (neutral — session filter adds nothing extra, all trades already in ASI=00:00)

**ADOPT: C_no_DST = ADX_ST+PBD_HI+ASI**
- PF=1.692 (+0.200 vs full), MDD=-3.2% (vs -8.9%), n=2049 (3× more trades)
- Boot P5=1.576, UES=91.3, max loss streak=10
- DST_NR was filtering OUT good trades — removing it both improves quality AND increases frequency
- This is NOT optimization: it's evidence-based condition removal (proxy-based — see R073)

**Symbol insights from R067:**
- Best tier: T3-Small (PF=1.638), worst: T2-Mid (PF=1.279)
- Worst symbols: NEAR (-83% WR), OP (-87% WR), FET (-78% WR), ATOM (-77% WR), SUI (-76% WR)
- Intra-session analysis inconclusive (timestamp artifact — all read as hour=0)

## R068 ADX_ST+PBD_HI Independent Validation (COMPLETE — superseded by R073)

**Result: 8/8 production criteria passed — CLEARED FOR PAPER TRADING** (⚠️ proxy-based; R073 retracts)

| Metric | Value |
|---|---|
| PF | 1.6919 |
| WR | 45.8% |
| n | 2,049 |
| MDD | -5.9% |
| Expectancy | $37.48/trade |
| Boot P5 | 1.573 (criterion >1.20 ✓) |
| MC P(profit) | 100% (criterion >95% ✓) |
| LOO-sym floor | 1.674 [LTC removed] |
| LOO-fold floor | 1.644 |
| All 5 folds profitable | YES ✓ |

**Final Q&A (R068 Section 12):**
- Q1 Genuine standalone edge: YES
- Q2 DST_NR truly redundant: UNCERTAIN — bootstrap CI spans 0; directionally better but not proven
- Q3 Survives independent validation: YES
- Q4 Deploy on demo today: YES
- Q5 New official Family C: NOT YET — run alongside original, monitor live
- Q6 Stop Family C research: NO — paper trade first
- Q7 vs Family A: Both real. Family A = high conviction/low frequency (PF=3.35, n=91). ADX+PBD = moderate conviction/high frequency (PF=1.69, n=2049). Run both.

**Promotion status: ADX_ST+PBD_HI → PAPER TRADING alongside Family A** (⚠️ RETRACTED by R073)
- Demo bot should run both strategies independently
- Worst-case simulated drawdown: -33.6% (know this going in)
- MC expected drawdown: -7.2%

## R069 Timestamp Fix + Full Re-Evaluation (COMPLETE)

**Timestamp fix:** `quantlab_fix_timestamps.py` — sets `datetime` column as index on all 64 parquet files. 1m files skipped (not used in research). All 1H files now have real UTC hours 0–23.

**All 24 hours present, evenly distributed (~4.2% each). Session conditions now work correctly.**

| Family | PF | n | vs Baseline | Status |
|---|---|---|---|---|
| A (BBW+RV_LO+DST_NR+PRG_VH) | 3.353 | 91 | No change ✓ | CLEARED (proxy) |
| B (RV_HI+DST_MD+ADX_WK+LON) | 3.200 | **26** | First real test | NOT DEPLOYABLE |
| C (ADX_ST+PBD_HI) | 1.692 | 2,049 | No change ✓ | CLEARED (proxy) |

**Family B finding:** 26 OOS trades — signal PF=3.2 is promising but n is far too small. The data bug was real. The condition combination is genuinely rare with the LON filter. Need more cache data over time before Family B can be evaluated properly.

**Session sensitivity finding (Family C):** C+ASI [0,6) gives PF=2.260 n=409. Worth tracking but not changing the frozen strategy before live validation.

**Demo bot:** Build now. Family A + Family C. Family B sits out. (⚠️ Family C later retracted by R073)

## Capital Allocation Finding (R066 Section 7)
- Equal weight (33/33/33): PF=1.618, MDD=-8.3%
- Kelly-weighted (64%A / 36%C): PF=1.718, MDD=-8.3%, RF=6.57 — practical choice if both families used
- Risk parity ranks best numerically but allocates 100% to empty Family B

## Stress Test (A+B+C combined)
- Bootstrap P5=1.437 ✓, MC P(profit)=100% ✓, LOO-fold=1.581 ✓, LOO-sym=1.580 ✓
- Permutation: artifact of fixed-RR ✗ (not informative)
- Verdict: MODERATE (4/5 meaningful tests pass) (proxy-based)

## Architecture (RR, entry, universe)
- RR=2.0, entry gate: RELVOL>1.5 × 20-bar avg + close>open + close>prev_close
- 52 symbols in cache (1H timeframe), IS_RATIO=0.80, 5-fold walk-forward
- Promotion criteria: PF>1.20, n≥200, boot_med>1.15, MC_p<35%, LOO-sym>1.0, LOO-fold>1.0, MDD<20%

## OKX API Data Limitation
- history-candles REST endpoint caps at ~1440 bars (~60 days of 1H) for new symbols (via `before`); `after` param pages deeper (~Dec 2023 verified)
- 70 symbols now in cache (52 original + 18 fetched); cache grows incrementally over time
- Family B's 0 OOS trades is partly explained by the narrow OOS window (20% of 2000+ bars)

## Key File Locations
- Research scripts: `quantlab_r0XX.py` (r073–r077 = current authoritative arc)
- **Exit-model audit:** `EXIT_MODEL_AUDIT.md` + `scripts/exit_model_audit.py`
- **Shared engine:** `scripts/ql_engine.py` (bot-faithful rolling walk-forward)
- **Universe/data:** `scripts/fetch_more_symbols.py` (OKX `after` pagination)
- Config + indicators: `quantlab_ai.py`
- Cached OHLCV: `quantlab_cache/SYMBOL_1H.parquet` (70 symbols)
- R077 outputs: `quantlab_output/r077_*` (lock-in: dollar DD, 15m test, RR sweep, locked config)
- R078 outputs: `quantlab_output/r078_*` (symbol forensics, unseen-universe test)
- R076 outputs: `quantlab_output/r076_*` (breadth overlay adoption, cross-sectional rejection)
- R075 outputs: `quantlab_output/r075_*` (4 new families — all rejected)
- R074 outputs: `quantlab_output/r074_*` (edge refinement + strict holdout)
- R073 outputs: `quantlab_output/r073_*` (corrected baselines on bot-faithful engine)
- R066–R072 outputs: proxy-based, superseded by R073+
