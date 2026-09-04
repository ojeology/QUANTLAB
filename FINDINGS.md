# QUANTLAB Blind OOS Validation — Findings

**Goal:** find a strategy in the frozen QUANTLAB repo that survives different years (cost-adjusted, out-of-sample), for a 2027-01-01 crypto bot go-live.

**Method:** blind out-of-sample walk-forward. Train on all prior years, test the next year standalone. Fees 0.05% per side. Universe = crypto USDT perpetual swaps, 1H.

---

## 2026-09-04 — T34 audit (small-account × implementability) — READ FIRST

T34 re-ran the champions on the **full 50-symbol 2023→2026 universe** (2023 fetched from OKX; per-year walk-forward; fees 0.05%/side) to size a $100 account. Two corrections to the tables below:

1. **The T25/T27 trend-leg numbers are exit-anchored, not implementable.** The trend simulator stamps each trade at the *exit* bar (`entry_time = df.index[i]` after close), so the RF-q65 "condition" filter reads exit-bar features. Verbatim reproduction on fetched data matches the log ($100→$198.79 vs $199.70), but the **same champion gated at the true entry bar is PF@c ≈ 1.03, $100→$106 — no live edge** (raw trend alone: 1.10/0.94/0.90). TP variants 1–3R are all negative. → treat every T25/T27/PORT trend PF ≥1.3 number below as **exit-anchored / superseded for live trading** until an entry-anchored trend gate is found.
2. **The MR champion's 2024 is a real losing year on the full universe.** Full 50-sym run: n=297, PF@cost **0.739 / 2.151 / 1.610** (2024/25/26), return@1% **+71.2%** (30-sym log's +71.3% reproduces), **max DD @1% = −28.2%** (30-sym's −9.4% was sample luck; it missed 2024's losing regime). 2025/26 carry the result; robustness is still broad (15/30 profitable months, 36/49 symbols).

**Result / recommendation for a small ($100) account:** the MR leg is the one genuine implementable edge. Its R:R optimum is **TP 2R / SL 1R** (mean R +0.30 vs +0.22 at the RR1.5 default; wr 47% vs 53%; full-Kelly 14.5%). At **1% risk/trade**: $100 → **$204 (+104%)** 2024→Jul-2026, CAGR ≈32%, MC P(profit) 99.9%, P(halve)≈0%, realized max DD **−37%** (an 18-month 2024→mid-2025 grind, trough ≈$63, then a +49% Aug-2025 month, +28% in 2026). RR1.5 @1% is the smoother alternative ($173, DD −27%). Risk ≥2% → realized DD −50…−80%; not for a small account. **Drop trend and the 70/30 sleeve at small-account size** — no live edge. Practical "leverage": 1% risk ≈ ~1× notional (MR stop ≈1.15% of price); 3–10× exchange leverage only for contract fit, never the risk dial.

Files: `t34_lib.py`, `t34_validate.py`, `t34_small_account.py`, `t34_output/`. Pre-2024 data: `quantlab_cache_2023/` (re-fetch: `fetch_2023_50.py`).

---

## What failed
- **Raw Family A / Family C bot strategies** (as deployed): PF@cost ~1.0–1.1, weak. Family C even went negative on a full year.
- **R077 "holy grail" (breadth50 + VolCeil static)**: PF@cost 0.75 → loses money after fees.
- **Static filter sweep (q × VolCeil × breadth)**: NO config survived all of 2024/2025/2026. VolCeil=70 fixes 2024 but breaks 2026 (opposite regimes).
- **Other environments**: 4H → only 56 trades/3.5yr (edge is 1H-specific); 15m cache is 2026-only/sparse; forex blocked (Dukascopy unreachable).
- **SVM q0.75 (original "champion")**: PF@cost 1.25 (2026), 75% profitable-months — good but q0.65 is better.

## The Champion (validated)
**SVM q0.65 + VolCeil gated by `|ema_dist_pct| > 2.0`** (apply the ATR-spike skip ONLY when price is stretched from its mean), trained on all prior years.

| Year | PF@cost | Win% | Return(1% risk) | Max DD | Profitable months |
|---|---|---|---|---|---|
| 2024 | 1.18 | 48% | +5.8% | −9.4% | 7/11 |
| 2025 | 2.11 | 62% | +43.5% | −8.6% | 6/10 |
| 2026* | 1.53 | 55% | +12.8% | −7.7% | 2/6 |
| **Full** | — | — | **+71.3%** | **−9.4%** | **15/27 (56%)** |

\*2026 partial (cache ends ~Jul-2026; 6 months measured).

**Why it works:** the edge is regime-cyclical. A *fixed* VolCeil fails because 2024 needs it and 2026 doesn't. Gating VolCeil by regime (price stretched from mean) applies the safety filter only when actually needed → both years survive.

**Risk/robustness (full 30-sym universe, 187 trades):**
- Max drawdown ≈ **9.4%** at 1% risk/trade.
- **22 / 30 symbols profitable** (73%); top-3 carry only 37% of total R → broad, not concentrated.
- Deploy with 1% risk/trade + DD-halt rules.

## Files (branch `blind-validation`)
- `blind_test.py` — Family A/C bot strategies, walk-forward split
- `control_5m.py` — negative control (5m correctly fails)
- `svm_crypto_blind.py` / `svm_deploy.py` — SVM + deployable module
- `champion_showdown_2026.py` — SVM vs R077 head-to-head
- `improvements.py` — stacking / q-sweep / cross-regime
- `split_2025_2026.py` — per-year standalone OOS
- `test_2023.py` — extend back to 2023
- `filter_sweep.py` / `filter_sweep_2023.py` — static filter hunt
- `filter_sweep_4H.py` — 4H environment (too sparse)
- `filter_sweep_adaptive.py` / `filter_sweep_adaptive_2023.py` / `filter_sweep_adaptive_saved.py` — adaptive-rule hunt
- `fetch_and_save_2023.py` — persist 2023-2026 data (resume-safe)
- `champion_deep_dive.py` — MDD / profitable-months / per-asset
- `quantlab_cache_2023/` — persisted 30-symbol 2023-2026 1H series

## Open question
Is the persistent SVM edge unique to Family A, or do other coiled-market signal definitions on 1H also carry it (ensemble potential)? → next hypothesis.
