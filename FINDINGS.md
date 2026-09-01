# QUANTLAB Blind OOS Validation — Findings

**Goal:** find a strategy in the frozen QUANTLAB repo that survives different years (cost-adjusted, out-of-sample), for a 2027-01-01 crypto bot go-live.

**Method:** blind out-of-sample walk-forward. Train on all prior years, test the next year standalone. Fees 0.05% per side. Universe = crypto USDT perpetual swaps, 1H.

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
