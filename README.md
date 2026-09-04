# QuantLab — Systematic Trading Strategy Research

> A reproducible research journal of systematic trading strategy exploration across **crypto (1H / 15m)**, **Deriv binary options**, and **forex**. Built on a backtesting framework (`quantlab.py`, `ql_engine.py`, `svm_deploy.py`).

**Conclusion up front (no sugar-coating):** the *only* validated, deployable edge is a **1H crypto trend + mean-reversion portfolio (T25 / T27)**. It is real, consistent, and low-risk — but it compounds slowly. Every faster or alternative path (15m crypto, binaries, synthetics, forex) is either **data-blocked** or has **no real edge** in this sandbox.

---

## Repo layout
```
quantlab.py        original framework: signal -> trade sim -> PF/DD
scripts/
  ql_engine.py     feature engineering (add_features, build_signal_mask, sim_symbol)
  svm_deploy.py    ML condition filter (RandomForest / SVM), build_mldf
  demo_bot.py       paper-trading bot skeleton (simulation-only, never executes live)
ml_conditions.py / t*.py / deriv_*.py / forex_*.py   the experiments below
RESEARCH_DERIV.md  Deriv / binary-options findings
README.md          this file
```

---

## The Exploration Tree

```
QuantLab Research
|
+-- MAIN BRANCH - Crypto 1H  (validated, deployable)
|   +-- T25  Condition-aware TREND (Donchian breakout + RF q65 filter)
|   |        PF 1.744 / 1.529 / 1.537  (2024/25/26)   -> bulletproof, DD -2%
|   +-- T27  MR + TREND portfolio (trend-weighted 70/30)
|   |        PF 1.40 / 1.32 / 1.81,  MAX DD <1%       -> bulletproof, lowest risk
|   +-- T13  Adaptive-VolCeil mean-reversion   PF 1.18 / 1.82 / 1.53
|   +-- T24  SVM-Q65 mean-reversion champion  PF 1.58 (full)
|   +-- T30/C Unified MR+trend model (one RF) PF 1.50
|
+-- SUB-BRANCH A - Crypto 1H ideas that FAILED
|   +-- T1-T12  Mean-reversion families (BBW/RV/DST/PRG variants) -> 2026 loses
|   +-- T14/T15 1H / 4H Donchian trend                         -> fail
|   +-- T19-T23 CAGE false-breakout                            -> fail
|   +-- T30/A  Cross-sectional momentum rotation               -> 80% DD, fail
|   +-- T30/B  Trailing-exit trend (vs fixed stop)             -> worse than T25
|   +-- T30/D  BTC-ETH pairs (spread z-score)                 -> 100% DD, fail
|   +-- T31  Combined 3-strategy portfolio (MR+Trend+Unified) -> PF 1.21 (< T27)
|
+-- SUB-BRANCH B - Crypto 15m  (blocked)
|   +-- T28/T29  Condition-aware 15m trend  PF 1.58 on 7-mo window (noisier than 1H)
|   +-- Full 3-yr 15m fetch -> OKX throttles 15m under load;
|            Binance / Bybit / Yahoo all BLOCKED in sandbox -> cannot finish
|
+-- SUB-BRANCH C - Deriv Binary Options
|   +-- Reachability: YES (WS wss://ws.binaryws.com, public app_id 1089)
|   +-- Synthetics (vol / step / jump indices) = engineered random walk -> NO edge
|   +-- Forex 15m binary = faint >55% win on ~35-day slices, but
|   |      Deriv free API caps 15m history to ~35 days -> cannot validate deeply
|   +-- Crypto method on forex 15m binary (T32) = inconclusive (same data cap)
|
+-- SUB-BRANCH D - Forex 1H SPOT  (T33)
|   +-- T25 method on forex 1H -> PF 1.099 (break-even); MR = 0 signals.
|            Our crypto edge does NOT transfer to forex.
|
+-- SUB-BRANCH E - Stocks / Equities
     +-- No equity data in sandbox (OKX is crypto-only; Yahoo/Binance blocked). BLOCKED.
```

---

## Master Test Table (T1 - T34)

| # | Idea | Result |
|---|-------|--------|
| T1 | 1H MR (Family A) | 2026 loses |
| T2 | 1H MR expanded families | 2026 loses |
| T3 | 1H MR alt exits | FAIL |
| T4 | 1H MR multi-family | FAIL |
| T5 | 1H MR tight params | FAIL |
| T6 | 1H MR time-of-day | FAIL |
| T7 | 15m (too sparse) | - |
| T8-T11 | 1H MR variants | FAIL |
| T12 | 1H MR + momentum filter | 2026 loses |
| T13 | 1H MR adaptive VolCeil | PF 1.18 / 1.82 / 1.53 |
| T14 | 1H Donchian trend | FAIL |
| T15 | 4H Donchian trend | FAIL |
| T16 | 1H MR+Trend portfolio | FAIL |
| T17 | 1D trend | strong but sparse |
| T18 | multi-TF portfolio | FAIL |
| T19-T23 | CAGE false-breakout | FAIL |
| T24 | Condition-aware ML (SVM) | PF 1.576 |
| T25 | Condition-aware TREND (Donchian+RF) | PF 1.744 / 1.529 / 1.537 ⚠ exit-anchored — see T34 |
| T26 | MR+Trend 50/50 | PF 1.21 / 1.33 / 1.81 |
| T27 | MR+Trend 70/30 portfolio | PF 1.40 / 1.32 / 1.81, DD<1% ⚠ trend leg exit-anchored — see T34 |
| T28 | 15m condition-aware | PF 1.58 / 1.75 / 0.94 (noisy) |
| T29 | 15m broad universe | PF 1.580, 23/38 profitable (7-mo) |
| T30/A | Cross-sectional momentum | FAIL (80% DD) |
| T30/B | Trailing-exit trend | worse than T25 |
| T30/C | Unified MR+trend model | PF 1.50 |
| T30/D | BTC-ETH pairs | FAIL (100% DD) |
| T31 | Combined 3-strategy portfolio | PF 1.21 (< T27) |
| T32 | Deriv reach + forex-15m binary (crypto method) | synthetics random; forex faint; data-capped |
| T33 | Forex 1H spot (T25 method) | PF 1.099 (break-even); MR silent |
| T34 | Small-account leverage × R:R ($100; implementability audit) | MR RR2 @1% risk: $100→$204, DD −37%; trend gate exit-anchored → NOT implementable |

---

## Key findings, by branch

### Main branch - Crypto 1H (the real edge)
- **MR champion (T24/T13, entry-anchored — the genuinely implementable edge):** FAM_A coiled signal + SVM q0.65 adaptive VolCeil, SL 1 ATR. On the FULL 50-symbol 2023→2026 universe: return@1% risk +71%, but 2024 is a real losing year (PF 0.74) and max DD @1% = −28% (T34b). **T34 recommendation (small account, $100): TP 2R instead of the 1.5R default, 1% risk/trade → $100→$204 (+104%), MC P(profit) 99.9%, P(halve)≈0.** Expect a −37% drawdown grind into mid-2025 before the +49% Aug-2025 month.
- **T25 (condition-aware trend) / T27 (70/30 portfolio):** headline PF >= 1.3-1.5 every year / DD<1% is **exit-anchored — the RF gate reads exit-bar features**. Re-run with entry-only information (T34a): trend edge ≈ PF 1.03, no edge. **Not deployable as published** until an entry-anchored trend gate is found.
- Historical numbers on 20-73 symbols stand as reported for their original exit-anchored pipeline; T34 is the implementability-corrected reading.

### Sub-branch B - Crypto 15m
- 15m showed a real but noisier edge (T29 PF 1.58 on a 7-month window, 23/38 symbols profitable, ~6x the 1H trade count).
- Blocked: full 3-year 15m history cannot be fetched - OKX throttles 15m under parallel load, and Binance/Bybit/Yahoo are unreachable from this sandbox. 15m remains a proof-of-concept.

### Sub-branch C - Deriv binary options
- Deriv is reachable (unlike every other exchange). Binary profitability requires a >55% directional win rate (payout < 2x stake).
- Volatility/step/jump indices are engineered random walks - 50% accuracy, no edge.
- Forex 15m showed faint >55% on small (~35-day) samples, but Deriv's free API caps 15m history, so it cannot be validated at depth.
- Verdict: binaries are not a viable fast-gains path here without a proven >55% edge, which we could not establish.

### Sub-branch D - Forex 1H spot (T33)
- Applied the exact T25/T27 pipeline to forex 1H (Deriv, ~1 year/pair). PF 1.099 - break-even. The crypto trend edge does not transfer to forex; MR produced zero signals (conditions tuned for crypto don't fire on forex's smoother structure).

### Sub-branch E - Stocks
- No equity data available in the sandbox; cannot fetch (OKX crypto-only, Yahoo/Binance blocked). Dead end here.

---

## Honest bottom line
We tested every reachable trading type: crypto 1H (MR, trend, CAGE, momentum, pairs, unified, portfolios), crypto 15m, Deriv synthetics, forex 15m binary, forex 1H spot, and (attempted) stocks. The inescapable result:

- **The one implementable, deployable edge is the 1H MR leg (SVM q0.65 adaptive VolCeil).** For a small account (T34, $100): TP 2R / SL 1R, 1% risk/trade → ~+104% over ~2.6 yr, near-zero ruin, accepting a −37% drawdown in year one-to-two.
- **The T25/T27 trend "PF≥1.3, DD<1%" numbers are exit-anchored and superseded** for live/paper go-live (T34a: entry-only trend ≈ PF 1.03). Until an entry-anchored trend gate is found, do not size a real account on the trend leg.
- Everything faster or on other instruments is **data-blocked or edge-less** in this environment.

---

## File guide (what to read)
- `scripts/ql_engine.py`, `scripts/svm_deploy.py` - the engine.
- `ml_conditions.py` - T24/T25 condition-aware training (the core ML filter).
- `t31_combined.py` - T31 combined multi-strategy portfolio.
- `t_all_1h.py` - T30 (cross-sectional momentum, trailing trend, unified model, pairs).
- `deriv_sweep.py`, `deriv_5m_backtest.py` - Deriv synthetic + 5m binary tests.
- `forex_15m_backtest.py`, `forex_15m_crypto_method.py` - forex 15m binary exploration.
- `forex_1h_spot.py` - T33 forex 1H spot (T25 method).
- `RESEARCH_DERIV.md` - Deriv / binary notes.

## Reproduction notes
- Crypto 1H data: `quantlab_cache/` (73 symbols, 2024-2026, from OKX).
- Deriv data: pulled live via WebSocket (`wss://ws.binaryws.com/websockets/v3?app_id=1089`, `ticks_history` style=candles; min granularity 60s for synthetics, 900s for forex).
- `pip install pandas numpy scikit-learn pyarrow requests websocket-client` to run the scripts.
- The sandbox egress is restricted to OKX (crypto) + Deriv; Binance/Bybit/Yahoo are blocked - this shaped every result above.
