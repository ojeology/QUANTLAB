# T34 — Small-Account Leverage × R:R Study (crypto 1H, 50-sym, 2023–2026)

**Date:** 2026-09-04 · **Branch context:** `blind-validation` worktree (T25/T27 lineage) · **Universe:** 50-symbol manifest, full history back to 2022-12/real OKX listing (45 full + 5 partial-listing) · **Fees:** 0.05%/side · **Protocol:** walk-forward OOS — filters trained on prior years only, each test year standalone (2023→2024, ≤2024→2025, ≤2025→2026) · **R:R levers:** trend TP 1–3R, MR TP 1–3R · **Risk levers:** risk-per-trade f ∈ {0.5%…12%} of equity, compounding.

> **Why this study exists.** The strategy the repo validated (~+100% over 3 years) is a *rate* a big account can live with, but on $100 it is barely bus fare. This study asks: does adding leverage (risk-per-trade) and retuning reward:risk turn it into a small-account grower — *without* breaking the branch's honesty rules (no lookahead, fees, train-before-test)?

---

## 0. TL;DR — two findings that change the answer

1. **The headline trend edge (+99.7%, PF 1.47) is not tradeable as published.** The branch's `backtest_donchian` labels each trade with the **exit** bar's timestamp; the RF "condition" filter therefore trains on and selects trades by *exit-bar* features — information a live trader does not have at entry. Reproduced on identical data, the **entry-anchored** (implementable) version of the same champion is **PF ≈ 1.03, $100→$106 over 3 years — no edge**. TP variants (capping winners at 1–3R) are all negative (mean R −0.01 to −0.03). (Section 1.)
2. **The mean-reversion leg is the genuine, implementable edge — and its R:R sweet spot is TP = 2R, not the branch-default 1.5R.** Walk-forward OOS, fees in: **MR-RR2 mean R +0.30, full-Kelly 14.5%, and the full 50-symbol sample reproduces the branch's ~+71% return at 1% risk** (branch deep-dive on the same data: +71.2%). The catch the 30-symbol branch log hid: **2024 is a real losing year on the full universe** (PF 0.73) and the true max drawdown at 1% risk is **−28% to −37%**, not −9.4%. Small accounts can still survive it; you just have to know it's coming.

**Recommendation (Section 6): trade the MR leg only, TP 2R, 1% risk per trade, from $100 — an implementable ~doubling (→$204) over ~2.6 years with near-zero ruin odds — and drop the trend/70-30 sleeves, which add only un-tradeable or negative edge at small-account sizes.**

---

## 1. Validation gate (why the trend number collapsed)

Reproduced `t25_full_universe_3yr` on the fetched 2023→2026 data with two anchors of the *same* champion code:

| Run | 2024 PF (n) | 2025 PF (n) | 2026 PF (n) | $100→ | Meaning |
|---|---|---|---|---|---|
| Branch log (reference) | 1.570 (2788) | 1.393 (2567) | 1.392 (1403) | $199.70 | published |
| **A) exit-anchor** (verbatim branch code) | 1.576 (2770) | 1.376 (2573) | 1.377 (1411) | **$198.79** | data+machinery faithful ✓ |
| **B) entry-anchor** (implementable) | 1.106 (2806) | 0.982 (2765) | 0.929 (1479) | $106.44 | **live edge = none** |
| C) raw trend, no filter | 1.096 | 0.941 | 0.903 | $100.49 | context |

Raw signal count 11,700 matches the branch exactly; the RF-q65 champion on exit-bar features reproduces the branch numbers within noise. The **entire** gap between A and B is the RF filter reading exit-bar conditions. The branch never intended this — its `backtest_donchian` happens to append `df.index[i]` at the exit bar — but for a study that must map to a live $100 account, **B is the honest number**.

MR leg cross-check: `champion_deep_dive.py` on the same 50-symbol data → n=297, per-year PF@cost **0.739 / 2.151 / 1.610**, return@1% **+71.2%**, max DD **−28.2%**. Return matches the branch's 30-symbol log (+71.3%); the DD and 2024 PF do not, because the 30-symbol sample never had a losing 2024.

## 2. R:R sweep — which target wins per leg?

| Config | n | WR | mean R | full-Kelly | verdict |
|---|---|---|---|---|---|
| TREND T25-ASIS (no TP) | 7,050 | 27% | **+0.055** | 0.01 | only positive trend config, and it is ~noise |
| TREND TP 1.0R | 12,374 | 50% | −0.032 | 0.00 | TP caps the right tail → negative |
| TREND TP 1.5R | 10,888 | 40% | −0.023 | 0.00 | ✗ |
| TREND TP 2.0R | 9,925 | 33% | −0.021 | 0.00 | ✗ |
| TREND TP 3.0R | 8,850 | 28% | −0.013 | 0.00 | ✗ |
| **MR RR 1.0** | 311 | 60% | +0.114 | 0.12 | ok, low EV |
| **MR RR 1.5** (branch default) | 316 | 53% | +0.217 | **0.141** | good |
| **MR RR 2.0** | 305 | 47% | **+0.301** | **0.145** | **best EV & Kelly** |
| MR RR 3.0 | 300 | 34% | +0.264 | 0.083 | too few winners at 34% |

Trend TP is a dead end (its wins are rare, fat right-tail breaks; capping them kills the only thing that worked — and even uncapped the implementable edge is ~0). MR is inverted: EV *rises* from RR1 → RR2 and only rolls over at RR3 (win rate 60%→34%). **RR2 ≈ the optimal target** on both mean-R and full-Kelly; the branch's RR1.5 default is fine but leaves EV on the table. (n≈300 → the RR1.5-vs-RR2 gap is not individually significant; treat RR2 as "≥ as good as RR1.5", not as a precise optimum.)

## 3. Risk sweep (MR leg — the implementable frontier)

*Realized* max DD is measured on the chronological equity path (loss clustering intact). *MC* = 4,000 reshuffled bootstrap paths; **MC understates drawdown** because reshuffling destroys the 2024-style loss clusters — always read the realized column first.

| Config | risk | End $100 | CAGR | **Realized maxDD** | MC P(profit) | P(2×) | P(3×) | MC P(halve) | MC P(DD>30%) | prof mo | worst streak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MR-RR1.5 | 0.5% | $136 | +13% | −14% | 99.8% | 0% | 0% | 0.00% | 0.0% | 15/30 | 4 |
| MR-RR1.5 | **1%** | **$173** | +24% | **−27%** | 99.8% | 44% | 2% | 0.00% | 0.1% | 15/30 | 4 |
| MR-RR2 | 0.5% | $150 | +17% | −20% | 99.9% | 3% | 0% | 0.00% | 0.0% | 11/30 | 5 |
| **MR-RR2** | **1%** | **$204** | **+32%** | **−37%** | **99.9%** | **77%** | **21%** | 0.00% | 0.1% | 11/30 | 5 |
| MR-RR1.5 | 2% | $240 | +41% | −50% | 99.6% | 90% | 65% | 0.00% | 10.3% | 15/30 | 4 |
| MR-RR2 | 2% | $301 | +54% | −63% | 99.9% | 97% | 88% | 0.00% | 12.6% | 11/30 | 5 |
| MR-RR2 | 3% | $351 | +63% | −80% | 99.9% | 99% | 96% | 0.03% | 56.8% | 11/30 | 5 |
| MR-RR2 | 5% | $271 | +48% | −95% | 99.9% | 99% | 89% | 0.10% | 98.9% | 11/30 | 5 |
| MR-RR3 | 2% | $138 | +13% | −80% | 98.7% | 85% | 66% | 0.07% | 54.9% | 11/30 | 6 |
| TREND T25-ASIS | 0.5% | $131 | +11% | −87% | 88.7% | 72%* | — | 3.1% | 99.7% | 11/30 | 6 |
| TREND T25-ASIS | 1% | $13 | −55% | −99% | 76.7% | 65%* | — | 14.5% | 100% | 9/30 | 6 |
| PORT 70/30 T25+MR | 1–3% | $52–105 | −22%…+2% | −79…−96% | ≤87% | ≤75% | — | 5–99% | 100% | — | — |

\* MC P(2×) for trend includes paths that first halve then recover — see P(halve). Full 58-row sweep in `t34_leverage_rr_sweep.csv`.

**Reading the frontier.** Ruin (MC P(halve)) stays ≈0 through **2%** risk for both RR1.5 and RR2, then starts to bite. Drawdown, however, grows roughly *linearly* in risk and is brutal past 2%: realized −50% (RR1.5@2%), −63% (RR2@2%), −80% (RR2@3%). The 2024 losing year is the binding constraint — at higher risk it is not a dip you sit through, it is an account you abandon.

## 4. What a $100 trader actually experiences (recommended config)

Chronological year-end equity and troughs, MR-RR2 @1% (and RR1.5@1% in brackets):

| Year | Year-end | Realized that year (1% risk) |
|---|---|---|
| 2024 | $78.70 ($83.50) | **bleed year** — PF 0.73 (0.77), ~ −20% (−16%); trough extends into H1-2025 |
| 2025 | $165.40 ($144.90) | monster year — +135% (+87%); the **Aug-2025** month alone is +49% (+43%) |
| 2026 (→Jul) | $203.70 ($173.10) | +28% (+22%) |

Worst month-end trough: **~$63 (~$73), mid-2025** (−37%/−27% from prior peak — deepest around Jun 2025). Worst single month −9.5%/−7% (Jan 2025); best month +49%/+43% (Aug 2025). Profitable months 11/30 (15/30); longest losing streak 5 (4) months.

So the honest experience is: **~18 months of flat-to-down grind (2024 → mid-2025), during which the account drops toward −37% and the strategy looks broken, then a single vertical month (+49%) and a 2026 that nearly doubles it again.** This is the single most important behavioural fact for a small-account trader. It is also why every config below ~1% risk is pointless (nothing ever happens) and every config above ~2% is dangerous (the grind becomes ruin).

## 5. What "leverage" means here, practically

Risk is set by the **per-trade risk fraction f**, not by the exchange leverage dial. For the MR leg the stop is 1 ATR ≈ **1.15% of price** (median; p10–p90 = 0.7–1.9%):

- Position notional = f × equity ÷ (stop distance / price) ≈ f × equity ÷ 0.0115
- **f = 1% ⇒ notional ≈ 0.9× equity (~1×); f = 2% ⇒ ≈1.7× equity**
- On OKX, cross/perpetual leverage of **3–10×** is then enough *only* to let a $100 account buy a single 0.01-contract slice of a large-ticket perp (BTC/ETH) and to keep margin headroom; liquidation sits far beyond the 1-ATR stop. Never use the exchange's max (20–100×): it would make the *stop distance* itself (~1%) the account-killer on gap-through.
- Not modelled (small but real drags to budget): perp funding (≈0.01%/8 h while a trade is open; MR holds hours–days → low single-digit %/yr), slippage beyond the 0.05% fee assumption, and OKX round-lot/min-notional dust on a $100 account.

## 6. Recommendation

**Primary — "the grower":** MR leg (FAM_A coil + SVM q0.65-adaptive filter), **TP 2R / SL 1R**, **1% of equity risk per trade**.
- Implementable walk-forward OOS result from $100: **→ ~$204 (+104%) over ~2.6 years, ~32% CAGR**, fees in.
- Near-zero ruin: MC P(profit) 99.9%, P(halve) ≈ 0.00%, P(3×) 21%.
- Cost: realized max drawdown **−37%**, almost all of it the 2024→H1-2025 grind. On a $100 stake that is a ~$37 paper loss at the trough.

**Alternative — "the smoother"** (for anyone who will quit during a −37% drawdown): same leg at **RR1.5** (branch default), 1% risk → $173 (+73%), realized DD **−27%**, 15/30 profitable months. Statistically near-indistinguishable from RR2; pick on temperament.

**Aggressive tier — only if $100 is "play money":** RR2 @ **2%** → $301 (+201%), but realized DD **−63%** and P(DD>30%) 12.6% — a −63% hole is where small accounts die psychologically. Not recommended as the base case.

**Rejected for a small account:**
- **Trend / T25 any TP and the T27 70/30 PORT sleeve.** The implementable trend edge is ≈0 (Section 1); the 70/30 sleeve inherits it and every PORT row loses (final $52–105, DD −80%+). The branch's PORT numbers (PF 1.29–1.81, DD <1%) rest on the exit-anchor trend leg.
- **MR-RR3+** — WR drops to 34%, Kelly halves, and DD per unit of return worsens.
- **Risk > 2%** — realized DD −50% or worse; MC ruin starts at 3%+.

**Bottom line for the dollar trader:** there is no free leverage. The one genuine implementable edge in this universe (MR) pays ~2.4× on mean R at TP2, doubles a $100 account in ~2.6 years at 1% risk, and asks you to survive a −37% drawdown in year one-to-two. That is the honest best available; everything the branch's headline numbers promised on the trend side does not survive contact with entry-only information.

---

*Full machine-readable sweep: `t34_leverage_rr_sweep.csv` (58 configs × 22 metrics). Equity curves: `t34_equity_curves.png` (log scale). Code: `/home/user/bv/t34_lib.py`, `t34_small_account.py`, `t34_validate.py`.*
