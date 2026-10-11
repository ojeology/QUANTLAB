# STACK v2 — Executive Brief
**Delta-neutral yield floor + two convex sleeves. 106 rounds of research, distilled to one page.**
*Verified against real OKX rate history; methodology and full artifacts available on request.*

---

## The thesis

At retail costs, **statistical edges are priced away** (proven across ~100 tested candidates — the survivors must earn *contractually*, not predictively). This stack has three layers; **each layer is paid by a different payer, for a different economic reason**, so no calendar regime starves all three at once.

## Layer 1 — The floor (80% of capital): hedged lending

**Construction:** hold 20 liquid coins (BTC, ETH, SOL, XRP, DOGE, BNB, ADA, LINK, LTC, AVAX, DOT, NEAR, APT, SUI, ATOM, FIL, INJ, ARB, OP, ETC) → lend each in OKX **flexible savings** → short 1:1 notional on each **USDT perp**, isolated margin. Fully delta-neutral; enter/exit at bar closes only.

**Two income streams:**
1. **Coin-borrow demand** — short sellers must borrow the coins (rent).
2. **Perp funding** — structurally long-crowded market pays the short side.

**The key property (observed in data, not assumed):** the two payers are *anti-correlated*. When funding flips negative (short-crowded), those same shorts must borrow coins → lending APR spikes. Example: ATOM funding −0.060%/wk offset by lending +0.086%/wk → combined positive. **Whoever is greedy pays you.**

**Measured — 15 weeks of real OKX rates, all 20 coins:**

| Metric | Result |
|---|---|
| Basket APR | **+5.4%** |
| Positive weeks | **15/15 (100%)** |
| Worst week | **+0.055%** (still positive) |
| Max drawdown | 0.00% (yield-only view) |

Static basket beats monthly top-12 reconstitution (+5.6%, churn eats the edge) → **ship static**. *Floor v2 is +40–80% over the previous generation's floor.*

## Layer 2 — Crash convexity (cascade sleeve, 0.15× risk)

Fades vertical capitulation wicks at bar close (3 slots, +2%/−3%/72h TTL, costs included). Standalone: n=473, WR 60%, +6% ann, but worst week −11.4% → **fails standalone gates, kept only as a small sleeve** because it pays *exactly* in crash weeks, when the floor is flat and trend systems suffer. Forced-flow premium: someone's liquidation cascade is your fill.

## Layer 3 — The champion (frozen, 0.15% risk/trade)

Breadth50 regime filter + walk-forward SVM (q0.55), close-only entries, SL 1·ATR / TP 1.5·ATR. Yearly ledger: 2024 −14R / 2025 +67.4R / 2026 +22.4R. Pays in expansions. Frozen by charter; monthly recheck only.

## The stack (11 weeks overlap, all costs in)

| Metric | Result |
|---|---|
| Positive weeks | 73% |
| Worst week | **−0.09%** |
| Max drawdown | **−0.09%** |
| Annualized | **+18.6%** |
| Income | **≈ $7 / $14.5 / $29 per month on $500 / $1k / $2k** |

Kill-switches: stack DD > 6% → floor only. Champion has its own throttle + recheck.

## Honest weak points (so you don't have to find them)

- **15-week funding sample** (OKX API cap ~3.3 months). 2026 is a quiet regime; 2021-style funding would pay far more, a dead market roughly half. Structure doesn't depend on regime; income does.
- **Exchange counterparty risk is the real tail.** All layers sit on OKX (coins, perps, lending pool). Post-FTX rule: this is a slice of net worth, never the whole thing. No self-custody alternative pays the lending stream.
- **Execution friction:** perp-side margin management (isolated, ~25% of notional per position), funding/taker fee slippage vs. backtest approximation.
- **Smart-contract-free by design** — no DeFi legs, no bridge risk; everything is native exchange products.

## Status

Paper-trading phase: **12-week trial**, weekly yield logging + sleeve sim. Promotion bar: ≥70% green weeks, worst week ≥ −1.5%. No real capital before promotion.

*Repo: github.com/ojeology/QUANTLAB — full audit trail: prespecified gates, per-round reports, rejection ledger (~100 failed candidates), charter: `CONSISTENCY_HUNT_BEGINS.md`.*
