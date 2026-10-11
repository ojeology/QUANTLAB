# R103 — CONSISTENCY HUNT, ROUND 1: the high-win-rate dream meets its own math

**Date:** 2026-10-09 · Objective (per charter): consistency, not magnitude ·
Close-only fills (zero path ambiguity) · window includes the 2024 storm ·
Gates: ≥65% positive weeks · worst week ≥ −3% · maxDD ≤ 12% · costs included.

---

## Results — both classic candidates fail, and the *reason* is the finding

### V2 — Dip-ladder (RSI2 dips in uptrends, TP +1.5% / SL −3%, 33,120 trades)

| Metric | Result |
|---|---|
| Win rate | **63%** |
| **Breakeven WR needed** | **67.2%** (3.0 ÷ 4.5) |
| Expectancy per trade | **−0.27% net** → ≈ −0.9%/week for a 70-symbol portfolio |
| Gates | 0/7 — dead in all three years |

**The law this exposes:** any TP<SL structure needs WR > SL/(TP+SL). At retail costs in
crypto, achievable WRs for close-only dip structures (63%) sit *below* that line. The
higher the win rate you chase, the fatter the left tail — and the fatter the tail, the
higher the WR you need. **"High win rate" and "consistent money" are not the same thing;
at retail cost they are mathematical opposites.** (Same cost-wall law as the forex hunt,
now proven for consistency structures: R081 scalps, F009 RSI2-fade, and this.)

### V1 — Low-vol basket (6 lowest-vol uptrending symbols, weekly, spot)

| Metric | 2024 | 2025 | 2026 | Full |
|---|---|---|---|---|
| Positive weeks | 47% | 48% | 51% | **48.9%** |
| Worst week | −19.2% | **−22.7%** | −14.1% | −22.7% |
| Year sum | +37.7% | **−31.7%** | +49.2% | ~0.5%/yr |
| Max drawdown | | | | **−63%** |

**Gates: 0/7.** Crypto's low-vol names are not defensive — low realized volatility in
alts usually means *dormant bleed risk*, and the basket got regime-whipsawed (up in
2024, destroyed in 2025, up in 2026). Calmar 0.01. In equities the low-vol anomaly is
documented; in alt-perps/spot universe it does not survive.

### C1 — Carry (anchor, from R102)

**15/15 positive weeks, worst week +0.014%, maxDD ≈ 0 (price-neutral).**
The ONLY mechanism that passes every consistency gate — priced in dollars instead:
~0.27%/month on notional in the current regime (≈$1.3/mo per $500).

---

## Round-1 verdict

| Candidate | Consistency gates | Verdict |
|---|---|---|
| C2 low-vol basket | 0/7 (48.9% posW, DD −63%) | ❌ REJECTED |
| C3 dip-ladder | 0/7 (below breakeven by construction) | ❌ REJECTED |
| C1 carry | all (15/15 weeks) | ✅ only survivor — the floor |

**The sharpened law of this hunt:** *in crypto at retail costs, statistical
consistency (many small wins) is mathematically priced away — whatever survives must
earn contractually (funding, basis) rather than statistically.* Therefore the queue
refocuses on contractual/carry-family income and on one remaining statistical angle
with a real asymmetry left:

**R104 queue:** (a) **basis trade** — spot+short-perp when annualized basis > 8%
(episodic, historically fat in bull phases; OKX spot+perp data in hand); (b) **OI /
liquidation data probe** for C4 cascade-fade; (c) carry sizing research — is a small
champion sleeve (frozen config, 0.25% risk) stacked on carry enough to bend the $/month
without breaking the never-red-week property?

*Files: `quantlab_r103_consistency.py` · `r103_stdout.log` · `r103_v1_weekly.csv` ·
`r103_v2_weekly.csv` · `r103_dip_trades.csv` (33k trades) · charter:
`CONSISTENCY_HUNT_BEGINS.md`*
