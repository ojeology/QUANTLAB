---
name: RR per-strategy decision
description: R071 full-bootstrap RR sweep results and per-strategy RR settings
---

## Decision (R071, 2026-07-30)

Family A → **RR=2.0** (unchanged)
Family C → **RR=3.0** (upgraded from 2.0)

## Why

Full 2,000-sample bootstrap across RR 1.0–3.0:

**Family A (n=91, WR=62.6%)**
- Every RR shows "no sig" vs baseline RR=2.0 (CIs all span 0).
- RR=3.0 P(better)=90.8% but CI=[-0.67, +4.71] — not statistically convincing.
- n=91 is too small to distinguish real improvement from noise.
- **Keep RR=2.0; revisit after accumulating live paper-trade data.**

**Family C (n=2049, WR=45.8%)**
- RR≤1.75: statistically significantly WORSE than RR=2.0 (entire CIs below 0).
- RR=1.0 and 1.25: Boot P5 < 1.0 — genuinely unprofitable at P5.
- RR=2.5+: statistically significantly BETTER (entire CIs above 0).
- RR=3.0: CI=[+0.59, +1.11], P(better)=100% — rock-solid improvement.
- **Upgraded to RR=3.0.**

## How to apply

- `STRATEGIES["FamilyA"]["rr"] = 2.0` in demo_bot.py
- `STRATEGIES["FamilyC"]["rr"] = 3.0` in demo_bot.py
- `process_symbol()` reads `rr = STRATEGIES[strategy_id].get("rr", RR)` and uses it for:
  - TP calculation: `entry_price + rr * atr`
  - Win PNL: `trade["risk_usd"] * rr`
  - Telegram display

## Monthly breakdown note
Family C at RR=3.0 monthly PF: Jan=0.72 (still loss month), Feb=3.25, Mar=2.42,
Apr=2.78, May=2.44, Jun=2.54, Jul=2.29. Six of seven months profitable.
