# R073 — Real-Edge Hunt (corrected SL/TP engine)

**Date:** 2026-08-06  |  **Engine:** bot-faithful rolling walk-forward (IS_LOOKBACK=500, RECAL=168b, entry=next close, SL/TP intrabar TP-first, 1 pos/symbol, 1% compounding risk)

**Symbols:** 52  **Time:** 47s


## Family A (RR=2.0)

| Variant | n | WR | PF | Exp $ | MDD% | Calmar | Net $ | Boot P5 | ΔPF vs E0 [P5, P95] |
|---|---|---|---|---|---|---|---|---|---|
| E0_base | 390 | 36.2% | 1.133 | +8.46 | -42.2% | 0.80 | +3359 | 0.955 | [+0.00, +0.00] |
| E1_be1r | 392 | 17.1% | 0.761 | -10.71 | -43.4% | -0.82 | -3573 | 0.598 | [-0.51, -0.23] ❌ |
| E2_partial | 392 | 17.1% | 0.571 | -19.26 | -57.3% | -0.94 | -5376 | 0.444 | [-0.70, -0.43] ❌ |
| E3_trail | 393 | 34.1% | 0.544 | -16.11 | -48.2% | -0.98 | -4740 | 0.425 | [-0.77, -0.42] ❌ |
| E4_time24 | 390 | 37.7% | 1.171 | +10.61 | -40.1% | 1.13 | +4532 | 0.985 | [+0.01, +0.07] ✅ |
| E5_widesl | 387 | 35.4% | 1.096 | +6.20 | -56.8% | 0.39 | +2219 | 0.921 | [-0.20, +0.10] |
| E6_sigentry | 395 | 42.8% | 1.496 | +28.35 | -38.4% | 5.03 | +19308 | 1.264 | [+0.15, +0.55] ✅ |

### RR sweep (E6_sigentry variant)
| RR | n | WR | PF | Exp $ | MDD% | Net $ |
|---|---|---|---|---|---|---|
| 1.00 | 408 | 59.3% | 1.458 | +18.63 | -22.8% | +10951 |
| 1.50 | 403 | 49.9% | 1.493 | +24.69 | -31.1% | +16181 |
| 2.00 | 395 | 42.8% | 1.496 | +28.35 | -38.4% | +19308 |
| 2.50 | 390 | 36.4% | 1.431 | +27.44 | -44.4% | +17562 |
| 3.00 | 390 | 31.5% | 1.382 | +26.15 | -51.4% | +15916 |

### Robustness of best variant **E6_sigentry**
- LOO-symbol PF floor: **1.446** (drop AAVE_USDT_SWAP)
- Monte Carlo (5,000 paths, 1% compounding): P(profit vs start) = **100.0%**, mean net = $+20,510, DD P5 = -20.1% / P95 = -8.0%


### Cost sensitivity (slippage+fees per side)
| Variant | 0.00% | 0.05% | 0.10% | 0.15% | 0.20% |
|---|---|---|---|---|---|
| E0_base | 1.133 | 0.980 | 0.851 | 0.741 | 0.646 |
| E6_sigentry | 1.496 | 1.296 | 1.128 | 0.985 | 0.861 |

*Breakeven round-trip cost for E6_sigentry: 0.145% per side*

## Family C (RR=3.0)

| Variant | n | WR | PF | Exp $ | MDD% | Calmar | Net $ | Boot P5 | ΔPF vs E0 [P5, P95] |
|---|---|---|---|---|---|---|---|---|---|
| E0_base | 10375 | 23.6% | 0.926 | -5.62 | -100.0% | -1.00 | -9993 | 0.891 | [+0.00, +0.00] |
| E1_be1r | 11210 | 10.0% | 0.588 | -21.04 | -100.0% | -1.00 | -10000 | 0.557 | [-0.37, -0.31] ❌ |
| E2_partial | 11210 | 10.0% | 0.392 | -31.05 | -100.0% | -1.00 | -10000 | 0.371 | [-0.56, -0.51] ❌ |
| E3_trail | 11755 | 32.7% | 0.569 | -16.43 | -100.0% | -1.00 | -10000 | 0.544 | [-0.40, -0.34] ❌ |
| E4_time24 | 10518 | 26.9% | 0.938 | -4.47 | -99.9% | -1.00 | -9977 | 0.904 | [-0.00, +0.02] |
| E5_widesl | 9164 | 22.7% | 0.876 | -9.62 | -100.0% | -1.00 | -10000 | 0.839 | [-0.07, -0.01] ❌ |
| E6_sigentry | 10895 | 23.9% | 0.938 | -4.69 | -100.0% | -1.00 | -9987 | 0.904 | [-0.02, +0.04] |

### RR sweep (E6_sigentry variant)
| RR | n | WR | PF | Exp $ | MDD% | Net $ |
|---|---|---|---|---|---|---|
| 1.00 | 12876 | 50.4% | 1.016 | +0.80 | -94.8% | +4783 |
| 1.50 | 12215 | 39.5% | 0.977 | -1.36 | -97.6% | -9239 |
| 2.00 | 11693 | 32.8% | 0.974 | -1.75 | -98.9% | -9594 |
| 2.50 | 11231 | 27.7% | 0.955 | -3.28 | -99.7% | -9935 |
| 3.00 | 10895 | 23.9% | 0.938 | -4.69 | -100.0% | -9987 |

### Robustness of best variant **E6_sigentry**
- LOO-symbol PF floor: **0.931** (drop BNB_USDT_SWAP)
- Monte Carlo (5,000 paths, 1% compounding): P(profit vs start) = **0.0%**, mean net = $-9,939, DD P5 = -100.0% / P95 = -98.7%


### Cost sensitivity (slippage+fees per side)
| Variant | 0.00% | 0.05% | 0.10% | 0.15% | 0.20% |
|---|---|---|---|---|---|
| E0_base | 0.926 | 0.840 | 0.765 | 0.698 | 0.640 |
| E6_sigentry | 0.938 | 0.851 | 0.775 | 0.709 | 0.650 |

*E6_sigentry not profitable even at zero cost (PF=0.938)*

## Verdict

**Family A:** E0 PF=1.133 → best (E6_sigentry) PF=1.496 | GO (positive PF, survives LOO, MC P>95%)

**Family C:** E0 PF=0.926 → best (E6_sigentry) PF=0.938 | NO-GO (unprofitable under realistic exits)

## Recommendation for demo_bot.py

- Family A: adopt exit variant `E6_sigentry` (entry at signal close for E6). Update `STRATEGIES['FamilyA']` exit logic in demo_bot.py after live confirm.

- Family C: **remove from demo bot or disable** — negative expectancy under the bot's own execution model.


## Methodological notes

- Rolling engine uses all data out-of-sample (thresholds from past 500 bars only) and mirrors demo_bot.py execution exactly — replaces the R066 proxy engine as the project's authoritative evaluation.

- ΔPF bootstrap is PAIRED on aligned trades (same entries, differing exits, matched by symbol+entry time). ✅ = 90% CI entirely above 0.

- Variants were pre-registered; no exit was tuned on outcomes. Any adopted change still needs live paper confirmation.
