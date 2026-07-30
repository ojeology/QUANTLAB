# QUANTLAB AI — R056 — E3 Regime Shift Forensic Investigation

**Frozen Environment:** `BBW_LO+RV_LO+DST_NR+PRG_VH`

## Fold Performance

| Fold | Group | PF | n | WR | Net$ |
|---|---|---|---|---|---|
| F1 | WIN | 3.022 | 26 | 65.4% | $+2130 |
| F2 | WIN | 4.184 | 22 | 72.7% | $+2259 |
| F3 | LOSE | 0.460 | 9 | 22.2% | $-445 |
| F4 | LOSE | 0.270 | 21 | 14.3% | $-1533 |
| F5 | REC | 1.276 | 18 | 44.4% | $+324 |

## Key Regime Changes

| Rank | Metric | F1+F2 | F3+F4 | Delta% | Cohen's d | Significance |
|---|---|---|---|---|---|---|
| 1 | ATR Rank (pct) | 45.5268 | 47.6835 | +4.7% | 1.483 | ** |
| 2 | Hurst Exponent | 0.5180 | 0.5285 | +2.0% | 0.892 | * |
| 3 | Avg ATR | 0.0049 | 0.0042 | -13.5% | -0.868 |  |
| 4 | Realised Volatility | 0.6929 | 0.8010 | +15.6% | 0.856 | * |
| 5 | Avg Candle Body (%) | 0.5130 | 0.5906 | +15.1% | 0.802 | * |
| 6 | Pct Bullish Candles | 0.4745 | 0.4827 | +1.7% | 0.785 |  |
| 7 | Daily Range (%) | 1.0675 | 1.1824 | +10.8% | 0.712 |  |
| 8 | Prev Candle Range (%) | 1.0674 | 1.1824 | +10.8% | 0.711 |  |

## Hypotheses

| ID | Hypothesis | Score |
|---|---|---|
| H3 | Volatility regime changed | 50/100 |
| H4 | Breakout quality deteriorated | 50/100 |
| H5 | Session behaviour changed | 50/100 |
| H2 | E3 fails during mean-reverting markets | 45/100 |
| H6 | No identifiable cause (random variation) | 20/100 |
| H1 | E3 only works during persistent trends | 0/100 |

## Conclusions

- **Win period (F1+F2):** PF=3.490, n=48
- **Lose period (F3+F4):** PF=0.323, n=30
- **Recovery (F5):** PF=1.276, n=18
- **Regime change:** YES — 11 metrics with d>0.30
- **Strongest hypothesis:** H3 — Volatility regime changed (50/100)
- **Recommendation:** REMAIN FROZEN → ONE FINAL RESEARCH PHASE
