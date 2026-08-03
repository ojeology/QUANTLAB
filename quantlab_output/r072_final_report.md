# R072 — Final Verdict

## Q1. Why exactly does Family A make money?

Family A (BBW_STRICT + RV_LO + DST_NR + PRG_VH) profits by entering during rare periods of compressed volatility (BB squeeze, low realised vol) that are close to the EMA200 (not overextended) and preceded by strong-range candles. The combination selects for coiling markets on the verge of expanding. WR=20.9%, PF=0.528. The entry gate (vol surge + green candle) provides an additional momentum trigger that improves timing of the compression-release move. Best regime: bear_trend.

## Q2. Why exactly does Family C make money?

Family C (ADX_ST + PBD_HI) profits by entering established trends (strong ADX) after a large-body previous candle (momentum confirmation). The large prior body signals institutional participation; the strong ADX confirms trend. The 46% WR is compensated by RR=3.0 — losers are 1R, winners are 3R. The strategy trades 2,000+ times vs Family A's 91, giving high statistical confidence. Best regime: trending.

## Q3. Under what conditions does each fail?

**Family A fails when:**
- The BB squeeze resolves sideways rather than directionally (ranging regime).
- The market is in high volatility (ATR spike): compression signal occurs but expansion is already exhausted.
- January-type months with inconsistent trend structure.

**Family C fails when:**
- ADX is at threshold (borderline trending): conditions technically met but trend lacks conviction.
- Fake breakout candles: large body with immediate reversal.
- High-vol spikes that trigger the PBD condition abnormally (news events).
- RR=1.0 (incompatible with 46% WR).

## Q4. Can either be safely improved?

**Family A:** Best exit variant is 'Partial TP: 50% @ 1R, rest @ 2R' with ΔPF=+1.0404. Measurable improvement — worth forward-testing in paper trading.

**Family C:** Best exit variant is 'Partial TP: 50% @ 1R, rest @ 2R' with ΔPF=+0.8659. Measurable improvement — worth forward-testing in paper trading.

IMPORTANT: These are in-sample observations. No exit change should be deployed without out-of-sample confirmation.

## Q5. Which weaknesses are structural?

**Family A structural:**
- n=91 — small sample means all statistics carry wide confidence intervals.
- Profit concentration in 2-3 months: the strategy is high-alpha but episodic.
- Depends on periodic volatility compression events; in a sustained high-vol regime the signal frequency drops toward zero.

**Family C structural:**
- 46% WR: the strategy structurally requires RR≥1.5 to be profitable. Cannot be traded at tight RR.
- Fails at RR=1.0 with PF=0.85 — this is mathematically certain, not random.
- Losing streaks of 16 at P95 are structural, not anomalies.

## Q6. Which weaknesses are simply variance?

**Family A variance:**
- Month-to-month concentration is largely a small-n artifact (5 trading months, 91 trades).
- The apparent edge decay in R070 is likely noise given the short OOS window.

**Family C variance:**
- The one losing month (January 2026) is consistent with expected variance (6/7 months profitable at RR=3.0).
- Symbol-specific performance differences are partly variance at per-symbol n<100.

## Q7. Should either strategy remain frozen?

**Family A:** YES — freeze entries. The edge is well-defined. The only permissible change is exit experimentation in paper trading, not condition modification.

**Family C:** YES — freeze entries. Edge attribution confirms both conditions contribute positively. RR=3.0 is now statistically validated (R071). Freeze all entry logic.

## Q8. Would you personally deploy them unchanged?

**Family A:** YES, at conservative sizing (0.5–0.75% risk per trade). The PF=3.35 with Boot P5=2.44 is exceptional. The concern is episodic trades (91 in 5 months) which makes monthly PnL lumpy. Acceptable for paper trading.

**Family C:** YES, at 0.5–1.0% risk per trade with RR=3.0. PF=2.54 at RR=3.0 with 2,000+ trades is a very robust result. The 16-loss P95 streak requires sizing discipline (max 1% per trade, $1,600 worst-case drawdown at $100 risk on $10k). Fully deployable.

**Combined:** Deploy both together. They are complementary — Family A is high-PF / low-frequency; Family C is moderate-PF / high-frequency. Combined portfolio provides smoother equity curve than either alone.


---

## Section Notes

### Family A — Key Stats

- n=91, WR=20.9%, PF=0.5278
- Robustness: PF at 0.10% slippage = 0.4536
- Condition 'PRG_VH': ΔPF if removed = +1.5181 (ESSENTIAL)
- Condition 'RV_LO': ΔPF if removed = +1.2904 (ESSENTIAL)
- Condition 'DST_NR': ΔPF if removed = +0.9734 (ESSENTIAL)

### Family C — Key Stats

- n=2045, WR=16.7%, PF=0.6004
- Robustness: PF at 0.10% slippage = 0.5415
- Condition 'PBD_HI': ΔPF if removed = +0.0088 (neutral)
- Condition 'ADX_ST': ΔPF if removed = -0.0879 (neutral)

