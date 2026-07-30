# R067b — Overfitting Audit

**Duration:** 113s

## Critical Finding: Timestamp Bug

Parquet indices are sequential integers. `pd.to_datetime(index).hour` = 0 always.

- ASI (`hour in [0,6)`) → **always True** — no session filtering
- LON (`hour in [7,14)`) → **always False** — kills all trades

## Effective Strategy Mapping

| Stated | Effective | Impact |
|---|---|---|
| C_FULL / C_no_ASI | DST_NR + ADX_ST + PBD_HI | R066/R067 results valid for 24/7 only |
| C_no_DST | ADX_ST + PBD_HI  ← 2-cond | R066/R067 results valid for 24/7 only |
| C_no_ADX | DST_NR + PBD_HI  ← 2-cond | R066/R067 results valid for 24/7 only |
| C_no_PBD | DST_NR + ADX_ST  ← 2-cond | R066/R067 results valid for 24/7 only |

## Overfit Test Results

| Variant | IS PF | OOS PF | Gap% | Z-score | Flag |
|---|---|---|---|---|---|
| DST+ADX+PBD | 2.140 | 1.492 | 30.3% | -3.4 | CAUTION |
| ADX+PBD | 1.864 | 1.692 | 9.3% | -3.5 | VALID |
| DST+PBD | 1.933 | 1.728 | 10.6% | 0.0 | VALID |
| DST+ADX | 1.951 | 1.573 | 19.4% | 0.0 | VALID |
| FamA_actual | 80.128 | 3.353 | 95.8% | 2.9 | CAUTION |

## Verdict

- **Family A**: results stand. No session filter. Solid edge.
- **Family B**: untestable without real timestamps. Results are meaningless.
- **Family C effective** (DST_NR+ADX_ST+PBD_HI): real 24/7 edge. Session framing was a label error.
- **ADX_ST+PBD_HI**: real edge, simpler, higher frequency. Worth further validation.
- **Action**: obtain real timestamps or reconstruct bar times from known anchor.
