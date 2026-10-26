# ACCOUNTING & BACKTEST STANDARDS (locked v1, audit-era)

Enforced by `ql_accounting.py` (`LadderAccount`, `performance_report`). Any script that emits
performance numbers MUST route P&L through it. Deviations require a SUPERSEDED label on old outputs.

## The seven conventions
1. **Signal/exec lag:** decisions at close of bar *t* → weights effective from bar *t+1*. No exceptions.
2. **Costs once:** charged on the execution bar only. Never per-bar-of-holding, never overwritten.
3. **Funding:** `−w·f` every bar the weight is held (funding is a drag when long in positive funding).
4. **One basis:** CAGR/vol/Sharpe/DD all computed from the same leveraged equity curve. No unleveraged APR next to leveraged DD.
5. **Intrabar conservatism:** when a bar's OHLC cannot order TP vs SL, book **SL-first** (label any optimistic exception).
6. **Holdout ledger:** every touch of a reserved window is logged in `docs/EXPOSURE_LEDGER.md` before results are read.
7. **Ledger invariants:** every bar marked exactly once; ±60% daily tripwire; no unapplied pending rebalance; `verify()` runs before any report.

## Caller discipline (the off-by-one law)
The lab's recurring defect (R104, R107, R118, and R119's own first draft) is marking bar *i*
with the position decided at close *i*. Under the standard this is **structurally impossible**:
compute bar-*i* P&L ONLY from `acct.effective_weights(i)`. Never from a local `pos` variable.
Regression test: `tests/test_accounting.py::test_weights_are_sole_pnl_source`.

## Reporting minimums
CAGR, vol, Sharpe, maxDD, posW, worstW, per-year table, trade/rebalance count, cost model stated,
holdout status stated (incl. contamination), and the charter gate count. No "live-ready" language, ever.

## Addendum (R129 lesson)
Retention gates MUST be evaluated before significance gates: a gate that vetoes 100% of trades
passes every significance test vacuously (PF of an empty book = infinity). R129 Cell A is the example.

## Addendum 2 (R131 lesson)
Probability lift is NOT expectancy. A precursor that raises P(big move) can still be untradeable:
direction-conditioning, per-trade costs at event frequency, and path (stop placement vs the mean
path) each independently consume lifts of 1.2-1.5x. Any new signal must clear PF@c gates on BOTH
halves before 'lift' language is used in reports.
