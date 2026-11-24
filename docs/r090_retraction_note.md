# CORRECTION — R090 H2_range_fade RETRACTED (operator-issued 2026-11-24)
The R090 H2_range_fade result (n=107, PF 4.67, PF@0.05% 2.24, holdout PF 4.10 — see
quantlab_output/r090_final_report.md) is RETRACTED. The operator's causal audit found that the
session-range feature used the FULL day's low before the session had ended (lookahead bias):
trade decisions referenced a price level not yet established at entry time.
Per audit-preserve rule, the original row remains in r090_final_report.md, marked by this note;
all downstream use of H2_range_fade is void. Causality checks are now a permanent pre-candidate
stage in the lab's process (see ACCOUNTING_AND_BACKTEST_STANDARDS.md lineage).
