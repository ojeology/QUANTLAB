# R128 — DISSECTION BATTERY II: SILVER + OIL — PRESPEC (LOCKED 2026-10-26, pre-analysis)
Operator directive: keep dissecting assets. Targets: SILVER (SI=F + OKX XAG perp) and WTI CRUDE
(CL=F + OKX CL perp) — the two remaining retail-favorite assets reachable AND tradeable on the
operator's rails. Status when locked: zero silver/oil signal tests in repo history; all discovery-grade,
same era structure as R127: SELECTION 2000→2015 · CONFIRM 2016→2026.

## Registered (verdict-bearing) cells per asset
1. TREND GRID — sign of past-Nd return, next-day position, N ∈ {20,60,120,250} × {long/flat,
   long/short}. Costs 0.10%/unit position change. Gates: selection Sharpe ≥ 0.5 AND
   monthly-block bootP(total>0) ≥ 0.95 AND confirm Sharpe > 0. All three → PROMISING-candidate
   WITH mandatory dominance check vs the asset's own buy-and-hold (R127 lesson: absolute gates
   can pass on beta alone).
2. COMPRESSION-POP on OKX 1H (champion machinery raw): gates n ≥ 30 ∧ PF@c ≥ 1.3 ∧ bootP ≥ 0.95
   (0.07%/side; TP-first disclosed). Expect n-starvation; reported honestly either way.

## Mapping-only (report, decides nothing)
Anatomy per asset (era CAGR/vol/Sharpe/deepest bear, 2026 YTD, $300 translation, BTC corr) ·
event anatomy (>2σ days fwd 5d/20d vs unconditional) · session anatomy (OKX 1H vol concentration).

## ML ADDENDUM (diagnostic, answers operator's question; covers GOLD too)
Per asset INCLUDING GOLD (GC=F): walk-forward supervised direction models — features = lagged
returns (1/5/10/20d), 20d+60d vol, momentum, dow; target = 20d-ahead sign (non-overlapping steps);
expanding-window yearly refits 2010→2026; models: logistic L2 + gradient boosting. Report OOS AUC
(+ economic check: threshold-0.55 long/flat vs buy-and-hold net of 0.10%). Diagnostic vocabulary
only: "ML finds information beyond drift: YES/NO per asset." Model-class conclusions already closed
by R124; this measures per-asset information content. NOT verdict-bearing for promotion.

## Verdict vocabulary
Per asset: PROMISING-CANDIDATE (rare; then forward-paper gauntlet) / MAPPED (default) — with the
dominance line printed under every survivor. No iteration, no re-windowing, no re-costing.
