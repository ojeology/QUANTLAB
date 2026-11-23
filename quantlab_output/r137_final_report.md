# R137 — FUNDING-RATE CARRY — FINAL (2026-11-23) + CHAMPION RETEST + R135 PAPER SEED
## 1) CHAMPION RETEST (Nov 23 — data through 2026-11-23 03:00)
**6-mo PF@c 0.92 (−4.8R, n=93) → HALT HOLDS.** Trajectory: 0.91 → 0.97 → 0.98 → **0.92** — the
recovery rolled back (15 trades, −3.9R since Nov 4). 2026 YTD: +23.0R (PF@c 1.21, n=197). The
kill-switch has now been right twice in one month: benched through the Nov 5→23 bleed at zero
capital risk. Jan-1 rule unchanged; the gate now needs a real recovery, not one trade.

## 2) FUNDING-RATE CARRY (32 coins × 3,378 epochs, refreshed spine): **CLOSED — decisively**
Honest epoch-following design (predict next epoch from last; delta-neutral short-perp+long-spot;
0.10% toggle costs; θ predeclared ∈ {0.01/0.03/0.05%}):
| θ | book APR (notional) | capital-adj (÷2) | 2024 | 2025 | 2026 | deployment |
|---|---|---|---|---|---|---|
| 0.01% | **−0.23%** | −0.11% | +1.09 | −0.31 | −0.31 | 9.0% |
| 0.03% P | **−0.33%** | −0.17% | +0.50 | −0.05 | −0.01 | 4.1% |
| 0.05% | **−0.25%** | −0.12% | −0.23 | −0.01 | +0.00 | 1.6% |
ALL gates failed (yearly, APR≥5%, sign-stability). WHY: per-epoch funding magnitudes are 0.01–0.05%
while a two-leg toggle costs 0.10% — the cost wall (R085–R095) reappears at 8H frequency; and the
persistence premium is consumed by adverse selection (you eat the negative surprise epochs too).
**The floor's boring ~5% lending yield remains the best income instrument in the lab** — it needs
no second leg, carries no basis risk, and just out-earned the sophisticated version by 5 points.
This is the round's product: the intuitive income idea dies clean, with a receipt.

## 3) R135 PAPER TRACKER — SEEDED (protocol disclosed: deterministic engine, retro-seeded from
2026-11-05, tracked forward from 2026-11-23; log: r135_paper_log.csv)
First forward window (18 days): **17 closed, net −5.5R, PF@c 0.47, WR 12% (2/17)** — winners: ARB
short +1.94, SUI short +2.82. 4 OPEN longs (ATOM/NEAR/OP/SUI, Nov 21-22). Read WITHOUT verdict:
n=17 is noise-scale for a PF-1.42 book; but WR 2/17 vs backtest 34% ≈ P~0.1 — unlucky-but-possible;
the paper machine now accumulates REAL forward scars for the Jan-1 review. Observation (no test):
the champion bled −3.9R in the same fortnight — both machines whipsawing together is exactly the
high-density-chop regime the queued frequency-gate hypothesis describes.

## LAWS EXTENDED
- Cost wall law, general form: at ANY frequency, an edge whose per-event magnitude ≈ 10% of
  round-trip costs is dead regardless of direction skill. Funding carry: magnitude 0.01–0.05%,
  cost 0.10%. Same law as 5m scalping. The floor yield wins income not because it's clever but
  because it has no toggle costs at all.
Artifacts: r137_carry_epochs.csv · r135_paper_log.csv · repro: quantlab_r137_funding_carry.py,
quantlab_r135_paper_tracker.py.
