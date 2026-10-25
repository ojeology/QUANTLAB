# R124 — ML CAPACITY LADDER (diagnostic, 2026-10-24)
Operator question: "upgrade the ML — a more sophisticated variant — and test again?" Method: five
model classes, ascending capacity, IDENTICAL data/features/splits. FX: train 2016-2024 (n=537),
OOS 2025+ (n=90). Crypto: train Jun-Dec 2025 (n=149), OOS 2026 (n=171). Metric: AUC (0.5 = coin flip).
Pre-declared prediction: OOS skill does NOT rise with capacity. RESULT: prediction confirmed exactly.

| model (dumb → fancy) | FX train | FX OOS | CRYPTO train | CRYPTO OOS |
|---|---|---|---|---|
| logreg L2 C=0.01 | 0.595 | 0.324 | 0.690 | 0.498 |
| logreg L2 C=1 | 0.593 | 0.318 | 0.733 | 0.509 |
| SVM linear | 0.582 | 0.353 | 0.718 | **0.572** |
| SVM rbf (incumbent) | 0.857 | 0.410 | 0.933 | 0.536 |
| grad-boost 300×d3 | **1.000** | 0.466 | **1.000** | 0.532 |

## Findings
1. **FX: every model is BELOW 0.5 out-of-sample** — dumb (0.32) to fancy (0.47). The fancy model
   memorized the training set perfectly (AUC 1.000) and still failed forward. The problem is EMPTY;
   sophistication cannot extract information that does not exist. Impossible is the right word here.
2. **Crypto: flat/noisy across capacity (0.50–0.57)** — the nominal best is the LEAN linear SVM
   (0.572), not the fancy ones (0.53); with n=171 all of these are within noise of 0.5. Direction
   of evidence: simplicity ≥ sophistication at this sample size, never the reverse.
3. In-sample AUC rises monotonically with capacity on BOTH problems (0.58→1.00) while OOS stays
   flat/inverted — the bias-variance tradeoff made visible: extra capacity converts entirely into
   memorization at n≈150–550.

## The upgrade paths that CAN work (and their status)
- MORE DATA: the only reliable lever — the walk-forward champion refits on every new paper month;
  this upgrade is ALREADY RUNNING via the calendar.
- LESS CAPACITY / FEWER FEATURES: the linear-SVM observation is recorded as a candidate for the
  re-entry stack — to be pre-registered at re-entry, not acted on now (seen data).
- NEW INFORMATION (features that carry signal, e.g. true order flow): blocked by data availability.

## Verdict
DIAGNOSTIC ONLY (no edge claimed, no closed direction reopened). "More sophisticated ML" is
formally answered: it cannot help on either problem — FX is informationally empty, crypto is
sample-starved with a weak effect, and in both cases added capacity becomes memorization.
