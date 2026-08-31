"""
IMPROVEMENTS — 4 experiments toward the 2027-01-01 go-live.

Baseline: SVM q0.75, trained pre-2026, tested on 2026 (the champion).
  #2 STACKING   : SVM-kept ∩ (breadth50 + VolCeil) on 2026
  #3 Q-SWEEP    : sweep keep-rate q on 2026, threshold from pre-2026 train preds
  #4 CROSS-REGIME: train on 2024 ONLY, test on 2025+2026 (stricter OOS)

All use fees (0.05%) and the exact engine from svm_deploy.SVMQ75.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from ql_engine import stats_from_trades, cost_adjusted_rs, pf_of_rs
from svm_deploy import SVMQ75, load_universe, FEATS

print("[load] building universe + features (fresh OKX top-up) …", flush=True)
feats, above20, raw, breadth, breadth_pct, mldf = load_universe()
mldf["breadth_frac"] = breadth.reindex(mldf["ts"]).fillna(0.5).values
print(f"[load] symbols={len(feats)}  total RAW trades={len(mldf)}", flush=True)

FREEZE = pd.Timestamp("2026-01-01", tz="UTC")


def monthly_profile(trades):
    if not trades:
        return dict(prof=float("nan"), worst=0, tpm=0.0)
    df = pd.DataFrame(trades); df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum(); flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0; worst = max(worst, cur)
    days = max(1, (df["entry_time"].max() - df["entry_time"].min()).days)
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df) / (days / 30.0))


def evaluate(name, trades):
    if not trades:
        print(f"\n[{name}] no trades"); return
    s = stats_from_trades(list(trades))
    pf_c = pf_of_rs(cost_adjusted_rs(list(trades), 0.05))
    mp = monthly_profile(trades)
    print(f"\n[{name}]\n  n={len(trades)}  WR={s['wr']:.1%}  PF={s['pf']:.3f}  "
          f"PF@0.05%={pf_c:.3f}  MDD={s['mdd']:.1%}  prof-months={mp['prof']:.1%}  "
          f"worst={mp['worst']}  t/mo={mp['tpm']:.1f}")


def trades_from(test_start, kept_ts):
    test_raw = [t for t in raw if t["entry_time"] >= test_start]
    return [t for t in test_raw if t["entry_time"] in kept_ts]


# ── Baseline: SVM q0.75 on 2026 ────────────────────────────────────────────────
TRAIN = mldf[mldf.ts < FREEZE]
TEST = mldf[mldf.ts >= FREEZE]
model = SVMQ75(q=0.75).fit_mldf(TRAIN)
kept_ts, _ = model.keep_mldf(TEST, 0.75)
print("\n" + "=" * 74)
print("BASELINE — SVM q0.75 (train pre-2026, test 2026)")
print("=" * 74)
evaluate("SVM q0.75 (current champion)", trades_from(FREEZE, kept_ts))

# ── #2 STACKING: SVM-kept ∩ (breadth50 + VolCeil) ─────────────────────────────
stack_mask = (TEST["breadth_frac"] > 0.50) & (TEST["atr_rank"] <= 70.0)
TEST_stack = TEST[stack_mask]
kept_stack, _ = model.keep_mldf(TEST_stack, 0.75)
print("\n" + "=" * 74)
print("#2 STACKING — SVM q0.75 + breadth50 + VolCeil (2026)")
print("=" * 74)
evaluate("STACK (SVM ∩ breadth50 ∩ VolCeil)", trades_from(FREEZE, kept_stack))

# ── #3 Q-SWEEP on 2026 ───────────────────────────────────────────────────────
preds_train = model.clf.predict_proba(model.scaler.transform(TRAIN[FEATS].fillna(0).values))[:, 1]
preds_test = model.clf.predict_proba(model.scaler.transform(TEST[FEATS].fillna(0).values))[:, 1]
print("\n" + "=" * 74)
print("#3 Q-SWEEP on 2026 (keep-threshold from pre-2026 train preds)")
print("=" * 74)
print(f"  {'q':>5}{'n':>6}{'WR':>6}{'PF':>8}{'PF@c':>8}{'prof%':>7}{'t/mo':>7}")
best_q, best_pfc = 0.75, -1
for q in [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]:
    thr = np.quantile(preds_train, 1 - q)
    kept = set(TEST[preds_test >= thr]["ts"])
    tr = trades_from(FREEZE, kept)
    if not tr:
        continue
    s = stats_from_trades(list(tr)); pf_c = pf_of_rs(cost_adjusted_rs(list(tr), 0.05))
    mp = monthly_profile(tr)
    print(f"  {q:>5.2f}{len(tr):>6}{s['wr']*100:>5.0f}%{s['pf']:>8.3f}{pf_c:>8.3f}{mp['prof']*100:>6.0f}%{mp['tpm']:>7.1f}")
    if pf_c > best_pfc:
        best_pfc, best_q = pf_c, q
print(f"  -> best q by PF@cost = {best_q}")

# ── #4 CROSS-REGIME: train 2024 only, test 2025+2026 ──────────────────────────
TRAIN24 = mldf[mldf.ts < pd.Timestamp("2025-01-01", tz="UTC")]
TEST_2526 = mldf[mldf.ts >= pd.Timestamp("2025-01-01", tz="UTC")]
model24 = SVMQ75(q=0.75).fit_mldf(TRAIN24)
kept24, _ = model24.keep_mldf(TEST_2526, 0.75)
print("\n" + "=" * 74)
print("#4 CROSS-REGIME — train 2024 ONLY, test 2025+2026 (stricter OOS)")
print("=" * 74)
evaluate("SVM q0.75 (train 2024, test 2025+2026)", trades_from(pd.Timestamp("2025-01-01", tz="UTC"), kept24))

preds_train24 = model24.clf.predict_proba(model24.scaler.transform(TRAIN24[FEATS].fillna(0).values))[:, 1]
preds_test24 = model24.clf.predict_proba(model24.scaler.transform(TEST_2526[FEATS].fillna(0).values))[:, 1]
print("\n  #4b cross-regime q-sweep (threshold from 2024 train preds)")
print(f"  {'q':>5}{'n':>6}{'WR':>6}{'PF':>8}{'PF@c':>8}{'prof%':>7}{'t/mo':>7}")
for q in [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]:
    thr = np.quantile(preds_train24, 1 - q)
    kept = set(TEST_2526[preds_test24 >= thr]["ts"])
    tr = trades_from(pd.Timestamp("2025-01-01", tz="UTC"), kept)
    if not tr:
        continue
    s = stats_from_trades(list(tr)); pf_c = pf_of_rs(cost_adjusted_rs(list(tr), 0.05))
    mp = monthly_profile(tr)
    print(f"  {q:>5.2f}{len(tr):>6}{s['wr']*100:>5.0f}%{s['pf']:>8.3f}{pf_c:>8.3f}{mp['prof']*100:>6.0f}%{mp['tpm']:>7.1f}")

print("\n[done]")
