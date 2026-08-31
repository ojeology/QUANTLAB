"""
Split the cross-regime model (trained on 2024 ONLY) into standalone
2025 and 2026 results, so each year's PF@cost / profitable-months is visible.

Method: SVM fit on 2024 mldf; keep-threshold for q computed from 2024
train predictions; applied separately to 2025 and 2026 test sets.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from ql_engine import stats_from_trades, cost_adjusted_rs, pf_of_rs
from svm_deploy import SVMQ75, load_universe, FEATS

print("[load] building universe + features …", flush=True)
feats, above20, raw, breadth, breadth_pct, mldf = load_universe()
mldf["breadth_frac"] = breadth.reindex(mldf["ts"]).fillna(0.5).values

S25 = pd.Timestamp("2025-01-01", tz="UTC")
S26 = pd.Timestamp("2026-01-01", tz="UTC")
E26 = pd.Timestamp("2027-01-01", tz="UTC")
TRAIN24 = mldf[mldf.ts < S25]
TEST2025 = mldf[(mldf.ts >= S25) & (mldf.ts < S26)]
TEST2026 = mldf[mldf.ts >= S26]
print(f"[split] train(2024)={len(TRAIN24)}  2025={len(TEST2025)}  2026={len(TEST2026)}", flush=True)


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


def trades_in(start, end, kept_ts):
    return [t for t in raw if t["entry_time"] >= start and t["entry_time"] < end
            and t["entry_time"] in kept_ts]


for q in [0.65, 0.75]:
    model = SVMQ75(q=q).fit_mldf(TRAIN24)
    preds_train = model.clf.predict_proba(model.scaler.transform(TRAIN24[FEATS].fillna(0).values))[:, 1]
    thr = np.quantile(preds_train, 1 - q)
    p25 = model.clf.predict_proba(model.scaler.transform(TEST2025[FEATS].fillna(0).values))[:, 1]
    p26 = model.clf.predict_proba(model.scaler.transform(TEST2026[FEATS].fillna(0).values))[:, 1]
    k25 = set(TEST2025[p25 >= thr]["ts"])
    k26 = set(TEST2026[p26 >= thr]["ts"])
    print("\n" + "=" * 70)
    print(f"CROSS-REGIME q={q}  (SVM trained on 2024 only; threshold from 2024)")
    print("=" * 70)
    evaluate(f"2025 ONLY  (trained-on-2024 → 2025)", trades_in(S25, S26, k25))
    evaluate(f"2026 ONLY  (trained-on-2024 → 2026)", trades_in(S26, E26, k26))
print("\n[done]")
