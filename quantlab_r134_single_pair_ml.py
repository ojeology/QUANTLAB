#!/usr/bin/env python3
# R134 — SINGLE-PAIR ML, BLIND-YEAR TEST (prespec: docs/RESEARCH_FRONTIER_R134.md, LOCKED)
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import binomtest

warnings.filterwarnings("ignore")
C = "quantlab_cache/bnfut/BTC_1D.csv"
SPLIT = pd.Timestamp("2026-01-01")

g = pd.read_csv(C, parse_dates=["date"]).set_index("date")["close"].sort_index()
g.index = g.index.tz_localize(None)
r = g.pct_change()
F = pd.DataFrame(index=g.index)
for k in (1,2,3,5,10,20): F[f"lag{k}"] = r.shift(k)
for k in (20,60,120): F[f"mom{k}"] = g.pct_change(k)
for k in (10,30,60): F[f"vol{k}"] = r.rolling(k).std()
for k in (20,60,120):
    F[f"hi{k}"] = g/ g.rolling(k).max() - 1
    F[f"lo{k}"] = g/ g.rolling(k).min() - 1
for k in (20,50,200): F[f"sma{k}"] = g/ g.rolling(k).mean() - 1
for d in range(5): F[f"dow{d}"] = (g.index.dayofweek == d).astype(float)
delta = g.diff()
up = delta.clip(lower=0).rolling(14).sum()
dn = (-delta.clip(upper=0)).rolling(14).sum()
F["rsi14"] = up/(up+dn)
macd = g.ewm(span=12).mean() - g.ewm(span=26).mean()
F["macd"] = macd / g
F["bpb"] = (g - g.rolling(20).mean()) / (2*g.rolling(20).std())
atr = ((g.rolling(2).max() - g.rolling(2).min())).rolling(14).mean()
F["atr"] = atr / g
F = F.dropna()
tgt1 = (g.shift(-1) > g).astype(int).reindex(F.index)
tgt5 = (g.shift(-5) > g).astype(int).reindex(F.index)
valid = tgt1.notna() & tgt5.notna()
F, tgt1, tgt5 = F[valid], tgt1[valid].astype(int), tgt5[valid].astype(int)
FEATS = list(F.columns)
print(f"sample: {len(F)} days {F.index.min().date()} -> {F.index.max().date()} | features: {len(FEATS)}")

REFIT = 21
test_mask = F.index >= SPLIT
test_idx = F.index[test_mask]
train_all = F.index < SPLIT
print(f"train: {train_all.sum()} days | blind test: {test_mask.sum()} days ({test_idx.min().date()} -> {test_idx.max().date()})")

preds = {m: [] for m in ("logreg","gbm")}
pred_dates = []
next_refit = 0
i = 0
positions = []
for d in test_idx:
    tr = F.index < d
    Xtr, ytr = F[tr], tgt1[tr]
    if tr.sum() > next_refit:
        sc = StandardScaler().fit(Xtr.values)
        m1 = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Xtr.values), ytr)
        m2 = GradientBoostingClassifier(n_estimators=300, max_depth=3).fit(Xtr.values, ytr)
        next_refit = tr.sum() + REFIT
    p1 = m1.predict_proba(sc.transform(F.loc[[d]].values))[0,1]
    p2 = m2.predict_proba(F.loc[[d]].values)[0,1]
    preds["logreg"].append(p1); preds["gbm"].append(p2)
    pred_dates.append(d)

y_test = tgt1.reindex(test_idx).values
y5_test = tgt5.reindex(test_idx).values
print("\n" + "="*100)
print("  R134 — SINGLE-PAIR ML, BLIND 2026 (BTC daily) — walk-forward, expanding, refit/21d")
print("="*100)
res = {}
for m in ("logreg","gbm"):
    p = np.array(preds[m])
    acc = accuracy_score(y_test, (p > 0.5).astype(int))
    auc = roc_auc_score(y_test, p)
    pval = binomtest(int((p > 0.5).astype(int).sum()), len(p), 0.5).pvalue
    # economic: long-if-up cash-if-down, net of costs on position changes
    pos = (p > 0.5).astype(float)
    fret = g.pct_change().reindex(test_idx).values
    turnover = np.abs(np.diff(np.concatenate([[0], pos])))
    net = pos*fret - turnover*0.0005
    bh = fret
    def cagr(x):
        x = x[~np.isnan(x)]
        yrs = len(x)/365
        return ((1+x).prod()**(1/yrs)-1)*100 if yrs > 0 else np.nan
    res[m] = dict(acc=acc, auc=auc, p=pval, cagr=cagr(net), bh=cagr(bh), n=len(p))
    print(f"  {m:<7} acc {acc*100:.1f}%  AUC {auc:.3f}  binom-p {pval:.3f}  | sign-strategy CAGR {cagr(net):+.1f}% vs B&H {cagr(bh):+.1f}%")
best = max(res, key=lambda m: res[m]["auc"])
ok = res[best]["auc"] >= 0.53 and res[best]["cagr"] >= res[best]["bh"]
print(f"\n  GATES: best-AUC model '{best}': AUC {res[best]['auc']:.3f} {'>=' if res[best]['auc']>=0.53 else '<'} 0.53 | "
      f"net CAGR {'>=' if res[best]['cagr']>=res[best]['bh'] else '<'} B&H")
print("  VERDICT: " + ("ML carries OOS information on this pair" if ok else
      "NO OOS INFORMATION — single-pair ML mapped-closed (consistent with R124's law; now measured)"))

# report-only: t2 (5-day target) with gbm
m2 = None
p5 = []
for j, d in enumerate(test_idx):
    tr = F.index < d
    if j % REFIT == 0:
        sc5 = StandardScaler().fit(F[tr].values)
        m5 = GradientBoostingClassifier(n_estimators=300, max_depth=3).fit(F[tr].values, tgt5[tr])
    p5.append(m5.predict_proba(F.loc[[d]].values)[0,1])
auc5 = roc_auc_score(y5_test, p5)
print(f"  report-only t2 (5d direction) AUC: {auc5:.3f}")
gbm_imp = None
try:
    gbm_imp = sorted(zip(FEATS, m2.feature_importances_ if m2 is not None else []), key=lambda x: -x[1])[:6]
except Exception:
    pass
pd.DataFrame({"date": test_idx, "y": y_test,
              "p_logreg": preds["logreg"], "p_gbm": preds["gbm"]}).to_csv("quantlab_output/r134_predictions.csv", index=False)
print("saved: r134_predictions.csv")
