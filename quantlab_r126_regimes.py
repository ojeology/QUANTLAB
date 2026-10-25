#!/usr/bin/env python3
# R126 — UNSUPERVISED REGIME OVERLAY (prespec: docs/RESEARCH_FRONTIER_R126.md, LOCKED)
import os, sys, glob, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, build_signal_mask, sim_symbol, IS_LOOKBACK, RECAL_EVERY
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from hmmlearn.hmm import GaussianHMM

CACHE = CONFIG["CACHE_FOLDER"]
FIT0, FIT1 = pd.Timestamp("2024-01-28", tz="UTC"), pd.Timestamp("2025-06-01", tz="UTC")
RR = 1.5
K = 3

# ── 1. daily panel features from 1H closes ───────────────────────────────────
closes = {}
for f in glob.glob(os.path.join(CACHE, "*_USDT_SWAP_1H.parquet")):
    sym = os.path.basename(f).replace("_USDT_SWAP_1H.parquet", "")
    if sym.startswith(("XAU","XAG","CL","SOXL","MU","SPCX","AAOI","UB","SNDK","SKHY","BEAT","ALLO","ZBT","SNXX","PUMP","HOME")):
        continue
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True)
    closes[sym] = df["close"].resample("1D").last().dropna()
PX = pd.DataFrame(closes).sort_index()
RETS = PX.pct_change()
f1 = RETS.mean(axis=1)                                   # market drift
f2 = RETS.rolling(20).std().mean(axis=1) * np.sqrt(24)   # vol level (daily units)
f3 = RETS.std(axis=1)                                    # dispersion
F = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3}).dropna()
fit_mask = (F.index >= FIT0) & (F.index < FIT1)
print(f"panel features: {len(F)} days | fit-window days: {fit_mask.sum()}")

scaler = StandardScaler().fit(F[fit_mask].values)
Xs = scaler.transform(F.values)
hmm = GaussianHMM(n_components=K, covariance_type="diag", n_iter=200, random_state=126)
hmm.fit(Xs[fit_mask])
print("HMM fit done. State means (standardized f1,f2,f3):")
print(np.round(hmm.means_, 2))

def state_at(ts):
    """Causal decode: Viterbi on prefix ending at ts (frozen params)."""
    sub = F[(F.index <= ts) & (F.index >= F.index[0])]
    if len(sub) < 100: return -1
    return int(hmm.predict(scaler.transform(sub.values))[-1])

state_series = {ts: state_at(ts) for ts in F.index[::1]}
print("causal decode cached.")

# ── 2. fit-era raw pops -> state ranking ─────────────────────────────────────
from quantlab_ai import CONFIG as CFG
feats_all = {}
for f in glob.glob(os.path.join(CACHE, "*_USDT_SWAP_1H.parquet")):
    sym = os.path.basename(f).replace("_USDT_SWAP_1H.parquet", "")
    if sym.startswith(("XAU","XAG","CL","SOXL","MU","SPCX","AAOI","UB","SNDK","SKHY","BEAT","ALLO","ZBT","SNXX","PUMP","HOME")):
        continue
    try:
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        ft = add_features(df)
        ft.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width",
                          "prev_range_r","prev_body_r"], inplace=True)
        if len(ft) >= IS_LOOKBACK + RECAL_EVERY: feats_all[sym] = ft
    except Exception:
        pass
cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
masks = {s: build_signal_mask(ft, cids, "green", 1.5) for s, ft in feats_all.items()}
raw = []
for sym, ft in feats_all.items():
    for t in sim_symbol(ft, masks[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"raw pops total: {len(raw)}")

def rc_of(tr):
    return tr["r"] - 2.0*(0.05/100.0)*tr["entry"]/tr["atr"]

fit_pops = [t for t in raw if FIT0 <= t["entry_time"] < FIT1]
rows = []
for t in fit_pops:
    ts = t["entry_time"]
    key = F.index[F.index <= ts]
    st = state_series.get(key[-1], -1) if len(key) else -1
    if st >= 0: rows.append((st, rc_of(t)))
rank = pd.DataFrame(rows, columns=["state","rc"])
print(f"fit-era pops with state: {len(rank)}")
state_pf = rank.groupby("state").rc.apply(lambda s: s[s>0].sum()/abs(s[s<0].sum()) if (s<0).any() else 999)
state_n = rank.groupby("state").size()
print("fit-era state economics:")
for st in sorted(state_pf.index):
    print(f"  state {st}: PF@c {state_pf[st]:.2f} (n={state_n[st]})")
favorable = sorted(int(s) for s in state_pf[state_pf >= 1.1].index)
print(f"FAVORABLE states (PF@c>=1.1): {favorable}")
if len(favorable) in (0, K):
    print("\nVERDICT: REJECT (uninformative — all-or-nothing state split)")
    sys.exit(0)

# ── 3. OOS test on incumbent kept book ───────────────────────────────────────
def pf(x):
    x = np.asarray(x,float); w = x[x>0].sum(); lo = abs(x[x<0].sum())
    return w/lo if lo>0 else 999
book = pd.read_csv("quantlab_output/r096_trades_q055.csv", parse_dates=["entry_time"])
book["rc"] = book.r - 2.0*(0.05/100.0)*book.entry/book.atr
book["state"] = [state_series.get(F.index[F.index <= ts][-1], -1) if (F.index <= ts).any() else -1
                 for ts in book.entry_time]
book["fav"] = book.state.isin(favorable)
oos = book[book.entry_time >= FIT1]
print(f"\nOOS kept book: n={len(oos)}  full PF@c={pf(oos.rc):.2f}")
favb, unfb = oos[oos.fav], oos[~oos.fav]
pf_f, pf_u = pf(favb.rc), pf(unfb.rc)
print(f"  favorable   states {favorable}: n={len(favb):>3}  PF@c={pf_f:.2f}  sum={favb.rc.sum():+.1f}R")
print(f"  unfavorable states:            n={len(unfb):>3}  PF@c={pf_u:.2f}  sum={unfb.rc.sum():+.1f}R")
y = oos[oos.entry_time >= pd.Timestamp("2026-01-01", tz="UTC")]
pf_f_y, pf_u_y = pf(y[y.fav].rc), pf(y[~y.fav].rc)
print(f"  2026 alone: fav PF@c={pf_f_y:.2f} (n={y.fav.sum()}) vs unfav PF@c={pf_u_y:.2f} (n={(~y.fav).sum()})")

g1 = pf_f > pf(oos.rc); g2 = len(favb) >= 60; g3 = pf_u < pf_f; g4 = pf_f_y > pf_u_y
print(f"\n  GATES: G1 {'PASS' if g1 else 'FAIL'} | G2 n>=60 {'PASS' if g2 else 'FAIL'} | "
      f"G3 order {'PASS' if g3 else 'FAIL'} | G4 2026 order {'PASS' if g4 else 'FAIL'}")
print("  VERDICT: " + ("PROMISING AS REGIME OVERLAY" if all([g1,g2,g3,g4]) else "REJECT"))

# ── 4. report-only: refit AUC with regime one-hots + K sensitivity ───────────
tr = oos[oos.entry_time < pd.Timestamp("2026-01-01", tz="UTC")]
te = oos[oos.entry_time >= pd.Timestamp("2026-01-01", tz="UTC")]
FE = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r","rel_vol",
      "bb_width","real_vol_20","hour","dow","breadth_q","dist_hi48","green_streak"]
# reconstruct features for the book from feats_all
feats_rows = []
for _, t in book.iterrows():
    sym = t.sym
    if sym not in feats_all: feats_rows.append({c: 0 for c in FE}); continue
    f = feats_all[sym]
    i = f.index.get_loc(pd.Timestamp(t.entry_time))
    row = f.iloc[i]
    feats_rows.append({c: float(row.get(c, 0)) for c in FE})
BF = pd.DataFrame(feats_rows, index=book.index)
B = book.join(BF)
Btr, Bte = B[B.entry_time < pd.Timestamp("2026-01-01", tz="UTC")], B[B.entry_time >= pd.Timestamp("2026-01-01", tz="UTC")]
from sklearn.svm import SVC
base_auc, reg_auc = np.nan, np.nan
try:
    sc1 = StandardScaler(); m1 = SVC(C=1.0, gamma="scale", probability=True)
    m1.fit(sc1.fit_transform(Btr[FE].fillna(0).values), (Btr.rc > 0).astype(int))
    base_auc = roc_auc_score((Bte.rc > 0).astype(int), m1.predict_proba(sc1.transform(Bte[FE].fillna(0).values))[:,1])
    sc2 = StandardScaler(); m2 = SVC(C=1.0, gamma="scale", probability=True)
    X2tr = np.hstack([Btr[FE].fillna(0).values, pd.get_dummies(Btr.state).reindex(columns=range(K), fill_value=0).values])
    X2te = np.hstack([Bte[FE].fillna(0).values, pd.get_dummies(Bte.state).reindex(columns=range(K), fill_value=0).values])
    m2.fit(sc2.fit_transform(X2tr), (Btr.rc > 0).astype(int))
    reg_auc = roc_auc_score((Bte.rc > 0).astype(int), m2.predict_proba(sc2.transform(X2te))[:,1])
except Exception as e:
    print(f"  AUC add-on skipped: {e}")
print(f"  report-only | 2026 rank-AUC: base {base_auc if base_auc==base_auc else 'n/a':.3f} "
      f"vs +regime {reg_auc if reg_auc==reg_auc else 'n/a':.3f} (incumbent artifact baseline 0.46)")
for K2 in (2, 4):
    h2 = GaussianHMM(n_components=K2, covariance_type="diag", n_iter=200, random_state=126).fit(Xs[fit_mask])
    st2 = {ts: int(h2.predict(scaler.transform(F[F.index <= ts].values))[-1]) for ts in F.index[F.index >= FIT1]}
    b2 = oos.copy(); b2["st"] = [st2.get(F.index[F.index <= ts][-1], -1) for ts in b2.entry_time]
    tab = b2.groupby("st").rc.apply(pf)
    print(f"  K={K2} state PF@c (OOS): {dict(tab.round(2))}")
book.to_csv("quantlab_output/r126_book_states.csv", index=False)
print("saved: r126_book_states.csv")
