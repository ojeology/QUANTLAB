#!/usr/bin/env python3
# R128 — DISSECTION BATTERY II: SILVER + OIL (+ ML addendum incl. GOLD)
# Prespec: docs/RESEARCH_FRONTIER_R128.md (LOCKED)
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, build_signal_mask, sim_symbol, IS_LOOKBACK, RECAL_EVERY
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CACHE = CONFIG["CACHE_FOLDER"]
SPLIT = pd.Timestamp("2016-01-01")
COST = 0.0010
ASSETS = {
    "SILVER": dict(yahoo="SILVER_1D.csv", okx="XAG_USDT_SWAP_1H.parquet"),
    "OIL":    dict(yahoo="OIL_1D.csv",    okx="CL_USDT_SWAP_1H.parquet"),
}

def ann(x):
    if len(x) < 20: return dict(n=0, cagr=np.nan, sharpe=np.nan, mdd=np.nan)
    yrs = len(x) / 252
    cagr = ((1+x).prod() ** (1/yrs) - 1) * 100
    sh = x.mean()/x.std()*np.sqrt(252) if x.std() > 0 else 0
    eq = (1+x).cumprod(); dd = (eq/eq.cummax()-1).min()*100
    return dict(n=len(x), cagr=cagr, sharpe=sh, mdd=dd, total=((1+x).prod()-1)*100)

def bootp(x, n=5000, seed=128):
    x = np.asarray(x, float); months = len(x)//21
    if months <= 2: return float("nan")
    B = np.array([x[i*21:(i+1)*21] for i in range(months)])
    rng = np.random.default_rng(seed)
    return float((B[rng.integers(0, months, size=(n, months))].sum(axis=(1,2)) > 0).mean())

def dissect(name, yfile, okxfile):
    g = pd.read_csv(f"{CACHE}/{yfile}", parse_dates=["date"]).set_index("date")["close"]
    r = g.pct_change().dropna()
    sel, conf = r[r.index < SPLIT], r[r.index >= SPLIT]
    a_s, a_c, y26 = ann(sel), ann(conf), ann(r[r.index >= "2026-01-01"])
    print("="*100)
    print(f"  {name} — ANATOMY")
    print("="*100)
    print(f"  selection 2000→2015: CAGR {a_s['cagr']:+.1f}% Sharpe {a_s['sharpe']:+.2f} maxDD {a_s['mdd']:+.1f}%")
    print(f"  confirm    2016→2026: CAGR {a_c['cagr']:+.1f}% Sharpe {a_c['sharpe']:+.2f} maxDD {a_c['mdd']:+.1f}%")
    print(f"  2026 YTD: {y26['total']:+.1f}%  | $300 translation: {a_c['cagr']/100*300/12:.2f}/mo at confirm-era drift")
    try:
        btc = pd.read_parquet(f"{CACHE}/BTC_USDT_SWAP_1H.parquet")
        btc.index = pd.to_datetime(btc.index, utc=True)
        b = btc["close"].resample("1D").last().dropna(); b.index = b.index.tz_localize(None).normalize()
        j = pd.concat([g.rename("a"), b.rename("b")], axis=1).dropna().pct_change().dropna()
        print(f"  corr with BTC (daily, {j.index.min().date()}→): {j.a.corr(j.b):+.2f}")
    except Exception:
        pass

    print(f"\n  {name} — TREND GRID (registered; 0.10%/unit change)")
    survivors = []
    for N in (20, 60, 120, 250):
        for shape in ("long/flat", "long/short"):
            sig = np.sign(g.pct_change(N)).shift(1)
            pos = (sig if shape == "long/short" else sig.clip(lower=0)).fillna(0)
            strat = pos * r - pos.diff().abs().fillna(0) * COST
            s1, s2 = strat[strat.index < SPLIT].dropna(), strat[strat.index >= SPLIT].dropna()
            A1, A2 = ann(s1), ann(s2)
            bp = bootp(s1.values)
            ok = A1["sharpe"] >= 0.5 and bp >= 0.95 and A2["sharpe"] > 0
            dom = ""
            if ok:
                dom = ("  [DOMINATED by buy-and-hold]" if A2["cagr"] < a_c["cagr"] and A2["sharpe"] < a_c["sharpe"]
                       else "  [beats buy-and-hold]")
                survivors.append((N, shape, dom))
            print(f"    N{N:<4}{shape:<12} sel Sh {A1['sharpe']:+.2f} bootP {bp:.3f} | conf Sh {A2['sharpe']:+.2f} CAGR {A2['cagr']:+.1f}%  {'PROMISING' if ok else 'dead'}{dom}")

    print(f"\n  {name} — COMPRESSION-POP on OKX 1H (registered)")
    try:
        df = pd.read_parquet(f"{CACHE}/{okxfile}")
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width",
                         "prev_range_r","prev_body_r"], inplace=True)
        mask = build_signal_mask(f, ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"], "green", 1.5)
        tr = list(sim_symbol(f, mask, 1.5, dict(entry_next=False, exit="base", hours=None)))
        rs = np.array([t["r"] - 2.0*(0.07/100.0)*t["entry"]/t["atr"] for t in tr]) if tr else np.array([])
        if len(rs):
            w = rs[rs>0].sum(); l = abs(rs[rs<0].sum()); pf = w/l if l>0 else 999
            rng = np.random.default_rng(128)
            bp = float(np.mean([rng.choice(rs, len(rs)).sum() > 0 for _ in range(5000)]))
            ok = len(rs) >= 30 and pf >= 1.3 and bp >= 0.95
            print(f"    n={len(rs)}  PF@c={pf:.2f}  WR={(rs>0).mean()*100:.0f}%  bootP={bp:.3f}  -> {'PROMISING-CANDIDATE' if ok else 'DEAD'}")
        else:
            print("    n=0 — pattern does not fire on this substrate")
    except Exception as e:
        print(f"    skipped: {type(e).__name__}: {str(e)[:60]}")

    # events + sessions (mapping)
    sig = r.rolling(252).std()
    big = r[(r.index < SPLIT) & (r.abs() > 2*sig)]
    for h in (5, 20):
        fwd = (g.shift(-h)/g - 1).reindex(big.index).mean()*100
        unc = (g.shift(-h)/g - 1).reindex(sel.index).mean()*100
        print(f"    after >2σ day fwd {h}d: {fwd:+.2f}% vs unc {unc:+.2f}% (n={len(big)})")
    try:
        hh = df["close"].pct_change().dropna()
        vshare = hh.groupby(hh.index.hour).apply(lambda s: np.sqrt((s**2).mean()))
        vshare = vshare/vshare.sum()*100
        blk = vshare[[h for h in range(7,17) if h in vshare.index]].sum()
        top2 = vshare.sort_values(ascending=False).head(2)
        print(f"    session: London+NY block {blk:.0f}% of vol | top hours " +
              " ".join(f"{h}h:{v:.1f}%" for h, v in top2.items()))
    except Exception:
        pass
    return survivors

def ml_addendum():
    print("\n" + "="*100)
    print("  ML ADDENDUM (diagnostic): can supervised models beat drift? walk-forward, yearly refits")
    print("="*100)
    for name, yfile in [("GOLD","GC_F_1D.csv"), ("SILVER","SILVER_1D.csv"), ("OIL","OIL_1D.csv")]:
        g = pd.read_csv(f"{CACHE}/{yfile}", parse_dates=["date"]).set_index("date")["close"]
        r = g.pct_change()
        F = pd.DataFrame({
            "lag1": r.shift(1), "lag5": r.shift(5), "lag10": r.shift(10), "lag20": r.shift(20),
            "mom60": g.pct_change(60), "vol20": r.rolling(20).std(), "vol60": r.rolling(60).std(),
            "dow": r.index.dayofweek,
        }).dropna()
        fwd20 = (g.shift(-20)/g - 1).reindex(F.index)
        y = (fwd20 > 0).astype(int)
        # non-overlapping steps: evaluate every 20th day from 2010
        idx_all = F.index[(F.index >= "2010-01-01") & (F.index <= "2026-09-01")]
        eval_idx = idx_all[::20]
        ys, aucs_lr, aucs_gb, strat, bh = [], [], [], [], []
        for t in eval_idx:
            tr = F[(F.index >= "2000-01-01") & (F.index < t - pd.Timedelta(days=40))]
            if len(tr) < 500: continue
            Xtr, ytr = tr.values, y.reindex(tr.index).values
            xe = F.loc[[t]]; ye = y.loc[t]
            if np.isnan(ye): continue
            sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); xe_s = sc.transform(xe.values)
            m1 = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr_s, ytr)
            p1 = m1.predict_proba(xe_s)[0,1]
            m2 = GradientBoostingClassifier(n_estimators=150, max_depth=2).fit(Xtr_s, ytr)
            p2 = m2.predict_proba(xe_s)[0,1]
            ys.append(int(ye)); aucs_lr.append(p1); aucs_gb.append(p2)
            pos = 1.0 if p1 >= 0.55 else 0.0
            fwd = (g.shift(-20)/g - 1)
            # realized fwd 20d return from t (approx by next eval step)
            nxt_i = list(eval_idx).index(t)
            if nxt_i+1 < len(eval_idx):
                realized = g.iloc[g.index.get_indexer([eval_idx[nxt_i+1]], method="nearest")[0]] / g.iloc[g.index.get_indexer([t], method="nearest")[0]] - 1
                strat.append(pos * realized - abs(pos-1.0)*COST)
                bh.append(realized)
        auc_lr = roc_auc_score(ys, aucs_lr); auc_gb = roc_auc_score(ys, aucs_gb)
        s, b = np.array(strat), np.array(bh)
        print(f"  {name:<7} OOS n={len(ys):>3}  AUC logreg={auc_lr:.3f}  AUC gbm={auc_gb:.3f}  "
              f"| ML-gated CAGR {((1+s).prod()**(252/len(s)/20)-1)*100:+.1f}% vs B&H {((1+b).prod()**(252/len(s)/20)-1)*100:+.1f}%")
    print("  (diagnostic only — R124's model-class law applies; this measures per-asset information)")

print("="*100)
print("  R128 — DISSECTION BATTERY II")
print("="*100)
all_surv = {}
for name, cfg in ASSETS.items():
    all_surv[name] = dissect(name, cfg["yahoo"], cfg["okx"])
ml_addendum()
print("\n" + "="*100)
for name, sv in all_surv.items():
    print(f"  {name}: surviving trend cells: {[(n, s) for n, s, d in sv] if sv else 'NONE'}")
print("="*100)
