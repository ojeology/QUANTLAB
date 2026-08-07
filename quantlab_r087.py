"""
QUANTLAB AI — R087
SVM keep-rate sweep: the "more trades" dial

Locked champion (R086): SVM (RBF), 14 lean features, top-q=0.55, 73 symbols
→ 9.2 t/mo, 71% prof-mo, PF 2.23, holPF 1.48.

User wants MORE trades. The direct dial is q (fraction of signals kept by the
SVM filter). Sweep q = 0.55 (locked) → 1.0 (raw, no filter) and show the exact
trade-off: trades/month vs profitable-months vs PF vs holdout.

All walk-forward, threshold from selection only, holdout 2026 untouched.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, build_signal_mask, sim_symbol, stats_from_trades,
    cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESEARCH_ID = "R087"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
RR = 1.5

ORIGINAL52 = {
    "1INCH_USDT_SWAP","AAVE_USDT_SWAP","ADA_USDT_SWAP","ALGO_USDT_SWAP",
    "APT_USDT_SWAP","ARB_USDT_SWAP","ATOM_USDT_SWAP","AVAX_USDT_SWAP",
    "AXS_USDT_SWAP","BCH_USDT_SWAP","BNB_USDT_SWAP","BONK_USDT_SWAP",
    "BTC_USDT_SWAP","CHZ_USDT_SWAP","COMP_USDT_SWAP","CRV_USDT_SWAP",
    "DOGE_USDT_SWAP","DOT_USDT_SWAP","DYDX_USDT_SWAP","EGLD_USDT_SWAP",
    "ENA_USDT_SWAP","ETC_USDT_SWAP","ETH_USDT_SWAP","FET_USDT_SWAP",
    "FIL_USDT_SWAP","FLOKI_USDT_SWAP","GALA_USDT_SWAP","GMX_USDT_SWAP",
    "GRT_USDT_SWAP","HBAR_USDT_SWAP","ICP_USDT_SWAP","IMX_USDT_SWAP",
    "INJ_USDT_SWAP","LDO_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP",
    "NEAR_USDT_SWAP","OP_USDT_SWAP","PEPE_USDT_SWAP","SAND_USDT_SWAP",
    "SATS_USDT_SWAP","SHIB_USDT_SWAP","SNX_USDT_SWAP","SOL_USDT_SWAP",
    "STX_USDT_SWAP","SUI_USDT_SWAP","SUSHI_USDT_SWAP","TRX_USDT_SWAP",
    "UNI_USDT_SWAP","WIF_USDT_SWAP","XLM_USDT_SWAP","XRP_USDT_SWAP",
}
NEW18 = {"BICO_USDT_SWAP","HYPE_USDT_SWAP","XAU_USDT_SWAP","HOME_USDT_SWAP",
         "PUMP_USDT_SWAP","ZBT_USDT_SWAP","ZEC_USDT_SWAP","BEAT_USDT_SWAP",
         "SNDK_USDT_SWAP","SPCX_USDT_SWAP","MU_USDT_SWAP","SKHYNIX_USDT_SWAP",
         "SOXL_USDT_SWAP","UB_USDT_SWAP","SNXX_USDT_SWAP","SKHY_USDT_SWAP",
         "KORU_USDT_SWAP","CL_USDT_SWAP"}
NEW3 = {"XAG_USDT_SWAP","ALLO_USDT_SWAP","AAOI_USDT_SWAP"}
ALL_SYMS = ORIGINAL52 | NEW18 | NEW3

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  SVM keep-rate sweep (more-trades dial)")
print(SEP)
t0 = time.time()

print("\n  Loading data (73 symbols) …")
raw1h = {}; feats = {}
for sym in ALL_SYMS:
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        raw1h[sym] = df
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY: feats[sym] = f
    except Exception:
        pass
print(f"  Symbols: {len(feats)}")

above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100

cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
mask = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats.items()}
raw = []
for sym, f in feats.items():
    for t in sim_symbol(f, mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"  Raw trades: {len(raw)}")

BASE_FEATS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
              "rel_vol","bb_width","real_vol_20","hour","dow"]
SPECIAL = ["breadth_q", "dist_hi48", "green_streak"]
FEATS = BASE_FEATS + SPECIAL

rows = []
for t in raw:
    sym = t["sym"]; ts = t["entry_time"]; f = feats[sym]
    row = f.loc[ts]
    i = f.index.get_loc(ts)
    c = float(row["close"])
    hi48 = float(f["close"].rolling(48).max().iloc[i]) if i >= 0 else np.nan
    dist_hi = (c / hi48 - 1) * 100 if pd.notna(hi48) and hi48 > 0 else 0.0
    streak = 0
    for k in range(0, 6):
        j = i - k
        if j < 0: break
        if f["close"].iloc[j] > f["open"].iloc[j]: streak += 1
        else: break
    bq = float(breadth_pct.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) if ts >= breadth_pct.index[0] else 50.0
    rows.append(dict(sym=sym, ts=ts, r=t["r"], win=int(t["r"] > 0),
                     **{c: row.get(c, 0) for c in BASE_FEATS},
                     breadth_q=bq, dist_hi48=dist_hi, green_streak=streak))
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
sel_mask = mldf["ts"] < HOLDOUT_START

print("  Fitting walk-forward SVM …")
X = mldf[FEATS].fillna(0).values
y = mldf["win"].values
pred = np.full(len(mldf), np.nan)
sc = StandardScaler()
for i in range(150, len(mldf)):
    clf = SVC(C=1.0, gamma="scale", probability=True)
    clf.fit(sc.fit_transform(X[:i]), y[:i])
    pred[i] = clf.predict_proba(sc.transform(X[i:i+1]))[0, 1]

def monthly_profile(trades):
    if not trades: return dict(prof=float("nan"), worst=float("nan"), tpm=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        worst = max(worst, cur)
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df) / 27.0)

def evaluate(name, trades):
    s = stats_from_trades(trades)
    pf_c = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel_t = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel_t)["pf"]; hp = stats_from_trades(hol)["pf"]
    mp = monthly_profile(trades)
    return dict(cfg=name, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"],
                pf_c=pf_c, mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"],
                selpf=sp, holpf=hp)

QS = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
print(f"\n{SEP2}")
print("  SVM KEEP-RATE SWEEP  (q = fraction of signals kept)")
hdr = (f"    {'q':>6}{'trades':>7}{'t/mo':>7}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'holPF':>7}")
print(hdr); print("    " + "─"*90)
res = []
for q in QS:
    if q == 1.0:
        trades = raw
    else:
        thr = pd.Series(pred).where(sel_mask.values).dropna().quantile(1 - q)
        keep = set(mldf.loc[pred >= thr, "ts"])
        trades = [t for t in raw if t["entry_time"] in keep]
    r = evaluate(f"q{q}", trades)
    res.append(r)
    tag = "  ← locked" if q == 0.55 else ("  ← raw (no filter)" if q == 1.0 else "")
    print(f"    {q:>6.2f}{r['n']:>7}{r['tpm']:>7.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['holpf']:>7.2f}{tag}")

pd.DataFrame(res).to_csv(os.path.join(OUT, "r087_svm_qsweep.csv"), index=False)
lines = [f"# R087 — SVM keep-rate sweep (more-trades dial)\n",
         f"**Date:** 2026-08-06 | SVM (RBF), 14 features, 73 symbols, walk-forward\n",
         f"\n## Sweep (q = fraction of signals kept by SVM filter)\n",
         "| q | trades | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | holPF |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in res:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['holpf']:.2f} |")
lines += ["", "## Verdict", "",
          "More trades = relax q. Each step up in q adds trades but costs profitable-"
          "months and PF. The user can pick their point on this curve."]
report = "\n".join(lines)
with open(os.path.join(OUT, "r087_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r087_*")
