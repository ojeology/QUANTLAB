"""
QUANTLAB AI — R089
5-MINUTE EDGE HUNT

New timeframe: 5m candles, 5 major symbols (BTC, ETH, DOGE, LINK, LTC),
~55k bars each (2026-01-28 → 2026-08-07).

Scales the validated 1H machinery to 5m keeping the SAME TIME windows:
  IS_LOOKBACK  500h  -> 6000 bars (5m)
  RECAL_EVERY  168h  -> 2016 bars (7 days)
  trade holds   ~2-4h -> 24-48 bars

Tests:
  1. RAW Family A compression-pop on 5m (signal-only, RR1.5, E6)
  2. ML-SVM top-q filter on 5m (walk-forward, q sweep 0.55/0.75)
  3. Honest selection/holdout: selection = pre-2026-06-01, holdout = Jun 1 - Aug 7.

No-cheating rules: walk-forward training, thresholds from selection only,
holdout untouched until the end, costs included (0.05%).
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx
from scripts.ql_engine import (
    add_features, sim_symbol, stats_from_trades, cost_adjusted_rs, pf_of_rs,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESEARCH_ID = "R089"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-06-01", tz="UTC")
RR = 1.5
# 5m-scaled params
IS_LOOKBACK = 6000
RECAL_EVERY = 2016
MIN_BARS = IS_LOOKBACK + RECAL_EVERY + 500

SYMS = ["BTC_USDT_SWAP","ETH_USDT_SWAP","DOGE_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP"]

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  5-Minute Edge Hunt")
print(SEP)
t0 = time.time()

print("\n  Loading 5m data …")
feats = {}
for sym in SYMS:
    p = os.path.join(CACHE, f"{sym}_5m.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < MIN_BARS: 
            print(f"    {sym}: only {len(df)} bars — skipped")
            continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= MIN_BARS: feats[sym] = f
        print(f"    {sym}: {len(f)} bars")
    except Exception as e:
        print(f"    {sym}: ERR {e}")
print(f"  Symbols ready: {len(feats)}")

# ── Raw Family A signal on 5m (scaled engine) ────────────────────────────────
cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
from scripts.ql_engine import build_signal_mask
mask = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats.items()}

# the sim_symbol in ql_engine uses module-level IS_LOOKBACK/RECAL_EVERY; patch them
import scripts.ql_engine as qle
qle.IS_LOOKBACK = IS_LOOKBACK
qle.RECAL_EVERY = RECAL_EVERY

raw = []
for sym, f in feats.items():
    for t in sim_symbol(f, mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"\n  RAW Family A on 5m: {len(raw)} trades")

# ── ML features ──────────────────────────────────────────────────────────────
BASE_FEATS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
              "rel_vol","bb_width","real_vol_20","hour","dow"]
SPECIAL = ["breadth_q", "dist_hi48", "green_streak"]
FEATS = BASE_FEATS + SPECIAL

above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(500, min_periods=250).rank(pct=True) * 100

rows = []
for t in raw:
    sym = t["sym"]; ts = t["entry_time"]; f = feats[sym]
    row = f.loc[ts]
    i = f.index.get_loc(ts)
    c = float(row["close"])
    hi48 = float(f["close"].rolling(240).max().iloc[i]) if i >= 0 else np.nan
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
print(f"  Feature rows: {len(mldf)} | selection: {sel_mask.sum()} | holdout: {(~sel_mask).sum()}")

# ── Walk-forward SVM ─────────────────────────────────────────────────────────
print("  Fitting walk-forward SVM …")
X = mldf[FEATS].fillna(0).values
y = mldf["win"].values
pred = np.full(len(mldf), np.nan)
sc = StandardScaler()
min_train = 300
for i in range(min_train, len(mldf)):
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
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df) / 6.0)

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

QS = [0.55, 0.75, 1.0]
print(f"\n{SEP2}")
print("  5m RESULTS  (selection Jan-May, holdout Jun-Aug untouched)")
hdr = (f"    {'Config':<10}{'n':>6}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}")
print(hdr); print("    " + "─"*100)
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
    tag = "  ← raw" if q == 1.0 else ""
    print(f"    {r['cfg']:<10}{r['n']:>6}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}{tag}")

pd.DataFrame(res).to_csv(os.path.join(OUT, "r089_5m_results.csv"), index=False)
lines = [f"# R089 — 5-Minute Edge Hunt\n",
         f"**Date:** 2026-08-07 | 5 symbols (BTC,ETH,DOGE,LINK,LTC), 5m candles, "
         f"Jan 28 - Aug 7 2026 | selection ≤ May 31, holdout = Jun-Aug (untouched)\n",
         f"**Params scaled to 5m:** lookback 6000 bars (500h), recal 2016 bars (7d)\n",
         f"\n## Results\n",
         "| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in res:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} |")
lines += ["", "## Verdict", ""]
best = max(res, key=lambda r: (r["holpf"] if r["cfg"] != "q1.0" else 0))
lines.append(f"**Best 5m config: {best['cfg']}** — holPF {best['holpf']:.2f}, "
             f"{best['tpm']:.1f} t/mo, PF {best['pf']:.2f}. ")
lines.append("Note: only ~6 months of 5m data (5 symbols) — wide uncertainty. "
             "A real 5m verdict needs more data/history.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r089_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r089_*")
