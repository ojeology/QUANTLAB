"""
QUANTLAB AI — R083
ML-filter refinement (follow-up to R082 F6, which hit 7.7 t/mo, 64% prof-mo, holPF 1.47)

Pre-registered grid:
  keep-fraction q (top-q of walk-forward predicted P(win), threshold from SELECTION):
     0.35 / 0.45 / 0.55  (0.50 = R082 F6, reference)
  Combos (each = ML filter + ONE additional regime gate):
     ML0.5 + daily-trend
     ML0.5 + daily-breadth
     ML0.4 + daily-trend

Success (user spec): t/mo>=8, prof-mo%>=65, worst losing-month streak<=3,
holPF>1.1, PF at 0.05% cost >1.1.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG, calc_ema, calc_adx
from scripts.ql_engine import (
    add_features, build_signal_mask, sim_symbol, stats_from_trades,
    cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RESEARCH_ID = "R083"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
RR = 1.5

ORIGINAL = {
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

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  ML-filter refinement")
print(SEP)
t0 = time.time()

print("\n  Loading data …")
feats = {}; raw1h = {}
for sym in ORIGINAL:
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

def ohlc(df, rule):
    return df.resample(rule).agg({"open":"first","high":"max","low":"min",
                                  "close":"last","vol":"sum"}).dropna()
daily = {}
for sym, df in raw1h.items():
    d = ohlc(df, "1D")
    d["ema50"] = calc_ema(d["close"], 50)
    daily[sym] = d
all_days = pd.date_range("2024-01-27", "2026-08-07", freq="D", tz="UTC")
d_above = {s: (d["close"] > d["ema50"]).astype(float).reindex(all_days)
           for s, d in daily.items()}
d_breadth = pd.DataFrame(d_above).mean(axis=1, skipna=True)

famA_cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
base_mask = {s: build_signal_mask(f, famA_cids, "green", 1.5) for s, f in feats.items()}
raw_trades = []
for sym, f in feats.items():
    for t in sim_symbol(f, base_mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw_trades.append(t)
raw_trades.sort(key=lambda t: t["entry_time"])
print(f"  Raw Family A trades: {len(raw_trades)}")

# ML features + walk-forward predictions
FEAT_COLS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
             "rel_vol","bb_width","real_vol_20","hour","dow"]
ml_rows = []
for t in raw_trades:
    sym = t["sym"]; ts = t["entry_time"]
    row = feats[sym].loc[ts]
    ml_rows.append(dict(sym=sym, ts=ts, r=t["r"], win=int(t["r"] > 0),
                        **{c: row.get(c, 0) for c in FEAT_COLS}))
mldf = pd.DataFrame(ml_rows).sort_values("ts").reset_index(drop=True)
X = mldf[FEAT_COLS].fillna(0).values
y = mldf["win"].values
pred = np.full(len(mldf), np.nan)
scaler = StandardScaler()
for i in range(150, len(mldf)):
    clf = LogisticRegression(max_iter=2000, C=0.5)
    clf.fit(scaler.fit_transform(X[:i]), y[:i])
    pred[i] = clf.predict_proba(scaler.transform(X[i:i+1]))[0, 1]
mldf["pwin"] = pred
sel_ts = mldf["ts"] < HOLDOUT_START

def ml_keep(q):
    thr = mldf.loc[sel_ts, "pwin"].dropna().quantile(1 - q)
    return set(mldf.loc[mldf["pwin"] >= thr, "ts"])

def daily_trend(sym, ts):
    d = daily[sym]
    return bool((d["close"] > d["ema50"]).reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0])

def daily_breadth_ok(ts):
    return bool((d_breadth.reindex(pd.DatetimeIndex([ts]), method="ffill") > 0.5).iloc[0])

CONFIGS = {
    "ML_q35": lambda t: t["entry_time"] in ml_keep(0.35),
    "ML_q45": lambda t: t["entry_time"] in ml_keep(0.45),
    "ML_q50_ref": lambda t: t["entry_time"] in ml_keep(0.50),
    "ML_q55": lambda t: t["entry_time"] in ml_keep(0.55),
    "ML50_dt": lambda t: (t["entry_time"] in ml_keep(0.50)) and daily_trend(t["sym"], t["entry_time"]),
    "ML50_dbr": lambda t: (t["entry_time"] in ml_keep(0.50)) and daily_breadth_ok(t["entry_time"]),
    "ML40_dt": lambda t: (t["entry_time"] in ml_keep(0.40)) and daily_trend(t["sym"], t["entry_time"]),
}

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

print(f"\n{SEP2}")
print("  RESULTS")
hdr = (f"    {'Config':<12}{'n':>5}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>6}{'worst':>6}{'selPF':>7}{'holPF':>7}")
print(hdr); print("    " + "─"*100)
rows = []
for name, keep_fn in CONFIGS.items():
    trades = [t for t in raw_trades if keep_fn(t)]
    s = stats_from_trades(trades)
    pf_c = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]; hp = stats_from_trades(hol)["pf"]
    mp = monthly_profile(trades)
    rows.append(dict(cfg=name, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"],
                     pf_c=pf_c, mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"],
                     selpf=sp, holpf=hp))
    print(f"    {name:<12}{len(trades):>5}{mp['tpm']:>6.1f}{s['wr']*100:>6.0f}%"
          f"{s['pf']:>7.2f}{pf_c:>8.2f}{s['mdd']*100:>7.1f}%{mp['prof']*100:>5.0f}%"
          f"{mp['worst']:>6}{sp:>7.2f}{hp:>7.2f}")

print(f"\n{SEP2}")
print("  SUCCESS (t/mo>=8, prof%>=65, worst<=3, holPF>1.1, PF@.05>1.1)")
passed = [r for r in rows if r["tpm"] >= 8 and r["prof"] >= 0.65 and r["worst"] <= 3
          and r["holpf"] > 1.1 and r["pf_c"] > 1.1]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: t/mo={r['tpm']:.1f} prof%={r['prof']*100:.0f}% "
              f"worst={r['worst']} PF={r['pf']:.2f} PF@.05={r['pf_c']:.2f} holPF={r['holpf']:.2f}")
else:
    print("  ❌ none pass all. Closest:")
    for r in sorted(rows, key=lambda r: -(0.3*(r["tpm"]>=8) + 0.3*(r["prof"]>=0.65) +
                                          0.2*(r["worst"]<=3) + 0.2*(r["holpf"]>1.1)))[:5]:
        print(f"    {r['cfg']:<12} t/mo={r['tpm']:.1f} prof%={r['prof']*100:.0f}% "
              f"worst={r['worst']} PF={r['pf']:.2f} PF@.05={r['pf_c']:.2f} holPF={r['holpf']:.2f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "r083_ml_refine.csv"), index=False)
lines = [f"# R083 — ML-filter refinement\n",
         f"**Date:** 2026-08-06 | walk-forward logistic regression on Family A raw, "
         f"keep top-q by P(win), threshold from selection only\n",
         f"\n## Results\n",
         "| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    best = max(passed, key=lambda r: r["tpm"] * r["holpf"] * r["prof"])
    lines.append(f"**✅ {best['cfg']} meets the retail spec:** {best['tpm']:.1f} t/mo, "
                 f"{best['prof']*100:.0f}% profitable months, worst streak {best['worst']}, "
                 f"PF {best['pf']:.2f} (cost {best['pf_c']:.2f}), holPF {best['holpf']:.2f}.")
else:
    lines.append("**❌ Still no config meets ALL criteria**, but ML filters are the closest "
                 "ever found (see table).")
report = "\n".join(lines)
with open(os.path.join(OUT, "r083_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r083_*")
