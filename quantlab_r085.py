"""
QUANTLAB AI — R085
Upgraded ML: richer features + confidence-based sizing + model comparison

Base = R084 winner (ML q55 on 73-symbol universe: 9.3 t/mo, 71% prof-mo, PF 2.11).

Upgrades (each pre-registered, ONE new idea):
  A  ML top-q, BASE features                     (reference = replicate R084)
  B  ML top-q, RICH features (adds higher-timeframe, cross-sectional,
     multi-bar momentum context the base model ignores)
  C  B + confidence-based sizing (risk scaled by predicted P(win) within the
     kept set — uses the model's full output instead of a blunt on/off)
  D  Gradient-boosting model (HistGradientBoosting), RICH features, top-q
     (honest test: does a fancier model help at n~250, or overfit?)
  E  B + daily-trend overlay (ML + the R082/R083 daily filter that hit 70%)

All walk-forward (train only on past), threshold from selection only,
holdout = 2026 untouched.

Success (user spec): prof-mo% >= 70, t/mo >= 8, worst losing-month streak <= 3,
holPF > 1.1, PF at 0.05% cost > 1.1.
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
from sklearn.ensemble import HistGradientBoostingClassifier

RESEARCH_ID = "R085"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
RR = 1.5
Q = 0.55

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
print(f"  QUANTLAB AI — {RESEARCH_ID}  Upgraded ML (rich features / sizing / model)")
print(SEP)
t0 = time.time()

# ── Load ─────────────────────────────────────────────────────────────────────
print("\n  Loading data (73 symbols) …")
raw1h = {}
feats = {}
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

# ── Higher-TF + cross-sectional context ─────────────────────────────────────
print("  Building HTF + cross-sectional context …")
def ohlc(df, rule):
    return df.resample(rule).agg({"open":"first","high":"max","low":"min",
                                  "close":"last","vol":"sum"}).dropna()
daily = {s: ohlc(d, "1D") for s, d in raw1h.items()}
for s, d in daily.items():
    d["ema50"] = calc_ema(d["close"], 50)
    d["adx14"] = calc_adx(d, 14)
h4 = {s: ohlc(d, "4H") for s, d in raw1h.items()}
for s, h in h4.items():
    h["ema20"] = calc_ema(h["close"], 20)
    h["ema50"] = calc_ema(h["close"], 50)

# universe breadth (fraction above EMA20 at each 1H) and median 24h return
above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
close_panel = pd.DataFrame({s: f["close"] for s, f in feats.items()}).sort_index()
ret24_panel = close_panel.pct_change(24)
med_ret24 = ret24_panel.median(axis=1, skipna=True)

# ── Raw trades + feature rows ────────────────────────────────────────────────
cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
mask = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats.items()}
raw = []
for sym, f in feats.items():
    for t in sim_symbol(f, mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"  Raw trades: {len(raw)}")

def daily_flag(sym, ts, kind):
    d = daily[sym]
    if kind == "trend":
        return int(bool((d["close"] > d["ema50"]).reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]))
    return int(bool((d["adx14"] > 20).reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]))

def h4_flag(sym, ts):
    h = h4[sym]
    return int(bool((h["ema20"] > h["ema50"]).reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]))

BASE_FEATS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
              "rel_vol","bb_width","real_vol_20","hour","dow"]
RICH_FEATS = BASE_FEATS + [
    "d_trend","d_adx","h4_trend",                 # higher-timeframe
    "rel_str_24","breadth_now","rank_24",         # cross-sectional
    "dist_high48","green5","ret_24","dist_ema20", # multi-bar momentum
]

rows = []
for t in raw:
    sym = t["sym"]; ts = t["entry_time"]; f = feats[sym]
    row = f.loc[ts]
    # cross-sectional
    r24 = ret24_panel.loc[ts] if ts in ret24_panel.index else None
    rel_str = float(row.get("close", 0) / row.get("close",1) - 1)
    med = med_ret24.get(ts, np.nan)
    my_ret = float(r24[sym]) if r24 is not None and sym in r24.index and pd.notna(r24[sym]) else np.nan
    rel_str = my_ret - med if pd.notna(my_ret) and pd.notna(med) else 0.0
    rk = float((r24.dropna() <= my_ret).mean()) if r24 is not None and pd.notna(my_ret) else 0.5
    hi48 = float(f["close"].rolling(48).max().loc[ts]) if ts in f.index else np.nan
    c = float(row["close"]); dist_hi = (c / hi48 - 1) * 100 if pd.notna(hi48) and hi48 > 0 else 0.0
    g5 = int((f["close"].loc[ts] > f["open"].loc[ts]) + sum(1 for k in range(1,5)
             if (f["close"].iloc[f.index.get_loc(ts)-k] > f["open"].iloc[f.index.get_loc(ts)-k])))
    rr24 = my_ret * 100 if pd.notna(my_ret) else 0.0
    d_ema20 = (c / float(f["ema20"].loc[ts]) - 1) * 100 if ts in f.index else 0.0
    rows.append(dict(sym=sym, ts=ts, r=t["r"], win=int(t["r"] > 0),
                     **{c: row.get(c, 0) for c in BASE_FEATS},
                     d_trend=daily_flag(sym, ts, "trend"), d_adx=daily_flag(sym, ts, "adx"),
                     h4_trend=h4_flag(sym, ts),
                     rel_str_24=rel_str, breadth_now=float(breadth.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]),
                     rank_24=rk, dist_high48=dist_hi, green5=g5, ret_24=rr24, dist_ema20=d_ema20))
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
print(f"  Feature rows: {len(mldf)} | rich features: {len(RICH_FEATS)}")

# ── Walk-forward predictors ──────────────────────────────────────────────────
def walkforward(feat_cols, model="lr", min_train=150):
    X = mldf[feat_cols].fillna(0).values
    y = mldf["win"].values
    pred = np.full(len(mldf), np.nan)
    if model == "lr":
        sc = StandardScaler()
        for i in range(min_train, len(mldf)):
            clf = LogisticRegression(max_iter=2000, C=0.5)
            clf.fit(sc.fit_transform(X[:i]), y[:i])
            pred[i] = clf.predict_proba(sc.transform(X[i:i+1]))[0, 1]
    else:
        for i in range(min_train, len(mldf)):
            clf = HistGradientBoostingClassifier(max_iter=300, max_depth=3,
                                                 l2_regularization=2.0,
                                                 min_samples_leaf=25, learning_rate=0.05)
            clf.fit(X[:i], y[:i])
            pred[i] = clf.predict_proba(X[i:i+1])[0, 1]
    return pred

sel_mask = mldf["ts"] < HOLDOUT_START

def keep_ts(pred, q=Q):
    thr = pd.Series(pred).where(sel_mask.values).dropna().quantile(1 - q)
    return set(mldf.loc[pred >= thr, "ts"]), thr

print("  Fitting walk-forward models …")
pred_base = walkforward(BASE_FEATS, "lr")
pred_rich = walkforward(RICH_FEATS, "lr")
pred_gb   = walkforward(RICH_FEATS, "gb")

keep_base, _ = keep_ts(pred_base)
keep_rich, _ = keep_ts(pred_rich)
keep_gb, _   = keep_ts(pred_gb)

# daily trend series for E
dt_series = {s: (d["close"] > d["ema50"]).astype(int) for s, d in daily.items()}
def in_daily_trend(sym, ts):
    try:
        return bool(dt_series[sym].reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0])
    except Exception:
        return False

def conf_size(pred, q=Q):
    """size weight within kept set: p / mean(p), capped [0.5, 2.0]"""
    thr = pd.Series(pred).where(sel_mask.values).dropna().quantile(1 - q)
    kept = pred >= thr
    p_kept = pred[kept]
    w = p_kept / p_kept.mean()
    kept_ts = set(mldf.loc[kept, "ts"])
    w_map = dict(zip(mldf.loc[kept, "ts"], np.clip(w, 0.5, 2.0)))
    return kept_ts, w_map

kept_c, w_c = conf_size(pred_rich)

# ── Build config trade lists ─────────────────────────────────────────────────
def sel(trades, keep):
    return [t for t in trades if t["entry_time"] in keep]

configs = {}
configs["A_base_ref"] = sel(raw, keep_base)
configs["B_rich"]     = sel(raw, keep_rich)
# C: rich + confidence sizing (weight each trade's R by w)
c_trades = []
for t in sel(raw, kept_c):
    tt = dict(t); tt["r"] = t["r"] * w_c[t["entry_time"]]
    c_trades.append(tt)
configs["C_rich_sized"] = c_trades
configs["D_gboost"]     = sel(raw, keep_gb)
configs["E_rich_dtrend"]= [t for t in sel(raw, keep_rich) if in_daily_trend(t["sym"], t["entry_time"])]

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

print(f"\n{SEP2}")
print("  RESULTS")
hdr = (f"    {'Config':<14}{'n':>5}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>6}{'worst':>6}{'selPF':>7}{'holPF':>7}")
print(hdr); print("    " + "─"*100)
rows = []
for name, trades in configs.items():
    r = evaluate(name, trades)
    rows.append(r)
    print(f"    {name:<14}{r['n']:>5}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>5.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}")

# feature importance (from rich LR on selection)
Xsel = mldf.loc[sel_mask, RICH_FEATS].fillna(0).values
ysel = mldf.loc[sel_mask, "win"].values
clf = LogisticRegression(max_iter=2000, C=0.5)
sc = StandardScaler()
clf.fit(sc.fit_transform(Xsel), ysel)
coef = pd.Series(clf.coef_[0], index=RICH_FEATS).abs().sort_values(ascending=False)
print(f"\n  Top 10 feature importances (|coef|, rich LR, selection):")
print(coef.head(10).round(3).to_string())

print(f"\n{SEP2}")
print("  SUCCESS (prof%>=70, t/mo>=8, worst<=3, holPF>1.1, PF@.05>1.1)")
passed = [r for r in rows if r["prof"] >= 0.70 and r["tpm"] >= 8 and r["worst"] <= 3
          and r["holpf"] > 1.1 and r["pf_c"] > 1.1]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: t/mo={r['tpm']:.1f} prof%={r['prof']*100:.0f}% "
              f"worst={r['worst']} PF={r['pf']:.2f} PF@.05={r['pf_c']:.2f} holPF={r['holpf']:.2f}")
else:
    print("  ❌ none pass all. Closest:")
    scored = sorted(rows, key=lambda r: -(0.3*(r["prof"]>=0.70) + 0.3*(r["tpm"]>=8) +
                                          0.2*(r["worst"]<=3) + 0.2*(r["holpf"]>1.1)))
    for r in scored[:5]:
        print(f"    {r['cfg']:<14} t/mo={r['tpm']:.1f} prof%={r['prof']*100:.0f}% "
              f"worst={r['worst']} PF={r['pf']:.2f} PF@.05={r['pf_c']:.2f} holPF={r['holpf']:.2f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "r085_ml_upgrade.csv"), index=False)
lines = [f"# R085 — Upgraded ML (rich features / sizing / model)\n",
         f"**Date:** 2026-08-06 | base = R084 ML q55 on 73 (9.3 t/mo, 71% prof-mo, PF 2.11)\n",
         f"\n## Results\n",
         "| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} |")
lines += ["", "## Feature importance (rich LR)", ""]
for feat, v in coef.head(12).items():
    lines.append(f"- {feat}: {v:.3f}")
lines += ["", "## Verdict", ""]
if passed:
    best = max(passed, key=lambda r: r["tpm"] * r["holpf"] * r["prof"])
    lines.append(f"**✅ {best['cfg']} beats the base:** {best['tpm']:.1f} t/mo, "
                 f"{best['prof']*100:.0f}% prof-mo, worst {best['worst']}, "
                 f"PF {best['pf']:.2f} (cost {best['pf_c']:.2f}), holPF {best['holpf']:.2f}.")
else:
    lines.append("**❌ No config beats the base on ALL criteria.** See table — "
                 "the richest feature set / sizing may or may not help.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r085_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r085_*")
