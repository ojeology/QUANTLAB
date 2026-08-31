"""
QUANTLAB AI — R082
New Dimensions: Multi-Timeframe (Daily/4H) Filters + ML Entry Filter + BE-Trail Exit

Base = Family A RAW (E6, RR1.5, no filters) = 14.9 t/mo, PF 1.48, MDD -31%, prof-mo 48%.
We want: keep frequency high, push profitable-months toward 65-70%, tame drawdown.

Pre-registered filters/exits (each = ONE new lever on the RAW base):
  F1 daily-trend   : trade only when daily close > daily EMA50
  F2 4H-trend      : trade only when 4H EMA20 > 4H EMA50
  F3 daily-ADX     : trade only when daily ADX14 > 20 (daily trend exists)
  F4 daily-breadth : trade only when >50% of 52 symbols are above daily EMA50
  F5 dailytrend + 4Htrend combined
  F6 ML filter     : walk-forward logistic regression, keep top-half by predicted P(win)
  F7 BE-trail exit : after +1R move SL to BE, then 1.5ATR trail, no TP cap, 48-bar stop
  F8 dailytrend + BE-trail exit

Success (user spec): t/mo>=8, prof-mo%>=65, worst losing-month streak<=3, holPF>1.1,
PF at 0.05% cost >1.1.
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

RESEARCH_ID = "R082"
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
print(f"  QUANTLAB AI — {RESEARCH_ID}  Multi-TF / ML / Exit new levers")
print(SEP)
t0 = time.time()

print("\n  Loading 1H data …")
feats = {}
raw1h = {}
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
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f
    except Exception:
        pass
print(f"  Symbols: {len(feats)}")

# ── Higher-timeframe series (resample 1H -> 4H and 1D) ──────────────────────
def ohlc(df, rule):
    return df.resample(rule).agg({"open":"first","high":"max","low":"min",
                                  "close":"last","vol":"sum"}).dropna()

print("  Building daily/4H series …")
daily = {}; h4 = {}
for sym, df in raw1h.items():
    d = ohlc(df, "1D")
    d["ema50"] = calc_ema(d["close"], 50)
    d["adx14"] = calc_adx(d, 14)
    daily[sym] = d
    h = ohlc(df, "4H")
    h["ema20"] = calc_ema(h["close"], 20)
    h["ema50"] = calc_ema(h["close"], 50)
    h4[sym] = h

# daily breadth (fraction above daily EMA50) — built on a FIXED common day range
# (avoids pandas tz-aware index-union artifact)
all_days = pd.date_range("2024-01-27", "2026-08-07", freq="D", tz="UTC")
d_above = {}
for s, d in daily.items():
    d_above[s] = (d["close"] > d["ema50"]).astype(float).reindex(all_days)
d_breadth = pd.DataFrame(d_above).mean(axis=1, skipna=True)

def daily_regime(sym, idx, kind):
    d = daily[sym]
    if kind == "trend":
        return (d["close"] > d["ema50"]).reindex(idx, method="ffill").fillna(False)
    if kind == "adx":
        return (d["adx14"] > 20).reindex(idx, method="ffill").fillna(False)

def h4_regime(sym, idx):
    h = h4[sym]
    return (h["ema20"] > h["ema50"]).reindex(idx, method="ffill").fillna(False)

def d_breadth_regime(idx):
    return (d_breadth.reindex(idx, method="ffill") > 0.5).fillna(False)

# ── Base RAW Family A mask + trades with full feature capture ───────────────
famA_cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
base_mask = {s: build_signal_mask(f, famA_cids, "green", 1.5) for s, f in feats.items()}

def run_raw(exit_mode="base"):
    cfg = dict(entry_next=False, exit=exit_mode, hours=None)
    if exit_mode == "be_trail":
        cfg = dict(entry_next=False, exit="trail_run", sl_mult=1.0,
                   trail_mult=1.5, time_bars=48)
    out = []
    for sym, f in feats.items():
        try:
            for t in sim_symbol(f, base_mask[sym], RR, cfg):
                t["sym"] = sym
                out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t["entry_time"])
    return out

raw_trades = run_raw("base")

# ── Apply filters to trade list (post-hoc on entry time is equivalent: filter is
#    a regime series; entry happens at signal bar close; regime at that timestamp) ──
def apply_filter(trades, reg_fn):
    out = []
    for t in trades:
        sym = t["sym"]; ts = t["entry_time"]
        if reg_fn(sym, ts):
            out.append(t)
    return out

reg_fns = {
    "F1_dailytrend": lambda sym, ts: daily_regime(sym, pd.DatetimeIndex([ts]), "trend").iloc[0],
    "F2_4htrend":    lambda sym, ts: h4_regime(sym, pd.DatetimeIndex([ts])).iloc[0],
    "F3_dailyadx":   lambda sym, ts: daily_regime(sym, pd.DatetimeIndex([ts]), "adx").iloc[0],
    "F4_dailybr":    lambda sym, ts: d_breadth_regime(pd.DatetimeIndex([ts])).iloc[0],
}
def F5(sym, ts):
    return reg_fns["F1_dailytrend"](sym, ts) and reg_fns["F2_4htrend"](sym, ts)

# ── ML filter (walk-forward logistic regression) ─────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

print("\n  Building ML filter (walk-forward logistic regression) …")
# features at signal time
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

# walk-forward: expanding window, min 150 trains, predict rest
pred = np.full(len(mldf), np.nan)
scaler = StandardScaler()
for i in range(150, len(mldf)):
    Xtr = X[:i]; ytr = y[:i]
    Xs = scaler.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=2000, C=0.5)
    clf.fit(Xs, ytr)
    Xn = scaler.transform(X[i:i+1])
    pred[i] = clf.predict_proba(Xn)[0, 1]
mldf["pwin"] = pred
sel_mask = mldf["ts"] < HOLDOUT_START
thr = mldf.loc[sel_mask, "pwin"].dropna().median()   # keep top-half by P(win), tuned on selection only
ml_keep_ts = set(mldf.loc[(mldf["pwin"] >= thr), "ts"])
def F6(sym, ts):
    return ts in ml_keep_ts

# ── Evaluate all configs ─────────────────────────────────────────────────────
configs = {
    "RAW_base":      raw_trades,
    "F1_dailytrend": apply_filter(raw_trades, reg_fns["F1_dailytrend"]),
    "F2_4htrend":    apply_filter(raw_trades, reg_fns["F2_4htrend"]),
    "F3_dailyadx":   apply_filter(raw_trades, reg_fns["F3_dailyadx"]),
    "F4_dailybr":    apply_filter(raw_trades, reg_fns["F4_dailybr"]),
    "F5_dt_4h":      apply_filter(raw_trades, F5),
    "F6_ml":         [t for t in raw_trades if F6(t["sym"], t["entry_time"])],
    "F7_betrail":    run_raw("be_trail"),
    "F8_dt_betrail": apply_filter(run_raw("be_trail"), reg_fns["F1_dailytrend"]),
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
print("  RESULTS (base = RAW Family A, E6, RR1.5)")
hdr = (f"    {'Config':<14}{'n':>5}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>6}{'worst':>6}{'selPF':>7}{'holPF':>7}")
print(hdr); print("    " + "─"*100)
rows = []
for name, trades in configs.items():
    s = stats_from_trades(trades)
    rs = np.array([t["r"] for t in trades])
    pf_c = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]
    hp = stats_from_trades(hol)["pf"]
    mp = monthly_profile(trades)
    rows.append(dict(cfg=name, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"],
                     pf_c=pf_c, mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"],
                     selpf=sp, holpf=hp))
    print(f"    {name:<14}{len(trades):>5}{mp['tpm']:>6.1f}{s['wr']*100:>6.0f}%"
          f"{s['pf']:>7.2f}{pf_c:>8.2f}{s['mdd']*100:>7.1f}%{mp['prof']*100:>5.0f}%"
          f"{mp['worst']:>6}{sp:>7.2f}{hp:>7.2f}")

# ── Success ──────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("  SUCCESS (t/mo>=8, prof%>=65, worst<=3, holPF>1.1, PF@.05>1.1)")
passed = [r for r in rows if r["tpm"] >= 8 and r["prof"] >= 0.65 and r["worst"] <= 3
          and r["holpf"] > 1.1 and r["pf_c"] > 1.1]
if passed:
    print("  ✅ PASS:")
    for r in passed:
        print(f"    {r['cfg']:<14} t/mo={r['tpm']:.1f} prof%={r['prof']*100:.0f}% "
              f"worst={r['worst']} PF={r['pf']:.2f} PF@.05={r['pf_c']:.2f} holPF={r['holpf']:.2f}")
else:
    print("  ❌ none pass all. Closest by weighted score:")
    scored = sorted(rows, key=lambda r: -(0.3*(r["tpm"]>=8) + 0.3*(r["prof"]>=0.65) +
                                          0.2*(r["worst"]<=3) + 0.2*(r["holpf"]>1.1)))
    for r in scored[:5]:
        print(f"    {r['cfg']:<14} t/mo={r['tpm']:.1f} prof%={r['prof']*100:.0f}% "
              f"worst={r['worst']} PF={r['pf']:.2f} PF@.05={r['pf_c']:.2f} holPF={r['holpf']:.2f}")

# ── Outputs ──────────────────────────────────────────────────────────────────
pd.DataFrame(rows).to_csv(os.path.join(OUT, "r082_multitf.csv"), index=False)
lines = [f"# R082 — Multi-Timeframe / ML / Exit new levers\n",
         f"**Date:** 2026-08-06 | base = Family A RAW (E6, RR1.5): 14.9 t/mo, PF 1.48, "
         f"MDD -31%, prof-mo 48%\n",
         f"**Target:** t/mo>=8, prof-mo%>=65, worst<=3, holPF>1.1, PF@0.05%>1.1\n",
         f"\n## Results\n",
         "| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    best = max(passed, key=lambda r: r["tpm"] * r["holpf"])
    lines.append(f"**✅ {best['cfg']} meets the retail spec:** {best['tpm']:.1f} t/mo, "
                 f"{best['prof']*100:.0f}% profitable months, worst streak {best['worst']}, "
                 f"PF {best['pf']:.2f} (cost {best['pf_c']:.2f}), holPF {best['holpf']:.2f}.")
else:
    lines.append("**❌ No config meets ALL criteria.** Honest result — see table. "
                 "Closest configs listed above.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r082_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r082_*")
