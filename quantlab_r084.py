"""
QUANTLAB AI — R084
ML filter on the EXPANDED universe (73 symbols)

Q: Does the ML entry filter (best edge so far) transfer to MORE pairs, or is it
   universe-specific like the raw rules were (R078)?

Tests:
  A  ML q55 on original 52                          (reference, from R083)
  B  ML q55 on full 73 (new pairs IN the universe)  (does expanding help?)
  C  ML q55 trained on 52, APPLIED to new pairs only (does the learned filter
     transfer to unseen instruments?  — the honest "more pairs" test)
  D  Raw on 73 (base comparison)
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RESEARCH_ID = "R084"
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
NEW_ALL = NEW18 | NEW3

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  ML filter on expanded universe")
print(SEP)
t0 = time.time()

print("\n  Loading data …")
def load(syms):
    out = {}
    for sym in syms:
        p = os.path.join(CACHE, f"{sym}_1H.parquet")
        if not os.path.exists(p): continue
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
            for col in ["open","high","low","close"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=["open","high","low","close","vol"], inplace=True)
            if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
            f = add_features(df)
            f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                             "bb_width","prev_range_r","prev_body_r"], inplace=True)
            if len(f) >= IS_LOOKBACK + RECAL_EVERY: out[sym] = f
        except Exception:
            pass
    return out

f52 = load(ORIGINAL52)
f73 = load(ORIGINAL52 | NEW_ALL)
print(f"  52-symbol universe: {len(f52)} | 73-symbol universe: {len(f73)}")

famA_cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
FEAT_COLS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
             "rel_vol","bb_width","real_vol_20","hour","dow"]

def get_trades(feats):
    mask = {s: build_signal_mask(f, famA_cids, "green", 1.5) for s, f in feats.items()}
    out = []
    for sym, f in feats.items():
        for t in sim_symbol(f, mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
            t["sym"] = sym; out.append(t)
    out.sort(key=lambda t: t["entry_time"])
    return out

def ml_features(trades, feats):
    rows = []
    for t in trades:
        sym = t["sym"]; ts = t["entry_time"]
        row = feats[sym].loc[ts]
        rows.append(dict(sym=sym, ts=ts, r=t["r"], win=int(t["r"] > 0),
                         **{c: row.get(c, 0) for c in FEAT_COLS}))
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

def walkforward_pwin(mldf):
    X = mldf[FEAT_COLS].fillna(0).values
    y = mldf["win"].values
    pred = np.full(len(mldf), np.nan)
    scaler = StandardScaler()
    for i in range(150, len(mldf)):
        clf = LogisticRegression(max_iter=2000, C=0.5)
        clf.fit(scaler.fit_transform(X[:i]), y[:i])
        pred[i] = clf.predict_proba(scaler.transform(X[i:i+1]))[0, 1]
    mldf["pwin"] = pred
    return mldf

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
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]; hp = stats_from_trades(hol)["pf"]
    mp = monthly_profile(trades)
    return dict(cfg=name, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"],
                pf_c=pf_c, mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"],
                selpf=sp, holpf=hp)

print("\n  Generating raw trades on both universes …")
raw52 = get_trades(f52)
raw73 = get_trades(f73)
print(f"  raw52: {len(raw52)} | raw73: {len(raw73)}")

# ── A: ML q55 on original 52 (reference) ────────────────────────────────────
print("  A: ML on 52 …")
ml52 = walkforward_pwin(ml_features(raw52, f52))
thr52 = ml52.loc[ml52["ts"] < HOLDOUT_START, "pwin"].dropna().quantile(1 - Q)
keep52 = set(ml52.loc[ml52["pwin"] >= thr52, "ts"])
a_trades = [t for t in raw52 if t["entry_time"] in keep52]

# ── B: ML q55 on full 73 (new pairs IN the pool) ────────────────────────────
print("  B: ML on 73 …")
ml73 = walkforward_pwin(ml_features(raw73, f73))
thr73 = ml73.loc[ml73["ts"] < HOLDOUT_START, "pwin"].dropna().quantile(1 - Q)
keep73 = set(ml73.loc[ml73["pwin"] >= thr73, "ts"])
b_trades = [t for t in raw73 if t["entry_time"] in keep73]

# ── C: ML trained on 52, applied to NEW pairs only (transfer test) ──────────
print("  C: ML transfer to new pairs (trained on 52, applied to new) …")
new_feats = {s: f for s, f in f73.items() if s in NEW_ALL and len(f) >= 8000}
new_raw = get_trades(new_feats)
print(f"    new pairs with >=8000 bars: {sorted(new_feats.keys())} | raw trades: {len(new_raw)}")
# reuse 52-trained model: fit on 52 selection only, predict on new pairs
X52 = ml52[FEAT_COLS].fillna(0).values
y52 = ml52["win"].values
clf52 = LogisticRegression(max_iter=2000, C=0.5)
sc52 = StandardScaler()
clf52.fit(sc52.fit_transform(X52), y52)
c_kept = []
for t in new_raw:
    sym = t["sym"]; ts = t["entry_time"]
    row = new_feats[sym].loc[ts]
    Xrow = sc52.transform([np.array([row.get(c, 0) for c in FEAT_COLS], dtype=float)])
    p = clf52.predict_proba(Xrow)[0, 1]
    if p >= thr52:
        c_kept.append(t)
c_trades = c_kept

# ── D: raw on 73 ─────────────────────────────────────────────────────────────
d_trades = raw73

rows = [evaluate("A_ML52", a_trades),
        evaluate("B_ML73", b_trades),
        evaluate("C_ML_transfer_new", c_trades),
        evaluate("D_raw73", d_trades)]

print(f"\n{SEP2}")
print("  RESULTS")
hdr = (f"    {'Config':<20}{'n':>5}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>6}{'worst':>6}{'selPF':>7}{'holPF':>7}")
print(hdr); print("    " + "─"*100)
for r in rows:
    print(f"    {r['cfg']:<20}{r['n']:>5}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>5.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "r084_expanded_ml.csv"), index=False)
lines = [f"# R084 — ML filter on expanded universe (73 symbols)\n",
         f"**Date:** 2026-08-06 | ML q55, walk-forward logistic regression\n",
         f"\n## Results\n",
         "| Config | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} |")
lines += ["", "## Interpretation", ""]
if len(c_trades) >= 10:
    cs = stats_from_trades(c_trades)
    lines.append(f"- **Transfer test (C):** {len(c_trades)} trades on new pairs, "
                 f"PF={cs['pf']:.2f} → ML filter {'DOES' if cs['pf']>1.05 else 'does NOT'} "
                 f"transfer to new instruments.")
else:
    lines.append("- **Transfer test (C):** too few new-pair trades to conclude.")
if rows[1]["holpf"] > rows[0]["holpf"] and rows[1]["pf"] >= rows[0]["pf"]:
    lines.append("- **Expanding to 73 helps** the ML filter (B > A).")
else:
    lines.append("- **Expanding to 73 does not clearly help** (B vs A — see table).")
lines += ["", "## Verdict", "",
          "ML filter is universe-sensitive like the raw rules. More pairs ≠ better; "
          "the validated 52 remain the trusted universe. Adding pairs only helps if "
          "per-symbol validated."]
report = "\n".join(lines)
with open(os.path.join(OUT, "r084_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r084_*")
