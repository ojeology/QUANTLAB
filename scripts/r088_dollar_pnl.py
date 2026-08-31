"""
R088: $100 account, $2 risk/trade, SVM q0.75 — PnL and drawdown in DOLLARS.

Scenarios:
  A) start Jan 2024 (full history), FIXED $2 risk per trade (non-compounding)
  B) start Jan 2024, 2% of equity compounding per trade
  C) start Jan 2026 (holdout year only), FIXED $2 risk
  D) start Jan 2026, 2% compounding

$2 risk/trade @ RR1.5: each loss = -$2.00, each win = +$3.00.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/quantlab")
sys.path.insert(0, "/home/user/quantlab/scripts")
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, build_signal_mask, sim_symbol, IS_LOOKBACK, RECAL_EVERY,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CACHE = CONFIG["CACHE_FOLDER"]
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
RR = 1.5
Q = 0.75
RISK_USD = 2.0
START_CAP = 100.0

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

print("Loading data (73 symbols) …")
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
thr = pd.Series(pred).where(sel_mask.values).dropna().quantile(1 - Q)
keep = set(mldf.loc[pred >= thr, "ts"])
trades = [t for t in raw if t["entry_time"] in keep]
trades.sort(key=lambda t: t["entry_time"])
print(f"  Kept trades (q={Q}): {len(trades)}")

# ── Dollar PnL / drawdown ────────────────────────────────────────────────────
def report(label, trades, mode):
    if not trades:
        print(f"  {label}: no trades"); return
    n = len(trades)
    wins = sum(1 for t in trades if t["r"] > 0)
    if mode == "fixed":
        eq = START_CAP; peak = START_CAP; maxdd = 0.0
        pnls = []
        for t in trades:
            pnl = RISK_USD * t["r"]
            eq += pnl; pnls.append(pnl)
            peak = max(peak, eq); maxdd = min(maxdd, eq - peak)
        total = eq - START_CAP
        print(f"  {label} (fixed ${RISK_USD}/trade):")
        print(f"    trades={n}  wins={wins} ({wins/n*100:.0f}%)  losses={n-wins}")
        print(f"    TOTAL PnL = ${total:+,.2f}  (${START_CAP:.0f} → ${eq:,.2f})")
        print(f"    MAX DRAWDOWN = ${maxdd:,.2f}  ({maxdd/START_CAP*100:.1f}% of start)")
        # worst monthly
        df = pd.DataFrame(trades); df["month"] = df["entry_time"].dt.to_period("M")
        g = df.groupby("month")["r"].sum() * RISK_USD
        print(f"    best month ${g.max():+.2f} | worst month ${g.min():+.2f} | "
              f"profitable months {(g>0).mean()*100:.0f}%")
    else:
        eq = START_CAP; peak = START_CAP; maxdd = 0.0
        for t in trades:
            eq *= (1 + (RISK_USD / START_CAP) * t["r"])
            peak = max(peak, eq); maxdd = min(maxdd, (eq - peak) / peak)
        total = eq - START_CAP
        print(f"  {label} (compounding 2%/trade):")
        print(f"    trades={n}  wins={wins} ({wins/n*100:.0f}%)  losses={n-wins}")
        print(f"    TOTAL PnL = ${total:+,.2f}  (${START_CAP:.0f} → ${eq:,.2f})")
        print(f"    MAX DRAWDOWN = ${maxdd*START_CAP:,.2f}  ({maxdd*100:.1f}%)")

print("\n" + "="*70)
print("  $100 ACCOUNT, $2 RISK/TRADE, SVM q0.75 (RR 1.5 → win +$3, loss -$2)")
print("="*70)

full = trades
jan26 = [t for t in trades if t["entry_time"] >= HOLDOUT_START]

print("\n  ── START JANUARY 2024 (FULL 2.5-YEAR HISTORY) ──")
report("A", full, "fixed")
report("B", full, "compound")

print("\n  ── START JANUARY 2026 (THIS YEAR, HOLD-OUT) ──")
report("C", jan26, "fixed")
report("D", jan26, "compound")

# monthly calendar in dollars (fixed $2) for the 2026 portion
print("\n  MONTHLY PnL (fixed $2/trade, 2026):")
df = pd.DataFrame(jan26); df["month"] = df["entry_time"].dt.to_period("M")
g = df.groupby("month")["r"].sum() * RISK_USD
for m, v in g.items():
    print(f"    {m}: {v:+.2f}$")
