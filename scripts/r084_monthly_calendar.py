"""
Monthly calendar for the winning config: ML q55 on the 73-symbol universe.
Also splits trades by universe segment (original 52 vs the 18 vs the 3).
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CACHE = CONFIG["CACHE_FOLDER"]
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

feats = load(ORIGINAL52 | NEW18 | NEW3)
cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
FEAT_COLS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
             "rel_vol","bb_width","real_vol_20","hour","dow"]

mask = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats.items()}
raw = []
for sym, f in feats.items():
    for t in sim_symbol(f, mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"raw trades on 73: {len(raw)}")

rows = []
for t in raw:
    row = feats[t["sym"]].loc[t["entry_time"]]
    rows.append(dict(sym=t["sym"], ts=t["entry_time"], r=t["r"], win=int(t["r"]>0),
                     **{c: row.get(c,0) for c in FEAT_COLS}))
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
X = mldf[FEAT_COLS].fillna(0).values; y = mldf["win"].values
pred = np.full(len(mldf), np.nan); sc = StandardScaler()
for i in range(150, len(mldf)):
    clf = LogisticRegression(max_iter=2000, C=0.5)
    clf.fit(sc.fit_transform(X[:i]), y[:i])
    pred[i] = clf.predict_proba(sc.transform(X[i:i+1]))[0,1]
mldf["pwin"] = pred
thr = mldf.loc[mldf["ts"] < HOLDOUT_START, "pwin"].dropna().quantile(1 - Q)
keep = mldf.loc[mldf["pwin"] >= thr]
print(f"kept: {len(keep)}")

df = keep.copy()
df["month"] = df["ts"].dt.to_period("M")
df["seg"] = np.where(df["sym"].isin(ORIGINAL52), "orig52",
            np.where(df["sym"].isin(NEW18), "new18", "new3"))

print("\n=== MONTHLY CALENDAR (ML q55 on 73) ===")
print(f"{'Month':<9}{'trades':>7}{'wins':>6}{'loss':>6}{'netR':>8}{'result':>10}")
g = df.groupby("month")
prof = 0; loss = 0
for m, grp in g:
    rs = grp["r"].values
    net = float(rs.sum())
    res = "PROFIT" if net > 0 else ("flat" if abs(net) < 0.01 else "LOSS")
    if net > 0: prof += 1
    elif net < 0: loss += 1
    print(f"{str(m):<9}{len(grp):>7}{int((rs>0).sum()):>6}{int((rs<0).sum()):>6}{net:>+8.1f}{res:>10}")
print(f"\nProfitable months: {prof} | Losing months: {loss} | Total: {prof+loss}")
print(f"Profitable-month rate: {prof/(prof+loss)*100:.0f}%")

print("\n=== TRADES BY UNIVERSE SEGMENT (the honest split) ===")
for seg in ["orig52", "new18", "new3"]:
    sub = df[df["seg"] == seg]
    if len(sub) == 0:
        print(f"  {seg}: 0 trades"); continue
    rs = sub["r"].values
    w = rs[rs>0].sum(); l = abs(rs[rs<0].sum())
    pf = w/l if l > 0 else float('inf')
    hol = sub[sub["ts"] >= HOLDOUT_START]
    print(f"  {seg}: n={len(sub)}  netR={rs.sum():+.1f}  PF={pf:.2f}  "
          f"holdout trades={len(hol)}  holdout netR={hol['r'].sum():+.1f}")

print("\n=== NEW-18/3 TRADES (which of them actually fired) ===")
newsub = df[df["seg"] != "orig52"]
if len(newsub):
    for _, t in newsub.sort_values("ts").iterrows():
        print(f"  {str(t['ts']):<20}{t['sym']:<18}{t['r']:>+6.2f}")
else:
    print("  none fired")

df.to_csv(os.path.join(CONFIG["OUTPUT_FOLDER"], "r084_monthly_calendar.csv"), index=False)
print("\nsaved → quantlab_output/r084_monthly_calendar.csv")
