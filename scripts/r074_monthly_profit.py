"""
Monthly profitability breakdown for the FINAL strategy
(Family A + E6_sigentry + RR 2.0 + VOLCEIL atr_rank<=70).
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/quantlab")
sys.path.insert(0, "/home/user/quantlab/scripts")
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, build_signal_mask, run_family, stats_from_trades,
    IS_LOOKBACK, RECAL_EVERY,
)

OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]

CIDS = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
RR = 2.0
CFG = dict(entry_next=False, exit="base", hours=None, atr_rank_ceil=70.0)

feats_by_sym = {}
for fn in sorted(os.listdir(CACHE)):
    if not fn.endswith("_1H.parquet"): continue
    sym = fn.replace("_1H.parquet", "")
    try:
        df = pd.read_parquet(os.path.join(CACHE, fn))
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats_by_sym[sym] = f
    except Exception:
        pass

mask = {sym: build_signal_mask(f, CIDS, "green", 1.5) for sym, f in feats_by_sym.items()}
trades = run_family(CIDS, RR, CFG, feats_by_sym, mask)
df = pd.DataFrame(trades)
df["month"] = df["entry_time"].dt.to_period("M")

print(f"TOTAL: {len(df)} trades | {df['month'].nunique()} months "
      f"({df['entry_time'].min()} → {df['entry_time'].max()})")
print()

rows = []
for m, g in df.groupby("month"):
    rs = g["r"].values
    wins = rs[rs > 0]; losses = rs[rs < 0]
    gw, gl = wins.sum(), abs(losses.sum())
    pf = gw / gl if gl > 0 else float("inf")
    rows.append(dict(month=str(m), n=len(g), wins=int((rs > 0).sum()),
                     losses=int((rs < 0).sum()), wr=float((rs > 0).mean()),
                     pf=pf, net_r=float(rs.sum())))
mdf = pd.DataFrame(rows)
mdf["profitable"] = mdf["net_r"] > 0

prof = mdf[mdf["profitable"]]
loss = mdf[~mdf["profitable"]]
print("MONTHLY RECORD (net R = profit in R-multiples; 1R = 1% of account):")
print(mdf.to_string(index=False))
print()
print(f"Profitable months: {len(prof)} / {len(mdf)}  ({len(prof)/len(mdf)*100:.0f}%)")
print(f"Losing months:     {len(loss)} / {len(mdf)}  ({len(loss)/len(mdf)*100:.0f}%)")
print(f"Avg net R in profit months: {prof['net_r'].mean():+.2f}")
print(f"Avg net R in loss months:   {loss['net_r'].mean():+.2f}")
print(f"Sum of profit months: {prof['net_r'].sum():+.1f}R")
print(f"Sum of loss months:   {loss['net_r'].sum():+.1f}R")
print(f"Net over all: {mdf['net_r'].sum():+.1f}R")
print(f"Worst month: {mdf.loc[mdf['net_r'].idxmin(), 'month']} ({mdf['net_r'].min():+.1f}R)")
print(f"Best month:  {mdf.loc[mdf['net_r'].idxmax(), 'month']} ({mdf['net_r'].max():+.1f}R)")

# streaks
prof_flag = mdf["profitable"].astype(int).values
cur = best_streak = 0
for v in prof_flag:
    cur = cur + 1 if v else 0
    best_streak = max(best_streak, cur)
cur = worst_streak = 0
for v in prof_flag:
    cur = cur + 1 if not v else 0
    worst_streak = max(worst_streak, cur)
print(f"Best profitable-month streak: {best_streak}")
print(f"Worst losing-month streak:    {worst_streak}")

# cumulative R by month
mdf["cum_r"] = mdf["net_r"].cumsum()
print()
print("Cumulative R by month:")
print(mdf[["month","net_r","cum_r"]].to_string(index=False))

# year breakdown
mdf["year"] = mdf["month"].str[:4]
print()
print("BY YEAR:")
for y, g in mdf.groupby("year"):
    print(f"  {y}: {len(g)} months, {len(g[g['profitable']])} profitable, "
          f"net {g['net_r'].sum():+.1f}R")

mdf.to_csv(os.path.join(OUT, "r074_final_monthly.csv"), index=False)
print(f"\nSaved → {OUT}/r074_final_monthly.csv")
