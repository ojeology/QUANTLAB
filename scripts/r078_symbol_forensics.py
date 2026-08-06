"""
Per-symbol forensics for the LOCKED config (Family A + E6 + RR1.5 + VolCeil + breadth50).
Which symbols are negative the entire time? Full period + selection + holdout.
"""
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/quantlab")
sys.path.insert(0, "/home/user/quantlab/scripts")
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, build_signal_mask, sim_symbol, stats_from_trades,
    IS_LOOKBACK, RECAL_EVERY,
)

CACHE = CONFIG["CACHE_FOLDER"]
OUT   = CONFIG["OUTPUT_FOLDER"]
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")

CIDS = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
RR = 1.5
CFG = dict(entry_next=False, exit="base", hours=None, atr_rank_ceil=70.0)
BREADTH_THR = 0.50

f1 = {}
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
            f1[sym] = f
    except Exception:
        pass

base_mask = {s: build_signal_mask(f, CIDS, "green", 1.5) for s, f in f1.items()}
above = {s: (f["close"] > f["ema20"]).astype(float) for s, f in f1.items()}
breadth = pd.DataFrame(above).sort_index().mean(axis=1, skipna=True)
final_mask = {}
for s, m in base_mask.items():
    f = f1[s]
    reg = (breadth.reindex(f.index, method="ffill") > BREADTH_THR).fillna(False)
    final_mask[s] = m & reg.values

trades = []
for sym, f in f1.items():
    try:
        for t in sim_symbol(f, final_mask[sym], RR, CFG):
            t["sym"] = sym
            trades.append(t)
    except Exception:
        pass

df = pd.DataFrame(trades)
print(f"Total trades: {len(df)} across {df['sym'].nunique()} symbols")
print(f"\n  PER-SYMBOL BREAKDOWN (locked config, full period):")
print(f"  {'Symbol':<18}{'n':>4}{'WR':>7}{'PF':>7}{'Net R':>8}{'selPF':>7}{'holPF':>7}{'Ever neg?':>10}")
rows = []
for sym, g in df.groupby("sym"):
    rs = g["r"].values
    w = rs[rs>0].sum(); l = abs(rs[rs<0].sum())
    pf = w/l if l>0 else float('inf')
    sel = g[g["entry_time"] < HOLDOUT_START]["r"].values
    hol = g[g["entry_time"] >= HOLDOUT_START]["r"].values
    def pfx(x):
        if len(x)==0: return float('nan')
        ww = x[x>0].sum(); ll = abs(x[x<0].sum())
        return ww/ll if ll>0 else float('inf')
    sp, hp = pfx(sel), pfx(hol)
    ever_neg = "YES" if (len(sel)==0 or sp<=1) and (len(hol)==0 or hp<=1) else ""
    rows.append(dict(sym=sym, n=len(g), wr=float((rs>0).mean()), pf=pf, net=float(rs.sum()),
                     selpf=sp, holpf=hp, ever_neg=ever_neg))
    print(f"  {sym:<18}{len(g):>4}{np.mean(rs>0)*100:>6.0f}%{pf:>7.2f}{rs.sum():>8.1f}"
          f"{sp:>7.2f}{hp:>7.2f}  {ever_neg:>10}")

rdf = pd.DataFrame(rows)
print(f"\n  Symbols with NO trades: {df['sym'].nunique()} traded; universe was {len(f1)}")
no_trade = set(f1.keys()) - set(df["sym"].unique())
print(f"  Symbols that never triggered: {sorted(no_trade) if no_trade else 'none'}")

neg_all = rdf[rdf["ever_neg"]=="YES"]
print(f"\n  Symbols negative 'the entire time' (PF<=1 in BOTH selection and holdout):")
if len(neg_all):
    print(neg_all.to_string(index=False))
else:
    print("  NONE — every symbol that traded was profitable in at least one period.")

# sum of negative-full-period symbols
full_neg = rdf[rdf["pf"]<1]
print(f"\n  Symbols negative over FULL period: {len(full_neg)}")
print(f"  {full_neg[['sym','n','pf','net']].to_string(index=False)}" if len(full_neg) else "  none")

# trades per symbol median
print(f"\n  Median trades/symbol: {rdf['n'].median():.0f} | mean: {rdf['n'].mean():.1f} | max: {rdf['n'].max()}")

rdf.to_csv(os.path.join(OUT, "r078_symbol_forensics.csv"), index=False)
print(f"\n  Saved → {OUT}/r078_symbol_forensics.csv")
