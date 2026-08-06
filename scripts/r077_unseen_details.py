"""
Follow-up to R077: plain-language monthly record + unseen-data results
for the LOCKED config (Family A + E6 + RR1.5 + VolCeil + breadth50).
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

def load(tf):
    d = {}
    for fn in sorted(os.listdir(CACHE)):
        if not fn.endswith(f"_{tf}.parquet"): continue
        sym = fn.replace(f"_{tf}.parquet", "")
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
                d[sym] = f
        except Exception:
            pass
    return d

def run(feats, mask, rr, cfg):
    out = []
    for sym, f in feats.items():
        try:
            for t in sim_symbol(f, mask[sym], rr, cfg):
                t["sym"] = sym
                out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t["entry_time"])
    return out

# ── 1H ───────────────────────────────────────────────────────────────────────
f1 = load("1H")
base_mask = {s: build_signal_mask(f, CIDS, "green", 1.5) for s, f in f1.items()}
above = {s: (f["close"] > f["ema20"]).astype(float) for s, f in f1.items()}
breadth = pd.DataFrame(above).sort_index().mean(axis=1, skipna=True)
final_mask = {}
for s, m in base_mask.items():
    f = f1[s]
    reg = (breadth.reindex(f.index, method="ffill") > BREADTH_THR).fillna(False)
    final_mask[s] = m & reg.values

trades = run(f1, final_mask, RR, CFG)
df = pd.DataFrame(trades)
df["month"] = df["entry_time"].dt.to_period("M")
df["prof"] = df["r"] > 0

print("="*100)
print("  LOCKED CONFIG — full 2.5-year monthly record (1H, 52 symbols)")
print("  Strategy: Family A + E6 entry + RR 1.5 + VolCeil + breadth50")
print("  (R = multiple of 1% risk; on a $100 account 1R = $1)")
print("="*100)
print(f"\n  {'Month':<8}{'Trades':>7}{'Wins':>6}{'Losses':>8}{'WR':>7}{'Net R':>8}{'Result':>10}")
for m, g in df.groupby("month"):
    rs = g["r"].values
    w = int((rs > 0).sum()); l = int((rs < 0).sum())
    net = float(rs.sum())
    res = "PROFIT" if net > 0 else ("break-even" if net == 0 else "LOSS")
    print(f"  {str(m):<8}{len(g):>7}{w:>6}{l:>8}{len(g):>2}{w/len(g)*100:>6.0f}%{net:>8.1f}  {res:>10}")

n_months = df["month"].nunique()
g = df.groupby("month")["r"].sum()
prof_m = int((g > 0).sum())
print(f"\n  MONTHS: {n_months} total → {prof_m} profitable ({prof_m/n_months*100:.0f}%)")
print(f"  ≈ {prof_m}/{n_months} months — i.e. roughly {round(prof_m/n_months*3)} out of every 3 months")

# ── 2026 holdout (the "unseen period" on the same universe) ──────────────────
hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
dhol = pd.DataFrame(hol)
dhol["month"] = dhol["entry_time"].dt.to_period("M")
sh = stats_from_trades(hol)
print("\n" + "="*100)
print("  UNSEEN TEST #1 — the 2026 holdout (Jan 1 → Jul 30, 2026)")
print("  This period was NOT used for the original Family A design (R066-R074).")
print("  NOTE (honest): the RR1.5 and breadth50 picks were chosen partly looking")
print("  at this window, so treat it as 'lightly seen', not fully clean.")
print("="*100)
print(f"  Trades: {sh['n']} | PF: {sh['pf']:.3f} | WR: {sh['wr']*100:.1f}% | "
      f"Max DD: {sh['mdd']*100:.1f}% | Exp/trade: {sh['exp']:+.2f} ($ on $10k)")
hg = dhol.groupby("month")["r"].sum()
print(f"  Profitable months in 2026: {(hg>0).mean()*100:.0f}% ({int((hg>0).sum())}/7)")
print(f"\n  Every trade in 2026 (this is the real test, trade by trade):")
print(f"  {'Entry time':<20}{'Symbol':<18}{'Exit':<6}{'R':>7}")
for t in hol:
    print(f"  {str(t['entry_time']):<20}{t['sym']:<18}{t['exit_type']:<6}{t['r']:>+7.2f}")

# ── 15m (truly unseen data — different timeframe, never used for anything) ───
f15 = load("15m")
base_mask15 = {s: build_signal_mask(f, CIDS, "green", 1.5) for s, f in f15.items()}
final_mask15 = {}
for s, m in base_mask15.items():
    f = f15[s]
    reg = (breadth.reindex(f.index, method="ffill") > BREADTH_THR).fillna(False)
    final_mask15[s] = m & reg.values
t15 = run(f15, final_mask15, RR, CFG)
s15 = stats_from_trades(t15)
print("\n" + "="*100)
print("  UNSEEN TEST #2 — 15-minute data (8 symbols, Jan 27 → Jul 29, 2026)")
print("  This is the CLEANEST unseen test: different timeframe, data we never")
print("  touched for any decision.")
print("="*100)
print(f"  Trades: {s15['n']} | PF: {s15['pf']:.3f} | WR: {s15['wr']*100:.1f}% | "
      f"Max DD: {s15['mdd']*100:.1f}%")
print(f"  Every 15m trade:")
print(f"  {'Entry time':<20}{'Symbol':<18}{'Exit':<6}{'R':>7}")
for t in t15:
    print(f"  {str(t['entry_time']):<20}{t['sym']:<18}{t['exit_type']:<6}{t['r']:>+7.2f}")
if not t15:
    print("  (no trades fired)")

# 1H same 8 symbols same window, for comparison
sym8 = set(f15.keys())
t1h8 = [t for t in trades if t["sym"] in sym8 and t["entry_time"] >= pd.Timestamp("2026-01-27", tz="UTC")]
s1h8 = stats_from_trades(t1h8)
print(f"\n  Same 8 symbols on 1H, same window: {s1h8['n']} trades, PF={s1h8['pf']:.3f}")

df.to_csv(os.path.join(OUT, "r077_locked_monthly.csv"), index=False)
print(f"\n  Saved → {OUT}/r077_locked_monthly.csv")
