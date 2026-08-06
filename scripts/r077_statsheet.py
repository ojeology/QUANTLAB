"""
Complete stat sheet for the LOCKED config
(Family A + E6 entry + RR1.5 + VolCeil(<=70) + breadth50, 52 symbols, 1H).
Answers: trades/month, WR, PF, drawdown in $ on $100, streaks, duration, exits.
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
    bootstrap_pf, loo_symbol_floor, monte_carlo, IS_LOOKBACK, RECAL_EVERY,
)

CACHE = CONFIG["CACHE_FOLDER"]
OUT   = CONFIG["OUTPUT_FOLDER"]

CIDS = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
RR = 1.5
CFG = dict(entry_next=False, exit="base", hours=None, atr_rank_ceil=70.0)
BREADTH_THR = 0.50

# original 52
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

f1 = {}
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
trades.sort(key=lambda t: t["entry_time"])

df = pd.DataFrame(trades)
df["month"] = df["entry_time"].dt.to_period("M")
rs = df["r"].values
s = stats_from_trades(trades)
b5, bmed, b95 = bootstrap_pf(rs)
floor, rm = loo_symbol_floor(trades)

g = df.groupby("month")["r"].sum()
months_all = (df["entry_time"].max().to_period("M") - df["entry_time"].min().to_period("M")).n  # span in months
n_active = len(g)
flags = (g > 0).astype(int).values
cur = worst_m_streak = 0
for v in flags:
    cur = cur + 1 if not v else 0
    worst_m_streak = max(worst_m_streak, cur)
cur = best_m_streak = 0
for v in flags:
    cur = cur + 1 if v else 0
    best_m_streak = max(best_m_streak, cur)
# trade-level losing streak
st = 0; worst_t_streak = 0
for r in rs:
    if r < 0: st += 1; worst_t_streak = max(worst_t_streak, st)
    else: st = 0
st = 0; best_t_streak = 0
for r in rs:
    if r > 0: st += 1; best_t_streak = max(best_t_streak, st)
    else: st = 0

# MC $ drawdown at 1% on $100
rng = np.random.default_rng(42)
dds, finals = [], []
for _ in range(5000):
    sh = rs[rng.integers(0, len(rs), len(rs))]
    cap = 100.0; eq = [cap]
    for r in sh:
        cap += cap * 0.01 * r
        eq.append(cap)
    eq = np.array(eq); pk = np.maximum.accumulate(eq)
    dds.append(float(((eq - pk) / pk).min()) * 100)
    finals.append(cap)
dds = np.array(dds); finals = np.array(finals)

print("=" * 78)
print("  LOCKED CONFIG — COMPLETE STAT SHEET")
print("  Family A + E6 entry + RR 1.5 + VolCeil(<=70) + breadth50 | 52 symbols | 1H")
print("=" * 78)
print(f"\n  ⏱  TRADE FREQUENCY")
print(f"    Total trades:            {len(df)}")
print(f"    Data span:               {df['entry_time'].min():%b %Y} → {df['entry_time'].max():%b %Y} ({months_all+1} months)")
print(f"    Months with trades:      {n_active} of {months_all+1} months ({n_active/(months_all+1)*100:.0f}%)")
print(f"    Trades/month (all months):  {len(df)/(months_all+1):.1f}")
print(f"    Trades/month (active only): {len(df)/n_active:.1f}")
print(f"    Median month:            {g.median():.0f} trades | min: {g.min()} | max: {g.max()}")

print(f"\n  📊  PERFORMANCE")
print(f"    Win rate:                {s['wr']*100:.1f}%  ({int((rs>0).sum())}W / {int((rs<0).sum())}L)")
print(f"    Profit factor:           {s['pf']:.3f}  (make ${s['pf']:.2f} per $1 lost)")
print(f"    Expectancy:              +{s['exp']:.2f} $/trade at 1% risk on $10k")
print(f"    Total net (1% risk):     {s['net']:+,.0f}$ on $10k → {s['net']/10000*100:+.0f}% over {months_all+1} months")
print(f"    Boot P5 / median / P95:  {b5:.3f} / {bmed:.3f} / {b95:.3f}")
print(f"    LOO-symbol floor:        {floor:.3f} (drop {rm})")

print(f"\n  💸  DRAWODOWN — $100 ACCOUNT at 1% risk/trade (1R = $1)")
print(f"    Worst month:             {g.idxmin()}  ({g.min():+.1f}R ≈ ${g.min():+.1f})")
print(f"    Best month:              {g.idxmax()}  ({g.max():+.1f}R ≈ ${g.max():+.1f})")
print(f"    MC worst DD:             P5 = ${-np.percentile(dds,5):.2f} | typical = ${-np.percentile(dds,50):.2f} | best = ${-np.percentile(dds,95):.2f}")
print(f"    P(end > $100):           {(finals>100).mean()*100:.0f}%")
print(f"    P(end > $120):           {(finals>120).mean()*100:.0f}%")
print(f"    P(end < $80):            {(finals<80).mean()*100:.0f}%")

print(f"\n  📅  MONTHLY RHYTHM")
print(f"    Profitable months:       {(g>0).mean()*100:.0f}%  ({int((g>0).sum())}/{n_active})")
print(f"    Best winning-month streak: {best_m_streak}")
print(f"    Worst losing-month streak: {worst_m_streak}")
print(f"    Best losing-trade streak:  {worst_t_streak} consecutive losses (1R each = -${worst_t_streak:.0f} on $100)")
print(f"    Best winning-trade streak: {best_t_streak} consecutive wins")

print(f"\n  ⏳  TRADE DURATION & EXITS")
print(f"    Avg bars held:           {df['bars_in'].mean():.1f} (median {df['bars_in'].median():.0f}, 90th pct {df['bars_in'].quantile(0.9):.0f})")
print(f"    Avg hours held:          {df['bars_in'].mean():.0f}h  (median {df['bars_in'].median():.0f}h)")
et = df["exit_type"].value_counts()
for k, v in et.items():
    print(f"    {k}: {v} ({v/len(df)*100:.0f}%)")

print(f"\n  💰  COST HEADROOM (from R073/R074)")
print(f"    Breakeven cost:          ~0.145% per side → at 0.05-0.10% costs PF ≈ 1.4-1.6")
print(f"    Family C was removed; only Family A runs.")

# top symbols by trades
print(f"\n  🪙  MOST ACTIVE SYMBOLS")
print(df.groupby("sym").size().sort_values(ascending=False).head(8).to_string())

df.to_csv(os.path.join(OUT, "r077_locked_statsheet.csv"), index=False)
print(f"\n  Saved → {OUT}/r077_locked_statsheet.csv")
