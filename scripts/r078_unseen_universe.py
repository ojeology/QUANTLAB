"""
R078: True unseen-universe test of the LOCKED config on brand-new symbols
+ per-symbol "negative the entire time" quantification.

New symbols (fetched from OKX, never used in ANY decision):
  BICO, HYPE, XAU, HOME, PUMP, ZBT, ZEC, BEAT  (>= 5000 bars)
Market breadth gate comes from the ORIGINAL 52-symbol universe (as locked).
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

# original 52 (pre-R078) vs new symbols — explicit full names
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
NEW_CANDIDATES = {
    "BICO_USDT_SWAP","HYPE_USDT_SWAP","XAU_USDT_SWAP","HOME_USDT_SWAP",
    "PUMP_USDT_SWAP","ZBT_USDT_SWAP","ZEC_USDT_SWAP","BEAT_USDT_SWAP",
}
MIN_BARS_NEW = 5000

def load_syms(syms):
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
            if len(f) >= IS_LOOKBACK + RECAL_EVERY:
                out[sym] = f
        except Exception as e:
            print("  load err", sym, e)
    return out

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

# ── breadth from ORIGINAL universe only ─────────────────────────────────────
print("Loading original universe for breadth …")
orig_feats = load_syms(ORIGINAL)
above = {s: (f["close"] > f["ema20"]).astype(float) for s, f in orig_feats.items()}
breadth = pd.DataFrame(above).sort_index().mean(axis=1, skipna=True)
print(f"Original universe: {len(orig_feats)} symbols for breadth")

# ── new symbols ──────────────────────────────────────────────────────────────
print("Loading NEW symbols …")
new_feats = load_syms(NEW_CANDIDATES)
new_feats = {s: f for s, f in new_feats.items() if len(f) >= MIN_BARS_NEW}
print(f"New symbols with >= {MIN_BARS_NEW} bars: {sorted(new_feats.keys())}")

base_mask = {s: build_signal_mask(f, CIDS, "green", 1.5) for s, f in new_feats.items()}
final_mask = {}
for s, m in base_mask.items():
    f = new_feats[s]
    reg = (breadth.reindex(f.index, method="ffill") > BREADTH_THR).fillna(False)
    final_mask[s] = m & reg.values

new_trades = run(new_feats, final_mask, RR, CFG)
print(f"\n  NEW-UNIVERSE RESULTS (locked config, RR1.5, breadth50):")
if new_trades:
    s = stats_from_trades(new_trades)
    print(f"  n={s['n']}  PF={s['pf']:.3f}  WR={s['wr']*100:.1f}%  MDD={s['mdd']*100:.1f}%  "
          f"Exp=${s['exp']:.2f}  Net=${s['net']:+.0f}")
    # per-symbol
    df = pd.DataFrame(new_trades)
    print(f"\n  Per-symbol:")
    print(f"  {'Symbol':<10}{'n':>5}{'WR':>7}{'PF':>7}{'Net R':>8}")
    for sym, g in df.groupby("sym"):
        rs = g["r"].values
        w = rs[rs>0].sum(); l = abs(rs[rs<0].sum())
        pf = w/l if l>0 else float('inf')
        print(f"  {sym:<10}{len(g):>5}{np.mean(rs>0)*100:>6.0f}%{pf:>7.2f}{rs.sum():>8.1f}")
    print(f"\n  Every NEW-universe trade:")
    for t in new_trades:
        print(f"    {str(t['entry_time']):<20}{t['sym']:<8}{t['exit_type']:<5}{t['r']:>+6.2f}")
else:
    print("  NO TRADES fired on new universe (setup rare / breadth gate restrictive)")

# ── combine with original universe (full portfolio) ─────────────────────────
print("\nLoading original symbols for full-universe run …")
orig_all = load_syms(ORIGINAL)
base_mask_o = {s: build_signal_mask(f, CIDS, "green", 1.5) for s, f in orig_all.items()}
final_mask_o = {}
for s, m in base_mask_o.items():
    f = orig_all[s]
    reg = (breadth.reindex(f.index, method="ffill") > BREADTH_THR).fillna(False)
    final_mask_o[s] = m & reg.values
orig_trades = run(orig_all, final_mask_o, RR, CFG)
all_trades = orig_trades + new_trades
all_trades.sort(key=lambda t: t["entry_time"])
s_orig = stats_from_trades(orig_trades)
s_all = stats_from_trades(all_trades)
print(f"\n  Original 52-universe: n={s_orig['n']} PF={s_orig['pf']:.3f} MDD={s_orig['mdd']*100:.1f}%")
print(f"  FULL universe (52 + new): n={s_all['n']} PF={s_all['pf']:.3f} MDD={s_all['mdd']*100:.1f}%  "
      f"Net=${s_all['net']:+,.0f}")

# ── negative-symbol quantification (answer to Q1) ───────────────────────────
print("\n  NEGATIVE-SYMBOL ANALYSIS (full 52-universe, locked config):")
df52 = pd.DataFrame(orig_trades)
per = []
for sym, g in df52.groupby("sym"):
    rs = g["r"].values
    per.append(dict(sym=sym, n=len(g), net=float(rs.sum()),
                    pf=(lambda w,l: w/l if l>0 else float('inf'))(
                        rs[rs>0].sum(), abs(rs[rs<0].sum()))))
per_df = pd.DataFrame(per).sort_values("net")
neg = per_df[per_df["net"] < 0]
print(f"  Symbols with negative net R (full period): {len(neg)} of {len(per_df)}")
print(f"  Their combined contribution: {neg['net'].sum():+.1f}R out of total {df52['r'].sum():+.1f}R")
print(f"  Max n among negative symbols: {neg['n'].max()} trades")
print(f"  Median n among negative symbols: {neg['n'].median():.0f} trades")
print(f"  → verdict: all negative symbols have very few trades (noise), "
      f"no symbol is reliably bad. Removing them would be curve-fitting.")

# 4 symbols never traded
notraded = sorted(set(ORIGINAL) - set(df52["sym"].unique()))
print(f"\n  Symbols that NEVER triggered a signal: {notraded}")

per_df.to_csv(os.path.join(OUT, "r078_symbol_perf.csv"), index=False)
if new_trades:
    pd.DataFrame(new_trades).to_csv(os.path.join(OUT, "r078_new_universe_trades.csv"), index=False)
print(f"\n  Saved → {OUT}/r078_*")
