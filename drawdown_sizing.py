"""
DRAWDOWN SIZING — find deploy-safe universe size + risk% so max DD is bounded.
Champion = SVMQ65Adaptive, walk-forward 2025/2026 (73-sym cache). Tests:
  universe: full(73) vs sub(30)   x   risk/trade: 1.0% vs 0.5%.
Reports PF@cost, return, and MAX DD for each combo. (2024 omitted; no 2023 here.)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from ql_engine import add_features, build_signal_mask, sim_symbol, cost_adjusted_rs, pf_of_rs, stats_from_trades, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import SVMQ65Adaptive, build_mldf

CACHE = "quantlab_cache"
FAM_A = ["BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"]

print("[load] 73-sym 1H cache …", flush=True)
syms = sorted({f[:-len("_1H.parquet")] for f in os.listdir(CACHE) if f.endswith("_1H.parquet")})
feats, above20 = {}, {}
for sym in syms:
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for c in ["open","high","low","close","vol"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["open","high","low","close","vol"], inplace=True)
    if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
    f = add_features(df)
    f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
    if len(f) >= IS_LOOKBACK + RECAL_EVERY:
        feats[sym] = f; above20[sym] = (f["close"] > f["ema20"]).astype(float)
print(f"[load] symbols: {len(feats)}", flush=True)

syms = sorted(feats.keys())        # only symbols that built features
SUBSET30 = syms[:30]  # representative 30-sym slice

def champion_trades(symbols):
    ab = {s: above20[s] for s in symbols}
    breadth = pd.DataFrame(ab).sort_index().mean(axis=1, skipna=True)
    breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100
    mask = {s: build_signal_mask(feats[s], FAM_A, "green", 1.5) for s in symbols}
    raw = []
    for s in symbols:
        for t in sim_symbol(feats[s], mask[s], 1.5, dict(entry_next=False, exit="base", hours=None)):
            t["sym"] = s; raw.append(t)
    raw.sort(key=lambda t: t["entry_time"])
    mldf = build_mldf(raw, {s: feats[s] for s in symbols}, breadth, breadth_pct)
    champ = []
    for Y in [2025, 2026]:
        tr = mldf[mldf.ts < pd.Timestamp(f"{Y}-01-01", tz="UTC")]
        te = mldf[mldf.ts.dt.year == Y]
        if len(tr) < 50 or len(te) == 0: continue
        model = SVMQ65Adaptive().fit_mldf(tr)
        kept, _ = model.keep_mldf(te)
        for t in raw:
            if t["entry_time"].year == Y and t["entry_time"] in kept:
                champ.append(t)
    adj = cost_adjusted_rs(list(champ), 0.05)
    for t, a in zip(champ, adj):
        t["adj_r"] = a
    return champ

def equity_mdd(trades, risk):
    eq = 1.0; peak = 1.0; mdd = 0.0
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        eq *= (1 + risk * t["adj_r"]); peak = max(peak, eq); mdd = min(mdd, eq/peak - 1)
    return eq - 1, mdd

print("\n" + "=" * 80)
print("DRAWDOWN SIZING — champion PF@cost, return, MAX DD by universe x risk")
print("=" * 80)
print(f"{'universe':>10}{'risk':>7}{'trades':>8}{'PF@c':>8}{'return':>10}{'MAX DD':>9}")
for name, symbols in [("FULL(73)", syms), ("SUB(30)", SUBSET30)]:
    champ = champion_trades(symbols)
    if not champ:
        continue
    pf = pf_of_rs([t["adj_r"] for t in champ])
    for risk in [0.01, 0.005]:
        ret, mdd = equity_mdd(champ, risk)
        print(f"{name:>10}{risk*100:>6.1f}%{len(champ):>8}{pf:>8.3f}{ret:>+9.1%}{mdd:>8.1%}")
print("\n[done]")
