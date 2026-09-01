"""
PAPER-TRADING REHEARSAL of the champion via the deploy module (svm_deploy.SVMQ65Adaptive).
Uses cached 73-sym 1H data (2024-2026, to 2026-07-30). Walk-forward:
train on all prior years, test 2025 & 2026. Reports simulated trades, PF@cost,
max DD (1% risk), win%, and the PER-MONTH breakdown — a dry-run of what the bot
would have traded. No live orders.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
from ql_engine import add_features, build_signal_mask, sim_symbol, cost_adjusted_rs, pf_of_rs, stats_from_trades, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import SVMQ65Adaptive, build_mldf

CACHE = "quantlab_cache"
FAM_A = ["BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"]
print("[rehearsal] loading 73-sym 1H cache (2024-2026) …", flush=True)
syms = {f[:-len("_1H.parquet")] for f in os.listdir(CACHE) if f.endswith("_1H.parquet")}
feats, above20 = {}, {}
for sym in sorted(syms):
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
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100
mask = {s: build_signal_mask(f, FAM_A, "green", 1.5) for s, f in feats.items()}
raw = []
for s, f in feats.items():
    for t in sim_symbol(f, mask[s], 1.5, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = s; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
mldf = build_mldf(raw, feats, breadth, breadth_pct)
print(f"[rehearsal] {len(feats)} symbols, {len(raw)} raw 1H signals, mldf={len(mldf)}", flush=True)

# Walk-forward champion (deploy module)
champion = []
for Y in [2025, 2026]:
    tr = mldf[mldf.ts < pd.Timestamp(f"{Y}-01-01", tz="UTC")]
    te = mldf[mldf.ts.dt.year == Y]
    if len(tr) < 50 or len(te) == 0:
        continue
    model = SVMQ65Adaptive().fit_mldf(tr)
    kept, _ = model.keep_mldf(te)
    for t in raw:
        if t["entry_time"].year == Y and t["entry_time"] in kept:
            champion.append(t)
print(f"[rehearsal] champion trades taken 2025-2026: {len(champion)}", flush=True)

adj = cost_adjusted_rs(list(champion), 0.05)
for t, a in zip(champion, adj):
    t["adj_r"] = a


def equity_mdd(trades, risk=0.01):
    eq = 1.0; peak = 1.0; mdd = 0.0
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        eq *= (1 + risk * t["adj_r"]); peak = max(peak, eq); mdd = min(mdd, eq/peak - 1)
    return eq - 1, mdd


print("\n" + "=" * 78)
print("PAPER-TRADING REHEARSAL — SVMQ65Adaptive (deploy module), 73-sym 1H")
print("=" * 78)
for Y in [2025, 2026]:
    yt = [t for t in champion if t["entry_time"].year == Y]
    ret, mdd = equity_mdd(yt)
    s = stats_from_trades(list(yt))
    print(f"\n[{Y}] trades={len(yt)} win={s['wr']:.0%} PF@cost={pf_of_rs([t['adj_r'] for t in yt]):.3f} "
          f"return(1% risk)={ret:+.1%} MAX DD={mdd:.1%}")
    msum = defaultdict(list)
    for t in yt:
        msum[(t["entry_time"].year, t["entry_time"].month)].append(t["adj_r"])
    print("   month      n   win%   netR   prof?")
    for (yy, mm) in sorted(msum):
        rs = msum[(yy, mm)]; nr = sum(rs); w = sum(1 for r in rs if r > 0)/len(rs)
        print(f"   {yy}-{mm:02d}   {len(rs):>4} {w*100:>5.0f}% {nr:>+7.2f}  {'OK' if nr>0 else 'LOSS'}")
ret, mdd = equity_mdd(champion)
s = stats_from_trades(list(champion))
print(f"\n[FULL 2025-2026] trades={len(champion)} win={s['wr']:.0%} PF@cost={pf_of_rs([t['adj_r'] for t in champion]):.3f} "
      f"return(1% risk)={ret:+.1%} MAX DD={mdd:.1%}")
print("\n[done]")
