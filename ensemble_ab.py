"""
ENSEMBLE: Family A OR Family B fires -> candidate. SVM q0.65 + adaptive VolCeil
(|ema_dist_pct|>2.0). Walk-forward 2025/2026 on 73-sym 1H cache.
Reports per-year PF@cost + the MONTH-BY-MONTH breakdown (n, win%, net R, profitable).
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from ql_engine import add_features, build_signal_mask, sim_symbol, stats_from_trades, cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import FEATS, build_mldf, SVMQ75

CACHE = "quantlab_cache"
FAM_A = (["BBW_STRICT","RV_LO","DST_NR","PRG_VH"], "green")
FAM_B = (["BBW_STRICT","DST_NR","PRG_VH"], "green")  # drop RV_LO


def load_feats():
    syms = {f[:-len("_1H.parquet")] for f in os.listdir(CACHE) if f.endswith("_1H.parquet")}
    feats, above20 = {}, {}
    for sym in sorted(syms):
        df = pd.read_parquet(os.path.join(CACHE, f"{sym}_1H.parquet"))
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for c in ["open","high","low","close","vol"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f; above20[sym] = (f["close"] > f["ema20"]).astype(float)
    return feats, above20


print("[load] 73-sym 1H cache …", flush=True)
feats, above20 = load_feats()
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100
print(f"[load] symbols: {len(feats)}", flush=True)

# raw candidates from A and B, deduped
raw = []
seen = set()
for name, (conds, gate) in [("A", FAM_A), ("B", FAM_B)]:
    mask = {s: build_signal_mask(f, conds, gate, 1.5) for s, f in feats.items()}
    for s, f in feats.items():
        for t in sim_symbol(f, mask[s], 1.5, dict(entry_next=False, exit="base", hours=None)):
            t["sym"] = s; k = (s, t["entry_time"])
            if k in seen: continue
            seen.add(k); raw.append(t)
print(f"[signals] A∪B raw trades 2024-2026: {len(raw)}", flush=True)

bfrac = breadth.reindex(pd.DatetimeIndex([t["entry_time"] for t in raw])).values
for t, bf in zip(raw, bfrac):
    t["breadth_frac"] = bf if pd.notna(bf) else 0.5
mldf = build_mldf(raw, feats, breadth, breadth_pct)
atr_map = dict(zip(mldf["ts"], mldf["atr_rank"]))
fmap = {c: dict(zip(mldf["ts"], mldf[c])) for c in ["ema_dist_pct"]}
for t in raw:
    t["atr_rank_feat"] = atr_map.get(t["entry_time"], 50.0)
    t["ema_dist_pct"] = fmap["ema_dist_pct"].get(t["entry_time"], 0.0)

# champion filter: adaptive VolCeil
fr = [t for t in raw if (abs(t["ema_dist_pct"]) <= 2.0) or (t["atr_rank_feat"] <= 70)]
mf = build_mldf(fr, feats, breadth, breadth_pct)
champion = []
for Y in [2025, 2026]:
    tr = mf[mf.ts < pd.Timestamp(f"{Y}-01-01", tz="UTC")]
    te = mf[mf.ts.dt.year == Y]
    if len(tr) < 50 or len(te) == 0:
        continue
    model = SVMQ75(q=0.65).fit_mldf(tr)
    kept, _ = model.keep_mldf(te, 0.65)
    kept = set(kept)
    for t in fr:
        if t["entry_time"].year == Y and t["entry_time"] in kept:
            champion.append(t)
print(f"[ensemble] champion trades 2025-2026: {len(champion)}", flush=True)

adj = cost_adjusted_rs(list(champion), 0.05)
for t, a in zip(champion, adj):
    t["adj_r"] = a


def equity_mdd(trades, risk=0.01):
    eq = 1.0; peak = 1.0; mdd = 0.0
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        eq *= (1 + risk * t["adj_r"]); peak = max(peak, eq); mdd = min(mdd, eq/peak - 1)
    return eq - 1, mdd


def month_table(trades, year):
    yt = [t for t in trades if t["entry_time"].year == year]
    if not yt:
        print(f"  [{year}] no trades"); return
    ret, mdd = equity_mdd(yt)
    s = stats_from_trades(list(yt))
    print(f"\n[{year}] n={len(yt)} win={s['wr']:.0%} PF@cost={pf_of_rs([t['adj_r'] for t in yt]):.3f} "
          f"ret(1% risk)={ret:+.1%} MDD={mdd:.1%}")
    # per month
    msum = defaultdict(list)
    for t in yt:
        msum[(t["entry_time"].year, t["entry_time"].month)].append(t["adj_r"])
    print(f"  {'month':>8} {'n':>4} {'win%':>6} {'netR':>8} {'prof?':>6}")
    for (yy, mm) in sorted(msum):
        rs = msum[(yy, mm)]; nr = sum(rs); w = sum(1 for r in rs if r > 0)/len(rs)
        flag = "OK" if nr > 0 else "LOSS"
        print(f"  {yy}-{mm:02d}   {len(rs):>4} {w*100:>5.0f}% {nr:>+8.2f} {flag:>6}")


print("\n" + "=" * 78)
print("ENSEMBLE A∪B — SVM q0.65 + adaptive VolCeil (73-sym, 2025/2026)")
print("=" * 78)
for Y in [2025, 2026]:
    month_table(champion, Y)
# full
ret, mdd = equity_mdd(champion)
print(f"\n[FULL 2025-2026] n={len(champion)} ret(1% risk)={ret:+.1%} MDD={mdd:.1%} "
      f"profitable-months across period shown above")
print("\n[done]")
