"""
CHAMPION DEEP DIVE — risk & robustness metrics the PF@cost number hides.
Config: SVM q0.65 + VolCeil gated by |ema_dist_pct|>2.0, walk-forward 2024/25/26.
Reports: per-year MAX DRAWDOWN (under 1% risk/trade), PROFITABLE MONTHS per year,
and PER-ASSET contribution (is it broad or carried by a few?).
Data: persisted 2023-2026 (30 symbols). Fees 0.05%.
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

SAVE_DIR = "quantlab_cache_2023"
FAM_A = ["BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"]

print("[load] persisted 2023-2026 series …", flush=True)
feats, above20 = {}, {}
for f in sorted(os.listdir(SAVE_DIR)):
    if not f.endswith("_1H_full.parquet"):
        continue
    try:
        df = pd.read_parquet(os.path.join(SAVE_DIR, f))
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for c in ["open","high","low","close","vol"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        sym = f.replace("_1H_full.parquet", "")
        ff = add_features(df)
        ff.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(ff) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = ff; above20[sym] = (ff["close"] > ff["ema20"]).astype(float)
    except Exception as e:
        print(f"  err {sym}: {e}", flush=True)
print(f"[load] symbols: {len(feats)}", flush=True)

breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100
mask = {s: build_signal_mask(f, FAM_A, "green", 1.5) for s, f in feats.items()}
raw = []
for s, f in feats.items():
    for t in sim_symbol(f, mask[s], 1.5, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = s; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
bfrac = breadth.reindex(pd.DatetimeIndex([t["entry_time"] for t in raw])).values
for t, bf in zip(raw, bfrac):
    t["breadth_frac"] = bf if pd.notna(bf) else 0.5
mldf_full = build_mldf(raw, feats, breadth, breadth_pct)
atr_map = dict(zip(mldf_full["ts"], mldf_full["atr_rank"]))
feat_map = {c: dict(zip(mldf_full["ts"], mldf_full[c])) for c in ["adx14","breadth_q","real_vol_20","ema_dist_pct","rsi14"]}
for t in raw:
    t["atr_rank_feat"] = atr_map.get(t["entry_time"], 50.0)
    for c in feat_map:
        t[c] = feat_map[c].get(t["entry_time"], 0.0)
print(f"[signals] RAW trades 2023-2026: {len(raw)}", flush=True)

# Champion filter: adaptive VolCeil gated by |ema_dist_pct| > 2.0
REGIME = lambda t: abs(t["ema_dist_pct"]) > 2.0
fr = [t for t in raw if (not REGIME(t)) or (t["atr_rank_feat"] <= 70)]
mf = build_mldf(fr, feats, breadth, breadth_pct)
champion = []
for Y in [2024, 2025, 2026]:
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
print(f"[champion] trades taken 2024-2026: {len(champion)}", flush=True)

adj = cost_adjusted_rs(list(champion), 0.05)
for t, a in zip(champion, adj):
    t["adj_r"] = a


def equity_mdd(trades, risk=0.01):
    eq = 1.0; peak = 1.0; mdd = 0.0
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        eq *= (1 + risk * t["adj_r"])
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    return eq - 1, mdd


print("\n" + "=" * 78)
print("CHAMPION DEEP DIVE — SVM q0.65 + VolCeil gated |ema_dist_pct|>2.0")
print("=" * 78)
for Y in [2024, 2025, 2026]:
    yt = [t for t in champion if t["entry_time"].year == Y]
    ret, mdd = equity_mdd(yt)
    s = stats_from_trades(list(yt))
    # profitable months
    msum = defaultdict(float)
    for t in yt:
        msum[(t["entry_time"].year, t["entry_time"].month)] += t["adj_r"]
    pos = sum(1 for v in msum.values() if v > 0)
    print(f"\n[{Y}] trades={len(yt)}  win={s['wr']:.0%}  PF@cost={pf_of_rs([t['adj_r'] for t in yt]):.3f}")
    print(f"      return(1% risk)={ret:+.1%}  MAX DRAWDOWN={mdd:.1%}  profitable-months={pos}/{len(msum)}")
# full 2024-2026
ret, mdd = equity_mdd(champion)
s = stats_from_trades(list(champion))
msum = defaultdict(float)
for t in champion:
    msum[(t["entry_time"].year, t["entry_time"].month)] += t["adj_r"]
pos = sum(1 for v in msum.values() if v > 0)
print(f"\n[FULL 2024-2026] trades={len(champion)}  return(1% risk)={ret:+.1%}  MAX DRAWDOWN={mdd:.1%}  profitable-months={pos}/{len(msum)}")

# Per-asset contribution
print("\n" + "-" * 78)
print("PER-ASSET (is it broad or carried by a few?)")
print("-" * 78)
agg = defaultdict(lambda: {"n":0,"r":0.0,"w":0})
for t in champion:
    a = agg[t["sym"]]; a["n"] += 1; a["r"] += t["adj_r"]; a["w"] += (1 if t["adj_r"] > 0 else 0)
rows = []
for sym, a in agg.items():
    rows.append((sym, a["n"], a["w"]/a["n"], a["r"]))
rows.sort(key=lambda x: -x[3])
total_r = sum(r[3] for r in rows)
pos_syms = sum(1 for r in rows if r[3] > 0)
print(f"  {len(rows)} symbols traded | {pos_syms} profitable | total R={total_r:.1f}")
print(f"  {'symbol':<22}{'n':>4}{'win%':>7}{'totR':>9}{'share':>8}")
for sym, n, w, r in rows:
    print(f"  {sym:<22}{n:>4}{w*100:>6.0f}%{r:>9.2f}{r/total_r*100:>7.0f}%")
top3 = sum(r[3] for r in rows[:3])
print(f"  -> top-3 symbols carry {top3/total_r*100:.0f}% of total R" if total_r > 0 else "")
print("\n[done]")
