"""
FRESH 1H HYPOTHESIS: is the persistent SVM edge robust to the EXACT signal
definition, or unique to Family A's 4 conditions? Test alternative combinations of
the 5 valid primitives (BBW_STRICT, RV_LO, DST_NR, PRG_VH, ADX_ST) on 1H.
If several definitions each carry a year-robust SVM edge -> the edge is real and
an ENSEMBLE of signal definitions is more robust than Family A alone.

Uses reliable 73-sym 1H cache (2024-2026). Testable years 2025/2026 (2024 needs
2023, unavailable here). Champion pipeline: SVM q0.65 + VolCeil gated |ema_dist_pct|>2.0.
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from ql_engine import add_features, build_signal_mask, sim_symbol, stats_from_trades, cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY
from svm_deploy import FEATS, build_mldf, SVMQ75

CACHE = "quantlab_cache"
FAMILIES = {
    "A  [BBW,RV,DST,PRG]green":  (["BBW_STRICT","RV_LO","DST_NR","PRG_VH"], "green"),
    "B  [BBW,DST,PRG]green":     (["BBW_STRICT","DST_NR","PRG_VH"], "green"),
    "C  [BBW,RV,PRG]green":      (["BBW_STRICT","RV_LO","PRG_VH"], "green"),
    "D  [BBW,RV,DST,PRG,ADX]grn":(["BBW_STRICT","RV_LO","DST_NR","PRG_VH","ADX_ST"], "green"),
    "E  [BBW,PRG]green":         (["BBW_STRICT","PRG_VH"], "green"),
    "F  [BBW,RV,DST,PRG]red":    (["BBW_STRICT","RV_LO","DST_NR","PRG_VH"], "red"),
}

print("[load] 73-sym 1H cache (2024-2026) …", flush=True)
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
print(f"[load] symbols: {len(feats)}", flush=True)


def run_family(name, conds, gate):
    mask = {s: build_signal_mask(f, conds, gate, 1.5) for s, f in feats.items()}
    raw = []
    for s, f in feats.items():
        for t in sim_symbol(f, mask[s], 1.5, dict(entry_next=False, exit="base", hours=None)):
            t["sym"] = s; raw.append(t)
    if len(raw) < 100:
        return None
    bfrac = breadth.reindex(pd.DatetimeIndex([t["entry_time"] for t in raw])).values
    for t, bf in zip(raw, bfrac):
        t["breadth_frac"] = bf if pd.notna(bf) else 0.5
    mldf = build_mldf(raw, feats, breadth, breadth_pct)
    atr_map = dict(zip(mldf["ts"], mldf["atr_rank"]))
    fmap = {c: dict(zip(mldf["ts"], mldf[c])) for c in ["ema_dist_pct"]}
    for t in raw:
        t["atr_rank_feat"] = atr_map.get(t["entry_time"], 50.0)
        t["ema_dist_pct"] = fmap["ema_dist_pct"].get(t["entry_time"], 0.0)
    fr = [t for t in raw if (abs(t["ema_dist_pct"]) <= 2.0) or (t["atr_rank_feat"] <= 70)]
    mf = build_mldf(fr, feats, breadth, breadth_pct)
    out = {}
    for Y in [2025, 2026]:
        tr = mf[mf.ts < pd.Timestamp(f"{Y}-01-01", tz="UTC")]
        te = mf[mf.ts.dt.year == Y]
        if len(tr) < 50 or len(te) == 0:
            out[Y] = None; continue
        model = SVMQ75(q=0.65).fit_mldf(tr)
        kept, _ = model.keep_mldf(te, 0.65)
        trd = [t for t in fr if t["entry_time"].year == Y and t["entry_time"] in kept]
        if not trd:
            out[Y] = None; continue
        s = stats_from_trades(list(trd)); pf_c = pf_of_rs(cost_adjusted_rs(list(trd), 0.05))
        out[Y] = (s["wr"], pf_c, len(trd))
    return out


print("\n" + "=" * 80)
print("FRESH HYPOTHESIS: multiple 1H signal definitions — which carry a persistent edge?")
print("(testable years 2025/2026; 2024 needs 2023)")
print("=" * 80)
print(f"{'family':<26}{'2025 PF@c':>11}{'2026 PF@c':>11}  both>1?")
for name, (conds, gate) in FAMILIES.items():
    try:
        res = run_family(name, conds, gate)
        if res is None:
            print(f"{name:<26}  (too few signals)"); continue
        def fmt(x): return f"{x[1]:.3f}(n{x[2]})" if x else "  -   "
        both = all(res[Y] and res[Y][1] > 1.0 for Y in [2025, 2026])
        print(f"{name:<26}{fmt(res[2025]):>11}{fmt(res[2026]):>11}  {'YES' if both else ' no'}")
    except Exception as e:
        print(f"{name:<26}  ERR {e}")
print("\n[done]")
