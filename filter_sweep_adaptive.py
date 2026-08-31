"""
OTHER RULES: adaptive (regime-gated) VolCeil on 1H.

The 1H static hunt showed VolCeil=70 HELPS 2024/2025 but HURTS 2026 — opposite
regimes. Fix: apply VolCeil only inside a specific regime (gate it), instead of
always. Sweep the regime rule + threshold; validate on 2025 & 2026 (2024 needs
2023, noted). Goal: a rule that survives both testable years.

Cache 1H 2024-2026, 73 symbols. Fees 0.05%.
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
FAM_A = ["BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"]

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
print(f"[signals] RAW trades: {len(raw)}", flush=True)

# Regime gate: apply VolCeil(<=70) ONLY when regime-feature > thr (else no VolCeil)
REGIMES = {
    "adx_high":   lambda t, thr: t["adx14"] > thr,
    "vol_high":   lambda t, thr: t["real_vol_20"] > thr,
    "dist_high":  lambda t, thr: abs(t["ema_dist_pct"]) > thr,
    "rsi_extr":   lambda t, thr: (t["rsi14"] < thr) or (t["rsi14"] > 100 - thr),
}
THRESH = {"adx_high":[20,25,30,35,40], "vol_high":[1.0,1.5,2.0,2.5],
          "dist_high":[2.0,4.0,6.0,8.0], "rsi_extr":[20,25,30]}


def eval_rule(regime, thr, q=0.65):
    fr = [t for t in raw
          if (not regime(t, thr)) or (t["atr_rank_feat"] <= 70)]  # VolCeil only in-regime
    mf = build_mldf(fr, feats, breadth, breadth_pct)
    out = {}
    for Y in [2025, 2026]:   # 2024 untestable (no pre-2024 in cache)
        tr = mf[mf.ts < pd.Timestamp(f"{Y}-01-01", tz="UTC")]
        te = mf[mf.ts.dt.year == Y]
        if len(tr) < 50 or len(te) == 0:
            out[Y] = None; continue
        model = SVMQ75(q=q).fit_mldf(tr)
        kept, _ = model.keep_mldf(te, q)
        trd = [t for t in fr if t["entry_time"].year == Y and t["entry_time"] in kept]
        if not trd:
            out[Y] = None; continue
        s = stats_from_trades(list(trd)); pf_c = pf_of_rs(cost_adjusted_rs(list(trd), 0.05))
        out[Y] = (s["wr"], s["pf"], pf_c, len(trd))
    return out


print("\n" + "=" * 90)
print("ADAPTIVE RULE HUNT — regime-gated VolCeil (train all-prior -> test 2025/2026)")
print("=" * 90)
print(f"{'regime':>9} {'thr':>6} | {'2025 PF@c':>10} {'2026 PF@c':>10} | survive 25&26?")
survivors = []
for rname, fn in REGIMES.items():
    for thr in THRESH[rname]:
        res = eval_rule(fn, thr)
        def fmt(x): return f"{x[2]:.3f}(n{x[3]})" if x else "  -   "
        survive = all(res[Y] and res[Y][2] > 1.0 for Y in [2025, 2026])
        if survive:
            survivors.append((rname, thr, {Y: res[Y][2] for Y in [2025, 2026]}))
        print(f"{rname:>9} {str(thr):>6} | {fmt(res[2025]):>10} {fmt(res[2026]):>10} |{' YES' if survive else '  no'}")
print("\n=== ADAPTIVE SURVIVORS (PF@cost>1 in BOTH 2025 and 2026) ===")
if survivors:
    for (r, thr, pfc) in sorted(survivors, key=lambda x: -np.mean(list(x[2].values()))):
        print(f"  regime={r} thr={thr}  PF@c 25={pfc[2025]:.3f} 26={pfc[2026]:.3f}  avg={np.mean(list(pfc.values())):.3f}")
else:
    print("  (none survived both testable years)")
print("\n[done]")
