"""
FILTER SWEET-SPOT HUNT — find the config that SURVIVES different years.

Sweep over:
  q          (SVM keep-rate)      in {0.55, 0.65, 0.75, 0.85}
  VolCeil     (atr_rank ceiling) in {None, 70}
  breadth50   (gate)              in {off, on}
For each config: walk-forward (train all prior years -> test 2024, 2025, 2026
standalone), report PF@cost per year. Flag configs with PF@cost > 1 in ALL years.

Data: cache-only 1H (2024-2026), 73 symbols. Fees 0.05%.
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

print("[load] 73-symbol 1H cache (2024-2026) …", flush=True)
syms = {f[:-len("_1H.parquet")] for f in os.listdir(CACHE) if f.endswith("_1H.parquet")}
feats, above20 = {}, {}
for sym in sorted(syms):
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for c in ["open", "high", "low", "close", "vol"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["open", "high", "low", "close", "vol"], inplace=True)
    if len(df) < IS_LOOKBACK + RECAL_EVERY + 100:
        continue
    f = add_features(df)
    f.dropna(subset=["ema200", "atr14", "adx14", "ema_dist_pct", "real_vol_20",
                     "bb_width", "prev_range_r", "prev_body_r"], inplace=True)
    if len(f) >= IS_LOOKBACK + RECAL_EVERY:
        feats[sym] = f
        above20[sym] = (f["close"] > f["ema20"]).astype(float)
print(f"[load] symbols with features: {len(feats)}", flush=True)

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
for t in raw:
    t["atr_rank_feat"] = atr_map.get(t["entry_time"], 50.0)
print(f"[signals] RAW trades: {len(raw)}", flush=True)


def eval_cfg(q, volceil, breadth_on):
    fr = [t for t in raw
          if (volceil is None or t["atr_rank_feat"] <= volceil)
          and (not breadth_on or t["breadth_frac"] > 0.5)]
    mf = build_mldf(fr, feats, breadth, breadth_pct)
    out = {}
    for Y in [2024, 2025, 2026]:
        tr = mf[mf.ts < pd.Timestamp(f"{Y}-01-01", tz="UTC")]
        te = mf[mf.ts.dt.year == Y]
        if len(tr) < 50 or len(te) == 0:
            out[Y] = None; continue
        model = SVMQ75(q=q).fit_mldf(tr)
        kept, _ = model.keep_mldf(te, q)
        trd = [t for t in fr if t["entry_time"].year == Y and t["entry_time"] in kept]
        if not trd:
            out[Y] = None; continue
        s = stats_from_trades(list(trd))
        pf_c = pf_of_rs(cost_adjusted_rs(list(trd), 0.05))
        out[Y] = (s["wr"], s["pf"], pf_c, len(trd))
    return out


configs = [(q, vc, bo) for q in [0.55, 0.65, 0.75, 0.85]
           for vc in [None, 70] for bo in [False, True]]
print("\n" + "=" * 96)
print("FILTER SWEET-SPOT HUNT — per-year walk-forward (PF@cost per year; * = survives ALL years)")
print("=" * 96)
print(f"{'q':>5} {'VolCeil':>7} {'brd50':>5} | {'2024 PF@c':>9} {'2025 PF@c':>9} {'2026 PF@c':>9} | survive?")
survivors = []
for (q, vc, bo) in configs:
    res = eval_cfg(q, vc, bo)
    def fmt(r):
        return f"{r[2]:.3f}(n{r[3]})" if r else "  -   "
    pfc = {Y: (res[Y][2] if res[Y] else 0.0) for Y in [2024, 2025, 2026]}
    survive = all(res[Y] and res[Y][2] > 1.0 for Y in [2024, 2025, 2026])
    if survive:
        survivors.append((q, vc, bo, pfc))
    flag = "  *" if survive else ""
    print(f"{q:>5} {str(vc):>7} {str(bo):>5} | {fmt(res[2024]):>9} {fmt(res[2025]):>9} {fmt(res[2026]):>9} |{' YES' if survive else '  no'}{flag}")

print("\n=== SURVIVORS (PF@cost > 1 in EVERY year 2024-2026) ===")
if survivors:
    for (q, vc, bo, pfc) in sorted(survivors, key=lambda x: -np.mean(list(x[3].values()))):
        print(f"  q={q} VolCeil={vc} breadth50={bo}  PF@c 2024={pfc[2024]:.3f} 2025={pfc[2025]:.3f} 2026={pfc[2026]:.3f}  avg={np.mean(list(pfc.values())):.3f}")
else:
    print("  (none — no single config beats PF@cost 1.0 in all three years)")
print("\n[done]")
