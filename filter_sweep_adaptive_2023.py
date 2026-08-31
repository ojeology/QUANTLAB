"""
CONFIRM ADAPTIVE RULE across 2024 too — fetch 2023 so 2024 is testable.
Sweeps regime-gated VolCeil (apply VolCeil only inside a regime) with per-year
walk-forward (train all-prior -> test 2024/25/26). Flags configs surviving ALL
three years (PF@cost>1 each). Subset universe (2023 fetch feasible). Fees 0.05%.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from ql_engine import add_features, build_signal_mask, sim_symbol, stats_from_trades, cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY
import demo_bot as bot
from svm_deploy import FEATS, build_mldf, SVMQ75

CACHE = "quantlab_cache"
FAM_A = ["BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"]
SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP"]


def fetch_before(end_ts_ms, n_bars, inst, bar="1H", page_limit=200):
    all_rows, after, pages = [], end_ts_ms, 0
    while len(all_rows) < n_bars and pages < page_limit:
        params = {"instId": inst, "bar": bar, "limit": bot.PAGE_LIMIT, "after": str(after)}
        raw = bot._get(bot.OKX_CANDLES, params) or bot._get(bot.OKX_CANDLES_CUR, params)
        if not raw:
            break
        all_rows.extend(raw); pages += 1; oldest = int(raw[-1][0]); after = oldest
        if len(all_rows) >= n_bars:
            break
        time.sleep(bot.PAGE_DELAY)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=bot.CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return (df[["datetime","open","high","low","close","vol"]].sort_values("datetime")
            .drop_duplicates("datetime").reset_index(drop=True).set_index("datetime"))


print("[load] fetch 2023 + cache 2024-2026 for subset …", flush=True)
feats, above20 = {}, {}
for sym in SUBSET:
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    if not os.path.exists(p):
        continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for c in ["open","high","low","close","vol"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        inst = sym.replace("_","-")
        end_ms = int(df.index[0].timestamp()*1000) - 1
        f2023 = fetch_before(end_ms, 9200, inst)
        if f2023 is not None and len(f2023):
            df = pd.concat([f2023, df]); df = df[~df.index.duplicated(keep="last")].sort_index()
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f; above20[sym] = (f["close"] > f["ema20"]).astype(float)
            print(f"  loaded {sym}: {len(f)} bars ({f.index[0].date()} -> {f.index[-1].date()})", flush=True)
    except Exception as e:
        print(f"  err {sym}: {e}", flush=True)
print(f"[load] subset: {len(feats)}", flush=True)

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

REGIMES = {
    "adx_high":  lambda t, thr: t["adx14"] > thr,
    "vol_high":  lambda t, thr: t["real_vol_20"] > thr,
    "dist_high": lambda t, thr: abs(t["ema_dist_pct"]) > thr,
    "rsi_extr":  lambda t, thr: (t["rsi14"] < thr) or (t["rsi14"] > 100 - thr),
}
THRESH = {"adx_high":[20,25,30,35,40], "vol_high":[1.0,1.5,2.0,2.5],
          "dist_high":[2.0,4.0,6.0,8.0], "rsi_extr":[20,25,30]}


def eval_rule(regime, thr, q=0.65):
    # apply VolCeil(<=70) ONLY inside the regime; otherwise no VolCeil
    fr = [t for t in raw if (not regime(t, thr)) or (t["atr_rank_feat"] <= 70)]
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
        s = stats_from_trades(list(trd)); pf_c = pf_of_rs(cost_adjusted_rs(list(trd), 0.05))
        out[Y] = (s["wr"], s["pf"], pf_c, len(trd))
    return out


print("\n" + "=" * 96)
print("ADAPTIVE RULE HUNT incl 2023 (train 2023 -> test 2024/25/26)  PF@cost per year")
print("=" * 96)
print(f"{'regime':>9} {'thr':>6} | {'2024':>9} {'2025':>9} {'2026':>9} | survive all?")
survivors = []
for rname, fn in REGIMES.items():
    for thr in THRESH[rname]:
        res = eval_rule(fn, thr)
        def fmt(x): return f"{x[2]:.3f}(n{x[3]})" if x else "  -   "
        pfc = {Y: (res[Y][2] if res[Y] else 0.0) for Y in [2024, 2025, 2026]}
        survive = all(res[Y] and res[Y][2] > 1.0 for Y in [2024, 2025, 2026])
        if survive:
            survivors.append((rname, thr, pfc))
        print(f"{rname:>9} {str(thr):>6} | {fmt(res[2024]):>9} {fmt(res[2025]):>9} {fmt(res[2026]):>9} |{' YES' if survive else '  no'}")
print("\n=== ADAPTIVE SURVIVORS (PF@cost > 1 in 2024 AND 2025 AND 2026) ===")
if survivors:
    for (r, thr, pfc) in sorted(survivors, key=lambda x: -np.mean(list(x[2].values()))):
        print(f"  regime={r} thr={thr}  PF@c 24={pfc[2024]:.3f} 25={pfc[2025]:.3f} 26={pfc[2026]:.3f}  avg={np.mean(list(pfc.values())):.3f}")
else:
    print("  (none survived all three years)")
print("\n[done]")
