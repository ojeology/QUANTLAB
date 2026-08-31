"""
OTHER ENVIRONMENT: 4H timeframe — same SVM sweet-spot hunt, different regime.

Fetches 4H bars (2023-2026) for the liquid subset (no 4H cache exists), then
sweeps q x VolCeil x breadth50 with per-year walk-forward (train 2023 -> test
2024/25/26). Tests whether the edge is more/less year-robust on 4H than 1H.
Fees 0.05%.
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
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP",
          "FIL_USDT_SWAP","INJ_USDT_SWAP","AXS_USDT_SWAP","SAND_USDT_SWAP","FET_USDT_SWAP",
          "GRT_USDT_SWAP","HBAR_USDT_SWAP","IMX_USDT_SWAP","COMP_USDT_SWAP","AAVE_USDT_SWAP"]


def fetch_bars(end_ts_ms, n_bars, inst, bar, page_limit=200):
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


print("[load] fetch 4H 2023-2026 for subset …", flush=True)
end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) - 1
feats, above20 = {}, {}
for sym in SUBSET:
    try:
        inst = sym.replace("_", "-")
        df = fetch_bars(end_ms, 8200, inst, "4H")
        if df is None or len(df) < IS_LOOKBACK + RECAL_EVERY + 100:
            continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f; above20[sym] = (f["close"] > f["ema20"]).astype(float)
        print(f"  loaded {sym}: {len(f)} 4H bars ({f.index[0].date()} -> {f.index[-1].date()})", flush=True)
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
for t in raw:
    t["atr_rank_feat"] = atr_map.get(t["entry_time"], 50.0)
print(f"[signals] RAW 4H trades 2023-2026: {len(raw)}", flush=True)


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
        s = stats_from_trades(list(trd)); pf_c = pf_of_rs(cost_adjusted_rs(list(trd), 0.05))
        out[Y] = (s["wr"], s["pf"], pf_c, len(trd))
    return out


configs = [(q, vc, bo) for q in [0.55, 0.65, 0.75, 0.85]
           for vc in [None, 70] for bo in [False, True]]
print("\n" + "=" * 96)
print("4H ENVIRONMENT — SWEET-SPOT HUNT (train 2023 -> test 2024/25/26)  PF@cost per year")
print("=" * 96)
print(f"{'q':>5} {'VolC':>5} {'brd':>4} | {'2024':>9} {'2025':>9} {'2026':>9} | survive?")
survivors = []
for (q, vc, bo) in configs:
    res = eval_cfg(q, vc, bo)
    def fmt(r): return f"{r[2]:.3f}(n{r[3]})" if r else "  -   "
    pfc = {Y: (res[Y][2] if res[Y] else 0.0) for Y in [2024, 2025, 2026]}
    survive = all(res[Y] and res[Y][2] > 1.0 for Y in [2024, 2025, 2026])
    if survive:
        survivors.append((q, vc, bo, pfc))
    print(f"{q:>5} {str(vc):>5} {str(bo):>4} | {fmt(res[2024]):>9} {fmt(res[2025]):>9} {fmt(res[2026]):>9} |{' YES' if survive else '  no'}")
print("\n=== 4H SURVIVORS (PF@cost > 1 in 2024 AND 2025 AND 2026) ===")
if survivors:
    for (q, vc, bo, pfc) in sorted(survivors, key=lambda x: -np.mean(list(x[3].values()))):
        print(f"  q={q} VolCeil={vc} breadth50={bo}  PF@c 24={pfc[2024]:.3f} 25={pfc[2025]:.3f} 26={pfc[2026]:.3f}  avg={np.mean(list(pfc.values())):.3f}")
else:
    print("  (none survived all three years)")
print("\n[done]")
