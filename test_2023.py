"""
EXTEND THE BLIND TEST BACK TO 2023.

Cache only goes back to Jan 2024, so 2023 is fetched live from OKX for a
liquid ~28-symbol subset (keeps breadth consistent, keeps the fetch feasible).
Then an expanding-window walk-forward on a CONSISTENT universe:
    train 2023            -> test 2024
    train 2023+2024       -> test 2025
    train 2023+2024+2025  -> test 2026
Each test year is standalone OOS (model never saw it). Fees 0.05%.
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
SUBSET = ["BTC_USDT_SWAP", "ETH_USDT_SWAP", "SOL_USDT_SWAP", "BNB_USDT_SWAP", "XRP_USDT_SWAP",
          "DOGE_USDT_SWAP", "ADA_USDT_SWAP", "AVAX_USDT_SWAP", "LINK_USDT_SWAP", "DOT_USDT_SWAP",
          "LTC_USDT_SWAP", "TRX_USDT_SWAP", "ATOM_USDT_SWAP", "NEAR_USDT_SWAP", "APT_USDT_SWAP",
          "ARB_USDT_SWAP", "OP_USDT_SWAP", "SUI_USDT_SWAP", "ETC_USDT_SWAP", "XLM_USDT_SWAP",
          "FIL_USDT_SWAP", "INJ_USDT_SWAP", "AXS_USDT_SWAP", "SAND_USDT_SWAP", "FET_USDT_SWAP",
          "GRT_USDT_SWAP", "HBAR_USDT_SWAP", "IMX_USDT_SWAP", "COMP_USDT_SWAP", "AAVE_USDT_SWAP"]


def fetch_before(end_ts_ms, n_bars, inst, bar="1H", page_limit=200):
    """Fetch n_bars of history ENDING just before end_ts_ms (i.e. oldest data)."""
    all_rows, after, pages = [], end_ts_ms, 0
    while len(all_rows) < n_bars and pages < page_limit:
        params = {"instId": inst, "bar": bar, "limit": bot.PAGE_LIMIT, "after": str(after)}
        raw = bot._get(bot.OKX_CANDLES, params)
        if not raw:
            raw = bot._get(bot.OKX_CANDLES_CUR, params)
            if not raw:
                break
        all_rows.extend(raw); pages += 1
        oldest = int(raw[-1][0]); after = oldest
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
    return (df[["datetime", "open", "high", "low", "close", "vol"]]
            .sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True).set_index("datetime"))


print(f"[setup] subset symbols requested: {len(SUBSET)}", flush=True)
feats, above20 = {}, {}
for sym in SUBSET:
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    if not os.path.exists(p):
        continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for c in ["open", "high", "low", "close", "vol"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["open", "high", "low", "close", "vol"], inplace=True)
        inst = sym.replace("_", "-")
        # fetch 2023: ~9200 bars ending just before the cache start
        end_ms = int(df.index[0].timestamp() * 1000) - 1
        f2023 = fetch_before(end_ms, 9200, inst)
        if f2023 is not None and len(f2023):
            df = pd.concat([f2023, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100:
            continue
        f = add_features(df)
        f.dropna(subset=["ema200", "atr14", "adx14", "ema_dist_pct", "real_vol_20",
                         "bb_width", "prev_range_r", "prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f
            above20[sym] = (f["close"] > f["ema20"]).astype(float)
        print(f"  loaded {sym}: {len(f)} bars ({f.index[0].date()} -> {f.index[-1].date()})", flush=True)
    except Exception as e:
        print(f"  err {sym}: {e}", flush=True)
print(f"[data] subset with features: {len(feats)}", flush=True)

breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100
mask = {s: build_signal_mask(f, FAM_A, "green", 1.5) for s, f in feats.items()}
raw = []
for s, f in feats.items():
    for t in sim_symbol(f, mask[s], 1.5, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = s; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
mldf = build_mldf(raw, feats, breadth, breadth_pct)
print(f"[signals] total RAW trades across 2023-2026: {len(mldf)}", flush=True)


def monthly_profile(trades):
    if not trades:
        return dict(prof=float("nan"), worst=0, tpm=0.0)
    df = pd.DataFrame(trades); df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum(); flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0; worst = max(worst, cur)
    days = max(1, (df["entry_time"].max() - df["entry_time"].min()).days)
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df) / (days / 30.0))


def evaluate(name, trades):
    if not trades:
        print(f"\n[{name}] no trades"); return
    s = stats_from_trades(list(trades))
    pf_c = pf_of_rs(cost_adjusted_rs(list(trades), 0.05))
    mp = monthly_profile(trades)
    print(f"\n[{name}]\n  n={len(trades)}  WR={s['wr']:.1%}  PF={s['pf']:.3f}  "
          f"PF@0.05%={pf_c:.3f}  MDD={s['mdd']:.1%}  prof-months={mp['prof']:.1%}  "
          f"worst={mp['worst']}  t/mo={mp['tpm']:.1f}")


def trades_in(year, kept_ts):
    y = pd.Timestamp(f"{year}-01-01", tz="UTC")
    e = pd.Timestamp(f"{year+1}-01-01", tz="UTC")
    return [t for t in raw if t["entry_time"] >= y and t["entry_time"] < e and t["entry_time"] in kept_ts]


print("\n" + "=" * 74)
print("EXTENDED WALK-FORWARD — train on all prior years, test each year (fees 0.05%)")
print("=" * 74)
for test_year in [2024, 2025, 2026]:
    train = mldf[mldf.ts < pd.Timestamp(f"{test_year}-01-01", tz="UTC")]
    test = mldf[mldf.ts.dt.year == test_year]
    if len(train) < 50 or len(test) == 0:
        print(f"\n[test {test_year}] skipped (train={len(train)} test={len(test)})"); continue
    model = SVMQ75(q=0.65).fit_mldf(train)
    kept_ts, _ = model.keep_mldf(test, 0.65)
    evaluate(f"TRAIN≤{test_year-1} -> TEST {test_year}", trades_in(test_year, kept_ts))

print("\n" + "=" * 74)
print("NON-CAUSAL SANITY CHECK — train 2024+ (future), test 2023 (look-ahead)")
print("  (answers 'does 2023's market contain the edge?' — NOT a live test)")
print("=" * 74)
train_fwd = mldf[mldf.ts >= pd.Timestamp("2024-01-01", tz="UTC")]
test_2023 = mldf[mldf.ts.dt.year == 2023]
if len(train_fwd) >= 50 and len(test_2023) > 0:
    m2 = SVMQ75(q=0.65).fit_mldf(train_fwd)
    k2, _ = m2.keep_mldf(test_2023, 0.65)
    evaluate("TRAIN 2024+ -> TEST 2023 (NON-CAUSAL)", trades_in(2023, k2))

print("\n[done]")
