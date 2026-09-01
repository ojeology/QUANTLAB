"""
NEW HYPOTHESIS — Trend-following (Donchian breakout) on 1H crypto.
Different regime bet from the mean-reversion SVM. Rule-based (no ML overfit),
ATR stop, long-only. Validated across 2024-2026 (2023 fetched in-run so 2024 is
testable). Reports PF@cost, MAX DD (1% risk), and profitable-months per year.
"""
import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from ql_engine import add_features, IS_LOOKBACK, RECAL_EVERY
import demo_bot as bot

CACHE = "quantlab_cache"
SUBSET = ["BTC_USDT_SWAP","ETH_USDT_SWAP","SOL_USDT_SWAP","BNB_USDT_SWAP","XRP_USDT_SWAP",
          "DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","LINK_USDT_SWAP","DOT_USDT_SWAP",
          "LTC_USDT_SWAP","TRX_USDT_SWAP","ATOM_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP",
          "ARB_USDT_SWAP","OP_USDT_SWAP","SUI_USDT_SWAP","ETC_USDT_SWAP","XLM_USDT_SWAP"]


def fetch_before(end_ts_ms, n_bars, inst, bar="1H", page_limit=200):
    all_rows, after, pages = [], end_ts_ms, 0
    while len(all_rows) < n_bars and pages < page_limit:
        params = {"instId": inst, "bar": bar, "limit": bot.PAGE_LIMIT, "after": str(after)}
        raw = bot._get(bot.OKX_CANDLES, params) or bot._get(bot.OKX_CANDLES_CUR, params)
        if not raw:
            break
        all_rows.extend(raw); pages += 1; after = int(raw[-1][0])
        if len(all_rows) >= n_bars:
            break
        time.sleep(bot.PAGE_DELAY)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=bot.CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return (df[["datetime","open","high","low","close","vol"]].sort_values("datetime")
            .drop_duplicates("datetime").reset_index(drop=True).set_index("datetime"))


def backtest_donchian(df, N=20, Nx=10, atr_mult=2.0, adx_min=25.0):
    """Long-only Donchian breakout with ATR stop. Returns list of trade dicts (r=gross return)."""
    df = df.copy()
    hh = df["high"].rolling(N).max().shift(1)
    ll = df["low"].rolling(Nx).min().shift(1)
    trades = []
    in_pos = False; entry_price = None; stop = None
    for i in range(N, len(df)):
        bar = df.iloc[i]
        if not in_pos:
            if (bar["close"] > hh.iloc[i]) and (bar["adx14"] > adx_min) and (bar["close"] > bar["ema200"]):
                entry_price = bar["close"]; stop = entry_price - atr_mult * bar["atr14"]
                in_pos = True
        else:
            exit_price = None; etype = None
            if bar["low"] <= stop:
                exit_price = stop; etype = "SL"
            elif bar["close"] < ll.iloc[i]:
                exit_price = bar["close"]; etype = "BRK"
            if exit_price is not None:
                trades.append(dict(entry_time=df.index[i], exit_time=df.index[i],
                                   r=(exit_price/entry_price - 1.0), etype=etype))
                in_pos = False
    return trades


print("[load] fetch 2023 + cache 2024-2026 for 20-sym subset …", flush=True)
feats = {}
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
        f.dropna(subset=["ema200","atr14","adx14"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f
        print(f"  loaded {sym}: {len(f)} bars ({f.index[0].date()} -> {f.index[-1].date()})", flush=True)
    except Exception as e:
        print(f"  err {sym}: {e}", flush=True)
print(f"[load] subset: {len(feats)}", flush=True)

# Backtest all symbols, collect trades
all_trades = []
for sym, f in feats.items():
    tr = backtest_donchian(f)
    for t in tr:
        t["sym"] = sym; all_trades.append(t)
all_trades.sort(key=lambda t: t["entry_time"])
print(f"[bt] total trend trades 2023-2026: {len(all_trades)}", flush=True)

FEE = 0.0005  # per side -> 0.1% round trip
for t in all_trades:
    t["adj_r"] = t["r"] - 2*FEE


def equity_mdd(trades, risk=0.01):
    eq = 1.0; peak = 1.0; mdd = 0.0
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        eq *= (1 + risk * t["adj_r"]); peak = max(peak, eq); mdd = min(mdd, eq/peak - 1)
    return eq - 1, mdd


print("\n" + "=" * 82)
print("NEW HYPOTHESIS: Trend-following (Donchian 20/10, ATRx2 stop) — 1H crypto")
print("=" * 82)
for Y in [2024, 2025, 2026]:
    yt = [t for t in all_trades if t["entry_time"].year == Y]
    if not yt:
        print(f"[{Y}] no trades"); continue
    ret, mdd = equity_mdd(yt)
    rs = [t["adj_r"] for t in yt]
    wins = sum(1 for r in rs if r > 0)/len(rs)
    pf = (sum(r for r in rs if r>0)) / max(1e-9, -sum(r for r in rs if r<0))
    msum = defaultdict(float)
    for t in yt:
        msum[(t["entry_time"].year, t["entry_time"].month)] += t["adj_r"]
    pos = sum(1 for v in msum.values() if v > 0)
    print(f"\n[{Y}] trades={len(yt)} win={wins:.0%} PF@cost={pf:.3f} ret(1%)={ret:+.1%} MAX DD={mdd:.1%} prof-months={pos}/{len(msum)}")
    for (yy, mm) in sorted(msum):
        nr = msum[(yy,mm)]
        print(f"   {yy}-{mm:02d}  netR={nr:>+7.2f}  {'OK' if nr>0 else 'LOSS'}")
ret, mdd = equity_mdd(all_trades)
rs = [t["adj_r"] for t in all_trades]
pf = (sum(r for r in rs if r>0)) / max(1e-9, -sum(r for r in rs if r<0))
print(f"\n[FULL 2024-2026] trades={len(all_trades)} PF@cost={pf:.3f} ret(1%)={ret:+.1%} MAX DD={mdd:.1%}")
print("\n[done]")
