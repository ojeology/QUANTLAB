"""
Pull DERIV 5m historical candles (WS ticks_history, granularity=300) and run a BINARY
directional backtest. The only metric that matters for binaries = WIN RATE. We predict
up/down at the next 5m close; trade only when the RF's P(up) clears a threshold.
Goal: does a directional edge (>55% win rate) exist on 5m? That answers "is binary profitable".
"""
import os, sys, time, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np, websocket
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

APP_ID = 1089
SYMBOL = "R_75"            # Volatility 75 index (24/7, iconic binary instrument)
GRAN = 300                 # 5 minutes
START = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
END   = int(pd.Timestamp("2026-09-02", tz="UTC").timestamp())


def pull_5m(symbol, start, end, gran=GRAN, chunk=5000):
    ws = websocket.create_connection(f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}", timeout=30)
    candles = []; cur = start; req = 0
    while cur < end:
        req += 1
        e = min(cur + chunk*gran, end)
        ws.send(json.dumps({"ticks_history": symbol, "style": "candles", "granularity": gran,
                            "start": cur, "end": e, "count": chunk, "req_id": req}))
        got = None
        while True:
            m = json.loads(ws.recv())
            if m.get("req_id") == req and ("candles" in m or "error" in m):
                got = m; break
        if "error" in got:
            print("  Deriv error:", got["error"]["message"] if "message" in got["error"] else got["error"]); break
        cs = got["candles"]
        if not cs: break
        candles.extend(cs)
        cur = int(cs[-1]["epoch"]) + gran
        time.sleep(0.25)
    ws.close()
    if not candles:
        return None
    df = pd.DataFrame(candles)
    df["epoch"] = pd.to_numeric(df["epoch"])
    df = df.drop_duplicates("epoch").sort_values("epoch")
    df["datetime"] = pd.to_datetime(df["epoch"], unit="s", utc=True)
    df = df[["datetime","open","high","low","close"]].set_index("datetime").astype(float)
    return df


def build_features(df):
    c = df["close"]
    df = df.copy()
    df["r1"] = c.pct_change(1); df["r3"] = c.pct_change(3); df["r5"] = c.pct_change(5)
    df["rsi"] = 100 - 100/(1 + c.diff().clip(lower=0).rolling(14).mean() / (-c.diff().clip(upper=0).rolling(14).mean()).abs())
    df["body"] = (df["close"]-df["open"])/df["open"]
    df["rng"] = (df["high"]-df["low"])/df["close"]
    df["ma20"] = c.rolling(20).mean(); df["dist_ma"] = c/df["ma20"] - 1
    df["vol"] = c.rolling(20).std()
    df["hr"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["up"] = (c.shift(-1) > c).astype(int)   # next 5m close > current close
    return df.dropna()


def backtest(df, thr=0.60):
    d = build_features(df)
    feats = ["r1","r3","r5","rsi","body","rng","dist_ma","vol","hr","dow"]
    d = d.dropna()
    cut = int(len(d)*0.7)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    m = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=0).fit(tr[feats], tr["up"])
    p = m.predict_proba(te[feats])[:,1]
    # trade when confident: P>thr -> long(up), P<1-thr -> short(down)
    longs = p > thr; shorts = p < (1-thr)
    traded = te.iloc[longs | shorts].copy()
    traded["pred"] = np.where(p[longs|shorts] > 0.5, 1, 0)
    win = (traded["pred"] == traded["up"]).mean()
    n = len(traded)
    # baseline: always-up accuracy
    base = te["up"].mean()
    print(f"\n[BACKTEST] {SYMBOL} 5m  candles={len(d)}  test={len(te)}")
    print(f"  naive 'always up' accuracy = {base:.1%}")
    print(f"  RF full-test accuracy     = {(m.predict(te[feats])==te['up']).mean():.1%}")
    print(f"  TRADED (P>={thr:.2f} or <{1-thr:.2f}): n={n}  WIN RATE = {win:.1%}")
    print(f"  -> edge vs 55% bar: {'YES' if win>=0.55 and n>200 else 'NO (need >=55% & enough trades)'}")
    return win, n


print(f"[pull] {SYMBOL} 5m from 2026-01-01 …", flush=True)
df = pull_5m(SYMBOL, START, END)
if df is None or len(df) < 100:
    print("[pull] FAILED — trying frxEURUSD …", flush=True)
    SYMBOL = "frxEURUSD"
    df = pull_5m(SYMBOL, START, END)
print(f"[pull] got {len(df) if df is not None else 0} 5m candles ({df.index[0]} -> {df.index[-1]})", flush=True)
if df is not None and len(df) > 1000:
    backtest(df, thr=0.60)
    # also try a couple thresholds
    for t in [0.55, 0.65]:
        print(f"\n--- threshold {t} ---")
        backtest(df, thr=t)
print("[done]")
