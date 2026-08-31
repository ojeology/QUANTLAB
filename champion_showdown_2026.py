"""
HEAD-TO-HEAD on 2026 (this year): which config is the real champion?

Train = pre-2026 (2024-2025). Test = 2026 (model blind to 2026).
All use RR=1.5, E6 entry (entry_next=False), exit="base", same fee model (0.05%).

Contenders (all built on RAW Family A signals):
  1. RAW       — Family A only (no extra filter)
  2. R077 LOCKED — Family A + breadth50 (frac>0.50 above EMA20) + VolCeil (atr_rank<=70)
  3. SVM q0.75  — Family A + walk-forward SVC keeps top 75% by P(win)
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))
from ql_engine import (add_features, build_signal_mask, sim_symbol,
                       stats_from_trades, cost_adjusted_rs, pf_of_rs,
                       IS_LOOKBACK, RECAL_EVERY)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import demo_bot as bot

FREEZE = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
RR = 1.5
CACHE = "quantlab_cache"
Q = 0.75
FAM_A = ["BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"]

ALL_SYMS = {f[:-len("_1H.parquet")] for f in os.listdir(CACHE) if f.endswith("_1H.parquet")}
print(f"[setup] symbols: {len(ALL_SYMS)}", flush=True)

# ── load data (cache + fresh OKX top-up) ─────────────────────────────────────
raw1h, feats = {}, {}
for sym in sorted(ALL_SYMS):
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    if not os.path.exists(p):
        continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open", "high", "low", "close", "vol"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open", "high", "low", "close", "vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100:
            continue
        inst = sym.replace("_", "-")
        fresh = bot.fetch_candles(inst, n_bars=800)
        if fresh is not None and len(fresh):
            fresh = fresh.astype(float)
            df = pd.concat([df, fresh]); df = df[~df.index.duplicated(keep="last")].sort_index()
        raw1h[sym] = df
        f = add_features(df)
        f.dropna(subset=["ema200", "atr14", "adx14", "ema_dist_pct", "real_vol_20",
                         "bb_width", "prev_range_r", "prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f
    except Exception as e:
        print(f"  load err {sym}: {e}", flush=True)
print(f"[data] symbols with features: {len(feats)}", flush=True)

# ── breadth (fraction of symbols above EMA20) ────────────────────────────────
above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)   # fraction > EMA20

# ── signal masks ──────────────────────────────────────────────────────────────
raw_mask = {s: build_signal_mask(f, FAM_A, "green", 1.5) for s, f in feats.items()}
breadth_reg = {}
for s, f in feats.items():
    b = breadth.reindex(f.index, method="ffill")
    breadth_reg[s] = (b > 0.50).fillna(False).values
r077_mask = {s: (raw_mask[s] & breadth_reg[s]) for s in feats}

raw_trades, r077_trades = [], []
for sym, f in feats.items():
    for t in sim_symbol(f, raw_mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw_trades.append(t)
    for t in sim_symbol(f, r077_mask[sym], RR, dict(entry_next=False, exit="base", hours=None, atr_rank_ceil=70.0)):
        t["sym"] = sym; r077_trades.append(t)
raw_trades.sort(key=lambda t: t["entry_time"])
r077_trades.sort(key=lambda t: t["entry_time"])
print(f"[signals] RAW trades={len(raw_trades)}  R077 trades={len(r077_trades)}", flush=True)

# ── 14-feature matrix for SVM (verbatim from R087) ──────────────────────────
BASE_FEATS = ["atr_rank", "adx14", "rsi14", "ema_dist_pct", "prev_body_r", "prev_range_r",
              "rel_vol", "bb_width", "real_vol_20", "hour", "dow"]
SPECIAL = ["breadth_q", "dist_hi48", "green_streak"]
FEATS = BASE_FEATS + SPECIAL

rows = []
for t in raw_trades:
    sym = t["sym"]; ts = t["entry_time"]; f = feats[sym]
    if ts not in f.index:
        continue
    row = f.loc[ts]; i = f.index.get_loc(ts)
    c = float(row["close"])
    hi48 = float(f["close"].rolling(48).max().iloc[i]) if i >= 0 else np.nan
    dist_hi = (c / hi48 - 1) * 100 if pd.notna(hi48) and hi48 > 0 else 0.0
    streak = 0
    for k in range(0, 6):
        j = i - k
        if j < 0:
            break
        if f["close"].iloc[j] > f["open"].iloc[j]:
            streak += 1
        else:
            break
    bq = float(breadth.rolling(100, min_periods=50).rank(pct=True).mul(100).reindex(
        pd.DatetimeIndex([ts]), method="ffill").iloc[0]) if ts >= breadth.index[0] else 50.0
    rows.append(dict(sym=sym, ts=ts, r=t["r"], win=int(t["r"] > 0),
                     **{c2: row.get(c2, 0) for c2 in BASE_FEATS},
                     breadth_q=bq, dist_hi48=dist_hi, green_streak=streak))
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

# ── train/test split at 2026 ──────────────────────────────────────────────────
train_m = mldf[mldf["ts"] < FREEZE]
test_m = mldf[mldf["ts"] >= FREEZE].copy()
print(f"[split] train(pre-2026)={len(train_m)}  test(2026)={len(test_m)}", flush=True)

test_raw = [t for t in raw_trades if t["entry_time"] >= FREEZE]
test_r077 = [t for t in r077_trades if t["entry_time"] >= FREEZE]

# ── SVM q0.75: train on pre-2026, keep top 75% of 2026 by P(win) ───────────
Xtr = train_m[FEATS].fillna(0).values; ytr = train_m["win"].values
sc = StandardScaler(); clf = SVC(C=1.0, gamma="scale", probability=True)
clf.fit(sc.fit_transform(Xtr), ytr)
pred = clf.predict_proba(sc.transform(test_m[FEATS].fillna(0).values))[:, 1]
thr = np.quantile(pred, 1 - Q)
kept_ts = set(test_m[pred >= thr]["ts"])
test_svm = [t for t in test_raw if t["entry_time"] in kept_ts]
print(f"[svm] q={Q} kept={len(test_svm)}/{len(test_raw)} test trades", flush=True)


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
    print(f"\n[{name}]")
    print(f"  n={len(trades)}  WR={s['wr']:.1%}  PF={s['pf']:.3f}  "
          f"PF@0.05%={pf_c:.3f}  MDD={s['mdd']:.1%}")
    print(f"  profitable-months={mp['prof']:.1%}  worst-streak={mp['worst']}  ~t/mo={mp['tpm']:.1f}")


if __name__ == "__main__":
    t0 = time.time()
    print("\n" + "=" * 74)
    print("CHAMPION SHOWDOWN — 2026 blind (train pre-2026, test 2026, fees 0.05%)")
    print("=" * 74)
    evaluate("RAW Family A", test_raw)
    evaluate(f"SVM q{Q} (current champion)", test_svm)
    evaluate("R077 LOCKED (breadth50 + VolCeil)", test_r077)
    print(f"\n[done in {time.time()-t0:.1f}s]")
