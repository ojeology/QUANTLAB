"""
BLIND OUT-OF-SAMPLE TEST — CRYPTO SVM q0.75 champion (R086/R087).

Reuses QUANTLAB's exact engine (ql_engine.add_features, build_signal_mask,
sim_symbol, stats_from_trades) and the exact research configuration:
  - RAW Family A signal: cids=[BBW_STRICT,RV_LO,DST_NR,PRG_VH], gate=green, rel_vol>1.5
  - 14 features (atr_rank, adx14, rsi14, ema_dist_pct, prev_body_r, prev_range_r,
    rel_vol, bb_width, real_vol_20, hour, dow, breadth_q, dist_hi48, green_streak)
  - SVC(C=1.0, gamma="scale", probability=True), StandardScaler
  - RR = 1.5, entry_next=False (E6), exit="base"

DIFFERENCE FROM RESEARCH: research evaluated on its 2026 holdout (already in
cache). Here we train the SVM ONLY on PRE-FREEZE data (before 2026-08-08) and
evaluate ONLY on POST-FREEZE data (2026-08-08 -> now) pulled fresh from OKX.
That is a true blind out-of-sample test the strategy has never seen.

Keep rule: top q=0.75 of post-freeze signals by predicted P(win) (per R087).
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
import demo_bot as bot   # for OKX fetch_candles + cache map

FREEZE = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
RR = 1.5
CACHE = "quantlab_cache"
Q = 0.75

ALL_SYMS = {f[:-len("_1H.parquet")] for f in os.listdir(CACHE) if f.endswith("_1H.parquet")}
print(f"[setup] symbols in cache: {len(ALL_SYMS)}", flush=True)

# ── load data: cache + fresh OKX top-up to cover post-freeze window ──────────
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
            df = pd.concat([df, fresh])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        raw1h[sym] = df
        f = add_features(df)
        f.dropna(subset=["ema200", "atr14", "adx14", "ema_dist_pct", "real_vol_20",
                         "bb_width", "prev_range_r", "prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f
    except Exception as e:
        print(f"  load err {sym}: {e}", flush=True)
print(f"[data] symbols with features: {len(feats)}", flush=True)

# ── breadth (cross-sectional) exactly as R087 ───────────────────────────────
above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100

# ── RAW Family A signals + trade simulation ─────────────────────────────────
cids = ["BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"]
mask = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats.items()}
raw = []
for sym, f in feats.items():
    for t in sim_symbol(f, mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym
        raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"[signals] raw trades: {len(raw)}", flush=True)

# ── build 14-feature matrix (verbatim from R087) ────────────────────────────
BASE_FEATS = ["atr_rank", "adx14", "rsi14", "ema_dist_pct", "prev_body_r",
              "prev_range_r", "rel_vol", "bb_width", "real_vol_20", "hour", "dow"]
SPECIAL = ["breadth_q", "dist_hi48", "green_streak"]
FEATS = BASE_FEATS + SPECIAL

rows = []
for t in raw:
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
    bq = float(breadth_pct.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) \
        if ts >= breadth_pct.index[0] else 50.0
    rows.append(dict(sym=sym, ts=ts, r=t["r"], win=int(t["r"] > 0),
                     **{c2: row.get(c2, 0) for c2 in BASE_FEATS},
                     breadth_q=bq, dist_hi48=dist_hi, green_streak=streak))
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)

# ── BLIND SPLIT: train on pre-freeze, test on post-freeze ───────────────────
train_m = mldf[mldf["ts"] < FREEZE]
test_m = mldf[mldf["ts"] >= FREEZE].copy()
print(f"[split] pre-freeze(train)={len(train_m)}  post-freeze(BLIND)={len(test_m)}", flush=True)

# ── train SVM on pre-freeze, predict on post-freeze ─────────────────────────
Xtr = train_m[FEATS].fillna(0).values; ytr = train_m["win"].values
sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr)
clf = SVC(C=1.0, gamma="scale", probability=True)
clf.fit(Xtr_s, ytr)
Xte_s = sc.transform(test_m[FEATS].fillna(0).values)
test_m["pred"] = clf.predict_proba(Xte_s)[:, 1]

thr = np.quantile(test_m["pred"].values, 1 - Q)
kept = test_m[test_m["pred"] >= thr]
kept_ts = set(kept["ts"])

test_raw_trades = [t for t in raw if t["entry_time"] >= FREEZE]
kept_trades = [t for t in raw if t["entry_time"] in kept_ts]
train_trades = [t for t in raw if t["entry_time"] < FREEZE]
print(f"[svm] q={Q} keep-threshold(pred Pwin)={thr:.3f}  kept={len(kept_trades)}/{len(test_raw_trades)} test trades", flush=True)


def monthly_profile(trades):
    if not trades:
        return dict(prof=float("nan"), worst=0, tpm=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        worst = max(worst, cur)
    days = max(1, (df["entry_time"].max() - df["entry_time"].min()).days)
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df) / (days / 30.0))


def evaluate(name, trades):
    if not trades:
        print(f"\n[{name}] no trades")
        return
    s = stats_from_trades(list(trades))
    pf_c = pf_of_rs(cost_adjusted_rs(list(trades), 0.05))
    mp = monthly_profile(trades)
    print(f"\n[{name}]")
    print(f"  n={len(trades)}  WR={s['wr']:.1%}  PF={s['pf']:.3f}  "
          f"PF@0.05%={pf_c:.3f}  MDD={s['mdd']:.1%}")
    print(f"  profitable-months={mp['prof']:.1%}  worst-streak={mp['worst']}  ~t/mo={mp['tpm']:.1f}")


if __name__ == "__main__":
    t0 = time.time()
    print("\n" + "=" * 72)
    print("CRYPTO SVM q0.75 — BLIND OOS TEST on 2026 (this year)")
    print(f"Train = pre-2026 (2024-2025) | Test = 2026, SVM blind to 2026 | SVM(RBF) 14 feats | RR={RR} | keep top q={Q}")
    print("=" * 72)
    evaluate("RAW Family A — post-freeze (no filter)", test_raw_trades)
    evaluate(f"SVM q{Q} — post-freeze (BLIND, kept)", kept_trades)
    evaluate("SVM — pre-freeze (in-sample ref)", train_trades)
    print(f"\n[done in {time.time()-t0:.1f}s]")
