#!/usr/bin/env python3
# R121 step 2 — FX transfer test: score FX pops with the FROZEN crypto SVM (prespec: docs/RESEARCH_FRONTIER_R121.md).
# Fills SL-first (convention #5). Keep rule: expanding-window 55% rank (no future scores used).
import os, sys, glob, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, build_signal_mask, IS_LOOKBACK, RECAL_EVERY

CACHE = CONFIG["CACHE_FOLDER"]
FXD = os.path.join(CACHE, "forex10")
RR = 1.5
Q = 0.55
RT_PIPS = {"EURUSD":0.6,"GBPUSD":0.9,"USDJPY":0.7,"AUDUSD":0.8,"USDCAD":1.0,"USDCHF":1.0,"NZDUSD":1.2,"EURGBP":1.2}
PIP = {"USDJPY":0.01}
PRIMARY0, PRIMARY1 = pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
WARMUP_SIG = 100

art = pickle.load(open("quantlab_output/r121_crypto_svm.pkl", "rb"))
FEATS = art["features"]

pairs = sorted(os.path.basename(f).replace("_1H.parquet","") for f in glob.glob(os.path.join(FXD,"*_1H.parquet")))
print(f"FX panel: {pairs}")

feats, raw_sym = {}, {}
for p in pairs:
    df = pd.read_parquet(os.path.join(FXD, f"{p}_1H.parquet"))
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    if len(df) < IS_LOOKBACK + RECAL_EVERY + 200:
        print(f"  {p}: only {len(df)} bars — skipped"); continue
    f = add_features(df)
    f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width",
                     "prev_range_r","prev_body_r"], inplace=True)
    feats[p] = f

cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
masks = {p: build_signal_mask(f, cids, "green", 1.5) for p, f in feats.items()}
above20 = {p: (f["close"] > f["ema20"]).astype(float) for p, f in feats.items()}
breadth_pct = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True).rolling(100, min_periods=50).rank(pct=True) * 100

def sim_sl_first(f, mask, rr, rt_pips, pipsize):
    """Entry at signal-bar close (E6). SL=1*ATR, TP=rr*ATR. SL-FIRST on ambiguous bars.
    Cost: round-trip pips charged once at exit, in R = rt_pips*pipsize / atr_entry."""
    out = []
    idx = f.index
    c, h, l, atr = f["close"].values, f["high"].values, f["low"].values, f["atr14"].values
    sigs = np.where(mask)[0]
    last_exit = -1
    for i in sigs:
        if i <= last_exit or i + 1 >= len(idx) or not np.isfinite(atr[i]) or atr[i] <= 0: continue
        entry = c[i]; sl = entry - atr[i]; tp = entry + rr * atr[i]
        cost_r = (rt_pips * pipsize) / atr[i]
        res = None; amb = 0
        for j in range(i + 1, len(idx)):
            hit_tp, hit_sl = h[j] >= tp, l[j] <= sl
            if hit_tp and hit_sl: amb += 1; res = (-1.0 - cost_r, "SL(amb)", j); break
            if hit_sl: res = (-1.0 - cost_r, "SL", j); break
            if hit_tp: res = (rr - cost_r, "TP", j); break
        if res is None: continue          # still open at data end — excluded, counted
        last_exit = res[2]
        out.append(dict(entry_time=idx[i], r=res[0] + cost_r, exit_type=res[1],
                        bars=res[2] - i, ambiguous=(res[1].endswith("(amb)")),
                        atr=atr[i], entry=entry))
    return out

trades = []
for p in pairs:
    pipsize = PIP.get(p, 0.0001)
    for t in sim_sl_first(feats[p], masks[p], RR, RT_PIPS[p], pipsize):
        t["sym"] = p; trades.append(t)
trades.sort(key=lambda t: t["entry_time"])
print(f"FX raw pops: {len(trades)}")

# features per trade (same builder semantics as the champion)
rows = []
for t in trades:
    p = t["sym"]; ts = t["entry_time"]; f = feats[p]
    i = f.index.get_loc(ts); row = f.loc[ts]
    c = float(row["close"])
    hi48 = float(f["close"].rolling(48).max().iloc[i]) if i >= 0 else np.nan
    dist_hi = (c / hi48 - 1) * 100 if pd.notna(hi48) and hi48 > 0 else 0.0
    streak = 0
    for k in range(0, 6):
        j = i - k
        if j < 0: break
        if f["close"].iloc[j] > f["open"].iloc[j]: streak += 1
        else: break
    bq = float(breadth_pct.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) if ts >= breadth_pct.index[0] else 50.0
    rows.append(dict(sym=p, entry_time=ts, r=t["r"], exit_type=t["exit_type"], ambiguous=t["ambiguous"],
                     **{cf: row.get(cf, 0) for cf in FEATS if cf in ("atr_rank","adx14","rsi14","ema_dist_pct",
                        "prev_body_r","prev_range_r","rel_vol","bb_width","real_vol_20","hour","dow")},
                     breadth_q=bq, dist_hi48=dist_hi, green_streak=streak))
T = pd.DataFrame(rows)
Xs = art["scaler"].transform(T[FEATS].fillna(0).values)
T["score"] = art["model"].predict_proba(Xs)[:, 1]

# expanding-window keep rule: keep if score >= q-quantile of TRAILING scores (min WARMUP_SIG seen)
keep_frac = np.full(len(T), np.nan)
for i in range(len(T)):
    if i < WARMUP_SIG: keep_frac[i] = 1.0            # warmup: keep all (labeled)
    else: keep_frac[i] = 1.0 if T.score.iloc[i] >= T.score.iloc[:i].quantile(1 - Q) else 0.0
T["kept"] = keep_frac
n_amb = int(T.ambiguous.sum())
print(f"ambiguous bars (SL-first enforced): {n_amb} | warmup trades kept-all: {WARMUP_SIG}")

def pf(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    w = x[x > 0].sum(); lo = abs(x[x < 0].sum())
    return w / lo if lo > 0 else (999.0 if w > 0 else 1.0)

prim = T[(T.entry_time >= PRIMARY0) & (T.entry_time < PRIMARY1)]
ctx  = T[T.entry_time < PRIMARY0]
kept, rawt = prim[prim.kept == 1], prim
rng = np.random.default_rng(121)
rand_pfs = [pf(kept.r.values[rng.random(len(kept)) < 1]) if False else pf(rng.choice(rawt.r.values, len(kept), replace=False)) for _ in range(200)]
rand_q = float(np.quantile(rand_pfs, Q))
boot = [pf(rng.choice(kept.r.values, len(kept))) for _ in range(2000)]
boot_p = float(np.mean(np.array(boot) > 1.0))
loao = {}
for p in kept.sym.unique():
    loao[p] = pf(kept[kept.sym != p].r.values)

print("\n" + "=" * 96)
print(f"  R121 — CRYPTO BRAIN → FX TRANSFER (primary eval 2021-01→2025-12, SL-first fills, RR {RR}, q {Q})")
print("=" * 96)
print(f"  raw pops   : n={len(rawt):>4}  PF@c={pf(rawt.r.values):.2f}  WR={(rawt.r > 0).mean()*100:.0f}%  sum={rawt.r.sum():+.1f}R")
print(f"  kept ( SVM): n={len(kept):>4}  PF@c={pf(kept.r.values):.2f}  WR={(kept.r > 0).mean()*100:.0f}%  sum={kept.r.sum():+.1f}R")
print(f"  random-55% : q-quantile PF@c = {rand_q:.2f}   | crypto-model kept-set vs raw delta: {pf(kept.r.values)-pf(rawt.r.values):+.2f}")
print(f"  bootP(kept PF>1) = {boot_p:.3f}   | LOO per pair: " + " ".join(f"{p}:{v:.2f}" for p, v in sorted(loao.items())))
if len(ctx):
    kc = ctx[ctx.kept == 1]
    print(f"  context 2016-2020 (report-only): raw PF {pf(ctx.r.values):.2f} (n={len(ctx)}) | kept PF {pf(kc.r.values):.2f} (n={len(kc)})")
print(f"  score sanity: crypto-model scores on FX — mean {T.score.mean():.3f}, sd {T.score.std():.3f} "
      f"(crypto train WR was {art['win_rate']:.2%})")

g1 = pf(kept.r.values) >= 1.10
g2 = len(kept) >= 100
g3 = boot_p >= 0.95
g4 = min(loao.values()) > 1.0
b1 = pf(kept.r.values) > pf(rawt.r.values)
b2 = pf(kept.r.values) > rand_q
print(f"\n  GATES: G1 PF>=1.10 {'PASS' if g1 else 'FAIL'} | G2 n>=100 {'PASS' if g2 else 'FAIL'} | "
      f"G3 boot>=0.95 {'PASS' if g3 else 'FAIL'} | G4 LOO>1 {'PASS' if g4 else 'FAIL'}")
print(f"  MUST-BEAT: raw {'PASS' if b1 else 'FAIL'} | random {'PASS' if b2 else 'FAIL'}")
verdict = "PROMISING FOR FURTHER VALIDATION" if all([g1,g2,g3,g4,b1,b2]) else "REJECT"
print(f"\n  VERDICT: {verdict}")
T.to_csv("quantlab_output/r121_fx_trades.csv", index=False)
print("saved: r121_fx_trades.csv")
