"""
R073 follow-up: monthly trade breakdown for the FINAL strategy
(Family A + E6_sigentry + RR 2.0, bot-faithful rolling engine).
"""
import os, sys, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/quantlab")
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]

IS_LOOKBACK  = 500
RECAL_EVERY  = 168

STRAT = {
    "label": "Family A (E6_sigentry, RR=2.0)",
    "cids": ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"],
    "rr": 2.0,
    "variant": "E6_sigentry",
}

COND_DEF = {
    "DST_NR":     ("ema_dist_pct", "lt_q", 0.33),
    "ADX_ST":     ("adx14",        "gt_q", 0.67),
    "PBD_HI":     ("prev_body_r",  "gt_q", 0.67),
    "BBW_STRICT": ("bb_width",     "lt_q", 0.25),
    "RV_LO":      ("real_vol_20",  "lt_q", 0.33),
    "PRG_VH":     ("prev_range_r", "gt_q", 0.80),
}

def add_features(df):
    df = df.copy()
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    df["ema200"]        = calc_ema(c, 200)
    df["atr14"]         = calc_atr(df, 14)
    bb_mid              = c.rolling(20).mean()
    bb_std              = c.rolling(20).std(ddof=0)
    df["bb_width"]      = (bb_std * 2) / bb_mid.replace(0, np.nan) * 100.0
    df["real_vol_20"]   = c.pct_change().rolling(20).std() * 100.0
    ema200_s            = df["ema200"].replace(0, np.nan)
    df["ema_dist_pct"]  = (c - ema200_s) / ema200_s * 100.0
    df["prev_range_r"]  = (h.shift(1)-l.shift(1)).abs() / c.shift(1).replace(0,np.nan) * 100.0
    df["prev_body_r"]   = (c.shift(1)-o.shift(1)).abs() / c.shift(1).replace(0,np.nan) * 100.0
    df["adx14"]         = calc_adx(df, 14)
    vol_avg             = df["vol"].rolling(20).mean()
    df["rel_vol"]       = df["vol"] / vol_avg.replace(0, np.nan)
    return df

def compute_thresholds(df_is, cids):
    out = {}
    for cid in cids:
        col, direction, q = COND_DEF[cid]
        vals = df_is[col].dropna()
        if len(vals) < 10:
            out[cid] = None
            continue
        out[cid] = float(vals.quantile(q))
    return out

def build_signal_mask(feats, cids):
    c = feats["close"].values
    o = feats["open"].values
    rv = feats["rel_vol"].values
    n = len(feats)
    sig = np.zeros(n, dtype=bool)
    cal = IS_LOOKBACK
    while cal < n:
        end = min(cal + RECAL_EVERY, n)
        thr = compute_thresholds(feats.iloc[cal - IS_LOOKBACK:cal], cids)
        m = np.ones(end - cal, dtype=bool)
        for cid in cids:
            col, direction, _ = COND_DEF[cid]
            vals = feats[col].values[cal:end]
            tq = thr.get(cid)
            if tq is None:
                m[:] = False
                break
            if direction == "lt_q":
                m &= vals < tq
            else:
                m &= vals > tq
        sig[cal:end] = m
        cal = end
    gate = (c > o) & (c > np.roll(c, 1)) & (rv > 1.5)
    gate[:1] = False
    gate = gate & ~np.isnan(c) & ~np.isnan(o) & ~np.isnan(rv)
    return sig & gate

def sim_symbol(feats, sig, rr, variant="E6_sigentry"):
    c = feats["close"].values
    h = feats["high"].values
    l = feats["low"].values
    atr = feats["atr14"].values
    n = len(feats)
    entry_next = variant != "E6_sigentry"
    trades = []
    pos = None
    for i in range(IS_LOOKBACK + 1, n):
        if pos is not None and i >= pos["first_check"]:
            e = pos["entry"]; a = pos["atr"]
            done = None
            if h[i] >= pos["tp"]: done = (rr, "TP")
            elif l[i] <= pos["sl"]: done = (-1.0, "SL")
            if done is not None:
                r_mult, xtype = done
                trades.append(dict(entry_time=feats.index[pos["entry_i"]], r=r_mult,
                                   exit_type=xtype, bars_in=i - pos["entry_i"],
                                   atr=pos["atr"], entry=pos["entry"]))
                pos = None
        if pos is not None or not sig[i]:
            continue
        if i + 1 >= n and entry_next:
            continue
        a = atr[i]
        if not (a > 0) or math.isnan(a):
            continue
        entry = c[i + 1] if entry_next else c[i]
        pos = dict(entry_i=i, first_check=i + (1 if not entry_next else 2),
                   entry=entry, atr=a, sl=entry - a, tp=entry + rr * a)
    if pos is not None:
        trades.append(dict(entry_time=feats.index[pos["entry_i"]],
                           r=(c[-1] - pos["entry"]) / pos["atr"],
                           exit_type="OPEN", bars_in=n - 1 - pos["entry_i"],
                           atr=pos["atr"], entry=pos["entry"]))
    return trades

# ── Load ─────────────────────────────────────────────────────────────────────
feats_by_sym = {}
for fn in sorted(os.listdir(CACHE)):
    if not fn.endswith("_1H.parquet"): continue
    sym = fn.replace("_1H.parquet", "")
    try:
        df = pd.read_parquet(os.path.join(CACHE, fn))
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats_by_sym[sym] = f
    except Exception:
        pass

mask = {sym: build_signal_mask(f, STRAT["cids"]) for sym, f in feats_by_sym.items()}
trades = []
for sym, feats in feats_by_sym.items():
    for t in sim_symbol(feats, mask[sym], STRAT["rr"], STRAT["variant"]):
        t["sym"] = sym
        trades.append(t)
trades.sort(key=lambda t: t["entry_time"])

df = pd.DataFrame(trades)
df["month"] = df["entry_time"].dt.to_period("M")

print(f"TOTAL TRADES: {len(df)}")
print(f"Time span: {df['entry_time'].min()} → {df['entry_time'].max()}")
months = df["month"].nunique()
print(f"Months with trades: {months}")
print(f"Average trades/month: {len(df)/months:.1f}")
print(f"Median trades/month: {df.groupby('month').size().median():.0f}")
print(f"Min/Max trades in a month: {df.groupby('month').size().min()}/{df.groupby('month').size().max()}")
print(f"Std dev trades/month: {df.groupby('month').size().std():.1f}")

print("\nMONTHLY BREAKDOWN (trades, WR, PF, net R):")
rows = []
for m, g in df.groupby("month"):
    rs = g["r"].values
    wins = rs[rs > 0]; losses = rs[rs < 0]
    gw, gl = wins.sum(), abs(losses.sum())
    pf = gw / gl if gl > 0 else float("inf")
    rows.append(dict(month=str(m), n=len(g), wr=float((rs > 0).mean()),
                     pf=pf, net_r=float(rs.sum()), exp=float(rs.mean())))
mdf = pd.DataFrame(rows)
print(mdf.to_string(index=False))

print("\nEXIT TYPE DISTRIBUTION:")
print(df["exit_type"].value_counts().to_string())

print("\nTRADES BY SYMBOL (top 15):")
print(df["sym"].value_counts().head(15).to_string())

print("\nHOUR DISTRIBUTION (entry hour UTC, top 8):")
print(df["entry_time"].dt.hour.value_counts().head(8).to_string())

print("\nTRADE DURATION (bars):")
print(f"  mean={df['bars_in'].mean():.1f}  median={df['bars_in'].median():.0f}  "
      f"p90={df['bars_in'].quantile(0.9):.0f}")

# also E0 for comparison
trades_e0 = []
for sym, feats in feats_by_sym.items():
    for t in sim_symbol(feats, mask[sym], STRAT["rr"], "E0_base"):
        t["sym"] = sym
        trades_e0.append(t)
dfe0 = pd.DataFrame(trades_e0)
months_e0 = dfe0["entry_time"].dt.to_period("M").nunique()
print(f"\n[Comparison] E0_base (current bot): {len(dfe0)} trades, "
      f"{len(dfe0)/months_e0:.1f}/month avg")

mdf.to_csv(os.path.join(OUT, "r073_final_strategy_monthly.csv"), index=False)
print(f"\nSaved → {OUT}/r073_final_strategy_monthly.csv")
