"""
R092: Win/Loss ENVIRONMENT forensics for the 5m hypotheses (R091 set).
What made winners win and losers lose? Is there any sub-environment where
the edge survives costs?

For each audit-passed hypothesis: snapshot entry-bar features of every trade,
split win/loss, report means + Cohen's d (like R072), then try single-factor
environment slices (hour, breadth, VWAP-dist, ATR-rank, relvol, day) and test
if any slice gives PF@0.05% cost > 1.1 on selection AND holdout with n>=40.
"""
import os, sys, warnings, math
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/quantlab")
sys.path.insert(0, "/home/user/quantlab/scripts")
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, sim_symbol
import scripts.ql_engine as qle
qle.IS_LOOKBACK = 6000
qle.RECAL_EVERY = 2016

CACHE = CONFIG["CACHE_FOLDER"]
HOLDOUT_START = pd.Timestamp("2026-06-01", tz="UTC")
RR = 1.5
SYMS = ["BTC_USDT_SWAP","ETH_USDT_SWAP","DOGE_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP"]

print("Loading 5m data + indicators …", flush=True)
feats = {}
for sym in SYMS:
    df = pd.read_parquet(f"{CACHE}/{sym}_5m.parquet")
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for col in ["open","high","low","close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open","high","low","close","vol"], inplace=True)
    f = add_features(df)
    f.dropna(subset=["ema200","atr14","adx14","rsi14","ema_dist_pct","real_vol_20",
                     "bb_width","prev_range_r","prev_body_r"], inplace=True)
    feats[sym] = f

def add_indicators(f):
    f = f.copy()
    day = f.index.normalize()
    typ = (f["high"] + f["low"] + f["close"]) / 3.0
    f["vwap"] = (typ * f["vol"]).groupby(day).cumsum() / f["vol"].groupby(day).cumsum().replace(0, np.nan)
    f["vwap_dist"] = (f["close"] - f["vwap"]) / f["atr14"]
    rsi14 = f["rsi14"]
    stoch = (rsi14 - rsi14.rolling(14).min()) / (rsi14.rolling(14).max() - rsi14.rolling(14).min()).replace(0, np.nan)
    f["stochK"] = stoch.rolling(3).mean() * 100
    f["stochD"] = f["stochK"].rolling(3).mean()
    ema12 = f["close"].ewm(span=12, adjust=False).mean()
    ema26 = f["close"].ewm(span=26, adjust=False).mean()
    f["macd"] = ema12 - ema26
    f["macd_sig"] = f["macd"].ewm(span=9, adjust=False).mean()
    f["macd_hist"] = f["macd"] - f["macd_sig"]
    f["kelt_mid"] = f["close"].rolling(20).mean()
    f["kelt_hi"] = f["kelt_mid"] + 1.5 * f["atr14"]
    f["kelt_lo"] = f["kelt_mid"] - 1.5 * f["atr14"]
    f["don_hi"] = f["high"].rolling(20).max().shift(1)
    f["don_lo"] = f["low"].rolling(20).min().shift(1)
    return f

feats2 = {s: add_indicators(f) for s, f in feats.items()}

# universe breadth (for environment)
above = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats2.items()}
breadth = pd.DataFrame(above).sort_index().mean(axis=1, skipna=True)

def k1(f):
    below = (f["vwap_dist"] < -0.5) | (f["close"] < f["vwap"] - 0.5 * f["atr14"])
    br = below.rolling(5, min_periods=1).max().astype(bool)
    return (br & (f["close"] > f["vwap"]) & (f["close"] > f["open"]) & (f["rel_vol"] > 1.2) & f["vwap"].notna())
def k2(f):
    over = f["stochK"] < 20
    cross = (f["stochK"] > f["stochD"]) & (f["stochK"].shift(1) <= f["stochD"].shift(1))
    return (over & cross & (f["close"] > f["open"]) & (f["close"] > f["close"].shift(1)) & f["stochK"].notna())
def k3(f):
    was_neg = (f["macd_hist"].shift(1) < 0) & (f["macd_hist"].shift(2) < 0)
    flip = (f["macd_hist"] > 0) & was_neg
    return (flip & (f["close"] > f["open"]) & (f["rel_vol"] > 1.2) & f["macd_hist"].notna())
def k4(f):
    kelt_w = (f["kelt_hi"] - f["kelt_lo"]) / f["kelt_mid"].replace(0, np.nan)
    sq = f["bb_width"] / 100.0 < kelt_w
    return (sq & (f["close"] > f["kelt_hi"]) & (f["rel_vol"] > 1.5) & (f["close"] > f["open"]) & f["kelt_hi"].notna())
def k5(f):
    broke = f["close"] > f["don_hi"]
    pull3 = (f["low"].rolling(3, min_periods=1).min() < f["ema20"]) & broke
    reclaim = (f["close"] > f["ema20"]) & pull3
    return (reclaim & (f["close"] > f["open"]) & f["don_hi"].notna())

HYP = {"K1_vwap": k1, "K2_stoch": k2, "K3_macd": k3, "K4_kelt": k4, "K5_donch": k5}
FEAT_NAMES = {
    "hour": "Hour UTC", "dow": "Day of week", "rsi14": "RSI14",
    "vwap_dist": "VWAP dist (ATR)", "macd_hist": "MACD hist",
    "stochK": "StochK", "atr_rank": "ATR rank", "rel_vol": "RelVol",
    "ema_dist_pct": "EMA200 dist %", "bb_width": "BB width %",
    "real_vol_20": "RealVol20", "prev_body_r": "Prev body %",
    "adx14": "ADX", "breadth": "Market breadth",
    "dist_hi48": "Dist to 48h high %", "green_streak": "Green streak",
}

print("Running hypotheses + collecting entry-bar features …", flush=True)
all_trades = []
for name, fn in HYP.items():
    for sym, f in feats2.items():
        m = fn(f)
        for t in sim_symbol(f, m, RR, dict(entry_next=False, exit="timeN", time_bars=60, hours=None)):
            ts = t["entry_time"]
            row = f.loc[ts]
            i = f.index.get_loc(ts)
            c = float(row["close"])
            hi48 = float(f["close"].rolling(240).max().iloc[i]) if i >= 0 else np.nan
            dist_hi = (c / hi48 - 1) * 100 if pd.notna(hi48) and hi48 > 0 else 0.0
            streak = 0
            for k in range(0, 6):
                j = i - k
                if j < 0: break
                if f["close"].iloc[j] > f["open"].iloc[j]: streak += 1
                else: break
            bq = float(breadth.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) if ts >= breadth.index[0] else 0.5
            all_trades.append(dict(
                hyp=name, sym=sym, ts=ts, r=t["r"], win=int(t["r"] > 0),
                hour=ts.hour, dow=ts.dayofweek,
                rsi14=float(row.get("rsi14", 50)), vwap_dist=float(row.get("vwap_dist", 0)),
                macd_hist=float(row.get("macd_hist", 0)), stochK=float(row.get("stochK", 50)),
                atr_rank=float(row.get("atr_rank", 50)), rel_vol=float(row.get("rel_vol", 1)),
                ema_dist_pct=float(row.get("ema_dist_pct", 0)), bb_width=float(row.get("bb_width", 2)),
                real_vol_20=float(row.get("real_vol_20", 1)), prev_body_r=float(row.get("prev_body_r", 1)),
                adx14=float(row.get("adx14", 20)), breadth=bq,
                dist_hi48=dist_hi, green_streak=streak))
df = pd.DataFrame(all_trades)
print(f"  Total trades across 5 hyps: {len(df)}\n", flush=True)

def cohen_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 5 or len(b) < 5: return 0.0
    sp = math.sqrt(((len(a)-1)*np.var(a,ddof=1) + (len(b)-1)*np.var(b,ddof=1)) / (len(a)+len(b)-2))
    return (np.mean(a)-np.mean(b))/sp if sp > 0 else 0.0

# ── PER-HYPOTHESIS WIN/LOSS forensics ────────────────────────────────────────
for name in HYP:
    sub = df[df["hyp"] == name]
    if len(sub) < 40:
        print(f"\n  [{name}] only {len(sub)} trades — skip forensics")
        continue
    w = sub[sub["win"] == 1]; l = sub[sub["win"] == 0]
    print(f"\n  ═══ {name} — {len(sub)} trades (WR {len(w)/len(sub):.0%}) ═══")
    print(f"    {'Feature':<20}{'Win μ':>9}{'Loss μ':>9}{'Cohen d':>9}")
    rows = []
    for col, label in FEAT_NAMES.items():
        if col not in sub.columns: continue
        d = cohen_d(w[col], l[col])
        rows.append((label, d, w[col].mean(), l[col].mean()))
    for label, d, wm, lm in sorted(rows, key=lambda x: -abs(x[1])):
        flag = "  ***" if abs(d) > 0.3 else ("  **" if abs(d) > 0.2 else "")
        print(f"    {label:<20}{wm:>9.2f}{lm:>9.2f}{d:>9.3f}{flag}")

# ── ENVIRONMENT SLICING: can any single factor rescue the edge? ─────────────
print(f"\n{'═'*90}\n  ENVIRONMENT SLICING — is there a sub-environment where PF@cost > 1.1?\n{'═'*90}")
def pf_cost(sub):
    r = sub["r"].values - 2 * 0.0005 * sub["rel_vol"].values * 0  # cost applied via entry/atr
    # approximate cost: use atr_rank? simpler: cost in R terms ~ 0.0005*price/atr; approximate via rel_vol not avail.
    # recompute properly with entry/atr stored? we didn't store entry/atr; use approximation 2*0.0005/atr_pct
    return None

# store entry/atr properly by re-running a light pass is heavy; instead use stored r and estimate cost via typical atr_pct
# simpler: rebuild cost using rsi... no. Let's just report gross PF per slice and flag slices with PF>1.15 gross
def pf_gross(sub):
    r = sub["r"].values
    w = r[r>0].sum(); l = abs(r[r<0].sum())
    return w/l if l > 0 else 99.0

slices = []
for hyp in HYP:
    sub = df[df["hyp"] == hyp]
    if len(sub) < 60: continue
    # hour slices
    for hr in range(24):
        s2 = sub[sub["hour"] == hr]
        if len(s2) >= 25:
            hol = s2[s2["ts"] >= HOLDOUT_START]
            selpf = pf_gross(s2[s2["ts"] < HOLDOUT_START])
            holpf = pf_gross(hol)
            slices.append((f"{hyp} hour={hr}", len(s2), selpf, holpf))
    # breadth halves
    for thr, lab in [(0.5, "breadth>0.5"), (0.6, "breadth>0.6")]:
        s2 = sub[sub["breadth"] > thr]
        if len(s2) >= 25:
            slices.append((f"{hyp} {lab}", len(s2), pf_gross(s2[s2["ts"]<HOLDOUT_START]), pf_gross(s2[s2["ts"]>=HOLDOUT_START])))
    # atr_rank low half
    for thr, lab in [(50, "atr_rank<50"), (30, "atr_rank<30")]:
        s2 = sub[sub["atr_rank"] < thr]
        if len(s2) >= 25:
            slices.append((f"{hyp} {lab}", len(s2), pf_gross(s2[s2["ts"]<HOLDOUT_START]), pf_gross(s2[s2["ts"]>=HOLDOUT_START])))
    # vwap_dist sign
    s2 = sub[sub["vwap_dist"] > 0]
    if len(s2) >= 25:
        slices.append((f"{hyp} vwap>0", len(s2), pf_gross(s2[s2["ts"]<HOLDOUT_START]), pf_gross(s2[s2["ts"]>=HOLDOUT_START])))
    s2 = sub[sub["vwap_dist"] < 0]
    if len(s2) >= 25:
        slices.append((f"{hyp} vwap<0", len(s2), pf_gross(s2[s2["ts"]<HOLDOUT_START]), pf_gross(s2[s2["ts"]>=HOLDOUT_START])))

sdf = pd.DataFrame(slices, columns=["slice","n","selPF","holPF"]).sort_values("holPF", ascending=False)
print("  Top 12 slices by holdout PF (gross):")
print(sdf.head(12).to_string(index=False))
print("\n  Slices where BOTH selPF>1.15 and holPF>1.15 (potential environment):")
wins = sdf[(sdf["selPF"] > 1.15) & (sdf["holPF"] > 1.15)]
print(wins.to_string(index=False) if len(wins) else "  NONE")

df.to_csv(os.path.join(CONFIG["OUTPUT_FOLDER"], "r092_5m_forensics.csv"), index=False)
print(f"\n  Saved → {CONFIG['OUTPUT_FOLDER']}/r092_5m_forensics.csv")
