"""
QUANTLAB AI — R090
FRESH 5-MINUTE HYPOTHESES (not the 1H signal ported)

5 fresh setups designed FOR the 5m scale:
  H1 MOMENTUM-BURST   : close > prior 10-bar high + relvol>2.0 + green
                        (breakout continuation at 5m speed)
  H2 RANGE-FADE       : price within 0.2*ATR of today's high/low + RSI2 extreme
                        (fade the 5m extreme — mean reversion)
  H3 ORB-5M           : first 120 min range, break above it after 09:00 UTC-ish
  H4 TREND-PULLBACK-5M: fast uptrend (EMA50>EMA200, slope>0) + dip to EMA20 +
                        reclaim (scaled pullback)
  H5 VOL-BURST        : ATR percentile > 90 + close>prev close (vol expansion)
                        then hold for continuation

Exits: E6 entry, RR 1.5, base SL/TP, 60-bar time stop (5h).
Strict: selection <= 2026-05-31, holdout Jun-Aug untouched, costs 0.05%.
Success: holPF > 1.1, PF@cost > 1.1, sel n >= 60.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, sim_symbol, stats_from_trades, cost_adjusted_rs, pf_of_rs
import scripts.ql_engine as qle

RESEARCH_ID = "R090"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-06-01", tz="UTC")
RR = 1.5
IS_LOOKBACK = 6000
RECAL_EVERY = 2016
MIN_BARS = IS_LOOKBACK + RECAL_EVERY + 500
SYMS = ["BTC_USDT_SWAP","ETH_USDT_SWAP","DOGE_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP"]

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  Fresh 5m Hypotheses")
print(SEP)
t0 = time.time()

print("\n  Loading 5m data …")
feats = {}
for sym in SYMS:
    p = os.path.join(CACHE, f"{sym}_5m.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < MIN_BARS: continue
        f = add_features(df)
        f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14",
                         "ema_dist_pct","real_vol_20","bb_width","prev_range_r",
                         "prev_body_r"], inplace=True)
        if len(f) >= MIN_BARS: feats[sym] = f
        print(f"    {sym}: {len(f)} bars")
    except Exception as e:
        print(f"    {sym}: ERR {e}")
print(f"  Symbols ready: {len(feats)}")

qle.IS_LOOKBACK = IS_LOOKBACK
qle.RECAL_EVERY = RECAL_EVERY

# ── Fresh 5m hypothesis masks ────────────────────────────────────────────────
def h1_momentum_burst(f):
    prior_hi = f["high"].rolling(10).max().shift(1)
    return ((f["close"] > prior_hi) & (f["rel_vol"] > 2.0) &
            (f["close"] > f["open"]) & (f["close"] > f["close"].shift(1)))

def h2_range_fade(f):
    day = f.index.normalize()
    day_hi = f["high"].groupby(day).transform("max")
    day_lo = f["low"].groupby(day).cummin()  # CAUSAL (running low, no future peek)
    near_hi = (f["high"] >= day_hi - 0.2 * f["atr14"])
    near_lo = (f["low"] <= day_lo + 0.2 * f["atr14"])
    rsi2 = f["rsi14"]  # use rsi14 for simplicity, extremes below 20 / above 80
    # fade: short side not supported in sim (long-only engine), so long fade = buy near day low
    return (near_lo & (f["rsi14"] < 30) & (f["close"] > f["open"]))

def h3_orb_5m(f):
    # ORB: first 120 min of UTC day; break above range high after hour 2
    day = f.index.normalize()
    hr = f.index.hour
    in_orb = (hr >= 0) & (hr < 2)
    orb_hi = f["high"].where(in_orb).groupby(day).transform("max")
    orb_lo = f["low"].where(in_orb).groupby(day).transform("min")
    rng = (orb_hi - orb_lo) >= 0.5 * f["atr14"]
    return ((hr >= 2) & (f["close"] > orb_hi) & rng & ~orb_hi.isna())

def h4_trend_pullback_5m(f):
    low3 = f["low"].rolling(3, min_periods=1).min()
    return ((f["ema50"] > f["ema200"]) & (f["ema50_slope"] > 0) &
            (low3 < f["ema20"]) & (f["close"] > f["ema20"]) & (f["close"] > f["open"]))

def h5_vol_burst(f):
    atr_pct = (f["atr14"] / f["close"]).rolling(500).rank(pct=True)
    return ((atr_pct > 0.90) & (f["close"] > f["close"].shift(1)) &
            (f["close"] > f["open"]) & (f["rel_vol"] > 1.5))

HYP = {
    "H1_momentum_burst": h1_momentum_burst,
    "H2_range_fade":     h2_range_fade,
    "H3_orb_5m":         h3_orb_5m,
    "H4_trend_pullback": h4_trend_pullback_5m,
    "H5_vol_burst":      h5_vol_burst,
}

def run_mask(mask_map):
    out = []
    for sym, f in feats.items():
        try:
            for t in sim_symbol(f, mask_map[sym], RR, dict(entry_next=False, exit="timeN",
                                                           time_bars=60, hours=None)):
                t["sym"] = sym; out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t["entry_time"])
    return out

def monthly_profile(trades):
    if not trades: return dict(prof=float("nan"), worst=float("nan"), tpm=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        worst = max(worst, cur)
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df) / 6.0)

def evaluate(name, trades):
    s = stats_from_trades(trades)
    pf_c = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel_t = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel_t)["pf"]; hp = stats_from_trades(hol)["pf"]
    mp = monthly_profile(trades)
    return dict(cfg=name, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"],
                pf_c=pf_c, mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"],
                selpf=sp, holpf=hp)

print("\n  Running 5 fresh hypotheses …")
results = {}
for name, fn in HYP.items():
    mask = {s: fn(f) for s, f in feats.items()}
    results[name] = run_mask(mask)
    print(f"    {name}: {len(results[name])} raw signals→trades")

print(f"\n{SEP2}")
print("  FRESH 5m HYPOTHESES  (selection Jan-May, holdout Jun-Aug untouched)")
hdr = (f"    {'Hypothesis':<18}{'n':>6}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}")
print(hdr); print("    " + "─"*100)
rows = []
for name in HYP:
    r = evaluate(name, results[name])
    rows.append(r)
    print(f"    {name:<18}{r['n']:>6}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}")

print(f"\n{SEP2}")
print("  SUCCESS (holPF>1.1, PF@cost>1.1, sel n>=60)")
passed = [r for r in rows if r["holpf"] > 1.1 and r["pf_c"] > 1.1 and
          stats_from_trades([t for t in results[r['cfg']] if t["entry_time"] < HOLDOUT_START])["n"] >= 60]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: holPF={r['holpf']:.2f} PF@.05={r['pf_c']:.2f}")
else:
    print("  ❌ none passed. Closest:")
    for r in sorted(rows, key=lambda r: -r["holpf"])[:3]:
        print(f"    {r['cfg']}: holPF={r['holpf']:.2f} PF={r['pf']:.2f} PF@.05={r['pf_c']:.2f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "r090_5m_hypotheses.csv"), index=False)
lines = [f"# R090 — Fresh 5m Hypotheses (not 1H port)\n",
         f"**Date:** 2026-08-07 | 5m, 5 symbols, selection ≤May31, holdout Jun-Aug\n",
         f"\n## Results\n",
         "| Hypothesis | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    lines.append(f"**✅ {passed[0]['cfg']} passes — candidate 5m edge.**")
else:
    lines.append("**❌ No fresh 5m hypothesis passes.** Honest negative result on 5m.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r090_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r090_*")
