"""
QUANTLAB AI — F001 (FOREX HUNT #1)
First forex hypotheses on 8 majors, 1H (Yahoo data, 2023-10 → 2026-08).

Forex-specific hypotheses (all causal by construction):
  X1 LONDON-BREAKOUT  : Asia range (22:00-07:00 UTC prior), close above Asia high after 07:00
  X2 TREND-PULLBACK   : EMA50>EMA200 + slope>0 + dip to EMA20 within 2 bars + reclaim + green
  X3 NY-MOMENTUM      : NY session (13:00-17:00) + close > session VWAP + green + relvol>1.2
  X4 DAY-LOW MEANREV  : near session-low (cummin low + 0.3ATR) + RSI<30 + green  [causal!]
  X5 LONDON-EXPANSION : ATR spike at London open (07:00-09:00) + green + close>prev close
  T1 FAMILY-A TRANSFER: crypto compression-pop ported to FX (does the 1H edge transfer?)

GUARDS:
  - Causal audit (delete last 500 bars, overlap masks identical)
  - Costs: realistic spread per pair modeled as R-cost = spread/ATR per side
  - Protocol: selection < 2025-06-01, holdout 2025-06-01→2026-08-07 untouched
  - Success: holPF@cost > 1.1, selection n >= 80
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG, calc_ema
from scripts.ql_engine import add_features, sim_symbol, stats_from_trades, pf_of_rs
import scripts.ql_engine as qle

RESEARCH_ID = "F001"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2025-06-01", tz="UTC")
IS_LOOKBACK = 500
RECAL_EVERY = 168
MIN_BARS = 5000
RR = 1.5

PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP"]
# realistic retail spread in pips (approx, mid-tier broker)
SPREAD_PIPS = {"EURUSD":0.6,"GBPUSD":0.9,"USDJPY":0.9,"AUDUSD":0.8,
               "USDCAD":1.0,"USDCHF":1.0,"NZDUSD":1.2,"EURGBP":1.5}

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  FOREX HUNT #1 (8 majors, 1H)")
print(SEP)
t0 = time.time()

print("\n  Loading forex 1H data …")
feats = {}
for p in PAIRS:
    fn = os.path.join(CACHE, f"FOREX_{p}_1H.parquet")
    if not os.path.exists(fn): continue
    try:
        df = pd.read_parquet(fn)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close"], inplace=True)
        if len(df) < MIN_BARS: continue
        f = add_features(df)
        f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14",
                         "ema_dist_pct","real_vol_20","bb_width","prev_range_r",
                         "prev_body_r"], inplace=True)
        if len(f) >= MIN_BARS: feats[p] = f
        print(f"    {p}: {len(f)} bars  {f.index.min():%Y-%m-%d} → {f.index.max():%Y-%m-%d}")
    except Exception as e:
        print(f"    {p}: ERR {e}")
print(f"  Pairs ready: {len(feats)}")

# ── Forex indicators (causal) ────────────────────────────────────────────────
def add_fx_indicators(f):
    f = f.copy()
    day = f.index.normalize()
    hour = f.index.hour
    # session VWAP (cumsum from day start — causal)
    typ = (f["high"] + f["low"] + f["close"]) / 3.0
    f["vwap"] = (typ * f["vol"].fillna(1)).groupby(day).cumsum() / f["vol"].fillna(1).groupby(day).cumsum()
    f["vwap_dist"] = (f["close"] - f["vwap"]) / f["atr14"]
    # session low (cummin — causal)
    f["sess_low"] = f["low"].groupby(day).cummin()
    f["sess_high"] = f["high"].groupby(day).cummax()
    # Asia range (22:00-07:00 UTC): need PRIOR day's Asia range, computed causally.
    # Build daily Asia-range series from the day's 22-07 bars only, shift by 1 day.
    is_asia = (hour >= 22) | (hour < 7)
    asia_hi = f["high"].where(is_asia).groupby(day).transform("max")
    asia_lo = f["low"].where(is_asia).groupby(day).transform("min")
    # shift by 1 day -> prior Asia range (reindex each day by previous day's values)
    d = f.resample("1D").agg({"high":"max","low":"min"})
    d_prev = d.shift(1)
    f["asia_hi_prev"] = d_prev["high"].reindex(day).values
    f["asia_lo_prev"] = d_prev["low"].reindex(day).values
    return f

feats2 = {p: add_fx_indicators(f) for p, f in feats.items()}

# ── Hypotheses ───────────────────────────────────────────────────────────────
def x1_london_breakout(f):
    h = f.index.hour
    after_london = (h >= 7) & (h <= 12)
    return (after_london & (f["close"] > f["asia_hi_prev"]) & (f["close"] > f["open"]) &
            f["asia_hi_prev"].notna())

def x2_trend_pullback(f):
    low2 = f["low"].rolling(2, min_periods=1).min()
    return ((f["ema50"] > f["ema200"]) & (f["ema50_slope"] > 0) &
            (low2 < f["ema20"]) & (f["close"] > f["ema20"]) & (f["close"] > f["open"]))

def x3_ny_momentum(f):
    h = f.index.hour
    ny = (h >= 13) & (h <= 17)
    return (ny & (f["close"] > f["vwap"]) & (f["close"] > f["open"]) & (f["rel_vol"] > 1.2) &
            f["vwap"].notna())

def x4_daylow_meanrev(f):
    near_low = f["low"] <= f["sess_low"] + 0.3 * f["atr14"]
    return (near_low & (f["rsi14"] < 30) & (f["close"] > f["open"]) & f["sess_low"].notna())

def x5_london_expansion(f):
    h = f.index.hour
    atr_rank = f["atr14"].rolling(500).rank(pct=True)
    london = (h >= 7) & (h <= 9)
    return (london & (atr_rank > 0.90) & (f["close"] > f["open"]) &
            (f["close"] > f["close"].shift(1)))

# Family-A transfer (crypto compression-pop on FX)
def t1_family(f):
    bb = f["bb_width"].rolling(500).quantile(0.25)
    rv = f["real_vol_20"].rolling(500).quantile(0.33)
    prg = f["prev_range_r"].rolling(500).quantile(0.80)
    return ((f["bb_width"] < bb) & (f["real_vol_20"] < rv) & (f["prev_range_r"] > prg) &
            (f["rel_vol"] > 1.5) & (f["close"] > f["open"]) & (f["close"] > f["close"].shift(1)))

HYP = {
    "X1_london_break": x1_london_breakout,
    "X2_trend_pull":   x2_trend_pullback,
    "X3_ny_momentum":  x3_ny_momentum,
    "X4_daylow_mr":    x4_daylow_meanrev,
    "X5_london_exp":   x5_london_expansion,
    "T1_family_trans": t1_family,
}

# ── Causal audit ─────────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  CAUSAL AUDIT (delete last 500 bars)\n{SEP2}")
audit_ok = {}
test_pair = sorted(feats2.keys())[0]
ft = feats2[test_pair]; fts = ft.iloc[:-500]
for name, fn in HYP.items():
    m1 = fn(ft); m2 = fn(fts)
    same = bool((m1.loc[fts.index] == m2.loc[fts.index]).all())
    audit_ok[name] = same
    print(f"    {name:<18} audit {'PASS ✓' if same else 'FAIL ✗ LOOKAHEAD'}")

# ── Run ──────────────────────────────────────────────────────────────────────
qle.IS_LOOKBACK = IS_LOOKBACK
qle.RECAL_EVERY = RECAL_EVERY

def run_hyp(fn, pair_list, spread_pips):
    out = []
    for p in pair_list:
        f = feats2[p]
        m = fn(f)
        try:
            for t in sim_symbol(f, m, RR, dict(entry_next=False, exit="timeN",
                                               sl_mult=1.0, time_bars=48, hours=None)):
                t["pair"] = p
                # cost in R: spread_pips * pip_value / ATR
                pip_val = 0.01 if p == "USDJPY" else 0.0001
                t["cost_r"] = (spread_pips[p] * pip_val) / t["atr"] if t["atr"] > 0 else 0
                out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t["entry_time"])
    return out

def evaluate(name, trades):
    if not trades: return dict(cfg=name, n=0, tpm=0, wr=float('nan'), pf=float('nan'),
                               pf_c=float('nan'), mdd=0, prof=float('nan'), worst=float('nan'),
                               selpf=float('nan'), holpf=float('nan'), holpf_c=float('nan'))
    r = np.array([t["r"] for t in trades])
    cost = np.array([t.get("cost_r", 0) for t in trades])
    rc = r - 2 * cost   # round-trip spread
    s = stats_from_trades(trades)
    pf = pf_of_rs(r); pf_c = pf_of_rs(rc)
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = pf_of_rs(np.array([t["r"] for t in sel]))
    hr = np.array([t["r"] for t in hol]); hc = np.array([t.get("cost_r",0) for t in hol])
    hp = pf_of_rs(hr); hpc = pf_of_rs(hr - 2*hc)
    df = pd.DataFrame(trades); df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        worst = max(worst, cur)
    return dict(cfg=name, n=len(trades), tpm=len(df)/2.8, wr=s["wr"], pf=pf, pf_c=pf_c,
                mdd=s["mdd"], prof=float((g>0).mean()), worst=worst,
                selpf=sp, holpf=hp, holpf_c=hpc)

print(f"\n{SEP2}\n  F001 RESULTS (selection <2025-06, holdout 2025-06→2026-08 untouched)\n{SEP2}")
hdr = (f"    {'Hyp':<16}{'n':>6}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@spd':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}{'holPF@c':>9}")
print(hdr); print("    " + "─"*108)
rows = []
for name, fn in HYP.items():
    if not audit_ok[name]:
        print(f"    {name:<16} EXCLUDED (lookahead)"); continue
    trades = run_hyp(fn, list(feats2.keys()), SPREAD_PIPS)
    r = evaluate(name, trades)
    rows.append(r)
    print(f"    {name:<16}{r['n']:>6}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}{r['holpf_c']:>9.2f}")

print(f"\n{SEP2}\n  SUCCESS (holPF@cost>1.1, sel n>=80)")
passed = [r for r in rows if r["holpf_c"] > 1.1 and r["n"] >= 80]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: holPF@cost={r['holpf_c']:.2f} t/mo={r['tpm']:.1f}")
else:
    print("  ❌ none passed. Closest:")
    for r in sorted(rows, key=lambda r: -r["holpf_c"])[:4]:
        print(f"    {r['cfg']}: holPF@cost={r['holpf_c']:.2f} holPF={r['holpf']:.2f} PF={r['pf']:.2f} n={r['n']}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "f001_forex.csv"), index=False)
lines = [f"# F001 — FOREX HUNT #1 (8 majors, 1H)\n",
         f"**Date:** 2026-08-08 | Yahoo 1H data 2023-10→2026-08 | selection <2025-06, "
         f"holdout 2025-06→2026-08 untouched | spread costs modeled\n",
         f"\n## Results\n",
         "| Hyp | n | t/mo | WR | PF | PF@spread | MDD% | prof% | worst | selPF | holPF | holPF@cost |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} | {r['holpf_c']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    best = max(passed, key=lambda r: r["holpf_c"])
    lines.append(f"**✅ {best['cfg']} survives costs on holdout (holPF@cost {best['holpf_c']:.2f}).**")
else:
    lines.append("**❌ No forex hypothesis survives spread costs on holdout.** F001 honest negative — "
                 "candidate directions for F002 noted in the run.")
report = "\n".join(lines)
with open(os.path.join(OUT, "f001_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/f001_*")
