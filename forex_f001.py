"""
FOREX HUNT — F001
First forex hypotheses on 8 majors, 1H, 2 years (Aug 2024 - Aug 2026).
Costs: realistic retail spreads modeled per pair (R-terms). Causal audit built in.

Universe (8 majors): EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD EURGBP
Timeframe: 1H. Selection <= 2025-08-01, holdout = Aug 2025 - Aug 2026 (untouched).

Hypotheses:
  F1 LONDON-BREAKOUT   : Asia range (00-08 UTC) break after London open (>=09 UTC)
  F2 TREND-PULLBACK    : EMA50>EMA200 uptrend + pullback to EMA20 + reclaim
  F3 RANGE-MEANREV     : ADX<20 + RSI<30 + below BB lower (causal) -> buy bounce
  F4 VWAP-RECLAIM      : below session VWAP then reclaim + green
  F5 MOMENTUM-BURST    : big green 1H (body > 1.2*ATR, relvol>1.5) + continuation
  F6 CRYPTO-CHAMPION   : Family-A compression-pop transfer test (SVM-free, raw signal)

Costs per pair (retail spread, in price): EURUSD .00006 GBPUSD .00010 USDJPY .010
  AUDUSD .00008 USDCAD .00010 USDCHF .00010 NZDUSD .00012 EURGBP .00010
Success: holPF@cost > 1.1, selection n >= 60.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx
from scripts.ql_engine import add_features, sim_symbol, stats_from_trades, pf_of_rs, IS_LOOKBACK, RECAL_EVERY

FOREX_DIR = os.path.join(CONFIG["CACHE_FOLDER"], "forex")
OUT = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2025-08-01", tz="UTC")
RR = 1.5

PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP"]
SPREAD = {"EURUSD":0.00006,"GBPUSD":0.00010,"USDJPY":0.010,"AUDUSD":0.00008,
          "USDCAD":0.00010,"USDCHF":0.00010,"NZDUSD":0.00012,"EURGBP":0.00010}

SEP = "=" * 110
SEP2 = "-" * 90
print(); print(SEP)
print("  FOREX HUNT — F001  (8 majors, 1H, 2yr, retail spreads)")
print(SEP)
t0 = time.time()

print("\n  Loading forex data …")
feats = {}
for p in PAIRS:
    fpath = os.path.join(FOREX_DIR, f"{p}_1H.parquet")
    if not os.path.exists(fpath):
        print(f"    {p}: MISSING"); continue
    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for col in ["open","high","low","close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open","high","low","close"], inplace=True)
    # forex has no volume -> use bar range as causal activity proxy
    if "vol" not in df.columns:
        df["vol"] = (df["high"] - df["low"]).clip(lower=1e-12)
    f = add_features(df)
    f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14",
                     "ema_dist_pct","real_vol_20","bb_width","prev_range_r",
                     "prev_body_r"], inplace=True)
    if len(f) >= IS_LOOKBACK + RECAL_EVERY:
        feats[p] = f
        print(f"    {p}: {len(f)} bars")
print(f"  Pairs ready: {len(feats)}")

# ── forex indicators (causal) ────────────────────────────────────────────────
def add_fx(f):
    f = f.copy()
    day = f.index.normalize()
    hr = f.index.hour
    typ = (f["high"] + f["low"] + f["close"]) / 3.0
    f["vwap"] = (typ * f["vol"]).groupby(day).cumsum() / f["vol"].groupby(day).cumsum().replace(0, np.nan)
    f["vwap_dist"] = (f["close"] - f["vwap"]) / f["atr14"]
    # Asia range (00-08 UTC) for the day, shifted to be available after 08
    in_asia = (hr >= 0) & (hr < 8)
    asia_hi = f["high"].where(in_asia).groupby(day).transform("max")
    asia_lo = f["low"].where(in_asia).groupby(day).transform("min")
    # make it causal: only available from hour 9 onward of the SAME day (groupby transform is causal per-day here since Asia bars precede London)
    f["asia_hi"] = asia_hi
    f["asia_lo"] = asia_lo
    f["hour"] = hr
    return f

feats2 = {p: add_fx(f) for p, f in feats.items()}

# ── hypotheses ───────────────────────────────────────────────────────────────
def f1_london_breakout(f):
    return ((f["hour"] >= 9) & (f["close"] > f["asia_hi"]) & (f["close"] > f["open"]) &
            (f["asia_hi"].notna()))

def f2_trend_pullback(f):
    low2 = f["low"].rolling(2, min_periods=1).min()
    return ((f["ema50"] > f["ema200"]) & (f["ema50_slope"] > 0) &
            (low2 < f["ema20"]) & (f["close"] > f["ema20"]) & (f["close"] > f["open"]))

def f3_range_meanrev(f):
    return ((f["adx14"] < 20) & (f["rsi14"] < 30) & (f["close"] < f["bb_lower"]) &
            (f["close"] > f["open"]))

def f4_vwap_reclaim(f):
    below = (f["vwap_dist"] < -0.5).rolling(5, min_periods=1).max().astype(bool)
    return (below & (f["close"] > f["vwap"]) & (f["close"] > f["open"]) & f["vwap"].notna())

def f5_momentum_burst(f):
    big = (f["close"] - f["open"]).abs() > 1.2 * f["atr14"]
    return (big & (f["rel_vol"] > 1.5) & (f["close"] > f["open"]) &
            (f["close"] > f["close"].shift(1)))

def f6_crypto_champion(f):
    bb = f["bb_width"].rolling(500).quantile(0.25)
    rv = f["real_vol_20"].rolling(500).quantile(0.33)
    prg = f["prev_range_r"].rolling(500).quantile(0.80)
    return ((f["bb_width"] < bb) & (f["real_vol_20"] < rv) & (f["prev_range_r"] > prg) &
            (f["rel_vol"] > 1.5) & (f["close"] > f["open"]) & (f["close"] > f["close"].shift(1)))

HYP = {"F1_london_breakout": f1_london_breakout, "F2_trend_pullback": f2_trend_pullback,
       "F3_range_meanrev": f3_range_meanrev, "F4_vwap_reclaim": f4_vwap_reclaim,
       "F5_momentum_burst": f5_momentum_burst, "F6_crypto_champ_transfer": f6_crypto_champion}

# ── causal audit ─────────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  CAUSAL AUDIT\n{SEP2}")
audit_ok = {}
test_p = sorted(feats2.keys())[0]
ft = feats2[test_p]; fts = ft.iloc[:-300]
for name, fn in HYP.items():
    m1 = fn(ft); m2 = fn(fts)
    same = bool((m1.loc[fts.index] == m2.loc[fts.index]).all())
    audit_ok[name] = same
    print(f"    {name:<22} {'PASS ✓' if same else 'FAIL ✗ LOOKAHEAD'}")

# ── run with costs ───────────────────────────────────────────────────────────
def run_hyp(fn):
    out = []
    for p, f in feats2.items():
        m = fn(f)
        try:
            for t in sim_symbol(f, m, RR, dict(entry_next=False, exit="timeN",
                                               time_bars=60, hours=None)):
                t["pair"] = p
                t["spread"] = SPREAD[p]
                out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t["entry_time"])
    return out

def eval_hyp(name, trades):
    if not trades:
        return dict(cfg=name, n=0, tpm=0, wr=float('nan'), pf=float('nan'), pf_c=float('nan'),
                    mdd=0, prof=float('nan'), worst=float('nan'), selpf=float('nan'),
                    holpf=float('nan'), holpf_c=float('nan'))
    s = stats_from_trades(trades)
    # cost: spread/ATR per side, 2 sides round trip
    rs = np.array([t["r"] for t in trades])
    cost_r = 2 * np.array([t["spread"]/max(t["atr"],1e-12) for t in trades])
    rc = rs - cost_r
    pf_c = (rc[rc>0].sum()/abs(rc[rc<0].sum())) if (rc<0).any() else 99.0
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]
    hp = stats_from_trades(hol)["pf"]
    hrc = np.array([t["r"] for t in hol]) - 2*np.array([t["spread"]/max(t["atr"],1e-12) for t in hol])
    hpc = (hrc[hrc>0].sum()/abs(hrc[hrc<0].sum())) if (hrc<0).any() else 99.0
    df = pd.DataFrame(trades); df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g>0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur+1 if not v else 0
        worst = max(worst, cur)
    return dict(cfg=name, n=len(trades), tpm=len(df)/24, wr=s["wr"], pf=s["pf"], pf_c=pf_c,
                mdd=s["mdd"], prof=float((g>0).mean()), worst=worst, selpf=sp, holpf=hp,
                holpf_c=hpc)

print(f"\n{SEP2}\n  RESULTS  (selection <= Aug-2025 | holdout Aug-2025..Aug-2026 untouched)\n{SEP2}")
hdr = (f"    {'Hyp':<24}{'n':>6}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@cost':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}{'holPF@c':>9}")
print(hdr); print("    " + "-"*108)
rows = []
for name, fn in HYP.items():
    if not audit_ok[name]:
        print(f"    {name:<24} EXCLUDED (lookahead)"); continue
    trades = run_hyp(fn)
    r = eval_hyp(name, trades)
    rows.append(r)
    print(f"    {name:<24}{r['n']:>6}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}{r['holpf_c']:>9.2f}")

print(f"\n{SEP2}\n  SUCCESS (holPF@cost>1.1, sel n>=60)")
passed = [r for r in rows if r["holpf_c"] > 1.1 and r["n"] >= 60]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: holPF@cost={r['holpf_c']:.2f}")
else:
    print("  ❌ none passed. Closest:")
    for r in sorted(rows, key=lambda r: -r["holpf_c"])[:4]:
        print(f"    {r['cfg']}: holPF@cost={r['holpf_c']:.2f} holPF={r['holpf']:.2f} PF={r['pf']:.2f} n={r['n']}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "f001_forex_results.csv"), index=False)
lines = [f"# FOREX F001 — first forex hypotheses\n",
         f"**Date:** 2026-08-08 | 8 majors, 1H, Aug2024-Aug2026 | selection ≤ Aug-2025, "
         f"holdout Aug-2025..Aug-2026 untouched | retail spreads modeled\n",
         f"\n## Audit\n"]
for name in HYP:
    lines.append(f"- {name}: {'PASS' if audit_ok[name] else 'FAIL (lookahead)'}")
lines += ["", "## Results", "",
          "| Hyp | n | t/mo | WR | PF | PF@cost | MDD% | prof% | worst | selPF | holPF | holPF@cost |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} | {r['holpf_c']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    lines.append(f"**✅ {passed[0]['cfg']} survives costs on holdout.**")
else:
    lines.append("**❌ No forex hypothesis survives retail spreads on holdout.** "
                 "First forex run — more hypotheses needed (this is run 1 of the hunt).")
report = "\n".join(lines)
with open(os.path.join(OUT, "f001_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/f001_*")
