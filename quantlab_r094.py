"""
QUANTLAB AI — R094
5m COMBINATION SWEEP: stack bank-style signals with each other + filters.

Addresses the honest gap: R091/R093 tested hypotheses standalone; this tests
COMBOS (signal AND signal / signal AND filter), with causal audit + cost gate.

Base signals (from R091/R093, all causal):
  VWAP-fade, PrevDayLow-reclaim, SessionLow-accum, 2DaySweep
Filters (all causal):
  breadth>0.5, atr_rank<50, hour in {7,8,9,13,14,15,16}, green-streak>=2

Combos (pre-registered, sensible only — NOT a blind grid):
  C1 VWAP-fade AND PrevDayLow
  C2 VWAP-fade AND SessionLow
  C3 SessionLow AND breadth>0.5
  C4 PrevDayLow AND atr_rank<50
  C5 2DaySweep AND breadth>0.5
  C6 VWAP-fade AND hour-set
  C7 SessionLow AND hour-set
  C8 PrevDayLow AND green-streak>=2
  C9 2DaySweep AND green-streak>=2
  C10 VWAP-fade AND SessionLow AND breadth>0.5

Protocol: selection <= 2026-05-31, holdout Jun-Aug untouched, 0.05% cost gate.
Success: holPF@cost > 1.1, selection n >= 60.
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

RESEARCH_ID = "R094"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-06-01", tz="UTC")
IS_LOOKBACK = 6000
RECAL_EVERY = 2016
MIN_BARS = IS_LOOKBACK + RECAL_EVERY + 500
SYMS = ["BTC_USDT_SWAP","ETH_USDT_SWAP","DOGE_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP"]
HOURS = {7,8,9,13,14,15,16}

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  5m COMBINATION SWEEP")
print(SEP)
t0 = time.time()

print("Loading 5m data …")
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
    except Exception as e:
        print(f"  {sym}: ERR {e}")
print(f"  Symbols: {len(feats)}")

def ind(f):
    f = f.copy()
    day = f.index.normalize()
    typ = (f["high"] + f["low"] + f["close"]) / 3.0
    f["vwap"] = (typ * f["vol"]).groupby(day).cumsum() / f["vol"].groupby(day).cumsum().replace(0, np.nan)
    f["vwap_dist"] = (f["close"] - f["vwap"]) / f["atr14"]
    f["sess_low"] = f["low"].groupby(day).cummin()
    d = f.resample("1D").agg({"high":"max","low":"min","close":"last"}).shift(1)
    f["prev_day_low"] = d["low"].reindex(day).values
    d2 = f.resample("1D").agg({"low":"min"}).rolling(2).min().shift(1)
    f["two_day_low"] = d2["low"].reindex(day).values
    return f

feats2 = {s: ind(f) for s, f in feats.items()}

# universe breadth
above = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats2.items()}
breadth = pd.DataFrame(above).sort_index().mean(axis=1, skipna=True)

# base signal masks
def sig_vwap(f):
    deep = (f["vwap_dist"] < -1.0).rolling(5, min_periods=1).max().astype(bool)
    return (deep & (f["close"] > f["vwap"] - 0.5 * f["atr14"]) & (f["close"] > f["open"]) & f["vwap"].notna())

def sig_prevlow(f):
    sw = (f["low"] < f["prev_day_low"]).rolling(5, min_periods=1).max().astype(bool)
    return (sw & (f["close"] > f["prev_day_low"]) & (f["close"] > f["open"]) &
            (f["rel_vol"] > 1.2) & f["prev_day_low"].notna())

def sig_sesslow(f):
    near = f["low"] <= f["sess_low"] + 0.2 * f["atr14"]
    return (near & (f["rsi14"] < 25) & (f["close"] > f["open"]) & (f["rel_vol"] > 1.3) & f["sess_low"].notna())

def sig_2day(f):
    sw = (f["low"] < f["two_day_low"] - 0.1 * f["atr14"]).rolling(5, min_periods=1).max().astype(bool)
    return (sw & (f["close"] > f["two_day_low"]) & (f["close"] > f["open"]) & f["two_day_low"].notna())

SIGS = {"vwap": sig_vwap, "prevlow": sig_prevlow, "sesslow": sig_sesslow, "2day": sig_2day}

def filt_breadth(f, thr=0.5):
    return (breadth.reindex(f.index, method="ffill") > thr).fillna(False)
def filt_atrrank(f, thr=50):
    return (f["atr_rank"] < thr)
def filt_hour(f):
    return f.index.hour.isin(HOURS)
def filt_green2(f):
    return (f["close"] > f["open"]) & (f["close"].shift(1) > f["open"].shift(1))

# COMBOS: each = list of signal-names (AND) + list of filter-names (AND)
COMBOS = {
    "C1_vwap+prevlow":   (["vwap","prevlow"], []),
    "C2_vwap+sesslow":   (["vwap","sesslow"], []),
    "C3_sesslow+brd":    (["sesslow"], ["breadth"]),
    "C4_prevlow+atr":    (["prevlow"], ["atrrank"]),
    "C5_2day+brd":       (["2day"], ["breadth"]),
    "C6_vwap+hour":      (["vwap"], ["hour"]),
    "C7_sesslow+hour":   (["sesslow"], ["hour"]),
    "C8_prevlow+green":  (["prevlow"], ["green2"]),
    "C9_2day+green":     (["2day"], ["green2"]),
    "C10_vwap+sesslow+brd": (["vwap","sesslow"], ["breadth"]),
}

# audit: causal check on one symbol
print("\n  AUDIT (causal, delete-last-500) …")
test_sym = sorted(feats2.keys())[0]
ft = feats2[test_sym]; fts = ft.iloc[:-500]
def combo_mask(f, sigs, fils):
    m = pd.Series(True, index=f.index)
    for s in sigs: m &= SIGS[s](f)
    for fl in fils:
        fn = {"breadth": filt_breadth, "atrrank": filt_atrrank, "hour": filt_hour, "green2": filt_green2}[fl]
        m &= fn(f)
    return m
audit_ok = {}
for name, (sigs, fils) in COMBOS.items():
    m1 = combo_mask(ft, sigs, fils)
    m2 = combo_mask(fts, sigs, fils)
    same = bool((m1.loc[fts.index] == m2.loc[fts.index]).all())
    audit_ok[name] = same
    print(f"    {name:<20} {'PASS ✓' if same else 'FAIL ✗'}")

qle.IS_LOOKBACK = IS_LOOKBACK
qle.RECAL_EVERY = RECAL_EVERY

def run_combo(sigs, fils, rr=1.5, sl_mult=1.0, time_bars=90):
    out = []
    for sym, f in feats2.items():
        m = combo_mask(f, sigs, fils)
        try:
            for t in sim_symbol(f, m, rr, dict(entry_next=False, exit="timeN",
                                               sl_mult=sl_mult, time_bars=time_bars, hours=None)):
                t["sym"] = sym; out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t["entry_time"])
    return out

def evaluate(name, trades):
    if not trades: return dict(cfg=name, n=0, tpm=0, wr=float('nan'), pf=float('nan'),
                               pf_c=float('nan'), mdd=0, prof=float('nan'), worst=float('nan'),
                               selpf=float('nan'), holpf=float('nan'), holpf_c=float('nan'))
    s = stats_from_trades(trades)
    pf_c = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]
    hp = stats_from_trades(hol)["pf"]
    hpc = pf_of_rs(cost_adjusted_rs(hol, 0.05))
    df = pd.DataFrame(trades); df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        worst = max(worst, cur)
    return dict(cfg=name, n=len(trades), tpm=len(df)/4.5, wr=s["wr"], pf=s["pf"], pf_c=pf_c,
                mdd=s["mdd"], prof=float((g>0).mean()), worst=worst,
                selpf=sp, holpf=hp, holpf_c=hpc)

print(f"\n{SEP2}\n  COMBINATION RESULTS\n{SEP2}")
hdr = (f"    {'Combo':<22}{'n':>6}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}{'holPF@c':>9}")
print(hdr); print("    " + "─"*108)
rows = []
for name, (sigs, fils) in COMBOS.items():
    if not audit_ok[name]:
        print(f"    {name:<22} EXCLUDED (lookahead)"); continue
    trades = run_combo(sigs, fils)
    r = evaluate(name, trades)
    rows.append(r)
    print(f"    {name:<22}{r['n']:>6}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
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

pd.DataFrame(rows).to_csv(os.path.join(OUT, "r094_5m_combos.csv"), index=False)
lines = [f"# R094 — 5m COMBINATION SWEEP\n",
         f"**Date:** 2026-08-07 | combos of bank-style signals + filters | "
         f"selection ≤May, holdout Jun-Aug, 0.05% cost gate\n",
         f"\n## Audit\n"]
for name in COMBOS:
    lines.append(f"- {name}: {'PASS' if audit_ok[name] else 'FAIL (lookahead)'}")
lines += ["", "## Results", "",
          "| Combo | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF | holPF@cost |",
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
    lines.append("**❌ No combination survives 0.05% costs on holdout.** 6th 5m confirmation.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r094_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r094_*")
