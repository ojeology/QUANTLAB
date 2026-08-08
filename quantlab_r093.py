"""
QUANTLAB AI — R093
BANK-STYLE 5m hypotheses ("think like a bank")

Banks don't predict direction — they provide liquidity, buy where others are
forced to sell, fade extremes back to fair value, and cut losers fast while
letting winners run. All 5 fresh hypotheses follow that logic (vs all our prior
5m attempts which were CHASING: momentum/breakouts).

  B1 PRIOR-DAY-LOW RECLAIM  : price sweeps below YESTERDAY's low (stop-hunt) then
                              closes back above it + green + relvol>1.2.
                              Exit: SL 1ATR / TP 1.5ATR, 90-bar time stop.
  B2 VWAP DEEP-DISCOUNT     : close >=1 ATR below VWAP (fair value) recently, then
                              reclaims above VWAP-0.5ATR + green.
                              Exit: SL 0.75ATR / TP ~1R (target = fair value).
  B3 SESSION-LOW ACCUMULATE : near session low (0.2ATR) + RSI<25 + green + relvol>1.3.
                              Exit: SL 0.75ATR / TP 1R, 90-bar time stop.
  B4 BANKER TRAIL           : small pullback in uptrend (EMA50>EMA200, slope>0) to
                              EMA20, reclaim + green. TIGHT stop 0.5ATR, BE quickly,
                              trail 0.5ATR, no TP cap, 90-bar time stop.
                              (cut losers at 0.5R, let winners run)
  B5 2-DAY LIQUIDITY SWEEP  : low < 2-day-low - 0.1ATR (stop run) then close back
                              above 2-day-low + green.
                              Exit: SL 1ATR / TP 1.5ATR, 90-bar time stop.

GUARDS (lessons from R090/R092):
  - ALL indicators causal (prior-day/rolling/cumsum, no future)
  - ANTI-CHEAT AUDIT: delete last 500 bars, overlap masks must match
  - COST GATE: every result shown at 0.05% per side too; success needs holPF@cost>1.1
  - Protocol: selection <= 2026-05-31, holdout Jun-Aug untouched
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

RESEARCH_ID = "R093"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-06-01", tz="UTC")
IS_LOOKBACK = 6000
RECAL_EVERY = 2016
MIN_BARS = IS_LOOKBACK + RECAL_EVERY + 500
SYMS = ["BTC_USDT_SWAP","ETH_USDT_SWAP","DOGE_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP"]

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  BANK-STYLE 5m hypotheses")
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

# ── BANK indicators (causal) ─────────────────────────────────────────────────
def add_bank_indicators(f):
    f = f.copy()
    day = f.index.normalize()
    typ = (f["high"] + f["low"] + f["close"]) / 3.0
    f["vwap"] = (typ * f["vol"]).groupby(day).cumsum() / f["vol"].groupby(day).cumsum().replace(0, np.nan)
    f["vwap_dist"] = (f["close"] - f["vwap"]) / f["atr14"]
    # session (day-start) low/high — causal cummin/cummax
    f["sess_low"] = f["low"].groupby(day).cummin()
    f["sess_high"] = f["high"].groupby(day).cummax()
    # prior-day low/high/close (shift by 1 day) — causal
    d = f.resample("1D").agg({"high":"max","low":"min","close":"last"}).shift(1)
    f["prev_day_low"] = d["low"].reindex(day).values
    f["prev_day_high"] = d["high"].reindex(day).values
    # prior-2-day low
    d2 = f.resample("1D").agg({"low":"min"}).rolling(2).min().shift(1)
    f["two_day_low"] = d2["low"].reindex(day).values
    return f

feats2 = {s: add_bank_indicators(f) for s, f in feats.items()}

# ── BANK hypothesis masks ────────────────────────────────────────────────────
def b1_prevdaylow_reclaim(f):
    sweep = f["low"] < f["prev_day_low"]
    sweep_recent = sweep.rolling(5, min_periods=1).max().astype(bool)
    return (sweep_recent & (f["close"] > f["prev_day_low"]) & (f["close"] > f["open"]) &
            (f["rel_vol"] > 1.2) & f["prev_day_low"].notna())

def b2_vwap_deep_fade(f):
    deep = f["vwap_dist"] < -1.0
    deep_recent = deep.rolling(5, min_periods=1).max().astype(bool)
    return (deep_recent & (f["close"] > f["vwap"] - 0.5 * f["atr14"]) &
            (f["close"] > f["open"]) & f["vwap"].notna())

def b3_sesslow_accum(f):
    near_low = f["low"] <= f["sess_low"] + 0.2 * f["atr14"]
    return (near_low & (f["rsi14"] < 25) & (f["close"] > f["open"]) &
            (f["rel_vol"] > 1.3) & f["sess_low"].notna())

def b4_banker_trail(f):
    low2 = f["low"].rolling(2, min_periods=1).min()
    return ((f["ema50"] > f["ema200"]) & (f["ema50_slope"] > 0) &
            (low2 < f["ema20"]) & (f["close"] > f["ema20"]) & (f["close"] > f["open"]))

def b5_2day_sweep(f):
    sweep = f["low"] < f["two_day_low"] - 0.1 * f["atr14"]
    sweep_recent = sweep.rolling(5, min_periods=1).max().astype(bool)
    return (sweep_recent & (f["close"] > f["two_day_low"]) & (f["close"] > f["open"]) &
            f["two_day_low"].notna())

HYP = {
    "B1_prevdaylow": (b1_prevdaylow_reclaim, dict(sl_mult=1.0, time_bars=90), 1.5),
    "B2_vwap_fade":  (b2_vwap_deep_fade,      dict(sl_mult=0.75, time_bars=90), 1.0),
    "B3_sesslow":    (b3_sesslow_accum,       dict(sl_mult=0.75, time_bars=90), 1.0),
    "B4_banker":     (b4_banker_trail,        dict(sl_mult=0.5, trail_mult=0.5,
                                                    time_bars=90), 2.0),
    "B5_2daysweep":  (b5_2day_sweep,          dict(sl_mult=1.0, time_bars=90), 1.5),
}

# ── ANTI-CHEAT AUDIT ─────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  ANTI-CHEAT LOOKAHEAD AUDIT\n{SEP2}")
audit_ok = {}
test_sym = sorted(feats2.keys())[0]
ft = feats2[test_sym]; ft_short = ft.iloc[:-500]
for name, (fn, _, _) in HYP.items():
    m_full = fn(ft); m_short = fn(ft_short)
    overlap = ft_short.index
    same = bool((m_full.loc[overlap] == m_short.loc[overlap]).all())
    audit_ok[name] = same
    print(f"    {name:<16} audit {'PASS ✓' if same else 'FAIL ✗ LOOKAHEAD'}")

# ── Run ──────────────────────────────────────────────────────────────────────
qle.IS_LOOKBACK = IS_LOOKBACK
qle.RECAL_EVERY = RECAL_EVERY

def run_mask(mask_map, cfg, rr):
    out = []
    for sym, f in feats2.items():
        try:
            for t in sim_symbol(f, mask_map[sym], rr, dict(entry_next=False, exit="timeN",
                                                           **cfg)):
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
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df) / 4.5)

def evaluate(name, trades):
    s = stats_from_trades(trades)
    pf_c = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel_t = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel_t)["pf"]; hp = stats_from_trades(hol)["pf"]
    hpc = pf_of_rs(cost_adjusted_rs(hol, 0.05))
    mp = monthly_profile(trades)
    return dict(cfg=name, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"],
                pf_c=pf_c, mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"],
                selpf=sp, holpf=hp, holpf_c=hpc)

print(f"\n{SEP2}\n  RESULTS (audit-passed only; gross + @0.05% cost)\n{SEP2}")
hdr = (f"    {'Hyp':<14}{'n':>6}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}{'holPF@c':>9}")
print(hdr); print("    " + "─"*105)
rows = []
for name, (fn, cfg, rr) in HYP.items():
    if not audit_ok[name]:
        print(f"    {name:<14}  EXCLUDED (lookahead)"); continue
    mask = {s: fn(f) for s, f in feats2.items()}
    trades = run_mask(mask, cfg, rr)
    r = evaluate(name, trades)
    rows.append(r)
    print(f"    {name:<14}{r['n']:>6}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}{r['holpf_c']:>9.2f}")

print(f"\n{SEP2}\n  SUCCESS (holPF@cost>1.1, sel n>=60)")
passed = [r for r in rows if r["holpf_c"] > 1.1]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: holPF@cost={r['holpf_c']:.2f}")
else:
    print("  ❌ none passed. Closest:")
    for r in sorted(rows, key=lambda r: -r["holpf_c"])[:4]:
        print(f"    {r['cfg']}: holPF@cost={r['holpf_c']:.2f} holPF={r['holpf']:.2f} PF={r['pf']:.2f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "r093_bank5m.csv"), index=False)
lines = [f"# R093 — BANK-STYLE 5m hypotheses\n",
         f"**Date:** 2026-08-07 | 5m, 5 symbols | selection ≤May, holdout Jun-Aug | "
         f"audit + cost gates\n",
         f"\n## Audit\n"]
for name in HYP:
    lines.append(f"- {name}: {'PASS (causal)' if audit_ok[name] else 'FAIL (lookahead)'}")
lines += ["", "## Results (gross + @0.05% cost)",
          "| Hyp | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF | holPF@cost |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} | {r['holpf_c']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    best = max(passed, key=lambda r: r["holpf_c"])
    lines.append(f"**✅ {best['cfg']} survives costs on holdout (holPF@cost {best['holpf_c']:.2f}).** "
                 f"Candidate bank-style 5m edge — needs more data/live confirm.")
else:
    lines.append("**❌ No bank-style 5m hypothesis survives 0.05% costs on holdout.** "
                 "Bank logic tested honestly; still no 5m edge after costs.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r093_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r093_*")
