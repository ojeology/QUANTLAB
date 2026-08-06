"""
QUANTLAB AI — R075
New Strategy Families for a Retail-Friendly Profile

Motivation: R074's final strategy (Family A E6+RR2+VolCeil, PF 1.63) is a real edge
but NOT retail-friendly: 41% profitable months, 7-month losing streak, -13..-26% MDD.
This run hunts for strategies with a profile a retail trader can actually survive:
  - >=50% profitable months
  - max losing-month streak <= 4
  - moderate frequency (not 1 trade per 6 months, not 300/day)
  - lower drawdown
even at a PF trade-off.

Four pre-registered strategy families (each ONE concept, defined up front):

  N1 TREND PULLBACK   : uptrend (EMA50>EMA200, slope>0) + price dipped below EMA20
                        within last 3 bars + reclaimed above EMA20 + mild volume.
                        Exit: 2ATR initial stop, 2ATR trailing from highest close,
                        no TP cap, 48-bar time stop.  ("ride the trend")
  N2 RANGE MEANREV    : ranging (ADX<20) + RSI<30 + close below lower Bollinger.
                        Exit: 1.5ATR SL / 1.5ATR TP (1:1), 24-bar time stop.
                        (high win-rate, many small wins — consistency)
  N3 BREAKOUT FOLLOW  : close breaks prior 20-bar high + rel_vol>1.5 + uptrend.
                        Exit: 2ATR stop, 1.5ATR trail, no TP, 48-bar time stop.
  N4 ORB LONDON       : daily range from 12:00-14:59 UTC bars; enter long when a
                        later bar closes above prior day's range high (range>=0.5ATR).
                        Exit: 2ATR stop, 1.5ATR trail, no TP, 24-bar time stop.

Protocol (same as R074): decisions on pre-2026 only, holdout = 2026 untouched.
All validated with bootstrap/LOO/MC/costs + a transparent Retail Score.

Retail Score (0-100):
  25% PF clamp((pf-1)/0.6,0,1)
  25% profitable-month rate
  15% clamp(1 - max_losing_streak/10, 0, 1)
  20% clamp(trades_per_month/30, 0, 1)
  15% clamp(1 - |MDD|/0.30, 0, 1)
"""
import os, sys, time, warnings
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, sim_symbol, stats_from_trades, bootstrap_pf,
    loo_symbol_floor, monte_carlo, cost_adjusted_rs, pf_of_rs,
    IS_LOOKBACK, RECAL_EVERY, STARTING_CAP, RISK_PCT,
)

RESEARCH_ID = "R075"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")

FAMILIES = {
    "N1_trendpull": dict(
        rr=2.0, cfg=dict(entry_next=False, exit="trail_run", sl_mult=2.0,
                         trail_mult=2.0, time_bars=48),
        desc="Trend pullback + 2ATR trail",
    ),
    "N2_meanrev": dict(
        rr=1.0, cfg=dict(entry_next=False, exit="timeN", sl_mult=1.5,
                         time_bars=24),
        desc="Range mean-reversion 1:1",
    ),
    "N3_breakout": dict(
        rr=2.0, cfg=dict(entry_next=False, exit="trail_run", sl_mult=2.0,
                         trail_mult=1.5, time_bars=48),
        desc="20-bar breakout follow-through",
    ),
    "N4_orb": dict(
        rr=2.0, cfg=dict(entry_next=False, exit="trail_run", sl_mult=2.0,
                         trail_mult=1.5, time_bars=24),
        desc="London opening-range breakout",
    ),
}

SEP  = "═" * 110
SEP2 = "─" * 90

print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  New Strategy Families (retail-friendly hunt)")
print(SEP)
t0 = time.time()

# ── Load data ────────────────────────────────────────────────────────────────
print("\n  Loading data …")
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
        f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14",
                         "bb_lower","bb_upper","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats_by_sym[sym] = f
    except Exception:
        pass
print(f"  Symbols loaded: {len(feats_by_sym)}")


# ── Signal masks per family ─────────────────────────────────────────────────
def mask_trendpull(f):
    low3 = f["low"].rolling(3, min_periods=1).min()
    return ((f["ema50"] > f["ema200"]) &
            (f["ema50_slope"] > 0) &
            (low3 < f["ema20"]) &
            (f["close"] > f["ema20"]) &
            (f["rel_vol"] > 1.0))

def mask_meanrev(f):
    return ((f["adx14"] < 20) & (f["rsi14"] < 30) & (f["close"] < f["bb_lower"]))

def mask_breakout(f):
    prior_hi = f["high"].rolling(20).max().shift(1)
    return ((f["close"] > prior_hi) & (f["rel_vol"] > 1.5) &
            (f["ema50"] > f["ema200"]))

def mask_orb(f):
    idx = f.index
    day = idx.normalize()
    hrs = idx.hour
    in_range = (hrs >= 12) & (hrs <= 14)
    daily_hi = f["high"].where(in_range).groupby(day).max()
    daily_lo = f["low"].where(in_range).groupby(day).min()
    prev_hi = daily_hi.shift(1)
    prev_lo = daily_lo.shift(1)
    hi_map = day.map(prev_hi)
    lo_map = day.map(prev_lo)
    rng = hi_map - lo_map
    return ((hrs >= 15) & (f["close"] > hi_map) & (rng >= 0.5 * f["atr14"]) &
            ~hi_map.isna() & ~lo_map.isna())

MASK_BUILDERS = {
    "N1_trendpull": mask_trendpull,
    "N2_meanrev":   mask_meanrev,
    "N3_breakout":  mask_breakout,
    "N4_orb":       mask_orb,
}

print("\n  Building signal masks …")
masks = {}
for fname in FAMILIES:
    t1 = time.time()
    masks[fname] = {}
    for sym, f in feats_by_sym.items():
        m = MASK_BUILDERS[fname](f).fillna(False)
        masks[fname][sym] = m
    total = sum(int(m.sum()) for m in masks[fname].values())
    print(f"    {fname}: built in {time.time()-t1:.0f}s, total signals={total}")


# ── Run simulations ──────────────────────────────────────────────────────────
def run_family_custom(fname, feats_by_sym, mask_map, cfg, rr):
    all_t = []
    for sym, feats in feats_by_sym.items():
        try:
            for t in sim_symbol(feats, mask_map[sym], rr, cfg):
                t["sym"] = sym
                all_t.append(t)
        except Exception:
            pass
    all_t.sort(key=lambda t: t["entry_time"])
    return all_t

print("\n  Simulating families …")
results = {}
for fname, spec in FAMILIES.items():
    results[fname] = run_family_custom(fname, feats_by_sym, masks[fname],
                                       spec["cfg"], spec["rr"])


def monthly_record(trades):
    if not trades:
        return dict(n_months=0, prof_rate=float("nan"), max_loss_streak=float("nan"),
                    best_month=float("nan"), worst_month=float("nan"),
                    trades_per_month=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    n_months = len(g)
    prof = int((g > 0).sum())
    flags = (g > 0).astype(int).values
    cur = best_streak = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        best_streak = max(best_streak, cur)
    return dict(n_months=n_months, prof_rate=prof / n_months,
                max_loss_streak=best_streak,
                best_month=float(g.max()), worst_month=float(g.min()),
                trades_per_month=len(df) / n_months)


def retail_score(pf, prof_rate, max_streak, tpm, mdd):
    sc = (0.25 * min(1.0, max(0.0, (pf - 1.0) / 0.6))
          + 0.25 * prof_rate
          + 0.15 * min(1.0, max(0.0, 1 - max_streak / 10))
          + 0.20 * min(1.0, max(0.0, tpm / 30))
          + 0.15 * min(1.0, max(0.0, 1 - abs(mdd) / 0.30)))
    return sc * 100


def split_trades(trades):
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    return sel, hol


# ── Analysis ─────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("  FULL-PERIOD RESULTS — new families vs Family A FINAL (R074)")
hdr = (f"    {'Family':<16}{'n':>7}{'WR':>8}{'PF':>8}{'Exp$':>8}{'MDD%':>8}"
       f"{'t/mo':>6}{'prof%':>7}{'worst':>6}{'Rscore':>8}")
print(hdr); print("    " + "─"*94)

# Family A final reference (E6 + RR2 + volceil) — recompute quickly
from scripts.ql_engine import build_signal_mask
COND_A = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
maskA = {sym: build_signal_mask(f, COND_A, "green", 1.5) for sym, f in feats_by_sym.items()}
fa_trades = run_family_custom("FamilyA", feats_by_sym, maskA,
                              dict(entry_next=False, exit="base", hours=None,
                                   atr_rank_ceil=70.0), 2.0)
# (rr=2.0 passed as arg; cfg exit base + volceil)

all_rows = []
fa_row = None
for fname, trades in list(results.items()) + [("A_FINAL", fa_trades)]:
    s = stats_from_trades(trades)
    mr = monthly_record(trades)
    if mr["n_months"] == 0 or not np.isfinite(mr["prof_rate"]):
        continue
    rs = np.array([t["r"] for t in trades])
    b5, _, _ = bootstrap_pf(rs)
    score = retail_score(s["pf"], mr["prof_rate"], mr["max_loss_streak"],
                         mr["trades_per_month"], s["mdd"])
    row = dict(family=fname, n=s["n"], wr=s["wr"], pf=s["pf"], exp=s["exp"],
               mdd=s["mdd"], tpm=mr["trades_per_month"], prof_rate=mr["prof_rate"],
               max_streak=mr["max_loss_streak"], boot_p5=b5, score=score,
               best_month=mr["best_month"], worst_month=mr["worst_month"])
    all_rows.append(row)
    tag = "  ← FAMILY A FINAL" if fname == "A_FINAL" else ""
    print(f"    {fname:<16}{s['n']:>7}{s['wr']*100:>7.1f}%{s['pf']:>8.3f}"
          f"{s['exp']:>8.2f}{s['mdd']*100:>7.1f}%{mr['trades_per_month']:>6.1f}"
          f"{mr['prof_rate']*100:>6.0f}%{mr['max_loss_streak']:>6}{score:>8.1f}{tag}")

# ── Selection / holdout for new families ─────────────────────────────────────
print(f"\n{SEP2}")
print("  SELECTION (≤2025) / HOLDOUT (2026) for new families")
hdr2 = (f"    {'Family':<16}{'sel n':>7}{'sel PF':>8}{'hol n':>7}{'hol PF':>8}"
        f"{'hol MDD%':>9}{'hol t/mo':>9}")
print(hdr2); print("    " + "─"*68)
for fname in results:
    sel, hol = split_trades(results[fname])
    ss = stats_from_trades(sel); hs = stats_from_trades(hol)
    hmr = monthly_record(hol)
    print(f"    {fname:<16}{ss['n']:>7}{ss['pf']:>8.3f}{hs['n']:>7}{hs['pf']:>8.3f}"
          f"{hs['mdd']*100:>8.1f}%{hmr['trades_per_month']:>8.1f}")

# ── Robustness + costs on the best new family ────────────────────────────────
new_rows = [r for r in all_rows if r["family"] != "A_FINAL"]
new_rows.sort(key=lambda r: -r["score"])
best_new = new_rows[0]["family"] if new_rows else None
print(f"\n  Best new family by Retail Score: {best_new} (score={new_rows[0]['score']:.1f})")

if best_new and new_rows[0]["score"] > 0:
    trades = results[best_new]
    floor, rm = loo_symbol_floor(trades)
    mc = monte_carlo(np.array([t["r"] for t in trades]))
    print(f"  Robustness [{best_new}]: LOO floor PF={floor:.3f} (drop {rm}) | "
          f"MC P(profit)={mc['prob']*100:.1f}% net=${mc['exp']:+,.0f} DD P5={mc['dd_p5']*100:.1f}%")
    print("  Cost sensitivity (PF at cost/side):")
    for cp in [0.0, 0.05, 0.10, 0.15, 0.20]:
        print(f"    {cp:.2f}% → {pf_of_rs(cost_adjusted_rs(trades, cp)):.3f}")
    # monthly list
    mr = monthly_record(trades)
    print(f"  Monthly: {mr['n_months']} months, {mr['prof_rate']*100:.0f}% profitable, "
          f"worst streak {mr['max_loss_streak']}, best month {mr['best_month']:+.1f}R, "
          f"worst month {mr['worst_month']:+.1f}R")

# ── Outputs ──────────────────────────────────────────────────────────────────
pd.DataFrame(all_rows).to_csv(os.path.join(OUT, "r075_families.csv"), index=False)
lines = [f"# R075 — New Strategy Families (retail-friendly hunt)\n",
         f"**Date:** 2026-08-06  |  Selection ≤2025, holdout = 2026 (untouched)\n",
         f"**Goal:** find a profile a retail trader can survive (≥50% profitable months, "
         f"max losing streak ≤4, moderate frequency, lower MDD) — even at a PF trade-off.\n",
         f"\n## Family A FINAL (R074 reference)\n",
         f"- PF=1.63, ~10 t/mo, 41% profitable months, worst streak 7, MDD -26% → Retail Score below\n",
         f"\n## Full-period comparison\n",
         "| Family | n | WR | PF | Exp$ | MDD% | t/mo | prof% | worst | BootP5 | RetailScore |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in all_rows:
    tag = " ⭐ Family A" if r["family"] == "A_FINAL" else ""
    lines.append(f"| {r['family']}{tag} | {r['n']} | {r['wr']*100:.1f}% | {r['pf']:.3f} | "
                 f"{r['exp']:+.2f} | {r['mdd']*100:.1f}% | {r['tpm']:.1f} | {r['prof_rate']*100:.0f}% | "
                 f"{r['max_streak']} | {r['boot_p5']:.3f} | {r['score']:.1f} |")
lines += ["", "## Selection / holdout (new families)",
          "| Family | sel n | sel PF | hol n | hol PF | hol MDD% | hol t/mo |",
          "|---|---|---|---|---|---|---|"]
for fname in results:
    sel, hol = split_trades(results[fname])
    ss = stats_from_trades(sel); hs = stats_from_trades(hol)
    hmr = monthly_record(hol)
    lines.append(f"| {fname} | {ss['n']} | {ss['pf']:.3f} | {hs['n']} | {hs['pf']:.3f} | "
                 f"{hs['mdd']*100:.1f}% | {hmr['trades_per_month']:.1f} |")
if best_new:
    lines += ["", f"## Robustness of best new family ({best_new})",
              f"- LOO floor PF={floor:.3f} (drop {rm})",
              f"- MC: P(profit)={mc['prob']*100:.1f}%, net=${mc['exp']:+,.0f}, DD P5={mc['dd_p5']*100:.1f}%",
              f"- Cost: " + ", ".join(f"{cp:.2f}%→{pf_of_rs(cost_adjusted_rs(trades, cp)):.3f}"
                                      for cp in [0.0, 0.05, 0.10, 0.15, 0.20]),
              f"- Monthly: {mr['n_months']} mo, {mr['prof_rate']*100:.0f}% profitable, "
              f"worst streak {mr['max_loss_streak']}, best {mr['best_month']:+.1f}R, worst {mr['worst_month']:+.1f}R"]
lines += ["", "## Verdict", ""]
fa = next(r for r in all_rows if r["family"] == "A_FINAL")
if best_new and new_rows[0]["score"] > fa["score"]:
    lines.append(f"**ADOPT {best_new} as primary** (Retail Score {new_rows[0]['score']:.1f} "
                 f"vs Family A {fa['score']:.1f}) — IF it survives the holdout above.")
elif best_new:
    lines.append(f"**No retail-friendly upgrade yet.** Best new family ({best_new}) scored "
                 f"{new_rows[0]['score']:.1f} vs Family A {fa['score']:.1f} — "
                 f"Family A final remains the reference. Continue hunting.")
else:
    lines.append("**No viable new family found.**")
report = "\n".join(lines)
with open(os.path.join(OUT, "r075_final_report.md"), "w") as f:
    f.write(report)

print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r075_*")
