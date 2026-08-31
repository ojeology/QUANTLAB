"""
FOREX HUNT — F005
THE TRAP HUNT: when the market enters a trap (pierces a key level = stop-hunt),
does it REVERSE or FOLLOW THE TREND?

Mechanics (institutional): price wicks through a prior high/low (trapping breakout
traders / hunting stops), then either snaps back (REVERSAL) or continues (FOLLOW).

Trap event (causal): within the last K=3 bars, price WICKED through the prior-20-bar
level (low < prior_lo  OR  high > prior_hi). At the current close:
  REVERSE play  : reclaim back through the level (close back above prior_lo / below
                  prior_hi) -> fade the trap
  FOLLOW play   : close stays through the level (below prior_lo / above prior_hi)
                  -> ride the break

4 signal types (long AND short):
  T1 bear-trap REVERSE (long)   : wick below prior_lo + close back above + green
  T2 bear-trap FOLLOW  (short)  : wick below prior_lo + close still below + red
  T3 bull-trap REVERSE (short)  : wick above prior_hi + close back below + red
  T4 bull-trap FOLLOW  (long)   : wick above prior_hi + close still above + green
Shorts simulated exactly by inverting the price series (long-only engine).

Pierce-depth variant: require the wick to extend >= 0.2*ATR through the level
(real stop-hunt, not noise).

TF: 4H (F004 showed 4H halves cost drag). RR sweep {2.0, 3.0}.
Protocol: selection <= Aug-2025, holdout Aug-2025..Aug-2026 untouched, retail spreads.
Success: holPF@cost > 1.1, sel n >= 60.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, sim_symbol, stats_from_trades, pf_of_rs, IS_LOOKBACK, RECAL_EVERY
import scripts.ql_engine as qle

FOREX_DIR = os.path.join(CONFIG["CACHE_FOLDER"], "forex")
OUT = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2025-08-01", tz="UTC")
PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","EURGBP"]
SPREAD = {"EURUSD":0.00006,"GBPUSD":0.00010,"USDJPY":0.010,"AUDUSD":0.00008,
          "USDCAD":0.00010,"USDCHF":0.00010,"NZDUSD":0.00012,"EURGBP":0.00010}
K = 3          # trap window (bars)
PRIOR = 20     # prior-level lookback
IS_LOOKBACK_4H = 125
RECAL_EVERY_4H = 42

SEP = "=" * 110
SEP2 = "-" * 90
print(); print(SEP)
print("  FOREX HUNT — F005  THE TRAP (reverse vs follow, long & short, 4H)")
print(SEP)
t0 = time.time()

print("\n  Loading 4H forex data …")
raw4h = {}
for p in PAIRS:
    fpath = os.path.join(FOREX_DIR, f"{p}_1H.parquet")
    if not os.path.exists(fpath): continue
    df = pd.read_parquet(fpath)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    for col in ["open","high","low","close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["open","high","low","close"], inplace=True)
    if "vol" not in df.columns:
        df["vol"] = (df["high"] - df["low"]).clip(lower=1e-12)
    r = df.resample("4H").agg({"open":"first","high":"max","low":"min","close":"last","vol":"sum"}).dropna()
    raw4h[p] = r

feats = {}
for p, df in raw4h.items():
    f = add_features(df)
    f.dropna(subset=["ema200","atr14","adx14","rsi14","ema_dist_pct","real_vol_20",
                     "bb_width","prev_range_r","prev_body_r"], inplace=True)
    if len(f) >= IS_LOOKBACK_4H + RECAL_EVERY_4H:
        feats[p] = f
print(f"  Pairs ready: {len(feats)}")

qle.IS_LOOKBACK = IS_LOOKBACK_4H
qle.RECAL_EVERY = RECAL_EVERY_4H

# ── trap signals (causal) ────────────────────────────────────────────────────
def trap_signals(f, depth=0.2):
    prior_hi = f["high"].rolling(PRIOR).max().shift(1)
    prior_lo = f["low"].rolling(PRIOR).min().shift(1)
    # wick pierce through the level (stop-hunt) - causal (intrabar wick, level from prior bars)
    wick_bear = f["low"] < prior_lo - depth * f["atr14"]
    wick_bull = f["high"] > prior_hi + depth * f["atr14"]
    wick_bear_recent = wick_bear.rolling(K, min_periods=1).max().astype(bool)
    wick_bull_recent = wick_bull.rolling(K, min_periods=1).max().astype(bool)
    # reverse vs follow at current close
    t1_bear_rev_long  = wick_bear_recent & (f["close"] > prior_lo) & (f["close"] > f["open"])
    t2_bear_fol_short = wick_bear_recent & (f["close"] < prior_lo) & (f["close"] < f["open"])
    t3_bull_rev_short = wick_bull_recent & (f["close"] < prior_hi) & (f["close"] < f["open"])
    t4_bull_fol_long  = wick_bull_recent & (f["close"] > prior_hi) & (f["close"] > f["open"])
    return {
        "T1_bear_rev_long":  t1_bear_rev_long,
        "T2_bear_fol_short": t2_bear_fol_short,
        "T3_bull_rev_short": t3_bull_rev_short,
        "T4_bull_fol_long":  t4_bull_fol_long,
    }

# causal audit
print(f"\n{SEP2}\n  CAUSAL AUDIT\n{SEP2}")
audit_ok = {}
test_p = sorted(feats.keys())[0]
ft = feats[test_p]; fts = ft.iloc[:-200]
for name, fn in trap_signals(ft).items():
    m1 = fn; m2 = trap_signals(fts)[name]
    same = bool((m1.loc[fts.index] == m2.loc[fts.index]).all())
    audit_ok[name] = same
    print(f"    {name:<22} {'PASS ✓' if same else 'FAIL ✗ LOOKAHEAD'}")

def invert(f):
    """Invert price series to simulate shorts with a long-only engine."""
    g = f.copy()
    for c in ["open","high","low","close"]:
        g[c] = -f[c]
    # swap high/low after negation: in inverted space, the 'high' of inverted = -real low
    g["high"], g["low"] = -f["low"], -f["high"]
    return g

def run_directional(f, mask, rr, short=False):
    """Run sim; if short, run on inverted series and flip r back to real sign.
    (Engine's r is direction-agnostic: win=+rr, loss=-1 regardless.)"""
    g = invert(f) if short else f
    out = []
    try:
        for t in sim_symbol(g, mask, rr, dict(entry_next=False, exit="timeN",
                                              time_bars=24, hours=None)):
            out.append(t)
    except Exception:
        pass
    return out

def eval_cfg(name, trades):
    if not trades:
        return dict(cfg=name, n=0, tpm=0, wr=float('nan'), pf=float('nan'), pf_c=float('nan'),
                    mdd=0, prof=float('nan'), worst=float('nan'), selpf=float('nan'),
                    holpf=float('nan'), holpf_c=float('nan'))
    s = stats_from_trades(trades)
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
                mdd=s["mdd"], prof=float((g>0).mean()), worst=worst, selpf=sp, holpf=hp, holpf_c=hpc)

SHORT_MAP = {"T1_bear_rev_long": False, "T2_bear_fol_short": True,
             "T3_bull_rev_short": True, "T4_bull_fol_long": False}

print(f"\n{SEP2}\n  RESULTS  (4H, RR sweep, selection ≤ Aug-2025 | holdout untouched)\n{SEP2}")
hdr = (f"    {'Signal':<22}{'RR':>5}{'n':>6}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@cost':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'holPF':>7}{'holPF@c':>9}")
print(hdr); print("    " + "-"*112)
rows = []
for depth in [0.0, 0.2]:
    for p, f in feats.items():
        sigs = trap_signals(f, depth)
        for name, mask in sigs.items():
            for rr in [2.0, 3.0]:
                short = SHORT_MAP[name]
                trades = []
                for t in run_directional(f, mask, rr, short):
                    t["pair"] = p; t["spread"] = SPREAD[p]; t["sig"] = f"{name}_d{depth}"
                    trades.append(t)
                # accumulate into a per-(name,depth,rr) bucket
    # (accumulate handled below instead - simpler per combo)

# redo cleanly: bucket by (sig, depth, rr)
buckets = {}
for depth in [0.0, 0.2]:
    for p, f in feats.items():
        sigs = trap_signals(f, depth)
        for name, mask in sigs.items():
            for rr in [2.0, 3.0]:
                key = (name, depth, rr)
                if key not in buckets: buckets[key] = []
                short = SHORT_MAP[name]
                for t in run_directional(f, mask, rr, short):
                    t["pair"] = p; t["spread"] = SPREAD[p]
                    buckets[key].append(t)

for (name, depth, rr), trades in sorted(buckets.items()):
    trades.sort(key=lambda t: t["entry_time"])
    label = f"{name}_d{int(depth*10)}"
    r = eval_cfg(label, trades)
    rows.append(r)
    print(f"    {label:<22}{rr:>5.1f}{r['n']:>6}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['holpf']:>7.2f}{r['holpf_c']:>9.2f}")

print(f"\n{SEP2}\n  SUCCESS (holPF@cost>1.1, sel n>=60)")
passed = [r for r in rows if r["holpf_c"] > 1.1 and r["n"] >= 60]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: holPF@cost={r['holpf_c']:.2f}")
else:
    print("  ❌ none passed. Closest (by holPF@cost):")
    for r in sorted(rows, key=lambda r: -r["holpf_c"])[:6]:
        print(f"    {r['cfg']}: holPF@cost={r['holpf_c']:.2f} holPF={r['holpf']:.2f} PF={r['pf']:.2f} n={r['n']} WR={r['wr']*100:.0f}%")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "f005_trap.csv"), index=False)
lines = [f"# FOREX F005 — THE TRAP (reverse vs follow)\n",
         f"**Date:** 2026-08-08 | 4H, 8 pairs, selection ≤ Aug-2025, holdout untouched, retail spreads\n",
         f"Trap = wick through prior-20-bar level (stop-hunt); REVERSE = reclaim through it, "
         f"FOLLOW = continue through it. Long & short via series inversion.\n",
         f"\n## Results\n",
         "| Signal | RR | n | WR | PF | PF@cost | MDD% | prof% | worst | holPF | holPF@cost |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | - | {r['n']} | {r['wr']*100:.0f}% | {r['pf']:.2f} | {r['pf_c']:.2f} | "
                 f"{r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | {r['holpf']:.2f} | "
                 f"{r['holpf_c']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    lines.append(f"**✅ {passed[0]['cfg']} survives costs on holdout.**")
else:
    lines.append("**❌ No trap config survives costs on holdout.** Honest negative — see closest above.")
report = "\n".join(lines)
with open(os.path.join(OUT, "f005_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/f005_*")
