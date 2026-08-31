"""
QUANTLAB AI — R077
$100 Drawdown, Lower Timeframes, RR Sweep → Lock-In

Three asks, one run:
  A  $100-account maximum drawdown for the FINAL strategy (Family A + E6 + RR2 +
     VolCeil + breadth50) — in dollars, at several risk-per-trade levels, plus
     Monte Carlo dollar-drawdown distribution.
  B  Lower timeframe: run the LOCKED config on 15m data (8 symbols, 2026 only —
     completely untouched OOS) with 1H 52-symbol breadth as the market gate.
     Compare vs same 8 symbols on 1H over the same window.
  C  RR sweep on the final 1H config (selection ≤2025 / holdout 2026).
  D  LOCK-IN the final configuration.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, build_signal_mask, sim_symbol, stats_from_trades,
    bootstrap_pf, loo_symbol_floor, IS_LOOKBACK, RECAL_EVERY,
    STARTING_CAP, RISK_PCT,
)

RESEARCH_ID = "R077"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")

FAM_A = dict(cids=["BBW_STRICT","RV_LO","DST_NR","PRG_VH"])
BASE_CFG = dict(entry_next=False, exit="base", hours=None, atr_rank_ceil=70.0)
BREADTH_THR = 0.50

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  $100 DD / 15m / RR sweep → lock-in")
print(SEP)
t0 = time.time()

# ── Load 1H data ─────────────────────────────────────────────────────────────
print("\n  Loading 1H data …")
feats1h = {}
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
            feats1h[sym] = f
    except Exception:
        pass
print(f"  1H symbols: {len(feats1h)}")

# base mask + breadth (1H)
base_mask1h = {sym: build_signal_mask(f, FAM_A["cids"], "green", 1.5)
               for sym, f in feats1h.items()}
above = {sym: (f["close"] > f["ema20"]).astype(float) for sym, f in feats1h.items()}
breadth1h = pd.DataFrame(above).sort_index().mean(axis=1, skipna=True)

final_mask1h = {}
for sym, m in base_mask1h.items():
    f = feats1h[sym]
    reg = (breadth1h.reindex(f.index, method="ffill") > BREADTH_THR).fillna(False)
    final_mask1h[sym] = m & reg.values

def run_cfg(feats, mask, rr, cfg):
    all_t = []
    for sym, f in feats.items():
        try:
            for t in sim_symbol(f, mask[sym], rr, cfg):
                t["sym"] = sym
                all_t.append(t)
        except Exception:
            pass
    all_t.sort(key=lambda t: t["entry_time"])
    return all_t


# ═════════════════════════════════════════════════════════════════════════════
# PART A — $100 drawdown
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}\n  PART A — $100 account, FINAL strategy (breadth50, RR=2.0)\n{SEP2}")
trades = run_cfg(feats1h, final_mask1h, 2.0, BASE_CFG)
rs = np.array([t["r"] for t in trades])

def dollar_stats(risk_pct, start=100.0):
    cap = start; eq = [cap]; eqm = {t["entry_time"].to_period("M"): [] for t in trades}
    for t, r in zip(trades, rs):
        cap += cap * risk_pct * r
        eq.append(cap)
    eq = np.array(eq)
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk).min())
    mdd_usd = mdd * start
    net = cap - start
    return dict(risk=risk_pct, net=net, mdd_pct=mdd, mdd_usd=mdd_usd,
                final=cap)

print(f"  Trades: {len(trades)} | full-period PF={stats_from_trades(trades)['pf']:.3f} "
      f"MDD%={stats_from_trades(trades)['mdd']*100:.1f}%")
print(f"\n  {'Risk/trade':<12}{'Net $':>10}{'Max DD $':>10}{'Max DD %':>10}{'Final $':>10}")
rows_a = []
for rp in [0.005, 0.01, 0.015, 0.02, 0.025]:
    d = dollar_stats(rp)
    rows_a.append(d)
    print(f"  {rp*100:>6.1f}%   {d['net']:>+10.2f}{d['mdd_usd']:>10.2f}"
          f"{d['mdd_pct']*100:>9.1f}%{d['final']:>10.2f}")

# worst month + worst streak in dollars (1% risk, $100)
cap = 100.0
monthly = {}
streak = 0; worst_streak = 0
for t, r in zip(trades, rs):
    cap += cap * 0.01 * r
    m = t["entry_time"].to_period("M")
    monthly[m] = monthly.get(m, 0.0) + r
    if r < 0: streak += 1; worst_streak = max(worst_streak, streak)
    else: streak = 0
mdf = pd.Series(monthly).sort_index()
print(f"\n  At 1% risk on $100 (1R = $1, compounding):")
print(f"    Worst month: {mdf.idxmin()} ({mdf.min():+.2f}R ≈ ${mdf.min():+.2f})")
print(f"    Best month:  {mdf.idxmax()} ({mdf.max():+.2f}R ≈ ${mdf.max():+.2f})")
print(f"    Worst losing-trade streak: {worst_streak} consecutive losers")
print(f"    Profitable months: {(mdf>0).mean()*100:.0f}%")

# Monte Carlo dollar drawdown at 1% risk, $100
rng = np.random.default_rng(42)
n_mc = 5000
dds, nets = [], []
for _ in range(n_mc):
    s = rs[rng.integers(0, len(rs), len(rs))]
    cap = 100.0; eq = [cap]
    for r in s:
        cap += cap * 0.01 * r
        eq.append(cap)
    eq = np.array(eq); pk = np.maximum.accumulate(eq)
    dds.append(float(((eq - pk) / pk).min()) * 100.0)
    nets.append(cap - 100.0)
dds = np.array(dds); nets = np.array(nets)
print(f"  Monte Carlo (5,000 paths, 1% risk, $100):")
print(f"    Max DD:  P5=${-np.percentile(dds,5):.2f}  P50=${-np.percentile(dds,50):.2f}  "
      f"P95=${-np.percentile(dds,95):.2f}")
print(f"    P(end > $100) = {(nets>0).mean()*100:.1f}% | P(end > $120) = {(nets>20).mean()*100:.1f}% | "
      f"P(end < $80) = {(nets<-20).mean()*100:.1f}% | P(end < $90) = {(nets<-10).mean()*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# PART B — 15m lower timeframe (locked config, untouched OOS)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}\n  PART B — 15m test (8 symbols, 2026 only — untouched OOS)\n{SEP2}")
feats15 = {}
for fn in sorted(os.listdir(CACHE)):
    if not fn.endswith("_15m.parquet"): continue
    sym = fn.replace("_15m.parquet", "")
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
            feats15[sym] = f
    except Exception:
        pass
print(f"  15m symbols: {len(feats15)}")

# breadth gate for 15m: use 1H 52-symbol breadth reindexed to 15m
base_mask15 = {sym: build_signal_mask(f, FAM_A["cids"], "green", 1.5)
               for sym, f in feats15.items()}
final_mask15 = {}
for sym, m in base_mask15.items():
    f = feats15[sym]
    reg = (breadth1h.reindex(f.index, method="ffill") > BREADTH_THR).fillna(False)
    final_mask15[sym] = m & reg.values

t15 = run_cfg(feats15, final_mask15, 2.0, BASE_CFG)
s15 = stats_from_trades(t15)
mr15 = None
if t15:
    d15 = pd.DataFrame(t15); d15["month"] = d15["entry_time"].dt.to_period("M")
    g15 = d15.groupby("month")["r"].sum()
    mr15 = dict(months=len(g15), prof=float((g15>0).mean()), tpm=len(d15)/len(g15),
                best=float(g15.max()), worst=float(g15.min()))
print(f"  15m result (LOCKED config, RR=2.0, breadth50):")
if t15:
    print(f"    n={s15['n']}  PF={s15['pf']:.3f}  WR={s15['wr']*100:.1f}%  "
          f"MDD={s15['mdd']*100:.1f}%  Exp=${s15['exp']:.2f}")
    print(f"    months={mr15['months']}  prof%={mr15['prof']*100:.0f}%  "
          f"t/mo={mr15['tpm']:.1f}  best={mr15['best']:+.1f}R  worst={mr15['worst']:+.1f}R")
else:
    print("    NO TRADES in 15m window with locked config.")

# apples-to-apples: same 8 symbols on 1H, same window (>=2026-01-27)
sym8 = set(feats15.keys())
feats1h8 = {k: v for k, v in feats1h.items() if k in sym8}
t1h8 = [t for t in run_cfg(feats1h8, {k: final_mask1h[k] for k in feats1h8}, 2.0, BASE_CFG)
        if t["entry_time"] >= pd.Timestamp("2026-01-27", tz="UTC")]
s1h8 = stats_from_trades(t1h8)
print(f"  Same 8 symbols on 1H (same window): n={s1h8['n']}  PF={s1h8['pf']:.3f}  "
      f"WR={s1h8['wr']*100:.1f}%  MDD={s1h8['mdd']*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# PART C — RR sweep on final 1H config
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}\n  PART C — RR sweep on final 1H config (breadth50)\n{SEP2}")
RR_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
print(f"  {'RR':>5}{'n':>6}{'WR':>8}{'PF':>8}{'MDD%':>8}{'selPF':>8}{'holPF':>8}  verdict")
rr_rows = []
for rr in RR_SWEEP:
    t = run_cfg(feats1h, final_mask1h, rr, BASE_CFG)
    s = stats_from_trades(t)
    sel = [x for x in t if x["entry_time"] < HOLDOUT_START]
    hol = [x for x in t if x["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]; hp = stats_from_trades(hol)["pf"]
    verdict = ""
    if hp > 1.0 and sp > 1.0: verdict = "OK"
    elif hp > 1.0: verdict = "hol-ok"
    elif sp > 1.0: verdict = "sel-only"
    else: verdict = "no"
    rr_rows.append(dict(rr=rr, n=s["n"], wr=s["wr"], pf=s["pf"], mdd=s["mdd"],
                        sel_pf=sp, hol_pf=hp, verdict=verdict))
    print(f"  {rr:>5.1f}{s['n']:>6}{s['wr']*100:>7.1f}%{s['pf']:>8.3f}"
          f"{s['mdd']*100:>7.1f}%{sp:>8.3f}{hp:>8.3f}  {verdict}")

ok_rows = [r for r in rr_rows if r["hol_pf"] > 1.0 and r["sel_pf"] > 1.0]
best_rr = max(ok_rows, key=lambda r: r["hol_pf"])["rr"] if ok_rows else 2.0
print(f"\n  → Best RR (max holdout PF among sel+hol profitable): {best_rr}")

# paired ΔPF: best RR vs RR=2.0 (aligned by sym+entry_time) — full period
def paired_dpf(trades_a, trades_b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    b_idx = {(t["sym"], t["entry_time"]): j for j, t in enumerate(trades_b)}
    pairs = [(i, b_idx[(t["sym"], t["entry_time"])]) for i, t in enumerate(trades_a)
             if (t["sym"], t["entry_time"]) in b_idx]
    if len(pairs) < 50: return (np.nan, np.nan, np.nan)
    A = np.array([trades_a[i]["r"] for i, _ in pairs])
    B = np.array([trades_b[j]["r"] for _, j in pairs])
    out = np.empty(n_boot)
    for k in range(n_boot):
        ix = rng.integers(0, len(pairs), len(pairs))
        def pf(x):
            s = x[ix]; w = s[s > 0].sum(); lo = abs(s[s < 0].sum())
            return w / lo if lo > 0 else (999.0 if w > 0 else 1.0)
        out[k] = pf(A) - pf(B)
    return (float(np.percentile(out, 5)), float(np.median(out)), float(np.percentile(out, 95)))

if best_rr != 2.0:
    t_best = run_cfg(feats1h, final_mask1h, best_rr, BASE_CFG)
    t_rr2  = run_cfg(feats1h, final_mask1h, 2.0, BASE_CFG)
    d5, dmed, d95 = paired_dpf(t_best, t_rr2)
    print(f"  Paired ΔPF (RR{best_rr} vs RR2.0, full period): [{d5:+.3f}, {d95:+.3f}] "
          f"{'SIG better' if d5>0 else 'not significant'}")
else:
    t_best = run_cfg(feats1h, final_mask1h, best_rr, BASE_CFG)
    d5 = dmed = d95 = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# PART D — LOCK-IN
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}\n  PART D — LOCKED CONFIGURATION\n{SEP2}")
final_trades = run_cfg(feats1h, final_mask1h, best_rr, BASE_CFG)
fs = stats_from_trades(final_trades)
floor, rm = loo_symbol_floor(final_trades)
b5, bmed, b95 = bootstrap_pf(np.array([t["r"] for t in final_trades]))
print(f"  Family A + E6 entry + RR={best_rr} + VolCeil(≤70) + breadth50")
print(f"    n={fs['n']}  PF={fs['pf']:.3f}  WR={fs['wr']*100:.1f}%  "
      f"MDD={fs['mdd']*100:.1f}%  Exp=${fs['exp']:.2f}")
print(f"    Boot P5={b5:.3f}  LOO floor={floor:.3f} (drop {rm})")

# monthly cadence
d = pd.DataFrame(final_trades); d["month"] = d["entry_time"].dt.to_period("M")
g = d.groupby("month")["r"].sum()
flags = (g > 0).astype(int).values
cur = worst_streak_m = 0
for v in flags:
    cur = cur + 1 if not v else 0
    worst_streak_m = max(worst_streak_m, cur)
print(f"    t/mo={len(d)/len(g):.1f}  prof months={(g>0).mean()*100:.0f}%  "
      f"worst month-streak={worst_streak_m}")

# ── Outputs ──────────────────────────────────────────────────────────────────
pd.DataFrame(rows_a).to_csv(os.path.join(OUT, "r077_dollar_dd.csv"), index=False)
pd.DataFrame(rr_rows).to_csv(os.path.join(OUT, "r077_rr_sweep.csv"), index=False)

lines = [f"# R077 — $100 DD / 15m / RR sweep → Lock-In\n",
         f"**Date:** 2026-08-06\n",
         f"## A — $100 account (final strategy, breadth50)\n",
         f"| Risk/trade | Net $ | Max DD $ | Max DD % | Final $ |",
         f"|---|---|---|---|---|"]
for d in rows_a:
    lines.append(f"| {d['risk']*100:.1f}% | {d['net']:+.2f} | {d['mdd_usd']:.2f} | "
                 f"{d['mdd_pct']*100:.1f}% | {d['final']:.2f} |")
lines += [f"\nAt 1% risk / $100 (1R=$1): worst month {mdf.idxmin()} {mdf.min():+.2f}R, "
          f"best month {mdf.idxmax()} {mdf.max():+.2f}R, worst streak {worst_streak} losers.",
          f"\nMonte Carlo (5,000 paths, 1% risk, $100):",
          f"- Max DD: P5=${-np.percentile(dds,5):.2f}, P50=${-np.percentile(dds,50):.2f}, "
          f"P95=${-np.percentile(dds,95):.2f}",
          f"- P(end>$100)={(nets>0).mean()*100:.1f}%  P(end>$120)={(nets>120).mean()*100:.1f}%  "
          f"P(end<$80)={(nets<80).mean()*100:.1f}%",
          "",
          f"## B — 15m lower timeframe (locked config, 8 symbols, 2026-only = untouched OOS)",
          f"| Timeframe | n | PF | WR | MDD% | t/mo |",
          f"|---|---|---|---|---|---|"]
if t15:
    lines.append(f"| 15m | {s15['n']} | {s15['pf']:.3f} | {s15['wr']*100:.1f}% | "
                 f"{s15['mdd']*100:.1f}% | {mr15['tpm']:.1f} |")
else:
    lines.append("| 15m | 0 | — | — | — | — |")
lines.append(f"| 1H (same 8 syms) | {s1h8['n']} | {s1h8['pf']:.3f} | {s1h8['wr']*100:.1f}% | "
             f"{s1h8['mdd']*100:.1f}% | — |")
lines += ["",
          f"## C — RR sweep (final 1H config)",
          "| RR | n | WR | PF | MDD% | selPF | holPF | verdict |",
          "|---|---|---|---|---|---|---|---|"]
for r in rr_rows:
    lines.append(f"| {r['rr']:.1f} | {r['n']} | {r['wr']*100:.1f}% | {r['pf']:.3f} | "
                 f"{r['mdd']*100:.1f}% | {r['sel_pf']:.3f} | {r['hol_pf']:.3f} | {r['verdict']} |")
lines += ["", f"## D — LOCKED CONFIGURATION",
          f"**Family A + E6 entry + RR={best_rr} + VolCeil(atr_rank≤70) + breadth50(>50% above EMA20)**",
          f"- n={fs['n']}, PF={fs['pf']:.3f}, WR={fs['wr']*100:.1f}%, MDD={fs['mdd']*100:.1f}%, "
          f"Exp=${fs['exp']:.2f}",
          f"- Boot P5={b5:.3f}, LOO floor={floor:.3f} (drop {rm})",
          f"- ~{len(d)/len(g):.1f} trades/month, {(g>0).mean()*100:.0f}% profitable months, "
          f"worst month-streak={worst_streak_m}"]
if best_rr != 2.0:
    lines.append(f"- Paired ΔPF vs RR2.0 (full period): [{d5:+.3f}, {d95:+.3f}] "
                 f"{'— statistically better' if d5>0 else '— not statistically significant'}")
    lines.append(f"- Chosen over RR2.0 for better holdout PF ({next(r['hol_pf'] for r in rr_rows if r['rr']==best_rr):.3f} "
                 f"vs {next(r['hol_pf'] for r in rr_rows if r['rr']==2.0):.3f}) and superior retail metrics "
                 f"(WR {fs['wr']*100:.0f}%, {(g>0).mean()*100:.0f}% prof months, streak {worst_streak_m})")
report = "\n".join(lines)
with open(os.path.join(OUT, "r077_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r077_*")
