"""
QUANTLAB AI — R081
High-Win-Rate Scalp Expansion (target: 70%+ profitable months)

User target: retail-friendly = >=70% profitable months, max 2-3 bad months,
more trades than Family A locked (4.3/mo).

NEW DIMENSION: sub-1R take profits (scalping). All prior runs used RR>=1.0.
At RR 0.3-0.75, win rate climbs to 65-80% and trade count rises (closer target =
more hits). High WR + many trades/month is the path to high profitable-month %.

Signals tested (each × RR {0.4,0.5,0.6,0.75} × exit {base, time6}):
  S1 Family A RAW (no breadth/volceil)  — already 14.9 t/mo at RR1.5
  S2 Breakout 20-bar (rv1.3)             — failed at RR1.5; scalp may save it
  H6 Micro-scalp: strong green momentum (close>open & close>prev_high & relvol>1.3 & above EMA20)
  H7 EMA20-touch scalp: uptrend + dip to EMA20 + reclaim, target 0.5R quickly
  H8 RSI2 mean-reversion scalp: RSI(2)<10 (deep short-term oversold), target 0.5R

Exits: E6 entry, base SL=1ATR, TP=rr*ATR (rr<1 = scalp), optional 6-bar time stop.

Metrics: t/mo, WR, PF (gross + at 0.05% cost), MDD, profitable-months%, worst
losing-month streak, selection PF, holdout PF.

Success (user spec): t/mo>=8, prof-months%>=70, worst streak<=3, holPF>1.05,
PF-at-cost>1.0.
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
    cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY, COND_DEF,
)

RESEARCH_ID = "R081"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")

ORIGINAL = {
    "1INCH_USDT_SWAP","AAVE_USDT_SWAP","ADA_USDT_SWAP","ALGO_USDT_SWAP",
    "APT_USDT_SWAP","ARB_USDT_SWAP","ATOM_USDT_SWAP","AVAX_USDT_SWAP",
    "AXS_USDT_SWAP","BCH_USDT_SWAP","BNB_USDT_SWAP","BONK_USDT_SWAP",
    "BTC_USDT_SWAP","CHZ_USDT_SWAP","COMP_USDT_SWAP","CRV_USDT_SWAP",
    "DOGE_USDT_SWAP","DOT_USDT_SWAP","DYDX_USDT_SWAP","EGLD_USDT_SWAP",
    "ENA_USDT_SWAP","ETC_USDT_SWAP","ETH_USDT_SWAP","FET_USDT_SWAP",
    "FIL_USDT_SWAP","FLOKI_USDT_SWAP","GALA_USDT_SWAP","GMX_USDT_SWAP",
    "GRT_USDT_SWAP","HBAR_USDT_SWAP","ICP_USDT_SWAP","IMX_USDT_SWAP",
    "INJ_USDT_SWAP","LDO_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP",
    "NEAR_USDT_SWAP","OP_USDT_SWAP","PEPE_USDT_SWAP","SAND_USDT_SWAP",
    "SATS_USDT_SWAP","SHIB_USDT_SWAP","SNX_USDT_SWAP","SOL_USDT_SWAP",
    "STX_USDT_SWAP","SUI_USDT_SWAP","SUSHI_USDT_SWAP","TRX_USDT_SWAP",
    "UNI_USDT_SWAP","WIF_USDT_SWAP","XLM_USDT_SWAP","XRP_USDT_SWAP",
}

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  High-Win-Rate Scalp Expansion")
print(SEP)
t0 = time.time()

print("\n  Loading data …")
feats = {}
for sym in ORIGINAL:
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        f = add_features(df)
        f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14",
                         "ema_dist_pct","real_vol_20","bb_width","prev_range_r",
                         "prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f
    except Exception:
        pass
print(f"  Symbols: {len(feats)}")

# ── RSI(2) needed — add column ───────────────────────────────────────────────
def calc_rsi2(series):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(2).mean()
    down = (-delta.clip(upper=0)).rolling(2).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

for sym, f in feats.items():
    f["rsi2"] = calc_rsi2(f["close"])

# ── Signal masks ─────────────────────────────────────────────────────────────
famA_cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
mask_famA = {s: build_signal_mask(f, famA_cids, "green", 1.5) for s, f in feats.items()}

def h2_breakout(f):
    prior_hi = f["high"].rolling(20).max().shift(1)
    return ((f["close"] > prior_hi) & (f["rel_vol"] > 1.3) &
            (f["close"] > f["ema20"]) & (f["close"] > f["open"]))

def h6_microscalp(f):
    return ((f["close"] > f["open"]) & (f["close"] > f["high"].shift(1)) &
            (f["rel_vol"] > 1.3) & (f["close"] > f["ema20"]))

def h7_ema20touch(f):
    low2 = f["low"].rolling(2, min_periods=1).min()
    return ((f["ema50"] > f["ema200"]) & (f["ema50_slope"] > 0) &
            (low2 < f["ema20"]) & (f["close"] > f["ema20"]) & (f["close"] > f["open"]))

def h8_rsi2(f):
    return ((f["rsi2"] < 10) & (f["close"] < f["ema200"]))

SIGNALS = {
    "S1_famA_raw":   (mask_famA, "Family A raw"),
    "S2_breakout":   ({s: h2_breakout(f) for s, f in feats.items()}, "20-bar breakout"),
    "H6_microscalp": ({s: h6_microscalp(f) for s, f in feats.items()}, "green-momentum micro-scalp"),
    "H7_ema20touch": ({s: h7_ema20touch(f) for s, f in feats.items()}, "EMA20-touch scalp"),
    "H8_rsi2":       ({s: h8_rsi2(f) for s, f in feats.items()}, "RSI2 mean-rev scalp"),
}

RR_SWEEP = [0.4, 0.5, 0.6, 0.75]
EXITS = ["base", "time6"]

def run(mask, rr, exit_mode):
    cfg = dict(entry_next=False, exit=exit_mode, hours=None)
    if exit_mode == "time6":
        cfg["time_bars"] = 6
    out = []
    for sym, f in feats.items():
        try:
            for t in sim_symbol(f, mask[sym], rr, cfg):
                t["sym"] = sym
                out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t["entry_time"])
    return out

def monthly_profile(trades):
    if not trades: return dict(n_months=0, prof=float("nan"), worst=float("nan"), tpm=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        worst = max(worst, cur)
    return dict(n_months=len(g), prof=float((g > 0).mean()), worst=worst,
                tpm=len(df) / max(len(g), 1))

print("\n  Running scalp sweep …")
results = {}
for sname, (mask, label) in SIGNALS.items():
    for rr in RR_SWEEP:
        for ex in EXITS:
            key = f"{sname}_rr{rr}_{ex}"
            results[key] = run(mask, rr, ex)

print(f"\n{SEP2}")
print("  SCALP SWEEP — full results (sorted by profitable-months% then t/mo)")
hdr = (f"    {'Variant':<24}{'n':>5}{'t/mo':>6}{'WR':>7}{'PF':>6}{'PF@0.05%':>9}"
       f"{'MDD%':>7}{'prof%':>6}{'worst':>6}{'selPF':>7}{'holPF':>7}")
print(hdr); print("    " + "─"*100)
rows = []
for key, trades in results.items():
    s = stats_from_trades(trades)
    rs = np.array([t["r"] for t in trades])
    pf_cost = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]
    hp = stats_from_trades(hol)["pf"]
    mp = monthly_profile(trades)
    rows.append(dict(variant=key, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"],
                     pf_cost=pf_cost, mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"],
                     selpf=sp, holpf=hp))
rows.sort(key=lambda r: (-r["prof"], -r["tpm"]))
for r in rows:
    print(f"    {r['variant']:<24}{r['n']:>5}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>6.2f}{r['pf_cost']:>9.2f}{r['mdd']*100:>6.1f}%"
          f"{r['prof']*100:>5.0f}%{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}")

# ── Success filter ───────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("  SUCCESS FILTER (user spec: t/mo>=8, prof%>=70, worst<=3, holPF>1.05, PF@cost>1.0)")
passed = [r for r in rows if (r["tpm"] >= 8 and r["prof"] >= 0.70 and r["worst"] <= 3
                              and r["holpf"] > 1.05 and r["pf_cost"] > 1.0)]
if passed:
    print(f"  ✅ {len(passed)} variants meet ALL user criteria:")
    for r in passed:
        print(f"    {r['variant']:<24} t/mo={r['tpm']:.1f} prof%={r['prof']*100:.0f}% "
              f"worst={r['worst']} PF={r['pf']:.2f} PF@cost={r['pf_cost']:.2f} holPF={r['holpf']:.2f}")
else:
    print("  ❌ NO variant meets all criteria. Closest:")
    near = sorted(rows, key=lambda r: -(0.4*(r["prof"]>=0.70) + 0.2*(r["worst"]<=3) +
                                        0.2*(r["tpm"]>=8) + 0.2*(r["holpf"]>1.05)))
    for r in near[:6]:
        print(f"    {r['variant']:<24} t/mo={r['tpm']:.1f} prof%={r['prof']*100:.0f}% "
              f"worst={r['worst']} PF={r['pf']:.2f} PF@cost={r['pf_cost']:.2f} holPF={r['holpf']:.2f}")

# ── Outputs ──────────────────────────────────────────────────────────────────
pd.DataFrame(rows).to_csv(os.path.join(OUT, "r081_scalp_sweep.csv"), index=False)
lines = [f"# R081 — High-Win-Rate Scalp Expansion\n",
         f"**Date:** 2026-08-06 | sub-1R targets (RR<1) — the untested dimension\n",
         f"**Target (user):** t/mo>=8, profitable-months%>=70, worst losing-month streak<=3, holPF>1.05\n",
         f"\n## Sweep (sorted by prof% then t/mo)\n",
         "| Variant | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['variant']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | "
                 f"{r['pf']:.2f} | {r['pf_cost']:.2f} | {r['mdd']*100:.1f}% | "
                 f"{r['prof']*100:.0f}% | {r['worst']} | {r['selpf']:.2f} | {r['holpf']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    lines.append("**✅ FOUND variants meeting the retail spec:**")
    for r in passed:
        lines.append(f"- **{r['variant']}**: {r['tpm']:.1f} t/mo, {r['prof']*100:.0f}% "
                     f"profitable months, worst streak {r['worst']}, PF {r['pf']:.2f} "
                     f"(cost {r['pf_cost']:.2f}), holPF {r['holpf']:.2f}")
else:
    lines.append("**❌ No variant meets ALL criteria** (t/mo>=8, prof%>=70, worst<=3, "
                 "holPF>1.05, PF@cost>1.0). Honest result: high WR + high frequency + "
                 "positive edge is very hard on this data after costs.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r081_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r081_*")
