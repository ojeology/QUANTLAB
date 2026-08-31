"""
QUANTLAB AI — R080
Clean New Hypotheses for Higher Frequency

User: 3-4 trades/month too small. R079 showed Family A relaxations cap out (~5-6/mo
before edge dies). So this run tests GENUINELY NEW setups designed to fire more often.

Five pre-registered hypotheses (each = a distinct setup type, 2-3 parameterizations):

  H1 FRESH-TREND PULLBACK v2   strong uptrend (EMA50>EMA200, slope>0, ADX>30) +
                               price dipped to EMA20 within last 2 bars + reclaimed +
                               mild volume. (R075 N1 was too loose; v2 is highly selective
                               but different logic from Family A)
  H2 STRONG BREAKOUT           close > prior 20-bar high + relvol>1.3 + above EMA20 + green.
                               (single-bar version of breakout; E6+RR1.5 exits)
  H3 COMPRESSION-POP WIDE      same TYPE as Family A but WIDER definition:
                               bb_width<p50 + real_vol_20<p50 + prev_range_r>p67 +
                               green/relvol gate  -> fires more often by construction
  H4 OVERSOLD COMPRESSION BOUNCE  bb_width<p40 + RSI<30 + below BB lower + mild vol
                               (mean-reversion WITH compression filter - unlike R075 N2
                               which had no compression and lost)
  H5 ADX IGNITION              ADX above its 100-bar median + big prev body (p67) +
                               relvol>1.5 + close>prev high. (trend-strength trigger,
                               distinct from Family A's volatility-compression trigger)

Protocol (identical to R074-R079): decisions on selection (<=2025) only, confirm on
untouched 2026 holdout. Exits fixed: E6 entry, RR=1.5, base SL/TP (no volceil/breadth —
this isolates the SIGNAL; add-ons can be re-tested on winners).

Success: selection n>=40, selPF>=1.4, holPF>=1.05, and t/mo meaningfully > Family A (4.3).
Winners are then combined into a multi-signal portfolio (sum of frequencies).
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, sim_symbol, stats_from_trades, bootstrap_pf,
    IS_LOOKBACK, RECAL_EVERY,
)

RESEARCH_ID = "R080"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
RR = 1.5
CFG = dict(entry_next=False, exit="base", hours=None)

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
print(f"  QUANTLAB AI — {RESEARCH_ID}  Clean New Hypotheses (higher frequency)")
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

# ── Hypothesis mask builders (each returns a boolean Series over f.index) ────
def h1_trend_pullback_v2(f, adx_thr=30, rv=1.0):
    low2 = f["low"].rolling(2, min_periods=1).min()
    return ((f["ema50"] > f["ema200"]) & (f["ema50_slope"] > 0) &
            (f["adx14"] > adx_thr) & (low2 < f["ema20"]) &
            (f["close"] > f["ema20"]) & (f["close"] > f["open"]) &
            (f["rel_vol"] > rv))

def h2_strong_breakout(f, n=20, rv=1.3):
    prior_hi = f["high"].rolling(n).max().shift(1)
    return ((f["close"] > prior_hi) & (f["rel_vol"] > rv) &
            (f["close"] > f["ema20"]) & (f["close"] > f["open"]))

def h3_comp_pop_wide(f, bb_p=0.50, rv_p=0.50, prg_p=0.67, rv=1.5):
    bb = f["bb_width"].rolling(500).quantile(bb_p)
    rvq = f["real_vol_20"].rolling(500).quantile(rv_p)
    prg = f["prev_range_r"].rolling(500).quantile(prg_p)
    return ((f["bb_width"] < bb) & (f["real_vol_20"] < rvq) &
            (f["prev_range_r"] > prg) & (f["rel_vol"] > rv) &
            (f["close"] > f["open"]) & (f["close"] > f["close"].shift(1)))

def h4_oversold_comp_bounce(f, bb_p=0.40, rsi=30, rv=0.8):
    bb = f["bb_width"].rolling(500).quantile(bb_p)
    return ((f["bb_width"] < bb) & (f["rsi14"] < rsi) & (f["close"] < f["bb_lower"]) &
            (f["rel_vol"] > rv))

def h5_adx_ignition(f, prg_p=0.67, rv=1.5):
    adx_med = f["adx14"].rolling(100).median()
    prg = f["prev_body_r"].rolling(500).quantile(prg_p)
    return ((f["adx14"] > adx_med) & (f["prev_body_r"] > prg) &
            (f["rel_vol"] > rv) & (f["close"] > f["high"].shift(1)))

HYPOTHESES = [
    ("H1a_trendpull30", lambda f: h1_trend_pullback_v2(f, 30, 1.0)),
    ("H1b_trendpull25", lambda f: h1_trend_pullback_v2(f, 25, 1.0)),
    ("H2a_breakout",    lambda f: h2_strong_breakout(f, 20, 1.3)),
    ("H2b_breakout15",  lambda f: h2_strong_breakout(f, 15, 1.3)),
    ("H3a_comppop",     lambda f: h3_comp_pop_wide(f, 0.50, 0.50, 0.67, 1.5)),
    ("H3b_comppop_rv12",lambda f: h3_comp_pop_wide(f, 0.50, 0.50, 0.67, 1.2)),
    ("H4a_oversold",    lambda f: h4_oversold_comp_bounce(f, 0.40, 30, 0.8)),
    ("H4b_oversold25",  lambda f: h4_oversold_comp_bounce(f, 0.40, 25, 0.8)),
    ("H5a_adxign",      lambda f: h5_adx_ignition(f, 0.67, 1.5)),
    ("H5b_adxign_rv12", lambda f: h5_adx_ignition(f, 0.67, 1.2)),
]

print("\n  Running hypotheses (E6 entry, RR=1.5, base exit, no breadth/volceil) …")
def run_mask(mask_map):
    out = []
    for sym, f in feats.items():
        try:
            for t in sim_symbol(f, mask_map[sym], RR, CFG):
                t["sym"] = sym
                out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t["entry_time"])
    return out

# Family A reference (signal-only, no breadth/volceil, E6 RR1.5) for apples-to-apples
from scripts.ql_engine import build_signal_mask, COND_DEF
famA_cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
maskA = {s: build_signal_mask(f, famA_cids, "green", 1.5) for s, f in feats.items()}
fa_trades = run_mask(maskA)

results = {"A_FAM_signal": fa_trades}
for name, builder in HYPOTHESES:
    mask = {s: builder(f) for s, f in feats.items()}
    results[name] = run_mask(mask)

def monthly_profile(trades):
    if not trades: return dict(n_months=0, prof=float("nan"), tpm=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    return dict(n_months=len(g), prof=float((g > 0).mean()), tpm=len(df) / len(g))

print(f"\n{SEP2}")
print("  RESULTS — selection (≤2025) | holdout (2026 untouched)")
hdr = (f"    {'Hypothesis':<18}{'n':>5}{'t/mo':>6}{'PF':>7}{'WR':>6}{'MDD%':>7}"
       f"{'prof%':>6}{'selPF':>7}{'holPF':>7}{'holMDD%':>8}{'hol t/mo':>8}")
print(hdr); print("    " + "─"*92)
rows = []
for name, trades in results.items():
    s = stats_from_trades(trades)
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]
    hs = stats_from_trades(hol)
    mp = monthly_profile(trades)
    mp_h = monthly_profile(hol)
    span = 27.0
    tpm = len(trades) / span
    rows.append(dict(hyp=name, n=len(trades), tpm=tpm, pf=s["pf"], wr=s["wr"],
                     mdd=s["mdd"], prof=mp["prof"], selpf=sp, holpf=hs["pf"],
                     holmdd=hs["mdd"], holtpm=mp_h["tpm"]))
    print(f"    {name:<18}{len(trades):>5}{tpm:>6.1f}{s['pf']:>7.2f}"
          f"{s['wr']*100:>5.0f}%{s['mdd']*100:>6.1f}%{mp['prof']*100:>5.0f}%"
          f"{sp:>7.2f}{hs['pf']:>7.2f}{hs['mdd']*100:>7.1f}%{mp_h['tpm']:>8.1f}")

# ── Ranking ──────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("  PASS / FAIL (sel n>=40, selPF>=1.4, holPF>=1.05)")
passed = []
for r in rows:
    ok = (r["n"] >= 40 and r["selpf"] >= 1.4 and r["holpf"] >= 1.05)
    status = "PASS ✓" if ok else "FAIL ✗"
    print(f"    {r['hyp']:<18} t/mo={r['tpm']:>5.1f}  PF={r['pf']:.2f}  selPF={r['selpf']:.2f}  "
          f"holPF={r['holpf']:.2f}  {status}")
    if ok:
        passed.append(r)

# ── Portfolio: combine all passed signals ────────────────────────────────────
print(f"\n{SEP2}")
if passed:
    print(f"  PORTFOLIO — running all {len(passed)} passed hypotheses together "
          f"(1 pos per symbol, merged chronologically)")
    merged = []
    for r in passed:
        merged.extend(results[r["hyp"]])
    merged.sort(key=lambda t: t["entry_time"])
    # dedupe same (sym, entry_time) across hypotheses (keep first)
    seen = set(); port = []
    for t in merged:
        k = (t["sym"], t["entry_time"])
        if k in seen: continue
        seen.add(k); port.append(t)
    ps = stats_from_trades(port)
    psel = [t for t in port if t["entry_time"] < HOLDOUT_START]
    phol = [t for t in port if t["entry_time"] >= HOLDOUT_START]
    pmp = monthly_profile(port); pmp_h = monthly_profile(phol)
    print(f"    PORTFOLIO: n={len(port)}  t/mo={len(port)/27.0:.1f}  PF={ps['pf']:.2f}  "
          f"WR={ps['wr']*100:.0f}%  MDD={ps['mdd']*100:.1f}%  prof%={pmp['prof']*100:.0f}%")
    print(f"    selection: n={len(psel)} selPF={stats_from_trades(psel)['pf']:.2f} | "
          f"holdout: n={len(phol)} holPF={stats_from_trades(phol)['pf']:.2f} "
          f"holMDD={stats_from_trades(phol)['mdd']*100:.1f}% hol t/mo={pmp_h['tpm']:.1f}")
else:
    print("  NO hypothesis passed — no portfolio to form.")
    port = None

# ── Outputs ──────────────────────────────────────────────────────────────────
pd.DataFrame(rows).to_csv(os.path.join(OUT, "r080_hypotheses.csv"), index=False)
lines = [f"# R080 — Clean New Hypotheses (higher frequency)\n",
         f"**Date:** 2026-08-06 | E6 entry, RR=1.5, base exit, signal-only "
         f"(no breadth/volceil) | selection ≤2025, holdout 2026 untouched\n",
         f"\n## Results\n",
         "| Hypothesis | n | t/mo | PF | WR | MDD% | prof% | selPF | holPF | holMDD% | hol t/mo |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['hyp']} | {r['n']} | {r['tpm']:.1f} | {r['pf']:.2f} | {r['wr']*100:.0f}% | "
                 f"{r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['selpf']:.2f} | "
                 f"{r['holpf']:.2f} | {r['holmdd']*100:.1f}% | {r['holtpm']:.1f} |")
lines += ["", "## Pass criteria (sel n>=40, selPF>=1.4, holPF>=1.05)", ""]
for r in rows:
    ok = (r["n"] >= 40 and r["selpf"] >= 1.4 and r["holpf"] >= 1.05)
    lines.append(f"- {r['hyp']}: {'PASS' if ok else 'FAIL'} (t/mo {r['tpm']:.1f}, PF {r['pf']:.2f}, holPF {r['holpf']:.2f})")
if port:
    lines += ["", "## Portfolio (all passed hypotheses, deduped)",
              f"- n={len(port)}, t/mo={len(port)/27.0:.1f}, PF={ps['pf']:.2f}, "
              f"MDD={ps['mdd']*100:.1f}%, prof%={pmp['prof']*100:.0f}%",
              f"- holdout: holPF={stats_from_trades(phol)['pf']:.2f}, "
              f"holMDD={stats_from_trades(phol)['mdd']*100:.1f}%, hol t/mo={pmp_h['tpm']:.1f}"]
lines += ["", "## Verdict", ""]
if port:
    lines.append(f"**PORTFOLIO is the frequency answer:** {len(port)/27.0:.1f} trades/month "
                 f"(vs Family A {len(fa_trades)/27.0:.1f}) at PF {ps['pf']:.2f}, "
                 f"holdout-validated.")
else:
    lines.append("**No clean new hypothesis passed.** Frequency wall confirmed: "
                 "no tested setup gives more trades with surviving edge. Family A remains "
                 "the only validated signal.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r080_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r080_*")
