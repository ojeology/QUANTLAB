"""
QUANTLAB AI — R079
Trade-Frequency Expansion: can we get MORE trades without killing the edge?

Current locked config (R077): Family A + E6 + RR1.5 + VolCeil(<=70) + breadth50.
Frequency is low: ~4.3/month all-months (116 trades / 27 months). The two biggest
frequency killers are the breadth50 gate and the tight entry conditions.

Each variant relaxes ONE constraint (single-factor) or a small pre-registered combo.
Strict protocol: selection <=2025 decides, 2026 holdout confirms (holPF must stay >1).

Levers tested:
  breadth threshold : 0.30 / 0.35 / 0.40 / 0.45 (vs 0.50)
  volceil           : 80 / 90 (vs 70)
  rel_vol gate      : 1.3 / 1.0 (vs 1.5)
  BBW_STRICT -> BBW_LO (p25 -> p33)
Combos (pre-registered, sensible only):
  C1 breadth40 + volceil80 | C2 breadth40 + volceil80 + relvol1.3
  C3 breadth45 + volceil80 | C4 breadth40 + BBW_LO

Success criteria for adoption:
  - n increases meaningfully (target >=1.5x trades/month)
  - full PF >= 1.5, holPF > 1.0, MDD not much worse than -10%
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
    bootstrap_pf, COND_DEF, IS_LOOKBACK, RECAL_EVERY,
)

RESEARCH_ID = "R079"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")

CIDS = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
RR = 1.5
BASE_CFG = dict(entry_next=False, exit="base", hours=None)
# BBW_LO = same conditions but BBW p25 -> p33 (full COND_DEF with one override)
BBW_LO_DEF = dict(COND_DEF)
BBW_LO_DEF["BBW_STRICT"] = ("bb_width", "lt_q", 0.33)

# original 52
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
print(f"  QUANTLAB AI — {RESEARCH_ID}  Trade-Frequency Expansion")
print(SEP)
t0 = time.time()

# ── Load 52 symbols ──────────────────────────────────────────────────────────
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
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f
    except Exception:
        pass
print(f"  Symbols: {len(feats)}")

# breadth series (from full 52) — reusable at any threshold
above = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above).sort_index().mean(axis=1, skipna=True)

# ── Variant definitions ──────────────────────────────────────────────────────
def build(breadth_thr, volceil, relvol, cond_def):
    base_mask = {s: build_signal_mask(f, CIDS, "green", relvol, cond_def)
                 for s, f in feats.items()}
    final_mask = {}
    for s, m in base_mask.items():
        f = feats[s]
        reg = (breadth.reindex(f.index, method="ffill") > breadth_thr).fillna(False)
        final_mask[s] = m & reg.values
    cfg = dict(BASE_CFG, atr_rank_ceil=volceil)
    trades = []
    for sym, f in feats.items():
        try:
            for t in sim_symbol(f, final_mask[sym], RR, cfg):
                t["sym"] = sym
                trades.append(t)
        except Exception:
            pass
    trades.sort(key=lambda t: t["entry_time"])
    return trades

VARIANTS = [
    ("BASE",      dict(breadth_thr=0.50, volceil=70,  relvol=1.5, cond_def=None)),
    ("V01_br30",  dict(breadth_thr=0.30, volceil=70,  relvol=1.5, cond_def=None)),
    ("V02_br35",  dict(breadth_thr=0.35, volceil=70,  relvol=1.5, cond_def=None)),
    ("V03_br40",  dict(breadth_thr=0.40, volceil=70,  relvol=1.5, cond_def=None)),
    ("V04_br45",  dict(breadth_thr=0.45, volceil=70,  relvol=1.5, cond_def=None)),
    ("V05_vc80",  dict(breadth_thr=0.50, volceil=80,  relvol=1.5, cond_def=None)),
    ("V06_vc90",  dict(breadth_thr=0.50, volceil=90,  relvol=1.5, cond_def=None)),
    ("V07_rv13",  dict(breadth_thr=0.50, volceil=70,  relvol=1.3, cond_def=None)),
    ("V08_rv10",  dict(breadth_thr=0.50, volceil=70,  relvol=1.0, cond_def=None)),
    ("V09_bbwlo", dict(breadth_thr=0.50, volceil=70,  relvol=1.5, cond_def=BBW_LO_DEF)),
    ("C1_br40vc80",  dict(breadth_thr=0.40, volceil=80,  relvol=1.5, cond_def=None)),
    ("C2_br40vc80rv",dict(breadth_thr=0.40, volceil=80,  relvol=1.3, cond_def=None)),
    ("C3_br45vc80",  dict(breadth_thr=0.45, volceil=80,  relvol=1.5, cond_def=None)),
    ("C4_br40bbw",   dict(breadth_thr=0.40, volceil=70,  relvol=1.5, cond_def=BBW_LO_DEF)),
]

print("\n  Running variants …")
results = {}
for vname, kw in VARIANTS:
    results[vname] = build(**kw)

def monthly_profile(trades):
    if not trades: return dict(n_months=0, prof=float("nan"), tpm=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    return dict(n_months=len(g), prof=float((g > 0).mean()), tpm=len(df) / len(g))

print(f"\n{SEP2}")
print("  FREQUENCY vs EDGE — selection (≤2025) / holdout (2026)")
hdr = (f"    {'Variant':<14}{'n':>5}{'t/mo':>6}{'PF':>7}{'WR':>7}{'MDD%':>8}"
       f"{'prof%':>6}{'selPF':>7}{'holPF':>7}{'holMDD%':>9}")
print(hdr); print("    " + "─"*84)
rows = []
for vname in [v[0] for v in VARIANTS]:
    trades = results[vname]
    s = stats_from_trades(trades)
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]
    hs = stats_from_trades(hol)
    mp = monthly_profile(trades)
    span_months = 27.0
    tpm = len(trades) / span_months
    rows.append(dict(variant=vname, n=len(trades), tpm=tpm, pf=s["pf"], wr=s["wr"],
                     mdd=s["mdd"], prof=mp["prof"], selpf=sp, holpf=hs["pf"],
                     holmdd=hs["mdd"]))
    print(f"    {vname:<14}{len(trades):>5}{tpm:>6.1f}{s['pf']:>7.2f}"
          f"{s['wr']*100:>6.0f}%{s['mdd']*100:>7.1f}%{mp['prof']*100:>5.0f}%"
          f"{sp:>7.2f}{hs['pf']:>7.2f}{hs['mdd']*100:>8.1f}%")

# ── Ranking: edge-retained frequency winners ─────────────────────────────────
print(f"\n{SEP2}")
print("  CANDIDATES — holPF > 1.0, selPF > 1.0, PF >= 1.5, sorted by t/mo")
cands = [r for r in rows if r["holpf"] > 1.0 and r["selpf"] > 1.0 and r["pf"] >= 1.5]
cands.sort(key=lambda r: -r["tpm"])
for r in cands:
    print(f"    {r['variant']:<14} t/mo={r['tpm']:>5.1f}  PF={r['pf']:.2f}  "
          f"MDD={r['mdd']*100:.1f}%  prof%={r['prof']*100:.0f}%  "
          f"selPF={r['selpf']:.2f}  holPF={r['holpf']:.2f}  holMDD={r['holmdd']*100:.1f}%")

base = next(r for r in rows if r["variant"] == "BASE")
print(f"\n  BASE for reference: t/mo={base['tpm']:.1f}  PF={base['pf']:.2f}  "
      f"MDD={base['mdd']*100:.1f}%  prof%={base['prof']*100:.0f}%")

# ── Outputs ──────────────────────────────────────────────────────────────────
pd.DataFrame(rows).to_csv(os.path.join(OUT, "r079_frequency_sweep.csv"), index=False)
lines = [f"# R079 — Trade-Frequency Expansion\n",
         f"**Date:** 2026-08-06 | locked baseline: Family A + E6 + RR1.5 + VolCeil70 + breadth50\n",
         f"\n## Sweep (n = full-period trades, t/mo = all-months)\n",
         "| Variant | n | t/mo | PF | WR | MDD% | prof% | selPF | holPF | holMDD% |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['variant']} | {r['n']} | {r['tpm']:.1f} | {r['pf']:.2f} | "
                 f"{r['wr']*100:.0f}% | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} | {r['holmdd']*100:.1f}% |")
lines += ["", "## Candidates (holPF>1, selPF>1, PF>=1.5)", ""]
if cands:
    best = cands[0]
    lines.append(f"**Best frequency gain: {best['variant']}** — t/mo {best['tpm']:.1f} "
                 f"(vs base {base['tpm']:.1f}), PF {best['pf']:.2f}, MDD {best['mdd']*100:.1f}%, "
                 f"holPF {best['holpf']:.2f}")
    lines.append("")
    for r in cands:
        lines.append(f"- {r['variant']}: t/mo {r['tpm']:.1f}, PF {r['pf']:.2f}, "
                     f"holPF {r['holpf']:.2f}, holMDD {r['holmdd']*100:.1f}%")
else:
    lines.append("**No variant passed all criteria — no free lunch confirmed.**")
lines += ["", "## Verdict", ""]
if cands:
    best = cands[0]
    lines.append(f"Recommendation: adopt **{best['variant']}** if higher frequency is worth "
                 f"the (small) edge trade-off; otherwise keep BASE. Every gain in t/mo costs "
                 f"some PF/MDD — the table shows the exact price.")
else:
    lines.append("**No safe frequency increase found.** Every relaxation either failed the "
                 "holdout or dropped PF below 1.5. BASE (locked) remains the config.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r079_final_report.md"), "w") as f:
    f.write(report)

print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r079_*")
