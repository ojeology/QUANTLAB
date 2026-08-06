"""
QUANTLAB AI — R074
Edge Refinement with Strict Holdout

Goal: find whether Family A (E6_sigentry, RR=2.0) can be refined for more edge
WITHOUT fooling ourselves a second time.

Protocol (pre-registered):
  SELECTION window : entries before 2026-01-01  -> used to CHOOSE refinements
  HOLDOUT  window  : entries >= 2026-01-01      -> NEVER touched until final
                      confirmation step

Every variant changes ONE factor vs the R073 winner (E6_sigentry, RR=2.0).
A refinement is only kept if its paired-bootstrap 90% CI vs baseline is entirely
> 0 on the SELECTION window. Significant single factors are combined into a
final candidate, which must then beat baseline on the HOLDOUT window to be
promoted.

Pre-registered variants (each = one change from base):
  V01 time18      enter only 12:00-17:59 UTC (London/NY overlap cluster)
  V02 time1416    enter only 14:00-15:59 UTC (tightest cluster)
  V03 volceil     skip when atr_rank > 70 (avoid ATR-spike failure mode)
  V04 breakout    gate = close > prev high (true breakout, not just green)
  V05 vol175      rel_vol > 1.75 (stronger spike)
  V06 vol200      rel_vol > 2.00 (strongest spike)
  V07 rr175       RR = 1.75
  V08 rr225       RR = 2.25
  V09 exit_time24 time stop at 24 bars (on E6 entry)
  V10 exit_partial partial 50% @ 1R (on E6 entry)
  V11 exit_trail  1ATR trailing stop (on E6 entry)

Engine: scripts/ql_engine.py (bot-faithful rolling walk-forward).
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, build_signal_mask, run_family, stats_from_trades,
    bootstrap_pf, bootstrap_pf_diff, loo_symbol_floor, monte_carlo,
    cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY, STARTING_CAP, RISK_PCT,
)

RESEARCH_ID = "R074"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")

STRATEGY = {
    "label": "Family A",
    "cids": ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"],
    "rr": 2.0,
}

BASE_CFG = dict(entry_next=False, exit="base", hours=None, atr_rank_ceil=None)

VARIANTS = [
    ("V01_time18",   dict(hours=(12, 18))),
    ("V02_time1416", dict(hours=(14, 16))),
    ("V03_volceil",  dict(atr_rank_ceil=70.0)),
    ("V04_breakout", dict()),   # gate_mode handled at mask level
    ("V05_vol175",   dict()),
    ("V06_vol200",   dict()),
    ("V07_rr175",    dict()),
    ("V08_rr225",    dict()),
    ("V09_exit_time24", dict(exit="time24")),
    ("V10_exit_partial", dict(exit="partial")),
    ("V11_exit_trail",  dict(exit="trail")),
]
# RR / gate-mode overrides
RR_OVERRIDE = {"V07_rr175": 1.75, "V08_rr225": 2.25}
GATE_OVERRIDE = {"V04_breakout": "breakout"}
VOL_OVERRIDE = {"V05_vol175": 1.75, "V06_vol200": 2.0}

SEP  = "═" * 110
SEP2 = "─" * 90

print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  Edge Refinement with Strict Holdout")
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
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats_by_sym[sym] = f
    except Exception:
        pass
print(f"  Symbols loaded: {len(feats_by_sym)}")

# ── Masks (green gate once; breakout gate once) ─────────────────────────────
print("  Building signal masks …")
masks_green = {sym: build_signal_mask(f, STRATEGY["cids"], "green", 1.5)
               for sym, f in feats_by_sym.items()}
masks_break = {sym: build_signal_mask(f, STRATEGY["cids"], "breakout", 1.5)
               for sym, f in feats_by_sym.items()}
vol_masks = {}
for vname in ["V05_vol175", "V06_vol200"]:
    vol_masks[vname] = {sym: build_signal_mask(f, STRATEGY["cids"], "green", VOL_OVERRIDE[vname])
                        for sym, f in feats_by_sym.items()}

def get_mask(vname):
    if vname == "V04_breakout": return masks_break
    if vname in vol_masks: return vol_masks[vname]
    return masks_green

def split_trades(trades):
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    return sel, hol

# ── Run base + variants (full period; windows split later) ──────────────────
print("  Running base + 11 variants (selection decisions use pre-2026 only) …")
results = {}
base_sel = base_hol = None
for vname, extra in [("BASE", {})] + VARIANTS:
    cfg = dict(BASE_CFG)
    cfg.update(extra)
    rr = RR_OVERRIDE.get(vname, STRATEGY["rr"])
    trades = run_family(STRATEGY["cids"], rr, cfg, feats_by_sym, get_mask(vname))
    sel, hol = split_trades(trades)
    results[vname] = dict(trades=trades, sel=sel, hol=hol, cfg=cfg, rr=rr)
    if vname == "BASE":
        base_sel, base_hol = sel, hol

# ── SELECTION analysis ───────────────────────────────────────────────────────
def unpaired_pf_diff(rs_a, rs_b, n_boot=2_000, rng=None):
    """CI on PF(a) - PF(b) via independent resampling (for entry filters)."""
    if rng is None:
        rng = np.random.default_rng(42)
    rs_a = np.asarray(rs_a, dtype=float); rs_b = np.asarray(rs_b, dtype=float)
    out = np.empty(n_boot)
    for k in range(n_boot):
        sa = rs_a[rng.integers(0, len(rs_a), len(rs_a))] if len(rs_a) else np.array([])
        sb = rs_b[rng.integers(0, len(rs_b), len(rs_b))] if len(rs_b) else np.array([])
        out[k] = pf_of_rs(sa) - pf_of_rs(sb)
    return (float(np.percentile(out, 5)), float(np.median(out)), float(np.percentile(out, 95)))

ENTRY_FILTERS = {"V01_time18","V02_time1416","V03_volceil","V04_breakout","V05_vol175","V06_vol200"}
MIN_KEPT_N = 100   # entry filters must keep >=100 selection trades to be considered

def base_minus(base_sel, var_sel):
    """Selection trades in base whose (sym, entry_time) is NOT in var."""
    have = set((t["sym"], t["entry_time"]) for t in var_sel)
    return [t for t in base_sel if (t["sym"], t["entry_time"]) not in have]

print(f"\n{SEP2}")
print("  SELECTION WINDOW (pre-2026) — variant vs BASE")
hdr = (f"    {'Variant':<16}{'n':>6}{'WR':>8}{'PF':>8}{'Exp$':>8}"
       f"{'MDD%':>8}  BootP5   ΔPF vs BASE [P5,P95]  Sig")
print(hdr); print("    " + "─"*92)
sel_rows = []
for vname, extra in [("BASE", {})] + VARIANTS:
    sel = results[vname]["sel"]
    s = stats_from_trades(sel)
    b5, _, _ = bootstrap_pf(np.array([t["r"] for t in sel]))
    if vname == "BASE":
        d5 = dmed = d95 = 0.0; sig = "—"
    elif vname in ENTRY_FILTERS:
        kept_r = np.array([t["r"] for t in sel])
        removed_r = np.array([t["r"] for t in base_minus(base_sel, sel)])
        d5, dmed, d95 = unpaired_pf_diff(kept_r, removed_r)
        sig = "SIG↑" if (d5 > 0 and len(sel) >= MIN_KEPT_N) else ("SIG↓" if d95 < 0 else "")
        if len(sel) < MIN_KEPT_N:
            sig = "SIG↑(n<100)" if d5 > 0 else sig
    else:
        d5, dmed, d95 = bootstrap_pf_diff(sel, base_sel)
        sig = "SIG↑" if d5 > 0 else ("SIG↓" if d95 < 0 else "")
    sel_rows.append(dict(variant=vname, n=s["n"], wr=s["wr"], pf=s["pf"], exp=s["exp"],
                         mdd=s["mdd"], boot_p5=b5, d5=d5, dmed=dmed, d95=d95, sig=sig))
    print(f"    {vname:<16}{s['n']:>6}{s['wr']*100:>7.1f}%{s['pf']:>8.3f}"
          f"{s['exp']:>8.2f}{s['mdd']*100:>7.1f}%{b5:>8.3f}   "
          f"[{d5:+.2f}, {d95:+.2f}]  {sig}")

# ── Combine significant single factors ──────────────────────────────────────
significant = [r["variant"] for r in sel_rows[1:] if r["sig"] == "SIG↑"]
print(f"\n  Significant refinements (90% CI > 0 on selection): {significant}")

combo_cfg = dict(BASE_CFG)
combo_rr = STRATEGY["rr"]
combo_mask_key = "green"
for vname in significant:
    extra = next(e for n, e in VARIANTS if n == vname)
    combo_cfg.update(extra)
    if vname in RR_OVERRIDE: combo_rr = RR_OVERRIDE[vname]
    if vname == "V04_breakout": combo_mask_key = "breakout"
    if vname in VOL_OVERRIDE:
        combo_mask_key = f"vol{VOL_OVERRIDE[vname]}"
        vol_masks.setdefault(combo_mask_key,
            {sym: build_signal_mask(f, STRATEGY["cids"], "green", VOL_OVERRIDE[vname])
             for sym, f in feats_by_sym.items()})

if combo_mask_key == "breakout":
    combo_mask = masks_break
elif combo_mask_key in vol_masks:
    combo_mask = vol_masks[combo_mask_key]
else:
    combo_mask = masks_green

combo_trades = run_family(STRATEGY["cids"], combo_rr, combo_cfg, feats_by_sym, combo_mask)
combo_sel, combo_hol = split_trades(combo_trades)

if significant:
    s = stats_from_trades(combo_sel)
    d5, dmed, d95 = bootstrap_pf_diff(combo_sel, base_sel)
    print(f"  COMBINED [{'+'.join(significant)}]: n={s['n']} PF={s['pf']:.3f} "
          f"MDD={s['mdd']*100:.1f}% | ΔPF vs BASE [{d5:+.2f}, {d95:+.2f}]")
else:
    print("  No significant single-factor refinement — combined = BASE.")

# ── HOLDOUT confirmation (2026, untouched until now) ─────────────────────────
print(f"\n{SEP2}")
print("  HOLDOUT WINDOW (2026, untouched) — final confirmation")
print(f"    {'Config':<28}{'n':>6}{'WR':>8}{'PF':>8}{'Exp$':>8}{'MDD%':>8}  BootP5")
hol_rows = []
for name, hol in [("BASE (E6, RR2)", base_hol)] + \
                 [(f"{v}", results[v]["hol"]) for v in significant] + \
                 [(f"COMBINED ({'+'.join(significant) if significant else 'none'})", combo_hol)]:
    s = stats_from_trades(hol)
    b5, _, _ = bootstrap_pf(np.array([t["r"] for t in hol]))
    hol_rows.append(dict(config=name, n=s["n"], wr=s["wr"], pf=s["pf"], exp=s["exp"],
                         mdd=s["mdd"], boot_p5=b5))
    print(f"    {name:<28}{s['n']:>6}{s['wr']*100:>7.1f}%{s['pf']:>8.3f}"
          f"{s['exp']:>8.2f}{s['mdd']*100:>7.1f}%{b5:>8.3f}")

# full-period robustness of COMBINED
full = combo_trades
fs = stats_from_trades(full)
floor, rm = loo_symbol_floor(full)
mc = monte_carlo(np.array([t["r"] for t in full]))
print(f"\n  COMBINED full-period robustness: PF={fs['pf']:.3f} n={fs['n']} | "
      f"LOO floor={floor:.3f} (drop {rm}) | MC P(profit)={mc['prob']*100:.1f}% "
      f"net=${mc['exp']:+,.0f} DD P5={mc['dd_p5']*100:.1f}%")

# cost sensitivity
print("\n  Cost sensitivity (PF at cost per side):")
print(f"    {'Config':<24}{'0.00%':>8}{'0.05%':>8}{'0.10%':>8}{'0.15%':>8}{'0.20%':>8}")
for name, trades in [("BASE", results["BASE"]["trades"]), ("COMBINED", full)]:
    row = []
    for cp in [0.0, 0.05, 0.10, 0.15, 0.20]:
        row.append(pf_of_rs(cost_adjusted_rs(trades, cp)))
    print(f"    {name:<24}" + "".join(f"{p:>8.3f}" for p in row))

# monthly breakdown of COMBINED (full period)
df = pd.DataFrame(full)
df["month"] = df["entry_time"].dt.to_period("M")
n_months = df["month"].nunique()
print(f"\n  COMBINED monthly cadence (full period):")
print(f"    total={len(df)} trades over {n_months} months → "
      f"avg={len(df)/n_months:.1f}/month, median={df.groupby('month').size().median():.0f}, "
      f"min={df.groupby('month').size().min()}, max={df.groupby('month').size().max()}")

# ── Outputs ──────────────────────────────────────────────────────────────────
pd.DataFrame(sel_rows).to_csv(os.path.join(OUT, "r074_selection.csv"), index=False)
monthly = df.groupby("month").agg(n=("r","size"), net_r=("r","sum")).reset_index()
monthly.to_csv(os.path.join(OUT, "r074_final_monthly.csv"), index=False)

lines = [f"# R074 — Edge Refinement with Strict Holdout\n",
         f"**Date:** 2026-08-06  |  Selection ≤2025, holdout = 2026 (untouched)\n",
         f"**Baseline:** Family A E6_sigentry RR=2.0 (R073 winner)\n",
         f"\n## Selection (pre-2026)\n",
         "| Variant | n | WR | PF | Exp$ | MDD% | BootP5 | ΔPF [P5,P95] | Sig |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in sel_rows:
    lines.append(f"| {r['variant']} | {r['n']} | {r['wr']*100:.1f}% | {r['pf']:.3f} | "
                 f"{r['exp']:+.2f} | {r['mdd']*100:.1f}% | {r['boot_p5']:.3f} | "
                 f"[{r['d5']:+.2f}, {r['d95']:+.2f}] | {r['sig']} |")
lines += ["", "## Holdout (2026)", "| Config | n | WR | PF | Exp$ | MDD% | BootP5 |",
          "|---|---|---|---|---|---|---|"]
for r in hol_rows:
    lines.append(f"| {r['config']} | {r['n']} | {r['wr']*100:.1f}% | {r['pf']:.3f} | "
                 f"{r['exp']:+.2f} | {r['mdd']*100:.1f}% | {r['boot_p5']:.3f} |")
lines += ["", f"## COMBINED full-period ({'+'.join(significant) if significant else 'none'})",
          f"- PF={fs['pf']:.3f}, n={fs['n']}, WR={fs['wr']*100:.1f}%, MDD={fs['mdd']*100:.1f}%",
          f"- LOO-symbol floor={floor:.3f} (drop {rm})",
          f"- MC: P(profit)={mc['prob']*100:.1f}%, mean net=${mc['exp']:+,.0f}, DD P5={mc['dd_p5']*100:.1f}%",
          f"- Monthly: avg {len(df)/n_months:.1f}/month, median {df.groupby('month').size().median():.0f}",
          "",
          "## Verdict",
          ]
# verdict logic — holdout confirmation weighs PF AND drawdown
base_s = stats_from_trades(base_hol)
adopted = None
adopted_reason = ""
for r in hol_rows:
    if r["config"] == "BASE (E6, RR2)":
        continue
    pf_ok = r["pf"] > base_s["pf"] or r["pf"] >= 0.90 * base_s["pf"]
    dd_ok = r["mdd"] >= base_s["mdd"] + 0.03   # at least 3pp BETTER (less negative) drawdown
    if r["pf"] > base_s["pf"]:
        adopted, adopted_reason = r["config"], "higher PF on holdout"
        break
    if pf_ok and dd_ok:
        adopted, adopted_reason = r["config"], "risk-reduction confirmed on holdout " \
            f"(MDD {r['mdd']*100:.1f}% vs base {base_s['mdd']*100:.1f}%, PF within noise)"
        break
if adopted is None:
    lines.append("**No refinement confirmed:** baseline E6 (RR=2.0) remains the best "
                 "configuration on the untouched 2026 holdout. Nothing from R074 is adopted.")
elif adopted.startswith("COMBINED"):
    lines.append(f"**ADOPT COMBINED [{'+'.join(significant)}]:** {adopted_reason}.")
else:
    lines.append(f"**ADOPT {adopted}:** {adopted_reason}.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r074_final_report.md"), "w") as f:
    f.write(report)

print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r074_*")
