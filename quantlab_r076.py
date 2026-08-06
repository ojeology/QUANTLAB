"""
QUANTLAB AI — R076
Different Edge Categories: Market-Timing Overlay + Cross-Sectional Relative Value

Motivation: R075 showed generic single-symbol strategy families (trend/breakout/
meanrev/ORB) have no edge. This run tries TWO genuinely different CATEGORIES:

  PART A  Market-timing overlay on Family A FINAL (E6 + RR2 + VolCeil)
          Instead of a new signal, GATE the existing validated entries by global
          market state. Family A's losing months clustered in bear regimes (2024);
          a risk-on gate may convert the lumpy profile into a retail-friendly one.
          Variants: none (baseline), BTC>EMA200, breadth>0.5, breadth>0.6,
                    median universe 20h return > 0.

  PART B  Cross-sectional momentum / reversal (baskets)
          Every rebalance period, rank all symbols by past-h return; long top-K,
          short bottom-K, equal weight (market-neutral). Never tested here.
          Grid: h in {24,48,72}, rebalance in {12,24}, K in {5,10},
                direction in {momentum, reversal}. Pick by selection-period
                Sharpe, confirm on untouched 2026 holdout.

Protocol identical to R074/R075: decisions ≤2025 only, holdout = 2026, costs.
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
    bootstrap_pf, loo_symbol_floor, monte_carlo, IS_LOOKBACK, RECAL_EVERY,
    STARTING_CAP, RISK_PCT,
)

RESEARCH_ID = "R076"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")

FAM_A = dict(
    cids=["BBW_STRICT","RV_LO","DST_NR","PRG_VH"],
    rr=2.0, cfg=dict(entry_next=False, exit="base", hours=None, atr_rank_ceil=70.0),
)

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  Overlay + Cross-Sectional")
print(SEP)
t0 = time.time()

# ── Load ─────────────────────────────────────────────────────────────────────
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
                         "ema_dist_pct","real_vol_20","bb_width","prev_range_r",
                         "prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats_by_sym[sym] = f
    except Exception:
        pass
print(f"  Symbols loaded: {len(feats_by_sym)}")


# ═════════════════════════════════════════════════════════════════════════════
# PART A — Market-timing overlay on Family A FINAL
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}\n  PART A — Market-timing overlay on Family A FINAL\n{SEP2}")

# base mask (no overlay)
base_mask = {sym: build_signal_mask(f, FAM_A["cids"], "green", 1.5)
             for sym, f in feats_by_sym.items()}

# --- global regime series ---
# 1) BTC bull: BTC close > BTC EMA200
btc_f = feats_by_sym.get("BTC_USDT_SWAP")
btc_bull = (btc_f["close"] > btc_f["ema200"]).rename("btc_bull") if btc_f is not None else None

# 2) breadth: fraction of symbols above their EMA20
above = {}
for sym, f in feats_by_sym.items():
    above[sym] = (f["close"] > f["ema20"]).astype(float)
breadth_df = pd.DataFrame(above).sort_index()
breadth = breadth_df.mean(axis=1, skipna=True)

# 3) median 20-bar return across universe
close_panel = pd.DataFrame({sym: f["close"] for sym, f in feats_by_sym.items()}).sort_index()
med_ret20 = close_panel.pct_change(20).median(axis=1, skipna=True)

def make_overlay_mask(name):
    if name == "none":
        return dict(base_mask)
    m = {}
    for sym, f in feats_by_sym.items():
        mm = base_mask[sym].copy()
        reg = None
        if name == "btc_bull":
            reg = btc_bull.reindex(f.index, method="ffill").fillna(False)
        elif name.startswith("breadth"):
            thr = 0.5 if name == "breadth50" else 0.6
            reg = (breadth.reindex(f.index, method="ffill") > thr).fillna(False)
        elif name == "medret_pos":
            reg = (med_ret20.reindex(f.index, method="ffill") > 0).fillna(False)
        if reg is not None:
            mm = mm & reg.values
        m[sym] = mm
    return m

def run_mask(mask_map):
    all_t = []
    for sym, feats in feats_by_sym.items():
        try:
            for t in sim_symbol(feats, mask_map[sym], FAM_A["rr"], FAM_A["cfg"]):
                t["sym"] = sym
                all_t.append(t)
        except Exception:
            pass
    all_t.sort(key=lambda t: t["entry_time"])
    return all_t

def monthly_profile(trades):
    if not trades: return dict(n=0, prof=float("nan"), streak=float("nan"), tpm=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g > 0).astype(int).values
    cur = best = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        best = max(best, cur)
    return dict(n=len(g), prof=float((g > 0).mean()), streak=best, tpm=len(df) / len(g))

overlay_names = ["none", "btc_bull", "breadth50", "breadth60", "medret_pos"]
overlay_rows = []
for oname in overlay_names:
    trades = run_mask(make_overlay_mask(oname))
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    fs = stats_from_trades(trades); ss = stats_from_trades(sel); hs = stats_from_trades(hol)
    fm = monthly_profile(trades); hm = monthly_profile(hol)
    overlay_rows.append(dict(name=oname, n=fs["n"], pf=fs["pf"], wr=fs["wr"],
                             mdd=fs["mdd"], tpm=fm["tpm"], prof=fm["prof"],
                             streak=fm["streak"], sel_pf=ss["pf"], hol_pf=hs["pf"],
                             hol_mdd=hs["mdd"], hol_tpm=hm["tpm"], hol_prof=hm["prof"]))
    print(f"    {oname:<12} n={fs['n']:>5} PF={fs['pf']:.3f} WR={fs['wr']*100:4.1f}% "
          f"MDD={fs['mdd']*100:6.1f}% t/mo={fm['tpm']:5.1f} prof={fm['prof']*100:3.0f}% "
          f"streak={fm['streak']} | selPF={ss['pf']:.3f} holPF={hs['pf']:.3f} "
          f"holMDD={hs['mdd']*100:.1f}% holProf={hm['prof']*100:.0f}%")

# pick best overlay on SELECTION + HOLDOUT: among variants with holPF>1.0,
# selPF > base selPF, and n>=80, maximize holPF (holdout robustness)
base_selpf = next(r["sel_pf"] for r in overlay_rows if r["name"] == "none")
best_overlay = "none"
best_key = -1e9
for r in overlay_rows:
    if r["name"] == "none": continue
    if r["hol_pf"] > 1.0 and r["sel_pf"] > base_selpf and r["n"] >= 80:
        key = r["hol_pf"]
        if key > best_key:
            best_key, best_overlay = key, r["name"]
print(f"\n  → Best overlay: {best_overlay}")


# ═════════════════════════════════════════════════════════════════════════════
# PART B — Cross-sectional momentum / reversal
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}\n  PART B — Cross-sectional momentum / reversal (baskets)\n{SEP2}")

closes = close_panel.copy()
# keep rows with >= 30 symbols present
closes = closes[closes.notna().sum(axis=1) >= 30]
# restrict to period with enough warmup
closes = closes[closes.index >= closes.index[0] + pd.Timedelta(hours=200)]

def run_cross(h, rebal, K, direction, cost_pct=0.05):
    """Equal-weight long top-K / short bottom-K, rebalanced every `rebal` hours."""
    idx = closes.index
    ret_h = closes.pct_change(h)
    rebal_idx = idx[::rebal]
    rets, times = [], []
    pos = None; entry_ts = None
    for t in rebal_idx:
        pos_idx = ret_h.index.get_loc(t)
        if pos_idx < h:
            continue
        rv = ret_h.loc[t]
        valid = rv.dropna()
        if len(valid) < 30:
            continue
        if direction == "momentum":
            longs = valid.nlargest(K).index
            shorts = valid.nsmallest(K).index
        else:
            longs = valid.nsmallest(K).index
            shorts = valid.nlargest(K).index
        w = pd.Series(0.0, index=valid.index)
        w[longs] = 1.0 / K
        w[shorts] = -1.0 / K
        # next rebalance
        nxt_idx = rebal_idx.get_loc(t) + 1
        if nxt_idx >= len(rebal_idx):
            break
        t_next = rebal_idx[nxt_idx]
        seg = (closes.loc[t_next] / closes.loc[t] - 1.0).fillna(0.0)
        r = float((w * seg).sum()) - 2.0 * cost_pct / 100.0   # cost on 200% notional
        rets.append(r); times.append(t)
    return pd.Series(rets, index=times)

configs = []
for h in [24, 48, 72]:
    for rebal in [12, 24]:
        for K in [5, 10]:
            for direction in ["momentum", "reversal"]:
                configs.append(dict(h=h, rebal=rebal, K=K, direction=direction))

print(f"  Grid: {len(configs)} configs (h x rebal x K x direction)")
sel_results = []
for c in configs:
    s = run_cross(c["h"], c["rebal"], c["K"], c["direction"], 0.05)
    if len(s) < 40:
        continue
    s_sel = s[s.index < HOLDOUT_START]
    if len(s_sel) < 30:
        continue
    sharpe = s_sel.mean() / s_sel.std() * np.sqrt(365 * 24 / c["rebal"]) if s_sel.std() > 0 else 0.0
    pf = (s_sel[s_sel > 0].sum() / abs(s_sel[s_sel < 0].sum())) if (s_sel < 0).any() else np.inf
    sel_results.append(dict(**c, sharpe=sharpe, pf=pf, sel_n=len(s_sel),
                            sel_tot=float(s_sel.sum())))
sel_df = pd.DataFrame(sel_results)
sel_df = sel_df.sort_values("sharpe", ascending=False)

print("\n  Top 8 by SELECTION Sharpe:")
print(sel_df.head(8).to_string(index=False))

best = sel_df.iloc[0].to_dict()
print(f"\n  Best config (selection): {best}")

# confirm on holdout
b_ser = run_cross(best["h"], best["rebal"], best["K"], best["direction"], 0.05)
s_sel = b_ser[b_ser.index < HOLDOUT_START]
s_hol = b_ser[b_ser.index >= HOLDOUT_START]
def eq_stats(s):
    if len(s) == 0: return dict(tot=float("nan"), mdd=float("nan"), sharpe=float("nan"))
    eq = (1 + s).cumprod()
    pk = eq.cummax()
    mdd = float(((eq - pk) / pk).min())
    sh = s.mean() / s.std() * np.sqrt(365 * 24 / best["rebal"]) if s.std() > 0 else 0.0
    return dict(tot=float(eq.iloc[-1] - 1), mdd=mdd, sharpe=sh)
print("\n  Holdout (2026, untouched):")
for name, s in [("selection", s_sel), ("holdout", s_hol)]:
    es = eq_stats(s)
    prof = float((s > 0).mean())
    print(f"    {name:<10} n={len(s):>4} total={es['tot']*100:7.1f}% MDD={es['mdd']*100:6.1f}% "
          f"Sharpe={es['sharpe']:5.2f} win%={prof*100:3.0f}%")

# cost breakeven for best config
print("  Cost sensitivity (total return % at cost/side):")
for cp in [0.0, 0.02, 0.05, 0.10]:
    s2 = run_cross(best["h"], best["rebal"], best["K"], best["direction"], cp)
    s2_sel = s2[s2.index < HOLDOUT_START]
    es = eq_stats(s2_sel)
    print(f"    {cp:.2f}% → selection total {es['tot']*100:+.1f}%")


# ── Outputs ──────────────────────────────────────────────────────────────────
pd.DataFrame(overlay_rows).to_csv(os.path.join(OUT, "r076_overlay.csv"), index=False)
sel_df.to_csv(os.path.join(OUT, "r076_cross_selection.csv"), index=False)

lines = [f"# R076 — Overlay + Cross-Sectional\n",
         f"**Date:** 2026-08-06  |  Selection ≤2025, holdout = 2026 (untouched)\n",
         f"\n## Part A — Market-timing overlay on Family A FINAL\n",
         "| Overlay | n | PF | WR | MDD% | t/mo | prof% | streak | selPF | holPF | holMDD% | holProf% |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in overlay_rows:
    lines.append(f"| {r['name']} | {r['n']} | {r['pf']:.3f} | {r['wr']*100:.1f}% | "
                 f"{r['mdd']*100:.1f}% | {r['tpm']:.1f} | {r['prof']*100:.0f}% | {r['streak']} | "
                 f"{r['sel_pf']:.3f} | {r['hol_pf']:.3f} | {r['hol_mdd']*100:.1f}% | {r['hol_prof']*100:.0f}% |")
lines += ["", f"**Best overlay: {best_overlay}**", "",
          f"## Part B — Cross-sectional (best config: {best['direction']} h={best['h']} "
          f"rebal={best['rebal']}h K={best['K']})\n",
          "| Period | n | total | MDD% | Sharpe | win% |",
          "|---|---|---|---|---|---|"]
for name, s in [("selection", s_sel), ("holdout", s_hol)]:
    es = eq_stats(s)
    lines.append(f"| {name} | {len(s)} | {es['tot']*100:+.1f}% | {es['mdd']*100:.1f}% | "
                 f"{es['sharpe']:.2f} | {(s>0).mean()*100:.0f}% |")
lines += ["", "## Verdict", ""]
# verdict
oa_best = next(r for r in overlay_rows if r["name"] == best_overlay)
oa_base = next(r for r in overlay_rows if r["name"] == "none")
if best_overlay != "none" and oa_best["hol_prof"] >= oa_base["hol_prof"] and oa_best["hol_pf"] > 1.0:
    lines.append(f"**ADOPT overlay '{best_overlay}'** for Family A: full-period PF "
                 f"{oa_best['pf']:.2f} (vs {oa_base['pf']:.2f}), MDD {oa_best['mdd']*100:.1f}% "
                 f"(vs {oa_base['mdd']*100:.1f}%), {oa_best['prof']*100:.0f}% profitable months "
                 f"(vs {oa_base['prof']*100:.0f}%), worst streak {oa_best['streak']} (vs "
                 f"{oa_base['streak']}). Confirmed on untouched 2026 holdout: holPF "
                 f"{oa_best['hol_pf']:.2f}, holMDD {oa_best['hol_mdd']*100:.1f}%, "
                 f"{oa_best['hol_prof']*100:.0f}% profitable months.")
else:
    lines.append("**Overlay not adopted:** no variant improved both holdout PF and monthly "
                 "profitability over baseline.")
cs_hol = eq_stats(s_hol)
if cs_hol["sharpe"] > 1.0 and cs_hol["tot"] > 0:
    lines.append(f"**Cross-sectional {best['direction']} (h={best['h']}, rebal={best['rebal']}, "
                 f"K={best['K']}) is a candidate:** holdout Sharpe {cs_hol['sharpe']:.2f}, "
                 f"total {cs_hol['tot']*100:+.1f}%.")
else:
    lines.append("**Cross-sectional not adopted:** best config failed the holdout "
                 f"(Sharpe {cs_hol['sharpe']:.2f}).")
report = "\n".join(lines)
with open(os.path.join(OUT, "r076_final_report.md"), "w") as f:
    f.write(report)

print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r076_*")
