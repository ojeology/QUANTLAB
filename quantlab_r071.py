"""
QUANTLAB AI — R071
Detailed RR Comparison: Full Bootstrap + Monthly Breakdown + Risk-Adjusted Metrics

Extends R070 Section 3 with:
  - Full bootstrap (2,000 samples) per RR per strategy
  - Monthly PF breakdown by RR
  - Risk-adjusted metrics: Calmar, Expectancy/MDD, Recovery Factor
  - Confidence intervals on PF difference vs RR=2.0 baseline
  - Clear deployment recommendation table

STRATEGIES ARE FROZEN. Entries unchanged. RR payout only.
"""

import os, sys, math, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx
warnings.filterwarnings("ignore")

RESEARCH_ID = "R071"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

STRATEGIES = {
    "FamilyA": {"label": "Family A",          "cids": ("BBW_STRICT","RV_LO","DST_NR","PRG_VH"), "color": "#f5a623"},
    "FamilyC": {"label": "Family C (ADX+PBD)", "cids": ("ADX_ST","PBD_HI"),                      "color": "#00c896"},
}

RR_SWEEP   = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
RR_BASE    = 2.0
TRADE_RISK = 100.0
IS_RATIO   = 0.80
N_FOLDS    = 5
N_BOOT     = 2_000
MIN_BARS   = 2_000
RAND_SEED  = 42
rng        = np.random.default_rng(RAND_SEED)

SEP  = "═" * 110
SEP2 = "─" * 90

C_BG   = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT = "#e0e0e0"
C_GRID = "#2a2a2a"; C_GREEN = "#00c896"; C_RED  = "#e05050"
C_GOLD = "#f5a623"; C_BLUE  = "#4a9eff"; C_PURP = "#9b59b6"

plt.rcParams.update({
    "figure.facecolor": C_BG, "axes.facecolor": C_PANEL,
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "axes.edgecolor": C_GRID, "grid.color": C_GRID, "font.family": "monospace",
})
def style_ax(ax):
    ax.set_facecolor(C_PANEL); ax.grid(True, ls="--", lw=0.4, color=C_GRID)
    for sp in ax.spines.values(): sp.set_edgecolor(C_GRID)
def save_fig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig); return p
def safe_pf(gw, gl):
    if gl == 0: return 999.0 if gw > 0 else 1.0
    return gw / gl
def max_dd(equity):
    eq = np.array(equity); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min())
def recovery_factor(equity):
    eq = np.array(equity)
    pk = np.maximum.accumulate(eq)
    mdd_abs = float((pk - eq).max())
    profit  = float(eq[-1] - eq[0])
    return profit / mdd_abs if mdd_abs > 0 else 999.0
def calmar(equity):
    """Annualised return proxy / |MDD|."""
    total_r = (np.array(equity)[-1] - np.array(equity)[0]) / np.array(equity)[0]
    mdd_abs = abs(max_dd(equity))
    return total_r / mdd_abs if mdd_abs > 0 else 999.0

COND_DEF = {
    "DST_NR":     ("ema_dist_pct", "lt_q", 0.33),
    "ADX_ST":     ("adx14",        "gt_q", 0.67),
    "PBD_HI":     ("prev_body_r",  "gt_q", 0.67),
    "BBW_STRICT": ("bb_width",     "lt_q", 0.25),
    "RV_LO":      ("real_vol_20",  "lt_q", 0.33),
    "PRG_VH":     ("prev_range_r", "gt_q", 0.80),
}
def apply_cond(df, cid, thr):
    col, direction, _ = COND_DEF[cid]
    v = df[col]
    return v < thr[f"{cid}_q"] if direction == "lt_q" else v > thr[f"{cid}_q"]
def compute_thresholds(df_is, cids):
    out = {}
    for cid in cids:
        col, _, param = COND_DEF[cid]
        out[f"{cid}_q"] = float(df_is[col].dropna().quantile(param))
    return out

def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; o = df["open"]
    df["ema200"]      = calc_ema(c, 200)
    df["atr14"]       = calc_atr(df, 14)
    bb_mid            = c.rolling(20).mean()
    bb_std            = c.rolling(20).std(ddof=0)
    df["bb_width"]    = (bb_std * 2) / bb_mid.replace(0, np.nan) * 100.0
    df["real_vol_20"] = c.pct_change().rolling(20).std() * 100.0
    ema200_s          = df["ema200"].replace(0, np.nan)
    df["ema_dist_pct"]= (c - ema200_s) / ema200_s * 100.0
    df["prev_range_r"]= (h.shift(1) - l.shift(1)).abs() / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"] = (c.shift(1) - o.shift(1)).abs() / c.shift(1).replace(0, np.nan) * 100.0
    df["adx14"]       = calc_adx(df, 14)
    df.dropna(subset=["ema200","atr14","real_vol_20","adx14","bb_width"], inplace=True)
    return df

def entry_gate(df):
    vol_avg = df["vol"].rolling(20).mean()
    return (df["vol"] > 1.5 * vol_avg) & (df["close"] > df["open"]) & \
           (df["close"] > df["close"].shift(1))

def backtest_sym(cids, df_feat, rr):
    n    = len(df_feat)
    is_e = int(n * IS_RATIO)
    df_is = df_feat.iloc[:is_e]
    df_oo = df_feat.iloc[is_e:]
    thr  = compute_thresholds(df_is, cids)
    gate = entry_gate(df_feat)
    masks = [apply_cond(df_feat, c, thr) for c in cids]
    sig   = masks[0].copy()
    for m in masks[1:]: sig = sig & m
    sig   = sig & gate
    sig_oo = sig.iloc[is_e:]
    n_oo   = len(df_oo)
    fsz    = max(1, n_oo // N_FOLDS)
    trades = []
    for idx in df_oo.index[sig_oo.values]:
        pos  = df_oo.index.get_loc(idx)
        if pos + 1 >= len(df_oo): continue
        ec   = df_oo["close"].iloc[pos + 1]
        en   = df_oo["close"].loc[idx]
        win  = ec > en
        pnl  = TRADE_RISK * rr if win else -TRADE_RISK
        fold = min(pos // fsz + 1, N_FOLDS)
        trades.append(dict(ts=idx, pnl=pnl, win=int(win), fold=fold))
    return trades

def run_all(cids, data, rr):
    all_trades = []
    for sym, df_raw in data.items():
        try:
            df_f = add_features(df_raw)
            if len(df_f) < MIN_BARS: continue
            for t in backtest_sym(cids, df_f, rr):
                t["sym"] = sym
            all_trades.extend(backtest_sym(cids, df_f, rr))
        except Exception:
            pass
    all_trades.sort(key=lambda t: t["ts"])
    return all_trades

def full_stats(trades, rr):
    if not trades:
        return dict(rr=rr, pf=0, wr=0, n=0, mdd=0, exp=0, rf=0, calmar_v=0,
                    boot_med=0, boot_p5=0, boot_p95=0,
                    ci_lo=0, ci_hi=0, p_better=0)
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    pf   = safe_pf(wins.sum(), abs(losses.sum()))
    wr   = float(sum(t["win"] for t in trades) / len(trades))
    eq   = np.cumsum(pnls) + 10000
    mdd  = max_dd(eq)
    rf   = recovery_factor(eq)
    cal  = calmar(eq)
    exp  = float(pnls.mean())
    # Bootstrap
    boot_pfs = []
    for _ in range(N_BOOT):
        s = rng.choice(pnls, size=len(pnls), replace=True)
        boot_pfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
    boot_arr = np.array(boot_pfs)
    return dict(
        rr=rr, pf=pf, wr=wr, n=len(trades), mdd=mdd, exp=exp,
        rf=rf, calmar_v=cal,
        boot_med=float(np.median(boot_arr)),
        boot_p5=float(np.percentile(boot_arr, 5)),
        boot_p95=float(np.percentile(boot_arr, 95)),
    )

def compare_vs_base(trades_rr, trades_base):
    """Bootstrap CI on PF(rr) - PF(base)."""
    pnls_rr   = np.array([t["pnl"] for t in trades_rr])
    pnls_base = np.array([t["pnl"] for t in trades_base])
    n_min = min(len(pnls_rr), len(pnls_base))
    diffs = []
    for _ in range(N_BOOT):
        s1 = rng.choice(pnls_rr,   size=n_min, replace=True)
        s2 = rng.choice(pnls_base, size=n_min, replace=True)
        diffs.append(
            safe_pf(s1[s1>0].sum(), abs(s1[s1<0].sum())) -
            safe_pf(s2[s2>0].sum(), abs(s2[s2<0].sum()))
        )
    d = np.array(diffs)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())

def monthly_pf(trades):
    months = defaultdict(list)
    for t in trades:
        ts = t["ts"]
        if hasattr(ts, "year"):
            months[(ts.year, ts.month)].append(t["pnl"])
    rows = {}
    for (yr, mo), pnls in sorted(months.items()):
        p = np.array(pnls)
        rows[f"{yr}-{mo:02d}"] = safe_pf(p[p>0].sum(), abs(p[p<0].sum()))
    return rows

# ─────────────────────────────────────────────────────────────────────────────
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  RR Deep Comparison"); print(SEP)
t0 = time.time()

print("\n  Loading data …")
data = {}
for fn in sorted(os.listdir(CACHE)):
    if not fn.endswith("_1H.parquet"): continue
    sym = fn.replace("_1H.parquet", "")
    try:
        df = pd.read_parquet(os.path.join(CACHE, fn))
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) >= MIN_BARS: data[sym] = df
    except Exception:
        pass
print(f"  Symbols: {len(data)}\n")

saved_charts = []
all_results  = {}    # sid → list of stat dicts
all_trades   = {}    # sid → rr → trades

for sid, scfg in STRATEGIES.items():
    print(SEP)
    print(f"  {scfg['label']}  ({' + '.join(scfg['cids'])})")
    print(SEP2)

    # Pre-run all RR and store trades
    all_trades[sid] = {}
    for rr in RR_SWEEP:
        all_trades[sid][rr] = run_all(scfg["cids"], data, rr)

    base_trades = all_trades[sid][RR_BASE]
    results_sid = []

    print(f"\n  {'RR':>6}  {'PF':>7}  {'Boot P5':>8}  {'Boot P95':>9}  "
          f"{'WR':>7}  {'Exp $':>8}  {'MDD':>7}  {'RF':>7}  {'Calmar':>7}  "
          f"{'vs RR=2.0':>12}  {'95% CI':>20}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*9}  "
          f"{'─'*7}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  "
          f"{'─'*12}  {'─'*20}")

    for rr in RR_SWEEP:
        trs  = all_trades[sid][rr]
        st   = full_stats(trs, rr)
        ci_lo, ci_hi, p_bet = compare_vs_base(trs, base_trades)
        st["ci_lo"] = ci_lo; st["ci_hi"] = ci_hi; st["p_better"] = p_bet

        if ci_lo > 0:   sig = "BETTER ✓"
        elif ci_hi < 0: sig = "WORSE  ✗"
        else:           sig = "no sig ─"

        base_marker = " ← BASE" if rr == RR_BASE else ""
        print(f"  {rr:>6.2f}  {st['pf']:>7.4f}  {st['boot_p5']:>8.4f}  {st['boot_p95']:>9.4f}  "
              f"{st['wr']:>7.1%}  {st['exp']:>8.2f}  {st['mdd']:>7.1%}  "
              f"{st['rf']:>7.2f}  {st['calmar_v']:>7.2f}  "
              f"{sig:>12}  [{ci_lo:+.3f}, {ci_hi:+.3f}]{base_marker}")
        results_sid.append(st)

    all_results[sid] = results_sid

    # ── Best RR by each metric ─────────────────────────────────────────────────
    print(f"\n  Best by metric:")
    for metric, label in [("pf","PF"), ("exp","Expectancy"), ("rf","Recovery F"),
                          ("calmar_v","Calmar"), ("boot_p5","Boot P5 (safety)")]:
        best = max(results_sid, key=lambda r: r[metric])
        print(f"    {label:<18}  RR={best['rr']:<4}  ({best[metric]:.4f})")

    # ── Monthly PF by RR ──────────────────────────────────────────────────────
    print(f"\n  Monthly PF breakdown by RR:")
    all_months = set()
    mo_pf_by_rr = {}
    for rr in RR_SWEEP:
        mp = monthly_pf(all_trades[sid][rr])
        mo_pf_by_rr[rr] = mp
        all_months.update(mp.keys())
    months_sorted = sorted(all_months)

    header = f"  {'Month':<10}" + "".join(f"  {rr:>6}" for rr in RR_SWEEP)
    print(header)
    print(f"  {'─'*10}" + "".join(f"  {'─'*6}" for _ in RR_SWEEP))
    for mo in months_sorted:
        row = f"  {mo:<10}"
        for rr in RR_SWEEP:
            v = mo_pf_by_rr[rr].get(mo, float("nan"))
            if math.isnan(v):    row += f"  {'N/A':>6}"
            elif v >= 1.5:       row += f"  {v:>6.2f}"   # green territory
            elif v >= 1.0:       row += f"  {v:>6.2f}"
            else:                row += f"  {v:>6.2f}"   # red territory
        print(row)
    print()

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP); print("  Generating charts …")

# ── Chart 1: Main RR comparison dashboard ────────────────────────────────────
fig, axes = plt.subplots(3, 6, figsize=(30, 16))
fig.suptitle("R071 — RR Deep Comparison: Full Bootstrap + Risk-Adjusted Metrics",
             fontsize=13, fontweight="bold", color=C_TEXT)

metrics_plot = [
    ("pf",        "Profit Factor",          None,   1.0),
    ("boot_p5",   "Bootstrap P5",           None,   1.0),
    ("exp",       "Expectancy $/trade",     None,   0.0),
    ("mdd",       "Max Drawdown",           None,  -0.20),
    ("rf",        "Recovery Factor",        None,   0.0),
    ("calmar_v",  "Calmar Ratio",           None,   0.0),
]

for col_idx, (metric, label, _, hline) in enumerate(metrics_plot):
    for row_idx, (sid, scfg) in enumerate(STRATEGIES.items()):
        ax = axes[row_idx][col_idx]; style_ax(ax)
        rs = all_results[sid]
        xs = [r["rr"] for r in rs]
        ys = [r[metric] for r in rs]
        col = scfg["color"]
        ax.plot(xs, ys, color=col, lw=2.5, marker="o", ms=6)
        # shade between p5 and p95 only for pf
        if metric == "pf":
            p5s  = [r["boot_p5"]  for r in rs]
            p95s = [r["boot_p95"] for r in rs]
            ax.fill_between(xs, p5s, p95s, color=col, alpha=0.12, label="P5–P95")
        if hline is not None:
            ax.axhline(hline, color=C_RED, lw=1, ls="--", alpha=0.6)
        ax.axvline(RR_BASE, color=C_GRID, lw=1, ls="--", alpha=0.7)
        ax.set_title(f"{scfg['label']} — {label}", fontsize=8, color=C_TEXT)
        ax.set_xlabel("RR", fontsize=7)
        ax.set_xticks(xs); ax.tick_params(axis="x", labelsize=6.5)

# Row 2: CI bars relative to RR=2.0
for idx, (sid, scfg) in enumerate(STRATEGIES.items()):
    ax = axes[2][idx * 2]; style_ax(ax)
    rs  = all_results[sid]
    xs_rr = [r["rr"] for r in rs]
    ci_lo = [r["ci_lo"] for r in rs]
    ci_hi = [r["ci_hi"] for r in rs]
    p_bet = [r["p_better"] for r in rs]
    ys    = [(lo + hi) / 2 for lo, hi in zip(ci_lo, ci_hi)]  # midpoint of CI
    errs_lo = [mid - lo for mid, lo in zip(ys, ci_lo)]
    errs_hi = [hi - mid for hi, mid in zip(ci_hi, ys)]
    bar_cols = [C_GREEN if lo > 0 else (C_RED if hi < 0 else C_GOLD)
                for lo, hi in zip(ci_lo, ci_hi)]
    ax.barh(range(len(xs_rr)), ys, xerr=[errs_lo, errs_hi],
            color=bar_cols, alpha=0.7, edgecolor=C_BG,
            error_kw=dict(ecolor=C_TEXT, capsize=3, lw=1))
    ax.set_yticks(range(len(xs_rr)))
    ax.set_yticklabels([f"RR={r}" for r in xs_rr], fontsize=7)
    ax.axvline(0, color=C_GRID, lw=1)
    ax.set_title(f"{scfg['label']} — ΔPF vs RR=2.0 (95% CI)", fontsize=8, color=C_TEXT)
    ax.set_xlabel("PF difference vs RR=2.0", fontsize=7)

# Row 2 right two panels: overlay PF curves on same axes for direct comparison
ax_cmp = axes[2][1]; style_ax(ax_cmp)
for sid, scfg in STRATEGIES.items():
    xs = [r["rr"] for r in all_results[sid]]
    ys = [r["pf"] for r in all_results[sid]]
    ax_cmp.plot(xs, ys, color=scfg["color"], lw=2, marker="o", ms=5,
                label=scfg["label"])
ax_cmp.axhline(1.0, color=C_RED, lw=1, ls="--", alpha=0.6)
ax_cmp.axvline(RR_BASE, color=C_GRID, lw=1, ls="--", alpha=0.7)
ax_cmp.set_title("Both Strategies — PF vs RR", fontsize=8, color=C_TEXT)
ax_cmp.set_xlabel("RR"); ax_cmp.set_ylabel("PF")
ax_cmp.legend(fontsize=7)
ax_cmp.set_xticks([r["rr"] for r in all_results["FamilyA"]])
ax_cmp.tick_params(axis="x", labelsize=6.5)

ax_exp = axes[2][3]; style_ax(ax_exp)
for _unused in [axes[2][4], axes[2][5]]: _unused.set_visible(False)
for sid, scfg in STRATEGIES.items():
    xs = [r["rr"] for r in all_results[sid]]
    ys = [r["exp"] for r in all_results[sid]]
    ax_exp.plot(xs, ys, color=scfg["color"], lw=2, marker="o", ms=5,
                label=scfg["label"])
ax_exp.axhline(0, color=C_RED, lw=1, ls="--", alpha=0.6)
ax_exp.set_title("Both Strategies — Expectancy vs RR", fontsize=8, color=C_TEXT)
ax_exp.set_xlabel("RR"); ax_exp.set_ylabel("Expectancy $/trade")
ax_exp.legend(fontsize=7)

plt.tight_layout()
saved_charts.append(save_fig(fig, "r071_rr_comparison.png"))
print("  → r071_rr_comparison.png")

# ── Chart 2: Equity curves at key RR levels ───────────────────────────────────
key_rrs  = [1.5, 2.0, 2.5, 3.0]
rr_cols  = [C_BLUE, C_GOLD, C_GREEN, C_RED]

fig2, axes2 = plt.subplots(2, len(key_rrs), figsize=(24, 10))
fig2.suptitle("R071 — OOS Equity Curves at Key RR Levels", fontsize=13,
              fontweight="bold", color=C_TEXT)

for row_i, (sid, scfg) in enumerate(STRATEGIES.items()):
    for col_i, rr in enumerate(key_rrs):
        ax = axes2[row_i][col_i]; style_ax(ax)
        trs = all_trades[sid][rr]
        if not trs:
            ax.set_title(f"No trades", fontsize=8); continue
        pnls = [t["pnl"] for t in trs]
        eq   = np.cumsum(pnls) + 10000
        col  = rr_cols[col_i]
        st   = full_stats(trs, rr)
        ax.plot(eq, color=col, lw=1.5)
        ax.axhline(10000, color=C_GRID, lw=0.8, ls="--")
        ax.fill_between(range(len(eq)), 10000, eq,
                        where=eq > 10000, color=col, alpha=0.10)
        ax.fill_between(range(len(eq)), 10000, eq,
                        where=eq < 10000, color=C_RED, alpha=0.10)
        ax.set_title(f"{scfg['label']} | RR={rr}\n"
                     f"PF={st['pf']:.3f}  Exp=${st['exp']:.0f}  MDD={st['mdd']:.1%}",
                     fontsize=8, color=C_TEXT)
        ax.set_ylabel("Portfolio $", fontsize=7)

plt.tight_layout()
saved_charts.append(save_fig(fig2, "r071_equity_by_rr.png"))
print("  → r071_equity_by_rr.png")

# ── Chart 3: Bootstrap P5 band across RR ─────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(18, 7))
fig3.suptitle("R071 — Bootstrap Safety Band (P5–P95) Across RR",
              fontsize=12, fontweight="bold", color=C_TEXT)

for ax_i, (sid, scfg) in enumerate(STRATEGIES.items()):
    ax = axes3[ax_i]; style_ax(ax)
    rs  = all_results[sid]
    xs  = [r["rr"] for r in rs]
    med = [r["boot_med"] for r in rs]
    p5  = [r["boot_p5"]  for r in rs]
    p95 = [r["boot_p95"] for r in rs]
    col = scfg["color"]
    ax.fill_between(xs, p5, p95, color=col, alpha=0.20, label="P5–P95 band")
    ax.plot(xs, med, color=col, lw=2.5, marker="o", ms=6, label="Median PF")
    ax.plot(xs, p5,  color=col, lw=1.5, ls="--", marker="s", ms=4, label="P5 (floor)")
    ax.axhline(1.0,     color=C_RED,  lw=1.5, ls="--", alpha=0.7, label="Break-even")
    ax.axhline(1.2,     color=C_GOLD, lw=1,   ls=":",  alpha=0.6, label="Criterion 1.20")
    ax.axvline(RR_BASE, color=C_GRID, lw=1,   ls="--", alpha=0.7, label="Current RR")
    ax.set_title(f"{scfg['label']} — Bootstrap PF Distribution vs RR",
                 fontsize=10, color=C_TEXT)
    ax.set_xlabel("RR", fontsize=9); ax.set_ylabel("PF", fontsize=9)
    ax.set_xticks(xs); ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

plt.tight_layout()
saved_charts.append(save_fig(fig3, "r071_bootstrap_band.png"))
print("  → r071_bootstrap_band.png")

# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(); print(SEP)
print("  RECOMMENDATION TABLE"); print(SEP2)

for sid, scfg in STRATEGIES.items():
    rs = all_results[sid]
    print(f"\n  [{scfg['label']}]\n")

    # Rank each RR on a composite score
    # Score = boot_p5 (safety) × 0.35 + normalised_exp × 0.30 + normalised_rf × 0.20 + (1+mdd) × 0.15
    max_exp = max(r["exp"] for r in rs); min_exp = min(r["exp"] for r in rs)
    max_rf  = max(r["rf"]  for r in rs)

    scored = []
    for r in rs:
        norm_exp = (r["exp"] - min_exp) / (max_exp - min_exp) if max_exp != min_exp else 0
        norm_rf  = r["rf"] / max_rf if max_rf > 0 else 0
        mdd_score = max(0, 1 + r["mdd"])  # 1 - |mdd|; mdd is negative
        score = (r["boot_p5"] / 3.0) * 0.35 + norm_exp * 0.30 + norm_rf * 0.20 + mdd_score * 0.15
        scored.append((r["rr"], score, r))

    scored.sort(key=lambda x: -x[1])
    recommended_rr = scored[0][0]

    print(f"  {'Rank':<5}  {'RR':<5}  {'PF':>7}  {'Boot P5':>8}  {'Exp $':>8}  "
          f"{'MDD':>7}  {'RF':>6}  {'Score':>7}  Notes")
    print(f"  {'─'*5}  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  "
          f"{'─'*7}  {'─'*6}  {'─'*7}  {'─'*20}")
    for rank, (rr, score, r) in enumerate(scored, 1):
        note = ""
        if rr == RR_BASE:      note += "CURRENT "
        if rank == 1:          note += "★ RECOMMENDED"
        if r["boot_p5"] < 1.0: note += "⚠ P5<1.0"
        print(f"  {rank:<5}  {rr:<5}  {r['pf']:>7.4f}  {r['boot_p5']:>8.4f}  "
              f"{r['exp']:>8.2f}  {r['mdd']:>7.1%}  {r['rf']:>6.2f}  "
              f"{score:>7.4f}  {note}")

    print(f"\n  → Recommended RR for {scfg['label']}: {recommended_rr}")
    if recommended_rr != RR_BASE:
        # Check if it's statistically convincing
        rec_st = next(r for r in rs if r["rr"] == recommended_rr)
        if rec_st["ci_lo"] > 0:
            print(f"    Statistically convincing improvement over RR=2.0 ✓")
            print(f"    CI: [{rec_st['ci_lo']:+.4f}, {rec_st['ci_hi']:+.4f}]  "
                  f"P(better)={rec_st['p_better']:.1%}")
        else:
            print(f"    NOT statistically convincing vs RR=2.0 — change is noise")
            print(f"    CI: [{rec_st['ci_lo']:+.4f}, {rec_st['ci_hi']:+.4f}]  "
                  f"P(better)={rec_st['p_better']:.1%}")
            print(f"    → Keep RR=2.0 until live data confirms")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY CSV + JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
print(); print(SEP); print("  Saving outputs …")
rows_csv = []
for sid, scfg in STRATEGIES.items():
    for r in all_results[sid]:
        rows_csv.append(dict(strategy=scfg["label"], **r))
pd.DataFrame(rows_csv).to_csv(os.path.join(OUT, "r071_rr_table.csv"), index=False)
print("  → r071_rr_table.csv")

elapsed = time.time() - t0
jpath = os.path.join(OUT, "r071_journal.md")
with open(jpath, "w") as f:
    f.write(f"# R071 — RR Deep Comparison\n\n")
    f.write(f"**Duration:** {elapsed:.0f}s  |  **Symbols:** {len(data)}  |  "
            f"**Bootstrap:** {N_BOOT:,} samples per RR\n\n")
    for sid, scfg in STRATEGIES.items():
        f.write(f"## {scfg['label']}\n\n")
        f.write(f"| RR | PF | Boot P5 | Boot P95 | Exp $ | MDD | RF | Calmar | vs 2.0 CI |\n")
        f.write(f"|---|---|---|---|---|---|---|---|---|\n")
        for r in all_results[sid]:
            ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
            f.write(f"| {r['rr']} | {r['pf']:.4f} | {r['boot_p5']:.4f} | "
                    f"{r['boot_p95']:.4f} | {r['exp']:.2f} | {r['mdd']:.1%} | "
                    f"{r['rf']:.2f} | {r['calmar_v']:.2f} | {ci} |\n")
        f.write("\n")
print("  → r071_journal.md")

# ─────────────────────────────────────────────────────────────────────────────
print(); print(SEP)
print(f"  R071 COMPLETE — {elapsed:.0f}s")
print(SEP)
print(f"\n  Files: {', '.join(os.path.basename(p) for p in saved_charts)}")
print(f"         r071_rr_table.csv  r071_journal.md\n")
