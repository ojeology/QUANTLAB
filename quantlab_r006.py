"""
=============================================================================
QUANTLAB AI – RESEARCH #006
Threshold Discovery — Liquidity Sweep Reversal

Objective:
  Run the exact same Liquidity Sweep Reversal backtest.
  Perform decile-level threshold analysis on the 6 strongest pre-entry
  features identified in Research #005.
  No trades are filtered.  No strategy is modified.
  This is purely descriptive statistical analysis to measure where
  natural performance breakpoints exist before any future hypothesis is formed.

Features analysed:
  1.  Distance From 20-Bar Low   (dist_from_ll_pct)
  2.  ATR Rank Percentile        (atr_rank_pct)
  3.  Distance From EMA200       (dist_from_ema_pct)
  4.  EMA200 Slope               (ema200_slope_pct)
  5.  Funding Rate               (funding_rate)
  6.  Hour of Day (UTC)          (hour_utc)

Engine, fees, costs, RR, position sizing, train/test: LOCKED — UNCHANGED.
=============================================================================
"""

import os
import sys
import math
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from datetime import datetime, timezone
from itertools import combinations

# ---------------------------------------------------------------------------
# Import locked engine from R004
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import (
    CONFIG, get_data, add_indicators, run_backtest,
    compute_metrics, monte_carlo, strategy_lsr,
    append_journal, _journal_row, _verdict_from_metrics,
)
from quantlab_r005 import (
    get_funding_rates, add_r005_indicators,
    enrich_trades_with_context, attach_funding_rate,
    FEATURE_NAMES,
)

RESEARCH_ID   = "R006"
OUTPUT_FOLDER = CONFIG["OUTPUT_FOLDER"]
BG            = "#0F1117"
N_BOOT        = 2000    # bootstrap iterations
N_DECILE      = 10      # decile groups
RR            = CONFIG["RISK_REWARD"]

# =============================================================================
# FEATURES UNDER ANALYSIS
# =============================================================================

R006_FEATURES = [
    ("dist_from_ll_pct",  "Distance From 20-Bar Low (%)",  "continuous"),
    ("atr_rank_pct",      "ATR Rank Percentile",            "continuous"),
    ("dist_from_ema_pct", "Distance From EMA200 (%)",       "continuous"),
    ("ema200_slope_pct",  "EMA200 Slope (%)",               "continuous"),
    ("funding_rate",      "Funding Rate",                   "continuous"),
    ("hour_utc",          "Hour of Day (UTC)",              "discrete"),
]

FEAT_COLS  = [f[0] for f in R006_FEATURES]
FEAT_LABEL = {f[0]: f[1] for f in R006_FEATURES}

# Interaction pairs requested
INTERACTIONS = [
    ("atr_rank_pct",      "hour_utc",         "ATR Rank vs Hour of Day"),
    ("dist_from_ema_pct", "atr_rank_pct",     "Dist EMA200 vs ATR Rank"),
    ("funding_rate",      "ema200_slope_pct", "Funding Rate vs EMA Slope"),
    ("dist_from_ll_pct",  "hour_utc",         "Dist from Low vs Hour"),
]


# =============================================================================
# SECTION 1 — PERFORMANCE HELPERS
# =============================================================================

def _group_metrics(sub: pd.DataFrame) -> dict:
    """Compute performance metrics for a sub-DataFrame of enriched trades."""
    n = len(sub)
    if n == 0:
        return dict(n=0, win_rate=0.0, profit_factor=0.0,
                    expectancy_r=0.0, avg_r=0.0,
                    net_pnl=0.0, max_drawdown=0.0)
    wins  = sub[sub["win"] == 1]
    loss  = sub[sub["win"] == 0]
    n_win = len(wins)
    wr    = n_win / n
    gw    = wins["pnl"].sum() if n_win > 0 else 0.0
    gl    = abs(loss["pnl"].sum()) if len(loss) > 0 else 1e-9
    pf    = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
    er    = wr * RR - (1.0 - wr)
    ar    = float(sub["r_multiple"].mean())
    net   = float(sub["pnl"].sum())

    # Drawdown on this sub-sequence
    equity = CONFIG["STARTING_CAPITAL"] + np.cumsum(sub["pnl"].values)
    peak   = np.maximum.accumulate(equity)
    dd     = ((equity - peak) / peak).min() if len(equity) > 0 else 0.0

    return dict(n=n, win_rate=wr, profit_factor=pf,
                expectancy_r=er, avg_r=ar,
                net_pnl=net, max_drawdown=dd)


# =============================================================================
# SECTION 2 — DECILE ANALYSIS
# =============================================================================

def _make_decile_bins(series: pd.Series, n_bins: int = N_DECILE,
                      discrete_hours: bool = False):
    """
    Return (bin_labels, bin_assignments) for a feature series.
    For hour_utc (discrete) we use fixed 4-hour blocks.
    For continuous features we use quantile-based equal-count bins.
    """
    valid = series.dropna()
    if len(valid) == 0:
        return [], pd.Series(dtype=object)

    if discrete_hours:
        # Fixed 4-hour UTC session blocks  →  6 groups
        edges  = [0, 4, 8, 12, 16, 20, 24]
        labels = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]
        bins   = pd.cut(series, bins=edges, labels=labels,
                        right=False, include_lowest=True)
        return labels, bins

    # Quantile cut — equal count, duplicate edges merged
    try:
        bins, edges = pd.qcut(series, q=n_bins, retbins=True,
                               duplicates="drop", labels=False)
        n_actual = int(bins.max()) + 1 if not bins.isna().all() else 0
        labels = []
        for i in range(n_actual):
            lo = edges[i]
            hi = edges[i + 1]
            labels.append(f"D{i+1}\n[{lo:.2f},{hi:.2f})")
        return labels, bins
    except Exception:
        return [], pd.Series(dtype=object)


def compute_decile_table(df: pd.DataFrame,
                         col: str,
                         n_bins: int = N_DECILE) -> pd.DataFrame:
    """
    Split df into deciles on `col` and return a DataFrame with one row
    per decile containing performance metrics.
    """
    discrete = (col == "hour_utc")
    labels, bins = _make_decile_bins(df[col], n_bins, discrete_hours=discrete)
    if not labels:
        return pd.DataFrame()

    df2 = df.copy()
    df2["_bin"] = bins

    rows = []
    for lbl in labels:
        sub = df2[df2["_bin"] == (lbl if discrete else labels.index(lbl))]
        m   = _group_metrics(sub)
        m["decile"] = lbl
        # Store midpoint of bin range for plotting (use rank for discrete)
        try:
            mid = float(df2.loc[df2["_bin"] == (lbl if discrete else labels.index(lbl)),
                                col].median())
        except Exception:
            mid = labels.index(lbl)
        m["midpoint"] = mid
        rows.append(m)

    return pd.DataFrame(rows)


# =============================================================================
# SECTION 3 — STATISTICAL TESTS
# =============================================================================

def bootstrap_mean_ci(arr: np.ndarray,
                      n_iter: int = N_BOOT,
                      ci: float = 95.0) -> tuple:
    """Return (mean, lower_ci, upper_ci) via bootstrap resampling."""
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng     = np.random.default_rng(42)
    samples = rng.choice(arr, size=(n_iter, len(arr)), replace=True)
    means   = samples.mean(axis=1)
    lo      = float(np.percentile(means, (100 - ci) / 2))
    hi      = float(np.percentile(means, 100 - (100 - ci) / 2))
    return float(arr.mean()), lo, hi


def compute_statistics(df: pd.DataFrame, col: str) -> dict:
    """
    Full statistical report for one feature vs binary outcome:
      - Pearson correlation (= point-biserial for binary outcome)
      - Cohen's d
      - Difference of means
      - Bootstrap 95% CI on means for wins and losses
      - Spearman rank correlation of decile rank vs PF (monotonicity test)
    """
    valid = df[[col, "win", "pnl", "r_multiple"]].dropna()
    if len(valid) < 4:
        return {}

    wins    = valid[valid["win"] == 1][col].values.astype(float)
    losses  = valid[valid["win"] == 0][col].values.astype(float)
    outcome = valid["win"].values.astype(float)
    feat    = valid[col].values.astype(float)

    # Pearson r
    corr = float(np.corrcoef(feat, outcome)[0, 1]) if feat.std() > 0 else 0.0

    # Cohen's d
    pooled_std = math.sqrt(
        ((len(wins) - 1) * wins.var(ddof=1) +
         (len(losses) - 1) * losses.var(ddof=1))
        / (len(wins) + len(losses) - 2)
    ) if (len(wins) > 1 and len(losses) > 1) else 1e-9
    cohens_d = (wins.mean() - losses.mean()) / pooled_std if pooled_std > 0 else 0.0

    # Bootstrap CI
    w_mean, w_lo, w_hi = bootstrap_mean_ci(wins)
    l_mean, l_lo, l_hi = bootstrap_mean_ci(losses)

    # Difference of means CI via bootstrap
    rng   = np.random.default_rng(42)
    diffs = []
    for _ in range(N_BOOT):
        ws = rng.choice(wins,   size=len(wins),   replace=True)
        ls = rng.choice(losses, size=len(losses), replace=True)
        diffs.append(ws.mean() - ls.mean())
    diffs     = np.array(diffs)
    diff_mean = float(diffs.mean())
    diff_lo   = float(np.percentile(diffs, 2.5))
    diff_hi   = float(np.percentile(diffs, 97.5))
    sig_diff  = not (diff_lo <= 0 <= diff_hi)   # CI excludes zero

    # Spearman rank of decile PF (monotonicity)
    dec_df  = compute_decile_table(valid, col)
    spear   = float("nan")
    if len(dec_df) > 2 and "profit_factor" in dec_df.columns:
        ranks  = np.arange(len(dec_df), dtype=float)
        pf_val = dec_df["profit_factor"].values.astype(float)
        ok     = np.isfinite(pf_val)
        if ok.sum() > 2:
            spear = float(np.corrcoef(ranks[ok], pf_val[ok])[0, 1])

    return {
        "col":          col,
        "label":        FEAT_LABEL.get(col, col),
        "n":            len(valid),
        "n_wins":       len(wins),
        "n_losses":     len(losses),
        "corr_r":       corr,
        "cohens_d":     cohens_d,
        "mean_wins":    w_mean,
        "mean_losses":  l_mean,
        "diff_mean":    diff_mean,
        "diff_ci_lo":   diff_lo,
        "diff_ci_hi":   diff_hi,
        "ci_excl_zero": sig_diff,
        "w_ci_lo":      w_lo,
        "w_ci_hi":      w_hi,
        "l_ci_lo":      l_lo,
        "l_ci_hi":      l_hi,
        "spearman_r":   spear,
    }


# =============================================================================
# SECTION 4 — INTERACTION ANALYSIS
# =============================================================================

def compute_interaction_grid(df: pd.DataFrame,
                              col_x: str, col_y: str,
                              n_bins: int = 5,
                              metric: str = "profit_factor") -> tuple:
    """
    Split df into n_bins × n_bins cells on (col_x, col_y) and compute
    `metric` in each cell.
    Returns (grid_matrix, x_labels, y_labels, count_matrix).
    """
    d_x = (col_x == "hour_utc")
    d_y = (col_y == "hour_utc")

    x_labels, bx = _make_decile_bins(df[col_x], n_bins, discrete_hours=d_x)
    y_labels, by = _make_decile_bins(df[col_y], n_bins, discrete_hours=d_y)

    if not x_labels or not y_labels:
        return None, [], [], None

    nx, ny    = len(x_labels), len(y_labels)
    grid      = np.full((ny, nx), np.nan)
    counts    = np.zeros((ny, nx), dtype=int)

    df2      = df.copy()
    df2["_bx"] = bx
    df2["_by"] = by

    # Bin index mapper
    def _idx(b_series, labels):
        if b_series.dtype.name == "category":
            # discrete (string labels)
            return {lbl: i for i, lbl in enumerate(labels)}
        else:
            return {i: i for i in range(len(labels))}

    for xi, xl in enumerate(x_labels):
        for yi, yl in enumerate(y_labels):
            if d_x:
                mask_x = df2["_bx"] == xl
            else:
                mask_x = df2["_bx"] == xi
            if d_y:
                mask_y = df2["_by"] == yl
            else:
                mask_y = df2["_by"] == yi

            sub = df2[mask_x & mask_y]
            m   = _group_metrics(sub)
            counts[yi, xi] = m["n"]
            if m["n"] >= 2:
                v = m.get(metric, np.nan)
                grid[yi, xi] = v if np.isfinite(v) and v < 10 else np.nan

    return grid, x_labels, y_labels, counts


# =============================================================================
# SECTION 5 — VISUALISATIONS
# =============================================================================

def _ax_style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=7)
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.grid(True, alpha=0.18, color="#444")


def _save(fig, fname: str) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


# ── Chart type: Threshold curves (4-panel per feature) ──────────────────────

def _plot_threshold_block(axes_row: list,
                          dec_df: pd.DataFrame,
                          col: str,
                          stats: dict) -> None:
    """
    Fill one row of 4 axes with decile threshold curves:
    [PF, Win Rate, Expectancy, Trade Count].
    """
    if dec_df is None or len(dec_df) == 0:
        return

    xlabels = dec_df["decile"].values
    x_pos   = np.arange(len(xlabels))
    col_d   = "#4A90D9"
    col_r   = "#FF4560"
    col_g   = "#00C49A"
    col_y   = "#FFB347"
    title   = FEAT_LABEL.get(col, col)

    panels = [
        (axes_row[0], dec_df["profit_factor"].values,
         "Profit Factor", 1.0,   col_d),
        (axes_row[1], dec_df["win_rate"].values * 100,
         "Win Rate (%)",  33.3,  col_g),
        (axes_row[2], dec_df["expectancy_r"].values,
         "Expectancy (R)", 0.0,  col_y),
        (axes_row[3], dec_df["n"].values,
         "Trade Count",   None,  col_r),
    ]

    for ax, vals, metric, ref, color in panels:
        _ax_style(ax)

        # Colour bars by above/below reference
        if ref is not None:
            bar_colors = [col_g if v > ref else col_r for v in vals]
        else:
            bar_colors = [color] * len(vals)

        ax.bar(x_pos, vals, color=bar_colors, alpha=0.85,
               width=0.7, zorder=2)

        # Overlay line
        finite = np.isfinite(vals.astype(float))
        if finite.sum() > 1:
            ax.plot(x_pos[finite], vals[finite],
                    color="white", lw=1.2, alpha=0.6,
                    marker="o", ms=3, zorder=3)

        if ref is not None:
            ax.axhline(ref, color="#FFD700", lw=0.9,
                       ls="--", alpha=0.8, zorder=4)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(xlabels, fontsize=5.5, rotation=30,
                            ha="right", color="white")
        ax.set_title(f"{title}\n{metric}", fontsize=7.5, pad=3)
        ax.set_ylabel(metric, fontsize=6.5)

        # Spearman annotation on PF panel
        if metric == "Profit Factor":
            sr = stats.get("spearman_r", float("nan"))
            if not math.isnan(sr):
                shape = "↗ monotone" if sr > 0.4 else \
                        "↘ monotone" if sr < -0.4 else "~nonlinear"
                ax.text(0.98, 0.97,
                        f"Spearman r={sr:+.2f}\n{shape}",
                        transform=ax.transAxes,
                        ha="right", va="top",
                        fontsize=6.5, color="#FFD700",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="#1A1D24", alpha=0.8))


def plot_threshold_curves(feature_set: list,
                           all_deciles: dict,
                           all_stats:   dict,
                           fname: str,
                           title: str) -> str:
    """
    Build one figure with len(feature_set) rows × 4 columns.
    """
    n_feat = len(feature_set)
    fig, axes = plt.subplots(n_feat, 4,
                              figsize=(18, 4.2 * n_feat))
    fig.patch.set_facecolor(BG)
    fig.suptitle(title, fontsize=11, fontweight="bold",
                 color="white", y=1.002)

    if n_feat == 1:
        axes = [axes]

    for row_idx, col in enumerate(feature_set):
        dec_df = all_deciles.get(col, pd.DataFrame())
        stats  = all_stats.get(col, {})
        _plot_threshold_block(axes[row_idx], dec_df, col, stats)

    plt.tight_layout(h_pad=2.0, w_pad=1.5)
    return _save(fig, fname)


# ── Chart type: Interaction heatmaps ────────────────────────────────────────

def plot_interaction_heatmaps(df: pd.DataFrame,
                               interactions: list) -> str:
    """
    2 × 2 grid of interaction heatmaps.
    Colour = Profit Factor.  Annotate cell with n trades.
    """
    n_pairs = len(interactions)
    ncols   = 2
    nrows   = math.ceil(n_pairs / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(14, 6 * nrows))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Feature Interaction Analysis — Profit Factor Heatmaps\n"
                 "Liq.Sweep Reversal  |  All Symbols  |  R006",
                 fontsize=10, fontweight="bold", color="white", y=1.01)

    flat = np.array(axes).flatten()

    for i, (cx, cy, pair_title) in enumerate(interactions):
        ax = flat[i]
        ax.set_facecolor(BG)

        grid, x_lbl, y_lbl, counts = compute_interaction_grid(
            df, cx, cy, n_bins=5, metric="profit_factor"
        )

        if grid is None or len(x_lbl) == 0:
            ax.set_visible(False)
            continue

        # Clip PF for colour scale
        grid_vis = np.where(np.isnan(grid), np.nan,
                            np.clip(grid, 0, 2.5))
        cmap = plt.cm.RdYlGn
        cmap.set_bad("#1a1a2e")
        masked = np.ma.masked_invalid(grid_vis)

        im = ax.imshow(masked, cmap=cmap, vmin=0.5, vmax=2.0,
                       aspect="auto", origin="upper")

        ax.set_xticks(range(len(x_lbl)))
        ax.set_yticks(range(len(y_lbl)))
        ax.set_xticklabels(x_lbl, color="white", fontsize=7, rotation=30, ha="right")
        ax.set_yticklabels(y_lbl, color="white", fontsize=7)
        ax.set_xlabel(FEAT_LABEL.get(cx, cx), color="white", fontsize=8)
        ax.set_ylabel(FEAT_LABEL.get(cy, cy), color="white", fontsize=8)
        ax.set_title(pair_title, fontsize=8.5, color="white", pad=4)

        # Annotate with PF and n
        for yi in range(len(y_lbl)):
            for xi in range(len(x_lbl)):
                n = int(counts[yi, xi])
                v = grid[yi, xi]
                if n == 0:
                    ax.text(xi, yi, "–", ha="center", va="center",
                            color="#555", fontsize=7)
                else:
                    pf_str = f"{v:.2f}" if np.isfinite(v) else "?"
                    col_txt = "black" if (np.isfinite(v) and (v > 1.5 or v < 0.5)) \
                              else "white"
                    ax.text(xi, yi,
                            f"{pf_str}\nn={n}",
                            ha="center", va="center",
                            color=col_txt, fontsize=6.5,
                            fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Profit Factor", color="white", fontsize=7)
        cbar.ax.tick_params(colors="white", labelsize=6)

    for j in range(n_pairs, len(flat)):
        flat[j].set_visible(False)

    plt.tight_layout()
    return _save(fig, "r006_interaction_heatmaps.png")


# ── Chart type: Bootstrap confidence intervals ───────────────────────────────

def plot_bootstrap_ci(all_stats: dict) -> str:
    """
    One panel per feature: error-bar plot of mean(wins) ± 95% CI
    and mean(losses) ± 95% CI, with difference CI band below.
    """
    feats  = [col for col in FEAT_COLS if col in all_stats]
    n_feat = len(feats)
    if n_feat == 0:
        return ""

    fig, axes = plt.subplots(n_feat, 1,
                              figsize=(10, 3.5 * n_feat))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Bootstrap 95% Confidence Intervals — Mean Feature Value\n"
                 "Wins vs Losses  |  Liq.Sweep  |  R006  "
                 f"(N={N_BOOT:,} iterations)",
                 fontsize=10, fontweight="bold", color="white")

    if n_feat == 1:
        axes = [axes]

    for ax, col in zip(axes, feats):
        _ax_style(ax)
        s = all_stats[col]

        # Error bars
        ax.errorbar([0], [s["mean_wins"]],
                    yerr=[[s["mean_wins"] - s["w_ci_lo"]],
                          [s["w_ci_hi"]  - s["mean_wins"]]],
                    fmt="o", color="#00C49A", ms=8, lw=2.5,
                    capsize=8, label=f"Wins  (mean={s['mean_wins']:.3f})")

        ax.errorbar([1], [s["mean_losses"]],
                    yerr=[[s["mean_losses"] - s["l_ci_lo"]],
                          [s["l_ci_hi"]     - s["mean_losses"]]],
                    fmt="o", color="#FF4560", ms=8, lw=2.5,
                    capsize=8, label=f"Losses (mean={s['mean_losses']:.3f})")

        # Difference CI band
        diff_sig = "★ CI EXCLUDES ZERO" if s.get("ci_excl_zero") else "CI includes zero"
        ax.axhspan(s["diff_ci_lo"] + s["mean_losses"],
                   s["diff_ci_hi"] + s["mean_losses"],
                   color="#FFD700", alpha=0.08, label=f"Diff CI  {diff_sig}")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Wins", "Losses"], color="white", fontsize=9)
        ax.set_title(
            f"{FEAT_LABEL.get(col, col)}\n"
            f"Δ={s['diff_mean']:+.3f}  "
            f"95% CI [{s['diff_ci_lo']:+.3f}, {s['diff_ci_hi']:+.3f}]  "
            f"| d={s['cohens_d']:+.3f}  r={s['corr_r']:+.3f}  {diff_sig}",
            fontsize=8, color="white"
        )
        ax.legend(fontsize=7, facecolor="#1A1D24",
                  edgecolor="#444", labelcolor="white")

    plt.tight_layout(h_pad=2.0)
    return _save(fig, "r006_bootstrap_ci.png")


# ── Chart type: Normalised PF comparison across all features ─────────────────

def plot_pf_overview(all_deciles: dict) -> str:
    """
    Single chart: normalised PF line for each feature across its deciles.
    Helps compare the shape of each curve (linear vs nonlinear).
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    _ax_style(ax)

    colors = ["#4A90D9", "#FF4560", "#00C49A",
              "#FFB347", "#E040FB", "#FFD700"]

    for (col, color) in zip(FEAT_COLS, colors):
        dec_df = all_deciles.get(col, pd.DataFrame())
        if dec_df is None or len(dec_df) < 2:
            continue
        pf   = dec_df["profit_factor"].values.astype(float)
        ok   = np.isfinite(pf)
        if ok.sum() < 2:
            continue
        x    = np.linspace(0, 1, ok.sum())
        ax.plot(x, pf[ok], color=color, lw=2.0, alpha=0.85,
                marker="o", ms=5,
                label=FEAT_LABEL.get(col, col))

    ax.axhline(1.0, color="#FFD700", lw=1.0, ls="--",
               alpha=0.7, label="PF = 1.0 (break-even)")
    ax.set_xlabel("Decile (low → high feature value)", color="white")
    ax.set_ylabel("Profit Factor", color="white")
    ax.set_title("Profit Factor Across Deciles — All 6 Features\n"
                 "Normalised X-axis  |  Liq.Sweep  |  R006",
                 fontsize=10, fontweight="bold", color="white")
    ax.legend(fontsize=8, facecolor="#1A1D24",
              edgecolor="#444", labelcolor="white")
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    return _save(fig, "r006_pf_overview.png")


# ── Chart type: Statistical summary heatmap ──────────────────────────────────

def plot_stats_summary(all_stats: dict) -> str:
    """
    Compact heatmap showing the 4 key statistics for all 6 features.
    Rows = features, columns = [|d|, |r|, Spearman, sig(diff)].
    """
    feats = [col for col in FEAT_COLS if col in all_stats]
    if not feats:
        return ""

    metrics_def = [
        ("abs_d",      "|Cohen's d|"),
        ("abs_r",      "|Pearson r|"),
        ("abs_spear",  "|Spearman r|"),
        ("sig",        "CI ≠ 0"),
    ]

    data = []
    row_labels = []
    for col in feats:
        s = all_stats[col]
        data.append([
            abs(s.get("cohens_d",   0.0)),
            abs(s.get("corr_r",     0.0)),
            abs(s.get("spearman_r", 0.0)),
            1.0 if s.get("ci_excl_zero") else 0.0,
        ])
        row_labels.append(FEAT_LABEL.get(col, col))

    mat  = np.array(data)
    col_labels = [m[1] for m in metrics_def]

    fig, ax = plt.subplots(figsize=(9, 0.8 * len(feats) + 2.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cmap = plt.cm.YlOrRd
    im   = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1,
                      aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, color="white", fontsize=9)
    ax.set_yticklabels(row_labels, color="white", fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v   = mat[i, j]
            tc  = "black" if v > 0.55 else "white"
            txt = f"{v:.2f}" if j < 3 else ("YES" if v > 0.5 else "no")
            ax.text(j, i, txt, ha="center", va="center",
                    color=tc, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Effect magnitude (0–1)", color="white", fontsize=8)
    cbar.ax.tick_params(colors="white")
    ax.set_title("Statistical Summary — All Features  |  R006\n"
                 "Liq.Sweep Reversal  |  All Symbols Combined",
                 fontsize=10, fontweight="bold", color="white")

    plt.tight_layout()
    return _save(fig, "r006_stats_summary.png")


# ── Chart type: Decile tables as printed plots ────────────────────────────────

def plot_decile_tables(all_deciles: dict) -> str:
    """
    6-panel figure: one table per feature showing the decile breakdown.
    """
    feats  = [col for col in FEAT_COLS if col in all_deciles]
    n_feat = len(feats)
    ncols  = 2
    nrows  = math.ceil(n_feat / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(18, 4.5 * nrows))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Decile Performance Tables — All Features  |  R006",
                 fontsize=10, fontweight="bold", color="white")

    flat = np.array(axes).flatten()

    for i, col in enumerate(feats):
        ax  = flat[i]
        ax.axis("off")
        ax.set_facecolor(BG)

        dec_df = all_deciles[col]
        if dec_df is None or len(dec_df) == 0:
            continue

        disp_cols  = ["decile", "n", "win_rate",
                      "profit_factor", "expectancy_r",
                      "avg_r", "net_pnl"]
        disp_hdr   = ["Decile", "N", "WR%",
                      "PF", "Exp(R)", "Avg R", "Net$"]

        table_data = []
        row_colors = []
        for _, row in dec_df.iterrows():
            pf = row["profit_factor"]
            rc = "#1a3a1a" if pf > 1.0 else "#3a1a1a"
            table_data.append([
                str(row["decile"]),
                str(int(row["n"])),
                f"{row['win_rate']:.1%}",
                f"{pf:.2f}",
                f"{row['expectancy_r']:+.2f}R",
                f"{row['avg_r']:+.2f}R",
                f"${row['net_pnl']:>6,.0f}",
            ])
            row_colors.append([rc] * len(disp_hdr))

        tbl = ax.table(
            cellText=table_data,
            colLabels=disp_hdr,
            cellLoc="center",
            loc="center",
            cellColours=row_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.35)

        # Style header
        for j in range(len(disp_hdr)):
            tbl[(0, j)].set_facecolor("#2a2a3a")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        for row_i in range(1, len(table_data) + 1):
            for col_j in range(len(disp_hdr)):
                tbl[(row_i, col_j)].set_text_props(color="white")

        ax.set_title(FEAT_LABEL.get(col, col),
                     fontsize=9, color="white", pad=4)

    for j in range(n_feat, len(flat)):
        flat[j].set_visible(False)

    plt.tight_layout(h_pad=1.5)
    return _save(fig, "r006_decile_tables.png")


# =============================================================================
# SECTION 6 — REPORT
# =============================================================================

def _linearity_verdict(spear_r: float, d: float) -> str:
    """Classify the PF-vs-decile relationship shape."""
    if math.isnan(spear_r):
        return "INSUFFICIENT DATA"
    if abs(spear_r) >= 0.7 and abs(d) >= 0.2:
        direction = "monotone ↗" if spear_r > 0 else "monotone ↘"
        return f"LIKELY LINEAR  ({direction}  Spearman r={spear_r:+.2f})"
    if abs(spear_r) < 0.35:
        return f"NONLINEAR  (Spearman r={spear_r:+.2f} — no monotone trend)"
    return f"MIXED / WEAK TREND  (Spearman r={spear_r:+.2f})"


def _breakpoint_summary(dec_df: pd.DataFrame, col: str) -> str:
    """Find the largest drop or jump in PF between adjacent deciles."""
    if dec_df is None or len(dec_df) < 3:
        return "insufficient deciles"
    pf   = dec_df["profit_factor"].values.astype(float)
    lbl  = dec_df["decile"].values
    diffs = np.diff(np.where(np.isfinite(pf), pf, np.nan))
    ok   = np.isfinite(diffs)
    if ok.sum() == 0:
        return "no finite adjacent pairs"
    idx  = int(np.nanargmax(np.abs(diffs)))
    size = float(diffs[idx])
    loc  = f"{lbl[idx]} → {lbl[idx+1]}"
    return f"largest Δ={size:+.2f}PF at {loc}"


def print_r006_report(all_deciles: dict,
                       all_stats:   dict,
                       combined_df: pd.DataFrame,
                       symbol_results: dict) -> None:
    S  = "=" * 104
    S2 = "─" * 104

    print(f"\n{S}")
    print("  QUANTLAB AI — RESEARCH #006")
    print("  Threshold Discovery — Liquidity Sweep Reversal")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(S)

    # ── Per-symbol summary ─────────────────────────────────────────────────
    print(f"\n  STRATEGY PERFORMANCE  (OOS — Locked Engine)")
    print(f"  {'─' * 90}")
    print(f"  {'Symbol':<24} {'Trades':>7} {'WR':>7} {'PF':>8} "
          f"{'Exp(R)':>9} {'Net$':>10} {'MDD':>8}  Verdict")
    print(f"  {'─' * 90}")
    for sym, res in symbol_results.items():
        m = res["metrics"]
        v = res["verdict"]
        s = "★" if v == "PROMOTE" else ("·" if v == "WEAK" else " ")
        print(f"  {sym:<24} {m['n_trades']:>7} "
              f"{m['win_rate']:>7.1%} {m['profit_factor']:>8.3f} "
              f"{m['expectancy_r']:>+9.3f}R ${m['net_profit']:>9,.0f} "
              f"{m['max_drawdown']:>8.2%}   {s}{v}")
    n_all = len(combined_df)
    n_win = int(combined_df["win"].sum())
    print(f"\n  Combined: {n_all} trades  |  {n_win} wins / {n_all-n_win} losses  "
          f"|  Overall WR {n_win/n_all:.1%}")

    # ── Statistical tests summary table ───────────────────────────────────
    print(f"\n  STATISTICAL TESTS — ALL FEATURES")
    print(f"  {'─' * 90}")
    print(f"  {'Feature':<36} {'r':>7} {'d':>7} {'ΔMean':>9} "
          f"{'95%CI':>18} {'Spearman':>10}  {'CIsig':>7}")
    print(f"  {'─' * 90}")
    for col in FEAT_COLS:
        s = all_stats.get(col, {})
        if not s:
            continue
        ci_str  = f"[{s['diff_ci_lo']:+.3f},{s['diff_ci_hi']:+.3f}]"
        sig_str = "★ YES" if s.get("ci_excl_zero") else "  no"
        sp_str  = f"{s['spearman_r']:+.3f}" if not math.isnan(s.get("spearman_r", float("nan"))) else "  N/A"
        print(f"  {FEAT_LABEL.get(col, col):<36} "
              f"{s['corr_r']:>+7.3f} "
              f"{s['cohens_d']:>+7.3f} "
              f"{s['diff_mean']:>+9.3f} "
              f"{ci_str:>18} "
              f"{sp_str:>10}  "
              f"{sig_str:>7}")
    print(f"  {'─' * 90}")
    print("  r = Pearson correlation with outcome  "
          "| d = Cohen's d  | CIsig = 95% bootstrap CI excludes zero")

    # ── Decile analysis per feature ────────────────────────────────────────
    print(f"\n  DECILE THRESHOLD ANALYSIS")
    for col in FEAT_COLS:
        dec_df = all_deciles.get(col, pd.DataFrame())
        s      = all_stats.get(col, {})
        if dec_df is None or len(dec_df) == 0:
            continue

        sp  = s.get("spearman_r", float("nan"))
        d   = s.get("cohens_d",   0.0)
        lv  = _linearity_verdict(sp, d)
        bp  = _breakpoint_summary(dec_df, col)

        print(f"\n  ── {FEAT_LABEL.get(col, col)}")
        print(f"     Shape: {lv}")
        print(f"     Breakpoint: {bp}")
        print(f"     {'Decile':<18} {'n':>4} {'WR':>7} {'PF':>8} "
              f"{'Exp(R)':>9} {'Avg R':>8} {'Net$':>10}")
        print(f"     {'─' * 70}")
        for _, row in dec_df.iterrows():
            pf  = row["profit_factor"]
            bar = "█" * min(12, max(0, int(pf * 4))) if np.isfinite(pf) else ""
            print(f"     {str(row['decile']):<18} {int(row['n']):>4} "
                  f"{row['win_rate']:>7.1%} {pf:>8.3f} "
                  f"{row['expectancy_r']:>+9.3f}R {row['avg_r']:>+8.3f}R "
                  f"${row['net_pnl']:>9,.0f}  {bar}")

    # ── Research questions ─────────────────────────────────────────────────
    print(f"\n{S}")
    print("  RESEARCH QUESTIONS — OBJECTIVE ANSWERS")
    print(f"  {'─' * 90}")

    # Rank features by |Cohen's d|
    ranked = sorted(
        [(col, all_stats[col]) for col in FEAT_COLS if col in all_stats],
        key=lambda x: abs(x[1].get("cohens_d", 0)),
        reverse=True,
    )
    strongest  = ranked[0]  if ranked else (None, {})
    weakest    = ranked[-1] if ranked else (None, {})

    # Q1: gradual vs threshold?
    print(f"\n  Q1  Does performance improve GRADUALLY or only beyond certain THRESHOLDS?")
    for col, s in ranked:
        dec_df = all_deciles.get(col, pd.DataFrame())
        sp     = s.get("spearman_r", float("nan"))
        d      = s.get("cohens_d", 0.0)
        lv     = _linearity_verdict(sp, d)
        bp     = _breakpoint_summary(dec_df, col)
        print(f"      {FEAT_LABEL.get(col, col):<36}  {lv}")
        print(f"{'':>6}  {bp}")

    # Q2: natural breakpoints?
    print(f"\n  Q2  Are there NATURAL BREAKPOINTS?")
    for col, s in ranked:
        dec_df = all_deciles.get(col, pd.DataFrame())
        if dec_df is None or len(dec_df) < 3:
            continue
        pf   = dec_df["profit_factor"].values.astype(float)
        lbl  = dec_df["decile"].values
        diffs = np.diff(np.where(np.isfinite(pf), pf, np.nan))
        ok   = np.isfinite(diffs)
        if ok.sum() == 0:
            continue
        max_d  = float(np.nanmax(np.abs(diffs)))
        max_i  = int(np.nanargmax(np.abs(diffs)))
        print(f"      {FEAT_LABEL.get(col, col):<36}  "
              f"Δ={diffs[max_i]:+.2f} at {lbl[max_i]} → {lbl[max_i+1]}"
              f"  {'← SHARP BREAKPOINT' if max_d > 0.4 else ''}")

    # Q3/Q4: linear vs nonlinear
    print(f"\n  Q3/Q4  LINEAR vs NONLINEAR relationships:")
    linear     = [(col, s) for col, s in ranked
                  if abs(s.get("spearman_r", 0)) >= 0.65]
    nonlinear  = [(col, s) for col, s in ranked
                  if abs(s.get("spearman_r", 0)) < 0.35 and
                     abs(s.get("cohens_d", 0)) > 0.1]
    mixed      = [(col, s) for col, s in ranked
                  if (col, s) not in linear and (col, s) not in nonlinear]

    print(f"      LINEAR (Spearman |r|≥0.65):")
    for col, s in (linear or [("—", {})]):
        print(f"        {FEAT_LABEL.get(col, col) if col != '—' else '  none'}")

    print(f"      NONLINEAR (Spearman |r|<0.35):")
    for col, s in (nonlinear or [("—", {})]):
        print(f"        {FEAT_LABEL.get(col, col) if col != '—' else '  none'}")

    print(f"      MIXED / WEAK:")
    for col, s in (mixed or [("—", {})]):
        print(f"        {FEAT_LABEL.get(col, col) if col != '—' else '  none'}")

    # Q5/Q6: strongest / weakest
    print(f"\n  Q5  STRONGEST predictive feature:")
    if strongest[0]:
        s = strongest[1]
        sig = " ★ bootstrap CI excludes zero" if s.get("ci_excl_zero") else ""
        print(f"      {FEAT_LABEL.get(strongest[0], strongest[0])}")
        print(f"      |d|={abs(s['cohens_d']):.3f}  |r|={abs(s['corr_r']):.3f}  "
              f"Spearman={s.get('spearman_r', float('nan')):+.3f}{sig}")

    print(f"\n  Q6  WEAKEST predictive feature:")
    if weakest[0]:
        s = weakest[1]
        print(f"      {FEAT_LABEL.get(weakest[0], weakest[0])}")
        print(f"      |d|={abs(s['cohens_d']):.3f}  |r|={abs(s['corr_r']):.3f}  "
              f"Spearman={s.get('spearman_r', float('nan')):+.3f}")

    # Interaction finding
    print(f"\n  INTERACTION NOTE:")
    print(f"      See r006_interaction_heatmaps.png for 2D feature-pair PF maps.")
    print(f"      Cells with n<2 are masked.  Look for off-diagonal hot/cold zones")
    print(f"      indicating conditions where two features amplify each other.")

    print(f"\n  NOTE: No filters have been added.  No strategy has been changed.")
    print(f"  These thresholds define the evidence base for Research #007.")
    print(S)


# =============================================================================
# SECTION 7 — SAVE DECILE TABLES TO CSV
# =============================================================================

def save_decile_csvs(all_deciles: dict) -> list:
    paths = []
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for col, dec_df in all_deciles.items():
        if dec_df is None or len(dec_df) == 0:
            continue
        safe = col.replace("_", "")
        path = os.path.join(OUTPUT_FOLDER, f"r006_decile_{safe}.csv")
        out  = dec_df.copy()
        out.insert(0, "feature", FEAT_LABEL.get(col, col))
        out.to_csv(path, index=False)
        paths.append(path)
    return paths


# =============================================================================
# SECTION 8 — MAIN PIPELINE
# =============================================================================

def process_symbol_r006(symbol: str, funding_df) -> dict:
    """Run Liq.Sweep + full enrichment on one symbol. Return results."""
    sep = "─" * 90
    print(f"\n{sep}\n  PROCESSING: {symbol}\n{sep}")

    df = get_data(symbol)
    n  = len(df)

    warm_up = CONFIG["EMA_LENGTH"] * 3 + 100
    if n < warm_up:
        print(f"  [SKIP] Insufficient candles ({n}).")
        return {}

    df = add_indicators(df)
    df = add_r005_indicators(df)

    split   = int(n * CONFIG["TRAIN_RATIO"])
    df_oos  = df.iloc[split:].reset_index(drop=True)

    oos_s = str(df_oos["datetime"].iloc[0].date())
    oos_e = str(df_oos["datetime"].iloc[-1].date())
    print(f"  Train : {df['datetime'].iloc[0].date()} → "
          f"{df['datetime'].iloc[split-1].date()} ({split:,} bars)")
    print(f"  OOS   : {oos_s} → {oos_e} ({len(df_oos):,} bars)")

    res     = run_backtest(df_oos, strategy_lsr, "Liq.Sweep")
    m       = compute_metrics(res["trades"], "Liq.Sweep")
    mc      = monte_carlo(m["pnls"], CONFIG["MC_ITERATIONS"])
    verdict = _verdict_from_metrics(m, mc)

    print(f"  Liq.Sweep  n={m['n_trades']:>4}  "
          f"PF {m['profit_factor']:>6.3f}  "
          f"Exp {m['expectancy_r']:>+6.3f}R  "
          f"MDD {m['max_drawdown']:.2%}  → {verdict}")

    attach_funding_rate(res["trades"], funding_df)
    enriched = enrich_trades_with_context(res["trades"], df_oos)
    enriched["symbol"] = symbol
    print(f"  Enriched {len(enriched)} trade records.")

    return {
        "metrics":  m,
        "mc":       mc,
        "verdict":  verdict,
        "enriched": enriched,
    }


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              QUANTLAB AI — RESEARCH #006                                      ║
║   Threshold Discovery — Liquidity Sweep Reversal                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  Features under analysis:
    1. Distance From 20-Bar Low (%)
    2. ATR Rank Percentile
    3. Distance From EMA200 (%)
    4. EMA200 Slope (%)
    5. Funding Rate
    6. Hour of Day (UTC)

  Engine, fees, costs, RR, position sizing, train/test: LOCKED — UNCHANGED.
  No trades filtered.  No strategy modified.  Descriptive analysis only.
""")

    random.seed(42)
    np.random.seed(42)

    symbols        = CONFIG["SYMBOLS"]
    symbol_results = {}
    all_enriched   = []

    for sym in symbols:
        print(f"\n[Funding Rates] {sym}")
        try:
            fund_df = get_funding_rates(sym)
        except Exception as e:
            print(f"  [WARN] {e}")
            fund_df = None

        try:
            res = process_symbol_r006(sym, fund_df)
            if res:
                symbol_results[sym] = res
                if len(res["enriched"]) > 0:
                    all_enriched.append(res["enriched"])
        except Exception as exc:
            import traceback
            print(f"\n  [ERROR] {sym}: {exc}")
            traceback.print_exc()

    if not all_enriched:
        print("\n  No trades. Cannot run threshold analysis.")
        return

    combined_df = pd.concat(all_enriched, ignore_index=True)
    n_total     = len(combined_df)
    print(f"\n  Combined: {n_total} trades across {len(all_enriched)} symbols")

    # ── Decile tables ──────────────────────────────────────────────────────
    print("\n  Computing decile tables…")
    all_deciles = {}
    for col in FEAT_COLS:
        if col not in combined_df.columns:
            print(f"  [SKIP] {col} not in data.")
            continue
        all_deciles[col] = compute_decile_table(combined_df, col)

    # ── Statistical tests ──────────────────────────────────────────────────
    print("  Running statistical tests & bootstrap CIs…")
    all_stats = {}
    for col in FEAT_COLS:
        if col not in combined_df.columns:
            continue
        all_stats[col] = compute_statistics(combined_df, col)

    # ── Charts ─────────────────────────────────────────────────────────────
    print("  Generating charts…")
    charts = []

    feats_1 = [col for col in FEAT_COLS[:3] if col in all_deciles]
    feats_2 = [col for col in FEAT_COLS[3:] if col in all_deciles]

    if feats_1:
        p = plot_threshold_curves(
            feats_1, all_deciles, all_stats,
            "r006_threshold_curves_1.png",
            "Threshold Curves — Dist from Low | ATR Rank | Dist EMA200  "
            "|  R006  |  Liq.Sweep",
        )
        charts.append(p); print(f"  → {p}")

    if feats_2:
        p = plot_threshold_curves(
            feats_2, all_deciles, all_stats,
            "r006_threshold_curves_2.png",
            "Threshold Curves — EMA Slope | Funding Rate | Hour of Day  "
            "|  R006  |  Liq.Sweep",
        )
        charts.append(p); print(f"  → {p}")

    p = plot_interaction_heatmaps(combined_df, INTERACTIONS)
    charts.append(p); print(f"  → {p}")

    p = plot_bootstrap_ci(all_stats)
    if p:
        charts.append(p); print(f"  → {p}")

    p = plot_pf_overview(all_deciles)
    if p:
        charts.append(p); print(f"  → {p}")

    p = plot_stats_summary(all_stats)
    if p:
        charts.append(p); print(f"  → {p}")

    p = plot_decile_tables(all_deciles)
    if p:
        charts.append(p); print(f"  → {p}")

    # ── Save CSVs ──────────────────────────────────────────────────────────
    csv_paths = save_decile_csvs(all_deciles)
    for cp in csv_paths:
        print(f"  → {cp}")

    # ── Report ─────────────────────────────────────────────────────────────
    print_r006_report(all_deciles, all_stats, combined_df, symbol_results)

    # ── Journal ────────────────────────────────────────────────────────────
    jnl_rows = []
    for sym, res in symbol_results.items():
        row = _journal_row("Liq.Sweep", sym, res["metrics"],
                           res["mc"], res["verdict"])
        row["research_id"] = RESEARCH_ID
        jnl_rows.append(row)
    if jnl_rows:
        append_journal(jnl_rows)
        print(f"\n  Research journal updated → {CONFIG['JOURNAL_FILE']}")
        print(f"  ({len(jnl_rows)} rows appended)")

    print(f"\n  All outputs → {OUTPUT_FOLDER}/")
    print("  Research #006 complete.\n")


if __name__ == "__main__":
    main()
