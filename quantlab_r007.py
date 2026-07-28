"""
=============================================================================
QUANTLAB AI – RESEARCH #007
Explainable Machine Learning — Liquidity Sweep Reversal

Objective:
  Use the completed trade dataset (same backtest engine, locked, unchanged).
  Train an explainable Random Forest classifier to understand which features
  and feature COMBINATIONS are associated with winning trades.

  This is NOT a trading AI.  It does NOT predict prices.
  It is an explainability experiment.  Truth over prediction.

Model:
  Random Forest Classifier
  70/30 chronological split  (no shuffling, no data leakage)
  Target: win=1, loss=0
  Features: all pre-entry context only

Engine, strategy, fees, costs, RR, sizing, entries, exits: LOCKED UNCHANGED.
=============================================================================
"""

import os
import sys
import math
import warnings
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Imports — locked engine + R005/R006 helpers
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
    FEATURE_NAMES as R005_FEAT_NAMES,
)

# ---------------------------------------------------------------------------
# scikit-learn
# ---------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    classification_report, brier_score_loss,
)
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
import shap

# =============================================================================
# CONSTANTS
# =============================================================================

RESEARCH_ID   = "R007"
OUTPUT_FOLDER = CONFIG["OUTPUT_FOLDER"]
BG            = "#0F1117"
RR            = CONFIG["RISK_REWARD"]

RANDOM_SEED   = 42
N_TREES       = 300
MAX_DEPTH     = 4          # shallow to limit overfitting on small N
MIN_SAMPLES_LEAF = 3       # each leaf needs ≥3 samples

# Session → numeric encoding
SESSION_ENC = {"Asia": 0, "London": 1, "New York": 2}

# Symbol → numeric encoding
SYMBOL_ENC  = {
    "BTC-USDT-SWAP": 0,
    "ETH-USDT-SWAP": 1,
    "SOL-USDT-SWAP": 2,
}

# Pre-entry features only (NO post-entry data: pnl, r_multiple, holding_minutes)
FEATURE_COLS = [
    "adx",
    "dist_from_ema_pct",
    "ema200_slope_pct",
    "atr_pct",
    "atr_rank_pct",
    "range_5_pct",
    "ret_vol_10",
    "candle_range_pct",
    "rel_vol",
    "dist_from_hh_pct",
    "dist_from_ll_pct",
    "funding_rate",
    "hour_utc",
    "day_of_week",
    "session_enc",     # Asia=0 London=1 NY=2
    "symbol_enc",      # BTC=0 ETH=1 SOL=2
]

FEATURE_LABELS = {
    "adx":               "ADX",
    "dist_from_ema_pct": "Dist EMA200 (%)",
    "ema200_slope_pct":  "EMA200 Slope (%)",
    "atr_pct":           "ATR (% price)",
    "atr_rank_pct":      "ATR Rank Pct",
    "range_5_pct":       "5-Bar Range (%)",
    "ret_vol_10":        "10-Bar Ret Vol",
    "candle_range_pct":  "Candle Range (%)",
    "rel_vol":           "Relative Volume",
    "dist_from_hh_pct":  "Dist 20-Bar HH (%)",
    "dist_from_ll_pct":  "Dist 20-Bar LL (%)",
    "funding_rate":      "Funding Rate",
    "hour_utc":          "Hour UTC",
    "day_of_week":       "Day of Week",
    "session_enc":       "Session",
    "symbol_enc":        "Symbol",
}


# =============================================================================
# SECTION 1 — DATA PIPELINE  (locked, identical to R006)
# =============================================================================

def collect_trades(symbols: list) -> pd.DataFrame:
    """Run Liq.Sweep backtest + enrichment on all symbols. Return combined df."""
    all_enriched = []
    symbol_meta  = {}

    for sym in symbols:
        print(f"\n[DATA] {sym}")
        try:
            fund_df = get_funding_rates(sym)
        except Exception as e:
            print(f"  [WARN] Funding: {e}")
            fund_df = pd.DataFrame(columns=["datetime", "funding_rate"])

        try:
            df  = get_data(sym)
            n   = len(df)
            df  = add_indicators(df)
            df  = add_r005_indicators(df)

            split   = int(n * CONFIG["TRAIN_RATIO"])
            df_oos  = df.iloc[split:].reset_index(drop=True)

            res     = run_backtest(df_oos, strategy_lsr, "Liq.Sweep")
            m       = compute_metrics(res["trades"], "Liq.Sweep")
            mc      = monte_carlo(m["pnls"], CONFIG["MC_ITERATIONS"])
            verdict = _verdict_from_metrics(m, mc)

            attach_funding_rate(res["trades"], fund_df)
            enriched = enrich_trades_with_context(res["trades"], df_oos)
            enriched["symbol"]     = sym
            enriched["symbol_enc"] = SYMBOL_ENC.get(sym, -1)
            enriched["session_enc"] = enriched["session"].map(SESSION_ENC).fillna(-1).astype(int)

            all_enriched.append(enriched)
            symbol_meta[sym] = {
                "metrics": m, "mc": mc, "verdict": verdict,
                "n": len(enriched),
            }
            print(f"  → {len(enriched)} trades  PF {m['profit_factor']:.3f}  {verdict}")

        except Exception as exc:
            import traceback
            print(f"  [ERROR] {sym}: {exc}")
            traceback.print_exc()

    if not all_enriched:
        raise RuntimeError("No trades collected.")

    combined = pd.concat(all_enriched, ignore_index=True)
    combined = combined.sort_values("entry_time").reset_index(drop=True)
    return combined, symbol_meta


# =============================================================================
# SECTION 2 — FEATURE MATRIX BUILDER
# =============================================================================

def build_feature_matrix(df: pd.DataFrame):
    """
    Extract pre-entry features and target from enriched trade DataFrame.
    Returns (X, y, feature_names) where X is numpy float32 array.
    Raises if post-entry columns are accidentally included.
    """
    POST_ENTRY_GUARD = {"pnl", "r_multiple", "holding_minutes",
                        "exit_type", "exit_time", "win"}

    available = [c for c in FEATURE_COLS if c in df.columns]
    guarded   = [c for c in available if c in POST_ENTRY_GUARD]
    if guarded:
        raise ValueError(f"Post-entry columns in feature set: {guarded}")

    sub = df[available + ["win"]].copy()
    sub = sub.dropna()

    X = sub[available].values.astype(np.float32)
    y = sub["win"].values.astype(int)
    return X, y, available


# =============================================================================
# SECTION 3 — CHRONOLOGICAL SPLIT  (no shuffling)
# =============================================================================

def chronological_split(X, y, df_ref, ratio: float = 0.70):
    """
    70/30 split preserving time order.
    df_ref is the source DataFrame aligned to X/y (post-dropna).
    Returns (X_train, X_test, y_train, y_test, split_idx).
    """
    n       = len(y)
    split   = int(n * ratio)
    return (
        X[:split], X[split:],
        y[:split], y[split:],
        split,
    )


# =============================================================================
# SECTION 4 — MODEL TRAINING
# =============================================================================

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train a shallow Random Forest with balanced class weights."""
    rf = RandomForestClassifier(
        n_estimators      = N_TREES,
        max_depth         = MAX_DEPTH,
        min_samples_leaf  = MIN_SAMPLES_LEAF,
        max_features      = "sqrt",
        class_weight      = "balanced",
        random_state      = RANDOM_SEED,
        n_jobs            = -1,
    )
    rf.fit(X_train, y_train)
    return rf


# =============================================================================
# SECTION 5 — EVALUATION
# =============================================================================

def evaluate_model(rf, X_train, X_test, y_train, y_test,
                   feature_names: list) -> dict:
    """Compute full evaluation metrics. Returns results dict."""
    y_pred_train  = rf.predict(X_train)
    y_prob_train  = rf.predict_proba(X_train)[:, 1]

    y_pred_test   = rf.predict(X_test)
    y_prob_test   = rf.predict_proba(X_test)[:, 1]

    def _metrics(y_true, y_pred, y_prob, split):
        n = len(y_true)
        try:
            roc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc = float("nan")
        return {
            "split":      split,
            "n":          n,
            "n_win":      int(y_true.sum()),
            "n_loss":     int((1 - y_true).sum()),
            "accuracy":   accuracy_score(y_true, y_pred),
            "precision":  precision_score(y_true, y_pred, zero_division=0),
            "recall":     recall_score(y_true, y_pred, zero_division=0),
            "f1":         f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":    roc,
            "brier":      brier_score_loss(y_true, y_prob),
            "cm":         confusion_matrix(y_true, y_pred),
            "y_true":     y_true,
            "y_pred":     y_pred,
            "y_prob":     y_prob,
        }

    # Permutation importance on test set
    perm = permutation_importance(
        rf, X_test, y_test,
        n_repeats=100,
        random_state=RANDOM_SEED,
        scoring="roc_auc",
    )

    return {
        "train":        _metrics(y_train, y_pred_train, y_prob_train, "train"),
        "test":         _metrics(y_test,  y_pred_test,  y_prob_test,  "test"),
        "gini_imp":     rf.feature_importances_,
        "perm_mean":    perm.importances_mean,
        "perm_std":     perm.importances_std,
        "feature_names": feature_names,
    }


# =============================================================================
# SECTION 6 — SHAP ANALYSIS
# =============================================================================

def compute_shap(rf, X_all, feature_names: list) -> dict:
    """
    Compute SHAP values for all observations using TreeExplainer.
    Uses the full dataset to maximise sample coverage for plotting.
    Returns dict with shap_values, expected_value, X_df.
    """
    explainer   = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_all, check_additivity=False)

    # shap_values may be:
    #   list [sv_class0, sv_class1]  — older shap
    #   ndarray (n, p, 2)            — newer shap 3D format
    #   ndarray (n, p)               — single-output regression style
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    X_df = pd.DataFrame(X_all, columns=feature_names)

    # Try interaction values (may be slow/memory-intensive on small N)
    try:
        shap_int = explainer.shap_interaction_values(X_all)
        if isinstance(shap_int, list):
            shap_int = shap_int[1]
    except Exception:
        shap_int = None

    return {
        "sv":            sv,          # (n_samples, n_features)
        "interaction":   shap_int,    # (n_samples, n_feat, n_feat) or None
        "expected_val":  float(explainer.expected_value[1])
                         if isinstance(explainer.expected_value, (list, np.ndarray))
                         else float(explainer.expected_value),
        "X_df":          X_df,
    }


# =============================================================================
# SECTION 7 — INTERACTION DISCOVERY
# =============================================================================

def discover_interactions(df: pd.DataFrame,
                           feature_names: list,
                           top_n: int = 8) -> pd.DataFrame:
    """
    Systematic pairwise interaction scan:
    For every feature pair, split each feature at its median.
    Compute performance in each of the 4 quadrants.
    Report the highest PF contrast (max_quadrant_PF - min_quadrant_PF).

    Returns a DataFrame of top interactions sorted by PF range.
    """
    def _pf(sub):
        n = len(sub)
        if n == 0:
            return 0.0, 0
        w = int(sub["win"].sum())
        gw = sub[sub["win"] == 1]["pnl"].sum() if w > 0 else 0.0
        gl = abs(sub[sub["win"] == 0]["pnl"].sum()) if (n - w) > 0 else 1e-9
        return (gw / gl if gl > 0 else float("inf") if gw > 0 else 0.0), n

    numeric_feats = [f for f in feature_names
                     if f in df.columns and df[f].dtype.kind in "fi"
                     and f not in ("session_enc", "symbol_enc")]

    rows = []
    for f1, f2 in itertools.combinations(numeric_feats, 2):
        sub = df[[f1, f2, "win", "pnl"]].dropna()
        if len(sub) < 8:
            continue
        m1 = float(sub[f1].median())
        m2 = float(sub[f2].median())

        q_results = {}
        for (lo1, hi1), (lo2, hi2), label in [
            ((True, True), (True, True),   "HH"),
            ((True, True), (False, False),  "HL"),
            ((False, False),(True, True),   "LH"),
            ((False, False),(False, False), "LL"),
        ]:
            mask1 = sub[f1] >= m1 if lo1 else sub[f1] < m1
            mask2 = sub[f2] >= m2 if lo2 else sub[f2] < m2
            pf_v, cnt = _pf(sub[mask1 & mask2])
            q_results[label] = {"pf": pf_v, "n": cnt}

        finite_pfs = [v["pf"] for v in q_results.values()
                      if np.isfinite(v["pf"]) and v["n"] >= 2]
        if len(finite_pfs) < 2:
            continue

        pf_range = max(finite_pfs) - min(finite_pfs)
        best_q   = max(q_results, key=lambda k: q_results[k]["pf"]
                       if np.isfinite(q_results[k]["pf"]) else -1)
        worst_q  = min(q_results, key=lambda k: q_results[k]["pf"]
                       if np.isfinite(q_results[k]["pf"]) else 99)

        rows.append({
            "f1":          f1,
            "f2":          f2,
            "f1_label":    FEATURE_LABELS.get(f1, f1),
            "f2_label":    FEATURE_LABELS.get(f2, f2),
            "pf_range":    pf_range,
            "best_q":      best_q,
            "best_pf":     q_results[best_q]["pf"],
            "best_n":      q_results[best_q]["n"],
            "worst_q":     worst_q,
            "worst_pf":    q_results[worst_q]["pf"],
            "worst_n":     q_results[worst_q]["n"],
            "all_q":       q_results,
        })

    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows).sort_values("pf_range", ascending=False).reset_index(drop=True)
    return result.head(top_n * 3)


# =============================================================================
# SECTION 8 — VISUALISATIONS
# =============================================================================

def _ax_style(ax, bg=None):
    ax.set_facecolor(bg or BG)
    ax.tick_params(colors="white", labelsize=8)
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


# ── Chart 1: Confusion Matrix + ROC ─────────────────────────────────────────

def plot_confusion_roc(eval_res: dict) -> str:
    tr = eval_res["train"]
    te = eval_res["test"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Random Forest — Classification Evaluation  |  R007  |  Liq.Sweep\n"
        f"Train: n={tr['n']} ({tr['n_win']}W/{tr['n_loss']}L)  "
        f"Test: n={te['n']} ({te['n_win']}W/{te['n_loss']}L)  "
        f"[Chronological 70/30 split — no shuffling]",
        fontsize=9, fontweight="bold", color="white",
    )

    # ── Panel 1: Test Confusion Matrix ─────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    cm  = te["cm"]
    cmap = plt.cm.Blues
    im  = ax.imshow(cm, cmap=cmap, vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Loss", "Pred Win"], color="white")
    ax.set_yticklabels(["True Loss", "True Win"], color="white")
    ax.set_title(
        f"Test Confusion Matrix\n"
        f"Acc={te['accuracy']:.2f}  F1={te['f1']:.2f}  "
        f"AUC={te['roc_auc']:.2f}",
        fontsize=8.5, color="white",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])),
                    ha="center", va="center",
                    color="white", fontsize=22, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.04)

    # ── Panel 2: ROC Curve ─────────────────────────────────────────────────
    ax2 = axes[1]
    _ax_style(ax2)
    for split, color, lw in [("train", "#4A90D9", 1.2), ("test", "#FFB347", 2.2)]:
        res = eval_res[split]
        try:
            fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
            ax2.plot(fpr, tpr, color=color, lw=lw,
                     label=f"{split.upper()}  AUC={res['roc_auc']:.3f}")
        except Exception:
            pass
    ax2.plot([0, 1], [0, 1], color="#555", lw=1.0, ls="--", label="Random")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve\n(gap = overfitting)", fontsize=9)
    ax2.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    # ── Panel 3: Calibration Curve ─────────────────────────────────────────
    ax3 = axes[2]
    _ax_style(ax3)
    for split, color, lw in [("train", "#4A90D9", 1.2), ("test", "#FFB347", 2.2)]:
        res = eval_res[split]
        try:
            n_bins = min(5, max(2, res["n"] // 4))
            frac_pos, mean_pred = calibration_curve(
                res["y_true"], res["y_prob"],
                n_bins=n_bins, strategy="uniform"
            )
            ax3.plot(mean_pred, frac_pos, color=color, lw=lw, marker="o",
                     ms=6, label=f"{split.upper()}  Brier={res['brier']:.3f}")
        except Exception:
            pass
    ax3.plot([0, 1], [0, 1], color="#555", lw=1.0, ls="--", label="Perfect cal.")
    ax3.set_xlabel("Mean Predicted Probability")
    ax3.set_ylabel("Fraction of Wins")
    ax3.set_title("Calibration Curve\n(diagonal = perfect)", fontsize=9)
    ax3.legend(fontsize=8, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

    plt.tight_layout()
    return _save(fig, "r007_confusion_roc_calibration.png")


# ── Chart 2: Feature Importance (Gini + Permutation) ───────────────────────

def plot_feature_importance(eval_res: dict) -> str:
    names  = [FEATURE_LABELS.get(f, f) for f in eval_res["feature_names"]]
    gini   = eval_res["gini_imp"]
    perm_m = eval_res["perm_mean"]
    perm_s = eval_res["perm_std"]

    # Sort by gini importance
    order_g = np.argsort(gini)
    order_p = np.argsort(perm_m)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(names) * 0.5 + 2)))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Feature Importance  |  Random Forest  |  R007  |  Liq.Sweep Reversal\n"
        "Left: Gini (Mean Decrease Impurity)  |  "
        "Right: Permutation Importance on test set (ROC-AUC drop)",
        fontsize=9, fontweight="bold", color="white",
    )

    y_pos = np.arange(len(names))

    # Panel 1: Gini
    ax = axes[0]
    _ax_style(ax)
    colors = plt.cm.YlOrRd(gini[order_g] / max(gini.max(), 1e-9))
    ax.barh(y_pos, gini[order_g], color=colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([names[int(i)] for i in order_g], color="white", fontsize=9)
    ax.set_xlabel("Gini Importance", color="white")
    ax.set_title("Gini (MDI)", fontsize=9)
    for i, v in enumerate(gini[order_g]):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", color="white", fontsize=7.5)

    # Panel 2: Permutation
    ax2 = axes[1]
    _ax_style(ax2)
    pm    = perm_m[order_p]
    ps    = perm_s[order_p]
    c_perm = ["#00C49A" if v > 0 else "#FF4560" for v in pm]
    ax2.barh(y_pos, pm, xerr=ps, color=c_perm, height=0.65,
             error_kw=dict(ecolor="white", alpha=0.5, lw=1.0))
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([names[int(i)] for i in order_p], color="white", fontsize=9)
    ax2.set_xlabel("Mean AUC Drop (permutation)", color="white")
    ax2.set_title("Permutation Importance (test set)", fontsize=9)
    ax2.axvline(0, color="#666", lw=0.8, ls="--")

    plt.tight_layout()
    return _save(fig, "r007_feature_importance.png")


# ── Chart 3: SHAP Summary Plot ───────────────────────────────────────────────

def plot_shap_summary(shap_res: dict) -> str:
    sv   = shap_res["sv"]
    X_df = shap_res["X_df"]

    renamed = X_df.rename(columns=FEATURE_LABELS)
    sv_df   = pd.DataFrame(sv, columns=[FEATURE_LABELS.get(c, c)
                                         for c in X_df.columns])

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(X_df.columns) * 0.55 + 2)))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "SHAP Analysis  |  Random Forest  |  R007  |  Liq.Sweep Reversal\n"
        "Left: Mean |SHAP| (global importance)  |  "
        "Right: SHAP value distribution per feature",
        fontsize=9, fontweight="bold", color="white",
    )

    feat_cols = [FEATURE_LABELS.get(c, c) for c in X_df.columns]
    mean_shap = np.abs(sv).mean(axis=0)
    order     = np.argsort(mean_shap)
    y_pos     = np.arange(len(feat_cols))

    # Panel 1: mean |SHAP|
    ax = axes[0]
    _ax_style(ax)
    bar_colors = plt.cm.plasma(mean_shap[order] / max(mean_shap.max(), 1e-9))
    ax.barh(y_pos, mean_shap[order], color=bar_colors, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat_cols[int(i)] for i in order], color="white", fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", color="white")
    ax.set_title("Global Feature Importance", fontsize=9)
    for i, v in enumerate(mean_shap[order]):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", color="white", fontsize=7.5)

    # Panel 2: SHAP dot strip (manual beeswarm substitute)
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    for sp in ax2.spines.values():
        sp.set_edgecolor("#333")
    ax2.tick_params(colors="white", labelsize=8)
    ax2.grid(True, alpha=0.12, color="#444", axis="x")

    cmap_beeswarm = plt.cm.RdYlGn
    for yi, fi in enumerate(order):
        fname  = feat_cols[fi]
        shap_v = sv[:, fi]
        feat_v = X_df.iloc[:, fi].values.astype(float)

        # Normalise feature to [0,1] for colouring
        fmin, fmax = feat_v.min(), feat_v.max()
        if fmax > fmin:
            feat_norm = (feat_v - fmin) / (fmax - fmin)
        else:
            feat_norm = np.full_like(feat_v, 0.5)

        # Jitter y slightly to avoid overplotting
        rng   = np.random.default_rng(42 + yi)
        jitter = rng.uniform(-0.25, 0.25, size=len(shap_v))
        cols   = cmap_beeswarm(feat_norm)

        ax2.scatter(shap_v, yi + jitter, c=cols, s=12, alpha=0.7,
                    linewidths=0, zorder=2)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([feat_cols[int(i)] for i in order], color="white", fontsize=9)
    ax2.set_xlabel("SHAP Value  (positive → predicts Win)", color="white")
    ax2.set_title("SHAP Value Distribution\n(colour = feature value: red=high, blue=low)",
                  fontsize=9)
    ax2.axvline(0, color="#FFD700", lw=0.9, ls="--", alpha=0.7)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_beeswarm,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax2, fraction=0.03, pad=0.02)
    cb.set_label("Feature value (low→high)", color="white", fontsize=7)
    cb.ax.tick_params(colors="white", labelsize=6)

    plt.tight_layout()
    return _save(fig, "r007_shap_summary.png")


# ── Chart 4: SHAP Dependence Plots (top-4 features) ─────────────────────────

def plot_shap_dependence(shap_res: dict, top_features: list) -> str:
    sv    = shap_res["sv"]
    X_df  = shap_res["X_df"]
    n_top = min(4, len(top_features))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "SHAP Dependence Plots — Top 4 Features  |  R007  |  Liq.Sweep Reversal\n"
        "X-axis = feature value at entry  |  "
        "Y-axis = SHAP contribution to Win probability  |  "
        "Colour = interaction feature",
        fontsize=9, fontweight="bold", color="white",
    )

    flat = np.array(axes).flatten()
    cmaps = [plt.cm.plasma, plt.cm.viridis, plt.cm.RdYlGn, plt.cm.coolwarm]

    for plot_i, feat in enumerate(top_features[:n_top]):
        if feat not in X_df.columns:
            continue
        ax     = flat[plot_i]
        _ax_style(ax)
        fi     = list(X_df.columns).index(feat)
        shap_f = sv[:, fi]
        feat_v = X_df[feat].values.astype(float)

        # Find the feature with highest SHAP interaction for colouring
        int_cols = [c for c in X_df.columns if c != feat]
        if int_cols:
            # Pick the column most correlated with residual SHAP for this feature
            resid = shap_f - np.polyval(np.polyfit(feat_v, shap_f, 1), feat_v)
            best_icol = max(
                int_cols,
                key=lambda c: abs(float(
                    np.corrcoef(resid, X_df[c].values.astype(float))[0, 1]
                )) if X_df[c].std() > 0 else 0.0
            )
            int_v    = X_df[best_icol].values.astype(float)
            int_min, int_max = int_v.min(), int_v.max()
            if int_max > int_min:
                int_norm = (int_v - int_min) / (int_max - int_min)
            else:
                int_norm = np.full_like(int_v, 0.5)
            colors = cmaps[plot_i](int_norm)
            color_lbl = FEATURE_LABELS.get(best_icol, best_icol)
        else:
            colors = "#4A90D9"
            color_lbl = ""

        sc = ax.scatter(feat_v, shap_f, c=colors, s=22, alpha=0.75,
                        linewidths=0.3, edgecolors="white", zorder=2)

        # Trend line
        try:
            z  = np.polyfit(feat_v, shap_f, 1)
            xr = np.linspace(feat_v.min(), feat_v.max(), 200)
            ax.plot(xr, np.polyval(z, xr), color="#FFD700",
                    lw=1.5, alpha=0.85, zorder=3)
        except Exception:
            pass

        ax.axhline(0, color="#888", lw=0.8, ls="--")
        ax.set_xlabel(FEATURE_LABELS.get(feat, feat), color="white", fontsize=9)
        ax.set_ylabel("SHAP value (→ Win)", color="white", fontsize=9)

        r = float(np.corrcoef(feat_v, shap_f)[0, 1]) if feat_v.std() > 0 else 0.0
        ax.set_title(
            f"{FEATURE_LABELS.get(feat, feat)}\n"
            f"r(SHAP)={r:+.3f}  colour={color_lbl}",
            fontsize=8.5, color="white",
        )

        if color_lbl:
            sm = plt.cm.ScalarMappable(cmap=cmaps[plot_i],
                                        norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label(f"{color_lbl} (low→high)", color="white", fontsize=6.5)
            cb.ax.tick_params(colors="white", labelsize=6)

    for j in range(n_top, 4):
        flat[j].set_visible(False)

    plt.tight_layout()
    return _save(fig, "r007_shap_dependence.png")


# ── Chart 5: SHAP Interaction Matrix ─────────────────────────────────────────

def plot_shap_interaction_matrix(shap_res: dict) -> str:
    sv   = shap_res["sv"]
    X_df = shap_res["X_df"]
    n    = sv.shape[1]

    feat_labels = [FEATURE_LABELS.get(c, c) for c in X_df.columns]

    # Build interaction proxy: for each pair (i,j), correlation of
    # sign(sv_i) with feat_j (how much does feat_j modulate feat_i's SHAP)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
                continue
            fi = X_df.iloc[:, i].values.astype(float)
            fj = X_df.iloc[:, j].values.astype(float)
            svi = sv[:, i]
            if fi.std() > 0 and fj.std() > 0 and svi.std() > 0:
                mat[i, j] = float(np.corrcoef(svi, fj)[0, 1])

    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), max(9, n * 0.65)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cmap = plt.cm.RdYlGn
    im   = ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feat_labels, rotation=45, ha="right",
                       color="white", fontsize=8)
    ax.set_yticklabels(feat_labels, color="white", fontsize=8)

    for i in range(n):
        for j in range(n):
            v  = mat[i, j]
            tc = "black" if abs(v) > 0.5 else "white"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color=tc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Corr(SHAP_i, feature_j)\n(how much feat_j modulates feat_i's contribution)",
                   color="white", fontsize=7)
    cbar.ax.tick_params(colors="white", labelsize=6)

    ax.set_title(
        "SHAP Interaction Proxy Matrix  |  R007  |  Liq.Sweep Reversal\n"
        "Cell [i,j] = how strongly feature j modulates feature i's SHAP value\n"
        "High |r| = interaction; diagonal = self-correlation (1.0)",
        fontsize=9, fontweight="bold", color="white",
    )

    plt.tight_layout()
    return _save(fig, "r007_shap_interaction_matrix.png")


# ── Chart 6: Top Interaction Heatmaps (pairwise quadrant analysis) ───────────

def plot_interaction_discovery(df: pd.DataFrame,
                                interactions: pd.DataFrame,
                                n_show: int = 6) -> str:
    top  = interactions.head(n_show)
    nrow = math.ceil(n_show / 2)
    ncol = 2

    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 5 * nrow))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Feature Interaction Discovery  |  R007  |  Liq.Sweep Reversal\n"
        "Each cell = quadrant defined by median splits on two features\n"
        "Colour = Profit Factor in that quadrant  |  Text = PF (n trades)",
        fontsize=9, fontweight="bold", color="white",
    )

    flat = np.array(axes).flatten()

    for plot_i, (_, row) in enumerate(top.iterrows()):
        if plot_i >= len(flat):
            break
        ax   = flat[plot_i]
        ax.set_facecolor(BG)
        f1, f2 = row["f1"], row["f2"]
        lbl1   = row["f1_label"]
        lbl2   = row["f2_label"]

        sub = df[[f1, f2, "win", "pnl"]].dropna()
        m1  = float(sub[f1].median())
        m2  = float(sub[f2].median())

        q_labels_x = [f"Low {lbl1}\n(< median)", f"High {lbl1}\n(≥ median)"]
        q_labels_y = [f"Low {lbl2}\n(< med)", f"High {lbl2}\n(≥ med)"]

        grid  = np.full((2, 2), np.nan)
        cnt   = np.zeros((2, 2), dtype=int)
        wr_g  = np.full((2, 2), np.nan)

        for xi, x_high in enumerate([False, True]):
            for yi, y_high in enumerate([False, True]):
                mx = sub[f1] >= m1 if x_high else sub[f1] < m1
                my = sub[f2] >= m2 if y_high else sub[f2] < m2
                s  = sub[mx & my]
                n_s = len(s)
                cnt[yi, xi] = n_s
                if n_s >= 2:
                    w  = int(s["win"].sum())
                    gw = s[s["win"] == 1]["pnl"].sum() if w > 0 else 0.0
                    gl = abs(s[s["win"] == 0]["pnl"].sum()) if (n_s - w) > 0 else 1e-9
                    pf = gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0)
                    grid[yi, xi]  = min(pf, 4.0)
                    wr_g[yi, xi]  = w / n_s

        cmap = plt.cm.RdYlGn
        cmap.set_bad("#1a1a2e")
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, cmap=cmap, vmin=0.3, vmax=2.5,
                       aspect="equal", origin="upper")

        for yi in range(2):
            for xi in range(2):
                n_c = cnt[yi, xi]
                pf_v = grid[yi, xi]
                wr_v = wr_g[yi, xi]
                if n_c < 2:
                    ax.text(xi, yi, "–", ha="center", va="center",
                            color="#555", fontsize=10)
                else:
                    pf_str = f"{pf_v:.2f}" if np.isfinite(pf_v) else "∞"
                    wr_str = f"{wr_v:.0%}" if np.isfinite(wr_v) else ""
                    tc = "black" if (np.isfinite(pf_v) and pf_v > 1.8) else "white"
                    ax.text(xi, yi,
                            f"PF {pf_str}\nWR {wr_str}\nn={n_c}",
                            ha="center", va="center",
                            color=tc, fontsize=9, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(q_labels_x, color="white", fontsize=7.5)
        ax.set_yticklabels(q_labels_y, color="white", fontsize=7.5)
        ax.set_title(
            f"{lbl1} × {lbl2}\nΔPF range = {row['pf_range']:.2f}",
            fontsize=8.5, color="white",
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Profit Factor", color="white", fontsize=6.5)
        cbar.ax.tick_params(colors="white", labelsize=6)

    for j in range(len(top), len(flat)):
        flat[j].set_visible(False)

    plt.tight_layout()
    return _save(fig, "r007_interaction_discovery.png")


# ── Chart 7: Top-3 SHAP Feature Deep-dive (scatter + marginal PF) ────────────

def plot_top3_deep_dive(shap_res: dict, top_feats: list,
                        df: pd.DataFrame) -> str:
    sv   = shap_res["sv"]
    X_df = shap_res["X_df"]

    feats = [f for f in top_feats[:3] if f in X_df.columns]
    if not feats:
        return ""

    n_feat = len(feats)
    fig, axes = plt.subplots(n_feat, 3, figsize=(16, 4.5 * n_feat))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Deep Dive — Top 3 Features  |  R007  |  Liq.Sweep Reversal\n"
        "Col 1: Feature dist (wins vs losses)  |  "
        "Col 2: SHAP vs feature value  |  "
        "Col 3: Win rate across deciles",
        fontsize=9, fontweight="bold", color="white",
    )

    if n_feat == 1:
        axes = [axes]

    for ri, feat in enumerate(feats):
        fi    = list(X_df.columns).index(feat)
        sv_f  = sv[:, fi]
        fv    = X_df[feat].values.astype(float)
        lbl   = FEATURE_LABELS.get(feat, feat)
        wins  = df["win"].values.astype(int) if len(df) == len(X_df) else \
                np.zeros(len(X_df), dtype=int)

        # Col 1: Distribution
        ax1 = axes[ri][0]
        _ax_style(ax1)
        idx_w = wins == 1
        idx_l = wins == 0
        bns   = min(20, max(5, len(fv) // 4))
        rng_  = (float(fv.min()), float(fv.max()))
        if rng_[0] < rng_[1]:
            ax1.hist(fv[idx_w], bins=bns, range=rng_, color="#00C49A",
                     alpha=0.65, label=f"Wins (n={idx_w.sum()})", density=True)
            ax1.hist(fv[idx_l], bins=bns, range=rng_, color="#FF4560",
                     alpha=0.65, label=f"Losses (n={idx_l.sum()})", density=True)
        ax1.set_title(f"{lbl}\nFeature Distribution", fontsize=8.5)
        ax1.set_xlabel(lbl, color="white")
        ax1.set_ylabel("Density", color="white")
        ax1.legend(fontsize=7, facecolor="#1A1D24", edgecolor="#444", labelcolor="white")

        # Col 2: SHAP vs feature
        ax2 = axes[ri][1]
        _ax_style(ax2)
        c_pts = ["#00C49A" if w else "#FF4560" for w in wins]
        ax2.scatter(fv, sv_f, c=c_pts, s=20, alpha=0.7, linewidths=0)
        try:
            z  = np.polyfit(fv, sv_f, 1)
            xr = np.linspace(fv.min(), fv.max(), 200)
            ax2.plot(xr, np.polyval(z, xr), color="#FFD700", lw=1.5)
        except Exception:
            pass
        ax2.axhline(0, color="#888", lw=0.8, ls="--")
        ax2.set_title(f"{lbl}\nSHAP vs Feature Value", fontsize=8.5)
        ax2.set_xlabel(lbl, color="white")
        ax2.set_ylabel("SHAP (→ Win)", color="white")

        # Col 3: Win rate by decile
        ax3 = axes[ri][2]
        _ax_style(ax3)
        try:
            tmp = pd.DataFrame({"fv": fv, "win": wins})
            tmp["decile"] = pd.qcut(tmp["fv"], q=5, labels=False, duplicates="drop")
            grp = tmp.groupby("decile").agg(
                wr=("win", "mean"), n=("win", "count"),
                med=("fv", "median"),
            ).reset_index()
            xd = grp["decile"].values
            wr = grp["wr"].values
            bar_c = ["#00C49A" if v >= 0.333 else "#FF4560" for v in wr]
            ax3.bar(xd, wr, color=bar_c, width=0.65, alpha=0.85)
            ax3.axhline(0.333, color="#FFD700", lw=1.0, ls="--", label="BE win rate")
            ax3.set_xticks(xd)
            ax3.set_xticklabels([f"Q{int(q)+1}" for q in xd],
                                 color="white", fontsize=8)
            ax3.set_ylabel("Win Rate", color="white")
            ax3.set_title(f"{lbl}\nWin Rate by Quintile", fontsize=8.5)
            ax3.legend(fontsize=7, facecolor="#1A1D24",
                       edgecolor="#444", labelcolor="white")
            for i, (v, n_v) in enumerate(zip(wr, grp["n"].values)):
                ax3.text(xd[i], v + 0.01, f"{v:.0%}\nn={n_v}",
                         ha="center", va="bottom", color="white", fontsize=7)
        except Exception:
            ax3.set_visible(False)

    plt.tight_layout()
    return _save(fig, "r007_top3_deep_dive.png")


# =============================================================================
# SECTION 9 — REPORT
# =============================================================================

def print_r007_report(eval_res: dict,
                      shap_res: dict,
                      interactions: pd.DataFrame,
                      df: pd.DataFrame,
                      symbol_meta: dict) -> None:
    S  = "=" * 108
    S2 = "─" * 108
    BL = "  "

    te = eval_res["test"]
    tr = eval_res["train"]
    fn = eval_res["feature_names"]
    gi = eval_res["gini_imp"]
    pm = eval_res["perm_mean"]
    sv = shap_res["sv"]

    # Rank features
    gini_order = np.argsort(gi)[::-1]
    perm_order = np.argsort(pm)[::-1]
    shap_order = np.argsort(np.abs(sv).mean(axis=0))[::-1]

    print(f"\n{S}")
    print(f"{BL}QUANTLAB AI — RESEARCH #007")
    print(f"{BL}Explainable Machine Learning — Liquidity Sweep Reversal")
    print(f"{BL}{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(S)

    # ── Warning on sample size ───────────────────────────────────────────────
    n_total = len(df)
    n_train = tr["n"]
    n_test  = te["n"]
    print(f"\n{BL}⚠  SAMPLE SIZE WARNING")
    print(f"{BL}   Total trades: {n_total}  |  Train: {n_train}  |  Test: {n_test}")
    print(f"{BL}   N is small for ML.  All findings are directional hypotheses,")
    print(f"{BL}   not statistically definitive conclusions.  They guide R008 design.")

    # ── Strategy performance ─────────────────────────────────────────────────
    print(f"\n{BL}STRATEGY PERFORMANCE  (OOS locked engine)")
    print(f"{BL}{'Symbol':<24} {'Trades':>7} {'WR':>7} {'PF':>8} {'Exp(R)':>9}  Verdict")
    print(f"{BL}{S2[2:]}")
    for sym, res in symbol_meta.items():
        m = res["metrics"]
        print(f"{BL}{sym:<24} {m['n_trades']:>7} {m['win_rate']:>7.1%} "
              f"{m['profit_factor']:>8.3f} {m['expectancy_r']:>+9.3f}R  {res['verdict']}")

    # ── Model summary ────────────────────────────────────────────────────────
    print(f"\n{BL}RANDOM FOREST MODEL SUMMARY")
    print(f"{BL}{S2[2:]}")
    print(f"{BL}  Estimators  : {N_TREES}   Max depth : {MAX_DEPTH}   "
          f"Min leaf : {MIN_SAMPLES_LEAF}   Class weight : balanced")
    print(f"{BL}  Split       : 70/30 chronological (no shuffle, no leakage)")
    print(f"{BL}  Features    : {len(fn)} pre-entry only")

    print(f"\n{BL}{'Metric':<22} {'Train':>12} {'Test':>12}  Note")
    print(f"{BL}{'─'*70}")
    for metric, t_val, e_val, note in [
        ("Accuracy",   tr["accuracy"],  te["accuracy"],  "fraction correct"),
        ("Precision",  tr["precision"], te["precision"],
         "of predicted wins, how many won"),
        ("Recall",     tr["recall"],    te["recall"],    "of actual wins, how many found"),
        ("F1 Score",   tr["f1"],        te["f1"],        "harmonic mean prec/recall"),
        ("ROC AUC",    tr["roc_auc"],   te["roc_auc"],
         ">0.5 = better than random"),
        ("Brier Score", tr["brier"],    te["brier"],     "lower = better calibration"),
    ]:
        gap  = abs(t_val - e_val) if (math.isfinite(t_val) and math.isfinite(e_val)) else 0
        flag = "  ← overfit?" if (metric == "ROC AUC" and gap > 0.15) else ""
        print(f"{BL}  {metric:<20} {t_val:>12.3f} {e_val:>12.3f}  {note}{flag}")

    print(f"\n{BL}  Confusion Matrix (test set):")
    cm = te["cm"]
    print(f"{BL}              Pred Loss  Pred Win")
    print(f"{BL}  True Loss :  {int(cm[0,0]):>8}   {int(cm[0,1]):>8}")
    print(f"{BL}  True Win  :  {int(cm[1,0]):>8}   {int(cm[1,1]):>8}")

    # ── Feature Importance Table ─────────────────────────────────────────────
    print(f"\n{BL}FEATURE IMPORTANCE — THREE METHODS")
    print(f"{BL}{S2[2:]}")
    print(f"{BL}{'Rank':<5} {'Feature':<28} {'Gini':>8} {'Perm(AUC)':>12} "
          f"{'SHAP|mean|':>12}  Consensus")
    print(f"{BL}{'─'*80}")

    # Build consensus rank
    gini_rank = {fn[i]: r+1 for r, i in enumerate(gini_order)}
    perm_rank = {fn[i]: r+1 for r, i in enumerate(perm_order)}
    shap_rank = {fn[i]: r+1 for r, i in enumerate(shap_order)}
    cons_score = {f: (gini_rank.get(f,99) + perm_rank.get(f,99) +
                      shap_rank.get(f,99)) for f in fn}
    cons_order = sorted(fn, key=lambda f: cons_score[f])

    shap_mean = np.abs(sv).mean(axis=0)
    for rank, feat in enumerate(cons_order, 1):
        fi   = fn.index(feat)
        lbl  = FEATURE_LABELS.get(feat, feat)
        g    = gi[fi]
        p    = pm[fi]
        sm_v = shap_mean[fi]
        cons = cons_score[feat]
        flag = (" ★★★" if rank <= 3 else " ★★" if rank <= 5 else " ★" if rank <= 8 else "")
        print(f"{BL}  {rank:<4} {lbl:<28} {g:>8.4f} {p:>12.4f} {sm_v:>12.4f}  "
              f"score={cons:>3}{flag}")

    # ── SHAP top-3 ──────────────────────────────────────────────────────────
    print(f"\n{BL}TOP-3 BY CONSENSUS RANK:")
    for i, feat in enumerate(cons_order[:3], 1):
        fi    = fn.index(feat)
        lbl   = FEATURE_LABELS.get(feat, feat)
        sv_f  = sv[:, fi]
        fv    = shap_res["X_df"][feat].values
        wins  = df["win"].values.astype(int) if len(df) == len(shap_res["X_df"]) else \
                np.zeros(len(shap_res["X_df"]), dtype=int)
        mean_w = float(fv[wins == 1].mean()) if (wins == 1).any() else float("nan")
        mean_l = float(fv[wins == 0].mean()) if (wins == 0).any() else float("nan")
        corr   = float(np.corrcoef(fv.astype(float), sv_f)[0, 1]) \
                 if fv.std() > 0 and sv_f.std() > 0 else float("nan")
        print(f"{BL}  {i}. {lbl}")
        print(f"{BL}     Mean@Win={mean_w:.3f}  Mean@Loss={mean_l:.3f}  "
              f"SHAP corr={corr:+.3f}  "
              f"('positive' SHAP = pushes model toward Win prediction)")

    # ── Top interactions ─────────────────────────────────────────────────────
    print(f"\n{BL}TOP FEATURE INTERACTIONS  (quadrant analysis — no trades removed)")
    print(f"{BL}{S2[2:]}")
    print(f"{BL}{'Rank':<5} {'Feature 1':<28} {'Feature 2':<28} "
          f"{'ΔPF range':>10} {'Best Q':>8} {'BestPF':>8} {'WorstPF':>8}")
    print(f"{BL}{'─'*95}")
    for rank, (_, row) in enumerate(interactions.head(10).iterrows(), 1):
        flag = " ★" if row["pf_range"] > 1.0 else ""
        print(f"{BL}  {rank:<4} {row['f1_label']:<28} {row['f2_label']:<28} "
              f"{row['pf_range']:>10.2f} {row['best_q']:>8} "
              f"{row['best_pf']:>8.2f} {row['worst_pf']:>8.2f}{flag}")
    print(f"{BL}  Note: quadrants defined by median splits; n ≥ 2 required per cell")

    # ── Research Questions ────────────────────────────────────────────────────
    print(f"\n{S}")
    print(f"{BL}RESEARCH QUESTIONS — OBJECTIVE ANSWERS")
    print(f"{BL}{S2[2:]}")

    top3_feats = [FEATURE_LABELS.get(f, f) for f in cons_order[:3]]

    # Q1: Top 3
    print(f"\n{BL}Q1  Which THREE features contribute most?")
    for i, feat in enumerate(cons_order[:3], 1):
        fi   = fn.index(feat)
        lbl  = FEATURE_LABELS.get(feat, feat)
        shap_direction = "→ Win" if sv[:, fi].mean() > 0 else "→ Loss"
        print(f"{BL}    {i}. {lbl}")
        print(f"{BL}       Gini rank={gini_rank[feat]}  Perm rank={perm_rank[feat]}  "
              f"SHAP rank={shap_rank[feat]}  "
              f"Mean SHAP direction: {shap_direction}")

    # Q2: Interactions
    print(f"\n{BL}Q2  Which feature INTERACTIONS matter?")
    if len(interactions) >= 3:
        for _, row in interactions.head(3).iterrows():
            # Describe the best quadrant
            qmap = {"HH": "High+High", "HL": "High+Low",
                    "LH": "Low+High",  "LL": "Low+Low"}
            print(f"{BL}    › {row['f1_label']} × {row['f2_label']}")
            print(f"{BL}      Best: {qmap.get(row['best_q'], row['best_q'])} "
                  f"→ PF {row['best_pf']:.2f}  "
                  f"Worst: {qmap.get(row['worst_q'], row['worst_q'])} "
                  f"→ PF {row['worst_pf']:.2f}  "
                  f"(ΔPF = {row['pf_range']:.2f})")
    else:
        print(f"{BL}    Insufficient data for reliable interaction analysis.")

    # Q3: Can one variable explain the strategy?
    top1_gini = FEATURE_LABELS.get(fn[gini_order[0]], fn[gini_order[0]])
    top1_gini_imp = gi[gini_order[0]]
    cumulative_top3 = gi[gini_order[:3]].sum()
    print(f"\n{BL}Q3  Can ONE variable explain the strategy?")
    if top1_gini_imp < 0.25:
        print(f"{BL}    → NO.  Top single feature ({top1_gini}) explains only "
              f"{top1_gini_imp:.1%} of Gini importance.")
        print(f"{BL}    Top 3 combined: {cumulative_top3:.1%}.  "
              f"The edge appears to require multiple features.")
    elif top1_gini_imp < 0.40:
        print(f"{BL}    → PARTIALLY.  Top feature ({top1_gini}) contributes "
              f"{top1_gini_imp:.1%} of Gini importance — dominant but not sufficient alone.")
    else:
        print(f"{BL}    → YES.  Top feature ({top1_gini}) contributes "
              f"{top1_gini_imp:.1%} — it largely drives the model.")

    # Q4: Edge from combinations?
    print(f"\n{BL}Q4  Is the edge created only by COMBINATIONS?")
    if len(interactions) > 0:
        top_int = interactions.iloc[0]
        if top_int["pf_range"] > 1.0:
            print(f"{BL}    → YES — evidence of meaningful interactions.")
            print(f"{BL}    Strongest: {top_int['f1_label']} × {top_int['f2_label']}  "
                  f"ΔPF = {top_int['pf_range']:.2f}")
            print(f"{BL}    Best quadrant PF {top_int['best_pf']:.2f} vs "
                  f"worst {top_int['worst_pf']:.2f} — "
                  f"no single feature reproduces this range alone.")
        elif top_int["pf_range"] > 0.4:
            print(f"{BL}    → POSSIBLY.  Modest interaction effect (ΔPF {top_int['pf_range']:.2f}).")
        else:
            print(f"{BL}    → WEAK evidence.  Interactions explain little beyond single-feature effects.")

    # Q5: Model discovers new relationships?
    print(f"\n{BL}Q5  Does the model discover relationships humans missed?")
    # Check if any high-importance feature ranked lower in R006 Cohen's d
    r006_order = ["dist_from_ll_pct", "atr_rank_pct", "dist_from_ema_pct",
                  "ema200_slope_pct", "funding_rate", "hour_utc"]
    surprises = []
    for rank, feat in enumerate(cons_order[:5]):
        r006_pos = r006_order.index(feat) + 1 if feat in r006_order else 99
        if r006_pos > 3 and rank < 3:
            surprises.append((feat, rank + 1, r006_pos))
    if surprises:
        print(f"{BL}    → YES — model elevates features ranked lower in univariate R006:")
        for feat, ml_rank, r006_rank in surprises:
            print(f"{BL}      {FEATURE_LABELS.get(feat, feat)}: "
                  f"ML consensus rank #{ml_rank} vs R006 univariate rank #{r006_rank}")
    # Check if interactions point to non-obvious combinations
    if len(interactions) > 0:
        top3_int = interactions.head(3)
        for _, row in top3_int.iterrows():
            if row["f1"] not in r006_order[:3] or row["f2"] not in r006_order[:3]:
                print(f"{BL}    ML found interaction: {row['f1_label']} × {row['f2_label']} "
                      f"(ΔPF={row['pf_range']:.2f}) — not discovered by single-feature R006 analysis")
                break
    if not surprises:
        print(f"{BL}    → Model is CONSISTENT with R006 univariate findings.")
        print(f"{BL}    No major surprises in top features.  "
              f"Interactions refine rather than contradict prior analysis.")

    # ── Overall summary ──────────────────────────────────────────────────────
    print(f"\n{BL}OVERALL SUMMARY")
    print(f"{BL}{S2[2:]}")
    print(f"{BL}  Test AUC = {te['roc_auc']:.3f}  |  "
          f"Test F1 = {te['f1']:.3f}  |  "
          f"Test Accuracy = {te['accuracy']:.3f}")
    if te["roc_auc"] < 0.55:
        print(f"{BL}  AUC near 0.5 — model has minimal predictive power on test set.")
        print(f"{BL}  This is expected with n={n_test} test trades.")
        print(f"{BL}  SHAP still provides valid directional explanations of training patterns.")
    elif te["roc_auc"] < 0.65:
        print(f"{BL}  Modest AUC — some signal captured but noise dominates at n={n_test}.")
    else:
        print(f"{BL}  Reasonable AUC for n={n_test} — model captures real patterns.")

    print(f"\n{BL}  IMPORTANT: This model is for EXPLAINABILITY only.")
    print(f"{BL}  Do NOT use it for live signals.  Use findings to design R008 hypotheses.")
    print(f"{BL}  Strategy, engine, fees, entries, exits remain UNCHANGED.")
    print(S)


# =============================================================================
# SECTION 10 — MAIN PIPELINE
# =============================================================================

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              QUANTLAB AI — RESEARCH #007                                      ║
║   Explainable Machine Learning — Liquidity Sweep Reversal                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  Purpose  : Understand which features and combinations explain wins vs losses
  Model    : Random Forest (explainable, no deep learning)
  Split    : 70/30 chronological — no shuffling, no leakage
  Target   : win=1 / loss=0
  Features : pre-entry context only (R005 enrichment)

  IMPORTANT: This is NOT a trading AI. No strategy modification. No prediction.
""")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # ── 1. Collect trades ─────────────────────────────────────────────────
    print("=" * 70)
    print("  STEP 1: Running backtest + enrichment (locked engine)")
    print("=" * 70)
    combined_df, symbol_meta = collect_trades(CONFIG["SYMBOLS"])
    n_total = len(combined_df)
    n_win   = int(combined_df["win"].sum())
    print(f"\n  Combined: {n_total} trades  "
          f"({n_win} wins / {n_total - n_win} losses)  "
          f"Base WR {n_win/n_total:.1%}")

    # ── 2. Build feature matrix ───────────────────────────────────────────
    print("\n  STEP 2: Building feature matrix (pre-entry features only)")
    X, y, feature_names = build_feature_matrix(combined_df)
    print(f"  Features: {len(feature_names)}")
    for i, fn in enumerate(feature_names):
        print(f"    {i+1:2d}. {FEATURE_LABELS.get(fn, fn)}")

    # Drop rows to keep df aligned with X
    available = [c for c in FEATURE_COLS if c in combined_df.columns]
    df_aligned = combined_df[available + ["win", "pnl", "entry_time",
                                           "session", "symbol"]].dropna().reset_index(drop=True)

    # ── 3. Chronological split ────────────────────────────────────────────
    print(f"\n  STEP 3: Chronological 70/30 split")
    X_train, X_test, y_train, y_test, split_idx = chronological_split(
        X, y, df_aligned, ratio=0.70
    )
    print(f"  Train: {len(y_train)} trades  ({int(y_train.sum())} wins)")
    print(f"  Test:  {len(y_test)} trades  ({int(y_test.sum())} wins)")

    # ── 4. Train model ────────────────────────────────────────────────────
    print(f"\n  STEP 4: Training Random Forest  "
          f"(n_trees={N_TREES}, max_depth={MAX_DEPTH})")
    rf = train_random_forest(X_train, y_train)
    print("  Training complete.")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    print("\n  STEP 5: Evaluating model")
    eval_res = evaluate_model(rf, X_train, X_test, y_train, y_test, feature_names)
    te = eval_res["test"]
    print(f"  Test — Acc={te['accuracy']:.3f}  Prec={te['precision']:.3f}  "
          f"Rec={te['recall']:.3f}  F1={te['f1']:.3f}  AUC={te['roc_auc']:.3f}")

    # ── 6. SHAP ───────────────────────────────────────────────────────────
    print("\n  STEP 6: Computing SHAP values (all trades for coverage)")
    shap_res = compute_shap(rf, X, feature_names)
    print(f"  SHAP complete.  Expected value = {shap_res['expected_val']:.3f}")

    # ── 7. Interaction discovery ──────────────────────────────────────────
    print("\n  STEP 7: Interaction discovery (pairwise quadrant scan)")
    interactions = discover_interactions(df_aligned, feature_names)
    if len(interactions) > 0:
        top_int = interactions.iloc[0]
        print(f"  Strongest interaction: {top_int['f1_label']} × {top_int['f2_label']}  "
              f"ΔPF = {top_int['pf_range']:.2f}")

    # ── 8. SHAP top features ──────────────────────────────────────────────
    shap_mean = np.abs(shap_res["sv"]).mean(axis=0)
    top_shap_feats = [feature_names[int(i)] for i in np.argsort(shap_mean)[::-1]]

    # ── 9. Charts ─────────────────────────────────────────────────────────
    print("\n  STEP 8: Generating charts")
    charts = []

    p = plot_confusion_roc(eval_res)
    charts.append(p); print(f"  → {p}")

    p = plot_feature_importance(eval_res)
    charts.append(p); print(f"  → {p}")

    p = plot_shap_summary(shap_res)
    charts.append(p); print(f"  → {p}")

    p = plot_shap_dependence(shap_res, top_shap_feats)
    charts.append(p); print(f"  → {p}")

    p = plot_shap_interaction_matrix(shap_res)
    charts.append(p); print(f"  → {p}")

    if len(interactions) > 0:
        p = plot_interaction_discovery(df_aligned, interactions, n_show=6)
        charts.append(p); print(f"  → {p}")

    p = plot_top3_deep_dive(shap_res, top_shap_feats, df_aligned)
    if p:
        charts.append(p); print(f"  → {p}")

    # ── 10. Report ────────────────────────────────────────────────────────
    print_r007_report(eval_res, shap_res, interactions, df_aligned, symbol_meta)

    # ── 11. Journal ───────────────────────────────────────────────────────
    jnl_rows = []
    for sym, res in symbol_meta.items():
        row = _journal_row("Liq.Sweep", sym, res["metrics"],
                           res["mc"], res["verdict"])
        row["research_id"] = RESEARCH_ID
        jnl_rows.append(row)
    if jnl_rows:
        append_journal(jnl_rows)
        print(f"\n  Research journal updated → {CONFIG['JOURNAL_FILE']}")

    print(f"\n  All outputs → {OUTPUT_FOLDER}/")
    print("  Research #007 complete.\n")


if __name__ == "__main__":
    main()
