"""
=============================================================================
QUANTLAB AI — RESEARCH #017
Objective : Market Regime Classification

NOT a trading strategy. No entries. No exits. No optimisation.

Goal: Determine whether crypto futures naturally separate into distinct
      market regimes using unsupervised machine learning.

Markets    : BTC, ETH, LINK, XRP, DOGE, LTC, AVAX, BCH  (OKX perps)
Timeframes : 15m | 1H

Features (per bar):
  ATR, ATR Percentile, ADX, EMA200 Slope, Realised Volatility,
  20-Bar Return, 50-Bar Return, Rolling Std, Rolling Skew, Rolling Kurtosis,
  Bollinger Width, Distance from EMA200, Volume, Volume Percentile,
  Momentum, RSI

Clustering:
  KMeans (optimal k via elbow + silhouette)
  Gaussian Mixture Model
  DBSCAN
  Hierarchical / Agglomerative

Output:
  Per-regime statistics, transition probabilities, PCA/t-SNE projections,
  correlation heatmaps, regime timelines, 7 final questions.
=============================================================================
"""

import os, sys, time, math, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantlab_ai import (
    CONFIG,
    calc_ema, calc_atr, calc_adx,
    append_journal,
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

try:
    import umap as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID   = "R017"
OUTPUT_FOLDER = CONFIG["OUTPUT_FOLDER"]
CACHE_FOLDER  = CONFIG["CACHE_FOLDER"]

SYMBOLS = [
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "LINK-USDT-SWAP", "AVAX-USDT-SWAP",
    "XRP-USDT-SWAP",  "DOGE-USDT-SWAP", "LTC-USDT-SWAP",  "BCH-USDT-SWAP",
]

TIMEFRAMES = [
    {"bar": "15m", "minutes": 15,  "months": 6, "min_cache": 5_000,  "label": "15-minute"},
    {"bar": "1H",  "minutes": 60,  "months": 6, "min_cache":   500,  "label": "1-hour"},
]

OKX_HIST_URL = "https://www.okx.com/api/v5/market/history-candles"
OKX_LIVE_URL = "https://www.okx.com/api/v5/market/candles"
CANDLE_COLS  = ["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"]
PAGE_LIMIT   = 300
API_DELAY    = 0.05
DL_WORKERS   = 4
MAX_RETRIES  = 5

BG       = "#0F1117"
PALETTE  = ["#4A90D9","#FFB347","#00C49A","#FF4560",
            "#E040FB","#FFD700","#00D4FF","#FF6B6B",
            "#A8E063","#F06292","#80DEEA","#BCAAA4"]

K_RANGE  = range(2, 9)   # test 2..8 clusters
TSNE_SAMPLE = 2_000      # max rows fed to t-SNE (speed)
REGIME_NAMES = {
    0: "Regime-0", 1: "Regime-1", 2: "Regime-2", 3: "Regime-3",
    4: "Regime-4", 5: "Regime-5", 6: "Regime-6", 7: "Regime-7",
}


# =============================================================================
# SECTION 1 — DATA DOWNLOAD (identical to R016)
# =============================================================================

def _parse_candles(raw: list) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open","high","low","close","vol"]:
        df[c] = pd.to_numeric(df[c])
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return (df[["datetime","open","high","low","close","vol"]]
            .sort_values("datetime").reset_index(drop=True))


def _fetch_page(symbol: str, bar: str, after_ms=None, use_history=True) -> list:
    url    = OKX_HIST_URL if use_history else OKX_LIVE_URL
    params = {"instId": symbol, "bar": bar, "limit": str(PAGE_LIMIT)}
    if after_ms is not None:
        params["after"] = str(after_ms)
    for attempt in range(MAX_RETRIES):
        try:
            r    = requests.get(url, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(2 ** attempt); continue
            d    = r.json()
            code = d.get("code", "-1")
            if code == "0":
                return d.get("data", [])
            if code in ("50011", "50013"):
                return []
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return []


def _cache_path(symbol: str, bar: str) -> str:
    safe = symbol.replace("-", "_") + f"_{bar}"
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    return os.path.join(CACHE_FOLDER, f"{safe}.parquet")


def _load_cache(symbol: str, bar: str):
    path = _cache_path(symbol, bar)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if df["datetime"].dt.tz is None:
        df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    return df.sort_values("datetime").reset_index(drop=True)


def _save_cache(df: pd.DataFrame, symbol: str, bar: str):
    df.to_parquet(_cache_path(symbol, bar), index=False)


def _download_symbol(symbol: str, bar: str, months: int) -> pd.DataFrame:
    now_ms    = int(time.time() * 1000)
    cutoff_ms = now_ms - int(months * 30.44 * 24 * 3600 * 1000)
    all_rows, after_ms_cursor, pages = [], None, 0
    while True:
        raw = _fetch_page(symbol, bar, after_ms=after_ms_cursor, use_history=True)
        if not raw:
            if pages == 0:
                raw = _fetch_page(symbol, bar, after_ms=None, use_history=False)
                if not raw: break
            else: break
        all_rows.extend(raw)
        pages += 1
        oldest_ts = int(raw[-1][0])
        after_ms_cursor = oldest_ts
        if oldest_ts <= cutoff_ms: break
        time.sleep(API_DELAY)
    if not all_rows:
        raise RuntimeError(f"No data for {symbol} [{bar}]")
    df        = _parse_candles(all_rows)
    cutoff_dt = pd.Timestamp(cutoff_ms, unit="ms", tz="UTC")
    df        = df[df["datetime"] >= cutoff_dt]
    return df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


def _refresh_symbol(symbol: str, bar: str, cached: pd.DataFrame) -> pd.DataFrame:
    last_ts  = cached["datetime"].iloc[-1]
    since_ms = int(last_ts.timestamp() * 1000)
    all_rows, cursor = [], None
    for _ in range(20):
        raw = _fetch_page(symbol, bar, after_ms=cursor, use_history=False)
        if not raw:
            raw = _fetch_page(symbol, bar, after_ms=cursor, use_history=True)
        if not raw: break
        all_rows.extend(raw)
        oldest_ts = int(raw[-1][0])
        if oldest_ts <= since_ms: break
        cursor = oldest_ts
        time.sleep(API_DELAY)
    if not all_rows:
        return cached
    new_df = _parse_candles(all_rows)
    new_df = new_df[new_df["datetime"] > last_ts]
    if len(new_df) == 0:
        return cached
    combined = (pd.concat([cached, new_df], ignore_index=True)
                .drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True))
    _save_cache(combined, symbol, bar)
    return combined


def get_data(symbol: str, bar: str, months: int, min_cache: int) -> pd.DataFrame:
    cached = _load_cache(symbol, bar)
    if cached is not None and len(cached) >= min_cache:
        last_ts = cached["datetime"].iloc[-1]
        gap_min = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 60
        bar_min = int(bar.rstrip("mH")) * (60 if bar.endswith("H") else 1)
        if gap_min < bar_min * 2:
            return cached
        print(f"  {symbol.split('-')[0]:6s}[{bar}] refreshing ({len(cached):,} cached)...", flush=True)
        return _refresh_symbol(symbol, bar, cached)
    print(f"  {symbol.split('-')[0]:6s}[{bar}] full download...", flush=True)
    df = _download_symbol(symbol, bar, months)
    _save_cache(df, symbol, bar)
    return df


def download_all_parallel(bar: str, months: int, min_cache: int) -> dict:
    results = {}
    def _worker(sym):
        try:    return sym, get_data(sym, bar, months, min_cache), None
        except Exception as e: return sym, None, str(e)
    with ThreadPoolExecutor(max_workers=DL_WORKERS) as pool:
        futures = {pool.submit(_worker, s): s for s in SYMBOLS}
        for fut in as_completed(futures):
            sym, df, err = fut.result()
            if err: print(f"  [WARN] {sym}[{bar}] failed: {err}", flush=True)
            else:   results[sym] = df
    return results


# =============================================================================
# SECTION 2 — FEATURE ENGINEERING
# =============================================================================

def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def rolling_pct_rank(series: pd.Series, window: int = 50) -> pd.Series:
    """Percentile rank of current value within rolling window (0–1). Vectorised via pandas."""
    return series.rolling(window, min_periods=window // 2).rank(pct=True)


def compute_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Compute all 16 regime features for a single symbol's OHLCV DataFrame.
    Returns a DataFrame with the feature columns + datetime + symbol.
    No look-ahead: all features use only past/current bar data.
    """
    df = df.copy().reset_index(drop=True)
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["vol"].replace(0, np.nan)

    # 1. ATR (14)
    atr = calc_atr(df, 14)

    # 2. ATR Percentile — 50-bar rolling rank
    atr_pct = rolling_pct_rank(atr, 50)

    # 3. ADX (14)
    adx = calc_adx(df, 14)

    # 4. EMA200 Slope — % change of EMA200 over 10 bars
    ema200       = calc_ema(close, 200)
    ema200_lag10 = ema200.shift(10)
    ema200_slope = (ema200 - ema200_lag10) / ema200_lag10.replace(0, np.nan) * 100.0

    # 5. Realised Volatility — std of log returns over 20 bars, annualised
    log_ret   = np.log(close / close.shift(1))
    rv20      = log_ret.rolling(20).std() * np.sqrt(365 * 24)   # continuous annualisation

    # 6. 20-Bar Return (%)
    ret20 = (close / close.shift(20) - 1.0) * 100.0

    # 7. 50-Bar Return (%)
    ret50 = (close / close.shift(50) - 1.0) * 100.0

    # 8. Rolling Std (20 bars, of log returns)
    roll_std20 = log_ret.rolling(20).std() * 100.0

    # 9. Rolling Skew (50 bars)
    roll_skew50 = log_ret.rolling(50).skew()

    # 10. Rolling Kurtosis (50 bars)
    roll_kurt50 = log_ret.rolling(50).kurt()

    # 11. Bollinger Width — (upper - lower) / mid
    sma20       = close.rolling(20).mean()
    std20       = close.rolling(20).std(ddof=1)
    bb_width    = (4 * std20) / sma20.replace(0, np.nan) * 100.0   # as %

    # 12. Distance from EMA200 (%)
    dist_ema200 = (close - ema200) / ema200.replace(0, np.nan) * 100.0

    # 13. Volume (log-transformed for scale)
    log_vol = np.log1p(vol)

    # 14. Volume Percentile — 50-bar rolling rank
    vol_pct = rolling_pct_rank(log_vol, 50)

    # 15. Momentum — 10-bar rate of change (%)
    momentum = (close / close.shift(10) - 1.0) * 100.0

    # 16. RSI (14)
    rsi = calc_rsi(close, 14)

    feat = pd.DataFrame({
        "datetime":     df["datetime"],
        "symbol":       symbol,
        "open":         df["open"],
        "high":         high,
        "low":          low,
        "close":        close,
        "vol":          df["vol"],
        # features
        "atr":          atr,
        "atr_pct":      atr_pct,
        "adx":          adx,
        "ema200_slope": ema200_slope,
        "rv20":         rv20,
        "ret20":        ret20,
        "ret50":        ret50,
        "roll_std20":   roll_std20,
        "roll_skew50":  roll_skew50,
        "roll_kurt50":  roll_kurt50,
        "bb_width":     bb_width,
        "dist_ema200":  dist_ema200,
        "log_vol":      log_vol,
        "vol_pct":      vol_pct,
        "momentum":     momentum,
        "rsi":          rsi,
    })
    return feat


FEATURE_COLS = [
    "atr_pct", "adx", "ema200_slope", "rv20",
    "ret20", "ret50", "roll_std20", "roll_skew50", "roll_kurt50",
    "bb_width", "dist_ema200", "vol_pct", "momentum", "rsi",
]
# Note: raw atr and log_vol excluded from clustering (scale issues),
# but pct-rank versions (atr_pct, vol_pct) are included.


# =============================================================================
# SECTION 3 — FIND OPTIMAL K (Elbow + Silhouette)
# =============================================================================

def find_optimal_k(X_scaled: np.ndarray, k_range=K_RANGE, random_state=42) -> dict:
    """
    Run KMeans for each k in k_range.
    Return inertias, silhouette scores, and best k.
    Uses a subsample for speed on large datasets.
    """
    inertias    = {}
    silhouettes = {}
    bics_gmm    = {}
    aics_gmm    = {}

    # Subsample for speed (k-selection doesn't need all data)
    rng      = np.random.RandomState(random_state)
    n_sample = min(20_000, len(X_scaled))
    idx_s    = rng.choice(len(X_scaled), n_sample, replace=False)
    X_s      = X_scaled[idx_s]

    print(f"  Finding optimal k on {n_sample:,} sample bars "
          f"(KMeans elbow + silhouette, GMM BIC/AIC)...", flush=True)
    for k in k_range:
        km  = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        lbl = km.fit_predict(X_s)
        inertias[k]    = km.inertia_
        silhouettes[k] = silhouette_score(X_s, lbl, sample_size=min(3000, n_sample))

        gm  = GaussianMixture(n_components=k, n_init=3, random_state=random_state,
                              covariance_type="diag")
        gm.fit(X_s)
        bics_gmm[k] = gm.bic(X_s)
        aics_gmm[k] = gm.aic(X_s)

        print(f"    k={k}  inertia={inertias[k]:,.0f}  sil={silhouettes[k]:.4f}"
              f"  BIC={bics_gmm[k]:,.0f}  AIC={aics_gmm[k]:,.0f}", flush=True)

    # Best k by silhouette
    best_k_sil = max(silhouettes, key=silhouettes.get)
    # Best k by BIC (lower = better)
    best_k_bic = min(bics_gmm, key=bics_gmm.get)
    # Elbow: largest second-derivative drop in inertia
    ks  = list(k_range)
    inn = [inertias[k] for k in ks]
    if len(inn) >= 3:
        d2  = [inn[i-1] - 2*inn[i] + inn[i+1] for i in range(1, len(inn)-1)]
        best_k_elbow = ks[1 + d2.index(max(d2))]
    else:
        best_k_elbow = ks[0]

    # Final recommendation — majority vote or silhouette wins tie
    candidates = [best_k_sil, best_k_bic, best_k_elbow]
    from collections import Counter
    voted = Counter(candidates).most_common(1)[0][0]

    print(f"\n  Best k → silhouette={best_k_sil}  BIC={best_k_bic}  elbow={best_k_elbow}"
          f"  → SELECTED={voted}")

    return {
        "inertias": inertias, "silhouettes": silhouettes,
        "bics": bics_gmm, "aics": aics_gmm,
        "best_k_sil": best_k_sil, "best_k_bic": best_k_bic,
        "best_k_elbow": best_k_elbow, "best_k": voted,
    }


# =============================================================================
# SECTION 4 — CLUSTERING
# =============================================================================

def run_all_clustering(X_scaled: np.ndarray, best_k: int, random_state: int = 42) -> dict:
    """Run all four clustering methods and return label arrays."""
    print(f"\n  Running clustering with k={best_k}...", flush=True)

    # KMeans
    km  = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    km_labels = km.fit_predict(X_scaled)
    print(f"  KMeans done.  Inertia={km.inertia_:,.0f}", flush=True)

    # Gaussian Mixture Model
    gm  = GaussianMixture(n_components=best_k, n_init=3, random_state=random_state,
                          covariance_type="diag")
    gm.fit(X_scaled)
    gm_labels = gm.predict(X_scaled)
    print(f"  GMM done.     BIC={gm.bic(X_scaled):,.0f}", flush=True)

    # DBSCAN — auto-tune eps via k-distance heuristic on a sample
    from sklearn.neighbors import NearestNeighbors
    rng_db   = np.random.RandomState(random_state)
    n_db_smp = min(15_000, len(X_scaled))
    idx_db   = rng_db.choice(len(X_scaled), n_db_smp, replace=False)
    X_db     = X_scaled[idx_db]
    nn  = NearestNeighbors(n_neighbors=5)
    nn.fit(X_db)
    dists, _ = nn.kneighbors(X_db)
    k_dist   = np.sort(dists[:, -1])
    eps_auto = float(np.percentile(k_dist, 90))
    db  = DBSCAN(eps=eps_auto, min_samples=8, n_jobs=-1)
    db_labels_smp = db.fit_predict(X_db)
    # Assign full dataset via nearest-centroid proxy
    db_labels = np.full(len(X_scaled), -1, dtype=int)
    db_labels[idx_db] = db_labels_smp
    n_db_clusters = len(set(db_labels_smp)) - (1 if -1 in db_labels_smp else 0)
    n_noise       = int((db_labels_smp == -1).sum())
    print(f"  DBSCAN done.  eps={eps_auto:.3f}  clusters={n_db_clusters}  "
          f"noise={n_noise}/{n_db_smp}", flush=True)

    # Hierarchical (Agglomerative) — subsample to avoid O(n²) cost
    n_hc  = min(20_000, len(X_scaled))
    idx_hc = np.random.RandomState(random_state).choice(len(X_scaled), n_hc, replace=False)
    X_hc  = X_scaled[idx_hc]
    hc    = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hc_labels_smp = hc.fit_predict(X_hc)
    # Propagate to full dataset via closest KMeans centroid (already fit)
    hc_labels = km_labels.copy()   # fallback: use kmeans labels for unsampled rows
    hc_labels[idx_hc] = hc_labels_smp
    print(f"  Hierarchical done (on {n_hc:,} sample).", flush=True)

    return {
        "kmeans":        km_labels,
        "gmm":           gm_labels,
        "dbscan":        db_labels,
        "hierarchical":  hc_labels,
        "kmeans_model":  km,
        "gmm_model":     gm,
        "dbscan_eps":    eps_auto,
        "n_dbscan_clusters": n_db_clusters,
        "n_dbscan_noise":    n_noise,
    }


# =============================================================================
# SECTION 5 — REGIME STATISTICS
# =============================================================================

def regime_stats(feat_df: pd.DataFrame, labels: np.ndarray, method_name: str) -> dict:
    """
    For each regime label compute:
      avg volatility, trend strength, return, duration, frequency,
      transition probability matrix.
    """
    df = feat_df.copy()
    df["regime"] = labels
    df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    regimes = sorted([r for r in np.unique(labels) if r != -1])
    n_total = len(df[df["regime"] != -1])

    stats = {}
    for r in regimes:
        sub = df[df["regime"] == r]
        freq = len(sub) / max(n_total, 1)

        # Duration: run-lengths of this regime in each symbol
        durations = []
        for sym in df["symbol"].unique():
            sym_df   = df[df["symbol"] == sym]
            reg_seq  = sym_df["regime"].values
            cnt      = 0
            for val in reg_seq:
                if val == r:
                    cnt += 1
                elif cnt > 0:
                    durations.append(cnt)
                    cnt = 0
            if cnt > 0:
                durations.append(cnt)

        stats[r] = {
            "n":            len(sub),
            "freq":         freq,
            "avg_rv20":     sub["rv20"].mean()        if "rv20"     in sub else np.nan,
            "avg_adx":      sub["adx"].mean()         if "adx"      in sub else np.nan,
            "avg_ret20":    sub["ret20"].mean()        if "ret20"    in sub else np.nan,
            "avg_ret50":    sub["ret50"].mean()        if "ret50"    in sub else np.nan,
            "avg_atr_pct":  sub["atr_pct"].mean()     if "atr_pct"  in sub else np.nan,
            "avg_bb_width": sub["bb_width"].mean()    if "bb_width" in sub else np.nan,
            "avg_slope":    sub["ema200_slope"].mean() if "ema200_slope" in sub else np.nan,
            "avg_rsi":      sub["rsi"].mean()          if "rsi"      in sub else np.nan,
            "avg_momentum": sub["momentum"].mean()     if "momentum" in sub else np.nan,
            "avg_dist_ema": sub["dist_ema200"].mean()  if "dist_ema200" in sub else np.nan,
            "avg_vol_pct":  sub["vol_pct"].mean()      if "vol_pct"  in sub else np.nan,
            "avg_duration_bars": float(np.mean(durations)) if durations else 0.0,
            "med_duration_bars": float(np.median(durations)) if durations else 0.0,
        }

    # Transition probability matrix (per symbol, pooled)
    trans_counts = np.zeros((len(regimes), len(regimes)), dtype=float)
    r_idx        = {r: i for i, r in enumerate(regimes)}

    for sym in df["symbol"].unique():
        sym_df  = df[df["symbol"] == sym]
        reg_seq = sym_df["regime"].values
        for i in range(1, len(reg_seq)):
            a, b = reg_seq[i-1], reg_seq[i]
            if a in r_idx and b in r_idx:
                trans_counts[r_idx[a], r_idx[b]] += 1

    # Row-normalise → probabilities
    row_sums     = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_probs  = trans_counts / row_sums

    # Persistence = diagonal of transition matrix
    for i, r in enumerate(regimes):
        stats[r]["persistence"] = float(trans_probs[i, i])

    return {
        "regimes":      regimes,
        "stats":        stats,
        "trans_probs":  trans_probs,
        "trans_counts": trans_counts,
        "method":       method_name,
    }


def symbol_regime_distribution(feat_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """How much time (%) each symbol spends in each regime."""
    df = feat_df.copy()
    df["regime"] = labels
    pivot = (df[df["regime"] != -1]
             .groupby(["symbol", "regime"])
             .size()
             .unstack(fill_value=0))
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100.0
    pivot.index = [s.split("-")[0] for s in pivot.index]
    return pivot


# =============================================================================
# SECTION 6 — DIMENSIONALITY REDUCTION
# =============================================================================

def compute_projections(X_scaled: np.ndarray, labels: np.ndarray,
                        random_state: int = 42) -> dict:
    """PCA, t-SNE (and UMAP if available)."""
    projections = {}

    # PCA (full, keep first 10 components)
    print("  PCA...", flush=True)
    pca   = PCA(n_components=min(10, X_scaled.shape[1]), random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    projections["pca"]              = X_pca
    projections["pca_explained"]    = pca.explained_variance_ratio_
    projections["pca_components"]   = pca.components_   # shape (n_comp, n_feat)

    # t-SNE on a sample
    print("  t-SNE (this may take a minute)...", flush=True)
    n      = len(X_scaled)
    idx    = np.random.RandomState(random_state).choice(n, min(TSNE_SAMPLE, n), replace=False)
    X_tsne = TSNE(n_components=2, perplexity=30, max_iter=300,
                  random_state=random_state).fit_transform(X_scaled[idx])
    projections["tsne"]         = X_tsne
    projections["tsne_idx"]     = idx
    projections["tsne_labels"]  = labels[idx]

    # UMAP if available
    if UMAP_AVAILABLE:
        print("  UMAP...", flush=True)
        X_umap = umap_lib.UMAP(n_components=2, random_state=random_state).fit_transform(X_scaled[idx])
        projections["umap"]        = X_umap
        projections["umap_labels"] = labels[idx]

    return projections


# =============================================================================
# SECTION 7 — VISUALISATIONS
# =============================================================================

def _ax_dark(ax):
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.tick_params(colors="#AAA", labelsize=8)
    ax.xaxis.label.set_color("#AAA")
    ax.yaxis.label.set_color("#AAA")
    ax.title.set_color("#EEE")
    return ax


def _save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  → {path}")


def plot_elbow_silhouette(k_results: dict, tf_bar: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    ks  = list(k_results["inertias"].keys())
    inn = [k_results["inertias"][k]    for k in ks]
    sil = [k_results["silhouettes"][k] for k in ks]
    bic = [k_results["bics"][k]        for k in ks]

    for ax in axes:
        _ax_dark(ax)

    axes[0].plot(ks, inn, "o-", color="#4A90D9", lw=2)
    axes[0].axvline(k_results["best_k_elbow"], color="#FFD700", lw=1.2, ls="--",
                    label=f"Elbow k={k_results['best_k_elbow']}")
    axes[0].set_title("KMeans Elbow (Inertia)", fontsize=10)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[0].legend(fontsize=8)

    axes[1].plot(ks, sil, "o-", color="#00C49A", lw=2)
    axes[1].axvline(k_results["best_k_sil"], color="#FFD700", lw=1.2, ls="--",
                    label=f"Best sil k={k_results['best_k_sil']}")
    axes[1].set_title("Silhouette Score", fontsize=10)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
    axes[1].legend(fontsize=8)

    axes[2].plot(ks, bic, "o-", color="#FFB347", lw=2)
    axes[2].axvline(k_results["best_k_bic"], color="#FFD700", lw=1.2, ls="--",
                    label=f"Best BIC k={k_results['best_k_bic']}")
    axes[2].set_title("GMM BIC", fontsize=10)
    axes[2].set_xlabel("k"); axes[2].set_ylabel("BIC")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"R017 Optimal k Selection [{tf_bar}]", color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"{OUTPUT_FOLDER}/r017_optimal_k_{tf_bar}.png")


def plot_pca_scatter(proj: dict, labels: np.ndarray, best_k: int, tf_bar: str):
    X_pca = proj["pca"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    regimes  = sorted(set(labels[labels != -1]))
    colors   = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(regimes)}

    for ax in axes:
        _ax_dark(ax)

    # PC1 vs PC2
    for r in regimes:
        mask = labels == r
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=colors[r], s=3, alpha=0.35, label=f"R{r}")
    axes[0].set_title("PCA — PC1 vs PC2", fontsize=10)
    axes[0].set_xlabel(f"PC1 ({proj['pca_explained'][0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({proj['pca_explained'][1]*100:.1f}% var)")
    axes[0].legend(markerscale=4, fontsize=8)

    # PC2 vs PC3
    if X_pca.shape[1] >= 3:
        for r in regimes:
            mask = labels == r
            axes[1].scatter(X_pca[mask, 1], X_pca[mask, 2],
                            c=colors[r], s=3, alpha=0.35, label=f"R{r}")
        axes[1].set_title("PCA — PC2 vs PC3", fontsize=10)
        axes[1].set_xlabel(f"PC2 ({proj['pca_explained'][1]*100:.1f}% var)")
        axes[1].set_ylabel(f"PC3 ({proj['pca_explained'][2]*100:.1f}% var)")
        axes[1].legend(markerscale=4, fontsize=8)

    fig.suptitle(f"R017 PCA Projection (KMeans k={best_k}) [{tf_bar}]",
                 color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"{OUTPUT_FOLDER}/r017_pca_{tf_bar}.png")


def plot_tsne(proj: dict, best_k: int, tf_bar: str):
    X_tsne  = proj["tsne"]
    lbl     = proj["tsne_labels"]
    regimes = sorted(set(lbl[lbl != -1]))
    colors  = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(regimes)}

    fig, ax = plt.subplots(figsize=(12, 9), facecolor=BG)
    _ax_dark(ax)

    for r in regimes:
        mask = lbl == r
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=colors[r], s=4, alpha=0.45, label=f"Regime {r}")
    if -1 in lbl:
        mask = lbl == -1
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c="#444", s=2, alpha=0.2, label="Noise")

    ax.set_title(f"t-SNE Projection (KMeans k={best_k}) [{tf_bar}]", fontsize=12)
    ax.legend(markerscale=4, fontsize=9)
    _save(fig, f"{OUTPUT_FOLDER}/r017_tsne_{tf_bar}.png")


def plot_correlation_heatmap(feat_df: pd.DataFrame, labels: np.ndarray, tf_bar: str):
    """Feature correlation matrix (pooled), coloured by regime average."""
    feat_only = feat_df[FEATURE_COLS].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG)
    for ax in axes:
        _ax_dark(ax)

    # Overall correlation
    corr = feat_only.corr()
    im   = axes[0].imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(FEATURE_COLS)))
    axes[0].set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontsize=7, color="#EEE")
    axes[0].set_yticks(range(len(FEATURE_COLS)))
    axes[0].set_yticklabels(FEATURE_COLS, fontsize=7, color="#EEE")
    axes[0].set_title("Feature Correlation Matrix (All Bars)", fontsize=10)
    plt.colorbar(im, ax=axes[0], fraction=0.04)

    # Mean feature values per regime (heatmap)
    df_tmp = feat_df[FEATURE_COLS].copy()
    df_tmp["regime"] = labels
    regime_means = df_tmp[df_tmp["regime"] != -1].groupby("regime")[FEATURE_COLS].mean()
    # Z-score each column so regimes are comparable
    rm_z = (regime_means - regime_means.mean()) / (regime_means.std() + 1e-9)
    im2  = axes[1].imshow(rm_z.values, cmap="RdYlGn", vmin=-2.5, vmax=2.5, aspect="auto")
    axes[1].set_xticks(range(len(FEATURE_COLS)))
    axes[1].set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontsize=7, color="#EEE")
    axes[1].set_yticks(range(len(regime_means)))
    axes[1].set_yticklabels([f"Regime {r}" for r in regime_means.index], fontsize=8, color="#EEE")
    axes[1].set_title("Regime Fingerprint (Z-scored Feature Means)", fontsize=10)
    plt.colorbar(im2, ax=axes[1], fraction=0.04)

    # Annotate values
    for i in range(len(regime_means)):
        for j in range(len(FEATURE_COLS)):
            v = rm_z.values[i, j]
            axes[1].text(j, i, f"{v:.1f}", ha="center", va="center",
                         fontsize=5, color="white" if abs(v) > 1.5 else "#333")

    fig.suptitle(f"R017 Correlation & Regime Fingerprint [{tf_bar}]",
                 color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"{OUTPUT_FOLDER}/r017_heatmap_{tf_bar}.png")


def plot_transition_matrix(reg_result: dict, tf_bar: str, method: str = "KMeans"):
    trans   = reg_result["trans_probs"]
    regimes = reg_result["regimes"]
    n       = len(regimes)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    for ax in axes: _ax_dark(ax)

    # Heatmap
    im = axes[0].imshow(trans, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels([f"R{r}" for r in regimes], fontsize=9, color="#EEE")
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([f"R{r}" for r in regimes], fontsize=9, color="#EEE")
    axes[0].set_xlabel("To Regime"); axes[0].set_ylabel("From Regime")
    axes[0].set_title(f"Transition Probability Matrix ({method})", fontsize=10)
    plt.colorbar(im, ax=axes[0], fraction=0.04)
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f"{trans[i,j]:.2f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if trans[i,j] > 0.5 else "#333")

    # Bar chart: persistence (diagonal) vs best_k
    persist = [trans[i, i] for i in range(n)]
    bars = axes[1].bar([f"R{r}" for r in regimes], persist,
                       color=[PALETTE[i % len(PALETTE)] for i in range(n)], alpha=0.85)
    axes[1].axhline(0.5, color="#FFF", lw=0.7, ls="--", label="50% persistence")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Regime Persistence (P(stay))", fontsize=10)
    axes[1].set_ylabel("P(regime stays same next bar)")
    axes[1].legend(fontsize=8)
    for bar, p in zip(bars, persist):
        axes[1].text(bar.get_x() + bar.get_width()/2, p + 0.01,
                     f"{p:.2f}", ha="center", va="bottom", fontsize=9, color="#EEE")

    fig.suptitle(f"R017 Regime Transitions ({method}) [{tf_bar}]",
                 color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"{OUTPUT_FOLDER}/r017_transitions_{tf_bar}.png")


def plot_regime_timeline(feat_df: pd.DataFrame, labels: np.ndarray,
                         best_k: int, tf_bar: str, n_syms_show: int = 4):
    """Show a strip-plot of regime labels over time for up to n_syms_show symbols."""
    df = feat_df.copy()
    df["regime"] = labels
    syms  = list(df["symbol"].unique())[:n_syms_show]
    n     = len(syms)
    regimes_all = sorted(set(labels[labels != -1]))
    cmap  = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(regimes_all)}

    fig, axes = plt.subplots(n, 1, figsize=(18, 3.5 * n), facecolor=BG, sharex=False)
    if n == 1:
        axes = [axes]

    for ax, sym in zip(axes, syms):
        _ax_dark(ax)
        sym_df = df[df["symbol"] == sym].sort_values("datetime").reset_index(drop=True)
        x      = np.arange(len(sym_df))
        rlbl   = sym_df["regime"].values
        # Draw colour blocks
        start = 0
        for i in range(1, len(rlbl)):
            if rlbl[i] != rlbl[start]:
                color = cmap.get(rlbl[start], "#444")
                ax.axvspan(start, i, alpha=0.6, color=color, lw=0)
                start = i
        ax.axvspan(start, len(rlbl), alpha=0.6,
                   color=cmap.get(rlbl[start], "#444"), lw=0)

        # Overlay close price (normalised)
        close_n = (sym_df["close"] - sym_df["close"].min()) / (
                    (sym_df["close"].max() - sym_df["close"].min()) + 1e-9)
        ax.plot(x, close_n, color="#FFF", lw=0.6, alpha=0.85)
        ax.set_ylabel(sym.split("-")[0], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, len(sym_df))

    # Legend
    from matplotlib.patches import Patch
    patches = [Patch(color=cmap[r], label=f"Regime {r}") for r in regimes_all]
    axes[0].legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.4)

    fig.suptitle(f"R017 Regime Timeline (KMeans k={best_k}) [{tf_bar}]",
                 color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"{OUTPUT_FOLDER}/r017_timeline_{tf_bar}.png")


def plot_symbol_regime_dist(sym_dist: pd.DataFrame, best_k: int, tf_bar: str):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    _ax_dark(ax)

    regimes = sym_dist.columns.tolist()
    x       = np.arange(len(sym_dist))
    bottoms = np.zeros(len(sym_dist))
    for i, r in enumerate(regimes):
        vals = sym_dist[r].values
        ax.bar(x, vals, bottom=bottoms,
               color=PALETTE[i % len(PALETTE)], alpha=0.88,
               label=f"Regime {r}", width=0.6)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sym_dist.index, fontsize=9, color="#EEE")
    ax.set_ylabel("% of bars in regime", fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Time Spent per Regime by Symbol (KMeans k={best_k}) [{tf_bar}]",
                 fontsize=10)
    _save(fig, f"{OUTPUT_FOLDER}/r017_symbol_dist_{tf_bar}.png")


def plot_regime_stats_bar(reg_result: dict, best_k: int, tf_bar: str):
    """Bar charts of key statistics for each regime."""
    regimes = reg_result["regimes"]
    stats   = reg_result["stats"]
    n       = len(regimes)
    colors  = [PALETTE[i % len(PALETTE)] for i in range(n)]
    rlbls   = [f"R{r}" for r in regimes]

    metrics = [
        ("avg_rv20",      "Realised Volatility (ann.%)",    False),
        ("avg_adx",       "Avg ADX (Trend Strength)",       False),
        ("avg_ret20",     "Avg 20-Bar Return (%)",          True),
        ("avg_bb_width",  "Avg Bollinger Width (%)",        False),
        ("avg_slope",     "EMA200 Slope (%/10-bar)",        True),
        ("avg_rsi",       "Avg RSI",                        False),
        ("avg_duration_bars", "Avg Duration (bars)",        False),
        ("persistence",   "Persistence P(stay)",            False),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)
    for ax, (key, title, symmetric) in zip(axes.flat, metrics):
        _ax_dark(ax)
        vals = [stats[r].get(key, 0) for r in regimes]
        bars = ax.bar(rlbls, vals, color=colors, alpha=0.87)
        if symmetric:
            ax.axhline(0, color="#FFF", lw=0.6)
        ax.set_title(title, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(bar.get_height())*0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#EEE")

    fig.suptitle(f"R017 Regime Statistics (KMeans k={best_k}) [{tf_bar}]",
                 color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"{OUTPUT_FOLDER}/r017_regime_stats_{tf_bar}.png")


def plot_pca_variance(proj: dict, tf_bar: str):
    """Explained variance plot for PCA components."""
    expl  = proj["pca_explained"]
    cumul = np.cumsum(expl)
    comps = [f"PC{i+1}" for i in range(len(expl))]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
    for ax in axes: _ax_dark(ax)

    axes[0].bar(comps, expl * 100, color="#4A90D9", alpha=0.88)
    axes[0].set_title("PCA — Per-Component Variance (%)", fontsize=10)
    axes[0].set_ylabel("Variance explained (%)")

    axes[1].plot(comps, cumul * 100, "o-", color="#FFD700", lw=2)
    axes[1].axhline(80, color="#00C49A", lw=0.8, ls="--", label="80% threshold")
    axes[1].axhline(95, color="#FF4560", lw=0.8, ls="--", label="95% threshold")
    axes[1].set_title("PCA — Cumulative Variance (%)", fontsize=10)
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"R017 PCA Explained Variance [{tf_bar}]", color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"{OUTPUT_FOLDER}/r017_pca_variance_{tf_bar}.png")


def plot_method_comparison(feat_df: pd.DataFrame, cluster_res: dict,
                           proj: dict, tf_bar: str):
    """2×2 PCA scatter coloured by each clustering method."""
    X_pca = proj["pca"]
    methods = ["kmeans", "gmm", "dbscan", "hierarchical"]
    titles  = ["KMeans", "Gaussian Mixture", "DBSCAN", "Hierarchical"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, method, title in zip(axes.flat, methods, titles):
        _ax_dark(ax)
        lbl     = cluster_res[method]
        regimes = sorted(set(lbl[lbl != -1]))
        colors  = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(regimes)}
        for r in regimes:
            mask = lbl == r
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[r], s=2, alpha=0.3, label=f"R{r}")
        if -1 in lbl:
            mask = lbl == -1
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c="#444", s=1, alpha=0.2, label="Noise")
        n_cl = len(regimes)
        ax.set_title(f"{title} ({n_cl} clusters)", fontsize=10)
        ax.legend(markerscale=4, fontsize=7, loc="upper right")

    fig.suptitle(f"R017 Clustering Method Comparison — PCA Space [{tf_bar}]",
                 color="#EEE", fontsize=12, y=1.01)
    _save(fig, f"{OUTPUT_FOLDER}/r017_method_comparison_{tf_bar}.png")


# =============================================================================
# SECTION 8 — REPORT
# =============================================================================

def _yn(cond): return "YES ✓" if cond else "NO  ✗"


def print_regime_summary(reg_result: dict, cluster_res: dict,
                         k_results: dict, best_k: int, tf_cfg: dict):
    bar   = tf_cfg["bar"]
    label = tf_cfg["label"]
    W     = 110
    stats = reg_result["stats"]

    print()
    print("=" * W)
    print(f"  QUANTLAB AI — RESEARCH #017  [{label}]")
    print(f"  Market Regime Classification — {label} candles")
    print(f"  {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * W)
    print()
    print(f"  Optimal k selected  : {best_k}")
    print(f"    KMeans best       : {k_results['best_k_elbow']}  (elbow)  "
          f"/ {k_results['best_k_sil']}  (silhouette)")
    print(f"    GMM best          : {k_results['best_k_bic']}  (BIC)")
    print(f"  DBSCAN clusters     : {cluster_res['n_dbscan_clusters']}  "
          f"(noise={cluster_res['n_dbscan_noise']}, eps={cluster_res['dbscan_eps']:.3f})")
    print()
    print(f"  {'Regime':<10} {'N bars':>8} {'Freq%':>7} {'AvgRV%':>8} "
          f"{'AvgADX':>8} {'AvgRet20':>9} {'BBW%':>7} "
          f"{'RSI':>6} {'AvgDur':>8} {'Persist':>8}")
    print("  " + "-" * (W - 2))

    for r, s in sorted(stats.items()):
        print(f"  Regime {r:<3d}  "
              f"{s['n']:>8,}  "
              f"{s['freq']*100:>6.1f}%  "
              f"{s['avg_rv20']:>8.4f}  "
              f"{s['avg_adx']:>8.1f}  "
              f"{s['avg_ret20']:>+9.3f}  "
              f"{s['avg_bb_width']:>7.3f}  "
              f"{s['avg_rsi']:>6.1f}  "
              f"{s['avg_duration_bars']:>8.1f}  "
              f"{s['persistence']*100:>7.1f}%")

    print()
    print("  Transition probabilities:")
    regimes = reg_result["regimes"]
    header  = "             " + "".join(f"  →R{r:d}" for r in regimes)
    print(f"  {header}")
    for i, ri in enumerate(regimes):
        row = f"  From R{ri}  :  " + "  ".join(
            f"{reg_result['trans_probs'][i,j]:5.2f}" for j in range(len(regimes)))
        print(row)
    print()


def print_seven_questions(all_tf_results: dict):
    W = 110
    print()
    print("=" * W)
    print("  R017 — SEVEN FINAL QUESTIONS")
    print("  Market Regime Classification — Regime Map")
    print("=" * W)
    print()

    for bar, res in all_tf_results.items():
        reg_result  = res["reg_result"]
        k_results   = res["k_results"]
        cluster_res = res["cluster_res"]
        best_k      = res["best_k"]
        sym_dist    = res["sym_dist"]
        proj        = res["proj"]
        label       = res["label"]

        stats     = reg_result["stats"]
        regimes   = reg_result["regimes"]
        trans     = reg_result["trans_probs"]
        explained = proj["pca_explained"]

        sil_scores  = list(k_results["silhouettes"].values())
        best_sil    = max(sil_scores)
        persistence = [stats[r]["persistence"] for r in regimes]
        avg_persist = np.mean(persistence)
        freqs       = [stats[r]["freq"] for r in regimes]
        # Entropy of regime freq distribution (lower = more dominant one regime)
        freq_entropy = -sum(f * math.log(f+1e-9) for f in freqs) / math.log(len(freqs)+1e-9)

        # Feature importance from PCA loadings (sum of abs loadings on PC1+PC2)
        pc12_load = np.abs(proj["pca_components"][:2, :]).sum(axis=0)
        top_feat_idx = np.argsort(pc12_load)[::-1][:5]
        top_feats = [FEATURE_COLS[i] for i in top_feat_idx]

        # DBSCAN: did it find structure?
        dbscan_clean = cluster_res["n_dbscan_clusters"] >= 2 and \
                       cluster_res["n_dbscan_noise"] / (len(res["labels"]) or 1) < 0.20

        print(f"  ══ [{label}] ══════════════════════════════════════════════════════")
        print()
        print(f"  Q1. Do crypto markets naturally separate into distinct regimes?")
        distinct = best_sil > 0.08 and dbscan_clean
        print(f"      Best silhouette score : {best_sil:.4f}  (>0.08 = meaningful separation)")
        print(f"      DBSCAN found clean clusters : {_yn(dbscan_clean)}")
        print(f"      → {_yn(distinct)}")
        print()

        print(f"  Q2. How many regimes exist?")
        print(f"      KMeans elbow     : k={k_results['best_k_elbow']}")
        print(f"      KMeans silhouette: k={k_results['best_k_sil']}")
        print(f"      GMM BIC          : k={k_results['best_k_bic']}")
        print(f"      DBSCAN found     : {cluster_res['n_dbscan_clusters']} clusters")
        print(f"      → Selected k={best_k}  (majority / silhouette)")
        print()

        print(f"  Q3. How persistent are they?")
        for r in regimes:
            print(f"      Regime {r}: P(stay)={stats[r]['persistence']:.3f}  "
                  f"avg duration={stats[r]['avg_duration_bars']:.1f} bars  "
                  f"({stats[r]['avg_duration_bars'] * (15 if bar=='15m' else 60) / 60:.1f} hrs)")
        print(f"      Avg persistence across all regimes: {avg_persist:.3f}")
        persistent = avg_persist > 0.6
        print(f"      → {_yn(persistent)}  (>0.60 = persistent)")
        print()

        print(f"  Q4. Are transitions predictable?")
        # Look for strong off-diagonal concentrations (non-uniform transitions)
        off_diag = []
        for i in range(len(regimes)):
            row = [trans[i,j] for j in range(len(regimes)) if j != i]
            if row:
                off_diag.append(max(row))
        max_off  = max(off_diag) if off_diag else 0
        predictable = max_off > 0.25 or avg_persist > 0.70
        print(f"      Strongest off-diagonal transition: {max_off:.3f}")
        print(f"      High persistence reduces transition uncertainty: {avg_persist:.3f}")
        print(f"      → {_yn(predictable)}  (some predictability observed)")
        print()

        print(f"  Q5. Which features best distinguish each regime?")
        print(f"      Top features by PCA PC1+PC2 loading:")
        for i, feat in enumerate(top_feats):
            print(f"        {i+1}. {feat}  (loading={pc12_load[top_feat_idx[i]]:.3f})")
        pca_80 = np.argmax(np.cumsum(explained) >= 0.80) + 1
        print(f"      PCA: {pca_80} components explain 80% of variance")
        print()

        print(f"  Q6. Which symbols spend most time in each regime?")
        for r in regimes:
            if r in sym_dist.columns:
                sym_vals = sym_dist[r].sort_values(ascending=False)
                top_sym  = sym_vals.index[0]
                bot_sym  = sym_vals.index[-1]
                print(f"      Regime {r}: most={top_sym} ({sym_vals.iloc[0]:.1f}%)  "
                      f"least={bot_sym} ({sym_vals.iloc[-1]:.1f}%)")
        print()

        print(f"  Q7. Is there evidence that future strategies should first identify")
        print(f"      the regime before searching for entries?")
        evidence = distinct and persistent and best_sil > 0.08
        print(f"      Distinct regimes exist     : {_yn(distinct)}")
        print(f"      Regimes are persistent     : {_yn(persistent)}")
        print(f"      PCA structure is clear     : {_yn(explained[0] > 0.15)}")
        verdict_str = (
            "YES ✓  — Regimes are statistically distinct and persistent. "
            "Regime-conditioning should improve strategy performance."
            if evidence else
            "WEAK ✗  — Regime separation is marginal. More data or features may help."
        )
        print(f"      → {verdict_str}")
        print()
        print("  " + "─" * (W - 2))
        print()

    print("=" * W)


# =============================================================================
# SECTION 9 — REGIME LABELLING / INTERPRETATION
# =============================================================================

def interpret_regimes(reg_result: dict) -> dict:
    """
    Assign intuitive names to regimes based on their statistical fingerprint.
    Uses simple rule-based classification on avg_adx, avg_rv20, avg_slope.
    """
    stats  = reg_result["stats"]
    labels = {}

    # Normalise for comparison
    adx_vals  = {r: stats[r]["avg_adx"]    for r in stats}
    rv_vals   = {r: stats[r]["avg_rv20"]   for r in stats}
    slope_vals= {r: stats[r]["avg_slope"]  for r in stats}
    rsi_vals  = {r: stats[r]["avg_rsi"]    for r in stats}

    max_adx   = max(adx_vals.values())  + 1e-9
    max_rv    = max(rv_vals.values())   + 1e-9

    for r in stats:
        adx_rel   = adx_vals[r]   / max_adx
        rv_rel    = rv_vals[r]    / max_rv
        slope     = slope_vals[r]
        rsi       = rsi_vals[r]

        if adx_rel > 0.65 and slope > 0:
            name = "Trending Bull"
        elif adx_rel > 0.65 and slope < 0:
            name = "Trending Bear"
        elif rv_rel > 0.70:
            name = "High Volatility / Chaotic"
        elif rv_rel < 0.35 and adx_rel < 0.40:
            name = "Low Vol / Compression"
        elif rsi > 55 and slope > 0:
            name = "Mild Uptrend"
        elif rsi < 45 and slope < 0:
            name = "Mild Downtrend"
        else:
            name = "Ranging / Choppy"

        labels[r] = name

    return labels


# =============================================================================
# SECTION 10 — MAIN PIPELINE
# =============================================================================

def run_timeframe(tf_cfg: dict) -> dict:
    bar     = tf_cfg["bar"]
    months  = tf_cfg["months"]
    min_c   = tf_cfg["min_cache"]
    label   = tf_cfg["label"]

    print(f"\n{'='*72}")
    print(f"  TIMEFRAME: {label}  [{bar}]")
    print(f"{'='*72}")

    # ── 1. Download ──────────────────────────────────────────────────────────
    print(f"\n  Downloading {bar} data...", flush=True)
    t0       = time.time()
    raw_data = download_all_parallel(bar, months, min_c)
    elapsed  = time.time() - t0
    for sym, df in raw_data.items():
        print(f"  {sym:20s}  {len(df):>8,} candles  "
              f"({df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()})")
    print(f"  Download complete in {elapsed:.0f}s")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print(f"\n  Computing features...", flush=True)
    feat_frames = []
    for sym, df_raw in raw_data.items():
        fdf = compute_features(df_raw, sym)
        fdf = fdf.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        feat_frames.append(fdf)
        print(f"  {sym.split('-')[0]:6s}  {len(fdf):>7,} clean bars", flush=True)

    if not feat_frames:
        raise RuntimeError("No feature data available")

    feat_df = pd.concat(feat_frames, ignore_index=True)
    print(f"\n  Total pooled bars: {len(feat_df):,}")

    # ── 3. Scale ─────────────────────────────────────────────────────────────
    X_raw    = feat_df[FEATURE_COLS].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── 4. Find optimal k ────────────────────────────────────────────────────
    k_results = find_optimal_k(X_scaled)
    best_k    = k_results["best_k"]

    # ── 5. Clustering ────────────────────────────────────────────────────────
    cluster_res = run_all_clustering(X_scaled, best_k)

    # Primary labels: KMeans
    labels = cluster_res["kmeans"]

    # ── 6. Regime statistics ─────────────────────────────────────────────────
    print("\n  Computing regime statistics...", flush=True)
    reg_result = regime_stats(feat_df, labels, "KMeans")
    interp     = interpret_regimes(reg_result)
    sym_dist   = symbol_regime_distribution(feat_df, labels)

    print(f"\n  Regime interpretations:")
    for r, name in interp.items():
        s = reg_result["stats"][r]
        print(f"    Regime {r}: {name:<30s}  "
              f"freq={s['freq']*100:.1f}%  ADX={s['avg_adx']:.1f}  "
              f"RV={s['avg_rv20']:.4f}  slope={s['avg_slope']:+.4f}  "
              f"persist={s['persistence']*100:.1f}%")

    # ── 7. Dimensionality reduction ──────────────────────────────────────────
    print(f"\n  Dimensionality reduction...", flush=True)
    proj = compute_projections(X_scaled, labels)

    # ── 8. Visualisations ────────────────────────────────────────────────────
    print(f"\n  Generating visualisations for [{bar}]...", flush=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    plot_elbow_silhouette(k_results, bar)
    plot_pca_variance(proj, bar)
    plot_pca_scatter(proj, labels, best_k, bar)
    plot_tsne(proj, best_k, bar)
    plot_correlation_heatmap(feat_df, labels, bar)
    plot_transition_matrix(reg_result, bar, method="KMeans")
    plot_regime_timeline(feat_df, labels, best_k, bar)
    plot_symbol_regime_dist(sym_dist, best_k, bar)
    plot_regime_stats_bar(reg_result, best_k, bar)
    plot_method_comparison(feat_df, cluster_res, proj, bar)

    if UMAP_AVAILABLE and "umap" in proj:
        # Save UMAP plot
        X_umap   = proj["umap"]
        umap_lbl = proj["umap_labels"]
        regimes_u = sorted(set(umap_lbl[umap_lbl != -1]))
        colors_u  = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(regimes_u)}
        fig, ax   = plt.subplots(figsize=(12, 9), facecolor=BG)
        _ax_dark(ax)
        for r in regimes_u:
            mask = umap_lbl == r
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       c=colors_u[r], s=4, alpha=0.45, label=f"Regime {r}")
        ax.set_title(f"UMAP Projection (KMeans k={best_k}) [{bar}]", fontsize=12)
        ax.legend(markerscale=4, fontsize=9)
        _save(fig, f"{OUTPUT_FOLDER}/r017_umap_{bar}.png")

    # ── 9. Console report ────────────────────────────────────────────────────
    print_regime_summary(reg_result, cluster_res, k_results, best_k, tf_cfg)

    return {
        "feat_df":    feat_df,
        "labels":     labels,
        "k_results":  k_results,
        "best_k":     best_k,
        "cluster_res":cluster_res,
        "reg_result": reg_result,
        "sym_dist":   sym_dist,
        "proj":       proj,
        "interp":     interp,
        "label":      label,
    }


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(CACHE_FOLDER,  exist_ok=True)

    print()
    print("╔" + "═" * 79 + "╗")
    print("║  QUANTLAB AI — RESEARCH #017" + " " * 50 + "║")
    print("║  Market Regime Classification" + " " * 49 + "║")
    print("╚" + "═" * 79 + "╝")
    print()
    print("  Objective  : Map the natural regime structure of crypto futures markets")
    print("  Output     : Market Regime Map for use in future research")
    print("  NOT a trading strategy — no entries, exits, or optimisation")
    print()
    if UMAP_AVAILABLE:
        print("  UMAP       : available ✓")
    else:
        print("  UMAP       : not available — skipping (PCA + t-SNE will be used)")
    print()

    all_tf_results = {}

    for tf_cfg in TIMEFRAMES:
        bar = tf_cfg["bar"]
        result = run_timeframe(tf_cfg)
        all_tf_results[bar] = result
        result["label"] = tf_cfg["label"]

    # ── Seven final questions ─────────────────────────────────────────────────
    print_seven_questions(all_tf_results)

    # ── Save regime map summary CSV ───────────────────────────────────────────
    rows = []
    for bar, res in all_tf_results.items():
        stats  = res["reg_result"]["stats"]
        interp = res["interp"]
        for r, s in stats.items():
            row = {
                "timeframe":    bar,
                "regime":       r,
                "interpretation": interp.get(r, "Unknown"),
                "n_bars":       s["n"],
                "freq_pct":     round(s["freq"] * 100, 2),
                "avg_rv20":     round(s["avg_rv20"], 5),
                "avg_adx":      round(s["avg_adx"], 2),
                "avg_ret20":    round(s["avg_ret20"], 4),
                "avg_ret50":    round(s["avg_ret50"], 4),
                "avg_bb_width": round(s["avg_bb_width"], 4),
                "avg_slope":    round(s["avg_slope"], 5),
                "avg_rsi":      round(s["avg_rsi"], 2),
                "avg_momentum": round(s["avg_momentum"], 4),
                "avg_dist_ema": round(s["avg_dist_ema"], 4),
                "avg_vol_pct":  round(s["avg_vol_pct"], 4),
                "avg_duration_bars": round(s["avg_duration_bars"], 1),
                "persistence":  round(s["persistence"], 4),
            }
            rows.append(row)

    regime_map_path = f"{OUTPUT_FOLDER}/r017_regime_map.csv"
    pd.DataFrame(rows).to_csv(regime_map_path, index=False)
    print(f"\n  Regime map saved → {regime_map_path}")

    print(f"\n  All outputs → {OUTPUT_FOLDER}/r017_*")
    print(f"  Research #017 complete.")
    print()


if __name__ == "__main__":
    main()
