"""
svm_deploy.py — Deployable SVM q0.75 champion strategy (reusable module).

Wraps QUANTLAB's exact engine (ql_engine.add_features / build_signal_mask /
sim_symbol) + the research SVM pipeline (StandardScaler + SVC, keep top-q by
P(win)) into a clean, importable class for the 2027-01-01 go-live.

Usage:
    from svm_deploy import SVMQ75, load_universe
    feats, above20, raw, breadth, breadth_pct, mldf = load_universe()  # full history
    model = SVMQ75(rr=1.5, q=0.75).fit_mldf(mldf[mldf.ts < TRAIN_END])
    kept_ts, pred = model.keep_mldf(mldf[mldf.ts >= TEST_START])
    # kept_ts = entry timestamps the strategy would have traded
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))
from ql_engine import add_features, build_signal_mask, sim_symbol, IS_LOOKBACK, RECAL_EVERY
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import demo_bot as bot

FAM_A = ["BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"]
BASE_FEATS = ["atr_rank", "adx14", "rsi14", "ema_dist_pct", "prev_body_r",
              "prev_range_r", "rel_vol", "bb_width", "real_vol_20", "hour", "dow"]
SPECIAL = ["breadth_q", "dist_hi48", "green_streak"]
FEATS = BASE_FEATS + SPECIAL


def load_universe(cache_dir="quantlab_cache", topup_bars=800):
    """Load 1H data (cache + fresh OKX top-up), build features, signals, trades, mldf.

    Returns (feats, above20, raw_trades, breadth, breadth_pct, mldf)."""
    syms = {f[:-len("_1H.parquet")] for f in os.listdir(cache_dir) if f.endswith("_1H.parquet")}
    feats, above20 = {}, {}
    for sym in sorted(syms):
        p = os.path.join(cache_dir, f"{sym}_1H.parquet")
        if not os.path.exists(p):
            continue
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
            for c in ["open", "high", "low", "close", "vol"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(subset=["open", "high", "low", "close", "vol"], inplace=True)
            if len(df) < IS_LOOKBACK + RECAL_EVERY + 100:
                continue
            inst = sym.replace("_", "-")
            fresh = bot.fetch_candles(inst, n_bars=topup_bars)
            if fresh is not None and len(fresh):
                fresh = fresh.astype(float)
                df = pd.concat([df, fresh]); df = df[~df.index.duplicated(keep="last")].sort_index()
            f = add_features(df)
            f.dropna(subset=["ema200", "atr14", "adx14", "ema_dist_pct", "real_vol_20",
                             "bb_width", "prev_range_r", "prev_body_r"], inplace=True)
            if len(f) >= IS_LOOKBACK + RECAL_EVERY:
                feats[sym] = f
                above20[sym] = (f["close"] > f["ema20"]).astype(float)
        except Exception:
            pass
    breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
    breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100
    mask = {s: build_signal_mask(f, FAM_A, "green", 1.5) for s, f in feats.items()}
    raw = []
    for s, f in feats.items():
        for t in sim_symbol(f, mask[s], 1.5, dict(entry_next=False, exit="base", hours=None)):
            t["sym"] = s
            raw.append(t)
    raw.sort(key=lambda t: t["entry_time"])
    mldf = build_mldf(raw, feats, breadth, breadth_pct)
    return feats, above20, raw, breadth, breadth_pct, mldf


def build_mldf(raw, feats, breadth, breadth_pct):
    """Construct the 14-feature matrix (verbatim from R087) from raw trades."""
    rows = []
    for t in raw:
        sym = t["sym"]; ts = t["entry_time"]; f = feats[sym]
        if ts not in f.index:
            continue
        row = f.loc[ts]; i = f.index.get_loc(ts); c = float(row["close"])
        hi48 = float(f["close"].rolling(48).max().iloc[i]) if i >= 0 else np.nan
        dist = (c / hi48 - 1) * 100 if pd.notna(hi48) and hi48 > 0 else 0.0
        streak = 0
        for k in range(0, 6):
            j = i - k
            if j < 0:
                break
            if f["close"].iloc[j] > f["open"].iloc[j]:
                streak += 1
            else:
                break
        bq = float(breadth_pct.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) \
            if ts >= breadth_pct.index[0] else 50.0
        rows.append(dict(sym=sym, ts=ts, r=t["r"], win=int(t["r"] > 0),
                         **{c2: row.get(c2, 0) for c2 in BASE_FEATS},
                         breadth_q=bq, dist_hi48=dist, green_streak=streak))
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


class SVMQ75:
    """SVM q0.75 champion. Fit on a training mldf, keep top-q of a test mldf."""

    def __init__(self, rr=1.5, q=0.75, fee=0.0005):
        self.rr = rr
        self.q = q
        self.fee = fee
        self.scaler = None
        self.clf = None
        self.fitted = False

    def fit_mldf(self, mldf_train):
        X = mldf_train[FEATS].fillna(0).values
        y = mldf_train["win"].values
        self.scaler = StandardScaler()
        self.clf = SVC(C=1.0, gamma="scale", probability=True)
        self.clf.fit(self.scaler.fit_transform(X), y)
        self.fitted = True
        return self

    def keep_mldf(self, mldf_test, q=None):
        """Return (kept_entry_times:set, preds:np.array) for test mldf."""
        if not self.fitted:
            raise RuntimeError("call fit_mldf first")
        q = q or self.q
        preds = self.clf.predict_proba(self.scaler.transform(mldf_test[FEATS].fillna(0).values))[:, 1]
        thr = np.quantile(preds, 1 - q)
        return set(mldf_test[preds >= thr]["ts"]), preds


class SVMQ65Adaptive(SVMQ75):
    """CHAMPION deployable filter — validated blind OOS, all of 2024-2026.

    SVM q0.65 + VolCeil gated by |ema_dist_pct| > 2.0.
    The static VolCeil (skip ATR-spike entries, atr_rank>70) is applied ONLY when
    price is stretched from its mean (|ema_dist_pct|>2.0); in calm regimes it is
    off. This resolves the 2024/2026 opposition that kills every static filter.

    Validated (30-sym, fees 0.05%):
        2024 PF@cost 1.18 | 2025 1.82 | 2026 1.53  (all > 1)
        max DD ~9.4% at 1% risk; 22/30 symbols profitable (broad, not concentrated).

    Usage:
        model = SVMQ65Adaptive().fit_mldf(mldf[mldf.ts < TRAIN_END])
        kept_ts, pred = model.keep_mldf(mldf[mldf.ts >= TEST_START])
    """
    def __init__(self, rr=1.5, q=0.65, fee=0.0005, ema_dist_thr=2.0):
        super().__init__(rr=rr, q=q, fee=fee)
        self.ema_dist_thr = ema_dist_thr

    def _keep_mask(self, mldf):
        # True = keep this candidate (VolCeil NOT triggered:
        #        either calm regime OR not an ATR spike)
        return (mldf["atr_rank"] <= 70) | (mldf["ema_dist_pct"].abs() <= self.ema_dist_thr)

    def fit_mldf(self, mldf_train):
        return super().fit_mldf(mldf_train[self._keep_mask(mldf_train)])

    def keep_mldf(self, mldf_test, q=None):
        if not self.fitted:
            raise RuntimeError("call fit_mldf first")
        q = q or self.q
        sub = mldf_test[self._keep_mask(mldf_test)]
        if len(sub) < 50:
            return set(), np.array([])
        preds = self.clf.predict_proba(self.scaler.transform(sub[FEATS].fillna(0).values))[:, 1]
        thr = np.quantile(preds, 1 - q)
        return set(sub[preds >= thr]["ts"]), preds

