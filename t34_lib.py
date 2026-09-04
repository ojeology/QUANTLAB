"""
t34_lib.py — shared machinery for the T34 small-account leverage × R:R study.
Built on the blind-validation engine (ql_engine/svm_deploy), faithful to the
branch's walk-forward protocol (train on prior years only, test each year).

Legs:
  TREND  = Donchian(20) breakout + ADX14>20 + close>EMA200, SL = atr_mult*ATR.
           exit AS-IS: SL else Donchian(20)-break-close  (verbatim T25d).
           RR variant: add TP = entry + rr_R * (atr_mult*ATR), TP-before-SL.
  MR     = FAM_A coiled (BBW_STRICT+RV_LO+DST_NR+PRG_VH) green gate rel_vol>1.5,
           SL=1 ATR, TP=rr*1 ATR (sim_symbol exit 'base'), rr sweep.
Filters (per test year, trained on prior years only):
  TREND: RandomForest q0.65 keep by P(win).   MR: SVMQ65Adaptive q0.65.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/bv")
sys.path.insert(0, "/home/user/bv/scripts")
from ql_engine import (add_features, build_signal_mask, sim_symbol,
                       IS_LOOKBACK, RECAL_EVERY)
from svm_deploy import build_mldf, FEATS
from sklearn.ensemble import RandomForestClassifier

SAVE_DIR = "/home/user/bv/quantlab_cache_2023"
FEE = 0.0005                       # 0.05% per side
FAM_A = ["BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH"]


# ------------------------------------------------------------------ data ----
def load_feats(save_dir=SAVE_DIR, cache_dir="/home/user/quantlab/quantlab_cache",
               strict=True):
    """Load feature frames for every symbol in the save_dir manifest.

    strict=True (STUDY DEFAULT): every manifest symbol MUST have its parquet in
    save_dir. 2024/2025/2026 walk-forward folds need full 2023 history for the
    2024 train + OOS split; silently substituting the 2024+ local cache drops
    ~1000 2024 trades and empties MR-2024 (observed 2026-09-04 after a snapshot
    lost 43/50 parquets). strict=False keeps the cache fallback for
    diagnostics only.
    """
    feats, above20, missing = {}, {}, []
    syms = []
    manifest_p = os.path.join(save_dir, "_done_2023.txt")
    if os.path.exists(manifest_p):
        syms = [l.strip() for l in open(manifest_p) if l.strip()]
    for sym in syms:
        p = os.path.join(save_dir, f"{sym}_1H_full.parquet")
        if not os.path.exists(p):
            if strict:
                missing.append(sym)
                continue
            p = os.path.join(cache_dir, f"{sym}_1H.parquet")
        if not os.path.exists(p):
            missing.append(sym)
            continue
        try:
            df = pd.read_parquet(p)
            df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
            for c in ["open", "high", "low", "close", "vol"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close", "vol"])
            f = add_features(df)
            f = f.dropna(subset=["ema200", "atr14", "adx14", "ema_dist_pct",
                                 "real_vol_20", "bb_width", "prev_range_r", "prev_body_r"])
            if len(f) >= IS_LOOKBACK + RECAL_EVERY + 100:
                feats[sym] = f
                above20[sym] = (f["close"] > f["ema20"]).astype(float)
        except Exception as e:
            print("  err", sym, str(e)[:60])
    if missing:
        raise RuntimeError(
            f"DATA INTEGRITY: {len(missing)}/{len(syms)} manifest symbols missing "
            f"from {save_dir}: {missing[:8]}{' ...' if len(missing) > 8 else ''}. "
            f"Re-run fetch_2023_50.py before any study; never fall back to the "
            f"2024+ cache silently.")
    breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
    breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100
    return feats, above20, breadth, breadth_pct


# ------------------------------------------------------------ TREND leg ----
def donchian(df, N=20, Nx=20, atr_mult=2.0, adx_min=20.0, tp_R=None, time_bars=None):
    """Entry verbatim T25d. Exit:
       tp_R=None   -> SL else Donchian-break close  (verbatim)
       tp_R given  -> TP(entry+tp_R*riskdist) first (intrabar TP-before-SL), else SL,
                      else Donchian-break close; optional time stop.
    Returns rich trades (entry/atr/sl_frac kept for R-normalisation)."""
    hh = df["high"].rolling(N).max().shift(1)
    ll = df["low"].rolling(Nx).min().shift(1)
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    o = df["open"].values
    adx = df["adx14"].values; ema = df["ema200"].values; atr = df["atr14"].values
    idx = df.index
    trades = []; in_pos = False; ep = None; sl = None; sl_d = None; ei = None
    for i in range(N, len(df)):
        if not in_pos:
            if c[i] > hh.iloc[i] and adx[i] > adx_min and c[i] > ema[i]:
                if not (atr[i] > 0) or np.isnan(atr[i]):
                    continue
                ep = c[i]; sl_d = atr_mult * atr[i]; sl = ep - sl_d
                in_pos = True; ei = i
        else:
            ex = None; xt = None
            if tp_R is not None and h[i] >= ep + tp_R * sl_d:
                ex = ep + tp_R * sl_d; xt = "TP"
            elif l[i] <= sl:
                ex = sl; xt = "SL"
            elif c[i] < ll.iloc[i]:
                ex = c[i]; xt = "BRK"
            elif time_bars is not None and (i - ei) >= time_bars:
                ex = c[i]; xt = "TIME"
            if ex is not None:
                trades.append(dict(entry_time=idx[ei], exit_time=idx[i],
                                   entry=ep, exit=ex, atr=atr[ei],
                                   sl_frac=sl_d / ep, r_price=(ex / ep - 1.0),
                                   exit_type=xt, bars_in=i - ei))
                in_pos = False
    return trades


def trend_raw(feats, **kw):
    raw = []
    for s, f in feats.items():
        for t in donchian(f, **kw):
            t = dict(t); t["sym"] = s
            t["r"] = t["r_price"]                 # build_mldf win label
            raw.append(t)
    raw.sort(key=lambda t: t["entry_time"])
    return raw


def trend_champion(raw, feats, breadth, breadth_pct, years=(2024, 2025, 2026),
                   q=0.65, rf_seed=0, anchor="entry"):
    """RF q0.65 champion, walk-forward per prior-years-only fold.

    anchor='entry' : mldf features at the TRUE entry bar  (implementable live).
    anchor='exit'  : VERBATIM branch behaviour — t25_*.py record 'entry_time' at
                     the EXIT bar, so the filter selects trades on exit-bar
                     features. Reproduces the branch logs but is NOT
                     implementable live (post-entry info in the gate). Used only
                     by the stage-1 validator to reproduce reference numbers.
    """
    ts_key = "exit_time" if anchor == "exit" else "entry_time"
    raw2 = [dict(t, entry_time=t[ts_key]) for t in raw]     # anchor for mldf
    mldf = build_mldf(raw2, feats, breadth, breadth_pct)
    out = []
    for Y in years:
        tr = mldf[mldf.ts < pd.Timestamp(f"{Y}-01-01", tz="UTC")]
        te = mldf[mldf.ts.dt.year == Y]
        if len(tr) < 50 or len(te) == 0:
            continue
        m = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                                   class_weight="balanced", n_jobs=-1,
                                   random_state=rf_seed).fit(tr[FEATS], tr["win"])
        P = m.predict_proba(te[FEATS])[:, 1]
        thr = np.quantile(P, 1 - q)
        kept = set(te[P >= thr]["ts"])
        for t in raw:
            if t[ts_key].year == Y and t[ts_key] in kept:
                tt = dict(t)
                if anchor == "exit":                  # branch convention: record
                    tt["entry_time"] = t["exit_time"]  # ts = exit bar
                out.append(tt)
    return out


# -------------------------------------------------------------- MR leg -----
def mr_raw(feats, rr=1.5, gate="green", relvol=1.5):
    masks = {s: build_signal_mask(f, FAM_A, gate, relvol) for s, f in feats.items()}
    raw = []
    for s in feats:
        for t in sim_symbol(feats[s], masks[s], rr,
                            dict(entry_next=False, exit="base", hours=None)):
            t = dict(t); t["sym"] = s; raw.append(t)
    raw.sort(key=lambda t: t["entry_time"])
    return raw


def mr_champion(raw, feats, breadth, breadth_pct, years=(2024, 2025, 2026), q=0.65,
                ema_dist_thr=2.0, atr_ceil=70.0):
    from svm_deploy import SVMQ65Adaptive
    # adaptive VolCeil pre-filter (verbatim SVMQ65Adaptive._keep_mask logic)
    mldf_all = build_mldf(raw, feats, breadth, breadth_pct)
    keep_m = (mldf_all["atr_rank"] <= atr_ceil) | (mldf_all["ema_dist_pct"].abs() <= ema_dist_thr)
    mldf = mldf_all[keep_m]
    kept_ids = set(mldf["ts"])
    fr = [t for t in raw if t["entry_time"] in kept_ids]
    out = []
    for Y in years:
        tr = mldf[mldf.ts < pd.Timestamp(f"{Y}-01-01", tz="UTC")]
        te = mldf[mldf.ts.dt.year == Y]
        if len(tr) < 50 or len(te) == 0:
            continue
        model = SVMQ65Adaptive(q=q, ema_dist_thr=ema_dist_thr).fit_mldf(tr)
        kept, _ = model.keep_mldf(te)
        for t in fr:
            if t["entry_time"].year == Y and t["entry_time"] in kept:
                out.append(t)
    return out


# ------------------------------------------------------- risk / R-math ----
def r_adj(t, sl_mult_extra=1.0):
    """Canonical fee-adjusted R. R = pnl / risk-distance; fee in R =
       2*FEE / sl_frac. For MR trades r already R (risk-dist = 1*ATR);
       sl_frac = atr/entry approx used for fee only (verbatim cost_adjusted_rs)."""
    if "r" in t and t.get("sl_frac") is None:
        pass
    if "sl_frac" in t:                       # trend-style rich trade
        return t["r_price"] / t["sl_frac"] - (2 * FEE) / t["sl_frac"]
    # MR-style sim_symbol trade: r in R; fee via cost_adjusted_rs(0.05)
    return t["r"] - 2.0 * FEE * t["entry"] / t["atr"]


def events(trades, risk):
    """List of (entry_time, delta) where delta = risk*R_adj per trade."""
    out = []
    for t in trades:
        out.append((t["entry_time"], risk * r_adj(t)))
    out.sort(key=lambda x: x[0])
    return out


def simulate(ev, start=100.0):
    """Chronological, batched per timestamp: eq *= (1 + sum deltas at ts).
    Curve keeps a unique, monotone timestamp index (baseline at first event
    ts, then one point per distinct event ts carrying the updated equity)."""
    eq = start; curve = []; i = 0
    def push(ts, val):
        if not curve or curve[-1][0] != ts:
            curve.append((ts, val))
    if ev:
        push(ev[0][0], eq)
    while i < len(ev):
        ts = ev[i][0]; d = 0.0
        while i < len(ev) and ev[i][0] == ts:
            d += ev[i][1]; i += 1
        eq *= (1.0 + d)
        if eq < 0.0:                    # concurrent losses > account -> ruin
            eq = 0.0
        push(ts, eq)
    if not curve:
        curve = [(ev[0][0], eq)]
    return eq, curve


def equity_metrics(ev, start=100.0, risk_label="", per_year=False):
    _, curve = simulate(ev, start)
    ts = np.array([x[0] for x in curve], dtype="datetime64[ns]")
    eq = np.array([x[1] for x in curve], dtype=float)
    # daily-ish resample to compute month stats + MDD on the real path
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk).min())
    final = eq[-1]
    yrs = max((ts[-1] - ts[0]).astype("timedelta64[D]").astype(int) / 365.25, 1e-9)
    cagr = (final / start) ** (1 / yrs) - 1

    s = pd.Series(eq, index=pd.DatetimeIndex(ts)).resample("ME").last()
    monthly_ret = s.pct_change().dropna()
    prof_months = int((monthly_ret > 0).sum())
    tot_months = len(monthly_ret)
    worst_month = float(monthly_ret.min()) if len(monthly_ret) else 0.0
    # longest losing-month streak
    streak = worst = 0
    for v in monthly_ret < 0:
        streak = streak + 1 if v else 0
        worst = max(worst, streak)
    return dict(final=final, ret=final / start - 1, cagr=cagr, mdd=mdd,
                prof_months=prof_months, tot_months=tot_months,
                worst_month=worst_month, worst_streak=worst)


def monte_carlo_risk(rs_deltas_flat, n_paths=4000, seed=7):
    """Bootstrap over per-trade deltas (unit-R_adj values, ALREADY scaled by
    the per-trade risk fraction the caller wants to test)."""
    rng = np.random.default_rng(seed)
    rs = np.asarray(rs_deltas_flat, dtype=float)
    fin = np.empty(n_paths); mdd = np.empty(n_paths); ruin = np.empty(n_paths)
    for k in range(n_paths):
        s = rs[rng.integers(0, len(rs), len(rs))]
        eq = np.empty(len(s) + 1); eq[0] = 100.0
        for j, r in enumerate(s):
            eq[j + 1] = eq[j] * (1 + r)
        pk = np.maximum.accumulate(eq)
        dd = ((eq - pk) / pk).min()
        fin[k] = eq[-1]; mdd[k] = dd
        ruin[k] = (eq < 50.0).any() or (dd < -0.99)
    return dict(end_p5=float(np.percentile(fin, 5)), end_p50=float(np.percentile(fin, 50)),
                end_p95=float(np.percentile(fin, 95)),
                p_profit=float((fin > 100).mean()),
                p_double=float((fin >= 200).mean()),
                p_triple=float((fin >= 300).mean()),
                p_halve=float((fin < 50).mean()),
                p_dd15=float((mdd <= -0.15).mean()),
                p_dd30=float((mdd <= -0.30).mean()),
                p_dd50=float((mdd <= -0.50).mean()),
                dd_p5=float(np.percentile(mdd, 5)))


def kelly(unit_rs):
    rs = np.asarray(unit_rs, dtype=float)
    mu = rs.mean(); var = rs.var()
    k = mu / var if var > 0 else 0.0
    # numerical full-kelly on log growth
    grid = np.linspace(0.001, min(max(3.0 * abs(k) + 0.1, 0.5), 5.0), 600)
    g = np.array([np.log1p(gf * rs).mean() for gf in grid])
    fk = float(grid[np.argmax(g)]) if g.max() > 0 else 0.0
    return dict(mean=float(mu), var=float(var), half_kelly=float(k * 0.5), kelly=float(k),
                full_kelly=float(fk))
