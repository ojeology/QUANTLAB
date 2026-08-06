"""
Shared bot-faithful engine for QUANTLAB research (R073 onward).

Single source of truth for the execution model that mirrors demo_bot.py:
  - thresholds recalibrated every RECAL_EVERY bars from last IS_LOOKBACK bars
  - signal at bar i -> entry at close of bar i (entry_next=False, E6)
                             or close of bar i+1 (entry_next=True, E0/bot)
  - exits checked from the first full bar after entry, intrabar, TP before SL
  - max one position per symbol; risk = RISK_PCT of running capital, compounding
"""
import math
import numpy as np
import pandas as pd

from quantlab_ai import calc_ema, calc_atr, calc_adx

IS_LOOKBACK  = 500
RECAL_EVERY  = 168
STARTING_CAP = 10_000.0
RISK_PCT     = 0.01

COND_DEF = {
    "DST_NR":     ("ema_dist_pct", "lt_q", 0.33),
    "ADX_ST":     ("adx14",        "gt_q", 0.67),
    "PBD_HI":     ("prev_body_r",  "gt_q", 0.67),
    "BBW_STRICT": ("bb_width",     "lt_q", 0.25),
    "RV_LO":      ("real_vol_20",  "lt_q", 0.33),
    "PRG_VH":     ("prev_range_r", "gt_q", 0.80),
}


def calc_rsi(series, period=14):
    delta = series.diff()
    up   = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs   = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_features(df):
    df = df.copy()
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    df["ema200"]        = calc_ema(c, 200)
    df["ema50"]         = calc_ema(c, 50)
    df["ema20"]         = calc_ema(c, 20)
    df["ema50_slope"]   = (df["ema50"] - df["ema50"].shift(5)) / df["ema50"].shift(5).replace(0, np.nan) * 100.0
    df["ema200_slope"]  = (df["ema200"] - df["ema200"].shift(5)) / df["ema200"].shift(5).replace(0, np.nan) * 100.0
    df["atr14"]         = calc_atr(df, 14)
    bb_mid              = c.rolling(20).mean()
    bb_std              = c.rolling(20).std(ddof=0)
    df["bb_width"]      = (bb_std * 2) / bb_mid.replace(0, np.nan) * 100.0
    df["bb_lower"]      = bb_mid - 2 * bb_std
    df["bb_upper"]      = bb_mid + 2 * bb_std
    df["real_vol_20"]   = c.pct_change().rolling(20).std() * 100.0
    ema200_s            = df["ema200"].replace(0, np.nan)
    df["ema_dist_pct"]  = (c - ema200_s) / ema200_s * 100.0
    df["prev_range_r"]  = (h.shift(1)-l.shift(1)).abs() / c.shift(1).replace(0,np.nan) * 100.0
    df["prev_body_r"]   = (c.shift(1)-o.shift(1)).abs() / c.shift(1).replace(0,np.nan) * 100.0
    df["adx14"]         = calc_adx(df, 14)
    df["rsi14"]         = calc_rsi(c, 14)
    vol_avg             = df["vol"].rolling(20).mean()
    df["rel_vol"]       = df["vol"] / vol_avg.replace(0, np.nan)
    # volatility rank (for the "skip when ATR already spiking" filter)
    df["atr_pct"]       = df["atr14"] / c.replace(0, np.nan) * 100.0
    df["atr_rank"]      = df["atr_pct"].rolling(100).rank(pct=True) * 100.0
    return df


def compute_thresholds(df_is, cids):
    out = {}
    for cid in cids:
        col, direction, q = COND_DEF[cid]
        vals = df_is[col].dropna()
        if len(vals) < 10:
            out[cid] = None
            continue
        out[cid] = float(vals.quantile(q))
    return out


def build_signal_mask(feats, cids, gate_mode="green", rel_vol_min=1.5):
    """Rolling-recalibrated condition mask AND entry gate.

    gate_mode "green"   : close>open & close>prev_close & rel_vol>min
    gate_mode "breakout": close>prev_high & rel_vol>min
    """
    c  = feats["close"].values
    o  = feats["open"].values
    h  = feats["high"].values
    rv = feats["rel_vol"].values
    n  = len(feats)

    sig = np.zeros(n, dtype=bool)
    cal = IS_LOOKBACK
    while cal < n:
        end = min(cal + RECAL_EVERY, n)
        thr = compute_thresholds(feats.iloc[cal - IS_LOOKBACK:cal], cids)
        m = np.ones(end - cal, dtype=bool)
        for cid in cids:
            col, direction, _ = COND_DEF[cid]
            vals = feats[col].values[cal:end]
            tq = thr.get(cid)
            if tq is None:
                m[:] = False
                break
            if direction == "lt_q":
                m &= vals < tq
            else:
                m &= vals > tq
        sig[cal:end] = m
        cal = end

    if gate_mode == "breakout":
        prev_high = np.roll(h, 1); prev_high[0] = np.nan
        gate = (c > prev_high) & (rv > rel_vol_min)
    else:
        gate = (c > o) & (c > np.roll(c, 1)) & (rv > rel_vol_min)
    gate[:1] = False
    gate = gate & ~np.isnan(c) & ~np.isnan(o) & ~np.isnan(rv)
    return sig & gate


def sim_symbol(feats, sig, rr, cfg):
    """Bot-faithful sequential simulation for one symbol.

    cfg keys:
      entry_next      bool   (False = enter at signal-bar close [E6])
      exit            str    "base" | "time24" | "timeN" | "partial" | "trail" | "trail_run"
      hours           tuple|None  UTC hour window for entries, e.g. (12,18)
      atr_rank_ceil   float|None  skip entries when atr_rank > ceil
      sl_mult         float  stop distance multiplier (default 1.0 ATR)
      trail_mult      float  trailing-stop distance for "trail_run" (ATR)
      time_bars       int|None  time stop for "timeN" / "trail_run" / "time24" override
    """
    c   = feats["close"].values
    h   = feats["high"].values
    l   = feats["low"].values
    atr = feats["atr14"].values
    if cfg.get("atr_rank_ceil") is not None:
        atr_r = feats["atr_rank"].values
    else:
        atr_r = None
    n = len(feats)
    entry_next = cfg.get("entry_next", False)
    exit_mode  = cfg.get("exit", "base")
    hours      = cfg.get("hours")
    ar_ceil    = cfg.get("atr_rank_ceil")
    sl_mult    = cfg.get("sl_mult", 1.0)
    trail_mult = cfg.get("trail_mult", 1.0)
    time_bars  = cfg.get("time_bars")

    trades = []
    pos = None

    def open_pos(i, entry, a):
        first_check = i + 1 if not entry_next else i + 2
        return dict(entry_i=i, first_check=first_check, entry=entry, atr=a,
                    sl=entry - sl_mult * a, tp=entry + rr * sl_mult * a,
                    be=False, partial=False, banked=0.0, trail_high=entry)

    for i in range(IS_LOOKBACK + 1, n):
        if pos is not None and i >= pos["first_check"]:
            e = pos["entry"]; a = pos["atr"]
            done = None
            if exit_mode == "time24":
                if h[i] >= pos["tp"]: done = (rr, "TP")
                elif l[i] <= pos["sl"]: done = (-1.0, "SL")
                elif i - pos["entry_i"] >= 24: done = ((c[i] - e) / a, "TIME")
            elif exit_mode == "timeN":
                if h[i] >= pos["tp"]: done = (rr, "TP")
                elif l[i] <= pos["sl"]: done = (-1.0, "SL")
                elif time_bars and i - pos["entry_i"] >= time_bars:
                    done = ((c[i] - e) / a, "TIME")
            elif exit_mode == "partial":
                if not pos["partial"] and h[i] >= e + a:
                    pos["sl"] = e; pos["partial"] = True; pos["banked"] = 0.5
                if pos["partial"] and h[i] >= e + rr * a:
                    done = (0.5 + 0.5 * rr, "TP")
                elif l[i] <= pos["sl"]:
                    done = (pos["banked"] - (0.5 if pos["partial"] else 1.0), "SL")
            elif exit_mode == "trail":
                pos["trail_high"] = max(pos["trail_high"], h[i])
                pos["sl"] = max(pos["sl"], pos["trail_high"] - a)
                if h[i] >= pos["tp"]: done = (rr, "TP")
                elif l[i] <= pos["sl"]: done = ((pos["sl"] - e) / a, "SL")
            elif exit_mode == "trail_run":
                # let winners run: no TP cap, trail from highest close
                pos["trail_high"] = max(pos["trail_high"], h[i])
                pos["sl"] = max(pos["sl"], pos["trail_high"] - trail_mult * a)
                if l[i] <= pos["sl"]: done = ((pos["sl"] - e) / a, "SL")
                elif time_bars and i - pos["entry_i"] >= time_bars:
                    done = ((c[i] - e) / a, "TIME")
            else:  # base
                if h[i] >= pos["tp"]: done = (rr, "TP")
                elif l[i] <= pos["sl"]: done = (-1.0, "SL")
            if done is not None:
                r_mult, xtype = done
                trades.append(dict(entry_time=feats.index[pos["entry_i"]],
                                   r=r_mult, exit_type=xtype,
                                   bars_in=i - pos["entry_i"],
                                   atr=pos["atr"], entry=pos["entry"]))
                pos = None

        if pos is not None or not sig[i]:
            continue

        # entry filters
        if hours is not None:
            hr = feats.index[i].hour
            if not (hours[0] <= hr < hours[1]):
                continue
        if ar_ceil is not None:
            v = atr_r[i]
            if not math.isnan(v) and v > ar_ceil:
                continue
        if i + 1 >= n and entry_next:
            continue
        a = atr[i]
        if not (a > 0) or math.isnan(a):
            continue
        entry = c[i + 1] if entry_next else c[i]
        pos = open_pos(i, entry, a)

    if pos is not None:  # mark-to-market at data end
        e = pos["entry"]; a = pos["atr"]; last_c = c[-1]
        if pos["partial"]:
            r_mm = 0.5 + 0.5 * (last_c - e) / a
        elif pos["be"]:
            r_mm = max((last_c - e) / a, 0.0)
        else:
            r_mm = (last_c - e) / a
        trades.append(dict(entry_time=feats.index[pos["entry_i"]],
                           r=r_mm, exit_type="OPEN", bars_in=n - 1 - pos["entry_i"],
                           atr=pos["atr"], entry=pos["entry"]))
    return trades


def run_family(cids, rr, cfg, feats_by_sym, masks):
    all_t = []
    for sym, feats in feats_by_sym.items():
        try:
            for t in sim_symbol(feats, masks[sym], rr, cfg):
                t["sym"] = sym
                all_t.append(t)
        except Exception:
            pass
    all_t.sort(key=lambda t: t["entry_time"])
    return all_t


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def max_dd_from_eq(eq):
    eq = np.asarray(eq, dtype=float)
    if len(eq) == 0: return 0.0
    pk = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(pk > 0, (eq - pk) / pk, 0.0)
    return float(dd.min())


def stats_from_trades(trades, start_cap=STARTING_CAP, risk_pct=RISK_PCT):
    n = len(trades)
    if n == 0:
        return dict(n=0, wr=float("nan"), pf=float("nan"), exp=0.0, mdd=0.0,
                    total_r=0.0, calmar=0.0, net=0.0)
    cap = start_cap
    eq = [cap]
    for t in trades:
        cap += cap * risk_pct * t["r"]
        eq.append(cap)
    eq = np.array(eq)
    rs = np.array([t["r"] for t in trades])
    wins = rs[rs > 0]; losses = rs[rs < 0]
    gw, gl = wins.sum(), abs(losses.sum())
    pf = gw / gl if gl > 0 else (999.0 if gw > 0 else 1.0)
    mdd = max_dd_from_eq(eq)
    net = eq[-1] - start_cap
    calmar = (net / start_cap) / abs(mdd) if mdd < 0 else 0.0
    return dict(n=n, wr=float((rs > 0).mean()), pf=pf,
                exp=float(np.mean(rs) * risk_pct * start_cap),
                mdd=mdd, total_r=float(rs.sum()), calmar=calmar, net=net)


def bootstrap_pf(rs, n_boot=2_000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    rs = np.asarray(rs, dtype=float)
    if len(rs) == 0: return (np.nan,)*3
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = rs[rng.integers(0, len(rs), len(rs))]
        w = s[s > 0].sum(); lo = abs(s[s < 0].sum())
        out[b] = w / lo if lo > 0 else (999.0 if w > 0 else 1.0)
    return (float(np.percentile(out, 5)), float(np.median(out)), float(np.percentile(out, 95)))


def bootstrap_pf_diff(trades_var, trades_base, n_boot=2_000, rng=None):
    """Paired bootstrap of (PF_var - PF_base) aligned by (sym, entry_time)."""
    if rng is None:
        rng = np.random.default_rng(42)
    base_idx = {}
    for j, t in enumerate(trades_base):
        base_idx[(t["sym"], t["entry_time"])] = j
    pairs = []
    for k, t in enumerate(trades_var):
        j = base_idx.get((t["sym"], t["entry_time"]))
        if j is not None:
            pairs.append((k, j))
    if len(pairs) < 50:
        return (np.nan, np.nan, np.nan)
    a = np.array([trades_var[k]["r"] for k, _ in pairs])
    b = np.array([trades_base[j]["r"] for _, j in pairs])
    n = len(pairs)
    out = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        def pf(x):
            s = x[idx]; w = s[s > 0].sum(); lo = abs(s[s < 0].sum())
            return w / lo if lo > 0 else (999.0 if w > 0 else 1.0)
        out[k] = pf(a) - pf(b)
    return (float(np.percentile(out, 5)), float(np.median(out)), float(np.percentile(out, 95)))


def loo_symbol_floor(trades):
    floor, rm = 999.0, None
    for sym in set(t["sym"] for t in trades):
        others = [t for t in trades if t["sym"] != sym]
        s = stats_from_trades(others)
        if s["pf"] < floor: floor, rm = s["pf"], sym
    return floor, rm


def monte_carlo(rs, n_paths=5_000, rng=None, start_cap=STARTING_CAP, risk_pct=RISK_PCT):
    if rng is None:
        rng = np.random.default_rng(42)
    rs = np.asarray(rs, dtype=float)
    if len(rs) == 0: return dict(prob=float("nan"), dd_p5=float("nan"), dd_p95=float("nan"), exp=float("nan"))
    profits = np.empty(n_paths); dds = np.empty(n_paths)
    for k in range(n_paths):
        s = rs[rng.integers(0, len(rs), len(rs))]
        cap = start_cap
        eq = np.empty(len(s) + 1); eq[0] = cap
        for j, r in enumerate(s):
            cap += cap * risk_pct * r
            eq[j + 1] = cap
        profits[k] = cap
        dds[k] = max_dd_from_eq(eq)
    return dict(prob=float((profits > start_cap).mean()),
                dd_p5=float(np.percentile(dds, 5)),
                dd_p95=float(np.percentile(dds, 95)),
                exp=float(profits.mean() - start_cap))


def cost_adjusted_rs(trades, cost_pct):
    """cost_pct = % of price per side (0.10 -> 0.10%)."""
    out = []
    for t in trades:
        cost_r = 2.0 * (cost_pct / 100.0) * t["entry"] / t["atr"]
        out.append(t["r"] - cost_r)
    return np.array(out)


def pf_of_rs(rs):
    rs = np.asarray(rs, dtype=float)
    if len(rs) == 0: return float("nan")
    w = rs[rs > 0].sum(); lo = abs(rs[rs < 0].sum())
    return w / lo if lo > 0 else (999.0 if w > 0 else 1.0)
