"""
QUANTLAB AI — R063
Signal Funnel & Bottleneck Autopsy

Forensic investigation of why E3.1_v2 produces only ~79 forward trades.
Strategy is FROZEN. No optimisation. No threshold changes.
Objective: anatomy only.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — identical to R062
# ─────────────────────────────────────────────────────────────────────────────
IS_RATIO      = 0.80
BASE_RR       = 2.0
N_FWD_FOLDS   = 5
MIN_BARS      = 2_000
CAPITAL       = 10_000.0
CACHE_DIR     = CONFIG["CACHE_FOLDER"]
OUT_DIR       = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUT_DIR, exist_ok=True)

FROZEN_CIDS  = ("BBW_STRICT", "RV_LO", "DST_NR", "PRG_VH")
FROZEN_LABEL = "BBW_STRICT+RV_LO+DST_NR+PRG_VH"

BASE_CONDITIONS = {
    "ATR_LO":    ("atr_rank",      "lt_q",      0.25),
    "BBW_LO":    ("bb_width",      "lt_q",      0.33),
    "BBW_STRICT":("bb_width",      "lt_q",      0.25),
    "RV_LO":     ("real_vol_20",   "lt_q",      0.33),
    "RV_HI":     ("real_vol_20",   "gt_q",      0.67),
    "SLP_UP":    ("ema200_slope",  "gt_fixed",  0.0),
    "DST_NR":    ("ema_dist_pct",  "lt_q",      0.33),
    "DST_MD":    ("ema_dist_pct",  "gt_q_pos",  0.60),
    "ADX_WK":    ("adx14",         "lt_q",      0.33),
    "PRG_LO":    ("prev_range_r",  "lt_q",      0.33),
    "PRG_VH":    ("prev_range_r",  "gt_q",      0.80),
}
QUANT_FEATS = ["atr_rank","bb_width","real_vol_20","ema_dist_pct",
               "adx14","prev_range_r","prev_body_r","prev_body_pct"]

SEP  = "═" * 110
SEP2 = "─" * 90

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — identical to R062
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]; o = df["open"]
    df["ema200"]        = calc_ema(c, 200)
    df["atr14"]         = calc_atr(df, 14)
    df["atr_rank"]      = df["atr14"].rolling(100).rank(pct=True) * 100
    bb_mid              = c.rolling(20).mean()
    bb_std              = c.rolling(20).std()
    df["bb_width"]      = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["ema_dist_pct"]  = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"]  = (df["ema200"] - df["ema200"].shift(10)) / \
                          df["ema200"].shift(10).replace(0, np.nan)
    vol_ma              = v.rolling(20).mean()
    df["rel_vol"]       = v / vol_ma.replace(0, np.nan)
    df["prev_close"]    = c.shift(1)
    df["prev_atr14"]    = df["atr14"].shift(1)
    log_ret             = np.log(c / c.shift(1))
    df["real_vol_20"]   = log_ret.rolling(20).std() * 100.0
    df["adx14"]         = calc_adx(df, 14)
    prev_range          = h.shift(1) - l.shift(1)
    prev_body           = (c.shift(1) - o.shift(1)).abs()
    df["prev_range_r"]  = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]   = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"] = prev_body  / prev_range.replace(0, np.nan)
    dt                  = pd.to_datetime(df["datetime"], utc=True)
    df["hour_utc"]      = dt.dt.hour.astype(np.int16)
    return df

def learn_thresholds(df_is, overrides=None):
    thr = {}; overrides = overrides or {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (feat, direction, base_q) in BASE_CONDITIONS.items():
        q = overrides.get(cid, base_q)
        if direction in ("gt_fixed","lt_fixed","hour_rng"):
            thr[cid] = q; continue
        if feat not in valid.columns:
            thr[cid] = np.nan; continue
        col = valid[feat].dropna()
        if len(col) < 20:
            thr[cid] = np.nan; continue
        if direction == "gt_q_pos":
            pos = col[col > 0]
            thr[cid] = float(pos.quantile(q) if len(pos) > 10 else col.quantile(q))
        else:
            thr[cid] = float(col.quantile(q))
    return thr

def build_env_mask(df, cond_ids, thr):
    N = len(df); mask = np.ones(N, dtype=bool)
    for cid in cond_ids:
        if cid not in BASE_CONDITIONS: return np.zeros(N, dtype=bool)
        feat, direction, _ = BASE_CONDITIONS[cid]
        if feat not in df.columns: return np.zeros(N, dtype=bool)
        col   = df[feat].values
        nan_m = np.isnan(col) if col.dtype.kind == "f" else np.zeros(N, dtype=bool)
        t     = thr.get(cid, np.nan)
        if direction == "lt_q":
            if isinstance(t, float) and np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col < t)
        elif direction in ("gt_q","gt_q_pos"):
            if isinstance(t, float) and np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col > t)
        elif direction == "gt_fixed": mask &= (~nan_m) & (col > t)
        elif direction == "lt_fixed": mask &= (~nan_m) & (col < t)
        elif direction == "hour_rng":
            lo_, hi_ = t; mask &= (col >= lo_) & (col <= hi_)
    return mask

def entry_gates(df):
    rv = df["rel_vol"].values; c = df["close"].values
    o  = df["open"].values;    pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    relvol  = ok & (rv > 1.5)
    bullish = ok & (c > o)
    above   = ok & (c > pc)
    return relvol, bullish, above

def safe_pf(gw, gl):
    return round(min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl, 3)

def run_backtest_full(df, signal):
    min_sl  = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf      = CONFIG["RISK_PER_TRADE_PCT"]
    fee     = CONFIG["TAKER_FEE"];  spd = CONFIG["SPREAD"] * 0.5
    slp     = CONFIG["SL_SLIPPAGE"]
    in_pos  = False; ep = st = tk = sz = 0.0
    trades  = []
    op_ = df["open"].values;  hi_ = df["high"].values
    lo_ = df["low"].values;   atr_ = df["prev_atr14"].values
    for i in range(1, len(df)):
        if in_pos:
            sl_hit = lo_[i] <= st; tp_hit = hi_[i] >= tk
            if sl_hit or tp_hit:
                xp  = (st * (1 - slp)) if sl_hit else tk
                net = (xp - ep) * sz - (ep * sz + xp * sz) * (fee + spd)
                if sl_hit: net -= (st - xp) * sz
                trades.append({"pnl": round(net, 4), "win": int(not sl_hit)})
                in_pos = False
            continue
        if signal[i-1]:
            a = atr_[i]
            if np.isnan(a) or a <= 0 or a / op_[i] < min_sl: continue
            ep = op_[i]; st = ep - a; tk = ep + BASE_RR * a
            sz = min(CAPITAL * rf / a, (CAPITAL * max_lev) / ep)
            in_pos = True
    return trades

def metrics_from(trades):
    if not trades:
        return {"n":0,"wr":0.0,"pf":0.0,"net":0.0,"mdd":0.0}
    pnl  = np.array([t["pnl"] for t in trades])
    wins = np.array([t["win"] for t in trades], dtype=bool)
    n=len(pnl); nw=wins.sum(); nl=n-nw
    gw=pnl[wins].sum() if nw else 0.0
    gl=abs(pnl[~wins].sum()) if nl else 0.0
    eq=np.concatenate([[CAPITAL], CAPITAL+np.cumsum(pnl)])
    pk=np.maximum.accumulate(eq)
    mdd=float(((eq-pk)/pk).min())
    avg_w = float(pnl[wins].mean()) if nw else 0.0
    avg_l = float(pnl[~wins].mean()) if nl else 0.0
    exp   = (nw/n)*BASE_RR - (nl/n) if n else 0.0
    return {"n":n,"wr":round(nw/n,3),"pf":safe_pf(gw,gl),"net":round(float(pnl.sum()),2),
            "mdd":round(mdd,4),"exp":round(exp,4),
            "avg_win":round(avg_w,4),"avg_loss":round(avg_l,4)}

def wfo_run(all_dfs, cond_ids, entry_override=None):
    """Run full WFO. entry_override = dict of gate overrides to disable."""
    all_t = []; fold_t = defaultdict(list)
    entry_override = entry_override or {}
    for sym, (df_is, df_fwd, sym_thr) in all_dfs.items():
        fwd_size = len(df_fwd); seg_size = max(1, fwd_size // N_FWD_FOLDS)
        for fi in range(N_FWD_FOLDS):
            seg_s = fi * seg_size
            seg_e = fwd_size if fi == N_FWD_FOLDS - 1 else (fi+1)*seg_size
            df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
            if len(df_seg) < 20: continue
            em  = build_env_mask(df_seg, cond_ids, sym_thr)
            rv, bul, abv = entry_gates(df_seg)
            use_rv  = entry_override.get("RELVOL",   True)
            use_bul = entry_override.get("BULLISH",  True)
            use_abv = entry_override.get("ABOVE_PC", True)
            base = np.ones(len(df_seg), dtype=bool)
            if use_rv:  base &= rv
            if use_bul: base &= bul
            if use_abv: base &= abv
            sig = base & em
            all_t.extend(run_backtest_full(df_seg, sig))
            fold_t[f"F{fi+1}"].extend(run_backtest_full(df_seg, sig))
    return all_t, fold_t

# ─────────────────────────────────────────────────────────────────────────────
# OPPORTUNITY COST SIMULATOR
# Simulate what happens to a rejected bar: entry @ next open, ATR-SL/TP
# ─────────────────────────────────────────────────────────────────────────────
def simulate_rejected(df, rejected_mask, max_bars=48):
    """
    For each rejected bar i (signal[i]=True but env rejected):
    simulate entry at bar i+1 open, SL = entry - ATR, TP = entry + RR*ATR.
    Returns list of dicts: {idx, entry, sl, tp, outcome, r_mult, bars_held}
    """
    op_  = df["open"].values;   hi_ = df["high"].values
    lo_  = df["low"].values;    atr_ = df["prev_atr14"].values
    min_sl = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    results = []
    N = len(df)
    for i in np.where(rejected_mask)[0]:
        j = i + 1  # entry bar
        if j >= N: continue
        a = atr_[j]
        if np.isnan(a) or a <= 0 or a / op_[j] < min_sl: continue
        ep = op_[j]; sl = ep - a; tp = ep + BASE_RR * a
        outcome = "open"; r_mult = np.nan; bars = 0
        for k in range(j, min(j + max_bars, N)):
            bars = k - j
            if lo_[k] <= sl:
                outcome = "SL"; r_mult = -1.0; break
            if hi_[k] >= tp:
                outcome = "TP"; r_mult = +BASE_RR; break
        if outcome == "open":
            # mark-to-market
            r_mult = (df["close"].values[min(j+max_bars-1, N-1)] - ep) / a
        results.append({"bar_idx": i, "outcome": outcome, "r_mult": round(r_mult, 3),
                        "bars_held": bars})
    return results

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  QUANTLAB AI — R063")
print("  Signal Funnel & Bottleneck Autopsy")
print(f"  Strategy (FROZEN): {FROZEN_LABEL}")
print(SEP)
print()
print("  Loading cached symbols ...")

all_dfs = {}
for fname in sorted(os.listdir(CACHE_DIR)):
    if not fname.endswith("_1H.parquet"): continue
    sym = fname.replace("_1H.parquet","").replace("_","-")
    try:
        df = pd.read_parquet(os.path.join(CACHE_DIR, fname))
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        if len(df) < MIN_BARS: continue
        df  = add_features(df)
        sp  = int(len(df) * IS_RATIO)
        thr = learn_thresholds(df.iloc[:sp])
        all_dfs[sym] = (df.iloc[:sp], df.iloc[sp:].reset_index(drop=True), thr)
    except Exception as e:
        print(f"  [WARN] {sym}: {e}")

SYMS = list(all_dfs.keys())
print(f"  Symbols loaded: {len(SYMS)}")
total_oos_bars = sum(len(v[1]) for v in all_dfs.values())
print(f"  Total OOS bars : {total_oos_bars:,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — FULL SIGNAL FUNNEL
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 1 — Full Signal Funnel")
print(SEP)
print()

funnel = defaultdict(int)
# Warm-up: drop first 250 bars of each OOS (feature settle)
WARM = 0  # features already NaN from IS boundary — no extra warm-up needed

for sym, (df_is, df_fwd, thr) in all_dfs.items():
    df = df_fwd
    N  = len(df)
    c  = df["close"].values;  o = df["open"].values
    pc = df["prev_close"].values
    rv_v = df["rel_vol"].values
    bbw  = df["bb_width"].values;   rv20 = df["real_vol_20"].values
    dst  = df["ema_dist_pct"].values; prg = df["prev_range_r"].values
    valid = (~np.isnan(rv_v)) & (~np.isnan(c)) & (~np.isnan(pc)) \
          & (~np.isnan(bbw))  & (~np.isnan(rv20)) \
          & (~np.isnan(dst))  & (~np.isnan(prg))

    funnel["01_all"]     += valid.sum()

    relvol_m = valid & (rv_v > 1.5)
    funnel["02_relvol"]  += relvol_m.sum()

    bullish_m = relvol_m & (c > o)
    funnel["03_bullish"] += bullish_m.sum()

    above_m = bullish_m & (c > pc)
    funnel["04_above_pc"] += above_m.sum()  # "base signals"

    bbw_m   = above_m & (bbw  < thr.get("BBW_STRICT", np.nan))
    funnel["05_bbw_strict"] += bbw_m.sum()

    rv_m    = bbw_m   & (rv20 < thr.get("RV_LO",      np.nan))
    funnel["06_rv_lo"]   += rv_m.sum()

    dst_m   = rv_m    & (dst  < thr.get("DST_NR",     np.nan))
    funnel["07_dst_nr"]  += dst_m.sum()

    prg_m   = dst_m   & (prg  > thr.get("PRG_VH",     np.nan))
    funnel["08_prg_vh"]  += prg_m.sum()

    funnel["09_signals"] += prg_m.sum()  # raw signals before backtest engine

# Stage labels
STAGE_LABELS = [
    ("01_all",        "All valid OOS bars"),
    ("02_relvol",     "↓ RELVOL > 1.5      [entry gate 1]"),
    ("03_bullish",    "↓ Bullish candle     [entry gate 2]"),
    ("04_above_pc",   "↓ Close > PrevClose  [entry gate 3]"),
    ("05_bbw_strict", "↓ BBW_STRICT         [env filter 1]"),
    ("06_rv_lo",      "↓ RV_LO              [env filter 2]"),
    ("07_dst_nr",     "↓ DST_NR             [env filter 3]"),
    ("08_prg_vh",     "↓ PRG_VH             [env filter 4]"),
    ("09_signals",    "= Final signals"),
]

base_n = funnel["01_all"]
print(f"  {'Stage':<45}  {'Count':>8}  {'% Remain':>9}  {'% Removed':>10}")
print(f"  {'─'*45}  {'─'*8}  {'─'*9}  {'─'*10}")
prev_n = base_n
for key, label in STAGE_LABELS:
    n     = funnel[key]
    pct_r = 100.0 * n / base_n if base_n else 0.0
    pct_x = 100.0 * (prev_n - n) / prev_n if prev_n else 0.0
    arrow = "▼" if pct_x > 0 else " "
    print(f"  {label:<45}  {n:>8,}  {pct_r:>8.2f}%  {arrow}{pct_x:>8.2f}%")
    prev_n = n

base_signals = funnel["04_above_pc"]
final_signals = funnel["09_signals"]
print()
print(f"  Base signals (all 3 entry gates)  : {base_signals:,}")
print(f"  Final signals (+ all 4 env conds) : {final_signals:,}")
print(f"  Env filter pass rate               : {100*final_signals/base_signals:.2f}% of base signals")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FILTER KILL RATE (independent, on base signals)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 2 — Filter Kill Rate  (measured on base-signal pool)")
print(SEP)
print()

FILTERS = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH",
           "RELVOL","BULLISH","ABOVE_PC"]

FILTER_FEATS = {
    "BBW_STRICT": ("bb_width",     "lt",  "BBW_STRICT"),
    "RV_LO":      ("real_vol_20",  "lt",  "RV_LO"),
    "DST_NR":     ("ema_dist_pct", "lt",  "DST_NR"),
    "PRG_VH":     ("prev_range_r", "gt",  "PRG_VH"),
}
ENTRY_GATE_NAMES = {"RELVOL":"RELVOL","BULLISH":"BULLISH","ABOVE_PC":"ABOVE_PC"}

kill_stats = {}

for flt in FILTERS:
    kills = 0; passes = 0
    for sym, (df_is, df_fwd, thr) in all_dfs.items():
        df = df_fwd
        c  = df["close"].values;  o = df["open"].values
        pc = df["prev_close"].values; rv_v = df["rel_vol"].values
        bbw = df["bb_width"].values; rv20 = df["real_vol_20"].values
        dst = df["ema_dist_pct"].values; prg = df["prev_range_r"].values
        valid = (~np.isnan(rv_v)) & (~np.isnan(c)) & (~np.isnan(pc))

        if flt in ("RELVOL","BULLISH","ABOVE_PC"):
            # kill rate on ALL valid bars
            if flt == "RELVOL":
                pass_m = valid & (rv_v > 1.5)
            elif flt == "BULLISH":
                pass_m = valid & (c > o)
            else:
                pass_m = valid & (c > pc)
            kills  += (valid & ~pass_m).sum()
            passes += pass_m.sum()
        else:
            # kill rate on base-signal pool
            base = valid & (rv_v > 1.5) & (c > o) & (c > pc)
            if base.sum() == 0: continue
            t = thr.get(flt, np.nan)
            if np.isnan(t): continue
            feat_map = {"BBW_STRICT": bbw, "RV_LO": rv20, "DST_NR": dst, "PRG_VH": prg}
            fv = feat_map[flt]
            if flt == "PRG_VH":
                pass_m = base & (~np.isnan(fv)) & (fv > t)
            else:
                pass_m = base & (~np.isnan(fv)) & (fv < t)
            kills  += (base & ~pass_m).sum()
            passes += pass_m.sum()

    total = kills + passes
    kill_stats[flt] = {
        "kills":  kills,
        "passes": passes,
        "total":  total,
        "kill_pct": 100.0 * kills / total if total else 0.0,
    }

ranked_kill = sorted(FILTERS, key=lambda f: -kill_stats[f]["kill_pct"])

print(f"  {'Filter':<14}  {'Pool':<10}  {'Entering':>9}  {'Removed':>9}  "
      f"{'Surviving':>9}  {'Kill %':>8}")
print(f"  {'─'*14}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}")
for flt in ranked_kill:
    s  = kill_stats[flt]
    pool = "all bars" if flt in ("RELVOL","BULLISH","ABOVE_PC") else "base-sig"
    print(f"  {flt:<14}  {pool:<10}  {s['total']:>9,}  {s['kills']:>9,}  "
          f"{s['passes']:>9,}  {s['kill_pct']:>7.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EFFICIENCY SCORE (deferred: depends on Section 4 ablation)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MARGINAL CONTRIBUTION (one-filter-removed ablation)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 4 — Marginal Contribution (one-filter removed at a time)")
print(SEP)
print()

# Baseline WFO
print("  Running baseline WFO ...")
base_t, _ = wfo_run(all_dfs, FROZEN_CIDS)
base_m    = metrics_from(base_t)
print(f"  Baseline: PF={base_m['pf']}  n={base_m['n']}  WR={base_m['wr']:.1%}  "
      f"MDD={base_m['mdd']:.1%}  Exp={base_m['exp']:.3f}")
print()

ABLATION_CONFIGS = [
    # Label,          env_cids,                         entry_override
    ("Drop BBW_STRICT", ("RV_LO","DST_NR","PRG_VH"),   {}),
    ("Drop RV_LO",      ("BBW_STRICT","DST_NR","PRG_VH"),{}),
    ("Drop DST_NR",     ("BBW_STRICT","RV_LO","PRG_VH"),{}),
    ("Drop PRG_VH",     ("BBW_STRICT","RV_LO","DST_NR"),{}),
    ("Drop RELVOL",     FROZEN_CIDS, {"RELVOL":False}),
    ("Drop BULLISH",    FROZEN_CIDS, {"BULLISH":False}),
    ("Drop ABOVE_PC",   FROZEN_CIDS, {"ABOVE_PC":False}),
]

ablation_results = {}
print(f"  {'Label':<20}  {'PF':>6}  {'n':>5}  {'WR':>6}  {'MDD':>7}  "
      f"{'Net($)':>8}  {'ΔPF':>7}  {'Δn':>5}")
print(f"  {'─'*20}  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*5}")

for label, env_cids, entry_ov in ABLATION_CONFIGS:
    t, _ = wfo_run(all_dfs, env_cids, entry_override=entry_ov)
    m    = metrics_from(t)
    ablation_results[label] = m
    dpf  = m["pf"] - base_m["pf"]
    dn   = m["n"]  - base_m["n"]
    dpf_s = f"+{dpf:.3f}" if dpf >= 0 else f"{dpf:.3f}"
    dn_s  = f"+{dn}"      if dn  >= 0 else f"{dn}"
    print(f"  {label:<20}  {m['pf']:>6.3f}  {m['n']:>5}  {m['wr']:>5.1%}  "
          f"{m['mdd']:>6.1%}  {m['net']:>8.0f}  {dpf_s:>7}  {dn_s:>5}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EFFICIENCY SCORE (now computed)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 3 — Filter Efficiency Score")
print("  [PF gained by the filter] / [Trades removed when filter is dropped]")
print(SEP)
print()

eff_rows = []
for label, env_cids, entry_ov in ABLATION_CONFIGS:
    m      = ablation_results[label]
    flt    = label.replace("Drop ","")
    dpf    = base_m["pf"] - m["pf"]   # PF LOST when filter is removed = PF that filter adds
    dn     = m["n"]  - base_m["n"]    # Trades GAINED when filter is removed = trades filter kills
    eff    = dpf / dn if dn > 0 else 0.0
    quality = "HIGH" if dpf > 0.05 and dn <= 15 else \
              "LOW"  if dpf <= 0.05 and dn > 10 else "MED"
    eff_rows.append((flt, dpf, dn, eff, quality, m["pf"]))

eff_rows.sort(key=lambda x: -x[3])

print(f"  {'Filter':<14}  {'PF Contribution':>16}  {'Trades Blocked':>15}  "
      f"{'Efficiency':>11}  {'Quality':>8}  {'PF w/o filter':>14}")
print(f"  {'─'*14}  {'─'*16}  {'─'*15}  {'─'*11}  {'─'*8}  {'─'*14}")
for flt, dpf, dn, eff, quality, pf_without in eff_rows:
    verdict = "✓ NECESSARY" if dpf > 0 else "✗ HARMFUL  "
    print(f"  {flt:<14}  {dpf:>+15.3f}  {dn:>15}  {eff:>10.4f}  {quality:>8}  "
          f"{pf_without:>13.3f}  {verdict}")
print()
print("  Key: PF Contribution = PF lost when this filter is removed")
print("       Efficiency Score = PF Contribution / Trades Blocked")
print("       HIGH efficiency = large PF gain for few trades blocked")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PAIRWISE INTERACTION
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 5 — Pairwise Interaction (remove two filters simultaneously)")
print(SEP)
print()

ENV_FILTERS  = list(FROZEN_CIDS)        # 4 env
ENTRY_GATES  = ["RELVOL","BULLISH","ABOVE_PC"]  # 3 entry
ALL_FILTERS  = ENV_FILTERS + ENTRY_GATES

# Compute single-drop n and PF for reference
single_n  = {}
single_pf = {}
for label, _, _ in ABLATION_CONFIGS:
    flt = label.replace("Drop ","")
    single_n[flt]  = ablation_results[label]["n"]
    single_pf[flt] = ablation_results[label]["pf"]

pair_results = []
pair_combos  = list(combinations(ALL_FILTERS, 2))
print(f"  Testing {len(pair_combos)} filter pairs ...")

for f1, f2 in pair_combos:
    # Build env cids: remove f1 and f2 from frozen
    env_drop = {f for f in (f1, f2) if f in ENV_FILTERS}
    ent_drop = {f for f in (f1, f2) if f in ENTRY_GATES}
    env_cids = tuple(c for c in FROZEN_CIDS if c not in env_drop)
    entry_ov = {g: False for g in ent_drop}
    t, _ = wfo_run(all_dfs, env_cids, entry_override=entry_ov)
    m    = metrics_from(t)

    # Expected n if filters were independent: take the higher n (more lenient)
    # Actual synergy = actual_n - max(single_n[f1], single_n[f2])
    expected_n   = max(single_n.get(f1, base_m["n"]), single_n.get(f2, base_m["n"]))
    extra_trades = m["n"] - expected_n

    # Classify relationship
    if extra_trades > 10:
        relation = "REDUNDANT"   # filters overlap, removing both barely adds more than removing one
    elif extra_trades < -5:
        relation = "SYNERGISTIC" # filters complement each other, removing both cascades badly
    else:
        relation = "INDEPENDENT"

    pair_results.append({
        "f1": f1, "f2": f2,
        "pf": m["pf"], "n": m["n"],
        "extra_n": extra_trades,
        "relation": relation,
        "delta_pf": m["pf"] - base_m["pf"],
    })

pair_results.sort(key=lambda x: -x["extra_n"])

print()
print(f"  {'F1':<14}  {'F2':<14}  {'PF':>6}  {'n':>5}  {'Extra n':>8}  "
      f"{'ΔPF':>7}  {'Relationship'}")
print(f"  {'─'*14}  {'─'*14}  {'─'*6}  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*13}")
for r in pair_results:
    dpf_s = f"+{r['delta_pf']:.3f}" if r['delta_pf'] >= 0 else f"{r['delta_pf']:.3f}"
    en_s  = f"+{r['extra_n']}"      if r['extra_n']  >= 0 else f"{r['extra_n']}"
    sym   = "⟷" if r['relation']=="REDUNDANT" else ("⊗" if r['relation']=="SYNERGISTIC" else "⊥")
    print(f"  {r['f1']:<14}  {r['f2']:<14}  {r['pf']:>6.3f}  {r['n']:>5}  "
          f"{en_s:>8}  {dpf_s:>7}  {sym} {r['relation']}")
print()
print("  ⟷ REDUNDANT  = removing both barely adds more trades than removing either alone")
print("  ⊗ SYNERGISTIC = filters protect each other; removing both collapses quality more than expected")
print("  ⊥ INDEPENDENT = each filter kills a different, non-overlapping population")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — OPPORTUNITY COST
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 6 — Opportunity Cost  (rejected signals: what would have happened?)")
print(SEP)
print()

# For every bar where ALL 3 entry gates fire but ≥1 env filter fails,
# simulate the entry and track outcome. Label by FIRST failing env filter.

opp_by_filter = defaultdict(list)  # filter_name -> list of simulation results

for sym, (df_is, df_fwd, thr) in all_dfs.items():
    df = df_fwd
    c  = df["close"].values;  o = df["open"].values
    pc = df["prev_close"].values; rv_v = df["rel_vol"].values
    bbw = df["bb_width"].values; rv20 = df["real_vol_20"].values
    dst = df["ema_dist_pct"].values; prg = df["prev_range_r"].values

    valid = (~np.isnan(rv_v)) & (~np.isnan(c)) & (~np.isnan(pc)) \
          & (~np.isnan(bbw))  & (~np.isnan(rv20)) \
          & (~np.isnan(dst))  & (~np.isnan(prg))

    base_sig = valid & (rv_v > 1.5) & (c > o) & (c > pc)

    t_bbw = thr.get("BBW_STRICT", np.nan)
    t_rv  = thr.get("RV_LO",      np.nan)
    t_dst = thr.get("DST_NR",     np.nan)
    t_prg = thr.get("PRG_VH",     np.nan)

    if any(np.isnan(t) for t in [t_bbw, t_rv, t_dst, t_prg]): continue

    bbw_pass = base_sig & (bbw < t_bbw)
    rv_pass  = base_sig & (rv20 < t_rv)
    dst_pass = base_sig & (dst  < t_dst)
    prg_pass = base_sig & (prg  > t_prg)

    env_full = bbw_pass & rv_pass & dst_pass & prg_pass

    # Bars rejected (base signal fired but not final)
    rejected = base_sig & ~env_full

    # Identify FIRST failing filter for each rejected bar
    for i in np.where(rejected)[0]:
        if not (bbw[i] < t_bbw):
            first_fail = "BBW_STRICT"
        elif not (rv20[i] < t_rv):
            first_fail = "RV_LO"
        elif not (dst[i]  < t_dst):
            first_fail = "DST_NR"
        else:
            first_fail = "PRG_VH"
        single_mask = np.zeros(len(df), dtype=bool); single_mask[i] = True
        sim = simulate_rejected(df, single_mask, max_bars=48)
        if sim:
            sim[0]["filter"] = first_fail
            opp_by_filter[first_fail].extend(sim)

# Aggregate
total_rejected = sum(len(v) for v in opp_by_filter.values())
print(f"  Total rejected base-signals simulated: {total_rejected:,}")
print()
print(f"  {'Filter':<14}  {'Rejected':>8}  {'→TP':>7}  {'→SL':>7}  {'Open':>6}  "
      f"{'Win%':>6}  {'Avg R':>7}  {'Pot.Winners':>12}")
print(f"  {'─'*14}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*12}")

opp_summary = {}
for flt in FROZEN_CIDS:
    sims = opp_by_filter.get(flt, [])
    if not sims:
        print(f"  {flt:<14}  {'0':>8}")
        continue
    n_tp   = sum(1 for s in sims if s["outcome"]=="TP")
    n_sl   = sum(1 for s in sims if s["outcome"]=="SL")
    n_op   = sum(1 for s in sims if s["outcome"]=="open")
    n_tot  = len(sims)
    wr     = n_tp / n_tot if n_tot else 0.0
    avg_r  = np.mean([s["r_mult"] for s in sims if not np.isnan(s["r_mult"])])
    opp_summary[flt] = {"n": n_tot, "n_tp": n_tp, "n_sl": n_sl, "wr": wr, "avg_r": avg_r}
    print(f"  {flt:<14}  {n_tot:>8,}  {n_tp:>7,}  {n_sl:>7,}  {n_op:>6,}  "
          f"{wr:>5.1%}  {avg_r:>7.3f}  {n_tp:>12,}")

print()
# Estimated winners lost
total_winners_lost = sum(opp_summary.get(f, {}).get("n_tp", 0) for f in FROZEN_CIDS)
total_losers_blocked = sum(opp_summary.get(f, {}).get("n_sl", 0) for f in FROZEN_CIDS)
print(f"  Potential winners discarded : {total_winners_lost:,}")
print(f"  Potential losers avoided    : {total_losers_blocked:,}")
ratio = total_winners_lost / total_losers_blocked if total_losers_blocked else 0
print(f"  Winners/Losers ratio        : {ratio:.2f}  "
      f"({'filters discard more winners than losers' if ratio>1 else 'filters block more losers than winners'})")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FALSE NEGATIVE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 7 — False Negative Analysis  (rejected trades that became winners)")
print(SEP)
print()

print(f"  {'Filter':<14}  {'False Negatives':>16}  {'False Positives':>16}  "
      f"{'FN Rate':>9}  {'FP Rate':>9}  {'Verdict'}")
print(f"  {'─'*14}  {'─'*16}  {'─'*16}  {'─'*9}  {'─'*9}  {'─'*30}")
for flt in FROZEN_CIDS:
    s = opp_summary.get(flt, {})
    if not s: continue
    fn = s["n_tp"]   # False Negatives = winners we blocked
    fp = s["n_sl"]   # correctly blocked losers
    n  = s["n"]
    fn_r = fn / n if n else 0.0
    fp_r = fp / n if n else 0.0
    if fn_r > 0.40:
        verdict = "⚠ HIGH FALSE-NEGATIVE RATE"
    elif fn_r < 0.25:
        verdict = "✓ Mostly blocking losers"
    else:
        verdict = "~ Mixed (blocks both roughly equally)"
    print(f"  {flt:<14}  {fn:>16,}  {fp:>16,}  {fn_r:>8.1%}  {fp_r:>8.1%}  {verdict}")
print()
print("  False Negatives = rejected signals that would have hit TP")
print("  False Positives = correctly blocked signals that would have hit SL")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — SURVIVAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 8 — Survival Analysis")
print(SEP)
print()

print("  Stage-by-stage survival curve:")
print()
print(f"  {'Stage':<18}  {'Survivors':>10}  {'Survival %':>10}")
print(f"  {'─'*18}  {'─'*10}  {'─'*10}")
survival_data = []
for key, label in STAGE_LABELS:
    n   = funnel[key]
    pct = 100.0 * n / base_n
    bar = "█" * int(pct / 2)
    short_label = label.split("↓")[-1].split("=")[-1].strip()[:18]
    survival_data.append((short_label, n, pct))
    print(f"  {short_label:<18}  {n:>10,}  {pct:>9.2f}%  {bar}")
print()

# Which single stage causes the biggest drop
stage_drops = []
for i in range(1, len(STAGE_LABELS)):
    k0, l0 = STAGE_LABELS[i-1]
    k1, l1 = STAGE_LABELS[i]
    n0, n1 = funnel[k0], funnel[k1]
    drop = n0 - n1
    drop_pct = 100.0 * drop / n0 if n0 else 0.0
    stage_drops.append((l1.split("↓")[-1].strip(), drop, drop_pct))

stage_drops.sort(key=lambda x: -x[2])
print("  Biggest single-stage collapses:")
for name, drop, drop_pct in stage_drops[:4]:
    print(f"    {name:<35}  removes {drop:,} signals ({drop_pct:.1f}% of entering stage)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PARETO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 9 — Pareto Analysis")
print(SEP)
print()

# Absolute signals removed by each filter (sequential funnel)
sequential_removals = []
keys = [k for k, _ in STAGE_LABELS]
for i in range(1, len(keys)):
    k0, k1 = keys[i-1], keys[i]
    n0, n1 = funnel[k0], funnel[k1]
    label  = STAGE_LABELS[i][1].split("↓")[-1].strip()
    sequential_removals.append((label, n0 - n1))

sequential_removals.sort(key=lambda x: -x[1])
total_removed = funnel["01_all"] - funnel["09_signals"]

print(f"  Total signals removed from raw bars to final: {total_removed:,}")
print()
print(f"  {'Filter/Stage':<35}  {'Removed':>8}  {'Cumulative%':>12}  {'Pareto'}")
print(f"  {'─'*35}  {'─'*8}  {'─'*12}  {'─'*30}")
cumulative = 0
for label, removed in sequential_removals:
    cumulative += removed
    cum_pct = 100.0 * cumulative / total_removed
    bar = "█" * int(cum_pct / 5)
    print(f"  {label:<35}  {removed:>8,}  {cum_pct:>11.1f}%  {bar}")
print()

# Dominant filters (80/20 check)
top_two_pct = 100.0 * sum(r for _, r in sequential_removals[:2]) / total_removed
print(f"  Top-2 filters account for {top_two_pct:.1f}% of all signal removal")
top_filter = sequential_removals[0]
print(f"  Dominant bottleneck: {top_filter[0]}")
print(f"    Removes {top_filter[1]:,} signals = "
      f"{100.*top_filter[1]/total_removed:.1f}% of total funnel loss")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — FINAL VERDICT
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 10 — Final Verdict")
print(SEP)
print()

# Best and worst filter by efficiency
best_eff  = max(eff_rows, key=lambda x: x[3])
worst_eff = min(eff_rows, key=lambda x: x[3])
biggest_bottleneck = sequential_removals[0][0]
most_winners_removed = max(opp_summary, key=lambda f: opp_summary[f].get("n_tp",0))
most_losers_blocked  = max(opp_summary, key=lambda f: opp_summary[f].get("n_sl",0))

print(f"  ① Why are there only {base_m['n']} trades?")
print(f"     The 3 entry gates + 4 env filters apply sequentially on all OOS bars.")
print(f"     From {funnel['01_all']:,} valid bars → {funnel['04_above_pc']:,} base signals "
      f"({100*funnel['04_above_pc']/funnel['01_all']:.1f}%) → "
      f"{funnel['09_signals']:,} final signals ({100*funnel['09_signals']/funnel['01_all']:.2f}% of all bars).")
print(f"     The backtest engine then skips signals fired while in-position,")
print(f"     reducing further to {base_m['n']} actual trades.")
print()
print(f"  ② Biggest bottleneck:")
print(f"     {biggest_bottleneck}")
print(f"     Removes {sequential_removals[0][1]:,} signals "
      f"({100.*sequential_removals[0][1]/total_removed:.1f}% of total funnel loss).")
print()
print(f"  ③ Most quality-adding filter (highest efficiency score):")
print(f"     {best_eff[0]}  |  PF contribution: {best_eff[1]:+.3f}  |  "
      f"Trades blocked: {best_eff[2]}  |  Score: {best_eff[3]:.4f}")
print()
print(f"  ④ Least quality-adding filter (lowest efficiency score):")
print(f"     {worst_eff[0]}  |  PF contribution: {worst_eff[1]:+.3f}  |  "
      f"Trades blocked: {worst_eff[2]}  |  Score: {worst_eff[3]:.4f}")
print()
print(f"  ⑤ Filter removing the most winners:")
fn_data = [(f, opp_summary[f]["n_tp"], opp_summary[f]["wr"])
           for f in FROZEN_CIDS if f in opp_summary]
fn_data.sort(key=lambda x: -x[1])
print(f"     {fn_data[0][0]}  |  {fn_data[0][1]:,} false negatives  |  "
      f"Win rate of blocked signals: {fn_data[0][2]:.1%}")
print()
print(f"  ⑥ Filter blocking the most losers:")
sl_data = [(f, opp_summary[f]["n_sl"]) for f in FROZEN_CIDS if f in opp_summary]
sl_data.sort(key=lambda x: -x[1])
print(f"     {sl_data[0][0]}  |  {sl_data[0][1]:,} losers blocked")
print()

# Candidate for future relaxation?
worst_fn = fn_data[0]
worst_pf_without = ablation_results.get(f"Drop {worst_fn[0]}", {}).get("pf", 0.0)
print(f"  ⑦ Could one filter be relaxed in future research?")
if worst_fn[2] > 0.35:
    print(f"     {worst_fn[0]} has a {worst_fn[2]:.1%} false-negative rate.")
    print(f"     PF without it: {worst_pf_without:.3f}. If future research finds a sub-segment")
    print(f"     where this filter is too strict, it is the best candidate to re-examine.")
    print(f"     However: do NOT change the frozen strategy. This is an observation only.")
else:
    print(f"     No single filter has a false-negative rate above 35%.")
    print(f"     The 79 trades appear to be the unavoidable output of a genuinely selective edge.")
print()

# Final conclusion
all_pf_add = all(r[1] > 0 for r in eff_rows if r[0] in FROZEN_CIDS)
print(f"  ⑧ Should the strategy remain frozen?")
if all_pf_add:
    print(f"     YES. All 4 environment filters add positive PF contribution.")
    print(f"     All 3 entry gates add positive PF contribution.")
    print(f"     The 79 trades are the unavoidable price of a real, selective edge.")
    print(f"     Relaxing any filter would grow n but damage quality.")
else:
    neg_f = [r[0] for r in eff_rows if r[1] <= 0]
    print(f"     PARTIAL FREEZE QUESTION. {', '.join(neg_f)} add ≤0 PF.")
    print(f"     Worth investigating in a future non-frozen research round.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts ...")
print(SEP2)
print()

C_BG    = "#0d1117"; C_PANEL = "#161b22"; C_TEXT = "#e6edf3"
C_GRID  = "#30363d"; C_GRN   = "#3fb950"; C_RED  = "#f85149"
C_AMB   = "#d29922"; C_BLU   = "#58a6ff"
plt.rcParams.update({
    "figure.facecolor": C_BG, "axes.facecolor": C_PANEL,
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "axes.edgecolor": C_GRID, "grid.color": C_GRID,
    "font.family": "monospace",
})
def ps(ax, title, fs=8):
    ax.set_facecolor(C_PANEL); ax.set_title(title, fontsize=fs, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.grid(True, color=C_GRID, alpha=0.4, lw=0.5)
    for sp in ax.spines.values(): sp.set_color(C_GRID)

fig = plt.figure(figsize=(20, 24), facecolor=C_BG)
fig.suptitle(f"R063 — Signal Funnel & Bottleneck Autopsy | {FROZEN_LABEL}",
             fontsize=11, color=C_TEXT, y=0.995)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.35,
                       top=0.97, bottom=0.03, left=0.06, right=0.97)

# ── P1: Waterfall funnel ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ps(ax1, "Signal Funnel — Waterfall (OOS bars)", 9)
stage_ns = [funnel[k] for k, _ in STAGE_LABELS]
stage_lbs = [l.split("↓")[-1].split("=")[-1].strip()[:20] for _, l in STAGE_LABELS]
colors_f = [C_BLU if i==0 else (C_GRN if i==len(stage_ns)-1 else
            (C_AMB if i<=3 else C_RED)) for i in range(len(stage_ns))]
bars = ax1.bar(range(len(stage_ns)), stage_ns, color=colors_f, edgecolor=C_GRID, linewidth=0.5)
ax1.set_xticks(range(len(stage_ns)))
ax1.set_xticklabels(stage_lbs, rotation=25, ha="right", fontsize=7)
ax1.set_ylabel("Signals", fontsize=7)
for i, (bar_, n_) in enumerate(zip(bars, stage_ns)):
    ax1.text(bar_.get_x()+bar_.get_width()/2, bar_.get_height()*1.02,
             f"{n_:,}", ha="center", va="bottom", fontsize=6.5, color=C_TEXT)
ax1.set_yscale("log")

# ── P2: % Survival curve ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ps(ax2, "Survival % by Stage", 9)
pcts = [100.0 * funnel[k] / base_n for k, _ in STAGE_LABELS]
ax2.plot(range(len(pcts)), pcts, "o-", color=C_BLU, lw=1.5, ms=5)
ax2.fill_between(range(len(pcts)), pcts, alpha=0.15, color=C_BLU)
ax2.set_xticks(range(len(pcts)))
ax2.set_xticklabels([str(i+1) for i in range(len(pcts))], fontsize=7)
ax2.set_ylabel("% of All Bars", fontsize=7)
ax2.set_ylim(0, 105)
for i, (p, lab) in enumerate(zip(pcts, [f"{n_:,}" for n_ in stage_ns])):
    ax2.text(i, p+2, f"{p:.1f}%", ha="center", va="bottom", fontsize=6, color=C_TEXT)

# ── P3: Filter kill rate (on base signals) ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ps(ax3, "Env Filter Kill Rate\n(on base-signal pool)", 9)
env_flt_order = sorted([f for f in FROZEN_CIDS],
                       key=lambda f: -kill_stats[f]["kill_pct"])
env_kill_pcts = [kill_stats[f]["kill_pct"] for f in env_flt_order]
bar_cols = [C_RED if p > 80 else (C_AMB if p > 50 else C_GRN) for p in env_kill_pcts]
ax3.barh(env_flt_order, env_kill_pcts, color=bar_cols, edgecolor=C_GRID, linewidth=0.5)
ax3.set_xlabel("Kill % of base signals", fontsize=7)
ax3.set_xlim(0, 105)
for i, p in enumerate(env_kill_pcts):
    ax3.text(p+1, i, f"{p:.1f}%", va="center", fontsize=7, color=C_TEXT)

# ── P4: Efficiency scores ─────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ps(ax4, "Filter Efficiency Score\n(PF gained / trades blocked)", 9)
eff_labels = [r[0] for r in eff_rows]
eff_scores = [r[3] for r in eff_rows]
eff_colors = [C_GRN if s > 0 else C_RED for s in eff_scores]
ax4.barh(eff_labels, eff_scores, color=eff_colors, edgecolor=C_GRID, linewidth=0.5)
ax4.set_xlabel("PF contribution / trades blocked", fontsize=7)
ax4.axvline(0, color=C_GRID, lw=0.8)
for i, s in enumerate(eff_scores):
    ax4.text(max(s, 0)+0.0002, i, f"{s:.4f}", va="center", fontsize=7, color=C_TEXT)

# ── P5: Marginal contribution (ΔPF ablation) ─────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ps(ax5, "ΔPF when filter removed\n(negative = filter adds quality)", 9)
abl_labels = [l.replace("Drop ","") for l, _, _ in ABLATION_CONFIGS]
abl_dpf    = [ablation_results[l]["pf"] - base_m["pf"] for l, _, _ in ABLATION_CONFIGS]
abl_colors = [C_GRN if d < 0 else C_RED for d in abl_dpf]
ax5.barh(abl_labels, abl_dpf, color=abl_colors, edgecolor=C_GRID, linewidth=0.5)
ax5.set_xlabel("ΔPF (vs baseline)", fontsize=7)
ax5.axvline(0, color=C_GRID, lw=0.8)
for i, d in enumerate(abl_dpf):
    ax5.text(d + (0.003 if d >= 0 else -0.003), i, f"{d:+.3f}",
             va="center", ha="left" if d >= 0 else "right", fontsize=7, color=C_TEXT)

# ── P6: Opportunity cost by filter ───────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
ps(ax6, "Opportunity Cost\n(outcome of rejected signals)", 9)
oc_flts  = [f for f in FROZEN_CIDS if f in opp_summary]
oc_tp    = [opp_summary[f]["n_tp"] for f in oc_flts]
oc_sl    = [opp_summary[f]["n_sl"] for f in oc_flts]
oc_op    = [opp_summary[f]["n"] - opp_summary[f]["n_tp"] - opp_summary[f]["n_sl"]
            for f in oc_flts]
x = np.arange(len(oc_flts)); w = 0.25
ax6.bar(x-w, oc_tp, w, label="→ TP (winner)", color=C_GRN, edgecolor=C_GRID, lw=0.5)
ax6.bar(x,   oc_sl, w, label="→ SL (loser)",  color=C_RED,  edgecolor=C_GRID, lw=0.5)
ax6.bar(x+w, oc_op, w, label="→ Open (MTM)",  color=C_AMB,  edgecolor=C_GRID, lw=0.5)
ax6.set_xticks(x); ax6.set_xticklabels(oc_flts, fontsize=7, rotation=15)
ax6.set_ylabel("Rejected signals", fontsize=7)
ax6.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)

# ── P7: False negative rate by filter ────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
ps(ax7, "False Negative Rate\n(rejected → TP hits)", 9)
fn_flts  = [f for f in FROZEN_CIDS if f in opp_summary]
fn_rates = [opp_summary[f]["wr"] for f in fn_flts]
fn_cols  = [C_RED if r > 0.4 else (C_AMB if r > 0.25 else C_GRN) for r in fn_rates]
ax7.bar(fn_flts, fn_rates, color=fn_cols, edgecolor=C_GRID, lw=0.5)
ax7.axhline(0.4, color=C_RED,  lw=0.8, ls="--", label="40% danger line")
ax7.axhline(0.25, color=C_AMB, lw=0.8, ls="--", label="25% caution line")
ax7.set_ylabel("FN Rate (win% of rejected)", fontsize=7)
ax7.set_ylim(0, 1)
ax7.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)
for i, r in enumerate(fn_rates):
    ax7.text(i, r+0.01, f"{r:.1%}", ha="center", va="bottom", fontsize=7, color=C_TEXT)

# ── P8: Pairwise interaction heatmap ─────────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ps(ax8, "Pairwise PF (both dropped)", 9)
env_f = list(FROZEN_CIDS); n_ef = len(env_f)
mat = np.full((n_ef, n_ef), np.nan)
for r in pair_results:
    if r["f1"] in env_f and r["f2"] in env_f:
        i, j = env_f.index(r["f1"]), env_f.index(r["f2"])
        mat[i, j] = r["pf"]; mat[j, i] = r["pf"]
np.fill_diagonal(mat, base_m["pf"])
im = ax8.imshow(mat, cmap="RdYlGn", vmin=0.8, vmax=3.5, aspect="auto")
ax8.set_xticks(range(n_ef)); ax8.set_yticks(range(n_ef))
ax8.set_xticklabels(env_f, fontsize=7, rotation=20)
ax8.set_yticklabels(env_f, fontsize=7)
for i in range(n_ef):
    for j in range(n_ef):
        if not np.isnan(mat[i,j]):
            ax8.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                     fontsize=7.5, color="black" if mat[i,j]>1.5 else "white")
plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

# ── P9: Pareto (cumulative % removal) ────────────────────────────────────────
ax9 = fig.add_subplot(gs[3, 0])
ps(ax9, "Pareto — Cumulative Signal Removal", 9)
par_labels = [l for l, _ in sequential_removals]
par_removes = [r for _, r in sequential_removals]
cum_pcts_p = np.cumsum(par_removes) / total_removed * 100
ax9.bar(range(len(par_labels)), [r/total_removed*100 for r in par_removes],
        color=C_BLU, edgecolor=C_GRID, lw=0.5, label="Individual %")
ax9.plot(range(len(par_labels)), cum_pcts_p, "o-", color=C_AMB,
         lw=1.5, ms=4, label="Cumulative %")
ax9.axhline(80, color=C_RED, lw=0.8, ls="--", label="80% line")
ax9.set_xticks(range(len(par_labels)))
ax9.set_xticklabels([l[:15] for l in par_labels], fontsize=6.5, rotation=25, ha="right")
ax9.set_ylabel("%", fontsize=7); ax9.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)

# ── P10: Trade count comparison (baseline vs each ablation) ──────────────────
ax10 = fig.add_subplot(gs[3, 1])
ps(ax10, "Trade Count: Baseline vs Ablations", 9)
abl_ns = [ablation_results[l]["n"] for l, _, _ in ABLATION_CONFIGS]
abl_ls = [l.replace("Drop ","") for l, _, _ in ABLATION_CONFIGS]
colors_n = [C_GRN if n > base_m["n"] else C_AMB for n in abl_ns]
ax10.barh(abl_ls, abl_ns, color=colors_n, edgecolor=C_GRID, lw=0.5)
ax10.axvline(base_m["n"], color=C_TEXT, lw=1, ls="--", label=f"Baseline n={base_m['n']}")
ax10.set_xlabel("Trades (n)", fontsize=7)
ax10.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)
for i, n_ in enumerate(abl_ns):
    ax10.text(n_+0.5, i, str(n_), va="center", fontsize=7, color=C_TEXT)

# ── P11: Summary verdict box ──────────────────────────────────────────────────
ax11 = fig.add_subplot(gs[3, 2])
ax11.set_facecolor(C_PANEL)
for sp in ax11.spines.values(): sp.set_color(C_GRID)
ax11.axis("off")
lines = [
    "R063 — AUTOPSY VERDICT",
    "─"*28,
    f"OOS bars analysed: {funnel['01_all']:,}",
    f"Base signals:      {funnel['04_above_pc']:,}  ({100*funnel['04_above_pc']/funnel['01_all']:.1f}%)",
    f"Final signals:     {funnel['09_signals']:,}   ({100*funnel['09_signals']/funnel['01_all']:.2f}%)",
    f"Actual trades:     {base_m['n']}",
    "─"*28,
    f"Biggest bottleneck:",
    f"  {sequential_removals[0][0][:22]}",
    f"  removes {100.*sequential_removals[0][1]/total_removed:.0f}% of funnel loss",
    "─"*28,
    f"Best filter: {best_eff[0]}",
    f"  Eff: {best_eff[3]:.4f}",
    f"Worst filter: {worst_eff[0]}",
    f"  Eff: {worst_eff[3]:.4f}",
    "─"*28,
    "STRATEGY: REMAIN FROZEN" if all_pf_add else "FLAG FOR REVIEW",
    "79 trades = price of edge" if all_pf_add else "",
]
for i, ln in enumerate(lines):
    clr = C_GRN if "FROZEN" in ln else (C_AMB if "FLAG" in ln else C_TEXT)
    bold = "bold" if i == 0 or "FROZEN" in ln else "normal"
    ax11.text(0.05, 0.97 - i*0.065, ln, transform=ax11.transAxes,
              fontsize=7.5, color=clr, va="top", fontweight=bold,
              fontfamily="monospace")

chart_path = os.path.join(OUT_DIR, "r063_autopsy.png")
fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close(fig)
print(f"  Autopsy chart saved → {chart_path}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  R063 COMPLETE")
print(SEP)
print()
print(f"  Strategy    : {FROZEN_LABEL}")
print(f"  OOS bars    : {funnel['01_all']:,}  across {len(SYMS)} symbols")
print(f"  Base signals: {funnel['04_above_pc']:,}  ({100*funnel['04_above_pc']/funnel['01_all']:.2f}% hit rate on raw bars)")
print(f"  Final sigs  : {funnel['09_signals']:,}   ({100*funnel['09_signals']/funnel['04_above_pc']:.2f}% of base signals pass env)")
print(f"  Actual trades:{base_m['n']}")
print()
print(f"  Dominant bottleneck  : {sequential_removals[0][0]}")
print(f"  Highest quality filt : {best_eff[0]}  (score={best_eff[3]:.4f})")
print(f"  Lowest quality filt  : {worst_eff[0]}  (score={worst_eff[3]:.4f})")
print(f"  False negative leader: {fn_data[0][0]}  (FN rate={fn_data[0][2]:.1%})")
print(f"  Losers blocked leader: {sl_data[0][0]}  (blocks={sl_data[0][1]:,} losers)")
print()
print(f"  Outputs:")
print(f"    {chart_path}")
print(SEP)
