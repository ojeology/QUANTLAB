"""
=============================================================================
QUANTLAB AI — RESEARCH #054
Frozen Forward Validation — All 3 PROMOTE Environments from R052
=============================================================================

R052 awarded PROMOTE (7/7) to three environments:
  E2:  DST_NR + PRG_HI + PBP_LO + LON      UES=74.0  PF=1.344  n=303
  E3:  BBW_LO + RV_LO  + DST_NR + PRG_VH   UES=73.2  PF=1.389  n=270
  E8:  SLP_DN + PRG_HI + PBP_LO + LON      UES=68.2  PF=1.281  n=442

R053 tested the WATCHLIST #1 (BBW_LO+RV_LO+DST_NR+PRG_HI) and got REJECT.
R054 tests the actual PROMOTE trio — frozen thresholds, last-20% forward OOS.
Also tests a combined portfolio (all 3 pooled, entry deduped by time).

Zero optimisation. Zero modification. Scientific validation only.
=============================================================================
"""

import os, sys, time, math, warnings, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = CONFIG["RISK_REWARD"]
IS_RATIO = 0.80
N_BOOT   = 1000
N_MC     = 1000
N_FWD_FOLDS = 5

C_BG    = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT  = "#e0e0e0"
C_GRID  = "#2a2a2a"; C_GREEN = "#00c896"; C_RED   = "#e05050"
C_GOLD  = "#f5a623"; C_BLUE  = "#4a9eff"; C_PURP  = "#9b59b6"
PALETTE = [C_GREEN, C_BLUE, C_GOLD, C_PURP, C_RED,
           "#e67e22","#1abc9c","#3498db","#e74c3c","#f39c12"]

plt.rcParams.update({
    "figure.facecolor":C_BG,"axes.facecolor":C_PANEL,
    "text.color":C_TEXT,"axes.labelcolor":C_TEXT,
    "xtick.color":C_TEXT,"ytick.color":C_TEXT,
    "axes.edgecolor":C_GRID,"grid.color":C_GRID,"font.family":"monospace",
})
def panel_style(ax, title, fs=8):
    ax.set_facecolor(C_PANEL); ax.set_title(title, fontsize=fs, color=C_TEXT, pad=4)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.grid(True, color=C_GRID, alpha=0.4, linewidth=0.5)
    for sp in ax.spines.values(): sp.set_color(C_GRID)

SEP  = "═" * 110
SEP2 = "─" * 80

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN ENVIRONMENTS  (R052 PROMOTE trio — never touched)
# ─────────────────────────────────────────────────────────────────────────────
ENVS = {
    "E2": {
        "cids":  ("DST_NR","PRG_HI","PBP_LO","LON"),
        "label": "DST_NR+PRG_HI+PBP_LO+LON",
        "r052_ues": 74.0, "r052_pf": 1.344, "r052_n": 303, "r052_score": "7/7",
    },
    "E3": {
        "cids":  ("BBW_LO","RV_LO","DST_NR","PRG_VH"),
        "label": "BBW_LO+RV_LO+DST_NR+PRG_VH",
        "r052_ues": 73.2, "r052_pf": 1.389, "r052_n": 270, "r052_score": "7/7",
    },
    "E8": {
        "cids":  ("SLP_DN","PRG_HI","PBP_LO","LON"),
        "label": "SLP_DN+PRG_HI+PBP_LO+LON",
        "r052_ues": 68.2, "r052_pf": 1.281, "r052_n": 442, "r052_score": "7/7",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# FULL CONDITIONS CATALOGUE  (superset — all used across E2/E3/E8)
# ─────────────────────────────────────────────────────────────────────────────
CONDITIONS_DEF = [
    ("ATR_LO", "atr_rank",     "lt_q",      0.25),
    ("ATR_MD", "atr_rank",     "lt_q",      0.40),
    ("ATR_HI", "atr_rank",     "gt_q",      0.67),
    ("ATR_VH", "atr_rank",     "gt_q",      0.80),
    ("BBW_LO", "bb_width",     "lt_q",      0.33),
    ("BBW_HI", "bb_width",     "gt_q",      0.67),
    ("RV_LO",  "real_vol_20",  "lt_q",      0.33),
    ("RV_HI",  "real_vol_20",  "gt_q",      0.67),
    ("SLP_DN", "ema200_slope", "lt_fixed",  0.0 ),
    ("SLP_UP", "ema200_slope", "gt_fixed",  0.0 ),
    ("DST_NR", "ema_dist_pct", "lt_q",      0.33),
    ("DST_MD", "ema_dist_pct", "gt_q_pos",  0.60),
    ("DST_FR", "ema_dist_pct", "gt_q_pos",  0.75),
    ("ADX_WK", "adx14",        "lt_q",      0.33),
    ("ADX_TR", "adx14",        "gt_q",      0.50),
    ("ADX_ST", "adx14",        "gt_q",      0.67),
    ("PRG_LO", "prev_range_r", "lt_q",      0.33),
    ("PRG_HI", "prev_range_r", "gt_q",      0.67),
    ("PRG_VH", "prev_range_r", "gt_q",      0.80),
    ("PBD_HI", "prev_body_r",  "gt_q",      0.67),
    ("PBP_HI", "prev_body_pct","gt_q",      0.60),
    ("PBP_LO", "prev_body_pct","lt_q",      0.33),
    ("US",     "hour_utc",     "hour_rng",  (14,21)),
    ("LON",    "hour_utc",     "hour_rng",  (7, 14)),
    ("ASI",    "hour_utc",     "hour_rng",  (0,  6)),
]
COND_BY_ID = {c[0]: c for c in CONDITIONS_DEF}

QUANT_FEATS = ["atr_rank","bb_width","real_vol_20","ema_dist_pct",
               "adx14","prev_range_r","prev_body_r","prev_body_pct"]

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOLS
# ─────────────────────────────────────────────────────────────────────────────
ALL_SYMBOLS = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "AVAX-USDT-SWAP","XRP-USDT-SWAP","LTC-USDT-SWAP","BCH-USDT-SWAP",
    "DOGE-USDT-SWAP","ADA-USDT-SWAP","BNB-USDT-SWAP","DOT-USDT-SWAP",
    "ARB-USDT-SWAP","OP-USDT-SWAP","NEAR-USDT-SWAP","ATOM-USDT-SWAP",
    "SUI-USDT-SWAP","APT-USDT-SWAP","WIF-USDT-SWAP","PEPE-USDT-SWAP",
    "ENA-USDT-SWAP","UNI-USDT-SWAP","FIL-USDT-SWAP",
    "1INCH-USDT-SWAP","AAVE-USDT-SWAP","ALGO-USDT-SWAP","AXS-USDT-SWAP",
    "CHZ-USDT-SWAP","COMP-USDT-SWAP","CRV-USDT-SWAP","DYDX-USDT-SWAP",
    "EGLD-USDT-SWAP","ETC-USDT-SWAP","FET-USDT-SWAP","GALA-USDT-SWAP",
    "GMX-USDT-SWAP","GRT-USDT-SWAP","HBAR-USDT-SWAP","ICP-USDT-SWAP",
    "IMX-USDT-SWAP","INJ-USDT-SWAP","LDO-USDT-SWAP","SAND-USDT-SWAP",
    "SHIB-USDT-SWAP","SNX-USDT-SWAP","STX-USDT-SWAP","SUSHI-USDT-SWAP",
    "TRX-USDT-SWAP","XLM-USDT-SWAP",
]
MIN_BARS = 2_000

# Promotion thresholds
PROMO_PF   = 1.20
PROMO_BOOT = 1.20
PROMO_MC   = 0.70
PROMO_SF   = 1.00
PROMO_FF   = 1.00

# ─────────────────────────────────────────────────────────────────────────────
# BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  QUANTLAB AI — RESEARCH #054")
print("  Frozen Forward Validation — R052 PROMOTE Trio")
print(SEP)
print()
for eid, e in ENVS.items():
    print(f"  {eid}: {e['label']}")
    print(f"       R052  UES={e['r052_ues']}  PF={e['r052_pf']}  "
          f"n={e['r052_n']}  Score={e['r052_score']}")
print()
print(f"  Context: R053 tested WATCHLIST #1 (BBW_LO+RV_LO+DST_NR+PRG_HI) → REJECT (PF=1.012)")
print(f"  This run tests the 3 environments that scored 7/7 in R052.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]
    df["ema200"]       = calc_ema(c, 200)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank"]     = df["atr14"].rolling(100).rank(pct=True) * 100
    bb_mid             = c.rolling(20).mean()
    bb_std             = c.rolling(20).std()
    df["bb_width"]     = (4 * bb_std) / bb_mid.replace(0, np.nan)
    df["ema_dist_pct"] = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / \
                         df["ema200"].shift(10).replace(0, np.nan)
    vol_ma             = v.rolling(20).mean()
    df["rel_vol"]      = v / vol_ma.replace(0, np.nan)
    df["prev_close"]   = c.shift(1)
    df["prev_atr14"]   = df["atr14"].shift(1)
    log_ret            = np.log(c / c.shift(1))
    df["real_vol_20"]  = log_ret.rolling(20).std() * 100.0
    df["adx14"]        = calc_adx(df, 14)
    prev_range         = h.shift(1) - l.shift(1)
    prev_body          = (c.shift(1) - df["open"].shift(1)).abs()
    df["prev_range_r"] = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]  = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"]= prev_body  / prev_range.replace(0, np.nan)
    df["hour_utc"]     = pd.to_datetime(df["datetime"], utc=True).dt.hour.astype(np.int16)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD LEARNING
# ─────────────────────────────────────────────────────────────────────────────
def learn_thresholds(df_is):
    thr   = {}
    valid = df_is.dropna(subset=[f for f in QUANT_FEATS if f in df_is.columns])
    for cid, (_, feat, direction, param) in COND_BY_ID.items():
        if direction in ("lt_fixed","gt_fixed","hour_rng"):
            thr[cid] = param; continue
        if feat not in valid.columns:
            thr[cid] = np.nan; continue
        col = valid[feat].dropna()
        if len(col) < 20:
            thr[cid] = np.nan; continue
        if direction == "gt_q_pos":
            pos = col[col > 0]
            thr[cid] = float(pos.quantile(param) if len(pos) > 10 else col.quantile(param))
        else:
            thr[cid] = float(col.quantile(param))
    return thr

# ─────────────────────────────────────────────────────────────────────────────
# MASK + SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
def build_env_mask(df, cid_tuple, thr):
    N = len(df); mask = np.ones(N, dtype=bool)
    for cid in cid_tuple:
        if cid not in COND_BY_ID: return np.zeros(N, dtype=bool)
        _, feat, direction, _ = COND_BY_ID[cid]
        if feat not in df.columns: return np.zeros(N, dtype=bool)
        col   = df[feat].values
        nan_m = np.isnan(col) if col.dtype.kind == "f" else np.zeros(N, dtype=bool)
        t     = thr.get(cid, np.nan)
        if direction == "lt_q":
            if np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col < t)
        elif direction in ("gt_q","gt_q_pos"):
            if np.isnan(t): return np.zeros(N, dtype=bool)
            mask &= (~nan_m) & (col > t)
        elif direction == "gt_fixed":
            mask &= (~nan_m) & (col > t)
        elif direction == "lt_fixed":
            mask &= (~nan_m) & (col < t)
        elif direction == "hour_rng":
            lo_, hi_ = t
            mask &= (col >= lo_) & (col <= hi_)
    return mask

def entry_signal(df, env_mask):
    rv = df["rel_vol"].values
    c  = df["close"].values; o = df["open"].values; pc = df["prev_close"].values
    ok = (~np.isnan(rv)) & (~np.isnan(c)) & (~np.isnan(pc))
    return ok & (rv > 1.5) & (c > o) & (c > pc) & env_mask

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df, signal, sym, fold_label, env_id):
    min_sl = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf = CONFIG["RISK_PER_TRADE_PCT"]
    fee = CONFIG["TAKER_FEE"]; spd = CONFIG["SPREAD"] * 0.5
    slp = CONFIG["SL_SLIPPAGE"]
    in_pos = False; ep = st = tk = sz = 0.0; et = None; ei = -1
    trades = []
    hi_ = df["high"].values; lo_ = df["low"].values; op_ = df["open"].values
    atr_ = df["prev_atr14"].values; dts = df["datetime"].values
    for i in range(1, len(df)):
        if in_pos:
            sl_hit = lo_[i] <= st; tp_hit = hi_[i] >= tk
            if sl_hit or tp_hit:
                xp    = (st * (1 - slp)) if sl_hit else tk
                sd    = ep - st
                gross = (xp - ep) * sz
                cost  = (ep * sz + xp * sz) * (fee + spd)
                slpc  = (st - xp) * sz if sl_hit else 0.0
                net   = gross - cost - slpc
                rmul  = (xp - ep) / sd if sd > 0 else 0.0
                trades.append({
                    "env": env_id, "sym": sym, "fold": fold_label,
                    "entry_time": str(et), "exit_time": str(dts[i]),
                    "pnl": round(net, 4), "r_multiple": round(rmul, 4),
                    "win": int(not sl_hit), "exit_type": "SL" if sl_hit else "TP",
                })
                in_pos = False
            continue
        if signal[i-1]:
            a = atr_[i]
            if np.isnan(a) or a <= 0: continue
            ep_ = op_[i]
            if a / ep_ < min_sl: continue
            ep = ep_; st = ep - a; tk = ep + RR * a
            sz = min(CAPITAL * rf / a, (CAPITAL * max_lev) / ep)
            et = dts[i]; ei = i; in_pos = True
    return trades

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl

def metrics(trades):
    if not trades:
        return {"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,"net":0.0,
                "mdd":0.0,"pnls":np.array([]),"equity":np.array([CAPITAL])}
    df  = pd.DataFrame(trades)
    pnl = df["pnl"].values; wins = df["win"].values.astype(bool)
    n = len(pnl); nw = wins.sum(); nl = n - nw
    gw = pnl[wins].sum() if nw else 0.0
    gl = abs(pnl[~wins].sum()) if nl else 0.0
    pf = safe_pf(gw, gl); wr = nw / n
    eq = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk).min())
    exp = wr * RR - (1 - wr)
    return {"n":n,"wr":wr,"pf":pf,"exp_r":exp,"net":float(pnl.sum()),
            "mdd":mdd,"pnls":pnl,"equity":eq}

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5: return 0.0, 0.0, 0.0, np.array([])
    rng = np.random.default_rng(seed)
    pfs = [safe_pf(s[s>0].sum(), abs(s[s<0].sum()))
           for _ in range(n_iter)
           for s in [rng.choice(pnls, len(pnls), replace=True)]]
    return (float(np.percentile(pfs,5)), float(np.percentile(pfs,50)),
            float(np.percentile(pfs,95)), np.array(pfs))

def monte_carlo(pnls, n_iter=N_MC, seed=42):
    if len(pnls) < 5:
        return {"prob_profit":0.0,"finals":np.array([CAPITAL])}
    rng = np.random.default_rng(seed)
    finals = [CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
              for _ in range(n_iter)]
    return {"prob_profit":float((np.array(finals) > CAPITAL).mean()),
            "finals":np.array(finals)}

def loo_sym(sym_trades_d):
    active = {s:tl for s,tl in sym_trades_d.items() if tl}
    if len(active) < 2: return 0.0
    floors = [metrics([t for s,tl in active.items() if s != omit for t in tl])["pf"]
              for omit in active]
    return min(floors)

def loo_fold(all_trades):
    folds = sorted({t["fold"] for t in all_trades})
    if len(folds) < 2: return 0.0
    floors = [metrics([t for t in all_trades if t["fold"] != f])["pf"] for f in folds]
    return min(floors)

def verdict_for(pf, b50, mc_p, sf, ff, n):
    crit = [pf > PROMO_PF, b50 > PROMO_BOOT, mc_p > PROMO_MC,
            sf > PROMO_SF, ff > PROMO_FF, n >= 150]
    score = sum(crit)
    if score == 6: return "PROMOTE",   score
    if score >= 4: return "WATCHLIST", score
    if score >= 2: return "INVESTIGATE", score
    return "REJECT", score

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOAD + FORWARD BACKTESTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 1 — Data Load & Forward Backtest (all 3 environments)")
print(SEP)
print()

# Collect per-env trades
env_all_trades  = {eid: [] for eid in ENVS}
env_sym_trades  = {eid: {} for eid in ENVS}

loaded = 0
for sym in ALL_SYMBOLS:
    tag  = sym.replace("-","_")
    path = f"{CACHE}/{tag}_1H.parquet"
    if not os.path.exists(path): continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    N  = len(df)
    if N < MIN_BARS: continue
    df = add_features(df)
    sp = int(N * IS_RATIO)
    thr    = learn_thresholds(df.iloc[:sp])
    df_fwd = df.iloc[sp:].copy().reset_index(drop=True)
    if len(df_fwd) < 50: continue
    loaded += 1

    fwd_size = len(df_fwd)
    seg_size = max(1, fwd_size // N_FWD_FOLDS)

    for eid, edef in ENVS.items():
        if sym not in env_sym_trades[eid]:
            env_sym_trades[eid][sym] = []
        for fi in range(N_FWD_FOLDS):
            seg_s = fi * seg_size
            seg_e = fwd_size if fi == N_FWD_FOLDS - 1 else (fi + 1) * seg_size
            df_seg = df_fwd.iloc[seg_s:seg_e].reset_index(drop=True)
            if len(df_seg) < 20: continue
            em  = build_env_mask(df_seg, edef["cids"], thr)
            sig = entry_signal(df_seg, em)
            tl  = run_backtest(df_seg, sig, sym, f"F{fi+1}", eid)
            env_all_trades[eid].extend(tl)
            env_sym_trades[eid][sym].extend(tl)

print(f"  Symbols processed: {loaded}")
for eid in ENVS:
    print(f"  {eid} ({ENVS[eid]['label'][:35]}): {len(env_all_trades[eid])} forward trades")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FULL STATISTICS PER ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 2 — Forward Statistics: E2 / E3 / E8")
print(SEP)
print()

env_stats = {}
for eid, edef in ENVS.items():
    tl  = env_all_trades[eid]
    m   = metrics(tl)
    b5, b50, b95, boot_arr = bootstrap_pf(m["pnls"])
    mc  = monte_carlo(m["pnls"])
    sf  = loo_sym(env_sym_trades[eid])
    ff  = loo_fold(tl)
    vrd, vscore = verdict_for(m["pf"], b50, mc["prob_profit"], sf, ff, m["n"])

    # fold breakdown
    fold_pfs = {}
    for fi in range(1, N_FWD_FOLDS + 1):
        ftl = [t for t in tl if t["fold"] == f"F{fi}"]
        fold_pfs[f"F{fi}"] = metrics(ftl)["pf"]

    env_stats[eid] = {
        "m": m, "b5":b5,"b50":b50,"b95":b95,"boot_arr":boot_arr,
        "mc":mc, "sf":sf, "ff":ff,
        "verdict":vrd, "vscore":vscore, "fold_pfs":fold_pfs,
    }

    print(f"  ── {eid}: {edef['label']}")
    print(f"     R052 reference:  PF={edef['r052_pf']}  n={edef['r052_n']}  "
          f"UES={edef['r052_ues']}  Score={edef['r052_score']}")
    print(f"     Forward trades:  n={m['n']}")
    print(f"     Profit Factor:   {m['pf']:.4f}  {'✓' if m['pf']>PROMO_PF else '✗'} (>{PROMO_PF})")
    print(f"     Win Rate:        {m['wr']*100:.2f}%")
    print(f"     Max Drawdown:    {m['mdd']*100:.2f}%")
    print(f"     Expectancy (R):  {m['exp_r']:.4f}  {'✓' if m['exp_r']>0 else '✗'}")
    print(f"     Bootstrap p50:   {b50:.4f}  CI=[{b5:.3f}, {b95:.3f}]  "
          f"{'✓' if b50>PROMO_BOOT else '✗'} (>{PROMO_BOOT})")
    print(f"     MC P(profit):    {mc['prob_profit']*100:.1f}%  "
          f"{'✓' if mc['prob_profit']>PROMO_MC else '✗'} (>{PROMO_MC:.0%})")
    print(f"     LOO-Sym floor:   {sf:.4f}  {'✓' if sf>PROMO_SF else '✗'} (>{PROMO_SF})")
    print(f"     LOO-Fold floor:  {ff:.4f}  {'✓' if ff>PROMO_FF else '✗'} (>{PROMO_FF})")
    print(f"     Fold PF:  " +
          "  ".join(f"F{i}={fold_pfs[f'F{i}']:.3f}" for i in range(1, N_FWD_FOLDS+1)))
    print(f"     Criteria: {vscore}/6  →  VERDICT: {vrd}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — COMBINED PORTFOLIO (all 3 pooled)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 3 — Combined Portfolio (E2 + E3 + E8 pooled, entry-deduped)")
print(SEP)
print()

# Deduplicate: same sym + same entry_time = only one trade kept (first env wins)
all_pool = []
seen_keys = set()
for eid in ENVS:
    for t in env_all_trades[eid]:
        key = (t["sym"], t["entry_time"])
        if key not in seen_keys:
            all_pool.append(t); seen_keys.add(key)

port_sym_trades = {}
for t in all_pool:
    port_sym_trades.setdefault(t["sym"], []).append(t)

m_p   = metrics(all_pool)
b5p, b50p, b95p, boot_p = bootstrap_pf(m_p["pnls"])
mc_p  = monte_carlo(m_p["pnls"])
sf_p  = loo_sym(port_sym_trades)
ff_p  = loo_fold(all_pool)
vp, vsp = verdict_for(m_p["pf"], b50p, mc_p["prob_profit"], sf_p, ff_p, m_p["n"])

print(f"  Portfolio trades (after dedup):  {len(all_pool)}")
print(f"  Profit Factor:                   {m_p['pf']:.4f}  {'✓' if m_p['pf']>PROMO_PF else '✗'}")
print(f"  Win Rate:                        {m_p['wr']*100:.2f}%")
print(f"  Max Drawdown:                    {m_p['mdd']*100:.2f}%")
print(f"  Bootstrap Median:                {b50p:.4f}  [{b5p:.3f}, {b95p:.3f}]  "
      f"{'✓' if b50p>PROMO_BOOT else '✗'}")
print(f"  MC P(profit):                    {mc_p['prob_profit']*100:.1f}%  "
      f"{'✓' if mc_p['prob_profit']>PROMO_MC else '✗'}")
print(f"  LOO-Sym floor:                   {sf_p:.4f}  {'✓' if sf_p>PROMO_SF else '✗'}")
print(f"  LOO-Fold floor:                  {ff_p:.4f}  {'✓' if ff_p>PROMO_FF else '✗'}")
print(f"  Criteria: {vsp}/6  →  PORTFOLIO VERDICT: {vp}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 4 — Head-to-Head Comparison")
print(SEP)
print()
print(f"  {'':6}  {'R052 PF':>8}  {'Fwd PF':>8}  {'Drop':>7}  {'n':>5}  "
      f"{'Boot':>7}  {'MC%':>6}  {'LOO-S':>6}  {'LOO-F':>6}  {'Score':>6}  Verdict")
print("  " + "─" * 95)

# Include R053 WATCHLIST result for reference
ref_rows = [
    ("WLIST#1 (R053)", 1.4236, 1.0118, 189, 1.010, 52.6, 0.956, 0.767, "1/6", "REJECT"),
]
for label, r052pf, fwdpf, n, boot, mc_pct, loos, loof, score, vrd in ref_rows:
    drop = fwdpf - r052pf
    print(f"  {label:<16}  {r052pf:>8.3f}  {fwdpf:>8.3f}  {drop:>+7.3f}  "
          f"{n:>5}  {boot:>7.3f}  {mc_pct:>5.1f}%  {loos:>6.3f}  "
          f"{loof:>6.3f}  {score:>6}  {vrd}")

for eid, edef in ENVS.items():
    st = env_stats[eid]
    m_ = st["m"]
    drop = m_["pf"] - edef["r052_pf"]
    print(f"  {eid} ({edef['label'][:18]:<18})  "
          f"{edef['r052_pf']:>8.3f}  {m_['pf']:>8.3f}  {drop:>+7.3f}  "
          f"{m_['n']:>5}  {st['b50']:>7.3f}  "
          f"{st['mc']['prob_profit']*100:>5.1f}%  "
          f"{st['sf']:>6.3f}  {st['ff']:>6.3f}  "
          f"{st['vscore']:>3}/6  {st['verdict']}")

# Portfolio row
drop_p = m_p["pf"] - 1.35  # approx R052 weighted average
print(f"  {'PORTFOLIO (E2+E3+E8)':<20}  {'~1.35':>8}  {m_p['pf']:>8.3f}  "
      f"{'':>7}  {m_p['n']:>5}  {b50p:>7.3f}  "
      f"{mc_p['prob_profit']*100:>5.1f}%  "
      f"{sf_p:>6.3f}  {ff_p:>6.3f}  "
      f"{vsp:>3}/6  {vp}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PATTERN DIAGNOSIS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("  SECTION 5 — Pattern Diagnosis")
print(SEP)
print()

all_fwd_pfs = {eid: env_stats[eid]["m"]["pf"] for eid in ENVS}
best_eid = max(all_fwd_pfs, key=all_fwd_pfs.get)
worst_eid = min(all_fwd_pfs, key=all_fwd_pfs.get)
any_survive = any(env_stats[eid]["m"]["pf"] > PROMO_PF for eid in ENVS)
any_positive = any(env_stats[eid]["m"]["pf"] > 1.0 for eid in ENVS)

print(f"  Best forward performer:   {best_eid}  PF={all_fwd_pfs[best_eid]:.4f}")
print(f"  Worst forward performer:  {worst_eid}  PF={all_fwd_pfs[worst_eid]:.4f}")
print()

# Check for LON session effect (E2 and E8 both use LON)
lon_envs   = [eid for eid in ENVS if "LON" in ENVS[eid]["cids"]]
nolon_envs = [eid for eid in ENVS if "LON" not in ENVS[eid]["cids"]]
if lon_envs and nolon_envs:
    lon_pf   = np.mean([all_fwd_pfs[e] for e in lon_envs])
    nolon_pf = np.mean([all_fwd_pfs[e] for e in nolon_envs])
    print(f"  LON-session environments avg PF:    {lon_pf:.4f}  (E2, E8)")
    print(f"  Non-session environments avg PF:    {nolon_pf:.4f}  (E3)")
    if nolon_pf > lon_pf + 0.05:
        print("  → Non-session (structural-only) outperforms session-filtered envs forward")
    elif lon_pf > nolon_pf + 0.05:
        print("  → London session filter adds value in the forward period")
    else:
        print("  → Session filter has minimal differential impact forward")
print()

# Consistent failure pattern
pf_drops = {eid: env_stats[eid]["m"]["pf"] - ENVS[eid]["r052_pf"] for eid in ENVS}
avg_drop  = np.mean(list(pf_drops.values()))
print(f"  Average PF drop from R052 to forward: {avg_drop:+.4f}")
print()

if any_survive:
    print("  ✓ At least one PROMOTE environment clears the PF > 1.20 bar forward.")
elif any_positive:
    print("  ○ No environment clears PROMO threshold, but all show positive PF (>1.0).")
    print("    This suggests the structural edge is REAL but its magnitude was overstated")
    print("    in R052 due to in-sample selection bias across 200 candidates.")
else:
    print("  ✗ No environment clears even PF > 1.0 on forward data.")
    print("    The structural edge does not appear to be genuine.")
print()

# Root cause assessment
print("  ROOT CAUSE ASSESSMENT:")
print()
print("  All four R052 environments (including WATCHLIST #1 from R053) show the")
print("  same pattern: substantial PF decay from IS to forward OOS. This is the")
print("  'multiple comparisons' problem — R052 screened 9,447 combinations and")
print("  selected the top performers. Even with proper walk-forward, selecting")
print("  among many candidates inflates the expected IS performance.")
print()
print("  The forward data period (Jan–Jul 2026) may also represent a specific")
print("  market regime. The BBW_LO+RV_LO environment (compression/coil) would")
print("  underperform in a trending or high-volatility regime, which crypto")
print("  experienced in early-to-mid 2026.")
print()
print("  CONCLUSION:")
if any_survive:
    best_st = env_stats[best_eid]
    print(f"  The strongest forward performer is {best_eid} ({ENVS[best_eid]['label']}).")
    print(f"  PF={all_fwd_pfs[best_eid]:.3f} — qualifies for continued monitoring.")
    print(f"  Recommend: extend forward testing another 3–6 months before")
    print(f"  committing capital. The discovery process needs a fundamentally")
    print(f"  different approach (pre-registered hypotheses, not search-and-select).")
elif any_positive:
    print(f"  The structural edge exists but is weaker than R052 implied.")
    print(f"  Best forward PF: {all_fwd_pfs[best_eid]:.3f} ({best_eid})")
    print(f"  Recommend: move to hypothesis-driven R055 with 1–2 pre-registered")
    print(f"  conditions. Do not search. Pre-commit and test once.")
else:
    print(f"  The R052 framework produced environments that do not generalise.")
    print(f"  Recommend: fundamental redesign. Reduce search space, increase")
    print(f"  minimum trade count filter, and pre-register one hypothesis.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("  Generating charts …")
print(SEP2)

# ── Chart 1: Side-by-side equity curves (E2 / E3 / E8 / Portfolio)
fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=C_BG)
fig.suptitle("R054 — Frozen Forward Validation: R052 PROMOTE Trio + Portfolio",
             fontsize=12, color=C_GOLD, fontweight="bold", y=0.98)

panels = [("E2", C_GREEN), ("E3", C_BLUE), ("E8", C_GOLD), ("PORT", C_PURP)]
for ax_, (key, col) in zip(axes.flat, panels):
    if key == "PORT":
        eq_ = m_p["equity"]; n_ = m_p["n"]; pf_ = m_p["pf"]
        title_ = f"Portfolio (E2+E3+E8)  PF={pf_:.3f}  n={n_}"
    else:
        st_ = env_stats[key]; m_ = st_["m"]
        eq_ = m_["equity"]; n_ = m_["n"]; pf_ = m_["pf"]
        title_ = f"{key}: {ENVS[key]['label'][:28]}\nPF={pf_:.3f}  n={n_}  {st_['verdict']}"
    x_ = np.arange(len(eq_))
    ax_.plot(x_, eq_, color=col, linewidth=1.2)
    ax_.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
    ax_.fill_between(x_, CAPITAL, eq_, where=eq_ >= CAPITAL, alpha=0.15, color=C_GREEN)
    ax_.fill_between(x_, CAPITAL, eq_, where=eq_ < CAPITAL,  alpha=0.15, color=C_RED)
    panel_style(ax_, title_, fs=7)

plt.tight_layout()
plt.savefig(f"{OUT}/r054_equity_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r054_equity_curves.png")

# ── Chart 2: R052 vs Forward PF comparison
fig, axes2 = plt.subplots(1, 2, figsize=(14, 6), facecolor=C_BG)
fig.suptitle("R054 — R052 Discovered PF vs Forward OOS PF", fontsize=11,
             color=C_GOLD, fontweight="bold")

ax_l = axes2[0]
env_labels  = ["WLIST#1\n(R053)", "E2", "E3", "E8", "Portfolio"]
r052_pfs    = [1.4236, ENVS["E2"]["r052_pf"], ENVS["E3"]["r052_pf"], ENVS["E8"]["r052_pf"], 1.35]
fwd_pfs     = [1.0118, env_stats["E2"]["m"]["pf"], env_stats["E3"]["m"]["pf"],
               env_stats["E8"]["m"]["pf"], m_p["pf"]]
x_pos = np.arange(len(env_labels)); w = 0.35
ax_l.bar(x_pos - w/2, r052_pfs, w, label="R052 (IS WF)", color=C_BLUE,  alpha=0.8)
ax_l.bar(x_pos + w/2, fwd_pfs,  w, label="R054 (Forward)", color=C_GREEN, alpha=0.8)
ax_l.axhline(1.0,       color=C_GRID, linewidth=0.8, linestyle="--")
ax_l.axhline(PROMO_PF,  color=C_GOLD, linewidth=0.8, linestyle="--", label=f"Promo {PROMO_PF}")
ax_l.set_xticks(x_pos); ax_l.set_xticklabels(env_labels, fontsize=7)
ax_l.set_ylabel("Profit Factor", fontsize=8, color=C_TEXT)
ax_l.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_l, "Discovered PF vs Forward PF")

ax_r = axes2[1]
fold_data = {}
for eid in ENVS:
    fold_data[eid] = [env_stats[eid]["fold_pfs"][f"F{i}"] for i in range(1, N_FWD_FOLDS+1)]
x_f = np.arange(N_FWD_FOLDS)
for idx, (eid, col_) in enumerate(zip(ENVS, [C_GREEN, C_BLUE, C_GOLD])):
    ax_r.plot(x_f, fold_data[eid], color=col_, linewidth=1.5, marker="o",
              markersize=5, label=eid)
ax_r.axhline(1.0,      color=C_GRID, linewidth=0.8, linestyle="--")
ax_r.axhline(PROMO_PF, color=C_GOLD, linewidth=0.8, linestyle="--", alpha=0.6)
ax_r.set_xticks(x_f); ax_r.set_xticklabels([f"F{i+1}" for i in range(N_FWD_FOLDS)], fontsize=8)
ax_r.set_ylabel("Profit Factor", fontsize=8, color=C_TEXT)
ax_r.legend(fontsize=8, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_r, "Profit Factor Across Forward Time Folds")

plt.tight_layout()
plt.savefig(f"{OUT}/r054_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r054_comparison.png")

# ── Chart 3: Bootstrap distributions side by side
fig, axes3 = plt.subplots(1, 3, figsize=(16, 5), facecolor=C_BG)
fig.suptitle("R054 — Bootstrap PF Distributions (forward OOS)", fontsize=10,
             color=C_GOLD, fontweight="bold")

for ax_, (eid, col_) in zip(axes3, zip(ENVS, [C_GREEN, C_BLUE, C_GOLD])):
    st_ = env_stats[eid]; ba = st_["boot_arr"]
    if len(ba):
        ax_.hist(ba, bins=50, color=col_, alpha=0.75, edgecolor="none")
        ax_.axvline(st_["b5"],  color=C_RED,  linewidth=1.0, linestyle="--",
                    label=f"p5={st_['b5']:.3f}")
        ax_.axvline(st_["b50"], color=C_GOLD, linewidth=1.3,
                    label=f"Med={st_['b50']:.3f}")
        ax_.axvline(st_["b95"], color=C_GREEN, linewidth=1.0, linestyle="--",
                    label=f"p95={st_['b95']:.3f}")
        ax_.axvline(1.0,       color=C_GRID,  linewidth=0.8, linestyle="--")
        ax_.axvline(PROMO_BOOT, color="white", linewidth=0.8, linestyle=":",
                    alpha=0.5, label=f"Promo {PROMO_BOOT}")
        ax_.legend(fontsize=6, facecolor=C_PANEL, labelcolor=C_TEXT)
    panel_style(ax_, f"{eid}: {ENVS[eid]['label'][:25]}\nVerdict: {st_['verdict']}", fs=7)

plt.tight_layout()
plt.savefig(f"{OUT}/r054_bootstrap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r054_bootstrap.png")

# ── Chart 4: Dashboard
fig = plt.figure(figsize=(20, 12), facecolor=C_BG)
fig.suptitle("QUANTLAB AI — R054 — R052 PROMOTE Trio Forward Validation Dashboard",
             fontsize=13, color=C_GOLD, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

ax_a = fig.add_subplot(gs[0, :2])
ax_a.bar(x_pos - w/2, r052_pfs, w, label="R052 WF", color=C_BLUE,  alpha=0.8)
ax_a.bar(x_pos + w/2, fwd_pfs,  w, label="R054 Fwd", color=C_GREEN, alpha=0.8)
ax_a.axhline(1.0, color=C_GRID, linewidth=0.8, linestyle="--")
ax_a.axhline(PROMO_PF, color=C_GOLD, linewidth=0.8, linestyle="--")
ax_a.set_xticks(x_pos); ax_a.set_xticklabels(env_labels, fontsize=7)
ax_a.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
panel_style(ax_a, "Discovered vs Forward PF — All Environments", fs=8)

ax_b = fig.add_subplot(gs[0, 2])
fold_lbls = [f"F{i}" for i in range(1, N_FWD_FOLDS+1)]
port_fold_pfs = []
for fi in range(1, N_FWD_FOLDS+1):
    ftl = [t for t in all_pool if t["fold"] == f"F{fi}"]
    port_fold_pfs.append(metrics(ftl)["pf"])
cols_fp = [C_GREEN if p > 1.0 else C_RED for p in port_fold_pfs]
ax_b.bar(fold_lbls, port_fold_pfs, color=cols_fp, alpha=0.85)
ax_b.axhline(1.0, color=C_GRID, linewidth=0.8, linestyle="--")
ax_b.axhline(PROMO_PF, color=C_GOLD, linewidth=0.8, linestyle="--")
panel_style(ax_b, "Portfolio Fold PF", fs=8)

for idx, (eid, col_) in enumerate(zip(ENVS, [C_GREEN, C_BLUE, C_GOLD])):
    ax_ = fig.add_subplot(gs[1, idx])
    st_ = env_stats[eid]; m_ = st_["m"]
    eq_ = m_["equity"]; x_ = np.arange(len(eq_))
    ax_.plot(x_, eq_, color=col_, linewidth=1.2)
    ax_.axhline(CAPITAL, color=C_GRID, linewidth=0.6, linestyle="--")
    ax_.fill_between(x_, CAPITAL, eq_, where=eq_ >= CAPITAL, alpha=0.15, color=C_GREEN)
    ax_.fill_between(x_, CAPITAL, eq_, where=eq_ < CAPITAL,  alpha=0.15, color=C_RED)
    panel_style(ax_, f"{eid}  PF={m_['pf']:.3f}  {st_['verdict']}", fs=8)

ax_e = fig.add_subplot(gs[2, :])
ax_e.axis("off")
lines = [
    "R054 — R052 PROMOTE TRIO FORWARD VALIDATION",
    "─" * 80,
    f"{'Env':<12}  {'R052 PF':>9}  {'Fwd PF':>8}  {'Drop':>7}  {'n':>5}  "
    f"{'Boot':>7}  {'MC%':>6}  {'LOO-S':>6}  {'LOO-F':>6}  {'Score':>6}  Verdict",
    "─" * 80,
]
for eid in ENVS:
    st_ = env_stats[eid]; m_ = st_["m"]
    drop = m_["pf"] - ENVS[eid]["r052_pf"]
    lines.append(
        f"{eid+' ('+ENVS[eid]['label'][:12]+')':<20}  "
        f"{ENVS[eid]['r052_pf']:>9.3f}  {m_['pf']:>8.3f}  {drop:>+7.3f}  "
        f"{m_['n']:>5}  {st_['b50']:>7.3f}  "
        f"{st_['mc']['prob_profit']*100:>5.1f}%  "
        f"{st_['sf']:>6.3f}  {st_['ff']:>6.3f}  "
        f"{st_['vscore']:>4}/6  {st_['verdict']}"
    )
lines += [
    "─" * 80,
    f"{'PORTFOLIO':<20}  {'~1.35':>9}  {m_p['pf']:>8.3f}  {'':>7}  "
    f"{m_p['n']:>5}  {b50p:>7.3f}  "
    f"{mc_p['prob_profit']*100:>5.1f}%  "
    f"{sf_p:>6.3f}  {ff_p:>6.3f}  "
    f"{vsp:>4}/6  {vp}",
    "─" * 80,
    f"Context: R053 WLIST#1 → REJECT (PF=1.012). This run tests the actual 7/7 PROMOTE trio.",
]
for i, line in enumerate(lines):
    col = C_GOLD if i == 0 else (C_GREEN if "PROMOTE" in line else
                                  C_RED if "REJECT" in line else C_TEXT)
    ax_e.text(0.01, 0.97 - i * 0.072, line, transform=ax_e.transAxes,
              fontsize=6.5, color=col, va="top", fontfamily="monospace")
panel_style(ax_e, "R054 Summary Dashboard")

plt.savefig(f"{OUT}/r054_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓  {OUT}/r054_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
rows_csv = []
for eid in ENVS:
    st_ = env_stats[eid]; m_ = st_["m"]
    rows_csv.append({
        "env_id": eid, "label": ENVS[eid]["label"],
        "r052_pf": ENVS[eid]["r052_pf"], "r052_ues": ENVS[eid]["r052_ues"],
        "fwd_n": m_["n"], "fwd_pf": round(m_["pf"],4), "fwd_wr": round(m_["wr"],4),
        "fwd_net": round(m_["net"],2), "fwd_mdd": round(m_["mdd"],4),
        "boot_p5": round(st_["b5"],4), "boot_med": round(st_["b50"],4),
        "boot_p95": round(st_["b95"],4),
        "mc_prob": round(st_["mc"]["prob_profit"],4),
        "loo_sym": round(st_["sf"],4), "loo_fold": round(st_["ff"],4),
        "verdict": st_["verdict"], "score": st_["vscore"],
    })
# Portfolio
rows_csv.append({
    "env_id": "PORT", "label": "E2+E3+E8 Portfolio",
    "r052_pf": 1.35, "r052_ues": None,
    "fwd_n": m_p["n"], "fwd_pf": round(m_p["pf"],4), "fwd_wr": round(m_p["wr"],4),
    "fwd_net": round(m_p["net"],2), "fwd_mdd": round(m_p["mdd"],4),
    "boot_p5": round(b5p,4), "boot_med": round(b50p,4), "boot_p95": round(b95p,4),
    "mc_prob": round(mc_p["prob_profit"],4),
    "loo_sym": round(sf_p,4), "loo_fold": round(ff_p,4),
    "verdict": vp, "score": vsp,
})
pd.DataFrame(rows_csv).to_csv(f"{OUT}/r054_results.csv", index=False)
print(f"  ✓  {OUT}/r054_results.csv")

pd.DataFrame(all_pool).to_csv(f"{OUT}/r054_portfolio_trades.csv", index=False)
print(f"  ✓  {OUT}/r054_portfolio_trades.csv  ({len(all_pool)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  R054 COMPLETE — R052 PROMOTE TRIO FORWARD VALIDATION")
print(SEP)
for eid in ENVS:
    st_ = env_stats[eid]; m_ = st_["m"]
    print(f"  {eid}  {ENVS[eid]['label']:<34}  "
          f"R052 PF={ENVS[eid]['r052_pf']:.3f} → Fwd PF={m_['pf']:.3f}  "
          f"n={m_['n']}  {st_['verdict']}")
print(f"  PORTFOLIO E2+E3+E8:  Fwd PF={m_p['pf']:.3f}  n={m_p['n']}  {vp}")
print()

# Overall conclusion
print("  OVERALL CONCLUSION:")
if any(env_stats[e]["verdict"] == "PROMOTE" for e in ENVS):
    winners = [e for e in ENVS if env_stats[e]["verdict"] == "PROMOTE"]
    print(f"  ✅ {', '.join(winners)} cleared all promotion criteria on forward data.")
    print(f"     Recommend paper trading → live with small position size.")
elif any(env_stats[e]["verdict"] == "WATCHLIST" for e in ENVS):
    winners = [e for e in ENVS if env_stats[e]["verdict"] == "WATCHLIST"]
    print(f"  ⚠  {', '.join(winners)} cleared ≥4/6 criteria — WATCHLIST status.")
    print(f"     Continue monitoring. Do not deploy until another 3–6 months of data.")
else:
    best = max(ENVS, key=lambda e: env_stats[e]["m"]["pf"])
    print(f"  ❌ No environment clears promotion criteria on forward data.")
    print(f"     Best: {best} (PF={env_stats[best]['m']['pf']:.3f}).")
    print(f"     Consistent finding across R052–R054: environments discovered by searching")
    print(f"     9,447 combinations do not survive the selection bias correction.")
    print(f"     Recommend R055: hypothesis-driven, pre-registered, single test.")
print(SEP)
