"""
QUANTLAB AI — R067
Family C Dissection: DST_NR + ADX_ST + PBD_HI + ASI

Objective:
  Family C (R066) has a genuine edge — PF=1.492, n=721, all-5-folds profitable.
  But MDD=-15.1%, WR=42.7%, and max-loss-streak=36 make it uncomfortable.
  This study dissects Family C without touching thresholds or adding new conditions.
  The goal is to find the best-quality sub-expression that preserves the edge.

Approach (no optimisation, no new conditions, no threshold changes):
  1  Full baseline re-verification
  2  Condition ablation — all 4 three-condition subsets
  3  Symbol-tier breakdown — where does the edge concentrate?
  4  Intra-session breakdown — which ASI hours are strongest?
  5  Fold-by-fold profitability — is the edge stable or regime-specific?
  6  Win/loss distribution — is the edge coming from avoiding big losses?
  7  Condition pair interaction — which 2-way pairs drive results?
  8  Volatility-regime split — does PBD_HI work better in low/high vol?
  9  Build best sub-variant, full validation suite
  10 Final verdict — best Family C expression for production
"""

import os, sys, math, warnings, time, itertools
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

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
RESEARCH_ID  = "R067"
OUT          = CONFIG["OUTPUT_FOLDER"]
CACHE        = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL      = CONFIG["STARTING_CAPITAL"]
RR           = CONFIG["RISK_REWARD"]
IS_RATIO     = 0.80
MIN_BARS     = 2_000
N_FWD_FOLDS  = 5
N_BOOT       = 3_000
N_MC         = 3_000
N_PERM       = 1_000
RAND_SEED    = 42
TRADE_RISK   = 100.0

SEP  = "═" * 110
SEP2 = "─" * 90

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN CONDITION REGISTRY — same thresholds as R066, no changes
# ─────────────────────────────────────────────────────────────────────────────
COND_DEF = {
    "BBW_STRICT": ("bb_width",       "lt_q",      0.25),
    "RV_LO":      ("real_vol_20",    "lt_q",      0.33),
    "DST_NR":     ("ema_dist_pct",   "lt_q",      0.33),
    "PRG_VH":     ("prev_range_r",   "gt_q",      0.80),
    "RV_HI":      ("real_vol_20",    "gt_q",      0.67),
    "DST_MD":     ("ema_dist_pct",   "gt_q_pos",  0.60),
    "ADX_WK":     ("adx14",          "lt_q",      0.33),
    "LON":        ("hour_utc",       "hour_rng",  (7, 14)),
    "ADX_ST":     ("adx14",          "gt_q",      0.67),
    "PBD_HI":     ("prev_body_r",    "gt_q",      0.67),
    "ASI":        ("hour_utc",       "hour_rng",  (0,  6)),
    # Extra conditions for section 8 volatility-regime split (frozen thresholds)
    "ATR_LO":     ("atr_rank",       "lt_q",      0.33),
    "ATR_HI":     ("atr_rank",       "gt_q",      0.67),
    "RV_MD_LO":   ("real_vol_20",    "lt_q",      0.50),   # below median
    "BBW_LO":     ("bb_width",       "lt_q",      0.33),
    "BBW_HI":     ("bb_width",       "gt_q",      0.67),
    "ADX_MD":     ("adx14",          "gt_q",      0.50),   # above median (less strict than ST)
    "PRG_HI":     ("prev_range_r",   "gt_q",      0.67),
}

# Full Family C — frozen
FAM_C_BASE = ("DST_NR", "ADX_ST", "PBD_HI", "ASI")

# All 3-condition ablations of Family C
ABLATIONS = {
    "C_no_DST": ("ADX_ST",  "PBD_HI", "ASI"),          # drop DST_NR
    "C_no_ADX": ("DST_NR",  "PBD_HI", "ASI"),          # drop ADX_ST
    "C_no_PBD": ("DST_NR",  "ADX_ST", "ASI"),          # drop PBD_HI
    "C_no_ASI": ("DST_NR",  "ADX_ST", "PBD_HI"),       # drop ASI (all sessions)
    "C_FULL":   FAM_C_BASE,                              # full 4-condition baseline
}

# Colour palette
C_BG    = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT  = "#e0e0e0"
C_GRID  = "#2a2a2a"; C_GREEN = "#00c896"; C_RED   = "#e05050"
C_GOLD  = "#f5a623"; C_BLUE  = "#4a9eff"; C_PURP  = "#9b59b6"
C_TEAL  = "#1abc9c"; C_ORAN  = "#e67e22"
PALETTE = [C_GREEN, C_GOLD, C_BLUE, C_PURP, C_TEAL, C_ORAN, C_RED,
           "#3498db","#e74c3c","#f39c12","#2ecc71","#e91e63","#00bcd4",
           "#ff5722","#8bc34a","#795548","#607d8b","#ff9800","#673ab7"]

plt.rcParams.update({
    "figure.facecolor":C_BG, "axes.facecolor":C_PANEL,
    "text.color":C_TEXT, "axes.labelcolor":C_TEXT,
    "xtick.color":C_TEXT, "ytick.color":C_TEXT,
    "axes.edgecolor":C_GRID, "grid.color":C_GRID, "font.family":"monospace",
})

def style_ax(ax):
    ax.set_facecolor(C_PANEL); ax.grid(True, ls="--", lw=0.4, color=C_GRID)
    for sp in ax.spines.values(): sp.set_edgecolor(C_GRID)

def save_fig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    return p

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]; o = df["open"]
    df["ema200"]       = calc_ema(c, 200)
    df["ema50"]        = calc_ema(c, 50)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank"]     = df["atr14"].rolling(100).rank(pct=True) * 100
    bb_mid             = c.rolling(20).mean()
    bb_std             = c.rolling(20).std(ddof=0)
    df["bb_width"]     = (bb_std * 2) / bb_mid.replace(0, np.nan) * 100.0
    df["real_vol_20"]  = c.pct_change().rolling(20).std() * 100.0
    df["rel_vol_rank"] = v.rolling(50).rank(pct=True) * 100
    df["ema200_slope"] = df["ema200"].diff(5) / df["ema200"].shift(5).replace(0, np.nan) * 100
    ema200_safe        = df["ema200"].replace(0, np.nan)
    df["ema_dist_pct"] = (c - ema200_safe) / ema200_safe * 100.0
    prev_range         = (h.shift(1) - l.shift(1)).abs()
    prev_body          = (c.shift(1) - o.shift(1)).abs()
    df["prev_range_r"] = prev_range / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_r"]  = prev_body  / c.shift(1).replace(0, np.nan) * 100.0
    df["prev_body_pct"]= prev_body  / prev_range.replace(0, np.nan)
    df["hour_utc"]     = pd.to_datetime(df.index).hour
    df["adx14"]        = calc_adx(df, 14)
    df.dropna(subset=["ema200","atr14","real_vol_20","adx14","bb_width"], inplace=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def apply_cond(df, cid, thresholds):
    col, direction, param = COND_DEF[cid]
    if direction == "hour_rng":
        lo, hi = param
        if lo < hi: return (df["hour_utc"] >= lo) & (df["hour_utc"] < hi)
        else:       return (df["hour_utc"] >= lo) | (df["hour_utc"] < hi)
    vals = df[col]
    if direction == "lt_q":
        return vals < thresholds.get(f"{cid}_q", np.nan)
    elif direction == "gt_q":
        return vals > thresholds.get(f"{cid}_q", np.nan)
    elif direction == "gt_q_pos":
        t = thresholds.get(f"{cid}_q", np.nan)
        return (vals > t) & (vals > 0)
    elif direction == "lt_fixed": return vals < param
    elif direction == "gt_fixed": return vals > param
    return pd.Series(False, index=df.index)

def compute_thresholds(df_is, cids):
    out = {}
    for cid in cids:
        col, direction, param = COND_DEF[cid]
        if direction in ("lt_q","gt_q","gt_q_pos"):
            vals = df_is[col].dropna()
            if direction == "gt_q_pos":
                vp = vals[vals > 0]
                t  = float(vp.quantile(param)) if len(vp) > 10 else float(vals.quantile(param))
            else:
                t  = float(vals.quantile(param))
            out[f"{cid}_q"] = t
    return out

def entry_gate(df):
    vol_avg = df["vol"].rolling(20).mean()
    return (df["vol"] > 1.5 * vol_avg) & (df["close"] > df["open"]) & (df["close"] > df["close"].shift(1))

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def safe_pf(gw, gl):
    if gl == 0: return 999.0 if gw > 0 else 1.0
    return gw / gl

def metrics_from_pnls(pnls):
    pnls = np.asarray(pnls, dtype=float)
    n = len(pnls)
    if n == 0:
        return dict(pf=0.0,wr=0.0,n=0,net=0.0,avg=0.0,mdd=0.0,mdd_abs=0.0,equity=np.array([CAPITAL]))
    wins = pnls[pnls>0]; losses = pnls[pnls<0]
    pf   = safe_pf(wins.sum(), abs(losses.sum()))
    wr   = len(wins)/n
    eq   = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnls)])
    peak = np.maximum.accumulate(eq)
    dd_a = eq - peak
    worst_idx = np.argmin(dd_a)
    mdd  = float(dd_a[worst_idx] / peak[worst_idx]) if peak[worst_idx] != 0 else 0.0
    return dict(pf=pf,wr=wr,n=n,net=float(pnls.sum()),avg=float(pnls.mean()),
                mdd=mdd,mdd_abs=float(abs(dd_a.min())),equity=eq)

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=RAND_SEED):
    rng  = np.random.default_rng(seed); pnls = np.asarray(pnls)
    bpfs = [safe_pf((s:=rng.choice(pnls,len(pnls),replace=True))[s>0].sum(),
                    abs(s[s<0].sum())) for _ in range(n_iter)]
    arr  = np.array(bpfs)
    return dict(med=float(np.percentile(arr,50)), p5=float(np.percentile(arr,5)),
                p95=float(np.percentile(arr,95)), pct_above1=float((arr>1).mean()), arr=arr)

def monte_carlo(pnls, n_iter=N_MC, seed=RAND_SEED+1):
    rng   = np.random.default_rng(seed); pnls = np.asarray(pnls)
    finals= []
    for _ in range(n_iter):
        s  = rng.choice(pnls,len(pnls),replace=True)
        finals.append(float(CAPITAL + np.cumsum(s)[-1]))
    finals = np.array(finals)
    return dict(prob_profit=float((finals>CAPITAL).mean()),
                median=float(np.median(finals)),
                p5=float(np.percentile(finals,5)),
                p95=float(np.percentile(finals,95)),
                finals=finals)

def ues_score(pf,wr,n,mdd,sym_fl,fold_fl,boot_p5,mc_prob):
    pf_s   = min(100,max(0,(pf-1.0)/0.80*35))
    wr_s   = min(100,max(0,(wr-0.30)/0.25*25))
    n_s    = min(100,max(0,(math.log1p(n)/math.log1p(200))*15))
    mdd_s  = min(100,max(0,(1-abs(mdd)/0.30)*10))
    sym_s  = min(100,max(0,(sym_fl-1.0)/0.50*5))
    fold_s = min(100,max(0,(fold_fl-1.0)/0.50*5))
    boot_s = min(100,max(0,(boot_p5-1.0)/0.50*5))
    mc_s   = min(100,max(0,mc_prob*5))
    return round(pf_s+wr_s+n_s+mdd_s+sym_s+fold_s+boot_s+mc_s,1)

def ulcer_index(equity):
    eq  = np.asarray(equity,dtype=float); pk = np.maximum.accumulate(eq)
    return float(np.sqrt(np.mean(((eq-pk)/pk*100)**2)))

def recovery_factor(net,mdd_abs):
    return abs(net)/mdd_abs if mdd_abs>0 else 999.0

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def backtest(cids, df_feat, sym="?"):
    n_bars  = len(df_feat)
    is_end  = int(n_bars * IS_RATIO)
    df_is   = df_feat.iloc[:is_end]
    df_oos  = df_feat.iloc[is_end:]
    oos_len = len(df_oos)
    fold_sz = max(1, oos_len // N_FWD_FOLDS)

    thresholds = compute_thresholds(df_is, cids)

    gate  = entry_gate(df_feat).iloc[is_end:]
    masks = [apply_cond(df_feat.iloc[is_end:], c, thresholds) for c in cids]
    sig   = masks[0].copy()
    for m in masks[1:]: sig = sig & m
    sig   = sig & gate

    trades = []
    for fi in range(N_FWD_FOLDS):
        sl      = slice(fi*fold_sz, (fi+1)*fold_sz if fi < N_FWD_FOLDS-1 else oos_len)
        fold_sig= sig.iloc[sl]
        fold_df = df_oos.iloc[sl]
        for idx in fold_df.index[fold_sig.values]:
            pos    = fold_df.index.get_loc(idx)
            exit_c = fold_df["close"].iloc[pos+1] if pos+1 < len(fold_df) else fold_df["close"].iloc[pos]
            entry_c= fold_df["close"].loc[idx]
            is_win = exit_c > entry_c
            row    = fold_df.loc[idx]
            trades.append({
                "symbol":    sym,
                "entry_idx": pos + fi*fold_sz,   # OOS-relative position (for sequencing)
                "exit_pnl":  TRADE_RISK*RR if is_win else -TRADE_RISK,
                "is_win":    is_win,
                "fold":      fi+1,
                "hour_utc":  int(row["hour_utc"]) if "hour_utc" in row.index else 0,
                "atr_rank":  float(row.get("atr_rank",50)),
                "real_vol":  float(row.get("real_vol_20",1)),
                "bb_width":  float(row.get("bb_width",1)),
                "adx14":     float(row.get("adx14",25)),
                "ema_dist":  float(row.get("ema_dist_pct",0)),
                "prev_body": float(row.get("prev_body_r",0)),
                "close":     float(entry_c),
            })
    return trades

def run_all_symbols(cids, data):
    all_trades = []
    for sym, df_raw in data.items():
        try:
            df_f = add_features(df_raw)
            all_trades.extend(backtest(cids, df_f, sym))
        except Exception:
            pass
    all_trades.sort(key=lambda t: (t["symbol"], t["entry_idx"]))
    return all_trades

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fold_breakdown(trades):
    by_fold = defaultdict(list)
    for t in trades: by_fold[t["fold"]].append(t["exit_pnl"])
    return {f: dict(pf=safe_pf((p:=np.array(v))[p>0].sum(),abs(p[p<0].sum())),
                    n=len(v), wr=float((np.array(v)>0).mean()))
            for f, v in sorted(by_fold.items())}

def sym_breakdown(trades):
    by_sym = defaultdict(list)
    for t in trades: by_sym[t["symbol"]].append(t["exit_pnl"])
    return {s: dict(pf=safe_pf((p:=np.array(v))[p>0].sum(),abs(p[p<0].sum())),
                    n=len(v), wr=float((np.array(v)>0).mean()))
            for s, v in by_sym.items()}

def loo_sym(trades):
    syms = list({t["symbol"] for t in trades}); floors = []
    for s in syms:
        p = np.array([t["exit_pnl"] for t in trades if t["symbol"]!=s])
        if len(p) >= 3: floors.append(safe_pf(p[p>0].sum(),abs(p[p<0].sum())))
    return float(min(floors)) if floors else 0.0

def loo_fold(trades):
    folds = list({t["fold"] for t in trades}); floors = []
    for f in folds:
        p = np.array([t["exit_pnl"] for t in trades if t["fold"]!=f])
        if len(p) >= 3: floors.append(safe_pf(p[p>0].sum(),abs(p[p<0].sum())))
    return float(min(floors)) if floors else 0.0

def max_streak(trades, win=True):
    pnls = [t["exit_pnl"] for t in trades]
    cur = mx = 0
    for p in pnls:
        cur = cur+1 if (p>0)==win else 0
        mx  = max(mx, cur)
    return mx

def full_eval(trades, label):
    pnls = np.array([t["exit_pnl"] for t in trades])
    m    = metrics_from_pnls(pnls)
    if m["n"] < 5:
        return dict(label=label, **m, boot={}, mc={}, ues=0.0,
                    sym_fl=0.0, fold_fl=0.0, folds={}, syms={},
                    ulcer=0.0, rf=0.0, win_streak=0, loss_streak=0, trades=trades)
    boot = bootstrap_pf(pnls)
    mc   = monte_carlo(pnls)
    folds= fold_breakdown(trades); syms = sym_breakdown(trades)
    sym_pfs  = [v["pf"] for v in syms.values()  if v["n"]>=3]
    fold_pfs = [v["pf"] for v in folds.values() if v["n"]>=3]
    sym_fl   = float(min(sym_pfs))  if sym_pfs  else 0.0
    fold_fl  = float(min(fold_pfs)) if fold_pfs else 0.0
    ues = ues_score(m["pf"],m["wr"],m["n"],m["mdd"],sym_fl,fold_fl,
                    boot["p5"],mc["prob_profit"])
    ul  = ulcer_index(m["equity"])
    rf  = recovery_factor(m["net"],m["mdd_abs"])
    ws  = max_streak(trades, True); ls = max_streak(trades, False)
    return dict(label=label, **m, boot=boot, mc=mc, ues=ues,
                sym_fl=sym_fl, fold_fl=fold_fl, folds=folds, syms=syms,
                ulcer=ul, rf=rf, win_streak=ws, loss_streak=ls, trades=trades)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    data = {}
    for fn in sorted(os.listdir(CACHE)):
        if not fn.endswith("_1H.parquet"): continue
        sym = fn.replace("_1H.parquet","")
        try:
            df = pd.read_parquet(os.path.join(CACHE,fn))
            df.index = pd.to_datetime(df.index,utc=True); df.sort_index(inplace=True)
            for col in ["open","high","low","close"]:
                if col in df.columns: df[col] = pd.to_numeric(df[col],errors="coerce")
            if "vol" not in df.columns and "volume" in df.columns:
                df.rename(columns={"volume":"vol"},inplace=True)
            df.dropna(subset=["open","high","low","close","vol"],inplace=True)
            if len(df) >= MIN_BARS: data[sym] = df
        except Exception: pass
    return data

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL TIER CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
TIER1 = {"BTC_USDT_SWAP","ETH_USDT_SWAP","BNB_USDT_SWAP","SOL_USDT_SWAP","XRP_USDT_SWAP"}
TIER2 = {"DOGE_USDT_SWAP","ADA_USDT_SWAP","AVAX_USDT_SWAP","DOT_USDT_SWAP",
         "LINK_USDT_SWAP","LTC_USDT_SWAP","BCH_USDT_SWAP","ATOM_USDT_SWAP",
         "UNI_USDT_SWAP","NEAR_USDT_SWAP","APT_USDT_SWAP","ARB_USDT_SWAP",
         "OP_USDT_SWAP","MATIC_USDT_SWAP"}

def tier(sym):
    if sym in TIER1: return "T1-Large"
    if sym in TIER2: return "T2-Mid"
    return "T3-Small"

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}")
print(f"  Family C Dissection: DST_NR + ADX_ST + PBD_HI + ASI")
print(SEP); print()
t0 = time.time()

print("  Loading data …")
data = load_data()
print(f"  Symbols: {len(data)}")
print()

saved_charts = []

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FULL BASELINE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  SECTION 1 — FULL BASELINE VERIFICATION"); print(SEP2)

base_trades = run_all_symbols(FAM_C_BASE, data)
ev_base     = full_eval(base_trades, "C_FULL")

pnls_base = np.array([t["exit_pnl"] for t in base_trades])
print(f"  DST_NR+ADX_ST+PBD_HI+ASI:")
print(f"  PF={ev_base['pf']:.3f}  WR={ev_base['wr']:.1%}  n={ev_base['n']}  "
      f"MDD={ev_base['mdd']:.1%}  UES={ev_base['ues']:.1f}")
print(f"  Boot P5={ev_base['boot'].get('p5',0):.3f}  "
      f"P50={ev_base['boot'].get('med',0):.3f}  "
      f"MC P(profit)={ev_base['mc'].get('prob_profit',0):.1%}")
print(f"  LOO-sym={ev_base['sym_fl']:.3f}  LOO-fold={ev_base['fold_fl']:.3f}")
print(f"  Win streak={ev_base['win_streak']}  Loss streak={ev_base['loss_streak']}")
folds_str = {f: f"{v['pf']:.3f}(n={v['n']})" for f,v in ev_base["folds"].items()}
print(f"  Folds: {folds_str}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONDITION ABLATION
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  SECTION 2 — CONDITION ABLATION (3-condition subsets)"); print(SEP2)

ablation_results = {}
for abl_id, cids in ABLATIONS.items():
    print(f"  Running {abl_id}: {'+'.join(cids)} …")
    trades = run_all_symbols(cids, data)
    ev     = full_eval(trades, abl_id)
    ablation_results[abl_id] = ev

print()
print(f"  {'Variant':<14}  {'Conditions':<30}  {'PF':>6}  {'WR':>6}  "
      f"{'n':>5}  {'MDD':>7}  {'UES':>6}  {'BtP5':>6}  {'LoStr':>5}")
print(f"  {'-'*14}  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*5}")
for abl_id, cids in ABLATIONS.items():
    ev = ablation_results[abl_id]
    cond_str = "+".join(cids)
    print(f"  {abl_id:<14}  {cond_str:<30}  {ev['pf']:>6.3f}  {ev['wr']:>6.1%}  "
          f"{ev['n']:>5}  {ev['mdd']:>7.1%}  {ev['ues']:>6.1f}  "
          f"{ev['boot'].get('p5',0):>6.3f}  {ev['loss_streak']:>5}")
print()

# Find best ablation
best_abl = max(ablation_results.items(),
               key=lambda x: x[1]["ues"] * (1 if x[1]["n"]>=50 else 0))
print(f"  Best ablation by UES (n≥50): {best_abl[0]}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SYMBOL-TIER BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  SECTION 3 — SYMBOL-TIER BREAKDOWN"); print(SEP2)

by_tier = defaultdict(list)
for t in base_trades:
    by_tier[tier(t["symbol"])].append(t)

print(f"\n  Full C per tier:")
tier_evals = {}
for t_label in ["T1-Large","T2-Mid","T3-Small"]:
    t_trades = by_tier[t_label]
    p = np.array([t["exit_pnl"] for t in t_trades])
    pf_ = safe_pf(p[p>0].sum(),abs(p[p<0].sum())) if len(p) else 0.0
    wr_ = float((p>0).mean()) if len(p) else 0.0
    tier_evals[t_label] = dict(pf=pf_,wr=wr_,n=len(t_trades))
    print(f"  {t_label:<12}: PF={pf_:.3f}  WR={wr_:.1%}  n={len(t_trades)}")

# Symbol-by-symbol for full C
sym_ev = ev_base["syms"]
sym_sorted = sorted(sym_ev.items(), key=lambda x: x[1]["pf"], reverse=True)
top_syms  = [(s,v) for s,v in sym_sorted if v["n"]>=5]
print(f"\n  Top-10 symbols by PF (n≥5):")
for s, v in top_syms[:10]:
    print(f"    {s:<22}: PF={v['pf']:.3f}  WR={v['wr']:.1%}  n={v['n']}")
print(f"\n  Bottom-5 symbols by PF (n≥5):")
for s, v in top_syms[-5:]:
    print(f"    {s:<22}: PF={v['pf']:.3f}  WR={v['wr']:.1%}  n={v['n']}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INTRA-SESSION BREAKDOWN (ASI hours 0–6 UTC)
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  SECTION 4 — INTRA-SESSION BREAKDOWN (ASI 00:00–06:00 UTC)"); print(SEP2)

# Group by hour UTC
hour_groups = defaultdict(list)
for t in base_trades:
    hour_groups[t["hour_utc"]].append(t)

print(f"\n  {'Hour UTC':<10}  {'PF':>6}  {'WR':>6}  {'n':>5}  {'Contrib%':>9}")
print(f"  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*9}")
hour_results = {}
for h in sorted(hour_groups.keys()):
    tl = hour_groups[h]
    p  = np.array([t["exit_pnl"] for t in tl])
    pf_ = safe_pf(p[p>0].sum(),abs(p[p<0].sum()))
    wr_ = float((p>0).mean()) if len(p) else 0.0
    contrib = len(tl) / len(base_trades) * 100
    hour_results[h] = dict(pf=pf_,wr=wr_,n=len(tl))
    print(f"  {h:02d}:00–{(h+1):02d}:00  {pf_:>6.3f}  {wr_:>6.1%}  {len(tl):>5}  {contrib:>8.1f}%")

# Best 3-hour window within ASI
best_window = None; best_window_pf = 0.0
for start_h in range(0, 6):
    for end_h in range(start_h+1, 7):
        window_trades = [t for t in base_trades if start_h <= t["hour_utc"] < end_h]
        if len(window_trades) < 30: continue
        p = np.array([t["exit_pnl"] for t in window_trades])
        pf_ = safe_pf(p[p>0].sum(),abs(p[p<0].sum()))
        if pf_ > best_window_pf and len(window_trades) >= 50:
            best_window_pf = pf_
            best_window = (start_h, end_h, len(window_trades), pf_)

if best_window:
    s,e,nn,pf_ = best_window
    print(f"\n  Best sub-window: {s:02d}:00–{e:02d}:00  PF={pf_:.3f}  n={nn}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FOLD-BY-FOLD PROFITABILITY
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  SECTION 5 — FOLD-BY-FOLD PROFITABILITY"); print(SEP2)

print(f"\n  Full C baseline folds:")
winning_folds = sum(1 for v in ev_base["folds"].values() if v["pf"] >= 1.0)
for f_, fd in ev_base["folds"].items():
    status = "✓" if fd["pf"] >= 1.0 else "✗"
    print(f"  {status} Fold {f_}: PF={fd['pf']:.3f}  WR={fd['wr']:.1%}  n={fd['n']}")
print(f"  → {winning_folds}/5 folds profitable")
print()

# Ablation fold comparison
print(f"  Ablation fold comparison (PF by fold):")
print(f"  {'Variant':<14}  {'F1':>6}  {'F2':>6}  {'F3':>6}  {'F4':>6}  {'F5':>6}  {'Win/5':>6}")
for abl_id in ABLATIONS:
    ev   = ablation_results[abl_id]
    fds  = ev["folds"]
    row  = [fds.get(f,{}).get("pf",0.0) for f in range(1,6)]
    wins = sum(1 for x in row if x >= 1.0)
    fstr = "  ".join(f"{x:.3f}" for x in row)
    print(f"  {abl_id:<14}  {fstr}  {wins}/5")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — WIN/LOSS DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  SECTION 6 — WIN/LOSS DISTRIBUTION ANALYSIS"); print(SEP2)

pnls_b = np.array([t["exit_pnl"] for t in base_trades])
wins_b = pnls_b[pnls_b > 0]; losses_b = pnls_b[pnls_b < 0]
print(f"\n  Trade PnL profile (Family C full):")
print(f"  Wins: {len(wins_b)} trades  Gross: ${wins_b.sum():,.1f}  Avg: ${wins_b.mean():.1f}")
print(f"  Losses: {len(losses_b)} trades  Gross: ${abs(losses_b.sum()):,.1f}  Avg: ${losses_b.mean():.1f}")
print(f"  Expectancy per trade: ${pnls_b.mean():.2f}")

# Streaks distribution — run-length encoding
def streak_counts(trades):
    cur_win = cur_loss = 0
    win_runs = []; loss_runs = []
    for t in trades:
        if t["exit_pnl"] > 0:
            cur_win += 1
            if cur_loss > 0: loss_runs.append(cur_loss); cur_loss = 0
        else:
            cur_loss += 1
            if cur_win > 0: win_runs.append(cur_win); cur_win = 0
    if cur_win > 0: win_runs.append(cur_win)
    if cur_loss > 0: loss_runs.append(cur_loss)
    return win_runs, loss_runs

win_runs, loss_runs = streak_counts(base_trades)
print(f"\n  Streak distribution:")
for thresh in [3, 5, 8, 10, 15, 20]:
    wc = sum(1 for r in loss_runs if r >= thresh)
    print(f"  Loss streaks ≥ {thresh:2d}: {wc} occurrences")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — CONDITION PAIR INTERACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  SECTION 7 — CONDITION PAIR INTERACTIONS"); print(SEP2)
print("  (Which 2-way pairs within Family C drive the most value?)")

pairs_2way = list(itertools.combinations(FAM_C_BASE, 2))
pair_results = {}
for pair in pairs_2way:
    trades_p = run_all_symbols(pair, data)
    p        = np.array([t["exit_pnl"] for t in trades_p])
    pf_      = safe_pf(p[p>0].sum(),abs(p[p<0].sum())) if len(p) else 0.0
    wr_      = float((p>0).mean()) if len(p) else 0.0
    pair_results[pair] = dict(pf=pf_,wr=wr_,n=len(trades_p))

print(f"\n  {'Pair':<25}  {'PF':>6}  {'WR':>6}  {'n':>6}")
print(f"  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*6}")
for pair, v in sorted(pair_results.items(), key=lambda x: x[1]["pf"], reverse=True):
    print(f"  {'+'.join(pair):<25}  {v['pf']:>6.3f}  {v['wr']:>6.1%}  {v['n']:>6}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — VOLATILITY-REGIME SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  SECTION 8 — VOLATILITY-REGIME SPLIT"); print(SEP2)
print("  (Does Family C work better in specific vol regimes?)")
print("  (No new conditions — this identifies WHEN to expect better/worse performance)")

# Split base trades by quartile of atr_rank and real_vol
def split_by_feature(trades, feat, quantiles=(0.25,0.50,0.75)):
    vals = np.array([t[feat] for t in trades])
    q_vals = np.quantile(vals, quantiles)
    bins   = [-np.inf] + list(q_vals) + [np.inf]
    labels = ["Q1(low)","Q2","Q3","Q4(high)"]
    groups = defaultdict(list)
    for t, v in zip(trades, vals):
        for i in range(len(bins)-1):
            if bins[i] <= v < bins[i+1]:
                groups[labels[i]].append(t); break
    return groups

for feat_name, feat_key in [("ATR Rank","atr_rank"),
                              ("Real Vol","real_vol"),
                              ("ADX14","adx14"),
                              ("Prev Body","prev_body")]:
    groups = split_by_feature(base_trades, feat_key)
    print(f"\n  Split by {feat_name}:")
    print(f"  {'Quartile':<12}  {'PF':>6}  {'WR':>6}  {'n':>5}")
    for label in ["Q1(low)","Q2","Q3","Q4(high)"]:
        tl = groups[label]; p = np.array([t["exit_pnl"] for t in tl])
        pf_ = safe_pf(p[p>0].sum(),abs(p[p<0].sum())) if len(p) else 0.0
        wr_ = float((p>0).mean()) if len(p) else 0.0
        print(f"  {label:<12}  {pf_:>6.3f}  {wr_:>6.1%}  {len(tl):>5}")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — BUILD BEST SUB-VARIANT, FULL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  SECTION 9 — CANDIDATE COMPARISON & FULL VALIDATION"); print(SEP2)

# Identify the best-quality ablation with n >= 100
candidates_to_validate = []
for abl_id, cids in ABLATIONS.items():
    ev = ablation_results[abl_id]
    if ev["n"] >= 100:
        candidates_to_validate.append((abl_id, cids, ev))

# Sort by UES * (1 + boost for lower loss_streak)
def quality_score(ev):
    streak_penalty = max(0, ev["loss_streak"] - 15) * 0.5
    mdd_penalty    = max(0, abs(ev["mdd"]) - 0.10) * 100
    return ev["ues"] - streak_penalty - mdd_penalty

candidates_to_validate.sort(key=lambda x: quality_score(x[2]), reverse=True)

print(f"\n  Candidates with n≥100, sorted by quality score:")
print(f"  {'Variant':<14}  {'QScore':>7}  {'PF':>6}  {'WR':>6}  {'n':>5}  "
      f"{'MDD':>7}  {'UES':>6}  {'LoStr':>5}  {'BtP5':>6}")
print(f"  {'-'*14}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*5}  "
      f"{'-'*7}  {'-'*6}  {'-'*5}  {'-'*6}")
for abl_id, cids, ev in candidates_to_validate:
    qs = quality_score(ev)
    print(f"  {abl_id:<14}  {qs:>7.1f}  {ev['pf']:>6.3f}  {ev['wr']:>6.1%}  {ev['n']:>5}  "
          f"{ev['mdd']:>7.1%}  {ev['ues']:>6.1f}  {ev['loss_streak']:>5}  "
          f"{ev['boot'].get('p5',0):>6.3f}")
print()

# Full validation on top candidate (if different from full)
if candidates_to_validate:
    top_id, top_cids, top_ev = candidates_to_validate[0]
    if top_id != "C_FULL":
        print(f"  → Re-running full validation on top candidate: {top_id}")
        top_trades = run_all_symbols(top_cids, data)
        top_ev     = full_eval(top_trades, top_id)
        # Extra: LOO
        top_ev["sym_fl_full"]  = loo_sym(top_trades)
        top_ev["fold_fl_full"] = loo_fold(top_trades)
        print(f"  PF={top_ev['pf']:.3f}  WR={top_ev['wr']:.1%}  n={top_ev['n']}  "
              f"MDD={top_ev['mdd']:.1%}  UES={top_ev['ues']:.1f}")
        print(f"  Boot P5={top_ev['boot'].get('p5',0):.3f}  "
              f"P50={top_ev['boot'].get('med',0):.3f}  "
              f"MC={top_ev['mc'].get('prob_profit',0):.1%}")
        print(f"  LOO-sym={top_ev['sym_fl']:.3f}  LOO-fold={top_ev['fold_fl']:.3f}")
        print(f"  Win streak={top_ev['win_streak']}  Loss streak={top_ev['loss_streak']}")
        folds_str2 = {f: f"{v['pf']:.3f}(n={v['n']})" for f,v in top_ev["folds"].items()}
        print(f"  Folds: {folds_str2}")
    else:
        top_trades = base_trades
        print(f"  → Full Family C (baseline) is already the best candidate")
else:
    top_id, top_cids, top_ev, top_trades = "C_FULL", FAM_C_BASE, ev_base, base_trades
    print("  No ablation improved quality. Full Family C is the best available variant.")

print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════
print(SEP); print("  SECTION 10 — FINAL VERDICT"); print(SEP2)

# Determine which condition adds most value
pf_full = ev_base["pf"]
ablation_deltas = {}
for abl_id, cids in ABLATIONS.items():
    if abl_id == "C_FULL": continue
    ev    = ablation_results[abl_id]
    drop  = set(FAM_C_BASE) - set(cids)   # which condition was removed
    delta = ev["pf"] - pf_full             # positive = removing it helps
    ablation_deltas[list(drop)[0]] = dict(
        removed=list(drop)[0], subset=abl_id, pf=ev["pf"],
        delta=delta, n=ev["n"], mdd=ev["mdd"], loss_streak=ev["loss_streak"])

print(f"\n  Condition contribution analysis:")
print(f"  (Delta = change in PF when this condition is REMOVED from Family C)")
print(f"  (Positive delta = condition was hurting; Negative = condition was helping)")
print(f"  {'Condition':<12}  {'Removed PF':>10}  {'Delta':>8}  {'n':>5}  {'MDD':>7}  {'LoStr':>6}")
print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*5}  {'-'*7}  {'-'*6}")
for cid, info in sorted(ablation_deltas.items(), key=lambda x: -x[1]["delta"]):
    arrow = "↑ HELPS remove" if info["delta"] > 0.05 else ("✓ HURTS remove" if info["delta"] < -0.05 else "≈ neutral")
    print(f"  {cid:<12}  {info['pf']:>10.3f}  {info['delta']:>+8.3f}  "
          f"{info['n']:>5}  {info['mdd']:>7.1%}  {info['loss_streak']:>6}  {arrow}")

# Recommendation
best_rec = max(ablation_deltas.items(), key=lambda x: (
    quality_score(ablation_results[x[1]["subset"]]) if x[1]["n"]>=100 else -999
), default=None)

if best_rec and ablation_results[best_rec[1]["subset"]]["n"] >= 100:
    rec_ev   = ablation_results[best_rec[1]["subset"]]
    rec_cids = ABLATIONS[best_rec[1]["subset"]]
    rec_label= "+".join(rec_cids)
    print(f"\n  ─────────────────────────────────────────────────────────")
    print(f"  RECOMMENDATION: {best_rec[1]['subset']} = {rec_label}")
    print(f"  PF={rec_ev['pf']:.3f}  WR={rec_ev['wr']:.1%}  n={rec_ev['n']}  "
          f"MDD={rec_ev['mdd']:.1%}  Loss streak={rec_ev['loss_streak']}")
    print(f"  vs Full C: PF={pf_full:.3f}  MDD={ev_base['mdd']:.1%}  "
          f"Loss streak={ev_base['loss_streak']}")
    improvement_pf  = rec_ev["pf"]  - pf_full
    improvement_mdd = ev_base["mdd"] - rec_ev["mdd"]
    print(f"  PF improvement: {improvement_pf:+.3f}  "
          f"MDD improvement: {improvement_mdd:+.1%}")
    verdict = "ADOPT" if rec_ev["pf"] > pf_full and rec_ev["n"] >= 100 else "KEEP_FULL"
else:
    rec_label  = "+".join(FAM_C_BASE)
    rec_cids   = FAM_C_BASE
    rec_ev     = ev_base
    verdict    = "KEEP_FULL"
    print(f"\n  No ablation strictly improves Family C. Retain full 4-condition version.")
    print(f"  Full C: PF={pf_full:.3f}  MDD={ev_base['mdd']:.1%}  n={ev_base['n']}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print(SEP); print("  GENERATING CHARTS …"); print(SEP2)

# ── Chart 1: Ablation Dashboard ───────────────────────────────────────────────
fig1 = plt.figure(figsize=(18, 12))
fig1.suptitle("R067 — Family C Dissection: Condition Ablation",
              fontsize=13, fontweight="bold", color=C_TEXT, y=0.98)
gs1  = gridspec.GridSpec(2, 4, figure=fig1, hspace=0.45, wspace=0.35)

abl_labels = list(ABLATIONS.keys())
abl_colors = [C_GREEN, C_GOLD, C_BLUE, C_PURP, C_RED]

def bar_panel(ax, labels, vals, colors, title, ylabel, refline=None, pct=False):
    style_ax(ax)
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor=C_BG, width=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=35, ha="right")
    ax.set_title(title, fontsize=8, color=C_TEXT)
    ax.set_ylabel(ylabel, fontsize=7, color=C_TEXT)
    if refline is not None: ax.axhline(refline, color=C_RED, lw=1, ls="--")
    for bar, val in zip(bars, vals):
        fmt = f"{val:.1f}%" if pct else (f"{val:.3f}" if abs(val)<10 else f"{val:.0f}")
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+abs(max(vals,default=1))*0.01,
                fmt, ha="center", va="bottom", fontsize=6.5, color=C_TEXT)

# PF
ax = fig1.add_subplot(gs1[0,0])
bar_panel(ax, abl_labels, [ablation_results[k]["pf"] for k in abl_labels],
          abl_colors, "Profit Factor by Ablation", "PF", refline=1.0)

# WR
ax = fig1.add_subplot(gs1[0,1])
bar_panel(ax, abl_labels, [ablation_results[k]["wr"]*100 for k in abl_labels],
          abl_colors, "Win Rate %", "WR %", refline=50)

# n
ax = fig1.add_subplot(gs1[0,2])
bar_panel(ax, abl_labels, [ablation_results[k]["n"] for k in abl_labels],
          abl_colors, "Trade Count", "n")

# MDD
ax = fig1.add_subplot(gs1[0,3])
bar_panel(ax, abl_labels, [abs(ablation_results[k]["mdd"])*100 for k in abl_labels],
          abl_colors, "Max Drawdown %", "MDD %")

# UES
ax = fig1.add_subplot(gs1[1,0])
bar_panel(ax, abl_labels, [ablation_results[k]["ues"] for k in abl_labels],
          abl_colors, "Universal Edge Score", "UES")

# Boot P5
ax = fig1.add_subplot(gs1[1,1])
bar_panel(ax, abl_labels, [ablation_results[k]["boot"].get("p5",0) for k in abl_labels],
          abl_colors, "Bootstrap P5 (floor)", "PF P5", refline=1.0)

# Loss streak
ax = fig1.add_subplot(gs1[1,2])
bar_panel(ax, abl_labels, [ablation_results[k]["loss_streak"] for k in abl_labels],
          abl_colors, "Max Loss Streak", "trades")

# Quality score
ax = fig1.add_subplot(gs1[1,3])
qs_vals = [quality_score(ablation_results[k]) for k in abl_labels]
bar_panel(ax, abl_labels, qs_vals, abl_colors, "Quality Score (UES adj.)", "Score")

saved_charts.append(save_fig(fig1, "r067_ablation.png"))
print("  → r067_ablation.png")

# ── Chart 2: Equity Curves — All Ablations ───────────────────────────────────
fig2, axes2 = plt.subplots(1, 5, figsize=(22, 6))
fig2.suptitle("R067 — Equity Curves: Family C Ablations", fontsize=10, color=C_TEXT,
              fontweight="bold")
for i, (abl_id, col) in enumerate(zip(abl_labels, abl_colors)):
    ax  = axes2[i]; style_ax(ax)
    ev  = ablation_results[abl_id]
    eq  = ev["equity"]
    ax.plot(eq, color=col, lw=1.4)
    ax.axhline(CAPITAL, color=C_GRID, lw=0.8, ls="--")
    cids_str = "+".join(ABLATIONS[abl_id])
    ax.set_title(f"{abl_id}\n{cids_str}\nPF={ev['pf']:.3f}  n={ev['n']}  MDD={ev['mdd']:.1%}",
                 fontsize=7, color=C_TEXT)
    ax.set_xlabel("Trade #", fontsize=6, color=C_TEXT)
    ax.set_ylabel("Equity $", fontsize=6, color=C_TEXT)
plt.tight_layout()
saved_charts.append(save_fig(fig2, "r067_equity_curves.png"))
print("  → r067_equity_curves.png")

# ── Chart 3: Fold Stability Heatmap ──────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 6))
fig3.suptitle("R067 — Fold Stability: PF by Ablation × Fold", fontsize=10,
              color=C_TEXT, fontweight="bold")
style_ax(ax3)
fold_matrix = np.zeros((len(abl_labels), N_FWD_FOLDS))
for i, abl_id in enumerate(abl_labels):
    ev = ablation_results[abl_id]
    for j in range(1, N_FWD_FOLDS+1):
        fold_matrix[i, j-1] = ev["folds"].get(j, {}).get("pf", 0.0)

im = ax3.imshow(fold_matrix, cmap="RdYlGn", vmin=0.5, vmax=3.0, aspect="auto")
ax3.set_xticks(range(N_FWD_FOLDS))
ax3.set_xticklabels([f"Fold {i+1}" for i in range(N_FWD_FOLDS)], fontsize=9)
ax3.set_yticks(range(len(abl_labels)))
ax3.set_yticklabels(abl_labels, fontsize=9)
for i in range(len(abl_labels)):
    for j in range(N_FWD_FOLDS):
        val = fold_matrix[i, j]
        color = "black" if 0.8 < val < 2.2 else "white"
        ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=8, color=color, fontweight="bold")
plt.colorbar(im, ax=ax3, label="OOS Profit Factor", shrink=0.8)
ax3.set_title("Green=profitable, Red=loss, Target: all green", fontsize=8, color=C_TEXT)
plt.tight_layout()
saved_charts.append(save_fig(fig3, "r067_fold_heatmap.png"))
print("  → r067_fold_heatmap.png")

# ── Chart 4: Symbol Breakdown ─────────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(18, 7))
fig4.suptitle("R067 — Symbol Breakdown (Full Family C)", fontsize=10, color=C_TEXT,
              fontweight="bold")

# Top-20 symbols by n
sym_sorted_n = sorted(sym_ev.items(), key=lambda x: x[1]["n"], reverse=True)[:20]
syms_n  = [s[:12] for s,_ in sym_sorted_n]
pfs_n   = [v["pf"] for _,v in sym_sorted_n]
ns_n    = [v["n"]  for _,v in sym_sorted_n]
ax4a    = axes4[0]; style_ax(ax4a)
cols_n  = [C_GREEN if p>=1.0 else C_RED for p in pfs_n]
bars4a  = ax4a.barh(range(len(syms_n)), pfs_n, color=cols_n, alpha=0.85, edgecolor=C_BG)
ax4a.set_yticks(range(len(syms_n))); ax4a.set_yticklabels(syms_n, fontsize=7)
ax4a.axvline(1.0, color=C_GOLD, lw=1, ls="--")
ax4a.invert_yaxis()
ax4a.set_title("Top-20 by Trade Count — PF", fontsize=8, color=C_TEXT)
ax4a.set_xlabel("Profit Factor", fontsize=7, color=C_TEXT)
for bar4, pf_v, n_v in zip(bars4a, pfs_n, ns_n):
    ax4a.text(pf_v + 0.01, bar4.get_y()+bar4.get_height()/2,
              f"n={n_v}", va="center", fontsize=6, color=C_TEXT)

# Tier distribution pie
ax4b = axes4[1]; ax4b.set_facecolor(C_PANEL); ax4b.set_aspect("equal")
tier_ns = [tier_evals.get(t_,{"n":0})["n"] for t_ in ["T1-Large","T2-Mid","T3-Small"]]
tier_pfs= [tier_evals.get(t_,{"pf":0})["pf"] for t_ in ["T1-Large","T2-Mid","T3-Small"]]
ax4b.pie(tier_ns, labels=[f"T1: PF={tier_pfs[0]:.2f}\n(n={tier_ns[0]})",
                            f"T2: PF={tier_pfs[1]:.2f}\n(n={tier_ns[1]})",
                            f"T3: PF={tier_pfs[2]:.2f}\n(n={tier_ns[2]})"],
         colors=[C_GREEN, C_GOLD, C_BLUE], startangle=90,
         textprops={"color": C_TEXT, "fontsize": 9}, autopct="%1.0f%%",
         pctdistance=0.75, wedgeprops={"edgecolor": C_BG})
ax4b.set_title("Trade Distribution by Symbol Tier", fontsize=8, color=C_TEXT)
plt.tight_layout()
saved_charts.append(save_fig(fig4, "r067_symbol_breakdown.png"))
print("  → r067_symbol_breakdown.png")

# ── Chart 5: Intra-session Hour Analysis ──────────────────────────────────────
fig5, axes5 = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle("R067 — Intra-session Analysis (Family C ASI Hours)", fontsize=10,
              color=C_TEXT, fontweight="bold")

ax5a = axes5[0]; style_ax(ax5a)
hours_sorted = sorted(hour_results.keys())
h_pfs  = [hour_results[h]["pf"] for h in hours_sorted]
h_ns   = [hour_results[h]["n"]  for h in hours_sorted]
h_cols = [C_GREEN if pf_>=1.0 else C_RED for pf_ in h_pfs]
bars5  = ax5a.bar(hours_sorted, h_pfs, color=h_cols, alpha=0.85, edgecolor=C_BG, width=0.8)
ax5a.axhline(1.0, color=C_GOLD, lw=1, ls="--", label="Break-even")
for h_, pf_, n_ in zip(hours_sorted, h_pfs, h_ns):
    ax5a.text(h_, pf_+0.01, f"n={n_}", ha="center", va="bottom", fontsize=7, color=C_TEXT)
ax5a.set_xticks(hours_sorted)
ax5a.set_xticklabels([f"{h:02d}:00" for h in hours_sorted], fontsize=8, rotation=45)
ax5a.set_title("PF by UTC Hour (ASI=00–06)", fontsize=9, color=C_TEXT)
ax5a.set_xlabel("UTC Hour", fontsize=7, color=C_TEXT)
ax5a.set_ylabel("Profit Factor", fontsize=7, color=C_TEXT)
ax5a.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

ax5b = axes5[1]; style_ax(ax5b)
h_wrs = [hour_results[h]["wr"] * 100 for h in hours_sorted]
bars5b = ax5b.bar(hours_sorted, h_wrs, color=h_cols, alpha=0.85, edgecolor=C_BG, width=0.8)
ax5b.axhline(50, color=C_GOLD, lw=1, ls="--", label="50% WR")
ax5b.set_xticks(hours_sorted)
ax5b.set_xticklabels([f"{h:02d}:00" for h in hours_sorted], fontsize=8, rotation=45)
ax5b.set_title("Win Rate % by UTC Hour", fontsize=9, color=C_TEXT)
ax5b.set_xlabel("UTC Hour", fontsize=7, color=C_TEXT)
ax5b.set_ylabel("Win Rate %", fontsize=7, color=C_TEXT)
ax5b.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
plt.tight_layout()
saved_charts.append(save_fig(fig5, "r067_intra_session.png"))
print("  → r067_intra_session.png")

# ── Chart 6: Bootstrap Comparison — top candidates ────────────────────────────
top_3_ids = [k for k,_ in sorted(ablation_results.items(),
             key=lambda x: quality_score(x[1]), reverse=True)][:3]
fig6, axes6 = plt.subplots(1, 3, figsize=(16, 6))
fig6.suptitle("R067 — Bootstrap PF Distribution (Top-3 Candidates)", fontsize=10,
              color=C_TEXT, fontweight="bold")
for i, abl_id in enumerate(top_3_ids):
    ax  = axes6[i]; style_ax(ax)
    ev  = ablation_results[abl_id]
    arr = ev["boot"].get("arr", np.array([ev["pf"]]))
    col = abl_colors[list(ABLATIONS.keys()).index(abl_id)]
    ax.hist(arr, bins=max(5,min(50,len(arr)//20)), color=col, alpha=0.75)
    b5  = ev["boot"].get("p5",  0)
    b50 = ev["boot"].get("med", 0)
    ax.axvline(1.0, color=C_RED,   lw=1.5, ls="--", label="Break-even")
    ax.axvline(b5,  color=C_GOLD,  lw=1.2, ls=":",  label=f"P5={b5:.3f}")
    ax.axvline(b50, color=C_GREEN, lw=1.2,           label=f"P50={b50:.3f}")
    ax.set_title(f"{abl_id}  n={ev['n']}  UES={ev['ues']:.1f}", fontsize=9, color=C_TEXT)
    ax.set_xlabel("Profit Factor", fontsize=7, color=C_TEXT)
    ax.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
plt.tight_layout()
saved_charts.append(save_fig(fig6, "r067_bootstrap_comparison.png"))
print("  → r067_bootstrap_comparison.png")

# ── Chart 7: Volatility regime split (bar chart) ─────────────────────────────
fig7, axes7 = plt.subplots(1, 4, figsize=(20, 6))
fig7.suptitle("R067 — Family C Edge by Regime Quartile", fontsize=10, color=C_TEXT,
              fontweight="bold")
reg_feats = [("ATR Rank","atr_rank"), ("Real Vol","real_vol"),
             ("ADX14","adx14"), ("Prev Body","prev_body")]
q_labels  = ["Q1(low)","Q2","Q3","Q4(high)"]
q_colors  = [C_BLUE, C_GOLD, C_ORAN, C_RED]

for i, (fname, fkey) in enumerate(reg_feats):
    ax_ = axes7[i]; style_ax(ax_)
    groups = split_by_feature(base_trades, fkey)
    pfs_q  = []
    ns_q   = []
    for ql in q_labels:
        tl = groups[ql]; p = np.array([t["exit_pnl"] for t in tl])
        pfs_q.append(safe_pf(p[p>0].sum(),abs(p[p<0].sum())) if len(p) else 0.0)
        ns_q.append(len(tl))
    bars7 = ax_.bar(q_labels, pfs_q, color=q_colors, alpha=0.85, edgecolor=C_BG, width=0.6)
    ax_.axhline(1.0, color=C_GOLD, lw=1, ls="--")
    for bar7, pf_q, n_q in zip(bars7, pfs_q, ns_q):
        ax_.text(bar7.get_x()+bar7.get_width()/2, bar7.get_height()+0.01,
                 f"{pf_q:.3f}\nn={n_q}", ha="center", va="bottom",
                 fontsize=6.5, color=C_TEXT)
    ax_.set_title(f"Split by {fname}", fontsize=9, color=C_TEXT)
    ax_.set_ylabel("Profit Factor", fontsize=7, color=C_TEXT)
    ax_.set_xticklabels(q_labels, fontsize=7, rotation=20, ha="right")
plt.tight_layout()
saved_charts.append(save_fig(fig7, "r067_regime_split.png"))
print("  → r067_regime_split.png")

print()

# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
print("  Saving CSV outputs …")

rows = []
for abl_id, cids in ABLATIONS.items():
    ev = ablation_results[abl_id]
    rows.append({
        "variant": abl_id, "conditions": "+".join(cids),
        "pf": round(ev["pf"],4), "wr": round(ev["wr"],4), "n": ev["n"],
        "net": round(ev["net"],2), "mdd": round(ev["mdd"],4),
        "ues": ev["ues"], "boot_p5": round(ev["boot"].get("p5",0),4),
        "boot_med": round(ev["boot"].get("med",0),4),
        "mc_prob": round(ev["mc"].get("prob_profit",0),4),
        "rf": round(ev["rf"],4), "ulcer": round(ev["ulcer"],4),
        "sym_fl": round(ev["sym_fl"],4), "fold_fl": round(ev["fold_fl"],4),
        "win_streak": ev["win_streak"], "loss_streak": ev["loss_streak"],
        "quality_score": round(quality_score(ev),1),
    })
df_abl = pd.DataFrame(rows).sort_values("quality_score",ascending=False)
df_abl.to_csv(os.path.join(OUT,"r067_ablation.csv"), index=False)
print("  → r067_ablation.csv")

# Symbol breakdown CSV
sym_rows = []
for sym, v in sorted(sym_ev.items(), key=lambda x: x[1]["pf"], reverse=True):
    sym_rows.append({"symbol":sym, "tier":tier(sym),
                     "pf":round(v["pf"],4), "wr":round(v["wr"],4), "n":v["n"]})
pd.DataFrame(sym_rows).to_csv(os.path.join(OUT,"r067_symbol_breakdown.csv"), index=False)
print("  → r067_symbol_breakdown.csv")

# Hour breakdown CSV
hour_rows = [{"hour_utc":h, "pf":round(v["pf"],4), "wr":round(v["wr"],4), "n":v["n"]}
             for h,v in sorted(hour_results.items())]
pd.DataFrame(hour_rows).to_csv(os.path.join(OUT,"r067_hours.csv"), index=False)
print("  → r067_hours.csv")

print()

# ─────────────────────────────────────────────────────────────────────────────
# JOURNAL
# ─────────────────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
journal_path = os.path.join(OUT,"r067_journal.md")
with open(journal_path,"w") as mf:
    mf.write(f"# R067 — Family C Dissection\n\n")
    mf.write(f"**Duration:** {elapsed:.0f}s  \n**Symbols:** {len(data)}\n\n")

    mf.write(f"## Section 1 — Baseline\n\n")
    mf.write(f"- **PF:** {ev_base['pf']:.3f}  **WR:** {ev_base['wr']:.1%}  "
             f"**n:** {ev_base['n']}  **MDD:** {ev_base['mdd']:.1%}  "
             f"**UES:** {ev_base['ues']:.1f}\n\n")

    mf.write(f"## Section 2 — Ablation Results\n\n")
    mf.write(f"| Variant | Conditions | PF | WR | n | MDD | UES | BootP5 | LoStr |\n")
    mf.write(f"|---|---|---|---|---|---|---|---|---|\n")
    for abl_id, cids in ABLATIONS.items():
        ev = ablation_results[abl_id]
        mf.write(f"| {abl_id} | {'+'.join(cids)} | {ev['pf']:.3f} | {ev['wr']:.1%} | "
                 f"{ev['n']} | {ev['mdd']:.1%} | {ev['ues']:.1f} | "
                 f"{ev['boot'].get('p5',0):.3f} | {ev['loss_streak']} |\n")
    mf.write(f"\n")

    mf.write(f"## Section 3 — Symbol Tiers\n\n")
    for t_lbl in ["T1-Large","T2-Mid","T3-Small"]:
        v = tier_evals.get(t_lbl,{"pf":0,"wr":0,"n":0})
        mf.write(f"- **{t_lbl}:** PF={v['pf']:.3f}  WR={v['wr']:.1%}  n={v['n']}\n")
    mf.write(f"\n")

    mf.write(f"## Section 4 — Intra-Session Hours\n\n")
    mf.write(f"| Hour UTC | PF | WR | n |\n|---|---|---|---|\n")
    for h in sorted(hour_results.keys()):
        v = hour_results[h]
        mf.write(f"| {h:02d}:00 | {v['pf']:.3f} | {v['wr']:.1%} | {v['n']} |\n")
    if best_window:
        s,e,nn,pf_ = best_window
        mf.write(f"\n**Best sub-window:** {s:02d}:00–{e:02d}:00  PF={pf_:.3f}  n={nn}\n\n")

    mf.write(f"## Section 7 — Condition Contribution\n\n")
    mf.write(f"| Removed | Delta PF | New PF | MDD | LoStr |\n|---|---|---|---|---|\n")
    for cid, info in sorted(ablation_deltas.items(), key=lambda x: -x[1]["delta"]):
        mf.write(f"| {cid} | {info['delta']:+.3f} | {info['pf']:.3f} | "
                 f"{info['mdd']:.1%} | {info['loss_streak']} |\n")
    mf.write(f"\n")

    mf.write(f"## Section 10 — Recommendation\n\n")
    mf.write(f"**Verdict:** {verdict}  \n")
    mf.write(f"**Best variant:** {rec_label}  \n")
    mf.write(f"**PF:** {rec_ev['pf']:.3f}  "
             f"**WR:** {rec_ev['wr']:.1%}  "
             f"**n:** {rec_ev['n']}  "
             f"**MDD:** {rec_ev['mdd']:.1%}  "
             f"**Loss streak:** {rec_ev['loss_streak']}\n\n")

    mf.write(f"## Outputs\n")
    for p in saved_charts:
        mf.write(f"- `{os.path.basename(p)}`\n")
    mf.write(f"- `r067_ablation.csv`\n")
    mf.write(f"- `r067_symbol_breakdown.csv`\n")
    mf.write(f"- `r067_hours.csv`\n")

print("  → r067_journal.md"); print()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL BANNER
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print(f"  R067 COMPLETE — {elapsed:.0f}s")
print(SEP); print()
print(f"  ╔══════════════════════════════════════════════════════════════════════╗")
print(f"  ║  FAMILY C DISSECTION VERDICT: {verdict:<41}║")
print(f"  ║  Best variant: {rec_label:<55}║")
print(f"  ║  PF={rec_ev['pf']:.3f}  WR={rec_ev['wr']:.1%}  n={rec_ev['n']}  "
      f"MDD={rec_ev['mdd']:.1%}  UES={rec_ev['ues']:.1f}{'':>18}║")
print(f"  ╠══════════════════════════════════════════════════════════════════════╣")
print(f"  ║  {'Variant':<14}  {'PF':>6}  {'n':>5}  {'MDD':>7}  {'UES':>6}  {'LoStr':>6}  {'QScore':>6} ║")
print(f"  ║  {'-'*14}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6} ║")
for abl_id in abl_labels:
    ev  = ablation_results[abl_id]
    qs  = quality_score(ev)
    print(f"  ║  {abl_id:<14}  {ev['pf']:>6.3f}  {ev['n']:>5}  "
          f"{ev['mdd']:>7.1%}  {ev['ues']:>6.1f}  {ev['loss_streak']:>6}  {qs:>6.1f} ║")
print(f"  ╚══════════════════════════════════════════════════════════════════════╝")
print()
print(f"  Condition contribution (delta PF when removed):")
for cid, info in sorted(ablation_deltas.items(), key=lambda x: -x[1]["delta"]):
    print(f"    Remove {cid:<12}: ΔPF={info['delta']:+.3f}  "
          f"{'↑ weaker condition' if info['delta']>0.05 else ('↓ essential condition' if info['delta']<-0.05 else '≈ neutral')}")
print()
print(f"  Files: {', '.join(os.path.basename(p) for p in saved_charts)}")
print(f"         r067_ablation.csv  r067_symbol_breakdown.csv  r067_hours.csv  r067_journal.md")
print()
