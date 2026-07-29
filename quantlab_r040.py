"""
=============================================================================
QUANTLAB AI — RESEARCH #040
Timeframe Transfer Validation: 1H → 15m
=============================================================================

Background:
  R039 production candidate (Var D) demonstrated strong edge:
    PF=2.220  n=27  Boot_p50=2.215  MC=98.2%  LOO-S=1.791  LOO-F=1.532
  Sole blocker: n<100 (insufficient trade frequency).

Objective:
  Pure timeframe transfer validation.
  Apply EXACT R039 Var D environment to 15-minute candles.
  Determine whether trade frequency increases 2× while preserving edge.

LOCKED ENVIRONMENT (zero changes from R039 Var D):
  • ATR Rank < p25
  • EMA200 Slope > 0
  • EMA Distance > p75
  • BB Width < p50
  • RELVOL Breakout entry (vol > 1.5× avg, bullish candle)
  • Long only
  • Stop = 1×ATR14  |  Target = 2×ATR14  (2R fixed)
  • Same fees, slippage, risk model

NO optimisation. NO threshold tuning. NO feature engineering.
The ONLY variable is the timeframe.

Promotion Criteria:
  PASS if ALL true: PF>1.20 · n≥100 · Boot_p50>1.20 · MC_P>60%
                    LOO-sym>1.00 · LOO-fold>1.00 · MDD<25%
=============================================================================
"""

import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr

RESEARCH_ID = "R040"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

CAPITAL  = CONFIG["STARTING_CAPITAL"]
RR       = CONFIG["RISK_REWARD"]          # 2.0
BEP_WR   = 1.0 / (1.0 + RR)              # 0.333

# ── LOCKED ENVIRONMENT (R039 Var D — unchanged) ──────────────────────────────
ATR_Q  = 0.25   # ATR Rank < p25
DIST_Q = 0.75   # EMA Distance > p75
BB_Q   = 0.50   # BB Width < p50
# EMA200 slope > 0 always enforced
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLS_WANT = [
    "BTC-USDT-SWAP","ETH-USDT-SWAP","SOL-USDT-SWAP","LINK-USDT-SWAP",
    "AVAX-USDT-SWAP","XRP-USDT-SWAP","LTC-USDT-SWAP","BCH-USDT-SWAP",
    "DOGE-USDT-SWAP","ADA-USDT-SWAP","BNB-USDT-SWAP","DOT-USDT-SWAP",
    "ARB-USDT-SWAP","OP-USDT-SWAP","NEAR-USDT-SWAP","ATOM-USDT-SWAP",
    "SUI-USDT-SWAP","APT-USDT-SWAP","WIF-USDT-SWAP","PEPE-USDT-SWAP",
    "ENA-USDT-SWAP","UNI-USDT-SWAP","FIL-USDT-SWAP",
]
MIN_BARS = 4_000
FOLDS    = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
N_BOOT   = 2_000

TARGET_PF   = 1.20
TARGET_N    = 100
TARGET_BOOT = 1.20
TARGET_MC   = 0.60

# R039 Var D reference (production candidate at 1H)
R039 = {
    "n": 27, "wr": 0.593, "pf": 2.220, "b50": 2.215, "mc_p": 0.982,
    "sym_floor": 1.791, "fold_floor": 1.532, "mdd": -0.047,
    "b5": None, "b95": None,
}

BG    = "#0d1117"
PANEL = "#161b22"
TXT   = "#e0e0e0"
AMBER = "#f0c040"
GREEN = "#2ea043"
RED   = "#cf222e"
BLUE  = "#1f77b4"

SYM_COLS = {
    "BTC-USDT-SWAP":"#F7931A","ETH-USDT-SWAP":"#627EEA","SOL-USDT-SWAP":"#9945FF",
    "LINK-USDT-SWAP":"#2A5ADA","AVAX-USDT-SWAP":"#E84142","XRP-USDT-SWAP":"#346AA9",
    "LTC-USDT-SWAP":"#BFBBBB","BCH-USDT-SWAP":"#8DC351","DOGE-USDT-SWAP":"#C3A634",
    "ADA-USDT-SWAP":"#0033AD","BNB-USDT-SWAP":"#F3BA2F","DOT-USDT-SWAP":"#E6007A",
    "ARB-USDT-SWAP":"#28A0F0","OP-USDT-SWAP":"#FF0420","NEAR-USDT-SWAP":"#00C08B",
    "ATOM-USDT-SWAP":"#6F4CFF","SUI-USDT-SWAP":"#6FBCF0","APT-USDT-SWAP":"#00B4D8",
    "WIF-USDT-SWAP":"#A67C52","PEPE-USDT-SWAP":"#4CAF50","ENA-USDT-SWAP":"#8B0000",
    "UNI-USDT-SWAP":"#FF007A","FIL-USDT-SWAP":"#0090FF",
}

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #040" + " "*50 + "║")
print("║  Timeframe Transfer Validation: 1H → 15m" + " "*37 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Locked environment: R039 Var D
    ATR Rank < p25  ·  EMA200 Slope > 0  ·  EMA Dist > p75  ·  BB Width < p50
  Entry:  RELVOL Breakout (vol > 1.5× avg, bullish candle)
  Exit:   Stop=1×ATR14  Target=2×ATR14  (2R fixed)
  Method: 5-fold expanding WF  ·  IS thresholds only (no OOS leakage)
  Timeframe: 15m  (same indicators, same periods — no re-tuning)
""")

# =============================================================================
# INDICATORS  (identical to R039)
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["vol"]

    df["ema200"]       = calc_ema(c, 200)
    df["atr14"]        = calc_atr(df, 14)
    df["atr_rank"]     = df["atr14"].rolling(100).rank(pct=True) * 100

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_width"]     = (bb_mid + 2*bb_std - (bb_mid - 2*bb_std)) / bb_mid.replace(0, np.nan)

    df["ema_dist_pct"] = (c - df["ema200"]) / df["ema200"].replace(0, np.nan) * 100
    df["ema200_slope"] = (df["ema200"] - df["ema200"].shift(10)) / df["ema200"].shift(10).replace(0, np.nan)

    vol_ma             = v.rolling(20).mean()
    df["rel_vol"]      = v / vol_ma.replace(0, np.nan)

    df["prev_close"]   = c.shift(1)
    df["prev_atr14"]   = df["atr14"].shift(1)

    # For session analysis (Q6)
    df["hour_utc"]     = df["datetime"].dt.hour

    return df

# =============================================================================
# THRESHOLDS  (learned from IS only — identical logic to R039)
# =============================================================================

def learn_thresholds(df_is: pd.DataFrame) -> dict:
    valid = df_is.dropna(subset=["atr_rank","ema_dist_pct","bb_width"])
    atr_thr  = float(valid["atr_rank"].quantile(ATR_Q))
    pos_dist = valid[valid["ema_dist_pct"] > 0]["ema_dist_pct"]
    dist_thr = float(pos_dist.quantile(DIST_Q) if len(pos_dist) > 10
                     else valid["ema_dist_pct"].quantile(DIST_Q))
    bb_thr   = float(valid["bb_width"].quantile(BB_Q))
    return {"atr": atr_thr, "dist": dist_thr, "bb": bb_thr}

def in_environment(df: pd.DataFrame, thr: dict) -> pd.Series:
    return (
        (df["atr_rank"]     < thr["atr"])  &
        (df["ema200_slope"] > 0)            &
        (df["ema_dist_pct"] > thr["dist"])  &
        (df["bb_width"]     < thr["bb"])
    ).fillna(False)

# =============================================================================
# SIGNAL — RELVOL Breakout (identical to R039)
# =============================================================================

def signal_relvol(df: pd.DataFrame, env: pd.Series) -> pd.Series:
    return (
        (df["rel_vol"]    > 1.5) &
        (df["close"]      > df["open"]) &
        (df["close"]      > df["prev_close"]) &
        env
    ).fillna(False)

# =============================================================================
# BACKTEST ENGINE  (identical to R039)
# =============================================================================

def run_backtest(df: pd.DataFrame, signal: pd.Series,
                 sym: str, fold: int) -> list:
    min_sl = CONFIG["MIN_SL_PCT"]; max_lev = CONFIG["MAX_LEVERAGE"]
    rf     = CONFIG["RISK_PER_TRADE_PCT"]; fee = CONFIG["TAKER_FEE"]
    spd    = CONFIG["SPREAD"] * 0.5; slp = CONFIG["SL_SLIPPAGE"]

    in_pos = False
    ep = st = tk = sz = 0.0; et = None; ei = -1
    trades = []

    for i in range(1, len(df)):
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_pos:
            sl_hit = bar["low"]  <= st
            tp_hit = bar["high"] >= tk
            if sl_hit or tp_hit:
                xp   = (st * (1 - slp)) if sl_hit else tk
                xt   = "SL" if sl_hit else "TP"
                sd   = ep - st
                gross = (xp - ep) * sz
                cost  = (ep*sz + xp*sz) * (fee + spd)
                slpc  = (st - xp) * sz if sl_hit else 0.0
                net   = gross - cost - slpc
                rmul  = (xp - ep) / sd if sd > 0 else 0.0
                trades.append({
                    "sym": sym, "fold": fold,
                    "entry_time": str(et), "exit_time": str(bar["datetime"]),
                    "entry_hour": et.hour if hasattr(et, "hour") else 0,
                    "pnl": round(net, 4), "r_multiple": round(rmul, 4),
                    "win": int(xt == "TP"), "exit_type": xt,
                    "holding_bars": i - ei,
                })
                in_pos = False
            continue

        if signal.iloc[i - 1]:
            atr_ = prev["atr14"]
            if pd.isna(atr_) or atr_ <= 0:
                continue
            sd = atr_
            if sd / bar["open"] < min_sl:
                continue
            ep = bar["open"]; st = ep - sd; tk = ep + RR * sd
            sz = min(CAPITAL * rf / sd, (CAPITAL * max_lev) / ep)
            et = bar["datetime"]; ei = i
            in_pos = True

    return trades

# =============================================================================
# STATISTICS  (identical to R039)
# =============================================================================

def safe_pf(gw, gl):
    return min(gw / 1e-9, 10.0) if gl <= 0 else gw / gl

def metrics(trades: list) -> dict:
    if not trades:
        return {"n":0,"wr":0.0,"pf":0.0,"exp_r":0.0,"net":0.0,
                "sharpe":0.0,"mdd":0.0,"pnls":np.array([]),"equity":np.array([CAPITAL])}
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values
    wins = df["win"].values.astype(bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw   = pnl[wins].sum()       if nw else 0.0
    gl   = abs(pnl[~wins].sum()) if nl else 0.0
    pf   = safe_pf(gw, gl)
    wr   = nw / n
    exp_r = wr * RR - (1 - wr)
    equity = np.concatenate([[CAPITAL], CAPITAL + np.cumsum(pnl)])
    peak   = np.maximum.accumulate(equity)
    mdd    = float(((equity - peak) / peak).min())
    bpy    = 365 * 24 * 4   # 15m bars per year ≈ 35040
    ann    = (equity[-1] / CAPITAL) ** (bpy / max(n, 1)) - 1
    vol    = pnl.std() * math.sqrt(bpy) if n > 1 else 1e-9
    sharpe = ann / vol if vol > 0 else 0.0
    return {"n":n,"wr":wr,"pf":pf,"exp_r":exp_r,"net":float(pnl.sum()),
            "sharpe":sharpe,"mdd":mdd,"pnls":pnl,"equity":equity}

def bootstrap_pf(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed); pfs = []
    for _ in range(n_iter):
        s = rng.choice(pnls, len(pnls), replace=True)
        pfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
    return float(np.percentile(pfs,5)), float(np.percentile(pfs,50)), float(np.percentile(pfs,95))

def monte_carlo(pnls, n_iter=N_BOOT, seed=42):
    if len(pnls) < 5:
        return {"prob_profit":0.0,"p5":CAPITAL,"p50":CAPITAL,"p95":CAPITAL,"finals":np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit":float((finals>CAPITAL).mean()),
            "p5":float(np.percentile(finals,5)),"p50":float(np.percentile(finals,50)),
            "p95":float(np.percentile(finals,95)),"finals":finals}

def loo_sym(sym_trades_dict):
    return {omit: metrics([t for s,tl in sym_trades_dict.items() if s!=omit for t in tl])["pf"]
            for omit in sym_trades_dict}

def loo_fld(all_trades):
    return {omit: metrics([t for t in all_trades if t["fold"]!=omit])["pf"]
            for omit in sorted({t["fold"] for t in all_trades})}

# =============================================================================
# DATA LOAD  (15m only)
# =============================================================================

print("─"*78)
print("  Loading 15m data …")
all_dfs = {}
missing  = []
for sym in SYMBOLS_WANT:
    tag  = sym.replace("-","_")
    path = f"{CACHE}/{tag}_15m.parquet"
    if not os.path.exists(path):
        missing.append(sym)
        continue
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) < MIN_BARS:
        print(f"  ✗ {sym}: only {len(df)} bars (< {MIN_BARS}), skipping")
        continue
    all_dfs[sym] = add_features(df)

SYMBOLS = list(all_dfs.keys())
print(f"  {len(SYMBOLS)} symbols loaded  ({sum(len(d) for d in all_dfs.values()):,} bars)")
if missing:
    print(f"  Symbols without 15m cache ({len(missing)}): "
          f"{', '.join(s.replace('-USDT-SWAP','') for s in missing)}")
print()

# =============================================================================
# WALK-FORWARD
# =============================================================================

sym_trades   = {sym: [] for sym in SYMBOLS}
fold_pf_list = []
fold_n_list  = []
all_trades   = []
env_bars_total = 0

print(f"  Walk-forward: {len(FOLDS)} folds × {len(SYMBOLS)} symbols")
print(f"  Environment: ATR<p{ATR_Q*100:.0f} · slope>0 · EMA_dist>p{DIST_Q*100:.0f} · BB<p{BB_Q*100:.0f}")
print()

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_t = []
    fold_e = 0

    for sym, df_full in all_dfs.items():
        N      = len(df_full)
        df_is  = df_full.iloc[:int(N*is_end)]
        df_oos = df_full.iloc[int(N*is_end):int(N*oos_end)].reset_index(drop=True)
        if len(df_oos) < 200:
            continue

        thr = learn_thresholds(df_is)
        env = in_environment(df_oos, thr)
        fold_e += int(env.sum())
        sig = signal_relvol(df_oos, env)
        tl  = run_backtest(df_oos, sig, sym, fold_idx)
        sym_trades[sym].extend(tl)
        fold_t.extend(tl)

    env_bars_total += fold_e
    m = metrics(fold_t)
    fold_pf_list.append(m["pf"])
    fold_n_list.append(m["n"])
    all_trades.extend(fold_t)
    print(f"  Fold {fold_idx}  (IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%)  "
          f"env_bars={fold_e:6,}  n={m['n']:4d}  PF={m['pf']:.3f}")

print()

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print("─"*78)
print("  Computing aggregate statistics …")

m15 = metrics(all_trades)
b5, b50, b95 = bootstrap_pf(m15["pnls"])
mc = monte_carlo(m15["pnls"])
ls = loo_sym(sym_trades)
lf = loo_fld(all_trades)
sym_floor  = min(ls.values()) if ls else 0.0
fold_floor = min(lf.values()) if lf else 0.0

# Per-symbol metrics
sym_metrics = {}
for sym in SYMBOLS:
    tl = sym_trades[sym]
    sm = metrics(tl)
    sym_metrics[sym] = sm

# =============================================================================
# PROMOTION CRITERIA
# =============================================================================

criteria = {
    "PF > 1.20":           m15["pf"]       > TARGET_PF,
    "n ≥ 100":             m15["n"]        >= TARGET_N,
    "Boot p50 > 1.20":     b50             > TARGET_BOOT,
    "MC P > 60%":          mc["prob_profit"] > TARGET_MC,
    "LOO-sym floor > 1.00": sym_floor      > 1.00,
    "LOO-fold floor > 1.00":fold_floor     > 1.00,
    "MDD < 25%":           abs(m15["mdd"]) < 0.25,
}
passed = sum(criteria.values())
overall = "PROMOTE" if passed == 7 else ("WATCHLIST" if passed >= 5 else "REJECT")

# =============================================================================
# RESULTS TABLE (side-by-side 15m vs 1H)
# =============================================================================

print("\n" + "═"*100)
print("  R040 — TIMEFRAME TRANSFER VALIDATION  (15m vs R039 1H reference)")
print("═"*100)

def fmt_delta(val15, val1h, higher_better=True):
    d = val15 - val1h
    sign = "+" if d >= 0 else ""
    better = (d > 0) == higher_better
    mark = "▲" if d > 0 else ("▼" if d < 0 else "=")
    return f"{sign}{d:.3f} {mark}"

print(f"\n  {'Metric':<22}  {'1H (R039 Var D)':>16}  {'15m (R040)':>16}  {'Delta':>14}  {'Criterion'}")
print("  " + "─"*90)

comparisons = [
    ("Trades (n)",       R039["n"],      m15["n"],           True,  "≥ 100"),
    ("Win Rate",         R039["wr"],     m15["wr"],          True,  "—"),
    ("Profit Factor",    R039["pf"],     m15["pf"],          True,  "> 1.20"),
    ("Expectancy (R)",   None,           m15["exp_r"],       True,  "—"),
    ("Max Drawdown",     R039["mdd"],    m15["mdd"],         False, "< −25%"),
    ("Boot p50",         R039["b50"],    b50,                True,  "> 1.20"),
    ("MC P(profit)",     R039["mc_p"],   mc["prob_profit"],  True,  "> 60%"),
    ("LOO-sym floor",    R039["sym_floor"],   sym_floor,     True,  "> 1.00"),
    ("LOO-fold floor",   R039["fold_floor"],  fold_floor,    True,  "> 1.00"),
]
for label, v1h, v15, hb, crit in comparisons:
    s1h = f"{v1h:.3f}" if isinstance(v1h, float) else (str(v1h) if v1h is not None else "  n/a")
    s15 = f"{v15:.3f}" if isinstance(v15, float) else str(v15)
    if isinstance(v1h, int):
        s1h = str(v1h); s15 = str(v15)
        d = v15 - v1h
        delta = f"+{d} ▲" if d > 0 else (f"{d} ▼" if d < 0 else "= ")
    elif v1h is not None:
        delta = fmt_delta(v15, v1h, hb)
    else:
        delta = "  —"
    print(f"  {label:<22}  {s1h:>16}  {s15:>16}  {delta:>14}  {crit}")

print()

# =============================================================================
# PROMOTION CRITERIA CHECKLIST
# =============================================================================

print("─"*78)
print("  PROMOTION CHECKLIST:")
print()
for crit, result in criteria.items():
    mark = "✓" if result else "✗"
    print(f"    {mark}  {crit}")
print()
print(f"  Score: {passed}/7   Verdict: {overall}")
print()

# =============================================================================
# RESEARCH QUESTIONS
# =============================================================================

print("═"*100)
print("  RESEARCH QUESTIONS")
print("═"*100)

freq_2x = m15["n"] >= 2 * R039["n"]
q2_pass = m15["pf"] > 1.20
q3_pass = b50 > 1.20
q4_pass = mc["prob_profit"] > 0.60
q5_pass = sym_floor > 1.00

# Q6: Session analysis (hour distribution of winning vs losing trades)
if all_trades:
    tdf = pd.DataFrame(all_trades)
    hour_counts = tdf.groupby("entry_hour").agg(
        n=("win","count"),
        wins=("win","sum")
    ).reset_index()
    hour_counts["wr"]  = hour_counts["wins"] / hour_counts["n"]
    hour_counts["pf"]  = hour_counts.apply(
        lambda row: safe_pf(
            tdf[(tdf["entry_hour"]==row["entry_hour"]) & (tdf["win"]==1)]["pnl"].sum(),
            abs(tdf[(tdf["entry_hour"]==row["entry_hour"]) & (tdf["win"]==0)]["pnl"].sum())
        ), axis=1
    )
    # Cluster test: do top-5 hours represent disproportionate fraction of trades?
    top5_hours = hour_counts.nlargest(5,"n")["entry_hour"].tolist()
    top5_n     = hour_counts.nlargest(5,"n")["n"].sum()
    top5_frac  = top5_n / max(m15["n"], 1)
    session_clustered = top5_frac > 0.65   # flag if >65% in 5 hours
else:
    tdf = pd.DataFrame()
    hour_counts = pd.DataFrame()
    top5_hours = []; top5_frac = 0.0; session_clustered = False

# Q7: Per-symbol comparison (15m PF vs no 1H reference available for individual syms)
sym_sorted_pf = sorted(
    [(s, sym_metrics[s]["pf"], sym_metrics[s]["n"]) for s in SYMBOLS if sym_metrics[s]["n"] > 0],
    key=lambda x: -x[1]
)

freq_change = m15["n"] / max(R039["n"], 1)

print(f"""
  Q1. Does 15m increase trade frequency by at least 2×?
      1H n={R039['n']}  →  15m n={m15['n']}  (×{freq_change:.1f})
      → {'YES ✓' if freq_2x else 'NO ✗'} — {"2× threshold met" if freq_2x else "2× threshold NOT met"}

  Q2. Does Profit Factor remain above 1.20?
      1H PF={R039['pf']:.3f}  →  15m PF={m15['pf']:.3f}
      → {'YES ✓' if q2_pass else 'NO ✗'}

  Q3. Does Bootstrap median remain above 1.20?
      1H p50={R039['b50']:.3f}  →  15m p50={b50:.3f}  CI=[{b5:.3f}, {b95:.3f}]
      → {'YES ✓' if q3_pass else 'NO ✗'}

  Q4. Does Monte Carlo Probability of Profit remain above 60%?
      1H MC={R039['mc_p']*100:.1f}%  →  15m MC={mc['prob_profit']*100:.1f}%
      → {'YES ✓' if q4_pass else 'NO ✗'}

  Q5. Does the edge remain distributed across symbols?
      LOO-sym floor: 1H={R039['sym_floor']:.3f}  →  15m={sym_floor:.3f}
      → {'YES ✓' if q5_pass else 'NO ✗'}

  Q6. Does the environment cluster around specific sessions or remain consistent?
      Top-5 hours by frequency: {top5_hours}
      Top-5 hours account for {top5_frac*100:.1f}% of all trades
      → {"CLUSTERED (>65% in 5 hours)" if session_clustered else "DISTRIBUTED — consistent throughout day"}

  Q7. Which symbols show strongest activity at 15m?""")

for sym, pf_, n_ in sym_sorted_pf[:8]:
    label = sym.replace("-USDT-SWAP","")
    print(f"      {label:<8}  PF={pf_:.3f}  n={n_:3d}")

if not sym_sorted_pf:
    print("      No trade data available by symbol")

# Timeframe conclusion
print(f"""
  ── TIMEFRAME VERDICT ──
  Environment IS timeframe-dependent: {'NO — edge transfers' if (q2_pass and q3_pass) else 'YES — edge degrades at 15m'}
  {'Recommend: REMAIN 1H strategy (edge not preserved at 15m)' if not (q2_pass and q3_pass) else ''}
  {'Overall: ' + overall}
""")

# =============================================================================
# CHARTS
# =============================================================================

print("─"*78)
print("  Generating charts …")

def _style(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#888", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=TXT, fontsize=8, pad=4)

# ── 1: Equity Curves (per symbol) ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG)
_style(ax, f"R040 — OOS Equity Curves by Symbol (15m, RELVOL, n={m15['n']})")
for sym in SYMBOLS:
    tl = sym_trades[sym]
    if not tl: continue
    eq_ = CAPITAL + np.cumsum([t["pnl"] for t in tl])
    label = sym.replace("-USDT-SWAP","")
    ax.plot(eq_, color=SYM_COLS.get(sym,"#888"), lw=1.1, alpha=0.80, label=label)
ax.axhline(CAPITAL, color="#444", lw=0.6, ls="--")
ax.set_xlabel("Trade #", color="#888", fontsize=8)
ax.set_ylabel("Equity ($)", color="#888", fontsize=8)
ax.legend(ncol=4, facecolor=PANEL, labelcolor=TXT, fontsize=7)
plt.tight_layout()
p = f"{OUT}/r040_equity_curves.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 2: Side-by-side Bar Comparison (1H vs 15m) ───────────────────────────────
metrics_cmp = [
    ("n",       R039["n"],     m15["n"],            "Trade Count",   100),
    ("pf",      R039["pf"],    m15["pf"],            "Profit Factor", 1.20),
    ("b50",     R039["b50"],   b50,                  "Boot p50",      1.20),
    ("mc_p",    R039["mc_p"]*100, mc["prob_profit"]*100, "MC P(profit)%", 60),
    ("loos",    R039["sym_floor"], sym_floor,         "LOO-sym floor", 1.00),
    ("loof",    R039["fold_floor"], fold_floor,       "LOO-fold floor",1.00),
]

fig, axes = plt.subplots(2, 3, figsize=(20, 9), facecolor=BG)
fig.suptitle("R040 — 1H (R039 Var D) vs 15m Side-by-Side Comparison",
             color=TXT, fontsize=11, y=0.98)

for ax_, (key, v1h, v15, title, tgt) in zip(axes.flat, metrics_cmp):
    _style(ax_, title)
    x_ = [0, 1]
    clrs = [BLUE, GREEN if (v15 > v1h) else RED]
    bars_ = ax_.bar(x_, [v1h, v15], color=clrs, alpha=0.85, width=0.5)
    ax_.axhline(tgt, color=AMBER, lw=0.8, ls=":", label=f"Target={tgt}")
    ax_.set_xticks([0, 1])
    ax_.set_xticklabels(["1H (R039)", "15m (R040)"], color=TXT, fontsize=9)
    for b, val in zip(bars_, [v1h, v15]):
        fmt = ".0f" if key == "n" else ".3f"
        ax_.text(b.get_x()+b.get_width()/2, max(val,0)+abs(tgt)*0.02,
                 f"{val:{fmt}}", ha="center", color=TXT, fontsize=9, fontweight="bold")
    ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

plt.tight_layout()
p = f"{OUT}/r040_comparison.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 3: Session / Hour Analysis (Q6) ──────────────────────────────────────────
if not hour_counts.empty:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=BG)
    fig.suptitle("R040 — Session Distribution (Q6): Are trades clustered by time of day?",
                 color=TXT, fontsize=10)

    ax_ = axes[0]
    _style(ax_, "Trade Count by UTC Hour")
    hrs = hour_counts["entry_hour"].values
    ns  = hour_counts["n"].values
    clrs_ = [GREEN if h in top5_hours else BLUE for h in hrs]
    ax_.bar(hrs, ns, color=clrs_, alpha=0.85)
    ax_.set_xlabel("UTC Hour", color="#888", fontsize=8)
    ax_.set_ylabel("Trades", color="#888", fontsize=8)
    ax_.set_xticks(range(0, 24, 2))

    ax_ = axes[1]
    _style(ax_, "Win Rate by UTC Hour")
    wrs = hour_counts["wr"].values
    ax_.bar(hrs, wrs * 100, color=BLUE, alpha=0.80)
    ax_.axhline(BEP_WR * 100, color=AMBER, lw=0.9, ls=":", label=f"BEP WR={BEP_WR*100:.1f}%")
    ax_.set_xlabel("UTC Hour", color="#888", fontsize=8)
    ax_.set_ylabel("Win Rate (%)", color="#888", fontsize=8)
    ax_.set_xticks(range(0, 24, 2))
    ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

    plt.tight_layout()
    p = f"{OUT}/r040_session_analysis.png"
    plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
    print(f"  → {p}")

# ── 4: Per-symbol PF at 15m (Q7) ─────────────────────────────────────────────
if sym_sorted_pf:
    syms_  = [x[0].replace("-USDT-SWAP","") for x in sym_sorted_pf]
    pfs_   = [x[1] for x in sym_sorted_pf]
    ns_    = [x[2] for x in sym_sorted_pf]
    clrs_  = [GREEN if pf_ > 1.20 else (AMBER if pf_ > 1.00 else RED) for pf_ in pfs_]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=BG)
    fig.suptitle("R040 — Per-Symbol Performance at 15m (Q7)", color=TXT, fontsize=10)

    ax_ = axes[0]
    _style(ax_, "Profit Factor by Symbol (15m)")
    x_ = np.arange(len(syms_))
    bars_ = ax_.bar(x_, pfs_, color=clrs_, alpha=0.85)
    ax_.axhline(1.20, color=AMBER, lw=0.8, ls=":", label="PF=1.20")
    ax_.axhline(1.00, color=RED, lw=0.7, ls="--", label="PF=1.00")
    ax_.set_xticks(x_)
    ax_.set_xticklabels(syms_, rotation=45, ha="right", color=TXT, fontsize=7)
    ax_.set_ylabel("PF", color="#888", fontsize=8)
    for b, v in zip(bars_, pfs_):
        ax_.text(b.get_x()+b.get_width()/2, v+0.02,
                 f"{v:.2f}", ha="center", color=TXT, fontsize=6, fontweight="bold")
    ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

    ax_ = axes[1]
    _style(ax_, "Trade Count by Symbol (15m)")
    ax_.bar(x_, ns_, color=BLUE, alpha=0.80)
    ax_.set_xticks(x_)
    ax_.set_xticklabels(syms_, rotation=45, ha="right", color=TXT, fontsize=7)
    ax_.set_ylabel("n trades", color="#888", fontsize=8)
    for b, v in zip(ax_.patches, ns_):
        ax_.text(b.get_x()+b.get_width()/2, v+0.1, str(v),
                 ha="center", color=TXT, fontsize=6)

    plt.tight_layout()
    p = f"{OUT}/r040_symbol_analysis.png"
    plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
    print(f"  → {p}")

# ── 5: Bootstrap CI & Monte Carlo ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=BG)
fig.suptitle("R040 — Robustness: Bootstrap PF CI & Monte Carlo (15m)", color=TXT, fontsize=10)

# Bootstrap CI
ax_ = axes[0]
_style(ax_, f"Bootstrap PF Distribution (n_iter={N_BOOT})")
if len(m15["pnls"]) >= 5:
    rng = np.random.default_rng(42)
    boot_pfs = []
    for _ in range(N_BOOT):
        s = rng.choice(m15["pnls"], len(m15["pnls"]), replace=True)
        boot_pfs.append(safe_pf(s[s>0].sum(), abs(s[s<0].sum())))
    ax_.hist(boot_pfs, bins=60, color=BLUE, alpha=0.75, edgecolor="#333")
    ax_.axvline(b50,  color=GREEN, lw=1.5, label=f"Median={b50:.3f}")
    ax_.axvline(b5,   color=RED,   lw=1.2, ls="--", label=f"p5={b5:.3f}")
    ax_.axvline(b95,  color=AMBER, lw=1.2, ls="--", label=f"p95={b95:.3f}")
    ax_.axvline(1.20, color=AMBER, lw=0.9, ls=":", label="PF=1.20")
ax_.set_xlabel("Profit Factor", color="#888", fontsize=8)
ax_.set_ylabel("Frequency", color="#888", fontsize=8)
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# Monte Carlo
ax_ = axes[1]
_style(ax_, f"Monte Carlo Equity Distribution (n_iter={N_BOOT})")
ax_.hist(mc["finals"], bins=60, color=GREEN, alpha=0.75, edgecolor="#333")
ax_.axvline(CAPITAL,   color=RED,   lw=1.2, ls="--", label=f"Initial={CAPITAL:,.0f}")
ax_.axvline(mc["p50"], color=AMBER, lw=1.5, label=f"Median=${mc['p50']:,.0f}")
ax_.axvline(mc["p5"],  color=RED,   lw=1.0, ls=":", label=f"p5=${mc['p5']:,.0f}")
ax_.set_xlabel("Final Equity ($)", color="#888", fontsize=8)
ax_.set_ylabel("Frequency", color="#888", fontsize=8)
ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)
pp = mc["prob_profit"] * 100
ax_.set_title(f"Monte Carlo  |  P(profit)={pp:.1f}%  {'✓' if pp>60 else '✗'}",
              color=TXT, fontsize=8, pad=4)

plt.tight_layout()
p = f"{OUT}/r040_bootstrap_ci.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 6: LOO Robustness ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor=BG)
fig.suptitle("R040 — Leave-One-Out Robustness (15m)", color=TXT, fontsize=10)

ax_ = axes[0]
_style(ax_, f"LOO-Symbol  (floor={sym_floor:.3f})")
if ls:
    syms_loo = [s.replace("-USDT-SWAP","") for s in ls.keys()]
    pfs_loo  = list(ls.values())
    clrs_loo = [GREEN if p > 1.0 else RED for p in pfs_loo]
    ax_.bar(range(len(syms_loo)), pfs_loo, color=clrs_loo, alpha=0.85)
    ax_.axhline(1.0,  color=AMBER, lw=0.8, ls=":")
    ax_.axhline(1.20, color=AMBER, lw=0.6, ls="--")
    ax_.set_xticks(range(len(syms_loo)))
    ax_.set_xticklabels(syms_loo, rotation=45, ha="right", color=TXT, fontsize=7)
    ax_.set_ylabel("PF", color="#888", fontsize=8)

ax_ = axes[1]
_style(ax_, f"LOO-Fold  (floor={fold_floor:.3f})")
if lf:
    folds_loo = [f"F{k}" for k in sorted(lf.keys())]
    pfs_lf    = [lf[k] for k in sorted(lf.keys())]
    clrs_lf   = [GREEN if p > 1.0 else RED for p in pfs_lf]
    ax_.bar(range(len(folds_loo)), pfs_lf, color=clrs_lf, alpha=0.85)
    ax_.axhline(1.0, color=AMBER, lw=0.8, ls=":")
    ax_.set_xticks(range(len(folds_loo)))
    ax_.set_xticklabels(folds_loo, color=TXT, fontsize=9)
    ax_.set_ylabel("PF", color="#888", fontsize=8)
    for i, (b, v) in enumerate(zip(ax_.patches, pfs_lf)):
        ax_.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
                 ha="center", color=TXT, fontsize=8, fontweight="bold")

plt.tight_layout()
p = f"{OUT}/r040_loo_robustness.png"
plt.savefig(p, dpi=130, facecolor=BG, bbox_inches="tight"); plt.close()
print(f"  → {p}")

# ── 7: Dashboard ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(28, 22), facecolor=BG)
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.35,
                         top=0.93, bottom=0.04, left=0.04, right=0.98)

# Row 0: Key metrics comparison bars
row0_items = [
    ("Trades (n)",  R039["n"],      m15["n"],       100,   False, False),
    ("PF",          R039["pf"],     m15["pf"],      1.20,  False, False),
    ("Boot p50",    R039["b50"],    b50,            1.20,  False, False),
    ("MC P(%)",     R039["mc_p"]*100, mc["prob_profit"]*100, 60, False, True),
]
for col, (title, v1h, v15, tgt, is_n, is_pct) in enumerate(row0_items):
    ax_ = fig.add_subplot(gs[0, col])
    _style(ax_, title)
    vals = [v1h, v15]
    clrs_b = [BLUE, GREEN if v15 >= v1h else RED]
    bars_ = ax_.bar([0, 1], vals, color=clrs_b, alpha=0.85, width=0.5)
    ax_.axhline(tgt, color=AMBER, lw=0.8, ls=":", label=f"Target={tgt}")
    ax_.set_xticks([0, 1])
    ax_.set_xticklabels(["1H", "15m"], color=TXT, fontsize=9)
    for b, val in zip(bars_, vals):
        fmt = ".0f" if is_n or is_pct else ".3f"
        ax_.text(b.get_x()+b.get_width()/2, max(val, 0)+abs(tgt)*0.015,
                 f"{val:{fmt}}", ha="center", color=TXT, fontsize=9, fontweight="bold")
    ax_.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# Row 1: Equity + Fold PF
ax_eq = fig.add_subplot(gs[1, 0:2])
_style(ax_eq, f"OOS Equity Curves (15m)  n={m15['n']}")
for sym in SYMBOLS:
    tl = sym_trades[sym]
    if not tl: continue
    eq_ = CAPITAL + np.cumsum([t["pnl"] for t in tl])
    ax_eq.plot(eq_, color=SYM_COLS.get(sym,"#888"), lw=0.9, alpha=0.75)
ax_eq.axhline(CAPITAL, color="#444", lw=0.5, ls="--")
ax_eq.set_ylabel("Equity ($)", color="#888", fontsize=7)

ax_fp = fig.add_subplot(gs[1, 2:4])
_style(ax_fp, "PF by Fold (15m)")
ax_fp.plot(range(1, len(fold_pf_list)+1), fold_pf_list,
           marker="o", ms=6, lw=1.5, color=BLUE, label="15m Folds")
ax_fp.axhline(TARGET_PF, color=AMBER, lw=0.8, ls=":", label="PF=1.20")
ax_fp.axhline(1.0, color="#555", lw=0.5, ls="--")
ax_fp.set_xticks(range(1, len(fold_pf_list)+1))
ax_fp.set_ylabel("PF", color="#888", fontsize=8)
for i, (n_, pf_) in enumerate(zip(fold_n_list, fold_pf_list)):
    ax_fp.annotate(f"n={n_}", (i+1, pf_),
                   xytext=(0, 8), textcoords="offset points",
                   color="#888", fontsize=7, ha="center")
ax_fp.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# Row 2: Bootstrap + LOO
ax_bc = fig.add_subplot(gs[2, 0:2])
_style(ax_bc, "Bootstrap PF CI (15m)")
if len(m15["pnls"]) >= 5:
    ax_bc.hist(boot_pfs, bins=50, color=BLUE, alpha=0.7, edgecolor="#333")
    ax_bc.axvline(b50,  color=GREEN, lw=1.5, label=f"Median={b50:.3f}")
    ax_bc.axvline(1.20, color=AMBER, lw=0.9, ls=":", label="PF=1.20")
ax_bc.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

ax_loo2 = fig.add_subplot(gs[2, 2:4])
_style(ax_loo2, f"LOO Floors  sym={sym_floor:.3f}  fold={fold_floor:.3f}")
if ls and lf:
    x_loo = np.arange(len(ls)); w = 0.38
    sym_loo_keys = list(ls.keys())
    sym_loo_vals = list(ls.values())
    x_sym = np.arange(len(sym_loo_keys))
    ax_loo2.bar(x_sym, sym_loo_vals, 0.6,
                color=[GREEN if v>1.0 else RED for v in sym_loo_vals], alpha=0.85, label="LOO-sym")
    ax_loo2.axhline(1.0, color=AMBER, lw=0.8, ls=":")
    ax_loo2.axhline(1.20, color=AMBER, lw=0.6, ls="--")
    ax_loo2.set_xticks(x_sym)
    ax_loo2.set_xticklabels([s.replace("-USDT-SWAP","") for s in sym_loo_keys],
                             rotation=45, ha="right", color=TXT, fontsize=6)
    ax_loo2.legend(facecolor=PANEL, labelcolor=TXT, fontsize=7)

# Row 3: Summary table + criteria checklist
ax_tbl = fig.add_subplot(gs[3, 0:3])
ax_tbl.set_facecolor(PANEL)
for sp in ax_tbl.spines.values(): sp.set_visible(False)
ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
ck = lambda c: "✓" if c else "✗"
tbl_text = (
    f"R040 — TIMEFRAME TRANSFER (15m) vs R039 Var D (1H)\n"
    f"{'─'*70}\n"
    f"{'Metric':<22}  {'1H':>10}  {'15m':>10}  {'Pass?':>6}\n"
    f"{'─'*50}\n"
    f"  {'n (trades)':<20}  {R039['n']:>10d}  {m15['n']:>10d}  {ck(m15['n']>=100)}\n"
    f"  {'Win Rate':<20}  {R039['wr']*100:>9.1f}%  {m15['wr']*100:>9.1f}%\n"
    f"  {'Profit Factor':<20}  {R039['pf']:>10.3f}  {m15['pf']:>10.3f}  {ck(m15['pf']>1.20)}\n"
    f"  {'Boot p50':<20}  {R039['b50']:>10.3f}  {b50:>10.3f}  {ck(b50>1.20)}\n"
    f"  {'Boot CI':<20}  {'—':>10}  [{b5:.3f},{b95:.3f}]\n"
    f"  {'MC P(profit)':<20}  {R039['mc_p']*100:>9.1f}%  {mc['prob_profit']*100:>9.1f}%  {ck(mc['prob_profit']>0.60)}\n"
    f"  {'LOO-sym floor':<20}  {R039['sym_floor']:>10.3f}  {sym_floor:>10.3f}  {ck(sym_floor>1.00)}\n"
    f"  {'LOO-fold floor':<20}  {R039['fold_floor']:>10.3f}  {fold_floor:>10.3f}  {ck(fold_floor>1.00)}\n"
    f"  {'Max Drawdown':<20}  {R039['mdd']:>9.1%}  {m15['mdd']:>9.1%}  {ck(abs(m15['mdd'])<0.25)}\n"
    f"{'─'*50}\n"
    f"  SCORE: {passed}/7   VERDICT: {overall}"
)
ax_tbl.text(0.01, 0.97, tbl_text, transform=ax_tbl.transAxes,
            color=TXT, fontsize=8, fontfamily="monospace", va="top")

ax_ans = fig.add_subplot(gs[3, 3])
ax_ans.set_facecolor(PANEL)
for sp in ax_ans.spines.values(): sp.set_visible(False)
ax_ans.set_xticks([]); ax_ans.set_yticks([])
q_text = (
    f"RESEARCH QUESTIONS\n\n"
    f"Q1. 2× frequency?\n"
    f"  1H n={R039['n']}  15m n={m15['n']}  ×{freq_change:.1f}\n"
    f"  {'YES ✓' if freq_2x else 'NO ✗'}\n\n"
    f"Q2. PF > 1.20?\n"
    f"  {m15['pf']:.3f}  {'✓' if q2_pass else '✗'}\n\n"
    f"Q3. Boot p50 > 1.20?\n"
    f"  {b50:.3f}  {'✓' if q3_pass else '✗'}\n\n"
    f"Q4. MC > 60%?\n"
    f"  {mc['prob_profit']*100:.1f}%  {'✓' if q4_pass else '✗'}\n\n"
    f"Q5. Edge distributed?\n"
    f"  LOO-sym={sym_floor:.3f}  {'✓' if q5_pass else '✗'}\n\n"
    f"Q6. Session clustered?\n"
    f"  {'YES' if session_clustered else 'NO — distributed'}\n"
    f"  Top-5h: {top5_frac*100:.0f}% of trades\n\n"
    f"VERDICT: {overall}"
)
ax_ans.text(0.05, 0.97, q_text, transform=ax_ans.transAxes,
            color=TXT, fontsize=8, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#0d1117", edgecolor="#444"))

verdict_color = GREEN if overall == "PROMOTE" else (AMBER if overall == "WATCHLIST" else RED)
fig.suptitle(
    f"QUANTLAB AI — R040 | 15m Timeframe Transfer | Environment: R039 Var D (locked)\n"
    f"n={m15['n']}  PF={m15['pf']:.3f}  Boot_p50={b50:.3f}  MC={mc['prob_profit']*100:.1f}%  "
    f"LOO-S={sym_floor:.3f}  LOO-F={fold_floor:.3f}  Score={passed}/7 — {overall}",
    color=TXT, fontsize=10, y=0.975
)
dash_path = f"{OUT}/r040_dashboard.png"
fig.savefig(dash_path, dpi=120, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"  → {dash_path}")

# =============================================================================
# TRADE LOG CSV
# =============================================================================

if all_trades:
    csv_path = f"{OUT}/r040_trade_log.csv"
    pd.DataFrame(all_trades).to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

# =============================================================================
# JOURNAL MARKDOWN
# =============================================================================

ck2 = lambda c: "✓" if c else "✗"
md_lines = [
    "# QUANTLAB AI — R040: Timeframe Transfer Validation (1H → 15m)\n",
    f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
    f"**Objective:** Pure timeframe transfer — apply R039 Var D environment to 15m candles  ",
    f"**Locked Environment:** ATR<p25 · EMA200_slope>0 · EMA_dist>p75 · BB<p50  ",
    f"**Entry:** RELVOL Breakout (vol>1.5× avg · bullish candle)  ",
    f"**Exit:** Stop=1×ATR14 · Target=2×ATR14 (2R)  ",
    f"**Method:** 5-fold expanding WF · IS thresholds only  ",
    f"**Timeframe:** 15m native candles  ",
    f"**Symbols available with 15m data:** {len(SYMBOLS)} ({', '.join(s.replace('-USDT-SWAP','') for s in SYMBOLS)})  \n",
    "## Side-by-Side Comparison\n",
    "| Metric | 1H (R039 Var D) | 15m (R040) | Δ | Pass? |",
    "|--------|----------------|-----------|---|-------|",
    f"| n (trades) | {R039['n']} | {m15['n']} | +{m15['n']-R039['n']} | {ck2(m15['n']>=100)} |",
    f"| Win Rate | {R039['wr']*100:.1f}% | {m15['wr']*100:.1f}% | {(m15['wr']-R039['wr'])*100:+.1f}% | — |",
    f"| Profit Factor | {R039['pf']:.3f} | {m15['pf']:.3f} | {m15['pf']-R039['pf']:+.3f} | {ck2(m15['pf']>1.20)} |",
    f"| Boot p50 | {R039['b50']:.3f} | {b50:.3f} | {b50-R039['b50']:+.3f} | {ck2(b50>1.20)} |",
    f"| Boot CI | — | [{b5:.3f}, {b95:.3f}] | — | — |",
    f"| MC P(profit) | {R039['mc_p']*100:.1f}% | {mc['prob_profit']*100:.1f}% | {(mc['prob_profit']-R039['mc_p'])*100:+.1f}% | {ck2(mc['prob_profit']>0.60)} |",
    f"| LOO-sym floor | {R039['sym_floor']:.3f} | {sym_floor:.3f} | {sym_floor-R039['sym_floor']:+.3f} | {ck2(sym_floor>1.00)} |",
    f"| LOO-fold floor | {R039['fold_floor']:.3f} | {fold_floor:.3f} | {fold_floor-R039['fold_floor']:+.3f} | {ck2(fold_floor>1.00)} |",
    f"| Max Drawdown | {R039['mdd']:.1%} | {m15['mdd']:.1%} | {m15['mdd']-R039['mdd']:+.1%} | {ck2(abs(m15['mdd'])<0.25)} |",
    "\n## Promotion Criteria\n",
    "| Criterion | Required | Actual | Pass? |",
    "|-----------|----------|--------|-------|",
]
crit_rows = [
    ("PF > 1.20",             "1.20",  f"{m15['pf']:.3f}"),
    ("n ≥ 100",               "100",   str(m15["n"])),
    ("Boot p50 > 1.20",       "1.20",  f"{b50:.3f}"),
    ("MC P > 60%",            "60%",   f"{mc['prob_profit']*100:.1f}%"),
    ("LOO-sym floor > 1.00",  "1.00",  f"{sym_floor:.3f}"),
    ("LOO-fold floor > 1.00", "1.00",  f"{fold_floor:.3f}"),
    ("MDD < 25%",             "25%",   f"{abs(m15['mdd'])*100:.1f}%"),
]
for crit_lbl, req, act in crit_rows:
    pass_ = criteria.get(crit_lbl.split(" ≥")[0].split(" > ")[0].strip() + \
                         (" > " + req if ">" in crit_lbl else
                          (" ≥ " + req if "≥" in crit_lbl else " < " + req)), None)
    # simpler: check directly
    md_lines.append(f"| {crit_lbl} | {req} | {act} | — |")

md_lines += [
    f"\n**Score: {passed}/7   Verdict: {overall}**\n",
    "\n## Research Questions\n",
    f"**Q1. Does 15m increase trade frequency ≥ 2×?**  ",
    f"1H n={R039['n']} → 15m n={m15['n']} (×{freq_change:.1f}) — {'YES ✓' if freq_2x else 'NO ✗'}\n",
    f"**Q2. Does PF remain > 1.20?**  ",
    f"1H PF={R039['pf']:.3f} → 15m PF={m15['pf']:.3f} — {'YES ✓' if q2_pass else 'NO ✗'}\n",
    f"**Q3. Does Bootstrap median remain > 1.20?**  ",
    f"1H p50={R039['b50']:.3f} → 15m p50={b50:.3f} CI=[{b5:.3f}, {b95:.3f}] — {'YES ✓' if q3_pass else 'NO ✗'}\n",
    f"**Q4. Does MC Probability remain > 60%?**  ",
    f"1H MC={R039['mc_p']*100:.1f}% → 15m MC={mc['prob_profit']*100:.1f}% — {'YES ✓' if q4_pass else 'NO ✗'}\n",
    f"**Q5. Does edge remain distributed across symbols?**  ",
    f"LOO-sym floor: 1H={R039['sym_floor']:.3f} → 15m={sym_floor:.3f} — {'YES ✓' if q5_pass else 'NO ✗'}\n",
    f"**Q6. Session clustering?**  ",
    f"Top-5 hours account for {top5_frac*100:.1f}% of trades. "
    f"{'Clustered — consider session filter.' if session_clustered else 'Distributed — consistent throughout day.'}\n",
    "\n**Q7. Per-symbol performance at 15m:**\n",
    "| Symbol | PF (15m) | n (15m) |",
    "|--------|----------|---------|",
]
for sym, pf_, n_ in sym_sorted_pf:
    md_lines.append(f"| {sym.replace('-USDT-SWAP','')} | {pf_:.3f} | {n_} |")

edge_transfers = q2_pass and q3_pass
md_lines += [
    "\n## Conclusion\n",
    f"**Timeframe transfer {'SUCCEEDED' if edge_transfers else 'FAILED'}.**  ",
    f"{'Edge preserves at 15m — environment is NOT timeframe-dependent.' if edge_transfers else 'Edge degrades at 15m — environment is timeframe-dependent. Recommend: remain 1H strategy.'}  \n",
    f"**Final Verdict: {overall}**\n",
    f"\n---\n*Generated by QUANTLAB AI R040 — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} UTC*",
]

jmd_path = f"{OUT}/r040_journal.md"
with open(jmd_path, "w") as f:
    f.write("\n".join(md_lines) + "\n")
print(f"  → {jmd_path}")

# =============================================================================
# JOURNAL CSV
# =============================================================================

journal_path = CONFIG["JOURNAL_FILE"]
row = {
    "research_id": RESEARCH_ID,
    "date":        pd.Timestamp.now().strftime("%Y-%m-%d"),
    "strategy":    "TF_TRANSFER_1H_15m_RELVOL_VarD",
    "timeframe":   "15m",
    "symbols":     str(len(SYMBOLS)),
    "method":      "tf-transfer-5fold-WF",
    "n_oos":       m15["n"],
    "wr":          round(m15["wr"], 4),
    "pf":          round(m15["pf"], 4),
    "sharpe":      round(m15["sharpe"], 4),
    "mdd":         round(m15["mdd"], 4),
    "net":         round(m15["net"], 2),
    "boot_p50":    round(b50, 4),
    "mc_prob":     round(mc["prob_profit"], 4),
    "loo_floor":   round(sym_floor, 4),
    "verdict":     overall,
}
jdf = pd.read_csv(journal_path) if os.path.exists(journal_path) else pd.DataFrame()
jdf = pd.concat([jdf, pd.DataFrame([row])], ignore_index=True)
jdf.to_csv(journal_path, index=False)
print(f"  → Journal: {journal_path}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "═"*100)
print(f"  R040 COMPLETE — Timeframe Transfer Validation (1H → 15m)")
print("═"*100)
print(f"  Dataset   : {len(SYMBOLS)} symbols · 15m candles · 5-fold WF")
print(f"  Environment: R039 Var D (LOCKED — no changes)")
print()
print(f"  {'Metric':<22}  {'1H (R039)':>12}  {'15m (R040)':>12}  {'Δ':>10}")
print("  " + "─"*60)

cmp_rows = [
    ("n (trades)",      R039["n"],      m15["n"]),
    ("Win Rate (%)",    R039["wr"]*100, m15["wr"]*100),
    ("Profit Factor",   R039["pf"],     m15["pf"]),
    ("Boot p50",        R039["b50"],    b50),
    ("MC P(profit)%",   R039["mc_p"]*100, mc["prob_profit"]*100),
    ("LOO-sym floor",   R039["sym_floor"], sym_floor),
    ("LOO-fold floor",  R039["fold_floor"], fold_floor),
    ("Max Drawdown (%)",abs(R039["mdd"])*100, abs(m15["mdd"])*100),
]
for lbl, v1h, v15 in cmp_rows:
    if isinstance(v1h, int) and isinstance(v15, int):
        print(f"  {lbl:<22}  {v1h:>12d}  {v15:>12d}  {v15-v1h:>+10d}")
    else:
        print(f"  {lbl:<22}  {v1h:>12.3f}  {v15:>12.3f}  {v15-v1h:>+10.3f}")

print()
print(f"  PROMOTION CRITERIA:")
for crit, result in criteria.items():
    mark = "✓" if result else "✗"
    print(f"    {mark}  {crit}")
print()
print(f"  SCORE: {passed}/7   VERDICT: {overall}")
print()
print(f"  Q1 (2× freq?):  {'YES ✓' if freq_2x else 'NO ✗'}   ×{freq_change:.1f}  (1H n={R039['n']} → 15m n={m15['n']})")
print(f"  Q2 (PF>1.20?):  {'YES ✓' if q2_pass else 'NO ✗'}   PF={m15['pf']:.3f}")
print(f"  Q3 (Boot>1.20): {'YES ✓' if q3_pass else 'NO ✗'}   p50={b50:.3f}")
print(f"  Q4 (MC>60%?):   {'YES ✓' if q4_pass else 'NO ✗'}   {mc['prob_profit']*100:.1f}%")
print(f"  Q5 (sym dist?): {'YES ✓' if q5_pass else 'NO ✗'}   LOO-sym={sym_floor:.3f}")
print(f"  Q6 (session?):  {'CLUSTERED' if session_clustered else 'DISTRIBUTED'}  top-5h={top5_frac*100:.1f}%")
print(f"  Q7 (best syms): {', '.join(x[0].replace('-USDT-SWAP','') for x in sym_sorted_pf[:5] if x[2]>0)}")
print()

edge_transfers = q2_pass and q3_pass
if edge_transfers and overall in ("PROMOTE", "WATCHLIST"):
    print(f"  ✓ Edge TRANSFERS to 15m — environment is NOT timeframe-dependent")
else:
    print(f"  ✗ Edge DEGRADES at 15m — environment is timeframe-dependent")
    print(f"    → Recommend: REMAIN 1H strategy (R039 Var D)")

print(f"""
  Output:
    {OUT}/r040_dashboard.png
    {OUT}/r040_equity_curves.png
    {OUT}/r040_comparison.png
    {OUT}/r040_session_analysis.png
    {OUT}/r040_symbol_analysis.png
    {OUT}/r040_bootstrap_ci.png
    {OUT}/r040_loo_robustness.png
    {OUT}/r040_journal.md
    {OUT}/r040_trade_log.csv
""" + "═"*100)
