"""
=============================================================================
QUANTLAB AI — RESEARCH #034
Objective: Validate Low ATR edge with maximum available history and
           expanded symbol set.  No strategy changes, no optimisation.

Method: Same 5-fold expanding walk-forward as R033.
        ATR p25 threshold learned from IS bars only (per fold, per symbol).
        All symbols with ≥ 4,000 1H bars included.

Research Questions:
  Q1  Portfolio PF > 1.20?
  Q2  Bootstrap median PF > 1.20?
  Q3  Monte Carlo P(profit) > 60%?
  Q4  WR significantly above 33.3% break-even? (binomial p < 0.01)
  Q5  Removing any single fold still leaves PF > 1.20?   ← NEW (addresses R033 audit)
  Q6  Removing any single symbol still leaves PF > 1.20?
  Q7  Which symbols consistently contribute / weaken the edge?
=============================================================================
"""

import os, sys, math, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import (CONFIG, calc_ema, calc_atr,
                          download_symbol, save_cache, load_cache,
                          OKX_HISTORY_URL, OKX_CANDLES_URL, CANDLE_COLS)

RESEARCH_ID = "R034"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

# ── Extended symbol candidates (≥ 4000 1H bars required to qualify) ──────────
SYMBOL_CANDIDATES = [
    # Core 9 from R033
    "BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
    "LINK-USDT-SWAP", "AVAX-USDT-SWAP", "XRP-USDT-SWAP",
    "LTC-USDT-SWAP", "BCH-USDT-SWAP", "DOGE-USDT-SWAP",
    # New candidates — included only if OKX has ≥ 4,000 1H bars
    "ADA-USDT-SWAP", "BNB-USDT-SWAP", "DOT-USDT-SWAP",
    "ARB-USDT-SWAP", "OP-USDT-SWAP", "NEAR-USDT-SWAP",
    "ATOM-USDT-SWAP", "SUI-USDT-SWAP", "APT-USDT-SWAP",
    "WIF-USDT-SWAP", "PEPE-USDT-SWAP", "ENA-USDT-SWAP",
    "UNI-USDT-SWAP", "FIL-USDT-SWAP",
]

MIN_BARS         = 4_000   # ~ 5.5 months; need enough for 5-fold WF
MONTHS_DOWNLOAD  = 30      # request 30 months to capture OKX's ~27-month history
CAPITAL          = CONFIG["STARTING_CAPITAL"]

COLOURS = {
    "BTC-USDT-SWAP": "#F7931A", "ETH-USDT-SWAP": "#627EEA",
    "SOL-USDT-SWAP": "#9945FF", "LINK-USDT-SWAP": "#2A5ADA",
    "AVAX-USDT-SWAP":"#E84142", "XRP-USDT-SWAP":  "#346AA9",
    "LTC-USDT-SWAP": "#BFBBBB", "BCH-USDT-SWAP":  "#8DC351",
    "DOGE-USDT-SWAP":"#C3A634", "ADA-USDT-SWAP":  "#0033AD",
    "BNB-USDT-SWAP": "#F3BA2F", "DOT-USDT-SWAP":  "#E6007A",
    "ARB-USDT-SWAP": "#28A0F0", "OP-USDT-SWAP":   "#FF0420",
    "NEAR-USDT-SWAP":"#00C08B", "ATOM-USDT-SWAP": "#6F4CFF",
    "SUI-USDT-SWAP": "#6FBCF0", "APT-USDT-SWAP":  "#00B4D8",
    "WIF-USDT-SWAP": "#A67C52", "PEPE-USDT-SWAP": "#4CAF50",
    "ENA-USDT-SWAP": "#8B0000", "UNI-USDT-SWAP":  "#FF007A",
    "FIL-USDT-SWAP": "#0090FF",
}

# Walk-forward folds — identical to R033
FOLDS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.00),
]

print("\n" + "╔" + "═"*79 + "╗")
print("║  QUANTLAB AI — RESEARCH #034" + " "*50 + "║")
print("║  Low ATR Validation — Maximum History + Expanded Symbol Set" + " "*19 + "║")
print("╚" + "═"*79 + "╝")
print(f"""
  Strategy    : FVG + EMA200 Slope + Low ATR (ATR Rank < IS p25)
  Method      : 5-fold expanding walk-forward (same as R033)
  Key change  : ~27 months of history (vs ~9/6 in R033) + up to 24 symbols
  Purpose     : Statistical power — does the edge survive larger OOS?
  PROMOTE bar : PF>1.20, n≥200, boot p50>1.20, MC P>60%,
                LOO-fold floor>1.20 (Q5 — NEW), LOO-sym floor>1.0
""")

# =============================================================================
# SECTION 1 — DATA: DOWNLOAD MAXIMUM HISTORY
# =============================================================================

def get_data_max(symbol: str) -> pd.DataFrame | None:
    """Load or download the maximum available history for a symbol.
    Forces a full re-download when the cached data covers < 15 months,
    ensuring we capture OKX's full ~27-month history-candles archive.
    """
    bar  = CONFIG["TIMEFRAME"]
    tag  = symbol.replace("-", "_")
    path = os.path.join(CACHE, f"{tag}_1H.parquet")

    cached = load_cache(symbol, bar)
    if cached is not None and len(cached) >= MIN_BARS:
        first_dt = cached["datetime"].iloc[0]
        months_cached = (pd.Timestamp.now(tz="UTC") - first_dt).days / 30.44
        if months_cached >= 15:
            # Cache has enough depth; fetch only the tail to update
            last_ts  = cached["datetime"].iloc[-1]
            bar_mins = 60
            gap = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds() / 60 / bar_mins
            if gap < 2:
                print(f"  {symbol.split('-')[0]:5s}  cache current  n={len(cached):,}  "
                      f"{cached.datetime.min().date()} → {cached.datetime.max().date()}")
                return cached
            # Incremental update
            since_ms = int(last_ts.timestamp() * 1000)
            new_df   = download_symbol(symbol, bar, months=0, since_ms=since_ms)
            if len(new_df):
                new_df = new_df[new_df["datetime"] > last_ts]
            if len(new_df):
                combined = (pd.concat([cached, new_df], ignore_index=True)
                            .drop_duplicates("datetime")
                            .sort_values("datetime")
                            .reset_index(drop=True))
                save_cache(combined, symbol, bar)
                print(f"  {symbol.split('-')[0]:5s}  +{len(new_df)} bars → n={len(combined):,}  "
                      f"{combined.datetime.min().date()} → {combined.datetime.max().date()}")
                return combined
            print(f"  {symbol.split('-')[0]:5s}  cache ok  n={len(cached):,}  "
                  f"{cached.datetime.min().date()} → {cached.datetime.max().date()}")
            return cached

    # Full (re-)download
    print(f"  {symbol.split('-')[0]:5s}  downloading {MONTHS_DOWNLOAD}M history …", end="", flush=True)
    try:
        df = download_symbol(symbol, bar, months=MONTHS_DOWNLOAD)
        save_cache(df, symbol, bar)
        print(f"  n={len(df):,}  {df.datetime.min().date()} → {df.datetime.max().date()}")
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# =============================================================================
# SECTION 2 — FEATURES (identical to R033 — confirmed causal in R034 audit)
# =============================================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    df["ema200"]        = calc_ema(c, 200)
    df["atr14"]         = calc_atr(df, 14)
    df["atr_rank_pct"]  = df["atr14"].rolling(100).rank(pct=True) * 100
    df["ema200_rising"] = df["ema200"] > df["ema200"].shift(10)
    df["high_2ago"]     = df["high"].shift(2)
    df["fvg_gap"]       = df["low"] > df["high_2ago"] * 1.0001
    df["prev_low"]      = df["low"].shift(1)
    return df


def signal_fvg_slope(df: pd.DataFrame) -> pd.Series:
    return (df["fvg_gap"] &
            (df["close"] > df["ema200"]) &
            df["ema200_rising"] &
            df["high_2ago"].notna()).fillna(False)


# =============================================================================
# SECTION 3 — BACKTEST ENGINE (identical to R033)
# =============================================================================

def run_backtest(df: pd.DataFrame, signal: pd.Series,
                 atr_threshold: float | None, sym_label: str,
                 fold: int) -> list:
    min_sl    = CONFIG["MIN_SL_PCT"]
    rr        = CONFIG["RISK_REWARD"]
    max_lev   = CONFIG["MAX_LEVERAGE"]
    capital   = CONFIG["STARTING_CAPITAL"]
    risk_frac = CONFIG["RISK_PER_TRADE_PCT"]
    fee_rate  = CONFIG["TAKER_FEE"]
    spd_rate  = CONFIG["SPREAD"] * 0.5
    slp_rate  = CONFIG["SL_SLIPPAGE"]

    in_pos = False
    entry_px = stop = take = pos_size = 0.0
    entry_tm = None; entry_i = -1
    trades = []

    for i in range(1, len(df)):
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        if in_pos:
            sl_hit = bar["low"]  <= stop
            tp_hit = bar["high"] >= take
            if sl_hit or tp_hit:
                exit_px   = (stop * (1.0 - slp_rate)) if sl_hit else take
                exit_type = "SL" if sl_hit else "TP"
                sl_dist   = entry_px - stop
                gross     = (exit_px - entry_px) * pos_size
                ne, nx    = entry_px * pos_size, exit_px * pos_size
                cost      = (ne + nx) * fee_rate + (ne + nx) * spd_rate
                slp_c     = (stop - exit_px) * pos_size if sl_hit else 0.0
                net       = gross - cost - slp_c
                rmul      = (exit_px - entry_px) / sl_dist if sl_dist > 0 else 0.0
                trades.append({
                    "sym":          sym_label,
                    "fold":         fold,
                    "entry_time":   str(entry_tm),
                    "exit_time":    str(bar["datetime"]),
                    "entry_price":  round(entry_px, 6),
                    "exit_price":   round(exit_px, 6),
                    "stop_loss":    round(stop, 6),
                    "take_profit":  round(take, 6),
                    "pnl":          round(net, 4),
                    "r_multiple":   round(rmul, 4),
                    "win":          int(exit_type == "TP"),
                    "exit_type":    exit_type,
                    "holding_hrs":  i - entry_i,
                    "atr_rank_pct": round(float(prev["atr_rank_pct"]), 2),
                })
                in_pos = False
            continue

        if signal.iloc[i - 1]:
            atr_pct = prev["atr_rank_pct"]
            if np.isnan(atr_pct):
                continue
            if atr_threshold is not None and atr_pct >= atr_threshold:
                continue

            ep      = bar["open"]
            sl      = prev["prev_low"]
            sl_dist = ep - sl
            if sl_dist <= 0 or sl_dist / ep < min_sl:
                sl      = prev["low"]
                sl_dist = ep - sl
                if sl_dist <= 0 or sl_dist / ep < min_sl:
                    continue

            tp       = ep + rr * sl_dist
            risk_usd = capital * risk_frac
            sz       = min(risk_usd / sl_dist, (capital * max_lev) / ep)

            entry_px = ep; stop = sl; take = tp
            pos_size = sz; entry_tm = bar["datetime"]; entry_i = i
            in_pos   = True

    return trades


# =============================================================================
# SECTION 4 — STATISTICS
# =============================================================================

def metrics(trades: list, label: str = "") -> dict:
    empty = {"label": label, "n": 0, "wr": 0.0, "pf": 0.0, "exp_r": 0.0,
             "net": 0.0, "sharpe": 0.0, "mdd": 0.0,
             "equity": np.array([CAPITAL]), "pnls": np.array([])}
    if not trades:
        return empty
    df   = pd.DataFrame(trades)
    pnl  = df["pnl"].values
    wins = df["win"].values.astype(bool)
    n, nw, nl = len(pnl), wins.sum(), (~wins).sum()
    gw  = pnl[wins].sum()       if nw else 0.0
    gl  = abs(pnl[~wins].sum()) if nl else 1e-9
    pf  = gw / gl
    wr  = nw / n
    rr  = CONFIG["RISK_REWARD"]
    exp_r  = wr * rr - (1.0 - wr)
    equity = CAPITAL + np.cumsum(pnl)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / peak
    mdd    = float(dd.min())
    bars_per_year = 365 * 24
    ann    = (equity[-1] / CAPITAL) ** (bars_per_year / max(n, 1)) - 1 if n > 1 else 0.0
    vol    = pnl.std() * math.sqrt(bars_per_year) if n > 1 else 1.0
    sharpe = ann / vol if vol != 0 else 0.0
    return {
        "label": label, "n": n, "wr": wr, "pf": pf, "exp_r": exp_r,
        "net": float(pnl.sum()), "sharpe": sharpe, "mdd": mdd,
        "equity": np.concatenate([[CAPITAL], equity]),
        "pnls": pnl,
    }


def bootstrap_pf(pnls: np.ndarray, n_iter: int = 5000, seed: int = 42) -> tuple:
    if len(pnls) < 10:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    pfs = []
    for _ in range(n_iter):
        s  = rng.choice(pnls, len(pnls), replace=True)
        wp = s[s > 0].sum(); lp = abs(s[s < 0].sum())
        pfs.append(wp / lp if lp > 0 else 2.0)
    return (float(np.percentile(pfs, 5)),
            float(np.percentile(pfs, 50)),
            float(np.percentile(pfs, 95)))


def monte_carlo(pnls: np.ndarray, n_iter: int = 5000, seed: int = 42) -> dict:
    if len(pnls) < 5:
        return {"prob_profit": 0.0, "p5": CAPITAL, "p50": CAPITAL,
                "p95": CAPITAL, "finals": np.array([CAPITAL])}
    rng    = np.random.default_rng(seed)
    finals = np.array([CAPITAL + rng.choice(pnls, len(pnls), replace=True).sum()
                       for _ in range(n_iter)])
    return {"prob_profit": float((finals > CAPITAL).mean()),
            "p5":  float(np.percentile(finals, 5)),
            "p50": float(np.percentile(finals, 50)),
            "p95": float(np.percentile(finals, 95)),
            "finals": finals}


def loo_pf_symbols(sym_trades: dict) -> dict:
    out = {}
    for omit in sym_trades:
        flat = [t for s, tl in sym_trades.items() if s != omit for t in tl]
        m    = metrics(flat)
        out[omit] = {"pf": m["pf"], "n": m["n"]}
    return out


def loo_pf_folds(all_trades: list) -> dict:
    """Leave-one-fold-out PF — key for Q5."""
    out = {}
    fold_ids = sorted({t["fold"] for t in all_trades})
    for omit_fold in fold_ids:
        flat = [t for t in all_trades if t["fold"] != omit_fold]
        m    = metrics(flat)
        out[omit_fold] = {"pf": m["pf"], "n": m["n"]}
    return out


# =============================================================================
# SECTION 5 — LOAD DATA
# =============================================================================

print("─"*78)
print("  Loading / updating symbol data …")
print()

all_dfs: dict[str, pd.DataFrame] = {}
skipped = []

for sym in SYMBOL_CANDIDATES:
    df = get_data_max(sym)
    if df is None or len(df) < MIN_BARS:
        reason = "no data" if df is None else f"only {len(df)} bars"
        skipped.append((sym, reason))
        continue
    all_dfs[sym] = add_features(df)

print()
if skipped:
    print(f"  Skipped ({len(skipped)} symbols — insufficient data):")
    for s, r in skipped:
        print(f"    {s}: {r}")

SYMBOLS = list(all_dfs.keys())
print(f"\n  Qualified: {len(SYMBOLS)} symbols")
print(f"  {'Symbol':22s}  {'Bars':>7}  {'From':>12}  {'To':>12}")
print("  " + "─"*52)
total_bar_years = 0
for sym, df in all_dfs.items():
    nbars = len(df)
    total_bar_years += nbars / 8760
    print(f"  {sym:22s}  {nbars:7,}  {df.datetime.min().date()}  {df.datetime.max().date()}")
print(f"\n  Total data: {sum(len(d) for d in all_dfs.values()):,} bars  "
      f"({total_bar_years:.1f} instrument-years)")
print()


# =============================================================================
# SECTION 6 — WALK-FORWARD BACKTESTS
# =============================================================================

print("─"*78)
print(f"  Running walk-forward — {len(FOLDS)} folds × {len(SYMBOLS)} symbols × 2 variants …")
print()

sym_base_trades: dict[str, list] = {s: [] for s in SYMBOLS}
sym_low_trades:  dict[str, list] = {s: [] for s in SYMBOLS}
fold_summaries   = []
fold_thresholds  = {s: [] for s in SYMBOLS}  # (fold, threshold)

for fold_idx, (is_end, oos_end) in enumerate(FOLDS, start=1):
    fold_base = []; fold_low = []
    fold_thrs = []

    for sym, df_full in all_dfs.items():
        N       = len(df_full)
        is_cut  = int(N * is_end)
        oos_cut = int(N * oos_end)

        df_is  = df_full.iloc[:is_cut]
        df_oos = df_full.iloc[is_cut:oos_cut].reset_index(drop=True)

        if len(df_oos) < 100:
            continue

        is_atr_p25 = float(df_is["atr_rank_pct"].dropna().quantile(0.25))
        fold_thresholds[sym].append((fold_idx, is_atr_p25))
        fold_thrs.append(is_atr_p25)

        sig = signal_fvg_slope(df_oos)
        bt  = run_backtest(df_oos, sig, None,       sym, fold_idx)
        lt  = run_backtest(df_oos, sig, is_atr_p25, sym, fold_idx)

        sym_base_trades[sym].extend(bt)
        sym_low_trades[sym].extend(lt)
        fold_base.extend(bt)
        fold_low.extend(lt)

    fb = metrics(fold_base); fl = metrics(fold_low)
    thr_mean = np.mean(fold_thrs) if fold_thrs else 0.0
    print(f"  Fold {fold_idx}  (IS={is_end*100:.0f}%→OOS={oos_end*100:.0f}%)  "
          f"avg_thr={thr_mean:.1f}  "
          f"Base: n={fb['n']:4d} PF={fb['pf']:.3f}  "
          f"LowATR: n={fl['n']:4d} PF={fl['pf']:.3f}")
    fold_summaries.append((fold_idx, is_end, oos_end, fb["pf"], fl["pf"], fb["n"], fl["n"]))

print()

# =============================================================================
# SECTION 7 — AGGREGATE METRICS
# =============================================================================

all_base_flat = [t for s in SYMBOLS for t in sym_base_trades[s]]
all_low_flat  = [t for s in SYMBOLS for t in sym_low_trades[s]]

port_base = metrics(all_base_flat, "Baseline (WF)")
port_low  = metrics(all_low_flat,  "Low ATR (WF)")

print("═"*78)
print(f"  PORTFOLIO RESULTS — Walk-Forward OOS (5 folds, {len(SYMBOLS)} symbols)")
print("═"*78)
print(f"\n  {'Variant':16s}  {'n':>5}  {'WR':>6}  {'PF':>7}  "
      f"{'ExpR':>7}  {'Sharpe':>7}  {'MDD':>7}  {'Net $':>9}")
print("  " + "─"*68)
for label, m in [("Baseline", port_base), ("Low ATR", port_low)]:
    print(f"  {label:16s}  {m['n']:5d}  {m['wr']*100:5.1f}%  {m['pf']:7.3f}  "
          f"{m['exp_r']:+7.3f}  {m['sharpe']:7.2f}  {m['mdd']*100:6.1f}%  "
          f"{m['net']:+9.0f}")

dpf = port_low["pf"] - port_base["pf"]
print(f"\n  δPF (Low ATR vs Baseline): {dpf:+.3f}")
print(f"  WR shift: {port_base['wr']*100:.1f}% → {port_low['wr']*100:.1f}%")
print(f"  Trade retention: {port_low['n']}/{port_base['n']} "
      f"= {port_low['n']/max(port_base['n'],1)*100:.0f}%")

# =============================================================================
# SECTION 8 — FOLD-BY-FOLD TABLE
# =============================================================================

print(f"\n  {'Fold':>5}  {'IS%':>5}→{'OOS%':>4}  "
      f"{'Base PF':>9}  {'LowATR PF':>10}  {'n(base)':>8}  {'n(low)':>7}  {'δPF':>8}")
print("  " + "─"*64)
for fi, is_end, oos_end, bpf, lpf, nb, nl in fold_summaries:
    arrow = "↑" if lpf > bpf else "↓"
    print(f"  {fi:5d}  {is_end*100:4.0f}%→{oos_end*100:3.0f}%  "
          f"{bpf:9.3f}  {lpf:10.3f}  {nb:8d}  {nl:7d}  {lpf-bpf:+8.3f} {arrow}")

# =============================================================================
# SECTION 9 — PER-SYMBOL TABLE (Q7)
# =============================================================================

print("\n" + "═"*78)
print(f"  PER-SYMBOL RESULTS (Q7) — Low ATR vs Baseline (all folds combined)")
print("═"*78)
print(f"\n  {'Symbol':6s}  {'Base n':>7}  {'Base PF':>8}  {'Low n':>7}  "
      f"{'Low PF':>8}  {'WR':>6}  {'δPF':>8}  {'Contribute?':>12}")
print("  " + "─"*73)

n_improved = 0
sym_ranks  = []

for sym in SYMBOLS:
    tag  = sym.split("-")[0]
    bm   = metrics(sym_base_trades[sym])
    lm   = metrics(sym_low_trades[sym])
    dpf_ = lm["pf"] - bm["pf"]
    ok   = dpf_ > 0 and lm["n"] > 0
    if ok:
        n_improved += 1
    flag = "✓ positive" if ok else "✗ drag"
    print(f"  {tag:6s}  {bm['n']:7d}  {bm['pf']:8.3f}  {lm['n']:7d}  "
          f"{lm['pf']:8.3f}  {lm['wr']*100:5.1f}%  {dpf_:+8.3f}  {flag:>12}")
    sym_ranks.append((sym, lm["n"], lm["pf"], lm["wr"], dpf_))

print(f"\n  Low ATR improved {n_improved}/{len(SYMBOLS)} symbols")

sym_ranks.sort(key=lambda x: x[2], reverse=True)
print(f"\n  Symbol ranking by Low ATR PF (Q7):")
print(f"  {'Rank':>4}  {'Symbol':8s}  {'n':>5}  {'PF':>7}  {'WR':>6}  {'δPF':>8}  {'Tier':>12}")
print("  " + "─"*55)
for rank, (sym, n_, pf_, wr_, dpf_) in enumerate(sym_ranks, 1):
    tag = sym.split("-")[0]
    tier = ("★ Core"    if pf_ >= 1.30 else
            "✓ Support" if pf_ >= 1.10 else
            "~ Neutral" if pf_ >= 0.95 else "✗ Drag")
    print(f"  {rank:4d}  {tag:8s}  {n_:5d}  {pf_:7.3f}  {wr_*100:5.1f}%  {dpf_:+8.3f}  {tier:>12}")

# =============================================================================
# SECTION 10 — STATISTICAL TESTS
# =============================================================================

print("\n" + "═"*78)
print("  STATISTICAL ANALYSIS")
print("═"*78)

pnls_low = port_low["pnls"]
b05, b50, b95 = bootstrap_pf(pnls_low)
mc = monte_carlo(pnls_low)

n_low  = port_low["n"]
wr_low = port_low["wr"]
bep_wr = 1.0 / (1.0 + CONFIG["RISK_REWARD"])  # 33.33%

binom_result = scipy_stats.binomtest(int(round(wr_low * n_low)),
                                     n_low, bep_wr, alternative="greater")

print(f"\n  Bootstrap PF  (5000 iter):  p5={b05:.3f}  p50={b50:.3f}  p95={b95:.3f}")
print(f"  Monte Carlo P(profit):     {mc['prob_profit']*100:.1f}%  "
      f"(p5=${mc['p5']:,.0f}  p50=${mc['p50']:,.0f}  p95=${mc['p95']:,.0f})")
print(f"\n  Binomial test — WR above break-even:")
print(f"    n={n_low}  WR={wr_low*100:.1f}%  break-even={bep_wr*100:.1f}%  "
      f"p={binom_result.pvalue:.6f}")
print(f"    {'SIGNIFICANT (p < 0.01) ✓' if binom_result.pvalue < 0.01 else 'NOT significant ✗'}")

# =============================================================================
# SECTION 11 — Q5: LEAVE-ONE-FOLD (Critical audit finding from R033)
# =============================================================================

print("\n" + "─"*78)
print("  Q5 — LEAVE-ONE-FOLD ANALYSIS")
print("─"*78)
print(f"\n  {'Excl. Fold':>11}  {'n':>5}  {'WR':>6}  {'PF':>7}  {'Boot p50':>9}  {'PROMOTE?'}")
print("  " + "─"*56)

loo_fold_results = loo_pf_folds(all_low_flat)
q5_all_above_120 = True
q5_floor = 9.99

for fold_id, res in sorted(loo_fold_results.items()):
    sub = [t for t in all_low_flat if t["fold"] != fold_id]
    sub_m = metrics(sub)
    _, bp50, _ = bootstrap_pf(sub_m["pnls"])
    ok  = sub_m["pf"] > 1.20
    if not ok:
        q5_all_above_120 = False
    q5_floor = min(q5_floor, sub_m["pf"])
    flag = "✓ PROMOTE" if sub_m["pf"] > 1.20 else ("~ WATCHLIST" if sub_m["pf"] > 1.0 else "✗ REJECT")
    # Mark which fold was fold 1 from R033 (highest)
    fold_n = sum(1 for t in all_low_flat if t["fold"] == fold_id)
    print(f"  Excl. fold {fold_id}  {sub_m['n']:5d}  {sub_m['wr']*100:5.1f}%  "
          f"{sub_m['pf']:7.3f}  {bp50:9.3f}  {flag}")

print(f"\n  LOO-fold PF floor: {q5_floor:.3f}")
print(f"  Q5 PASS (all folds > 1.20): {'✓ YES' if q5_all_above_120 else '✗ NO'}")

# =============================================================================
# SECTION 12 — Q6: LEAVE-ONE-SYMBOL
# =============================================================================

print("\n" + "─"*78)
print("  Q6 — LEAVE-ONE-SYMBOL ANALYSIS")
print("─"*78)
print(f"\n  {'Excl. Symbol':>12}  {'n':>5}  {'PF':>7}  {'PROMOTE?'}")
print("  " + "─"*38)

loo_sym_results = loo_pf_symbols(sym_low_trades)
q6_all_positive = True
q6_floor = 9.99

for sym in sorted(loo_sym_results, key=lambda s: loo_sym_results[s]["pf"]):
    res = loo_sym_results[sym]
    tag = sym.split("-")[0]
    ok  = res["pf"] > 1.0
    if not ok:
        q6_all_positive = False
    q6_floor = min(q6_floor, res["pf"])
    flag = "✓" if res["pf"] > 1.20 else ("~" if res["pf"] > 1.0 else "✗")
    print(f"  Excl. {tag:8s}  {res['n']:5d}  {res['pf']:7.3f}  {flag}")

print(f"\n  LOO-symbol PF floor: {q6_floor:.3f}")
print(f"  Q6 PASS (all symbols > 1.0): {'✓ YES' if q6_all_positive else '✗ NO'}")

# =============================================================================
# SECTION 13 — PROMOTE / WATCHLIST / REJECT VERDICT
# =============================================================================

print("\n" + "═"*78)
print("  VERDICT CRITERIA")
print("═"*78)

criteria = [
    # (label, result, required_for_promote)
    ("Q1  PF > 1.20",              port_low["pf"] > 1.20,             True),
    ("Q1b n ≥ 200",                port_low["n"]  >= 200,             True),
    ("Q2  Boot p50 > 1.20",        b50 > 1.20,                        True),
    ("Q3  MC P(profit) > 60%",     mc["prob_profit"] > 0.60,          True),
    ("Q4  WR binomial p < 0.01",   binom_result.pvalue < 0.01,        True),
    ("Q5  LOO-fold floor > 1.20",  q5_all_above_120,                  True),  # NEW
    ("Q6  LOO-sym floor > 1.0",    q6_all_positive,                   False), # advisory
    (f"Q7  ≥50% symbols positive", n_improved >= len(SYMBOLS) // 2,   False), # advisory
]

n_pass  = sum(1 for _, v, req in criteria if v and req)
n_req   = sum(1 for _, _, req in criteria if req)
n_total = len(criteria)

print()
for label, passed, required in criteria:
    mark = "✓" if passed else "✗"
    req  = "(required)" if required else "(advisory)"
    print(f"  {mark} {label:40s} {req}")

print()
if all(v for _, v, req in criteria if req):
    VERDICT = "PROMOTE"
elif port_low["pf"] > 1.0 and n_pass >= n_req - 1:
    VERDICT = "WATCHLIST"
else:
    VERDICT = "REJECT"

print(f"  Required criteria: {n_pass}/{n_req} passed")
VERDICT_LINE = f"  ► VERDICT: {VERDICT}"
print()
print("  " + "═"*40)
print(VERDICT_LINE)
print("  " + "═"*40)

# =============================================================================
# SECTION 14 — CHARTS
# =============================================================================

print("\n  Generating charts …")

# ── 14a. Per-symbol equity curves ─────────────────────────────────────────────
for sym in SYMBOLS:
    tag  = sym.split("-")[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor("#0d1117")
    colour = COLOURS.get(sym, "#888888")
    for ax, label, trades_ in zip(axes,
                                   ["Baseline", "Low ATR"],
                                   [sym_base_trades[sym], sym_low_trades[sym]]):
        m = metrics(trades_, label)
        ax.set_facecolor("#161b22")
        if m["n"] > 0:
            ax.plot(m["equity"], color=colour, lw=1.2)
            ax.axhline(CAPITAL, color="#444", lw=0.8, ls="--")
        ax.set_title(f"{tag} {label}  n={m['n']}  PF={m['pf']:.3f}  WR={m['wr']*100:.1f}%",
                     color="#e0e0e0", fontsize=9)
        ax.tick_params(colors="#888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
    fig.suptitle(f"R034 — {tag}", color="#e0e0e0", fontsize=11)
    fig.tight_layout()
    path = f"{OUT}/r034_{tag.lower()}.png"
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)

print(f"  → {len(SYMBOLS)} per-symbol charts saved")

# ── 14b. Main dashboard ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22), facecolor="#0d1117")
gs  = gridspec.GridSpec(5, 3, figure=fig,
                        hspace=0.45, wspace=0.32,
                        top=0.94, bottom=0.04, left=0.06, right=0.97)

ax_port = fig.add_subplot(gs[0, :2])
ax_dd   = fig.add_subplot(gs[1, :2])
ax_dist = fig.add_subplot(gs[0, 2])
ax_mc   = fig.add_subplot(gs[1, 2])
ax_fold = fig.add_subplot(gs[2, :2])
ax_sym  = fig.add_subplot(gs[2, 2])
ax_loo_fold = fig.add_subplot(gs[3, :])
ax_loo_sym  = fig.add_subplot(gs[4, :])

_dark = "#0d1117"; _panel = "#161b22"; _txt = "#e0e0e0"
_green = "#2ea043"; _red = "#cf222e"; _blue = "#58a6ff"

def _style(ax, title=""):
    ax.set_facecolor(_panel)
    ax.tick_params(colors="#888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    if title:
        ax.set_title(title, color=_txt, fontsize=9, pad=4)

# Portfolio equity
_style(ax_port, f"Portfolio Equity — Low ATR Walk-Forward OOS  "
                f"n={port_low['n']}  PF={port_low['pf']:.3f}  "
                f"WR={port_low['wr']*100:.1f}%  MDD={port_low['mdd']*100:.1f}%")
eq = port_low["equity"]
ax_port.plot(eq, color=_blue, lw=1.5, label="Low ATR")
ax_port.plot(port_base["equity"], color="#888", lw=0.8, ls="--", label="Baseline", alpha=0.5)
ax_port.axhline(CAPITAL, color="#444", lw=0.8, ls=":")
ax_port.fill_between(range(len(eq)), CAPITAL, eq,
                     where=eq >= CAPITAL, alpha=0.15, color=_green)
ax_port.fill_between(range(len(eq)), CAPITAL, eq,
                     where=eq < CAPITAL, alpha=0.15, color=_red)
ax_port.legend(fontsize=7, facecolor=_panel, labelcolor=_txt)
ax_port.set_ylabel("Equity ($)", color="#888", fontsize=8)

# Drawdown
_style(ax_dd, "Portfolio Drawdown")
peak = np.maximum.accumulate(eq)
dd   = (eq - peak) / peak * 100
ax_dd.fill_between(range(len(dd)), dd, 0, color=_red, alpha=0.4)
ax_dd.plot(dd, color=_red, lw=0.8)
ax_dd.axhline(0, color="#444", lw=0.8)
ax_dd.set_ylabel("Drawdown (%)", color="#888", fontsize=8)

# PnL distribution
_style(ax_dist, f"PnL Distribution  n={n_low}")
ax_dist.hist(pnls_low[pnls_low >= 0], bins=30, color=_green, alpha=0.7, label="Wins")
ax_dist.hist(pnls_low[pnls_low < 0],  bins=30, color=_red,   alpha=0.7, label="Losses")
ax_dist.axvline(0, color="#888", lw=0.8)
ax_dist.legend(fontsize=7, facecolor=_panel, labelcolor=_txt)

# Monte Carlo
_style(ax_mc, f"Monte Carlo (5000 iter)  P(profit)={mc['prob_profit']*100:.1f}%")
ax_mc.hist(mc["finals"], bins=50, color=_blue, alpha=0.6, edgecolor="none")
ax_mc.axvline(CAPITAL, color="#888", lw=1, ls="--", label="Break-even")
ax_mc.axvline(mc["p50"], color=_green, lw=1.2, ls="--", label=f"p50=${mc['p50']:,.0f}")
ax_mc.legend(fontsize=7, facecolor=_panel, labelcolor=_txt)
ax_mc.set_xlabel("Final Equity ($)", color="#888", fontsize=7)

# Fold PF bar chart
_style(ax_fold, "Fold-by-Fold PF — Baseline vs Low ATR")
fold_x    = [f"Fold {f[0]}\n({f[1]*100:.0f}%→{f[2]*100:.0f}%)" for f in fold_summaries]
base_pfs  = [f[3] for f in fold_summaries]
low_pfs   = [f[4] for f in fold_summaries]
xs = np.arange(len(fold_x))
ax_fold.bar(xs - 0.2, base_pfs, 0.38, color="#888", alpha=0.6, label="Baseline")
ax_fold.bar(xs + 0.2, low_pfs,  0.38,
            color=[_green if p >= 1.20 else (_blue if p >= 1.0 else _red) for p in low_pfs],
            alpha=0.85, label="Low ATR")
ax_fold.axhline(1.20, color="#f0c040", lw=1, ls="--", label="1.20 threshold")
ax_fold.axhline(1.0,  color="#888",    lw=0.8, ls=":")
ax_fold.set_xticks(xs); ax_fold.set_xticklabels(fold_x, fontsize=7, color="#888")
ax_fold.set_ylabel("Profit Factor", color="#888", fontsize=8)
ax_fold.legend(fontsize=7, facecolor=_panel, labelcolor=_txt)

# Symbol PF scatter
_style(ax_sym, "Symbol PF (Low ATR)")
sym_names = [s.split("-")[0] for s, *_ in sym_ranks]
sym_pfs   = [r[2] for r in sym_ranks]
colours_sym = [_green if p >= 1.20 else (_blue if p >= 1.0 else _red) for p in sym_pfs]
ax_sym.barh(sym_names, sym_pfs, color=colours_sym, alpha=0.8)
ax_sym.axvline(1.20, color="#f0c040", lw=1, ls="--")
ax_sym.axvline(1.0,  color="#888",    lw=0.8, ls=":")
ax_sym.set_xlabel("Profit Factor", color="#888", fontsize=8)
ax_sym.invert_yaxis()

# LOO-fold bar chart
_style(ax_loo_fold, "Q5 — Leave-One-Fold: Portfolio PF when each fold is excluded")
loo_fold_xs  = sorted(loo_fold_results.keys())
loo_fold_pfs = [loo_fold_results[f]["pf"] for f in loo_fold_xs]
loo_fold_ns  = [loo_fold_results[f]["n"]  for f in loo_fold_xs]
loo_bar_c    = [_green if p >= 1.20 else (_blue if p >= 1.0 else _red) for p in loo_fold_pfs]
bars = ax_loo_fold.bar([f"Excl Fold {f}" for f in loo_fold_xs],
                        loo_fold_pfs, color=loo_bar_c, alpha=0.85)
ax_loo_fold.axhline(1.20, color="#f0c040", lw=1.2, ls="--", label="1.20 threshold")
ax_loo_fold.axhline(1.0,  color="#888",    lw=0.8, ls=":")
for bar_, n_ in zip(bars, loo_fold_ns):
    ax_loo_fold.text(bar_.get_x() + bar_.get_width()/2, bar_.get_height() + 0.003,
                      f"n={n_}", ha="center", va="bottom", fontsize=7, color="#aaa")
ax_loo_fold.set_ylabel("Portfolio PF", color="#888", fontsize=8)
ax_loo_fold.legend(fontsize=7, facecolor=_panel, labelcolor=_txt)

# LOO-symbol bar chart
_style(ax_loo_sym, "Q6 — Leave-One-Symbol: Portfolio PF when each symbol is excluded")
loo_sym_sorted = sorted(loo_sym_results.items(), key=lambda x: x[1]["pf"])
loo_sym_labels = [s.split("-")[0] for s, _ in loo_sym_sorted]
loo_sym_pfs    = [v["pf"]  for _, v in loo_sym_sorted]
loo_sym_ns     = [v["n"]   for _, v in loo_sym_sorted]
loo_sym_c      = [_green if p >= 1.20 else (_blue if p >= 1.0 else _red) for p in loo_sym_pfs]
bars2 = ax_loo_sym.barh(loo_sym_labels, loo_sym_pfs, color=loo_sym_c, alpha=0.8)
ax_loo_sym.axvline(1.20, color="#f0c040", lw=1.2, ls="--", label="1.20")
ax_loo_sym.axvline(1.0,  color="#888",    lw=0.8, ls=":")
ax_loo_sym.set_xlabel("Portfolio PF", color="#888", fontsize=8)
ax_loo_sym.legend(fontsize=7, facecolor=_panel, labelcolor=_txt)

fig.suptitle(
    f"QUANTLAB AI — R034 — Low ATR Walk-Forward Validation\n"
    f"n={port_low['n']} trades  PF={port_low['pf']:.3f}  WR={port_low['wr']*100:.1f}%  "
    f"Symbols={len(SYMBOLS)}  Boot p50={b50:.3f}  MC={mc['prob_profit']*100:.1f}%  "
    f"Verdict: {VERDICT}",
    color=_txt, fontsize=12, y=0.975
)

dash_path = f"{OUT}/r034_dashboard.png"
fig.savefig(dash_path, dpi=120, facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  → {dash_path}")

# =============================================================================
# SECTION 15 — TRADE LOG & JOURNAL
# =============================================================================

trade_log_path = f"{OUT}/r034_trade_log.csv"
pd.DataFrame(all_low_flat).to_csv(trade_log_path, index=False)
print(f"  → {trade_log_path}  ({len(all_low_flat)} trades)")

journal_path = CONFIG["JOURNAL_FILE"]
journal_row  = {
    "research_id": RESEARCH_ID,
    "date":        pd.Timestamp.now().strftime("%Y-%m-%d"),
    "strategy":    "FVG+EMA200Slope+LowATR",
    "timeframe":   "1H",
    "symbols":     ",".join(s.split("-")[0] for s in SYMBOLS),
    "method":      "walk-forward-5fold-maxhistory",
    "n_oos":       port_low["n"],
    "wr":          round(port_low["wr"], 4),
    "pf":          round(port_low["pf"], 4),
    "sharpe":      round(port_low["sharpe"], 4),
    "mdd":         round(port_low["mdd"], 4),
    "net":         round(port_low["net"], 2),
    "boot_p50":    round(b50, 4),
    "mc_prob":     round(mc["prob_profit"], 4),
    "loo_floor":   round(q5_floor, 4),
    "verdict":     VERDICT,
}
journal_df = pd.read_csv(journal_path) if os.path.exists(journal_path) else pd.DataFrame()
journal_df = pd.concat([journal_df, pd.DataFrame([journal_row])], ignore_index=True)
journal_df.to_csv(journal_path, index=False)
print(f"  Journal updated → {journal_path}")

# =============================================================================
# SECTION 16 — SUMMARY
# =============================================================================

print("\n" + "═"*78)
print(f"  R034 complete.")
print(f"  Verdict      : {VERDICT}  ({n_pass}/{n_req} required criteria)")
print(f"  Symbols      : {len(SYMBOLS)}")
print(f"  n (Low ATR)  : {port_low['n']}  Baseline: {port_base['n']}")
print(f"  PF           : Base={port_base['pf']:.3f}  LowATR={port_low['pf']:.3f}  δ={dpf:+.3f}")
print(f"  WR           : Base={port_base['wr']*100:.1f}%  LowATR={port_low['wr']*100:.1f}%")
print(f"  MDD          : {port_low['mdd']*100:.1f}%")
print(f"  Boot p50     : {b50:.3f}")
print(f"  MC P(profit) : {mc['prob_profit']*100:.1f}%")
print(f"  Q5 LOO-fold  : floor={q5_floor:.3f}  {'PASS ✓' if q5_all_above_120 else 'FAIL ✗'}")
print(f"  Q6 LOO-sym   : floor={q6_floor:.3f}  {'PASS ✓' if q6_all_positive else 'FAIL ✗'}")
print(f"  Output       : {OUT}/r034_*")
print("═"*78)
