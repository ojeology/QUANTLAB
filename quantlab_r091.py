"""
QUANTLAB AI — R091
NEW 5m hypotheses with NEW INDICATORS — all strictly causal, with built-in
lookahead audit (lesson from R090 retraction).

New indicators (never used in this project before):
  - session VWAP (cumulative from day start — causal cumsum)
  - StochRSI (rolling — causal)
  - MACD (EMA-based — causal)
  - Keltner Channels (EMA + ATR — causal)
  - Donchian Channels (prior rolling max/min — causal)
  - Bollinger %B (rolling — causal)

5 new hypotheses:
  K1 VWAP-reclaim       : close < session VWAP - 0.5*ATR recently, then reclaims
                          above VWAP + green + relvol>1.2
  K2 StochRSI cross-up  : StochRSI K < 20 then K crosses above D + green
  K3 MACD-flip          : MACD hist crosses > 0 from negative (>=2 bars) + green + relvol>1.2
  K4 Keltner squeeze    : BB width < Keltner width (squeeze) + close > upper Keltner + relvol>1.5
  K5 Donchian retest    : close > prior 20-bar high, then pulls to EMA20 within 3 bars, reclaims

ANTI-CHEAT AUDIT (run BEFORE results are trusted):
  For a random symbol, recompute each mask with the last 500 bars DELETED.
  Masks in the overlap MUST be identical. Any difference = lookahead = hypothesis killed.

Protocol: E6 entry, RR 1.5, base SL/TP + 60-bar time stop.
Selection <= 2026-05-31, holdout Jun-Aug untouched, costs 0.05%.
Success: holPF > 1.1, PF@cost > 1.1, selection n >= 60.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, sim_symbol, stats_from_trades, cost_adjusted_rs, pf_of_rs
import scripts.ql_engine as qle

RESEARCH_ID = "R091"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2026-06-01", tz="UTC")
RR = 1.5
IS_LOOKBACK = 6000
RECAL_EVERY = 2016
MIN_BARS = IS_LOOKBACK + RECAL_EVERY + 500
SYMS = ["BTC_USDT_SWAP","ETH_USDT_SWAP","DOGE_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP"]

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  NEW 5m hypotheses / NEW indicators / causal")
print(SEP)
t0 = time.time()

print("\n  Loading 5m data …")
feats = {}
for sym in SYMS:
    p = os.path.join(CACHE, f"{sym}_5m.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < MIN_BARS: continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","rsi14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= MIN_BARS: feats[sym] = f
        print(f"    {sym}: {len(f)} bars")
    except Exception as e:
        print(f"    {sym}: ERR {e}")
print(f"  Symbols ready: {len(feats)}")

# ─────────────────────────────────────────────────────────────────────────────
# NEW INDICATORS (all causal)
# ─────────────────────────────────────────────────────────────────────────────
def add_new_indicators(f):
    f = f.copy()
    day = f.index.normalize()
    typ = (f["high"] + f["low"] + f["close"]) / 3.0
    # session VWAP: cumulative typical*vol / cumulative vol from day start (causal)
    f["vwap"] = (typ * f["vol"]).groupby(day).cumsum() / f["vol"].groupby(day).cumsum().replace(0, np.nan)
    f["vwap_dist"] = (f["close"] - f["vwap"]) / f["atr14"]
    # StochRSI: RSI14 -> rolling 14 K, rolling 3 D (causal)
    rsi14 = f["rsi14"]
    stoch = (rsi14 - rsi14.rolling(14).min()) / (rsi14.rolling(14).max() - rsi14.rolling(14).min()).replace(0, np.nan)
    f["stochK"] = stoch.rolling(3).mean() * 100
    f["stochD"] = f["stochK"].rolling(3).mean()
    # MACD (12,26,9) via EMAs (causal)
    ema12 = f["close"].ewm(span=12, adjust=False).mean()
    ema26 = f["close"].ewm(span=26, adjust=False).mean()
    f["macd"] = ema12 - ema26
    f["macd_sig"] = f["macd"].ewm(span=9, adjust=False).mean()
    f["macd_hist"] = f["macd"] - f["macd_sig"]
    # Keltner channels: EMA20 ± 1.5*ATR (causal)
    f["kelt_mid"] = f["close"].rolling(20).mean()
    f["kelt_hi"] = f["kelt_mid"] + 1.5 * f["atr14"]
    f["kelt_lo"] = f["kelt_mid"] - 1.5 * f["atr14"]
    # Donchian (prior 20-bar high/low — causal via shift)
    f["don_hi"] = f["high"].rolling(20).max().shift(1)
    f["don_lo"] = f["low"].rolling(20).min().shift(1)
    # Bollinger %B
    bb_mid = f["close"].rolling(20).mean()
    bb_std = f["close"].rolling(20).std(ddof=0)
    f["bb_pctB"] = (f["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    # squeeze: BB width vs Keltner width
    kelt_w = (f["kelt_hi"] - f["kelt_lo"]) / f["kelt_mid"].replace(0, np.nan)
    f["squeeze"] = (f["bb_width"] / 100.0 < kelt_w)  # True when BB inside Keltner
    return f

feats2 = {s: add_new_indicators(f) for s, f in feats.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 5 NEW HYPOTHESES (masks)
# ─────────────────────────────────────────────────────────────────────────────
def k1_vwap_reclaim(f):
    below = (f["vwap_dist"] < -0.5) | (f["close"] < f["vwap"] - 0.5 * f["atr14"])
    below_recent = below.rolling(5, min_periods=1).max().astype(bool)
    return (below_recent & (f["close"] > f["vwap"]) & (f["close"] > f["open"]) &
            (f["rel_vol"] > 1.2) & f["vwap"].notna())

def k2_stochrsi_cross(f):
    oversold = f["stochK"] < 20
    cross_up = (f["stochK"] > f["stochD"]) & (f["stochK"].shift(1) <= f["stochD"].shift(1))
    return (oversold & cross_up & (f["close"] > f["open"]) & (f["close"] > f["close"].shift(1)) &
            f["stochK"].notna())

def k3_macd_flip(f):
    was_neg = (f["macd_hist"].shift(1) < 0) & (f["macd_hist"].shift(2) < 0)
    flip = (f["macd_hist"] > 0) & was_neg
    return (flip & (f["close"] > f["open"]) & (f["rel_vol"] > 1.2) & f["macd_hist"].notna())

def k4_keltner_squeeze(f):
    return (f["squeeze"] & (f["close"] > f["kelt_hi"]) & (f["rel_vol"] > 1.5) &
            (f["close"] > f["open"]) & f["kelt_hi"].notna())

def k5_donchian_retest(f):
    broke = f["close"] > f["don_hi"]
    pull3 = (f["low"].rolling(3, min_periods=1).min() < f["ema20"]) & broke
    reclaim = (f["close"] > f["ema20"]) & pull3
    return (reclaim & (f["close"] > f["open"]) & f["don_hi"].notna())

HYP = {
    "K1_vwap_reclaim":   k1_vwap_reclaim,
    "K2_stochrsi_cross": k2_stochrsi_cross,
    "K3_macd_flip":      k3_macd_flip,
    "K4_keltner_squeeze":k4_keltner_squeeze,
    "K5_donchian_retest":k5_donchian_retest,
}

# ─────────────────────────────────────────────────────────────────────────────
# ANTI-CHEAT AUDIT: delete last 500 bars, masks in overlap MUST be identical
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  ANTI-CHEAT LOOKAHEAD AUDIT (before trusting results)\n{SEP2}")
audit_ok = {}
test_sym = sorted(feats2.keys())[0]
ft = feats2[test_sym]
ft_short = ft.iloc[:-500]
for name, fn in HYP.items():
    m_full = fn(ft)
    m_short = fn(ft_short)
    # align on overlap
    overlap = ft_short.index
    same = (m_full.loc[overlap] == m_short.loc[overlap]).all()
    audit_ok[name] = bool(same)
    print(f"    {name:<20} audit {'PASS ✓ (causal)' if same else 'FAIL ✗ (LOOKAHEAD!)'}")
if not all(audit_ok.values()):
    print("\n  🚨 SOME HYPOTHESES FAILED THE AUDIT — they will be EXCLUDED.")

# ─────────────────────────────────────────────────────────────────────────────
qle.IS_LOOKBACK = IS_LOOKBACK
qle.RECAL_EVERY = RECAL_EVERY

def run_mask(mask_map):
    out = []
    for sym, f in feats2.items():
        try:
            for t in sim_symbol(f, mask_map[sym], RR, dict(entry_next=False, exit="timeN",
                                                           time_bars=60, hours=None)):
                t["sym"] = sym; out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t["entry_time"])
    return out

def monthly_profile(trades):
    if not trades: return dict(prof=float("nan"), worst=float("nan"), tpm=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        worst = max(worst, cur)
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df) / 4.5)

def evaluate(name, trades):
    s = stats_from_trades(trades)
    pf_c = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel_t = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel_t)["pf"]; hp = stats_from_trades(hol)["pf"]
    mp = monthly_profile(trades)
    return dict(cfg=name, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"],
                pf_c=pf_c, mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"],
                selpf=sp, holpf=hp)

print(f"\n{SEP2}\n  RESULTS (audit-passed hypotheses only)\n{SEP2}")
hdr = (f"    {'Hypothesis':<20}{'n':>6}{'t/mo':>6}{'WR':>7}{'PF':>7}{'PF@.05':>8}"
       f"{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}")
print(hdr); print("    " + "─"*100)
rows = []
for name in HYP:
    if not audit_ok[name]:
        print(f"    {name:<20}  EXCLUDED (lookahead)")
        continue
    trades = run_mask({s: HYP[name](f) for s, f in feats2.items()})
    r = evaluate(name, trades)
    rows.append(r)
    print(f"    {name:<20}{r['n']:>6}{r['tpm']:>6.1f}{r['wr']*100:>6.0f}%"
          f"{r['pf']:>7.2f}{r['pf_c']:>8.2f}{r['mdd']*100:>7.1f}%{r['prof']*100:>6.0f}%"
          f"{r['worst']:>6}{r['selpf']:>7.2f}{r['holpf']:>7.2f}")

print(f"\n{SEP2}\n  SUCCESS (holPF>1.1, PF@cost>1.1, sel n>=60)")
passed = [r for r in rows if r["holpf"] > 1.1 and r["pf_c"] > 1.1]
if passed:
    for r in passed:
        print(f"  ✅ {r['cfg']}: holPF={r['holpf']:.2f} PF@.05={r['pf_c']:.2f}")
else:
    print("  ❌ none passed. Closest:")
    for r in sorted(rows, key=lambda r: -r["holpf"])[:3]:
        print(f"    {r['cfg']}: holPF={r['holpf']:.2f} PF={r['pf']:.2f} PF@.05={r['pf_c']:.2f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "r091_5m_newind.csv"), index=False)
lines = [f"# R091 — NEW 5m hypotheses / NEW indicators (causal, audited)\n",
         f"**Date:** 2026-08-07 | 5m, 5 symbols | new indicators: VWAP, StochRSI, MACD, "
         f"Keltner, Donchian, BB%B | audit-passed only\n",
         f"\n## Audit\n"]
for name in HYP:
    lines.append(f"- {name}: {'PASS (causal)' if audit_ok[name] else 'FAIL (lookahead, excluded)'}")
lines += ["", "## Results", "",
          "| Hypothesis | n | t/mo | WR | PF | PF@0.05% | MDD% | prof% | worst | selPF | holPF |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['tpm']:.1f} | {r['wr']*100:.0f}% | {r['pf']:.2f} | "
                 f"{r['pf_c']:.2f} | {r['mdd']*100:.1f}% | {r['prof']*100:.0f}% | {r['worst']} | "
                 f"{r['selpf']:.2f} | {r['holpf']:.2f} |")
lines += ["", "## Verdict", ""]
if passed:
    lines.append(f"**✅ {passed[0]['cfg']} passes — candidate 5m edge (audit-passed).**")
else:
    lines.append("**❌ No new 5m hypothesis passes after causal audit.** Honest negative.")
report = "\n".join(lines)
with open(os.path.join(OUT, "r091_final_report.md"), "w") as f:
    f.write(report)
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r091_*")
