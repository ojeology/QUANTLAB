#!/usr/bin/env python3
# R127 — THE GOLD MAP (prespec: docs/RESEARCH_FRONTIER_R127.md, LOCKED)
import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, build_signal_mask, sim_symbol, IS_LOOKBACK, RECAL_EVERY

CACHE = CONFIG["CACHE_FOLDER"]
SPLIT = pd.Timestamp("2016-01-01")
COST = 0.0010          # 0.10% per unit position change (flip = 0.20% RT)

g = pd.read_csv(f"{CACHE}/GC_F_1D.csv", parse_dates=["date"]).set_index("date")["close"]
r = g.pct_change().dropna()

def ann(x):
    if len(x) < 20: return dict(n=0)
    yrs = len(x) / 252
    tot = (1 + x).prod() - 1
    cagr = (1 + x).prod() ** (1 / yrs) - 1
    vol = x.std() * np.sqrt(252)
    sh = x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    eq = (1 + x).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return dict(n=len(x), cagr=cagr * 100, vol=vol * 100, sharpe=sh, mdd=dd * 100, total=tot * 100)

def bootp(x, n=5000, seed=127):
    x = np.asarray(x, float)
    months = len(x) // 21
    if months <= 2: return float("nan")
    B = np.array([x[i*21:(i+1)*21] for i in range(months)])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, months, size=(n, months))
    return float((B[idx].sum(axis=(1, 2)) > 0).mean())

print("=" * 100)
print("  R127 — THE GOLD MAP (GC=F 2000→2026 · OKX XAU 1H 2024→2026)")
print("=" * 100)

# ── D2 anatomy (report-only) ─────────────────────────────────────────────────
print("\n── D2 ANATOMY (mapping) ──")
sel, conf = r[r.index < SPLIT], r[r.index >= SPLIT]
for tag, x in [("selection 2000-08→2015", sel), ("confirm    2016→2026", conf)]:
    a = ann(x)
    print(f"  {tag}: CAGR {a['cagr']:+.1f}%  vol {a['vol']:.1f}%  Sharpe {a['sharpe']:+.2f}  maxDD {a['mdd']:+.1f}%")
eq = (1 + r).cumprod()
print("  deepest bears (peak-to-trough):")
for end in eq.resample("YE").last().index:
    pass
win = 512
dds = []
for i in range(win, len(eq), 21):
    w = eq.iloc[i-win:i]; pk = w.cummax(); dds.append((w.index[-1], (w/pk-1).min()*100))
for d, v in sorted(dds, key=lambda t: t[1])[:3]:
    print(f"    {d.date()}: {v:+.1f}%")
y26 = r[r.index >= "2026-01-01"]
print(f"  2026 YTD: {ann(y26)['total']:+.1f}%  | $300 translation: gold's long-run drift {ann(conf)['cagr']/100:.1%}/yr = ${300*ann(conf)['cagr']/100/12:.2f}/mo")
try:
    btc = pd.read_parquet(f"{CACHE}/BTC_USDT_SWAP_1H.parquet")
    btc.index = pd.to_datetime(btc.index, utc=True)
    b = btc["close"].resample("1D").last().dropna(); b.index = b.index.tz_localize(None).normalize()
    j = pd.concat([g.rename("au"), b.rename("btc")], axis=1).dropna().pct_change().dropna()
    print(f"  corr(gold, BTC) daily {j.index.min().date()}→{j.index.max().date()}: {j.au.corr(j.btc):+.2f}")
except Exception as e:
    print(f"  corr skipped: {type(e).__name__}")

# ── D1 registered momentum cells ─────────────────────────────────────────────
print("\n── D1 TIME-SERIES MOMENTUM (REGISTERED cells; cost 0.10%/unit change) ──")
print(f"  {'cell':<22}{'sel Sharpe':>10}{'sel bootP':>10}{'conf Sharpe':>12}{'conf CAGR':>10}{'verdict':>10}")
promising = []
for N in (20, 60, 120, 250):
    for shape in ("long/flat", "long/short"):
        sig = np.sign(g.pct_change(N)).shift(1)
        pos = sig if shape == "long/short" else sig.clip(lower=0)
        pos = pos.fillna(0)
        strat = pos * r - pos.diff().abs().fillna(0) * COST
        s_sel, s_conf = strat[strat.index < SPLIT].dropna(), strat[strat.index >= SPLIT].dropna()
        a_s, a_c = ann(s_sel), ann(s_conf)
        bp = bootp(s_sel.values)
        ok = a_s.get("sharpe", 0) >= 0.5 and (bp == bp and bp >= 0.95) and a_c.get("sharpe", -9) > 0
        tag = "PROMISING" if ok else "dead"
        if ok: promising.append(f"N{N} {shape}")
        print(f"  N{N:<3} {shape:<16}{a_s.get('sharpe', float('nan')):>10.2f}{bp:>10.3f}"
              f"{a_c.get('sharpe', float('nan')):>12.2f}{a_c.get('cagr', float('nan')):>9.1f}%{tag:>10}")

# ── D3 event anatomy (report-only) ───────────────────────────────────────────
print("\n── D3 EVENT ANATOMY (mapping) ──")
sig = r.rolling(252).std()
big = r[(r.index < SPLIT) & (r.abs() > 2 * sig)]
for h in (5, 20):
    fwd = (g.shift(-h) / g - 1).reindex(big.index)
    cond = fwd.mean(); unc = (g.shift(-h) / g - 1).reindex(r[r.index < SPLIT].index).mean()
    print(f"  after >2σ day: fwd {h}d {cond*100:+.2f}%  vs unconditional {unc*100:+.2f}%  (n={len(big)})")
big_all = r[r.abs() > 2 * sig].dropna()
print("  biggest single days ever: " + ", ".join(f"{d.date()} {v*100:+.1f}%" for d, v in big_all.abs().sort_values(ascending=False).head(0).items()) or "")
top5 = r.abs().sort_values(ascending=False).head(5)
print("  top-5 |daily| moves: " + "  ".join(f"{d.date()} {r[d]*100:+.1f}%" for d in top5.index))

# ── D4 compression-pop on gold 1H (REGISTERED) ───────────────────────────────
print("\n── D4 COMPRESSION-POP ON GOLD 1H (REGISTERED; engine sim, 0.07%/side, TP-first disclosed) ──")
try:
    df = pd.read_parquet(f"{CACHE}/XAU_USDT_SWAP_1H.parquet")
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    f = add_features(df)
    f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20","bb_width",
                     "prev_range_r","prev_body_r"], inplace=True)
    mask = build_signal_mask(f, ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"], "green", 1.5)
    tr = list(sim_symbol(f, mask, 1.5, dict(entry_next=False, exit="base", hours=None)))
    rs = np.array([t["r"] - 2.0*(0.07/100.0)*t["entry"]/t["atr"] for t in tr])
    w = rs[rs > 0].sum(); l = abs(rs[rs < 0].sum()); pf = w/l if l > 0 else 999
    rng = np.random.default_rng(127)
    bp = float(np.mean([rng.choice(rs, len(rs)).sum() > 0 for _ in range(5000)])) if len(rs) else float("nan")
    print(f"  raw gold pops: n={len(rs)}  PF@c={pf:.2f}  WR={(rs>0).mean()*100 if len(rs) else 0:.0f}%  bootP={bp:.3f}")
    ok4 = len(rs) >= 30 and pf >= 1.3 and bp >= 0.95
    print(f"  gates (n>=30, PF>=1.3, boot>=0.95): {'PROMISING-CANDIDATE' if ok4 else 'DEAD'}")
except Exception as e:
    print(f"  D4 skipped: {type(e).__name__}: {e}")

# ── D5 session anatomy (report-only) ─────────────────────────────────────────
print("\n── D5 SESSION ANATOMY (gold 1H 2024→2026, mapping) ──")
hr = df["close"].pct_change().groupby(df.index.hour)
vol_by = (hr.apply(lambda s: np.sqrt((s**2).mean())) * 100).round(3)
ret_by = (hr.mean() * 100).round(3)
share = (vol_by / vol_by.sum() * 100)
top = share.sort_values(ascending=False)
london_ny = share[[h for h in range(7, 17)]].sum()
print(f"  vol share top-4 UTC hours: " + "  ".join(f"{h}h:{v:.1f}%" for h, v in top.head(4).items()))
print(f"  London+NY block (07–16 UTC) holds {london_ny:.0f}% of hourly vol")
best = ret_by.sort_values(ascending=False)
print(f"  best/worst avg hour: {best.index[0]}h {best.iloc[0]:+.3f}% / {best.index[-1]}h {best.iloc[-1]:+.3f}%")

print("\n" + "=" * 100)
print(f"  GOLD MAP VERDICT: D1 promising cells: {promising if promising else 'NONE'} | "
      f"D4: see above | D2/D3/D5: mapping only")
print("=" * 100)
