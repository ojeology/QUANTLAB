"""
FOREX HUNT — F009
RSI2 deep-oversold fade at 1H (the scalp logic, on a cost-friendly timeframe).
Crypto + forex, RR 1.3.

F008 found: crypto RSI2<10 fade has a REAL gross edge at 5m (holPF 1.18) but died to
~0.94R/trade cost. At 1H the cost per trade drops ~10x (bigger ATR), so the same logic
may survive.

Signals (1H):
  S1 rsi2-fade : RSI2 < 10 (deep oversold) + green candle -> long
  S2 rsi2-fade v2 : RSI2 < 15 + green + close > open + mild relvol
  S3 rsi2-fade + lower-BB : RSI2 < 10 + close below BB lower (stronger oversold)
  S4 (crypto only) rsi2-fade + volceil : skip when atr_rank > 70 (F008 winner refined)
RR 1.3, 1H, time stop 24h.
Markets: 8 forex majors (retail spreads), crypto 1H (0.05%/side) - fetch fresh.
Selection = first ~10 months, holdout = last ~4 months (untouched) if data allows;
for crypto 1H (~2yr avail) use 2024-25 sel / 2026 hol.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import yfinance as yf
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features, sim_symbol, IS_LOOKBACK, RECAL_EVERY
import scripts.ql_engine as qle

OUT = CONFIG["OUTPUT_FOLDER"]
os.makedirs(OUT, exist_ok=True)
RR = 1.3
IS_LOOKBACK_1H = 500
RECAL_EVERY_1H = 168

FOREX_PAIRS = ["EURUSD=X","GBPUSD=X","JPY=X","AUDUSD=X","CAD=X","CHF=X","NZDUSD=X","EURGBP=X"]
FOREX_SPREAD = {"EURUSD=X":0.00006,"GBPUSD=X":0.00010,"JPY=X":0.010,"AUDUSD=X":0.00008,
                "CAD=X":0.00010,"CHF=X":0.00010,"NZDUSD=X":0.00012,"EURGBP=X":0.00010}
CRYPTO_TICKS = ["BTC-USD","ETH-USD","DOGE-USD","SOL-USD"]
CRYPTO_COST = 0.0005

print("Fetching 1H data (yfinance, ~2y) …", flush=True)
t0 = time.time()
def fetch(ticker):
    df = yf.download(ticker, interval="1h", period="2y", progress=False, auto_adjust=False)
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[["Open","High","Low","Close"]].copy()
    df.columns = ["open","high","low","close"]
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("UTC")
    df = df[~df.index.duplicated(keep="last")].dropna()
    return df

data = {}
for t in FOREX_PAIRS + CRYPTO_TICKS:
    try:
        df = fetch(t)
        if df is not None and len(df) > 3000:
            data[t] = df
            print(f"  ✓ {t}: {len(df)} bars {df.index.min():%Y-%m}→{df.index.max():%Y-%m}")
    except Exception as e:
        print(f"  ✗ {t}: {str(e)[:60]}")
print(f"  fetched {len(data)} in {time.time()-t0:.0f}s", flush=True)

def prep(df):
    if "vol" not in df.columns:
        df["vol"] = (df["high"]-df["low"]).clip(lower=1e-12)
    f = add_features(df)
    delta = f["close"].diff(); up = delta.clip(lower=0).rolling(2).mean(); down = (-delta.clip(upper=0)).rolling(2).mean()
    f["rsi2"] = 100 - 100/(1 + up/down.replace(0,np.nan))
    f = f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14","ema_dist_pct","rel_vol","bb_lower"])
    return f

feats = {}
for t, df in data.items():
    try:
        f = prep(df)
        if len(f) > 3500: feats[t] = f
    except Exception as e:
        print(f"  prep err {t}: {str(e)[:60]}")

qle.IS_LOOKBACK = IS_LOOKBACK_1H
qle.RECAL_EVERY = RECAL_EVERY_1H

def s1(f):
    return ((f["rsi2"]<10) & (f["close"]>f["open"]))
def s2(f):
    return ((f["rsi2"]<15) & (f["close"]>f["open"]) & (f["rel_vol"]>1.0))
def s3(f):
    return ((f["rsi2"]<10) & (f["close"]<f["bb_lower"]) & (f["close"]>f["open"]))
def s4(f):
    return ((f["rsi2"]<10) & (f["close"]>f["open"]) & (f["atr_rank"]<=70))
HYP = {"S1_rsi2fade": s1, "S2_rsi2v2": s2, "S3_bb": s3, "S4_volceil": s4}

def run_market(t, f, market):
    out = []
    for name, fn in HYP.items():
        m = fn(f)
        try:
            for tr in sim_symbol(f, m, RR, dict(entry_next=False, exit="timeN",
                                                time_bars=24, hours=None)):
                tr["market"]=market; tr["tick"]=t; tr["hyp"]=name; tr["ts"]=tr["entry_time"]
                out.append(tr)
        except Exception:
            pass
    return out

all_trades = []
for t, f in feats.items():
    mkt = "forex" if t in FOREX_PAIRS else "crypto"
    all_trades += run_market(t, f, mkt)

df = pd.DataFrame(all_trades)
df["cost_r"] = df.apply(lambda r: 2*(FOREX_SPREAD[r["tick"]]/max(r["atr"],1e-12)) if r["market"]=="forex"
                        else 2*CRYPTO_COST*r["entry"]/max(r["atr"],1e-12), axis=1)
df["r_net"] = df["r"] - df["cost_r"]
# selection = up to 2025-12-31, holdout = 2026
cutoff = pd.Timestamp("2025-12-31", tz="UTC")
df["period"] = np.where(df["ts"] < cutoff, "sel", "hol")

def pf(rs):
    rs=np.array(rs)
    return (rs[rs>0].sum()/abs(rs[rs<0].sum())) if (rs<0).any() else 99.0

print(f"\n{'='*100}\n  RSI2-FADE at 1H, RR {RR} — gross vs cost\n{'='*100}")
print(f"  Total: {len(df)} | forex {(df['market']=='forex').sum()} | crypto {(df['market']=='crypto').sum()}")
print(f"\n  {'Config':<26}{'n':>6}{'WR':>6}{'PF':>7}{'PF@cost':>8}{'holPF':>7}{'holPF@c':>9}")
rows = []
for mkt in ["forex","crypto"]:
    for name in HYP:
        sub = df[(df["market"]==mkt)&(df["hyp"]==name)]
        if len(sub) < 60: continue
        hol = sub[sub["period"]=="hol"]
        r = dict(cfg=f"{mkt[:4]}|{name}", n=len(sub), wr=(sub["r"]>0).mean(),
                 pf=pf(sub["r"]), pfc=pf(sub["r_net"]),
                 holpf=pf(hol["r"]) if len(hol) else float('nan'),
                 holpfc=pf(hol["r_net"]) if len(hol) else float('nan'))
        rows.append(r)
        print(f"  {r['cfg']:<26}{r['n']:>6}{r['wr']*100:>5.0f}%{r['pf']:>7.2f}{r['pfc']:>8.2f}"
              f"{r['holpf']:>7.2f}{r['holpfc']:>9.2f}")
for mkt in ["forex","crypto"]:
    sub = df[df["market"]==mkt]
    if len(sub)==0: continue
    hol = sub[sub["period"]=="hol"]
    print(f"  {'ALL '+mkt.upper():<26}{len(sub):>6}{(sub['r']>0).mean()*100:>5.0f}%"
          f"{pf(sub['r']):>7.2f}{pf(sub['r_net']):>8.2f}{pf(hol['r']):>7.2f}{pf(hol['r_net']):>9.2f}")

print(f"\n  COST per trade (R):")
for mkt in ["forex","crypto"]:
    sub = df[df["market"]==mkt]
    if len(sub): print(f"    {mkt}: avg {sub['cost_r'].mean():.2f}R  (F008 5m was ~0.95R — 1H should be ~10x lower)")

df.to_csv(f"{OUT}/f009_rsi2_1h.csv", index=False)
lines=[f"# F009 — RSI2 deep-oversold fade at 1H (crypto + forex, RR {RR})\n",
       f"**Date:** 2026-08-08 | 1H, ~2y, selection ≤2025, holdout 2026 untouched\n",
       f"F008 found the gross edge at 5m; 1H cuts cost/trade ~10x.\n",
       f"\n## Results\n", f"| Config | n | WR | PF | PF@cost | holPF | holPF@cost |",
       f"|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['wr']*100:.0f}% | {r['pf']:.2f} | {r['pfc']:.2f} | "
                 f"{r['holpf']:.2f} | {r['holpfc']:.2f} |")
lines += ["", "## Verdict", "",
          "If any config has holPF@cost > 1.1: the RSI2-fade scalp edge survives at 1H."]
open(f"{OUT}/f009_final_report.md","w").write("\n".join(lines))
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/f009_*")
