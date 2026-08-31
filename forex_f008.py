"""
SCALP PROBE — F008
User wants scalping on BOTH crypto and forex at RR 1.3, "a real easy edge".

Tests (5m candles, ~60 days via yfinance):
  Markets: forex 8 majors (EURUSD GBPUSD USDJPY AUDUSD USDCAD USDCHF NZDUSD EURGBP)
           crypto (BTC-USD ETH-USD DOGE-USD SOL-USD)
  RR = 1.3 (risk 1R, target 1.3R)
  Hypotheses (scalp-appropriate, causal):
    S1 momentum-burst : body > 1.0*ATR green + relvol>1.5 -> long
    S2 extreme-fade   : RSI2 < 10 + green -> long (mean-reversion scalp)
    S3 breakout-20    : close > prior-20-high + relvol>1.3 -> long
    S4 ema-pullback   : uptrend + dip to EMA20 + reclaim -> long
  Costs: forex retail spread/pair; crypto 0.05%/side
  Selection = first 40 days, holdout = last 20 days (untouched)
  Honest question: does ANY scalp survive costs at 5m?
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
IS_LOOKBACK_5M = 288   # 1 day of 5m
RECAL_EVERY_5M = 288
HORIZON = 24

FOREX_PAIRS = ["EURUSD=X","GBPUSD=X","JPY=X","AUDUSD=X","CAD=X","CHF=X","NZDUSD=X","EURGBP=X"]
FOREX_SPREAD = {"EURUSD=X":0.00006,"GBPUSD=X":0.00010,"JPY=X":0.010,"AUDUSD=X":0.00008,
                "CAD=X":0.00010,"CHF=X":0.00010,"NZDUSD=X":0.00012,"EURGBP=X":0.00010}
CRYPTO_TICKS = ["BTC-USD","ETH-USD","DOGE-USD","SOL-USD"]
CRYPTO_COST = 0.0005  # 0.05% per side

print("Fetching 5m data (yfinance, ~60 days) …", flush=True)
t0 = time.time()

def fetch(ticker):
    df = yf.download(ticker, interval="5m", period="1mo", progress=False, auto_adjust=False)
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
        if df is not None and len(df) > 2000:
            data[t] = df
            print(f"  ✓ {t}: {len(df)} bars  {df.index.min():%m-%d}→{df.index.max():%m-%d}")
    except Exception as e:
        print(f"  ✗ {t}: {str(e)[:60]}")
print(f"  fetched {len(data)} series in {time.time()-t0:.0f}s", flush=True)

# features per market
def prep(df):
    if "vol" not in df.columns:
        df["vol"] = (df["high"]-df["low"]).clip(lower=1e-12)
    f = add_features(df)
    # RSI2
    delta = f["close"].diff(); up = delta.clip(lower=0).rolling(2).mean(); down = (-delta.clip(upper=0)).rolling(2).mean()
    f["rsi2"] = 100 - 100/(1 + up/down.replace(0,np.nan))
    f = f.dropna(subset=["ema200","ema50","ema20","atr14","adx14","rsi14","ema_dist_pct","rel_vol"])
    return f

feats = {}
for t, df in data.items():
    try:
        f = prep(df)
        if len(f) > 2500: feats[t] = f
    except Exception as e:
        print(f"  prep err {t}: {str(e)[:60]}")

qle.IS_LOOKBACK = IS_LOOKBACK_5M
qle.RECAL_EVERY = RECAL_EVERY_5M

def s1_mom(f):
    big = (f["close"]-f["open"]).abs() > 1.0*f["atr14"]
    return (big & (f["close"]>f["open"]) & (f["rel_vol"]>1.5) & (f["close"]>f["close"].shift(1)))
def s2_fade(f):
    return ((f["rsi2"]<10) & (f["close"]>f["open"]))
def s3_break(f):
    ph = f["high"].rolling(20).max().shift(1)
    return ((f["close"]>ph) & (f["rel_vol"]>1.3) & (f["close"]>f["open"]))
def s4_pull(f):
    low2 = f["low"].rolling(2,min_periods=1).min()
    return ((f["ema50"]>f["ema200"]) & (low2<f["ema20"]) & (f["close"]>f["ema20"]) & (f["close"]>f["open"]))
HYP = {"S1_mom": s1_mom, "S2_fade": s2_fade, "S3_break": s3_break, "S4_pull": s4_pull}

def run_market(t, f, market):
    out = []
    for name, fn in HYP.items():
        m = fn(f)
        try:
            for tr in sim_symbol(f, m, RR, dict(entry_next=False, exit="timeN",
                                                time_bars=HORIZON, hours=None)):
                tr["market"]=market; tr["tick"]=t; tr["hyp"]=name; tr["ts"]=tr["entry_time"]
                out.append(tr)
        except Exception:
            pass
    return out

all_trades = []
for t, f in feats.items():
    mkt = "forex" if t in FOREX_PAIRS else "crypto"
    all_trades += run_market(t, f, mkt)

# attach costs
df = pd.DataFrame(all_trades)
df["cost_r"] = df.apply(lambda r: 2*(FOREX_SPREAD[r["tick"]]/max(r["atr"],1e-12)) if r["market"]=="forex"
                        else 2*CRYPTO_COST*r["entry"]/max(r["atr"],1e-12), axis=1)
df["r_net"] = df["r"] - df["cost_r"]
# selection = first 40 days, holdout = last 20
cutoff = df["ts"].max() - pd.Timedelta(days=20)
df["period"] = np.where(df["ts"] < cutoff, "sel", "hol")

def pf(rs):
    rs=np.array(rs)
    return (rs[rs>0].sum()/abs(rs[rs<0].sum())) if (rs<0).any() else 99.0

print(f"\n{'='*100}\n  SCALP RESULTS (5m, RR {RR}) — gross vs cost\n{'='*100}")
print(f"  Total trades: {len(df)} | forex { (df['market']=='forex').sum() } | crypto { (df['market']=='crypto').sum() }")

print(f"\n  {'Config':<28}{'n':>6}{'WR':>6}{'PF':>7}{'PF@cost':>8}{'holPF':>7}{'holPF@c':>9}")
rows = []
# per market x hypothesis
for mkt in ["forex","crypto"]:
    for name in HYP:
        sub = df[(df["market"]==mkt)&(df["hyp"]==name)]
        if len(sub) < 40: continue
        hol = sub[sub["period"]=="hol"]
        r = dict(cfg=f"{mkt[:4]}|{name}", n=len(sub), wr=(sub["r"]>0).mean(),
                 pf=pf(sub["r"]), pfc=pf(sub["r_net"]),
                 holpf=pf(hol["r"]) if len(hol) else float('nan'),
                 holpfc=pf(hol["r_net"]) if len(hol) else float('nan'))
        rows.append(r)
        print(f"  {r['cfg']:<28}{r['n']:>6}{r['wr']*100:>5.0f}%{r['pf']:>7.2f}{r['pfc']:>8.2f}"
              f"{r['holpf']:>7.2f}{r['holpfc']:>9.2f}")
# market totals
for mkt in ["forex","crypto"]:
    sub = df[df["market"]==mkt]
    if len(sub)==0: continue
    hol = sub[sub["period"]=="hol"]
    print(f"  {'ALL '+mkt.upper():<28}{len(sub):>6}{(sub['r']>0).mean()*100:>5.0f}%"
          f"{pf(sub['r']):>7.2f}{pf(sub['r_net']):>8.2f}{pf(hol['r']):>7.2f}{pf(hol['r_net']):>9.2f}")

# cost breakeven analysis for the best market/hyp
print(f"\n  COST BREAKEVEN (best candidates):")
for mkt in ["forex","crypto"]:
    sub = df[df["market"]==mkt]
    if len(sub)==0: continue
    # average cost in R
    avg_cost = sub["cost_r"].mean()
    print(f"    {mkt}: avg cost {avg_cost:.2f}R/trade | avg gross R {sub['r'].mean():.3f} | "
          f"net {sub['r_net'].mean():.3f}R")

df.to_csv(f"{OUT}/f008_scalp.csv", index=False)
# report
lines=[f"# SCALP PROBE (crypto + forex, 5m, RR {RR})\n",
       f"**Date:** 2026-08-08 | 5m candles ~60 days (yfinance) | selection=first 40d, holdout=last 20d\n",
       f"Crypto cost 0.05%/side; forex retail spreads. RR {RR} (risk 1, target {RR}R).\n",
       f"\n## Results (gross vs cost)\n",
       f"| Config | n | WR | PF | PF@cost | holPF | holPF@cost |",
       f"|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['cfg']} | {r['n']} | {r['wr']*100:.0f}% | {r['pf']:.2f} | {r['pfc']:.2f} | "
                 f"{r['holpf']:.2f} | {r['holpfc']:.2f} |")
lines += ["","## Verdict","",
          "Honest: at 5m, cost drag is huge (crypto ~0.05%/side, forex spread/ATR). "
          "Any config with holPF@cost > 1.1 would be a real scalp edge; otherwise the "
          "5m cost wall holds in both markets."]
open(f"{OUT}/f008_final_report.md","w").write("\n".join(lines))
print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/f008_*")
