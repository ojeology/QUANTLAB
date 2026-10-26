#!/usr/bin/env python3
# R130 — BIG-MOVE CENSUS & PRECURSOR ATLAS (prespec: docs/RESEARCH_FRONTIER_R130.md, LOCKED)
import os, sys, glob, json, subprocess, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantlab_ai import CONFIG
from scripts.ql_engine import add_features
from scipy.stats import binomtest

CACHE = CONFIG["CACHE_FOLDER"]
SPLIT = pd.Timestamp("2025-10-01", tz="UTC")
W = 720  # 30d hourly window

# ── market series (refetched; /tmp is per-session) ───────────────────────────
subprocess.run(["curl","-s","--max-time","40","https://api.alternative.me/fng/?limit=0","-o","/tmp/fng.json"])
subprocess.run(["curl","-s","-L","--max-time","40","https://stablecoins.llama.fi/stablecoincharts/all","-o","/tmp/stbl.json"])
fng = json.load(open("/tmp/fng.json"))["data"]
FG = pd.Series([float(x["value"]) for x in fng],
               index=pd.to_datetime([int(x["timestamp"]) for x in fng], unit="s").normalize()).sort_index()
st = json.load(open("/tmp/stbl.json"))
SV = pd.Series([float(d["totalCirculating"]["peggedUSD"]) for d in st],
               index=pd.to_datetime([int(d["date"]) for d in st], unit="s").normalize()).sort_index()
LIQ30 = (SV / SV.shift(30) - 1)
print(f"market series: F&G {len(FG)}d, stablecoins {len(SV)}d")

# ── per-coin features + events ───────────────────────────────────────────────
coins = {}
for f in sorted(glob.glob(os.path.join(CACHE, "*_USDT_SWAP_1H.parquet"))):
    sym = os.path.basename(f).replace("_USDT_SWAP_1H.parquet", "")
    if sym.startswith(("XAU","XAG","CL","SOXL","MU","SPCX","AAOI","UB","SNDK","SKHY","BEAT","ALLO","ZBT","SNXX","PUMP","HOME")):
        continue
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
    df = df[~df.index.duplicated()]
    if len(df) < W + 200: continue
    ft = add_features(df)
    ret = ft["close"].pct_change()
    sig = ret.rolling(W).std()
    ft = ft.assign(ret=ret, hsig=sig,
                   bbw_pct=ft["bb_width"].rolling(W).rank(pct=True),
                   rv_pct=ft["real_vol_20"].rolling(W).rank(pct=True),
                   drift24=ret.rolling(24).sum())
    coins[sym] = ft
print(f"coins with warmup: {len(coins)}")

rows = []
for sym, ft in coins.items():
    ev = ft[(ft.ret.abs() >= 4*ft.hsig) & (ft.ret.abs() >= 0.015)].dropna(subset=["ret"])
    last = None
    for ts, r in ev.ret.items():
        if last is not None and (ts - last).total_seconds() < 24*3600: continue
        last = ts
        rows.append((sym, ts, r))
EV = pd.DataFrame(rows, columns=["sym","ts","ret"]).sort_values("ts").reset_index(drop=True)
EV["dir"] = np.where(EV.ret > 0, "UP", "DOWN")
same_hour = EV.groupby(EV.ts.dt.floor("h")).sym.transform("size")
EV["cluster"] = same_hour >= 5
print(f"CENSUS: {len(EV)} events ({(EV.dir=='UP').sum()} UP / {(EV.dir=='DOWN').sum()} DOWN) | "
      f"clustered {EV.cluster.mean()*100:.0f}% | span {EV.ts.min().date()} → {EV.ts.max().date()}")
hh = EV.ts.dt.hour.value_counts().sort_index()
print("  hour anatomy (UTC): " + " ".join(f"{h:>2}h:{n}" for h, n in hh.items()))

# ── hourly grid of antecedent states ─────────────────────────────────────────
breadth = pd.DataFrame({s: (f["close"] > f["ema20"]).astype(float) for s, f in coins.items()}).mean(axis=1)
btc1h = coins["BTC"]["ret"]
grid = []
for sym, ft in coins.items():
    g = pd.DataFrame(index=ft.index)
    g["bbw"] = ft["bbw_pct"].values
    g["rv"] = ft["rv_pct"].values
    g["edist"] = ft["ema_dist_pct"].values
    g["drift"] = ft["drift24"].values
    g["breadth"] = breadth.reindex(ft.index).values
    g["btc1h"] = btc1h.reindex(ft.index).values
    d = ft.index.tz_convert("UTC")
    dna = d.tz_localize(None).normalize()
    g["fng"] = FG.reindex(dna, method="ffill").values
    g["liq"] = LIQ30.reindex(dna, method="ffill").values
    g["hour"] = d.hour; g["dow"] = d.dayofweek
    g["sym"] = sym
    grid.append(g)
G = pd.concat(grid)
liq_med = LIQ30.expanding(min_periods=90).median()
G["liq_med"] = G.index.tz_localize(None).normalize().map(liq_med)
G["liq_hi"] = (G["liq"] > G["liq_med"]).astype(float)
# forward event indicator per coin-hour (up/down within next 24h)
ev_up = {(s, t): 1 for s, t in EV.loc[EV.dir == "UP", ["sym","ts"]].itertuples(index=False)}
ev_dn = {(s, t): 1 for s, t in EV.loc[EV.dir == "DOWN", ["sym","ts"]].itertuples(index=False)}
def fwd_hit(sym, idx, table):
    keys = [(sym, t) for t in idx]
    hits = np.zeros(len(idx), dtype=bool)
    for j, k in enumerate(keys):
        if k in table:
            hits[j] = True; continue
        for h in range(1, 24):
            if (sym, idx[j] + pd.Timedelta(hours=h)) in table:
                hits[j] = True; break
    return hits
GRID_CACHE = "quantlab_cache/r130_grid.parquet"
if os.path.exists(GRID_CACHE):
    G = pd.read_parquet(GRID_CACHE)
    print(f"grid loaded from cache: {len(G):,} rows")
else:
    print("computing forward-event indicators (this is the slow part)...")
    fwd_up, fwd_dn = [], []
    for sym, ft in coins.items():
        idx = G[G.sym == sym].index
        fwd_up.append(pd.Series(fwd_hit(sym, idx, ev_up), index=idx))
        fwd_dn.append(pd.Series(fwd_hit(sym, idx, ev_dn), index=idx))
    G["fup"] = pd.concat(fwd_up).reindex(G.index)
    G["fdn"] = pd.concat(fwd_dn).reindex(G.index)
    G = G.dropna(subset=["bbw","rv","edist","drift","breadth","fng","liq"])
    G["era"] = np.where(G.index < SPLIT, "DISC", "CONF")
    G.to_parquet(GRID_CACHE)
print(f"evaluation grid: {len(G):,} coin-hours  (DISC {int((G.era=='DISC').sum()):,} / CONF {int((G.era=='CONF').sum()):,})")

def lift_table(Gsub, submask, up=True):
    tgt = "fup" if up else "fdn"
    base = Gsub[tgt].mean()
    sub = Gsub[submask]
    k, n = int(sub[tgt].sum()), len(sub)
    if n < 500: return base, (np.nan, n, np.nan)
    lift = (k/n) / base if base > 0 else np.nan
    p = binomtest(k, n, base, alternative="two-sided").pvalue
    return base, (lift, n, p)

BINS = [
    ("bbw<=25%",      G.bbw <= 0.25), ("bbw 25-75%", (G.bbw > 0.25) & (G.bbw < 0.75)), ("bbw>=75%", G.bbw >= 0.75),
    ("rv<=25%",       G.rv <= 0.25),  ("rv>=75%", G.rv >= 0.75),
    ("coil (POSCTRL)", (G.bbw <= 0.25) & (G.rv <= 0.33) & (G.edist <= 0)),
    ("edist>=p75", G.edist >= G.edist.quantile(0.75)),
    ("drift24<-3%",   G.drift < -0.03), ("drift24>+6%", G.drift > 0.06),
    ("breadth<30%",   G.breadth < 0.30), ("breadth>70%", G.breadth > 0.70),
    ("btc1h>+0.5%",   G.btc1h > 0.005), ("btc1h<-0.5%", G.btc1h < -0.005),
    ("FNG<=40",       G.fng <= 40), ("FNG>75", G.fng > 75),
    ("LIQ expanding", G.liq_hi == 1), ("LIQ contracting", G.liq_hi == 0),
    ("hours 0-6 UTC", G.hour < 6), ("hours 12-18 UTC", (G.hour >= 12) & (G.hour < 18)),
    ("weekend",       G.dow >= 5),
]
DISC_MASK = (G.era == "DISC").values
Dd, Cc = G[DISC_MASK], G[~DISC_MASK]
print("\n" + "="*108)
print("  PRECURSOR ATLAS — forward-24h BIG-MOVE lift vs base rate (DISC → CONF reproduction)")
print("="*108)
print(f"  {'state':<18}{'UP lift DISC':>13}{'CONF':>7}{'p(disc)':>10}{'n(disc)':>9}   {'DOWN lift DISC':>14}{'CONF':>7}")
candidates = []
for name, mask in BINS:
    mv = mask.values
    _, oU = lift_table(Dd, mv[DISC_MASK], up=True)
    _, oUc = lift_table(Cc, mv[~DISC_MASK], up=True)
    _, oD = lift_table(Dd, mv[DISC_MASK], up=False)
    _, oDc = lift_table(Cc, mv[~DISC_MASK], up=False)
    lU, nU, pU = oU; lUc = oUc[0]
    lD = oD[0]; lDc = oDc[0]
    print(f"  {name:<18}{lU:>13.2f}{lUc:>7.2f}{pU:>10.4f}{nU:>9,}   {lD:>14.2f}{lDc:>7.2f}")
    if name != "coil (POSCTRL)" and lU >= 1.5 and lUc >= 1.3 and nU >= 50 and pU < 0.001:
        candidates.append(name)

# positive control
ctl_mask = ((G.bbw <= 0.25) & (G.rv <= 0.33) & (G.edist <= 0)).values
bctl = Dd[ctl_mask[DISC_MASK]]
k, n = int(bctl.fup.sum()), len(bctl)
p_ctl = binomtest(k, n, Dd.fup.mean()).pvalue
lift_ctl = (k/n)/Dd.fup.mean()
print(f"\n  POSITIVE CONTROL (coil state → fwd UP lift, discovery): lift={lift_ctl:.2f}  n={n:,}  p={p_ctl:.2e}  "
      f"-> {'PIPELINE VALID' if (lift_ctl > 1.0 and p_ctl < 0.01) else 'PIPELINE INVALID'}")
print(f"\n  CANDIDATES (disc>=1.5, conf>=1.3, n>=50, p<0.001): {candidates if candidates else 'NONE'}")

# follow-through anatomy
up_ev = EV[EV.dir == "UP"].copy()
f24, f72 = [], []
for _, e in up_ev.iterrows():
    ft = coins[e.sym]
    i = ft.index.get_loc(e.ts)
    if i+24 < len(ft): f24.append(ft.close.iloc[i+24]/ft.close.iloc[i]-1)
    if i+72 < len(ft): f72.append(ft.close.iloc[i+72]/ft.close.iloc[i]-1)
un24 = coins["BTC"].ret.rolling(24).sum().mean()*1  # approx unconditional 24h drift
print(f"\n  FOLLOW-THROUGH after big UP events: fwd24h mean {np.mean(f24)*100:+.2f}% (n={len(f24)}) | "
      f"fwd72h {np.mean(f72)*100:+.2f}% (n={len(f72)}) | panel 24h drift ~{un24*100:+.2f}%")
cl = up_ev[up_ev.cluster]
print(f"  clustered UP events: {len(cl)} of {len(up_ev)}")
EV.to_csv("quantlab_output/r130_events.csv", index=False)
G.sample(min(200_000, len(G)), random_state=130).to_csv("quantlab_output/r130_grid_sample.csv")
print("saved: r130_events.csv (+grid sample)")
