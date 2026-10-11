"""
QUANTLAB AI — R102
FUNDING-RATE CARRY: the consistency strategy we never tested.

Mechanism: when perp funding is positive, LONGS pay SHORTS every 8h. Delta-neutral
position (long spot + short perp of equal notional) collects the rate with ~no price
exposure. Income is contractual (a published rate), not statistical.

Data limits (disclosed): OKX public history = ~3.3 months (Jun29→Oct10 2026, 311 epochs
× 24 symbols). Binance/Bybit geo-blocked from this sandbox → the 2024 carry question
cannot be answered from here; we measure the CURRENT regime and its weekly steadiness.

Variants:
  A) ALWAYS-ON  — hold all 24 hedged, eat negative funding too (zero management)
  B) THRESHOLD  — hold a symbol only while its trailing-3d avg rate > +0.005%/8h
                  (≈5.5%/yr); costs 0.25% round-trip per episode amortized over holds
Output: weekly income %, monthly income %, APR, $ tables for $200/$500/$1000,
and a blend vs the champion's 2026 months.
"""
import os, glob
import numpy as np
import pandas as pd

FD = "quantlab_cache/funding"
RNG = np.random.default_rng(42)
SEP="═"*100; SEP2="─"*90

frames=[]
for f in sorted(glob.glob(os.path.join(FD,"*.csv"))):
    sym=os.path.basename(f)[:-4]
    d=pd.read_csv(f, parse_dates=["fundingTime"])
    d["sym"]=sym
    frames.append(d)
allf=pd.concat(frames)
allf["week"]=allf["fundingTime"].dt.to_period("W")
allf["month"]=allf["fundingTime"].dt.to_period("M")
print(SEP); print("  R102 — FUNDING CARRY (delta-neutral income, OKX 24 symbols)"); print(SEP)
print(f"  epochs: {len(allf)}  span: {allf['fundingTime'].min():%Y-%m-%d} → {allf['fundingTime'].max():%Y-%m-%d}")

# ── per-symbol stats ─────────────────────────────────────────────────────────
g = allf.groupby("sym")["rate"]
st = pd.DataFrame({"mean8h":g.mean(), "pos%":g.apply(lambda x:(x>0).mean()*100), "sum":g.sum()})
st["APR"] = st["mean8h"]*3*365*100
print(f"\n{SEP2}\n  per-symbol funding (always-on income, %/yr on notional)\n{SEP2}")
for s,r in st.sort_values("APR",ascending=False).iterrows():
    print(f"  {s:<6} avg {r['mean8h']*100:+.4f}%/8h  pos {r['pos%']:5.1f}%  APR {r['APR']:+6.1f}%")

# ── A) always-on portfolio income ────────────────────────────────────────────
port = allf.groupby(["fundingTime"])["rate"].mean()   # equal-weight across syms per epoch
w = port.groupby(port.index.to_period("W")).sum()*100
m = port.groupby(port.index.to_period("M")).sum()*100
print(f"\n{SEP2}\n  A) ALWAYS-ON portfolio (equal-weight 24, income % of notional)\n{SEP2}")
print("  weekly:"); 
for k,v in w.items(): print(f"    {str(k)[:15]}: {v:+.3f}%")
print(f"  positive weeks: {(w>0).sum()}/{len(w)}  min week {w.min():+.3f}%  max {w.max():+.3f}%")
print("  monthly:")
for k,v in m.items(): print(f"    {k}: {v:+.3f}%")
apr = port.mean()*3*365*100
print(f"  → APR ≈ {apr:+.1f}% on notional")

# ── B) threshold variant ─────────────────────────────────────────────────────
print(f"\n{SEP2}\n  B) THRESHOLD variant (hold only while trailing-3d avg > +0.005%/8h; 0.25% round-trip)\n{SEP2}")
inc_rows=[]
for sym, d in allf.groupby("sym"):
    d=d.sort_values("fundingTime").reset_index(drop=True)
    d["ma3"]=d["rate"].rolling(3*9, min_periods=9).mean()   # 3 days = 9 epochs
    on=False; entry_cost=0.0025
    for i,r in d.iterrows():
        want = (not np.isnan(r["ma3"])) and r["ma3"]>0.00005
        if want and not on:
            on=True; pend_cost=entry_cost
        if on:
            inc_rows.append(dict(sym=sym, ts=r["fundingTime"], rate=r["rate"], new=(not want==False and False)))
            if pend_cost:  # amortize entry cost over expected hold: charge on first epoch
                inc_rows[-1]["rate"] -= 0.0   # deferred: we instead charge half now, half at exit (simple: charge on both ends)
                pend_cost=False
        if on and (not want):
            on=False
            if len(inc_rows): inc_rows[-1]["rate"] -= 0.0
        # (exit cost folded into last epoch below via simple bookkeeping)
inc=pd.DataFrame(inc_rows)
inc["week"]=inc["ts"].dt.to_period("W"); inc["month"]=inc["ts"].dt.to_period("M")
# simple, transparent cost: subtract 0.25% once per (episode) — approximate by subtracting 0.25%/min hold length from each episode
# easier defensible: subtract 0.25% from the FIRST epoch of each on-run (most conservative for short holds)
on_runs=[]
for sym, d in allf.groupby("sym"):
    d=d.sort_values("fundingTime").reset_index(drop=True)
    d["ma3"]=d["rate"].rolling(27, min_periods=9).mean()
    on=False
    for i,r in d.iterrows():
        want = (not np.isnan(r["ma3"])) and r["ma3"]>0.00005
        if want and not on: on=True; start_i=i; started_ts=r["fundingTime"]
        if on:
            on_runs.append(dict(sym=sym, ts=r["fundingTime"], rate=r["rate"], first=(i==start_i)))
        if on and (not want) and i>=start_i: on=False
od=pd.DataFrame(on_runs)
od["rate_adj"]=od["rate"]-np.where(od["first"],0.0025,0.0)
od["week"]=od["ts"].dt.to_period("W"); od["month"]=od["ts"].dt.to_period("M")
pw = od.groupby(["week"])["rate_adj"].mean()
pm = od.groupby(["month"])["rate_adj"].mean()
print("  weekly (portfolio avg):")
for k,v in pw.items(): print(f"    {str(k)[:15]}: {v:+.3f}%")
print(f"  positive weeks: {(pw>0).sum()}/{len(pw)}")
print("  monthly:")
for k,v in pm.items(): print(f"    {k}: {v:+.3f}%")
print(f"  → APR ≈ {od.groupby('ts')['rate_adj'].mean().mean()*3*365*100:+.1f}%")

# ── $ translation ────────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  $ INCOME (notional = 1× capital; spot 2/3 of capital + perp margin 1/3 @3x isolated)\n{SEP2}")
monthly_apr_frac = od.groupby("ts")["rate_adj"].mean().mean()*3*30.4
for cap in [200,500,1000,2000]:
    print(f"  ${cap:>5} account → ≈ ${cap*monthly_apr_frac:5.2f}/month (current funding regime)")

# ── blend with champion (2026 overlap months) ────────────────────────────────
print(f"\n{SEP2}\n  BLEND: 50% carry + 50% champion (months in common, % of that half's capital)\n{SEP2}")
ch = pd.read_csv("quantlab_output/r100_champion_trades.csv", parse_dates=["ts"])
ch["month"]=ch["ts"].dt.to_period("M")
chm = ch.groupby("month")["r_net"].sum()*1.0   # % at 1%/trade ≈ R
carm = pm
common = chm.index.intersection(carm.index)
print(f"    {'month':<10}{'carry%':>8}{'champ%':>8}{'50/50':>8}")
for k in common:
    print(f"    {str(k):<10}{carm[k]:>8.2f}{chm[k]:>8.1f}{(carm[k]+chm[k])/2:>8.2f}")
print("\n  (carry leg quoted on its own capital; champ leg at 1%/trade)")
