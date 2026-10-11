"""
QUANTLAB AI — R101
SMALL-ACCOUNT PROTOCOL: same validated champion trades, restructured so a
small account ($200 / $500 / $1,000) can actually run them.

NOTHING here re-fits or re-selects: we filter the ALREADY-VALIDATED champion
trade log (R100, 277 trades) by symbol liquidity, and stress the sizing ladder.
Steps:
  1) OKX instrument specs: contract value (ctVal/minSz) → min notional per symbol
     → feasibility table vs account size (which symbols a small account CAN trade).
  2) Subsets: TOP12 most liquid vs all-73 — does the edge hold on the big caps alone?
  3) Risk ladder: 1% / 2% / 3% per trade (flat + throttle) on 2026, in DOLLARS
     for $200/$500/$1,000, with Monte-Carlo survival stats (P(DD>30%), P(DD>50%)).
  4) Verdict: the small-account config.
"""
import os, json, time, subprocess
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
UA = "Mozilla/5.0"
def get(url, tries=3):
    for a in range(tries):
        p = subprocess.run(["curl","-s","-4","--max-time","20","-A",UA,url], capture_output=True, timeout=30)
        try:
            d = json.loads(p.stdout.decode())
            if d.get("code") == "0": return d["data"]
            time.sleep(1+a)
        except Exception: time.sleep(1+a)
    return []

SEP = "═"*110; SEP2 = "─"*90
print(SEP); print("  QUANTLAB AI — R101  SMALL-ACCOUNT PROTOCOL"); print(SEP)

t = pd.read_csv("quantlab_output/r100_champion_trades.csv", parse_dates=["ts"])
t26 = t[t["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")].copy()
t["r_net"] = t["r_net"].astype(float)

# ── 1) instrument specs / min notional ───────────────────────────────────────
print("\n  [1] OKX contract specs (ctVal, minSz) → min notional")
inst = get("https://www.okx.com/api/v5/public/instruments?instType=SWAP")
specs = {}
for i in inst:
    iid = i.get("instId","")
    if not iid.endswith("-USDT-SWAP"): continue
    try:
        ct = float(i["ctVal"]); ctccy = i.get("ctValCcy","")
        minsz = float(i.get("minSz","1"))
        specs[iid.replace("-","_")] = dict(ct=ct, ccy=ctccy, minsz=minsz)
    except Exception: pass

# current prices for the champion's symbols
tick = get("https://www.okx.com/api/v5/market/tickers?instType=SWAP")
price = {}
for x in tick:
    try: price[x["instId"].replace("-","_")] = float(x["last"])
    except Exception: pass

syms = sorted(t["sym"].unique())
rows=[]
for s in syms:
    sp = specs.get(s); px = price.get(s)
    if not sp or not px: continue
    # contract value in USDT
    if sp["ccy"] == "USDT":
        cval = sp["ct"]
    else:
        cval = sp["ct"] * px
    min_notional = cval * sp["minsz"]
    rows.append(dict(sym=s, price=px, ct_val=cval, min_notional=min_notional))
sp = pd.DataFrame(rows)
liq = t.groupby("sym")["ts"].count().rename("trades").reset_index()
sp = sp.merge(liq, on="sym", how="left")
sp = sp.sort_values("min_notional")
print(f"    {len(sp)} champion symbols with specs")
print("    CHEAPEST 12 by min notional (small-account friendly):")
for _, r in sp.head(12).iterrows():
    print(f"      {r['sym']:<22} min ~${r['min_notional']:7.1f}   ({int(r['trades'])} champ trades)")
print("    MOST EXPENSIVE 5:")
for _, r in sp.tail(5).iterrows():
    print(f"      {r['sym']:<22} min ~${r['min_notional']:7.1f}")

# ── 2) TOP12 subset performance ──────────────────────────────────────────────
print(f"\n{SEP2}\n  [2] Does the edge hold on a small-account subset? (TOP12 cheapest-min + liquid)\n{SEP2}")
# pick TOP12: lowest min_notional among symbols with >=8 champion trades (practical + proven activity)
cand = sp[sp["trades"]>=8].sort_values("min_notional")
TOP12 = set(cand.head(12)["sym"])
sub = t26[t26["sym"].isin(TOP12)]
def pf(x):
    x=np.asarray(x,float)
    return x[x>0].sum()/abs(x[x<0].sum()) if (x<0).any() else 99.0
print(f"    TOP12: {sorted(TOP12)}")
for tag, d in [("all-73 (2026)", t26), ("TOP12 (2026)", sub)]:
    print(f"    {tag:<16} n={len(d):>3}  PF@c={pf(d['r_net'].values):.2f}  sumR={d['r_net'].sum():+.1f}R  WR={(d['r_net']>0).mean()*100:.0f}%")
for tag, d in [("all-73 (17mo)", t), ("TOP12 (17mo)", t[t['sym'].isin(TOP12)])]:
    print(f"    {tag:<16} n={len(d):>3}  PF@c={pf(d['r_net'].values):.2f}  sumR={d['r_net'].sum():+.1f}R  WR={(d['r_net']>0).mean()*100:.0f}%")

# ── 3) risk ladder in dollars + MC survival ──────────────────────────────────
print(f"\n{SEP2}\n  [3] RISK LADDER — 2026 YTD, TOP12 subset, $200/$500/$1,000 accounts\n{SEP2}")
def run_equity(d, start, w, throttle):
    cap=start; peak=start; mdd=0; tier_used=0
    for _,x in d.iterrows():
        dd = 1-cap/peak
        ww = w if not throttle else (0.25*w if dd>0.10 else (0.5*w if dd>0.05 else w))
        if ww<w: tier_used+=1
        cap *= (1+ww*x["r_net"]); peak=max(peak,cap); mdd=max(mdd,1-cap/peak)
    return cap, mdd, tier_used

print(f"    {'sizing':<22}{'account':>9}{'final':>10}{'P&L':>10}{'maxDD':>8}")
for w in [0.01, 0.02, 0.03]:
    tag = f"{w*100:.0f}%/trade" + (" + throttle" if w>=0.02 else "")
    for start in [200, 500, 1000]:
        cap, mdd, _ = run_equity(sub, start, w, throttle=(w>=0.02))
        print(f"    {tag:<22}{start:>9}${cap:>9,.0f}{cap-start:>+10,.0f}{-mdd*100:>7.1f}%")

print(f"\n    Monte Carlo survival (TOP12 2026 Rs, 3000 resamples):")
for w in [0.02, 0.03]:
    for throttle in [False, True]:
        rs = sub["r_net"].values
        dds=[]; bust=0; finals=[]
        for _ in range(3000):
            x = rs[RNG.integers(0,len(rs),len(rs))]
            cap=1.0; peak=1.0; m=0
            for r in x:
                dd=1-cap/peak
                ww = w if not throttle else (0.25*w if dd>0.10 else (0.5*w if dd>0.05 else w))
                cap*=(1+ww*r); peak=max(peak,cap); m=max(m,1-cap/peak)
            dds.append(m); finals.append(cap)
            if m>0.5: bust+=1
        dds=np.array(dds); finals=np.array(finals)
        print(f"    {w*100:.0f}%{' +thr' if throttle else '     '}: median={np.median(finals):.2f}x  P(DD>30%)={(dds>0.3).mean()*100:.0f}%  P(DD>50%)={bust/3000*100:.0f}%  worst5%DD={np.percentile(dds,95)*100:.0f}%")

# ── 4) min-order reality check ───────────────────────────────────────────────
print(f"\n{SEP2}\n  [4] FEASIBILITY: min notional vs account (TOP12, avg stop 0.98%)\n{SEP2}")
mx = sp[sp["sym"].isin(TOP12)]["min_notional"].max()
print(f"    largest min-notional in TOP12: ~${mx:.0f}")
for start in [200, 500, 1000]:
    risk = 0.02*start
    need_notional = risk/0.0098
    print(f"    ${start:>5} @ 2% risk (${risk:.0f}/trade): position ≈ ${need_notional:.0f} → "
          f"{'✅ fits all TOP12' if need_notional>=mx else '⚠ margin carries the gap (isolated 3-5x, stop still caps loss at '+f'${risk:.0f})'}")

t26.to_csv("quantlab_output/r101_trades_2026.csv", index=False)
print(f"\n  Done → quantlab_output/r101_*")
