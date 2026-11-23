#!/usr/bin/env python3
# R137 — FUNDING-RATE CARRY (prespec: docs/RESEARCH_FRONTIER_R137.md, LOCKED)
import numpy as np, pandas as pd, glob, os

TOGGLE = 0.0010  # 0.10% notional per state toggle (two legs)

def load_coin(f):
    sym = os.path.basename(f).replace("_fund.csv", "")
    d = pd.read_csv(f)
    d["ts"] = pd.to_datetime(d.ts_ms, unit="ms").dt.tz_localize(None)
    return sym, d.set_index("ts").rate.sort_index()

def run_theta(th):
    per_coin = {}
    for f in sorted(glob.glob("quantlab_cache/bnfut/*_fund.csv")):
        sym, rate = load_coin(f)
        r = rate.values
        sig = r[:-1] > th          # signal at epoch t -> hold next
        pos = sig.astype(float)    # position during epoch t+1
        ret = pos * r[1:]          # collect f_{t+1} (short receives positive funding)
        tog = np.abs(np.diff(np.concatenate([[0.0], pos])))
        ret = ret - tog * TOGGLE
        s = pd.Series(ret, index=rate.index[1:])
        yrs = (rate.index[-1] - rate.index[0]).days / 365.25
        per_coin[sym] = dict(series=s, net=s.sum(), apr=s.sum()/yrs*100, yrs=yrs, dep=pos.mean())
    return per_coin

def book_table(pc, label):
    df = pd.DataFrame({k: v["series"] for k, v in pc.items()})
    book = df.mean(axis=1)  # equal slots; undeployed slots are 0 by construction
    years = (book.index[-1] - book.index[0]).days / 365.25
    yr_net = book.groupby(book.index.year).sum() * 100  # % per year (sum of mean-slot epoch returns)
    apr = book.sum() / years * 100
    print(f"\n=== {label} ===")
    print(f"  book net APR (notional): {apr:+.2f}%   | capital-adjusted (÷2): {apr/2:+.2f}%")
    print(f"  yearly net %: " + "  ".join(f"{y}: {v:+.2f}" for y, v in yr_net.items()))
    neg = book[book < 0]
    dep = np.mean([v["dep"] for v in pc.values()])
    print(f"  deployment: {dep*100:.1f}% of epochs  | worst epoch {book.min()*100:+.3f}%  worst 10-epoch run {book.rolling(10).sum().min()*100:+.2f}%")
    coins = sorted(((v["net"], k) for k, v in pc.items()), reverse=True)
    top_share = coins[0][0] / sum(v["net"] for v in pc.values()) * 100
    print(f"  top coin share: {coins[0][1]} {top_share:.0f}%  | coins net+: {sum(1 for n,_ in coins if n>0)}/32")
    return apr, yr_net, top_share

print("="*100)
print("  R137 — FUNDING-RATE CARRY — 32 coins × 3,378 epochs, delta-neutral short-perp+long-spot")
print("="*100)
res = {}
for th, tag in ((0.0001, "θ=0.01%"), (0.0003, "θ=0.03% (PRIMARY)"), (0.0005, "θ=0.05%")):
    pc = run_theta(th)
    apr, yr, top = book_table(pc, tag)
    res[th] = (apr, yr, top, pc)
    if th == 0.0003:
        book = pd.DataFrame({k: v["series"] for k, v in pc.items()}).mean(axis=1)
        book.rename("r").to_csv("quantlab_output/r137_carry_epochs.csv", header=True)

apr, yr, top, pc = res[0.0003]
g1 = all(yr.get(y, 0) >= 0 for y in (2024, 2025, 2026))
g2 = apr >= 5.0
g3 = res[0.0001][0] > 0 and res[0.0005][0] > 0
g4 = top <= 50
print(f"\n  GATES: yearly≥0 {'PASS' if g1 else 'FAIL'} | APR≥5% {'PASS' if g2 else 'FAIL'} | θ-sign-stable "
      f"{'PASS' if g3 else 'FAIL'} | no-50%-coin {'PASS' if g4 else 'FAIL'}")
print("  VERDICT: " + ("CARRY IS A REAL INCOME CANDIDATE" if all((g1,g2,g3,g4)) else
      "CARRY CLOSED — persistence absent/insufficient after costs"))
print("saved: r137_carry_epochs.csv")
