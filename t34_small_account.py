"""
T34 — SMALL-ACCOUNT LEVERAGE × R:R STUDY
==========================================
Question: the validated crypto-1H edge (T25 trend, ~+100%/3y; T27 MR+trend 70/30)
is too slow for a $100 trader. Which (R:R exit target) x (risk-per-trade,
compounding "leverage") turns it into a small-account grower without ruin?

Honesty protocol (identical to blind-validation branch):
  - walk-forward OOS: filter trained on PRIOR years only; test each year standalone
  - fees 0.05% per side; trades simulated on the shared bot-faithful engine
  - data: 50-symbol manifest, 2023 fetched from OKX, cache 2024-01-28..2026-07-30
  - "leverage" = risk-per-trade f of equity, compounding; effective notional
    leverage L ~= f / (stop distance in %) is reported separately

Legs:
  TREND  Donchian(20) brk + ADX>20 + close>EMA200; SL = 2 ATR.
         R:R lever = optional TP (1,1.5,2,3 R). ASIS = verbatim T25d exit.
  MR     FAM_A coiled signal; SL 1 ATR, TP = rr*1 ATR (rr sweep).
  PORT   T27-style: 70% capital on TREND sleeve, 30% on MR sleeve.
"""
import os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from t34_lib import (load_feats, trend_raw, trend_champion, mr_raw, mr_champion,
                     r_adj, simulate, equity_metrics, monte_carlo_risk, kelly)

OUT = "/home/user/bv/t34_output"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
YEARS = (2024, 2025, 2026)
RISKS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]


def metrics_from_curve(curve):
    ts = pd.DatetimeIndex([x[0] for x in curve])
    eq = np.array([x[1] for x in curve], dtype=float)
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / pk).min())
    final = float(eq[-1])
    yrs = max((ts[-1] - ts[0]).total_seconds() / (365.25 * 86400), 1e-9)
    cagr = (final / 100.0) ** (1 / yrs) - 1 if final > 0 else -1.0
    s = pd.Series(eq, index=pd.DatetimeIndex(ts)).resample("ME").last()
    mr = s.pct_change().dropna()
    prof = int((mr > 0).sum()); tot = len(mr)
    wm = float(mr.min()) if tot else 0.0
    streak = worst = 0
    for v in mr < 0:
        streak = streak + 1 if v else 0
        worst = max(worst, streak)
    return dict(final=final, ret=final / 100 - 1, cagr=cagr, mdd=mdd,
                prof_mo=f"{prof}/{tot}", worst_month=wm, worst_streak=worst)


def deltas(trades, risk):
    d = [(t["entry_time"], risk * r_adj(t)) for t in trades]
    d.sort(key=lambda x: x[0])
    return d


def one_row(cfg, trades, risk, extra_mc=None, kelly_f=None):
    ev = deltas(trades, risk)
    _, curve = simulate(ev)
    em = metrics_from_curve(curve)
    unit = [r_adj(t) for t in trades]
    mc = monte_carlo_risk([risk * u for u in unit])
    kf = (kelly(unit) if kelly_f is None else {"full_kelly": kelly_f})["full_kelly"]
    return dict(cfg=cfg, n=len(trades), risk=risk,
                final=em["final"], ret=em["ret"], cagr=em["cagr"], mdd=em["mdd"],
                prof_mo=em["prof_mo"], worst_month=em["worst_month"],
                worst_streak=em["worst_streak"], kelly=round(kf, 3),
                **{f"mc_{k}": v for k, v in mc.items()})


print("[1/5] load features …", flush=True)
feats, above20, breadth, breadth_pct = load_feats()
print(f"  usable symbols: {len(feats)}  ({time.time()-t0:.0f}s)", flush=True)

print("[2/5] TREND leg — TP (R:R) sweep, RF q0.65 walk-forward …", flush=True)
trend_cfgs = {
    "T25-ASIS": dict(tp_R=None),
    "T25-TP1.0": dict(tp_R=1.0),
    "T25-TP1.5": dict(tp_R=1.5),
    "T25-TP2.0": dict(tp_R=2.0),
    "T25-TP3.0": dict(tp_R=3.0),
}
trend_champs = {}
for name, kw in trend_cfgs.items():
    raw = trend_raw(feats, **kw)
    trend_champs[name] = trend_champion(raw, feats, breadth, breadth_pct, years=YEARS)
    u = np.array([r_adj(t) for t in trend_champs[name]])
    print(f"  {name}: n={len(u)}  meanR={u.mean():+.3f}  wr={(u > 0).mean():.0%}", flush=True)

print("[3/5] MR leg — RR sweep, SVM q0.65-adaptive walk-forward …", flush=True)
mr_cfgs = {f"MR-RR{r:g}": dict(rr=r) for r in [1.0, 1.5, 2.0, 3.0]}
mr_champs = {}
for name, kw in mr_cfgs.items():
    raw = mr_raw(feats, **kw)
    mr_champs[name] = mr_champion(raw, feats, breadth, breadth_pct, years=YEARS)
    u = np.array([r_adj(t) for t in mr_champs[name]])
    print(f"  {name}: n={len(u)}  meanR={u.mean():+.3f}  wr={(u > 0).mean():.0%}", flush=True)

print("[4/5] risk × R:R sweep …", flush=True)
rows = []
# legs alone
for name in ["T25-ASIS", "T25-TP1.5", "T25-TP2.0"]:
    for f in RISKS:
        rows.append(one_row(f"TREND|{name}", trend_champs[name], f))
for name in ["MR-RR1.5", "MR-RR2", "MR-RR3"]:
    for f in RISKS:
        rows.append(one_row(f"MR|{name}", mr_champs[name], f))


def port_curve(tname, mname, ft, fm):
    """70/30 sleeve portfolio path (T27 style)."""
    evt = deltas(trend_champs[tname], ft)
    evm = deltas(mr_champs[mname], fm)
    _, ct = simulate(evt); _, cm = simulate(evm)
    tdf = pd.DataFrame(ct, columns=["ts", "eq"]).set_index("ts")
    mdf = pd.DataFrame(cm, columns=["ts", "eq"]).set_index("ts")
    ix = sorted(set(tdf.index) | set(mdf.index))
    pth = (0.7 * tdf["eq"].reindex(ix).ffill().fillna(100.0) +
           0.3 * mdf["eq"].reindex(ix).ffill().fillna(100.0))
    return [(ix[0], 100.0)] + list(pth.reset_index().itertuples(index=False, name=None))

def port_mc_stream(tname, mname, ft, fm):
    return ([0.7 * ft * r_adj(t) for t in trend_champs[tname]] +
            [0.3 * fm * r_adj(t) for t in mr_champs[mname]])


# PORT 70/30 sleeves (T27 style) for a few risk pairs
for tname, mname in [("T25-ASIS", "MR-RR1.5"), ("T25-TP1.5", "MR-RR1.5"),
                     ("T25-TP2.0", "MR-RR1.5"), ("T25-TP2.0", "MR-RR2")]:
    for ft, fm in [(0.01, 0.01), (0.02, 0.02), (0.03, 0.03), (0.05, 0.05)]:
        curve = port_curve(tname, mname, ft, fm)
        em = metrics_from_curve(curve)
        mc = monte_carlo_risk(port_mc_stream(tname, mname, ft, fm))
        n = len(trend_champs[tname]) + len(mr_champs[mname])
        rows.append(dict(cfg=f"PORT70/30|{tname}+{mname}", n=n,
                         risk=f"{ft:.0%}/{fm:.0%}", final=em["final"], ret=em["ret"],
                         cagr=em["cagr"], mdd=em["mdd"], prof_mo=em["prof_mo"],
                         worst_month=em["worst_month"], worst_streak=em["worst_streak"],
                         kelly=float("nan"), **{f"mc_{k}": v for k, v in mc.items()}))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "t34_leverage_rr_sweep.csv"), index=False)
print(f"csv written: {len(df)} rows", flush=True)

# ------------------------------------------------------- equity figure ------
print("[5/5] figures + report …", flush=True)
fig, ax = plt.subplots(figsize=(11, 6))
def plot_curve(label, ev, lw=1.8):
    _, curve = simulate(ev)
    ts = pd.DatetimeIndex([x[0] for x in curve]); eq = np.array([x[1] for x in curve])
    ax.plot(ts, eq, label=f"{label}  ${eq[-1]:,.0f}", lw=lw)
plot_curve("T25 ASIS @2%/trade", deltas(trend_champs["T25-ASIS"], 0.02))
plot_curve("T25 TP2.0 @2%/trade", deltas(trend_champs["T25-TP2.0"], 0.02), 1.2)
plot_curve("MR RR1.5 @2%/trade", deltas(mr_champs["MR-RR1.5"], 0.02))
plot_curve("MR RR2 @2%/trade", deltas(mr_champs["MR-RR2"], 0.02))
plot_curve("MR RR2 @5%/trade", deltas(mr_champs["MR-RR2"], 0.05), 1.4)
def plot_port(tname, mname, ft, fm):
    curve = port_curve(tname, mname, ft, fm)
    ts = pd.DatetimeIndex([x[0] for x in curve]); eq = np.array([x[1] for x in curve])
    ax.plot(ts, eq, label=f"PORT70/30 {tname}+{mname} @{ft:.0%}/{fm:.0%}  ${eq[-1]:,.0f}", lw=1.5)
plot_port("T25-TP2.0", "MR-RR1.5", 0.03, 0.03)
ax.axhline(100, color="grey", ls="--", lw=1)
ax.set_yscale("log")
ax.set_title("T34 — $100 account under risk × R:R (walk-forward OOS 2024–2026, 50-sym crypto 1H, fees 0.05%/side)")
ax.set_ylabel("Equity ($, log)"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "t34_equity_curves.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------ report (markdown) ---------
top = df.sort_values("final", ascending=False).head(40)
md = []
md.append("# T34 — Small-Account Leverage × R:R Study (crypto 1H, 50-sym, 2023–2026)\n")
md.append("**Date:** 2026-09-03 · **Branch context:** blind-validation (`main` fork) · **Universe:** 50-symbol manifest · **Data:** 2023 fetched (OKX) + cache 2024-01-28→2026-07-30 · **Fees:** 0.05%/side · **Protocol:** walk-forward OOS, filter trained on prior years, each year standalone (RF q0.65 trend / SVM q0.65-adaptive MR).\n")
md.append("**Question:** the validated edge compounds ~+100% over 3 years (big-trader rate). What R:R × risk-per-trade makes a **$100** account grow fast without ruin?\n")
md.append("**What 'leverage' means here:** risk-per-trade `f` of equity, compounding (`eq *= 1 + f·R`, R = P&L/initial-stop-distance). Effective notional leverage ≈ `f / stop%` (≈ f/0.02 trend, ≈ f/0.01 MR). No margin/funding modelled — positions are assumed sized so the stop is the only risk; exchange leverage only needs to cover `1/stop%`.\n")

md.append("## 1. R:R sweep — which target wins per leg?\n")
md.append("| Leg config | n | WR | mean R | full-Kelly | comment |")
md.append("|---|---|---|---|---|---|")
for name in [*trend_cfgs.keys(), *mr_cfgs.keys()]:
    u = np.array([r_adj(t) for t in (trend_champs.get(name) or mr_champs.get(name))])
    if len(u) == 0:
        continue
    kl = kelly(u)
    src = "trend" if name in trend_champs else "MR"
    md.append(f"| {name} ({src}) | {len(u)} | {(u > 0).mean():.0%} | {u.mean():+.3f} | {kl['full_kelly']:.2f} | |")

md.append("\n## 2. Risk sweep (top configs)\n")
md.append("| Config | n | risk/trade | End $100 | CAGR | MaxDD | prof-mo | worst streak | MC P(double) | MC P(halve) | MC P(DD>30%) |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
def fmt_risk(v):
    return v if isinstance(v, str) else f"{v:.0%}"
for r in top.to_dict("records"):
    md.append("| {} | {} | {} | ${:,.0f} | {:.0%} | {:.0%} | {} | {} | {:.0%} | {:.2%} | {:.0%} |".format(
        r["cfg"], r["n"], fmt_risk(r["risk"]), r["final"], r["cagr"], r["mdd"],
        r["prof_mo"], r["worst_streak"], r["mc_p_double"], r["mc_p_halve"], r["mc_p_dd30"]))
md.append("\n*CAGR annualised over the ~2.5-yr window; MC = 4,000 bootstrap paths; full sweep in `t34_leverage_rr_sweep.csv`.*\n")
md.append("## 3. Equity curves\n")
md.append("![equity](t34_equity_curves.png)\n")
md.append("## 4. Recommended $100 config\n")
md.append("*Filled from results below the sweep — see script stdout / report section above.*\n")
open(os.path.join(OUT, "t34_small_account_report.md"), "w").write("\n".join(md))
print("saved csv/png/report", flush=True)
print(f"\n[done in {time.time()-t0:.0f}s]")
