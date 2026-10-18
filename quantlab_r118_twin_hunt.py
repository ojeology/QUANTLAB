#!/usr/bin/env python3
"""TWIN HUNT (2026-10-16) — engineering the two known premia into livable form. Prespec locked.
H-T1: cross-sectional momentum top-8, vol-targeted, BTC-filter on/off (R113's concentration fix).
H-T2: DON55 BTC long/flip with softened exits (classic / soft-band / time-stop) (R117's DD fix).
Charter gates; selection 2024-01->2026-03; holdout 2026-07->09 untouched; 2020-2023 context."""
import pathlib
import numpy as np
import pandas as pd

BC = pathlib.Path("quantlab_cache/bnfut"); OUT = pathlib.Path("quantlab_output")
T0 = pd.Timestamp("2024-01-01"); SEL_END = pd.Timestamp("2026-03-31"); HOLD0 = pd.Timestamp("2026-07-01")
SPINE1 = pd.Timestamp("2026-09-30"); CTX0 = pd.Timestamp("2020-06-01")

def nd(v):
    v = str(v)
    if "-" in v:
        return pd.Timestamp(v[:10])
    x = int(float(v))
    return pd.Timestamp(pd.to_datetime(x, unit="ms" if x > 1e11 else "s").date())

def load_panel():
    px = {}
    for f in sorted(BC.glob("*_1D.csv")):
        s = f.stem.split("_")[0]
        p = pd.read_csv(f, dtype={"date": str})
        px[s] = pd.Series(p.close.values, index=p.date.map(nd))
    P = pd.DataFrame(px).sort_index()
    P = P.mask(P.pct_change().abs() > 0.5)
    win = P.loc[T0:SPINE1]
    keep = [s for s in P.columns
            if win[s].notna().mean() >= 0.97 and P[s].first_valid_index() <= pd.Timestamp("2023-09-01")]
    P = P[keep].loc[CTX0:SPINE1].ffill()
    R = P.pct_change().fillna(0.0).clip(-0.5, 0.5)
    return P, R

def load_btc_fund():
    fd = pd.read_csv(BC / "BTC_fund.csv")
    fd["ts_ms"] = pd.to_numeric(fd.ts_ms, errors="coerce"); fd["rate"] = pd.to_numeric(fd.rate, errors="coerce")
    fd = fd.dropna()
    fd["d"] = pd.to_datetime(fd.ts_ms, unit="ms").dt.normalize()
    return fd.groupby("d").rate.sum()

P, RETS = load_panel()
D = P.index
FUND = load_btc_fund().reindex(D).fillna(0.0)

def gates(d):
    d = d[d.index >= T0]
    if len(d) < 30:
        return dict(n=0)
    wk = d.resample("W").sum() * 100
    eq = (1 + d).cumprod()
    yr = d.groupby(d.index.year).sum()
    g = dict(posW=(wk > 0).mean() * 100, worstW=wk.min(), maxDD=(eq / eq.cummax() - 1).min() * 100,
             y2024=yr.get(2024, 0) * 100, y2025=yr.get(2025, 0) * 100, y2026=yr.get(2026, 0) * 100,
             apr=d.mean() * 365 * 100, sharpe=d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else 0)
    g["passes"] = sum([g["y2024"] >= 0, g["y2025"] >= 0, g["y2026"] >= 0, g["maxDD"] >= -12,
                       g["worstW"] >= -3, g["posW"] >= 65])
    return g

def show(tag, g, hold_ret, ctx_series=None):
    print(f"{tag:28s}: APR {g['apr']:+6.1f}% Sh {g['sharpe']:+5.2f} maxDD {g['maxDD']:+6.1f}% worstW {g['worstW']:+6.2f}% "
          f"posW {g['posW']:5.1f}% | Y {g['y2024']:+5.1f}/{g['y2025']:+5.1f}/{g['y2026']:+5.1f} | gates {g['passes']}/6 "
          f"| HOLDOUT {hold_ret*100:+5.2f}% {'✓' if hold_ret >= 0 else '✗'}")
    if ctx_series is not None:
        c = ctx_series[ctx_series.index >= CTX0]
        yc = c.groupby(c.index.year).sum() * 100
        print(f"{'':28s}  context: " + " ".join(f"{y}:{v:+.0f}%" for y, v in yc.items()))

# ============ H-T1: DIVERSIFIED, VOL-TARGETED MOMENTUM ============
print("=" * 74)
print("H-T1 — MOMENTUM, DIVERSIFIED + VOL-TARGETED (top-8, 1/8 base, 8%/yr sleeves)")
print("=" * 74)
def ht1(filtered):
    w = pd.Series(0.0, index=P.columns)
    daily = pd.Series(0.0, index=D)
    iso = D.strftime("%G-%V")
    is_last = np.concatenate([(iso[1:] != iso[:-1]), [True]])
    pend = None
    vol = RETS.rolling(30).std() * np.sqrt(365)
    for i in range(1, len(D)):
        if pend is not None:
            daily.iloc[i] -= 0.0005 * float((pend - w).abs().sum())
            w = pend; pend = None
        daily.iloc[i] = float((RETS.iloc[i] * w).sum())
        if is_last[i - 0] and i + 1 < len(D):
            t = D[i]
            if filtered and not (P["BTC"].loc[t] > P["BTC"].loc[:t].rolling(100).mean().loc[t]):
                pend = pd.Series(0.0, index=P.columns); continue
            mom = P.loc[t] / P.iloc[max(0, i - 14)] - 1.0
            if mom.notna().sum() < 8:
                continue
            top = mom.nlargest(8).index
            new = pd.Series(0.0, index=P.columns)
            for c in top:
                v = vol.loc[t, c] if pd.notna(vol.loc[t, c]) and vol.loc[t, c] > 0 else 0.5
                new[c] = (1 / 8) * float(np.clip(0.08 / v, 0.25, 2.0))
            pend = new
    return daily

for filt in (False, True):
    d = ht1(filt)
    g = gates(d)
    h = d[(d.index >= HOLD0)].sum()
    show(f"H-T1 {'BTC-filter' if filt else 'always-in'}", g, h, d)

# ============ H-T2: SOFTENED DONCHIAN EXITS (BTC) ============
print()
print("=" * 74)
print("H-T2 — DON55 BTC LONG/FLIP WITH SOFTENED EXITS (funding + costs in)")
print("=" * 74)
btc = P["BTC"]; ret = RETS["BTC"]; fund = FUND
hh55 = btc.rolling(55).max().shift(1); ll55 = btc.rolling(55).min().shift(1)
sma100 = btc.rolling(100).mean(); hh20 = btc.rolling(20).max().shift(1); ll20 = btc.rolling(20).min().shift(1)
atr30 = (btc.diff().abs().rolling(30).mean())

def ht2(variant):
    """OFF-BY-ONE LAW (R107/R104): the position decided at close i earns bar i+1's return."""
    pos = 0.0; entry = 0.0; entry_i = None
    daily = pd.Series(0.0, index=D)
    cost = 0.0007
    pos_decided = 0.0   # position as of yesterday's close (earns today)
    changed_today = False
    for i in range(1, len(D)):
        t = D[i]; px = btc.iloc[i]
        # P&L for bar i uses the position decided at close i-1
        daily.iloc[i] = pos_decided * (ret.iloc[i] - fund.iloc[i])
        if pos_decided != 0 and changed_today:
            pass
        # now decide tonight's position from today's close
        old = pos
        if variant in ("classic", "timestop"):
            if pos == 0:
                if px > hh55.iloc[i]: pos, entry, entry_i = 1.0, px, i
                elif px < ll55.iloc[i]: pos, entry, entry_i = -1.0, px, i
            else:
                flip = (pos == 1 and px < ll55.iloc[i]) or (pos == -1 and px > hh55.iloc[i])
                stuck = False
                if variant == "timestop" and entry_i is not None and (i - entry_i) >= 30:
                    if abs(px - entry) < 0.5 * atr30.iloc[i]:
                        stuck = True
                if flip or stuck:
                    pos = 0.0
        changed_today = (pos != old)
        if changed_today:
            daily.iloc[i + 1 if i + 1 < len(D) else i] -= cost  # cost lands on the execution bar
        pos_decided = pos
    return daily

best2, best_sh = None, -9
for v in ("classic", "softband", "timestop"):
    if v == "softband":
        continue  # prespec wording degenerate (exit-always) -> VOID, disclosed in report

    d = ht2(v)
    g = gates(d)
    h = d[(d.index >= HOLD0)].sum()
    show(f"H-T2 {v}", g, h, d)
    if g["sharpe"] > best_sh:
        best2, best_sh = v, g["sharpe"]

print(f"\n>>> H-T2 SELECTED (selection Sharpe): {best2}")
d2 = ht2(best2)
g2 = gates(d2)
h2 = d2[(d2.index >= HOLD0)].sum()
print(f">>> H-T1/T2 FINAL: H-T1 gates best {max(gates(ht1(False))['passes'], gates(ht1(True))['passes'])}/6, "
      f"H-T2 {g2['passes']}/6")
d1 = ht1(True); d1.to_csv(OUT / "twin_ht1_filtered.csv")
d2.to_csv(OUT / "twin_ht2.csv")
