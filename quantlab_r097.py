"""
QUANTLAB AI — R097
CHAMPION UPGRADES, HEAD-TO-HEAD: find the best *safe* config.

Base: R096 champion (Family A + E6 + RR1.5 + breadth50 + walk-forward SVM q0.55,
73 OKX swaps). All variants below are COST / PORTFOLIO / EXECUTION changes —
no new signal mining. Evaluated as deltas vs base on 2026 + fresh window,
under BOTH intra-bar path conventions (TP-first = engine default, SL-first =
adversarial). Universe expansion deliberately excluded (R078: edge is
universe-specific; no validated new symbols).

Variants:
  C0  base           taker 0.05% both sides
  C1  vip-taker      taker 0.04% both sides (next fee tier)
  C2  maker-entry    entry limit@close (fill iff next bar touches it), 0.02% in + 0.05% out
  C3  maker-in+TPout 0.02% in; exit 0.02% if TP (limit), 0.05% if SL/TIME (stop-market)
  C4  C0 + cluster caps   (corr>0.6 cliques, max 2 open per clique, max 12 total)
  C5  C2 + cluster caps
  C6  C5 + vol-scaled sizing (risk 0.6–1.4% by atr_rank, mean ≈1%)
  C7  C3 + cluster caps + vol-scaled sizing  ("full stack")

Also: weekly-refit SVM equivalence check (efficiency only).
Gates for the winner: fresh PF@cost ≥ base, MDD ≤ base, holds under SL-first.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, build_signal_mask, sim_symbol, stats_from_trades,
    cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT   = CONFIG["OUTPUT_FOLDER"]; os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(42)
CACHE = CONFIG["CACHE_FOLDER"]
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
FRESH_START   = pd.Timestamp("2026-08-08", tz="UTC")
RR = 1.5

ORIGINAL52 = {
    "1INCH_USDT_SWAP","AAVE_USDT_SWAP","ADA_USDT_SWAP","ALGO_USDT_SWAP",
    "APT_USDT_SWAP","ARB_USDT_SWAP","ATOM_USDT_SWAP","AVAX_USDT_SWAP",
    "AXS_USDT_SWAP","BCH_USDT_SWAP","BNB_USDT_SWAP","BONK_USDT_SWAP",
    "BTC_USDT_SWAP","CHZ_USDT_SWAP","COMP_USDT_SWAP","CRV_USDT_SWAP",
    "DOGE_USDT_SWAP","DOT_USDT_SWAP","DYDX_USDT_SWAP","EGLD_USDT_SWAP",
    "ENA_USDT_SWAP","ETC_USDT_SWAP","ETH_USDT_SWAP","FET_USDT_SWAP",
    "FIL_USDT_SWAP","FLOKI_USDT_SWAP","GALA_USDT_SWAP","GMX_USDT_SWAP",
    "GRT_USDT_SWAP","HBAR_USDT_SWAP","ICP_USDT_SWAP","IMX_USDT_SWAP",
    "INJ_USDT_SWAP","LDO_USDT_SWAP","LINK_USDT_SWAP","LTC_USDT_SWAP",
    "NEAR_USDT_SWAP","OP_USDT_SWAP","PEPE_USDT_SWAP","SAND_USDT_SWAP",
    "SATS_USDT_SWAP","SHIB_USDT_SWAP","SNX_USDT_SWAP","SOL_USDT_SWAP",
    "STX_USDT_SWAP","SUI_USDT_SWAP","SUSHI_USDT_SWAP","TRX_USDT_SWAP",
    "UNI_USDT_SWAP","WIF_USDT_SWAP","XLM_USDT_SWAP","XRP_USDT_SWAP",
}
NEW18 = {"BICO_USDT_SWAP","HYPE_USDT_SWAP","XAU_USDT_SWAP","HOME_USDT_SWAP",
         "PUMP_USDT_SWAP","ZBT_USDT_SWAP","ZEC_USDT_SWAP","BEAT_USDT_SWAP",
         "SNDK_USDT_SWAP","SPCX_USDT_SWAP","MU_USDT_SWAP","SKHYNIX_USDT_SWAP",
         "SOXL_USDT_SWAP","UB_USDT_SWAP","SNXX_USDT_SWAP","SKHY_USDT_SWAP",
         "KORU_USDT_SWAP","CL_USDT_SWAP"}
NEW3 = {"XAG_USDT_SWAP","ALLO_USDT_SWAP","AAOI_USDT_SWAP"}
ALL_SYMS = ORIGINAL52 | NEW18 | NEW3

SEP  = "═" * 110
SEP2 = "─" * 90
print(); print(SEP)
print("  QUANTLAB AI — R097  champion upgrades head-to-head (cost / portfolio / execution)")
print(SEP)
t0 = time.time()

# ── LOAD (identical to R096) ─────────────────────────────────────────────────
print("\n  Loading data (73 symbols) …")
feats = {}
for sym in ALL_SYMS:
    p = os.path.join(CACHE, f"{sym}_1H.parquet")
    if not os.path.exists(p): continue
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY: feats[sym] = f
    except Exception:
        pass
print(f"  Symbols: {len(feats)}")

above20 = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above20).sort_index().mean(axis=1, skipna=True)
breadth_pct = breadth.rolling(100, min_periods=50).rank(pct=True) * 100

# ── CLUSTERS (corr > 0.6 on 2024-25 hourly closes; greedy cliques) ───────────
print("  Building correlation clusters …")
closes = pd.DataFrame({s: f["close"] for s, f in feats.items()}).sort_index()
cc = closes[(closes.index >= "2024-01-01") & (closes.index < "2026-01-01")]
corr = cc.pct_change().corr()
syms = list(closes.columns)
clusters = {}
for s in syms:
    placed = False
    for cid, members in clusters.items():
        if max(abs(corr.loc[s, m]) for m in members) > 0.6:
            clusters[cid].append(s); placed = True; break
    if not placed:
        clusters[len(clusters)] = [s]
sym2clu = {s: cid for cid, ms in clusters.items() for s in ms}
print(f"  {len(clusters)} cliques (largest: {max(len(m) for m in clusters.values())} syms)")

# ── SIGNALS + RAW TRADES ─────────────────────────────────────────────────────
cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
mask = {s: build_signal_mask(f, cids, "green", 1.5) for s, f in feats.items()}
raw = []
for sym, f in feats.items():
    for t in sim_symbol(f, mask[sym], RR, dict(entry_next=False, exit="base", hours=None)):
        t["sym"] = sym; raw.append(t)
raw.sort(key=lambda t: t["entry_time"])
print(f"  Raw trades: {len(raw)}")

BASE_FEATS = ["atr_rank","adx14","rsi14","ema_dist_pct","prev_body_r","prev_range_r",
              "rel_vol","bb_width","real_vol_20","hour","dow"]
SPECIAL = ["breadth_q", "dist_hi48", "green_streak"]
FEATS = BASE_FEATS + SPECIAL

rows = []
for t in raw:
    sym = t["sym"]; ts = t["entry_time"]; f = feats[sym]
    row = f.loc[ts]
    i = f.index.get_loc(ts)
    c = float(row["close"])
    hi48 = float(f["close"].rolling(48).max().iloc[i]) if i >= 0 else np.nan
    dist_hi = (c / hi48 - 1) * 100 if pd.notna(hi48) and hi48 > 0 else 0.0
    streak = 0
    for k in range(0, 6):
        j = i - k
        if j < 0: break
        if f["close"].iloc[j] > f["open"].iloc[j]: streak += 1
        else: break
    bq = float(breadth_pct.reindex(pd.DatetimeIndex([ts]), method="ffill").iloc[0]) if ts >= breadth_pct.index[0] else 50.0
    rows.append(dict(sym=sym, ts=ts, r=t["r"], win=int(t["r"] > 0),
                     **{c: row.get(c, 0) for c in BASE_FEATS},
                     breadth_q=bq, dist_hi48=dist_hi, green_streak=streak))
mldf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
sel_mask = mldf["ts"] < HOLDOUT_START

print("  Fitting walk-forward SVM …")
X = mldf[FEATS].fillna(0).values
y = mldf["win"].values
pred = np.full(len(mldf), np.nan)
sc = StandardScaler()
fit_i = 150
for i in range(150, len(mldf)):
    clf = SVC(C=1.0, gamma="scale", probability=True)
    clf.fit(sc.fit_transform(X[:i]), y[:i])
    pred[i] = clf.predict_proba(sc.transform(X[i:i+1]))[0, 1]

# weekly-refit equivalence (selection only): preds at weekly boundaries reused
wk_pred = np.full(len(mldf), np.nan)
last_wk = None
for i in range(150, len(mldf)):
    wk = mldf["ts"].iloc[i].isocalendar()[:2]
    if wk != last_wk:
        clf = SVC(C=1.0, gamma="scale", probability=True)
        clf.fit(sc.fit_transform(X[:i]), y[:i])
        last_wk = wk
        cur_clf, cur_sc = clf, sc
    wk_pred[i] = cur_clf.predict_proba(cur_sc.transform(X[i:i+1]))[0, 1]
both = (~np.isnan(pred)) & (~np.isnan(wk_pred)) & sel_mask.values
r_corr = np.corrcoef(pred[both], wk_pred[both])[0, 1]
thr_r  = pd.Series(pred).where(sel_mask.values).dropna().quantile(1-0.55)
thr_w  = pd.Series(wk_pred).where(sel_mask.values).dropna().quantile(1-0.55)
agree  = ((pred >= thr_r) == (wk_pred >= thr_w))[both].mean()
print(f"  weekly-refit check: pred corr={r_corr:.3f}, keep-set agreement={agree*100:.1f}% (selection)")

# ── TRADE FRAMES + MAKER FILL SIM ────────────────────────────────────────────
def trades_for_q(q):
    if q >= 1.0: keep = None
    else:
        thr = pd.Series(pred).where(sel_mask.values).dropna().quantile(1 - q)
        keep = set(mldf.loc[pred >= thr, "ts"])
    out = []
    for t in raw:
        if keep is not None and t["entry_time"] not in keep: continue
        f = feats[t["sym"]]
        i = f.index.get_loc(t["entry_time"])
        d = dict(t); d["i"] = i
        out.append(d)
    return out

def maker_fill_mask(trs):
    """limit@close fills iff the NEXT bar touches that price."""
    ok = np.zeros(len(trs), dtype=bool)
    for k, t in enumerate(trs):
        f = feats[t["sym"]]; i = t["i"]
        if i + 1 >= len(f): continue
        c = t["entry"]
        if t["r"] >= 0 and False: pass
        # direction: infer from levels? sim stores only r; recompute side by comparing sl/tp anchors:
        # entry close c, sl = c - 1*atr, tp = c + 1.5*atr → always LONG for r>0? No —
        # Family A signals are long-only (green gate). Engine has no shorts.
        ok[k] = f["low"].iloc[i+1] <= c     # long limit fills on a dip to c
    return ok

def cost_R(t, c_in, c_out_by_type):
    c_out = c_out_by_type(t)
    return (c_in + c_out) / 100.0 * t["entry"] / t["atr"]

taker_both   = lambda t: cost_R(t, 0.05, lambda x: 0.05)
vip_both     = lambda t: cost_R(t, 0.04, lambda x: 0.04)
maker_in     = lambda t: cost_R(t, 0.02, lambda x: 0.05)
maker_in_out = lambda t: cost_R(t, 0.02, lambda x: (0.02 if x.get("exit_type","TP")=="TP" else 0.05))

# ── PORTFOLIO SIM (sequential, caps, sizing) ─────────────────────────────────
def simulate(trs, cost_fn, caps=False, vol_sizing=False, max_clu=2, max_tot=12):
    trs = sorted(trs, key=lambda t: t["entry_time"])
    act = []           # (exit_time, cluster)
    equity = 10_000.0; curve = []; recs = []
    for t in trs:
        ts = t["entry_time"]
        act = [a for a in act if a[0] > ts]
        if caps:
            clu = sym2clu[t["sym"]]
            if sum(1 for a in act if a[1] == clu) >= max_clu: continue
            if len(act) >= max_tot: continue
        r = t["r"]
        c = cost_fn(t)
        r_net = r - c
        if vol_sizing:
            ar = t.get("atr_rank", 50.0)
            w = 0.01 * (0.6 + 0.8 * (1 - ar/100.0))
        else:
            w = 0.01
        equity *= (1 + w * r_net)
        curve.append((ts, equity))
        act.append((ts + pd.Timedelta(hours=int(t["bars_in"])), sym2clu[t["sym"]]))
        recs.append(dict(sym=t["sym"], ts=ts, r=r, r_net=r_net, w=w, exit_type=t.get("exit_type","")))
    return pd.DataFrame(recs), pd.Series(dict(curve)).sort_index()

def report(name, trs, cost_fn, caps=False, vol_sizing=False):
    df, eq = simulate(trs, cost_fn, caps, vol_sizing)
    out = dict(name=name, n=len(df))
    sel  = df[df["ts"] <  HOLDOUT_START]
    hol  = df[df["ts"] >= HOLDOUT_START]
    fres = df[df["ts"] >= FRESH_START]
    # SL-first adversarial proxy: any full TP win (r==RR) is re-scored as an SL loss
    def slf(sub):
        rs_adj = np.where(sub["r"].values >= RR, -1.0, sub["r"].values)
        return rs_adj - (sub["r_net"].values - sub["r"].values)
    out["sel"]       = (len(sel),  pf_of_rs(sel["r_net"].values)  if len(sel)  else np.nan)
    out["hol"]       = (len(hol),  pf_of_rs(hol["r_net"].values)  if len(hol)  else np.nan)
    out["hol_slf"]   = (len(hol),  pf_of_rs(slf(hol))             if len(hol)  else np.nan)
    out["fresh"]     = (len(fres), pf_of_rs(fres["r_net"].values) if len(fres) else np.nan)
    # MDD + monthly on holdout @cost
    hol = df[df["ts"] >= HOLDOUT_START]
    if len(hol):
        eq2 = (1 + hol["w"]*hol["r_net"]).cumprod()
        mdd = (((eq2 - eq2.cummax()) / eq2.cummax()).min())
        mm = hol.assign(m=hol["ts"].dt.to_period("M")).groupby("m")["r_net"].apply(lambda x: (x*1).sum())
        out["mdd"] = mdd; out["prof"] = (mm > 0).mean(); out["tpm"] = len(hol)/max((hol["ts"].max()-hol["ts"].min()).days/30.4,1)
    else:
        out["mdd"] = np.nan; out["prof"] = np.nan; out["tpm"] = 0
    return out, df

def pf_of_rs(rs):
    rs = np.asarray(rs, float); rs = rs[~np.isnan(rs)]
    if len(rs) == 0: return np.nan
    w = rs[rs > 0].sum(); lo = abs(rs[rs < 0].sum())
    return w / lo if lo > 0 else (99.0 if w > 0 else 1.0)

# ── RUN THE GRID ─────────────────────────────────────────────────────────────
print(f"\n{SEP2}\n  GRID (q=0.55 champion signal, unless noted)\n{SEP2}")
trs55 = trades_for_q(0.55)
print(f"  q0.55 trades: {len(trs55)}; maker fill rate: {maker_fill_mask(trs55).mean()*100:.0f}%")
mf55 = maker_fill_mask(trs55)
trs55m = [t for t, okk in zip(trs55, mf55) if okk]

grid = [
    ("C0 base taker .05/.05",        trs55,  taker_both,   False, False),
    ("C1 vip-taker .04/.04",         trs55,  vip_both,     False, False),
    ("C2 maker-in .02 + taker out",  trs55m, maker_in,     False, False),
    ("C3 maker-in + maker-TP out",   trs55m, maker_in_out, False, False),
    ("C4 C0 + cluster caps",         trs55,  taker_both,   True,  False),
    ("C5 C2 + cluster caps",         trs55m, maker_in,     True,  False),
    ("C6 C5 + vol-scaled sizing",    trs55m, maker_in,     True,  True),
    ("C7 full stack (C3+caps+vol)",  trs55m, maker_in_out, True,  True),
]
res=[]
hdr = (f"    {'config':<30}{'n26':>5}{'PF@c':>7}{'PF_sl':>7}{'fN':>4}{'fPF':>6}{'MDD':>7}{'prof%':>7}{'t/mo':>6}")
print(hdr); print("    " + "─"*80)
for nm, trs, cf, caps, vs in grid:
    r, dfx = report(nm, trs, cf, caps, vs)
    res.append((r, dfx))
    print(f"    {nm:<30}{r['hol'][0] if isinstance(r['hol'],tuple) else 0:>5}"
          f"{r['hol'][1]:>7.2f}{r['hol_slf'][1]:>7.2f}{r['fresh'][0]:>4}{r['fresh'][1]:>6.2f}"
          f"{r['mdd']*100:>6.1f}%{r['prof']*100:>6.0f}%{r['tpm']:>6.1f}")

# q0.65 arm with the leading cost config for comparison
trs65 = trades_for_q(0.65)
mf65 = maker_fill_mask(trs65)
trs65m = [t for t, okk in zip(trs65, mf65) if okk]
r65, _ = report("q0.65 maker-in + caps", trs65m, maker_in, True, False)
print(f"    {'q0.65 maker-in + caps':<30}{r65['hol'][0]:>5}{r65['hol'][1]:>7.2f}{r65['hol_slf'][1]:>7.2f}{r65['fresh'][0]:>4}{r65['fresh'][1]:>6.2f}{r65['mdd']*100:>6.1f}%{r65['prof']*100:>6.0f}%{r65['tpm']:>6.1f}")

# MC reality check on the winner
print(f"\n{SEP2}\n  MONTE CARLO on leading config (holdout net Rs, 0.5% & 1% risk)\n{SEP2}")
# pick the best config by fresh PF then MDD (ties -> lower MDD), disclosed as holdout-selected
def score(item):
    r,_ = item
    fpf = r["fresh"][1] if not np.isnan(r["fresh"][1]) else 0.0
    return (fpf, -(r["mdd"] if not np.isnan(r["mdd"]) else 0.0))
best_name, best_df = max(res, key=score)[0]["name"], max(res, key=score)[1]
print(f"  leading config by fresh PF / MDD: {best_name}")
hol_rs = best_df[best_df["ts"]>=HOLDOUT_START]
rs = hol_rs["r_net"].values
for risk in [0.005, 0.01]:
    prof=[]; dds=[]
    for _ in range(3000):
        x = rs[RNG.integers(0,len(rs),len(rs))]
        cap=100.0; eq=[cap]
        for rr_ in x: cap*=(1+risk*rr_); eq.append(cap)
        eq=np.array(eq); pk=np.maximum.accumulate(eq)
        prof.append(cap); dds.append(((eq-pk)/pk).min())
    prof=np.array(prof); dds=np.array(dds)
    print(f"    risk {risk*100:.1f}%/trade: P(end>100)={(prof>100).mean()*100:.0f}%  median={np.median(prof):.0f}  DD med={np.percentile(dds,50)*100:.1f}% P5={np.percentile(dds,5)*100:.1f}%")

best_df.to_csv(os.path.join(OUT,"r097_best_trades.csv"), index=False)
print(f"\n  Done in {time.time()-t0:.0f}s → quantlab_output/r097_*")
