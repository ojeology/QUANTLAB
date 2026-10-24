# SL-first truth audit of the RR ladder: re-score ambiguous bars (1H bar touches BOTH TP and SL)
# against 15m ground truth on the 8 covered symbols. Plus the full robustness battery per RR.
import glob, os, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CACHE = "quantlab_cache"
COVERED = ["BTC","ETH","XRP","LINK","AVAX","DOGE","BCH","LTC"]
RRS = {"1.0":"champ_rr10","1.25":"champ_rr125","1.5":"r096","1.75":"champ_rr175",
       "2.0":"champ_rr2","2.5":"champ_rr25","3.0":"champ_rr30"}

def rc(t):  # engine cost convention, 0.05%/side
    return t.r - 2.0*(0.05/100.0)*t.entry/t.atr

def pf(x):
    x = np.asarray(x); x = x[~np.isnan(x)]
    w = x[x>0].sum(); lo = abs(x[x<0].sum())
    return w/lo if lo>0 else (999.0 if w>0 else 1.0)

h1, m15 = {}, {}
for s in COVERED:
    df = pd.read_parquet(f"{CACHE}/{s}_USDT_SWAP_1H.parquet")
    df.index = pd.to_datetime(df.index, utc=True)
    h1[s] = df.sort_index()
    d15 = pd.read_parquet(f"{CACHE}/{s}_USDT_SWAP_15m.parquet")
    d15.index = pd.to_datetime(d15.index, utc=True)
    m15[s] = d15.sort_index()

def truth_flip(sym, entry_time, entry, atr, rr):
    """Replicate the 1H walk; return (ambiguous?, engine_outcome, truth_outcome)."""
    book = h1[sym]
    idx = book.index.get_indexer([entry_time], method="nearest")
    i0 = int(idx[0])
    if abs((book.index[i0] - entry_time).total_seconds()) > 3600: return None
    sl, tp = entry - 1.0*atr, entry + rr*atr
    j = i0 + 1
    while j < len(book):
        hi, lo = book.high.iloc[j], book.low.iloc[j]
        hit_tp, hit_sl = hi >= tp, lo <= sl
        if hit_tp or hit_sl:
            eng = "TP" if hit_tp else "SL"          # engine books TP-first
            if hit_tp and hit_sl:
                b15 = m15[sym]
                seg = b15[(b15.index >= book.index[j]) & (b15.index < book.index[j] + pd.Timedelta(hours=1))]
                tru = None
                for _, r15 in seg.iterrows():
                    t_sl, t_tp = r15.low <= sl, r15.high >= tp
                    if t_sl and not t_tp: tru = "SL"; break
                    if t_tp and not t_sl: tru = "TP"; break
                    if t_sl and t_tp:      tru = "SL"; break   # 15m bar touches both: SL-first (conservative)
                return True, eng, tru
            return False, eng, eng
        j += 1
    return None

print("="*100)
print("  SL-FIRST TRUTH AUDIT (ambiguous 1H bars re-scored on 15m) — per RR, covered-symbol trades")
print("="*100)
print(f"{'RR':>5} {'n_cov':>6} {'n_ambig':>8} {'SL-first':>9} {'PF opt':>7} {'PF 15m':>7} {'PF ultra':>9}")
truth_out = {}
for rr, pre in RRS.items():
    t = pd.read_csv(f"quantlab_output/{pre}_trades_q055.csv", parse_dates=["entry_time"])
    cov = t[t.sym.str.startswith(tuple(f"{s}_" for s in COVERED))].copy()
    rows_opt, rows_truth, rows_ultra = [], [], []
    namb = 0; slfirst = 0
    for _, tr in cov.iterrows():
        sym = tr.sym.split("_")[0]
        res = truth_flip(sym, tr.entry_time, tr.entry, tr.atr, float(rr))
        if res is None: continue
        amb, eng, tru = res
        cost = 2.0*(0.05/100.0)*tr.entry/tr.atr
        r_eng = (float(rr) if eng=="TP" else -1.0) - cost
        r_tru = (float(rr) if tru=="TP" else -1.0) - cost
        r_ultra = (float(rr) if (not amb and eng=="TP") else -1.0) - cost
        rows_opt.append(r_eng); rows_truth.append(r_tru); rows_ultra.append(r_ultra)
        namb += amb; slfirst += (amb and tru=="SL")
    n = len(rows_opt)
    print(f"{rr:>5} {n:>6} {namb:>8} {slfirst:>6}/{namb:<3} {pf(rows_opt):>7.2f} {pf(rows_truth):>7.2f} {pf(rows_ultra):>9.2f}")
    truth_out[rr] = dict(n=n, namb=namb, slfirst=slfirst, opt=pf(rows_opt), tru=pf(rows_truth), ultra=pf(rows_ultra))

print()
print("="*100)
print("  ROBUSTNESS BATTERY per RR (full book, q=0.55, 0.05%/side)")
print("="*100)
rng = np.random.default_rng(11)
print(f"{'RR':>5} {'bootP(PF>1)':>12} {'LOAO-minPF':>11} {'PF@03':>6} {'PF@075':>7} {'PF@10':>6} | {'MC P(profit)':>13} {'medMDD':>7} {'p95MDD':>7}")
for rr, pre in RRS.items():
    t = pd.read_csv(f"quantlab_output/{pre}_trades_q055.csv", parse_dates=["entry_time"])
    base = t.r.values; rel = (2.0*(0.05/100.0)*t.entry/t.atr).values
    # bootstrap P(PF>1)
    wins = 0; N = 5000
    for _ in range(N):
        s = base[rng.integers(0, len(base), len(base))]
        c = rel[rng.integers(0, len(rel), len(rel))]
        x = s - c
        if pf(x) > 1.0: wins += 1
    # LOAO min PF
    loao = []
    for s in t.sym.unique():
        sub = t[t.sym != s]
        x = sub.r.values - 2.0*(0.05/100.0)*sub.entry.values/sub.atr.values
        loao.append(pf(x))
    # cost grid
    def pf_at(cp): return pf(base - 2.0*(cp/100.0)*t.entry.values/t.atr.values)
    # MC at 1% risk
    prof = 0; mdds = []
    for _ in range(N):
        idx = rng.integers(0, len(base), len(base))
        eq = 100 + (base[idx]-rel[idx]).cumsum()
        if eq[-1] > 100: prof += 1
        mdds.append((eq/np.maximum.accumulate(eq)-1).min()*100)
    mdds = np.array(mdds)
    print(f"{rr:>5} {wins/N:>12.2f} {min(loao):>11.2f} {pf_at(0.03):>6.2f} {pf_at(0.075):>7.2f} {pf_at(0.10):>6.2f} | {prof/N*100:>12.0f}% {np.median(mdds):>7.1f} {np.percentile(mdds,5):>7.1f}")

import json
json.dump(truth_out, open("quantlab_output/champ_rr_truth.json","w"), indent=1)
print("\nsaved: champ_rr_truth.json")
