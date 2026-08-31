"""
R081b: push profitable-months% higher on the winning scalp config
(Family A raw + RR0.4/0.5) by adding MILD filters (volceil, breadth)
that trim the worst trades without collapsing frequency.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/quantlab")
sys.path.insert(0, "/home/user/quantlab/scripts")
from quantlab_ai import CONFIG
from scripts.ql_engine import (
    add_features, build_signal_mask, sim_symbol, stats_from_trades,
    cost_adjusted_rs, pf_of_rs, IS_LOOKBACK, RECAL_EVERY,
)

OUT = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")

ORIGINAL = {
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

feats = {}
for sym in ORIGINAL:
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
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats[sym] = f
    except Exception:
        pass

famA_cids = ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"]
base_mask = {s: build_signal_mask(f, famA_cids, "green", 1.5) for s, f in feats.items()}
above = {s: (f["close"] > f["ema20"]).astype(float) for s, f in feats.items()}
breadth = pd.DataFrame(above).sort_index().mean(axis=1, skipna=True)

def run(rr, volceil=None, breadth_thr=None):
    mask = {}
    for s, m in base_mask.items():
        f = feats[s]
        mm = m.copy()
        if volceil is not None:
            mm = mm & (f["atr_rank"].fillna(100) <= volceil).values
        if breadth_thr is not None:
            reg = (breadth.reindex(f.index, method="ffill") > breadth_thr).fillna(False)
            mm = mm & reg.values
        mask[s] = mm
    cfg = dict(entry_next=False, exit="base", hours=None)
    out = []
    for sym, f in feats.items():
        try:
            for t in sim_symbol(f, mask[sym], rr, cfg):
                t["sym"] = sym
                out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t["entry_time"])
    return out

def monthly_profile(trades):
    if not trades: return dict(prof=float("nan"), worst=float("nan"), tpm=0.0)
    df = pd.DataFrame(trades)
    df["month"] = df["entry_time"].dt.to_period("M")
    g = df.groupby("month")["r"].sum()
    flags = (g > 0).astype(int).values
    cur = worst = 0
    for v in flags:
        cur = cur + 1 if not v else 0
        worst = max(worst, cur)
    return dict(prof=float((g > 0).mean()), worst=worst, tpm=len(df) / len(g))

CONFIGS = [
    ("rr04_raw",        0.4, None,   None),
    ("rr04_vc80",       0.4, 80,    None),
    ("rr04_vc90",       0.4, 90,    None),
    ("rr04_br40",       0.4, None,  0.40),
    ("rr04_vc80_br40",  0.4, 80,    0.40),
    ("rr04_vc90_br40",  0.4, 90,    0.40),
    ("rr05_raw",        0.5, None,  None),
    ("rr05_vc80",       0.5, 80,    None),
    ("rr05_vc90",       0.5, 90,    None),
    ("rr05_vc80_br40",  0.5, 80,    0.40),
]

print(f"{'Config':<18}{'n':>6}{'t/mo':>7}{'WR':>7}{'PF':>7}{'PF@c':>7}{'MDD%':>8}{'prof%':>7}{'worst':>6}{'selPF':>7}{'holPF':>7}")
rows = []
for name, rr, vc, br in CONFIGS:
    trades = run(rr, vc, br)
    s = stats_from_trades(trades)
    rs = np.array([t["r"] for t in trades])
    pf_c = pf_of_rs(cost_adjusted_rs(trades, 0.05))
    sel = [t for t in trades if t["entry_time"] < HOLDOUT_START]
    hol = [t for t in trades if t["entry_time"] >= HOLDOUT_START]
    sp = stats_from_trades(sel)["pf"]
    hp = stats_from_trades(hol)["pf"]
    mp = monthly_profile(trades)
    rows.append(dict(name=name, n=len(trades), tpm=mp["tpm"], wr=s["wr"], pf=s["pf"],
                     pf_c=pf_c, mdd=s["mdd"], prof=mp["prof"], worst=mp["worst"],
                     selpf=sp, holpf=hp))
    print(f"{name:<18}{len(trades):>6}{mp['tpm']:>7.1f}{s['wr']*100:>6.0f}%"
          f"{s['pf']:>7.2f}{pf_c:>7.2f}{s['mdd']*100:>7.1f}%{mp['prof']*100:>6.0f}%"
          f"{mp['worst']:>6}{sp:>7.2f}{hp:>7.2f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "r081b_scalp_filters.csv"), index=False)
print("\nSaved →", os.path.join(OUT, "r081b_scalp_filters.csv"))
