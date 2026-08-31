"""
Exit-model audit for QUANTLAB frozen strategies.

Compares how the SAME frozen signal conditions resolve under different exit
simulations, to explain the discrepancy between:
  - Frozen baselines (R066/R068/R071): PF_A=3.353, PF_C=1.692   [proxy engine]
  - R072 structural forensics:          PF_A=0.528, PF_C=0.600  [SL/TP engine]

Models (all on identical IS-thresholds / OOS window, IS_RATIO=0.80):
  M1 PROXY         : R066 baseline. win = next bar close > entry close, pnl=±RR
  M2 SLTP_CAP100   : R072 verbatim. entry=next close, SL=1ATR/TP=rr*ATR,
                     intrabar TP-first, 100-bar horizon, OPEN counted as loss
  M3 SLTP_NOCAP    : entry=next close, SL/TP, NO horizon (bot-faithful entry),
                     start checks at bar AFTER entry bar (bot behavior)
  M4 SLTP_NOCAP_INC: entry=next close, SL/TP, no horizon, but include entry bar
                     in checks (R072's loop style, without the 100-bar cap)
  M5 SLTP_SIGENTRY : entry=signal bar close, SL/TP, no horizon, skip entry bar

Unresolved trades at data end are reported separately, then counted as losses
for PF (pessimistic) — same convention R072 uses for OPEN.
"""
import os, sys, math, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/quantlab")
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx

CACHE = CONFIG["CACHE_FOLDER"]
MIN_BARS = 2_000
IS_RATIO = 0.80
TRADE_RISK = 100.0
RAND_SEED = 42

STRATEGIES = {
    "FamilyA": {"label": "Family A", "cids": ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"], "rr": 2.0},
    "FamilyC": {"label": "Family C", "cids": ["ADX_ST","PBD_HI"], "rr": 3.0},
}

COND_DEF = {
    "DST_NR":     ("ema_dist_pct", "lt_q", 0.33),
    "ADX_ST":     ("adx14",        "gt_q", 0.67),
    "PBD_HI":     ("prev_body_r",  "gt_q", 0.67),
    "BBW_STRICT": ("bb_width",     "lt_q", 0.25),
    "RV_LO":      ("real_vol_20",  "lt_q", 0.33),
    "PRG_VH":     ("prev_range_r", "gt_q", 0.80),
}

def calc_rsi(series, period=14):
    delta = series.diff()
    up   = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs   = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def add_features(df):
    df = df.copy()
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    df["ema200"]        = calc_ema(c, 200)
    df["atr14"]         = calc_atr(df, 14)
    bb_mid              = c.rolling(20).mean()
    bb_std              = c.rolling(20).std(ddof=0)
    df["bb_width"]      = (bb_std * 2) / bb_mid.replace(0, np.nan) * 100.0
    df["real_vol_20"]   = c.pct_change().rolling(20).std() * 100.0
    ema200_s            = df["ema200"].replace(0, np.nan)
    df["ema_dist_pct"]  = (c - ema200_s) / ema200_s * 100.0
    df["prev_range_r"]  = (h.shift(1)-l.shift(1)).abs() / c.shift(1).replace(0,np.nan) * 100.0
    df["prev_body_r"]   = (c.shift(1)-o.shift(1)).abs() / c.shift(1).replace(0,np.nan) * 100.0
    df["adx14"]         = calc_adx(df, 14)
    return df

def entry_gate(df):
    vol_avg = df["vol"].rolling(20).mean()
    return (df["vol"] > 1.5*vol_avg) & (df["close"] > df["open"]) & \
           (df["close"] > df["close"].shift(1))

def compute_thresholds(df_is, cids):
    out = {}
    for cid in cids:
        col, direction, param = COND_DEF[cid]
        out[f"{cid}_q"] = float(df_is[col].dropna().quantile(param))
    return out

def apply_cond(df, cid, thr):
    col, direction, _ = COND_DEF[cid]
    v = df[col]
    return v < thr[f"{cid}_q"] if direction == "lt_q" else v > thr[f"{cid}_q"]

def max_dd_pct(pnls):
    eq = np.cumsum(np.array(pnls, dtype=float))
    if len(eq) == 0: return 0.0
    pk = np.maximum.accumulate(eq)
    dd = (eq - pk) / pk
    return float(dd.min()) if pk[-1] != 0 else 0.0

def summarize(pnls, rr):
    pnls = np.array(pnls, dtype=float)
    n = len(pnls)
    if n == 0:
        return dict(n=0, wr=float("nan"), pf=float("nan"), exp=0.0, mdd=0.0)
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    gw, gl = wins.sum(), abs(losses.sum())
    pf = (gw / gl) if gl > 0 else (999.0 if gw > 0 else 1.0)
    return dict(n=n, wr=float((pnls > 0).mean()), pf=pf,
                exp=float(pnls.mean()), mdd=max_dd_pct(pnls))

def run_models(cids, rr, data):
    """Returns {model: summary, ...} plus unresolved info."""
    results = {m: {"pnls": [], "n_unres": 0} for m in
               ["M1_PROXY","M2_SLTP_CAP100","M3_SLTP_NOCAP","M4_SLTP_NOCAP_INC","M5_SLTP_SIGENTRY"]}

    for sym, df_raw in data.items():
        df_f = add_features(df_raw)
        df_f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct"], inplace=True)
        if len(df_f) < MIN_BARS: continue
        n = len(df_f); is_e = int(n * IS_RATIO)
        df_is = df_f.iloc[:is_e]; df_oo = df_f.iloc[is_e:]
        thr = compute_thresholds(df_is, cids)
        gate = entry_gate(df_f)
        masks = [apply_cond(df_f, c, thr) for c in cids]
        sig = masks[0].copy()
        for m in masks[1:]: sig = sig & m
        sig = sig & gate
        sig_oo = sig.iloc[is_e:]
        oos_len = len(df_oo)
        oo_close = df_oo["close"].values
        oo_high  = df_oo["high"].values
        oo_low   = df_oo["low"].values
        oo_atr   = df_oo["atr14"].values
        idx_list = list(df_oo.index[sig_oo.values])

        for idx in idx_list:
            pos = df_oo.index.get_loc(idx)
            # M1 proxy
            if pos + 1 < oos_len:
                win = oo_close[pos + 1] > oo_close[pos]
                results["M1_PROXY"]["pnls"].append(TRADE_RISK * rr if win else -TRADE_RISK)
            # shared entry for SLTP models: next bar close
            if pos + 1 >= oos_len:
                continue
            entry_price = oo_close[pos + 1]
            atr = oo_atr[pos]
            if not (atr > 0) or math.isnan(atr): continue
            sl = entry_price - atr; tp = entry_price + rr * atr

            for mname, skip_entry, cap in (
                    ("M2_SLTP_CAP100", False, 100),
                    ("M3_SLTP_NOCAP",  True,  None),
                    ("M4_SLTP_NOCAP_INC", False, None)):
                start = pos + 1 if not skip_entry else pos + 2
                # if skipping entry bar but nothing left, trade unresolved
                if start >= oos_len:
                    results[mname]["n_unres"] += 1
                    results[mname]["pnls"].append(-TRADE_RISK)  # pessimistic, R072 convention
                    continue
                end = oos_len if cap is None else min(start + cap, oos_len)
                hit_tp = hit_sl = False
                for b in range(start, end):
                    if oo_high[b] >= tp: hit_tp = True; break
                    if oo_low[b]  <= sl: hit_sl = True; break
                if hit_tp:
                    results[mname]["pnls"].append(TRADE_RISK * rr)
                elif hit_sl:
                    results[mname]["pnls"].append(-TRADE_RISK)
                else:
                    results[mname]["n_unres"] += 1
                    results[mname]["pnls"].append(-TRADE_RISK)

            # M5: entry at signal bar close, no cap, skip entry bar
            entry_price5 = oo_close[pos]
            sl5 = entry_price5 - atr; tp5 = entry_price5 + rr * atr
            start5 = pos + 1
            if start5 >= oos_len:
                results["M5_SLTP_SIGENTRY"]["n_unres"] += 1
                results["M5_SLTP_SIGENTRY"]["pnls"].append(-TRADE_RISK)
                continue
            hit_tp = hit_sl = False
            for b in range(start5, oos_len):
                if oo_high[b] >= tp5: hit_tp = True; break
                if oo_low[b]  <= sl5: hit_sl = True; break
            if hit_tp:
                results["M5_SLTP_SIGENTRY"]["pnls"].append(TRADE_RISK * rr)
            elif hit_sl:
                results["M5_SLTP_SIGENTRY"]["pnls"].append(-TRADE_RISK)
            else:
                results["M5_SLTP_SIGENTRY"]["n_unres"] += 1
                results["M5_SLTP_SIGENTRY"]["pnls"].append(-TRADE_RISK)

    out = {}
    for m, r in results.items():
        s = summarize(r["pnls"], rr)
        s["n_unres"] = r["n_unres"]
        out[m] = s
    return out

# --------------------------------------------------------------------------
print("Loading data …")
data = {}
for fn in sorted(os.listdir(CACHE)):
    if not fn.endswith("_1H.parquet"): continue
    sym = fn.replace("_1H.parquet", "")
    try:
        df = pd.read_parquet(os.path.join(CACHE, fn))
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) >= MIN_BARS: data[sym] = df
    except Exception:
        pass
print(f"Symbols loaded: {len(data)}")

lines = []
sep = "=" * 108
for sid, scfg in STRATEGIES.items():
    print(f"\n{sep}\n{scfg['label']}  (RR={scfg['rr']})  conditions={scfg['cids']}\n{sep}")
    lines.append(f"## {scfg['label']} (RR={scfg['rr']})")
    res = run_models(scfg["cids"], scfg["rr"], data)
    hdr = f"{'Model':<22}{'n':>6}{'unres':>7}{'WR':>8}{'PF':>8}{'Exp $':>9}{'MDD%':>9}"
    print(hdr); lines.append(hdr)
    for m in ["M1_PROXY","M2_SLTP_CAP100","M3_SLTP_NOCAP","M4_SLTP_NOCAP_INC","M5_SLTP_SIGENTRY"]:
        s = res[m]
        row = (f"{m:<22}{s['n']:>6}{s['n_unres']:>7}{s['wr']*100:>7.1f}%"
               f"{s['pf']:>8.3f}{s['exp']:>9.2f}{s['mdd']*100:>8.1f}%")
        print(row); lines.append(row)
    lines.append("")

with open("/home/user/exit_model_audit.md", "w") as f:
    f.write("# QUANTLAB Exit-Model Audit\n\n")
    f.write("Same frozen signals, five exit simulations (OOS only, IS_RATIO=0.80).\n")
    f.write("Unresolved = trade still open at data end (counted as loss, pessimistic).\n\n")
    f.write("\n".join(lines) + "\n")
print("\nSaved → /home/user/exit_model_audit.md")
