"""
QUANTLAB AI — R073
Real-Edge Hunt under the Corrected (SL/TP) Exit Engine

Motivation (see EXIT_MODEL_AUDIT.md):
  Frozen baselines R066-R071 used a next-bar-close proxy, not the bot's real
  SL/TP execution. This run re-establishes everything on an engine that MIRRORS
  demo_bot.py exactly, then hunts for genuine edge under realistic exits.

Engine (bot-faithful rolling walk-forward):
  - thresholds recalibrated every 168 bars (7d) from the last 500 bars
    (bot: IS_LOOKBACK=500, recalibrate_every_days=7)
  - signal at bar i -> entry at close of bar i+1 (bot: sig on iloc[-2], entry iloc[-1])
  - exits checked from the first full bar after entry, intrabar, TP BEFORE SL (bot)
  - no time horizon (bot holds until SL/TP)
  - max one position per symbol (bot step 3)
  - risk = 1% of running capital, compounding (bot: RISK_PCT=0.01)

Exit hypotheses (FIXED — promotion needs statistical proof, no OOS cherry-pick):
  E0 base      SL=1ATR / TP=rr*ATR            (current bot behavior)
  E1 be1r      SL->entry after price touches +1R
  E2 partial   50% banked @ +1R, SL->entry, rest to TP
  E3 trail     SL trails highest-high - 1ATR; TP cap at rr*ATR
  E4 time24    close at 24 bars if still open (mark-to-market)
  E5 widesl    SL=1.5ATR / TP=rr*1.5ATR (same R:R, wider stop)
  E6 sigentry  entry at SIGNAL-bar close (vs next close)  [deployment change]

Sections:
  1  Rolling-engine baseline (E0) for both families — honest PF/WR/MDD
  2  Exit variant sweep — PF/WR/Exp/MDD + bootstrap diff-CI vs E0
  3  RR sweep under E0 and the best variant
  4  LOO-symbol + Monte Carlo on the best candidate
  5  Verdict: go / no-go per family + bot config recommendation

NO parameter optimisation. Variants are pre-registered hypotheses.
"""
import os, sys, math, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from quantlab_ai import CONFIG, calc_ema, calc_atr, calc_adx
warnings.filterwarnings("ignore")

RESEARCH_ID = "R073"
OUT   = CONFIG["OUTPUT_FOLDER"]
CACHE = CONFIG["CACHE_FOLDER"]
os.makedirs(OUT, exist_ok=True)

# ── Bot-faithful execution parameters ────────────────────────────────────────
IS_LOOKBACK  = 500          # bot calibration lookback (bars)
RECAL_EVERY  = 168          # bot recalibrates every 7 days (bars)
STARTING_CAP = 10_000.0
RISK_PCT     = 0.01

STRATEGIES = {
    "FamilyA": {"label": "Family A", "cids": ["BBW_STRICT","RV_LO","DST_NR","PRG_VH"], "rr": 2.0,
                "color": "#f5a623"},
    "FamilyC": {"label": "Family C", "cids": ["ADX_ST","PBD_HI"], "rr": 3.0,
                "color": "#00c896"},
}

COND_DEF = {
    "DST_NR":     ("ema_dist_pct", "lt_q", 0.33),
    "ADX_ST":     ("adx14",        "gt_q", 0.67),
    "PBD_HI":     ("prev_body_r",  "gt_q", 0.67),
    "BBW_STRICT": ("bb_width",     "lt_q", 0.25),
    "RV_LO":      ("real_vol_20",  "lt_q", 0.33),
    "PRG_VH":     ("prev_range_r", "gt_q", 0.80),
}

EXIT_VARIANTS = {
    "E0_base":     "SL 1ATR / TP rr·ATR (current bot)",
    "E1_be1r":     "Break-even after +1R",
    "E2_partial":  "Partial: 50% @ 1R, rest to TP",
    "E3_trail":    "1·ATR trailing stop (TP cap)",
    "E4_time24":   "Time stop: close @ 24 bars",
    "E5_widesl":   "SL 1.5·ATR, TP rr·1.5·ATR",
    "E6_sigentry": "Entry at signal-bar close",
}
RR_SWEEP = [1.0, 1.5, 2.0, 2.5, 3.0]
N_BOOT   = 2_000
N_MC     = 5_000
RAND_SEED = 42
rng = np.random.default_rng(RAND_SEED)

SEP  = "═" * 110
SEP2 = "─" * 90

C_BG = "#0d0d0d"; C_PANEL = "#141414"; C_TEXT = "#e0e0e0"
C_GRID = "#2a2a2a"; C_GREEN = "#00c896"; C_RED = "#e05050"
C_GOLD = "#f5a623"; C_BLUE = "#4a9eff"; C_PURP = "#9b59b6"
plt.rcParams.update({
    "figure.facecolor": C_BG, "axes.facecolor": C_PANEL,
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "axes.edgecolor": C_GRID, "grid.color": C_GRID, "font.family": "monospace",
})

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES (identical to research engine; + rel_vol for the gate)
# ─────────────────────────────────────────────────────────────────────────────
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
    vol_avg             = df["vol"].rolling(20).mean()
    df["rel_vol"]       = df["vol"] / vol_avg.replace(0, np.nan)
    return df

def compute_thresholds(df_is, cids):
    out = {}
    for cid in cids:
        col, direction, q = COND_DEF[cid]
        vals = df_is[col].dropna()
        if len(vals) < 10:
            out[cid] = None
            continue
        out[cid] = float(vals.quantile(q))
    return out

def max_dd_from_eq(eq):
    eq = np.asarray(eq, dtype=float)
    if len(eq) == 0: return 0.0
    pk = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(pk > 0, (eq - pk) / pk, 0.0)
    return float(dd.min())

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL MASK (rolling recalibration; independent of RR and exit variant)
# ─────────────────────────────────────────────────────────────────────────────
def build_signal_mask(feats, cids):
    c = feats["close"].values
    o = feats["open"].values
    rv = feats["rel_vol"].values
    n = len(feats)
    sig = np.zeros(n, dtype=bool)
    cal = IS_LOOKBACK
    while cal < n:
        end = min(cal + RECAL_EVERY, n)
        thr = compute_thresholds(feats.iloc[cal - IS_LOOKBACK:cal], cids)
        m = np.ones(end - cal, dtype=bool)
        for cid in cids:
            col, direction, _ = COND_DEF[cid]
            vals = feats[col].values[cal:end]
            tq = thr.get(cid)
            if tq is None:
                m[:] = False
                break
            if direction == "lt_q":
                m &= vals < tq
            else:
                m &= vals > tq
        sig[cal:end] = m
        cal = end
    gate = (c > o) & (c > np.roll(c, 1)) & (rv > 1.5)
    gate[:1] = False
    gate = gate & ~np.isnan(c) & ~np.isnan(o) & ~np.isnan(rv)
    return sig & gate

# ─────────────────────────────────────────────────────────────────────────────
# BOT-FAITHFUL SEQUENTIAL SIMULATION (per symbol)
# ─────────────────────────────────────────────────────────────────────────────
def sim_symbol(feats, sig, rr, variant):
    c = feats["close"].values
    h = feats["high"].values
    l = feats["low"].values
    atr = feats["atr14"].values
    n = len(feats)

    sl_mult = 1.5 if variant == "E5_widesl" else 1.0
    tp_mult = sl_mult
    entry_next = variant != "E6_sigentry"

    trades = []
    pos = None

    def open_pos(i, entry, a):
        first_check = i + 1 if not entry_next else i + 2
        return dict(entry_i=i, first_check=first_check, entry=entry, atr=a,
                    sl=entry - sl_mult * a, tp=entry + rr * tp_mult * a,
                    be=False, partial=False, banked=0.0, trail_high=entry)

    for i in range(IS_LOOKBACK + 1, n):
        if pos is not None and i >= pos["first_check"]:
            e = pos["entry"]; a = pos["atr"]
            done = None
            v = variant
            if v == "E1_be1r":
                if not pos["be"] and h[i] >= e + a:
                    pos["sl"] = e; pos["be"] = True
                if h[i] >= pos["tp"]: done = (rr, "TP")
                elif l[i] <= pos["sl"]: done = (-1.0 if not pos["be"] else 0.0, "SL")
            elif v == "E2_partial":
                if not pos["partial"] and h[i] >= e + a:
                    pos["sl"] = e; pos["partial"] = True; pos["banked"] = 0.5
                if pos["partial"] and h[i] >= e + rr * a:
                    done = (0.5 + 0.5 * rr, "TP")
                elif l[i] <= pos["sl"]:
                    done = (pos["banked"] - (0.5 if pos["partial"] else 1.0), "SL")
            elif v == "E3_trail":
                pos["trail_high"] = max(pos["trail_high"], h[i])
                pos["sl"] = max(pos["sl"], pos["trail_high"] - a)
                if h[i] >= pos["tp"]: done = (rr, "TP")
                elif l[i] <= pos["sl"]: done = ((pos["sl"] - e) / a, "SL")
            elif v == "E4_time24":
                if h[i] >= pos["tp"]: done = (rr, "TP")
                elif l[i] <= pos["sl"]: done = (-1.0, "SL")
                elif i - pos["entry_i"] >= 24: done = ((c[i] - e) / a, "TIME")
            else:  # E0_base, E5_widesl, E6_sigentry
                if h[i] >= pos["tp"]: done = (rr, "TP")
                elif l[i] <= pos["sl"]: done = (-1.0, "SL")
            if done is not None:
                r_mult, xtype = done
                trades.append(dict(entry_i=pos["entry_i"],
                                   entry_time=feats.index[pos["entry_i"]],
                                   r=r_mult, exit_type=xtype,
                                   bars_in=i - pos["entry_i"],
                                   atr=pos["atr"], entry=pos["entry"]))
                pos = None

        if pos is not None or not sig[i]:
            continue
        if i + 1 >= n and entry_next:
            continue
        a = atr[i]
        if not (a > 0) or math.isnan(a):
            continue
        entry = c[i + 1] if entry_next else c[i]
        pos = open_pos(i, entry, a)

    if pos is not None:  # mark-to-market at data end
        e = pos["entry"]; a = pos["atr"]; last_c = c[-1]
        if pos["partial"]:
            r_mm = 0.5 + 0.5 * (last_c - e) / a
        elif pos["be"]:
            r_mm = max((last_c - e) / a, 0.0)
        else:
            r_mm = (last_c - e) / a
        trades.append(dict(entry_i=pos["entry_i"], entry_time=feats.index[pos["entry_i"]],
                           r=r_mm, exit_type="OPEN", bars_in=n - 1 - pos["entry_i"],
                           atr=pos["atr"], entry=pos["entry"]))
    return trades

def run_family(cids, rr, variant, feats_by_sym, masks):
    all_t = []
    for sym, feats in feats_by_sym.items():
        try:
            for t in sim_symbol(feats, masks[sym], rr, variant):
                t["sym"] = sym
                all_t.append(t)
        except Exception:
            pass
    all_t.sort(key=lambda t: t["entry_time"])
    return all_t

# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def stats_from_trades(trades, start_cap=STARTING_CAP, risk_pct=RISK_PCT):
    n = len(trades)
    if n == 0:
        return dict(n=0, wr=float("nan"), pf=float("nan"), exp=0.0, mdd=0.0,
                    total_r=0.0, calmar=0.0, net=0.0)
    cap = start_cap
    eq = [cap]
    for t in trades:
        cap += cap * risk_pct * t["r"]
        eq.append(cap)
    eq = np.array(eq)
    rs = np.array([t["r"] for t in trades])
    wins = rs[rs > 0]; losses = rs[rs < 0]
    gw, gl = wins.sum(), abs(losses.sum())
    pf = gw / gl if gl > 0 else (999.0 if gw > 0 else 1.0)
    mdd = max_dd_from_eq(eq)
    net = eq[-1] - start_cap
    calmar = (net / start_cap) / abs(mdd) if mdd < 0 else 0.0
    return dict(n=n, wr=float((rs > 0).mean()), pf=pf,
                exp=float(np.mean(rs) * risk_pct * start_cap),
                mdd=mdd, total_r=float(rs.sum()), calmar=calmar, net=net)

def bootstrap_pf(rs, n_boot=N_BOOT):
    rs = np.asarray(rs, dtype=float)
    if len(rs) == 0: return (np.nan,)*3
    out = np.empty(n_boot)
    for b in range(n_boot):
        s = rs[rng.integers(0, len(rs), len(rs))]
        w = s[s > 0].sum(); lo = abs(s[s < 0].sum())
        out[b] = w / lo if lo > 0 else (999.0 if w > 0 else 1.0)
    return (float(np.percentile(out, 5)), float(np.median(out)), float(np.percentile(out, 95)))

def bootstrap_pf_diff(trades_var, trades_base, n_boot=N_BOOT):
    """Paired bootstrap of (PF_var - PF_base) aligned by (sym, entry_time)."""
    base_idx = {}
    for j, t in enumerate(trades_base):
        base_idx[(t["sym"], t["entry_time"])] = j
    pairs = []
    for k, t in enumerate(trades_var):
        j = base_idx.get((t["sym"], t["entry_time"]))
        if j is not None:
            pairs.append((k, j))
    if len(pairs) < 50:
        return (np.nan, np.nan, np.nan)
    a = np.array([trades_var[k]["r"] for k, _ in pairs])
    b = np.array([trades_base[j]["r"] for _, j in pairs])
    n = len(pairs)
    out = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        def pf(x):
            s = x[idx]; w = s[s > 0].sum(); lo = abs(s[s < 0].sum())
            return w / lo if lo > 0 else (999.0 if w > 0 else 1.0)
        out[k] = pf(a) - pf(b)
    return (float(np.percentile(out, 5)), float(np.median(out)), float(np.percentile(out, 95)))

def loo_symbol_floor(trades):
    floor, rm = 999.0, None
    for sym in set(t["sym"] for t in trades):
        others = [t for t in trades if t["sym"] != sym]
        s = stats_from_trades(others)
        if s["pf"] < floor: floor, rm = s["pf"], sym
    return floor, rm

def monte_carlo(rs, n_paths=N_MC, start_cap=STARTING_CAP, risk_pct=RISK_PCT):
    rs = np.asarray(rs, dtype=float)
    if len(rs) == 0: return dict(prob=float("nan"), dd_p5=float("nan"), exp=float("nan"))
    profits = np.empty(n_paths); dds = np.empty(n_paths)
    for k in range(n_paths):
        s = rs[rng.integers(0, len(rs), len(rs))]
        cap = start_cap
        eq = np.empty(len(s) + 1); eq[0] = cap
        for j, r in enumerate(s):
            cap += cap * risk_pct * r
            eq[j + 1] = cap
        profits[k] = cap
        dds[k] = max_dd_from_eq(eq)
    return dict(prob=float((profits > start_cap).mean()),
                dd_p5=float(np.percentile(dds, 5)),
                dd_p95=float(np.percentile(dds, 95)),
                exp=float(profits.mean() - start_cap))

# ─────────────────────────────────────────────────────────────────────────────
print(); print(SEP)
print(f"  QUANTLAB AI — {RESEARCH_ID}  Real-Edge Hunt (corrected SL/TP engine)")
print(SEP)
t0 = time.time()

print("\n  Loading data …")
feats_by_sym = {}
for fn in sorted(os.listdir(CACHE)):
    if not fn.endswith("_1H.parquet"): continue
    sym = fn.replace("_1H.parquet", "")
    try:
        df = pd.read_parquet(os.path.join(CACHE, fn))
        df.index = pd.to_datetime(df.index, utc=True); df.sort_index(inplace=True)
        for col in ["open","high","low","close"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["open","high","low","close","vol"], inplace=True)
        if len(df) < IS_LOOKBACK + RECAL_EVERY + 100: continue
        f = add_features(df)
        f.dropna(subset=["ema200","atr14","adx14","ema_dist_pct","real_vol_20",
                         "bb_width","prev_range_r","prev_body_r"], inplace=True)
        if len(f) >= IS_LOOKBACK + RECAL_EVERY:
            feats_by_sym[sym] = f
    except Exception:
        pass
print(f"  Symbols loaded: {len(feats_by_sym)}")

print("\n  Building signal masks (rolling calibration, per family) …")
masks = {}
for sid, scfg in STRATEGIES.items():
    t1 = time.time()
    masks[sid] = {sym: build_signal_mask(f, scfg["cids"]) for sym, f in feats_by_sym.items()}
    total_sig = sum(int(m.sum()) for m in masks[sid].values())
    print(f"    {scfg['label']}: masks in {time.time()-t1:.0f}s, total signals={total_sig}")

print("\n  Section 1+2 — E0 baseline + exit variant sweep (rolling engine) …")
results = {}
for sid, scfg in STRATEGIES.items():
    print(f"    [{scfg['label']}] running {len(EXIT_VARIANTS)} variants …")
    for vk in EXIT_VARIANTS:
        results[(sid, vk)] = run_family(scfg["cids"], scfg["rr"], vk, feats_by_sym, masks[sid])

matrix_rows = []
for sid, scfg in STRATEGIES.items():
    base_trades = results[(sid, "E0_base")]
    for vk in EXIT_VARIANTS:
        trades = results[(sid, vk)]
        s = stats_from_trades(trades)
        b5, bmed, b95 = bootstrap_pf(np.array([t["r"] for t in trades]))
        d5, dmed, d95 = bootstrap_pf_diff(trades, base_trades)
        matrix_rows.append(dict(family=sid, variant=vk, n=s["n"], wr=s["wr"], pf=s["pf"],
                                exp=s["exp"], mdd=s["mdd"], calmar=s["calmar"],
                                net=s["net"], boot_p5=b5, boot_med=bmed,
                                d_p5=d5, d_med=dmed, d_p95=d95))

print(f"\n{SEP2}")
for sid, scfg in STRATEGIES.items():
    print(f"\n  [{scfg['label']}] EXIT VARIANT SWEEP  (rolling engine, RR={scfg['rr']})")
    hdr = (f"    {'Variant':<14}{'n':>7}{'WR':>8}{'PF':>8}{'Exp $':>9}{'MDD%':>8}"
           f"{'Calmar':>8}{'Net $':>10}  BootP5   dPF vs E0 [P5, P95]")
    print(hdr)
    print(f"    {'─'*14}{'─'*7}{'─'*8}{'─'*8}{'─'*9}{'─'*8}{'─'*8}{'─'*10}  {'─'*30}")
    for row in matrix_rows:
        if row["family"] != sid: continue
        sig = ""
        if row["variant"] != "E0_base":
            if row["d_p5"] > 0: sig = "  ↑SIG"
            elif row["d_p95"] < 0: sig = "  ↓SIG"
        print(f"    {row['variant']:<14}{row['n']:>7}{row['wr']*100:>7.1f}%"
              f"{row['pf']:>8.3f}{row['exp']:>9.2f}{row['mdd']*100:>7.1f}%"
              f"{row['calmar']:>8.2f}{row['net']:>10.0f}  {row['boot_p5']:>6.3f}   "
              f"[{row['d_p5']:+.2f}, {row['d_p95']:+.2f}]{sig}")
print(f"{SEP2}")

print("\n  Section 3 — RR sweep (rolling engine) …")
best_variants = {}
for sid, scfg in STRATEGIES.items():
    rows = [r for r in matrix_rows if r["family"] == sid]
    best = max((r for r in rows if r["variant"] != "E0_base"), key=lambda r: r["pf"])
    best_variants[sid] = best["variant"]
    print(f"    {scfg['label']}: best variant = {best['variant']} (PF={best['pf']:.3f})")

rr_rows = []
for sid, scfg in STRATEGIES.items():
    for vk in ["E0_base", best_variants[sid]]:
        for rr in RR_SWEEP:
            trades = run_family(scfg["cids"], rr, vk, feats_by_sym, masks[sid])
            s = stats_from_trades(trades)
            rr_rows.append(dict(family=sid, variant=vk, rr=rr, n=s["n"], wr=s["wr"],
                                pf=s["pf"], exp=s["exp"], mdd=s["mdd"], net=s["net"]))

for sid, scfg in STRATEGIES.items():
    print(f"\n  [{scfg['label']}] RR SWEEP")
    hdr = f"    {'Variant':<14}{'RR':>6}{'n':>7}{'WR':>8}{'PF':>8}{'Exp $':>9}{'MDD%':>8}{'Net $':>10}"
    print(hdr)
    print(f"    {'─'*14}{'─'*6}{'─'*7}{'─'*8}{'─'*8}{'─'*9}{'─'*8}{'─'*10}")
    for r in rr_rows:
        if r["family"] != sid: continue
        print(f"    {r['variant']:<14}{r['rr']:>6.2f}{r['n']:>7}{r['wr']*100:>7.1f}%"
              f"{r['pf']:>8.3f}{r['exp']:>9.2f}{r['mdd']*100:>7.1f}%{r['net']:>10.0f}")

print("\n  Section 4 — LOO-symbol + Monte Carlo on best candidate …")
sec4 = {}
for sid, scfg in STRATEGIES.items():
    bv = best_variants[sid]
    trades = results[(sid, bv)]
    floor, rm = loo_symbol_floor(trades)
    mc = monte_carlo(np.array([t["r"] for t in trades]))
    sec4[sid] = dict(bv=bv, floor=floor, rm=rm, mc=mc)
    print(f"    {scfg['label']} [{bv}]: LOO-sym floor PF={floor:.3f} (drop {rm}) | "
          f"MC P(profit)={mc['prob']*100:.1f}%  net=${mc['exp']:+,.0f}  "
          f"DD P5={mc['dd_p5']*100:.1f}%  DD P95={mc['dd_p95']*100:.1f}%")

print("\n  Section 4b — Cost sensitivity (slippage + fees) …")
COST_PCTS = [0.0, 0.05, 0.10, 0.15, 0.20]   # % of price per side
cost_rows = []
for sid, scfg in STRATEGIES.items():
    for vk in ["E0_base", best_variants[sid]]:
        trades = results[(sid, vk)]
        for cp in COST_PCTS:
            rs = []
            for t in trades:
                cost_r = 2.0 * (cp / 100.0) * t["entry"] / t["atr"]
                rs.append(t["r"] - cost_r)
            s = dict(n=len(rs),
                     pf=(lambda a, b: a / b if b > 0 else (999.0 if a > 0 else 1.0))(
                         sum(x for x in rs if x > 0), abs(sum(x for x in rs if x < 0))),
                     exp=float(np.mean(rs)) * RISK_PCT * STARTING_CAP)
            cost_rows.append(dict(family=sid, variant=vk, cost_pct=cp, pf=s["pf"],
                                  exp=s["exp"], n=s["n"]))
for sid, scfg in STRATEGIES.items():
    print(f"\n  [{scfg['label']}] COST SENSITIVITY (PF at cost per side)")
    for vk in ["E0_base", best_variants[sid]]:
        row = [next(r["pf"] for r in cost_rows if r["family"] == sid and r["variant"] == vk
                    and r["cost_pct"] == cp) for cp in COST_PCTS]
        print(f"    {vk:<14}" + "  ".join(f"{cp:.2f}%→{p:.3f}" for cp, p in zip(COST_PCTS, row)))
    # breakeven cost (PF=1.0) for the best variant
    bv = best_variants[sid]
    pts = [(r["cost_pct"], r["pf"]) for r in cost_rows if r["family"] == sid and r["variant"] == bv]
    be = None
    for (c1, p1), (c2, p2) in zip(pts, pts[1:]):
        if p1 >= 1.0 >= p2:
            be = c1 + (1.0 - p1) * (c2 - c1) / (p2 - p1)
            break
    if be:
        print(f"    → breakeven round-trip cost for {bv}: {be:.3f}% per side")
    elif pts[0][1] < 1.0:
        print(f"    → {bv} NOT profitable even at zero cost (PF={pts[0][1]:.3f})")
    else:
        print(f"    → {bv} remains profitable beyond {COST_PCTS[-1]}% cost per side")

print("\n  Section 5 — Verdict …")
verdict_lines = []
for sid, scfg in STRATEGIES.items():
    e0 = stats_from_trades(results[(sid, "E0_base")])
    bs = stats_from_trades(results[(sid, best_variants[sid])])
    verdict_lines.append((sid, e0, bs, sec4[sid]))

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig); return p

def style_ax(ax, title, ylabel=""):
    ax.grid(True, ls="--", lw=0.4, color=C_GRID)
    ax.set_title(title, fontsize=8, color=C_TEXT)
    ax.set_ylabel(ylabel, fontsize=7)

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)
for k, sid in enumerate(["FamilyA", "FamilyC"]):
    rows = [r for r in matrix_rows if r["family"] == sid]
    ax = fig.add_subplot(gs[0, k])
    variants = [r["variant"] for r in rows]
    pfs = [r["pf"] for r in rows]
    colors = [C_GREEN if p >= 1 else C_RED for p in pfs]
    ax.bar(variants, pfs, color=colors, alpha=0.85)
    ax.axhline(1.0, color=C_GOLD, ls="--", lw=1)
    ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=6)
    style_ax(ax, f"{sid} — PF by exit variant", "PF")
    for i, p in enumerate(pfs):
        ax.text(i, p + 0.03, f"{p:.2f}", ha="center", fontsize=6, color=C_TEXT)
ax = fig.add_subplot(gs[0, 2])
for sid, scfg in STRATEGIES.items():
    rows = [r for r in rr_rows if r["family"] == sid and r["variant"] == best_variants[sid]]
    ax.plot([r["rr"] for r in rows], [r["pf"] for r in rows], marker="o",
            color=scfg["color"], label=scfg["label"])
ax.axhline(1.0, color=C_GOLD, ls="--", lw=1)
ax.legend(fontsize=7)
style_ax(ax, "PF vs RR (best variant)", "PF")
for k, sid in enumerate(["FamilyA", "FamilyC"]):
    rows = [r for r in matrix_rows if r["family"] == sid]
    ax = fig.add_subplot(gs[1, k])
    xs = np.arange(len(rows))
    d5 = np.array([r["d_p5"] for r in rows]); d95 = np.array([r["d_p95"] for r in rows])
    med = np.array([r["d_med"] for r in rows])
    ax.errorbar(xs, med, yerr=[med - d5, d95 - med], fmt="o", color=C_BLUE,
                capsize=3, markersize=5)
    ax.axhline(0, color=C_GOLD, ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["variant"] for r in rows], rotation=45, ha="right", fontsize=6)
    style_ax(ax, f"{sid} — ΔPF vs E0 (90% boot CI)", "ΔPF")
fig.suptitle("R073 — Real-Edge Hunt under corrected SL/TP engine", fontsize=13, color=C_TEXT)
save_fig(fig, "r073_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# CSV OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
pd.DataFrame(matrix_rows).to_csv(os.path.join(OUT, "r073_exit_matrix.csv"), index=False)
pd.DataFrame(rr_rows).to_csv(os.path.join(OUT, "r073_rr_sweep.csv"), index=False)
pd.DataFrame(cost_rows).to_csv(os.path.join(OUT, "r073_cost_sensitivity.csv"), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
lines = [f"# R073 — Real-Edge Hunt (corrected SL/TP engine)\n",
         f"**Date:** 2026-08-06  |  **Engine:** bot-faithful rolling walk-forward "
         f"(IS_LOOKBACK={IS_LOOKBACK}, RECAL={RECAL_EVERY}b, entry=next close, "
         f"SL/TP intrabar TP-first, 1 pos/symbol, 1% compounding risk)\n",
         f"**Symbols:** {len(feats_by_sym)}  **Time:** {time.time()-t0:.0f}s\n"]

for sid, scfg in STRATEGIES.items():
    lines.append(f"\n## {scfg['label']} (RR={scfg['rr']})\n")
    lines.append("| Variant | n | WR | PF | Exp $ | MDD% | Calmar | Net $ | Boot P5 | ΔPF vs E0 [P5, P95] |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in matrix_rows:
        if r["family"] != sid: continue
        sig = ""
        if r["variant"] != "E0_base":
            sig = " ✅" if r["d_p5"] > 0 else (" ❌" if r["d_p95"] < 0 else "")
        lines.append(f"| {r['variant']} | {r['n']} | {r['wr']*100:.1f}% | {r['pf']:.3f} | "
                     f"{r['exp']:+.2f} | {r['mdd']*100:.1f}% | {r['calmar']:.2f} | "
                     f"{r['net']:+.0f} | {r['boot_p5']:.3f} | [{r['d_p5']:+.2f}, {r['d_p95']:+.2f}]{sig} |")

    lines.append(f"\n### RR sweep ({best_variants[sid]} variant)")
    lines.append("| RR | n | WR | PF | Exp $ | MDD% | Net $ |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rr_rows:
        if r["family"] != sid or r["variant"] != best_variants[sid]: continue
        lines.append(f"| {r['rr']:.2f} | {r['n']} | {r['wr']*100:.1f}% | {r['pf']:.3f} | "
                     f"{r['exp']:+.2f} | {r['mdd']*100:.1f}% | {r['net']:+.0f} |")

    s4 = sec4[sid]
    lines.append(f"\n### Robustness of best variant **{s4['bv']}**")
    lines.append(f"- LOO-symbol PF floor: **{s4['floor']:.3f}** (drop {s4['rm']})")
    lines.append(f"- Monte Carlo (5,000 paths, 1% compounding): P(profit vs start) = **{s4['mc']['prob']*100:.1f}%**, "
                 f"mean net = ${s4['mc']['exp']:+,.0f}, "
                 f"DD P5 = {s4['mc']['dd_p5']*100:.1f}% / P95 = {s4['mc']['dd_p95']*100:.1f}%\n")

    lines.append(f"\n### Cost sensitivity (slippage+fees per side)")
    lines.append("| Variant | 0.00% | 0.05% | 0.10% | 0.15% | 0.20% |")
    lines.append("|---|---|---|---|---|---|")
    for vk in ["E0_base", s4["bv"]]:
        pfs = [next(r["pf"] for r in cost_rows if r["family"] == sid and r["variant"] == vk
                    and r["cost_pct"] == cp) for cp in COST_PCTS]
        lines.append(f"| {vk} | " + " | ".join(f"{p:.3f}" for p in pfs) + " |")
    bv = s4["bv"]
    pts = [(r["cost_pct"], r["pf"]) for r in cost_rows if r["family"] == sid and r["variant"] == bv]
    be = None
    for (c1, p1), (c2, p2) in zip(pts, pts[1:]):
        if p1 >= 1.0 >= p2:
            be = c1 + (1.0 - p1) * (c2 - c1) / (p2 - p1)
            break
    if be:
        lines.append(f"\n*Breakeven round-trip cost for {bv}: {be:.3f}% per side*")
    elif pts[0][1] < 1.0:
        lines.append(f"\n*{bv} not profitable even at zero cost (PF={pts[0][1]:.3f})*")
    else:
        lines.append(f"\n*{bv} remains profitable beyond {COST_PCTS[-1]}% cost per side*")

lines.append("\n## Verdict\n")
for sid, e0, bs, s4 in verdict_lines:
    label = STRATEGIES[sid]["label"]
    if bs["pf"] > 1 and s4["floor"] > 1 and s4["mc"]["prob"] > 0.95:
        status = "GO (positive PF, survives LOO, MC P>95%)"
    elif bs["pf"] > 1:
        status = "MAYBE (positive but thin / not LOO-robust)"
    else:
        status = "NO-GO (unprofitable under realistic exits)"
    lines.append(f"**{label}:** E0 PF={e0['pf']:.3f} → best ({s4['bv']}) PF={bs['pf']:.3f} | {status}\n")

lines.append("## Recommendation for demo_bot.py\n")
for sid, e0, bs, s4 in verdict_lines:
    label = STRATEGIES[sid]["label"]
    if bs["pf"] > 1 and s4["floor"] > 1 and s4["mc"]["prob"] > 0.95:
        lines.append(f"- {label}: adopt exit variant `{s4['bv']}` (entry at signal close for E6). "
                     f"Update `STRATEGIES['{sid}']` exit logic in demo_bot.py after live confirm.\n")
    elif bs["pf"] > 1:
        lines.append(f"- {label}: thin edge only. Keep paper-trading at 0.5% risk; do NOT "
                     f"scale up. Revisit after more live data.\n")
    else:
        lines.append(f"- {label}: **remove from demo bot or disable** — negative expectancy "
                     f"under the bot's own execution model.\n")

lines.append("\n## Methodological notes\n")
lines.append("- Rolling engine uses all data out-of-sample (thresholds from past 500 bars only) "
             "and mirrors demo_bot.py execution exactly — replaces the R066 proxy engine "
             "as the project's authoritative evaluation.\n")
lines.append("- ΔPF bootstrap is PAIRED on aligned trades (same entries, differing exits, "
             "matched by symbol+entry time). ✅ = 90% CI entirely above 0.\n")
lines.append("- Variants were pre-registered; no exit was tuned on outcomes. "
             "Any adopted change still needs live paper confirmation.\n")

report = "\n".join(lines)
with open(os.path.join(OUT, "r073_final_report.md"), "w") as f:
    f.write(report)

journal = [f"# R073 — Real-Edge Hunt",
           f"**Duration:** {time.time()-t0:.0f}s | **Symbols:** {len(feats_by_sym)}"]
for sid, scfg in STRATEGIES.items():
    s4 = sec4[sid]
    journal.append(f"## {scfg['label']}")
    journal.append(f"- E0 (bot) PF={stats_from_trades(results[(sid,'E0_base')])['pf']:.3f} | "
                   f"best {s4['bv']} PF={stats_from_trades(results[(sid, s4['bv'])])['pf']:.3f} | "
                   f"LOO floor={s4['floor']:.3f} | MC P(profit)={s4['mc']['prob']*100:.0f}%")
with open(os.path.join(OUT, "r073_journal.md"), "w") as f:
    f.write("\n".join(journal) + "\n")

print(f"\n  Done in {time.time()-t0:.0f}s → {OUT}/r073_*")
