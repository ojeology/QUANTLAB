"""
NEGATIVE CONTROL for the blind-test harness.

Research (R089) PROVED 5m crypto has NO edge: raw Family A on 5m = PF 0.59,
holPF 0.50, WR 28%. If my blind_test.py harness is valid, running the SAME
frozen-strategy logic on 5m data MUST also fail (PF < 1). If it instead shows
PF ~2.9, the harness is a false-positive machine and the 1H results are garbage.

Method is identical to blind_test.py except the data is 5m (cached pre-freeze
history + fresh post-freeze OKX 5m bars), split at the same 2026-08-08 freeze.
"""
import warnings, logging, time, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import demo_bot as bot
logging.getLogger("demo_bot").setLevel(logging.ERROR)

FREEZE    = pd.Timestamp("2026-08-08 00:00:00", tz="UTC")
START_CAP = 10000.0
RISK_PCT  = 0.01
CACHE     = "quantlab_cache"

# No 5m parquet cache exists, so pull fresh 5m OKX bars for liquid symbols.
# n_bars=12000 ≈ 42 days → spans pre-freeze (thresholds) + post-freeze (blind test).
SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP",
           "XRP-USDT-SWAP", "BNB-USDT-SWAP", "AVAX-USDT-SWAP", "LINK-USDT-SWAP"]
fivem = {s: None for s in SYMBOLS}
print(f"[control] 5m symbols (fresh fetch) = {len(fivem)}: {sorted(fivem)}", flush=True)


def fetch_5m(inst, n_bars=12000):
    all_rows, after, pages = [], None, 0
    while len(all_rows) < n_bars and pages < 80:
        params = {"instId": inst, "bar": "5m", "limit": bot.PAGE_LIMIT}
        if after:
            params["after"] = str(after)
        raw = bot._get(bot.OKX_CANDLES, params)
        if not raw:
            raw = bot._get(bot.OKX_CANDLES_CUR, params)
            if not raw:
                break
        all_rows.extend(raw); pages += 1
        oldest = int(raw[-1][0]); after = oldest
        if len(all_rows) >= n_bars:
            break
        time.sleep(bot.PAGE_DELAY)
    if not all_rows:
        return None
    df = pd.DataFrame(all_rows, columns=bot.CANDLE_COLS)
    df["ts"] = pd.to_numeric(df["ts"])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return (df[["datetime", "open", "high", "low", "close", "vol"]]
            .sort_values("datetime").drop_duplicates("datetime")
            .reset_index(drop=True).set_index("datetime"))


def load_symbol(inst):
    fresh = fetch_5m(inst)
    if fresh is None or len(fresh) < 600:
        return None
    return fresh[["open", "high", "low", "close", "vol"]].astype(float)


class Strat:
    def __init__(self, sid):
        self.sid = sid; self.cap = START_CAP; self.peak = START_CAP
        self.trades = []; self.pos = {}; self.max_dd = 0.0
        self.day_pnl = {}; self.paused_until = None
    @property
    def dd(self):
        return (self.cap - self.peak) / self.peak if self.peak > 0 else 0.0


def run():
    strats = {sid: Strat(sid) for sid in bot.STRATEGIES}
    stats = {sid: {"n": 0, "w": 0, "l": 0, "gp": 0.0, "gl": 0.0} for sid in bot.STRATEGIES}
    for inst in sorted(fivem):
        df = load_symbol(inst)
        if df is None or len(df) < 600:
            continue
        dff = bot.add_features(df)
        dff.dropna(subset=["ema200", "atr14", "adx14"], inplace=True)
        if len(dff) < 550:
            continue
        train = dff[dff.index < FREEZE]
        test = dff[dff.index >= FREEZE]
        if len(train) < 400 or len(test) < 50:
            print(f"  [skip {inst}] train={len(train)} test={len(test)}", flush=True)
            continue
        thr = {sid: bot.compute_thresholds(train, bot.STRATEGIES[sid]["conditions"])
               for sid in bot.STRATEGIES}
        closes = dff["close"]; idx = list(dff.index); n = len(dff)
        first = idx.index(test.index[0])
        for i in range(first + 1, n):
            L = dff.iloc[i]; Lhi, Llo, Lc = float(L["high"]), float(L["low"]), float(L["close"])
            now = idx[i]
            for sid, st in strats.items():
                tr = st.pos.get(inst)
                if tr is None:
                    continue
                tp, sl = tr["tp"], tr["sl"]
                if Lhi >= tp:
                    pnl = tr["risk"] * bot.STRATEGIES[sid]["rr"]
                elif Llo <= sl:
                    pnl = -tr["risk"]
                else:
                    continue
                st.cap += pnl; st.peak = max(st.peak, st.cap)
                st.max_dd = min(st.max_dd, st.dd)
                st.trades.append(pnl); s = stats[sid]; s["n"] += 1
                if pnl > 0:
                    s["w"] += 1; s["gp"] += pnl
                else:
                    s["l"] += 1; s["gl"] += abs(pnl)
                del st.pos[inst]
            sig_i = i - 1
            if sig_i < 0:
                continue
            sb = dff.iloc[sig_i]
            cp = float(closes.iloc[sig_i - 1]) if sig_i - 1 >= 0 else np.nan
            sig = sb.to_dict(); sig["close_prev"] = cp
            for sid, st in strats.items():
                if inst in st.pos:
                    continue
                if st.paused_until is not None and now < st.paused_until:
                    continue
                if st.dd < -0.15:
                    st.paused_until = now + pd.Timedelta(days=365); continue
                day = now.floor("D")
                if st.day_pnl.get(day, 0.0) < START_CAP * -0.03:
                    st.paused_until = (day + pd.Timedelta(days=1)).floor("D"); continue
                ok, _ = bot.check_conditions(sig, thr[sid], bot.STRATEGIES[sid]["conditions"])
                if not ok:
                    continue
                if not bot.check_entry_gate(sig):
                    continue
                atr = float(sb["atr14"])
                if atr <= 0:
                    continue
                entry = Lc; sl = entry - atr; tp = entry + bot.STRATEGIES[sid]["rr"] * atr
                risk = st.cap * RISK_PCT
                st.pos[inst] = {"entry": entry, "sl": sl, "tp": tp, "risk": risk, "entry_time": now}
    return strats, stats


if __name__ == "__main__":
    t0 = time.time()
    strats, stats = run()
    print("\n" + "=" * 70)
    print("5m NEGATIVE CONTROL — same harness logic, 5m data (research: NO EDGE)")
    print(f"Freeze = {FREEZE.date()} | Test window = post-freeze 5m bars (blind)")
    print("=" * 70)
    for sid in bot.STRATEGIES:
        st = strats[sid]; s = stats[sid]
        pf = (s["gp"] / s["gl"]) if s["gl"] > 0 else (float("inf") if s["gp"] > 0 else 0.0)
        wr = (s["w"] / s["n"]) if s["n"] else 0.0
        verdict = "FAIL (correct)" if pf < 1 else "*** FALSE POSITIVE ***"
        print(f"\n[{bot.STRATEGIES[sid]['label']}]")
        print(f"  trades={s['n']}  win_rate={wr:.1%}  PF={pf:.3f}  "
              f"net=${st.cap-START_CAP:.2f}  maxDD={st.max_dd:.1%}")
        print(f"  -> {verdict}")
    print(f"\n[done in {time.time()-t0:.1f}s]")
