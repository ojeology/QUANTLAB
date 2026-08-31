"""
BLIND OUT-OF-SAMPLE TEST of the FROZEN QuantLab strategies (Family A + Family C)
exactly as deployed in demo_bot.py, on 1H OKX data the strategy has NEVER seen.

Method (true walk-forward, zero parameter tuning on the test set):
  - Load each symbol's full cached 1H history (ends ~2026-07-30).
  - Top up with FRESH 1H bars from OKX (covers the gap + post-freeze window).
  - FREEZE = 2026-08-08 00:00 UTC (the strategy-freeze date in MEMORY.md).
  - Threshold/train window = all bars BEFORE the freeze (what the strategy knew).
  - Blind test window     = bars ON/AFTER the freeze (2026-08-08 -> now) = fresh.
  - Per (strategy, symbol): compute frozen quantile thresholds on the TRAIN
    window, then replay the TEST window bar-by-bar with the bot's EXACT logic:
        signal on bar i-1 (last closed), entry at bar i close,
        SL = entry - ATR(i-1),  TP = entry + RR*ATR(i-1),
        one position per symbol, fixed 1% risk/trade, TP/SL checked on next bars.
  - Aggregate PF, win rate, net PnL, max DD, profitable days.

ALL strategy logic (conditions, quantile thresholds, RR, entry gate) is imported
verbatim from demo_bot (the deployed environment). Nothing is re-tuned.
"""
import warnings, logging, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import demo_bot as bot
logging.getLogger("demo_bot").setLevel(logging.ERROR)

FREEZE     = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
START_CAP  = 10000.0
RISK_PCT   = 0.01
FRESH_BARS = 800

bot._CACHE_MAP = bot._build_cache_map()
universe = sorted(bot._CACHE_MAP.keys())
print(f"[setup] universe = {len(universe)} cached symbols", flush=True)


def load_symbol(inst_id):
    path = bot._CACHE_MAP[inst_id]
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ("datetime", "ts"):
            if c in df.columns:
                df = df.set_index(c)
                break
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index().drop_duplicates()
    keep = [c for c in ["open", "high", "low", "close", "vol"] if c in df.columns]
    df = df[keep].astype(float)
    try:
        df_new = bot.fetch_candles(inst_id, n_bars=FRESH_BARS)
    except Exception:
        df_new = None
    if df_new is not None and len(df_new):
        df_new = df_new.astype(float)
        df = pd.concat([df, df_new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["open", "high", "low", "close", "vol"]]


class Strat:
    def __init__(self, sid):
        self.sid = sid
        self.cap = START_CAP
        self.peak = START_CAP
        self.trades = []
        self.pos = {}
        self.paused_until = None
        self.day_pnl = {}
        self.max_dd = 0.0

    @property
    def dd(self):
        return (self.cap - self.peak) / self.peak if self.peak > 0 else 0.0


def run():
    strats = {sid: Strat(sid) for sid in bot.STRATEGIES}
    stats = {sid: {"n": 0, "w": 0, "l": 0, "gp": 0.0, "gl": 0.0} for sid in bot.STRATEGIES}
    done = 0
    for inst_id in universe:
        df = load_symbol(inst_id)
        if df is None or len(df) < 600:
            continue
        dff = bot.add_features(df)
        dff.dropna(subset=["ema200", "atr14", "adx14"], inplace=True)
        if len(dff) < 550:
            continue
        train = dff[dff.index < FREEZE]
        test = dff[dff.index >= FREEZE]
        if len(train) < 400 or len(test) < 50:
            continue
        thr = {sid: bot.compute_thresholds(train, bot.STRATEGIES[sid]["conditions"])
               for sid in bot.STRATEGIES}
        closes = dff["close"]
        idx = list(dff.index)
        n = len(dff)
        first_test_pos = idx.index(test.index[0])
        for i in range(first_test_pos + 1, n):
            L = dff.iloc[i]
            Lhi, Llo, Lclose = float(L["high"]), float(L["low"]), float(L["close"])
            nowts = idx[i]
            # 1) manage open positions on bar i
            for sid, st in strats.items():
                tr = st.pos.get(inst_id)
                if tr is None:
                    continue
                tp, sl = tr["tp"], tr["sl"]
                if Lhi >= tp:
                    pnl = tr["risk"] * bot.STRATEGIES[sid]["rr"]; et = "TP"
                elif Llo <= sl:
                    pnl = -tr["risk"]; et = "SL"
                else:
                    continue
                st.cap += pnl
                st.peak = max(st.peak, st.cap)
                st.max_dd = min(st.max_dd, st.dd)
                d1 = nowts.floor("D")
                st.day_pnl[d1] = st.day_pnl.get(d1, 0.0) + pnl
                st.trades.append(pnl)
                s = stats[sid]
                s["n"] += 1
                if pnl > 0:
                    s["w"] += 1; s["gp"] += pnl
                else:
                    s["l"] += 1; s["gl"] += abs(pnl)
                del st.pos[inst_id]
            # 2) risk gates + new signals (signal on bar i-1, entry at bar i close)
            sig_i = i - 1
            if sig_i < 0:
                continue
            sig_bar = dff.iloc[sig_i]
            close_prev = float(closes.iloc[sig_i - 1]) if sig_i - 1 >= 0 else np.nan
            sig = sig_bar.to_dict()
            sig["close_prev"] = close_prev
            for sid, st in strats.items():
                if inst_id in st.pos:
                    continue
                if st.paused_until is not None and nowts < st.paused_until:
                    continue
                if st.dd < -0.15:
                    st.paused_until = nowts + pd.Timedelta(days=365)
                    continue
                day = nowts.floor("D")
                if st.day_pnl.get(day, 0.0) < START_CAP * -0.03:
                    st.paused_until = (day + pd.Timedelta(days=1)).floor("D")
                    continue
                env_ok, _ = bot.check_conditions(sig, thr[sid], bot.STRATEGIES[sid]["conditions"])
                if not env_ok:
                    continue
                if not bot.check_entry_gate(sig):
                    continue
                atr = float(sig_bar["atr14"])
                if atr <= 0:
                    continue
                entry = Lclose
                sl = entry - atr
                tp = entry + bot.STRATEGIES[sid]["rr"] * atr
                risk = st.cap * RISK_PCT
                st.pos[inst_id] = {"entry": entry, "sl": sl, "tp": tp,
                                   "risk": risk, "entry_time": nowts}
        done += 1
        if done % 10 == 0:
            print(f"[progress] {done}/{len(universe)} symbols processed", flush=True)
    return strats, stats


if __name__ == "__main__":
    t0 = time.time()
    strats, stats = run()
    print("\n" + "=" * 70)
    print("BLIND OUT-OF-SAMPLE TEST — frozen strategies on 2026 (this year)")
    print(f"Train = pre-2026 (2024-2025) | Test = 2026 (this year), model blind to 2026")
    print("=" * 70)
    for sid in bot.STRATEGIES:
        st = strats[sid]
        s = stats[sid]
        pf = (s["gp"] / s["gl"]) if s["gl"] > 0 else (float("inf") if s["gp"] > 0 else 0.0)
        wr = (s["w"] / s["n"]) if s["n"] else 0.0
        print(f"\n[{bot.STRATEGIES[sid]['label']}]  {bot.STRATEGIES[sid]['description']}")
        print(f"  trades={s['n']}  wins={s['w']}  losses={s['l']}  win_rate={wr:.1%}")
        print(f"  gross_profit=${s['gp']:.2f}  gross_loss=${s['gl']:.2f}  PF={pf:.3f}")
        print(f"  start=${START_CAP:.2f}  equity=${st.cap:.2f}  "
              f"net_pnl=${st.cap - START_CAP:.2f}  maxDD={st.max_dd:.1%}")
    print(f"\n[done in {time.time() - t0:.1f}s]")
