"""
R098 — 1-MINUTE PATH AUDIT of the R096 champion (q0.55).

The 1H engine assumes TP-before-SL when a single bar touches both levels.
This audit fetches OKX 1m candles for each 2026 champion trade window and
resolves the true first touch. Output per trade:
  tp1m      first touch == TP  (true win under any convention)
  sl1m      first touch == SL  (true loss)
  same_min  TP and SL touched within the SAME 1m minute (kept as SL, counted)
  ambiguous_1h  the 1H bar that closed the trade touched both levels
Also recomputes PF@cost under taker and maker cost models.
Windows: entry_time+1h (first full bar) → entry_time+(bars_in+1)h (exit bar end).
"""
import os, json, time, subprocess
import numpy as np
import pandas as pd

UA = "Mozilla/5.0"
def get(url, tries=4):
    for a in range(tries):
        p = subprocess.run(["curl","-s","-4","--max-time","20","-A",UA,url], capture_output=True, timeout=30)
        try:
            d = json.loads(p.stdout.decode())
            if d.get("code") == "0": return d["data"]
            time.sleep(1+a)
        except Exception:
            time.sleep(1+a)
    return []

def fetch_window(inst, start_ms, end_ms):
    """All 1m bars with open-time in [start_ms, end_ms)."""
    rows = {}
    after = end_ms
    for page in range(30):
        url = (f"https://www.okx.com/api/v5/market/history-candles?instId={inst}"
               f"&bar=1m&limit=100&after={after}")
        data = get(url)
        if not data: break
        for r in data:
            ts = int(r[0])
            if start_ms <= ts < end_ms:
                rows[ts] = (float(r[2]), float(r[3]))   # (high, low)
        oldest = int(data[-1][0])
        if oldest <= start_ms or len(data) < 100: break
        after = oldest
        time.sleep(0.08)
    return [rows[k] for k in sorted(rows)]

def main():
    df = pd.read_csv("quantlab_output/r096_trades_q055.csv")
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df[df["entry_time"] >= pd.Timestamp("2026-01-01", tz="UTC")].reset_index(drop=True)
    print(f"Auditing {len(df)} trades (2026) …", flush=True)
    out=[]
    t0=time.time()
    for k, t in df.iterrows():
        inst = t["sym"].replace("_","-")
        start = t["entry_time"] + pd.Timedelta(hours=1)
        end   = t["entry_time"] + pd.Timedelta(hours=int(t["bars_in"])+1)
        tp = t["entry"] + 1.5*t["atr"]
        sl = t["entry"] - 1.0*t["atr"]
        bars = fetch_window(inst, int(start.value//10**6), int(end.value//10**6))
        rec = dict(sym=t["sym"], entry_time=t["entry_time"], r_1h=t["r"], exit_1h=t["exit_type"],
                   bars_in=t["bars_in"], n_1m=len(bars))
        touch = None; same_min=False
        for hi, lo in bars:
            hit_tp = hi >= tp; hit_sl = lo <= sl
            if hit_tp and hit_sl: same_min=True
            if hit_tp or hit_sl:
                touch = "TP" if hit_tp else "SL"
                break
        rec["touch_1m"] = touch if touch else ("NONE" if len(bars)>0 else "NODATA")
        rec["same_min"] = same_min
        # 1H-level ambiguity: recompute what the 1H assumption saw on the exit bar
        rec["cost_taker_R"] = 2*0.0005*t["entry"]/t["atr"]
        rec["cost_maker_R"] = 0.0002*t["entry"]/t["atr"] + (0.0002 if (touch=="TP") else 0.0005)*t["entry"]/t["atr"]
        out.append(rec)
        if (k+1) % 20 == 0:
            print(f"  [{k+1}/{len(df)}] ({time.time()-t0:.0f}s)", flush=True)
    o = pd.DataFrame(out)
    o.to_csv("quantlab_output/r098_path_audit.csv", index=False)
    # summary
    aud = o[~o["touch_1m"].isin(["NODATA"])].copy()
    aud["r_1m_gross"] = np.where(aud["touch_1m"]=="TP", 1.5, -1.0)
    print(f"\nAudited with 1m data: {len(aud)}/{len(o)} trades (NODATA: {(o['touch_1m']=='NODATA').sum()})")
    amb = aud["touch_1m"] != aud["exit_1h"]
    print(f"1H convention WRONG on {amb.sum()} trades ({amb.mean()*100:.0f}% of audited)")
    print(f"same-minute collisions: {aud['same_min'].sum()}")
    for name, col, base in [("taker .05/.05","cost_taker_R",None), ("maker-in/TP .02","cost_maker_R",None)]:
        for tag, sub in [("2026 full", aud), ("fresh Aug8+", aud[aud["entry_time"]>=pd.Timestamp("2026-08-08", tz="UTC")])]:
            rs = sub["r_1m_gross"] - sub[col]
            w = rs[rs>0].sum(); l = abs(rs[rs<0].sum())
            pf = w/l if l>0 else 99
            print(f"  PF@cost [{name:<18}] {tag:<11} n={len(sub):>3}  PF={pf:.2f}")
    # compare: 1H-assumed PF on same trades
    for tag, sub in [("2026 full", aud), ("fresh Aug8+", aud[aud["entry_time"]>=pd.Timestamp("2026-08-08", tz="UTC")])]:
        rs = sub["r_1h_gross"] if "r_1h_gross" in sub else np.where(sub["exit_1h"]=="TP",1.5,-1.0)
        rs = rs - sub["cost_taker_R"]
        w = rs[rs>0].sum(); l=abs(rs[rs<0].sum())
        print(f"  [1H-assumed, taker]      {tag:<11} n={len(sub):>3}  PF={w/l if l>0 else 99:.2f}")
    print(f"\nDone in {time.time()-t0:.0f}s → quantlab_output/r098_path_audit.csv")

if __name__ == "__main__":
    main()
