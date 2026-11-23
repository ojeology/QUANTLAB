#!/usr/bin/env python3
# R135 PAPER TRACKER — logs the locked w5/20-coin engine's signals as paper trades.
# Seed protocol (disclosed): the engine is deterministic on stored dailies; entries since the
# last session (2026-11-05) are seeded retro-simultaneously and tracked forward from here.
import numpy as np, pandas as pd, glob, os
from datetime import datetime, timezone

LAST_SESSION = pd.Timestamp("2026-11-05")
LOG = "quantlab_output/r135_paper_log.csv"
w = 5

rows, open_pos = [], []
for f in sorted(glob.glob("quantlab_cache/daily/*_1D.csv")):
    sym = os.path.basename(f).replace("_1D.csv", "")
    df = pd.read_csv(f, parse_dates=["ts"]).set_index("ts").sort_index()
    c, lo, hi = df.c.values, df.l.values, df.h.values
    n = len(c)
    atr = pd.Series(np.maximum(hi - lo, np.abs(hi - np.roll(c, 1)))[1:], index=df.index[1:]).rolling(14).mean()
    pl = np.zeros(n, bool); ph = np.zeros(n, bool)
    for i in range(w, n - w):
        if lo[i] == lo[i-w:i+w+1].min(): pl[i] = True
        if hi[i] == hi[i-w:i+w+1].max(): ph[i] = True
    pos = None
    for t in range(120, n):
        if pos is not None:
            d_, ep, ei, ae, lp = pos
            px = c[t]
            if (px >= lp if d_ < 0 else px <= lp) or (t - ei) >= 10:
                r = (ep - px)/ae if d_ < 0 else (px - ep)/ae
                if df.index[ei] >= LAST_SESSION:
                    rows.append(dict(sym=sym, entry_ts=df.index[ei], dir=d_, status="EXITED",
                                     exit_ts=df.index[t], r=r - 2*0.0005*ep/ae))
                pos = None
            continue
        idxs = [i for i in range(max(0, t-120), t-w+1) if pl[i]]
        if len(idxs) >= 2:
            i2, i1 = idxs[-1], idxs[-2]
            if lo[i2] > lo[i1]:
                s_ = (lo[i2]-lo[i1])/(i2-i1)
                lt = lo[i2] + s_*(t-i2); lp_ = lo[i2] + s_*(t-1-i2)
                if c[t] < lt and c[t-1] >= lp_:
                    a = atr.iloc[t-1]
                    if not np.isnan(a) and a > 0: pos = (-1, c[t], t, a, lt); continue
        idxs = [i for i in range(max(0, t-120), t-w+1) if ph[i]]
        if len(idxs) >= 2:
            i2, i1 = idxs[-1], idxs[-2]
            if hi[i2] < hi[i1]:
                s_ = (hi[i2]-hi[i1])/(i2-i1)
                lt = hi[i2] + s_*(t-i2); lp_ = hi[i2] + s_*(t-1-i2)
                if c[t] > lt and c[t-1] <= lp_:
                    a = atr.iloc[t-1]
                    if not np.isnan(a) and a > 0: pos = (1, c[t], t, a, lt)
    if pos is not None and df.index[pos[2]] >= LAST_SESSION:
        open_pos.append(dict(sym=sym, entry_ts=df.index[pos[2]], dir=pos[0], status="OPEN",
                             exit_ts=None, r=np.nan))

now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
new = pd.DataFrame(rows + open_pos)
new["logged_at"] = now
if os.path.exists(LOG):
    old = pd.read_csv(LOG)
    old = old[~old.entry_ts.isin(new.entry_ts) | (old.status != new.status.reindex(old.index, fill_value=""))]
    out = pd.concat([old, new], ignore_index=True)
else:
    out = new
out = out.drop_duplicates(subset=["sym", "entry_ts", "status"], keep="last")
out.to_csv(LOG, index=False)
print(f"paper log: {len(out)} rows (logged_at {now})")
if len(new):
    print(new[["sym", "entry_ts", "dir", "status", "exit_ts", "r"]].to_string(index=False))
else:
    print("no new R135 signals since last session")
