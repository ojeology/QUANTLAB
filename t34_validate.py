"""
T34 stage 1 — VALIDATION gate.

Part A (anchor='exit', VERBATIM branch behaviour): replicate T25d (trend,
50-sym broad universe) exactly as t25_full_universe_3yr.py computes it — that
script records each trade's 'entry_time' at the EXIT bar (backtest_donchian
appends df.index[i] after the position closed), so mldf features + year folds
are exit-anchored. Acceptance = branch log within noise:
  2024 PF 1.570 n2788 | 2025 1.393 n2567 | 2026 1.392 n1403 | FULL 1.472
  $100 -> $199.70

Part B (anchor='entry', IMPLEMENTABLE live): same champion code but mldf
features at the true ENTRY bar. The difference between A and B is the
post-entry information the branch's exit-anchor gate silently used.
"""
import os, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings("ignore")
from t34_lib import load_feats, trend_raw, trend_champion, SAVE_DIR, FEE

t0 = time.time()
feats, above20, breadth, breadth_pct = load_feats()
print(f"[load] usable: {len(feats)}  ({time.time()-t0:.0f}s)", flush=True)

raw = trend_raw(feats)
print(f"[signals] raw trend trades (2023-2026): {len(raw)} ({time.time()-t0:.0f}s)", flush=True)

def report(tag, champ):
    print(f"\n---- {tag} ----", flush=True)
    for Y in [2024, 2025, 2026]:
        yt = [t for t in champ if t["entry_time"].year == Y]
        rs = [t["r"] - 2 * FEE for t in yt]
        wins = sum(1 for r in rs if r > 0) / len(rs)
        pf = sum(r for r in rs if r > 0) / max(1e-9, -sum(r for r in rs if r < 0))
        msum = defaultdict(float)
        for t in yt:
            msum[(t["entry_time"].year, t["entry_time"].month)] += t["r"] - 2 * FEE
        print(f"[{Y}] n={len(yt)} win={wins:.0%} PF@c={pf:.3f} "
              f"prof-months={sum(1 for v in msum.values() if v>0)}/{len(msum)}", flush=True)
    rs = [t["r"] - 2 * FEE for t in champ]
    pf = sum(r for r in rs if r > 0) / max(1e-9, -sum(r for r in rs if r < 0))
    print(f"[FULL] n={len(champ)} PF@c={pf:.3f}", flush=True)
    eq = 100.0
    for Y in [2024, 2025, 2026]:
        yt = sorted([t for t in champ if t["entry_time"].year == Y],
                    key=lambda x: x["entry_time"])
        st = eq
        for t in yt:
            eq += 2.0 * (t["r"] - 2 * FEE)          # their fixed-$2 sim
        print(f"[{Y}] $100 -> ${eq:,.2f} ({(eq/st - 1):+.1%})", flush=True)
    print(f"[FULL $] $100 -> ${eq:,.2f} ({(eq/100 - 1):+.1%})", flush=True)

report("A) BRANCH-VERBATIM (exit-anchor RF filter; reproduces t25_full_universe_3yr.log)",
       trend_champion(raw, feats, breadth, breadth_pct, anchor="exit"))
report("B) IMPLEMENTABLE (entry-anchor RF filter; same code, gate at true entry)",
       trend_champion(raw, feats, breadth, breadth_pct, anchor="entry"))

# raw trend for context (no RF gate at all)
report("C) RAW trend signals, NO champion filter (context)",
       raw)
print(f"\n[done in {time.time()-t0:.0f}s]", flush=True)
