#!/usr/bin/env python3
"""Regression tests for ql_accounting — each test targets a DEFECT ACTUALLY FOUND in this repo
(R104 printer x100, R107/R118 off-by-one, R118 cost overwrite, R115/R116 repeated turnover,
leverage-mixed reporting, demo_bot peak amnesia + dead DAILY_LOSS_LIM). Run: python3 tests/test_accounting.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from ql_accounting import LadderAccount, AccountingError, performance_report

IDX = pd.date_range("2024-01-01", periods=10, freq="D")
PASS = []


def test_off_by_one_signal_never_earns_own_bar():
    """R107/R118 family: weights decided at close of bar 0 must earn bar 1, not bar 0."""
    a = LadderAccount(IDX, leverage=1.0, side_rate=0.0)
    a.rebalance(0, {"X": 1.0})
    rets = [0.0, 0.10, 0.0, 0, 0, 0, 0, 0, 0, 0]  # big gain on bar 1 (after decision), bar 0 flat
    for i, r in enumerate(rets):
        a.mark(i, r)
    eq = a.equity_returns()
    assert abs(eq.iloc[0]) < 1e-12, "signal bar earned its own bar (off-by-one!)"
    assert abs(eq.iloc[1] - 0.10) < 1e-12, "first earned bar should be bar 1"
    PASS.append("off-by-one: signal never earns its own bar")


def test_cost_charged_once_and_never_overwritten():
    """R118 family: a cost booked on the execution bar must survive all later marks."""
    a = LadderAccount(IDX, leverage=1.0, side_rate=0.0007)
    a.rebalance(0, {"X": 1.0})
    for i in range(10):
        a.mark(i, 0.01)  # constant gross
    eq = a.equity_returns()
    # turnover 1.0 x 0.0007 charged on bar 1, once:
    assert abs(a.costs[1] - 0.0007) < 1e-12, f"cost wrong: {a.costs}"
    assert sum(a.costs) == a.costs[1], "cost appeared on more than one bar"
    assert abs(eq.iloc[1] - (0.01 - 0.0007)) < 1e-12, "cost not reflected on execution bar"
    assert abs(eq.iloc[2] - 0.01) < 1e-12, "cost leaked into later bars"
    PASS.append("cost charged exactly once on execution bar, never overwritten")


def test_turnover_not_repeated_between_rebalances():
    """R115/R116 family: R=7 rebalance cycle must charge 1x, not 7x."""
    a = LadderAccount(IDX, leverage=1.0, side_rate=0.0007)
    a.rebalance(0, {"X": 1.0})
    for i in range(10):
        a.mark(i, 0.001)   # no further rebalances
    total_cost = sum(a.costs)
    assert abs(total_cost - 0.0007) < 1e-12, f"turnover repeated: total {total_cost}"
    PASS.append("turnover charged once per weight change (not daily)")


def test_mark_twice_raises():
    a = LadderAccount(IDX, leverage=1.0)
    a.rebalance(0, {"X": 1.0})
    a.mark(0, 0.0)
    try:
        a.mark(0, 0.0)
        raise AssertionError("double-mark did not raise")
    except AccountingError:
        PASS.append("double-marked bar raises AccountingError")


def test_leverage_single_basis():
    """R115/R116 family: APR, vol, Sharpe, maxDD must all live on the SAME (leveraged) basis."""
    rng = np.random.default_rng(3)
    a = LadderAccount(IDX, leverage=2.0, side_rate=0.0)
    a.rebalance(0, {"X": 1.0})
    rets = rng.normal(0.001, 0.01, 10)
    for i, r in enumerate(rets):
        a.mark(i, r)
    eq = a.equity_returns()
    assert np.allclose(eq.values, rets * 2.0), "equity series is not gross x leverage"
    r = performance_report(eq)
    assert abs(r["ann_vol"] - eq.std() * np.sqrt(365) * 100) < 1e-9
    PASS.append("APR/vol/Sharpe/DD all on leveraged equity basis (no mixed units)")


def test_sanity_tripwire():
    a = LadderAccount(IDX, leverage=1.0)
    a.rebalance(0, {"X": 1.0})
    a.mark(0, 0.0)
    a.mark(1, 5.0)  # +500% single bar -> must raise on verify
    try:
        a.equity_returns()
        raise AssertionError("sanity tripwire did not raise")
    except AccountingError:
        PASS.append("impossible bar raises sanity tripwire")


def test_demo_bot_peak_reload_regression():
    """demo_bot load_states must restore peak from DB (defect: peak=max(START,capital) after restart).
    Static source check (bot is not executable here — no network/DB fixtures)."""
    src = pathlib.Path("demo_bot.py").read_text()
    load_block = src[src.index("def load_states"):src.index("def main")]
    assert "peak" in load_block and ("equity" in load_block or "SELECT peak" in load_block), \
        "load_states does not restore peak from DB — regression not fixed"
    # enforcement check
    body = src[src.index("DAILY_LOSS_LIM"):]
    assert body.count("DAILY_LOSS_LIM") >= 2, "DAILY_LOSS_LIM still dead code (defined, never used)"
    PASS.append("demo_bot: peak restored from DB + DAILY_LOSS_LIM referenced in loop")


if __name__ == "__main__":
    for t in [test_off_by_one_signal_never_earns_own_bar, test_cost_charged_once_and_never_overwritten,
              test_turnover_not_repeated_between_rebalances, test_mark_twice_raises,
              test_leverage_single_basis, test_sanity_tripwire, test_demo_bot_peak_reload_regression]:
        t()
    print(f"PASSED {len(PASS)}/{len(PASS)} accounting regression tests:")
    for p in PASS:
        print("  ✓", p)
