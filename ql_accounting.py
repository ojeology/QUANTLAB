#!/usr/bin/env python3
"""ql_accounting — THE single accounting/execution convention for all QUANTLAB backtests.
Established 2026-10-16 (audit directive, Phase 3). Every new experiment MUST use this module
or document a deviation. Historical scripts are preserved untouched; superseded results are
marked in docs/QUANTLAB_RESEARCH_AUDIT.md.

CONVENTIONS (locked):
1. SIGNAL TIMING      : a weight/position decided from data up to close of bar t is effective
                        for bar t+1 (earns t+1's return). It NEVER earns bar t. (off-by-one law)
2. EXECUTION BAR      : a weight change decided at close t is executed at the open of bar t+1
                        (close-to-close approximation); its COST is charged exactly once, on
                        bar t+1, and is never overwritten by later marks.
3. COSTS              : cost = turnover_L1 * side_rate, charged once per weight change.
                        side_rate must include fee + spread + slippage (default 0.05%+0.02%).
4. FUNDING            : perp funding cash flow = -weight * funding_rate(bar) applied daily to
                        held positions (uniform -w.f law; positive rate = longs pay).
5. LEVERAGE/EQUITY    : capital is explicit. gross_notional = sum|w|; leverage = gross/capital.
                        equity_return(bar) = portfolio_return(bar) * leverage.
                        APR, volatility, Sharpe, Sortino, maxDD are ALL computed on the SAME
                        leveraged equity series. Mixed bases are forbidden.
6. INTRABAR TP/SL     : OHLC data cannot resolve which level was hit first inside a bar.
                        Default convention: CONSERVATIVE (SL first). Any TP-first assumption
                        must be labeled OPTIMISTIC and bounded by the SL-first variant
                        (see EXIT_MODEL_AUDIT.md M2 vs M3).
7. HOLDOUT DISCIPLINE : a price/return window used for evaluation in any report is recorded in
                        docs/EXPOSURE_LEDGER.md. Windows examined >=2 times are CONTAMINATED
                        for new-discovery purposes (usable only for confirmatory re-tests).

The LadderAccount class enforces 1-5 and raises on violations via verify().
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

SIDE_RATE_DEFAULT = 0.0007  # 0.05% taker + 0.02% slippage, per side
MAX_ABS_DAILY = 0.60        # sanity tripwire: no single-bar portfolio return beyond +-60%


class AccountingError(AssertionError):
    pass


@dataclass
class LadderAccount:
    """Event-driven ladder accounting with enforced invariants.

    Usage:
        acct = LadderAccount(index, leverage=2.0)
        acct.rebalance(i_signal_bar, weights_dict)   # decided at close of bar i_signal_bar
        for i in range(len(index)):
            acct.mark(i, portfolio_gross_return_i, funding_drag_i)   # per unit gross
        res = acct.report()
    """
    index: pd.DatetimeIndex
    leverage: float = 1.0
    side_rate: float = SIDE_RATE_DEFAULT
    w: dict = field(default_factory=dict)
    pending_w: dict = None
    pending_signal_bar: int = None
    gross: list = field(default_factory=list)      # per-unit-gross portfolio return, pre-cost
    costs: list = field(default_factory=list)      # charged once, on execution bar
    funding: list = field(default_factory=list)
    written: list = field(default_factory=list)    # bars that received a mark
    n_rebalances: int = 0

    def __post_init__(self):
        self.gross = [0.0] * len(self.index)
        self.costs = [0.0] * len(self.index)
        self.funding = [0.0] * len(self.index)
        self.written = [False] * len(self.index)

    def rebalance(self, signal_bar: int, weights: dict):
        """Weights decided at close of `signal_bar`. Cost charged on signal_bar+1 (exec bar)."""
        if not (0 <= signal_bar < len(self.index)):
            raise AccountingError(f"signal bar {signal_bar} out of range")
        self.pending_w = dict(weights)
        self.pending_signal_bar = signal_bar

    def _apply_pending(self, i: int):
        if self.pending_w is None or i != self.pending_signal_bar + 1:
            return
        keys = set(self.pending_w) | set(self.w)
        turnover = sum(abs(self.pending_w.get(k, 0.0) - self.w.get(k, 0.0)) for k in keys)
        self.costs[i] += turnover * self.side_rate      # ONCE — this slot is never reassigned later
        self.w = dict(self.pending_w)
        self.pending_w = None
        self.n_rebalances += 1

    def effective_weights(self, i: int) -> dict:
        """Weights effective for bar i (applies any pending rebalance whose exec bar is i).
        CALLERS MUST compute bar-i P&L from THIS, never from a local position variable —
        using a post-decision local position re-creates the off-by-one (R104/R107/R118/R119)."""
        self._apply_pending(i)
        return dict(self.w)

    def mark(self, i: int, gross_ret: float, funding_drag: float = 0.0):
        """Record bar i. Applies any pending rebalance whose execution bar is i, first."""
        if self.written[i]:
            raise AccountingError(f"bar {i} marked twice — every bar must be written exactly once")
        self._apply_pending(i)
        self.gross[i] = gross_ret
        self.funding[i] = funding_drag
        self.written[i] = True

    def equity_returns(self):
        """Leveraged equity return series = (gross - costs - funding) * leverage. Single basis."""
        self.verify()
        net = [(g - c - f) * self.leverage for g, c, f in zip(self.gross, self.costs, self.funding)]
        return pd.Series(net, index=self.index)

    def verify(self):
        unwritten = [i for i, x in enumerate(self.written) if not x]
        if unwritten:
            raise AccountingError(f"{len(unwritten)} bars never marked (first: {unwritten[0]})")
        for i, g in enumerate(self.gross):
            if abs(g) > MAX_ABS_DAILY:
                raise AccountingError(f"bar {i}: gross return {g:.2%} beyond sanity tripwire ±{MAX_ABS_DAILY:.0%}")
        if self.pending_w is not None:
            raise AccountingError("unapplied pending rebalance at end of run")
        return True

    def report(self, freq="W", bench=None):
        return performance_report(self.equity_returns(), freq=freq, bench=bench)


def performance_report(eq: pd.Series, freq="W", bench=None):
    """Single standardized metric block (Phase 3.4). eq = LEVERAGED daily equity returns.
    Annualization formula (stated): CAGR = (prod(1+r))^(365/n_days) - 1 on the leveraged series."""
    eq = eq.dropna()
    gross_eq = (1 + eq).cumprod()
    years = len(eq) / 365.0
    ann = (gross_eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1) * 100
    vol = eq.std() * np.sqrt(365) * 100
    sharpe = eq.mean() / eq.std() * np.sqrt(365) if eq.std() > 0 else 0.0
    downside = eq[eq < 0]
    sortino = (eq.mean() / downside.std() * np.sqrt(365)) if len(downside) > 1 and downside.std() > 0 else float("nan")
    peak = gross_eq.cummax()
    dd = gross_eq / peak - 1
    wk = eq.resample(freq).sum() * 100 if freq else None
    out = dict(net_total=(gross_eq.iloc[-1] - 1) * 100, ann_cagr=ann, ann_vol=vol,
               sharpe=sharpe, sortino=sortino, maxDD=dd.min() * 100,
               time_underwater_pct=(dd < -1e-9).mean() * 100)
    if wk is not None:
        out.update(posW=(wk > 0).mean() * 100, worstW=wk.min(),
                   yearly={int(y): round(v * 100, 1) for y, v in eq.groupby(eq.index.year).sum().items()})
    if bench is not None:
        b = bench.reindex(eq.index).dropna()
        if len(b):
            out["bench_corr"] = float(eq.corr(b))
    return out


def fmt_report(tag, r):
    yr = r.get("yearly", {})
    return (f"{tag:30s}: CAGR {r['ann_cagr']:+6.1f}%  vol {r['ann_vol']:5.1f}%  Sharpe {r['sharpe']:+5.2f}  "
            f"maxDD {r['maxDD']:+6.1f}%  worstW {r.get('worstW', float('nan')):+6.2f}%  posW {r.get('posW', float('nan')):5.1f}%  "
            f"| Y " + "/".join(f"{yr.get(y, 0):+.0f}" for y in (2024, 2025, 2026)))
