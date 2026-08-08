# QuantLab
### Quantitative Strategy Research Laboratory

---

## ═══════════════════════════════════════════════════════════════
## 🏁 QUANTLAB — CLOSED (2026-08-08)
## 🚀 NEW HUNT BEGINS — FOREX
## ═══════════════════════════════════════════════════════════════

**After 23 research runs (R073 → R095), the crypto hunt is closed.**
- **5-minute crypto: NO EDGE** — proven 7 independent ways (rules, bank logic, new
  indicators, combos, advanced ML). The 5m bar is too small vs retail costs.
- **1-Hour crypto: VALIDATED EDGE FOUND** — **SVM q0.75 on 73 symbols**:
  ~10.4 trades/month, PF 1.94, ~70% profitable months, cost-surviving,
  holdout-validated. Saved for the end-of-year 2026 re-run.

**The Forex hunt now begins** — same rigorous pipeline (walk-forward, holdout,
cost gates, causal audit), applied to a market with 15-20 years of free data,
better cost-per-move, and real session structure.

---

> **Good quantitative research attempts to disprove ideas before trusting them.**

---

## Overview

QuantLab was a quantitative research project focused on identifying robust trading strategies through evidence-driven experimentation rather than curve fitting.

The project documented dozens of research iterations covering strategy discovery, validation, portfolio construction, and paper trading.

Unlike many trading repositories, this project documents both successful and failed experiments, emphasizing reproducibility and scientific methodology.

---

# Research Goals

- Discover statistically robust trading strategies
- Minimize overfitting through walk-forward testing
- Validate results using multiple independent methods
- Compare competing strategy families
- Measure robustness instead of optimizing for one backtest
- Build a framework suitable for continuous research

---

# Validation Framework

Every strategy is evaluated using multiple layers of validation including:

- Walk-Forward Optimization
- Out-of-Sample Testing
- Bootstrap Confidence Intervals
- Monte Carlo Simulation
- Leave-One-Out Symbol Validation
- Leave-One-Out Fold Validation
- Drawdown Analysis
- Risk Metrics
- Monthly Stability Analysis
- Regime Analysis
- Portfolio Validation
- Reward/Risk Sensitivity Testing

---

# Research Highlights

- 70+ structured research iterations
- Multiple independent strategy families discovered
- Thousands of combinations evaluated
- Automated strategy ranking engine
- Bootstrap robustness testing
- Monte Carlo simulations
- Portfolio construction framework
- Statistical stress testing
- Paper trading engine
- Automated reporting pipeline

---

# Strategy Families

## Family A

High-conviction, highly selective strategy.

Characteristics:

- Very high profit factor
- Low drawdown
- Low signal frequency
- Designed for quality over quantity

---

## Family B

Experimental session-based strategy.

Current status:

- Promising concept
- Requires additional historical data
- Under continued investigation

---

## Family C

Higher-frequency strategy.

Characteristics:

- Large sample size
- Strong statistical validation
- More frequent opportunities
- Suitable for continuous paper trading

---

# Project Structure

```
research/
    Research iterations

engine/
    Research framework

strategies/
    Strategy implementations

demo_bot/
    Paper trading engine

docs/
    Documentation

quantlab_cache/
    Historical market data

quantlab_output/
    Generated reports and charts
```

---

# Research Methodology

The philosophy behind this repository is simple:

> Good quantitative research attempts to disprove ideas before trusting them.

Strategies are not accepted because they produce attractive backtests.

Instead they must survive multiple independent validation procedures before being considered for deployment.

---

# Lessons Learned

This project reinforced several important principles:

- Small sample sizes can be misleading.
- High profit factors alone are insufficient.
- Robustness matters more than optimization.
- Different exit models can completely change results.
- Scientific validation is an iterative process.
- Failed experiments often provide the most valuable insights.

---

# Current Status

Current work focuses on:

- Continuous paper trading
- Long-term live validation
- Timestamp reconstruction improvements
- Alternative exit models
- Portfolio research
- Additional market data

---

# Technologies

- Python
- Pandas
- NumPy
- Parquet
- SQLite
- APScheduler
- OKX Market Data
- Statistical Analysis
- Bootstrap Methods
- Monte Carlo Simulation

---

# Disclaimer

This repository is intended for educational and quantitative research purposes.

Nothing contained here should be interpreted as financial advice or a recommendation to trade any financial instrument.

Past performance does not guarantee future results.

---

# Author

**Ojeology**

Independent Quantitative Researcher

GitHub: https://github.com/ojeology

---

*"The goal isn't to find a strategy that looks good. The goal is to find one that survives every attempt to prove it wrong."*
