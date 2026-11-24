> **SUPERSEDED by `article_realmarketapi.md`** (submission version, trimmed to RealMarketAPI's 2,500-word guideline, metadata block added, charts generated). Preserved per audit-preserve rule.

# The Hardest Part of Algorithmic Trading Is Proving Your Backtest Wrong

*What 137 numbered experiments across crypto, forex and synthetic markets taught me about execution costs, statistical robustness, lookahead bias — and the difference between a profitable backtest and a dependable trading system.*

**By Ojeology | Independent Quantitative Researcher**

A trading strategy can have a profit factor above 3 and still be a poor candidate for live trading.

It can post an impressive win rate because its simulator does not represent how the trading bot will actually enter and exit positions. It can show a profitable result because information from the future has leaked into the decision process. It can survive a parameter sweep and still fail the moment it is judged year by year.

I ran into these problems repeatedly while building QuantLab, an independent quantitative research project that tests systematic trading strategies under locked, pre-declared conditions: before any experiment runs, its rules, parameters, costs and pass/fail gates are written down and committed; after it runs, every claim has to trace back to committed artifacts — the raw trade book, the engine code, the dated report.

The project began with a straightforward objective: find trading rules that could survive historical testing and eventually operate under realistic execution conditions. Over time the research grew across crypto perpetuals, forex, synthetic (Deriv-style) markets, machine-learning filters, trend-following and mean-reversion systems — more than a hundred numbered experiments in total.

The most important findings were not the profitable ones.

Some of the strongest-looking results were later invalidated by audits. Others died because transaction costs consumed their apparent advantage. A few survived the demanding tests — and even those were classified as *candidates* requiring forward evidence, not as systems with permission to trade.

The experience changed how I evaluate algorithmic strategies. I no longer treat a good-looking backtest as the conclusion of research. I treat it as the beginning of an attempt to disprove it.

## 1. First problem: a backtest can simulate the wrong strategy

One of the most consequential investigations in the project was an audit of its own trade-execution model (`EXIT_MODEL_AUDIT.md`).

An earlier simulator judged a trade by whether the *next candle closed above the entry price*. That is convenient — and it does not reproduce what a bot with a stop-loss and take-profit actually experiences. A strategy with a 2:1 reward-to-risk ratio doesn't win because the next candle closes green. Price has to reach the target *before* it reaches the stop.

So I took two frozen signal families from earlier rounds — Family A (`BBW_STRICT+RV_LO+DST_NR+PRG_VH`, RR 2.0, 91 signals) and Family C (`ADX_ST+PBD_HI`, RR 3.0, 2,044 signals; 52 symbols, 1H crypto) — and ran the *same frozen signals* through five exit models, from the original proxy to a bot-faithful intrabar simulation that checks whether the stop or target was actually reached first.

| | Proxy PF | Bot-faithful PF | Win rate change |
|---|---|---|---|
| Family A (n=91) | 3.353 | **1.193** | 62.6% → 37.4% |
| Family C (n=2,044) | 2.534 | **0.785** | 45.8% → 20.7% |

Family A went from exceptional to thin. Family C went below 1.0 — its gross losses exceeded its gross profits once the exits were real.

The signals had not changed. The exit model had. And the audit answered the obvious objection, too: removing a 100-bar time cap barely moved the numbers. The gap was structural — the proxy never checked whether the stop or the target came first.

The requirement this creates is fundamental: the backtest must model the strategy that will actually be traded — entry timing, stop conditions, target conditions, execution assumptions. **A strategy is not just an entry signal. Its exits are part of its definition.**

*Receipt: `EXIT_MODEL_AUDIT.md` (M1 vs M3 tables), `scripts/exit_model_audit.py`.*

## 2. Second problem: a profitable result built with future information

Another lesson came from an apparent five-minute range-fade strategy.

The initial result looked extraordinary: 107 trades, a profit factor of 4.67, a 76% win rate, max drawdown −4.4% — and it even held up on a holdout period (PF 4.10). It looked exactly like the discovery a quantitative researcher wants to find.

Then a causal audit asked a simple question: what information did each decision actually have? The answer was disqualifying. The implementation used the day's low — the *full day's* low — in a session that had not finished. The strategy was reading a price level that would not exist until hours after the trades it justified.

The result was retracted. Not adjusted — retracted. This is not a small haircut to the performance statistics; it changes what the experiment was *about*. A backtest with a lookahead defect does not describe a worse version of your strategy. It describes a system that cannot exist in real time.

The lesson is narrow and absolute: every feature, indicator and decision rule must be checked for temporal causality. A model can be mathematically correct and operationally impossible.

QuantLab now treats causal auditing as a permanent stage of its process — a result is not "verified" until someone has reconstructed what was knowable at each decision point.

*Receipt: `quantlab_output/r090_final_report.md` (H2_range_fade row); retraction recorded in the project's research history.*

## 3. Third problem: passing historical validation but failing the implementability audit

The blind-validation campaign took a separate group of strategies through per-year holdouts (2023–2026, costs included) rather than one pooled number. Some results looked genuinely strong.

The T25 trend configuration, for example, reported yearly profit factors of 1.744 / 1.529 / 1.537 (2024/25/26) with about −2% drawdown, and its companion portfolio T27 looked even sturdier (PF 1.40 / 1.32 / 1.81, max drawdown under 1%).

Then a later audit — T34 — asked the same question as the range-fade audit, one level deeper: was the machine-learning filter using information available *at the actual entry bar*?

It was not. The trend simulator was associating each trade with *exit-bar* features, so the Random Forest gate was evaluating conditions from after the trade had already entered. Verbatim reproduction matched the original log — and the corrected version, gated at the true entry bar, collapsed to roughly PF 1.03: a $100 account ending near $106 instead of ~$199. No edge.

And the audit found a second, independent problem in the mean-reversion champion, which had used a different simulation path and was *not* affected by the exit-anchoring defect: re-tested on the full 50-symbol universe, its 2024 year turned losing (PF 0.739) and its real max drawdown was −28.2%, not the −9.4% the narrower test had reported.

Two strategies. Two different ways a passing backtest was wrong. Neither fix was a tuning change — both were corrections to what the backtest was *measuring*.

A proper audit has to reconstruct not only the market outcome but the information set available at each decision point — and the universe the statistics claim to describe.

*Receipt: README "T34 audit" block; `blind-validation` branch (`t34_lib.py`, `t34_validate.py`, `t34_output/`).*

## 4. Why a profit factor above 2 did not convince me

The R136 Supertrend experiment is the cleanest illustration of why headline metrics can mislead — this one from the current, audit-hardened process.

I tested the canonical Supertrend configuration (period 10, multiplier 3.0), long-only, exits on trend reversal, on daily crypto perpetual-swap candles for 20 liquid instruments, June 2023 → November 2026, with 0.10% round-trip costs charged in risk units. Parameters were frozen in advance; no sweeps were allowed.

The pooled results looked impressive:

| Metric | Result |
|---|---|
| Trades | 282 |
| Profit factor after costs | **2.04** |
| Net performance | +448.6R |
| Win rate | 34.4% |
| Median trade | **−1.57R** |

The strategy even stayed above PF 2 under a doubled-cost stress test (2.02).

But the distribution told the real story. Eight trades — 2.8% of the sample — contributed **78.5% of total net profit** (40% of all gross winnings). Seven of the eight exited in 2024. They were long positions held 43–133 days on SOL, NEAR, AVAX, SUI, BNB, LINK, ADA and XRP — the exact instruments and months of crypto's 2023–24 bull runs. The strategy was clearly *capable* of capturing powerful trends. The problem was that its profitability depended on when those trends appeared — which is not something the indicator controls.

The pre-declared yearly stability requirement exposed it:

| Exit year | Trades | Net R | Profit factor |
|---|---|---|---|
| 2023 | 24 | −24.8 | 0.34 |
| 2024 | 106 | **+475.1** | 4.21 |
| 2025 | 76 | +10.7 | 1.09 |
| 2026 | 76 | **−12.5** | 0.90 |

The nuance matters, and I want to state it carefully because the careless version is everywhere. Removing the eight largest winners did *not* make the book negative — the remaining 274 trades still produced +96.3R (PF 1.22). So it would be inaccurate to claim the strategy was "pure luck" or that dropping eight trades "proved there is no edge." What the analysis actually shows: a thin positive base, a typical trade that loses money, and a profit pool whose size and timing are lottery-dependent. And 2025's nominally positive +10.7R contains the single largest jackpot — strip the XRP trade and 2025 is about −66R.

A year-block bootstrap (resampling the four calendar years with replacement, 2,000 draws, seed fixed) produced a 95% confidence interval on total return of roughly **−75R to +1,413R** — with about a 28% probability of a non-positive replication. The interval including zero does not *prove* the strategy has no edge; it means the data cannot rule out that possibility, because the profit mass sits in one year-block that may or may not repeat.

Under the gates declared before the experiment ran, it was closed. Not declared universally worthless — it failed to demonstrate sufficiently reliable profitability under the criteria that were set in advance. That is a different, more defensible claim, and it is the one the evidence supports.

*Receipt: `quantlab_output/r136_evidence_package.md` (audit-grade; every figure recalculated from `r136_supertrend_book.csv`).*

## 5. Transaction costs can destroy an edge before it reaches the market

The same tests kept returning to one pattern: at high frequency, the machinery eats the edge.

Five-minute strategies generated thousands of signals; several looked encouraging gross and died net. The information wasn't necessarily absent — the *expected movement per trade* was smaller than the cost of capturing it. In the R090 battery, for instance, that same 4.67-PF range-fade fell to PF 2.24 at a 0.05%-side cost; its neighbors fell below 1.0.

The newest example is the funding-rate carry experiment (R137): collect perpetual-swap funding payments with a delta-neutral position — short the perp, hold the spot. No directional view, a genuinely intuitive income trade. The test covered 32 currencies and 3,378 eight-hour funding epochs, three pre-declared thresholds, and the real cost of toggling two legs.

The primary configuration produced a book APR of **−0.33%** before even accounting for the ~2× capital the two legs require. Every threshold was negative. The reason is almost elegant: funding opportunities were typically 0.01–0.05% per epoch, while a two-leg round trip cost 0.10% — and following the previous epoch's rate means eating the negative surprise epochs too.

An intuitive income strategy failed exactly where the five-minute strategies failed: **an edge must be larger than the machinery required to capture it.** The cost wall does not care what frequency you attack it at.

*Receipt: `quantlab_output/r137_final_report.md`, `r137_carry_epochs.csv`.*

## 6. Failed experiments are not wasted experiments

QuantLab has produced results that survive everything above — and the process treats even those with suspicion.

The R135 trendline-break experiment mechanized the classic chartist move — a break of a line fitted to recent swing pivots — across the same 20 liquid instruments on daily data. Its primary configuration recorded 1,076 trades, PF 1.42 after costs, +229.1R, with **every yearly block positive** (2023–2026), stable halves (PF 1.44 / 1.41), and a year-block bootstrap interval of [+165R, +293R] that excludes zero. It passed every gate the Supertrend experiment failed.

It was classified as a *candidate* — not promoted. Paper-tracking began immediately, and the first forward window (17 trades as of this writing) sits at −5.5R with a 12% win rate. That is far too few trades to conclude anything — the point is that nobody gets to skip the forward evidence, including the strategies that passed.

That is the actual product of a research framework: not a pile of rejected strategies, but a *boundary map* — hypothesis → promising historical result → validated candidate → deployable system — where each stage requires different evidence. Negative results are what draw the boundaries. The campaigns across crypto, forex and synthetic markets also showed that edges do not automatically transfer between environments: the same pipeline that worked on crypto 1H came out break-even on forex (PF 1.099), and Deriv-style synthetics behaved like a random walk. Each of those is a dead branch pruned with a receipt.

*Receipts: `quantlab_output/r135_final_report.md`, `r135_paper_log.csv`; README blind-validation campaign notes.*

## 7. What I now check before trusting a trading strategy

The research reordered my checklist. I no longer start from the profit factor — I start from the ways the result could be fake, and work forward:

1. **Causality** — did every decision use only information available at that moment? (Audited, not assumed.)
2. **Execution realism** — do the simulator's entries, exits and costs match the system that will actually trade? Are exits part of the strategy definition?
3. **Stability across time** — does it survive year-by-year scrutiny, not just a pooled number? Does the profit depend on one regime?
4. **Profit distribution** — is the result broadly distributed, or does a handful of trades carry the book? (And what exactly remains if they're removed?)
5. **Cost sensitivity** — does the edge survive fees, spread and slippage *at its own frequency*? Is the edge larger than the machinery that captures it?
6. **Statistical uncertainty** — does the evidence support a positive result, or is the interval still wide enough to include zero? Say which.
7. **Forward validation** — does it behave the same on data that arrives after the backtest is frozen?
8. **Traceability** — can every number be regenerated from committed artifacts by someone who doesn't trust me?

None of these guarantees future profitability. But skipping them is how a historical coincidence, an implementation error or a market-specific accident gets dressed up as a durable edge.

## Conclusion: the goal is not a beautiful backtest

The most valuable output of this project is not an indicator or a model. It is a process that can challenge its own conclusions — and a paper trail that lets other people check whether it did.

A profit factor of 3.35 became 1.19 when the exit model was corrected. A five-minute system with a 4.67 profit factor was retracted for lookahead. A trend portfolio lost its edge when its filter was evaluated at the true entry bar; its mean-reversion sibling turned out to have a bigger drawdown than advertised. A Supertrend system with a profit factor above 2 failed its stability requirements. A funding-carry income strategy went negative after costs.

None of this means algorithmic trading is impossible. It means the quality of a backtest is determined by things its final statistics cannot show you: what information the decisions had, what exits were modeled, what years were survived, what fraction of profit is concentration, and whether anyone else can regenerate the numbers.

The strongest quantitative research is not the research with the most attractive equity curve. It is the research that makes its assumptions visible, tests its conclusions against independent evidence, and is willing to close a promising experiment when the evidence does not hold.

That is the standard I want QuantLab to meet.

---

**Ojeology** is an independent quantitative researcher and developer of QuantLab, an open-evidence research project investigating systematic trading strategies across crypto, FX and synthetic markets. Every experiment cited here is reproducible from committed pre-specifications, engines, trade books and reports in the public repository.
