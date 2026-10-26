# TRADING STYLE TAXONOMY — full disposition (2026-10-26)
Operator question: "have we tested all major trading types?" Answer below, from the ledger.
Three buckets: TESTED (receipt exists) / BLOCKED (needs data this sandbox cannot reach) /
EXCLUDED (by operator's standing rules).

## TESTED — every major style has at least one receipt
| trading type | where tested | outcome |
|---|---|---|
| Trend / momentum following | R107/R109/R113/R117/R120/R127/R128 (7 assets, 3 horizons) | dead everywhere; crypto CSM real-but-uninhabitable |
| Mean reversion / fading | R112 weekly extremes, F-series session fades | dead |
| Breakout systems | R117 Donchian, F-series range breaks, compression-pop family | dead — EXCEPT compression-pop+filter (the champion, benched) |
| Scalping (seconds-minutes) | R085-R095: 7 builds, 5m OKX | dead — cost wall (round-trip > bar edge) |
| Intraday day-trading | the champion itself (1H, avg hold ~3h) | the ONE survivor |
| Swing (days) | R110 daily cascade, R117/R109 dailies | dead |
| Position (weeks-months) | R122 COT weekly, R127/R128 N250 trends | dead/dominated |
| News & event trading | R111 FOMC/CPI sniper | dead (decaying, dead in holdout) |
| Seasonality / calendar | R114: 20 patterns, Bonferroni-corrected | 0/20 |
| Carry / funding harvest | R102, R115, R108 (FX carry) | thin-positive/uninhabitable; FX carry fails survival |
| Basis / hedged yield | R106 the floor | instrument, not strategy ($1.35/mo @ $300) |
| Grid systems | R012-R035 early grid sweeps (family) | dead — selection-era mirage |
| Martingale / DCA ladders | R103 dip-ladder, R114 DCA | dead (0/7 gates; BH/DCA fail charter) |
| Pairs / stat-arb | R051-R063 (incl. BTC-ETH spread) | dead |
| Buy & hold (as strategy) | R114 + R127/R128 asset dissections | fails charter laws; good as ASSET (gold/silver/oil cards) |
| ML-driven (any model) | R124 capacity ladder, R126 HMM, R128 walk-forward ML, R121/R123 transfers | sophistication can't create information; no OOS skill beyond noise |
| Sentiment/flow overlays | R129 F&G + stablecoin gates | explain the edge; cannot modulate it |
| Positioning-following | R122 COT (40y) | dead (era-instability) |

## BLOCKED — real styles, unreachable data from this sandbox
| trading type | what's missing | note |
|---|---|---|
| Market making / passive quoting | L2 books + fill simulation | nearest receipt: R098 maker-vs-taker cost split (PF 1.27/1.39) |
| High-frequency / latency arb | tick feeds, co-location-class data | cost-wall result at 5m makes the wall tighter lower |
| Cross-exchange / triangular arb | simultaneous multi-venue books | single-venue sandbox |
| Options / vol-selling strategies | option chains & IV surfaces | the champion trades the SAME vol premium without them |
| On-chain / DeFi-native flows | Coin Metrics, blockchain.com, DEX APIs | probed & blocked (R129) |

## EXCLUDED — by operator's standing rules
| market | rule |
|---|---|
| Stocks / indices | excluded forever (operator directive) |
| Opay / personal-finance side doors | never enters the repo |

## THE ONE-LINE SUMMARY
Every major trading TYPE has a receipt or a named blocker. The space is not "tested" in the sense
of exhausted-but-secretly-shallow: it is dispositioned, with the depth proportional to promise —
17 styles tested (1 survivor), 5 styles data-blocked with named feeds, 2 excluded by your rules.
The survivor is benched at its own pre-committed gate, awaiting the Jan-1 quarterly checkpoint.
