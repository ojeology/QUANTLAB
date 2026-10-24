# R122 — COT POSITIONING → FX DIRECTION — PRESPEC (LOCKED 2026-10-24, pre-pairing)
Operator opened the one untested FX lever: real, timestamped positioning (CFTC Commitments of
Traders, weekly). Status when locked: zero COT bytes in repo history; zero positioning↔return
pairs computed. Nothing below may change after first pairing.

## Prior (honest): LOW-MODERATE. FX family test #8 (multiplicity acknowledged — bar unchanged);
COT literature mixed; effect, if real, is slow (weekly) and cost-light by construction.

## Registered cells (TWO, pre-named; multiplicity handled by era requirement)
- **A (PRIMARY) — Follow the hedgers:** per currency, comm_net_pct = (comm_long − comm_short)/OI,
  z-scored over trailing 156 reports (≥100 required). LONG currency if z>0, SHORT if z<0.
- **B (SECONDARY, reported-not-deciding) — Fade spec extremes:** spec net pct ≥ its 90th trailing
  percentile (260 reports) → SHORT; ≤ 10th → LONG; else flat. If A fails while B "looks good",
  verdict = INCONCLUSIVE-with-note, never a flip.

## Timing law (causal, locked)
COT is effective Tuesday, released Friday 15:30 ET. Signal from report released at week w is
tradeable from the FIRST daily close STRICTLY AFTER that Friday → position held one week
(Mon-close→Mon-close). No bar earns its signal week (off-by-one law via explicit index shift).

## Universe & data
7 CME-traded majors (EURGBP has no COT — excluded): EUR, GBP, JPY, AUD, CAD, CHF, NZD vs USD.
Positioning: CFTC Legacy Futures-Only report (Socrata 6dca-aqww), full history per market.
Prices: HistData daily closes (existing 1H cache resampled + refetch 2003→2015).
Costs: locked pip table from R121 (0.6–1.2 RT pips), charged in return terms on every position
CHANGE (flip = full RT; hold = 0). Pair covered only when BOTH series exist.

## Eras (contamination-law structure)
- **DISCOVERY 2003-01 → 2015-12:** FX prices never examined by any test (ledger-verified). Gates run HERE.
- **CONFIRMATORY 2016-01 → 2026-10:** examined once (R121) — usable only as confirmatory re-test (convention 7).

## Gates (on DISCOVERY era, all required for "PROMISING")
- G1: portfolio PF@cost (weekly blocks) ≥ 1.10
- G2: bootstrap P(portfolio net > 0) ≥ 0.95 (weekly blocks, 5,000 draws)
- G3: annualized Sharpe ≥ 0.5
- G4: leave-one-currency-out — all 7 minima > 0
- G5 (era rule): CONFIRMATORY era PF@cost > 1.0 (sign consistency; weaker bar is the point)
Secondary B reported with identical stats but decides nothing.

## Verdicts
All 5 gates → PROMISING FOR FURTHER VALIDATION (next rung: charter monetization on FX micro lots).
Fail on G1–G4 → REJECT (direction closes: positioning-as-signal joins the closed map).
Pass G1–G4, fail G5 → INCONCLUSIVE (works in history, dead in the modern era — like TSMOM).
No iteration, no parameter adjustments, one run.

### DATA DEVIATIONS (amended 2026-10-24, before any pairing — disclosed)
1. Discovery-era prices: HistData M1 for 2003-2015 proved unfetchable in-session (token throttling);
   replaced by Yahoo Finance daily closes (repo-documented FX source, F-series). Seam: 2003-15
   indicative closes vs 2016+ HistData — weekly-horizon returns unaffected; per-pair starts:
   JPY 2003-01, CAD/CHF 2003-09, EUR/GBP/NZD 2003-12, AUD 2006-05. Discovery era per pair =
   first price date → 2015-12-31.
2. NZD COT report ENDS 2022-02-01 (CFTC discontinuation) → NZD active 2003→2022 only.
3. AMENDED (pre-pairing): 2016-2026 confirmatory prices ALSO from Yahoo dailies — same vendor both eras,
   removing the cross-vendor seam entirely (HistData 1H cache lost to a sandbox wipe; dailies are all R122 needs).

## RESULT (2026-10-24) - REJECT (Cell A discovery: PF 0.94, Sharpe -0.14, bootP 0.309)
Hedger-following is flat-to-negative in the clean era and flips sign after 2015 (era-instability disqualifies; the confirmatory +14.2% is an observation, not a lead). Cell B also dead (0.94). POSITIONING CLOSED. FX overall: price- and positioning-derived levers now fully mapped.
