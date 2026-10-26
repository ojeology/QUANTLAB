# R129 — THE FLOW LAYER — FINAL (2026-10-26)
Prespec: docs/RESEARCH_FRONTIER_R129.md (locked pre-pairing; on-chain data-blocked, documented).
Overlays evaluated on the champion kept book (today's environment rebuild: PF@c 1.81 — env-drift
disclosure; all comparisons are within-book, hence internally valid).

## CELL A — STABLECOIN LIQUIDITY GATE: **CLOSED (retention gate)**
- 2025 slice: gated n=76/119, PF@c 11.07 vs full 3.25 — spectacular, and seductive.
- 2026 slice: gated n=**0**/174. The gate stops trading entirely — stablecoin 30d supply growth
  has sat BELOW its own expanding median for the whole of 2026.
- Reading (the real finding): the champion's profits CLUSTER in liquidity-expansion eras — the
  Aug-2025 +40R month lives exactly there. But as a forward overlay this is not a filter, it is a
  market-timing call ("the engine is off") that would have kept the champion out during Sep-2026's
  +14.2R month. It explains WHERE the edge came from; it cannot modulate it. G2 FAIL → CLOSED.
- Design lesson recorded in standards: retention gates must precede significance gates (a 0-trade
  gate passes any significance test vacuously — G1's "PASS" at n=0 is the proof).

## CELL C — FEAR&GREED EUPHORIA VETO: **CLOSED (vetoes nothing)**
- Gated n=174/174 (100% pass-through) on PRIMARY — F&G never exceeded 75 at any pop entry in 2026;
  100% on the full book too. All gates FAIL trivially.
- Reading: compression-pops are structurally non-euphoric events — they fire in quiet, neutral
  tape, exactly as the coil logic says. The champion already avoids euphoria *by construction*.
  A sentiment veto is a solution to a problem this strategy doesn't have.

## A×C interaction (report-only): both gates → n=76 (26% retention) — same 2025-only cluster.

# THE FLOW LAYER VERDICT: CLOSED
The new-information map from this sandbox is now COMPLETE:
| information axis | status |
|---|---|
| exchange-internal (funding, OI, breadth) | tested R115/R116/R120 — closed |
| exchange-external: stablecoin liquidity | tested TODAY — closed (explains, cannot modulate) |
| exchange-external: sentiment (F&G) | tested TODAY — closed (structurally orthogonal) |
| exchange-external: on-chain (Coin Metrics, blockchain.com) | DATA-BLOCKED |
| exchange-external: ETF flows, liquidations, options skew, L2 | DATA-BLOCKED (paid/CF walls) |

Every experiment axis reachable with free, verifiable data has now been run. What remains open is
not an experiment: it is (1) the Jan-1 quarterly checkpoint, and (2) an OPERATOR-LEVEL decision —
buying new information (on-chain APIs, liquidation feeds, L2) — which is the only door the lab
cannot open from inside the sandbox.
