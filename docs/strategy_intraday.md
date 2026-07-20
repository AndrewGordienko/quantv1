# Intraday V1 — Frozen Spec (event-absorption day-trade engine)

**Status: SCOPED, DATA-GATED. No intraday result may be computed yet.** At intraday
horizons the spread-at-trade-time *is* the experiment, and there is no `quotes`
table. This spec + the NBBO/trades data contract (`src/quantv1/ingest/nbbo.py`)
is the design; nothing runs until real quotes + a calibrated fill simulator exist.
Do **not** fabricate intraday numbers from bars — the Latent-Flow experiment
already proved bar volume is not order flow (−18.2 bps, rejected).

This is a **separate engine**, not TSMOM sped up. The one architectural rule:
**replay and live call the same feature/decision/risk/execution code — only the
data clock changes.** The engine says **NO_TRADE by default** and executes only
when predicted net edge clears cost.

## Product definition (frozen answers)

| Question | V1 answer |
|---|---|
| Universe | ~200 most-liquid US equities + SPY + 11 sector ETFs |
| Trigger | a verified public **event** (SEC 8-K, earnings, index announcement, halt-resume) OR an abnormal residual price/volume shock |
| Decision time | 1–5 minutes after the trigger (let a liquid price form) |
| Holding periods | frozen {5, 15, 30, 60} min; hard exit by 15:50 ET |
| Positions | ≤ 3 simultaneous |
| Overnight | **never** — flat by close |
| Order type | marketable limit vs current NBBO |
| Model | elastic-net baseline; LightGBM challenger only |
| Target | executable **quote-to-quote** net return over the horizon |
| Broker | paper first (IBKR/Alpaca), reconcile before any live |
| Frequency | opportunity-driven; **0 trades is a valid day** |

**Decision rule:** trade only when `E[r_{t→h} | x_t] − spread − fees − slippage −
impact` > **2×** the estimated round-trip cost. Else NO_TRADE.

## Primary strategy — event absorption

Not the rejected generic news fade (that treated events generically on bar data).
The new, sharper question: *given the event type, the signed surprise, and the
actual order-flow response, has the market finished absorbing the information?*

Flow: verified event → wait 1–5 min → measure {initial return vs SPY+sector,
aggressor trade imbalance, spread & quote drift, same-clock-minute relative
volume, event type & signed direction, gap & pre-event trend, peer confirmation}
→ predict continue / reverse / untradeable → enter only if net edge > 2× cost →
exit on frozen horizon or 15:50.

Secondary families (later, same discipline): liquidity-shock recovery (needs
trades+quotes; not the rejected bar mean-reversion) and cross-market propagation
(5–30 min, requires a structural reason for the lag).

## Data contract (the load-bearing gap)

`src/quantv1/ingest/nbbo.py` defines two point-in-time tables (guarded, needs a
paid Polygon quotes/trades tier — free tier is insufficient):

- `quotes_nbbo(ticker, ts, bid_price, bid_size, ask_price, ask_size, …, known_at)`
- `trades_tick(ticker, ts, price, size, exchange, aggressor, …, known_at)`

`aggressor` (buy/sell-initiated) is derived by the quote rule / Lee-Ready against
the NBBO — that is the true order-flow feature. Every row carries `known_at`
(when OUR collector observed it) so replay can never see the future.

## Fill simulator (build BEFORE any signal)

The counterintuitive but correct order: the simulator IS the answer intraday. A
signal that is +18 bps under optimistic fills and −6 bps under realistic fills is
an assumption, not a discovery. Requirements: marketable-limit only, cross the
spread (no passive queue credit), explicit **50 ms** decision→exchange latency,
partial fills, cancels/rejects, fees, borrow (shorts need a locate — fail closed),
adverse-selection cost. **Calibration:** replay real IBKR/Alpaca paper fills
through the simulator; until it predicts them within tolerance, treat every
backtest bp as fiction.

## Validation gates (a candidate advances only if ALL hold)

- ≥ 100 independent **event clusters** (and ideally ≥ 500 executions), with
  **standard errors clustered by day** (1,000 trades across 40 days ≠ 1,000 obs).
- Positive net return on unseen data after realistic fills.
- Net Sharpe > 1; positive under **2× costs** and under **added latency**.
- No single ticker, month, or event family dominates.
- Event strategy beats **matched non-event shocks** (placebo).
- Stable across volatility regimes.
- **Deflated Sharpe** against the global trial ledger (this is trial #N, not #1).

Fail → archive in the ledger, no rescue filters. Same discipline as every other
lane.

## Risk config (paper)

~10 bps NAV risk/trade · ≤3 positions · ≤50 bps daily-loss stop · never overnight
· immediate kill on stale market data, broker disconnect, or reconciliation
mismatch. **"Risky mode" scales size only AFTER an edge is demonstrated** — it
never lowers the admission threshold, forces trades, adds leverage to a negative
strategy, or removes loss controls.

## Build sequence

1. **NBBO + trades ingest** (`ingest/nbbo.py`) — paid quotes/trades, 50–200-symbol
   pilot, exact exchange timestamps, corporate actions, sessions.
2. **Replay engine** — stream quotes/trades/events in timestamp order; no future
   access; simulate marketable limits, partial fills, latency, rejects; replay ==
   live decision interface.
3. **Fill simulator + calibration** vs paper fills.
4. **Small frozen feature set + elastic-net baseline.**
5. **Decisive validation** (gates above). Fail → archive.
6. **Paper-forward engine** — every session, record every candidate incl.
   NO_TRADE, real paper orders/fills/reconciliation, auto-flat by close.

Meaningful forward evidence ≈ 60 sessions / 100–200 fills. Live capital only after
paper matches replay within tolerance, at minimum size.

## Honest base rate

Most of this ends in an archived null — that is the design, not pessimism. The
fastest route to a day-trading engine is the same as any engine: force one
hypothesis to a verdict under honest costs, not add more hypotheses.
