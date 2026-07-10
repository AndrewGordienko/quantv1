# Earnings alpha sprint — frozen protocol v1

Frozen 2026-07-10. Do not tune these definitions against the untouched cells.
The general actor graph is paused until this sprint is promoted or rejected.

## Sample and provenance

- Sample window: 2021-07-01 through 2026-06-30.
- Target breadth: 200–500 U.S. liquid common stocks, admitted by trailing
  point-in-time price and dollar-volume rules. Do not select on future market cap
  or current index membership.
- Canonical event time is the earliest verified company IR or primary press-wire
  release. SEC 8-K acceptance is stored as a conservative fallback, never
  silently called the earliest release.
- Store BMO, AMC, during-session or unknown status with its provenance.
- Consensus is optional. A value is usable only when it is an archived vendor
  snapshot with `estimate_as_of < earliest_public_time`, marked point-in-time and
  not a final revised series. If this cannot be licensed, run the no-consensus
  baseline and leave surprise fields missing.
- Download one-minute bars and NBBO quotes only around each event. Bars without
  quotes are eligible for the first coarse screen using 15 bps per side. NBBO
  is required only if the bar-cost screen survives and advances toward promotion.

## Frozen splits

- Training ends 2024-06-30. The following year is validation and may be used for
  model/layer selection. The final test begins 2025-07-01 and is opened once only
  after a model specification is written to the immutable lock artifact.
- Unseen-company holdout is the deterministic 20% SHA-256 bucket under
  `earnings-alpha-v1`; it is fixed before outcomes are inspected.
- The primary test is future time across every eligible company. Unseen-company
  results are a separate transfer-robustness diagnostic, never intersected with
  the primary time test for model selection.

## Decision and target

- V5's primary decision is 30 minutes after the first liquid regular session
  begins after release. Primary target is the five-trading-day sector-adjusted
  return. Secondary targets are 2 hours, 1 day and 20 trading days.
- Delayed-entry robustness uses 60 minutes for the five-day model.
- Coarse execution enters at the next one-minute bar open with 15 bps per side;
  doubled-cost robustness uses 30 bps per side and exits at the fifth subsequent
  common asset/sector-ETF close.
- Raw, sector-residual, modeled hedge and actually quote-executed returns remain
  separate. Pre-event volatility uses information strictly before release.

## Model ladder

1. Descriptive continuation/reversal tables by BMO/AMC, gap, first-five-minute
   move, surprise bucket, volatility, year and sector.
2. Price-only elastic net baseline.
3. Earnings elastic net adding session, verified EPS/revenue/margin/cash-flow/
   bookings surprise and new guidance versus prior guidance/consensus.
4. Add pre-event options-implied move and positioning.
5. CatBoost or quantile regression may be added only when elastic net improves
   untouched predictive loss and has positive untouched net P&L.
6. CEO/CFO prepared-versus-Q&A behavior comes afterward and must show incremental
   lift over the complete financial/reaction model.

## Portfolio constraints

- Maximum five concurrent positions; 5% NAV per name; 25% gross exposure;
  15% gross per sector; 15% net exposure.
- Entry at ask for longs/bid for shorts; exit on the opposite side. Add one basis
  point adverse-selection slippage per side plus configured fees.
- No averaging down. The five-day primary model carries positions overnight;
  intraday outputs are diagnostics only and cannot advance the strategy.

## Promotion gates

Validation determines whether better timestamps and quotes are worth acquiring.
Every promotion gate must later pass on the once-opened final test:

1. Positive net portfolio return.
2. Earnings elastic net improves predictive loss and net P&L over price-only.
3. Every eligible holdout year is positive and at least 70% of eligible sectors
   are positive; eligible groups require at least 30 trades.
4. Deflated-Sharpe probability exceeds 0.95 using the full experiment-trial count.
5. Sixty-minute delayed entry and doubled non-spread costs remain positive.
6. No company, event category, or calendar quarter contributes more than 25% of
   positive P&L.
7. Historical NBBO quote coverage is at least 95% for executed entries/exits.
8. Net Sharpe exceeds 1.

Failure routes the next sprint to structured forced-flow/event edges: index
reconstitutions, corporate actions, then event-conditioned supplier/competitor
lead-lag. It does not route back to generic sentiment.
