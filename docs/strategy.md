# Earnings expectation-reaction mismatch

Frozen 2026-07-10. Do not tune this specification against the final period.

## Event and decision

1. Establish the earliest verified public earnings-release time from company IR
   or a primary wire. SEC acceptance is a conservative fallback, not proof of
   earliest publication.
2. Find the first liquid regular session after publication.
3. Wait 30 minutes. Use only information whose `known_at` is no later than the
   decision timestamp.
4. Enter at the next executable quote. The bar screen is a coarse research stage
   using 15 basis points per side; it cannot promote without 95% quote coverage.
5. Hedge market/sector beta and exit at the fifth subsequent common trading-day
   close.

The 60-minute entry and 30-basis-point-per-side cases are mandatory robustness
checks.

## Features and target

Fundamental information includes point-in-time EPS and revenue surprise, new
guidance versus prior guidance and pre-release consensus, analyst dispersion,
revision breadth, and analyst counts.

The observed reaction is the 30-minute stock return less the beta-adjusted sector
return, standardized by pre-event volatility. Abnormal 30-minute volume and
liquidity are retained separately.

```text
fundamental surprise = mean(directional standardized surprises, revision breadth)
mismatch              = fundamental surprise - residual reaction score
target                = five-day stock return - sector ETF return
```

Training-only 1%/99% winsorization, imputation, and standardization are mandatory.

## Model ladder

- M0: price, reaction, volume, volatility, liquidity, session, and sector.
- M1: M0 plus structured earnings information.
- M2: M1 plus beta-residual reaction, fundamental composite, and mismatch.

The ladder is strictly nested. M1/M2 remain unavailable unless point-in-time EPS
and revenue coverage is at least 80% in training and validation and representative
across eligible years, sectors, and company-size buckets. Elastic net is the only
permitted model family until M2 passes.

## Trading and risk

A prediction becomes a trade only when:

```text
abs(expected residual return) > 2 * estimated all-in round-trip cost
```

Long positive tails and short negative tails; otherwise no trade.

- At most five positions.
- 5% NAV per stock.
- 25% maximum gross.
- 15% maximum sector gross.
- 15% maximum net.
- No leverage and no averaging down.

Portfolio statistics come from signed stock/hedge quantities in a daily ledger.
NAV, returns, drawdown, gross/net/sector exposure, and stock/hedge turnover are
marked every common session rather than booked only on exit dates.

## Validation and final holdout

Validation compares M0 versus M1 and M1 versus M2 using RMSE, MAE, Spearman IC,
net portfolio return, delayed entry, doubled costs, year/sector stability, and
company/event/quarter concentration. Deterministic block-feature and timestamp
permutation controls are recorded for every fitted model.

The final period may be opened once, only after M2 has all of:

- net Sharpe above 1;
- deflated-Sharpe probability above 0.95;
- positive doubled-cost and delayed-entry returns;
- stable years and sectors;
- no positive-P&L concentration above 25%; and
- executable quote coverage of at least 95%.

If properly measured M2 fails, retire it. The next independent strategy is forced
flow: index reconstitutions, ETF demand, corporate actions, and required shares
divided by ADV. Behavior is an M3 incremental overlay; political context is a
slow prior; world-model teachers remain shadow research only.
