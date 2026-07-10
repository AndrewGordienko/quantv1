# V4_PLAN.md — All-day public-information reaction engine

The pivot: **don't day-trade politicians — day-trade the market's reaction to
public events, using political/government information as context.** Congressional
disclosures arrive days–weeks late, so they can never be a day-trading trigger;
they are a slow *conditioning prior*. The tradeable frequency comes from the
market's reaction to freshly-timestamped public events.

Legality unchanged: only lawfully-obtained PUBLIC information. The objective is
public information the market has not yet fully processed — never MNPI.

## Data + execution stack (Canada-resolved)

Alpaca is US-only, so for Canada:
- **Data: Polygon.io** (global vendor, not a broker). Minute bars + news API with
  publish timestamps. Free tier = recent/2y with 5 req/min; **Starter ($29/mo) =
  5y history, unlimited calls** — what we're on. `POLYGON_API_KEY` in gitignored
  `.env`; ingester `ingest/polygon_data.py`.
- **Execution: Interactive Brokers** (paper + live, Canada). API via TWS/IB
  Gateway; connector is a later step (`execution/ibkr.py`, after a strategy clears
  the bar).

Everything writes to the SAME `bars_minute` + `events` tables, so the replay
engine is vendor-agnostic (Alpaca ingester kept as an alt).

## 1. Timestamped real-time event bus

One store, every event carries: `public_time` (critical — the backtest may see
NOTHING published after it), `source`, `event_type`, `tickers`,
`politicians/agencies`, `novelty`, `direction`, `magnitude`, `confidence`,
`source_reliability`. Sources:

- Real-time company news (Alpaca news stream / historical news API — has publish
  time + affected symbols)
- SEC EDGAR: 8-K, 6-K, Form 4, 13D/G (EDGAR APIs, filing timestamps)
- Earnings releases, guidance, analyst revisions
- Federal contracts / grants / procurement (USAspending — already built)
- Federal Register / regulatory actions (already built, 1933 rules)
- Congress bills / hearings / committee activity / disclosures (already have
  disclosures; bills/hearings via Congress.gov)
- 1-minute trades/quotes/spreads/volume/sector returns (Alpaca market data)

The existing `events` store (P congress + F insider + G contracts/FR) IS this bus
at daily resolution; V4 adds intraday news/market events to the same table and two
columns (`confidence`, `source_reliability`).

## 2. Event-reaction model (the flagship day strategy)

For every public event at time t, predict market- and sector-adjusted return over
**5m / 30m / 2h / close / next-open**, and classify: immediate continuation /
overreaction→reversal / delayed peer reaction / no-trade. Features: event
(type, sentiment, novelty, magnitude, source, entities); immediate reaction
(1m/5m return, rel-volume, spread, vol, VWAP distance, market/sector residual);
context (earnings proximity, short interest, liquidity, overnight gap, regime);
political context (committee membership, contract exposure, regulatory exposure,
recent politician purchases, sector-level political activity).

Target = **expected market-neutral return − spread − slippage − fees −
adverse-selection buffer**, NOT accuracy. (A 60%-accurate model can lose; a
48% model with asymmetric payoffs can win.)

## 3. Three independent strategies

- **A. Event shock continuation/reversal** — extract event, watch first 1–3 min,
  measure abnormal price/volume, predict continuation vs reversal, enter next bar
  only when expected return clears costs with margin.
- **B. Earnings gap + post-earnings drift** — more observations than politics.
  Surprise, guidance, language change, premarket gap + rel-volume, options-implied
  move, sector reaction, first 5–15 min behavior → continue/reverse/no-trade.
  Reaction relative to expectations, not the headline beat.
- **C. Event-conditioned lead/lag** — graph company→suppliers/customers/
  competitors, agency→contractors, rule→industries. When an event moves one name,
  trade related names that haven't reacted. A *reason* for the lag (vs generic
  mean reversion, which we already proved dies on costs).

## 4. Algorithm ladder (no deep RL — sample size won't justify it)

LLM/FinBERT for event EXTRACTION (not buy/sell) → elastic-net logistic baseline →
CatBoost/LightGBM for nonlinear event×reaction → quantile regression (distributions,
not direction) → PCA/Kalman residuals for peer/sector-neutral → HMM regime gate →
triple-barrier labeling → meta-labeling (which primaries to execute) → EW ensemble.

## 5. The backtester we need FIRST (this build)

Leak-free intraday **event-replay** engine:
1. Replays news/filings/minute quotes in `public_time` order.
2. Reveals each record only at its real public time.
3. Features computed only from already-available info.
4. Enters on the next executable quote/bar.
5. Historical bid/ask spreads + conservative slippage.
6. Tracks partial fills, rejected orders, halted stocks.
7. Locks an untouched time-based test.
8. Reports net Sharpe, drawdown, turnover, capacity, **deflated Sharpe**.

Reuses the intraday mean-reversion scaffolding; replaces the signal with
event-conditioned models. Built now as `v4/replay.py`; validated on the events +
hourly bars we already have (a daily/hourly PoC) until Alpaca minute data lands.

## 6. How it trades (watch every minute; trade rarely)

Universe 500–1500 liquid US names; 1-min bars; holds 5 min–EOD; **0–5 good
trades/day** initially; ≤3–5 concurrent; one entry per symbol/event; symbol
cooldown; no averaging down; no overnight unless a separate model approves; daily
loss/turnover limits; disable on abnormal spreads/feeds. "Trading every minute" is
destructive — the hourly-reversal result already proved turnover-without-edge dies
on costs.

## 7. What politics contributes

A conditioning variable, never the sole trigger:
`public event + unusual reaction + political/government relevance`. E.g. a defense
contract weighted by committee+procurement context; a healthcare ruling weighted
by committee/regulatory exposure; a chip restriction weighted by gov+supply-chain
exposure; a politician purchase as a slow prior when fresh company news arrives.
One sparse political event conditions many later intraday decisions.

## Build sequence

The active path is frozen in `docs/EARNINGS_ALPHA_SPRINT.md`. General actor-graph
expansion is paused.

1. Five years of point-in-time earnings events for 200–500 liquid companies.
2. Event-window minute bars and historical NBBO quotes only.
3. Descriptive continuation/reversal tables.
4. Price-only and structured-earnings elastic-net comparison.
5. Quote-executable constrained portfolio on unseen companies and unseen time.
6. CatBoost/quantile regression only if elastic net establishes signal.
7. CEO/CFO behavioral features only after the structured event model.
8. On failure: index reconstitutions/corporate actions, then event-conditioned
   supplier/competitor lead-lag—not generic sentiment.

Honest objective: demonstrate **net-positive alpha and Sharpe > 1 on untouched
data after costs** BEFORE any target return. Chasing a monthly % before finding
the edge forces leverage/overfitting. The LARGE sleeve stays as the slow sleeve
and its frozen forward record; V4 is a separate engine and separate forward record.

## Status

- Built this session: `v4/replay.py` (leak-free event-replay backtester),
  `ingest/alpaca_data.py` (minute bars + news ingester — needs keys), event-bus
  columns. Validated replay on Federal-Register→sector-ETF reactions (hourly).
- Active sprint: earnings acquisition → elastic-net baseline → quote-executable
  portfolio → promotion/rejection gates. Actor research is contextual only.
