"""DuckDB connection, schema, and point-in-time helpers.

Everything analytical hangs off a single DuckDB file. Raw tables (`trades`,
`prices`, `members`) are ground truth; derived tables (`features`, `scores`,
`signals`, `portfolios`) are regenerable.

Point-in-time discipline: the *filing_date* is the earliest date a trade could
have been acted on (the STOCK Act gives members 30-45 days to disclose). Any
"as of date D" query must therefore filter on `filing_date <= D`, never on the
transaction date. `trades_known_as_of()` enforces that.
"""

from __future__ import annotations

import duckdb

from .config import DB_PATH

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id       VARCHAR PRIMARY KEY,  -- deterministic hash of the row
    chamber        VARCHAR,              -- 'senate' | 'house'
    member         VARCHAR,              -- normalized display name
    member_key     VARCHAR,              -- normalized join key (lower, no punct)
    ticker         VARCHAR,
    asset_desc     VARCHAR,
    asset_type     VARCHAR,              -- stock, option, etc.
    tx_type        VARCHAR,              -- 'purchase' | 'sale' | 'exchange'
    tx_date        DATE,                 -- when the trade happened (per filing)
    filing_date    DATE,                 -- when it was disclosed (actionable date)
    filing_estimated BOOLEAN,            -- true if filing_date was estimated (senate)
    disclosure_lag INTEGER,              -- days between tx_date and filing_date
    amount_lo      DOUBLE,
    amount_hi      DOUBLE,
    amount_mid     DOUBLE,
    owner          VARCHAR,              -- self | spouse | child | joint
    first_seen_at  TIMESTAMP,            -- when OUR collector first observed this row
    raw            JSON
);

CREATE TABLE IF NOT EXISTS prices (
    ticker  VARCHAR,
    date    DATE,
    open    DOUBLE,
    high    DOUBLE,
    low     DOUBLE,
    close   DOUBLE,      -- split/dividend adjusted
    volume  DOUBLE,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS members (
    member_key VARCHAR PRIMARY KEY,
    member     VARCHAR,
    chamber    VARCHAR,
    party      VARCHAR,
    state      VARCHAR,
    committees JSON        -- list of {name, jurisdiction_sectors:[...]}
);

CREATE TABLE IF NOT EXISTS ticker_sectors (
    ticker VARCHAR PRIMARY KEY,
    sector VARCHAR,
    industry VARCHAR,
    market_cap DOUBLE
);

-- Fama-French / Carhart daily factors (percent -> stored as decimals).
CREATE TABLE IF NOT EXISTS factors (
    date   DATE PRIMARY KEY,
    mkt_rf DOUBLE,
    smb    DOUBLE,
    hml    DOUBLE,
    mom    DOUBLE,
    rf     DOUBLE
);

-- === v3: political-intelligence engine ===================================
-- Entity resolution anchor: SEC ticker <-> CIK <-> legal name. The backbone
-- of the temporal knowledge graph (congress trade -> company -> SEC filings,
-- contracts, lobbying). Deterministic, from SEC's public company_tickers.json.
CREATE TABLE IF NOT EXISTS sec_entities (
    ticker VARCHAR PRIMARY KEY,
    cik    VARCHAR,
    title  VARCHAR
);

-- Point-in-time EVENT STORE. Every signal-bearing public event from any layer
-- (P politician / G government / F fundamentals / M market / E event-novelty)
-- lands here with its PUBLIC timestamp so nothing can leak future information.
-- Alpha_i,t = sum over layers of decayed, novelty-weighted event contributions.
CREATE TABLE IF NOT EXISTS events (
    event_id       VARCHAR PRIMARY KEY,   -- deterministic hash of the source row
    layer          VARCHAR,               -- 'P' | 'G' | 'F' | 'M' | 'E'
    event_type     VARCHAR,               -- e.g. congress_purchase, insider_buy, gov_contract
    ticker         VARCHAR,               -- resolved company (nullable)
    entity         VARCHAR,               -- raw actor (politician, agency, insider…)
    direction      DOUBLE,                -- signed thesis direction, [-1, 1]
    magnitude      DOUBLE,                -- size/importance, [0, 1]
    novelty        DOUBLE,                -- surprise vs expectation, [0, 1]
    effective_date DATE,                  -- when the effect begins
    source_time    TIMESTAMP,             -- public timestamp (point-in-time gate)
    source_url     VARCHAR,
    payload        JSON
);
CREATE INDEX IF NOT EXISTS idx_events_ticker ON events(ticker);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(source_time);

-- Intraday (hourly) bars for the fast-trigger proof-of-concept. Free yfinance
-- history is ~2 years at 60m resolution; a true minute feed needs a paid source.
CREATE TABLE IF NOT EXISTS bars_hourly (
    ticker VARCHAR,
    ts     TIMESTAMP,      -- bar start, US/Eastern regular-hours
    open   DOUBLE,
    high   DOUBLE,
    low    DOUBLE,
    close  DOUBLE,
    volume DOUBLE,
    PRIMARY KEY (ticker, ts)
);

-- Ingestion watermarks: per (ticker, source) what range is loaded, so ingestion
-- is resumable and completeness is auditable. Fetch stages to Parquet; a single
-- bulk load brings Parquet -> bars_minute (no page-by-page conflict checks).
CREATE TABLE IF NOT EXISTS ingest_watermarks (
    ticker        VARCHAR,
    source        VARCHAR,          -- 'polygon_minute'
    req_start     DATE,
    req_end       DATE,
    loaded_through TIMESTAMP,       -- max bar ts staged
    row_count     BIGINT,
    trading_days  INTEGER,
    status        VARCHAR,          -- complete | partial | failed
    updated_at    TIMESTAMP,
    PRIMARY KEY (ticker, source)
);

-- === V4: minute bars for the intraday event-reaction engine ================
CREATE TABLE IF NOT EXISTS bars_minute (
    ticker VARCHAR,
    ts     TIMESTAMP,        -- bar start, UTC
    open   DOUBLE,
    high   DOUBLE,
    low    DOUBLE,
    close  DOUBLE,
    volume DOUBLE,
    trades INTEGER,
    vwap   DOUBLE,
    PRIMARY KEY (ticker, ts)
);

-- === Forward paper-trading record (APPEND-ONLY, immutable) ================
-- The scientifically-honest forward test. Decisions are recorded BEFORE
-- execution and never edited or backfilled. A rule change must create a new
-- `version`; it may not rewrite history. Plain LARGE is the primary (capital)
-- strategy; the four shadow variants are observational (no capital).
-- Decision HEADER: one row per (decision_date, strategy, version) — recorded even
-- when the book is empty, so an empty decision is still immutable and a new
-- strategy version on the same date coexists instead of being silently skipped.
CREATE TABLE IF NOT EXISTS forward_decision_headers (
    decision_date     DATE,
    strategy          VARCHAR,
    version           VARCHAR,
    decision_ts       TIMESTAMP,     -- wall-clock when the snapshot was taken
    signal_as_of      DATE,          -- data cutoff used to pick signals
    execution_session DATE,          -- session the orders execute at (next open)
    n_positions       INTEGER,
    gross             DOUBLE,
    cash              DOUBLE,
    status            VARCHAR,        -- RECORDED | PRELAUNCH_INVALID
    PRIMARY KEY (decision_date, strategy, version)
);

CREATE TABLE IF NOT EXISTS forward_decisions (
    decision_date      DATE,
    strategy           VARCHAR,
    version            VARCHAR,       -- content fingerprint of the strategy code
    ticker             VARCHAR,
    decision_ts        TIMESTAMP,
    signal_as_of       DATE,
    execution_session  DATE,          -- next session; theoretical entry is its open
    target_weight      DOUBLE,
    gross              DOUBLE,
    cash               DOUBLE,
    source_trade_id    VARCHAR,
    source_member      VARCHAR,
    source_filing_date DATE,
    first_seen_at      TIMESTAMP,     -- real collector observation time
    decision_price     DOUBLE,        -- last close known at decision time
    theoretical_entry  DOUBLE,        -- execution_session open (filled at settle)
    rationale          JSON,
    PRIMARY KEY (decision_date, strategy, version, ticker)
);

CREATE TABLE IF NOT EXISTS forward_fills (
    decision_date     DATE,
    fill_date         DATE,
    strategy          VARCHAR,
    ticker            VARCHAR,
    status            VARCHAR,        -- filled | rejected
    theoretical_price DOUBLE,         -- next-open reference
    fill_price        DOUBLE,         -- incl. modeled spread/slippage
    spread_bps        DOUBLE,
    slippage_bps      DOUBLE,
    PRIMARY KEY (decision_date, strategy, ticker)
);

CREATE TABLE IF NOT EXISTS forward_pnl (
    date         DATE,
    strategy     VARCHAR,
    ret          DOUBLE,
    equity       DOUBLE,
    gross        DOUBLE,
    turnover     DOUBLE,
    spy_ret      DOUBLE,
    qqq_ret      DOUBLE,
    beta_spy_ret DOUBLE,             -- beta-matched SPY (primary benchmark)
    PRIMARY KEY (date, strategy)
);

CREATE TABLE IF NOT EXISTS forward_positions (
    date       DATE,
    strategy   VARCHAR,
    ticker     VARCHAR,
    weight     DOUBLE,
    entry_date DATE,
    PRIMARY KEY (date, strategy, ticker)
);

CREATE TABLE IF NOT EXISTS forward_exits (
    exit_date       DATE,
    strategy        VARCHAR,
    ticker          VARCHAR,
    entry_date      DATE,
    reason          VARCHAR,          -- time_stop | ...
    source_trade_id VARCHAR,          -- original signal that opened it
    ret             DOUBLE,
    PRIMARY KEY (exit_date, strategy, ticker)
);

-- Derived / regenerable ------------------------------------------------------
CREATE TABLE IF NOT EXISTS skill_scores (
    member_key   VARCHAR PRIMARY KEY,
    member       VARCHAR,
    n_trades     INTEGER,
    n_purchases  INTEGER,
    raw_car      DOUBLE,   -- unshrunk mean abnormal return (purchases)
    shrunk_car   DOUBLE,   -- empirical-Bayes posterior mean
    ci_low       DOUBLE,
    ci_high      DOUBLE,
    hit_rate     DOUBLE,   -- fraction beating SPY over label horizon
    updated_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signals (
    trade_id     VARCHAR PRIMARY KEY,
    filing_date  DATE,
    ticker       VARCHAR,
    member       VARCHAR,
    score        DOUBLE,        -- model P(beats SPY over horizon)
    contribs     JSON,          -- per-feature SHAP contributions
    updated_at   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolios (
    as_of_date   DATE,
    ticker       VARCHAR,
    weight       DOUBLE,
    score        DOUBLE,
    rationale    JSON,
    PRIMARY KEY (as_of_date, ticker)
);

CREATE TABLE IF NOT EXISTS backtest_equity (
    strategy VARCHAR,
    date     DATE,
    equity   DOUBLE,
    PRIMARY KEY (strategy, date)
);
"""


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the project DuckDB, ensuring the schema exists."""
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
    return con


def init_db() -> None:
    con = connect()
    con.close()


def trades_known_as_of(con: duckdb.DuckDBPyConnection, as_of: str):
    """Return trades that were *disclosed* on or before `as_of` (YYYY-MM-DD).

    This is the point-in-time gate: a backtest standing at date D may only see
    trades whose filing_date <= D.
    """
    return con.execute(
        "SELECT * FROM trades WHERE filing_date <= ? ORDER BY filing_date",
        [as_of],
    ).df()


if __name__ == "__main__":
    init_db()
    print(f"Initialized DuckDB schema at {DB_PATH}")
