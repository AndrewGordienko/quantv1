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

-- News CATALYSTS. Membership is append-only and algorithm-versioned.  The
-- denormalized events.catalyst_id is only the current assignment pointer; the
-- history of assignments lives in catalyst_events.
CREATE TABLE IF NOT EXISTS catalysts (
    catalyst_id        VARCHAR PRIMARY KEY,
    build_id           VARCHAR,
    cluster_version    VARCHAR,
    earliest_public_time TIMESTAMP,
    headline           VARCHAR,
    n_article_revisions INTEGER,
    n_assets           INTEGER,
    created_at          TIMESTAMP,
    metadata            JSON
);
CREATE TABLE IF NOT EXISTS catalyst_builds (
    build_id           VARCHAR PRIMARY KEY,
    cluster_version    VARCHAR,
    data_snapshot_hash VARCHAR,
    created_at         TIMESTAMP,
    status             VARCHAR,
    metadata           JSON
);
CREATE TABLE IF NOT EXISTS catalyst_events (
    catalyst_id       VARCHAR,
    build_id          VARCHAR,
    event_id          VARCHAR,
    event_public_time TIMESTAMP,
    PRIMARY KEY (catalyst_id, event_id)
);
-- CRITICAL: each ticker carries its OWN first-public link time. At time t an asset
-- is usable only if first_link_public_time <= t (prevents a later-linked ticker
-- from being exposed at the catalyst's earliest timestamp).
CREATE TABLE IF NOT EXISTS catalyst_assets (
    catalyst_id           VARCHAR,
    build_id               VARCHAR,
    ticker                VARCHAR,
    first_link_public_time TIMESTAMP,
    source_event_id       VARCHAR,
    PRIMARY KEY (catalyst_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_catalyst_events_event ON catalyst_events(event_id);
CREATE INDEX IF NOT EXISTS idx_catalyst_assets_time
    ON catalyst_assets(ticker, first_link_public_time);

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

-- === Actor State & Influence Engine ======================================
-- Actor identity is stable; aliases, roles and asset exposures are separately
-- time-valid and sourced.  Hand-assigned authority priors may be retained only
-- inside metadata and are never model features.
CREATE TABLE IF NOT EXISTS actors (
    actor_id         VARCHAR PRIMARY KEY,
    name             VARCHAR,
    actor_type       VARCHAR,
    metadata         JSON,
    registry_version VARCHAR,
    registry_status  VARCHAR,
    source           VARCHAR,
    first_seen_at    TIMESTAMP
);
CREATE TABLE IF NOT EXISTS actor_aliases (
    actor_id              VARCHAR,
    alias                 VARCHAR,
    valid_from            DATE,
    valid_to              DATE,
    source                VARCHAR,
    record_version        VARCHAR,
    entity_link_required  BOOLEAN,
    first_seen_at         TIMESTAMP,
    PRIMARY KEY (actor_id, alias, valid_from, record_version)
);
CREATE TABLE IF NOT EXISTS actor_roles (
    actor_id       VARCHAR,
    organization   VARCHAR,
    role           VARCHAR,
    valid_from     DATE,
    valid_to       DATE,
    source         VARCHAR,
    record_version VARCHAR,
    first_seen_at  TIMESTAMP,
    PRIMARY KEY (actor_id, organization, role, valid_from, record_version)
);
CREATE TABLE IF NOT EXISTS actor_asset_exposure (
    actor_id       VARCHAR,
    ticker         VARCHAR,
    valid_from     DATE,
    valid_to       DATE,
    channel        VARCHAR,
    confidence     DOUBLE,
    source         VARCHAR,
    record_version VARCHAR,
    first_seen_at  TIMESTAMP,
    PRIMARY KEY (actor_id, ticker, channel, valid_from, record_version)
);
-- One row per (public actor event, affected ticker). public_time is the exact
-- moment it was public.  actor_event_role is causal/event participation, not
-- the actor's institutional title.  Only explicitly eligible roles enter B2.
CREATE TABLE IF NOT EXISTS actor_events (
    actor_event_id             VARCHAR,
    actor_id                   VARCHAR,
    ticker                     VARCHAR,
    public_time                TIMESTAMP,
    event_type                 VARCHAR,
    headline                   VARCHAR,
    catalyst_id                VARCHAR,
    source                     VARCHAR,
    first_seen_at              TIMESTAMP,
    source_event_id            VARCHAR,
    actor_event_role           VARCHAR,
    role_confidence            DOUBLE,
    role_evidence              VARCHAR,
    primary_hypothesis_eligible BOOLEAN,
    extraction_version         VARCHAR,
    metadata                   JSON,
    PRIMARY KEY (actor_event_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_actor_events_time ON actor_events(public_time);
CREATE TABLE IF NOT EXISTS actor_event_features (
    actor_event_id       VARCHAR,
    feature_version      VARCHAR,
    semantic_event_type  VARCHAR,
    stance               DOUBLE,
    magnitude            DOUBLE,
    topic                VARCHAR,
    certainty            DOUBLE,
    baseline_deviation   DOUBLE,
    regime               VARCHAR,
    sector               VARCHAR,
    pre_event_volatility DOUBLE,
    time_of_day_bucket   VARCHAR,
    created_at           TIMESTAMP,
    metadata             JSON,
    PRIMARY KEY (actor_event_id, feature_version)
);
CREATE TABLE IF NOT EXISTS actor_event_outcomes (
    actor_event_id             VARCHAR,
    ticker                     VARCHAR,
    horizon                    VARCHAR,
    outcome_version            VARCHAR,
    public_time                TIMESTAMP,
    raw_return                 DOUBLE,
    market_beta_residual       DOUBLE,
    sector_beta_residual       DOUBLE,
    standardized_residual      DOUBLE,
    modeled_beta_hedged_return DOUBLE,
    actually_hedged_return     DOUBLE,
    execution_source           VARCHAR,
    quote_source               VARCHAR,
    created_at                 TIMESTAMP,
    metadata                   JSON,
    PRIMARY KEY (actor_event_id, ticker, horizon, outcome_version)
);

-- Primary-source Fed samples.  The speaker panel supports B2; timestamped
-- chair press-conference segments support B3 state-change analysis.
CREATE TABLE IF NOT EXISTS fed_communications (
    communication_id  VARCHAR PRIMARY KEY,
    actor_id           VARCHAR,
    public_time        TIMESTAMP,
    communication_type VARCHAR,
    title              VARCHAR,
    source_url         VARCHAR,
    transcript         VARCHAR,
    prepared_or_qa     VARCHAR,
    source_sha256      VARCHAR,
    first_seen_at      TIMESTAMP,
    metadata           JSON
);
CREATE TABLE IF NOT EXISTS fed_transcript_segments (
    segment_id         VARCHAR PRIMARY KEY,
    communication_id   VARCHAR,
    segment_index      INTEGER,
    segment_public_time TIMESTAMP,
    actor_id           VARCHAR,
    segment_role       VARCHAR,
    text               VARCHAR,
    source_url         VARCHAR,
    first_seen_at      TIMESTAMP,
    metadata           JSON
);

-- === Focused earnings alpha sprint ========================================
-- A canonical event is promoted only when its earliest public release has
-- verified provenance. SEC acceptance alone is retained as a conservative
-- fallback but is not labelled "earliest" and cannot clear promotion gates.
CREATE TABLE IF NOT EXISTS earnings_events (
    earnings_event_id       VARCHAR PRIMARY KEY,
    ticker                  VARCHAR,
    cik                     VARCHAR,
    fiscal_period_end       DATE,
    fiscal_quarter          VARCHAR,
    earliest_public_time    TIMESTAMP,
    release_session         VARCHAR,   -- BMO | AMC | DURING | UNKNOWN
    timestamp_status        VARCHAR,   -- VERIFIED_EARLIEST | CONSERVATIVE_SEC_ONLY
    release_session_status  VARCHAR,   -- VERIFIED | INFERRED | UNKNOWN
    primary_source_id       VARCHAR,
    primary_source_url      VARCHAR,
    event_version           VARCHAR,
    first_seen_at           TIMESTAMP,
    metadata                JSON
);
CREATE TABLE IF NOT EXISTS earnings_event_sources (
    earnings_event_id   VARCHAR,
    source_id           VARCHAR,
    public_time         TIMESTAMP,
    source_type         VARCHAR,       -- company_ir | press_release_wire | sec_8k
    source_url          VARCHAR,
    is_direct_release   BOOLEAN,
    retrieved_at        TIMESTAMP,
    source_sha256       VARCHAR,
    metadata            JSON,
    PRIMARY KEY (earnings_event_id, source_id)
);
CREATE TABLE IF NOT EXISTS earnings_acquisition_watermarks (
    ticker          VARCHAR,
    source          VARCHAR,
    sample_start    DATE,
    sample_end      DATE,
    candidate_count INTEGER,
    status          VARCHAR,
    updated_at      TIMESTAMP,
    metadata        JSON,
    PRIMARY KEY (ticker, source, sample_start, sample_end)
);
CREATE INDEX IF NOT EXISTS idx_earnings_events_time
    ON earnings_events(earliest_public_time);
CREATE TABLE IF NOT EXISTS earnings_universe_snapshots (
    universe_version  VARCHAR,
    ticker            VARCHAR,
    eligibility_as_of DATE,
    trailing_adv      DOUBLE,
    last_price        DOUBLE,
    company_bucket    VARCHAR,       -- TRAIN_COMPANY | UNSEEN_COMPANY
    market_cap        DOUBLE,
    company_size_bucket VARCHAR,
    size_known_at     TIMESTAMP,
    size_source       VARCHAR,
    size_source_record_id VARCHAR,
    included          BOOLEAN,
    exclusion_reason  VARCHAR,
    first_seen_at     TIMESTAMP,
    metadata          JSON,
    PRIMARY KEY (universe_version, ticker)
);
CREATE TABLE IF NOT EXISTS earnings_consensus_snapshots (
    earnings_event_id   VARCHAR,
    ticker              VARCHAR,
    fiscal_period_end   DATE,
    metric              VARCHAR,       -- diluted_eps | revenue | ...
    estimate_value      DOUBLE,
    currency            VARCHAR,
    analyst_count       INTEGER,
    estimate_as_of      TIMESTAMP,     -- vendor snapshot time, must precede event
    vendor              VARCHAR,
    vendor_record_id    VARCHAR,
    is_point_in_time    BOOLEAN,
    is_final_revised    BOOLEAN,
    ingested_at         TIMESTAMP,
    metadata            JSON,
    PRIMARY KEY (vendor, vendor_record_id, metric, estimate_as_of)
);
CREATE TABLE IF NOT EXISTS earnings_actuals (
    earnings_event_id VARCHAR,
    metric            VARCHAR,
    actual_value      DOUBLE,
    currency          VARCHAR,
    public_time       TIMESTAMP,
    source            VARCHAR,
    source_url        VARCHAR,
    ingested_at       TIMESTAMP,
    metadata          JSON,
    PRIMARY KEY (earnings_event_id, metric, source)
);
CREATE TABLE IF NOT EXISTS earnings_guidance_snapshots (
    earnings_event_id VARCHAR,
    metric            VARCHAR,
    guidance_period   VARCHAR,
    lower_value       DOUBLE,
    upper_value       DOUBLE,
    currency          VARCHAR,
    public_time       TIMESTAMP,
    source            VARCHAR,
    source_url        VARCHAR,
    ingested_at       TIMESTAMP,
    metadata          JSON,
    PRIMARY KEY (earnings_event_id, metric, guidance_period, source)
);
CREATE TABLE IF NOT EXISTS earnings_options_expectations (
    earnings_event_id VARCHAR,
    observed_at       TIMESTAMP,
    expiration_date   DATE,
    straddle_mid      DOUBLE,
    underlying_mid    DOUBLE,
    implied_move      DOUBLE,
    implied_volatility DOUBLE,
    source            VARCHAR,
    source_record_id  VARCHAR,
    ingested_at       TIMESTAMP,
    metadata          JSON,
    PRIMARY KEY (source, source_record_id, observed_at)
);
CREATE TABLE IF NOT EXISTS earnings_positioning_snapshots (
    earnings_event_id       VARCHAR,
    observed_at             TIMESTAMP,
    short_interest_shares   DOUBLE,
    days_to_cover           DOUBLE,
    institutional_ownership DOUBLE,
    passive_ownership       DOUBLE,
    borrow_available        BOOLEAN,
    borrow_fee_bps_annual   DOUBLE,
    borrow_known_at         TIMESTAMP,
    source                  VARCHAR,
    source_record_id        VARCHAR,
    ingested_at             TIMESTAMP,
    metadata                JSON,
    PRIMARY KEY (source, source_record_id, observed_at)
);
CREATE TABLE IF NOT EXISTS earnings_call_segments (
    segment_id        VARCHAR PRIMARY KEY,
    earnings_event_id VARCHAR,
    call_public_time  TIMESTAMP,
    segment_index     INTEGER,
    segment_time      TIMESTAMP,
    speaker_name      VARCHAR,
    speaker_role      VARCHAR,
    section           VARCHAR,       -- prepared | question | answer
    text              VARCHAR,
    source            VARCHAR,
    source_url        VARCHAR,
    ingested_at       TIMESTAMP,
    metadata          JSON
);
CREATE TABLE IF NOT EXISTS earnings_event_outcomes (
    earnings_event_id VARCHAR,
    horizon           VARCHAR,       -- 30m | 2h | 1d | 5d | 20d
    entry_time        TIMESTAMP,
    exit_time         TIMESTAMP,
    raw_return        DOUBLE,
    sector_residual   DOUBLE,
    actually_hedged_return DOUBLE,
    execution_status  VARCHAR,
    outcome_version   VARCHAR,
    created_at        TIMESTAMP,
    metadata          JSON,
    PRIMARY KEY (earnings_event_id, horizon, outcome_version)
);
-- Bars and NBBO quotes are staged as event-window Parquet, not full-history
-- ticker dumps. This table is the completeness/provenance manifest.
CREATE TABLE IF NOT EXISTS earnings_market_windows (
    earnings_event_id VARCHAR PRIMARY KEY,
    ticker            VARCHAR,
    window_start      TIMESTAMP,
    window_end        TIMESTAMP,
    bars_source       VARCHAR,
    quotes_source     VARCHAR,
    bars_path         VARCHAR,
    quotes_path       VARCHAR,
    benchmark_ticker  VARCHAR,
    benchmark_bars_path VARCHAR,
    benchmark_quotes_path VARCHAR,
    bar_rows          BIGINT,
    quote_rows        BIGINT,
    quote_coverage    DOUBLE,
    status            VARCHAR,         -- COMPLETE_QUOTES | BARS_ONLY | FAILED
    retrieved_at      TIMESTAMP,
    metadata          JSON
);
CREATE TABLE IF NOT EXISTS earnings_experiments (
    experiment_id     VARCHAR PRIMARY KEY,
    sprint_version    VARCHAR,
    created_at        TIMESTAMP,
    status            VARCHAR,
    dataset_hash      VARCHAR,
    code_hash         VARCHAR,
    model_family      VARCHAR,
    holdout_definition JSON,
    metrics           JSON,
    promotion_gates   JSON
);

-- === Management Guidance Revision-Reaction Mismatch ======================
-- This is an independent public-data experiment. It never reads or writes the
-- vendor-consensus tables used by EERM.
CREATE TABLE IF NOT EXISTS mgrm_filings (
    accession_number VARCHAR PRIMARY KEY,
    earnings_event_id VARCHAR,
    ticker           VARCHAR,
    cik              VARCHAR,
    form             VARCHAR,
    items            VARCHAR,
    acceptance_time  TIMESTAMP,
    filing_date      DATE,
    primary_document VARCHAR,
    source_url       VARCHAR,
    first_seen_at    TIMESTAMP,
    discovery_version VARCHAR,
    status           VARCHAR,
    metadata         JSON
);
CREATE TABLE IF NOT EXISTS mgrm_documents (
    document_id      VARCHAR PRIMARY KEY,
    accession_number VARCHAR,
    earnings_event_id VARCHAR,
    ticker           VARCHAR,
    document_type    VARCHAR,
    source_url       VARCHAR,
    source_sha256    VARCHAR,
    raw_path         VARCHAR,
    public_time      TIMESTAMP,
    first_seen_at    TIMESTAMP,
    status           VARCHAR,
    metadata         JSON
);
CREATE TABLE IF NOT EXISTS mgrm_guidance_extractions (
    extraction_id    VARCHAR PRIMARY KEY,
    document_id      VARCHAR,
    earnings_event_id VARCHAR,
    ticker           VARCHAR,
    metric           VARCHAR,
    guidance_period  VARCHAR,
    lower_value      DOUBLE,
    upper_value      DOUBLE,
    midpoint         DOUBLE,
    units            VARCHAR,
    currency         VARCHAR,
    guidance_status  VARCHAR,
    stated_action    VARCHAR,
    supporting_sentence VARCHAR,
    deterministic_confidence DOUBLE,
    ai_confidence    DOUBLE,
    deterministic_payload JSON,
    ai_payload       JSON,
    agreement_status VARCHAR,
    extractor_version VARCHAR,
    public_time      TIMESTAMP,
    created_at       TIMESTAMP
);
CREATE TABLE IF NOT EXISTS mgrm_guidance_links (
    extraction_id    VARCHAR PRIMARY KEY,
    previous_extraction_id VARCHAR,
    midpoint_revision DOUBLE,
    range_width_change DOUBLE,
    revision_classification VARCHAR,
    link_status      VARCHAR,
    linker_version   VARCHAR,
    created_at       TIMESTAMP,
    metadata         JSON
);
CREATE TABLE IF NOT EXISTS mgrm_experiments (
    experiment_id    VARCHAR PRIMARY KEY,
    created_at       TIMESTAMP,
    status           VARCHAR,
    dataset_hash     VARCHAR,
    code_hash        VARCHAR,
    model_name       VARCHAR,
    holdout_definition JSON,
    metrics          JSON,
    promotion_gates  JSON
);
CREATE TABLE IF NOT EXISTS public_expectation_snapshots (
    snapshot_id      VARCHAR PRIMARY KEY,
    ticker           VARCHAR,
    expectation_type VARCHAR,
    metric           VARCHAR,
    period           VARCHAR,
    source           VARCHAR,
    source_url       VARCHAR,
    source_time      TIMESTAMP,
    first_seen_at    TIMESTAMP,
    raw_sha256       VARCHAR,
    raw_path         VARCHAR,
    payload          JSON
);

-- Independent V5 track 2: forced flows. Announcement, effective date and
-- required-flow estimate are distinct point-in-time fields.
CREATE TABLE IF NOT EXISTS forced_flow_events (
    forced_flow_event_id VARCHAR PRIMARY KEY,
    index_family         VARCHAR,
    index_name           VARCHAR,
    event_type           VARCHAR,     -- addition | deletion | weight_change
    ticker               VARCHAR,
    announcement_time    TIMESTAMP,
    effective_date       DATE,
    old_weight           DOUBLE,
    new_weight           DOUBLE,
    estimated_flow_usd   DOUBLE,
    flow_pct_trailing_adv DOUBLE,
    passive_ownership    DOUBLE,
    source               VARCHAR,
    source_url           VARCHAR,
    version              VARCHAR,
    first_seen_at        TIMESTAMP,
    metadata             JSON
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
        _migrate_schema(con)
    return con


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return table columns without relying on DuckDB version-specific DDL."""
    return {row[1] for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()}


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return table in {row[0] for row in con.execute("SHOW TABLES").fetchall()}


def _add_columns(con: duckdb.DuckDBPyConnection, table: str,
                 columns: dict[str, str]) -> None:
    """Non-destructively extend old databases created by earlier versions."""
    existing = _columns(con, table)
    for name, sql_type in columns.items():
        if name not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


def _migrate_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Apply additive migrations only.

    Historical catalyst and actor rows are research provenance.  In particular,
    this migration deliberately does not drop, truncate, or recreate the legacy
    ``news_catalysts``/``actor_relationships`` tables.
    """
    _add_columns(con, "events", {"catalyst_id": "VARCHAR"})
    _add_columns(con, "catalysts", {"build_id": "VARCHAR"})
    _add_columns(con, "catalyst_events", {"build_id": "VARCHAR"})
    _add_columns(con, "catalyst_assets", {"build_id": "VARCHAR"})
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_catalyst_build_event
        ON catalyst_events(build_id,event_id)
    """)
    _add_columns(con, "actors", {
        "actor_type": "VARCHAR",
        "metadata": "JSON",
        "registry_version": "VARCHAR",
        "registry_status": "VARCHAR",
        "source": "VARCHAR",
        "first_seen_at": "TIMESTAMP",
    })
    con.execute("""
        UPDATE actors SET registry_status='DEPRECATED_UNSAFE_V1'
        WHERE registry_status IS NULL
    """)
    _add_columns(con, "actor_events", {
        "source_event_id": "VARCHAR",
        "actor_event_role": "VARCHAR",
        "role_confidence": "DOUBLE",
        "role_evidence": "VARCHAR",
        "primary_hypothesis_eligible": "BOOLEAN",
        "extraction_version": "VARCHAR",
        "metadata": "JSON",
    })

    # Existing headline-regex rows were mentions, never verified actor actions.
    # Backfill their interpretation in place while retaining the original rows.
    con.execute("""
        UPDATE actor_events
        SET actor_event_role = COALESCE(actor_event_role, 'merely_mentioned'),
            role_confidence = COALESCE(role_confidence, 0.5),
            role_evidence = COALESCE(role_evidence, 'legacy headline alias match'),
            primary_hypothesis_eligible = COALESCE(primary_hypothesis_eligible, FALSE),
            extraction_version = COALESCE(extraction_version, 'legacy-news-mention-v1')
        WHERE actor_event_role IS NULL OR extraction_version IS NULL
    """)
    if _table_exists(con, "actor_relationships"):
        _add_columns(con, "actor_relationships", {
            "record_version": "VARCHAR",
            "deprecated_at": "TIMESTAMP",
            "deprecated_reason": "VARCHAR",
        })
        con.execute("""
            UPDATE actor_relationships
            SET record_version=COALESCE(record_version, 'legacy-unsafe-v1'),
                deprecated_at=COALESCE(deprecated_at, CURRENT_TIMESTAMP),
                deprecated_reason=COALESCE(
                    deprecated_reason,
                    'superseded by sourced actor_roles and actor_asset_exposure'
                )
            WHERE deprecated_at IS NULL
        """)
    if _table_exists(con, "earnings_market_windows"):
        _add_columns(con, "earnings_market_windows", {
            "benchmark_ticker": "VARCHAR",
            "benchmark_bars_path": "VARCHAR",
            "benchmark_quotes_path": "VARCHAR",
        })
    if _table_exists(con, "earnings_universe_snapshots"):
        _add_columns(con, "earnings_universe_snapshots", {
            "market_cap": "DOUBLE",
            "company_size_bucket": "VARCHAR",
            "size_known_at": "TIMESTAMP",
            "size_source": "VARCHAR",
            "size_source_record_id": "VARCHAR",
        })
    if _table_exists(con, "earnings_consensus_snapshots"):
        _add_columns(con, "earnings_consensus_snapshots", {
            "known_at": "TIMESTAMP",
            "forecast_dispersion": "DOUBLE",
            "revision_breadth": "DOUBLE",
            "feature_version": "VARCHAR",
        })
    if _table_exists(con, "earnings_actuals"):
        _add_columns(con, "earnings_actuals", {
            "known_at": "TIMESTAMP",
            "source_record_id": "VARCHAR",
            "feature_version": "VARCHAR",
        })
    if _table_exists(con, "earnings_guidance_snapshots"):
        _add_columns(con, "earnings_guidance_snapshots", {
            "known_at": "TIMESTAMP",
            "source_record_id": "VARCHAR",
            "guidance_status": "VARCHAR",
            "guidance_role": "VARCHAR",
            "feature_version": "VARCHAR",
        })
    if _table_exists(con, "earnings_positioning_snapshots"):
        _add_columns(con, "earnings_positioning_snapshots", {
            "borrow_available": "BOOLEAN",
            "borrow_fee_bps_annual": "DOUBLE",
            "borrow_known_at": "TIMESTAMP",
        })


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
