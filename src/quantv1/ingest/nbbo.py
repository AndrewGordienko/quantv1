"""NBBO quotes + tick trades ingest — DATA CONTRACT (guarded skeleton).

The load-bearing gap for any intraday research (docs/strategy_intraday.md): at
short horizons the spread-at-trade-time decides every verdict, and there is no
quotes table. This module defines the point-in-time contract and a guarded
Polygon fetcher. It is NOT YET RUNNABLE against history — Polygon quotes/trades
need a PAID tier; with no key it prints setup and touches no network. No data is
fabricated here.

Tables (additive; created on demand), matching bars_minute conventions + a
mandatory ``known_at`` (when OUR collector observed the row) so replay can never
see the future:

    quotes_nbbo(ticker, ts, bid_price, bid_size, ask_price, ask_size,
                bid_exchange, ask_exchange, tape, known_at, source)
    trades_tick(ticker, ts, price, size, exchange, conditions, tape,
                aggressor, known_at, source)

``aggressor`` (buy/sell-initiated) is the true order-flow feature; it is derived
by the quote rule against the prevailing NBBO (see ``classify_aggressor``), NOT
from bar volume. Bars are not order flow (Latent-Flow was rejected at -18.2 bps).
"""

from __future__ import annotations

import os
from pathlib import Path

from ..config import ROOT

# Polygon endpoints (paid tier for full history depth).
QUOTES_URL = "https://api.polygon.io/v3/quotes/{ticker}"       # NBBO
TRADES_URL = "https://api.polygon.io/v3/trades/{ticker}"       # tick trades

SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes_nbbo (
    ticker       VARCHAR,
    ts           TIMESTAMP,     -- SIP/exchange timestamp (UTC), point-in-time
    bid_price    DOUBLE,
    bid_size     INTEGER,
    ask_price    DOUBLE,
    ask_size     INTEGER,
    bid_exchange VARCHAR,
    ask_exchange VARCHAR,
    tape         VARCHAR,
    known_at     TIMESTAMP,     -- when OUR collector observed it
    source       VARCHAR,       -- 'polygon_nbbo'
    PRIMARY KEY (ticker, ts)
);
CREATE TABLE IF NOT EXISTS trades_tick (
    ticker     VARCHAR,
    ts         TIMESTAMP,       -- SIP/exchange timestamp (UTC)
    price      DOUBLE,
    size       INTEGER,
    exchange   VARCHAR,
    conditions VARCHAR,
    tape       VARCHAR,
    aggressor  VARCHAR,         -- 'BUY' | 'SELL' | NULL (quote-rule / Lee-Ready)
    known_at   TIMESTAMP,
    source     VARCHAR,         -- 'polygon_trades'
    PRIMARY KEY (ticker, ts, price, size, exchange)
);
"""


def ensure_tables(con) -> None:
    """Create the quotes/trades tables if absent (additive, non-destructive)."""
    con.execute(SCHEMA)


def _key() -> str | None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("POLYGON_API_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("POLYGON_API_KEY")


def classify_aggressor(price: float, bid: float, ask: float,
                       prev_price: float | None = None) -> str | None:
    """Quote rule (+ Lee-Ready tick-test fallback at the midpoint). Pure/testable.

    BUY-initiated if the trade prints at/above the ask (or above the midpoint);
    SELL-initiated if at/below the bid (or below the midpoint); at the midpoint,
    fall back to the tick test vs the previous trade price. Returns None if the
    NBBO is crossed/locked or inputs are missing.
    """
    if bid is None or ask is None or ask <= bid:
        return None
    mid = (bid + ask) / 2.0
    if price >= ask or price > mid:
        return "BUY"
    if price <= bid or price < mid:
        return "SELL"
    # exactly at midpoint -> Lee-Ready tick test
    if prev_price is not None:
        if price > prev_price:
            return "BUY"
        if price < prev_price:
            return "SELL"
    return None


def status() -> dict:
    """Report readiness without touching the network."""
    has_key = _key() is not None
    return {
        "module": "ingest/nbbo",
        "purpose": "point-in-time NBBO quotes + tick trades for intraday research",
        "tables": ["quotes_nbbo", "trades_tick"],
        "endpoints": {"quotes": QUOTES_URL, "trades": TRADES_URL},
        "polygon_key_present": has_key,
        "state": "READY_TO_BACKFILL" if has_key else "DATA_GATED_NO_KEY_OR_TIER",
        "note": ("Polygon quotes/trades require a PAID tier for history depth. "
                 "No key -> no network, no data. Backfill is intentionally NOT "
                 "implemented until the paid tier is provisioned; no intraday "
                 "result may be computed from bars in the meantime."),
    }


def _setup() -> None:
    print("NBBO/trades ingest is DATA-GATED. To enable intraday research:\n"
          "  1. Provision a PAID Polygon plan with quotes+trades history.\n"
          "  2. Put POLYGON_API_KEY=... in a gitignored .env at the repo root.\n"
          "  3. Then implement the guarded backfill (v3/quotes, v3/trades) writing\n"
          "     quotes_nbbo / trades_tick with known_at, and build the fill\n"
          "     simulator BEFORE any signal (docs/strategy_intraday.md).")


def backfill(*_args, **_kwargs) -> dict:
    """Guarded entry point. Refuses to run without a key; never fabricates data."""
    if _key() is None:
        _setup()
        return {"error": "no POLYGON_API_KEY", "state": "DATA_GATED"}
    # Intentionally not implemented until the paid tier is confirmed. Writing this
    # against the free/insufficient tier would silently produce partial history.
    return {"error": "backfill not implemented pending paid quotes/trades tier",
            "state": "READY_TO_IMPLEMENT"}


if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2))
