"""Frozen, auditable S&P 500 addition/deletion census (forced-flow flagship).

The discovery source (fja05680/sp500 changes-since-2019 list) is treated as a
DISCOVERY source, not authoritative point-in-time evidence. We pin its commit
SHA and verify the raw-file sha256 on every run, store the market
``effective_date`` separately from ``knowledge_time``, and NEVER fabricate an
announcement time. Every leg is kept in the census; legs without market data are
marked ``MARKET_DATA_UNAVAILABLE`` rather than dropped. Additions co-dated on one
effective date share an ``event_batch_id`` — inference downstream must cluster by
effective date, because 161 legs are only 117 independent date-batches.

Timestamp discipline: with no verified public announcement time, every event is
``EFFECTIVE_DATE_ONLY``. Effective-close pressure and post-effective reversal are
testable; announcement->effective continuation is NOT leak-free until a verified
announcement timestamp exists.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import csv
import hashlib
import json

from ..config import DATA_DIR
from ..db import connect

CENSUS_VERSION = "forced-flow-sp500-census-v1"
INDEX_FAMILY = "S&P"
INDEX_NAME = "S&P 500"
SOURCE_REPO = "fja05680/sp500"
SOURCE_COMMIT = "c403a121c2e766840f34837738cdd4725eeda818"
SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/"
    f"{SOURCE_COMMIT}/sp500_changes_since_2019.csv"
)
FROZEN_CSV = "goldset/forced_flow/sp500_changes_since_2019.csv"
EXPECTED_SHA256 = "7766d1603ceacea9715feed55e0383b95f44903d2452e33bf73cca4d4381c036"

COVERAGE_MIN_BARS = 20
COVERAGE_WINDOW_DAYS = 30
GOLDSET_DIR = "goldset/forced_flow"


class CensusIntegrityError(RuntimeError):
    """Raised when the frozen discovery file drifts from its pinned hash."""


def _third_friday(year: int, month: int) -> date:
    fridays = [date(year, month, day) for day in range(1, 29)
               if date(year, month, day).weekday() == 4]
    return fridays[2]


def _change_type(effective: date) -> str:
    """Deterministic, clearly-inferred split of scheduled vs ad-hoc changes."""
    if effective.month in (3, 6, 9, 12):
        if abs((effective - _third_friday(effective.year, effective.month)).days) <= 4:
            return "QUARTERLY_REBALANCE"
    return "AD_HOC"


def _load_frozen(path: str) -> list[dict]:
    with open(path, "rb") as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise CensusIntegrityError(
            f"{path} sha256 {digest} != pinned {EXPECTED_SHA256}; the discovery "
            "source drifted. Re-pin deliberately, do not ingest silently.")
    text = raw.decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def _legs(rows: list[dict]) -> list[dict]:
    """Explode date rows into per-ticker addition/deletion legs with batch ids."""
    legs = []
    for row in rows:
        effective = date.fromisoformat(row["date"])
        batch_id = f"SP500|{row['date']}"
        adds = [t.strip() for t in (row.get("add") or "").split(",") if t.strip()]
        removes = [t.strip() for t in (row.get("remove") or "").split(",") if t.strip()]
        for ticker in adds:
            legs.append({"effective": effective, "batch_id": batch_id,
                         "event_type": "addition", "ticker": ticker,
                         "batch_size": len(adds)})
        for ticker in removes:
            legs.append({"effective": effective, "batch_id": batch_id,
                         "event_type": "deletion", "ticker": ticker,
                         "batch_size": len(removes)})
    return legs


def _coverage(con, ticker: str, effective: date) -> str:
    n = con.execute(
        "SELECT COUNT(*) FROM prices WHERE ticker=? AND date BETWEEN "
        "CAST(? AS DATE)-? AND CAST(? AS DATE)+?",
        [ticker, effective, COVERAGE_WINDOW_DAYS, effective, COVERAGE_WINDOW_DAYS]
    ).fetchone()[0]
    return "COVERED" if n >= COVERAGE_MIN_BARS else "MARKET_DATA_UNAVAILABLE"


def _event_id(effective: date, ticker: str, event_type: str) -> str:
    raw = f"{CENSUS_VERSION}|{effective.isoformat()}|{ticker}|{event_type}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def build(verbose: bool = True) -> dict:
    """Ingest and FREEZE the addition/deletion census. No returns computed here."""
    rows = _load_frozen(FROZEN_CSV)
    legs = _legs(rows)
    con = connect()
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    records = []
    for leg in legs:
        coverage = _coverage(con, leg["ticker"], leg["effective"])
        records.append({
            **leg,
            "forced_flow_event_id": _event_id(leg["effective"], leg["ticker"],
                                              leg["event_type"]),
            "change_type": _change_type(leg["effective"]),
            "coverage_status": coverage,
        })

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM forced_flow_events WHERE version=?", [CENSUS_VERSION])
        for rec in records:
            metadata = {
                "source_repo": SOURCE_REPO,
                "change_type_inferred": True,
                "batch_size": rec["batch_size"],
            }
            con.execute("""
                INSERT INTO forced_flow_events
                    (forced_flow_event_id, index_family, index_name, event_type,
                     ticker, announcement_time, effective_date, knowledge_time,
                     event_batch_id, batch_size, change_type, change_reason,
                     timestamp_status, coverage_status, historical_ticker,
                     source, source_url, source_commit, source_sha256, version,
                     first_seen_at, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT DO NOTHING
            """, [rec["forced_flow_event_id"], INDEX_FAMILY, INDEX_NAME,
                  rec["event_type"], rec["ticker"], None, rec["effective"],
                  None,                                   # knowledge_time: never fabricated
                  rec["batch_id"], rec["batch_size"], rec["change_type"],
                  "UNVERIFIED", "EFFECTIVE_DATE_ONLY", rec["coverage_status"],
                  rec["ticker"], SOURCE_REPO, SOURCE_URL, SOURCE_COMMIT,
                  EXPECTED_SHA256, CENSUS_VERSION, now, json.dumps(metadata)])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        con.close()
        raise
    con.close()

    summary = _summarize(records)
    _freeze(records, summary, verbose)
    if verbose:
        a = summary["additions"]
        print(f"forced-flow census {CENSUS_VERSION}: "
              f"{a['legs']} addition legs across {a['unique_effective_dates']} "
              f"effective-date batches; {a['unique_tickers']} unique tickers; "
              f"covered {a['coverage']['COVERED']}, "
              f"unavailable {a['coverage'].get('MARKET_DATA_UNAVAILABLE', 0)}")
    return summary


def _side(records: list[dict], event_type: str) -> dict:
    side = [r for r in records if r["event_type"] == event_type]
    by_year, by_change, by_cov = {}, {}, {}
    for r in side:
        y = str(r["effective"].year)
        by_year[y] = by_year.get(y, 0) + 1
        by_change[r["change_type"]] = by_change.get(r["change_type"], 0) + 1
        by_cov[r["coverage_status"]] = by_cov.get(r["coverage_status"], 0) + 1
    return {
        "legs": len(side),
        "unique_tickers": len({r["ticker"] for r in side}),
        "unique_effective_dates": len({r["effective"] for r in side}),
        "event_batches": len({r["batch_id"] for r in side}),
        "by_year": dict(sorted(by_year.items())),
        "by_change_type": by_change,
        "coverage": by_cov,
    }


def _summarize(records: list[dict]) -> dict:
    return {
        "census_version": CENSUS_VERSION,
        "index": INDEX_NAME,
        "source": {"repo": SOURCE_REPO, "commit": SOURCE_COMMIT,
                   "url": SOURCE_URL, "sha256": EXPECTED_SHA256,
                   "role": "DISCOVERY_SOURCE_NOT_POINT_IN_TIME"},
        "window": {"start": min(r["effective"] for r in records).isoformat(),
                   "end": max(r["effective"] for r in records).isoformat()},
        "timestamp_status": "EFFECTIVE_DATE_ONLY",
        "tradability": {
            # Daily OHLC supports effective-day and multi-day-reversal returns.
            # It does NOT support closing-auction / last-hour pressure -- that
            # needs minute or auction data for these tickers.
            "testable_now_daily_bars": ["effective_day_return", "d1_d5_reversal"],
            "blocked_needs_intraday_coverage": ["closing_auction_pressure"],
            "blocked_needs_verified_announcement_time": ["announcement_to_effective_continuation"],
        },
        "additions": _side(records, "addition"),
        "deletions": _side(records, "deletion"),
    }


def _freeze(records: list[dict], summary: dict, verbose: bool) -> None:
    # Content-addressable freeze over the sorted accepted corpus + coverage ledger.
    canonical = sorted(
        ({"effective": r["effective"].isoformat(), "ticker": r["ticker"],
          "event_type": r["event_type"], "batch_id": r["batch_id"],
          "change_type": r["change_type"], "coverage_status": r["coverage_status"]}
         for r in records),
        key=lambda r: (r["effective"], r["event_type"], r["ticker"]))
    census_sha = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    summary["census_sha256"] = census_sha

    ledger = [r for r in canonical if r["coverage_status"] != "COVERED"]
    for name, payload in (
        (f"{GOLDSET_DIR}/census_manifest_v1.json", {**summary, "corpus": canonical}),
        (f"{GOLDSET_DIR}/coverage_ledger_v1.json",
         {"census_version": CENSUS_VERSION, "reason": "MARKET_DATA_UNAVAILABLE",
          "note": "kept in census, not tradable; no minute/daily bars in window",
          "count": len(ledger), "events": ledger}),
    ):
        with open(name, "w") as file:
            json.dump(payload, file, indent=2)
    # A convenience copy of the summary in data/ (gitignored) for quick reads.
    with open(DATA_DIR / "forced_flow_census.json", "w") as file:
        json.dump(summary, file, indent=2)
    if verbose:
        print(f"frozen census_sha256: {census_sha}")


if __name__ == "__main__":
    build()
