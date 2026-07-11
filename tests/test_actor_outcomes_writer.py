"""Tests for the actor_event_outcomes writer (research.actor_outcomes).

These cover the persistence layer and its integrity guarantees, distinct from
tests/test_actor_outcomes.py which exercises the underlying _outcome engine:

  * idempotency  -- rebuilding a version does not duplicate rows
  * versioning   -- a second outcome_version coexists; rebuilding one replaces
                    only its own rows
  * no leakage   -- pre-event beta is invariant to post-event price corruption
  * primary-only -- merely_mentioned events never produce outcomes
  * hedge map    -- sector ETFs / broad proxies hedge vs SPY; SPY self-hedge is
                    skipped; single names hedge vs their sector ETF
  * timestamp gate -- the upstream manifest ingester the outcomes depend on
                    rejects page-date precision (exact public times only)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quantv1.db import connect
from quantv1.research import actor_outcomes as ao

_SESSION_START = datetime(2025, 3, 3, 13, 30)   # UTC regular session open
_N_BARS = 390
_EVENT_INDEX = 200


def _prices(rng, n, beta, base, resid_sd, market_ret):
    ret = beta * market_ret + rng.normal(0, resid_sd, n)
    return base * np.cumprod(1.0 + ret)


def _seed(con):
    """Insert one session of synthetic bars plus one primary + one mention event."""
    rng = np.random.default_rng(7)
    stamps = [_SESSION_START + timedelta(minutes=i) for i in range(_N_BARS)]
    market_ret = rng.normal(0, 0.0006, _N_BARS)
    series = {
        "SPY": 100.0 * np.cumprod(1.0 + market_ret),
        "NVDA": _prices(rng, _N_BARS, 1.30, 50.0, 0.0004, market_ret),
        "XLK": _prices(rng, _N_BARS, 1.05, 40.0, 0.0002, market_ret),
    }
    rows = []
    for ticker, close in series.items():
        for i, ts in enumerate(stamps):
            c = float(close[i])
            rows.append((ticker, ts, c, c * 1.001, c * 0.999, c, 1000.0))
    con.executemany(
        "INSERT INTO bars_minute (ticker, ts, open, high, low, close, volume) "
        "VALUES (?,?,?,?,?,?,?)", rows)
    con.execute("INSERT INTO ticker_sectors (ticker, sector) VALUES ('NVDA','Technology')")

    public_time = stamps[_EVENT_INDEX] - timedelta(seconds=1)
    now = datetime(2025, 3, 4, 0, 0)
    con.executemany("""
        INSERT INTO actor_events
            (actor_event_id, actor_id, ticker, public_time, event_type, headline,
             source, first_seen_at, actor_event_role, role_confidence,
             primary_hypothesis_eligible, extraction_version, metadata)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        ("evt_primary", "jerome_powell", "NVDA", public_time, "speech", "Remarks",
         "fed-primary-v1", now, "speaker_author", 1.0, True, "fed-primary-v1", "{}"),
        ("evt_mention", "jerome_powell", "NVDA", public_time, "news", "Powell mentioned",
         "news-v3", now, "merely_mentioned", 0.4, False, "news-context-v3", "{}"),
    ])


def _count_by_version(con):
    return dict(con.execute(
        "SELECT outcome_version, COUNT(*) FROM actor_event_outcomes "
        "GROUP BY outcome_version").fetchall())


class HedgeMapTests(unittest.TestCase):
    def _panel(self, present):
        return types.SimpleNamespace(has=lambda t, present=present: t in present)

    def test_sector_etfs_and_broad_proxies_hedge_vs_spy(self):
        panel = self._panel({"SPY", "XLF", "QQQ", "XLK"})
        self.assertEqual(ao._hedge_benchmark("XLF", {}, panel), "SPY")
        self.assertEqual(ao._hedge_benchmark("QQQ", {}, panel), "SPY")

    def test_spy_self_hedge_is_skipped(self):
        panel = self._panel({"SPY"})
        self.assertIsNone(ao._hedge_benchmark("SPY", {}, panel))

    def test_single_name_hedges_vs_sector_etf_then_spy(self):
        with_sector = self._panel({"SPY", "XLK"})
        self.assertEqual(
            ao._hedge_benchmark("NVDA", {"NVDA": "technology"}, with_sector), "XLK")
        no_sector = self._panel({"SPY"})   # sector ETF absent -> fall back to SPY
        self.assertEqual(
            ao._hedge_benchmark("NVDA", {"NVDA": "technology"}, no_sector), "SPY")


class WriterIntegrationTests(unittest.TestCase):
    def _run(self, fn):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.duckdb"
            with patch("quantv1.db.DB_PATH", db_path), \
                 patch.object(ao, "DATA_DIR", Path(directory)):
                con = connect()
                _seed(con)
                con.close()
                return fn(db_path, directory)

    def test_writes_only_primary_events_and_is_idempotent(self):
        def check(db_path, directory):
            first = ao.build(verbose=False)
            con = connect(read_only=True)
            rows = con.execute(
                "SELECT DISTINCT actor_event_id FROM actor_event_outcomes").fetchall()
            con.close()
            # exactly the primary event, three horizons, no mention leakage
            self.assertEqual({r[0] for r in rows}, {"evt_primary"})
            self.assertEqual(first["rows_written"], len(ao.HORIZONS_MINUTES))
            second = ao.build(verbose=False)
            con = connect(read_only=True)
            total = con.execute("SELECT COUNT(*) FROM actor_event_outcomes").fetchone()[0]
            con.close()
            self.assertEqual(second["rows_written"], first["rows_written"])
            self.assertEqual(total, len(ao.HORIZONS_MINUTES))   # no duplication
        self._run(check)

    def test_versioning_replaces_only_matching_version(self):
        def check(db_path, directory):
            ao.build(outcome_version="A", verbose=False)
            ao.build(outcome_version="B", verbose=False)
            con = connect(read_only=True)
            counts = _count_by_version(con)
            con.close()
            self.assertEqual(set(counts), {"A", "B"})
            self.assertEqual(counts["A"], len(ao.HORIZONS_MINUTES))
            # rebuilding A must not touch B and must not duplicate A
            ao.build(outcome_version="A", verbose=False)
            con = connect(read_only=True)
            recounts = _count_by_version(con)
            con.close()
            self.assertEqual(recounts, counts)
        self._run(check)

    def test_pre_event_beta_is_invariant_to_post_event_prices(self):
        def check(db_path, directory):
            ao.build(outcome_version="clean", verbose=False)
            con = connect()
            baseline = con.execute("""
                SELECT market_beta_residual,
                       json_extract(metadata,'$.market_beta') AS beta, raw_return
                FROM actor_event_outcomes
                WHERE outcome_version='clean' AND horizon='30m'
            """).fetchone()
            # Corrupt every bar strictly AFTER the entry bar; the pre-event beta
            # window and the entry open must be untouched, so beta cannot move
            # but the horizon exit price (and thus raw_return) must.
            entry_ts = _SESSION_START + timedelta(minutes=_EVENT_INDEX)
            con.execute(
                "UPDATE bars_minute SET close=close*10, open=open*10 WHERE ts>?",
                [entry_ts])
            con.close()
            ao.build(outcome_version="corrupt", verbose=False)
            con = connect(read_only=True)
            corrupted = con.execute("""
                SELECT json_extract(metadata,'$.market_beta') AS beta, raw_return
                FROM actor_event_outcomes
                WHERE outcome_version='corrupt' AND horizon='30m'
            """).fetchone()
            con.close()
            self.assertIsNotNone(baseline[1])
            # beta identical (no post-event leakage)...
            self.assertAlmostEqual(float(baseline[1]), float(corrupted[0]), places=9)
            # ...while the horizon return genuinely changed (corruption took effect)
            self.assertNotAlmostEqual(float(baseline[2]), float(corrupted[1]), places=6)
        self._run(check)


class TimestampGateTests(unittest.TestCase):
    """The outcomes rest on the strict manifest gate; assert it here too."""

    def test_manifest_rejects_page_date_precision(self):
        from quantv1.ingest.fed_primary import ManifestError, _validate
        from tests.test_fed_primary import _record
        record = _record()
        record["timestamp_precision"] = "date_only"
        with self.assertRaises(ManifestError):
            _validate(record)


if __name__ == "__main__":
    unittest.main()
