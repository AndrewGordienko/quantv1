"""Tests for the frozen forced-flow S&P 500 census (ingest.forced_flow).

Covers the audit guarantees the flagship rests on:
  * hash integrity   -- a drifted discovery file is refused
  * no fabrication   -- announcement_time and knowledge_time are never invented
  * effective-only   -- the whole corpus is EFFECTIVE_DATE_ONLY
  * coverage kept    -- uncovered legs stay in the census, marked, not dropped
  * batching         -- co-dated legs share an event_batch_id; power = batches
  * change_type      -- deterministic quarterly-vs-ad-hoc split
  * idempotency      -- rebuilding the version does not duplicate rows
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quantv1.db import connect
from quantv1.ingest import forced_flow as ff


class PureFunctionTests(unittest.TestCase):
    def test_change_type_quarterly_vs_ad_hoc(self):
        # 2024-09-20 is the 3rd Friday of Sep -> quarterly rebalance window
        self.assertEqual(ff._change_type(date(2024, 9, 20)), "QUARTERLY_REBALANCE")
        # a February date can never be a quarterly rebalance
        self.assertEqual(ff._change_type(date(2024, 2, 15)), "AD_HOC")

    def test_legs_explode_and_share_batch_id(self):
        rows = [{"date": "2024-09-20", "add": "AAA,BBB", "remove": "CCC"}]
        legs = ff._legs(rows)
        adds = [l for l in legs if l["event_type"] == "addition"]
        self.assertEqual({l["ticker"] for l in adds}, {"AAA", "BBB"})
        self.assertEqual(len({l["batch_id"] for l in legs}), 1)      # one date, one batch
        self.assertEqual(adds[0]["batch_size"], 2)

    def test_event_id_is_deterministic_and_side_specific(self):
        a = ff._event_id(date(2024, 9, 20), "AAA", "addition")
        self.assertEqual(a, ff._event_id(date(2024, 9, 20), "AAA", "addition"))
        self.assertNotEqual(a, ff._event_id(date(2024, 9, 20), "AAA", "deletion"))

    def test_frozen_file_matches_pinned_hash(self):
        rows = ff._load_frozen(ff.FROZEN_CSV)     # real frozen copy; must verify
        self.assertGreater(len(rows), 100)

    def test_drifted_discovery_file_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "changes.csv"
            tampered.write_text("date,add,remove\n2024-09-20,AAA,BBB\n")
            with self.assertRaises(ff.CensusIntegrityError):
                ff._load_frozen(str(tampered))


class BuildIntegrationTests(unittest.TestCase):
    def test_build_freezes_corpus_without_fabricating_times(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.duckdb"
            gold = Path(directory) / "gold"
            gold.mkdir()
            with patch("quantv1.db.DB_PATH", db_path), \
                 patch.object(ff, "GOLDSET_DIR", str(gold)), \
                 patch.object(ff, "DATA_DIR", Path(directory)):
                import csv
                # Give exactly one real added ticker (the first) full price coverage.
                with open(ff.FROZEN_CSV) as handle:
                    sample = next(r["add"] for r in csv.DictReader(handle)
                                  if r["add"]).split(",")[0]
                con = connect()
                d0 = date(2019, 1, 1)
                con.executemany(
                    "INSERT INTO prices (ticker, date, open, high, low, close, volume) "
                    "VALUES (?,?,?,?,?,?,?)",
                    [(sample, d0 + timedelta(days=i), 10.0, 10.1, 9.9, 10.0, 1e6)
                     for i in range(400)])
                con.close()

                summary = ff.build(verbose=False)

                con = connect(read_only=True)
                V = ff.CENSUS_VERSION
                total = con.execute(
                    "SELECT COUNT(*) FROM forced_flow_events WHERE version=?", [V]
                ).fetchone()[0]
                fabricated = con.execute(
                    "SELECT COUNT(*) FROM forced_flow_events WHERE version=? AND "
                    "(announcement_time IS NOT NULL OR knowledge_time IS NOT NULL)", [V]
                ).fetchone()[0]
                statuses = {r[0] for r in con.execute(
                    "SELECT DISTINCT timestamp_status FROM forced_flow_events "
                    "WHERE version=?", [V]).fetchall()}
                cov = dict(con.execute(
                    "SELECT coverage_status, COUNT(*) FROM forced_flow_events "
                    "WHERE version=? AND ticker=? GROUP BY 1", [V, sample]).fetchall())
                con.close()

                self.assertGreater(total, 100)
                self.assertEqual(fabricated, 0)                     # nothing invented
                self.assertEqual(statuses, {"EFFECTIVE_DATE_ONLY"})
                # the covered ticker is COVERED; uncovered legs remain in the census
                self.assertIn("COVERED", cov)
                self.assertLess(summary["additions"]["coverage"].get("COVERED", 0),
                                summary["additions"]["legs"])       # some UNAVAILABLE kept
                # power reported in batches, and batches < legs
                self.assertLess(summary["additions"]["event_batches"],
                                summary["additions"]["legs"])

                # idempotent rebuild
                ff.build(verbose=False)
                con = connect(read_only=True)
                total2 = con.execute(
                    "SELECT COUNT(*) FROM forced_flow_events WHERE version=?", [V]
                ).fetchone()[0]
                con.close()
                self.assertEqual(total, total2)


if __name__ == "__main__":
    unittest.main()
