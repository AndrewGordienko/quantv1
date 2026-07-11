"""Tests for the announcement-timestamp corpus (ingest.forced_flow_announcements).

Covers the frozen-source discipline that unblocks the continuation test:
  * blindness      -- the module never touches prices / returns
  * worklist       -- enumerates ALL addition batches, all UNRESOLVED
  * tier rules     -- date-only / bad-tier / no-provenance / off-census rejected
  * accept + entry -- a valid Tier-1 record verifies and gets a next-open entry
  * freeze         -- ingest freezes a manifest + rejection ledger; unresolved = N - verified
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quantv1.db import connect
from quantv1.ingest import forced_flow as ff
from quantv1.ingest import forced_flow_announcements as fa


def _batches():
    return {"SP500|2024-09-20": {"effective_date": "2024-09-20",
                                 "tickers": ["AAA", "BBB"], "covered_tickers": ["AAA"]}}


def _record(**over):
    rec = {
        "event_batch_id": "SP500|2024-09-20",
        "announcement_public_time": "2024-09-13T16:15:00-04:00",
        "announcement_timezone": "America/New_York",
        "source_tier": 1,
        "source_url": "https://www.spglobal.com/spdji/en/example.pdf",
        "source_sha256": "a" * 64,
        "original_or_correction": "ORIGINAL",
        "affected_tickers": ["AAA"],
        "timestamp_precision": "exact_minute",
    }
    rec.update(over)
    return rec


class BlindnessTests(unittest.TestCase):
    def test_module_never_references_prices_or_returns(self):
        # Strip the docstring (which legitimately describes the blindness intent
        # in prose) and check for actual price/return CODE tokens.
        import ast
        src = Path(fa.__file__).read_text()
        body_wo_doc = src.replace(ast.get_docstring(ast.parse(src)) or "", "")
        for forbidden in ("bars_minute", "raw_return", "sector_beta",
                          "FROM prices", "from ..model", "read_csv"):
            self.assertNotIn(forbidden, body_wo_doc,
                             f"announcement sourcing must stay blind: {forbidden!r}")


class ValidatorTests(unittest.TestCase):
    def test_accepts_valid_tier1_and_sets_next_open_entry(self):
        row = fa.validate_record(_record(), _batches())
        self.assertEqual(row["verification_status"], "VERIFIED")
        self.assertEqual(row["source_tier"], 1)
        # 16:15 ET is after the close -> next session open
        self.assertEqual(row["first_executable_time"], "AFTER_HOURS_NEXT_SESSION_OPEN")

    def test_rejects_date_only_precision(self):
        with self.assertRaises(fa.AnnouncementRejection) as ctx:
            fa.validate_record(_record(timestamp_precision="date_only"), _batches())
        self.assertEqual(ctx.exception.reason_code, "DATE_ONLY")

    def test_rejects_time_without_offset(self):
        with self.assertRaises(fa.AnnouncementRejection) as ctx:
            fa.validate_record(
                _record(announcement_public_time="2024-09-13T16:15:00"), _batches())
        self.assertEqual(ctx.exception.reason_code, "DATE_ONLY")

    def test_rejects_bad_tier_and_missing_provenance(self):
        with self.assertRaises(fa.AnnouncementRejection) as ctx:
            fa.validate_record(_record(source_tier=4), _batches())
        self.assertEqual(ctx.exception.reason_code, "BAD_TIER")
        with self.assertRaises(fa.AnnouncementRejection) as ctx:
            fa.validate_record(_record(source_sha256=""), _batches())
        self.assertEqual(ctx.exception.reason_code, "NO_SOURCE_PROVENANCE")

    def test_rejects_ticker_not_in_batch_and_off_census(self):
        with self.assertRaises(fa.AnnouncementRejection) as ctx:
            fa.validate_record(_record(affected_tickers=["ZZZ"]), _batches())
        self.assertEqual(ctx.exception.reason_code, "TICKER_NOT_IN_BATCH")
        with self.assertRaises(fa.AnnouncementRejection) as ctx:
            fa.validate_record(_record(event_batch_id="SP500|1900-01-01"), _batches())
        self.assertEqual(ctx.exception.reason_code, "OFF_CENSUS")

    def test_intraday_classification(self):
        t = datetime(2024, 9, 20, 15, 0, tzinfo=timezone.utc)   # 15:00 UTC = mid-session
        self.assertEqual(fa.classify_entry(t), "INTRADAY_NEXT_SESSION_OPEN")


class WorklistFreezeTests(unittest.TestCase):
    def test_worklist_enumerates_all_batches_then_freezes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "test.duckdb"
            work = Path(directory) / "worklist.jsonl"
            manifest = Path(directory) / "manifest.jsonl"
            ledger = Path(directory) / "ledger.json"
            resolved = Path(directory) / "resolved.jsonl"
            with patch("quantv1.db.DB_PATH", db_path), \
                 patch.object(ff, "GOLDSET_DIR", str(directory)), \
                 patch.object(ff, "DATA_DIR", Path(directory)), \
                 patch.object(fa, "WORKLIST_PATH", str(work)), \
                 patch.object(fa, "MANIFEST_PATH", str(manifest)), \
                 patch.object(fa, "LEDGER_PATH", str(ledger)):
                connect().close()
                census = ff.build(verbose=False)
                work_summary = fa.generate_worklist(verbose=False)
                # worklist size == addition batches in the census
                self.assertEqual(work_summary["batches"],
                                 census["additions"]["event_batches"])
                # one valid resolved record for the first real batch
                resolved.write_text(
                    '{"event_batch_id":"SP500|2019-01-18",'
                    '"announcement_public_time":"2019-01-15T16:15:00-05:00",'
                    '"announcement_timezone":"America/New_York","source_tier":1,'
                    '"source_url":"https://www.spglobal.com/x.pdf",'
                    f'"source_sha256":"{"a"*64}","original_or_correction":"ORIGINAL",'
                    '"affected_tickers":["TFX"],"timestamp_precision":"exact_minute"}\n')
                summary = fa.ingest_resolved(str(resolved), verbose=False)
                self.assertEqual(summary["verified_batches"], 1)
                self.assertEqual(summary["unresolved"],
                                 summary["total_batches"] - 1)
                self.assertTrue(manifest.exists() and ledger.exists())

                # selection-bias comparison partitions the same denominator
                cmp = fa.coverage_comparison({"SP500|2019-01-18"})
                self.assertEqual(cmp["resolved"]["n_batches"]
                                 + cmp["unresolved"]["n_batches"],
                                 cmp["total_batches"])
                self.assertEqual(cmp["resolved"]["n_batches"], 1)


if __name__ == "__main__":
    unittest.main()
