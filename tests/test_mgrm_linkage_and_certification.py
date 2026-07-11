"""Tests for hardened previous-guidance linkage and extractor certification."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quantv1.ingest import guidance, guidance_goldset


def _t(day: int) -> datetime:
    return datetime(2024, 1, day, 12, 0, tzinfo=timezone.utc)


def _rec(eid, acc, period, mid, low, high, t, *, metric="revenue",
         status="AVAILABLE", action="UNSPECIFIED", ticker="T") -> dict:
    return {"extraction_id": f"{eid}:{acc}:{metric}:{period}:{t.day}",
            "earnings_event_id": eid, "accession": acc, "ticker": ticker,
            "metric": metric, "period": period, "midpoint": mid, "low": low,
            "high": high, "status": status, "action": action, "public_time": t}


def _by_id(links: list[dict]) -> dict[str, dict]:
    return {link["extraction_id"]: link for link in links}


class LinkageGuardrailTests(unittest.TestCase):
    def test_genuine_revision_across_events(self):
        rows = [_rec("E1", "A1", "FY2024", 100.0, 90.0, 110.0, _t(1)),
                _rec("E2", "A2", "FY2024", 110.0, 100.0, 120.0, _t(8))]
        links = _by_id(guidance.link_guidance_records(rows))
        later = links["E2:A2:revenue:FY2024:8"]
        self.assertEqual(later["link_status"], "LINKED")
        self.assertAlmostEqual(later["midpoint_revision"], 0.10)
        self.assertEqual(later["revision_classification"], "RAISED")

    def test_duplicate_exhibit_is_canonicalized_and_never_a_revision(self):
        # Two documents from the SAME filing (event+accession) for the same
        # metric/period: one canonical, one DUPLICATE_EXHIBIT, no revision.
        rows = [_rec("E1", "A1", "FY2024", 100.0, 90.0, 110.0, _t(1)),
                _rec("E1", "A1", "FY2024", 100.0, 90.0, 110.0, _t(1))]
        links = guidance.link_guidance_records(rows)
        statuses = sorted(link["link_status"] for link in links)
        self.assertEqual(statuses, ["DUPLICATE_EXHIBIT", "NO_PREVIOUS_GUIDANCE"])
        self.assertTrue(all(link["midpoint_revision"] is None for link in links))

    def test_same_filing_different_periods_not_cross_linked(self):
        # A multi-period outlook table in one filing must not link its own rows.
        rows = [_rec("E1", "A1", "FY2024", 100.0, 90.0, 110.0, _t(1)),
                _rec("E1", "A1", "Q1-2024", 25.0, 24.0, 26.0, _t(1))]
        links = guidance.link_guidance_records(rows)
        self.assertTrue(all(link["link_status"] == "NO_PREVIOUS_GUIDANCE"
                            for link in links))

    def test_unknown_period_is_never_linked(self):
        rows = [_rec("E1", "A1", "FYUNKNOWN", 100.0, 90.0, 110.0, _t(1)),
                _rec("E2", "A2", "PERIOD-UNKNOWN", 110.0, 100.0, 120.0, _t(8))]
        links = guidance.link_guidance_records(rows)
        self.assertTrue(all(link["link_status"] == "UNKNOWN_PERIOD_NOT_LINKABLE"
                            for link in links))

    def test_repeated_reaffirmations_carry_forward_zero_revision(self):
        rows = [_rec("E1", "A1", "FY2024", 100.0, 90.0, 110.0, _t(1)),
                _rec("E2", "A2", "FY2024", None, None, None, _t(8),
                     status="REAFFIRMED", action="REAFFIRMED"),
                _rec("E3", "A3", "FY2024", None, None, None, _t(15),
                     status="REAFFIRMED", action="REAFFIRMED")]
        links = _by_id(guidance.link_guidance_records(rows))
        for eid, acc, day in (("E2", "A2", 8), ("E3", "A3", 15)):
            link = links[f"{eid}:{acc}:revenue:FY2024:{day}"]
            self.assertEqual(link["link_status"], "LINKED")
            self.assertEqual(link["revision_classification"], "REAFFIRMED")
            self.assertEqual(link["midpoint_revision"], 0.0)

    def test_equal_timestamp_is_not_a_revision(self):
        rows = [_rec("E1", "A1", "FY2024", 100.0, 90.0, 110.0, _t(5)),
                _rec("E2", "A2", "FY2024", 110.0, 100.0, 120.0, _t(5))]
        links = _by_id(guidance.link_guidance_records(rows))
        later = links["E2:A2:revenue:FY2024:5"]
        self.assertEqual(later["link_status"], "NO_PREVIOUS_GUIDANCE")


class CertificationGateTests(unittest.TestCase):
    def test_audit_reports_three_evaluations_and_gates_on_reconciled(self):
        result = guidance_goldset.audit()
        self.assertIn("deterministic", result["evaluations"])
        self.assertIn("ai", result["evaluations"])
        self.assertIn("reconciled", result["evaluations"])
        self.assertEqual(result["certified_output"], "reconciled")
        # No AI backend -> deterministic parses but reconciled AGREED is empty.
        self.assertEqual(result["evaluations"]["deterministic"]["detection"]["recall"], 1.0)
        self.assertEqual(result["evaluations"]["reconciled"]["detection"]["recall"], 0.0)

    def test_synthetic_fixtures_do_not_count_as_real_documents(self):
        result = guidance_goldset.audit()
        self.assertEqual(result["real_documents"], 0)
        self.assertEqual(result["status"], "GOLDSET_TOO_SMALL")
        self.assertFalse(result["certified"])
        self.assertFalse(result["pilot_justified"])

    def test_certification_status_detects_absent_stale_and_wrong_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cert.json"
            with patch.object(guidance_goldset, "CERTIFICATION_PATH", path):
                self.assertEqual(guidance_goldset.certification_status()["reason"],
                                 "CERTIFICATION_ABSENT")
                valid = {"certified": True,
                         "goldset_sha256": guidance_goldset.goldset_sha256(),
                         "extractor_version": guidance_goldset.EXTRACTOR_VERSION,
                         "provider": guidance.provider_tag()}
                path.write_text(json.dumps(valid))
                self.assertTrue(guidance_goldset.certification_status()["certified"])
                path.write_text(json.dumps({**valid, "goldset_sha256": "deadbeef"}))
                self.assertEqual(guidance_goldset.certification_status()["reason"],
                                 "CERTIFICATION_STALE_GOLDSET")
                path.write_text(json.dumps({**valid, "provider": "openai:other"}))
                self.assertEqual(guidance_goldset.certification_status()["reason"],
                                 "CERTIFICATION_WRONG_PROVIDER")


if __name__ == "__main__":
    unittest.main()
