from __future__ import annotations

import unittest
from unittest.mock import patch

import json
import tempfile
from pathlib import Path

from quantv1.ingest.sec_census import _classify, select_companies, _issuer_split, _write_goldset_skeleton
from quantv1.events.goldset import score_goldset


class SecCensusTests(unittest.TestCase):
    def test_selection_deduplicates_share_classes_by_cik(self):
        rows = [
            {"cik": "1", "ticker": "ZZZ", "title": "Example Corp"},
            {"cik": "1", "ticker": "AAA", "title": "Example Corp"},
            {"cik": "2", "ticker": "BBB", "title": "Other Corp"},
        ]
        with patch("quantv1.ingest.sec_census.fetch_entities", return_value=rows):
            got = select_companies(10)
        self.assertEqual({r["cik"] for r in got}, {"1", "2"})
        self.assertEqual(next(r["ticker"] for r in got if r["cik"] == "1"), "AAA")

    def test_classifier_retains_multiple_event_types(self):
        text = "The Company announced a merger, a cybersecurity incident, and a buyback authorization."
        types = {r["event_type"] for r in _classify(text, "1.01,1.05,5.03")}
        self.assertIn("merger_announced", types)
        self.assertIn("cyber_incident", types)
        self.assertIn("buyback_authorization", types)

    def test_goldset_has_controls_and_issuer_disjoint_partitions(self):
        events = [{"atlas_event_id": str(i), "event_type": "guidance_raised", "cik": str(i),
                   "accession_number": str(i), "exhibits": []} for i in range(6)]
        controls = [{"cik": "900", "accession_number": "c0"}]
        filings = [{"cik": str(i), "accession_number": f"f{i}"} for i in range(30)]
        with tempfile.TemporaryDirectory() as d:
            import quantv1.ingest.sec_census as census
            old = census.GOLDSET
            census.GOLDSET = Path(d) / "gold.jsonl"
            try:
                _write_goldset_skeleton(events, controls, filings, min_controls=20)
                rows = [json.loads(x) for x in census.GOLDSET.read_text().splitlines()]
            finally:
                census.GOLDSET = old
        self.assertGreaterEqual(sum(r["record_type"].startswith("routine_control") for r in rows), 20)
        issuers = {}
        for row in rows:
            cik = str(row["source"]["cik"])
            issuers.setdefault(cik, set()).add(row["issuer_split"])
        self.assertTrue(all(len(v) == 1 for v in issuers.values()))

    def test_goldset_scorer_leaves_unlabeled_pending(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gold.jsonl"
            p.write_text(json.dumps({"record_type": "event", "issuer_split": "sealed", "human_label": None}) + "\n")
            report = score_goldset(p)
        self.assertEqual(report["pending"], 1)
        self.assertIsNone(report["partitions"]["sealed"]["event_detection_precision"])


if __name__ == "__main__":
    unittest.main()
