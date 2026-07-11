"""Tests for the MGRM labelling logic and corpus portability/rehydrate."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from quantv1 import labeling
from quantv1.ingest import mgrm_corpus


class LabelValidationTests(unittest.TestCase):
    def test_no_guidance_requires_empty_expected(self):
        self.assertEqual(labeling.validate_label({"no_guidance": True, "expected": []}), [])
        errors = labeling.validate_label({"no_guidance": True, "expected": [{"metric": "revenue"}]})
        self.assertTrue(any("empty expected" in e for e in errors))

    def test_guidance_requires_a_record(self):
        errors = labeling.validate_label({"no_guidance": False, "expected": []})
        self.assertTrue(any("at least one" in e for e in errors))

    def test_numeric_consistency_enforced(self):
        good = {"no_guidance": False, "expected": [{
            "metric": "revenue", "period": "FY2024", "units": "absolute",
            "low": 100, "high": 120, "midpoint": 110, "status": "AVAILABLE",
            "action": "RAISED", "evidence": "..."}]}
        self.assertEqual(labeling.validate_label(good), [])
        bad = json.loads(json.dumps(good))
        bad["expected"][0]["midpoint"] = 999  # not within [low, high] / not the mean
        errors = labeling.validate_label(bad)
        self.assertTrue(any("midpoint" in e for e in errors))

    def test_withdrawn_may_omit_numbers(self):
        errors = labeling.validate_label({"no_guidance": False, "expected": [{
            "metric": "revenue", "period": "FY2024", "units": "absolute",
            "status": "WITHDRAWN", "action": "WITHDRAWN", "evidence": "withdrew"}]})
        self.assertEqual(errors, [])

    def test_invalid_enum_rejected(self):
        errors = labeling.validate_label({"no_guidance": False, "expected": [{
            "metric": "not_a_metric", "period": "FY2024", "units": "absolute",
            "low": 1, "high": 2, "midpoint": 1.5, "status": "AVAILABLE",
            "action": "RAISED", "evidence": "x"}]})
        self.assertTrue(any("metric not in" in e for e in errors))


class PrefillIsolationTests(unittest.TestCase):
    MANIFEST = [
        {"document_id": "dev1", "company": "A", "sector": "Tech", "format_hint": "prose",
         "split": "development", "document_url": "u", "document_type": "EX-99.1",
         "raw_path": "data/raw/mgrm/x/y.html"},
        {"document_id": "cert1", "company": "B", "sector": "Energy", "format_hint": "table",
         "split": "sealed_certification", "document_url": "u", "document_type": "EX-99.1",
         "raw_path": "data/raw/mgrm/x/z.html"},
    ]

    def test_certification_never_exposes_prefill(self):
        prefill = {"dev1": [{"metric": "revenue"}], "cert1": [{"metric": "eps"}]}
        with patch.object(mgrm_corpus, "load_manifest", return_value=self.MANIFEST), \
                patch.object(labeling, "_prefill_index", return_value=prefill):
            self.assertIsNotNone(labeling.document_view("dev1")["prefill"])
            self.assertIsNone(labeling.document_view("cert1")["prefill"])


class ExportDeterminismTests(unittest.TestCase):
    def test_export_is_sorted_and_skips_invalid(self):
        manifest = [
            {"document_id": "b", "company": "B", "sector": "S", "format_hint": "prose",
             "split": "development", "document_url": "u"},
            {"document_id": "a", "company": "A", "sector": "S", "format_hint": "prose",
             "split": "development", "document_url": "u"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(mgrm_corpus, "load_manifest", return_value=manifest), \
                    patch.object(labeling, "DRAFT_DIR", root / "drafts"), \
                    patch.object(labeling, "DEV_LABELS_PATH", root / "dev.jsonl"), \
                    patch.object(labeling, "ROOT", root):
                labeling.save_draft("a", {"no_guidance": True, "expected": []})
                labeling.save_draft("b", {"no_guidance": True, "expected": []})
                result = labeling.export_jsonl("development")
                self.assertEqual(result["exported"], 2)
                lines = (root / "dev.jsonl").read_text().splitlines()
                ids = [json.loads(x)["doc_id"] for x in lines if not x.startswith("#")]
                self.assertEqual(ids, ["a", "b"])  # deterministic sort


class RehydrateRefusalTests(unittest.TestCase):
    def test_sha_mismatch_on_refetch_is_refused(self):
        manifest = [{"document_id": "d", "ticker": "T", "raw_path": "data/raw/mgrm/x/y.html",
                     "source_sha256": "expected", "primary_url": "http://x",
                     "accession_number": "acc"}]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(mgrm_corpus, "load_manifest", return_value=manifest), \
                    patch.object(mgrm_corpus, "ROOT", Path(directory)), \
                    patch.object(mgrm_corpus, "_reproduce", return_value=None):
                result = mgrm_corpus.rehydrate(verbose=False)
        self.assertEqual(result["refused"], 1)
        self.assertEqual(result["downloaded"], 0)
        self.assertFalse(result["reselected"])
        self.assertEqual(result["problems"][0]["reason"], "SHA_MISMATCH_ON_REFETCH")


class PortablePathTests(unittest.TestCase):
    def test_manifest_paths_are_repository_relative(self):
        for record in mgrm_corpus.load_manifest():
            self.assertFalse(record["raw_path"].startswith("/"))
            self.assertTrue(record["raw_path"].startswith("data/raw/mgrm/"))
            self.assertIn(record["accession_number"].replace("-", ""),
                          record["primary_url"])


if __name__ == "__main__":
    unittest.main()
