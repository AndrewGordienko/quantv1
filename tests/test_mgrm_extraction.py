"""Tests for MGRM extraction hardening: provider-agnostic backend, HTML-table
extraction before prose, and the frozen gold-set accuracy audit."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from quantv1.ingest import guidance, guidance_goldset


class ProviderAgnosticConfigTests(unittest.TestCase):
    def test_no_backend_fails_closed(self):
        with patch.dict("os.environ", {"MGRM_LLM_PROVIDER": "none"}, clear=False):
            self.assertIsNone(guidance.llm_config())
            self.assertIsNone(guidance.ai_extract("anything"))
        self.assertEqual(guidance.provider_tag(None), "none:none")

    def test_openai_and_ollama_routing_recorded(self):
        with patch.dict("os.environ", {"MGRM_LLM_PROVIDER": "openai",
                                       "OPENAI_API_KEY": "sk-test",
                                       "MGRM_LLM_MODEL": "gpt-x"}, clear=False):
            config = guidance.llm_config()
            self.assertEqual(config["provider"], "openai")
            self.assertEqual(guidance.provider_tag(config), "openai:gpt-x")
        with patch.dict("os.environ", {"MGRM_LLM_PROVIDER": "ollama",
                                       "MGRM_OLLAMA_MODEL": "llama3.1"}, clear=False):
            config = guidance.llm_config()
            self.assertEqual(config["provider"], "ollama")
            self.assertIsNone(config["api_key"])
            self.assertIn("11434", config["base_url"])
            self.assertEqual(guidance.provider_tag(config), "ollama:llama3.1")

    def test_unknown_provider_raises(self):
        with patch.dict("os.environ", {"MGRM_LLM_PROVIDER": "bogus"}, clear=False):
            with self.assertRaises(ValueError):
                guidance.llm_config()


class TableExtractionTests(unittest.TestCase):
    OUTLOOK = ("<p>The company is providing guidance for the fourth quarter of "
               "fiscal 2024:</p><table><tr><th></th><th>Q4 FY2024 Guidance</th></tr>"
               "<tr><td>Revenue</td><td>$1.20 billion to $1.25 billion</td></tr>"
               "<tr><td>Diluted EPS</td><td>$1.10 to $1.20</td></tr></table>")

    def test_table_rows_are_extracted_with_ranges(self):
        records = {(r["metric"], r["guidance_period"]): r
                   for r in guidance.extract_tables(self.OUTLOOK)}
        self.assertIn(("revenue", "Q4-2024"), records)
        revenue = records[("revenue", "Q4-2024")]
        self.assertAlmostEqual(revenue["lower_value"], 1.20e9)
        self.assertAlmostEqual(revenue["upper_value"], 1.25e9)
        self.assertEqual(revenue["source_kind"], "table")
        eps = records[("eps", "Q4-2024")]
        self.assertEqual(eps["units"], "per_share")

    def test_non_guidance_table_is_ignored(self):
        table = ("<p>Condensed balance sheet</p><table>"
                 "<tr><td>Revenue</td><td>$5.0 billion</td></tr></table>")
        self.assertEqual(guidance.extract_tables(table), [])

    def test_structured_extract_prefers_table_over_prose(self):
        records = {(r["metric"], r["guidance_period"]): r
                   for r in guidance.structured_extract(self.OUTLOOK)}
        self.assertEqual(records[("revenue", "Q4-2024")]["source_kind"], "table")


class GoldSetAuditTests(unittest.TestCase):
    def test_seed_extractor_matches_labels_but_is_uncertified(self):
        result = guidance_goldset.audit()
        # The parser correctly matches the synthetic labels (machinery ok)...
        machinery = result["evaluations"]["synthetic_machinery"]
        self.assertEqual(machinery["detection"]["precision"], 1.0)
        self.assertEqual(machinery["detection"]["recall"], 1.0)
        for field in ("period", "units", "range", "action"):
            self.assertEqual(machinery["field_accuracy"][field], 1.0)
        # ...but every seed doc is synthetic, so there are zero real documents
        # and it cannot certify.
        self.assertEqual(result["real_documents"], 0)
        self.assertEqual(result["status"], "GOLDSET_TOO_SMALL")
        self.assertFalse(result["certified"])
        self.assertFalse(result["pilot_justified"])

    def test_missing_prediction_counts_as_false_negative(self):
        gold = [{"doc_id": "x", "sector": "S", "format": "prose",
                 "document_html": "<p>No forward statements here.</p>",
                 "expected": [{"metric": "revenue", "period": "FY2025",
                               "units": "absolute", "low": 1.0, "high": 2.0}]}]
        result = guidance_goldset.audit(gold)
        self.assertEqual(result["detection"]["fn"], 1)
        self.assertEqual(result["detection"]["recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
