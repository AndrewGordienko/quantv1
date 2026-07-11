"""Tests for deterministic MGRM corpus selection (data-phase tooling)."""

from __future__ import annotations

import unittest

from quantv1.ingest import mgrm_corpus


def _pool(n_companies: int, sectors: int, docs_per: int = 3) -> list[dict]:
    documents = []
    for company in range(n_companies):
        ticker = f"T{company:03d}"
        sector = f"S{company % sectors}"
        for doc in range(docs_per):
            accession = f"{ticker}-{doc:04d}"
            documents.append({
                "document_id": f"{accession}-doc", "accession_number": accession,
                "ticker": ticker, "cik": str(company), "earnings_event_id": f"e{accession}",
                "document_type": "EX-99.1", "source_url": f"http://x/{accession}",
                "public_time": "2024-01-01", "source_sha256": accession * 2,
                "raw_path": f"/tmp/{accession}", "sector": sector,
                "format_hint": ["table", "prose", "mixed"][doc % 3],
                "selection_key": mgrm_corpus._key(accession),
            })
    return documents


class DeterministicSelectionTests(unittest.TestCase):
    def test_split_is_company_disjoint_and_sized(self):
        selection = mgrm_corpus.select_corpus(_pool(40, 8), dev_n=20, cert_n=30)
        dev, cert = selection["development"], selection["sealed_certification"]
        self.assertEqual(len(dev), 20)
        self.assertEqual(len(cert), 30)
        self.assertFalse(set(selection["dev_companies"]) &
                         set(selection["cert_companies"]))

    def test_selection_is_deterministic(self):
        pool = _pool(40, 8)
        first = mgrm_corpus.select_corpus(pool)
        second = mgrm_corpus.select_corpus(list(reversed(pool)))
        self.assertEqual([d["document_id"] for d in first["development"]],
                         [d["document_id"] for d in second["development"]])
        self.assertEqual([d["document_id"] for d in first["sealed_certification"]],
                         [d["document_id"] for d in second["sealed_certification"]])

    def test_sector_minimum_reported(self):
        distribution = mgrm_corpus.distribution(
            mgrm_corpus.select_corpus(_pool(40, 8)))
        self.assertTrue(distribution["meets_sector_minimum"])
        self.assertGreaterEqual(distribution["sectors_covered"], 6)
        self.assertTrue(distribution["company_disjoint"])

    def test_max_docs_per_company_capped(self):
        # One company with many docs must not dominate a split.
        selection = mgrm_corpus.select_corpus(_pool(40, 8, docs_per=10))
        counts = {}
        for document in selection["development"] + selection["sealed_certification"]:
            counts[document["ticker"]] = counts.get(document["ticker"], 0) + 1
        self.assertLessEqual(max(counts.values()), mgrm_corpus.MAX_DOCS_PER_COMPANY)


if __name__ == "__main__":
    unittest.main()
