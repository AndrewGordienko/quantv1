"""Tests for the four certification/power correctness fixes:

1. certification is code-sensitive (extractor implementation hash);
2. accuracy gates only on real (non-synthetic) reconciled output;
3. run()/extraction_audit() never double-apply guidance features;
4. power uses announcement dates, not entry dates.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from quantv1.ingest import guidance, guidance_goldset
from quantv1.research import mgrm


def _valid_cert() -> dict:
    return {"certified": True,
            "goldset_sha256": guidance_goldset.goldset_sha256(),
            "extractor_implementation_sha256":
                guidance_goldset.extractor_implementation_sha256(),
            "extractor_version": guidance_goldset.EXTRACTOR_VERSION,
            "provider": guidance.provider_tag()}


class ImplementationHashTests(unittest.TestCase):
    def test_stale_implementation_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cert.json"
            with patch.object(guidance_goldset, "CERTIFICATION_PATH", path):
                path.write_text(json.dumps(_valid_cert()))
                self.assertTrue(guidance_goldset.certification_status()["certified"])
                stale = {**_valid_cert(),
                         "extractor_implementation_sha256": "deadbeef"}
                path.write_text(json.dumps(stale))
                self.assertEqual(guidance_goldset.certification_status()["reason"],
                                 "CERTIFICATION_STALE_IMPLEMENTATION")

    def test_implementation_hash_tracks_thresholds(self):
        before = guidance_goldset.extractor_implementation_sha256()
        with patch.object(guidance_goldset, "MIN_RANGE_ACCURACY", 0.5):
            after = guidance_goldset.extractor_implementation_sha256()
        self.assertNotEqual(before, after)


class RealOnlyAccuracyTests(unittest.TestCase):
    def _pred(self, expected):
        return [{"metric": e["metric"], "guidance_period": e["period"],
                 "lower_value": e["low"], "upper_value": e["high"],
                 "midpoint": e["midpoint"], "units": e["units"],
                 "stated_action": e["action"]} for e in expected]

    def test_perfect_synthetic_cannot_rescue_bad_real_accuracy(self):
        expected = [{"metric": "revenue", "period": "FY2025", "units": "absolute",
                     "low": 1.0, "high": 2.0, "midpoint": 1.5, "action": "UNSPECIFIED"}]
        gold = [{"company": "EXAMPLE-A", "synthetic": True, "sector": "Tech",
                 "format": "prose", "expected": expected},
                {"company": "RealCo", "sector": "Tech", "format": "prose",
                 "expected": expected}]

        def fake(document, config):
            if guidance_goldset._is_synthetic(document):
                predicted = self._pred(document["expected"])
                return predicted, predicted, predicted
            return [], [], []  # real reconciled output misses everything

        with patch.multiple(guidance_goldset, MIN_GOLD_DOCUMENTS=1, MIN_SECTORS=1,
                            MIN_FORMATS=1), \
                patch.object(guidance_goldset, "llm_config",
                             return_value={"provider": "x", "model": "y"}), \
                patch.object(guidance_goldset, "_predict_variants", side_effect=fake):
            result = guidance_goldset.audit(gold)
        self.assertEqual(result["certified_output"], "real_reconciled")
        self.assertEqual(result["evaluations"]["synthetic_machinery"]
                         ["detection"]["recall"], 1.0)
        self.assertEqual(result["evaluations"]["real_reconciled"]
                         ["detection"]["recall"], 0.0)
        self.assertEqual(result["status"], "ACCURACY_BELOW_THRESHOLD")
        self.assertFalse(result["certified"])


class FrameContractTests(unittest.TestCase):
    def _enriched(self) -> pd.DataFrame:
        return pd.DataFrame({
            "earnings_event_id": ["e1", "e2", "e3"],
            "ticker": ["AAA", "BBB", "CCC"], "sector": ["Tech", "Tech", "Ind"],
            "entry_time": pd.to_datetime(["2024-02-01", "2024-02-02", "2024-03-01"],
                                         utc=True),
            "announcement_date": ["2024-01-31", "2024-02-01", "2024-02-29"],
            "time_bucket": ["TRAIN_TIME"] * 3,
            "target_beta_hedged_5d": [0.01, -0.02, 0.03],
            "quote_complete": [True, True, False], "beta": [1.0, 1.0, 1.0],
            "reaction_score": [0.1, -0.1, 0.2],
            "guidance_revision_score": [0.5, -0.5, 0.1],
        })

    def test_enriched_frame_is_not_remerged(self):
        sample = mgrm._executable_sample(self._enriched())
        suffixed = [c for c in sample.columns if c.endswith(("_x", "_y"))]
        self.assertEqual(suffixed, [])
        self.assertEqual(sample.earnings_event_id.nunique(), 3)
        self.assertEqual(len(sample), 3)  # each event exactly once

    def test_run_on_enriched_frame_does_not_keyerror(self):
        result = mgrm.run(frame=self._enriched(), verbose=False)
        self.assertIn("status", result)  # returns cleanly (blocked on sample power)


class AnnouncementDatePowerTests(unittest.TestCase):
    def _amc(self, date_column: str) -> pd.DataFrame:
        # Two events, SAME entry session but DIFFERENT announcement dates.
        return pd.DataFrame({
            "earnings_event_id": ["e1", "e2"], "ticker": ["A", "B"],
            "sector": ["T", "T"],
            "entry_time": pd.to_datetime(["2024-05-02", "2024-05-02"], utc=True),
            date_column: (["2024-05-01", "2024-04-30"] if date_column
                          == "announcement_date" else
                          pd.to_datetime(["2024-05-01", "2024-04-30"], utc=True)),
            "target_beta_hedged_5d": [0.01, 0.02],
            "quote_complete": [True, True], "beta": [1.0, 1.0],
            "guidance_revision_score": [0.1, 0.2],
        })

    def test_announcement_dates_not_entry_dates(self):
        power = mgrm._sample_power(self._amc("announcement_date"))
        self.assertEqual(power["unique_announcement_dates"], 2)

    def test_public_time_fallback(self):
        power = mgrm._sample_power(self._amc("public_time"))
        self.assertEqual(power["unique_announcement_dates"], 2)


if __name__ == "__main__":
    unittest.main()
