from __future__ import annotations

import unittest

from quantv1.ingest.fed_primary import (
    B2_SAMPLE, B3_SAMPLE, ManifestError, _validate,
)


def _record(sample=B2_SAMPLE):
    record = {
        "sample": sample,
        "actor": {"actor_id": "speaker", "name": "Fed Speaker",
                  "actor_type": "central_banker"},
        "institutional_role": {
            "organization": "Federal Reserve Board", "role": "Governor",
            "valid_from": "2024-01-01", "valid_to": None,
            "source": "https://www.federalreserve.gov/aboutthefed.htm",
        },
        "public_time": "2026-01-02T14:00:00-05:00",
        "timestamp_precision": "exact",
        "communication_type": "speech",
        "actor_event_role": "speaker_author",
        "title": "Economic outlook",
        "source_url": "https://www.federalreserve.gov/newsevents/speech/example.htm",
        "transcript": "Primary source text",
        "asset_exposures": [
            {"ticker": "TLT", "channel": "monetary_policy", "confidence": 1.0},
            {"ticker": "XLF", "channel": "monetary_policy", "confidence": 0.8},
        ],
    }
    if sample == B3_SAMPLE:
        record["communication_type"] = "chair_press_conference"
        record["segments"] = [
            {"public_time": "2026-01-02T14:00:10-05:00",
             "segment_role": "prepared", "actor_id": "speaker", "text": "Opening."},
            {"public_time": "2026-01-02T14:01:00-05:00",
             "segment_role": "question", "actor_id": None, "text": "Question?"},
            {"public_time": "2026-01-02T14:01:05-05:00",
             "segment_role": "answer", "actor_id": "speaker", "text": "Answer."},
        ]
    return record


class FedPrimaryManifestTests(unittest.TestCase):
    def test_b2_and_b3_are_separate_valid_samples(self):
        self.assertEqual(_validate(_record())["sample"], B2_SAMPLE)
        self.assertEqual(_validate(_record(B3_SAMPLE))["sample"], B3_SAMPLE)

    def test_rejects_secondary_source_and_page_date_precision(self):
        record = _record()
        record["source_url"] = "https://example.com/news"
        with self.assertRaises(ManifestError):
            _validate(record)
        record = _record()
        record["timestamp_precision"] = "date_only"
        with self.assertRaises(ManifestError):
            _validate(record)

    def test_requires_rates_duration_and_financial_asset(self):
        record = _record()
        record["asset_exposures"] = [
            {"ticker": "SPY", "channel": "monetary_policy", "confidence": 0.5}
        ]
        with self.assertRaises(ManifestError):
            _validate(record)


if __name__ == "__main__":
    unittest.main()
