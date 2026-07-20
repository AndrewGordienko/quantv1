from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest

from quantv1.events.atlas import EVENT_TYPES, MANIFEST_VERSION, TAXONOMY_VERSION, validate_record


def _record(raw_path: str, digest: str, **over):
    row = {
        "atlas_event_id": "evt-1", "taxonomy_version": TAXONOMY_VERSION,
        "manifest_version": MANIFEST_VERSION, "cik": "320193", "issuer_name": "Example",
        "ticker": "EXM", "accession_number": "0000000000-25-000001", "form": "8-K",
        "item_codes": "1.01", "event_type": "major_customer_win",
        "public_time": "2025-04-01T20:00:00Z", "known_at": "2025-04-01T20:00:01Z",
        "source_url": "https://www.sec.gov/Archives/example.txt", "source_sha256": digest,
        "raw_path": raw_path, "extraction_version": "rules-v1",
    }
    row.update(over)
    return row


class EventAtlasTests(unittest.TestCase):
    def test_taxonomy_is_explicit_and_versioned(self):
        self.assertGreaterEqual(len(EVENT_TYPES), 30)
        self.assertEqual(len(set(EVENT_TYPES.values())), 15)

    def test_validates_source_anchored_unsigned_event(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "source.txt"
            raw.write_text("primary SEC source")
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            result = validate_record(_record(raw.name, digest), root=root)
        self.assertEqual(result["event_family"], "commercial")
        self.assertEqual(result["split"], "validation")
        self.assertEqual(result["status"], "VERIFIED")

    def test_rejects_direction_and_future_leakage_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "source.txt"
            raw.write_text("source")
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "DIRECTIONAL_FIELD"):
                validate_record(_record(raw.name, digest, direction=1), root=root)

    def test_rejects_known_at_before_public_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "source.txt"
            raw.write_text("source")
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "KNOWN_AT_BEFORE"):
                validate_record(_record(raw.name, digest, known_at="2025-03-01T00:00:00Z"), root=root)


if __name__ == "__main__":
    unittest.main()
