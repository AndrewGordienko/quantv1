from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from quantv1.ingest.microstructure_audit import SELECTION_VERSION, VERSION, audit_manifest


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sample(root: Path) -> Path:
    stocks = [f"T{i:02d}" for i in range(10)]
    sector_etfs = {ticker: f"X{i:02d}" for i, ticker in enumerate(stocks)}
    tickers = stocks + ["SPY"] + list(sector_etfs.values())
    sessions = pd.date_range("2026-01-02", periods=20, freq="B")
    quotes, trades = [], []
    for ticker in tickers:
        for sequence, day in enumerate(sessions, start=1):
            ts = (day + pd.Timedelta(hours=14, minutes=30)).isoformat()
            quotes.append({"ticker": ticker, "exchange_ts": ts, "sequence": sequence,
                           "bid": 100, "ask": 100.01, "bid_size": 200, "ask_size": 300,
                           "condition_codes": "R"})
            trades.append({"ticker": ticker, "exchange_ts": ts, "sequence": sequence,
                           "price": 100.01, "size": 100, "condition_codes": "@",
                           "correction_code": "", "cancellation_code": ""})
    files = {
        "quotes": pd.DataFrame(quotes), "trades": pd.DataFrame(trades),
        "halts": pd.DataFrame(columns=["ticker", "start_ts", "end_ts", "halt_code"]),
        "corporate_actions": pd.DataFrame(columns=["ticker", "effective_date", "adjustment_type", "adjustment_factor", "source"]),
    }
    entries = {}
    for kind, frame in files.items():
        path = root / f"{kind}.csv"
        frame.to_csv(path, index=False)
        entries[kind] = {"path": path.name, "sha256": _digest(path)}
    selection = {
        "version": SELECTION_VERSION, "selection_id": "test-selection", "created_at": "2026-01-01T00:00:00Z",
        "stock_symbols": stocks, "benchmark": "SPY", "sector_etfs": sector_etfs,
        "sessions": [day.strftime("%Y-%m-%d") for day in sessions],
        "no_return_or_move_selection": True,
    }
    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps(selection))
    manifest = {
        "version": VERSION, "provider": "test-provider", "dataset_id": "test-dataset", "snapshot_id": "snapshot-1",
        "retrieved_at": "2026-07-11T12:00:00Z", "source_documentation_url": "https://example.test/docs",
        "historical_availability": {"point_in_time": True, "available_from": "2024-01-01T00:00:00Z", "documentation_url": "https://example.test/history"},
        "ordering": {"exchange_timestamp": True, "sequence_numbers": True, "sequence_domain": ["ticker", "session"], "documentation_url": "https://example.test/order"},
        "quote_completeness": {"nbbo_updates_complete": True, "methodology": "complete NBBO", "documentation_url": "https://example.test/nbbo"},
        "conditions": {"trade_condition_codes_documented": True, "quote_condition_codes_documented": True, "corrections_and_cancellations_included": True},
        "selection": {"path": selection_path.name, "sha256": _digest(selection_path)},
        "files": entries,
    }
    target = root / "manifest.json"
    target.write_text(json.dumps(manifest))
    return target


class MicrostructureAuditTests(unittest.TestCase):
    def test_accepts_complete_hashed_ordered_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_manifest(_write_sample(Path(directory)))
        self.assertEqual(result["status"], "ACCEPTED_FOR_F2_FEATURE_RESEARCH")
        self.assertEqual(result["errors"], [])

    def test_rejects_missing_quote_completeness_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_sample(Path(directory))
            manifest = json.loads(path.read_text())
            manifest["quote_completeness"] = {"nbbo_updates_complete": False}
            path.write_text(json.dumps(manifest))
            result = audit_manifest(path)
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("QUOTE_COMPLETENESS_NOT_PROVEN", result["errors"])

    def test_rejects_tampered_quote_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_sample(root)
            quotes = pd.read_csv(root / "quotes.csv")
            quotes.loc[0, "bid"] = 101  # also invalidates the pinned hash
            quotes.to_csv(root / "quotes.csv", index=False)
            result = audit_manifest(path)
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("HASH_MISMATCH:quotes", result["errors"])

    def test_rejects_unordered_quote_file_even_with_a_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_sample(root)
            quotes_path = root / "quotes.csv"
            quotes = pd.read_csv(quotes_path)
            # Create two T00 updates in one session, deliberately in reverse
            # raw sequence. The matching hash below proves this is an ordering
            # audit, not merely an integrity-hash check.
            early = quotes.iloc[0].copy()
            late = early.copy()
            late["exchange_ts"] = (pd.Timestamp(early["exchange_ts"]) + pd.Timedelta(minutes=1)).isoformat()
            late["sequence"] = 2
            quotes = pd.concat([pd.DataFrame([late, early]), quotes.iloc[1:]], ignore_index=True)
            quotes.to_csv(quotes_path, index=False)
            manifest = json.loads(path.read_text())
            manifest["files"]["quotes"]["sha256"] = _digest(quotes_path)
            path.write_text(json.dumps(manifest))
            result = audit_manifest(path)
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("NON_MONOTONIC_ORDER:quotes", result["errors"])

    def test_accepts_venue_scoped_sequence_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_sample(root)
            manifest = json.loads(path.read_text())
            for kind in ("quotes", "trades"):
                file_path = root / manifest["files"][kind]["path"]
                frame = pd.read_csv(file_path)
                frame["venue"] = "XNAS"
                frame.to_csv(file_path, index=False)
                manifest["files"][kind]["sha256"] = _digest(file_path)
            manifest["ordering"]["sequence_domain"] = ["ticker", "venue", "session"]
            path.write_text(json.dumps(manifest))
            result = audit_manifest(path)
        self.assertEqual(result["status"], "ACCEPTED_FOR_F2_FEATURE_RESEARCH")

    def test_accepts_tied_timestamps_when_sequence_resolves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_sample(root)
            manifest = json.loads(path.read_text())
            for kind in ("quotes", "trades"):
                file_path = root / manifest["files"][kind]["path"]
                frame = pd.read_csv(file_path)
                tied = frame.iloc[0].copy()
                tied["sequence"] = 2
                frame = pd.concat([frame.iloc[:1], pd.DataFrame([tied]), frame.iloc[1:]], ignore_index=True)
                frame.to_csv(file_path, index=False)
                manifest["files"][kind]["sha256"] = _digest(file_path)
            path.write_text(json.dumps(manifest))
            result = audit_manifest(path)
        self.assertEqual(result["status"], "ACCEPTED_FOR_F2_FEATURE_RESEARCH")


if __name__ == "__main__":
    unittest.main()
